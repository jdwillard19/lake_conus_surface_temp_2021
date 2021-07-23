import xarray as xr
import pandas as pd
import numpy as np
import os
import sys
import re
import math
import shutil
import pdb
import datetime

###############################################################################3
# june 2021 - Jared - this script takes the previously created weather files per lake and
# normalizes them for use in ML. Two command line arguments indicate starting and ending index
# for lakes to be processed as listed in the metadata file
#
###################################################################################

#load metadata, get command line arguments indicating indices of lakes you wish to preprocess, get ids
metadata = pd.read_csv("../../metadata/lake_metadata.csv")
start = int(sys.argv[1])
end = int(sys.argv[2])
site_ids = metadata['site_id'].values[start:end]

#load wst obs
obs = pd.read_csv("../../data/raw/obs/lake_surface_temp_obs.csv")



obs.sort_values('Date',inplace=True)
n_lakes = site_ids.shape[0]


n_dyn_feats = 5 #AT,LW,SW,WSU,WSV
n_stc_feats = 4 #AREA,LAT,LON,ELEV

mean_feats = None
calc_stats = True

#get dates

base_path = '../../data/raw/data_release/'
w1 = xr.open_dataset(base_path+'01_weather_N40-53_W98-126.nc4')
dates = w1['time'].values
n_dates = len(dates)

areas = np.empty((n_lakes))
areas[:] = np.nan
lats = np.empty((n_lakes))
lats[:] = np.nan
lons = np.empty((n_lakes))
lons[:] = np.nan
elevs = np.empty((n_lakes))
elevs[:] = np.nan
sws = np.empty((n_lakes,2))
sws[:] = np.nan
lws = np.empty((n_lakes,2))
lws[:] = np.nan
ats = np.empty((n_lakes,2))
ats[:] = np.nan
wsus = np.empty((n_lakes,2))
wsus[:] = np.nan
wsvs = np.empty((n_lakes,2))
wsvs[:] = np.nan
feat_base_path = '../../data/raw/feats/'


if not calc_stats:  #pre-calculated statistics
    mean_feats = np.array([11.6896, 39.822, 
                           -94.141577, 476.3117, 
                           3661.1549, 1759.52895, 
                           197.5949, 3.4463, 
                           4.315085986565])
    std_feats = np.sqrt(np.array([1.1173**2, 6.7915**2, 
                          12.1085**2, 575.42646**2,
                          7136.574, 3205.9666, 
                          110.0645, 6.2657, 
                          8.1407]))
else: 
    metadata.set_index('site_id',inplace=True)
    for site_ct, site_id in enumerate(site_ids):
            # if site_ct == 0:
            #     site_id = 'nhdhr_143249470'
            if site_ct % 10000==0:
                print(site_ct)
            w_id = metadata.loc[site_id]['weather_id'].encode()
            area = np.log(metadata.loc[site_id].area_m2)
            areas[site_ct] = area
            lat = metadata.loc[site_id].lake_lat_deg
            lats[site_ct] = lat
            lon = metadata.loc[site_id].lake_lon_deg
            lons[site_ct] = lon
            elev = metadata.loc[site_id].elevation_m
            elevs[site_ct] = elev

            #get dynamic feats
            sw = np.load(feat_base_path+"SW_"+w_id.decode()+".npy",allow_pickle=True)
            sws[site_ct,0] = sw.mean()
            sws[site_ct,1] = sw.std()**2
            lw = np.load(feat_base_path+"LW_"+w_id.decode()+".npy",allow_pickle=True)
            lws[site_ct,0] = lw.mean()
            lws[site_ct,1] = lw.std()**2
            at = np.load(feat_base_path+"AT_"+w_id.decode()+".npy",allow_pickle=True)
            ats[site_ct,0] = at.mean()
            ats[site_ct,1] = at.std()**2
            wsu = np.load(feat_base_path+"WSU_"+w_id.decode()+".npy",allow_pickle=True)
            wsus[site_ct,0] = wsu.mean()
            wsus[site_ct,1] = wsu.std()**2
            wsv = np.load(feat_base_path+"WSV_"+w_id.decode()+".npy",allow_pickle=True)
            wsvs[site_ct,0] = wsv.mean()
            wsvs[site_ct,1] = wsv.std()**2

print(areas.mean())
print(areas.std())
print(lats.mean())
print(lats.std())
print(lons.mean())
print(lons.std())
print(elevs.mean())
print(elevs.std())
print(sws[:,0].mean())
print(sws[:,1].mean())
print(lws[:,0].mean())
print(lws[:,1].mean())
print(ats[:,0].mean())
print(ats[:,1].mean())
print(wsus[:,0].mean())
print(wsus[:,1].mean())
print(wsvs[:,0].mean())
print(wsvs[:,1].mean())
np.save("sws",sws)
np.save("lws",lws)
np.save("ats",ats)
np.save("wsus",wsus)
np.save("wsvs",wsvs)
sys.exit()
n_features = mean_feats.shape[0]

#get dates
base_path = '../../data/raw/data_release/'
w1 = xr.open_dataset(base_path+'01_weather_N40-53_W98-126.nc4')
dates = w1['time'].values


# loop to preprocess each site
for site_ct, site_id in enumerate(site_ids):
    site_id = 'nhdhr_143249470'

    print(site_ct,"/",len(site_ids)," starting ", site_id)
    # if os.path.exists("../../data/processed/"+site_id+"/features.npy"):
    #     print("already done")
    #     continue
    #get weather_id
    w_id = metadata[metadata['site_id'] == site_id]['weather_id'].values[0].encode()


    #get and sort obs
    site_obs = obs[obs['site_id'] == site_id]
    print(site_obs.shape[0], " obs")
    site_obs = site_obs.sort_values("Date")    

    start_date = dates[0]
    end_date = dates[-1]
    print("start date: ",start_date)
    print("end date: ", end_date)

    #cut files to between first and last observation
    lower_cutoff = np.where(dates == pd.Timestamp(start_date).to_datetime64())[0][0] #457
    print("lower cutoff: ", lower_cutoff)
    upper_cutoff = dates.shape[0]
    print("upper cutoff: ", upper_cutoff)
    site_dates = dates[lower_cutoff:upper_cutoff]
    n_dates = len(site_dates)
    print("n dates after cutoff: ", n_dates)

    site_feats = np.empty((n_dates,n_features))
    site_feats[:] = np.nan

    #get static feats
    area = np.log(metadata[metadata['site_id']==site_id].area_m2)
    lat = metadata[metadata['site_id']==site_id].lake_lat_deg
    lon = metadata[metadata['site_id']==site_id].lake_lon_deg
    elev = metadata[metadata['site_id']==site_id].elevation_m

    #get dynamic feats
    sw = np.load(feat_base_path+"SW_"+w_id.decode()+".npy",allow_pickle=True)
    lw = np.load(feat_base_path+"LW_"+w_id.decode()+".npy",allow_pickle=True)
    at = np.load(feat_base_path+"AT_"+w_id.decode()+".npy",allow_pickle=True)
    wsu = np.load(feat_base_path+"WSU_"+w_id.decode()+".npy",allow_pickle=True)
    wsv = np.load(feat_base_path+"WSV_"+w_id.decode()+".npy",allow_pickle=True)

    #fill data
    site_feats[:,0] = area
    site_feats[:,1] = lat
    site_feats[:,2] = lon 
    site_feats[:,3] = elev
    site_feats[:,4] = sw
    site_feats[:,5] = lw
    site_feats[:,6] = at
    site_feats[:,7] = wsu
    site_feats[:,8] = wsv

    #normalize data
    feats_norm = (site_feats - mean_feats[:]) / std_feats[:]

    #data structs to fill
    site_obs_mat = np.empty((n_dates))
    site_obs_mat[:] = np.nan




    # get unique observation days
    unq_obs_dates = np.unique(site_obs['Date'].values)
    n_unq_obs_dates = unq_obs_dates.shape[0]
    n_obs = n_unq_obs_dates
    n_obs_placed = 0
    for o in range(0,n_obs):
        if len(np.where(dates == pd.Timestamp(site_obs['Date'].values[o]).to_datetime64())[0]) < 1:
            print("not within meteo dates")
            continue
        date_ind = np.where(dates == pd.Timestamp(site_obs['Date'].values[o]).to_datetime64())[0][0]
        site_obs_mat[date_ind] = site_obs['wtemp_obs'].values[o]
        n_obs_placed += 1

    #make directory if not exist
    if not os.path.exists("../../data/processed/"+site_id): 
        os.mkdir("../../data/processed/"+site_id)
    if not os.path.exists("../../models/"+site_id):
        os.mkdir("../../models/"+site_id)

    feat_path = "../../data/processed/"+site_id+"/features"
    norm_feat_path = "../../data/processed/"+site_id+"/processed_features"
    obs_path = "../../data/processed/"+site_id+"/obs"
    dates_path = "../../data/processed/"+site_id+"/dates"

    #assert and save
    assert np.isfinite(site_feats).all()
    assert np.isfinite(feats_norm).all()
    assert np.isfinite(dates).all()


    np.save(feat_path, site_feats)
    np.save(norm_feat_path, feats_norm)
    np.save(dates_path, dates)
    np.save(obs_path, site_obs_mat)

