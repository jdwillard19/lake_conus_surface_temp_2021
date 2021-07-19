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
lats = np.empty((n_lakes))
lons = np.empty((n_lakes))
elevs = np.empty((n_lakes))
sws = np.empty((n_lakes,n_dates))
lws = np.empty((n_lakes,n_dates))
ats = np.empty((n_lakes,n_dates))
wsus = np.empty((n_lakes,n_dates))
wsvs = np.empty((n_lakes,n_dates))
feat_base_path = '../../data/raw/feats/'


if not calc_stats:  #pre-calculated statistics
    mean_feats = np.array([13.1700613, 41.67473611, -90.43172611, 397.97342139, 1.76944297e+02, 3.07248340e+02, 2.82968074e+02, 7.85104236e-01, 2.86081133e-01])
    std_feats = np.array([1.630222, 6.45012084, 9.8714776, 474.08400329,9.10455152, 7.54579132, 3.32533227, 1.62018831, 1.70615275])
else: 
    for site_ct, site_id in enumerate(site_ids):
            if site_ct % 1000 == 0:
                print(site_ct)
            area = np.log(metadata[metadata['site_id']==site_id].area_m2)
            areas[site_ct] = area
            lat = metadata[metadata['site_id']==site_id].lake_lat_deg
            lats[site_ct] = lat
            lon = metadata[metadata['site_id']==site_id].lake_lon_deg
            lons[site_ct] = lon
            elev = metadata[metadata['site_id']==site_id].elevation_m
            elevs[site_ct] = elev

            #get dynamic feats
            sw = np.load(feat_base_path+"SW_"+w_id.decode()+".npy",allow_pickle=True)
            sws[site_ct,:] = sw
            lw = np.load(feat_base_path+"LW_"+w_id.decode()+".npy",allow_pickle=True)
            lws[site_ct,:] = lw
            at = np.load(feat_base_path+"AT_"+w_id.decode()+".npy",allow_pickle=True)
            ats[site_ct,:] = at
            wsu = np.load(feat_base_path+"WSU_"+w_id.decode()+".npy",allow_pickle=True)
            wsus[site_ct,:] = wsu
            wsv = np.load(feat_base_path+"WSV_"+w_id.decode()+".npy",allow_pickle=True)
            wsvs[site_ct,:] = wsv

pdb.set_trace()
n_features = mean_feats.shape[0]

#get dates
base_path = '../../data/raw/data_release/'
w1 = xr.open_dataset(base_path+'01_weather_N40-53_W98-126.nc4')
dates = w1['time'].values


# loop to preprocess each site
for site_ct, site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids)," starting ", site_id)
    if os.path.exists("../../data/processed/"+site_id+"/features.npy"):
        print("already done")
        continue
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
    unq_obs_dates = np.unique(site_obs.values[:,0])
    n_unq_obs_dates = unq_obs_dates.shape[0]
    n_obs = n_unq_obs_dates
    n_obs_placed = 0
    for o in range(0,n_obs):
        if len(np.where(dates == pd.Timestamp(site_obs.values[o,0]).to_datetime64())[0]) < 1:
            print("not within meteo dates")
            obs_d += 1
            continue
        date_ind = np.where(dates == pd.Timestamp(site_obs.values[o,0]).to_datetime64())[0][0]
        site_obs_mat[date_ind] = site_obs.values[o,2]
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

