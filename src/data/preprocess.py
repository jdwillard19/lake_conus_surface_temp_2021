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
start = sys.argv[1]
end = sys.argv[2]
site_ids = metadata['site_id'].values[start:end]

#load wst obs
obs = pd.read_csv("../../data/raw/obs/lake_surface_temp_obs.csv")



obs.sort_values('Date',inplace=True)
n_lakes = site_ids.shape[0]


n_dyn_feats = 5 #AT,LW,SW,WSU,WSV
n_stc_feats = 4 #AREA,LAT,LON,ELEV

#pre-calculated statistics
mean_feats = np.array([13.1700613, 41.67473611, -90.43172611, 397.97342139, 1.76944297e+02, 3.07248340e+02, 2.82968074e+02, 7.85104236e-01, 2.86081133e-01])
std_feats = np.array([1.630222, 6.45012084, 9.8714776, 474.08400329,9.10455152, 7.54579132, 3.32533227, 1.62018831, 1.70615275])
   
n_features = mean_feats.shape[0]

#get dates
base_path = '../../data/raw/data_release/'
w1 = xr.open_dataset(base_path+'01_weather_N40-53_W98-126.nc4')
dates = w1['time'].values


# loop to preprocess each site
for site_ct, site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids)," starting ", site_id)

    #get NLDAS coords
    x_re
    y_re = 
    weather_str = metadata[metadata['site_id'] == site_id]['x'].values[0]
    re.search(x_re,metadata[metadata['site_id'] == site_id]['weather_id'].values[0]).group(1)
    x = str(metadata[metadata['site_id'] == site_id]['x'].values[0])+".0"
    y = str(metadata[metadata['site_id'] == site_id]['y'].values[0])+".0"

    #read/format meteorological data for numpy
    site_obs = obs[obs['site_id'] == site_id]
    print(site_obs.shape[0], " obs")

    #lower/uppur cutoff indices (to match observations)
    # if site_obs.shape[0] == 0:
    #     print("|\n|\nNO SURFACE OBSERVATIONS\n|\n|\n|")
    #     pdb.set_trace()w
    #     no_obs_ids.append(site_id)
    #     no_obs_ct +=1 
    #     continue

    site_obs = site_obs.sort_values("Date")    
    #sort observations
    # obs_start_date = site_obs.values[0,0]
    meteo_start_date = dates[0]
    start_date = meteo_start_date
    # obs_end_date = site_obs.values[-1,0]

    print("start date: ",start_date)
    print("end date: ", dates[-1])
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
    sw = np.load("../../data/raw/feats/SW_"+str(x)+"x_"+str(y)+"y.npy")
    lw = np.load("../../data/raw/feats/LW_"+str(x)+"x_"+str(y)+"y.npy")
    at = np.load("../../data/raw/feats/AT_"+str(x)+"x_"+str(y)+"y.npy")
    wsu = np.load("../../data/raw/feats/WSU_"+str(x)+"x_"+str(y)+"y.npy")
    wsv = np.load("../../data/raw/feats/WSV_"+str(x)+"x_"+str(y)+"y.npy")


    site_feats[:,0] = np.log(metadata[metadata['site_id']==site_id].area_m2)
    site_feats[:,1] = metadata[metadata['site_id']==site_id].lat
    site_feats[:,2] = metadata[metadata['site_id']==site_id].lon
    site_feats[:,3] = metadata[metadata['site_id']==site_id].elevation
    site_feats[:,4] = sw[lower_cutoff:upper_cutoff]
    site_feats[:,5] = lw[lower_cutoff:upper_cutoff]
    site_feats[:,6] = at[lower_cutoff:upper_cutoff]
    site_feats[:,7] = wsu[lower_cutoff:upper_cutoff]
    site_feats[:,8] = wsv[lower_cutoff:upper_cutoff]

    #normalize data
    feats_norm = (site_feats - mean_feats[:]) / std_feats[:]


    # obs_trn_mat = np.empty((n_dates))
    site_obs_mat = np.empty((n_dates))
    site_obs_mat[:] = np.nan
    # obs_trn_mat[:] = np.nan
    # obs_tst_mat = np.empty((n_dates))
    # obs_tst_mat[:] = np.nan



    obs_g = 0
    obs_d = 0

    # get unique observation days
    unq_obs_dates = np.unique(site_obs.values[:,0])
    n_unq_obs_dates = unq_obs_dates.shape[0]
    n_obs = n_unq_obs_dates

    n_obs_placed = 0
    # n_trn_obs_placed = 0
    for o in range(0,n_obs):
        if len(np.where(site_dates == pd.Timestamp(site_obs.values[o,0]).to_datetime64())[0]) < 1:
            print("not within meteo dates")
            obs_d += 1
            continue
        date_ind = np.where(site_dates == pd.Timestamp(site_obs.values[o,0]).to_datetime64())[0][0]
        site_obs_mat[date_ind] = site_obs.values[o,2]
        n_obs_placed += 1


    if not os.path.exists("../../data/processed/"+site_id): 
        os.mkdir("../../data/processed/"+site_id)
    if not os.path.exists("../../models/"+site_id):
        os.mkdir("../../models/"+site_id)

    feat_path = "../../data/processed/"+site_id+"/features_ea_conus_021621"
    norm_feat_path = "../../data/processed/"+site_id+"/processed_features_ea_conus_021621"
    full_path = "../../data/processed/"+site_id+"/full"
    # full_path = "../../data/processed/"+site_id+"/full_061121_cold_lab_filtered"

    dates_path = "../../data/processed/"+site_id+"/dates"


    # np.save(feat_path, site_feats)
    # np.save(norm_feat_path, feats_norm)
    # np.save(dates_path, site_dates)
    np.save(full_path, site_obs_mat)

