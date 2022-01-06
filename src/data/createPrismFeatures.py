import pandas as pd
import sys
import numpy as np
import pdb
import rasterio
import os

TO_KELVIN = 273.15
#load metadata
metadata = pd.read_csv("../../metadata/lake_metadata.csv")
metadata = metadata[metadata['num_obs'] > 0]
site_ids = metadata['site_id'].values

site_id0 = site_ids[0]

feat_path = "../../data/processed/"+site_id0+"/features.npy"
feat_old = np.load(feat_path, allow_pickle=True)

dates_path = "../../data/processed/"+site_id0+"/dates.npy"
dates = np.load(dates_path, allow_pickle=True)

#temp data struct to fill
at_arr = np.empty((site_ids.shape[0], dates.shape[0]))
at_arr[:] = np.nan
inds_arr = np.array_split(range(dates.shape[0]),300)


#load prism air temps
for i in range(300):
    load_at = np.load("../../data/raw/prism/ats_split"+str(i)+".npy",allow_pickle=True)
    at_arr[:,inds_arr[i]] = load_at

#repreprocess per site
for site_ct, site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids)," starting ", site_id)
    feat_path = "../../data/processed/"+site_id+"/features.npy"
    obs_path = "../../data/processed/"+site_id+"/obs.npy"
    dates_path = "../../data/processed/"+site_id+"/dates.npy"

    feat = np.load(feat_path, allow_pickle=True)
    obs = np.load(obs_path, allow_pickle=True)
    dates = np.load(dates_path, allow_pickle=True)

    new_feat_path = "../../data/processed/"+site_id+"/features_wPrism"
    new_obs_path = "../../data/processed/"+site_id+"/obs_wPrism"
    new_dates_path = "../../data/processed/"+site_id+"/dates_wPrism"

    feat[:,6] = at_arr[site_ct,:]+TO_KELVIN

    #clip to prism data range (exclude 1980)
    feat = feat[366:]
    obs = obs[366:]
    dates = dates[366:]

    assert np.isfinite(feat).all()

    #save
    np.save(new_feat_path,feat)
    np.save(new_obs_path,obs)
    np.save(new_dates_path,dates)

