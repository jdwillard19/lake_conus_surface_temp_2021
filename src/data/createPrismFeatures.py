import pandas as pd
import sys
import numpy as np
import pdb
import rasterio
import os

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
    load_at = np.load("../../data/raw/prism/ats_split"+str(i),at_arr)
    at_arr[:,inds_arr[i]] = load_at

for site_ct, site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids)," starting ", site_id)
    feat_path = "../../data/processed/"+site_id+"/features.npy"
    feat_old = np.load(feat_path, allow_pickle=True)
    pdb.set_trace()

