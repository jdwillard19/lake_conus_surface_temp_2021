import pandas as pd
import sys
import numpy as np
import pdb
import rasterio
import os

#get prism data to align with dates
metadata = pd.read_csv("../../metadata/lake_metadata.csv")
start = int(sys.argv[1])
end = int(sys.argv[2])
site_ids = metadata['site_id'].values[start:end]

for site_ct, site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids)," starting ", site_id)
    feat_path = "../../data/processed/"+site_id+"/features.npy"
    feat_old = np.load(feat_path, allow_pickle=True)

    dates_path = "../../data/processed/"+site_id+"/dates.npy"
    dates = np.load(dates_path, allow_pickle=True)

    new_temps = np.empty(feat_old.shape[0])
    lat = feat_old[0,1]
    lon = feat_old[0,2]

    for date_ct, date in enumerate(dates):
        print("date ", date)
        date_str = str(date)[:4]+str(date)[5:7]+str(date)[8:10]
        file_path = '../../data/raw/prism/PRISM_tmax_stable_4kmD2_'+date_str+'_bil.bil'
        if date_ct > 500:
            pdb.set_trace()
        if os.path.exists(file_path):
            dataset = rasterio.open(file_path)
            pdb.set_trace()
        else:
            print("date not in PRISM")
