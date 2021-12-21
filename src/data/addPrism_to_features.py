import pandas as pd
import sys
import numpy as np
import pdb

#get prism data to align with dates
metadata = pd.read_csv("../../metadata/lake_metadata.csv")
start = int(sys.argv[1])
end = int(sys.argv[2])
site_ids = metadata['site_id'].values[start:end]

for site_ct, site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids)," starting ", site_id)
	feat_path = "../../data/processed/"+site_id+"/features.npy"
	feat_old = np.load(feat_path, allow_pickle=True)

    dates_path = "../../data/processed/"+site_id+"/dates"
    dates = np.load(dates_path, allow_pickle=True)

    
    pdb.set_trace()