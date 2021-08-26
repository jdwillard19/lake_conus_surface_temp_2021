import numpy as np
import pandas as pd
import pdb


#
site_ids = ['nhdhr_112699096']



#load data
dates = np.load("../../data/processed/"+site_ids[0]+"/dates.npy",allow_pickle=True)

for site_id in site_ids:
	obs = np.load("../../data/processed/"+site_id+"/obs.npy")
	feats = np.load("../../data/processed/"+site_id+"/features.npy")
	air_temp = feats[:,6]
	for y in range(1980,2021):
		pdb.set_trace()
		start_ind = 0

