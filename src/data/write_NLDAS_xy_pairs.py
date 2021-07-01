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
# june 2021 - Jared - this script creates weather data files per NLDAS cell for each 
# weather driver. Two command line arguments indicate starting and ending index
# for lakes to be processed as listed in the metadata file
#
###################################################################################

#load metadata, get ids
metadata = pd.read_csv("../../metadata/lake_metadata.csv")

#get site ids
site_ids = np.unique(metadata['site_id'].values)
n_lakes = site_ids.shape[0]


#load weather files
base_path = '../../data/raw/data_release/'
feat_base_path = "../../data/raw/feats/"
w1_fn = '01_weather_N40-53_W98-126.nc4'
w2_fn = '02_weather_N24-40_W98-126.nc4'
w3_fn = '03_weather_N40-53_W82-98.nc4'
w4_fn = '04_weather_N24-40_W82-98.nc4'
w5_fn = '05_weather_N24-53_W67-82.nc4'
w1 = xr.open_dataset(base_path+w1_fn)
w2 = xr.open_dataset(base_path+w2_fn)
w3 = xr.open_dataset(base_path+w3_fn)
w4 = xr.open_dataset(base_path+w4_fn)
w5 = xr.open_dataset(base_path+w5_fn)

start = int(sys.argv[1])
end = int(sys.argv[2])
print("running site id's ",start,"->",end)
site_ids = site_ids[start:end]
skipped = []
verbose = True
for site_ct, site_id in enumerate(site_ids):
    print("(",site_ct,"/",str(len(site_ids)),") ","writing... ", site_id)

    site_id = 'nhdhr_134311540'
    pdb.set_trace()
    lon = metadata[metadata['site_id'] == site_id]['weather_lon_deg'].values[0]
    lat = metadata[metadata['site_id'] == site_id]['weather_lat_deg'].values[0]
    w_id = metadata[metadata['site_id'] == site_id]['weather_id'].values[0].encode()
    #check if cell already processed
    if os.path.exists(feat_base_path+"LW_"+w_id.decode()+".npy") and os.path.exists(feat_base_path+"SW_"+w_id.decode()+".npy") and os.path.exists(feat_base_path+"WSV_"+w_id.decode()+".npy") and os.path.exists(feat_base_path+"WSU_"+w_id.decode()+".npy") and  os.path.exists(feat_base_path+"AT_"+w_id.decode()+".npy"):
        print("ALREADY PROCESSED")
        continue
    #select weather file
    weather = None

    if (w1['instance_name']==w_id).any():
        if verbose:
            print("loading ",w1_fn)
        weather = w1
    elif (w2['instance_name']==w_id).any(): 
        weather = w2
        if verbose:
            print("loading ",w2_fn)
    elif (w3['instance_name']==w_id).any():
        weather = w3
        if verbose:
            print("loading ",w3_fn)
    elif (w4['instance_name']==w_id).any():
        weather = w4
        if verbose:
            print("loading ",w4_fn)
    elif (w5['instance_name']==w_id).any():
        weather = w5
        if verbose:
            print("loading ",w5_fn)
    else:
        print("invalid instance name / weather_id")
        pdb.set_trace()

    #index by lat/lon
    ind = (weather['instance_name']==w_id)
    assert ind.any()

    #select data 
    sw_vals = weather['dswrfsfc'][ind,:].values
    lw_vals = weather['dlwrfsfc'][ind,:].values
    at_vals = weather['tmp2m'][ind,:].values
    wsu_vals = weather['ugrd10m'][ind,:].values
    wsv_vals = weather['vgrd10m'][ind,:].values
    if np.isnan(sw_vals).any():
        print("nan sw?")
        raise Exception("CANT CONTINUE") 
    if np.isnan(lw_vals).any():
        print("nan lw?")
        raise Exception("CANT CONTINUE") 
    if np.isnan(at_vals).any():
        print("nan at?")
        raise Exception("CANT CONTINUE") 
    if np.isnan(wsu_vals).any():
        print("nan wsu?")
        raise Exception("CANT CONTINUE") 
    if np.isnan(wsv_vals).any():
        print("nan wsv?") 
        raise Exception("CANT CONTINUE") 
    # pdb.set_trace()
    np.save(feat_base_path+"SW_"+w_id.decode(),sw_vals)
    np.save(feat_base_path+"LW_"+w_id.decode(),lw_vals)
    np.save(feat_base_path+"AT_"+w_id.decode(),at_vals)
    np.save(feat_base_path+"WSU_"+w_id.decode(),wsu_vals)
    np.save(feat_base_path+"WSV_"+w_id.decode(),wsv_vals)
    if verbose:
        print("x/y: ",w_id,":\nSW: ", sw_vals, "\nLW: ",lw_vals,"\nAT: ",at_vals,"\nWSU: ", wsu_vals, "\nWSV: ", wsv_vals)

print("DATA COMPLETE")
