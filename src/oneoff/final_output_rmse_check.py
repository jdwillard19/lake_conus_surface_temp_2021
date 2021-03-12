import pandas as pd 
import numpy as np
import os
import pdb

metadata = pd.read_csv("../../metadata/lake_metadata_full_conus_185k.csv")
site_ids = metadata['site_id'].values

sites = []
rmses = []
final_df = pd.DataFrame()
def calc_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean()) 
for site_ct,site_id in enumerate(site_ids):
    print(site_ct,"/",len(site_ids), " site")
    if not metadata[metadata['site_id'] == site_id]['observed'].values[0]:
        print("unobserved")
        continue
    else:
        df = pd.read_feather("../../results/SWT_results/outputs_"+site_id+".feather") 
        loss_outputs = df['temp_pred'].values[np.isfinite(df['temp_actual'].values)]
        loss_actual = df['temp_actual'].values[np.isfinite(df['temp_actual'].values)]
        loss_dates = df['index'].values[np.isfinite(df['temp_actual'].values)]


        df = pd.DataFrame(data={'date':loss_dates, 'pred':loss_outputs, 'actual':loss_actual})
        final_df = pd.concat([final_df,df])
        pdb.set_trace()

        
        # sites.append(site_id)
        # rmse = calc_rmse(loss_outputs, loss_actual)
        # print("rmse: ",rmse)
        # rmses.append(rmse)
