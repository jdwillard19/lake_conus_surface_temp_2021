import pandas as pd
import numpy as np
import pdb
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from joblib import dump, load
import re
import datetime
import xgboost as xgb


##################################################################3
# (Jan 2020 - Jared) - 
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#file to save model  to
save_file_path = '../../models/xgb_surface_temp.joblib'




#############################
#load data
train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]

train_df = pd.DataFrame()
train_df.columns = ['ShortWave','LongWave','AirTemp','WindSpeed','Surface_Area','Surface_Temp']

for ct, lake_id in enumerate(train_lakes):
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features_ea")
    labs = np.load("../../data/processed/"+lake_id+"/full")
    pdb.set_trace()


#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here



#W TRANSFER OPTIM, W PRETRAIN
# feats = ['n_obs_sp', 'obs_temp_mean', 'obs_temp_std', 'obs_temp_mean_airdif',
#        'dif_surface_area', 'dif_sw_mean', 'dif_sw_mean_au', 'dif_lw_std_au',
#        'dif_at_std_au', 'dif_snow_mean_au', 'dif_zero_temp_doy',
#        'perc_dif_surface_area']

#NO TRAN NO PRETRAIN
# feats = ['n_obs', 'n_obs_sp', 'n_obs_su', 'n_obs_au', 'obs_temp_mean',
#        'obs_temp_skew', 'obs_temp_kurt', 'obs_temp_mean_airdif',
#        'dif_surface_area', 'dif_lw_std', 'dif_at_std', 'dif_snow_mean',
#        'dif_rh_std_su', 'dif_snow_mean_su', 'dif_sw_mean_au', 'dif_lw_mean_au',
#        'dif_lw_std_au', 'dif_at_std_au', 'dif_rh_std_au', 'dif_rain_mean_au',
#        'dif_snow_mean_au', 'dif_lw_std_wi', 'dif_rain_mean_wi',
#        'perc_dif_surface_area']

#NO TRAN W PRETRAIN
# feats = ['n_obs', 'n_obs_sp', 'n_obs_su', 'n_obs_au', 'obs_temp_mean',
#        'obs_temp_std', 'obs_temp_skew', 'obs_temp_kurt', 'ad_zero_temp_doy',
#        'obs_temp_mean_airdif', 'dif_SDF', 'dif_k_d', 'dif_surface_area',
#        'dif_sw_mean', 'dif_lw_std', 'dif_at_std', 'dif_rh_mean', 'dif_rh_std',
#        'dif_rain_mean', 'dif_rain_std', 'dif_sw_std_sp', 'dif_at_mean_sp',
#        'dif_sw_mean_su', 'dif_sw_std_su', 'dif_lw_std_su', 'dif_rh_mean_su',
#        'dif_rh_std_su', 'dif_ws_mean_su', 'dif_rain_mean_su',
#        'dif_snow_mean_su', 'dif_snow_std_su', 'dif_sw_mean_au',
#        'dif_lw_std_au', 'dif_at_mean_au', 'dif_at_std_au', 'dif_rh_mean_au',
#        'dif_rh_std_au', 'dif_ws_mean_au', 'dif_rain_mean_au',
#        'dif_rain_std_au', 'dif_snow_mean_au', 'dif_sw_std_wi',
#        'dif_rh_mean_wi', 'dif_ws_mean_wi', 'dif_zero_temp_doy',
#        'dif_ws_sp_mix', 'perc_dif_surface_area', 'dif_sqrt_surface_area']

###################################################################################


#######################################################################3
#paste hyperparameters found in "pbmtl_hyperparameter_search.py" here
#
#
# n_estimators = 300 #full models
n_estimators = 1000 #no tran no pre
objective = 'reg:squarederror'
# learning_rate = .05 #no tran no pre and full model
learning_rate = .025 #no tran w pre
colsample_bytree=.7
max_depth=6
min_child_weight=11
subsample=.8
#####################################################################



########################
##########################
#metamodel training code
##########################
#######################






#compile training data
train_df = pd.DataFrame()
for _, lake_id in enumerate(train_lakes):
    new_df = pd.DataFrame()

    #get performance results (metatargets), filter out target as source
    lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results",header=None,names=['source_id','rmse'])
    # lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results_noTran_noPre",header=None,names=['source_id','rmse'])
    # lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results_noTran_wPre",header=None,names=['source_id','rmse'])
    lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

    #get metadata differences between target and all the sources
    lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
    lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
    lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
    lake_df_res['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res['source_id'].values]
    lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id2')
    new_df = lake_df
    train_df = pd.concat([train_df, new_df], ignore_index=True)



#train model
X_trn = pd.DataFrame(train_df[feats])
y_trn = np.array([float(x) for x in np.ravel(pd.DataFrame(train_df['rmse']))])
# model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr)
# print("Model training in progress...")
model = xgb.XGBRegressor(booster='gbtree',n_estimators=n_estimators,objective=objective,learning_rate=learning_rate,colsample_bytree=colsample_bytree,
             max_depth=6,min_child_weight=min_child_weight,subsample=subsample)
# model = RandomForestRegressor(n_estimators=n_estimators)
print("Training metamodel...")
model.fit(X_trn, y_trn)
dump(model, save_file_path)
print("Training Complete, saved to ", save_file_path)
