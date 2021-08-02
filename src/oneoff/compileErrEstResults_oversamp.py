import pandas as pd
import numpy as np
import pdb
import os

#load metadata
metadata = pd.read_csv("../../metadata/lake_metadata.csv")

#trim to observed lakes
metadata = metadata[metadata['num_obs'] > 0]# obs = pd.read_feather("../../data/raw/obs/surface_lake_temp_daily_020421.feather")
obs = pd.read_csv("../../data/raw/obs/lake_surface_temp_obs.csv")

site_ids = np.unique(obs['site_id'].values)
print(len(site_ids), ' lakes')
# site_ids = metadata['site_id'].values #CHANGE DIS----------
n_folds = 5

combined_df = pd.DataFrame()
combined_lm = pd.DataFrame()
combined_gb = pd.DataFrame()
combined_ea = pd.DataFrame()
folds_arr = np.arange(n_folds)+1 
for k in folds_arr: #CHANGE DIS----------------

	print("fold ",k)
	# lm_df = pd.read_feather("../../results/lm_conus_022221_fold"+str(k)+".feather")
	lm_df = pd.read_feather("../../results/bachmann_071121_fold"+str(k)+".feather")
	# gb_df = pd.read_feather("../../results/xgb_conus_022221_fold"+str(k)+".feather")

	# ea_df = pd.read_feather("../../results/err_est_outputs_2layer128hid_2.4rmse_EALSTM_fold"+str(k)+".feather")
	# ea_df = pd.read_feather("../../results/err_est_outputs_072621_EALSTMgrouploss_fold"+str(k)+".feather")
	ea_df = pd.read_feather("../../results/err_est_outputs_072621_EALSTM_fold"+str(k)+".feather")
	# ea_df_gl = pd.read_feather("../../results/err_est_outputs_072621_EALSTMgrouploss_fold"+str(k)+".feather")
	# ea_df_fgl = pd.read_feather("../../results/err_est_outputs_072921_EALSTMfullgrouploss_fold"+str(k)+".feather")
	
	# ea_df_os1 = pd.read_feather("../../results/err_est_outputs_072621_EALSTM_fold"+str(k)+"_oversamp1.feather")

	# ea_df = pd.read_feather("../../results/err_est_outputs_070221_EALSTM_fold"+str(k)+".feather")
	# pdb.set_trace()
	# ea_df.drop(ea_df[ea_df['Date'] < gb_date_df['Date'].min()].index,axis=0,inplace=True)
	# assert ea_df.shape[0] == lm_df.shape[0]
	ea_df = pd.merge(ea_df,lm_df,left_on=['Date','site_id'],right_on=['date','site_id'],how='left')
	# ea_df = pd.merge(ea_df,ea_df_os1,left_on=['Date','site_id'],right_on=['Date','site_id'],how='left')
	combined_ea = combined_ea.append(ea_df)
	combined_ea.reset_index(inplace=True,drop=True)
	# combined_lm = combined_lm.append(lm_df)
	# combined_lm.reset_index(inplace=True,drop=True)

combined_df['Date'] = combined_ea['Date']
combined_df['site_id'] = combined_ea['site_id']
combined_df['wtemp_predicted-ealstm'] = combined_ea['wtemp_predicted']
# combined_df['wtemp_predicted-ealstm_os1'] = combined_ea['wtemp_predicted']
combined_df['wtemp_predicted-linear_model'] = combined_ea['temp_pred_lm']
# combined_df['wtemp_actual'] = combined_ea['wtemp_actual']
combined_df['wtemp_actual'] = combined_ea['wtemp_actual']
combined_df.reset_index(inplace=True)
combined_df.to_feather("../../results/all_outputs_and_obs_072921.feather")
combined_df.to_csv("../../results/all_outputs_and_obs_072921.csv")

combined_df = pd.read_feather("../../results/all_outputs_and_obs_072921.feather")
print(">32 RMSE")
print(np.sqrt(((combined_df[combined_df['wtemp_actual']>32]['wtemp_predicted-ealstm'] - combined_df[combined_df['wtemp_actual']>32]['wtemp_actual'])**2).mean()))
print(">33 RMSE")
print(np.sqrt(((combined_df[combined_df['wtemp_actual']>33]['wtemp_predicted-ealstm'] - combined_df[combined_df['wtemp_actual']>33]['wtemp_actual'])**2).mean()))
print(">34 RMSE")
print(np.sqrt(((combined_df[combined_df['wtemp_actual']>34]['wtemp_predicted-ealstm'] - combined_df[combined_df['wtemp_actual']>34]['wtemp_actual'])**2).mean()))

pdb.set_trace()

per_site_df = pd.DataFrame(columns=['site_id','n_obs','rmse_ealstm','rmse_lm'])
# per_site_df = pd.DataFrame(columns=['site_id','n_obs','rmse_ealstm','rmse_xgboost'])
for i,site_id in enumerate(site_ids):
	print(i)
	per_site_res = combined_df[(combined_df['site_id'] == site_id)]
	# per_site_res2 = per_site_res[per_site_res['wtemp_actual'] > 32]
	site_df = pd.DataFrame(columns=['site_id','n_obs','rmse_ealstm','rmse_lm'])
	# site_df = pd.DataFrame(columns=['site_id','n_obs','rmse_ealstm','rmse_xgboost'])
	site_df['rmse_ealstm'] = [np.sqrt(((per_site_res['wtemp_predicted-ealstm'] - per_site_res['wtemp_actual']) ** 2).mean())]
	# site_df['rmse_ealstm_gt32'] = [np.sqrt(((per_site_res2['wtemp_predicted-ealstm'] - per_site_res2['wtemp_actual']) ** 2).mean())]
	# site_df['rmse_ealstm_os1'] = [np.sqrt(((per_site_res['wtemp_predicted-ealstm'] - per_site_res['wtemp_actual']) ** 2).mean())]
	site_df['rmse_lm'] = [np.sqrt(((per_site_res['wtemp_predicted-linear_model'] - per_site_res['wtemp_actual']) ** 2).mean())]
	if np.isnan(site_df['rmse_ealstm']).any():
		continue
	site_df['site_id'] = [site_id]
	site_df['n_obs'] = [per_site_res.shape[0]]
	per_site_df = per_site_df.append(site_df)

print("median RMSE")
print(per_site_df['rmse_ealstm'].quantile(.5))
print("Q1 RMSE")
print(per_site_df['rmse_ealstm'].quantile(.25))
print("Q2 RMSE")
print(per_site_df['rmse_ealstm'].quantile(.75)
per_site_df.reset_index(inplace=True)
per_site_df.to_csv("../../results/err_per_site_072921.csv")
# per_site_df.to_feather("../../results/err_per_site_wBachmann_071121.feather")
