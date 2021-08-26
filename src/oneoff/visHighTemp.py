import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt

#
site_ids = ['nhdhr_112699096']
# site_ids = ['nhdhr_143249470']



#load data
dates = np.load("../../data/processed/"+site_ids[0]+"/dates.npy",allow_pickle=True)
meta = pd.read_csv("../../metadata/lake_metadata.csv")
pdb.set_trace()
for site_id in site_ids:
	feats = np.load("../../data/processed/"+site_id+"/features.npy")
	obs = np.load("../../data/processed/"+site_id+"/obs.npy")
	# res = pd.read_feather("../../results/SWT_results/outputs_"+site_id+'.feather')

	#get fold
	fold = meta[meta['site_id']==site_id].group_id.values[0]
	res = pd.read_feather("err_est_outputs_072621_EALSTM_fold"+str(fold)+"_oversamp_norm2.feather")
	pdb.set_trace()
	pred = res['temp_pred'].values
	# obs = res['temp_actual'].values
	air_temp = feats[:,6] - 273.15
	print("max air temp: ",air_temp.max())
	for y in range(1980,2021):
		start_date = pd.Timestamp(str(y)+'-04-01').to_datetime64()
		end_date = pd.Timestamp(str(y)+'-09-01').to_datetime64()
		start_ind = np.where(dates==start_date)[0][0]
		end_ind = np.where(dates==end_date)[0][0]
		p_dates = [str(d)[:10] for d in dates[start_ind:end_ind]]
		p_obs = obs[start_ind:end_ind]
		if np.isnan(p_obs).all():
			continue
		x = np.arange(end_ind-start_ind)
		p_at = air_temp[start_ind:end_ind]
		p_pred = pred[start_ind:end_ind]

		x_p_obs = x[np.isfinite(p_obs)]
		y_p_obs = p_obs[np.isfinite(p_obs)]
		plt.plot(p_pred,color='green',label='EALSTM prediction')
		plt.plot(p_at,color='blue',label='Air Temperature')
		plt.scatter(x_p_obs,y_p_obs,c='red',marker='+',s=15,label='Observation')
		plt.xticks(ticks=x, labels=p_dates)
		plt.locator_params(axis='x', nbins=4)
		plt.xlabel("Day of Summer")
		plt.ylabel("Degrees C")
		plt.title(site_id+" May 1st - Sept 01 : "+str(y))
		plt.legend()
		plt.savefig("plot_"+site_id+"_"+str(y))
		plt.clf()