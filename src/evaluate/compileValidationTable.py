import pandas as pd
import numpy as np 
import pdb
import sys
import os



#######################################
# creates validation table in CSV format 
#
# this script assumes download of lake_surface_temp_preds.csv from 
# the data release (https://www.sciencebase.gov/catalog/item/60341c3ed34eb12031172aa6)
#
##########################################################


#load data
df = pd.read_csv("lake_surface_temp_preds.csv")
vals = df['wtemp_ERA5'].values[:]
df['wtemp_ERA5'] = vals[:]+3.47
df['wtemp_ERA5b'] = vals[:]
pdb.set_trace()
site_ids = np.unique(df['site_id'].values)
meta = pd.read_csv("../../metadata/lake_metadata.csv")
meta = meta[meta['num_obs']>0]

#calculate error per site
# err_per_site_ea = [np.abs((df[df['site_id']==i_d]['wtemp_EALSTM']-df[df['site_id']==i_d]['wtemp_obs'])).mean()for i_d in site_ids]
err_per_site_ea = [np.sqrt(((df[df['site_id']==i_d]['wtemp_EALSTM']-df[df['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in site_ids]
err_per_site_LM = [np.sqrt(np.nanmean((df[df['site_id']==i_d]['wtemp_LM']-df[df['site_id']==i_d]['wtemp_obs'])**2)) for i_d in site_ids]
err_per_site_e5 = [np.sqrt(((df[df['site_id']==i_d]['wtemp_ERA5']-df[df['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in site_ids]
err_per_site_e5b = [np.sqrt(((df[df['site_id']==i_d]['wtemp_ERA5b']-3.46-df[df['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in site_ids]
err_per_site = pd.DataFrame()
err_per_site['site_id'] = site_ids
err_per_site['RMSE_EA'] = err_per_site_ea
print("median err_per_site EALSTM: ",np.median(err_per_site['RMSE_EA'].values))
err_per_site['RMSE_ERA5'] = err_per_site_e5
print("median err_per_site ERA5*: ",np.median(err_per_site['RMSE_ERA5'].values))
err_per_site['RMSE_ERA5b'] = err_per_site_e5b
print("median err_per_site ERA5: ",np.median(err_per_site['RMSE_ERA5b'].values))

err_per_site['RMSE_LM'] = err_per_site_LM

#calc bias per site in 4 different ranges
t1 = df[df['wtemp_obs']<=10]
sites_t1 = np.unique(t1['site_id'].values)
t2 = df[(df['wtemp_obs']>=10)&(df['wtemp_obs']<=20)]
sites_t2 = np.unique(t2['site_id'].values)
t3 = df[(df['wtemp_obs']>=20)&(df['wtemp_obs']<=30)]
sites_t3 = np.unique(t3['site_id'].values)
t4 = df[(df['wtemp_obs']>=30)]
sites_t4 = np.unique(t4['site_id'].values)

print("calc bias per site t1")
bias_per_site_ea_t1 = [np.sqrt(((t1[t1['site_id']==i_d]['wtemp_EALSTM']-t1[t1['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t1]
bias_per_site_lm_t1 = [np.sqrt(np.nanmean((t1[t1['site_id']==i_d]['wtemp_LM']-t1[t1['site_id']==i_d]['wtemp_obs'])**2)) for i_d in sites_t1]
bias_per_site_e5_t1 = [np.sqrt(((t1[t1['site_id']==i_d]['wtemp_ERA5']-t1[t1['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t1]
bias_per_site_e5b_t1 = [np.sqrt(((t1[t1['site_id']==i_d]['wtemp_ERA5b']-t1[t1['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t1]
bias_per_site_t1 = pd.DataFrame()
bias_per_site_t1['site_id'] = sites_t1
bias_per_site_t1['EA'] = bias_per_site_ea_t1
bias_per_site_t1['E5'] = bias_per_site_e5_t1
bias_per_site_t1['E5b'] = bias_per_site_e5b_t1
bias_per_site_t1['LM'] = bias_per_site_lm_t1
bias_per_site_t1.to_feather("bias_per_site_t1")

# print("calc bias per site t2")
# bias_per_site_ea_t2 = [np.sqrt(((t2[t2['site_id']==i_d]['wtemp_EALSTM']-t2[t2['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t2]
# bias_per_site_lm_t2 = [np.sqrt(np.nanmean((t1[t1['site_id']==i_d]['wtemp_LM']-t2[t2['site_id']==i_d]['wtemp_obs'])**2)) for i_d in sites_t2]
# bias_per_site_e5_t2 = [np.sqrt(((t2[t2['site_id']==i_d]['wtemp_ERA5']-t2[t2['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t2]
# bias_per_site_e5b_t2 = [np.sqrt(((t2[t2['site_id']==i_d]['wtemp_ERA5b']-t2[t2['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t2]
# bias_per_site_t2 = pd.DataFrame()
# bias_per_site_t2['site_id'] = sites_t2
# bias_per_site_t2['EA'] = bias_per_site_ea_t2
# bias_per_site_t2['E5'] = bias_per_site_e5_t2
# bias_per_site_t2['E5b'] = bias_per_site_e5b_t2
# bias_per_site_t2['LM'] = bias_per_site_lm_t2
# bias_per_site_t2.to_feather("bias_per_site_t2")

# print("calc bias per site t3")
# bias_per_site_ea_t3 = [np.sqrt(((t3[t3['site_id']==i_d]['wtemp_EALSTM']-t3[t3['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t3]
# bias_per_site_lm_t3 = [np.sqrt(np.nanmean((t3[t3['site_id']==i_d]['wtemp_LM']-t3[t3['site_id']==i_d]['wtemp_obs'])**2)) for i_d in sites_t3]
# bias_per_site_e5_t3 = [np.sqrt(((t3[t3['site_id']==i_d]['wtemp_ERA5']-t3[t3['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t3]
# bias_per_site_e5b_t3 = [np.sqrt(((t3[t3['site_id']==i_d]['wtemp_ERA5b']-t3[t3['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t3]
# bias_per_site_t3 = pd.DataFrame()
# bias_per_site_t3['site_id'] = sites_t3
# bias_per_site_t3['EA'] = bias_per_site_ea_t3
# bias_per_site_t3['E5'] = bias_per_site_e5_t3
# bias_per_site_t3['E5b'] = bias_per_site_e5b_t3
# bias_per_site_t3['LM'] = bias_per_site_lm_t3
# bias_per_site_t3.to_feather("bias_per_site_t3")

# print("calc bias per site t4")
# bias_per_site_ea_t4 = [np.sqrt(((t4[t4['site_id']==i_d]['wtemp_EALSTM']-t4[t4['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t4]
# bias_per_site_lm_t4 = [np.sqrt(np.nanmean((t1[t1['site_id']==i_d]['wtemp_LM']-t4[t4['site_id']==i_d]['wtemp_obs'])**2)) for i_d in sites_t4]
# bias_per_site_e5_t4 = [np.sqrt(((t4[t4['site_id']==i_d]['wtemp_ERA5']-t4[t4['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t4]
# bias_per_site_e5b_t4 = [np.sqrt(((t4[t4['site_id']==i_d]['wtemp_ERA5b']-t4[t4['site_id']==i_d]['wtemp_obs'])**2).mean())for i_d in sites_t4]
# bias_per_site_t4 = pd.DataFrame()
# bias_per_site_t4['site_id'] = sites_t4
# bias_per_site_t4['EA'] = bias_per_site_ea_t4
# bias_per_site_t4['E5'] = bias_per_site_e5_t4
# bias_per_site_t4['E5b'] = bias_per_site_e5b_t4
# bias_per_site_t4['LM'] = bias_per_site_lm_t4
# bias_per_site_t4.to_feather("bias_per_site_t4")

err_per_site.to_feather("./err_per_site")
err_per_site = pd.read_feather('./err_per_site')
bias_per_site_t1 = pd.read_feather("bias_per_site_t1")
bias_per_site_t2 = pd.read_feather("bias_per_site_t2")
bias_per_site_t3 = pd.read_feather("bias_per_site_t3")
bias_per_site_t4 = pd.read_feather("bias_per_site_t4")

ov_RMSE_EA = np.sqrt(((df['wtemp_EALSTM']-df['wtemp_obs'])**2).mean())
ov_RMSE_E5 = np.sqrt(((df['wtemp_ERA5']-df['wtemp_obs'])**2).mean())
ov_RMSE_E5b = np.sqrt(((df['wtemp_ERA5b']-3.46-df['wtemp_obs'])**2).mean())
ov_RMSE_LM = np.sqrt(np.nanmean((df['wtemp_LM']-df['wtemp_obs'])**2))

md_RMSE_EA = np.median(err_per_site['RMSE_EA'])
md_RMSE_E5 = np.median(err_per_site['RMSE_ERA5'])
md_RMSE_E5b = np.median(err_per_site['RMSE_ERA5b'])
md_RMSE_LM = np.nanmedian(err_per_site['RMSE_LM'])

pdb.set_trace()
#row label
rows = ['EA-LSTM', 'ERA5*','ERA5', 'LM']
ha = 10000
lakes_area1 = meta[(meta['area_m2']< 10*ha)]['site_id'].values 
lakes_area2 = meta[(meta['area_m2'] > 10*ha)&(meta['area_m2']< 100*ha)]['site_id'].values #n=69548
lakes_area3 = meta[(meta['area_m2'] > 100*ha)&(meta['area_m2']< 1000*ha)]['site_id'].values #n=8631
lakes_area4 = meta[(meta['area_m2'] > 1000*ha)]['site_id'].values #n=1451



EA_a1_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area1)]['RMSE_EA'])
E5_a1_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area1)]['RMSE_ERA5'])
E5b_a1_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area1)]['RMSE_ERA5b'])
LM_a1_RMSE = np.nanmedian(err_per_site[np.isin(site_ids,lakes_area1)]['RMSE_LM'])

EA_a2_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area2)]['RMSE_EA'])
E5_a2_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area2)]['RMSE_ERA5'])
E5b_a2_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area2)]['RMSE_ERA5b'])
LM_a2_RMSE = np.nanmedian(err_per_site[np.isin(site_ids,lakes_area2)]['RMSE_LM'])

EA_a3_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area3)]['RMSE_EA'])
E5_a3_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area3)]['RMSE_ERA5'])
E5b_a3_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area3)]['RMSE_ERA5b'])
LM_a3_RMSE = np.nanmedian(err_per_site[np.isin(site_ids,lakes_area3)]['RMSE_LM'])

EA_a4_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area4)]['RMSE_EA'])
E5_a4_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area4)]['RMSE_ERA5'])
E5b_a4_RMSE = np.median(err_per_site[np.isin(site_ids,lakes_area4)]['RMSE_ERA5b'])
LM_a4_RMSE = np.nanmedian(err_per_site[np.isin(site_ids,lakes_area4)]['RMSE_LM'])


#bias per method in diff temp ranges
EA_t1_bias = np.median(t1['wtemp_EALSTM']-t1['wtemp_obs'])
E5_t1_bias = np.median(t1['wtemp_ERA5']-t1['wtemp_obs'])
E5b_t1_bias = np.median(t1['wtemp_ERA5b']-3.46-t1['wtemp_obs'])
LM_t1_bias = np.nanmedian(t1['wtemp_LM']-t1['wtemp_obs'])

EA_t2_bias = np.median(t2['wtemp_EALSTM']-t2['wtemp_obs'])
E5_t2_bias = np.median(t2['wtemp_ERA5']-t2['wtemp_obs'])
E5b_t2_bias = np.median(t2['wtemp_ERA5b']-3.46-t2['wtemp_obs'])
LM_t2_bias = np.nanmedian(t2['wtemp_LM']-t2['wtemp_obs'])

EA_t3_bias = np.median(t3['wtemp_EALSTM']-t3['wtemp_obs'])
E5_t3_bias = np.median(t3['wtemp_ERA5']-t3['wtemp_obs'])
E5b_t3_bias = np.median(t3['wtemp_ERA5b']-3.46-t3['wtemp_obs'])
LM_t3_bias = np.nanmedian(t3['wtemp_LM']-t3['wtemp_obs'])

EA_t4_bias = np.median(t4['wtemp_EALSTM']-t4['wtemp_obs'])
E5_t4_bias = np.median(t4['wtemp_ERA5']-t4['wtemp_obs'])
E5b_t4_bias = np.median(t4['wtemp_ERA5b']-3.46-t4['wtemp_obs'])
LM_t4_bias = np.nanmedian(t4['wtemp_LM']-t4['wtemp_obs'])

#bias per lake in diff temp ranges
# EA_t1_bias = np.median(bias_per_site_ea_t1[np.isin(sites_t1,lakes_area1)]['bias_EA'])
# E5_t1_bias = np.median(err_per_site[np.isin(site_ids,lakes_area1)]['bias_ERA5'])
# LM_t1_bias = np.nanmedian(err_per_site[np.isin(site_ids,lakes_area1)]['bias_LM'])

# EA_t2_bias = np.median(err_per_site[np.isin(site_ids,lakes_area2)]['bias_EA'])
# E5_t2_bias = np.median(err_per_site[np.isin(site_ids,lakes_area2)]['bias_ERA5'])
# LM_t2_bias = np.nanmedian(err_per_site[np.isin(site_ids,lakes_area2)]['bias_LM'])

# EA_t3_bias = np.median(err_per_site[np.isin(site_ids,lakes_area3)]['bias_EA'])
# E5_t3_bias = np.median(err_per_site[np.isin(site_ids,lakes_area3)]['bias_ERA5'])
# LM_t3_bias = np.nanmedian(err_per_site[np.isin(site_ids,lakes_area3)]['bias_LM'])

# EA_t4_bias = np.median(err_per_site[np.isin(site_ids,lakes_area4)]['bias_EA'])
# E5_t4_bias = np.median(err_per_site[np.isin(site_ids,lakes_area4)]['bias_ERA5'])
# LM_t4_bias = np.nanmedian(err_per_site[np.isin(site_ids,lakes_area4)]['bias_LM'])

#row arrays to fill
cols = ['Median lake-specific RMSE','Overall RMSE',
		'Lakes <10 ha Median RMSE (n=1946)','Lakes 10-100 ha Median RMSE (n=6707)',
		'Lakes 100-1000 ha Median RMSE (n=2949)','Lakes > 1000 ha Median RMSE (n=685)',
		'Median Bias (0-10 deg C obs) (28,196 obs)','Median Bias (10-20 deg C obs) (98,298 obs)',
		'Median Bias (20-30 deg C obs) (170,114 obs)','Median Bias (30-40 deg C obs) (15,451 obs)']

EA_arr = [md_RMSE_EA,ov_RMSE_EA,\
		  EA_a1_RMSE,EA_a2_RMSE,\
		  EA_a3_RMSE,EA_a4_RMSE,
		  EA_t1_bias,EA_t2_bias,
		  EA_t3_bias,EA_t4_bias]
E5_arr = [md_RMSE_E5,ov_RMSE_E5,\
          E5_a1_RMSE,E5_a2_RMSE,\
          E5_a3_RMSE,E5_a4_RMSE,
          E5_t1_bias,E5_t2_bias,
		  E5_t3_bias,E5_t4_bias]
E5b_arr = [md_RMSE_E5b,ov_RMSE_E5b,\
          E5b_a1_RMSE,E5b_a2_RMSE,\
          E5b_a3_RMSE,E5b_a4_RMSE,
          E5b_t1_bias,E5b_t2_bias,
		  E5b_t3_bias,E5b_t4_bias]
LM_arr = [md_RMSE_LM,ov_RMSE_LM,\
          LM_a1_RMSE,LM_a2_RMSE,\
          LM_a3_RMSE,LM_a4_RMSE,
          LM_t1_bias,LM_t2_bias,
		  LM_t3_bias,LM_t4_bias]
df = pd.DataFrame([EA_arr, E5_arr, E5b_arr, LM_arr], rows, cols)

df.to_csv("validationTable.csv")





