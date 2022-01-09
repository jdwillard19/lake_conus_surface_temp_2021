import pandas as pd
import numpy as np
import pdb
import sys
import plotly.express as px
from scipy import stats
import matplotlib.pyplot as plt

import os


def centeroidnp(lats, lons):
    length = len(lats)
    sum_x = np.sum(lats)
    sum_y = np.sum(lons)
    return sum_x/length, sum_y/length

metadata = pd.read_csv("../../metadata/lake_metadata.csv")
err_per_site = pd.read_csv("../../results/err_per_site_wBachmann_072621.csv")

cluster_id_int = []
cluster_col = metadata['cluster_id'].values
for i in range(metadata.shape[0]):
	val = cluster_col[i]
	if np.isnan(val):
		cluster_id_int.append(val)
	else:
		cluster_id_int.append(int(val))

cluster_id_str = []
for i in range(metadata.shape[0]):
	val = cluster_col[i]
	if np.isnan(val):
		cluster_id_str.append(val)
	else:
		cluster_id_str.append(str(val))

metadata['cluster_id'] = cluster_id_str

all_lats = metadata['lake_lat_deg'].values
all_lons = metadata['lake_lon_deg'].values

#create df to plot
df = pd.DataFrame()
df['cluster'] = np.arange(1,17).astype(int)
#add cluster summary 

#vars to append to for above df
avg_areas = []
lats = []
lons = []
centroid_xs = []
centroid_ys = []
err_per_cluster = []
for i in range(1,17):
	cluster_str = str(i)+".0"
	site_ids = metadata[metadata['cluster_id']==cluster_str]['site_id']
	avg_areas.append(metadata[metadata['cluster_id']==cluster_str]['area_m2'].mean())
	lons = metadata[metadata['cluster_id']==cluster_str]['lake_lon_deg'].values
	lats = metadata[metadata['cluster_id']==cluster_str]['lake_lat_deg'].values
	centroid_x,centroid_y = centeroidnp(lats,lons)
	centroid_xs.append(centroid_x)
	centroid_ys.append(centroid_y)
	err_per_cluster.append(err_per_site[np.isin(err_per_site['site_id'],site_ids)]['rmse_ealstm'].median())
df['average_area_m2'] = avg_areas
df['centroid_y'] = centroid_xs
df['centroid_x'] = centroid_ys
df['Median per-lake RMSE'] = err_per_cluster

n_lakes_per_cluster = [metadata[metadata['cluster_id']==str(i)+'.0'].shape[0] for i in range(1,17)]
df['n_lakes'] = n_lakes_per_cluster

fig = px.scatter(df, title="Lake Cluster Centroids",
				x='centroid_x', y='centroid_y',color='cluster',hover_data=['average_area_m2', 'n_lakes','Median per-lake RMSE'])
fig.update_traces(marker=dict(size=20))
fig.add_trace(px.scatter(metadata, x='lake_lon_deg', y='lake_lat_deg',opacity=0.02,hover_data=[]).data[0])
#                   selector=dict(mode='markers'))

fig.write_html("./cluster_visual.html")


