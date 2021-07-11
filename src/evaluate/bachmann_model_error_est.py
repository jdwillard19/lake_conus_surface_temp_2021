import pandas as pd
import numpy as np
import pdb
import sys
import os
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import re
import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score

##################################################################3
# (July 2021 - Jared) - error estimation linear model 
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#file to save model  to
save_file_path = '../../models/lm_surface_temp.joblib'

#load metadata
metadata = pd.read_csv("../../metadata/lake_metadata.csv")

#trim to observed lakes
metadata = metadata[metadata['num_obs'] > 0]


columns = ['Latitude','Longitude', 'Elevation',
           'AirTemp','Month', 'Surface_Temp']

train_df = pd.DataFrame(columns=columns)

param_search = True
k = int(sys.argv[1])

#build training set
final_output_df = pd.DataFrame()
result_df = pd.DataFrame(columns=['site_id','temp_pred_lm','temp_actual'])

train_lakes = metadata[metadata['cluster_id']!=k]['site_id'].values
# lakenames = metadata['site_id'].values
test_lakes = metadata[metadata['cluster_id']==k]['site_id'].values
train_df = pd.DataFrame(columns=columns)
test_df = pd.DataFrame(columns=columns)

def getBachmannFeatures(data,dates):
    data = np.delete(data,(0,4,5,7,8),axis=1)
    new_x = []
    for i in range(0,data.shape[0]-7):
        new_x.append(data[i:i+8,3].mean())
    data = data[7:,:]
    dates = dates[7:]
    data[:,3] = new_x
    month = [int(str(x)[5:7]) for x in dates]
    data = np.append(data,np.expand_dims(np.array(month),axis=1),axis=1)
    return data

for ct, lake_id in enumerate(train_lakes):
    if ct %100 == 0:
      print("fold ",k," assembling training lake ",ct,"/",len(train_lakes),": ",lake_id)
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features.npy")
    labs = np.load("../../data/processed/"+lake_id+"/obs.npy")
    dates = np.load("../../data/processed/"+lake_id+"/dates.npy",allow_pickle=True)
    data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]

    X = getBachmannFeatures(X,dates)

    y = data[:,-1]
    y = y[7:] #since we're taking 
    inds = np.where(np.isfinite(y))[0]
    if inds.shape[0] == 0:
        continue
    X = np.array([X[i,:] for i in inds],dtype = np.float)
    y = y[inds]
    #remove days without obs
    data = np.concatenate((X,y.reshape(len(y),1)),axis=1)

    data = data[np.where(np.isfinite(data[:,-1]))]
    new_df = pd.DataFrame(columns=columns,data=data)
    train_df = pd.concat([train_df, new_df], ignore_index=True)
X = train_df[columns[:-1]].values
y = np.ravel(train_df[columns[-1]].values)

print("train set dimensions: ",X.shape)

#declare model and fit
model = LinearRegression()

print("Training linear model...fold ",k)
model.fit(X, y)
dump(model, save_file_path)
print("model trained and saved to ", save_file_path)

#test
for ct, lake_id in enumerate(test_lakes):
    print("fold ",k," testing test lake ",ct,"/",len(test_lakes),": ",lake_id)
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features.npy")
    labs = np.load("../../data/processed/"+lake_id+"/obs.npy")
    dates = np.load("../../data/processed/"+name+"/dates.npy")
    data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]

    X = getBachmannFeatures(X,dates)
    
    y = data[:,-1]
    y = y[7:] #since we're taking 
    inds = np.where(np.isfinite(y))[0]
    if inds.shape[0] == 0:
        continue

    X = np.array([X[i,:] for i in inds],dtype = np.float)
    y = y[inds]

    #remove days without obs
    data = np.concatenate((X,y.reshape(len(y),1)),axis=1)
    data = data[np.where(np.isfinite(data[:,-1]))]
    new_df = pd.DataFrame(columns=columns,data=data)
    X = new_df[columns[:-1]].values
    y_act = np.ravel(new_df[columns[-1]].values)
    y_pred = model.predict(X)

    df = pd.DataFrame()
    df['temp_pred_lm'] = y_pred
    df['temp_actual'] = y_act
    df['site_id'] = lake_id
    result_df = result_df.append(df)

result_df.reset_index(inplace=True)
result_df.to_feather("../../results/bachmann_071121_fold"+str(k)+".feather")
