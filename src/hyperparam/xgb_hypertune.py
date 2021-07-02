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
from sklearn.model_selection import GridSearchCV, cross_val_score

##################################################################3
# (July 2021 - Jared) - tune XGB Model for each fold. Takes one command line 
# argument saying which fold to tune for
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#load metadata
metadata = pd.read_csv("../../metadata/lake_metadata.csv")

#trim to observed lakes
metadata = metadata[metadata['num_obs'] > 0]


columns = ['Surface_Area','Latitude','Longitude', 
     'Elevation','ShortWave','LongWave','AirTemp','WindSpeedU','WindspeedV',
     'Surface_Temp']

k = int(sys.argv[1])
param_search = True


train_lakes = metadata[metadata['cluster_id']!=k]['site_id'].values

train_df = pd.DataFrame(columns=columns)

for ct, lake_id in enumerate(train_lakes):
    print(" assembling training lake ",ct,"/",len(train_lakes),": ",lake_id)

    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features.npy")
    labs = np.load("../../data/processed/"+lake_id+"/obs.npy")
    data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]
    y = data[:,-1]
    inds = np.where(np.isfinite(y))[0]
    if inds.shape[0] == 0:
      continue
    # inds = inds[np.where(inds > farthest_lookback)[0]]
    X = np.array([X[i,:] for i in inds],dtype = np.float)
    y = y[inds]
    #remove days without obs
    data = np.concatenate((X,y.reshape(len(y),1)),axis=1)
    new_df = pd.DataFrame(columns=columns,data=data)
    train_df = pd.concat([train_df, new_df], ignore_index=True)

X = train_df[columns[:-1]].values
y = np.ravel(train_df[columns[-1]].values)
print("train set dimensions: ",X.shape)


gbm = xgb.XGBRegressor(booster='gbtree')
nfolds = 3
parameters = {'objective':['reg:squarederror'],
              'learning_rate': [.025, 0.05,.10], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5000,10000,15000], #number of trees, change it to 1000 for better results
              }
def gb_param_selection(X, y, nfolds):
    grid_search = GridSearchCV(gbm, parameters, n_jobs=-1, cv=nfolds,verbose=1)
    grid_search.fit(X, y)
    return grid_search.best_params_

print("commencing parameter tuning")
parameters = gb_param_selection(X, y, nfolds)
print(parameters)


