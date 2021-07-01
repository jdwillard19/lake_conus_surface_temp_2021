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
# (July 2021 - Jared) - tune XGB Model for each fold
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#file to save model  to


# metadata = pd.read_csv("../../metadata/conus_source_metadata.csv")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_021521_wCluster.csv")

# train_lakes = metadata['site_id'].values

#############################
#load data
# train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")



columns = ['Surface_Area','Latitude','Longitude', 
     'Elevation','ShortWave','LongWave','AirTemp','WindSpeedU','WindspeedV',
     'Surface_Temp']

k = int(sys.argv[1])
param_search = True


train_lakes = metadata[metadata['5fold_fold']!=k]['site_id'].values

# lakenames = metadata['site_id'].values
# test_lakes = metadata[metadata['5fold_fold']==k]['site_id'].values
# assert(np.isin(train_lakes,test_lakes,invert=True).all())
train_df = pd.DataFrame(columns=columns)
# test_df = pd.DataFrame(columns=columns)

for ct, lake_id in enumerate(train_lakes):
    # if ct %100 == 0:
    print(" assembling training lake ",ct,"/",len(train_lakes),": ",lake_id)
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features_ea_conus_021621.npy")
    labs = np.load("../../data/processed/"+lake_id+"/full.npy")
    # dates = np.load("../../data/processed/"+name+"/dates.npy")
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

    # data = data[np.where(np.isfinite(data[:,-1]))]
    new_df = pd.DataFrame(columns=columns,data=data)
    train_df = pd.concat([train_df, new_df], ignore_index=True)

X = train_df[columns[:-1]].values
y = np.ravel(train_df[columns[-1]].values)
print("train set dimensions: ",X.shape)

if param_search:
    gbm = xgb.XGBRegressor(booster='gbtree')
    nfolds = 3
    parameters = {'objective':['reg:squarederror'],
                  'learning_rate': [.025, 0.05], #so called `eta` value
                  'max_depth': [6],
                  'min_child_weight': [11],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [5000,10000], #number of trees, change it to 1000 for better results
                  # 'n_estimators': [5000,10000,15000], #number of trees, change it to 1000 for better results
                  }
    def gb_param_selection(X, y, nfolds):
        # ests = np.arange(1000,6000,600)
        # lrs = [.05,.01]
        # max_d = [3, 5]
        # param_grid = {'n_estimators': ests, 'learning_rate' : lrs}
        # grid_search = GridSearchCV(gbm, param_grid, cv=nfolds, n_jobs=-1,verbose=1)
        grid_search = GridSearchCV(gbm, parameters, n_jobs=-1, cv=nfolds,verbose=1)
        grid_search.fit(X, y)
        # print(grid_search.best_params_)
        return grid_search.best_params_


    parameters = gb_param_selection(X, y, nfolds)
    print(parameters)


