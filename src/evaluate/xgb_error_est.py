import pandas as pd
import numpy as np
import pdb
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load
import re
import datetime
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
##################################################################3
# (Jan 2020 - Jared) - 
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

train_df = pd.DataFrame(columns=columns)

param_search = True

#build training set
k = int(sys.argv[1])
n_estimators = int(sys.argv[2])
learning_rate = float(sys.argv[3])
train = True
save_file_path = '../../models/xgb_lagless_surface_temp_fold'+str(k)+"_070221.joblib"

final_output_df = pd.DataFrame()
result_df = pd.DataFrame(columns=['site_id','temp_pred_xgb','temp_actual'])

train_lakes = metadata[metadata['cluster_id']!=k]['site_id'].values
# lakenames = metadata['site_id'].values
test_lakes = metadata[metadata['cluster_id']==k]['site_id'].values
assert(np.isin(train_lakes,test_lakes,invert=True).all())
train_df = pd.DataFrame(columns=columns)
test_df = pd.DataFrame(columns=columns)

if train:
    for ct, lake_id in enumerate(train_lakes):
        # if ct %100 == 0:
        print("fold ",k," assembling training lake ",ct,"/",len(train_lakes),": ",lake_id)
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

        # data = data[np.where(np.isfinite(data[:,-1]))]
        new_df = pd.DataFrame(columns=columns,data=data)
        train_df = pd.concat([train_df, new_df], ignore_index=True)

    X = train_df[columns[:-1]].values
    y = np.ravel(train_df[columns[-1]].values)

    print("train set dimensions: ",X.shape)
    #construct lookback feature set??
    model = xgb.XGBRegressor(booster='gbtree',n_estimators=n_estimators,learning_rate=learning_rate)
    # model = xgb.XGBRegressor(booster='gbtree',n_estimators=5000,learning_rate=.025,max_depth=6,min_child_weight=11,subsample=.8,colsample_bytree=.7,random_state=2)

    if train:
        print("Training XGB regression model...fold ",k)
        model.fit(X, y)
        dump(model, save_file_path)
        print("model trained and saved to ", save_file_path)

else:
    model = load(save_file_path)



y_pred = model.predict(X)
rmse  = np.sqrt(((y_pred-y)**2).mean())
print("trn rmse: ",rmse)
#test
for ct, lake_id in enumerate(test_lakes):
    # if ct %100 == 0:
    print("fold ",k,"  test lake ",ct,"/",len(test_lakes),": ",lake_id)
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features.npy")
    labs = np.load("../../data/processed/"+lake_id+"/obs.npy")
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
    X = new_df[columns[:-1]].values
    y_act = np.ravel(new_df[columns[-1]].values)
    y_pred = model.predict(X)

    df = pd.DataFrame()
    df['temp_pred_xgb'] = y_pred
    df['temp_actual'] = y_act
    df['site_id'] = lake_id
    result_df = result_df.append(df)
    rmse  = np.sqrt(((y_pred-y_act)**2).mean())
    print("tst rmse: ",rmse)


#calculate and save results
result_df.reset_index(inplace=True)
print("tst rmse: ",np.sqrt(((result_df['temp_pred_xgb']-result_df['temp_actual'])**2).mean()))

save_path = "../../results/xgb_lagless_062421_fold"+str(k)+".feather"
result_df.to_feather(save_path)
print("saved to ",save_path) 
