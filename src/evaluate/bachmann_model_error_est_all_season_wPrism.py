import pandas as pd
import numpy as np
import pdb
import sys
import os
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import re
import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
                    
##################################################################3
# (July 2021 - Jared) - error estimation linear model 
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#file to save model  to
save_file_path = '../../models/lm_surface_temp_all_season.joblib'

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
# lakenames = metadata['site_id'].valu
test_lakes = metadata[metadata['cluster_id']==k]['site_id'].values
train_df = pd.DataFrame(columns=columns)
test_df = pd.DataFrame(columns=columns)

def getBachmannFeatures(data,dates):
    data = np.delete(data,(0,4,5,7,8),axis=1)
    new_x = []
    for i in range(0,data.shape[0]):
        if i >= 8:
            new_x.append(data[i:i+8,3].mean())
        elif i > 0 and i < 8:
            new_x.append(data[:i+1,3].mean())
        else:
            new_x.append(data[0,3])

    data[:,3] = new_x
    month = [int(str(x)[5:7]) for x in dates]
    data = np.append(data,np.expand_dims(np.array(month),axis=1),axis=1)
    return data

X = None
y = None
if not os.path.exists("bachmannX_"+str(k)+"_all_season_wPrism.npy"):
    for ct, lake_id in enumerate(train_lakes):
        if ct %100 == 0:
          print("fold ",k," assembling training lake ",ct,"/",len(train_lakes),": ",lake_id)
        #load data
        feats = np.load("../../data/processed/"+lake_id+"/features_wPrism.npy")

        #convert K to C
        labs = np.load("../../data/processed/"+lake_id+"/obs_wPrism.npy")
        labs_old = np.load("../../data/processed/"+lake_id+"/obs.npy")
        
        dates = np.load("../../data/processed/"+lake_id+"/dates_wPrism.npy",allow_pickle=True)
        data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
        X = data[:,:-1]

        X = getBachmannFeatures(X,dates)
        X[:,3] = X[:,3] - 273.15
        y = data[:,-1]
        dates_str = [str(d) for d in dates]

        #summer only ind
        # inds = np.where(((np.core.defchararray.find(dates_str,'-06-')!=-1)|\
        #                  (np.core.defchararray.find(dates_str,'-07-')!=-1)|\
        #                  (np.core.defchararray.find(dates_str,'-08-')!=-1)|\
        #                  (np.core.defchararray.find(dates_str,'-09-')!=-1))&\
        #                   (np.isfinite(y)))[0]


        #all data ind
        inds = np.where(np.isfinite(y))[0]

        if inds.shape[0] == 0:
            print("empty")
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
    np.save("bachmannX_"+str(k)+"_all_season_wPrism",X)
    np.save("bachmannY_"+str(k)+"_all_season_wPrism",y)
else:
    X = np.load("bachmannX_"+str(k)+"_all_season_wPrism.npy",allow_pickle=True)
    y = np.load("bachmannY_"+str(k)+"_all_season_wPrism.npy",allow_pickle=True)

#convert K to C
print("train set dimensions: ",X.shape)


data = np.concatenate((X,np.expand_dims(y,-1)),axis=1)#add oversamping 
vals = data[np.isfinite(data[:,-1])][:,-1]
mu, std = norm.fit(vals) 
nsize = vals.shape[0]
# plt.hist(df['wtemp_actual'].values,bins=bins)
# Plot the histogram.

hist, bins, _ = plt.hist(vals, bins=40)
xmin, xmax = plt.xlim()

p = norm.pdf(bins, mu, std)           
new_handler, = plt.plot(bins, p/p.sum() * nsize , 'r', linewidth=2)
# new_handler now contains a Line2D object
# and the appropriate way to get data from it is therefore:
xdata, ydata = new_handler.get_data()

add_40 = (ydata[-1]+ydata[-2])/2 - hist[-2]
print("40: ",add_40)
ind40 = np.where((data[:,-1]>39)&(data[:,-1] <= 40))[0]
if ind40.shape[0] != 0:
    new_data = data[np.append(ind40,np.random.choice(ind40,int(np.round(add_40)))),:]
    augment = new_data
else:
    augment = np.empty((0,data.shape[1]))

add_39 = (ydata[-2]+ydata[-3])/2 - hist[-3]
if add_39 > 0:
    print("39: ",add_39)
    ind39 = np.where((data[:,-1]>38)&(data[:,-1] <= 39))[0]
    if ind39.shape[0] > 0:
        new_data = data[np.append(ind39,np.random.choice(ind39,int(np.round(add_39)))),:]
        augment = np.concatenate((augment,new_data),axis=0)

add_38 = (ydata[-3]+ydata[-4])/2 - hist[-4]
print("38: ",add_38)
ind38 = np.where((data[:,-1]>37)&(data[:,-1] <= 38))[0]
new_data = data[np.append(ind38,np.random.choice(ind38,int(np.round(add_38)))),:]
augment = np.concatenate((augment,new_data),axis=0)

add_37 = (ydata[-4] + ydata[-5])/2 - hist[-5]
if add_37 > 0:
    print("37: ",add_37)
    ind37 = np.where((data[:,-1]>36)&(data[:,-1] <= 37))[0]
    new_data = data[np.append(ind37,np.random.choice(ind37,int(np.round(add_37)))),:]
    augment = np.concatenate((augment,new_data),axis=0)

add_36 = (ydata[-5]+ydata[-6])/2 - hist[-6]
if add_36 > 0:
    print("36: ",add_36)
    ind36 = np.where((data[:,-1]>35)&(data[:,-1] <= 36))[0]
    new_data = data[np.append(ind36,np.random.choice(ind36,int(np.round(add_36)))),:]
    augment = np.concatenate((augment,new_data),axis=0)

add_35 = (ydata[-6]+ydata[-7])/2 - hist[-6]
if add_35 > 0:
    print("35: ",add_35)
    ind35 = np.where((data[:,-1]>34)&(data[:,-1] <= 35))[0]
    new_data = data[np.append(ind35,np.random.choice(ind35,int(np.round(add_35)))),:]
    augment = np.concatenate((augment,new_data),axis=0)

add_34 = (ydata[-7]+ydata[-8])/2 - hist[-7]
if add_34 > 0:
    print("34: ",add_34)
    ind34 = np.where((data[:,-1]>33)&(data[:,-1] <= 34))[0]
    if ind34.shape[0] > 0:
        new_data = data[np.append(ind34,np.random.choice(ind34,int(np.round(add_34)))),:]
        augment = np.concatenate((augment,new_data),axis=0)

add_33 = (ydata[-8]+ydata[-9])/2 - hist[-8]
if add_33 > 0:
    print("33: ",add_33)
    ind33 = np.where((data[:,-1]>32)&(data[:,-1] <= 33))[0]
    if ind33.shape[0] > 0:
        new_data = data[np.append(ind33,np.random.choice(ind33,int(np.round(add_33)))),:]
        augment = np.concatenate((augment,new_data),axis=0)


#remove non-hot obs in augment
ind3 = np.where(augment[:,-1] < 32)
augment[ind3[0],-1] = np.nan

#add noise optional
augment[:,:-1] = augment[:,:-1] + (.0125**.5)*np.random.randn(augment.shape[0],augment.shape[1]-1)
augment[:,-1] = augment[:,-1] + (.125**.5)*np.random.randn(augment.shape[0])

data = np.concatenate((data,augment), axis=0)

X = data[:,:-1]
y = data[:,-1]
X[:,3] = X[:,3] - 273.15



print("train set dimensions: ",X.shape)
#declare model and fit
model = LinearRegression()

print("Training linear model...fold ",k)

model.fit(X, y)
dump(model, save_file_path)
print("model trained and saved to ", save_file_path)

#tes
for ct, lake_id in enumerate(test_lakes):
    print("fold ",k," testing test lake ",ct,"/",len(test_lakes),": ",lake_id)
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features_wPrism.npy")
    labs = np.load("../../data/processed/"+lake_id+"/obs_wPrism.npy")
    dates = np.load("../../data/processed/"+lake_id+"/dates_wPrism.npy")
    data = np.concatenate((feats[:,:],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]
    dates_str = [str(d) for d in dates]

    X = getBachmannFeatures(X,dates)
    
    #convert K to C
    X[:,3] = X[:,3]-273.15

    y = data[:,-1]
    #summer only ind


    #all data ind
    inds = np.where(np.isfinite(y))[0]


    if inds.shape[0] == 0:
        continue

    X = np.array([X[i,:] for i in inds],dtype = np.float)
    y = y[inds]
    dates = dates[inds]
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
    df['date'] = dates
    result_df = result_df.append(df)

result_df.reset_index(inplace=True)
result_df.to_feather("../../results/bachmann_fold"+str(k)+"_all_season_wPrism.feather")
