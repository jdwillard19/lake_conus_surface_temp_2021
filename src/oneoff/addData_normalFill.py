import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
from scipy.stats import skew
from scipy import stats
import sys
from scipy.stats import norm
k_arr = np.arange(5)+1

for k in k_arr:
	data = np.load("../evaluate/ealstm_trn_data_072621_5fold_k"+str(int(k))+".npy")


	pdb.set_trace()
	vals = data[np.isfinite(data[:,:,-1])][:,-1]

	mu, std = norm.fit(vals) 
	nsize = vals.shape[0]
	# plt.hist(df['wtemp_actual'].values,bins=bins)
	# Plot the histogram.

	hist, bins, _ = plt.hist(vals, bins=40, color='b', edgecolor='black')
	xmin, xmax = plt.xlim()

	p = norm.pdf(bins, mu, std)           
	new_handler, = plt.plot(bins, p/p.sum() * nsize , 'r', linewidth=2)
	# new_handler now contains a Line2D object
	# and the appropriate way to get data from it is therefore:
	xdata, ydata = new_handler.get_data()

	add_40 = (ydata[-1]+ydata[-2])/2 - hist[-2]
	print("40: ",add_40)
	ind40 = np.where((data[:,:,-1]>39)&(data[:,:,-1] <= 40))[0]
	pdb.set_trace()
	# to_add_arr = np.repeat(40,add_40)
	# vals = np.append(vals,to_add_arr)

	add_39 = (ydata[-2]+ydata[-3])/2 - hist[-3]
	print("39: ",add_39)
	to_add_arr = np.repeat(38.99,add_39)
	vals = np.append(vals,to_add_arr)

	add_38 = (ydata[-3]+ydata[-4])/2 - hist[-4]
	print("38: ",add_38)
	to_add_arr = np.repeat(37.99,add_38)
	vals = np.append(vals,to_add_arr)

	add_37 = (ydata[-4] + ydata[-5])/2 - hist[-5]
	print("37: ",add_37)
	to_add_arr = np.repeat(36.99,add_37)
	vals = np.append(vals,to_add_arr)

	add_36 = (ydata[-5]+ydata[-6])/2 - hist[-6]
	print("36: ",add_36)
	to_add_arr = np.repeat(35.99,add_36)
	vals = np.append(vals,to_add_arr)

	add_35 = (ydata[-6]+ydata[-7])/2 - hist[-6]
	print("35: ",add_35)
	to_add_arr = np.repeat(34.99,add_35)
	vals = np.append(vals,to_add_arr)
	
	add_34 = (ydata[-7]+ydata[-8])/2 - hist[-7]
	print("34: ",add_34)
	to_add_arr = np.repeat(33.99,add_34)
	vals = np.append(vals,to_add_arr)
	
	add_33 = (ydata[-8]+ydata[-9])/2 - hist[-8]
	print("33: ",add_33)
	to_add_arr = np.repeat(32.99,add_33)
	vals = np.append(vals,to_add_arr)

	# ind33 = np.where((data[:,:,-1]>32)&(data[:,:,-1] <= 33))[0]
	# add_33 = 1951
	# ind34 = np.where((data[:,:,-1]>33)&(data[:,:,-1] <= 34))[0]
	# add_34 = 2282
	# ind35 = np.where((data[:,:,-1]>34)&(data[:,:,-1] <= 35))[0]
	# add_35 = 1942
	# ind36 = np.where((data[:,:,-1]>35)&(data[:,:,-1] <= 36))[0]
	# add_36 = 1369
	# ind37 = np.where((data[:,:,-1]>36)&(data[:,:,-1] <= 37))[0]
	# add_37 = 1071
	# ind38 = np.where((data[:,:,-1]>37)&(data[:,:,-1] <= 38))[0]
	# add_38 = 773
	# ind39 = np.where((data[:,:,-1]>38)&(data[:,:,-1] <= 39))[0]
	# add_39 = 549
	# ind40 = np.where((data[:,:,-1]>39)&(data[:,:,-1] <= 40))[0]
	# add_40 = 367

	# augment = data[ind1,:,:]
	augment = np.repeat(data[ind1,:,:],10,axis=0)


	#remove non-hot obs in augment
	ind3 = np.where(augment[:,:,-1] < 32)
	augment[ind3[0],ind3[1],-1] = np.nan

	augment[:,:,:-1] = augment[:,:,:-1] + (.025**.5)*np.random.randn(augment.shape[0],augment.shape[1],augment.shape[2]-1)
	augment[:,:,-1] = augment[:,:,-1] + (.25**.5)*np.random.randn(augment.shape[0],augment.shape[1])

	data = np.concatenate((data,augment), axis=0)
	np.save("../evaluate/ealstm_trn_data_oversamp8_k"+str(int(k))+".npy",data)


	np.sum[1851, 2282, 1942, 1369, 1071, 773, 549, 367]