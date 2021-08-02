import pdb
import numpy as np
# from imblearn.over_sampling import SMOTE 

k_arr = np.arange(5)+1

for k in k_arr:
	data = np.load("../evaluate/ealstm_trn_data_072621_5fold_k"+str(int(k))+".npy")
	ind1 = np.where(data[:,:,-1]>32)[0]
	# augment = data[ind1,:,:]
	augment = np.repeat(data[ind1,:,:],10,axis=0)


	#remove non-hot obs in augment
	ind3 = np.where(augment[:,:,-1] < 32)
	augment[ind3] = np.nan

	augment[:,:,:-1] = augment[:,:,:-1] + (.1**.5)*np.random.randn(augment.shape[0],augment.shape[1],augment.shape[2]-1)
	augment[:,:,-1] = augment[:,:,-1] + np.random.randn(augment.shape[0],augment.shape[1])

	data = np.concatenate((data,augment), axis=0)
	np.save("../evaluate/ealstm_trn_data_oversamp2_k"+str(int(k))+".npy",data)