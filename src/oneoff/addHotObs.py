import pdb
import numpy as np


k_arr = np.arange(5)+1

for k in k_arr:
	data = np.load("../evaluate/ealstm_trn_data_072621_5fold_k"+str(int(k))+".npy")
	augment = np.repeat(data[ind1,:,:],10,axis=0)
	data = np.concatenate((data,augment), axis=0)
	np.save("../evaluate/ealstm_trn_data_072621_hotaug_k"+str(int(k))+".npy",data)