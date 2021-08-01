import pdb
import numpy as np
# from imblearn.over_sampling import SMOTE 

k_arr = np.arange(5)+1

for k in k_arr:
	data = np.load("../evaluate/ealstm_trn_data_072621_5fold_k"+str(int(k))+".npy")
	ind1 = np.where(data[:,:,-1]>33)[0]
	augment = data[ind1,:,:]

	pdb.set_trace()
	#remove nonhot obs from augset
	# data[ind1,:,:-1] = x + (0.1**0.5)*torch.randn(5, 10, 20)	
	data = np.concatenate((data,augment), axis=0)
	np.save("../evaluate/ealstm_trn_data_072621_hotaug_k"+str(int(k))+".npy",data)