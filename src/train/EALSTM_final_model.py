from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.init import xavier_normal_
from datetime import date
import pandas as pd
import pdb
import random
import math
import sys
import re
import os
sys.path.append('../../data')
sys.path.append('../data')
sys.path.append('../../models')
sys.path.append('../models')
from pytorch_model_operations import saveModel
import pytorch_data_operations
import datetime
from torch.utils.data import DataLoader
from pytorch_data_operations import buildLakeDataForRNN_multilakemodel_conus
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


#script start
currentDT = datetime.datetime.now()
print(str(currentDT))

####################################################
# (July 2021 - Jared) trains final EA-LSTM model
###################################################

#enable/disable cuda 
use_gpu = True 
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=10)



### debug tools
verbose = True
save = True
test = True



#####################3
#params
###########################33
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 5  #number of physical drivers
n_static_feats = 4
n_total_feats =n_static_feats+n_features
win_shift = 175 #how much to slide the window on training set each time
save = True 
grad_clip = 1.0 #how much to clip the gradient 2-norm in training
dropout = 0.
num_layers = 1
n_hidden = 256
lambda1 = 0.000


#epoch settings
n_eps = 1000
first_save_epoch = 0
targ_ep = 200
targ_rmse = 1.89

#load metadata
metadata = pd.read_csv("../../metadata/lake_metadata.csv")

#trim to observed lakes
metadata = metadata[metadata['num_obs'] > 0]


first_save_epoch = 0
epoch_since_best = 0
yhat_batch_size = 1 #obsolete since this isnt PGDL


final_output_df = pd.DataFrame()
lakenames = metadata['site_id'].values

ep_arr = []   


if not os.path.exists("./ealstm_trn_data_072621.npy"):
    (trn_data, _) = buildLakeDataForRNN_multilakemodel_conus(lakenames,\
                                                    seq_length, n_total_feats,\
                                                    win_shift = win_shift, begin_loss_ind = begin_loss_ind,\
                                                    static_feats=True,n_static_feats = 4,verbose=True,cold_filter=False) 

    np.save("./ealstm_trn_data_072621.npy",trn_data)
else:
    trn_data = torch.from_numpy(np.load("./ealstm_trn_data_072621.npy"))

print("train_data size: ",trn_data.size())

#add oversampling data
data = trn_data.numpy()

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
if ind40.shape[0] != 0:
    new_data = data[np.append(ind40,np.random.choice(ind40,int(np.round(add_40)))),:,:]
    augment = new_data
else:
    augment = np.empty((0,350,10))

add_39 = (ydata[-2]+ydata[-3])/2 - hist[-3]
print("39: ",add_39)
ind39 = np.where((data[:,:,-1]>38)&(data[:,:,-1] <= 39))[0]
new_data = data[np.append(ind39,np.random.choice(ind39,int(np.round(add_39)))),:,:]
augment = np.concatenate((augment,new_data),axis=0)

add_38 = (ydata[-3]+ydata[-4])/2 - hist[-4]
print("38: ",add_38)
ind38 = np.where((data[:,:,-1]>37)&(data[:,:,-1] <= 38))[0]
new_data = data[np.append(ind38,np.random.choice(ind38,int(np.round(add_38)))),:,:]
augment = np.concatenate((augment,new_data),axis=0)

add_37 = (ydata[-4] + ydata[-5])/2 - hist[-5]
print("37: ",add_37)
ind37 = np.where((data[:,:,-1]>36)&(data[:,:,-1] <= 37))[0]
new_data = data[np.append(ind37,np.random.choice(ind37,int(np.round(add_37)))),:,:]
augment = np.concatenate((augment,new_data),axis=0)

add_36 = (ydata[-5]+ydata[-6])/2 - hist[-6]
print("36: ",add_36)
ind36 = np.where((data[:,:,-1]>35)&(data[:,:,-1] <= 36))[0]
new_data = data[np.append(ind36,np.random.choice(ind36,int(np.round(add_36)))),:,:]
augment = np.concatenate((augment,new_data),axis=0)

add_35 = (ydata[-6]+ydata[-7])/2 - hist[-6]
print("35: ",add_35)
ind35 = np.where((data[:,:,-1]>34)&(data[:,:,-1] <= 35))[0]
new_data = data[np.append(ind35,np.random.choice(ind35,int(np.round(add_35)))),:,:]
augment = np.concatenate((augment,new_data),axis=0)

add_34 = (ydata[-7]+ydata[-8])/2 - hist[-7]
print("34: ",add_34)
ind34 = np.where((data[:,:,-1]>33)&(data[:,:,-1] <= 34))[0]
new_data = data[np.append(ind34,np.random.choice(ind34,int(np.round(add_34)))),:,:]
augment = np.concatenate((augment,new_data),axis=0)

add_33 = (ydata[-8]+ydata[-9])/2 - hist[-8]
print("33: ",add_33)
ind33 = np.where((data[:,:,-1]>32)&(data[:,:,-1] <= 33))[0]
new_data = data[np.append(ind33,np.random.choice(ind33,int(np.round(add_33)))),:,:]
augment = np.concatenate((augment,new_data),axis=0)


#remove non-hot obs in augment
ind3 = np.where(augment[:,:,-1] < 32)
augment[ind3[0],ind3[1],-1] = np.nan

#add noise optional
augment[:,:,:-1] = augment[:,:,:-1] + (.0125**.5)*np.random.randn(augment.shape[0],augment.shape[1],augment.shape[2]-1)
augment[:,:,-1] = augment[:,:,-1] + (.125**.5)*np.random.randn(augment.shape[0],augment.shape[1])

data = np.concatenate((data,augment), axis=0)
np.save("../evaluate/ealstm_trn_data_normoversample.npy",data)

trn_data = torch.from_numpy(data)
print("train_data size: ",trn_data.size())
print(len(lakenames), " lakes of data")


batch_size = int(math.floor(trn_data.size()[0])/150)




#Dataset classes
class TemperatureTrainDataset(Dataset):
#training dataset class, allows Dataloader to load both input/target
    def __init__(self, trn_data):
        self.len = trn_data.shape[0]
        self.x_data = trn_data[:,:,:-1].float()
        self.y_data = trn_data[:,:,-1].float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len




#format training data for loading
train_data = TemperatureTrainDataset(trn_data)


#format total y-hat data for loading
n_batches = math.floor(trn_data.size()[0] / batch_size)

#batch samplers used to draw samples in dataloaders
batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)


#define EA-LSTM class
"""
This file is part of the accompanying code to the manuscript:
Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)
You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""


#define LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(myLSTM_Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = n_total_feats, hidden_size=hidden_size, batch_first=True,num_layers=num_layers,dropout=dropout) #batch_first=True?
        self.out = nn.Linear(hidden_size, 1) #1?
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(num_layers, batch_size, self.hidden_size)),
                xavier_normal_(torch.empty(num_layers, batch_size, self.hidden_size)))
        if use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0,item1)
        return ret

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        return x, hidden

#method to calculate l1 norm of model
def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    # for name, p in model.named_parameters():
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            #take absolute value of weights and sum
            to_regularize.append(p.view(-1))
    l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val


class EALSTM(nn.Module):
    """Implementation of the Entity-Aware-LSTM (EA-LSTM)
    TODO: Include paper ref and latex equations
    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(EALSTM, self).__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size_dyn, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_sh = nn.Parameter(torch.FloatTensor(input_size_stat, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d, x_s):
        """[summary]
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)
            # x_s = x_s.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []
        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # calculate input gate only once because inputs are static
        bias_s_batch = (self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size()))

        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) +
                     torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n

class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initialize model.
        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None
        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        h_n = self.dropout(h_n)
        
        # last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(h_n)
        return out, h_n, c_n

    # © 2020 GitHub, Inc.
    # Terms
    # Privacy
    # Security
    # Status
    # Help

    # Contact GitHub
    # Pricing
    # API
    # Training
    # Blog
    # About




# lstm_net = myLSTM_Net(n_total_feats, n_hidden, batch_size)
lstm_net = Model(input_size_dyn=n_features,input_size_stat=n_static_feats,hidden_size=n_hidden)

#tell model to use GPU if needed
if use_gpu:
    lstm_net = lstm_net.cuda()




#define loss and optimizer
mse_criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_net.parameters(), lr=.005)#, weight_decay=0.01)

#training loop

min_mse = 99999
min_mse_tsterr = None
ep_min_mse = -1
ep_since_min = 0
best_pred_mat = np.empty(())
manualSeed = [random.randint(1, 99999999) for i in range(n_eps)]

#stop training if true
min_train_rmse = 999
min_train_ep = -1
done = False
for epoch in range(n_eps):
    if done:
        break
    # if verbose and epoch % 10 == 0:
    if verbose:
        print("train epoch: ", epoch)

    running_loss = 0.0

    #reload loader for shuffle
    #batch samplers used to draw samples in dataloaders
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)

    trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)


    #zero the parameter gradients
    optimizer.zero_grad()
    lstm_net.train(True)
    avg_loss = 0
    batches_done = 0
    ct = 0
    for m, data in enumerate(trainloader, 0):
        #now for mendota data
        #this loop is dated, there is now only one item in testloader

        #parse data into inputs and targets
        inputs = data[0].float()
        targets = data[1].float()
        targets = targets[:, begin_loss_ind:]
        # tmp_dates = tst_dates_target[:, begin_loss_ind:]
        # depths = inputs[:,:,0]


        #cuda commands
        if(use_gpu):
            inputs = inputs.cuda()
            targets = targets.cuda()

        #forward  prop
        # lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
        # lstm_net.reset_parameters()
        # h_state = None
        outputs, h_state, _ = lstm_net(inputs[:,:,n_static_feats:], inputs[:,0,:n_static_feats])
        outputs = outputs.view(outputs.size()[0],-1)

        #calculate losses
        reg1_loss = 0
        if lambda1 > 0:
            reg1_loss = calculate_l1_loss(lstm_net)


        loss_outputs = outputs[:,begin_loss_ind:]
        loss_targets = targets[:,begin_loss_ind:].cpu()


        #get indices to calculate loss
        loss_indices = np.array(np.isfinite(loss_targets.cpu()), dtype='bool_')

        if use_gpu:
            loss_outputs = loss_outputs.cuda()
            loss_targets = loss_targets.cuda()
        loss = mse_criterion(loss_outputs[loss_indices], loss_targets[loss_indices]) + lambda1*reg1_loss 
        #backward

        loss.backward(retain_graph=False)
        if grad_clip > 0:
            clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

        #optimize
        optimizer.step()

        #zero the parameter gradients
        optimizer.zero_grad()
        avg_loss += loss
        batches_done += 1

    #check for convergence
    avg_loss = avg_loss / batches_done
    train_avg_loss = avg_loss


    if verbose:
        print("train rmse loss=", avg_loss)

    if avg_loss < min_train_rmse:
        min_train_rmse = avg_loss
        print("model saved")
        save_path = "../../models/EALSTM_"+str(n_hidden)+"hid_"+str(num_layers)+"_final_072621"
        saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)

    if avg_loss < targ_rmse and epoch > targ_ep:
        print("training complete")
        break

print("training complete")

