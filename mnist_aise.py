#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from aise import AISE
import pickle
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNAISE(nn.Module):
    def __init__(self,train_data,train_targets,hidden_layers,aise_params,checkpoint=None,device=DEVICE):
        super(CNNAISE, self).__init__()
        self.conv1 = nn.Conv2d(1 , 64 , 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64 , 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128,  3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128,  3, padding = 1)
        self.fc1   = nn.Linear(128 * 7 * 7, 256)
        self.fc2   = nn.Linear(256       , 256)
        self.fc3   = nn.Linear(256       , 10)

        self.device = device
        self.to(self.device)

        if checkpoint:
            self._load_state_dict(checkpoint)

        self.x_train = self._load_data_from_file(train_data)
        self.y_train = self._load_data_from_file(train_targets)
        self.hidden_layers = hidden_layers
        self.aise_params = aise_params

        self.aise = []
        for i,layer in enumerate(self.hidden_layers):
            self.aise.append(AISE(self.x_train, self.y_train, hidden_layer=layer, model=self, **self.aise_params[str(i)]))

    def _load_state_dict(self,file_path):
        checkpoint = torch.load(file_path,map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])

    def _load_data_from_file(self,file_path):
        with open(file_path,"rb") as f:
            temp = pickle.load(f)
            if isinstance(temp,np.ndarray):
                return torch.from_numpy(temp)
            elif isinstance(temp,torch.Tensor):
                return temp.cpu()
            else:
                raise ValueError("Unsupported data type!")
       
    def truncated_forward(self,truncate=None):
        assert truncate is not None,"truncate must be specified"
        if truncate == 0:
            return self.partial_forward_1
        elif truncate == 1:
            return self.partial_forward_2
        elif truncate == 2:
            return self.partial_forward_3
        else:
            return self.partial_forward_4
        
    def partial_forward_1(self,x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        return out_conv1
    
    def partial_forward_2(self,x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training = self.training)
        return out_conv2
    
    def partial_forward_3(self,x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training = self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training = self.training)
        return out_conv3
    
    def partial_forward_4(self,x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training = self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training = self.training)
        out_conv4 = F.dropout(F.relu(self.conv4(out_conv3)), 0.1, training = self.training)
        return out_conv4
    
    def forward(self, x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training = self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training = self.training)
        out_conv4 = F.dropout(F.relu(self.conv4(out_conv3)), 0.1, training = self.training)
        out_pool2 = F.max_pool2d(out_conv4, kernel_size = (2, 2))
        out_view  = out_pool2.view(-1, 128 * 7 * 7)
        out_fc1    = F.dropout(F.relu(self.fc1(out_view)), 0.1, training = self.training)
        out_fc2    = F.dropout(F.relu(self.fc2(out_fc1)), 0.1, training = self.training)
        out       = self.fc3(out_fc2)
        
        return out
    
    def predict(self, x):
        pred_sum = 0.
        for i in range(len(self.hidden_layers)):
            pred_sum = pred_sum + self.aise[i](x)
        return pred_sum/len(self.hidden_layers)