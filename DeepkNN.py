#!/usr/bin/env python
# coding: utf-8

# In[ ]:




from collections import Counter

import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors        import NearestNeighbors

import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim


class Calibration():
    def __init__(self, x_cali, y_cali):
        self.x = x_cali
        self.y = y_cali
        self.n_sample = len(y_cali)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_sample  



class CKNN():
    def __init__(self, model, device, x_train, y_train, 
                 batch_size, n_neighbors, n_embs):
        self.model         = model
        self.model.to(device)
        self.model.eval()
        self.device        = device
        self.input_shape = x_train.shape[1:]

        self.batch_size    = batch_size
        self.n_neighbors   = n_neighbors
        self.n_embs        = n_embs
        
        self.rng = np.random.RandomState(1)

        x_train, y_train, self.calib_dataset = self._sep_data(x_train, y_train)
        
        self.conv_features = self._buid_rep_train(x_train)
        self.train_targets = y_train
        self.neighs          = self._build_neighs()
        self.alpha_calib_sum = self._build_calibration()

    def _sep_data(self, xx, yy):
        calieps = 720
        cali_indice = np.array([i for i in range(0,xx.size(0),math.floor(xx.size(0)/calieps))])
        cali_images = np.zeros((calieps,)+self.input_shape)
        cali_labels = self.rng.randint(0,10, calieps)
        for i in range(calieps):
            cali_images[i] = xx[cali_indice[i]]
            cali_labels[i] = yy[i]
        x_train = np.concatenate((xx, cali_images), axis=0)
        y_train = np.concatenate((yy, cali_labels), axis=0)
        x_train = np.delete(x_train, cali_indice, axis=0)
        y_train = np.delete(y_train, cali_indice, axis=0)
        calib_dataset = Calibration(cali_images.astype(np.float32), cali_labels)
        return x_train, y_train, calib_dataset

    def _buid_rep_train(self, xx_batch_train, batchs=2000):
        xhs = []
        for i in range(0, len(xx_batch_train), batchs):
            xx = xx_batch_train[i:i + batchs]
            conv_train = self._feature_space(self.model, 4, torch.Tensor(xx), self.device)
            if i == 0:
                xhs = conv_train
            else:
                xhs[0] = np.concatenate((xhs[0],conv_train[0]),axis=0)
                xhs[1] = np.concatenate((xhs[1],conv_train[1]),axis=0)
                xhs[2] = np.concatenate((xhs[2],conv_train[2]),axis=0)
                xhs[3] = np.concatenate((xhs[3],conv_train[3]),axis=0)
        return xhs

    def _feature_space(self, cnnmod, num_rep, data, device):
        print('Building the feature spaces from the selected set.')

        conv_features = [[] for _ in range(num_rep)]
        targets       = []
        predictions   = []
        print('\tRunning predictions')
        cnnmod.eval()
        data = data.to(device)
        *out_convs, _ = cnnmod(data)
        for i, out_conv in enumerate(out_convs):
            conv_feat = out_conv.contiguous().reshape(out_conv.size(0), -1).cpu().detach().numpy()
            conv_features[i].append(conv_feat)
        conv_features = [np.concatenate(out_convs) for out_convs in conv_features]

        return conv_features

    def _build_calibration(self):
        print('Building calibration set.')
        sequential_calib_loader = torch.utils.data.DataLoader(
            self.calib_dataset,
            shuffle    = False,
            batch_size = self.batch_size
        )
        alpha_by_batch = [self._alpha(X, y) for X, y in sequential_calib_loader]
        alpha_values   =  np.concatenate(alpha_by_batch)
        c              = Counter(alpha_values)
        alpha_sum_cum  = []

        for alpha_value in range(self.n_embs * self.n_neighbors, -1, -1):
            alpha_sum_cum.append(c[alpha_value] + (alpha_sum_cum[-1] if len(alpha_sum_cum) > 0 else 0))
            
        return np.array(alpha_sum_cum[::-1])
    
    def _build_neighs(self):
        print('Building Nearest Neighbor finders.')
        return [
            NearestNeighbors(
                n_neighbors = self.n_neighbors, 
                metric      = 'cosine'
            ).fit(feats) 
            for feats in self.conv_features
        ]
        
    def _alpha(self, X, y):
        neighbors_by_layer     = self._get_closest_points(X)
        closest_points_classes = self.train_targets[neighbors_by_layer]
        same_class_neighbors   = torch.Tensor(closest_points_classes) != y.reshape(y.shape[0], 1, 1)
        print((closest_points_classes.dtype,closest_points_classes.shape))
        print(y.dtype,y.shape)
        same_class_neighbors   = same_class_neighbors.reshape(-1, self.n_neighbors * self.n_embs)
        alpha                  = same_class_neighbors.sum(axis = 1)
        
        return alpha

    def _compute_nonconformity(self, X):
        neighbors_by_layer  = self._get_closest_points(X)
        closest_points_label   = self.train_targets[neighbors_by_layer]
        closest_points_label   = closest_points_label.reshape(-1, self.n_embs*self.n_neighbors)
        nonconformity          = [(closest_points_label != label).sum(axis = 1) for label in range(10)]
        nonconformity          = np.stack(nonconformity, axis = 1)
        
        return nonconformity
    
    def _compute_p_value(self, X):
        nonconformity = self._compute_nonconformity(X)
        empirical_p_value      = self.alpha_calib_sum[nonconformity] / len(self.calib_dataset)
        
        return empirical_p_value
    
    def _get_closest_points(self, X):
        *out_convs,_ = self.model(X.to(self.device))
        neighbors_by_layer = []
        out_convs = out_convs#out_convs#[out_convs[0],]#out_convs#[out_convs[3],]
        for i, (neigh, layer_emb) in enumerate(zip(self.neighs, out_convs)):
            emb       = layer_emb.detach().cpu().reshape(X.size(0), -1).numpy()
            neighbors = neigh.kneighbors(emb, return_distance = False) 
            neighbors_by_layer.append(neighbors)
        return torch.tensor(np.stack(neighbors_by_layer, axis = 1))#, y_pred

    def predict(self, X):
        p_value     = self._compute_p_value(X)
        y_pred      = p_value.argmax(axis = 1)
        # Partitioning according to the second to last value in order to compute
        # credibility and confidence
        partition   = np.partition(p_value, -2)
        credibility = partition[:, -1]
        confidence  = 1 - partition[:, -2]
        
        return p_value, confidence, credibility
    
    def predict_interpretation(self, X, y = None):
        img_data = lambda img: reverse_normalize(img.to(self.device).squeeze().permute(1,2,0)).data.cpu().numpy().clip(0, 1)
        plt.figure(figsize = (12, 6))
        gs       = gridspec.GridSpec(4, 4 + self.n_neighbors)
        ax_big   = plt.subplot(gs[:, :4])
        ax_grid  = [[plt.subplot(gs[i, j]) for j in range(4, 4 + self.n_neighbors)] for i in range(4)]
        ax_big.imshow(img_data(X))
        ax_big.axis('off')
        neighbors_by_layer, y_pred = self._get_closest_points(X.unsqueeze(0))
        print(neighbors_by_layer)
        print(y_pred)
        for ax_line, closest_layer in zip(ax_grid, neighbors_by_layer[0]):
            for ax_cell, train_ex_id in zip(ax_line, closest_layer):
                img = self.train_dataset[train_ex_id][0]
                ax_cell.imshow(img_data(img))
                ax_cell.axis('off')

        if y is not None:
            print(f'Real class: {cifar_cats[y]}')
        print(f'Predicted class: {cifar_cats[y_pred.argmax(dim = 1).cpu().item()]}')
        
        return y_pred

