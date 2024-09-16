#%%
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributions as tdist 

import models.gnn as gnn 
import dataprep.speedy as spd

#%%
path_to_data = "D:\MyGithub\DDP_PyGeom\datasets\speedy_numpy_file_train.npz"
device_for_loading = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rollout_length = 2

#%%
print('Loading dataset...')
data_list, _ = spd.get_pygeom_dataset(
        path_to_data = path_to_data,
        device_for_loading = device_for_loading,
        time_lag = rollout_length, 
        fraction_valid = 0.0)
print('Done loading dataset.')

#%%
# check data shape
sample = data_list[0]

print(sample.x.shape)
print(len(sample.y))
print(sample.y[0].shape)



# # Print field names
# print(sample.field_names)
# print(sample.field_shape)

# #%%
# f_plt = sample.x[:,0].reshape(sample.field_shape)
# # %%
# mu = 0.0
# std = 1e-2
# noise_dist = tdist.Normal(torch.tensor([mu]), torch.tensor([std]))
# # %%
# SMALL = 1e-10
# data = data_list[0]
# x_scaled = (data.x - data.data_mean)/(data.data_std + SMALL)
# noise = noise_dist.sample((x_scaled.shape[0],))
# x_old = torch.clone(x_scaled) + noise
# # %%
# # rollout_length = 2
# # x = x_old
# print(x_old.shape)
# print(data.edge_attr.shape)
# mean_edge_length = torch.mean(data.edge_attr[:,2])
# print(mean_edge_length)

# # %%
# model = gnn.TopkMultiscaleGNN( 
#             input_node_channels = x_old.shape[1],
#             input_edge_channels = data.edge_attr.shape[1],
#             hidden_channels = 128,
#             output_node_channels = x_old.shape[1],
#             n_mlp_hidden_layers = 2,
#             n_mmp_layers = 2,
#             n_messagePassing_layers = 4,
#             max_level_mmp = 0,
#             l_char = mean_edge_length,
#             max_level_topk = 0,
#             rf_topk = 0,
#             name='gnn')

# # %%
# model.train()
# x_src, mask = model(x_scaled, 
#                     data.edge_index, 
#                     data.pos, 
#                     data.edge_attr, 
#                     data.batch)

# x_new = x_old + x_src
# # %%
# target = (data.y[0] - data.data_mean) / (data.data_std + SMALL)  # Scaled target

# # %%
# loss_fn = torch.nn.MSELoss()
# loss = torch.tensor([0.0])
# loss_scale = torch.tensor([1.0/rollout_length])

# loss = loss_scale * loss_fn(x_new, target)  # Compute loss
# print(loss)
# # %%

# # print y
# print(data.y[0])