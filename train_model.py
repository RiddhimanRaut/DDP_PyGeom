from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributions as tdist
plt.rcParams.update({'font.size': 22})

import models.gnn as gnn 
import dataprep.speedy as spd

torch.manual_seed(122)
np.random.seed(122)
SMALL = 1e-10

path_to_data = "D:\MyGithub\DDP_PyGeom\datasets\speedy_numpy_file_train.npz"
device_for_loading = "cpu"
rollout_length = 1
print('Loading dataset...')
data_list, _ = spd.get_pygeom_dataset(
        path_to_data = path_to_data,
        device_for_loading = device_for_loading,
        time_lag = rollout_length, 
        fraction_valid = 0.0)
print('Done loading dataset.')

#%%
mu = 0.0
std = 1e-2
noise_dist = tdist.Normal(torch.tensor([mu]), torch.tensor([std]))

#%%
def train_sample(sample, model, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_scaled = (sample.x - sample.data_mean)/(sample.data_std + SMALL)
        noise = noise_dist.sample((x_scaled.shape[0],))
        x_old = torch.clone(x_scaled) + noise
        x_src, mask = model(x_old, sample.edge_index, sample.pos, sample.edge_attr, sample.batch)
        x_new = x_old + x_src
        target = (sample.y[0] - sample.data_mean)/(sample.data_std + SMALL)
        loss = loss_fn(x_new, target)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return loss

l_char = data_list[0].edge_attr[:,2].mean()
model = gnn.TopkMultiscaleGNN( 
            input_node_channels = data_list[0].x.shape[1],
            input_edge_channels = data_list[0].edge_attr.shape[1],
            hidden_channels = 128,
            output_node_channels = data_list[0].x.shape[1],
            n_mlp_hidden_layers = 2,
            n_mmp_layers = 2,
            n_messagePassing_layers = 2,
            max_level_mmp = 0,
            l_char = l_char,
            max_level_topk = 0,
            rf_topk = 16,
            name='gnn')
model.to('cpu')

# for i in range(len(data_list)):
for i in range(10):
    print(f'Training sample {i+1}/{len(data_list)}')
    sample = data_list[i]
    train_sample(sample, model, torch.optim.Adam(model.parameters(), lr=1e-3), torch.nn.MSELoss(), 10)

# %%
# Save model
torch.save(model.state_dict(), 'D:\MyGithub\DDP_PyGeom\models\gnn.pth')

# # %%
# # load test data
# path_to_data = "D:\MyGithub\DDP_PyGeom\datasets\speedy_numpy_file_test.npz"
# test_data_list, _ = spd.get_pygeom_dataset(
#         path_to_data = path_to_data,
#         device_for_loading = device_for_loading,
#         time_lag = rollout_length, 
#         fraction_valid = 0.0)

# #%%
# def test_sample(sample, model):
#     x_scaled = (sample.x - sample.data_mean)/(sample.data_std + SMALL)
#     noise = noise_dist.sample((x_scaled.shape[0],))
#     x_old = torch.clone(x_scaled) + noise
#     x_src, mask = model(x_old, sample.edge_index, sample.pos, sample.edge_attr, sample.batch)
#     x_new = x_old + x_src
#     target = (sample.y[0] - sample.data_mean)/(sample.data_std + SMALL)
#     return x_new, target

# #%%
# # Evaluate model

# model.eval()
# loss = torch.tensor([0.0])
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss = []
# for i in range(test_data_list):
#     sample = data_list[i]
#     x_new, target = test_sample(sample, model)
#     loss.append(loss_fn(x_new, target))

# print(f'Loss: {loss.item():.4f}')
