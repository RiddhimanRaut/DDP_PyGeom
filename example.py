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

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a random model, and evaluate on a snapshot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# Plot data: 
sample = data_list[0]
field_id = 3 
f_plt = sample.x[:,field_id].reshape(sample.field_shape) 
fig, ax = plt.subplots()
ax.imshow(f_plt)
plt.show(block=False)

# Create model 
device = 'cpu' 

# ~~~~ Baseline model settings (no top-k pooling) 
# 1 MMP level (no graph coarsening), and 0 top-k levels 
input_node_channels = sample.x.shape[1]
input_edge_channels = sample.edge_attr.shape[1]
hidden_channels = 64 # embedded node/edge dimensionality   
output_node_channels = input_node_channels
n_mlp_hidden_layers = 2 # size of MLPs
n_mmp_layers = 2 # number of MMP layers per topk level.  
n_messagePassing_layers = 4 # number of message passing layers in each processor block 
max_level_mmp = 0 # maximum number of MMP levels. "1" means only single-scale operations are used. 
max_level_topk = 0 # maximum number of topk levels.
rf_topk = 16 # topk reduction factor 
""" 
if n_mmp_layers=2 and max_level_topk=0, we have this:
    (2 x Down MMP) 0 ---------------> output
"""


# # ~~~~ Interpretable model settings (with top-k pooling)
# # 1 MMP level (no graph coarsening), and 1 top-k level 
# input_node_channels = sample.x.shape[1]
# input_edge_channels = sample.edge_attr.shape[1]
# hidden_channels = 64 # embedded node/edge dimensionality   
# output_node_channels = input_node_channels
# n_mlp_hidden_layers = 2 # size of MLPs
# n_mmp_layers = 1 # number of MMP layers per topk level.  
# n_messagePassing_layers = 4 # number of message passing layers in each processor block 
# max_level_mmp = 0 # maximum number of MMP levels. "1" means only single-scale operations are used. 
# max_level_topk = 1 # maximum number of topk levels.
# rf_topk = 16 # topk reduction factor 
# """ 
# if n_mmp_layers=1 and max_level_topk=1, we have this:
#     (1 x Down MMP) 0 ---------------> (1 x Up MMP) 0 ---> output
#             |                               | 
#             |                               |
#             |----> (1 x Down MMP) 1  ------>|
# """


# get l_char -- characteristic edge lengthscale used for graph coarsening (not used when max_level_mmp=1) 
edge_attr = sample.edge_attr
mean_edge_length = edge_attr[:,2].mean()
l_char = mean_edge_length

model = gnn.TopkMultiscaleGNN( 
            input_node_channels,
            input_edge_channels,
            hidden_channels,
            output_node_channels,
            n_mlp_hidden_layers,
            n_mmp_layers,
            n_messagePassing_layers,
            max_level_mmp,
            l_char,
            max_level_topk,
            rf_topk,
            name='gnn')

model.to(device)
num_epochs = 500

# %%
mu = 0.0
std = 1
noise_dist = tdist.Normal(torch.tensor([mu]), torch.tensor([std]))
noise = noise_dist.sample((sample.x.shape[0],))

#%%
# Evaluate model
data = data_list[0]

loss = torch.tensor([0.0])
loss_scale = torch.tensor([1.0/rollout_length])
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Training step
    x_scaled = (data.x - data.data_mean) / (data.data_std + SMALL)  # Scaled input
    x_old = torch.clone(x_scaled) + noise
    x_src, mask = model(x_old, data.edge_index, data.pos, data.edge_attr, data.batch)
    x_new = x_old + x_src

    target = (data.y[0] - data.data_mean) / (data.data_std + SMALL)  # Scaled target

    loss = loss_scale * loss_fn(x_new, target)  # Compute loss

    loss.backward()  # Backpropagation
    optimizer.step()  # Optimize the model

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  # Print epoch loss


# %%
# load test data
path_to_data = "D:\MyGithub\DDP_PyGeom\datasets\speedy_numpy_file_test.npz"
data_list, _ = spd.get_pygeom_dataset(
        path_to_data = path_to_data,
        device_for_loading = device_for_loading,
        time_lag = rollout_length, 
        fraction_valid = 0.0)

# Evaluate model on test data
data = data_list[0]
x_scaled = (data.x - data.data_mean) / (data.data_std + SMALL)  # Scaled input
x_old = torch.clone(x_scaled) + noise
x_src, mask = model(x_old, data.edge_index, data.pos, data.edge_attr, data.batch)
x_new = x_old + x_src
target = (data.y[0] - data.data_mean) / (data.data_std + SMALL)  # Scaled target
loss = loss_scale * loss_fn(x_new, target)  # Compute loss

print(f'Test Loss: {loss.item():.4f}')

#%%
# Plot results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(x_old[:, 0].reshape(sample.field_shape))
ax[0].set_title('Input')
ax[1].imshow(x_new[:, 0].detach().numpy().reshape(sample.field_shape))
ax[1].set_title('Output')
ax[2].imshow(target[:, 0].reshape(sample.field_shape))
ax[2].set_title('Target')
plt.show()
# %%
