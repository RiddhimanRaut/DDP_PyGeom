from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
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
num_epochs = 10


# Evaluate model
data = data_list[0]

loss = torch.tensor([0.0])
loss_scale = torch.tensor([1.0/rollout_length])
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Initialize lists to store true and predicted outputs
true_outputs = []
predicted_outputs = []

for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Training step
    x_scaled = (data.x - data.data_mean) / (data.data_std + SMALL)  # Scaled input
    x_src, mask = model(x_scaled, data.edge_index, data.pos, data.edge_attr, data.batch)
    x_new = x_scaled + x_src

    target = (data.y[0] - data.data_mean) / (data.data_std + SMALL)  # Scaled target

    loss = loss_scale * loss_fn(x_new, target)  # Compute loss

    loss.backward()  # Backpropagation
    optimizer.step()  # Optimize the model

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  # Print epoch loss

    # Store true and predicted outputs
    true_outputs.append(target.detach().cpu().numpy())
    predicted_outputs.append(x_new.detach().cpu().numpy())

# Convert lists to numpy arrays for easier plotting
true_outputs = np.array(true_outputs)
predicted_outputs = np.array(predicted_outputs)

# Plot the true vs. predicted outputs for the last epoch
plt.figure(figsize=(10, 6))
plt.plot(true_outputs[-1].flatten(), predicted_outputs[-1].flatten(), 'o', color='blue')
plt.xlabel('True Output')
plt.ylabel('Predicted Output')
plt.show()
# # plt.plot(true_outputs[-1].flatten(), label='True Output', color='blue')
# # plt.plot(predicted_outputs[-1].flatten(), label='Predicted Output', color='red', linestyle='--')
# plt.legend()
# plt.xlabel('Data Points')
# plt.ylabel('Value')
# plt.title('True vs. Predicted Outputs')
# plt.show()