from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributions as tdist
plt.rcParams.update({'font.size': 22})

import models.gnn as gnn 
import dataprep.speedy as spd

import dataprep.speedy as spd
import torch
import numpy as np

torch.manual_seed(122)
np.random.seed(122)
SMALL = 1e-10

#%%
# load test data
path_to_data = "D:\MyGithub\DDP_PyGeom\datasets\speedy_numpy_file_test.npz"
device_for_loading = "cpu"
rollout_length = 100
print('Loading dataset...')
test_data_list, _ = spd.get_pygeom_dataset(
        path_to_data = path_to_data,
        device_for_loading = device_for_loading,
        time_lag = rollout_length, 
        fraction_valid = 0.0)
#Initialize model
l_char = test_data_list[0].edge_attr[:,2].mean()

#%%
mu = 0.0
std = 1e-2
noise_dist = tdist.Normal(torch.tensor([mu]), torch.tensor([std]))

#%%
model = gnn.TopkMultiscaleGNN( 
            input_node_channels = test_data_list[0].x.shape[1],
            input_edge_channels = test_data_list[0].edge_attr.shape[1],
            hidden_channels = 128,
            output_node_channels = test_data_list[0].x.shape[1],
            n_mlp_hidden_layers = 2,
            n_mmp_layers = 1,
            n_messagePassing_layers = 2,
            max_level_mmp = 2,
            l_char = l_char,
            max_level_topk = 1,
            rf_topk = 16,
            name='gnn')
model.to('cpu')

#%%
# Load the state dictionary
model_path = r"D:\MyGitHub\DDP_PyGeom\models\gnn.pth"
state_dict = torch.load(model_path, map_location=device_for_loading)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

#%%
# test model
def test_sample(sample, model, noise_dist, rollout_length, loss_fn):
    loss_scale = torch.tensor([1.0/rollout_length])
    loss = torch.tensor([0.0])
    x_new = (sample.x - sample.data_mean)/(sample.data_std + SMALL)
    for t in range(rollout_length):        
        noise = noise_dist.sample((x_new.shape[0],))
        x_old = torch.clone(x_new) + noise
        x_src, mask = model(x_old, sample.edge_index, sample.pos, sample.edge_attr, sample.batch)
        x_new = x_old + x_src
        target = (sample.y[t] - sample.data_mean)/(sample.data_std + SMALL)
        loss += loss_scale * loss_fn(x_new, target)
        print(f'Loss: {loss.item():.4f}')
    return x_new, target, mask

sample = test_data_list[5]
with torch.no_grad():
    x_new, target, mask = test_sample(sample = sample,
                                model = model,
                                noise_dist = noise_dist,
                                rollout_length = rollout_length,
                                loss_fn = torch.nn.MSELoss())

#%% 
# plot results
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
f_plt = x_new[:,0].detach().numpy().reshape(sample.field_shape)
ax[0].imshow(f_plt)
ax[0].set_title('Model prediction')
ax[1].imshow(target[:,0].reshape(sample.field_shape))
ax[1].set_title('Target')
# plot mask
mask = mask.detach().numpy().reshape(sample.field_shape)
ax[2].imshow(mask)
ax[2].set_title('Mask')

# plt.savefig(r'D:\MyGitHub\DDP_PyGeom\results\prediction_without_TopK.png', 
#             dpi = 600, bbox_inches = 'tight')
plt.savefig(r'D:\MyGitHub\DDP_PyGeom\results\prediction.png', 
            dpi = 600, bbox_inches = 'tight')
plt.show()