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

# load test data
path_to_data = "D:\MyGithub\DDP_PyGeom\datasets\speedy_numpy_file_test.npz"
device_for_loading = "cpu"
rollout_length = 1
print('Loading dataset...')
test_data_list, _ = spd.get_pygeom_dataset(
        path_to_data = path_to_data,
        device_for_loading = device_for_loading,
        time_lag = rollout_length, 
        fraction_valid = 0.0)
#Initialize model
l_char = test_data_list[0].edge_attr[:,2].mean()

model = gnn.TopkMultiscaleGNN( 
            input_node_channels = test_data_list[0].x.shape[1],
            input_edge_channels = test_data_list[0].edge_attr.shape[1],
            hidden_channels = 128,
            output_node_channels = test_data_list[0].x.shape[1],
            n_mlp_hidden_layers = 2,
            n_mmp_layers = 2,
            n_messagePassing_layers = 2,
            max_level_mmp = 0,
            l_char = l_char,
            max_level_topk = 0,
            rf_topk = 16,
            name='gnn')
model.to('cpu')

# Load the state dictionary
model_path = r"D:\MyGitHub\DDP_PyGeom\models\gnn.pth"
state_dict = torch.load(model_path, map_location=device_for_loading)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# test model
def test_sample(sample, model):
    x_scaled = (sample.x - sample.data_mean)/(sample.data_std + SMALL)
    x_old = torch.clone(x_scaled)
    x_src, mask = model(x_old, sample.edge_index, sample.pos, sample.edge_attr, sample.batch)
    x_new = x_old + x_src
    target = (sample.y[0] - sample.data_mean)/(sample.data_std + SMALL)
    return x_new, target

sample = test_data_list[0]
x_new, target = test_sample(sample, model)

#%% 
# plot results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
f_plt = x_new[:,0].detach().numpy().reshape(sample.field_shape)
ax[0].imshow(f_plt)
ax[0].set_title('Model prediction')
ax[1].imshow(target[:,0].reshape(sample.field_shape))
ax[1].set_title('Target')
plt.savefig(r'D:\MyGitHub\DDP_PyGeom\results\prediction_without_TopK.png', 
            dpi = 600, bbox_inches = 'tight')
plt.show()


