from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import numpy as np
import torch
import torch.distributions as tdist
import matplotlib.pyplot as plt
import models.gnn as gnn 
import dataprep.speedy as spd

# Set font size for plots
plt.rcParams.update({'font.size': 22})

torch.manual_seed(122)
np.random.seed(122)
SMALL = 1e-10

# Base path to save models
model_save_path = "D:\\MyGithub\\DDP_PyGeom\\models\\"
data_path = "D:\\MyGithub\\DDP_PyGeom\\datasets\\"

# Prepare the dataset
device_for_loading = "cpu"
path_to_data = data_path + "speedy_numpy_file_train.npz"
print('Loading dataset...')
data_list, _ = spd.get_pygeom_dataset(path_to_data=path_to_data, device_for_loading=device_for_loading, time_lag=1, fraction_valid=0.0)
print('Done loading dataset.')

# Model and training configurations
rollout_steps_values = [2]
std_values = [1e-2]
max_level_values = [2]
epoch_values = [100]

def train_and_save_models():
    for rollout_length in rollout_steps_values:
        for std in std_values:
            for max_level_mmp in max_level_values:
                for max_level_topk in max_level_values:
                    for epochs in epoch_values:
                        # Define noise distribution
                        noise_dist = tdist.Normal(torch.tensor([0.0]), torch.tensor([std]))
                        
                        # Initialize model
                        l_char = data_list[0].edge_attr[:, 2].mean()
                        model = gnn.TopkMultiscaleGNN(
                            input_node_channels=data_list[0].x.shape[1],
                            input_edge_channels=data_list[0].edge_attr.shape[1],
                            hidden_channels=128,
                            output_node_channels=data_list[0].x.shape[1],
                            n_mlp_hidden_layers=2,
                            n_mmp_layers=2,
                            n_messagePassing_layers=2,
                            max_level_mmp=max_level_mmp,
                            l_char=l_char,
                            max_level_topk=max_level_topk,
                            rf_topk=16,
                            name='gnn'
                        )
                        model.to('cpu')

                        # Train model
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                        loss_fn = torch.nn.MSELoss()
                        for sample in data_list[:5]:  # Using first 10 samples for training
                            train_sample(sample, model, optimizer, loss_fn, epochs, noise_dist)

                        # Save model
                        model_filename = f'gnn_rollout{rollout_length}_std{std}_maxmmp{max_level_mmp}_maxtopk{max_level_topk}_epochs{epochs}.pth'
                        torch.save(model.state_dict(), model_save_path + model_filename)
                        print(f'Model saved: {model_filename}')

def train_sample(sample, model, optimizer, loss_fn, epochs, noise_dist):
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_scaled = (sample.x - sample.data_mean) / (sample.data_std + SMALL)
        noise = noise_dist.sample((x_scaled.shape[0],))
        x_old = torch.clone(x_scaled) + noise
        x_src, mask = model(x_old, sample.edge_index, sample.pos, sample.edge_attr, sample.batch)
        x_new = x_old + x_src
        target = (sample.y[0] - sample.data_mean) / (sample.data_std + SMALL)
        loss = loss_fn(x_new, target)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return loss

train_and_save_models()
