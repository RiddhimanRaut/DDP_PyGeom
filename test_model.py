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
