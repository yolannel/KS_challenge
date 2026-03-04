import numpy as np
import torch
from scipy.stats import vonmises
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
from scipy.stats import beta
from aux_functions.ks_data_generator import generate_ks_data

def setup_dataloaders(num_data, nu_range=[1,3], num_train_steps=101, num_test_steps=100):
    dataset=[]
    params = {
        'L': 30,
        'N': 2048,
        'nu': 0.5,
        'dt': 0.5,
        'T': 100.0,
        'num_steps': 201  # Total steps for 0 to 100
    }
    for i in range(num_data):
        params['nu'] = np.random.uniform(nu_range[0], nu_range[1])
        x, t, u = generate_ks_data(params)
        dataset.append(u)
    
    dataset =  SupervisedDataset(torch.tensor(dataset))
    return x, t, dataset


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None):
        super(SupervisedDataset, self).__init__()
        if y is None:
            y = torch.zeros(x.shape[0]).long()
        assert x.shape[0] == y.shape[0]


        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index].unsqueeze(0), self.y[index]

    def to(self, device):
        return SupervisedDataset(
            self.role,
            self.x.to(device),
            self.y.to(device)
        )
