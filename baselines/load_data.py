from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd 

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os


class load_mnist_1d:
    def __init__(self):
        # Fetch data
        batch_size = 1
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,
                                      shuffle=False, num_workers=2)
        self.dataiter = iter(train_loader)
        self.n_arm = 10
        self.dim = 7840
 
    def step(self):
        x, y = next(self.dataiter)
        d = x.numpy()[0]
        d = d.reshape(784)
        target = y.item()
        X_n = []
        for i in range(10):
            front = np.zeros((784*i))
            back = np.zeros((784*(9 - i)))
            new_d = np.concatenate((front,  d, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)    
        rwd = np.zeros(self.n_arm)
        #print(target)
        rwd[target] = 1
        return X_n, rwd  
    
    

class load_yelp:
    def __init__(self):
        # Fetch data
        
        self.m = np.load("../data/Yelp/yelp_2000users_10000items_entry.npy")
        self.U = np.load("../data/Yelp/yelp_2000users_10000items_features.npy")
        self.I = np.load("../data/Yelp/yelp_10000items_2000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else:
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), 9, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), replace=False)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]])))
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X), rwd
    
    
    
class load_movielen:
    def __init__(self):
        # Fetch data
        self.m = np.load("../data/MovieLens/movie_2000users_10000items_entry_1000.npy")
        self.U = np.load("../data/MovieLens/movie_2000users_10000items_features_1000.npy")
        self.I = np.load("../data/MovieLens/movie_10000items_2000users_features_1000.npy")
        self.n_arm = 10
        self.dim = 2000
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else:
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), 9, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), replace=False)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]])))
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X), rwd


class load_synthetic:
    def __init__(self,method,d,K,T,context_sparsity,action_sparsity):
        # Fetch data
        self.X = np.load(f"../data/synthetic/synthetic_arms_{method}_d_{d}_k_{K}_cs_{context_sparsity}_as_{action_sparsity}.npy")
        self.Y = np.load(f"../data/synthetic/synthetic_action_{method}_d_{d}_k_{K}_cs_{context_sparsity}_as_{action_sparsity}.npy")

        
        self.n_arm = K
        self.dim = d
        self.pos = 0
 
    def step(self):
        x, y = self.X[self.pos], self.Y[self.pos]
        self.pos = self.pos + 1
        rwd = y + np.random.randn()
        # rwd = y
        return x, rwd 
    
class load_synthetic_dataloader:
    def __init__(self,method,d,K,T,context_sparsity,action_sparsity):
        batch_size = 1
        data_dir = f"../data/synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}"
        dataset = NumpyDataset(data_dir)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=2)
        self.dataiter = iter(train_loader)
        self.n_arm = 4
        self.dim = d
 
    def step(self):
        x, y = next(self.dataiter)
        x = x.squeeze().numpy()
        y = y.squeeze().numpy()
        rwd = y + np.random.randn()
        return x, rwd  
    
class NumpyDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (str): Path to the directory containing .npy files.
            transform (callable, optional): Optional transform to apply to data.
        """
        self.directory = directory
        self.transform = transform
        self.arms = sorted([f for f in os.listdir(directory) if f.startswith("arm_")])
        self.rewards = sorted([f for f in os.listdir(directory) if f.startswith("rewards_")])
        
        print(len(self.arms),len(self.rewards))
        assert len(self.arms) == len(self.rewards), "Mismatch between arms and rewards files"

    def __len__(self):
        return len(self.arms)

    def __getitem__(self, idx):
        arm_path = os.path.join(self.directory, self.arms[idx])
        reward_path = os.path.join(self.directory, self.rewards[idx])
        
        arm_data = np.load(arm_path)
        reward_data = np.load(reward_path)
        
        return arm_data, reward_data



