from numpy import random, linalg
import os
import numpy as np
import argparse

np.random.seed(54)
parser = argparse.ArgumentParser(description="Argument parser for sparsity and dimensionality parameters.")


parser.add_argument("--d", type=int, default=200, help="Dimension of the space (default: 20000)")
parser.add_argument("--K", type=int, default=4, help="Number of clusters or actions (default: 4)")
parser.add_argument("--T", type=int, default=1000, help="Number of iterations or time steps (default: 100000)")
parser.add_argument("--action_sparsity", type=float, default=0.5, help="Sparsity level for actions (default: 0.5)")
parser.add_argument("--context_sparsity", type=float, default=0.5, help="Sparsity level for context (default: 0.5)")

method = 'latent_space'
args = parser.parse_args()
context_sparsity = args.context_sparsity
action_sparsity = args.action_sparsity
d = args.d
K = args.K
T = args.T

context_latent_dim = int((1-context_sparsity)*d)
action_latent_dim = int((1-action_sparsity)*d)


os.makedirs(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}",exist_ok=True)

W = np.random.randn(d, context_latent_dim) #latent context space
A = np.random.randn(d, action_latent_dim) # latent action space

a = np.random.randn(1,action_latent_dim)
a = a@A.T

for iters in range(T//1000):
    X = np.random.randn(1000*K, context_latent_dim)
    
    X_high_dim = X @ W.T
    X_reshaped = X_high_dim.reshape(1000,K,d)
    Y = X_reshaped@a.T
    
    for i in range(X_reshaped.shape[0]):
        np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/arm_{1000*iters + i}.npy",X_reshaped[i])
        np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/rewards_{1000*iters + i}.npy",Y[i])
        

np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/true_parameter.npy",a)


