from numpy import random, linalg
import os
import numpy as np
import argparse

np.random.seed(42)
parser = argparse.ArgumentParser(description="Argument parser for sparsity and dimensionality parameters.")


parser.add_argument("--d", type=int, default=2000, help="Dimension of the space (default: 20000)")
parser.add_argument("--K", type=int, default=4, help="Number of clusters or actions (default: 4)")
parser.add_argument("--T", type=int, default=1000, help="Number of iterations or time steps (default: 100000)")


args = parser.parse_args()

d = args.d
K = args.K
T = args.T

action_sparsity = 0.0
context_sparsity = 0.0
method = 'l1_ball'

os.makedirs(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}",exist_ok=True)

# %%
# https://chatgpt.com/share/6794bb9f-fe10-8004-ba39-16a92e9c3d46
def sample_from_l1_ball(num_points,dimension,radius=1):
    """
    Sample uniformly from the L1 ball of radius 1 in n dimensions.

    Parameters:
        n (int): Dimension of the space.
        num_samples (int): Number of samples to generate.

    Returns:
        numpy.ndarray: Array of shape (num_samples, n) containing the samples.
    """
    # Step 1: Generate directions on the L1 sphere
    exp_samples = np.random.exponential(scale=1.0, size=(num_points, dimension))
    directions = exp_samples / np.sum(exp_samples, axis=1, keepdims=True)  # Normalize to form a simplex
    signs = np.random.choice([-1, 1], size=(num_points, dimension))  # Randomly assign signs
    directions *= signs

    # Step 2: Sample radii
    u = np.random.uniform(0, 1, size=num_points)
    radii = u ** (1 / dimension)

    # Step 3: Combine radii and directions
    samples = directions * radii[:, np.newaxis]

    return samples



a = sample_from_l1_ball(num_points=1,dimension=d,radius=1)



for iters in range(T//1000):
    X = sample_from_l1_ball(num_points=K*1000,dimension=d,radius=1)

    X_reshaped = X.reshape(1000,K,d)
    Y = d*10*X_reshaped@a.T
    
    for i in range(X_reshaped.shape[0]):
        np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/arm_{1000*iters + i}.npy",X_reshaped[i])
        np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/rewards_{1000*iters + i}.npy",Y[i])
        
    
np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/true_parameter.npy",a)
