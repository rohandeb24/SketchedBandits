from numpy import random, linalg
import os
import numpy as np
import argparse

np.random.seed(42)
parser = argparse.ArgumentParser(description="Argument parser for sparsity and dimensionality parameters.")


parser.add_argument("--d", type=int, default=2000, help="Dimension of the space (default: 20000)")
parser.add_argument("--K", type=int, default=4, help="Number of clusters or actions (default: 4)")
parser.add_argument("--T", type=int, default=1000, help="Number of iterations or time steps (default: 100000)")
parser.add_argument("--action_sparsity", type=float, default=0.5, help="Sparsity level for actions (default: 0.5)")
parser.add_argument("--context_sparsity", type=float, default=0.5, help="Sparsity level for context (default: 0.5)")

args = parser.parse_args()

d = args.d
K = args.K
T = args.T

action_sparsity = args.action_sparsity
context_sparsity = args.context_sparsity
method = 'l2_ball'

os.makedirs(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}",exist_ok=True)

# %%
# https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def random_ball(num_points, dimension, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


a = random_ball(num_points=1,dimension=d,radius=1)
if int(-action_sparsity*d) > 0 :
    a[:,int(-action_sparsity*d):] = 0


for iters in range(T//1000):
    X = random_ball(num_points=K*1000,dimension=d,radius=1)
    if int(-action_sparsity*d) > 0:
        X[:,int(-context_sparsity*d):] = 0
    X_reshaped = X.reshape(1000,K,d)
    Y = X_reshaped@a.T
    
    for i in range(X_reshaped.shape[0]):
        np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/arm_{1000*iters + i}.npy",X_reshaped[i])
        np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/rewards_{1000*iters + i}.npy",Y[i])
        

np.save(f"synthetic/{method}_d_{d}_k_{K}_t_{T}_cs_{context_sparsity}_as_{action_sparsity}/true_parameter.npy",a)
