# %%
d = 200
K = 4
T = 100000
method = 'l2_ball'
action_sparsity = 0.99
context_sparsity = 0.99

# %%
from numpy import random, linalg
import os
import numpy as np

# %%
np.random.seed(42)

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

# %%
X = random_ball(num_points=K*T,dimension=d,radius=1)
X[:,int(-context_sparsity*d):] = 0
X_reshaped = X.reshape(T,K,d)


# %%
a = random_ball(num_points=1,dimension=d,radius=1)
a[:,int(-action_sparsity*d):] = 0

# %%
os.makedirs("synthetic",exist_ok=True)

# %%
Y = X_reshaped@a.T
Y = Y.squeeze()
print(Y.shape)

# %%
np.save(f"synthetic/synthetic_arms_{method}_d_{d}_k_{K}_cs_{context_sparsity}_as_{action_sparsity}",X_reshaped)
np.save(f"synthetic/synthetic_action_{method}_d_{d}_k_{K}_cs_{context_sparsity}_as_{action_sparsity}",Y)


