import numpy as np
import scipy as sp
from packages import *

class Sklinucbdynamic:
    def __init__(self, dim, b, lamdba=0.001, nu=1, style='ts'):
        self.dim = dim
        self.b = b  # The dimension of the sketched space
        self.lamdba = lamdba
        self.nu = nu
        self.style = style
        self.contexts = []  # Store original contexts for future sketching
        self.rewards = []  # Store rewards for each context
        self.U = lamdba * np.eye(b)  # Covariance matrix in sketched space
        self.Uinv = 1 / lamdba * np.eye(b)  # Inverse of U
        self.jr = np.zeros((b, ))  # Reward vector in sketched space
        self.mu = np.zeros((b, ))  # Estimated rewards in sketched space
        np.random.seed(42)
        self.sketch = (1/np.sqrt(self.b)) * np.random.randn(self.b, self.dim) 

    def select(self, context):
        # Apply a new sketching matrix for the current time step
          # New random sketching matrix
        sketched_context = context @ self.sketch.T
        
        # Compute uncertainty (variance) for selection
        sig = np.diag(np.matmul(np.matmul(sketched_context, self.Uinv), sketched_context.T))
        
        # Calculate upper confidence bound
        r = np.dot(sketched_context, self.mu) + np.sqrt(self.lamdba * self.nu) * sig
        return np.argmax(r)

    def train(self, context, reward):
        # Store the original context and reward
        self.contexts.append(context)
        self.rewards.append(reward)

        # Initialize temporary variables for U, jr, and Uinv
        U_temp = self.lamdba * np.eye(self.b)
        jr_temp = np.zeros((self.b,))
        self.sketch = (1/np.sqrt(self.b)) * np.random.randn(self.b, self.dim) 
        # Iterate over all past contexts to recalculate U and jr using current sketching matrix
        for t in range(len(self.contexts)):
            # Apply a new sketching matrix at each step (dynamically change the sketching matrix)
             # New sketching matrix at each iteration
            sketched_context = self.contexts[t] @ self.sketch.T  # Project context into the sketched space
            
            # Update jr and U_temp with the sketched context
            jr_temp += self.rewards[t] * sketched_context
            U_temp += np.matmul(sketched_context.reshape((-1, 1)), sketched_context.reshape((1, -1)))
        
        # Perform fast inverse for symmetric matrix (U_temp)
        zz, _ = sp.linalg.lapack.dpotrf(U_temp, False, False)
        Linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(Linv) + np.triu(Linv, k=1).T
        self.U = U_temp
        self.mu = np.dot(self.Uinv, jr_temp)

        

        return 0
