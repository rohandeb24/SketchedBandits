from packages import *


class TS:
    # Brute-force Linear TS with full inverse
    def __init__(self, dim, lamdba=0.001, nu=1, style='ts'):
        self.dim = dim
        self.U = lamdba * np.eye(dim)
        self.Uinv = 1 / lamdba * np.eye(dim)
        self.nu = nu
        self.jr = np.zeros((dim, ))
        self.mu = np.zeros((dim, ))
        self.lamdba = lamdba
        self.style = style

    def select(self, context):
        # Sample theta from posterior: N(mu, nu^2 * Uinv)
        sampled_theta = np.random.multivariate_normal(self.mu, self.nu**2 * self.Uinv)
        r = np.dot(context, sampled_theta)  # Expected reward for each action
        return np.argmax(r)
        
    
    def train(self, context, reward):
        self.jr += reward * context
        self.U += np.matmul(context.reshape((-1, 1)), context.reshape((1, -1)))
        # fast inverse for symmetric matrix
        zz , _ = sp.linalg.lapack.dpotrf(self.U, False, False)
        Linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(Linv) + np.triu(Linv, k=1).T
        self.mu = np.dot(self.Uinv, self.jr)
        return 0
    

    
    
    