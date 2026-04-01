from packages import *


class SkTS:
    # Brute-force Linear TS with full inverse
    def __init__(self, dim, b, lamdba=0.001, nu=1, style='ts'):
        self.dim = dim
        self.U = lamdba * np.eye(b)
        self.Uinv = 1 / lamdba * np.eye(b)
        self.nu = nu
        self.jr = np.zeros((b, ))
        self.mu = np.zeros((b, ))
        self.lamdba = lamdba
        self.style = style
        np.random.seed(42)
        self.sketch = (1/np.sqrt(b))*np.random.randn(b,dim)
        

    def select(self, context):
        context =  (context@self.sketch.T)
        sig = np.diag(np.matmul(np.matmul(context, self.Uinv), context.T))
        
        sampled_theta = np.random.multivariate_normal(self.mu, self.nu**2 * self.Uinv)
        r = np.dot(context, sampled_theta)  # Expected reward for each action
        return np.argmax(r)
        
    
    def train(self, context, reward):
        #sketch the context vector
        context = self.sketch @ context
        self.jr += reward * context
        self.U += np.matmul(context.reshape((-1, 1)), context.reshape((1, -1)))
        # fast inverse for symmetric matrix
        zz , _ = sp.linalg.lapack.dpotrf(self.U, False, False)
        Linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(Linv) + np.triu(Linv, k=1).T
        self.mu = np.dot(self.Uinv, self.jr)

        return 0
    

    
    
    