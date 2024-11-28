import numpy as np 
from utils import Cov_exact

class rBergomi_direct:
    def __init__(self, M, T, params, P):
        #Time discretization
        self.M = M #number of time steps 
        self.T = T
        self.tau = self.T/self.M
        self.grid = np.linspace(0, T, self.M+1)        
        self.P = P #number of paths to generate 
        
        #Rough Bergomi model parameters 
        self.X0 = params["X0"]
        self.V0 = params["V0"]
        self.xi = params["xi"]
        self.nu = params["nu"]
        self.rho = params["rho"]
        self.H = params["H"]
        
        #Precomputation
        self.minue = self.nu**2/2 * self.grid[1:] **(2*self.H) #(M,) 1-d array        
    
    
    def W_Z(self):
        cov = Cov_exact(self.M, self.H, self.rho)
        WZ = np.random.multivariate_normal(np.zeros(2*self.M), cov, self.P)
        W = WZ[:, :self.M]
        Z = WZ[:, self.M:]
        dZ = Z - np.c_[np.zeros(self.P), Z][:, :-1]
        return dZ, W #samples of BM increment, Volterra process respectively 
    
    
    
    def S_(self): #by Forward Euler method         
        X = np.zeros((self.P, self.M))
        dZ, W = self.W_Z()
        V = self.xi * np.exp(self.nu * W - self.minue)
       
        X[:,0] = self.X0 - self.V0 * self.tau/2 + np.sqrt(self.V0)*dZ[:,0]
        
        for j in range(1, self.M):
            X[:,j] = X[:,j-1] - V[:,j-1] * self.tau/2 + np.sqrt(V[:,j-1])*dZ[:,j]
            
        S = np.exp(X)
        return S
