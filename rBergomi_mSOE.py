import scipy as sp
import numpy as np
from utils import Cov_SOE
import multiprocessing as mp
from joblib import Parallel, delayed

class rBergomi_mSOE:
    def __init__(self, M, T, params, P, Lambda, Omega, cores, loop, rand_seed):
        #Time discretization
        self.M = M # number of time intervals 
        self.T = T # expiration        
        self.tau = self.T/self.M
        self.grid = np.linspace(0, T, self.M+1)        
        self.P = P #number of paths to generate 
        self.Lambda = Lambda
        self.Omega = Omega
        self.Nexp = self.Lambda.size
        
        #Rough Bergomi model parameters 
        self.X0 = params["X0"]
        self.V0 = params["V0"]
        self.xi = params["xi"]
        self.nu = params["nu"]
        self.rho = params["rho"]
        self.H = params["H"]
        
              
        #Precomputation 
        # size = (1, Nexp)
        self.coef = np.exp(-self.tau * Lambda.reshape(1, -1))
        self.minue = self.nu**2/2 * (self.grid[1:])**(2* self.H)
        
        # compute covariance matrix 
        self.cov = Cov_SOE(self.Nexp, self.Lambda, self.tau, self.H)
        
        # enerate the stock price paths in parallel
        self.num_cores = cores
        self.my_loops = loop
        self.seed = rand_seed 



    # generate volatility paths without the forward variance
    # generate the paths of Brownian motion that drives the stock price 
    def generate_V_chunk(self, chunk_size):
        
        W = np.zeros((chunk_size, self.M))
        mul = np.zeros((chunk_size, self.M))
        hist = np.zeros((chunk_size, self.Nexp)) 
        sample = np.random.multivariate_normal(np.zeros(self.Nexp +2), self.cov, chunk_size)
        W[:, 0] = sample[:, 0]
        mul[:, 0] = sample[:, -1]
        
        
        for i in range(2, self.M + 1):    
           
            # size = (chunk_size, Nexp)
            hist = (hist + sample[:, 1:-1]) * self.coef
            # size = (chunk_size, )
            hist_part = np.sqrt(2*self.H) * np.sum(self.Omega.reshape(1, -1) * hist, axis = 1)
            sample = np.random.multivariate_normal(np.zeros(self.Nexp +2), self.cov, chunk_size)
            W[:, i-1] = sample[:, 0]
            mul[:, i-1] = sample[:, -1] + hist_part
       

        V = np.exp(self.nu * mul - self.minue) # (chunk, M) 
        
        return V, W

    
    def generate_paths_chunk(self, chunk_size):        
        
        X_chunk = np.zeros((chunk_size, self.M)) 
        V, W = self.generate_V_chunk(chunk_size)
        W_perp = np.sqrt(self.tau) * np.random.randn(chunk_size, self.M)
        V_chunk = self.xi * V
        Z = self.rho * W + np.sqrt(1 - self.rho**2) * W_perp

        # by Forward Euler method，log of stock price
        X_chunk[:,0] = self.X0 - self.V0 * self.tau/2 + np.sqrt(self.V0) * Z[:, 0]

        for j in range(1, self.M):
            X_chunk[:,j] = X_chunk[:,j-1] - V_chunk[:,j-1] * self.tau/2 + np.sqrt(V_chunk[:,j-1]) * Z[:, j]

        return np.exp(X_chunk)


    # Generate the paths of stock price in parallel
    def S_(self):        
        np.random.seed(self.seed) 

        my_S = np.zeros(self.M).reshape(1, -1)
        
        chunk_size = int(np.ceil(self.P / self.num_cores/ self.my_loops))
        
        for i in range(self.my_loops):        
            S_chunks = Parallel(n_jobs=self.num_cores)(delayed(self.generate_paths_chunk)(chunk_size) for j in range(self.num_cores))

            # Concatenate the path chunks to form the final path array
            S_p = np.concatenate(S_chunks)            
            
            my_S = np.concatenate((my_S, S_p), axis = 0)    
        
        return my_S[1:, :] 
        