import scipy as sp
import numpy as np
from utils import Convar_SOE
import multiprocessing as mp
from joblib import Parallel, delayed

class rBergomi_SOE:
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
        self.H = params["H"] # Hurst index       
        
              
        #Precomputation 
        
        self.coef = np.exp(-np.arange(self.M).reshape(-1,1) * self.Lambda * self.tau) #(M, N)
        self.coef[0,:] = np.zeros(self.Nexp)

        self.minue = self.nu**2/2 * (self.grid[1:])**(2* self.H)
        
        # compute covariance matrix 
        self.cov = Convar_SOE(self.Nexp, self.Lambda, self.tau, self.H)      
        
        # enerate the stock price paths in parallel
        self.num_cores = cores
        self.my_loops = loop
        self.seed = rand_seed 



    # generate volatility paths without the forward variance   
    def generate_V_chunk(self, chunk_size, for_S, for_V):
        
        my_W_chunk = np.random.multivariate_normal(np.zeros(self.Nexp +2), self.cov, (chunk_size, self.M))
        
        # Z and W are correlated with rho 
        W = my_W_chunk[:,:,0] # (chunk_size, M)
        W_v= np.sqrt(self.tau) * np.random.randn(chunk_size, self.M)    
        Z = self.rho * W + np.sqrt(1 - self.rho**2) * W_v   
    
        # compute V
        random_matrix = my_W_chunk[:, :, 1:-1] # (chunk_size, M, Nexp)        
        
        #discrete convolution 
        coeff = np.repeat(self.coef.reshape(1, self.M, self.Nexp), chunk_size, axis = 0)  # (chunk_size, M, N) 
        convol = sp.signal.fftconvolve(random_matrix, coeff, axes = 1)[:, :self.M, :] # (chunk, M, N)  
        mul = np.sqrt(2 * self.H) * np.sum(self.Omega * convol, axis = -1) + my_W_chunk[:,:, -1] # (chunk, M) 

        V = np.exp(self.nu * mul - self.minue) # (chunk, M) 
        
        if for_S:
            return V, Z 
        elif for_V:
            return V
        else:
            return Z 
        
  
    
    def generate_paths_chunk(self, chunk_size):        
        
        X_chunk = np.zeros((chunk_size, self.M)) 
        V, Z = self.generate_V_chunk(chunk_size, True, False)
        V_chunk = self.xi * V

        # by Forward Euler method，log of stock price
        X_chunk[:,0] = self.X0 - self.V0 * self.tau/2 + np.sqrt(self.V0) * Z[:, 0]

        for j in range(1, self.M):
            X_chunk[:,j] = X_chunk[:,j-1] - V_chunk[:,j-1] * self.tau/2 + np.sqrt(V_chunk[:,j-1]) * Z[:, j]

        return np.exp(X_chunk)
     
        
    
    def V_(self):
        np.random.seed(self.seed)
        # Generate the vol paths in parallel (without the forward variance)
        
        my_V = np.zeros(self.M).reshape(1, -1)
        
        num_cores = mp.cpu_count()
        chunk_size = int(np.ceil(self.P / self.num_cores / self.my_loops))
        
        for i in range(self.my_loops):        
            V_chunks = Parallel(n_jobs=self.num_cores)(delayed(self.generate_V_chunk)(chunk_size, False, True)\
                                                  for j in range(self.num_cores))

            # Concatenate the path chunks to form the final path array
            V_p = np.concatenate(V_chunks)          
            
            my_V = np.concatenate((my_V, V_p), axis = 0)
    
        return my_V[1:, :] 
    
    def Z_(self):
        np.random.seed(self.seed)
        # Generate the Brownian motion paths in parallel
        
        my_Z = np.zeros(self.M).reshape(1, -1)
        
        
        chunk_size = int(np.ceil(self.P / self.num_cores / self.my_loops))
        
        for i in range(self.my_loops):        
            Z_chunks = Parallel(n_jobs=self.num_cores)(delayed(self.generate_V_chunk)(chunk_size, False, False) for j in range(self.num_cores))

            # Concatenate the path chunks to form the final path array
            Z_p = np.concatenate(Z_chunks)          
            
            my_Z = np.concatenate((my_Z, Z_p), axis = 0)
    
        return my_Z[1:, :] 
        

    def S_(self):        
        np.random.seed(self.seed) 
        # Generate the stock price paths in parallel
        
        my_S = np.zeros(self.M).reshape(1, -1)
        
        
        chunk_size = int(np.ceil(self.P / self.num_cores/ self.my_loops))
        
        for i in range(self.my_loops):        
            S_chunks = Parallel(n_jobs=self.num_cores)(delayed(self.generate_paths_chunk)(chunk_size) for j in range(self.num_cores))

            # Concatenate the path chunks to form the final path array
            S_p = np.concatenate(S_chunks)            
            
            my_S = np.concatenate((my_S, S_p), axis = 0)    
        
        return my_S[1:, :] 
        