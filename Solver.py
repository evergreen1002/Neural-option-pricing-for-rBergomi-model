import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, IterableDataset



# load data 
class MyIterable(IterableDataset):    
    def __init__(self, file_path: str, start, end):
        super(MyIterable, self).__init__()
        self.file_path = file_path        
        self.start = start
        self.end = end
   
    def __iter__(self):
        sample = []
        with open(self.file_path, 'r') as f:             
            for i, line in enumerate(f):
                if i < self.start: continue 
                if i >= self.end: 
                    break                
                sample.append([float(i) for i in line.split()]) 
            
        return np.array(sample)        
    
    def __len__(self):
        num = 0 
        with open(self.file_path, 'r') as f:
            for _ in enumerate(f):
                num += 1
        return num 
    

def MyDataset(start, rank_size, device, paths):
    end = start + rank_size
    # load data 
    ys = MyIterable(paths[0], start, end).__iter__()
    mul = MyIterable(paths[1], start, end).__iter__()
    z = MyIterable(paths[2], start, end).__iter__()
    
    # convert numpy array to tensor 
    ys = torch.from_numpy(ys).to(torch.float32).to(device)
    mul = np.c_[np.ones(rank_size), mul[:, :-1]]
    mul = torch.from_numpy(mul).to(torch.float32).to(device)
    z = torch.from_numpy(z).to(torch.float32).to(device)
    
    train_rank = TensorDataset(ys, mul, z)
    return train_rank
 

class my_MLP(torch.nn.Module):
    def __init__(self, in_size, mlp_size, num_layers):
        # in_size: input size 
        # mlp_size: size of hidden layers 
        # num_layers: num of hidden layers 
        super().__init__()
    
        model = [torch.nn.Linear(in_size, mlp_size), torch.nn.LeakyReLU(0.1)]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(torch.nn.LeakyReLU(0.1))
            
        #output size: 1
        model.append(torch.nn.Linear(mlp_size, 1))
        # model.append(torch.nn.Tanh())
    
        self._model = torch.nn.Sequential(*model)
        
    def forward(self, x):
        return self._model(x) 
    

# numerical SDE solver 
# specifically applies to rough bergomi model 
# can only deal with the 1-dim BM case and use forward Euler method 
class sdeint:
    def __init__(self, neural_sde, x0, ts, mul, Z):
        
        # forward variance curve as neural sde 
        self.neural_sde = neural_sde
        # initial log stock price 
        self.x0 = x0 #(batch_size, )
        self.batch_size = x0.shape[0]        
        
        # discretized time grid
        self.ts = ts #(M, )
        self.num_grid = ts.shape[0] # =M
        self.tau = ts[1] - ts[0]
        
        # precomputed paths of volatility and Brownian motion  
        self.mul = mul
        self.Z = Z        
    
        
    def __call__(self):
        # forward Euler 
        neural_xs = torch.zeros(self.batch_size, self.num_grid + 1, device = device)       
        
        for i in range(1, self.num_grid+1):
            t = self.ts[i-1]
            V = self.neural_sde(t.reshape(-1,1)).squeeze(-1) * self.mul[:, i-1] #(batch_size, )
            neural_xs[:, i] = neural_xs[:, i-1] - V * self.tau/2 + torch.sqrt(V) * self.Z[:, i-1]
            
        return neural_xs[:, 1:] #(batch_size, M)


def price(ys, strikes):
    sample_size = ys.size(0) 
    strike_size = strikes.shape[0] # strikes is a 1-d array, has shape (strike_size, )    
    
    with torch.no_grad():
        y_T = ys[:, -1].cpu().numpy() #(P, )
        
    Y = np.tile(y_T, (strike_size, 1)) #(strike_size, P)
    K = np.tile(np.reshape(strikes, (-1, 1)), (1, sample_size)) #(strike_size, P)
    Y_K = Y - K
    Y_K[Y_K < 0] = 0
    price = np.mean(Y_K, -1) # 1-d array, has shape (strike_size, )
    return price


#plot the marginal distribution at T and the option price 
def my_plot(neural_ys, real_ys, neural_price, real_price, strikes):
    
    with torch.no_grad():
        neural_ys_1 = neural_ys[:, -1].cpu().numpy()            
        real_ys_1 = real_ys[:, -1].cpu().numpy()   
 
    plt.figure(figsize = (12, 5))
    plt.subplot(1,2,1)     
    _, bins, _ = plt.hist(neural_ys_1, bins = 100, alpha = 0.7, color = "crimson", density = True)
    bin_width = bins[1] - bins[0]
    num_bins = int((real_ys_1.max() - real_ys_1.min()) // bin_width)
    plt.hist(real_ys_1, bins = 100 , alpha = 0.7, color = "dodgerblue", density = True)
    plt.legend(["Neural SDE", "Real"], fontsize = 12)
    plt.xlabel("Value", fontweight = "heavy")
    plt.ylabel("Density", fontweight = "heavy")
    plt.title("Empirical distribution at t = T", fontsize = 14, fontweight = "heavy")
    plt.grid(True)
    
    plt.subplot(1,2,2)    
    plt.plot(strikes, neural_price, color = "crimson", lw = 2)
    plt.plot(strikes, real_price, color = "dodgerblue", lw = 2)
    plt.legend(["Neural SDE price", "Real price"], fontsize = 12)
    plt.xlabel("Strikes", fontweight = "heavy")
    plt.ylabel("Price", fontweight = "heavy")
    plt.title("Option price", fontsize = 14, fontweight = "heavy") 
    plt.grid(True)
    plt.show() 
    

# return the Wasserstein_p distance between two empircial ditributions 
def Wasserstein_p(real_ys, neural_ys, p):
    # real_ys, neural_ys have size (batch_size, M)
    
    real_ys_sorted, _ = torch.sort(real_ys, 0) 
    neural_ys_sorted, _ = torch.sort(neural_ys, 0)
    loss = torch.mean(torch.abs(real_ys_sorted - neural_ys_sorted)**p, 0)**(1/p) #(M, )
    return loss





