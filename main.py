import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from Solver import *



is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'
if not is_cuda:
    print("Warning: CUDA is not available, use CPU instead") 

M = 2000
T = 1
tau = T/M
X0 = 0
V0 = 0.235**2
xi = 0.235**2
rho = -0.9
nu = 1.9
H = 0.07
batch_size = 2**13
num_batch = 120
P_train = num_batch * batch_size# size of train set 
P_valid = 10000
P_test = 10000 # size of test set 

rank_size = int(15 * batch_size) # sample size for every loading 
num_rank = int (num_batch/15) # number of loading 

# epochs = 10
# for test set, all moves to cpu
x0_t = torch.zeros(P_test, 1, device = device)
x0_b = torch.zeros(batch_size, 1, device = device)

logmoneyness = np.arange(-0.5, 0.31, 0.01)
strikes = np.exp(logmoneyness)


S_path  = "/lustre1/u/u3553440/const_S/S_total"
V_path = "/lustre1/u/u3553440/V/V_total"
Z_path = "/lustre1/u/u3553440/Z/Z_total"
paths = [S_path, V_path, Z_path]

# load the validate set 
start_valid = 1000000
valid_set = MyDataset(start_valid, P_valid, device, paths)

# load test set
start_test = 1010000
test_set = MyDataset(start_test, P_test, device, paths)


# neural_ys 
forward_var = my_MLP(1, 100, 10)
forward_var = forward_var.to(device)

ts = torch.linspace(tau, T, M, device = device)
neural_ys = sdeint(forward_var, x0_t, ts, test_set[:P_test][1], test_set[:P_test][2])()
neural_ys = torch.exp(neural_ys)


#real_price
real_price = price(test_set[:P_test][0], strikes)

# neural_price 
neural_price = price(neural_ys, strikes)

# plot before training 
my_plot(neural_ys, test_set[:P_test][0], neural_price, real_price, strikes)

#optimization
my_optimizer = torch.optim.Adam(forward_var.parameters(), lr= 1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(my_optimizer, 'min', verbose = True, patience = 1)


loss_history = []
max_error_history = []
# Wasserstein-1 distance as loss function
epoch = 5
random_rank = list(range(num_rank))
random.shuffle(random_rank)
for i in range(epoch):    
    print(f"Epoch {i+1}\n-------------------------")
    for rank in random_rank:
        start = rank * rank_size
        train_rank = MyDataset(start, rank_size, device, paths)
        train_rank_loader = DataLoader(train_rank, batch_size = batch_size, shuffle = True)
    
        for batch, train_samples in enumerate(train_rank_loader):         
        
            real_samples, mul_samples, Z_samples = train_samples # (batch_size, M)
        
            neural_samples = sdeint(forward_var, x0_b, ts, mul_samples, Z_samples)()
            neural_samples = torch.exp(neural_samples) # (batch_size, M)
        
            # compute Wasserstein-1 distance at all time grids 
            loss = Wasserstein_p(real_samples, neural_samples, p = 1)[-1]
        
            # Wasserstein-1 distance at t = T
            loss_history.append(loss.item())
        
            # average Wasserstein distance at all time steps
            # ave_loss = torch.mean(loss)            
            price_error = np.abs(price(neural_samples, strikes) - price(real_samples, strikes))
            max_error_history.append(price_error.max())        

            my_optimizer.zero_grad()
            # use the Wasserstein distance at T as loss
            loss.backward()
            my_optimizer.step()        
         
            current = batch * batch_size 
            print(f"loss: {loss:>7f}  [{current:>5d}/{P_train:>5d}]")
        
    random.shuffle(random_rank)
    
    # check the model's performance on validate set after every epoch 
    
    neural_ys = sdeint(forward_var, x0_t, ts, test_set[:P_test][1], test_set[:P_test][2])()
    neural_ys = torch.exp(neural_ys)
        
    valid_loss = Wasserstein_p(test_set[:P_test][0], neural_ys, p = 1)[-1]
    scheduler.step(valid_loss)
    print(f"Valid Loss: {valid_loss:>7f}")
        
  



