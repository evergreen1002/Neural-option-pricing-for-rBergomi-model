import numpy as np
from scipy.stats import norm
import scipy.special as sc
from scipy.special import gamma
from scipy.optimize import newton



def G(x, H):
    """
    Confluent hypergeometric function
    :param H: Hurst index
    """
    gamma = 0.5 - H
    return (1 - 2*gamma)/(1-gamma) * (x**gamma) * sc.hyp2f1(1, gamma, 2-gamma, x)    



def Cov_exact(m, H, rho):
    """
    Covariance matrix for exact method
    :param m: number of time steps
    :param H: Hurst index
    :param rho: correlation between Brownian motions
    return: the covariance matrix with size (2*m, 2*m)
    """
    tau = 1/m
    H_2 = 2 * H
    H_h = H + 0.5 
    rho_D = rho * np.sqrt(H_2)/H_h
    
    cov = np.zeros([2*m, 2*m], dtype = 'float32')
    for i in range(m):
        for j in range(i+1): # j<=i
            cov[i,j] = (tau * (j+1))**H_2 * G((j+1)/(i+1), H)
            cov[j,i] = cov[i,j] 
            
            cov[i+m, j+m] = tau * (j+1)
            cov[j+m, i+m] = cov[i+m, j+m]
            
            cov[i, j+m] = rho_D * (((i+1)*tau)**H_h - ((i - j)*tau)**H_h)
            cov[j+m, i] = cov[i, j+m]
            
            cov[j, i+m] = rho_D * ((j+1)*tau)**H_h
            cov[i+m, j] = cov[j, i+m]    
    
    return cov 



def Cov_SOE(N, Lambda, tau, H):
    """
    Covariance matrix for mSOE scheme
    :param N: number of summation terms
    :param Lambda: an array contains the nodes with size (N,)
    :param tau: time step size
    :param H: Hurst index
    return: the covariance matrix with size (N+2, N+2)
    """
    cov = np.zeros((N+2, N+2))
    cov[0,0] = tau
    cov[N+1, 0] = np.sqrt(2*H)/(H + 0.5) * (tau ** (H + 0.5))
    cov[0, N+1] = cov[N+1, 0]
    cov[N+1, N+1] = tau**(2*H)
        
    for i in range(N):
        cov[0, i+1] = 1/Lambda[i] * (1 - np.exp(-Lambda[i] * tau))
        cov[i+1, 0] = cov[0, i+1]
        
    for i in range(N):
        for j in range(i+1):
            cov[i+1, j+1] = 1/(Lambda[i] +  Lambda[j]) * (1 - np.exp(-(Lambda[i] +  Lambda[j]) * tau))
            cov[j+1, i+1] = cov[i+1, j+1]
        
    for i in range(N):
        cov[N+1, i+1] = np.sqrt(2*H) * Lambda[i]**(-H - 0.5) * sc.gammainc(H + 0.5, Lambda[i] * tau) * sc.gamma(H + 0.5)
        cov[i+1, N+1] = cov[N+1, i+1]
        
    return cov



def fbm(H, m, P):
    """
    Generate samples of fractional Brownian motion with Hurst index H
    :param H: Hurst index
    :param m: number of time steps
    :param P: number of samples
    return: an array contains P samples with size (P, m)
    """
    H_2 = 2 * H
    grid = np.linspace(0, 1, M + 1)[1:]
    mean = np.zeros(m)
    cov = np.zeros([m, m])

    # covariance matrix
    for i in range(m):
        for j in range(m):
            cov[i, j] = 0.5 * (grid[i] ** H_2 + grid[j] ** H_2 - np.abs(grid[i] - grid[j]) ** H_2)


    return np.random.multivariate_normal(mean, cov, P)


def bs(F, K, V, o = 'call'):
    """
    Compute the Black-Scholes price for call and put options
    :param F: forward stock price
    :param K: strike price
    :param V: integrated variance
    :param o: option token
    return: the BS option price
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P


def bsinv(P, F, K, t, o = 'call'):
    """
    Compute the BS implied volatility
    :param P: option price
    :param F: forward stock price
    :param K: strike price
    :param t: time to maturity
    :param o: option token
    return: BS implied volatility
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    s = newton(error, 1e-4)
    return s



