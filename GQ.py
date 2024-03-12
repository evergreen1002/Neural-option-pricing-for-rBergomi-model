import numpy as np
from scipy.special import gamma
from numpy import linalg as LA
import scipy.integrate as integrate

# Use Golub-Welsch algorithm to implement Gaussian quadrature 
# weight function is prespecified
class Golub_Welsch_0:
    def __init__(self, H, N):
        self.H = H # Hurst index 
        self.N = N # total number of nodes 
        
        #precomputation 
        alpha = 1.8 
        beta = 0.9
        a = 1 
        b = 1
        A_H = (1/self.H + 1/(1.5-self.H))**0.5
        self.m = int(np.ceil(beta/A_H * np.sqrt(self.N))) # Gaussian quadrature level on each interval 
        self.n = int(self.N/self.m) # number of intervals 
        self.true_N = self.m * self.n # true number of N         
        
        
        #intervals 
        xi_0 = a * np.exp(-alpha/((1.5-self.H) * A_H)*np.sqrt(self.N))
        xi_n = b * np.exp(alpha/(self.H * A_H) * np.sqrt(self.N))
        self.my_xi = np.empty(self.n+1)

        for i in range(self.n+1):
            self.my_xi[i] = xi_0 * (xi_n/xi_0)**(i/self.n)            
        
        #w_0 
        self.w_0 = 1/(gamma(0.5 - self.H)*(0.5-self.H)) * xi_0**(0.5-self.H)
        
    # weight function 
    def weight(self, x):        
        return 1/gamma(0.5 - self.H)*x**(-self.H-0.5)   
    
    
    # compute a_r and b_r for each interval 
    def a_r(self, p_r, l_b, u_b):      
    #l_b, u_b: lower bound and upper bound of the scalar product   
        
        a = integrate.quad(lambda x: x*p_r(x)**2*self.weight(x), l_b, u_b)[0]/\
            integrate.quad(lambda x: p_r(x)**2*self.weight(x), l_b, u_b)[0]
        return a
    
    
    def b_r(self, p_r, p_r_minus_1, l_b, u_b):
    
        b = integrate.quad(lambda x: p_r(x)**2*self.weight(x), l_b, u_b)[0]/\
            integrate.quad(lambda x: p_r_minus_1(x)**2*self.weight(x), l_b, u_b)[0]
        return b    
    
    # generate the Jacobi matrix for each interval 
    def Jacobi(self, l_b, u_b):       
        
        Jacobi_mat = np.zeros((self.m, self.m)) 
        polys = np.empty(self.m+1, dtype = object)
        my_a = np.zeros(self.m)
        my_b = np.zeros(self.m)       
      
               
        # Starting values 
        def p_0(x):
            return 1
        polys[0] = p_0        
        my_a[0] = self.a_r(polys[0], l_b, u_b)
        
        def p_1(x):
            return (x-my_a[0]) * polys[0](x)
        polys[1] = p_1
         
        
        # recurrence relation
        def p(x, a, b, poly_i, poly_i_minus_1):
            return (x-a)*poly_i(x) - b*poly_i_minus_1(x)       
        
        Jacobi_mat[0, 0] = my_a[0]
        if self.m == 1:
            pass
        else:            
            for r in range(1, self.m):
                my_a[r] = self.a_r(polys[r], l_b, u_b)
                my_b[r] = self.b_r(polys[r], polys[r-1], l_b, u_b)                
                polys[r+1] = lambda x, r=r: p(x, my_a[r], my_b[r], polys[r], polys[r-1])
               
            for r in range(1, self.m):
                Jacobi_mat[r, r] = my_a[r]
                Jacobi_mat[r, r-1] = np.sqrt(my_b[r])
                Jacobi_mat[r-1, r] = Jacobi_mat[r, r-1]
        
        return Jacobi_mat 
        
   
        
    # compute the nodes and the corresponding weights        
    def G_W(self):
        nodes = np.empty((self.n, self.m))
        ws = np.empty((self.n, self.m))
        
        for i in range(self.n):
            l_b = self.my_xi[i]
            u_b = self.my_xi[i+1]
           
            Jacobi_mat = self.Jacobi(l_b, u_b)
            eigval, eigvec = LA.eig(Jacobi_mat)
            
            nodes[i, :] = eigval
            mu_0 = integrate.quad(lambda x: self.weight(x), l_b, u_b)[0]
            ws[i, :] = mu_0 * eigvec[0,:]**2
            
        return nodes, ws
        
 