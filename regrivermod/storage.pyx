from __future__ import division
import numpy as np
import time
from scipy.stats import lognorm, gamma, truncnorm
import pylab
cimport numpy as np
cimport cython
import math
from econlearn.tilecode cimport Tilecode


cdef extern from "math.h":
    double c_min "fmin" (double, double)

cdef extern from "math.h":
    double c_max "fmax" (double, double)

cdef extern from "math.h":
    double c_log "log" (double)

cdef extern from "math.h":
    double c_cos "cos" (double)

cdef extern from "math.h":
    double c_exp "exp" (double)

cdef extern from "stdlib.h":
    double c_rand "drand48" ()

cdef extern from "stdlib.h":
    void c_seed "srand48" (int)

cdef inline double c_sum(int N, double[:] x):
    
    cdef double sumx = 0
    cdef int i = 0

    for i in range(N):
        sumx += x[i]

    return sumx

cdef inline double loss_12(F1, delta_a, delta_b, F_bar):

    if F1 < F_bar:
        return c_min(F1, delta_a)
    else:
        return delta_a + delta_b * (F1 - F_bar)

cdef class Storage:

    def __init__(self, para):

        self.K = para.K                     # Storage capacity
        
        self.delta0 = para.delta0           # Storage loss (e.g. evaporation) function parameters
        self.alpha = para.alpha
        
        self.rho = para.rho_I               # Inflow distribution parameters
        self.k_I = para.k_I
        self.theta_I = para.theta_I
        
        self.delta1a = para.delta1a         # Delivery losses
        self.delta1b = para.delta1b
        
        self.omega_mu = para.ch7['omega_mu']        # Summer-Winter inflow split
        self.omega_sig = para.ch7['omega_sig']
        self.omega_ab = np.array(para.ch7_param['omega_ab'])

        self.delta_a   = para.ch7['delta_a']                     # ch7 river flow parameters
        self.delta_b  = para.ch7['delta_b'] 
        self.delta_Ea = para.ch7['delta_Ea']
        self.delta_Eb = para.ch7['delta_Eb']
        self.delta_R = para.ch7['delta_R']
        self.F_bar = np.zeros(2)
        self.F_bar[0] = para.ch7['F_bar'] * (5/12)
        self.F_bar[1] = para.ch7['F_bar'] * (7/12)

        self.I = para.I_bar
        self.C = para.I_bar
        self.I_bar = para.I_bar
        
        self.Imax = self.K * 2      
        
        self.S = self.K                     # Initial storage level
        self.Spill = 0                          # Storage spills
        self.Loss = 0

        self.pi = math.pi
        self.X = np.zeros(2)

    def seed(self, i):
        seed = int(time.time()/(i + 1))
        c_seed(seed)
        np.random.seed(seed)
    
    def precompute_I_shocks(self, T, seed=0):
         "Draw a random series for eps_I of length t "
         if seed == 0:
             np.random.seed()
         else:
             np.random.seed(seed)

         self.EPS = np.zeros(T)
         self.EPS = gamma(self.k_I, loc=0, scale=self.theta_I).rvs(size=T)
    
    def precompute_I_split(self, T, seed=0):
         "Draw a random series for omega_t, (winter share of annual flow)"
         if seed == 0:
             np.random.seed()
         else:
             np.random.seed(seed)

         self.OMEGA = np.zeros(T)
         self.OMEGA = truncnorm(self.omega_ab[0], self.omega_ab[1], loc=self.omega_mu, scale=self.omega_sig).rvs(size=T) 

    @cython.initializedcheck(False) 
    cdef double update(self, double W, int t):
         "Draw I randomly and update storage level given W"
         
         self.I = c_min(self.rho * self.I + self.EPS[t], self.K * 2)

         self.storage_transition(W)        

         return self.S

    def update_test(self, W, t):
         "Draw I randomly and update storage level given W"

         self.update(W, t)

         return self.S

    @cython.initializedcheck(False) 
    cdef double storage_transition(self, double W):

         self.Loss = self.delta0 * self.alpha * (self.S)**(0.666666666666666)
        
         self.Spill = c_max(self.I - (self.K - (self.S - W - self.Loss)), 0)
        
         self.S = c_max(c_min(self.S - W - self.Loss + self.I, self.K), 0)
    
    
    @cython.initializedcheck(False) 
    cdef double release(self, double W):
        "Calculate storage release need to satisfy aggregate orders W"

        return c_max((1 - self.delta1b) * W - self.delta1a,0)
    
    @cython.initializedcheck(False) 
    cpdef double release_ch7(self, double sum_w, int M):
        
        cdef double W = 0

        # Determine release required to satisfy user withdrawals
        if sum_w > 0:
            if M == 0:
                W = sum_w + self.delta_a + self.delta_Ea
                if W > self.F_bar[0]:
                    W = (W - self.F_bar[0] * self.delta_b)*((1 - self.delta_b)**-1)
            else:
                W = sum_w + self.delta_a

        self.max_E = sum_w

        self.min_F2 = (W + self.Spill) - loss_12(W + self.Spill, self.delta_a, self.delta_b, self.F_bar[M]) - self.max_E

        return W
    
    def loss12(self, F1, int M):
        """Python wrapper for inline function loss_12"""

        return loss_12(F1, self.delta_a, self.delta_b, self.F_bar[M])

    @cython.initializedcheck(False) 
    cpdef double extract_ch7(self, double E):
        "ch7: Returns water available at demand node, given extraction E"

        return c_max((1 - self.delta_Eb) * E - self.delta_Ea , 0)   
    
    @cython.initializedcheck(False) 
    cdef void river_flow(self, double W, double E, int M):
        "ch7: Compute river flows at each node, given releases W and extraction"
        
        self.F1 = W + self.Spill

        self.loss_12 = loss_12(self.F1, self.delta_a, self.delta_b, self.F_bar[M])

        self.F2 = c_max(self.F1 - self.loss_12 - E, 0)

        self.F3 = c_max(self.F2 - c_min(self.F2, self.delta_a) +  self.delta_R * E, 0)

    @cython.cdivision(True) 
    @cython.initializedcheck(False) 
    cdef double update_ch7(self, double W, double E, int M, int t):
         
         if M == 0:
            self.C = c_min(self.rho * self.C + self.EPS[t], self.I_bar * 4)
            self.I = self.C * (1 - self.OMEGA[t])
         else:
            self.I = self.C * self.OMEGA[t] 
         
         self.storage_transition(W)
         
         # Natural flows
         self.river_flow(self.I, 0, M)
         self.F1_tilde = self.F1
         self.F2_tilde = self.F2
         self.F3_tilde = self.F3

         # Actual flows
         self.river_flow(W, E, M)

         return self.S

    def update_ch7_test(self, W, E, M, t):

        self.update_ch7(W, E, M, t)
       
        print 'Annual inflow: ' + str(self.C)
        print 'Seasonal inflow: ' + str(self.I)
        
        print 'Storage loss: ' + str(self.Loss)
        print 'Storage spill: ' + str(self.Spill)
        print 'Storage volume: ' + str(self.S)

        print 'River flow, node 1, actual: ' + str(self.F1) + ', natural: ' + str(self.F1_tilde)
        print 'River flow, node 2, actual: ' + str(self.F2) + ', natural: ' + str(self.F2_tilde)
        print 'River flow, node 3, actual: ' + str(self.F3) + ', natural: ' + str(self.F3_tilde)
        print 'River loss, node 1 to 2, actual: ' + str(self.loss_12) 

    def river_flows_ch7_test(self, int T):
        
        cdef int t, M
        cdef double[:, :] I_sim = np.zeros([T, 2])
        cdef double[:, :] F1_sim = np.zeros([T, 2])
        cdef double[:, :] F2_sim = np.zeros([T, 2])
        cdef double[:, :] F3_sim = np.zeros([T, 2])

        self.S = 0
        
        for t in range(T):
            for M in range(2):
                self.update_ch7(-1, 0, M, t)
                
                I_sim[t, M] = self.I
                F1_sim[t, M] = self.F1
                F2_sim[t, M] = self.F2
                F3_sim[t, M] = self.F3

        for M in range(2):
            print 'Inflow, ' + str(M) 
            pylab.hist(I_sim[:, M], bins=20)
            pylab.show()
            print 'Flow 1, ' + str(M) 
            pylab.hist(F1_sim[:, M], bins=20)
            pylab.xlim(0, np.max(I_sim))
            pylab.show()
            print 'Flow 2, ' + str(M) 
            pylab.hist(F2_sim[:, M], bins=20)
            pylab.xlim(0, np.max(I_sim))
            pylab.show()
            print 'Flow 3, ' + str(M) 
            pylab.hist(F3_sim[:, M], bins=20)
            pylab.xlim(0, np.max(I_sim))
            pylab.show()

        return [I_sim, F1_sim, F2_sim, F3_sim]

    def get_loss(self, S, W):

        self.Loss = self.delta0 * self.alpha * (self.S - W)**(2/3)

        return self.Loss

    def discrete_I_pdf(self, points):
        "Build discrete version of I distribution"

        ## Rows are I(-1) columns are I
        I_grid = np.vstack([np.linspace(0, self.Imax, points)]*points)
        d = I_grid[0,1] - I_grid[0,0]
        I_pr  = np.zeros([points, points])

        for i in range(points):
            for j in range(points):
                I_old = I_grid[i,i]
                I_low = I_grid[i,j] - d/2 - I_old * self.rho
                I_high = I_grid[i,j] + d/2 - I_old * self.rho
                if j == (points - 1):
                    I_high = self.K * 100
                I_pr[i,j] = gamma.cdf(I_high, self.k_I, loc=0, scale=self.theta_I) - gamma.cdf(I_low, self.k_I, loc=0, scale=self.theta_I)
        
        self.I_grid = I_grid
        self.I_pr = I_pr
    
    def transition(self, S, W, I):
         "Update storage given S, W and I"
         
         self.Loss = self.delta0 * self.alpha * (self.S - W)**(2/3)
         
         return np.maximum( np.minimum(S - W - self.Loss + I, self.K) ,0)

