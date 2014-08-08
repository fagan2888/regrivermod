from __future__ import division
import numpy as np
import time
from scipy.stats import lognorm, gamma
import pylab
cimport numpy as np
cimport cython
import math

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


cdef class Storage:

    def __init__(self, para):

        self.K = para.K                     # Storage capacity
        self.delta0 = para.delta0           # Storage loss (e.g. evaporation) function parameters
        self.alpha = para.alpha
        self.rho = para.rho_I               # Inflow distribution parameters
        self.k_I = para.k_I
        self.theta_I = para.theta_I
        self.delta1a = para.delta1a
        self.delta1b = para.delta1b

        self.I = para.I_bar 
        self.I_bar = para.I_bar
        
        self.Imax = self.K * 2      
        
        self.S = self.K                     # Initial storage level
        self.Spill                          # Storage spills
        self.Loss = 0

        self.pi = math.pi
    
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

    @cython.initializedcheck(False) 
    cdef double update(self, double W, int t):
         "Draw I randomly and update storage level given W"
         
         self.I = c_min(self.rho * self.I + self.EPS[t], self.K * 2)

         self.Loss = self.delta0 * self.alpha * (self.S)**(2/3)
        
         self.Spill = c_max(self.I - (self.K - (self.S - W - self.Loss)), 0)
        
         self.S = c_max(c_min(self.S - W - self.Loss + self.I, self.K), 0)

         return self.S

    def update_test(self, W, t):
         "Draw I randomly and update storage level given W"

         self.update(W, t)

         return self.S
    
    def transition(self, S, W, I):
         "Update storage given S, W and I"
         
         self.Loss = self.delta0 * self.alpha * (self.S - W)**(2/3)
         
         return np.maximum( np.minimum(S - W - self.Loss + I, self.K) ,0)
    
    cdef double release(self, double W):

        return c_max((1 - self.delta1b) * W - self.delta1a,0)

    def get_loss(self, S, W):

        self.Loss = self.delta0 * self.alpha * (self.S - W)**(2/3)

        return self.Loss

    def I_series(self, k):
         "Draw a random series for F of length k "
         
         mu = c_exp(self.mu)

         I0 = lognorm.mean(self.sig, loc=0, scale=mu) / (1-self.rho)
         eps = lognorm.rvs(self.sig, loc=0, scale=mu, size = k) 
         I = np.zeros(k)

         for i in range(k):
            I[i] = self.rho * I0 + eps[i]
            I0 = I[i]

         return I

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
