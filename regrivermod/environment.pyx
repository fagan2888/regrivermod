#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from __future__ import division
import numpy as np
import time
from scipy.stats import lognorm, gamma, truncnorm
import pylab
import math

cimport numpy as np
cimport cython
from econlearn.tilecode cimport Tilecode

from libc.math cimport fmin as c_min
from libc.math cimport fmax as c_max
from libc.math cimport log as c_log
from libc.math cimport exp as c_exp
from libc.math cimport sin as c_sin
from libc.math cimport cos as c_cos

from libc.stdlib cimport srand as c_seed
from libc.stdlib cimport rand
from libc.stdlib cimport RAND_MAX

cdef inline double c_rand() nogil:
   
    return rand() / (<double> RAND_MAX)

cdef inline double c_sum(int N, double[:] x) nogil:
    
    cdef double sumx = 0
    cdef int i = 0

    for i in range(N):
        sumx += x[i]

    return sumx


cdef class Environment:

    def __init__(self, para):

        self.w = 0                          # Withdrawal
        self.a = 0                          # Allocation 
        self.q = 0                          # Water consumption
        self.u = 0                          # Payoff
        self.p = 0                          # Marginal value of water (at allocation a)

        #b_w = para.b_w                      # Objective function parameters
        #b_value = para.b_value
        #self.b2 = b_w * b_value
        #self.b3 = (1 - b_w) * b_value

        #self.Lambda_I = para.env_inflow_share       # Inflow shares
        #self.Lambda_K = para.env_capacity_share
        
        self.delta_R = para.ch7['delta_R']         
        self.delta_Eb = para.ch7['delta_Eb']         
        self.delta_a = para.ch7['delta_a']
 
        self.DELTA = (1 - self.delta_R) / (1 - self.delta_Eb)

        self.pi = math.pi
        
        self.d = 0
        self.explore = 1
        self.state_zero = np.zeros(3)
    
    def test_random(self, int seed):
        
        c_seed(seed)

        for i in range(50):
            print c_rand()


    cdef double consume(self, double P, double t_cost, double p, double a):

        # Effective demand

        cdef double F3, q = 0

        if p > (P + t_cost):
            F3 = self.d_c + self.d_b * (P + t_cost)
            q = c_max((F3 - self.min_F2 + self.delta_a) * (self.DELTA**-1), 0)
        elif p < P:
            q = c_max(self.d_c + self.d_b * P, 0)
        else:
            q = c_max(c_min(a, self.d_c), 0)
        
        if q < self.min_q:
            return 0
        
        self.q = q

        return q
    
    cdef void allocate(self, double a, double min_F2, double F3_tilde):

        self.min_F2 = min_F2 #F1 - L(F1) - max E 

        cdef double F3 = 0
        
        self.a = a
        self.min_q = c_max((self.delta_a - min_F2) / self.DELTA , 0)
        self.d_c = (F3_tilde - min_F2 + self.delta_a)*(self.DELTA**-1)
        self.d_b = -1 * (1/2) * (F3_tilde**2) * ( self.b3**-1) * (self.DELTA**-2)

        # Inverse demand
        F3 = c_max(min_F2 + self.DELTA * a - self.delta_a, 0)
        if F3 > 0 and F3 < F3_tilde:
            self.p = (2 * self.b3 / F3_tilde) * (1 - F3 / F3_tilde) * self.DELTA 
        else:
            self.p = 0

    cdef double payoff(self, double F1, double F3, double F1_tilde, double F3_tilde):

        self.u = self.b1 * (1 - F1 / F1_tilde)**2  + self.b3 * (1 - F3 / F3_tilde)**2

        return self.u
    
    cdef double withdraw(self, double S, double s, double I, int M):
        
        cdef double[:] state = self.state_zero  
        cdef double U, V, Z 

        state[0] = S
        state[1] = s
        state[2] = I
        
        self.w = c_max(c_min(self.policy.one_value(state), s), 0)
        
        if self.explore == 1:
            U = c_rand()
            V = c_rand()
            Z = ((-2 * c_log(U))**0.5)*c_cos(2*self.pi*V)
            self.w = c_min(c_max(Z * (self.d * s) + self.w, 0), s)
        

    def set_policy(self, Tilecode policy, double d, int explore):

        self.policy = policy
        self.explore = explore
        self.d = d
