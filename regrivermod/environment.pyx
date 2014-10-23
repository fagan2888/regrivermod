#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from __future__ import division
import numpy as np
import time
from scipy.stats import lognorm, gamma, truncnorm
import pylab
import math

cimport cython
from econlearn.tilecode cimport Tilecode
from regrivermod.storage cimport Storage

from libc.math cimport fmin as c_min
from libc.math cimport fmax as c_max
from libc.math cimport log as c_log
from libc.math cimport exp as c_exp
from libc.math cimport sin as c_sin
from libc.math cimport cos as c_cos
from libc.math cimport fabs as c_abs

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

    def __init__(self, para, turn_off=False):

        self.w = 0                          # Withdrawal
        self.a = 0                          # Allocation 
        self.q = 0                          # Water consumption
        self.u = 0                          # Payoff
        self.p = 0                          # Marginal value of water (at allocation a)

        b_w = para.ch7['b_w']                      # Objective function parameters
        b_value = para.ch7['b_value']
        self.b1 = b_w * b_value
        self.b3 = (1 - b_w) * b_value

        self.Lambda_I = para.ch7['inflow_share']       # Inflow shares
        self.Lambda_K = para.ch7['capacity_share']
        
        self.delta_R = para.ch7['delta_R']         
        self.delta_Eb = para.ch7['delta_Eb']         
        self.delta_a = para.ch7['delta_a']
 
        self.DELTA = (1 - self.delta_R) / (1 - self.delta_Eb)
        self.pi = math.pi
        
        self.d = 0
        self.explore = 1
        self.state_zero = np.zeros(3)

        self.t_cost = para.t_cost
        self.min_F2 = 0
        self.d_b = 0
        self.d_c = 0
        self.min_q = 0

        if turn_off:
            self.turn_off = 1
        else:
            self.turn_off = 0

    cdef double consume(self, double P, int planner):

        # Effective demand

        cdef double F3, q = 0
        cdef t_cost

        if planner == 1:
            t_cost = 0
        else:
            t_cost = self.t_cost

        if self.p > (P + t_cost):
            F3 = self.d_c + self.d_b * (P + t_cost)
            q = c_max((F3 - self.min_F2 + self.delta_a) * (self.DELTA**-1), 0)
        elif self.p <= P:
            q = c_max(self.d_c + self.d_b * P, 0)
        else:
            q = c_max(c_min(self.a, self.d_c), 0)
        
        if q <= self.min_q:
            q = 0
        
        self.q = q

        if planner == 1:
            self.a = q

        return q
    
    cdef allocate(self, double a, double min_F2, double F3_tilde):

        self.min_F2 = min_F2 #F1 - L(F1) - max E 

        cdef double F3 = 0

        self.a = a
        self.min_q = c_max((self.delta_a - min_F2) / self.DELTA , 0)
        self.d_c = (F3_tilde - min_F2 + self.delta_a)*(self.DELTA**-1)
        self.d_b = -0.5 * (F3_tilde**2) * (self.b3**-1) * (self.DELTA**-2)

        if self.turn_off == 1:
            self.d_c = 0
            self.d_b = 0

        # Inverse demand
        F3 = c_max(min_F2 + self.DELTA * a - self.delta_a, 0)
        if 0 < F3 < F3_tilde:
            self.p = (2 * self.b3 / F3_tilde) * (1 - F3 / F3_tilde) * self.DELTA
        else:
            self.p = 0

        self.Pmax = 2 * self.b3 / F3_tilde * self.DELTA
    
    @cython.cdivision(True)
    cdef double payoff(self, double F1, double F3, double F1_tilde, double F3_tilde, double P):

        cdef double u1 = 0
        cdef double u3 = 0

        if self.turn_off == 1:
            self.u = 0
        else:
            if F1_tilde  > 0:
                u1 = -1 * (self.b1 * ((F1_tilde - F1)/F1_tilde)**2)
            if F3_tilde  > 0:
                u3 = -1 * (self.b3 * ((F3_tilde - F3)/F3_tilde)**2)
            if self.q > self.a:     # Water buyer
                self.u = u1 + u3 + (self.a - self.q) * (P + self.t_cost)
            else:                   # Water seller or non-trader
                self.u = u1 + u3 + (self.a - self.q) * P

        return self.u
    
    cdef double withdraw(self, double S, double s, double I, int M):
        
        cdef double[:] state = self.state_zero  
        cdef double U, V, Z 

        state[0] = S
        state[1] = s
        state[2] = I

        if self.turn_off == 1:
            self.w = 0
        else:
            if M == 0:
                self.w = c_max(c_min(self.policy0.one_value(state), s), 0)
            else:
                self.w = c_max(c_min(self.policy1.one_value(state), s), 0)

            if self.explore == 1:
                U = c_rand()
                V = c_rand()
                Z = ((-2 * c_log(U))**0.5)*c_cos(2*self.pi*V)
                self.w = c_min(c_max(Z * (self.d * s) + self.w, 0), s)

    def plot_demand(self, ):

        P = np.linspace(0, 3000, 600)
        Q = np.zeros(600)
        for i in range(600):
            Q[i] = self.consume(P[i], 0)
        print str(np.array(Q))
        print str(np.array(P))
        import pylab
        pylab.plot(Q, P)
        pylab.show()

    def set_policy(self, Tilecode policy0, Tilecode policy1, double d, int explore):

        self.policy0 = policy0
        self.policy1 = policy1
        self.explore = explore
        self.d = d

    def init_policy(self, Tilecode W_f0, Tilecode V_f0, Tilecode W_f1, Tilecode V_f1, Storage storage, linT, CORES, radius):

        cdef int i, N = 100000    
        cdef double[:] state = np.zeros(2)
        cdef double s, I, S
        cdef double[:,:] X = np.zeros([N, 3])
        cdef double[:] w0 = np.zeros(N)
        cdef double[:] v0 = np.zeros(N)
        cdef double[:] w1 = np.zeros(N)
        cdef double[:] v1 = np.zeros(N)
        cdef double wp, vp = 0
        cdef double fl = storage.delta_a[0] + storage.delta_Ea

        for i in range(N):
            
            X[i, 0] = c_rand() * storage.K
            X[i, 1] = c_rand() * self.Lambda_K * (storage.K - fl)
            X[i, 2] = c_rand() * storage.Imax  * (storage.I_bar**-1)
            
            state[0] = X[i, 1] * (self.Lambda_I**-1) + fl
            state[1] = X[i, 2]    
            
            wp0 = W_f0.one_value(state)
            vp0 = V_f0.one_value(state)
            
            state[0] = X[i, 1] * (self.Lambda_I**-1) + storage.delta_a[1]
            
            wp1 = W_f1.one_value(state)
            vp1 = V_f1.one_value(state)
            
            w0[i] = c_max(c_min((wp0 - fl) * self.Lambda_I, X[i, 1]), 0)
            w1[i] = c_max(c_min((wp1 - storage.delta_a[1]) * self.Lambda_I, X[i, 1]), 0)
            v0[i] = vp0 * self.Lambda_I
            v1[i] = vp1 * self.Lambda_I
        
        Twv = int((1 / radius) / 2)
        T = [Twv for t in range(4)]
        L = int(130 / Twv)
        self.policy0 = Tilecode(3, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
        self.policy1 = Tilecode(3, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
        self.value0 = Tilecode(3, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
        self.value1 = Tilecode(3, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
        
        self.policy0.fit(X, w0)
        self.value0.fit(X, v0)
        self.policy1.fit(X, w1)
        self.value1.fit(X, v1)
