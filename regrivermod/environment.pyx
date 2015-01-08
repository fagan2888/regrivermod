#!python
#cython: cdivision=True, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False

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
    
        self.t = 0

        self.b1 = para.ch7['b_1']                      # Objective function parameters
        self.b_value = para.ch7['b_value']
        self.Bhat_alpha = para.ch7['Bhat_alpha']
        self.b3 = (1- self.b1)

        self.Lambda_I = para.ch7['inflow_share']       # Inflow shares
        self.Lambda_K = para.ch7['capacity_share']
        
        self.delta_R = para.ch7['delta_R']         
        self.delta_Eb = para.ch7['delta_Eb']         
        self.delta_a = np.zeros(2)
        self.delta_a[0] = para.ch7['delta_a'] * para.ch7['omegadelta']
        self.delta_a[1] = para.ch7['delta_a'] * (1 - para.ch7['omegadelta'])
        self.DELTA0 = (1 - self.delta_R) / (1 - self.delta_Eb)
        self.DELTA1 = 1 / (1 - self.delta_Eb)
        self.pi = math.pi
        
        self.d = 0
        self.explore = 1
        self.state_zero = np.zeros(5)

        self.t_cost = para.t_cost
        self.d_b = 0
        self.d_c = 0
        self.min_q = 0
        self.Bhat = 0.5
        self.Pmax = 0
        self.e_sig = para.ch7['e_sig']
        self.budget = 0

        if turn_off:
            self.turn_off = 1
        else:
            self.turn_off = 0

    def seed(self, i):
        seed = int(time.time()/(i + 1))
        c_seed(seed)
        np.random.seed(seed)
    
    def precompute_e_shocks(self, T, seed=0):
        
        "Draw a random series for eps_I of length t "
        
        T = T * 2
        
        if seed == 0:
            np.random.seed()
        else:
            np.random.seed(seed)

        self.e = np.zeros(T)
        self.e = truncnorm(-1*(self.e_sig**-1), 1*(self.e_sig**-1), loc=1, scale=self.e_sig).rvs(size=T) 
    
    def set_shares(self, Lambda):

        self.Lambda_I = Lambda
        self.Lambda_K = Lambda

    cdef double consume(self, double P, int M, int planner):

        # Effective demand

        cdef double F3, q = 0
        cdef double t_cost = 0

        if planner == 1:
            t_cost = 0
        else:
            t_cost = self.t_cost

        if self.p > (P + t_cost):
            q = c_max(self.d_c + self.d_b * (P + t_cost), 0)
        elif self.p < P:
            q = c_max(self.d_c + self.d_b * P, 0)
        else:
            q = c_max(c_min(self.a, self.d_c), 0)
        
        if M == 1:
            self.q = c_max(q, self.a)
        else:
            self.q = q
        
        if planner == 1:
            self.a = q

        return q
    
    cdef void allocate(self, double a, double Z, double max_R, double F1_tilde, double F3_tilde, double Pr, int M):

        cdef double F1, F3 = 0
        cdef double b1hat, bhat3
        cdef double F1_marg = 0
        cdef double F3_marg = 0

        self.Bhat *= Pr
        self.a = a

        if self.turn_off == 1:
            self.d_c = 0
            self.d_b = 0
        else:
            if M == 0:
                if F3_tilde > 0:
                    b3hat = (2 * self.b3 * self.b_value * self.Bhat) / F3_tilde
                    self.min_q =  0 #c_max((self.delta_a[M] - Z) / self.DELTA0 , 0)
                    
                    # Demand coefficients
                    self.d_c = (F3_tilde - Z - max_R)*(self.DELTA0**-1)
                    self.d_b = -1* (b3hat**-1) * (self.DELTA0**-2) * F3_tilde
                
                    # Inverse demand
                    F3 = c_max(Z + self.DELTA0 * a, 0) + max_R
                    self.p = b3hat * self.DELTA0 * (1 - F3 / F3_tilde) 
                    if self.d_b < 0:
                        self.Pmax = (self.min_q - self.d_c)*self.d_b**-1
                    else:
                        self.Pmax = 0 
                else:
                    self.d_c = 0
                    self.d_b = 0
                    self.p = 0
                    self.Pmax = 0
                    self.min_q = 0

            elif M == 1:
                if F1_tilde > 0 and F3_tilde > 0:
                    b1hat = (2 * self.b1 * self.b_value * self.Bhat) / F1_tilde
                    b3hat = (2 * self.b3 * self.b_value * self.Bhat) / F3_tilde
                    self.min_q = 0
                
                    # Demand coefficients
                    self.d_c =  (self.DELTA1**-1) * ((b1hat + b3hat - 2*self.delta_a[M]*b1hat*(F1_tilde**-1)) / ((b1hat*F1_tilde**-1) + b3hat*(F3_tilde**-1)) - Z)
                    self.d_b = -1 / ((self.DELTA1**2) * (b1hat*F1_tilde**-1 + b3hat*F3_tilde**-1) ) 

                    # Inverse demand
                    F1 = Z + self.DELTA1 * a + 2 * self.delta_a[1] 
                    F3 = F1 - 2 * self.delta_a[1]
                   
                    F1_marg = (1 - F1 * (F1_tilde**-1))
                    F3_marg = (1 - F3 * (F3_tilde**-1))
                
                    self.p = c_max(b1hat * self.DELTA1 * F1_marg + b3hat * self.DELTA1 * F3_marg, 0)

                    if self.d_b < 0:
                        self.Pmax = (self.min_q - self.d_c)*self.d_b**-1
                    else:
                        self.Pmax = 0
                else:
                    self.d_c = 0
                    self.d_b = 0
                    self.min_q = 0
                    self.p = 0
                    self.Pmax = 0

    
    @cython.cdivision(True)
    cdef double payoff(self, double F1, double F3, double F1_tilde, double F3_tilde, double P):

        cdef double u = 0
        cdef double deltaF1
        cdef double deltaF3
        cdef double trade = 0

        if F1_tilde  > 0:
            deltaF1 = c_min(((F1_tilde - F1)/F1_tilde)**2, 1)
        elif F1 > 0:
            deltaF1 = 1
        if F3_tilde > 0:
            deltaF3 = c_min(((F3_tilde - F3)/F3_tilde)**2, 1)
        elif F3 > 0:
            deltaF3 = 1

        self.u  = self.b_value  * self.Bhat * (1 - (self.b1 * deltaF1 + self.b3 * deltaF3))
        
        if self.turn_off == 0:
            if self.q > self.a:     # Water buyer
                trade = (self.a - self.q) * (P + self.t_cost)
            else:                   # Water seller or non-trader
                trade = (self.a - self.q) * P
        
        self.budget = trade 
        self.u += trade

        return self.u

    cdef double update(self):

        self.Bhat = self.e[self.t]  #self.Bhat_alpha * (self.b1 * deltaF1 + self.b3 * deltaF3) + (1- self.Bhat_alpha) * self.Bhat
        self.t += 1
    
    def update_policy(self, W_f):

        self.policy0 = W_f[0]
        self.policy1 = W_f[1]
        self.budget = 0

    cdef double withdraw(self, double S, double s, double I, int M):
        
        cdef double[:] state = self.state_zero  
        cdef double U, V, Z 

        state[0] = S
        state[1] = s
        state[2] = I
        state[3] = self.Bhat

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

        P = np.linspace(0, self.Pmax*1.1, 600)
        Q = np.zeros(600)
        for i in range(600):
            Q[i] = self.consume(P[i], 0, 1)
        print 'Pmax: ' + str(self.Pmax)
        import pylab
        pylab.plot(Q, P)
        pylab.show()

    def set_policy(self, Tilecode policy, double d, int explore):

        self.policy = policy
        self.explore = explore
        self.d = d

    def init_policy(self, Tilecode W_f, Tilecode V_f, Storage storage, Utility utility, linT, CORES, radius, m):

        cdef int i, N = 200000    
        cdef double[:] state = np.zeros(3)
        cdef double[:,:] X = np.zeros([N, 4])
        cdef double[:] w = np.zeros(N)
        cdef double[:] v = np.zeros(N)
        cdef double wp, vp = 0
        cdef double fl = 0

        if m == 0:
            fl = utility.fixed_loss
        else:
            fl = storage.delta_a[1] * 2

        for i in range(N):
            
            X[i, 0] = c_rand() * storage.K
            X[i, 1] = c_rand() * self.Lambda_K * (storage.K - utility.fixed_loss)
            X[i, 2] = c_rand() * storage.Imax  * (storage.I_bar**-1)
            X[i, 3] = c_rand() * 2
            
            state[0] = X[i, 1] * (self.Lambda_I**-1) + utility.fixed_loss
            state[1] = X[i, 2]   
            state[2] = X[i, 3] 
            
            wp = W_f.one_value(state)
            vp = V_f.one_value(state)
            
            w[i] = c_max(c_min((wp - fl) * self.Lambda_I, X[i, 1]), 0)
            v[i] = vp * self.Lambda_I
        
        Twv = int((1 / radius) / 2)
        T = [Twv for t in range(4)]
        L = int(130 / Twv)
        
        if m == 0:
            self.policy0 = Tilecode(4, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
            self.value0 = Tilecode(4, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)

            #Env default is to withdraw no water in summer
            self.policy0.fit(X, np.zeros(N))   
            #self.policy0.fit(X, w)
            self.value0.fit(X, v)
        else:
            self.policy1 = Tilecode(4, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
            self.value1 = Tilecode(4, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
            self.policy1.fit(X, w)
            self.value1.fit(X, v)
