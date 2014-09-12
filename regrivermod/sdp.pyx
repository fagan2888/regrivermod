#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=False, initializedcheck=False

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from time import time
import sys
import pylab

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

@cython.cdivision(True)
cdef inline double c_rand() nogil:
   
    return rand() / (<double> RAND_MAX)

cdef class SDP:

    """
    Stochastic Dynamic Programming (SDP) - policy iteration 

    Solves planners' storage problem by SDP
    
    """
    cdef double beta                # Discount rate
    
    cdef public Tilecode U_f       # Continuous payoff function U = f(Q, I)
    cdef public Tilecode V_f       # Continuous value function V = f(S, I)
    cdef public Tilecode W_f       # Continuous value function V = f(S, I)

    cdef double K                   # Storage capacity
    cdef double delta0              # Storage loss parameters
    cdef double alpha
    cdef double delta1a             # Delivery loss parameters
    cdef double delta1b

    cdef double[:] I_p              # Inflow distribution grid
    cdef double[:,:] I_pr           # Inflow probabilities
    cdef public int[:] I_i          # Inflow grid index
    cdef double I_bar_1             # /bar I^-1

    cdef public double[:,:] state   # State space points S * I
    cdef public double[:] S         # Storage points
    cdef public double[:] I         # Inflow points
    cdef int N                      # Number of state points S * I
    cdef int N_p                    # Number of next state grid points S * I 
    cdef int points                 # Points per state variable
    cdef int GRID                   # Policy variable (W) grid points
    
    cdef double CV                  # Continuation value 
    cdef double[:] V                # Value function points (over state space)
    cdef double[:] W_opt            # Optimal policy points (over state space)

    cdef public double[:,:] W 
    cdef public double[:,:,:] UX  
    cdef public double[:,:,:] state_p_eval   
    cdef public double[:,:,:] statetemp   
    cdef public double[:,:] Xi
    cdef public double[:,:] UX_eval   
    cdef public double[:,:] UXtemp
    
    cdef double GRID_1              # GRID^-1
   
    cdef double[:] Q_0
    cdef double[:] U_0
    cdef double[:] loss_0
    cdef double[:,:] UX_0 
    cdef double[:,:,:] state_p_0
    cdef double[:,:] V_p_0
    
    cdef double solvetime

    cdef int ITERNUM

    def __init__(self, points, users, storage, para):
      
        self.beta = para.beta
        self.points = points

        ###     Build state grid --- S x I

        storage.discrete_I_pdf(points)
        self.I_pr = storage.I_pr
        self.I_p = storage.I_grid[0,:]
        self.I_bar_1 = 1 / storage.I_bar 

        S_grid = np.linspace(0, 1, points) ** 1.2
        S_grid = S_grid * storage.K
        I_grid = np.array(storage.I_grid[0,:]) / storage.I_bar

        [Si, Ii] = np.mgrid[0:points, 0:points]
        S = S_grid[Si.flatten()]
        I = I_grid[Ii.flatten()]
        
        self.I_i = np.asarray(Ii.flatten(), dtype='int32')
        
        self.N = len(S)
        self.N_p = len(self.I_p)
        
        self.V_f = Tilecode(2, [int(points/1.6), int(points/1.6)], 20, offset='optimal', lin_spline=True, linT=6, cores=para.CPU_CORES)
        self.V_f.fit(np.array([S, I]).T, np.zeros(self.N), unsupervised=True, copy=True)
        
        self.U_f = users.SW_f
        self.U_f.extrap = np.zeros(self.N, dtype='int32')

        self.GRID = 350
        
        self.state = np.array([S, I]).T
        self.S = S
        self.I = I
        self.W_opt = np.ones(self.N)
        self.V = np.ones(self.N)
        
        self.K = storage.K                      
        self.delta0 = storage.delta0            
        self.alpha = storage.alpha
        self.delta1a = storage.delta1a
        self.delta1b = storage.delta1b

        self.GRID_1 = 1.0 / (self.GRID - 1)
    
        self.W = np.zeros([self.N, self.GRID])
        self.UX = np.zeros([self.N, self.GRID, 2]) 
        self.UX_eval = np.zeros([self.N, 2]) 
        self.UXtemp = np.zeros([self.N, 2]) 
        self.state_p_eval = np.zeros([self.N, self.N_p, 2])  
        self.statetemp = np.zeros([self.N, self.N_p, 2])  
        self.Xi = np.zeros([self.N_p, 2])  
   
        self.Q_0 = np.zeros(self.N)
        self.U_0 = np.zeros(self.N)
        self.loss_0 = np.zeros(self.N)
        self.UX_0 = np.zeros([self.N, 2]) 
        self.state_p_0 = np.zeros([self.N, self.N_p, 2])  
        self.V_p_0 = np.zeros([self.N, self.N_p])
        
        self.solvetime = 0

    def policy_iteration(self, double TOL_PI, int ITER_PE, plot=False):
        
        cdef double ERROR_PI = 10000
        cdef int ITER_PI = 0
        cdef int i, j
        cdef double[:] Vold = np.zeros(self.N)
        cdef double[:] V = np.zeros(self.N)
        cdef double[:] W_opt = np.zeros(self.N)
        cdef double Verror
        cdef double Verror_max
        cdef double ERROR_PI_P = 100

        tic = time()

        while ERROR_PI_P > TOL_PI and ITER_PI < 10:

            # Policy improvement step
            self.bellman_max_grid(self.S)
            W_opt[...] = self.W_opt
            V[...] = self.V
            
            # Value norm
            ERROR_PI = -100
            for i in range(self.N):
                Verror = abs(V[i] - Vold[i])
                if Verror > ERROR_PI:
                    ERROR_PI = Verror
                    if Vold[i] > 0:
                        ERROR_PI_P = Verror / Vold[i]
            
            #Update value function
            self.V_f.partial_fit(V, 1)

            # Policy Evaluation step
            self.bellman_eval(ITER_PE, TOL_PI)

            V[...] = self.V
            Vold[...] = V
            
            ITER_PI += 1

            print 'PI Iteration: ' + str(ITER_PI) + ', Error: ' + str(round(ERROR_PI_P,4)) + ', PE Iterations: ' + str(self.ITERNUM)


        # Fit policy functions W
        self.W_f = Tilecode(2, [int(self.points/1.8), int(self.points/1.8)], 20, offset='optimal', lin_spline=True, linT=6)
        self.W_f.fit(np.array([self.S, self.I]).T, W_opt, sgd=True, eta=0.4, n_iters=2, scale=0)

        # Plot results
        if plot:
            pylab.figure()
            pylab.clf()
            pylab.title('Planners value function')
            self.V_f.plot(['x', 1], showdata=True)
            pylab.show()

            pylab.figure()
            pylab.clf()
            pylab.title('Planners policy function')
            self.W_f.plot(['x', 1], showdata=True)
            pylab.ylim(0,self.K)
            pylab.xlim(0,self.K)
            pylab.show()

    cdef bellman_eval(self, int ITER_PE, double TOL):
        
        cdef int i, j
        cdef double[:] Vold = np.zeros(self.N)
        cdef double[:] U = np.zeros(self.N)
        cdef double[:] Utemp = np.zeros(self.N)
        cdef double[:,:] V_p = np.zeros([self.N, self.N_p])
        cdef double[:,:] V_ptemp = np.zeros([self.N, self.N_p])
        cdef double CV = 0
        cdef double bellman = 0
        cdef int num_iter = 0
        cdef double Verror = 0
        cdef double ERROR_PE = 10000
        cdef int ITER = 0
         
        while ERROR_PE > TOL and ITER < ITER_PE:

            Vold[...] = self.V

            self.state_transition(self.W_opt)

            #Values
            V_p = self.V_f.matrix_values(self.state_p_eval, self.N, self.N_p, V_ptemp, self.statetemp, self.Xi)
            
            #Payoffs
            U = self.U_f.N_values(self.UX_eval, self.N, Utemp, self.UXtemp)
             
            #Bellman operator
            for i in range(self.N):
                CV = 0
                for k in range(self.N_p):
                    CV += V_p[i, k] * self.I_pr[self.I_i[i], k]

                self.V[i] = U[i] + self.beta * CV

            self.V_f.partial_fit(self.V, 1)

            ERROR_PE = 1
            for i in range(self.N):
                Verror = abs(self.V[i] - Vold[i])
                if Verror > ERROR_PE:
                    if Vold[i] > 0:
                        ERROR_PE = Verror / Vold[i]

            ITER += 1

        self.ITERNUM = ITER

    cdef bellman_max_grid(self, double[:] S):
        
        cdef int i, j
        cdef double[:] U = np.zeros(self.N)
        cdef double[:] Utemp = np.zeros(self.N)
        cdef double[:] V_temp = np.zeros(self.N) * (-100000)
        cdef double[:] W_temp = np.zeros(self.N)
        cdef double[:,:] V_p = np.zeros([self.N, self.N_p])
        cdef double[:,:] V_ptemp = np.zeros([self.N, self.N_p])
        cdef double CV = 0
        cdef double bellman = 0
        cdef int num_iter = 0
        cdef double step = 0.002
        cdef int it = 0
        cdef int count = 0

        W_temp[...] = S

        for j in range(self.GRID):
            for i in range(self.N):
                W_temp[i] = j * S[i] * self.GRID_1
            
            self.state_transition(W_temp)
            
            #Values
            V_p = self.V_f.matrix_values(self.state_p_eval, self.N, self.N_p, V_ptemp, self.statetemp, self.Xi)
            
            #Payoffs
            U = self.U_f.N_values(self.UX_eval, self.N, Utemp, self.UXtemp)
            if j == (self.GRID - 1):
                idx = (np.array(self.I) == 0)
            
            #Bellman operator
            for i in range(self.N):
                CV = 0
                for k in range(self.N_p):
                    CV += V_p[i, k] * self.I_pr[self.I_i[i], k]

                bellman = U[i] + self.beta * CV
                if bellman >= V_temp[i]:
                    V_temp[i] = bellman
                    self.W_opt[i] = W_temp[i]

        self.V[...] = V_temp
    
    cdef bellman_max_trisection(self, double[:] S):
        
        cdef int i, j
        cdef double[:] U = np.zeros(self.N)
        cdef double[:] Utemp = np.zeros(self.N)
        cdef double[:] V_temp = np.zeros(self.N) * (-100000)
        cdef double[:] W_temp = np.zeros(self.N)
        cdef double[:,:] V_p = np.zeros([self.N, self.N_p])
        cdef double[:,:] V_ptemp = np.zeros([self.N, self.N_p])
        cdef double CV = 0
        cdef double bellman = 0
        cdef int num_iter = 0
        cdef double step = 0.002
        cdef int it = 0
        cdef int count = 0
        cdef double[:] l = np.zeros(self.N)
        cdef double[:] r = np.ones(self.N)
        cdef double[:] W_low = np.ones(self.N)
        cdef double[:] W_high = np.ones(self.N)
        cdef int itr

        r[...] = S


        for itr in range(20):
            for j in range(2):
                for i in range(self.N):
                    if j == 0:
                        W_low[i] = l[i] + (1 / 3.0) * (r[i] - l[i]) 
                        if S[i] < self.delta1a:
                            W_low[i] = 0
                        W_temp[i] = W_low[i]
                    else:
                        W_high[i] = l[i] + (2 / 3.0) * (r[i] - l[i])
                        if S[i] < self.delta1a:
                            W_high[i] = 0
                        W_temp[i] = W_high[i]
                
                self.state_transition(W_temp)

                #Values
                V_p = self.V_f.matrix_values(self.state_p_eval, self.N, self.N_p, V_ptemp, self.statetemp, self.Xi)
                
                #Payoffs
                U = self.U_f.N_values(self.UX_eval, self.N, Utemp, self.UXtemp)

                #Bellman operator
                for i in range(self.N):
                    CV = 0
                    for k in range(self.N_p):
                        CV += V_p[i, k] * self.I_pr[self.I_i[i], k]

                    bellman = U[i] + self.beta * CV
                    
                    if j == 0:
                        V_temp[i] = bellman
                        self.W_opt[i] = W_temp[i]
                    else:
                        if bellman > V_temp[i]:
                            l[i] = W_low[i] 
                            r[i] = S[i]
                            V_temp[i] = bellman
                            self.W_opt[i] = W_temp[i]
                        else:
                            l[i] = l[i]
                            r[i] = W_high[i]

        self.V[...] = V_temp

    cdef inline void state_transition(self, double[:] W):
        
        cdef int i, j
        cdef double[:] Q = self.Q_0
        cdef double[:] U = self.U_0
        cdef double[:] loss = self.loss_0
        cdef double[:,:] UX = self.UX_0
        cdef double[:,:,:] state_p = self.state_p_0
        cdef double[:,:] V_p = self.V_p_0
        
        for i in range(self.N):
            
            #Water use
            Q[i] = c_max(W[i] * (1 - self.delta1b) - self.delta1a, 0)
            UX[i, 0] = Q[i]
            UX[i, 1] = self.I[i]
        
            #Storage transition
            loss[i] = self.delta0 * self.alpha * (self.S[i])**(2/3)
            for k in range(self.N_p):
                state_p[i, k, 0] = c_max(c_min(self.S[i] - W[i] - loss[i] + self.I_p[k], self.K), 0)
                state_p[i, k, 1] = self.I_p[k] * self.I_bar_1

        self.state_p_eval[...] = state_p
        self.UX_eval[...] = UX

