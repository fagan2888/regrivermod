#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from __future__ import division
import numpy as np
import pylab 
import time
import math
import random

cimport numpy as np
cimport cython
from econlearn.tilecode cimport Tilecode, Function_Group
from econlearn.samplegrid import buildgrid
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

cdef inline double c_sum(int N, double[:] x):
    
    cdef double sumx = 0
    cdef int i = 0

    for i in range(N):
        sumx += x[i]

    return sumx

cdef inline double[:] demand(double P, double t_cost, int N, double[:] mv, double[:] d_c, double[:] d_b, double[:] a, double[:] q):
    "Calculate all user demands"

    cdef int i

    for i in range(N):
        
        if mv[i] > (P + t_cost):
            q[i] = c_max(d_c[i] + d_b[i] * (P + t_cost), 0)
        elif mv[i] < P:
            q[i] = c_max(d_c[i] + d_b[i] * P, 0)
        else:
            q[i] = c_max(c_min(a[i], d_c[i]), 0)
    
    return q

cdef inline double excess_demand(double P, double Qbar, double t_cost, int N, double[:] mv, double[:] d_c, double[:] d_b, double[:] a):
        "Calculate excess demand given P and W"

        cdef double Qtemp = 0
        cdef int i = 0
        
        for i in range(N):
            
            if mv[i] > (P + t_cost):
                Qtemp += c_max(d_c[i] + d_b[i] * (P + t_cost), 0)
            elif mv[i] < P:
                Qtemp += c_max(d_c[i] + d_b[i] * P, 0)
            else:
                Qtemp += c_min(a[i], d_c[i])
        
        Qtemp = Qtemp - Qbar

        return Qtemp 

cdef double[:] payoff(int N, double P, double t_cost, double[:] q, double[:]  a, double[:] e, double I, double[:,:] theta, double[:] L, double[:] pi, double[:] risk, int utility):
    "Calculate all user payoffs"

    cdef int i
    cdef double qmax = 0

    for i in range(N):
        if q[i] > a[i]:    # Water buyer
            pi[i] = L[i] * e[i] * ( theta[i, 0] + theta[i, 3] * I + theta[i, 4] * I**2 + (theta[i, 1] + theta[i, 5] * I) * (q[i] / L[i]) + theta[i, 2] * (q[i] / L[i])**2)   + (a[i] - q[i]) * (P + t_cost)
        else:              # Water seller or non-trader
            pi[i] =  L[i] * e[i] * (  theta[i, 0] + theta[i, 3] * I + theta[i, 4] * I**2 + (theta[i, 1] + theta[i, 5] * I) * (q[i] / L[i]) + theta[i, 2] * (q[i] / L[i])**2 )  + (a[i] - q[i]) * P
    
    if utility == 1:
        for i in range(N):
            pi[i] = 1 - c_exp(-risk[i] * pi[i])

    return pi

cdef class Users:

    """
    Consumptive water users class
    includes all consumptive users (high and low reliability)
    """
    
    def __init__(self, para):

        cdef int i, j

        #--------------------------------------#
        # User index
        #--------------------------------------#
        self.N = para.N                               
        self.N_low = para.N_low
        self.N_high = para.N_high
        
        self.I_low = np.zeros(self.N_low, dtype='int32')
        self.I_high = np.zeros(self.N_high, dtype='int32')
        
        for i in range(self.N_low):
            self.I_low[i] = i

        for i in range(self.N_high):
            self.I_high[i]  = i + self.N_low

        #--------------------------------------#
        # User variables
        #--------------------------------------#

        self.w = np.zeros(self.N)                   # User withdrawal
        self.w_scaled = np.zeros(self.N)            # User withdrawal - scaled to [0,1] over search range
        self.a = np.zeros(self.N)                   # User allocation (withdrawal less delivery losses)
        self.q = np.zeros(self.N)                   # User water consumption
        self.profit = np.zeros(self.N)              # User payoff
        self.e = np.ones(self.N)                    # Productivity shocks
        self.trade = np.zeros(self.N)               # a - q
                
        #--------------------------------------#
        # Delivery loss deductions
        #--------------------------------------#
        self.delta = para.delta1b                       
        self.delta1a = para.delta1a                       

        #--------------------------------------#
        # Inflow and capacity shares
        #--------------------------------------#
        low_c = 1 - para.Lambda_high 
        
        self.c_F_low = low_c / <double> self.N_low                  # Low reliability inflow shares
        self.c_K_low = low_c / <double> self.N_low                  # Low reliability capacity shares
        self.c_F_high = (1 - low_c) / <double> self.N_high          # High reliability inflow shares
        self.c_K_high = (1 - low_c) / <double> self.N_high          # High reliability capacity shares
        self.K = para.K
        self.c_F = np.zeros(self.N)
        self.c_K = np.zeros(self.N)
        
        for i in range(0, self.N_low):
            self.c_F[i] = self.c_F_low                                    # Inflow shares
            self.c_K[i] = self.c_K_low                                    # Capacity shares

        for i in range(self.N_low, self.N):
            self.c_F[i] = self.c_F_high
            self.c_K[i] = self.c_K_high

        #--------------------------------------#
        # Yield function 
        #--------------------------------------#
        m = len(para.theta)        
        self.theta = np.zeros([self.N, m])
        for i in range(0, self.N_low):
            for j in range(m):
                self.theta[i,j] = para.theta[j,0]
        for i in range(self.N_low, self.N):
            for j in range(m):
                self.theta[i,j] = para.theta[j,1]

        #--------------------------------------#
        # Risk aversion
        #--------------------------------------#
        self.risk = np.zeros(self.N)
        for i in range(0, self.N_low):
            self.risk[i] = para.risk_aversion_low
        for i in range(self.N_low, self.N):
            self.risk[i] = para.risk_aversion_high
        self.utility = para.utility

        #--------------------------------------#
        # Productivity shocks  
        #--------------------------------------#
        self.rho_eps = para.rho_eps
        self.sig_eta = para.sig_eta

        #--------------------------------------#
        # Land  
        #--------------------------------------#
        self.L = np.ones(self.N)
        for i in range(0, self.N_low):
            self.L[i] = para.L 
        
        for i in range(self.N_low, self.N):
            self.L[i] = para.L * para.high_size
        
        #--------------------------------------#
        # Demand function D(p) = q  
        #--------------------------------------#
        
        self.d_beta = np.zeros(self.N)                                           
        self.d_cons = np.zeros(self.N)
        self.MV = np.zeros([self.N])                                # Marginal value of water
        self.Pmax = self.theta[self.N-1,1] * 2                      # Maximum possible market price
        self.t_cost = para.t_cost		                            # Transaction cost per unit water
        self.demand_para(I = 1.0)
        self.mv(1)

        #--------------------------------------#
        # Exploration parameters  
        #--------------------------------------#

        self.exploring = 1
        self.N_e = 5                                 # 5 explorers per class
        self.I_e_l = np.random.choice(np.array(self.I_low), size=self.N_e, replace=False)
        self.I_e_h = np.random.choice(np.array(self.I_high), size=self.N_e, replace=False)

        self.share_e_l = np.zeros(2, dtype='int32')
        self.share_e_l[0] = 5
        self.share_e_l[1] = 6
        self.share_e_h = np.zeros(2, dtype='int32')
        self.share_e_h[0] = self.N_low + 5 
        self.share_e_h[1] = self.N_low + 6 
        self.share_adj = 0

        self.I_e = np.hstack([self.I_e_l, self.I_e_h])
        self.testing = 0
        self.test_explore = 0
        self.test_idx = 0

        self.c_pi = math.pi
        self.two_zeros = np.zeros(2)                            # Aggregate state [S, I]
        self.N_zeros = np.zeros(self.N)                            
        self.N_ints = np.zeros(self.N, dtype='int32')                            
        self.state_planner_zero = np.zeros([self.N, 2])         # Aggregate state [S, I]
        self.state_zero = np.zeros([self.N, 4])                 # N User states [S, s, e, I]
        self.state_single_zero = np.zeros(4)                    # Single user state [S, s, e, I]

        ##################      Estimate market demand curve    ###############

        GRID = 40
        points = GRID * GRID

        maxQ = para.K * (1 - para.delta1b) - para.delta1a
        maxI = (para.K * 2) / para.I_bar
        Q_grid = np.linspace(0, maxQ, GRID)
        I_grid = np.linspace(0, maxI, GRID)

        [Qi, Ii] = np.mgrid[0:GRID, 0:GRID]

        Q = Q_grid[Qi.flatten()]
        I = I_grid[Ii.flatten()]

        P = np.zeros(points)

        for i in range(points):
            atemp = Q[i] / self.N
            self.a = np.ones(self.N) * atemp
            self.demand_para(I[i])
            self.mv(I[i])
            P[i] = self.solve_price(Q[i], para.price, self.Pmax, self.t_cost)

        self.market_d = Tilecode(2, [23, 23], 30, offset='optimal', lin_spline=True, linT=6, cores=para.CPU_CORES)
        self.market_d.fit(np.array([Q, I]).T, P)

        #pylab.figure()
        #pylab.clf()
        #pylab.title('Effective market demand curve')
        #self.market_d.plot(['x', 1], showdata=True)
        #pylab.show()

        ##################      Estimate perfect market demand curve    ###############

        P_perf = np.zeros(points)

        for i in range(points):
            atemp = Q[i] / <double> self.N
            self.a = np.ones(self.N) * atemp
            self.demand_para(I[i])
            self.mv(I[i])
            P_perf[i] = self.solve_price(Q[i], para.price, self.Pmax, 0)

        self.perf_market = Tilecode(2, [23, 23], 20, offset='optimal', lin_spline=True, linT=6, cores=para.CPU_CORES)
        self.perf_market.fit(np.array([Q, I]).T, P_perf)

        #import pickle
        #home = '/home/nealbob'
        #folder = '/Dropbox/Model/Results/chapter5/'
        #f = open(home + folder + 'Demand.pkl', 'wb')
        #pickle.dump([Q, I, P_perf], f)
        #f.close()

        #pylab.figure()
        #pylab.clf()
        #pylab.title('Perfect market demand curve')
        #self.perf_market.plot(['x', 1.0], showdata = True)
        #pylab.show()

        ###################      Construct the planners payoff function     ##################

        # Build grids 
        Q, I, SW = self.build_SW(para, self.perf_market, 65)
        points = len(Q)

        # Now estimate continuous approximation
        self.SW_f = Tilecode(2, [35, 35], 20, offset='optimal', lin_spline=True, linT=2, cores=para.CPU_CORES)
        self.SW_f.fit(np.array([Q, I]).T, SW, sgd=True, eta=0.2, scale=0, n_iters=2)

        # Plot fitted vs actual
        #pylab.figure()
        #pylab.clf()
        #pylab.title('Social welfare function (planners payoff)')
        #self.SW_f.plot(['x', 1], showdata=True)
        #pylab.show()
        
    def set_shares(self, double Lambda_high, unbundled=False, Lambda_K_high=0):


        self.c_F_low = (1 - Lambda_high) / <double> self.N_low                # Low reliability inflow shares
        self.c_K_low = (1 - Lambda_high) / <double> self.N_low                # Low reliability capacity shares
        self.c_F_high = Lambda_high / <double> self.N_high                  # High reliability inflow shares
        self.c_K_high = Lambda_high / <double> self.N_high                  # High reliability capacity shares
        self.c_F = np.zeros(self.N)
        self.c_K = np.zeros(self.N)
        
        if unbundled:
            self.c_K_low = (1- Lambda_K_high) / <double> self.N_low         # Low reliability capacity shares
            self.c_K_high = Lambda_K_high / <double> self.N_high            # High reliability capacity shares


        for i in range(0, self.N_low):
            self.c_F[i] = self.c_F_low
            self.c_K[i] = self.c_K_low

        for i in range(self.N_low, self.N):
            self.c_F[i] = self.c_F_high
            self.c_K[i] = self.c_K_high

        #Inflow share explorers
        #if self.share_explore:
        #    self.c_F[self.share_e_l[0]] += self.share_adj / <double> self.N_low
        #    self.c_F[self.share_e_h[0]] -= self.share_adj / <double> self.N_low
        #    self.c_K[self.share_e_l[0]] += self.share_adj / <double> self.N_low
        #    self.c_K[self.share_e_h[0]] -= self.share_adj / <double> self.N_low


    cdef double[:] withdraw(self, double S, double[:] s, double I):
        "User policy function, returns user withdrawal w, given current state: S, s, e and I"
        
        cdef int i
        cdef double wstar = 0
        cdef double[:] wplanner = self.N_zeros
        cdef int[:] extrap_planner = self.N_ints
        cdef double[:, :] state = self.state_zero
        cdef double[:] state_single = self.state_single_zero
        cdef double[:,:] state_planner = self.state_planner_zero
        cdef double U, V, Z

        for i in range(self.N):
            state[i, 0] = S
            state[i, 1] = s[i]
            state[i, 2] = self.e[i]
            state[i, 3] = I

        # Optimal policy
        self.w = self.policy.get_values(state, self.w)
        
        if self.testing == 1:
            i = self.test_idx    
            if self.test_explore == 1:
                U = c_rand()
                V = c_rand()
                Z = ((-2 * c_log(U))**0.5)*c_cos(2*self.c_pi*V)
                self.w[i] = c_min(c_max(Z * (self.d * s[i]) + self.w[i], 0), s[i])
                #self.w[i] = c_rand() * s[i] 
            else:
                state_single[0] = S
                state_single[1] = s[i]
                state_single[2] = self.e[i]
                state_single[3] = I
                self.w[i] = c_max(c_min(self.w_f.one_value(state_single), s[i]), 0)
        else:                   
            if self.exploring == 1:
                self.explore(s)
        
        return self.w
   
    cdef void user_stats(self, double[:] s, double[:] x):
        "User policy function, returns user withdrawal w, given current state: S, s, e and I"
        
        cdef int i
        
        self.W_low = 0
        self.W_high = 0
        self.S_low = 0
        self.S_high = 0
        self.X_low = 0
        self.X_high = 0
        self.tradeVOL = 0
        self.trade_low = 0
        self.trade_high = 0

        for i in range(0, self.N_low):
            self.W_low += self.w[i] 
        for i in range(self.N_low, self.N):
            self.W_high += self.w[i] 
        
        for i in range(0, self.N_low):
            self.S_low += s[i] 
        for i in range(self.N_low, self.N):
            self.S_high += s[i] 
        
        for i in range(0, self.N_low):
            self.X_low += x[i] 
        for i in range(self.N_low, self.N):
            self.X_high += x[i] 
    
        for i in range(self.N):
            self.tradeVOL += c_abs(self.trade[i])
        
        for i in range(0, self.N_low):
            self.trade_low += self.trade[i] 
        for i in range(self.N_low, self.N):
            self.trade_high += self.trade[i] 
    
    cdef mv(self, double I):

        cdef int i

        for i in range(self.N):
            self.MV[i] = c_max(self.e[i] * (self.theta[i,1] + self.theta[i,5]*I + (2 * self.theta[i,2] * self.a[i] / self.L[i])), 0)


    cdef void explore(self, double[:] s):

        cdef int i
        cdef double delta = self.d
        cdef double w_min = 0
        cdef double w_max = 0
        cdef int l_idx = 0
        cdef int h_idx = 0
        cdef double U, V, Z1, Z2
        
        if delta == 0:
            for i in range(self.N_e):
                
                l_idx = self.I_e_l[i]
                self.w[l_idx] = s[l_idx] * c_rand() 
                
                h_idx = self.I_e_h[i]
                self.w[h_idx] = s[h_idx] * c_rand() 
        else: 
            for i in range(self.N_e):
                U = c_rand()
                V = c_rand()
                Z1 = ((-2 * c_log(U))**0.5)*c_cos(2*self.c_pi*V)
                Z2 = ((-2 * c_log(U))**0.5)*c_sin(2*self.c_pi*V)
                
                l_idx = self.I_e_l[i]
                self.w[l_idx] = c_min(c_max(Z1 * (delta * s[l_idx]) + self.w[l_idx], 0), s[l_idx]) 
                
                h_idx = self.I_e_h[i]
                self.w[h_idx] = c_min(c_max(Z2 * (delta * s[h_idx]) + self.w[h_idx], 0), s[h_idx]) 
    
    cdef double consume(self, double P, double I, int planner):
        "Determine water consumption q, and payoff u"
       
        cdef double SW
        cdef int i = 0
        cdef double t_cost = 0
        cdef double low_gain, high_gain

        if planner == 1:
            t_cost = 0
        else:
            t_cost = self.t_cost

        I = c_max(c_min(I, 2), 0.5)
        
        self.q = demand(P, t_cost, self.N, self.MV, self.d_cons, self.d_beta, self.a, self.q)
        if planner == 1:
            self.a[...] = self.q

        self.profit = payoff(self.N, P, t_cost, self.q, self.a, self.e, I, self.theta, self.L, self.profit, self.risk, self.utility)

        self.U_low = 0
        self.U_high = 0
        for i in range(0, self.N_low):
            self.U_low += self.profit[i] 
        for i in range(self.N_low, self.N):
            self.U_high += self.profit[i] 
        SW = self.U_low + self.U_high 
        for i in range(self.N):
            self.trade[i] = c_max(c_min(self.a[i], self.d_cons[i]), 0) - self.q[i]

        return SW
    
    def set_explorers(self, N_e, d=0, testing=False, test_idx=0):
        
        """Set the number of explorers per user class (maximum of 5)"""

        self.N_e = N_e
        self.d = d
        
        self.I_e_l = np.random.choice(np.array(self.I_low), size=self.N_e, replace=False)
        self.I_e_h = np.random.choice(np.array(self.I_high), size=self.N_e, replace=False)
        
        if testing:
            self.testing = 1
            self.test_explore = 1
            self.test_idx = test_idx
    
    cdef void update(self):
        """
        Update user productivity shock
        """

        cdef double U 
        cdef double V 
        cdef double Z1
        cdef double eps1
        cdef double Z2
        cdef double eps2
        cdef int i = 0

        while i < self.N:
            U = c_rand()
            V = c_rand()
            Z1 = ((-2 * c_log(U))**0.5)*c_cos(2*self.c_pi*V)
            Z2 = ((-2 * c_log(U))**0.5)*c_sin(2*self.c_pi*V)
            eps1 = Z1 * self.sig_eta 
            eps2 = Z2 * self.sig_eta 
            
            self.e[i] = c_max(1 + self.rho_eps * (self.e[i] - 1) + eps1, 0) 
            self.e[i + 1] = c_max(1 + self.rho_eps * (self.e[i + 1] - 1) + eps2, 0) 
            i += 2
    
    cdef demand_para(self, double I = 1.0):
        
        cdef int i

        for i in range(self.N):
            self.d_beta[i] = self.L[i]*((2 * self.theta[i,2]* self.e[i])**-1) 
            self.d_cons[i] = c_max(self.L[i]*(self.theta[i,1] + self.theta[i,5] * I) * ((-2.0 * self.theta[i,2])**-1),0) 

    cdef void allocate(self, double[:] a, double I):

        cdef int i
        
        for i in range(self.N):
            self.a[i] = a[i]

        I = c_max(c_min(I, 2),0.5)
        
        # Update user demand parameters
        self.demand_para(I)
        
        # User marginal value for water pre trade
        self.mv(I)
    
    cdef double clear_market(self, double I, Tilecode market_d, int planner):

        cdef int i
        cdef double Q = c_sum(self.N, self.a)
        cdef double[:] state = self.two_zeros
        cdef double P_guess =  0 
        cdef double P = 0
        cdef double t_cost = 0

        state[0] = Q
        state[1] = I
        P_guess = market_d.one_value(state)
        if planner == 1:
            t_cost = 0
        else:
            t_cost = self.t_cost

        P = self.solve_price(Q, P_guess, self.Pmax, t_cost)
        self.Q = Q

        return c_max(P, 0)

    cdef double solve_price(self, double Q, double P_guess, double Pmax, double t_cost):
        """
        Solve for exact market clearing price given W

        """
        
        cdef double tol = 0.01
        cdef double tol2 = 0.005 * Q
        cdef double P1 = P_guess
        cdef double P2 = P1*0.9
        cdef double P0 = 0
        cdef double EX2 = excess_demand(P2, Q, t_cost, self.N, self.MV, self.d_cons, self.d_beta, self.a)
        cdef double EX1 = excess_demand(P1, Q, t_cost, self.N, self.MV, self.d_cons, self.d_beta, self.a)
        cdef double EX0 = EX1
        cdef int iters = 0

        while 1:                                # Use secant method

            if EX1 == EX2:
                if EX1 < 0:
                    P0 = P1 * 0.9
                elif EX1 > 0:
                    P0 = P1 * (1.1)
            else:
                P0 = c_max(P1 - (EX1 * (P1 - P2) * ((EX1 - EX2)**-1)), 0)
            EX0 = excess_demand(P0, Q, t_cost, self.N, self.MV, self.d_cons, self.d_beta, self.a)

            if P0 == 0 and EX0 <= 0:
                EX0 = 0
                
            P2 = P1
            P1 = P0
            EX2 = EX1
            EX1 = EX0
            iters += 1

            if (c_abs(EX0) < tol and (EX0 + Q) > 0) or iters > 50:
                break


        if c_abs(EX0) > tol2 and Q > tol:         # Use bisection method
            
            iters = 0
            if c_min(EX0, EX2) < 0 and c_max(EX0, EX2) > 0:
                P0 = c_min(P0, P2)
                P1 = c_max(P0, P2)
            else:
                if EX0 > 0:
                    P0 = P0
                    P1 = Pmax
                else:
                    P0 = 0
                    P1 = P0

            if P0 == P1:# or Q <= tol:
                P0 = 0
                P1 = Pmax
            
            EX2 = 2 * tol2

            while c_abs(EX2) > tol2 and iters < 100:
                
                P2 = (P1 + P0)* 0.5
                EX2 = excess_demand(P2, Q, t_cost, self.N, self.MV, self.d_cons, self.d_beta, self.a)
                if EX2 > 0:
                    P0 = P2
                    P1 = P1
                else:
                    P0 = P0
                    P1 = P2

                if P0 == P1:# or Q <= tol:
                    P0 = 0
                    P1 = Pmax*(1+c_rand()*0.1)
                
                iters += 1

            if c_abs(EX2) > tol2 and P0 > 0 and Q > tol:

                print '   Warning: Clearing price not found   '
                print 'Excess demand: ' + str(EX2/Q)
                print 'iters: ' + str(iters)
                print 'Q: ' + str(Q)
                print 'P_guess: ' + str(P_guess)
                print 'P0: ' + str(P0)
                print 'P1: ' + str(P1)
                print 'P2: ' + str(P1)
                print 'EX2: ' + str(EX2)
                print 'EX0: ' + str(EX0)
                raise NameError('SpotMarketFail')

        return P0
    
    def update_policy(self, w_f_low, w_f_high, Np=10, test=False, test_idx=0, explore=True, N_e=5, d=0):
        
        self.N_e = N_e
        self.d = d
        
        if test:
            self.w_f = w_f_low
            self.test_idx = test_idx
            self.testing = 1
            if explore:
                self.test_explore = 1
            else:
                self.test_explore = 0

        if explore:
            self.exploring = 1
        else:
            self.exploring = 0

        if Np == self.N: 
            index_low = self.I_low
            index_high = self.I_high
            self.I_e_l = np.random.choice(np.array(self.I_low), size=self.N_e, replace=False)
            self.I_e_h = np.random.choice(np.array(self.I_high), size=self.N_e, replace=False)
        else:
            index_low = np.random.choice(np.array(self.I_low), size=Np, replace=False)
            self.I_e_l = index_low[0:self.N_e]
            if self.share_explore == 1:
                index_low = np.append(index_low, self.share_e_l)

            index_high = np.random.choice(np.array(self.I_high), size=Np, replace=False)
            self.I_e_h = index_high[0:self.N_e]
            if self.share_explore == 1:
                index_low = np.append(index_high, self.share_e_h)

        self.policy.update(index_low, w_f_low, index_high, w_f_high)

    def set_policy(self, w_f_low, w_f_high):
        
        self.policy = Function_Group(self.N, self.N_low, w_f_low, w_f_high)
    
    def seed(self, i):
        seed = int(time.time()/(i + 1))
        c_seed(seed)
    
    def build_SW(self, para, Tilecode perf_market, GRID=25):

        cdef int points = GRID*GRID

        maxQ = para.K * (1 - para.delta1b) - para.delta1a
        maxI = (para.K * 2) / para.I_bar
        Q_grid = np.linspace(0, maxQ, GRID)
        I_grid = np.linspace(0, maxI, GRID)

        [Qi, Ii] = np.mgrid[0:GRID, 0:GRID]

        cdef double[:] Q = Q_grid[Qi.flatten()]
        cdef double[:] I = I_grid[Ii.flatten()]
        cdef double[:] SW = np.zeros(points)
        cdef double P, P0
        P0 = para.price
        cdef int i, j
        
        self.e = np.ones(self.N)

        for i in range(points):
            for j in range(self.N):
                self.a[j] = Q[i] * (1/ <double> self.N)
            P = self.clear_market(I[i], perf_market,  1)
            self.consume(P, I[i], 1)
            SW[i] += c_sum(self.N, self.profit)

        return [np.array(Q), np.array(I), np.array(SW)]

    def init_policy(self, Tilecode W_f, Tilecode V_f, Storage storage, linT, CORES, radius):
        
        cdef int i, N = 100000    
        cdef double wplanner = 0
        cdef double[:] state = np.zeros(2)
        cdef double s, I, e, S
        cdef double[:,:] X = np.zeros([N, 4])
        cdef double[:] w = np.zeros(N)
        cdef double[:] v = np.zeros(N)
        cdef double wp, vp = 0
        
        for i in range(N):
            
            X[i, 0] = c_rand() * storage.K
            X[i, 1] = c_rand() * self.c_F_low * (storage.K - storage.delta1a)
            X[i, 2] = c_rand() * 2
            X[i, 3] = c_rand() * storage.Imax / storage.I_bar
            
            state[0] = X[i, 1] * (self.c_F_low**-1) + self.delta1a
            state[1] = X[i, 3]    
            
            wp = W_f.one_value(state)
            vp = V_f.one_value(state)
            
            w[i] = c_max(c_min((wp - self.delta1a) * self.c_F_low, X[i, 1]), 0)
            v[i] = vp * self.c_F_low
        
        
        Twv = int((1 / radius) / 2)
        T = [Twv for t in range(4)]
        L = int(130 / Twv)
        w_f_low = Tilecode(4, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
        v_f_low = Tilecode(4, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
        w_f_low.fit(X, w)
        v_f_low.fit(X, v)
        
        for i in range(N):
            
            X[i, 0] = c_rand() * storage.K
            X[i, 1] = c_rand() * self.c_F_high * (storage.K - storage.delta1a)
            X[i, 2] = c_rand() * 2
            X[i, 3] = c_rand() * storage.Imax / storage.I_bar
            
            state[0] = X[i, 1] * (self.c_F_high**-1) + self.delta1a
            state[1] = X[i, 3]    
            
            wp = W_f.one_value(state)
            vp = V_f.one_value(state)
            
            w[i] = c_max(c_min((wp - self.delta1a) * self.c_F_high, X[i, 1]), 0)
            v[i] = vp * self.c_F_high
        
        
        Twv = int((1 / radius) / 2)
        T = [Twv for t in range(4)]
        L = int(130 / Twv)
        w_f_high = Tilecode(4, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
        v_f_high = Tilecode(4, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=CORES)
        w_f_high.fit(X, w)
        v_f_high.fit(X, v)
        
        self.policy = Function_Group(self.N, self.N_low, w_f_low, w_f_high)
        
        return [[w_f_low, w_f_high], [v_f_low,  v_f_high]]

