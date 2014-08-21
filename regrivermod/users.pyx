#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=False, initializedcheck=False

from __future__ import division
import numpy as np
import pylab 
import time
import math
import random
cimport numpy as np
cimport cython
from econlearn.tilecode cimport Tilecode, Function_Group

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "math.h":
    double c_fmax "fmax" (double, double)

cdef extern from "math.h":
    double c_fmin "fmin" (double, double)
    
cdef extern from "math.h":
    double c_log "log" (double)

cdef extern from "math.h":
    double c_exp "exp" (double)

cdef extern from "math.h":
    double c_sin "sin" (double)

cdef extern from "math.h":
    double c_cos "cos" (double)

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

cdef inline double[:] demand(double P, double t_cost, int N, double[:] mv, double[:] d_c, double[:] d_b, double[:] a, double[:] q):
    "Calculate all user demands"

    cdef int i

    for i in range(N):
        
        if mv[i] > (P + t_cost):
            q[i] = c_fmax(d_c[i] + d_b[i] * (P + t_cost), 0)
        elif mv[i] < P:
            q[i] = c_fmax(d_c[i] + d_b[i] * P, 0)
        else:
            q[i] = c_fmax(c_fmin(a[i], d_c[i]), 0)
    
    return q

cdef inline double excess_demand(double P, double Qbar, double t_cost, int N, double[:] mv, double[:] d_c, double[:] d_b, double[:] a):
        "Calculate excess demand given P and W"

        cdef double Qtemp = 0
        cdef int i = 0
        
        for i in range(N):
            
            if mv[i] > (P + t_cost):
                Qtemp += c_fmax(d_c[i] + d_b[i] * (P + t_cost), 0)
            elif mv[i] < P:
                Qtemp += c_fmax(d_c[i] + d_b[i] * P, 0)
            else:
                Qtemp += c_fmin(a[i], d_c[i])
        
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
                
        #--------------------------------------#
        # Delivery loss deductions
        #--------------------------------------#
        self.delta = para.delta1b                       
        self.delta1a = para.delta1a                       

        #--------------------------------------#
        # Inflow and capacity shares
        #--------------------------------------#
        low_c = 1 - para.Lambda_high 
        
        self.c_F_low = low_c / self.N_low                  # Low reliability inflow shares
        self.c_K_low = low_c / self.N_low                  # Low reliability capacity shares
        self.c_F_high = (1 - low_c) / self.N_high          # High reliability inflow shares
        self.c_K_high = (1 - low_c) / self.N_high          # High reliability capacity shares
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
        self.I_e_l = np.zeros(self.N_e, dtype='int32')
        self.I_e_h = np.zeros(self.N_e, dtype='int32')
        self.I_e_l[0] = 0
        self.I_e_l[1] = 1
        self.I_e_l[2] = 2
        self.I_e_l[3] = 3
        self.I_e_l[4] = 4
        self.I_e_h[0] = self.N_low
        self.I_e_h[1] = self.N_low + 1
        self.I_e_h[2] = self.N_low + 2
        self.I_e_h[3] = self.N_low + 3
        self.I_e_h[4] = self.N_low + 4
        self.I_e = np.hstack([self.I_e_l, self.I_e_h])
        self.testing = 0
        self.test_idx = 0

        self.c_pi = math.pi
        self.two_zeros = np.zeros(2)                            # Aggregate state [S, I]
        self.N_zeros = np.zeros(self.N)                            # Aggregate state [S, I]
        self.state_planner_zero = np.zeros([self.N, 2])         # Aggregate state [S, I]
        self.state_zero = np.zeros([self.N, 4])                 # N User states [S, s, e, I]
        self.state_single_zero = np.zeros(4)                    # Single user state [S, s, e, I]
        self.init = 1

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
            atemp = Q[i] / self.N
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

    def set_shares(self, Lambda_high):

        low_c = 1 - Lambda_high

        self.c_F_low = low_c / self.N_low                  # Low reliability inflow shares
        self.c_K_low = low_c / self.N_low                  # Low reliability capacity shares
        self.c_F_high = (1 - low_c) / self.N_high          # High reliability inflow shares
        self.c_K_high = (1 - low_c) / self.N_high          # High reliability capacity shares
        self.c_F = np.zeros(self.N)
        self.c_K = np.zeros(self.N)

        for i in range(0, self.N_low):
            self.c_F[i] = self.c_F_low
            self.c_K[i] = self.c_K_low

        for i in range(self.N_low, self.N):
            self.c_F[i] = self.c_F_high
            self.c_K[i] = self.c_K_high

    cdef double[:] withdraw(self, double S, double[:] s, double I):
        "User policy function, returns user withdrawal w, given current state: S, s, e and I"
        
        cdef int i
        cdef double wstar = 0
        cdef double[:] wplanner = self.N_zeros
        cdef double[:, :] state = self.state_zero
        cdef double[:] state_single = self.state_single_zero
        cdef double[:,:] state_planner = self.state_planner_zero
       
        # Initial user policy functions derived from planners solution    
    
        if self.init == 1:
            for i in range(self.N):
                state_planner[i, 0] = s[i] * (self.c_F[i]**-1) + self.delta1a
                state_planner[i, 1] = I    
            wplanner = self.W_f.N_values_policy(state_planner, self.N, wplanner)
            for i in range(self.N):
                self.w[i] = c_fmax(c_fmin((wplanner[i] - self.delta1a) * self.c_F[i], s[i]), 0)
        
        # Actual user policy functions 

        else:
            for i in range(self.N):
                state[i, 0] = S
                state[i, 1] = s[i]
                state[i, 2] = self.e[i]
                state[i, 3] = I

            # Optimal policy
            self.w = self.policy.get_values(state, self.w)
        
        # Exploration
        self.explore(s)
       
        if self.testing == 1:
            i = self.test_idx    
            state_single[0] = S
            state_single[1] = s[i]
            state_single[2] = self.e[i]
            state_single[3] = I
            self.w[i] = self.w_f.one_value(state_single) 
        
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
    
    
    cdef mv(self, double I):

        cdef int i

        for i in range(self.N):
            self.MV[i] = c_fmax(self.e[i] * (self.theta[i,1] + self.theta[i,5]*I + (2 * self.theta[i,2] * self.a[i] / self.L[i])), 0)


    cdef void explore(self, double[:] s):

        cdef int i
        cdef double delta = self.d
        cdef double w_min = 0
        cdef double w_max = 0
        cdef int l_idx = 0
        cdef int h_idx = 0
        
        if delta == 0 or delta == 1:
            for i in range(self.N_e):
                
                l_idx = self.I_e_l[i]
                self.w[l_idx] = s[l_idx] * c_rand() 
                self.w_scaled[l_idx] = self.w[l_idx]
                
                h_idx = self.I_e_h[i]
                self.w[h_idx] = s[h_idx] * c_rand() 
                self.w_scaled[h_idx] = self.q[h_idx]
        else: 
            for i in range(self.N_e):
                
                l_idx = self.I_e_l[i]
                w_min = c_fmax(self.w[l_idx] - delta * s[l_idx], 0)
                w_max = c_fmin(self.w[l_idx] + delta * s[l_idx], s[l_idx])
                self.w_scaled[l_idx] = c_rand()
                self.w[l_idx] = w_min + (w_max - w_min) * self.w_scaled[l_idx] 
                
                h_idx = self.I_e_h[i]
                w_min = c_fmax(self.w[h_idx] - delta * s[h_idx], 0)
                w_max = c_fmin(self.w[h_idx] + delta * s[h_idx], s[h_idx])
                self.w_scaled[h_idx] = c_rand()
                self.w[h_idx] = w_min + (w_max - w_min) *  self.w_scaled[h_idx]
    
    cdef double consume(self, double P, double I, int planner):
        "Determine water consumption q, and payoff u"
       
        cdef double SW
        cdef int i = 0
        cdef double t_cost = 0

        if planner == 1:
            t_cost = 0
        else:
            t_cost = self.t_cost

        I = c_fmax(c_fmin(I, 2), 0.5)

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

        return SW
    
    def calc_trade(self):

        cdef int i
        self.trade = 0
        for i in range(self.N):
            self.trade = self.trade + abs(self.a[i] - self.q[i])
    
        return self.trade
    
    def set_explorers(self, N_e, d=0):
        
        """Set the number of explorers per user class (maximum of 5)"""

        self.N_e = N_e
        self.d = d
    
    
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
            
            self.e[i] = c_fmax(1 + self.rho_eps * (self.e[i] - 1) + eps1, 0) 
            self.e[i + 1] = c_fmax(1 + self.rho_eps * (self.e[i + 1] - 1) + eps2, 0) 
            i += 2
    
    cdef demand_para(self, double I = 1.0):
        
        cdef int i

        for i in range(self.N):
            self.d_beta[i] = self.L[i]*((2 * self.theta[i,2]* self.e[i])**-1) 
            self.d_cons[i] = c_fmax(self.L[i]*(self.theta[i,1] + self.theta[i,5] * I) * ((-2.0 * self.theta[i,2])**-1),0) 

    cdef double clear_market(self, double I, Tilecode market_d, int planner):

        cdef int i
        cdef double Q = c_sum(self.N, self.a)
        cdef double[:] state = self.two_zeros
        cdef double P_guess =  0 
        cdef double P = 0
        cdef double t_cost = 0

        I = c_fmax(c_fmin(I, 2),0.5)
        
        # Update user demand parameters
        self.demand_para(I)
        
        # User marginal value for water pre trade
        self.mv(I)
        
        state[0] = Q
        state[1] = I
        P_guess = market_d.one_value(state)
        if planner == 1:
            t_cost = 0
        else:
            t_cost = self.t_cost

        P = self.solve_price(Q, P_guess, self.Pmax, t_cost)

        return c_fmax(P, 0)

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
                P0 = c_fmax(P1 - (EX1 * (P1 - P2) * ((EX1 - EX2)**-1)), 0)
            EX0 = excess_demand(P0, Q, t_cost, self.N, self.MV, self.d_cons, self.d_beta, self.a)

            if P0 == 0 and EX0 <= 0:
                EX0 = 0
                
            P2 = P1
            P1 = P0
            EX2 = EX1
            EX1 = EX0
            iters += 1

            if (abs(EX0) < tol and (EX0 + Q) > 0) or iters > 50:
                break


        if abs(EX0) > tol2 and Q > tol:         # Use bisection method
            
            iters = 0
            if c_fmin(EX0, EX2) < 0 and c_fmax(EX0, EX2) > 0:
                P0 = c_fmin(P0, P2)
                P1 = c_fmax(P0, P2)
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

            while abs(EX2) > tol2 and iters < 100:
                
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

            if abs(EX2) > tol2 and P0 > 0 and Q > tol:

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
    
    def update_policy(self, w_f_low, w_f_high, prob = 0.3, init=False, test = False, test_idx = 0):
        
        if init:
            self.policy = Function_Group(self.N, self.N_low, w_f_low, w_f_high)
        if test:
            self.w_f = w_f_low
            self.test_idx = test_idx
            self.testing = 1
        else:
            if prob == 1: 
                index_low = self.I_low
                index_high = self.I_high
            else:
                # Low reliability users
                index_low = np.array(self.I_low)[random.sample(range(1,self.N_low), int(prob * (self.N_low-1)))]
                index_low = np.append(index_low, self.I_e_l)
                
                # High reliability users
                index_high = np.array(self.I_high)[random.sample(range(1,self.N_high), int(prob * (self.N_high-1)))]
                index_high = np.append(index_high, self.I_e_h)

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
                self.a[j] = Q[i] * (1/self.N)
            P = self.clear_market(I[i], perf_market,  1)
            self.consume(P, I[i], 1)
            SW[i] += c_sum(self.N, self.profit)

        return [np.array(Q), np.array(I), np.array(SW)]

