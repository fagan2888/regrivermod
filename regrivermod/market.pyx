#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=False, initializedcheck=False

from __future__ import division
import numpy as np
import pylab 
import time
import math
import random

cimport cython
from econlearn.tilecode cimport Tilecode, Function_Group

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

@cython.cdivision(True)
cdef inline double c_rand() nogil:
   
    return rand() / (<double> RAND_MAX)

cdef inline double c_sum(int N, double[:] x):
    
    cdef double sumx = 0
    cdef int i = 0

    for i in range(N):
        sumx += x[i]

    return sumx

cdef inline double excess_demand(double P, double Q, double t_cost, int N, double[:] p, double[:] d_c, double[:] d_b, double[:] a, double p_e, double d_c_e, double d_b_e, double min_q, double a_e) nogil:
        
        "Calculate excess demand given P and Q"

        cdef double Qt = 0
        cdef int i = 0
        cdef double q = 0
        
        # Irrigation users

        for i in range(N):
            
            if p[i] > (P + t_cost):
                Qt += c_max(d_c[i] + d_b[i] * (P + t_cost), 0)
            elif p[i] < P:
                Qt += c_max(d_c[i] + d_b[i] * P, 0)
            else:
                Qt += c_max(c_min(a[i], d_c[i]), 0)
       
        # Environment

        if p_e > (P + t_cost):
            q = c_max(d_c_e + d_b_e * (P + t_cost), 0)
        elif p_e < P:
            q = c_max(d_c_e + d_b_e * P, 0)
        else:
            q = c_max(c_min(a_e, d_c_e), 0)
        
        if q < min_q:
            q = 0
        
        Qt += q
        Qt = Qt - Q
        
        return Qt 

cdef class Market:

    """
    Spot market class, finds the clearing price for a market with N linear demands,
    exogenous supply and a transfer cost
    """
    
    def __init__(self, para, Users users):

        self.N = para.N     # Consumptive users
        self.d_beta = np.zeros(self.N)                   
        self.d_cons = np.zeros(self.N)
        self.p = np.zeros([self.N])                                
        self.a = np.zeros([self.N]) 
        self.Pmax = users.theta[self.N-1,1] * 2                      
        self.t_cost = para.t_cost		                           
        
                            #Environment
        self.min_q = 0
        self.d_b_e = 0
        self.d_c_e = 0
        self.p_e = 0
        self.a_e = 0

        self.twozeros = np.zeros(2)

    cdef void open_market(self, Users users, Environment env):
        
        cdef int i

        for i in range(self.N):
            self.p[i] = users.MV[i]
            self.d_beta[i] = users.d_beta[i]
            self.d_cons[i] = users.d_cons[i]
            self.a[i] = users.a[i]

        self.p_e = env.p
        self.d_b_e = env.d_b
        self.d_c_e = env.d_c
        self.min_q = env.min_q
        self.a_e = env.a
    
    cpdef estimate_market_demand(self, Storage storage, Users users, Environment env, Utility utility, para):

        cdef int i, j, points
        cdef double[:] uw = np.zeros(users.N)
        cdef double W, w, maxE
        cdef double[:] P, Q, I, Qenv, Qlow, Qhigh

        ##################      Estimate market demand curve    ###############
    
        GRID = 40
        points = GRID * GRID
        maxQ = para.K * (1 - storage.delta_Eb) - utility.fixed_loss
        maxI = 4
        Q_grid = np.linspace(0, maxQ, GRID)
        I_grid = np.linspace(0, maxI, GRID)
        [Qi, Ii] = np.mgrid[0:GRID, 0:GRID]
        Q = Q_grid[Qi.flatten()]
        Qenv = Q_grid[Qi.flatten()]
        Qlow = Q_grid[Qi.flatten()]
        Qhigh = Q_grid[Qi.flatten()]
        I = I_grid[Ii.flatten()]
        P = np.zeros(points)
        cdef qtemp, itemp
        qtemp = 100000
        itemp = 1.5

        w = qtemp *  ((<double> utility.N)**-1)
        for j in range(users.N):
            uw[j] = w
        W = utility.deliver_ch7(uw, w, storage, 0)
        storage.I = itemp * (1 - storage.omega_mu) * storage.I_bar
        storage.natural_flows(W, utility.max_E, 0)
        users.allocate(utility.a, itemp)
        env.allocate(utility.a[utility.I_env], storage.min_F2, storage.F3_tilde)

        print 'b3: ' + str(env.b3)
        print 'p: ' + str(env.p)
        print 'a: ' + str(env.a)
        print 'd_c: ' + str(env.d_c)
        print 'd_b: ' + str(env.d_b)
        print 'min_F2: ' + str(storage.min_F2)
        print 'F3_tilde: ' + str(storage.F3_tilde)

        env.plot_demand()
        """
        for i in range(points):
            w = Q[i] *  ((<double> utility.N)**-1)
            for j in range(users.N):
                uw[j] = w
            W = utility.deliver_ch7(uw, w, storage, 0)
            storage.I = I[i] * (1 - storage.omega_mu) * storage.I_bar
            storage.natural_flows(W, utility.max_E, 0)
            users.allocate(utility.a, I[i]) 
            env.allocate(utility.a[utility.I_env], storage.min_F2, storage.F3_tilde)
            self.open_market(users, env)
            P[i] = self.solve_price(utility.A, I[i], 1)
            users.consume(P[i], I[i], 0)
            env.consume(P[i])
            Qlow[i] = c_sum(users.N_low, users.q)
            Qhigh[i] = c_sum(users.N_high, users.q[users.N_low::])
            Qenv[i] = env.q

            if P[i] > 50000:
                print Q[i]
                print I[i]
                print P[i]
                print '-----'
                print Qlow[i]
                print Qhigh[i]
                print Qenv[i]
        
        self.market_d = Tilecode(2, [23, 23], 30, offset='optimal', lin_spline=True, linT=6, cores=para.CPU_CORES)
        self.market_d.fit(np.array([Q, I]).T, P)

        d_low = Tilecode(2, [23, 23], 30, offset='optimal', lin_spline=True, linT=6, cores=para.CPU_CORES)
        d_high = Tilecode(2, [23, 23], 30, offset='optimal', lin_spline=True, linT=6, cores=para.CPU_CORES)
        d_env = Tilecode(2, [23, 23], 30, offset='optimal', lin_spline=True, linT=6, cores=para.CPU_CORES)
        d_low.fit(np.array([Qlow, I]).T, P)
        d_high.fit(np.array([Qhigh, I]).T, P)
        d_env.fit(np.array([Qenv, I]).T, P)

        pylab.figure()
        pylab.clf()
        pylab.title('Effective market demand curve')
        self.market_d.plot(['x', 1], showdata=True)
        pylab.show()

        pylab.figure()
        pylab.clf()
        pylab.title('Low user demand')
        d_low.plot(['x', 1], showdata=True)
        pylab.show()

        pylab.figure()
        pylab.clf()
        pylab.title('High user demand')
        d_high.plot(['x', 1], showdata=True)
        pylab.show()

        pylab.figure()
        pylab.clf()
        pylab.title('Environmental demand')
        d_env.plot(['x', 1], showdata=True)
        pylab.show()
        """
        """
        ##################      Estimate perfect market demand curve    ###############

        P_perf = np.zeros(points)

        for i in range(points):
            atemp = Q[i] / self.N
            users.a = np.ones(self.N) * atemp
            users.demand_para(I[i])
            users.mv(I[i])
            P_perf[i] = self.solve_price(Q[i], para.price)

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
        """

    cdef double solve_price(self, double Q, double I, int init):
        
        """
        Solve for exact market clearing price given W

        """
        cdef double[:] state = self.twozeros  
        cdef double P_guess 
        
        state[0] = Q
        state[1] = I
        if init == 1:
            P_guess = 10
        else:
            P_guess = self.market_d.one_value(state)
        
        cdef double P1 = P_guess
        cdef double tol = 0.01
        cdef double tol2 = 0.005 * Q
        cdef double P2 = P1*0.9
        cdef double P0 = 0

        cdef double EX2 = excess_demand(P2, Q, self.t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e)
        cdef double EX1 = excess_demand(P1, Q, self.t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e)
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
            EX0 = excess_demand(P0, Q, self.t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e)

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
                    P1 = self.Pmax
                else:
                    P0 = 0
                    P1 = P0

            if P0 == P1:# or Q <= tol:
                P0 = 0
                P1 = self.Pmax
            
            EX2 = 2 * tol2

            while c_abs(EX2) > tol2 and iters < 100:
                
                P2 = (P1 + P0)* 0.5
                EX2 = excess_demand(P2, Q, self.t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e)
                if EX2 > 0:
                    P0 = P2
                    P1 = P1
                else:
                    P0 = P0
                    P1 = P2

                if P0 == P1:# or Q <= tol:
                    P0 = 0
                    P1 = self.Pmax*(1+c_rand()*0.1)
                
                iters += 1
            
            if c_abs(EX2) > tol2 and P0 > 0 and Q > tol and P0 > 5000:
                print '   Warning: Clearing price not found   '
                print 'Excess demand: ' + str(EX2)
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
    
