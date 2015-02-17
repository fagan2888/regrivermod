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

cdef inline double excess_demand(double P, double Q, double t_cost, int N, double[:] p, double[:] d_c, double[:] d_b, double[:] a, double p_e, double d_c_e, double d_b_e, double min_q, double a_e, double P_adj) nogil:
        
        "Calculate excess demand given P and Q"

        cdef double Qt = 0
        cdef int i = 0
        cdef double q = 0
        
        # Irrigation users

        for i in range(N):
            
            if p[i] > (P + t_cost):
                Qt += c_max(d_c[i] + d_b[i] * (P + t_cost), 0)
            elif p[i] < (P - t_cost):
                Qt += c_max(d_c[i] + d_b[i] * (P - t_cost), 0)
            else:
                Qt += c_max(c_min(a[i], d_c[i]), 0)
       
        # Environment

        #if Q <= min_q:
        #    q = 0
        #else:
        P = c_max(P + P_adj, 0)
        if p_e > (P + t_cost):
            q = c_max(d_c_e + d_b_e * (P + t_cost), 0)
        elif p_e < (P - t_cost):
            q = c_max(d_c_e + d_b_e * (P - t_cost), 0)
        else:
            q = c_max(c_min(a_e, d_c_e), 0)
        
        #if q <= min_q:
        #    q = 0

        Qt += q
        Qt = Qt - Q
        
        return Qt 

cdef inline double e_demand(double P, double t_cost, double p_e, double d_c_e, double d_b_e, double min_q, double a_e, double P_adj) nogil:


        "Calculate excess demand given P and Q"

        cdef int i = 0
        cdef double q = 0
        
        P = c_max(P + P_adj, 0)
        if p_e > (P + t_cost):
            q = c_max(d_c_e + d_b_e * (P + t_cost), 0)
        elif p_e < (P - t_cost):
            q = c_max(d_c_e + d_b_e * (P - t_cost), 0)
        else:
            q = c_max(c_min(a_e, d_c_e), 0)

        #if q <= min_q:
        #    q = 0

        return q

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
        self.users_Pmax = users.theta[self.N-1,1] * 2                      
        self.Pmax = self.users_Pmax
        self.t_cost = para.t_cost		                           
        self.P_adj = 0
        self.nat = 0

        #Environment
        self.min_q = 0
        self.d_b_e = 0
        self.d_c_e = 0
        self.p_e = 0
        self.a_e = 0

        self.threezeros = np.zeros(3)

    cdef void open_market(self, Users users, Environment env, int M):
        
        cdef int i

        if self.nat == 1:
            for i in range(self.N):
                self.p[i] = 0
                self.d_beta[i] = 0
                self.d_cons[i] = 0
                self.a[i] = 0
        else:
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

        self.Pmax = c_max(env.Pmax, users.Pmax)
        self.ePmax = env.Pmax
        self.M = M

        if M == 1:
            for i in range(self.N):
                self.p[i] = 0
                self.d_beta[i] = 0
                self.d_cons[i] = 0

    def estimate_market_demand(self, double[:] P, double[:] Q, double[:] Qlow, double[:] Qhigh, double[:]  Qenv, double[:] B, double[:] I, para):

        ##################      Estimate market demand curve    ###############
   
        self.market_d = Tilecode(3, [12, 12, 12], 30, offset='optimal', lin_spline=True, linT=6, cores=para.CPU_CORES)
        self.market_d.fit(np.array([Q, I, B]).T, P)

        self.d_low = Tilecode(3, [10, 10, 10], 30, offset='optimal', lin_spline=True, linT=4, cores=para.CPU_CORES)
        self.d_high = Tilecode(3, [10, 10, 10], 30, offset='optimal', lin_spline=True, linT=4, cores=para.CPU_CORES)
        self.d_env = Tilecode(3, [10, 10, 10], 30, offset='optimal', lin_spline=True, linT=4, cores=para.CPU_CORES)
        self.d_low.fit(np.array([Qlow, I, B]).T, P)
        self.d_high.fit(np.array([Qhigh, I, B]).T, P)
        self.d_env.fit(np.array([Qenv, I, B]).T, P)
        
        """
        pylab.figure()
        pylab.clf()
        pylab.title('Effective market demand curve')
        self.market_d.plot(['x', 1, 0.5], showdata=True)
        pylab.show()

        pylab.figure()
        pylab.clf()
        pylab.title('Low user demand')
        self.d_low.plot(['x', 1, 0.5], showdata=True)
        pylab.show()

        pylab.figure()
        pylab.clf()
        pylab.title('High user demand')
        self.d_high.plot(['x', 1, 0.5], showdata=True)
        pylab.show()

        pylab.figure()
        pylab.clf()
        pylab.title('Environmental demand')
        self.d_env.plot(['x', 1, 0.5], showdata=True)
        pylab.show()

        
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

    cdef double solve_price(self, double Q, double I, double Bhat, int init, int plan):
        
        """
        Solve for exact market clearing price given W

        """

        cdef double t_cost = self.t_cost
        if plan == 1:
            t_cost = 0

        if self.M == 1:
            if self.d_b_e < 0:
                return c_max((Q - self.d_c_e)*(self.d_b_e**-1) - t_cost, 0)
            else:
                return 0

        cdef double[:] state = self.threezeros  
        cdef double P_guess 
        
        state[0] = Q
        state[1] = I
        state[2] = Bhat

        if init == 1:
            P_guess = 10
        else:
            P_guess = self.market_d.one_value(state)
        
        
        cdef double qetemp = e_demand(self.ePmax-0.01, 0, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e, self.P_adj)
        cdef double EXtemp = excess_demand(self.ePmax-0.01, Q, 0, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e, self.P_adj)

        if EXtemp > 0:          # Environment exits the market
            'Env has left the building'
            self.d_c_e = 0
            self.d_b_e = 0


        cdef double P1 = P_guess
        cdef double tol = 0.01
        cdef double tol2 = 0.005 * Q
        cdef double P2 = P1*0.9
        cdef double P0 = 0

        cdef double EX2 = excess_demand(P2, Q, t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e, self.P_adj)
        cdef double EX1 = excess_demand(P1, Q, t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e, self.P_adj)
        cdef double EX0 = EX1
        cdef int iters = 0

        while 1:                                # Use secant method

            if EX1 == EX2:
                if EX1 < 0:
                    P0 = P1 * 0.9
                elif EX1 > 0:
                    P0 = P1 * (1.1)
            else:
                P0 = c_min(c_max(P1 - (EX1 * (P1 - P2) * ((EX1 - EX2)**-1)), 0), self.Pmax)
            EX0 = excess_demand(P0, Q, t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e, self.P_adj)

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
            if c_min(EX0, EX2) < 0 < c_max(EX0, EX2):
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
                
                P2 = (P1 + P0)*0.5
                EX2 = excess_demand(P2, Q, t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e, self.P_adj)
                if EX2 > 0:
                    P0 = P2
                else:
                    P1 = P2

                if P0 == P1:# or Q <= tol:
                    P0 = 0
                    P1 = self.Pmax*(1-c_rand()*0.1)
                
                iters += 1
             
            if c_abs(EX2) > tol2 and P0 > 0 and Q > tol:
                """
                print '   Warning: Clearing price not found   '
                
                print 'Excess demand: ' + str(EX2)
                print 'iters: ' + str(iters)
                print 'Q: ' + str(Q)
                print 'P_guess: ' + str(P_guess)
                print 'P0: ' + str(P0)
                #print 'P1: ' + str(P1)
                #print 'P2: ' + str(P1)
                #print 'EX2: ' + str(EX2)
                #print 'EX0: ' + str(EX0)
                print 'Pmax: ' + str(self.Pmax)
                print 'q_e: ' + str(e_demand(P0, t_cost, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e, self.P_adj))
                print 'min q: ' + str(self.min_q)
                print 'd_c_e: ' + str(self.d_c_e)
                print 'd_b_e: ' + str(self.d_b_e)
                print 'ePmax: ' + str(self.ePmax)
                print 'EXtemp: ' + str(EXtemp)
                print 'qetemp: ' + str(qetemp)
                print 'ae: ' + str(self.a_e)
                print 'tcost: ' + str(t_cost)

                #raise NameError('SpotMarketFail')
                """
        self.EX =  excess_demand(P0, Q, t_cost, self.N, self.p, self.d_cons, self.d_beta, self.a, self.p_e, self.d_c_e, self.d_b_e, self.min_q, self.a_e, self.P_adj)
        return P0
    
