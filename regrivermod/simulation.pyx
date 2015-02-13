#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from __future__ import division 
import numpy as np
import pylab
import time
import scipy.optimize as opt
import multiprocessing
import math
from sklearn import ensemble, tree, neighbors
from sklearn import preprocessing as pre
from sklearn import neighbors as near
from multiprocessing.queues import Queue
import errno

cimport cython
from regrivermod.storage cimport Storage
from regrivermod.users cimport Users
from regrivermod.utility cimport Utility
from regrivermod.environment cimport Environment
from regrivermod.market cimport Market
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

cdef inline double c_sum(int N, double[:] x):

    cdef double sumx = 0
    cdef int i = 0

    for i in range(N):
        sumx += x[i]

    return sumx

def retry_on_eintr(function, *args, **kw):
    while True:
        try:
            return function(*args, **kw)
        except IOError, e:            
            if e.errno == errno.EINTR:
                continue
            else:
                raise    

class RetryQueue(Queue):
    """Queue which will retry if interrupted with EINTR."""

    def get(self, block=True, timeout=None):
        return retry_on_eintr(Queue.get, self, block, timeout)

def run_ch7_sim(int job, int T, Users users, Storage storage, Utility utility, Market market, Environment env, int init, int stats, int nat, que, multi=True, planner=False, budgetonly=False):

    users.seed(job)
    utility.seed(job)
    storage.seed(job)
    env.seed(job)

    # Representative user explorers
    cdef int[:] I_e_l = users.I_e_l
    cdef int[:] I_e_h = users.I_e_h
    cdef int N_e_l = users.N_e
    cdef int N_e_h = users.N_e
    cdef int N_e = N_e_l + N_e_h
 
    cdef int bo = 0
    if budgetonly:
        bo = 1

    cdef double[:,:] W_sim = np.zeros([T, 2])
    cdef double[:,:] Q_sim = np.zeros([T, 2])
    cdef double[:,:] Q_low_sim = np.zeros([T, 2])
    cdef double[:,:] Q_high_sim = np.zeros([T, 2])
    cdef double[:,:] Q_env_sim = np.zeros([T, 2])
    cdef double[:,:] A_low_sim = np.zeros([T, 2])
    cdef double[:,:] A_high_sim = np.zeros([T, 2])
    cdef double[:,:] A_env_sim = np.zeros([T, 2])
    cdef double[:,:] S_low_sim = np.zeros([T, 2])
    cdef double[:,:] S_high_sim = np.zeros([T, 2])
    cdef double[:,:] S_env_sim = np.zeros([T, 2])
    cdef double[:,:] U_low_sim = np.zeros([T, 2])
    cdef double[:,:] U_high_sim = np.zeros([T, 2])
    cdef double[:,:] Budget_sim = np.zeros([T, 2])
    cdef double[:] P_adj_sim = np.zeros(T)
    cdef double[:,:] Budget_in = np.zeros([T, 2])
    cdef double[:,:] Budget_out = np.zeros([T, 2])
    cdef double[:,:] Budget_tc = np.zeros([T, 2])
    cdef double[:,:] A_sim = np.zeros([T, 2])
    cdef double[:,:] SW_sim = np.zeros([T, 2])
    cdef double[:,:] S_sim = np.zeros([T, 2])
    cdef double[:,:] I_sim = np.zeros([T, 2])
    cdef double[:,:] Z_sim = np.zeros([T, 2])
    cdef double[:,:] P_sim = np.zeros([T, 2])
    cdef double[:,:] E_sim = np.zeros([T, 2])
    cdef double[:,:] Bhat_sim = np.zeros([T, 2])
    cdef double[:,:] B_sim = np.zeros([T, 2])
    cdef double[:,:] F1_tilde = np.zeros([T, 2])
    cdef double[:,:] F3_tilde = np.zeros([T, 2])
    cdef double[:,:] F1 = np.zeros([T, 2])
    cdef double[:,:] F3 = np.zeros([T, 2])
    cdef double[:,:] Profit_sim = np.zeros([T, 2])

    cdef int plan = 0
    cdef int itrs = 0

    # Planner samples
    cdef double[:,:] XA0
    cdef double[:,:] X10
    cdef double[:] U0

    cdef double[:,:] XA1
    cdef double[:,:] X11
    cdef double[:] U1
    
    # Consumptive user samples
    cdef double[:,:,:] XA_l0, X1_l0, XA_h0, X1_h0, XA_l1, X1_l1, XA_h1, X1_h1
    cdef double[:,:] u_t_l0, u_t_h0, u_t_l1, u_t_h1

    # Environmental water holder samples
    cdef double[:,:] XA_e0, X1_e0, XA_e1, X1_e1
    cdef double[:] u_e0, u_e1

    if planner:
        plan = 1
        # Planner samples
        XA0 = np.zeros([T, 4])
        X10 = np.zeros([T, 3])
        U0 = np.zeros(T)
        XA1 = np.zeros([T, 4])
        X11 = np.zeros([T, 3])
        U1 = np.zeros(T)
    else:
        if stats == 0:
            # Consumptive user samples
            XA_l0 = np.zeros([N_e_l, T, 5])
            X1_l0 = np.zeros([N_e_l, T, 4])
            XA_h0 = np.zeros([N_e_h, T, 5])
            X1_h0 = np.zeros([N_e_h, T, 4])
            u_t_l0 = np.zeros([N_e_l, T])
            u_t_h0 = np.zeros([N_e_h, T])

            XA_l1 = np.zeros([N_e_l, T, 5])
            X1_l1 = np.zeros([N_e_l, T, 4])
            XA_h1 = np.zeros([N_e_h, T, 5])
            X1_h1 = np.zeros([N_e_h, T, 4])
            u_t_l1 = np.zeros([N_e_l, T])
            u_t_h1 = np.zeros([N_e_h, T])
           
           # Environmental water holder samples
            XA_e0 = np.zeros([T, 5])
            X1_e0 = np.zeros([T, 4])
            u_e0 = np.zeros(T)
            XA_e1 = np.zeros([T, 5])
            X1_e1 = np.zeros([T, 4])
            u_e1 = np.zeros(T)

    cdef int t = 0
    cdef int i = 0
    cdef int idx
    cdef double qe, ae = 0

    storage.precompute_I_shocks(T)
    storage.precompute_I_split(T)
    env.precompute_e_shocks(T)
    env.precompute_P_adj_shocks(T, env.P_adj, 30)
    
    # Run simulation
    for t in range(T):
        if bo == 1: 
            env.draw_P_adj(t)
            market.P_adj = env.P_adj
            P_adj_sim[t] = market.P_adj
        
        utility.fail = 0

        # ================================================
        #       M = 0, Summer
        # ================================================

        ###### Initial state ######

        S_sim[t, 0] = storage.S
        I_sim[t, 0] = storage.I
        Z_sim[t, 0] = storage.Spill
        
        storage.natural_flows(0)
        F1_tilde[t, 0] = storage.F1_tilde
        F3_tilde[t, 0] = storage.F3_tilde
        
        Bhat_sim[t, 0] = env.Bhat
        
        ###### Withdrawals and Allocations ######

        if plan == 1:
            W_sim[t, 0] = utility.withdraw_ch7(storage.S, storage.I_tilde, env.Bhat, 0, env.turn_off)
            
            # Planner state and action
            XA0[t, 0] = W_sim[t, 0]
            XA0[t, 1] = storage.S
            XA0[t, 2] = storage.I_tilde
            XA0[t, 3] = env.Bhat
        else:
            # Users make withdrawals
            users.withdraw_ch7(storage.S, utility.s, storage.I_tilde, 0)
            env.withdraw(storage.S, utility.s[utility.I_env], storage.I_tilde, 0)
            utility.make_allocations(users.w, env.w)


            if stats == 0: 
                ##########  Record user state and action pairs [w, S, s, e, I, M] ###########
                for i in range(N_e_l):
                    idx = I_e_l[i]
                    XA_l0[i, t, 0] = users.w[idx]
                    XA_l0[i, t, 1] = storage.S
                    XA_l0[i, t, 2] = utility.s[idx]
                    XA_l0[i, t, 3] = users.e[idx]
                    XA_l0[i, t, 4] = storage.I_tilde
                
                for i in range(N_e_h):
                    idx = I_e_h[i]
                    XA_h0[i, t, 0] = users.w[idx]
                    XA_h0[i, t, 1] = storage.S
                    XA_h0[i, t, 2] = utility.s[idx]
                    XA_h0[i, t, 3] = users.e[idx]
                    XA_h0[i, t, 4] = storage.I_tilde

                XA_e0[t, 0] = env.w
                XA_e0[t, 1] = storage.S
                XA_e0[t, 2] = utility.s[utility.I_env]
                XA_e0[t, 3] = env.Bhat
                XA_e0[t, 4] = storage.I_tilde
                #########################################################################
        
        # Inform users of allocations
        users.allocate(utility.a, storage.I_tilde)
        env.allocate(utility.a[utility.I_env], storage.Spill, utility.max_R, storage.F1_tilde, storage.F3_tilde, storage.Pr, 0)

        if plan == 0:
            # Determine releases
            W_sim[t, 0] = utility.deliver_ch7(storage, 0, 0)
        
        A_sim[t, 0] = utility.A

        ###### Clear spot market ######
        
        market.open_market(users, env, 0)
        P_sim[t, 0] = market.solve_price(utility.A, storage.I_tilde, env.Bhat, init, plan)
        
        ###### Compute payoffs ######

        # Extract and consume water
        qe = env.consume(P_sim[t, 0], 0, plan)
        E_sim[t, 0] = utility.extract(qe)
        Profit_sim[t, 0] = users.consume(P_sim[t, 0], storage.I_tilde, plan)
        Q_sim[t, 0] = c_sum(users.N, users.q)
        U_low_sim[t, 0] = users.U_low
        U_high_sim[t, 0] = users.U_high
        
        #if stats == 1:
        users.user_stats(utility.s, utility.x)
        Q_low_sim[t, 0] = users.Q_low
        Q_high_sim[t, 0] = users.Q_high
        Q_env_sim[t, 0] = qe
        A_low_sim[t, 0] = users.A_low
        A_high_sim[t, 0] = users.A_high
        A_env_sim[t, 0] = env.a 
        S_low_sim[t, 0] = users.S_low
        S_high_sim[t, 0] = users.S_high
        S_env_sim[t, 0] = utility.s[utility.I_env]

        # Calculate actual river flows
        storage.river_flow(W_sim[t,0], E_sim[t, 0], 0, 0)
        F1[t, 0] = storage.F1
        F3[t, 0] = storage.F3

        # Env payoff
        B_sim[t, 0] = env.payoff(storage.F1, storage.F3, storage.F1_tilde, storage.F3_tilde, P_sim[t,0])
        SW_sim[t, 0] = Profit_sim[t, 0] + B_sim[t, 0]
        Budget_sim[t, 0] = env.budget

        ###### State transition ######

        # New inflows
        storage.update_ch7(W_sim[t, 0], E_sim[t, 0], 0, t)
        
        # New productivity shocks
        users.update()
        env.update()

        if plan == 0:
            # User accounts are updated
            utility.update(storage.S, storage.I, storage.Loss, storage.Spill, utility.w, 0)
            
            if stats == 0:
                ############### Record state transition [S1, s1, e1, I1, M1] and payoff #####
                for i in range(N_e_l):
                    idx = I_e_l[i]
                    X1_l0[i, t, 0] = storage.S
                    X1_l0[i, t, 1] = utility.s[idx]
                    X1_l0[i, t, 2] = users.e[idx]
                    X1_l0[i, t, 3] = storage.I_tilde
                    u_t_l0[i, t] = users.profit[idx]
                for i in range(N_e_h):
                    idx = I_e_h[i]
                    X1_h0[i, t, 0] = storage.S
                    X1_h0[i, t, 1] = utility.s[idx]
                    X1_h0[i, t, 2] = users.e[idx]
                    X1_h0[i, t, 3] = storage.I_tilde
                    u_t_h0[i, t] = users.profit[idx]

                X1_e0[t, 0] = storage.S
                X1_e0[t, 1] = utility.s[utility.I_env]
                X1_e0[t, 2] = env.Bhat
                X1_e0[t, 3] = storage.I_tilde
                u_e0[t] = B_sim[t, 0]
                #########################################################################
        else:
            # Planner state transition
            X10[t, 0] = storage.S
            X10[t, 1] = storage.I_tilde
            X10[t, 2] = env.Bhat
            #Planner payoff
            if env.turn_off == 1:
                U0[t] = Profit_sim[t, 0]
            else:
                U0[t] = SW_sim[t, 0]
        
        itrs += 1

        if plan == 0 and utility.fail == 1:
            #print '-------     Test output     ------'
            #print 'Time: ' + str(t)
            """
            print 'M = 0, Summer'
            print 'Storage: ' + str(S_sim[t, 0])
            print 'Inflow: ' + str(I_sim[t, 0])
            print 'Spill: ' + str(Z_sim[t, 0])
            print 'User accounts: ' + str(np.array(utility.s))
            print 'User account sum: ' + str(np.sum(utility.s))
            print 'Natural flows, F1: ' + str(storage.F1_tilde)
            print 'Natural flows, F3: ' + str(storage.F3_tilde)
            print 'Bhat: ' + str(Bhat_sim[t,0])
            print 'Withdrawals: ' + str(W_sim[t, 0])
            print 'User allocations: ' + str(np.array(users.a))
            print 'User withdrawals: ' + str(np.array(users.w))
            print 'Env allocations: ' + str(env.a)
            print 'Allocations: ' + str(A_sim[t, 0])
            print 'Price: ' + str(P_sim[t, 0])
            #print 'Excess demand: ' + str(env.EX)
            print 'Water use: ' + str(Q_sim[t, 0])
            print 'Environmental use: ' + str(qe)
            print 'Extraction: ' + str(E_sim[t, 0])
            print 'User consumption: ' + str(np.array(users.q))
            print 'User consumption sum:' + str(np.sum(users.q))
            print 'River flows, F1: ' + str(storage.F1)
            print 'River flows, F2: ' + str(storage.F2)
            print 'River flows, F3: ' + str(storage.F3)
            print 'User Welfare: ' + str(Profit_sim[t, 0])
            print 'Social Welfare: ' + str(SW_sim[t, 0])
            print 'Env Welfare: ' + str(env.u)
            print 'Env budget: ' + str(env.budget)
            """
        
        # ================================================
        #       M = 1, Winter
        # ================================================

        ###### Initial state ######

        S_sim[t, 1] = storage.S
        I_sim[t, 1] = storage.I
        Z_sim[t, 1] = storage.Spill
        
        storage.natural_flows(1)
        F1_tilde[t, 1] = storage.F1_tilde
        F3_tilde[t, 1] = storage.F3_tilde

        Bhat_sim[t, 1] = env.Bhat

        ###### Withdrawals and Allocations ######
        
        if plan == 1:
            W_sim[t, 1] = utility.withdraw_ch7(storage.S, storage.I_tilde, env.Bhat, 1, env.turn_off)
            
            # Planner state and action
            XA1[t, 0] = W_sim[t, 1]
            XA1[t, 1] = S_sim[t, 1]
            XA1[t, 2] = storage.I_tilde
            XA1[t, 3] = env.Bhat
        else:
            users.withdraw_ch7(storage.S, utility.s, storage.I_tilde, 1)
            env.withdraw(storage.S, utility.s[utility.I_env], storage.I_tilde, 1)
            utility.make_allocations(users.w, env.w)
            S_low_sim[t, 1] = users.S_low
            S_high_sim[t, 1] = users.S_high
            S_env_sim[t, 1] = utility.s[utility.I_env]
            
            if stats == 0 :
                ##########  Record user state and action pairs [w, S, s, e, I, M] ###########
                for i in range(N_e_l):
                    idx = I_e_l[i]
                    XA_l1[i, t, 0] = users.w[idx]
                    XA_l1[i, t, 1] = storage.S
                    XA_l1[i, t, 2] = utility.s[idx]
                    XA_l1[i, t, 3] = users.e[idx]
                    XA_l1[i, t, 4] = storage.I_tilde
                
                for i in range(N_e_h):
                    idx = I_e_h[i]
                    XA_h1[i, t, 0] = users.w[idx]
                    XA_h1[i, t, 1] = storage.S
                    XA_h1[i, t, 2] = utility.s[idx]
                    XA_h1[i, t, 3] = users.e[idx]
                    XA_h1[i, t, 4] = storage.I_tilde

                XA_e1[t, 0] = env.w
                XA_e1[t, 1] = storage.S
                XA_e1[t, 2] = utility.s[utility.I_env]
                XA_e1[t, 3] = env.Bhat
                XA_e1[t, 4] = storage.I_tilde
                #########################################################################

        A_sim[t, 1] = utility.A       
        
        # Inform users of allocations
        env.allocate(utility.a[utility.I_env], storage.Spill, 0, storage.F1_tilde, storage.F3_tilde, storage.Pr, 1)
        users.allocate(utility.a, storage.I_tilde)

        ###### Clear spot market ######
        
        Q_sim[t, 1] = 0
        E_sim[t, 1] = 0
        A_env_sim[t, 1] = env.a 
        
        if plan == 0:
            # Open the spot market
            market.open_market(users, env, 1)
            P_sim[t, 1] = market.solve_price(utility.A, storage.I_tilde, env.Bhat, init, plan)
            
        ###### Compute payoffs ######
            # Extract and consume water
            qe = env.consume(P_sim[t, 1], 1, plan)
            We = utility.record_trades(users, env, storage)
            Profit_sim[t, 1] = c_max(users.take_sale_cash(utility.a, P_sim[t, 1]), 0)
            W_sim[t, 1] = utility.deliver_ch7(storage, 1, We)
            U_low_sim[t, 1] = c_max(users.U_low, 0)
            U_high_sim[t, 1] = c_max(users.U_high, 0)
        else:
            env.q = c_max(W_sim[t, 1] - 2 * storage.delta_a[1], 0)* (1 - storage.delta1b)
            env.a = env.q
            P_sim[t, 1] = 0

        Q_env_sim[t, 1] = env.q 
        
        users.user_stats(utility.s, utility.x)
        Q_low_sim[t, 1] = users.Q_low
        Q_high_sim[t, 1] = users.Q_high
        A_low_sim[t, 1] = users.A_low
        A_high_sim[t, 1] = users.A_high
        S_low_sim[t, 1] = users.S_low
        S_high_sim[t, 1] = users.S_high
        S_env_sim[t, 1] = utility.s[utility.I_env]

        # Calculate actual river flows
        storage.river_flow(W_sim[t,1], E_sim[t, 1], 1, 0)
        F1[t, 1] = storage.F1
        F3[t, 1] = storage.F3

        # Env payoff
        B_sim[t, 1] = env.payoff(storage.F1, storage.F3, storage.F1_tilde, storage.F3_tilde, P_sim[t, 1])
        SW_sim[t, 1] = B_sim[t, 1] + Profit_sim[t, 1]
        Budget_sim[t, 1] = env.budget

        ###### State transition ######

        storage.update_ch7(W_sim[t, 1], E_sim[t, 1], 1, t)
        env.update()

        if plan == 0:
            
            # User accounts are updated
            utility.update(storage.S, storage.I, storage.Loss, storage.Spill, utility.w, 1)
            if stats == 0:
                ############### Record state transition [S1, s1, e1, I1, M1] and payoff #####
                for i in range(N_e_l):
                    idx = I_e_l[i]
                    X1_l1[i, t, 0] = storage.S
                    X1_l1[i, t, 1] = utility.s[idx]
                    X1_l1[i, t, 2] = users.e[idx]
                    X1_l1[i, t, 3] = storage.I_tilde
                    u_t_l1[i, t] = users.profit[idx]
                for i in range(N_e_h):
                    idx = I_e_h[i]
                    X1_h1[i, t, 0] = storage.S
                    X1_h1[i, t, 1] = utility.s[idx]
                    X1_h1[i, t, 2] = users.e[idx]
                    X1_h1[i, t, 3] = storage.I_tilde
                    u_t_h1[i, t] = users.profit[idx]

                X1_e1[t, 0] = storage.S
                X1_e1[t, 1] = utility.s[utility.I_env]
                X1_e1[t, 2] = env.Bhat
                X1_e1[t, 3] = storage.I_tilde
                u_e1[t] = B_sim[t, 1]
                #########################################################################
        else:
            #Planner state transition
            X11[t, 0] = storage.S
            X11[t, 1] = storage.I_tilde
            X11[t, 2] = env.Bhat
            #Planner payoff
            if env.turn_off == 1:
                U1[t] = Profit_sim[t, 1]
            else:
                U1[t] = SW_sim[t, 1]
        
        itrs += 1
    
        if plan == 0 and utility.fail == 1:
            print '-------     Test output     ------'
            print 'Time: ' + str(t)
            print 'M = 1, Winter'
            print 'Storage: ' + str(S_sim[t, 1])
            print 'Inflow: ' + str(I_sim[t, 1])
            print 'Spill: ' + str(Z_sim[t, 1])
            print 'User accounts: ' + str(np.array(utility.s))
            print 'User account sum: ' + str(np.sum(utility.s))
            print 'Natural flows, F1: ' + str(storage.F1_tilde)
            print 'Natural flows, F3: ' + str(storage.F3_tilde)
            print 'Bhat: ' + str(Bhat_sim[t,1])
            print 'Withdrawals: ' + str(W_sim[t, 1])
            print 'User allocations: ' + str(np.array(users.a))
            print 'User withdrawals: ' + str(np.array(users.w))
            print 'All withdrawals: ' + str(np.array(utility.w))
            print 'Env allocations: ' + str(env.a)
            print 'Env withdrawals: ' + str(env.w)
            print 'Allocations: ' + str(A_sim[t, 1])
            print 'Price: ' + str(P_sim[t, 1])
            #print 'Excess demand: ' + str(env.EX)
            print 'Water use: ' + str(Q_sim[t, 1])
            print 'Environmental use: ' + str(qe)
            print 'Extraction: ' + str(E_sim[t, 1])
            print 'User consumption: ' + str(np.array(users.q))
            print 'User consumption sum:' + str(np.sum(users.q))
            print 'River flows, F1: ' + str(storage.F1)
            print 'River flows, F2: ' + str(storage.F2)
            print 'River flows, F3: ' + str(storage.F3)
            print 'User Welfare: ' + str(Profit_sim[t, 1])
            print 'Social Welfare: ' + str(SW_sim[t, 1])
            print 'Env Welfare: ' + str(env.u)
            print 'Env budget: ' + str(env.budget)
    
    if budgetonly:
        data = {'Budget' : np.asarray(Budget_sim),
                'P_adj' : np.asarray(P_adj_sim)}
    else:
        data = {'W': np.asarray(W_sim),
                'SW': np.asarray(SW_sim),
                'Profit' : np.asarray(Profit_sim),
                'S': np.asarray(S_sim),
                'I': np.asarray(I_sim),
                'Z': np.asarray(Z_sim),
                'P' : np.asarray(P_sim),
                'E' : np.asarray(E_sim),
                'Q' : np.asarray(Q_sim),
                'A' : np.asarray(A_sim),
                'F1' : np.asarray(F1),
                'F3' : np.asarray(F3),
                'F1_tilde' : np.asarray(F1_tilde),
                'F3_tilde' : np.asarray(F3_tilde),
                'B' : np.asarray(B_sim), 
                'Budget' : np.asarray(Budget_sim),
                'Q_low' : np.asarray(Q_low_sim),
                'Q_high' : np.asarray(Q_high_sim),
                'Q_env' : np.asarray(Q_env_sim),
                'Bhat' : np.asarray(Bhat_sim),
                'A_low' : np.asarray(A_low_sim),
                'A_high' : np.asarray(A_high_sim),
                'A_env' : np.asarray(A_env_sim),
                'S_low' : np.asarray(S_low_sim),
                'S_high' : np.asarray(S_high_sim),
                'S_env' : np.asarray(S_env_sim),
                'U_low' : np.asarray(U_low_sim),
                'U_high' : np.asarray(U_high_sim)}
        
        if plan == 1: 
            data['XA1'] = np.asarray(XA1)
            data['X11'] = np.asarray(X11)
            data['U1'] = np.asarray(U1) 
            data['XA0'] = np.asarray(XA0)
            data['X10'] = np.asarray(X10)
            data['U0'] = np.asarray(U0) 
        else:
            if stats == 0:
                data['XA_l'] = np.asarray(XA_l0)
                data['X1_l' ] = np.asarray(X1_l0)
                data['u_t_l'] = np.asarray(u_t_l0) 
                data['XA_h'] = np.asarray(XA_h0)
                data['X1_h'] = np.asarray(X1_h0)
                data['u_t_h'] = np.asarray(u_t_h0) 

                data['1'] = { 
                'XA_l' : np.asarray(XA_l1),
                'X1_l' : np.asarray(X1_l1),
                'u_t_l' : np.asarray(u_t_l1), 
                'XA_h' : np.asarray(XA_h1),
                'X1_h' : np.asarray(X1_h1),
                'u_t_h' : np.asarray(u_t_h1), 
                }
                
                data['XA_e0'] = np.asarray(XA_e0) 
                data['X1_e0'] = np.asarray(X1_e0) 
                data['u_e0'] = np.asarray(u_e0) 
                data['XA_e1'] = np.asarray(XA_e1) 
                data['X1_e1'] = np.asarray(X1_e1) 
                data['u_e1'] = np.asarray(u_e1) 
    
    if multi:
        que.put(data)
    else:
        return data

def run_sim(int job, int T, int stats, Users users, Storage storage, Utility utility, Tilecode market_d, que, multi = True):
    
    "Run simulation episode for T periods"

    users.seed(job)
    storage.seed(job)
    
    # Representative user explorers 
    cdef int[:] I_e_l = users.I_e_l 
    cdef int[:] I_e_h = users.I_e_h 
    cdef int N_e_l = users.N_e #len(I_e_l)
    cdef int N_e_h = users.N_e #len(I_e_h)
    cdef int N_e = N_e_l + N_e_h

    cdef double[:,:,:] XA_l
    cdef double[:,:,:] X1_l
    cdef double[:,:,:] XA_h
    cdef double[:,:,:] X1_h
    cdef double[:,:] u_t_l
    cdef double[:,:] u_t_h

    if users.testing == 0:
        XA_l = np.zeros([N_e_l, T, 5])
        X1_l = np.zeros([N_e_l, T, 4])
        XA_h = np.zeros([N_e_h, T, 5])
        X1_h = np.zeros([N_e_h, T, 4])
        u_t_l = np.zeros([N_e_l, T])
        u_t_h = np.zeros([N_e_h, T])
    elif users.test_explore == 1:
        XA_l = np.zeros([1, T, 5])
        X1_l = np.zeros([1, T, 4])
        XA_h = np.zeros([1, T, 5])
        X1_h = np.zeros([1, T, 4])
        u_t_l = np.zeros([1, T])
        u_t_h = np.zeros([1, T])


    cdef double[:] W_sim = np.zeros(T)
    cdef double[:] Q_sim = np.zeros(T)
    cdef double[:] SW_sim = np.zeros(T)
    cdef double[:] S_sim = np.zeros(T)
    cdef double[:] I_sim = np.zeros(T)
    cdef double[:] Z_sim = np.zeros(T)
    cdef double[:] U_low_sim = np.zeros(T)
    cdef double[:] U_high_sim = np.zeros(T)
    cdef double[:] S_low_sim = np.zeros(T)
    cdef double[:] S_high_sim = np.zeros(T)
    cdef double[:] X_low_sim = np.zeros(T)
    cdef double[:] X_high_sim = np.zeros(T)
    cdef double[:] trade_sim = np.zeros(T)
    cdef double[:] W_low_sim = np.zeros(T)
    cdef double[:] W_high_sim = np.zeros(T)
    cdef double[:] Q_low_sim = np.zeros(T)
    cdef double[:] Q_high_sim = np.zeros(T)
    cdef double[:] A_low_sim = np.zeros(T)
    cdef double[:] trade_high_sim = np.zeros(T)
    cdef double[:] trade_low_sim = np.zeros(T)
    cdef double[:] A_high_sim = np.zeros(T)
    cdef double[:] P_sim = np.zeros(T)
    cdef double S1
    cdef double I1

    cdef double[:] test_payoff = np.zeros(T)
    cdef int record_state = 1

    if users.testing == 1 and users.test_explore == 0:
        record_state = 0
    elif stats == 1:
        record_state = 0


    cdef double I_tilde
    cdef int t = 0
    cdef int i = 0
    cdef int idx

    storage.precompute_I_shocks(T)
    for t in range(T):
        # Initial state
        S_sim[t] = storage.S
        I_sim[t] = storage.I
        Z_sim[t] = storage.Spill 
        I_tilde = I_sim[t] * (storage.I_bar**-1)

        # Users make withdrawals
        users.withdraw(storage.S, utility.s, I_tilde)

        W_sim[t] = utility.release(users.w, storage.S)
        users.allocate(utility.a, I_tilde)
        
        if record_state == 1: 
            ##########  Record user state and action pairs [w, S, s, e, I] ###########
            for i in range(N_e_l):
                idx = I_e_l[i]
                XA_l[i, t, 0] = users.w[idx]
                XA_l[i, t, 1] = S_sim[t]
                XA_l[i, t, 2] = utility.s[idx]
                XA_l[i, t, 3] = users.e[idx]
                XA_l[i, t, 4] = I_tilde
            
            for i in range(N_e_h):
                idx = I_e_h[i]
                XA_h[i, t, 0] = users.w[idx]
                XA_h[i, t, 1] = S_sim[t]
                XA_h[i, t, 2] = utility.s[idx]
                XA_h[i, t, 3] = users.e[idx]
                XA_h[i, t, 4] = I_tilde
            #########################################################################


        # Water is delivered and spot market opens
        P_sim[t] = users.clear_market(I_tilde, market_d, 0)
        SW_sim[t] = users.consume(P_sim[t], I_tilde, 0)
        Q_sim[t] = c_sum(users.N, utility.a)


        if stats == 1:
            users.user_stats(utility.s, utility.x)
            U_low_sim[t] = users.U_low
            U_high_sim[t] = users.U_high
            W_low_sim[t] = users.W_low
            W_high_sim[t] = users.W_high
            A_low_sim[t] = users.A_low
            A_high_sim[t] = users.A_high
            Q_low_sim[t] = users.Q_low
            Q_high_sim[t] = users.Q_high
            S_low_sim[t] = users.S_low
            S_high_sim[t] = users.S_high
            X_low_sim[t] = users.X_low
            X_high_sim[t] = users.X_high
            trade_low_sim[t] = users.trade_low
            trade_high_sim[t] = users.trade_high
            trade_sim[t] = users.tradeVOL
        
        if users.testing == 1:
            test_payoff[t] = users.profit[users.test_idx] 

        ### State transition ###

        # New Inflows
        S1 = storage.update(W_sim[t], t)
        I1_tilde = storage.I * (storage.I_bar**-1)

        # User accounts are updated
        utility.update(storage.S, storage.I, storage.Loss, storage.Spill, users.w, 0)


        # New productivity shocks
        users.update()

        if record_state == 1:
            ############### Record state transition [S1, s1, e1, I1] and payoff #####
            for i in range(N_e_l):
                idx = I_e_l[i]
                X1_l[i, t, 0] = S1
                X1_l[i, t, 1] = utility.s[idx]
                X1_l[i, t, 2] = users.e[idx]
                X1_l[i, t, 3] = I1_tilde
                u_t_l[i, t] = users.profit[idx]
            for i in range(N_e_h):
                idx = I_e_h[i]
                X1_h[i, t, 0] = S1
                X1_h[i, t, 1] = utility.s[idx]
                X1_h[i, t, 2] = users.e[idx]
                X1_h[i, t, 3] = I1_tilde
                u_t_h[i, t] = users.profit[idx]
            #########################################################################

    if stats == 1:
        data = {'W': np.asarray(W_sim), 'SW': np.asarray(SW_sim), 'S': np.asarray(S_sim), 'I': np.asarray(I_sim),  'Z': np.asarray(Z_sim), 'U_low' : np.asarray(U_low_sim) , 'U_high' : np.asarray(U_high_sim), 'W_low' : np.asarray(W_low_sim) , 'W_high' : np.asarray(W_high_sim), 'S_low' : np.asarray(S_low_sim) , 'S_high' : np.asarray(S_high_sim), 'X_low' : np.asarray(X_low_sim) , 'X_high' : np.asarray(X_high_sim), 'P' : np.asarray(P_sim),  'Q' : np.asarray(Q_sim), 'trade' : np.asarray(trade_sim), 'A_low' : np.asarray(A_low_sim) , 'A_high' : np.asarray(A_high_sim), 'Q_low' : np.asarray(Q_low_sim) , 'Q_high' : np.asarray(Q_high_sim), 'trade_low' : np.asarray(trade_low_sim) , 'trade_high' : np.asarray(trade_high_sim)}
    else:
        if record_state == 1:
            data = {'XA_l': np.asarray(XA_l), 'X1_l' : np.asarray(X1_l), 'u_t_l' : np.asarray(u_t_l), 'XA_h': np.asarray(XA_h), 'X1_h' : np.asarray(X1_h), 'u_t_h' : np.asarray(u_t_h), 'W': np.asarray(W_sim), 'SW': np.asarray(SW_sim), 'S': np.asarray(S_sim), 'I': np.asarray(I_sim),  'Z': np.asarray(Z_sim), 'P' : np.asarray(P_sim), 'Q' : np.asarray(Q_sim)}
        else:
            data = {'test' : np.asarray(test_payoff)}    
    
    if multi:
        que.put(data)
    else:
        return data 


def run_planner_sim(int job, int T, Users users, Storage storage, Utility utility, Tilecode policy, double delta,  int planner, int explore, q, seed, multi = True, myopic = False, SOP=False, Sbar=0):

    "Run simulation episode for T periods, store data for Q-learning"
    
    if seed == 0:
        seed = time.time()

    c_seed(job)
    users.seed(job * seed)
    storage.seed(job * seed)
    
    cdef int useall = 0
    if myopic:
        useall = 1

    # Temp variables for the simulation episode
    cdef double[:] W_sim = np.zeros(T)
    cdef double[:] Q_sim = np.zeros(T)
    cdef double[:] S_sim = np.zeros(T)
    cdef double[:] I_sim = np.zeros(T)
    cdef double[:] P_sim = np.zeros(T)
    cdef double[:] Z_sim = np.zeros(T)
    cdef double[:] SW_sim = np.zeros(T)
    cdef double[:] U_low_sim = np.zeros(T)
    cdef double[:] U_high_sim = np.zeros(T)
    cdef double[:] A_low_sim = np.zeros(T)
    cdef double[:] A_high_sim = np.zeros(T)
    cdef double[:] trade_sim = np.zeros(T)
    cdef double[:] zeros = np.zeros(users.N)

    cdef double[:,:] XA = np.zeros([T, 3])
    cdef double[:,:] X1 = np.zeros([T, 2])
    
    cdef double I_tilde
    cdef int t
    cdef double w_min = 0
    cdef double w_max = 0
    cdef double w_star = 0
    cdef int finetune = 0
    cdef double ran = 0
    cdef double[:] state = np.zeros(2)

    cdef Tilecode perf_market = users.perf_market

    cdef double U, V, Z
    cdef double trade
    cdef double c_pi = math.pi
    cdef int i
    if delta > 0:
        finetune = 1

    cdef double SSbar = Sbar
    cdef int SSOP = 0
    if SOP:
        SSOP = 1

    storage.precompute_I_shocks(T)

    # Run simulation
    for t in range(T):

        # Initial state
        S_sim[t] = storage.S
        I_sim[t] = storage.I
        I_tilde = I_sim[t] / storage.I_bar
        Z_sim[t] = storage.Spill

        # Planner makes storage release (with exploration)
        if explore == 1:
            if finetune == 0:
                W_sim[t] = c_rand() * storage.S

                ##########  Record state and action pairs [W, S, I] ###########
                XA[t, 0] = W_sim[t]
                XA[t, 1] = S_sim[t]
                XA[t, 2] = I_tilde
                ###############################################################
            else:
                state[0] = storage.S
                state[1] = I_tilde
                w_star = policy.one_value(state)

                U = c_rand()
                V = c_rand()
                Z = ((-2 * c_log(U))**0.5)*c_cos(2*c_pi*V)
                W_sim[t] = c_min(c_max(Z * (delta * storage.S) + w_star, 0), storage.S)

                ##########  Record state and action pairs [W, S, I,] ###########
                XA[t, 0] = W_sim[t]
                XA[t, 1] = S_sim[t]
                XA[t, 2] = I_tilde
                ###############################################################
        else:
            if useall == 1:
                W_sim[t] = S_sim[t]*1.0
            elif SSOP == 1:
                W_sim[t] = c_min(S_sim[t], SSbar)
            else:
                state[0] = storage.S
                state[1] = I_tilde
                W_sim[t] = c_max(c_min(policy.one_value(state), S_sim[t]), 0)

        # Water delivered
        Q_sim[t] = storage.release(W_sim[t])

        # Make water allocations
        utility.update(S_sim[t], I_sim[t], storage.Loss, storage.Spill, zeros, Q_sim[t])
        users.allocate(utility.a, I_tilde)

        # Spot market opens
        P_sim[t] = users.clear_market(I_tilde, perf_market, planner)
        SW_sim[t] = users.consume(P_sim[t], I_tilde, planner)

        # stats
        U_low_sim[t] = users.U_low
        U_high_sim[t] = users.U_high
        trade = 0
        for i in range(users.N): 
            trade += abs(users.trade[i])
        trade_sim[t] = trade
        A_low_sim[t] = utility.a[0]
        A_high_sim[t] = utility.a[users.N - 1]

        # Inflows are received
        storage.update(W_sim[t], t)

        # New productivity shocks for users
        users.update()

        ##########  Record State transition [S', I'] ###########
        X1[t, 0] = storage.S
        X1[t, 1] = storage.I / storage.I_bar
        ##################################################
        
    # Place simulation data into dictionary
    data = {'XA': np.asarray(XA), 'X1' : np.asarray(X1), 'SW': np.asarray(SW_sim), 'W': np.asarray(W_sim), 'S': np.asarray(S_sim),
            'I': np.asarray(I_sim), 'P': np.asarray(P_sim), 'Z': np.asarray(Z_sim), 'U_low' : np.asarray(U_low_sim), 'U_high' : np.asarray(U_high_sim), 'Q' : np.asarray(Q_sim), 'trade' : np.asarray(trade_sim), 'A_low' : np.asarray(A_low_sim), 'A_high' : np.asarray(A_high_sim)}

    if multi:
        q.put(data)
    else:
        return data


class Simulation:

    "Decentralised model simulation class"

    def __init__(self, para, ch7=False):
       
        self.sample_rate = para.sample_rate

        self.N = para.N                       # Number of users
        self.beta = para.beta                 # Discount rate

        self.CPUs = para.CPU_CORES

        self.ITERMAX = para.ITER2 + 4
        self.ITER = 0
        self.ITERNEW = 0
        self.ITEROLD = 0
        self.S = np.zeros(150)
        self.W = np.zeros(150)
        self.E = np.zeros(150)
        self.B = np.zeros(150)

        ####################     Result containers

        # Summary stats
        self.stats = {}
        names  = ['Mean', 'SD', '25th', '75th', '2.5th', '97.5th', 'Min', 'Max']
        formats = ['f4','f4','f4','f4','f4','f4','f4','f4']
        vars = ['S', 'W', 'I', 'SW', 'Z', 'U_low', 'U_high', 'A_low', 'A_high', 'Q_low', 'Q_high', 'W_low', 'W_high', 'S_low', 'S_high',
                 'X_low', 'X_high', 'trade', 'P', 'Q', 'F1', 'F3', 'F1_tilde', 'F3_tilde', 'E', 'Profit', 'trade_low', 'trade_high', 'A', 'Q_env', 'Bhat', 'B', 'Budget', 'A_env', 'A_low', 'A_high', 'S_env']
        if ch7:
            for var in vars:
                self.stats[var] = { 'Summer' : np.zeros(self.ITERMAX, dtype={'names':names, 'formats':formats}),
                                    'Winter' : np.zeros(self.ITERMAX, dtype={'names':names, 'formats':formats}),
                                    'Annual' : np.zeros(self.ITERMAX, dtype={'names':names, 'formats':formats})}
        else:
            for var in vars:
                self.stats[var] = np.zeros(self.ITERMAX, dtype={'names':names, 'formats':formats})

        # Data series
        self.series = {'S' : 0, 'W' : 0, 'I' : 0, 'SW' : 0, 'Z' : 0, 'P' : 0, 'Q' : 0} 
        self.series_old = {'S' : 0, 'W' : 0, 'I' : 0, 'SW' : 0, 'Z' : 0, 'P' : 0, 'Q' : 0} 
        self.full_series = {'S' : 0, 'W' : 0, 'I' : 0, 'SW' : 0, 'Z' : 0, 'U_low' : 0, 'U_high' : 0, 'W_low' : 0, 'W_high' : 0, 'S_low': 0, 'S_high' : 0, 'X_low': 0, 'X_high' : 0, 'P' : 0, 'Q' : 0, 'trade' : 0, 'A_low' : 0, 'A_high' : 0 , 'Q_low' : 0, 'Q_high' : 0 , 'trade_low' : 0, 'trade_high' : 0} 
        self.p_series = {'S' : 0, 'W' : 0, 'I' : 0, 'SW' : 0, 'Z' : 0, 'U_low' : 0, 'U_high' : 0, 'Q' : 0, 'trade' : 0, 'A_low' : 0, 'A_high' : 0, 'P' : 0}


    def test_sim(self, int T, Users users, Storage storage, Utility utility, Tilecode market_d):
        
        users.seed(0)
        storage.seed(0)
        
        # Representative user explorers 
        cdef int[:] I_e_l = users.I_e_l 
        cdef int[:] I_e_h = users.I_e_h 
        cdef int N_e_l = users.N_e #len(I_e_l)
        cdef int N_e_h = users.N_e #len(I_e_h)
        cdef int N_e = N_e_l + N_e_h


        cdef double[:] W_sim = np.zeros(T)
        cdef double[:] Q_sim = np.zeros(T)
        cdef double[:] SW_sim = np.zeros(T)
        cdef double[:] S_sim = np.zeros(T)
        cdef double[:] I_sim = np.zeros(T)
        cdef double[:] Z_sim = np.zeros(T)
        cdef double[:] P_sim = np.zeros(T)

        cdef double I_tilde 
        cdef int t = 0
        cdef int i = 0
        cdef int idx

        storage.precompute_I_shocks(T)

        # Run simulation
        for t in range(T):
            
            print '-------------------------------------------------'
            print 'Time: ' + str(t)

            # Initial state
            S_sim[t] = storage.S
            I_sim[t] = storage.I
            Z_sim[t] = storage.Spill 
            I_tilde = I_sim[t] * (storage.I_bar**-1)
            
            print 'Storage: ' + str(S_sim[t])
            print 'Inflow: ' + str(I_sim[t])
            print 'Spill: ' + str(Z_sim[t])
            
            print 'User accounts: ' + str(np.array(utility.s))
            print 'User account sum: ' + str(np.sum(utility.s))

            # Users make withdrawals 
            users.withdraw(storage.S, utility.s, I_tilde)
            
            print 'User withdrawals: ' + str(np.array(users.w))
            print 'User withdrawal sum: ' + str(np.sum(users.w))
            
            W_sim[t] = utility.release(users.w, storage.S)
            users.allocate(utility.a, I_tilde)

            print 'Withdrawals: ' + str(W_sim[t])
            print 'User allocations: ' + str(np.array(users.a))
            
            # Water is delivered and spot market opens
            P_sim[t] = users.clear_market(I_tilde, market_d, 0)
            SW_sim[t] = users.consume(P_sim[t], I_tilde, 0)
            Q_sim[t] = c_sum(users.N, utility.a)

            print '-------------------------'
            print 'W_sim[t]: ' + str(W_sim[t])
            print 'User withdrawal sum: ' + str(np.sum(users.w))
            print 'Utility fixed loss:' + str(utility.fixed_loss)
            print 'Utility marginal loss:' + str(utility.delta1b)
            print 'User allocation sum: ' + str(np.sum(utility.a))
            print 'User use sum: ' + str(users.Q)
            print 'W * (1-d) - fl - Q: ' + str(W_sim[t]*(1 - utility.delta1b) - utility.fixed_loss - np.sum(users.a))
            print '-------------------------'

            print 'Price: ' + str(P_sim[t])
            print 'Welfare: ' + str(SW_sim[t])

            ### State transition ###

            # New Inflows
            storage.update(W_sim[t], t)

            print 'Storage: ' + str(storage.S)
            print 'Inflow: ' + str(storage.I)
            print 'Spill: ' + str(storage.Spill)

            # User accounts are updated
            utility.update(storage.S, storage.I, storage.Loss, storage.Spill, users.w, 0)

            # New productivity shocks
            users.update()

            print 'Prod. shocks : ' + str(np.array(users.e))
            print 'Prod. shock mean: ' + str(np.mean(np.array(users.e)[np.array(users.I_high)]))

            #if utility.it == 100:
            import pdb; pdb.set_trace()

    def stack_sims(self, data, planner = False, stats=False, solve_planner = True, testing=False, partial=False):
        """
        Combine simulation data from multiple processes
        """
        
        if planner:
            self.series = self.p_series
            if solve_planner:
                self.XA_t = np.vstack(d['XA'] for d in data)
                self.X_t1 = np.vstack(d['X1'] for d in data)
            for x in self.series:
                self.series[x] = np.hstack(d[x] for d in data)  
            self.T = len(self.series['S'])
            if stats:
                self.summary_stats(sample = 1, percentiles=True)
        else:
            if testing:
                test_array = np.hstack([d['test'] for d in data])
                test_payoff = np.mean(test_array)
                print 'Test user welfare, mean: ' + str(test_payoff)
                print 'Test user welfare, min: ' + str(np.min(test_array))
                print 'Test user welfare, max: ' + str(np.max(test_array))
                print 'Test user welfare, SD: ' + str(np.var(test_array)**0.5)
                self.test_payoff = test_payoff
            else:
                if stats:
                    self.series = self.full_series
                    for x in self.series:
                        self.series[x] = np.hstack(d[x] for d in data)  
                else:
                    self.stack_user_samples(data, partial=partial)

                self.T = len(self.series['S'])
                self.summary_stats(sample=1, percentiles=stats)

    def stack_user_samples(self, data, partial=False, group=['_l', '_h'], summer=False, ch7=False):

        m = len(group)

        if not(partial):
            if not(summer):
                self.XA_t = [0,0]       # State action samples
                self.X_t1 = [0,0]       # State transition samples
                self.u_t = [0,0]        # Payoff samples
            else:
                self.XA_t1 = [0,0]       # State action samples
                self.X_t11 = [0,0]       # State transition samples
                self.u_t1 = [0,0]        # Payoff samples
            if ch7:
                if not(summer):
                    for x in self.series:
                        self.series[x] = np.vstack(d[x] for d in data)  
            else:
                for x in self.series:
                    self.series[x] = np.hstack(d[x] for d in data)  
        else:
            if ch7: 
                if not(summer):
                    for x in self.series:
                        temp =  np.vstack(d[x] for d in data)
                        N = temp.shape[0]
                        self.series[x] = np.vstack([self.series[x][N::], temp])  
            else:
                for x in self.series:
                    self.series[x] = np.hstack(d[x] for d in data)
                    #temp =  np.hstack(d[x] for d in data)
                    #N = temp.shape[0]
                    #self.series[x] = np.hstack([self.series[x][N::], temp])  
        for h in range(m):
            XA_t = np.hstack(d['XA' + group[h]] for d in data)
            X_t1 = np.hstack(d['X1' + group[h]] for d in data)
            u_t = np.hstack(d['u_t' + group[h]] for d in data)
            N_e = XA_t.shape[0]
            if N_e > 0:
                XA_t = np.vstack(XA_t[i,:,:] for i in range(N_e))
                X_t1 = np.vstack(X_t1[i,:,:] for i in range(N_e))
                u_t = np.hstack(u_t[i,:] for i in range(N_e))
            if partial:
                N1 = self.u_t[0].shape[0]
                N2 = u_t.shape[0]
                sample = np.random.choice(range(N1), size=N2, replace=False)
                if not(summer):
                    self.XA_t[h][sample, :] = XA_t 
                    self.X_t1[h][sample, :] = X_t1 
                    self.u_t[h][sample] = u_t
                else:
                    self.XA_t1[h][sample, :] = XA_t 
                    self.X_t11[h][sample, :] = X_t1 
                    self.u_t1[h][sample] = u_t
            else: 
                if not(summer):
                    self.XA_t[h] = XA_t 
                    self.X_t1[h] = X_t1 
                    self.u_t[h] = u_t
                else:
                    self.XA_t1[h] = XA_t 
                    self.X_t11[h] = X_t1 
                    self.u_t1[h] = u_t

    def summary_stats(self, sample = 0.5, percentiles=False, ch7=False):
        
        tic = time.time()
        N = int(sample * self.T)

        if not(ch7):
            for x in self.series:
                self.stats[x]['Mean'][self.ITER] = np.mean(self.series[x])
                self.stats[x]['SD'][self.ITER] = np.std(self.series[x])
                self.stats[x]['Min'][self.ITER] = np.min(self.series[x])
                self.stats[x]['Max'][self.ITER] = np.max(self.series[x])
                if percentiles:
                    self.stats[x]['25th'][self.ITER] = np.percentile(self.series[x][0:N], 25)
                    self.stats[x]['75th'][self.ITER] = np.percentile(self.series[x][0:N], 75)
                    self.stats[x]['2.5th'][self.ITER] = np.percentile(self.series[x][0:N], 2.5)
                    self.stats[x]['97.5th'][self.ITER] = np.percentile(self.series[x][0:N], 97.5)
        else:
            names = ['Summer', 'Winter', 'Annual']

            for x in self.series:
                for i in range(3):
                    name = names[i]
                    if name == 'Annual':
                        data = np.sum(self.series[x], axis=1)
                        if x == 'S' or x == 'S_low' or x == 'S_high' or x == 'S_env':
                            data = data / 2
                    else:
                        data = self.series[x][:,i]
                    self.stats[x][name]['Mean'][self.ITER] = np.mean(data)
                    self.stats[x][name]['SD'][self.ITER] = np.std(data)
                    self.stats[x][name]['Min'][self.ITER] = np.min(data)
                    self.stats[x][name]['Max'][self.ITER] = np.max(data)
                    if percentiles:
                        data = data[0:N]
                        self.stats[x][name]['25th'][self.ITER] = np.percentile(data, 25)
                        self.stats[x][name]['75th'][self.ITER] = np.percentile(data, 75)
                        self.stats[x][name]['2.5th'][self.ITER] = np.percentile(data, 2.5)
                        self.stats[x][name]['97.5th'][self.ITER] = np.percentile(data, 97.5)

        toc = time.time()

        print 'Summary stats time ' + str(toc - tic)
        
        if not(ch7):
            self.ITEROLD = self.ITER
            self.ITERNEW = self.ITERNEW + 1
            self.ITER = min(self.ITER + 1, self.ITERMAX - 1 ) 
            print 'ITEROLD: ' + str(self.ITEROLD)
            print 'ITER: ' + str(self.ITER)
    
    def finalise_stats(self, ):
        
        tic = time.time()
        stats = ['Mean', 'SD', 'Max', 'Min', '25th', '75th', '2.5th', '97.5th']
        
        if self.ITEROLD < (self.ITERMAX - 1):
            for x in self.series:
                for stat in stats:
                    self.stats[x][stat][self.ITERMAX - 1] = self.stats[x][stat][self.ITEROLD]
        
        toc = time.time()
        print 'Finalise stats time ' + str(toc - tic)
    
    def simulate(self, users, storage, utility, T, num_process, planner = False, partial = False, seed = 0, policy = False, polf = 0, delta = 0, stats = False, planner_explore = True, t_cost_off = False, myopic=False, SOP=False, Sbar=0):
        
        tic = time.time()
        
        print 'Running simulation for ' + str(T) + ' periods...'

        st = 0
        if stats:
            st = 1
        
        T = int(T / num_process)
        datalist = []
        
        ques = [RetryQueue() for i in range(num_process)]

        if planner:
            self.planner = True
            if t_cost_off:
                tc_off = 1
            else:
                tc_off = 0
            if planner_explore:
                explore = 1
            else:
                explore = 0
            if not(policy):
                polf = Tilecode(2, [2,2], 2)

            args = [(i, T, users, storage, utility, polf, delta, tc_off, explore, ques[i], seed, True, myopic, SOP, Sbar) for i in range(num_process)]
            jobs = [multiprocessing.Process(target=run_planner_sim, args=(a)) for a in args]

            for j in jobs: j.start()
            for q in ques: datalist.append(q.get())
            for j in jobs: j.join()
        
        else:
            if self.planner == True:
                self.planner = False
                self.series = self.series_old
            if num_process == 1:
                datalist.append(run_sim(0, T, st, users, storage, utility, users.market_d, 0, False))
            else:
                args = [(i, T, st, users, storage, utility, users.market_d, ques[i], True) for i in range(num_process)]
                jobs = [multiprocessing.Process(target=run_sim, args=(a)) for a in args]
        
                for j in jobs: j.start()
                for q in ques: datalist.append(q.get())
                for j in jobs: j.join()
        toc = time.time()
        
        print 'Simulation time: ' + str(round(toc - tic,2))
        
        tic1 = time.time()
        if users.testing == 1 and users.test_explore == 0:
            testing = True
        else:
            testing = False
        self.stack_sims(datalist, planner = planner, stats = stats, solve_planner = planner_explore, testing=testing, partial=partial)
        toc1 = time.time()
       
        del datalist
        
        if users.testing == 0:
            print 'Data stacking time: ' + str(toc1 - tic1)
            print 'Storage mean: ' + str(np.mean(self.series['S']))  
            print 'Inflow mean: ' + str(np.mean(self.series['I'])) 
            print 'Withdrawal mean: ' + str(np.mean(self.series['W']))  
            print 'Welfare mean: ' + str(np.mean(self.series['SW']))
        
        if self.ITER > 0:
            self.S[self.ITERNEW - 1] = np.mean(self.series['S'])
            self.W[self.ITERNEW - 1] = np.mean(self.series['SW'])

        if self.ITER > 1:
            pylab.plot(self.S[1:self.ITERNEW])
            pylab.show()
            pylab.plot(self.W[1:self.ITERNEW])
            pylab.show()

    def simulate_ch7(self, users, storage, utility, market, env, T, num_process, planner=False, stats=False, initP=False, partial=False, budgetonly=False):

        tic = time.time()

        self.T = T

        print 'Running simulation for ' + str(T) + ' periods...'
        
        st = 0
        init = 0
        nat = 0
        if stats:
            st = 1
        if initP:
            init = 1
         
        if num_process == 1:
            datalist = [run_ch7_sim(0, T, users, storage, utility, market, env, init, st, nat, 0, False, planner, budgetonly)]
        else:
            T = int(T / num_process)
            datalist = []
            ques = [RetryQueue() for i in range(num_process)]
            args = [(i, T, users, storage, utility, market, env, init, st, nat, ques[i], True, planner, budgetonly) for i in range(num_process)]
            jobs = [multiprocessing.Process(target=run_ch7_sim, args=(a)) for a in args]
            for j in jobs: j.start()
            for q in ques: datalist.append(q.get())
            for j in jobs: j.join()

        toc = time.time()

        print 'Simulation time: ' + str(round(toc - tic,2))

        tic1 = time.time()
        if budgetonly:
            Budget = np.sum(np.vstack(d['Budget'] for d in datalist), axis=1)
            P_adj = np.hstack(d['P_adj'] for d in datalist)
            NN = len(P_adj)
            P_adj = P_adj.reshape([NN, 1])
            print '--------------------------------------------------------'
            print '--------------------------------------------------------'
            print 'Env trade surplus mean: ' + str(np.mean(Budget))
            print '--------------------------------------------------------'
            print '--------------------------------------------------------'
            return [P_adj, Budget] 
        else:
            if planner: 
                series = ['W','SW','S','I','Z','P','E','Q', 'A', 'F1','F3','F1_tilde','F3_tilde', 'Profit', 'B', 'Budget']
                series = series + ['Q_low', 'Q_high', 'Q_env', 'Bhat', 'A_env', 'A_low', 'A_high'] 
                self.series = dict.fromkeys(series)
                
                self.XA = [np.vstack(d['XA0'] for d in datalist), np.vstack(d['XA1'] for d in datalist)]
                self.X1 = [np.vstack(d['X10'] for d in datalist), np.vstack(d['X11'] for d in datalist)]
                self.U = [np.hstack(d['U0'] for d in datalist), np.hstack(d['U1'] for d in datalist)]
                for x in self.series:
                    self.series[x] = np.vstack(d[x] for d in datalist)

                for m in range(2):
                    if env.turn_off == 1:
                        self.X1[m] = np.delete(self.X1[m], 2, axis=1)
                        self.XA[m] = np.delete(self.XA[m], 3, axis=1)
            else:
                if not(partial):
                    series = ['W','SW','S','I','Z','P','E','Q', 'A', 'F1','F3','F1_tilde','F3_tilde', 'Profit', 'B', 'Budget']
                    series = series + ['Q_low', 'Q_high', 'Q_env', 'Bhat', 'A_env', 'A_low', 'A_high', 'S_low', 'S_high', 'S_env', 'U_low', 'U_high' ] 
                    self.series = dict.fromkeys(series)
                    self.XA_e = [0,0] 
                    self.X1_e = [0,0] 
                    self.u_e = [0,0] 
                
                if stats:
                    for x in self.series:
                        self.series[x] = np.vstack(d[x] for d in datalist)
                else:
                    self.stack_user_samples(datalist, partial, ch7=True)
                    datalist2 = [d['1'] for d in datalist]
                    self.stack_user_samples(datalist2, partial=partial, summer=True, ch7=True)
                    
                    for m in range(2):
                        # Stack environmental samples
                        XA_e = np.vstack(d['XA_e' + str(m)] for d in datalist)
                        X1_e = np.vstack(d['X1_e'+ str(m)] for d in datalist)
                        u_e = np.hstack(d['u_e' + str(m)] for d in datalist)
                        if partial:
                            N1 = self.u_e[m].shape[0]
                            N2 = u_e.shape[0]
                            sample = np.random.choice(range(N1), size=N2, replace=False)
                            self.XA_e[m][sample, :] = XA_e 
                            self.X1_e[m][sample, :] = X1_e 
                            self.u_e[m][sample] = u_e
                        else: 
                            self.XA_e[m] = XA_e
                            self.X1_e[m] = X1_e
                            self.u_e[m] = u_e

            if stats:
                self.summary_stats(sample=0.5, percentiles=True, ch7=True)
            else:
                self.summary_stats(sample=0.5, percentiles=False, ch7=True)

            self.ITEROLD = self.ITER
            self.ITERNEW = self.ITERNEW + 1
            self.ITER = min(self.ITER + 1, self.ITERMAX - 1 ) 
            
            toc1 = time.time()

            print 'Processing results time: ' + str(toc1 - tic1)

            print ' ----- Summer ----- '
            print 'Storage mean: ' + str(np.mean(self.series['S'][:, 0]))
            print 'Inflow mean: ' + str(np.mean(self.series['I'][:, 0]))
            print 'Withdrawal mean: ' + str(np.mean(self.series['W'][:, 0]))
            print 'Welfare mean: ' + str(np.mean(self.series['SW'][:,0]))
            print 'Extraction mean: ' + str(np.mean(self.series['E'][:,0]))
            print 'Price mean: ' + str(np.mean(self.series['P'][:,0]))
            print 'Env trade surplus mean: ' + str(np.mean(self.series['Budget'][:,0]))

            print ' ----- Winter ----- '
            print 'Storage mean: ' + str(np.mean(self.series['S'][:, 1]))
            print 'Inflow mean: ' + str(np.mean(self.series['I'][:, 1]))
            print 'Withdrawal mean: ' + str(np.mean(self.series['W'][:, 1]))
            print 'Welfare mean: ' + str(np.mean(self.series['SW'][:,1]))
            print 'Extraction mean: ' + str(np.mean(self.series['E'][:,1]))
            print 'Price mean: ' + str(np.mean(self.series['P'][:,1]))
            print 'Env trade surplus mean: ' + str(np.mean(self.series['Budget'][:,1]))

            print ' ----- Annual ----- ' 
            print 'Storage mean: ' + str(self.stats['S']['Annual']['Mean'][self.ITEROLD])
            print 'Inflow mean: ' + str(self.stats['I']['Annual']['Mean'][self.ITEROLD])
            print 'Withdrawal mean: ' + str(self.stats['W']['Annual']['Mean'][self.ITEROLD])
            print 'Welfare mean: ' + str(self.stats['SW']['Annual']['Mean'][self.ITEROLD])
            print 'Extraction mean: ' + str(self.stats['E']['Annual']['Mean'][self.ITEROLD])
            print 'Price mean: ' + str(self.stats['P']['Annual']['Mean'][self.ITEROLD])
            print 'Env trade surplus mean: ' + str(self.stats['Budget']['Annual']['Mean'][self.ITEROLD])

            if self.ITER > 0:
                self.S[self.ITERNEW - 1] = np.mean(self.series['S'])
                self.W[self.ITERNEW - 1] = np.mean(self.series['W'][:, 1])
                self.E[self.ITERNEW - 1] = np.mean(self.series['E'][:, 0])
                self.B[self.ITERNEW - 1] = np.mean(self.series['Budget'])


            #if self.ITER > 2:
            #    pylab.plot(self.S[4:self.ITERNEW])
            #    pylab.plot(self.W[4:self.ITERNEW])
            #    pylab.plot(self.E[4:self.ITERNEW])
            #    pylab.show()
            #    pylab.plot(self.B[4:self.ITERNEW])
            #    pylab.show()
