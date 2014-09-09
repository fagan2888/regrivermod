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
from econlearn.tilecode cimport Tilecode

cdef inline double c_sum(int N, double[:] x):
    
    cdef double sumx = 0
    cdef int i = 0

    for i in range(N):
        sumx += x[i]

    return sumx

cdef extern from "math.h":
    double c_fmax "fmax" (double, double)

cdef extern from "math.h":
    double c_fmin "fmin" (double, double)

cdef extern from "stdlib.h":
    double c_rand "drand48" ()

cdef extern from "stdlib.h":
    void c_seed "srand48" (int)

cdef extern from "fast_floattoint.h":
    int c_int "Real2Int" (double)

cdef extern from "math.h":
    double c_log "log" (double)

cdef extern from "math.h":
    double c_exp "exp" (double)

cdef extern from "math.h":
    double c_sin "sin" (double)

cdef extern from "math.h":
    double c_cos "cos" (double)

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
    cdef double[:] W_low_sim = np.zeros(T)
    cdef double[:] W_high_sim = np.zeros(T)
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

    # Run simulation
    for t in range(T):
        
        # Initial state
        S_sim[t] = storage.S
        I_sim[t] = storage.I
        Z_sim[t] = storage.Spill 
        I_tilde = I_sim[t] * (storage.I_bar**-1)
        
        # Users make withdrawals 
        users.withdraw(storage.S, utility.s, I_tilde)
        
        W_sim[t] = utility.release(users.w, storage.S)
        users.a[...] = utility.a
        
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
        
        if stats == 1:
            users.user_stats(utility.s, utility.x)
            U_low_sim[t] = users.U_low
            U_high_sim[t] = users.U_high
            W_low_sim[t] = users.W_low
            W_high_sim[t] = users.W_high
            S_low_sim[t] = users.S_low
            S_high_sim[t] = users.S_high
            X_low_sim[t] = users.X_low
            X_high_sim[t] = users.X_high
        
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
        data = {'W': np.asarray(W_sim), 'SW': np.asarray(SW_sim), 'S': np.asarray(S_sim), 'I': np.asarray(I_sim),  'Z': np.asarray(Z_sim), 'U_low' : np.asarray(U_low_sim) , 'U_high' : np.asarray(U_high_sim), 'W_low' : np.asarray(W_low_sim) , 'W_high' : np.asarray(W_high_sim), 'S_low' : np.asarray(S_low_sim) , 'S_high' : np.asarray(S_high_sim), 'X_low' : np.asarray(X_low_sim) , 'X_high' : np.asarray(X_high_sim), 'P' : np.asarray(P_sim),  'Q' : np.asarray(Q_sim)}
    else:
        if record_state == 1:
            data = {'XA_l': np.asarray(XA_l), 'X1_l' : np.asarray(X1_l), 'u_t_l' : np.asarray(u_t_l), 'XA_h': np.asarray(XA_h), 'X1_h' : np.asarray(X1_h), 'u_t_h' : np.asarray(u_t_h), 'W': np.asarray(W_sim), 'SW': np.asarray(SW_sim), 'S': np.asarray(S_sim), 'I': np.asarray(I_sim),  'Z': np.asarray(Z_sim), 'P' : np.asarray(P_sim), 'Q' : np.asarray(Q_sim)}
        else:
            data = {'test' : np.asarray(test_payoff)}    
    
    if multi:
        que.put(data)
    else:
        return data 


def run_planner_sim(int job, int T, Users users, Storage storage, Utility utility, Tilecode policy, double delta,  int planner, int explore, q, seed, multi = True, myopic = False):

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
    cdef double c_pi = math.pi
    cdef int i
    if delta > 0:
        finetune = 1

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
                #ran = c_rand()
                #w_min = c_fmax(w_star - delta * storage.S, 0)
                #w_max = c_fmin(w_star + delta * storage.S, storage.S)
                #for i in range(100000):
                U = c_rand()
                V = c_rand()
                Z = ((-2 * c_log(U))**0.5)*c_cos(2*c_pi*V)
                W_sim[t] = c_fmin(c_fmax(Z * (delta * storage.S) + w_star, 0), storage.S)
                #    if W_sim[t] < storage.S and W_sim[t] > 0:
                #        break
                #    Z = ((-2 * c_log(U))**0.5)*c_sin(2*c_pi*V)
                #    W_sim[t] = Z * (delta * storage.S) + w_star
                #    if W_sim[t] < storage.S and W_sim[t] > 0:
                #        break

                ##########  Record state and action pairs [W, S, I,] ###########
                XA[t, 0] = W_sim[t]
                XA[t, 1] = S_sim[t]
                XA[t, 2] = I_tilde
                ###############################################################
        else:
            if useall == 1:
                W_sim[t] = S_sim[t]*1.0
            else:
                state[0] = storage.S
                state[1] = I_tilde
                W_sim[t] = c_fmax(c_fmin(policy.one_value(state), S_sim[t]), 0)

        # Water delivered
        Q_sim[t] = storage.release(W_sim[t])

        # Make water allocations
        utility.update(S_sim[t], I_sim[t], storage.Loss, storage.Spill, zeros, Q_sim[t])
        users.a[...] = utility.a

        # Spot market opens
        P_sim[t] = users.clear_market(I_tilde, perf_market, planner)
        SW_sim[t] = users.consume(P_sim[t], I_tilde, planner)
        
        U_low_sim[t] = users.U_low
        U_high_sim[t] = users.U_high
        

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
            'I': np.asarray(I_sim), 'P': np.asarray(P_sim), 'Z': np.asarray(Z_sim), 'U_low' : np.asarray(U_low_sim), 'U_high' : np.asarray(U_high_sim), 'Q' : np.asarray(Q_sim)} 

    if multi:
        q.put(data)
    else:
        return data

class Simulation:

    "Decentralised model simulation class"

    def __init__(self, para):
       
        self.sample_rate = para.sample_rate

        self.N = para.N                       # Number of users
        self.beta = para.beta                 # Discount rate

        self.CPUs = para.CPU_CORES
        
        ####################     Result containers

        # Summary stats
        self.stats = {}
        names  = ['Mean', 'SD', '25th', '75th', '2.5th', '97.5th', 'Min', 'Max']
        formats = ['f4','f4','f4','f4','f4','f4','f4','f4']
        self.stats['S'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['W'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['I'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['SW'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['Z'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['U_low'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['U_high'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['W_low'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['W_high'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['S_low'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['S_high'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['X_low'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['X_high'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['P'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        self.stats['Q'] = np.zeros(para.ITER2 + 4, dtype={'names':names, 'formats':formats})
        
        # Data series
        self.series = {'S' : 0, 'W' : 0, 'I' : 0, 'SW' : 0, 'Z' : 0, 'P' : 0, 'Q' : 0} 
        self.series_old = {'S' : 0, 'W' : 0, 'I' : 0, 'SW' : 0, 'Z' : 0, 'P' : 0, 'Q' : 0} 
        self.full_series = {'S' : 0, 'W' : 0, 'I' : 0, 'SW' : 0, 'Z' : 0, 'U_low' : 0, 'U_high' : 0, 'W_low' : 0, 'W_high' : 0, 'S_low': 0, 'S_high' : 0, 'X_low': 0, 'X_high' : 0, 'P' : 0, 'Q' : 0} 
        self.p_series = {'S' : 0, 'W' : 0, 'I' : 0, 'SW' : 0, 'Z' : 0, 'U_low' : 0, 'U_high' : 0, 'Q' : 0} 
        
        self.ITER = 0
        self.S = np.zeros(para.ITER2 + 4)

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
            users.a[...] = utility.a
            print 'Withdrawals: ' + str(W_sim[t])
            print 'User allocations: ' + str(np.array(users.a))
            
            # Water is delivered and spot market opens
            P_sim[t] = users.clear_market(I_tilde, market_d, 0)
            SW_sim[t] = users.consume(P_sim[t], I_tilde, 0)
            
            print 'Price: ' + str(P_sim[t])
            print 'Welfare: ' + str(SW_sim[t])

            ### State transition ###

            # New Inflows
            storage.update(W_sim[t], t)

            print 'Storage: ' + str(S_sim[t])
            print 'Inflow: ' + str(I_sim[t])
            print 'Spill: ' + str(Z_sim[t])
            # User accounts are updated
            utility.update_test(storage.S, storage.I, storage.Loss, storage.Spill, users.w, 0)

            # New productivity shocks
            users.update()

            import pdb; pdb.set_trace()

    def stack_sims(self, data, planner = False, stats=False, solve_planner = True, testing=False, partial=False):
        """
        Combine simulation data from multiple processes
        """
        
        if planner:
            if solve_planner: 
                self.XA_t = np.vstack(d['XA'] for d in data)
                self.X_t1 = np.vstack(d['X1'] for d in data)
                self.series = self.p_series
            
            for x in self.series:
                self.series[x] = np.hstack(d[x] for d in data)  
            self.T = len(self.series['S'])
            self.summary_stats(sample = 1, percentiles = stats)
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
                    for x in self.series:
                        self.series[x] = np.hstack(d[x] for d in data)  
                    
                    group = ['_l', '_h']
                    if not(partial):
                        self.XA_t = [0,0]       # State action samples
                        self.X_t1 = [0,0]       # State transition samples
                        self.u_t = [0,0]        # Payoff samples
                    for h in range(2):
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
                            self.XA_t[h][sample, :] = XA_t 
                            self.X_t1[h][sample, :] = X_t1 
                            self.u_t[h][sample] = u_t
                        else: 
                            self.XA_t[h] = XA_t 
                            self.X_t1[h] = X_t1 
                            self.u_t[h] = u_t

                self.T = len(self.series['S'])
                self.summary_stats(sample = 1, percentiles=stats)
            self.ITER +=1 
    
    def summary_stats(self, sample = 0.5, percentiles=False):
        
        tic = time.time()
        N = int(sample * self.T)
        
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
        toc = time.time()
        print 'Summary stats time ' + str(toc - tic)

    
    def simulate(self, users, storage, utility, T, num_process, planner = False, partial = False, seed = 0, policy = False, polf = 0, delta = 0, stats = False, planner_explore = True, t_cost_off = False, myopic=False):
        
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

            args = [(i, T, users, storage, utility, polf, delta, tc_off, explore, ques[i], seed, True, myopic) for i in range(num_process)]
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
        
        if self.ITER <=2:
            self.S[self.ITER - 1] = np.mean(self.series['S'])
        else:
            self.S[self.ITER - 1] = np.mean(self.series['S']) * self.sample_rate + (1 - self.sample_rate) * self.S[self.ITER - 2]

        if self.ITER > 1:
            pylab.plot(self.S[0:self.ITER])
            pylab.show()

