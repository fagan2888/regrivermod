#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from __future__ import division
import numpy as np
import pylab
import time
import math
from regrivermod.storage cimport Storage
from libc.math cimport fmin as c_min
from libc.math cimport fmax as c_max
from libc.math cimport log as c_log
from libc.math cimport fabs as c_abs
from libc.math cimport cos as c_cos
from libc.stdlib cimport srand as c_seed
from libc.stdlib cimport rand
from libc.stdlib cimport RAND_MAX
cimport cython

cdef inline double c_rand() nogil:

    return rand() / (<double> RAND_MAX)

cdef inline double c_sum(int N, double[:] x) nogil:
    
    cdef double sumx = 0
    cdef int i = 0

    for i in range(N):
        sumx += x[i]

    return sumx

cdef inline double[:] storage_loss(int N, double L, double[:] s, double[:] w, double[:] loss, double[:] c_F, int ls):
    """User storage loss deduction"""

    cdef int i = 0
    cdef double temp_sum_s = 0

    if ls == 0:
        for i in range(N):
            loss[i] = L * c_F[i]
    else:
        for i in range(N):
            temp_sum_s += s[i]

        if temp_sum_s > 0:
            for i in range(N):
                loss[i] =  L * (s[i] / temp_sum_s) 
        else:
            for i in range(N):
                loss[i] = 0 

    return loss

cdef inline double[:] calc_x(int N, double[:] s, double[:] acc_max, double[:] c_F, double[:] x, double diff, double[:] J):

    cdef int i = 0
    cdef double sumc_F = 0

    for i in range(N):
        J[i] = 0

    if diff > 0:
        for i in range(N):
            if s[i] < acc_max[i]:
                sumc_F += c_F[i]
                J[i] = 1
    if diff < 0:
        for i in range(N):
            if s[i] > 0:
                sumc_F += c_F[i]
                J[i] = 1

    for i in range(N):
        if J[i] == 1:
            x[i] += (c_F[i] / sumc_F ) * diff

    return x

cdef inline double[:] update_accounts(int N, double[:] s, double[:] w, double[:] l, double[:] x, double[:] acc_max, double[:] lamI, double[:] temp):

    cdef int i
    cdef double[:] values = temp
    
    for i in range(N):
        values[i] = c_max(c_min(s[i] - w[i] - l[i] + lamI[i] + x[i], acc_max[i]), 0)

    return values

cdef class Utility:

    def __init__(self, users, storage, para, ch7=False, env=0):
        
        self.N = users.N          
        self.N_low = users.N_low
        self.N_high = users.N_high

        self.c_pi = math.pi

        if ch7:
            self.ch7 = 1
            self.I_env = self.N
            self.N +=  1
            if para.ch7['High']:
                self.N_high += 1
            else:
                self.N_low += 1
        else:
            self.ch7 = 0
        
        self.I_low = np.zeros(self.N_low, dtype='int32')
        self.I_high = np.zeros(self.N_high, dtype='int32')

        cdef int i
        
        if not(ch7) or (ch7 and para.ch7['High']):
            for i in range(self.N_low):
                self.I_low[i] = i
            for i in range(self.N_high):
                self.I_high[i]  = i + self.N_low
        else:
            for i in range(self.N_low):
                self.I_low[i] = i
            for i in range(self.N_high):
                self.I_high[i]  = i + self.N_low
            self.I_low[self.N_low - 1] = self.N - 1

        self.K = storage.K
        self.a = np.zeros(self.N)
        self.s = np.zeros(self.N)
        self.x = np.zeros(self.N)
        self.l = np.ones(self.N)
        self.J = np.zeros(self.N)
        self.temp = np.zeros(self.N)
        self.temp2 = np.zeros(self.N)
        self.temp3 = np.zeros(self.N)
        self.temp4 = np.zeros(self.N)
        self.temp5 = np.zeros(self.N)
        self.c_F = np.zeros(self.N)
        self.c_K = np.zeros(self.N)
        self.w = np.zeros(self.N)

        if ch7:
            self.delta_a = storage.delta_a
            self.delta_Ea = storage.delta_Ea
            self.delta1b = storage.delta_Eb
            for i in range(self.N - 1):
                self.c_F[i] = users.c_F[i] 
                self.c_K[i] = users.c_K[i] 
            self.c_F[self.N - 1] = env.Lambda_I
            self.c_K[self.N - 1] = env.Lambda_K
            self.fixed_loss = (storage.delta_a[0] + storage.delta_Ea) / ( 1 - storage.delta_Eb)
        else:
            self.c_F = users.c_F                        # User inflow shares
            self.c_K = users.c_K                        # User capacity shares
            self.delta1a = storage.delta1a
            self.delta1b = storage.delta1b
            self.fixed_loss = self.delta1a / (1 - self.delta1b)
            self.fixed_loss_co = 0
        

        for i in range(self.N):
            self.s[i] = (storage.S - self.fixed_loss) * self.c_F[i]      # Initialize user accounts
        
        self.acc_max = np.zeros(self.N)
        if para.sr == 'CS':
            for i in range(self.N):
                self.acc_max[i] = self.c_K[i] * (storage.K - self.fixed_loss)
            self.sr = 0
        elif para.sr == 'SWA':
            for i in range(self.N):
                self.acc_max[i] = storage.K - self.fixed_loss
            self.sr = 1
        elif para.sr == 'OA':
            for i in range(self.N):
                self.acc_max[i] = storage.K - self.fixed_loss
            self.sr = 2
        elif para.sr == 'NS':
            for i in range(self.N):
                self.acc_max[i] = storage.K - self.fixed_loss
            self.sr = 3
        elif para.sr == 'CS-SWA':
            for i in range(self.N):
                self.acc_max[i] = self.c_K[i] * (storage.K - self.fixed_loss)
            self.sr = 4
        elif para.sr == 'RS':
            self.sr = -1
       
        self.ls = para.ls            # Loss deduction type
        self.HL = para.HL            # Priority or proportional
        if ch7:
            if self.sr == -1:
                self.A_bar = c_max((self.K - storage.loss12(self.K, 0)) * (1 - storage.delta_Eb) - storage.delta_Ea, 0)
            else:
                self.A_bar = self.K - self.fixed_loss
        else:
            if self.sr == -1:
                self.A_bar = (self.K*(1 - self.delta1b) - self.delta1a)
            else:
                self.A_bar = (self.K - self.fixed_loss)

        self.Lambda_high = para.Lambda_high
        self.M = 0

        self.state_zero = np.zeros(3)
        self.explore = 0

    cdef double[:] allocate(self, double A, double[:] a) nogil:
        
        cdef int i = 0 

        if self.HL == 0:
            for i in range(self.N):
                a[i] = A * self.c_F[i]
        else:
            for i in range(self.N_high):
                a[self.I_high[i]] =  c_min(A, self.A_bar * self.Lambda_high) * (self.c_F[self.I_high[i]] / self.Lambda_high)
            for i in range(self.N_low):
                a[self.I_low[i]] = c_max(A - self.A_bar * self.Lambda_high, 0) * (self.c_F[self.I_low[i]] / (1 - self.Lambda_high))

        return a

    def seed(self, i):
        seed = int(time.time()/(i + 1))
        c_seed(seed)

    def set_shares(self, Lambda_high, users, env=0):

        self.Lambda_high = Lambda_high

        if self.ch7 == 1:
            self.c_F = np.array([users.c_F + env.c_F])
            self.c_K = np.array([users.c_K + env.c_K])
        else:
            self.c_F = users.c_F                        # User inflow shares
            self.c_K = users.c_K                        # User capacity shares

        for i in range(self.N):
            self.s[i] = (self.K - self.fixed_loss) * self.c_F[i]      # Initialize user accounts

        self.acc_max = np.zeros(self.N)
        if self.sr == 0:
            for i in range(self.N):
                self.acc_max[i] = self.c_K[i] * (self.K  - self.fixed_loss)
        elif self.sr == 1:
            for i in range(self.N):
                self.acc_max[i] = self.K - self.fixed_loss
        elif self.sr == 2:
            for i in range(self.N):
                self.acc_max[i] = self.K - self.fixed_loss
        elif self.sr == 3:
            for i in range(self.N):
                self.acc_max[i] = self.K - self.fixed_loss
        elif self.sr == 4:
            for i in range(self.N):
                self.acc_max[i] = self.c_K[i] * (self.K - self.fixed_loss)



    cdef double release(self, double[:] w, double S):

        cdef double W = 0

        W = c_sum(self.N, w)

        if W > 0.001:
            W += self.fixed_loss
            self.delivered = 1
        else:
            W = 0
            self.delivered = 0

        W = c_min(c_max(W, 0), S)

        # User allocations (adjusted for delivery losses)
        for i in range(self.N):
            self.a[i] = w[i] * (1 - self.delta1b)

        return W

    def init_policy(self, Tilecode W_f, Storage storage, para):

        cdef int i, N = 100000
        cdef double[:] state = np.zeros(2)
        cdef double I, S
        cdef double[:,:] X = np.zeros([N, 3])
        cdef double[:] W = np.zeros(N)

        for i in range(N):

            X[i, 0] = c_rand() * storage.K
            X[i, 1] = c_rand() * storage.Imax  * (storage.I_bar**-1)
            if c_rand() < 0.5:
                X[i, 2] = 0
            else:
                X[i, 2] = 1

            state[0] = X[i, 0]
            state[1] = X[i, 1]

            W[i] = c_min(c_max(W_f.one_value(state), 0), X[i, 0])

        self.policy = Tilecode(3, [10, 10, 2], 13, mem_max=1, lin_spline=True, linT=para.linT, cores=para.CPU_CORES)

        self.policy.fit(X, W)

    cdef double withdraw_ch7(self, double S, double I, int M):

        cdef double[:] state = self.state_zero
        cdef double U, V, Z, W, A

        state[0] = S
        state[1] = I
        state[2] = M

        W = c_max(c_min(self.policy.one_value(state), S), 0)

        if self.explore == 1:
            if self.d == 0:
                W = c_rand() * S
            else:
                U = c_rand()
                V = c_rand()
                Z = ((-2 * c_log(U))**0.5)*c_cos(2*self.c_pi*V)
                W = c_min(c_max(Z * (self.d * S) + W, 0), S)

        self.A = c_max((W - self.fixed_loss) * (1 - self.delta1b), 0)

        self.max_E = c_max(W - self.delta_a[M], 0)

        self.a = self.allocate(self.A, self.a)

        return W

    cdef double deliver_ch7(self, double[:] users_w, double env_w, Storage storage,  int M):

        cdef double W = 0
        cdef int i = 0
         
        for i in range(self.N - 1):
            self.w[i] = users_w[i]
            W += users_w[i]

        if W > 0:
            self.delivered = 1
        else:
            self.delivered = 0
        
        self.w[self.I_env] = env_w
        W += env_w
        
        self.max_E = storage.delta_Ea * self.delivered + W
        
        # User allocations (adjusted for delivery losses)
        for i in range(self.N):
            self.a[i] = self.w[i] * (1 - storage.delta_Eb)

        self.A = c_sum(self.N, self.a)

        self.M = M

        # Physical withdrawals to satisfy orders
        if self.delivered == 1 and M == 0:
            W += self.fixed_loss
        elif self.delivered == 0 and M == 0:
            W += storage.delta_a[0]
        elif M == 1 and storage.S > self.fixed_loss:
            W = W + storage.delta_a[1]
        elif storage.S < self.fixed_loss:
            W = 0
        
        return W
    
    cdef double extract(self, double qe):

        cdef double E

        E = c_max(self.max_E - qe / (1 - self.delta1b), 0)

        return E

    cdef void update(self, double S, double I, double L, double Spill, double[:] w, double A):
    
        if self.sr == -1:       # Release sharing (planner storage)
            self.a = self.allocate(A, self.a)
        else:                   # Storage rights 
            self.update_storage_accounts(S, I, L, Spill, w)

    def update_test(self, S, I, L, Spill, w,  A):

        if self.sr == -1:       # Release sharing (planner storage)
            self.a = self.allocate(A, self.a)
        else:                   # Storage rights
            self.update_storage_accounts_test(S, I, L, Spill, w)

    cdef void update_storage_accounts(self, double S, double I, double L, double Spill, double[:] w):

        cdef int i = 0                           # user index
        cdef double diff = 10                    # difference between sum of accounts and target
        cdef double sum_s = 0                    # sum of user accounts
        cdef double target                       # target for sum of accounts (Actual storage less fixed losses)
        cdef double f_loss_ded = 0               # Fixed delivery loss deduction 
        cdef double[:] s = self.temp             # Storage account balance
        cdef double[:] s0 = self.temp5
        cdef double[:] lamI = self.temp2         # User inflow credits
        cdef double[:] lamI_Spill = self.temp3   # User inflow credits (before spills)
        cdef double[:] s_k = self.temp4          # User account balance less storage share (for CS-SWA)
        cdef double sum_s_k = 0                  # Sum of above

        cdef double tol = 0.001
        cdef int it = 0         

        # Initialize arrays
        for i in range(self.N):
            s[i] = 0
            s0[i] = 0
            self.x[i] = 0
            self.l[i] = 0
            lamI[i] = 0
            lamI_Spill[i] = 0
            s_k[i] = 0
        
        if S > self.fixed_loss:   # Normal case: enough water to meet fixed losses
            
            # Fixed delivery loss deductions
            target = S - self.fixed_loss
            if self.M == 0:
                if self.delivered == 1:
                    f_loss_ded = self.fixed_loss - self.fixed_loss_co
                    self.fixed_loss_co = 0
                else:
                    f_loss_ded = 0
                    self.fixed_loss_co = self.fixed_loss
                    if self.ch7 == 1:
                        f_loss_ded = self.fixed_loss - self.delta_Ea
                        self.fixed_loss_co = self.delta_Ea
            if self.M == 1:
                f_loss_ded = self.delta_a[1] - self.fixed_loss_co
                self.fixed_loss_co = self.fixed_loss - self.delta_a[1] * 2

            # User inflow shares = lambda_i * I_t (unless HL=1)
            lamI = self.allocate(I, lamI)
            
            # User evaporation loss deductions (socialised or proportional to account balance)
            self.l = storage_loss(self.N, L, self.s, w, self.l, self.c_F, self.ls)

            # Add fixed delivery loss deductions (by inflow shares)
            for i in range(self.N):
                self.l[i] += f_loss_ded * self.c_F[i]

            s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
            s0[...] = s

            # Apply spill event rules
            if Spill > 0:
                if self.sr == 0:                    # CS
                    for i in range(self.N):
                        s[i] = self.c_K[i] * target
                elif self.sr == 1:                  # SWA
                    lamI_Spill  = self.allocate(I - Spill, lamI)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI_Spill, s)
                    for i in range(self.N):
                        self.x[i] = -Spill * (s[i] / (self.K - self.fixed_loss))
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                elif self.sr == 2:                  # OA
                    for i in range(self.N):
                        self.x[i] = -Spill * self.c_F[i]
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                elif self.sr == 4:                  # CS-SWA
                    lamI_Spill  = self.allocate(I - Spill, lamI)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI_Spill, s)
                    for i in range(self.N):
                        s_k[i] = c_max(s[i] - self.acc_max[i], 0)
                        sum_s_k += s_k[i]
                    for i in range(self.N):
                        self.x[i] = -1*c_min(Spill, sum_s_k) * (s_k[i] / sum_s_k)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)

            if self.sr == 3:                        # NS
                for i in range(self.N):
                    self.s[i] = target * self.c_F[i]
                    self.x[i] = self.s[i] - s[i]
                    s[i] = self.s[i]

            # Prepare for final reconciliation - delete overhang

            diff = target - sum_s

            for i in range(self.N):
                s0[i] = self.s[i] - w[i] - self.l[i] + lamI[i]
                if s0[i] < 0:
                    self.x[i] = -s0[i]
                if s0[i] > self.acc_max[i]:
                    self.x[i] = self.acc_max[i] - s0[i]

            # Final reconciliation (including any internal spills)
            while 1:
                sum_s = c_sum(self.N, s)
                diff = (target - sum_s)

                if c_abs(diff) < tol or it >= 100:
                    break

                self.x = calc_x(self.N, s, self.acc_max, self.c_F, self.x, diff, self.J)
                s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                it += 1

            if it == 100:
                print 'Accounts failed to reconcile!'
                """
                print diff
                print np.sum(s)
                print 'target: ' + str(target)
                print 'w sum' + str(np.array(w))
                print 's0 sum' + str(np.array(s0))
                print 'I sum' + str(np.array(lamI))
                print 'Spill' + str(Spill)
                print 'self.s sum' + str(np.array(self.s))
                print 'acc max sum' + str(np.array(self.acc_max))
                print np.array(er)
                print 's sum: ' + str(np.array(s))
                print 'S: ' + str(S)
                print 'I: ' + str(I)
                print 'x: ' + str(np.array(self.x))
                print 'J: ' + str(np.array(self.J))
                print 'f_loss_ded: ' + str(f_loss_ded)
                print 'fixed_loss: ' + str(self.fixed_loss)
                print 'loss: ' + str(np.array(self.l))
                """

            # Update user accounts
            for i in range(self.N):
                self.s[i] = s[i]
                self.x[i] = s[i] - s0[i]

        else:   # Extreme drought: not enough water to meet current fixed losses
            
            self.fixed_loss_co = S      # All water is put aside to meet next period fixed losses
            for i in range(self.N):     # All user accounts are zero
                self.x[i] = self.s[i]
                self.s[i] = 0
                self.a[i] = 0


    def update_storage_accounts_test(self, double S, double I, double L, double Spill, double[:] w):

        cdef int i = 0                           # user index
        cdef double diff = 10                    # difference between sum of accounts and target
        cdef double sum_s = 0                    # sum of user accounts
        cdef double target                       # target for sum of accounts (Actual storage less fixed losses)
        cdef double f_loss_ded = 0               # Fixed delivery loss deduction 
        cdef double[:] s = self.temp             # Storage account balance
        cdef double[:] lamI = self.temp2         # User inflow credits
        cdef double[:] lamI_Spill = self.temp3   # User inflow credits (before spills)
        cdef double[:] s_k = self.temp4          # User account balance less storage share (for CS-SWA)
        cdef double sum_s_k = 0                  # Sum of above
        
        cdef double tol = 0.001
        cdef int it = 0         
        
        # Initialize arrays
        for i in range(self.N):
            s[i] = 0
            self.x[i] = 0
            self.l[i] = 0
            lamI[i] = 0
            lamI_Spill[i] = 0
            s_k[i] = 0
        
        if S > self.fixed_loss:   # Normal case: enough water to meet fixed losses
            print 'S > fixed loss'
            print ' ---------- '
            # Fixed delivery loss deductions
            target = S - self.fixed_loss
            f_loss_ded = (self.fixed_loss - self.fixed_loss_co) * self.delivered 
            self.fixed_loss_co = 0
            print 'target: ' + str(target) 
            print 'f_loss_ded: ' + str(f_loss_ded) 
            print 'self.fixed_loss_co: ' + str(self.fixed_loss_co) 
            print ' ---------- '
            # User inflow shares = lambda_i * I_t (unless HL=1)
            lamI = self.allocate(I, lamI)
            print 'lamI: ' + str(np.array(lamI)) 
            print 'lamI: sum ' + str(np.sum(lamI))

            # User evaporation loss deductions (socialised or proportional to account balance)
            self.l = storage_loss(self.N, L, self.s, w, self.l, self.c_F, self.ls) 
            print 'loss: ' + str(np.array(self.l)) 
            print ' ---------- '
            for i in range(self.N):
                # Add fixed delivery loss deductions (by inflow shares)
                self.l[i] += f_loss_ded * self.c_F[i]
            print 'loss: ' + str(np.array(self.l)) 
            print ' ---------- '
            print 'x: ' + str(np.array(self.x))
            print 'acc_max: ' + str(np.array(self.acc_max)) 
            print 's: ' + str(np.array(self.s)) 
            s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
            print 's: ' + str(np.array(s)) 
            # Apply spill event rules
            if Spill > 0:
                print '--- Spill ---'
                if self.sr == 0:                    # CS
                    for i in range(self.N):
                        s[i] = self.c_K[i] * target
                elif self.sr == 1:                  # SWA
                    lamI_Spill  = self.allocate(I - Spill, lamI)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI_Spill, s) 
                    for i in range(self.N):
                        self.x[i] = -Spill * (s[i] / (self.K - self.fixed_loss))
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s) 
                elif self.sr == 2:                  # OA
                    for i in range(self.N):
                        self.x[i] = -Spill * self.c_F[i] 
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                elif self.sr == 4:                  # CS-SWA
                    lamI_Spill  = self.allocate(I - Spill, lamI)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI_Spill, s) 
                    for i in range(self.N):
                        s_k[i] = c_max(s[i] - self.acc_max[i], 0)
                        sum_s_k += s_k[i]
                    for i in range(self.N):
                        self.x[i] = -1*c_min(Spill, sum_s_k) * (s_k[i] / sum_s_k)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s) 

            if self.sr == 3:                        # NS
                for i in range(self.N):
                    self.s[i] = target * self.c_F[i]
                    self.x[i] = self.s[i] - s[i]
                    s[i] = self.s[i]

            # Final reconciliation (including any internal spills)
            while c_abs(diff) > tol and it < 100:
                sum_s = c_sum(self.N, s)
                diff = (target - sum_s)
                print diff
                print sum_s
                self.x = calc_x(self.N, s, self.acc_max, self.c_F, self.x, diff, self.J)
                s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                it += 1

            if it == 100:
                print 'Accounts failed to reconcile!'
                print 's: ' + str(np.array(s))
                self.it = 1
            else:
                self.it = 0
            
            # Update user accounts
            for i in range(self.N):
                self.s[i] = s[i]

        else:   # Extreme drought: not enough water to meet current fixed losses
            
            self.fixed_loss_co = S      # All water is put aside to meet next period fixed losses
            for i in range(self.N):     # All user accounts are zero
                self.x[i] = self.s[i]
                self.s[i] = 0
                self.a[i] = 0

        if c_sum(self.N, self.s) > S:
            print 'S: ' + str(S)
            print 'I: ' + str(I)
            print 's sum : ' + str(np.sum(self.s))
            print 's sum : ' + str(sum_s)
            print 'target: ' + str(self.target)
            print 'fl: ' + str(self.fixed_loss)
            raise NameError('UpSh*tCreekB')

