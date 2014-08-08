#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from __future__ import division
import numpy as np
import pylab
cimport numpy as np
import time
cimport cython


cdef extern from "math.h":
    double c_min "fmin" (double, double)

cdef extern from "math.h":
    double c_max "fmax" (double, double)

cdef extern from "math.h":
    double c_log "log" (double)

cdef inline double c_sum(int N, double[:] x):
    
    cdef double sumx = 0
    cdef int i = 0

    for i in range(N):
        sumx += x[i]

    return sumx

cdef inline double[:] storage_loss(int N, double L, double[:] s, double[:] w, double[:] loss, double[:] c_F, int ls):
    "User storage loss deduction"

    cdef int i = 0
    cdef double temp_sum_s = 0
    
    if ls == 0:
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

cdef inline double[:] calc_x(int N, double[:] s, double[:] acc_max, double[:] c_F, double[:] x, double diff, double[:] J, double[:] minx, double[:] maxx):

    cdef int i = 0
    cdef double sumc_F = 0

    if diff > 0:
        for i in range(N):
            if s[i] < acc_max[i]: 
                sumc_F += c_F[i]
                J[i] = 1
                maxx[i] = acc_max[i] - s[i]
            else:
                J[i] = 0
                maxx[i] = 0
    if diff < 0:
        for i in range(N):
            if s[i] > 0: 
                sumc_F += c_F[i]
                J[i] = 1
                minx[i] = 0 - s[i]
            else:
                J[i] = 0
                minx[i] = 0

    for i in range(N):
        if J[i] == 1:
            x[i] += c_max(c_min((c_F[i] / sumc_F ) * diff, maxx[i]), minx[i])

    return x

cdef inline double[:] update_accounts(int N, double[:] s, double[:] w, double[:] l, double[:] x, double[:] acc_max, double[:] lamI, double[:] temp):

    cdef int i
    cdef double[:] values = temp
    
    for i in range(N):
        values[i] = c_max(c_min(s[i] - w[i]- l[i] + lamI[i] + x[i], acc_max[i]), 0)

    return values

cdef inline double[:] allocate(int N, int N_low, double A, int HL, double[:] c_F, double Lambda_high, double A_bar, double[:] a):
    
    cdef int i = 0 

    if HL == 0:
        for i in range(N):
            a[i] = A * c_F[i]
    else:
        for i in range(N_low, N):
            a[i] =  c_min(A, A_bar * Lambda_high) * (c_F[i] / Lambda_high)
        for i in range(0, N_low):
            a[i] = c_max(A - A_bar * Lambda_high, 0) * ( c_F[i] / (1 - Lambda_high))

    return a


cdef class Utility:

    def __init__(self, users, storage, para):

        self.N = users.N          
        self.N_low = users.N_low
        self.N_high = users.N_high
        self.I_low = np.zeros(self.N_low, dtype='int32')
        self.I_high = np.zeros(self.N_high, dtype='int32')

        cdef int i
        for i in range(self.N_low):
            self.I_low[i] = i

        for i in range(self.N_high):
            self.I_high[i]  = i + self.N_low

        self.K = storage.K
        self.a = np.zeros(self.N)
        self.s = np.zeros(self.N)
        self.x = np.zeros(self.N)
        self.maxx = np.zeros(self.N)
        self.minx = np.zeros(self.N)
        self.l = np.ones(self.N)
        self.J = np.zeros(self.N)
        self.temp = np.zeros(self.N)
        self.temp2 = np.zeros(self.N)
        self.temp3 = np.zeros(self.N)
        self.temp4 = np.zeros(self.N)
        self.c_F = users.c_F                        # User inflow shares
        self.c_K = users.c_K                        # User capacity shares
        self.delta1a = storage.delta1a
        self.delta1b = storage.delta1b
        self.fixed_loss = self.delta1a
        self.fixed_loss_co = 0

        for i in range(self.N):
            self.s[i] = (storage.S - self.fixed_loss) * self.c_F[i]      # Initialize user accounts
        
        self.acc_max = np.zeros(self.N)
        if para.sr == 'CS':
            for i in range(self.N):
                self.acc_max[i] = self.c_K[i] * (storage.K - self.delta1a)
            self.sr = 0
        elif para.sr == 'SWA':
            for i in range(self.N):
                self.acc_max[i] = storage.K - self.delta1a
            self.sr = 1
        elif para.sr == 'OA':
            for i in range(self.N):
                self.acc_max[i] = storage.K - self.delta1a
            self.sr = 2
        elif para.sr == 'NS':
            for i in range(self.N):
                self.acc_max[i] = storage.K - self.delta1a
            self.sr = 3
        elif para.sr == 'CS-SWA':
            for i in range(self.N):
                self.acc_max[i] = self.c_K[i] * (storage.K - self.delta1a)
            self.sr = 4
        elif para.sr == 'RS':
            self.sr = -1

        self.ls = para.ls            # Loss deduction type
        self.HL = para.HL            # Priority or proportional
        self.A_bar = (self.K*(1 - self.delta1b) - self.delta1a)
        self.Lambda_high = para.Lambda_high

    def set_shares(self, Lambda_high, users):

        self.Lambda_high = Lambda_high

        self.c_F = users.c_F                            # User capacity shares
        self.c_K = users.c_K                            # User capacity shares

        for i in range(self.N):
            self.s[i] = (self.K - self.fixed_loss) * self.c_F[i]      # Initialize user accounts

        self.acc_max = np.zeros(self.N)
        if self.sr == 0:
            for i in range(self.N):
                self.acc_max[i] = self.c_K[i] * self.K * (1 - self.delta1a)
        elif self.sr == 1:
            for i in range(self.N):
                self.acc_max[i] = self.K * (1 - self.delta1a)
        elif self.sr == 2:
            for i in range(self.N):
                self.acc_max[i] = self.K * (1 - self.delta1a)
        elif self.sr == 3:
            for i in range(self.N):
                self.acc_max[i] = self.K * (1 - self.delta1a)
        elif self.sr == 4:
            for i in range(self.N):
                self.acc_max[i] = self.c_K[i] * self.K * (1 - self.delta1a)

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
    
    cdef void update(self, double S, double I, double L, double Spill, double[:] w, double A):
    
        if self.sr == -1:       # Release sharing (planner storage)
            self.a = allocate(self.N, self.N_low, A, self.HL, self.c_F, self.Lambda_high, self.A_bar, self.a)
        else:                   # Storage rights 
            self.update_storage_accounts(S, I, L, Spill, w)

    def update_test(self, S, I, L, Spill, w,  A):

        if self.sr == -1:       # Release sharing (planner storage)
            self.a = allocate(self.N, self.N_low, A, self.HL, self.c_F, self.Lambda_high, self.A_bar, self.a)
        else:                   # Storage rights
            self.update_storage_accounts_test(S, I, L, Spill, w)

    cdef void update_storage_accounts(self, double S, double I, double L, double Spill, double[:] w):

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
        
        cdef double tol = 0.0001                 
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
            
            # Fixed delivery loss deductions
            target = S - self.fixed_loss
            f_loss_ded = (self.fixed_loss - self.fixed_loss_co) * self.delivered 
            self.fixed_loss_co = 0

            # User inflow shares = lambda_i * I_t (unless HL=1)
            lamI = allocate(self.N, self.N_low, I, self.HL, self.c_F, self.Lambda_high, self.A_bar, lamI)
            
            # User evaporation loss deductions (socialised or proportional to account balance)
            self.l = storage_loss(self.N, L, self.s, w, self.l, self.c_F, self.ls) 
            
            for i in range(self.N):
                
                # Add fixed delivery loss deductions (by inflow shares)
                self.l[i] += f_loss_ded * self.c_F[i]
            
            s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
            
            # Apply spill event rules
            if Spill > 0:
                if self.sr == 0:                    # CS
                    for i in range(self.N):
                        s[i] = self.c_K[i] * target
                elif self.sr == 1:                  # SWA
                    lamI_Spill  = allocate(self.N, self.N_low, I - Spill, self.HL, self.c_F, self.Lambda_high, self.A_bar, lamI)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI_Spill, s) 
                    for i in range(self.N):
                        self.x[i] = -Spill * (s[i] / (self.K - self.fixed_loss))
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s) 
                elif self.sr == 2:                  # OA
                    for i in range(self.N):
                        self.x[i] = Spill * self.c_F[i] 
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                elif self.sr == 4:                  # CS-SWA
                    lamI_Spill  = allocate(self.N, self.N_low, I - Spill, self.HL, self.c_F, self.Lambda_high, self.A_bar, lamI)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI_Spill, s) 
                    for i in range(self.N):
                        s_k[i] = c_max(s[i] - self.c_F[i] * (self.K - self.fixed_loss), 0)
                        sum_s_k += s_k[i]
                    for i in range(self.N):
                        self.x[i] = -Spill * (s_k[i] / sum_s_k)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s) 

            if self.sr == 3:                        # NS
                for i in range(self.N):
                    self.s[i] = target * self.c_F[i]
                    self.x[i] = self.s[i] - s[i]
                    s[i] = self.s[i]

            # Final reconciliation (including any internal spills)
            while 1:
                sum_s = c_sum(self.N, s)
                diff = (target - sum_s)
                
                if diff < tol or it >= 100:
                    break
                
                self.x = calc_x(self.N, s, self.acc_max, self.c_F, self.x, diff, self.J, self.maxx, self.minx)
                s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                it += 1

            if it == 100:
                print 'Accounts failed to reconcile!'
            
            # Update user accounts
            for i in range(self.N):
                self.s[i] = s[i]

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
        
        cdef double tol = 0.0001                 
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
            lamI = allocate(self.N, self.N_low, I, self.HL, self.c_F, self.Lambda_high, self.A_bar, lamI)
            print 'lamI: ' + str(np.array(lamI)) 
            # User evaporation loss deductions (socialised or proportional to account balance)
            self.l = storage_loss(self.N, L, self.s, w, self.l, self.c_F, self.ls) 
            print 'loss: ' + str(np.array(self.l)) 
            print ' ---------- '
            for i in range(self.N):
                # Add fixed delivery loss deductions (by inflow shares)
                self.l[i] += f_loss_ded * self.c_F[i]
            print 'loss: ' + str(np.array(self.l)) 
            print ' ---------- '
            s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
            
            print 's: ' + str(np.array(s)) 
            # Apply spill event rules
            if Spill > 0:
                print '--- Spill ---'
                if self.sr == 0:                    # CS
                    for i in range(self.N):
                        s[i] = self.c_K[i] * target
                elif self.sr == 1:                  # SWA
                    lamI_Spill  = allocate(self.N, self.N_low, I - Spill, self.HL, self.c_F, self.Lambda_high, self.A_bar, lamI)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI_Spill, s) 
                    for i in range(self.N):
                        self.x[i] = -Spill * (s[i] / (self.K - self.fixed_loss))
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s) 
                elif self.sr == 2:                  # OA
                    for i in range(self.N):
                        self.x[i] = Spill * self.c_F[i] 
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                elif self.sr == 4:                  # CS-SWA
                    lamI_Spill  = allocate(self.N, self.N_low, I - Spill, self.HL, self.c_F, self.Lambda_high, self.A_bar, lamI)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI_Spill, s) 
                    for i in range(self.N):
                        s_k[i] = c_max(s[i] - self.c_F[i] * (self.K - self.fixed_loss), 0)
                        sum_s_k += s_k[i]
                    for i in range(self.N):
                        self.x[i] = -Spill * (s_k[i] / sum_s_k)
                    s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s) 

            if self.sr == 3:                        # NS
                for i in range(self.N):
                    self.s[i] = target * self.c_F[i]
                    self.x[i] = self.s[i] - s[i]
                    s[i] = self.s[i]

            # Final reconciliation (including any internal spills)
            while diff > tol and it < 100:
                sum_s = c_sum(self.N, s)
                diff = (target - sum_s)
                print diff
                print sum_s
                self.x = calc_x(self.N, s, self.acc_max, self.c_F, self.x, diff, self.J, self.maxx, self.minx)
                s = update_accounts(self.N, self.s, w, self.l, self.x, self.acc_max, lamI, s)
                it += 1

            if it == 100:
                print 'Accounts failed to reconcile!'
            
            # Update user accounts
            for i in range(self.N):
                self.s[i] = s[i]
    
        else:   # Extreme drought: not enough water to meet current fixed losses
            
            self.fixed_loss_co = S      # All water is put aside to meet next period fixed losses
            for i in range(self.N):     # All user accounts are zero
                self.x[i] = self.s[i]
                self.s[i] = 0
                self.a[i] = 0
   
