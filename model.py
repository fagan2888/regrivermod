import pylab
from time import time
import numpy as np
import sys
from regrivermod import *
from econlearn import Qlearn

class Model:

    def __init__(self, para):
        
        pylab.ioff()

        #################           Create objects          #################
        
        print '\nDecentralised storage model with ' + str(para.N) + ' users. \n'
        
        self.para = para
        self.storage = Storage(para)
        self.users = Users(para)
        self.sim = Simulation(para)
        self.utility = Utility(self.users, self.storage, para)

    def plannerSDP(self, seed=0):

        SR = self.utility.sr
        self.utility.sr = -1
        
        #################           Solve planners problem          #################

        print 'Solving the planner\'s problem...'
        
        self.sdp = SDP(self.para.SDP_GRID,  self.users, self.storage, self.para)
        
        tic = time()
        self.sdp.policy_iteration(self.para.SDP_TOL, self.para.SDP_ITER, plot=False)
        toc = time()
        st = toc - tic
        print 'Solve time: ' + str(st)

        self.sim.simulate(self.users, self.storage, self.utility, self.para.T0, self.para.CPU_CORES, planner=True, policy=True, polf=self.sdp.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=True, seed=seed)

        self.utility.sr = SR

        return [self.sim.stats, self.sdp, st]

    def simulate_myopic(self, simT, seed=0):
        
        self.sim.ITER = 0
        self.sim.simulate(self.users, self.storage, self.utility, simT, self.para.CPU_CORES, planner=True, policy=False, polf=0, delta=0, stats=True, planner_explore=False, t_cost_off=True, seed=seed, myopic=True)

        return self.sim.stats['SW']['Mean'][0]

    def plannerQV(self, t_cost_off=True, T1=200000, T2=400000, stage1=True, stage2=True, d=0, seed=0, type='ASGD'):
        
        if type == 'A':
            Ta = [11, 11, 7]
            La = 12
            Tb = [6, 11, 7]
            Lb = 20
            minsamp = int(60 * (T1 / 100000)**0.5) 
            asgd = False
        elif type == 'ASGD':
            Ta = [5, 5, 5]
            La = 25
            Tb = [4, 5, 4]
            Lb = 25
            minsamp = 1
            asgd = True
        
        tic = time()
        SR = self.utility.sr
        self.utility.sr = -1

        # Feasibility constraints
        Alow = lambda X: 0                  # W > 0
        Ahigh = lambda X: X[0]              # W < S
        
        if stage1:

            self.sim.simulate(self.users, self.storage, self.utility, T1, self.para.CPU_CORES, planner=True, policy=False, polf=self.sdp.W_f,
                    delta=0, stats=False, planner_explore=True, t_cost_off=t_cost_off)
            
            if type=='RF':
                self.qv = Qlearn.QVtree(2, self.para.s_points1, self.para.s_radius1, self.para, num_split=40, num_leaf=20, num_est=200)
            else:
                self.qv = Qlearn.QVtile(2, Ta, La, 1, minsamp, self.para.s_points1, self.para.s_radius1, self.para, asgd=asgd)
            
            self.qv.iterate(self.sim.XA_t, self.sim.X_t1, self.sim.series['SW'], Alow, Ahigh, ITER=45, Ascaled=False,
                    plot=False, xargs=['x', 1])
        
        if stage2:
            
            self.qv.resetQ(2, Ta, La, 1, minsamp)

            self.sim.simulate(self.users, self.storage, self.utility, T2, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, 
                    delta=d, stats=False, planner_explore=True, t_cost_off=t_cost_off)

            # Search range 
            Alow = lambda X, Ws: max(Ws - X[0]*d , 0)             # W > W* - d*S
            Ahigh = lambda X, Ws: min(Ws + X[0]*d, X[0])          # W < W* + d*S
            
            self.qv.iterate(self.sim.XA_t, self.sim.X_t1, self.sim.series['SW'], Alow, Ahigh, ITER=5, Ascaled=True,
                    plot=False, xargs=['x', 1])
            
        toc = time()
        st = toc - tic    
        print 'Solve time: ' + str(st)
        
        self.sim.ITER = 0
        self.sim.simulate(self.users, self.storage, self.utility, self.para.T0, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=t_cost_off, seed=seed)

        self.utility.sr = SR

        return [self.sim.stats, self.qv, st]

    def multiQV(self,  N_e, d, ITER, init=False, policy = 0, type='A'):
        
        tic = time()
        
        if type == 'A':
            Ta = [10, 5, 10, 5, 5]
            La = 12
            Tb = [6, 5, 11, 5, 5]
            Lb = 20
            minsamp = 75
            asgd = False
        elif type == 'ASGD':
            Ta = [5, 5, 5, 5, 5]
            La = 25
            Tb = [5, 5, 5, 5, 5]
            Lb = 25
            minsamp = 1
            asgd = True
        
        if init:
            self.qvHL = [0, 0]
            self.HL = [0, 1]
            self.users.init = 1
            for h in self.HL:
                self.qvHL[h] = Qlearn.QVtile(4, Ta, La, 1, minsamp, self.para.s_points2, self.para.s_radius2, self.para, asgd=True)
            self.users.W_f = policy
        else:
            self.users.init = 0

        self.users.set_explorers(N_e, d)
        bigT = int(self.para.T2 / N_e)
        
        self.sim.simulate(self.users, self.storage, self.utility, bigT, self.para.CPU_CORES, stats=False)
        
        # Feasibility constraints
        if d == 0:
            Alow = lambda X: 0              # w > 0
            Ahigh = lambda X: X[1]          # w < s
            Asc = False
        else:
            Alow = lambda X, w_st: max(w_st - X[1]*d , 0)             # w > w* - d*s
            Ahigh = lambda X, w_st: min(w_st + X[1]*d, X[1])          # w < w* + d*s
            Asc = True
            for h in self.HL:
                self.qvHL[h].resetQ(4, Tb, Lb, 1, minsamp) 
        
        for h in self.HL:
            self.qvHL[h].iterate(self.sim.XA_t[h], self.sim.X_t1[h], self.sim.u_t[h], Alow, Ahigh, ITER=ITER, Ascaled=Asc, plot=False, xargs=[1000000, 'x', 1, 1], a = [0, 0, 0, 0.25, 0.25], b = [100, 100, 100, 99.75, 100], pc_samp=0.25)

        toc = time()
        st = toc - tic    
        print 'Solve time: ' + str(st)
        
        return [self.sim.stats, self.qvHL]
        

    def plannerQVTree(self, t_cost_off=True, T1=200000, T2=400000, simT=800000, stage1=True, stage2=True, d=0.15, seed=0):

        tic = time()

        self.utility.sr = -1
        
        # Feasibility constraints
        Alow = lambda X: 0              # W > 0
        Ahigh = lambda X: X[0]          # W < S
        if stage1:

            self.sim.simulate(self.users, self.storage, self.utility, T1, self.para.CPU_CORES, planner=True, policy=False, delta=0, stats=False, planner_explore=True, t_cost_off=t_cost_off)
            
            self.qv = Qlearn.QVtree(2, self.para.s_points1, self.para.s_radius1, self.para)
            self.qv.iterate(self.sim.XA_t, self.sim.X_t1, self.sim.series['SW'], Alow, Ahigh, ITER=10, Ascaled=False, plot=True)
        
        if stage2:
            
            self.sim.simulate(self.users, self.storage, self.utility, T2, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, delta=d, stats=False, planner_explore=True, t_cost_off=t_cost_off)

            # Search range 
            Alow = lambda X, Ws: max(Ws - X[0]*d , 0)             # W > W* - d*S
            Ahigh = lambda X, Ws: min(Ws + X[0]*d, X[0])          # W < W* + d*S
            
            self.qv.iterate(self.sim.XA_t, self.sim.X_t1, self.sim.series['SW'], Alow, Ahigh, ITER=5, Ascaled=True, plot=True)
            
        toc = time()
        st = toc - tic    
        print 'Solve time: ' + str(st)
        
        self.sim.ITER = 0
        self.sim.simulate(self.users, self.storage, self.utility, simT, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=t_cost_off, seed=seed)

        return [self.sim.stats, self.qv, st]



    def chapter6(self, stage2=True, T1=200000, T2=400000, d=0.15):
       
        print '\n --- Scenario --- \n'
        print str(self.para.sr) + ' storage rights, loss deductions = ' + str(self.para.ls) + ', priority = ' + str(self.para.HL) + '\n'
        
        big_tic = time()
        
        # Planners problem
        p_stats, p_sdp, p_st = self.plannerSDP()
        
        #################           Solve Release sharing problem          #################
        
        #     Search for optimal shares by stochastic hill climbing
        
        print '\n Solve release sharing problem... '
        
        ITER = 1
        sg1 = True
        sg2 = False
        if self.para.opt_lam == 1:
            print 'Search for optimal shares... \n'
            ITER = 12
            Lambda_old = self.para.Lambda_high
            Lambda_new = self.para.Lambda_high
            delta = Lambda_old / 1
            SW_old = -1
            SW_check = np.zeros(ITER)
            Lambda_check = np.zeros(ITER)


        for i in range(ITER):
        
            stats, qv, st = self.plannerQV(t_cost_off=False, stage1=sg1, stage2=sg2, T1=self.para.T1, T2=self.para.T1)

            if self.para.opt_lam:
                SW = self.sim.stats['SW']['Mean'][0]
                SW_check[i] = SW
                Lambda_check[i] = Lambda_new
                if SW > SW_old:
                    delta *= 0.75
                    Lambda_old = Lambda_new
                    SW_old = SW
                else:
                    delta *= -0.75

                Lambda_new = max(min(Lambda_old + np.random.rand() * delta, 0.99), 0.01)

                self.users.set_shares(Lambda_new)
                self.utility.set_shares(Lambda_new, self.users)
                self.sim.ITER = 0
                sg1 = False
                sg2 = True

                print '--- Optimal share search ---'
                print 'Current best: ' + str(Lambda_old)
                print 'Next guess: ' + str(Lambda_new)
                print 'delta: ' + str(delta)

        if self.para.opt_lam:
            pylab.scatter(Lambda_check, SW_check)
            pylab.show()
        
        #################           Solve storage right problem          #################
        if self.utility.sr >= 0:
            
            self.users.W_f = self.qv.W_f
            N_e = 5                             # Max Number of explorers per user group
            d = 0                               # Search all feasible space

        ##################          User starting values                #################

            print 'User starting values, fitted Q-iteration ...'
         
            stats, qv = self.multiQV(N_e, d, ITER=self.para.ITER1, init=True, policy = qv.W_f)

            self.users.set_policy(qv[0].W_f, qv[0].W_f)
            
        ##################          Main Q-learning                     #################
            
            print '\nSolve decentralised problem, multiple agent fitted Q-iteration ...'
            
            for i in range(self.para.ITER2):
                print '\n  ---  Iteration: ' + str(i) + '  ---\n'
                print 'Number of Explorers: '+ str(N_e * 2) + ' of ' + str(self.para.N)  
                print 'Exploration range: ' + str(d)   
                print '-----------------------------------------'

                stats, qv = self.multiQV(N_e, d, ITER=self.para.iters)
                
                N_e = self.para.N_e[i]
                d = self.para.d[i]
                update_rate = self.para.update_rate[i]

                self.users.update_policy(qv[0].W_f, qv[1].W_f, prob = update_rate)
            
            self.sim.simulate(self.users, self.storage, self.utility, self.para.T2, self.para.CPU_CORES, stats = True)
            
            big_toc = time.time()
            print "Total time (minutes): " + str(round((big_toc - big_tic) / 60,2))

            stats = self.sim.stats

            del self

            return stats

    def chapter5(self):
       
        big_tic = time()
        
        # Planners problem
        p_stats, sdp, p_st = self.plannerSDP()
        
        ##################          User starting values              #################
        
        print 'User starting values, fitted Q-iteration ...'
     
        stats, qv = self.multiQV(5, 0, ITER=self.para.ITER1, init=True, policy=sdp.W_f, type='ASGD')
        
        self.users.update_policy(qv[0].W_f, qv[1].W_f, init=True)
        
        ##################          Main Q-learning              #################

        V_e = np.zeros([self.para.ITER2, 2])      # Value error
        P_e = np.zeros([self.para.ITER2, 2])      # Policy error

        print '\nSolve decentralised problem, multiple agent fitted Q-iteration ...'
        
        for i in range(self.para.ITER2):
            
            N_e = self.para.N_e[i]
            d = self.para.d[i]
            
            print '\n  ---  Iteration: ' + str(i) + '  ---\n'
            print 'Number of Explorers: '+ str(N_e * 2) + ' of ' + str(self.para.N)  
            print 'Exploration range: ' + str(d)   
            print '-----------------------------------------'

            stats, qv = self.multiQV(N_e, d, ITER=self.para.iters, type='ASGD')
            
            for h in range(2):
                V_e[i, h] = qv[h].ve
                P_e[i, h] = qv[h].pe

            self.users.update_policy(qv[0].W_f, qv[1].W_f, prob = self.para.update_rate[i])
         
        self.sim.simulate(self.users, self.storage, self.utility, self.para.T0, self.para.CPU_CORES, stats = True)
        
        big_toc = time()
        print "Total time (minutes): " + str(round((big_toc - big_tic) / 60, 2))
        
        """
        
        ################        Build some results      ##########################

        stats = self.sim.stats
        
        # Aggregate policy functions
        
        from econlearn.tilecode import Tilecode as Tile

        W_f_p = Tile(1, [9], 9)
        X = self.sim.p_series['S'].reshape([self.para.T0, 1])
        Y = sim.p_series['W']
        W_f_p.fit(X,Y)

        W_f = Tile(1, [9], 9)
        X = self.sim.series['S'].reshape([bigT, 1])
        Y = self.sim.series['W']
        W_f.fit(X,Y)
        
        W_f_low = Tile(1, [9], 9)
        X = self.sim.series['S_low'].reshape([bigT, 1])
        Y = self.sim.series['W_low']
        W_f_low.fit(X, Y)
        
        W_f_high = Tile(1, [9], 9)
        X = self.sim.series['S_high'].reshape([bigT, 1])
        Y = self.sim.series['W_high']
        W_f_high.fit(S, Y)

        policy = [W_f_p, W_f, W_f_low, W_f_high]

        del self

        return [V_e, P_e, stats, policy]
        """

    def chapter8(self, stage2=False, T1=100000, T2=100000, d=0):

        # Planner's problem testing for chapter 8

        SW = [0, 0, 0]
        S = [0, 0, 0]
        solvetime = [0, 0, 0]
        n = 10

        seed = int(time())
        
        for i in range(n):
            
            # SDP
            self.sim.ITER = 0
            p_stats, p_sdp, p_st = self.plannerSDP(seed=seed)
            
            SW[0] += p_stats['SW']['Mean'][0] / n
            S[0] += p_stats['S']['Mean'][0]  / n
            solvetime[0] += p_st /n
            
            # QV learning - TC-A
            self.sim.ITER = 0
            stats, qv1, st = self.plannerQV(t_cost_off=True, T1=T1, T2=T2, stage1=True, stage2=stage2, d=d, seed=seed, type='A')

            SW[1] += stats['SW']['Mean'][0] / n
            S[1] += stats['S']['Mean'][0] / n
            solvetime[1] += st / n
        
            # QV learning - TC-ASGD
            self.sim.ITER = 0
            stats, qv2, st = self.plannerQV(t_cost_off=True, T1=T1, T2=T2, stage1=True, stage2=stage2, d=d, seed=seed, type='ASGD')
            
            SW[2] += stats['SW']['Mean'][0] / n
            S[2] += stats['S']['Mean'][0] / n
            solvetime[2] += st / n
            
        
        self.sdp.W_f.plot(['x', 1])
        qv1.W_f.plot(['x', 1])
        qv2.W_f.plot(['x', 1])
        pylab.show()
        
        return [SW, S, solvetime]


