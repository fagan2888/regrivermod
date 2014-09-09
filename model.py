import pylab
from time import time
import numpy as np
import sys
from regrivermod import *
from econlearn import Qlearn
from results.chartbuilder import *

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

    def plannerSDP(self, seed=0, plot=False):
        
        SR = self.utility.sr
        self.utility.sr = -1
        
        #################           Solve planners problem          #################

        print 'Solving the planner\'s problem...'
        
        self.sdp = SDP(self.para.SDP_GRID,  self.users, self.storage, self.para)
        
        tic = time()
        self.sdp.policy_iteration(self.para.SDP_TOL, self.para.SDP_ITER, plot=plot)
        toc = time()
        st = toc - tic
        print 'Solve time: ' + str(st)
        
        self.sim.simulate(self.users, self.storage, self.utility, self.para.T0, self.para.CPU_CORES, planner=True, policy=True, polf=self.sdp.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=True, seed=seed) 

        self.utility.sr = SR
       
        self.p_series = self.sim.series

        return [self.sim.stats, self.sdp, st]

    def simulate_myopic(self, simT, seed=0):
        
        SR = self.utility.sr
        self.utility.sr = -1
        
        self.sim.ITER = 0
        self.sim.simulate(self.users, self.storage, self.utility, simT, self.para.CPU_CORES, planner=True, policy=False, polf=0, delta=0, stats=True, planner_explore=False, t_cost_off=True, seed=seed, myopic=True)

        self.utility.sr = SR
        
        self.series = self.sim.series
        
        return [self.sim.stats['SW']['Mean'][0], self.sim.stats['S']['Mean'][0]]

    def plannerQV(self, t_cost_off=True, T1=200000, T2=400000, stage1=True, stage2=True, d=0, seed=0, type='ASGD', simulate=True): 
        
        if type == 'A':
            Ta = [11, 11, 7]
            La = 12
            Tb = [6, 11, 7]
            Lb = 20
            minsamp = int(60 * (T1 / 100000)**0.5) 
            asgd = False
        elif type == 'ASGD':
            Ta = [6, 5, 4]
            La = 25
            Tb = Ta
            Lb = La
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
                self.qv = Qlearn.QVtree(2, self.para.s_points1, self.para.s_radius1, self.para, num_split=40, num_leaf=20, num_est=215)
            else:
                self.qv = Qlearn.QVtile(2, Ta, La, 1, minsamp, self.para.sg_radius1, self.para, asgd=asgd, linT=8)
            
            self.qv.iterate(self.sim.XA_t, self.sim.X_t1, self.sim.series['SW'], Alow, Ahigh, ITER=self.para.QV_ITER1, Ascaled=False,
                    plot=True, xargs=['x', 1], eta=0.8, sg_points=self.para.sg_points1)
        
            XA = self.sim.XA_t
            X = self.sim.X_t1
            SW = self.sim.series['SW']

        if stage2:
            
            self.qv.resetQ(2, Tb, Lb, 1, minsamp)

            self.sim.simulate(self.users, self.storage, self.utility, T2, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, 
                    delta=d, stats=False, planner_explore=True, t_cost_off=t_cost_off)

            XA = np.vstack([XA, self.sim.XA_t])
            X = np.vstack([X, self.sim.X_t1])
            SW = np.hstack([SW, self.sim.series['SW']])
            
            self.qv.iterate(XA, X, SW, Alow, Ahigh, ITER=self.para.QV_ITER2, Ascaled=False,
                    plot=True, xargs=['x', 1], eta=0.8, sg_points=self.para.sg_points1)
            
        toc = time()
        st = toc - tic    
        print 'Solve time: ' + str(st)
        
        if simulate:
            self.sim.ITER = 1
            self.sim.simulate(self.users, self.storage, self.utility, self.para.T0, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=t_cost_off, seed=seed)

        self.utility.sr = SR
        
        return [self.sim.stats, self.qv, st]

    def multiQV(self,  N_e, d, ITER, init=False, type='ASGD', eta=0.7, testing=False, test_idx=0, partial=False):
        
        tic = time()
        
        if type == 'A':
            Ta = [10, 5, 10, 5, 5]
            La = 12
            Tb = [6, 5, 11, 5, 5]
            Lb = 20
            minsamp = 50
            asgd = False
        elif type == 'ASGD':
            Ta = [6, 5, 6, 5, 4]
            La = 25
            Tb = [5, 5, 6, 4, 4]
            Lb = 25
            minsamp = 5
            asgd = True
        
        if init:
            self.HL = [0, 1]
            self.users.testing = 0
            w_f, v_f = self.users.init_policy(self.sdp.W_f, self.sdp.V_f, self.storage, self.para.linT, self.para.CPU_CORES, self.para.s_radius2)
            if not(partial):
                self.qvHL = [0, 0]
                for h in self.HL:
                    self.qvHL[h] = Qlearn.QVtile(4, Ta, La, 1, minsamp, self.para.s_points2, self.para.s_radius2, self.para, asgd=asgd, linT=self.para.linT, init=True, W_f=w_f[h], V_f=v_f[h])

        self.users.set_explorers(N_e, d, testing, test_idx=test_idx)
        if testing:
            bigT = self.para.T2
        elif partial:
            bigT = int((self.para.T2 / N_e) * self.para.sample_rate)
        else:
            bigT = int(self.para.T2 / N_e)
        
        self.sim.simulate(self.users, self.storage, self.utility, bigT, self.para.CPU_CORES, stats=False, partial=partial)
        
        # Feasibility constraints
        Alow = lambda X: 0              # w > 0
        Ahigh = lambda X: X[1]          # w < s
        
        for h in self.HL:
            self.qvHL[h].iterate(self.sim.XA_t[h], self.sim.X_t1[h], self.sim.u_t[h], Alow, Ahigh, ITER=ITER, Ascaled=False, plot=True, xargs=[1000000, 'x', 1, 1], a = [0, 0, 0, 0.25, 0.25], b = [100, 100, 100, 99.75, 100], pc_samp=0.25, maxT=500000, eta=eta)

        toc = time()
        st = toc - tic    
        print 'Total time: ' + str(st)
        
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



    def chapter6(self, ):
       
        print '\n --- Scenario --- \n'
        print str(self.para.sr) + ' storage rights, loss deductions = ' + str(self.para.ls) + ', priority = ' + str(self.para.HL) + '\n'
        
        home = '/home/nealbob'
        folder = '/Dropbox/model/results/chapter6/'
        out = '/Dropbox/Thesis/IMG/chapter6/'
        img_ext = '.pdf'
        
        big_tic = time()
        self.sim.percentiles = True
        
        # Planners problem
        p_stats, p_sdp, p_st = self.plannerSDP()
       

        #################           Solve Release sharing problem          #################
        
        self.sim.percentiles = False
        #     Search for optimal shares by stochastic hill climbing
        
        print '\n Solve release sharing problem... '
        
        stats, qv, st = self.plannerQV(t_cost_off=False, stage1=True, stage2=True, T1=self.para.T1, T2=self.para.T1, d=self.para.policy_delta, simulate=False)
        
        ITER = 0 
        if self.para.opt_lam == 1:
            
            print 'Search for optimal shares... \n'
            ITER = self.para.opt_lam_ITER
            delta = self.para.Lambda_high / 1.5
            SW = np.zeros(ITER + 1)
            SW[0] = self.sim.stats['SW']['Mean'][1]
            Lambda = np.zeros(ITER + 1)
            Lambda[0] = self.para.Lambda_high
            SW_max = SW[0]
            Lambda_max = self.para.Lambda_high

        for i in range(1, ITER + 1):
                
            Lambda[i] = max(min(Lambda_max + np.random.rand() * delta, 0.99), 0.01)

            self.users.set_shares(Lambda[i]) 
            self.utility.set_shares(Lambda[i], self.users)
            stats, qv, st = self.plannerQV(t_cost_off=False, stage1=True, stage2=True, T1=self.para.T1, T2=self.para.T1, d=self.para.policy_delta)
            
            SW[i] = self.sim.stats['SW']['Mean'][1]
            if SW[i] > SW_max:
                delta *= 0.8
                Lambda_max = Lambda[i]
                SW_max = SW[i]
            else:
                delta *= -0.8

            print '--- Optimal share search ---'
            print 'Lambda: ' + str(Lambda[i])
            print 'Welfare: ' + str(SW[i])
            print 'Best Lambda: ' + str(Lambda_max)
            print 'Best welfare: ' + str(SW_max)

        if self.para.opt_lam:
            data = [[Lambda, SW]]
            chart = {'OUTFILE': home + out + 'Lambda' + str(self.para.HL) + img_ext,
             'YLABEL': 'Mean welfare',
             'XLABEL': 'High reliability user inflow share' }
            build_chart(chart, data, chart_type='scatter')

        self.sim.ITER = 1
        self.sim.percentiles = True
        self.sim.simulate(self.users, self.storage, self.utility, self.para.T0, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=False)
        self.sim.percentiles = False

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
            if self.para.opt_lam:
                users.share_adj = 0.01              # Inflow share adjustment rate
                self.users.share_expore = 1
                self.users.set_shares(self.para.Lambda_high)

            print '\nSolve decentralised problem, multiple agent fitted Q-iteration ...'
            
            for i in range(self.para.ITER2):
                N_e = self.para.N_e[i]
                d = self.para.d[i]
                update_rate = self.para.update_rate[i]
                
                print '\n  ---  Iteration: ' + str(i) + '  ---\n'
                print 'Number of Explorers: '+ str(N_e * 2) + ' of ' + str(self.para.N)  
                print 'Exploration range: ' + str(d)   
                print '-----------------------------------------'
                
                stats, qv = self.multiQV(N_e, d, ITER=self.para.iters)
                
                self.users.update_policy(qv[0].W_f, qv[1].W_f, prob = update_rate)
                
                if self.para.opt_lam:
                    if users.low_gain > 0 and users.high_gain > 0:
                        users.share_adj *= 1
                    if users.low_gain < 0 and users.high_gain < 0:
                        users.share_adj *= -1
                    newLambda =  self.para.Lambda_high-self.users.share_adj
                    self.utility.set_shares(newLambda, self.users)
                    self.users.set_shares(newLambda)
                    print 'Share adjustment: ' + str(self.users.share_adj) + ' Lambda high: ' + str(newLambda)

            self.sim.simulate(self.users, self.storage, self.utility, self.para.T2, self.para.CPU_CORES, stats = True)
        

        big_toc = time()
        print "Total time (minutes): " + str(round((big_toc - big_tic) / 60,2))
        
        Lambda_high = self.utility.Lambda_high

        stats = self.sim.stats

        del self

        return stats, Lambda_high
        
    def chapter5(self):
       
        big_tic = time()
        
        # Planners problem
        p_stats, sdp, p_st = self.plannerSDP()
        
        ##################          User starting values              #################
        
        print 'User starting values, fitted Q-iteration ...'
     
        stats, qv = self.multiQV(5, 0.25, ITER=self.para.ITER1, init=True, type='ASGD')
        
        #self.users.update_policy(qv[0].W_f, qv[1].W_f, prob = 0.0)
        
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

            stats, qv = self.multiQV(N_e, d, ITER=self.para.iters, type='ASGD', partial=True)
            
            for h in range(2):
                V_e[i, h] = qv[h].ve
                P_e[i, h] = qv[h].pe

            self.users.update_policy(qv[0].W_f, qv[1].W_f, prob = self.para.update_rate[i])
        
        self.users.exploring = 0
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

    def testing(self, n=6, N=75, stage2=True):
        
        SWmax = 0
        SW = 0 
        for i in range(N):
            Ta = [6, 5, 4] #[int(3 + 4*np.random.rand()), int(3 + 4*np.random.rand()), int(3 + 4*np.random.rand())]
            Tb = [int(3 + 4*np.random.rand()), int(3 + 4*np.random.rand()), int(3 + 4*np.random.rand())]
            linT = 6 #int(4 + 3*np.random.rand())
            eta = 0.6 + np.random.rand()*0.3
            
            print 'Ta: ' + str(Ta)
            print 'Tb: ' + str(Tb)
            print 'linT: ' + str(linT)
            print 'eta: ' + str(eta)
            
            SW = 0
            for j in range(n):
                stats, qv2, st = self.plannerQV(t_cost_off=True, T1=50000, T2=50000, stage1=True, stage2=stage2, d=0.15, seed=time(), type='ASGD', Ta=Ta, Tb=Tb, eta=eta, linT=linT)
                SW += stats['SW']['Mean'][1] / n
            
            if SW > SWmax:
                SWmax = SW
                Tamax = Ta
                Tbmax = Tb
                linTmax = linT
                etamax = eta
                print '--------------------------------------------------'
                print 'Ta: ' + str(Ta)
                print 'Tb: ' + str(Tb)
                print 'linT: ' + str(linT)
                print 'eta: ' + str(eta)
                print 'SW: ' + str(SW)
         
        print '--------------------------------------------------'
        print 'Ta: ' + str(Tamax)
        print 'Tb: ' + str(Tbmax)
        print 'linT: ' + str(linTmax)
        print 'eta: ' + str(etamax)
        print 'SW: ' + str(SWmax)
    
    def chapter8(self, stage2=False, T1=100000, T2=100000, d=0, decentral_test=False):

        # Planner's problem testing for chapter 8

        SW = [0, 0, 0]#, 0]
        S = [0, 0, 0]#, 0]
        solvetime = [0, 0, 0]#, 0]
        n = 1

        seed = int(time())
        self.sim.series = self.sim.series_old
        p_stats, p_sdp, p_st = self.plannerSDP(seed=seed)
        
        """ 
        self.users.testing = 0 
        for i in range(n):
            
            # SDP
            p_stats, p_sdp, p_st = self.plannerSDP(seed=seed)
            
            SW[0] += p_stats['SW']['Mean'][0] / n
            S[0] += p_stats['S']['Mean'][0]  / n
            solvetime[0] += p_st /n
            
            # QV learning - TC-A
            stats, qv1, st = self.plannerQV(t_cost_off=True, T1=T1, T2=T2, stage1=True, stage2=stage2, d=d, seed=seed, type='A')

            SW[1] += stats['SW']['Mean'][1] / n
            S[1] += stats['S']['Mean'][1] / n
            solvetime[1] += st / n
        
            # QV learning - TC-ASGD
            stats, qv2, st = self.plannerQV(t_cost_off=True, T1=T1, T2=T2, stage1=True, stage2=stage2, d=d, seed=seed, type='ASGD')
            
            SW[2] += stats['SW']['Mean'][1] / n
            S[2] += stats['S']['Mean'][1] / n
            solvetime[2] += st / n
            
            # QV learning - RF
            #stats, qv3, st = self.plannerQV(t_cost_off=True, T1=T1, T2=T2, stage1=True, stage2=stage2, d=d, seed=seed, type='RF')
            
            #SW[3] += stats['SW']['Mean'][1] / n
            #S[3] += stats['S']['Mean'][1] / n
            #solvetime[3] += st / n
        
        self.sdp.W_f.plot(['x', 1])
        qv1.W_f.plot(['x', 1])
        qv2.W_f.plot(['x', 1], showdata=True)
        #qv3.W_f.plot(['x', 1])
        pylab.show()
        
        
        self.sdp.W_f.plot(['x', 0.5])
        qv1.W_f.plot(['x', 0.5])
        qv2.W_f.plot(['x', 0.5], showdata=True)
        #qv3.W_f.plot(['x', 1])
        pylab.show()
        
        self.sdp.W_f.plot(['x', 2])
        qv1.W_f.plot(['x', 2])
        qv2.W_f.plot(['x', 2], showdata=True)
        #qv3.W_f.plot(['x', 1])
        pylab.show()
        """
        SWb = [0, 0]
                
        if decentral_test:
            self.para.T2 = T1

            for i in range(n):
            
                NN = self.users.N_low
                # QV learning - TC-A
                #self.sim.ITER = 0
                #stats, qv1 = self.multiQV(1, d, ITER=self.para.ITER1, init=True, policy=p_sdp.W_f, type='A', testing=True, test_idx=NN) 

                #self.sim.ITER = 1
                #self.users.update_policy(qv1[1].W_f, qv1[1].W_f, test = True, test_idx = NN)
                #self.sim.simulate(self.users, self.storage, self.utility, 500000, self.para.CPU_CORES, delta=0, stats=False,  seed=seed) 

                #SWb[0] += self.sim.test_payoff / n

                # QV learning - TC-ASGD
                self.sim.ITER = 0
                stats, qv2 = self.multiQV(1, d, ITER=self.para.ITER1, init=True, policy=p_sdp.W_f, type='ASGD', testing=True, test_idx=NN)
                self.sim.ITER = 1
                self.users.update_policy(qv2[1].W_f, qv2[1].W_f, test = True, test_idx = NN)
                self.sim.simulate(self.users, self.storage, self.utility, 500000, self.para.CPU_CORES, delta=0, stats=False,  seed=seed) 
                
                #self.sim.ITER = 0
                #self.para.T2 = int(T1 / 5)
                #stats, qv2 = self.multiQV(1, d, ITER=5, init=True, policy=p_sdp.W_f, type='ASGD', testing=True, test_idx=NN, partial=True)
                
                #self.sim.ITER = 1
                #self.users.update_policy(qv2[1].W_f, qv2[1].W_f, test = True, test_idx = NN)
                #self.sim.simulate(self.users, self.storage, self.utility, 500000, self.para.CPU_CORES, delta=0, stats=False,  seed=seed)
                
                SWb[1] += self.sim.test_payoff / n
                
        #self.qv1 = qv1
        self.qv2 = qv2
                
        del self
        

        return [SW, S, solvetime, SWb]


