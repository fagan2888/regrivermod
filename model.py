import pylab
from time import time
import numpy as np
import sys
from regrivermod import *
from econlearn import Qlearn
from results.chartbuilder import *

class Model:

    def __init__(self, para, ch7=False, turn_off_env=False):
        
        pylab.ioff()
        
        #################           Create objects          #################
        
        print '\nDecentralised storage model with ' + str(para.N) + ' users. \n'

        self.para = para
        self.storage = Storage(para, ch7)
        self.users = Users(para, ch7)
        self.sim = Simulation(para, ch7)
        if ch7:
            self.env = Environment(para, turn_off=turn_off_env)
            self.market = Market(para, self.users)
            self.utility = Utility(self.users, self.storage, para, ch7, self.env)
            self.learn_market_demand()
        else:
            self.utility = Utility(self.users, self.storage, para)
        
        print '----------------------------------------------------'
        print 'Sum of user shares: ' + str(np.sum(self.utility.c_F))
        print 'User shares: ' + str(np.array(self.utility.c_F))
        print '----------------------------------------------------'

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
        
        #self.sim.ITER = 0
        self.sim.simulate(self.users, self.storage, self.utility, simT, self.para.CPU_CORES, planner=True, policy=False, polf=0, delta=0, stats=True, planner_explore=False, t_cost_off=True, seed=seed, myopic=True)

        self.utility.sr = SR
        
        self.series = self.sim.series

        return [self.sim.stats['SW']['Mean'][0], self.sim.stats['S']['Mean'][0]]

    def simulate_SOP(self, simT, Sbar, Lambda, seed=0):

        self.para.central_case()
        self.para.set_property_rights('RS-HL')
        self.para.Lambda_high = Lambda

        self.storage = Storage(self.para, False)
        self.users = Users(self.para)
        self.sim = Simulation(self.para)
        self.utility = Utility(self.users, self.storage, self.para)

        self.sim.ITER = 0
        self.sim.simulate(self.users, self.storage, self.utility, simT, self.para.CPU_CORES, planner=True, policy=False, polf=0, delta=0, stats=True, planner_explore=False, t_cost_off=False, seed=seed, myopic=False, SOP=True, Sbar=Sbar)

        SW = self.sim.stats['SW']['Mean'][0]
        U_low = self.sim.stats['U_low']['Mean'][0]
        U_high = self.sim.stats['U_high']['Mean'][0]
        yield_low = self.sim.stats['A_low']['Mean'][0] / self.sim.stats['A_low']['Max'][0]
        yield_high = self.sim.stats['A_high']['Mean'][0] / self.sim.stats['A_high']['Max'][0]
        SD_low = self.sim.stats['A_low']['SD'][0] / self.sim.stats['A_low']['Mean'][0]
        SD_high = self.sim.stats['A_high']['SD'][0] / self.sim.stats['A_high']['Mean'][0]
        P_low = np.sum((self.sim.series['P'] * self.sim.series['A_low']) * (self.para.beta**np.arange(simT))) / self.sim.stats['A_low']['Max'][0]

        P_high = np.sum((self.sim.series['P'] * self.sim.series['A_high']) * (self.para.beta**np.arange(simT))) / self.sim.stats['A_high']['Max'][0]

        return [SW, U_low, U_high, yield_low, yield_high, SD_low, SD_high, P_low, P_high]

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

        D = 2
        xargs=['x', 1]

        if stage1:

            self.sim.simulate(self.users, self.storage, self.utility, T1, self.para.CPU_CORES, planner=True, policy=False, polf=self.sdp.W_f,
                    delta=0, stats=False, planner_explore=True, t_cost_off=t_cost_off)

            if type=='RF':
                self.qv = Qlearn.QVtree(D, self.para.s_points1, self.para.s_radius1, self.para, num_split=40, num_leaf=20, num_est=215)
            else:
                self.qv = Qlearn.QVtile(D, Ta, La, 1, minsamp, self.para.sg_radius1, self.para, asgd=asgd, linT=8)
            
            self.qv.iterate(self.sim.XA_t, self.sim.X_t1, self.sim.series['SW'], Alow, Ahigh, ITER=self.para.QV_ITER1, Ascaled=False,
                    plot=False, xargs=xargs, eta=0.8, sg_points=self.para.sg_points1, maxT=250000)
        
            XA = self.sim.XA_t
            X = self.sim.X_t1
            SW = self.sim.series['SW']

        if stage2:
            
            self.qv.resetQ(D, Tb, Lb, 1, minsamp)

            self.sim.simulate(self.users, self.storage, self.utility, T2, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, 
                    delta=d, stats=False, planner_explore=True, t_cost_off=t_cost_off)

            if not(stage1):
                XA = self.sim.XA_t
                X = self.sim.X_t1
                SW = self.sim.series['SW']
            else:
                XA = np.vstack([XA, self.sim.XA_t])
                X = np.vstack([X, self.sim.X_t1])
                SW = np.hstack([SW, self.sim.series['SW']])
            
            self.qv.iterate(XA, X, SW, Alow, Ahigh, ITER=self.para.QV_ITER2, Ascaled=False,
                    plot=False, xargs=xargs, eta=0.8, sg_points=self.para.sg_points1, maxT=250000)
            
        toc = time()
        st = toc - tic    
        print 'Solve time: ' + str(st)
        
        if simulate:
            self.sim.simulate(self.users, self.storage, self.utility, self.para.T0, self.para.CPU_CORES, planner=True, policy=True, polf=self.qv.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=t_cost_off, seed=seed)

        self.utility.sr = SR
        
        return [self.sim.stats, self.qv, st]
    
    def learn_market_demand(self, T=100000):
       
        self.utility.init_policy(self.storage, self.para)
        
        self.utility.explore = 1
        self.utility.d = 0

        self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, T, self.para.CPU_CORES , planner=True, stats=True, initP=True)
        
        self.market.estimate_market_demand(self.sim.series['P'][:, 0], self.sim.series['A'][:, 0], self.sim.series['Q_low'][:, 0], self.sim.series['Q_high'][:, 0], self.sim.series['Q_env'][:, 0], self.sim.series['Bhat'][:, 0], self.sim.series['I'][:, 0]*(self.storage.I_bar_ch7[0]**-1), self.para)

    def plannerQV_ch7(self, t_cost_off=True, T=250000, stage2=False, d=0, simulate=True, envoff=False):

        tic = time()
        
        if envoff:
            Tiles = [[6, 5, 4], [6,5,4]]
            D = 2
            a = [0, 0, 0.25]
            b = [100, 100, 99.75]
            pc_samp = 0.5
            xarg = ['x', 1]
        else:
            Tiles = [[6, 6, 6, 5], [6, 6, 6, 5]]
            D = 3
            xarg = ['x', 1, 1]
            a = [0, 0, 0.25, 0]
            b = [100, 100, 99.75, 100]
            pc_samp = 0.5

        L = 25

        # Feasibility constraints
        Alow = lambda X: 0                  # W > 0
        Ahigh = lambda X: X[0]              # W < S

        self.utility.init_policy(self.storage, self.para)
        self.utility.explore = 1
        self.utility.d = 0

        self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, T, self.para.CPU_CORES, planner=True)
        
        self.qv = Qlearn.QVtile_ch7(D, Tiles, L, 1, 1, self.para.sg_radius1_ch7, self.para, asgd=True, linT=7)

        self.qv.iterate(self.sim.XA, self.sim.X1, self.sim.U, Alow, Ahigh, ITER=self.para.QV_ITER1, 
                            plot=False, eta=0.8, sg_points=self.para.sg_points1_ch7/2, maxT=250000, a=a, b=b, pc_samp=pc_samp, plotiter=False, xargs=xarg)
        
        #for m in range(2): 
        #    self.qv.W_f[m].plot(xarg)
        #    pylab.show()
        #    self.qv.V_f[m].plot(xarg)
        #    pylab.show()
        
        self.utility.policy0 = self.qv.W_f[0]
        self.utility.policy1 = self.qv.W_f[1]
        
        if stage2:

            XA = self.sim.XA
            X = self.sim.X1
            SW = self.sim.U

            self.qv.resetQ(D, Tiles, L, 1, 1)
            self.utility.d = d
            self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, T, self.para.CPU_CORES, planner=True)

            XA = np.vstack([XA, self.sim.XA])
            X = np.vstack([X, self.sim.X1])
            SW = np.hstack([SW, self.sim.U])

            self.qv.iterate(XA, X, SW, Alow, Ahigh, ITER=self.para.QV_ITER2, Ascaled=False,
                            plot=False, eta=0.8, sg_points=self.para.sg_points1_ch7, maxT=250000, a=a, b=b, pc_samp=pc_samp)
            
            self.utility.policy = self.qv.W_f

        toc = time()
        st = toc - tic
        print 'Solve time: ' + str(st)

        if simulate:
            self.utility.explore = 0
            self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, self.para.T0, self.para.CPU_CORES, planner=True, stats=True)
        
        return [self.sim.stats, self.qv, st]
    
    def chapter7_initialise(self, ):

        self.para.sg_radius1_ch7 = 0.02
        self.para.sg_points1_ch7 = 750
        stats, _, _ = self.plannerQV_ch7(T=200000, stage2=False, d=0.2, simulate=True, envoff=True)

        E_cons = stats['E']['Summer']['Mean'][self.sim.ITEROLD]

        self.env.turn_off = 0 
        self.sim.ITER = 0
        self.sim.ITERNEW = 0
        self.sim.ITEROLD = 0
        self.para.sg_radius1_ch7 = 0.045
        self.para.sg_points1_ch7 = 2500
        stats, _, _ = self.plannerQV_ch7(T=200000, stage2=False, d=0.2, simulate=True, envoff=False)
        
        E_opt = stats['E']['Summer']['Mean'][self.sim.ITEROLD]
        E_lambda = 1 - E_opt / E_cons 
    
        print 'E_cons: ' + str(E_cons) + ', E_opt: ' + str(E_opt) + ', % change: ' + str(E_lambda)
        
        return E_lambda

    def chapter7(self, P_adj, psearch=True):
        
        from econlearn.tilecode import Tilecode as Tile

        self.plannerQV_ch7(T=200000, stage2=False, d=0.2, simulate=True, envoff=False)
        
        if self.para.sr == 'NS':
            NS = True
        else:
            NS = False
        
        big_tic = time()
        
        ##################          User starting values              #################
        
        print 'Starting values, fitted QV iteration ...'
        
        self.users.set_explorers(2, self.para.d[0])#.3)
        self.env.explore = 1
        self.env.d = 0.3
        
        #stats, qv = 
        self.multiQV_ch7(ITER=self.para.ITER1, init=True, NS=NS)
        
        ##################          Main Q-learning              #################
        
        print '\nSolve decentralised problem, multiple agent fitted QV iteration ...'
        
        counter  = 0
        scale = 1
        P_adj_plot = np.zeros(self.para.ITER2)
        self.env.P_adj = P_adj
        self.market.P_adj = P_adj

        for i in range(self.para.ITER2):
            
            print '\n  ---  Iteration: ' + str(i) + '  ---\n'
            print '-----------------------------------------'

            stats, qv = self.multiQV_ch7(ITER=1, partial=True, NS=NS)
            
            self.users.update_policy_ch7(qv[0].W_f, qv[1].W_f, Np=self.para.update_rate_ch7[i], N_e=2, d=self.para.d[i])
            self.env.update_policy(qv[2].W_f)
            self.env.d = self.para.envd[i]
            
            print '------- Summer --------'
            print 'Mean A low: ' + str(np.mean(self.sim.series['A_low'][:,0]))
            print 'Mean A high: ' + str(np.mean(self.sim.series['A_high'][:,0]))
            print 'Mean A env: ' + str(np.mean(self.sim.series['A_env'][:,0]))
            print 'Mean Q low: ' + str(np.mean(self.sim.series['Q_low'][:,0]))
            print 'Mean Q high: ' + str(np.mean(self.sim.series['Q_high'][:,0]))
            print 'Mean Q env: ' + str(np.mean(self.sim.series['Q_env'][:,0]))
            print 'Mean EWH budget outcome: ' + str(np.mean(self.sim.series['Budget'][:, 0]))
            print '------- Winter --------'
            print 'Mean A low: ' + str(np.mean(self.sim.series['A_low'][:,1]))
            print 'Mean A high: ' + str(np.mean(self.sim.series['A_high'][:,1]))
            print 'Mean A env: ' + str(np.mean(self.sim.series['A_env'][:,1]))
            print 'Mean Q low: ' + str(np.mean(self.sim.series['Q_low'][:,1]))
            print 'Mean Q high: ' + str(np.mean(self.sim.series['Q_high'][:,1]))
            print 'Mean Q env: ' + str(np.mean(self.sim.series['Q_env'][:,1]))
            print 'Mean EWH budget outcome: ' + str(np.mean(self.sim.series['Budget'][:, 1]))
            
            if psearch:
                counter += 1
                if counter > 3:
                    counter = 0
                    self.users.exploring = 0
                    self.env.explore = 0
                    P_adj_sim, Budget_sim = self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, 100000,self.para.CPU_CORES, stats=True, budgetonly=True) 
                    self.users.exploring = 1
                    self.env.explore = 1

                    approx = Tile(1, [11], 30, min_sample=120)
                    approx.fit(P_adj_sim, Budget_sim)
                    #approx.plot()
                    #pylab.show()
                    
                    X = np.linspace(P_adj - 2.5*30, P_adj + 2.5*30, 1000).reshape([1000, 1])
                    Y = approx.predict(X)
                    idx = np.abs(Y) > 0
                    idx2 = np.argmin(np.abs(Y[idx]))
                    P_adj2 = X[idx][idx2] 
                    Y2 = Y[idx][idx2] 
                    #Y1 = Y[500] 
                    #if Y2 > 5000000: # linear extrapolation
                    #    P_adj = P_adj - 75 #*(abs(Y2)/5000000)#  Y1 * ((P_adj - P_adj2) / (Y1 - Y2)) 
                    #elif Y2 < -5000000:
                    #    P_adj = P_adj + 75 #*(abs(Y2)/5000000)
                    #else:
                    P_adj = P_adj2

                    self.env.P_adj = P_adj
                    self.market.P_adj = P_adj

                    print '======================================================' 
                    print 'P_adj: ' + str(self.market.P_adj) 
                    print 'Budget hat: ' + str(Y2) 
                    print '======================================================' 
                    #pylab.plot(X, Y) 
                    #pylab.show()
                    #import pdb; pdb.set_trace()
                #P_adj_plot[i] = P_adj
                #pylab.plot(P_adj_plot)
                #pylab.show()
             
                """
                counter += 1
                if counter > 7:
                    budget = np.mean(self.sim.series['Budget'])
                    #self.users.exploring = 0
                    #self.env.explore = 0
                    #budget = self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, 100000,self.para.CPU_CORES, stats=True, budgetonly=True) 
                    #self.users.exploring = 1
                    #self.env.explore = 1
                    
                    if  budget > 0:
                        P_adj -= delta * scale 
                    else:
                        P_adj += delta * scale
                        
                    self.env.P_adj = P_adj
                    self.market.P_adj = P_adj
                    print 'P_adj: ' + str(self.market.P_adj) 
                    print 'Budget: ' + str(budget)
                    counter = 0
                    scale *= 0.79
               """ 
        if psearch and self.para.sr == 'OA':
            print '============================ Final stage ======================================'
            self.users.exploring = 0
            self.env.explore = 0
            
            iters = 0 
            while iters < 50:
                self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, 100000, self.para.CPU_CORES, partial=False, stats=True)
                budget = np.mean(self.sim.series['Budget'])
                
                if abs(budget) < 100000:
                    print '======================================================' 
                    print 'P_adj: ' + str(self.market.P_adj) 
                    print 'Budget hat: ' + str(budget) 
                    print '======================================================' 
                    iters += 201
                    break
                else:
                    if budget > 0:
                        P_adj -= 10
                    else:
                        P_adj += 10
                
                    self.env.P_adj = P_adj
                    self.market.P_adj = P_adj

                    print '======================================================' 
                    print 'P_adj: ' + str(self.market.P_adj) 
                    print 'Budget hat: ' + str(budget) 
                    print '======================================================' 
                
                iters += 1
        
        if psearch:

            self.users.exploring = 0
            self.env.explore = 0
            P_adj_sim, Budget_sim = self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, 500000,self.para.CPU_CORES, stats=True, budgetonly=True) 

            approx = Tile(1, [11], 30, min_sample=120)
            approx.fit(P_adj_sim, Budget_sim)
            X = np.linspace(P_adj - 2.5*30, P_adj + 2.5*30, 1000).reshape([1000, 1])
            Y = approx.predict(X)
            idx = np.abs(Y) > 0
            idx2 = np.argmin(np.abs(Y[idx]))
            P_adj2 = X[idx][idx2] 
            Y2 = Y[idx][idx2] 
            Y1 = Y[500] 
            P_adj = P_adj2

            self.env.P_adj = P_adj
            self.market.P_adj = P_adj

            print '======================================================' 
            print 'Last chance'
            print 'P_adj: ' + str(self.market.P_adj) 
            print 'Budget hat: ' + str(Y2) 
            print '======================================================' 
        
        self.users.exploring = 0
        self.env.explore = 0
        self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, self.para.T2, self.para.CPU_CORES, partial=False, stats=True)
        
        big_toc = time()
        print "Total time (minutes): " + str(round((big_toc - big_tic) / 60, 2))
        
        return [self.sim.stats, P_adj, self.sim.series]
        #return [self.sim.stats, P_adj]

        

    def multiQV_ch7(self, ITER, init=False, eta=0.7, partial=False, NS=False):
        
        tic = time()
        
        if init:
            Tuser = [[6, 5, 6, 3, 4], [6, 5, 6, 1, 3]]
            Luser = 25
            minsamp = 3
            asgd = True
            Tenv = [[5, 4, 5, 4, 4], [5, 4, 5, 4, 4]]
            Lenv = 25


            for m in range(2):
                self.users.init_policy_ch7(self.qv.W_f[m], self.qv.V_f[m], self.storage, self.utility, self.para.linT, self.para.CPU_CORES, 
                        self.para.sg_radius2_ch7, m)
                
                self.env.init_policy(self.qv.W_f[m], self.qv.V_f[m], self.storage, self.utility, self.para.linT, self.para.CPU_CORES, 
                        self.para.sg_radius2_ch7, m)
        
            self.qv_multi = [0, 0, 0]
            
            self.qv_multi[0] = Qlearn.QVtile_ch7(4, Tuser, Luser, 0.5, minsamp, self.para.sg_radius2_ch7, self.para, asgd=asgd, 
                    linT=self.para.linT)
            self.qv_multi[1] = Qlearn.QVtile_ch7(4, Tuser, Luser, 0.5, minsamp, self.para.sg_radius2_ch7, self.para, asgd=asgd, 
                    linT=self.para.linT) 
            self.qv_multi[2] = Qlearn.QVtile_ch7(4, Tenv, Lenv, 0.5, minsamp, self.para.sg_radius2_ch7, self.para, asgd=asgd, 
                    linT=self.para.linT) 
        
        if partial:
            bigT = int((self.para.T2_ch7 / 2) * self.para.sample_rate)
        else:
            bigT = int(self.para.T2_ch7 / 2)
        
        self.sim.simulate_ch7(self.users, self.storage, self.utility, self.market, self.env, bigT,self.para.CPU_CORES, partial=partial, stats=False) #
        
        # Feasibility constraints
        Alow = lambda X: 0              # w > 0
        Ahigh = lambda X: X[1]          # w < s
        
        print "\nSolving low reliability users problem"
        print "-------------------------------------\n"
        self.qv_multi[0].iterate([self.sim.XA_t[0], self.sim.XA_t1[0]], [self.sim.X_t1[0], self.sim.X_t11[0]],  [self.sim.u_t[0], self.sim.u_t1[0]],  
                Alow, Ahigh, ITER=ITER, a = [0, 0, 0, 0.25, 0.25],b = [100, 100, 100, 99.40, 99.40], pc_samp=0.25, 
                maxT=1200000,eta=eta, tilesg=True, sg_samp=self.para.sg_samp2_ch7, sg_prop=self.para.sg_prop2_ch7, sgmem_max=0.15, plotiter=False, 
                xargs=[300000,'x', 1, 1, 1], test=False, plot=False, NS=NS)
     
        print "\nSolving high reliability users problem"
        print "-------------------------------------\n"
        self.qv_multi[1].iterate([self.sim.XA_t[1], self.sim.XA_t1[1]], [self.sim.X_t1[1], self.sim.X_t11[1]], [self.sim.u_t[1], self.sim.u_t1[1]],
                Alow, Ahigh, ITER=ITER, a = [0, 0, 0, 0.25, 0.25], b = [100, 100, 100, 99.40, 99.40], pc_samp=0.25, 
                maxT=1200000, eta=eta, tilesg=True, sg_samp=self.para.sg_samp2_ch7, sg_prop=self.para.sg_prop2, sgmem_max=0.15, plotiter=False, 
                xargs=[300000,'x', 1, 1, 1], plot=False, NS=NS)
        
        print "\nSolving the EWHs problem"
        print "-------------------------------------\n"
        self.qv_multi[2].iterate(self.sim.XA_e, self.sim.X1_e, self.sim.u_e, Alow, Ahigh, ITER=ITER, maxT=600000, eta=eta,
                tilesg=True, sg_samp=self.para.sg_samp2_ch7, sg_prop=self.para.sg_prop2, sgmem_max= 0.15, plotiter=False, xargs=[1000000,'x', 1, 1], plot=False, NS=NS)

        toc = time()
        st = toc - tic    
        print 'Total time: ' + str(st)
        
        return [self.sim.stats, self.qv_multi]
        
    def multiQV(self, ITER, init=False, type='ASGD', eta=0.7, testing=False, test_idx=0, partial=False, NS=False):
        
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
            minsamp = 3
            asgd = True

        
        if init:
            self.HL = [0, 1]
            self.users.testing = 0
            w_f, v_f = self.users.init_policy(self.sdp.W_f, self.sdp.V_f, self.storage, self.para.linT, self.para.CPU_CORES, self.para.sg_radius2)
            if not(partial):
                self.qvHL = [0, 0]
                for h in self.HL:
                    self.qvHL[h] = Qlearn.QVtile(4, Ta, La, 1, minsamp, self.para.sg_radius2, self.para, asgd=asgd, linT=self.para.linT, init=True, W_f=w_f[h], V_f=v_f[h])

        if testing:
            bigT = self.para.T2
        elif partial:
            bigT = int((self.para.T2 / self.users.N_e) * self.para.sample_rate)
        else:
            bigT = int(self.para.T2 / self.users.N_e)
        
        self.sim.simulate(self.users, self.storage, self.utility, bigT, self.para.CPU_CORES, stats=False, partial=partial)
        
        # Feasibility constraints
        Alow = lambda X: 0              # w > 0
        Ahigh = lambda X: X[1]          # w < s
        
        for h in self.HL:
            self.qvHL[h].iterate(self.sim.XA_t[h], self.sim.X_t1[h], self.sim.u_t[h], Alow, Ahigh, ITER=ITER, Ascaled=False, plot=False, xargs=[1000000, 'x', 1, 2.0], a = [0, 0, 0, 0.25, 0.25], b = [100, 100, 100, 99.75, 100], pc_samp=0.25, maxT=500000, eta=eta, tilesg=True, sg_samp=self.para.sg_samp2, sg_prop=self.para.sg_prop2, NS=NS)

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

            self.sim.simulate(self.users, self.storage, self.utility, T1, self.para.CPU_CORES, planner=True, policy=False,
                              delta=0, stats=False, planner_explore=True, t_cost_off=t_cost_off)
            
            self.qv = Qlearn.QVtree(2, self.para.s_points1, self.para.s_radius1, self.para)
            self.qv.iterate(self.sim.XA_t, self.sim.X_t1, self.sim.series['SW'], Alow, Ahigh, ITER=10, Ascaled=False, plot=True)
        
        if stage2:
            
            self.sim.simulate(self.users, self.storage, self.utility, T2, self.para.CPU_CORES, planner=True, policy=True,
                              polf=self.qv.W_f, delta=d, stats=False, planner_explore=True, t_cost_off=t_cost_off)

            # Search range 
            Alow = lambda X, Ws: max(Ws - X[0]*d , 0)             # W > W* - d*S
            Ahigh = lambda X, Ws: min(Ws + X[0]*d, X[0])          # W < W* + d*S
            
            self.qv.iterate(self.sim.XA_t, self.sim.X_t1, self.sim.series['SW'], Alow, Ahigh, ITER=5, Ascaled=True, plot=True)
            
        toc = time()
        st = toc - tic    
        print 'Solve time: ' + str(st)
        
        self.sim.ITER = 0
        self.sim.simulate(self.users, self.storage, self.utility, simT, self.para.CPU_CORES, planner=True, policy=True,
                          polf=self.qv.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=t_cost_off, seed=seed)

        return [self.sim.stats, self.qv, st]



    def chapter6(self, sens=False):
       
        print '\n --- Scenario --- \n'
        print str(self.para.sr) + ' storage rights, loss deductions = ' + str(self.para.ls) + ', priority = ' + str(self.para.HL) + '\n'
        
        home = '/home/nealbob'
        folder = '/Dropbox/model/results/chapter6/'
        out = '/Dropbox/Thesis/IMG/chapter6/'
        img_ext = '.pdf'
        
        big_tic = time()
        
        # Planners problem
        p_stats, p_sdp, p_st = self.plannerSDP()

        #################           Solve Release sharing problem          #################
        
        #     Search for optimal shares by stochastic hill climbing
        
        print '\n Solve release sharing problem... '
        
        if sens:
            self.users.set_shares(self.para.Lambda_high_RS) 
            self.utility.set_shares(self.para.Lambda_high_RS, self.users)
        
        stats, qv, st = self.plannerQV(t_cost_off=False, stage1=True, stage2=True, T1=self.para.T1, T2=self.para.T1, d=self.para.policy_delta, simulate=True)
         
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
        else:
            Lambda_max = self.utility.Lambda_high

        for i in range(1, ITER + 1):
                
            Lambda[i] = max(min(Lambda_max + np.random.rand() * delta, 0.99), 0.01)

            self.users.set_shares(Lambda[i]) 
            self.utility.set_shares(Lambda[i], self.users)
            
            stats, qv, st = self.plannerQV(t_cost_off=False, stage1=True, stage2=True, T1=self.para.T1, T2=self.para.T1, d=self.para.policy_delta)

            SW[i] = self.sim.stats['SW']['Mean'][self.sim.ITEROLD]
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

        #if self.para.opt_lam:
            #data = [[Lambda, SW]]
            #chart = {'OUTFILE': home + out + 'Lambdatest' + str(self.para.HL) + img_ext,
            # 'YLABEL': 'Mean welfare',
            # 'XLABEL': 'High reliability user inflow share' }
            #build_chart(chart, data, chart_type='scatter')
        self.RSLambda = Lambda_max

        #################           Solve storage right problem          #################
        if self.utility.sr >= 0:
            if sens:
                self.users.set_shares(self.para.Lambda_high) 
                self.utility.set_shares(self.para.Lambda_high, self.users)
            
            temp = self.para.ITER2
            #self.sim.ITER = 1
            #self.sim.ITEROLD = 0
            #self.sim.ITERNEW = 1

            ### ============================
            if self.para.opt_lam: 

                if self.para.unbundled:
                    self.para.ITER2 = 140
                else:
                    self.para.ITER2 = 100

                n = np.count_nonzero(self.sim.stats['SW']['Mean']) - 1
                # Optimal inflow shares
                SW = np.zeros(self.para.ITER2)
                SW[0] = self.sim.stats['SW']['Mean'][n]
                SW_max = SW[0]
                Lambda = np.zeros(self.para.ITER2)
                Lambda[0] = Lambda_max
                delta = Lambda[0] / 10
                LambdaK = np.zeros(self.para.ITER2)
                LambdaK[0] = Lambda_max
                deltaK = delta
                j = 0
                counter = 0
                Kchange = False
            # ===============================

        ##################          User starting values                #################
            print 'User starting values, fitted Q-iteration ...'
            
            self.users.set_explorers(self.para.N_e[0], self.para.d[0])
         
            stats, qv = self.multiQV(ITER=self.para.ITER1, init=True, type='ASGD')
            
        ##################          Main Q-learning                     #################
        
            V_e = np.zeros([self.para.ITER2, 2])      # Value error
            P_e = np.zeros([self.para.ITER2, 2])      # Policy error

            print '\nSolve decentralised problem, multiple agent fitted Q-iteration ...'
            
            for i in range(self.para.ITER2):
                print '\n  ---  Iteration: ' + str(i) + '  ---\n'
                print 'Number of Explorers: '+ str(self.para.N_e[i] * 2) + ' of ' + str(self.para.N)  
                print 'Exploration temperature: ' + str(self.para.d[i])   
                print '-----------------------------------------'

                stats, qv = self.multiQV(ITER=self.para.iters, type='ASGD', partial=True)
                
                for h in range(2):
                    V_e[i, h] = qv[h].ve
                    P_e[i, h] = qv[h].pe

                self.users.update_policy(qv[0].W_f, qv[1].W_f, Np=self.para.update_rate[i], N_e=self.para.N_e[i + 1], d=self.para.d[i + 1])
                
                ### ================================
                # Optimal Inflow shares
                if self.para.opt_lam:
                    counter += 1 
                    if counter > 8 and i > 16:
                        counter = 0
                        ITER = self.sim.ITEROLD
                        SW[j] = self.sim.stats['SW']['Mean'][ITER]
                        
                        if j == 0:
                            Lambda[j + 1] = max(min(Lambda[j] + delta*0.5, 0.99), 0.01)
                            LambdaK[j + 1] = LambdaK[j]
                            SW_max = SW[j]
                        else:
                            if self.para.unbundled:
                                if Kchange:
                                    if SW[j] > SW_max:
                                        deltaK *= 0.9
                                        LambdaK[j + 1] = LambdaK[j]
                                        SW_max = SW[j]
                                    else:
                                        deltaK *= -0.75
                                        LambdaK[j + 1] = LambdaK[j - 1]
                                    Lambda[j + 1] = max(min(Lambda[j] + 1 * delta, 0.99), 0.01)
                                    Kchange = False
                                else:  
                                    if SW[j] > SW_max:
                                        delta *= 0.9
                                        Lambda[j + 1] = Lambda[j]
                                        SW_max = SW[j]
                                    else:
                                        delta *= -0.75
                                        Lambda[j + 1] = Lambda[j - 1]
                                    LambdaK[j + 1] = max(min(LambdaK[j] + 1 * deltaK, 0.99), 0.01)
                                    Kchange = True
                                
                                pylab.scatter(Lambda[0:j+1], SW[0:j+1])
                                pylab.show()
                                pylab.scatter(LambdaK[0:j+1], SW[0:j+1])
                                pylab.show()
                                
                                print '--- Optimal Inflow share search ---'
                                print 'Lambda previous: ' + str(Lambda[j -1])
                                print 'Lambda: ' + str(Lambda[j])
                                print 'Lambda next: ' + str(Lambda[j + 1])
                                print '--- Optimal Capacity share search ---'
                                print 'Lambda K previous: ' + str(LambdaK[j -1])
                                print 'Lambda K: ' + str(LambdaK[j])
                                print 'Lambda K next: ' + str(LambdaK[j + 1])
                                print 'Old welfare: ' + str(SW[j-1])
                                print 'Welfare: ' + str(SW[j])
                                print 'Welfare: ' + str(SW_max)
                            else:
                                if SW[j] > SW[j - 1]:
                                    delta *= 0.9
                                else:
                                    delta *= -0.75

                                Lambda[j + 1] = max(min(Lambda[j] + 1 * delta, 0.99), 0.01)
                                
                                #pylab.scatter(Lambda[0:j+1], SW[0:j+1])
                                #pylab.show()
                    
                                print '--- Optimal share search ---'
                                print 'Lambda previous: ' + str(Lambda[j -1])
                                print 'Lambda: ' + str(Lambda[j])
                                print 'Lambda next: ' + str(Lambda[j + 1])
                                print 'Old welfare: ' + str(SW[j-1])
                                print 'Welfare: ' + str(SW[j])
                            
                        
                        self.users.set_shares(Lambda[j + 1], self.para.unbundled, LambdaK[j + 1]) 
                        self.utility.set_shares(Lambda[j + 1], self.users)
                        j += 1
                ### =================================== 

            self.users.exploring = 0
            self.sim.simulate(self.users, self.storage, self.utility, self.para.T2, self.para.CPU_CORES, stats = True)
            self.para.ITER2 = temp
        
        big_toc = time()
        print "Total time (minutes): " + str(round((big_toc - big_tic) / 60,2))
        
        Lambda_high = self.utility.Lambda_high
        if self.para.unbundled:
            Lambda_K = LambdaK[j]
        else:
            Lambda_K = Lambda_high
        
        self.sim.finalise_stats()

        stats = self.sim.stats

        self.CSLambda = Lambda_high
        print '-------- Final Lambda: ' + str(self.utility.Lambda_high)

        return stats, Lambda_high, Lambda_K

    def chapter6_extra(self, points):

        # Buld grid
        Sbar_grid = np.linspace(0, self.para.K, points)
        Lambda_grid = np.linspace(0.01, 0.99, points)
        self.Lambda, self.Sbar = np.meshgrid(Lambda_grid, Sbar_grid)


        #Result arrays
        self.SW = np.zeros([points, points])
        self.U_low = np.zeros([points, points])
        self.U_high = np.zeros([points, points])
        self.yield_low = np.zeros([points, points])
        self.yield_high = np.zeros([points, points])
        self.SD_low = np.zeros([points, points])
        self.SD_high = np.zeros([points, points])
        self.P_low = np.zeros([points, points])
        self.P_high = np.zeros([points, points])

        for i in range(points):
            for j in range(points):
                if (self.Lambda[i,j] * self.storage.K) < self.Sbar[i, j]:
                    self.SW[i, j], self.U_low[i, j], self.U_high[i, j], self.yield_low[i, j], self.yield_high[i, j], self.SD_low[i, j], self.SD_high[i, j], self.P_low[i, j], self.P_high[i, j] = self.simulate_SOP(100000, self.Sbar[i, j], self.Lambda[i, j])
                if np.isnan(self.yield_high[i,j]) or self.yield_high[i, j] == 0:
                    self.yield_high[i,j] = 10
                if np.isnan(self.yield_low[i,j]) or self.yield_low[i, j] == 0:
                    self.yield_low[i,j] = np.nan

        index = self.SW > 10000000000000
        self.SW[index] = -10
        maxSW = np.max(self.SW)
        maxindex = np.where(self.SW == maxSW)
        yh = self.yield_high[maxindex]

        index1 = (self.yield_high > yh - 0.005)
        index2 = (self.yield_high < yh + 0.005)
        index3 = index1*index2

        for i in range(points):
            for j in range(points):
                if self.yield_high[i,j] == 10:
                    self.yield_high[i,j] = np.min(self.yield_high[i, :])


        print 'Optimal Lambda: ' + str(self.Lambda[maxindex])
        print 'Optimal Sbar: ' + str(self.Sbar[maxindex])
        print 'Optimal yield high: ' + str(self.yield_high[maxindex])

        from results.chartbuilder import chart_params
        chart_params()
        home = '/home/nealbob'
        out = '/Dropbox/Thesis/IMG/chapter6/'

        pylab.figure()
        CS = pylab.contour(self.Lambda, self.Sbar, self.yield_high, [0.7, 0.8, 0.9, 0.98])
        pylab.clabel(CS)
        pylab.xlabel(r'$\Lambda_{high} K$')
        pylab.ylabel(r'$\overline{S}$')
        pylab.savefig(home + out + 'yield_high_c.pdf', bbox_inches='tight')
        pylab.show()

        pylab.figure()
        CS = pylab.contour(self.Lambda, self.Sbar, self.yield_low, [0.9, 0.7, 0.5, 0.3])
        pylab.clabel(CS)
        pylab.xlabel(r'$\Lambda_{high} K$')
        pylab.ylabel(r'$\overline{S}$')
        pylab.savefig(home + out + 'yield_low_c.pdf', bbox_inches='tight')
        pylab.show()

        pylab.figure()
        pylab.scatter(self.Lambda[index3], self.Sbar[index3])
        pylab.xlabel(r'$\Lambda_{high} K$')
        pylab.savefig(home + out + 'yield_low_s.pdf', bbox_inches='tight')
        pylab.show()

        pylab.figure()
        pylab.plot(self.Lambda[index3], self.yield_low[index3], 'o', label='low')
        pylab.plot(self.Lambda[index3], self.yield_high[index3], 'o', label='high')
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
        pylab.xlabel(r'$\Lambda_{high} K$')
        pylab.savefig(home + out + 'yield_s.pdf', bbox_inches='tight')
        pylab.show()

        pylab.figure()
        pylab.plot(self.Lambda[index3], self.P_low[index3], 'o', label='low')
        pylab.plot(self.Lambda[index3], self.P_high[index3], 'o', label='high')
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
        pylab.xlabel(r'$\Lambda_{high} K$')
        pylab.savefig(home + out + 'P_high_s.pdf', bbox_inches='tight')
        pylab.show()


    def chapter5(self):
       
        print '\n --- Scenario --- \n'
        print str(self.para.sr) + ' storage rights, loss deductions = ' + str(self.para.ls) + ', priority = ' + str(self.para.HL) + '\n'
        
        if self.para.sr == 'NS':
            NS = True
        else:
            NS = False

        big_tic = time()
        
        # Planners problem
        p_stats, sdp, p_st = self.plannerSDP()
        
        ##################          User starting values              #################
        
        print 'User starting values, fitted Q-iteration ...'
        
        self.users.set_explorers(self.para.N_e[0], self.para.d[0])
     
        stats, qv = self.multiQV(ITER=self.para.ITER1, init=True, type='ASGD', NS=NS)
        
        ##################          Main Q-learning              #################

        V_e = np.zeros([self.para.ITER2, 2])      # Value error
        P_e = np.zeros([self.para.ITER2, 2])      # Policy error

        print '\nSolve decentralised problem, multiple agent fitted Q-iteration ...'
        
        for i in range(self.para.ITER2):
            
            print '\n  ---  Iteration: ' + str(i) + '  ---\n'
            print 'Number of Explorers: '+ str(self.para.N_e[i] * 2) + ' of ' + str(self.para.N)  
            print 'Exploration temperature: ' + str(self.para.d[i])   
            print '-----------------------------------------'

            stats, qv = self.multiQV(ITER=self.para.iters, type='ASGD', partial=True, NS=NS)
            
            for h in range(2):
                V_e[i, h] = qv[h].ve
                P_e[i, h] = qv[h].pe

            self.users.update_policy(qv[0].W_f, qv[1].W_f, Np=self.para.update_rate[i], N_e=self.para.N_e[i + 1], d=self.para.d[i + 1])
            
        self.users.exploring = 0
        self.sim.simulate(self.users, self.storage, self.utility, self.para.T0, self.para.CPU_CORES, stats = True)
        
        print '-------- Final Lambda: ' + str(self.utility.Lambda_high)
        big_toc = time()
        print "Total time (minutes): " + str(round((big_toc - big_tic) / 60, 2))
        
        ################        Build some results      ##########################

        stats = self.sim.stats
        
        # Aggregate policy functions
        
        from econlearn.tilecode import Tilecode as Tile

        W_f_p = Tile(1, [9], 9)
        T = self.p_series['S'].shape[0]
        X = self.p_series['S'].reshape([T, 1])
        Y = self.p_series['W']
        W_f_p.fit(X,Y)

        W_f = Tile(1, [9], 9)
        T = self.sim.series['S'].shape[0]
        X = self.sim.series['S'].reshape([T, 1])
        Y = self.sim.series['W']
        W_f.fit(X,Y)
        
        W_f_low = Tile(1, [9], 9)
        X = self.sim.series['S_low'].reshape([T, 1])
        Y = self.sim.series['W_low']
        W_f_low.fit(X, Y)
        
        W_f_high = Tile(1, [9], 9)
        X = self.sim.series['S_high'].reshape([T, 1])
        Y = self.sim.series['W_high']
        W_f_high.fit(X, Y)

        policy = [W_f_p, W_f, W_f_low, W_f_high]

        return [V_e, P_e, stats, policy]
        

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


