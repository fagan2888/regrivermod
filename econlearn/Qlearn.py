from __future__ import division
import numpy as np
from time import time
import sys
import pylab
from tilecode import Tilecode
from samplegrid import buildgrid 
from sklearn.ensemble import ExtraTreesRegressor as Tree
#from sklearn.ensemble import RandomForestRegressor as Tree

class QVtile:

    """
    Solve any MDP by Q-V iteration
    
    Attributes
    -----------

    W_f :
        Policy function
    V_f :
        Value function
    Q_f :
        Action-value function
    """
 
    def __init__(self, D, T, L, mem_max, ms, maxgrid, radius, para, asgd=False):

        self.Q_f = Tilecode(D + 1, T, L, mem_max, min_sample=ms, cores=para.CPU_CORES)
        
        Twv = int((1 / radius) / 2)
        T = [Twv for t in range(D)]
        L = int(130 / Twv)
        
        points = maxgrid
        self.W_f = Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=7, cores=para.CPU_CORES)
        self.V_f = Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=7, cores=para.CPU_CORES)
        self.maxgrid = maxgrid
        self.radius = radius
        self.D = D

        self.first = True
        
        self.beta = para.beta
        self.CORES = para.CPU_CORES
        self.asgd = asgd

    
    def resetQ(self, D, T, L, mem_max, ms):

        self.Q_f = Tilecode(D + 1, T, L, mem_max, min_sample=ms, cores=self.CORES)
    
    
    def iterate(self, XA, X1, u, A_low, A_high, ITER=50, Ascaled=False, plot=True, xargs=[], output=True, a = 0, b = 0, pc_samp=1):

        tic = time()

        self.v_e = 0        # Value function error
        self.p_e = 0        # Policy function error
        
        T = XA.shape[0]
        
        tic = time()
        N = X1.shape[0]
        grid, m = buildgrid(X1, self.maxgrid, self.radius, scale=True, stopnum=700)
        points = grid.shape[0]
        toc = time()
        print 'State grid points: ' + str(points) + ', of maximum: ' + str(m) + ', Time taken: ' + str(toc - tic)

        if self.first:
            self.W_f.fit(grid, np.zeros(points))
            self.V_f.fit(grid, np.zeros(points))
            self.first = False
        
        Al = np.zeros(points)
        Ah = np.zeros(points)
        if Ascaled:
            for i in range(points):
                Ws = self.W_f.predict(grid[i,:])
                Al[i] = A_low(grid[i,:], Ws)
                Ah[i] = A_high(grid[i,:], Ws)
                minpol = 0
                maxpol = 1
        else:
            for i in range(points):
                Al[i] = A_low(grid[i,:])
                Ah[i] = A_high(grid[i,:])
                minpol = min(Al)
                maxpol = max(Ah)
        
        
        if ITER == 1:
            precompute = False
        else:
            precompute = True
        
        # ------------------
        #   Q-learning
        # ------------------
        
        #First iteration
        j = 0

        # Q values
        Q = u + self.beta * self.V_f.predict(X1, store_XS=precompute)
        
        # Fit Q function
        self.Q_f.fit(XA, Q, pa=minpol, pb=maxpol , copy=False, unsupervised=precompute, sgd=self.asgd, asgd=self.asgd, eta=0.8, n_iters=1, scale=1* (1 / T), storeindex=(self.asgd and precompute), a=a, b=b, pc_samp=pc_samp)
        
        # Optimise Q function
        self.ve, W_opt, state = self.maximise(grid, Al, Ah, Ascaled, output=output)
        
        for j in range(1, ITER):
            # Q values
            Q = u + self.beta * self.V_f.fast_values()
            
            # Fit Q function
            self.Q_f.partial_fit(Q, 0)

            # Optimise Q function
            self.ve, W_opt, state = self.maximise(grid, Al, Ah, Ascaled, output=output)
            
        W_opt_old = self.W_f.predict(state)
        self.W_f.fit(state, W_opt, sgd=0, eta=0.1, n_iters=5, scale=0)
        self.pe = np.mean(abs(W_opt_old - W_opt)/W_opt_old)
        
        toc = time()

        print 'Solve time: ' + str(toc - tic) + ', Policy change: ' + str(self.pe)
        
        if plot:
            xargstemp = xargs
            self.W_f.plot(xargs, showdata=True)
            pylab.show()
            self.V_f.plot(xargstemp, showdata=True)
            pylab.show()

    def maximise(self, grid, Al, Ah, Ascaled, plot=False, output=True):

        """
        Maximises current Q-function for a subset of state space points and returns new value and policy functions

        Parameters
        -----------

        grid : array, shape=(N, D)
            State space grid

        Al : array, shape=(N,)
            action lower bound

        Ah : array, shape=(N,)
            action upper bound
        
        Ascaled : boolean
            True if action is scaled to [0,1]

        Returns
        -----------

        ERROR: float
            Mean absolute deviation

        """
        
        tic = time()

        W_old = self.W_f.predict(grid)
        
        if Ascaled:
            Alow = np.zeros(grid.shape[0])
            Ahigh = np.ones(grid.shape[0])
            W_old = (W_old - Al) / (Ah - Al) 
        else:
            Alow = Al
            Ahigh = Ah
        
        X =  np.vstack([W_old, grid.T]).T
        
        [W_opt, V, state, idx] = self.Q_f.opt(X, Alow, Ahigh)
        nidx = np.array([not(i) for i in idx])

        if Ascaled:
            W_opt  = Al[idx] + (Ah[idx] - Al[idx]) * W_opt

        V_old = self.V_f.predict(state)

        self.V_f.fit(state, V, sgd=0, eta=0.1, n_iters=5, scale=0)
        

        if np.count_nonzero(V_old) < V_old.shape[0]:
            self.v_e = 1
        else:
            self.v_e = np.mean(abs(V_old - V)/V_old)

        toc = time()
        
        if output:
            print 'Value change: ' + str(round(self.v_e, 3)) + '\t---\tMax time: ' + str(round(toc - tic, 4))

        return [self.v_e, W_opt, state]

class QVtree:

    def __init__(self, D,  maxgrid, radius, para, num_split=40, num_leaf=20, num_est=215):

        self.Q_f = Tree(n_estimators=num_est, min_samples_split=num_split, min_samples_leaf=num_leaf, n_jobs=para.CPU_CORES)
        
        Twv = (1 / radius) / 1.8
        T = [Twv for t in range(D)]
        L = int(140 / Twv)
        
        points = maxgrid
        self.W_f = Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=7, cores=para.CPU_CORES)
        self.V_f = Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=7, cores=para.CPU_CORES)

        self.maxgrid = maxgrid
        self.radius = radius
        self.D = D

        self.first = True
        
        self.beta = para.beta
        self.CORES = para.CPU_CORES
        
    def iterate(self, XA, X1, u, A_low, A_high, ITER=50, Ascaled=False, plot=True, xargs =[], output=True, gridsamp=1):

        tic = time()

        self.v_e = 0        # Value function error
        self.p_e = 0        # Policy function error

        tic = time()
        N = int(gridsamp * X1.shape[0])
        grid, m = buildgrid(X1[0:N, :], self.maxgrid, self.radius, scale=True)
        points = grid.shape[0]
        toc = time()
        print 'State grid points: ' + str(points) + ', of maximum: ' + str(m) + ', Time taken: ' + str(toc - tic)

        if self.first:
            self.W_f.fit(grid, np.zeros(points))
            self.V_f.fit(grid, np.zeros(points))
            self.first = False
        
        Al = np.zeros(points)
        Ah = np.zeros(points)
        if Ascaled:
            for i in range(points):
                Ws = self.W_f_old.predict(grid[i,:])
                Al[i] = A_low(grid[i,:], Ws)
                Ah[i] = A_high(grid[i,:], Ws)
        else:
            for i in range(points):
                Al[i] = A_low(grid[i,:])
                Ah[i] = A_high(grid[i,:])

        # ------------------
        #   Q-learning
        # ------------------
        
        #First iteration
        j = 0
        
        # Q values
        Q = u + self.beta * self.V_f.predict(X1, store_XS=True)
        
        # Fit Q function
        self.Q_f.fit(XA, Q)
        
        # Optimise Q function
        ERROR = self.maximise(grid, Al, Ah, Ascaled, output=output)

        for j in range(ITER):

            # Q values
            Q = u + self.beta * self.V_f.fast_values()

            # Fit Q function
            tic = time()
            self.Q_f.fit(XA, Q)
            toc = time()
            print 'Fit time: ' + str(toc - tic)

            # Optimise Q function
            ERROR = self.maximise(grid, Al, Ah, Ascaled, output=output)

        toc = time()

        print 'Solve time: ' + str(toc - tic)
        
        if plot:
            self.W_f.plot(xargs, showdata=True)
            pylab.show()
            #self.V_f.plot(['x', 1], showdata=True)
            #pylab.show()

    def maximise(self, grid, Al, Ah, Ascaled, plot=False, output=True):

        tic = time()

        if Ascaled:
            Alow = np.zeros(grid.shape[0])
            Ahigh = np.ones(grid.shape[0])
        else:
            Alow = Al
            Ahigh = Ah
        
        N = grid.shape[0]
        W_opt = np.zeros(N)
        V = np.zeros(N)
        Wgrid = np.zeros(0)
        for i in range(N):
            Wgrid = np.append(Wgrid, np.linspace(Alow[i], Ahigh[i], 300))
        x = np.repeat(grid, 300, axis=0)
        X = np.hstack([Wgrid.reshape([N*300,1]), x])
        
        tic = time()
        Qhat = self.Q_f.predict(X)
        toc = time()
        print str(toc - tic)

        j = 0
        for i in range(N):
            idx = np.argmax(Qhat[j:j+300])
            W_opt[i] = Wgrid[j+idx]
            V[i] = Qhat[j+idx]
            j = j + 300
        
        if Ascaled:
            W_opt  = Al[idx] + (Ah[idx] - Al[idx]) * W_opt

        W_opt_old = self.W_f.predict(grid)
        V_old = self.V_f.predict(grid)

        self.V_f.fit(grid, V)
        self.W_f.fit(grid, W_opt, sgd=1, eta=0.4, n_iters=1, scale=0)

        self.p_e = np.mean(abs(W_opt_old - W_opt)/W_opt_old)
        self.v_e = np.mean(abs(V_old - V)/V_old)

        toc = time()
        
        if output:
            print 'Maximisation time: ' + str(toc - tic)
            print 'Value function change: ' + str(round(self.v_e, 4)) + ', Policy change: ' + str(round(self.p_e, 4))

        if plot:
            self.W_f.plot(['x', 1], showdata=True)
            pylab.show()

            self.V_f.plot(['x', 1], showdata=True)
            pylab.show()

        return self.v_e

"""
class QVtile_afterstate:

 
    def __init__(self, R_D, R_T, R_L, R_ms, V_D, V_T, V_L, V_ms, maxgrid, radius, para):

        self.R_f = BIG_Tilecode(R_D, R_T, R_L, min_sample=R_ms)
        self.V_f = BIG_Tilecode(V_D, V_T, V_L, min_sample=V_ms)
        
        Tw = (1 / radius) / 1.8
        T = [Tw for t in range(D)]
        L = int(140 / Tw)
        
        points = maxgrid
        self.W_f = SMALL_Tilecode_m(points, D, T, L)

        self.maxgrid = maxgrid
        self.radius = radius
        self.R_D = R_D
        self.V_D = V_D

        self.first = True
        
        self.beta = para.beta
        self.CORES = para.CPU_CORES
    
    def resetR(self, D, T, L, ms):

        self.R_f = BIG_Tilecode(D, T, L, min_sample=ms)
    
    def iterate(self, XA, X1, u, A_low, A_high, ITER=50, Ascaled=False, plot=True, xargs=[], output=True, gridsamp=1):

        tic = time()

        self.v_e = 0        # Value function error
        self.p_e = 0        # Policy function error

        N = int(gridsamp * X1.shape[0])
        grid, m = buildgrid(X1[0:N, :], self.maxgrid, self.radius, scale=True)
        points = grid.shape[0]
        print 'State grid points: ' + str(points) + ', of maximum: ' + str(m)

        if self.first:
            self.W_f.fit(grid, np.zeros(points))
                 
        Al = np.zeros(points)
        Ah = np.zeros(points)
        if Ascaled:
            for i in range(points):
                Ws = self.W_f.predict(grid[i,:])
                Al[i] = A_low(grid[i,:], Ws)
                Ah[i] = A_high(grid[i,:], Ws)
        else:
            for i in range(points):
                Al[i] = A_low(grid[i,:])
                Ah[i] = A_high(grid[i,:])
       
        if self.first: 
            self.R_f.fit(XA, u)
            self.first = False
        
        # ------------------
        #   Q-learning
        # ------------------

        for j in range(ITER):
            
            # Optimise Q function
            V = self.maximise(grid, Al, Ah, Ascaled, output=output)

            # Q values
            self.V_f.fit(as_trans(XA), V)  

        toc = time()

        print 'Solve time: ' + str(toc - tic)
        
        if plot:
            self.W_f.plot(xargs, showdata=True)
            pylab.show()
            #self.V_f.plot(xargs, showdata=True)
            #pylab.show()

    def maximise(self, grid, Al, Ah, Ascaled, plot=False, output=True):

        tic = time()

        if Ascaled:
            Alow = np.zeros(grid.shape[0])
            Ahigh = np.ones(grid.shape[0])
        else:
            Alow = Al
            Ahigh = Ah
        
        tic1 = time()
        [W_opt, V, state, idx] = self.Q_f.opt(grid, Alow, Ahigh)
        toc1 = time()
        print str(toc1 - tic1)

        if Ascaled:
            W_opt  = Al[idx] + (Ah[idx] - Al[idx]) * W_opt

        W_opt_old = self.W_f.predict(state)
        V_old = self.V_f.predict(state)

        tic2 = time()
        self.V_f.fit(state, V)
        self.W_f.fit(state, W_opt, sgd=1, eta=0.4, n_iters=1, scale=0)
        toc2 = time()
        print str(toc2 - tic2)

        self.p_e = np.mean(abs(W_opt_old - W_opt)/W_opt_old)
        self.v_e = np.mean(abs(V_old - V)/V_old)

        toc = time()
        
        if output:
            print 'Maximisation time: ' + str(toc - tic)
            print 'Value function change: ' + str(round(self.v_e, 4)) + ', Policy change: ' + str(round(self.p_e, 4))

        #self.W_f.plot([1000000, 'x', 1.5, 1], showdata=True)
        #pylab.show()
        #self.V_f.plot([1000000, 'x', 1.5, 1], showdata=True)
        #pylab.show()

        return self.v_e
"""
