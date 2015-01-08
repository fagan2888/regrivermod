from __future__ import division
import numpy as np
from time import time
import sys
import pylab
from tilecode import Tilecode
from samplegrid import buildgrid 
from sklearn.ensemble import ExtraTreesRegressor as Tree
#from sklearn.ensemble import RandomForestRegressor as Tree
from tile import TilecodeSamplegrid

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
 
    def __init__(self, D, T, L, mem_max, ms, radius, para, asgd=False, linT=6, init=False, W_f=0, V_f=0):

        self.Q_f = Tilecode(D + 1, T, L, mem_max, min_sample=ms, cores=para.CPU_CORES)
        
        self.radius = radius
        
        if init:
            self.W_f = W_f
            self.V_f = V_f
            self.first = False
        else:
            Twv = int((1 / self.radius) / 2)
            T = [Twv for t in range(D)]
            L = int(130 / Twv)
            self.W_f = Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=para.CPU_CORES)
            self.V_f = Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=para.CPU_CORES)
            self.first = True
        
        self.D = D
        self.beta = para.beta
        self.CORES = para.CPU_CORES
        self.asgd = asgd
   

    def resetQ(self, D, T, L, mem_max, ms):

        self.Q_f = Tilecode(D + 1, T, L, mem_max, min_sample=ms, cores=self.CORES)
    
    
    def iterate(self, XA, X1, u, A_low, A_high, ITER=50, Ascaled=False, plot=True, xargs=[], output=True, a = 0, b = 0, pc_samp=1, maxT=60000, eta=0.8, tilesg=False, sg_prop=0.96, sg_samp=1, sg_points=100, sgmem_max=0.4, plotiter=False, test=False, NS=False):

        tic = time()

        self.v_e = 0        # Value function error
        self.p_e = 0        # Policy function error
        
        T = XA.shape[0]
        
        self.value_error = np.zeros(ITER)
        
        if not(tilesg):
            grid, m = buildgrid(X1, sg_points, self.radius, scale=True, stopnum=X1.shape[0])
        else: 
            nn = int(X1.shape[0]*sg_samp)
            tic = time()
            tile = TilecodeSamplegrid(X1.shape[1], 25, mem_max=sgmem_max, cores=self.CORES)
            grid = tile.fit(X1[0:nn], self.radius, prop=sg_prop)
            toc = time()
            print 'State grid points: ' + str(grid.shape[0]) + ', of maximum: ' + str(tile.max_points) + ', Time taken: ' + str(toc - tic)
            del tile
         
        points = grid.shape[0]

        ticfit = time()
        if self.first:
            self.W_f.fit(grid, np.zeros(points), NS=NS)
            self.V_f.fit(grid, np.zeros(points), NS=NS)
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
            minpol = np.min(Al)
            maxpol = np.max(Ah)
        
        tocfit = time()
        print 'Constraint time: ' + str(tocfit - ticfit)
        
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
        ticfit = time()
        Q = u + self.beta * self.V_f.predict(X1, store_XS=precompute)
        tocfit = time()
        print 'V prediction time: ' + str(tocfit - ticfit)
        
        # Fit Q function
        ticfit = time()
        self.Q_f.fit(XA, Q, pa=minpol, pb=maxpol , copy=True, unsupervised=precompute, sgd=self.asgd, asgd=self.asgd, eta=eta, n_iters=1, scale=1* (1 / min(T, maxT)), storeindex=(self.asgd and precompute), a=a, b=b, pc_samp=pc_samp)
        tocfit = time()
        print 'Q Fitting time: ' + str(tocfit - ticfit)

        # Optimise Q function
        self.value_error[0], W_opt, state = self.maximise(grid, Al, Ah, Ascaled, output=output, plotiter=plotiter, xargs=xargs, NS=NS)
         
        for j in range(1, ITER):
            # Q values
            Q = u + self.beta * self.V_f.fast_values()
            
            # Fit Q function
            self.Q_f.partial_fit(Q, 0)

            # Optimise Q function
            self.value_error[j], W_opt, state = self.maximise(grid, Al, Ah, Ascaled, output=output, plotiter=plotiter, xargs=xargs, NS=NS)
            
        ticfit = time()
        NN = min(X1.shape[0], 20000)
        W_opt_old = self.W_f.predict(X1[0:NN,:])
        self.W_f.fit(state, W_opt, sgd=0, eta=0.1, n_iters=5, scale=0, NS=NS)
        W_opt_new = self.W_f.predict(X1[0:NN,:])
        self.pe = np.mean((W_opt_old - W_opt_new))/np.mean(W_opt_old)
        toc = time()
        tocfit = time()
        print 'Policy time: ' + str(tocfit - ticfit)
        
        print 'Solve time: ' + str(toc - tic) + ', Policy change: ' + str(self.pe)
        
        if plot:
            xargstemp = xargs
            self.W_f.plot(xargs, showdata=True)
            pylab.show()
            self.V_f.plot(xargstemp, showdata=True)
            pylab.show()

    def maximise(self, grid, Al, Ah, Ascaled, plot=False, output=True, plotiter=False, xargs=0, NS=False):

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
        print 'OPT POINTS: ' + str(len(W_opt))

        if Ascaled:
            W_opt  = Al[idx] + (Ah[idx] - Al[idx]) * W_opt

        V_old = self.V_f.predict(state)
        
        self.V_f.fit(state, V, sgd=0, eta=0.1, n_iters=5, scale=0, NS=NS)
        

        if np.count_nonzero(V_old) < V_old.shape[0]:
            self.ve = 1
        else:
            self.ve = np.mean(abs(V_old - V)/V_old)

        toc = time()
        
        if output:
            print 'Value change: ' + str(round(self.ve, 3)) + '\t---\tMax time: ' + str(round(toc - tic, 4))

        if plotiter:
            self.W_f.fit(state, W_opt, sgd=0, eta=0.1, n_iters=5, scale=0, NS=NS)
            xargstemp = xargs
            self.W_f.plot(xargs, showdata=True)
            pylab.show()
            self.V_f.plot(xargstemp, showdata=True)
            pylab.show()
        
        return [self.ve, W_opt, state]

class QVtile_ch7:

    """
    Solve MDP by Q-V iteration, with one binary state variable
    
    Attributes
    -----------

    W_f :
        [Policy function, Policy function]
    V_f :
        Value function
    Q_f :
        Action-value function
    """
 
    def __init__(self, D, T, L, mem_max, ms, radius, para, asgd=False, linT=6, init=False, W_f=0, V_f=0):
        
        self.M = [0,1]
        M = self.M
        self.Q_f = [0,0]
        self.W_f = [0,0]
        self.V_f = [0,0]

        self.Q_f = [Tilecode(D + 1, T[m], L, mem_max, min_sample=ms, cores=para.CPU_CORES) for m in M]
        
        self.radius = radius
        
        if init:
            self.W_f = W_f
            self.V_f = V_f
            self.first = False
        else:
            Twv = int((1 / self.radius) / 2)
            T = [Twv for t in range(D)]
            L = int(130 / Twv)
            self.W_f = [Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=para.CPU_CORES) for m in M]
            self.V_f = [Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=para.CPU_CORES) for m in M]
            self.first = True
        
        self.D = D
        self.beta = para.beta
        self.CORES = para.CPU_CORES
        self.asgd = asgd

    def resetQ(self, D, T, L, mem_max, ms):

        self.Q_f = [Tilecode(D + 1, T[m], L, mem_max, min_sample=ms, cores=self.CORES) for m in self.M]
    
    def iterate(self, XA, X1, u, A_low, A_high, ITER=50, plot=True, xargs=[], output=True, a = 0, b = 0, pc_samp=1, maxT=60000, eta=0.8, tilesg=False, sg_prop=0.96, sg_samp=1, sg_points=100, sgmem_max=0.4, plotiter=False, test=False):

        tic = time()
        M = self.M
        self.v_e = 0        # Value function error
        self.p_e = 0        # Policy function error
        
        T = [XA[m].shape[0] for m in M]
        
        self.value_error = np.zeros(ITER)
        
        grid = [0,0] 
        
        if not(tilesg):
            grid[0], _ = buildgrid(X1[0], sg_points, self.radius, scale=True, stopnum=X1[0].shape[0])
            grid[1], _ = buildgrid(X1[1], sg_points, self.radius, scale=True, stopnum=X1[1].shape[0])
        else: 
            for m in range(2):
                nn = int(X1[m].shape[0]*sg_samp)
                tic = time()
                tile = TilecodeSamplegrid(X1[m].shape[1], 25, mem_max=sgmem_max, cores=self.CORES)
                grid[m] = tile.fit(X1[m][0:nn], self.radius, prop=sg_prop)
                toc = time()
                print 'State grid points: ' + str(grid[m].shape[0]) + ', of maximum: ' + str(tile.max_points) + ', Time taken: ' + str(toc - tic)
                del tile
        points = [grid[m].shape[0] for m in M]

        ticfit = time()
        if self.first:
            [self.W_f[m].fit(grid[m], np.zeros(points[m])) for m in M]
            [self.V_f[m].fit(grid[m], np.zeros(points[m])) for m in M]
            self.first = False
        
        Al = [np.zeros(points[m]) for m in M]
        Ah = [np.zeros(points[m]) for m in M]
        for m in range(2):
            for i in range(points[m]):
                Al[m][i] = A_low(grid[m][i,:])
                Ah[m][i] = A_high(grid[m][i,:])
        minpol = [np.min(Al[m]) for m in M]
        maxpol = [np.max(Ah[m]) for m in M]
        
        tocfit = time()
        print 'Constraint time: ' + str(tocfit - ticfit)
        
        if ITER == 1:
            precompute = False
        else:
            precompute = True

        W_opt = [0,0]
        state = [0,0]
        Q = [0,0]
        # ------------------
        #   Q-learning
        # ------------------

        #First iteration
        j = 0

        ticfit = time()
        m1 = 1
        for m in range(2):
            
            # Q values
            Q[m] = u[m] + self.beta * self.V_f[m1].predict(X1[m], store_XS=precompute)
            tocfit = time()
            print 'V prediction time: ' + str(tocfit - ticfit)
            
            # Fit Q function
            ticfit = time()
            self.Q_f[m].fit(XA[m], Q[m], pa=minpol[m], pb=maxpol[m] , copy=True, unsupervised=precompute, sgd=self.asgd, asgd=self.asgd, eta=eta, n_iters=1, scale=1* (1 / min(T[m], maxT)), storeindex=(self.asgd and precompute), a=a, b=b, pc_samp=pc_samp)
            tocfit = time()
            print 'Q Fitting time: ' + str(tocfit - ticfit)

            # Optimise Q function
            value_error, W_opt[m], state[m] = self.maximise(m, grid[m], Al[m], Ah[m], output=output, plotiter=plotiter, xargs=xargs)
            m1 = 0
            if test:
                import pdb; pdb.set_trace()

        for j in range(1, ITER):
            m1 = 1
            for m in range(2):
                # Q values
                Q[m] = u[m] + self.beta * self.V_f[m1].fast_values()
                
                # Fit Q function
                self.Q_f[m].partial_fit(Q[m], 0)

                # Optimise Q function
                value_error, W_opt[m], state[m] = self.maximise(m, grid[m], Al[m], Ah[m], output=output, plotiter=plotiter, xargs=xargs)
                m1 = 0
                if test:
                    import pdb; pdb.set_trace()

        self.pe = [0,0]
        for m in range(2):
            ticfit = time()
            NN = min(X1[m].shape[0], 20000)
            W_opt_old = self.W_f[m].predict(X1[m][0:NN,:])
            self.W_f[m].fit(state[m], W_opt[m], sgd=0, eta=0.1, n_iters=5, scale=0)
            W_opt_new = self.W_f[m].predict(X1[m][0:NN,:])
            self.pe[m] = np.mean((W_opt_old - W_opt_new)/W_opt_old)
            toc = time()
            tocfit = time()
            print 'Policy time: ' + str(tocfit - ticfit)
        
        print 'Solve time: ' + str(toc - tic) + ', Policy change: ' + str(self.pe)
        
        if plot:
            xargstemp1 = xargs
            xargstemp2 = xargs
            for m in range(2):
                xargs1 = xargstemp1
                self.W_f[m].plot(xargs1, showdata=True)
                pylab.show()
                xargs2 = xargstemp2
                self.V_f[m].plot(xargs2, showdata=True)
                pylab.show()
                xargstemp1 = xargs
                xargstemp2 = xargs
    
    def maximise(self, m, grid, Al, Ah, plot=False, output=True, plotiter=False, xargs=0):

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
        
        Returns
        -----------

        ERROR: float
            Mean absolute deviation

        """
        tic = time()

        W_old = self.W_f[m].predict(grid)
        
        Alow = Al
        Ahigh = Ah
        
        X =  np.vstack([W_old, grid.T]).T
        
        [W_opt, V, state, idx] = self.Q_f[m].opt(X, Alow, Ahigh)
        nidx = np.array([not(i) for i in idx])
        print 'OPT POINTS: ' + str(len(W_opt))

        V_old = self.V_f[m].predict(state)
        
        self.V_f[m].fit(state, V, sgd=0, eta=0.1, n_iters=5, scale=0)
        

        if np.count_nonzero(V_old) < V_old.shape[0]:
            self.ve = 1
        else:
            self.ve = np.mean(abs(V_old - V)/V_old)

        toc = time()
        
        if output:
            print 'Value change: ' + str(round(self.ve, 3)) + '\t---\tMax time: ' + str(round(toc - tic, 4))

        if plotiter:
            self.W_f[m].fit(state, W_opt, sgd=0, eta=0.1, n_iters=5, scale=0)
            xargstemp = xargs
            self.W_f[m].plot(xargs, showdata=True)
            pylab.show()
            self.V_f[m].plot(xargstemp, showdata=True)
            pylab.show()
        
        return [self.ve, W_opt, state]

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
