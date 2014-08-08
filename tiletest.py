import numpy as np
from econlearn.tilecode import Tilecode
from econlearn.tilecode import Function_Group as FG
import pylab
from econlearn.samplegrid import buildgrid
import time

#GRID = 10
#N = GRID ** D

D = 4
radius = 0.1
N = 150000
T = [int((1 / radius) / 2) for d in range(D)]
print T
L = 10
noise = 0.05

#X_grid = np.linspace(0, 1000, GRID)
#[Xi1, Xi2] = np.mgrid[0:GRID, 0:GRID]
#X1 = X_grid[Xi1.flatten()]
#X2 = X_grid[Xi2.flatten()]
#X = np.array([X1, X2]).T

x = np.random.rand(N)
X = np.array([(x + np.random.rand(N)*0.5) for d in range(D)]).T

grid, m = buildgrid(X, 500, radius, scale=True, stopnum = 400) 
print 'Done'
Ymean  = grid[:,0] - 0.5 * grid[:,0]**2
Y = Ymean + np.random.rand(500)*noise - noise/2.0

newTile = Tilecode(D, T, L, mem_max = 1, min_sample=1, offset='optimal', lin_spline=True, linT=5, cores=4)

newTile.fit(grid, Y, score=True) #, unsupervised=True) #, score=True, copy=True, sgd=True, eta=0.1, scale=(1/500000.0), n_iters=1, asgd=True, storeindex=True, a=[0, 0, 0, 0], b=[100, 100, 100, 90], pc_samp=0.25)
print 'Done'
Yhat = newTile.predict(grid)#, store_XS=True)
#actions, _, _, _ = newTile.opt(X, np.zeros(N), np.ones(N)*2)
#print 'Optimize: ' + str(np.mean((actions-1.0)**2))

idx = Yhat != 0
error = Yhat[idx] - Ymean[idx]
score = np.mean(error**2) 
print 'Score : ' + str(score)

xargs = [0.5 for d in range(D)]
xargs[0] = 'x'
newTile.plot(xargs, showdata=True)
#Xplot = np.linspace(0,2, 500)
#Yplot = Xplot - 0.5*Xplot**2  
#pylab.plot(Xplot, Yplot)
pylab.show()

#newTile.partial_fit(Y, 1) #score=True, copy=True, unsupervised=False, sgd=False, eta=0.5, scale=0.0001, n_iters=1, asgd=True, storeindex=True)

#Yhat = newTile.fast_values()

#Yhat = newTile.predict(X, store_XS=True)
#actions, _, _, _ = newTile.opt(X, np.zeros(N), np.ones(N)*2)
#print 'Optimize: ' + str(np.mean((actions-1.0)**2))
#idx = Yhat != 0
#error = Yhat[idx] - Ymean[idx]
#score = np.mean(error**2) 
#print 'Score : ' + str(score)

#xargs = [0.5 for d in range(D)]
#xargs[0] = 'x'
#newTile.plot(xargs)
#Xplot = np.linspace(0,2, 500)
#Yplot = Xplot - 0.5*Xplot**2  
#pylab.plot(Xplot, Yplot)
#pylab.show()


"""
T = [8 for d in range(D)]
L = 30

new2Tile = Tilecode(D, T, L, mem_max = 1, min_sample=1, offset='optimal', lin_spline=False, cores=4 )
new2Tile.fit(X, Y, score=True, copy=True, unsupervised=False, sgd=False, eta=0.3, scale=0.0001, n_iters=1, ASGD=False)
#eta = 0.25 scale = 0.01
#eta = 0.7, scale = 0.005
#eta = 1, scale = 0.1
Yhat = new2Tile.predict(X)
#actions, _, _, _ = new2Tile.opt(X, np.zeros(N), np.ones(N)*2)
#print 'Optimize: ' + str((np.mean(actions-1.0))**2)
idx = Yhat != 0
error = Yhat[idx] - Ymean[idx]
score2 = np.mean(error**2) 
print 'Score : ' + str(score2)

xargs = [0.5 for d in range(D)]
xargs[0] = 'x'
new2Tile.plot(xargs)

Xplot = np.linspace(0,1.5, 500)
Yplot = Xplot - 0.5*Xplot**2  
pylab.plot(Xplot, Yplot)
pylab.show()

if check_partial:
    Y2 = Y * 2
    newTile.partial_fit(Y2)
    xargs = [0.5 for d in range(D)]
    xargs[0] = 'x'
    newTile.plot(xargs)
    pylab.show()
"""

NN = 100
Nlow = 50
tic = time.time()
group = FG(NN, Nlow, newTile, newTile)
toc = time.time()
print str(toc - tic)

XX = np.array([np.random.rand(NN) for d in range(D)]).T
v = np.zeros(NN)

#np.vstack([X for i in range(NN)])
#group.predict(XX)
#group.plot(['x', 500], showdata=True)
#pylab.show()

"""
#newTile.plot(['x', 0.5, 0.5, 0.5, 0.5], quad=True)
#xargs = [0.5 for d in range(D)]
#newTile.predict_quad(np.array(xargs))
#newTile.local_quadratic(XX, 2)
#newTile.plot_quadratic(['x', 0.5, 0.5, 0.5, 2)
"""


