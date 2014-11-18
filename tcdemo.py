import numpy as np
import econlearn.tilecode as tc
import matplotlib.pyplot as plt
import pylab
import results.chartbuilder as cb
import sklearn.linear_model as sk 

x = np.random.rand(150)
y = np.maximum(-0.5 + x, 0) + np.random.rand(150)*.2
X = x.reshape([150, 1])
minx = min(X)
maxx = max(X)
XS = np.linspace(minx, maxx, 200).reshape([200,1])

home = '/home/nealbob'
out = '/Dropbox/Thesis/IMG/chapter8/'
img_ext = '.pdf'

print 'The noise of a high resolution grid'
test = tc.Tilecode(1, [60], 1, min_sample=1, offset='uniform', lin_spline=True, linT=5)
test.fit(X, y)
yhat = test.predict(XS)

pylab.figure()
pylab.plot(x, y, '+')
pylab.plot(XS, yhat)
pylab.savefig(home + out + 'noisytile' + img_ext, bbox_inches='tight')
pylab.show()

print 'The bias of a low resolution grid'
test = tc.Tilecode(1, [5], 1, min_sample=1, offset='uniform', lin_spline=True, linT=5)
test.fit(X, y)
yhat = test.predict(XS)

plt.plot(x, y, '+')
plt.plot(XS, yhat)
pylab.savefig(home + out + 'biasedtile' + img_ext, bbox_inches='tight')
plt.show()

print 'High resolution tilecoding'
test = tc.Tilecode(1, [3], 40, min_sample=1, offset='uniform', lin_spline=True, linT=5)
test.fit(X, y, sgd=True, asgd=False, eta=0.3, n_iters=10, scale=0)
yhat = test.predict(XS)
#test2 = tc.Tilecode(1, [5], 30, min_sample=1, offset='uniform', lin_spline=True, linT=5)
#test2.fit(X, y) #sgd=True, eta=0.5, n_iters=10, scale=0)
#yhat2 = test2.predict(XS)

plt.plot(x, y, '+')
plt.plot(XS, yhat)
pylab.savefig(home + out + 'goodtile' + img_ext, bbox_inches='tight')
#plt.plot(XS, yhat2)
plt.show()

print 'The bias of averaging'
test = tc.Tilecode(1, [2], 50, min_sample=1, offset='uniform', lin_spline=True, linT=5)
test.fit(X, y)
yhat = test.predict(XS)

plt.plot(x, y, '+')
plt.plot(XS, yhat)
pylab.savefig(home + out + 'averagetile' + img_ext, bbox_inches='tight')
plt.show()

print 'A linear model'
test = sk.LinearRegression()
test.fit(X, y)
yhat = test.predict(XS)

plt.plot(x, y, '+')
plt.plot(XS, yhat)
plt.ylim(0, 0.7)
pylab.savefig(home + out + 'lineartile' + img_ext, bbox_inches='tight')
plt.show()



