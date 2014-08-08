#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=False, initializedcheck=False

import numpy as np
cimport cython
cimport numpy as np
import pylab

cdef extern from "math.h":
    double c_min "fmin" (double, double)

cdef extern from "math.h":
    double c_max "fmax" (double, double)

def buildgrid(double[:,:] x, int M, double radius, scale = False, stopnum = 1000):

    """
    Generates an approximately equidistant subset of points. 
    
    Returns at most M points at least radius distance apart. 
    Ranks each point by the number of samples within radius
    If there are more than M grid points found, returns the highest scoring points.
    
    Has option for early stopping (by decreasing stopnum) 
    
    ============
    Parameters
    ============

    """

    cdef int N = x.shape[0]
    cdef int D = x.shape[1]
    cdef int m = 1
    cdef double[:,:] xc = np.zeros([N, D])
    cdef double[:] xc_counter = np.zeros([N])
    cdef double r = 0
    cdef double r_min = 10
    cdef int i, j, k
    cdef double n = 0
    cdef double[:] a = np.zeros(D)
    cdef double[:] b = np.zeros(D)
    cdef double[:] d = np.ones(D)
    cdef double[:] d1 = np.ones(D)
    cdef double[:] xs = np.zeros(D)
    cdef int scale_1 = 0
    cdef double[:,:] grid
    cdef double cmax
    cdef int cmax_idx = 0
    cdef int j_star = 0

    cdef int stop_counter = 0
    cdef int stop_num = stopnum

    if scale: 
        atemp = np.min(x, axis=0)
        btemp = np.max(x, axis=0)
        dtemp = (btemp - atemp)**-1
        dtemp1 = (btemp - atemp)
        
        a = atemp
        b = btemp
        d = dtemp
        d1 = dtemp1

        scale_1 = 1
        
    for k in range(D):
        xc[0,k] = (x[0,k] - a[k]) * d[k]
    
    for i in range(1, N):
        for k in range(D):
            xs[k] = (x[i,k] - a[k]) * d[k]
        r_min = 10
        j_star = 0
        for j in range(m):
            r = 0
            for k in range(D):
                r += (xs[k] - xc[j, k])**2
            r **= 0.5
            if r < r_min:
                r_min = r
                j_star = j
        if r_min > radius:
            m += 1
            for k in range(D):
                xc[m - 1,k] = xs[k]
            stop_counter = 0
        else:
            xc_counter[j_star] += 1
            stop_counter += 1 
        if stop_counter > stop_num and m > M:
            break
    
    M = <int> c_min(M, m)
    grid = np.zeros([M, D])

    for j in range(M): 
        for i in range(m):
            if xc_counter[i] > cmax:
                cmax = xc_counter[i]
                cmax_idx = i
        cmax = 0
        xc_counter[cmax_idx] = -1
        if scale_1 == 1: 
            for k in range(D):
                grid[j,k] = xc[cmax_idx, k] * d1[k] + a[k]
        else:
            for k in range(D):
                grid[j,k] = xc[cmax_idx, k]
    
    return [np.asarray(grid), m]


def test():

    pylab.ioff()
    fig_width_pt = 350 			     # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27                # Convert pt to inch
    golden_mean = ((5**0.5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt   # width in inches   
    fig_height = fig_width*1 #golden_mean       # height in inches
    fig_size =  [fig_width,fig_height]

    params = { 'backend': 'ps',
           'axes.labelsize': 10,
           'text.fontsize': 10,
           'legend.fontsize': 10,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8,
           'text.usetex': True,
           'figure.figsize': fig_size }

    home = '/home/nealbob'
    out = '/Dropbox/Thesis/IMG/chapter8/'
    img_ext = '.pdf'
    
    pylab.rcParams.update(params)
    x1 = np.random.normal(size=10000)
    x2 = np.random.normal(size=10000)
    X = np.array([x1, x2]).T
    pylab.scatter(x1, x2)
    pylab.ylim(-4, 4)
    pylab.xlim(-4, 4)
    pylab.savefig(home + out + 'scatter.pdf', bbox_inches='tight')
    pylab.show()

    grid, m = buildgrid(X, 100, 0.4, scale=False, stopnum=2000)
    pylab.scatter(grid[:,0], grid[:, 1])
    pylab.ylim(-4, 4)
    pylab.xlim(-4, 4)
    pylab.savefig(home + out + 'approx_grid.pdf', bbox_inches='tight')
    pylab.show()

    grid, m = buildgrid(X, 100, 0.4, scale=False, stopnum=10000)
    pylab.scatter(grid[:,0], grid[:, 1])
    pylab.ylim(-4, 4)
    pylab.xlim(-4, 4)
    pylab.savefig(home + out + 'grid.pdf', bbox_inches='tight')
    pylab.show()

