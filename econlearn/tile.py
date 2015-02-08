"""Tilecoding based machine learning"""

# Authors:  Neal Hughes <neal.hughes@anu.edu.au>

from __future__ import division
import numpy as np
from time import time
import sys
import pylab
from tilecode import Tilecode

class TilecodeSamplegrid:

    """
    Construct a sample grid (sample of approximately equidistant points) from 
    a  large data set, using a tilecoding data structure

    Parameters
    -----------

    D : int,
        Number of input dimensions

    L : int,
        Number of tilings or 'layers'

    mem_max : float, optional (default = 1)
        Tile array size, values less than 1 turn on hashing

    cores : int, optional (default=1)
        Number of CPU cores to use (fitting stage is parallelized)

    offset : {'optimal', 'random', 'uniform'}, optional
        Type of displacement vector used

    Examples
    --------

    See also
    --------

    Notes
    -----

    This is an approximate method: it is possible that the resulting sample will contain
    some points less than ``radius`` distance apart. The accuracy improves with the number 
    of layers ``L``.

    Currently the tile widths are defined as ``int(1 / radius)**-1``, so small changes in 
    radius may have no effect.

    """
 
    def __init__(self, D, L, mem_max=1, cores=1, offset='optimal'):

        if D == 1 and offset == 'optimal':
            offset = 'uniform'
        
        self.D = D
        self.L = L
        self.mem_max = mem_max
        self.cores = cores
        self.offset= offset
        
    def fit(self, X, radius, prop=1):
        
        """
        Fit a density function to X and return a sample grid with a maximum of M points

        Parameters
        ----------

        X : array of shape [N, D]
            Input data
    
        radius : float in (0, 1)
            minimum distance between points (determines tile widths)

        prop : float in (0, 1), optional (default=1.0)
            Proportion of sample points to return (lowest density points are excluded)
        
        Returns
        -------

        GRID, array of shape [M, D]
            The sample grid with M < N points

        """

        a = np.min(X, axis=0)
        b = np.max(X, axis=0)
        Tr = int(1 / radius)
        T = [Tr + 1] * self.D

        self.tile = Tilecode(self.D, T, self.L, mem_max=self.mem_max , cores=self.cores, offset = self.offset) 
        
        N = X.shape[0]
        GRID, max_points =  self.tile.fit_samplegrid(X, prop)
        self.max_points = max_points

        return GRID


class TilecodeRegressor:

    """    
    Tile coding for function approximation (Supervised Learning).  
    Fits by averaging and/or Stochastic Gradient Descent.
    Supports multi-core fit and predict. Options for uniform, random or 'optimal' displacement vectors.
    Provides option for linear spline extrapolation / filling

    Parameters
    -----------
    
    D : integer
        Total number of input dimensions 
    
    T : list of integers, length D
        Number of tiles per dimension 
    
    L : integer
        Number of tiling 'layers'

    mem_max : double, (default=1)
        Proportion of tiles to store in memory: less than 1 means hashing is used.
    
    min_sample : integer, (default=50) 
        Minimum number of observations per tile

    offset : string, (default='uniform')
        Type of displacement vector, one of 'uniform', 'random' or 'optimal'

    lin_spline : boolean, (default=False)
        Use sparse linear spline model to extrapolate / fill empty tiles

    linT : integer, (default=6)
        Number of linear spline knots per dimension
    
    Attributes
    -----------

    tile : Cython Tilecode instance
    
    """

    def __init__(self, D, T, L, mem_max = 1, min_sample=1, offset='optimal', lin_spline=False, linT=7, cores=4):
        
        if D == 1 and offset == 'optimal':
            offset = 'uniform'

        self.tile = Tilecode(D, T, L, mem_max, min_sample, offset, lin_spline, linT, cores)

    def fit(self, X, Y, method='A', score=False, copy=True, a=0, b=0, pc_samp=1, eta=0.01, n_iters=1, scale=0):

        """    
        Estimate tilecode weights. 
        Supports `Averaging', Stochastic Gradient Descent (SGD) and Averaged SGD.

        Parameters
        -----------
        X : array, shape=(N, D) 
            Input data (unscaled)

        Y : array, shape=(N) 
            Output data (unscaled)

        method : string (default='A')
            Estimation method, one of 'A' (for Averaging), 'SGD' or 'ASGD'.

        score : boolean, (default=False)
            Calculate R-squared

        copy : boolean (default=False)
            Store X and Y

        a : array, optional shape=(D) 
            Percentile to use for minimum tiling range (if not provided set to 0)
        
        b : array, optional, shape=(D) 
            Percentile to use for maximum tiling range (if not provided set to 100)

        pc_samp : float, optional, (default=1)
            Proportion of sample to use when calculating percentile ranges

        eta : float (default=.01)
            SGD Learning rate

        n_iters : int (default=1)
            Number of passes over the data set in SGD

        scale : float (default=0)
            Learning rate scaling factor in SGD
        """

        if method == 'A':
            sgd = False
            asgd = False
        elif method == 'SGD':
            sgd = True
            asgd = False
        elif method == 'ASGD':
            sgd = True
            asgd = True
        
        if X.ndim == 1:
            X = X.reshape([X.shape[0], 1])

        self.tile.fit(X, Y, score=score, copy=copy, a=a, b=b, pc_samp=pc_samp, sgd=sgd, asgd=asgd, eta=eta, scale=scale, n_iters=n_iters)

    def check_memory(self, ):
        
        """
        Provides information on the current memory usage of the tilecoding scheme.
        If memory usage is an issue call this function after fitting and then consider rebuilding the scheme with a lower `mem_max` parameter.
        """

        print 'Number of Layers: ' + str(self.tile.L)
        print 'Tiles per layer: ' + str(self.tile.SIZE)
        print 'Total tiles: ' + str(self.tile.L * self.tile.SIZE)
        print 'Weight array size after hashing: ' + str(self.tile.mem_max)
        temp = np.count_nonzero(self.tile.count) / self.tile.mem_max
        print 'Percentage of weight array active: ' + str(np.count_nonzero(self.tile.count) / self.tile.mem_max)
        mem_max = self.tile.mem_max / (self.tile.L*self.tile.SIZE)
        print '----------------------------------------------'
        print 'Estimated current memory usage (Mb): ' + str((self.tile.mem_max * 2 * 8)/(1024**2))
        print '----------------------------------------------'
        print 'Current hashing parameter (mem_max): ' + str(mem_max)
        print 'Minimum hashing parameter (mem_max): ' + str(temp*mem_max)

    def predict(self, X):
        
        """    
        Return tilecode predicted value 

        Parameters
        -----------
        X : array, shape=(N, D) or (D,)
            Input data

        Returns
        --------
    
        Y : array, shape=(N,)
            Predicted values
        """

        return self.tile.predict(X)

    def plot(self, xargs=['x'], showdata=True):

        """
        Plot the function on along one dimension, holding others fixed 

        Parameters
        -----------
        xargs : list, length = D
            List of variable default values, set plotting dimension to 'x'
            Not required if D = 1

        showdata : boolean, (default=False)
            Scatter training points
        """

        self.tile.plot(xargs=xargs, showdata=showdata)
        pylab.show()

class TilecodeDensity:

    """    
    Tile coding approximation of the pdf of X  
    Fits by averaging. Supports multi-core fit and predict.
    Options for uniform, random or 'optimal' displacement vectors.

    Parameters
    -----------
    
    D : integer
        Total number of input dimensions 
    
    T : list of integers, length D
        Number of tiles per dimension 
    
    L : integer
        Number of tiling 'layers'

    mem_max : double, (default=1)
        Proportion of tiles to store in memory: less than 1 means hashing is used.
    
    min_sample : integer, (default=50) 
        Minimum number of observations per tile

    offset : string, (default='uniform')
        Type of displacement vector, one of 'uniform', 'random' or 'optimal'

    Attributes
    -----------

    tile : Tilecode instance
    
    Examples
    --------

    See also
    --------

    Notes
    -----

    """

    def __init__(self, D, T, L, mem_max = 1, offset='optimal', cores=1):

        if D == 1 and offset == 'optimal':
            offset = 'uniform'
        
        self.tile = Tilecode(D, T, L, mem_max=mem_max, min_sample=1, offset=offset, cores=cores)
    
    def fit(self, X, cdf=False):
        
        N = X.shape[0]

        if X.ndim == 1:
            X = X.reshape([X.shape[0], 1])
        
        self.tile.fit(X, np.zeros(N))
        d = np.array(self.tile.d)
        w = (d**-1) / np.array(self.tile.T) 
        adj = np.product(w)**-1  
        self.tile.countsuminv = (1 / N) * adj

    def predict(self, X):

        return self.tile.predict_prob(X)

    def plot(self, xargs=['x']):

        """
        Plot the pdf along one dimension, holding others fixed 

        Parameters
        -----------
        xargs : list, length = D
            List of variable default values, set plotting dimension to 'x'
            Not required if D = 1
        """

        self.tile.plot_prob(xargs=xargs)
        pylab.show()


#class TilecodeNearestNeighbour:
#
#    """
#    Fast approximate nearest neighbour search using tile coding data structure  
#    
#
#    Parameters
#    -----------
#
#    D : int,
#        Number of input dimensions
#
#    L : int,
#        Number of tilings or 'layers'
#
#    mem_max : float, optional (default = 1)
#        Tile array size, values less than 1 turn on hashing
#
#    cores : int, optional (default=1)
#        Number of CPU cores to use (fitting stage is parallelized)
#
#    offset : {'optimal', 'random', 'uniform'}, optional
#        Type of displacement vector used
#
#    Examples
#    --------
#
#    See also
#    --------
#
#    Notes
#    -----
#
#    This is an approximate method: it is possible that some points > than radius may be included 
#    and some < than radius may be excluded.
#
#    """
# 
#    def __init__(self, D, L, mem_max=1, cores=1, offset='optimal'):
#
#        if D == 1 and offset == 'optimal':
#            offset = 'uniform'
#        
#        self.D = D
#        self.L = L
#        self.mem_max = mem_max
#        self.cores = cores
#        self.offset= offset
#        
#    def fit(self, X, radius, prop=1):
#        
#        """
#        Fit a tile coding data structure to X
#
#        Parameters
#        ----------
#
#        X : array of shape [N, D]
#            Input data
#    
#        radius : float in (0, 1)
#            radius for nearest neighbor queries (determines tile widths)
#
#        """
#
#        a = np.min(X, axis=0)
#        b = np.max(X, axis=0)
#        Tr = int(1 / radius)
#        T = [Tr + 1] * self.D
#
#        #Work out T...
#
#        self.tile = Tilecode(self.D, T, self.L, mem_max=self.mem_max , cores=self.cores, offset = self.offset) 
#        
#        self.tile.fit(X, np.ones(X.shape[0]), unsupervised=True, copy=True)
#
#    def predict(self, X, refine=False):
#
#        """
#        Obtain nearest neighbors (points within distance radius
#
#        Parameters
#        ----------
#
#        X : array of shape [N, D]
#            Query points
#
#        thresh : int, (default=1)
#            Only include points if they are active in at least thresh layers (max is L)
#            Higher thresh values will tend to exclude the points furthest from the query point
#
#        Returns
#        -------
#
#        Y : list of arrays (length = N)
#            Nearest neighbors for each query point
#        """
#
#        self.tile.nearest(X, X.shape[0])

