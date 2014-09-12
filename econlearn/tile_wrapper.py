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
    a  large data set, using a tilecoding (over lapping grid) data structure

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
    
    This algorithm iterates over the input samples each time applying a fast approximate nearest 
    neighbours search (based on tilecoding) to delete neighboring points, leaving 
    a small sample of approximately equally spaced points. 

    This is an approximate method: it is possible that the resulting sample will contain
    some points less than ``radius`` distance apart. The accuracy improves with the number 
    of layers ``L``.

    Currently the tile widths are defined as ``int(1 / radius)**-1``, so small changes in 
    radius may have no effect.

    """
 
    def __init__(self, D, L, mem_max=1, cores=1, offset='optimal'):

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



