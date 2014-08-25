from chartbuilder import *
import numpy as np
import pandas
import pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree

def notrade(results, scenarios):

    """
        Generate charts and tables for no trade scenarios
    """

    home = '/home/nealbob'
    folder = '/Dropbox/model/results/chapter6/risk/'
    out = '/Dropbox/Thesis/IMG/chapter6/risk'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/risk/'
    
    n = len(scenarios)+1
    m = 5 

    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['SW', 'S']
    scale = {'SW' : 1, 'S' : 1000}

    for x in series:

        data0 = []
        
        # Planner
        record = {}
        for col in cols:
            record[col] = results[scenrios[0]][x][col][0] / scale[x]
        data0.append(record)

        # Policy scenarios
        for scen in scenarios:
            record = {}
            for col in cols:
                record[col] = results[scen][x][col][1] / scale[x]
            data0.append(record)
        data = pandas.DataFrame(data0)
        data.index = scenarios

        with open(home + table_out + 'notrade_' + var + '.txt', 'w') as f:
            f.write(tab.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()
