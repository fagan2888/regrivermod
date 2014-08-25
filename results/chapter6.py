from chartbuilder import *
import numpy as np
import pandas
import pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree

def notrade(results, scenarios, Lambda):

    """
        Generate charts and tables for no trade scenarios
    """
    home = '/home/nealbob'
    folder = '/Dropbox/model/results/chapter6/'
    out = '/Dropbox/Thesis/IMG/chapter6/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/'
    
    n = len(scenarios)
    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['SW', 'S']
    scale = {'SW' : 1000000, 'S' : 1000}
    rows = ['Planner'] + scenarios

    for x in series:

        data0 = []
        # Planner
        record = {}
        for col in cols:
            record[col] = results[scenarios[0]][x][col][0] / scale[x]
        data0.append(record)

        # Policy scenarios
        for scen in scenarios:
            record = {}
            for col in cols:
                record[col] = results[scen][x][col][1] / scale[x]
            data0.append(record)
        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + 'notrade_' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

    data = pandas.DataFrame([Lambda])
    #data.index = 

    with open(home + table_out + 'notrade_Lambda' + '.txt', 'w') as f:
        f.write(data.to_latex(float_format='{:,.2f}'.format))
        f.close()


def trade(results, scenarios, Lambda):

    """
        Generate charts and tables for central case scenarios
    """
    home = '/home/nealbob'
    folder = '/Dropbox/model/results/chapter6/'
    out = '/Dropbox/Thesis/IMG/chapter6/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/'
    
    n = len(scenarios)
    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['SW', 'S', 'U_low', 'U_high']
    scale = {'SW' : 1000000, 'S' : 1000, 'U_low' : 1000000, 'U_high' : 1000000}
    rows = ['Planner'] + scenarios

    for x in series:

        data0 = []
        # Planner
        record = {}
        for col in cols:
            record[col] = results[scenarios[0]][x][col][0] / scale[x]
        data0.append(record)

        # Policy scenarios
        for scen in scenarios:
            record = {}
            for col in cols:
                record[col] = results[scen][x][col][1] / scale[x]
            data0.append(record)
        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + 'trade_' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

    data = pandas.DataFrame([Lambda])
    #data.index = 

    with open(home + table_out + 'trade_Lambda' + '.txt', 'w') as f:
        f.write(data.to_latex(float_format='{:,.2f}'.format))
        f.close()
