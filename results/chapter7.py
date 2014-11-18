from __future__ import division
from chartbuilder import *
import numpy as np
import pandas
import pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree

def planner(results):

    """
        Generate charts and tables for central case scenarios
    """
    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter7/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'
     
    #with open(home + folder + 'central_result.pkl', 'rb') as f:
    #    results = pickle.load(f)
    #    f.close()

    stats_envoff, timeseries_envoff, stats, timeseries = results


    ###### Summary results #####
    
    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    rows = ['Consumptive', 'Optimal']
    series = ['SW', 'Profit', 'B', 'S', 'W', 'E']
    scale = {'SW' : 1000000, 'Profit' : 1000000, 'S' : 1000, 'W' : 1000, 'E' : 1000, 'B' : 1000000}

    m = 1

    for x in series:
        data0 = []

        record = {}
        for col in cols:
            record[col] = stats_envoff[x]['Annual'][col][m] / scale[x]
        data0.append(record)

        record = {}
        for col in cols:
            record[col] = stats[x]['Annual'][col][m] / scale[x]
        data0.append(record)
        
        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + ' ' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()
    
    ###### Environmental flows #####
    
    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    rows = ['Summer', 'Winter', 'Annual']
    series = ['Q_env', 'Q']
    scale = {'Q_env' : 1000, 'Q' : 1000}

    m = 1

    for x in series:
        data0 = []
        for row in rows:
            record = {}
            for col in cols:
                record[col] = stats[x][row][col][m] / scale[x]
            data0.append(record)
        
        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + ' ' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()


    ###### River flows #########

    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    rows = ['Summer', 'Winter', 'Annual']
    series = ['F1', 'F1_tilde', 'F3', 'F3_tilde']
    scale = 1000

    m = 1

    for x in series:
        data0 = []

        for row in rows:
            record = {}
            for col in cols:
                record[col] = stats[x][row][col][m] / scale
            data0.append(record)

        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + ' ' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()
    
    for x in series:
        data0 = []

        for row in rows:
            record = {}
            for col in cols:
                record[col] = stats_envoff[x][row][col][m] / scale
            data0.append(record)

        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + ' ' + x + '_envoff.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

    # Flow duration curves
    data = {'Natural' : timeseries['F1_tilde'][:, 0],
            'Consumptive' : timeseries_envoff['F1'][:, 0],
            'Optimal' :  timeseries['F1'][:, 0] } 
    duration_curve(data, OUTFILE=home + out + 'up_sum' + img_ext)

    data = {'Natural' : timeseries['F1_tilde'][:, 1],
            'Consumptive' : timeseries_envoff['F1'][:, 1],
            'Optimal' :  timeseries['F1'][:, 1] } 
    duration_curve(data, OUTFILE=home + out + 'up_win' + img_ext)
    
    data = {'Natural' : timeseries['F3_tilde'][:, 0],
            'Consumptive' : timeseries_envoff['F3'][:, 0],
            'Optimal' :  timeseries['F3'][:, 0] } 
    duration_curve(data, OUTFILE=home + out + 'down_sum' + img_ext)
    
    data = {'Natural' : timeseries['F3_tilde'][:, 1],
            'Consumptive' : timeseries_envoff['F3'][:, 1],
            'Optimal' :  timeseries['F3'][:, 1] } 
    duration_curve(data, OUTFILE=home + out + 'down_win' + img_ext)


def duration_curve(data, bins=100, XMAX=0, OUTFILE=''):

    chart_params()
    
    xmax = 0
    pylab.figure()
    for x in data:
        values, base = np.histogram(data[x], bins=bins)
        cum = np.cumsum(values) / float(len(data[x]))

        fig = pylab.plot(base[:-1], 1 - cum, label=x)
        
        xmax = max(xmax, np.max(data[x]))
    
    if XMAX == 0:
        XMAX = xmax

    setFigLinesBW(fig[0])
    #pylab.legend()
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.) 
    pylab.xlim(0, XMAX)
    pylab.savefig(OUTFILE, bbox_inches='tight')
    pylab.show()

