from __future__ import division
from chartbuilder import *
import numpy as np
import pandas
import pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree


def env_demands(stats, timeseries):

    """
        Generate charts and tables for central case scenarios
    """
    home = '/home/nealbob'
    folder = '/Dropbox/model/results/chapter7/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'

    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    rows = ['Summer', 'Winter', 'Annual']
    series = ['F1', 'F1_tilde', 'F3', 'F3_tilde']
    scale = 1000

    m = 0

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

    # Flow duration curves
    duration_curve(timeseries['F1_tilde'][:, 0], timeseries['F1'][:, 0], OUTFILE=home + out + 'up_sum' + img_ext)
    duration_curve(timeseries['F1_tilde'][:, 1], timeseries['F1'][:, 1], OUTFILE=home + out + 'up_win' + img_ext)
    duration_curve(timeseries['F3_tilde'][:, 0], timeseries['F3'][:, 0], OUTFILE=home + out + 'down_sum' + img_ext)
    duration_curve(timeseries['F3_tilde'][:, 1], timeseries['F3'][:, 1], OUTFILE=home + out + 'down_win' + img_ext)


def duration_curve(data1, data2, bins=100, XMAX=0, OUTFILE=''):

    chart_params()

    if XMAX == 0:
        XMAX = max(np.max(data1), np.max(data2))

    values1, base1 = np.histogram(data1, bins=bins)
    cum1 = np.cumsum(values1) / float(len(data1))

    values2, base2 = np.histogram(data2, bins=bins)
    cum2 = np.cumsum(values2) / float(len(data2))

    pylab.plot(base1[:-1], 1 - cum1)
    pylab.plot(base2[:-1], 1 - cum2)
    pylab.xlim(0, XMAX)

    pylab.savefig(OUTFILE, bbox_inches='tight')
    pylab.show()

