from chartbuilder import *
import numpy as np
import pandas
import pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree


def tables(results, scenarios, Lambda, LambdaK, label='central', risk=False):

    """
        Generate charts and tables for central case scenarios
    """
    home = '/home/nealbob'
    folder = '/Dropbox/model/results/chapter6/'
    out = '/Dropbox/Thesis/IMG/chapter6/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/'

    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['SW', 'S', 'U_low', 'U_high', 'X_low', 'X_high', 'S_low', 'S_high', 'trade']
    scale = {'SW': 1000000, 'S': 1000, 'U_low': 1000000, 'U_high': 1000000, 'X_low': 1000, 'X_high': 1000,
             'S_low': 1, 'S_high': 1, 'trade': 1000}
    if risk:
        scale['SW'] = 1
        scale['U_low'] = 1
        scale['U_high'] = 1

    rows = ['Planner'] + scenarios
    m = len(results[scenarios[0]][series[0]][cols[0]]) - 1

    # Calculate user storage percentages
    for x in ['S_low', 'S_high']:
        for scen in scenarios:
            for col in cols:
                results[scen][x][col][:] /= results[scen][x]['Max'][m]

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
                record[col] = results[scen][x][col][m] / scale[x]
            data0.append(record)
        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + label + ' ' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

    data = pandas.DataFrame([Lambda])

    with open(home + table_out + label + '_Lambda' + '.txt', 'w') as f:
        f.write(data.to_latex(float_format='{:,.2f}'.format))
        f.close()
    
    labels = ['CS', 'RS-O', 'RS-HL-O', 'CS-O', 'CS-HL-O', 'CS-U']
    values = [Lambda[l] for l in labels]
    labels[0] = 'Arbitrary'

    chart = {'OUTFILE': home + out + label + '_Lambdahigh' + img_ext,
             'XLABEL': '$\Lambda_{high}$',
             'LABELS': labels}
    data = [np.arange(len(labels)), values]
    build_chart(chart, data, chart_type='barh')

    data = pandas.DataFrame([LambdaK])

    with open(home + table_out + label + '_LambdaK' + '.txt', 'w') as f:
        f.write(data.to_latex(float_format='{:,.2f}'.format))
        f.close()

    # # Social welfare chart

    arbSW = np.array([results['RS']['SW']['Mean'][m] / scale['SW'],
                      results['RS-HL']['SW']['Mean'][m] / scale['SW'],
                      results['CS']['SW']['Mean'][m] / scale['SW'],
                      results['CS-HL']['SW']['Mean'][m] / scale['SW']
    ])

    optSW = np.array([results['RS-O']['SW']['Mean'][m] / scale['SW'],
                      results['RS-HL-O']['SW']['Mean'][m] / scale['SW'],
                      results['CS-O']['SW']['Mean'][m] / scale['SW'],
                      results['CS-HL-O']['SW']['Mean'][m] / scale['SW']
    ])

    ymin = 170
    ymax = 187
    if risk:
        ymin = 87
        ymax = 93

    chart = {'OUTFILE': home + out + label + '_SW' + img_ext,
             'YLABEL': 'Mean social welfare',
             'XLABEL': 'Policy scenario',
             'LABELS': ('', 'RS', '', 'RS-HL', '', 'CS', '', 'CS-HL'),
             'LEGEND': ('Arbitrary shares', 'Optimal shares'),
             'WIDTH': 0.3,
             'YMIN': ymin,
             'YMAX': ymax}
    data = [[np.arange(4), arbSW], [np.arange(4) + chart['WIDTH'], optSW]]
    build_chart(chart, data, chart_type='bar', ylim=True)

def tables2(results, scenarios, label='central'):

    """
        Generate charts and tables for central case scenarios
    """
    home = '/home/nealbob'
    folder = '/Dropbox/model/results/chapter6/'
    out = '/Dropbox/Thesis/IMG/chapter6/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/'

    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['S_low', 'S_high']
    rows = ['Planner'] + scenarios
    m = len(results[scenarios[0]][series[0]][cols[0]]) - 1

    for x in series:

        data0 = []
        # Planner
        record = {}
        for col in cols:
            record[col] = results[scenarios[0]][x][col][0]
        data0.append(record)

        # Policy scenarios
        for scen in scenarios:
            record = {}
            for col in cols:
                record[col] = results[scen][x][col][m]
            data0.append(record)
        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + label + ' ' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

