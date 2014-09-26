from chartbuilder import *
import numpy as np
import pandas
import pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree

def tables(results, scenarios, Lambda, LambdaK, label='central'):

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
    series = ['SW', 'S', 'U_low', 'U_high', 'X_low', 'X_high', 'S_low', 'S_high', 'trade']
    scale = {'SW' : 1000000, 'S' : 1000, 'U_low' : 1000000, 'U_high' : 1000000, 'X_low' : 1000, 'X_high' : 1000, 'S_low' : 1000, 'S_high' : 1000, 'trade' : 1000}
    rows = ['Planner'] + scenarios
    m = len(results[scenarios[0]][series[0]][cols[0]]) - 1

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
    
    data = pandas.DataFrame([LambdaK])

    with open(home + table_out + label + '_LambdaK' + '.txt', 'w') as f:
        f.write(data.to_latex(float_format='{:,.2f}'.format))
        f.close()

    ## Social welfare chart
   
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
    
    chart = {'OUTFILE': home + out + label + '_SW' + img_ext,
     'YLABEL': 'Mean social welfare ($m)',
     'XLABEL': 'Policy scenario',
     'LABELS': ('', 'RS', '', 'RS-HL', '', 'CS', '', 'CS-HL'),
     'LEGEND': ('Arbitrary shares', 'Optimal shares'),
     'WIDTH': 0.3}
    data = [[np.arange(4), arbSW], [np.arange(4) + chart['WIDTH'], optSW]]
    build_chart(chart, data, chart_type='bar')
