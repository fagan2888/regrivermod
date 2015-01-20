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


def simple_share_model(n=10):

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter6/lambda/'
    model = '/Dropbox/Model/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'
   
    results = []
    paras = []

    for i in range(n):
        if i != 9:
            with open(home + folder + 'lambda_result_' + str(i) +'.pkl', 'rb') as f:
                results.extend(pickle.load(f))
                f.close()

            with open(home + folder + 'lambda_para_' + str(i) + '.pkl', 'rb') as f:
                paras.extend(pickle.load(f))
                f.close()
    
    nn = (n - 1) * 10

    Y = np.zeros([nn, 4])
    X = np.zeros([nn, 12])

    for i in range(nn):
       
        Y[i, 0] = results[i][0][1][0]
        Y[i, 1] = results[i][0][1][1]
        Y[i, 2] = results[i][1][1][0]
        Y[i, 3] = results[i][1][1][1]
        
        X[i, :] = np.array([paras[i][p] for p in paras[i]])

    """    
    tree = Tree(min_samples_split=3, min_samples_leaf=2, n_estimators = 300)
    tree.fit(X, Y)
    
    with open(home + model + 'sharemodel.pkl', 'wb') as f:
       pickle.dump(tree, f)
       f.close()
    
    scen = ['RS-O', 'CS-O', 'RS-HL-O', 'CS-HL-O']

    for i in range(4):
    
        chart = {'OUTFILE' : (home + out + 'lambda_' + scen[i] + img_ext),
                 'XLABEL' : 'Optimal flow share',
                 'XMIN' : min(Y[:,i]),
                 'XMAX' : max(Y[:,i]),
                 'BINS' : 10}
        data = [Y[:,i]]
        build_chart(chart, data, chart_type='hist')

        chart = {'OUTFILE' : (home + out + 'lambda_scat_' + scen[i] + img_ext),
                 'XLABEL' : 'Number of high reliability users',
                 'YLABEL' : 'Optimal flow share'}
        data = [[X[:, 2], Y[:,i]]]
        build_chart(chart, data, chart_type='scatter')
    
    
    rank = tree.feature_importances_ * 100
    
    data0 = []
    for i in range(len(paras[0])):
        record = {}
        record['Importance'] = rank[i]
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = [p for p in paras[i]]
    tab = tab.sort(columns=['Importance'], ascending=False)
    
    with open(home + table_out + 'lambda' + '.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.2f}'.format))
        f.close()
    """  

    from sklearn.linear_model import LinearRegression as OLS
    ols = OLS()
    
    ols.fit(X[:,2].reshape([190, 1]), Y[:,1])
    CS_c = ols.intercept_
    CS_b = ols.coef_[0]
    xp = np.linspace(30, 70, 300)
    yp = CS_c + CS_b * xp
    
    chart_params()
    pylab.figure()
    pylab.plot(X[:,2], Y[:, 1], 'o') 
    pylab.plot(xp, yp)
    pylab.xlabel('Number of high reliability users')
    pylab.ylabel('Optimal flow share')
    pylab.ylim(0, 0.8)
    pylab.savefig(home + out + 'sharemodel1.pdf')
    pylab.show()
    
    ols.fit(X[:,2].reshape([190, 1]), Y[:,3])
    CSHL_c = ols.intercept_
    CSHL_b = ols.coef_[0]
    xp = np.linspace(30, 70, 300)
    yp = CSHL_c + CSHL_b * xp
    
    chart_params() 
    pylab.figure()
    pylab.plot(X[:,2], Y[:, 3], 'o') 
    pylab.plot(xp, yp)
    pylab.xlabel('Number of high reliability users')
    pylab.ylabel('Optimal flow share')
    pylab.ylim(0, 0.8)
    pylab.savefig(home + out + 'sharemodel2.pdf')
    pylab.show()

    return [CS_c, CS_b, CSHL_c, CSHL_b]

def central_case():

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter7/chapter7/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'
    
    rows = ['CS', 'SWA', 'OA', 'NS', 'CS-HL', 'SWA-HL']
    results = {row : 0 for row in rows}
    
    for row in rows:
        with open(home + folder + '0' + row + '_0_result.pkl', 'rb') as f:
            results[row] = pickle.load(f)
            f.close()
    
    ###### Summary results #####
    
    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['SW', 'Profit', 'B', 'S', 'W', 'E', 'Z', 'Q_low', 'Q_high', 'Q_env', 'A_low', 'A_high', 'A_env', 'S_low', 'S_high', 'S_env', 'U_low', 'U_high']
    scale = {'SW' : 1000000, 'Profit' : 1000000, 'S' : 1000, 'W' : 1000, 'E' : 1000, 'B' : 1000000, 'Z' : 1000, 'Q_low' : 1000, 'Q_high' : 1000, 'Q_env' : 1000, 'A_low' : 1000, 'A_high' : 1000, 'A_env' : 1000, 'S_low' : 1000, 'S_high' : 1000, 'S_env' : 1000, 'U_low' : 1000000, 'U_high' : 1000000}

    m = len(results['CS'][0]['S']['Annual']['Mean']) - 1

    for x in series:
        data0 = []
        
        record = {}
        for col in cols:
            record[col] = results[row][0][x]['Annual'][col][2] / scale[x]
        data0.append(record)
        
        for row in rows:
            record = {}
            for col in cols:
                record[col] = results[row][0][x]['Annual'][col][m] / scale[x]
            data0.append(record)

        data = pandas.DataFrame(data0)
        data.index = ['Planner'] + rows

        with open(home + table_out + 'central_' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()


    for x in series:
        data0 = []
        
        record = {}
        for col in cols:
            record[col] = results[row][0][x]['Summer'][col][2] / scale[x]
        data0.append(record)
        
        for row in rows:
            record = {}
            for col in cols:
                record[col] = results[row][0][x]['Summer'][col][m] / scale[x]
            data0.append(record)

        data = pandas.DataFrame(data0)
        data.index = ['Planner'] + rows

        with open(home + table_out + 'central_sum_' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()
    
    for x in series:
        data0 = []
        
        record = {}
        for col in cols:
            record[col] = results[row][0][x]['Winter'][col][2] / scale[x]
        data0.append(record)
        
        for row in rows:
            record = {}
            for col in cols:
                record[col] = results[row][0][x]['Winter'][col][m] / scale[x]
            data0.append(record)

        data = pandas.DataFrame(data0)
        data.index = ['Planner'] + rows

        with open(home + table_out + 'central_win_' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

    return results
