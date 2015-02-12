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
    series = ['SW', 'Profit', 'B', 'S', 'W', 'E', 'P']
    scale = {'SW' : 1000000, 'Profit' : 1000000, 'S' : 1000, 'W' : 1000, 'E' : 1000, 'B' : 1000000, 'P' : 1}

    m = 2

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

    m = 2

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

    m = 2

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
    data = {'Natural' : timeseries['F1_tilde'][:, 0]/1000,
            'Consumptive' : timeseries_envoff['F1'][:, 0]/1000,
            'Optimal' :  timeseries['F1'][:, 0]/1000 } 
    duration_curve(data, OUTFILE=home + out + 'up_sum' + img_ext)

    data = {'Natural' : timeseries['F1_tilde'][:, 1]/1000,
            'Consumptive' : timeseries_envoff['F1'][:, 1]/1000,
            'Optimal' :  timeseries['F1'][:, 1]/1000 } 
    duration_curve(data, OUTFILE=home + out + 'up_win' + img_ext)
    
    data = {'Natural' : timeseries['F3_tilde'][:, 0]/1000,
            'Consumptive' : timeseries_envoff['F3'][:, 0]/1000,
            'Optimal' :  timeseries['F3'][:, 0]/1000 } 
    duration_curve(data, OUTFILE=home + out + 'down_sum' + img_ext)
    
    data = {'Natural' : timeseries['F3_tilde'][:, 1] /1000,
            'Consumptive' : timeseries_envoff['F3'][:, 1] /1000,
            'Optimal' :  timeseries['F3'][:, 1]/1000 } 
    duration_curve(data, OUTFILE=home + out + 'down_win' + img_ext)

    from econlearn import TilecodeRegressor as TR
    
    tr = TR(1, [11], 20)
    
    I = timeseries['I'][1:100000,1] / 1000
    idx = I < 1400
    I = I[idx]
    Q = timeseries['W'][1:100000,1] / 1000
    Q = Q[idx]
    S = timeseries['S'][1:100000,1] / 1000
    S = S[idx]
    S2 = timeseries['S'][1:100000,0] / 1000
    S2 = S2[idx]
    
    Popt = timeseries['P'][:,0]
    Pcons = timeseries_envoff['P'][:,0]

    tr.fit(I[1:30000], Q[1:30000])
    tr.tile.plot(['x'], showdata=True)
    pylab.xlabel('Inflow, $I_t$ (GL)')
    pylab.ylabel('Release, $W_t$ (GL)')
    pylab.xlim(0, 1400)
    #setFigLinesBW_list(fig)
    #pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
    pylab.savefig(home + out + 'env_dem_planner.pdf', bbox_inches='tight')
    pylab.show()
    
    tr.fit(S[1:30000], Q[1:30000])
    tr.tile.plot(['x'], showdata=True)
    pylab.xlabel('Storage, $S_t$ (GL)')
    pylab.ylabel('Release, $W_t$ (GL)')
    #setFigLinesBW_list(fig)
    #pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
    pylab.savefig(home + out + 'env_dem_plannerS.pdf', bbox_inches='tight')
    pylab.show()

    pylab.hexbin(S2, I, C=Q)
    pylab.xlabel('Summer storage, $S_t$ (GL)')
    pylab.ylabel('Winter inflow, $I_t$ (GL)')
    cb = pylab.colorbar()
    cb.set_label('Winter release, $W_t$ (GL)') 
    #pylab.ylim(0, 1000)
    pylab.savefig(home + out + 'env_dem_plannerZ.pdf', bbox_inches='tight')
    pylab.show()

    xopt = pylab.hist(Popt, bins=120)
    xcons = pylab.hist(Pcons, bins=120)
    pylab.show()
    
    fig = pylab.figure() 
    pylab.plot(xopt[1][1::], xopt[0]/500000, label = 'Optimal')
    pylab.plot(xcons[1][1::], xcons[0]/500000, label = 'Consumptive')
    setFigLinesBW_list(fig)
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.) 
    pylab.xlabel('Summer shadow price, $ per ML')
    pylab.ylabel('Frequency')
    pylab.xlim(0, 500)
    pylab.savefig(home + out + 'plan_price.pdf', bbox_inches='tight')
    pylab.show()


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
    pylab.xlabel('River flow, $F_{jt}$ (GL)')
    #pylab.legend()
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
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

def central_case(notrade=False):

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter7/chapter7/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'
    
    rows = ['CS', 'SWA', 'OA', 'NS', 'CS-HL', 'SWA-HL']
    results = {row : 0 for row in rows}
    
    if notrade:
        filename = 'notrade'
    else:
        filename = '0'
    
    for row in rows:
        with open(home + folder + '0' + row + '_' + filename + '_result.pkl', 'rb') as f:
            results[row] = pickle.load(f)[0:2]
            f.close()
    
    ###### Summary results #####
    
    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['SW', 'Profit', 'B', 'S', 'W', 'E', 'Z', 'Q_low', 'Q_high', 'Q_env', 'A_low', 'A_high', 'A_env', 'S_low', 'S_high', 'S_env', 'U_low', 'U_high', 'Budget']
    scale = {'SW' : 1000000, 'Profit' : 1000000, 'S' : 1000, 'W' : 1000, 'E' : 1000, 'B' : 1000000, 'Z' : 1000, 'Q_low' : 1000, 'Q_high' : 1000, 'Q_env' : 1000, 'A_low' : 1000, 'A_high' : 1000, 'A_env' : 1000, 'S_low' : 1000, 'S_high' : 1000, 'S_env' : 1000, 'U_low' : 1000000, 'U_high' : 1000000, 'Budget' : 1000000}

    m = len(results['CS'][0]['S']['Annual']['Mean']) - 1

    if not(notrade):
        filename = 'central_'
    else:
        filename = 'notrade_'

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

        with open(home + table_out + filename + x + '.txt', 'w') as f:
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

        with open(home + table_out + filename + 'sum_' + x + '.txt', 'w') as f:
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

        with open(home + table_out + filename + 'win_' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

    if not(notrade):
        filename = ''
    else:
        filename = 'notrade_'
    
    # central case trade-off chart
    X = np.zeros(7)
    X[0] = results['CS'][0]['B']['Annual']['Mean'][2] / scale['B']
    for i in range(1, 7):
        X[i] = results[rows[i-1]][0]['B']['Annual']['Mean'][m] / scale['B']
    
    Y = np.zeros(7)
    Y[0] = results['CS'][0]['Profit']['Annual']['Mean'][2] / scale['Profit']
    for i in range(1, 7):
        Y[i] = results[rows[i-1]][0]['Profit']['Annual']['Mean'][m] / scale['Profit']

    chart_params()
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y, X, 'o') 
    
    ax.annotate('Planner', xy=(Y[0], X[0]), xytext=(-43, 2) , textcoords='offset points', xycoords=('data'),)
    ax.annotate(rows[0], xy=(Y[1], X[1]), xytext=(-20, 2) , textcoords='offset points', xycoords=('data'),)
    ax.annotate(rows[1], xy=(Y[2], X[2]), xytext=(10, -15) , textcoords='offset points', xycoords=('data'),)
    ax.annotate(rows[2], xy=(Y[3], X[3]), xytext=(0, 5) , textcoords='offset points', xycoords=('data'),)
    ax.annotate(rows[3], xy=(Y[4], X[4]), xytext=(10, -4) , textcoords='offset points', xycoords=('data'),)
    ax.annotate(rows[4], xy=(Y[5], X[5]), xytext=(-40, 5) , textcoords='offset points', xycoords=('data'),)
    ax.annotate(rows[5], xy=(Y[6], X[6]), xytext=(-38, -14) , textcoords='offset points', xycoords=('data'),)
    
    pylab.xlabel('Mean irrigation profit (\$m)')
    pylab.ylabel('Mean environmental benefit (\$m)')
    pylab.ylim(26, 40)
    pylab.savefig(home + out + filename + 'tradeoff.pdf', bbox_inches='tight')
    pylab.show()
    
    # Storage chart
    n = len(results['CS'][0]['S']['Annual']['Mean'])
    data0 = []
    for i in range(2, n-1):
        record = {}
        for sr in ['CS', 'SWA', 'OA', 'NS']:
            record[sr] = results[sr][0]['S']['Annual']['Mean'][i] / 1000
        data0.append(record)
    
    data = pandas.DataFrame(data0)
    chart = {'OUTFILE': home + out + filename + 'Storage' + img_ext,
     'YLABEL': 'Mean storage $S_t$ (GL)',
     'XLABEL': 'Iteration'}
    build_chart(chart, data, chart_type='date')
    
    # Budget
    n = len(results['CS'][0]['S']['Annual']['Mean'])
    data0 = []
    for i in range(2, n-1):
        record = {}
        for sr in ['CS', 'SWA', 'OA', 'NS']:
            record[sr] = results[sr][0]['Budget']['Annual']['Mean'][i] / 1000000
        data0.append(record)
    
    data = pandas.DataFrame(data0)
    chart = {'OUTFILE': home + out + filename + 'Budget' + img_ext,
     'YLABEL': 'Environmental trade $P_t(a_{0t} - q_{0t})$',
     'XLABEL': 'Iteration'}
    build_chart(chart, data, chart_type='date')
            
    # Extraction
    n = len(results['CS'][0]['S']['Annual']['Mean'])
    data0 = []
    for i in range(2, n-1):
        record = {}
        for sr in ['CS', 'SWA', 'OA', 'NS']:
            record[sr] = results[sr][0]['E']['Annual']['Mean'][i] / 1000
        data0.append(record)
    
    data = pandas.DataFrame(data0)
    chart = {'OUTFILE': home + out + filename + 'Extraction' + img_ext,
     'YLABEL': 'Extraction, $E_t$ (GL)',
     'XLABEL': 'Iteration'}
    build_chart(chart, data, chart_type='date')
    
    # Price
    n = len(results['CS'][0]['S']['Annual']['Mean'])
    data0 = []
    for i in range(2, n-1):
        record = {}
        for sr in ['CS', 'SWA', 'OA', 'NS']:
            record[sr] = results[sr][0]['P']['Annual']['Mean'][i]
        data0.append(record)
    
    data = pandas.DataFrame(data0)
    chart = {'OUTFILE': home + out + filename + 'Price' + img_ext,
     'YLABEL': 'Price, $P_t$ (\$/ML)',
     'XLABEL': 'Iteration'}
    build_chart(chart, data, chart_type='date')

    return results

def trade_gain():

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter7/chapter7/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'
    
    scen = ['CS', 'SWA', 'OA', 'NS', 'CS-HL', 'SWA-HL']
    notrade = np.array([205.67, 206.12, 208.03, 196.54, 208.11, 204.62])
    trade = np.array([211.95, 212.55, 211.97, 209.32, 213.79, 212.25])
    gain = trade - notrade
    labels = scen
    chart_params()
    
    width = 0.5
    pylab.bar(np.arange(6), gain, width)
    pylab.xticks(np.arange(6) + width/2, scen)
    pylab.xlabel('Scenario')
    pylab.ylabel('Gain from trade (\$M)')
    pylab.savefig(home + out + 'tradegain.pdf', bbox_inches='tight')
    pylab.show()

def env_demand():

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter7/chapter7/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'
    
    rows = ['CS', 'CS-HL', 'SWA', 'OA']
    
    data = {}

    for row in rows:
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            data[row] = pickle.load(f)[2]['F3'][:, 1]

    duration_curve(data, OUTFILE=home + out + 'down_win_dec' + img_ext)
    
    for row in rows:
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            data[row] = pickle.load(f)[2]['F3'][:, 0]
            f.close()

    duration_curve(data, OUTFILE=home + out + 'down_sum_dec' + img_ext)
    
    from econlearn import TilecodeRegressor as TR
    
    tr = TR(1, [11], 20)
    fig = pylab.figure()

    for row in rows:
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            Q = np.sum(pickle.load(f)[2]['Q_env'], axis = 1) /1000
            f.close()
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            S = np.sum(pickle.load(f)[2]['S'], axis = 1) / 2000
            f.close()
        #with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
        #    I = np.sum(pickle.load(f)[2]['I'], axis = 1)
        #    f.close()

        tr.fit(S, Q)
        tr.tile.plot(['x'], showdata=False, label=row)
    
    pylab.xlabel('Storage volume, $S_t$ (GL)')
    pylab.ylabel('Mean environmental flow, $q_{0t}$ (GL)')
    setFigLinesBW_list(fig)
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
    pylab.savefig(home + out + 'env_demand_S.pdf', bbox_inches='tight')
    pylab.show()
    
    
    tr = TR(1, [11], 20)
    fig = pylab.figure()

    for row in rows:
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            Q = np.sum(pickle.load(f)[2]['Q_env'], axis = 1) /1000
            f.close()
        #with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
        #    S = np.sum(pickle.load(f)[2]['S'], axis = 1) / 2.0
        #    f.close()
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            I = np.sum(pickle.load(f)[2]['I'], axis = 1) / 1000
            f.close()

        tr.fit(I, Q)
        tr.tile.plot(['x'], showdata=False, label=row)
    
    pylab.xlabel('Inflow, $I_t$ (GL)')
    pylab.ylabel('Mean environmental flow, $q_{0t}$ (GL)')
    pylab.xlim(0, 2000)
    setFigLinesBW_list(fig)
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
    pylab.savefig(home + out + 'env_demand_I.pdf', bbox_inches='tight')
    pylab.show()

    tr = TR(1, [11], 20)
    fig = pylab.figure()

    for row in rows:
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            B = np.sum(pickle.load(f)[2]['Budget'], axis = 1) / 1000000
            f.close()
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            S = np.sum(pickle.load(f)[2]['S'], axis = 1) / 2000
            f.close()
        #with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
        #    I = np.sum(pickle.load(f)[2]['I'], axis = 1)
        #    f.close()

        tr.fit(S, B)
        tr.tile.plot(['x'], showdata=False, label=row)
    
    pylab.xlabel('Storage volume, $S_t$ (GL)')
    pylab.ylabel('Mean environmental trade, $P_t(a_{0t} - q_{0t})$ (\$m)')
    setFigLinesBW_list(fig)
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
    pylab.savefig(home + out + 'env_trade_S.pdf', bbox_inches='tight')
    pylab.show()
    
    tr = TR(1, [11], 20)
    fig = pylab.figure()

    for row in rows:
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            B = np.sum(pickle.load(f)[2]['Budget'], axis = 1) / 1000000
            f.close()
        with open(home + folder + '0' + row + '_' + '0' + '_result.pkl', 'rb') as f:
            I = np.sum(pickle.load(f)[2]['I'], axis = 1) /1000
            f.close()

        tr.fit(I, B)
        tr.tile.plot(['x'], showdata=False, label=row)
    
    pylab.xlabel('Inflow, $I_t$ (GL)')
    pylab.ylabel('Mean environmental trade, $P_t(a_{0t} - q_{0t})$ (\$m)')
    pylab.xlim(0, 2000)
    setFigLinesBW_list(fig)
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
    pylab.savefig(home + out + 'env_trade_I.pdf', bbox_inches='tight')
    pylab.show()

def tradeoff():

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter7/chapter7/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'
    
    rows = ['CS', 'SWA', 'OA', 'NS', 'CS-HL', 'SWA-HL']
    rows2 = ['CS', 'SWA', 'OA', 'CS-HL']
    shares = ['10', '20', '26.3', '30', '40', '50']
    results = {share: {row : 0 for row in rows} for share in shares} 

    for share in shares: 
        for row in rows:
            if share == '26.3':
                share_ext = ''
            else:
                share_ext = share
            with open(home + folder + '0' + row + '_0_result' + share_ext + '.pkl', 'rb') as f:
                results[share][row] = pickle.load(f)[0:2]
                f.close()
    
    ###### Summary results #####
    
    series = ['SW', 'Profit', 'B', 'Budget', 'S']
    scale = {'SW' : 1000000, 'Profit' : 1000000, 'S' : 1000, 'W' : 1000, 'E' : 1000, 'B' : 1000000, 'Z' : 1000, 'Q_low' : 1000, 'Q_high' : 1000, 'Q_env' : 1000, 'A_low' : 1000, 'A_high' : 1000, 'A_env' : 1000, 'S_low' : 1000, 'S_high' : 1000, 'S_env' : 1000, 'U_low' : 1000000, 'U_high' : 1000000, 'Budget' : 1000000}

    m = len(results['20']['CS'][0]['S']['Annual']['Mean']) - 1

    for x in series:
        data0 = []
        
        for row in rows:
            record = {}
            for share in shares:
                record[share] = results[share][row][0][x]['Annual']['Mean'][m] / scale[x]
            data0.append(record)

        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + 'tradeoff_' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=shares))
            f.close()

    chart_params()
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    
    # central case trade-off chart
    rows = ['CS',  'CS-HL', 'SWA', 'OA']
    for row in rows:
        X = np.zeros(6)
        Y = np.zeros(6)
        i = 0 
        for share in shares:  
            X[i] = results[share][row][0]['Profit']['Annual']['Mean'][m] / scale['Profit']
            Y[i] = results[share][row][0]['B']['Annual']['Mean'][m] / scale['B']
            i += 1

        ax.plot(X, Y, label=row) 
    X = np.array(results['26.3']['CS'][0]['Profit']['Annual']['Mean'][2] / scale['Profit'])
    Y = np.array(results['26.3']['CS'][0]['B']['Annual']['Mean'][2] / scale['B'])

    setAxLinesBW(ax)
    ax.plot(X, Y, 'o') 
    ax.annotate('Planner', xy=(X, Y), xytext=(-5, 10) , textcoords='offset points', xycoords=('data'),)
    
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
    
    pylab.xlabel('Mean irrigation profit (\$m)')
    pylab.ylabel('Mean environmental benefit (\$m)')
    pylab.xlim(140, 190)
    pylab.ylim(20, 50)
    pylab.savefig(home + out + 'tradeoff_multi.pdf', bbox_inches='tight')
    
    pylab.show()
    
    chart_params()
    fig, ax = pylab.subplots()   
    
    # central case trade-off chart
    rows = ['CS',  'CS-HL', 'SWA', 'SWA-HL', 'OA', 'NS']
    
    pos = np.arange(6)
    width = 0.1
    for share in shares:  
        i = 0
        X = np.zeros(6)
        for row in rows:
            X[i] = results[share][row][0]['SW']['Annual']['Mean'][m] / scale['SW']
            i += 1
        ax.bar(pos, X, width=width)
        pos = pos + width
        
    
    #two = ax.bar(data_set[1][0], data_set[1][1], chart['WIDTH'], color='w')
    #ax.set_ylabel(chart['YLABEL'])
    #ax.set_xticklabels(chart['LABELS'])
    #ax.legend((one[0], two[0]), chart['LEGEND'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

    pylab.ylim(190, 215)
    pylab.savefig(home + out + 'tradeoff_multi2.pdf', bbox_inches='tight')
    
    pylab.show()

    return results

def sens(sample=20):

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter7/chapter7/'
    out = '/Dropbox/Thesis/IMG/chapter7/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter7/'
    
    rows = ['CS', 'SWA', 'OA', 'CS-HL']
    results = {run_no: {row : 0 for row in rows} for run_no in range(1,sample)} 
    samprange = []
    
    for run_no in range(1, sample): 
        try:
            for row in rows:
                with open(home + folder + str(run_no) + '_' + row + '_result.pkl', 'rb') as f:
                    results[run_no][row] = pickle.load(f)
                    f.close()
            samprange.append(run_no)
        except:
            print 'Run no: ' + str(run_no) + ' failed.'

    n = len(samprange)
    print str(n) + ' good runs of ' + str(sample - 1) + ' total'
     
    ###### Summary tables #####
    
    series = ['SW', 'Profit', 'B', 'Budget', 'S']
    title = {'SW' : 'Social welfare relative to CS', 'Profit' : 'Profit relative to CS', 'B' : 'Environmental benefits relative to CS', 'S' : 'Storage relative to CS', 'Budget' : 'Environmental trade relative to CS'}
    scale = {'SW' : 1000000, 'Profit' : 1000000, 'S' : 1000, 'W' : 1000, 'E' : 1000, 'B' : 1000000, 'Z' : 1000, 'Q_low' : 1000, 'Q_high' : 1000, 'Q_env' : 1000, 'A_low' : 1000, 'A_high' : 1000, 'A_env' : 1000, 'S_low' : 1000, 'S_high' : 1000, 'S_env' : 1000, 'U_low' : 1000000, 'U_high' : 1000000, 'Budget' : 1000000}

    m = len(results[1]['CS'][0]['S']['Annual']['Mean']) - 1
    
    X = {}
    XI = {}

    for x in series:
        data0 = []
        data1 = []
        data2 = []
        
        for row in rows:
            temp = np.zeros(n)
            record = {}
            record1 = {}
            i = 0
            for run_no in samprange:
                temp[i] = results[run_no][row][0][x]['Annual']['Mean'][m] / scale[x]
                i += 1
                record[run_no] = results[run_no][row][0][x]['Annual']['Mean'][m] / scale[x]
            record1['Mean'] = np.mean(temp)
            record1['Min'] = np.min(temp)
            record1['Q1'] = np.percentile(temp, 25)
            record1['Q3'] = np.percentile(temp, 75)
            record1['Max'] = np.max(temp)
            
            X[row] = temp

            data0.append(record)
            data1.append(record1)

        data = pandas.DataFrame(data0)
        data.index = rows
        data1 = pandas.DataFrame(data1)
        data1.index = rows #['Mean', 'Min', 'Q1', 'Q3', 'Max']

        for row in rows:
            record2 = {}
            temp1 = np.zeros(n)
            for i in range(n):
                temp1[i] = X[row][i] / X['CS'][i]
            
            XI[row] = temp1
            
            record2['Mean'] = np.mean(temp)
            record2['Min'] = np.min(temp)
            record2['Q1'] = np.percentile(temp, 25)
            record2['Q3'] = np.percentile(temp, 75)
            record2['Max'] = np.max(temp)
            data2.append(record2)
        
        data2 = pandas.DataFrame(data2)
        data2.index = rows #['Mean', 'Min', 'Q1', 'Q3', 'Max']

        with open(home + table_out + 'sens_full' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=samprange))
            f.close()
        
        with open(home + table_out + 'sens_sum' + x + '.txt', 'w') as f:
            f.write(data1.to_latex(float_format='{:,.2f}'.format))
            f.close()
        
        with open(home + table_out + 'sens_table' + x + '.txt', 'w') as f:
            f.write(data2.to_latex(float_format='{:,.2f}'.format))
            f.close()
        
        minx = np.percentile([min(XI[i]) for i in XI], 1)
        maxx = np.percentile([max(XI[i]) for i in XI],99)
        if x == 'SW':
            minx = 0.8
        chart_ch7(XI, 0.985 * minx, 1.015 * maxx, title[x], out, str(x) + '_sens')

    ##################################################################################### Regression

    Y = np.zeros([n, 4])
    
    j = 0
    for row in rows:
        i = 0
        for run_no in samprange:
            Y[i, j] = results[run_no][row][0]['SW']['Annual']['Mean'][m] /  results[run_no]['CS'][0]['SW']['Annual']['Mean'][m]
            i += 1
        j += 1
    
    paras = []
    for run_no in range(1, sample): 
        with open(home + folder + str(run_no) + '_para.pkl', 'rb') as f:
            paras.append(pickle.load(f))
            f.close()
    
    pname1 = ['delta0', 'I_K', 'SD_I', 't_cost', 'N_high', 'rho_I', 'alpha', 'rho_eps', 'sig_eta', 'LL']
    numpara1 = len(pname1)    
    pname2 = ['omega_mu', 'omega_sig', 'omegadelta', 'delta_a', 'delta_Ea', 'delta_Eb', 'delta_R', 'b_1', 'b_value', 'e_sig']
    numpara2 = len(pname2)    
    para_labels = pname1 + pname2 + ['lambda', 'lambdaHL', 'lambdae']
    numpara = numpara1 + numpara2 + 3

    X = np.zeros([n, numpara])
    
    para_names = ['$\delta0$', '$E[I]/K$',  '$c_v$', '$\tau$', '$n_{high}$', '$\rho_I$', '$\alpha$', '$\rho_e$', '$\sigma_{\eta}$', '${\aA_{low} \over E[I]/K}$', '$\mu_\omega$', '$\sigma_\omega$', '$\omega_\delta$', '$\delta_a$', '$\delta_{Ea}$', '$\delta_{Eb}', '$\delta_R$', '$b_1$', '$b_{\$} \over \bar I$', '$\sigma_{e0}$', '$\Lambda_{high} - \hat \Lambda_{high}$', '$\Lambda_{high}^{CS-HL} - \hat \Lambda_{high}^{CS-HL}$', '$\lambda_0 - \hat \lambda_0$' ]
    
    for j in range(numpara1):
        for i in range(n):
            if pname1[j] == 'LL':
                X[i, j] = paras[samprange[i]-1].para_list[pname1[j]] / paras[samprange[i]-1].para_list['I_K']
            else:
                X[i, j] = paras[samprange[i]-1].para_list[pname1[j]]
    
    for j in range(numpara1, numpara2+numpara1):
        for i in range(n):
            if pname2[j - numpara1] == 'b_value':
                X[i, j] = paras[samprange[i]-1].ch7[pname2[j - numpara1]] / (paras[samprange[i]-1].para_list['I_K']*1000000)
            else:
                X[i, j] = paras[samprange[i]-1].ch7[pname2[j - numpara1]]
        
    CS_c = -0.153007555
    CS_b = 0.00930613
    CSHL_c = -0.0891846
    CSHL_b = 0.0047009
    
    for i in range(n): 
        if i > 20:
            y = paras[samprange[i]-1].y
        else:
            y = CS_c + CS_b * paras[samprange[i]-1].para_list['N_high']
        X[i, numpara2 + numpara1] = paras[samprange[i]-1].Lambda_high - y 
    
    for i in range(n): 
        if i > 20:
            yhl = paras[samprange[i]-1].yhl
        else:
            yhl = CSHL_c + CSHL_b * paras[samprange[i]-1].para_list['N_high']
        X[i, numpara2 + numpara1 + 1] = paras[samprange[i]-1].Lambda_high_HL - yhl
    
    yelist = [0.4443, 0.1585, 0.1989, 0.2708, 0.3926, 0.0697, 0.1290, 0.1661, 0.2687, 0.0868, 0.1239, 0.3598, 0.3543, 0.2883, 0.2367, 0.2139, 0.2485, 0.2641, 0.5730, 0.1745] 

    for i in range(n): 
        if i >= 20:
            ye = paras[samprange[i]-1].E_lambda_hat
        else:
            ye = yelist[samprange[i]-1]  
        X[i, numpara2 + numpara1 + 2] = paras[samprange[i]-1].ch7['inflow_share'] - ye
    
    tree = Tree(n_estimators=500, n_jobs=4)
    tree.fit(X, Y)
    rank = tree.feature_importances_ * 100
    
    data0 = []
    inn = 0
    for p in para_names:
        record = {}
        record['Importance'] = rank[inn]
        data0.append(record)
        inn = inn + 1

    tab = pandas.DataFrame(data0)
    tab.index = para_names
    tab = tab.sort(columns=['Importance'], ascending=False)
    tab_text = tab.to_latex(float_format='{:,.2f}'.format, escape=False)
    print tab_text 
    with open(home + table_out + 'importance.txt', 'w') as f:
        f.write(tab_text)
        f.close()
     
    for i in range(numpara):
        Xtemp = np.zeros([200, numpara])
        for j in range(numpara):
            Xtemp[:, j] = np.ones(200) * np.mean(X[:, j])

        Xtemp[:, i] = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 200)
        Ytemp = tree.predict(Xtemp)
        
        data = [[Xtemp[:, i], Ytemp]]
        data0 = []
        for k in range(200):
            record = {}
            record['SWA'] = Ytemp[k, 1]
            record['OA'] = Ytemp[k, 2]
            record['CS-HL'] = Ytemp[k, 3]
            data0.append(record)

        data = pandas.DataFrame(data0)
        data.index = Xtemp[:, i]
        chart_data = {'OUTFILE': home + out + 'SW_' + para_labels[i] + img_ext,
         'XLABEL': '',
         'YLABEL': '',
         'YMIN': 0.97,
         'YMAX': 1.01}
        print para_labels[i]
        
        build_chart(chart_data, data, chart_type='date', ylim=True)
     
    ##################################################################################### Classifier
    
    srnum = {'CS' : 0, 'SWA' : 1, 'OA' : 2, 'CS-HL' : 3}
    Y = np.zeros(n)

    for i in range(n):
        SW = 0
        SWmax = -1
        for row in rows:
            SW = results[samprange[i]][row][0]['SW']['Annual']['Mean'][m]      
            if SW > SWmax:
                SWmax = SW
                Y[i] = srnum[row]
    
    for row in rows:
        idx = np.where(Y == srnum[row])
        print row + ': ' + str(np.count_nonzero(Y[idx]))

     
    treec = Tree_classifier(n_estimators=500, n_jobs=4) #min_samples_split=3, min_samples_leaf=2)
    treec.fit(X, Y)
    rank = treec.feature_importances_ * 100
    print rank


    """ 
    data0 = []
    inn = 0
    for p in para_names:
        record = {}
        record['Importance'] = rank[inn]
        record['CS'] = np.mean(Xpara[np.where(Y == 0), inn])
        record['RS'] = np.mean(Xpara[np.where(Y == 1), inn])
        record['CS-HL'] = np.mean(Xpara[np.where(Y == 2), inn])
        record['RS-HL'] = np.mean(Xpara[np.where(Y == 3), inn])
        data0.append(record)
        inn = inn + 1

    tab = pandas.DataFrame(data0)
    tab.index = para_names
    tab = tab.sort(columns=['Importance'], ascending=False)
    tab_text = tab.to_latex(float_format='{:,.2f}'.format, escape=False)
    
    with open(home + table_out + 'classifier_table.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.2f}'.format, escape=False, columns=['Importance', 'CS', 'RS', 'CS-HL', 'RS-HL']))
        f.close()

    pylab.ioff()
    fig_width_pt = 350
    inches_per_pt = 1.0 / 72.27
    golden_mean = 1.2360679774997898 / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
     'axes.labelsize': 10,
     'text.fontsize': 10,
     'legend.fontsize': 10,
     'xtick.labelsize': 8,
     'ytick.labelsize': 8,
     'text.usetex': True,
     'figure.figsize': fig_size}
    pylab.rcParams.update(params)
    plot_colors = 'rybg'
    cmap = pylab.cm.RdYlBu
    
    (xx, yy,) = np.meshgrid(np.arange(min(Xpara[:, 1]), max(Xpara[:, 1]), 0.02), np.arange(min(Xpara[:, 6]), max(Xpara[:, 6]), 1))

    nnn = xx.ravel().shape[0]
    
    Xlist = [np.mean(Xpara[:,i])*np.ones(nnn) for i in range(pnum)]
    Xlist[2] = np.zeros(nnn)
    Xlist[1] = xx.ravel()
    Xlist[6] = yy.ravel()
    X = np.array(Xlist).T

    Z = treec.predict(X).reshape(xx.shape)
    fig = pylab.contourf(xx, yy, Z, [0, 0.9999, 1.9999, 2.9999, 3.9999], colors=('red', 'yellow', 'blue', 'green'), alpha=0.5, antialiased=False, extend='both')
    for (i, c,) in zip(xrange(4), plot_colors):
        idx0 = np.where(Y == i)
        pylab.scatter(Xpara[idx0, 1], Xpara[idx0, 6], c=c, cmap=cmap, label=srlist[i], s = 12, lw=0.5 )
        pylab.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, ncol=4, mode='expand', borderaxespad=0.0)

    pylab.xlabel('Mean inflow over capacity')
    pylab.ylabel('Number of high reliability users')
    OUT = home + out + 'class_fig.pdf'
    pylab.savefig(OUT, bbox_inches='tight')
    pylab.show()
    
    Xlist = [np.mean(Xpara[:,i])*np.ones(nnn) for i in range(pnum)]
    Xlist[2] = np.ones(nnn) * 1
    Xlist[1] = xx.ravel()
    Xlist[6] = yy.ravel()
    X = np.array(Xlist).T

    Z = treec.predict(X).reshape(xx.shape)
    fig = pylab.contourf(xx, yy, Z, [0, 0.9999, 1.9999, 2.9999, 3.9999], colors=('red', 'yellow', 'blue', 'green'), alpha=0.5, antialiased=False, extend='both')
    for (i, c,) in zip(xrange(4), plot_colors):
        idx0 = np.where(Y == i)
        pylab.scatter(Xpara[idx0, 1], Xpara[idx0, 6], c=c, cmap=cmap, label=srlist[i], s=12, lw=0.5)
        pylab.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, ncol=4, mode='expand', borderaxespad=0.0)

    pylab.xlabel('Mean inflow over capacity')
    pylab.ylabel('Number of high reliability users')
    OUT = home + out + 'class_fig2.pdf'
    pylab.savefig(OUT, bbox_inches='tight')
    pylab.show()
    """

    ##################################################################################### Trade-off
