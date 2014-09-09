from chartbuilder import *
import numpy as np
import pandas
import pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree

def planner(SW, S, solvetime, myopic):

    home = '/home/nealbob'
    out = '/Dropbox/Thesis/IMG/chapter8/fine/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter8/fine/'

    # Input data

    SW = np.array(SW)
    S = np.array(S)
    solvetime = np.array(solvetime)
    series = ['SDP', 'TC-A', 'TC-ASGD'] 
    n = S.shape[0]
    m = S.shape[1]
    index = [5000, 10000, 20000, 50000, 80000]
    Y = np.zeros([n, m])
    
    # ======================

    chart = {'OUTFILE' : (home + out + 'planner_testb' + img_ext),
      'YLABEL' : 'Social welfare as percentage of SDP',
      'XLABEL' : 'Number of samples'}

    for i in range(n): Y[i, :] = SW[i, :] / SW[i, 0]
    data = dataframe(n, m, series, index, Y)
    build_chart(chart, data, chart_type='date')
    
    # ======================
    
    #chart = {'OUTFILE' : (home + out + 'planner_test2b' + img_ext),
    #  'YLABEL' : 'Social welfare as percentage of SDP',
    #  'XLABEL' : 'Computation time',
    #  'XMIN' : np.min(solvetime[:,1::])*0.8,
    #  'XMAX' : np.max(solvetime[:,1::])*1.2,
    #  'YMIN' : np.min(Y),
    #  'YMAX' : 1.005}

    #data = [[solvetime[:, j], Y[:,j], series[j]] for j in range(1, m)] 
    #build_chart(chart, data, chart_type='plot', legend=True)
    
    # =====================

    # Welfare
    data0 = []
    
    for i in range(n):
        record = {}
        record['Myopic'] = myopic[0] / 1000000
        for j in range(3):
            record[series[j]] = SW[i, j] / 1000000
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = index 
    tab = tab.transpose()
    tab_text = tab.to_latex(float_format =  '{:,.1f}'.format, index=True)  

    with open(home + table_out + "planner_test_SW.txt", "w") as f:
        f.write(tab_text)
        f.close()

    # =====================

    # Storage
    data0 = []
    for i in range(n):
        record = {}
        record['Myopic'] = myopic[1] / 1000
        for j in range(3):
            record[series[j]] = S[i, j] / 1000
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = index 
    tab = tab.transpose()
    tab_text = tab.to_latex(float_format =  '{:,.1f}'.format, index=True)  

    with open(home + table_out + "planner_test_S.txt", "w") as f:
        f.write(tab_text)
        f.close()

    # =====================

    # Solvetime
    data0 = []
    for i in range(n):
        record = {}
        for j in range(3):
            record[series[j]] = solvetime[i, j]
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = index 
    tab = tab.transpose()
    tab_text = tab.to_latex(float_format =  '{:,.1f}'.format, index=True)  

    with open(home + table_out + "planner_test_time.txt", "w") as f:
        f.write(tab_text)
        f.close()


def decentralised(SWb):

    home = '/home/nealbob'
    out = '/Dropbox/Thesis/IMG/chapter8/test/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter8/test/'

    # Input data

    SWb = np.array(SWb)
    series = ['TC-A', 'TC-ASGD'] 
    n = SWb.shape[0]
    m = SWb.shape[1]
    index = [300000/5, 500000/5, 700000/5]
    Y = np.zeros([n, m])
    
    # ======================

    chart = {'OUTFILE' : (home + out + 'decentral_test_low' + img_ext),
      'YLABEL' : 'User payoff (\$m)',
      'XLABEL' : 'Simulation length (T)'}

    for i in range(n): Y[i, :] = SWb[i, :] / 1000000
    data = dataframe(n, m, series, index, Y)
    build_chart(chart, data, chart_type='date')
