from chartbuilder import *
import numpy as np
import pandas
import pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree

def planner(SW, S, solvetime, myopic):

    home = '/home/nealbob'
    out = '/Dropbox/Thesis/IMG/chapter8/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter8/'

    # Input data

    SW = np.array(SW)
    S = np.array(S)
    solvetime = np.array(solvetime)
    series = ['SDP', 'TC-A', 'TC-ASGD'] 
    n = S.shape[0]
    m = S.shape[1]
    index = [4000, 10000, 20000, 40000, 75000]
    Y = np.zeros([n, m])
    
    # ======================

    chart = {'OUTFILE' : (home + out + 'planner_testb' + img_ext),
      'YLABEL' : 'Social welfare as percentage of SDP',
      'XLABEL' : 'Number of samples'}

    for i in range(n): Y[i, :] = SW[i, :] / SW[i, 0]
    data = dataframe(n, m, series, index, Y)
    build_chart(chart, data, chart_type='date')
    
    # ======================
    
    chart = {'OUTFILE' : (home + out + 'planner_test2b' + img_ext),
      'YLABEL' : 'Social welfare as percentage of SDP',
      'XLABEL' : 'Computation time',
      'XMIN' : np.min(solvetime[:,1::])*0.8,
      'XMAX' : np.max(solvetime[:,1::])*1.2,
      'YMIN' : np.min(Y),
      'YMAX' : 1.005}

    data = [[solvetime[:, j], Y[:,j], series[j]] for j in range(1, m)] 
    build_chart(chart, data, chart_type='plot', legend=True)
    
    tabindex = ['SDP', 'Myopic', 'Q-V TC-A', 'Q-V TC-A', 'Q-V TC-A', 'Q-V TC-A', 'Q-V TC-A', 
            'Q-V TC-ASGD', 'Q-V TC-ASGD', 'Q-V TC-ASGD', 'Q-V TC-ASGD', 'Q-V TC-ASGD', 'RF', 'RF', 'RF', 'RF', 'RF']
    
    data0 = []
    
    #SDP
    record = {}
    record['Sample (T)'] = '-'
    record['Time (secs)'] = solvetime[0, 0]
    record['Welfare ($m)'] = SW[0, 0] / 1000000
    data0.append(record)
        
    #Myopic
    record = {}
    record['Sample (T)'] = '-'
    record['Time (secs)'] = '-'
    record['Welfare ($m)'] = myopic / 1000000
    data0.append(record)
        
    #TC-A
    for i in range(n):
        record = {}
        record['Sample (T)'] = index[i]
        record['Time (secs)'] = solvetime[i,1]
        record['Welfare ($m)'] = SW[i, 1] / 1000000
        data0.append(record)

    #TC-ASGD
    for i in range(n):
        record = {}
        record['Sample (T)'] = index[i]
        record['Time (secs)'] = solvetime[i,2]
        record['Welfare ($m)'] = SW[i, 2] / 1000000
        data0.append(record)
    
    #RF
    for i in range(n):
        record = {}
        record['Sample (T)'] = index[i]
        record['Time (secs)'] = solvetime[i,3]
        record['Welfare ($m)'] = SW[i, 3] / 1000000
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = tabindex 

    tab_text = tab.to_latex(float_format =  '{:,.1f}'.format, columns=['Sample (T)', 'Time (secs)', 'Welfare ($m)'], index=True)  

    with open(home + table_out + "planner_testb.txt", "w") as f:
        f.write(tab_text)
        f.close()
