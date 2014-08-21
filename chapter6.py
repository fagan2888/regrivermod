# Decentralised water storage model: master file - Neal Hughes

# Chapter 6 model runs

from __future__ import division
import numpy as np
from para import Para
import model
from results import chapter6
from results.chartbuilder import *

para = Para(rebuild=True, charts=False)
para.central_case(N=100, printp=False)
para.set_property_rights(scenario='CS')
para.solve_para()

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter6/'
out = '/Dropbox/Thesis/IMG/chapter6/'

#==========================================
# No trade - central case
#==========================================

scenarios = ['RS-HL-O', 'RS-HL', 'RS-O', 'RS']# 'CS', 'CS-O']
results = {'RS-HL-O': 0, 'RS-HL' : 0, 'RS-O' : 0, 'RS' : 0}#, 'CS' : 0, 'CS-O' : 0} 
Lambda = {'RS-HL-O': 0, 'RS-HL' : 0, 'RS-O' : 0, 'RS' : 0}

for scen in scenarios:
    para.set_property_rights(scenario=scen)
    para.t_cost = 10000000000
    
    mod = model.Model(para)

    results[scen], Lambda[scen] = mod.chapter6()

chapter6.notrade(results, scenarios, Lambda)


#==========================================
# With trade - central case 
#==========================================

#==========================================
# General case 
#==========================================


"""
for i in range(1):
    #try:
    for scen in scenarios:
        para.set_property_rights(scenario=scen)
        para.t_cost = 10000000000

        res = mod.chapter6()

        results[scen].append(res)

        #with open(home + folder + '_result.pkl', 'wb') as f:
        #   pickle.dump(res, f)
        #   f.close()

        #para.randomize(N = 100)

    #except KeyboardInterrupt:
    #    raise
    #except:
    #    pass
"""


"""
#==========================================
# Risk aversion plot
#==========================================
Y = np.zeros([3, 200])
X = np.zeros([3, 200])
risk = [0.25, 1.5, 3]
names = ['0', '1.5', '3.0']

for i in range(3):
    para.central_case(N=100, printp=False, utility=True, risk=risk[i])
    mod = model.Model(para)
    X[i,:], Y[i,:]  = mod.users.SW_f.plot(['x', 1], returndata=True)
    Y[i,:] = Y[i,:] / Y[i , 199]

df = dataframe(200, 3, names, X[0,:], Y.T)

chart = {'OUTFILE': home + out + 'Risk.pdf',
         'XLABEL': 'Total water use, $Q_t$',
         'YLABEL': 'Social welfare (utility)',
         'YMIN' : 0.2,
         'YMAX' : 1.1}

build_chart(chart, df, chart_type='date', ylim=True)

para.central_case(N=100, printp=False)
para.set_property_rights(scenario='CS')

Y = np.zeros([3, 200])
X = np.linspace(0, 1, 200)
risk = [0.001, 1.5, 3]
names = ['0', '1.5', '3.0']

for i in range(3):
    Y[i,:]  = 1 - np.exp(-risk[i]* X) 
    Y[i,:] = Y[i,:] / Y[i , 199]

df = dataframe(200, 3, names, X, Y.T)

chart = {'OUTFILE': home + out + 'Risk0.pdf',
         'XLABEL': 'Scaled user profit, $u_{it}/{\\bar \\pi_h}$',
         'YLABEL': 'Scaled utility',
         'YMIN' : 0,
         'YMAX' : 1.1}

build_chart(chart, df, chart_type='date', ylim=True)

para.central_case(N=100, printp=False)
para.set_property_rights(scenario='CS')

"""




