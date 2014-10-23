# Decentralised water storage model: master file - Neal Hughes

# Chapter 6 model runs

from __future__ import division
import numpy as np
from para import Para
from model import Model
from results import chapter6
from results.chartbuilder import *

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter6/'
out = '/Dropbox/Thesis/IMG/chapter6/'

para = Para()

scenarios = ['RS-HL-O', 'RS-HL', 'RS-O', 'RS', 'CS', 'CS-O', 'CS-HL', 'CS-HL-O', 'CS-U']
results = {scen: 0 for scen in scenarios}
Lambda = {scen: 0 for scen in scenarios}
LambdaK = {scen: 0 for scen in scenarios}

"""
#==========================================
# Central case (with trade)
#==========================================

for scen in scenarios:
    para.set_property_rights(scenario=scen)

    mod = Model(para)

    results[scen], Lambda[scen], LambdaK[scen] = mod.chapter6()
        
    del mod

chapter6.tables(results, scenarios, Lambda, LambdaK, label='central')

with open(home + folder + 'central_result.pkl', 'wb') as f:
    pickle.dump(results, f)
    f.close()


#==========================================
# No trade case
#==========================================

para.t_cost = 10000000000

for scen in scenarios:
    para.set_property_rights(scenario=scen)
    mod = Model(para)

    results[scen], Lambda[scen], LambdaK[scen] = mod.chapter6()

    del mod

chapter6.tables(results, scenarios, Lambda, LambdaK, label='notrade')

with open(home + folder + 'notrade_result.pkl', 'wb') as f:
    pickle.dump(results, f)
    f.close()
"""
#==========================================
# Risk aversion
#==========================================

para.central_case(utility=True, risk=3)

for scen in scenarios:
    para.set_property_rights(scenario=scen)
    mod = Model(para)

    results[scen], Lambda[scen], LambdaK[scen] = mod.chapter6()

    del mod

chapter6.tables(results, scenarios, Lambda, LambdaK, label='risk1', risk=True)

with open(home + folder + 'risk1_result.pkl', 'wb') as f:
    pickle.dump(results, f)
    f.close()
"""
#==========================================
# General case 
#==========================================


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




