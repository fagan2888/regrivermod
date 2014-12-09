# Decentralised water storage model: master file - Neal Hughes

# Chapter 6 model runs

from __future__ import division
import numpy as np
from para import Para
from model import Model
from results import chapter7
from results.chartbuilder import *

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter7/'
out = '/Dropbox/Thesis/IMG/chapter7/'

para = Para(rebuild=True)

#==========================================
# Planner with central parameters
#==========================================

para.set_property_rights(scenario='CS')
para.ch7['inflow_share'] = 0.25
para.ch7['capacity_share'] = 0.25

"""
para.sg_radius1_ch7 = 0.02
para.sg_points1_ch7 = 750
mod = Model(para, ch7=True, turn_off_env=True)
mod.plannerQV_ch7(T=150000, stage2=False, d=0.2, simulate=True, envoff=True)
stats_envoff = mod.sim.stats
series_envoff = mod.sim.series
del mod

para.sg_radius1_ch7 = 0.045
para.sg_points1_ch7 = 2000

mod = Model(para, ch7=True, turn_off_env=False)
mod.plannerQV_ch7(T=150000, stage2=False, d=0.2, simulate=True, envoff=False)
series = mod.sim.series
stats = mod.sim.stats

results = [stats_envoff, series_envoff, stats, series]

chapter7.planner(results)
"""

mod = Model(para, ch7=True, turn_off_env=False)
mod.chapter7()
"""
b_value = np.linspace(10, 2000, 25)

E = np.zeros(25)
Env = np.zeros(25)
SW = np.zeros(25)
Profit = np.zeros(25)
F1dev = np.zeros(25)
F3dev = np.zeros(25)
Qlow = np.zeros(25)
Qhigh = np.zeros(25)

for i in range(25):
    
    para.ch7['b_value'] = b_value[i] * para.I_bar

    #mod = Model(para, ch7=True, turn_off_env=True)
    #mod.plannerQV_ch7(T=125000, stage2=True, d=0.2, simulate=True, envoff=True)
    #ITER_envoff = mod.sim.ITEROLD 
    #stats_envoff = mod.sim.stats
    #series_envoff = mod.sim.series
    #del mod
        
    mod = Model(para, ch7=True, turn_off_env=False)
    mod.plannerQV_ch7(T=125000, stage2=True, d=0.2, simulate=True, envoff=False)
    series = mod.sim.series
    stats = mod.sim.stats
    ITER = mod.sim.ITEROLD 
    del mod

    E[i] = stats['E']['Annual']['Mean'][1]
    SW[i] = stats['SW']['Annual']['Mean'][1]
    env = series['SW'].flatten() - series['Profit'].flatten()
    Env[i] = np.mean(env)
    Profit[i] = stats['Profit']['Annual']['Mean'][1]
    F1dev[i]  = np.mean((series['F1'].flatten() - series['F1_tilde'].flatten())**2)
    F3dev[i] = np.mean((series['F3'].flatten() - series['F3_tilde'].flatten())**2)
    Qlow[i] = stats['Q_low']['Annual']['Mean'][1]
    Qhigh[i] = stats['Q_high']['Annual']['Mean'][1]

    print 'E: ' + str(E[i])
    print 'SW: ' + str(SW[i])
    print 'Env: ' + str(Env[i])
    print 'Profit: ' + str(Profit[i])
    print 'F1dev: ' + str(F1dev[i])
    print 'F3dev: ' + str(F3dev[i])
    print 'Qlow: ' + str(Qlow[i])
    print 'Qhigh: ' + str(Qhigh[i])

#chapter6.tables(results, scenarios, Lambda, LambdaK, label='central')

#results = [stats_envoff, series_envoff, stats, series]

#with open(home + folder + 'central_result.pkl', 'wb') as f:
#    pickle.dump(results, f)
#    f.close()
"""

"""
scenarios = ['RS-HL-O', 'RS-HL', 'RS-O', 'RS', 'CS', 'CS-O', 'CS-HL', 'CS-HL-O', 'CS-U']
results = {scen: 0 for scen in scenarios}
Lambda = {scen: 0 for scen in scenarios}
LambdaK = {scen: 0 for scen in scenarios}

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




