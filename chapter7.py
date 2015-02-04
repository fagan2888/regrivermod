# Decentralised water storage model: master file - Neal Hughes

# Chapter 7 model runs

from __future__ import division
import numpy as np
from para import Para
from model import Model
from results import chapter7
from results.chartbuilder import *
import sys
import multiprocessing
from multiprocessing.queues import Queue

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter7/'
out = '/Dropbox/Thesis/IMG/chapter7/'

NCIhome = '/short/fr3/ndh401'
NCIfolder = '/chapter7/'

def retry_on_eintr(function, *args, **kw):
    while True:
        try:
            return function(*args, **kw)
        except IOError, e:            
            if e.errno == errno.EINTR:
                continue
            else:
                raise    

class RetryQueue(Queue):

    def get(self, block=True, timeout=None):
        return retry_on_eintr(Queue.get, self, block, timeout)

def solve_model(para, scenarios, E_lambda, nonoise, que):
    results = {scen: 0 for scen in scenarios}
    
    for scen in scenarios:
        para.set_property_rights(scenario=scen)
        para.aproximate_shares_ch7(nonoise=nonoise)
        mod = Model(para, ch7=True, turn_off_env=False)
        results[scen] = mod.chapter7(E_lambda)
        del mod
        
    que.put([results])

#==========================================
# NCI central case - decentralised
#==========================================

try:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
except IndexError:
    print "Provide arguments <runnum> <numofjobs> <scenario>"

para = Para()
share_no = int(arg1)
scen = arg2 #, 'CS-HL', 'SWA','SWA-HL', 'OA', 'NS']

P_adj_scen = {'CS' : 58.3, 'SWA' : 57.1, 'OA' : 0, 'NS' : 38.4, 'CS-HL' : 94.1, 'SWA-HL' : 61} #0
E_lambda_share = [0.1, 0.2, 0.263, 0.3, 0.4, 0.5] 
E_lambda_name = ['10', '20', '', '30', '40', '50'] 

print '============================================================'
print 'Scenario: ' + scen
print 'E_lambda: ' + str(E_lambda_share[share_no])
print 'E_lambda_name: ' + str(E_lambda_name[share_no])
print '============================================================'

P_adj = P_adj_scen[scen]

#for i in range(N):
    
para.central_case(N = 100)
nonoise = True
#if i > 0:
#    para.randomize(N = 100)
#    nonoise = False

#para.set_property_rights(scenario='CS')
#mod = Model(para, ch7=True, turn_off_env=True)
#E_lambda = mod.chapter7_initialise()
#del mod
E_lambda = E_lambda_share[share_no]

para.ch7['inflow_share'] = E_lambda
para.ch7['capacity_share'] = E_lambda
#para.t_cost = 10000000000000000000 #
para.t_cost = para.t_cost/2.0

para.set_property_rights(scenario=scen)
para.aproximate_shares_ch7(nonoise=nonoise)
mod = Model(para, ch7=True, turn_off_env=False)
results = mod.chapter7(P_adj, psearch=True)
del mod

#notrade
#0
#env_flow
with open(NCIhome + NCIfolder + '0' + str(arg2) + '_' + 'env_flow' +  '_result' + E_lambda_name[share_no] + '.pkl', 'wb') as f:
    pickle.dump(results, f)
    f.close()
    

#==========================================
# Planner with central parameters
#==========================================

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

#==========================================
# Environmental trade-off curves
#==========================================

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

