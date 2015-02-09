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

"""
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

"""    
"""
#==========================================
# Planner with central parameters
#==========================================

para = Para()
para.central_case(N = 100)
para.set_property_rights('CS')
para.aproximate_shares_ch7(nonoise=True)

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
# General case
#==========================================

try:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
except IndexError:
    print "Provide arguments <runnum> <numofjobs> <scenario>"

run_no = int(arg1)
scen = arg2 

P_adj_scen = {'CS' : 58.3, 'SWA' : 57.1, 'OA' : 0, 'NS' : 38.4, 'CS-HL' : 94.1, 'SWA-HL' : 61} #0

print '============================================================'
print 'Scenario: ' + scen
print '============================================================'

P_adj = P_adj_scen[scen]

with open(NCIhome + NCIfolder +  str(run_no) + '_para.pkl', 'rb') as f:
    para = pickle.load(f)
    f.close()

print '============================================================'
print 'E_lambda: ' + str(para.ch7['inflow_share'])
print 'Run no: ' + str(run_no)
print '============================================================'

para.set_property_rights(scenario=scen)

if scen == 'CS-HL':
    para.Lambda_high = self.Lambda_high_HL
    para.para_list['Lambda_high'] = para.Lambda_high

mod = Model(para, ch7=True, turn_off_env=False)
results = mod.chapter7(P_adj, psearch=True)
del mod

with open(NCIhome + NCIfolder + str(run_no) +'_' + str(arg2) + '_result.pkl', 'wb') as f:
    pickle.dump(results, f)
    f.close()

