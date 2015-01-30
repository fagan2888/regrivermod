# Decentralised water storage model: master file - Neal Hughes

# Chapter 5 model runs

from __future__ import division
import numpy as np
from para import Para
import model
import pickle

para = Para()
para.central_case(N=100, printp=False)
para.set_property_rights(scenario='OA')
para.solve_para()

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter5/'

scenarios = ['CS', 'SWA', 'OA', 'NS', 'CS-SL', 'SWA-SL', 'CS-SWA']
results = {scen: 0 for scen in scenarios}
policies = {scen: 0 for scen in scenarios}

for i in range(1):
    #try:

    para.central_case(N = 100)
    para.t_cost = 100000000000
    para.aproximate_shares(nonoise=True)
    #if i > 0:
    #    para.randomize(N = 100)
    #    para.aproximate_shares()

    for scen in scenarios:
        para.set_property_rights(scenario=scen)
        res = {'paras' : [], 'stats' : [], 'VE': [], 'PE' : []}
        pol = []
        
        mod = model.Model(para)
        
        VE, PE, stats, policy = mod.chapter5()
        res['stats'].append(stats)
        res['paras'].append(para.para_list)
        res['VE'].append(VE)
        res['PE'].append(PE)
        pol.append(policy)
    
        results[scen] = res
        policies[scen] = pol

    with open(home + folder + str(i) + '_result_notrade.pkl', 'wb') as f:
        pickle.dump(results, f)
        f.close()
    
    #except KeyboardInterrupt:
    #    raise
    #except:
    #    pass

"""
#=================================
#   NCI
#=================================

import sys
import multiprocessing
from multiprocessing.queues import Queue
home = '/short/fr3/ndh401'
folder = '/chapter5/'

def solve_model(para, scen, que):
    
    para.set_property_rights(scenario=scen)
    para.aproximate_shares()
    res = {'paras' : [], 'stats' : [], 'VE': [], 'PE' : []}
    pol = []
    mod = Model(para)
    VE, PE, stats, policy = mod.chapter5()
    res['stats'].append(stats)
    res['paras'].append(para.para_list)
    res['VE'].append(VE)
    res['PE'].append(PE)
    pol.append(policy)
    
    del mod
    
    que.put([res, pol])


try:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
except IndexError:
    print "Provide arguments <runnum> <numofjobs>"

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

N = int(arg2)
for i in range(N):
    scenarios = ['CS', 'SWA', 'OA', 'NS']
    results = {scen: 0 for scen in scenarios}
    policies = {scen: 0 for scen in scenarios}
 
    try:
        para.central_case(N = 100)
        if i > 0:
            para.randomize(N = 100)
        para.CPU_CORES = 4
        temp = []
        ques = [RetryQueue() for i in range(4)]
        args = [(para, scenarios[i], ques[i]) for i in range(4)]
        jobs = [multiprocessing.Process(target=solve_model, args=(a)) for a in args]
        for j in jobs: j.start()
        for q in ques: temp.append(q.get())
        for j in jobs: j.join()
        
        for i in range(4):
            results[scenarios[i]] = temp[i][0]
            policies[scenarios[i]] = temp[i][1]

     
        with open(home + folder + str(arg1) + '_' + str(i) +  '_result.pkl', 'wb') as f:
            pickle.dump(results, f)
            f.close()
    
    except KeyboardInterrupt:
        raise
    except:
        pass
"""

