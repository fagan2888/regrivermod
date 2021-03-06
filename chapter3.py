# Decentralised water storage model: master file - Neal Hughes

# Chapter 5 model runs

from __future__ import division
import numpy as np
from para import Para
from model import Model
from results import chapter3
import pickle

# Store results here

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter3/'
NCI = '/short/fr3/ndh401/chapter3/'

# Initialise parameters

para = Para(rebuild=True, charts=False)
para.central_case(N=100, printp=False)
para.set_property_rights(scenario='CS')
para.solve_para()
para.SDP_GRID = 40

# Create model instance

mod = Model(para)

# Model runs

result = {'paras' : [], 'stats' : [], 'series' : []}
#temp = [0,0]

for i in range(1000):
    try:
        mod.plannerSDP() 
        #temp[0] =  mod.sim.series       
        
        mod.simulate_myopic(500000)
        
        result['stats'].append(mod.sim.stats)
        result['paras'].append(para.para_list)
        #result['series'].append(mod.sim.series)
        
        #mod2 = Model(para)
        #mod2.simulate_myopic(500000)
        #temp[1] = mod2.sim.series
        #result['series'].append(temp)

        if i == 0:
            W_f = mod.sdp.W_f
            V_f = mod.sdp.V_f
            SW_f = mod.users.SW_f

        para.SDP_GRID = 35
        para.randomize(N = 100)
        mod = Model(para)
    
    except KeyboardInterrupt:
        raise
    except:
        raise
    
    with open(NCI + 'result.pkl', 'wb') as f:
        pickle.dump(result, f)
        f.close()

#chapter3.build(results)
