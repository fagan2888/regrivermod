# Decentralised water storage model: master file - Neal Hughes

# Chapter 5 model runs

from __future__ import division
import numpy as np
from para import Para
from model import Model
from results import chapter3

# Store results here

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter3/'

# Initialise parameters

para = Para(rebuild=True, charts=False)
para.central_case(N=100, printp=False)
para.set_property_rights(scenario='CS')
para.solve_para()

# Create model instance

mod = model.Model(para)

# Model runs

result = {'paras' : [], 'stats' : [], 'series' : []}

for i in range(1000):
    try:
        
        mod.plannerSDP() 
                
        result['stats'].append(mod.sim.stats)
        result['paras'].append(parameters.para_list)
        result['series'].append(mod.sim.series)
        
        if i == 0:
            W_f = mod.sdp.W_f
            V_f = mod.sdp.V_f
            SW_f = mod.users.SW_f

        para.randomize(N = 100)
        mod = model.Model(para)
    
    except KeyboardInterrupt:
        raise
    except:
        raise

    with open(home + folder + 'result.pkl', 'wb') as f:
        pickle.dump(result, f)
        f.close()

chapter3.build(results)
