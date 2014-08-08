# Decentralised water storage model: master file - Neal Hughes

# Chapter 5 model runs

from __future__ import division
import numpy as np
from para import Para
import model

para = Para(rebuild=True, charts=False)
para.central_case(N=100, printp=False)
para.set_property_rights(scenario='CS')
para.solve_para()

mod = model.Model(para)

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter5/'

mod.chapter5()

"""
scenarios = ['CS'] #, 'RS-HL']
results = {'CS': []} #, 'RS-HL': 0}

for i in range(1):
    #try:
    para.central_case(N = 100)
    if i > 0:
        para.randomize(N = 100)
    for scen in scenarios:
        para.set_property_rights(scenario=scen)
        result = mod.chapter6(para)
        
        results[scen].append(result)
    
    #with open(home + folder + '_result.pkl', 'wb') as f:
    #   pickle.dump(res, f)
    #   f.close()
    
    #except KeyboardInterrupt:
    #    raise
    #except:
    #    pass
"""




