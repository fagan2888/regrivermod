# Decentralised water storage model: master file - for control of multiple model runs.   Neal Hughes

from __future__ import division
import numpy as np
import model
from para import Para
from results import chapter8
from results.chartbuilder import *

para = Para(rebuild=True, charts=False)
para.central_case(N = 100)
para.solve_para()
para.set_property_rights(scenario = 'CS')

mod = model.Model(para)

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter8/'
out = '/Dropbox/Thesis/IMG/chapter8/'

SW = []
SWb = []
S = []
TIME = []

m = 5
sp = [True for i in range(m)]
t1 = [5000, 10000, 20000, 50000, 80000]
t2 = [5000, 10000, 20000, 50000, 80000]
d  = [0.2] * m

for j in range(m):

    sw, s, time, _ = mod.chapter8(stage2=sp[j], T1=t1[j], T2=t2[j], d=d[j])

    SW.append(sw)
    S.append(s)
    TIME.append(time)

myopicSW = mod.simulate_myopic(500000)

chapter8.planner(SW, S, TIME, myopicSW)


"""
for j in range(3):
    
    t1 = [300000, 500000, 700000]

    _, _, _, swb = mod.chapter8(stage2=sp[j], T1=t1[j], T2=t2[j], d=d[j], decentral_test=True)

    SWb.append(swb)

print SWb



#===========================================
# S x I state space chart
#===========================================

mod.plannerSDP()

mod.sim.simulate(mod.users, mod.storage, mod.utility, 10000, mod.para.CPU_CORES, planner=True, policy=True, polf=mod.sdp.W_f, delta=0, stats=True, planner_explore=False, t_cost_off=True) 

S = mod.sim.series['S'] / 1000
I = mod.sim.series['I'] / 1000

data = [[S, I]]
chart = {'OUTFILE': home + out + 'SbyI.pdf',
 'XLABEL': 'Storage (GL)',
 'YLABEL': 'Inflow (GL)' }
build_chart(chart, data, chart_type='scatter')
"""
