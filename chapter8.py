# Decentralised water storage model: master file - for control of multiple model runs.   Neal Hughes

from __future__ import division
import numpy as np
import model
from para import Para
import results 

para = Para(rebuild=True, charts=False)
para.central_case(N = 100)
para.solve_para()
para.set_property_rights(scenario = 'CS')

mod = model.Model(para)

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter8/'

SW = []
S = []
TIME = []

sp = [False for i in range(5)]
t1 = [4000, 10000, 20000, 40000, 75000]
t2 = [4000, 10000, 20000, 40000, 75000]
d = [0.2, 0.2, 0.2, 0.2, 0.2]

for j in range(5):

    sw, s, time = mod.chapter8(stage2=sp[j], T1=t1[j], T2=t2[j], d=d[j])

    SW.append(sw)
    S.append(s)
    TIME.append(time)

myopicSW = mod.simulate_myopic(600000)

results.chapter8.planner(SW, S, TIME, myopicSW)

