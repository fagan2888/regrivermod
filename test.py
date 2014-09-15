# Decentralised water storage model: master file - Neal Hughes

# Model testing

from __future__ import division
import numpy as np
from para import Para
import model
import time
from econlearn.samplegrid import test

if __name__ == '__main__': 

    para = Para(rebuild=True, charts=False)
    para.central_case(N=100, printp=False)
    para.set_property_rights(scenario='CS')
    para.solve_para()

    mod = model.Model(para)
#stats, _, _ = mod.plannerSDP(plot=True)

#w_f_low, v_f_low, w_f_high, v_f_high = mod.users.init_policy(mod.sdp.W_f, mod.sdp.V_f, mod.storage, 6, 4, para.s_radius2)

    mod.chapter5()
#multiQV(5, 0.2, 40, init=True)

#SW = 0
#for i in range(5):
#    stats, _, _ = mod.plannerQV(t_cost_off=True, T1=10000, T2=10000, stage1=True, stage2=True, d=0.16, type='ASGD') #0.155
#    SW += stats['SW']['Mean'][1] / 5
#print SW

    t1 = 500000
    t2 = 0
    d = 0
#_, _, _, swb = mod.chapter8(stage2=False, T1=t1, T2=t2, d=d, decentral_test=True)

    """
    mod.storage.set_test_policy(mod.sdp.W_f)
    mod.storage.precompute_I_shocks(10000)
    mod.storage.precompute_I_split(10000)
    mod.storage.update_ch7_test(mod.storage.S*0.5, 0.0, 0, 0)
    mod.storage.update_ch7_test(mod.storage.S*0.5, 0.0, 1, 0)
    mod.storage.update_ch7_test(mod.storage.S*0.5, 0.0, 0, 1)
    mod.storage.update_ch7_test(mod.storage.S*0.5, 0.0, 1, 1)
    mod.storage.update_ch7_test(mod.storage.S*0.5, 0.0, 0, 2)
    mod.storage.update_ch7_test(mod.storage.S*0.5, 0.0, 1, 2)
    natural = mod.storage.natural_flows_ch7_test(10000, -1)
    reg = mod.storage.natural_flows_ch7_test(10000, -2)
    """

#mod.users.init = 1
#mod.users.testing = 0
#mod.users.W_f = mod.sdp.W_f
#mod.users.set_explorers(5, 0)

#mod.sim.simulate(mod.users, mod.storage, mod.utility, 100000, para.CPU_CORES, stats=True)

#stats, qv = mod.multiQV(5, 0, ITER=para.ITER1, init=True, policy = mod.sdp.W_f)

#mod.sim.test_sim(100, mod.users, mod.storage, mod.utility, mod.users.market_d)

#stats, qv1, st = mod.plannerQV(t_cost_off=True, T1=50000, T2=50000, stage1=True, stage2=True, d=0.2, seed=time.time(), type='A')
