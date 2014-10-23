# Decentralised water storage model: master file - Neal Hughes

# Model testing

from __future__ import division
import numpy as np
from para import Para
import model
import time
from econlearn.samplegrid import test

if __name__ == '__main__': 

    para = Para(rebuild=True)

    #para.set_property_rights(scenario='RS')
    #mod1 = model.Model(para)
    #mod1.plannerSDP()

    #sr1 = mod1.sim.series
    #del mod1
    #para.t_cost = 100000000000
    para.set_property_rights(scenario='RS')
    para.ch7['inflow_share'] = 0
    para.ch7['capacity_share'] = 0
    mod = model.Model(para, ch7=True, turn_off_env=True)

    mod.plannerSDP(plot=True)
    #mod.users.init_policy(mod.sdp.W_f, mod.sdp.V_f, mod.storage, para.CPU_CORES, para.linT, para.sg_radius2)
    #mod.env.init_policy(mod.sdp.W_f, mod.sdp.V_f, mod.sdp.W_f, mod.sdp.V_f, mod.storage, para.linT, para.CPU_CORES, para.sg_radius2)
    #mod.utility.init_policy(mod.sdp.W_f, mod.storage, para)
    #mod.utility.explore = 1
    #mod.utility.d = 0
    #mod.sim.simulate_ch7(mod.users, mod.storage, mod.utility, mod.market, mod.env, 10000, 1, planner=True)
    mod.plannerQV_ch7(T=100000, stage2=True, d=0.15, simulate=True)

    f1_win_dev = np.minimum(abs((mod.sim.series['F1_tilde'][:,1] - mod.sim.series['F1'][:,1]) / mod.sim.series['F1_tilde'][:,1]), 1)
    f3_win_dev = np.minimum(abs((mod.sim.series['F3_tilde'][:,1] - mod.sim.series['F3'][:,1]) / mod.sim.series['F3_tilde'][:,1]), 1)
    f1_sum_dev = np.minimum(abs((mod.sim.series['F1_tilde'][:,0] - mod.sim.series['F1'][:,0]) / mod.sim.series['F1_tilde'][:,0]), 1)
    f3_sum_dev = np.minimum(abs((mod.sim.series['F3_tilde'][:,0] - mod.sim.series['F3'][:,0]) / mod.sim.series['F3_tilde'][:,0]), 1)

    Bhat = np.ones(200000)
    I = np.ones(200000)
    t = 1
    for i in range(1, 100000):
        for m in range(2):
            if mod.sim.series['F1_tilde'][i,m] == 0:
                F1dot = min(mod.sim.series['F1'][i,m], 1)
            else:
                F1dot = min(abs((mod.sim.series['F1_tilde'][i,m] - mod.sim.series['F1'][i,m]) / mod.sim.series['F1_tilde'][i,m]), 1)
            if mod.sim.series['F3_tilde'][i,m] == 0:
                F3dot = min(mod.sim.series['F3'][i,m], 1)
            else:
                F3dot = min(abs((mod.sim.series['F3_tilde'][i,m] - mod.sim.series['F3'][i,m]) / mod.sim.series['F3_tilde'][i,m]), 1)
            tot_dev = 0.5 * F1dot + 0.5 * F3dot
            Bhat[t] = 0.25 * tot_dev + 0.75 * Bhat[t-1]
            I[t] = mod.sim.series['I'][i,m]
            t += 1

    B = 1 - Bhat**2

    #res, lam, lamk = mod.chapter6()
    #mod.chapter6_extra(65)
    #sr = mod.sim.series

    #mod.sim.test_ch7_sim(10, mod.users, mod.storage, mod.utility, mod.market, mod.env)
    #mod.sim.simulate_ch7(mod.users, mod.storage, mod.utility, mod.market, mod.env, 100000, 4, planner=True)

    #stats, qv, st = mod.plannerQV(t_cost_off=True, stage1=True, stage2=True, T1=para.T1, T2=para.T1, d=para.policy_delta, simulate=True, record_e=False)

    #mod.users.init_policy(mod.sdp.W_f, mod.sdp.V_f, mod.storage, 4, para.linT, para.sg_radius2)

    #mod.sim.test_sim(100, mod.users, mod.storage, mod.utility, mod.users.market_d)

    #mod.chapter5()

#stats, _, _ = mod.plannerSDP(plot=True)

#w_f_low, v_f_low, w_f_high, v_f_high = mod.users.init_policy(mod.sdp.W_f, mod.sdp.V_f, mod.storage, 6, 4, para.s_radius2)

    #mod.chapter6()
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
