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
from scipy.stats import truncnorm

home = '/home/nealbob'
folder = '/Dropbox/Model/results/chapter7/'
out = '/Dropbox/Thesis/IMG/chapter7/'
NCIhome = '/short/fr3/ndh401'
NCIfolder = '/chapter7/'

#==========================================
# NCI general case - initialisation
#==========================================

try:
    arg1 = sys.argv[1]
except IndexError:
    print "Provide arguments <runnum> <numofjobs> <scenario>"

para = Para()
para.central_case(N = 100)
para.randomize()
para.set_property_rights(scenario='CS')

run_no = int(arg1)

print '============================================================'
print 'Initialisation for run no: ' + str(run_no)
print '============================================================'

mod = Model(para, ch7=True, turn_off_env=True)
E_lambda = mod.chapter7_initialise()
print '============================================================'
print 'E_lambda: ' + str(E_lambda)
print '============================================================'

#Truncated normal
E_lambda = truncnorm((0.01 - E_lambda) / 0.05, (0.99 - E_lambda) / 0.05, loc=E_lambda, scale=0.05).rvs()

print '============================================================'
print 'E_lambda: ' + str(E_lambda)
print '============================================================'

para.ch7['inflow_share'] = E_lambda
para.ch7['capacity_share'] = E_lambda
para.t_cost = para.t_cost/2.0
para.aproximate_shares_ch7()

print '============================================================'
print 'Lambda high: ' + str(para.Lambda_high)
print 'Lambda high HL: ' + str(para.Lambda_high_HL)
print '============================================================'


with open(NCIhome + NCIfolder +  str(run_no) + '_para.pkl', 'wb') as f:
    pickle.dump(para, f)
    f.close()
