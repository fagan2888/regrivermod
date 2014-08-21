from __future__ import division
import numpy as np
from scipy.stats import lognorm, norm, uniform, gamma
from scipy.optimize import brentq, newton
from math import log, exp
import pandas        
import pylab
import pickle
import multiprocessing as mp

class Para:

    "Class for generating model parameters"

    def __init__(self, charts = False, rebuild=False):

        self.opt_lam = 0

        if rebuild:
            home = '/home/nealbob'
            #home = '/users/MAC'
            folder = '/Dropbox/Thesis/STATS/chapter3/'
            folder2 = '/Dropbox/Thesis/STATS/chapter2/'
            out = '/Dropbox/Thesis/IMG/chapter3/'
            img_ext = '.pdf'
            
            MDB_dams = pandas.read_csv(home + folder + 'MDB_dams.csv')
            AUS_RIVERS = pandas.read_csv(home + folder + 'AUS_RIVERS.csv')
            
            #MDB_table = pandas.read_csv(home + folder + 'MDB_table.csv')
            #open(home + folder + "MDB_table.tex", "w").write(MDB_table.to_latex(index=False, float_format=lambda x: '%10.1f' % x ))

            #################       WATER SUPPLY        ################

            #---------------#       K - capacity
            
            K = MDB_dams['K']
            sample = (K > 50000)
            K = K[sample] / 1000

            bins = np.linspace(min(K), max(K), 18)
            data = [K]
            
            chart = {'OUTFILE' : (home + out + 'K' + img_ext),
              'XLABEL' : 'Storage capacity (GL)',
              'XMIN' : min(K),
              'XMAX' : max(K),
              'BINS' : bins}

            if charts:
                self.build_chart(chart, data, chart_type='hist')
            
            #---------------#       I_over_K - mean inflow over storage capacity    
            
            K = MDB_dams['K']
            E_I = MDB_dams['mean_I']
            sample = (K > 50000)
            I_over_K = E_I[sample] / K[sample]
            
            I_K_param = np.zeros(2)
            I_K_param[0] = np.percentile(I_over_K, 10)
            I_K_param[1] = np.percentile(I_over_K, 83)


            x = np.linspace(0.2, 5, 500)
            y = uniform.pdf(x, loc=I_K_param[0], scale=(I_K_param[1]-I_K_param[0]))
            
            bins = np.linspace(min(I_over_K), 4, 18)
            bins[17] = 15
            data = [I_over_K, x, y]
            data2 = [I_over_K]

            chart = {'OUTFILE' : (home + out + 'I_over_K' + img_ext),
              'XLABEL' : 'Mean annual inflow over storage capacity',
              'XMIN' : 0,
              'XMAX' : 4,
              'BINS' : bins}
            
            chart2 = {'OUTFILE' : (home + out + 'I_over_K_2' + img_ext),
              'XLABEL' : 'Mean annual inflow over storage capacity',
              'XMIN' : 0,
              'XMAX' : 4,
              'BINS' : bins}

            if charts:
                self.build_chart(chart, data, chart_type='hist')
                self.build_chart(chart2, data2, chart_type='hist')

            #---------------#       SD_over_I - SD of inflow over mean inflow  
            sample = AUS_RIVERS['MAR'] < 700
            SD_over_I = AUS_RIVERS['Cv'][sample] 
            
            bins = np.linspace(min(SD_over_I), max(SD_over_I), 13)
            x = np.linspace(min(SD_over_I), max(SD_over_I), 500)
            y = uniform.pdf(x, loc=0.43, scale=(1-0.4))
            
            data = [SD_over_I, x, y]
            data2 = [SD_over_I]
            
            chart = {'OUTFILE' : (home + out + 'SD_over_I' + img_ext),
              'XLABEL' : 'Standard deviation of annual inflow over mean',
              'XMIN' : 0,
              'XMAX' : max(SD_over_I), 
              'BINS' : bins}

            chart2 = {'OUTFILE' : (home + out + 'SD_over_I_2' + img_ext),
              'XLABEL' : 'Standard deviation of annual inflow over mean',
              'XMIN' : 0,
              'XMAX' : max(SD_over_I), 
              'BINS' : bins}

            if charts:
                self.build_chart(chart, data, chart_type='hist')
                self.build_chart(chart2, data2, chart_type='hist')
        
            #---------------#       SA_over_K - surface area over capacity
            
            SA = MDB_dams['sa']
            sample = (K > 50000)
            SA_over_K = SA[sample] / K[sample]
            bins = np.linspace(min(SA_over_K), max(SA_over_K), 28)
            SA_K_param = np.zeros(2)
            SA_K_param[0] = np.percentile(SA_over_K, 10) 
            SA_K_param[1] = np.percentile(SA_over_K, 81.5) 
            x = np.linspace(min(SA_over_K), max(SA_over_K), 500)
            y = uniform.pdf(x, loc=SA_K_param[0], scale=(SA_K_param[1] - SA_K_param[0]))

            data = [SA_over_K, x, y]
            data2 = [SA_over_K]
            
            chart = {'OUTFILE' : (home + out + 'SA_over_K' + img_ext),
              'XLABEL' : 'Storage surface area over storage capacity',
              'XMIN' : min(SA_over_K),
              'XMAX' : max(SA_over_K), 
              'BINS' : bins}
            
            chart2 = {'OUTFILE' : (home + out + 'SA_over_K_2' + img_ext),
              'XLABEL' : 'Storage surface area over storage capacity',
              'XMIN' : min(SA_over_K),
              'XMAX' : max(SA_over_K), 
              'BINS' : bins}
            
            if charts:
                self.build_chart(chart, data, chart_type='hist')
                self.build_chart(chart2, data2, chart_type='hist')
        
            #---------------#       evap - net evaporation rate
            
            evap = 0.75 * (MDB_dams['net_evap'] / 1000)
            folder = '/Dropbox/Thesis/STATS/chapter3/'
            evap = evap[sample]
            folder = '/Dropbox/Thesis/STATS/chapter3/'
            sample2 = evap > 0
            evap = evap[sample2]

            bins = np.linspace(min(evap), max(evap), 18)

            evap_param = np.zeros(2)
            evap_param[0] = np.percentile(evap, 15) 
            evap_param[1] = np.percentile(evap, 85) 

            x = np.linspace(min(evap), max(evap), 500)
            y = uniform.pdf(x, loc=evap_param[0], scale=(evap_param[1]-evap_param[0]))
            data = [evap, x, y]
            data2 = [evap]
            
            chart = {'OUTFILE' : (home + out + 'evap' + img_ext),
              'XLABEL' : 'Average annual evaporation rate (meters)',
              'XMIN' : min(evap),
              'XMAX' : max(evap), 
              'BINS' : bins}
            
            chart2 = {'OUTFILE' : (home + out + 'evap_2' + img_ext),
              'XLABEL' : 'Average annual evaporation rate (meters)',
              'XMIN' : min(evap),
              'XMAX' : max(evap), 
              'BINS' : bins}

            if charts:
                self.build_chart(chart, data, chart_type='hist')
                self.build_chart(chart2, data2, chart_type='hist')
       
            #---------------#      d_loss - delivery losses in irrigation areas

            d_loss = np.zeros([5, 2])
            
            Murray = pandas.read_csv(home + folder2 + 'Murray_loss.csv')
            y = Murray['Loss'] 
            N = len(y)
            X = np.vstack([Murray['Pumped'], np.ones(N)]).T
            d_loss[0,:] = np.linalg.lstsq(X, y)[0]
            d_loss[0,1] = d_loss[0,1] / np.mean(Murray['Pumped'])

            Shep = pandas.read_csv(home + folder2 + 'Shep_loss.csv')
            y = Shep['Loss'] 
            N = len(y)
            X = np.vstack([Shep['Pumped'], np.ones(N)]).T
            d_loss[1,:] = np.linalg.lstsq(X, y)[0]
            d_loss[1,1] = d_loss[1,1] / (np.mean(Shep['Pumped']))
            
            Jemm = pandas.read_csv(home + folder2 + 'Jemm_loss.csv')
            y = Jemm['Loss'] 
            N = len(y)
            X = np.vstack([Jemm['Pumped'], np.ones(N)]).T
            d_loss[2,:] = np.linalg.lstsq(X, y)[0]
            d_loss[2,1] = d_loss[2,1] / (np.mean(Jemm['Pumped']))
        
            Coll = pandas.read_csv(home + folder2 + 'Coll_loss.csv')
            y = Coll['Loss'] 
            N = len(y)
            X = np.vstack([Coll['Pumped'], np.ones(N)]).T
            d_loss[3,:] = np.linalg.lstsq(X, y)[0]
            d_loss[3,1] = d_loss[3,1] / (np.mean(Coll['Pumped']))
            
            MIA = pandas.read_csv(home + folder2 + 'MIA_loss.csv')
            y = MIA['Loss'] 
            N = len(y)
            X = np.vstack([MIA['Pumped'], np.ones(N)]).T
            d_loss[4,:] = np.linalg.lstsq(X, y)[0]
            d_loss[4,1] = d_loss[4,1] / (np.mean(MIA['Pumped']))

            #---------------#   Inflow autocorrelation

            rho_param = [0.2, 0.3]

            #################       WATER DEMAND         ################
            
            
            #---------------#      Theta parameters (yield functions)

            theta_mu = np.zeros([6, 2])
            theta_mu[:,0] = np.array([154.7, 236.7, -35.8, 20.7, 14.9, -48.5])
            #theta_mu[:,1] = np.array([-1785.4, 2545.3, -157.2, 1924.3, -537.5, -118.9])
            theta_mu[:,1] = np.array([-1773.8, 2135.0, -133.3, 1597.1, -520.9, -100.8]) 

            #clow = (41.5+14.9+231.7)*0.1
            #chigh = (-76.5+2153.9-537.5)*0.1

            theta_sig = np.zeros([6, 2])
            #theta_sig[:,0] = np.array([0, 51.1, 16.8, 96.8, 38.8, 17.3])
            #theta_sig[:,1] = np.array([0, 306.2, 22.3, 2370.1, 952.2, 151.4])
            theta_sig[:,0] = np.array([0, 51.1, 16.8, 0, 0, 17.3])
            theta_sig[:,1] = np.array([0, 306.2, 22.3, 0, 0, 151.4])
            
            q_bar_limits = np.zeros([2,2])
            q_bar_limits[:, 0] = np.array([0.5, 6.5])
            q_bar_limits[:, 1] = np.array([5, 14])

            w_ha = np.linspace(0,3,100)
            profit_ha = theta_mu[0,0] + theta_mu[1,0]*w_ha + theta_mu[2,0] * w_ha**2 + theta_mu[3,0]*1 + theta_mu[4,0]*1**2 + theta_mu[5,0]*1*w_ha
            
            data = [[w_ha, profit_ha]]
            chart = {'OUTFILE' : (home + out + 'low_yield' + img_ext),
              'XLABEL' : 'Water use per unit land (ML / HA)',
              'XMIN' : min(w_ha),
              'XMAX' : max(w_ha),
              'YMIN' : 0,
              'YMAX' : max(profit_ha)*1.05,
              'YLABEL' : 'Profit per unit land (\$ / HA)'}
            if charts:
                self.build_chart(chart, data, chart_type='plot')
            
            w_ha = np.linspace(0,9,100)
            profit_ha = theta_mu[0,1] + theta_mu[1,1]*w_ha + theta_mu[2,1] * w_ha**2 + theta_mu[3,1]*1 + theta_mu[4,1]*1**2 + theta_mu[5,1]*1*w_ha

            data = [[w_ha, profit_ha]]
            chart = {'OUTFILE' : (home + out + 'high_yield' + img_ext),
              'XLABEL' : 'Water use per unit land (ML / HA)',
              'XMIN' : min(w_ha),
              'XMAX' : max(w_ha),
              'YMIN' : min(profit_ha)*1.5,
              'YMAX' : max(profit_ha)*1.05,
              'YLABEL' : 'Profit per unit land (\$ / HA)'}
            if charts: 
                self.build_chart(chart, data, chart_type='plot')
            

            #---------------#       Epsilon parameters (yield functions)

            rho_eps_param = np.array([0.3, 0.5])
            sig_eta_param = np.array([0.1, 0.2])
        
            #################       Final Parameters       ################

            self.I_K_param = I_K_param
            self.SD_I_param = np.array([0.4, 1])
            self.rho_param = [0.2, 0.3]
            self.SA_K_param = SA_K_param
            self.evap_param = evap_param
            self.d_loss_param_a = np.array([0, 0.15])
            self.d_loss_param_b = np.array([0.15, 0.30])
            self.theta_mu = theta_mu
            self.theta_sig = theta_sig / 3
            self.q_bar_limits = q_bar_limits
            self.rho_eps_param = np.array([0.3, 0.5])
            self.sig_eta_param = np.array([0.1, 0.2])
            self.prop_high = np.array([0.05, 0.35])
            self.target_price =  10 #np.array([50,200])
            self.t_cost_param = np.array([10, 100])
            self.Lambda_high_param = np.array([1, 2])
            self.relative_risk_aversion = np.array([0, 3])

            para_dist = [self.I_K_param, self.SD_I_param, self.rho_param, self.SA_K_param, self.evap_param, self.d_loss_param_a, self.d_loss_param_b,  self.t_cost_param, self.Lambda_high_param, [100, 100], [30, 70], [30, 70], self.theta_mu, self.theta_sig, self.q_bar_limits, self.rho_eps_param, self.sig_eta_param, self.prop_high, self.target_price, self.relative_risk_aversion]

            with open('para_dist.pkl', 'wb') as f:
                pickle.dump(para_dist, f)
                f.close()

            rows = ['$E[I_t]/K$', '$c_v$', '$\rho_I$' ,'$\alpha K^{2/3} / K $' ,  '$\delta_0$', '$\delta_{1a}$', '$\delta_{1b}$', '$\tau$', '$\Lambda_{high}$', '$n$', '$n_low$', '$n_high$', ]
            cols = ['Min', 'Central case', 'Max']
            n_rows = 12
            data = []
            for i in range(n_rows):
                record = {}
                record['Min'] = para_dist[i][0]
                record['Central case'] = np.mean(para_dist[i])
                record['Max'] = para_dist[i][1]
                data.append(record)
            tab = pandas.DataFrame(data)
            tab.index = rows
            tab_text = tab.to_latex(float_format =  '{:,.2f}'.format, columns=['Min', 'Central case', 'Max' ], index=True)
            with open(home + folder + "para_table.txt", "w") as f:
                f.write(tab_text)
                f.close()

        else:
            with open('para_dist.pkl', 'rb') as f:
                para_dist = pickle.load(f)
                f.close()
    
            self.I_K_param      = para_dist[0]
            self.SD_I_param     = para_dist[1]
            self.rho_param      = para_dist[2]
            self.SA_K_param     = para_dist[3]
            self.evap_param     = para_dist[4]
            self.d_loss_param_a = para_dist[5]
            self.d_loss_param_b = para_dist[6]
            self.t_cost_param   = para_dist[7] 
            self.Lambda_high_pa = para_dist[8] 
            self.theta_mu       = para_dist[12]
            self.theta_sig      = para_dist[13]
            self.q_bar_limits   = para_dist[14]
            self.rho_eps_param  = para_dist[15]
            self.sig_eta_param  = para_dist[16]
            self.prop_high      = para_dist[17]
            self.target_price   = para_dist[18]
            self.relative_risk_aversion = para_dist[19] 
            
    def set_property_rights(self, scenario='CS'):

        if scenario == 'CS':
            self.sr = 'CS'
            self.ls = 1
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'SWA':
            self.sr = 'SWA'
            self.ls = 1
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'NS':
            self.sr = 'NS'
            self.ls = 0
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'OA':
            self.sr = 'OA'
            self.ls = 0
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'CS-SWA':
            self.sr = 'CS-SWA'
            self.ls = 1
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'CS-SL':
            self.sr = 'CS'
            self.ls = 0
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'SWA-SL':
            self.sr = 'SWA'
            self.ls = 0
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'CS-SWA-SL':
            self.sr = 'CS-SWA'
            self.ls = 0
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'RS':
            self.sr = 'RS'
            self.ls = 0
            self.HL = 0
            self.opt_lam = 0
        elif scenario == 'RS-HL':
            self.sr = 'RS'
            self.ls = 0
            self.HL = 1
            self.opt_lam = 0
        elif scenario == 'RS-O':
            self.sr = 'RS'
            self.ls = 0
            self.HL = 0
            self.opt_lam = 1
        elif scenario == 'RS-HL-O':
            self.sr = 'RS'
            self.ls = 0
            self.HL = 1
            self.opt_lam = 1
        elif scenario == 'CS-HL-O':
            self.sr = 'CS'
            self.ls = 1
            self.HL = 1
            self.opt_lam = 1
        elif scenario == 'CS-O':
            self.sr = 'CS'
            self.ls = 1
            self.HL = 0
            self.opt_lam = 1
        
        if self.opt_lam == 1:
            if self.HL == 1:
                self.Lambda_high = self.prop * 1
            else:
                self.Lambda_high = self.prop * 1.5
        else:
            self.Lambda_high = self.prop

    def central_case(self, N=100, utility=False, printp=True, risk=0):
        
        self.K = 1000000
        self.beta = 0.945
        
        I_K = np.mean(self.I_K_param)
        SD_I = np.mean(self.SD_I_param)
        
        self.I_m = I_K * self.K
        self.I_sig = SD_I * self.I_m
        
        SA_K = np.mean(self.SA_K_param)
        
        self.I_bar = I_K * self.K
        self.SD_I = self.I_bar * SD_I
        self.rho_I = np.mean(self.rho_param)
        self.delta0 = np.mean(self.evap_param)

        m = (I_K * self.K) * (1-self.rho_I)
        v = (((SD_I * (I_K*self.K))**2)*(1-self.rho_I**2))

        # Log normal inflow shocks
        #self.mu_I = log((m**2)/((v+m**2)**0.5)) 
        #self.sig_I = (log(1 + (v/(m**2))))**0.5
        
        # Gamma inflow shocks
        self.theta_I = v / m
        self.k_I = m**2 / v
        
        self.alpha = (SA_K*self.K) / (self.K**(2.0/3))

        self.delta1a = np.mean(self.d_loss_param_a) * self.I_bar
        self.delta1b = np.mean(self.d_loss_param_b)

        self.theta = np.zeros([6,2])
        self.theta = self.theta_mu

        self.rho_eps = np.mean(self.rho_eps_param)
        self.sig_eta = np.mean(self.sig_eta_param)

        self.price = np.mean(self.target_price)
        self.prop = np.mean(self.prop_high) 
        p = max(self.price, 0)
        q_bar_low = (-1*(self.theta[1,0]+self.theta[5,0])) / (2*self.theta[2,0]) + p/(self.theta[2,0]*2)
        q_bar_high = (-1*(self.theta[1,1]+self.theta[5,1])) / (2*self.theta[2,1]) + p/(self.theta[2,1]*2)
        self.high_size = self.prop / (q_bar_high / q_bar_low)

        self.N = N
        self.N_low = int(N/2)
        self.N_high = int(N/2)
       
        self.L = newton(self.excessp, 2000, args=(self.price, ))
        self.Land = self.L    
        self.t_cost = np.mean(self.t_cost_param)
        
        p = max(self.max_price(self.L, self.price), 0)
        q_bar_low = (-1* (self.theta[1,0]+self.theta[5,0])) / (2*self.theta[2,0]) + p/(self.theta[2,0]*2) 
        q_bar_high = (-1* (self.theta[1,1]+self.theta[5,1])) / (2*self.theta[2,1]) + p/(self.theta[2,1]*2)
        self.prop = (self.N_high * q_bar_high * (self.high_size * self.L)) / ((self.N_high * q_bar_high * (self.high_size * self.L)) + self.N_low * q_bar_low * self.L)

        if self.opt_lam == 1:
            self.Lambda_high = self.prop * 1 #np.mean(self.Lambda_high_param)
        else:
            self.Lambda_high = self.prop * 1.5
        
        
        if utility:
            self.utility = 1 
            risk = risk #self.relative_risk_aversion[1]
            
            low_land = self.L
            high_land = self.L * self.high_size
            profit_bar_high = low_land * (self.theta[0, 0] + self.theta[1, 0]*q_bar_low + self.theta[2, 0]*(q_bar_low**2)+ self.theta[3, 0] + self.theta[4, 0] + self.theta[5, 0]*q_bar_low)
            profit_bar_low = high_land * (self.theta[0, 1] + self.theta[1, 1]*q_bar_high + self.theta[2, 1]*(q_bar_high**2)+ self.theta[3, 1] + self.theta[4, 1] + self.theta[5, 1]*q_bar_high)
            
            self.risk_aversion_low = risk / profit_bar_high
            self.risk_aversion_high = risk / profit_bar_low
        else: 
            self.utility = 0 
            self.risk_aversion_low = 0
            self.risk_aversion_high = 0
        
        self.para_list = {'I_K': I_K, 'SD_I' : SD_I, 'Prop_high' :  self.prop, 't_cost' : self.t_cost, 'L' : self.Land,'N_high' : self.N_high, 'rho_I' : self.rho_I, 'SA_K' : SA_K, 'alpha' : self.alpha, 'delta1a' : self.delta1a, 'delta1b' : self.delta1b}
        
        if printp:
            print '\n --- Main parameters --- \n'
            print 'Inflow to capacity: ' + str(I_K)
            print 'Coefficient of variation: ' + str(SD_I)
            print 'Proportion of high demand: ' + str(self.prop)
            print 'Target water price: ' + str(self.price)
            print 'Transaction cost: ' + str(self.t_cost)
            print 'High user inflow share: ' + str(self.Lambda_high)
            print 'Land: ' + str(self.L)
            print 'High Land: ' + str(self.high_size)
    
    def excessp(self, L, target_price):

        p = self.max_price_max(L, target_price)

        return p - target_price

    def max_price(self, L, target_price):
        
        #Q = self.K - self.delta1a*self.K - self.delta1b*self.K
        W = min(self.I_bar, self.K)
        Q = max(W - self.delta1a - self.delta1b*W, 0)
        p  = newton(self.excessD, target_price, args = (Q, L))

        return p
    
    def max_price_max(self, L, target_price):
        
        Q = self.K - self.delta1a - self.delta1b*self.K
        #Q = max(self.I_bar - self.delta1a*self.K - self.delta1b*self.I_bar, 0)
        p  = newton(self.excessD, target_price, args = (Q, L))

        return p

    def excessD(self, p, Q, L):

        qsum = self.low_demand(p, L) + self.high_demand(p, L)

        return qsum - Q

    def low_demand(self, p, L):
        qi = (-(self.theta[1,0] + self.theta[5,0])/(2*self.theta[2,0]) + p / (self.theta[2,0]*2) )
        q = (qi * L * self.N_low) 

        return q

    def high_demand(self, p, L):
        qi = (-(self.theta[1,1] + self.theta[5,1])/(2*self.theta[2,1]) + p / (self.theta[2,1]*2) )
        q = (qi * (self.high_size*L) * self.N_high)
    
        return q

    def randomize(self, N = 100):
        
        ex = 10
        self.L = 1000000
    
        self.K = 1000000
        self.beta = uniform.rvs(loc=0.93, scale =0.03)

        I_K = uniform.rvs(loc=self.I_K_param[0], scale=(self.I_K_param[1] - self.I_K_param[0]))
        SD_I = uniform.rvs(loc=self.SD_I_param[0], scale=(self.SD_I_param[1]-self.SD_I_param[0]))
        SA_K = uniform.rvs(loc=self.SA_K_param[0], scale=(self.SA_K_param[1] - self.SA_K_param[0]))
        
        self.I_bar = I_K * self.K
        
        self.rho_I = uniform.rvs(loc=self.rho_param[0], scale=(self.rho_param[1]-self.rho_param[0]))
        self.delta0 = uniform.rvs(loc=self.evap_param[0], scale=(self.evap_param[1] - self.evap_param[0])) 
        
        m = (I_K * self.K) * (1-self.rho_I)
        v = (((SD_I * (I_K*self.K))**2)*(1-self.rho_I**2))

        #print 'm ' + str(m)
        #print 'v ' + str(m)
        # Log normal inflow 
        #self.mu_I = log((m**2)/((v+m**2)**0.5)) 
        #self.sig_I = (log(1 + (v/(m**2))))**0.5
        
        # Gamma inflow
        self.theta_I =  v / m
        self.k_I = m**2 / v
        #print 'theta_I ' + str(self.theta_I)
        #print 'k_I ' + str(self.k_I)
        
        self.alpha = (SA_K*self.K) / (self.K**(2.0/3))

        self.delta1a = uniform.rvs(loc=self.d_loss_param_a[0], scale=(self.d_loss_param_a[1]-self.d_loss_param_a[0])) * self.I_bar
        self.delta1b = uniform.rvs(loc=self.d_loss_param_b[0], scale=(self.d_loss_param_b[1]-self.d_loss_param_b[0]))

        self.theta = np.zeros([6, 2])
        self.theta[0,:] = self.theta_mu[0,:]
        
        constraint_binding = True
        iters = 0

        theta = np.zeros([6, 2])
        while constraint_binding and iters < 10000:
            iters += 1
            for k in [3, 4]:
                theta[k,0] = self.theta_mu[k,0] 
                theta[k,1] = self.theta_mu[k,1] 

            for k in [0, 1, 2, 5]:
                theta[k,0] = norm.rvs(loc=self.theta_mu[k,0], scale=self.theta_sig[k,0])
                theta[k,1] = norm.rvs(loc=self.theta_mu[k,1], scale=self.theta_sig[k,1])
            
            if theta[1,0] > 0 and theta[1, 1] > 0:
                if theta[2, 0] < 0 and theta[2, 1] < 0:
                    if theta[5, 0] < 0 and theta[5, 1] < 0:
                        qbarlow = -(theta[1,0]+theta[5,0]) / (2 * theta[2,0]) 
                        if qbarlow > self.q_bar_limits[0,0] and qbarlow < self.q_bar_limits[1,0]:
                            qbarhigh = -(theta[1,1]+theta[5,1]) / (2 * theta[2,1]) 
                            if qbarhigh > self.q_bar_limits[0,1] and qbarhigh < self.q_bar_limits[1,1]:
                                constraint_binding = False
        self.theta = theta
        
        if iters == 10000:
            print 'Warning parameters not generated...'
            self.theta = self.theta_mu

        self.rho_eps = uniform.rvs(loc=self.rho_eps_param[0], scale=(self.rho_eps_param[1]-self.rho_eps_param[0]))
        self.sig_eta = uniform.rvs(loc=self.sig_eta_param[0], scale=(self.sig_eta_param[1]-self.sig_eta_param[0]))

        #self.price = uniform.rvs(loc=self.target_price[0], scale=(self.target_price[1]-self.target_price[0]))
        
        self.N_low = int(uniform.rvs(loc=30, scale=40))
        self.N_high = N - self.N_low
        self.L_ratio =  uniform.rvs(loc=0.75, scale=0.5)
        self.L = self.L_ratio * I_K *  (self.Land / (np.mean(self.I_K_param)))
        
        p = max(self.max_price(self.L, self.price), 0)
        
        q_bar_low = max((-1* (self.theta[1,0]+self.theta[5,0])) / (2*self.theta[2,0]) + p/(self.theta[2,0]*2),0) 
        q_bar_high = max((-1* (self.theta[1,1]+self.theta[5,1])) / (2*self.theta[2,1]) + p/(self.theta[2,0]*2),0)
        self.prop = (self.N_high * q_bar_high * (self.high_size * self.L)) / ((self.N_high * q_bar_high * (self.high_size * self.L)) + self.N_low * q_bar_low * self.L)
        
        low_land = self.L
        high_land = self.L * self.high_size
        profit_bar_high = low_land * (self.theta[0, 0] + self.theta[1, 0]*q_bar_low + self.theta[2, 0]*(q_bar_low**2)+ self.theta[3, 0] + self.theta[4, 0] + self.theta[5, 0]*q_bar_low) 
        profit_bar_low = high_land * (self.theta[0, 1] + self.theta[1, 1]*q_bar_high + self.theta[2, 1]*(q_bar_high**2)+ self.theta[3, 1] + self.theta[4, 1] + self.theta[5, 1]*q_bar_high) 
        
        self.utility = 1    
        risk = uniform.rvs(loc=self.relative_risk_aversion[0], scale=self.relative_risk_aversion[1] - self.relative_risk_aversion[0])
        self.risk_aversion_low = risk / profit_bar_high
        self.risk_aversion_high = risk / profit_bar_low

        self.t_cost = (np.random.rand() * (self.t_cost_param[1]-self.t_cost_param[0]) + self.t_cost_param[0]) 
        
        self.Lambda_high = self.max(min(self.prop, 0.95), 0)  #* (np.random.rand() * (self.Lambda_high_param[1]-self.Lambda_high_param[0]) + self.Lambda_high_param[0]),0.95),0)
        
        self.para_list = {'I_K' : I_K, 'SD_I' : SD_I, 'Prop_high' :  self.prop, 't_cost' : self.t_cost, 'L' : self.Land,'N_high' : self.N_high, 'rho_I' : self.rho_I, 'SA_K' : SA_K, 'alpha' : self.alpha, 'delta1a' : self.delta1a, 'delta1b' : self.delta1b}
        
        print '\n --- Main parameters --- \n'
        print 'Inflow to capacity: ' + str(I_K)
        print 'Coefficient of variation: ' + str(SD_I)
        print 'Proportion of high demand:' + str(self.prop)
        print 'Target water price: ' + str(p)
        print 'Transaction cost: ' + str(self.t_cost)
        print 'High user inflow share: ' + str(self.Lambda_high)
        print 'Number of high users: ' + str(self.N_high)
        print 'Land: ' + str(self.L)

    def check_random(self, num=2000, I = 1):

        xlow = np.linspace(0,4, 200)
        xhigh = np.linspace(0,10, 200)

        ylow = np.zeros([200,num])
        yhigh = np.zeros([200,num])

        pylab.figure(0)
        pylab.figure(1)
        for i in range(num):
            self.randomize()
            ylow[:,i] = xlow * (self.theta[1,0]) + xlow * self.theta[5,0] * I + (xlow**2) * self.theta[2,0] + self.theta[0,0]+self.theta[3,0]+self.theta[4,0] 
            yhigh[:,i] = xhigh * (self.theta[1,1])  + xhigh * self.theta[5,1] * I + (xhigh**2) * self.theta[2,1] + self.theta[0,1]+self.theta[3,1]+self.theta[4,1] 

        ymeanlow = np.mean(ylow, axis=1) 
        ymeanhigh = np.mean(yhigh, axis=1) 
        yminlow = np.percentile(ylow, 10.0, axis=1) 
        yminhigh = np.percentile(yhigh, 10.0, axis=1) 
        ymaxlow = np.percentile(ylow, 90.0, axis=1) 
        ymaxhigh = np.percentile(yhigh, 90.0, axis=1) 
        ymulow = xlow * (self.theta_mu[1,0] + self.theta_mu[5,0]) + (xlow**2) * self.theta_mu[2,0] + self.theta_mu[0,0]+self.theta_mu[3,0]+self.theta_mu[4,0]
        ymuhigh = xhigh * (self.theta_mu[1,1] + self.theta_mu[5,1]) + (xhigh**2) * self.theta_mu[2,1] + self.theta_mu[0,1]+self.theta_mu[3,1]+self.theta_mu[4,1]
        
        pylab.figure(0)
        pylab.plot(xlow, ymeanlow)
        pylab.plot(xlow, ymulow)
        pylab.plot(xlow, yminlow)
        pylab.plot(xlow, ymaxlow)
        pylab.plot(xlow, ylow[:,10])
        pylab.ylim(0, 800)
        pylab.figure(1)
        pylab.plot(xhigh, ymeanhigh)
        pylab.plot(xhigh, ymuhigh)
        pylab.plot(xhigh, yminhigh)
        pylab.plot(xhigh, ymaxhigh)
        pylab.plot(xhigh, yhigh[:,10])
            
        pylab.show()
    
    def solve_para(self):
        
        """
        
                Specify model solution parameters, sample sizes, iterations, exploration etc.
        
        """

        self.CPU_CORES = 4 #mp.cpu_count()

        # Evaluation simulation size (for generating model results)
        self.T0 = 500000   
        
        #======================================================
        #       Planner SDP parameters
        #======================================================
       
        # SDP Tolerance level 
        self.SDP_TOL = 0.0012

        # Max SDP Policy evaluations per policy improvement
        self.SDP_ITER = 100

        # SDP GRID points per dimension, S_t, I_t and I_t+1
        self.SDP_GRID = 33
        
        #======================================================
        #       Planner QV learning parameters
        #======================================================

        # Simulation sample size
        self.T1 = 50000

        # Max number of sample grid points
        self.s_points1 = 425

        # Sample grid radius
        self.s_radius1 = 0.02

        # Stage 2 search range
        self.policy_delta = 0.2

        self.QV_ITER1 = 40
        self.QV_ITER2 = 10

        #======================================================
        #       Optimal share search parameters
        #======================================================
        
        self.opt_lam_ITER = 12

        #======================================================
        #       Decentralised model QV learning parameters
        #======================================================

        self.ITER1 = 45             # Initialization stage QV iterations
        self.ITER2 = 20              # Main learning iterations
        self.iters = 1              # QV iterations per learning iteration 

        #Proportion of users to update
        self.update_rate = [0.12] * 5 + [0.1] * 15 
        
        # Number of exploring agents per class
        self.N_e = [5] * 10 + [2] * 10 

        # Exploration range
        self.d = [0] * 5 + [0.5] * 10 + [0.25] * 5 

        # Total sample size, actual sim length = T1 / (2*N_e)
        self.T2 = 600000
        
        # State sample grid radius
        self.s_radius2 = 0.05

        # State sample gird max number of points
        self.s_points2 = 3000

    def test(self, prop = 0.1):

        "Generate parameters for the test version of the model"

        self.K = 50                                      # Storage capacity
        self.S = 50                                      # Initial storage level

        self.ITERS = 10
        self.prop = prop

        self.F = [0, 25, 50]                            # Inflow distribution coefficients
        self.loss = [0]

        self.N = 50  				                    # Number of users
        self.N_low = 25                                  # Number of low reliability users (rest are high)

        self.rs = 'SWA' 				                    # Property rights system (CS, SWA, CPS, NS)

        self.beta = 0.90                                 # Discount rate

        self.u_para_low = [0, 1.2, -0.4]                # Low reliability payoff function coefficients
        self.u_para_high = [-0.5, 2.5, -1.4]            # High reliability payoff function coefficients
        self.e_para = [0.8, 1.2]                       # Maximum productivity level

        self.t_cost = 0.3                                # Spot market transaction cost per unit water

        self.low_c = 0.5                               # Low users share of inflow / storage capacity

        self.temperature = 0.18                         # Degree of exploration

        self.CPU_CORES = 4
        self.TOL = 0.01
        self.T1 = 60000
        self.T2 = 60000
        self.D = 3

        self.a = [0, 0, 0, self.e_para[0]]

        self.samples = int(self.T1/2)
        self.s_radius = 0.075



    def fit_lognorm(self, x, scale = True):
        
        
        if scale:
            minx = min(x)
            m = np.mean(x - minx)
            v = np.var(x - minx)
        else:
            minx = -1.5
            m = np.mean(x)
            v = np.var(x)

        e_mu = m**2 / ((v+m**2)**0.5)
        sig = np.log(1+ v/(m**2))**0.5

        return (sig, minx, e_mu)

    def build_chart(self, chart, data_set, chart_type='plot', ticks = False):
    
        import pylab
        
        pylab.ioff()
        fig_width_pt = 350 					     # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27                # Convert pt to inch
        golden_mean = ((5**0.5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt   # width in inches   
        fig_height = fig_width*golden_mean       # height in inches
        fig_size =  [fig_width,fig_height]

        params = { 'backend': 'ps',
               'axes.labelsize': 10,
               'text.fontsize': 10,
               'legend.fontsize': 10,
               'xtick.labelsize': 8,
               'ytick.labelsize': 8,
               'text.usetex': True,
               'figure.figsize': fig_size }

        pylab.rcParams.update(params)

    
        if chart_type == 'plot':
            pylab.figure()
            [pylab.plot(series[0], series[1]) for series in data_set]
        elif chart_type == 'scatter':
            pylab.figure()
            [pylab.plot(series[0], series[1], 'o') for series in data_set]
        elif chart_type == 'bar':
            pylab.figure()
            [pylab.bar(series[0], series[1], width = 2) for series in data_set]
        elif chart_type =='hist':
            if len(data_set) == 1:
                pylab.figure()
                pylab.hist(data_set[0], bins = chart['BINS'], normed=False)
                pylab.xlim(chart['XMIN'], chart['XMAX'])
            else:
                pylab.hist(data_set[0], bins = chart['BINS'], normed=True)
                pylab.xlim(chart['XMIN'], chart['XMAX'])
                pylab.plot(data_set[1], data_set[2])
        elif chart_type == 'date':
            data_set.plot() 
        
        if not(chart_type == 'date'):    
            pylab.xlim(chart['XMIN'], chart['XMAX'])
        
        if chart_type == 'bar':
            pylab.xticks(chart['XMIN'] + np.arange(chart['XTICKS']) * chart['XSTEP'], chart['LABELS'])
        elif ticks:
            pylab.xticks(chart['XMIN'] + np.arange(chart['XTICKS']) * chart['XSTEP'])
            pylab.yticks(chart['YMIN'] + np.arange(chart['YTICKS']) * chart['YSTEP'])
        
        pylab.xlabel(chart['XLABEL'])
        if not chart_type == 'hist':
            pylab.ylim(chart['YMIN'], chart['YMAX'])
            pylab.ylabel(chart['YLABEL'])
        
        pylab.savefig(chart['OUTFILE'], bbox_inches='tight')

        pylab.show()
