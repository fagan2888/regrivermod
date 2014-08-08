from chartbuilder import *

def build(results, W_f, V_f, SW_f):

    solution(W_f, V_f, SW_f)
    tables(result)
    time_series(result, 0, 50)
    time_series(result, 50, 100)
    
def solution(W_f, V_f, SW_f):

    home = '/home/nealbob'
    out = '/Dropbox/Thesis/IMG/chapter3/'
    img_ext = '.pdf'

    K = max(W_f.xdata[:,0])
    X =[np.linspace(0, K, 200), np.ones(200)]
    X0 =[np.linspace(0, K, 200), np.zeros(200)]
    W = W_f(X)
    W0 = W_f(X0)

    # Policy chart
    data0 = []
    for i in range(200):
        record = {}
        record['Optimal $I_t = E[I_t]$'] = W[i] / 1000
        record['Optimal $I_t = 0$'] = W0[i] / 1000
        record['Myopic'] = X[0][i] / 1000
        data0.append(record)
    data = pandas.DataFrame(data0)
    data.index = X[0]/1000
    chart = {'OUTFILE' : (home + out + 'Policy' + img_ext),
      'YLABEL' : 'Withdrawal (GL)',
      'XLABEL' : 'Storage (GL)', 
      'YMIN' : 0, 
      'YMAX' : K / 1000}
    
    build_chart(chart, data, chart_type='date', ylim = True)
    
    V = V_f(X)
    V0 = V_f(X0)

    #Value chart
    data0 = []
    for i in range(200):
        record = {}
        record['Optimal $I_t = E[I_t]$'] = V[i] / 1000000
        record['Optimal $I_t = 0$'] = V0[i] / 1000000
        data0.append(record)
    data = pandas.DataFrame(data0)
    data.index = X[0]/1000

    chart = {'OUTFILE' : (home + out + 'Value' + img_ext),
      'YLABEL' : 'Value (\$m)',
      'XLABEL' : 'Storage (GL)'}
    
    build_chart(chart, data, chart_type='date')
    
    K = max(SW_f.xdata[:,0])
    X =[np.linspace(0, K, 200), np.ones(200)]
    X0 =[np.linspace(0, K, 200), np.ones(200)*0.5]
    X2 =[np.linspace(0, K, 200), np.ones(200)*2]
    SW = SW_f(X)
    SW0 = SW_f(X0)
    SW2 = SW_f(X2)

    #Social welfare
    #data0 = []
    #for i in range(200):
    #    record = {}
    #    record['$I_t = E[I_t]$'] = SW[i] / 1000000
        #record['$\tilde I_t = 0.5$'] = SW0[i] / 1000000
        #record['$\tilde I_t = 2$'] = SW2[i] / 1000000
    #    data0.append(record)
    data = [[X[0]/1000, SW/1000000]] #pandas.DataFrame(data0)

    chart = {'OUTFILE' : (home + out + 'Welfare_f' + img_ext),
      'YLABEL' : 'Social welfare (\$m)',
      'XLABEL' : 'Total water use (GL)',
      'XMIN' : min(X[0])/1000,
      'XMAX': max(X[0])/1000,
      'YMIN' : 0,
      'YMAX': 1.1*max(SW)/1000000}
    build_chart(chart, data, chart_type='plot')

def time_series(result, a, b):
    
    """
        Generate charts and tables for report
    
    """

    home = '/home/nealbob'
    out = '/Dropbox/Thesis/IMG/chapter3/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter3/'       

    # Storage
    data0 = []
    for i in range(a,b):
        record = {}
        record['Optimal'] = result['series'][0][0]['S'][i] / 1000#  - result['series'][0][0]['W'][i] / 1000
        #record['x'] = result['series'][0][0]['S'][i] / 1000  - result['series'][0][0]['I'][i] / 1000
        record['Myopic'] = result['series'][0][1]['S'][i] / 1000
        data0.append(record)
    data = pandas.DataFrame(data0)

    chart = {'OUTFILE' : (home + out + 'Storage' + img_ext),
      'YLABEL' : 'Storage (GL)',
      'XLABEL' : 'Time' }

    build_chart(chart, data, chart_type='date')

    # Price
    data0 = []
    for i in range(a,b):
        record = {}
        record['Optimal'] = result['series'][0][0]['P'][i] 
        record['Myopic'] = result['series'][0][1]['P'][i]
        data0.append(record)
    data = pandas.DataFrame(data0)

    chart = {'OUTFILE' : (home + out + 'Price' + img_ext),
      'YLABEL' : 'Price (\$/ML)',
      'XLABEL' : 'Time' }

    build_chart(chart, data, chart_type='date')
    
    # Social Welfare
    data0 = []
    for i in range(a,b):
        record = {}
        record['Optimal'] = result['series'][0][0]['SW'][i] / 1000000
        record['Myopic'] = result['series'][0][1]['SW'][i] / 1000000
        data0.append(record)
    data = pandas.DataFrame(data0)

    chart = {'OUTFILE' : (home + out + 'Welfare' + img_ext),
      'YLABEL' : 'Welfare (\$ Million)',
      'XLABEL' : 'Time' }

    build_chart(chart, data, chart_type='date')

    # Withdrawal
    data0 = []
    for i in range(a,b):
        record = {}
        record['Optimal'] = result['series'][0][0]['W'][i] / 1000
        record['Myopic'] = result['series'][0][1]['W'][i] / 1000
        data0.append(record)
    data = pandas.DataFrame(data0)

    chart = {'OUTFILE' : (home + out + 'Withdrawal' + img_ext),
      'YLABEL' : 'Withdrawal (GL)',
      'XLABEL' : 'Time' }

    build_chart(chart, data, chart_type='date')
 
def tables(result=0):

    home = '/home/nealbob'
    table_out = '/Dropbox/Thesis/STATS/chapter3/'

    home = '/home/nealbob'
    inf = '/Dropbox/Model/Results/chapter3/'
    out = '/Dropbox/Thesis/IMG/chapter3/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter3/'       
   
    with open(home + inf + 'result.pkl', 'rb') as f:
        result = pickle.load(f)
        f.close()

    ### Result Tables

    series = ['S', 'SW', 'W', 'Z', 'P']
    stats = ['Mean', 'SD', '25th', '75th', '2.5th', '97.5th']
    series_name = {'W' : 'Withdrawals, (GL)', 'S' : 'Storage (GL)', 'SW' : 'Social Welfare (\\$m)', 'Z' : 'Spills (GL)', 'P' : 'Shadow Price (\\$/ML)'}
    
    for x in series:
        if x=='SW' or x=='U_low' or x=='U_high':
            scale = 1000000
        elif x=='P':
            scale = 1
        else:
            scale = 1000
        data0 = []
        
        record = {}
        for stat in stats:
            record[stat] = result['stats'][0][x][stat][0] / scale
        data0.append(record)
        record = {}
        for stat in stats:
            record[stat] = result['stats'][0][x][stat][1] / scale
        data0.append(record)
        tab = pandas.DataFrame(data0)
        tab.index = ['Optimal', 'Myopic']
        tab_text = tab.to_latex(float_format =  '{:,.2f}'.format, columns=['Mean', 'SD', '2.5th', '25th', '75th', '97.5th'])  
        text_list = tab_text.split('\n')

        if x == 'S':
            with open(home + table_out + "central_case.txt", "w") as f:
                f.write(text_list[0]  + "\n" + text_list[1] + "\n"  + text_list[2] +"\n")
                f.close()
        
        with open(home + table_out + "central_case.txt", "a") as f:
            f.write("\midrule \n" + series_name[x] +"\\\\" + "\n\midrule\n") 
            f.write(text_list[4] + "\n"  + text_list[5] + "\n") 
            f.close()
        
        if x == 'P':
            with open(home + table_out + "central_case.txt", "a") as f:
                f.write("\\bottomrule \n \end{tabular}") 
                f.close()
        #with open(home + table_out + x +"_table.txt", "w") as f:
        #    f.write(tab.to_latex(float_format =  '{:,.1f}'.format, columns=['Mean', 'SD', '2.5th', '25th', '75th', '97.5th'])) 
        #    f.close()
    
def sens_results():

    home = '/home/nealbob'
    inf = '/Dropbox/Model/Results/chapter3/'
    out = '/Dropbox/Thesis/IMG/chapter3/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter3/'       
   
    import chart

    with open(home + inf + 'result.pkl', 'rb') as f:
        result = pickle.load(f)
        f.close()
    
    data = {'W' : 0, 'S' : 0, 'SW' : 0, 'Z' : 0}
    data0 = []

    n = len(result['stats'])
    series = ['W', 'S', 'SW', 'Z',]
    series_name = {'W' : 'Withdrawals (GL)', 'S' : 'Storage (GL)', 'SW' : 'Social Welfare (\\$m)', 'Z' : 'Spills (GL)'}
    stats = ['Mean', 'SD']
    SAMP = np.ones(n) > 0

    for y in stats:
        for x in series:
            print x
            print y
            if x=='SW' or x=='U_low' or x=='U_high':
                scale = 1000000
            elif x=='P':
                scale = 1
            else:
                scale = 1000
            data0 = []

            opt = np.zeros(n)
            myo = np.zeros(n)
            idx = np.zeros(n)
            for i in range(n):
                opt[i] = result['stats'][i][x][y][0] / scale
                myo[i] = result['stats'][i][x][y][1] / scale
                idx[i] = myo[i] / opt[i]
            
            if x == 'W': 
                SAMP = opt > 25

            opt = opt[SAMP]
            myo = myo[SAMP]
            idx = idx[SAMP]

            chart.chart(idx, 0.98*min(idx), 1.02*max(idx), 'Myopic relative to optimal', x + '_' + y + '_index')
        
            data0 = []
            record = {}
            record['Mean'] = np.mean(opt)
            record['Min'] = np.min(opt)
            record['Q1'] = np.percentile(opt, 25)
            record['Q3'] = np.percentile(opt, 75)
            record['Max'] = np.max(opt)
            data0.append(record)
            
            record = {}
            record['Mean'] = np.mean(myo)
            record['Min'] = np.min(myo)
            record['Q1'] = np.percentile(myo, 25)
            record['Q3'] = np.percentile(myo, 75)
            record['Max'] = np.max(myo)
            data0.append(record)
            
            record = {}
            record['Mean'] = np.mean(idx)
            record['Min'] = np.min(idx)
            record['Q1'] = np.percentile(idx, 25)
            record['Q3'] = np.percentile(idx, 75)
            record['Max'] = np.max(idx)
            data0.append(record)
            
            tab = pandas.DataFrame(data0)
            tab.index = ['Optimal', 'Myopic', 'Myopic / Optimal']
            tab_text = tab.to_latex(float_format =  '{:,.2f}'.format , columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'])
            text_list = tab_text.split('\n')
           
            if x == 'W':
                with open(home + table_out + y + "_sens.txt", "w") as f:
                    f.write(text_list[0]  + "\n" + text_list[1] + "\n"  + text_list[2] +"\n")
                    f.close()
            
            with open(home + table_out + y + "_sens.txt", "a") as f:
                f.write("\midrule \n" + series_name[x] +"\\\\" + "\n\midrule\n") 
                f.write(text_list[4] + "\n"  + text_list[5] + "\n" + text_list[6] + "\n") 
                f.close()
            
            if x == 'Z':
                with open(home + table_out + y + "_sens.txt", "a") as f:
                    f.write("\\bottomrule \n \end{tabular}") 
                    f.close()

            if y == 'Mean':
                paras = ['I_K', 'L', 'alpha', 'N_high', 'SA_K', 'rho_I', 'delta1b', 'delta1a', 'SD_I']
                para_names = ['$E[I]/K$', '$\mathcal{L_{low}}$', '$\\alpha$', '$N_{high}$', '$SA_K$', '$\\rho_I$', '$\\delta_{1b}$', '$\\delta_{1a}$', '{$\\sqrt{\\Var[I]} \over E[I]$}']
                N = sum(np.ones(n)[SAMP])
                Xpara = np.zeros([N, 9])
                inn = 0
                for i in range(n):
                    if SAMP[i]:
                        pn = 0 
                        for p in paras:
                            Xpara[inn, pn] = result['paras'][i][p]
                            if p =='L':
                                Xpara[inn, pn] =  result['paras'][i][p] / result['paras'][i]['I_K']
                            if p =='delta1a':
                                Xpara[inn, pn] =  result['paras'][i][p] * 1000

                            pn = pn + 1
                        inn +=1

                tree = Tree(min_samples_leaf = 2, n_estimators = 150, n_jobs=4)
                tree.fit(Xpara, idx)
                rank = tree.feature_importances_ * 100

                data0 = []
                inn = 0
                for p in para_names:
                    record = {}
                    record['Importance'] = rank[inn]
                    data0.append(record)
                    inn = inn + 1
                tab = pandas.DataFrame(data0)
                tab.index = para_names
                tab = tab.sort(columns=['Importance'], ascending=False)
                tab_text = tab.to_latex(float_format =  '{:,.2f}'.format)
                with open(home + table_out + x + "sens_table.txt", "w") as f:
                    f.write(tab.to_latex(float_format =  '{:,.2f}'.format)) 
                    f.close()

                low = np.percentile(idx, 10)
                idx_low = idx < low
                x_low = np.zeros(9)
                for i in range(9):
                    x_low[i] = np.mean(Xpara[idx_low,i])
                x_low[1] = np.mean(Xpara[idx_low,1] * Xpara[idx_low,0])
                high = np.percentile(idx, 90)
                idx_high = idx > high
                x_high = np.zeros(9)
                for i in range(9):
                    x_high[i] = np.mean(Xpara[idx_high,i])
                x_high[1] = np.mean(Xpara[idx_high,1] * Xpara[idx_high,0])

                data0 = []
                inn = 0
                record = {}
                record['Importance'] = 1000
                record['$<$ 10th percentile'] = np.mean(idx[idx_low])
                record['Mean'] = np.mean(idx)
                record['$>$ 90th percentile'] = np.mean(idx[idx_high])
                data0.append(record)
                for p in paras:
                    record = {}
                    record['Importance'] = rank[inn]
                    record['$<$ 10th percentile'] = x_low[inn]
                    if p == 'L':
                        record['Mean'] = np.mean(Xpara[:,inn] * Xpara[:,0])
                    else:
                        record['Mean'] = np.mean(Xpara[:,inn])
                        
                    record['$>$ 90th percentile'] = x_high[inn]
                    data0.append(record)
                    inn = inn + 1
                tab = pandas.DataFrame(data0)
                tab.index = ['Index'] + para_names
                tab = tab.sort(columns=['Importance'], ascending=False)
                with open(home + table_out + x + "_sens_table.txt", "w") as f:
                    f.write(tab.to_latex(float_format =  '{:,.2f}'.format, columns=['Importance', 'Bottom 10%', 'Mean', 'Top 10%'])) 
                    f.close()
                
                for i in [0,3,8, 7]:
                    X = np.zeros([200, 9])
                    for j in range(9):
                        X[:,j] = np.ones(200) * np.mean(Xpara[:,j])
                    X[:,i] = np.linspace(np.min(Xpara[:,i]), np.max(Xpara[:,i]),200)

                    Y = tree.predict(X) 

                    data = [[X[:,i], Y]]

                    chart_data = {'OUTFILE' : (home + out + x + '_' + paras[i] +  img_ext),
                      'XLABEL' : '',
                      'YLABEL' : '', 
                      'YMIN' : 0.9, #np.min(idx), 
                      'YMAX' : 1, #np.max(idx), 
                      'XMIN' : np.min(X[:,i]),
                      'XMAX' : np.max(X[:,i])}
                    
                    build_chart(chart_data, data, chart_type='plot')
   
    print n
