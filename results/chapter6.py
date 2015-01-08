from chartbuilder import *
import numpy as np
import pandas
import matplotlib.pyplot as pylab
import pickle
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree
import copy

def removekeys(d, keys):
    r = dict(d)
    for key in keys:
        del r[key]
    return r 

def tables(results, scenarios, Lambda, LambdaK, label='central', risk=False):

    """
        Generate charts and tables for central case scenarios
    """
    home = '/home/nealbob'
    folder = '/Dropbox/model/results/chapter6/'
    out = '/Dropbox/Thesis/IMG/chapter6/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/'

    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['SW', 'S', 'U_low', 'U_high', 'X_low', 'X_high', 'S_low', 'S_high', 'trade']
    scale = {'SW': 1000000, 'S': 1000, 'U_low': 1000000, 'U_high': 1000000, 'X_low': 1000, 'X_high': 1000,
             'S_low': 1, 'S_high': 1, 'trade': 1000}
    if risk:
        scale['SW'] = 1
        scale['U_low'] = 1
        scale['U_high'] = 1

    rows = ['Planner'] + scenarios
    m = len(results[scenarios[0]][series[0]][cols[0]]) - 1

    # Calculate user storage percentages
    for x in ['S_low', 'S_high']:
        for scen in scenarios:
            for col in cols:
                results[scen][x][col][:] /= results[scen][x]['Max'][m]

    for x in series:

        data0 = []
        # Planner
        record = {}
        for col in cols:
            record[col] = results[scenarios[0]][x][col][0] / scale[x]
        data0.append(record)

        # Policy scenarios
        for scen in scenarios:
            record = {}
            for col in cols:
                record[col] = results[scen][x][col][m] / scale[x]
            data0.append(record)
        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + label + ' ' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

    data = pandas.DataFrame([Lambda])

    with open(home + table_out + label + '_Lambda' + '.txt', 'w') as f:
        f.write(data.to_latex(float_format='{:,.2f}'.format))
        f.close()
    
    labels = ['CS', 'RS-O', 'RS-HL-O', 'CS-O', 'CS-HL-O', 'CS-U']
    values = [Lambda[l] for l in labels]
    labels[0] = 'Arbitrary'

    chart = {'OUTFILE': home + out + label + '_Lambdahigh' + img_ext,
             'XLABEL': '$\Lambda_{high}$',
             'LABELS': labels}
    data = [np.arange(len(labels)), values]
    build_chart(chart, data, chart_type='barh')

    data = pandas.DataFrame([LambdaK])

    with open(home + table_out + label + '_LambdaK' + '.txt', 'w') as f:
        f.write(data.to_latex(float_format='{:,.2f}'.format))
        f.close()

    # # Social welfare chart

    arbSW = np.array([results['RS']['SW']['Mean'][m] / scale['SW'],
                      results['RS-HL']['SW']['Mean'][m] / scale['SW'],
                      results['CS']['SW']['Mean'][m] / scale['SW'],
                      results['CS-HL']['SW']['Mean'][m] / scale['SW']
    ])

    optSW = np.array([results['RS-O']['SW']['Mean'][m] / scale['SW'],
                      results['RS-HL-O']['SW']['Mean'][m] / scale['SW'],
                      results['CS-O']['SW']['Mean'][m] / scale['SW'],
                      results['CS-HL-O']['SW']['Mean'][m] / scale['SW']
    ])

    ymin = 170
    ymax = 187
    if risk:
        ymin = 87
        ymax = 93

    chart = {'OUTFILE': home + out + label + '_SW' + img_ext,
             'YLABEL': 'Mean social welfare',
             'XLABEL': 'Policy scenario',
             'LABELS': ('', 'RS', '', 'RS-HL', '', 'CS', '', 'CS-HL'),
             'LEGEND': ('Arbitrary shares', 'Optimal shares'),
             'WIDTH': 0.3,
             'YMIN': ymin,
             'YMAX': ymax}
    data = [[np.arange(4), arbSW], [np.arange(4) + chart['WIDTH'], optSW]]
    build_chart(chart, data, chart_type='bar', ylim=True)

def tables2(results, scenarios, label='central'):

    """
        Generate charts and tables for central case scenarios
    """
    home = '/home/nealbob'
    folder = '/Dropbox/model/results/chapter6/'
    out = '/Dropbox/Thesis/IMG/chapter6/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/'

    cols = ['Mean', 'SD', '2.5th', '25th', '75th', '97.5th']
    series = ['S_low', 'S_high']
    rows = ['Planner'] + scenarios
    m = len(results[scenarios[0]][series[0]][cols[0]]) - 1

    for x in series:

        data0 = []
        # Planner
        record = {}
        for col in cols:
            record[col] = results[scenarios[0]][x][col][0]
        data0.append(record)

        # Policy scenarios
        for scen in scenarios:
            record = {}
            for col in cols:
                record[col] = results[scen][x][col][m]
            data0.append(record)
        data = pandas.DataFrame(data0)
        data.index = rows

        with open(home + table_out + label + ' ' + x + '.txt', 'w') as f:
            f.write(data.to_latex(float_format='{:,.2f}'.format, columns=cols))
            f.close()

def lambda_search(n=10):

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter6/lambda/'
    model = '/Dropbox/Model/'
    out = '/Dropbox/Thesis/IMG/chapter6/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/'
   
    results = []
    paras = []

    for i in range(n):
        if i != 9:
            with open(home + folder + 'lambda_result_' + str(i) +'.pkl', 'rb') as f:
                results.extend(pickle.load(f))
                f.close()

            with open(home + folder + 'lambda_para_' + str(i) + '.pkl', 'rb') as f:
                paras.extend(pickle.load(f))
                f.close()
    
    nn = (n - 1) * 10

    Y = np.zeros([nn, 4])
    X = np.zeros([nn, 12])

    for i in range(nn):
       
        Y[i, 0] = results[i][0][1][0]
        Y[i, 1] = results[i][0][1][1]
        Y[i, 2] = results[i][1][1][0]
        Y[i, 3] = results[i][1][1][1]
        
        X[i, :] = np.array([paras[i][p] for p in paras[i]])

        
    tree = Tree(min_samples_split=3, min_samples_leaf=2, n_estimators = 300)
    tree.fit(X, Y)
    
    with open(home + model + 'sharemodel.pkl', 'wb') as f:
       pickle.dump(tree, f)
       f.close()
    
    scen = ['RS-O', 'CS-O', 'RS-HL-O', 'CS-HL-O']

    for i in range(4):
    
        chart = {'OUTFILE' : (home + out + 'lambda_' + scen[i] + img_ext),
                 'XLABEL' : 'Optimal flow share',
                 'XMIN' : min(Y[:,i]),
                 'XMAX' : max(Y[:,i]),
                 'BINS' : 10}
        data = [Y[:,i]]
        build_chart(chart, data, chart_type='hist')

        chart = {'OUTFILE' : (home + out + 'lambda_scat_' + scen[i] + img_ext),
                 'XLABEL' : 'Number of high reliability users',
                 'YLABEL' : 'Optimal flow share'}
        data = [[X[:, 2], Y[:,i]]]
        build_chart(chart, data, chart_type='scatter')
    
    
    rank = tree.feature_importances_ * 100
    
    data0 = []
    for i in range(len(paras[0])):
        record = {}
        record['Importance'] = rank[i]
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = [p for p in paras[i]]
    tab = tab.sort(columns=['Importance'], ascending=False)
    
    with open(home + table_out + 'lambda' + '.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.2f}'.format))
        f.close()
    
    return [results, paras, Y, X]


def sens(n=10):

    home = '/home/nealbob'
    folder = '/Dropbox/Model/results/chapter6/sens/'
    model = '/Dropbox/Model/'
    out = '/Dropbox/Thesis/IMG/chapter6/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter6/'
   
    temp = []
    for i in range(1, n + 1):
        with open(home + folder + 'main_result_' + str(i) +'.pkl', 'rb') as f:
            temp.extend(pickle.load(f))
            f.close()
    n = len(temp)

    ### Convert to chapter 5 format
    results = []
    srlist = ['CS', 'RS', 'CS-HL', 'RS-HL']
    for i in range(n):
        res = {sr : 0 for sr in srlist}
        
        # CS
        res0 = {}
        res0['stats'] = [temp[i][0][0]]
        res0['paras'] = [temp[i][0][1]]
        res['CS'] = res0
        
        # RS
        res1 = {}
        res1['stats'] = [copy.copy(temp)[i][0][0]]
        res1['paras'] = [temp[i][0][1]]
        res['RS'] = res1
        
        # CS-HL
        res2 = {}
        res2['stats'] = [temp[i][1][0]]
        res2['paras'] = [temp[i][1][1]]
        res['CS-HL'] = res2
        
        # RS-HL
        res3 = {}
        res3['stats'] = [copy.copy(temp)[i][1][0]]
        res3['paras'] = [temp[i][1][1]]
        res['RS-HL'] = res3
        
        results.append(res)
    
    idx = {}   
    idx['CS'] = results[0]['CS']['stats'][0]['SW'].shape[0] - 1
    idx['RS'] = 1
    idx['CS-HL'] = results[0]['CS']['stats'][0]['SW'].shape[0] - 1
    idx['RS-HL'] = 1
    
    # Delete dud records

    nfull = len(results)
    delidx = []
    for i in range(nfull):
        n2 = idx['CS']
        if results[i]['CS']['stats'][0]['SW']['Mean'][n2] == float('Inf') or results[i]['CS-HL']['stats'][0]['SW']['Mean'][n2] == float('Inf'):
            delidx.append(i)
        n2 = idx['RS']
        if results[i]['RS']['stats'][0]['SW']['Mean'][n2] == float('Inf') or results[i]['RS-HL']['stats'][0]['SW']['Mean'][n2] == float('Inf'):
            delidx.append(i)
    results = [i for j, i in enumerate(results) if j not in delidx]
    n = len(results)
    print str(n) + ' good results of ' + str(nfull) + ' total runs.'

    series = []
    for s in results[0]['CS']['stats'][0]:
        if results[0]['CS']['stats'][0][s]['SD'][idx['CS']] > 0:
            series.append(s)
    
    stats = ['Mean', 'SD', '25th', '75th', '2.5th','97.5th', 'Min', 'Max']

    #===========================
    # Preferred scenario
    #============================
    
    data0 = []
    SW = {}
    for sr in srlist:
        array = np.zeros(n)
        for i in range(n):
            n2 = idx[sr]
            if results[i][sr]['paras'][0].para_list['risk'] == 0:
                array[i] = results[i][sr]['stats'][0]['SW'][n2]['Mean'] / 1000000
            else:
                array[i] = results[i][sr]['stats'][0]['SW'][n2]['Mean']

        SW[sr] = np.round(array, 3)

    SWmax = []
    for i in range(n):
        SWm = -1
        SWi = -1
        for sr in srlist:
            if SW[sr][i] > SWm:
                SWm = SW[sr][i]
                SWi = sr
            elif SW[sr][i] == SWm:
                SWm = SWm
                SWi = SWi + sr

        SWmax.append(SWi)

    Y = np.zeros(n)
    srnum = {'CS': 0, 'RS': 1, 'CS-HL': 2, 'RS-HL': 3}
    for sr in srlist:
        count = 0
        for z in range(n):
            if SWmax[z] == sr:
                count += 1
                Y[z] = srnum[sr]

        print sr
        print count

    paralist = results[0]['CS']['paras'][0].para_list  
    #temp = results[0]['CS']['paras'][0].para_list  
    #temp = removekeys(temp, ['sig_eta', 'rho_eps', 'delta0', 'LL', 'Lambda_high', 'Lambda_high_RS'])

    paralist = removekeys(paralist,  ['L',  'SA_K', 'Prop_high', 'Lambda_high_RS'])
    
    home = '/home/nealbob'
    model = '/Dropbox/Model/'
    #with open(home + model + 'sharemodel.pkl', 'rb') as f:
    #   tree = pickle.load(f)
    #   f.close()
    
    with open(home + model + 'results/chapter6/Yy.pkl', 'rb') as f:
       Yy = pickle.load(f)
       f.close()
    
    #m = len(results[i]['CS']['paras'][0].para_list) - 6
    #Xx = np.zeros([n, m])
    #for i in range(n):
    #    temp = results[i]['CS']['paras'][0].para_list  
    #    temp = removekeys(temp, ['sig_eta', 'rho_eps', 'delta0', 'LL', 'Lambda_high', 'Lambda_high_RS'])
    #    Xx[i,:] = np.array([temp[p] for p in temp])
    
    #home = '/home/nealbob'
    #model = '/Dropbox/Model/results/chapter6/'
    #with open(home + model + 'xX.pkl', 'wb') as f:
    #   pickle.dump(Xx, f)
    #   f.close()
    
    #Yy = tree.predict(Xx)
    
    for i in range(n):

        results[i]['CS']['paras'][0].para_list['Lambda_high'] = results[i]['CS']['paras'][0].para_list['Lambda_high']  - Yy[i,1] #/ results[i]['CS']['paras'][0].para_list['Prop_high'] 
        results[i]['CS']['paras'][0].para_list['Lambda_high_RS'] = results[i]['CS']['paras'][0].para_list['Lambda_high_RS'] - Yy[i,0] #/ results[i]['CS']['paras'][0].para_list['Prop_high'] 
        results[i]['CS-HL']['paras'][0].para_list['Lambda_high'] = results[i]['CS-HL']['paras'][0].para_list['Lambda_high'] - Yy[i,3] #/ results[i]['CS-HL']['paras'][0].para_list['Prop_high'] #/  
        results[i]['CS-HL']['paras'][0].para_list['Lambda_high_RS'] = results[i]['CS-HL']['paras'][0].para_list['Lambda_high_RS']  - Yy[i,2] #/ results[i]['CS-HL']['paras'][0].para_list['Prop_high'] #/ 

    pnum = len(paralist)
    paras = [i for i in paralist]
    print paras

    para_names = ['$\delta0$', '$E[I]/K$', '$\bar \pi \phi$', '$\rho_e$',  '$\alpha$',  '$\sigma_{\eta}$', '$n_{high}$',
     '$\Lambda_{high}^CS -\hat \Lambda_{high}^{CS} $', '$\rho_I$', '$\delta_{1b}$', '$\tau$', '$\delta_{1a}$', '$c_v$', '${\aA_{low} \over E[I]/K}$', '$\Lambda_{high}^{RS} - \hat \Lambda_{high}^{RS}$', '$\Lambda_{high}^{CS-HL} - \hat \Lambda_{high}^{CS-HL}$', '$\Lambda_{high}^{RS-HL} - \hat \Lambda_{high}^{RS-HL}$']
    print len(para_names)
    

    pnum = pnum + 3
    Xpara = np.zeros([n, pnum])
    for i in range(n):
        pn = 0
        for p in paras:
            Xpara[i, pn] = results[i]['CS']['paras'][0].para_list[p]
            if p == 'LL':
                Xpara[i, pn] = results[i]['CS']['paras'][0].para_list['LL'] / results[i]['CS']['paras'][0].para_list['I_K']
            pn = pn + 1
        
        Xpara[i, pn] = results[i]['CS']['paras'][0].para_list['Lambda_high_RS']
        pn = pn + 1
        Xpara[i, pn] = results[i]['CS-HL']['paras'][0].para_list['Lambda_high']
        pn = pn + 1
        Xpara[i, pn] = results[i]['CS-HL']['paras'][0].para_list['Lambda_high_RS']
        pn = pn + 1
    
    #===========================
    # Classifier
    #============================
    
    treec = Tree_classifier(n_estimators=500, n_jobs=4) #min_samples_split=3, min_samples_leaf=2)
    treec.fit(Xpara, Y)
    rank = treec.feature_importances_ * 100
    print rank
    
    data0 = []
    inn = 0
    for p in para_names:
        record = {}
        record['Importance'] = rank[inn]
        record['CS'] = np.mean(Xpara[np.where(Y == 0), inn])
        record['RS'] = np.mean(Xpara[np.where(Y == 1), inn])
        record['CS-HL'] = np.mean(Xpara[np.where(Y == 2), inn])
        record['RS-HL'] = np.mean(Xpara[np.where(Y == 3), inn])
        data0.append(record)
        inn = inn + 1

    tab = pandas.DataFrame(data0)
    tab.index = para_names
    tab = tab.sort(columns=['Importance'], ascending=False)
    tab_text = tab.to_latex(float_format='{:,.2f}'.format, escape=False)
    
    with open(home + table_out + 'classifier_table.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.2f}'.format, escape=False, columns=['Importance', 'CS', 'RS', 'CS-HL', 'RS-HL']))
        f.close()

    pylab.ioff()
    fig_width_pt = 350
    inches_per_pt = 1.0 / 72.27
    golden_mean = 1.2360679774997898 / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
     'axes.labelsize': 10,
     'text.fontsize': 10,
     'legend.fontsize': 10,
     'xtick.labelsize': 8,
     'ytick.labelsize': 8,
     'text.usetex': True,
     'figure.figsize': fig_size}
    pylab.rcParams.update(params)
    plot_colors = 'rybg'
    cmap = pylab.cm.RdYlBu
    
    (xx, yy,) = np.meshgrid(np.arange(min(Xpara[:, 1]), max(Xpara[:, 1]), 0.02), np.arange(min(Xpara[:, 6]), max(Xpara[:, 6]), 1))

    nnn = xx.ravel().shape[0]
    
    Xlist = [np.mean(Xpara[:,i])*np.ones(nnn) for i in range(pnum)]
    Xlist[2] = np.zeros(nnn)
    Xlist[1] = xx.ravel()
    Xlist[6] = yy.ravel()
    X = np.array(Xlist).T

    Z = treec.predict(X).reshape(xx.shape)
    fig = pylab.contourf(xx, yy, Z, [0, 0.9999, 1.9999, 2.9999, 3.9999], colors=('red', 'yellow', 'blue', 'green'), alpha=0.5, antialiased=False, extend='both')
    for (i, c,) in zip(xrange(4), plot_colors):
        idx0 = np.where(Y == i)
        pylab.scatter(Xpara[idx0, 1], Xpara[idx0, 6], c=c, cmap=cmap, label=srlist[i], s = 12, lw=0.5 )
        pylab.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, ncol=4, mode='expand', borderaxespad=0.0)

    pylab.xlabel('Mean inflow over capacity')
    pylab.ylabel('Number of high reliability users')
    OUT = home + out + 'class_fig.pdf'
    pylab.savefig(OUT, bbox_inches='tight')
    pylab.show()
    
    Xlist = [np.mean(Xpara[:,i])*np.ones(nnn) for i in range(pnum)]
    Xlist[2] = np.ones(nnn) * 1
    Xlist[1] = xx.ravel()
    Xlist[6] = yy.ravel()
    X = np.array(Xlist).T

    Z = treec.predict(X).reshape(xx.shape)
    fig = pylab.contourf(xx, yy, Z, [0, 0.9999, 1.9999, 2.9999, 3.9999], colors=('red', 'yellow', 'blue', 'green'), alpha=0.5, antialiased=False, extend='both')
    for (i, c,) in zip(xrange(4), plot_colors):
        idx0 = np.where(Y == i)
        pylab.scatter(Xpara[idx0, 1], Xpara[idx0, 6], c=c, cmap=cmap, label=srlist[i], s=12, lw=0.5)
        pylab.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, ncol=4, mode='expand', borderaxespad=0.0)

    pylab.xlabel('Mean inflow over capacity')
    pylab.ylabel('Number of high reliability users')
    OUT = home + out + 'class_fig2.pdf'
    pylab.savefig(OUT, bbox_inches='tight')
    pylab.show()
    
    #===========================
    # Welfare
    #============================
    
    data0 = []
    miny = 10
    maxy = -10
    SW = {}
    SWI = {}
    for sr in srlist:
        n2 = idx[sr]
        array = np.zeros(n)
        arrayI = np.zeros(n)
        for i in range(n):
            array[i] = results[i][sr]['stats'][0]['SW'][n2]['Mean']
            arrayI[i] = results[i][sr]['stats'][0]['SW'][n2]['Mean'] / results[i]['CS']['stats'][0]['SW'][idx['CS']]['Mean'] 

        SW[sr] = array
        SWI[sr] = arrayI
        mn = np.min(arrayI)
        mx = np.max(arrayI)
        if mn < miny:
            miny = mn
        if mx > maxy:
            maxy = mx

    SW_p = np.zeros(n)
    for i in range(n):
        SW_p[i] = results[i][sr]['stats'][0]['SW'][0]['Mean']
    
    chart_ch6(SWI, 0.99 * miny, 1.01 * maxy, 'Social welfare relative to CS', out, 'Welfare_sens')
    
    data0 = []
    for x in srlist:
        record = {}
        record['Mean'] = np.mean(SW[x])
        record['Min'] = np.min(SW[x])
        record['Q1'] = np.percentile(SW[x], 25)
        record['Q3'] = np.percentile(SW[x], 75)
        record['Max'] = np.max(SW[x])
        data0.append(record)

    record = {}
    record['Mean'] = np.mean(SW_p)
    record['Min'] = np.min(SW_p)
    record['Q1'] = np.percentile(SW_p, 25)
    record['Q3'] = np.percentile(SW_p, 75)
    record['Max'] = np.max(SW_p)
    data0.append(record)
    tab = pandas.DataFrame(data0)
    tab.index = srlist + ['Planner']
    
    with open(home + table_out + 'welfare_sens.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.2f}'.format, columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'], escape=False))
        f.close()
    
    data0 = []
    for x in srlist:
        record = {}
        record['Mean'] = np.mean(SWI[x])
        record['Min'] = np.min(SWI[x])
        record['Q1'] = np.percentile(SWI[x], 25)
        record['Q3'] = np.percentile(SWI[x], 75)
        record['Max'] = np.max(SWI[x])
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = srlist
    with open(home + table_out + 'welfareI_sens.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.4f}'.format, columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'], escape=False))
        f.close()
    
    #=============================
    # Regressor
    #==============================

    Y = np.array([SWI['CS'], SWI['RS'], SWI['CS-HL'], SWI['RS-HL']]).T
    
    tree = Tree(n_estimators=500, n_jobs=4)
    tree.fit(Xpara, Y)
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
    tab_text = tab.to_latex(float_format='{:,.2f}'.format, escape=False)
    print tab_text 
    with open(home + table_out + 'importance.txt', 'w') as f:
        f.write(tab_text)
        f.close()
    paras.append('Lambda_high_RS')
    paras.append('Lambda_high_CSHL')
    paras.append('Lambda_high_RSHL')
    for i in range(pnum):
        X = np.zeros([200, pnum])
        for j in range(pnum):
            X[:, j] = np.ones(200) * np.mean(Xpara[:, j])

        X[:, i] = np.linspace(np.min(Xpara[:, i]), np.max(Xpara[:, i]), 200)
        Y = tree.predict(X)
        data = [[X[:, i], Y]]
        data0 = []
        for k in range(200):
            record = {}
            record['CS'] = Y[k, 0]
            record['RS'] = Y[k, 1]
            record['CS-HL'] = Y[k, 2]
            record['RS-HL'] = Y[k, 3]
            data0.append(record)

        data = pandas.DataFrame(data0)
        data.index = X[:, i]
        chart_data = {'OUTFILE': home + out + 'SW_' + paras[i] + img_ext,
         'XLABEL': '',
         'YLABEL': '',
         'YMIN': 0.97,
         'YMAX': 1.01}
        print paras[i]
        build_chart(chart_data, data, chart_type='date', ylim=True)

    #===========================
    # Storage
    #=============================

    miny = 10
    maxy = -10
    S = {}
    SI = {}
    for sr in srlist:
        array = np.zeros(n)
        arrayI = np.zeros(n)
        for i in range(n):
            n2 = idx[sr] 
            arrayI[i] = results[i][sr]['stats'][0]['S'][n2]['Mean'] / (results[i]['CS']['stats'][0]['S'][idx['CS']]['Mean'])
            array[i] = results[i][sr]['stats'][0]['S'][n2]['Mean'] / 1000

        S[sr] = array
        SI[sr] = arrayI
        mn = np.min(arrayI)
        mx = np.max(arrayI)
        if mn < miny:
            miny = mn
        if mx > maxy:
            maxy = mx

    S_p = np.zeros(n)
    for i in range(n):
        S_p[i] = results[i][sr]['stats'][0]['S'][0]['Mean'] / 1000

    chart_ch6(SI, 0.99 * miny, 1.01 * maxy, 'Storage relative to CS', out, 'Storage_sens')

    data0 = []
    for x in srlist:
        record = {}
        record['Mean'] = np.mean(S[x])
        record['Min'] = np.min(S[x])
        record['Q1'] = np.percentile(S[x], 25)
        record['Q3'] = np.percentile(S[x], 75)
        record['Max'] = np.max(S[x])
        data0.append(record)

    record = {}
    record['Mean'] = np.mean(S_p)
    record['Min'] = np.min(S_p)
    record['Q1'] = np.percentile(S_p, 25)
    record['Q3'] = np.percentile(S_p, 75)
    record['Max'] = np.max(S_p)
    data0.append(record)
    tab = pandas.DataFrame(data0)
    tab.index = srlist + ['Planner']

    with open(home + table_out + 'storage_sens.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.2f}'.format, columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'], escape=False))
        f.close()
    
    data0 = []
    for x in srlist:
        record = {}
        record['Mean'] = np.mean(SI[x])
        record['Min'] = np.min(SI[x])
        record['Q1'] = np.percentile(SI[x], 25)
        record['Q3'] = np.percentile(SI[x], 75)
        record['Max'] = np.max(SI[x])
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = srlist
    with open(home + table_out + 'storageI_sens.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.3f}'.format, columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'], escape=False))
        f.close()

    #===========================
    # low_welfare
    #=============================

    miny = 10
    maxy = -10
    S = {}
    SI = {}
    for sr in srlist:
        array = np.zeros(n)
        arrayI = np.zeros(n)
        for i in range(n):
            n2 = idx[sr] 
            arrayI[i] = results[i][sr]['stats'][0]['U_low'][n2]['Mean'] / 1000 / (results[i]['CS']['stats'][0]['U_low'][idx['CS']]['Mean'] / 1000)
            array[i] = results[i][sr]['stats'][0]['U_low'][n2]['Mean'] / 1000

        S[sr] = array
        SI[sr] = arrayI
        mn = np.min(arrayI)
        mx = np.max(arrayI)
        if mn < miny:
            miny = mn
        if mx > maxy:
            maxy = mx

    S_p = np.zeros(n)
    for i in range(n):
        S_p[i] = results[i][sr]['stats'][0]['U_low'][0]['Mean'] / 1000

    chart_ch6(SI, 0.99 * miny, 1.01 * maxy, 'Low reliability profit relative to CS', out, 'lowwelfare_sens')

    data0 = []
    for x in srlist:
        record = {}
        record['Mean'] = np.mean(S[x])
        record['Min'] = np.min(S[x])
        record['Q1'] = np.percentile(S[x], 25)
        record['Q3'] = np.percentile(S[x], 75)
        record['Max'] = np.max(S[x])
        data0.append(record)

    record = {}
    record['Mean'] = np.mean(S_p)
    record['Min'] = np.min(S_p)
    record['Q1'] = np.percentile(S_p, 25)
    record['Q3'] = np.percentile(S_p, 75)
    record['Max'] = np.max(S_p)
    data0.append(record)
    tab = pandas.DataFrame(data0)
    tab.index = srlist + ['Planner']

    with open(home + table_out + 'low_welfare_sens.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.2f}'.format, columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'], escape=False))
        f.close()
    
    data0 = []
    for x in srlist:
        record = {}
        record['Mean'] = np.mean(SI[x])
        record['Min'] = np.min(SI[x])
        record['Q1'] = np.percentile(SI[x], 25)
        record['Q3'] = np.percentile(SI[x], 75)
        record['Max'] = np.max(SI[x])
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = srlist
    with open(home + table_out + 'low_welfare_I_sens.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.3f}'.format, columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'], escape=False))
        f.close()

    #===========================
    # high_welfare
    #=============================

    miny = 10
    maxy = -10
    S = {}
    SI = {}
    for sr in srlist:
        array = np.zeros(n)
        arrayI = np.zeros(n)
        for i in range(n):
            n2 = idx[sr] 
            arrayI[i] = results[i][sr]['stats'][0]['U_high'][n2]['Mean'] / 1000 / (results[i]['CS']['stats'][0]['U_high'][idx['CS']]['Mean'] / 1000)
            array[i] = results[i][sr]['stats'][0]['U_high'][n2]['Mean'] / 1000

        S[sr] = array
        SI[sr] = arrayI
        mn = np.min(arrayI)
        mx = np.max(arrayI)
        if mn < miny:
            miny = mn
        if mx > maxy:
            maxy = mx

    S_p = np.zeros(n)
    for i in range(n):
        S_p[i] = results[i][sr]['stats'][0]['U_high'][0]['Mean'] / 1000

    chart_ch6(SI, 0.99 * miny, 1.01 *maxy, 'High reliability profit relative to CS', out, 'highwelfare_sens')

    data0 = []
    for x in srlist:
        record = {}
        record['Mean'] = np.mean(S[x])
        record['Min'] = np.min(S[x])
        record['Q1'] = np.percentile(S[x], 25)
        record['Q3'] = np.percentile(S[x], 75)
        record['Max'] = np.max(S[x])
        data0.append(record)

    record = {}
    record['Mean'] = np.mean(S_p)
    record['Min'] = np.min(S_p)
    record['Q1'] = np.percentile(S_p, 25)
    record['Q3'] = np.percentile(S_p, 75)
    record['Max'] = np.max(S_p)
    data0.append(record)
    tab = pandas.DataFrame(data0)
    tab.index = srlist + ['Planner']

    with open(home + table_out + 'high_welfare_sens.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.2f}'.format, columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'], escape=False))
        f.close()
    
    data0 = []
    for x in srlist:
        record = {}
        record['Mean'] = np.mean(SI[x])
        record['Min'] = np.min(SI[x])
        record['Q1'] = np.percentile(SI[x], 25)
        record['Q3'] = np.percentile(SI[x], 75)
        record['Max'] = np.max(SI[x])
        data0.append(record)

    tab = pandas.DataFrame(data0)
    tab.index = srlist
    with open(home + table_out + 'high_welfare_I_sens.txt', 'w') as f:
        f.write(tab.to_latex(float_format='{:,.3f}'.format, columns=['Mean', 'Min', 'Q1', 'Q3', 'Max'], escape=False))
        f.close()

    return Z, Xpara
