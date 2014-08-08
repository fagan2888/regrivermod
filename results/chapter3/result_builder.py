import numpy as np
import pandas
import pylab
import pickle
import cloud
import pylab
import sklearn

def build_chart(chart, data_set, chart_type='plot', ticks = False, show=True, ylim=False):

    
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
        #pylab.figure()
        fig, ax = pylab.subplots()   
        one = ax.bar(data_set[0][0], data_set[0][1], chart['WIDTH'], color='k')
        two = ax.bar(data_set[1][0], data_set[1][1], chart['WIDTH'], color='w')
        ax.set_ylabel(chart['YLABEL'])
        ax.set_xticklabels(chart['LABELS'])
        ax.legend((one[0], two[0]), chart['LEGEND'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
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
        fig = data_set.plot()
        setFigLinesBW(fig)
        fig.grid(False)
        fig.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
        #fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fancybox=True)
        pylab.xlim(min(data_set.index), max(data_set.index))

    if not(chart_type == 'date' or chart_type == 'bar'):    
        pylab.xlim(chart['XMIN'], chart['XMAX'])
    
    #if chart_type == 'bar':
        #pylab.xticks(chart['XMIN'] + np.arange(chart['XTICKS']) * chart['XSTEP'], chart['LABELS'])
    if ticks:
        pylab.xticks(chart['XMIN'] + np.arange(chart['XTICKS']) * chart['XSTEP'])
        pylab.yticks(chart['YMIN'] + np.arange(chart['YTICKS']) * chart['YSTEP'])
    
    pylab.xlabel(chart['XLABEL'])
    if not (chart_type == 'hist' or chart_type == 'date' or chart_type == 'bar'):
        pylab.ylim(chart['YMIN'], chart['YMAX'])
    if ylim:
        pylab.ylim(chart['YMIN'], chart['YMAX'])

    if not (chart_type == 'hist'):
        pylab.ylabel(chart['YLABEL'])
 
    if show:
        pylab.savefig(chart['OUTFILE'], bbox_inches='tight')
        pylab.show()

def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    COLORMAP = {
        'b': {'marker': None, 'dash': (None,None)},
        'g': {'marker': None, 'dash': [5,5]},
        'r': {'marker': None, 'dash': [5,3,1,3]},
        'c': {'marker': None, 'dash': [1,3]},
        'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
        'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }

    for line in ax.get_lines():# + ax.get_legend().get_lines():
        origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    setAxLinesBW(fig.get_axes())

def setFigLinesBW_list(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    axlist = fig.get_axes()
    [setAxLinesBW(x) for x in axlist]


def build(n=1000):

    home = '/home/nealbob'
    out = '/Dropbox/Thesis/IMG/chapter3/'
    img_ext = '.pdf'
    table_out = '/Dropbox/Thesis/STATS/chapter3/'       
    
    f = open('result.pkl', 'rb')
    result = pickle.load(f)

    a = np.zeros(n)
    a0 = np.zeros(n)
    aI = np.zeros(n)
    for i in range(n):
        a0[i] = (result['stats'][i]['SW']['Mean'][0] / 1000000) 
        a[i] = (result['stats'][i]['SW']['Mean'][1] / 1000000) 
    
    aI = a / a0
    aI0 = a0 / a0
   
    ix = aI < 2

    SW = {'Planner' : a[ix], 'Myopic' : a0[ix]}
    SWI = {'Planner' : aI[ix], 'Myopic' : aI0[ix]}

    data0 = []
    
    for x in SW:
        record = {}
        record['Mean'] = np.mean(SW[x])
        record['Min'] = np.min(SW[x])
        record['Q1'] = np.percentile(SW[x], 25)
        record['Q3'] = np.percentile(SW[x], 75)
        record['Max'] = np.max(SW[x])
        data0.append(record)
    
    tab = pandas.DataFrame(data0)
    tab.index = ['Planner', 'Myopic']
    with open(home + table_out +"welfare.txt", "w") as f:
        f.write(tab.to_latex(float_format =  '{:,.2f}'.format , columns=['Mean', 'Min', 'Q1', 'Q3', 'Max']) )
        f.close()

    data0 = []
    
    for x in SWI:
        record = {}
        record['Mean'] = np.mean(SWI[x])
        record['Min'] = np.min(SWI[x])
        record['Q1'] = np.percentile(SWI[x], 25)
        record['Q3'] = np.percentile(SWI[x], 75)
        record['Max'] = np.max(SWI[x])
        data0.append(record)
    
    tab = pandas.DataFrame(data0)
    tab.index = ['Planner', 'Myopic']
    with open(home + table_out +"welfareI.txt", "w") as f:
        f.write(tab.to_latex(float_format =  '{:,.2f}'.format , columns=['Mean', 'Min', 'Q1', 'Q3', 'Max']) )
        f.close()
   
    ################################## Storage

    a = np.zeros(n)
    a0 = np.zeros(n)
    aI = np.zeros(n)
    for i in range(n):
        a0[i] = (result['stats'][i]['S']['Mean'][0] / 1000) 
        a[i] = (result['stats'][i]['S']['Mean'][1] / 1000) 
    
    aI = a / a0
    aI0 = a0 / a0
   
    S = {'Planner' : a[ix], 'Myopic' : a0[ix]}
    SI = {'Planner' : aI[ix], 'Myopic' : aI0[ix]}

    data0 = []
    
    for x in S:
        record = {}
        record['Mean'] = np.mean(S[x])
        record['Min'] = np.min(S[x])
        record['Q1'] = np.percentile(S[x], 25)
        record['Q3'] = np.percentile(S[x], 75)
        record['Max'] = np.max(S[x])
        data0.append(record)
    
    tab = pandas.DataFrame(data0)
    tab.index = ['Planner', 'Myopic']
    with open(home + table_out +"storage.txt", "w") as f:
        f.write(tab.to_latex(float_format =  '{:,.2f}'.format , columns=['Mean', 'Min', 'Q1', 'Q3', 'Max']) )
        f.close()

    data0 = []
    
    for x in SI:
        record = {}
        record['Mean'] = np.mean(SI[x])
        record['Min'] = np.min(SI[x])
        record['Q1'] = np.percentile(SI[x], 25)
        record['Q3'] = np.percentile(SI[x], 75)
        record['Max'] = np.max(SI[x])
        data0.append(record)
    
    tab = pandas.DataFrame(data0)
    tab.index = ['Planner', 'Myopic']
    with open(home + table_out +"storageI.txt", "w") as f:
        f.write(tab.to_latex(float_format =  '{:,.2f}'.format , columns=['Mean', 'Min', 'Q1', 'Q3', 'Max']) )
        f.close()
    
    ################################## W / Storage

    a = np.zeros(n)
    a0 = np.zeros(n)
    aI = np.zeros(n)
    for i in range(n):
        a0[i] = (result['stats'][i]['W']['Mean'][0] / 1000) / (result['stats'][i]['S']['Mean'][0] / 1000)  
        a[i] = (result['stats'][i]['W']['Mean'][1] / 1000) / (result['stats'][i]['S']['Mean'][1] / 1000) 
    
    aI = a / a0
    aI0 = a0 / a0
   
    S = {'Planner' : a[ix], 'Myopic' : a0[ix]}
    SI = {'Planner' : aI[ix], 'Myopic' : aI0[ix]}

    data0 = []
    
    for x in S:
        record = {}
        record['Mean'] = np.mean(S[x])
        record['Min'] = np.min(S[x])
        record['Q1'] = np.percentile(S[x], 25)
        record['Q3'] = np.percentile(S[x], 75)
        record['Max'] = np.max(S[x])
        data0.append(record)
    
    tab = pandas.DataFrame(data0)
    tab.index = ['Planner', 'Myopic']
    with open(home + table_out +"w_storage.txt", "w") as f:
        f.write(tab.to_latex(float_format =  '{:,.2f}'.format , columns=['Mean', 'Min', 'Q1', 'Q3', 'Max']) )
        f.close()

    data0 = []
    
    for x in SI:
        record = {}
        record['Mean'] = np.mean(SI[x])
        record['Min'] = np.min(SI[x])
        record['Q1'] = np.percentile(SI[x], 25)
        record['Q3'] = np.percentile(SI[x], 75)
        record['Max'] = np.max(SI[x])
        data0.append(record)
    
    tab = pandas.DataFrame(data0)
    tab.index = ['Planner', 'Myopic']
    with open(home + table_out +"w_storageI.txt", "w") as f:
        f.write(tab.to_latex(float_format =  '{:,.2f}'.format , columns=['Mean', 'Min', 'Q1', 'Q3', 'Max']) )
        f.close()

    ##################### Regression


    X = np.zeros([n, 10])
    Y = np.zeros(n)


    for i in range(n):
        X[i, 0] = result['paras'][i]['I_K']  
        X[i, 1] = result['paras'][i]['SD_I']  
        X[i, 2] = result['paras'][i]['N_high']  
        X[i, 3] = result['paras'][i]['rho_I']  
        X[i, 4] = result['paras'][i]['t_cost']  
        X[i, 5] = result['paras'][i]['SA_K']  
        X[i, 6] = result['paras'][i]['alpha']  
        X[i, 7] = result['paras'][i]['delta1a']  
        X[i, 8] = result['paras'][i]['delta1b']  
        X[i, 9] = result['paras'][i]['L']  

        #Y[i] = (result['stats'][i]['W']['Mean'][0]) / (result['stats'][i]['S']['Mean'][0]) 
        Y[i] = (result['stats'][i]['SW']['Mean'][0]) / (result['stats'][i]['SW']['Mean'][1]) 

    tree = sklearn.ensemble.ExtraTreesRegressor(500, min_samples_leaf=5)
    tree.fit(X, Y)
    
    r = tree.feature_importances_ 
    
    rank = {'I_K' :  r[0], 'SD_I' : r[1], 'N_high' : r[2], 'rho_I' : r[3], 't_cost' : r[4], 'SA_K' : r[5], 'alpha' : r[6], 'delta1a' : r[7], 'delta1b' : r[8], 'L' : r[9]}

    data0 = [rank]
    tab = pandas.DataFrame(data0)
    tab = tab.transpose()
    tab.columns = ['W / S']
    tab = tab.sort(ascending=False, columns = ['W / S'])
    with open(home + table_out +"rank.txt", "w") as f:
        f.write(tab.to_latex(float_format =  '{:,.3f}'.format ))
        f.close()
    
    name = ['Inflow to capacity, $E[I_t]/K$', 'Coefficient of variation', 'Number of high users, $n_high$', 'Inflow autocorrelation, $rho_I$', 'Transaction cost, $\tau$', 'Surface area to capacity', 'Evaporation loss, $\alpha$', 'Fixed delivery loss, $\delta_{1a}$', 'Variable delivery loss, $\delta_{1b}', 'Land, $L_{low}$']
    
    
    M = X.shape[1]
    for k in range(M):
        xk = np.linspace(np.min(X[:, k]), np.max(X[:, k]), 50)
        Xhat = np.ones([50, M])
        for i in range(M):
            Xhat[:, i] = np.mean(X[:,i])    
         
        Xhat[:, k] = xk
        Yhat = tree.predict(Xhat)
        
        data = [[xk, Yhat]] 
        chart = {'OUTFILE' : (home + out + name[k] + img_ext),
          'XLABEL' : name[k],
          'YLABEL' : 'Mean social welfare index',
          'XMIN' : min(xk),
          'XMAX' : max(xk),
          'YMIN' : 0.95,
          'YMAX' : 1.15}

        build_chart(chart, data, chart_type='plot')

    return [tree, tab] #[X[ix,:], Y[ix]]
