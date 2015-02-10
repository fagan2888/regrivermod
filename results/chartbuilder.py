import numpy as np
import pandas
import pylab
import pickle
import pylab
from sklearn.ensemble import ExtraTreesClassifier as Tree_classifier
from sklearn.ensemble import ExtraTreesRegressor as Tree

def dataframe(npoints, nseries, names, index, X):
    data0 = []
    for i in range(npoints):
        record = {}
        for j in range(nseries):
            record[names[j]] = X[i, j] 
        data0.append(record)
    data = pandas.DataFrame(data0)
    data.index = index

    return data

def chart_params():

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
               'text.usetex': False,
               'figure.figsize': fig_size}

    pylab.rcParams.update(params)

def build_chart(chart, data_set, chart_type='plot', ticks = False, show=True, ylim=False, xlim=False, legend=False):

    chart_params()

    if chart_type == 'plot':
        pylab.figure()
        if legend:
            figs = [pylab.plot(series[0], series[1], label=series[2]) for series in data_set]
            #setFigLinesBW(fig[0])
            pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.) 
        else:
            [pylab.plot(series[0], series[1]) for series in data_set]
    elif chart_type == 'scatter':
        pylab.figure()
        [pylab.plot(series[0], series[1], 'o') for series in data_set] 
    elif chart_type == 'bar':
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
        if xlim:
            pylab.xlim(chart['XMIN'], chart['XMAX'])
        else:
            pylab.xlim(min(data_set.index), max(data_set.index))

    if not(chart_type == 'date' or chart_type == 'bar'):    
        if xlim:
            pylab.xlim(chart['XMIN'], chart['XMAX'])
    
    #if chart_type == 'bar':
        #pylab.xticks(chart['XMIN'] + np.arange(chart['XTICKS']) * chart['XSTEP'], chart['LABELS'])
    if ticks:
        pylab.xticks(chart['XMIN'] + np.arange(chart['XTICKS']) * chart['XSTEP'])
        pylab.yticks(chart['YMIN'] + np.arange(chart['YTICKS']) * chart['YSTEP'])
    
    pylab.xlabel(chart['XLABEL'])

    if ylim:
        pylab.ylim(chart['YMIN'], chart['YMAX'])

    if not (chart_type == 'hist' or chart_type =='barh'):
        pylab.ylabel(chart['YLABEL'])
    
    if chart_type == 'barh':
        pylab.barh(data_set[0], data_set[1], align='center')
        pylab.xlabel(chart['XLABEL'])
        pylab.yticks(data_set[0], chart['LABELS'])
    if show:
        pylab.savefig(chart['OUTFILE'], bbox_inches='tight')
        pylab.show()

def chart_ch6(SW, a, b, label, folder, FILE):
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

    home = '/home/nealbob'
    img_ext = '.pdf'

    pylab.figure()
    pylab.boxplot([SW['RS'], SW['CS-HL'], SW['RS-HL']], whis=5)
    pylab.axhline(y=1.0, color='0.5', linewidth=0.5, alpha=0.75, linestyle=':')
    pylab.ylim(a, b)
    pylab.ylabel(label)
    pylab.tick_params(axis='x', which = 'both', labelbottom='off')
    pylab.figtext(0.225, 0.06, 'RS', fontsize = 10)
    pylab.figtext(0.495, 0.06, 'CS-HL', fontsize = 10)
    pylab.figtext(0.76, 0.06, 'RS-HL', fontsize = 10)
    pylab.savefig(home + folder + FILE + img_ext)
    pylab.show()

def chart(SW, a, b, label, folder, FILE):
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

    home = '/home/nealbob'
    img_ext = '.pdf'

    pylab.figure()
    pylab.boxplot([SW['SWA'], SW['OA'], SW['NS']], whis=5)
    pylab.axhline(y=1.0, color='0.5', linewidth=0.5, alpha=0.75, linestyle=':')
    pylab.ylim(a, b)
    pylab.ylabel(label)
    pylab.tick_params(axis='x', which = 'both', labelbottom='off')
    pylab.figtext(0.225, 0.06, 'SWA', fontsize = 10)
    pylab.figtext(0.495, 0.06, 'OA', fontsize = 10)
    pylab.figtext(0.76, 0.06, 'NS', fontsize = 10)
    pylab.savefig(home + folder + FILE + img_ext)
    pylab.show()

def chart_ch7(SW, a, b, label, folder, FILE):
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

    home = '/home/nealbob'
    img_ext = '.pdf'

    pylab.figure()
    pylab.boxplot([SW['SWA'], SW['OA'], SW['CS-HL']], whis=5)
    pylab.axhline(y=1.0, color='0.5', linewidth=0.5, alpha=0.75, linestyle=':')
    pylab.ylim(a, b)
    pylab.ylabel(label)
    pylab.tick_params(axis='x', which = 'both', labelbottom='off')
    pylab.figtext(0.225, 0.06, 'SWA', fontsize = 10)
    pylab.figtext(0.495, 0.06, 'OA', fontsize = 10)
    pylab.figtext(0.76, 0.06, 'CS-HL', fontsize = 10)
    pylab.savefig(home + folder + FILE + img_ext)
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

    

        
