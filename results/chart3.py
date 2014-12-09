import pylab

def chart(idx, a, b, label, FILE):
    pylab.ioff()
    fig_width_pt = 350 					     # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27                # Convert pt to inch
    golden_mean = ((5**0.5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt   # width in inches   
    fig_height = fig_width*golden_mean       # height in inches
    fig_size =  [fig_width*0.42,fig_height]

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
    folder = '/Dropbox/Thesis/IMG/chapter3/'
    img_ext = '.pdf'

    pylab.figure()
    pylab.boxplot(idx, whis=100)
    pylab.ylim(a, b)
    #pylab.ylabel(label)
    pylab.tick_params(axis='x', which = 'both', labelbottom='off')
    pylab.savefig(home + folder + FILE + img_ext)
    pylab.show()
