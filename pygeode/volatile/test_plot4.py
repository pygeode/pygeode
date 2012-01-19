# Test out plot legends

from plot_shortcuts import plot
from plot_wrapper import Legend, Overlay
from pygeode.tutorial import t1

t = t1.Temp

# Approach 1 - building up from individual line plots, each with their own labels
p1 = plot (t(lat=-30), label='30S')
p2 = plot (t(lat=  0), label='EQ')
p3 = plot (t(lat= 30), label='30N')
myplot = Overlay(p1,p2,p3, title='Temperature')
myplot = Legend(myplot)
myplot.render()

# Approach 2 - single line plot, labels added directly to the legend
myplot = plot (t(lat=-30), t(lat=0), t(lat=30), title='Temperature')
myplot = Legend(myplot, ['30S','EQ','30N'])
myplot.render()

from matplotlib.pyplot import show
show()
