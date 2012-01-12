import pickle
from plot_wrapper import Overlay, Multiplot, Colorbar, QuiverKey
from plot_shortcuts import pcolor, contourf, quiver, Map

from pygeode.data.ccmval2 import ref2

data = ref2(year=2000,month=1,day=1,hour=0)

temp = pcolor(data.TEMP(eta=1,ensemble=0))
temp = Colorbar(temp)

u = data.U(eta=1,ensemble=0)
v = data.V(eta=1,ensemble=0)

winds = quiver (u, v)
winds = QuiverKey (winds, 1.25, 0.9, 10, r'$10 \frac{m}{s}$')

theplot = Overlay(temp, winds, title="Winds and Temperature")

theplot = Map(theplot, projection='cyl', llcrnrlon=0, urcrnrlon=360, llcrnrlat=-90,urcrnrlat=90, xlabel='', ylabel='')


# Save and re-load the plot
outfile = open('myplot.pickle','w')
pickle.dump(theplot, outfile)
outfile.close()

infile = open('myplot.pickle','ro')
theplot = pickle.load(infile)
infile.close()


theplot.render()

from matplotlib.pyplot import show
show()
