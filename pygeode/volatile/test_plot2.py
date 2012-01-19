# Test quivers, pseudo-color plots, and a map projection.

from plot_wrapper import Overlay, Multiplot, Colorbar, QuiverKey, load
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

theplot = Map(theplot, projection='cyl')


# Save and re-load the plot
theplot.save('myplot.dat')
theplot = load('myplot.dat')

theplot.render()

from matplotlib.pyplot import show
show()
