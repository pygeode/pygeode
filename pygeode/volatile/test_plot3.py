# Test 'spaghetti' plots

from plot_wrapper import Overlay, Multiplot, load
from plot_shortcuts import spaghetti
from pygeode.data.ccmval2 import ref2

temp = ref2.TEMP(year=2000,month=6,day=1,hour=0)


axes_args = dict(ylabel='Pressure (bar)', xlabel='Temperature (K)')

# Overlay temperature profiles (of all ensemble members)
toronto = spaghetti(temp(lat=43.72, lon=280.66), 'zaxis', title='Toronto', **axes_args)
alert = spaghetti(temp(lat=82.50, lon=297.66), 'zaxis', title='Alert', **axes_args)

theplot = Multiplot([[toronto,alert]])


# Save and re-load the plot
theplot.save('myplot.dat')
theplot = load('myplot.dat')

theplot.render()

from matplotlib.pyplot import show
show()
