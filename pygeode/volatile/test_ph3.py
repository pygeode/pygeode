
import plot_wr_ph as pl, plot_sc_ph as pygpl
import numpy as np
import pylab as pyl
import pygeode as pyg
from pygeode.tutorial import t1

reload(pl)
reload(pygpl)

t = pyg.ModelTime365(np.arange(100), units='days', startdate=dict(year=2))
x = pyg.cos(2 * np.pi * t / 30.)
x = x.rename('Ts')

Ax1 = pygpl.plot(x)
Ax2 = pygpl.contour(t1.Temp)
Axc = pl.colorbar(Ax2, Ax2.plots[0], pos='b')

Ax = pl.grid([[Ax1, Axc]])
# NB: It seems you can't pickle lambdas - need to look this up
# ironically this pooches every axis *but* time axes
#pl.save(Ax, 'test_ph3.fig')
#Ax = pl.load('test_ph3.fig')
Ax.render()

