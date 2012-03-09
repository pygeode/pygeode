
import plot_wr_ph as pl, plot_sc_ph as pygpl
import numpy as np
import pylab as pyl
import pygeode as pyg
from pygeode.tutorial import t1

reload(pl)
reload(pygpl)

lv = np.arange(240, 310, 2)
ln = np.arange(240, 310, 10)
Ax = pygpl.contour(t1.Temp, lv, ln)
Ax.drawcoastlines()
Ax.drawmeridians([0, 90, 180, 270])
Ax.drawparallels([-60, -30, 0, 30, 60])

#pl.save(Ax, 'test_ph3.fig')
#Ax = pl.load('test_ph3.fig')
Ax.render(1)
