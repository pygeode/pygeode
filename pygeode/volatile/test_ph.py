# Colour bar test
import plot_wr_ph as pl
import numpy as np
import pylab as pyl

reload(pl)

# Test data
x = np.arange(10) / (20*np.pi)
y = np.arange(40) / (20*np.pi)

x, y = np.meshgrid(x, y)
z = np.sin(x) * np.sin(y)
z2 = np.sin(x) * np.cos(y)

# Create contour plots
Ax1 = pl.contourf(x, y, z, 21)
Ax2 = pl.contourf(x, y, z2, 21, cmap=pyl.cm.Reds)
Ax2.setp_xaxis(major_locator = pyl.MultipleLocator(0.02),
         minor_locator = pyl.MultipleLocator(0.005))
Ax = pl.grid([[Ax1, Ax2]])

# Add colour bars
ticks = np.arange(0, 1, 0.1)/10.
AxG = pl.colorbar(Ax, Ax1.plots[0], orientation='horizontal', pos='b', ticks=ticks)
AxG = pl.colorbar(AxG, Ax2.plots[0], pos='l')

pl.save(AxG, 'test_ph.fig')
Ax = pl.load('test_ph.fig')
Ax.render()

