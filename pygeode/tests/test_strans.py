from pygeode.s_trans2 import Strans
#from pygeode.spectral.s_trans import Strans as Strans_orig
import numpy as np
from matplotlib import pyplot as pl
from pygeode.timeaxis import ModelTime365
from pygeode.plot import plotvar
from math import pi
from pygeode.var import Var


t = np.arange(1280)/10.
h = np.empty(1280)
h[:630] = np.cos(2*pi * t[:630] * 6.0 / 128.0)
h[630:] = np.cos(2*pi * t[630:] * 25.0 / 128.0)
h[200:300] += 0.5 * np.cos(2*pi*t[200:300] * 52.0 / 128.0)

T = ModelTime365(t, units='days', startdate={'year':2010})#.modify(exclude='year')
H = Var([T], values=h, name = 'signal')

fig = pl.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plotvar(H, figure=fig, ax=ax1)
S = abs(Strans(H,maxfreq=0.5))
print H
print H.get().shape
print H.time.values
print H.time.delta('days')
print S.shape
print S.get().shape
#quit()

#plotvar(S, pcolor=False, figure=fig, ax=ax2, colorbar=True)
plotvar(S, pcolor=False, figure=fig, ax=ax2)

fig = pl.figure()
ax = fig.add_subplot(111)
plotvar(S(freq=0.41), figure=fig, ax=ax)
plotvar(S(freq=0.2), figure=fig, ax=ax)
plotvar(S(freq=0.05), figure=fig, ax=ax)

#pl.plot(h, figure=pl.figure())
#pl.pcolor(np.abs(Strans_orig(h).ST), figure=pl.figure())

pl.show()
