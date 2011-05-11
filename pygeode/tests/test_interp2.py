#!/usr/bin/python

from pygeode.formats import cccma, netcdf
from pygeode.cccma import spectral
from pygeode.plot import plotvar
from pygeode.axis import Pres, Hybrid, Lon, Lat
from pygeode.timeaxis import Time
from pygeode.interp import Interp

from matplotlib.pyplot import show

import numpy as np

from pygeode import libpath
ss = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_ss", iyear=1950, delt=720)

lnsp = spectral.to_grid(ss.LNSP)
temp = spectral.to_grid(ss.TEMP)

print temp

temp2 = Interp(temp, inaxis='lat', outaxis=Lat(np.arange(-85,86)))
print temp2

plotvar (temp(day=1,hour=0,lev=1000), pcolor=True)
plotvar (temp2(day=1,hour=0,lev=1000), pcolor=True)
show()
