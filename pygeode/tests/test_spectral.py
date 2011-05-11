#!/usr/bin/python

from pygeode.formats import cccma
from pygeode.cccma import spectral
from pygeode.plot import plotvar as plot
from matplotlib.pyplot import ion
ion()

from pygeode import libpath
dataset = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_ss", delt=720, iyear=1950, coord='ET15', plid=.0575)

#lnsp = dataset.LNSP.slice[0,:,:]
dataset = dataset(year=1960, month=1, day=1, hour=6)
lnsp = dataset.LNSP
m = dataset.m
n = dataset.n

lnsp = lnsp * (m<=5) * (n<=5)
lnsp.name = 'LNSP (N<=5)'

plot(spectral.to_grid(lnsp))



