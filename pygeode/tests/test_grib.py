#!/usr/bin/python

#TODO: fix segmentation fault on exit

from pygeode.formats import grib
from pygeode.plot import plotvar

from matplotlib.pyplot import ion
ion()

from pygeode import libpath
dataset = grib.open(libpath+"/data/gfsanl.2009060100")

print dataset

ta = dataset.TMP
zt = ta.mean('lon')
#zt = ta[:,:,::10,::10].mean('lon')

plotvar(zt)
#plotvar (ta(lev=64), pcolor=False)


