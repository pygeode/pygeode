#!/usr/bin/python

from pygeode.formats import cccma
from pygeode.formats.multifile import openall
from pygeode.axis import Lon, Lat, Hybrid
from pygeode.timeaxis import Time
from pygeode.plot import plotvar as plot

from pygeode.cccma import spectral

from matplotlib.pyplot import ion
ion()

from pygeode import libpath
#dataset = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_?s", autogrid=True)
dataset = openall(format=cccma,files=libpath+"/data/mm_t31ref2d4_010_m01_?s", autogrid=True)
v = dataset.O3

v = v.slice[0,-1,...](lat=(-45,45))
print v
plot(v)
print v.mean()

