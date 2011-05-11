#!/usr/bin/python

#from pygeode.deriv import DerivativeVar
from pygeode.axis import Lon
from pygeode.plot import plotvar
from matplotlib.pyplot import ion
ion()

from pygeode.formats import cccma
from pygeode import libpath
ss = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_ss", autogrid=True)


# 1D test
var = (Lon(360)-100.)**2/1000 - 50
#dvar = DerivativeVar(var, 0)
dvar = var.deriv(0)
plotvar(var.slice[::2], wait=True)
plotvar(dvar.slice[::2], wait=True)

# 2D test
lnsp = ss.LNSP
#dlnsp = DerivativeVar(lnsp, Lon)
dlnsp = lnsp.deriv(Lon)
plotvar(lnsp.slice[0,:,:], wait=True)
plotvar(dlnsp.slice[0,:,:])

