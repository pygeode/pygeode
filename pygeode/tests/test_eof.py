from pygeode import libpath
from pygeode.formats import netcdf, cccma
from pygeode.climat import climatology, climtrend, from_trend, detrend
#from pygeode.eof2 import eof as EOF
from pygeode.eof import EOF
from pygeode.varoperations import unfill
from pygeode.var import Var
import numpy as np
from math import pi

x1 = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_TO2Ms_tos.nc").tos
x2 = netcdf.open(libpath+"/data/CCMVal2_REF-B2_CMAM_1_TO2Ms_tos.nc").tos

taxis1 = x1.time
taxis1 = taxis1.modify(resolution='month')
x1 = x1.replace_axes(time=taxis1)
taxis2 = x2.time
taxis2 = taxis2.modify(resolution='month')
x2 = x2.replace_axes(time=taxis2)

mask = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_ss").DZG.squeeze(lev=1)==0

# only water?
x1 *= mask
x2 *= mask

# only look at tropics
x1 = x1(lat=(-15,15))
x2 = x2(lat=(-15,15))

# Subtract climatology
x1 = detrend(x1)
x2 = detrend(x2)

x1.name = x2.name = "SST"

num = 3
eof1, eig1, pc1 = EOF(x1, num=num)
pc1 *= eig1
eof2, eig2, pc2 = EOF(x2, num=num)
pc2 *= eig2

eof1 = unfill(eof1,0) # ignore the zeros from the mask
eof2 = unfill(eof2,0) # ignore the zeros from the mask


#"""
from pygeode.plot import plotvar
import matplotlib.pyplot as pl

nt = len(pc1.getaxis('time'))

for o in range(num):
  fig = pl.figure()
  ax = fig.add_subplot(221)
  plotvar(eof1(order=o), pcolor=True, ifig=fig, ax=ax)
  ax = fig.add_subplot(222)
  plotvar(eof2(order=o), pcolor=True, ifig=fig, ax=ax)
  ax = fig.add_subplot(223)
  plotvar(pc1(order=o), ifig=fig, ax=ax)
  ax = fig.add_subplot(224)
  plotvar(pc2(order=o,i_time=(0,nt-1)), ifig=fig, ax=ax)


pl.show()
#"""

