from pygeode import libpath
from pygeode.formats import netcdf, cccma
from pygeode.climat import detrend
from pygeode import eof
from pygeode.varoperations import unfill
from pygeode.var import Var

x = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_TO2Ms_tos.nc").tos

taxis = x.time
taxis = taxis.modify(resolution='month')
x = x.replace_axes(time=taxis)

mask = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_ss", delt=720, iyear=1950, coord='ET15', plid=.0575).DZG.squeeze(lev=1)==0

# only water?
x *= mask

# only look at tropics
x = x(lat=(-15,15))
#x = x(lat=(30,90))
#x = x(lat=(-50,0))

# Subtract climatology
x = detrend(x)

x.name = "SST"

num = 3

eof1, eig1, pc1 = eof.EOF_iter(x, num, out='EOF,EIG,PC')
#quit()
pc1 *= eig1
eof2, eig2, pc2 = eof.EOF_guess(x, num, out='EOF,EIG,PC')
pc2 *= eig2
#quit()
eof1 = unfill(eof1,0) # ignore the zeros from the mask
eof2 = unfill(eof2,0) # ignore the zeros from the mask

# Correct for negative correlations
corr = (pc1 * pc2).sum('time').get()
corr = Var(pc1.axes[0:1], values=[-1 if c < 0 else 1 for c in list(corr)])
pc2 *= corr
eof2 *= corr

# Rename
eof1 = eof1.rename("iterative EOF")
eof2 = eof2.rename("single-pass EOF")

diff = (eof2 - eof1).rename("diff")

print eig1.values
print eig2.values

#"""
from pygeode.plot import plotvar
import matplotlib.pyplot as pl

for o in range(num):
  fig = pl.figure()
  ax = fig.add_subplot(221)
  plotvar(eof1(order=o), pcolor=True, ifig=fig, ax=ax)
  ax = fig.add_subplot(222)
  plotvar(eof2(order=o), pcolor=True, ifig=fig, ax=ax)
#  plotvar(diff(order=o), pcolor=True, ifig=fig, ax=ax)
  ax = fig.add_subplot(212)
  plotvar(pc1(order=o), ifig=fig, ax=ax)
  plotvar(pc2(order=o), ifig=fig, ax=ax)


pl.show()
#"""

