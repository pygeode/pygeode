from pygeode import libpath
from pygeode.formats import netcdf, cccma
from pygeode.climat import climtrend, from_trend, detrend
from pygeode.svd import SVD
from pygeode.varoperations import unfill
from math import pi

x = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_TO2Ms_tos.nc").tos
#y = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_TO2Ms_tos.nc").tos
#x = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_T2Mz_ta.nc").ta
#y = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_T2Mz_ta.nc").ta
y = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_T2Mz_ua.nc").ua

# Ignore time and day info, since these are monthly means
taxis = x.time
taxis = taxis.modify(resolution='month')
x = x.replace_axes(time=taxis)
y = y.replace_axes(time=taxis)

# Mask out land??
mask = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_ss").DZG(lev=1).squeeze()==0
mask.name = 'mask'
x = x * mask
#y = y * mask

# Subset the data
x = x(lat=(-15,15))
y = y(plev=(1000,10))
#x = x(month=1)
#y = y(month=1)

# Subtract climatology
x = detrend(x)
y = detrend(y)
#x -= x.mean('time').load()
#y -= y.mean('time').load()

# Weight by latitude
w1 = (x.lat * (pi/180)).cos()
w2 = (y.lat * (pi/180)).cos()


eof1, pc1, eof2, pc2 = SVD(x, y, num=3, weight1=w1, weight2=w2, matrix='cor')

eof1 = unfill(eof1,0) # ignore the zeros from the mask
eof2 = unfill(eof2,0) # ignore the zeros from the mask


"""
print "testing orthonormality of eofs"
print

for o1 in range(3):
  for o2 in range(3):
    print (eof1(order=o1).squeeze()*eof1(order=o2).squeeze()*w1).sum()

print

for o1 in range(3):
  for o2 in range(3):
    print (eof2(order=o1).squeeze()*eof2(order=o2).squeeze()*w2).sum()

print

#print "creating prinicpal components from the eofs, comparing to precalculated versions"
#print
#
#for o in range(3):
#  test = (x * eof1(order=o).squeeze() * w1).sum('lat','lon').getallvalues()
#  pc = pc1(order=o).squeeze().getallvalues()
#  print pc - test
#
#print
#
#for o in range(3):
#  test = (y * eof2(order=o).squeeze() * w2).sum('lat','plev').getallvalues()
#  pc = pc2(order=o).squeeze().getallvalues()
#  print pc - test
#
#print

print "checking cross-orthogonality of principal components"
print

for o1 in range(3):
  for o2 in range(3):
    print (pc1(order=o1).squeeze()*pc2(order=o2).squeeze()).sum()
#"""

"""
# Write to disk
approx1 = (pc1*eof1).sum('order')
approx1.name = 'var1'
approx2 = (pc2*eof2).sum('order')
approx2.name = 'var2'
approx1.lat.name = 'lat1'
approx2.lat.name = 'lat2'
netcdf.save ("test.nc", [approx1, approx2])
#"""


#"""
from pygeode.plot import plotvar
import matplotlib.pyplot as pl

for order in range(3):

  fig = pl.figure()

  eofax1 = fig.add_subplot(221)
  eofax2 = fig.add_subplot(222)
  plotvar(eof1(order=order), pcolor=True, ifig=fig, ax=eofax1)
  plotvar(eof2(order=order), pcolor=True, ifig=fig, ax=eofax2)

  pcax = fig.add_subplot(212)
  plotvar(pc1(order=order), ifig=fig, ax=pcax)
  plotvar(pc2(order=order), ifig=fig, ax=pcax, hold=True)

pl.show()
#"""

