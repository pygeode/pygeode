from pygeode.climat import climatology, dailymean, monthlymean, climtrend, from_trend, detrend, seasonalmean
from pygeode.formats import netcdf
import numpy as np
from pygeode.plot import plotvar
from matplotlib.pyplot import ion, show, figure
#ion()

from pygeode import libpath
d = netcdf.open(libpath+"/data/CCMVal2_REF-B2_CMAM_1_T2Mz_ta.nc")
ta = d.ta(year=(1960,2000))
print ta
plotvar (ta(lat=0,plev=1000))
s = seasonalmean(ta)
print s
# Hack the seasonal 'values' array, so we can overplot with the original data
s.time.values.flags.writeable = True
s.time.values[:-1] = ta.time.values[::3]
s.time.values[-1] = ta.time.values[-1]
plotvar (s(lat=0,plev=1000),ifig=1,hold=True)
show()
quit()
d = detrend(ta)
print d
plotvar (d(lat=0,plev=1000))
show()
quit()

#######################
from pygeode.datasets import ccmval2
ox = ccmval2.ref1.OX(ensemble=0, day=1, hour=6, eta=.0002, year=(1960,1961))
cox = climatology(ox).squeeze()
netcdf.save("ox_06.nc", cox)
quit()

ta = ccmval2.ref2.TEMP(ensemble=0, day=1, hour=6, eta=1).squeeze()
#netcdf.save("ta06.nc", ta)
quit()
tta = climtrend(ta)
tta = netcdf.save("trend06.nc", tta)()
lin = from_trend(ta.time, tta)
lin.name = 'linear_ta'
netcdf.save("linear06.nc", lin)

quit()
#######################

from pygeode import libpath
d = netcdf.open(libpath+"/data/CCMVal2_REF-B2_CMAM_1_T2Mz_ta.nc")
ta = d.ta

# Define the climatology
c = climatology(ta)

# Anchor the climatology to a file
print 'getting climatology'
c = netcdf.save("clim.nc", c)()
c = c.load()

# Save some anomolies
anom = ta-c
anom.name = "anom"
#t = anom.time
#for year, month, day in zip(t.year.values, t.month.values, t.day.values):
#  print year, month, day
#quit()
print 'getting anom'
anom = netcdf.save ("anom.nc", anom)()
#t = anom.time
#for year, month, day in zip(t.year.values, t.month.values, t.day.values):
#  print year, month, day
#quit()

# linear trend
print 'getting trend'
tr = trend(anom).load()
netcdf.save ("trend.nc", tr)

print 'getting linearized dataset'
lin = from_trend(tr, anom.time)
netcdf.save ("linear.nc", lin)

# detrend the anomolies
anom_detrended = anom - lin
anom_detrended.name = "anom_detrended"
print 'detrending the anomolies'
anom_detrended = netcdf.save ("anom_detrended.nc", anom_detrended)()

quit()

# compute EOFs?
var = anom_detrended.load()
data = var.get().reshape(var.shape[0],-1)
cov = np.dot (data.transpose(), data)

print 'start solver'
w, v = np.linalg.eigh(cov)
print 'finished solver'

# Reverse the order, so most significant is first
v = v[:,::-1]
w = w[::-1]
from pygeode.axis import NamedAxis
from pygeode.timeaxis import Time
from pygeode.var import Var
eigaxis = NamedAxis(np.arange(len(w)), name='ind')
ti = var.whichaxis(Time)
shape = list(var.shape)
shape = shape[:ti] + shape[ti+1:] + [len(w)]
eig = v.reshape(shape)
axes = list(var.axes)
axes = axes[:ti] + axes[ti+1:] + [eigaxis]
eig = Var(axes, values=eig)

pc = np.dot(data, v)
pc = Var([var.getaxis(Time), eigaxis], values=pc)

fig = figure()
for i in range(12):
  ax = fig.add_subplot(3,4,i+1)
  plotvar(eig(ind=i), ifig=fig, ax=ax, wait=True)

fig = figure()
for i in range(12):
  ax = fig.add_subplot(3,4,i+1)
  plotvar(eig(ind=i+12), ifig=fig, ax=ax, wait=True)

fig = figure()
for i in range(12):
  ax = fig.add_subplot(12,1,i+1)
  plotvar(pc(ind=i), ifig=fig, ax=ax, wait=True)

show()
quit()

var = anom_detrended
MAX = var.max()
MIN = var.min()

for y in [2000,2001]:
  afig = figure()
  for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
    ax = afig.add_subplot(3,4,m)
    plotvar(var(year=y,month=m), ifig=afig, ax=ax, pcolor=True, vmin=MIN, vmax=MAX, wait=True)


show()
