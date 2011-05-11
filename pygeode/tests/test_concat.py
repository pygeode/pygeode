#!/usr/bin/python

from pygeode.formats import netcdf
from pygeode import libpath
from pygeode import plot as p
from pygeode.axis import Pres

from numpy import arange
from pygeode import dataset

from matplotlib.pyplot import ion, show
ion()


cs = arange(-0.026, 0.027, 0.002)
wr = netcdf.open(libpath+'/data/dm_yphs40ac2_015_m12wres.nc.001', dimtypes={'Z01':Pres})
wr2 = netcdf.open(libpath+'/data/dm_yphs40ac2_016_m01wres.nc.001', dimtypes={'Z01':Pres})
wra = dataset.concat(wr, wr2)
print wr
print wr2
print wra
p.plotvar(wr.WRES.slice[:, :, 2], ifig=1, clevs=cs, clines=cs, wait=True)
p.plotvar(wr2.WRES.slice[:, :, 2], ifig=2, clevs=cs, clines=cs, wait=True)
#TODO: fix crash here
p.plotvar(wra.WRES.slice[:, :, 2], ifig=3, clevs=cs, clines=cs, wait=True)

N = len(wra.WRES.time)
a = N/4
b = N*3/4
#p.plotvar(wra.WRES[a:b, :, 2], ifig=4, clevs=cs, clines=cs, wait=True)

p.plotvar(wra.WRES.slice[N/2+1:, :, 2], ifig=4, clevs=cs, clines=cs, wait=True)

show()
