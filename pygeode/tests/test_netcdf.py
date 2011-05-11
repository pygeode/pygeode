#!/usr/bin/python

from pygeode.formats import netcdf as nc
from pygeode.plot import plotvar
from pygeode.axis import Pres, Lat
from pygeode.timeaxis import ModelTime365

from matplotlib.pyplot import ion
ion()

from pygeode import libpath
#d = nc.open(libpath+'/../data/CCMVal2_REF-B2_GEOSCCM_1_T3M_ta.nc')
#d = nc.open(libpath+'/../data/CCMVal2_REF-B2_CMAM_1_T2Iz_ta.nc')

class MyTime(ModelTime365):
  def __init__(self, values, *args, **kwargs):
    delt = kwargs.pop('delt',None)
    if delt is None:
      ModelTime365.__init__(self, values, *args, **kwargs)
    else:
      ModelTime365.__init__(self, values*delt, *args, **kwargs)

time = (MyTime, dict(startdate=dict(year=1950,month=1), delt=720., units='seconds'))
d1 = nc.open(libpath+"/data/dm_yphs40ac2_015_m12wres.nc.001", dimtypes={'Z01':Pres,'lat':Lat,'time':time})
#d1 = nc.open("wres.nc")

print d1
#plotvar(d1.WRES[0,:,:])

d1.atts = {'test':123}

nc.save("wres.nc", d1)
quit()

d2 = nc.open(libpath+"/data/dm_yphs40ac2_016_m01wres.nc.001")

#TODO: check the globbing - was getting errors?

print d1
print d2

from pygeode.dataset import concat
d = concat(d1,d2)

print d
plotvar (d.WRES[:,0,:])
