#!/usr/bin/python

"""
from pygeode.formats import hdf4
d = hdf4.open("/ipy/disk2/gmao/apr08/DAS.ops.asm.tavg3d_prs_v.GEOS510.20080430_1800.V01.hdf")

from pygeode.plot import plotvar
import matplotlib.pyplot as pl

print d

plotvar (d.PS)
pl.show()

quit()
"""
from pygeode.formats import hdf4
from pygeode.plot import plotvar as plot

from matplotlib.pyplot import ion
ion()

from pygeode import libpath
d = hdf4.open(libpath+"/data/MOP03M-200905-L3V92.0.1.hdf")

# MOPITT uses a fill value of -9999.0 (which isn't documented in the file?)
def get(s): return d[s].unfill(-9999.0)
#def get(s): return d[s]

ps_day = get('Surface Pressure Day')
ps_night = get('Surface Pressure Night')

t_day = get('Retrieved Surface Temperature Day')
t_night = get('Retrieved Surface Temperature Night')

npix_day = get('Number of Pixels Day')
npix_night = get('Number of Pixels Night')

satza_day = get('Satellite Zenith Angle Day')
satza_night = get('Satellite Zenith Angle Night')

sza_day = get('Solar Zenith Angle Day')
sza_night = get('Solar Zenith Angle Night')

co_day = get('Retrieved CO Mixing Ratio Profile Day')
co_night = get('Retrieved CO Mixing Ratio Profile Night')
co_tc_day = get('Retrieved CO Total Column Day')
co_tc_night = get('Retrieved CO Total Column Night')

kern_day = get('Retrieval Averaging Kernel Matrix Day')

si_day = get('Surface Index Day')

#print d
#for v in d.vars: print v.name

#plot (ps_day)
from pygeode.axis import Lat, Lon
ps_night = ps_night.replace_axes(Latitude=Lat, Longitude=Lon)
print ps_night
plot (ps_night)
#plot (t_day)
#plot (t_night)
#plot (t_day - t_night)

#plot (npix_day)
#plot (npix_night)

#plot(co_day[40,50,:])
#plot(co_tc_day)
#plot(co_tc_night)

#plot (kern_day[40,50,:,:])

#plot (si_day)
