#!/usr/bin/python

#print globals()
#quit()

from pygeode.formats import cccma
from pygeode.plot import plotvar as plot
from pygeode.axis import Lon
from pygeode.timeaxis import StandardTime

from matplotlib.pyplot import ion
ion()

from pygeode import libpath
dataset = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_ss", autogrid=True, coord='ET15', plid=.0575)

t = dataset.TEMP

t1 = t.slice[0,...]( lat=(-90,45), eta=(0.5,1) , lon=11.25)

plot (t1, pcolor=True, wait=True)
plot (t1.slice[:,:,::2,:], pcolor=True, wait=True)
plot (t1.slice[:,:,::3,:], pcolor=True, wait=True)
plot (t1.slice[:,:,::4,:], pcolor=True, wait=False)

print t1.get().shape
