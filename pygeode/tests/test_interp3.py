#!/usr/bin/python

from pygeode.formats import cccma, netcdf
from pygeode.cccma import spectral
from pygeode.plot import plotvar
from pygeode.axis import Pres, Hybrid, Lon
from pygeode.timeaxis import Time
from pygeode.interp import Interp

from matplotlib.pyplot import show

from pygeode import libpath

etarange=(1,.0001)
#etarange=(1,0)


ccmval = netcdf.open(libpath+"/data/CCMVal2_REF-B2_CMAM_1_T2Iz_ta.nc")
ss = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_ss", coord="ET15", plid=.0575, iyear=1950, delt=720)

lnsp = spectral.to_grid(ss.LNSP)
temp = spectral.to_grid(ss.TEMP)

plotvar (temp(lon=0,i_time=0,eta=etarange), pcolor=True, wait=True)

p0 = lnsp.exp()*100
A = temp.eta.auxasvar('A')
B = temp.eta.auxasvar('B')
p = (A + p0 * B) / 100

paxis = ccmval.ta.plev

ta = Interp (temp, inaxis='eta', outaxis = paxis, inx = p.log(), outx = paxis.log())

plotvar (ta(lon=0,i_time=0), pcolor=True, wait=True)

te = Interp (ta, inaxis='plev', outaxis = temp.eta, inx = paxis.log(), outx = p.log())



plotvar (te(lon=0,i_time=0,eta=etarange), pcolor=True, wait=True)


diff = temp - te
plotvar (diff(lon=0,i_time=0,eta=etarange), pcolor=True, wait=True)



show()
