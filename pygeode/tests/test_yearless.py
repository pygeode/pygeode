from pygeode.formats import netcdf
from pygeode.timeaxis import Yearless
from pygeode import libpath
from pygeode.plot import plotvar
from matplotlib.pyplot import show

d = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_T2Mz_ta.nc")

taxis = d.time.modify(resolution='day')
print taxis

startdate = taxis.startdate.copy()
units = taxis.units
ref = taxis.reltime(units=units)

del startdate['year']
del startdate['month']

yt = Yearless (ref, units=units, startdate=startdate).modify(resolution='day')
#yt = taxis
print yt

ta = d.ta.replace_axes(time=yt)(plev=1000)

plotvar(ta)

#print ta

show()
