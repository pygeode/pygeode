from pygeode import libpath
from pygeode.formats import cccma, netcdf
from pygeode.plot import plotvar
import matplotlib.pyplot as pl

d = cccma.open(libpath+"/data/mm_t31ref2d4_010_m01_gs", autogrid = True)

x = d.O3(lat=10,lon=10)
#print x.size

netcdf.save("test.nc", x)
