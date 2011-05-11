from pygeode.formats import netcdf
from pygeode import libpath

f = netcdf.open(libpath+"/data/era40_2000_t.nc")
netcdf.save ("compressed.nc", f, compress=True)
