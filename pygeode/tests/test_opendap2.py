from pygeode.formats import opendap, netcdf
from pygeode import libpath

d = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_T2Mz_ta.nc")

opendap.serve("/test.nc",d)

