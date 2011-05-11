from pygeode.ensemble import ensemble
from pygeode import libpath
from pygeode.formats import netcdf

f = netcdf.open(libpath+"/data/CCMVal2_REF-B1_CMAM_1_T2Mz_ta.nc")
t = f.ta

e = ensemble(f, f)

print e
print e.ta
