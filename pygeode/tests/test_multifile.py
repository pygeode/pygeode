from pygeode.formats import netcdf, mpeg
from pygeode.formats.multifile import open_multi
from pygeode import libpath
from pygeode.axis import Pres
from pygeode.plot import plotvar
from matplotlib.pyplot import show

d = open_multi (libpath+"/data/era40_200?_t.nc", format=netcdf, pattern="era40_$Y_t.nc")
#d = netcdf.open(libpath+"/data/era40_2000_t.nc")

t = d.t
t = t.replace_axes(levelist=Pres)

plotvar (t(levelist=1000, year=2000,month=3,day=1))
show()

#t = t.slice(levelist=1000).load()
#mpeg.save("test.mpeg", t, fps=10)
