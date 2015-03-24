# Issue 22 - plot attributes get overridden by NamedAxis defaults
# https://github.com/pygeode/pygeode/issues/22

# Make a sample file with a non-annotated pressure axis
from pygeode.axis import NamedAxis, Pres
from pygeode.formats import netcdf as nc

lat = NamedAxis(values=[-80,-70,-60,-50], name='lat')
p1 = NamedAxis(values=[1000.,900.,800.], name='p1')
x = lat * p1
x.name = 'x'

nc.save("issue022_test.nc", x)

# Load it back in
d = nc.open("issue022_test.nc", dimtypes={'p1':Pres})

# plotatts should be from the new (dimtypes) axis, not copied from the old one?
# (also implicitly asserts that other metadata (such as the name) are still
#  copied from the old axis)
assert d.p1.plotatts == Pres.plotatts


