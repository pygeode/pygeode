# Issue 25 - Netcdf module does not filter names that start with a digit
# https://github.com/pygeode/pygeode/issues/25

from pygeode.formats import netcdf as nc
from pygeode.axis import Lat
from pygeode.var import Var

def test_issue025():
  lat = Lat([80,70,60])
  var = Var(axes=[lat], values=[1,2,3], name='2B')

  # Save the variable
  nc.save ("issue025_test.nc", var)

  # This may crash in some versions of the netcdf library.

  # Even if it doesn't crash, it's a good idea to enforce the legal
  # netcdf names

  f = nc.open("issue025_test.nc")

  assert len(f.vars) == 1
  # Must not start with a digit (should have been filtered)
  assert not f.vars[0].name[0].isdigit()

