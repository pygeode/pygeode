# Issue 90 - pygeode.multifile.openall shouldn't require a format if opener is
# provided.
# https://github.com/pygeode/pygeode/issues/90

def test_opener():

  from pygeode.formats import netcdf
  from pygeode.tutorial import t1
  from pygeode.formats.multifile import openall
  netcdf.save("issue090.data", t1)
  my_opener = lambda filename: netcdf.open(filename)
  f = openall("issue090.d???", opener=my_opener)

