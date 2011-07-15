# Issue 16 - Var.get() failing
# http://code.google.com/p/pygeode/issues/detail?id=16

def test_issue016():
  from pygeode.axis import Lat, Lon, Pres

  # Create a simple variable
  lat = Lat([75,85,95])
  lon = Lon([100, 110, 120, 130])
  pres = Pres([1000,900,800,700])
  x = (lat-80)**2 + lon - pres/2.

  # Try getting the values
  x.get()
