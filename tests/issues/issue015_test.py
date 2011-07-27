# Issue 15 - netcdf_new crashes because of axes mismatch
# http://code.google.com/p/pygeode/issues/detail?id=15

# Backported to 0.6
# (test the older netcdf module in the same way, although it never had this issue)

def test_issue015():
  from pygeode.formats import netcdf as nc
  from pygeode.axis import Lat, Lon, Pres

  # Create a simple variable
  lat = Lat([75,85,95])
  lon = Lon([100, 110, 120, 130])
  pres = Pres([1000,900,800,700])
  x = (lat-80)**2 + lon - pres/2.
  x.name = "stuff"

  # Save as a netcdf file
  nc.save("issue015_test.nc", x)

  # Reload
  f = nc.open("issue015_test.nc")
  y = f.stuff.load()
