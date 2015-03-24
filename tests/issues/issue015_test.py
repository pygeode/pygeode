# Issue 15 - netcdf crashes because of axes mismatch
# https://github.com/pygeode/pygeode/issues/15

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
