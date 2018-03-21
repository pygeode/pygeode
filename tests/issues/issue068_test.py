# Issue 68  - Implementation of a Station axis
# https://github.com/pygeode/pygeode/issues/68

# Sample station locations
obs_locations = (
  ('Alert'           , (82.451065,  -62.506771,   200, 'Canada')),
  ('Candle_Lake'     , (53.987108, -105.117939,   600, 'Canada')),
  ('Egbert'          , (44.231006,  -79.783839,   251, 'Canada')),
  ('Chibougamau'     , (49.69251,   -74.342296,   393, 'Canada')),
  ('Estevan_Point'   , (49.382935, -126.544097,     7, 'Canada')),
  ('Fraserdale'      , (49.875168,  -81.569774,   210, 'Canada')),
  ('Lac_Labiche'     , (54.953809, -112.466649,   540, 'Canada')),
  ('Sable_Island'    , (43.93227,   -60.01256,      5, 'Canada')),
  ('Bratts_Lake'     , (50.201631, -104.711259,   595, 'Canada')),
  ('Esther'          , (51.669987, -110.206175,   707, 'Canada')),
  ('Toronto'         , (43.780491,  -79.46801,    198, 'Canada')),
)

# Generate a Station axis from the data above
def make_station_axis ():
  from pygeode.axis import Station
  station_names, lats, lons, elevations, countries = list(zip(*[
    (station_name, lat, lon, elevation, country)
    for station_name, (lat, lon, elevation, country) in obs_locations
  ]))
  return Station(station=station_names, lat=lats, lon=lons, elevation=elevations, country=countries)

# Generate a dummy variable with a station axis
def make_var ():
  from pygeode.var import Var
  from pygeode.timeaxis import StandardTime
  import numpy as np
  time = StandardTime(startdate=dict(year=2009,month=1,day=1), values=list(range(10)), units='days')
  station = make_station_axis()
  return Var([time,station], values=np.arange(len(time)*len(station)).reshape(len(time),len(station)), name="dummy")

# Try to create a Station axis
def test_creation():
  stations = make_station_axis()
  # Make sure we've got all the auxiliary information in the right place.
  assert 'lat' in stations.auxarrays
  assert 'lon' in stations.auxarrays
  assert 'elevation' in stations.auxarrays
  assert 'country' in stations.auxarrays
  # Make sure we can access the auxiliary information
  assert len(stations.lat) == len(stations.lon) == len(stations.elevation) == len(stations.country) == len(stations)

# Test equality operator
def test_equality():
  stations1 = make_station_axis().slice[1::]
  stations2 = make_station_axis().slice[::-1]
  assert (stations1 == stations1) is True
  assert (stations2 == stations2) is True
  assert (stations1 == stations2) is False
  assert (stations1 == stations2) is False

# Test creation of a variable with a station axis
def test_var_creation():
  x = make_var()
  # Try loading the data
  x.get()

# Test selecting a single station
def test_select_station():
  import numpy as np
  x = make_var()
  y = x(station='Alert')
  print(y)
  # Try loading the data
  y.get()
  assert y.get().shape == y.shape
  assert len(y.station) == 1
  # Try another station
  z = x(station='Egbert')
  assert not np.all(y.get() == z.get())
  # Try a fictional station, make sure there is no match
  w = x(station='Moon')
  assert len(w.station) == 0

# Test concatenating stations together
def test_concat():
  import numpy as np
  from pygeode.var import concat
  x = make_var()
  # First, pull out two stations
  y = x(station='Sable_Island')
  z = x(station='Bratts_Lake')
  print("Sable_Island auxarrays: %s" % y.station.auxarrays)
  print("Bratts_Lake auxarrays: %s" % z.station.auxarrays)
  # Now, concatenate the data along the station axis.
  w = concat([y,z])
  assert len(w.station) == 2
  print("Station values: %s" % w.station.values)
  print("Station auxarrays: %s" % w.station.auxarrays)
  print("Station names: %s" % w.station.station)
  assert list(w.station.station) == ['Sable_Island', 'Bratts_Lake']
  # Make sure the getview() functionality works.
  test1 = x.get()
  test2 = y.get()
  test3 = z.get()
  print('orig: %s' % test1)
  print('y: %s %s' % (y, test2))
  print('z: %s %s' % (z, test3))
  test4 = np.concatenate((test2,test3),axis=1)
  print(test4)
  test5 = w.get()
  assert np.all(test4 == test5)
  # Make sure this works the same as pulling the two stations without
  # splitting / concatenating
  test1 =  w.get()
  test2 = x.slice[:,7:9].get()
  print('w.get(): %s' % test1)
  print('x.slice[:,7:9]: %s' % test2)
  assert np.all(test1 == test2)

# Test writing to netcdf, and reading back in
def test_encode_decode():
  from pygeode.formats import netcdf
  import numpy as np
  x = make_var()
  netcdf.save("issue068_test.nc", x)
  y = netcdf.open("issue068_test.nc").dummy
  type1 = type(x.station)
  type2 = type(y.station)
  assert type1 is type2, (type1, type2)
  assert x.station == y.station
  assert np.all(x.get() == y.get())
