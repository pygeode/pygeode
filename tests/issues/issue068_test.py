# Issue 68  - Implementation of a Station axis
# https://github.com/pygeode/pygeode/issues/68

# Sample station locations
obs_locations = dict(
  Alert           = (82.451065,  -62.506771,   200, 'Canada'),
  Candle_Lake     = (53.987108, -105.117939,   600, 'Canada'),
  Egbert          = (44.231006,  -79.783839,   251, 'Canada'),
  Chibougamau     = (49.69251,   -74.342296,   393, 'Canada'),
  Estevan_Point   = (49.382935, -126.544097,     7, 'Canada'),
  Fraserdale      = (49.875168,  -81.569774,   210, 'Canada'),
  Lac_Labiche     = (54.953809, -112.466649,   540, 'Canada'),
  Sable_Island    = (43.93227,   -60.01256,      5, 'Canada'),
  Bratts_Lake     = (50.201631, -104.711259,   595, 'Canada'),
  Esther          = (51.669987, -110.206175,   707, 'Canada'),
  Toronto         = (43.780491,  -79.46801,    198, 'Canada'),
)

# Generate a Station axis from the data above
def make_station_axis ():
  from pygeode.axis import Station
  station_names, lats, lons, elevations, countries = zip(*[
    (station_name, lat, lon, elevation, country)
    for station_name, (lat, lon, elevation, country) in obs_locations.items()
  ])
  return Station(values=station_names, lat=lats, lon=lons, elevation=elevations, country=countries)

# Try to create a Station axis
def test_creation():
  stations = make_station_axis()

