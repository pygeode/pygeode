# Issue 125 - Issue with loading data into memory after converting Pygeode dataset to Xarray dataset
# https://github.com/pygeode/pygeode/issues/125

def test_xarray_values():
  import pygeode as pyg
  import pygeode.ext_xarray as pyg_xr
  import xarray as xr
  from pygeode.tutorial import t1

  t1_xr = pyg_xr.to_xarray(t1)
  assert hasattr(t1_xr.Temp,'values')
  values = t1_xr.Temp.values
  assert (values.shape == t1.Temp.shape)

