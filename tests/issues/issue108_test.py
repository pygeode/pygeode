# Issue 108 - Unable to access data from a scalar variable (with no axes) 
# https://github.com/pygeode/pygeode/issues/108

def test_scalar_variable_read():
  import pygeode as pyg
  # Create a scalar variable and test reading from it
  v = pyg.Var((), name = 'scalar', values = 10.)
  assert v[()] == 10.

  # Write to netcdf
  pyg.save('test_issue_108.nc', v)

def test_scalar_from_netcdf():
  import pygeode as pyg
  # read from netcdf and access data
  v = pyg.open('test_issue_108.nc').scalar

  assert v[()] == 10.
