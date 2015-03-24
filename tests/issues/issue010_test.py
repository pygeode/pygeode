# Issue 10 - netcdf.dim2axes crashes if a dimension doesn't have a corresponding variable
# https://github.com/pygeode/pygeode/issues/10
def test_issue010():
  from pygeode.var import Var
  from pygeode.axis import Axis
  from pygeode.dataset import Dataset
  from pygeode.formats import netcdf as nc

  # Make some axes
  time_axis = Axis(values=[0], name='time')
  bnds_axis = Axis(values=[0,1], name='bnds')

  # Make some vars (note we don't have a 'bnds' variable corresponding to the 'bnds' dimension
  time_var = Var(axes=[time_axis], values=[1], name='time')
  time_bnds = Var(axes=[time_axis,bnds_axis], values=[[3,4]], name='time_bnds')

  # Make a dataset to hold the vars
  dataset = Dataset([time_var, time_bnds])

  # Manually appy dims2axes to detect our axes
  dataset = nc.dims2axes(dataset)


  # (crash)

