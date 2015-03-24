# Issue 5 - cf decode fails on datestamps specified from year 0000
# https://github.com/pygeode/pygeode/issues/5

def test_issue005():

  from pygeode.timeaxis import ModelTime365
  from pygeode.axis import TAxis
  import numpy as np
  from pygeode.var import Var
  from pygeode.formats import netcdf as nc
  from pygeode import timeutils

  # Make a time axis starting at year 0
  startdate = dict(year=0,month=1,day=1)
  taxis = ModelTime365(values=10200, startdate=startdate, units='days')

  # Make some dummy variable
  np.random.seed(len(taxis))
  values = np.random.randn(len(taxis))
  var = Var(axes=[taxis], values=values, name='x')

  # Save it
  nc.save("issue005_test.nc", var)

  # Load it
  f = nc.open("issue005_test.nc")

  # Make sure we have a regular time axis
  # (no climatologies!)
  assert f.time.__class__ == ModelTime365
  assert hasattr(f.time,'year')

  # Okay, now reload it, but override the axis coming in
  f = nc.open("issue005_test.nc", dimtypes=dict(time=TAxis(taxis.values)))

  # Make sure we dimtypes is still working properly
  assert f.x.axes[0].__class__ == TAxis


  # For good measure, test that climatologies are still produced
  taxis = timeutils.modify(taxis,exclude='year',uniquify=True)
  values = np.random.randn(len(taxis))
  var = Var(axes=[taxis], values=values, name='c')

  nc.save("issue005_test.nc", var)
  f = nc.open("issue005_test.nc")

  assert not hasattr(f.time,'year')


