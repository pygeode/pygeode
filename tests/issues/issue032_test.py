# Issue 32 - nan-friendly reductions to a scalar don't work

def test_issue032():
  from pygeode.tutorial import t1

  t1.Temp.nansum()
  t1.Temp.nanmean()
  t1.Temp.nanstdev()
  t1.Temp.nanvariance()

  # This ones should always work
  # (just testing them for the hell of it)
  x = t1.Temp.sum()
  assert isinstance(x,float)
  x = t1.Temp.mean()
  assert isinstance(x,float)
  x = t1.Temp.stdev()
  assert isinstance(x,float)
  x = t1.Temp.variance()
  assert isinstance(x,float)
