# Issue 104 - Unable to select data along an axis with underscores in the name
# https://github.com/pygeode/pygeode/issues/104

def test_underscores_in_axes():
  from pygeode.axis import ZAxis
  from pygeode.var import Var
  # Axis names without underscores works
  class HeightwrtGround(ZAxis): pass
  ax = HeightwrtGround(values=[1.5])
  x = Var(axes=[ax], name='blah', values=[123.45])
  y = x(heightwrtground=0.0)
  # Axis names with underscores fails
  class Height_wrt_Ground(ZAxis): pass
  ax = Height_wrt_Ground(values=[1.5])
  x = Var(axes=[ax], name='blah', values=[123.45])
  y = x(height_wrt_ground=0.0)


