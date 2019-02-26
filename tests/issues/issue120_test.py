# Issue 120 - Cannot switch off showvar's colorbar as intended
# https://github.com/pygeode/pygeode/issues/120

def test_suppresscolorbar():
  from pygeode.tutorial import t1
  import pygeode as pyg
  import pylab as pyl

  pyl.ioff()

  # default behaviour
  ax1 = pyg.showvar(t1.Temp)

  ax2 = pyg.showvar(t1.Temp,colorbar=False)

  # assuming that if there is a second axis, it will be a colorbar
  # and axes list will have non-zero length
  assert len(ax1.axes)==2

  assert len(ax2.axes)==0

