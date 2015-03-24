# Issue 69  - Time axes do not support unordered values
# https://github.com/pygeode/pygeode/issues/69

def test_sort():
  import numpy as np
  import pygeode as pyg

  t1 = pyg.StandardTime(units='days', values=[1, 3, 5, 8, 12, 14, 19], startdate=dict(year=2000, month=1, day=1))
  t2 = pyg.StandardTime(units='days', values=[6, 2, 14.5, 8.2, 13, 11., 9], startdate=dict(year=2000, month=1, day=1))
  v1 = 1.*t1
  v2 = 1.*t2
  vc = pyg.concat.concat([v1, v2])
  vs = vc.sorted(time=1)

  assert all(np.diff(vs[:]) > 0.)
