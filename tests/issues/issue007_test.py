# Issue 7 - val_as_date_std fails on negative values
# https://github.com/pygeode/pygeode/issues/7

def test_val_as_date():
  import pygeode as pyg, numpy as np
  tm = pyg.StandardTime(values=np.arange(10), units='days', startdate={'year':1980})
  dt = tm.val_as_date(-5)
  assert dt['year'] == 1979 and dt['month'] == 12 and dt['day'] == 27

