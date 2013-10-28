# Issue 44 - SmoothVar fails when accessing only first element along smoothed axis

def test_issue044():
  import pygeode as pyg
  import numpy as np
  time = pyg.ModelTime365(values=np.arange(100), units='days', startdate=dict(year=1, month=1))
  ts = time.smooth('time', 20)
  ts[0]

