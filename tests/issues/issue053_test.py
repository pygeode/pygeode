# Issue 53 - plotatts dict should not propagate from axis objects

def test_issue053():
  import pygeode as pyg
  import numpy as np
  l = pyg.regularlat(30)
  t = pyg.ModelTime365(values=np.arange(100), units='days', startdate=dict(year=1, month=1))
  v = pyg.Var((t, l), name='Test', values=np.ones((100, 30)))
  v.plotatts['scalefactor'] = 2.
  v.plotatts['plottitle'] = 'V'

  a = l * v
  b = t + v
  
  assert a.plotatts == v.plotatts
  assert b.plotatts == v.plotatts
