# Issue 46 - Error handling in the interpolate code consists of crashing python

def test_issue046():
  import pygeode as pyg
  import numpy as np
  V = pyg.Var((pyg.Lat(32),), name='Test', values=np.zeros(32) * np.nan)
  V.interpolate('lat', pyg.gausslat(32))[:]

