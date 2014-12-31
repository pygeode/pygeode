# Issue 58 - Ufunc does not like numpy scalars

def test_issue058():
  from pygeode.tutorial import t1
  import numpy as np
  # This throws an Exception:
  celcius = t1.Temp - np.float32(273.15)
