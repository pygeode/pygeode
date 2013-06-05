# Issue 36 - plot_v1 modifies source data

def test_issue036():
  import numpy as np
  from pygeode.axis import Height
  from pygeode.var import Var
  from pygeode.plot import plotvar

  height = Height(np.arange(10))
  var = Var([height], values=np.zeros(10))
  plotvar(var)
  assert np.all(height.values == np.arange(10))

