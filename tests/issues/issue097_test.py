# Issue 097 - Hook climatology operators into the Var object
# https://github.com/pygeode/pygeode/issues/97

def test_monthlymean():
  from pygeode.tutorial import t2
  from pygeode.climat import monthlymean
  import numpy as np
  var = t2.Temp(pres=1000.)
  from_var = var.monthlymean()
  from_climat = monthlymean(var)
  assert np.all(from_var.get() == from_climat.get())

def test_climatology():
  from pygeode.tutorial import t2
  from pygeode.climat import climatology
  import numpy as np
  var = t2.Temp(pres=1000.)
  from_var = var.climatology()
  from_climat = climatology(var)
  assert np.all(from_var.get() == from_climat.get())

