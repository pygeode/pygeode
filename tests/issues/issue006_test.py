# Issue 6 - Advanced slicing sometimes fails
# https://github.com/pygeode/pygeode/issues/6

def test_slice():
  from pygeode.axis import XAxis, YAxis
  from pygeode.var import Var
  import numpy as np
  values = np.zeros([10,10])
  xaxis = XAxis(np.arange(10))
  yaxis = YAxis(np.arange(10))
  var = Var(axes=[xaxis,yaxis], values=values)
#  slicedvar = var.slice[[ 7, -2, -4,  9,  4,  0], [ 7, -2, -4,  9,  4,  0]]
  slicedvar = var.slice[[ 7, -2, -4,  9,  4,  0], [ -1,  0,  4]]
  slicedvar.get()

