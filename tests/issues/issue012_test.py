# Issue 12 - interpolation code fails when both loop_xout and reversed axis are triggered
# https://github.com/pygeode/pygeode/issues/12

# Input axis values in decreasing order
def test_1():
  from pygeode.axis import XAxis
  from pygeode.interp import interpolate

  # Input data
  inaxis = XAxis([3,2,1])
  indata = inaxis ** 2

  # Output axis
  outaxis = XAxis([1.5, 2.5])

  outdata = interpolate(indata, inaxis=inaxis, outaxis=outaxis, interp_type='linear')
  outdata = list(outdata.get())


# Same as above, but multiple input/output axes  (resets 'j')
def test_2():
  from pygeode.axis import XAxis, YAxis
  from pygeode.interp import interpolate

  # Input data
  inaxis = XAxis([3,2,1])
  extra_axis = YAxis([2,4,6])
  indata = inaxis ** 2 + extra_axis*0

  # Output axis
  outaxis = XAxis([1.5, 2.5])
  outx = outaxis + extra_axis*0

  outdata = interpolate(indata, inaxis=inaxis, outaxis=outaxis, outx=outx, interp_type='linear')
  outdata = list(outdata.get())



