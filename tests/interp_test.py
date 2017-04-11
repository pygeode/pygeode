# Test some basic properties of the interpolation routine.

import unittest
import numpy as np
from pygeode.var import Var
from pygeode.axis import XAxis, YAxis
from pygeode.interp import interpolate
from nose.tools import raises

# Helper function - array comparison with nan support
def alleq (x1, x2):
  equal = (x1==x2)
  both_nan = np.isnan(x1) & np.isnan(x2)
  return np.all(equal | both_nan)

class TestInterp(unittest.TestCase):

  def setUp(self):
    # Use linear coordinates
    x = np.array([0,1,2])
    y = np.array([0,1,2,3])
    # Some simple data with a hole in it
    self.data    = np.array([[1,2,3],[4,float('nan'),6],[7,8,9],[10,11,12]])
    self.data_nm = np.array([[1,2,3],[4,5,6],[7,11,9],[10,8,12]])
    # Construct a Var object
    x = XAxis(x)
    y = YAxis(y)
    var    = Var(axes=[y,x], values=self.data)

    self.x = x
    self.y = y
    self.var = var

    # Some interpolation coordinates
    # mid-points (and values just outside the range)
    self.x2 = XAxis(values=[-0.5,0.5,1.5,2.5])
    self.y2 = YAxis(values=[-0.5,0.5,1.5,2.5,3.5])
    # Reverse of original values
    self.x3 = XAxis(values=[2,1,0])
    self.y3 = YAxis(values=[3,2,1,0])
    # Reverse of mid-points
    self.x4 = XAxis(values=[2.5,1.5,0.5,-0.5])
    self.y4 = YAxis(values=[3.5,2.5,1.5,0.5,-0.5])
    # Single non-midpoint
    self.x5 = XAxis(values=[1.4])
    self.y5 = YAxis(values=[2.2])

    # Non-monotonic axis
    self.y_nm = Var(axes=[y,x], values=[[0,0,0], [1,1,2], [2,2,1], [3,3,3]])

    # 2D interpolation
    xfield = np.array([[-1,0,1],[0,1,2],[1,2,3],[0,1,2]])
    yfield = np.array([[-1,0,1,2],[0,1,2,3],[1,2,3,4]]).transpose()
    self.xfield = Var(axes=[y,x], values=xfield)
    self.yfield = Var(axes=[y,x], values=yfield)
    self.x6 = XAxis(values=[-1,0,1,2,3])

    #TODO: interpolation via 2D coordinate field

  # Trivial case (no interpolation)
  def test_nointerp (self):
    input = self.var.get()

    output = interpolate(self.var, inaxis=self.x, outaxis=self.x, interp_type='linear').transpose(YAxis,XAxis).get()
    # GSL interoplates over nans, so we need to undo this for a direct comparison
    masked_output = np.array(output)
    masked_output[1,1] = float('nan')
    self.assertTrue(alleq(masked_output,input), output)
    self.assertEqual(output[1,1], 5.)  # check how GSL interoplates over the nan

    # Repeat, but for the y axis

    output = interpolate(self.var, inaxis=self.y, outaxis=self.y, interp_type='linear').transpose(YAxis,XAxis).get()
    # GSL interoplates over nans, so we need to undo this for a direct comparison
    masked_output = np.array(output)
    masked_output[1,1] = float('nan')
    self.assertTrue(alleq(masked_output,input), output)
    self.assertEqual(output[1,1], 5.)  # check how GSL interoplates over the nan


  # Test out interpolation/extrapolation
  def test_interp (self):
    input = self.var.get()

    # Interpolation along X axis
    for slope in (float('nan'),0,1,2):
      output = interpolate(self.var, inaxis=self.x, outaxis=self.x2, interp_type='linear', d_above=slope, d_below=slope).transpose(YAxis,XAxis).get()
      if np.isnan(slope):
        self.assertTrue(np.all(np.isnan(output[:,0])), output)
        self.assertTrue(np.all(np.isnan(output[:,-1])), output)
        self.assertTrue(np.all(np.isfinite(output[:,1:-1])), output)
      else:
        self.assertTrue(np.all(np.isfinite(output)), output)
        # Check extrapolation (out of range by half a coordinate unit)
        self.assertTrue(np.all(output[:,0] == input[:,0] - 0.5*slope), output)
        self.assertTrue(np.all(output[:,-1] == input[:,-1] + 0.5*slope), output)

    # Interpolation along Y axis
    for slope in (float('nan'),0,1,2):
      output = interpolate(self.var, inaxis=self.y, outaxis=self.y2, interp_type='linear', d_above=slope, d_below=slope).transpose(YAxis,XAxis).get()
      if np.isnan(slope):
        self.assertTrue(np.all(np.isnan(output[0,:])), output)
        self.assertTrue(np.all(np.isnan(output[-1,:])), output)
        self.assertTrue(np.all(np.isfinite(output[1:-1,:])), output)
      else:
        self.assertTrue(np.all(np.isfinite(output)), output)
        # Check extrapolation (out of range by half a coordinate unit)
        self.assertTrue(np.all(output[0,:] == input[0,:] - 0.5*slope), output)
        self.assertTrue(np.all(output[-1,:] == input[-1,:] + 0.5*slope), output)

  # Test out non-monotonic interpolation
  @raises(ValueError)
  def test_nm_interp (self):
    # Interpolation with non-monotonic source data - should fail with ValueError
    output = interpolate(self.var, inaxis=self.y, outaxis=self.y2, inx=self.y_nm, interp_type='linear').transpose(YAxis,XAxis).get()

  # Test interpolation via 2D coordinate field
  def test_2d_interp(self):
    input = self.var.get()

    output = interpolate(self.var, inaxis=self.x, outaxis=self.x6, inx=self.xfield, interp_type='linear').transpose(YAxis,XAxis).get()

    nan = float('nan')

    expected = np.array([[1,2,3,nan,nan],[nan,4,5,6,nan],[nan,nan,7,8,9],[nan,10,11,12,nan]])
    self.assertTrue(alleq(output,expected), output)

if __name__ == '__main__': unittest.main()
