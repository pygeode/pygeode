from pygeode.tools import product

# Test the concatenation operator

# Helper function - Test concatenation for given array sizes

def do_concat (shape1, shape2, iaxis, count=[0]):
  from pygeode.axis import XAxis, YAxis, ZAxis, TAxis
  from pygeode.var import Var
  from pygeode.concat import concat
  from var_test import varTest
  import numpy as np

  # Increment test counter (and generate a unique test name)
  count[0] += 1
  testname = "concattest%05d"%(count[0])

  # Create some test data
  np.random.seed(count[0])
  array1 = np.random.randn(*shape1)
  array2 = np.random.randn(*shape2)

  # Get the # of dimensions, and assign a unique axis class for each dim
  assert array1.ndim == array2.ndim
  ndim = array1.ndim
  axis_classes = (XAxis, YAxis, ZAxis, TAxis)[:ndim]

  # Construct the first var
  axes = [cls(n) for cls,n in zip(axis_classes,array1.shape)]
  var1 = Var(axes = axes, values = array1, name = "myvar", atts={'a':1, 'b':2, 'c':3})

  # The second var should have the same axes, except for the concatenation one
  n1 = array1.shape[iaxis]
  n2 = array2.shape[iaxis]
  axes[iaxis] = axis_classes[iaxis](np.arange(n1, n1+n2))
  var2 = Var(axes = axes, values = array2, name = "myvar", atts={'a':1, 'b':3, 'd':4})

  # Try concatenating
  var = concat(var1,var2)

  # The expected result
  expected = np.concatenate ( (array1, array2), iaxis)

  # Test this
  test = varTest(testname=testname, var=var, values=expected)

  # Store this test
  globals()[testname] = test


# Now, do some tests

sizes = (1, 2, 3, 20)

for naxes in (1,2,3):
  # Shape of the output
  for shape in product(*([sizes]*naxes)):
    # Concatenation axis
    for iaxis in range(naxes):
      n = shape[iaxis]
      # Length of first array
      for n1 in sorted(set([0, 1, 2, n//3, n-1, n])):
        if n1 < 0 or n1 > n: continue
        n2 = n - n1
        shape1 = shape[:iaxis]+(n1,)+shape[iaxis+1:]
        shape2 = shape[:iaxis]+(n2,)+shape[iaxis+1:]
        do_concat (shape1, shape2, iaxis)

