import numpy as np
from pygeode.tools import product
from pygeode.var import Var
from pygeode.axis import Axis
from var_test import varTest

# Helper function: given an axis length, return a bunch of
# different slices to try out.
def gimme_some_slices(size):
  slices = []

  # No slicing
  slices.append(slice(None))

  # Slicing a single value
  for scalar in (-1,0,1,2):
    if scalar >= size: continue
    slices.append(scalar)
    slices.append(slice(scalar,scalar+1))
    slices.append([scalar])

  # Empty slice (nothing selected)
  #slices.append([])

  # The stuff below requires a non-degenerate array
  if (size == 0): return slices

  # Range of values
  np.random.seed(size)
  start, stop = sorted(np.random.randint(0, size, 2))

  for stride in (1,2):
    slices.append(slice(start, stop, stride))
    slices.append(slice(stop, start, -stride))

  # Random integer indices
  np.random.seed(size*42)
  slices.append(np.random.randint(-size,size, size//2+1))

  return slices

# Okay, now start constructing some test cases

# Different axis lengths to try out
sizes = (1, 2, 3, 10, 200)

# Compile a list of slices to try for each dimension size.
# Each size listed above is a dictionary entry, containing a list of slices
# to try for that axis size.
slices = {}
for size in sizes:
  slices[size] = gimme_some_slices(size)

# Each axis needs to be a distinct class, or view.get() gives bizarre error messages
from pygeode.axis import XAxis, YAxis, ZAxis
axis_classes = (XAxis, YAxis, ZAxis)

# Counter for giving each test a unique name
count = 1

for naxes in (1,2):
  print("Testing %s dimensions"%naxes)
  for shape in product(*([sizes]*naxes)):
    print("  Testing shape %s"%str(shape))

    np.random.seed(shape)
    values = np.random.randn(*shape)
#    print "full values:", values

    axes = [axis_classes[i](sorted(np.random.randn(n))) for i,n in enumerate(shape)]
    for i,axis in enumerate(axes):
      axis.name = 'axis%s'%i
#      print "axis %s values: %s"%(i,axis.values)

    var = Var(axes, values=values)
    var.name = 'myvar'

    slicelists = [slices[size] for size in shape]
    print("    # tests: %d" % len(list(product(*slicelists))))
    for sl in product(*slicelists):
      print("    Testing slices %s"%repr(sl))
      # Trap any known failures here to further diagnose
#      assert (count!=4860), "shape: %s, slices: %s, values: %s, axes: %s"%(shape, str(sl), values, [a.values for a in var.axes])

      # slice the var immediately (before massaging the slices for numpy)
      slicedvar = var.slice[sl]

      # Force integer slices to be integer index arrays
      # (so numpy doesn't remove any dimensions)
      sl = tuple([i] if isinstance(i,int) else i for i in sl)

      # Apply the slices one dimension at a time.
      # Avoids a 'feature' in numpy slicing when there are advanced integer
      # indices.  If there are 2 such things, numpy goes into a special
      # mode where it zips the arrays to get a list of specific coordinates,
      # then picks out those individual elements.
      # E.g., you would think that x[ [0,1], [0,1] ] would give you a 2x2
      # square array of the first 2 rows and first 2 columns of x, but noooo...
      # what it does is give you a 1D array, consisting of elements (0,0) and
      # (1,1)!!!
      expected = values
      for dim in range(naxes):
        current_sl = [slice(None)]*dim + [sl[dim]] + [slice(None)]*(naxes-1-dim)
        expected = expected[current_sl]

      # Things are just about to start getting crazy-go-nuts.
      # Pass the var and expected values to 'varTest', which in turn
      # dynamically defines a test class (subclass of unittest.TestCase)
      # to check the var for consistency.
      # We then have to take this test class, and assign it to a variable
      # (this has to be done dynamically, since we're looping over many tests).
      # 'nosetests' will then find this file, import it, and look for any
      # global variables that represent a subclass of unittest.TestCase(?),
      # then invoke the corresponding tests.
      testname = 'slicetest%05d'%count
      globals()[testname] = varTest(testname=testname, var=slicedvar, values=expected)

      count += 1

