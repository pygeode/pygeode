import numpy as np
from pygeode.tools import product
from pygeode.var import Var
from pygeode.axis import Axis
from var_test import varTest

def try_slice (shape):
  import numpy as np
  np.random.seed(shape)
  indata = np.random.randn(*shape)

sizes = (1, 2, 3, 10, 200)

# Compile a list of slices to try for each dimension size
slices = dict([(size,[]) for size in sizes])

for size in sizes:
  # No slicing
  slices[size].append(slice(None))

  # Slicing a single value
  for scalar in (-1,0,1,2):
    if scalar >= size: continue
    slices[size].append(scalar)
    slices[size].append(slice(scalar,scalar+1))
    slices[size].append([scalar])

  # Empty slice (nothing selected)
#  slices[size].append([])

  if (size == 0): continue

  # Range of values
  np.random.seed(size)
  start, stop = sorted(np.random.randint(0, size, 2))

  for stride in (1,2):
    slices[size].append(slice(start, stop, stride))
    slices[size].append(slice(stop, start, -stride))

  # Random integer indices
  np.random.seed(size*42)
  slices[size].append(np.random.randint(-size,size, size/2+1))

# Each axis needs to be a distinct class, or view.get() gives bizarre error messages
from pygeode.axis import XAxis, YAxis, ZAxis
axis_classes = (XAxis, YAxis, ZAxis)

for naxes in (1,2):
  print "Testing %s dimensions"%naxes
  for shape in product(*([sizes]*naxes)):
    print "  Testing shape %s"%str(shape)

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
    print "    # tests:", len(list(product(*slicelists)))
    for sl in product(*slicelists):
      print "    Testing slices %s"%repr(sl)

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

      # Check that the var is sliced as expected
      varTest(testname='blah', var=var, values=expected)

