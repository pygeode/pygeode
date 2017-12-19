# Issue 65 - Allow Var objects for auxarrays
# https://github.com/pygeode/pygeode/issues/65

def test_auxarray_var_arg():
  from pygeode.axis import ZAxis, Hybrid
  from pygeode.var import Var
  import numpy as np
  zaxis = ZAxis(list(range(10)))
  A = [0., 10., 20., 30., 40., 50., 50., 50., 50., 40.]
  B = np.linspace(0,1,10)

  # Try passing A and B as arrays.
  # This should work already.
  eta = Hybrid(values=list(range(10)), A=A, B=B)
  assert 'A' in eta.auxarrays and 'B' in eta.auxarrays

  A = Var(axes=[zaxis], values=A)
  B = Var(axes=[zaxis], values=B)

  # Try again with Var object arguments.
  eta = Hybrid(values=list(range(10)), A=A, B=B)
  assert 'A' in eta.auxarrays and 'B' in eta.auxarrays


