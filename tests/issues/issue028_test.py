# Issue 28 - can't mix complex scalars with PyGeode variables

def test_issue028():
  from pygeode.var import Var
  from pygeode.axis import Lat

  # Create a mock variable
  lat = Lat([-45.])
  var = Var(axes=[lat], values=[123.])

  # Try scaling it by a complex number
  var2 = var * 1j
  assert var2.dtype.name == 'complex128'
