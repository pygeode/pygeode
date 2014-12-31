# Issue 61 - common_dict should drop attributes that don't appear in all variables

def test_issue061():
  from pygeode.tutorial import t1
  from pygeode.var import Var

  # Construct 3 variables, with some attributes that are identical and some
  # that are different.
  x = Var(t1.axes, atts=dict(standard_name='x',units='deg C'), dtype=float)
  y = Var(t1.axes, atts=dict(standard_name='y',units='deg C'), dtype=float)
  z = Var(t1.axes, atts=dict(standard_name='z',units='deg C'), dtype=float)

  # Apply a ufunc operator on these variables, to get a new variable
  w = x*y*z

  # Make sure the standard_name is removed.
  assert w.atts == dict(units='deg C')

