# Issue 49 - Slicing behaviour for Var objects

def test_issue049():
  from pygeode.tutorial import t1, t2

  # Check that degenerate axes get squeezed
  assert t1.Temp[2,:].ndim == 1

  # Check that ellipses are handled properly
  assert t2.Temp[0,...,0].shape == (20,31)

