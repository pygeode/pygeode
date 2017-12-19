# Issue 73 - Implement forward  difference operator
# https://github.com/pygeode/pygeode/issues/73

def do_forward_difference(v, n):
  import numpy as np
  print('Testing for n=%d'%n)
  d = v.diff('time',n=n)
  assert len(d.time) == len(v.time)-n
  expected_values = np.diff(v[:,10,10,10], axis=0, n=n)
  computed_values = d[:,10,10,10]
  assert np.all(expected_values == computed_values), (expected_values, computed_values)
  print('n=%d passed.'%n)

def test_forward_difference():
  from pygeode.tutorial import t2
  do_forward_difference(t2.Temp,n=1)
  do_forward_difference(t2.Temp,n=2)
  do_forward_difference(t2.Temp,n=3)

