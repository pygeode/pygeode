# Issue 23 - nansum doesn't actually use NANSumVar
# https://github.com/pygeode/pygeode/issues/23

def test_nan():
  from pygeode.axis import XAxis, ZAxis
  from pygeode.reduce import nansum, NANSumVar, WeightedNANSumVar
  x = XAxis([1,2,3,4])
  z = ZAxis([1,2,3])
  var = x*z

  sum1 = nansum(var, ZAxis, weights=False)
  assert isinstance(sum1, NANSumVar)
  sum2 = nansum(var, ZAxis, weights=z**2)
  assert isinstance(sum2, WeightedNANSumVar)

