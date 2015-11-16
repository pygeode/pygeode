# Issue 89 - climatology operators don't preserve variable metadata
# https://github.com/pygeode/pygeode/issues/89

def test_climat_metadata():
  from pygeode.tutorial import t2
  from pygeode.climat import monthlymean
  Temp = t2.Temp
  Temp.atts = {'units':'K'}
  Temp = monthlymean(Temp)
  assert 'units' in Temp.atts

