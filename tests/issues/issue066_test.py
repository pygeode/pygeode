# Issue 66 - Don't allow computing a mean over a dimension with length 0

def test_issue066():
  from pygeode.tutorial import t1
  x = t1.Temp.slice[:0,:]
  assert len(x.lat) == 0

  try:
    y = x.mean('lat')
    raise Exception("Should not be allowed to take a mean over this axis!")
  except ValueError:
    pass

