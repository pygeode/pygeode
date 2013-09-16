# Issue 39 - Ufunc drops metadata

def test_issue039():
  from pygeode.tutorial import t1
  x = t1.Temp
  x.atts['test_att'] = 123
  x = x * 1
  assert 'test_att' in x.atts

