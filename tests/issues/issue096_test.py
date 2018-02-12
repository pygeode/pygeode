# Issue 096 - Make a writeable copy of the data in Var.get()
# https://github.com/pygeode/pygeode/issues/96

def test_writeable():
  from pygeode.tutorial import t1
  data = t1.Temp.get()
  data -= 273.15

  # Also test the slice notation.
  data = t1.Temp[...]
  data -= 273.15
