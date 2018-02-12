# Issue 093 - Add getaxis, hasaxis, and whichaxis to Dataset interface
# https://github.com/pygeode/pygeode/issues/93

def test_hasaxis():
  from pygeode.tutorial import t1, t2
  assert not t1.hasaxis('time')
  assert t2.hasaxis('time')

def test_getaxis():
  from pygeode.tutorial import t1, t2
  from pygeode.axis import TAxis
  from pygeode.timeaxis import Time
  try:
    time = t1.getaxis('time')
    raise Exception("Somehow got a time axis?")
  except KeyError: pass
  time = t2.hasaxis('time')

  assert t2.getaxis(0) is t2.getaxis('time')
  assert t2.getaxis(TAxis) is t2.getaxis('time')
  assert t2.getaxis(Time) is t2.getaxis('time')
