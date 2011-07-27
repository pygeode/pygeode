# Issue 17 - can't concatenate time axes with different units
# http://code.google.com/p/pygeode/issues/detail?id=17

# Common stuff

from pygeode.axis import concat

# Starting date
startdate = {'year':2000, 'month':1, 'day':1}

# Relative time segment - in hours
values1 = [0., 1., 2., 3.]

# Relative time segment - in minutes (continuing from first segment)
values2 = [240., 300., 360., 420., 480.]


# Test 1 - standard calendar
def test_standard():
  from pygeode.timeaxis import StandardTime
  t1 = StandardTime(values1, units='hours', startdate=startdate)
  t2 = StandardTime(values2, units='minutes', startdate=startdate)
  t = concat ([t1, t2])

# Test 2 - 365-day calendar
def test_365day():
  from pygeode.timeaxis import ModelTime365
  t1 = ModelTime365(values1, units='hours', startdate=startdate)
  t2 = ModelTime365(values2, units='minutes', startdate=startdate)
  t = concat ([t1, t2])

