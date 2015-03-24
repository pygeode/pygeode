# Issue 17 - can't concatenate time axes with different units
# https://github.com/pygeode/pygeode/issues/17

# Common stuff

from pygeode.axis import concat
import numpy as np

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
  # Assuming the default units are 'days'.  Change this line as necessary.
  assert np.allclose(t.values * 24, [0,1,2,3,4,5,6,7,8])

# Test 2 - 365-day calendar
def test_365day():
  from pygeode.timeaxis import ModelTime365
  t1 = ModelTime365(values1, units='hours', startdate=startdate)
  t2 = ModelTime365(values2, units='minutes', startdate=startdate)
  t = concat ([t1, t2])
  # Assuming the default units are 'days'.  Change this line as necessary.
  assert np.allclose(t.values * 24, [0,1,2,3,4,5,6,7,8])

