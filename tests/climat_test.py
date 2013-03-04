# Test climatologies (and array partial sums)

import unittest
import numpy as np
from pygeode.var import Var
from pygeode.axis import Lat, Lon
from pygeode.timeaxis import ModelTime365
from pygeode.climat import Monthly, TimeOp, loopover

# Create a monthly sum operator for testing (better test for partial sums)
# Based on climat.Mean
class Sum(TimeOp):
  name_suffix2 = '_sum'

  def getview (self, view, pbar):
    from pygeode.tools import partial_sum
    import numpy as np

    ti = self.ti

    sum = np.zeros (view.shape, self.dtype)
    count = np.zeros (view.shape, dtype='int32')

    for slices, [data], bins in loopover (self.var, view, pbar):
      partial_sum (data, slices, sum, count, ti, bins)

    return sum

class monthlysum(Monthly,Sum): pass

class TestClimat(unittest.TestCase):

  def setUp(self):
#    values = np.random.rand(365,180,360)
    values = np.ones([365,180,360])
    time = ModelTime365(startdate=dict(year=2000,month=1),values=np.arange(365),units='days')
    lon = Lon(values=np.arange(360))
    lat = Lat(values=np.arange(180)-89.5)
    self.var = Var(axes=[time,lat,lon], values=values)

  def test_monthlysum (self):
    var = self.var
    var = monthlysum(var).get()
    self.assertEqual(var.shape[0], 12)
    monthly_max = [var[i,:,:].max() for i in range(12)]
    monthly_min = [var[i,:,:].min() for i in range(12)]
    self.assertEqual(monthly_max, monthly_min)
    self.assertEqual(monthly_max, [31,28,31,30,31,30,31,31,30,31,30,31])

if __name__ == '__main__': unittest.main()
