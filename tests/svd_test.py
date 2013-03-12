# Test the SVD routine

import unittest
import numpy as np
from pygeode.svd import SVD
from pygeode.formats import netcdf

class TestSVD(unittest.TestCase):

  def setUp(self):
    from pygeode.tutorial import t2
    self.var1 = t2.Temp

  # Test the SVD of a variable with itself
  def test_self_svd (self):
    var1 = self.var1
    eof1, pc1, eof2, pc2 = SVD(var1,var1,num=1,subspace=1)
    netcdf.save("svd_test.nc", [eof1,pc1,eof2,pc2])
    self.assertTrue(np.allclose(eof1.get(),eof2.get()))
    self.assertTrue(np.allclose(pc1.get(),pc2.get()))

if __name__ == '__main__': unittest.main()
