# Test the EOF routine
# Based on the SVD test

import unittest
import numpy as np
from pygeode.eof import EOF
from pygeode.formats import netcdf

class TestEOF(unittest.TestCase):

  def setUp(self):
    from pygeode.tutorial import t2
    self.var = t2.Temp.squeeze(pres=500)

  # Test a degenerate EOF from the test module
  def test_self_eof (self):
    var = self.var
    eof, eig, pc = EOF(var,num=1)
    netcdf.save("eof_test.nc", [eof,eig,pc])
    std = (var - eof*eig*pc).stdev()
    rms = (var**2).mean()
    self.assertLess(std, rms*1E-3)

from cProfile import run
#if __name__ == '__main__': run("unittest.main()")
if __name__ == '__main__': unittest.main()

