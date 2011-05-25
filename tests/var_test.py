# some basic variable operation tests
from nose import with_setup

import pygeode as pyg
import numpy as np

def test_name():
  ax = pyg.NamedAxis(np.arange(10), 'axis')
  var = pyg.Var((ax, ), values=np.arange(10), name='name')
  assert var.name == 'name'
  
def test_values():
  ax = pyg.NamedAxis(np.arange(10), 'axis')
  var = pyg.Var((ax, ), values=np.arange(10), name='name')
  assert all(var[:] == np.arange(10))
