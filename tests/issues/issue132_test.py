# -*- coding: utf-8 -*-
# Issue 132 - Handling of non-ASCII characters in netcdf and netcdf4 breaks in Python 3
# https://github.com/pygeode/pygeode/issues/132

def test_write_netcdf_unicode_atts():
  import pygeode as pyg
  import numpy as np
  import sys

  # Create a variable and test reading from it
  lat = pyg.regularlat(64)
  v = pyg.Var((lat,), name = 'var', values = np.arange(64))
  if sys.version < '3':
    v.atts['Ann\u00e9e'] = '2000\u33af'
  else:
    v.atts['Année'] = '2000ʰ'
  ds = pyg.asdataset([v])

  # Write to netcdf
  pyg.save('issue132_test_1.nc', ds, format='netcdf', version=3)
  pyg.save('issue132_test_1b.nc', ds, format='netcdf', version=4)
  pyg.save('issue132_test_2.nc', ds, format='netcdf4')

def test_read_netcdf_nonascii_atts():
  import pygeode as pyg
  import sys
  # read from netcdf and access data
  d1 = pyg.open('issue132_test_1.nc', format='netcdf')
  print(d1.var.atts)
  if sys.version < '3':
    assert d1.var.atts['Ann\u00e9e'] == '2000\u33af'
  else:
    assert d1.var.atts['Année'] == '2000ʰ'

def test_read_netcdf4_nonascii_atts():
  import pygeode as pyg
  import sys
  # read from netcdf and access data
  d2 = pyg.open('issue132_test_2.nc', format='netcdf4')
  print(d2.var.atts)
  if sys.version < '3':
    assert d2.var.atts['Ann\u00e9e'] == '2000\u33af'
  else:
    assert d2.var.atts['Année'] == '2000ʰ'

