# Issue 134 - Packed variables are rescaled incorrectly (multiple times) by netcdf4 module
# https://github.com/pygeode/pygeode/issues/134

def test_write_netcdf_packed():
  import pygeode as pyg
  import numpy as np

  N = 4000
  lin = np.linspace(300, 301, N)

  # Create a variable and test reading from it
  t = pyg.yearlessn(N)
  v = pyg.Var((t,), name = 'var', values = lin)

  # Write to netcdf
  pyg.save('issue134_test_1.nc', v, format='netcdf', pack=True, compress=True)
  pyg.save('issue134_test_2.nc', v, format='netcdf4', pack=True, compress=True)

def test_read_netcdf():
  import pygeode as pyg
  import numpy as np

  N = 4000
  lin = np.linspace(300, 301, N)

  # read from netcdf and access data
  d1 = pyg.open('issue134_test_1.nc', format='netcdf')
  assert np.allclose(d1.var[:], lin)

  d2 = pyg.open('issue134_test_2.nc', format='netcdf')
  assert np.allclose(d2.var[:], lin)

def test_read_netcdf4():
  import pygeode as pyg
  import numpy as np

  N = 4000
  lin = np.linspace(300, 301, N)

  # read from netcdf and access data
  d1 = pyg.open('issue134_test_1.nc', format='netcdf4')
  assert np.allclose(d1.var[:], lin)

  d2 = pyg.open('issue134_test_2.nc', format='netcdf4')
  assert np.allclose(d2.var[:], lin)
