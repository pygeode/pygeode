import unittest
import numpy as np
import os
import pygeode as pyg
from pygeode.formats import netcdf as nc

def assertSameVar(v1, v2):
   assert v1.name == v2.name
   assert v1.shape == v2.shape
   assert all([a1 == a2 for a1, a2 in zip(v1.axes, v2.axes)])
   assert (v1[:] == v2[:]).all()

def varTest(testname='', var=None, name = None, serialize = False,\
         shape = None, axes = None, dtype = None, values = None):
   # Validate inputs; all tests must have a name and a variable
   assert testname is not ''
   assert var is not None
   formats = {'nc':nc}

   # Validate consistency of test outputs
   if axes is not None:    # Define shape from axes list if available
      s = tuple([len(a) for a in axes])
      if shape is None: shape = s   
      assert shape == s

   if values is not None:  # Check shape and dtype of output array
      if shape is None: shape = values.shape
      assert values.shape == shape
      if dtype is None: dtype = values.dtype
      assert values.dtype == dtype

   # Build test class
   class tc(unittest.TestCase):
      def setUp(self):
         self.var = var
         self.name = name
         self.shape = shape
         self.axes = axes
         self.dtype = dtype
         self.values = values

      if name is not None:
         def test_name(self):
            self.assertEqual(self.name, self.var.name)

      if dtype is not None:
         def test_dtype(self):
            self.assertEqual(self.dtype, self.var.dtype)

      if shape is not None:
         def test_shape(self):
            self.assertEqual(len(self.shape), self.var.naxes)
            self.assertEqual(self.shape, self.var.shape)

         def test_shape_data(self):
            data = self.var.get()
            self.assertEqual(len(self.shape), len(data.shape), 'Wrong number of dimensions.')
            self.assertEqual(self.shape, data.shape)

      if axes is not None:                # Test axes against expected axes
         def test_axes(self):
            for i in range(len(axes)):
               self.assertEqual(self.axes[i], self.var.axes[i])

      if values is not None:              # Test variable data against expected output
         assert values.shape == shape
         def test_values(self):
            data = self.var.get()
            self.assertTrue(np.allclose(data, self.values))

      if serialize:  
         # Write variable to disk, read it back in, test for equality
         def test_serialize(self):
            for k, f in formats.items():
               fname = '%s.%s' % (testname, k)

               f.save(fname, [self.var])    
               d = f.open(fname)         
               assertSameVar(d.vardict[self.name], self.var)
               
               os.remove(fname)              
            

   tc.__name__ = testname
   return tc

ax1 = pyg.StandardTime(values=np.arange(365.), units='days', startdate={'year':2001})
ax2 = pyg.gausslat(32)
ax3 = pyg.Pres(np.arange(0, 100, 10.))

shape = (365, 32, 10)
data = np.random.randn(*shape)
ltwts = ax2.auxarrays['weights']
var = pyg.Var((ax1, ax2, ax3), values=data, name='var')

tv = varTest('1_Simple', var, \
         name = 'var', axes = (ax1, ax2, ax3), \
         values = data, serialize=True)
         
sl1 = varTest('slice_simple', var(i_time=(0, 5), i_lat=(0, 5), i_pres=(0, 5)), \
         shape = (5, 5, 5), \
         axes = (ax1(i_time=(0, 5)), ax2(i_lat=(0, 5)), ax3(i_pres=(0, 5))), \
         name = 'var', \
         values = data[:5, :5, :5])

sl2 = varTest('slice_stride', var(i_time=(1, -1, 4)), \
         values = data[1:-1:4, :, :])

sl3 = varTest('slice_negative_stride', var(i_time=(2, -5, -3)), \
         values = data[2:-5:-3, :, :])

bc1 = varTest('broadcast1', ax1*ax2, \
         axes = (ax1, ax2), \
         values = ax1.values.reshape(-1, 1) * ax2.values.reshape(1, -1))
      
bc2 = varTest('broadcast2', ax1(time=(0,100)) + ax1(time=(50,100)), \
         axes = (ax1(time=(50,100)),), \
         values = 2*np.arange(50,101,dtype=ax1.dtype))

lnrm = np.sum(ltwts[-5:])

pf1 = varTest('mean', var(m_lat=(60, 90)), \
         axes = (ax1, ax3), \
         values = np.sum(data[:, -5:, :] * ltwts[-5:].reshape(1, 5, 1), 1)/lnrm)

pf2 = varTest('complement', var(n_lat=(60, 90), month=1, n_day=(4, 31)), \
         shape = (3, 27, 10), \
         axes = (ax1(i_time=(0,3)), ax2(i_lat=(0, 27)), ax3), \
         values = data[:3, :27, :])

pf3 = varTest('squeeze', var(si_lat=10), \
         shape = (365, 10), \
         axes = (ax1, ax3), \
         values = data[:, 10, :])
