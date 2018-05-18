# Pygeode interface for HDF4 files
#TODO: bring out EOSGRID stuff into a separate module?

from pygeode.libhelper import load_lib
try:
  lib1 = load_lib('df', Global=True)  # so we have symbol 'error_top'
  lib = load_lib('mfhdf')
except OSError, e:
  print 'Failed to load HDF libraries; no HDF support available.'

del load_lib

import numpy as np
numpy_type = {20:np.int8, 21:np.uint8, 22:np.int16, 23:np.uint16,
              24:np.int32, 25:np.uint32, 5:np.float32, 6:np.float64}
del np

hdf_type = {'int8':20, 'uint8':21, 'int16':22, 'uint16':23,
            'int32':24, 'uint32':25, 'float32':5, 'float64':6}

# Return an attribute dictionary given an HDF id (and expected # of attributes)
def get_attributes (obj_id, natts):
  from ctypes import create_string_buffer, c_long, byref
  import numpy as np
  from pygeode.tools import point
  atts = {}
  for i in range(natts):
    name = create_string_buffer(256)
    type = c_long()
    count = c_long()
    ret = lib.SDattrinfo(obj_id, i, name, byref(type), byref(count))
    assert ret == 0
    # Can only handle strings (type=3 or 4?) for now.
    # Note: should find HDF4 files that actually have numerical attributes before doing this
    name = name.value
    count = count.value
    type = type.value
    if type in (3,4):
      value = create_string_buffer(count)
      ret = lib.SDreadattr(obj_id, i, value)
      assert ret == 0
      value = value.value
    else:
      value = np.empty([count], dtype=numpy_type[type])
      ret = lib.SDreadattr(obj_id, i, point(value))
      assert ret == 0
      if len(value) == 1: value = value[0]
    atts[name] = value
  return atts

# Get information about a dimension.
def get_dim_info (dim_id):
  from ctypes import create_string_buffer, c_long, byref
  import numpy as np
  from pygeode.tools import point
  name = create_string_buffer(256)
  size = c_long()
  type = c_long()
  natts = c_long()
  ret = lib.SDdiminfo(dim_id, name, byref(size), byref(type), byref(natts))
  assert ret == 0
  name = str(name.value.decode())
  size = size.value
  type = type.value
  natts = natts.value

  return name, size, type, natts

# Read some data
#(adapted from pygeode netcdf module)
def load_values (sds_id, start, count, out):
  import numpy as np
  from ctypes import c_int
  from pygeode.tools import point
  A = c_int * len(start)
  _start = A(*start)
  _stride = A(*([1]*len(start)))
  _count = A(*count)
  ret = lib.SDreaddata(sds_id, _start, _stride, _count, point(out))
  assert ret == 0, 'HDF4 read error: SDreaddata returned code %d'%ret
  return out

# HDF4 file wrapper
# (allows the file to be cleanly closed when no longer referenced)
# (for read-only access)
class HDF4_File:
  def __init__ (self, filename):
    from os.path import exists
    assert exists(filename), "file '%s' does not exist!"%filename
    self.lib = lib  # so we can reference the HDF4 library on cleanup
    self.filename = filename
    self.sd_id = lib.SDstart (filename, 1)
    assert self.sd_id > 0, 'error opening %s.  HDF4 Error code %d'%(filename,self.sd_id)
  def __del__ (self):
    if self.sd_id > 0:
      ret = self.lib.SDend (self.sd_id)
      assert ret == 0

# HDF-4 scientific data set
# (intermediate structure to help in constructing a pygeode Var)
class HDF4_SD:
  def __init__(self, f, i):
    from ctypes import create_string_buffer, c_long, byref
    sds_id = lib.SDselect (f.sd_id, i)

    name = create_string_buffer(256)
    rank = c_long()
    dimsizes = (c_long * 256)()
    data_type = c_long()
    natts = c_long()
    ret = lib.SDgetinfo (sds_id, name, byref(rank), dimsizes,
                         byref(data_type), byref(natts))
    assert ret == 0
    self.f = f
    self.sds_id = sds_id
    self.name = name.value
    self.rank = rank.value
    self.shape = dimsizes[:rank.value]
    self.type = data_type.value
    self.natts = natts.value

    self.dimids = [lib.SDgetdimid(sds_id, d) for d in range(rank.value)]

    self.iscoord = (lib.SDiscoordvar(sds_id))
    #doesn't seem to be used (at least for MOPITT data)
#    ret = lib.SDgetfillvalue(sds_id, VOIDP fill_value)


  def __str__(self):
    info = self.sds_id, self.name, self.rank, self.shape, self.type, self.natts, self.dimids
    return ' '.join(str(i) for i in info)

# HDF-4 pygeode variable
from pygeode.var import Var
class HDF4_Var (Var):

  def __init__(self, sd, axes):
    self.sd = sd
    self.name = sd.name
    #  attributes
    atts = get_attributes (sd.sds_id, sd.natts)
#    if len(atts) > 0: self.atts = atts
#    else: atts = None
    Var.__init__(self, axes, numpy_type[sd.type], atts=atts)

  def getvalues (self, start, count):
    import numpy as np
    out = np.empty(count, self.dtype)
    try:
      load_values (self.sd.sds_id, start, count, out)
#    except Exception as e:
    except Exception, e:
      args = list(e.args)
      args[0] = "error reading file '%s', var '%s' --- "%(self.sd.f.filename, self.sd.name) + args[0]
      e.args = tuple(args)
      raise
    return out


def open (filename, value_override = {}, dimtypes = {}, namemap = {},  varlist = [], cfmeta = True):
  from numpy import empty
  from ctypes import c_long, byref
  from pygeode.axis import DummyAxis
  from pygeode.dataset import asdataset
  from pygeode.formats import finalize_open

  f = HDF4_File (filename)

  num_datasets = c_long()
  num_global_attrs = c_long()
  ret = lib.SDfileinfo (f.sd_id, byref(num_datasets), byref(num_global_attrs))
  assert ret == 0
  num_datasets = num_datasets.value
  num_global_attrs = num_global_attrs.value
  global_atts = get_attributes(f.sd_id, num_global_attrs)

  # Get the HDF vars
  SD_arr = [None] * num_datasets
  for i in range(num_datasets):
    SD_arr[i] = HDF4_SD(f, i)

  # If there are 2 vars of the name XXXX and XXXX:EOSGRID, then
  # ignore the first one and use the latter one.
  # (Based some some GMAO files from the IPY dataset)
  SD_arr = [sd for sd in SD_arr
            if sd.name.endswith(':EOSGRID')
            or not any(sd2.name == sd.name+':EOSGRID' for sd2 in SD_arr)
           ]

  # Find any 'axes'
  # (look for unique 1D vars which contain a particular dimension id)
  sd_1d = [sd for sd in SD_arr if sd.rank == 1]
  # Determine which dimensions map to a unique 1D array
  dimids = [sd.dimids[0] for sd  in sd_1d]
  dimsds = [s for s in sd_1d if dimids.count(s.dimids[0]) == 1 or s.iscoord == 1]

  # Load axis values
  for s in dimsds:
    s.values = empty(s.shape, numpy_type[s.type])
    load_values (s.sds_id, [0], s.shape, s.values)

  #for s in dimsds: print s; print s.values

  # Create axis objects
  from pygeode.axis import NamedAxis
  axes = [None] * len(dimsds)
  for i,s in enumerate(dimsds):
    # Append attributes for the axis
    atts = get_attributes (s.sds_id, s.natts)
#    if len(atts) > 0: axes[i].atts = atts
    axes[i] = NamedAxis (s.values, s.name, atts=atts)

  # Reference axes by dimension ids
  axis_lookup = {}
  for i,a in enumerate(axes): axis_lookup[dimids[i]] = a

  # Add dummy axes for dimensions without coordinate info.
  for s in SD_arr:
    for d in s.dimids:
      if d not in axis_lookup:
        dimname, dimsize, dimtype, dim_natts = get_dim_info(d)
        axis_lookup[d] = DummyAxis(dimsize,dimname)

  # Create var objects
  vars = [None]*len(SD_arr)
  for i,s in enumerate(SD_arr):
    axes = [axis_lookup[d] for d in s.dimids]
    vars[i] = HDF4_Var(s, axes)
  vars = [v for v in vars if v.sd not in dimsds]

  # Return a dataset
  d = asdataset(vars)
  d.atts = global_atts

  return finalize_open(d, dimtypes, namemap, varlist, cfmeta)

