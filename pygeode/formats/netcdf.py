#TODO: when saving, convert NaN to another fill value?
# Pygeode interface for netCDF files

from ctypes import c_char_p
from pygeode.libhelper import load_lib
lib = load_lib('netcdf')
lib.nc_strerror.restype = c_char_p
del c_char_p

# Map netcdf types to numpy types
import numpy as np
numpy_type = {1:np.int8, 2:np.dtype('|S1'), 3:np.int16, 4:np.int32,
              5:np.float32, 6:np.float64}
del np

NC_MAX_NAME = 256
NC_MAX_DIMS = 1024
NC_MAX_VAR_DIMS = NC_MAX_DIMS

# note: int64 currently converted to float64, since netcdf doesn't have an int64?
nc_type = {'int8':1, 'string8':2, 'int16':3, 'int32':4,
           'float32':5, 'float64':6, 'int64':6}


get_att_f = {1:lib.nc_get_att_uchar, 2:lib.nc_get_att_text,
             3:lib.nc_get_att_short, 4:lib.nc_get_att_int,
             5:lib.nc_get_att_float, 6:lib.nc_get_att_double}

put_att_f = {1:lib.nc_put_att_uchar, 2:lib.nc_put_att_text,
             3:lib.nc_put_att_short, 4:lib.nc_put_att_int,
             5:lib.nc_put_att_float, 6:lib.nc_put_att_double}

# Read global/variable attributes, return a dictionary
def get_attributes (fileid, varid):
# {{{
  from ctypes import create_string_buffer, c_int, c_long, byref
  from pygeode.tools import point
  from numpy import empty
  natts = c_int()

  # Global attributes?
  if (varid < 0):
    ret = lib.nc_inq_natts(fileid, byref(natts))
    assert ret == 0

  # Variable attributes?
  else:
    ret = lib.nc_inq_varnatts (fileid, varid, byref(natts))
    assert ret == 0

  natts = natts.value

  atts = {}

  name = create_string_buffer(NC_MAX_NAME)
  vtype = c_int()
  size = c_long()

  # Loop over all attributes
  for n in range(natts):
    ret = lib.nc_inq_attname(fileid, varid, n, name);
    assert ret == 0
    ret = lib.nc_inq_att (fileid, varid, name, byref(vtype), byref(size))
    assert ret == 0
    # String?
    if vtype.value == 2:
      valstr = create_string_buffer(size.value)
      ret = get_att_f[vtype.value](fileid, varid, name, valstr);
      assert ret == 0
      value = valstr.value
    else:
      valnp = empty([size.value], numpy_type[vtype.value])
      ret = get_att_f[vtype.value](fileid, varid, name, point(valnp))
      assert ret == 0
      value = valnp
      if value.size == 1: value = value[0]

    atts[name.value] = value

  return atts
# }}}

# Write global/variable attributes from a dictionary
def put_attributes (fileid, varid, atts):
# {{{
  from numpy import asarray
#  from ctypes import c_long
  from pygeode.tools import point
  from warnings import warn
  for name, value in atts.iteritems():
    # String?
    if isinstance(value, str):
      vtype = 2
      ret = put_att_f[vtype](fileid, varid, name, len(value), value)
      assert ret == 0
    else:
      oldvalue = value
      value = asarray(value)
      # Numpy likes to use int64's a lot - force the arrays back down to int32?
      if isinstance(oldvalue,int) and oldvalue >= -2147483648 and oldvalue <= 2147483647:
        value = asarray(value, dtype='int32')
      # Drop unsupported data types
      if value.dtype.name not in nc_type:
        if value.dtype.name.startswith('string'):
          warn ("no support for writing attributes containing an array of strings", stacklevel=3)
        warn ("skipping attribute %s = %s  (unsupported type %s)"%(name,value,value.dtype.name), stacklevel=3)
        return
      # Scalar?
      if value.shape == (): value = value.reshape([1])
      vtype = nc_type[value.dtype.name]
      # Get the dtype again, but this time it should be compatible with the function we're writing to
      # (in case there is an implicit cast involved, i.e. int64's need to be cast to something else for netcdf)
      dtype = numpy_type[vtype]
      value = asarray(value, dtype=dtype)
      ret = put_att_f[vtype](fileid, varid, name, vtype, len(value), point(value))
      assert ret == 0
# }}}

# Load some values from the file
# Return a numpy array
# TODO: allow non-contiguous reads?
def load_values (fileid, varid, vartype, start, count, out=None):
  # Map the netCDF types to a numpy type
  # NOTE: character type not supported
  from ctypes import c_long
  from pygeode.tools import point
  import numpy as np
  if out is None: out = np.empty(count, dtype = numpy_type[vartype])
  f = {1:lib.nc_get_vara_uchar, 2:lib.nc_get_vara_text, 3:lib.nc_get_vara_short,
       4:lib.nc_get_vara_int, 5:lib.nc_get_vara_float,
       6:lib.nc_get_vara_double}
  A = c_long * len(start)
  _start = A(*start)
  _count = A(*count)
  ret = f[vartype](fileid, varid, _start, _count, point(out))
  if ret != 0: raise IOError, lib.nc_strerror(ret)
  return out

# Simple file object - just holds the file id
# allows the file to be properly closed during cleanup
# (i.e., when no more references to the file exist)
class NCFile:
# {{{
  import ctypes  # import here, so we don't lose the ctypes module during cleanup
  def __init__ (self, filename):
    from ctypes import c_int
    self.filename = filename
    self.fileid = c_int(-1)
    self.lib = lib
  def __del__(self): 
    if self is not None: self.close()
  def open(self):
    from ctypes import c_int, byref
    if self.fileid.value == -1:
      mode = c_int(0)  # 0 = read
      ret = self.lib.nc_open(self.filename, mode, byref(self.fileid))
      if ret != 0: raise IOError, self.lib.nc_strerror(ret)
  def opened(self): return self.fileid.value > -1
  def close(self):
#    from ctypes import c_int
    if self.fileid.value != -1:
      ret = self.lib.nc_close(self.fileid)
      if ret != 0: raise IOError, self.lib.nc_strerror(ret)
    self.fileid = self.ctypes.c_int(-1)  # use class-level ctypes reference to avoid errors during cleanup
  def __enter__(self): return self
  def __exit__(self): self.close()

# }}}


# A netcdf variable
from pygeode.var import Var
class NCVar(Var):
  # Read variable info (name, dimension ids, attributes)
  # NOTE: this is an incomplete init - we don't have axis information yet
  # Once dimensions are loaded, and mapped to appropriate variables, then we can finish the Var.__init__
  def __init__(self, f, varid, namemap):
  # {{{
    from ctypes import c_int, byref, create_string_buffer
    from warnings import warn

    self.f = f
    self.varid = varid

    assert f.fileid.value != -1

    name = create_string_buffer(NC_MAX_NAME+1)
    vtype = c_int()
    ndims = c_int()
    dimids = (c_int * NC_MAX_VAR_DIMS)()
    natts = c_int()
    ier = lib.nc_inq_var (f.fileid, varid, name, byref(vtype), byref(ndims), dimids, byref(natts))
    assert ier == 0

    self.nc_name = nc_name = name.value
    # Change the name?
    self.name = name = namemap[nc_name] if nc_name in namemap else nc_name
    self.vtype  = vtype = vtype.value
    self.dimids = [dimids[j] for j in range(ndims.value)]

    # Load attributes
    atts = get_attributes (f.fileid, varid)
    self.atts = atts

  # }}}

  # Finish the initialization, once the axes are derived.
  # (axes are a global list of all axes, corresponding to each dimension in the file)
  def finish_init (self, axes):
    from pygeode.var import Var
    dtype = numpy_type[self.vtype]

    axes = [axes[i] for i in self.dimids] # select only the axes that we need for this var
    Var.__init__(self, axes, dtype=dtype)


  #TODO: a more general (non-contiguous) read routine
  def getvalues (self, start, count):
  # {{{
    import numpy as np
    already_opened = self.f.opened()
    if not already_opened: self.f.open()
    # allocate space for loading the data
    out = np.empty(count, numpy_type[self.vtype])
    load_values (self.f.fileid, self.varid, self.vtype, start, count, out)
    if not already_opened: self.f.close()

    return out
  # }}}
del Var

# A netcdf dimension
def nc_dim (f, dimid, vardict, dimtypes, namemap):
  from pygeode.axis import Axis, NamedAxis
  from pygeode.var import copy_meta
  from pygeode.tools import make_axis
  from ctypes import c_int, c_long, byref, create_string_buffer
  import numpy as np
  assert f.fileid.value != -1

  name = create_string_buffer(NC_MAX_NAME+1)
  length = c_long()
  ret = lib.nc_inq_dim (f.fileid, dimid, name, byref(length))
  assert ret == 0
  name = name.value
  length = length.value

  # use values and attributes from corresponding variable (if one exists)   
  if name in vardict: 
    # if a corresponding variable exists (i.e. same name)    
    var = vardict.pop(name) # get variable and remove it from dictionary
    # A function for getting the values (if we need to)
    valfunc = lambda: load_values(f.fileid, var.varid, var.vtype, [0], [length])
    atts = var.atts    
  else: 
    # if no corresponding variable exists, assume indices
    valfunc = lambda: np.arange(length)
    atts = {}    
  
  # select axis type for this particular axis
  if name in dimtypes:
    dt = dimtypes[name]
  else: dt = None
  
  # change the name according to namemap
  if name in namemap: name = namemap[name]
      
  return make_axis(name, dt, valfunc, atts, length)


def open(filename, dimtypes = {}, namemap = {},  varlist = []):
  ''' open (filename, [dimtypes = {}, namemap = {}, dimmap = {},  varlist = [] ])

  Returns a Dataset of PyGeode variables contained in the specified files. The axes of the 
  variables are created from the dimensions of the NetCDF file. NetCDF variables in the file that do
  not correspond to dimensions are imported as PyGeode variables.

  filename - NetCDF file to open
  dimtypes - a dictionary mapping dimension names to axis classes. The keys should be axis names
              as defined in the NetCDF file; values should be one of:
              1) an axis instance, 
              2) an axis class, or 
              3) a tuple of an axis class and a dictionary with keyword arguments to pass 
                to that axis' constructor              
              If no dictionary is included, an attempt is made to automatically identify the axis 
              types.
  namemap - an optional dictionary to map NetCDF variable names (keys) to PyGeode variable names
            (values); also works for axes/dimensions
  varlist - a list containing the variables that should be loaded into the data set (if the list is
            empty, all NetCDF variables will be loaded)
  Note: The identifiers used in varlist and dimtypes are the original names used in the NetCDF file, 
        not the names given in namemap.'''

  from os.path import exists
  if not filename.startswith('http://'):
    assert exists(filename)

  from ctypes import c_int, byref, create_string_buffer
  from numpy import empty, arange
  from pygeode.formats.cfmeta import decode_cf

  f = NCFile(filename)
  f.open()
  try:
    fileid = f.fileid

    # Get variable info (name, type, dimensions)
    nvars = c_int()
    ret = lib.nc_inq_nvars(fileid, byref(nvars))
    assert ret == 0
    nvars = nvars.value

    # Partial construction of variable objects, storage in dictonary
    # (we need to explicitly invoke Var.__init__ once we have axis information)    
    vardict = dict([[var.nc_name, var] for var in [NCVar(f,i,namemap) for i in range(nvars)]])    
    
    # Get the number of dimensions, and their length
    ndims = c_int()
    ret = lib.nc_inq_ndims(fileid, byref(ndims))
    assert ret == 0
    ndims = ndims.value
    
    # Get the dimension values, and represent the dimensions as pygeode axes
    dims = [nc_dim(f, i, vardict, dimtypes, namemap) for i in range(ndims)]
    
    # create list from vardict, only keep requested variables (if list is provided) 
    if varlist: 
      vars = [var for var in vardict.itervalues() if var.nc_name in varlist]
    else:
      vars = vardict.values()
     
    # Finish constructing the variables, now that we have/removed all the axes
    for var in vars: var.finish_init(dims)    

    # Now, create a dataset to contain all these variables
    # (and do any translation from cf metadata)
    dataset = decode_cf(vars)

    # Store global attributes in the dataset
    # (variable attributes are handled in the NC_Var initializer
    dataset.atts = get_attributes (fileid, -1)

  finally:
    f.close()

  return dataset

# }}}


# Write a dataset to netcdf
def save (filename, in_dataset, version=3, compress=False):
# {{{
  from ctypes import c_int, c_long, byref
  from pygeode.view import View
  from pygeode.tools import combine_axes, point
  from pygeode.axis import Axis
  import numpy as np
  from pygeode.progress import PBar, FakePBar
  from pygeode.formats import cfmeta

  assert isinstance(filename,str)

  # Version?
  if compress: version = 4
  assert version in (3,4)

  dataset = cfmeta.encode_cf(in_dataset)

  fileid = c_int()

  vars = list(dataset.vars)
  # The output axes
  axes = combine_axes(v.axes for v in vars)

  # Include axes in the list of vars (for writing to netcdf)
  vars.extend(axes)

  # Variables (and axes) must all have unique names
  assert len(set([v.name for v in vars])) == len(vars), "vars must have unique names: %s"% [v.name for v in vars]

  # Functions for writing entire array
  allf = {1:lib.nc_put_var_uchar, 2:lib.nc_put_var_text, 3:lib.nc_put_var_short,
       4:lib.nc_put_var_int, 5:lib.nc_put_var_float,
       6:lib.nc_put_var_double}

  # Functions for writing chunks
  chunkf = {1:lib.nc_put_vara_uchar, 2:lib.nc_put_vara_text, 3:lib.nc_put_vara_short,
       4:lib.nc_put_vara_int, 5:lib.nc_put_vara_float,
       6:lib.nc_put_vara_double}

  # Create the file
  if version == 3:
    ret = lib.nc_create (filename, 0, byref(fileid))
    if ret != 0: raise IOError, lib.nc_strerror(ret)
  elif version == 4:
    ret = lib.nc_create (filename, 0x1000, byref(fileid))  # 0x1000 = NC_NETCDF4
    if ret != 0: raise IOError, lib.nc_strerror(ret)
  else: raise Exception

  # Define the dimensions
  dimids = [None] * len(axes)
  for i,a in enumerate(axes):
    dimids[i] = c_int()
    ret = lib.nc_def_dim (fileid, a.name, c_long(len(a)), byref(dimids[i]))
    assert ret == 0, lib.nc_strerror(ret)

  # Define the variables (including axes)
  chunks = [None] * len(vars)
  varids = [None] * len(vars)
  for i, var in enumerate(vars):
    t = nc_type[var.dtype.name]
    # Generate the array of dimension ids for this var
    d = [dimids[list(axes).index(a)] for a in var.axes]
    # Make it C-compatible
    d = (c_int * var.naxes)(*d)
    varids[i] = c_int()
    ret = lib.nc_def_var (fileid, var.name, t, var.naxes, d, byref(varids[i]))
    assert ret == 0, lib.nc_strerror(ret)
    # Compress the data? (only works for netcdf4 or (higher?))
    if compress:
      ret = lib.nc_def_var_deflate (fileid, varids[i], 1, 1, 2)
      assert ret == 0, lib.nc_strerror(ret)

  # Write the attributes

  # global attributes
  put_attributes (fileid, -1, dataset.atts)

  # variable attributes
  for i, var in enumerate(vars):
    # modify axes to be netcdf friendly (CF-compliant, etc.)
    put_attributes (fileid, varids[i], var.atts)

  # Don't pre-fill the file
  oldmode = c_int()
  ret = lib.nc_set_fill (fileid, 256, byref(oldmode))
  assert ret == 0, "can't set fill mode"
  # Finished defining the variables, about to start writing the values
  ret = lib.nc_enddef (fileid)
  assert ret == 0, "error leaving define mode: %d"%ret

  # Relative progress of each variable
  sizes = [v.size for v in vars]
  prog = np.cumsum([0.]+sizes) / np.sum(sizes) * 100

#  print "Saving '%s':"%filename
  pbar = PBar(message="Saving '%s':"%filename)
#  pbar = FakePBar()
  # Write the data
  for i, var in enumerate(vars):
    t = nc_type[var.dtype.name]
    dtype = numpy_type[t]

#    print 'writing', var.name

    # number of actual variables (non-axes) for determining our progress
    N = len([v for v in vars if not isinstance(v,Axis)])
    varpbar = pbar.subset(prog[i], prog[i+1])

    views = list(View(var.axes).loop_mem())
    for j,v in enumerate(views):

      vpbar = varpbar.part(j, len(views))
#      print '???', repr(str(v))

      # Should always be slices (since we're looping over whole thing contiguously?)
      for sl in v.slices: assert isinstance(sl, slice)
      for sl in v.slices: assert sl.step in (1,None)

      start = [sl.start for sl in v.slices]
      count = [sl.stop - sl.start for sl in v.slices]

      start = (c_long*var.naxes)(*start)
      count = (c_long*var.naxes)(*count)

      if isinstance(var, Axis):
        assert len(start) == len(count) == 1
        data = var.values
        data = data[start[0]:start[0]+count[0]] # the above gives us the *whole* axis,
                                                # but under extreme conditions we may be looping over smaller pieces
        vpbar.update(100)
      else: data = v.get(var, pbar=vpbar)

      # Ensure the data is stored contiguously in memory
      data = np.ascontiguousarray(data, dtype=dtype)
      ret = chunkf[t](fileid, varids[i], start, count, point(data))
      assert ret == 0, "error writing var '%s' to netcdf, code %d"%(var.name,ret)


  # Finished
  lib.nc_close(fileid)

  # Return a function stub for reloading the saved data
  from pygeode.var import Var
  if isinstance(in_dataset, Var):
    return lambda : open(filename).vars[0]
  else: return lambda : open(filename)


# }}}


