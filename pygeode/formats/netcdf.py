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
              5:np.float32, 6:np.float64, 7:np.uint8, 8:np.uint16,
              9:np.uint32, 10:np.int64, 11:np.uint64}
del np

NC_MAX_NAME = 256
NC_MAX_DIMS = 1024
NC_MAX_VAR_DIMS = NC_MAX_DIMS

nc_type_v3 = {'int8':1, 'string8':2, 'int16':3, 'int32':4,
           'float32':5, 'float64':6, 'uint8':3, 'uint16':4,
           'uint32':4, 'int64':6, 'uint64':6}

nc_type_v4 = {'int8':1, 'string8':2, 'int16':3, 'int32':4,
           'float32':5, 'float64':6, 'uint8':7, 'uint16':8,
           'uint32':9, 'int64':10, 'uint64':11}

nc_type = {3:nc_type_v3, 4:nc_type_v4}

get_att_f = {1:lib.nc_get_att_schar, 2:lib.nc_get_att_text,
             3:lib.nc_get_att_short, 4:lib.nc_get_att_int,
             5:lib.nc_get_att_float, 6:lib.nc_get_att_double,
             7:lib.nc_get_att_uchar, 8:lib.nc_get_att_ushort,
             9:lib.nc_get_att_uint, 10:lib.nc_get_att_longlong,
            11:lib.nc_get_att_ulonglong}

put_att_f = {1:lib.nc_put_att_schar, 2:lib.nc_put_att_text,
             3:lib.nc_put_att_short, 4:lib.nc_put_att_int,
             5:lib.nc_put_att_float, 6:lib.nc_put_att_double,
             7:lib.nc_put_att_uchar, 8:lib.nc_put_att_ushort,
             9:lib.nc_put_att_uint, 10:lib.nc_put_att_longlong,
            11:lib.nc_put_att_ulonglong}

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
    assert ret == 0, lib.nc_strerror(ret)

  # Variable attributes?
  else:
    ret = lib.nc_inq_varnatts (fileid, varid, byref(natts))
    assert ret == 0, lib.nc_strerror(ret)

  natts = natts.value

  atts = {}

  name = create_string_buffer(NC_MAX_NAME)
  vtype = c_int()
  size = c_long()

  # Loop over all attributes
  for n in range(natts):
    ret = lib.nc_inq_attname(fileid, varid, n, name);
    assert ret == 0, lib.nc_strerror(ret)
    ret = lib.nc_inq_att (fileid, varid, name, byref(vtype), byref(size))
    assert ret == 0, lib.nc_strerror(ret)
    # String?
    if vtype.value == 2:
      valstr = create_string_buffer(size.value)
      ret = get_att_f[vtype.value](fileid, varid, name, valstr);
      assert ret == 0, lib.nc_strerror(ret)
      value = valstr.value
    else:
      valnp = empty([size.value], numpy_type[vtype.value])
      ret = get_att_f[vtype.value](fileid, varid, name, point(valnp))
      assert ret == 0, lib.nc_strerror(ret)
      value = valnp
      if value.size == 1: value = value[0]

    atts[name.value] = value

  return atts
# }}}

# Write global/variable attributes from a dictionary
def put_attributes (fileid, varid, atts, version):
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
      assert ret == 0, lib.nc_strerror(ret)
    else:
      oldvalue = value
      value = asarray(value)
      # Numpy likes to use int64's a lot - force the arrays back down to int32?
      if isinstance(oldvalue,int) and oldvalue >= -2147483648 and oldvalue <= 2147483647:
        value = asarray(value, dtype='int32')
      # Drop unsupported data types
      if value.dtype.name not in nc_type[version]:
        if value.dtype.name.startswith('string'):
          warn ("no support for writing attributes containing an array of strings", stacklevel=3)
        warn ("skipping attribute %s = %s  (unsupported type %s)"%(name,value,value.dtype.name), stacklevel=3)
        return
      # Scalar?
      if value.shape == (): value = value.reshape([1])
      vtype = nc_type[version][value.dtype.name]
      # Get the dtype again, but this time it should be compatible with the function we're writing to
      # (in case there is an implicit cast involved, i.e. int64's need to be cast to something else for netcdf)
      dtype = numpy_type[vtype]
      value = asarray(value, dtype=dtype)
      ret = put_att_f[vtype](fileid, varid, name, vtype, len(value), point(value))
      assert ret == 0, lib.nc_strerror(ret)

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
  f = {1:lib.nc_get_vara_schar, 2:lib.nc_get_vara_text, 3:lib.nc_get_vara_short,
       4:lib.nc_get_vara_int, 5:lib.nc_get_vara_float,
       6:lib.nc_get_vara_double, 7:lib.nc_get_vara_uchar,
       8:lib.nc_get_vara_ushort, 9:lib.nc_get_vara_uint,
      10:lib.nc_get_vara_longlong, 11:lib.nc_get_vara_ulonglong}
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
    if self.ctypes.c_int is not None:
      self.fileid = self.ctypes.c_int(-1)  # use class-level ctypes reference to avoid errors during cleanup
  def __enter__(self): return self
  def __exit__(self): self.close()

# }}}


# A generic dimension (has no attributes except a name and a length)
from pygeode.axis import DummyAxis
class NCDim (DummyAxis):
  # Make an axis from given fileid and dimid
  @staticmethod
  def from_id (f, dimid):
    from ctypes import create_string_buffer, c_long, byref
    name = create_string_buffer(NC_MAX_NAME+1)
    length = c_long()
    ret = lib.nc_inq_dim (f.fileid, dimid, name, byref(length))
    assert ret == 0, lib.nc_strerror(ret)

    name = name.value
    length = length.value
    return NCDim (length, name=name)

del DummyAxis

# constructor for the dims (wrapper for NCDim so it's only created once)
def makedim (f, dimid, dimdict={}):
  if (f,dimid) not in dimdict:
    dimdict[(f,dimid)] = NCDim.from_id(f,dimid)
  return dimdict[(f,dimid)]

# A netcdf variable
from pygeode.var import Var
class NCVar(Var):
# {{{
  # Read variable info (name, dimension ids, attributes)
  def __init__(self, f, varid):
  # {{{
    from pygeode.var import Var
    from ctypes import c_int, byref, create_string_buffer
    from warnings import warn

    self._f = f
    self._varid = varid

    assert f.fileid.value != -1

    name = create_string_buffer(NC_MAX_NAME+1)
    vtype = c_int()
    ndims = c_int()
    dimids = (c_int * NC_MAX_VAR_DIMS)()
    natts = c_int()
    ier = lib.nc_inq_var (f.fileid, varid, name, byref(vtype), byref(ndims), dimids, byref(natts))
    assert ier == 0

    self._vtype  = vtype = vtype.value
    dtype = numpy_type[vtype]
    self._dimids = dimids = [dimids[j] for j in range(ndims.value)]
    name = name.value

    # Load attributes
    atts = get_attributes (f.fileid, varid)

    # Axes (just generic dimensions right now, until some filter is applied)
    axes = [makedim(f,d) for d in dimids]

    Var.__init__(self, axes, name=name, dtype=dtype, atts=atts)

  # }}}

  #TODO: a more general (non-contiguous) read routine
  def getvalues (self, start, count):
  # {{{
    import numpy as np
    already_opened = self._f.opened()
    if not already_opened: self._f.open()
    # allocate space for loading the data
    out = np.empty(count, numpy_type[self._vtype])
    load_values (self._f.fileid, self._varid, self._vtype, start, count, out)
    if not already_opened: self._f.close()

    return out
  # }}}
# }}}

### internal data filters ###

# Override values from the source?
def override_values (dataset, value_override):
# {{{
  from warnings import warn
  import numpy as np
  from pygeode.var import Var, copy_meta
  vardict = {}
  for name, values in value_override.iteritems():
    if name not in dataset:
      warn ("var '%s' not found - values not overridden"%name, stacklevel=3)
      continue
    values = np.asarray(values)
    oldvar = dataset[name]
    assert values.shape == oldvar.shape, "bad shape for '%s'.  Expected %s, got %s"%(name,oldvar.shape,values.shape)
    var = Var(oldvar.axes, values=values)
    copy_meta (oldvar, var)
    vardict[name] = var
  dataset = dataset.replace_vars(vardict)
  return dataset
# }}}


# Find axes (1D vars with the same name as a dimension)
def dims2axes (dataset):
# {{{
  from pygeode.axis import NamedAxis
  from pygeode.var import copy_meta
  # Loop over current set of generic "dimensions"  
  replacements = {}
  for i,dim in enumerate(dataset.axes):
    # Do we have a Var with this name?
#    if dim.name in dataset:
    if any (var.name == dim.name for var in dataset.vars):
      # Get the var
      var = dataset[dim.name]
      if var.naxes != 1: continue  # abort if we have > 1 dimension
      # Turn it into a proper axis
      axis = NamedAxis (name=var.name, values=var.get())  # need the values pre-loaded for axes
      copy_meta (var, axis)
      replacements[dim.name] = axis
  dataset = dataset.replace_axes(axisdict=replacements)
  # Remove the axes from the list of variables
  dataset = dataset.remove(*replacements.keys())
  return dataset
# }}}

def open(filename, value_override = {}, dimtypes = {}, namemap = {},  varlist = [], cfmeta = True):
# {{{
  ''' open (filename, [value_override = {}, dimtypes = {}, namemap = {}, varlist = [] ])

  Returns a Dataset of PyGeode variables contained in the specified files. The axes of the 
  variables are created from the dimensions of the NetCDF file. NetCDF variables in the file that do
  not correspond to dimensions are imported as PyGeode variables.

  filename - NetCDF file to open
  value_override - an optional dictionary with replacement values for one or more variables.
           The only known use for this dictionary is to avoid loading in values from a severely
           scattered variable (such as a 'time' axis or other slowest-varying dimension).
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
  from ctypes import c_int, byref
  from pygeode.dataset import asdataset
  from pygeode.formats import finalize_open
  from pygeode.axis import Axis
  if not filename.startswith('http://'):
    assert exists(filename), 'File open failed. "%s" does not exist.' % filename


  # Read variable dimensions and metadata from the file
  f = NCFile(filename)
  f.open()
  try:
    fileid = f.fileid

    # Get number of variables
    nvars = c_int()
    ret = lib.nc_inq_nvars(fileid, byref(nvars))
    assert ret == 0, lib.nc_strerror(ret)

    nvars = nvars.value

    # Construct all the variables, put in a list
    vars = [NCVar(f,i) for i in range(nvars)]

    # Construct a dataset from these Vars
    dataset = asdataset(vars)
    dataset.atts = get_attributes (fileid, -1)

  finally:
    f.close()

  # Add the object stuff from dimtypes to value_override, so we don't trigger a
  # load operation on those dims.
  # (We could use any values here, since they'll be overridden again later,
  #  but we might as well use something relevant).
  value_override = dict(value_override)  # don't use  the default (static) empty dict
  for k,v in dimtypes.items():
    if isinstance(v,Axis):
      value_override[k] = v.values

  #### Filters to apply to the data ####

  # Override values from the source?
  if len(value_override) > 0:
    dataset = override_values(dataset, value_override)

  # Set up the proper axes (get coordinate values / metadata from a 1D variable
  # with the same name as the dimension)
  dataset = dims2axes(dataset)

  return finalize_open(dataset, dimtypes, namemap, varlist, cfmeta)
# }}}

#TODO: factor out cf-meta encoding and other processing steps
# Write a dataset to netcdf
def save (filename, in_dataset, version=3, pack=None, compress=False, cfmeta = True, unlimited=None):
# {{{
  from ctypes import c_int, c_long, byref
  from pygeode.view import View
  from pygeode.tools import combine_axes, point
  from pygeode.axis import Axis, DummyAxis
  import numpy as np
  from pygeode.progress import PBar, FakePBar
  from pygeode.formats import finalize_save
  from pygeode.dataset import asdataset

  assert isinstance(filename,str)

  dataset = finalize_save(in_dataset, cfmeta, pack)

  # Version?
  if compress: version = 4
  assert version in (3,4)

  fileid = c_int()

  vars = list(dataset.vars)
  # The output axes
  axes = combine_axes(v.axes for v in vars)

  # Include axes in the list of vars (for writing to netcdf).
  # Exclude axes which don't have any intrinsic values.
  vars = vars + [a for a in axes if not isinstance(a,DummyAxis)]
  #vars.extend(axes)

  # Variables (and axes) must all have unique names
  assert len(set([v.name for v in vars])) == len(vars), "vars must have unique names: %s"% [v.name for v in vars]

  if unlimited is not None:
    assert unlimited in [a.name for a in axes]

  # Functions for writing entire array
  allf = {1:lib.nc_put_var_schar, 2:lib.nc_put_var_text, 3:lib.nc_put_var_short,
       4:lib.nc_put_var_int, 5:lib.nc_put_var_float,
       6:lib.nc_put_var_double, 7:lib.nc_put_var_uchar,
       8:lib.nc_put_var_ushort, 9:lib.nc_put_var_uint,
      10:lib.nc_put_var_longlong, 11:lib.nc_put_var_ulonglong}

  # Functions for writing chunks
  chunkf = {1:lib.nc_put_vara_schar, 2:lib.nc_put_vara_text, 3:lib.nc_put_vara_short,
       4:lib.nc_put_vara_int, 5:lib.nc_put_vara_float,
       6:lib.nc_put_vara_double, 7:lib.nc_put_vara_uchar,
       8:lib.nc_put_vara_ushort, 9:lib.nc_put_vara_uint,
      10:lib.nc_put_vara_longlong, 11:lib.nc_put_vara_ulonglong}

  # Create the file
  if version == 3:
    ret = lib.nc_create (filename, 0, byref(fileid))
    if ret != 0: raise IOError, lib.nc_strerror(ret)
  elif version == 4:
    ret = lib.nc_create (filename, 0x1000, byref(fileid))  # 0x1000 = NC_NETCDF4
    if ret != 0: raise IOError, lib.nc_strerror(ret)
  else: raise Exception

  try:
    # Define the dimensions
    dimids = [None] * len(axes)
    for i,a in enumerate(axes):
      dimids[i] = c_int()
      if unlimited == a.name:
        ret = lib.nc_def_dim (fileid, a.name, c_long(0), byref(dimids[i]))
      else:
        ret = lib.nc_def_dim (fileid, a.name, c_long(len(a)), byref(dimids[i]))
      assert ret == 0, lib.nc_strerror(ret)

    # Define the variables (including axes)
    chunks = [None] * len(vars)
    varids = [None] * len(vars)
    for i, var in enumerate(vars):
      t = nc_type[version][var.dtype.name]
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
    put_attributes (fileid, -1, dataset.atts, version)

    # variable attributes
    for i, var in enumerate(vars):
      # modify axes to be netcdf friendly (CF-compliant, etc.)
      put_attributes (fileid, varids[i], var.atts, version)

    # Don't pre-fill the file
    oldmode = c_int()
    ret = lib.nc_set_fill (fileid, 256, byref(oldmode))
    assert ret == 0, "Can't set fill mode: %s (error %d)" % (lib.nc_strerror(ret), ret)
    # Finished defining the variables, about to start writing the values
    ret = lib.nc_enddef (fileid)
    assert ret == 0, "Error leaving define mode: %s (error %d)" % (lib.nc_strerror(ret), ret)

    # Relative progress of each variable
    sizes = [v.size for v in vars]
    prog = np.cumsum([0.]+sizes) / np.sum(sizes) * 100

  #  print "Saving '%s':"%filename
    pbar = PBar(message="Saving '%s':"%filename)
  #  pbar = FakePBar()
    # Write the data
    for i, var in enumerate(vars):
      t = nc_type[version][var.dtype.name]
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
        assert ret == 0, "Error writing var '%s' to netcdf: %s (error %d)" % (var.name, lib.nc_strerror(ret), ret)

  finally:
    # Finished
    lib.nc_close(fileid)

#  # Return a function stub for reloading the saved data
#  from pygeode.var import Var
#  if isinstance(in_dataset, Var):
#    return lambda : open(filename).vars[0]
#  else: return lambda : open(filename)

# }}}
