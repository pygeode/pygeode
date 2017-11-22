
# Include "pygeode.formats" from other PyGeode-based packages.
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
del extend_path

# Include packages via the entry_points mechanism of pkg_resources.
import pkg_resources
for ep in pkg_resources.iter_entry_points('pygeode.formats'):
  globals()[ep.name] = ep.load()
  del ep
del pkg_resources

# Formats included when doing "from pygeode.formats import *"
# Note: additional formats from the plugin directories are added dynamically
# (see pygeode/__init__.py)

from multifile import openall, open_multi

__all__ = ['openall', 'open_multi', 'open', 'save']

extdict = {'.nc':'netcdf',
           '.hdf': 'hdf4',
           '.grib':'grib' }

from pygeode.var import Var
class PackVar(Var):
  def __init__(self, var):
  # {{{
    from pygeode.var import copy_meta
    import numpy as np
    self.var = var

    # At present data is packed into short integers following the packing
    # algorithm described in the NetCDF Operator documentation
    dtype = np.int16

    min = var.nanmin()
    max = var.nanmax()
    self.scale = (max - min) / (2**16 - 2.)
    self.offset = 0.5 * (min + max)

    Var.__init__(self, var.axes, dtype=dtype)

    copy_meta(var, self)
    self.atts['packing_convention'] = 'NetCDF (16 bit)'
    self.atts['scale_factor'] = self.scale
    self.atts['add_offset'] = self.offset
  # }}}

  def getview (self, view, pbar):
  # {{{
    return ((view.get(self.var, pbar) - self.offset) / self.scale).astype(self.dtype)
  # }}}

def autodetectformat(filename):
# {{{
  ''' Returns best guess at file format based on file name.

      Parameters
      ==========
      filename : string
        Filename to identify

      Returns
      =======
      string
        String specifying identified file format.   

      Raises
      ======
      ValueError
        If the format cannot be determined from the extension.

      See Also
      ========
      extdict
  '''

  from os import path

  rt, ext = path.splitext(filename)
  format = extdict.get(ext, None)
  if format is None:
    raise ValueError('Unrecognized extension "%s"; please specify a file format.' % ext)

  return format
# }}}

def open(filename, format = None, value_override = {}, dimtypes = {}, namemap = {}, varlist = [],
         cfmeta = True, **kwargs):
# {{{
  ''' Returns a :class:`Dataset` containing variables defined in a single file.

  Parameters
  ==========
  filename : string
    Path of file to open

  format : string, optional
    String specifying format of file to open. If none is given the format will be automatically
    detected from the file (see :func:`autodetectformat`)

  value_override : dict, optional
    A dictionary containing arrays with which to override values for one or more variables (specified
    by the keys). This can be used for instance to avoid loading the values of an axis whose values
    are severely scattered across a large file.

  dimtypes : dict, optional
    A dictionary mapping dimension names to axis classes. The keys should be
    axis names as defined in the file; values should be one of:

    1. an axis instance, which will be used directly
    2. an axis class, which will be used to create a new instance with the values given by the file
    3. a tuple of an axis class and a dictionary with keyword arguments to pass to that axis' constructor              

    If dimtypes is not specified, an attempt is made to automatically identify the axis types (see optional
    `cfmeta` argument below)

  namemap : dict, optional
    A dictionary to map variable names as specified in the file (keys) to PyGeode variable names
    (values); also works for axes/dimensions

  varlist : list, optional
    A list (of strings) specifying the variables that should be loaded into the
    data set (if the list is empty, all NetCDF variables will be loaded)

  cfmeta : boolean
    If true, an attempt to identify the type of each dimension is made
    following the CF metadata conventions.

  Returns
  =======
  dataset
    A dataset containing the variables contained in the file. The variable data itself is not loaded
    into memory. 

  Notes
  =====
  The format of the file is automatically detected from the filename (if
  possible); otherwise it must be specified by the ``format`` argument. 
  The identifiers used in ``varlist`` and ``dimtypes`` are the original names used in
  the NetCDF file, not the names given in ``namemap``.

  See Also
  ========
  openall
  open_multi
  '''

  if format is None: format = autodetectformat(filename)

  if not hasattr(format, 'open'):
    try:
      format = __import__("pygeode.formats.%s" % format, fromlist=["pygeode.formats"])
    except ImportError:
      raise ValueError('Unrecognized format module %s.' % format)

  return format.open(filename, value_override=value_override, dimtypes=dimtypes, \
              namemap=namemap, varlist=varlist, cfmeta=cfmeta, **kwargs)
# }}}

def save(filename, dataset, format=None, cfmeta=True, **kwargs):
# {{{
  ''' Saves a :class:`Var` or :class:`Dataset` to file.

  Parameters
  ==========
  filename : string
    Path of file to save to.

  dataset : :class:`Var`, :class:`Dataset`, or collection of :class:`Var` objects
    Variables to write to disk. The dataset is consolidated using :func:`dataset.asdataset`.

  format : string, optional
    String specifying format of file to open. If none is given the format will be automatically
    detected from the file (see :func:`autodetectformat`)

  cfmeta : boolean
    If true, metadata is automatically written specifying the axis dimensions following CF
    metadata conventions.

  Notes
  =====
  The format of the file is automatically detected from the filename (if
  possible). The NetCDF format is at present the best supported.
  '''
  if format is None: format = autodetectformat(filename)

  if not hasattr(format, 'save'):
    try:
      format = __import__("pygeode.formats.%s" % format, fromlist=["pygeode.formats"])
    except ImportError:
      raise ValueError('Unrecognized format module %s.' % format)

  format.save(filename, dataset, cfmeta=cfmeta, **kwargs)
# }}}

# Coerce axes into particular types
# (useful if there's no existing ruleset for detecting your axes)
# (based on deprecated tools.make_axis)
def set_axistypes (dataset, dimtypes):
# {{{
  from pygeode.axis import Axis
  from pygeode.var import copy_meta
  from types import FunctionType
  assert isinstance(dimtypes, dict)

  replacements = {}

  for oldaxis in dataset.axes:
    name = oldaxis.name
    if name not in dimtypes: continue
    dt = dimtypes[name]
    # Determine axis type      
    if isinstance(dt, Axis):   # Axis instance
      if len(dt) != len(oldaxis):
        raise ValueError('Provided axis instance %s is the wrong length (expected length %d, got length %d)' % (repr(dt),len(oldaxis),len(dt)))
      axis = dt 
    elif hasattr(dt, '__bases__') and issubclass(dt, Axis): # Axis class
      dimclass = dt
      axis = dimclass(values=oldaxis.values)
      # Copy the file metadata (but discard plot attributes from the old axis)
      # (See issue 22)
      copy_meta (oldaxis, axis, plotatts=False)
    elif hasattr(dt, '__len__'):
      if len(dt) != 2: raise ValueError('Got a list/tuple for dimtypes, but did not have 2 elements as expected (Axis class, parameters).  Instead, got %s.'%dt)
      dimclass, dimargs = dt
      dimargs = dimargs.copy()
      assert issubclass (dimclass, Axis), "expected an Axis subclass, got %s instead."%dimclass
      assert isinstance (dimargs, dict)
      if 'values' not in dimargs:  dimargs['values'] = oldaxis.values
      axis = dimclass(**dimargs)
    # Axis-creating function?
    elif isinstance (dt, FunctionType):
      axis = dt(oldaxis)
    else: raise ValueError('Unrecognized dimtypes parameter. Expected a dictionary, axis class, or axis instance.  Got %s instead.'%type(dt))

    assert len(axis) == len(oldaxis), "expected axis of length %s, ended up with axis of length %s"%(len(oldaxis),len(axis))
    replacements[name] = axis

  return dataset.replace_axes(axisdict=replacements)
# }}}

# Apply variable whitelist to a dataset (only keep specific variables)
def whitelist (dataset, varlist):
# {{{
  from pygeode.dataset import Dataset
  assert isinstance(varlist,(list,tuple))
  vars = [dataset[v] for v in varlist if dataset.vardict.has_key(v)]
  dataset = Dataset(vars, atts=dataset.atts)
  return dataset
# }}}

def finalize_open(dataset, dimtypes = {}, namemap = {}, varlist = [], cfmeta = True):
# {{{
  from pygeode.formats import cfmeta as cf
  # Process CF-metadata?
  if cfmeta is True:
    # Skip anything that we're going to override in dimtypes
    # (so we don't get any meaningless warnings or other crap from cfmeta)
    dataset = cf.decode_cf(dataset, ignore=dimtypes.keys())

  # Apply custom axis types?
  if len(dimtypes) > 0:
    dataset = set_axistypes(dataset, dimtypes)

  # Keep only specific variables?
  if len(varlist) > 0:
    dataset = whitelist(dataset, varlist)

  # Rename variables?
  if len(namemap) > 0:
    # Check both axes and variables
    dataset = dataset.rename_vars(vardict=namemap)
    dataset = dataset.rename_axes(axisdict=namemap)

  return dataset
# }}}

def finalize_save(dataset, cfmeta = True, pack = None):
# {{{
  from pygeode.formats import cfmeta as cf
  from pygeode.dataset import asdataset

  # Only pack if pack is true
  if pack:
    if hasattr(pack, '__len__'): # Assume this is a list of variables to pack
      vars = [PackVar(v) if v.name in pack else v for v in dataset.vars]
    else:
      vars = [PackVar(v) for v in dataset.vars]
    dset = asdataset(vars)
    dset.atts = dataset.atts.copy()
  else:
    dset = dataset

  # Encode standard axes back into netcdf metadata?
  if cfmeta is True:
    return cf.encode_cf(dset)
  else:
    return asdataset(dset)
# }}}

