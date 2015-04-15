
# Include "pygeode.formats" from other PyGeode-based packages.
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
del extend_path

# Formats included when doing "from pygeode.formats import *"
# Note: additional formats from the plugin directories are added dynamically
# (see pygeode/__init__.py)

from multifile import openall, open_multi

__all__ = ['openall', 'open_multi', 'open', 'save']

extdict = {'.nc':'netcdf',
           '.hdf': 'hdf4',
           '.grib':'grib' }

def autodetectformat(filename):
# {{{
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

def finalize_save(dataset, cfmeta = True):
# {{{
  from pygeode.formats import cfmeta as cf
  from pygeode.dataset import asdataset

  # Encode standard axes back into netcdf metadata?
  if cfmeta is True:
    return cf.encode_cf(dataset)
  else:
    return asdataset(dataset)
# }}}

