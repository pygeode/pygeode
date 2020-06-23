# Pygeode interface for the netCDF4 Python module.

# A generic dimension (has no attributes except a name and a length)
from pygeode.axis import DummyAxis
class NCDim (DummyAxis): pass
del DummyAxis

# constructor for the dims (wrapper for NCDim so it's only created once)
def make_dim (name, size, dimdict={}):
  if (name,size) not in dimdict:
    dimdict[(name,size)] = NCDim(size, name=str(name))
  return dimdict[(name,size)]

# Extract attributes
def make_atts (v):
  import sys
  atts = dict()
  for name in v.ncattrs():
    att = v.getncattr(name)

    if sys.version_info[0] < 3:
      # netcdf4-python module in Python 2 uses unicode instead of strings.
      # Need to force this back to string type.
      if isinstance(att, unicode): 
        att = att.encode('utf-8')
      #att = att.encode('ascii', 'replace').decode('ascii')

    atts[str(name)] = att
  return atts

# Wrapper for netcdf variables.
# Delegates access to the arrays so they don't get loaded until needed.
from pygeode.var import Var
class NCVar(Var):
  def __init__ (self, axes, name, ncvar, atts):
    from pygeode.var import Var
    self._ncvar = ncvar
    # Turn off netcdf4 masking and scaling algorithm so that they don't get applied twice
    ncvar.set_auto_maskandscale(False)
    Var.__init__(self, axes=axes, name=name, dtype=ncvar.dtype, atts=atts)
  def getvalues (self, start, count):
    sl = [slice(s,s+c) for s,c in zip(start,count)]
    if len(sl) > 0:
      return self._ncvar[sl]
    else:
      return self._ncvar[:]
del Var

# A netcdf variable
def make_var (ncvar):
# {{{
  from pygeode.var import Var
  axes = [make_dim(str(name),size) for name,size in zip(ncvar.dimensions,ncvar.shape)]
  return NCVar(axes=axes, name=str(ncvar.name), ncvar=ncvar, atts=make_atts(ncvar))
# }}}

# A netcdf variable
def make_dataset (ncfile):
# {{{
  from pygeode.dataset import asdataset
  # Construct all the variables, put in a list
  vars = list(map(make_var, list(ncfile.variables.values())))
  
  # Construct a dataset from these Vars
  dataset = asdataset(vars)
  dataset.atts = make_atts(ncfile)
  return dataset

# }}}

# Prepare var axes for writing 
def tidy_axes(dataset, unlimited=None):
# {{{
  from pygeode.tools import combine_axes
  from pygeode.axis import DummyAxis
  from pygeode.dataset import asdataset
  
  vars = list(dataset.vars)
  # The output axes
  axes = combine_axes(v.axes for v in vars)

  # Include axes in the list of vars (for writing to netcdf).
  # Exclude axes which don't have any intrinsic values.
  # Look at original dataset to check original type of axes (because
  # finalize_save may force everything to be NamedAxis).
  vars = vars + [a for a in axes if not isinstance(dataset[a.name],DummyAxis)]

  # Variables (and axes) must all have unique names
  assert len(set([v.name for v in vars])) == len(vars), "vars must have unique names: %s"% [v.name for v in vars]

  if unlimited is not None:
    assert unlimited in [a.name for a in axes]

  return asdataset(vars)
# }}}

def write_var (ncfile, dataset, unlimited=None, compress=False):
# {{{
  from pygeode.view import View
  from pygeode.axis import Axis 
  import numpy as np
  from pygeode.progress import PBar, FakePBar
  from pygeode.tools import combine_axes
  
  vars = list(dataset.vars)
  axes = combine_axes(v.axes for v in vars)

  # Define the dimensions
  for a in axes:
    ncfile.createDimension(a.name, size=(None if a.name == unlimited else len(a)))

  # Define the variables (including axes)
  for var in vars:
    dimensions = [a.name for a in var.axes]
    v = ncfile.createVariable(var.name, datatype=var.dtype, dimensions=dimensions, zlib=compress, fill_value=var.atts.get('_FillValue',None))
    v.set_auto_maskandscale(False)
    v.setncatts(var.atts)

  # global attributes
  ncfile.setncatts(dataset.atts)

  # Relative progress of each variable
  sizes = [v.size for v in vars]
  prog = np.cumsum([0.]+sizes) / np.sum(sizes) * 100

  pbar = PBar(message="Saving '%s':"%ncfile.filepath())

  # number of actual variables (non-axes) for determining our progress
  N = len([v for v in vars if not isinstance(v,Axis)])

  # Write the data
  for i,var in enumerate(vars):
    ncvar = ncfile.variables[var.name]
    varpbar = pbar.subset(prog[i], prog[i+1])

    views = list(View(var.axes).loop_mem())

    for j,v in enumerate(views):
      vpbar = varpbar.part(j, len(views))
      ncvar[v.slices] = v.get(var, pbar=vpbar)

# }}}

###########################################################
# Internal methods needed for open command.  Copied from pygeode.formats.netcdf

# Override values from the source?
def override_values (dataset, value_override):
# {{{
  from warnings import warn
  import numpy as np
  from pygeode.var import Var, copy_meta
  vardict = {}
  for name, values in value_override.items():
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
  dataset = dataset.remove(*list(replacements.keys()))
  return dataset
# }}}


def open(filename, value_override = {}, dimtypes = {}, namemap = {},  varlist = [], cfmeta = True):
# {{{
  ''' open (filename, [value_override = {}, dimtypes = {}, namemap = {}, varlist = [] ])

  Returns a Dataset or dictionary of Datasets of PyGeode variables contained in the specified files. The axes of the 
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
  Note: -The identifiers used in varlist and dimtypes are the original names used in the NetCDF file, 
        not the names given in namemap.
        -The optional arguments are not currently supported for netcdf4 files containing groups.'''

  import netCDF4 as nc
  from pygeode.dataset import asdataset
  from pygeode.formats import finalize_open
  from pygeode.axis import Axis

  # Read the file
  try:
    f = nc.Dataset(filename,"r")
    if f.groups:
      dataset =  {str(key): make_dataset(value) for key, value in f.groups.items()}
      dataset =  {str(key): dims2axes(value) for key, value in dataset.items()}
      
      return {str(key): finalize_open(value) for key, value in dataset.items()}
          
    else: 
      dataset = make_dataset(f)
      # Add the object stuff from dimtypes to value_override, so we don't trigger a
      # load operation on those dims.
      # (We could use any values here, since they'll be overridden again later,
      #  but we might as well use something relevant).
      value_override = dict(value_override)  # don't use  the default (static) empty dict
      for k,v in list(dimtypes.items()):
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
  except IOError:  # Problem accessing the file?
    raise
# }}}

#TODO: factor out cf-meta encoding and other processing steps
# Write a dataset to netcdf
def save (filename, in_dataset, version=4, pack=None, compress=False, cfmeta = True, unlimited=None):
# {{{
  import netCDF4 as nc
  from pygeode.view import View
  from pygeode.tools import combine_axes
  from pygeode.axis import Axis, DummyAxis
  import numpy as np
  from pygeode.progress import PBar, FakePBar
  from pygeode.formats import finalize_save
  from pygeode.dataset import asdataset
  
  # Version?
  if compress: version = 4
  assert version in (3,4)
  if version == 3:
    format = 'NETCDF3_CLASSIC'
  else:
    format = 'NETCDF4'

  assert format in ('NETCDF3_CLASSIC','NETCDF4')
  
  with nc.Dataset(filename,'w',format=format) as f:
    
    if isinstance(in_dataset, dict):
      dataset =  {key: finalize_save(asdataset(value), cfmeta, pack) for key, value in in_dataset.items()}
      dataset =  {key: tidy_axes(value, unlimited=unlimited) for key, value in dataset.items()}
      for key, value in dataset.items():
        group = f.createGroup(key)
        write_var(group, value, unlimited=unlimited, compress=compress)
        
    else:
      in_dataset = asdataset(in_dataset)

      dataset = finalize_save(in_dataset, cfmeta, pack)
      dataset = tidy_axes(dataset, unlimited=unlimited)
      write_var(f, dataset, unlimited=unlimited, compress=compress)
  
# }}}
