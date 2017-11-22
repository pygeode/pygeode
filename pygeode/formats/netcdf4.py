# Pygeode interface for the netCDF4 Python module.

# constructor for the dims (wrapper for NCDim so it's only created once)
def make_dim (name, size, dimdict={}):
  from pygeode.formats.netcdf import NCDim
  if name not in dimdict:
    dimdict[name] = NCDim(size, name=str(name))
  return dimdict[name]

# Extract attributes
def make_atts (v):
  return dict((str(name),v.getncattr(name)) for name in v.ncattrs())


# A netcdf variable
def make_var (ncvar):
# {{{
  from pygeode.var import Var
  axes = [make_dim(name,size) for name,size in zip(ncvar.dimensions,ncvar.shape)]
  return Var(axes=axes, name=str(ncvar.name), values=ncvar, atts=make_atts(ncvar))
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

  import netCDF4 as nc
  from pygeode.dataset import asdataset
  from pygeode.formats import finalize_open
  from pygeode.formats.netcdf import override_values, dims2axes
  from pygeode.axis import Axis

  # Read the file
  with nc.Dataset(filename,"r") as f:

    # Construct all the variables, put in a list
    vars = map(make_var, f.variables.values())

    # Construct a dataset from these Vars
    dataset = asdataset(vars)
    dataset.atts = make_atts(f)

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
def save (filename, in_dataset, version=4, pack=None, compress=False, cfmeta = True, unlimited=None):
# {{{
  import netCDF4 as nc
  from pygeode.view import View
  from pygeode.tools import combine_axes
  from pygeode.axis import Axis, DummyAxis
  import numpy as np
  from pygeode.progress import PBar, FakePBar
  from pygeode.formats import finalize_save

  dataset = finalize_save(in_dataset, cfmeta, pack)

  # Version?
  if compress: version = 4
  assert version in (3,4)
  if version == 3:
    format = 'NETCDF3_CLASSIC'
  else:
    format = 'NETCDF4'

  vars = list(dataset.vars)
  # The output axes
  axes = combine_axes(v.axes for v in vars)

  # Include axes in the list of vars (for writing to netcdf).
  # Exclude axes which don't have any intrinsic values.
  # Look at original dataset to check original type of axes (because
  # finalize_save may force everything to be NamedAxis).
  vars = vars + [a for a in axes if not isinstance(in_dataset[a.name],DummyAxis)]

  # Variables (and axes) must all have unique names
  assert len(set([v.name for v in vars])) == len(vars), "vars must have unique names: %s"% [v.name for v in vars]

  if unlimited is not None:
    assert unlimited in [a.name for a in axes]

  with nc.Dataset(filename,'w',format=format) as f:
    # Define the dimensions
    for a in axes:
      f.createDimension(a.name, size=(None if a.name == unlimited else len(a)))

    # Define the variables (including axes)
    for var in vars:
      dimensions = [a.name for a in var.axes]
      v = f.createVariable(var.name, datatype=var.dtype, dimensions=dimensions, zlib=compress, fill_value=var.atts.get('_FillValue',None))
      v.setncatts(var.atts)

    # global attributes
    f.setncatts(dataset.atts)

    # Relative progress of each variable
    sizes = [v.size for v in vars]
    prog = np.cumsum([0.]+sizes) / np.sum(sizes) * 100

    pbar = PBar(message="Saving '%s':"%filename)

    # number of actual variables (non-axes) for determining our progress
    N = len([v for v in vars if not isinstance(v,Axis)])

    # Write the data
    for i,var in enumerate(vars):
      ncvar = f.variables[var.name]
      varpbar = pbar.subset(prog[i], prog[i+1])

      views = list(View(var.axes).loop_mem())

      for j,v in enumerate(views):
        vpbar = varpbar.part(j, len(views))
        ncvar[v.slices] = v.get(var, pbar=vpbar)

# }}}
