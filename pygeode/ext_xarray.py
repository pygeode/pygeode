# Functions for converting PyGeode objects to other (external) projects, and
# vice-versa.

def to_xarray(dataset):
  """
  Converts a PyGeode Dataset into an xarray Dataset.

  Parameters
  ----------
  dataset : pygeode.Dataset
    The dataset to be converted.

  Returns
  -------
  out : xarray.Dataset
    An object which can be used with the xarray package.
  """
  from pygeode.dataset import asdataset
  from pygeode.formats.cfmeta import encode_cf
  from pygeode.view import View
  from dask.base import tokenize
  import dask.array as da
  import xarray as xr
  dataset = asdataset(dataset)
  # Encode the axes/variables with CF metadata.
  dataset = encode_cf(dataset)
  out = dict()
  # Loop over each axis and variable.
  for var in list(dataset.axes) + list(dataset.vars):
    # Generate a unique name to identify it with dask.
    name = var.name + "-" + tokenize(var)
    dsk = dict()
    dims = [a.name for a in var.axes]

    # Special case: already have the values in memory.
    if hasattr(var,'values'):
      out[var.name] = xr.DataArray(var.values, dims=dims, attrs=var.atts, name=var.name)
      continue

    # Keep track of all the slices that were made over each dimension.
    # This information will be used to determine the "chunking" that was done
    # on the variable from inview.loop_mem().
    slice_order = [[] for a in var.axes]
    chunks = []
    # Break up the variable into into portions that are small enough to fit
    # in memory.  These will become the "chunks" for dask.
    inview = View(var.axes)
    for outview in inview.loop_mem():
      integer_indices = map(tuple,outview.integer_indices)
      # Determine *how* loop_mem is splitting the axes, and define the chunk
      # sizes accordingly.
      # A little indirect, but loop_mem doesn't make its chunking choices
      # available to the caller.
      for o, sl in zip(slice_order, integer_indices):
        if sl not in o:
          o.append(sl)
      ind = [o.index(sl) for o, sl in zip(slice_order, integer_indices)]
      # Add this chunk to the dask array.
      key = tuple([name] + ind)
      dsk[key] = (var.getview, outview, False)
    # Construct the dask array.
    chunks = [map(len,sl) for sl in slice_order]
    arr = da.Array(dsk, name, chunks, dtype=var.dtype)
    # Wrap this into an xarray.DataArray (with metadata and named axes).
    out[var.name] = xr.DataArray(arr, dims = dims, attrs = var.atts, name=var.name)
  # Build the final xarray.Dataset.
  out = xr.Dataset(out, attrs=dataset.atts)
  # Re-decode the CF metadata on the xarray side.
  out = xr.conventions.decode_cf(out)
  return out

# Helper method - convert unicode attributes to str.
def _fix_atts (atts):
  import sys
  if sys.version_info[0] >= 3:
    unicode = str
  atts = dict((str(k),v) for k,v in atts.items())
  for k,v in list(atts.items()):
    if isinstance(v,unicode):
      atts[k] = str(v)
  return atts

from pygeode.var import Var
class XArray_DataArray(Var):
  """
  A wrapper for accessing xarray.DataArray objects as pygeode.Var objects.
  """
  def __init__ (self, name, arr):
    from pygeode.var import Var
    from pygeode.axis import NamedAxis
    self._arr = arr
    # Extract axes and metadata.
    # Convert unicode strings to str for compatibility with PyGeode.
    axes = [NamedAxis(n,str(d)) for n,d in zip(arr.shape,arr.dims)]
    atts = _fix_atts(arr.attrs)
    Var.__init__(self, axes, name=str(name), dtype=arr.dtype, atts=atts)
  def getview (self, view, pbar):
    import numpy as np
    out = np.asarray(self._arr[view.slices])
    pbar.update(100)
    return out

del Var

def from_xarray(dataset):
  """
  Converts an xarray Dataset into a PyGeode Dataset.

  Parameters
  ----------
  dataset : xarray.Dataset
    The dataset to be converted.

  Returns
  -------
  out : pygeode.Dataset
    An object which can be used with the pygeode package.
  """
  import xarray as xr
  from pygeode.dataset import Dataset
  from pygeode.formats.netcdf import dims2axes
  from pygeode.formats.cfmeta import decode_cf
  # Encode the axes/variables with CF metadata.
  out = []
  # Loop over each axis and variable, and wrap as a pygeode.Var object.
  for varname, var in dataset.variables.items():
    # Apply a subset of conventions that are relevant to PyGeode.
    try:
      var = xr.conventions.maybe_encode_datetime(var)
      var = xr.conventions.maybe_encode_timedelta(var)
    except AttributeError:
      var = xr.coding.times.CFDatetimeCoder().encode(var)
      var = xr.coding.times.CFTimedeltaCoder().encode(var)
    try:
      var = xr.conventions.maybe_encode_string_dtype(var)
    except AttributeError:
      pass # Using an older version of xarray (<0.10.0)?
    out.append(XArray_DataArray(varname, var))
  # Wrap all the Var objects into a pygeode.Dataset object.
  out = Dataset(out, atts=_fix_atts(dataset.attrs))
  # Re-construct the axes as pygeode.axis.NamedAxis objects.
  out = dims2axes(out)
  # Re-decode the CF metadata on the PyGeode end.
  # This will get the approperiate axis types for lat, lon, time, etc.
  out = decode_cf(out)
  return out

