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
      if hasattr(var,'values'):
        dsk[key] = var.values
      else:
        dsk[key] = (var.getview, outview, False)
    chunks = [map(len,sl) for sl in slice_order]
    # Construct the dask array.
    arr = da.Array(dsk, name, chunks, dtype=var.dtype)
    # Wrap this into an xarray.DataArray (with metadata and named axes).
    out[var.name] = xr.DataArray(arr, dims = [a.name for a in var.axes], attrs = var.atts, name=var.name)
  # Build the final xarray.Dataset.
  out = xr.Dataset(out, attrs=dataset.atts)
  # Re-decode the CF metadata on the xarray side.
  out = xr.conventions.decode_cf(out)
  return out

