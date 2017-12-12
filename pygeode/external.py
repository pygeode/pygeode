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
  dataset = encode_cf(dataset)
  out = dict()
  for var in list(dataset.axes) + list(dataset.vars):
    name = var.name + "-" + tokenize(var)
    dsk = dict()
    slice_order = [[] for a in var.axes]
    chunks = []
    inview = View(var.axes)
    for outview in inview.loop_mem():
      integer_indices = map(tuple,outview.integer_indices)
      for o, sl in zip(slice_order, integer_indices):
        if sl not in o:
          o.append(sl)
      ind = [o.index(sl) for o, sl in zip(slice_order, integer_indices)]
      key = tuple([name] + ind)
      if hasattr(var,'values'):
        dsk[key] = var.values
      else:
        dsk[key] = (var.getview, outview, False)
    chunks = [map(len,sl) for sl in slice_order]
    arr = da.Array(dsk, name, chunks, dtype=var.dtype)
    out[var.name] = xr.DataArray(arr, dims = [a.name for a in var.axes], attrs = var.atts, name=var.name)
  out = xr.Dataset(out, attrs=dataset.atts)
  out = xr.conventions.decode_cf(out)
  return out

