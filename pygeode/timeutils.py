# Time axis utilities

# Things which are useful for manipulating time axes, but aren't needed in the
# core timeaxis module.

# For now, share the same C library as the timeaxis module
from pygeode.timeaxis import lib

# Take a list of fields (year, month, day, etc.) and return the fields with duplicates removed.
# NOTE: the fields are assumed to be pre-sorted
def _uniquify (fields):
# {{{
  from pygeode.tools import point
  from pygeode.timeaxis import _argsort
  from ctypes import c_int, byref
  import numpy as np
  assert len(fields) > 0, 'no fields provided'
  S = _argsort(fields)
  in_atts = np.vstack(f[S] for f in fields).transpose()
  in_atts = np.ascontiguousarray(in_atts,dtype='int32')
  out_atts = np.empty(in_atts.shape, dtype='int32')
  selection = np.empty(len(in_atts), dtype='int32')
  nout = c_int()
  ret = lib.uniquify (len(fields), point(in_atts), len(in_atts), 
                                   point(out_atts), byref(nout))
  assert ret == 0

  nout = nout.value

  # Convert back to a list of fields
  return list(out_atts[:nout,:].transpose())

# }}}


# Mask out certain fields from a time axis
# resolution: maximum resolution of the output 'day', 'hour', 'year', etc.)
# exclude: list of fields to remove (i.e. mask out 'year' when making a climatology axis)
# include: explicit list of fields to include (everything else is excluded)
def modify (taxis, resolution=None, exclude=[], include=[], uniquify=False):
# {{{
  # Determine which fields to use
  fnames = set(taxis.auxarrays.keys())
  if isinstance(exclude,str): exclude = [exclude]
  if isinstance(include,str): include = [include]
  if len(include) > 0:
    fnames = fnames & set(include)
  fnames -= set(exclude)
  if resolution is not None:
    i = list(taxis.allowed_fields).index(resolution)
    fnames -= set(taxis.allowed_fields[i+1:])

  # Convert back to an ordered list
  fnames = [f for f in taxis.allowed_fields if f in fnames]

  # Get the fields
  fields = [taxis.auxarrays[f] for f in fnames]

  if uniquify: fields = _uniquify(fields)

  # Get a dictionary for the fields
  fields = dict([name,f] for name,f in zip(fnames,fields))

  kwargs = dict(taxis.auxatts, **fields)
  return type(taxis)(**kwargs)
# }}}


# Get a relative time array with the given parameters
def reltime (taxis, startdate=None, units=None):
  if units is None: units = taxis.units
  return taxis.date_as_val (startdate=startdate, units=units)


# Get time increment
# Units: day, hour, minute, second
def delta (taxis, units=None):
# {{{
  import numpy as np

  delt = np.diff(reltime(taxis, units=units))
  # If we have more than one 'unique' value, check if it's just a case of numerical precision
  if len(delt) > 1:
    if np.allclose(delt, delt[0]): delt = delt[0:1]
  # Not enough values to determine a delt?
  if len(delt) == 0: return 0

  assert len(delt) == 1, 'Non-unique deltas found: '+str(delt)

  return delt[0]
# }}}

# Helper function; normalizes date object so that all fields are within the standard range
def wrapdate(taxis, dt, allfields=False):
# {{{
  return taxis.val_as_date(taxis.date_as_val(dt), allfields=allfields)
# }}}

# Helper function; returns time between two dates in specified units
def date_diff(taxis, dt1, dt2, units = None):
# {{{
  return taxis.date_as_val(dt2, startdate=dt1, units = units)
# }}}


from pygeode.var import Var
from pygeode.timeaxis import Yearless

class Lag(Yearless): name = 'Lag'

class LagVar(Var):
  def __init__(self, var, iaxis, lags):
  # {{{
    import numpy as np
    from pygeode import Var
    from pygeode.timeaxis import Time

    self.iaxis = var.whichaxis(iaxis)
    taxis = var.axes[self.iaxis]
    assert isinstance(taxis, Time), 'must specify a Time axis'
    delt = taxis.delta()
    
    self.lags = np.array(lags).astype('i')
    lag = Lag(values = delt*self.lags, units=taxis.units, startdate={'day':0})
    axes = var.axes[:self.iaxis+1] + (lag, ) + var.axes[self.iaxis+1:]
    self.var = var

    Var.__init__(self, axes, dtype=var.dtype, name=var.name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    lind = self.lags[view.integer_indices[self.iaxis+1]]
    loff, roff = np.min(lind), np.max(lind)

    tind = view.integer_indices[self.iaxis]
    tmin, tmax = np.min(tind), np.max(tind)
    tsl = slice(max(tmin + loff, 0), min(tmax + roff, self.shape[self.iaxis]))
    inview = view.remove(self.iaxis+1).modify_slice(self.iaxis, tsl)
    src = inview.get(self.var, pbar=pbar)

    out = np.empty(view.shape, self.dtype)
    outsl = [0 if i == self.iaxis + 1 else slice(None) for i in range(self.naxes)]
    insl = [slice(None) for i in range(self.naxes-1)]
    for i, l in enumerate(lind):
      valid = (tind + l >= 0) & (tind + l < src.shape[self.iaxis])
      ivalid = np.where(valid)[0]
      insl[self.iaxis] = tind[ivalid] + l
      outsl[self.iaxis] = ivalid
      outsl[self.iaxis+1] = i
      out[outsl] = src[insl]
      outsl[self.iaxis] = np.where(~valid)[0]
      out[outsl] = np.nan
    
    return out
  # }}}

del Var, Yearless
