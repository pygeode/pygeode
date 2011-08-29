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

# Conform two time axes so their values are comparable
def conform_values (taxis1, taxis2):
  from pygeode.timeaxis import Time
  assert isinstance(taxis1, Time)
  assert isinstance(taxis2, Time)
  assert type(taxis1) is type(taxis2), "can't conform time axes of different types"  

  allowed_fields = type(taxis1).allowed_fields

  if set(taxis1.auxarrays.keys()) < set(taxis2.auxarrays.keys()):
    return conform_values(taxis2, taxis1)[::-1]
  assert set(taxis1.auxarrays.keys()) >= set(taxis2.auxarrays.keys()), "incompatible fields"

  # From here on out, can assume that taxis1 contains the superset of all the fields.
  # Check that the only extra fields of taxis1 occur at the end (fastest varying).
  blah = False
  for name in allowed_fields:
    if name not in taxis1.auxarrays: continue
    if blah: assert name not in taxis2.auxarrays, "incompatible fields"
    if name not in taxis2.auxarrays: blah = True

  # Use the units and start date from the second axis
  units = taxis2.units
  startdate = taxis2.startdate

  # Change the values of the first time axis
  taxis1 = type(taxis1)(units=units, startdate=startdate, **taxis1.auxarrays)

  return taxis1, taxis2


from pygeode.var import Var
from pygeode.timeaxis import Yearless

class Lag(Yearless): 
# {{{
  name = 'lag'
  @classmethod
  def class_has_alias(cls, name):
  # {{{
    if cls.name.lower() == name.lower(): return True
    return False
  # }}}
# }}}


# Remove leap days from data on a standard calendar, coerce the data onto a
# 365-day calendar.
def removeleapyears(data):
  from pygeode.timeaxis import StandardTime, ModelTime365
  import numpy as np

  taxis = data.getaxis(StandardTime)

  # Determine which times to keep
  year = taxis.year
  isleapyear = (year % 4 == 0) & ( (year % 100 != 0) | (year % 400 == 0) )
  isleapday = isleapyear & (taxis.month == 2) & (taxis.day == 29)
  indices = np.nonzero(~isleapday)

  # Remove the leap days from the time axis
  taxis = taxis._getitem_asvar(indices)

  # Re-init as a 365-day calendar (use the absolute date fields, not the relative values)
  taxis = ModelTime365(units=taxis.units, **taxis.auxarrays)

  # Remove the leap days from the data
  slices = [slice(None)] * data.naxes
  slices[data.whichaxis(StandardTime)] = indices
  data = data._getitem_asvar(*slices)

  # Replace the time axis of the data
  data = data.replace_axes(time = taxis)

  return data

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

del Yearless


# Split a time axis into a 2D representation (year,<everything else>)
class SplitTime (Var):
  def __init__ (self, var, iaxis):
    from pygeode.axis import NamedAxis
    from pygeode.timeaxis import CalendarTime
    from pygeode.varoperations import sorted
    from pygeode.var import copy_meta, Var
    import numpy as np

    # Get the time axis to split
    iaxis = var.whichaxis(iaxis)
    taxis = var.getaxis(iaxis)
    assert isinstance(taxis,CalendarTime)
    assert hasattr(taxis,'year'), "no years to split off!"

    # Get the years, turn them into an axis
    years = np.unique(taxis.year)
    years = NamedAxis(values=years, name='year')

    # Get the rest, as a 'climatological' axis
    days = modify(taxis, exclude='year', uniquify=True)

    # Construct the output axes
    axes = list(var.axes)
    axes = axes[:iaxis] + [years, days] + axes[iaxis+1:]

    copy_meta(var,self)

    Var.__init__(self, axes=axes, dtype=var.dtype)

    self.iaxis = iaxis
    self.var = var

  def getview (self, view, pbar):
    import numpy as np

    # Get the selected years and days
    iaxis = self.iaxis
    years = view.subaxis(iaxis).values  # numpy array
    nyears = len(years)
    days = view.subaxis(iaxis+1).auxarrays  # dictionary of numpy arrays
    ndays = view.shape[iaxis+1]

    # Input time axis (full, not sliced)
    in_taxis = self.var.getaxis(iaxis)

    # Broadcast all combinations of selected years and days, get a list of
    # dates to request from the input var
    fields = {}
    fields['year'] = years.reshape(-1,1).repeat(ndays,axis=1).flatten()
    for fname, farray in days.iteritems():
      fields[fname] = farray.reshape(1,-1).repeat(nyears,axis=0).flatten()

    # Construct a time axis with this field information
    out_times = type(in_taxis)(units=in_taxis.units, **fields)

    # Determine how much of the input axis we need, and how much
    # that provides for the output (the rest will be NaNs).
    in_sl, out_sl = in_taxis.common_map(out_times)
    assert len(in_sl) == len(out_sl)  # just for the hell of it

    # Start with a field of NaNs
    out = np.empty(view.shape, dtype=self.dtype)
    out[()] = float('NaN')  # will fail horribly if we have integer data.

    # Get some input data
    inview = view.remove(iaxis+1).replace_axis(iaxis, in_taxis, sl=in_sl)
    indata = inview.get(self.var)

    # Now, put this where we need it.
    # Input data has a single time axis, so we need to reshape the output a bit.
    squished_shape = out.shape[:iaxis] + (-1,) + out.shape[iaxis+2:]
    out = out.reshape(squished_shape)
    sl = [slice(None)] * out.ndim
    sl[iaxis] = out_sl
    out[sl] = indata

    # Restore the shape
    out = out.reshape(view.shape)

    return out

def splittimeaxis (var, iaxis='time'):
  return SplitTime (var, iaxis)

del Var
