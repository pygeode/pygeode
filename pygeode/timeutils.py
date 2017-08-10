# Time axis utilities

''' A set of tools for manipulating time axes. '''

# Things which are useful for manipulating time axes, but aren't needed in the
# core timeaxis module.

# For now, share the same C library as the timeaxis module
from pygeode.timeaxis import lib

# Take a list of fields (year, month, day, etc.) and return the fields with duplicates removed.
# NOTE: the fields are assumed to be pre-sorted
def _uniquify (fields):
# {{{
  from pygeode.timeaxis import _argsort
  import numpy as np

  assert len(fields) > 0, 'no fields provided'
  S = _argsort(fields)
  in_atts = np.vstack(f[S] for f in fields).transpose()
  in_atts = np.ascontiguousarray(in_atts,dtype='int32')
  out_atts = np.empty(in_atts.shape, dtype='int32')
  nout = lib.uniquify (len(fields), in_atts, len(in_atts), out_atts)

  # Convert back to a list of fields
  return list(out_atts[:nout,:].transpose())

# }}}


# Mask out certain fields from a time axis
# resolution: maximum resolution of the output 'day', 'hour', 'year', etc.)
# exclude: list of fields to remove (i.e. mask out 'year' when making a climatology axis)
# include: explicit list of fields to include (everything else is excluded)
def modify (taxis, resolution=None, exclude=[], include=[], uniquify=False):
# {{{
  ''' Modifies the auxiliary arrays associated with a time axis.

    Parameters
    ==========
    taxis : time axis instance
      Time axis to modify

    resolution : {None, 'year', 'month', 'day', 'hour', 'minute', 'second'}, optional
      Finest division to retain.

    exclude : list of strings, optional
      List of arrays to remove

    include : list of strings, optional
      Explicit list of arrays to include

    Returns
    =======
    taxis : time axis instance
      Modified time axis
  '''
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
# {{{
  ''' Returns time axis values relative to a given reference date. The units can be
  be specified, if none are given the units of the given time axis are used.'''
  if units is None: units = taxis.units
  return taxis.date_as_val (startdate=startdate, units=units)
# }}}


# Get time increment
# Units: day, hour, minute, second
def delta (taxis, units=None):
# {{{
  ''' Returns the interval between values of a given time axis. If
  non-unique intervals are found an exception is raised. The units
  can be specified; if none are given the units of the given time axis
  are used.'''
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
  ''' Returns a modified date dictionary such that all fields
  lie within standard values. '''
  return taxis.val_as_date(taxis.date_as_val(dt), allfields=allfields)
# }}}

# Helper function; returns time between two dates in specified units
def date_diff(taxis, dt1, dt2, units = None):
# {{{
  ''' Returns time interval between two dates. A time axis must be given
  to specify the calendar. If no units are specified the units of the given
  time axis are used.'''
  return taxis.date_as_val(dt2, startdate=dt1, units = units)
# }}}

# Conform two time axes so their values are comparable
def conform_values (taxis1, taxis2):
# {{{
  '''Given two time axes, return new axes such that their values are comparable.'''
  from pygeode.timeaxis import Time
  assert isinstance(taxis1, Time)
  assert isinstance(taxis2, Time)
  assert type(taxis1) is type(taxis2), "can't conform time axes of different types"  

  allowed_fields = type(taxis1).allowed_fields

  # Make taxis1 be the one with the 'larger' set of fields
  if set(taxis1.auxarrays.keys()) < set(taxis2.auxarrays.keys()):
    taxis1, taxis2 = taxis2, taxis1
  assert set(taxis1.auxarrays.keys()) >= set(taxis2.auxarrays.keys()), "incompatible fields"

  # From here on out, can assume that taxis1 contains the superset of all the fields.

  # Check that the only extra fields of taxis1 occur at the end (fastest varying).
  # Otherwise, we get a non-trivial many-to-one mapping between the axes
  # (they can't share a linear 'values' relationship).
  end_resolution = False
  for name in allowed_fields:
    if name not in taxis1.auxarrays: continue
    # If we've already gone beyond the resolution of taxis2, then there
    # shouldn't be any further fields defined in it.
    if end_resolution: assert name not in taxis2.auxarrays, "incompatible fields"
    # Check if we're now beyond the resolution of taxis2
    if name not in taxis2.auxarrays: end_resolution = True

  # Use the units and start date from the second axis
  units = taxis2.units
  startdate = taxis2.startdate

  # Change the values of the first time axis
  taxis1 = type(taxis1)(units=units, startdate=startdate, **taxis1.auxarrays)

  return taxis1, taxis2
# }}}


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
from pygeode.timeaxis import ModelTime365
def removeleapyears(data, omitdoy_leap=[60], omitdoy_noleap=[], new_axis_type=ModelTime365):
# {{{
  '''Removes leap day(s) from data on a standard calendar. Casts variable with
  a :class:`StandardTime` time axis onto a time axis with a uniform year length
  by removing days from leap years.
  
  Parameters
  ----------
  data :  :class:`Var`
    The variable to modify. Should have a :class:`Time` axis.
  
  omitdoy_leap : list, optional [ [60] ]
    A list of days of the year (e.g. 1 is 1 January, 60 is 29 February) to remove
    from leap years.

  omitdoy_noleap : list, optional [ [] ]
    A list of days of the year (e.g. 1 is 1 January, 60 is 1 March) to remove
    from non-leap years. Can be empty.
  
  new_axis_type : :class:`CalendarTime`, optional
    Time axis class to use instead. Default is :class:`ModelTime365`. Should
    expect a year length consistent with the lists passed to omitdoy_leap and
    omitdoy_noleap

  Returns
  -------
  New :class:`Var` object with modified time axis and specified days removed
  from leap year.'''

  from pygeode.timeaxis import Time, StandardTime
  import numpy as np

  taxis = data.getaxis(Time)

  # Determine which times to keep
  year = taxis.year
  isleapyear = isinstance(taxis,StandardTime) & (year % 4 == 0) & ( (year % 100 != 0) | (year % 400 == 0) )

  doy = []
  for y in sorted(set(taxis.year)):
    doy.extend(reltime(taxis(year=y), startdate={'year':y, 'month':1, 'day':1}, units='days')+1)
  doy = np.array(np.floor(doy),dtype=int)

  omit_on_leap = [False] * len(taxis)
  for d in omitdoy_leap:
    omit_on_leap |= (doy == d)

  omit_on_noleap = [False] * len(taxis)
  for d in omitdoy_noleap:
    omit_on_noleap |= (doy == d)

  omit = (isleapyear & omit_on_leap) | ((~isleapyear) & omit_on_noleap)
  indices = np.nonzero(~omit)

  # Remove the leap days from the data
  slices = [slice(None)] * data.naxes
  slices[data.whichaxis(Time)] = indices
  data = data._getitem_asvar(slices)

  # Recreate the axis as the new type.
  # Convert doy (old axis) to doy (new axis)
  new_taxis_pieces = []
  for y in sorted(set(taxis.year)):
    # Check for leap year - use one list of omitted days or the other
    if isleapyear[taxis.year==y][0]:
      omitted_days = sorted(omitdoy_leap, reverse=True)
    else:
      omitted_days = sorted(omitdoy_noleap, reverse=True)
    # Start with the old doy
    doy = reltime(taxis(year=y), startdate={'year':y, 'month':1, 'day':1}, units='days')+1
    # Strip out the omitted days, and re-index the other days
    for d in omitted_days:
      doy = doy[np.floor(doy) != d]
      doy[doy>d] -= 1
    # Create an axis of the new type
    new_taxis_pieces.append(new_axis_type(values=doy-1, units='days', startdate={'year':y, 'month':1, 'day':1}))

  # Rebuild the final new axis
  # (Had to break it down by year, to do the doy arithmetic)
  new_taxis = new_axis_type.concat(new_taxis_pieces)

  # Replace the time axis of the data
  data = data.replace_axes(time = new_taxis)

  return data
# }}}

del ModelTime365

class LagVar(Var):
# {{{
  def __init__(self, var, iaxis, lags, reverse=False):
  # {{{
    import numpy as np
    from pygeode import Var
    from pygeode.timeaxis import Time

    self.iaxis = var.whichaxis(iaxis)
    taxis = var.axes[self.iaxis]
    assert isinstance(taxis, Time), 'must specify a Time axis'
    delt = (taxis.values[1] - taxis.values[0])
    if reverse: 
      delt = -delt
      lags = lags[::-1]
    
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

    tsl = slice(max(tmin + loff, 0), min(tmax + roff + 1, self.shape[self.iaxis]))
    imin = tsl.start
    inview = view.remove(self.iaxis+1).modify_slice(self.iaxis, tsl)
    src = inview.get(self.var, pbar=pbar)

    out = np.empty(view.shape, self.dtype)
    outsl = [0 if i == self.iaxis + 1 else slice(None) for i in range(self.naxes)]
    insl = [slice(None) for i in range(self.naxes-1)]
    for i, l in enumerate(lind):
      valid = (tind + l >= 0) & (tind + l < self.shape[self.iaxis])
      ivalid = np.where(valid)[0]
      insl[self.iaxis] = tind[ivalid] + l - imin
      outsl[self.iaxis] = ivalid
      outsl[self.iaxis+1] = i
      out[outsl] = src[insl]
      outsl[self.iaxis] = np.where(~valid)[0]
      out[outsl] = np.nan
    
    return out
  # }}}
# }}}

del Yearless

def lag (var, iaxis, lags, reverse=False):
# {{{
  ''' Adds a lag axis with offset values. '''
  return LagVar (var, iaxis, lags, reverse=reverse)
# }}}


# Split a time axis into year,<everything else>
def _splittime (taxis):
# {{{
  from pygeode.timeaxis import CalendarTime
  from pygeode.axis import NamedAxis
  import numpy as np

  assert isinstance(taxis,CalendarTime)
  assert hasattr(taxis,'year'), "no years to split off!"

  # Get the years, turn them into an axis
  years = np.unique(taxis.year)
  years = NamedAxis(values=years, name='year')

  # Get the rest, as a 'climatological' axis
  days = modify(taxis, exclude='year', uniquify=True).rename('day')

  return years, days
# }}}


# Join 2 axes (year,<everything else>) into a single time axis
# 'years' is an axis, 'days' is a CalendarTime axis.
# NOTE: magical things can happen when the calendar year doesn't have a fixed length
def _jointime (years, days):
# {{{
  from pygeode.axis import Axis
  from pygeode.timeaxis import CalendarTime
  assert isinstance(years, Axis)
  assert isinstance(days, CalendarTime)
  assert 'year' not in days

  nyears = len(years)
  ndays = len(days)
  timetype = type(days)
  timeunits = days.units

  # Convert 'years' to a raw numpy array, and 'days' to a dictionary of numpy
  # arrays, representing the other date/time fields.
  years = years.values
  days = days.auxarrays

  # Broadcast all combinations of selected years and days, get a list of
  # dates to request from the input var
  fields = {}
  fields['year'] = years.reshape(-1,1).repeat(ndays,axis=1).flatten()
  for fname, farray in days.iteritems():
    fields[fname] = farray.reshape(1,-1).repeat(nyears,axis=0).flatten()

  # Construct a time axis with this field information
  out_times = timetype(units=timeunits, **fields)

  return out_times
# }}}

# Now, define some Var classes that apply the above split/join to
# variables that contain time axes.

# Split a time axis into a 2D representation (year,<everything else>)
class SplitTime (Var):
# {{{
  def __init__ (self, var, iaxis):
# {{{
    from pygeode.var import copy_meta, Var

    # Get the time axis to split
    iaxis = var.whichaxis(iaxis)
    taxis = var.getaxis(iaxis)

    years, days = _splittime(taxis)

    # Construct the output axes
    axes = list(var.axes)
    axes = axes[:iaxis] + [years, days] + axes[iaxis+1:]

    copy_meta(var,self)

    Var.__init__(self, axes=axes, dtype=var.dtype)

    self.iaxis = iaxis
    self.var = var
# }}}

  def getview (self, view, pbar):
# {{{
    import numpy as np

    # Get the selected years and days
    iaxis = self.iaxis
    years = view.subaxis(iaxis)
    days = view.subaxis(iaxis+1)

    # Input time axis (full, not sliced)
    in_taxis = self.var.getaxis(iaxis)

    # Selected output times (joined into a 1D time axis)
    out_times = _jointime(years,days)

    # Determine how much of the input axis we need, and how much
    # that provides for the output (the rest will be NaNs).
    in_sl, out_sl = in_taxis.common_map(out_times)
    assert len(in_sl) == len(out_sl)  # just for the hell of it

    # Start with a field of NaNs
    out = np.empty(view.shape, dtype=self.dtype)
    out[()] = float('NaN')  # will fail horribly if we have integer data.

    # Get some input data
    inview = view.remove(iaxis+1).replace_axis(iaxis, in_taxis, sl=in_sl)
    indata = inview.get(self.var, pbar=pbar)

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
# }}}
# }}}

def splittimeaxis (var, iaxis='time'):
# {{{
  '''Convert a variable with a 1D time axis into one with a 2D time axis.'''
  return SplitTime (var, iaxis)
# }}}


# Join a 2D time representation (year,<everything else>) into a single
# 1D time axis.
class JoinTime(Var):
# {{{
  def __init__(self, var, yaxis, daxis):
# {{{
    from pygeode.var import copy_meta, Var

    yaxis = var.whichaxis(yaxis)
    daxis = var.whichaxis(daxis)
    assert (yaxis < daxis)  # need a certain order

    years = var.getaxis(yaxis)
    days = var.getaxis(daxis)

    # Generate the joined axis
    taxis = _jointime(years, days)

    axes = list(var.axes)
    axes = axes[:daxis] + axes[daxis+1:]
    axes[yaxis] = taxis

    Var.__init__(self, axes=axes, dtype=var.dtype)
    copy_meta(var,self)

    self.yaxis = yaxis
    self.daxis = daxis
    self.var = var
# }}}

  def getview (self, view, pbar):
# {{{
    import numpy as np

    yaxis = self.yaxis
    daxis = self.daxis

    # Get the requested times.
    # Since the time axis is in the same position as 'years' when comparing
    # the input and output vars, we can do:
    taxis = yaxis
    times = view.subaxis(taxis)

    # Determine which years and (day-of-year?) these times correspond to
    years, days = _splittime(times)

    # The full input years/days
    in_years = self.var.getaxis(yaxis)
    in_days = self.var.getaxis(daxis)

    # Get the input data
    inview = view.replace_axis(taxis, years).add_axis(daxis, days,slice(None))
    data = inview.get(self.var, pbar=pbar)

    # Reshape the data so it has a single time axis.
    # First, move the year and day axes next to each other
    data = np.rollaxis(data, daxis, yaxis+1)
    # Then, combine then
    shape = list(data.shape)
    shape = shape[:yaxis] + [-1] + shape[daxis+1:]
    data = data.reshape(shape)

    # Now, we may have slightly more data than we actually need
    # (e.g., we may not actually want data at the start of the first year)
    # Define a time axis that represents *all* the data we just got:
    uber_time = _jointime(years, days)

    # Slice out what we actually wanted from this
    slices = [slice(None)] * data.ndim
    slices[taxis] = uber_time.map_to(times)
    return data[slices]
# }}}
# }}}

def jointimeaxes(var, yaxis='year', daxis='day'):
# {{{
  '''Convert a variable with a 2D time axis into one with a 1D time axis.'''
  return JoinTime(var, yaxis, daxis)
# }}}

del Var
