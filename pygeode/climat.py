__all__ = ('climatology', 'dailymean', 'monthlymean',
           'yearlymean','diurnalmean', 'seasonalmean',
           'nanclimatology', 'dailynanmean', 'monthlynanmean',
           'yearlynanmean','diurnalnanmean', 'seasonalnanmean',
           'climstdev', 'dailystdev', 'monthlystdev',
           'yearlystdev','diurnalstdev', 'seasonalstdev',
           'climnanstdev', 'dailynanstdev', 'monthlynanstdev',
           'yearlynanstdev','diurnalnanstdev', 'seasonalnanstdev',
           'climcount', 'dailycount', 'monthlycount',
           'yearlycount','diurnalcount', 'seasonalcount',
           'climtrend', 'from_trend')

from pygeode.var import Var

# Loop over a variable, applying the specified view.
# Outputs iterations of:
# - slices for fitting the data chunk into the output (after the partial reduction is done)
# - a chunk of output data to be partially reduced
# - bins along the time axis for reducing the data chunk
def loopover(varlist, view, pbar):
  from pygeode.timeaxis import Time
  import numpy as np
  from itertools import izip

  if not hasattr(varlist,'__len__'): varlist = [varlist]
  nvars = len(varlist)

  # clip the output view so there is nothing 'outside' of the selected region
  # (so we can use the slices of the current view chunk as a 1:1 correspondence to
  #  the slices into the accumulation array)
  view = view.clip()

  # map the climatology axis to the non-climatology axis
  ti = view.index(Time)
  ct = view.axes[ti]
  t = varlist[0].getaxis(Time)
  inmap, outmap = Time.common_map(t, ct)
  # We should be covering the entire output array (unless there's something horribly wrong with the logic above)
  assert np.all(np.unique(outmap) == np.unique(view.integer_indices[ti]))
  inview = view.replace_axis(Time, t, inmap)
  climview = view.replace_axis(Time, ct, outmap)  #??? do we need to do this?  can't we just set climview=view?
                                                  # (we might need to, if common_map changes the *order* of the axis elements)
                                                  # (I like talking to myself)
  # We want to loop over these two views simultaneously
  assert inview.shape == climview.shape
  # To avoid possible coding bugs, make sure the dimension number of the time axis is consistent between views
  assert inview.index(Time) == climview.index(Time)

  # Break the view up into memory-friendly chunks
  loop = zip(inview.loop_mem(), climview.loop_mem())
  # Relative sizes of each var - so we can more accurately divide progress
  sizes = [var.size for var in varlist]
  prog = np.cumsum([0.]+sizes) / np.sum(sizes) * 100
  for i, (inv, climv) in enumerate(loop):
    subpbar = pbar.part(i, len(loop))
    data = [inv.get(var, pbar = subpbar.subset(prog[j],prog[j+1])) for j,var in enumerate(varlist)]
    data = [np.ascontiguousarray(d) for d in data]
    slices = list(climv.slices)
    slices[ti] = slice(None)
    yield slices, data, climv.integer_indices[ti]



###############################################################################
# Define how the time unit is mapped

class TimeMap(Var):
  # Returns the output time axis
  @staticmethod
  def get_outtime (intime): return intime  # Overridden in subclasses
  # Returns the input time axis
  # (sometimes, we have to massage the input time axis into a form that makes
  #  the mapping easier  (see Seasonal class)
  @staticmethod
  def get_intime (intime): return intime

class Clim(TimeMap):
  name_suffix1 = '_clim'
  @staticmethod
  def get_outtime (intime):
    from pygeode.timeutils import modify
    return modify(intime, exclude='year', uniquify=True)

class Diurnal(TimeMap):
  name_suffix1 = '_diurnal'
  @staticmethod
  def get_outtime (intime):
    from pygeode.timeutils import modify
    return modify(intime, exclude=['year','month','day'], uniquify=True)

class Daily(TimeMap):
  name_suffix1 = '_daily'
  @staticmethod
  def get_outtime (intime):
    from pygeode.timeutils import modify
    outtime = modify(intime, resolution='day', uniquify=True)
    assert hasattr(outtime, 'day')  # can we even do a daily mean?
    return outtime

class Monthly(TimeMap):
  name_suffix1 = '_monthly'
  @staticmethod
  def get_outtime (intime):
    from pygeode.timeutils import modify
    outtime = modify(intime, resolution='month', uniquify=True)
    assert hasattr(outtime, 'month')  # can we even do a monthly mean?
    return outtime

class Yearly(TimeMap):
  name_suffix1 = '_yearly'
  @staticmethod
  def get_outtime (intime):
    from pygeode.timeutils import modify
    outtime = modify(intime, resolution='year', uniquify=True)
    assert hasattr(outtime, 'year')  # can we even do a yearly mean?
    return outtime

# Note: in the seasonal stuff below, 'syear' represents a year field that has
# been adjusted so DJFs are contiguous (see 'Seasonal' class below)

# Mapping to a seasonal time axis
class Seasonal(TimeMap):
  name_suffix1 = '_seasonal'

  @classmethod
  def get_outtime (cls, intime):
    from pygeode.timeaxis import SeasonalTAxes
    from pygeode.timeutils import _uniquify
    try:
      SeasonalAxis = SeasonalTAxes[intime.__class__]
    except:
      raise NotImplemented('Seasonal Time axis not implemented for this calendar')

    # Wrap in axis with a seasonal axis to allow reduction
    outax = SeasonalAxis(values = intime.values, \
                        startdate = intime.startdate, \
                        units = intime.units)
    if 'year' in outax.auxarrays:
      fields = [outax.auxarrays['year'], outax.auxarrays['season']]
      fields = _uniquify(fields)
      auxarrays = {'year':fields[0], 'season':fields[1]}
    else:
      fields = [outax.auxarrays['season']]
      fields = _uniquify(fields)
      auxarrays = {'season':fields[0]}

    return SeasonalAxis (startdate=intime.startdate, units=intime.units, **auxarrays)

  @classmethod
  # Endow input time axis with a season (and a season-aligned year)
  def get_intime (cls, intime):
    from pygeode.timeaxis import SeasonalTAxes
    try:
      SeasonalAxis = SeasonalTAxes[intime.__class__]
    except:
      raise NotImplemented('Seasonal Time axis not implemented for this calendar')

    # Wrap in axis with a seasonal axis to allow reduction
    return SeasonalAxis(values = intime.values, \
                        startdate = intime.startdate, \
                        units = intime.units)

###############################################################################
# Define the operation that's done in the mapping

class TimeOp(Var):
# {{{
  extra_dims = ()  # Override this in a subclass if needed (see Trend class)
  def __init__ (self, var):
    from pygeode.var import Var
    from pygeode.timeaxis import CalendarTime
    self.ti  = ti = var.whichaxis(CalendarTime)
    intime = var.axes[ti]
    outtime = self.get_outtime(intime)
    # Fudge the input axis?  (endow it with extra information that it normally wouldn't have?)
    new_intime = self.get_intime(intime)
    if new_intime is not intime: var = var.replace_axes({CalendarTime:new_intime})

    if var.name != '': self.name = var.name + self.name_suffix1 + self.name_suffix2

    self.var = var

    axes = list(var.axes)
    axes[ti] = outtime
    Var.__init__ (self, axes+list(self.extra_dims), dtype = var.dtype, atts=var.atts)
# }}}

class Mean(TimeOp):
# {{{
  name_suffix2 = '_mean'

  def getview (self, view, pbar):
    from pygeode.tools import partial_sum
    import numpy as np

    ti = self.ti

    sum = np.zeros (view.shape, self.dtype)
    count = np.zeros (view.shape, dtype='int32')

    for slices, [data], bins in loopover (self.var, view, pbar):
      partial_sum (data, slices, sum, count, ti, bins)

    sum /= count
    return sum
# }}}

class NANMean(TimeOp):
# {{{
  name_suffix2 = '_nanmean'

  def getview (self, view, pbar):
    from pygeode.tools import partial_nan_sum
    import numpy as np

    ti = self.ti

    sum = np.zeros (view.shape, self.dtype)
    count = np.zeros (view.shape, dtype='int32')

    for slices, [data], bins in loopover (self.var, view, pbar):
      partial_nan_sum (data, slices, sum, count, ti, bins)

    inodata = (count == 0)
    if np.any(inodata):
      count = count.astype('d')
      count[inodata] = np.nan

    sum /= count
    return sum
# }}}

class Stdev(TimeOp):
# {{{
  name_suffix2 = '_stdev'

  def getview (self, view, pbar):
    from pygeode.tools import partial_sum
    import numpy as np

    ti = self.ti

    xx = np.zeros (view.shape, self.dtype)
    x = np.zeros (view.shape, self.dtype)
    nx = np.zeros (view.shape, dtype='int32')
    nx2 = np.zeros (view.shape, dtype='int32')

    for slices, [data], bins in loopover (self.var, view, pbar):
      partial_sum (data, slices, x, nx, ti, bins)
      partial_sum (data**2, slices, xx, nx2, ti, bins)

    x /= nx
    var = (xx - nx*x**2) / (nx - 1)
    return np.sqrt(var)
# }}}

class NANStdev(TimeOp):
# {{{
  name_suffix2 = '_nanstdev'

  def getview (self, view, pbar):
    from pygeode.tools import partial_nan_sum
    import numpy as np

    ti = self.ti

    xx = np.zeros (view.shape, self.dtype)
    x = np.zeros (view.shape, self.dtype)
    nx = np.zeros (view.shape, dtype='int32')
    nx2 = np.zeros (view.shape, dtype='int32')

    for slices, [data], bins in loopover (self.var, view, pbar):
      partial_nan_sum (data, slices, x, nx, ti, bins)
      partial_nan_sum (data**2, slices, xx, nx2, ti, bins)

    inodata = (nx == 0)
    if np.any(inodata):
      nx = nx.astype('d')
      nx[inodata] = np.nan

    x /= nx
    var = (xx - nx*x**2) / (nx - 1)
    return np.sqrt(var)
# }}}

class Count(TimeOp):
# {{{
  name_suffix2 = '_nancount'

  def getview (self, view, pbar):
    from pygeode.tools import partial_nan_sum
    import numpy as np

    ti = self.ti

    sum = np.zeros (view.shape, self.dtype)
    count = np.zeros (view.shape, dtype='int32')

    for slices, [data], bins in loopover (self.var, view, pbar):
      partial_nan_sum (data, slices, sum, count, ti, bins)

    return count.astype(self.dtype)
# }}}

class Trend(TimeOp):
# {{{
  name_suffix2 = '_trend'
  from pygeode.axis import Coef
  extra_dims = (Coef(2),)
  del Coef

  def __init__ (self, var):
    # verify that we have more than one timestep, otherwise the 'trend' is not well defined!
    TimeOp.__init__ (self, var)
    assert var.shape[self.ti] > 1, "need more than one timestep for a trend"

  def getview (self, view, pbar):
    from pygeode.tools import partial_sum
    import numpy as np
    from pygeode.axis import Coef
    from pygeode.var import Var

    ti = self.ti
    taxis = self.var.axes[ti]
    # Get number of seconds since start of data
    secs = taxis.reltime(units='seconds')
    # Wrap it as a var, so we can use it in the loop below
    secs = Var([taxis], values=secs)

    cview = view.remove(Coef)  # view without regard to a 'coefficient' axis

    X = np.zeros(cview.shape, self.dtype)
    nX = np.zeros(cview.shape, int)
    F = np.zeros(cview.shape, self.dtype)
    nF = np.zeros(cview.shape, int)
    XF = np.zeros(cview.shape, self.dtype)
    nXF = np.zeros(cview.shape, int)
    X2 = np.zeros(cview.shape, self.dtype)
    nX2 = np.zeros(cview.shape, int)


    for slices, (data,t), bins in loopover ([self.var, secs], cview, pbar):
      partial_sum (data,   slices, F,  nF,  ti, bins)
      partial_sum (t,      slices, X,  nX,  ti, bins)
      partial_sum (data*t, slices, XF, nXF, ti, bins)
      partial_sum (t**2,   slices, X2, nX2, ti, bins)

    F /= nF
    X /= nX
    XF /= nXF
    X2 /= nX2

#    print '??', X2 - X**2

    A = XF - X*F
    B = X2*F - X*XF

    icoef = view.index(Coef)
    coef = view.integer_indices[icoef]

    out = np.empty(view.shape, self.dtype)

    #  Stick the two coefficients together into a single array
    out[...,np.where(coef==0)[0]] = B[...,None]
    out[...,np.where(coef==1)[0]] = A[...,None]

    out /= (X2 - X**2)[...,None]

    return out
# }}}

###############################################################################
# Combine the above classes

class climatology(Clim,Mean):
  """
  Computes a climatological mean.  Averages over all years, returning a single
  value for each distinct month, day, hour, etc.
  """
class dailymean(Daily,Mean):
  """
  Computes an average value for each day.
  """
class monthlymean(Monthly,Mean):
  """
  Averages over each month.
  """
class yearlymean(Yearly,Mean):
  """
  Averages over each year.
  """
class diurnalmean(Diurnal,Mean):
  """
  Computes an average value for each time of day (averages over all years,
  months, days).
  """
class seasonalmean(Seasonal,Mean):
  """
  Averages over each season.  Currently, the seasons are hard-coded as (DJF,
  MAM, JJA, SON).
  """

class nanclimatology(Clim,NANMean):
  """
  Computes a climatological nan-aware mean.  Averages over all years, returning a single
  value for each distinct month, day, hour, etc.
  """
class dailynanmean(Daily,NANMean):
  """
  Computes a nan-aware average value for each day.
  """
class monthlynanmean(Monthly,NANMean):
  """
  Nan-aware Averages over each month.
  """
class yearlynanmean(Yearly,NANMean):
  """
  Nan-aware Averages over each year.
  """
class diurnalnanmean(Diurnal,NANMean):
  """
  Computes a nan-aware average value for each time of day (averages over all years,
  months, days).
  """
class seasonalnanmean(Seasonal,NANMean):
  """
  Nan-aware averages over each season.  Currently, the seasons are hard-coded as (DJF,
  MAM, JJA, SON).
  """

class climtrend(Clim,Trend):
  """
  For each month, day, hour, etc., compute a least-squares fit to a linear trend
  over all years.  This is similar to the climatology, but instead of averaging
  over all years, it computes the rate of change over all years.
  """

class climstdev(Clim,Stdev):
  """
  Computes a climatological standard deviation. Computes standard deviation over all years,
  returning a single value for each distinct month, day, hour, etc.
  """
class dailystdev(Daily,Stdev):
  """ Computes daily standard deviation. """

class monthlystdev(Monthly,Stdev):
  """ Computes monthly standard deviation. """

class seasonalstdev(Seasonal,Stdev):
  """ Computes seasonal standard deviation. """

class yearlystdev(Yearly,Stdev):
  """ Computes yearly standard deviation. """

class diurnalstdev(Diurnal,Stdev):
  """ Computes diurnal standard deviation. """

class climnanstdev(Clim,NANStdev):
  """ Computes nan-aware climatological standard deviation. """

class dailynanstdev(Daily,NANStdev):
  """ Computes nan-aware daily standard deviation. """

class monthlynanstdev(Monthly,NANStdev):
  """ Computes nan-aware monthly standard deviation. """

class seasonalnanstdev(Seasonal,NANStdev):
  """ Computes nan-aware seasonal standard deviation. """

class yearlynanstdev(Yearly,NANStdev):
  """ Computes nan-aware yearly standard deviation. """

class diurnalnanstdev(Diurnal,NANStdev):
  """ Computes nan-aware diurnal standard deviation. """

class climcount(Clim,Count):
  """ Counts number of non-nan data points contributing to climatology. """

class dailycount(Daily,Count):
  """ Counts number of non-nan data points contributing to daily mean. """

class monthlycount(Monthly,Count):
  """ Counts number of non-nan data points contributing to monthly mean. """

class seasonalcount(Seasonal,Count):
  """ Counts number of non-nan data points contributing to seasonal mean. """

class yearlycount(Yearly,Count):
  """ Counts number of non-nan data points contributing to yearly mean. """

class diurnalcount(Diurnal,Count):
  """ Counts number of non-nan data points contributing to diurnal mean. """


# Reconstruct a linear dataset given the trend coefficients
# I.e., A*t + B
class from_trend (Var):
# {{{
  """
  Reconstructs linear timeseries from a given trend.
  """
  def __init__ (self, taxis, coef=None, A=None, B=None):
    from pygeode.tools import merge_coefs
    from pygeode.var import Var
    from pygeode.timeaxis import Time
    from pygeode.timeutils import modify
    # Get the coefficients
    if coef is None:
      assert A is not None and B is not None
      coef = merge_coefs (B, A)
    else: assert A is None and B is None
    # Ignore 'year' field if it's constant?
    #   (I.e., if there's a 'year' field that's all zeros, then drop it)
    #  - This is an artifact of reading climatological data from a file which can't/doesn't specify it's a climatology
    coeft = coef.getaxis(Time)
    if hasattr(coeft,'year'):
      import numpy as np
      from warnings import warn
      if len(np.unique(coeft.year.values)) == 1:
        warn ("ignoring degenerate 'year' field", stacklevel=2)
        coeft = modify(coeft, exclude='year')
        coef = coef.replace_axes({Time:coeft})

    self.coef = coef
    self.secs = Var([taxis], values=taxis.reltime(units='seconds'))
    self.ti = ti = coef.whichaxis(Time)
    self.ci = ci = coef.whichaxis('coef')
    self.caxis = coef.axes[ci]
    axes = list(coef.axes)
    assert axes[ti].map_to(taxis) is not None, (
      "the given time axis is not compatible with the coefficients.\n"+
      "time axis: %s\n coefficient time axis: %s" % (repr(str(taxis)),repr(str(axes[ti])))
    )
    axes[ti] = taxis
    axes = axes[:ci] + axes[ci+1:]
#    Var.__init__(self, axes, dtype=coef.dtype)
    Var.__init__(self, axes, dtype='float64')  # because secs is float64
  def getview (self, view, pbar):
    cview = view.add_axis(0, self.caxis, slice(None))
    coef = cview.get(self.coef, pbar=pbar)
    B = coef[0,...]
    A = coef[1,...]
    secs = view.get(self.secs)
    return A*secs + B
# }}}

del Var

# Remove the climatological trend from the data
# NOTE: precomputes climatology and stores it in memory
# Don't use this if the climatology won't fit in memory.
# Instead, call 'climtrend', save the output to disk, and then feed that into 'from_trend'
from pygeode.timeaxis import Time
def detrend (var, taxis=Time, return_clim = False):
# {{{
  clim = from_trend(var.getaxis(taxis), climtrend(var).load())
  clim.name = 'clim'
  out = var - clim
  if return_clim is True: return out, clim
  else: return out
# }}}

del Time
