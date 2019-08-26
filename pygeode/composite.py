#!/usr/bin/python

# Composite axis

# Note: since negative offsets can't be used directly in a slice (they mean something else),
# you need to use the 'select' command.
# i.e., for a dataset with 10 timesteps, the offsets available are
#  -9 to 9  ( [-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
# so, to use offset -8, do var.select(offset=-8)

from pygeode.var import Var
from pygeode.axis import Axis, NamedAxis
from pygeode.timeaxis import ModelTime365

class Event(Axis): name = 'event'
class Offset(Axis): name = 'offset'

class CompositeVar(Var):
  '''CompositeVar(var, caxis, cinds)
    Creates a new view of pygeode variable var'''

  def __init__ (self, var, iaxis, ievents, evlen, evoff=0, saxes=None, sindices=None):
  #{{{
    # Replace the time axis with a reference time (of an event?), and the offset
    # from that event.
    import numpy as np
    ievents = np.array(ievents)
    n = ievents.shape[0]
    caxis = var.axes[iaxis]

    # Event offsets can either be specified per event or as a single offset
    if hasattr(evoff, '__len__'):
      evoff = np.array(evoff)
      mevoff = evoff.max()
      assert evoff.shape == ievents.shape, "The number of event offsets provided does not match the number of events."
    else:
      mevoff = evoff
      evoff = np.ones(n, 'i') * mevoff

    # Event lengths can either be specified per event or as a single length
    if hasattr(evlen, '__len__'):
      evlen = np.array(evlen)
      mevlen = (evlen - evoff + mevoff).max()
      assert evlen.shape == ievents.shape, "The number of event lengths provided does not match the number of events."
    else:
      mevlen = evlen
      evlen = np.ones(n, 'i') * mevlen

    # Construct event and offset axes
    from pygeode.timeaxis import Time, Yearless
    from pygeode import timeutils
    ev = Event(np.arange(n)+1, indices=ievents)
    if isinstance(caxis, Time):
      units = caxis.units
      delta = timeutils.delta(caxis, units = units)
      off = Yearless(values=delta*np.arange(-mevoff, mevlen-mevoff), units=units, startdate={'day':0})
    else:
      off = Offset(np.arange(-mevoff, mevlen-mevoff))
    axes = var.axes[:iaxis] + (ev, off) + var.axes[iaxis+1:]

    # Build var object
    self.var = var
    self.iaxis = iaxis
    self.evlens = evlen
    self.mevlen = mevlen
    self.evoffs = evoff
    self.mevoff = mevoff

    for i, (iev, el, eo) in enumerate(zip(ievents, evlen, evoff)):
      if iev - eo < 0:
         self.evoffs[i] += iev - eo
         self.evlens[i] -= iev - eo
      if iev - eo + el >= len(caxis):
         self.evlens[i] = len(caxis) - (iev - eo)

      #assert iev - eo >= 0 and iev - eo + el < len(caxis), \
         #'Event %d (i: %d) is not fully defined' % (np.where(ievents==iev)[0][0], iev)
    Var.__init__(self, axes, dtype=var.dtype, name=var.name, atts=var.atts, plotatts=var.plotatts)
  #}}}

  def getview (self, view, pbar):
  #{{{
    import numpy as np

    src = self.var
    iev = self.iaxis     # Index of event axis
    iof = iev + 1        # Index of offset axis
    icm = iev            # Index of composited (source) axis

    # We will need to slice into the output array
    out = np.empty(view.shape, self.dtype)
    out[:] = np.nan
    outsl = [slice(None) for i in range(self.naxes)]
    iofsl = view.clip().integer_indices[iof]

    # Prepare view on composited variable
    icmsl = view.integer_indices[iof] - self.mevoff
    inviewt = view.remove(iev, iof).add_axis(icm, src.axes[icm], slice(None))

    # Loop over events, get requested subset of composited axis
    ievsl = view.integer_indices[iev]
    events = self.axes[iev].indices[ievsl]
    evlens = self.evlens[ievsl]
    evoffs = self.evoffs[ievsl]
    n = len(ievsl)

    progs = np.linspace(0, 100, n+1)
    for i, ev, el, eo in zip(np.arange(n), events, evlens, evoffs):
      # Construct slice on composite variable for this event
      mask = (icmsl >= -eo) & (icmsl < el-eo)
      icsl = icmsl[mask] + ev
      inview = inviewt.modify_slice(icm, icsl)

      # Construct slice into output array; clip to this events' duration
      outsl[iev] = i
      outsl[iof] = iofsl[mask]
      out[tuple(outsl)] = inview.get(src, pbar=pbar.subset(progs[i], progs[i+1]))

    return out
  #}}}

def composite (var, **kwargs):
# {{{
  '''Creates a composite based on this variable.

  Parameters
  ----------
  <axis selection> : string
    A single axis selection string (similar to :func:`Var.__call__`) that
    specifies the central 'date' of each event to composite (although
    composites need not be constructed along time axes). See Notes.
  evlen : int
    Length of segement around each central 'date' to extract.
  evoff : int
    Offset from the central 'dates' to include. A positive value
    will lead to dates prior to the central 'date' being included.

  Returns
  -------
  cvar : :class:`Var`
    Composite variable. The axis along which composites are to be constructed
    is replaced by an Event axis and an Offset axis.

  Notes
  -----
  The axis matched is used as the composite axis and the returned indices
  are the key dates. evlen is required; it can either be an integer or a
  list of integers specifying the length of each event. evoff is a single integer.
  If the requested composite extends past the ends of the underlying axis, the
  variable will contain NaNs.

  Examples
  ========
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2
  >>> dts = ['12 May 2012', '15 Aug 2015', '1 Feb 2018', '28 Dec 2020']
  >>> cT = t2.Temp.composite(l_time = dts, evlen = 15, evoff = 10)
  >>> print(cT)
  <Var 'Temp'>:
    Shape:  (event,time,pres,lat,lon)  (4,15,20,31,60)
    Axes:
      event <Event>  :  1  to 4  (4 values)
      time <Yearless>:  day -10, 00:00:00 to day 4, 00:00:00 (15 values)
      pres <Pres>    :  1000 hPa to 50 hPa (20 values)
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  CompositeVar (dtype="float64")
  >>> cT(s_event=1, s_pres=50, s_lat=0, s_lon = 180)[:]
  array([199.66766628, 199.69594407, 199.72479591, 199.75418884,
         199.78408917, 199.81446257, 199.84527408, 199.87648814,
         199.90806866, 199.939979  , 199.97218208, 200.00464036,
         200.03731592, 200.07017048, 200.10316548])
  >>> cT(s_event=4, s_pres=50, s_lat=0, s_lon = 180)[:]
  array([201.02847025, 201.04367089, 201.05782423, 201.07091237,
         201.08291874, 201.09382812, 201.10362669, 201.112302  ,
         201.11984303, 201.12624021, 201.1314854 , 201.13557193,
         201.1384946 , 201.14024969,          nan])

  '''

  from pygeode.view import expand
  evlen = kwargs.pop('evlen')
  evoff = kwargs.pop('evoff', 0)
  assert len(kwargs) == 1, 'Must specify event centres.'

  iaxis = -1
  for i, ax in enumerate(var.axes):
    try:
      sl = ax.get_slice(kwargs.copy())
    except Exception as e:
      continue
    iaxis = i
    ievents = expand(sl, len(ax))

  assert 0 <= iaxis < var.naxes, 'Unrecognized axis'
  if hasattr(evlen, '__len__'):
    assert len(evlen) == len(ievents), 'Must specify event length for each event'
  if hasattr(evoff, '__len__'):
    assert len(evoff) == len(ievents), 'Must specify event offset for each event'

  return CompositeVar (var, iaxis, ievents, evlen, evoff)
# }}}

class FlattenVar(Var):
  '''FlattenVar(var, outer, inner)
    Creates a new view of pygeode variable var'''

  def __init__ (self, var, inner = -1, naxis=None):
  #{{{
    from numpy import concatenate
    if inner == -1:
      if isinstance(var, CompositeVar):
         out = var.whichaxis(var.event.__class__)
         inner = var.whichaxis(var.offset.__class__)
         stride = len(var.axes[inner])
         vals = concatenate([e + var.axes[inner].values for e in var.axes[out].values])
      else:
         raise NotImplementedError("You must specify which axis to flatten")
    elif inner == 0: raise NotImplementedError("inner axis must not be the outermost")
    else:
      out = inner - 1
      stride = len(var.axes[inner])
      vals = concatenate([i*stride + var.axes[inner].values for i in range(len(var.axes[out]))])

    if naxis is None:
      if isinstance(var.axes[out], NamedAxis):
        naxis = var.axes[out].__class__(vals, var.axes[out].name)
      else:
        naxis = var.axes[out].__class__(vals)
    axes = var.axes[:out] + (naxis,) + var.axes[inner+1:]
    self.iout = out
    self.iin = inner
    self.stride = stride
    self.source_var = var

    self.name = var.name

    Var.__init__(self, axes, var.dtype)
  #}}}

  def getview (self, view, pbar):
  # {{{
    # TODO: handle strides, random access...
    io = self.iout  # Index of outer axis in source var
    ii = self.iin   # Index of inner axis in source var
    ifl = self.iout # Index of flattened axis
    stride = self.stride
    src = self.source_var

    import numpy as np
    out = np.empty(view.shape, self.dtype)
    slf = view.slices[ifl]  # Again, need these indices in the space of the variable (?)
    slo = slice(int(np.floor(slf.start / float(stride))),
                          int(np.ceil(slf.stop / float(stride))))

    inviewt = view.replace_axis(ifl, src.axes[io], slice(None))
    inviewt = inviewt.add_axis(ii, src.axes[ii], slice(None))

    sllen = lambda sl: sl.stop-sl.start
    if sllen(slo) == 1: # Request lies within a single stride
      # Build inner slice
      sli = slice(slf.start - slo.start * stride, slf.stop - slo.start * stride)

      # Set bounds on inner view
      inview = inviewt.modify_slice(io, slo).modify_slice(ii, sli)

      return inview.get(src, pbar=pbar.subset(0, 100)).reshape(view.shape) # Get data, reshape
    else:     # Request spans multiple strides;
      # Need to get interior slice, initial and final partial strides
      # Build slices
      slos = [slice(slo.start, slo.start + 1),  # Initial
              slice(slo.start + 1, slo.stop - 1),  # Interior (should be degenerate if slo.cnt=2)
              slice(slo.stop - 1, slo.stop)]  # Final

      slis = [slice(slf.start - slo.start * stride, stride),
              slice(0, stride),
              slice(0, slf.stop - (slo.stop - 1) * stride)]

      # Compute appropriate shape of each request
      shp = list(view.shape)
      shapes = [list(shp) for shp[ifl] in [sllen(slis[0]), sllen(slos[1]) * stride, sllen(slis[2])]]
      progs = np.cumsum([0] + [s[io] for s in shapes])
      progs = [(100.*progs[i]/progs[-1], 100.*progs[i+1]/progs[-1]) for i in range(len(progs) - 1)]

      # Construct views
      inviews = [inviewt.modify_slice(io, slo).modify_slice(ii, sli) for slo, sli in zip(slos, slis)]

      # Get data, reshape, concatenate along flattened axis
      return np.concatenate([inv.get(src, pbar=pbar.subset(*p)).reshape(shp) for inv, shp, p in zip(inviews, shapes, progs)], ifl)
  # }}}

def flatten (var, inner = -1, naxis=None):
  return FlattenVar (var, inner, naxis)

#====

def clim_detrend(var, yrlen, itime = -1, sig=False):
# {{{
  ''' clim_detrend() - returns detrended time series with a daily trend.'''
  from pygeode.timeaxis import Time
  from . import stats
  from numpy import arange
  if itime == -1: itime = var.whichaxis(Time)
  tlen = var.shape[itime]

  vary = composite(var, itime, list(range(0, tlen, yrlen)), yrlen)
  yrs = vary.axes[itime]
  yrs.values=arange(len(yrs)).astype(yrs.dtype)

  print('Computing regression')
  from pygeode.progress import PBar
  m, b, p = stats.regress(yrs, vary, pbar=PBar())
  varz = flatten(vary - (m*yrs + b), itime + 1)

  varz.axes = var.axes

  # Since the axes have been modified after initialization, redo the init to get
  # shortcuts to the axes names
  Var.__init__(varz, varz.axes, varz.dtype)

  if var.name != '':
    varz.name = var.name+"'"

  if sig:
    return m, b, varz, p
  else:
    return m, b, varz
# }}}

def clim_anoms(var, yrlen, itime = -1):
# {{{
  ''' clim_anoms() - quick and dirty implementation;
        returns climatology and anomalies of given variable.'''
  from pygeode.timeaxis import Time
  if itime == -1: itime = var.whichaxis(Time)
  tlen = (var.shape[itime] // yrlen) * yrlen
  vary = composite(var, itime, list(range(0, tlen, yrlen)), yrlen)
  varc = vary.mean(itime).load()
  varz = flatten(vary - varc, itime + 1)
  varz.axes = var.axes
  # Since the axes have been modified after initialization, redo the init to get
  # shortcuts to the axes names
  Var.__init__(varz, varz.axes, varz.dtype)
  if var.name != '':
    varz.name = var.name+'_anom'
  return varc, varz
# }}}

def time_ave(var, type, itime = -1):
# {{{
  ''' time_ave() - quick and dirty implementation of time averaging;
            for now assumes 6h data (!). type can be one of the following:
            'd' - daily averages
            '5d' - 5-day averages
            'm' - monthly averages
            'y' - annual averages '''
  from pygeode.timeaxis import Time
  if itime == -1: itime = var.whichaxis(Time)
  tlen = var.shape[itime]

  if type == 'd':
    ilen = 4
  elif type == '5d':
    ilen = 20
  elif type == 'm':
    raise NotImplemented('Monthly averaging not yet implemented')
  elif type == 'y':
    ilen = 1460
  else:
    raise NotImplemented('Unrecognized type')

  vary = composite(var, itime, list(range(0, tlen, ilen)), ilen)
  varave = vary.mean(itime + 1)
  return varave
# }}}

def test():

  from pygeode.formats.cccma import ccc_open

  dataset = ccc_open ("../data/mm_s40a_001_m01_gs")

  gt = dataset.GT
  gt = composite(gt, [0,10,20,30])

  print(gt)

  from .plot import plot

  x1 = gt[1:4,25:30,:,:]

#test()
