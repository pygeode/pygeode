from pygeode import timeutils as tu

class TickGenerator:
# {{{
  fmt = None
  ofmt = None
  def __init__(self, taxis, mults = [1], fmt = None, ofmt = None):
  # {{{
    self._taxis = taxis
    self.mults = mults
    self.imult = 0
    self.mult = self.mults[0]
    if fmt is not None: 
      self.fmt = fmt
    if ofmt is not None: 
      self.ofmt = ofmt
  # }}}

  def nticks(self, vmin, vmax, mult=None):
  # {{{
    # Default implementation is general, but can be slow if there
    # are many ticks btw. vmin and vmax
    return len([t for t in self.ticks(vmin, vmax, mult)])
  # }}}

  def ticks(self, vmin, vmax, mult=None):
  # {{{
    if mult is not None: self.mult = mult
    d1 = self.get_tick_prior(vmin)
    d2 = self.get_tick_prior(vmax)

    v = self._taxis.date_as_val(d1)
    vf = self._taxis.date_as_val(d2)

    while v < vf:
      v = self.next_tick(v)
      yield v
  # }}}

  def best_mult(self, vmin, vmax, nticks):
  # {{{
    if self.mults[-1] == 'p': # 
      mults = self.mults[:-1]
    elif self.mults[-1] == 'm':
      mults = self.mults[:-1]
    else:
      mults = self.mults
    
    i = self.imult
    nt = self.nticks(vmin, vmax, mults[i])
    while True:
      if nt > nticks and i > 0:
        nt2 = self.nticks(vmin, vmax, mults[i-1])
        if nt - nticks <= nticks - nt2: break
        nt = nt2
        i -= 1
        if nt2 <= nticks: break
      elif nt < nticks and i < len(mults) - 1:
        nt2 = self.nticks(vmin, vmax, mults[i+1])
        if nticks - nt < nt2 - nticks: break
        nt = nt2
        i += 1
        if nt2 > nticks: break
      else: break

    self.imult = i
    self.mult = self.mults[i]
    return nt, i
  # }}}
# }}}

#########################################
## Class YearTickGen
class YearTickGen(TickGenerator):
# {{{
  ''' YearTickGen() - locator helper object which ticks at intervals
  of a multiple of years.'''

  fmt = '$Y'
  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the year of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the index of the
    previous tick.'''

    dt = self._taxis.val_as_date(val, allfields=True)
    yr = dt.get('year', 1)
    if val > self._taxis.date_as_val({'year':yr}): yr += 1

    from numpy import floor_divide
    return {'year':floor_divide(yr-1, self.mult) * self.mult}
  # }}}

  def next_tick(self, val):
  # {{{
    d = self._taxis.val_as_date(val, allfields=True)
    d['year'] += self.mult
    return self._taxis.date_as_val(d)
  # }}}

  def nticks(self, vmin, vmax, mult=None):
  #  {{{
    if mult is not None: self.mult = mult
    y1 = self.get_tick_prior(vmin)['year']
    y2 = self.get_tick_prior(vmax)['year']

    return (y2 - y1) / self.mult
   # }}}
# }}}
    
#########################################
## Class MonthTickGen
class MonthTickGen(TickGenerator):
# {{{
  ''' MonthTickGen() - locator helper object which ticks at intervals
  of a multiple of months.'''
  fmt = '$b $Y'
  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the date of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the date of the
    previous tick. The date is in the form of a dictionary with 'year' and
    'month' fields.'''

    dt = self._taxis.val_as_date(val, allfields=True)
    yr = dt.get('year', 1)
    mn = dt.get('month', 1)
    if val > self._taxis.date_as_val({'year':yr, 'month':mn}): 
      dt = tu.wrapdate(self._taxis, {'year':yr, 'month':mn+1}, allfields=True)
      yr = dt.get('year', 1)
      mn = dt.get('month', 1)

    # Find first month on given multiple prior to the given month
    from numpy import floor_divide
    mn = floor_divide(mn - 2, self.mult) * self.mult + 1

    return tu.wrapdate(self._taxis, {'year':yr, 'month':mn}, allfields=True)
  # }}}

  def next_tick(self, val):
  # {{{
    d = self._taxis.val_as_date(val, allfields=True)
    d['month'] += self.mult
    return self._taxis.date_as_val(d)
  # }}}

  def nticks(self, vmin, vmax, mult=None):
  #  {{{
    if mult is not None: self.mult = mult
    d1 = self.get_tick_prior(vmin)
    d2 = self.get_tick_prior(vmax)

    y1, m1 = d1['year'], d1['month']
    y2, m2 = d2['year'], d2['month']
    
    # NB: simpler, but not good enough if mult is not a factor of 12
    months = (12 * y2 + m2) - (12 * y1 + m1) 
    return months / self.mult
  # }}}
# }}}

#########################################
## Class SeasonTickGen
class SeasonTickGen(TickGenerator):
# {{{
  ''' SeasonTickGen() - locator helper object which ticks at intervals
    of a multiple of seasons.'''
  fmt = '$s $Y'
  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the date of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the date of the
    previous tick. The date is in the form of a dictionary with 'year' and
    'season' fields.'''

    dt = self._taxis.val_as_date(val, allfields=True)
    yr = dt.get('year', 1)
    sn = dt.get('season', 1)
    if val > self._taxis.date_as_val({'year':yr, 'season':sn}): 
      dt = tu.wrapdate(self._taxis, {'year':yr, 'month':sn+1}, allfields=True)
      yr = dt.get('year', 1)
      sn = dt.get('season', 1)

    # Find first month on given multiple prior to the given month
    from numpy import floor_divide
    sn = floor_divide(sn - 2, self.mult) * self.mult + 1

    return tu.wrapdate(self._taxis, {'year':yr, 'season':sn}, allfields=True)
  # }}}

  def next_tick(self, val):
  # {{{
    d = self._taxis.val_as_date(val, allfields=True)
    d['season'] += self.mult
    return self._taxis.date_as_val(d)
  # }}}

  def nticks(self, vmin, vmax, mult=None):
  #  {{{
    if mult is not None: self.mult = mult
    d1 = self.get_tick_prior(vmin)
    d2 = self.get_tick_prior(vmax)

    y1, s1 = d1['year'], d1['season']
    y2, s2 = d2['year'], d2['season']
    
    # NB: simpler, but not good enough if mult is not a factor of nseas
    nseas = self._taxis.nseasons
    seasons = (nseas * y2 + s2) - (nseas * y1 + s1) 
    return seasons / self.mult
  # }}}
# }}}

#########################################
## Class DayOfMonthTickGen
class DayOfMonthTickGen(TickGenerator):
# {{{
  ''' DayOfMonthTickGen() - locator helper object which ticks at intervals
  of a multiple of the day of the year.'''
  fmt = '$d $b'
  ofmt = '$Y'

  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the date of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the date of the
    previous tick.'''

    def unpack(dt): return dt.get('year', 1), dt.get('month', 1), dt.get('day', 1)

    dt = self._taxis.val_as_date(val, allfields=True)
    yr, mn, dy = unpack(dt)

    if val > self._taxis.date_as_val({'year':yr, 'month':mn, 'day':dy}): 
      yr, mn, dy = unpack(tu.wrapdate(self._taxis, {'year':yr, 'month':mn, 'day':dy+1}, allfields=True))

    # Find first day on given multiple prior to the given day
    from numpy import floor_divide
    dy = floor_divide(dy - 2, self.mult) * self.mult + 1

    # If we've wrapped, decrement the year
    d = tu.wrapdate(self._taxis, {'year':yr, 'month':mn, 'day':dy}, allfields=True)
    d1 = tu.wrapdate(self._taxis, {'year':yr, 'month':mn + 1, 'day':1}, allfields=True)

    if tu.date_diff(self._taxis, d, d1, 'days') < self.mult / 2:
      return d1
    else:
      return d
  # }}}

  def next_tick(self, val):
  # {{{
    d = self._taxis.val_as_date(val, allfields=True)
    d1 = d.copy()
    d['day'] += self.mult
    d1['month'] += 1; d1['day'] = [1]
    if tu.date_diff(self._taxis, d, d1, 'days') < self.mult / 2.:
      return self._taxis.date_as_val(d1)
    else:
      return self._taxis.date_as_val(d)
  # }}}
# }}}

#########################################
## Class DayOfYearTickGen
class DayOfYearTickGen(TickGenerator):
# {{{
  ''' DayOfYearTickGen() - locator helper object which ticks at intervals
  of a multiple of the day of the year.'''
  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the date of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the date of the
    previous tick.'''
  # }}}

  def nticks(self, vmin, vmax):
  #  {{{
    pass
  # }}}

  def ticks(self, vmin, vmax):
  # {{{
    pass
  # }}}
# }}}

#########################################
## Class DayTickGen
class DayTickGen(TickGenerator):
# {{{ 
  ''' DayTickGen() - locator helper object which ticks at intervals
  of a multiple of days (with no sense of months or years).'''
  fmt = '$d'

  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the year of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the index of the
    previous tick.'''

    dt = self._taxis.val_as_date(val, allfields=True)
    d = dt.get('day', 1)
    if val > self._taxis.date_as_val({'day':d}): d += 1

    from numpy import floor_divide
    return {'day':floor_divide(d-1, self.mult) * self.mult}
  # }}}

  def next_tick(self, val):
  # {{{
    d = self._taxis.val_as_date(val, allfields=True)
    d['day'] += self.mult
    return self._taxis.date_as_val(d)
  # }}}

  def nticks(self, vmin, vmax, mult=None):
  #  {{{
    if mult is not None: self.mult = mult
    d1 = self.get_tick_prior(vmin)['day']
    d2 = self.get_tick_prior(vmax)['day']

    return (d2 - d1) / self.mult + 1
  # }}}

# }}}

#########################################
## Class HourTickGen
class HourTickGen(TickGenerator):
# {{{
  ''' HourTickGen() - locator helper object which ticks at intervals
  of a multiple of hours.'''
  fmt = '$H:$M'
  ofmt = '$d $b $Y'

  def nticks(self, vmin, vmax, mult=None):
  # {{{
    if mult is None: mult = self.mult
    d1 = self.get_tick_prior(vmin)
    d2 = self.get_tick_prior(vmax)

    return tu.date_diff(self._taxis, d1, d2, units='hours') / mult
  # }}}

  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the date of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the date of the
    previous tick.'''

    def unpack(dt): return dt.get('year', 1), dt.get('month', 1), dt.get('day', 1), dt.get('hour', 0)
    def pack(yr, mn, dy, hr): return {'year':yr, 'month':mn, 'day':dy, 'hour':hr}

    dt = self._taxis.val_as_date(val, allfields=True)
    yr, mn, dy, hr = unpack(dt)

    if val > self._taxis.date_as_val(pack(yr, mn, dy, hr)): 
      yr, mn, dy, hr = unpack(tu.wrapdate(self._taxis, pack(yr, mn, dy, hr+1), allfields=True))

    # Find first hour on given multiple prior to the given hour
    from numpy import floor_divide
    hr = floor_divide(hr - 1, self.mult) * self.mult

    # If we've wrapped, decrement the year
    d = tu.wrapdate(self._taxis, pack(yr, mn, dy, hr), allfields=True)
    d1 = tu.wrapdate(self._taxis, pack(yr, mn, dy+1, 0), allfields=True)

    if tu.date_diff(self._taxis, d, d1, 'hours') < self.mult / 2:
      return d1
    else:
      return d
  # }}}

  def next_tick(self, val):
  # {{{
    d = self._taxis.val_as_date(val, allfields=True)
    d1 = d.copy()
    d['hour'] += self.mult
    d1['day'] += 1; d1['hour'] = 0
    if tu.date_diff(self._taxis, d, d1, 'hours') < self.mult / 2:
      return self._taxis.date_as_val(d1)
    else:
      return self._taxis.date_as_val(d)
  # }}}
# }}}

#########################################
## Class MinuteTickGen
class MinuteTickGen(TickGenerator):
# {{{
  ''' MinuteTickGen() - locator helper object which ticks at intervals
  of a multiple of minutes.'''
  fmt = '$H:$M'
  ofmt = '$d $b $Y'

  def nticks(self, vmin, vmax, mult=None):
  # {{{
    if mult is None: mult = self.mult
    d1 = self.get_tick_prior(vmin)
    d2 = self.get_tick_prior(vmax)

    return tu.date_diff(self._taxis, d1, d2, units='minutes') / mult
  # }}}

  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the date of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the date of the
    previous tick.'''

    def unpack(dt): return dt.get('year', 1), dt.get('month', 1), dt.get('day', 1), dt.get('hour', 0), dt.get('minute', 0)
    def pack(yr, mn, dy, hr, mi): return {'year':yr, 'month':mn, 'day':dy, 'hour':hr, 'minute':mi}

    dt = self._taxis.val_as_date(val, allfields=True)
    yr, mn, dy, hr, mi = unpack(dt)

    if val > self._taxis.date_as_val(pack(yr, mn, dy, hr, mi)): 
      yr, mn, dy, hr, mi = unpack(tu.wrapdate(self._taxis, pack(yr, mn, dy, hr, mi+1), allfields=True))

    # Find first hour on given multiple prior to the given hour
    from numpy import floor_divide
    mi = floor_divide(mi - 1, self.mult) * self.mult

    # If we've wrapped, decrement the year
    d = tu.wrapdate(self._taxis, pack(yr, mn, dy, hr, mi), allfields=True)
    d1 = tu.wrapdate(self._taxis, pack(yr, mn, dy, hr+1, 0), allfields=True)

    if tu.date_diff(self._taxis, d, d1, 'minutes') < self.mult / 2:
      return d1
    else:
      return d
  # }}}

  def next_tick(self, val):
  # {{{
    d = self._taxis.val_as_date(val, allfields=True)
    d1 = d.copy()
    d['minute'] += self.mult
    d1['hour'] += 1; d1['minute'] = 0
    if tu.date_diff(self._taxis, d, d1, 'minutes') < self.mult / 2:
      return self._taxis.date_as_val(d1)
    else:
      return self._taxis.date_as_val(d)
  # }}}
# }}}

#########################################
## Class SecondTickGen
class SecondTickGen(TickGenerator):
# {{{
  ''' SecondTickGen() - locator helper object which ticks at intervals
  of a multiple of seconds.'''
  fmt = '$H:$M:$S'
  ofmt = '$d $b $Y'

  def nticks(self, vmin, vmax, mult=None):
  # {{{
    if mult is None: mult = self.mult
    d1 = self.get_tick_prior(vmin)
    d2 = self.get_tick_prior(vmax)

    nt = tu.date_diff(self._taxis, d1, d2, units='seconds') / mult
    return nt
  # }}}

  def get_tick_prior(self, val):
  # {{{
    ''' get_tick_prior(val) - returns the date of the first tick prior to the
    date represented by val. If a tick lies on val, it returns the date of the
    previous tick.'''

    def unpack(dt): return dt.get('year', 1), dt.get('month', 1), dt.get('day', 1), dt.get('hour', 0), dt.get('minute', 0), dt.get('second', 0)
    def pack(yr, mn, dy, hr, mi, sc): return {'year':yr, 'month':mn, 'day':dy, 'hour':hr, 'minute':mi, 'second':sc}

    dt = self._taxis.val_as_date(val, allfields=True)
    yr, mn, dy, hr, mi, sc = unpack(dt)

    if val > self._taxis.date_as_val(pack(yr, mn, dy, hr, mi, sc)): 
      yr, mn, dy, hr, mi, sc = unpack(tu.wrapdate(self._taxis, pack(yr, mn, dy, hr, mi, sc+1), allfields=True))

    # Find first hour on given multiple prior to the given hour
    from numpy import floor_divide
    sc = floor_divide(sc - 1, self.mult) * self.mult

    # If we've wrapped, decrement the year
    d = tu.wrapdate(self._taxis, pack(yr, mn, dy, hr, mi, sc), allfields=True)
    d1 = tu.wrapdate(self._taxis, pack(yr, mn, dy, hr, mi+1, 0), allfields=True)

    if tu.date_diff(self._taxis, d, d1, 'seconds') < self.mult / 2:
      return d1
    else:
      return d
  # }}}

  def next_tick(self, val):
  # {{{
    d = self._taxis.val_as_date(val, allfields=True)
    d1 = d.copy()
    d['second'] += self.mult
    d1['minute'] += 1; d1['second'] = 0
    if tu.date_diff(self._taxis, d, d1, 'seconds') < self.mult / 2:
      return self._taxis.date_as_val(d1)
    else:
      return self._taxis.date_as_val(d)
  # }}}
# }}}

from pylab import Locator
from pylab import Formatter

#########################################
## TimeFormatter - matplotlib formatter object for the time axis;
## most formatting work is done by the time axis itself
class TimeFormatter(Formatter):
# {{{
  def __init__(self, taxis, fmt=None, ofsfmt=None, auto=True):
    if fmt is None: 
      if not auto: 
        fmt = taxis.plotatts.get('plotfmt', None)
        if fmt is None: fmt = taxis.formatstr
    else: # If fmt is set explicitly, turn off auto-formatting
      auto = False
      if ofsfmt is None: ofsfmt = ''

    if ofsfmt is None and not auto:
      ofsfmt = taxis.plotatts.get('plotofsfmt', None)

    self.taxis = taxis
    self.fmt = fmt
    self.ofsfmt = ofsfmt
    self.auto = auto

    self.offset = ''

  def __call__(self, x, pos=0):
    return self.taxis.formatvalue(x, fmt=self.fmt)

  def set_locs(self, locs):
    if len(locs) > 0:
      if self.auto:
        delt = max(locs) - min(locs)
        delt = delt * self.taxis.unitfactor[self.taxis.units] / ((len(locs)-1) * 86400.)
        self.fmt, self.ofsfmt = self.autoformat_range(delt)
      self.offset = self.taxis.formatvalue(locs[0], self.ofsfmt)
    else:
      self.offset = ''

  def autoformat_range(self, delt):
    ''' Returns a format string approriate for a date/time range of size delt days. '''
    import numpy as np

    for d, f, o in self.taxis.autofmts:
      if np.abs(delt) * 1.1 >= d: return f, o
    
    return f, o

  def get_offset(self):
    return self.offset
# }}}

#########################################
## AutoCalendarLocator - matplotlib locator object for the time axis
class AutoCalendarLocator(Locator):
# {{{
  def __init__(self, taxis, fmt=None, ofmt=None, nticks=5):
  # {{{
    self._taxis = taxis
    self._nticks = nticks
    self._itkgen = 0
    self.fmt = fmt    # Overrides tick generator defaults
    self.ofmt = ofmt
  # }}}

  def __call__(self):
  # {{{
    # Return tick locations
    try:
      vmin, vmax = self.axis.get_view_interval()
    except AttributeError:
      vmin, vmax = self.viewInterval.get_bounds()

    # Todo:  Auto selection of the multiple up to some factor
    # - stickiness of tick generators

    # Sticky, automult algorithm?
    # start with previous ticker: get best multiplier
    #  if nticks is greater than target, get best multiplier
    #    of prev. ticker. If this brackets the target, pick the better
    #    of the two. If not, keep stepping previous
    #  if nticks is less than target, do the same, but with the next ticker

    itk = self._itkgen
    ntks = len(self._taxis.tick_generators)
    tk = self._taxis.tick_generators[itk]
    nt, im = tk.best_mult(vmin, vmax, self._nticks)

    while True:
      # If the tick generator is at an external point, continue search
      if nt < self._nticks and im == len(tk.mults) - 1 and itk < ntks - 1:
        tk2 = self._taxis.tick_generators[itk + 1]
        nt2, im2 = tk2.best_mult(vmin, vmax, self._nticks)

        # If the new generator is worse, use current guess
        if im2 == 0 and self._nticks - nt <= nt2 - self._nticks: break

        tk = tk2; 
        im = im2; 
        nt = nt2 
        itk = itk + 1
      elif nt > self._nticks and im == 0 and itk > 0:
        tk2 = self._taxis.tick_generators[itk - 1]
        nt2, im2 = tk2.best_mult(vmin, vmax, self._nticks)

        # If the new generator is worse, use current guess
        if im2 == len(tk2.mults) - 1 and abs(nt - self._nticks) <= abs(self._nticks - nt2): break

        tk = tk2; 
        im = im2; 
        nt = nt2 
        itk = itk - 1
      else: 
        break
   
    self._itkgen = itk
    tcks = [t for t in tk.ticks(vmin, vmax)]
    if self.fmt is None: self._taxis.plotatts['plotfmt'] = tk.fmt
    else: self._taxis.plotatts['plotfmt'] = self.fmt
    if self.ofmt is None: self._taxis.plotatts['plotofsfmt'] = tk.ofmt
    else: self._taxis.plotatts['plotofsfmt'] = self.ofmt
    return tcks
  # }}}
# }}}
    
#########################################
## YearLocator - matplotlib locator object for the time axis
class YearLocator(Locator):
# {{{
  def __init__(self, taxis, mult=1):
  # {{{
    self._taxis = taxis
    self._mult = mult
    self.tkgen = YearTickGen(taxis, [mult])
  # }}}

  def __call__(self):
  # {{{
    # Return tick locations
    try:
      vmin, vmax = self.axis.get_view_interval()
    except AttributeError:
      vmin, vmax = self.viewInterval.get_bounds()

    tcks = [t for t in self.tkgen.ticks(vmin, vmax)]
    return tcks
  # }}}
# }}}
    
#########################################
## MonthLocator - matplotlib locator object for the time axis
class MonthLocator(Locator):
# {{{
  def __init__(self, taxis, mult=1):
  # {{{
    self._taxis = taxis
    self._mult = mult
    self.tkgen = MonthTickGen(taxis, [mult])
  # }}}

  def __call__(self):
  # {{{
    # Return tick locations
    try:
      vmin, vmax = self.axis.get_view_interval()
    except AttributeError:
      vmin, vmax = self.viewInterval.get_bounds()

    tcks = [t for t in self.tkgen.ticks(vmin, vmax)]
    return tcks
  # }}}
# }}}
    
