# fancy schmancy progress bar, so you know when it's a good time to go grab a coffee

# Uses the 'progressbar' module, which should be available in your distribution's repository
# (probably as 'python-progressbar')

# Extends the functionality to allow for a sub-progress value.
# I.e., if we are going to call multiple calculations, and each one is responsible
# for a small segment of the total time, we can pass the corresponding 'subset' of the
# progress bar to the relevant subroutine, and the subroutine can internally reference the
# progress as 0 to 100% without needing to know what other progress exists outside that part.

# Change criteria for updating the bar to: once a second
# Also, don't show a bar for tasks of a short duration (10 seconds)
try:
  from progressbar import (
          Bar, 
          ETA,
          ProgressBar, 
          Percentage
  )
  from progressbar.bar import StdRedirectMixin, ResizableMixin, ProgressBarBase
  import progressbar.base as base
  import time
  import timeit
  from datetime import datetime

  _NOSHOWTIME = 10.
  class PygProgressBar(ProgressBar):

    if hasattr(ProgressBar, '__slots__'):
       __slots__ = ProgressBar.__slots__ + ('first_update_time', 'prev_update_time', 'message', 'printed_message')
    
    def __init__(self, **kwargs):
      ProgressBar.__init__(self, **kwargs)
      self.first_update_time = None
      self.prev_update_time = None
      self.message = None
      self.printed_message = False
      self.need_update_bool = False
      self.seconds_elapsed = 0.0

    def _needs_update(self):
    
      'Returns whether the ProgressBar should redraw the line.'
      self.seconds_elapsed = time.time() - time.mktime(self.start_time.timetuple())

      # Do no show progressbar until after _NOSHOWTIME
      if self.seconds_elapsed >_NOSHOWTIME:
        if not self.printed_message:
          if self.message is not None: print('\n'+self.message)
          self.printed_message = True
        if self.poll_interval:
          delta = timeit.default_timer() - self._last_update_timer
          poll_status = delta > self.poll_interval.total_seconds()
        else:
          poll_status = False

        # Do not update if value increment is not large enough to
        # add more bars to progressbar (according to current
        # terminal width)
        try:
          divisor = self.max_value / self.term_width  # float division
          if self.value // divisor == self.previous_value // divisor:
            return poll_status or self.end_time
        except Exception:
          # ignore any division errors
          pass

        self.need_update_bool = self.value > self.next_update or poll_status or self.end_time
      else:
        self.need_update_bool = False
      
      return self.need_update_bool  

    def start(self, init=True): 
      '''
      Start the progress bar but do not displaying until _needs_update() returns
      True during update.
      '''
      if init:
          self.init()

      # Prevent multiple starts
      if self.start_time is not None:  # pragma: no cover
          return self

      StdRedirectMixin.start(self, max_value=self.max_value)
      ResizableMixin.start(self, max_value=self.max_value)
      ProgressBarBase.start(self, max_value=self.max_value)
      # Constructing the default widgets is only done when we know max_value
      if self.widgets is None:
          self.widgets = self.default_widgets()

      for widget in self.widgets:
          interval = getattr(widget, 'INTERVAL', None)
          if interval is not None:
              self.poll_interval = min(
                  self.poll_interval or interval,
                  interval,
              )

      self.num_intervals = max(100, self.term_width)
      self.next_update = 0

      if self.max_value is not base.UnknownLength and self.max_value < 0:
          raise ValueError('Value out of range')

      self.start_time = self.last_update_time = datetime.now()
      self._last_update_timer = timeit.default_timer()
      self.update(self.min_value, force=False)
      return self


    def finished(self): # Overloaded to fix unneccesary newlines when not displayed
      if self.seconds_elapsed > _NOSHOWTIME:
        self.finish()
      else:
        return

except ImportError:
  from warnings import warn
  warn ("progressbar module not found; progress bars will not be displayed.")
  PygProgressBar = None

class PBar:
  def __init__ (self, pbar=None, lower=0, upper=100, message=None):
    import time
    self.lower = lower
    self.upper = upper

    if pbar is not None:
      self.pbar = pbar
      return

    if PygProgressBar is not None:
      pbar = PygProgressBar(widgets=[
                                Percentage(), 
                                Bar(), 
                                ETA()], max_value=100.0)
      self.pbar = pbar.start()
      self.pbar.message = message  # staple a title message to the progress bar
                                   # (to be used in _need_update)
    else:
      self.pbar = None

  def update(self, x):
    import time
    lower = self.lower
    upper = self.upper
    dx = upper - lower
    y = lower + dx * x/100.
    y = max(0,y)
    y = min(y,100)
    if self.pbar is not None:
      if y == 100: self.pbar.finished()
      else: self.pbar.update(y)

  def subset(self, L, U):
    lower = self.lower
    upper = self.upper
    dx = upper - lower
    # adjust lower/upper to match the absolute progress
    L = lower + dx * L/100.
    U = lower + dx * U/100.
    return PBar(pbar=self.pbar, lower=L, upper=U)

  # Similar to a subset, but take as arguments the number of divisions, and the division number
  # i.e., part i of n
  def part(self, i,n):  return self.subset(100./n*i, 100./n*(i+1))

class FakePBar:
  def update (self, x): pass
  def subset (self, L, U): return self
  def part (self, i, n): return self

# Wrap some possible non-progressbar things (message string, True/False/None) into a progress bar
def as_pbar (x):
  if isinstance(x,PBar): return x
  if x is False or x is None: return FakePBar()
  if x is True: return PBar()
  if isinstance(x,str): return PBar(message=x)
  raise Exception ("can't convert %s to a PBar"%type(x))
