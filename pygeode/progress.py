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

from pygeode import _config

try:
  import progressbar as pb
  from datetime import datetime

  class PygProgressBar(pb.ProgressBar):
    ''' Version of progressbar.ProgressBar that does not display until a minimum 
    time has elapsed. '''
    def __init__(self, **kwargs):
      self._no_show_time = _config.getfloat('ProgressBar', 'no_show_time')
      self.message = kwargs.pop('message', '')
      if self.message is not '': self.message = '{:<30}'.format(self.message)
      self.init_kwargs = kwargs
      pb.bar.ProgressBarMixinBase.__init__(self)

    def default_widgets(self):
      return [pb.FormatCustomText(format = self.message),
              pb.Percentage(**self.widget_kwargs), 
              pb.Bar(**self.widget_kwargs), 
              pb.AdaptiveETA(**self.widget_kwargs)]

    def start(self, max_value=None, init=True):
      # Avoid initializing progress bar until _no_show_time has elapsed.
      self.init()
      self.max_value = max_value
      self.start_time = datetime.now()
      return self

    def update(self, value=None, force=False, **kwargs):
      if self.start_time is None:
        self.start()

      if self.end_time is None:
        et = datetime.now()
      else:
        et = self.end_time

      total_seconds_elapsed = pb.utils.timedelta_to_seconds(et - self.start_time)
      if total_seconds_elapsed < self._no_show_time: 
        return
      else:
        if self.last_update_time is None:
          # Create underlying progress bar
          st = self.start_time 
          pb.ProgressBar.__init__(self, **self.init_kwargs)
          pb.ProgressBar.start(self, self.max_value, init=True)
          self.start_time = st

        return pb.ProgressBar.update(self, value, force, **kwargs)

    def finish(self, end='\n'):
      if self._finished: return

      if self.start_time is None:
        total_seconds_elapsed = 0.
      else:
        if self.end_time is None:
          self.end_time = datetime.now()
        total_seconds_elapsed = pb.utils.timedelta_to_seconds(self.end_time - self.start_time)

      if total_seconds_elapsed >= self._no_show_time: 
        pb.ProgressBar.finish(self, end = end)

      self._finished = True

except ImportError as e:
  from warnings import warn
  warn ("progressbar module not found (%s). Progress bars will not be displayed." % repr(e))
  PygProgressBar = None

class PBar:
  def __init__ (self, pbar=None, lower=0, upper=100, message=''):
    self.lower = lower
    self.upper = upper
    self.parent = False

    if pbar is not None:
      self.pbar = pbar
      return

    if PygProgressBar is not None:
      self.parent = True
      pbar = PygProgressBar(message = message, max_value=100.0, \
                            poll_interval = _config.getfloat('ProgressBar', 'poll_interval'))
      self.pbar = pbar.start()
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
      self.pbar.update(y)
      if y >= 100 and self.parent:
        self.pbar.finish()

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
