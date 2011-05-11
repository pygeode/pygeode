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
def _need_update(self):
  import time
  # Force this to be updated independant of the need_update decision
  # (this is normally computed in update(), but is skipped if we don't need to display anything)
  if not self.start_time:
      self.start_time = time.time()
  self.seconds_elapsed = time.time() - self.start_time

  current_time = time.time()

  #TODO: used seconds_elapsed instead of whatever we're using here.
  # Also, recycle the logic from the wrapped need_update? (update only on every percentage)?
  if not hasattr(self,'first_update_time'):
    self.first_update_time = current_time
    self.prev_update_time = current_time
    return False
  if current_time - self.first_update_time < 10: return False
  if current_time - self.prev_update_time > 1 or self.currval == self.maxval:
    self.prev_update_time = current_time
    if self.message is not None and not hasattr(self,'printed_message'):
      print self.message
      self.printed_message = True
    return True
  return False
###  return int(self.percentage()) != int(self.prev_percentage)


class PBar:
  def __init__ (self, pbar=None, lower=0, upper=100, message=None):
    self.lower = lower
    self.upper = upper

    if pbar is not None:
      self.pbar = pbar
      return

    try:
      from progressbar import ProgressBar, Percentage, Bar, ETA
      ProgressBar._need_update = _need_update
      pbar = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', ETA()])
      self.pbar = pbar.start()
      self.pbar.message = message  # staple a title message to the progress bar
                                   # (to be used in _need_update)
    except ImportError:
      from warnings import warn
      warn ("progressbar module not found; no progress will be displayed.")
      self.pbar = None

  def update(self, x):
    lower = self.lower
    upper = self.upper
    dx = upper - lower
    y = lower + dx * x/100.
    y = max(0,y)
    y = min(y,100)
    if self.pbar is not None:
      self.pbar.update(y)
      if y == 100: self.pbar.finish()

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
