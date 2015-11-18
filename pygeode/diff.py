# diff.py - implementation of ForwardDifferenceVar, and the 'diff' method.

from pygeode.var import Var
class ForwardDifferenceVar (Var):
  '''Forward difference variable.'''

  def __init__ (self, var, axis, n):
  # {{{
    ''' __init__()'''

    from pygeode.var import Var

    df = 'right'  # Hard-coded to match numpy behaviour.
                  # May be extended in the future?

    self.daxis = daxis = var.whichaxis(axis)
    assert var.shape[daxis] > n, "need at least %d value(s) along difference axis"%n

    self.n = n
    self.df = df
    self.var = var

    # Construct new variable

    if var.name != '':
      name = 'd' + var.name
    else:
      name = 'd(UnknownVar)'

    axes = list(var.axes)
    if df == 'left':
      axes[daxis] = axes[daxis].slice[n:]
    else:
      axes[daxis] = axes[daxis].slice[:-n]

    Var.__init__(self, axes, var.dtype, name=name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    from functools import reduce

    daxis = self.daxis
    n = self.n

    # Get integer indices along the difference axis
    left = view.integer_indices[daxis]

    # All the points we need to request (unique occurrences only)
    allpoints = reduce(np.union1d, [left+i for i in range(n+1)])

    # Set a view on the original variable, to request these points
    allview = view.replace_axis(daxis, self.var.axes[daxis], sl=allpoints)

    # Get the data values for these points
    allvalues = allview.get(self.var, pbar=pbar)

    # Compute the difference
    diff = np.diff(allvalues, axis=daxis, n=n)

    # Define a map back to our points
    getleft = np.searchsorted(allpoints,left)
    # Make this 1D map into the right shape for the view (if multi-dimensional)
    getleft = [slice(None)]*daxis + [getleft] + [slice(None)]*(self.naxes-daxis-1)

    # Finally, map the data to our points, and return.
    # Hopefully the code above works for all cases (including non-contiguous
    # views).
    # Otherwise - good luck, asshole!
    return diff[getleft]
  # }}}

def diff(var, axis=0, n=1):
  '''Computes the forward difference along the given axis.
     Mimics the same behaviour of the numpy 'diff' function.'''
  return ForwardDifferenceVar(var, axis, n)

