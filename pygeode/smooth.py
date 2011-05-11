# smooth.py - implementation of SmoothVar

from pygeode.var import Var

class SmoothVar (Var):
  '''Smoothing variable. Convolves source variable along 
      given axis with the specified smoothing kernel.'''

  def __init__ (self, var, saxis, kernel='hann', klen=None):
  # {{{
    ''' __init__()'''
    import numpy as np

    # Moved this here temporarily, otherwise pygeode won't start on sparc01 (no scipy available)
    from scipy import signal as sg
    ktypes = {'hann': sg.hann, 'boxcar': sg.boxcar, 'triang': sg.triang}

    # Construct new variable
    self.saxis = saxis
    self.var = var
    if kernel in ktypes.keys():
      self.kernel = ktypes[kernel](klen)
    else:
      self.kernel = kernel

    assert klen <= var.shape[saxis], 'Kernel must be shorter than dimension being smoothed.'
    self.klen = klen
    self.kshape = [klen if i == saxis else 1 for i in range(var.naxes)]

    # Normalize kernel
    self.kernel /= np.sum(self.kernel)

    Var.__init__(self, var.axes, var.dtype, name=var.name, atts=var.atts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np

    # Moved this here temporarily, otherwise pygeode won't start on sparc01 (no scipy available)
    from scipy import signal as sg

    saxis = self.saxis

    # Get bounds of slice on smoothing axis
    ind = view.integer_indices[saxis]
    st, sp = np.min(ind), np.max(ind)

    # Extend view along smoothing axis if possible
    offs = self.klen / 2
    insl = slice(max(st - offs, 0), min(sp + offs, self.shape[saxis]), 1)

    # Construct slice into convolved output
    outsl = [ind - insl.start if i == saxis else slice(None) for i in range(self.naxes)]

    # Get source data
    aview = view.modify_slice(saxis, insl)
    src = aview.get(self.var, pbar=pbar)

    # Return convolved array (forcing the dtype seems to be required)
    return sg.convolve(src, self.kernel.reshape(self.kshape), 'same')[outsl].astype(self.dtype)
  # }}}

def smooth(var, saxis, klen=15, kernel='hann'):
  if klen <= 1:
    return var
  return SmoothVar(var, saxis=var.whichaxis(saxis), kernel=kernel, klen=klen)
