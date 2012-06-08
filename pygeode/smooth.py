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
    ktypes = {'hann': np.hanning, 'flat': np.ones, 'bartlett': np.bartlett, 'hamm': np.hamming}

    # Construct new variable
    self.saxis = saxis
    self.var = var
    if kernel in ktypes.keys():
      self.kernel = ktypes[kernel](klen)
    else:
      self.kernel = kernel

    assert klen <= var.shape[saxis], 'Kernel must be shorter than dimension being smoothed.'
    self.klen = klen

    # Normalize and reshape kernel
    self.kernel /= np.sum(self.kernel)
    self.kernel.shape = [klen if i == saxis else 1 for i in range(var.naxes)]

    Var.__init__(self, var.axes, var.dtype, name=var.name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    from scipy import signal as sg
    import numpy as np

    saxis = self.saxis

    # Get bounds of slice on smoothing axis
    ind = view.integer_indices[saxis]
    st, sp = np.min(ind), np.max(ind)

    # Extend view along smoothing axis; use data past slice if present, otherwise mirror data
    loffs = self.klen / 2
    roffs = self.klen - loffs
    src_shape = view.shape[:saxis] + (sp - st + self.klen,) + view.shape[saxis+1:]
    src = np.empty(src_shape, self.var.dtype)
    
    # Construct slices for mirroring pre-convolved data
    ssli = [slice(None) for i in range(self.naxes)]
    sslo = [slice(None) for i in range(self.naxes)]
    mleft_len = max(loffs - st, 0)
    mright_len = max(sp + roffs + 1, self.shape[saxis]) - self.shape[saxis]

    # Construct slice into source variable
    insl = slice(max(st - loffs, 0), min(sp + roffs, self.shape[saxis]), 1)
    aview = view.modify_slice(saxis, insl)

    # Construct slice into convolved output
    outsl = [ind - (st - loffs) if i == saxis else slice(None) for i in range(self.naxes)]

    # Get source data
    sslo[saxis] = slice(mleft_len, src_shape[saxis] - mright_len + 1)
    src[sslo] = aview.get(self.var, pbar=pbar)

    # Mirror boundaries, if necessary
    if mleft_len > 0:
      ssli[saxis] = slice(2*mleft_len, mleft_len,-1)
      sslo[saxis] = slice(0, mleft_len)
      src[sslo] = src[ssli]

    # Mirror boundaries, if necessary
    if mright_len > 0:
      ssli[saxis] = slice(-mright_len,-2*mright_len,-1)
      sslo[saxis] = slice(-mright_len, None)
      src[sslo] = src[ssli]

    # Return convolved array (forcing the dtype seems to be required)
    try:
       # Older versions of scipy require a flag to define the correct convolution behaviour 
       return sg.convolve(src, self.kernel, 'same', old_behavior=False)[outsl].astype(self.dtype)
    except TypeError:
       return sg.convolve(src, self.kernel, 'same')[outsl].astype(self.dtype)
  # }}}

def smooth(var, saxis, klen=15, kernel='hann'):
  if klen <= 2:
    return var
  return SmoothVar(var, saxis=var.whichaxis(saxis), kernel=kernel, klen=klen)
