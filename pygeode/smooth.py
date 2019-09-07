# smooth.py - implementation of SmoothVar

from pygeode.var import Var

class SmoothVar (Var):
  '''Smoothing variable. Convolves source variable along 
      given axis with the specified smoothing kernel.'''

  def __init__ (self, var, saxis, kernel, fft):
  # {{{
    ''' __init__()'''
    import numpy as np

    assert len(kernel) <= var.shape[saxis], 'Kernel must not be longer than dimension being smoothed.'

    # Construct new variable
    self.saxis = saxis
    self.var = var
    self.kernel = kernel
    self.fft = fft
    self.klen = len(kernel)

    # Normalize and reshape kernel
    self.kernel /= np.sum(self.kernel)
    self.kernel.shape = [self.klen if i == saxis else 1 for i in range(var.naxes)]

    # Determine which convolution function to use
    from scipy import signal as sg
    tdata = np.ones(len(kernel), 'd')
    if self.fft:
      try:
        sg.fftconvolve(tdata, kernel, 'same', old_behaviour=False)
        self._convolve = lambda x, y, z: sg.fftconvolve(x, y, z, old_behaviour=False)
      except TypeError:
        self._convolve = sg.fftconvolve
    else:
      try:
        sg.convolve(tdata, kernel, 'same', old_behaviour=False)
        self._convolve = lambda x, y, z: sg.convolve(x, y, z, old_behaviour=False)
      except TypeError:
        self._convolve = sg.convolve

    Var.__init__(self, var.axes, var.dtype, name=var.name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    from pygeode.tools import loopover

    saxis = self.saxis

    # Get bounds of slice on smoothing axis
    ind = view.integer_indices[saxis]
    st, sp = np.min(ind), np.max(ind)

    # Extend view along smoothing axis; use data past slice if present, otherwise mirror data
    loffs = self.klen // 2
    roffs = self.klen - loffs
    
    # Construct slices for mirroring pre-convolved data
    ssli = [slice(None) for i in range(self.naxes)]
    sslo = [slice(None) for i in range(self.naxes)]
    mleft_len = max(loffs - st, 0)
    mright_len = max(sp + roffs, self.shape[saxis]) - self.shape[saxis]

    # Construct slice into source variable
    insl = slice(max(st - loffs, 0), min(sp + roffs, self.shape[saxis]), 1)
    aview = view.modify_slice(saxis, insl)

    # Construct slice into convolved output
    cnvsl = [ind - (st - loffs) if i == saxis else slice(None) for i in range(self.naxes)]

    # Extended source data if needed
    src_saxislen = sp - st + self.klen

    out = np.zeros(view.shape, self.dtype)

    for outsl, (indata,) in loopover(self.var, aview, preserve=[saxis], pbar=pbar):
      src_shape = indata.shape[:saxis] + (src_saxislen,) + indata.shape[saxis+1:]
      src = np.zeros(src_shape, self.var.dtype)
      sslo[saxis] = slice(mleft_len, src_saxislen - mright_len)
      src[tuple(sslo)] = indata

      # Mirror boundaries, if necessary
      if mleft_len > 0:
        ssli[saxis] = slice(2*mleft_len-1, mleft_len-1,-1)
        sslo[saxis] = slice(0, mleft_len)
        src[tuple(sslo)] = src[tuple(ssli)]

      # Mirror boundaries, if necessary
      if mright_len > 0:
        ssli[saxis] = slice(-mright_len-1,-2*mright_len-1,-1)
        sslo[saxis] = slice(-mright_len, None)
        src[tuple(sslo)] = src[tuple(ssli)]

      out[outsl] = self._convolve(src, self.kernel, 'same')[tuple(cnvsl)]

    return out
  # }}}

def smooth(var, saxis, kernel=15, fft=True):
  ''' Smooths this variable along ``saxis`` by convolving it with an averaging
      kernel. The returned variable is defined on the same axes.

      Parameters
      ----------
      saxis : any axis identifier (string, :class:`Axis`, or int)
        Axis over which the smoothing should be performed

      kernel : sequence or int (optional)
        Averaging kernel with which to convolve this variable. Does not need to
        be normalized.  If an integer is provided, a Hanning window is used of
        length ``kernel`` (:func:`numpy.hanning`)

      fft : boolean (optional, True by default)
        If True, :func:`scipy.signal.fftconvolve` is used to compute the convolution.
        Otherwise, :func:`scipy.signal.convolve` is used. In many cases the former
        is more efficient.

      Returns
      -------
      out : :class:`Var`
        Smoothed variable.

      Notes
      -----
      When the convolution is performed, the source data is extended on either
      end of the axis being smoothed by reflecting the data by enough to ensure
      the returned variable is defined on the same grid as the source variable.
      That is, if the original data is t1, t2, .., tN, and the kernel is L
      items long, the convolved sequence is tj, t_j-1, t1, t1, t2, .. tN-1, tN,
      tN, tN-1, .. tN-l, where j = floor(L/2) and l = L - j - 1.

      Examples
      --------
  '''
  import numpy as np
  if hasattr(kernel, '__len__'):
    kernel = np.array(kernel, dtype='d')
  else:
    kernel = np.hanning(kernel)

  if len(kernel) <= 2:
    return var

  return SmoothVar(var, var.whichaxis(saxis), kernel, fft=fft)
