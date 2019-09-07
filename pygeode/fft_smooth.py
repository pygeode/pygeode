# fft_smooth.py - implementation of SmoothVar

from pygeode.var import Var

class FFTSmoothVar (Var):
  '''Smoothing variable.'''

  def __init__(self, var, saxis, maxharm):
  # {{{
    ''' __init__()'''
    # Construct new variable
    self.saxis = saxis
    self.var = var
    self.maxharm = maxharm
    Var.__init__(self, var.axes, var.dtype, name=var.name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy
    saxis = self.saxis
    # Get bounds of slice on smoothing axis
    ind = view.integer_indices[saxis]
    st, sp = numpy.min(ind), numpy.max(ind)
    # input is the whole range
    insl = slice(0, self.shape[saxis],1)
    # output is the required slice
    outsl = [ ind if i == saxis else slice(None) for i in range(self.naxes)]
    outsl = tuple(outsl)
    # Get source data
    aview = view.modify_slice(saxis, insl)
    src = aview.get(self.var, pbar=pbar)
    maxharm= self.maxharm
    smsl = [slice(maxharm,None) if i == saxis else slice(None) for i in range(self.naxes)]
    smsl = tuple(smsl)
    # calculate harmonics and output required data
    from numpy import fft
    if 'complex' in self.dtype.name:
      ct=fft.fft(src,self.shape[saxis],saxis)
      smsl=[ slice(maxharm,-maxharm+1) if i == saxis else slice(None) for i in range(self.naxes)]
      ct[smsl]=0
      st = fft.ifft(ct, self.shape[saxis], saxis)
    else:
      ct=fft.rfft(src,self.shape[saxis],saxis)
      ct[smsl]=0
      st = fft.irfft(ct, self.shape[saxis], saxis)

    return st[outsl].astype(self.dtype)
  # }}}

def fft_smooth(var, saxis, maxharm):
  ''' Smooths this variable along ``saxis`` by retaining leading Fourier
  components.

  Parameters
  ----------
  saxis : any axis identifier (string, :class:`Axis`, or int)
    Axis over which the smoothing should be performed

  maxharm : int
    Maximum harmonic to retain.

  Returns
  -------
  out : :class:`Var`
    Smoothed variable.

  Notes
  -----
  The variable data is Fourier transformed using :func:`np.fft.rfft` (if real) or
  :func:`np.fft.fft` (if complex). The coefficients for harmonics equal to and
  greater than ``maxharm`` are set to zero, then an inverse transform is applied.

  Examples
  --------
  >>> import pygeode as pyg, numpy as np
  >>> tm = pyg.modeltime365n('1 Jan 2000', 365)
  >>> v = pyg.cos(3 * 2 * np.pi * tm / 365.)
  >>> np.std(v[:])
  0.7071067811865476
  >>> np.std(v.fft_smooth('time', 3)[:]) # This retains only the annual and semi-annual cycle
  1.392158115250162e-16
  >>> np.std(v.fft_smooth('time', 4)[:]) # This retains up to the third harmonic
  0.7071067811865472
  '''
  return FFTSmoothVar(var, saxis=var.whichaxis(saxis), maxharm=maxharm)
