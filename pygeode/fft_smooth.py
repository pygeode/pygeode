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
    # Get source data
    aview = view.modify_slice(saxis, insl)
    src = aview.get(self.var, pbar=pbar)
    maxharm= self.maxharm
    smsl = [ slice(maxharm,None) if i == saxis else slice(None) for i in range(self.naxes)]
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
  return FFTSmoothVar(var, saxis=var.whichaxis(saxis), maxharm=maxharm)



 
