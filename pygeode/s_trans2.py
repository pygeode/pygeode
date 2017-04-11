

from pygeode.var import Var
class Strans(Var):
    
  def __init__(self,x,units='days',minfreq=None,maxfreq=None,nsamples=100):
    from pygeode.var import Var, copy_meta
    from pygeode.timeaxis import Time
    from pygeode.axis import Freq
    import numpy as np
    from math import log, exp, ceil
    assert x.hasaxis(Time), "no time axis found!"
    assert not x.hasaxis(Freq), "%s already has a frequency axis?"%repr(x)
    self.x = x
    self.taxis = taxis = x.getaxis(Time)
    self.T = T = len(taxis)

    self.dt = dt = taxis.delta(units)

    if minfreq is None: minfreq = 1./(T*dt)
    if maxfreq is None: maxfreq = 1./(2*dt)

    F1 = int(round(minfreq*T*dt))
    F2 = int(round(maxfreq*T*dt))
    assert nsamples > 1
    stride = (F2-F1) / (nsamples-1)
    if stride == 0: stride = 1
    F = np.arange(F2, F1-1, -stride)[::-1]

    if 0 in F:
      from pygeode.progress import PBar
      print 'Calculating mean'
      # constant s-transform value for n = 0
      self.const = x.mean(Time).get(pbar=True)
    
    f = F / (T * dt)
    self.faxis = faxis = Freq(f, units)

    axes = list(x.axes)
    ti = x.whichaxis(Time)
    # Move time axis to the end, and include frequency axis
    axes = axes[:ti] + axes[ti+1:] + [faxis] + [taxis]
    Var.__init__(self, axes, dtype=complex)
    copy_meta(x, self)
    self.name = 'Strans('+(x.name or '??')+')'


  def getview (self, view, pbar):
    from scipy.fftpack import fft,ifft
    import numpy as np
    from math import exp, pi
    from pygeode.axis import Freq
    from pygeode.timeaxis import Time
    out = np.empty(view.shape, self.dtype)

    timeslice = view.slices[-1]

    W = np.empty(self.T, self.dtype)

    T = self.T
    ti = self.x.whichaxis(Time)

    # Use entire time axis, and loop over memory-friendly chunks of data
    loop = list(view.modify_slice(Time,slice(None)).loop_mem())
    # Can't subset over time, because we need the whole timeseries to compute FFT
    assert len(loop[0].integer_indices[-1]) == T, "not enough memory"
    for iloop,v in enumerate(loop):
      subpbar = pbar.part(iloop,len(loop))

      X = v.get(self.x, pbar=subpbar)
      X = fft(X, axis=-1)
      X = np.concatenate((X,X), axis=-1)

      F = v.get(self.faxis) * self.dt * T
      F = np.asarray(np.round(F.squeeze()),int)

      #TODO
      outsl = list(v.slices) # what part of the output we're looking at
      # Loop over each frequency
      for i in np.arange(len(F)):
        outsl[-2] = [i] # this particular frequency
        if F[i] == 0:
          out[outsl] = self.const
          continue
        # create Fourier Transform of Gaussian Window
        M = np.arange(T)
        M[T/2:] -= T  # T/2 or T/2 + 1??
        W = np.exp(-2 * (pi*M/F[i])**2 )

        # compute S-transform
        sample = X[...,F[i]:F[i]+T]
        out[outsl] = ifft(sample * W, axis=-1)[...,timeslice]

    return out



del Var
