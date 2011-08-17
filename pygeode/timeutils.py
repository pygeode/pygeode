
from pygeode.var import Var
from pygeode.timeaxis import Time, Yearless

class Lag(Yearless): 
# {{{
  name = 'lag'
  @classmethod
  def class_has_alias(cls, name):
  # {{{
    if cls.name.lower() == name.lower(): return True
    return False
  # }}}
# }}}


class LagVar(Var):
  def __init__(self, var, iaxis, lags):
  # {{{
    import numpy as np

    self.iaxis = var.whichaxis(iaxis)
    taxis = var.axes[self.iaxis]
    assert isinstance(taxis, Time), 'must specify a Time axis'
    delt = taxis.delta()
    
    self.lags = np.array(lags).astype('i')
    lag = Lag(values = delt*self.lags, units=taxis.units, startdate={'day':0})
    axes = var.axes[:self.iaxis+1] + (lag, ) + var.axes[self.iaxis+1:]
    self.var = var

    Var.__init__(self, axes, dtype=var.dtype, name=var.name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    lind = self.lags[view.integer_indices[self.iaxis+1]]
    loff, roff = np.min(lind), np.max(lind)

    tind = view.integer_indices[self.iaxis]
    tmin, tmax = np.min(tind), np.max(tind)
    tsl = slice(max(tmin + loff, 0), min(tmax + roff, self.shape[self.iaxis]))
    inview = view.remove(self.iaxis+1).modify_slice(self.iaxis, tsl)
    src = inview.get(self.var, pbar=pbar)

    out = np.empty(view.shape, self.dtype)
    outsl = [0 if i == self.iaxis + 1 else slice(None) for i in range(self.naxes)]
    insl = [slice(None) for i in range(self.naxes-1)]
    for i, l in enumerate(lind):
      valid = (tind + l >= 0) & (tind + l < src.shape[self.iaxis])
      ivalid = np.where(valid)[0]
      insl[self.iaxis] = tind[ivalid] + l
      outsl[self.iaxis] = ivalid
      outsl[self.iaxis+1] = i
      out[outsl] = src[insl]
      outsl[self.iaxis] = np.where(~valid)[0]
      out[outsl] = np.nan
    
    return out
  # }}}
