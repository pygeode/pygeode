# deriv.py - implementation of DerivativeVar

from pygeode.var import Var
class IntegrateVar (Var):
  '''Definite integration variable. For now performs a cumulative integral along 
    the specified axis using a trapezoid rule.'''

  def __init__ (self, var, iaxis, dx=None, v0=None, order=1, type='trap'):
  # {{{
    ''' __init__()'''

    from pygeode.var import Var

    self.iaxis = var.whichaxis(iaxis)
    assert var.shape[iaxis] > 1, "need at least two values along integration axis"

    self.var = var

    if v0 is not None:
      # Confirm the initial values are consistently shaped
      assert v0.naxes == var.naxes - 1
      assert all([a in var.axes for a in v0.axes])

    self.v0 = v0

    if dx is not None:  # Optionally one can specify a coordinate system
      if dx.naxes == 1:
        assert dx.shape[0] == var.shape[iaxis]
        self.dx = dx.replace_axes(newaxes=(var.axes[iaxis],))
      else: # Must be mappable to integrand, with a matching integration axis
        for a in var.axes:
          assert dx.hasaxis(a)
        self.dx = dx
    else:
      self.dx = var.axes[iaxis]

    self.order=order

    assert type in ['trapz', 'rectr', 'rectl']
    self.type = type

    # Construct new variable
    if var.name != '':
      name = 'i' + var.name
    else:
      name = 'i(UnknownVar)'

    Var.__init__(self, var.axes, dtype='d', name=name)
  # }}}

  def getview (self, view, pbar):
  # {{{
    from numpy import arange, min, max, clip
    import numpy as np

    iaxis = self.iaxis
    Ni = self.shape[iaxis]

    # Get all data along the integration axis
    inview = view.unslice(iaxis)  
    data = inview.get(self.var, pbar=pbar.subset(0, 70))

    # Get initial values
    if self.v0 is not None:
      d0 = view.get(self.v0, pbar=pbar.subset(70, 90))
    else:
      shp = [s if i != iaxis else 1 for i, s in enumerate(view.shape)]
      d0 = np.zeros(shp, 'd')

    # Compute differences
    d = np.diff(inview.get(self.dx), axis=iaxis)

    sl1 = [slice(None)] * self.naxes
    sl2 = [slice(None)] * self.naxes
    slr = [slice(None)] * self.naxes
    sl1[iaxis] = slice(None, -1)
    sl2[iaxis] = slice(1, None)
    slr[iaxis] = slice(None, None, self.order)
    
    # Accumulate
    if self.type == 'trapz':
      dat = np.concatenate([d0, (0.5*d*(data[sl1] + data[sl2]))[slr]], iaxis)
    elif self.type == 'rectl':
      dat = np.concatenate([d0, (d*data[sl1])[slr]], iaxis)
    elif self.type == 'rectr':
      dat = np.concatenate([d0, (d*data[sl2])[slr]], iaxis)

    out = np.add.accumulate(dat, iaxis, 'd')[slr]

    # Select the requested values along the integration axis 
    sl1[iaxis] = view.integer_indices[iaxis]
    return out[sl1]
  # }}}

def integrate(var, iaxis, dx=None, v0=None, order = 1, type='trapz'):
  return IntegrateVar(var, var.whichaxis(iaxis), dx, v0, order, type)
