# deriv.py - implementation of DerivativeVar

from pygeode.var import Var
class IntegrateVar (Var):
  '''Definite integration variable. For now performs a cumulative integral along
    the specified axis using a trapezoid rule.'''

  def __init__ (self, var, iaxis, dx=None, v0=None, order=1, type='trapz'):
  # {{{
    ''' __init__()'''

    from pygeode.var import Var

    self.iaxis = var.whichaxis(iaxis)
    if var.shape[iaxis] < 2:
      raise ValueError('At least two values are needed to integrate %s along axis %s' % (var.name, var.axis[iaxis]))

    self.var = var

    if v0 is not None:
      # Confirm the initial values are consistently shaped
      if isinstance(v0, Var):
        if v0.hasaxis(var.axes[self.iaxis]):
          raise ValueError('v0\n %s \n must not share integration axis with \n\n %s' % (v0, var))
        if not all([a in var.axes for a in v0.axes]):
          raise ValueError('v0\n %s \n\n axes must all match those of \n\n %s' % (v0, var))
      self.v0 = v0
    else:
      self.v0 = 0.

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

    if type not in ['trapz', 'rectr', 'rectl']:
      raise ValueError("type (%s) must be one of 'trapz', 'rectr', 'rectl'." % type)

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
    if isinstance(self.v0, Var):
      d0 = view.get(self.v0, pbar=pbar.subset(70, 90))
    else:
      shp = [s if i != iaxis else 1 for i, s in enumerate(view.shape)]
      d0 = np.full(shp, self.v0, 'd')

    # Compute differences
    d = np.diff(inview.get(self.dx), axis=iaxis)

    sl1 = [slice(None)] * self.naxes
    sl2 = [slice(None)] * self.naxes
    slr = [slice(None)] * self.naxes
    sl1[iaxis] = slice(None, -1)
    sl2[iaxis] = slice(1, None)
    slr[iaxis] = slice(None, None, self.order)
    sl1 = tuple(sl1)
    sl2 = tuple(sl2)
    slr = tuple(slr)

    # Accumulate
    if self.type == 'trapz':
      dat = np.concatenate([d0, (0.5*d*(data[sl1] + data[sl2]))[slr]], iaxis)
    elif self.type == 'rectl':
      dat = np.concatenate([d0, (d*data[sl1])[slr]], iaxis)
    elif self.type == 'rectr':
      dat = np.concatenate([d0, (d*data[sl2])[slr]], iaxis)

    out = np.add.accumulate(dat, iaxis, 'd')[slr]

    # Select the requested values along the integration axis
    sl1 = list(sl1)
    sl1[iaxis] = view.integer_indices[iaxis]
    sl1 = tuple(sl1)

    return out[sl1]
  # }}}

def integrate(var, iaxis, dx=None, v0=None, order = 1, type='trapz'):
  '''Computes an indefinite integral along the given axis.

  Parameters
  ----------
  iaxis : string, :class:`Axis` class, or int
    Axis along which to compute integral.
  dx : :class:`Var`, or None (optional)
    Coordinate with respect to which to integrate (see notes). Must
    share axis along which the derivative is being taken. If ``None``, the
    coordinate axis is used.
  v0 : float, :class:`Var`, or None (optional)
    Constant of integration. See notes.
  order: int (1 or -1)
    Direction along axis to integrate. ``1`` corresponds to an integration from
    the first to the last element, while ``-1`` integrates in the other
    direction.
  type : string (optional)
    Type of numerical integral to take. One of 'trapz', 'rectr', or 'rectl';
    defaults to 'trapz'. See notes.

  Returns
  -------
  ivar : :class:`Var`
    Numerical integral of ``var``

  Notes
  -----
  Possible integration methods are
    * 'trapz': trapezoidal rule
    * 'rectl': left rectangle method or Riemann sum
    * 'rectr': right rectangle method or Riemann sum

  Examples
  ========
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t1
  >>> print(t1.Temp.integrate('lon')) # Compute simple derivative
  <Var 'iTemp'>:
    Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  IntegrateVar (dtype="float64")
  '''
  return IntegrateVar(var, var.whichaxis(iaxis), dx, v0, order, type)
