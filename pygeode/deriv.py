# deriv.py - implementation of DerivativeVar

from pygeode.var import Var
class DerivativeVar (Var):
  '''Derivative variable. For now performs centred differences in the
    interior of the axis and one sided differences on the boundaries.'''

  def __init__ (self, var, daxis, dx=None, df='centre'):
  # {{{
    ''' __init__()'''

    from pygeode.var import Var

    self.daxis = daxis = var.whichaxis(daxis)
    assert var.shape[daxis] > 1, "need at least two values along differentiation axis"

    if dx is not None:
      if dx.naxes == 1:
        assert dx.shape[0] == var.shape[daxis]
        self.dx = dx.replace_axes(newaxes=(var.axes[daxis],))
      else:
        assert all([dx.hasaxis(a) for a in var.axes])
        self.dx = dx
    else:
      self.dx = var.axes[daxis]

    assert df in ['centre', 'left', 'right']
    self.df = df

    self.var = var

    # Construct new variable

    if var.name != '':
      name = 'd' + var.name
    else:
      name = 'd(UnknownVar)'

    Var.__init__(self, var.axes, var.dtype, name=name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    from numpy import arange, min, max, clip
    import numpy as np

    daxis = self.daxis
    Nd = self.shape[daxis]

    # Get integer indices along the differentiation axis
    ind = view.integer_indices[daxis]

    # Want to do the finite difference with values to the left & right
    if self.df == 'centre':
      left = ind-1
      right = ind+1
    elif self.df == 'left':
      left = ind-1
      right = ind
    elif self.df == 'right':
      left = ind
      right = ind+1
    else:
      assert False, 'Unrecognized differencing type.'

    # Truncate to the left & right boundaries
    right[left==-1] = 1
    left[left==-1] = 0
    left[right==Nd] = Nd-2
    right[right==Nd] = Nd-1

    # All the points we need to request (unique occurrences only)
    allpoints = np.union1d(left, right)

    allview = view.modify_slice(daxis, allpoints)

    # Get the data and axis values for these points
    allvalues = allview.get(self.var, pbar=pbar)
    allaxis = allview.get(self.dx)

    # Define a map from these unique points back to the left & right arrays
    getleft = np.searchsorted(allpoints,left)
    getright = np.searchsorted(allpoints,right)
    # Make this 1D map into the right shape for the view (if multi-dimensional)
    getleft = tuple([slice(None)]*daxis + [getleft] + [slice(None)]*(self.naxes-daxis-1))
    getright = tuple([slice(None)]*daxis + [getright] + [slice(None)]*(self.naxes-daxis-1))

    # Finally, get the left & right values, and do the finite difference
    L = allvalues[getleft]
    R = allvalues[getright]
    La = allaxis[getleft]
    Ra = allaxis[getright]

    values = (R-L)/(Ra-La)
    values = np.asarray(values, self.dtype)
    return values
  # }}}

class SecondDerivativeVar (Var):
  '''Finds second derivative, using centred differences in the
    interior of the axis and one sided differences on the boundaries.'''

  def __init__ (self, var, daxis, dx=None):
  # {{{
    ''' __init__()'''

    from pygeode.var import Var

    self.daxis = daxis = var.whichaxis(daxis)
    assert var.shape[daxis] > 1, "need at least two values along differentiation axis"

    if dx is not None:
      if dx.naxes == 1:
        assert dx.shape[0] == var.shape[daxis]
        self.dx = dx.replace_axes(newaxes=(var.axes[daxis],))
      else:
        assert all([dx.hasaxis(a) for a in var.axes])
        self.dx = dx
    else:
      self.dx = var.axes[daxis]

    self.var = var

    # Construct new variable

    if var.name != '':
      name = 'd2' + var.name
    else:
      name = 'd2(UnknownVar)'

    Var.__init__(self, var.axes, var.dtype, name=name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    from numpy import arange, min, max, clip
    import numpy as np

    daxis = self.daxis
    Nd = self.shape[daxis]

    # Get integer indices along the differentiation axis
    ind = view.integer_indices[daxis]

    # Want to do the finite difference with values to the left & right
    left = ind-1
    centre = ind
    right = ind+1

    # Truncate to the left & right boundaries
    left[left==-1] = 2
    right[right==Nd] = Nd-3

    # All the points we need to request (unique occurrences only)
    allpoints = np.union1d(left, np.union1d(centre, right))

    allview = view.modify_slice(daxis, allpoints)

    # Get the data and axis values for these points
    allvalues = allview.get(self.var, pbar=pbar)
    allaxis = allview.get(self.dx)

    # Define a map from these unique points back to the left & right arrays
    getleft = np.searchsorted(allpoints,left)
    getcentre = np.searchsorted(allpoints,centre)
    getright = np.searchsorted(allpoints,right)
    # Make this 1D map into the right shape for the view (if multi-dimensional)
    getleft   = tuple([slice(None)]*daxis + [getleft] + [slice(None)]*(self.naxes-daxis-1))
    getcentre = tuple([slice(None)]*daxis + [getcentre] + [slice(None)]*(self.naxes-daxis-1))
    getright  = tuple([slice(None)]*daxis + [getright] + [slice(None)]*(self.naxes-daxis-1))

    # Finally, get the left & right values, and do the finite difference
    L = allvalues[getleft]
    C = allvalues[getcentre]
    R = allvalues[getright]
    La = allaxis[getleft]
    Ca = allaxis[getcentre]
    Ra = allaxis[getright]

    den = 2. / ((Ra - Ca) * (Ca - La) * (Ra - La))
    dL = (Ra - Ca) * den
    dC = (La - Ra) * den
    dR = (Ca - La) * den
    return np.asarray(dL * L + dC * C + dR * R, self.dtype)
  # }}}

def deriv(var, daxis, dx=None, df='centre'):
  '''Computes derivative along the given axis.

  Parameters
  ----------
  daxis : string, :class:`Axis` class, or int
    Axis along which to compute derivative.
  dx : :class:`Var`, or None (optional)
    Coordinate with respect to which to take the derivative (see notes). Must
    share axis along which the derivative is being taken. If ``None``, the
    coordinate axis is used.
  df : string (optional)
    Type of derivative to take. One of 'left', 'right', 'centre', or '2'. See
    notes.

  Returns
  -------
  dvar : :class:`Var`
    Numerical derivative of ``var``

  Notes
  -----
  The derivative is computed using a forward (df = 'right'), backward (df =
  'left'), or centred (df = 'centre') difference approximation; for instance,
  the forward difference is computed as: ::

    dvar[i] = (var[i+1] - var[i]) / (dx[i+1] - dx[i]).

  One-sided differences are used at the axis boundaries so that ``dvar`` is
  defined on the same axis as ``var``. The second derivative can also be
  computed (df = '2')

  Examples
  ========
  >>> import pygeode as pyg, numpy as np
  >>> from pygeode.tutorial import t1
  >>> print(t1.Temp.deriv('lon')) # Compute simple derivative
  <Var 'dTemp'>:
    Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  DerivativeVar (dtype="float64")
  >>> x = 6317e3 * pyg.cosd(t1.lat) * np.pi / 180. * t1.lon
  >>> print(t1.Temp.deriv('lon', dx=x, df='2')) # Compute 2nd derivative with respect to geometric length
  <Var 'd2Temp'>:
    Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  SecondDerivativeVar (dtype="float64")
  '''
  if df == '2':
    return SecondDerivativeVar(var, var.whichaxis(daxis), dx=dx)
  else:
    return DerivativeVar(var, var.whichaxis(daxis), dx=dx, df=df)
