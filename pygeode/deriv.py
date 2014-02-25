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
    getleft = [slice(None)]*daxis + [getleft] + [slice(None)]*(self.naxes-daxis-1)
    getright = [slice(None)]*daxis + [getright] + [slice(None)]*(self.naxes-daxis-1)

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
    getleft = [slice(None)]*daxis + [getleft] + [slice(None)]*(self.naxes-daxis-1)
    getcentre = [slice(None)]*daxis + [getcentre] + [slice(None)]*(self.naxes-daxis-1)
    getright = [slice(None)]*daxis + [getright] + [slice(None)]*(self.naxes-daxis-1)

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
  if df == '2':
    return SecondDerivativeVar(var, var.whichaxis(daxis), dx=dx)
  else:
    return DerivativeVar(var, var.whichaxis(daxis), dx=dx, df=df)
