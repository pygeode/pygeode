#TODO: keep the degenerate axis after the reduction, with a value provided by the user.
#TODO: allow the selection of a range *within* the reducedvar framework
#     (saves some steps for the user)
#     I.e., could use something like:
#                      var.mean(lat=(-45,45))

from pygeode.var import Var

#########################
class ReducedVar(Var):
# {{{
  def __new__ (type, var, indices, *args, **kwargs):
  # {{{
    from pygeode.var import Var

    # Very 'special' case: reducing on a scalar.
    if var.naxes == 0: return var.get()

    new = object.__new__(type)

    varlist = [var] if isinstance(var,Var) else var  # simplify logic below

    # If no indices are specified, work over the whole domain and
    # return the scalar result.
    if len(indices) == 0:
      new.__init__(var, list(range(var.naxes)), *args, **kwargs)
      return new.get()

    # If all variables are in memory, then do the calculation right away
    if all(hasattr(v,'values') for v in varlist):
      new.__init__(var, indices, *args, **kwargs)
      return new.load(pbar=None)

    # Otherwise, proceeed as usual
    return new
  # }}}

  def __init__ (self, var, indices):
  # {{{
    from pygeode.var import Var
    import numpy as np
    from pygeode.tools import combine_axes, common_dtype
    # Are we given a list of variables to work on in parallel?
    if isinstance(var,(tuple,list)):
      axes = combine_axes(var)
      dtype = common_dtype(var)
    else:
      axes = var.axes
      dtype = var.dtype

#    if not isinstance(indices,(list,tuple)): indices = [indices]
    indices = np.sort([var.whichaxis(i) for i in indices])
    assert len(indices) > 0, "no reduction axes specified"

    N = [len(axes[i]) for i in indices]
    # Check for degenerate reductions (ill-defined)
    for i,n in enumerate(N):
      if n == 0:  raise ValueError("Can't do a reduction over axis '%s' - length is 0."%axes[i].name)
    N = int(np.product(N))
    self.N =  N # number of values to reduce over
    self.var = var
    self.indices = tuple(indices)

    self.in_axes = axes

    # Remove the reduction axis from the output variable
    axes = [a for i,a in enumerate(axes) if i not in indices]

    Var.__init__(self, axes, dtype=dtype, name=var.name, atts=var.atts, plotatts=var.plotatts)

  # }}}
# }}}

class MinVar(ReducedVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npmin
    out = np.empty(view.shape, self.dtype)
    out[()] = float('inf')
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      out[outsl] = np.minimum(out[outsl], npmin(indata, self.indices))
    return out
# }}}

class NANMinVar(ReducedVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npnanmin
    out = np.full(view.shape, np.nan, self.dtype)
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      #Ignore NaNs when finding mininum
      out[outsl] = np.nanmin([out[outsl], npnanmin(indata, self.indices)], 0)
    return out
# }}}

class MaxVar(ReducedVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npmax
    out = np.empty(view.shape, self.dtype)
    out[()] = float('-inf')
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      out[outsl] = np.maximum(out[outsl], npmax(indata, self.indices))
    return out
# }}}

class NANMaxVar(ReducedVar):
# {{{ 
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npnanmax
    out = np.full(view.shape, np.nan, self.dtype)
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      #Ignore NaNs when finding maximum
      out[outsl] = np.nanmax([out[outsl], npnanmax(indata, self.indices)], 0)
    return out
# }}}

class ArgMaxVar(ReducedVar):
# {{{
  def __init__(self, var, index):
  # {{{
    from pygeode.var import Var
    import numpy as np
    axes = var.axes
    index = var.whichaxis(index)

    self.N = len(axes[index])
    self.var = var
    self.indices = [index]
    self.in_axes = axes

    # Remove the reduction axis from the output variable
    axes = [a for i,a in enumerate(axes) if i != index]

    Var.__init__(self, axes, dtype='i', name=var.name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npmax
    out = np.empty(view.shape, self.dtype)
    index = self.indices[0]
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      out[outsl] = np.argmax(indata, index)
    return out
# }}}
class ArgMinVar(ReducedVar):
# {{{
  def __init__(self, var, index):
  # {{{
    from pygeode.var import Var
    import numpy as np
    axes = var.axes
    index = var.whichaxis(index)

    self.N = len(axes[index])
    self.var = var
    self.indices = [index]
    self.in_axes = axes

    # Remove the reduction axis from the output variable
    axes = [a for i,a in enumerate(axes) if i != index]

    Var.__init__(self, axes, dtype='i', name=var.name, atts=var.atts, plotatts=var.plotatts)
  # }}}

  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npmax
    out = np.empty(view.shape, self.dtype)
    index = self.indices[0]
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      out[outsl] = np.argmin(indata, index)
    return out
# }}}

class SumVar(ReducedVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npsum
    out = np.zeros(view.shape, self.dtype)
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      out[outsl] += npsum(indata, self.indices)
    return out
# }}}
class WeightedSumVar(ReducedVar):
# {{{
  '''WeightedSumVar(ReducedVar) - computes weighted sum.'''
  def __init__(self, var, indices, weights):
  # {{{
    ReducedVar.__init__(self, var, indices)

    # Confirm that weights are defined for the reduction axes
    raxes = [a for a in var.axes if a not in self.axes]
    assert all([a in raxes for a in weights.axes]), 'The provided weights do not match the reduced axes'

    self.mweights = weights
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    from pygeode.tools import loopover, npsum
    out = np.zeros(view.shape, self.dtype)
    for outsl, (indata, inw) in loopover([self.var, self.mweights], view, self.var.axes, pbar=pbar):
      out[outsl] += npsum(indata * inw, self.indices)  # Product of data and weights
    return out
  # }}}
# }}}

class NANSumVar(ReducedVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npnansum
    out = np.full(view.shape, np.nan, self.dtype)
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      # Accumulation must be nan-safe
      out[outsl] = np.nansum([out[outsl], npnansum(indata, self.indices)], 0)
    return out
# }}}
class WeightedNANSumVar(ReducedVar):
# {{{
  '''WeightedNANSumVar(ReducedVar) - computes weighted sum, neglecting NaNs.'''
  def __init__(self, var, indices, weights):
  # {{{
    ReducedVar.__init__(self, var, indices)

    # Confirm that weights are defined for the reduction axes
    raxes = [a for a in var.axes if a not in self.axes]
    assert all([a in raxes for a in weights.axes]), 'The provided weights do not match the reduced axes'

    self.mweights = weights
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    from pygeode.tools import loopover, npnansum
    out = np.full(view.shape, np.nan, self.dtype)
    for outsl, (indata, inw) in loopover([self.var, self.mweights], view, self.var.axes, pbar=pbar):
      # Accumulation must be nan-safe
      out[outsl] = np.nansum([out[outsl], \
              npnansum(indata * inw, self.indices)], 0)     # Product of data and weights

    return out
  # }}}
# }}}

class MeanVar(ReducedVar):
# {{{
  '''MeanVar(ReducedVar) - computes unweighted mean.'''
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npsum
    out = np.zeros(view.shape, self.dtype)
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      out[outsl] += npsum(indata, self.indices)  

    return out / self.N
 # }}}
class WeightedMeanVar(ReducedVar):
# {{{
  '''WeightedMeanVar(ReducedVar) - computes weighted mean.'''
  def __init__(self, var, indices, weights):
  # {{{
    ReducedVar.__init__(self, var, indices)

    # Confirm that weights are defined for the reduction axes
    raxes = [a for a in var.axes if a not in self.axes]
    assert all([a in raxes for a in weights.axes]), 'The provided weights do not match the reduced axes'

    self.mweights = weights
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    from pygeode.tools import loopover, npsum
    out = np.zeros(view.shape, self.dtype)
    W = np.zeros(view.shape, self.dtype)
    for outsl, (indata, inw) in loopover([self.var, self.mweights], view, self.var.axes, pbar=pbar):
      out[outsl] += npsum(indata * inw, self.indices)  # Product of data and weights
      f = indata.size / (inw.size * out[outsl].size)
      W[outsl] += npsum(inw, self.indices) * f # Sum of weights

    return out / W
  # }}}
# }}}

class NANMeanVar(ReducedVar):
# {{{
  '''NANMeanVar(ReducedVar) - computes unweighted mean, ignoring NANs.'''
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npnansum
    out = np.full(view.shape, np.nan, self.dtype)
    N   = np.full(view.shape, np.nan, self.dtype)
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      # Must increment in a nan-safe way
      out[outsl] = np.nansum([out[outsl], npnansum(indata, self.indices)], 0)
      # Sum of weights (kludge to get masking right)
      N[outsl] = np.nansum([N[outsl], npnansum(~np.isnan(indata), self.indices)], 0)

    nmsk = (N > 0.)
    out[nmsk] /= N[nmsk]
    out[~nmsk] = np.nan

    return out
 # }}}
class WeightedNANMeanVar(ReducedVar):
# {{{ 
  '''WeightedNANMeanVar(ReducedVar) - computes weighted mean.'''
  def __init__(self, var, indices, weights):
  #  {{{
    ReducedVar.__init__(self, var, indices)

    # Confirm that weights are defined for the reduction axes
    raxes = [a for a in var.axes if a not in self.axes]
    assert all([a in raxes for a in weights.axes]), 'The provided weights do not match the reduced axes'

    self.mweights = weights
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    from pygeode.tools import loopover, npnansum
    out = np.full(view.shape, np.nan, self.dtype)
    W   = np.full(view.shape, np.nan, self.dtype)
    for outsl, (indata, inw) in loopover([self.var, self.mweights], view, self.var.axes, pbar=pbar):
      # Must increment in a nan-safe way
      out[outsl] = np.nansum([out[outsl], \
              npnansum(indata * inw, self.indices)], 0)     # Product of data and weights
      W[outsl] = np.nansum([W[outsl], \
              npnansum((~np.isnan(indata)) * inw, self.indices)], 0)  # Sum of weights (kludge to get masking right)

    nmsk = (W > 0.)
    out[nmsk] /= W[nmsk]
    out[~nmsk] = np.nan

    return out
  # }}}
# }}}

# naive single-pass variance
#TODO: make it more numerically stable, add weights
class VarianceVar(ReducedVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npsum
    x = np.zeros(view.shape, self.dtype)
    x2 = np.zeros(view.shape, self.dtype)
    N = self.N
    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      x[outsl] += npsum(indata, self.indices)  
      x2[outsl] += npsum(indata**2, self.indices)  

    x /= N
    return (x2 - N*x**2) / (N - 1)
# }}}
class WeightedVarianceVar(ReducedVar):
# {{{
  '''WeightedVarianceVar(ReducedVar) - computes weighted mean.'''
  def __init__(self, var, indices, weights):
  # {{{
    ReducedVar.__init__(self, var, indices)

    # Confirm that weights are defined for the reduction axes
    raxes = [a for a in var.axes if a not in self.axes]
    assert all([a in raxes for a in weights.axes]), 'The provided weights do not match the reduced axes'

    self.mweights = weights
  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    from pygeode.tools import loopover, npsum
    x = np.zeros(view.shape, self.dtype)
    x2 = np.zeros(view.shape, self.dtype)
    W = np.zeros(view.shape, self.dtype)
    for outsl, (indata, inw) in loopover([self.var, self.mweights], view, self.var.axes, pbar=pbar):
      x[outsl] += npsum(indata * inw, self.indices)  # Product of data and weights
      x2[outsl] += npsum(indata**2 * inw, self.indices)  # Product of data and weights
      f = indata.size / (inw.size * x[outsl].size)
      W[outsl] += npsum(inw, self.indices) * f # Sum of weights

    return (x2 - x*x/W) / (W - 1)
  # }}}
# }}}

# NB: There is possibly an issue with this for integer datatypes
class NANVarianceVar(ReducedVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.tools import loopover, npnansum
    x  = np.full(view.shape, np.nan, self.dtype)
    x2 = np.full(view.shape, np.nan, self.dtype)
    N  = np.full(view.shape, np.nan, self.dtype)

    out = np.zeros(view.shape, self.dtype)

    for outsl, (indata,) in loopover(self.var, view, pbar=pbar):
      x[outsl] = np.nansum([x[outsl],\
                  npnansum(indata, self.indices)], 0)
      x2[outsl] = np.nansum([x2[outsl],\
                  npnansum(indata**2, self.indices)], 0)  
      N[outsl] = np.nansum([N[outsl],
                  npnansum(~np.isnan(indata), self.indices)], 0)

    nmsk = (N > 1.)
    out[nmsk] = x2[nmsk] - x[nmsk]**2/N[nmsk]
    out[nmsk] /= N[nmsk] - 1.
    out[~nmsk] = np.nan

    return out
# }}}

# naive single-pass standard deviation
class SDVar(VarianceVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    variance = VarianceVar.getview (self, view, pbar)
    # Very small variances may become negative due to numerical issues with
    # the single-pass variance routine used.
    if np.isscalar(variance):
      if variance < 0.: variance = 0.
    else:
      variance[np.where(variance<=0)] = 0
    return np.sqrt(variance)
# }}}
class NANSDVar(NANVarianceVar):
# {{{
  def getview (self, view, pbar):
    import numpy as np
    variance = NANVarianceVar.getview (self, view, pbar)
    # Very small variances may become negative due to numerical issues with
    # the single-pass variance routine used.
    if np.isscalar(variance):
      if variance < 0.: variance = 0.
    else:
      variance[variance<=0] = 0
    return np.sqrt(variance)
# }}}

def min (var, *axes): 
# {{{
  '''
    Computes the minimum value of this variable. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the minimum should be found. If none are provided, the 
      global minimum is found.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the global minimum
      is requested, a python scalar is returned.

    See Also
    --------
    max
    nanmax
    nanmin
  '''
  return MinVar(var, axes)
# }}}
def nanmin (var, *axes): 
# {{{
  '''
    Computes the minimum value of this variable, ignoring NaNs. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the minimum should be found. If none are provided, the 
      global minimum is found.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the global minimum
      is requested, a python scalar is returned.

    See Also
    --------
    min
    max
    nanmax
  '''
  return NANMinVar(var, axes)
# }}}
def max (var, *axes): 
# {{{
  '''
    Computes the maximum value of this variable. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the maximum should be found. If none are provided, the 
      global maximum is found.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the global maximum
      is requested, a python scalar is returned.

    See Also
    --------
    min
    nanmax
    nanmin
  '''
  return MaxVar(var, axes)
# }}}
def nanmax (var, *axes): 
# {{{
  '''
    Computes the maximum value of this variable, ignoring NaNs. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the maximum should be found. If none are provided, the 
      global maximum is found.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the global maximum
      is requested, a python scalar is returned.

    See Also
    --------
    max
    min
    nanmin
  '''
  return NANMaxVar(var, axes)
# }}}
def argmax (var, axis): 
# {{{
  '''
    Finds the index of the maximum value of this variable along the given axis. 

    Parameters
    ----------
    axis : a single axis identifier (string, :class:`Axis`, or int) (optional)
      Axis over which the index of the maximum should be found. 

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on all axes of this variable except ``axis``,
      containing the index of the maximum.

    See Also
    --------
    argmin
  '''
  return ArgMaxVar(var, axis)
# }}}
def argmin (var, axis): 
# {{{
  '''
    Finds the index of the minumum value of this variable along the given axis. 

    Parameters
    ----------
    axis : a single axis identifier (string, :class:`Axis`, or int) (optional)
      Axis over which the index of the minimum should be found. 

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on all axes of this variable except ``axis``,
      containing the index of the minimum.

    See Also
    --------
    argmax
  '''
  return ArgMinVar(var, axis)
# }}}


def sum (var, *axes, **kwargs): 
# {{{
  '''
    Computes the sum of this variable. NB: Unlike mean, weights are not used by default.

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the sum should be computed. If none are provided, the 
      sum is computed over the whole domain.

    weights : boolean or :class:`Var` (optional)
      If provided, a weighted sum is performed. If True, the default
      weights associated with the variable are used (getweights). If False or None (the default), no
      weighting is performed. Finally, custom weights can be provided in the form of a :class:`Var`;
      this var must be defined on a subset of the axes being summed over.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the sum is computed over
      the whole domain, a python scalar is returned.

    See Also
    --------
    getweights
  '''
  weights = kwargs.pop('weights', False)
  if weights is True:
    weights = var.getweights(axes)

  if weights is False or weights is None or weights.naxes == 0:
    # If weights aren't provided or predefined, return unweighted mean
    return SumVar(var, axes)

  return WeightedSumVar(var, axes, weights=weights, **kwargs)
# }}}
def nansum (var, *axes, **kwargs):
# {{{
  '''
    Computes the sum of this variable, ignoring any NaNs. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the sum should be computed. If none are provided, the 
      sum is computed over the whole domain.

    weights : boolean or :class:`Var` (optional)
      If provided, a weighted sum is performed. If True, the default
      weights associated with the variable are used (getweights). If False or None (the default), no
      weighting is performed. Finally, custom weights can be provided in the form of a :class:`Var`;
      this var must be defined on a subset of the axes being summed over.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the sum is computed over
      the whole domain, a python scalar is returned.

    See Also
    --------
    getweights
  '''
  weights = kwargs.pop('weights', True)
  if weights is True:
    weights = var.getweights(axes)

  if weights is False or weights is None or weights.naxes == 0:
    # If weights aren't provided or predefined, return unweighted mean
    return NANSumVar(var, axes)

  return WeightedNANSumVar(var, axes, weights=weights, **kwargs)
# }}}

def mean (var, *axes, **kwargs): 
# {{{
  '''
    Computes the mean of this variable. If weights are present on any of the
    axes, a weighted mean is computed by default.

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the average should be computed. If none are provided, the 
      mean is computed over the whole domain.

    weights : boolean or :class:`Var` (optional)
      If provided, a weighted mean is performed. If True (the default), the default
      weights associated with the variable are used (getweights). If False, or None, no 
      weighting is performed. Finally, custom weights can be provided in the form of a 
      :class:`Var`; this var must be defined on a subset of the axes being averaged over.


    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the mean is computed over
      the whole domain, a python scalar is returned.

    See Also
    --------
    nanmean
    getweights
  '''
  weights = kwargs.pop('weights', True)
  if weights is True:
    weights = var.getweights(axes)

  if weights is False or weights is None or weights.naxes == 0:
    # If weights aren't provided or predefined, return unweighted mean
    return MeanVar(var, axes)

  return WeightedMeanVar(var, axes, weights=weights, **kwargs)
# }}}
def nanmean (var, *axes, **kwargs):
# {{{
  '''
    Computes the mean of this variable, ignoring any NaNs in the domain. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the average should be computed. If none are provided, the 
      mean is computed over the whole domain.

    weights : boolean or :class:`Var` (optional)
      If provided, a weighted mean is performed. If True (the default), the default
      weights associated with the variable are used (getweights). If False, or None, no 
      weighting is performed. Finally, custom weights can be provided in the form of a 
      :class:`Var`; this var must be defined on a subset of the axes being averaged over.


    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the mean is computed over
      the whole domain, a python scalar is returned.

    See Also
    --------
    mean
    getweights
  '''
  weights = kwargs.pop('weights', True)
  if weights is True:
    weights = var.getweights(axes)

  if weights is False or weights is None or weights.naxes == 0:
    # If weights aren't provided or predefined, return unweighted mean
    return NANMeanVar(var, axes)

  return WeightedNANMeanVar(var, axes, weights=weights, **kwargs)
# }}}

def stdev (var, *axes): 
# {{{
  '''
    Computes the standard deviation of this variable. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the standard deviation should be computed. If none are provided, the 
      standard deviation is computed over the whole domain.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the variance is
      computed over the whole domain, a python scalar is returned.

    See Also
    --------
    variance
    nanstdev
    nanvariance
  '''
  return SDVar (var, axes)
# }}}
def nanstdev (var, *axes): 
# {{{
  '''
    Computes the standard deviation of this variable, ignoring any NaNs present. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the standard deviation should be computed. If none are provided, the 
      standard deviation is computed over the whole domain.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the variance is
      computed over the whole domain, a python scalar is returned.

    See Also
    --------
    stdev
    variance
    nanvariance
  '''
  return NANSDVar (var, axes)
# }}}

def variance (var, *axes, **kwargs): 
# {{{
  '''
    Computes the variance of this variable. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the variance should be computed. If none are provided, the 
      variance is computed over the whole domain.

    weights : boolean or :class:`Var` (optional, default False)
      If provided, a weighted variance is calculated. If True, the default
      weights associated with the variable are used (getweights). If False, or None, no 
      weighting is performed. Finally, custom weights can be provided in the form of a 
      :class:`Var`; this var must be defined on a subset of the axes being reduced.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the variance is
      computed over the whole domain, a python scalar is returned.

    See Also
    --------
    stdev
    nanvariance
    nanstdev
  '''
  weights = kwargs.pop('weights', False)
  if weights is True:
    weights = var.getweights(axes)

  if weights is False or weights is None or weights.naxes == 0:
    # If weights aren't provided or predefined, return unweighted variance
    return VarianceVar(var, axes)

  return WeightedVarianceVar(var, axes, weights=weights, **kwargs)
# }}}
def nanvariance (var, *axes): 
# {{{
  '''
    Computes the variance of this variable, ignoring any NaNs. 

    Parameters
    ----------
    *axes : any number of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which the variance should be computed. If none are provided, the 
      variance is computed over the whole domain.

    Returns
    -------
    out : :class:`Var`
      :class:`Var` defined on a subgrid of this variable. If the variance is
      computed over the whole domain, a python scalar is returned.

    See Also
    --------
    variance
    stdev
    nanstdev
  '''
  return NANVarianceVar(var, axes)
# }}}

class_flist = [min, max, nanmin, nanmax, argmax, argmin, sum, mean, variance, stdev, nansum, nanmean, nanvariance, nanstdev]

del Var
