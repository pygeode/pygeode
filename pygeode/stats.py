__all__ = ('correlate', 'regress', 'difference', 'isnonzero')

import numpy as np
from scipy.stats import t as tdist

sigs = (-1.1, -0.05, -0.01, 0., 0.01, 0.05, 1.1)
sigs_c = (  (1.0, 1.0, 1.0),
            (0.9, 0.9, 1.0), 
            (0.6, 0.6, 0.9), 
            (0.9, 0.6, 0.6), 
            (1.0, 0.9, 0.9), 
            (1.0, 1.0, 1.0))

def correlate(X, Y, axes=None, pbar=None):
# {{{
  r''' correlate(X, Y) - returns correlation between variables X and Y
      computed over all axes shared by x and y. Returns \rho_xy, and p values
      for \rho_xy assuming x and y are normally distributed as Storch and Zwiers 1999
      section 8.2.3.'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npnansum
  from pygeode.view import View

  # Put all the axes being reduced over at the end 
  # so that we can reshape 
  srcaxes = combine_axes([X, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [X.axes, Y.axes])
  if axes is not None:
    ri_new = []
    for a in axes:
      ri_new.append(whichaxis([srcaxes[i] for i in riaxes], a))
    oiaxes.extend([r for r in riaxes if r not in ri_new])
    riaxes = ri_new
    
  oaxes = [srcaxes[i] for i in oiaxes]
  inaxes = oaxes + [srcaxes[i] for i in riaxes]
  oview = View(oaxes) 
  iview = View(inaxes) 
  siaxes = range(len(oaxes), len(srcaxes))

  assert len(riaxes) > 0, '%s and %s share no axes to be correlated over' % (X.name, Y.name)
  xview = oview
  for i in range(len(oaxes)): 
    if not X.hasaxis(oaxes[i]): xview = xview.modify_slice(i, 0)
  yview = oview
  for i in range(len(oaxes)): 
    if not Y.hasaxis(oaxes[i]): yview = yview.modify_slice(i, 0)
  sxaxes = [whichaxis(inaxes, a) for a in axes]
  syaxes = [whichaxis(inaxes, a) for a in axes]
  xshape = []
  yshape = []
  for i, s in enumerate(iview.shape):
    if i < len(oaxes) and Y.hasaxis(inaxes[i]): yshape.append(s)
    else: yshape.append(1)
    if i < len(oaxes) and X.hasaxis(inaxes[i]): xshape.append(s)
    else: xshape.append(1)

  # Construct work arrays
  x  = np.zeros(xview.shape, 'd')*np.nan
  Nx = np.zeros(xview.shape, 'd')*np.nan
  y  = np.zeros(yview.shape, 'd')*np.nan
  Ny = np.zeros(yview.shape, 'd')*np.nan
  xx = np.zeros(oview.shape, 'd')*np.nan
  yy = np.zeros(oview.shape, 'd')*np.nan
  xy = np.zeros(oview.shape, 'd')*np.nan
  Na = np.zeros(oview.shape, 'd')*np.nan

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  # Accumulate 1st moments
  for outsl, (xdata,) in loopover([X], xview, inaxes, pbar):
    xdata = xdata.astype('d')
    x[outsl] = np.nansum([x[outsl], npnansum(xdata, sxaxes)], 0)
    Nx[outsl] = np.nansum([Nx[outsl], npnansum(1. + xdata*0., sxaxes)], 0) 
  x = (x / Nx).reshape(*xshape)

  for outsl, (ydata,) in loopover([Y], yview, inaxes, pbar):
    ydata = ydata.astype('d')
    y[outsl]  = np.nansum([y[outsl], npnansum(ydata, syaxes)], 0)
    Ny[outsl] = np.nansum([Ny[outsl], npnansum(1. + ydata*0., syaxes)], 0) 
  y = (y / Ny).reshape(*yshape)

  for outsl, (xdata, ydata) in loopover([X, Y], oview, inaxes, pbar):
    xdata = xdata.astype('d') - x
    ydata = ydata.astype('d') - y
    xydata = xdata*ydata

    # It seems np.nansum does not broadcast its arguments automatically
    # so there must be a better way of doing this...
    xbc = [s1 / s2 for s1, s2 in zip(xx[outsl].shape, xdata.shape)]
    ybc = [s1 / s2 for s1, s2 in zip(yy[outsl].shape, ydata.shape)]
    xx[outsl] = np.nansum([xx[outsl], np.tile(npnansum(xdata**2, siaxes), xbc)], 0)
    yy[outsl] = np.nansum([yy[outsl], np.tile(npnansum(ydata**2, siaxes), ybc)], 0)
    xy[outsl] = np.nansum([xy[outsl], npnansum(xydata, siaxes)], 0)

    # Sum of weights (kludge to get masking right)
    Na[outsl] = np.nansum([Na[outsl], npnansum(1. + xydata*0., siaxes)], 0) 

  # Compute correlation coefficient, t-statistic, p-value
  rho = xy/np.sqrt(xx*yy)
  den = 1 - rho**2
  den[den < 1e-14] = 1e-14 # Saturate the denominator to avoid div by zero warnings
  t = np.abs(rho) * np.sqrt((Na - 2.)/den)
  p = tdist.cdf(t, Na-2) * np.sign(rho)

  # Construct and return variables
  xn = X.name if X.name != '' else 'X' # Note: could write:  xn = X.name or 'X'
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  Rho = Var(oaxes, values=rho, name='C(%s, %s)' % (xn, yn))
  P = Var(oaxes, values=p, name='P(C(%s,%s) != 0)' % (xn, yn))
  return Rho, P
# }}}

def regress(X, Y, axes=None, pbar=None):
# {{{
  ''' regress(X, Y) - returns correlation between variables X and Y
      computed over axes. Returns rho_xy, and p values
      for rho_xy assuming x and y are normally distributed as Storch and Zwiers 1999
      section 8.2.3.'''
  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npsum
  from pygeode.view import View

  srcaxes = combine_axes([X, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [X.axes, Y.axes])
  if axes is not None:
    ri_new = []
    for a in axes:
      ri_new.append(whichaxis([srcaxes[i] for i in riaxes], a))
    oiaxes.extend([r for r in riaxes if r not in ri_new])
    riaxes = ri_new
    
  oaxes = [srcaxes[i] for i in oiaxes]
  inaxes = oaxes + [srcaxes[i] for i in riaxes]
  oview = View(oaxes) 
  siaxes = range(len(oaxes), len(srcaxes))

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  assert len(riaxes) > 0, '%s and %s share no axes to be regressed over' % (X.name, Y.name)

  # Construct work arrays
  x = np.zeros(oview.shape, 'd')
  y = np.zeros(oview.shape, 'd')
  xx = np.zeros(oview.shape, 'd')
  xy = np.zeros(oview.shape, 'd')
  yy = np.zeros(oview.shape, 'd')

  # Accumulate data
  for outsl, (xdata, ydata) in loopover([X, Y], oview, inaxes, pbar=pbar):
    xdata = xdata.astype('d')
    ydata = ydata.astype('d')
    x[outsl] += npsum(xdata, siaxes)
    y[outsl] += npsum(ydata, siaxes)
    xx[outsl] += npsum(xdata**2, siaxes)
    yy[outsl] += npsum(ydata**2, siaxes)
    xy[outsl] += npsum(xdata*ydata, siaxes)

  N = np.prod([len(srcaxes[i]) for i in riaxes])

  # remove the mean (NOTE: numerically unstable if mean >> stdev)
  xx -= x**2/N
  yy -= y**2/N
  xy -= (x*y)/N

  m = xy/xx
  b = (y - m*x)/float(N)
  sige = (yy - m * xy) / (N - 2.)
  t = np.abs(m) * np.sqrt(xx / sige)
  p = tdist.cdf(t, N-2) * np.sign(m)
  xn = X.name if X.name != '' else 'X'
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  M = Var(oaxes, values=m, name='%s vs. %s' % (yn, xn))
  B = Var(oaxes, values=b, name='Intercept (%s vs. %s)' % (yn, xn))
  P = Var(oaxes, values=p, name='P(%s vs. %s != 0)' % (yn, xn))
  return M, B, P
# }}}

def difference(X, Y, axes, alpha=0.05, Nx_fac = None, Ny_fac = None, pbar=None):
# {{{
  ''' difference(X, Y) - calculates difference between the mean values of X and Y
      averaged over the dimensions specified by axes. Returns X - Y, p values, confidence
      intervals, and degrees of freedom.'''

  from pygeode.tools import combine_axes, whichaxis, loopover, npsum
  from pygeode.view import View

  srcaxes = combine_axes([X, Y])
  riaxes = [whichaxis(srcaxes, n) for n in axes]
  raxes = [a for i, a in enumerate(srcaxes) if i in riaxes]
  oaxes = [a for i, a in enumerate(srcaxes) if i not in riaxes]
  oview = View(oaxes) 

  ixaxes = [X.whichaxis(n) for n in axes]
  Nx = np.product([len(X.axes[i]) for i in ixaxes])

  iyaxes = [Y.whichaxis(n) for n in axes]
  Ny = np.product([len(Y.axes[i]) for i in iyaxes])
  
  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  assert Nx > 1, '%s has only one element along the reduction axes' % X.name
  assert Ny > 1, '%s has only one element along the reduction axes' % Y.name

  # Construct work arrays
  x = np.zeros(oview.shape, 'd')
  y = np.zeros(oview.shape, 'd')
  xx = np.zeros(oview.shape, 'd')
  yy = np.zeros(oview.shape, 'd')

  # Accumulate data
  for outsl, (xdata,) in loopover([X], oview, pbar=pbar):
    xdata = xdata.astype('d')
    x[outsl] += npsum(xdata, ixaxes)
    xx[outsl] += npsum(xdata**2, ixaxes)

  for outsl, (ydata,) in loopover([Y], oview, pbar=pbar):
    ydata = ydata.astype('d')
    y[outsl] += npsum(ydata, iyaxes)
    yy[outsl] += npsum(ydata**2, iyaxes)

  # remove the mean (NOTE: numerically unstable if mean >> stdev)
  xx = (xx - x**2/Nx) / (Nx - 1)
  yy = (yy - y**2/Ny) / (Ny - 1)
  x /= Nx
  y /= Ny

  if Nx_fac is not None: eNx = Nx/Nx_fac
  else: eNx = Nx
  if Ny_fac is not None: eNy = Ny/Ny_fac
  else: eNy = Ny
  print 'eff. Nx = %.1f, eff. Ny = %.1f' % (eNx, eNy)

  d = x - y
  den = np.sqrt(xx/eNx + yy/eNy)
  df = (xx/eNx + yy/eNy)**2 / ((xx/eNx)**2/(eNx - 1) + (yy/eNy)**2/(eNy - 1))

  p = tdist.cdf(abs(d/den), df)*np.sign(d)
  ci = tdist.ppf(1. - alpha/2, df) * den

  xn = X.name if X.name != '' else 'X'
  yn = Y.name if Y.name != '' else 'Y'
  if xn == yn: name = xn
  else: name = '%s-%s'%(xn, yn)

  if len(oaxes) > 0:
    from pygeode import Var, Dataset
    D = Var(oaxes, values=d, name=name)
    DF = Var(oaxes, values=df, name='df_%s' % name)
    P = Var(oaxes, values=p, name='p_%s' % name)
    CI = Var(oaxes, values=ci, name='CI_%s' % name)
    return Dataset([D, DF, P, CI])
  else: # Degenerate case
    return d, df, p, ci
# }}}

def isnonzero(X, axes, alpha=0.05, N_fac = None, pbar=None):
# {{{
  ''' isnonzero(X) - determins if X is non-zero, assuming X is normally distributed.
      Returns mean of X along axes, p value, and confidence interval.'''

  from pygeode.tools import combine_axes, whichaxis, loopover, npsum, npnansum
  from pygeode.view import View

  riaxes = [X.whichaxis(n) for n in axes]
  raxes = [a for i, a in enumerate(X.axes) if i in riaxes]
  oaxes = [a for i, a in enumerate(X.axes) if i not in riaxes]
  oview = View(oaxes) 

  N = np.product([len(X.axes[i]) for i in riaxes])

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  assert N > 1, '%s has only one element along the reduction axes' % X.name

  # Construct work arrays
  x = np.zeros(oview.shape, 'd')*np.nan
  xx = np.zeros(oview.shape, 'd')*np.nan
  Na = np.zeros(oview.shape, 'd')*np.nan

  # Accumulate data
  for outsl, (xdata,) in loopover([X], oview, pbar=pbar):
    xdata = xdata.astype('d')
    x[outsl] = np.nansum([x[outsl], npnansum(xdata, riaxes)], 0)
    xx[outsl] = np.nansum([xx[outsl], npnansum(xdata**2, riaxes)], 0)
    # Sum of weights (kludge to get masking right)
    Na[outsl] = np.nansum([Na[outsl], npnansum(1. + xdata*0., riaxes)], 0) 

  # remove the mean (NOTE: numerically unstable if mean >> stdev)
  xx = (xx - x**2/Na) / (Na - 1)
  x /= Na

  if N_fac is not None: 
    eN = N/N_fac
    eNa = Na/N_fac
  else: 
    eN = N
    eNa = Na
  print 'eff. N = %.1f' % eN

  sdom = np.sqrt(xx/eNa)

  p = tdist.cdf(abs(x/sdom), eNa - 1)*np.sign(x)
  ci = tdist.ppf(1. - alpha/2, eNa - 1) * sdom

  name = X.name if X.name != '' else 'X'

  if len(oaxes) > 0:
    from pygeode import Var, Dataset
    X = Var(oaxes, values=x, name=name)
    P = Var(oaxes, values=p, name='p_%s' % name)
    CI = Var(oaxes, values=ci, name='CI_%s' % name)
    return Dataset([X, P, CI])
  else: # Degenerate case
    return x, p, ci
# }}}

"""
def regress_old(y, xs, resid=False):
# {{{
   ''' regress(y, xs) - performs a multiple-linear regression of the variable y against
         the predictors xs. returns the coefficients of the fit and the residuals. 
         xs must be a single-dimensioned variable.'''
   from np import linalg as la

   xy = [(y * x).sum(x.axes[0].__class__) for x in xs]
   xx = np.array([[(xi * xj).sum() for xi in xs] for xj in xs])

   xxi = la.inv(xx)
   
   beta = numpy.dot(xxi, xy)
   if resid: return beta, y - sum([b*x for b,x in zip(beta, xs)])
   else: return beta
# }}}

# Linear trend
class Trend (ReducedVar):
  def __new__ (cls, var):
    from pygeode.timeaxis import Time
    return ReducedVar.__new__(cls, var, [Time])
  def __init__ (self, var):
    from pygeode.timeaxis import Time
    ReducedVar.__init__(self, var, [Time])
  def getview (self, view, pbar):
    import numpy as np
    from pygeode.timeaxis import Time
    ti = self.var.whichaxis(Time)
    X = np.zeros(view.shape, self.dtype)
    XX = np.zeros(view.shape, self.dtype)
    F = np.zeros(view.shape, self.dtype)
    XF = np.zeros(view.shape, self.dtype)
    for outsl, [fdata,xdata] in loopover([self.var,self.var.axes[ti]], view, inaxes=self.var.axes, pbar=pbar):
      X[outsl] += npnansum(xdata, self.indices)
      XX[outsl] += npnansum(xdata**2, self.indices)
      F[outsl] += npnansum(fdata, self.indices)
      XF[outsl] += npnansum(fdata*xdata, self.indices)
    N = self.N
    out = ( XF/N - X*F/N**2 ) / ( XX/N - X**2/N**2 )
    assert out.shape == view.shape, "%s != %s"%(out.shape, view.shape)
    return out

def trend (var): return Trend (var)

# Construct a var from a trend
# (almost the reverse of the above trend function, except for a possible offset if the mean is not zero)
class From_Trend (Var):
  def __init__ (self, var, taxis):
    from pygeode.var import Var
    self.trend = var
    self.taxis = taxis
    axes = [taxis] + list(var.axes)
    Var.__init__(self, axes, dtype=var.dtype)
  def getview (self, view, pbar):
    import numpy as np
    taxis = self.taxis
    X = view.get(taxis)
    X = X - np.nansum(taxis.values, 0)/len(taxis)
    trend = view.get(self.trend, pbar=pbar)
    return trend * X

def from_trend (var, taxis): return From_Trend (var, taxis)
"""
