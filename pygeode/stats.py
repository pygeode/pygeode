import numpy as np
from scipy.stats import t as tdist

sigs = (-1.1, -0.05, -0.01, 0., 0.01, 0.05, 1.1)
sigs_c = (  (1.0, 1.0, 1.0),
            (0.9, 0.9, 1.0), 
            (0.6, 0.6, 0.9), 
            (0.9, 0.6, 0.6), 
            (1.0, 0.9, 0.9), 
            (1.0, 1.0, 1.0))

def correlate(X, Y, pbar=None):
# {{{
  ''' correlate(X, Y) - returns correlation between variables X and Y
      computed over all axes shared by x and y. Returns \rho_xy, and p values
      for \rho_xy assuming x and y are normally distributed as Storch and Zwiers 1999
      section 8.2.3.'''

  from pygeode.tools import combine_axes, shared_axes, npsum
  from pygeode.view import View
  from pygeode.reduce import loopover

  # Put all the axes being reduced over at the end 
  # so that we can reshape 
  srcaxes = combine_axes([X, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [X.axes, Y.axes])
  oaxes = [srcaxes[i] for i in oiaxes]
  inaxes = oaxes + [srcaxes[i] for i in riaxes]
  oview = View(oaxes) 
  siaxes = range(len(oaxes), len(srcaxes))

  if len(oaxes) == 0:
    raise ValueException('%s and %s share no common axes to be correlated over' % (X.name, Y.name))

  # Construct work arrays
  x = np.zeros(oview.shape, 'd')
  y = np.zeros(oview.shape, 'd')
  xx = np.zeros(oview.shape, 'd')
  xy = np.zeros(oview.shape, 'd')
  yy = np.zeros(oview.shape, 'd')

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  # Accumulate 1st and 2nd moments
  for outsl, (xdata, ydata) in loopover([X, Y], oview, inaxes, pbar):
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

  # Compute correlation coefficient, t-statistic, p-value
  rho = xy/np.sqrt(xx*yy)
  t = np.abs(rho) * np.sqrt((N - 2.)/(1 - rho**2))
  p = tdist.cdf(t, N-2) * np.sign(rho)

  # Construct and return variables
  xn = X.name if X.name != '' else 'X' # Note: could write:  xn = X.name or 'X'
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  Rho = Var(oaxes, values=rho, name='C(%s, %s)' % (xn, yn))
  P = Var(oaxes, values=p, name='P(C(%s,%s) != 0)' % (xn, yn))
  return Rho, P
# }}}

def regress(X, Y, pbar=None):
# {{{
  ''' regress(X, Y) - returns correlation between variables X and Y
      computed over all axes shared by x and y. Returns \rho_xy, and p values
      for \rho_xy assuming x and y are normally distributed as Storch and Zwiers 1999
      section 8.2.3.'''
  from pygeode.tools import combine_axes, shared_axes, npsum
  from pygeode.view import View
  from pygeode.reduce import loopover

  srcaxes = combine_axes([X, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [X.axes, Y.axes])
  oaxes = [srcaxes[i] for i in oiaxes]
  inaxes = oaxes + [srcaxes[i] for i in riaxes]
  oview = View(oaxes) 
  siaxes = range(len(oaxes), len(srcaxes))

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  if len(oaxes) == 0:
    raise ValueException('%s and %s share no common axes to be regressed over' % (X.name, Y.name))

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
