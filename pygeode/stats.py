''' Tools for computing some basic statistical quantities. '''

__all__ = ('correlate', 'regress', 'multiple_regress', 'difference', 'paired_difference', 'isnonzero')

import numpy as np
from scipy.stats import norm, t as tdist

def correlate(X, Y, axes=None, output = 'r2,p', pbar=None):
# {{{
  r'''Computes correlation between variables X and Y.

  Parameters
  ==========
  X, Y : :class:`Var`
    Variables to correlate. Must have at least one axis in common.

  axes : list, optional
    Axes over which to compute correlation; if nothing is specified, the correlation
    is computed over all axes common to  shared by X and Y.

  output : string, optional
    A string determining which parameters are returned; see list of possible outputs
    in the Returns section. The specifications must be separated by a comma. Defaults
    to 'r2,p'.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : :class:`Dataset` 
    The names of the variables match the output request string (i.e. if ``ds``
    is the returned dataset, the correlation coefficient can be obtained
    through ``ds.r2``).

    * 'r2': The correlation coefficient :math:`\rho_{XY}`
    * 'p':  The p-value; see notes.

  Notes
  =====
  The coefficient :math:`\rho_{XY}` is computed following von Storch and Zwiers
  1999, section 8.2.2. The p-value is the probability of finding a correlation
  coeefficient of equal or greater magnitude (two-sided) to the given result
  under the hypothesis that the true correlation coefficient between X and Y is
  zero. It is computed from the t-statistic given in eq (8.7), in section
  8.2.3, and assumes normally distributed quantities.'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npnansum
  from pygeode.view import View

  # Split output request now
  ovars = ['r2', 'p']
  output = [o for o in output.split(',') if o in ovars]
  if len(output) < 1: raise ValueError('No valid outputs are requested from correlation. Possible outputs are %s.' % str(ovars))

  # Put all the axes being reduced over at the end 
  # so that we can reshape 
  srcaxes = combine_axes([X, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [X.axes, Y.axes])
  if axes is not None:
    ri_new = []
    for a in axes:
      i = whichaxis(srcaxes, a)
      if i not in riaxes: 
        raise KeyError('%s axis not shared by X ("%s") and Y ("%s")' % (a, X.name, Y.name))
      ri_new.append(i)
    oiaxes.extend([r for r in riaxes if r not in ri_new])
    riaxes = ri_new
    
  oaxes = [srcaxes[i] for i in oiaxes]
  inaxes = oaxes + [srcaxes[i] for i in riaxes]
  oview = View(oaxes) 
  iview = View(inaxes) 
  siaxes = list(range(len(oaxes), len(srcaxes)))

  # Construct work arrays
  x  = np.full(oview.shape, np.nan, 'd')
  y  = np.full(oview.shape, np.nan, 'd')
  xx = np.full(oview.shape, np.nan, 'd')
  yy = np.full(oview.shape, np.nan, 'd')
  xy = np.full(oview.shape, np.nan, 'd')
  Na = np.full(oview.shape, np.nan, 'd')

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  for outsl, (xdata, ydata) in loopover([X, Y], oview, inaxes, pbar=pbar):
    xdata = xdata.astype('d')
    ydata = ydata.astype('d')
    xydata = xdata*ydata

    xbc = [s1 // s2 for s1, s2 in zip(xydata.shape, xdata.shape)]
    ybc = [s1 // s2 for s1, s2 in zip(xydata.shape, ydata.shape)]
    xdata = np.tile(xdata, xbc)
    ydata = np.tile(ydata, ybc)
    xdata[np.isnan(xydata)] = np.nan
    ydata[np.isnan(xydata)] = np.nan

    # It seems np.nansum does not broadcast its arguments automatically
    # so there must be a better way of doing this...
    x[outsl]  = np.nansum([x[outsl],  npnansum(xdata, siaxes)], 0)
    y[outsl]  = np.nansum([y[outsl],  npnansum(ydata, siaxes)], 0)
    xx[outsl] = np.nansum([xx[outsl], npnansum(xdata**2, siaxes)], 0)
    yy[outsl] = np.nansum([yy[outsl], npnansum(ydata**2, siaxes)], 0)
    xy[outsl] = np.nansum([xy[outsl], npnansum(xydata, siaxes)], 0)

    # Count of non-NaN data points
    Na[outsl] = np.nansum([Na[outsl], npnansum(~np.isnan(xydata), siaxes)], 0)

  imsk = (Na > 0)

  xx[imsk] -= (x*x)[imsk]/Na[imsk]
  yy[imsk] -= (y*y)[imsk]/Na[imsk]
  xy[imsk] -= (x*y)[imsk]/Na[imsk]

  # Ensure variances are non-negative
  xx[xx <= 0.] = 0.
  yy[yy <= 0.] = 0.

  # Compute correlation coefficient, t-statistic, p-value
  den = np.zeros(oview.shape, 'd')
  rho = np.zeros(oview.shape, 'd')

  den[imsk] = np.sqrt((xx*yy)[imsk])
  dmsk = (den > 0.)

  rho[dmsk] = xy[dmsk] / np.sqrt(xx*yy)[dmsk]

  den = 1 - rho**2
  # Saturate the denominator (when correlation is perfect) to avoid div by zero warnings
  den[den < eps] = eps

  t = np.zeros(oview.shape, 'd')
  p = np.zeros(oview.shape, 'd')

  t[imsk] = np.abs(rho)[imsk] * np.sqrt((Na[imsk] - 2.)/den[imsk])
  p[imsk] = 2. * (1. - tdist.cdf(t[imsk], Na[imsk] - 2))

  p[~imsk] = np.nan
  rho[~imsk] = np.nan

  p[~dmsk] = np.nan
  rho[~dmsk] = np.nan

  # Construct and return variables
  xn = X.name if X.name != '' else 'X' # Note: could write:  xn = X.name or 'X'
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  from pygeode.dataset import asdataset

  rvs = []

  if 'r2' in output:
    r2 = Var(oaxes, values=rho, name='r2')
    r2.atts['longname'] = 'Correlation coefficient between %s and %s' % (xn, yn)
    rvs.append(r2)

  if 'p' in output:
    p = Var(oaxes, values=p, name='p')
    p.atts['longname'] = 'p-value for correlation coefficient between %s and %s' % (xn, yn)
    rvs.append(p)

  ds = asdataset(rvs)
  ds.atts['description'] = 'correlation analysis %s against %s' % (yn, xn)

  return ds
# }}}

def regress(X, Y, axes=None, N_fac=None, output='m,b,p', pbar=None):
# {{{
  r'''Computes least-squares linear regression of Y against X.

  Parameters
  ==========
  X, Y : :class:`Var`
    Variables to regress. Must have at least one axis in common.

  axes : list, optional
    Axes over which to compute correlation; if nothing is specified, the correlation
    is computed over all axes common to X and Y.

  N_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom; the effective
    number will be given by the number estimated from the dataset divided by ``N_fac``.

  output : string, optional
    A string determining which parameters are returned; see list of possible outputs
    in the Returns section. The specifications must be separated by a comma. Defaults 
    to 'm,b,p'.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : :class:`Dataset` 
    The returned variables are specified by the ``output`` argument. The names of the 
    variables match the output request string (i.e. if ``ds`` is the returned dataset, the 
    linear coefficient of the regression can be obtained by ``ds.m``). 
    
    A fit of the form :math:`Y = m X + b + \epsilon` is assumed, and the
    following parameters can be returned:

    * 'm': Linear coefficient of the regression
    * 'b': Constant coefficient of the regression
    * 'r2': Fraction of the variance in Y explained by X (:math:`R^2`)
    * 'p': p-value of regression; see notes.
    * 'sm': Standard deviation of linear coefficient estimate
    * 'se': Standard deviation of residuals

  Notes
  =====
  The statistics described are computed following von Storch and Zwiers 1999,
  section 8.3. The p-value 'p' is computed using the t-statistic given in
  section 8.3.8, and confidence intervals for the slope and intercept can be
  computed from 'se' and 'se' (:math:`\hat{\sigma}_E` and
  :math:`\hat{\sigma}_E/\sqrt{S_{XX}}` in von Storch and Zwiers, respectively).
  The data is assumed to be normally distributed.'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npnansum
  from pygeode.view import View

  # Split output request now
  ovars = ['m', 'b', 'r2', 'p', 'sm', 'se']
  output = [o for o in output.split(',') if o in ovars]
  if len(output) < 1: raise ValueError('No valid outputs are requested from regression. Possible outputs are %s.' % str(ovars))

  srcaxes = combine_axes([X, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [X.axes, Y.axes])
  if axes is not None:
    ri_new = []
    for a in axes:
      i = whichaxis(srcaxes, a)
      if i not in riaxes: 
        raise KeyError('%s axis not shared by X ("%s") and Y ("%s")' % (a, X.name, Y.name))
      ri_new.append(i)
    oiaxes.extend([r for r in riaxes if r not in ri_new])
    riaxes = ri_new
    
  oaxes = [srcaxes[i] for i in oiaxes]
  inaxes = oaxes + [srcaxes[i] for i in riaxes]
  oview = View(oaxes) 
  siaxes = list(range(len(oaxes), len(srcaxes)))

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  assert len(riaxes) > 0, '%s and %s share no axes to be regressed over' % (X.name, Y.name)

  # Construct work arrays
  x  = np.full(oview.shape, np.nan, 'd')
  y  = np.full(oview.shape, np.nan, 'd')
  xx = np.full(oview.shape, np.nan, 'd')
  yy = np.full(oview.shape, np.nan, 'd')
  xy = np.full(oview.shape, np.nan, 'd')
  Na = np.full(oview.shape, np.nan, 'd')

  # Accumulate data
  for outsl, (xdata, ydata) in loopover([X, Y], oview, inaxes, pbar=pbar):
    xdata = xdata.astype('d')
    ydata = ydata.astype('d')
    xydata = xdata*ydata

    xbc = [s1 // s2 for s1, s2 in zip(xydata.shape, xdata.shape)]
    ybc = [s1 // s2 for s1, s2 in zip(xydata.shape, ydata.shape)]
    xdata = np.tile(xdata, xbc)
    ydata = np.tile(ydata, ybc)
    xdata[np.isnan(xydata)] = np.nan
    ydata[np.isnan(xydata)] = np.nan

    # It seems np.nansum does not broadcast its arguments automatically
    # so there must be a better way of doing this...
    x[outsl]  = np.nansum([x[outsl],  npnansum(xdata, siaxes)], 0)
    y[outsl]  = np.nansum([y[outsl],  npnansum(ydata, siaxes)], 0)
    xx[outsl] = np.nansum([xx[outsl], npnansum(xdata**2, siaxes)], 0)
    yy[outsl] = np.nansum([yy[outsl], npnansum(ydata**2, siaxes)], 0)
    xy[outsl] = np.nansum([xy[outsl], npnansum(xydata, siaxes)], 0)

    # Sum of weights
    Na[outsl] = np.nansum([Na[outsl], npnansum(~np.isnan(xydata), siaxes)], 0)

  if N_fac is None:
    N_eff = Na - 2.
  else:
    N_eff = Na / N_fac - 2.

  nmsk = (N_eff > 0.)

  xx[nmsk] -= (x*x)[nmsk]/Na[nmsk]
  yy[nmsk] -= (y*y)[nmsk]/Na[nmsk]
  xy[nmsk] -= (x*y)[nmsk]/Na[nmsk]

  dmsk = (xx > 0.)

  m  = np.zeros(oview.shape, 'd')
  b  = np.zeros(oview.shape, 'd')
  r2 = np.zeros(oview.shape, 'd')

  m[dmsk] = xy[dmsk]/xx[dmsk]
  b[nmsk] = (y[nmsk] - m[nmsk]*x[nmsk]) / Na[nmsk]

  r2den = xx * yy
  d2msk = (r2den > 0.)

  r2[d2msk] = xy[d2msk]**2 / r2den[d2msk]

  sige = np.zeros(oview.shape, 'd')
  sigm = np.zeros(oview.shape, 'd')
  t = np.zeros(oview.shape, 'd')
  p = np.zeros(oview.shape, 'd')

  sige[nmsk] = (yy[nmsk] - m[nmsk] * xy[nmsk]) / N_eff[nmsk]
  sigm[dmsk] = np.sqrt(sige[dmsk] / xx[dmsk])
  sige[nmsk] = np.sqrt(sige[dmsk])
  t[dmsk] = np.abs(m[dmsk]) / sigm[dmsk]
  p[nmsk] = 2. * (1. - tdist.cdf(t[nmsk], N_eff[nmsk]))

  msk = nmsk & dmsk
   
  m[~msk] = np.nan
  b[~msk] = np.nan
  sige[~msk] = np.nan
  sigm[~msk] = np.nan
  p[~msk] = np.nan

  msk = nmsk & d2msk
  r2[~msk] = np.nan

  xn = X.name if X.name != '' else 'X'
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  from pygeode.dataset import asdataset
  
  rvs = []

  if 'm' in output:
    M = Var(oaxes, values=m, name='m')
    M.atts['longname'] = 'slope'
    rvs.append(M)

  if 'b' in output:
    B = Var(oaxes, values=b, name='b')
    B.atts['longname'] = 'intercept'
    rvs.append(B)

  if 'r2' in output:
    R2 = Var(oaxes, values=r2, name='r2')
    R2.atts['longname'] = 'fraction of variance explained'
    rvs.append(R2)

  if 'p' in output:
    P = Var(oaxes, values=p, name='p')
    P.atts['longname'] = 'p-value'
    rvs.append(P)

  if 'sm' in output:
    SM = Var(oaxes, values=sigm, name='sm')
    SM.atts['longname'] = 'standard deviation of slope parameter'
    rvs.append(SM)

  if 'se' in output:
    SE = Var(oaxes, values=sige, name='se')
    SE.atts['longname'] = 'standard deviation of residual'
    rvs.append(SE)

  ds = asdataset(rvs)
  ds.atts['description'] = 'linear regression parameters for %s regressed against %s' % (yn, xn)

  return ds
# }}}

def multiple_regress(Xs, Y, axes=None, N_fac=None, output='B,p', pbar=None):
# {{{
  r'''Computes least-squares multiple regression of Y against variables Xs.

  Parameters
  ==========
  Xs : list of :class:`Var` instances
    Variables to treat as independent regressors. Must have at least one axis
    in common with each other and with Y.

  Y : :class:`Var`
    The dependent variable. Must have at least one axis in common with the Xs.

  axes : list, optional
    Axes over which to compute correlation; if nothing is specified, the correlation
    is computed over all axes common to the Xs and Y.

  N_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom; the effective
    number will be given by the number estimated from the dataset divided by ``N_fac``.

  output : string, optional
    A string determining which parameters are returned; see list of possible outputs
    in the Returns section. The specifications must be separated by a comma. Defaults 
    to 'B,p'.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : tuple of floats or :class:`Var` instances.
    The return values are specified by the ``output`` argument. The names of the 
    variables match the output request string (i.e. if ``ds`` is the returned dataset, the 
    linear coefficient of the regression can be obtained by ``ds.m``). 
    
    A fit of the form :math:`Y = \sum_i \beta_i X_i + \epsilon` is assumed.
    Note that a constant term is not included by default. The following
    parameters can be returned:

    * 'B': Linear coefficients :math:`\beta_i` of each regressor
    * 'r2': Fraction of the variance in Y explained by all Xs (:math:`R^2`)
    * 'p': p-value of regession; see notes.
    * 'sb': Standard deviation of each linear coefficient
    * 'covb': Covariance matrix of the linear coefficients
    * 'se': Standard deviation of residuals

    The outputs 'B', 'p', and 'sb' will produce as many outputs as there are
    regressors. 

  Notes
  =====
  The statistics described are computed following von Storch and Zwiers 1999,
  section 8.4. The p-value 'p' is computed using the t-statistic appropriate
  for the multi-variate normal estimator :math:`\hat{\vec{a}}` given in section
  8.4.2; it corresponds to the probability of obtaining the regression
  coefficient under the null hypothesis that there is no linear relationship.
  Note this may not be the best way to determine if a given parameter is
  contributing a significant fraction to the explained variance of Y.  The
  variances 'se' and 'sb' are :math:`\hat{\sigma}_E` and the square root of the
  diagonal elements of :math:`\hat{\sigma}^2_E (\chi^T\chi)` in von Storch and
  Zwiers, respectively.  The data is assumed to be normally distributed.'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npsum
  from pygeode.view import View

  # Split output request now
  ovars = ['beta', 'r2', 'p', 'sb', 'covb', 'se']
  output = [o for o in output.split(',') if o in ovars]
  if len(output) < 1: raise ValueError('No valid outputs are requested from correlation. Possible outputs are %s.' % str(ovars))

  Nr = len(Xs)

  Xaxes = combine_axes(Xs)

  srcaxes = combine_axes([Xaxes, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [Xaxes, Y.axes])
  if axes is not None:
    ri_new = []
    for a in axes:
      ia = whichaxis(srcaxes, a)
      if ia in riaxes: ri_new.append(ia)
      else: raise KeyError('One of the Xs or Y does not have the axis %s.' % a)
    oiaxes.extend([r for r in riaxes if r not in ri_new])
    riaxes = ri_new
    
  oaxes = tuple([srcaxes[i] for i in oiaxes])
  inaxes = oaxes + tuple([srcaxes[i] for i in riaxes])
  oview = View(oaxes) 
  siaxes = list(range(len(oaxes), len(srcaxes)))

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  assert len(riaxes) > 0, 'Regressors and %s share no axes to be regressed over' % (Y.name)

  # Construct work arrays
  os = oview.shape
  os1 = os + (Nr,)
  os2 = os + (Nr,Nr)
  y  = np.zeros(os, 'd')
  yy = np.zeros(os, 'd')
  xy = np.zeros(os1, 'd')
  xx = np.zeros(os2, 'd')
  xxinv = np.zeros(os2, 'd')

  N = np.prod([len(srcaxes[i]) for i in riaxes])

  # Accumulate data
  for outsl, datatuple in loopover(Xs + [Y], oview, inaxes, pbar=pbar):
    ydata =  datatuple[-1].astype('d')
    xdata = [datatuple[ i].astype('d') for i in range(Nr)]
    y[outsl]  += npsum(ydata,    siaxes)
    yy[outsl] += npsum(ydata**2, siaxes)
    for i in range(Nr):
      xy[outsl+(i,)] += npsum(xdata[i]*ydata, siaxes)
      for j in range(i+1):
        xx[outsl+(i,j)] += npsum(xdata[i]*xdata[j], siaxes)

  # Fill in opposite side of xTx
  for i in range(Nr):
    for j in range(i):
      xx[..., j, i] = xx[..., i, j]

  # Compute inverse of covariance matrix (could be done more intellegently? certainly the python
  # loop over oview does not help)
  xx    = xx.reshape(-1, Nr, Nr)
  xxinv = xxinv.reshape(-1, Nr, Nr)
  for i in range(xx.shape[0]):
    xxinv[i,:,:] = np.linalg.inv(xx[i,:,:])
  xx = xx.reshape(os2)
  xxinv = xxinv.reshape(os2)

  beta = np.sum(xy.reshape(os + (1, Nr)) * xxinv, -1)
  vare = np.sum(xy * beta, -1)

  if N_fac is None: N_eff = N
  else: N_eff = N // N_fac

  sigbeta = [np.sqrt((yy - vare) * xxinv[..., i, i] / N_eff) for i in range(Nr)]

  xns = [X.name if X.name != '' else 'X%d' % i for i, X in enumerate(Xs)]
  yn = Y.name if Y.name != '' else 'Y'

  from .var import Var
  from .dataset import asdataset
  from .axis import NonCoordinateAxis

  ra  = NonCoordinateAxis(values=np.arange(Nr), regressor = xns, name = 'regressor')
  ra2 = NonCoordinateAxis(values=np.arange(Nr), regressor = xns, name = 'regressor2')
  Nd = len(oaxes)

  rvs = []

  if 'beta' in output:
    B = Var(oaxes + (ra,), values=beta, name='beta')
    B.atts['longname'] = 'regression coefficient'
    rvs.append(B)

  if 'r2' in output:
    vary = (yy - y**2/N)
    R2 = 1 - (yy - vare) / vary
    R2 = Var(oaxes, values=R2, name='R2')
    R2.atts['longname'] = 'fraction of variance explained'
    rvs.append(R2)

  if 'p' in output:
    p = [2. * (1. - tdist.cdf(np.abs(beta[...,i]/sigbeta[i]), N_eff-Nr)) for i in range(Nr)]
    p = np.transpose(np.array(p), [Nd] + list(range(Nd)))
    p = Var(oaxes + (ra,), values=p, name='p')
    p.atts['longname'] = 'p-values'
    rvs.append(p)

  if 'sb' in output:
    sigbeta = np.transpose(np.array(sigbeta), [Nd] + list(range(Nd)))
    sb = Var(oaxes + (ra,), values=sigbeta, name='sb')
    sb.atts['longname'] = 'standard deviation of linear coefficients'
    rvs.append(sb)

  if 'covb' in output:
    sigmat = np.zeros(os2, 'd')
    for i in range(Nr):
      for j in range(Nr):
        #sigmat[..., i, j] = np.sqrt((yy - vare) * xxinv[..., i, j] / N_eff)
        sigmat[..., i, j] = (yy - vare) * xxinv[..., i, j] / N_eff
    covb = Var(oaxes + (ra, ra2), values=sigmat, name='covb')
    covb.atts['longname'] = 'Covariance matrix of the linear coefficients'
    rvs.append(covb)

  if 'se' in output:
    se = np.sqrt((yy - vare) / N_eff)
    se = Var(oaxes, values=se, name='se')
    se.atts['longname'] = 'standard deviation of residual'
    rvs.append(se)

  ds = asdataset(rvs)
  ds.atts['description'] = 'multiple linear regression parameters for %s regressed against %s' % (yn, xns)

  return ds
# }}}

def difference(X, Y, axes=None, alpha=0.05, Nx_fac = None, Ny_fac = None, output='d,p,ci', pbar=None):
# {{{
  r'''Computes the mean value and statistics of X - Y.

  Parameters
  ==========
  X, Y : :class:`Var`
    Variables to difference. Must have at least one axis in common.

  axes : list, optional, defaults to None
    Axes over which to compute means; if othing is specified, the mean
    is computed over all axes common to X and Y.

  alpha : float, optional; defaults to 0.05
    Confidence level for which to compute confidence interval.

  Nx_fac : integer, optional: defaults to None
    A factor by which to rescale the estimated number of degrees of freedom of
    X; the effective number will be given by the number estimated from the
    dataset divided by ``Nx_fac``.

  Ny_fac : integer, optional: defaults to None
    A factor by which to rescale the estimated number of degrees of freedom of
    Y; the effective number will be given by the number estimated from the
    dataset divided by ``Ny_fac``.

  output : string, optional
    A string determining which parameters are returned; see list of possible outputs
    in the Returns section. The specifications must be separated by a comma. Defaults 
    to 'd,p,ci'.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : :class:`Dataset` 
    The returned variables are specified by the ``output`` argument. The names
    of the variables match the output request string (i.e. if ``ds`` is the
    returned dataset, the average of the difference can be obtained by
    ``ds.d``). The following four quantities can be computed:

    * 'd': The difference in the means, X - Y
    * 'df': The effective number of degrees of freedom, :math:`df`
    * 'p': The p-value; see notes.
    * 'ci': The confidence interval of the difference at the level specified by ``alpha``

  See Also
  ========
  isnonzero
  paired_difference

  Notes
  =====
  The effective number of degrees of freedom is estimated using eq (6.20) of 
  von Storch and Zwiers 1999, in which :math:`n_X` and :math:`n_Y` are scaled by
  Nx_fac and Ny_fac, respectively. This provides a means of taking into account
  serial correlation in the data (see sections 6.6.7-9), but the number of effective
  degrees of freedom are not calculated explicitly by this routine. The p-value and 
  confidence interval are computed based on the t-statistic in eq (6.19).'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npnansum
  from pygeode.view import View

  # Split output request now
  ovars = ['d', 'df', 'p', 'ci']
  output = [o for o in output.split(',') if o in ovars]
  if len(output) < 1: raise ValueError('No valid outputs are requested from correlation. Possible outputs are %s.' % str(ovars))

  srcaxes = combine_axes([X, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [X.axes, Y.axes])
  if axes is not None:
    ri_new = []
    for a in axes:
      i = whichaxis(srcaxes, a)
      if i not in riaxes: 
        raise KeyError('%s axis not shared by X ("%s") and Y ("%s")' % (a, X.name, Y.name))
      ri_new.append(i)
    oiaxes.extend([r for r in riaxes if r not in ri_new])
    riaxes = ri_new

  oaxes = [a for i, a in enumerate(srcaxes) if i not in riaxes]
  oview = View(oaxes) 

  ixaxes = [X.whichaxis(n) for n in axes if X.hasaxis(n)]
  iyaxes = [Y.whichaxis(n) for n in axes if Y.hasaxis(n)]

  Nx = np.product([len(X.axes[i]) for i in ixaxes])
  Ny = np.product([len(Y.axes[i]) for i in iyaxes])
  assert Nx > 1, '%s has only one element along the reduction axes' % X.name
  assert Ny > 1, '%s has only one element along the reduction axes' % Y.name

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  # Construct work arrays
  x  = np.full(oview.shape, np.nan, 'd')
  y  = np.full(oview.shape, np.nan, 'd')
  xx = np.full(oview.shape, np.nan, 'd')
  yy = np.full(oview.shape, np.nan, 'd')
  Nx = np.full(oview.shape, np.nan, 'd')
  Ny = np.full(oview.shape, np.nan, 'd')

  # Accumulate data
  for outsl, (xdata,) in loopover([X], oview, pbar=pbar):
    xdata = xdata.astype('d')
    x[outsl]  = np.nansum([x[outsl],  npnansum(xdata, ixaxes)], 0)
    xx[outsl] = np.nansum([xx[outsl], npnansum(xdata**2, ixaxes)], 0)

    # Count of non-NaN data points
    Nx[outsl] = np.nansum([Nx[outsl], npnansum(~np.isnan(xdata), ixaxes)], 0) 

  for outsl, (ydata,) in loopover([Y], oview, pbar=pbar):
    ydata = ydata.astype('d')
    y[outsl]  = np.nansum([y[outsl],  npnansum(ydata, iyaxes)], 0)
    yy[outsl] = np.nansum([yy[outsl], npnansum(ydata**2, iyaxes)], 0)

    # Count of non-NaN data points
    Ny[outsl] = np.nansum([Ny[outsl], npnansum(~np.isnan(ydata), iyaxes)], 0) 

  # remove the mean (NOTE: numerically unstable if mean >> stdev)
  imsk = (Nx > 1) & (Ny > 1)
  xx[imsk] -= (x*x)[imsk] / Nx[imsk]
  xx[imsk] /= (Nx[imsk] - 1)

  x[imsk]  /= Nx[imsk]

  yy[imsk] -= (y*y)[imsk] / Ny[imsk]
  yy[imsk] /= (Ny[imsk] - 1)

  y[imsk]  /= Ny[imsk]

  # Ensure variances are non-negative
  xx[xx <= 0.] = 0.
  yy[yy <= 0.] = 0.

  if Nx_fac is not None: eNx = Nx//Nx_fac
  else: eNx = Nx
  if Ny_fac is not None: eNy = Ny//Ny_fac
  else: eNy = Ny

  emsk = (eNx > 1) & (eNy > 1)

  # Compute difference
  d = x - y

  den = np.zeros(oview.shape, 'd')
  df  = np.zeros(oview.shape, 'd')
  p   = np.zeros(oview.shape, 'd')
  ci  = np.zeros(oview.shape, 'd')

  # Convert to variance of the mean of each sample
  xx[emsk] /= eNx[emsk]
  yy[emsk] /= eNy[emsk]

  den[emsk] = xx[emsk]**2/(eNx[emsk] - 1) + yy[emsk]**2/(eNy[emsk] - 1)
  dmsk = (den > 0.)

  df[dmsk] = (xx[dmsk] + yy[dmsk])**2 / den[dmsk]

  den[emsk] = np.sqrt(xx[emsk] + yy[emsk])

  dmsk &= (den > 0.)

  p[dmsk] = np.abs(d[dmsk]/den[dmsk])
  p[dmsk] = 2. * (1. - tdist.cdf(p[dmsk], df[dmsk]))

  ci[dmsk] = tdist.ppf(1. - alpha/2, df[dmsk]) * den[dmsk]

  df[~dmsk] = np.nan
  p [~dmsk] = np.nan
  ci[~dmsk] = np.nan

  # Construct dataset to return
  xn = X.name if X.name != '' else 'X'
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  from pygeode.dataset import asdataset

  rvs = []

  if 'd' in output:
    d = Var(oaxes, values = d, name = 'd')
    d.atts['longname'] = 'Difference (%s - %s)' % (xn, yn)
    rvs.append(d)

  if 'df' in output:
    df = Var(oaxes, values = df, name = 'df')
    df.atts['longname'] = 'Degrees of freedom used for t-test'
    rvs.append(df)

  if 'p' in output:
    p = Var(oaxes, values = p, name = 'p')
    p.atts['longname'] = 'p-value for t-test of difference (%s - %s)' % (xn, yn)
    rvs.append(p)

  if 'ci' in output:
    ci = Var(oaxes, values = ci, name = 'ci')
    ci.atts['longname'] = 'Confidence Interval (alpha = %.2f) of difference (%s - %s)' % (alpha, xn, yn)
    rvs.append(ci)

  ds = asdataset(rvs)
  ds.atts['alpha'] = alpha
  ds.atts['Nx_fac'] = Nx_fac
  ds.atts['Ny_fac'] = Ny_fac
  ds.atts['description'] = 't-test of difference (%s - %s)' % (yn, xn)

  return ds
# }}}

def paired_difference(X, Y, axes=None, alpha=0.05, N_fac = None, output='d,p,ci', pbar=None):
# {{{
  r'''Computes the mean value and statistics of X - Y, assuming that individual elements
  of X and Y can be directly paired. In contrast to :func:`difference`, X and Y must have the same
  shape.

  Parameters
  ==========
  X, Y : :class:`Var`
    Variables to difference. Must share all axes over which the means are being computed.

  axes : list, optional
    Axes over which to compute means; if nothing is specified, the mean
    is computed over all axes common to X and Y.

  alpha : float
    Confidence level for which to compute confidence interval.

  N_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom of
    X and Y; the effective number will be given by the number estimated from the
    dataset divided by ``N_fac``.

  output : string, optional
    A string determining which parameters are returned; see list of possible outputs
    in the Returns section. The specifications must be separated by a comma. Defaults 
    to 'd,p,ci'.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : :class:`Dataset` 
    The returned variables are specified by the ``output`` argument. The names
    of the variables match the output request string (i.e. if ``ds`` is the
    returned dataset, the average of the difference can be obtained by
    ``ds.d``). The following four quantities can be computed:

    * 'd': The difference in the means, X - Y
    * 'df': The effective number of degrees of freedom, :math:`df`
    * 'p': The p-value; see notes.
    * 'ci': The confidence interval of the difference at the level specified by ``alpha``

  See Also
  ========
  isnonzero
  difference

  Notes
  =====
  Following section 6.6.6 of von Storch and Zwiers 1999, a one-sample t test is used to test the
  hypothesis. The number of degrees of freedom is the sample size scaled by N_fac, less one. This
  provides a means of taking into account serial correlation in the data (see sections 6.6.7-9), but
  the appropriate number of effective degrees of freedom are not calculated explicitly by this
  routine. The p-value and confidence interval are computed based on the t-statistic in eq
  (6.21).'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npnansum
  from pygeode.view import View

  # Split output request now
  ovars = ['d', 'df', 'p', 'ci']
  output = [o for o in output.split(',') if o in ovars]
  if len(output) < 1: raise ValueError('No valid outputs are requested from correlation. Possible outputs are %s.' % str(ovars))

  srcaxes = combine_axes([X, Y])
  oiaxes, riaxes = shared_axes(srcaxes, [X.axes, Y.axes])
  if axes is not None:
    ri_new = []
    for a in axes:
      i = whichaxis(srcaxes, a)
      if i not in riaxes: 
        raise KeyError('%s axis not shared by X ("%s") and Y ("%s")' % (a, X.name, Y.name))
      ri_new.append(i)
    oiaxes.extend([r for r in riaxes if r not in ri_new])
    riaxes = ri_new

  oaxes = [a for i, a in enumerate(srcaxes) if i not in riaxes]
  oview = View(oaxes) 

  ixaxes = [X.whichaxis(n) for n in axes if X.hasaxis(n)]
  Nx = np.product([len(X.axes[i]) for i in ixaxes])

  iyaxes = [Y.whichaxis(n) for n in axes if Y.hasaxis(n)]
  Ny = np.product([len(Y.axes[i]) for i in iyaxes])

  assert ixaxes == iyaxes and Nx == Ny, 'For the paired difference test, X and Y must have the same size along the reduction axes.'
  
  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  assert Nx > 1, '%s has only one element along the reduction axes' % X.name
  assert Ny > 1, '%s has only one element along the reduction axes' % Y.name

  # Construct work arrays
  d  = np.full(oview.shape, np.nan, 'd')
  dd = np.full(oview.shape, np.nan, 'd')
  N  = np.full(oview.shape, np.nan, 'd')

  # Accumulate data
  for outsl, (xdata, ydata) in loopover([X, Y], oview, inaxes=srcaxes, pbar=pbar):
    ddata = xdata.astype('d') - ydata.astype('d')
    d[outsl]  = np.nansum([d[outsl],  npnansum(ddata, ixaxes)], 0)
    dd[outsl] = np.nansum([dd[outsl], npnansum(ddata**2, ixaxes)], 0)

    # Count of non-NaN data points
    N[outsl] = np.nansum([N[outsl], npnansum(~np.isnan(ddata), ixaxes)], 0) 

  # remove the mean (NOTE: numerically unstable if mean >> stdev)
  imsk = (N > 1)
  dd[imsk] -= (d*d)[imsk]/N[imsk]
  dd[imsk] /= (N[imsk] - 1)
  d[imsk]  /= N[imsk]

  # Ensure variance is non-negative
  dd[dd <= 0.] = 0.

  if N_fac is not None: eN = N//N_fac
  else: eN = N

  emsk = (eN > 1)

  den = np.zeros(oview.shape, 'd')
  p   = np.zeros(oview.shape, 'd')
  ci  = np.zeros(oview.shape, 'd')

  den = np.zeros(oview.shape, 'd')
  den[emsk] = np.sqrt(dd[emsk]/(eN[emsk] - 1))
  dmsk = (den > 0.)

  p[dmsk]  = np.abs(d[dmsk]/den[dmsk])
  p[dmsk]  = 2. * (1. - tdist.cdf(p[dmsk], eN[dmsk] - 1))
  ci[dmsk] = tdist.ppf(1. - alpha/2, eN[dmsk] - 1) * den[dmsk]

  # Construct dataset to return
  xn = X.name if X.name != '' else 'X'
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  from pygeode.dataset import asdataset

  rvs = []

  if 'd' in output:
    d = Var(oaxes, values = d, name = 'd')
    d.atts['longname'] = 'Difference (%s - %s)' % (xn, yn)
    rvs.append(d)

  if 'df' in output:
    df = Var(oaxes, values = eN - 1, name = 'df')
    df.atts['longname'] = 'Degrees of freedom used for t-test'
    rvs.append(df)

  if 'p' in output:
    p = Var(oaxes, values = p, name = 'p')
    p.atts['longname'] = 'p-value for t-test of paired difference (%s - %s)' % (xn, yn)
    rvs.append(p)

  if 'ci' in output:
    ci = Var(oaxes, values = ci, name = 'ci')
    ci.atts['longname'] = 'Confidence Interval (alpha = %.2f) of paired difference (%s - %s)' % (alpha, xn, yn)
    rvs.append(ci)

  ds = asdataset(rvs)
  ds.atts['alpha'] = alpha
  ds.atts['N_fac'] = N_fac
  ds.atts['description'] = 't-test of paired difference (%s - %s)' % (yn, xn)

  return ds
# }}}

def isnonzero(X, axes=None, alpha=0.05, N_fac = None, output='m,p', pbar=None):
# {{{
  r'''Computes the mean value of X and statistics relevant for a test against
  the hypothesis that it is 0.

  Parameters
  ==========
  X : :class:`Var`
    Variable to average.

  axes : list, optional
    Axes over which to compute the mean; if nothing is specified, the mean is
    computed over all axes.

  alpha : float
    Confidence level for which to compute confidence interval.

  N_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom;
    the effective number will be given by the number estimated from the dataset
    divided by ``N_fac``.

  output : string, optional
    A string determining which parameters are returned; see list of possible outputs
    in the Returns section. The specifications must be separated by a comma. Defaults 
    to 'm,p'.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : :class:`Dataset` 
    The names of the variables match the output request string (i.e. if ``ds``
    is the returned dataset, the mean value can be obtained through ``ds.m``).
    The following quantities can be calculated.

    * 'm': The mean value of X
    * 'p': The probability of the computed value if the population mean was zero
    * 'ci': The confidence interval of the mean at the level specified by alpha

    If the average is taken over all axes of X resulting in a scalar,
    the above values are returned as a tuple in the order given. If not, the
    results are provided as :class:`Var` objects in a dataset. 

  See Also
  ========
  difference

  Notes
  =====
  The number of effective degrees of freedom can be scaled as in :meth:`difference`. 
  The p-value and confidence interval are computed for the t-statistic defined in 
  eq (6.61) of von Storch and Zwiers 1999.'''

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
  x = np.zeros(oview.shape, 'd')
  xx = np.zeros(oview.shape, 'd')
  Na = np.zeros(oview.shape, 'd')

  x[()] = np.nan
  xx[()] = np.nan
  Na[()] = np.nan

  # Accumulate data
  for outsl, (xdata,) in loopover([X], oview, pbar=pbar):
    xdata = xdata.astype('d')
    x[outsl] = np.nansum([x[outsl], npnansum(xdata, riaxes)], 0)
    xx[outsl] = np.nansum([xx[outsl], npnansum(xdata**2, riaxes)], 0)

    # Sum of weights (kludge to get masking right)
    Na[outsl] = np.nansum([Na[outsl], npnansum(~np.isnan(xdata), riaxes)], 0)

  imsk = (Na > 0.)

  # remove the mean (NOTE: numerically unstable if mean >> stdev)
  xx[imsk] -= x[imsk]**2 / Na[imsk]
  xx[imsk] = xx[imsk] / (Na[imsk] - 1)

  x[imsk] /= Na[imsk]

  if N_fac is not None: 
    eN = N//N_fac
    eNa = Na//N_fac
  else: 
    eN = N
    eNa = Na

  sdom = np.zeros((oview.shape), 'd')
  p    = np.zeros((oview.shape), 'd')
  t    = np.zeros((oview.shape), 'd')
  ci   = np.zeros((oview.shape), 'd')

  sdom[imsk] = np.sqrt(xx[imsk] / eNa[imsk])
  dmsk = (sdom > 0.)

  t[dmsk] = np.abs(x[dmsk]) / sdom[dmsk]
  p[imsk]  = 2. * (1. - tdist.cdf(t[imsk], eNa[imsk] - 1))
  ci[imsk] = tdist.ppf(1. - alpha/2, eNa[imsk] - 1) * sdom[imsk]

  name = X.name if X.name != '' else 'X'

  from pygeode.var import Var
  from pygeode.dataset import asdataset

  rvs = []

  if 'm' in output:
    m = Var(oaxes, values=x, name='m')
    m.atts['longname'] = 'Mean value of %s' % (name, )
    rvs.append(m)

  if 'p' in output:
    p = Var(oaxes, values=p, name='p')
    p.atts['longname'] = 'p-value of test %s is 0' % (name,)
    rvs.append(p)

  if 'ci' in output:
    ci = Var(oaxes, values=ci, name='ci')
    ci.atts['longname'] = 'Confidence intervale of the mean value of %s' % (name,)
    rvs.append(ci)

  return asdataset(rvs)
# }}}
