''' Tools for computing some basic statistical quantities. '''

__all__ = ('correlate', 'regress', 'multiple_regress', 'difference', 'paired_difference', 'isnonzero')

import numpy as np
from scipy.stats import norm, t as tdist

def correlate(X, Y, axes=None, pbar=None):
# {{{
  r'''Computes correlation between variables X and Y.

  Parameters
  ==========
  X, Y : :class:`Var`
    Variables to correlate. Must have at least one axis in common.

  axes : list, optional
    Axes over which to compute correlation; if nothing is specified, the correlation
    is computed over all axes common to  shared by X and Y.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  rho, p : :class:`Var`
    The correlation coefficient :math:`\rho_{XY}` and p-value, respectively.

  Notes
  =====
  The coefficient :math:`\rho_{XY}` is computed following von Storch and Zwiers
  1999, section 8.2.2. The p-value is the probability of finding the given
  result under the hypothesis that the true correlation coefficient between X
  and Y is zero. It is computed from the t-statistic given in eq (8.7), in
  section 8.2.3, and assumes normally distributed quantities.'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npnansum
  from pygeode.view import View

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
  siaxes = range(len(oaxes), len(srcaxes))

  print oaxes
  print inaxes

  # Construct work arrays
  x  = np.zeros(oview.shape, 'd')*np.nan
  y  = np.zeros(oview.shape, 'd')*np.nan
  xx = np.zeros(oview.shape, 'd')*np.nan
  yy = np.zeros(oview.shape, 'd')*np.nan
  xy = np.zeros(oview.shape, 'd')*np.nan
  Na = np.zeros(oview.shape, 'd')*np.nan

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  for outsl, (xdata, ydata) in loopover([X, Y], oview, inaxes, pbar=pbar):
    xdata = xdata.astype('d')
    ydata = ydata.astype('d')
    xydata = xdata*ydata

    xbc = [s1 / s2 for s1, s2 in zip(xydata.shape, xdata.shape)]
    ybc = [s1 / s2 for s1, s2 in zip(xydata.shape, ydata.shape)]
    xdata = np.tile(xdata, xbc)
    ydata = np.tile(ydata, ybc)
    xdata[np.isnan(xydata)] = np.nan
    ydata[np.isnan(xydata)] = np.nan

    # It seems np.nansum does not broadcast its arguments automatically
    # so there must be a better way of doing this...
    x[outsl] = np.nansum([x[outsl], npnansum(xdata, siaxes)], 0)
    y[outsl]  = np.nansum([y[outsl], npnansum(ydata, siaxes)], 0)
    xx[outsl] = np.nansum([xx[outsl], npnansum(xdata**2, siaxes)], 0)
    yy[outsl] = np.nansum([yy[outsl], npnansum(ydata**2, siaxes)], 0)
    xy[outsl] = np.nansum([xy[outsl], npnansum(xydata, siaxes)], 0)

    # Sum of weights (kludge to get masking right)
    Na[outsl] = np.nansum([Na[outsl], npnansum(1. + xydata*0., siaxes)], 0) 

  print 'x NaNs:  %d of %d' % (np.sum(np.isnan(x)), x.size)
  print 'y NaNs:  %d of %d' % (np.sum(np.isnan(y)), y.size)
  print 'xx NaNs: %d of %d' % (np.sum(np.isnan(xx)), xx.size)
  print 'xx < 0:  %d of %d' % (np.sum(xx < 0.), xx.size)
  print 'xy NaNs: %d of %d' % (np.sum(np.isnan(xy)), xy.size)
  print 'yy NaNs: %d of %d' % (np.sum(np.isnan(yy)), yy.size)
  print 'yy < 0:  %d of %d' % (np.sum(yy < 0.), yy.size)
  print 'Na NaNs: %d of %d' % (np.sum(np.isnan(Na)), Na.size)

  xx -= x**2/Na
  yy -= y**2/Na
  xy -= (x*y)/Na

  print 'xx < 0:  %d of %d' % (np.sum(xx <= 0.), xx.size)
  print 'yy < 0:  %d of %d' % (np.sum(yy <= 0.), yy.size)
  
  # Compute correlation coefficient, t-statistic, p-value
  den = np.sqrt(xx*yy)
  rho = xy / [rho > 0.]
  rho = xy.copy()
  rho[rho > 0.] = rho[rho > 0.] / np.sqrt(xx*yy)[rho > 0.]
  print 'den NaNs:  %d of %d' % (np.sum(np.sqrt(xx*yy)[rho > 0.] <= 0.), rho[rho > 0.].size)
  print 'rho NaNs:  %d of %d' % (np.sum(np.isnan(rho)), rho.size)

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

def regress(X, Y, axes=None, pbar=None, N_fac=None, output='m,b,p'):
# {{{
  r'''Computes least-squares linear regression of Y against X.

  Parameters
  ==========
  X, Y : :class:`Var`
    Variables to regress. Must have at least one axis in common.

  axes : list, optional
    Axes over which to compute correlation; if nothing is specified, the correlation
    is computed over all axes common to X and Y.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  N_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom; the effective
    number will be given by the number estimated from the dataset divided by ``N_fac``.

  output : string, optional
    A string determining which parameters are returned; see list of possible outputs
    in the Returns section. The specifications must be separated by a comma. Defaults 
    to 'm,b,p'.

  Returns
  =======
  results : list of :class:`Var` instances.
    The return values are specified by the ``output`` argument. A fit of the form
    :math:`Y = m X + b + \epsilon` is assumed, and the following parameters
    can be returned:

    * 'm': Linear coefficient of the regression
    * 'b': Constant coefficient of the regression
    * 'r': Fraction of the variance in Y explained by X (:math:`R^2`)
    * 'p': Probability of this fit if the true linear coefficient was zero
    * 'sm': Variance in linear coefficient
    * 'se': Variance of residuals

  Notes
  =====
  The statistics described are computed following von Storch and Zwiers 1999,
  section 8.3. The p-value 'p' is computed using the t-statistic given in
  section 8.3.8, and confidence intervals for the slope and intercept can be
  computed from 'se' and 'se' (:math:`\hat{\sigma}_E` and
  :math:`\hat{\sigma}_E/\sqrt{S_{XX}}` in von Storch and Zwiers, respectively).
  The data is assumed to be normally distributed.'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npsum
  from pygeode.view import View

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

  if N_fac is None: N_eff = N
  else: N_eff = N / N_fac
  sige = (yy - m * xy) / (N_eff - 2.)
  sigm = np.sqrt(sige / xx)
  t = np.abs(m) / sigm
  p = tdist.cdf(t, N-2) * np.sign(m)
  xn = X.name if X.name != '' else 'X'
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  output = output.split(',')
  ret = []

  if 'm' in output:
    M = Var(oaxes, values=m, name='%s vs. %s' % (yn, xn))
    ret.append(M)
  if 'b' in output:
    B = Var(oaxes, values=b, name='Intercept (%s vs. %s)' % (yn, xn))
    ret.append(B)
  if 'r' in output:
    ret.append(Var(oaxes, values=xy**2/(xx*yy), name='R2(%s vs. %s)' % (yn, xn)))
  if 'p' in output:
    P = Var(oaxes, values=p, name='P(%s vs. %s != 0)' % (yn, xn))
    ret.append(P)
  if 'sm' in output:
    ret.append(Var(oaxes, values=sigm, name='Sig. Intercept (%s vs. %s != 0)' % (yn, xn)))
  if 'se' in output:
    ret.append(Var(oaxes, values=np.sqrt(sige), name='Sig. Resid. (%s vs. %s != 0)' % (yn, xn)))

  return ret
# }}}

def multiple_regress(Xs, Y, axes=None, pbar=None, N_fac=None, output='B,p'):
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

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  N_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom; the effective
    number will be given by the number estimated from the dataset divided by ``N_fac``.

  output : string, optional
    A string determining which parameters are returned; see list of possible outputs
    in the Returns section. The specifications must be separated by a comma. Defaults 
    to 'B,p'.

  Returns
  =======
  results : tuple of floats or :class:`Var` instances.
    The return values are specified by the ``output`` argument. A fit of the form
    :math:`Y = \sum_i \beta_i X_i + \epsilon` is assumed. Note that a constant term
    is not included by default. The following parameters can be returned:

    * 'B': Linear coefficients :math:`\beta_i` of each regressor
    * 'r': Fraction of the variance in Y explained by all Xs (:math:`R^2`)
    * 'p': Probability of this fit if the true linear coefficient was zero for each regressor
    * 'sb': Standard deviation of each linear coefficient
    * 'covb': Covariance matrix of the linear coefficients
    * 'se': Standard deviation of residuals

    If the regression is computed over all axes so that the result is a scalar,
    the above are returned as a tuple of floats in the order specified by
    ``output``. Otherwise they are returned as :class:`Var` instances. The outputs
    'B', 'p', and 'sb' will produce as many outputs as there are regressors. 

  Notes
  =====
  The statistics described are computed following von Storch and Zwiers 1999,
  section 8.4. The p-value 'p' is computed using the t-statistic appropriate
  for the multi-variate normal estimator :math:`\hat{\vec{a}}` given in section
  8.4.2; note this may not be the best way to determine if a given parameter is
  contributing a significant fraction to the explained variance of Y.  The
  variances 'se' and 'sb' are :math:`\hat{\sigma}_E` and the square root of the
  diagonal elements of :math:`\hat{\sigma}^2_E (\chi^T\chi)` in von Storch and
  Zwiers, respectively.  The data is assumed to be normally distributed.'''

  from pygeode.tools import loopover, whichaxis, combine_axes, shared_axes, npsum
  from pygeode.view import View

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
    
  oaxes = [srcaxes[i] for i in oiaxes]
  inaxes = oaxes + [srcaxes[i] for i in riaxes]
  oview = View(oaxes) 
  siaxes = range(len(oaxes), len(srcaxes))

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  assert len(riaxes) > 0, 'Regressors and %s share no axes to be regressed over' % (Y.name)

  # Construct work arrays
  os = oview.shape
  os1 = os + (Nr,)
  os2 = os + (Nr,Nr)
  y = np.zeros(os, 'd')
  yy = np.zeros(os, 'd')
  xy = np.zeros(os1, 'd')
  xx = np.zeros(os2, 'd')
  xxinv = np.zeros(os2, 'd')

  N = np.prod([len(srcaxes[i]) for i in riaxes])

  # Accumulate data
  for outsl, datatuple in loopover(Xs + [Y], oview, inaxes, pbar=pbar):
    ydata = datatuple[-1].astype('d')
    xdata = [datatuple[i].astype('d') for i in range(Nr)]
    y[outsl] += npsum(ydata, siaxes)
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
  xx = xx.reshape(-1, Nr, Nr)
  xxinv = xxinv.reshape(-1, Nr, Nr)
  for i in xrange(xx.shape[0]):
    xxinv[i,:,:] = np.linalg.inv(xx[i,:,:])
  xx = xx.reshape(os2)
  xxinv = xxinv.reshape(os2)

  beta = np.sum(xy.reshape(os + (1, Nr)) * xxinv, -1)
  vare = np.sum(xy * beta, -1)

  if N_fac is None: N_eff = N
  else: N_eff = N / N_fac

  sigbeta = [np.sqrt((yy - vare) * xxinv[..., i, i] / N_eff) for i in range(Nr)]

  xns = [X.name if X.name != '' else 'X%d' % i for i, X in enumerate(Xs)]
  yn = Y.name if Y.name != '' else 'Y'

  from pygeode.var import Var
  output = output.split(',')
  ret = []

  for o in output:
    if o == 'B':
      if len(oaxes) == 0:
        ret.append(beta)
      else:
        ret.append([Var(oaxes, values=beta[...,i], name='beta_%s' % xns[i]) for i in range(Nr)])
    elif o == 'r':
      vary = (yy - y**2/N)
      R2 = 1 - (yy - vare) / vary
      if len(oaxes) == 0:
        ret.append(R2)
      else:
        ret.append(Var(oaxes, values=R2, name='R2'))
    elif o == 'p':
      ps = [tdist.cdf(np.abs(beta[...,i]/sigbeta[i]), N_eff-Nr) * np.sign(beta[...,i]) for i in range(Nr)]
      if len(oaxes) == 0:
        ret.append(ps)
      else:
        ret.append([Var(oaxes, values=ps[i], name='p_%s' % xns[i]) for i in range(Nr)])
    elif o == 'sb':
      if len(oaxes) == 0:
        ret.append(sigbeta)
      else:
        ret.append([Var(oaxes, values=sigbeta[i], name='sig_%s' % xns[i]) for i in range(Nr)])
    elif o == 'covb':
      from axis import NonCoordinateAxis as nca
      cr1 = nca(values=range(Nr), regressor1=[X.name for X in Xs], name='regressor1')
      cr2 = nca(values=range(Nr), regressor2=[X.name for X in Xs], name='regressor2')
      sigmat = np.zeros(os2, 'd')
      for i in range(Nr):
        for j in range(Nr):
          #sigmat[..., i, j] = np.sqrt((yy - vare) * xxinv[..., i, j] / N_eff)
          sigmat[..., i, j] = (yy - vare) * xxinv[..., i, j] / N_eff
      ret.append(Var(oaxes + [cr1, cr2], values=sigmat, name='smat'))
    elif o == 'se':
      se = np.sqrt((yy - vare) / N_eff)
      if len(oaxes) == 0:
        ret.append(se)
      else:
        ret.append(Var(oaxes, values=se, name='sig_resid'))
    else:
      print 'multiple_regress: unrecognized output "%s"' % o

  return ret
# }}}

def difference(X, Y, axes, alpha=0.05, Nx_fac = None, Ny_fac = None, pbar=None):
# {{{
  r'''Computes the mean value and statistics of X - Y.

  Parameters
  ==========
  X, Y : :class:`Var`
    Variables to difference. Must have at least one axis in common.

  axes : list, optional
    Axes over which to compute means; if nothing is specified, the mean
    is computed over all axes common to X and Y.

  alpha : float
    Confidence level for which to compute confidence interval.

  Nx_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom of
    X; the effective number will be given by the number estimated from the
    dataset divided by ``Nx_fac``.

  Ny_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom of
    Y; the effective number will be given by the number estimated from the
    dataset divided by ``Ny_fac``.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : tuple or :class:`Dataset` instance.
    Four quantities are computed:

    * The difference in the means, X - Y
    * The effective number of degrees of freedom, :math:`df`
    * The probability of the computed difference if the population difference was zero
    * The confidence interval of the difference at the level specified by alpha

    If the average is taken over all axes of X and Y resulting in a scalar,
    the above values are returned as a tuple in the order given. If not, the
    results are provided as :class:`Var` objects in a dataset. 

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

  from pygeode.tools import combine_axes, whichaxis, loopover, npsum, npnansum
  from pygeode.view import View

  srcaxes = combine_axes([X, Y])
  riaxes = [whichaxis(srcaxes, n) for n in axes]
  raxes = [a for i, a in enumerate(srcaxes) if i in riaxes]
  oaxes = [a for i, a in enumerate(srcaxes) if i not in riaxes]
  oview = View(oaxes) 

  ixaxes = [X.whichaxis(n) for n in axes if X.hasaxis(n)]
  Nx = np.product([len(X.axes[i]) for i in ixaxes])

  iyaxes = [Y.whichaxis(n) for n in axes if Y.hasaxis(n)]
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

  Nx = np.zeros(oview.shape, 'd')
  Ny = np.zeros(oview.shape, 'd')

  x[()] = np.nan
  y[()] = np.nan
  xx[()] = np.nan
  yy[()] = np.nan
  Nx[()] = np.nan
  Ny[()] = np.nan

  # Accumulate data
  for outsl, (xdata,) in loopover([X], oview, pbar=pbar):
    xdata = xdata.astype('d')
    x[outsl] = np.nansum([x[outsl], npnansum(xdata, ixaxes)], 0)
    xx[outsl] = np.nansum([xx[outsl], npnansum(xdata**2, ixaxes)], 0)
    # Sum of weights (kludge to get masking right)
    Nx[outsl] = np.nansum([Nx[outsl], npnansum(1. + xdata*0., ixaxes)], 0) 

  for outsl, (ydata,) in loopover([Y], oview, pbar=pbar):
    ydata = ydata.astype('d')
    y[outsl] = np.nansum([y[outsl], npnansum(ydata, iyaxes)], 0)
    yy[outsl] = np.nansum([yy[outsl], npnansum(ydata**2, iyaxes)], 0)
    # Sum of weights (kludge to get masking right)
    Ny[outsl] = np.nansum([Ny[outsl], npnansum(1. + ydata*0., iyaxes)], 0) 

  # remove the mean (NOTE: numerically unstable if mean >> stdev)
  xx = (xx - x**2/Nx) / (Nx - 1)
  yy = (yy - y**2/Ny) / (Ny - 1)
  x /= Nx
  y /= Ny

  if Nx_fac is not None: eNx = Nx/Nx_fac
  else: eNx = Nx
  if Ny_fac is not None: eNy = Ny/Ny_fac
  else: eNy = Ny
  #print 'average eff. Nx = %.1f, average eff. Ny = %.1f' % (eNx.mean(), eNy.mean())

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

def paired_difference(X, Y, axes, alpha=0.05, N_fac = None, pbar=None):
# {{{
  r'''Computes the mean value and statistics of X - Y, assuming that individual elements
  of X and Y can be directly paired. In contrast to :func:`difference`, X and Y must have the same
  shape.

  Parameters
  ==========
  X, Y : :class:`Var`
    Variables to difference. Must have at least one axis in common.

  axes : list, optional
    Axes over which to compute means; if nothing is specified, the mean
    is computed over all axes common to X and Y.

  alpha : float
    Confidence level for which to compute confidence interval.

  Nx_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom of
    X; the effective number will be given by the number estimated from the
    dataset divided by ``Nx_fac``.

  Ny_fac : integer
    A factor by which to rescale the estimated number of degrees of freedom of
    Y; the effective number will be given by the number estimated from the
    dataset divided by ``Ny_fac``.

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : tuple or :class:`Dataset` instance.
    Four quantities are computed:

    * The difference in the means, X - Y
    * The effective number of degrees of freedom, :math:`df`
    * The probability of the computed difference if the population difference was zero
    * The confidence interval of the difference at the level specified by alpha

    If the average is taken over all axes of X and Y resulting in a scalar,
    the above values are returned as a tuple in the order given. If not, the
    results are provided as :class:`Var` objects in a dataset. 

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

  from pygeode.tools import combine_axes, whichaxis, loopover, npsum, npnansum
  from pygeode.view import View

  srcaxes = combine_axes([X, Y])
  riaxes = [whichaxis(srcaxes, n) for n in axes]
  raxes = [a for i, a in enumerate(srcaxes) if i in riaxes]
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
  d = np.zeros(oview.shape, 'd')
  dd = np.zeros(oview.shape, 'd')

  N = np.zeros(oview.shape, 'd')

  d[()] = np.nan
  dd[()] = np.nan
  N[()] = np.nan

  # Accumulate data
  for outsl, (xdata, ydata) in loopover([X, Y], oview, inaxes=srcaxes, pbar=pbar):
    ddata = xdata.astype('d') - ydata.astype('d')
    d[outsl] = np.nansum([d[outsl], npnansum(ddata, ixaxes)], 0)
    dd[outsl] = np.nansum([dd[outsl], npnansum(ddata**2, ixaxes)], 0)
    # Sum of weights (kludge to get masking right)
    N[outsl] = np.nansum([N[outsl], npnansum(1. + ddata*0., ixaxes)], 0) 

  # remove the mean (NOTE: numerically unstable if mean >> stdev)
  dd = (dd - d**2/N) / (N - 1)
  d /= Nx

  if N_fac is not None: eN = N/N_fac
  else: eN = N
  #print 'average eff. Nx = %.1f, average eff. Ny = %.1f' % (eNx.mean(), eNy.mean())

  den = np.sqrt(dd/(eN - 1))

  p = tdist.cdf(abs(d/den), eN - 1)*np.sign(d)
  ci = tdist.ppf(1. - alpha/2, eN - 1) * den

  xn = X.name if X.name != '' else 'X'
  yn = Y.name if Y.name != '' else 'Y'
  if xn == yn: name = xn
  else: name = '%s-%s'%(xn, yn)

  if len(oaxes) > 0:
    from pygeode import Var, Dataset
    D = Var(oaxes, values=d, name=name)
    DF = Var(oaxes, values=eN-1, name='df_%s' % name)
    P = Var(oaxes, values=p, name='p_%s' % name)
    CI = Var(oaxes, values=ci, name='CI_%s' % name)
    return Dataset([D, DF, P, CI])
  else: # Degenerate case
    return d, eN-1, p, ci
# }}}

def isnonzero(X, axes, alpha=0.05, N_fac = None, pbar=None):
# {{{
  r'''Computes the mean value and statistics of X, against the hypothesis that it is 0.

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

  pbar : progress bar, optional
    A progress bar object. If nothing is provided, a progress bar will be displayed
    if the calculation takes sufficiently long.

  Returns
  =======
  results : tuple or :class:`Dataset` instance.
    Three quantities are computed:

    * The mean value of X
    * The probability of the computed value if the population mean was zero
    * The confidence interval of the mean at the level specified by alpha

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
  #print 'eff. N = %.1f' % eN

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
