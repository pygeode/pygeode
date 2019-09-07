# Shortcuts for plotting PyGeode vars
# Extends wrapper.py to automatically use information from the Pygeode Vars.

from . import wrappers as wr
from . import cnt_helpers as ch
import numpy as np

def _getplotatts(var):
# {{{
  ''' Builds plotatts dictionary, using variable attributes as suitable defaults. '''
  plt = dict(plotname = var.name, plotunits = var.units, plotfmt = var.formatstr)
  plt.update([(k, v) for k, v in var.plotatts.items() if v is not None])
  return plt
# }}}

def _buildaxistitle(plotname=None, plottitle=None, plotunits=None, **dummy):
# {{{
  if plottitle is not None: return plottitle
  elif plotname is not None: title = plotname
  else: title = ''

  if plotunits not in [None, '']: title += ' [%s]' % plotunits

  return title
# }}}

def _buildvartitle(axes=None, plotname=None, plottitle=None, plotunits=None, **dummy):
# {{{
  if plottitle is not None: return plottitle
  elif plotname is not None: title = plotname
  else: title = ''

  if plotunits not in [None, '']: title += ' [%s]' % plotunits

  # Add information on degenerate axes to the title
  if axes is not None:
    for a in [a for a in axes if len(a) == 1]:
      title += ', ' + a.formatvalue(a.values[0])

  return title
# }}}

def scalevalues(var):
# {{{
   sf = var.plotatts.get('scalefactor', None)
   of = var.plotatts.get('offset', None)
   v = var.get().copy()
   if sf is not None: v *= sf
   if of is not None: v += of
   return v
# }}}

def axes_parm(axis):
# {{{
  vals = scalevalues(axis).ravel()
  lims = np.nanmin(vals), np.nanmax(vals)
  plt = _getplotatts(axis)
  return plt.get('plotscale', 'linear'), \
         _buildaxistitle(**plt), \
         lims[::plt.get('plotorder', 1)], \
         axis.formatter(), \
         axis.locator()
# }}}

def set_xaxis(axes, axis, lbl, xscale=True):
# {{{
  scale, label, lim, form, loc = axes_parm(axis)
  pl, pb, pr, pt = axes.pad

  prm = {}
  axprm = dict(major_locator = loc)
  if xscale:
    prm['xscale'] = scale
    prm['xlim']   = lim

  if lbl:
    prm['xlabel'] = label
    axprm['major_formatter'] = form
  else:
    prm['xticklabels'] = []

  axes.setp(**prm)
  axes.setp_xaxis(**axprm)

  if len(label) > 0 and lbl:
    axes.pad = [pl, 0.5, pr, 0.3]
  else:
    axes.pad = [pl, 0.3, pr, 0.3]
# }}}

def set_yaxis(axes, axis, lbl, yscale=True):
# {{{
  scale, label, lim, form, loc = axes_parm(axis)
  pl, pb, pr, pt = axes.pad

  prm = {}
  axprm = dict(major_locator = loc)
  if yscale:
    prm['yscale'] = scale
    prm['ylim']   = lim

  if lbl:
    prm['ylabel'] = label
    axprm['major_formatter'] = form
  else:
    prm['yticklabels'] = []

  axes.setp(**prm)
  axes.setp_yaxis(**axprm)

  if len(label) > 0 and lbl:
    axes.pad = [0.8, pb, 0.1, pt]
  else:
    axes.pad = [0.5, pb, 0.1, pt]
# }}}

def build_basemap(lons, lats, **kwargs):
# {{{
  if not hasattr(wr, 'BasemapAxes'):
    return wr.AxesWrapper()

  prd = dict(projection = 'cyl', resolution = 'c')
  prd.update(kwargs.pop('map', {}))
  proj = prd['projection']
  bnds = {}

  if proj not in ['sinu', 'moll', 'hammer', 'npstere', 'spstere', 'nplaea', 'splaea', 'npaeqd', \
                        'spaeqd', 'robin', 'eck4', 'kav7', 'mbtfpq', 'ortho', 'nsper']:
    bnds = {'llcrnrlat':lats.min(),
            'urcrnrlat':lats.max(),
            'llcrnrlon':lons.min(),
            'urcrnrlon':lons.max()}

  if proj == 'npstere':
    bnds = {'boundinglat':20, 'lon_0':0}

  if proj == 'spstere':
    bnds = {'boundinglat':-20, 'lon_0':0}

  bnds.update(prd)
  prd.update(bnds)

  return wr.BasemapAxes(**prd)
# }}}

def decorate_basemap(axes, **kwargs):
# {{{
  prd = dict(projection = 'cyl', resolution = 'c')
  prd.update(kwargs.pop('map', {}))

  if kwargs.pop('bluemarble', False):
    axes.bluemarble()

  # Add coastlines, meridians, parallels
  cld = {}
  merd = dict(meridians=[-180,-90,0,90,180,270,360],
              labels=[1,0,0,1])
  pard = dict(circles=[-90,-60,-30,0,30,60,90],
              labels=[1,0,0,1])

  cld.update(kwargs.pop('coastlines', {}))
  merd.update(kwargs.pop('meridians', {}))
  pard.update(kwargs.pop('parallels', {}))

  axes.pad=(0.6, 0.4, 0.4, 0.4)
  if prd.get('resolution', 'c') is not None:
    axes.drawcoastlines(**cld)
    axes.drawmeridians(**merd)
    axes.drawparallels(**pard)
# }}}

def _parse_autofmt_kwargs(Z, kwargs):
# {{{
  ''' Used by showvar and showgrid to parse kwargs for auto-contouring options. '''

  # Process auto contouring dictionaries
  if 'clevs' in list(kwargs.keys()) or 'clines' in list(kwargs.keys()):
    tdef = 'raw'
  else:
    tdef = 'clf'

  typ = kwargs.pop('type', tdef)

  if typ == 'raw':
    return kwargs
      
  elif typ == 'clf':
    cdelt = kwargs.pop('cdelt', None)
    if cdelt is None:
      z = scalevalues(Z)
      style = kwargs.pop('style',  None)
      ndiv  = kwargs.pop('ndiv',  None)
      cdelt, dct = ch.guessclimits(z, style=style, ndiv=ndiv)
    else:
      dct = {}
      if 'min' in kwargs and 'style' not in kwargs:
        dct['style'] = 'seq'
        dct['ndiv'] = 5
    dct.update(kwargs)
    kwargs = ch.clfdict(cdelt, **dct)
    return kwargs

  elif typ == 'cl':
    cdelt = kwargs.pop('cdelt', None)
    if cdelt is None:
      z = scalevalues(Z)
      style = kwargs.pop('style',  None)
      ndiv = kwargs.pop('ndiv', None)
      cdelt, dct = ch.guessclimits(z, style=style, ndiv=ndiv, clf=False)
      verbose = True
    else:
      ndiv = kwargs.pop('ndiv', 10)
      range = kwargs.pop('range', ndiv*cdelt)
      dct = dict(range=range)
      verbose = False
    dct.update(kwargs)
    kwargs = ch.cldict(cdelt, **dct)
    if verbose:
      print('Contour Interval: %0.2g' % cdelt)
      print('Minimum value: %3g, Maximum value: %3g' % (np.min(z), np.max(z)))
      print('Minimum contour: %3g, Maximum contour: %3g' % (kwargs['clines'][0], kwargs['clines'][-1]))

    return kwargs

  elif typ in ['log', 'log1s']:
    dct = {}
    dct.update(kwargs)
    if 'cmin' not in dct:
      raise ValueError('Must specify cmin (lower bound) for logarithmically spaced contours')
    kwargs = ch.log1sdict(**dct)
    return kwargs

  elif typ == 'log2s':
    dct = {}
    dct.update(kwargs)
    if not ('cmin' in dct) :
      raise ValueError('Must specify cmin (inner boundary for linear spaced interval) for two-sided logarithmically spaced contours')
    kwargs = ch.log2sdict(**dct)
    return kwargs

  else:
    raise ValueError("Unrecognized 2d plot type '%s'" % typ)
# }}}

# Do a 1D line plot
def vplot(var, fmt='', axes=None, transpose=False, lblx=True, lbly=True, **kwargs):
# {{{
  '''
  Create a line plot of a variable.

  Parameters
  ----------
  var :  :class:`Var`
     The variable to plot. Should have 1 non-degenerate axis.

  fmt :  string, optional
     Format of the line. See :func:`pylab.plot`.

  '''

  Y = var.squeeze().load()
  assert Y.naxes == 1, 'Variable to plot must have exactly one non-degenerate axis.'
  X = Y.axes[0].load()

  yord = True

  # If a vertical axis is present transpose the plot
  from pygeode.axis import ZAxis
  if isinstance(X, ZAxis):
    X, Y = Y, X
    yord = False

  if transpose:
    X, Y = Y, X
    yord = not yord

  if yord:
    scalex = kwargs.get('scalex', True)
    scaley = kwargs.get('scaley', False)
  else:
    scalex = kwargs.get('scalex', False)
    scaley = kwargs.get('scaley', True)

  hold = kwargs.pop('hold', False)

  x = scalevalues(X)
  y = scalevalues(Y)

  axes = wr.plot(x, y, fmt, axes=axes, **kwargs)

  # Apply the custom axes args
  if not hold:
    axes.pad = (0.1, 0.1, 0.1, 0.1)
    set_xaxis(axes, X, lblx, scalex)
    set_yaxis(axes, Y, lbly, scaley)
    plt = _getplotatts(var)
    lbl = _buildvartitle(var.axes, **plt)
    axes.setp(title=lbl, label=lbl)

  return axes
# }}}

def vhist(var, axes=None, lblx=True, lbly=True, **kwargs):
# {{{
  '''
  Create a histogram of values taken by a variable.

  Parameters
  ----------
  var :  :class:`Var`
     The variable to compute the histogram of.
  '''

  V = var.squeeze().load()

  v = scalevalues(V).ravel()

  axes = wr.hist(v, axes=axes, **kwargs)

  # Apply the custom axes args
  axes.pad = (0.1, 0.1, 0.1, 0.1)
  set_xaxis(axes, V, lblx)
  plt = _getplotatts(var)
  lbl = _buildvartitle(var.axes, **plt)
  axes.setp(title=lbl, label=lbl)

  return axes
# }}}

# Do a scatter plot
def vscatter(varx, vary, axes=None, lblx=True, lbly=True, **kwargs):
# {{{
  '''
  Create a scatter plot from two variables with the same shape.

  Parameters
  ----------
  varx :  :class:`Var`
     Variable to use as abscissa values. Must have the same size as vary.

  vary :  :class:`Var`
     Variable to use as ordinate values.

  Notes
  -----
  Wraps matplotlib.scatter
  '''

  assert varx.size == vary.size, 'Variables must have same size to do a scatter plot.'

  x = scalevalues(varx).ravel()
  y = scalevalues(vary).ravel()

  axes = wr.scatter(x, y, axes=axes, **kwargs)

  # Apply the custom axes args
  axes.pad = (0.1, 0.1, 0.1, 0.1)
  set_xaxis(axes, varx, lblx)
  set_yaxis(axes, vary, lbly)

  return axes
# }}}

# Do a 2D contour plot
def vcontour(var, clevs=None, clines=None, axes=None, lblx=True, lbly=True, label=True, transpose=None, **kwargs):
# {{{
  '''Create a contour plot (lines, filled, or both) from a variable.

  Parameters
  ----------
  var :  :class:`Var`
    The variable to plot. Should have 2 non-degenerate axes.

  clevs : integer or collection of numbers, optional
    Levels at which to construct filled contours through an underlying call to
    :func:`matplotlib.contourf`. If None is specified, no filled contours will
    be produced, unless clines is also None. If a number is specified, that
    number of equally spaced contours are chosen. Otherwise the explicit
    values are used.

  clines : integer or collection of numbers, optional
    Levels at which to construct contour lines through an underlying call to
    :func:`matplotlib.contour`. If None is specified, no contour lines are
    produced.  If a number is specified, that number of equally spaced contours
    are chosen. Otherwise the explicit values are used.

  axes : :class:`AxisWrapper`, optional
    Axes on which to produce contour plot. If none is specified, one is created.

  lblx : bool, optional
    If True, add appropriate tick labels and an axis label on the x axis; if
    False, the x axis a is left unlabeled. Defaults to True.

  lbly : bool, optional
    If True, add appropriate tick labels and an axis label on the y axis; if
    False, the y axis a is left unlabeled. Defaults to True.

  transpose : bool, optional
    The x and y axes are chosen based on the two degenerate axes of the variable
    to plot. This order can be reversed by setting transpose to True.

  map : dict, optional
    If

  *args, **kwargs : arguments to pass on to underlying matplotlib contour
    plotting routines, see Notes.

  Returns
  -------
  axes : :class:`AxesWrapper`
    The axes object containing the contour plots.

  Notes
  -----
  If the two axes of the variable are a :class:`Lat` and :class:`Lon` axes,
  a map projection is created automatically.

  See Also
  --------
  showvar, colorbar
  '''
  Z = var.squeeze()
  assert Z.naxes == 2, 'Variable to contour must have two non-degenerate axes.'
  X, Y = Z.axes

  # If a vertical axis is present transpose the plot
  from pygeode.axis import ZAxis, Lat, Lon
  if transpose is None:
    if isinstance(X, ZAxis):
      X, Y = Y, X
    if isinstance(X, Lat) and isinstance(Y, Lon):
      X, Y = Y, X
  elif transpose:
    X, Y = Y, X

  x = scalevalues(X)
  y = scalevalues(Y)
  z = scalevalues(Z.transpose(Y, X))

  if axes is None:
    if isinstance(X, Lon) and isinstance(Y, Lat) and kwargs.get('map', None) is not False:
      axes = build_basemap(x, y, **kwargs)
    else:
      axes = wr.AxesWrapper()

  if clevs is None and clines is None:
    # If both clevs and clines are None, use default
    axes.contourf(x, y, z, 21, **kwargs)

  if not clevs is None:
    lw = kwargs.pop('linewidths', None)
    axes.contourf(x, y, z, clevs, **kwargs)
    if lw is not None:
      kwargs['linewidths'] = lw
    # Special case; if plotting both filled and unfilled contours
    # with a single call, set the color of the latter to black
    kwargs['colors'] = 'k'
    kwargs['cmap'] = None

  if not clines is None:
    axes.contour(x, y, z, clines, **kwargs)

  # Apply the custom axes args
  if label:
    axes.pad = (0.1, 0.1, 0.1, 0.1)
    if wr.isbasemapaxis(axes):
      decorate_basemap(axes, **kwargs)
    else:
      axes.pad = (0.1, 0.1, 0.1, 0.1)
      set_xaxis(axes, X, lblx)
      set_yaxis(axes, Y, lbly)
    plt = _getplotatts(var)
    axes.setp(title = _buildvartitle(var.axes, **plt))

  return axes
# }}}

# Do a 2D significance mask
def vsigmask(pval, axes, mjsig=0.95, mnsig=None, mjsigp = None, mnsigp = None, nsigp = None, transpose=None):
# {{{
  '''
  Add significance shading to a contour plot from a variable.

  Parameters
  ----------
  pval :  :class:`Var`
    The variable containing a p-value of the significance mask. The mask will
    be applied where abs(pval) > mjsig (and optionally an additional mask will
    be applied for mnsig < abs(pval) < mjsig. Signed p-values for two-sided
    tests ensure that a gap will appear between significant regions of opposite sign.
    Should have two non-degenerate axes that match the quantity plotted.

  axes :  :class:`AxesWrapper`
    The axis on which to add the mask.

  mjsig : float, optional [0.95]
    The p-value dividing significant from non-significant values.

  mnsig : float or None, optional [None]
    The p-value dividing minor significance from non-significant values.

  mjsigp : dictionary, optional
    A dictionary of keyword arguments that determine the properties of the
    (major) significant filled contours. See notes.

  mnsigp : dictionary, optional
    A dictionary of keyword arguments that determine the properties of the
    minor significant filled contours. See notes.

  nsigp : dictionary, optional
    A dictionary of keyword arguments that determine the properties of the
    non-significant filled contour. See notes.

  transpose : bool or None, optional [None]
    If True, transpose the axes of the plot.

  Notes
  -----
  The significance mask is plotted as three (or five) filled contours, with
  boundaries at [-1.1, -mjsig, mjsig, 1.1] or [-1.1, -mjsig, -mnsig, mnsig,
  mjsig, 1.1]. Their respective graphical properties can be set using the
  dictionary kw arguments.

  By default, the non-significant contours are set to be invisible and the
  significant contours are set to a hatching pattern; this is equivalent to
  passing in mjsigp = dict(alpha = 0., hatch = '...') and nsigp = dict(visible
  = False). Any property of the filled contour can be set.

  Returns
  -------
  :class:`AxesWrapper` object with plot.
  '''

  Z = pval.squeeze()
  assert Z.naxes == 2, 'Variable to contour must have two non-degenerate axes.'
  X, Y = Z.axes

  # If a vertical axis is present transpose the plot
  from pygeode.axis import ZAxis, Lat, Lon
  if transpose is None:
    if isinstance(X, ZAxis):
      X, Y = Y, X
    if isinstance(X, Lat) and isinstance(Y, Lon):
      X, Y = Y, X
  elif transpose:
    X, Y = Y, X

  x = scalevalues(X)
  y = scalevalues(Y)
  z = scalevalues(Z.transpose(Y, X))

  if mjsigp is None: mjsigp = dict(alpha = 0., hatch = '...')
  if mnsigp is None: mnsigp = dict(alpha = 0., hatch = '..')
  if nsigp  is None: nsigp  = dict(visible = False)

  if mnsig is None:
    cl = [-1.1, -mjsig, mjsig, 1.1]
    axes.contourf(x, y, z, cl, colors='w')
    cnt = axes.plots[-1]
    axes.modifycontours(cnt, ind=[0, 2], **mjsigp)
    axes.modifycontours(cnt, ind=[1],    **nsigp)
  else:
    cl = [-1.1, -mnsig,-mjsig, mjsig, mnsig, 1.1]
    axes.contourf(x, y, z, cl, colors='w')
    cnt = axes.plots[-1]
    axes.modifycontours(cnt, ind=[0, 4], **mjsigp)
    axes.modifycontours(cnt, ind=[1, 3], **mnsigp)
    axes.modifycontours(cnt, ind=[2],    **nsigp)

  return axes
# }}}

# Do a stream plot
def vstreamplot(varu, varv, axes=None, lblx=True, lbly=True, label=True, transpose=None, **kwargs):
# {{{
  '''
  Create a streamplot from two variables.

  Parameters
  ----------
  '''
  U = varu.squeeze()
  V = varv.squeeze()
  assert U.naxes == 2 and V.naxes == 2, 'Variables to quiver must have two non-degenerate axes.'
  X, Y = U.axes

  from pygeode.axis import Lat, Lon
  # If a vertical axis is present transpose the plot
  from pygeode.axis import ZAxis, Lat, Lon
  if transpose is None:
    if isinstance(X, ZAxis):
      X, Y = Y, X
    if isinstance(X, Lat) and isinstance(Y, Lon):
      X, Y = Y, X
  elif transpose:
    X, Y = Y, X

  x = scalevalues(X)
  y = scalevalues(Y)
  u = scalevalues(U.transpose(Y, X))
  v = scalevalues(V.transpose(Y, X))

  map = kwargs.pop('map', None)

  if axes is None:
    if isinstance(X, Lon) and isinstance(Y, Lat) and map is not False:
      axes = build_basemap(x, y, map = map, **kwargs)
    else:
      axes = wr.AxesWrapper()

  axes.streamplot(x, y, u, v, **kwargs)

  # Apply the custom axes args
  if label:
    axes.pad = (0.1, 0.1, 0.1, 0.1)
    if wr.isbasemapaxis(axes):
      decorate_basemap(axes, map = map, **kwargs)
    else:
      set_xaxis(axes, X, lblx)
      set_yaxis(axes, Y, lbly)
    plt = _getplotatts(varu)
    axes.setp(title = _buildvartitle(varu.axes, **plt))

  return axes
# }}}

# Do a quiver plot
def vquiver(varu, varv, varc=None, axes=None, lblx=True, lbly=True, label=True, transpose=None, **kwargs):
# {{{
  '''
  Create a quiver plot from two variables.

  Parameters
  ----------
  '''
  U = varu.squeeze()
  V = varv.squeeze()
  assert U.naxes == 2 and V.naxes == 2, 'Variables to quiver must have two non-degenerate axes.'
  assert U.axes == V.axes, 'Variables U and V must have the same axes.'
  X, Y = U.axes

  if varc is not None:
    C = varc.squeeze()
    assert U.axes == C.axes, 'Color values must have same axes as U and V'

  from pygeode.axis import Lat, Lon
  # If a vertical axis is present transpose the plot
  from pygeode.axis import ZAxis, Lat, Lon
  if transpose is None:
    if isinstance(X, ZAxis):
      X, Y = Y, X
    if isinstance(X, Lat) and isinstance(Y, Lon):
      X, Y = Y, X
  elif transpose:
    X, Y = Y, X

  x = scalevalues(X)
  y = scalevalues(Y)
  u = scalevalues(U.transpose(Y, X))
  v = scalevalues(V.transpose(Y, X))
  if varc is not None:
    c = scalevalues(C.transpose(Y, X))

  map = kwargs.pop('map', None)

  if axes is None:
    if isinstance(X, Lon) and isinstance(Y, Lat) and map is not False:
      axes = build_basemap(x, y, map = map, **kwargs)
    else:
      axes = wr.AxesWrapper()

  if varc is not None:
    axes.quiver(x, y, u, v, c, **kwargs)
  else:
    axes.quiver(x, y, u, v, **kwargs)

  # Apply the custom axes args
  if label:
    axes.pad = (0.1, 0.1, 0.1, 0.1)
    if wr.isbasemapaxis(axes):
      decorate_basemap(axes, map = map, **kwargs)
    else:
      set_xaxis(axes, X, lblx)
      set_yaxis(axes, Y, lbly)
    plt = _getplotatts(varu)
    axes.setp(title = _buildvartitle(varu.axes, **plt))

  return axes
# }}}

# Generic catch all interface (plotvar replacement)
def showvar(var, *args, **kwargs):
# {{{
  '''Plot variable, showing a contour plot for 2d variables or a line plot for
  1d variables.

  Parameters
  ----------
  var :  :class:`Var`
     The variable to plot. Should have either 1 or 2 non-degenerate axes.
     Arguments below relevant only for the 1 dimensional case are labelled [1D],
     those relevant only for the 2 dimensional case are labelled [2D].

  fmt : string, optional
    [1D] matplotlib format to plot line. See :func:`matplotlib.plot()`. Will also
    be recognized as the second positional argument (after var).

  type : string, optional ['clf']
    [2D] style of plot to produce. See Notes.

  axes : :class:`AxesWrapper` instance, optional
    Axes object on which to plot variable. A new one is created if this is not specified.

  transpose: boolean, optional [False]
    If True, reverse axes.

  lblx: boolean, optional [True]
    If True, label horizontal axes

  lbly: boolean, optional [True]
    If True, label vertical axes

  *args, **kwargs :
    Further arguments are passed on to the underlying plotting routine. See
    Notes.

  Notes
  -----
  This function is intended as the simplest way to display the contents of a
  variable, choosing appropriate parameter values as automatically as possible.
  For 1d variables it calls :func:`Var.vplot()`, and for 2d variables
  :func:`Var.vcontour`. In the latter case, if filled contours were produced, it
  calls :func:`AxesWrapper.colorbar()`. A dictionary ``colorbar`` can be provided to
  pass arguments through. Setting ``colorbar`` to ``False`` suppresses the
  colorbar.

  Returns
  -------
  :class:`AxesWrapper` object with plot.

  See Also
  --------
  vplot, vcontour, colorbar
  '''

  Z = var.squeeze()
  assert Z.naxes in [1, 2], 'Variable %s has %d non-generate axes; must have 1 or 2.' % (var.name, Z.naxes)

  var = var.load()

  fig = kwargs.pop('fig', None)
  size = kwargs.pop('size', None)

  if Z.naxes == 1:
    ax = vplot(var, *args, **kwargs)
    if size is not None: ax.size = size

  elif Z.naxes == 2:
    kwargs = _parse_autofmt_kwargs(Z, kwargs)

    cbar = kwargs.pop('colorbar', dict(orientation='vertical'))

    ax = vcontour(var, *args, **kwargs)

    if size is not None: ax.size = size

    cf = ax.find_plot(wr.Contourf)
    cl = ax.find_plot(wr.Contour)
    if cbar and cf is not None:
      if cl is not None: cbar['lcnt'] = cl
      ax = wr.colorbar(ax, cf, **cbar)

  import pylab as pyl
  if pyl.isinteractive():
    ax.render(fig)
  return ax
# }}}

def showcol(vs, size=(4.1,2), **kwargs):
# {{{
  '''
  Plot column of contour plots. Superseded by :func:`showgrid`.

  Parameters
  ----------
  v :  list of lists of :class:`Var`
     The variables to plot. Should have either 1 or 2 non-degenerate axes.

  Notes
  -----
  This function is intended as the simplest way to display the contents of a variable,
  choosing appropriate parameter values as automatically as possible.
  '''

  Z = [v.squeeze() for v in vs]

  assert Z[0].naxes in [1, 2], 'Variables %s has %d non-generate axes; must have 1 or 2.' % (Z.name, Z.naxes)

  for z in Z[1:]:
    assert Z[0].naxes == z.naxes, 'All variables must have the same number of non-generate dimensions'
    #assert all([a == b for a, b in zip(Z[0].axes, z.axes)])

  fig = kwargs.pop('fig', None)

  if Z[0].naxes == 1:
    axs = []
    ydat = []
    for v in vs:
      lblx = (v is vs[-1])
      ax = vplot(v, lblx = lblx, **kwargs)
      ax.size = size
      axs.append([ax])
      ydat.append(ax.find_plot(wr.Plot).plot_args[1])

    Ax = wr.grid(axs)
    ylim = (np.min([np.min(y) for y in ydat]), np.max([np.max(y) for y in ydat]))
    Ax.setp(ylim = ylim, children=True)

  elif Z[0].naxes == 2:
    axs = []
    for v in vs:
      lblx = (v is vs[-1])
      ax = vcontour(v, lblx = lblx, **kwargs)
      ax.size = size
      axs.append([ax])

    Ax = wr.grid(axs)

    cbar = kwargs.pop('colorbar', dict(orientation='vertical'))
    cf = Ax.axes[0].find_plot(wr.Contourf)
    if cbar and cf is not None:
      Ax = wr.colorbar(Ax, cf, **cbar)

  import pylab as pyl
  if pyl.isinteractive(): Ax.render(fig)
  return Ax
# }}}

def showgrid(vf, vl=[], ncol=1, size=(3.5,1.5), lbl=True, **kwargs):
# {{{
  '''
  Create grid of contour plots of multiple variables.

  Parameters
  ----------
  vf :  list of lists of :class:`Var`
    The variables to plot. Should have 2 non-degenerate axes.

  ncol : integer
    Number of columns
  '''

  from pygeode import Var
  if isinstance(vf, Var): vf = [vf]
  if isinstance(vl, Var): vl = [vl]

  assert all([v.squeeze().naxes == 2 for v in vf]), 'Variables (vf) should have 2 degenerate axes.'
  nVf = len(vf)

  assert all([v.squeeze().naxes == 2 for v in vl]), 'Variables (vl) should have 2 degenerate axes.'
  nVl = len(vl)
  if nVf > 1 and nVl > 1: assert nVl == nVf, 'Must have the same number of filled and contour-line variables.'

  fig = kwargs.pop('fig', None)

  kwargs = _parse_autofmt_kwargs(vf[0], kwargs)

  cbar = kwargs.pop('colorbar', dict(orientation='vertical'))
  xpad = 0.
  ypad = 0.

  kwlines = {}
  if nVl > 0:
    kwlines['colors'] = kwargs.pop('colors', 'k')
    kwlines['clines'] = kwargs.pop('clines', 11)
    for k in ['linewidths', 'linestyles']:
      if k in kwargs: kwlines[k] = kwargs.pop(k)

  kwfill = {}
  if nVf > 0:
    kwfill['clevs'] = kwargs.pop('clevs', 31)
    kwfill['cmap'] = kwargs.pop('cmap', None)
    kwlines['label'] = False
    if cbar:
      if cbar.get('orientation', 'vertical') == 'vertical':
        ypad = cbar.get('width', 0.8)
      else:
        xpad = cbar.get('height', 0.4)


  kwcb = {}

  nV = max(nVl, nVf)
  nrow = np.ceil(nV / float(ncol))

  axpad = kwargs.pop('axpad', 0.2)
  aypad = kwargs.pop('aypad', 0.4)
  if lbl:
    axpadl = axpad + kwargs.pop('lblxpad', 0.7)
    aypadl = aypad + kwargs.pop('lblypad', 0.15)
  else:
    axpadl = axpad
    aypadl = aypad

  axw, axh = size

  axs = []
  row = []
  for i in range(nV):
    lblx = (i / ncol == nrow - 1)
    lbly = (i % ncol == 0)
    ax = None
    if nVf > 0:
      v = vf[i % nVf]
      kwfill.update(kwargs)
      ax = vcontour(v, axes=ax, lblx = lblx, lbly = lbly, **kwfill)
    if nVl > 0:
      v = vl[i % nVl]
      kwlines.update(kwargs)
      ax = vcontour(v, axes=ax, lblx = lblx, lbly = lbly, **kwlines)

    if hasattr(wr, 'BasemapAxes') and isinstance(ax, wr.BasemapAxes):
      w = axw
      h = axh
    else:
      pl, pb, pr, pt = ax.pad
      if lblx:
        py = aypadl
      else:
        py = aypad
      if lbly:
        px = axpadl
      else:
        px = axpad
      h = axh + py
      w = axw + px
      ax.pad = [px-pr, py-pt, pr, pt]

    ax.size = (w, h)
    row.append(ax)
    if i % ncol == ncol - 1:
      axs.append(row)
      row = []

  if len(row) > 0:
    row.extend([None] * (ncol - len(row)))
    axs.append(row)

  Ax = wr.grid(axs)

  cf = Ax.axes[0].find_plot(wr.Contourf)
  if cbar and cf is not None:
    Ax = wr.colorbar(Ax, cf, **cbar)

  import pylab as pyl
  if pyl.isinteractive(): Ax.render(fig)
  return Ax
# }}}

def showlines(vs, fmts=None, labels=None, size=(4.1,2), lblx=True, lbly=True, **kwargs):
# {{{
  '''
  Produce line plots of a list of 1D variables on a single figure.

  Parameters
  ----------
  vs :  list of :class:`Var`
     The variables to plot. Should all have 1 non-degenerate axis.
  '''

  Z = [v.squeeze() for v in vs]

  for z in Z:
    assert z.naxes == 1, 'Variable %s has %d non-generate axes; must have 1.' % (z.name, z.naxes)

  fig = kwargs.pop('fig', None)

  ax = wr.AxesWrapper(size=size)
  ydat = []
  for i, v in enumerate(vs):
    if fmts is None:
      fmt = ''
    elif hasattr(fmts, '__len__'):
      fmt = fmts[i]
    else:
      fmt = fmts

    if labels is None:
      lbl = v.name
    elif hasattr(labels, '__len__'):
      lbl = labels[i]
    else:
      lbl = labels

    vplot(v, axes=ax, fmt=fmt, label=lbl)
    ydat.append(ax.find_plot(wr.Plot).plot_args[1])

  #ylim = (np.min([np.min(y) for y in ydat]), np.max([np.max(y) for y in ydat]))
  #kwargs.update(dict(ylim=ylim))

  kwleg = kwargs.pop('legend', dict(loc='best', frameon=False))
  ax.legend(**kwleg)
  ax.setp(**kwargs)

  import pylab as pyl
  if pyl.isinteractive(): ax.render(fig)
  return ax
# }}}

def savepages(figs, fn, psize='A4', marg=0.5, scl=1.):
# {{{
  sizes = dict(A4 = (8.3, 11.7),
              A4l = (11.7, 8.3))
  if psize in sizes:
    pwidth, pheight = sizes[psize]
  else:
    pwidth, pheight = psize

  hmarg = marg
  wmarg = marg
  psize = (pwidth, pheight)

  fwidth = pwidth - 2*wmarg
  fheight = pheight - 2*hmarg

  ymarg = hmarg/pheight
  xmarg = wmarg/pwidth

  y = 1. - ymarg
  x = xmarg

  ax = wr.AxesWrapper(size=psize)
  from matplotlib.backends.backend_pdf import PdfPages
  pp = PdfPages(fn)

  page = 1
  nfigs = 0
  hlast = 0
  for f in figs:
    w = f.size[0] / pwidth * scl
    h = f.size[1] / pheight * scl

    if x + w < 1. - xmarg:
      r = [x, y - h, x + w, y]
      ax.add_axis(f, r)
      x += w
      hlast = max(h, hlast)
      nfigs += 1
    else:
      x = xmarg
      y = y - hlast

      if nfigs > 0 and y - h < ymarg:
        fig = ax.render('page%d' % page)
        pp.savefig(fig)
        ax = wr.AxesWrapper(size=psize)
        y = 1. - ymarg
        print('Page %d, %d figures.' % (page, nfigs))
        page += 1
        nfigs = 0

      r = [x, y - h, x + w, y]
      ax.add_axis(f, r)
      x = x + w
      hlast = h
      nfigs += 1

  print('Page %d, %d figures.' % (page, nfigs))
  fig = ax.render('page%d'%page, show=False)
  pp.savefig(fig)
  pp.close()
# }}}

__all__ = ['showvar', 'showcol', 'showgrid', 'showlines', 'vplot', 'vscatter', \
          'vhist', 'vcontour', 'vsigmask', 'vstreamplot', 'vquiver', 'savepages']
