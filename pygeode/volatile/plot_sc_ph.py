# Shortcuts for plotting PyGeode vars
# Extends plot_wrapper to automatically use information from the Pygeode Vars.

def _buildaxistitle(name = '', plotname = '', plottitle = '', plotunits = '', **dummy):
# {{{
  if name is None: name = ''
  if plotname is None: plotname = ''
  if plottitle is None: plottitle = ''
  if plotunits is None: plotunits = ''

  assert type(plotname) is str
  assert type(plottitle) is str
  assert type(plotunits) is str
  assert type(name) is str
  
  if plotname is not '': title = plotname # plotname is shorter, hence more suitable for axes
  elif plottitle is not '': title = plottitle
  elif name is not '': title = name
  else: title = ''

  if plotunits is not '': title += ' [%s]' % plotunits

  return title
# }}}

def _buildvartitle(axes = None, name = '', plotname = '', plottitle = '', plotunits = '', **dummy):
# {{{
  if name is None: name = ''
  if plotname is None: plotname = ''
  if plottitle is None: plottitle = ''
  if plotunits is None: plotunits = ''
  
  assert type(plotname) is str
  assert type(plottitle) is str
  assert type(plotunits) is str
  assert type(name) is str

  if plottitle is not '': title = plottitle # plottitle is longer, hence more suitable for axes
  elif plotname is not '': title = plotname
  elif name is not '': title = name
  else: title = 'Unnamed Var'

  if plotunits is not '': title += ' (%s)' % plotunits
    
  # Add information on degenerate axes to the title
  if axes is not None:
    for a in [a for a in axes if len(a) == 1]:
      title += ', ' + a.formatvalue(a.values[0])

  return title
# }}}

def set_xaxis(axes, axis):
# {{{
  vals = axis.get()
  lims = min(vals), max(vals)
  axes.setp(
      xscale = axis.plotatts.get('plotscale', 'linear'),
      xlabel = _buildaxistitle(**axis.plotatts),
      xlim = lims[::axis.plotatts['plotorder']])
  axes.setp_xaxis(
      major_formatter = axis.formatter(),
      major_locator = axis.locator())
# }}}

def set_yaxis(axes, axis):
# {{{
  vals = axis.get()
  lims = min(vals), max(vals)
  axes.setp(
      yscale = axis.plotatts.get('plotscale', 'linear'),
      ylabel = _buildaxistitle(**axis.plotatts),
      ylim = lims[::axis.plotatts['plotorder']])
  axes.setp_yaxis(
      major_formatter = axis.formatter(),
      major_locator = axis.locator())
# }}}

# Do a 1D line plot
def plot (var, fmt='', axes=None, **kwargs):
# {{{
  import plot_wr_ph as pl

  Y = var.squeeze()
  assert Y.naxes == 1, 'Variable to plot must have exactly one non-degenerate axis.'
  X = Y.axes[0]

  # If a vertical axis is present transpose the plot
  from pygeode.axis import ZAxis
  if isinstance(X, ZAxis):
    X, Y = Y, X

  x = X.get()
  y = Y.get()

  axes = pl.plot(x, y, fmt, axes=axes, **kwargs)

  # Apply the custom axes args
  set_xaxis(axes, X)
  set_yaxis(axes, Y)
  axes.setp(title = _buildvartitle(var.axes, var.name, **var.plotatts))

  return axes
# }}}

# Do a 2D contour plot
def contour (var, clevs=None, clines=None, axes=None, **kwargs):
# {{{
  import plot_wr_ph as pl

  Z = var.squeeze()
  assert Z.naxes == 2, 'Variable to contour must have two non-degenerate axes.'
  Y, X = Z.axes

  # If a vertical axis is present transpose the plot
  from pygeode.axis import ZAxis, Lat, Lon
  if isinstance(X, ZAxis):
    X, Y = Y, X

  x = X.get()
  y = Y.get()
  z = Z.transpose(Y, X).get()

  if axes is None: 
    if isinstance(X, Lon) and isinstance(Y, Lat):
      axes = pl.BasemapAxes()
    else:
      axes = pl.AxesWrapper()

  if clevs is None and clines is None: 
     # If both clevs and clines are None, use default
     axes.contourf(x, y, z, **kwargs)

  if not clevs is None:
     axes.contourf(x, y, z, clevs, **kwargs)
     # Special case; if plotting both filled and unfilled contours
     # with a single call, set the color of the latter to black
     kwargs['colors'] = 'k'
  if not clines is None:
     axes.contour(x, y, z, clines, **kwargs)

  # Apply the custom axes args
  set_xaxis(axes, X)
  set_yaxis(axes, Y)
  axes.setp(title = _buildvartitle(var.axes, var.name, **var.plotatts))

  return axes
# }}}
