# Shortcuts for plotting PyGeode vars
# Extends plot_wrapper to automatically use information from the Pygeode Vars.

# Set up arguments for axis
def get_axes_args (var):

  title = var.plotatts.get('plottitle', None)
  if title is None: title = var.name

  # Put degenerate PyGeode axis info into title,
  # strip the degenerate axes out of the data.
  for axis in var.axes:
    if len(axis) == 1:
      title += ", %s = %s" % (axis.name, axis.values[0])
  var = var.squeeze()
  del axis

  # 1D stuff
  if var.naxes == 1:
    xaxis = var.axes[0]
    xatts = xaxis.plotatts

    xlabel = var.plotatts.get('xlabel',xaxis.name)
    ylabel = var.name
    xlim = min(xaxis.values), max(xaxis.values)
    xscale = xatts.get('plotscale','linear')
    yscale = var.plotatts.get('plotscale','linear')
    xlim = xlim[::xatts.get('plotorder',1)]

    del xaxis, xatts

  # 2D stuff
  if var.naxes == 2:

    # For 2D plots, the x/y need to be switched?
    xaxis = var.axes[1]
    yaxis = var.axes[0]
    xatts = xaxis.plotatts
    yatts = yaxis.plotatts

    xlabel = xatts.get('plottitle','')
    if xlabel == '': xlabel = var.axes[1].name
    ylabel = yatts.get('plottitle','')
    if ylabel == '': ylabel = var.axes[0].name
    xlim = min(var.axes[1].values), max(var.axes[1].values)
    ylim = min(var.axes[0].values), max(var.axes[0].values)

    # linear/log scale, reversed order
    xscale = xatts.get('plotscale', 'linear')
    yscale = yatts.get('plotscale', 'linear')
    xlim = xlim[::xatts.get('plotorder', 1)]
    ylim = ylim[::yatts.get('plotorder', 1)]

    del xaxis, yaxis, xatts, yatts

  # Collect these local variables into a dictionary
  # (these will be keyword parameters for constructing an Axes)
  axes_args = dict(locals())
  del axes_args['var']

  return axes_args, var

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

# Do a 1D line plot
# Assumes the X coordinate is provided by the Var, not in the parameter list
def plot (axis, var, fmt=None, **kwargs):
  from plot_wr_ph import Plot
  render = False

  if axis is None:
    from plot_wr_ph import SingleFigure
    
    # Build Figure object
    fig = SingleFigure()
    axis = fig.axis
    render = True

  # Special case: have a vertical coordinate
  # (transpose the plot)
  from pygeode.axis import ZAxis
  Y = var.squeeze()
  X = Y.axes[0]
  if isinstance(X, ZAxis):
    X, Y = Y, X

  axis.set_haxis(X)
  axis.set_vaxis(Y)
  axis.set(title = _buildvartitle(var.axes, var.name, **var.plotatts))

  # Apply the custom axes args
  X = X.get()
  Y = Y.get()

  axis.add_plot(Plot(X, Y, fmt, **kwargs))

  if render:
    fig.render()
