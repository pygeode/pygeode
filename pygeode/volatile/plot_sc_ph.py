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

# Do a 1D line plot
# Assumes the X coordinate is provided by the Var, not in the parameter list
def plot (axis, var, fmt='', **kwargs):
  from plot_wr_ph import Plot, _buildvartitle
  render = False

  if axis is None:
    from plot_wr_ph import SingleFigure
    
    # Build Figure object
    fig = SingleFigure(1)
    axis = fig.axes[0]
    render = True

  # Special case: have a vertical coordinate
  # (transpose the plot)
  Y = var.squeeze()
  X = Y.axes[0]
  from pygeode.axis import ZAxis
  if isinstance(X, ZAxis):
    X, Y = Y, X

  # Apply the custom axes args
  axis.set_haxis(X)
  axis.set_vaxis(Y)
  axis.set(title = _buildvartitle(var.axes, var.name, **var.plotatts))

  X = X.get()
  Y = Y.get()

  axis.add_plot(Plot(X, Y, fmt, **kwargs))

  if render:
    fig.render()

  return fig

# Do a 2D contour plot
def contour (axis, var, *args, **kwargs):
  from plot_wr_ph import Contour, _buildvartitle
  render = False

  if axis is None:
    from plot_wr_ph import SingleFigure
    
    # Build Figure object
    fig = SingleFigure(1)
    axis = fig.axes[0]
    render = True

  Z = var.squeeze()
  assert Z.naxes == 2, 'Variable to contour must have two dimensions'

  Y, X = Z.axes
  from pygeode.axis import ZAxis
  if isinstance(X, ZAxis):
    X, Y = Y, X

  # Apply the custom axes args
  axis.set_haxis(X)
  axis.set_vaxis(Y)
  axis.set(title = _buildvartitle(var.axes, var.name, **var.plotatts))

  x = X.get()
  y = Y.get()
  z = Z.transpose(Y, X).get()

  axis.add_plot(Contour(x, y, z, *args, **kwargs))

  if render:
    fig.render()

  return fig
