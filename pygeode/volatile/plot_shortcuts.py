# Shortcuts for plotting PyGeode vars
# Extends plot_wrapper to automatically use information from the Pygeode Vars.

# Set up arguments for axis
def get_axes_args (var):

  title = var.plotatts.get('plottitle', None)
  if title is None: title = var.name

  #TODO: put degenerate PyGeode axis info into title,
  # strip the degenerate axes out of the data.

  # 1D stuff
  if var.naxes == 1:
    xlabel = var.plotatts.get('xlabel',var.axes[0].name)
    ylabel = title
    xlim = min(var.axes[0].values), max(var.axes[0].values)

  # 2D stuff
  if var.naxes == 2:
    # For 2D plots, the x/y need to be switched?
    xlabel = var.plotatts.get('xlabel',var.axes[1].name)
    ylabel = var.plotatts.get('ylabel',var.axes[0].name)
    xlim = min(var.axes[1].values), max(var.axes[1].values)
    ylim = min(var.axes[0].values), max(var.axes[0].values)

  #TODO: linear/log scale, reversed order

  # Collect these local variables into a dictionary
  # (these will be keyword parameters for constructing an Axes)
  axes_args = dict(locals())
  del axes_args['var']

  # Create an Axes wrapper with this information
  from plot_wrapper import Axes
  axes = Axes(**axes_args)

  return axes


# Get 2D data
# (note: notion of 'x' and 'y' are switched for matplotlib 2D plots...)
def get_XYC (var):
  X = var.axes[1].get()
  Y = var.axes[0].get()
  C = var.get()

  # Special case: we have regular longitudes on a global grid.
  # Add a repeated longitude for this data
  from pygeode.axis import Lon
  import numpy as np
  if isinstance(var.axes[1], Lon):
    dlon = np.diff(X)
    if np.allclose(dlon, dlon[0]):
      dlon = dlon[0]
      firstlon = X[0] % 360.
      lastlon = (X[-1] + dlon) % 360.
      if np.allclose(lastlon,360.): lastlon = 0.
      if np.allclose(firstlon,lastlon):
        # Add the extra longitude
        X = np.concatenate([X, [X[0]+360.]])
        C = np.concatenate([C, C[:,0:1]], axis=1)

  return X, Y, C

# A decorator for a plot maker
# (does the work of setting up the generic Axes info)
def plot_maker (f):
  def g (var, **kwargs):
    from plot_wrapper import split_axes_args
    # Separate out the plot arguments from the Axes arguments
    axes_args, plot_args = split_axes_args(kwargs)
    axes = get_axes_args(var).modify(**axes_args)
    return f(var, axes, **plot_args)
  g.__name__ = f.__name__
  return g


# Do a contour plot
@plot_maker
def contour (var, axes, **options):
  from plot_wrapper import Contour
  X, Y, C = get_XYC(var)
  return Contour(X, Y, C, axes=axes, **options)

# Do a filled contour plot
@plot_maker
def contourf (var, axes, **options):
  from plot_wrapper import Contourf
  X, Y, C = get_XYC(var)
  return Contourf(X, Y, C, axes=axes, **options)

# Do a pseudocolor plot
@plot_maker
def pcolor (var, axes, **options):
  from plot_wrapper import Pcolor
  X, Y, C = get_XYC(var)
  return Pcolor (X, Y, C, axes=axes, **options)


