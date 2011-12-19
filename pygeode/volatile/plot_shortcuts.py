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
    xlabel = var.plotatts.get('xlabel',var.axes[0].name)
    ylabel = var.name
    xlim = min(var.axes[0].values), max(var.axes[0].values)
    xscale = var.axes[0].plotatts.get('plotscale','linear')
    yscale = var.plotatts.get('plotscale','linear')
    plotorder = var.axes[0].plotatts.get('plotorder',1)
    xlim = xlim[::plotorder]
    del plotorder

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

  return axes, var


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

    # Get the default axes args
    axes, var = get_axes_args(var)

    # Apply the custom axes args
    axes = axes.modify(**axes_args)

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


# Do a 1D line plot
# Assumes the X coordinate is provided by the Var, not in the parameter list
def plot (*args, **kwargs):
  from plot_wrapper import Plot, split_axes_args

  outargs = []
  i = 0
  while i < len(args):

    Y = args[i]
    X = Y.squeeze().axes[0]
    i += 1

    axes, Y = get_axes_args(Y)

    # Special case: have a vertical coordinate
    # (transpose the plot)
    from pygeode.axis import ZAxis
    from plot_wrapper import Axes
    if isinstance(X, ZAxis):
      X, Y = Y, X
      axes_args = {}
      # Change the role of 'x' and 'y' attributes
      for k, v in axes.args.iteritems():
        if k.startswith('x'): k = 'y'+k[1:]
        elif k.startswith('y'): k = 'x'+k[1:]
        axes_args[k] = v
      axes = Axes(**axes_args)

    Y = Y.get()
    X = X.get()
    outargs.extend([X,Y])

    # Have a format?
    if i < len(args) and isinstance(args[i],str):
      outargs.append(args[i])
      i += 1

  # Split out axes args and plot args
  axes_args, plot_args = split_axes_args(kwargs)

  # Apply the custom axes args
  axes = axes.modify(**axes_args)


  return Plot(*outargs, axes=axes, **plot_args)


# Do many 1D line plots
# (A higher-level extension, no analogue in matplotlib)
def spaghetti (var, data_axis, **kwargs):
  from pygeode.var import Var

  data_axis = var.whichaxis(data_axis)

  # Get the raw data
  data = var.get()

  # Transpose so the data axis is the fastest-varying axis
  order = range(0,data_axis) + range(data_axis+1,var.naxes) + [data_axis]
  data = data.transpose(order)

  # Flatten out the other dimensions
  data = data.reshape(-1, data.shape[-1])

  # Tear apart the data, into individual strands
  data = list(data)

  # Construct individual Vars from this data
  data = [Var([var.axes[data_axis]], values=d) for d in data]

  # Do the plot
  return plot (*data, **kwargs)



# Do a quiver plot
def quiver (u, v, **kwargs):
  from plot_wrapper import Quiver, split_axes_args
  import numpy as np
  # Filter the vars
  u_axes, u = get_axes_args(u)
  v_axes, v = get_axes_args(v)

  # Get a proper title
  dummy = u.rename(u.name+','+v.name)
  axes, dummy = get_axes_args(dummy)

  # Get the data
  X, Y, u = get_XYC(u)
  X2, Y2, v = get_XYC(v)

  assert np.allclose(X,X2), "U/V domain mismatch"
  assert np.allclose(Y,Y2), "U/V domain mismatch"

  # Apply custom arguments
  axes_args, plot_args = split_axes_args(kwargs)
  axes = axes.modify(**axes_args)

  return Quiver (X, Y, u, v, axes=axes, **plot_args)

