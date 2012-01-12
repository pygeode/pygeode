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

  return axes_args, var


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

    # Get the default axes args
    axes_args, var = get_axes_args(var)

    kwargs = dict(axes_args, **kwargs)

    return f(var, **kwargs)

  g.__name__ = f.__name__
  return g


# Do a contour plot
@plot_maker
def contour (var, **kwargs):
  from plot_wrapper import Contour
  X, Y, C = get_XYC(var)
  return Contour(X, Y, C, **kwargs)

# Do a filled contour plot
@plot_maker
def contourf (var, **kwargs):
  from plot_wrapper import Contourf
  X, Y, C = get_XYC(var)
  return Contourf(X, Y, C, **kwargs)

# Do a pseudocolor plot
@plot_maker
def pcolor (var, **kwargs):
  from plot_wrapper import Pcolor
  X, Y, C = get_XYC(var)
  return Pcolor (X, Y, C, **kwargs)


# Helper function
# Transpose the x/y axes parameters
def transpose_axes (axes_args):
  # Start with non-x,y arguments
  new_axes_args = dict(axes_args)
  for k in new_axes_args.keys():
    # Get rid of the x,y arguments
    if k.startswith('x') or k.startswith('y'):
      del new_axes_args[k]

  # Now, go back and scan in the rest of the arguments
  # Change the role of 'x' and 'y' attributes
  for k, v in axes_args.iteritems():

    # Swap the meaning of x and y
    if k.startswith('x'): k = 'y'+k[1:]
    elif k.startswith('y'): k = 'x'+k[1:]
    else: continue

    new_axes_args[k] = v

  return new_axes_args


# Do a 1D line plot
# Assumes the X coordinate is provided by the Var, not in the parameter list
def plot (*args, **kwargs):
  from plot_wrapper import Plot

  outargs = []
  i = 0
  while i < len(args):

    Y = args[i]
    X = Y.squeeze().axes[0]
    i += 1

    axes_args, Y = get_axes_args(Y)

    # Special case: have a vertical coordinate
    # (transpose the plot)
    from pygeode.axis import ZAxis
    if isinstance(X, ZAxis):
      X, Y = Y, X
      axes_args = transpose_axes(axes_args)

    Y = Y.get()
    X = X.get()
    outargs.extend([X,Y])

    # Have a format?
    if i < len(args) and isinstance(args[i],str):
      outargs.append(args[i])
      i += 1


  # Apply the custom axes args
  kwargs = dict(axes_args, **kwargs)


  return Plot(*outargs, **kwargs)


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
  from plot_wrapper import Quiver
  import numpy as np
  # Get a proper title
  dummy = u.rename(u.name+','+v.name)
  axes_args, dummy = get_axes_args(dummy)

  # Filter the vars
  u_axes_args, u = get_axes_args(u)
  v_axes_args, v = get_axes_args(v)

  # Get the data
  X, Y, u = get_XYC(u)
  X2, Y2, v = get_XYC(v)

  assert np.allclose(X,X2), "U/V domain mismatch"
  assert np.allclose(Y,Y2), "U/V domain mismatch"

  # Apply custom arguments
  kwargs = dict(axes_args, **kwargs)

  return Quiver (X, Y, u, v, **kwargs)


# Put the plot onto a decorated map.
# NOTE: this expects an existing PlotWrapper instance, it can't take
# a PyGeode Var as input.
# This is just a convenient shortcut for automatically drawing certain map
# features (coastlines, meridians, etc.).
def Map (plot, **kwargs):
  from plot_wrapper import Map
  result = Map(plot, **kwargs)
  # Modify the parallels/meridians depending on the size of the map region,
  # and the type of projection.
  # (TODO)
  result.drawcoastlines()
  projection = kwargs.get('projection','cyl')
  parallels = [-90,-60,-30,0,30,60,90]
  meridians = [0,60,120,180,240,300,360]
  if projection.startswith('ortho'):
    parallels_labels = [False, False, False, False]
    meridians_labels = [False, False, False, False]
  else:
    parallels_labels = [True, True, False, False]
    meridians_labels = [False, False, False, True]

  result.drawparallels(parallels, labels=parallels_labels)
  result.drawmeridians(meridians, labels=meridians_labels)
  result.drawmapboundary()
  return result
