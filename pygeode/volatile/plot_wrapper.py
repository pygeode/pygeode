# Wrapper for matplotlib.pyplot
# Make plots picklable (pickleable? pickle-able?)
# Makes it so you can pickle them.


# Axes container
# (Just contains a dictionary of init args)
class Axes:
  def __init__(self, **axes_args):
    self.args = axes_args

  # Method to override with other axes args
  def modify (self, **axes_args):
    axes_args = dict(self.args, **axes_args)
    return Axes(**axes_args)

  # Method for removing certain axes args
  def remove (self, *remove_arg_list):
    args = dict(self.args)
    for remove_arg in remove_arg_list:
      args.pop(remove_arg,None)
    return Axes(**args)


# Helper function
# Given a dictionary of keyword parameters, split it into Axes and non-Axes
# dictionaries of keywords.
def split_axes_args (all_args):
  import matplotlib.pyplot as pl
  axes_args = {}
  other_args = {}
  for argname, argval in all_args.items():
    if hasattr(pl.Axes, 'set_'+argname):
      axes_args[argname] = argval
    else:
      other_args[argname] = argval
  return axes_args, other_args

# The default coordinate transform
# (Doesn't do any tranformation)
def notransform (*inputs): return inputs

# Generic object for holding plot information
class PlotWrapper:
  def __init__(self, *plot_args, **plot_kwargs):
    axes = plot_kwargs.pop('axes', None)
    plot_kwargs.pop('figure', None)
    if axes is None:
      axes = Axes()
    assert isinstance(axes, Axes)
    self.plot_args = plot_args
    self.plot_kwargs = plot_kwargs
    self.default_axes = axes

  # Draw the thing
  # This is pretty much the only public-facing method.
  def render (self, figure=None):
    import matplotlib.pyplot as pl
    if figure is None:
      figure = pl.figure()
    axes = pl.subplot(111)
    self._doplot(figure, pl, axes, transform=notransform)
    # Set the axes properties
    self._apply_axes (axes)

  # Apply the axes stuff
  def _apply_axes (self, axes):
    for k, v in self.default_axes.args.items():
      getattr(axes,'set_'+k)(v)

  # Routine for doing the actual plot
  # Arguments:
  #  figure: the figure to draw into
  #  pl: the plotting package (probably matplotlib.pyplot, but could be
  #      something else, like Basemap)
  #  axes: the axes to do the plot on
  #  transform: a coordinate transformation.  Needed for rendering data
  #          on map projections with Basemap.
  #          Not sure why Basemap can't take care of that step for us, but
  #          that's how their package is designed...
  def _doplot (self, figure, pl, axes, transform):
    return  # Nothing to plot in this generic wrapper!
            # (this should never be called)

# 1D plot
class Plot(PlotWrapper):
  def _doplot (self, figure, pl, axes, transform):
    pl.plot (*self.plot_args, **self.plot_kwargs)


# Both contour-type plots (contour/contourf) use the same input argument
# conventions, so only need to define the transformation method once.
class ContourType(PlotWrapper):

  # Transformation of the coordinates
  @staticmethod
  def _transform(inputs, transform):
    from numpy import meshgrid
    from warnings import warn
    # Z
    if len(inputs) == 1: return inputs
    # X, Y, Z
    if len(inputs) == 3:
      X, Y, Z = inputs
      X, Y = meshgrid(X, Y)
      X, Y = transform(X, Y)
      return X, Y, Z
    # Z, N
    if len(inputs) == 2 and isinstance(inputs[1],int): return inputs
    # X, Y, Z, N
    if len(inputs) == 4 and isinstance(inputs[3],int):
      X, Y, Z, N = inputs
      X, Y = meshgrid(X, Y)
      X, Y = transform(X, Y)
      return X, Y, Z, N
    #TODO: finish the rest of the cases
    warn("don't know what to do for the coordinate transformation")
    return inputs

  def _doplot (self, figure, pl, axes, transform):
    # Coordinate transformation?
    inputs = self._transform(self.plot_args, transform)
    plotfcn = getattr(pl,self.plotfcn)
    return plotfcn(*inputs, axes=axes, **self.plot_kwargs)

# Contour plot
class Contour(ContourType):
  plotfcn = 'contour'

# Filled contour plot
class Contourf(ContourType):
  plotfnc = 'contourf'


# Pseudo-colour plot
class Pcolor(PlotWrapper):
  def _doplot (self, figure, pl, axes, transform):
    # Coordinate transformation?
    inputs = self._transform(self.plot_args, transform)
    return pl.pcolor(*inputs, axes=axes, **self.plot_kwargs)

  # Transformation of the coordinates
  @staticmethod
  def _transform(inputs, transform):
    from numpy import meshgrid
    from warnings import warn
    # C
    if len(inputs) == 1: return inputs
    # X, Y, C
    if len(inputs) == 3:
      X, Y, C = inputs
      X, Y = meshgrid(X, Y)
      X, Y = transform(X, Y)
      return X, Y, C

    warn("don't know what to do for the coordinate transformation")
    return inputs


# A quiver plot
class Quiver(PlotWrapper):
  @staticmethod
  def _transform (inputs, transform):
    from numpy import meshgrid
    # U, V
    if len(inputs) == 2: return inputs
    # U, V, C
    if len(inputs) == 3: return inputs
    # X, Y, U, V
    if len(inputs) == 4:
      X, Y, U, V = inputs
      X, Y = meshgrid(X, Y)
      X, Y = transform(X, Y)
      return X, Y, U, V
    if len(inputs) == 5:
      X, Y, U, V, C = inputs
      X, Y = meshgrid(X, Y)
      X, Y = transform(X, Y)
      return X, Y, U, V, C

  def _doplot (self, figure, pl, axes, transform):
    # Coordinate transformation?
    inputs = self._transform(self.plot_args, transform)
    return pl.quiver(*inputs, axes=axes, **self.plot_kwargs)

# A quiver key
class QuiverKey(PlotWrapper):
  def __init__ (self, plot, *plot_args, **plot_kwargs):
    self.plot = plot
    self.plot_args = plot_args
    self.plot_kwargs = plot_kwargs
  def _doplot (self, figure, pl, axes, transform):
    theplot = self.plot._doplot(figure,pl,axes,transform)
    # Ignore 'pl' argument, use matplotlib.pyplot exclusively
    # (Basemap doesn't have a quiverkey method)
    import matplotlib.pyplot as pl
    return pl.quiverkey(theplot, *self.plot_args, **self.plot_kwargs)


# Colorbar
# Treated as a plot-type thing.
# Use it in an overlay; will use the values from the plot 'underneath' it.
class Colorbar(PlotWrapper):
  def __init__ (self, plot, cax=None, ax=None, mappable=None, **kwargs):
    self.plot = plot
    # Ignore cax/ax/mappable arguments
    self.cbar_kwargs = kwargs
    self.default_axes = plot.default_axes
  def _doplot (self, figure, pl, axes, transform):
    theplot = self.plot._doplot(figure,pl,axes,transform)
    return figure.colorbar (theplot, ax=axes, **self.cbar_kwargs)


# Overlay object
# Similar to Plot, but renders a bunch of things in order.
# Plots must be pre-created
class Overlay(PlotWrapper):
  def __init__ (self, *plots, **axes_args):
    self.plots = plots
    # Create an Axes from the first plot, and additional keyword arguments
    self.default_axes = plots[0].default_axes.modify(**axes_args)

  def _doplot (self, figure, pl, axes, transform):
    # Loop over all plots, and render them on top of each other
    for plot in self.plots:
      p = plot._doplot(figure=figure, axes=axes, pl=pl, transform=transform)
      axes.hold(True)
    return p

# Multiplot
# (more than one plot in a figure)
class Multiplot:
  def __init__ (self, plots):
    self.plots = plots

  def render (self, figure=None):
    import matplotlib.pyplot as pl
    if figure is None:
      figure = pl.figure()
    nrows = len(self.plots)
    for i, row in enumerate(self.plots):
      ncols = len(row)
      for j, plot in enumerate(row):
        axes = pl.subplot(nrows, ncols, i*ncols + j + 1)
        # Do the plot (assume no coordinate transformation, unless otherwise overridden)
        plot._doplot(figure=figure, axes=axes, pl=pl, transform=notransform)
        # Apply the axes decorations
        plot._apply_axes(axes)


# Helper for Basemap wrapper
# defines a decorator for drawing map elements on the plot
def drawing (f):
  from functools import wraps
  @wraps(f)
  def g (self, *args, **kwargs):
    # Don't allow the axes object to be overridden
    kwargs.pop('ax',None)
    # Save this call
    self.map_stuff.append([f.__name__, args, kwargs])
  return g

# Basemap wrapper
# Takes a plot object, wraps it so it appears within a map projection
class Map(PlotWrapper):
  def __init__ (self, plot, **kwargs):
    axes_args, basemap_args = split_axes_args(kwargs)

    self.plot = plot
    self.basemap_args = basemap_args

    # Start with the axes decorators from the original plot
    default_axes = plot.default_axes.modify(**axes_args)

    # Remove some arguments which no longer make sense
    default_axes = default_axes.remove('xlim','ylim')

    # Remove axes labels on circular-type projections
    for prefix in 'aeqd', 'gnom', 'ortho', 'geos', 'nsper', 'npstere', 'spstere', 'nplaea', 'splaea', 'npaeqd', 'spaeqd':
      if basemap_args.get('projection','cyl').startswith(prefix):
        default_axes = default_axes.remove('xlabel','ylabel')

    self.default_axes = default_axes

    # Start with no extra map stuff
    self.map_stuff = []

  def _doplot (self, figure, pl, axes, transform):
    from warnings import warn
    # Try setting up the map
    try:
      from mpl_toolkits.basemap import Basemap
      pl = Basemap(ax=axes, **self.basemap_args)
      transform = pl
      # Draw map-related stuff
      for fcn, args, kwargs in self.map_stuff:
        fcn = getattr(pl,fcn)
        fcn(*args, **kwargs)
    except ImportError:
      warn ("Can't import Basemap", stacklevel=2)
      import matplotlib.pyplot as pl
      transform = lambda x, y: (x, y)
    # Apply the plot to this modified field
    return self.plot._doplot (figure=figure, axes=axes, pl=pl, transform=transform)


# Impement the drawing of map elements (boundaries, meridians, etc.)

for fcn in 'drawcoastlines', 'drawcountries', 'drawgreatcircle', 'drawlsmask', 'drawmapboundary', 'drawmapscale', 'drawmeridians', 'drawparallels', 'drawrivers', 'drawstates', 'etopo', 'fillcontinents', 'drawmapscale', 'nightshade', 'shadedrelief', 'tissot', 'warpimage':
  def F(fcn_name=fcn):
    def f(self, *args, **kwargs):
      # Don't allow the axes object to be overridden
      kwargs.pop('ax',None)
      # Save this call
      self.map_stuff.append([fcn_name, args, kwargs])
    f.__name__ = fcn_name
    return f

  setattr(Map,fcn,F(fcn))
  del F

del fcn

