# Wrapper for matplotlib.pyplot
# Make plots picklable (pickleable? pickle-able?)
# Makes it so you can pickle them.

#TODO: move axes_args into a dict parameter (not a bunch of keyword parameters).  Keyword parameters should be reserved for any plot-specific arguments.

#TODO: make sure no stray fig/axes arguments get passed to the wrappers

# Axes container
# (Just contains a dictionary of init args)
class Axes:
  def __init__(self, **axes_args):
    self.args = axes_args

  # Method to override with other axes args
  def modify (self, **axes_args):
    axes_args = dict(self.args, **axes_args)
    return Axes(**axes_args)


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

  def render (self, axes=None, hold=False, pl=None):
    if pl is None:
      import matplotlib.pyplot as pl
    if axes is None:
      axes = pl.subplot(111)
    self._doplot(pl, axes)
    # Set the axes properties
    for k, v in self.default_axes.args.items():
      getattr(axes,'set_'+k)(v)
    axes.hold(hold)
    return

  # Routine for doing the actual plot
  def _doplot (self, pl, axes):  #TODO mapper arg
    return  # Nothing to plot in this generic wrapper!
            # (this should never be called)


# Both contour-type plots (contour/contourf) use the same input argument
# conventions, so only need to define the transformation method once.
class ContourType(PlotWrapper):

  # Transformation of the coordinates
  @staticmethod
  def _transform(inputs, mapper):
    from warnings import warn
    # Z
    if len(inputs) == 1: return inputs
    # X, Y, Z
    if len(inputs) == 3:
      X, Y, Z = inputs
      X, Y = mapper(X, Y)
      return X, Y, Z
    # Z, N
    if len(inputs) == 2 and isinstance(inputs[1],int): return inputs
    # X, Y, Z, N
    if len(inputs) == 4 and isinstance(inputs[3],int):
      X, Y, Z, N = inputs
      X, Y = mapper(X, Y)
      return X, Y, Z, N
    #TODO: finish the rest of the cases
    warn("don't know what to do for the coordinate transformation")
    return inputs

# Contour plot
class Contour(ContourType):
  def _doplot (self, pl, axes):
    pl.contour(*self.plot_args, axes=axes, **self.plot_kwargs)

# Filled contour plot
class Contourf(ContourType):
  def _doplot (self, pl, axes):
    pl.contourf(*self.plot_args, axes=axes, **self.plot_kwargs)


# Pseudo-colour plot
class Pcolor(PlotWrapper):
  def _doplot (self, pl, axes):
    pl.pcolor(*self.plot_args, axes=axes, **self.plot_kwargs)

  # Transformation of the coordinates
  @staticmethod
  def _transform(inputs, mapper):
    from warnings import warn
    # C
    if len(inputs) == 1: return inputs
    # X, Y, C
    if len(inputs) == 3:
      X, Y, C = inputs
      X, Y = mapper(X, Y)
      return X, Y, C

    warn("don't know what to do for the coordinate transformation")
    return inputs


# Colorbar
# Treated as a plot-type thing.
# Use it in an overlay; will use the values from the plot 'underneath' it.
class Colorbar(PlotWrapper):
  def __init__ (self, cax=None, ax=None, mappable=None, **kwargs):
    # Ignore cax/ax/mappable arguments
    self.cbar_kwargs = kwargs
    self.default_axes = Axes()  # Don't need any special axes configuration
  def _doplot (self, pl, axes):
    pl.colorbar (ax=axes, **self.cbar_kwargs)

# Overlay object
# Similar to Plot, but renders a bunch of things in order.
# Plots must be pre-created
class Overlay(PlotWrapper):
  def __init__ (self, *plots, **axes_args):
    self.plots = plots
    # Create an Axes from the first plot, and additional keyword arguments
    self.default_axes = plots[0].default_axes.modify(**axes_args)

  def _doplot (self, pl, axes):
    # Loop over all plots, and render them on top of each other
    for plot in self.plots:
      plot.render(axes=axes, hold=True, pl=pl)


# Multiplot
# (more than one plot in a figure)
class Multiplot:
  def __init__ (self, plots):
    self.plots = plots

  def render (self):
    import matplotlib.pyplot as pl
    nrows = len(self.plots)
    for i, row in enumerate(self.plots):
      ncols = len(row)
      for j, plot in enumerate(row):
        axes = pl.subplot(nrows, ncols, i*ncols + j + 1)
        plot.render(axes=axes)


# Basemap wrapper
# Takes a plot object, wraps it so it appears within a map projection
class Map:
  def __init__ (plot, **basemap_args):
    self.plot = plot
    self.basemap_args = basemap_args
  def render (self, axes=None):
    from warnings import warn
    # Try setting up the map
    try:
      from mpl_toolkits.basemap import Basemap
      pl = Basemap(axes=axes, **self.basemap_args)
      mapper = pl
    except ImportError:
      warn ("Can't import Basemap", stacklevel=2)
      import matplotlib.pyplot as pl
      mapper = lambda x, y: x, y
    # Apply the plot to this modified field
    self.plot.render (axes=axes, pl=pl)


