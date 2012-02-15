# Wrapper for matplotlib.pyplot
# Allows plots to be constructed in a slightly more object-oriented way.
# Also allows plot objects to be saved to / loaded from files, something which
# normal matplotlib plots can't do.


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

  # Some extra tweaks
  # Don't treat 'label' as an axes arg.
  # Let it be a plot arg, so we get labels applied early enough that they
  # can be picked up by a legend object.
  # (If it was an axes arg, it won't get applied until the very end, when the
  # axes get decorated - too late to be of use in a legend.)
  if 'label' in axes_args:
    other_args['label'] = axes_args.pop('label')
  return axes_args, other_args

# The default coordinate transform
# (Doesn't do any tranformation)
def notransform (*inputs): return inputs

# Generic object for holding plot information
class PlotWrapper:
  def __init__(self, *plot_args, **kwargs):
    axes_args, plot_kwargs = split_axes_args(kwargs)
    self.axes_args = axes_args
    plot_kwargs.pop('figure', None)
    self.plot_args = plot_args
    self.plot_kwargs = plot_kwargs

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
    args = dict(self.axes_args)
    # Handle scaling first, because setting this screws up other custom attributes like ticks
    if 'xscale' in args: axes.set_xscale(args.pop('xscale'))
    if 'yscale' in args: axes.set_yscale(args.pop('yscale'))
    # Apply the rest of the axes attributes
    for k, v in args.items():
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

  # Routine for saving this plot to a file
  def save (self, filename):
    import pickle
    outfile = open(filename,'w')
    pickle.dump(self, outfile)
    outfile.close()

# Module-level routine for loading a plot from file
def load (filename):
  import pickle
  infile = open(filename,'ro')
  theplot = pickle.load(infile)
  infile.close()
  return theplot



# 1D plot
class Plot(PlotWrapper):
  def _doplot (self, figure, pl, axes, transform):
    pl.plot (*self.plot_args, **self.plot_kwargs)

# Errorbar plot
class ErrorBar(PlotWrapper):
  def _doplot (self, figure, pl, axes, transform):
    pl.errorbar(*self.plot_args, **self.plot_kwargs)

# Both contour-type plots (contour/contourf) use the same input argument
# conventions, so only need to define the transformation method once.
class ContourType(PlotWrapper):

  # Transformation of the coordinates
  @staticmethod
  def _transform(inputs, transform):
    from numpy import meshgrid, ndarray
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
    if len(inputs) == 2 and isinstance(inputs[1],(int,list,tuple,ndarray)): return inputs
    # X, Y, Z, N
    if len(inputs) == 4 and isinstance(inputs[3],(int,list,tuple,ndarray)):
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
  plotfcn = 'contourf'


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
    warn("don't know what to do for the coordinate transformation")
    return inputs

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
    self.axes_args = plot.axes_args
  def _doplot (self, figure, pl, axes, transform):
    theplot = self.plot._doplot(figure,pl,axes,transform)
    # Ignore 'pl' argument, use matplotlib.pyplot exclusively
    # (Basemap doesn't have a quiverkey method)
    import matplotlib.pyplot as pl
    return pl.quiverkey(theplot, *self.plot_args, **self.plot_kwargs)


# Colorbar
# Wraps an existing plot, from which the data values will be retrieved.
class Colorbar(PlotWrapper):
  def __init__ (self, plot, cax=None, ax=None, mappable=None, **kwargs):
    self.plot = plot
    # Ignore cax/ax/mappable arguments
    self.cbar_kwargs = kwargs
    self.axes_args = plot.axes_args
  def _doplot (self, figure, pl, axes, transform):
    theplot = self.plot._doplot(figure,pl,axes,transform)
    return figure.colorbar (theplot, ax=axes, **self.cbar_kwargs)


# A legend
# (labels the lines of a line plot)
class Legend(PlotWrapper):
  def __init__ (self, plot, *args, **kwargs):
    self.plot = plot
    self.axes_args = plot.axes_args
    self.legend_args = args
    self.legend_kwargs = kwargs
  def _doplot (self, figure, pl, axes, transform):
    theplot = self.plot._doplot(figure,pl,axes,transform)
    return axes.legend (*self.legend_args, **self.legend_kwargs)

# Overlay object
# Defines a sequence of plots, drawn consecutively on top of each other.
class Overlay(PlotWrapper):
  def __init__ (self, *plots, **axes_args):
    self.plots = plots
    # Create an Axes from the first plot, and additional keyword arguments
    self.axes_args = dict(plots[0].axes_args, **axes_args)

  def _doplot (self, figure, pl, axes, transform):
    # Loop over all plots, and render them on top of each other
    for plot in self.plots:
      p = plot._doplot(figure=figure, axes=axes, pl=pl, transform=transform)
      axes.hold(True)
    return p

# Multiplot
# Tiles a bunch of plots together in the same figure.
# Plots are passed in a nested list, representing a 2D grid.
# E.g., to tile plots A, B, C, D together so they look like
#   A B
#   C D
# then you would do something like
# myplot = Multiplot( [[A,B],[C,D]] )
#
# Note:
#   This plot object can't be further wrapped (e.g. Overlay, Colorbar,...).
#   It must be the outer-most (last) operation applied to the plots of a figure.
class Multiplot (PlotWrapper):
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

  def _apply_axes (self, axes):
    raise NotImplementedError,"Multiplot can't be embedded in other plot objects"

  def _doplot (self, figure, pl, axes, transform):
    raise NotImplementedError,"Multiplot can't be embedded in other plot objects"




# Basemap wrapper
# Takes a plot object, wraps it so it appears within a map projection
class Map(PlotWrapper):
  def __init__ (self, plot, **kwargs):

    axes_args, basemap_args = split_axes_args(kwargs)

    self.plot = plot

    # Start with the axes decorators from the original plot
    if plot is not None:
      axes_args = dict(plot.axes_args, **axes_args)
    else:
      axes_args = dict()

    # Remove some arguments which no longer make sense
    # These will cause basemap to choke
    axes_args.pop('xlim',None)
    axes_args.pop('ylim',None)
    axes_args.pop('xscale',None)
    axes_args.pop('yscale',None)

    self.axes_args = axes_args
    self.basemap_args = basemap_args

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
      transform = notransform
    # Apply the plot to this modified field
    if self.plot is not None:
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

