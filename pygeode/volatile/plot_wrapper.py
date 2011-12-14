# Wrapper for matplotlib.pyplot
# Make plots picklable (pickleable? pickle-able?)
# Makes it so you can pickle them.


# Generic object for holding plot information
class Plot:
  def __init__(self, plot_type, inputs, **axes_args):
    self.plot_type = plot_type
    self.inputs = inputs
    self.axes_args = axes_args

  def render (self, axes=None, hold=False, pl=None):
    if pl is None:
      import matplotlib.pyplot as pl
    if axes is None:
      axes = pl.subplot(111, **self.axes_args)
    plotfunc = getattr(pl,self.plot_type)
    result = plotfunc (*self.inputs, axes=axes)
    axes.hold(hold)
    #pl.colorbar(ax=axes)
    return

# Colorbar
# Treated as a plot-type thing.
# Use it in an overlay; will use the values from the plot 'underneath' it.
class Colorbar:
  def render (self, axes=None, hold=False, pl=None):
    if pl is None:
      import matplotlib.pyplot as pl
    pl.colorbar (ax=axes)
    return

# Overlay object
# Similar to Plot, but renders a bunch of things in order.
# Plots must be pre-created
class Overlay(Plot):
  def __init__ (self, *plots, **axes_args_overrides):
    self.plots = plots
    self.axes_args = dict(plots[0].axes_args, **axes_args_overrides)
  def render (self, axes=None, hold=False, pl=None):
    if pl is None:
      import matplotlib.pyplot as pl
    # If no existing axes provided, use the ones from the first plot
    if axes is None:
      axes = pl.subplot(111, **self.axes_args)
    # Loop over all plots, and render them on top of each other
    for plot in self.plots:
      plot.render(axes=axes, hold=True)

    axes.hold(hold)  # Do we need to hold for additional plot stuff outside here?
    return

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
        axes = pl.subplot(nrows, ncols, i*ncols + j + 1, **plot.axes_args)
        plot.render(axes=axes)


"""
# Basemap wrapper
# Takes a plot object, wraps it so it appears within a map projection
class Map:
  # TODO: other projections
  def __init__ (plot):
    self.plot = plot
  def render (self, axes=None):
"""



