# Wrapper for matplotlib.pyplot
import matplotlib.pyplot as pyl

# Allows plots to be constructed in a slightly more object-oriented way.
# Also allows plot objects to be saved to / loaded from files, something which
# normal matplotlib plots can't do.

# Helper function
# Given a dictionary of keyword parameters, extract those that need to go into
# the call to pyl.figure()
def split_fig_args (**kwargs):
  fig_props = {}
  fig_args = {}
  for k, v in kwargs.items():
    if k in ['num', 'figsize', 'dpi']:
      fig_props[k] = v
    else:
      fig_args[k] = v

  return fig_args, fig_props

# Interface for wrapping a matplotlib figure
class FigureWrapper:
  def __init__(self, *args, **kwargs):
    # Should extract anything that needs to go into the calling command
    self.fig_args = args
    self.fig_props = kwargs
    self.naxes = 0
    self.axes = []
    self.axes_rects = []
    self.axes_args = []

  def add_axis(self, axis, rect, **kwargs):
    self.axes.append(axis)
    self.axes_rects.append(rect)
    self.axes_args.append(kwargs)
    self.naxes = len(self.axes)

  def _do_layout(self):
    axes = []
    for r, a in zip(self.axes_rects, self.axes_args):
      axes.append(self.fig.add_axes(r, **a))
            
    return axes

  def render(self):
    pyl.ioff()

    self.fig = pyl.figure(*self.fig_args)
    self.fig.clf()
    pyl.setp(self.fig, **self.fig_props)

    axes = self._do_layout()

    for pl_ax, ax in zip(axes, self.axes):
      ax.render(pl_ax)

    pyl.ion()
    pyl.draw()
    pyl.show()

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

# Interface for wrapping a matplotlib axes object
class AxesWrapper:
  def __init__(self, **kwargs):
    self.axes_args = kwargs
    self.nplots = 0
    self.plots = []

  def add_plot(self, plot, order=None):
    if order is None:
      self.plots.append(plot)
    else:
      self.plots.insert(plot, order)
    self.nplots = len(self.plots)

  def render(self, axes):
    # Perform plotting operations
    for p in self.plots:
      p.render(axes)

    # Handle scaling first, because setting this screws up other custom attributes like ticks
    args = self.axes_args.copy()
    if 'xscale' in args: axes.set_xscale(args.pop('xscale'))
    if 'yscale' in args: axes.set_yscale(args.pop('yscale'))

    # Apply the rest of the axes attributes
    #pyl.setp(axes, **args)
    
# Generic object for holding plot information
class PlotWrapper:
  def __init__(self, *plot_args, **kwargs):
    self.plot_args = plot_args
    self.plot_kwargs = kwargs

  # Draw the thing
  # This is pretty much the only public-facing method.
  def render (self, axes=None):
    pass

# 1D plot
class Plot(PlotWrapper):
  def render (self, axes):
    pyl.plot (*self.plot_args, **self.plot_kwargs)


