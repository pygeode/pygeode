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

def _buildaxistitle(name = '', plotname = '', plottitle = '', plotunits = '', **dummy):
# {{{
  if name is None: name = ''
  if plotname is None: plotname = ''
  if plottitle is None: plottitle = ''
  if plotunits is None: plotunits = ''

  assert type(plotname) is str
  assert type(plottitle) is str
  assert type(plotunits) is str
  assert type(name) is str
  
  if plotname is not '': title = plotname # plotname is shorter, hence more suitable for axes
  elif plottitle is not '': title = plottitle
  elif name is not '': title = name
  else: title = ''

  if plotunits is not '': title += ' [%s]' % plotunits

  return title
# }}}

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
    self._haxis = None
    self._vaxis = None

  def add_plot(self, plot, order=None):
    if order is None:
      self.plots.append(plot)
    else:
      self.plots.insert(plot, order)
    self.nplots = len(self.plots)

  def set_haxis(self, axis):
    vals = axis.get()
    lims = min(vals), max(vals)
    ax_args = dict(
        xscale = axis.plotatts.get('plotscale', 'linear'),
        xlabel = _buildaxistitle(**axis.plotatts),
        xlim = lims[::axis.plotatts['plotorder']])
    self.axes_args.update(ax_args)
    self._haxis = axis

  def set_vaxis(self, axis):
    vals = axis.get()
    lims = min(vals), max(vals)
    ax_args = dict(
        yscale = axis.plotatts.get('plotscale', 'linear'),
        ylabel = _buildaxistitle(**axis.plotatts),
        ylim = lims[::axis.plotatts['plotorder']])
    self.axes_args.update(ax_args)
    self._vaxis = axis

  def set(self, **kwargs):
    self.axes_args.update(kwargs)

  def render(self, axes):
    # Perform plotting operations
    for p in self.plots:
      p.render(axes)

    # Handle scaling first, because setting this screws up other custom attributes like ticks
    args = self.axes_args.copy()
    if 'xscale' in args: axes.set_xscale(args.pop('xscale'))
    if 'yscale' in args: axes.set_yscale(args.pop('yscale'))

    import pygeode as pyg
    if isinstance(self._haxis, pyg.Axis):
      axes.xaxis.set_major_formatter(self._haxis.formatter())
      self._haxis.set_locator(axes.xaxis)

    if isinstance(self._vaxis, pyg.Axis):
      axes.yaxis.set_major_formatter(self._vaxis.formatter())
      self._vaxis.set_locator(axes.yaxis)

    # Apply the rest of the axes attributes
    pyl.setp(axes, **args)

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


class SingleFigure(FigureWrapper):
  def __init__(self, *args, **kwargs):
    FigureWrapper.__init__(self, *args, **kwargs)
    self.axes = [AxesWrapper()]

  def _do_layout(self):
    axis = self.fig.add_subplot(111)
    return [axis]
    
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

# Contour
class Contour(PlotWrapper):
  def render (self, axes):
    self._cnt = pyl.contour (*self.plot_args, **self.plot_kwargs)
