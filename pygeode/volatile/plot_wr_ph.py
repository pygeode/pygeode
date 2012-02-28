# Wrapper for matplotlib.pyplot
from matplotlib import pyplot as pyl
import matplotlib as mpl

# Allows plots to be constructed in a slightly more object-oriented way.
# Also allows plot objects to be saved to / loaded from files, something which
# normal matplotlib plots can't do.

def grid(axes, size = None):
# {{{
  # Expect a 2d grid; first index is the row, second the column
  ny = len(axes)
  nx = len(axes[0])
  assert all([len(x) == nx for x in axes[1:]]), 'Each row must have the same number of axes'

  rowh = [max([a.size[1] for a in row]) for row in axes]
  colw = [max([axes[i][j].size[0] for i in range(ny)]) for j in range(nx)]

  tsize = [sum(colw), sum(rowh)]
  if size is None: size = tsize

  Ax = AxesWrapper(size = size)

  x, y = 0., 1.
  for i in range(ny):
    for j in range(nx):
      ax = axes[i][j]
      w = ax.size[0] / tsize[0]
      px = (colw[j] - ax.size[0]) / tsize[0] / 2.

      h = ax.size[1] / tsize[1]
      py = (rowh[i] - ax.size[1]) / tsize[1] / 2.
        
      r = [x + px, y - h - py, x + w + px, y - py]
      Ax.add_axis(ax, r)
      x += colw[j] / tsize[0]
    x = 0.
    y -= rowh[i] / tsize[1]

  return Ax
# }}}

# Interface for wrapping a matplotlib axes object
class AxesWrapper:
# {{{
  def __init__(self, parent = None, rect = None, size = None, make_axis=False, **kwargs):
# {{{
    self.parent = parent

    self.nplots = 0
    self.plots = []
    self.make_axis = make_axis

    self.naxes = 0
    self.axes = []
    self.ax_boxes = []

    if rect is None:
      rect = [pyl.rcParams['figure.subplot.' + k] for k in ['left', 'bottom', 'right', 'top']]
      rect[2] -= rect[0]
      rect[3] -= rect[1]
    self.rect = rect

    if size is None: size = pyl.rcParams['figure.figsize']
    self.size = (float(size[0]), float(size[1]))

    self.axes_args = kwargs
    self.xaxis_args = {}
    self.yaxis_args = {}
# }}} 

  def add_axis(self, axis, rect):
# {{{
    bb = mpl.transforms.Bbox.from_extents(rect)
    assert all([s > 0. for s in bb.size]), 'Bounding box must not vanish'
    axis.parent = self
    self.axes.append(axis)
    self.ax_boxes.append(rect)
    self.naxes = len(self.axes)
# }}}

  def add_plot(self, plot, order=None):
# {{{
    plot.axes = self
    if order is None:
      self.plots.append(plot)
    else:
      self.plots.insert(plot, order)
    self.make_axis = True
    self.nplots = len(self.plots)
# }}}

  def render(self, fig = None, **kwargs):
# {{{
    pyl.ioff()

    if not isinstance(fig, mpl.figure.Figure):
      figparm = dict(figsize = self.size)
      figparm.update(kwargs)
      if not fig is None:
        figparm['num'] = fig
      fig = pyl.figure(**figparm)

    fig.clf()

    self._build_axes(fig)

    self._do_plots(fig)

    pyl.ion()
    pyl.draw()
    pyl.show()
# }}}

  def get_transform(self):
# {{{
    if self.parent is None:
      return mpl.transforms.IdentityTransform()

    ia = self.parent.axes.index(self)
    rect = self.parent.ax_boxes[ia]
    box = mpl.transforms.Bbox.from_extents(rect)
    t_self = mpl.transforms.BboxTransformTo(box)
    t_parent = self.parent.get_transform()
    return mpl.transforms.CompositeAffine2D(t_self, t_parent)
# }}}

  def _build_axes(self, fig):
# {{{
    if self.make_axis:
      tfm = self.get_transform()
      l, b = self.rect[0], self.rect[1]
      r, t = self.rect[0] + self.rect[2], self.rect[1] + self.rect[3]
      l, b = tfm.transform_point((l, b))
      r, t = tfm.transform_point((r, t))
      self.ax = fig.add_axes([l, b, r - l, t - b])

    # Build children
    for a in self.axes: a._build_axes(fig)
# }}}

  def _do_plots(self, fig):
# {{{
    # Plot children
    for a in self.axes: a._do_plots(fig)

    if self.nplots == 0: return

    # Perform plotting operations
    for p in self.plots:
      p.render(self.ax)

    # Handle scaling first, because setting this screws up other custom attributes like ticks
    args = self.axes_args.copy()
    if 'xscale' in args: self.ax.set_xscale(args.pop('xscale'))
    if 'yscale' in args: self.ax.set_yscale(args.pop('yscale'))
    if len(args) > 0: pyl.setp(self.ax, **args)

    if len(self.xaxis_args) > 0: pyl.setp(self.ax.xaxis, **self.xaxis_args)
    if len(self.yaxis_args) > 0: pyl.setp(self.ax.yaxis, **self.yaxis_args)
# }}}

  def setp(self, **kwargs):
# {{{
    self.axes_args.update(kwargs)
# }}}

  def setp_xaxis(self, **kwargs):
# {{{
    self.xaxis_args.update(kwargs)
# }}}

  def setp_yaxis(self, **kwargs):
# {{{
    self.yaxis_args.update(kwargs)
# }}}
# }}} 

# Generic object for holding plot information
class PlotWrapper:
# {{{
  def __init__(self, *plot_args, **kwargs):
    self.plot_args = plot_args
    self.plot_kwargs = kwargs
    self.axes = None

  # Draw the thing
  # This is pretty much the only public-facing method.
  def render (self, axes=None):
    pass
# }}} 

# 1D plot
class Plot(PlotWrapper):
# {{{
  def render (self, axes):
    axes.plot (*self.plot_args, **self.plot_kwargs)
# }}}

# Contour
class Contour(PlotWrapper):
# {{{
  def render (self, axes):
    self._cnt = axes.contour (*self.plot_args, **self.plot_kwargs)
# }}}

# Filled Contour
class Contourf(PlotWrapper):
# {{{
  def render (self, axes):
    self._cnt = axes.contourf (*self.plot_args, **self.plot_kwargs)
# }}}

# Filled Contour
class Colorbar(PlotWrapper):
# {{{
  def __init__(self, cnt, cax, *plot_args, **kwargs):
    self.cnt = cnt
    self.cax = cax
    PlotWrapper.__init__(self, *plot_args, **kwargs)

  def render (self, axes):
    pyl.colorbar(self.cnt._cnt, cax=self.cax.ax, *self.plot_args, **self.plot_kwargs)
# }}}

def colorbar(axes, cnt, cax=None, rect=None, *args, **kwargs):
# {{{
  if cax is None:
    pos = kwargs.pop('pos', 'r')
    if pos in ['r', 'l']: orient = kwargs.get('orientation', 'vertical')
    if pos in ['b', 't']: orient = kwargs.get('orientation', 'horizontal')
    kwargs['orientation'] = orient

    if orient == 'horizontal': 
      height = kwargs.pop('height', 0.4)
      size = axes.size[0], height

      if rect is None:
        l = kwargs.pop('rl', 0.05)
        b = kwargs.pop('rb', 0.5)
        r = kwargs.pop('rr', 0.9)
        t = kwargs.pop('rt', 0.4)
        rect = [l, b, r, t]
    else: 
      width = kwargs.pop('width', 0.8)
      size = width, axes.size[1]
      if rect is None:
        l = kwargs.pop('rl', 0.1)
        b = kwargs.pop('rb', 0.05)
        r = kwargs.pop('rr', 0.2)
        t = kwargs.pop('rt', 0.9)
        rect = [l, b, r, t]

    cax = AxesWrapper(size=size, rect=rect, make_axis=True)

    if pos == 'b': ret = grid([[axes], [cax]])
    elif pos == 'l': ret = grid([[cax, axes]])
    elif pos == 't': ret = grid([[cax], [axes]])
    else: ret = grid([[axes, cax]])

  else: ret = None

  cnt.axes.add_plot(Colorbar(cnt, cax, *args, **kwargs))
  return ret
# }}}

def make_plot_func(fclass):
  def f(*args, **kwargs):
    axes = kwargs.pop('axes', None)
    if axes is None: axes = AxesWrapper()
    axes.add_plot(fclass(*args, **kwargs))
    return axes
  return f

def make_plot_member(f):
  def g(self, *args, **kwargs): f(*args, axes = self, **kwargs)
  return g

plot = make_plot_func(Plot)
contour = make_plot_func(Contour)
contourf = make_plot_func(Contourf)

AxesWrapper.plot = make_plot_member(plot)
AxesWrapper.contour = make_plot_member(contour)
AxesWrapper.contourf = make_plot_member(contourf)

# Routine for saving this plot to a file
def save (fig, filename):
# {{{
  import pickle
  outfile = open(filename,'w')
  pickle.dump(fig, outfile)
  outfile.close()
# }}}

# Module-level routine for loading a plot from file
def load (filename):
# {{{
  import pickle
  infile = open(filename,'ro')
  theplot = pickle.load(infile)
  infile.close()
  return theplot
# }}}
