# Wrapper for matplotlib.pyplot

from matplotlib import pyplot as pyl
import matplotlib as mpl

# Allows plots to be constructed in a slightly more object-oriented way.
# Also allows plot objects to be saved to / loaded from files, something which
# normal matplotlib plots can't do.

# Interface for wrapping a matplotlib axes object
class AxesWrapper:
# {{{
  def __init__(self, parent=None, rect=None, size=None, pad=None, make_axis=False, name='', **kwargs):
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
    self.pad = pad

    if size is None: size = pyl.rcParams['figure.figsize']
    self.size = (float(size[0]), float(size[1]))

    self.args = {}
    self.axes_args = kwargs
    self.xaxis_args = {}
    self.yaxis_args = {}

    self.name = name
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

  def add_plot(self, plot, order=None, make_axes=True):
# {{{
    plot.axes = self
    if order is None:
      self.plots.append(plot)
    else:
      self.plots.insert(plot, order)
    if make_axes: self.make_axis = True
    self.nplots = len(self.plots)
# }}}

  def pop_plot(self, order=-1):
# {{{
    self.plots.pop(order)
    self.nplots = len(self.plots)
# }}}

  def render(self, fig = None, show = True, **kwargs):
# {{{
    wason = pyl.isinteractive()
    if wason: pyl.ioff()

    if not isinstance(fig, mpl.figure.Figure):
      figparm = dict(figsize = self.size)
      figparm.update(kwargs)
      if not fig is None:
        figparm['num'] = fig
      fig = pyl.figure(**figparm)

    fig.clf()

    self._build_axes(fig, self)

    self._do_plots(fig)

    if wason:
      pyl.ion()

      if show:
        pyl.show()
        pyl.draw()

    return fig
# }}}

  def get_transform(self, root = None):
 # {{{
    if self is root or self.parent is None:
      return mpl.transforms.IdentityTransform()

    ia = self.parent.axes.index(self)
    rect = self.parent.ax_boxes[ia]
    box = mpl.transforms.Bbox.from_extents(rect)
    t_self = mpl.transforms.BboxTransformTo(box)
    t_parent = self.parent.get_transform()
    return mpl.transforms.CompositeAffine2D(t_self, t_parent)
# }}}

  def _build_axes(self, fig, root):
# {{{
    if self.make_axis:
      tfm = self.get_transform(root)
      if self.pad is not None:
         l, b = tfm.transform_point((0., 0.))
         r, t = tfm.transform_point((1., 1.))
         #print l, b, r, t
         fsize = fig.get_size_inches()
         l += self.pad[0] / fsize[0]
         b += self.pad[1] / fsize[1]
         w = r - l - self.pad[2] / fsize[0]
         h = t - b - self.pad[3] / fsize[1]
      else:
         l, b = self.rect[0], self.rect[1]
         r, t = self.rect[0] + self.rect[2], self.rect[1] + self.rect[3]
         l, b = tfm.transform_point((l, b))
         r, t = tfm.transform_point((r, t))
         w = r - l
         h = t - b

      self.ax = fig.add_axes([l, b, w, h])
    else:
      self.ax = None

    # Build children
    for a in self.axes: a._build_axes(fig, root)

    #Draw bounding boxes of children for debugging purposes
    #if root is self:
      #ax = pyl.gca()
      #for b in self.ax_boxes:
        #xy = b[0], b[1]
        #w = b[2] - b[0]
        #h = b[3] - b[1]
        #ax.add_patch(pyl.Rectangle(xy, w, h, lw=2., fill=False, transform=fig.transFigure, clip_on=False))
# }}}

  def _do_plots(self, fig):
# {{{
    # Plot children
    for a in self.axes: a._do_plots(fig)

    if self.ax is None: return

    preops = [p for p in self.plots if p.pre]
    postops = [p for p in self.plots if not p.pre]

    # Perform plotting operations
    for p in preops:
      p.render(self.ax)

    # Handle scaling first, because setting this screws up other custom attributes like ticks
    args = self.args.copy()
    if 'xscale' in args: self.ax.set_xscale(args.pop('xscale'))
    if 'yscale' in args: self.ax.set_yscale(args.pop('yscale'))
    if len(args) > 0: pyl.setp(self.ax, **args)

    if len(self.xaxis_args) > 0: pyl.setp(self.ax.xaxis, **self.xaxis_args)
    if len(self.yaxis_args) > 0: pyl.setp(self.ax.yaxis, **self.yaxis_args)

    # Perform plotting operations
    for p in postops:
      p.render(self.ax)
# }}}

  def setp(self, children=True, **kwargs):
# {{{
    self.args.update(kwargs)
    if children:
      for a in self.axes: a.setp(children, **kwargs)
# }}}

  def setp_xaxis(self, children=True, **kwargs):
# {{{
    self.xaxis_args.update(kwargs)
    if children:
      for a in self.axes: a.setp_xaxis(children, **kwargs)
# }}}

  def setp_yaxis(self, children=True, **kwargs):
# {{{
    self.yaxis_args.update(kwargs)
    if children:
      for a in self.axes: a.setp_yaxis(children, **kwargs)
# }}}

  def find_plot(self, cl):
  # {{{
    ''' Returns last instance of plot class cl in this axes plots. '''
    for p in reversed(self.plots):
      if isinstance(p, cl): return p
    return None
  # }}}
# }}} 

# Generic object for holding plot information
class PlotOp:
# {{{
  def __init__(self, *plot_args, **kwargs):
    self.plot_args = plot_args
    self.plot_kwargs = kwargs
    self.axes = None
    self.pre = True

  # Draw the thing
  # This is pretty much the only public-facing method.
  def render (self, axes=None):
    pass
# }}} 

# 1D plots
class Plot(PlotOp):
# {{{
  def render (self, axes):
    axes.plot (*self.plot_args, **self.plot_kwargs)
    #print 'Autoscaling.'
    #axes.autoscale_view()
# }}}

class FillBetween(PlotOp):
# {{{
  def render (self, axes):
    axes.fill_between(*self.plot_args, **self.plot_kwargs)
    axes.autoscale()
# }}}

class Scatter(PlotOp):
# {{{
  def render (self, axes):
    axes.scatter (*self.plot_args, **self.plot_kwargs)
# }}}

class Errorbar(PlotOp):
# {{{
  def render (self, axes):
    axes.errorbar (*self.plot_args, **self.plot_kwargs)
# }}}

class Histogram(PlotOp):
# {{{
  def render (self, axes):
    axes.hist (*self.plot_args, **self.plot_kwargs)
# }}}

class AxHLine(PlotOp):
# {{{
  def render (self, axes):
    axes.axhline (*self.plot_args, **self.plot_kwargs)
# }}}

class AxVLine(PlotOp):
# {{{
  def render (self, axes):
    axes.axvline (*self.plot_args, **self.plot_kwargs)
# }}}

class Legend(PlotOp):
# {{{
  def render (self, axes):
    axes.legend (*self.plot_args, **self.plot_kwargs)
# }}}

class Text(PlotOp):
# {{{
  def render (self, axes):
    kwargs = self.plot_kwargs.copy()
    tr = kwargs.pop('transform', 'Data')
    if tr == 'Axes': kwargs['transform'] = axes.transAxes
    if tr == 'Data': kwargs['transform'] = axes.transData
       
    axes.text (*self.plot_args, **kwargs)
# }}}

# Contour
class Contour(PlotOp):
# {{{
  def render (self, axes):
    self._cnt = axes.contour (*self.plot_args, **self.plot_kwargs)
# }}}

# Filled Contour
class Contourf(PlotOp):
# {{{
  def render (self, axes):
    self._cnt = axes.contourf (*self.plot_args, **self.plot_kwargs)
# }}}

# Op to modify contours
class ModifyContours(PlotOp):
# {{{
  def __init__(self, cnt, ind=None, **kwargs):
    self.cnt = cnt
    self.ind = ind
    PlotOp.__init__(self, **kwargs)

  def render (self, axes):
    coll = self.cnt._cnt.collections
    if self.ind is None: pyl.setp(coll, **self.plot_kwargs)
    else: pyl.setp([coll[i] for i in self.ind], **self.plot_kwargs)
# }}}

# Op to add contour labels
class CLabel(PlotOp):
# {{{
  def __init__(self, cnt, **kwargs):
    self.cnt = cnt
    PlotOp.__init__(self, **kwargs)
    self.pre = False

  def render (self, axes):
    pyl.clabel(self.cnt._cnt, **self.plot_kwargs)
# }}}

# PColor
class PColor(PlotOp):
# {{{
  def render (self, axes):
    self._cnt = axes.pcolor (*self.plot_args, **self.plot_kwargs)
# }}}

# Streamplot
class Streamplot(PlotOp):
# {{{
  def render (self, axes):
    self._sp = axes.streamplot (*self.plot_args, **self.plot_kwargs)
# }}}

# Quiver
class Quiver(PlotOp):
# {{{
  def render (self, axes):
    self._cnt = axes.quiver (*self.plot_args, **self.plot_kwargs)
# }}}

# Op to add a quiver key
class QuiverKey(PlotOp):
# {{{
  def __init__(self, cnt, *args, **kwargs):
    self.cnt = cnt
    PlotOp.__init__(self, *args, **kwargs)

  def render (self, axes):
    pyl.quiverkey(self.cnt._cnt, *self.plot_args, **self.plot_kwargs)
# }}}

# Colorbar
class Colorbar(PlotOp):
# {{{
  def __init__(self, cnt, cax, *plot_args, **kwargs):
    self.cnt = cnt
    self.cax = cax
    self.lcnt = kwargs.pop('lcnt', None)
    PlotOp.__init__(self, *plot_args, **kwargs)

  def render (self, axes):
    self._cbar = pyl.colorbar(self.cnt._cnt, cax=self.cax.ax, *self.plot_args, **self.plot_kwargs)
    if self.lcnt is not None: self._cbar.add_lines(self.lcnt._cnt)
    pyl.sca(axes)
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
        l = kwargs.pop('rl', 0.15)
        b = kwargs.pop('rb', 0.5)
        r = kwargs.pop('rr', 0.75)
        t = kwargs.pop('rt', 0.4)
        rect = [l, b, r, t]
    else: 
      width = kwargs.pop('width', 0.8)
      size = width, axes.size[1]
      if rect is None:
        l = kwargs.pop('rl', 0.1)
        b = kwargs.pop('rb', 0.15)
        r = kwargs.pop('rr', 0.2)
        t = kwargs.pop('rt', 0.75)
        rect = [l, b, r, t]

    cax = AxesWrapper(size=size, rect=rect, make_axis=True)

    if pos == 'b': ret = grid([[axes], [cax]])
    elif pos == 'l': ret = grid([[cax, axes]])
    elif pos == 't': ret = grid([[cax], [axes]])
    else: ret = grid([[axes, cax]])

  else: ret = None

  ticklabels = kwargs.pop('ticklabels', None)
  kwargs['spacing'] = kwargs.pop('spacing', 'proportional')

  cnt.axes.add_plot(Colorbar(cnt, cax, *args, **kwargs))

  if ticklabels is not None:
    if orient == 'horizontal':
      cax.setp(xticklabels = ticklabels)
    else:
      cax.setp(yticklabels = ticklabels)

  return ret
# }}}

def make_plot_func(fclass, make_axes=True):
  def f(*args, **kwargs):
    axes = kwargs.pop('axes', None)
    if axes is None: axes = AxesWrapper()
    axes.add_plot(fclass(*args, **kwargs), make_axes=make_axes)
    return axes
  return f

def make_plot_member(f):
  def g(self, *args, **kwargs): f(*args, axes = self, **kwargs)
  return g

plot = make_plot_func(Plot)
fill_between = make_plot_func(FillBetween)
scatter = make_plot_func(Scatter)
errorbar = make_plot_func(Errorbar)
hist = make_plot_func(Histogram)
axhline = make_plot_func(AxHLine)
axvline = make_plot_func(AxVLine)
legend = make_plot_func(Legend)
text = make_plot_func(Text, make_axes=False)
contour = make_plot_func(Contour)
contourf = make_plot_func(Contourf)
modifycontours = make_plot_func(ModifyContours)
clabel = make_plot_func(CLabel)
pcolor = make_plot_func(PColor)
streamplot = make_plot_func(Streamplot)
quiver = make_plot_func(Quiver)
quiverkey = make_plot_func(QuiverKey)

__all__ = ['AxesWrapper', 'plot', 'fill_between', 'scatter', 'hist', 'axhline', 'axvline', 'legend', 'text', 'contour', 'contourf', 'pcolor', 'quiver', 'quiverkey', 'colorbar']

AxesWrapper.plot = make_plot_member(plot)
AxesWrapper.fill_between = make_plot_member(fill_between)
AxesWrapper.scatter = make_plot_member(scatter)
AxesWrapper.errorbar = make_plot_member(errorbar)
AxesWrapper.hist = make_plot_member(hist)
AxesWrapper.axhline = make_plot_member(axhline)
AxesWrapper.axvline = make_plot_member(axvline)
AxesWrapper.legend = make_plot_member(legend)
AxesWrapper.text = make_plot_member(text)
AxesWrapper.contour = make_plot_member(contour)
AxesWrapper.contourf = make_plot_member(contourf)
AxesWrapper.modifycontours = make_plot_member(modifycontours)
AxesWrapper.clabel = make_plot_member(clabel)
AxesWrapper.pcolor = make_plot_member(pcolor)
AxesWrapper.streamplot = make_plot_member(streamplot)
AxesWrapper.quiver = make_plot_member(quiver)
AxesWrapper.quiverkey = make_plot_member(quiverkey)

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

def grid(axes, size = None):
# {{{
  # Expect a 2d grid; first index is the row, second the column
  ny = len(axes)
  nx = len(axes[0])
  assert all([len(x) == nx for x in axes[1:]]), 'Each row must have the same number of axes'

  rowh = [max([a.size[1] for a in row if a is not None]) for row in axes]
  colw = [max([axes[i][j].size[0] for i in range(ny) if axes[i][j] is not None]) for j in range(nx)]

  tsize = [float(sum(colw)), float(sum(rowh))]
  if size is None: size = tsize

  Ax = AxesWrapper(size = size)

  x, y = 0., 1.
  for i in range(ny):
    for j in range(nx):
      ax = axes[i][j]
      if ax is not None:
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

def annotate(axes, text, pos='b'):
# {{{
   size = axes.size[0], axes.size[1] + 0.5
   Ax = AxesWrapper(size=size)
   r = (0, 0.5 / float(size[1]), 1., 1.)
   Ax.add_axis(axes, r)
   Ax.text(0.5, 0., text, ha='center', va='bottom')#, transform='Ax')
   return Ax
# }}}
__all__.extend(['save', 'load', 'grid'])

try:
  from basemap import *
  from basemap import __all__ as bm_all
  __all__.extend(bm_all)

  def isbasemapaxis(axes):
  # {{{
    return isinstance(axes, BasemapAxes)
  # }}}
except ImportError:
  import warnings
  warnings.warn('Basemap functionality is unavailable.')

  def isbasemapaxis(axes):
  # {{{
    return False
  # }}}
