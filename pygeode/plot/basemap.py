from wrappers import AxesWrapper, PlotOp, Contour, Contourf, Streamplot, Quiver, make_plot_func, make_plot_member

from mpl_toolkits.basemap import Basemap
class BasemapAxes(AxesWrapper):
  def _build_axes(self, fig, root):
# {{{
    AxesWrapper._build_axes(self, fig, root)

    proj = {'projection':'cyl', 'resolution':'c'}
    proj.update(self.axes_args)
    self.bm = Basemap(ax = self.ax, **proj)
# }}}

  def setp(self, children=True, **kwargs):
# {{{
    proj = self.axes_args.get('projection', 'cyl')
    if proj in ['cyl', 'merc', 'mill', 'gall']:
      bnds = {}
      if kwargs.has_key('xlim'):
        x0, x1 = kwargs.pop('xlim')
        bnds['llcrnrlon'] = x0
        bnds['urcrnrlon'] = x1
      if kwargs.has_key('ylim'):
        y0, y1 = kwargs.pop('ylim')
        bnds['llcrnrlat'] = y0
        bnds['urcrnrlat'] = y1
      self.axes_args.update(bnds)

    kwargs.pop('xscale', None)
    kwargs.pop('yscale', None)

    self.args.update(kwargs)
    if children:
      for a in self.axes: a.setp(children, **kwargs)
# }}}

  def setp_xaxis(self, children=True, **kwargs):
# {{{
    kwargs.pop('major_locator', None)
    kwargs.pop('minor_locator', None)
    kwargs.pop('major_formatter', None)
    kwargs.pop('minor_formatter', None)
    AxesWrapper.setp_xaxis(self, **kwargs)
    if children:
      for a in self.axes: a.setp(children, **kwargs)
# }}}

  def setp_yaxis(self, children=True, **kwargs):
# {{{
    kwargs.pop('major_locator', None)
    kwargs.pop('minor_locator', None)
    kwargs.pop('major_formatter', None)
    kwargs.pop('minor_formatter', None)
    AxesWrapper.setp_yaxis(self, **kwargs)
    if children:
      for a in self.axes: a.setp(children, **kwargs)
# }}}

# Contour
class BMContour(Contour):
# {{{
  @staticmethod
  def transform(bm, args):
    from numpy import meshgrid, ndarray
    from warnings import warn
    # Z
    if len(args) == 1: return args
    # X, Y, Z
    if len(args) == 3:
      X, Y, Z = args
      X, Y = meshgrid(X, Y)
      X, Y = bm(X, Y)
      return X, Y, Z
    # Z, N
    if len(args) == 2 and isinstance(args[1],(int,list,tuple,ndarray)): return args
    # X, Y, Z, N
    if len(args) == 4 and isinstance(args[3],(int,list,tuple,ndarray)):
      X, Y, Z, N = args
      X, Y = meshgrid(X, Y)
      X, Y = bm(X, Y)
      return X, Y, Z, N
    #TODO: finish the rest of the cases
    warn("Don't know what to do for the coordinate transformation")
    return args

  def render (self, axes):
    bm = self.axes.bm
    args = BMContour.transform(bm, self.plot_args)
    self._cnt = bm.contour (*args, **self.plot_kwargs)
# }}}

class BMContourf(Contourf):
# {{{
  def render (self, axes):
    bm = self.axes.bm
    args = BMContour.transform(bm, self.plot_args)
    self._cnt = bm.contourf (*args, **self.plot_kwargs)
# }}}

class BMStreamplot(Streamplot):
# {{{
  @staticmethod
  def transform(bm, args):
    from numpy import meshgrid, ndarray
    from mpl_toolkits.basemap import shiftgrid
    from warnings import warn
    # X, Y, U, V
    if len(args) == 4:
      X, Y, U, V = args
      U, newX = shiftgrid(180, U, X, start=False)
      V, newX = shiftgrid(180, V, X, start=False)
      #newX, Y = meshgrid(newX, Y)
      #UP, VP, XX, YY = bm.transform_vector(U, V, newX, Y, 31, 31, returnxy=True)
      UP, VP, XX, YY = bm.rotate_vector(U, V, newX, Y, returnxy=True)
      return XX, YY, UP, VP
    #TODO: finish the rest of the cases
    warn("Don't know what to do for the coordinate transformation")
    return args

  def render (self, axes):
    bm = self.axes.bm
    args = BMStreamplot.transform(bm, self.plot_args)
    self._sp = bm.streamplot (*args, **self.plot_kwargs)
# }}}

class BMQuiver(Quiver):
# {{{
  @staticmethod
  def transform(bm, args, ngrid, **kwargs):
    from numpy import meshgrid, ndarray
    from mpl_toolkits.basemap import shiftgrid
    from warnings import warn
    # X, Y, U, V
    if len(args) == 4:
      X, Y, U, V = args
      U, newX = shiftgrid(180, U, X, start=False)
      V, newX = shiftgrid(180, V, X, start=False)
      #newX, Y = meshgrid(newX, Y)
      UP, VP, XX, YY = bm.transform_vector(U, V, newX, Y, ngrid, ngrid, returnxy=True)
      return XX, YY, UP, VP
    elif len(args) == 5:
      X, Y, U, V, C = args
      U, newX = shiftgrid(180, U, X, start=False)
      V, newX = shiftgrid(180, V, X, start=False)
      C, newX = shiftgrid(180, C, X, start=False)
      #newX, Y = meshgrid(newX, Y)
      UP, VP, XX, YY = bm.transform_vector(U, V, newX, Y, ngrid, ngrid, returnxy=True)
      CP = bm.transform_scalar(C, newX, Y, ngrid, ngrid, returnxy=False)
      return XX, YY, UP, VP, CP
    #TODO: finish the rest of the cases
    warn("Don't know what to do for the coordinate transformation")
    return args

  def __init__(self, *args, **kwargs):
    self.ngrid = kwargs.pop('ngrid', 31)
    PlotOp.__init__(self, *args, **kwargs)

  def render (self, axes):
    bm = self.axes.bm
    args = BMQuiver.transform(bm, self.plot_args, self.ngrid)
    self._cnt = bm.quiver (*args, **self.plot_kwargs)
# }}}

class BMDrawCoast(PlotOp):
# {{{
  def render (self, axes):
    bm = self.axes.bm
    bm.drawcoastlines(*self.plot_args, ax = axes, **self.plot_kwargs)
# }}}

class BMDrawMeridians(PlotOp):
# {{{
  def render (self, axes):
    bm = self.axes.bm
    bm.drawmeridians(*self.plot_args, ax = axes, **self.plot_kwargs)
# }}}

class BMDrawParallels(PlotOp):
# {{{
  def render (self, axes):
    bm = self.axes.bm
    bm.drawparallels(*self.plot_args, ax = axes, **self.plot_kwargs)
# }}}

class BMBlueMarble(PlotOp):
# {{{
  def render (self, axes):
    bm = self.axes.bm
    bm.bluemarble(*self.plot_args, ax = axes, **self.plot_kwargs)
# }}}

class BMWarpImage(PlotOp):
# {{{
  def render (self, axes):
    bm = self.axes.bm
    bm.warpimage(*self.plot_args, ax = axes, **self.plot_kwargs)
# }}}

bmcontour = make_plot_func(BMContour)
bmcontourf = make_plot_func(BMContourf)
bmstreamplot = make_plot_func(BMStreamplot)
bmquiver = make_plot_func(BMQuiver)
drawcoastlines = make_plot_func(BMDrawCoast)
drawmeridians = make_plot_func(BMDrawMeridians)
drawparallels = make_plot_func(BMDrawParallels)
bluemarble = make_plot_func(BMBlueMarble)
warpimage = make_plot_func(BMWarpImage)

BasemapAxes.contour = make_plot_member(bmcontour)
BasemapAxes.contourf = make_plot_member(bmcontourf)
BasemapAxes.streamplot = make_plot_member(bmstreamplot)
BasemapAxes.quiver = make_plot_member(bmquiver)
BasemapAxes.drawcoastlines = make_plot_member(drawcoastlines)
BasemapAxes.drawmeridians = make_plot_member(drawmeridians)
BasemapAxes.drawparallels = make_plot_member(drawparallels)
BasemapAxes.bluemarble = make_plot_member(bluemarble)
BasemapAxes.warpimage = make_plot_member(warpimage)

__all__ = ['BasemapAxes', 'bmcontour', 'bmcontourf', 'bmstreamplot', 'bmquiver', \
            'drawcoastlines', 'drawmeridians', 'drawparallels', 'bluemarble', 'warpimage']
