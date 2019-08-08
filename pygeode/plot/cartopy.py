from .wrappers import AxesWrapper, PlotOp, Contour, Contourf, Streamplot, Quiver, make_plot_func, make_plot_member

import cartopy as crt
import cartopy.crs as ccrs

class CartopyAxes(AxesWrapper):
  def __init__(self, transform = None, **kwargs):
    AxesWrapper.__init__(self, **kwargs)

    if transform is None:
      transform = ccrs.PlateCarree()

    self.transform = transform

  def _build_axes(self, fig, root):
# {{{
    prj = self.axes_args.get('projection', 'PlateCarree')
    prj_args = self.axes_args.get('prj_args', {})

    self.projection = ccrs.__dict__[prj](**prj_args)
    
    AxesWrapper._build_axes(self, fig, root)
    self.ax.set_global()
# }}}

# Contour
class CRTContour(Contour):
# {{{
  def render (self, axes):
    self._cnt = axes.contour (*self.plot_args, transform = self.axes.transform, **self.plot_kwargs)
# }}}

class CRTContourf(Contourf):
# {{{
  def render (self, axes):
    self._cnt = axes.contourf (*self.plot_args, transform = self.axes.transform, **self.plot_kwargs)
# }}}

class CRTCoastlines(PlotOp):
# {{{
  def render (self, axes):
    axes.coastlines(*self.plot_args, **self.plot_kwargs)
# }}}

class CRTGridlines(PlotOp):
# {{{
  def render (self, axes):
    axes.gridlines(*self.plot_args, **self.plot_kwargs)
# }}}

contour    = make_plot_func(CRTContour)
contourf   = make_plot_func(CRTContourf)
coastlines = make_plot_func(CRTCoastlines)
gridlines  = make_plot_func(CRTGridlines)

CartopyAxes.contour    = make_plot_member(contour)
CartopyAxes.contourf   = make_plot_member(contourf)
CartopyAxes.coastlines = make_plot_member(coastlines)
CartopyAxes.gridlines  = make_plot_member(gridlines)

__all__ = ['CartopyAxes', 'contour', 'contourf', 'coastlines', 'gridlines']
