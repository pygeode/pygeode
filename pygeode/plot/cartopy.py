from .wrappers import AxesWrapper, PlotOp, Contour, Contourf, PColor, Streamplot, Quiver, make_plot_func, make_plot_member

import cartopy as crt
import cartopy.crs as ccrs

prj_grid = ['PlateCarree', 'Mercator', 'Miller', 'TransverseMercator']

prj_regn = ['AlbersEqualArea', 'EquidistantConic', 'LambertConformal', \
            'LambertCylindrical', 'RotatedPole']

prj_glob = ['AzimuthalEquidistant', 'Mollweide', 'Orthographic',  \
            'Stereographic', 'Robinson', 'Sinusoidal', 'NearsidePerspective', \
            'Geostationary', 'LambertAzimuthalEqualArea', 'EckertIII']

class CartopyAxes(AxesWrapper):
  def __init__(self, projection = 'PlateCarree', prj_args = None, transform = None, setglobal = None, **kwargs):
    AxesWrapper.__init__(self, **kwargs)

    # Allow users to pass in an already constructed projection
    if isinstance(projection, ccrs.Projection):
      self.projection = projection
      self.prj_name = projection.__class__.__name__
      self.prj_args = None
    else:
      self.prj_name = projection

      if prj_args is None:
        if self.prj_name in ['PlateCarree']:
          prj_args = dict(central_longitude = 180.)
        else:
          prj_args = dict()

      self.prj_args = prj_args
      self.projection = ccrs.__dict__[self.prj_name](**self.prj_args)

    if setglobal is None:
      if self.prj_name in prj_glob:
        setglobal = True
      else:
        setglobal = False

    if transform is None:
      transform = ccrs.PlateCarree()

    self.transform = transform

    self.setglobal = setglobal

  def _build_axes(self, fig, root):
# {{{ 
    AxesWrapper._build_axes(self, fig, root)
    if self.setglobal:
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

class CRTPColor(PColor):
# {{{
  def render (self, axes):
    self._cnt = axes.pcolor (*self.plot_args, transform = self.axes.transform, **self.plot_kwargs)
# }}}

class CRTStreamplot(Streamplot):
# {{{
  def render (self, axes):
    self._cnt = axes.streamplot (*self.plot_args, transform = self.axes.transform, **self.plot_kwargs)
# }}}

class CRTQuiver(Quiver):
# {{{
  def render (self, axes):
    self._cnt = axes.quiver (*self.plot_args, transform = self.axes.transform, **self.plot_kwargs)
# }}}

class CRTCoastlines(PlotOp):
# {{{
  def render (self, axes):
    axes.coastlines(*self.plot_args, **self.plot_kwargs)
# }}}

class CRTGridlines(PlotOp):
# {{{
  def render (self, axes):
    self._gl = axes.gridlines(*self.plot_args, **self.plot_kwargs)
# }}}

class CRTModifyGridlines(PlotOp):
# {{{
  def __init__(self, gl, **kwargs):
    self.gl = gl
    PlotOp.__init__(self, **kwargs)

  def render (self, axes):
    gl = self.gl._gl
    gl.__dict__.update(self.plot_kwargs)
# }}}

class CRTAddFeature(PlotOp):
# {{{
  def render (self, axes):
    axes.add_feature(*self.plot_args, **self.plot_kwargs)
# }}}

class CRTAddGeometries(PlotOp):
# {{{
  def render (self, axes):
    axes.add_geometries(*self.plot_args, **self.plot_kwargs)
# }}}

class CRTSetExtent(PlotOp):
# {{{
  def render (self, axes):
    axes.set_extent(*self.plot_args, **self.plot_kwargs)
# }}}

contour        = make_plot_func(CRTContour)
contourf       = make_plot_func(CRTContourf)
coastlines     = make_plot_func(CRTCoastlines)
gridlines      = make_plot_func(CRTGridlines)
modifygridlines = make_plot_func(CRTModifyGridlines)
pcolor         = make_plot_func(CRTPColor)
streamplot     = make_plot_func(CRTStreamplot)
quiver         = make_plot_func(CRTQuiver)
add_feature    = make_plot_func(CRTAddFeature)
add_geometries = make_plot_func(CRTAddGeometries)
set_extent     = make_plot_func(CRTSetExtent)

CartopyAxes.contour        = make_plot_member(contour)
CartopyAxes.contourf       = make_plot_member(contourf)
CartopyAxes.coastlines     = make_plot_member(coastlines)
CartopyAxes.gridlines      = make_plot_member(gridlines)
CartopyAxes.modifygridlines = make_plot_member(modifygridlines)
CartopyAxes.pcolor         = make_plot_member(pcolor)
CartopyAxes.streamplot     = make_plot_member(streamplot)
CartopyAxes.quiver         = make_plot_member(quiver)
CartopyAxes.add_feature    = make_plot_member(add_feature)
CartopyAxes.add_geometries = make_plot_member(add_geometries)
CartopyAxes.set_extent     = make_plot_member(set_extent)

__all__ = ['CartopyAxes', 'contour', 'contourf', 'coastlines', 'gridlines', 'modifygridlines', 'pcolor', 'streamplot', 'quiver', 'add_feature', 'add_geometries', 'set_extent']
