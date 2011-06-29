# Interface to the GNU Scientific Library interpolation routines
# Requires the GSL shared libraries and header files.

#TODO

from pygeode.var import Var

from pygeode.libhelper import load_lib


gslcblas = load_lib('gslcblas', Global=True)  # GSL needs some symbols from here
gsl = load_lib('gsl', Global=True)

interp = load_lib("interp")

del load_lib


# Interpolation types from GSL
import ctypes
interp_types = dict([t,ctypes.c_void_p.in_dll(gsl,'gsl_interp_'+t)]
  for t in ['linear', 'polynomial', 'cspline', 'cspline_periodic', 'akima', 'akima_periodic']
)
del ctypes


# Sorted var
# (sorted along a certain axis)
class SortedVar (Var):
# {{{
  def __init__ (self, invar, iaxis, reverse=False):
    from pygeode.var import Var
    self.var = invar
    iaxis = invar.whichaxis(iaxis)
    axes = list(invar.axes)
    oldaxis = axes[iaxis]
    newaxis = oldaxis.sorted(reverse=reverse)
    axes[iaxis] = newaxis
    Var.__init__(self, axes, dtype=invar.dtype, name=invar.name, atts=invar.atts, plotatts=invar.plotatts)

  def getview (self, view):
    return view.get(self.var)
# }}}

def sorted (var, iaxis, reverse=False):
# {{{
  newvar = SortedVar (var, iaxis, reverse=reverse)
  if newvar.getaxis(iaxis) is var.getaxis(iaxis): return var
  return newvar
# }}}

# Interpolation Var
class Interp (Var):
# {{{
  def __init__ (self, invar, inaxis, outaxis, inx=None, outx=None, interp_type='cspline', \
                d_below=0., d_above=0.):
# {{{
    from pygeode.var import Var
    from pygeode.axis import Axis

    # Check the types of the input parameters
    assert isinstance(invar,Var)
    inaxis = invar.getaxis(inaxis)
    assert isinstance(outaxis,Axis)

    if inx is None: inx = inaxis
    if outx is None: outx = outaxis

    assert isinstance(inx,Var)
    assert isinstance(outx,Var)

    # Validate the input axis
    assert invar.hasaxis(inaxis)

    # We need the interpolation axis to be the fastest-varying axis
    # (For the C code to work properly)
    iaxis = invar.whichaxis(inaxis)
    order = range(0,iaxis) + range(iaxis+1,invar.naxes) + [iaxis]
    invar = invar.transpose (*order)
    del iaxis, order

    # Generate the output axes
    outaxes = list(invar.axes)
    outaxes[-1] = outaxis

    # Validate the coordinate fields
    assert all(invar.hasaxis(a) for a in inx.axes)
#    assert all(invar.hasaxis(a) or a is outaxis for a in outx.axes)
    for a in outx.axes:
      if a is outaxis: continue
      assert invar.hasaxis(a), "invar doesn't have axis '%s'"%repr(a)
    assert inx.hasaxis(inaxis)
    assert outx.hasaxis(outaxis)

    self.invar = invar
    self.inx = inx
    self.outx = outx
    self.interp_type = interp_type

    self.d_below = d_below
    self.d_above = d_above

    Var.__init__ (self, outaxes, name=invar.name, dtype='d', atts=invar.atts, plotatts=invar.plotatts)
# }}}

  def getview (self, view, pbar=None):
# {{{
    from pygeode.tools import point
    from ctypes import c_double
    import numpy as np

    #TODO: intelligent mapping of the input/output interpolation axes
    # Right now, we must read in the whole axis for the input.
    #TODO: at least split this up along the other axes, so we don't run out of memory

    insl = list(view.slices)
    insl[-1] = slice(None)
    inview = view.replace_axis(self.naxes-1, self.invar.axes[-1])
    indata = inview.get(self.invar)
    outdata = np.empty(view.shape, dtype=self.dtype)

    # Get the input/output interpolation coordinates
    inx = self.inx
    outx = self.outx

    # Input coordinates
    inx_data = inview.get(inx)
    if inx.naxes == 1: loop_inx = 0
    else:
      loop_inx = 1
      # Expand this out (can't use broadcasting in the C code)
      for i in range(inx.naxes):
        if inx_data.shape[i] == 1:
          inx_data = np.repeat(inx_data, indata.shape[i], axis=i)

    # Output coordinates
    outx_data = view.get(outx)
    if outx.naxes == 1: loop_outx = 0
    else:
      loop_outx = 1
      # Expand this out
      for i in range(outx.naxes):
        if outx_data.shape[i] == 1:
          outx_data = np.repeat(outx_data, outdata.shape[i], axis=i)


    narrays = indata.size / indata.shape[-1]
    ninx = inx_data.shape[-1]
    noutx = outx_data.shape[-1]
    inx_data = np.ascontiguousarray(inx_data, dtype='float64')
    indata = np.ascontiguousarray(indata, dtype='float64')
    outx_data = np.ascontiguousarray(outx_data, dtype='float64')
    outdata = np.ascontiguousarray(outdata, dtype='float64')
    interp_type = interp_types[self.interp_type]

    # Do the interpolation
    ret = interp.interpgsl_nan (narrays, ninx, noutx,
            point(inx_data), point(indata), point(outx_data), point(outdata),
            loop_inx, loop_outx,
            c_double(self.d_below), c_double(self.d_above),
            interp_type)

    assert ret == 0

    return outdata
# }}}
# }}}

def interpolate(var, inaxis, outaxis, inx=None, outx=None, interp_type='cspline', \
                d_below = 0., d_above = 0.):
# {{{
  """
  Interpolates a variable along a single dimension.

  Parameters
  ----------
  invar : Var
    The input variable
  inaxis : Axis
    The input axis being interpolated from
  outaxis : Axis
    The output axis being interpolated to. 

  inx : Var (optional)
    The coordinates from which we are interpolating (must be conformable to the
    input var). If not provided, the values of inaxis are used. This can be
    either a one dimensional field defined on inaxis, or a multidimensional
    field.
  outx : Var (optional)
    The coordinates to which we are interpolating (must be conformable to the
    output var).  If not provided, the values of outaxis are used. This can be
    either a one dimensional field defined on outaxis, or a multidimensional
    field.
  interp_type : string (optional)
    The type of interpolation. One of 'linear', 'polynomial', 'cspline',
    'cspline_periodic', 'akima', 'akima_periodic'.
    Default is 'cspline' (cubic spline interpolation)


  Returns
  -------
  interpolated : Var
    The interpolated variable

  Notes
  -----
  if inx or outx are not explicitly given, they will take on the values of
  inaxis and outaxis respectively.  Note that inx and outx are assumed to
  have the same units, so if inaxis and outaxis have different units, you'll
  need to override either inx or outx explicitly, and provide some kind of
  mapping field there.


  Examples::

    1) Simple interpolation along one axis (re-gridding):

    To interpolate some data to 20 evenly spaced longitude values (starting at 0):
    newvar = Interp (invar = data, inaxis = 'lon', outaxis = Lon(20))

    2) Interpolation involving a change of coordinates:

    Suppose we have some ozone data (o3) on a model vertical coordinate.
    Suppose we also have pre-computed a pressure field (pfield) over these coordinates.
    Suppose finally we have our desired pressure axis (paxis) that we want to interpolate to.

    To interpolate onto pressure levels:
    newvar = Interp (invar = o3, inaxis = 'eta', outaxis = paxis, inx = pfield)

    To go backwards, and convert from pressure levels back to model levels:
    whyohwhy = Interp (invar = newvar, inaxis = paxis,
                       outaxis = o3.getaxis('eta'), outx = pfield)

    Observe that here, we have 2 different units of coordinates ('eta' and 
    pressure (hPa)).  In this case, we had to provide an explicit mapping
    between these 2 coordinates, by use of pfield (pressure field on eta
    levels).

    Now, you may be asking yourself "shouldn't pressure be interpolated on a
    log scale?".  This is true, but I tried to simplify things by assuming a
    linear scale just to show the basics.  If you wanted to interpolate over
    a log scale, then your interpolation coordinate would be log(pressure),
    instead of pressure.  You'll have to explicitly provide this log coordinate
    for both the inputs and outputs, for example:

    newvar = Interp (invar = o3, inaxis = 'eta', outaxis = paxis,
                     inx = pfield.log(), outx = paxis.log()  )

    Here, our input and output axes remain the same as before, but now we're
    using log(pressure) internally as the coordinate over which to perform
    the interpolation.
  """
  return Interp(var, inaxis, outaxis, inx, outx, interp_type, d_below, d_above)
# }}}

del Var

