# Simple Var operations
# (things that manipulate the properties/shape of the data, without changing the values)

from pygeode.var import Var

# Squeeze out degenerate axes (where length=1)
class SqueezedVar(Var):
  def __init__ (self, var, *iaxes, **kwargs):
    from pygeode.var import Var, copy_meta
    # Get the axes to be squeezed
    if len(iaxes) == 1 and isinstance(iaxes[0],(list,tuple)): iaxes = iaxes[0]
    if len(iaxes) == 0: iaxes = [i for i,a in enumerate(var.axes) if len(a) == 1]

    # Only remove degenerate axes
    iaxes = [var.whichaxis(a) for a in iaxes]
    iaxes = [i for i in iaxes if len(var.axes[i]) == 1]

    # Slice the var along some axes (passed by keyword argument)?
    if len(kwargs) > 0:
      for k,v in kwargs.items():
        assert var.hasaxis(k), "unknown axis '%s'"%k
        a = var.whichaxis(k)
        if a not in iaxes: iaxes.append(a)
        assert isinstance(v,(int,float)), "expected a numerical value for keyword '%s' - received %s instead"%(k,type(v))
      var = var(**kwargs)  # Do the slicing first, before doing this wrapper

    self.var = var

    Var.__init__(self,[a for i,a in enumerate(var.axes) if i not in iaxes], var.dtype)
    copy_meta (var, self)

  def getview (self, view, pbar):
    return view.get(self.var, strict=False, conform=False, pbar=pbar).reshape(view.shape)

def squeeze (self, *iaxes, **kwargs):
  """
  Removes degenerate axes from a variable, reducing its dimensionality.

  Parameters
  ----------
  *iaxes : one or more axes (optional)
    The axes to remove (they must already be degenerate).  If no explicit axes
    are provided, then *all* degenerate axes are removed.
  **kwargs : keyword arguments
    Keyword-based axis slicing.  Selects a particular value along the axis and
    then removes the axis from the output.  See :meth:`Var.__call__` for a
    similar method which uses this keyword syntax.

  Returns
  -------
  squeezed : Var
    The squeezed variable
  """
  var = self
  return SqueezedVar(var, *iaxes, **kwargs)

# Extend the var with more axes
class ExtendedVar(Var):
  def __init__(self, var, pos, *newaxes):
    from pygeode.var import Var, copy_meta
    self.var = var
    self.pos = pos
    self.newaxes = newaxes
    axes = var.axes[:pos] + tuple(newaxes) + var.axes[pos:]
    Var.__init__(self, axes, dtype=var.dtype)
    copy_meta (var, self)
  def getview(self, view, pbar):
    import numpy as np
    out = np.empty(view.shape, self.dtype)
    values = view.get(self.var, pbar=pbar)  # view.get should give us degenerate extended axes here
    # Broadcast the values to the extended shape
    out[()] = values
    return out

def extend (self, pos, *newaxes):
  """
  Adds more axes to a variable.  Data will be duplicated along these new axes.

  Parameters
  ----------
  pos : int
    The position (within the variable's current axes) to start inserting the
    new axes.
  *newaxes : one or more axes
    The new axes to insert

  Returns
  -------
  extended_var : Var
    The variable, extended to include the new axes.

  Examples
  --------
  >>> from pygeode.tutorial import t1
  >>> print(t1.Temp)
  <Var 'Temp'>:
    Units: K  Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  Add_Var (dtype="float64")
  >>> from pygeode import Pres
  >>> paxis = Pres([1000,850,700])  # Create a pressure axis
  >>> print(paxis)
  pres <Pres>    :  1000 hPa to 700 hPa (3 values)
  >>> extended_var = t1.Temp.extend(0, paxis)  # Extend the data with this axis
  >>> print(extended_var)
  <Var 'Temp'>:
    Units: K  Shape:  (pres,lat,lon)  (3,31,60)
    Axes:
      pres <Pres>    :  1000 hPa to 700 hPa (3 values)
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  ExtendedVar (dtype="float64")
  """
  var = self
  if isinstance(newaxes[0], (tuple,list)): newaxes = newaxes[0]
  return ExtendedVar(var, pos, *newaxes)

# Transpose a variable
class TransposedVar(Var):
  def __init__(self, var, alist):
    from pygeode.var import Var, copy_meta
    self.var = var
    Var.__init__(self, [var.axes[a] for a in alist], dtype=var.dtype)
    copy_meta(var, self)
  def getview (self, view, pbar): return view.get(self.var, pbar=pbar)

def transpose (self, *axes):
  """
  Transposes the axes of a variable.

  Parameters
  ----------
  *axes : one or more axis identifiers (strings, :class:`Axis` classes, integer indices)
    The new order of the axes.  Any unspecified axes will be appended after these.

  Returns
  -------
  transposed_var : Var
    The transposed variable.

  Examples
  --------
  >>> from pygeode.tutorial import t1
  >>> print(t1.Temp)
  <Var 'Temp'>:
    Units: K  Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  Add_Var (dtype="float64")
  >>> transposed_var = t1.Temp.transpose('lon','lat')
  >>> print(transposed_var)
  <Var 'Temp'>:
    Units: K  Shape:  (lon,lat)  (60,31)
    Axes:
      lon <Lon>      :  0 E to 354 E (60 values)
      lat <Lat>      :  90 S to 90 N (31 values)
    Attributes:
      {}
    Type:  TransposedVar (dtype="float64")
  """
  var = self
  assert len(axes) > 0, "no axes specified"
  alist = [var.whichaxis(a) for a in axes]
  for a in alist: assert 0 <= a < var.naxes, "axis not found"
  # Append any other axes not mentioned
  alist = alist + [a for a in range(var.naxes) if a not in alist]
  #TODO: allow an argument of -1/None to indicate where to put the "other" axes?
  # No transpose necessary?
  if alist == list(range(var.naxes)): return var
  return TransposedVar (var, alist)

# Sort the axis values of a variable
class SortedVar(Var):
  def __init__(self, var, order):
    from pygeode.var import Var, copy_meta
    self.var = var
    outaxes = list(var.axes)

    for iaxis, o in order.items():
      reverse = {1:False, 0:None, -1:True}[o]
      outaxes[iaxis] = var.axes[iaxis].sorted(reverse=reverse)

    Var.__init__(self, outaxes, dtype=var.dtype)
    copy_meta (var, self)
  def getview (self, view, pbar):
    r = view.get(self.var, pbar=pbar)
    return r
#TODO: rename to sort_by_axis
def sorted (self, *iaxes, **kwargs):
  """
  Sorts the data so that the axes have monotonically increasing values.

  Parameters
  ----------
  *iaxes : (optional)
    Which axes to sort.  If not specified, then *all* axes are sorted.
    Axes can also be passed as keyword arguments, with a value of 1/0/-1 to
    specifiy a sort order of increasing/default/decreasing.
    If positional arguments are passed, or if an order of '0' is specified,
    then the natural (default) order will be used for that type of axis.

  Returns
  -------
  sorted_var : Var
    The sorted version of the data.

  Examples
  --------
  >>> from pygeode.tutorial import t2
  >>> print(t2.Temp)
  <Var 'Temp'>:
    Shape:  (time,pres,lat,lon)  (3650,20,31,60)
    Axes:
      time <ModelTime365>:  Jan 1, 2011 00:00:00 to Dec 31, 2020 00:00:00 (3650 values)
      pres <Pres>    :  1000 hPa to 50 hPa (20 values)
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  TransposedVar (dtype="float64")
  >>> print(t2.Temp.sorted('pres'))  # Sort by default/natural order
  <Var 'Temp'>:
    Shape:  (time,pres,lat,lon)  (3650,20,31,60)
    Axes:
      time <ModelTime365>:  Jan 1, 2011 00:00:00 to Dec 31, 2020 00:00:00 (3650 values)
      pres <Pres>    :  1000 hPa to 50 hPa (20 values)
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  SortedVar (dtype="float64")
  >>> # ^^ no change, since pressure was already in its natrual order (decreasing)
  >>> print(t2.Temp.sorted(pres=1))  # Sort pressure explicitly as increasing order
  <Var 'Temp'>:
    Shape:  (time,pres,lat,lon)  (3650,20,31,60)
    Axes:
      time <ModelTime365>:  Jan 1, 2011 00:00:00 to Dec 31, 2020 00:00:00 (3650 values)
      pres <Pres>    :  50 hPa to 1000 hPa (20 values)
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  SortedVar (dtype="float64")
  """

  # Build a dictionary of axes involved in this sort
  # (key = axis index, value = order (1/0/-1)
  order = {}
  for iaxis in iaxes:
    iaxis = self.whichaxis(iaxis)  # Get axis index
    order[iaxis] = 0 # Default order assumed for positional arguments
  # Do the same thing for keyword arguments, but use the order given
  for iaxis,o in kwargs.items():
    iaxis = self.whichaxis(iaxis)  # Get axis index
    assert o in (1,0,-1), "invalid order: %s"%o
    order[iaxis] = o
  # If no axes provided, then sort everything!
  if len(order) == 0:
    for iaxis in range(self.naxes):
      order[iaxis] = 0

  return SortedVar(self, order)

# Wrapper for replacing a variable's axes with new ones
# (the axes must be in 1:1 correspondence with the old ones)
class Replace_axes (Var):
  def __init__(self, var, axisdict={}, ignore_mismatch=False, newaxes=None, keep_old_name=True, **kwargs):
    from pygeode.var import Var, copy_meta
    from inspect import isclass
    axisdict = dict(axisdict, **kwargs)
    if newaxes is None:
      newaxes = list(var.axes)
    else:
      assert len(newaxes) == var.naxes, "wrong number of axes provided"

    for a,newa in axisdict.items():
      if not var.hasaxis(a) and ignore_mismatch: continue
      i = var.whichaxis(a)
      olda = var.axes[i]
      # Keep the old axis name?
      name = olda.name if keep_old_name else newa.name
      # Convert axis class to axis object, using the existing values?
      if isclass(newa):
        # Cram in any 'auxiliary' attributes, in case they're needed by the new class.
        # (Needed if, say, converting from StandardTime to ModelTime365)
        # Note: even if these attributes aren't pick up by the new init,
        # they'll get stored in the 'auxatts' field and stay there as benign,
        # unused values.  Ideally, if we knew ahead of time what attributes are
        # needed, we could pass only *those* attributes to the new class...
        newa = newa(olda.values, name=name, **olda.auxatts)
      # Use this new axis
      newaxes[i] = newa
    for a1, a2 in zip(newaxes, var.axes): assert len(a1) == len(a2)
    self.var = var
    Var.__init__(self, newaxes, dtype=var.dtype)
    copy_meta (var, self)
  def getview (self, view, pbar):
    from pygeode.view import View
    import numpy as np
    # Do a brute-force mapping of the indices to the internal axes
    # (should work if the axes are in 1:1 correspondence)
    data = View(self.var.axes, force_slices=view.slices,
                force_integer_indices=view.integer_indices).get(self.var, pbar=pbar)
    return data

# Function wrapper
# (use explicit arguments (instead of *x,**y) so we expose our 'ignore_mismatch'
# flag for callers to see.
def replace_axes (self, axisdict={}, ignore_mismatch=False, newaxes=None, keep_old_name=True, **kwargs):
  """
  Replaces one or more axes of a variable with new axes.  The axis length
  must remain the same.

  Parameters
  ----------
  axisdict : dict (optional)
    The keys are identifiers for the current axes, and the values are the
    replacement axes.
  ignore_mismatch : boolean (optional)
    If ``True``, will ignore axis identifiers that don't match any axes of
    this variable.  If ``False``, will raise an exception on a mismatch.
    Default is ``False``.
  newaxes : list of axes (optional)
    An explicit list of axes to use as replacements.  Useful if you want to
    replace *all* the axes at the same time.
  keep_old_name : boolean (optional)
    If ``True``, will keep the old axis name (with the new values).  If
    ``False``, the name ofthe new axis will be used.  Default is ``True``.
  **kwargs : keyword arguments
    Similar to 'axisdict', but using keyword parameters instead of a
    dictionary.

  Returns
  -------
  new_var : Var
    The same variable, but with different axes.  The variable data will remain
    unchanged.

  Examples
  --------
  >>> from pygeode.tutorial import t1
  >>> print(t1.Temp)
  <Var 'Temp'>:
    Units: K  Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  Add_Var (dtype="float64")
  >>> from pygeode import XAxis, YAxis
  >>> new_var = t1.Temp.replace_axes(lon=XAxis, lat=YAxis)
  >>> print(new_var)
  <Var 'Temp'>:
    Units: K  Shape:  (lat,lon)  (31,60)
    Axes:
      lat <YAxis>    :  -90  to 90  (31 values)
      lon <XAxis>    :  0  to 354  (60 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  >>> new_var = t1.Temp.replace_axes(lon=XAxis, lat=YAxis, keep_old_name=False)
  >>> print(new_var)
  <Var 'Temp'>:
    Units: K  Shape:  (yaxis,xaxis)  (31,60)
    Axes:
      yaxis <YAxis>  :  -90  to 90  (31 values)
      xaxis <XAxis>  :  0  to 354  (60 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  """
  var = self
  return Replace_axes (var, axisdict, ignore_mismatch, newaxes, keep_old_name, **kwargs)


# Rename a variable
# (wrapped in a new class, because the original var might *need* the original name, so we can't just make a shallow copy of the object and change the name.  Well, we can, but it might cause some very tricky bugs down the road...)
class RenamedVar(Var):
  def __init__ (self, var, newname):
    from pygeode.var import Var, copy_meta
    self.var = var
    Var.__init__(self, var.axes, dtype=var.dtype)
    copy_meta (var, self)
    self.name = newname
  def getview (self, view, pbar):  return view.get(self.var, pbar=pbar)

def rename (self, newname):
  """
  Assigns a new name to a variable

  Parameters
  ----------
  newname : string
    The new name of the variable.

  Returns
  -------
  renamed_var : Var
    The same variable, but with a different name.

  Notes
  -----
  In most cases, you could probably change the ``name`` attribute of the
  variable directly instead of calling this method.  However, if the variable
  is being used in other places, this method guarantees that the name change
  will only affect a local version of the variable, and won't have any
  side-effects on other existing references.

  Examples
  --------
  >>> from pygeode.tutorial import t1
  >>> print(t1.Temp.rename('i_prefer_really_long_variable_names'))
  <Var 'i_prefer_really_long_variable_names'>:
    Units: K  Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  RenamedVar (dtype="float64")
  """
  var = self
  return RenamedVar(var, newname)

# Rename the axes of a variable
def rename_axes (self, ignore_mismatch=False, axisdict={}, **namemap):
  """
  Renames the axes of a variable.

  Parameters
  ----------
  ignore_mismatch : boolean (optional)
    If ``True``, will ignore axis identifiers that don't match any axes of
    this variable.  If ``False``, will raise an exception on a mismatch.
    Default is ``False``.
  axisdict : dictionary
    An explicit dictionary mapping old names to new names.
  **namemap : keyword arguments
    One or more keyword-based arguments.  The parameters are the existing axis
    names, and the values are the new names to substitute.

  Returns
  -------
  new_var : Var
    The same variable, but with new names for the axes.

  Examples
  --------
  >>> from pygeode.tutorial import t1
  >>> print(t1.Temp)
  <Var 'Temp'>:
    Units: K  Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  Add_Var (dtype="float64")
  >>> new_var = t1.Temp.rename_axes(lat="latitude",lon="longitude")
  >>> print(new_var)
  <Var 'Temp'>:
    Units: K  Shape:  (latitude,longitude)  (31,60)
    Axes:
      latitude <Lat> :  90 S to 90 N (31 values)
      longitude <Lon>:  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  """
  var = self
  namemap = dict(axisdict, **namemap)
  for n1 in namemap.keys():
    if ignore_mismatch: continue
    assert var.hasaxis(n1), "'%s' not an axis of %s"%(n1,var)
  axisdict = dict([n1,var.getaxis(n1).rename(n2)] for n1,n2 in namemap.items() if var.hasaxis(n1))
  return var.replace_axes(keep_old_name=False, **axisdict)




# A sliced var
# i.e., var[i:j,m:n]
#TODO: remove degenerate axes from output?
class SlicedVar(Var):
# {{{

  def __init__ (self, var, slices):
  # {{{
    from pygeode.var import Var, copy_meta
    self.var = var
    copy_meta (var, self)

    #TODO: remove degenerate dimensions when slicing by integer values

    if not hasattr(slices,'__len__'): slices = [slices]
#    assert len(slices) == len(var.axes), "expected %i parameters, received %i."%(len(var.axes),len(slices))
    slices = list(slices)  # make a copy of the list

    # Append an implicit Ellipsis at the end (makes the logic a big simpler below)
#    if Ellipsis not in slices: slices.append(Ellipsis)
    if not any (sl is Ellipsis for sl in slices):
      slices.append(Ellipsis)

    # Handle Ellipsis argument
#    assert slices.count(Ellipsis) == 1, "can't handle more than one Ellipsis argument"
    ellipsis_index = [i for i,sl in enumerate(slices) if sl is Ellipsis]
    assert len(ellipsis_index) == 1, "can't handle more than one Ellipsis argument"

    num_missing = var.naxes - len(slices) + 1
    assert num_missing >= 0, "too many slices provided"
#    i = slices.index(Ellipsis)
    i = ellipsis_index.pop()
    slices = slices[:i] + [slice(None)]*num_missing + slices[i+1:]

    # Slice the output axes
    axes = [a.slice[s] for a,s in zip(var.axes,slices)]

    Var.__init__(self, axes, dtype=var.dtype, atts=self.atts, plotatts=self.plotatts)
  # }}}

  def getview (self, view, pbar):
  # {{{
    return view.get(self.var, pbar=pbar)
  # }}}
# }}}




# -----------
# Operations that superficially modify the data (casting, fill/unfil missing values, etc.)

# Replace 'NaN' with the specified fill value
class FillVar(Var):
  def __init__(self, var, fill):
    from pygeode.var import Var, copy_meta
    self.var = var
    self._fill = fill
    Var.__init__(self, var.axes, var.dtype)
    copy_meta(var, self)
  def getview (self, view, pbar):
    import numpy as np
    data = view.get(self.var, pbar=pbar).copy()
#    data[where(data == float('NaN'))] = self.fill
    w = np.where(np.isnan(data))
    data[w] = self._fill
    return data

def fill (self, fill):
  """
  Replaces ``NaN`` (missing values) with some fill value.

  Parameters
  ----------
  fill : float or int
    The fill value

  Returns
  -------
  filled_var : Var
    A new representation of the variable with missing values filled in.

  See Also
  --------
  unfill
  """
  var = self
  return FillVar(var, fill)

# Replace fill value with 'NaN'
# Ideally, this should be done at file load time, but sometimes the fill value
# isn't known or available for automatic extraction.
# NOTE: if input values are integers, they are converted to floats
class UnfillVar(Var):
  def __init__(self, var, fill):
    from numpy import float32, float64
    from pygeode.var import Var, copy_meta
    self.var = var
    self._fill = fill
    # We need floating-point values to have a nan
    if var.dtype not in (float32, float64):
      dtype = float32
    else: dtype = var.dtype
    Var.__init__(self, var.axes, dtype)
    copy_meta (var, self)
  def getview(self, view, pbar):
    import numpy as np
    data = view.get(self.var, pbar=pbar).copy()
    data = np.asarray(data, self.dtype)
    data[np.where(data == self._fill)] = float('NaN')
    return data

def unfill (self, fill):
  """
  Replaces all occurrences of the specified value with an ``NaN`` (missing
  value).

  Parameters
  ----------
  fill - float or int
    The value to treat as missing

  Returns
  -------
  unfilled_var : Var
    A new represntation of the variable with the 'fill' values removed
    (replaced with ``NaN``).

  See Also
  --------
  fill
  """
  var = self
  return UnfillVar(var, fill)


# Convert the data to the specified type
class Cast(Var):
  def __init__(self, var, dtype):
    from pygeode.var import Var, copy_meta
    self.var = var
    Var.__init__(self, var.axes, dtype)
    copy_meta (var, self)
  def getview(self, view, pbar):
    import numpy as np
    data = view.get(self.var, pbar=pbar)
    try:
      data = np.asarray(data, self.dtype)
    except ValueError:
      raise ValueError("unable to cast %s to %s"%(data.dtype, self.dtype))
    return data
  def __repr__(self):
    from pygeode.var import Var
    s = Var.__repr__(self)
    return s[:-1] + ' ' + self.dtype.name + s[-1]

def as_type(self, dtype):
  """
  Casts a variable to a new data type (I.e., float32, float64, etc.)

  Parameters
  ----------
  dtype : ``numpy.dtype`` or string
    The new data type to use.

  Returns
  -------
  casted_var : Var
    The same var, but with its values cast into a new type.

  Examples
  --------
  >>> from pygeode.tutorial import t1
  >>> # Treat the values as single-precision
  >>> print(t1.Temp.as_type('float32'))
  <Var 'Temp' float32>:
    Units: K  Shape:  (lat,lon)  (31,60)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      lon <Lon>      :  0 E to 354 E (60 values)
    Attributes:
      {}
    Type:  Cast (dtype="float32")
  """
  var = self
  return Cast(var, dtype)


# -----------




del Var
