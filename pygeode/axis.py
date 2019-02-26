#TODO: remove the 'concat' class method - put the code directly in the static 'concat' method.  There are no longer any cases where Axis subclasses need to overload the merging logic.
#TODO: change the arguments for 'concat' from axes to *axes
#TODO: remove NamedAxis class - mostly redundant
#TODO: make map_to a light wrapper for common_map, since the latter is a more powerful version of the method

from pygeode.var import Var

try:
  import pylab as pyl
  class AxisFormatter(pyl.Formatter):
  # {{{
    def __init__(self, axis, fmt=None, unitstr=None, units=False):
      if fmt is None:
        fmt = axis.plotatts.get('plotfmt', None)
        if fmt is None: fmt = axis.formatstr
      if unitstr is None:
        unitstr = axis.plotatts.get('plotunits', None)
        if unitstr is None: unitstr = axis.units

      self.fmt = fmt
      self.unitstr = unitstr
      self.showunits = units
      self.pygaxis = axis

    def __call__(self, x, pos=None):
      return self.pygaxis.formatvalue(x, fmt=self.fmt, unitstr=self.unitstr, units=self.showunits)
  # }}}
except ImportError:
  from warnings import warn
  warn ("Matplotlib not available; plotting functionality will be absent.")

# Axis parent class
class Axis(Var):
# {{{
  """
    An object that describes a single dimension of a :class:`Var` object.
    It is a subclass of :class:`Var`, so it can be used anywhere a Var would be
    used.

    Parameters
    ----------
    values : numpy.ndarray
      The coordinate values for each point along the axis.  Should be monotonic.
    name : string
      A name used to reference this axis.  Should be unique among the other
      axes of a variable.
    atts : dict
      Any additional metadata to associate with the axis.  The dictionary keys
      should be strings.

    See Also
    --------
    :doc:`var`

  """
  
  # Default dictionaries: these are class defaults and are overwritten by child class defaults    
  
  #: Auxiliary arrays. These contain additionnal fields beyond the regular value array.
  auxarrays = {}

  #: Auxiliary attributes. These are preserved during merge/slice/etc operations.
  auxatts = {}  

  #: Format specification for plotting values.
  formatstr = '%g'

  #: Relative tolerance for identifying two values of this axis as equal
  rtol = 1e-5

  #: Dictionary of attributes for plotting; see plotting documentation.
  plotatts = Var.plotatts.copy()

  def __init__(self, values, name=None, atts=None, plotatts=None, rtol=None, **kwargs):
# {{{ 
    """
    Create a new Axis object with the given values.

    Parameters
    ----------
    values : numpy.ndarray
        A one-dimensional coordinate defining the axis grid.
    name : string (optional)
        What to call the axis (i.e. for plot titles & when saving to file)
    atts : dict (optional)
        Any additional metadata to associate with the axis. The dictionary
        keys should be strings.
    plotatts : dict (optional)
        Parameters that control plotting behaviour; default values are available. 
        The dictionary keys should be strings.
    rtol : float
        A relative tolerance used for identifying an element of this axis.

    Notes
    -----
    All subclasses of :class:`Axis` need to call this __init__ method within
    their own __init__, to properly initialize all attributes. 

    """
    import numpy as np
    from pygeode.var import Var

    # If a single integer given, expand to an integer range
    #TODO: get rid of this?  if you want to use integer indices, then make an appropriate 'Index' axis subclass?
    if isinstance(values,int):
      values = list(range(values))

    values = np.asarray(values)

    # Read configuration details
    self.__class__._readaxisconfig(self)

    # Note: Call init before hasattr (or don't use hasattr at all in here)
    # (__getattr__ is overridden to call getaxis, which assumes axes are defined, otherwise __getattr__ is called to find an 'axes' property, ....)
    Var.__init__(self, [self], values=values, name=name, atts=atts, plotatts=plotatts)
 
    # Compute size of spacing relative to magnitude for relative tolerances when mapping
    if rtol is None:
      rtol = 1e-5
      rtol = self.rtol
      inz = np.where(values != 0.)[0]
      if len(inz) > 1:
        vnz = np.sort(values[inz]).astype('d')
        logr = np.floor(np.min( np.log10(np.abs(np.diff(vnz) / vnz[:-1])) ))
        if not np.isinf(logr) and 10**logr < rtol: rtol = 10**logr

    #: The relative tolerance for identifying an element of this axis.
    self.rtol = rtol 

    # Add auxilliary arrays after calling Var.__init__ - the weights
    # array, if present, will be added here, not by the logic in Var.__init___
    auxarrays = {}; auxatts = {}
    for key, val in kwargs.items():
      if isinstance(val,Var): val = val.get()
      if isinstance(val,(list,tuple,np.ndarray)):
        val = np.asarray(val)
        if val.shape != self.values.shape:
          raise ValueError('Auxilliary array %s has the wrong shape.  Expected %s, got %s' % (key,self.values.shape, val.shape))
        auxarrays[key] = val
      else:
        auxatts[key] = val
        
    # update auxiliary attribute (make copy to not change class defaults)        
    self.auxarrays = self.__class__.auxarrays.copy()
    self.auxarrays.update(auxarrays.copy())
    self.auxatts = self.__class__.auxatts.copy() 
    self.auxatts.update(auxatts.copy())    
# }}}

  # 
  @classmethod
  def isparentof(cls,other):
  # {{{
    """ 
    Determines if an axis object is an instance of a base class (or the same
    class) of another axis.

    Parameters
    ==========
    other : :class:`Axis` object to compare against this one.

    Returns
    =======
    bool : boolean
      True if ``other`` is an instance of this object's class
    """
    return isinstance(other,cls)
  # }}}

  @classmethod
  def _readaxisconfig(cls, ax):
  # {{{
    from pygeode import _config
    c = cls
    nm = c.__name__
    while c is not Axis:
      if _config.has_option('Axes',  nm + '.name'):
        ax.name = str(_config.get('Axes', nm + '.name'))
        break
      else:
        c = c.__bases__[0]
        nm = c.__name__

    if c is Axis: ax.name = cls.__name__.lower()

    # Set basic attributes
    for p in ['formatstr', 'units']:
      if _config.has_option('Axes',  nm + '.' + p):
        setattr(ax, p, _config.get('Axes', nm + '.' + p))

    for p in ['rtol']:
      if _config.has_option('Axes',  nm + '.' + p):
        setattr(ax, p, _config.getfloat('Axes', nm + '.' + p))

    # Set plot attributes
    for p in ['plottitle', 'plotfmt', 'plotscale']:
      if _config.has_option('Axes',  nm + '.' + p):
        ax.plotatts[p] = _config.get('Axes', nm + '.' + p)

    for p in ['plotorder']:
      if _config.has_option('Axes',  nm + '.' + p):
        ax.plotatts[p] = int(_config.getfloat('Axes', nm + '.' + p))
  # }}}

  #TODO: fix inconsistency between Axis and Var, for == and !=
  #      Vars produce a boolean mask under those operations, Axes return scalar True/False
  # I.e., "lat == 30" and "(lat*1) == 30" give very different results!
  def __ne__ (self, other): return not self.__eq__(other)
  def __eq__ (self, other):
  # {{{
    '''override Var's ufunc stuff here
       this is a weak comparison, in that we only require the other
       axis to be a *subclass* of this one.
       this allows things like "lat in [time,gausslat,lev]" to evaluate to True
       If you want a more strict comparison, then check the classes explicitly yourself.'''

    #TODO: do some testing to see just how many times this is called, if it will be a bottleneck for large axes

#    print '<< Axis.__eq__ on', repr(self), 'and', repr(other), '>>'

    # exact same object?
    if self is other: return True
    # incomparable?
    if not isinstance(other,Axis):
#      print 'not an axis?'
      return False
    if not self.isparentof(other) and not other.isparentof(self):
#      print 'parent issues'
      return False

    # If they are generic Axis objects, an additional requirement is that they have the same name
    if self.__class__ is Axis and other.__class__ is Axis:
      if self.name != other.name: return False

    # Check if they have the same lengths
    if len(self.values) != len(other.values):
#      print 'false by length'
      return False

    # Check the values
    from numpy import allclose
    if not allclose(self.values, other.values):
#      print 'values mismatch'
      return False

    # Check auxiliary attributes
    if set(self.auxatts.keys()) != set(other.auxatts.keys()): return False
    for fname in self.auxatts.keys():
      if self.auxatts[fname] != other.auxatts[fname]: return False

    # Check any associated arrays
    if set(self.auxarrays.keys()) != set(other.auxarrays.keys()):
#      print 'false by mismatched set of auxarrays'
      return False

    # Check values of associated arrays
    for fname in self.auxarrays.keys():
      if not allclose(self.auxarrays[fname], other.auxarrays[fname]):
#        print 'false by mismatched auxarray "%s":'%fname
        return False

    return True

  # }}}

  def alleq (self, *others):
  # {{{
    ''' alleq(self, *others) - returns True if self matches with all axes in others.'''
    for other in others:
      if not self.__eq__(other): return False
    return True
  # }}}

  #TODO: include associated arrays when doing the mapping?
  def map_to (self, other):
  # {{{
    '''Returns indices of this axis which correspond to the axis ``other``. 
    
       Parameters
       ----------
       other : :class:`Axis` 
         Axis to find mapping to

       Returns
       -------
       mapping : integer array or None

       Notes
       -----

       Returns an ordered indices of the elements of this axis that correspond to those of
       the axis ``other``, if one exists, otherwise None. This axis must be a
       parent class of ``other`` or vice versa in order for the mapping to
       exist. The mapping may include only a subset of this axis object, but
       must be as long as the other axis, if it is not None. The mapping
       identifies equivalent elements based on equality up to a tolerance
       specified by self.rtol. 
       '''

    from pygeode.tools import map_to
    import numpy as np
    if not self.isparentof(other) and not other.isparentof(self): return None

    # special case: both axes are identical
    if self == other: return np.arange(len(self))

    # Use less conservative tolerance?
    #rtol = max(self.auxatts.get('rtol', 1e-5), other.auxatts.get('rtol', 1e-5))

    return map_to(self.values, other.values, self.rtol)
  # }}}

  def sorted (self, reverse=None):
# {{{
    """
    Sorts the points of the Axis.

    Parameters
    ----------
    reverse : boolean (optional)
      If ``True``, sorts in descending order. If ``False``, sorts in ascending order.
      By default the sign of self.plotorder is used.

    Returns
    -------
    sorted_axis : Axis
      A sorted version of the input axis.

    Examples
    --------
    >>> from pygeode import Lat
    >>> x = Lat([30,20,10])
    >>> print x 
    >>> y = x.sorted() 
    >>> print y

    See Also
    --------
    argsort
    """
    S = self.argsort(reverse=reverse)
    return self.slice[S]
# }}}

  def argsort (self, reverse=None):
# {{{
    """
      Generates a list of indices that would sort the Axis.

      Parameters
      ----------
      reverse : boolean (optional)
        If ``False``, indices are in ascending order. If ``True``, will produce 
        indices for a *reverse* sort instead. By default, sign of self.plotorder is used.

      Returns
      -------
      indices : list
        The indices which will produces a sorted version of the Axis.

      Examples
      --------
      >>> from pygeode import Lat
      >>> x = Lat([20,30,10])
      >>> print x
      lat <Lat>      :  20 N to 10 N (3 values)
      >>> indices = x.argsort()
      >>> print indices
      [2 0 1]
      >>> print x.slice[indices]
      lat <Lat>      :  10 N to 30 N (3 values)


      See Also
      --------
      sorted
    """
    import numpy as np
    S = np.argsort(self.values)
    step = 1
    if reverse is None: step = self.plotatts.get('plotorder', 1)
    if reverse is True: step = -1
    return S[::step]
# }}}

  #TODO: implement and test this (if it's ever needed?)
  # (make sure to check auxiliary arrays)
  """
  def common_map (self, other):
    '''return the indices that map common elements from one axis to another'''
    from pygeode.tools import common_map
    assert self.isparentof(other) or other.isparentof(self)
    return common_map(self.values, other.values)
  """

  # The length of an axis (equal to the length of the array of values)
  def __len__ (self): return len(self.values)

  """
  # Avoid accidentally iterating over axes
  # (Would cause a new, 1-element axis to be created for each iteration)
  def __iter__ (self):
    raise Exception ("Axes cannot be iterated over")
  """
  # Iterating over an axis iterates over the values
  def __iter__ (self):  return iter(self.values)

  # Slice an axis -> construct a new one with the sliced values
  # (Overridden from Var to preserve our Axis status)
  def _getitem_asvar (self, slices):
  # {{{
    import numpy as np

    values = np.array(self.values[slices], ndmin=1, copy=False)

    # Check if we even need to do any slicing
    if len(values) == len(self.values) and np.all(values==self.values): return self

    aux = {}

    # Keep auxiliary attributes
    for key,val in self.auxatts.items():
      aux[key] = val

    # Slice auxiliary arrays
    for key,val in self.auxarrays.items():
      aux[key] = np.array(val[slices], ndmin=1)

    axis = type(self)(values, name=self.name, atts=self.atts, **aux)

    return axis
  # }}}

  def str_as_val(self, key, s):
# {{{
    '''str_as_val(self, key, s) - converts string s to a value corresponding to this axis. Default
        implementation returns float(s); derived classes can return different conversions depending
        on whether key corresponds to the axis itself or an auxiliary array.'''
    return float(s)
# }}}

  def get_slice(self, kwargs, ignore_mismatch=False):
# {{{
    import numpy as np
    from pygeode.view import simplify, expand

    # boolean flags indicating which axis indices will be used
    n = len(self)
    keep = np.ones(n,bool)
    matched = []

    for k, v in kwargs.items():
       # Split off prefix if present
      if '_' in k and not self.has_alias(k):
        prefix, ax = k.split('_', 1)
      else:
        prefix, ax = '', k

      if 'i' in prefix: 
        ################### Select by index; key must correspond to this axis
        if not self.has_alias(ax):
          if ignore_mismatch: continue
          raise Exception("'%s' is not associated with this %s axis" % (ax, self.name))

        # Build mask
        kp = np.zeros(n, bool)    

        if not hasattr(v, '__len__'): # Treat as an index
          kp[v] = True
        elif len(v) > 3 or len(v) < 2 or 'l' in prefix: # Treat as integer array
          kp[np.array(v, 'int')] = True
        elif len(v) == 2:         # Treat as slice
          kp[v[0]:v[1]] = True
        elif len(v) == 3:         # Treat as slice with stride
          kp[v[0]:v[1]:v[2]] = True
        else:
          raise ValueException("'%s' is not associated with this %s axis" % (ax, self.name))
      else:
        ################### Select by value
        if self.has_alias(ax):     # Does key match this axis?
          vals = self.values
        elif ax in self.auxarrays: # What about an aux. array?
          vals = self.auxarrays[ax]
        else:
          if ignore_mismatch: continue
          raise Exception("'%s' is not associated with this %s axis" % (ax, self.name))

        # Build mask
        kp = np.zeros(n, bool)  
        # Convert string representation if necessary
        if isinstance(v, str): v = self.str_as_val(ax, v) 

        if isinstance(v,str) or not hasattr(v,'__len__'): # Single value given
          if vals.dtype.name.startswith('float'): # closest match 
            kp[np.argmin( np.abs(v-vals) )] = True
          else:                 # otherwise require an exact match
            kp[vals == v] = True

        elif 'l' in prefix:
          for V in v:
            # Convert string representation if necessary
            if isinstance(V, str): V = self.str_as_val(ax, V) 

            if vals.dtype.name.startswith('float'): # closest match 
              kp[np.argmin( np.abs(V-vals) )] = True
            else:                 # otherwise require an exact match
              kp[vals == V] = True

        elif len(v) == 2:       # Select within range
          # Convert string representations if necessary
          v = [self.str_as_val(ax, V) if isinstance(V, str) else V for V in v]
          lower, upper = min(v), max(v)
          kp[(lower <= vals) & (vals <= upper)] = True

        else:                   # Don't know what to do with more than 2 values
          raise Exception('A range must be specified')

      # Use complement of requested set
      if 'n' in prefix: kp = ~kp

      # Compute intersection of index sets
      keep &= kp
      matched.append(k)       # Mark for removal from slice list

    # Pop kw arguments that have been handled
    for m in matched:
      kwargs.pop(m)

    # Convert boolean mask to integer indices
    sl = np.flatnonzero(keep)

    # Filter through view.simplify() to construct slice objects wherever possible
    # (otherwise, it's a generic integer array)
    return simplify(sl)
# }}}

  # Keyword/value based slicing of an axis
  # (Overridden from Var to preserve our Axis status)
  def __call__ (self, **kwargs):
# {{{
    sl = self.get_slice(kwargs)
    return self._getitem_asvar(sl)
# }}}

  # Get an axis attribute
  # Overloaded from pygeode.Var to allow shortcuts to auxiliary arrays
  def __getattr__ (self, name):
# {{{
    # Disregard metaclass stuff
    if name.startswith('__'): raise AttributeError

#    print 'axis getattr ??', name
    from pygeode.var import Var
    if name in self.auxarrays: return self.auxarrays[name]
    if name in self.auxatts: return self.auxatts[name]
    return Var.__getattr__(self, name)
# }}}

  def auxasvar (self, name):
# {{{
    ''' Returns auxiliary array as a new :class:`Var` object.

        Parameters
        ==========
        name : string
            Name of auxiliary array to return

        Returns
        =======
        var : :class:`Var`
            Variable with values of requested auxilliary array

        See Also
        ========
        auxarrays
    '''
    from pygeode.var import Var
    return Var([self], values=self.auxarrays[name], name=name)
# }}}

  # Pretty printing
  def __repr__ (self): return '<' + self.__class__.__name__ + '>'

  def __str__ (self):
  # {{{
    if len(self) > 0:
      first = self.formatvalue(self.values[0])
      last = self.formatvalue(self.values[-1])
    else: first = last = "<empty>"
    num = str(len(self.values))
    if self.name != '': head = self.name + ' ' + repr(self)
    else: head = repr(self)

    if len(self) > 1:
      out = head.ljust(15) + ':  '+first+' to '+last+' ('+num+' values)'
    else:
      out = head.ljust(15) + ':  '+first

#    return out+"\n"
    return out
  # }}}

  # Rename an axis
  def rename (self, name):
# {{{
    """
    Assigns a new name to this axis.

    Parameters
    ----------
    name : string
      The new name of this axis.

    Returns
    -------
    renamed_axis : Axis
      An instance of the same axis class with the new name.
    """
    aux = {}
    for k,v in self.auxatts.items(): aux[k] = v
    for k,v in self.auxarrays.items(): aux[k] = v

    return type(self)(values=self.values, name=name, atts=self.atts, **aux)
# }}}

  # Check if a given string is a meaningful alias to the axis
  # (i.e., if the string matches the name, or the class name, or one of the base class names)
  @classmethod
  def class_has_alias (cls, name):
  # {{{
#    if cls is Axis: return False  # need a uniquely identifiable subclass of Axis
    # Default string name for the class
    if cls.name.lower() == name.lower(): return True
    # A stringified version of the class name (i.e. Time => 'time')
    if cls.__name__.lower() == name.lower(): return True
    for subclass in cls.__bases__:
      if not issubclass(subclass,Axis): continue  # only iterate over Axis subclasses
      if subclass.class_has_alias(name): return True
    return False
  # }}}

  # Now, a function which can work on instances of a class
  # Extends the above to also check the name given to the particular instance
  # (which can depend on the source of the data loaded at runtime)
  def has_alias (self, name):
  # {{{
    assert isinstance(name, str)
    if self.name.lower() == name.lower(): return True
    return self.class_has_alias(name)
  # }}}

  # Concatenate multiple axes together
  # Use numpy arrays
  # Assume the segments are pre-sorted
  @classmethod
  def concat (cls, axes):
  # {{{
    from numpy import concatenate
    from pygeode.tools import common_dict
    # Must all be same type of axis
    for a in axes: assert isinstance(a,cls), 'axes must be the same type'

    values = concatenate([a.values for a in axes])

    # Get common attributes
    atts = common_dict([a.atts for a in axes])

    aux = {}

    # Check that all pieces have the same auxiliary attributes, and propogate them to the output.
    auxkeys = set(axes[0].auxatts.keys())
    for a in axes[1:]:
      auxkeys = auxkeys.intersection(list(a.auxatts.keys()))
    for k in auxkeys:
      vals = [a.auxatts[k] for a in axes]
      v1 = vals[0]
#      assert all(v == v1 for v in vals), "inconsistent '%s' attribute"%k
      # Only use consistent aux atts
      if all(v == v1 for v in vals):
        aux[k] = axes[0].auxatts[k]


    # Find and concatenate auxilliary arrays common to all axes being concatenated
    auxkeys = set(axes[0].auxarrays.keys())
    for a in axes[1:]:      # set.intersection takes multiple arguments only in python 2.6 and later..
      auxkeys = auxkeys.intersection(list(a.auxarrays.keys()))

    for k in auxkeys:
      aux[k] = concatenate([a.auxarrays[k] for a in axes])

    name = axes[0].name  #TODO: check all names?

    return cls(values, name=name, atts=atts, **aux)
  # }}}

  # Replace the values of an axis
  # (any auxiliary arrays from the old axis are ignored)
  def withnewvalues (self, values):
  # {{{
    # Assume any auxiliary scalars are the same for the new axis
    return type(self)(values, name=self.name, atts=self.atts, **self.auxatts)
  # }}}
# }}}


# Useful axis subclasses

# Named axis
class NamedAxis (Axis):
# {{{
  '''Generic axis object identified by its name.'''

  def __init__ (self, values, name, **kwargs):
  # {{{
    Axis.__init__(self, values, **kwargs)
    self.name = name
  # }}}

  def __eq__ (self, other):
  # {{{
    if type(other) is not NamedAxis: return False

    # Check the names
    if self.name != other.name: return False
    # If the names match, check the values
    return Axis.__eq__(self,other)
  # }}}

  def __repr__ (self): return "<%s '%s'>"%(self.__class__.__name__, self.name)

  # Need more restrictions on mapping for named axes
  # (not only do both axes need to be a NamedAxis, but they need to have the same name)
  # (The name is the only way to uniquely identify them)
  def map_to (self, other):
        
    if not isinstance(other, NamedAxis): return None
    if other.name != self.name: return None
    return Axis.map_to(self, other)
# }}}

# Dummy axis (values are just placeholders).
# Useful when there is no intrinsic coordinate system for a dimension.
# Example could be a dimension in a netCDF file that has no variable
# associated with it.
class DummyAxis (NamedAxis): pass

class XAxis (Axis): pass
class YAxis (Axis): pass

class Lon (XAxis): 
# {{{
  ''' Longitude axis. '''
  name = 'lon'
  #name = _config.get('Axes', 'Lon.name')
  formatstr = '%.3g E<360'

  plotatts = XAxis.plotatts.copy()
  plotatts['plottitle'] = ''
  plotatts['plotfmt'] = '%.3g E'

  def formatvalue(self, value, fmt=None, units=False, unitstr=None):
# {{{
    '''
    Returns formatted string representation of longitude ``value``.
      
    Parameters
    ----------
    value : float or int
      Value to format.
    fmt : string (optional)
      Format specification; see Notes. If the default ``None`` is specified, 
      ``self.formatstr`` is used.
    units : boolean (optional)
      If ``True``, will include the units in the string returned.
      Default is ``False``.
    unitstr : string (optional)
      String to use for the units; default is ``self.units``.

    Returns
    -------
    Formatted representation of the latitude. See notes.

    Notes
    -----
    If fmt is includes the character 'E', the hemisphere ('E' or 'W') is
    appended to the string. The value is wrapped to lie between 0 and 360., and
    the eastern hemisphere is defined as values less than 180. Optionally, the
    boundary between the hemispheres can be specified by adding, for example,
    '<360' after the 'E'; in this case all values less than 360 would have 'E'
    appended. Otherwise the behaviour is like :func:`Var.formatvalue`.
    
    Examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print t1.lon.formatvalue(270)
      270 E
    >>> print t1.lon.formatvalue(-20.346, '%.4gE')
      20.35W
    >>> print t1.lon.formatvalue(-192.4, '%.3g')
      -192
    '''
    if fmt is None: fmt = self.formatstr
    if 'E' in fmt:
      fmt, orig = fmt.split('E')
      try: orig = 360. - float(orig[1:])
      except ValueError: orig = 180.

      v = (value + orig) % 360. - orig
      if v >= 0: strval = fmt % v + 'E'
      elif v < 0: strval = fmt % -v + 'W'
    else:
      strval = fmt % value

    if units:
      strval += ' ' + self.units

    return strval
# }}}

  def locator(self):
  # {{{
    import pylab as pyl
    return pyl.MaxNLocator(nbins=9, steps=[1, 3, 6, 10])
  # }}}
# }}}

def regularlon(n, origin=0., order=1, repeat_origin=False):
# {{{
  '''Constructs a regularly spaced :class:`Lon` axis with n longitudes. The
  values range from origin to origin + 360. If repeat_origin is set to True,
  the final point is equal to origin + 360. '''
  import numpy as np
  vals = np.linspace(0., 360, n, endpoint=repeat_origin)[::order] + origin

  return Lon(vals)
# }}}

class Lat (YAxis):
# {{{
  ''' Latitude axis. '''
  name = 'lat'
  formatstr = '%.2g N'
  plotatts = YAxis.plotatts.copy() 
  plotatts['plottitle'] = ''

  # Make sure we get some weights
  def __init__(self, values, weights=None, **kwargs):
  # {{{
    from numpy import cos, asarray, pi
    # Input weights are only along latitude, not area weight
    # Output weights are area weights
    # If no input weights given, assume uniform
    #TODO: handle non-uniform latitudes?
    if weights is None: 
      weights = cos(asarray(values) * pi / 180.)

    Axis.__init__(self, values, weights=weights, **kwargs)
  # }}}

  def formatvalue(self, value, fmt=None, units=False, unitstr=None):
# {{{
    '''
    Returns formatted string representation of latitude ``value``.
      
    Parameters
    ----------
    value : float or int
      Value to format.
    fmt : string (optional)
      Format specification; see Notes. If the default ``None`` is specified, 
      ``self.formatstr`` is used.
    units : boolean (optional)
      If ``True``, will include the units in the string returned.
      Default is ``False``.
    unitstr : string (optional)
      String to use for the units; default is ``self.units``.

    Returns
    -------
    Formatted representation of the latitude. See notes.

    Notes
    -----
    If the last character of fmt is 'N', then the absolute value of ``value``
    is formatted using the fmt[:-1] as the format specification, and the hemisphere is 
    added, using 'N' for values greater than 0 and 'S' for values less than 0. The value
    0 is formatted as 'EQ'. Otherwise the behaviour is like :func:`Var.formatvalue`.
    
    Examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print t1.lat.formatvalue(0)
      EQ
    >>> print t1.lat.formatvalue(-43.61, '%.3gN')
      43.6S
    >>> print t1.lat.formatvalue(-43.61, '%.3g')
      -43.6
    '''

    if fmt is None: fmt = self.formatstr
    if fmt[-1] == 'N': 
      fmt = fmt[:-1]
      if value > 0: strval = fmt % value + 'N'
      elif value < 0: strval = fmt % -value + 'S'
      else: strval = 'EQ' 
    else:
      strval = fmt % value

    if units:
      strval += ' ' + self.units

    return strval
# }}}

  def locator(self):
  # {{{
    import pylab as pyl
    return pyl.MaxNLocator(nbins=9, steps=[1, 1.5, 3, 5, 10])
  # }}}
# }}}

def gausslat (n, order=1, axis_dict={}):
# {{{
  '''Constructs a Gaussian :class:`Lat` axis with n latitudes.'''
  from pygeode.quadrulepy import legendre_compute
  import numpy as np
  from math import pi
  if (n,order) in axis_dict: return axis_dict[(n,order)]
  x, w = legendre_compute(n)
  x = np.arcsin(x) / pi * 180

  x = x[::order]
  w = w[::order]

  axis = Lat (x, weights=w)
  axis_dict[(n,order)] = axis
  return axis
# }}}

def regularlat(n, order=1, inc_poles=True):
# {{{
  '''Constructs a regularly spaced :class:`Lat` axis with n latitudes.
  If inc_poles is set to True, the grid includes the poles. '''
  import numpy as np
  if inc_poles: vals = np.linspace(-90, 90, n)[::order]
  else: vals = np.linspace(-90, 90, n+2)[1:-1][::order]

  return Lat(vals)
# }}}

# Spectral axes
# Note: XAxis/YAxis is used by cccma code to put these in the proper order (XAxis is fastest-increasing)
class SpectralM(YAxis): name = 'm'
class SpectralN(XAxis): name = 'n'

# Vertical axes
class ZAxis (Axis): 
# {{{
  name = 'lev'
  formatstr = '%3g'
# }}}

# Geometric height
#TODO: weights
#TODO: attributes
class Height(ZAxis):
# {{{  
  ''' Geometric height axis. '''
  name = 'z' # default name
  formatstr = '%d' 
  units = 'm'
  plotatts = ZAxis.plotatts.copy()
  plotatts['plotname'] = 'Height' # name displayed in plots (axis label)
# }}}

# Model hybrid levels
#TODO: weights!
class Hybrid (ZAxis):
# {{{
  ''' Hybridized vertical coordinate axis. '''
  name = 'eta'  #TODO: rename this to 'hybrid'?  (keep 'eta' for now, for compatibility with existing code)
  formatstr = '%g'
  plotatts = ZAxis.plotatts.copy()
  plotatts['plotorder'] = -1
  plotatts['plotscale'] = 'log'

  def __init__ (self, values, A, B, **kwargs):
  # {{{
    # Just pass all the stuff to the superclass
    # (All we do here is enforce the existence of 'A' and 'B' associated arrays
    ZAxis.__init__ (self, values, A=A, B=B, **kwargs)
  # }}}

  def __eq__ (self, other):
  # {{{
    if not ZAxis.__eq__(self, other): return False
    from numpy import allclose
    if not allclose(self.A, other.A): return False
    if not allclose(self.B, other.B): return False
    return True
  # }}}

  def locator(self):
    import pylab as pyl, numpy as np
    ndecs = np.log10(np.max(self.values) / np.min(self.values))
    if ndecs < 1.2: return pyl.LogLocator(subs=[1., 2., 4., 7.])
    elif ndecs < 3.: return pyl.LogLocator(subs=[1., 3.])
    else: return pyl.LogLocator()
# }}}

class Pres (ZAxis): 
# {{{
  ''' Pressure height axis. '''
  name = 'pres'
  units = 'hPa'
  formatstr = '%.2g<100'
  plotatts = ZAxis.plotatts.copy() 
  plotatts['plotname'] = 'Pressure'
  plotatts['plotscale'] = 'log'
  plotatts['plotorder'] = -1

  def logPAxis(self, p0=1000., H=7.1):
# {{{
    '''logPAxis(p0, H) - returns a pygeode axis with log pressure heights
          corresponding to this axis. By default p0 = 1000 hPa and H = 7.1 (km).'''
    import numpy as np
    z = ZAxis(H * np.log(p0 / self.values))
    z.plotatts['plotname'] = 'Log-p Height'
    return z
# }}}

  def locator(self):
# {{{
    import pylab as pyl, numpy as np
    ndecs = np.log10(np.max(self.values) / np.min(self.values))
    if ndecs < 1.2: return pyl.LogLocator(subs=[1., 2., 4., 7.])
    elif ndecs < 3.: return pyl.LogLocator(subs=[1., 3.])
    else: return pyl.LogLocator()
# }}}

  def formatvalue(self, value, fmt=None, units=True, unitstr=None):
# {{{
    '''
    Returns formatted string representation of pressure ``value``.
      
    Parameters
    ----------
    value : float or int
      Value to format.
    fmt : string (optional)
      Format specification; see Notes. If the default ``None`` is specified, 
      ``self.formatstr`` is used.
    units : boolean (optional)
      If ``True``, will include the units in the string returned.
      Default is ``True``.
    unitstr : string (optional)
      String to use for the units; default is ``self.units``.

    Returns
    -------
    Formatted representation of the pressure. See notes.

    Notes
    -----
    If ``fmt`` includes the character '<', it is interpreted as 'fmt<break',
    such that values less than float(break) are formatted using the format
    string fmt, while those greater than float(break) are formatted as integers.
    Otherwise the behaviour is like :func:`Var.formatvalue`. 
    
    Examples
    --------
    >>> from pygeode.tutorial import t2
    >>> print t2.pres.formatvalue(1000)
      1000 hPa
    >>> print t2.pres.formatvalue(1.52)
      1.5 hPa
    >>> print t2.pres.formatvalue(20, '%.1g')
      2e+01 hPa
    >>> print t2.pres.formatvalue(20, '%.1g<10')
      20 hPa
    '''

    if fmt is None: fmt = self.formatstr
    if '<' in fmt:
      lfmt, brk = fmt.split('<')
      if value >= float(brk): strval = '%d' % value
      else: strval = lfmt % value
    else: 
      strval = fmt % value

    if units:
      strval += ' ' + self.units

    return strval
# }}}
# }}}

class TAxis(Axis): pass
# NOTE: Time axis is in pygeode.timeaxis
# It's a fairly heavyweight class, worthy of its own module.
# Importing it here would create a circular reference that would bugger things up


class Freq(Axis):
# {{{
  name = 'freq'
#  plotscale = 'log'
  def __init__ (self, values, inv_units=None, *args, **kwargs):
    if inv_units is not None:
      self.units = '/'+inv_units
      self.plotatts['plottitle'] = 'Frequency (per %s)'%inv_units
    kwargs = kwargs.copy()
    kwargs['inv_units'] = inv_units
    Axis.__init__ (self, values, *args, **kwargs)
# }}}

# Indexing arrays (represent discrete items, such as EOF number, ensemble number, etc.)
class Index(Axis):
# {{{
  def __init__ (self, n, *args, **kwargs):
    values = n if hasattr(n,'__len__') else list(range(n))
    Axis.__init__(self, values, *args, **kwargs)
# }}}

# Coefficient number (when returning a set of coefficients, such as for a polynomial fit)
class Coef(Index): pass

class NonCoordinateAxis(Axis):
# {{{
  '''Non-coordinate axis (disables nearest-neighbour value matching, etc.)'''
  # Refresh the coordinate values (should always be monotonically increasing integers).
  def __init__ (self, *args, **kwargs):
# {{{
    import numpy as np
    lengths = [len(kw) for kw in list(kwargs.values()) if isinstance(kw,(list,tuple,np.ndarray))]
    if len(lengths) == 0:
      raise ValueError("Unable to determine a length for the non-coordinate axis.")
    N = lengths[0]
    kwargs['values'] = np.arange(N)
    Axis.__init__(self, **kwargs)
    # Remember original name
    self._name = self.name
# }}}

  # Modify test for equality to look for an exact match
  #TODO: Make the default Axis.__eq__ logic do this, and move the "close"
  # matching to a subclass of Axis.
  def __eq__ (self, other):
# {{{
    # For simplicity, expect them to be the same type of axis.
    if type(self) != type(other): return False
    if set(self.auxarrays.keys()) != set(other.auxarrays.keys()): return False
    for k in list(self.auxarrays.keys()):
      if list(self.auxarrays[k]) != list(other.auxarrays[k]): return False
    return True
# }}}

  # How to map string values to dummy indices
  def str_as_val(self, key, s):
# {{{
    # Special case: referencing an aux array with the same name as the axis.
    if self._name in self.auxarrays:
      values = list(self.auxarrays[self._name])
      if s in values:
        return values.index(s)
    # Otherwise, return an invalid index (no match)
    return -1
# }}}

  # Modify formatvalue to convert dummy indices to the appropriate values
  def formatvalue(self, value, fmt=None, units=False, unitstr=None):
# {{{
    # Check if the value is in range.
    if value not in list(range(len(self))):
      return "?%s?"%value
    # Check if we have a special aux array with the same name as the axis.
    # In this case, use those values as the string.
    if self._name in self.auxarrays:
      return str(self.auxarrays[self._name][value])
    # Otherwise, return all the key/value pairs for all aux arrays.
    out = [name+"="+str(array[value]) for name,array in self.auxarrays.items()]
    out = ",".join(out)
    return "("+out+")"
# }}}

  # Modify map_to do use exact matching.
  # (Avoids use of tools.map_to, which assumes the values are numerical)
  #TODO: Make this the default for Axis (don't assume we have numerical values?)
  def map_to (self, other):
# {{{
    import numpy as np
    # Only allow mapping non-coordinate axes if they're the exact same type.
    if not isinstance(other,type(self)): return None
    # Get keys to use for comparing aux arrays
    keys = list(set(self.auxarrays.keys()) & set(other.auxarrays.keys()))
    if len(keys) == 0: return None
    values = list(zip(*[self.auxarrays[k] for k in keys]))
    other_values = list(zip(*[other.auxarrays[k] for k in keys]))
    #TODO: Speed this up? (move this to tools.c?)
    values_set = set(values)
    indices = []
    for v in other_values:
      if v not in values_set: continue
      indices.append(values.index(v))
    return indices
# }}}
# }}}

class Station(NonCoordinateAxis):
# {{{
  '''Station axis (for timeseries data at fixed station locations)'''
  name = "station"
# }}}

# Concatenate a bunch of axes together.
# Find a common parent class for all of them, and call that class's concat function.
def concat (axes):
# {{{
  axes = list(axes) # in case we're given a set, generator, etc.
  assert len(axes) > 0, 'nothing to concatenate!'
  # Degenerate case: only 1 axis provided
  if len(axes) == 1: return axes[0]

  cls = type(axes[0])
  for a in axes:
    cls2 = type(a)
    if issubclass(cls, cls2): cls = cls2
    assert issubclass(cls2, cls), "can't concatenate incompatible axes"
  return cls.concat(axes)
# }}}


# List of axes provided in this module (for easy importing)
standard_axes = [Axis, NamedAxis, XAxis, YAxis, ZAxis, TAxis, Lon, NonCoordinateAxis, regularlon, Lat, gausslat, regularlat, Pres, Hybrid, Height, SpectralM, SpectralN, Freq, Index]
