#TODO: remove the 'concat' class method - put the code directly in the static 'concat' method.  There are no longer any cases where Axis subclasses need to overload the merging logic.
#TODO: change the arguments for 'concat' from axes to *axes
#TODO: remove NamedAxis class - mostly redundant
#TODO: make map_to a light wrapper for common_map, since the latter is a more powerful version of the method

from pygeode.var import Var

import pylab as pyl
class AxisFormatter(pyl.Formatter):
# {{{
  def __init__(self, plotatts, units):
    self.plotatts = plotatts
    self.units = units

  def format(self, val, units=False):
    val = val*self.plotatts['scalefactor'] + self.plotatts['offset'] # apply scalefactor and offset 
    strval = self.plotatts['formatstr'] % val # convert to string
    if units:
      if self.plotatts['plotunits']: strval += self.plotatts['plotunits'] # properly formatted units
      elif self.plotatts['scalefactor']==1 and self.plotatts['offset']==0: strval += self.units # fallback option
      else: pass # warn('no unit specified') # should I have a warning here?
    return strval

  def __call__(self, x, pos=None):
    return self.format(x, units=False)
# }}}

class LatFormatter(AxisFormatter):
# {{{
  def format(self, val, units=False):
    if val > 0: return self.plotatts['formatstr'] % val + ' N'
    elif val < 0: return self.plotatts['formatstr'] % -val + ' S'
    else: return 'EQ' 
# }}}

class LonFormatter(AxisFormatter):
# {{{
  def format(self, val, units=False):
    v = (val + 180.) % 360. - 180.
    if v >= 0: return self.plotatts['formatstr'] % v + ' E'
    elif v < 0: return self.plotatts['formatstr'] % -v + ' W'
# }}}

class PresFormatter(AxisFormatter):
# {{{
  def format(self, val, units=False):
    su = ''
    if units: 
      if self.plotatts.has_key('plotunits'): su = ' ' + self.plotatts['plotunits']
      elif self.units != '': su = ' ' + self.units

    if val >= 10: return '%d ' % val + su
    #elif val >= 1: return '%.1g ' % val + su
    else: return '%.1g ' % val + su
# }}}

# Axis parent class
class Axis(Var):
# {{{

  """
    A one-dimensional object associated with each dimension of a data array.
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
  # Auxiliary arrays (provides some more context than just the regular value array)
  auxarrays = {}
  # Auxiliary attributes (attributes which should be preserved during merge/slice/etc.)
  auxatts = {}  

  _formatter = AxisFormatter
    
  def __init__(self, values, name=None, atts=None, plotatts=None, **kwargs):
# {{{ 
    import numpy as np

    # If a single integer given, expand to an integer range
    #TODO: get rid of this?  if you want to use integer indices, then make an appropriate 'Index' axis subclass?
    if isinstance(values,int):
      values = range(values)

    values = np.asarray(values)

    # Note: Call init before hasattr (or don't use hasattr at all in here)
    # (__getattr__ is overridden to call getaxis, which assumes axes are defined, otherwise __getattr__ is called to find an 'axes' property, ....)
    Var.__init__(self, [self], values=values, name=name, atts=atts, plotatts=plotatts)
 
    # Add auxilliary arrays after calling Var.__init__ - the weights
    # array, if present, will be added here, not by the logic in Var.__init___
    auxarrays = {}; auxatts = {}
    for key, val in kwargs.iteritems():
      if isinstance(val,(list,tuple,np.ndarray)):
        val = np.asarray(val)
        assert val.shape == self.values.shape, 'Auxilliary array %s has the wrong shape.' % key
        auxarrays[key] = val
      else:
        auxatts[key] = val
        
    # update auxiliary attribute (make copy to not change class defaults)        
    self.auxarrays = self.__class__.auxarrays.copy()
    self.auxarrays.update(auxarrays.copy())
    self.auxatts = self.__class__.auxatts.copy() 
    self.auxatts.update(auxatts.copy())    
    
    # name defaults
    if self.name == '': self.name = self.__class__.__name__.lower()
    if self.plotatts['plottitle'] is None: self.plotatts['plottitle'] = self.name

# }}}

  # Determine if one axis object is from a base class (or same class) of another axis
  @classmethod
  def isparentof(cls,other):
  # {{{
    return isinstance(other,cls)
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
    for fname in self.auxatts.iterkeys():
      if self.auxatts[fname] != other.auxatts[fname]: return False

    # Check any associated arrays
    if set(self.auxarrays.keys()) != set(other.auxarrays.keys()):
#      print 'false by mismatched set of auxarrays'
      return False

    # Check values of associated arrays
    for fname in self.auxarrays.iterkeys():
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
    '''check if there is a way to map from one to the other, but the values are
        not necessarily in 1:1 correspondence '''

    from pygeode.tools import map_to
    import numpy as np
    if not self.isparentof(other) and not other.isparentof(self): return None

    # special case: both axes are identical
    if self == other: return np.arange(len(self))

    # Use less conservative tolerance?
    rtol = max(self.auxatts.get('rtol', 1e-5), other.auxatts.get('rtol', 1e-5))

    return map_to(self.values, other.values, rtol)
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
    lat <Lat>      :  30 N to 10 N (3 values)
    >>> y = x.sorted()
    >>> print y
    lat <Lat>      :  10 N to 30 N (3 values)

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
    for key,val in self.auxatts.iteritems():
      aux[key] = val

    # Slice auxiliary arrays
    for key,val in self.auxarrays.iteritems():
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

    for k, v in kwargs.iteritems():
      if '_' in k: # Split off prefix if present
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

        if not hasattr(v,'__len__'): # Single value given
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
    from pygeode.var import Var
    return Var([self], values=self.auxarrays[name], name=name)
# }}}

  # How the axis is represented as a string
  def __repr__ (self): return '<' + self.__class__.__name__ + '>'

  # Convert an axis value to a string representation
  def _val2str (self, val):
    return '%g %s'%(val,self.units) if self.units != '' else '%g'%val

  def __str__ (self):
  # {{{
    if len(self) > 0:
      first = self._val2str(self.values[0])
      last = self._val2str(self.values[-1])
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
    aux = {}
    for k,v in self.auxatts.iteritems(): aux[k] = v
    for k,v in self.auxarrays.iteritems(): aux[k] = v

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

  # Plotting functions

  # Helper function to format values nicely (should be overridden in child classes)
  # By default, it uses self.plotatts['formatstr'] to format the value
  def formatvalue(self, val, units=True):
  # {{{
    ''' formatvalue(val)
        Returns an appropriately formatted string representation of val in the axis space. '''    
    fmt = self._formatter(self.plotatts, self.units)
    return fmt.format(val, units)
  # }}}

  # Returns a matplotlib axis Formatter object
  # By default, it uses self.plotatts['formatstr'] to format the value
  def formatter(self):
  # {{{
    ''' Returns a matplotlib axis Formatter object; by default a FuncFormatter which calls formatvalue(). '''
    return self._formatter(self.plotatts, self.units)
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
      auxkeys = auxkeys.intersection(a.auxatts.keys())
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
      auxkeys = auxkeys.intersection(a.auxarrays.keys())

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
  '''Slightly less useless than a raw Axis, we can at least use the name
      to uniquely identify them.'''
  def __init__ (self, values, name, **kwargs):
  # {{{
    Axis.__init__(self, values, **kwargs)
    self.name = name
    self.plotatts['plottitle'] = name
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

class XAxis (Axis): pass
class YAxis (Axis): pass

class Lon (XAxis): 
# {{{
  name = 'lon'
  plotatts = XAxis.plotatts.copy()
  plotatts['formatstr'] = '%d'
  plotatts['plottitle'] = ''
  _formatter = LonFormatter

  def locator(self):
  # {{{
    import pylab as pyl
    return pyl.MaxNLocator(nbins=9, steps=[1, 3, 6, 10])
  # }}}

  @staticmethod
  def _val2str (val):
    return '%g '%val + ('W' if val<0 else 'E')
# }}}

class Lat (YAxis):
# {{{
  name = 'lat'
  plotatts = YAxis.plotatts.copy() 
  plotatts['formatstr'] = '%d'
  plotatts['plottitle'] = ''
  _formatter = LatFormatter

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

  @staticmethod
  def _val2str (val):
  # {{{
    return '%g N'%val if val>0 else '%g S'%-val if val<0 else 'EQ'
  # }}}
# }}}

def gausslat (n, order=1, axis_dict={}):
# {{{
  '''Gaussian latitude axis'''
  from pygeode.quadrulepy import legendre_compute
  import numpy as np
  from math import pi
  from pygeode.tools import point as safe_point
  point = lambda x: safe_point(x).value
  if n in axis_dict: return axis_dict[n]
  x = np.empty([n],'d')
  w = np.empty([n],'d')
  legendre_compute(n,point(x),point(w))
  x = np.arcsin(x) / pi * 180

  x = x[::order]
  w = w[::order]

  axis = Lat (x, weights=w)
  axis_dict[n] = axis
  return axis
# }}}



# Spectral axes
# Note: XAxis/YAxis is used by cccma code to put these in the proper order (XAxis is fastest-increasing)
class SpectralM(YAxis): name = 'm'
class SpectralN(XAxis): name = 'n'

# Vertical axes
class ZAxis (Axis): 
# {{{
  name = 'lev'
  plotatts = Axis.plotatts.copy()
  plotatts['formatstr'] = '%3g'
# }}}

# Geometric height
#TODO: weights
#TODO: attributes
class Height(ZAxis):
# {{{  
  name = 'z' # default name
  units = 'm'
  plotatts = ZAxis.plotatts.copy()
  plotatts['formatstr'] = '%d' # just print integers
  # Formatting attributes for axis labels and ticks (see formatter method for application)
  plotatts['plottitle'] = 'Height' # name displayed in plots (axis label)
  plotatts['plotunits'] = 'km' # displayed units (after offset and scalefactor have been applied)
  plotatts['scalefactor'] = 1e-3 # conversion factor; assumed units are meters 
# }}}

# Model hybrid levels
#TODO: weights!
class Hybrid (ZAxis):
# {{{
  name = 'eta'  #TODO: rename this to 'hybrid'?  (keep 'eta' for now, for compatibility with existing code)
  plotatts = ZAxis.plotatts.copy()
  plotatts['formatstr'] = '%g'
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
  name = 'pres'
  units = 'hPa'
  plotatts = ZAxis.plotatts.copy() 
  plotatts['plottitle'] = 'Pressure'
  plotatts['plotunits'] = 'hPa'
  plotatts['plotscale'] = 'log'
  plotatts['plotorder'] = -1
  _formatter = PresFormatter

  def logPAxis(self, p0=1000., H=7.1):
    '''logPAxis(p0, H) - returns a pygeode axis with log pressure heights
          corresponding to this axis. By default p0 = 1000 hPa and H = 7.1 (km).'''
    import numpy as np
    z = ZAxis(H * np.log(p0 / self.values))
    z.plotatts['plottitle'] = 'Log-p Height (km)'
    return z

  def locator(self):
    import pylab as pyl, numpy as np
    ndecs = np.log10(np.max(self.values) / np.min(self.values))
    if ndecs < 1.2: return pyl.LogLocator(subs=[1., 2., 4., 7.])
    elif ndecs < 3.: return pyl.LogLocator(subs=[1., 3.])
    else: return pyl.LogLocator()

# }}}

class TAxis(Axis): pass
# NOTE: Time axis is in pygeode.timeaxis
# It's a fairly heavyweight class, worthy of its own module.
# Importing it here would create a circular reference that would bugger things up


class Freq(Axis):
  name = 'freq'
#  plotscale = 'log'
  def __init__ (self, values, inv_units=None, *args, **kwargs):
    if inv_units is not None:
      self.units = '/'+inv_units
      self.plotatts['plottitle'] = 'Frequency (per %s)'%inv_units
    kwargs = kwargs.copy()
    kwargs['inv_units'] = inv_units
    Axis.__init__ (self, values, *args, **kwargs)

# Indexing arrays (represent discrete items, such as EOF number, ensemble number, etc.)
class Index(Axis):
# {{{
  def __init__ (self, n, *args, **kwargs):
    values = n if hasattr(n,'__len__') else range(n)
    Axis.__init__(self, values, *args, **kwargs)
# }}}

# Coefficient number (when returning a set of coefficients, such as for a polynomial fit)
class Coef(Index): pass


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
standard_axes = [Axis, NamedAxis, XAxis, YAxis, ZAxis, TAxis, Lon, Lat, gausslat, Pres, Hybrid, Height, SpectralM, SpectralN, Freq, Index]
