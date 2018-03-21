
# Python functions for handling geophysical data

# These are dynamically generated objects, because we need to wrap the
# appropriate 'self' at init-time (here) to abuse the [] notation.
#TODO - something more pythonic?
class SL:
  def __init__(self, v): self.v = v
  def __getitem__ (self, slices):  return self.v._getitem_asvar(slices)
  def __len__ (self): return len(self.v)

#TODO: make Vars truly immutable (i.e. use 'slots' or something?).  Then:
#TODO: get rid of dynamic references to axes (__getattr__) - causes too many headaches!
#NOTE: keep the name mutable, since it's something that will commonly be changed
#     (subclasses should never rely on an immutable name attribute)..
#TODO: seperate the metadata/attributes from the data handler.
#  (I.e., have a more simple Var object with just a 'getview' or equivalent.
#    then, wrap this in another object with annotates it with a name,
#     attributes, axes, etc.)
#  This would make certain trivial operations (renaming, transposing, axis
#  replacement, etc.) more efficient, since the vars may go through several
#  permuations during their lifetime.  Currently, they accumulate layers of
#  wrappers.
class Var(object):
# {{{
  """
    The base class of all data objects in PyGeode.  This is not usually
    instantiated directly, but rather it is extended by a subclass to do some
    particular operation.  The only case where you would use this directly is
    if you already have some data loaded in memory (perhaps through some other
    interface), and you wish to wrap it as a PyGeode data to do further
    operations on it.

    See Also
    --------
    :doc:`axis`
  """

  # Default attributes

  #: A description of the variable (may not be set).  Usually determined at the
  #: data source (e.g. input file), and may be used to identify the variable
  #: when saving to an output file.
  name = '' # default name (blank)

  #: A string representation of the units of the variable.
  units = '' # default units (none)

  #: Formatting code to use for printing values.
  formatstr = '%g' 

  #: Dictionary of metadata associated with the variable.
  atts = {} # shared dictionary - replace this in init!

  #: Dictionary of attributes for plotting; see plotting documentation.
  plotatts = {'plotscale': 'linear',  # default scale for plotting
              'plotorder': 1}  # By default, plot with axis values increasing away from origin

  #: The axes of the variable. A ``tuple`` of :class:`Axis` instances.
  axes = None

  #: The number of axes of this variable.
  naxes = 0

  #: The dimensions of this variable, as a ``tuple``. Similar to :attr:`numpy.ndarray.shape`.
  shape = None

  #: The total number of data points represented by this variable. 
  size = 0

  #: The numerical type of the data as a :class:`numpy.dtype`. See also :meth:`Var.__init__`.
  dtype = None

  #: A helper to select subsets of this variable using slice notation. See :meth:`Var._getitem_asvar`.
  slice = None

  # This method should be called by all subclasses
  def __init__ (self, axes, dtype=None, name=None, values=None, atts=None, plotatts=None):
  # {{{
    """
    Create a new Var object with the given axes and values.

    Parameters
    ----------
    axes : list/tuple
        The :class:`Axis` objects to associate with each of the data dimensions
    dtype : string / Python type / numpy.dtype (optional)
        The numerical type of the data (can be automatically determined from
        the array)
    name : string (optional)
        What to call the variable (i.e. for plot titles & when saving to file)
    values : numpy.ndarray
        The data to be wrapped. 
    atts : dict (optional)
        Any additional metadata to associate with the variable. The dictionary
        keys should be strings.
    plotatts : dict (optional)
        Parameters that control plotting behaviour; default values are available. 
        The dictionary keys should be strings.

    Returns
    -------
    out : Var
      The array, wrapped as a Var object.

    Notes
    -----
    The :class:`Var` class can be instantiated directly (see `constructing-vars`),
    in which case providing an array for the values argument is necessary. Sub-classes
    of `Var` which define their values based on some operation need not provide any
    data; however all subclasses of :class:`Var` need to call this __init__ method within
    their own __init__, to properly initialize all attributes.
    """


    import numpy as np
    from pygeode.axis import Axis

    # Convert the list of axes to a tuple
    # Since it is normally immutable - modifying the axes implies
    # changing the variable itself
    # Do this before calling 'hasattr', or you're in for a world of pain
    assert all(isinstance(a,Axis) for a in axes)
    self.axes = tuple(axes)
    self.naxes = len(axes)
    
    # If we're given a Var as the input data, then we need to grab the data.
    if isinstance(values,Var): values = values.get()

    # Values stored in memory?
    if values is not None:
      self.values = np.asarray(values,dtype=dtype)
      # Make values read-only (or at least difficult to change accidentally)
      self.values.flags.writeable = False

    # Get the shape of the variable
    # Have to do this after setting self.values, otherwise this crashes
    # when initializing Axis objects (which call this init)
    self.shape = tuple(len(a) for a in axes)

    # Check the shape of the value array
    if values is not None:
      assert self.values.ndim == self.naxes, "ndim=%d, naxes=%d?"%(self.values.ndim, self.naxes)
      assert self.values.shape == self.shape, "array shape does not match the given axes"

    # Determine the type, if we're supplied the values...
    if dtype is None:
      if values is not None:
        dtype = self.values.dtype
      else:
        raise TypeError("Can't determine dtype")

    # Convert dtype shortcuts like 'd' to a standard name
    dtype = np.dtype(dtype)
    self.dtype = dtype

    # handle meta data (create unique copy of each dictionary for each instance)
    if name is not None: self.name = name
    self.atts = self.__class__.atts.copy()
    if atts is not None: 
      self.atts.update(atts)
    self.plotatts = self.__class__.plotatts.copy()
    if plotatts is not None: 
      self.plotatts.update(plotatts)
    # Note: the default empty dict {} used to be set as a static thing right
    # after the 'class Var' line, but this meant that all vars which weren't
    # assigned explicit attributes were sharing the same dict, so any post-init
    # modifications would be applied to *all* such vars!
#    if self.naxes == 0:
#      print 'STOP!'
#      print 'Hammer time'
#      raise Exception("You can't touch this")

#    # Shortcuts to the axes, referenced by name
#    axis_names = [a.name for a in axes]
#    for a in axes:
#      name = a.name
#      if axis_names.count(name) == 1:  # Unique occurrences only
#        setattr(self,name,a)
# note: this is currently done dynamically, to allow some fudging of the axes

    # Get the size of the var
#    self.size = np.prod(self.shape)
    self.size = reduce(lambda x,y: x*y, self.shape, 1)

    # Slicing notation
    self.slice = SL(self)

    # If this is a Var (and not a subclass), then it is safe to lock
    # the attributes now, and prevent furthur changes.
    #if type(self) == Var: self._finalize()

  # }}}

  # Subset by integer indices - wrapped as Var object
  def _getitem_asvar (self, slices):
# {{{
    '''
    Slice-based data subsetting.

    Parameters
    ----------
    slices : list of slices

    Returns
    -------
    subset_var : Var
      A new Var, restricted to the specified domain.

    Notes
    -----
    A helper function so that standard python slicing notation
    can be used to subset a Var without loading the underlying
    data. A new Var object is returned.

    See Also
    --------
    Var.slice, Var.__call__, Var.__getitem__

    Examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print t1.Temp
    <Var 'Temp'>:
      Units: K  Shape:  (lat,lon)  (31,60)
      Axes:
        lat <Lat>      :  90 S to 90 N (31 values)
        lon <Lon>      :  0 E to 354 E (60 values)
      Attributes:
        {}
      Type:  Add_Var (dtype="float64")
    >>> print t1.Temp.slice[10:-10, ::10]
    <Var 'Temp'>:
      Units: K  Shape:  (lat,lon)  (11,6)
      Axes:
        lat <Lat>      :  30 S to 30 N (11 values)
        lon <Lon>      :  0 E to 300 E (6 values)
      Attributes:
        {}
      Type:  SlicedVar (dtype="float64")
    >>> print t1.Temp.slice[17, :]
    <Var 'Temp'>:
      Units: K  Shape:  (lat,lon)  (1,60)
      Axes:
        lat <Lat>      :  12 N
        lon <Lon>      :  0 E to 354 E (60 values)
      Attributes:
        {}
      Type:  SlicedVar (dtype="float64")
    '''
    from pygeode.varoperations import SlicedVar
    newvar = SlicedVar(self, slices)

    # Degenerate case: no slicing done?
    if all(a1 is a2 for a1,a2 in zip(newvar.axes, self.axes)): return self

#    # Was the source var preloaded?
#    # If so, preload this slice as well
#    if hasattr(self,'values'): newvar = newvar.load(pbar=None)

    return newvar
# }}}

  # Subset by integer indices
  def __getitem__ (self, slices):
# {{{
    """
    Gets a raw numpy array containing a subset of values of the variable.

    Parameters
    ----------
    slices : list of slices

    Returns
    -------
    out : numpy.ndarray
      The requested values, as a numpy array.

    See Also
    --------
    Var.get, Var.slice, Var.__call__

    Examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print t1.Temp[:].shape
    (31, 60)
    >>> print t1.Temp[20:-6, ::12]
    [[ 285.64721554  287.07380031  286.52889342  284.76553766  284.22063076]
     [ 281.09169696  282.80359869  282.14971042  280.03368351  279.37979523]
     [ 276.73945224  278.73667093  277.97380127  275.50510321  274.74223356]
     [ 272.82122084  275.10375648  274.23190545  271.41053624  270.5386852 ]
     [ 269.47711035  272.04496294  271.06413053  267.89009017  266.90925775]]
    """
    # Get the raw numpy array (with degenerate axes intact)
    array = self._getitem_asvar(slices).get()
    # If any single integer indices were passed, then reduce out those
    # dimensions.  This is consistent with what would happen with numpy slicing.
    if isinstance(slices,tuple):
      extra_slicing = tuple(0 if isinstance(sl,int) else Ellipsis if sl is Ellipsis else slice(None) for sl in slices)
      array = array[extra_slicing]
    elif isinstance(slices,int):
      array = array[0]
    return array
# }}}

  # Select a subset by keyword arguments (i.e., lat = (-45.0, 45.0))
  # Keys should be the name of the axis shortcut in the var (i.e., lat, lon, time).
  def __call__ (self, ignore_mismatch = False, **kwargs):
  # {{{
    """
    Keyword-based data subsetting.

    Parameters
    ----------
    ignore_mismatch : boolean (optional)
      If ``True``, any keywords that don't match an axis are ignored.
      Default is ``False``

    **kwargs : one or more keyword parameters
      The keys are the Axis names (or Axis class names), and the values are
      either a `tuple` of the desired (lower,upper) range, or a single Axis
      value.  E.g., ``lat = (-45,45)`` or ``lat = 10.5``

    Returns
    -------
    subset_var : Var
      A new Var, restricted  to the specified domain.

    Notes
    -----
    There are a couple of special prefixes which can be prepended to each
    keyword to alter the subsetting behaviour. They can be used together.

      * **i_** indicates that the values are *indices* into the axis, and not
        the axis values themselves.  Indices start at 0.
        E.g. ``myvar(i_time = 0)`` selects the first time step of the variable,
        and ``myvar(i_lon=(10,20))`` selects the 11th through 21st longitudes.

      * **l_** indicates that you are providing an explicit list of
        coordinates, instead of a range.
        E.g. ``myvar(l_lon = (105.,106.,107.,108))``

      * **n_** returns the complement of the set you request; that is,
        everything except the specified selection.
        E.g. ``myvar(n_lat = (60, 90))`` returns all latitudes except those between 60 and 90N.

      * **m_** triggers an arithmetic mean over the specified range.
        E.g., ``myvar(m_lon = (10, 80))`` is a shortcut for doing
        ``myvar(lon = (10,80)).mean('lon')``.

      * **s_** triggers a call to squeeze on the specified axis, so
        that if only one value is selected the degenerate axis is removed.
        E.g., ``myvar(s_lon = 5)`` is a shortcut for doing
        ``myvar(lon = 5).squeeze()`` or ``myvar.squeeze(lon=5)``.


    Examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print t1.vars
    [<Var 'Temp'>]
    >>> T = t1.Temp
    >>> print T
    <Var 'Temp'>:
      Shape:  (lat,lon)  (32,64)
      Axes:
        lat <Lat>      :  85 S to 85 N (32 values)
        lon <Lon>      :  0 E to 354 E (64 values)
      Attributes:
        {'units': 'K'}
      Type:  Var (dtype="float64")
    >>> print T(lat=30,lon=(100,200))
    <Var 'Temp'>:
      Shape:  (lat,lon)  (1,18)
      Axes:
        lat <Lat>      :  30 N
        lon <Lon>      :  101 E to 196 E (18 values)
      Attributes:
        {'units': 'K'}
      Type:  SlicedVar (dtype="float64")
    """
    # Check for mean axes
    # (kind of a hack... prepend an 'm_' to the keyword)
    # i.e., var(m_lon=(30,40)) is equivalent to var(lon=(30,40)).mean('lon')

    means = []
    squeezes = []

    newargs = {}
    for k, v in kwargs.iteritems():
      # Detect special prefixes.
      if '_' in k and not self.hasaxis(k):
        prefix, ax = k.split('_', 1)
      else:
        prefix, ax = '', k

      if 'm' in prefix:
        if not self.hasaxis(ax) and ignore_mismatch: continue
        assert self.hasaxis(ax), "'%s' is not a valid axis for var '%s'"%(ax,self.name)
        means.append(ax)
        prefix = prefix.replace('m', '')

      if 's' in prefix:
        if not self.hasaxis(ax) and ignore_mismatch: continue
        assert self.hasaxis(ax), "'%s' is not a valid axis for var '%s'"%(ax,self.name)
        if ax not in means: squeezes.append(ax)
        prefix = prefix.replace('s', '')

      if len(prefix) > 0:
        k = '_'.join([prefix, ax])
      else:
        k = ax

      newargs[k] = v

    # Get the slices, apply to the variable
    slices = [a.get_slice(newargs, ignore_mismatch=True) for a in self.axes]
    assert len(newargs) == 0 or ignore_mismatch, "Unmatched slices remain: %s" % str(newargs)

    ret = self._getitem_asvar(slices)

    # Any means or squeezes requested?
    if len(means) > 0: ret = ret.mean(*means)
    if len(squeezes) > 0: ret = ret.squeeze(*squeezes)

    return ret
  # }}}

  # Avoid accidentally iterating over vars
  def __iter__ (self):
# {{{
    raise Exception ("Var instances cannot be iterated over")
# }}}

  # Include axes names
  def __dir__(self):
# {{{
    l = self.__dict__.keys() + dir(self.__class__)
    return l + [a.name for a in self.axes]
# }}}

  # Shortcuts to axes
  # (handled dynamically, in case some fudging is done to the var)
  def __getattr__(self, name):
# {{{
    # Disregard metaclass stuff
    if name.startswith('__'): raise AttributeError

#    print 'var getattr ??', name


    # Some things that we *know* should not be axis shortcuts
    if name in ('axes', 'values'):
      raise AttributeError(name)
    try: return self.getaxis(name)
    except KeyError: raise AttributeError ("'%s' not found in %s"%(name,repr(self)))

    raise AttributeError (name)
# }}}


  # Modifies __setattr__ and __delattr__ to avoid further modifications to the Var
  # (should be called by any Var subclasses at the end of their __init__)
  def _finalize (self):
# {{{
    if self.__setattr__ == self._dont_setattr:
      from warnings import warn
      warn ("already finalized the var", stacklevel=2)
      return
    self.__delattr__ = self._dont_delattr
    self.__setattr__ = self._dont_setattr
# }}}

  # Send all attribute requests through here, to enforce immutability
  def _dont_setattr(self,name,value):
# {{{
    # Can only modify the var name
    if name != 'name':
#      raise TypeError ("can't modify an immutable Var")
      from warnings import warn
      warn ("shouldn't be modifying an immutable Var", stacklevel=2)
    object.__setattr__(self,name,value)
# }}}

  def _dont_delattr(self,name):
# {{{
#    raise TypeError ("can't modify an immutable Var")
    from warnings import warn
    warn ("shouldn't be modifying an immutable Var", stacklevel=2)
    object.__delattr__(self,name)
# }}}

  # Get an axis based on its class
  # ie, somevar.getaxis(ZAxis)
  def getaxis (self, iaxis):
  # {{{
    """
      Grabs a reference to a particular :class:`Axis` associated with this variable.

      Parameters
      ----------
      iaxis : string, :class:`Axis` class, or int
        The search criteria for finding the class.

      Returns
      -------
      axis : :class:`Axis` object that matches.  Raises ``KeyError`` if there is no match.

      See Also
      --------
      whichaxis, hasaxis, Axis
    """
    return self.axes[self.whichaxis(iaxis)]
  # }}}

  # Get the index of an axis given its class (or a string)
  # (similar to above, but return the index, not the Axis itself)
  def whichaxis (self, iaxis):
  # {{{
    """
    Locates a particular :class:`Axis` associated with this variable.

    Parameters
    ----------
    iaxis : string, :class:`Axis` class, or int
      The search criteria for finding the class.

    Returns
    -------
    i : The index of the matching Axis.  Raises ``KeyError`` if there is no match.

    Examples
    --------
    >>> from pygeode.tutorial import t1
    >>> T = t1.Temp
    >>> print T.axes
    (<Lat>, <Lon>)
    >>> print T.whichaxis('lat')
    0
    >>> print T.whichaxis('lon')
    1
    >>> print T.whichaxis('time')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/usr/local/lib/python2.6/dist-packages/pygeode/var.py", line 352, in whichaxis
        raise KeyError, "axis %s not found in %s"%(repr(iaxis),self.axes)
    KeyError: "axis 'time' not found in (<Lat>, <Lon>)"

    See Also
    --------
    getaxis, hasaxis, Axis
    """
    from pygeode.tools import whichaxis
    return whichaxis(self.axes, iaxis)
  # }}}

  # Indicate if an axis is found
  def hasaxis (self, iaxis):
  # {{{
    """
      Determines if a particular :class:`Axis` is associated with this variable.

      Parameters
      ----------
      iaxis : string, :class:`Axis` class, or int
        The search criteria for finding the class.

      Returns
      -------
      found : boolean.  ``True`` if found, otherwise ``False``

      See Also
      --------
      getaxis, whichaxis, Axis
    """
    try: i = self.whichaxis(iaxis)
    except KeyError: return False
    return True
  # }}}

  def formatvalue(self, value, fmt=None, units=True, unitstr=None):
  # {{{
    '''
    Returns formatted string representation of ``value``.
      
    Parameters
    ----------
    value : float or int
      Value to format.
    fmt : string (optional)
      Format specification. If the default ``None`` is specified, 
      ``self.formatstr`` is used.
    units : boolean (optional)
      If ``True``, will include the units in the string returned.
      Default is ``True``.
    unitstr : string (optional)
      String to use for the units; default is ``self.units``.

    Returns
    -------
    If units is True, fmt % value + ' ' + unitstr. Otherwise fmt % value.

    Notes
    -----
    This is overridden in a number of Axis classes for more sophisticated formatting.
    '''

    if fmt is None: fmt = self.formatstr
    strval = fmt % value

    if units: 
      if unitstr is None: unitstr = self.units
      strval += ' ' + unitstr

    return strval
  # }}}

  # Pretty printing
  def __str__ (self):
  # {{{
    from textwrap import TextWrapper
#    axes_list = '(' + ', '.join(a.name if a.name != '' else repr(a) for a in self.axes) + ')'
#    axes_details = ''.join(['  '+str(a) for a in self.axes])
#    s = repr(self) + ':\n\n  axes: ' + axes_list + '\n  shape: ' + str(self.shape) + '\n\n' + axes_details
    s = repr(self) + ':\n'
    if self.units != '':
      s += '  Units: ' + self.units
    s += '  Shape:'
    s += '  (' + ','.join(a.name for a in self.axes) + ')'
    s += '  (' + ','.join(str(len(a)) for a in self.axes) + ')\n'
    s += '  Axes:\n'
    for a in self.axes:
      s += '    '+str(a) + '\n'
    
    w = TextWrapper(width=80)
    w.initial_indent = w.subsequent_indent = '    '
    w.break_on_hyphens = False
    s += '  Attributes:\n' + w.fill(str(self.atts)) + '\n'
    s += '  Type:  %s (dtype="%s")' % (self.__class__.__name__, self.dtype.name)
    return s
  # }}}

  def __repr__ (self):
  # {{{
    if self.name != '':
#      return '<' + self.__class__.__name__ + " '" + self.name + "'>"
      return "<Var '" + self.name + "'>"
    return '<Var>'
  # }}}

  def get (self, pbar=None, **kwargs):
  # {{{
    """
    Gets a raw numpy array containing the values of the variable.

    Parameters
    ----------
    pbar : boolean (optional)
      If ``True``, will display a progress bar while the data is being
      retrieved.  This requires the *python-progressbar* package (not included
      with PyGeode).

    **kwargs : keyword arguments (optional)
      One or more keyword arguments may be included to subset the variable
      before grabbing the data.  See :func:`Var.__call__` for a similar
      method which uses this keyword subsetting.

    Returns
    -------
    out : numpy.ndarray
      The requested values, as a numpy array.

    Notes
    -----

    Once you grab the data as a numpy array, you can no longer use the PyGeode
    functions to do further work on it directly.  You can, however, use
    :func:`Var.__init__` to re-wrap your numpy array as a PyGeode Var.  This
    may be useful if you want to do some very complicated operations on the
    data using the numpy interface as an intermediate step.

    PyGeode variables can be huge!  They can be larger than the available RAM
    in your computer, or even larger than your hard disk.  Numpy arrays, on
    the other hand, need to fit in memory, so make sure you are only getting
    a reasonable piece of data at a time.

    Examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print t1.Temp
    <Var 'Temp'>:
      Shape:  (lat,lon)  (32,64)
      Axes:
        lat <Lat>      :  85 S to 85 N (32 values)
        lon <Lon>      :  0 E to 354 E (64 values)
      Attributes:
        {'units': 'K'}
      Type:  Var (dtype="float64")
    >>> x = t1.Temp.get()
    >>> print x
    [[ 261.05848727  259.81373805  258.6761858  ...,  264.37317879
       263.44078874  262.30323649]
     [ 261.66049058  260.49545075  259.43074336 ...,  264.76292084
       263.89023779  262.82553041]
     [ 262.53448988  261.44963014  260.45819779 ...,  265.42340543
       264.61078196  263.61934962]
     ..., 
     [ 262.53448988  263.61934962  264.61078196 ...,  259.64557433
       260.45819779  261.44963014]
     [ 261.66049058  262.82553041  263.89023779 ...,  258.55806031
       259.43074336  260.49545075]
     [ 261.05848727  262.30323649  263.44078874 ...,  257.74379575  258.6761858
       259.81373805]]
    """
    from pygeode.view import View
    import numpy as np
    var = self.__call__(**kwargs)
    data = View(var.axes).get(var,pbar=pbar)
    if isinstance(data,np.ndarray):
      data = np.array(data,copy=True)
    return data
  # }}}

  def getweights (self, iaxes = None):
  # {{{
    ''' Returns weights associated with the axes of this variable.

    Parameters
    ----------
    iaxes : list of axis identifiers (string, :class:`Axis`, or int) (optional)
      Axes over which to compute the weights

    Returns
    -------
    weights : :class:`Var`
      defined on the subgrid of this variable on which weights are defined.

    See Also
    --------
    Axis.__init__
    '''
    # Add weights object if any of the axes are weighted
    # Take outer product of weights if present on more than one axis
    if iaxes is None or len(iaxes) == 0:
      axes = self.axes
    else:
      axes = [self.getaxis(i) for i in iaxes]
    wt = tuple([a.auxasvar('weights') for a in axes if hasattr(a, 'weights')])
    if len(wt) == 0:
#      return None
      return Var(axes=[], values=1.0)  # Return degenerate variable?
    else:
      from pygeode.ufunc import vprod
      return vprod(*wt) # Variable-wise product of weights
  # }}}

  # Preload the values of a variable
  # If the values are already loaded, then return self
  def load (self, pbar=True):
  # {{{
    ''' Returns a version of this variable with all data loaded into memory.

    Parameters
    ----------
    pbar : boolean
      If True, display a progress bar while loading data.
    '''
    from pygeode.progress import PBar
    if hasattr(self, 'values'): return self
    if pbar is True:
      pbar = PBar(message="Loading %s:"%repr(self))
    var = Var(self.axes, values=self.get(pbar=pbar))
    copy_meta (self, var)
    return var
  # }}}

  # Plotting helper functions
  def formatter(self):
  # {{{
    '''Returns a matplotlib formatter (pygeode.AxisFormatter) for use in plotting. '''
    from pygeode.axis import AxisFormatter
    return AxisFormatter(self)
  # }}}

  def locator(self):
  # {{{
    '''Returns a matplotlib locator object for use in plotting.'''
    import pylab as pyl
    scl = self.plotatts.get('plotscale', 'linear')
    if scl == 'log': return pyl.LogLocator()
    else: return pyl.AutoLocator()
  # }}}
# }}}

# a function to copy metadata from one variable to another
def copy_meta (invar, outvar, plotatts=True):
# {{{
  outvar.name = invar.name
  outvar.units = invar.units
  outvar.atts = invar.atts.copy()
  if plotatts: outvar.plotatts = invar.plotatts.copy()
# }}}

# a function to try and combine metadata from multiple inputs into an output
# (similar to copy_meta, but takes multiple input variables)
def combine_meta (invars, outvar):
# {{{
  from pygeode.tools import common_dict
  from pygeode import Axis
  # Intrinsic attributes
  for att in 'name', 'units':
    s = list(set([getattr(v,att) for v in invars]))
    if len(s) == 1: setattr(outvar,att,s[0])
  # Get common metadata from the input segments
  # *Set* these attributes, don't 'update' the dictionaries!
  # This method may be called from the __init__ of a Var subclass, before the
  # dictionaries are properly created - the existing 'atts' and 'plotatts' may
  # be a shared dictionary! Skip Axis objects for the plotatts dict (see issue 53)
  outvar.atts = common_dict([v.atts for v in invars])
  outvar.plotatts = common_dict([v.plotatts for v in invars if not isinstance(v, Axis)])
# }}}


##################################################
# Hook various useful methods into the Var class.
# We can't define these in the class definition, because it would create
# a circular reference.
##################################################

# Numeric operations
from numpy import ndarray as nd
from pygeode.ufunc import wrap_unary, wrap_binary
Var.__add__  = wrap_binary(nd.__add__,  symbol='+')
Var.__radd__ = wrap_binary(nd.__radd__, symbol='+')
Var.__sub__  = wrap_binary(nd.__sub__,  symbol='-')
Var.__rsub__ = wrap_binary(nd.__rsub__, symbol='-')
Var.__mul__  = wrap_binary(nd.__mul__,  symbol='*')
Var.__rmul__ = wrap_binary(nd.__rmul__, symbol='*')
Var.__div__  = wrap_binary(nd.__div__,  symbol='/')
Var.__rdiv__ = wrap_binary(nd.__rdiv__, symbol='/')
Var.__pow__  = wrap_binary(nd.__pow__,  symbol='**')
Var.__rpow__ = wrap_binary(nd.__rpow__, symbol='**')
Var.__mod__  = wrap_binary(nd.__mod__,  symbol='%')
Var.__rmod__ = wrap_binary(nd.__rmod__, symbol='%')

Var.__lt__ = wrap_binary(nd.__lt__, symbol='<')
Var.__le__ = wrap_binary(nd.__le__, symbol='<=')
Var.__gt__ = wrap_binary(nd.__gt__, symbol='>')
Var.__ge__ = wrap_binary(nd.__ge__, symbol='>=')
Var.__eq__ = wrap_binary(nd.__eq__, symbol='==')
Var.__ne__ = wrap_binary(nd.__ne__, symbol='!=')

Var.__abs__ = wrap_unary(nd.__abs__, "Absolute value", symbol=('|','|'))
Var.__neg__ = wrap_unary(nd.__neg__, "Flips the sign of the values", symbol = ('-',''))
Var.__pos__ = wrap_unary(nd.__pos__, "Does nothing", symbol = ('+',''))

def __trunc__ (x):
  import numpy as np
  return np.asarray(x, dtype=int)
Var.__trunc__ = wrap_unary(__trunc__, "Truncate to integer value", symbol=('trunc(',')'))
del __trunc__

del nd, wrap_unary, wrap_binary


global_hooks = []
class_hooks = []

# Universal functions (pointwise arithmetic, etc.)
#from pygeode.ufunc import class_flist, generic_flist
#global_hooks += generic_flist
#class_hooks += class_flist
#del class_flist, generic_flist
from pygeode.ufunc import unary_flist
class_hooks.extend(unary_flist)
del unary_flist


# --- Array reduction  ---
# (summation, average, min, max, etc)
from pygeode.reduce import class_flist
class_hooks += class_flist
del class_flist

# Other operations deemed important enough to hook into Var
# Static methods
from pygeode.concat import concat
global_hooks += [concat]
del concat
# Class methods
from pygeode.smooth import smooth
from pygeode.deriv import deriv
from pygeode.diff import diff
from pygeode.intgr import integrate
from pygeode.composite import composite, flatten
from pygeode.fft_smooth import fft_smooth
from pygeode.timeutils import lag
try:
  from pygeode.interp import interpolate
except ImportError:
  from warnings import warn
  warn ("Can't import the GSL library.  Interpolation is disabled.")
  interpolate = None
from pygeode.varoperations import squeeze, extend, transpose, sorted, replace_axes, rename, rename_axes, fill, unfill, as_type
class_hooks += filter(None,[smooth, deriv, diff, integrate, composite, flatten, fft_smooth, lag, interpolate, squeeze, extend, transpose, sorted, replace_axes, rename, rename_axes, fill, unfill, as_type])
del smooth, deriv, diff, integrate, composite, flatten, fft_smooth, lag, interpolate, squeeze, extend, transpose, sorted, replace_axes, rename, rename_axes, fill, unfill, as_type

# Climatology operators
from pygeode import climat
for n in climat.__all__:
  c = getattr(climat,n)
  def do_wrapper (c=c):
    def f (*args, **kwargs):
      return c(*args, **kwargs)
    f.__name__ = n
    return f
  class_hooks.append(do_wrapper())
del climat, n, c


# Apply the global hooks
for f in global_hooks:
  globals()[f.__name__] = f

# Apply the class hooks
def wrap_static(f):
  def g(self, *args, **kwargs):
    return f(self, *args, **kwargs)
  from functools import update_wrapper
  g = update_wrapper(g, f)
  g.__doc__ = 'Test'
  return g

for f in class_hooks:
  setattr(Var,f.__name__,f)

del f


# Apply unary functions from ufunc
#TODO: remove these?  (kind of an awkward syntax, i.e. t.lat.cosd().sqrt() )
from pygeode.ufunc import unary_flist
for f in unary_flist:
  # We need to retain a copy of *this* f in a local scope,
  # so encapsulate this part of the process in another (trivial) function
  def newf_creator(f):
    _f = f  # store a copy of the source function in this local scope
    def newf(self):
      return _f(self)
    newf.__name__ = f.__name__
    newf.__doc__ = f.__doc__ + ' Called on the Var object.'
    return newf

  setattr(Var,f.__name__,newf_creator(f))
  del newf_creator
del f
