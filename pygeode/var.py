
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
    the base class of all data objects in pygeode.  this is not usually
    instantiated directly, but rather it is extended by a subclass to do some
    particular operation.  the only case where you would use this directly is
    if you already have some data loaded in memory (perhaps through some other
    interface), and you wish to wrap it as a pygeode data to do further
    operations on it.

    see also
    --------
    :doc:`axis`
  """

  # default attributes

  #: a description of the variable (may not be set).  usually determined at the
  #: data source (e.g. input file), and may be used to identify the variable
  #: when saving to an output file.
  name = '' # default name (blank)

  #: a string representation of the units of the variable.
  units = '' # default units (none)

  #: formatting code to use for printing values.
  formatstr = '%g'

  #: dictionary of metadata associated with the variable.
  atts = {} # shared dictionary - replace this in init!

  #: dictionary of attributes for plotting; see plotting documentation.
  plotatts = {'plotscale': 'linear',  # default scale for plotting
              'plotorder': 1}  # by default, plot with axis values increasing away from origin

  #: the axes of the variable. a ``tuple`` of :class:`axis` instances.
  axes = none

  #: the number of axes of this variable.
  naxes = 0

  #: the dimensions of this variable, as a ``tuple``. similar to :attr:`numpy.ndarray.shape`.
  shape = none

  #: the total number of data points represented by this variable.
  size = 0

  #: the numerical type of the data as a :class:`numpy.dtype`. see also :meth:`var.__init__`.
  dtype = none

  #: a helper to select subsets of this variable using slice notation. see :meth:`var._getitem_asvar`.
  slice = none

  # this method should be called by all subclasses
  def __init__ (self, axes, dtype=none, name=none, values=none, atts=none, plotatts=none):
  # {{{
    """
    create a new var object with the given axes and values.

    parameters
    ----------
    axes : list/tuple
        the :class:`axis` objects to associate with each of the data dimensions
    dtype : string / python type / numpy.dtype (optional)
        the numerical type of the data (can be automatically determined from
        the array)
    name : string (optional)
        what to call the variable (i.e. for plot titles & when saving to file)
    values : numpy.ndarray
        the data to be wrapped.
    atts : dict (optional)
        any additional metadata to associate with the variable. the dictionary
        keys should be strings.
    plotatts : dict (optional)
        parameters that control plotting behaviour; default values are available.
        the dictionary keys should be strings.

    returns
    -------
    out : var
      the array, wrapped as a var object.

    notes
    -----
    the :class:`var` class can be instantiated directly (see `constructing-vars`),
    in which case providing an array for the values argument is necessary. sub-classes
    of `var` which define their values based on some operation need not provide any
    data; however all subclasses of :class:`var` need to call this __init__ method within
    their own __init__, to properly initialize all attributes.
    """


    import numpy as np
    from pygeode.axis import axis

    # convert the list of axes to a tuple
    # since it is normally immutable - modifying the axes implies
    # changing the variable itself
    # do this before calling 'hasattr', or you're in for a world of pain
    assert all(isinstance(a,axis) for a in axes)
    self.axes = tuple(axes)
    self.naxes = len(axes)

    # if we're given a var as the input data, then we need to grab the data.
    if isinstance(values,var): values = values.get()

    # values stored in memory?
    if values is not none:
      self.values = np.asarray(values,dtype=dtype)
      # make values read-only (or at least difficult to change accidentally)
      self.values.flags.writeable = false

    # get the shape of the variable
    # have to do this after setting self.values, otherwise this crashes
    # when initializing axis objects (which call this init)
    self.shape = tuple(len(a) for a in axes)

    # check the shape of the value array
    if values is not none:
      assert self.values.ndim == self.naxes, "ndim=%d, naxes=%d?"%(self.values.ndim, self.naxes)
      assert self.values.shape == self.shape, "array shape does not match the given axes"

    # determine the type, if we're supplied the values...
    if dtype is none:
      if values is not none:
        dtype = self.values.dtype
      else:
        raise typeerror("can't determine dtype")

    # convert dtype shortcuts like 'd' to a standard name
    dtype = np.dtype(dtype)
    self.dtype = dtype

    # handle meta data (create unique copy of each dictionary for each instance)
    if name is not none: self.name = name
    self.atts = self.__class__.atts.copy()
    if atts is not none:
      self.atts.update(atts)
    self.plotatts = self.__class__.plotatts.copy()
    if plotatts is not none:
      self.plotatts.update(plotatts)
    # note: the default empty dict {} used to be set as a static thing right
    # after the 'class var' line, but this meant that all vars which weren't
    # assigned explicit attributes were sharing the same dict, so any post-init
    # modifications would be applied to *all* such vars!
#    if self.naxes == 0:
#      print 'stop!'
#      print 'hammer time'
#      raise exception("you can't touch this")

#    # shortcuts to the axes, referenced by name
#    axis_names = [a.name for a in axes]
#    for a in axes:
#      name = a.name
#      if axis_names.count(name) == 1:  # unique occurrences only
#        setattr(self,name,a)
# note: this is currently done dynamically, to allow some fudging of the axes

    # get the size of the var
#    self.size = np.prod(self.shape)
    self.size = reduce(lambda x,y: x*y, self.shape, 1)

    # slicing notation
    self.slice = sl(self)

    # if this is a var (and not a subclass), then it is safe to lock
    # the attributes now, and prevent furthur changes.
    #if type(self) == var: self._finalize()

  # }}}

  # subset by integer indices - wrapped as var object
  def _getitem_asvar (self, slices):
# {{{
    '''
    slice-based data subsetting.

    parameters
    ----------
    slices : list of slices

    returns
    -------
    subset_var : var
      a new var, restricted to the specified domain.

    notes
    -----
    a helper function so that standard python slicing notation
    can be used to subset a var without loading the underlying
    data. a new var object is returned.

    see also
    --------
    var.slice, var.__call__, var.__getitem__

    examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print(t1.temp)
    <var 'temp'>:
      units: k  shape:  (lat,lon)  (31,60)
      axes:
        lat <lat>      :  90 s to 90 n (31 values)
        lon <lon>      :  0 e to 354 e (60 values)
      attributes:
        {}
      type:  add_var (dtype="float64")
    >>> print(t1.temp.slice[10:-10, ::10])
    <var 'temp'>:
      units: k  shape:  (lat,lon)  (11,6)
      axes:
        lat <lat>      :  30 s to 30 n (11 values)
        lon <lon>      :  0 e to 300 e (6 values)
      attributes:
        {}
      type:  slicedvar (dtype="float64")
    >>> print(t1.temp.slice[17, :])
    <var 'temp'>:
      units: k  shape:  (lat,lon)  (1,60)
      axes:
        lat <lat>      :  12 n
        lon <lon>      :  0 e to 354 e (60 values)
      attributes:
        {}
      type:  slicedvar (dtype="float64")
    '''
    from pygeode.varoperations import slicedvar
    newvar = slicedvar(self, slices)

    # degenerate case: no slicing done?
    if all(a1 is a2 for a1,a2 in zip(newvar.axes, self.axes)): return self

#    # was the source var preloaded?
#    # if so, preload this slice as well
#    if hasattr(self,'values'): newvar = newvar.load(pbar=none)

    return newvar
# }}}

  # subset by integer indices
  def __getitem__ (self, slices):
# {{{
    """
    gets a raw numpy array containing a subset of values of the variable.

    parameters
    ----------
    slices : list of slices

    returns
    -------
    out : numpy.ndarray
      the requested values, as a numpy array.

    see also
    --------
    var.get, var.slice, var.__call__

    examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print(t1.temp[:].shape)
    (31, 60)
    >>> print(t1.temp[20:-6, ::12])
    [[285.64721554 287.07380031 286.52889342 284.76553766 284.22063076]
     [281.09169696 282.80359869 282.14971042 280.03368351 279.37979523]
     [276.73945224 278.73667093 277.97380127 275.50510321 274.74223356]
     [272.82122084 275.10375648 274.23190545 271.41053624 270.5386852 ]
     [269.47711035 272.04496294 271.06413053 267.89009017 266.90925775]]
    """
    # get the raw numpy array (with degenerate axes intact)
    array = self._getitem_asvar(slices).get()
    # if any single integer indices were passed, then reduce out those
    # dimensions.  this is consistent with what would happen with numpy slicing.
    if isinstance(slices,tuple):
      extra_slicing = tuple(0 if isinstance(sl,int) else ellipsis if sl is ellipsis else slice(none) for sl in slices)
      array = array[extra_slicing]
    elif isinstance(slices,int):
      array = array[0]
    return array
# }}}

  # select a subset by keyword arguments (i.e., lat = (-45.0, 45.0))
  # keys should be the name of the axis shortcut in the var (i.e., lat, lon, time).
  def __call__ (self, ignore_mismatch = false, **kwargs):
  # {{{
    """
    keyword-based data subsetting.

    parameters
    ----------
    ignore_mismatch : boolean (optional)
      if ``true``, any keywords that don't match an axis are ignored.
      default is ``false``

    **kwargs : one or more keyword parameters
      the keys are the axis names (or axis class names), and the values are
      either a `tuple` of the desired (lower,upper) range, or a single axis
      value.  e.g., ``lat = (-45,45)`` or ``lat = 10.5``

    returns
    -------
    subset_var : var
      a new var, restricted  to the specified domain.

    notes
    -----
    there are a couple of special prefixes which can be prepended to each
    keyword to alter the subsetting behaviour. they can be used together.

      * **i_** indicates that the values are *indices* into the axis, and not
        the axis values themselves.  indices start at 0.
        e.g. ``myvar(i_time = 0)`` selects the first time step of the variable,
        and ``myvar(i_lon=(10,20))`` selects the 11th through 21st longitudes.

      * **l_** indicates that you are providing an explicit list of
        coordinates, instead of a range.
        e.g. ``myvar(l_lon = (105.,106.,107.,108))``

      * **n_** returns the complement of the set you request; that is,
        everything except the specified selection.
        e.g. ``myvar(n_lat = (60, 90))`` returns all latitudes except those between 60 and 90n.

      * **m_** triggers an arithmetic mean over the specified range.
        e.g., ``myvar(m_lon = (10, 80))`` is a shortcut for doing
        ``myvar(lon = (10,80)).mean('lon')``.

      * **s_** triggers a call to squeeze on the specified axis, so
        that if only one value is selected the degenerate axis is removed.
        e.g., ``myvar(s_lon = 5)`` is a shortcut for doing
        ``myvar(lon = 5).squeeze()`` or ``myvar.squeeze(lon=5)``.


    examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print(t1.vars)
    [<var 'temp'>]
    >>> t = t1.temp
    >>> print(t)
    <var 'temp'>:
      units: k  shape:  (lat,lon)  (31,60)
      axes:
        lat <lat>      :  90 s to 90 n (31 values)
        lon <lon>      :  0 e to 354 e (60 values)
      attributes:
        {}
      type:  add_var (dtype="float64")
    >>> print(t(lat=30,lon=(100,200)))
    <var 'temp'>:
      units: k  shape:  (lat,lon)  (1,17)
      axes:
        lat <lat>      :  30 n
        lon <lon>      :  102 e to 198 e (17 values)
      attributes:
        {}
      type:  slicedvar (dtype="float64")
    """
    # check for mean axes
    # (kind of a hack... prepend an 'm_' to the keyword)
    # i.e., var(m_lon=(30,40)) is equivalent to var(lon=(30,40)).mean('lon')

    means = []
    squeezes = []

    newargs = {}
    for k, v in kwargs.items():
      # detect special prefixes.
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

    # get the slices, apply to the variable
    slices = [a.get_slice(newargs, ignore_mismatch=true) for a in self.axes]
    assert len(newargs) == 0 or ignore_mismatch, "unmatched slices remain: %s" % str(newargs)

    ret = self._getitem_asvar(slices)

    # any means or squeezes requested?
    if len(means) > 0: ret = ret.mean(*means)
    if len(squeezes) > 0: ret = ret.squeeze(*squeezes)

    return ret
  # }}}

  # avoid accidentally iterating over vars
  def __iter__ (self):
# {{{
    raise exception ("var instances cannot be iterated over")
# }}}

  # include axes names
  def __dir__(self):
# {{{
    l = list(self.__dict__.keys()) + dir(self.__class__)
    return l + [a.name for a in self.axes]
# }}}

  # shortcuts to axes
  # (handled dynamically, in case some fudging is done to the var)
  def __getattr__(self, name):
# {{{
    # disregard metaclass stuff
    if name.startswith('__'): raise attributeerror

#    print 'var getattr ??', name


    # some things that we *know* should not be axis shortcuts
    if name in ('axes', 'values'):
      raise attributeerror(name)
    try: return self.getaxis(name)
    except keyerror: raise attributeerror ("'%s' not found in %s"%(name,repr(self)))

    raise attributeerror (name)
# }}}


  # modifies __setattr__ and __delattr__ to avoid further modifications to the var
  # (should be called by any var subclasses at the end of their __init__)
  def _finalize (self):
# {{{
    if self.__setattr__ == self._dont_setattr:
      from warnings import warn
      warn ("already finalized the var", stacklevel=2)
      return
    self.__delattr__ = self._dont_delattr
    self.__setattr__ = self._dont_setattr
# }}}

  # send all attribute requests through here, to enforce immutability
  def _dont_setattr(self,name,value):
# {{{
    # can only modify the var name
    if name != 'name':
#      raise typeerror ("can't modify an immutable var")
      from warnings import warn
      warn ("shouldn't be modifying an immutable var", stacklevel=2)
    object.__setattr__(self,name,value)
# }}}

  def _dont_delattr(self,name):
# {{{
#    raise typeerror ("can't modify an immutable var")
    from warnings import warn
    warn ("shouldn't be modifying an immutable var", stacklevel=2)
    object.__delattr__(self,name)
# }}}

  # get an axis based on its class
  # ie, somevar.getaxis(zaxis)
  def getaxis (self, iaxis):
  # {{{
    """
      grabs a reference to a particular :class:`axis` associated with this variable.

      parameters
      ----------
      iaxis : string, :class:`axis` class, or int
        the search criteria for finding the class.

      returns
      -------
      axis : :class:`axis` object that matches.  raises ``keyerror`` if there is no match.

      see also
      --------
      whichaxis, hasaxis, axis
    """
    return self.axes[self.whichaxis(iaxis)]
  # }}}

  # get the index of an axis given its class (or a string)
  # (similar to above, but return the index, not the axis itself)
  def whichaxis (self, iaxis):
  # {{{
    """
    locates a particular :class:`axis` associated with this variable.

    parameters
    ----------
    iaxis : string, :class:`axis` class, or int
      the search criteria for finding the class.

    returns
    -------
    i : the index of the matching axis.  raises ``keyerror`` if there is no match.

    examples
    --------
    >>> from pygeode.tutorial import t1
    >>> t = t1.temp
    >>> print(t.axes)
    (<lat>, <lon>)
    >>> print(t.whichaxis('lat'))
    0
    >>> print(t.whichaxis('lon'))
    1
    >>> print(t.whichaxis('time'))
    traceback (most recent call last):
      file "<stdin>", line 1, in <module>
      file "/usr/local/lib/python2.6/dist-packages/pygeode/var.py", line 352, in whichaxis
        raise keyerror, "axis %s not found in %s"%(repr(iaxis),self.axes)
    keyerror: "axis 'time' not found in (<lat>, <lon>)"

    see also
    --------
    getaxis, hasaxis, axis
    """
    from pygeode.tools import whichaxis
    return whichaxis(self.axes, iaxis)
  # }}}

  # indicate if an axis is found
  def hasaxis (self, iaxis):
  # {{{
    """
      determines if a particular :class:`axis` is associated with this variable.

      parameters
      ----------
      iaxis : string, :class:`axis` class, or int
        the search criteria for finding the class.

      returns
      -------
      found : boolean.  ``true`` if found, otherwise ``false``

      see also
      --------
      getaxis, whichaxis, axis
    """
    try: i = self.whichaxis(iaxis)
    except keyerror: return false
    return true
  # }}}

  def formatvalue(self, value, fmt=none, units=true, unitstr=none):
  # {{{
    '''
    returns formatted string representation of ``value``.

    parameters
    ----------
    value : float or int
      value to format.
    fmt : string (optional)
      format specification. if the default ``none`` is specified,
      ``self.formatstr`` is used.
    units : boolean (optional)
      if ``true``, will include the units in the string returned.
      default is ``true``.
    unitstr : string (optional)
      string to use for the units; default is ``self.units``.

    returns
    -------
    if units is true, fmt % value + ' ' + unitstr. otherwise fmt % value.

    notes
    -----
    this is overridden in a number of axis classes for more sophisticated formatting.
    '''

    if fmt is none: fmt = self.formatstr
    strval = fmt % value

    if units:
      if unitstr is none: unitstr = self.units
      strval += ' ' + unitstr

    return strval
  # }}}

  # pretty printing
  def __str__ (self):
  # {{{
    from textwrap import textwrapper
#    axes_list = '(' + ', '.join(a.name if a.name != '' else repr(a) for a in self.axes) + ')'
#    axes_details = ''.join(['  '+str(a) for a in self.axes])
#    s = repr(self) + ':\n\n  axes: ' + axes_list + '\n  shape: ' + str(self.shape) + '\n\n' + axes_details
    s = repr(self) + ':\n'
    if self.units != '':
      s += '  units: ' + self.units
    s += '  shape:'
    s += '  (' + ','.join(a.name for a in self.axes) + ')'
    s += '  (' + ','.join(str(len(a)) for a in self.axes) + ')\n'
    s += '  axes:\n'
    for a in self.axes:
      s += '    '+str(a) + '\n'

    w = textwrapper(width=80)
    w.initial_indent = w.subsequent_indent = '    '
    w.break_on_hyphens = false
    s += '  attributes:\n' + w.fill(str(self.atts)) + '\n'
    s += '  type:  %s (dtype="%s")' % (self.__class__.__name__, self.dtype.name)
    return s
  # }}}

  def __repr__ (self):
  # {{{
    if self.name != '':
#      return '<' + self.__class__.__name__ + " '" + self.name + "'>"
      return "<var '" + self.name + "'>"
    return '<var>'
  # }}}

  def get (self, pbar=none, **kwargs):
  # {{{
    """
    gets a raw numpy array containing the values of the variable.

    parameters
    ----------
    pbar : boolean (optional)
      if ``true``, will display a progress bar while the data is being
      retrieved.  this requires the *python-progressbar* package (not included
      with pygeode).

    **kwargs : keyword arguments (optional)
      one or more keyword arguments may be included to subset the variable
      before grabbing the data.  see :func:`var.__call__` for a similar
      method which uses this keyword subsetting.

    returns
    -------
    out : numpy.ndarray
      the requested values, as a numpy array.

    notes
    -----

    once you grab the data as a numpy array, you can no longer use the pygeode
    functions to do further work on it directly.  you can, however, use
    :func:`var.__init__` to re-wrap your numpy array as a pygeode var.  this
    may be useful if you want to do some very complicated operations on the
    data using the numpy interface as an intermediate step.

    pygeode variables can be huge!  they can be larger than the available ram
    in your computer, or even larger than your hard disk.  numpy arrays, on
    the other hand, need to fit in memory, so make sure you are only getting
    a reasonable piece of data at a time.

    examples
    --------
    >>> from pygeode.tutorial import t1
    >>> print(t1.temp)
    <var 'temp'>:
      units: k  shape:  (lat,lon)  (31,60)
      axes:
        lat <lat>      :  90 s to 90 n (31 values)
        lon <lon>      :  0 e to 354 e (60 values)
      attributes:
        {}
      type:  add_var (dtype="float64")
    >>> x = t1.temp.get()
    >>> print(x)
    [[260.73262556 258.08759192 256.45287123 ... 265.01237988 265.01237988
      263.37765919]
     [261.22683172 258.75813366 257.23239435 ... 265.22126909 265.22126909
      263.69552978]
     [261.98265134 259.69028886 258.27353093 ... 265.69177175 265.69177175
      264.27501382]
     ...
     [261.98265134 264.27501382 265.69177175 ... 258.27353093 258.27353093
      259.69028886]
     [261.22683172 263.69552978 265.22126909 ... 257.23239435 257.23239435
      258.75813366]
     [260.73262556 263.37765919 265.01237988 ... 256.45287123 256.45287123
      258.08759192]]
    """
    from pygeode.view import view
    import numpy as np
    var = self.__call__(**kwargs)
    data = view(var.axes).get(var,pbar=pbar)
    if isinstance(data,np.ndarray):
      data = np.array(data,copy=true)
    return data
  # }}}

  def getweights (self, iaxes = none):
  # {{{
    ''' returns weights associated with the axes of this variable.

    parameters
    ----------
    iaxes : list of axis identifiers (string, :class:`axis`, or int) (optional)
      axes over which to compute the weights

    returns
    -------
    weights : :class:`var`
      defined on the subgrid of this variable on which weights are defined.

    see also
    --------
    axis.__init__
    '''
    # add weights object if any of the axes are weighted
    # take outer product of weights if present on more than one axis
    if iaxes is none or len(iaxes) == 0:
      axes = self.axes
    else:
      axes = [self.getaxis(i) for i in iaxes]
    wt = tuple([a.auxasvar('weights') for a in axes if hasattr(a, 'weights')])
    if len(wt) == 0:
#      return none
      return var(axes=[], values=1.0)  # return degenerate variable?
    else:
      from pygeode.ufunc import vprod
      return vprod(*wt) # variable-wise product of weights
  # }}}

  # preload the values of a variable
  # if the values are already loaded, then return self
  def load (self, pbar=true):
  # {{{
    ''' returns a version of this variable with all data loaded into memory.

    parameters
    ----------
    pbar : boolean
      if true, display a progress bar while loading data.
    '''
    from pygeode.progress import pbar
    if hasattr(self, 'values'): return self
    if pbar is true:
      pbar = pbar(message="loading %s:"%repr(self))
    var = var(self.axes, values=self.get(pbar=pbar))
    copy_meta (self, var)
    return var
  # }}}

  # plotting helper functions
  def formatter(self):
  # {{{
    '''returns a matplotlib formatter (pygeode.axisformatter) for use in plotting. '''
    from pygeode.axis import axisformatter
    return axisformatter(self)
  # }}}

  def locator(self):
  # {{{
    '''returns a matplotlib locator object for use in plotting.'''
    import pylab as pyl
    scl = self.plotatts.get('plotscale', 'linear')
    if scl == 'log': return pyl.loglocator()
    else: return pyl.autolocator()
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
from functools import reduce
Var.__add__  = wrap_binary(nd.__add__,  symbol='+')
Var.__radd__ = wrap_binary(nd.__radd__, symbol='+')
Var.__sub__  = wrap_binary(nd.__sub__,  symbol='-')
Var.__rsub__ = wrap_binary(nd.__rsub__, symbol='-')
Var.__mul__  = wrap_binary(nd.__mul__,  symbol='*')
Var.__rmul__ = wrap_binary(nd.__rmul__, symbol='*')
if hasattr(nd,'__div__'):  # Python2
  Var.__div__  = wrap_binary(nd.__div__,  symbol='/')
  Var.__rdiv__ = wrap_binary(nd.__rdiv__, symbol='/')
else:  # Python3
  Var.__truediv__  = wrap_binary(nd.__truediv__,  symbol='/')
  Var.__rtruediv__ = wrap_binary(nd.__rtruediv__, symbol='/')
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
class_hooks += [_f for _f in [smooth, deriv, diff, integrate, composite, flatten, fft_smooth, lag, interpolate, squeeze, extend, transpose, sorted, replace_axes, rename, rename_axes, fill, unfill, as_type] if _f]
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
