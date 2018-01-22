# An interface to describe a 'view' of a grid.
# Consists of a bunch of axes and ranges.
# Used in variable read routines, to simplify the mapping between input and output variables.


# Attributes:
#   axes    - a list of axis objects to use for the view (whole axis, not subsetted)
#   slices  - a list of slices describing what parts of the axes we're actually viewing
#             (can include non-uniform lists of indices in addition to slices)
#   integer_indices - an explicit array of integer indices, indicating which values along the axes have been selected.

def expand (ind, upper_bound):
# {{{
  '''Expand a slice or integer into an explicit array of indices'''
  import numpy as np
  # already a list?
  if hasattr(ind,'__len__'): return np.array(ind, dtype='int32')
  # a single integer index?
  if isinstance(ind,int): return np.array([ind], dtype='int32')
  # Otherwise, we should have a slice now
  assert isinstance(ind,slice), "unknown slicing mechanism - "+repr(ind)
  # Make sure start, stop, and step are positive integers
#  ind = fix_slice(ind, upper_bound)
#  return np.arange(ind.start, ind.stop, ind.step)

  start, stop, step = ind.start, ind.stop, ind.step
  if step is None: step = 1
  assert step != 0
  if start is None:
    if step > 0: start = 0
    else: start = upper_bound-1
  if start < 0: start += upper_bound

  if stop is None:
    if step > 0: stop = upper_bound
    else: stop = -1
  elif stop < 0: stop += upper_bound

  return np.arange(start, stop, step, dtype='int32')
# }}}


def simplify (ind):
# {{{
  '''Simplify an index list into a slice, if possible
    does *not* return single integer indices, since that screws up the dimensions
    when applying these slices to numpy arrays.'''
  assert hasattr(ind,'__len__'), "not an array of indices"
  # Single value?
  if len(ind) == 1:
    ind = int(ind[0])
    return slice(ind,ind+1,1)
  # otherwise, check the distance between each value - see if it's regular
  import numpy as np
  delt = np.unique(np.diff(ind))
  if len(delt) != 1 or delt == 0: return ind  # irregular indices, can't do anything
  delt = int(delt[0])
  start, stop, step = int(ind[0]), int(ind[-1])+delt, delt
  # Special case: going in reverse, ending at the start of the array - stop will be just before index 0, i.e. a negative stop value
  # (because of the way negative indices are wrapped in numpy, this won't behave the way we'd want)
  # so, for this case only, set 'stop' equal to None, which should tell numpy to go until you can't go anymore
  if stop < 0: stop = None
  return slice(start, stop, step)
# }}}

def contiguate (ind):
# {{{
  '''how to contiguate(?) the slices without actually returning the contiguous slices
      (returns what you would pass to slice_into to get the contiguous slices)'''
  assert hasattr(ind,'__len__'), "not an array of indices"
  import numpy as np
  delt = np.diff(ind)
  # Find all non-contiguous 'jumps'
  stops = list(np.where(delt!=1)[0] + 1) + [len(ind)]
  starts = [0] + stops[:-1]
  return [slice(i,j) for i,j in zip(starts,stops)]
# }}}


#TODO: write a set of integer indices as a superposition of slices
# ideally, this should return a minimum number of slices to cover all indices
# Assume the indices are in order!
def indices_to_slices (ind):
  import numpy as np
  d = np.diff(ind)
  assert not np.any (d <= 0)
  slices = []
  #TODO

class View:
# {{{
  def __init__(self, axes, slices=None, force_slices=None, force_integer_indices=None):
  # {{{  

    if hasattr(axes,'axes'): axes = axes.axes

    self.axes = tuple(axes)
    self.naxes = len(axes)

    # if no slices given, default to the whole range
    if slices is None: slices = [slice(None)] * len(axes)

    # don't allow a generic 'None' as a slice - it causes problems in numpy
    assert not any(s is None for s in slices)
    assert len(axes) == len(slices)

    if force_integer_indices is not None:
      self.integer_indices = tuple(force_integer_indices)
    else:
      self.integer_indices = tuple(expand(sl,len(a)) for sl,a in zip(slices,axes))

    if force_slices is not None:
      self.slices = tuple(force_slices)
    else:
      self.slices = tuple(simplify(e) for e in self.integer_indices)

    self.shape = tuple(len(a) for a in self.integer_indices) 
    self.size = 1
    for s in self.shape: self.size *= s

    # Don't allow empty views?
    # Remove this if it interferes with normal operations, I just added it
    # to shorten the stack trace when a bad mapping was taking
#    assert not any(len(ind) == 0 for ind in self.integer_indices), self.slices

  # }}}

  def map_to (self, axes, strict=True, order=[]):
  # {{{
    '''Map one view onto another.
        Modifies the slices to work with the new list of axes.
        'strict' indicates if all the output axes must correspond to an input axes
          (if not, the default behaviour is to include the entire axis if it's not mentioned in the input view)
        'order' returns the ordering of the output axes to get the original axes back (-1 for things that can't be mapped)'''

    assert isinstance(order,list), "can't output order to a non-list"

    if hasattr(axes,'axes'): axes = axes.axes  # check if a var or view was passed

    slices = []
    order[:] = [-1] * len(self.axes)

    for out_i,out_a in enumerate(axes):
      #TODO: remove this special case, let Axis.map_to handle it
      if out_a in self.axes:
        in_i = list(self.axes).index(out_a)
        slices.append(self.slices[in_i])
        order[in_i] = out_i
        continue
      # Look for a mapping
      map = None
      for in_i,in_a in enumerate(self.subaxes()):
        map = out_a.map_to(in_a)
        if map is not None:
          slices.append(simplify(map))
          order[in_i] = out_i
          break
      if map is not None: continue  # found a map
      # otherwise, nothing of interest was found, so default to a full slice on the axis
      assert strict is False, "Can't do a strict mapping from %s to %s"%(self.axes, axes)
      slices.append(slice(None))

# eventually, want:
# ->  out[:,x,y] = in[[0,1,2,3,4,5,6,7,8,9,10,11,0,1,2,3,4,5,6,7,8,9,10,11,...],x,y]
#
# so, out_times can map to in_times (climatology)
# so, we would have out_time.map_to(in_time) = [0,1,2,3,4,5,6,7,8,9,10,11,0,1,2,3,4,5,6,7,8,9,10,11,...]
# assume our view is on the output
# view.map_to(in) = [[0,1,2,3,4,5,6,7,8,9,10,11,0,1,2,3,4,5,6,7,8,9,10,11,...],x,y]
#             = [out_time.map_to(in_time), out_x.map_to(in_x), ...]
    assert not any(s is None for s in slices)
    return View(axes, slices = slices)

  # }}}

  def subaxis (self, iaxis):
    '''Returns the specified axis, after selecting the elements that are in the current view.'''
    iaxis = self.index(iaxis)
    return self.axes[iaxis].slice [self.slices[iaxis]]

  def subaxes (self):
    return tuple(self.subaxis(i) for i in range(self.naxes))

  def clip (self):
  # {{{
    '''Return the same view, but with the axes clipped to cover only the selected region
    (the axes could cover more than the selection)
    i.e. latitude axis could be from 90S to 90N, but the view region is 45S to 45N
    Having a larger axis than needed could be useful in some cases, such as
    when the current view needs to be extended to take more values
    (i.e., taking a derivative might need to extend latitudes to 46N to 46S to do a finite difference)
    One useful case for clipping is if you want a 1:1 mapping from the view to an output array
    (and the output array would only be defined on the view region)'''
    return View (self.subaxes())
  # }}}

  def get (self, var, pbar=False, strict=True, conform=True):
  # {{{
    '''Applies this view to the given variable, returning 
      the data conformed to the shape of this view. (strict, conform keywords?)'''

#    from pygeode.progress import FakePBar
#    assert not isinstance(pbar, FakePBar)

    import numpy as np
    from warnings import warn

    from pygeode.progress import FakePBar
    if pbar is False or pbar is None: pbar = FakePBar()

    assert strict in (True, False)
    assert conform in (True, False)

    # If we allow the output axes to contain things not found in the input axes,
    # then it is impossible(?) to conform the array to match the inputs
    assert not (strict is False and conform is True), "Can't conform an array from an unstrict get"

    # Map to the var's axes, and read the data
    # Keep track of how to reverse this operation (if it can be reversed)
    order = []
    newview = self.map_to(var.axes, strict=strict, order=order)  # It rubs the lotion on its skin

    # If the shape is degenerate, then we can just return the empty array here
    if any(s == 0 for s in newview.shape):
#      print 'FAIL:'
#      print 'input var:', repr(var), var.axes
#      print 'view:', self.axes, self.slices
#      print 'mapped view:', newview.axes, newview.slices
#      raise Exception

      return np.empty([s if o > -1 else 1 for s,o in zip(newview.shape,order)], var.dtype)
    assert newview.axes == var.axes                              # Or it gets the hose again

    # Only request unique elements from the input
    unique_indices = [np.unique(ind) for ind in newview.integer_indices]
    if all(np.all(u == ind) for u,ind in zip(unique_indices, newview.integer_indices)):
      # already unique
      duplicator = ()
      unique_view = newview
    else:
      # uniquify
      duplicator = [np.searchsorted(u, ind) for u,ind in zip(unique_indices, newview.integer_indices)]
      # need to massage this into something numpy can handle
      duplicator = tuple([d.reshape([1]*i+[-1]+[1]*(newview.naxes-i-1)) for i,d in enumerate(duplicator)])
      unique_view = View(newview.axes, force_integer_indices = unique_indices)

#    from pygeode.var import Var
    if hasattr(var,'values'):
#      values = var.values[unique_view.slices]
#  ^^ can't do this if we are slicing by integer indices.
# (Refer to issue 6 - https://github.com/pygeode/pygeode/issues/6 )
# Instead, apply the slices one at a time.
# (TODO: instead, conform the index array shapes the way duplicator is done
#  above, but need to have addictional check if we have slices or indices.)
      values = var.values
      for i,sl in enumerate(unique_view.slices):
        values = values[[slice(None)]*i + [sl]]

    elif hasattr(var,'getview'):
      # Can we use the progress bar?
      if 'pbar' in var.getview.func_code.co_varnames:
        values = var.getview(unique_view, pbar=pbar)
      else:
        print 'no pbar in', type(var)
        values = var.getview(unique_view)

    elif hasattr(var,'getvalues'):
      values = np.empty(unique_view.shape, var.dtype)
      # Loop over contiguous pieces, build the result
      loop = list(unique_view.loop_contiguous())
      for i, (outsl, start, count) in enumerate(loop):
        values[outsl] = var.getvalues(start, count)
        pbar.update (100./len(loop) * i)

    else:
      raise IOError, "can't determine how to extract values from "+repr(var)

    pbar.update(100)

    assert isinstance(values,(np.ndarray,np.number)), "did not receive a valid array from %s"%repr(var)

    if values.dtype != var.dtype:
      warn ("%s is supposed to return a %s, but is returning %s"%(repr(var),var.dtype.name,values.dtype.name))
      values = np.asarray(values, var.dtype)

    if not isinstance(values,np.ndarray):
      warn ("%s is returning a numpy scalar instead of an ndarray - re-wrapping it now"%type(var))
      values = np.array(values)

    # Use these unique elements to generate the expected array, including duplicates
    values = values[duplicator]

    #TODO: better error messages
    assert len(values.shape) == len(var.axes)
    if strict is True:
      assert values.shape == newview.shape, "expected shape %s, got shape %s (culprit is %s) view is %s"%(newview.shape,values.shape,var,self)

    # conform the array to the view shape
    if conform is True:
#      print "conforming!"
#      print "order from %s to %s is %s"%(self.axes,var.axes,order)
      values = values.transpose([o for o in order if o > -1])
      assert len(self.shape) == len(order)
      values = values.reshape([s if o > -1 else 1 for s,o in zip(self.shape,order)])  # It puts the lotion in the basket

#    pbar.update(100)

    # Write-protect the values, just in case they're shared by multiple components
    # Can't write-protect scalars, though?
    if values.ndim > 0: values.flags.writeable = False

#    if values.max() == 0 and values.min() == 0:
#      warn ("%s chunk is all zeros"%repr(var))

    return values
  # }}}


  ###########################################################################
  # View modification functions
  #

  def add_axis (self, iaxis, axis, sl):
  # {{{
    '''Add a new axis (and slice) to the view'''
    iaxis = self.index(iaxis)
    axes = self.axes[:iaxis] + (axis,) + self.axes[iaxis:]
    slices = self.slices[:iaxis] + (sl,) + self.slices[iaxis:]
    return View(axes, slices)
  # }}}

  def remove(self, *iaxes):
  # {{{
    '''  Remove specified axes from the view entirely'''
    ind = []
    for iaxis in iaxes:
      ind.append(self.index(iaxis))
    rind = [i for i in range(len(self.slices)) if i not in ind]
    axes = [self.axes[i] for i in rind]
    slices = [self.slices[i] for i in rind]
    integer_indices = [self.integer_indices[i] for i in rind]
#    return View (axes, slices)
    return View (axes, force_slices = slices, force_integer_indices = integer_indices)
  # }}}

  def replace_axis (self, iaxis, axis, sl=slice(None)):
  # {{{
    iaxis = self.index(iaxis)
    axes = list(self.axes)
    axes[iaxis] = axis
    slices = list(self.slices)
    slices[iaxis] = sl
    return View(axes, slices)
  # }}}

  def modify_slice (self, iaxis, newslice):
  # {{{
    '''Modify a slice of the view'''
    iaxis = self.index(iaxis)
    return View(self.axes, self.slices[:iaxis] + (newslice,) + self.slices[iaxis+1:])
  # }}}

  def unslice(self, *iaxes):
  # {{{
    '''  Remove any slicing on the specified axes'''
    from pygeode.axis import Axis
    slices = list(self.slices)
    for iaxis in iaxes:
      ind = self.index(iaxis)
      if ind >= 0: slices[ind] = slice(None)
    return View (self.axes, slices)
  # }}}

  def only_slice(self, *iaxes):
  # {{{
    '''Remove slicing for all but the specified axes'''
    slices = [slice(None)] * len(self.slices)
    for iaxis in iaxes:
      ind = self.index(iaxis)
      if ind >= 0: slices[ind] = self.slices[ind]
    return View(self.axes, slices)
  # }}}

  #Note: this is very similar to Var.whichaxis().
  #Is there some (sensible) way to remove these kinds of redundancies between Vars and Views?
  # (both have a list of axes associated with them, and methods to lookup / modify them)
  def index (self, axis):
  # {{{
    ''' Returns index of matching axis if present; -1 otherwise. '''
    from pygeode.axis import Axis
    if isinstance(axis, int):
      assert 0 <= axis <= len(self.axes)  # TODO: need more strict upper bound?
      return axis

    # axis object?
    if isinstance(axis, Axis): 
      try: 
        return list(self.axes).index(axis)
      except ValueError: 
        return -1

    # axis class?
    for i, a in enumerate(self.axes):
      if isinstance(a, axis): return i

    return -1
  # }}}

  #############
  # iterators
  #############


  def loop_contiguous(self):
  # {{{
    '''Break a non-contiguous view up into contiguous pieces
      (so that we don't have to handle it at a lower level)
      Input: this view
      Generates: outsl, start, count
         outsl is the current slice into an array that contains the view in a
               contiguous piece of memory
         start, count are corresponding arrays giving the start of the slice & its length.
      NOTE: These pieces might not fit in memory (see loop_mem for handling that).
            The only guarantee is that the pieces will be contiguous.  If the input
            view is already contiguous, then the slice will be over the *whole* view.'''
    #TODO: use itertools.product when everyone has python >= 2.6
    from itertools import product
    #from pygeode.tools import product
    outslices = [contiguate(e) for e in self.integer_indices]
    inslices = [[simplify(ind[osl]) for osl in outsl] for ind,outsl in zip(self.integer_indices, outslices)]
    for outsl, insl in zip(product(*outslices), product(*inslices)):
      start = [sl.start if isinstance(sl,slice) else sl for sl in insl]
      count = [sl.stop - sl.start if isinstance(sl,slice) else 1 for sl in insl]
      yield outsl, start, count


  # }}}

  def _loop_mem (self, preserve=None):
  # {{{
    '''Loop over smaller pieces of the view that fit in memory'''
    from pygeode import MAX_ARRAY_SIZE
    #TODO: use itertools.product when everyone has python >= 2.6
    from itertools import product
    #from pygeode.tools import product
    # Determine the largest chunk that can be loaded, given the size constraint
    maxsize = MAX_ARRAY_SIZE
 
    # Get the shape of a single chunk
    indices = []
    for i in reversed(range(len(self.axes))):
      # Number of values along this axis we can get at a time
      N = self.shape[i]  # total length of this axis
      n = min(maxsize,N)  # amount we can fix in a single chunk
      # break up the axis into subslices
      input_indices = self.integer_indices[i]
      ind = [input_indices[j:min(j+n,N)] for j in range(0,N,n)]
      # Build the subslices from the last axis to the first
      indices = [ind] + indices
      # Take into account the count along this axis when looking at the faster-varying axes
      maxsize /= n
 
    # Loop over all combinations of slices to cover the whole view
    # (take the cartesian product)
    for ind in product(*indices):
      yield View(self.axes, ind)
  # }}}

  def loop_mem (self, preserve=None):
  # {{{
    '''Loop over smaller pieces of the view that fit in memory. preserve
        can optionally be a list of integer indices of axes that should be loaded
        in their entirety in each chunk. A warning is thrown if this ends up being
        larger than MAX_ARRAY_SIZE, but not an exception; memory allocation problems
        may result in this case. '''

    from pygeode import MAX_ARRAY_SIZE
#TODO: use itertools.product when everyone has python >= 2.6
    from itertools import product
    #from pygeode.tools import product
    from warnings import warn
    # Determine the largest chunk that can be loaded, given the size constraint
    maxsize = MAX_ARRAY_SIZE

    # Get the shape of a single chunk
    indices = [[] for s in self.shape]

    if preserve is not None:
      for i in preserve:
        N = self.shape[i]
        indices[i] = [self.integer_indices[i]]
        if maxsize < N: 
          warn('Data request involves arrays larger than MAX_ARRAY_SIZE; continuing for now but memory allocation problems may result.')
          maxsize = 1
        else:
          maxsize /= N
      others = [i for i in range(len(self.axes)) if i not in preserve]
    else:
      others = range(len(self.axes))

    # Build the subslices from the last axis to the first
    for i in reversed(others):
      # Number of values along this axis we can get at a time
      N = self.shape[i]  # total length of this axis
      n = min(maxsize,N)  # amount we can fix in a single chunk
      # break up the axis into subslices
      input_indices = self.integer_indices[i]
      indices[i] = [input_indices[j:min(j+n,N)] for j in range(0,N,n)]
      # Take into account the count along this axis when looking at the faster-varying axes
      maxsize /= n

    # Loop over all combinations of slices to cover the whole view
    # (take the cartesian product)
    for ind in product(*indices):
      yield View(self.axes, ind)
  # }}}

  #######################
  # string representations
  #######################
  
  def strtok (self):
  # {{{  
    out = []
    for a,s in zip(self.axes, self.slices):
      v = a.values[s]
      if hasattr(v,'__len__'):
        if len(v) > 0:
          out.append(a.name+': '+str(a.formatvalue(v[0]))+' to '+str(a.formatvalue(v[-1])))
        else: out.append(a.name+': <none>')
      else:
       out.append(a.name+': '+str(a.formatvalue(v)))
    return out
  # }}}

  def __str__ (self):
  # {{{
    return '\n'.join(self.strtok())
  # }}}

  def __repr__ (self):
  # {{{
    return 'View(' + ', '.join(self.strtok()) + ')'
  # }}}
# }}}


