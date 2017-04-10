#TODO: check the order of the concat axis
#TODO: skip over var segs not involved in the query.
#      right now, it's a linear search through the whole thing.
#TODO: more efficient comparison of axes

from var import Var


class ConcatVar(Var):
  def __init__(self, vars, iaxis=None):
    import pygeode.axis
    from pygeode.tools import common_dtype
    from pygeode.var import combine_meta
    import numpy as np

    # Use first var segment for the axes
    axes = list(vars[0].axes)
    naxes = len(axes)
    # For now, assume all segments have the same order of axes
    assert all(v.naxes == naxes for v in vars)
    for i in range(naxes):
      assert all(axes[i].isparentof(v.axes[i]) for v in vars)

    if iaxis is None:
      iaxis = set(i for v in vars for i in range(naxes) if v.axes[i] not in axes)
  
      assert len(iaxis) <= 1, "more than one varying axis id=%s for %s; can't concatenate"%(iaxis,repr(vars[0]))
  
      # Degenerate case: all segments have identical axes
      if len(iaxis) == 0:
        from warnings import warn
        warn ('all axes are identical.  Creating a fake "concat" axis', stacklevel=2)
        iaxis = naxes
        axes.append(pygeode.axis.NamedAxis(len(vars), name='concat'))
  
      # Standard case: exactly one concatenation axis
      else:
        iaxis = iaxis.pop()

    if not iaxis is naxes:
      # Get a numerical dimension number
      iaxis = vars[0].whichaxis(iaxis)

      # Update the list of axes with the concatenated axis included
      axes[iaxis] = pygeode.axis.concat([v.axes[iaxis] for v in vars])

    # Get the data type
    dtype = common_dtype(vars)

    Var.__init__(self, axes, dtype=dtype)

    # Grab metadata from the input variables
    combine_meta (vars, self)

#    # Assign a name (and other attributes??) to the var
#    name = set(v.name for v in vars if v.name != '')
#    if len(name) == 1: self.name = name.pop()
#
#    # Combine the attributes (if applicable)
#    atts = common_dict([v.atts for v in vars])
#    self.atts = atts
#    # Combine the plot attributes (if applicable)
#    plotatts = common_dict([v.plotatts for v in vars])
#    self.plotatts = plotatts

    # Other stuff
    self.vars = vars
    self.iaxis = iaxis



  def getview (self, view, pbar):
    import numpy as np
    # Degenerate case: separate concat axis
    # (would be appended to the end of the axis list)
    if self.iaxis == self.vars[0].naxes:
      N = len(view.integer_indices[self.iaxis])
      subview = view.remove(self.iaxis)
      chunks = [np.expand_dims(subview.get(self.vars[i], pbar=pbar.part(n,N)), self.iaxis)
              for n,i in enumerate(view.integer_indices[self.iaxis])]
      return np.concatenate(chunks, axis=self.iaxis)
    #TODO: fix this, once there's a common_map
    chunks = [view.get(v,strict=True,conform=False) for v in self.vars]
    pbar.update(100)  # can't really use this here, since we don't know which var segments are actually used
    return np.concatenate(chunks, axis=self.iaxis)

def concat (*vars, **kwargs):
  from pygeode.var import Var
  from pygeode.tools import islist
  # Already passed a list?
  if len(vars) == 1 and islist(vars[0]):
    vars = vars[0]
  # Degenerate case: only one segment
  if len(vars) == 1: return vars[0]
  # Make sure we have vars
  assert all(isinstance(v,Var) for v in vars), "can only concatenate Var objects in this routine"

  return ConcatVar(vars, **kwargs)

