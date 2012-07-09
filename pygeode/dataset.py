#TODO: filter function (similar to map, but use a boolean selection function
#      instead of a transformation).

# Dataset
class Dataset(object):
  # Get a variable by name
  def __getitem__ (self, key):
    if key in self.vardict: return self.vardict[key]
    if key in self.axisdict: return self.axisdict[key]
    raise KeyError("%s not in %s.  Valid keys are %s"%(key,repr(self),self.vardict.keys()))


  # Check if we have a variable of the given name
  def __contains__ (self, key):
    return key in self.vardict or key in self.axisdict
  # Reference a axis/variable as an attribute (for the super lazy)
  def __getattr__ (self, name):
    # Disregard metaclass stuff
    if name.startswith('__'): raise AttributeError
#    print 'dataset getattr ??', name
    if name in self: return self[name]
    raise AttributeError (name)

  def __dir__(self):
    l = self.__dict__.keys() + dir(self.__class__)
    return l + self.vardict.keys() + self.axisdict.keys()

  # Iterate over the variables
  def __iter__(self): return iter(self.vars)

  # Dataset Initialization
  # Takes a list of Vars, and any global attributes
  # Vars and axes may be renamed within the Dataset to ensure uniqueness
  def __init__ (self, vars, atts={}, print_warnings=True):
    from pygeode.var import Var
    from warnings import warn
    atts = atts.copy()

    vars = list(vars)
    for v in vars: assert isinstance(v,Var)

    namedict = axis_name_clumping(vars)

    # Rename axes that share a common name
    newaxes = {}
    for oldname, eqlist in namedict.iteritems():
      # Case 1: the name is unique
      if len(eqlist) == 1:
        newaxis = eqlist[0][0]
        for a in eqlist[0]: newaxes[id(a)] = newaxis
      # Case 2: there is more than one axis with that name
      else:
        for i, eq in enumerate(eqlist):
          newname = oldname + '%02d'%(i+1)
          # It's possible that we still have a name conflict
          while newname in namedict:  newname += 'x'
          if print_warnings:
            warn ("renaming non-unique axis '%s' to '%s'"%(oldname,newname), stacklevel=2)
          newaxis = eq[0].rename(newname)
          for a in eq:
            newaxes[id(a)] = newaxis

    # Wrap the vars with these new axes
    for i,v in enumerate(vars):
      axes = [newaxes[id(a)] for a in v.axes]
      # Check if we're already using the right axes
      if all(a1 is a2 for a1,a2 in zip(v.axes,axes)): continue
      vars[i] = v.replace_axes(newaxes=axes)


    # Gather all axes together into a list (unique axes only, semi-ordered)
    axes = []
    axis_ids = []
    for v in vars:
      for a in v.axes:
        if id(a) not in axis_ids:
          axes.append(a)
          axis_ids.append(id(a))

    self.axes = axes
    self.axisdict = dict([a.name,a] for a in axes)
    self.atts = atts  # global attributes

    # Handle name clobbering here

    # Get list of names, fill in blanks
    for i, v in enumerate(list(vars)):
      if v.name == '':
        if print_warnings:
          warn ('unnamed variables found - using default name "var"')
        vars[i] = v.rename('var')

    # Check for duplicate variable names
    oldnames = [v.name for v in vars]
    namecount = {}
    for n in oldnames:
      if n not in namecount: namecount[n] = 0
      namecount[n] += 1
    replace_name = [False if namecount[n] == 1 else True for n in oldnames]
    namecount = dict([n,0] for n in set(oldnames))

    for i, v in enumerate(list(vars)):
      if replace_name[i]:
        n = v.name
        if print_warnings:
          warn ('multiple variables with the name "%s" found - adding integer suffixes'%n)
        namecount[n] += 1
        newname = "%s%02d"%(n,namecount[n])
        # It's possible that we still have a naming conflict
        while newname in namecount:  newname += 'x'
        vars[i] = v.rename(newname)


    self.vars = vars
    self.vardict = dict([v.name,v] for v in vars)


  # String arrays representing the variables, dimensions, etc.
  # Feeds into __str__ below, to simplify it a bit
  def __str_vararr__ (self):
    for v in self.vars:
      oldname = v.name
      # Get the name used as the reference for the dataset
      # (not necessarily the var's name)
      name = [n for n,v2 in self.vardict.iteritems() if v2 is v].pop()
      axes = '(' + ','.join(a.name for a in v.axes) + ')'
      shape = ' (' + ','.join(str(len(a)) for a in v.axes) + ')'
      yield name, axes, shape

  # String representation
  def __str__ (self):
    from textwrap import TextWrapper
    # Degenerate case - no variables??
    if len(self.vars) == 0:
      return '<empty Dataset>'
    lines = list(self.__str_vararr__())
    pad1 = max(len(a[0]) for a in lines) + 1
    pad2 = max(len(a[1]) for a in lines) + 1

    s = '<' + self.__class__.__name__ + '>:\n'

    s = s + 'Vars:\n'
    for name,dims,shape in lines:
     s = s + '  ' + name.ljust(pad1) + dims.ljust(pad2) + shape + '\n'

    s = s + 'Axes:\n  ' + '  '.join([str(a)+'\n' for a in self.axes])

    w = TextWrapper(width=80)
    w.initial_indent = w.subsequent_indent = '  '
    w.break_on_hyphens = False
    s = s + 'Global Attributes:\n' + w.fill(str(self.atts))

    return s

  # Make a copy of a dataset
  # (copies the internal lists and dictionaries, does *not* copy the vars)
  def copy (self): return asdataset(self, copy=True)

  # Rename some variables in the dataset
  # (need to update vars, vardict)
  def rename_vars (self, vardict={}, **kwargs):
    vardict = dict(vardict, **kwargs)
    varlist = list(self.vars)
    for i, v in enumerate(varlist):
      # Rename this var?
      oldname = v.name
      if oldname in vardict:
        newname = vardict[oldname]
        assert isinstance(newname,str)
        varlist[i] = v.rename(newname)
    return Dataset(varlist, atts=self.atts)

  # Remove some variables from the dataset
  def remove (self, *varnames):
    for n in varnames:
      assert isinstance(n,str)
      assert n in self.vardict, "'%s' not found in the dataset"%n
    vars = [v for v in self.vars if v.name not in varnames]
    d = Dataset(vars, atts=self.atts)
    return d
  def __sub__ (self, varnames):
    if isinstance (varnames,(list,tuple)): return self.remove(*varnames)
    return self.remove(varnames)

  # Add some more variables to the dataset
  def add (self, *vars):
    from pygeode.var import Var
    from pygeode.tools import common_dict
    # Collect global attributes (from any Datasets passsed to us)
    atts = [self.atts] + [d.atts for d in vars if isinstance(d,Dataset)]
    atts = common_dict(*atts)
    for v in vars:
      assert isinstance(v,(Var,Dataset)), "'%s' is not a Var"%repr(v)
    # Expand all Datasets to Vars
    vars = [v for v in vars if isinstance(v,Var)] + sum([
           d.vars for d in vars if isinstance(d,Dataset)],[])
    vars = list(self.vars) + list(vars)
    d = Dataset(vars, atts=self.atts)
    return d
  def __add__ (self, vars):
    if isinstance(vars,(list,tuple)): return self.add(*vars)
    return self.add(vars)
  def __radd__ (self, vars): return self.__add__(vars)

  # Replace one or more variables
  def replace_vars (self, vardict={}, **kwargs):
    vardict = dict(vardict, **kwargs)
    varlist = list(self.vars)
    for i, v in enumerate(varlist):
      if v.name in vardict:
        varlist[i] = vardict[v.name]
    return Dataset(varlist, atts=self.atts)

  # Apply the specified var->var function to all variables in the dataset,
  # and make a new dataset.  Anything that gets mapped to 'None' is ignored.
  def map (self, f, *args, **kwargs):
    from pygeode.var import Var
    # Special case: f is a string representing a Var method
    if isinstance(f,str):
      fname = f
      assert hasattr(Var,fname), "unknown function '%s'"%fname
      f = getattr(Var,fname)
      assert hasattr(f,'__call__'), "Var.%s is not a function"%fname
      del fname
    # Allow the function to gracefully fail on vars it can't be applied to.
    if 'ignore_mismatch' in f.func_code.co_varnames:
      kwargs['ignore_mismatch'] = True
    varlist = [f(v, *args, **kwargs) for v in self.vars]
    varlist = [v for v in varlist if v is not None]
    for v in varlist: assert isinstance(v,Var), "%s does not map vars to vars"%f
    return Dataset(varlist, atts=self.atts.copy())


  # Slicing
  # Applies the keyword-based axis slicing to *all* the vars in the dataset.
  #TODO: more efficient method?
  # Right now, each var is sliced independantly, so the same axis will be sliced multiple times.
  def __call__ (self, **kwargs):
    return self.map ('__call__', **kwargs)

# Wrap a variable (or a list of variables) into a dataset
# Use this if you want to make sure something is a dataset, in case it's
#  possible that it's currently a Var list.
def asdataset (vars, copy=False, print_warnings=True):
  ''' asdataset(vars, copy=False, print_warnings=True)
       Tries to make vars into a dataset. If it is a single variable or list of variables, 
       asdataset() returns a Dataset wrapping them. If there are datasets present in the list,
       it merges them into a single dataset. '''
  from copy import copy
  from pygeode.var import Var
  if hasattr(vars, '__len__') and any([isinstance(d, Dataset) for d in vars]):
    d = [d for d in vars if isinstance(d, Dataset)]
    v = [v for v in vars if isinstance(v, Var)]
    assert len(d) + len(v) == len(vars), '%s must consist of vars and datasets only.'
    args = d[1:] + v
    return d[0].add(*args)
  if isinstance(vars,Dataset):
    if not copy: return vars
    dataset = copy(vars)
    #TODO: update if more members are added to this class
    dataset.vars = list(dataset.vars)
    dataset.vardict = dataset.vardict.copy()
    dataset.axes = list(dataset.axes)
    dataset.axisdict = dataset.axisdict.copy()
    dataset.atts = dataset.atts.copy()
    return dataset
  if isinstance(vars, Var): vars = [vars]
  return Dataset(vars, print_warnings=print_warnings)



def axis_name_clumping (varlist):
  # Name-based dictionary pointing to all related axes
  # each name maps to a list of 'distinct' axis lists
  # each of these sublists contains all the equivalent axes
  namedict = {}
  for v in varlist:
    for a in v.axes:
      name = a.name
      if name not in namedict: namedict[name] = []
      eqlist = namedict[name]
      # Check if it's in an existing family
      if any (a is A for e in eqlist for A in e): continue
      # Check if it's comparable to something in a family
      # If so, then append it
      match = False
      for e in eqlist:
        if a == e[0]:
          e.append(a)
          match = True
          break
      # If it's not comparable to anything, start a new family
      if not match: eqlist.append([a])
  return namedict


# Concatenate a bunch of datasets together
def concat(*datasets):
  from pygeode.concat import concat
  from pygeode.tools import common_dict, islist

  # Did we get passed a list of datasets already?
  # (need to break out of the outer list)
  if len(datasets) == 1 and islist(datasets[0]): datasets = list(datasets[0])

  #  If we only have one dataset, then return it
  if len(datasets) == 1: return datasets[0]

  # Collect a list of variable names (in the order they're found in the datasets)
  # and, a corresponding dictionary mapping the names to vars

  varnames = []
  vardict = {}
  for dataset in datasets:
    for v in dataset.vars:
      if v.name not in vardict:
        vardict[v.name] = []
        varnames.append(v.name)
      vardict[v.name].append(v)

  # Merge the var segments together
  # If only one segment, just use the variable itself
  vars = [vardict[n] for n in varnames]
  vars = [concat(v) if len(v)>1 else v[0] for v in vars]

  d = Dataset(vars)

  # Keep any common global attributes
  # Collect all attributes found in the datasets
  atts = common_dict(*[x.atts for x in datasets])
  if len(atts) > 0:
    d.atts = atts
  return d




##################################################
# Hook in Var methods
##################################################

# Wrapper to convert functions from working on Vars to Datasets
def dataset_method (f):
  from pygeode.var import Var
  def new_f (dataset, *args, **kwargs):
    # Degenerate case: passed a single var (return a single var)
    if isinstance(dataset,Var): return f(dataset,*args,**kwargs)
    # Otherwise, apply the function to the vars in the dataset, construct a new dataset
    return dataset.map(f, *args, **kwargs)
  new_f.__name__ = f.__name__
  new_f.__doc__ = f.__doc__
  return new_f

from pygeode.var import class_hooks
for f in class_hooks:
  # Don't use built-in operator methods (__xxx__)
  if f.__name__.startswith('__'): continue
  if f.__name__.endswith('__'): continue
  # Don't override functions already defined here
  if hasattr(Dataset, f.__name__): continue
  setattr(Dataset, f.__name__, dataset_method(f))

del class_hooks, f


