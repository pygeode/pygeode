#TODO: allow variables to be omitted in some ensembles

from pygeode.axis import Index
class Ensemble(Index): pass
del Index

def make_ensemble (n, ensdict={}):
  if n not in ensdict: ensdict[n] = Ensemble(n)
  return ensdict[n]

from pygeode.var import Var
class EnsembleVar(Var):
  def __init__(self, varlist):
    from pygeode.var import Var, combine_meta
    self.varlist = varlist
    # assume the vars have already been checked for consistency
    var0 = varlist[0]
    axes = list(var0.axes)
#    axes = [Ensemble(len(varlist))] + axes
    axes = [make_ensemble(len(varlist))] + axes
    Var.__init__(self, axes, dtype=var0.dtype)
#    copy_meta (var0, self)
#    self.atts = common_dict(var.atts for var in varlist)
#    self.plotatts = common_dict(var.plotatts for var in varlist)
    combine_meta (varlist, self)
    self.name = varlist[0].name
  def getview (self, view, pbar):
    import numpy as np
    subview = view.remove(0)
    N = len(view.integer_indices[0])
    chunks = [[subview.get(self.varlist[i], pbar=pbar.part(n,N))]
              for n,i in enumerate(view.integer_indices[0])]
    return np.concatenate(chunks, axis=0)

del Var

# Collect vars into an ensemble

def ensemble (*varlists):
  """
  Creates an ensemble out of a set of similar variables.
  The corresponding variable must have the same axes and the same name.
  If a bunch of vars are passed as inputs, then a single ensemble var is returned.
  If a bunch of datasets are passed as inputs, then a single dataset is returned, consisting of an ensemble of the internal vars.  Each input dataset must have matching vars.
  """
  from pygeode.var import Var
  from pygeode.dataset import Dataset, asdataset
  from pygeode.tools import common_dict
  datasets = [asdataset(v) for v in varlists]

  varnames = [v.name for v in datasets[0].vars]

  # Make sure we have the same varnames in each dataset
  for dataset in datasets: assert set(dataset.vardict.keys()) == set(varnames), "inconsistent variable names between datasets"

  # Make sure the varlists are all in the same order
  for i, dataset in enumerate(datasets):
    varlist = [dataset[varname] for varname in varnames]
    datasets[i] = Dataset(varlist, atts=dataset.atts)

  for varname in varnames:
    var0 = datasets[0][varname]
    for dataset in datasets:
      var = dataset[varname]
      # Make sure the axes are the same between ensemble vars
      assert var.axes == var0.axes, "inconsistent axes for %s"%varname

  # Collect the ensembles together
  ensembles = []
  for varname in varnames:
    ensemble = EnsembleVar([dataset[varname] for dataset in datasets])
    ensembles.append(ensemble)

  # Global attributes
  atts = common_dict(dataset.atts for dataset in datasets)
  if isinstance(varlists[0], Dataset): return Dataset(ensembles, atts=atts)
  if isinstance(varlists[0], Var):
    assert len(ensembles) == 1
    return ensembles[0]

  return ensembles
