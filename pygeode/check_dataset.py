# Iterates over a dataset, checks for any I/O problems.
# Summarizes any Exceptions caught along the way.
#NOTE: This is not (yet) part of the official PyGeode API, and may be changed/moved/removed at any time!

def check_dataset (dataset):
  from pygeode.view import View
  from pygeode.tools import combine_axes
  from pygeode.progress import PBar
  from pygeode.dataset import asdataset
  import numpy as np

  # Make sure we have a dataset (in case we're sent a simple list of vars)
  dataset = asdataset(dataset)

  vars = list(dataset.vars)

  # Include axes in the list of vars (to check these values too)
  axes = combine_axes(v.axes for v in vars)
  vars.extend(axes)

  # Relative progress of each variable
  sizes = [v.size for v in vars]
  prog = np.cumsum([0.]+sizes) / np.sum(sizes) * 100

  pbar = PBar(message="Checking %s for I/O errors:"%repr(dataset))

  failed_indices = {}
  error_messages = {}

  # Loop over the data
  for i,var in enumerate(vars):

    varpbar = pbar.subset(prog[i], prog[i+1])

    # Scan the outer axis (record axis?) for failures.
    N = var.shape[0]
    failed_indices[var.name] = []
    error_messages[var.name] = []

    for j in range(N):
      vpbar = varpbar.part(j, N)
      try:
        # Try fetching the data, see if something fails
        var[j] if var.naxes == 1 else var[j,...]
      except Exception as e:
        failed_indices[var.name].append(j)
        error_messages[var.name].append(str(e))
      vpbar.update(100)

  # Print summary information for each variable
  everything_ok = True
  for var in vars:
    indices = failed_indices[var.name]
    messages = error_messages[var.name]
    if len(indices) == 0: continue

    everything_ok = False

    print "\nFailures encountered with variable '%s':"%var.name

    # Group together record indices that give the same error message
    unique_messages = []
    aggregated_indices = []
    for ind,msg in zip(indices,messages):
      if len(unique_messages) == 0 or msg != unique_messages[-1]:
        unique_messages.append(msg)
        aggregated_indices.append([ind])
      else:
        aggregated_indices[-1].append(ind)

    # Print each error message encountered (and the record indices that give the error)
    for ind,msg in zip(aggregated_indices,unique_messages):

      # Group records together that have are consecutive (instead of printing each record separately)
      groups = []
      for i in ind:
        if len(groups) == 0 or i-1 not in groups[-1]:
          groups.append([i])
        else:
          groups[-1].append(i)
      for g in groups:
        print "=> at %s:\n    %s"% (var.axes[0].slice[g[0]:g[-1]+1], msg)

  if not everything_ok: raise Exception("Problem encountered with the dataset.")


