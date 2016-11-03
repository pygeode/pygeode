###############################################################################
# Copyright 2016 - Climate Research Division
#                  Environment and Climate Change Canada
#
# This file is part of the "EC-CAS diags" package.
#
# "EC-CAS diags" is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "EC-CAS diags" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with "EC-CAS diags".  If not, see <http://www.gnu.org/licenses/>.
###############################################################################


# Helper method(s) for scanning through data files, constructing a table of
# contents, and sorting them into logical collections.

# Stuff that in the official API for this module
__all__ = ['AxisManager','DataInterface','from_files']

# Current version of the manifest file format.
# If this version doesn't match the existing manifest file, then the manifest
# is re-generated.
_MANIFEST_VERSION="3"

# Interface for creating / reading a manifest file.
class _Manifest(object):

  # Start using a manifest file (and read the existing entries if available).
  def __init__(self, filename=None, axis_manager=None):
    from os.path import exists, getmtime
    import gzip
    import cPickle as pickle

    self.filename = filename

    # If there's already a manifest file on disk, read it.
    if filename is not None and exists(filename):
      with gzip.open(filename,'r') as f:
        version = pickle.load(f)
        table = f.read()
        try:
          self.table = pickle.loads(table)
        except (ImportError, AttributeError, EOFError):
          version = None  # Unable to read the symbols, so treat this as an
                          # incompatible version
      # Use the modification time for the manifest to determine if a data file
      # has been updated since last time we saved the manifest.
      self.mtime = getmtime(filename)
    # If we don't have an existing file, or it's the wrong version, then we
    # start with an empty table.
    if filename is None or not exists(filename) or version != _MANIFEST_VERSION:
      self.table = {}
      self.mtime = 0

    # Collect the axes from the manifest, so we can re-use the objects where
    # possible for new files.
    if axis_manager is not None:
      self.axis_manager = axis_manager
    else:
      self.axis_manager = AxisManager()

    for filename, entries in self.table.iteritems():
      for varname, axes, atts in entries:
        self.axis_manager.register_axes(axes)

    # We haven't done anything new with the table yet.
    self.modified_table = False

    # No data files have been selected yet (even if we have files listed in
    # an existing manifest, we don't yet know if the user wants those
    # particular files included in their query).
    self.selected_files = []

  # Scan through all the given files, add the info to the manifest.
  def scan_files (self, files, opener):
    from os.path import getmtime, normpath
    from pygeode.progress import PBar

    table = self.table

    # Special case: no files given
    if len(files) == 0: return

    self.selected_files.extend(files)

    # Strip out extra separators, etc. from the filenames.
    # Otherwise, if the files are scanned a second time with different
    # separators, it may cause the same file to be included more than once.
    files = [normpath(f) for f in files]

    if self.filename is not None:
      pbar = PBar (message = "Scanning files for %s"%self.filename)
    else:
      pbar = PBar (message = "Scanning files")

    # Construct / add to the table
    for i,f in enumerate(files):
      pbar.update(i*100./len(files))
      if f in table:
        # File has changed since last time?
        if int(getmtime(f)) > self.mtime:
          # Remove existing info
          del table[f]
        else:
          # Otherwise, we've already dealt with the file, so skip it.
          continue
      # Always use the latest modification time to represent the valid time of
      # the whole table.
      self.mtime = max(self.mtime,int(getmtime(f)))

      # Record all variables from the file.
      entries = []
      table[f] = entries
      for var in opener(f):

        axes = self.axis_manager.lookup_axes(var.axes)
        entries.append((var.name, axes, var.atts))

      self.modified_table = True

    pbar.update(100)

  # Get the relevant entries from the manifest (only files that were previously
  # specified in scan_files).
  def get_table (self):
    return dict((f,self.table[f]) for f in self.selected_files)

  # Reset the list of selected files (so we can scan a new batch and produce
  # a new table).
  def unselect_all (self):
    self.selected_files = []

  # Save the data back to disk, if it's been modified.
  # This is called once the manifest is no longer in use.
  def save (self):
    from os.path import exists, getatime, getmtime, normpath
    from os import utime
    import gzip
    import cPickle as pickle
    from pygeode.progress import PBar

    if self.modified_table is True and self.filename is not None:
      with gzip.open(self.filename,'w') as f:
        pickle.dump(_MANIFEST_VERSION, f)
        blob = pickle.dumps(self.table)
        f.write(blob)
      # Set the modification time to the latest file that was used.
      atime = getatime(self.filename)
      utime(self.filename,(atime,self.mtime))

      self.modified_table = False

# A list of variables (acts like an "axis" for the purpose of domain
# aggregating).
class _Varlist (object):
  name = 'varlist'
  def __init__ (self, varnames):
    self.values = tuple(varnames)
  def __iter__ (self):  return iter(self.values)
  def __len__ (self): return len(self.values)
  def __repr__ (self): return "<%s>"%self.__class__.__name__
  def __cmp__ (self, other): return cmp(self.values, other.values)
  _memoized = dict()
  @classmethod
  def singlevar (cls, varname):
    var = cls._memoized.get(varname,None)
    if var is None:
      var = cls._memoized[varname] = cls([varname])
    return var


# An interface for axis-manipulation methods.
# These methods are tied to a common object, which allows the re-use of
# previous values.
class AxisManager (object):
  """
  Keeps track of existing PyGeode.Axis objects.
  Allows you to avoiding having many duplicate Axis objects in your code.

  Example:
  >>> am = AxisManager()
  >>> lat1 = am.lookup_axis(Lat([10.,20.,30.]))
  >>> lat2 = am.lookup_axis(Lat([10.,20.,30.]))
  >>> lat1 is lat2
  True
  """
  def __init__ (self):
    self._hash_bins = {}  # Bin axis objects by hash value
    self._id_lookup = {}  # Reverse-lookup of an axis by id
    self._all_axes = []   # List of encountered axes (so the ids don't get
                          # recycled)

    # For axes that are converted to/from a set
    self._settified_axes = {}
    self._unsettified_axes = {}

    # For union / intersection of axes
    self._unions = {}
    self._intersections = {}

  # Helper function: recycle an existing axis object if possible.
  # This allows axes to be compared by their ids, and makes pickling them
  # more space-efficient.
  def lookup_axis (self, axis):
    """
    If the axis isn't registered yet, then register it and return it.
    If the axis is already registered, or an identical axis is already
    registered, then return a reference to that original axis.
    """
    # Check if we've already looked at this exact object.
    axis_id = id(axis)
    entry = self._id_lookup.get(axis_id,None)
    if entry is not None: return entry
    # Store a reference to this axis, so the object id doesn't get recycled.
    self._all_axes.append(axis)
    values = tuple(axis.values)
    # Get a hash value that will be equal among axes that are equivalent
    axis_hash = hash((axis.name,type(axis),values))
    # Get all axes that have this hash (most likely, only 1 match (or none))
    hash_bin = self._hash_bins.setdefault(axis_hash,[])
    # Find one that is truly equivalent, otherwise add this new one to the cache.
    try:
      axis = hash_bin[hash_bin.index(axis)]
    except ValueError:
      hash_bin.append(axis)
    # Record this object id, in case we're passed it in again.
    self._id_lookup[axis_id] = axis

    return axis

  # Register an axis in this object (so we're aware of it for future reference)
  def register_axis (self, axis):
    """
    Make the axis manager aware of the given axis.
    """
    self.lookup_axis (axis)

  # Look up multiple axes at a time, return as a tuple
  def lookup_axes (self, axes):
    """
    Looks up multiple axes at a time.
    """
    return tuple(map(self.lookup_axis, axes))

  # Register multiple axes at a time
  def register_axes (self, axes):
    """
    Registers multiple axes at a time.
    """
    self.lookup_axes (axes)

  # Convert an axis to unordered set of tuples
  def _settify_axis (self, axis):
    from pygeode.timeaxis import Time
    axis = self.lookup_axis(axis)
    axis_id = id(axis)
    entry = self._settified_axes.get(axis_id,None)
    if entry is not None: return entry

    # Varlist objects have no aux arrays, and we don't *need* the aux arrays
    # for time axes (can reconstruct this information later).
    if isinstance(axis,(_Varlist,Time)):
      auxarrays = []
    else:
      auxarrays = [[(name,v) for v in axis.auxarrays[name]] for name in sorted(axis.auxarrays.keys())]
    assert all(len(aux) == len(axis.values) for aux in auxarrays)

    # If there are aux arrays, need to pair the elements in the flatteded
    # version.
    if len(auxarrays) > 0:
      flat = zip(axis.values, *auxarrays)
    # Otherwise, just need the values themselves.
    else:
      flat = axis.values

    out = frozenset(flat)
    self._settified_axes[axis_id] = out
# disabled this - otherwise we get the original (unsorted) axis where we may
# expect a sorted axis. (e.g. in DataVar)
#    self._unsettified_axes.setdefault(type(axis),dict())[out] = axis
    return out

  # Convert some settified coordinates back into an axis
  def _unsettify_axis (self, sample, values):
    import numpy as np
    # Check if we can already get one
    key = values
    axis = self._unsettified_axes.setdefault(type(sample),dict()).get(key,None)
    if axis is not None: return axis

    values = sorted(values)
    # Detect reverse-ordered axes
    if len(sample) > 1 and sample.values[0] > sample.values[1]:
      values = values[::-1]

    # Check if the axis is degenerate
    if len(values) == 0:
      axis = sample.withnewvalues(values)
    # Do we have aux array pairs to deal with?
    elif isinstance(values[0],tuple):
      x = zip(*values)
      values = x[0]
      auxarrays = {}
      for aux in x[1:]:
        name, arr = zip(*aux)
        auxarrays[name[0]] = np.array(arr)
      axis = sample.withnewvalues(values)
      # Only update the auxarrays if we have something to put
      # For empty axes, we don't want to erase the (empty) auxarrays already
      # created e.g. for the time axis.
      if len(auxarrays) > 0: axis.auxarrays = auxarrays

    # Do we have a Varlist pseudo-axis?
    elif isinstance(sample,_Varlist):
      axis = _Varlist(values)

    # Otherwise, we have an axis with no aux arrays (so we can just use the
    # values we have).
    else:
      axis = sample.withnewvalues(values)

    axis = self.lookup_axis(axis)

    self._unsettified_axes[type(sample)][key] = axis
    return axis

  # Find common values between axes
  def _get_axis_intersection (self, axes):
    key = tuple(sorted(map(id,axes)))
    if key in self._intersections: return self._intersections[key]
    values = map(self._settify_axis, axes)
    values = reduce(frozenset.intersection, values, values[0])
    intersection = self._unsettify_axis (axes[0], values)
    if len(intersection) > 0:
      self._intersections[key] = intersection
    return intersection


# A domain (essentially a tuple of axes, with no deep comparisons)
class _Domain (object):
  def __init__ (self, axis_samples, axis_values):
    # Sample axis objects, for reconstructing axes of these types.
    # (may not contain the actual data that will be reconstructed).
    self.axis_samples = tuple(axis_samples)
    self.axis_names = tuple([a.name for a in axis_samples])
    # Store the axis values as sets, to make unions/intersections faster.
    self.axis_values = tuple(axis_values)
  def __cmp__ (self, other):
    key1 = (self.axis_names, self.axis_values)
    key2 = (other.axis_names, other.axis_values)
    return cmp(key1, key2)
  def __hash__ (self):
    return hash(self.axis_values)
  def __repr__ (self):
    return "("+",".join(map(str,map(len,filter(None,self.axis_values))))+")"
  def which_axis (self, iaxis):
    if isinstance(iaxis,int): return iaxis
    assert isinstance(iaxis,str)
    if iaxis in self.axis_names:
      return self.axis_names.index(iaxis)
    return None
  # Mask out an axis (convert it to a 'None' placeholder)
  def without_axis (self, iaxis):
    axis_values = list(self.axis_values)
    axis_values[self.which_axis(iaxis)] = None
    axis_values = tuple(axis_values)
    return type(self)(self.axis_samples, axis_values)
  # Unmask an axis type (re-insert an axis object where the 'None' placeholder was
  def with_axis (self, iaxis, values):
    assert isinstance(values,frozenset)
    axis_values = list(self.axis_values)
    axis_values[self.which_axis(iaxis)] = values
    axis_values = tuple(axis_values)
    return type(self)(self.axis_samples, axis_values)
  # Reconstructs the axes from the samples and values. 
  def make_axes (self, axis_manager):
    return [axis_manager._unsettify_axis(s,v) for (s,v) in zip(self.axis_samples, self.axis_values)]
  # Determine if the given axis is in this domain (given its name)
  def has_axis (self, axis_name):
    return axis_name in self.axis_names
  # Return the (unordered) values of a particular axis.
  def get_axis_values (self, iaxis):
    return self.axis_values[self.which_axis(iaxis)]




# Helper method - return all names of axes in a set of domains
# Returned in approximate order that they're found in the domains.
def _get_axis_names (domains):
  ordered_names = set()
  for domain in domains:
    ordered_names.update(enumerate(domain.axis_names))
  names = []
  for i,name in sorted(ordered_names,reverse=True):
    if name not in names:
      names.append(name)
  names = list(reversed(names))
  # Special case: time axis should be handled first (makes aggregating faster)
  if 'time' in names:
    names.remove('time')
    names = ['time'] + names
  return tuple(names)


# Helper method - aggregate along a particular axis
# Inputs:
#   domains: the set of original domains.
#   axis_name: the name of axis to aggregate the domains over.
# Input/Output:
#   used_domains: set of original domains that are covered by the output
#                 (i.e., ones that could be safely removed later).
#                 These are appended to an existing set passed in.
# Output: the domains that could be aggregated along to given axis.
#         Domains without that axis type are ignored.
def _aggregate_along_axis (domains, axis_name):
  output = set()
  bins = {}
  for domain in domains:
    iaxis = domain.which_axis(axis_name)
    if iaxis is None:
      output.add(domain)
      continue
    domain_group = domain.without_axis(iaxis)
    axis_bin = bins.setdefault(domain_group,set())
    axis_bin.add(domain.axis_values[iaxis])
  # For each domain group, aggregate the axes together
  # NOTE: assumes that all the axis segments are consistent
  # (same origin, units, etc.)
  # Also, assumes the axis values should be monotonically increasing.
  for domain_group, axis_bin in bins.iteritems():
    if len(axis_bin) == 1:  # Only one axis piece (nothing to aggregate)
      axis_values = axis_bin.pop()
    # Otherwise, need to aggregate pieces together.
    else:
      axis_values = frozenset.union(*axis_bin)
    output.add(domain_group.with_axis(axis_name,axis_values))

  return output


# Find a minimal set of domains that cover all available data
def _get_prime_domains (domains):
  axis_names = _get_axis_names(domains)
  # Aggregate along one axis at a time.
  for axis_name in axis_names:
    domains = _aggregate_along_axis(domains, axis_name)

  return domains

# Try merging multiple domains together.
# For each pair of domains, look for an axis over which they could be
# concatenated.  All other axes will be intersected between domains.
def _merge_domains (d1, d2):
  from copy import copy
  domains = set()
  axis_names = _get_axis_names([d1,d2])
  # We need at least one of the two domains to contain all the types
  # (so we have a deterministic ordering of axes).
  # Make the first domain the one with all axes.
  #TODO: Check the relative order of the axes as well?
  if _get_axis_names([d1]) == axis_names:
    pass  # Already done
  elif _get_axis_names([d2]) == axis_names:
    d1, d2 = d2, d1  # Swap
  else:
    return set()  # Nothing can be done
  del axis_names

  # Pre-compute union and intersection of the axes.
  # Determine which axes may be useful to merge over.
  # Early termination if 2 or more axes are non-intersectable
  intersections = []
  unions = []
  merge_axes = []
  non_intersectable = 0
  for iaxis,v1 in enumerate(d1.axis_values):
    axis_name = d1.axis_samples[iaxis].name
    if d2.has_axis(axis_name):
      v2 = d2.get_axis_values(axis_name)
    else:
      v2 = v1
    if v1 is v2:
      intersection = v1
      union = v2
    else:
      intersection = v1 & v2
      union = v1 | v2
    # Would we get anything useful from merging this axis?
    # (note: other conditions would need to be met as well)
    if len(union) > len(v1) and len(union) > len(v2):
      merge_axes.append(iaxis)
    # Check for non-overlapping axes (can have up to 1 such dimension).
    if len(intersection) == 0:
      non_intersectable += 1
    # Store the union / intersection for use later.
    intersections.append(intersection)
    unions.append(union)
  if non_intersectable > 1:
    return set()  # We will alway have an empty domain after a merge
                  # where multiple dimensions have no overlap.


  # Test each axis to see if it can be a merge axis
  # (if the other axes have non-zero intersection).
  for iaxis in merge_axes:
    axis_values = list(intersections)
    axis_values[iaxis] = unions[iaxis]
    # Skip if we don't have any overlap
    if any(len(v) == 0 for v in axis_values): continue
    domain = copy(d1)
    domain.axis_values = tuple(axis_values)
    domains.add(domain)

  return domains

def _merge_all_domains (domains):
  merged_domains = set(domains)
  while True:
    new_merged_domains = set()
    for d1 in domains:
      for d2 in merged_domains:
        if d1 is d2: continue
        new_merged_domains.update(_merge_domains(d1,d2))
    new_merged_domains -= merged_domains
    if len(new_merged_domains) == 0: break  # Nothing new added
    merged_domains.update(new_merged_domains)

  return domains | merged_domains

# Remove any domains that are proper subsets of other domains
# (Clean up anything that is not needed).
def _cleanup_subdomains (domains):
  junk_domains = set()
  for d1 in domains:
    for d2 in domains:
      if d1 is d2: continue
      assert d1 != d2
      if _get_axis_names([d1]) != _get_axis_names([d2]): continue
      axis_names = _get_axis_names([d2])
      values1 = [d1.get_axis_values(a) for a in axis_names]
      values2 = [d2.get_axis_values(a) for a in axis_names]
      if all(v1 <= v2 for v1, v2 in zip(values1,values2)):
        junk_domains.add(d1)
  return domains - junk_domains



# Scan a file manifest, return all possible domains available.
def _get_domains (manifest, axis_manager):

  # Start by adding all domain pieces to the list
  domains = set()
  for entries in manifest.itervalues():
    for var, axes, atts in entries:
      # Map each entry to a domain.
      axes = (_Varlist.singlevar(var),)+axes
      axis_values = map(axis_manager._settify_axis, axes)
      domains.add(_Domain(axis_samples=axes, axis_values=axis_values))

  # Reduce this to a minimal number of domains for data coverage
  domains = _get_prime_domains(domains)
  # Try merging domains together in different ways to get different coverage.
  # Continue until we found all the unique combinations.
  while True:
    old_ndomains = len(domains)
    domains = _merge_all_domains(domains)
    domains = _cleanup_subdomains(domains)
    if len(domains) == old_ndomains: break
  return domains


# Extract variable attributes and table of files from the given manifest.
def _get_var_info(manifest,opener):
  from pygeode.tools import common_dict
  atts = dict()
  table = dict()
  for filename, entries in manifest.iteritems():
    for _varname, _axes, _atts in entries:
      _attslist = atts.setdefault(_varname,[])
      if _atts not in _attslist: _attslist.append(_atts)
      table.setdefault(_varname,[]).append((filename, opener, _axes))
  atts = dict((_varname,common_dict(_attslist)) for (_varname,_attslist) in atts.iteritems())
  return atts, table

# Find all datasets that can be constructed from a set of files.
def from_files (filelist, interface, manifest=None, save_manifest=True, opener_args={}):
  """
  Scans the given files using the specified interface.  Determines all the
  different ways the data can be mixed and matched, and returns a list of
  PyGeode.Dataset objects, one for each combination of variables / axes.

  Inputs:

    filelist:  Can be either an explicit list of files, or a glob expression.

    interface: How to read the data (either a format name, PyGeode format
               module, or an explicit function for reading a file.

    manifest: The name of a temporary file for storing the information about the
              files (variable names, axes, attributes).  On subsequent calls,
              this information can be re-used, so the data files don't have to
              be re-scanned.
              NOTE: If you change the interface in between calls, then you
              should delete this file before re-running.

    opener_args: Any extra arguments to pass to the file interface.

  """

  # Check if we're given a single glob expression
  # (evaluate to a list of files).
  if isinstance(filelist,str):
    from glob import glob
    filelist = glob(filelist)

  # Determine what kind of parameter was passed for the 'interface', and
  # find a file opener from this.
  # Format name passed as a string?
  if isinstance(interface,str):
    import importlib
    interface = importlib.import_module('pygeode.formats.'+interface)
  # PyGeode format object?
  if hasattr(interface,'open'):
    interface = interface.open
  # EC-CAS diags interface object?
  if hasattr(interface,'open_file'):
    interface = interface.open_file
  if not hasattr(interface,'__call__'):
    raise TypeError("Unable to determine the type of interface provided.")

  # Wrap any extra args into the opener
  opener = lambda filename: interface(filename,**opener_args)

  # If we're given a filename, then wrap it in a Manifest object.
  # If we're not given any filename, then create a new Manifest with no file
  # association.
  if isinstance(manifest,str) or manifest is None:
    manifest = _Manifest(filename=manifest)

  axis_manager = manifest.axis_manager
  # Scan the given data files, and add them to the table.
  manifest.scan_files(filelist, opener)
  if save_manifest: manifest.save()
  # Get the final table of available data.
  table = manifest.get_table()
  # Done with these files.
  manifest.unselect_all()

  domains = _get_domains(table, axis_manager)
  # Find all variable attributes that are consistent throughout all files.
  # Also, invert the table so the lookup key is the varname (value is a list
  # of all filenames that contain it).
  atts, table = _get_var_info(table,opener)
  datasets = [_domain_as_dataset(d,atts,table,axis_manager) for d in domains]
  return datasets

# Wrap a domain as a dataset.
def _domain_as_dataset (domain, atts, table, axis_manager):
  from pygeode.dataset import Dataset
  axes = domain.make_axes(axis_manager)
  ivarlist = domain.which_axis('varlist')
  assert ivarlist is not None, "Unable to determine variable names"
  varlist = axes[ivarlist]
  axes = axes[:ivarlist] + axes[ivarlist+1:]
  return Dataset([_DataVar.construct(name, axes, atts[name], table[name], axis_manager) for name in varlist])



# Wrap a variable from a domain into a Var object
from pygeode.var import Var
class _DataVar(Var):
  @classmethod
  def construct (cls, varname, axes, atts, table, axis_manager):

    # Reduce the axes to only those that the variable actually has
    axis_names = set(a.name for f,o,_axes in table for a in _axes)
    axes = [a for a in axes if a.name in axis_names]
    axis_manager.register_axes(axes)

    obj = cls(axes, name=varname, dtype=float, atts=atts)
    obj._table = table
    obj._axis_manager = axis_manager
    obj._varname = varname  # In case the object gets renamed.

    return obj

  def getview (self, view, pbar):

    import numpy as np
    from pygeode.view import View, simplify
    out = np.empty(view.shape, dtype=self.dtype)
    out[()] = float('nan')
    out_axes = view.clip().axes
    # Loop over all available files.
    N = 0  # Number of points covered so far
    for filename, opener, axes in self._table:
      subaxes = [self._axis_manager._get_axis_intersection([a1,a2]) for a1,a2 in zip(out_axes,axes)]
      reorder = []
      mask = []
      if any(len(a)==0 for a in subaxes): continue
      for a1,a2 in zip(out_axes,subaxes):
        # Figure out where the input chunk fits into the output
        re = np.searchsorted(a2.values, a1.values)
        # Mask out elements that we don't actually have in the chunk
        m = [r<len(a2.values) and a2.values[r]==v for r,v in zip(re,a1.values)]
        m = np.array(m)
        # Convert mask to integer indices
        m = np.arange(len(m))[m]
        # and then to a slice (where possible)
        m = simplify(m)
        re = re[m]
        # Try to simplify the re-ordering array
        if np.all(re == np.sort(re)):
          re = simplify(re)
        reorder.append(re)
        mask.append(m)
      var = [v for v in opener(filename) if v.name == self._varname][0]
      v = View(subaxes)
      chunk = v.get(var)
      # Note: this may break if there is more than one axis with integer indices.
      assert len([r for r in reorder if isinstance(r,(tuple,np.ndarray))]) <= 1, "Unhandled advanced indexing case."
      assert len([m for m in mask if isinstance(m,(tuple,np.ndarray))]) <= 1, "Unhandled advanced indexing case."
      out[mask] = chunk[reorder]
      N = N + chunk.size
      pbar.update(100.*N/out.size)

    return out

del Var


# A generic data interface.
# Essentially, a collection of datasets, with some convenience methods.
class DataInterface (object):
  """
  Wraps a list of PyGeode.Dataset objects in a higher-level interface for
  querying the data.
  """
  # Generic initializer - takes a list of Datasets, stores it.
  def __init__ (self, datasets):
    from pygeode.dataset import asdataset
    self.datasets = tuple(map(asdataset,datasets))

  # Allow the underlying datasets to be iterated over
  def __iter__ (self):
    return iter(self.datasets)


  # Get the requested variable(s).
  # The possible matches are returned one at a time, and the calling method
  # will have to figure out which one is the best.
  def find (self, *vars, **kwargs):
    """
    Iterates over all datasets, looking for the given variable(s).

    Inputs:
      vars: Either a single variable name, or a list of variable names to
            look for concurrently.

      requirement: A boolean function that returns True or False for a given
                   dataset.  Useful if you have some extra criteria that the
                   data must satisfy.  (optional)

    Returns: an iterator over all matching copies of the variable(s).
    """
    requirement=kwargs.pop('requirement',None)
    if len(kwargs) > 0:
      raise TypeError("Unexpected keyword arguments: %s"%kwargs.keys())

    for dataset in self.datasets:
      # Check if this dataset meets any extra requirements
      if requirement is not None:
        if not requirement(dataset):
          continue
      # Check if all the variables are in the dataset
      if all(v in dataset for v in vars):
        varlist = [dataset[v] for v in vars]
        if len(varlist) == 1: yield varlist[0]
        else: yield varlist

  # Determine if a given variable is in the data somewhere
  def have (self, var):
    """
    Checks if the specified variable is available in the datasets.
    """
    for dataset in self.datasets:
      if var in dataset: return True
    return False

  # Helper function - find the best field matches that fit some criteria
  def find_best (self, varnames, requirement=None, maximize=None, minimize=None):
    """
    Returns the "best" match for the specified variable name(s).

    Inputs:
      varnames: Either a single variable name, or a list of variable names to
                match concurrently.

      requirement: A boolean function that returns True or False for a given
                   dataset.  Useful if you have some extra criteria that the
                   data must satisfy.  (optional)

      maximize / minimize: A function (or list of functions) for ranking
                           a dataset.  This ranking value will be used for
                           determining the "best" match.

    Example:  Find CO2 and temperature fields on concurrent time steps, with
              the additional requirement that we have surface-level data
              (assuming a pressure axis):

    >>> have_surface = lambda dataset: 1000.0 in dataset.pres.values
    >>> co2, temp = mydata.find_best(['CO2','Temperature'], requirement=have_surface)

    Example: Find temperature data with as much vertical extent as possible:

    >>> number_of_levels = lambda dataset: len(dataset.zaxis)
    >>> temp = mydata.find_best('Temperature', maximize=number_of_levels)

    """

    # If we are given a single field name (not in a list), then return a
    # single field (also not in a list structure).
    collapse_result = False
    if isinstance(varnames,str):
      varnames = [varnames]
      collapse_result = True

    if len(varnames) == 1:
      candidates = zip(self.find(*varnames,requirement=requirement))
    else:
      candidates = list(self.find(*varnames,requirement=requirement))

    # At the very least, order by domain shapes
    # (so we never have an arbitrary order of matches)
    def domain_size (varlist):
      return sorted((v.name,v.shape) for v in varlist)
    candidates = sorted(candidates, key=domain_size, reverse=True)

    if isinstance(maximize,tuple):
      # Will be sorted by increasing order, so need to reverse the cost
      # functions here.
      maximize = lambda x,F=maximize: [-f(x) for f in F]
    elif maximize is not None:
      # Always need to invert the sign to get the maximum value first.
      maximize = lambda x,f=maximize: -f(x)

    if isinstance(minimize,tuple):
      minimize = lambda x,F=minimize: [f(x) for f in F]


    # Sort by the criteria (higher value is better)
    if maximize is not None:
      candidates = sorted(candidates, key=maximize)
    elif minimize is not None:
      candidates = sorted(candidates, key=minimize)

    if len(candidates) == 0:
      raise KeyError("Unable to find any matches for varnames=%s, requirement=%s, maximize=%s, minimize=%s"%(varnames, requirement, maximize, minimize))

    # Use the best result
    result = candidates[0]

    if collapse_result: result = result[0]
    return result


