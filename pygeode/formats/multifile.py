#TODO: a simple function interface that takes as arguments a format,
#     a set of filenames, and a date format (i.e., $Y$m$d) to extract
#     from the file names.

# Expand globbed file expressions to a flat list of files
def expand_file_list (file_list, sort=True):
#  from glob import iglob
  from glob import iglob
  from os.path import exists
  if not isinstance(file_list, (list,tuple)): file_list = [file_list]
  files = [ f for file_glob in file_list for f in iglob(file_glob)]
  assert len(files) > 0, 'No matches found'
  for f in files: assert exists(f), str(f)+" doesn't exist"
  if sort: files.sort()
  return files


# Interface for loading multiple files at a time (file a file glob)
# Inputs: the format, the glob, and any additional keywords to pass
# I.e.: openall(files = "file???.nc", format = netcdf)
#NOTE: for a large number of homogeneous files, use the alternative interface below
def openall (files, format=None, opener=None, **kwargs):
  ''' Returns a :class:`Dataset` containing variables merged across multiple files.

  Parameters
  ==========
  files : string, list, or tuple
    Either a single filename or a list of filenames. Wildcards are supported, :func:`glob.iglob` is
    used to expand these into an explicit list of files.

  format : string, optional
    String specifying format of file to open. If none is given the format will be automatically
    detected from the first filename (see :func:`autodetectformat`)

  opener : function, optional
    Function to open individual files. If none is provided, uses the
    format-specific version of :func:`open`. The datasets returned by this
    function are then concatenated and returned. See Notes.

  sorted : boolean, optional
    If True, the filenames are sorted (by alpha) prior to opening each file, and
    the axes on the returned dataset are sorted by calling :meth:`Dataset.sorted`.

  **kwargs : keyword arguments
    These are passed on to the function ``opener``;

  Returns
  =======
  dataset
    A dataset containing the variables concatenated across all specified files.
    The variable data itself is not loaded into memory. 

  Notes
  =====
  The function ``opener`` must take a single positional argument - the filename of the file
  to open - and keyword arguments that are passed through from this function. It must return
  a :class:`Dataset` object with the loaded variables. By default the standard
  :func:`open` is used, but providing a custom opener can be useful for any reshaping of the 
  variables that must be done prior to concatenating the whole dataset. 

  Once every file has been opened, the resulting datasets are concatenated
  using :func:`dataset.concat`. 
  
  This function is best suited for a moderate number of files. Because each
  file must be explicitly opened to read the metadata, even this can take a
  significant amount of time if a large number of files are being opened. For
  these cases using :func:`open_multi` can be much more efficient, though it
  requires more coding effort initially. The underlying concatenation is also
  more efficient when the data is actually accessed. 

  See Also
  ========
  open
  open_multi
  '''
  from pygeode.dataset import concat
  from pygeode.formats import autodetectformat

  sort = kwargs.pop('sorted', True)
  files = expand_file_list (files, sort)

  if opener is None:

    if format is None: format = autodetectformat(files[0])

    if not hasattr(format, 'open'):
      try:
        format = __import__("pygeode.formats.%s" % format, fromlist=["pygeode.formats"])
      except ImportError:
        raise ValueError('Unrecognized format module %s.' % format)

    opener = format.open
  
  datasets = [ opener(f, **kwargs) for f in files]

  ds = concat(*[d for d in datasets if d is not None])
  if sort: ds = ds.sorted()
  return ds

def open_multi (files, format=None, opener=None, pattern=None, file2date=None, **kwargs):
# {{{
  ''' Returns a :class:`Dataset` containing variables merged across many files.

  Parameters
  ==========
  files : string, list, or tuple
    Either a single filename or a list of filenames. Wildcards are supported, :func:`glob.iglob` is
    used to expand these into an explicit list of files.

  format : string, optional
    String specifying format of file to open. If none is given the format will be automatically
    detected from the first filename (see :func:`autodetectformat`)

  opener : function, optional
    Function to open individual files. If none is provided, uses the
    format-specific version of :func:`open`. The datasets returned by this
    function are then concatenated and returned. See Notes.

  pattern : string, optional
    A regex pattern to extract date stamps from the filename; used by default file2date.
    Matching patterns must be named <year>, <month>, <day>, <hour> or <minute>.
    Abbreviations are available for the above; $Y matches a four digit year, $m, $d, $H,
    and $M match a two-digit month, day, hour and minute, respectively.

  file2date : function, optional
    Function which returns a date dictionary given a filename. By default this is produced
    by applying the regex pattern ``pattern`` to the filename.

  sorted : boolean, optional
    If True, the filenames are sorted (by alpha) prior to opening each file, and
    the axes on the returned dataset are sorted by calling :meth:`Dataset.sorted`.

  **kwargs : keyword arguments
    These are passed on to the function ``opener``;

  Returns
  =======
  dataset
    A dataset containing the variables concatenated across all specified files.
    The variable data itself is not loaded into memory. 

  Notes
  =====
  This is intended to provide access to large datasets whose files are
  separated by timestep.  To avoid opening every file individually, the time
  axis is constructed by opening the first and the last file in the list of
  files provided. This is done to provide a template of what variables and what
  times are stored in each file - it is assumed that the number of timesteps
  (and their offsets) is the same accross the whole dataset. The time axis is
  then constructed from the filenames themselves, using the function
  ``file2date`` to generate a date from each filename. As a result only two files
  need to be opened, which makes this a very efficient way to work with very large
  datasets.

  However, no explicit check is made of the integrity of the files - if there
  are corrupt or missing data within individual files, this will not become
  clear until that data is actually accessed. This can be done explicitly with
  :func:`check_multi`, which explicitly attempts to access all the data and
  returns a list of any problems encountered; this can take a long time, but is
  a useful check (and is more likely to provide helpful error messages). 

  The function ``opener`` must take a single positional argument - the filename
  of the file to open - and keyword arguments that are passed through from this
  function. It must return a :class:`Dataset` object with the loaded variables.
  By default the standard :func:`open` is used, but providing a custom opener
  can be useful for any reshaping of the variables that must be done prior to
  concatenating the whole dataset. 

  See Also
  ========
  open
  openall
  '''

  from pygeode.timeaxis import Time, StandardTime
  from pygeode.timeutils import reltime, delta
  from pygeode.dataset import Dataset
  from pygeode.tools import common_dict
  from pygeode.formats import open, autodetectformat
  import numpy as np

  files = expand_file_list(files)
  nfiles = len(files)
  assert nfiles > 0

  if opener is None: 
    if format is None: format = autodetectformat(files[0])

    if not hasattr(format, 'open'): 
      try:
        format = __import__("pygeode.formats.%s" % format, fromlist=["pygeode.formats"])
      except ImportError:
        raise ValueError('Unrecognized format module %s.' % format)

    opener = format.open

  # Apply keyword arguments
  if len(kwargs) > 0:
    old_opener = opener
    opener = lambda f: old_opener (f, **kwargs)


  # Degenerate case: only one file was given
  if nfiles == 1: return opener(files[0])


  # We'll need a function to translate filenames to dates
  # (if we don't have one, use the supplied pattern to make one)
  if file2date is None:
    import re
    assert pattern is not None, "I don't know how to get the dates from the filenames"
    regex = pattern
    regex = regex.replace('$Y', '(?P<year>[0-9]{4})')
    regex = regex.replace('$m', '(?P<month>[0-9]{2})')
    regex = regex.replace('$d', '(?P<day>[0-9]{2})')
    regex = regex.replace('$H', '(?P<hour>[0-9]{2})')
    regex = regex.replace('$M', '(?P<minute>[0-9]{2})')
    regex = re.compile(regex)
    def file2date (f):
      d = regex.search(f)
      assert d is not None, "can't use the pattern on the filenames?"
      d = d.groupdict()
      d = dict([k,int(v)] for k,v in d.iteritems() if v is not None)
      # Apply default values (i.e. for minutes, seconds if they're not in the file format?)
      d = dict({'hour':0, 'minute':0,'second':0}, **d)
      return d


  # Get the starting date of each file
  dates = [file2date(f) for f in files]
  dates = dict((k,[d[k] for d in dates]) for k in dates[0].keys())

  # Open a file to get a time axis
  file = opener(files[0])
  T = None
  for v in file.vars:
    if v.hasaxis(Time):
      T = type(v.getaxis(Time))
      break
  if T is None: T = StandardTime
#  T = [v.getaxis(Time) for v in file.vars if v.hasaxis(Time)]
#  T = type(T[0]) if len(T) > 0 else StandardTime
  del file

  # Generate a lower-resolution time axis (the start of *each* file)
  faxis = T(units='days',**dates)

  # Re-sort the files, if they weren't in order
  S = faxis.argsort()
  faxis = faxis.slice[S]
  files = [files[s] for s in S]
  # Re-init the faxis to force the proper start date
  faxis = type(faxis)(units=faxis.units, **faxis.auxarrays)

  # Open the first and last file, so we know what the variables & timesteps are
  first = opener(files[0])
  last  = opener(files[-1])
  names = [v.name for v in first.vars]
  for n in names: assert n in last, "inconsistent vars"
  # Get global attributes
  global_atts = common_dict (first.atts, last.atts)

  #---
  timedict = {None:faxis}
  for v1 in first:
    if not v1.hasaxis(Time): continue
    t1 = v1.getaxis(Time)
    if t1.name in timedict: continue  # already handled this one
    t2 = last[v1.name].getaxis(Time)
    # Construct a full time axis from these pieces

    # One timestep per file? (check for an offset for the var time compared
    #  to the file time)
    if max(len(t1),len(t2)) == 1:
      offset = reltime(t1, startdate=faxis.startdate, units=faxis.units)[0]
      taxis = faxis.withnewvalues(faxis.values + offset)
    # At least one of first/last files has multiple timesteps?
    else:
      assert t1.units == t2.units
      dt = max(delta(t1),delta(t2))
      assert dt > 0
      val1 = t1.values[0]
      val2 = reltime(t2, startdate=t1.startdate)[-1]
      nt = (val2-val1)/dt + 1
      assert round(nt) == nt
      nt = int(round(nt))
      assert nt > 0
      taxis = t1.withnewvalues(np.arange(nt)*dt + val1)

    timedict[t1.name] = taxis

  #---

  # Create the multifile version of the vars
  vars = [Multifile_Var(v1, opener, files, faxis, timedict) for v1 in first]


  return Dataset(vars,atts=global_atts)
# }}} 

from pygeode.var import Var
class Multifile_Var (Var):
# {{{
  def __init__ (self, v1, opener, files, faxis, timedict):
# {{{
    # v1 - var chunk from the first file
    # opener - method for opening a file given a filename
    # files - list of filenames
    # faxis - starting time of each file, as a time axis
    # timedict - dictionary of pre-constructed time axes
    from pygeode.var import Var
    from pygeode.timeaxis import Time

    self.opener = opener
    self.files = files
    self.faxis = faxis

    n = v1.getaxis(Time).name if v1.hasaxis(Time) else None
    taxis = timedict[n]
    T = type(taxis)

    axes = list(v1.axes)

    # Replace the existing time axis?
    # (look for either a Time axis or a generic axis with the name 'time')
    if v1.hasaxis(T): axes[v1.whichaxis(T)] = taxis
    elif v1.hasaxis('time'): axes[v1.whichaxis('time')] = taxis
    else: axes = [taxis] + axes

    Var.__init__ (self, axes, dtype=v1.dtype, name=v1.name, atts=v1.atts, plotatts=v1.plotatts)
# }}}

  def getview (self, view, pbar):
# {{{
    from pygeode.timeaxis import Time
    from pygeode.timeutils import reltime
    import numpy as np
    from warnings import warn

    out = np.empty(view.shape, self.dtype)
    out[()] = float('nan')

    # Get the times
    itime = view.index(Time)
    times = view.subaxis(Time)
    handled_times = np.zeros([len(times)], dtype='bool')

#    print times

    # Map these times to values along the 'file' axis
    x = reltime(times, startdate=self.faxis.startdate, units=self.faxis.units)
    file_indices = np.searchsorted(self.faxis.values, x, side='right') - 1 # -1 because we want the file that has a date *before* the specified timestep

    diff = np.diff(file_indices)

    # Where a new file needs to be loaded
    newfile_pos = list(np.where(diff != 0)[0] + 1)
    newfile_pos = [0] + newfile_pos

    # Loop over each file, git 'er done
    for i,p in enumerate(newfile_pos):
      file_index = file_indices[p]
      try:
        file = self.opener(self.files[file_index])
      except Exception as e:
        raise Exception("Multifile: error encountered with file '%s': %s"%(self.files[file_index], str(e)))
      if self.name not in file:
        raise Exception("Multifile: var '%s' was expected to be in file '%s', but it's not there!"%(self.name, self.files[file_index]))
      var = file[self.name] # abandon all hope, ye who use non-unique variable names
      # How does this var map to the overall time axis?
      if var.hasaxis(Time):
        timechunk = var.getaxis(Time)
      else:
        timechunk = self.faxis.slice[file_index]
        # Remove any vestigial internal time axis
        if var.hasaxis('time'):
          assert len(var.getaxis('time')) == 1, "unresolved time axis.  this should have been caught at init time!"
          var = var.squeeze()
      bigmap, smallmap = times.common_map(timechunk)
      # Check for any funky problems with the map
#      assert len(bigmap) > 0, "?? %s <-> %s"%(times,timechunk)
      if len(bigmap) == 0:
        raise Exception("Multifile: Can't find an entire chunk of data for variable '%s'.  Perhaps a file is missing?"%self.name)

      slices = [slice(None)] * self.naxes
      slices[itime] = bigmap
      newview = view.replace_axis(Time, times, bigmap)
      try:
        data = newview.get(var, pbar=pbar.part(i,len(newfile_pos)))
      except Exception as e:
        raise Exception("Multifile: problem fetching variable '%s' from file '%s': %s"%(self.name, self.files[file_index], str(e)))
      # Stick this data into the output
      out[slices] = data
      handled_times[bigmap] = True

    if not np.all(handled_times):
      raise Exception("Multifile: Can't find some data for variable '%s'.  Perhaps a file is missing?"%self.name)
    return out
# }}}
# }}}

del Var

# Decorator for turning an 'open' method into a multifile open method
def multifile (opener):
  def new_opener (files, pattern=None, file2date=None, **kwargs):
    o = lambda f: opener(f, **kwargs) # attach the kwargs to the underlying format opener
    return open_multi (files, pattern=pattern, file2date=file2date, opener=o)
  new_opener.__doc__ = opener.__doc__
  return new_opener


def check_multi (*args, **kwargs):
  ''' Validates the files for completeness and consistency with the assumptions
      made by pygeode.formats.multifile.open_multi.
  '''
  from pygeode.timeutils import reltime
  import numpy as np
  # First, query open_multi to find out what we *expect* to see in all the files
  full_dataset = open_multi (*args, **kwargs)
  # Dig into this object, to find the list of files and the file opener.
  # (this will break if open_multi or Multifile_Var are ever changed!)
  sample_var = full_dataset.vars[0]
  assert isinstance(sample_var,Multifile_Var)
  files = sample_var.files
  faxis = sample_var.faxis
  opener = sample_var.opener
  full_taxis = sample_var.getaxis('time')
  del sample_var
  # Helper method - associate a time axis value with a particular file.
  def find_file (t):
    i = np.searchsorted(faxis.values, t, side='right') - 1
    if i == -1:  return '(some missing file?)'
    return files[i]
  # Similar to above, but return all files that should cover all given timesteps.
  def find_files (t_array):
    return sorted(set(map(find_file,t_array)))
  # Loop over each file, and check the contents.
  all_ok = True
  all_expected_times = set(full_taxis.values)

  # Check for uniformity in the data, and report any potential holes.
  dt = np.diff(full_taxis.values)
  expected_dt = min(dt[dt > 0])
  gaps = full_taxis.values[np.where(dt > expected_dt)]
  if len(gaps) > 0:
    print "ERROR: detected gaps on or after file(s):"
    for filename in find_files(gaps):
      print filename
    print "There may be missing files near those files."
    all_ok = False

  covered_times = set()
  for i,filename in enumerate(files):
    print "Scanning "+filename
    try:
      current_file = opener(filename)
    except Exception as e:
      print "  ERROR: Can't even open the file.  Reason: %s"%str(e)
      all_ok = False
      continue
    for var in current_file:
      if var.name not in full_dataset:
        print "  ERROR: unexpected variable '%s'"%var.name
        all_ok = False
        continue
    for var in full_dataset:
      if var.name not in current_file:
        print "  ERROR: missing variable '%s'"%var.name
        all_ok = False
        continue
      try:
        source_data = current_file[var.name].get().flatten()
      except Exception as e:
        print "  ERROR: unable to read source variable '%s'.  Reason: %s"%(var.name, str(e))
        all_ok = False
        continue
      try:
        file_taxis = current_file[var.name].getaxis('time')
        times = reltime(file_taxis, startdate=full_taxis.startdate, units=full_taxis.units)
        multifile_data = var(l_time=list(times)).get().flatten()
      except Exception as e:
        print "  ERROR: unable to read multifile variable '%s'.  Reason: %s"%(var.name, str(e))
        all_ok = False
        continue
      if len(source_data) != len(multifile_data):
        print "  ERROR: size mismatch for variable '%s'"%var.name
        all_ok = False
        continue
      source_mask = ~np.isfinite(source_data)
      multifile_mask = ~np.isfinite(multifile_data)
      if not np.all(source_mask == multifile_mask):
        print "  ERROR: different missing value masks found in multifile vs. direct access for '%s'"%var.name
        all_ok = False
        continue
      source_data = np.ma.masked_array(source_data, mask=source_mask)
      multifile_data = np.ma.masked_array(multifile_data, mask=multifile_mask)
      if not np.all(source_data == multifile_data):
        print "  ERROR: get different data from multifile vs. direct access for '%s'"%var.name
        all_ok = False
        continue

    covered_times.update(times)
    if i < len(files)-1 and np.any(times >= faxis[i+1]):
      print "  ERROR: found timesteps beyond the expected range of this file."
      all_ok = False
    if np.any(times < faxis[i]):
      print "  ERROR: found timestep(s) earlier than the expected start of this file."
      all_ok = False

  missing_times = all_expected_times - covered_times
  if len(missing_times) > 0:
    print "ERROR: did not get full time coverage.  Missing some timesteps for file(s):"
    for filename in find_files(missing_times):
      print filename
    all_ok = False
  extra_times = covered_times - all_expected_times
  if len(extra_times) > 0:
    print "ERROR: found extra (unexpected) timesteps in the following file(s):"
    for filename in find_files(extra_times):
      print filename
    all_ok = False

  if all_ok:
    print "Scan completed without any errors."
  else:
    print "One or more errors occurred while scanning the files."

