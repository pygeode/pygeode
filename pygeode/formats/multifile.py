#TODO: a simple function interface that takes as arguments a format,
#     a set of filenames, and a date format (i.e., $Y$m$d) to extract
#     from the file names.

# Expand globbed file expressions to a flat list of files
def expand_file_list (file_list):
#  from glob import iglob
  from glob import iglob
  from os.path import exists
  if not isinstance(file_list, list): file_list = [file_list]
  files = [ f for file_glob in file_list for f in iglob(file_glob)]
  assert len(files) > 0, 'No matches found'
  for f in files: assert exists(f), str(f)+" doesn't exist"
  files.sort()
  return files


# Interface for loading multiple files at a time (file a file glob)
# Inputs: the format, the glob, and any additional keywords to pass
# I.e.: openall(files = "file???.nc", format = netcdf)
#NOTE: for a large number of homogeneous files, use the alternative interface below
def openall (files, format, *args, **kwargs):
  from pygeode.dataset import concat
  files = expand_file_list (files)
  sort = kwargs.pop('sorted', True)
  datasets = [ format.open(f, *args, **kwargs) for f in files]
  if sort: datasets = [d.sorted() for d in datasets]
  return concat(*datasets)

def open_multi (files, format=None, opener=None, pattern=None, file2date=None, **kwargs):
# {{{
  ''' open_multi (files, [format, opener, pattern, file2date], **kwargs)
      Opens multiple data sources and tries to merge them together.

      format - a pygeode-supported data format module, such as netcdf (found in pygeode.formats)
      opener - a function which opens an individual file and returns a pygeode dataset. 
               format.open() is used by default; if any custom behaviour is needed it can
               be provided here.
      pattern - a regex pattern to extract date stamps from the filename; used by default file2date.
                Matching patterns must be named <year>, <month>, <day>, <hour> or <minute>.
                Abbreviations are available for the above; $Y matches a four digit year, $m, $d, $H,
                and $M match a two-digit month, day, hour and minute, respectively.
      file2date - a function which returns a date dictionary given a filename. By default a regex pattern is
               used.
  '''

  from pygeode.timeaxis import Time, StandardTime
  from pygeode.dataset import Dataset
  from pygeode.tools import common_dict
  import numpy as np

  files = expand_file_list(files)
  nfiles = len(files)
  assert nfiles > 0

  assert format is not None or opener is not None, "Don't know how to open the files!"
  if opener is None: opener = format.open

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
      d = dict([k,int(v)] for k,v in d.iteritems())
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
  faxis = T(**dates)

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
      offset = t1.reltime(startdate=faxis.startdate, units=faxis.units)[0]
      taxis = faxis.withnewvalues(faxis.values + offset)
    # At least one of first/last files has multiple timesteps?
    else:
      assert t1.units == t2.units
      dt = max(t1.delta(),t2.delta())
      assert dt > 0
      val1 = t1.values[0]
      val2 = t2.reltime(startdate=t1.startdate)[-1]
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
  def __init__ (self, v1, opener, files, faxis, timedict):
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

    Var.__init__ (self, axes, dtype=v1.dtype, name=v1.name, atts=v1.atts)

  def getview (self, view, pbar):
    from pygeode.timeaxis import Time
    import numpy as np
    from warnings import warn

    out = np.empty(view.shape, self.dtype)
    out[()] = float('nan')

    # Get the times
    itime = view.index(Time)
    times = view.subaxis(Time)

#    print times

    # Map these times to values along the 'file' axis
    x = times.reltime(startdate=self.faxis.startdate, units=self.faxis.units)
    file_indices = np.searchsorted(self.faxis.values, x, side='right') - 1 # -1 because we want the file that has a date *before* the specified timestep

    diff = np.diff(file_indices)

    # Where a new file needs to be loaded
    newfile_pos = list(np.where(diff != 0)[0] + 1)
    newfile_pos = [0] + newfile_pos

    # Loop over each file, git 'er done
    for i,p in enumerate(newfile_pos):
     try:
      file_index = file_indices[p]
      file = self.opener(self.files[file_index])
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
      assert len(bigmap) > 0, "?? %s <-> %s"%(times,timechunk)
      slices = [slice(None)] * self.naxes
      slices[itime] = bigmap
      newview = view.replace_axis(Time, times, bigmap)
      data = newview.get(var, pbar=pbar.part(i,len(newfile_pos)))
      # Stick this data into the output
      out[slices] = data
     except Exception:
      from warnings import warn
      warn ("error encountered with file '%s', var '%s'"%(
        self.files[file_index], self.name
      ))
      raise
    return out
del Var


# Decorator for turning an 'open' method into a multifile open method
def multifile (opener):
  def new_opener (files, pattern=None, file2date=None, **kwargs):
    o = lambda f: opener(f, **kwargs) # attach the kwargs to the underlying format opener
    return open_multi (files, pattern=pattern, file2date=file2date, opener=o)
  new_opener.__doc__ = opener.__doc__
  return new_opener

