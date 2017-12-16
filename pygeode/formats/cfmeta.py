# Helper module for converting to/from cf-compliant variables
#Note: this isn't a "format" per se, but a bunch of code that's used by several formats
#  (netcdf, hdf, opendap)

#TODO: filter the characters used in axis/metadata names
#TODO: use 'bounds' attribute to determine the resolution of the time axis.
#      (ignore cell_methods from vars.  Pygeode doesn't care how the data was derived.)

# Check for axes registered via the entry_points mechanism of pkg_resources.
custom_axes = {}
import pkg_resources
for ep in pkg_resources.iter_entry_points('pygeode.axis'):
  custom_axes[ep.name] = ep.load()
  del ep
del pkg_resources

# Wrapper for replacing a variable's axes with new ones
# (the axes must be in 1:1 correspondence with the old ones)
from pygeode.var import Var
class var_newaxes (Var):
  def __init__(self, var, newaxes, name=None, fillvalue=None, scale=None, offset=None, atts={}, plotatts={}):
    from pygeode.var import Var, copy_meta 
    atts = atts.copy()
    plotatts = plotatts.copy()
    assert len(newaxes) == len(var.axes)
    for a1, a2 in zip(newaxes, var.axes): assert len(a1) == len(a2)
    self.var = var
    dtype = var.dtype
    if fillvalue is not None or scale is not None or offset is not None: dtype = 'float32'
    self.fillvalue = fillvalue
    self.scale = scale
    self.offset = offset
    Var.__init__(self, newaxes, dtype=dtype)
    copy_meta(var, self)
    self.atts = atts
    self.plotatts = plotatts
    if name is not None: self.name = name
#    self.name = var.name  # we should have a name at this point??
#    if len(atts) > 0: self.atts = atts
  def getview (self, view, pbar):
    from pygeode.view import View
    import numpy as np
    fillvalue = self.fillvalue
    scale = self.scale
    offset = self.offset
    # Do a brute-force mapping of the indices to the internal axes
    # (should work if the axes are in 1:1 correspondence)
    data = View(self.var.axes, force_slices=view.slices,
                force_integer_indices=view.integer_indices).get(self.var, pbar=pbar)
    if fillvalue is not None or scale is not None or offset is not None: data = np.copy(data)
    if fillvalue is not None: w = np.where(data==fillvalue)
    data = np.asarray(data, self.dtype)
    if scale is not None: data *= scale
    if offset is not None: data += offset
    if fillvalue is not None: data[w] = float('nan')

    return data

del Var



# Convert a name to a cf-compliant name
def fix_char(n):
  if 'a' <= n and n <= 'z': return n
  if 'A' <= n and n <= 'Z': return n
  if '0' <= n and n <= '9': return n
  return '_'
def fix_name (name):
  # Fix bad characters
  name = ''.join(fix_char(n) for n in name)
  # Special case: can't start with a number
  if name[0].isdigit(): name = '_'+name
  return name


# Convert a 1D string variable into a 2D character variable
# (useful for encoding string arrays into netcdf)
# Taken from EC-CAS diagnostic package.
def encode_string_var (var):
  import numpy as np
  from pygeode.var import Var
  from pygeode.axis import DummyAxis

  # Construct a 2D character array to hold strings
  strlen = max(len(string) for string in var.values)
  #TODO: make this a simple dimension (no coordinate values needed!)
  strlen_axis = DummyAxis (strlen, name=var.name+"_strlen")
  dtype = '|S'+str(strlen)  # For a convenient view on the character array
                                      # (to help popluate it from strings)

  data = np.zeros(list(var.shape)+[strlen], dtype='|S1')
  data.view(dtype)[...,0] = var.values
  var = Var(list(var.axes)+[strlen_axis], values=data, name=var.name+"_name")
  return var

# Convert a 2D character variable back into a 1D string variable
# Taken from EC-CAS diagnostic package.
def decode_string_var (var):
  from pygeode.var import Var

  name = var.name
  if name.endswith('_name'):
    name = name[:-5]
  data = [''.join(s) for s in var.get()]
  return Var(axes=var.axes[:-1], values=data, name=name)


###############################################################################
# Encode a set of variables as cf-compliant
def encode_cf (dataset):
  from pygeode.dataset import asdataset, Dataset
  from pygeode.axis import Lat, Lon, Pres, Hybrid, XAxis, YAxis, ZAxis, TAxis, NonCoordinateAxis, Station
  from pygeode.timeaxis import Time, ModelTime365, ModelTime360, StandardTime, Yearless
  from pygeode.axis import NamedAxis, DummyAxis
  from pygeode.var import Var
  from pygeode.timeutils import reltime
  from copy import copy
  dataset = asdataset(dataset)
  varlist = list(dataset)
  axisdict = dataset.axisdict.copy()
  global_atts = dataset.atts.copy()
  del dataset

  # Fix the variable names
  for i,v in enumerate(list(varlist)):
    oldname = v.name
    newname = fix_name(oldname)
    if newname != oldname:
      from warnings import warn
      warn ("renaming '%s' to '%s'"%(oldname,newname))
      varlist[i] = v.rename(newname)

  # Fix the axis names
  #TODO

  # Fix the variable metadata
  #TODO

  # Fix the global metadata
  # Specify the conventions we're (supposedly) using
  global_atts['Conventions'] = "CF-1.0"

  for v in varlist: assert v.name not in axisdict, "'%s' refers to both a variable and an axis"%v.name

  # Metadata based on axis classes
  for name,a in axisdict.items():
    atts = a.atts.copy()
    plotatts = a.plotatts.copy() # passed on to Axis constructor
    
    if isinstance(a,Lat):
      atts['standard_name'] = 'latitude'
      atts['units'] = 'degrees_north'
    if isinstance(a,Lon):
      atts['standard_name'] = 'longitude'
      atts['units'] = 'degrees_east'
    if isinstance(a,Pres):
      atts['standard_name'] = 'air_pressure'
      atts['units'] = 'hPa'
      atts['positive'] = 'down'
    if isinstance(a,Hybrid):
      #TODO: formula_terms (how do we specify LNSP instead of P0?????)
      atts['standard_name'] = 'atmosphere_hybrid_sigma_pressure_coordinate'
    if isinstance(a,Time):
      atts['standard_name'] = 'time'
      #TODO: change the unit depending on the time resolution?
      start = a.startdate
      atts['units'] = '%s since %04i-%02i-%02i %02i:%02i:%02i'% (a.units,
        start.get('year',0), start.get('month',1), start.get('day',1),
        start.get('hour',0), start.get('minute',0), start.get('second',0)
      )
    if isinstance(a,StandardTime): atts['calendar'] = 'standard'
    if isinstance(a,ModelTime365): atts['calendar'] = '365_day'
    if isinstance(a,ModelTime360): atts['calendar'] = '360_day'
    if isinstance(a,Yearless): atts['calendar'] = 'none'

    if isinstance(a,XAxis): atts['axis'] = 'X'
    if isinstance(a,YAxis): atts['axis'] = 'Y'
    if isinstance(a,ZAxis): atts['axis'] = 'Z'
    if isinstance(a,TAxis): atts['axis'] = 'T'

    # Change the time axis to be relative to a start date
    #TODO: check 'units' attribute of the time axis, use that in the 'units' of the netcdf metadata
    if isinstance(a, Time):
      #TODO: cast into an integer array if possible
      axisdict[name] = NamedAxis(values=reltime(a), name=name, atts=atts, plotatts=plotatts)
      continue

    # Encode non-coordinate axes, including station (timeseries) data.
    # Loosely follow http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series
    # Move station lat/lon/name data into separate variables.
    if isinstance(a, NonCoordinateAxis):

      # Keep track of extra variables created from auxarray data.
      extra_vars = []

      # Detect certain arrays that should be treated as "coordinates".
      coordinates = []

      # Encode station latitude.
      if 'lat' in a.auxarrays:
        lat = a.auxasvar('lat')
        lat.atts = dict(standard_name="latitude", long_name=a.name+" latitude", units="degrees_north")
        extra_vars.append(lat)
        coordinates.append('lat')
      # Encode station longitude.
      if 'lon' in a.auxarrays:
        lon = a.auxasvar('lon')
        lon.atts = dict(standard_name="longitude", long_name=a.name+" longitude", units="degrees_east")
        extra_vars.append(lon)
        coordinates.append('lon')

      coordinates = " ".join(coordinates)

      # Encode other auxarrays as generic "ancillary" arrays.
      ancillary_variables = []
      for auxname in a.auxarrays.keys():
        if auxname in coordinates: continue  # Handled above
        var = a.auxasvar(auxname)
        if var.dtype.name.startswith('string'):
          var = encode_string_var(var)
        # Some extra CF encoding for the station name, to use it as the unique identifier.
        if auxname == 'station':
          var.atts = dict(cf_role = "timeseries_id")
        extra_vars.append(var)
        ancillary_variables.append(auxname)

      ancillary_variables = " ".join(ancillary_variables)

      # Attach these coordinates to all variables that use this axis.
      #TODO: cleaner way of adding this information without having to do a shallow copy.
      for i,var in enumerate(varlist):
        if var.hasaxis(a):
          var = copy(var)
          var.atts = copy(var.atts)
          if len(coordinates) > 0:
            var.atts['coordinates'] = coordinates
          if len(ancillary_variables) > 0:
            var.atts['ancillary_variables'] = ancillary_variables
          varlist[i] = var

      # Add these coordinates / ancillary variables to the output.
      varlist.extend(extra_vars)

      # The values in the axis itself are meaningless, so mark them as such
      axisdict[name] = DummyAxis(len(a),name=name)

      # Special case: Station (timeseries) data.
      if isinstance(a, Station):
        global_atts['featureType'] = "timeSeries"
      # Nothing more to do for this axis type
      continue

    # Encode custom axes from add-ons
    for n,c in custom_axes.items():
      if isinstance(a,c):
        atts['standard_name'] = n

    # Add associated arrays as new variables
    auxarrays = a.auxarrays
    for aux,values in auxarrays.iteritems():
      auxname = name+'_'+aux
      assert not any(v.name == auxname for v in varlist), "already have a variable named %s"%auxname
      varlist.append( Var([a], values=values, name=auxname) )
    if len(auxarrays) > 0:
      atts['ancillary_variables'] = ' '.join(name+'_'+aux for aux in auxarrays.iterkeys())

    # Create new, generic axes with the desired attributes
    # (Replaces the existing entry in the dictionary)
    axisdict[name] = NamedAxis(values=a.values, name=name, atts=atts, plotatts=plotatts)

  # Apply these new axes to the variables
  for i,oldvar in enumerate(list(varlist)):
    name = oldvar.name
    try:
      #TODO: use Var.replace_axes instead?
      varlist[i] = var_newaxes(oldvar, [axisdict.get(a.name,a) for a in oldvar.axes], atts=oldvar.atts, plotatts=oldvar.plotatts)
    except KeyError:
      print '??', a.name, axisdict
      raise

  dataset = Dataset(varlist, atts=global_atts)
  return dataset

###############################################################################
# Decode cf-compliant variables
def decode_cf (dataset, ignore=[]):
  from pygeode.dataset import asdataset, Dataset
  from pygeode.axis import Axis, NamedAxis, Lat, Lon, Pres, Hybrid, XAxis, YAxis, ZAxis, TAxis, Station, DummyAxis, NonCoordinateAxis
  from pygeode.timeaxis import Time, ModelTime365, ModelTime360, StandardTime, Yearless
  from pygeode import timeutils
  from warnings import warn
  import re

#  dataset = asdataset(dataset, copy=True)
  dataset = asdataset(dataset)
  varlist = list(dataset)
  axisdict = dataset.axisdict.copy()
  global_atts = dataset.atts
  del dataset

  # Decode string variables
  for i,var in enumerate(varlist):
    if var.name.endswith("_name") and var.dtype.name == "string8" and var.axes[-1].name.endswith("_strlen"):
      varlist[i] = decode_string_var(var)

  # data for auxiliary arrays
  auxdict = {}
  for name in axisdict.iterkeys(): auxdict[name] = {}

  # fill values / scale / offset (if applicable)
  fillvalues = {}
  scales = {}
  offsets = {}
  for v in varlist:
    name = v.name
    fillvalues[name] = None
    scales[name] = None
    offsets[name] = None

  for name,a in axisdict.items():

    # Skip over this axis?
    if name in ignore: continue

    atts = a.atts.copy()
    plotatts = a.plotatts.copy() # just carry along and pass to new Axis instance

    # Find any auxiliary arrays
    aux = auxdict[name]
    if 'ancillary_variables' in atts:
      _anc = atts.pop('ancillary_variables')
      remove_from_dataset = []  # vars to remove from the dataset
      for auxname in _anc.split(' '):
        assert any(v.name == auxname for v in varlist), "ancillary variable '%s' not found"%auxname
        newname = auxname
        # Remove the axis name prefix, if it was used
        if newname.startswith(name+'_'): newname = newname[len(name)+1:]
        aux[newname] = [v for v in varlist if v.name == auxname].pop().get()
        # Don't need this as a var anymore
        remove_from_dataset.append(auxname)

      # Remove some stuff
      varlist = [v for v in varlist if v.name not in remove_from_dataset]

    # Determine the best Axis subclass to use
#    cls = NamedAxis
    cls = type(a)

    # Generic 'axis' identifiers first
    if 'axis' in atts:
      _axis = atts.pop('axis')
      if _axis == 'X': cls = XAxis
      if _axis == 'Y': cls = YAxis
      if _axis == 'Z': cls = ZAxis
      if _axis == 'T': cls = TAxis

    # Check specific standard names, and also units?
    #TODO: don't *pop* the standard_name, units, etc. until the end of this routine - in case we didn't end up mapping them to an axis
    _ln = atts.get('long_name', a.name).lower()
    _st = atts.get('standard_name',_ln).lower()
    _units = atts.pop('units','')
    if _st == 'latitude' or _units == 'degrees_north': cls = Lat
    if _st == 'longitude' or _units == 'degrees_east': cls = Lon
    if _st == 'air_pressure' or _units in ('hPa','mbar'):
      cls = Pres
      # Don't need this in the metadata anymore (it will be put back in encode_cf)
      atts.pop('positive',None)

    if _st == 'atmosphere_hybrid_sigma_pressure_coordinate':
      #TODO: check formula_terms??
      #TODO: for ccc2nc files, look for long_name == "Model Level", use_AB = <formula>,
      #       A & B embedded as metadata or as data arrays not attached to ancillary_variables
      if 'A' in aux and 'B' in aux:
        cls = Hybrid
      else:
        warn ("Cannot create a proper Hybrid vertical axis, since 'A' and 'B' coefficients aren't found.")

    if _st == 'station':
      cls = Station

    if (_st == 'time' or cls == TAxis or _units.startswith('days since') or _units.startswith('hours since') or _units.startswith('minutes since') or _units.startswith('seconds since')) and ' since ' in _units:
      _calendar = atts.pop('calendar', 'standard')
      if _calendar in ('standard', 'gregorian', 'proleptic_gregorian'): cls = StandardTime
      elif _calendar in ('365_day', 'noleap', '365day'): cls = ModelTime365
      elif _calendar in ('360_day', '360day'): cls = ModelTime360
      elif _calendar in ('none'): cls = Yearless
      else:
        warn ("unknown calendar '%s'"%_calendar)
        continue
      # Extract the time resolution (day, hour, etc), and the reference date
      res, date = re.match("([a-z]+)\s+since\s+(.*)", _units).groups()
      # Pluralize the increment (i.e. day->days)?
      if not res.endswith('s'): res += 's'
      # Extract the rest of the date
      date = date.rstrip()
      year, month, day, hour, minute, second = 0,1,1,0,0,0
      if len(date) > 0: year, date = re.match("(\d+)-?(.*)", date).groups()
      if len(date) > 0: month, date = re.match("(\d+)-?(.*)", date).groups()
      if len(date) > 0: day, date = re.match("(\d+)\s*(.*)", date).groups()
      if len(date) > 0: hour, date = re.match("(\d+):?(.*)", date).groups()
      if len(date) > 0: minute, date = re.match("(\d+):?(.*)", date).groups()
      if len(date) > 0 and date[0] != ' ': second, date = re.match("(\d+)(.*)", date).groups()
      # convert from strings to integers
      #TODO: milliseconds? time zone?
      year, month, day, hour, minute, second = map(int, [year, month, day, hour, minute, float(second)])
      # Create the time axis
      startdate={'year':year, 'month':month, 'day':day, 'hour':hour, 'minute':minute, 'second':second}
      axisdict[name] = cls(a.values, startdate=startdate, units=res, name=name, atts=atts)
      # Special case: start year=0 implies a climatology
      #NOTE: 'climatology' attribute not used, since we don't currently keep
      #      track of the interval that was used for the climatology.
      if year == 0:
        # Don't climatologize(?) the axis if there's more than a year
        if not all(axisdict[name].year == 0):
          warn ("cfmeta: data starts at year 0 (which usually indicates a climatology), but there's more than one year's worth of data!  Keeping it on a regular calendar.", stacklevel=3)
          continue
        axisdict[name] = timeutils.modify(axisdict[name], exclude='year')
      continue  # we've constructed the time axis, so move onto the next axis

    # Check for a match from the custom axes (from add-ons).
    if _st in custom_axes:
      cls = custom_axes[_st]

    # Find any other information that should be put inside this axis.
    # Look for anything that's identified as a coordinate or anicllary
    # variable, and that has this axis as its only dimension.
    dependencies = set()
    for var in varlist:
      if var.hasaxis(a.name):
        dependencies.update(var.atts.get('coordinates','').split())
        dependencies.update(var.atts.get('ancillary_variables','').split())
    # Look up these dependencies.  Only consider 1D information, since we
    # don't yet have a way to associate multidimensional arrays as auxarrays
    # in an axis.
    dependencies = [v for v in varlist if v.name in dependencies and v.naxes == 1 and v.hasaxis(a.name)]

    # If we found any such information, then this is no longer a simple
    # "dummy" axis.
    if issubclass(cls, DummyAxis) and len(dependencies) > 0:
      cls = NonCoordinateAxis

    # Attach the information from these dependent variables as auxiliary arrays.
    aux.update((dep.name,dep.get()) for dep in dependencies)

    # Anything that got attached to this axis should be removed from the
    # list of variables, since it's just extra info specific to the axis.
    varlist = [v for v in varlist if v.name not in aux]


    # put the units back (if we didn't use them)?
    if cls in [Axis, NamedAxis, XAxis, YAxis, ZAxis, TAxis] and _units != '': atts['units'] = _units

    # create new axis instance if need be.
    if cls != type(a):
      axisdict[name] = cls(values=a.values, name=name, atts=atts, **aux)

  # Apply these new axes to the variables
  # Check for fill values, etc.
  # Extract to a list first, then back to a dataset
  # (ensures the dataset axis list is up to date)
  for i,oldvar in enumerate(list(varlist)):
#    name = [n for n,v in dataset.vardict.iteritems() if v is oldvar].pop()
    name = oldvar.name
    atts = oldvar.atts.copy()
    plotatts = oldvar.atts.copy()
    fillvalue = [atts.pop(f,None) for f in ('FillValue', '_FillValue', 'missing_value')]
    fillvalue = filter(None, fillvalue)
    fillvalue = fillvalue[0] if len(fillvalue) > 0 else None
    scale = atts.pop('scale_factor', None)
    offset = atts.pop('add_offset', None)

    varlist[i] = var_newaxes(oldvar, [axisdict[a.name] for a in oldvar.axes],
                    name=name, fillvalue=fillvalue, scale=scale, offset=offset, atts=atts, plotatts=plotatts)

  dataset = Dataset(varlist, atts=global_atts)

  return dataset

