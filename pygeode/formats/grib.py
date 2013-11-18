from pygeode.var import Var
from pygeode.formats import gribcore as lib

class GribFile:
  def __init__(self, filename):
    from os.path import exists, basename
    indexname = filename + ".index"
    if not exists(indexname):
      indexname = basename(indexname)
    if not exists(indexname):
      print "GribFile: generating "+indexname
      lib.make_index(filename, indexname)
    index = lib.read_Index(indexname)
    file = lib.open_grib(filename)

    self.index = index
    self.file = file

    self.lib = lib


class GribVar(Var):
  def __init__(self, file, v):
    from pygeode import axis, timeaxis
    import numpy as np

    # Get the variable code
    center, table, varid, level_type = lib.get_varcode (file.index, v)
    print "varcode:", center, table, varid, level_type

    # Get time axis
    y, m, d, H, M = lib.get_var_t(file.index, v)
    print "time:", y, m, d, H, M
    #TODO: proper time axis
    time = timeaxis.StandardTime(year=y, month=m, day=d, hour=H, minute=M)

    # Get vertical levels
    #TODO: check for z *range*, handle that as a separate axis type??
    lev1, lev2 = lib.get_var_z(file.index, v)
    print "levels:", lev1, lev2
    zorder = lev2.tolist()
    lev2.sort()

    # Read a,b for hybrid coordinates
    neta = lib.get_var_neta (file.index, v)
    if (neta > 0 and len(lev2) == neta-1):
      a, b = lib.get_var_eta(file.index, v)
      #TODO: specify linear (not logarithmic) axis
      zaxis = axis.Hybrid(lev2, a[1:], b[1:])
    else:
      zaxis = axis.ZAxis(lev2)

    # Strip z axis for surface fields
    if (level_type <= 99 or level_type in (102,200,201)) or (
        level_type == 109 and lev2.size == 1 and lev2[0] == 1):
      zaxis = None
      zorder = [1]

    # Get horizontal axes
    grid_type, ni, nj, la1, lo1, la2, lo2 = lib.get_grid (file.index, v)
    la1 /= 1000.
    la2 /= 1000.
    lo1 /= 1000.
    if lo1 < 0: lo1 += 360
    lo2 /= 1000.
    if lo2 < 0: lo2 += 360
    print ni, nj
    print la1, la2, lo1, lo2
    assert grid_type in (0,4), grid_type

    if grid_type == 0:  # regular lat/lon
      lat = axis.Lat(la1 + np.arange(nj,dtype='d')/(nj-1)*(la2-la1))
    elif grid_type == 4: # gauss grid
      if la1 < la2:
        lat = axis.gausslat(nj, order = 1)
      else:
        lat = axis.gausslat(nj, order = -1)

    lon = axis.Lon(lo1 + np.arange(ni,dtype='d')/(ni-1)*(lo2-lo1))

    axes = [a for a in (time, zaxis, lat, lon) if a is not None]

    if table in international_params:
      longname, units, name = international_params[table][varid]
    else:
      longname, units, name = params[center][table][varid]

    self.file = file
    self.v = v
    self.zorder = zorder
    self.center = center
    self.table = table
    self.varid = varid
    self.level_type = level_type

    #TODO: more attributes
    self.atts = {
      'long_name': longname,
      'units': units,
      'center_id': center,
      'table_id': table,
      'var_id': varid,
      'level_type': level_type
    }
    self.name = name

    Var.__init__(self, axes, 'd', atts=self.atts, plotatts=self.plotatts)


  def getview (self, view, pbar):
    from pygeode.axis import ZAxis, Lat, Lon
    from pygeode.timeaxis import Time
    import numpy as np
    ti = view.index(Time)
    t = np.ascontiguousarray(view.integer_indices[ti], 'int32')
    zi = view.index(ZAxis)
    if zi >= 0:
      z = view.integer_indices[zi]
      # Get the requested level numbers
      outlevs = view.axes[zi].values[z]
      # Figure out how to map these level numbers to the original (scrambled) order
      zorder = np.array(self.zorder)
      #TODO: more elegant solution?
      z = [np.where(zorder==lev)[0][0] for lev in outlevs]
    else: z = [0]

    # Get the levels

    z = np.ascontiguousarray(z, 'int32')
    nt = len(t)
    nz = len(z)
    yi = view.index(Lat)
    y = np.ascontiguousarray(view.integer_indices[yi], 'int32')
    ny = len(y)
    xi = view.index(Lon)
    x = np.ascontiguousarray(view.integer_indices[xi], 'int32')
    nx = len(x)

    # Output data
    out = np.empty(view.shape, self.dtype)
    assert out.dtype.name == 'float64'

    lib.read_data_loop(self.file.file, self.file.index, self.v, nt, t, 
                  nz, z, ny, y, nx, x, out)

    return out


def open(filename, value_override = {}, dimtypes = {}, namemap = {}, varlist = [], cfmeta = True, **kwargs):
# {{{
  from pygeode.dataset import Dataset
  from pygeode.formats import finalize_open

  file = GribFile(filename)
  vars = [GribVar(file,i) for i in range(lib.get_nvars(file.index))]
  # append level type to vars with the same name
  names = [v.name for v in vars]
  for i, v in enumerate(vars):
    if names.count(v.name) > 1: v.name = v.name + '_' + level_types[v.level_type][1]
  d = Dataset(vars)

  return finalize_open(d, dimtypes, namemap, varlist, cfmeta)
# }}}

# stolen from http://www.nco.ncep.noaa.gov/pmb/docs/on388/table3.html

surface_level_types = {
   1:	("Ground or water surface",	"SFC"),
   2:	("Cloud base level",	"CBL"),
   3:	("Cloud top level",	"CTL"),
   4:	("Level of 0 deg (C) isotherm",	"0DEG"),
   5:	("Level of adiabatic condensation lifted from the surface",	"ADCL"),
   6:	("Maximum wind level",	"MWSL"),
   7:	("Tropopause",	"TRO"),
   8:	("Nominal top of atmosphere",	"NTAT"),
   9:	("Sea bottom",	"SEAB"),
 102:	("mean sea level",	"MSL"),
 200:	("entire atmosphere (considered as a single layer)",	"EATM"),
 201:	("entire ocean (considered as a single layer)",	"EOCN")
}

layer_level_types = {
  101:	("layer between two isobaric levels",	"ISBY"),
  104:	("layer between two specified altitudes above MSL",	"GPMY"),
  106:	("layer between two specified height levels above ground",	"HTGY"),
  108:	("layer between two sigma levels",	"SIGY"),
  110:	("layer between two hybrid levels",	"HYBY"),
  112:	("layer between two depths below land surface"	"DBLY"),
  114:	("layer between two isentropic levels",	"THEY"),
  116:	("layer between two levels at specified pressure difference from ground to level",	"SPDY"),
  120:	("layer between two NAM levels",	"NAMY"),
  121:	("layer between two isobaric surfaces (high precision)",	"IBYH"),
  128:	("layer between two sigma levels (high precision)",	"SGYH"),
  141:	("layer between two isobaric surfaces (mixed precision)",	"IBYM")
}

point_level_types = {
   20:	("Isothermal level (temperature in 1/100 K in octets 11 and 12)",	"TMPL"),
  100:	("isobaric level",	"ISBL"),
  103:	("Specified altitude above MSL",	"GPML"),
  105:	("specified height level above ground",	"HTGL"),
  107:	("sigma level",	"SIGL"),
  109:	("Hybrid level",	"HYBL"),
  111:	("depth below land surface",	"DBLL"),
  113:	("isentropic (theta) level",	"THEL"),
  115:	("level at specified pressure difference from ground to level",	"SPDL"),
  117:	("potential vorticity(pv) surface",	"PVL"),
  119:	("NAM level",	"NAML"),
  125:	("specified height level above ground (high precision)",	"HGLH"),
  126:	("isobaric level",	"ISBP"),
  160:	("depth below sea level",	"DBSL")
}

level_types = dict((k,v) for d in (surface_level_types,layer_level_types,point_level_types) for k,v in d.items())

# stolen from http://www.nco.ncep.noaa.gov/pmb/docs/on388/table2.html

international_params = {

  2: {

      1:	("Pressure",	"Pa",	"PRES"),
      2:	("Pressure reduced to MSL",	"Pa",	"PRMSL"),
      3:	("Pressure tendency",	"Pa/s",	"PTEND"),
      4:	("Potential vorticity",	"K m2 kg-1 s-1",	"PVORT"),
      5:	("ICAO Standard Atmosphere Reference Height",	"m",	"ICAHT"),
      6:	("Geopotential",	"m2/s2",	"GP"),
      7:	("Geopotential height",	"gpm",	"HGT"),
      8:	("Geometric height",	"m",	"DIST"),
      9:	("Standard deviation of height",	"m",	"HSTDV"),
     10:	("Total ozone",	"Dobson",	"TOZNE"),
     11:	("Temperature",	"K",	"TMP"),
     12:	("Virtual temperature",	"K",	"VTMP"),
     13:	("Potential temperature",	"K",	"POT"),
     14:	("Pseudo-adiabatic potential temperature or equivalent potential temperature",	"K",	"EPOT"),
     15:	("Maximum temperature",	"K",	"T_MAX"),
     16:	("Minimum temperature",	"K",	"T_MIN"),
     17:	("Dew point temperature",	"K",	"DPT"),
     18:	("Dew point depression (or deficit)",	"K",	"DEPR"),
     19:	("Lapse rate",	"K/m",	"LAPR"),
     20:	("Visibility",	"m",	"VIS"),
     21:	("Radar Spectra (1)",	"-",	"RDSP1"),
     22:	("Radar Spectra (2)",	"-",	"RDSP2"),
     23:	("Radar Spectra (3)",	"-",	"RDSP3"),
     24:	("Parcel lifted index (to 500 hPa)",	"K",	"PLI"),
     25:	("Temperature anomaly",	"K",	"TMP_A"),
     26:	("Pressure anomaly",	"Pa",	"PRESA"),
     27:	("Geopotential height anomaly",	"gpm",	"GP_A"),
     28:	("Wave Spectra (1)",	"-",	"WVSP1"),
     29:	("Wave Spectra (2)",	"-",	"WVSP2"),
     30:	("Wave Spectra (3)",	"-",	"WVSP3"),
     31:	("Wind direction (from which blowing)",	"deg true",	"WDIR"),
     32:	("Wind speed",	"m/s",	"WIND"),
     33:	("u-component of wind",	"m/s",	"U_GRD"),
     34:	("v-component of wind",	"m/s",	"V_GRD"),
     35:	("Stream function",	"m2/s",	"STRM"),
     36:	("Velocity potential",	"m2/s",	"V_POT"),
     37:	("Montgomery stream function",	"m2/s2",	"MNTSF"),
     38:	("Sigma coordinate vertical velocity",	"/s",	"SGCVV"),
     39:	("Vertical velocity (pressure)",	"Pa/s",	"V_VEL"),
     40:	("Vertical velocity (geometric)",	"m/s",	"DZDT"),
     41:	("Absolute vorticity",	"/s",	"ABS_V"),
     42:	("Absolute divergence",	"/s",	"ABS_D"),
     43:	("Relative vorticity",	"/s",	"REL_V"),
     44:	("Relative divergence",	"/s",	"REL_D"),
     45:	("Vertical u-component shear",	"/s",	"VUCSH"),
     46:	("Vertical v-component shear",	"/s",	"VVCSH"),
     47:	("Direction of current",	"Degree true",	"DIR_C"),
     48:	("Speed of current",	"m/s",	"SP_C"),
     49:	("u-component of current",	"m/s",	"UOGRD"),
     50:	("v-component of current",	"m/s",	"VOGRD"),
     51:	("Specific humidity",	"kg/kg",	"SPF_H"),
     52:	("Relative humidity",	"%",	"R_H"),
     53:	("Humidity mixing ratio",	"kg/kg",	"MIXR"),
     54:	("Precipitable water",	"kg/m2",	"P_WAT"),
     55:	("Vapor pressure",	"Pa",	"VAPP"),
     56:	("Saturation deficit",	"Pa",	"SAT_D"),
     57:	("Evaporation",	"kg/m2",	"EVP"),
     58:	("Cloud Ice",	"kg/m2",	"C_ICE"),
     59:	("Precipitation rate",	"kg/m2/s",	"PRATE"),
     60:	("Thunderstorm probability",	"%",	"TSTM"),
     61:	("Total precipitation",	"kg/m2",	"A_PCP"),
     62:	("Large scale precipitation (non-conv.)",	"kg/m2",	"NCPCP"),
     63:	("Convective precipitation",	"kg/m2",	"ACPCP"),
     64:	("Snowfall rate water equivalent",	"kg/m2/s",	"SRWEQ"),
     65:	("Water equiv. of accum. snow depth",	"kg/m2",	"WEASD"),
     66:	("Snow depth",	"m",	"SNO_D"),
     67:	("Mixed layer depth",	"m",	"MIXHT"),
     68:	("Transient thermocline depth",	"m",	"TTHDP"),
     69:	("Main thermocline depth",	"m",	"MTHD"),
     70:	("Main thermocline anomaly",	"m",	"MTH_A"),
     71:	("Total cloud cover",	"%",	"T_CDC"),
     72:	("Convective cloud cover",	"%",	"CDCON"),
     73:	("Low cloud cover",	"%",	"L_CDC"),
     74:	("Medium cloud cover",	"%",	"M_CDC"),
     75:	("High cloud cover",	"%",	"H_CDC"),
     76:	("Cloud water",	"kg/m2",	"C_WAT"),
     77:	("Best lifted index (to 500 hPa)",	"K",	"BLI"),
     78:	("Convective snow",	"kg/m2",	"SNO_C"),
     79:	("Large scale snow",	"kg/m2",	"SNO_L"),
     80:	("Water Temperature",	"K",	"WTMP"),
     81:	("Land cover (land=1, sea=0) (see note)",	"proportion",	"LAND"),
     82:	("Deviation of sea level from mean",	"m",	"DSL_M"),
     83:	("Surface roughness",	"m",	"SFC_R"),
     84:	("Albedo",	"%",	"ALBDO"),
     85:	("Soil temperature",	"K",	"TSOIL"),
     86:	("Soil moisture content",	"kg/m2",	"SOIL_M"),
     87:	("Vegetation",	"%",	"VEG"),
     88:	("Salinity",	"kg/kg",	"SALTY"),
     89:	("Density",	"kg/m3",	"DEN"),
     90:	("Water runoff",	"kg/m2",	"WATR"),
     91:	("Ice cover (ice=1, no ice=0) (See Note)",	"proportion",	"ICE_C"),
     92:	("Ice thickness",	"m",	"ICETK"),
     93:	("Direction of ice drift",	"deg. true",	"DICED"),
     94:	("Speed of ice drift",	"m/s",	"SICED"),
     95:	("u-component of ice drift",	"m/s",	"U_ICE"),
     96:	("v-component of ice drift",	"m/s",	"V_ICE"),
     97:	("Ice growth rate",	"m/s",	"ICE_G"),
     98:	("Ice divergence",	"/s",	"ICE_D"),
     99:	("Snow melt",	"kg/m2",	"SNO_M"),
    100:	("Significant height of combined wind waves and swell",	"m",	"HTSGW"),
    101:	("Direction of wind waves (from which)",	"Degree true",	"WVDIR"),
    102:	("Significant height of wind waves",	"m",	"WVHGT"),
    103:	("Mean period of wind waves",	"s",	"WVPER"),
    104:	("Direction of swell waves",	"Degree true",	"SWDIR"),
    105:	("Significant height of swell waves",	"m",	"SWELL"),
    106:	("Mean period of swell waves",	"s",	"SWPER"),
    107:	("Primary wave direction",	"Degree true",	"DIRPW"),
    108:	("Primary wave mean period",	"s",	"PERPW"),
    109:	("Secondary wave direction",	"Degree true",	"DIRSW"),
    110:	("Secondary wave mean period",	"s",	"PERSW"),
    111:	("Net short-wave radiation flux (surface)",	"W/m2",	"NSWRS"),
    112:	("Net long wave radiation flux (surface)",	"W/m2",	"NLWRS"),
    113:	("Net short-wave radiation flux (top of atmosphere)",	"W/m2 ",	"NSWRT"),
    114:	("Net long wave radiation flux (top of atmosphere)",	"W/m2 ",	"NLWRT"),
    115:	("Long wave radiation flux",	"W/m2 ",	"LWAVR"),
    116:	("Short wave radiation flux",	"W/m2 ",	"SWAVR"),
    117:	("Global radiation flux",	"W/m2 ",	"G_RAD"),
    118:	("Brightness temperature",	"K",	"BRTMP"),
    119:	("Radiance (with respect to wave number)",	"W/m/sr",	"LWRAD"),
    120:	("Radiance (with respect to wave length)",	"W/m3/sr",	"SWRAD"),
    121:	("Latent heat net flux",	"W/m2 ",	"LHTFL"),
    122:	("Sensible heat net flux",	"W/m2 ",	"SHTFL"),
    123:	("Boundary layer dissipation",	"W/m2 ",	"BLYDP"),
    124:	("Momentum flux, u component",	"N/m2 ",	"U_FLX"),
    125:	("Momentum flux, v component",	"N/m2 ",	"V_FLX"),
    126:	("Wind mixing energy",	"J",	"WMIXE"),
    127:	("Image data",	"-",	"IMG_D"),
    128:	("Mean Sea Level Pressure (Standard Atmosphere Reduction)",	"Pa",	"MSLSA"),
    129:	("Mean Sea Level Pressure (MAPS System Reduction)",	"Pa",	"MSLMA"),
    130:	("Mean Sea Level Pressure (NAM Model Reduction)",	"Pa",	"MSLET"),
    131:	("Surface lifted index",	"K",	"LFT_X"),
    132:	("Best (4 layer) lifted index",	"K",	"4LFTX"),
    133:	("K index",	"K",	"K_X"),
    134:	("Sweat index",	"K",	"S_X"),
    135:	("Horizontal moisture divergence",	"kg/kg/s",	"MCONV"),
    136:	("Vertical speed shear",	"1/s",	"VW_SH"),
    137:	("3-hr pressure tendency Std. Atmos. Reduction",	"Pa/s",	"TSLSA"),
    138:	("Brunt-Vaisala frequency (squared)",	"1/s2",	"BVF_2"),
    139:	("Potential vorticity (density weighted)",	"1/s/m",	"PV_MW"),
    140:	("Categorical rain (yes=1; no=0)",	"non-dim",	"CRAIN"),
    141:	("Categorical freezing rain (yes=1; no=0)",	"non-dim",	"CFRZR"),
    142:	("Categorical ice pellets (yes=1; no=0)",	"non-dim",	"CICEP"),
    143:	("Categorical snow (yes=1; no=0)",	"non-dim",	"CSNOW"),
    144:	("Volumetric soil moisture content",	"fraction",	"SOILW"),
    145:	("Potential evaporation rate",	"W/m**2",	"PEVPR"),
    146:	("Cloud workfunction",	"J/kg",	"CWORK"),
    147:	("Zonal flux of gravity wave stress",	"N/m**2",	"U_GWD"),
    148:	("Meridional flux of gravity wave stress",	"N/m**2",	"V_GWD"),
    149:	("Potential vorticity",	"m**2/s/kg",	"PVORT"),
    150:	("Covariance between meridional and zonal components of the wind. Defined as [uv]-[u][v], where \"[]\" indicates the mean over the indicated time span.",	"m2/s2",	"COVMZ"),
    151:	("Covariance between temperature and zonal components of the wind. Defined as [uT]-[u][T], where \"[]\" indicates the mean over the indicated time span.",	"K*m/s",	"COVTZ"),
    152:	("Covariance between temperature and meridional components of the wind. Defined as [vT]-[v][T], where \"[]\" indicates the mean over the indicated time span.",	"K*m/s",	"COVTM"),
    153:	("Cloud water",	"Kg/kg",	"CLWMR"),
    154:	("Ozone mixing ratio",	"Kg/kg",	"O3MR"),
    155:	("Ground Heat Flux",	"W/m2",	"GFLUX"),
    156:	("Convective inhibition",	"J/kg",	"CIN"),
    157:	("Convective Available Potential Energy",	"J/kg",	"CAPE"),
    158:	("Turbulent Kinetic Energy",	"J/kg",	"TKE"),
    159:	("Condensation pressure of parcel lifted from indicated surface",	"Pa",	"CONDP"),
    160:	("Clear Sky Upward Solar Flux",	"W/m2",	"CSUSF"),
    161:	("Clear Sky Downward Solar Flux",	"W/m2",	"CSDSF"),
    162:	("Clear Sky upward long wave flux",	"W/m2",	"CSULF"),
    163:	("Clear Sky downward long wave flux",	"W/m2",	"CSDLF"),
    164:	("Cloud forcing net solar flux",	"W/m2",	"CFNSF"),
    165:	("Cloud forcing net long wave flux",	"W/m2",	"CFNLF"),
    166:	("Visible beam downward solar flux",	"W/m2",	"VBDSF"),
    167:	("Visible diffuse downward solar flux",	"W/m2",	"VDDSF"),
    168:	("Near IR beam downward solar flux",	"W/m2",	"NBDSF"),
    169:	("Near IR diffuse downward solar flux",	"W/m2",	"NDDSF"),
    170:	("Rain water mixing ratio",	"Kg/Kg",	"RWMR"),
    171:	("Snow mixing ratio",	"Kg/Kg",	"SNMR"),
    172:	("Momentum flux",	"N/m2",	"M_FLX"),
    173:	("Mass point model surface",	"non-dim",	"LMH"),
    174:	("Velocity point model surface",	"non-dim",	"LMV"),
    175:	("Model layer number (from bottom up)",	"non-dim",	"MLYNO"),
    176:	("latitude (-90 to +90)",	"deg",	"NLAT"),
    177:	("east longitude (0-360)",	"deg",	"ELON"),
    178:	("Ice mixing ratio",	"Kg/Kg",	"ICMR"),
    179:	("Graupel mixing ratio",	"Kg/Kg",	"GRMR"),
    180:	("Surface wind gust",	"m/s",	"GUST"),
    181:	("x-gradient of log pressure",	"1/m",	"LPS_X"),
    182:	("y-gradient of log pressure",	"1/m",	"LPS_Y"),
    183:	("x-gradient of height",	"m/m",	"HGT_X"),
    184:	("y-gradient of height",	"m/m",	"HGT_Y"),
    185:	("Turbulence Potential Forecast Index",	"non-dim",	"TPFI"),
    186:	("Total Icing Potential Diagnostic",	"non-dim",	"TIPD"),
    187:	("Lightning",	"non-dim",	"LTNG"),
    188:	("Rate of water dropping from canopy to ground",	"-",	"RDRIP"),
    189:	("Virtual potential temperature",	"K",	"VPTMP"),
    190:	("Storm relative helicity",	"m2/s2",	"HLCY"),
    191:	("Probability from ensemble",	"numeric",	"PROB"),
    192:	("Probability from ensemble normalized with respect to climate expectancy",	"numeric",	"PROBN"),
    193:	("Probability of precipitation",	"%",	"POP"),
    194:	("Percent of frozen precipitation",	"%",	"CPOFP"),
    195:	("Probability of freezing precipitation",	"%",	"CPOZP"),
    196:	("u-component of storm motion",	"m/s",	"USTM"),
    197:	("v-component of storm motion",	"m/s",	"VSTM"),
    198:	("Number concentration for ice particles",	"-",	"NCIP"),
    199:	("Direct evaporation from bare soil",	"W/m2",	"EVBS"),
    200:	("Canopy water evaporation",	"W/m2",	"EVCW"),
    201:	("Ice-free water surface",	"%",	"ICWAT"),
    202:	("Convective weather detection index",	"non-dim",	"CWDI"),
    203:	("VAFTAD",	"log10(kg/m3)",	"VAFTD"),
    204:	("downward short wave rad. flux",	"W/m2",	"DSWRF"),
    205:	("downward long wave rad. flux",	"W/m2",	"DLWRF"),
    206:	("Ultra violet index (1 hour integration centered at solar noon)",	"J/m2",	"UVI"),
    207:	("Moisture availability",	"%",	"MSTAV"),
    208:	("Exchange coefficient",	"(kg/m3)(m/s)",	"SFEXC"),
    209:	("No. of mixed layers next to surface",	"integer",	"MIXLY"),
    210:	("Transpiration",	"W/m2",	"TRANS"),
    211:	("upward short wave rad. flux",	"W/m2",	"USWRF"),
    212:	("upward long wave rad. flux",	"W/m2",	"ULWRF"),
    213:	("Amount of non-convective cloud",	"%",	"CDLYR"),
    214:	("Convective Precipitation rate",	"kg/m2/s",	"CPRAT"),
    215:	("Temperature tendency by all physics",	"K/s",	"TTDIA"),
    216:	("Temperature tendency by all radiation",	"K/s",	"TTRAD"),
    217:	("Temperature tendency by non-radiation physics",	"K/s",	"TTPHY"),
    218:	("precip.index(0.0-1.00) (see note)",	"fraction",	"PREIX"),
    219:	("Std. dev. of IR T over 1x1 deg area",	"K",	"TSD1D"),
    220:	("Natural log of surface pressure",	"ln(kPa)",	"NLGSP"),
    221:	("Planetary boundary layer height",	"m",	"HPBL"),
    222:	("5-wave geopotential height",	"gpm",	"5WAVH"),
    223:	("Plant canopy surface water",	"kg/m2",	"CNWAT"),
    224:	("Soil type (as in Zobler)",	"Integer (0-9)",	"SOTYP"),
    225:	("Vegitation type (as in SiB)",	"Integer (0-13)",	"VGTYP"),
    226:	("Blackadar's mixing length scale",	"m",	"BMIXL"),
    227:	("Asymptotic mixing length scale",	"m",	"AMIXL"),
    228:	("Potential evaporation",	"kg/m2",	"PEVAP"),
    229:	("Snow phase-change heat flux",	"W/m2",	"SNOHF"),
    230:	("5-wave geopotential height anomaly",	"gpm",	"5WAVA"),
    231:	("Convective cloud mass flux",	"Pa/s",	"MFLUX"),
    232:	("Downward total radiation flux",	"W/m2",	"DTRF"),
    233:	("Upward total radiation flux",	"W/m2",	"UTRF"),
    234:	("Baseflow-groundwater runoff",	"kg/m2",	"BGRUN"),
    235:	("Storm surface runoff",	"kg/m2",	"SSRUN"),
    236:	("Supercooled Large Droplet (SLD) Icing Potential Diagnostic",	"NumericSee Note (1)",	"SIPD"),
    237:	("Total ozone",	"Kg/m2",	"03TOT"),
    238:	("Snow cover",	"percent",	"SNOWC"),
    239:	("Snow temperature",	"K",	"SNO_T"),
    240:	("Covariance between temperature and vertical component of the wind. Defined as [wT]-[w][T], where \"[]\" indicates the mean over the indicated time span",	"K*m/s",	"COVTW"),
    241:	("Large scale condensate heat rate",	"K/s",	"LRGHR"),
    242:	("Deep convective heating rate",	"K/s",	"CNVHR"),
    243:	("Deep convective moistening rate",	"kg/kg/s",	"CNVMR"),
    244:	("Shallow convective heating rate",	"K/s",	"SHAHR"),
    245:	("Shallow convective moistening rate",	"kg/kg/s",	"SHAMR"),
    246:	("Vertical diffusion heating rate",	"K/s",	"VDFHR"),
    247:	("Vertical diffusion zonal acceleration",	"m/s2",	"VDFUA"),
    248:	("Vertical diffusion meridional acceleration",	"m/s2",	"VDFVA"),
    249:	("Vertical diffusion moistening rate",	"kg/kg/s",	"VDFMR"),
    250:	("Solar radiative heating rate",	"K/s",	"SWHR"),
    251:	("Long wave radiative heating rate",	"K/s",	"LWHR"),
    252:	("Drag coefficient",	"non-dim",	"CD"),
    253:	("Friction velocity",	"m/s",	"FRICV"),
    254:	("Richardson number",	"non-dim.",	"RI"),
    255:	("Missing",	"-",	"-")

  }
}

# Stolen from http://www.ecmwf.int/products/data/technical/GRIB_tables/table_128.html

ECMWF = {

  128: {
      1:	("Stream function",	"m**2 s**-1",	"STRF"),
      2:	("Velocity potential",	"m**2 s**-1",	"VPOT"),
      3:	("Potential temperature",	"K",	"PT"),
      4:	("Equivalent potential temperature",	"K",	"EQPT"),
      5:	("Saturated equivalent potential temperature",	"K",	"SEPT"),
     11:	("U component of divergent wind",	"m s**-1",	"UDVW"),
     12:	("V component of divergent wind",	"m s**-1",	"VDVW"),
     13:	("U component of rotational wind",	"m s**-1",	"URTW"),
     14:	("V component of rotational wind",	"m s**-1",	"VRTW"),
     21:	("Unbalanced component of temperature",	"K",	"UCTP"),
     22:	("Unbalanced component of logarithm of surface pressure",	"-",	"UCLN"),
     23:	("Unbalanced component of divergence",	"s**-1",	"UCDV"),
     26:	("Lake cover",	"(0-1)",	"CL"),
     27:	("Low vegetation cover",	"(0-1)",	"CVL"),
     28:	("High vegetation cover",	"(0-1)",	"CVH"),
     29:	("Type of low vegetation",	"-",	"TVL"),
     30:	("Type of high vegetation",	"-",	"TVH"),
     31:	("Sea-ice cover",	"(0-1)",	"CI"),
     32:	("Snow albedo",	"(0-1)",	"ASN"),
     33:	("Snow density",	"kg m**-3",	"RSN"),
     34:	("Sea surface temperature",	"K",	"SSTK"),
     35:	("Ice surface temperature layer 1",	"K",	"ISTL1"),
     36:	("Ice surface temperature layer 2",	"K",	"ISTL2"),
     37:	("Ice surface temperature layer 3",	"K",	"ISTL3"),
     38:	("Ice surface temperature layer 4",	"K",	"ISTL4"),
     39:	("Volumetric soil water layer 1",	"m**3 m**-3",	"SWVL1"),
     40:	("Volumetric soil water layer 2",	"m**3 m**-3",	"SWVL2"),
     41:	("Volumetric soil water layer 3",	"m**3 m**-3",	"SWVL3"),
     42:	("Volumetric soil water layer 4",	"m**3 m**-3",	"SWVL4"),
     43:	("Soil type",	"-",	"SLT"),
     44:	("Snow evaporation",	"m of water",	"ES"),
     45:	("Snowmelt",	"m of water",	"SMLT"),
     46:	("Solar duration",	"s",	"SDUR"),
     47:	("Direct solar radiation",	"w m**-2",	"DSRP"),
     48:	("Magnitude of surface stress",	"N m**-2 s",	"MAGSS"),
     49:	("Wind gust at 10 metres",	"m s**-1",	"10FG"),
     50:	("Large-scale precipitation fraction",	"s",	"LSPF"),
     51:	("Maximum 2 metre temperature",	"K",	"MX2T24"),
     52:	("Minimum 2 metre temperature",	"K",	"MN2T24"),
     53:	("Montgomery potential",	"m**2 s**-2",	"MONT"),
     54:	("Pressure",	"Pa",	"PRES"),
     60:	("Potential vorticity",	"K m**2 kg**-1 s**-1",	"PV"),
    127:	("Atmospheric tide",	"-",	"AT"),
    128:	("Budget values",	"-",	"BV"),
    129:	("Geopotential",	"m**2 s**-2",	"Z"),
    130:	("Temperature",	"K",	"T"),
    131:	("U velocity",	"m s**-1",	"U"),
    132:	("V velocity",	"m s**-1",	"V"),
    133:	("Specific humidity",	"kg kg**-1",	"Q"),
    134:	("Surface pressure",	"Pa",	"SP"),
    135:	("Vertical velocity",	"Pa s**-1",	"W"),
    136:	("Total column water",	"kg m**-2",	"TCW"),
    137:	("Total column water vapour",	"kg m**-2",	"TCWV"),
    138:	("Vorticity (relative)",	"s**-1",	"VO"),
    139:	("Soil temperature level 1",	"K",	"STL1"),
    140:	("Soil wetness level 1",	"m of water",	"SWL1"),
    141:	("Snow depth",	"m of water equivalent",	"SD"),
    142:	("Stratiform precipitation",	"m",	"LSP"),
    143:	("Convective precipitation",	"m",	"CP"),
    144:	("Snowfall (convective + stratiform)",	"m of water equivalent",	"SF"),
    145:	("Boundary layer dissipation",	"W m**-2 s",	"BLD"),
    146:	("Surface sensible heat flux",	"W m**-2 s",	"SSHF"),
    147:	("Surface latent heat flux",	"W m**-2 s",	"SLHF"),
    148:	("Charnock",	"-",	"CHNK"),
    149:	("Surface net radiation",	"W m**-2 s",	"SNR"),
    150:	("Top net radiation",	"-",	"TNR"),
    151:	("Mean sea-level pressure",	"Pa",	"MSL"),
    152:	("Logarithm of surface pressure",	"-",	"LNSP"),
    153:	("Short-wave heating rate",	"K",	"SWHR"),
    154:	("Long-wave heating rate",	"K",	"LWHR"),
    155:	("Divergence",	"s**-1",	"D"),
    156:	("Height",	"m",	"GH"),
    157:	("Relative humidity",	"%",	"R"),
    158:	("Tendency of surface pressure",	"Pa s**-1",	"TSP"),
    159:	("Boundary layer height",	"m",	"BLH"),
    160:	("Standard deviation of orography",	"-",	"SDOR"),
    161:	("Anisotropy of sub-gridscale orography",	"-",	"ISOR"),
    162:	("Angle of sub-gridscale orography",	"rad",	"ANOR"),
    163:	("Slope of sub-gridscale orography",	"-",	"SLOR"),
    164:	("Total cloud cover",	"(0 - 1)",	"TCC"),
    165:	("10 metre U wind component",	"m s**-1",	"10U"),
    166:	("10 metre V wind component",	"m s**-1",	"10V"),
    167:	("2 metre temperature",	"K",	"2T"),
    168:	("2 metre dewpoint temperature",	"K",	"2D"),
    169:	("Surface solar radiation downwards",	"W m**-2 s",	"SSRD"),
    170:	("Soil temperature level 2",	"K",	"STL2"),
    171:	("Soil wetness level 2",	"m of water",	"SWL2"),
    172:	("Land/sea mask",	"(0, 1)",	"LSM"),
    173:	("Surface roughness",	"m",	"SR"),
    174:	("Albedo",	"(0 - 1)",	"AL"),
    175:	("Surface thermal radiation downwards",	"W m**-2 s",	"STRD"),
    176:	("Surface solar radiation",	"W m**-2 s",	"SSR"),
    177:	("Surface thermal radiation",	"W m**-2 s",	"STR"),
    178:	("Top solar radiation",	"W m**-2 s",	"TSR"),
    179:	("Top thermal radiation",	"W m**-2 s",	"TTR"),
    180:	("East/West surface stress",	"N m**-2 s",	"EWSS"),
    181:	("North/South surface stress",	"N m**-2 s",	"NSSS"),
    182:	("Evaporation",	"m of water",	"E"),
    183:	("Soil temperature level 3",	"K",	"STL3"),
    184:	("Soil wetness level 3",	"m of water",	"SWL3"),
    185:	("Convective cloud cover",	"(0 - 1)",	"CCC"),
    186:	("Low cloud cover",	"(0 - 1)",	"LCC"),
    187:	("Medium cloud cover",	"(0 - 1)",	"MCC"),
    188:	("High cloud cover",	"(0 - 1)",	"HCC"),
    189:	("Sunshine duration",	"s",	"SUND"),
    190:	("EW component of subgrid orographic variance",	"m**2",	"EWOV"),
    191:	("NS component of subgrid orographic variance",	"m**2",	"NSOV"),
    192:	("NWSE component of subgrid orographic variance",	"m**2",	"NWOV"),
    193:	("NESW component of subgrid orographic variance",	"m**2",	"NEOV"),
    194:	("Brightness temperature",	"K",	"BTMP"),
    195:	("Lat. component of gravity wave stress",	"N m**-2 s",	"LGWS"),
    196:	("Meridional component of gravity wave stress",	"N m**-2 s",	"MGWS"),
    197:	("Gravity wave dissipation",	"W m**-2 s",	"GWD"),
    198:	("Skin reservoir content",	"m of water",	"SRC"),
    199:	("Vegetation fraction",	"(0 - 1)",	"VEG"),
    200:	("Variance of sub-gridscale orography",	"m**2",	"VSO"),
    201:	("Maximum 2 metre temperature since previous post-processing",	"K",	"MX2T"),
    202:	("Minimum 2 metre temperature since previous post-processing",	"K",	"MN2T"),
    203:	("Ozone mass mixing ratio",	"kg kg**-1",	"O3"),
    204:	("Precipiation analysis weights",	"-",	"PAW"),
    205:	("Runoff",	"m",	"RO"),
    206:	("Total column ozone",	"Dobson",	"TCO3"),
    207:	("10 meter windspeed",	"m s**-1",	"10SI"),
    208:	("Top net solar radiation, clear sky",	"W m**-2",	"TSRC"),
    209:	("Top net thermal radiation, clear sky",	"W m**-2",	"TTRC"),
    210:	("Surface net solar radiation, clear sky",	"W m**-2",	"SSRC"),
    211:	("Surface net thermal radiation, clear sky",	"W m**-2",	"STRC"),
    212:	("Solar insolation",	"W m**-2",	"SI"),
    214:	("Diabatic heating by radiation",	"K",	"DHR"),
    215:	("Diabatic heating by vertical diffusion",	"K",	"DHVD"),
    216:	("Diabatic heating by cumulus convection",	"K",	"DHCC"),
    217:	("Diabatic heating large-scale condensation",	"K",	"DHLC"),
    218:	("Vertical diffusion of zonal wind",	"m s**-1",	"VDZW"),
    219:	("Vertical diffusion of meridional wind",	"m s**-1",	"VDMW"),
    220:	("EW gravity wave drag tendency",	"m s**-1",	"EWGD"),
    221:	("NS gravity wave drag tendency",	"m s**-1",	"NSGD"),
    222:	("Convective tendency of zonal wind",	"m s**-1",	"CTZW"),
    223:	("Convective tendency of meridional wind",	"m s**-1",	"CTMW"),
    224:	("Vertical diffusion of humidity",	"kg kg**-1",	"VDH"),
    225:	("Humidity tendency by cumulus convection",	"kg kg**-1",	"HTCC"),
    226:	("Humidity tendency large-scale condensation",	"kg kg**-1",	"HTLC"),
    227:	("Change from removing negative humidity",	"kg kg**-1",	"CRNH"),
    228:	("Total precipitation",	"m",	"TP"),
    229:	("Instantaneous X surface stress",	"N m**-2",	"IEWS"),
    230:	("Instantaneous Y surface stress",	"N m**-2",	"INSS"),
    231:	("Instantaneous surface heat flux",	"W m**-2",	"ISHF"),
    232:	("Instantaneous moisture flux",	"kg m**-2 s",	"IE"),
    233:	("Apparent surface humidity",	"kg kg**-1",	"ASQ"),
    234:	("Logarithm of surface roughness length for heat",	"-",	"LSRH"),
    235:	("Skin temperature",	"K",	"SKT"),
    236:	("Soil temperature level 4",	"K",	"STL4"),
    237:	("Soil wetness level 4",	"m",	"SWL4"),
    238:	("Temperature of snow layer",	"K",	"TSN"),
    239:	("Convective snowfall",	"m of water equivalent",	"CSF"),
    240:	("Large-scale snowfall",	"m of water equivalent",	"LSF"),
    241:	("Accumulated cloud fraction tendency",	"(-1 to 1)",	"ACF"),
    242:	("Accumulated liquid water tendency",	"(-1 to 1)",	"ALW"),
    243:	("Forecast albedo",	"(0 - 1)",	"FAL"),
    244:	("Forecast surface roughness",	"m",	"FSR"),
    245:	("Forecast log of surface roughness for heat",	"-",	"FLSR"),
    246:	("Cloud liquid water content",	"kg kg**-1",	"CLWC"),
    247:	("Cloud ice water content",	"kg kg**-1",	"CIWC"),
    248:	("Cloud cover",	"(0 - 1)",	"CC"),
    249:	("Accumulated ice water tendency",	"(-1 to 1)",	"AIW"),
    250:	("Ice age",	"1,0",	"ICE"),
    251:	("Adiabatic tendency of temperature",	"K",	"ATTE"),
    252:	("Adiabatic tendency of humidity",	"kg kg**-1",	"ATHE"),
    253:	("Adiabatic tendency of zonal wind",	"m s**-1",	"ATZE"),
    254:	("Adiabatic tendency of meridional wind",	"m s**-1",	"ATMW"),
    255:	("Indicates a missing value",	"-",	"-")

  }

}

params = {
  98: ECMWF
}


