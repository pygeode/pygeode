from pygeode._version import __version__

import os
from os.path import sep

libpath = os.getenv('PYGEODELIBPATH', os.path.dirname(__file__))
__path__.append(libpath)

pluginpath = os.getenv('PYGEODEPLUGINS', libpath+sep+'plugins')
__path__.append(pluginpath)

_config, _configfiles = None, None

def readConfig():
# {{{
  try:
    import configparser as Cfg
  except ImportError:
    # Python2
    import ConfigParser as Cfg
  import sys, os
  from os.path import expanduser, dirname, sep
  global _config, _configfiles

  if sys.platform.startswith('linux'):
    cfgpaths = ['/etc/', '/usr/local/etc/', dirname(__file__) + sep, \
                expanduser('~') + '/.config/pygeode/', os.curdir + sep]
  else:
    cfgpaths = [dirname(__file__) + sep, expanduser('~') + sep, os.curdir + sep]
    
  try:
    c = Cfg.ConfigParser(interpolation=None)
  except TypeError:
    # Python 2.7
    c = Cfg.ConfigParser()

  files = c.read([p + 'pygrc' for p in cfgpaths])
  _config = c
  _configfiles = files
  #return c, files
# }}}

readConfig()

del os, sep

# Allow PyGeode stuff from other (local) directory structures in PYTHONPATH
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
del extend_path


# Global parameters
# Maximum size allowed for arrays in memory
MAX_ARRAY_SIZE = _config.getint('Memory', 'max_array_size')

# Maximum size allowed for arrays in temp files
# (Not currently used, but could be useful for 'medium' sized intermediate
#  products which are too big to fit in memory, but are a pain in the ass to
#  recalculate over again.)
MAX_SWAP_SIZE = _config.getint('Memory', 'max_swap_size')


# Shortcuts

__all__ = ['libpath', 'pluginpath', 'MAX_ARRAY_SIZE', 'MAX_SWAP_SIZE']  # more will be added below

# Classes
from pygeode.dataset import Dataset
from pygeode.var import Var
from pygeode.axis import standard_axes
for A in standard_axes:
  globals()[A.__name__] = A
from pygeode.timeaxis import *
from pygeode.timeaxis import __all__ as taxis_all
__all__.extend(taxis_all)

# Static methods

# Top-level concat method, which can work on both Datasets and Vars.
def concatenate(*items, **kwargs):
  """
  Concatenates multiple Var or Dataset objects together.
  """
  from pygeode.tools import islist
  from pygeode.var import Var
  from pygeode.dataset import Dataset, concat as concat_datasets
  from .concat import concat as concat_vars
  # Items already wrapped as a list (now inside another list)?
  if len(items) == 1 and islist(items[0]):
    return concatenate(*items[0], **kwargs)
  if isinstance(items[0],Var):
    return concat_vars(*items, **kwargs)
  if isinstance(items[0],Dataset):
    return concat_datasets(*items, **kwargs)
  raise TypeError("Unable to concatenate objects of type '%s'"%type(items[0]))

from pygeode.ufunc import *
from pygeode.ufunc import __all__ as ufunc_all
__all__.extend(ufunc_all)

# Plotting
from pygeode.plot import *
from pygeode.plot import __all__ as plot_all
__all__.extend(plot_all)

from pygeode.climat import *
from pygeode.climat import __all__ as climat_all
__all__.extend(climat_all)

from pygeode.eof import EOF
__all__.append('EOF')

from pygeode.svd import SVD
__all__.append('SVD')

from pygeode.ensemble import ensemble
__all__.append('ensemble')

from pygeode.stats import *
from pygeode.stats import __all__ as stats_all
__all__.append(stats_all)

from pygeode.spectral import *
from pygeode.spectral import __all__ as spectral_all
__all__.extend(spectral_all)

# Dataset shortcuts
from pygeode.dataset import asdataset

#from pygeode.composite import composite

# File I/O
try:
  from pygeode.formats import netcdf
except Exception:
  print("Warning: can't import netcdf module")
from pygeode.formats import *

#### Dynamic shortcuts to plugins ####

from glob import glob
from os.path import join, sep
import sys
from . import plugins
for _plugin_path in plugins.__path__:
  _plugins = glob(join(_plugin_path,"*",""))
  for _plugin in _plugins:
    _plugin = _plugin.split(sep)[-2]
    # Trigger the __init__ for the plugin
    exec("from pygeode.plugins import %s as _x"%_plugin)
    del _x, _plugin

del glob, join, sep, _plugin_path, _plugins, sys, A

