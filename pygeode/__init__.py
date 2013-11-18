import os
from os.path import sep

libpath = os.getenv('PYGEODELIBPATH', os.path.dirname(__file__))
__path__.append(libpath)

pluginpath = os.getenv('PYGEODEPLUGINS', libpath+sep+'plugins')
__path__.append(pluginpath)

del os, sep

# Allow PyGeode stuff from other (local) directory structures in PYTHONPATH
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
del extend_path


# Global parameters
# Maximum size allowed for arrays in memory
MAX_ARRAY_SIZE = 2**22
#MAX_ARRAY_SIZE=2**10
#MAX_ARRAY_SIZE=2**5

# Maximum size allowed for arrays in temp files
# (Not currently used, but could be useful for 'medium' sized intermediate
#  products which are too big to fit in memory, but are a pain in the ass to
#  recalculate over again.)
MAX_SWAP_SIZE = 2**30


# Shortcuts

__all__ = ['libpath', 'pluginpath', 'MAX_ARRAY_SIZE', 'MAX_SWAP_SIZE']  # more will be added below

# File I/O
try:
  from pygeode.formats import netcdf
except Exception:
  print "Warning: can't import netcdf module"
from pygeode.formats import *

# Classes
from pygeode.dataset import Dataset
from pygeode.var import Var
from pygeode.axis import standard_axes
for A in standard_axes:
  globals()[A.__name__] = A
from pygeode.timeaxis import StandardTime, ModelTime365, ModelTime360, Yearless

# Static methods

#TODO: concat method, which can work on both Datasets and Vars?

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

#from pygeode.composite import composite

#### Dynamic shortcuts to plugins ####

from glob import glob
from os.path import join, sep
import sys
import plugins
for _plugin_path in plugins.__path__:
  _plugins = glob(join(_plugin_path,"*",""))
  for _plugin in _plugins:
    _plugin = _plugin.split(sep)[-2]
    # Trigger the __init__ for the plugin
    exec "from pygeode.plugins import %s as _x"%_plugin
    del _x, _plugin

del glob, join, sep, _plugin_path, _plugins, sys

