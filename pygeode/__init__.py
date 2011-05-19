import os
from os.path import sep

libpath = os.getenv('PYGEODELIBPATH', os.path.dirname(__file__))

pluginpath = os.getenv('PYGEODEPLUGINS', libpath+sep+'plugins')
__path__.append(pluginpath)

del os

#__tmppath__ = "/tmp/" +  "pygeode-" + os.getlogin()
#if not os.path.exists(__tmppath__):
#  os.mkdir(__tmppath__)
#else: assert os.path.isdir(__tmppath__)

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
from pygeode.formats import netcdf
from pygeode.formats.multifile import openall, open_multi

# Classes
from pygeode.dataset import Dataset
from pygeode.var import Var
from pygeode.axis import standard_axes
for A in standard_axes:
  globals()[A.__name__] = A
from pygeode.timeaxis import StandardTime, ModelTime365, Yearless

# Static methods

#TODO: concat method, which can work on both Datasets and Vars?

from pygeode.ufunc import *
from pygeode.ufunc import __all__ as ufunc_all
__all__.extend(ufunc_all)

# Plotting
from pygeode.plot import plotvar, plotquiver
__all__.extend(['plotvar', 'plotquiver'])

from pygeode.climat import *
from pygeode.climat import __all__ as climat_all
__all__.extend(climat_all)

from pygeode.eof import EOF
__all__.append('EOF')

from pygeode.svd import SVD
__all__.append('SVD')

from pygeode.ensemble import ensemble
__all__.append('ensemble')

#from pygeode.composite import composite
