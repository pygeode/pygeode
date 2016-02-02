''' Module of plotting routines for pygeode variables. '''

__all__ = []

# Import old style plotting routines
from pygeode.plot.plot_v1 import plotvar, plotquiver, plotsigmask
__all__.extend(['plotvar', 'plotquiver', 'plotsigmask'])

# Import new plotting utils
from pygeode.plot.wrappers import *

from pygeode.plot.pyg_helpers import *
from pygeode.plot.pyg_helpers import __all__ as phelp_all
__all__.extend(phelp_all)

from pygeode.plot.cnt_helpers import *
from pygeode.plot.cnt_helpers import __all__ as chelp_all
__all__.extend(chelp_all)
