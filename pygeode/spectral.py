# Code for handling spherical harmonics -- makes use of pyspharm python wrapper of SPHEREPACK
# to do computations

from pygeode.axis import Axis
import numpy as np

class Spectral(Axis):
# {{{
  name = 'spharm' 
  formatstr = '%d'
  plotatts = Axis.plotatts.copy()
  plotatts['plotname'] = 'Spherical Harm.'

  def __init__ (self, values, trunc=None, M=None, N=None, **kwargs):
  # {{{
    if isinstance(values, int):
      trunc = values
      values = np.arange((trunc+1)*(trunc+2)/2)

    if trunc is None:
      # Infer triangular truncation by length of values
      trunc = int(np.sqrt(len(values) * 2 + 0.25) - 1.5)

    if N is None:
      N = np.concatenate([np.arange(n, trunc+1) for n in np.arange(0, trunc+1)])
      M = np.concatenate([n*np.ones(trunc+1-n, 'i') for n in np.arange(0, trunc+1)])

    # Just pass all the stuff to the superclass
    Axis.__init__ (self, values, N=N, M=M, **kwargs)
  # }}}
# }}}

__all__ = ('Spectral',)
