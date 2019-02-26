# Set of helper routines for contour plots

import numpy as np
from numpy import ma
import matplotlib.cbook as cbook
from matplotlib.colors import Normalize
from pylab import cm as pcm
from . import cm

class LogNorm2Sided(Normalize):
 # {{{
  """
  Normalize a given value to the 0-1 range on a two-sided log scale;
    values from vmin to -vin are mapped logarithmically from 0 to 0.5-cin;
    values from -vin to +vin are mapped linearly from 0.5-cin to 0.5+cin,
    and values from +vin to vmax are mapped logarithmically from 0.5+cin to 1.   
  """
  def __init__(self, vmin=None, vmax=None, clip=False, vin=None, cin=0.01):
    self.vin = vin
    self.cin = cin
    Normalize.__init__(self, vmin, vmax, clip)
   
  def __call__(self, value, clip=None):
    if clip is None:
      clip = self.clip

    if cbook.iterable(value):
      vtype = 'array'
      val = ma.asarray(value).astype(np.float)
    else:
      vtype = 'scalar'
      val = ma.array([value]).astype(np.float)

    self.autoscale_None(val)
    vmin, vmax = self.vmin, self.vmax
    vin, cin = self.vin, self.cin
    if vmin > vmax:
      raise ValueError("minvalue must be less than or equal to maxvalue")
    elif vmin > 0:
      raise ValueError("minvalue must be less than 0")
    elif vmax < 0:
      raise ValueError("maxvalue must be greater than 0")
    elif vmin==vmax:
      result = 0.0 * val
    else:
      if clip:
        mask = ma.getmask(val)
        val = ma.array(np.clip(val.filled(vmax), vmin, vmax),
                        mask=mask)
      ipos = (val > vin)
      ineg = (val < -vin)
      izero = ~(ipos | ineg)

      result = ma.empty_like(val)
      result[izero] = 0.5 + cin * val[izero] / vin
      result[ipos] = 0.5 + cin + (0.5 - cin) * \
                    (ma.log(val[ipos]) - np.log(vin)) / (np.log(vmax) - np.log(vin))
      result[ineg] = 0.5 - cin - (0.5 - cin) * \
                    (ma.log(-val[ineg]) - np.log(vin)) / (np.log(-vmin) - np.log(vin))
      result.mask = ma.getmask(val)
    if vtype == 'scalar':
      result = result[0]
    return result

  def inverse(self, value):
    if not self.scaled():
      raise ValueError("Not invertible until scaled")
    vmin, vmax = self.vmin, self.vmax
    vin, cin = self.vin, self.cin

    if cbook.iterable(value):
      val = ma.asarray(value)
      ipos = (val > (0.5 + cin))
      ineg = (val < (0.5 - cin))
      izero = ~(ipos | ineg)

      result = ma.empty_like(val)
      result[izero] = (val[izero] - 0.5) * vin/cin
      result[ipos] = vin * pow((vmax/vin), (val[ipos] - (0.5 + cin))/(0.5 - cin)) 
      result[ineg] = -vin * pow((-vmin/vin), ((0.5 - cin) - val[min])/(0.5 - cin))
                    
      r = vmin * ma.power((vmax/vmin), val)
    else:
      if value > 0.5 + cin: r = vin * pow((vmax/vin), (value - (0.5 + cin))/(0.5 - cin))
      elif value < 0.5 - cin: r = -vin * pow((-vmin/vin), ((0.5 - cin) - value)/(0.5 - cin))
      else: r = (value - 0.5) * vin / cin
    return r

  def autoscale(self, A):
    '''
    Set *vmin*, *vmax* to min, max of *A*.
    '''
    A = ma.masked_less_equal(np.abs(A), 1e-16, copy=False)
    self.vmin = -ma.max(A)
    self.vmax = ma.max(A)
    self.vin = ma.min(A)

  def autoscale_None(self, A):
    ' autoscale only None-valued vmin or vmax'
    if self.vmin is not None and self.vmax is not None and self.vin is not None:
      return
    A = ma.masked_less_equal(np.abs(A), 1e-16, copy=False)
    if self.vmin is None:
      self.vmin = -ma.max(A)
    if self.vmax is None:
      self.vmax = ma.max(A)
    if self.vin is None:
      self.vin = ma.min(A)
 # }}}

def guessclimits(z, style=None, ndiv=None, clf=True):
# {{{
  ''' Guesses a round contour interval and style to display
  the array z. '''

  if style not in [None, 'div', 'seq']:
    raise ValueError("style '%s' not recognized. Must be one of None, 'div', or 'seq'." % style)

  if ndiv is not None and ndiv <= 0.:
    raise ValueError("ndiv must be None or an integer greater than 0.")

  zn = z.ravel()
  zn = zn[np.isfinite(zn)]

  if len(zn) < 1: 
    if style is None: style = 'div'
    if style == 'div': 
      if ndiv is None: ndiv = 3
      kws = dict(ndiv=ndiv, style=style)
    elif style == 'seq':
      if ndiv is None: ndiv = 5
      kws = dict(ndiv=ndiv, min=0., style=style)

    return 1., kws

  pcs = [0, 2., 50, 98, 100]
  mn, p2, md, p98, mx = np.percentile(zn, pcs)

  dmn = md - mn
  dp2 = md - p2
  dp98 = p98 - md
  dmx = mx - md

  # If ratio of the spacing between the minimum/maximum and the median
  # and the 2nd/98th percentile and the median are too large, 
  # consider the minimum/maximum values outliers, do not include them
  # in the range
  if mx - mn > 0.:
    if dp2 > 0 and dmn / dp2  > 50:
      dl = dp2
      lo = p2
    else:
      dl = dmn
      lo = mn
    if dp98 > 0 and dmx / dp98 > 50:
      dm = dp98
      hi = p98
    else:
      dm = dmx
      hi = mx
  else:
    lo = mn - 1.
    dl = 1.0
    dm = 1.0
    hi = mn + 1.
    if style is None: style = 'div'

  if style is None: # Guess style of cmap to use
    if p2 < 0 and p98 > 0: style = 'div'
    else: style = 'seq'

  if style is 'div':
    if ndiv is None: 
      if clf: ndiv = 3
      else: ndiv = 10

    div = max(-lo, hi) / ndiv

    # Handle degenerate case
    if div == 0.: return 1., ndiv, style

    p = np.floor(np.log10(np.abs(div)))
    m = div / 10**p
    
    if m > 6.: spc = 10.
    elif m > 5.: spc = 6.
    elif m > 4.: spc = 5.
    elif m > 3.: spc = 4.
    elif m > 2.: spc = 3.
    elif m > 1.5: spc = 2.
    elif m > 1.: spc = 1.5
    else: spc = 1.

    #print style, div, p, m, spc

    spc *= 10**p
    if clf: kws = dict(ndiv=ndiv, style=style)
    else: kws = dict(range=ndiv*spc)

  elif style is 'seq':
    if ndiv is None: 
      if clf: ndiv = 5
      else: ndiv = 10

    div = (dm + dl) / ndiv

    p = np.floor(np.log10(np.abs(div)))
    m = div / 10**p
    
    if m > 6.: spc = 10.
    elif m > 5.: spc = 6.
    elif m > 4.: spc = 5.
    elif m > 3.: spc = 4.
    elif m > 2.: spc = 3.
    elif m > 1.5: spc = 2.
    else: spc = 1.5

    #print style, p, m, spc

    spc *= 10**p
    min = lo - np.remainder(lo, spc)
    if clf: kws = dict(ndiv=ndiv, min=min, style=style)
    else: kws = dict(range=ndiv*spc/2., min=min)
    
  return spc, kws
# }}}

def cldict(cdelt, range=None, min=None, mid=0, cidelt=0., nozero=False, **kwargs):
# {{{
  ''' Returns kwargs to :meth:`showvar` for a line contour plot.
  '''
  if nozero: cidelt = 2*cdelt
  if range is None: range = 10*cdelt

  if cidelt > 0.:
    cl = np.concatenate([np.arange(-range, -cdelt, cdelt),
                         np.arange(-cdelt, cdelt, cidelt),
                         np.arange(cdelt, range + cdelt/2., cdelt)])
  elif min is None:
    cl = np.arange(mid - range, mid + range + cdelt/2, cdelt)
  else:
    cl = np.arange(min, min + 2 * range + cdelt/2, cdelt)

  kwa = dict(clevs = None,
              clines = cl,
              linewidths = 1.,
              colors='k')
  kwa.update(kwargs)
  return kwa
# }}}

def clfdict(cdelt, min=None, mid=0., nf=6, nl=2, ndiv=3, nozero=False, style='div', clr=True, **kwargs):
# {{{
  ''' 
  Returns kwargs to :meth:`showvar` for a filled contour plot.

  Parameters
  ----------
  cdelt : float
    Spacing of a single division. Each division is spanned by a certain number of filled
    contours and contour lines, and the colorbar will span a set number of divisions. See notes.

  min : float or None (optional)
    If specified, contours span ndiv equal divisions starting with the value
    min. If None, the colorbar is centred on the value of mid. Default is None.

  mid : float (optional)
    If min is None, contours span ndiv equal divisions both above and below the value of mid.

  nf : integer (optional)
    Number of filled contours per division. Default is 6.

  nl : integer (optional)
    Number of contour lines per division. Default is 2.

  nozero : boolean (optional)
    If True, the contour line at mid is omitted. Defalt is False.

  style : string (optional)
    Either 'seq' or 'div'. Deterimines which style of colourmap to use; a
    sequential or divergent. Default is 'div'.
  '''

  if not style in ['div', 'seq']: raise ValueError("style must be one of 'div' or 'seq'")

  if style == 'div': crange = float(cdelt * ndiv * 2)
  else: crange = float(cdelt * ndiv)

  if min is not None:
    min = float(min)
    cmin = min
    cmax = min + crange
    mid = (cmin + cmax) / 2.
  else:
    cmin = mid - crange / 2.
    cmax = mid + crange / 2.

  cmp = cm.get_cm(style, ndiv)

  if style == 'div':
    tks = np.linspace(cmin, cmax, 2*ndiv + 1)
    cf = np.concatenate([np.linspace(cmin, mid, nf * ndiv + 1)[:-1],
                         np.linspace(mid, cmax, nf * ndiv + 1)[1:]])
    if nozero:
      cl = np.concatenate([np.linspace(cmin, 0, nl * ndiv + 1)[:-1],
                           np.linspace(0, cmax, nl * ndiv + 1)[1:]])
    else:
      cl = np.linspace(cmin, cmax, 2 * nl * ndiv + 1)
  else:
    tks = np.linspace(cmin, cmax, ndiv + 1)
    cf = np.linspace(cmin, cmax, nf * ndiv + 1)
    cl = np.linspace(cmin, cmax, nl * ndiv + 1)

  if nl == 0: cl = None

  kwcb = kwargs.pop('colorbar', {})
  cb = dict(ticks = tks)
  if kwcb is False:
    cb = False
  else:
    cb.update(kwcb)

  if not clr: 
    cmp = pcm.gray
    cb = False

  kwa = dict(clevs = cf,
              clines = cl,
              colorbar = cb,
              linewidths = 1.,
              cmap=cmp)
  kwa.update(kwargs)
  return kwa
# }}}

def lfmt(x, pos=None):
# {{{
  x = float(x)
  if x == 0.: return '0'

  e = float(np.floor(np.log10(np.abs(x))))
  if abs(e) < 2:
    return '%.1f' % x
  else: 
    return r'%d$\!\times$10$^{%d}$' % (np.round(x/10**e), e)
# }}}

def log1sdict(cmin, cdelt = 10., nf=6, nl=2, ndiv=5, **kwargs):
# {{{
  ''' Returns kwargs to :meth:`showvar` for a one-sided
  logarithmically-spaced contour plot. '''
  from matplotlib.colors import LogNorm
  cmax = cmin * cdelt ** ndiv

  tks = cmin * cdelt ** np.arange(0., ndiv+1)
  if nf > 0:
    cf = cmin * cdelt ** np.linspace(0., ndiv, nf*ndiv+1)
  else:
    cf = None

  if nl > 0: 
    cl = cmin * cdelt ** np.linspace(0., ndiv, nl*ndiv+1)
  else:
    cl = None

  nrm = LogNorm(vmin=cmin, vmax=cmax)

  cmp = cm.get_cm('seq', ndiv)

  kwcb = kwargs.pop('colorbar', {})
  cb = dict(ticks=tks, width=1.4, rl=0.01, rr=0.16, ticklabels = [lfmt(x) for x in tks])
  if kwcb is False:
    cb = False
  else:
    cb.update(kwcb)

  kwa = dict(clevs = cf,
              clines = cl,
              norm = nrm,
              colorbar = cb,
              cmap=cmp)
  kwa.update(kwargs)
  return kwa
# }}}

def log2sdict(cmin, cdelt = 10, nf=6, nl=2, ndiv=3, nozero=False, **kwargs):
# {{{
  ''' Returns kwargs to :meth:`showvar` for a two-sided logarithmically-spaced contour plot. '''

  ce = cmin * cdelt ** np.linspace(0, ndiv-1, nf*(ndiv - 1)+1)
  cel = cmin * cdelt ** np.linspace(0, ndiv-1, nl*(ndiv - 1)+1)
  tks = cmin * cdelt ** np.arange(ndiv)
  cmax = tks[-1]

  tks = np.concatenate([-tks[::-1], [0.], tks])

  if nf > 0:
    #ci = np.linspace(-cmin, cmin, 2*nf+1)[1:-1]
    ci = np.concatenate([np.linspace(-cmin, 0, nf + 1)[1:-1],
                         np.linspace(0,  cmin, nf + 1)[1:-1]])
    cf = np.concatenate([-ce[::-1], ci, ce])
  else:
    cf = None

  if nl > 0: 
    if nozero:
      cil = np.concatenate([np.linspace(-cmin, 0, nl+1)[1:-1], np.linspace(0, cmin, nl+1)[1:-1]])
    else:
      cil = np.linspace(-cmin, cmin, 2*nl+1)[1:-1]
    cl = np.concatenate([-cel[::-1], cil, cel])
  else:
    cl = None

  cin = 1. / (2 * ndiv)
  nrm = LogNorm2Sided(vmin=-cmax, vmax=cmax, vin=cmin, cin=cin)

  cmp = cm.get_cm('div', ndiv)

  kwcb = kwargs.pop('colorbar', {})
  cb = dict(ticks=tks, width=1.4, rl=0.01, rr=0.16, ticklabels = [lfmt(x) for x in tks])
  cb.update(kwcb)
  if kwcb is False:
    cb = False
  else:
    cb.update(kwcb)

  kwa = dict(clevs = cf,
              clines = cl,
              norm = nrm,
              colorbar = cb,
              cmap=cmp)
  kwa.update(kwargs)
  return kwa
# }}}

__all__ = ['clfdict', 'cldict', 'log1sdict', 'log2sdict', 'cm']
