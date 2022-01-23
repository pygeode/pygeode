import pygeode as pyg
import numpy as np
import os.path, glob
from matplotlib import cm as mpl_cm

cmaps_div = {4: ([0.3, 0.43, 0.52, -0.64, -0.99, 0.05, 0.12, 0.18], [0.9, 0.3], [0.5, 0.98]), \
            #3: ([0.4, 0.5, -0.556, -1.0, 0.94, 0.87], [0.96, 0.4], [0.4, 0.7]), \
             3: ([0.48, 0.38, -0.6, -1.0, 0.09, 0.15], [0.9, 0.3], [0.5, 0.98]), \
             2: ([0.55, -0.6, -1.0, 0.9], [0.96, 0.4], [0.5, 0.98]), \
             1: ([-0.6, -0.99], [0.9, 0.6], [0.5, 0.98]), \
     'default': 'RdBu'}

cmaps_seq = {6: ([0.5, 0.556, 1.0, 0.94, 0.87, 0.82], [0.96, 0.4], [0.4, 0.7]), \
            #5: ([0.5, 0.556, 1.0, 0.94, 0.87], [0.96, 0.4], [0.4, 0.7]), \
            #5: ([0.62, 0.75, 0.93, 0.05, 0.13], [0.96, 0.3], [0.5, 0.99]), \
             5: ([0.6, 0.38, 0.18, 0.08, 0.01], [1., 0.3], [0.5, 0.95]), \
             4: ([0.56, 0.72, 0.83, 0.0], [0.96, 0.3], [0.5, 0.99]), \
             3: ([0.5, 0.556, 1.0], [0.96, 0.4], [0.4, 0.7]),
             2: ([0.5, 0.6], [0.96, 0.4], [0.4, 0.9]), \
             1: ([0.6], [0.96, 0.2], [0.4, 0.9]), \
     'default': 'inferno'}

# put them in config file?
# colour brewer options?

def cmap_from_hues(hues=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], sat=[0.2, 0.9], rval=[0.3, 0.6]):
# {{{
  stops = build_stops(hues, sat, rval)
  return build_cmap(stops)
# }}}

def build_cmap(stops): 
# {{{ 
  from matplotlib.colors import LinearSegmentedColormap
  ct = dict(red=[], blue=[], green=[])
  for s in stops:
    if 'r2' in s:
      ct['red'].append((s['s'], s['r'], s['r2']))
      ct['green'].append((s['s'], s['g'], s['g2']))
      ct['blue'].append((s['s'], s['b'], s['b2']))
    else:
      ct['red'].append((s['s'], s['r'], s['r']))
      ct['green'].append((s['s'], s['g'], s['g']))
      ct['blue'].append((s['s'], s['b'], s['b']))
  cm = LinearSegmentedColormap('loaded_colourmap', ct, 256)
  cm.set_under((stops[ 0]['r'], stops[ 0]['g'], stops[ 0]['b']))
  cm.set_over ((stops[-1]['r'], stops[-1]['g'], stops[-1]['b']))
  return cm
# }}}

def build_stops(hues, sat, rval):
# {{{
  from colorsys import hsv_to_rgb
  nstops = len(hues) + 1
  stops = [dict(s=S) for S in np.linspace(0, 1, nstops)]
  flipped = False
  j = [0, 1]
  for i, h in enumerate(hues):
    r1, g1, b1 = hsv_to_rgb(np.fabs(h), sat[j[0]], rval[j[0]])
    r2, g2, b2 = hsv_to_rgb(np.fabs(h), sat[j[1]], rval[j[1]])
    if h < 0.:
      if flipped:
        r1, g1, b1 = 1., 1., 1.
      else:
        r2, g2, b2 = 1., 1., 1.
        j = [1, 0]
        flipped = True

    if i == 0:
      stops[i].update(dict(r=r1, g=g1, b=b1))
    else:
      stops[i].update(dict(r2=r1, g2=g1, b2=b1))
    stops[i+1].update(dict(r=r2, g=g2, b=b2))

  return stops
# }}}

def read_grad(path, rev=False):
# {{{
  import matplotlib
  import re
  ct = dict(red=[], blue=[], green=[])
  f = open(path)
  pattern = r'color-stop\((?P<stop>[0-9]\.?[0-9]*), rgb\((?P<r>[0-9]*),(?P<g>[0-9]*),(?P<b>[0-9]*)\)'
  pattern2 = r'color-stop\((?P<stop>[0-9]\.?[0-9]*), rgb\((?P<r>[0-9]*),(?P<g>[0-9]*),(?P<b>[0-9]*)\), rgb\((?P<r2>[0-9]*),(?P<g2>[0-9]*),(?P<b2>[0-9]*)\)'

  st = lambda g: float(g['stop'])
  c = lambda g, cl: float(g[cl])/255.

  for l in f.readlines():
    d = re.match(pattern2, l)
    twoside = True
    if d is None: 
      twoside = False
      d = re.match(pattern, l)
    gd = d.groupdict()

    ctr = [st(gd), c(gd, 'r'), c(gd, 'r')]
    ctg = [st(gd), c(gd, 'g'), c(gd, 'g')]
    ctb = [st(gd), c(gd, 'b'), c(gd, 'b')]

    if twoside:
      ctr[2] = c(gd, 'r2')
      ctg[2] = c(gd, 'g2')
      ctb[2] = c(gd, 'b2')

    ct['red'].append(tuple(ctr))
    ct['green'].append(tuple(ctg))
    ct['blue'].append(tuple(ctb))

  if rev:
    rt = lambda ctc: [(1.-c[0], c[2], c[1]) for c in ctc[::-1]]
    for c in list(ct.keys()):
      ct[c] = rt(ct[c])

  return matplotlib.colors.LinearSegmentedColormap('loaded_colourmap', ct, 256)
# }}}

def get_cm(style, ndiv):
# {{{
  from matplotlib.colors import Colormap

  if style == 'seq':
    cm = cmaps_seq.get(ndiv, cmaps_seq['default'])
  elif style == 'div':
    cm = cmaps_div.get(ndiv, cmaps_div['default'])
  else:
    raise ValueError("style must be one of 'seq' or 'div'")

  if type(cm) is tuple:
    return cmap_from_hues(*cm)
  elif isinstance(cm, Colormap):
    return cm
  else:
    return mpl_cm.get_cmap(cm)
# }}}

def read_config():
# {{{
  from warnings import warn

  cmpdir = pyg._config.get('Plotting', 'cmaps_path')

  for fn in glob.glob(cmpdir + '*.cmap'):
    name = os.path.splitext(os.path.basename(fn))[0]
    #print fn, name
    globals()[name] = read_grad(fn)
    globals()[name + '_r'] = read_grad(fn, rev=True)

  def parse_cmaps(param):
    dct = {}

    cstrs = [d.strip() for d in divstr.split('\n') if len(d) > 0]
    for cstr in cstrs:
      key, val = [c.strip() for c in cstr.split(':')]

      if not key == 'default':
        key = int(key)

      if val[0] == '(':
        val = eval(val)

      dct[key] = val

  # Read in divergent colormaps
  divstr = pyg._config.get('Plotting', 'cmaps_div')
  cmaps_div = parse_cmaps(divstr)
  try:
    pass
  except:
    warn('Parsing of default divergent colormaps in config files failed. See pyg._configfiles for list of configuration files in current use.')

  # Read in sequential colormaps
  seqstr = pyg._config.get('Plotting', 'cmaps_seq')
  try:
    cmaps_seq = parse_cmaps(seqstr)
  except:
    warn('Parsing of default sequential colormaps in config files failed. See pyg._configfiles for list of configuration files in current use.')
# }}}

# Read in configuration file on import
read_config()
