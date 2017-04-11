import pygeode as pyg
import numpy as np
import os.path, glob

hsv_div = {4: ([0.3, 0.43, 0.52, -0.64, -0.99, 0.05, 0.12, 0.18], [0.9, 0.3], [0.5, 0.98]), \
           #3: ([0.4, 0.5, -0.556, -1.0, 0.94, 0.87], [0.96, 0.4], [0.4, 0.7]), \
           3: ([0.48, 0.38, -0.6, -1.0, 0.09, 0.15], [0.9, 0.3], [0.5, 0.98]), \
           2: ([0.55, -0.6, -1.0, 0.9], [0.96, 0.4], [0.5, 0.98]), \
           1: ([-0.6, -0.99], [0.9, 0.6], [0.5, 0.98])}
hsv_seq = {6: ([0.5, 0.556, 1.0, 0.94, 0.87, 0.82], [0.96, 0.4], [0.4, 0.7]), \
           #5: ([0.5, 0.556, 1.0, 0.94, 0.87], [0.96, 0.4], [0.4, 0.7]), \
           #5: ([0.62, 0.75, 0.93, 0.05, 0.13], [0.96, 0.3], [0.5, 0.99]), \
           5: ([0.6, 0.38, 0.18, 0.08, 0.01], [1., 0.3], [0.5, 0.95]), \
           4: ([0.56, 0.72, 0.83, 0.0], [0.96, 0.3], [0.5, 0.99]), \
           3: ([0.5, 0.556, 1.0], [0.96, 0.4], [0.4, 0.7]),
           2: ([0.5, 0.6], [0.96, 0.4], [0.4, 0.9]), \
           1: ([0.6], [0.96, 0.2], [0.4, 0.9])}
# create defaults for centered (2, 4, 6 stops)
# uncentered (1, 2, 3, 4, 5, 6)
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
    if s.has_key('r2'):
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
    for c in ct.keys():
      ct[c] = rt(ct[c])

  return matplotlib.colors.LinearSegmentedColormap('loaded_colourmap', ct, 256)
# }}}

def get_cm(style, ndiv):
# {{{
  if style == 'seq':
    return cmap_from_hues(*hsv_seq[ndiv])
  elif style == 'div':
    return cmap_from_hues(*hsv_div[ndiv])
  else:
    raise ValueError("style must be one of 'seq' or 'div'")
# }}}

cmpdir = pyg._config.get('Plotting', 'cmaps_path')
#print cmpdir

for fn in glob.glob(cmpdir + '*.cmap'):
  name = os.path.splitext(os.path.basename(fn))[0]
  #print fn, name
  globals()[name] = read_grad(fn)
  globals()[name + '_r'] = read_grad(fn, rev=True)
