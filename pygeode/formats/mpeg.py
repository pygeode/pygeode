# Encode a 3D variable as a movie file (mpeg format)
# Requires 'mencoder' application

# Note: no colourbar, titles, or labels of any kind are used.
#TODO: add these somehow?

def save (filename, var, iaxis=None, fps=15, palette='bw', minmax=None):
  from pygeode.axis import TAxis
  from pygeode.var import Var
  from pygeode.progress import PBar
  import tempfile, shutil
  import Image
  import numpy as np
  import os

  assert isinstance(var, Var)

  # Remove any degenerate dimensions, make sure the axes are in a consistent order
  var = var.squeeze().sorted()
  assert var.naxes == 3, "can only work with 3D data"
  
  if iaxis is None: iaxis = var.whichaxis(TAxis)
  assert iaxis >= 0, "no time axis found"

  tmpdir = tempfile.mkdtemp (prefix='pygeode_mpeg')
  sl = [slice(None)] * 3

  # Get max & min values of the whole dataset
  if minmax is None:
    #TODO: calculate both of these at once, with a progress bar to help the process
    min = float(var.min())
    max = float(var.max())
  else:
    assert len(minmax) == 2, "invalid minmax argument"
    min, max = minmax

  print "Saving %s:"%filename
  pbar = PBar()

  # Loop over each timestep, generate a temporary image file
  for i in range(len(var.axes[iaxis])):
    fpbar = pbar.part(i,len(var.axes[iaxis]))
    sl[iaxis] = i
    # Get data, flip y axis, add an 'RGB' axis
    data = var[sl].squeeze()[::-1,:,np.newaxis]
    data =  (data-min)/(max-min) * 255
    if palette == 'bw':
      # Same data for R, G, and B channels
      data = np.concatenate([data,data,data], axis=2)
    elif palette == 'rainbow':
      # Piecewise linear palette
      part1 = data <= 85
      part2 = (85 < data) & (data <= 170)
      part3 = 170 < data
      b = np.zeros(data.shape)
      b[part1] = 255
      b[part2] = 255 - (data[part2] - 85)*3
      g = np.zeros(data.shape)
      g[part1] = data[part1] * 3
      g[part2] = 255
      g[part3] = 255 - (data[part3] - 170) * 3
      r = np.zeros(data.shape)
      r[part2] = (data[part2] - 85) * 3
      r[part3] = 255

      data = np.concatenate([r,g,b], axis=2)

    # Encode as an 8-bit array
    data = np.asarray(np.round(data), 'uint8')
    # Save
    framefile = tmpdir+"/frame%04d.jpg"%i
    Image.fromarray(data,"RGB").save(framefile, quality=95)
#    os.system("display "+framefile)
#    break
    fpbar.update(100)

  shape = list(var.shape)
  shape = shape[:iaxis] + shape[iaxis+1:]
  h, w = shape

#  """
  # Make the movie file
  os.system("mencoder mf://%s/*.jpg -mf w=%s:h=%s:type=jpg:fps=%s \
          -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=8000 -oac copy \
           -o %s" % (tmpdir, w, h, fps, filename)    )
#  """

  # Clean up files
  shutil.rmtree (tmpdir)
