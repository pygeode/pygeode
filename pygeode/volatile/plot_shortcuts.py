# Shortcuts for plotting PyGeode vars
# Extends plot_wrapper to automatically use information from the Pygeode Vars.

# Set up arguments for axis
def get_axes_args (var, **overrides):
  kwargs = {}
  atts = dict(var.plotatts, **overrides)

  kwargs['title'] = atts.get('title',None) or var.name
  #TODO: put degenerate PyGeode axis info into title,
  # strip the degenerate axes out of the data.

  # 1D stuff
  if var.naxes == 1:
    kwargs['xlabel'] = atts.get('xlabel',None) or var.axes[0].name
    kwargs['ylabel'] = kwargs['title']

  # 2D stuff
  if var.naxes == 2:
    # For 2D plots, the x/y need to be switched?
    kwargs['xlabel'] = atts.get('xlabel',None) or var.axes[1].name
    kwargs['ylabel'] = atts.get('ylabel',None) or var.axes[0].name

  #TODO: linear/log scale, reversed order

  return kwargs

# Get 2D data
# (note: notion of 'x' and 'y' are switched for matplotlib 2D plots...)
def get_XYC (var):
  X = var.axes[1].get()
  Y = var.axes[0].get()
  C = var.get()
  return X, Y, C


# Do a contour plot
def make_contour (var, **kwargs):
  from pygeode.plot_wrapper import Plot
  X, Y, C = get_XYC(var)
  return Plot('contour', [X,Y,C], **get_axes_args(var, **kwargs))
#"""

# Do a filled contour plot
def make_contourf (var, **kwargs):
  from pygeode.plot_wrapper import Plot
  X, Y, C = get_XYC(var)
  return Plot('contourf', [X,Y,C], **get_axes_args(var, **kwargs))

# Do a pseudocolor plot
def make_pcolor (var, **kwargs):
  from pygeode.plot_wrapper import Plot
  X, Y, C = get_XYC(var)
  return Plot('pcolor', [X,Y,C], **get_axes_args(var, **kwargs))



# Shortcuts for plotting
def contour (var):
  return make_contour(var).render()

def contourf (var):
  return make_contourf(var).render()

def pcolor (var):
  return make_pcolor(var).render()


