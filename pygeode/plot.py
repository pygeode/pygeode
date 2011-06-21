# General plotting

#TODO: check if the axes are subclasses of XAxis/YAxis/ZAxis/TAxis, and use
#      this as a hint for transposing the data?

def plotvar (var, **kwargs):
# {{{ 
  ''' plotvar(var, title, clevs, cmap, ax, ifig, hold)

  Produces a plot of the pygeode variable var. The routine can plot
  1d or 2d data; degenerate axes (of length 1) are ignored; their value is 
  displayed in the title of the plot.

  If the axes are longitude and latitude, the Basemap package is used to plot
  variable on a map of the world. 

  If one of the axes is a ZAxis, it is plotted on the y-axes, logarithmically if
  appropriate.

  keyword arguments:
    title: Title of the plot
    ax: A matplotlib axes object on which to produce the plot
    lblx: Show xaxis titles and labels 
    lbly: Show yaxis titles and labels 
    colorbar: Show colorbar
    clevs: Filled contour levels, if None, no filled contours are plotted
    cmap: A colormap passed on to the contour pylab function
    clines: Outlined levels, if None, no contour lines are plotted
    perx: Roll values in x axis (appropriate for periodic axes)
    ifig: Index of the matplotlib figure on which to produce the plot
    hold: If True, don't clear the contents of the axis
    wait: if True, don't invoke the show() command
          (the plotting main loop is not called, so subsequent pygeode commands 
           can be invoked)
  '''
  from matplotlib.pyplot import figure, show, ion, ioff, draw, cm, clf, isinteractive
###  from matplotlib.numerix import ma
  from numpy import ma
  from numpy import isnan, isinf, where
  from pygeode.progress import PBar

  # Get # of dimensions - can only do 1D or 2D
  nd = len([s for s in var.shape if s > 1])
  assert nd == 1 or nd == 2, "can only plot 1D or 2D arrays.  Try slicing along some dimensions."

  axes = var.axes
  ret = None
        
  # Create title if none has been specified
  title = kwargs.pop('title', None)
  if title is None:
    if var.plotatts.has_key('plottitle') and var.plotatts['plottitle']:
      title = var.plotatts['plottitle']
    elif var.name != '':
      title = var.name
    else: title = 'Unnamed Var'

    if var.plotatts.has_key('plotunits'): # units as an optional attribute for which a string can be provided?
      title += ' (%s)' % var.plotatts['plotunits']
      
    # Add information on degenerate axes to the title
    for a in axes:
      if len(a) == 1 and title:
        title += ', ' + a.formatvalue(a.values[0])

  pbar = kwargs.pop('pbar', True)
  if pbar is True:
#    print 'Loading plot values from %s:'%repr(var)
    pbar = PBar(message='Loading plot values from %s:'%repr(var))
    values = var.get(pbar=pbar).squeeze()
  else:
    values = var.get().squeeze()

  # Mask out missing values (NaN)
  values = ma.masked_where(isnan(values), values)
  
  # Apply linear rescaling for plotting
  values = var.plotatts.get('scalefactor',1)*values + var.plotatts.get('offset',0)

  wasint = isinteractive()
  ioff()

  ax = kwargs.pop('ax', None)
  ifig = kwargs.pop('ifig', None)
  hold = kwargs.pop('hold', False)
  wait = kwargs.pop('wait', False)
  if ax is None:
    if ifig is None:
      fig = figure()
    else:
      fig=figure(ifig)
      if not hold: clf()
      
    ax = fig.add_subplot(111)
  else:
    fig = ax.figure

  if not hold and title: ax.set_title(title)

  # 1D case:
  if nd == 1:
    from axis import ZAxis, Pres, Hybrid
    xaxis = [a for a in axes if len(a)>1][0]
    # Vertical?
    if isinstance(xaxis,ZAxis):
      lblx = kwargs.pop('lblx', False) # preserve previous behaviour
      lbly = kwargs.pop('lbly', True)
      
      ax.plot(values, xaxis.values, **kwargs)

      ax.set_yscale(xaxis.plotatts['plotscale'])
      ylims = min(xaxis.values),max(xaxis.values)
      ax.set_ylim(ylims[::xaxis.plotatts['plotorder']])
      
      # coordinate axis
      ax.yaxis.set_major_formatter(xaxis.formatter())
      if lbly:
        xaxis.set_locator(ax.yaxis)
        ax.set_ylabel(xaxis.plotatts['plottitle']+' ['+xaxis.plotatts['plotunits']+']')
      # value axis
      if lblx:
        if var.plotatts.has_key('plottitle') and var.plotatts['plottitle']: 
          varname = var.plotatts['plottitle']
        else: varname = var.name
        if var.plotatts.has_key('plotunits'): varname += ' ['+var.plotatts['plotunits']+']' 
        ax.set_xlabel(varname)
            
    else:
      lblx = kwargs.pop('lblx', True)
      lbly = kwargs.pop('lbly', False) # preserve previous behaviour
      
      ax.plot(xaxis.values, values, **kwargs)

      ax.set_xscale(xaxis.plotatts['plotscale'])
      xlims = min(xaxis.values),max(xaxis.values)
      ax.set_xlim(xlims[::xaxis.plotatts['plotorder']])

      ax.xaxis.set_major_formatter(xaxis.formatter())
      if lblx:
        xaxis.set_locator(ax.xaxis)
        ax.set_xlabel(xaxis.plotatts['plottitle']+' ['+xaxis.plotatts['plotunits']+']')      
                      
  # 2D case:
  elif nd == 2:
    from numpy import meshgrid, concatenate
    from matplotlib.pyplot import contourf, colorbar, xlim, ylim, xlabel, ylabel, gca
    from axis import Lat, Lon, ZAxis, Pres, Hybrid, SpectralM, SpectralN
    yaxis, xaxis = [a for a in axes if len(a) > 1]
 
    # Transpose vertical axis?
    if isinstance(xaxis, ZAxis):
      values = values.transpose()
      xaxis, yaxis = yaxis, xaxis
    if isinstance(xaxis, SpectralN) and isinstance(yaxis, SpectralM):
      values = values.transpose()
      xaxis, yaxis = yaxis, xaxis
    if isinstance(xaxis, Lat) and isinstance(yaxis, Lon):
      values = values.transpose()
      xaxis, yaxis = yaxis, xaxis


    perx = kwargs.pop('perx', False)
    if perx:
      xvals = concatenate([xaxis.values, [xaxis.values[-1] + (xaxis.values[1] - xaxis.values[0])]])
      yvals = yaxis.values
      meshx, meshy = meshgrid (xvals, yvals)
    else:
      xvals = xaxis.values
      yvals = yaxis.values
      meshx, meshy = meshgrid (xvals, yvals)

    #cmap = kwargs.pop('cmap', cm.gist_rainbow_r)
    clevs = kwargs.pop('clevs', 21)
    clines = kwargs.pop('clines', None)
    cbar = kwargs.pop('colorbar', {'orientation':'vertical'})
    pcolor = kwargs.pop('pcolor', False)

    mask = kwargs.pop('mask', None)
    if mask is not None:
      values = ma.masked_where(mask(values), values)
    if perx: 
      concatenate([values, values[0:1, :]], axis=0)

    #
    # Map?
    Basemap = None
    # New toolkit path
    try:
      from mpl_toolkits.basemap import Basemap
    except ImportError: pass
    # Old toolkit path
    try:
      from matplotlib.toolkits.basemap import Basemap
    except ImportError: pass

    if isinstance(xaxis,Lon) and isinstance(yaxis,Lat) and Basemap is not None:
      from numpy import arange
      
      # pop some arguments related to projection grid labelling 
      projargs = kwargs.pop('projection', {})
      # meridians setup (latitude / y)
      meridians = projargs.pop('meridians',[-180,-90,0,90,180,270,360])
      # parallels setup (longitude / x)
      parallels = projargs.pop('parallels',[-90,-60,-30,0,30,60,90])
      # show labels for meridians and parallels in given location
      # labels[0]: left, labels[1]: right, labels[2]: top, labels[3]: bottom    
      labels = projargs.pop('labels',[1,0,0,1]) 
      
      # default axes boundaries 
      bnds = {'llcrnrlat':yvals.min(),
              'urcrnrlat':yvals.max(),
              'llcrnrlon':xvals.min(),
              'urcrnrlon':xvals.max()}
      # default projection      
      proj = {'projection':'cyl', 'resolution':'l'}
      
      # read projection arguments
      proj.update(projargs)
      if proj['projection'] in ['cyl', 'merc', 'mill', 'gall']:
        bnds.update(proj)
        proj.update(bnds)
            
      # construct projection axis
      m = Basemap(ax=ax, **proj)
      m.drawcoastlines(ax=ax)      
      # draw meridians and parallels (using arguments from above) 
      m.drawmeridians(meridians,labels=labels,ax=ax)
      m.drawparallels(parallels,labels=labels,ax=ax)
      m.drawmapboundary()

      # Transform mesh
      px, py = m(meshx, meshy)

      cont = None

      # Colour individual grid boxes? (no contours)
      if pcolor:
        clevs = None  # can't have both
        cont = m.pcolor(px, py, values, **kwargs)
        ret = cont

      # Filled contours?
      if clevs is not None:
        cont = m.contourf(px, py, values, clevs, **kwargs)
        ret = cont

      # Colour bar?
      if cbar and cont is not None: 
        fig.colorbar(cont, ax=ax, **cbar)

      # Contour lines?
      if clines is not None:
        ret = m.contour(px, py, values, clines, colors='k')
    else:
      cont = None

      # Colour individual grid boxes? (no contours)
      if pcolor:
        clevs = None  # can't have both
        cont = ax.pcolor(meshx, meshy, values, **kwargs)
        ret = cont

      # Filled contours?
      if clevs is not None:
        cont = ax.contourf(meshx, meshy, values, clevs, **kwargs)
        ret = cont

      # Colour bar?
      if cbar and cont is not None: 
        fig.colorbar(cont, ax=ax, **cbar)

      # Contour lines?
      if clines is not None:
        ret = ax.contour(meshx, meshy, values, clines, colors='k')

      # Disable autoscale.  Otherwise, if we set a log scale below, then
      # the range of our axes will get screwed up.
      # (This is a 'feature' of matplotlib!)
      # http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg10527.html
      gca().set_autoscale_on(False)

      # Set the axis limits
      ax.set_xscale(xaxis.plotatts['plotscale'])
      xlims = min(xvals),max(xvals)
      ax.set_xlim(xlims[::xaxis.plotatts['plotorder']])

      ax.set_yscale(yaxis.plotatts['plotscale'])
      ylims = min(yaxis.values),max(yaxis.values)
      ax.set_ylim(ylims[::yaxis.plotatts['plotorder']])

      # Set x and y labels and formatters     
      if kwargs.pop('lblx', True):
        ax.set_xlabel(xaxis.plotatts['plottitle'])      
        ax.xaxis.set_major_formatter(xaxis.formatter())
        xaxis.set_locator(ax.xaxis)
      else:
        ax.set_xticklabels('')      
      if kwargs.pop('lbly', True):
        ax.set_ylabel(yaxis.plotatts['plottitle'])      
        ax.yaxis.set_major_formatter(yaxis.formatter())
        yaxis.set_locator(ax.yaxis)
      else:
        ax.set_yticklabels('')

  if wasint:
     ion()
     draw()
     if not wait: show()

  if ret is not None: return ret
# }}}

def plotsigmask (var, ax, **kwargs):
# {{{ 
  ''' plotsigmask(var, ax, **kwargs)

  Adds a mask of statistical significance to a contour plot. 
  var must be a 2D variable ranging from -1 to 1 where abs(v) > p 
  indicates significance at the p * 100% level. an axis object must
  be provided.
  '''

  from matplotlib.pyplot import figure, show, ion, ioff, draw, cm, clf, isinteractive, setp
  from numpy import meshgrid
  from matplotlib.pyplot import contourf, xlim, ylim, xlabel, ylabel, gca
  from axis import Lat, Lon, ZAxis, Pres, Hybrid, SpectralM, SpectralN

  # Get # of dimensions - can only do 1D or 2D
  nd = len([s for s in var.shape if s > 1])
  assert nd == 2

  axes = var.axes
  values = var.get().squeeze()

  wasint = isinteractive()
  ioff()

  yaxis, xaxis = [a for a in axes if len(a) > 1]

  # Transpose vertical axis?
  if isinstance(xaxis, ZAxis):
    values = values.transpose()
    xaxis, yaxis = yaxis, xaxis
  if isinstance(xaxis, SpectralN) and isinstance(yaxis, SpectralM):
    values = values.transpose()
    xaxis, yaxis = yaxis, xaxis

  meshx, meshy = meshgrid (xaxis.values, yaxis.values)

  shi = kwargs.pop('majsig', 0.99)
  slo = kwargs.pop('minsig', 0.95)
  sigc = ['1.', '0.7', '0.', '0.7', '1.']
  #sigc = ['0.', '0.7', '1.', '0.7', '0.']
  sigl = [-1.1, -shi, -slo, slo, shi, 1.1]
  ct = ax.contourf(meshx, meshy, values, sigl, colors=sigc, hold=True)
  setp([ct.collections[0], ct.collections[4]], visible=False)
  setp([ct.collections[1], ct.collections[3]], alpha=0.4, edgecolor='none')
  setp(ct.collections[2], alpha=0.4, edgecolor='none')
  ax.set_rasterization_zorder(0)

  # Set the axis limits
  # Pressure / eta -> log scale, reversed
  # NOTE: scaling is now specified inside the axes
  ax.set_xscale(xaxis.plotatts['plotscale'])
  ax.set_yscale(yaxis.plotatts['plotscale'])
  if isinstance(xaxis, Pres) or isinstance(xaxis, Hybrid):
    ax.set_xlim(max(xaxis.values),min(xaxis.values))
  else:
    ax.set_xlim(min(xaxis.values),max(xaxis.values))

  if isinstance(yaxis, Pres) or isinstance(yaxis, Hybrid):
    ax.set_ylim(max(yaxis.values),min(yaxis.values))
  else:
    ax.set_ylim(min(yaxis.values),max(yaxis.values))

  # Set x and y labels and formatters
  if kwargs.pop('lblx', True):
    ax.set_xlabel(xaxis.plotatts['plottitle'])      
    ax.xaxis.set_major_formatter(xaxis.formatter())
    xaxis.set_locator(ax.xaxis)
  else:
    ax.set_xticklabels('')

  if kwargs.pop('lbly', True):
    ax.set_ylabel(yaxis.plotatts['plottitle'])      
    ax.yaxis.set_major_formatter(yaxis.formatter())
    yaxis.set_locator(ax.yaxis)
  else:
    ax.set_yticklabels('')

  if wasint:
     ion()
     draw()
     if not kwargs.pop('wait', False): show()
# }}}

def plotprofiles (var,paxis,**kwargs):
# {{{ 
  ''' plotprofiles(var,paxis,**kwargs)

  Produces profiles of the pygeode variable along the axis paxis.

  If one of the axes is a ZAxis, it is plotted on the y-axes, logarithmically if
  appropriate.

  keyword arguments:
    title: Title of the plot
    ax: A matplotlib axes object on which to produce the plot
    ifig: Index of the matplotlib figure on which to produce the plot
    hold: If True, don't clear the contents of the axis
    wait: if True, don't invoke the show() command
          (the plotting main loop is not called, so subsequent pygeode commands 
           can be invoked)
  '''
# }}}

def plotquiver (vu, vv, **kwargs):
# {{{ 
  ''' plotquiver(vu, vv)

  Produces a quiver plot of the vector field vu, vv, which must be 2-dimensional
  variables.

  keyword arguments:
    title: Title of the plot
    ax: A matplotlib axes object on which to produce the plot
    ifig: Index of the matplotlib figure on which to produce the plot
    hold: If True, don't clear the contents of the axis
    wait: if True, don't invoke the show() command
          (the plotting main loop is not called, so subsequent pygeode commands 
           can be invoked)
  '''
  from matplotlib.pyplot import figure, show, ion, ioff, draw, cm, clf, isinteractive
  from matplotlib.numerix import ma
  from numpy import meshgrid
  from matplotlib.pyplot import contourf, colorbar, xlim, ylim, xlabel, ylabel, gca
  from axis import Lat, Lon, ZAxis, Pres, Hybrid, SpectralM, SpectralN
  from numpy import isnan, isinf, where

  # Get # of dimensions - can only do 1D or 2D
  ndx = len([s for s in vu.shape if s > 1])
  ndy = len([s for s in vv.shape if s > 1])
  assert ndx == 2 and ndy == 2
  assert vu.axes == vv.axes

  axes = vu.axes

  # Create title if none has been specified
  title = kwargs.pop('title', None)
  if title is None:
    if vu.plotatts.has_key('plottitle'):
      title = vu.plotatts['plottitle']
    elif vu.name != '':
      title = vu.name
    else: title = 'Unnamed Var'

    if vu.plotatts.has_key('plotunits'): # units as an optional attribute for which a string can be provided?
      title += ' (%s)' % vu.plotatts['plotunits']
    
    for a in axes:
      if len(a) == 1:
        title += ', ' + a.formatvalue(a.values[0])

  every = kwargs.pop('every', 1)
  valu = vu.squeeze()[::every, ::every]
  valv = vv.squeeze()[::every, ::every]

  # Mask out missing values (NaN)
  valu = ma.masked_where(isnan(valu), valu)
  valv = ma.masked_where(isnan(valv), valv)

  wasint = isinteractive()
  ioff()

  ax = kwargs.pop('ax', None)
  ifig = kwargs.pop('ifig', None)
  hold = kwargs.pop('hold', False)
  wait = kwargs.pop('wait', False)
  if ax is None:
    if ifig is None:
      fig = figure()
    else:
      fig=figure(ifig)
      if not hold: clf()
      
    ax = fig.add_subplot(111)
  else:
    fig = ax.figure

  ax.set_title(title)

  yaxis, xaxis = [a for a in axes if len(a) > 1]
 
  # Transpose vertical axis?
  if isinstance(xaxis, ZAxis) or (isinstance(xaxis, SpectralN) and isinstance(yaxis, SpectralM)):
    valu = valu.transpose()
    valv = valv.transpose()
    xaxis, yaxis = yaxis, xaxis

  meshx, meshy = meshgrid (xaxis.values[::every], yaxis.values[::every])
  angles = kwargs.pop('angles', 'xy')

  lblx = kwargs.pop('lblx', True)
  lbly = kwargs.pop('lbly', True)

  ax.quiver(meshx, meshy, valu, valv, units='x', angles=angles, pivot='middle', **kwargs)

  #
  # Map?
  if isinstance(xaxis,Lon) and isinstance(yaxis,Lat):
    from mpl_toolkits.basemap import Basemap
    from numpy import arange
    lons = xaxis.values
    lats = yaxis.values 
    m = Basemap(projection='cyl', \
        llcrnrlat=lats.min(), urcrnrlat=lats.max(), \
        llcrnrlon=lons.min(), urcrnrlon=lons.max(), \
        resolution='c', ax=ax)
    m.drawcoastlines(ax=ax)
    m.drawmeridians(arange(-180,361,45),ax=ax)
    m.drawmeridians([-180,-90,0,90,180,270,360],labels=[0,0,0,1],ax=ax)
    m.drawparallels([-90,-60,-30,0,30,60,90],labels=[1,0,0,0],ax=ax)

  else:
    if lblx: ax.set_xlabel(xaxis.plotatts['plottitle'])      
    if lbly: ax.set_ylabel(yaxis.plotatts['plottitle'])

    # Disable autoscale.  Otherwise, if we set a log scale below, then
    # the range of our axes will get screwed up.
    # (This is a 'feature' of matplotlib!)
    # http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg10527.html
    gca().set_autoscale_on(False)

    # Set the axis limits
    # Pressure / eta -> log scale, reversed
    # NOTE: scaling is now specified inside the axes
    ax.set_xscale(xaxis.plotatts['plotscale'])
    ax.set_yscale(yaxis.plotatts['plotscale'])
    if isinstance(xaxis, Pres) or isinstance(xaxis, Hybrid):
      ax.set_xlim(max(xaxis.values),min(xaxis.values))
    else:
      ax.set_xlim(min(xaxis.values),max(xaxis.values))

    if isinstance(yaxis, Pres) or isinstance(yaxis, Hybrid):
      ax.set_ylim(max(yaxis.values),min(yaxis.values))
    else:
      ax.set_ylim(min(yaxis.values),max(yaxis.values))

  if wasint:
     ion()
     draw()
     if not wait: show()
# }}}
