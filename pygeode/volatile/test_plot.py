# Demo program for the wrapped plot objects

from pygeode.tutorial import t1
import pickle

from pygeode.volatile.plot_wrapper import Overlay, Multiplot, Colorbar
from pygeode.volatile.plot_shortcuts import contour, pcolor

# Define a contour plot
cont = contour (t1.Temp, title='contours', colors='black')

# Define a pseudo-colour plot
pcol = pcolor  (t1.Temp, title='pseudo-color')

# Define a plot that's an overlay of the above 2 plots
comb = Overlay (pcol,  Colorbar(), cont, title='combined')

# Define a plot that's a multiplot of the above 3 objects
theplot = Multiplot ([[cont,pcol],[comb]])

# Save and re-load the plot
outfile = open('myplot.pickle','w')
pickle.dump(theplot, outfile)
outfile.close()

infile = open('myplot.pickle','ro')
theplot = pickle.load(infile)
infile.close()

# Render the plot on the screen
theplot.render()

import matplotlib.pyplot as pl
pl.show()
