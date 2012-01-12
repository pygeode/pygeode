# Demo program for the wrapped plot objects

from pygeode.tutorial import t1
import pickle

from plot_wrapper import Overlay, Multiplot, Colorbar
from plot_shortcuts import contour, pcolor, Map

# Define a contour plot
cont = contour (t1.Temp, title='Contours', colors='black')

# Define a pseudo-colour plot
pcol = pcolor  (t1.Temp, title='Pseudo-color')
# Give it a colour bar
pcol = Colorbar(pcol)

# Define a plot that's an overlay of the above 2 plots
comb = Overlay (pcol, cont, title='Combined')

# Project it onto a map
mapped = Map (comb, projection='ortho', lon_0=-105, lat_0=40, resolution='c', title = 'On a map')

# Define a plot that's a multiplot of the above 4 objects
theplot = Multiplot ([[cont,pcol],[comb,mapped]])


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
