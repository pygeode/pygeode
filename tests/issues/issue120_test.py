from pygeode.tutorial import t1
imoprt pylab as pyl

pyl.ioff()

# default behaviour
ax1 = pyg.showvar(t1.Temp)

ax2 = pyg.showvar(t1.Temp,colorbar=False)

# assuming that if there is a second axis, it will be a colorbar
# and axes list will have non-zero length
assert len(ax1.axes)==2

assert len(ax2.axes)==0

