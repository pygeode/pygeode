#!/usr/bin/python

from pygeode.formats import netcdf, opendap
from pygeode.plot import plotvar
from matplotlib.pyplot import ion, show
ion()
#import profile

#dataset = opendap.open("http://localhost:8080/era40_test")
#dataset = opendap.open("http://localhost:8080/t31ref1a2_gs")
#dataset = opendap.open("http://localhost:8080/t31ref1a2_gs_01")

dataset = opendap.open("http://sparc01.atmosp.physics.utoronto.ca:8080/CCMVal2/REF-B1/CMAM/model_levels")


print dataset
#quit()

o3 = dataset.O3
print o3

#plotvar (o3(time=0,ensemble=0).mean('lon'), pcolor=True, wait=True)
#plotvar (o3(time=0,ensemble=1).mean('lon'), pcolor=True, wait=True)
#plotvar (o3(time=0,ensemble=2).mean('lon'), pcolor=True, wait=True)
#plotvar (o3(time=0).mean('lon','ensemble'), pcolor=True, wait=True)
plotvar (o3(time=0,ensemble=0,eta=1))
show()


#print dataset
#plotvar (dataset.t(year=2000,month=7,day=1, pres=1000))

#from cProfile import run
#run('netcdf.save ("t.nc", dataset)')

