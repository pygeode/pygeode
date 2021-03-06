import pygeode as pyg
import numpy as np

def buildT1():
  lat = pyg.regularlat(31)
  lon = pyg.regularlon(60)

  T_c = 260. + 40. * pyg.exp(-(lat/45.)**2)
  T_wave = 0.05 * lat * pyg.sind(6*lon)
  T = T_c + T_wave
  T.name = 'Temp'
  T.units = 'K'

  return pyg.Dataset([T], atts={'history':'Synthetic Temperature data generated by pygeode'})

def buildT2():
  nyrs = 10
  lat = pyg.regularlat(31)
  lon = pyg.regularlon(60)
  time = pyg.ModelTime365(values=np.arange(nyrs*365), \
          units='days', startdate={'year':2011, 'month':1, 'day':1})
  pres = pyg.Pres(np.arange(1000, 0, -50.))
  z = 6.6 * pyg.log(1000./pres)

  ts1 = 2*pyg.sin(2*np.pi*time/365.) + 4*pyg.Var((time,), values=np.random.randn(nyrs*365))
  ts1 = ts1.smooth('time', 20)
  ts2 = -5 + 0.6*time/365. + 5*pyg.Var((time,), values=np.random.randn(nyrs*365))
  ts2 = ts2.smooth('time', 20)

  T_c = 260. + 40. * pyg.exp(-((lat - 10*np.sin(2*np.pi*time/365))/45.)**2)
  T_wave = 0.05 * lat * pyg.sind(6*lon - time)# * ts1
  T_lapse = -5*z

  Tf = (T_lapse + T_c + T_wave).transpose('time', 'pres', 'lat', 'lon')
  Tf.name = 'Temp'

  U_c = 40 * pyg.sind(2*lat)**2 * pyg.sin(2*np.pi * z / 12)**2
  U_wave = 0.08 * lat * pyg.sind(6*lon - time)
  U = (U_c + ts2*U_wave).transpose('time', 'pres', 'lat', 'lon')
  U.name = 'U'
  return pyg.Dataset([Tf, U], atts={'history':'Synthetic Temperature and Wind data generated by pygeode'})

def buildT3():
  lon = pyg.regularlon(60)
  lat = pyg.gausslat(42)
  tm  = pyg.modeltime365n('1 Jan 2000', 200)

  def eps_like(v):
    return pyg.Var(v.axes, values = np.random.randn(*v.shape))

  X1 = (5. * tm / 2000.) * pyg.cosd(lat)
  X2 = pyg.cosd(2*np.pi * tm / 500.) * pyg.sind(0.5*lon)
  X3 = pyg.sind(2*np.pi * tm / 120.) * pyg.cosd(3*lon)**2

  X1e = X1 + 0.1 * eps_like(X1)
  X2e = X2 + 0.2 * eps_like(X2)
  X3e = X3 + 0.2 * eps_like(X3)

  Y1 = -1. * pyg.sind(lon) * X1e + eps_like(X1).smooth('lat', 4)
  Y2 = -0.2 * X1e + 0.8 * X2e + -0.6 * X3e + 0.1 * eps_like(X1)

  X1 = X1.rename('X1')
  X2 = X2.rename('X2')
  X3 = X3.rename('X3')
  X1e = X1e.rename('X1e')
  X2e = X2e.rename('X2e')
  X3e = X3e.rename('X3e')
  Y1 = Y1.rename('Y1')
  Y2 = Y2.rename('Y2')
  Y3 = Y2.rename('Y3')

  return pyg.Dataset([X1e, X2e, X3e, Y1, Y2], atts={'history':'Synthetic dataset generated by pygeode'})

t1 = buildT1()
t2 = buildT2()
t3 = buildT3()

