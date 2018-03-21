# Issue 4 - The new netcdf module does not write out axis cf metadata
# https://github.com/pygeode/pygeode/issues/4

def test_io():

  from pygeode.formats import netcdf as nc
  from pygeode.timeaxis import StandardTime
  from pygeode.axis import Pres
  from pygeode.dataset import Dataset
  import numpy as np

  tm = StandardTime(values=np.arange(365), units='days', startdate={'year':2001})
  p = Pres(np.arange(100.))
  v = (tm * p).rename('v')

  # Save the dataset, then reload it immediately
  before = Dataset([v])
  nc.save('issue004_test.nc', before)
  after = nc.open('issue004_test.nc')

  # Compare all vars/axes/attributes

  for var in before:
    assert var.name in after, "Can't find var '%s'"%var.name
    var2 = getattr(after,var.name)
    assert var2.atts == var.atts, "mismatched metadata.  Input: %s, Output %s"%(var.atts,var2.atts)

  for axis in before.axes:
    axis2 = [a for a in after.axes if a.name == axis.name]
    assert len(axis2) == 1, "can't find axis '%s'"%axis.name
    axis2 = axis2[0]
#    assert axis2.atts == axis.atts, "mismatched metadata.  Input: %s, Output %s"%(axis.atts,axis2.atts)
    for attname in list(axis.atts.keys()):
      assert attname in axis2.atts, "attribute '%s' not found"%attname
      assert axis.atts[attname] == axis2.atts[attname], "attribute '%s' changed from '%s' to '%s'"%(attname, axis.atts[attname], axis2.atts[attname])
    assert type(axis2) == type(axis), "mismatched axis types.  Input: %s, Output %s"%(type(axis), type(axis2))
