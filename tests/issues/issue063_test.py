# Issue 63 - Ability to edit Datasets in-place

# Try editing a variable in-place (as an attribute assignment)
def test_issue063_edit_attr():
  from pygeode.tutorial import t1
  import numpy as np
  old_values = t1.Temp.get()
  t1.Temp -= 273.15
  assert np.all(t1.Temp.get() == old_values - 273.15)

# Try editing a variable in-place (as a dictionary key)
def test_issue063_edit_dictkey():
  from pygeode.tutorial import t1
  import numpy as np
  old_values = t1.Temp.get()
  t1['Temp'] -= 273.15
  assert np.all(t1['Temp'].get() == old_values - 273.15)

# Try adding a new variable as an attribute
def test_issue063_new_attr():
  from pygeode.tutorial import t1
  t1.P0 = t1.Temp * 0 + 1000
  assert t1.P0.name == 'P0'
  assert t1.P0.get() == t1.Temp.get()*0+1000

# Try adding a new variable as a dictionary key
def test_issue063_new_dictkey():
  from pygeode.tutorial import t1
  t1['P0'] = t1.Temp * 0 + 1000
  assert t1.P0.name == 'P0'
  assert t1.P0.get() == t1.Temp.get()*0+1000



