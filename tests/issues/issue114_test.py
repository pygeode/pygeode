# Issue 114
# https://github.com/pygeode/pygeode/issues/114

from pygeode.formats import netcdf4 as nc
from pygeode.axis import Lat
from pygeode.var import Var
from pygeode.dataset import Dataset

lat = Lat([80,70,60])
var = Var(axes=[lat], values=[1,2,3], name='A')
dataset = Dataset([var])

dataset_groups = {'Group 1': dataset, 'Group 2': dataset}

# Save the variable. 
nc.save('issue114_test.nc', dataset_groups, cfmeta=True)

# Read in the file again

dataset_groups_read = nc.open('issue114_test.nc',cfmeta=True)

# Check that the variables are the same
assert (dataset_groups['Group 1'].A == dataset_groups_read['Group 1'].A)


