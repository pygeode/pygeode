# Importing this module will force PyGeode vars to be immediately evaluated
# as soon as they're created, effectively disabling the lazy evaluation feature.
# NOTE: Only use this option if all your data can fit into memory.

# Usage:
#   from pygeode.options import immediate_evaluation

from functools import wraps
from pygeode.var import Var
old_init = Var.__init__
@wraps(Var.__init__)
def new_init (self, *args, **kwargs):
  from pygeode.view import View
  old_init (self, *args, **kwargs)
  if not hasattr(self,'values'):
    self.values = View(self.axes).get(self)
Var.__init__ = new_init

