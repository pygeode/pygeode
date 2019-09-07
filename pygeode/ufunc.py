# This module defines all operations that are applied element by element.
# This is essentially a wrapper to apply numpy universal functions (ufunc)
# to PyGeode variables.

from .var import Var
class UfuncVar (Var):
# {{{
  op = None
  symbol = None

  # If any of the arguments are not scalars or Vars, then defer the op
  # to some other module that may handle it (such as Dataset).
  def __new__ (self, *args):
# {{{
    from pygeode.var import Var
    import numpy as np
    for arg in args:
      if not isinstance(arg,(int,float,complex,np.number,Var)):
        return NotImplemented
    return object.__new__(self)
# }}}

  def __init__ (self, *args):
  # {{{
    from pygeode.tools import combine_axes
    from pygeode.var import combine_meta
    import numpy as np

    assert self.op is not None, "can't instantiate UfuncVar directly"

    ivars = [i for i,v in enumerate(args) if isinstance(v, Var)]
    vars = [args[i] for i in ivars]

    axes = combine_axes(vars)

    self.args = args
    self.ivars = ivars

#    dtype = common_dtype(args)
    # create some dummy scalar args to test the dtype
    dummy_dtypes = ['int64' if isinstance(a,int) else 'float64' if isinstance(a,float) else 'complex128' if isinstance(a,complex) else a.dtype for a in args]
    dummy_args = [np.array(1,dtype=d) for d in dummy_dtypes]
    dtype = self.op(*dummy_args).dtype

    # TODO: Type check arguments. numpy arrays probably shouldn't be allowed

    # Generate a default name
    symbol = self.symbol
    names = [(arg.name or '??') if isinstance(arg,Var) else str(arg) for arg in args]
    # Strip out parentheses if there's only one name?
    if len(names) == 1:
      if names[0].startswith('(') and names[0].endswith(')'):
        names[0] = names[0][1:-1]

    if symbol is None:
      name = self.op.__name__ + '(' + ','.join(names) + ')'

    elif isinstance(symbol,(list,tuple)):
      assert len(names) == 1
      name = symbol[0] + names[0] + symbol[1]

    else:
      assert isinstance(symbol, str)
      name = '(' + symbol.join(names) + ')'

    # Special case: applying a scalar to a Var object with a simple name.
    # In this case, keep the original name.
    if len(args) == 2 and len(vars) == 1:  # One scalar, one var
      if '(' not in vars[0].name and ')' not in vars[0].name:
        if self.symbol in ('+','-','*','/'):  # Basic arithmetic only
          name = vars[0].name

#    # Copy any common generic metadata
#    self.atts = common_dict(v.atts for v in vars)
#    self.plotatts = common_dict(v.plotatts for v in vars)

    Var.__init__(self, axes, dtype=dtype)

    # Copy any common generic metadata
    combine_meta(vars, self)
    # Use our locally derived name (override combine_meta)
    self.name = name

  # }}}

  def getview (self, view, pbar):
  # {{{
    import numpy as np
    args = list(self.args)

    # Relative progress of each variable
    sizes = [args[i].size for i in self.ivars]
    prog = np.cumsum([0.]+sizes) / np.sum(sizes) * 100

    for I,i in enumerate(self.ivars):
      # For each variable, get appropriate subranges, reshape, and transpose array
      args[i] = view.get(args[i], pbar=pbar.subset(prog[I],prog[I+1]))
      # Fix for 0-dimensions variables, which will return a scalar
      args[i] = np.array(args[i])
    # Ensure that the output is still a numpy array
    # (if the input arrays are degenerate, numpy wants to return a scalar
    #  by default)
    import numpy
    values = numpy.asarray(self.op(*args), self.dtype)
    return values
  # }}}
# }}}





# Take a function that operates on numpy arrays, create a
# function that operates on PyGeode arrays.
def wrap_npfunc (nterms, npfunc, doc='', symbol=None):
  import numpy as np
  from types import FunctionType

  # Create a ufunc subclass
  _symbol = symbol  # change the name so it doesn't collide with the class member below
  class C(UfuncVar):
    op = staticmethod(npfunc)
    symbol = _symbol
  C.__name__ = npfunc.__name__.strip('_').capitalize() + "_Var"
  # Create a function for creating the class
  assert isinstance(nterms, int)
  if nterms == 1:
    def f(x): return C(x)
  elif nterms == 2:
    def f(x,y): return C(x,y)
  elif nterms == -1:
    def f(*args): return C(*args)
  else: raise Exception
  f.__name__ = npfunc.__name__
  f.__doc__ = doc

  # Is this a trivial wrapper of something from numpy?
  if getattr(np, npfunc.__name__, None) is npfunc:
    # Use proper sentence structure?
    if f.__doc__ != '':
      f.__doc__ = f.__doc__.rstrip(' ').rstrip('.')
      f.__doc__ += ".  "
    if type(npfunc) is FunctionType:
      f.__doc__ += "Wraps :func:`numpy.%s`"%npfunc.__name__
    else:
      # Ufunc things (such as sin) are ufunc types, not functions?
      f.__doc__ += "Wraps :data:`numpy.%s`"%npfunc.__name__

  return f

def wrap_unary (npfunc, doc='', symbol=None):
  return wrap_npfunc (1, npfunc, doc, symbol)

def wrap_binary (npfunc, doc='', symbol=None):
  return wrap_npfunc (2, npfunc, doc, symbol)

# Chain a sequence of unary functions
#TODO: use some standard function for this?
def chain (*f):
  assert len(f) > 0
  if len(f) == 1: return f[0]
  def make_chain (f1,f2):
    def f3(x): return f1(f2(x))
    return f3
  return make_chain (f[0],chain(*f[1:]))

import numpy as np


abs  = wrap_unary(np.abs,    "Absolute value")
absolute  = wrap_unary(np.abs,    "Absolute value")
sign  = wrap_unary(np.sign,  "Sign (+1 = *positive*, -1 = *negative*)")
exp   = wrap_unary(np.exp,   "Natural exponent")
log   = wrap_unary(np.log,   "Natural logarithm")
log10 = wrap_unary(np.log10, "Base-10 logarithm")
cos   = wrap_unary(np.cos,   "Cosine of angle (in radians)")
sin   = wrap_unary(np.sin,   "Sine of angle (in radians)")
tan   = wrap_unary(np.tan,   "Tangent of angle (in radians)")
cosh  = wrap_unary(np.cosh,  "Hyperbolic cosine")
sinh  = wrap_unary(np.sinh,  "Hyperbolic sine")
tanh  = wrap_unary(np.tanh,  "Hyperbolic tangent")
sqrt  = wrap_unary(np.sqrt,  "Square root")
real  = wrap_unary(np.real,  "Real part of a complex array")
imag  = wrap_unary(np.imag,  "Imaginary part of a complex array")
conj  = wrap_unary(np.conj,  "Complex conjugate of a complex array"); conj.__name__ = 'conj'
angle = wrap_unary(np.angle, "Angles (arguments) of a complex array")
arccos  = wrap_unary(np.arccos,  "Inverse cosine (in radians)")
arcsin  = wrap_unary(np.arcsin,  "Inverse sine (in radians)")
arctan  = wrap_unary(np.arctan,  "Inverse tangent (in radians)")
arccosh = wrap_unary(np.arccosh, "Inverse hyperbolic cosine")
arcsinh = wrap_unary(np.arcsinh, "Inverse hyperbolic sine")
arctanh = wrap_unary(np.arctanh, "Inverse hyperbolic tangent")
nan_to_num = wrap_unary(np.nan_to_num, "Replace nan with zero and inf with finite numbers")
arctan2 = wrap_binary(np.arctan2, "Inverse tangent. Explicit x/y given. Returns radians")

def _deg2rad(x):
  from math import pi
  return x * pi / 180.
deg2rad = wrap_unary (_deg2rad, "Converts values from degrees to radians")
cosd = chain(np.cos,_deg2rad); cosd.__name__ = 'cosd'
sind = chain(np.sin,_deg2rad); sind.__name__ = 'sind'
tand = chain(np.tan,_deg2rad); tand.__name__ = 'tand'
cosd = wrap_unary (cosd, "Cosine of angle (in degrees)")
sind = wrap_unary (sind, "Sine of angle (in degrees)")
tand = wrap_unary (tand, "Tangent of angle (in degrees)")


def _rad2deg(x):
  from math import pi
  return x * 180. / pi
rad2deg = wrap_unary (_rad2deg, "Converts values from radians to degress")
arccosd = chain(_rad2deg,np.arccos); arccosd.__name__ = 'arccosd'
arcsind = chain(_rad2deg,np.arcsin); arcsind.__name__ = 'arcsind'
arctand = chain(_rad2deg,np.arctan); arctand.__name__ = 'arctand'
arccosd = wrap_unary (arccosd, "Inverse cosine (in degrees).")
arcsind = wrap_unary (arcsind, "Inverse sine (in degrees).")
arctand = wrap_unary (arctand, "Inverse tangent (in degrees).")
arctand2 = chain(_rad2deg,np.arctan2); arctand2.__name__ = 'arctand2'
arctand2 = wrap_binary (arctand2, "Inverse tangent.  Explicit x/y given.  Returns degrees.")

minimum = wrap_binary (np.minimum, "Element-wise minimum of the two given arguments.")
maximum = wrap_binary (np.maximum, "Element-wise maximum of the two given arguments.")

def vprod(*args):
  if len(args) == 0: return 1
  ans = args[0]
  for arg in args[1:]:
    ans = ans * arg
  return ans
vprod = wrap_npfunc(-1, vprod, "Multiplies an arbitrary number of variables together")

def vsum(*args):
#  from numpy import sum
#  return sum(args, 0)
  if len(args) == 0: return 0
  ans = args[0]
  for arg in args[1:]:
    ans = ans + arg
  return ans
vsum = wrap_npfunc(-1, vsum, "Adds an arbitrary number of variables together")

clip = wrap_npfunc(-1, np.clip, "Clips values to given interval.")



del np

unary_flist = (abs, sign, exp, log, log10, cos, sin, tan, cosd, sind, tand,
    cosh, sinh, tanh, arccos, arcsin, arctan,
    arccosd, arcsind, arctand, arccosh, arcsinh, arctanh,
    sqrt, nan_to_num, real, imag, conj, angle,
    clip
)

binary_flist = (arctan2, arctand2, minimum, maximum)

vararg_flist = (vprod, vsum)

all_flist = unary_flist + binary_flist + vararg_flist

__all__ = tuple(sorted(f.__name__ for f in all_flist))

#__all__ = 

# Collect these functions into a list, which Var can load into itself dynamically
#class_flist = (__add__, __sub__, __mul__, __div__, __pow__,
#    __abs__, __neg__, __pos__, __mod__, __rmod__, __trunc__,
#    __lt__, __le__, __gt__, __ge__ , __eq__, __ne__,
#    __radd__, __rsub__, __rmul__, __rdiv__, __rpow__,
#    sign, exp, log, log10, cos, sin, tan, cosd, sind, tand,
#    cosh, sinh, tanh, arccos, arcsin, arctan, arctan2,
#    arccosd, arcsind, arctand, arctand2, arccosh, arcsinh, arctanh,
#    sqrt, nan_to_num, real, imag, angle
#)
#generic_flist = (
#    exp, log, log10, cos, sin, tan, cosd, sind, tand,
#    cosh, sinh, tanh, arccos, arcsin, arctan, arctan2,
#    arccosd, arcsind, arctand, arctand2, arccosh, arcsinh, arctanh,
#    sqrt, nan_to_num, real, imag, angle
#)

