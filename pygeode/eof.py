#TODO: handle missing values
#TODO: for certain sizes, calculate the full covariance matrix, but then use an iterative method directly on it?

# EOF order axis
from pygeode.axis import Axis
class order(Axis): 
  name = 'eof'
  formatstr = '%d'

  plotatts = Axis.plotatts.copy()
  plotatts['plotname'] = 'EOF'
del Axis


def apply_weights (var, weight):
  # Normalize by area weight?
  if weight is not False:
    if weight is True:  weight = var.getweights()
    W = weight.sum() / weight.size
    weight /= W
    weight = weight.sqrt()
    name = var.name
    var *= weight
    var.name = name
  return var

def remove_weights (var, weight):
  # Undo normalization by area weight?
  # (should be the exact opposite of the 'weight' function
  if weight is not False:
    if weight is True:  weight = var.getweights()
    W = weight.sum() / weight.size
    weight /= W
    weight = weight.sqrt()
    name = var.name
    var /= weight
    var.name = name
  return var

# Parse the output list
def whichout (out):
  # Valid options
  namemap = [ 'eof|eofs', 'eig|eigenvalue|eigenvalues', 'pc|timeseries', 'var|variance', 'frac|varfrac', 'eig2' ]
  namemap = [ n.split('|') for n in namemap ]
  names = [ n for N in namemap for n in N ]
  # Defaults
  if out is None: out = ('eof','eig','pc')
  # Single string (comma separated values?)
  if isinstance(out,str): out = out.split(',')
  assert isinstance(out,(list,tuple)), "Unknown output type"
  for o in out: assert isinstance(o,str), "Expected string argument, found %s"%type(o)
  # Lower case
  out = [o.lower() for o in out]
  for o in out: assert o in names, "Unrecognized output '%s'"%o
  return out

##prep the var
def prep (var, iaxis, weight, out):
  from pygeode.timeaxis import Time
  from pygeode.var import Var
  from pygeode.axis import Axis
  from pygeode.view import View
  from warnings import warn
  from pygeode import MAX_ARRAY_SIZE

  assert isinstance(var,Var)
  assert var.naxes >= 2, "need at least 2 axes"

  # Check the outputs
  out = whichout(out)
  del out # not actually used here

  # Keep the name
  name = var.name

  # Normalize by area weight?
  var = apply_weights(var, weight=weight)
  del weight

  timeaxes = iaxis
  del iaxis
  if timeaxes is None:
    if var.hasaxis(Time): timeaxes = Time
    else:
      warn ("No explicit record axis provided.  Using the first axis.", stacklevel=2)
      timeaxes = 0

  # Keep the record axis/axes as a tuple
  # (in case we have more than one axis, i.e. time and ensemble)
  if not isinstance(timeaxes,(list,tuple)):
    assert isinstance(timeaxes,(int,long)) or issubclass(timeaxes,Axis), 'unknown iaxis type %s'%type(timeaxes)
    timeaxes = [timeaxes]

  # Convert the axes to integer ids
  timeaxes = [var.whichaxis(a) for a in timeaxes]
  spaceaxes = [i for i in range(var.naxes) if i not in timeaxes]

  # Convert to axis objects
  timeaxes = [var.axes[i] for i in timeaxes]
  spaceaxes = [var.axes[i] for i in spaceaxes]

  # Create a view, to hold the axes together
  # (provides us with other useful stuff, like a notion of 'shape' and 'size')
  time = View(axes=timeaxes)
  space = View(axes=spaceaxes)

#  var = SquishedVar(var, timeaxes, spaceaxes)

  # Preload the data, if possible
  if var.size <= MAX_ARRAY_SIZE: var = var.load()

  return var, time, space



# Remove weights, wrap as Vars
def finalize (var, time, space, eof, eig, pc, variance, weight, out):
  from pygeode.var import Var
  import numpy as np

  # Keep the name
  name = var.name
  num = eof.shape[0]

  # Conform to the proper shape
  eof = eof.reshape((num,)+space.shape)
  pc = pc.reshape((num,)+time.shape)

  # Use a consistent sign convention
  # (the first element of the eof is non-negative)
  sign = [-1 if eof.reshape(num,-1)[i,0] < 0 else 1 for i in range(num)]
  for i,s in enumerate(sign):
    eof[i,...] *= s
    pc[i,...] *= s

  # Copy the data into fresh array
  # (that way, if these are view on much larger arrays, the rest of
  #  the unused data can be garbage collected)
  eof = np.array(eof)
  pc = np.array(pc)
  eig = np.array(eig)

  # EOF axis
  orderaxis = order(num,name="order")
  eof = Var((orderaxis,)+space.axes, values=eof)
  eig = Var([orderaxis], values=eig)
  pc = Var((orderaxis,)+time.axes, values = pc)

  # Undo normalization by area weight?
  eof = remove_weights(eof, weight=weight).load()

  eof.name = name + "_EOF"
  pc.name = name + "_timeseries"
  eig.name = name + "_eigenvalues"

  # Other things
  # Fraction of total variance
  frac = ((eig**2) / variance).load()
  frac.name = name + "_fraction_of_variance"
  # Eigenvalues of the covariance matrix
  eig2 = (eig**2).load()
  eig2.name = name + "_eigenvalues"

  # Gather up all possible outputs
  outdict = dict(eof=eof, eig=eig, eig2=eig2, pc=pc, var=variance, frac=frac)
  out = whichout(out)
  out = [outdict[o] for o in out]

  return out


##################################################
# Iterative EOF method
##################################################

# After each iteration, does an explicit eigenvalue decomposition on a much
# smaller matrix.  Hopefully, this will converge faster than a simple
# power-method.
# Uses EOF_guess for initial patterns

# Assume the mean has already been subtracted!
def EOF_iter (x, num=1, iaxis=None, subspace = -1, max_iter=1000, weight=True, out=None):
  """
  (See svd.SVD for documentation on a similar function, but replace each xxx1 and xxx2 parameter with a single xxx parameter.)
  """
  import numpy as np
  from pygeode import libpath
  from pygeode.view import View
  from math import sqrt
  from pygeode.varoperations import fill
  from pygeode import svdcore as lib

  # Need vector subspace to be at least as large as the number of EOFs extracted.
  if subspace < num: subspace = num

  # Run the single-pass guess to seed the first iteration
  guess_eof, guess_eig, guess_pc = EOF_guess (x, subspace, iaxis, weight=weight, out=None)
  # Convert NaNs to zeros so they don't screw up the matrix operations
  guess_eof = fill (guess_eof, 0)

  x, time, space = prep(var=x, iaxis=iaxis, weight=weight, out=out)
  del iaxis

  eofshape =  (subspace,) + space.shape
  pcshape =  time.shape + (subspace,)

  pcs = np.empty(pcshape,dtype='d')

  oldeofs = np.empty(eofshape,dtype='d')
  # Seed with initial guess (in the weighted space)
  neweofs = apply_weights (guess_eof, weight=weight).get()
  neweofs = np.array(neweofs, dtype='d')  # so we can write
#  neweofs = np.random.rand(*eofshape)

  # Workspace for smaller representative matrix
  work1 = np.empty([subspace,subspace], dtype='d')
  work2 = np.empty([subspace,subspace], dtype='d')

  NX = space.size

  # Variance accumulation (on first iteration only)
  variance = 0.0

  for iter_num in range(1,max_iter+1):

    print 'iter_num:', iter_num

    neweofs, oldeofs = oldeofs, neweofs

    # Reset the accumulation arrays for the next approximations
    neweofs[()] = 0

    # Apply the covariance matrix
    for inview in View(x.axes).loop_mem():
      X = np.ascontiguousarray(inview.get(x), dtype='d')
      assert X.size >= space.size, "spatial pattern is too large"

      nt = inview.shape[0]
      time_offset = inview.slices[0].start
      ier = lib.build_eofs (subspace, nt, NX, X, oldeofs,
                            neweofs, pcs[time_offset,...])
      assert ier == 0

      # Compute variance?
      if iter_num == 1:
        variance += (X**2).sum()

    # Useful dot products
    lib.dot(subspace, NX, oldeofs, neweofs, work1)
    lib.dot(subspace, NX, neweofs, neweofs, work2)

    # Compute surrogate matrix (using all available information from this iteration)
    A, residues, rank, s = np.linalg.lstsq(work1,work2,rcond=1e-30)

    # Eigendecomposition on surrogate matrix
    w, P = np.linalg.eig(A)

    # Sort by eigenvalue
    S = np.argsort(w)[::-1]
    w = w[S]
    print w
#    assert P.dtype.name == 'float64', P.dtype.name
    P = np.ascontiguousarray(P[:,S], dtype='d')

    # Translate the surrogate eigenvectors to an estimate of the true eigenvectors
    lib.transform(subspace, NX, P, neweofs)

    # Normalize
    lib.normalize (subspace, NX, neweofs)

#    # verify orthogonality
#    for i in range(num):
#      print [np.dot(neweofs[i,...].flatten(), neweofs[j,...].flatten()) for j in range(num)]

    if np.allclose(oldeofs[:num,...],neweofs[:num,...], atol=0):
      print 'converged after %d iterations'%iter_num
      break

  assert iter_num != max_iter, "no convergence"

  # Wrap as pygeode vars, and return
  # Only need some of the eofs for output (the rest might not have even converged yet)
  eof = neweofs[:num]
  pc = pcs[...,:num].transpose()

  # Extract the eigenvalues
  # (compute magnitude of pc arrays)
  #TODO: keep eigenvalues as a separate variable in the iteration loop
  eig = np.array([sqrt( (pc[i,...]**2).sum() ) for i in range(pc.shape[0]) ])
  pc = np.dot(np.diag(1/eig), pc)

  return finalize (x, time, space, eof, eig, pc, variance, weight=weight, out=out)






##################################################
# Single-pass EOF guess
# Gives a reasonable qualitative representation of the EOF patterns
##################################################
def EOF_guess (x, num=1, iaxis=None, weight=True, out=None):
  import numpy as np
  from pygeode.var import Var
  from pygeode.view import View
  from pygeode import eofcore as lib

  x, time, space = prep (x, iaxis, weight=weight, out=out)
  del iaxis

  print "working on array shape", x.shape

  # Initialize workspace
  work = lib.start (num, space.size)

  eof = np.empty((num,)+space.shape, dtype='d')
  eig = np.empty([num], dtype='d')
  pc = np.empty((num,)+time.shape, dtype='d')

  # Variance accumulation
  variance = 0.0

  # Loop over chunks of the data
  for inview in View(x.axes).loop_mem():
    X = np.ascontiguousarray(inview.get(x), dtype='d')
    assert X.size >= space.size, "Spatial pattern is too large"
    nrec = X.size / space.size
    lib.process (work, nrec, X)

    # Accumulate variance
    variance += (X**2).sum()

  # Get result
  lib.endloop (work, eof, eig, pc)

  # Free workspace
  lib.finish (work)

  # Wrap the stuff
  return finalize (x, time, space, eof, eig, pc, variance, weight=weight, out=out)



##################################################
# Compute EOFs directly from a singular value decomposition
# (works best on small data arrays)
##################################################
def EOF_svd (x, num=1, iaxis=None, weight=True, out=None):
  import numpy as np

  x, time, space = prep (x, iaxis, weight=weight, out=out)
  del iaxis

  # SVD
  data = x.get().reshape([time.size, space.size])
  variance = (data**2).sum()
  u, s, v = np.linalg.svd(data, full_matrices=False)
#  print '?', u.shape, s.shape, v.shape

  eof = v[:num,...]
  eig = s[:num]
  pc = u.transpose()[:num,...]

#  print eof.shape, eig.shape, pc.shape

  return finalize (x, time, space, eof, eig, pc, variance, weight=weight, out=out)


##################################################
# Compute EOFs explicitly from the covariance matrix
# (works best when the covariance matrix is small)
# Steps:
#  1) Scan through the data, construct an explicit covariance matrix
#  2) Compute the eigenvalues & eigenvectors
#  3) Scan through the data, construct the timeseries data
##################################################
def EOF_cov (x, num=1, iaxis=None, weight=True, out=None):
  import numpy as np
  from pygeode.view import View

  x, time, space = prep (x, iaxis, weight=weight, out=out)
  del iaxis

  # Initialize space for accumulating the covariance matrix
  cov = np.zeros ([space.size, space.size], dtype='d')

  # Accumulate the covariance
  for inview in View(x.axes).loop_mem():
    X = inview.get(x)
    assert X.size >= space.size, "Spatial pattern is too large"
    X = X.reshape(-1,space.size)

    cov += np.dot(X.transpose(),X)

  # Decompose the eigenvectors & eigenvalues
  w, v = np.linalg.eigh(cov/(time.size-1))

  variance = w.sum()
  eig = np.sqrt(w[::-1][:num])
  eof = v.transpose()[::-1,:][:num,:]

  # Compute the timeseries
  pc = []
  for inview in View(x.axes).loop_mem():
    X = inview.get(x).reshape(-1,space.size)
    pc.append(np.dot(eof, X.transpose()))

  pc = np.concatenate(pc, axis=1)
  # Normalize
  pc /= eig.reshape(num,1)

  return finalize (x, time, space, eof, eig, pc, variance, weight=weight, out=out)


##################################################
# Compute EOFs
# Determine which routine to use (svd, covariance, iterative)
# based on the size of the data
##################################################
def EOF (x, num=1, iaxis=None, weight=True, out=None):
  """
    Computes the leading Empirical Orthogonal Function(s) for the given
    variable.

    Parameters
    ----------
    x : Var object
        The data of interest

    num : integer, optional
        The number of leading EOFs to calculate. Default is ``1``.

    iaxis : Axis object, Axis class, string, or integer, optional
        Which axis/axes to treat as the record axis.  Multiple axes can be
        passed as a tuple.  Default is the ``Time`` axis of the data
        (if found), otherwise the leftmost axis.

    weight : Var object or boolean, optional
        Weights to use use for the orthogonality condition.
        If ``True``, it uses whatever internal weights the variable posesses.
        If ``False`` or ``None``, it doesn't use any weights.
        You can also pass in a Var object with explicit weights.
        Default is ``True``.

    out : string, optional
        Which outputs to return.  This is a comma-separated string,
        built from the following keywords:

          =======   ==========================================================
          keyword   meaning
          =======   ==========================================================
          EOF       The EOFs (normalized to unit variance)
          EIG       Eigenvalues (of the singular value decomposition of
                    the variable).  If you want the eigenvalues of the
                    covariance matrix, square these, or use EIG2).
          EIG2      Eigenvalues (of the covariance matrix)
          VAR       Variance (a single scalar value).
          FRAC      Fraction of total variance explained by each EOF.
          PC        Principal components (timeseries data), normalized
                    to unit variance.
          =======   ==========================================================

        Default is ``'EOF,EIG,PC'``

    Returns
    -------
    eof_decomposition : tuple
      A combination of EOFs, eignenvalues or other computed quantities specified
      by ``out``.

    Notes
    -----

    This routine doesn't do any pre-processing of the data, such as removing
    the mean or detrending.  If you want to work with anomalies, then you'll
    have to first compute the anomalies!

    This routine tries to automatically determine the best way to solve the
    EOFs.  If you want to use a particular method, you can call the following
    functions (with the same parameters):

      =========   ============================================================
      function    behaviour
      =========   ============================================================
      EOF_iter    Iterative solver (uses a variant of the power method).
      EOF_cov     Calculates the full covariance matrix, and then does an
                  explicit eigendecomposition.
      EOF_svd     Does an explicit singular value decomposition on the data.
      EOF_guess   Returns an approximation of the EOF decomposition from one
                  pass through the data.  This may be useful if you have a
                  large dataset, and you just want the qualitative features
                  of the EOF spatial patterns.
      =========   ============================================================
  """
  from math import sqrt
  from pygeode import MAX_ARRAY_SIZE
  # Maximum size allowed for explicit solvers
  #TODO: run some actual tests to determine this number
  max_explicit_size = int(sqrt(MAX_ARRAY_SIZE))
  # Get the time and space dimensions
  junk, time, space = prep (x, iaxis=iaxis, weight=weight, out=out)
  if x.size <= max_explicit_size: f = EOF_svd
  elif space.size**2 <= max_explicit_size: f = EOF_cov
  else: f = EOF_iter
  return f (x, num=num, iaxis=iaxis, weight=weight, out=out)


