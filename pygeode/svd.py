# Singular value decomposition


# 'order' axis  i.e., the EOF number
from pygeode.axis import Index
class order (Index): pass
del Index


from pygeode.timeaxis import Time
# Finds the coupled EOFs between two fields
def SVD (var1, var2, num=1, subspace=-1, iaxis=Time, weight1=True, weight2=True, matrix='cov'):
  """
  Finds coupled EOFs of two fields.  Note that the mean/trend/etc. is NOT
  removed in this routine.

  Parameters
  ----------
  var1, var2 : :class:`Var`
    The variables to analyse.

  num : integer
    The number of EOFs to compute (default is ``1``).

  weight1, weight2 : optional 
    Weights to use for defining orthogonality in the var1, var2 domains,
    respectively.  Patterns X and Y in the var1 domain are orthogonal if the
    sum over X*Y*weights1 is 0. Patterns Z and W in the var2 domain are
    orthogonal if the sum over Z*W*weights2 is 0. Default is to use internal
    weights defined for var1 accessed by :meth:`Var.getweights()`. If set to
    ``False`` no weighting is used.

  matrix : string, optional ['cov']
    Which matrix we are diagonalizing (default is 'cov'). 
     * 'cov': covariance matrix of var1 & var2
     * 'cov': correlation matrix of var1 & var2

  iaxis : Axis identifier
    The principal component / expansion coefficient axis, i.e., the 'time'
    axis. Can be an integer (the axis number, leftmost = 0), the axis name
    (string), or a Pygeode axis class.  If not specified, will try to use
    pygeode.timeaxis.Time, and if that fails, the leftmost axis.

  Returns
  -------
  (eof1, pc1, eof2, pc2): tuple
    * eof1: The coupled eof patterns for var1. 
    * pc1: The principal component / expansion coefficients for var1.
    * eof2: The coupled eof patterns for var2.
    * pc2: The principal component / expansion coefficients for var2.

  Notes
  -----
    Multiple orders of EOFs are concatenated along an 'order' axis.
  """
  import numpy as np
  from pygeode.timeaxis import Time
  from pygeode.var import Var
  from pygeode.view import View
  from pygeode import MAX_ARRAY_SIZE
  from warnings import warn
  from pygeode import svdcore as lib

  if matrix in ('cov', 'covariance'): matrix = 'cov'
  elif matrix in ('cor', 'corr', 'correlation'): matrix = 'cor'
  else:
    warn ("invalid matrix type '%'.  Defaulting to covariance."%matrix, stacklevel=2)
    matrix = 'cov'

  MAX_ITER = 1000

  # Iterate over more EOFs than we need
  # (this helps with convergence)
  # TODO: a more rigorous formula for the optimum number of EOFs to use
  if subspace <= 0: subspace = 2*num + 8
  if subspace < num: subspace = num  # Just in case

  # Remember the names
  prefix1 = var1.name+'_' if var1.name != '' else ''
  prefix2 = var2.name+'_' if var2.name != '' else ''

  # Apply weights?
#  if weight1 is not None: var1 *= weight1.sqrt()
#  if weight2 is not None: var2 *= weight2.sqrt()
  if weight1 is True: weight1 = var1.getweights()
  if weight1 is not False:
    assert not weight1.hasaxis(iaxis), "Can't handle weights along the record axis"
    # Normalize the weights
    W = weight1.sum() / weight1.size
    weight1 /= W
    # Apply the weights
    var1 *= weight1.sqrt()
  if weight2 is True: weight2 = var2.getweights()
  if weight2 is not False:
    assert not weight2.hasaxis(iaxis), "Can't handle weights along the record axis"
    # Normalize the weights
    W = weight2.sum() / weight2.size
    weight2 /= W
    # Apply the weights
    var2 *= weight2.sqrt()


  #TODO: allow multiple iteration axes (i.e., time and ensemble)
#  if iaxis is None:
#    if var1.hasaxis(Time) and var2.hasaxis(Time):
#      iaxis1 = var1.whichaxis(Time)
#      iaxis2 = var2.whichaxis(Time)
#    else:
#      iaxis1 = 0
#      iaxis2 = 0
#  else:
  iaxis1 = var1.whichaxis(iaxis)
  iaxis2 = var2.whichaxis(iaxis)

  assert var1.axes[iaxis1] == var2.axes[iaxis2], "incompatible iteration axes"
  del iaxis  # so we don't use this by accident


  # Special case: can load entire variable in memory
  # This will save some time, especially if the field is stored on disk, or is heavily derived
  if var1.size <= MAX_ARRAY_SIZE:
    print 'preloading '+repr(var1)
    var1 = var1.load()
  if var2.size <= MAX_ARRAY_SIZE:
    print 'preloading '+repr(var2)
    var2 = var2.load()

  # Use correlation instead of covariance?
  # (normalize by standard deviation)
  if matrix == 'cor':
    print 'computing standard deviations'
    std1 = var1.stdev(iaxis1).load()
    std2 = var2.stdev(iaxis2).load()
    # account for grid points with zero standard deviation?
    std1.values = std1.values + (std1.values == 0)
    std2.values = std2.values + (std2.values == 0)
    var1 /= std1
    var2 /= std2


  eofshape1 =  (subspace,) + var1.shape[:iaxis1] + var1.shape[iaxis1+1:]
  eofshape2 =  (subspace,) + var2.shape[:iaxis2] + var2.shape[iaxis2+1:]

  pcshape1 =  (var1.shape[iaxis1], subspace)
  pcshape2 =  (var2.shape[iaxis2], subspace)

  # number of spatial grid points
  NX1 = var1.size / var1.shape[iaxis1]
  assert NX1 <= MAX_ARRAY_SIZE, 'field is too large!'
  NX2 = var2.size / var2.shape[iaxis2]
  assert NX2 <= MAX_ARRAY_SIZE, 'field is too large!'

  # Total number of timesteps
  NT = var1.shape[iaxis1]
  # Number of timesteps we can do in one fetch
  dt = MAX_ARRAY_SIZE / max(NX1,NX2)

  pcs1 = np.empty(pcshape1,dtype='d')
  pcs2 = np.empty(pcshape2,dtype='d')

  X = np.empty(eofshape2,dtype='d')
  U = np.empty(eofshape1,dtype='d')
  # Seed with sinusoids superimposed on random values
  Y = np.random.rand(*eofshape1)
  V = np.random.rand(*eofshape2)
  from math import pi
  for i in range(subspace):
    Y[i,...].reshape(NX1)[:] += np.cos( np.arange(NX1,dtype='d') / NX1 * 2 * pi * (i+1))
    V[i,...].reshape(NX2)[:] += np.cos( np.arange(NX2,dtype='d') / NX2 * 2 * pi * (i+1))

#  raise Exception

  # Workspace for C code
  UtAX  = np.empty([subspace,subspace], dtype='d')
  XtAtU = np.empty([subspace,subspace], dtype='d')
  VtV   = np.empty([subspace,subspace], dtype='d')
  YtY   = np.empty([subspace,subspace], dtype='d')

  # Views over whole variables
  # (rearranged to be compatible with our output eof arrays)
  view1 = View( (var1.axes[iaxis1],) + var1.axes[:iaxis1] + var1.axes[iaxis1+1:] )
  view2 = View( (var2.axes[iaxis2],) + var2.axes[:iaxis2] + var2.axes[iaxis2+1:] )


  for iter_num in range(1,MAX_ITER+1):

    print 'iter_num:', iter_num

    assert Y.shape == U.shape
    assert X.shape == V.shape
    U, Y = Y, U
    X, V = V, X

    # Reset the accumulation arrays for the next approximations
    Y[()] = 0
    V[()] = 0

    # Apply the covariance/correlation matrix
    for t in range(0,NT,dt):
      # number of timesteps we actually have
      nt = min(dt,NT-t)

      # Read the data
      chunk1 = view1.modify_slice(0, slice(t,t+nt)).get(var1)
      chunk1 = np.ascontiguousarray(chunk1, dtype='d')
      chunk2 = view2.modify_slice(0, slice(t,t+nt)).get(var2)
      chunk2 = np.ascontiguousarray(chunk2, dtype='d')

      ier = lib.build_svds (subspace, nt, NX1, NX2, chunk1, chunk2,
                            X, Y, pcs2[t,...])
      assert ier == 0
      ier = lib.build_svds (subspace, nt, NX2, NX1, chunk2, chunk1,
                            U, V, pcs1[t,...])
      assert ier == 0


    # Useful dot products
    lib.dot(subspace, NX1, U, Y, UtAX)
    lib.dot(subspace, NX2, V, V, VtV)
    lib.dot(subspace, NX1, Y, U, XtAtU)
    lib.dot(subspace, NX1, Y, Y, YtY)

    # Compute surrogate matrices (using all available information from this iteration)
    A1, residues, rank, s = np.linalg.lstsq(UtAX,VtV,rcond=1e-30)
    A2, residues, rank, s = np.linalg.lstsq(XtAtU,YtY,rcond=1e-30)

    # Eigendecomposition on surrogate matrices
    Dy, Qy = np.linalg.eig(np.dot(A1,A2))
    Dv, Qv = np.linalg.eig(np.dot(A2,A1))

    # Sort by eigenvalue (largest first)
    S = np.argsort(np.real(Dy))[::-1]
    Dy = Dy[S]
    Qy = np.ascontiguousarray(Qy[:,S], dtype='d')
    S = np.argsort(np.real(Dv))[::-1]
    Dv = Dv[S]
    Qv = np.ascontiguousarray(Qv[:,S], dtype='d')

    # get estimate of true eigenvalues
    D = np.sqrt(Dy)  # should also = np.sqrt(Dv) in theory
    print D

    # Translate the surrogate eigenvectors to an estimate of the true eigenvectors
    lib.transform(subspace, NX1, Qy, Y)
    lib.transform(subspace, NX2, Qv, V)

    # Normalize
    lib.normalize (subspace, NX1, Y)
    lib.normalize (subspace, NX2, V)

    if not np.allclose(U[:num,...],Y[:num,...], atol=0): continue
    if not np.allclose(X[:num,...],V[:num,...], atol=0): continue
    print 'converged after %d iterations'%iter_num
    break

  assert iter_num != MAX_ITER, "no convergence"

  # Flip the sign of the var2 EOFs and PCs so that the covariance is positive
  lib.fixcov (subspace, NT, NX2, pcs1, pcs2, V)

  # Wrap as pygeode vars, and return
  # Only need some of the eofs for output (the rest might not have even converged yet)
  orderaxis = order(num)

  eof1 = np.array(Y[:num])
  pc1 = np.array(pcs1[...,:num]).transpose()
  eof1 = Var((orderaxis,)+var1.axes[:iaxis1]+var1.axes[iaxis1+1:], values=eof1)
  pc1 = Var((orderaxis,var1.axes[iaxis1]), values = pc1)

  eof2 = np.array(V[:num])
  pc2 = np.array(pcs2[...,:num]).transpose()
  eof2 = Var((orderaxis,)+var2.axes[:iaxis2]+var2.axes[iaxis2+1:], values=eof2)
  pc2 = Var((orderaxis,var2.axes[iaxis2]), values = pc2)

  # Apply weights?
  if weight1 is not False: eof1 /= weight1.sqrt()
  if weight2 is not False: eof2 /= weight2.sqrt()

  # Use correlation instead of covariance?
  # Re-scale the fields by standard deviation
  if matrix == 'cor':
    eof1 *= std1
    eof2 *= std2

  # Give it a name
  eof1.name = prefix1 + "EOF"
  pc1.name = prefix1 + "PC"
  eof2.name = prefix2 + "EOF"
  pc2.name = prefix2 + "PC"

  return eof1, pc1, eof2, pc2

del Time
