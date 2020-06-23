#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

// LAPACK routine for computing eigenvalues and eigenvectors
int dsyev_ (char *JOBZ, char *UPLO, int *N, double *A, int *LDA, double *W, double *WORK, int *LWORK, int *INFO);

// Local wrapper
// Eigenvectors are stored as the *rows* of A (overwriting original data)
// Ordered from smallest to largest
int dsyev_pyg (int N, double *A, double *W) {
  char JOBZ = 'V';
  char UPLO = 'L';
  int LDA = N;
  int LWORK;
  int INFO = 999; // Set to some nonzero value, to verify that when we get INFO=0, it means success
  
  // query the optimal workspace size
  {
    LWORK = -1;
    double WORK[1];
    dsyev_ (&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);
    LWORK = WORK[0];
//    printf ("INFO from workspace query: %d\n", INFO);
//    printf ("optimal LWORK: %d\n", LWORK);
    assert (LWORK > 0);
  }
  // compute the eigenvectors
  double WORK[LWORK];
  dsyev_ (&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);
//  printf ("INFO from DSYEV: %d\n", INFO);

  return 0;
}

/*************************************************

     Helper functions

*************************************************/

// Print a square matrix
void print_matrix (int n, double *d) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf ("%12.5g", d[i*n+j]);
    }
    printf ("\n");
  }
  printf ("\n");
}

// Print a vector (inline)
void print_vector (int n, double *d) {
  for (int i = 0; i < n; i++) printf ("%12.5g", d[i]);
  printf ("\n");
}

// Dot product of two matrices
// First dimension of leftmost matrix will be fastest varying for the output.
void dot (int ni, int nj, int nk, double *X, double *Y, double *Z) {
  for (int i = 0; i < ni; i++) {
    double *y = Y;
    double *z = Z;
    for (int k = 0; k < nk; k++) {
      double s = 0;
//      for (int j = 0; j < nj; j++) s += X[i*nj+j] * Y[k*nj+j];
      for (int j = 0; j < nj; j++) s += X[j] * y[j];
//      Z[k*ni+i] = s;
      z[i] = s;
      y += nj;  // Next column vector of Y
      z += ni;  // Next row of output Z
    }
    X += nj; // Next column vector of X
  }
}

// Transform a set of vectors
void transform (int nx, int M, int N, double *X, double *A, double *Y) {
  // loop over vector elements
  for (int x = 0; x < nx; x++) {
    double *a = A;
    double *y = Y;
    // loop over output vector #
    for (int j = 0; j < N; j++) {
      double s = 0;
      double *_X = X;
      // loop over input vectors, transform
//      for (int i = 0; i < M; i++) s += X[i*nx+x] * A[j*M+i];
      for (int i = 0; i < M; i++) {
        s += _X[x] * a[i];
        _X += nx;  // Next column of X (inner loop)
      }
//      Y[j*nx+x] = s;
      y[x] = s;
      a += M;  // Next column of A
      y += nx;  // Next column of output Y
    }
  }
}

// Scale a vector (in-place)
void scale (int N, double *X, double k) {
  for (int i = 0; i < N; i++) X[i] *= k;
}

// Multiply a matrix in-place by a diagonal
void diagscale (int nx, int N, double *A, double *d) {
  for (int i = 0; i < N; i++) scale (nx, A + i*nx, d[i]);
}
// Similar to above, but divide by a diagonal
// if the divisor is exactly 0, then do nothing
void diagunscale (int nx, int N, double *A, double *d) {
  for (int i = 0; i < N; i++) {
    if (d[i] != 0) scale (nx, A + i*nx, 1/d[i]);
  }
}

// Multiply a matrix in-place by a diagonal (on the left)
void ldiagscale (int nx, int N, double *d, double *A) {
  for (int i = 0; i < N; i++) {
    for (int x = 0; x < nx; x++) A[i*nx+x] *= d[x];
  }
}

// Transpose an array
void transpose (int nx, int N, double *A, double *B) {
  for (int i = 0; i < N; i++) {
    for (int x = 0; x < nx; x++) {
      B[x*N+i] = A[i*nx+x];
    }
  }
}

// Perform a singular value decomposition
// nx: length of each column of A
// N: number of columns of A
// m: number of singular values to compute
// U: m columns of orthogonal vectors
// eig: m largest singular values
// V: m columns of size n
void svd (int nx, int N, int m, double *A, double *U, double *eig, double *V) {
  double C[N][N];
  double e[N];
  assert (m <= N);
  // Compute the inner covariance matrix
  dot (N, nx, N, A, A, C[0]);
  // Compute inner eigenvalue decomposition
  dsyev_pyg (N, C[0], e);
  // Compute V and outer eigenvalues
  for (int i = 0; i < m; i++) {
    memcpy (V+i*N, C[N-1-i], N*sizeof(double));
    eig[i] = sqrt(e[N-1-i]);
  }
  // Compute U
  transform (nx, N, m, A, V, U);

  // Normalize
  diagunscale (nx, m, U, eig);

}



/*************************************************

     EOF functions

*************************************************/


// Maximum stack level.  A value of N allows for handling 2**N records.
#define MAX_STACKLEVEL 100

// Structure to hold workspace, parameters
typedef struct {
  int num_eofs; // number of EOF patterns to compute
  int nx;  // number of spatial grid points
  int stacksize;  // current depth of the stack
  double *eofstack[MAX_STACKLEVEL];  // EOF stack (normalized)
  double *eigstack[MAX_STACKLEVEL];  // eigenvalue stack
  int ntstack[MAX_STACKLEVEL];  // number of timesteps represented by each stack level
  double *pcstack[MAX_STACKLEVEL]; // timeseries stack (normalized)
  int count;  // Counts the number of eof pieces done so far
              // (used for determine when to merge pieces together, and how
              //  far up the stack to merge)
/*
  double *scratch;  // Scratch space for holding some EOFs during calculations
  double *scratch2; // More scratch space, twice the size
*/
  double *target;  // Target EOF patterns (from previous pass)
} Workspace;


// Start an EOF analysis
// (allocate workspace, set some parameters)
int start (int num_eofs, int nx, Workspace **work) {
  assert (work != NULL);  // need somewhere to store the pointer
  assert (num_eofs > 0);
  assert (nx > 0);

  Workspace *w = malloc(sizeof(Workspace));
  w->num_eofs = num_eofs;
  w->nx = nx;
  w->stacksize = 0;
  w->count = 0;

  // Initialize the # of timesteps, just in case there's a bad reference
  for (int i = 0; i < MAX_STACKLEVEL; i++) w->ntstack[i] = -1;

/*
  // Create some scratch space for merge_eof()
  w->scratch = malloc(sizeof(double)*num_eofs*nx);
  w->scratch2 = malloc(sizeof(double)*num_eofs*2*nx);
*/

  w->target = NULL;  // First pass, no target EOFs yet.

  *work = w;

  return 0;
}


// Merge two partial EOF patterns into one.
// Takes last 2 EOFs off the stack, merges them, then places the result back on the stack
// Mathematically, this is equivalent to:
// - reconstructing the two data chunks from their EOFs and timeseries
// - merging the data chunks into a single data array
// - performing EOF analysis on this larger array, to estimate the overall EOFs
void merge_eof (Workspace *work) {

  const int N = work->num_eofs;
  const int N2 = N*2;
  const int nx = work->nx;

  int stacksize = work->stacksize;
  assert (stacksize > 1);  // Nothing to merge?!

  double *eof1 = work->eofstack[stacksize-2];
  double *eig1 = work->eigstack[stacksize-2];
  int nt1 = work->ntstack[stacksize-2];
  assert (nt1 > 0);
  double *pc1 = work->pcstack[stacksize-2];

  double *eof2 = work->eofstack[stacksize-1];
  double *eig2 = work->eigstack[stacksize-1];
  int nt2 = work->ntstack[stacksize-1];
  assert (nt2 > 0);
  double *pc2 = work->pcstack[stacksize-1];

  double *new_eof = malloc(sizeof(double)*N*nx);
  double *new_eig = malloc(sizeof(double)*N);
  int nt = nt1 + nt2;
  double *new_pc1 = malloc(sizeof(double)*N2*nt1);
  double *new_pc2 = malloc(sizeof(double)*N2*nt2);
  double *new_pc = malloc(sizeof(double)*N*nt);

  printf ("  merging stacklevels %d and %d\n", stacksize-1, stacksize-2);

  // Merge the two sets of EOFs together
  double *E = malloc(sizeof(double)*N2*nx);
  memcpy (E, eof1, N*nx*sizeof(double));
  memcpy (E+N*nx, eof2, N*nx*sizeof(double));

  // Scale the eigenvectors by the eigenvalues
  for (int i = 0; i < N; i++) {
    scale (nx, E+i*nx, eig1[i]);
    scale (nx, E+(i+N)*nx, eig2[i]);
  }

  // Compute the overall eigenvectors (no truncation yet)
  double *E_eof = malloc(sizeof(double)*N2*nx);
  double E_eig[N2];
  // Junk from the svd - we don't use this array here.
  double V[N2][N2];
  svd (nx, N2, N2, E, E_eof, E_eig, V[0]);

  double *E_pc = malloc(sizeof(double)*N2*nt);

  // Compute the timeseries (principal components)
  double C[N2][N];

  diagscale (nt1, N, pc1, eig1);
  dot (N, nx, N2, eof1, E_eof, C[0]);
  transform (nt1, N, N2, pc1, C[0], new_pc1);

  diagscale (nt2, N, pc2, eig2);
  dot (N, nx, N2, eof2, E_eof, C[0]);
  transform (nt2, N, N2, pc2, C[0], new_pc2);

  for (int i = 0; i < N2; i++) {
    memcpy (E_pc + i*nt, new_pc1 + i*nt1, nt1*sizeof(double));
    memcpy (E_pc + i*nt + nt1, new_pc2 + i*nt2, nt2*sizeof(double));
  }

  diagunscale (nt, N2, E_pc, E_eig);


  // First pass
  if (work->target == NULL) {

    // keep the best N of the 2*N
    memcpy (new_eof, E_eof, sizeof(double)*N*nx);
    memcpy (new_eig, E_eig, sizeof(double)*N);
    memcpy (new_pc, E_pc, sizeof(double)*N*nt);

  }

  // Subsequent passes
  else {

    double rank[N2];
    for (int i = 0; i < N2; i++) {
      double v[N];
      dot (N, nx, 1, work->target, E_eof+i*nx, v);
//      scale (N, v, E_eig[i]);
      rank[i] = 0;
      for (int j = 0; j < N; j++) rank[i] += v[j]*v[j];
    }

    print_vector (N2, rank);

    // Copy the best-ranked EOFs
    // TODO: more efficient routine
    printf ("'optimal' order: ");
    for (int i = 0; i < N; i++) {
      int bestind = i;
      for (int j = 0; j < N2; j++) {
        if ((float)(rank[j]) > (float)(rank[bestind])) bestind = j;
      }
      printf ("%d ", bestind);
      memcpy (new_eof+i*nx, E_eof+bestind*nx, sizeof(double)*nx);
      new_eig[i] = E_eig[bestind];
      memcpy (new_pc+i*nt, E_pc+bestind*nt, sizeof(double)*nt);
      rank[bestind] = -1;  // disqualify this one from further inclusion (we already have it)
    }
    printf ("\n");

    // TODO
//    // keep the best N of the 2*N
//    memcpy (new_eof, E_eof, sizeof(double)*N*nx);
//    memcpy (new_eig, E_eig, sizeof(double)*N);
//    memcpy (new_pc, E_pc, sizeof(double)*N*nt);

  }

/*
  // Check that the new eigenvectors are orthogonal
  double test[N][N];
  dot (N, nx, N, new_eof, new_eof, test[0]);
  printf ("    EOF orthogonality check:\n");
  print_matrix (N, test[0]);
*/
/*
  // Check that the new timeseries are orthogonal
  dot (N, nt, N, new_pc, new_pc, test[0]);
  printf ("    PC orthogonality check:\n");
  print_matrix (N, test[0]);
*/

  // Free the old entries in the stack
  free (eof1); free (eof2);
  free (eig1); free (eig2);
  free (pc1); free (pc2);

  // Free the temporary space
  free (new_pc1); free (new_pc2);
  free (E); free (E_eof); free (E_pc);


  // Push the new entry onto the stack
  work->eofstack[stacksize-1] = NULL;  // Just in case
  work->eofstack[stacksize-2] = new_eof;

  work->eigstack[stacksize-1] = NULL;  // Just in case
  work->eigstack[stacksize-2] = new_eig;

  work->ntstack[stacksize-1] = -1; // Just in case
  work->ntstack[stacksize-2] = nt;

  work->pcstack[stacksize-1] = NULL; // Just in case
  work->pcstack[stacksize-2] = new_pc;

  // Update the stack
  stacksize--;
  work->stacksize = stacksize;

  return;
}


// Add some EOFs to the workspace stack
void add_to_stack (Workspace *work, double *eofs, double *eig, int nt, double *pcs) {

  int c = work->count;  // for iterating of the bits, from least to greatest

  printf ("adding to stack... (count=%d, stacksize=%d)\n", c, work->stacksize);

  // Store on the stack
  int stacksize = work->stacksize;
  assert (stacksize < MAX_STACKLEVEL);
  work->eofstack[stacksize] = eofs;
  work->eigstack[stacksize] = eig;
  work->ntstack[stacksize] = nt;
  work->pcstack[stacksize] = pcs;
  work->stacksize++;

  // Merging part
  while (c & 1) {
    merge_eof (work);
    c >>= 1;  // shift the bits to look at the next most significant
  }


  // Update the count
  work->count++;

  printf ("done adding to stack... new count = %d, new stacksize = %d\n", work->count, work->stacksize);
  return;
}


// Given some data, process it and put it in the stack
int process (Workspace *work, int NREC, double *data) {
  const int N = work->num_eofs;
  const int nx = work->nx;

  double *_data;

  // Loop over N-sized chunks of data
  for (int offset = 0; offset < NREC; offset+= N) {
    int n = (offset+N <= NREC) ? N : NREC-offset;
    printf ("handling %d records\n", n);
    _data = data + offset*nx;
    // To simplify the code below, always make sure we have N timesteps
    if (n < N) {
      printf ("padding the last set of records\n");
      _data = malloc(sizeof(double)*nx*N);
      memcpy (_data, data + offset*nx, sizeof(double)*nx*n);
      memset (_data+n*nx, 0, sizeof(double)*nx*(N-n));
    }

    double *eof = malloc(sizeof(double)*N*nx);
    double *eig = malloc(sizeof(double)*N);
    int nt = n;
    double PC[N][N];
    double *pc = malloc(sizeof(double)*N*nt);

    svd (nx, N, N, _data, eof, eig, PC[0]);
    for (int i = 0; i < N; i++) memcpy (pc+i*nt, PC[i], sizeof(double)*nt);
//    printf ("pc check: ");
//    for (int i = 0; i < nt; i++) printf ("%12.5g", pc[i]);
//    printf ("\n");

/*
    // Test orthogonality of these EOFs
    double C[N][N];
    dot (N, nx, N, eof, eof, C[0]);
    printf ("EOF orthogonality check:\n");
    print_matrix (N, C[0]);
*/

/*
    printf ("PC orthonormality check:\n");
    dot (N, nt, N, pc, pc, C[0]);
    print_matrix (N, C[0]);
*/

    // add this to the stack
    add_to_stack (work, eof, eig, nt, pc);

    // Cleanup?
    if (n < N) {
      printf ("freeing the padded space\n");
      free (_data);
    }

  }

  return 0;
}


// Finish processing the data for this scan
// Collapses the stack, returns whatever EOF estimates we have so far.
int endloop (Workspace *work, double *EOFs, double *EIGs, double *PCs) {
  const int N = work->num_eofs;
  const int nx = work->nx;

  printf ("freeing workspace\n");
  printf ("nx was %d, num_eofs was %d\n", nx, N);
//  printf ("leftover stacksize is %d\n", stacksize);

  //Finish off anything left on the stack (up to 0th stacklevel)
  assert (work->stacksize > 0);  // we need something to return!
  while (work->stacksize > 1) merge_eof (work);

  assert (work->stacksize == 1);

  double *eofs = work->eofstack[0];
  double *eigs = work->eigstack[0];
  int nt = work->ntstack[0];
  assert (nt > 0);
  double *pcs = work->pcstack[0];

  memcpy (EOFs, eofs, sizeof(double)*nx*N);
  memcpy (EIGs, eigs, sizeof(double)*N);
  memcpy (PCs, pcs, sizeof(double)*nt*N);

  // Use consistent sign - make the first eof value positive
  for (int i = 0; i < N; i++) {
    if (EOFs[i*nx] < 0) {
      for (int x = 0; x < nx; x++) EOFs[i*nx+x] *= -1;
      for (int t = 0; t < nt; t++) PCs[i*nt+t] *= -1;
    }
  }

  // Copy the orthonormal EOF patterns into the workspace, for the next pass
  if (work->target != NULL) free (work->target);
  work->target = eofs;

  // Free the 0th stacklevel
  free (pcs);
  work->stacksize --;
  work->count = 0;  // Reset counter

  assert (work->stacksize == 0);

  return 0;
}

// Clean up the workspace
int finish (Workspace *work) {
  assert (work->stacksize == 0);  // Stack must be collapsed first
  if (work->target != NULL) free (work->target);

  // Free the structure
  free (work);

  return 0;
}

/*** Python wrappers ***/

static PyObject *eofcore_start (PyObject *self, PyObject *args) {
  int num_eofs, nx;
  Workspace *work;
  if (!PyArg_ParseTuple(args, "ii", &num_eofs, &nx)) return NULL;
  // Make sure the input arrays are contiguous and of the right type

  // Call the C function
  start (num_eofs, nx, &work);

  return Py_BuildValue("L", (long long)(work));
}

static PyObject *eofcore_process (PyObject *self, PyObject *args) {
  Workspace *work;
  int NREC;
  double *data;
  PyArrayObject *data_array;
  if (!PyArg_ParseTuple(args, "LiO!",
    &work, &NREC, &PyArray_Type, &data_array)) return NULL;

  // Call the C function
  data = (double*)data_array->data;
  process (work, NREC, data);

  Py_RETURN_NONE;
}

static PyObject *eofcore_endloop (PyObject *self, PyObject *args) {
  Workspace *work;
  double *EOFs, *EIGs, *PCs;
  PyArrayObject *EOFs_array, *EIGs_array, *PCs_array;
  if (!PyArg_ParseTuple(args, "LO!O!O!",
    &work, &PyArray_Type, &EOFs_array, &PyArray_Type, &EIGs_array,
    &PyArray_Type, &PCs_array)) return NULL;

  // Call the C function
  EOFs = (double*)EOFs_array->data;
  EIGs = (double*)EIGs_array->data;
  PCs = (double*)PCs_array->data;
  endloop (work, EOFs, EIGs, PCs);

  Py_RETURN_NONE;
}

static PyObject *eofcore_finish (PyObject *self, PyObject *args) {
  Workspace *work;
  if (!PyArg_ParseTuple(args, "L", &work)) return NULL;

  // Call the C function
  finish (work);

  Py_RETURN_NONE;
}


static PyMethodDef EOFMethods[] = {
  {"start", eofcore_start, METH_VARARGS, ""},
  {"process", eofcore_process, METH_VARARGS, ""},
  {"endloop", eofcore_endloop, METH_VARARGS, ""},
  {"finish", eofcore_finish, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "eofcore",           /* m_name */
        NULL,                /* m_doc */
        -1,                  /* m_size */
        EOFMethods,          /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif


static PyObject *
moduleinit(void)
{
    PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("eofcore", EOFMethods);
#endif

    import_array();

    return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initeofcore(void)
    {
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_eofcore(void)
    {
        return moduleinit();
    }
#endif

