#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>   // for sqrt

// low-level functions to assist in calculating EOFs

// Given a piece of a time series, accumulate new SVDs and PC values.
// Assume the new SVD fields have already been initialized to zero.
// Arrays:
// input1[t][x1]
// input2[t][x2]
// oldeofs[n][x2]
// neweofs[n][x1]
// pcs[t][n]
//
// neweofs = <input1, pc> over t
// where pc = <input2, oldeofs> over x2
//
// Note the dimension change in the output!

int build_svds (int num_svds, int nt, int nx1, int nx2,
                double *input1, double *input2,
                double *oldeofs, double *neweofs, double *pcs) {

  for (int n = 0; n < num_svds; n++) {
    double *oldeofs_ = oldeofs + n*nx2;
    double *neweofs_ = neweofs + n*nx1;
    for (int t = 0; t < nt; t++) {
      double *input1_ = input1 + t*nx1;
      double *input2_ = input2 + t*nx2;
      double PC = 0;
      for (int x = 0; x < nx2; x++) PC += input2_[x] * oldeofs_[x];
      for (int x = 0; x < nx1; x++) neweofs_[x] += PC * input1_[x];
      pcs[t*num_svds + n] = PC;
    }
  }

  return 0;
}


// Given a piece of a time series, accumulate new EOF fields and PC values.
// Assume the new EOF fields have already been initialized to zero.
// Arrays:
// input[t][x]
// oldeofs[e][x]
// neweofs[e][x]
// pcs[t][e]
// This is the symmetric case of build_svds, when the left and right
// orthogonal vectors are identical
int build_eofs (int num_eofs, int nt, int nx, double *input, double *oldeofs, double *neweofs,
                double *pcs) {

  for (int e = 0; e < num_eofs; e++) {
    double *oldeofs_ = oldeofs + e*nx;
    double *neweofs_ = neweofs + e*nx;
//    double *pcs_ = pcs + e*nt;
    for (int t = 0; t < nt; t++) {
      double *input_ = input + t*nx;
      double PC = 0;
      for (int x = 0; x < nx; x++) PC += input_[x] * oldeofs_[x];
      for (int x = 0; x < nx; x++) neweofs_[x] += PC * input_[x];
//      pcs_[t] = PC;
      pcs[t*num_eofs + e] = PC;
    }
  }

  return 0;
}


// Dot product of two matrices
// Arrays:
// U[e1][x]
// V[e2][x]
// out[e1][e2]
int dot (int num_eofs, int nx, double *U, double *V, double *out) {
  for (int e1 = 0; e1 < num_eofs; e1++) {
    double *U_ = U + e1*nx;
    double *out_ = out + e1*num_eofs;
    for (int e2 = 0; e2 < num_eofs; e2++) {
      double *V_ = V + e2*nx;
      double w = 0;
      for (int x = 0; x < nx; x++) w += U_[x] * V_[x];
      out_[e2] = w;
    }
  }

  return 0;
}

// Normalize the rows of a matrix
// (in-place)
// Use consistent sign convention (1st element of each row is positive)
// TODO: more robust version, i.e. where 1st element is very close to zero

// Arrays:
// U[e][x]
int normalize (int num_eofs, int nx, double *U) {
  for (int e = 0; e < num_eofs; e++) {
    double *U_ = U + e*nx;
    // Calculate magnitude
    double n = 0;
    for (int x = 0; x < nx; x++) n += U_[x] * U_[x];
    n = sqrt(n);
    // Normalize
    for (int x = 0; x < nx; x++) U_[x] /= n;
  }

  // Consistent sign
  for (int e = 0; e < num_eofs; e++) {
    double *U_ = U + e*nx;
    if (U_[0] < 0) {
      for (int x = 0; x < nx; x++) U_[x] *= -1;
    }
  }

  return 0;
}

// Transform a row matrix
// Arrays:
// P[e][e]
// U[e][x]
// Note: transposing P before doing matrix multiplication?
int transform (int num_eofs, int nx, double *P, double *U) {
  double work[num_eofs];
  for (int x = 0; x < nx; x++) {
    for (int e1 = 0; e1 < num_eofs; e1++) {
      double sum = 0;
      for (int e2 = 0; e2 < num_eofs; e2++) {
        sum += P[e2*num_eofs + e1] * U[e2*nx + x];
      }
      work[e1] = sum;
    }
    for (int e1 = 0; e1 < num_eofs; e1++) U[e1*nx + x] = work[e1];
  }

  return 0;
}

// Do a sign flip on EOFs and PCs so that the covariance between PC pairs is positive
// Arrays:
// pcA[t][e]
// pcB[t][e]
// eofB[e][x]
int fixcov (int num_eofs, int nt, int nx, double *pcA, double *pcB, double *eofB) {
  for (int e = 0; e < num_eofs; e++) {
    double cov = 0;
    for (int t = 0; t < nt; t++) cov += pcA[t*num_eofs + e] * pcB[t*num_eofs + e];
    if (cov < 0) {
      for (int t = 0; t < nt; t++) pcB[t*num_eofs + e] *= -1;
      double *eofB_ = eofB + e*nx;
      for (int x = 0; x < nx; x++) eofB_[x] *= -1;
    }
  }
  return 0;
}

/*** Python wrappers ***/

static PyObject *svdcore_build_svds (PyObject *self, PyObject *args) {
  int num_svds, nt, nx1, nx2;
  double *input1, *input2, *oldeofs, *neweofs, *pcs;
  PyArrayObject *input1_array, *input2_array, *oldeofs_array, *neweofs_array, *pcs_array;

  // Assume the arrays are contiguous and of the right type
  if (!PyArg_ParseTuple(args, "iiiiO!O!O!O!O!",
    &num_svds, &nt, &nx1, &nx2,
    &PyArray_Type, &input1_array, &PyArray_Type, &input2_array,
    &PyArray_Type, &oldeofs_array, &PyArray_Type, &neweofs_array,
    &PyArray_Type, &pcs_array)) return NULL;

  input1 = (double*)input1_array->data;
  input2 = (double*)input2_array->data;
  oldeofs = (double*)oldeofs_array->data;
  neweofs = (double*)neweofs_array->data;
  pcs = (double*)pcs_array->data;

  // Call the C function
  int ret = build_svds (num_svds, nt, nx1, nx2, input1, input2, oldeofs, neweofs, pcs);

  return Py_BuildValue("i", ret);
}

static PyObject *svdcore_build_eofs (PyObject *self, PyObject *args) {
  int num_eofs, nt, nx;
  double *input, *oldeofs, *neweofs, *pcs;
  PyArrayObject *input_array, *oldeofs_array, *neweofs_array, *pcs_array;

  // Assume the arrays are contiguous and of the right type
  if (!PyArg_ParseTuple(args, "iiiO!O!O!O!",
    &num_eofs, &nt, &nx,
    &PyArray_Type, &input_array,
    &PyArray_Type, &oldeofs_array, &PyArray_Type, &neweofs_array,
    &PyArray_Type, &pcs_array)) return NULL;

  input = (double*)input_array->data;
  oldeofs = (double*)oldeofs_array->data;
  neweofs = (double*)neweofs_array->data;
  pcs = (double*)pcs_array->data;

  // Call the C function
  int ret = build_eofs (num_eofs, nt, nx, input, oldeofs, neweofs, pcs);

  return Py_BuildValue("i", ret);
}



static PyObject *svdcore_dot (PyObject *self, PyObject *args) {
  int num_eofs, nx;
  double *U, *V, *out;
  PyArrayObject *U_array, *V_array, *out_array;

  // Assume the arrays are contiguous and of the right type
  if (!PyArg_ParseTuple(args, "iiO!O!O!",
    &num_eofs, &nx, &PyArray_Type, &U_array, &PyArray_Type, &V_array,
    &PyArray_Type, &out_array)) return NULL;

  U = (double*)U_array->data;
  V = (double*)V_array->data;
  out = (double*)out_array->data;

  // Call the C function
  dot (num_eofs, nx, U, V, out);

  Py_RETURN_NONE;
}

static PyObject *svdcore_transform (PyObject *self, PyObject *args) {
  int num_eofs, nx;
  double *P, *U;
  PyArrayObject *P_array, *U_array;

  // Assume the arrays are contiguous and of the right type
  if (!PyArg_ParseTuple(args, "iiO!O!",
    &num_eofs, &nx, &PyArray_Type, &P_array, &PyArray_Type, &U_array)) return NULL;

  P = (double*)P_array->data;
  U = (double*)U_array->data;

  // Call the C function
  transform (num_eofs, nx, P, U);

  Py_RETURN_NONE;
}


static PyObject *svdcore_normalize (PyObject *self, PyObject *args) {
  int num_eofs, nx;
  double *U;
  PyArrayObject *U_array;

  // Assume the arrays are contiguous and of the right type
  if (!PyArg_ParseTuple(args, "iiO!",
    &num_eofs, &nx, &PyArray_Type, &U_array)) return NULL;

  U = (double*)U_array->data;

  // Call the C function
  normalize (num_eofs, nx, U);

  Py_RETURN_NONE;
}

static PyObject *svdcore_fixcov (PyObject *self, PyObject *args) {
  int num_eofs, nt, nx;
  double *pcA, *pcB, *eofB;
  PyArrayObject *pcA_array, *pcB_array, *eofB_array;

  // Assume the arrays are contiguous and of the right type
  if (!PyArg_ParseTuple(args, "iiiO!O!O!",
    &num_eofs, &nt, &nx, &PyArray_Type, &pcA_array, &PyArray_Type, &pcB_array,
    &PyArray_Type, &eofB_array)) return NULL;

  pcA = (double*)pcA_array->data;
  pcB = (double*)pcB_array->data;
  eofB = (double*)eofB_array->data;

  // Call the C function
  fixcov (num_eofs, nt, nx, pcA, pcB, eofB);

  Py_RETURN_NONE;
}

static PyMethodDef SVDMethods[] = {
  {"build_svds", svdcore_build_svds, METH_VARARGS, ""},
  {"build_eofs", svdcore_build_eofs, METH_VARARGS, ""},
  {"dot", svdcore_dot, METH_VARARGS, ""},
  {"transform", svdcore_transform, METH_VARARGS, ""},
  {"normalize", svdcore_normalize, METH_VARARGS, ""},
  {"fixcov", svdcore_fixcov, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "svdcore",           /* m_name */
        NULL,                /* m_doc */
        -1,                  /* m_size */
        SVDMethods,          /* m_methods */
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
    m = Py_InitModule("svdcore", SVDMethods);
#endif

    import_array();

    return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initsvdcore(void)
    {
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_svdcore(void)
    {
        return moduleinit();
    }
#endif

