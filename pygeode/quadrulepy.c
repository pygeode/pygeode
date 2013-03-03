#include <Python.h>

# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <math.h>
# include <time.h>

# include "quadrule.h"


static PyObject *quadrulepy_legendre_compute (PyObject *self, PyObject *args) {
  int order;
  double *xtab, *weight;
  long long xtab_L, weight_L;
  if (!PyArg_ParseTuple(args, "iLL", &order, &xtab_L, &weight_L)) return NULL;
  // Do some unsafe casting to pointers.
  // What's the worst that could happen?
  xtab = (double*)xtab_L;
  weight = (double*)weight_L;
  legendre_compute (order, xtab, weight);
  Py_RETURN_NONE;
}



static PyMethodDef QuadrulepyMethods[] = {
  {"legendre_compute", quadrulepy_legendre_compute, METH_VARARGS, "Compute Gaussian quadrature"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initquadrulepy(void) {
  (void) Py_InitModule("quadrulepy", QuadrulepyMethods);
}

