#include <Python.h>
#include <numpy/arrayobject.h>

# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <math.h>
# include <time.h>

# include "quadrule.h"


static PyObject *quadrulepy_legendre_compute (PyObject *self, PyObject *args) {
  int order;
  double *xtab, *weight;
  PyArrayObject *xtab_array, *weight_array;
  PyObject *result;
  if (!PyArg_ParseTuple(args, "i", &order)) return NULL;
  // Allocate the output arrays
  xtab_array = (PyArrayObject*)PyArray_FromDims(1,&order,NPY_DOUBLE);
  weight_array = (PyArrayObject*)PyArray_FromDims(1,&order,NPY_DOUBLE);
  if (xtab_array == NULL || weight_array == NULL) return NULL;
  // Extract C arrays
  xtab = (double*)xtab_array->data;
  weight = (double*)weight_array->data;

  legendre_compute (order, xtab, weight);

  // Return the 2 arrays
  result = Py_BuildValue("(O,O)", xtab_array, weight_array);
  if (result == NULL) return NULL;
  Py_DECREF(xtab_array);
  Py_DECREF(weight_array);
  return result;
}



static PyMethodDef QuadrulepyMethods[] = {
  {"legendre_compute", quadrulepy_legendre_compute, METH_VARARGS, "Compute Gaussian quadrature"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initquadrulepy(void) {
  (void) Py_InitModule("quadrulepy", QuadrulepyMethods);
  import_array();
}

