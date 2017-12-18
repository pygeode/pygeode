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
  npy_intp order_np;
  PyArrayObject *xtab_array, *weight_array;
  PyObject *result;
  if (!PyArg_ParseTuple(args, "i", &order)) return NULL;
  order_np = order;
  // Allocate the output arrays
  xtab_array = (PyArrayObject*)PyArray_SimpleNew(1,&order_np,NPY_DOUBLE);
  weight_array = (PyArrayObject*)PyArray_SimpleNew(1,&order_np,NPY_DOUBLE);
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

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "quadrulepy",        /* m_name */
        NULL,                /* m_doc */
        -1,                  /* m_size */
        QuadrulepyMethods,   /* m_methods */
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
    m = Py_InitModule("quadrulepy", QuadrulepyMethods);
#endif

    import_array();

    return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initquadrulepy(void)
    {
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_quadrulepy(void)
    {
        return moduleinit();
    }
#endif

