#include <Python.h>
#include <numpy/arrayobject.h>

// Wrapper for GSL functions

#include <gsl/gsl_interp.h>
#include <gsl/gsl_errno.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

/*
  Interpolate a bunch of arrays which may contain Nan
  (Call the GSL interp functions in a loop)

  narrays: number of data arrays to interpolate
  nxin:  length of a single input array (constant)
  nxout: length of a single output array
  xin: the x coordinates of the inputs
  yin: the corresponding array values
  xout: the x coordinates to interpolate to
  loop_xin: does xin change for each array? (0=no, 1=yes)
  loop_xout: does xout change for each array? (0=no, 1=yes)
  type: type of interpolation (i.e., gsl_interp_cspline)
  d_below: slope of extrapolation below the input field
  d_above: slope of extrapolation above the input field

*/

// Raw C function
int interpgsl (int narrays, int nxin, int nxout,
               double *xin, double *yin, double *xout, double *yout,
               int loop_xin, int loop_xout,
               double d_below, double d_above,
               const gsl_interp_type *type, int omit_nm)           {

  int i, j, n, start, end, step, status; 
  double *local_xin, *local_xout, *local_yin;
  double dx, xmin, ymin, xmax, ymax;

  // Check the order of the x coordinates
  // If the order seems to be reversed, then flip the sign of the coordinates
  if (nxin > 1) {
    i = 0;
    do {
      dx = xin[i + 1] - xin[i];
    } while (!isfinite(dx) && ++i < nxin);
  } else {
    dx = 1.;
  }

  if (!isfinite(dx) || dx == 0. ) { return -1; }
  
  if (dx < 0.) {
    start = nxin - 1; 
    end = -1;
    step = -1;
  } else {
    start = 0;
    end = nxin;
    step = 1;
  }

  local_xin = (double *) malloc(nxin * sizeof(double));
  local_yin = (double *) malloc(nxin * sizeof(double));
  local_xout = (double *) malloc(nxout * sizeof(double));

  if (!loop_xout) {
    for (i = 0; i < nxout; i++) local_xout[i] = xout[i]; 
  }

  gsl_set_error_handler_off();

  gsl_interp *interp = NULL;
  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  if (NULL == acc) { PyErr_SetString(PyExc_MemoryError, "Failed to allocate gsl_interp_accel object."); goto clean_err;}

  for (n = 0; n < narrays; n++) {
    // Search for nans, resort if necessary
    for (i = start, j = 0; i != end; i += step) {
      if (isfinite(yin[i]) && isfinite(xin[i])) {
        // !!! Omit non-monotonic bits if flag is set
        if (omit_nm && j > 0 && xin[i] < local_xin[j - 1]) {
          continue;
        }
        local_xin[j] = xin[i];
        local_yin[j] = yin[i];
        j++;
      }
    }

    // Save extremal points for extrapolation
    xmin = local_xin[0];
    xmax = local_xin[j-1];

    if (loop_xout) {
      for (i = 0; i < nxout; i++) local_xout[i] = xout[i]; 
    }

    // If we have insufficient points for the interpolation type, then fill
    // the output array with NaNs.
    if (j < type->min_size) {
      for (j = 0; j < nxout; j++, xout++, yout++) *yout = NAN;
      // Update the input/output pointers, be consistent with the normal case
      // below.
      if (loop_xin) { xin += nxin; }
      if (!loop_xout) { xout -= nxout; }
      yin += nxin;
      continue;
    }

    interp = gsl_interp_alloc (type, j);
    if (NULL == interp) { PyErr_SetString(PyExc_MemoryError, "Failed to allocate interp object."); goto clean_err;}

    status = gsl_interp_init (interp, local_xin, local_yin, j);
    if (status) { PyErr_Format(PyExc_ValueError, "Interpolation error (%s), usually due to non-monotonic source data. In array %d.", gsl_strerror(status), n); goto clean_err;}

    gsl_interp_eval_e (interp, local_xin, local_yin, xmin, acc, &ymin);
    gsl_interp_eval_e (interp, local_xin, local_yin, xmax, acc, &ymax);

    for (j = 0; j < nxout; j++, xout++, yout++) {
      // Linearly extrapolate outside bounds
      if (*xout < xmin) {
        *yout = ymin + d_below * (*xout - xmin);
      } else if (*xout > xmax) {
        *yout = ymax + d_above * (*xout - xmax);
      } else {
        gsl_interp_eval_e (interp, local_xin, local_yin, *xout, acc, yout);
      }
    }

    gsl_interp_accel_reset (acc);
    gsl_interp_free (interp);

    if (loop_xin) { xin += nxin; }
    if (!loop_xout) { xout -= nxout; }
    yin += nxin;
  }

  gsl_interp_accel_free(acc);

  // Clean up local coordinate arrays
  free (local_xin);
  free (local_yin);
  free (local_xout);

  return 0;

clean_err:
  if (interp) { gsl_interp_free (interp); }
  if (acc)    { gsl_interp_accel_free (acc); }

  free (local_xin);
  free (local_yin);
  free (local_xout);

  return -1;
}

static PyObject *interpcore_interpgsl (PyObject *self, PyObject *args) {
  int narrays, nxin, nxout, status;
  double *xin, *yin, *xout, *yout;
  int loop_xin, loop_xout, omit_nm;
  double d_below, d_above;
  const gsl_interp_type *type;
  PyObject *xin_obj, *yin_obj, *xout_obj;
  PyArrayObject *xin_array, *yin_array, *xout_array, *yout_array;
  const char *type_str;
  // Assume the output array is contiguous and of the right type
  if (!PyArg_ParseTuple(args, "iiiOOOO!iiddsi",
    &narrays, &nxin, &nxout, &xin_obj, &yin_obj, &xout_obj,
    &PyArray_Type, &yout_array,
    &loop_xin, &loop_xout, &d_below, &d_above, &type_str, &omit_nm)) return NULL;
  // Make sure the input arrays are contiguous and of the right type
  xin_array = (PyArrayObject*)PyArray_ContiguousFromObject(xin_obj,NPY_DOUBLE,1,0);
  yin_array = (PyArrayObject*)PyArray_ContiguousFromObject(yin_obj,NPY_DOUBLE,1,0);
  xout_array = (PyArrayObject*)PyArray_ContiguousFromObject(xout_obj,NPY_DOUBLE,1,0);
  if (xin_array == NULL || yin_array == NULL || xout_array == NULL) return NULL;

  xin = (double*)xin_array->data;
  yin = (double*)yin_array->data;
  xout = (double*)xout_array->data;
  yout = (double*)yout_array->data;

  // Determine the interpolation type
  if (strcmp(type_str,"linear")==0) type = gsl_interp_linear;
  else if (strcmp(type_str,"polynomial")==0) type = gsl_interp_polynomial;
  else if (strcmp(type_str,"cspline")==0) type = gsl_interp_cspline;
  else if (strcmp(type_str,"cspline_periodic")==0) type = gsl_interp_cspline_periodic;
  else if (strcmp(type_str,"akima")==0) type = gsl_interp_akima;
  else if (strcmp(type_str,"akima_periodic")==0) type = gsl_interp_akima_periodic;
  else {
    PyErr_SetString(PyExc_KeyError, type_str);
    return NULL;
  }

  // Call the C function
  status = interpgsl (narrays, nxin, nxout, xin, yin, xout, yout, loop_xin, loop_xout, d_below, d_above, type, omit_nm);

  // Clean up internal objects
  Py_DECREF(xin_array);
  Py_DECREF(xout_array);
  Py_DECREF(yin_array);

  // If the interpolation fails the error string should be set by interpgsl
  if (status) { return NULL; }

  Py_RETURN_NONE;
}

static PyMethodDef InterpMethods[] = {
  {"interpgsl", interpcore_interpgsl, METH_VARARGS, "Interpolate an array"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "interpcore",        /* m_name */
        NULL,                /* m_doc */
        -1,                  /* m_size */
        InterpMethods,       /* m_methods */
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
    m = Py_InitModule("interpcore", InterpMethods);
#endif

    import_array();

    return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initinterpcore(void)
    {
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_interpcore(void)
    {
        return moduleinit();
    }
#endif

