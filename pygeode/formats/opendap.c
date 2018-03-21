#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

int str2int8 (unsigned char *str, unsigned char *x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = str[i];
  }
  return 0;
}


// Convert a string to an array (assume it's encoded as network endian (big endian)
int str2int32 (unsigned char *str, unsigned int *x, int n) {
  for (int i = 0; i < n; i++) {
    int j = 4*i;
    x[i] = str[j] * (1<<24) + str[j+1] * (1<<16) + str[j+2] * (1<<8) + str[j+3];
  }
  return 0;
}

int str2int64 (unsigned char *str, unsigned long long *x, int n) {
  for (int i = 0; i < n; i++) {
    int j = 8*i;
    x[i] = str[j] * (1LL<<56) + str[j+1] * (1LL<<48) + str[j+2] * (1LL<<40) + str[j+3] * (1LL<<32)
         + str[j+4] * (1LL<<24) + str[j+5] * (1LL<<16) + str[j+6] * (1LL<<8) + str[j+7];
  }
  return 0;
}

int str2float32 (unsigned char *str, float *f, int n) {
  assert (sizeof(float) == 4);
  assert (sizeof(unsigned int) == 4);
  unsigned int *x = (unsigned int*)f;
  for (int i = 0; i < n; i++) {
    int j = 4*i;
    x[i] = str[j] * (1<<24) + str[j+1] * (1<<16) + str[j+2] * (1<<8) + str[j+3];
  }
  return 0;
}

int str2float64 (unsigned char *str, double *f, int n) {
  assert (sizeof(double) == 8);
  assert (sizeof(unsigned long long) == 4);
  unsigned long long *x = (unsigned long long*)f;
  for (int i = 0; i < n; i++) {
    int j = 8*i;
    x[i] = str[j] * (1LL<<56) + str[j+1] * (1LL<<48) + str[j+2] * (1LL<<40) + str[j+3] * (1LL<<32)
         + str[j+4] * (1LL<<24) + str[j+5] * (1LL<<16) + str[j+6] * (1LL<<8) + str[j+7];
  }
  return 0;
}


int int8toStr (unsigned char *x, char *str, int n) {
  for (int i = 0; i < n; i++) {
    str[i] = x[i];
  }
  return 0;
}

// Convert an array to a string (big endian encoding)
int int32toStr (unsigned int *x, char *str, int n) {
  for (int i = 0; i < n; i++) {
    int j = 4*i;
    str[j]   = x[i]>>24;
    str[j+1] = x[i]>>16;
    str[j+2] = x[i]>>8;
    str[j+3] = x[i];
  }
  return 0;
}

int int64toStr (unsigned long long *x, char *str, int n) {
  for (int i = 0; i < n; i++) {
    int j = 8*i;
    str[j]   = x[i]>>56;
    str[j+1] = x[i]>>48;
    str[j+2] = x[i]>>40;
    str[j+3] = x[i]>>32;
    str[j+4] = x[i]>>24;
    str[j+5] = x[i]>>16;
    str[j+6] = x[i]>>8;
    str[j+7] = x[i];
  }
  return 0;
}

int float32toStr (float *f, char *str, int n) {
  unsigned int *x = (unsigned int*)f;
  for (int i = 0; i < n; i++) {
    int j = 4*i;
    str[j]   = x[i]>>24;
    str[j+1] = x[i]>>16;
    str[j+2] = x[i]>>8;
    str[j+3] = x[i];
  }
  return 0;
}

int float64toStr (double *f, char *str, int n) {
  unsigned long long *x = (unsigned long long*)f;
  for (int i = 0; i < n; i++) {
    int j = 8*i;
    str[j]   = x[i]>>56;
    str[j+1] = x[i]>>48;
    str[j+2] = x[i]>>40;
    str[j+3] = x[i]>>32;
    str[j+4] = x[i]>>24;
    str[j+5] = x[i]>>16;
    str[j+6] = x[i]>>8;
    str[j+7] = x[i];
  }
  return 0;
}


// Python wrapper

static PyObject *opendapcore_str2int8 (PyObject *self, PyObject *args) {
  unsigned char *str;
  int slen;
  npy_intp n;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "s#", &str, &slen)) return NULL;
  n = slen;
  x = (PyArrayObject*)PyArray_SimpleNew(1,&n,NPY_INT8);

  // Call the C function
  str2int8 (str, (unsigned char*)x->data, n);

  return (PyObject*)x;
}

static PyObject *opendapcore_str2int32 (PyObject *self, PyObject *args) {
  unsigned char *str;
  int slen;
  npy_intp n;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "s#", &str, &slen)) return NULL;
  n = slen/4;
  x = (PyArrayObject*)PyArray_SimpleNew(1,&n,NPY_INT32);

  // Call the C function
  str2int32 (str, (unsigned int*)x->data, n);

  return (PyObject*)x;
}

static PyObject *opendapcore_str2int64 (PyObject *self, PyObject *args) {
  unsigned char *str;
  int slen;
  npy_intp n;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "s#", &str, &slen)) return NULL;
  n = slen/8;
  x = (PyArrayObject*)PyArray_SimpleNew(1,&n,NPY_INT64);

  // Call the C function
  str2int64 (str, (unsigned long long*)x->data, n);

  return (PyObject*)x;
}

static PyObject *opendapcore_str2float32 (PyObject *self, PyObject *args) {
  unsigned char *str;
  int slen;
  npy_intp n;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "s#", &str, &slen)) return NULL;
  n = slen/4;
  x = (PyArrayObject*)PyArray_SimpleNew(1,&n,NPY_FLOAT32);

  // Call the C function
  str2float32 (str, (float*)x->data, n);

  return (PyObject*)x;
}

static PyObject *opendapcore_str2float64 (PyObject *self, PyObject *args) {
  unsigned char *str;
  int slen;
  npy_intp n;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "s#", &str, &slen)) return NULL;
  n = slen/8;
  assert (n*8 == slen);
  x = (PyArrayObject*)PyArray_SimpleNew(1,&n,NPY_FLOAT64);

  // Call the C function
  str2float64 (str, (double*)x->data, n);

  return (PyObject*)x;
}

static PyObject *opendapcore_int8toStr (PyObject *self, PyObject *args) {
  npy_intp n;
  PyObject *obj;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
  x = (PyArrayObject*)PyArray_ContiguousFromObject(obj,NPY_INT8,0,0);
  if (x == NULL) return NULL;
  n = 1;
  for (int i = 0; i < x->nd; i++) n *= x->dimensions[i];
  char str[n];

  // Call the C function
  int8toStr ((unsigned char*)x->data, str, n);

  // Clean up local objects
  Py_DECREF (x);

  return Py_BuildValue("s#", str, n);
}

static PyObject *opendapcore_int32toStr (PyObject *self, PyObject *args) {
  npy_intp n;
  PyObject *obj;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
  x = (PyArrayObject*)PyArray_ContiguousFromObject(obj,NPY_INT32,0,0);
  if (x == NULL) return NULL;
  n = 1;
  for (int i = 0; i < x->nd; i++) n *= x->dimensions[i];
  char *str = (char*)malloc(sizeof(char)*n*4);

  // Call the C function
  int32toStr ((unsigned int*)x->data, str, n);

  // Clean up local objects
  Py_DECREF (x);

  PyObject *ret = Py_BuildValue("s#", str, n*4);
  free(str);
  return ret;
}

static PyObject *opendapcore_int64toStr (PyObject *self, PyObject *args) {
  npy_intp n;
  PyObject *obj;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
  x = (PyArrayObject*)PyArray_ContiguousFromObject(obj,NPY_INT64,0,0);
  if (x == NULL) return NULL;
  n = 1;
  for (int i = 0; i < x->nd; i++) n *= x->dimensions[i];
  char *str = (char*)malloc(sizeof(char)*n*8);

  // Call the C function
  int64toStr ((unsigned long long*)x->data, str, n);

  // Clean up local objects
  Py_DECREF (x);

  PyObject *ret = Py_BuildValue("s#", str, n*8);
  free(str);
  return ret;
}

static PyObject *opendapcore_float32toStr (PyObject *self, PyObject *args) {
  npy_intp n;
  PyObject *obj;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
  x = (PyArrayObject*)PyArray_ContiguousFromObject(obj,NPY_FLOAT32,0,0);
  if (x == NULL) return NULL;
  n = 1;
  for (int i = 0; i < x->nd; i++) n *= x->dimensions[i];
  char *str = (char*)malloc(sizeof(char)*n*4);

  // Call the C function
  float32toStr ((float*)x->data, str, n);

  // Clean up local objects
  Py_DECREF (x);

  PyObject *ret = Py_BuildValue("s#", str, n*4);
  free(str);
  return ret;
}

static PyObject *opendapcore_float64toStr (PyObject *self, PyObject *args) {
  npy_intp n;
  PyObject *obj;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
  x = (PyArrayObject*)PyArray_ContiguousFromObject(obj,NPY_FLOAT64,0,0);
  if (x == NULL) return NULL;
  n = 1;
  for (int i = 0; i < x->nd; i++) n *= x->dimensions[i];
  char *str = (char*)malloc(sizeof(char)*n*8);

  // Call the C function
  float64toStr ((double*)x->data, str, n);

  // Clean up local objects
  Py_DECREF (x);

  PyObject *ret = Py_BuildValue("s#", str, n*8);
  free(str);
  return ret;
}


static PyMethodDef OpenDAPMethods[] = {
  {"str2int8", opendapcore_str2int8, METH_VARARGS, ""},
  {"str2int32", opendapcore_str2int32, METH_VARARGS, ""},
  {"str2int64", opendapcore_str2int64, METH_VARARGS, ""},
  {"str2float32", opendapcore_str2float32, METH_VARARGS, ""},
  {"str2float64", opendapcore_str2float64, METH_VARARGS, ""},
  {"int8toStr", opendapcore_int8toStr, METH_VARARGS, ""},
  {"int32toStr", opendapcore_int32toStr, METH_VARARGS, ""},
  {"int64toStr", opendapcore_int64toStr, METH_VARARGS, ""},
  {"float32toStr", opendapcore_float32toStr, METH_VARARGS, ""},
  {"float64toStr", opendapcore_float64toStr, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "opendapcore",       /* m_name */
        NULL,                /* m_doc */
        -1,                  /* m_size */
        OpenDAPMethods,      /* m_methods */
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
    m = Py_InitModule("opendapcore", OpenDAPMethods);
#endif

    import_array();

    return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initopendapcore(void)
    {
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_opendapcore(void)
    {
        return moduleinit();
    }
#endif

