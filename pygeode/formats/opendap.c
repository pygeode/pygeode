#include <Python.h>
#include <numpy/arrayobject.h>

int str2int8 (char *str, unsigned char *x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = str[i];
  }
  return 0;
}


// Convert a string to an array (assume it's encoded as network endian (big endian)
int str2int32 (char *str, unsigned int *x, int n) {
  for (int i = 0; i < n; i++) {
    int j = 4*i;
    x[i] = str[j] * (1<<24) + str[j+1] * (1<<16) + str[j+2] * (1<<8) + str[j+3];
  }
  return 0;
}

int str2int64 (char *str, unsigned long long *x, int n) {
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


// Python wrapper

static PyObject *opendapcore_str2int8 (PyObject *self, PyObject *args) {
  char *str;
  npy_intp n;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "s", &str)) return NULL;
  n = strlen(str);
  x = (PyArrayObject*)PyArray_SimpleNew(1,&n,NPY_INT8);

  // Call the C function
  str2int8 (str, (unsigned char*)x->data, n);

  return (PyObject*)x;
}

static PyObject *opendapcore_str2int32 (PyObject *self, PyObject *args) {
  char *str;
  npy_intp n;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "s", &str)) return NULL;
  n = strlen(str)/4;
  x = (PyArrayObject*)PyArray_SimpleNew(1,&n,NPY_INT32);

  // Call the C function
  str2int32 (str, (unsigned int*)x->data, n);

  return (PyObject*)x;
}

static PyObject *opendapcore_str2int64 (PyObject *self, PyObject *args) {
  char *str;
  npy_intp n;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "s", &str)) return NULL;
  n = strlen(str)/8;
  x = (PyArrayObject*)PyArray_SimpleNew(1,&n,NPY_INT64);

  // Call the C function
  str2int64 (str, (unsigned long long*)x->data, n);

  return (PyObject*)x;
}

static PyObject *opendapcore_int8toStr (PyObject *self, PyObject *args) {
  npy_intp n;
  PyObject *obj;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
  x = (PyArrayObject*)PyArray_ContiguousFromObject(obj,NPY_INT8,1,1);
  if (x == NULL) return NULL;
  n = x->dimensions[0];
  char str[n+1];
  str[n] = 0;

  // Call the C function
  int8toStr ((unsigned char*)x->data, str, n);

  // Clean up local objects
  Py_DECREF (x);

  return Py_BuildValue("s", str);
}

static PyObject *opendapcore_int32toStr (PyObject *self, PyObject *args) {
  npy_intp n;
  PyObject *obj;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
  x = (PyArrayObject*)PyArray_ContiguousFromObject(obj,NPY_INT32,1,1);
  if (x == NULL) return NULL;
  n = x->dimensions[0];
  char str[n*4+1];
  str[n*4] = 0;

  // Call the C function
  int32toStr ((unsigned int*)x->data, str, n);

  // Clean up local objects
  Py_DECREF (x);

  return Py_BuildValue("s", str);
}

static PyObject *opendapcore_int64toStr (PyObject *self, PyObject *args) {
  npy_intp n;
  PyObject *obj;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
  x = (PyArrayObject*)PyArray_ContiguousFromObject(obj,NPY_INT64,1,1);
  if (x == NULL) return NULL;
  n = x->dimensions[0];
  char str[n*8+1];
  str[n*8] = 0;

  // Call the C function
  int64toStr ((unsigned long long*)x->data, str, n);

  // Clean up local objects
  Py_DECREF (x);

  return Py_BuildValue("s", str);
}


static PyMethodDef OpenDAPMethods[] = {
  {"str2int8", opendapcore_str2int8, METH_VARARGS, ""},
  {"str2int32", opendapcore_str2int32, METH_VARARGS, ""},
  {"str2int64", opendapcore_str2int64, METH_VARARGS, ""},
  {"int8toStr", opendapcore_int8toStr, METH_VARARGS, ""},
  {"int32toStr", opendapcore_int32toStr, METH_VARARGS, ""},
  {"int64toStr", opendapcore_int64toStr, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initopendapcore(void) {
  (void) Py_InitModule("opendapcore", OpenDAPMethods);
  import_array();
}

