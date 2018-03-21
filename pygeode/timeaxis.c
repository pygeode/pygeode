#include <Python.h>
#include <numpy/arrayobject.h>

/*

  Helper functions for manipulating time axes

  (Things that are too awkward to handle in numpy)

*/

#include <assert.h>
#include <stdio.h>
#include <string.h>  // for memcpy
#include <stdlib.h>
#include <math.h>

//TODO:  allow unsorted time axes as inputs
//     (pre-sort them before continuing)

// Find a particular record, return the index
// (Binary search on sorted data)
int get_index (int natts, int *values, int n, int *key) {
/*
  printf ("looking for: ");
  for (int a = 0; a < natts; a++) printf ("%d ", key[a]);
  printf ("\n");
*/
  int lower = 0;
  int upper = n-1;

  int match = 0;
  int i;
  while (!match) {
    if (lower > upper) {
//      printf ("no match\n");
      return -1;  // no match
    }

    i = (lower+upper)/2;

//    printf ("try i = %d (lower=%d,upper=%d)\n", i, lower, upper);

    for (int a = 0; a < natts; a++) {
      if (key[a] < values[i*natts+a]) {
        upper = i-1;
        break;
      }
      if (key[a] > values[i*natts+a]) {
        lower = i+1;
        break;
      }
      // Matched?
      if (a == natts-1) match = 1;
    }
  }
//  printf ("got a match %d\n", i);
  return i;
}

// Find indices mapping the input times to the output times
int get_indices (int natts, int *invalues, int n_in, int *outvalues, int n_out, int *indices) {
  for (int i = 0; i < n_out; i++) {
    int ind = get_index(natts, invalues, n_in, outvalues + i*natts);
// allow things to not match (partial map_to's are allowed now)
//    if (ind == -1) return -1;
    indices[i] = ind;
  }

  return 0;
}

// Return only unique elements
// Assume the inputs are somewhat sorted, in the sense that the output times
// can be constructed from the inputs without the need to sort them
int uniquify (int natts, int *in_atts, int n_in, int *out_atts, int *n_out) {
  int nout = 0;
  for (int i = 0; i < n_in; i++) {

    int index = -1;
    if (nout > 0) {
      // Try and find this time value in the current list of outputs
      index = get_index (natts, out_atts, nout, &(in_atts[i*natts]));
    }
    // new value?
    if (nout == 0 || index == -1) {


      for (int a = 0; a < natts; a++) {
        out_atts[nout*natts + a] = in_atts[i*natts + a];
      }
      
      nout++;

    }

  }
  *n_out = nout;
  return 0;
}

// Return the index where the specified value(s) can be inserted to maintain order
// (Designed to work similarly to np.searchsorted, but over time axis with multiple fields)
//TODO
int searchsorted (int natts, int *sorted, int n_sorted, int *lookup, int n_lookup) {
  return 0;
}

// Return the index of the nearest lower match to a time axis -- which is just lower (or equal)
// to the desired value.
// Returns -1 where there is no lower match available
int nearest_lower (int natts, int *sorted, int n_sorted, int *lookup, int n_lookup, int *results) {
  // Loop over each requested field
  for (int i = 0; i < n_lookup; i++) {
//    printf ("i = %d\n", i);

    int j = i*natts;
    // Binary search into sorted data
    int lower = 0;
    int upper = n_sorted-1;
    while (lower <= upper) {
      int middle = (lower+upper+1)/2;
      int ind = middle*natts;
      int le = 1;
      for (int a = 0; a < natts; a++) {
        if (sorted[ind+a] < lookup[j+a]) break;
        if (sorted[ind+a] > lookup[j+a]) {le = 0; break;}
      }

      if (le == 1) {
//        for (int a = 0; a < natts; a++) printf (" %d", sorted[ind+a]);
//        printf (" <=");
//        for (int a = 0; a < natts; a++) printf (" %d", lookup[j+a]);
//        printf ("  [%d,%d,%d]\n", upper, middle, lower);
        lower = middle;
        if (lower == upper) {
//          printf ("break\n");
          break;
        }
      }
      else {
//        for (int a = 0; a < natts; a++) printf (" %d", sorted[ind+a]);
//        printf (" >");
//        for (int a = 0; a < natts; a++) printf (" %d", lookup[j+a]);
//        printf ("  [%d,%d,%d]\n", upper, middle, lower);
        upper = middle-1;
      }
    }
    if (lower == upper) results[i] = lower;
    else results[i] = -1;
  }
  return 0;
}


int cmpval (int natts, int *x, int *y) {
  for (int att = 0; att < natts; att++) {
    int c = x[att] - y[att];
    if (c != 0) return c;
  }
  return 0;
}

// Common map between two time axes
int common_map (int natts, int na, int *a, int nb, int *b, int *nmap, int *a_map, int *b_map) {

  #define CMP(x,xi,y,yi) cmpval(natts, (x)+(xi)*(natts), (y)+(yi)*(natts))

  int ai = 0, bi = 0;
  int m = 0;
  while (ai < na && bi < nb) {
    int cmp = CMP(a,ai,b,bi);
    // A match?
    if (cmp == 0) {
      assert (m < na || m < nb);
      a_map[m] = ai;
      b_map[m] = bi;
      m++;
      // Special cases: one of the arrays  has repeated values
      if (ai < na-1) if (CMP(a,ai,a,ai+1) == 0) {
        ai++;
        continue;
      }
      if (bi < nb-1) if (CMP(b,bi,b,bi+1) == 0) {
        bi++;
        continue;
      }
      // If no repeated values, it is safe to increment both arrays
      ai++;
      bi++;
      continue;
    }

    // No match, increment whichever array had the smaller value
    if (cmp < 0) ai++;
    else bi++;
  }

  *nmap = m;

  return 0;

  #undef CMP
}




/*

   Standard calendar functions
   (Handles leap years, etc.)

*/

// Check if a year is a leap year
#define isleap(y) (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0))

// Count the number of days in between two years
int ndays (int year1, int year2) {
//  // We need the years to be positive, for mod operator to work properly
//  assert (year1 >= 0 && year2 >= 0);
/*
  if (year1 > year2) {
//    printf ("warning: ndays: %d > %d\n", year1, year2);
    return -ndays(year2, year1);
  }
*/

  // Decrement years (makes everything else easier)
  year1--; year2--;

  // First, ignore leap years
  int out = (year2-year1)*365;

  // Now, count how many years that are divisible by 4
  out += year2/4 - year1/4;

  // Exclude years that are divisible by 100
  out -= year2/100 - year1/100;

  // .. but include years that are divisible by 400  (whew!)
  out += year2/400 - year1/400;

  return out;

}


// Make sure we have a valid month (or we'll get a segmentation fault!)
#define check_month(m) if (imonth <= 0 || imonth > 12) { PyErr_SetString (PyExc_IndexError, "month is out of range"); return 0; }

// Convert an absolute date to a relative date (seconds since start date)
int date_as_val_std (int n, int iyear, int imonth, int iday,
                     int ihour, int iminute, int isecond,
                     int *year, int *month, int *day,
                     int *hour, int *minute, int *second,
                     long long int *val) {

  static const int month2doy[2][13] = {
    {0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
    {0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335}
  };

  check_month(imonth);
  long long int ref = ( month2doy[isleap(iyear)][imonth] + iday - 1) * 86400 + ihour*3600 + iminute*60 + isecond;

  // Take the difference between the date array and the start date
  for (int i = 0; i < n; i++) {

    int y = year[i];

    // Wrap month
    int m = month[i];
    int dy = (m < 1) ? (m-12)/12 : (m-1)/12;
    y += dy;
    m -= dy*12;

    long long int v = ( ndays(iyear,y) + month2doy[isleap(y)][m] + day[i] - 1) * 86400LL;
    v += hour[i]*3600 + minute[i]*60 + second[i];
    v -= ref;
    val[i] = v;
  }

  return 0;
}


// Convert a relative date (seconds since start date) to an absolute date
int val_as_date_std (int n, int iyear, int imonth, int iday,
                     int ihour, int iminute, int isecond,
                     long long int *val,
                     int *year, int *month, int *day,
                     int *hour, int *minute, int *second) {

  static const int month2doy[2][13] = {
    {0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
    {0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335}
  };

  check_month(imonth);
  long long int ref = ( month2doy[isleap(iyear)][imonth] + iday - 1) * 86400 + ihour*3600 + iminute*60 + isecond;

  for (int i = 0; i < n; i++) {
    long long int s = val[i] + ref;  // Seconds since the beginning of the year of start date
    int y = iyear;
    // Make sure s is positive
    // (estimate number of years to offset, ok so long as overestimate - just don't want a negative s)
//    assert (s >= 0);  // remove this later
    while (s < 0LL) {
      int y_back = y - 1 - (int) ((-s)/365LL/86400LL);
      //if (y_back < 0) y_back = 0;
      s += ndays(y_back, y) * 86400LL;
      y = y_back;
    }
    assert (s >= 0);

    long long int d = s / 86400;  // seconds to days
//    printf ("%4d-%02d-%02d %5lld  ->  %4d-01-01 %5lld", iyear,imonth,iday,val[i]/86400,y,d);

    // Store the time info
    s -= d * 86400;
    hour[i] = s / 3600;
    s %= 3600;
    minute[i] = s / 60;
    second[i] = s % 60;

    // Now, take the day offset ('d'), and get a date
    // Note: 'd' is the number of days *since* start date
    assert (d >= 0);

    // Adjust the year, if the # of days is *way* off
    // Underestimate the year jump, so we don't go *too* far and make d negative
    {
      int y_forward = y + d/366;
      d -= ndays (y, y_forward);
      y = y_forward;
    }
//    printf ("  -> %4d-01-01 %3lld", y, d);

    // Now, minor adjustment if d is still too large
//    printf ("( %lld >= %d?)", d, ndays(y,y+1));
    while (d >= ndays(y,y+1)){
      d -= ndays(y,y+1);
      y++;
    }
//    printf ("  -> %4d-01-01 %3lld", y, d);

    assert (d >= 0);

    year[i] = y;

    // Get the month / day
    {
      int m = d / 29 + 1;  // Estimate (allow overestimation)
      if (m > 12) m = 12;  // Within boundaries
      int l = isleap(y);
      while ( month2doy[l][m] > d) m--; // Fix overestimation
      month[i] = m;
      day[i] = d - month2doy[l][m] + 1;
    }

//    printf ("  -> %4d-%02d-%02d", y, month[i], day[i]);
//    printf ("\n");

  }

//  printf ("\n");
  return 0;
}
#undef isleap


/*

  365-day calendar functions

*/

// Convert an absolute date to a relative date (seconds since start date)
int date_as_val_365 (int n, int iyear, int imonth, int iday,
                     int ihour, int iminute, int isecond,
                     int *year, int *month, int *day,
                     int *hour, int *minute, int *second,
                     long long int *val) {

  static const int month2doy[] = {0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

  check_month(imonth);
  long long int ref = ( month2doy[imonth] + iday - 1) * 86400 + ihour*3600 + iminute*60 + isecond;

  // Take the difference between the date array and the start date
  for (int i = 0; i < n; i++) {

    int y = year[i];

    // Wrap month
    int m = month[i];
    int dy = (m < 1) ? (m-12)/12 : (m-1)/12;
    y += dy;
    m -= dy*12;

    long long int v = ( 365*(y-iyear) + month2doy[m] + day[i] - 1 ) * 86400LL;
    v += hour[i]*3600 + minute[i]*60 + second[i];
    v -= ref;
    val[i] = v;
  }

  return 0;
}

//TODO: adjust ref to use iday-1 instead of iday (to be consistent with std calendar cord)

// Convert a relative date (seconds since start date) to an absolute date
int val_as_date_365 (int n, int iyear, int imonth, int iday,
                     int ihour, int iminute, int isecond,
                     long long int *val,
                     int *year, int *month, int *day,
                     int *hour, int *minute, int *second) {

  static const int month2doy[] = {0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

  check_month(imonth);
  long long int ref = ( month2doy[imonth] + iday ) * 86400 + ihour*3600 + iminute*60 + isecond;

  for (int i = 0; i < n; i++) {
    int y = iyear;
    long long int d;
    {
      long long int s = val[i] + ref;  // Seconds since the beginning of the year of start date
      d = s / 86400;  // seconds to days
      s -= d * 86400;
      if (s < 0) { s += 86400; d--; }  // handle negative reference values
      hour[i] = s / 3600;
      s %= 3600;
      minute[i] = s / 60;
      second[i] = s % 60;
    }
    // Get year / day-of-year
    if (d <= 0) {
      int ny = (365-d)/365;
      d += 365*ny;
      y -= ny;
    }
    if (d > 365) {
      int ny = (d-1)/365;
      d -= 365*ny;
      y += ny;
    }
    year[i] = y;
    // Get the month / day
    {
      int m = d / 30 + 1;  // Estimate (allow overestimation)
      if (m > 12) m = 12;  // Within boundaries
      while (month2doy[m] >= d) m--; // Fix overestimation
      month[i] = m;
      day[i] = d - month2doy[m];
    }
  }

  return 0;
}


/*

  360-day calendar functions

*/

// Convert an absolute date to a relative date (seconds since start date)
int date_as_val_360 (int n, int iyear, int imonth, int iday,
                     int ihour, int iminute, int isecond,
                     int *year, int *month, int *day,
                     int *hour, int *minute, int *second,
                     long long int *val) {

  long long int ref = ( (imonth-1)*30 + iday ) * 86400 + ihour*3600 + iminute*60 + isecond;

  // Take the difference between the date array and the start date
  for (int i = 0; i < n; i++) {

    int y = year[i];
    long long int v = ( 360*(y-iyear) + (month[i]-1)*30 + day[i] ) * 86400LL;
    v += hour[i]*3600 + minute[i]*60 + second[i];
    v -= ref;
    val[i] = v;
  }

  return 0;
}

// Convert a relative date (seconds since start date) to an absolute date
int val_as_date_360 (int n, int iyear, int imonth, int iday,
                     int ihour, int iminute, int isecond,
                     long long int *val,
                     int *year, int *month, int *day,
                     int *hour, int *minute, int *second) {

  long long int ref = ( (imonth-1)*30 + iday ) * 86400 + ihour*3600 + iminute*60 + isecond;

  for (int i = 0; i < n; i++) {
    int y = iyear;
    long long int d;
    {
      long long int s = val[i] + ref;  // Seconds since the beginning of the year of start date
      d = s / 86400;  // seconds to days
      s -= d * 86400;
      if (s < 0) { s += 86400; d--; }  // handle negative reference values
      hour[i] = s / 3600;
      s %= 3600;
      minute[i] = s / 60;
      second[i] = s % 60;
    }
    // Get year / day-of-year
    if (d <= 0) {
      int ny = (360-d)/360;
      d += 360*ny;
      y -= ny;
    }
    if (d > 360) {
      int ny = (d-1)/360;
      d -= 360*ny;
      y += ny;
    }
    year[i] = y;
    // Get the month / day
    {
      int m = (d-1) / 30 + 1;
      month[i] = m;
      day[i] = d - 30*(m-1);
    }
  }

  return 0;
}


/*

  'yearless' calendar functions

*/


// Convert an absolute date to a relative date (seconds since start date)
int date_as_val_yearless (int n, int iyear, int imonth, int iday,
                          int ihour, int iminute, int isecond,
                          int *year, int *month, int *day,
                          int *hour, int *minute, int *second,
                          long long int *val) {

  assert (iyear == 0);
  assert (imonth == 1);
  for (int i = 0; i < n; i++) {
//    assert (year[i] == 0);
    if (year[i] != 0) printf ("nonzero year at position %5d: %d\n", i, year[i]);
//    assert (month[i] == 1);
    if (month[i] != 1) {
      printf ("non-1 month at position %5d: %d\n", i, month[i]);
      return -1;
    }
  }

  for (int i = 0; i < n; i++) {
    val[i] = (day[i] - iday) * 86400LL + (hour[i] - ihour) * 3600
           + (minute[i]-iminute) * 60 + (second[i] - isecond);
  }

  return 0;

}

// Convert a relative date (seconds since start date) to an absolute date
int val_as_date_yearless (int n, int iyear, int imonth, int iday,
                          int ihour, int iminute, int isecond,
                          long long int *val,
                          int *year, int *month, int *day,
                          int *hour, int *minute, int *second) {

  assert (iyear == 0);
  assert (imonth == 1);

  long long int ref = iday * 86400LL + ihour*3600 + iminute*60 + isecond;

  for (int i = 0; i < n; i++) {
    long long int x = val[i] + ref;
    day[i] = (int) floor(x / 86400.);
    x -= day[i] * 86400LL;
    second[i] = x % 60; x /= 60;
    minute[i] = x % 60; x /= 60;
    hour[i] = x % 24; x /= 24;
    month[i] = 1;
    year[i] = 0;
  }

  return 0;
}


/*****************   Python wrappers   ****************/

// get_indices
static PyObject *timeaxiscore_get_indices (PyObject *self, PyObject *args) {
  int natts, *invalues, n_in, *outvalues, n_out, *indices;
  PyObject *invalues_obj, *outvalues_obj;
  PyArrayObject *invalues_array, *outvalues_array, *indices_array;
  int ret;
  if (!PyArg_ParseTuple(args, "iOiOiO!", &natts, &invalues_obj, &n_in, &outvalues_obj, &n_out, &PyArray_Type, &indices_array)) return NULL;
  // Make sure input arrays are contiguous and of the right type
  invalues_array = (PyArrayObject*)PyArray_ContiguousFromObject(invalues_obj,NPY_INT32,0,0);
  if (invalues_array == NULL) return NULL;
  outvalues_array = (PyArrayObject*)PyArray_ContiguousFromObject(outvalues_obj,NPY_INT32,0,0);
  if (outvalues_array == NULL) return NULL;
  if (indices_array->descr->type_num != NPY_INT32) return NULL;
  if (!PyArray_ISCONTIGUOUS(indices_array)) return NULL;
  invalues = (int*)(invalues_array->data);
  outvalues = (int*)(outvalues_array->data);
  indices = (int*)(indices_array->data);
  ret = get_indices (natts, invalues, n_in, outvalues, n_out, indices);
  // Free temporary references
  Py_DECREF(invalues_array);
  Py_DECREF(outvalues_array);
  return Py_BuildValue("i", ret);
}

// uniquify
static PyObject *timeaxiscore_uniquify (PyObject *self, PyObject *args) {

  int natts, *in_atts, n_in, *out_atts, n_out;
  PyObject *in_atts_obj;
  PyArrayObject *in_atts_array, *out_atts_array;
  int ret;
  if (!PyArg_ParseTuple(args, "iOiO!", &natts, &in_atts_obj, &n_in, &PyArray_Type, &out_atts_array)) return NULL;
  // Make sure input array is contiguous and of the right type
  in_atts_array = (PyArrayObject*)PyArray_ContiguousFromObject(in_atts_obj,NPY_INT32,0,0);
  if (in_atts_array == NULL) return NULL;
  // Make sure the output arrays are contiguous and of the right type
  if (out_atts_array->descr->type_num != NPY_INT32) return NULL;
  if (!PyArray_ISCONTIGUOUS(out_atts_array)) return NULL;

  in_atts = (int*)(in_atts_array->data);
  out_atts = (int*)(out_atts_array->data);
  ret = uniquify (natts, in_atts, n_in, out_atts, &n_out);

  // Free temporary references
  Py_DECREF(in_atts_array);

  if (ret != 0) return NULL;
  return Py_BuildValue("i", n_out);
}

// common_map
static PyObject *timeaxiscore_common_map (PyObject *self, PyObject *args) {
  int natts, na, *a, nb, *b, nmap, *a_map, *b_map;

  PyObject *a_obj, *b_obj;
  PyArrayObject *a_array, *b_array, *a_map_array, *b_map_array;
  int ret;
  if (!PyArg_ParseTuple(args, "iiOiOO!O!", &natts, &na, &a_obj, &nb, &b_obj, &PyArray_Type, &a_map_array, &PyArray_Type, &b_map_array)) return NULL;

  // Make sure input arrays are contiguous and of the right type
  a_array = (PyArrayObject*)PyArray_ContiguousFromObject(a_obj,NPY_INT32,0,0);
  if (a_array == NULL) return NULL;
  b_array = (PyArrayObject*)PyArray_ContiguousFromObject(b_obj,NPY_INT32,0,0);
  if (b_array == NULL) return NULL;

  // Make sure the output arrays are contiguous and of the right type
  if (a_map_array->descr->type_num != NPY_INT32) return NULL;
  if (!PyArray_ISCONTIGUOUS(a_map_array)) return NULL;
  if (b_map_array->descr->type_num != NPY_INT32) return NULL;
  if (!PyArray_ISCONTIGUOUS(b_map_array)) return NULL;

  a = (int*)(a_array->data);
  b = (int*)(b_array->data);
  a_map = (int*)(a_map_array->data);
  b_map = (int*)(b_map_array->data);

  ret = common_map (natts, na, a, nb, b, &nmap, a_map, b_map);

  // Free temporary references
  Py_DECREF(a_array);
  Py_DECREF(b_array);

  if (ret != 0) return NULL;

  return Py_BuildValue("i", nmap);
}

typedef int (val_as_date_func) (int n, int iyear, int imonth, int iday,
                     int ihour, int iminute, int isecond,
                     long long int *val,
                     int *year, int *month, int *day,
                     int *hour, int *minute, int *second);

int checkArray(PyArrayObject *a, const char *name, int np_type)
{
  if (a->descr->type_num != np_type) { 
    char err[100];
    sprintf(err, "%s array type wrong.", name);
    PyErr_SetString(PyExc_TypeError, err); 
    return 0; 
  }
  if (!PyArray_ISCONTIGUOUS(a)) {
    char err[100];
    sprintf(err, "%s array not contiguous.", name);
    PyErr_SetString(PyExc_TypeError, err);
    return 0; 
  }
  return 1;
}

static PyObject *val_as_date_wrapper (PyObject *args, val_as_date_func *f) {

  int n, iyear, imonth, iday, ihour, iminute, isecond, *year, *month, *day, *hour, *minute, *second;
  long long int *val;
  PyArrayObject *val_array, *year_array, *month_array, *day_array, *hour_array, *minute_array, *second_array;
  int ret;
  if (!PyArg_ParseTuple(args, "iiiiiiiO!O!O!O!O!O!O!", &n, &iyear, &imonth, &iday, &ihour, &iminute, &isecond, &PyArray_Type, &val_array, &PyArray_Type, &year_array, &PyArray_Type, &month_array, &PyArray_Type, &day_array, &PyArray_Type, &hour_array, &PyArray_Type, &minute_array, &PyArray_Type, &second_array)) return NULL;

  // Make sure the arrays are contiguous and of the right type
  if (!checkArray(val_array, "Val", NPY_INT64)) return NULL;
  if (!checkArray(year_array, "Year", NPY_INT32)) return NULL;
  if (!checkArray(month_array, "Month", NPY_INT32)) return NULL;
  if (!checkArray(day_array, "Day", NPY_INT32)) return NULL;
  if (!checkArray(hour_array, "Hour", NPY_INT32)) return NULL;
  if (!checkArray(minute_array, "Minute", NPY_INT32)) return NULL;
  if (!checkArray(second_array, "Second", NPY_INT32)) return NULL;

  val = (long long int*)(val_array->data);
  year   = (int*)(year_array->data);
  month  = (int*)(month_array->data);
  day    = (int*)(day_array->data);
  hour   = (int*)(hour_array->data);
  minute = (int*)(minute_array->data);
  second = (int*)(second_array->data);

  ret = f (n, iyear, imonth, iday, ihour, iminute, isecond, val, year, month, day, hour, minute, second);
  return Py_BuildValue("i", ret);

}

//val_as_date_std
static PyObject *timeaxiscore_val_as_date_std (PyObject *self, PyObject *args) {
  return val_as_date_wrapper (args, val_as_date_std);
}

//val_as_date_365
static PyObject *timeaxiscore_val_as_date_365 (PyObject *self, PyObject *args) {
  return val_as_date_wrapper (args, val_as_date_365);
}

//val_as_date_360
static PyObject *timeaxiscore_val_as_date_360 (PyObject *self, PyObject *args) {
  return val_as_date_wrapper (args, val_as_date_360);
}

//val_as_date_yearless
static PyObject *timeaxiscore_val_as_date_yearless (PyObject *self, PyObject *args) {
  return val_as_date_wrapper (args, val_as_date_yearless);
}


typedef int (date_as_val_func) (int n, int iyear, int imonth, int iday,
                     int ihour, int iminute, int isecond,
                     int *year, int *month, int *day,
                     int *hour, int *minute, int *second,
                     long long int *val);


static PyObject *date_as_val_wrapper (PyObject *args, date_as_val_func *f) {

  int n, iyear, imonth, iday, ihour, iminute, isecond, *year, *month, *day, *hour, *minute, *second;
  long long int *val;
  PyArrayObject *val_array, *year_array, *month_array, *day_array, *hour_array, *minute_array, *second_array;
  int ret;
  if (!PyArg_ParseTuple(args, "iiiiiiiO!O!O!O!O!O!O!", &n, &iyear, &imonth, &iday, &ihour, &iminute, &isecond, &PyArray_Type, &year_array, &PyArray_Type, &month_array, &PyArray_Type, &day_array, &PyArray_Type, &hour_array, &PyArray_Type, &minute_array, &PyArray_Type, &second_array, &PyArray_Type, &val_array)) return NULL;

  // Make sure the arrays are contiguous and of the right type
  if (!checkArray(val_array, "Val", NPY_INT64)) return NULL;
  if (!checkArray(year_array, "Year", NPY_INT32)) return NULL;
  if (!checkArray(month_array, "Month", NPY_INT32)) return NULL;
  if (!checkArray(day_array, "Day", NPY_INT32)) return NULL;
  if (!checkArray(hour_array, "Hour", NPY_INT32)) return NULL;
  if (!checkArray(minute_array, "Minute", NPY_INT32)) return NULL;
  if (!checkArray(second_array, "Second", NPY_INT32)) return NULL;

  val = (long long int*)(val_array->data);
  year   = (int*)(year_array->data);
  month  = (int*)(month_array->data);
  day    = (int*)(day_array->data);
  hour   = (int*)(hour_array->data);
  minute = (int*)(minute_array->data);
  second = (int*)(second_array->data);

  ret = f (n, iyear, imonth, iday, ihour, iminute, isecond, year, month, day, hour, minute, second, val);
  // Check if there was an internal problem
  if (PyErr_Occurred()) return NULL;

  return Py_BuildValue("i", ret);

}

//date_as_val_std
static PyObject *timeaxiscore_date_as_val_std (PyObject *self, PyObject *args) {
  return date_as_val_wrapper (args, date_as_val_std);
}

//date_as_val_365
static PyObject *timeaxiscore_date_as_val_365 (PyObject *self, PyObject *args) {
  return date_as_val_wrapper (args, date_as_val_365);
}

//date_as_val_360
static PyObject *timeaxiscore_date_as_val_360 (PyObject *self, PyObject *args) {
  return date_as_val_wrapper (args, date_as_val_360);
}

//date_as_val_yearless
static PyObject *timeaxiscore_date_as_val_yearless (PyObject *self, PyObject *args) {
  return date_as_val_wrapper (args, date_as_val_yearless);
}



static PyMethodDef TimeaxisMethods[] = {
  {"get_indices", timeaxiscore_get_indices, METH_VARARGS, "Find indices mapping the input times to the output times"},
  {"uniquify", timeaxiscore_uniquify, METH_VARARGS, "Gives unique elements of a time axis"},
  {"common_map", timeaxiscore_common_map, METH_VARARGS, "Common map between two time axes"},
  {"val_as_date_std", timeaxiscore_val_as_date_std, METH_VARARGS, ""},
  {"val_as_date_365", timeaxiscore_val_as_date_365, METH_VARARGS, ""},
  {"val_as_date_360", timeaxiscore_val_as_date_360, METH_VARARGS, ""},
  {"val_as_date_yearless", timeaxiscore_val_as_date_yearless, METH_VARARGS, ""},
  {"date_as_val_std", timeaxiscore_date_as_val_std, METH_VARARGS, ""},
  {"date_as_val_365", timeaxiscore_date_as_val_365, METH_VARARGS, ""},
  {"date_as_val_360", timeaxiscore_date_as_val_360, METH_VARARGS, ""},
  {"date_as_val_yearless", timeaxiscore_date_as_val_yearless, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "timeaxiscore",      /* m_name */
        NULL,                /* m_doc */
        -1,                  /* m_size */
        TimeaxisMethods,     /* m_methods */
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
    m = Py_InitModule("timeaxiscore", TimeaxisMethods);
#endif

    import_array();

    return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    inittimeaxiscore(void)
    {
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_timeaxiscore(void)
    {
        return moduleinit();
    }
#endif

