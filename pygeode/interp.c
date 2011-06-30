// Wrapper for GSL functions

#include <gsl/gsl_interp.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

/*
  Interpolate a bunch of arrays
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

*/
int interpgsl (int narrays, int nxin, int nxout,
               double *xin, double *yin, double *xout, double *yout,
               int loop_xin, int loop_xout,
               const gsl_interp_type *type)                          {

  int flip = 0; double *local_xin; double *local_xout;

  // Check the order of the x coordinates
  // If the order seems to be reversed, then flip the sign of the coordinates
  if (nxin > 1) if (xin[0] > xin[1]) {
    flip = 1;
    int N_xin = (loop_xin ? narrays : 1) * nxin;
    int N_xout = (loop_xout ? narrays : 1) * nxout;
    local_xin = malloc(N_xin * sizeof(double));
    local_xout = malloc(N_xout * sizeof(double));
    for (int i = 0; i < N_xin; i++) local_xin[i] = xin[i] * -1;
    for (int i = 0; i < N_xout; i++) local_xout[i] = xout[i] * -1;
    xin = local_xin;
    xout = local_xout;
  }


  gsl_interp *interp = gsl_interp_alloc (type, nxin);
  gsl_interp_accel *acc = gsl_interp_accel_alloc();

  if (loop_xin == 0) gsl_interp_init (interp, xin, yin, nxin);

  for (int n = 0; n < narrays; n++) {
    if (loop_xin == 1) gsl_interp_init (interp, xin, yin, nxin);

    for (int i = 0; i < nxout; i++) {
      gsl_interp_eval_e (interp, xin, yin, *(xout++), acc, yout++);
    }

    if (loop_xin == 1) {
      xin += nxin;
      gsl_interp_accel_reset (acc);
    }
    if (loop_xout == 0) xout -= nxout;
    yin += nxin;
  }

  gsl_interp_accel_free(acc);
  gsl_interp_free (interp);

  // Clean up the flipped coordinate arrays?
  if (flip == 1) {
    free (local_xin);
    free (local_xout);
  }

  return 0;

}

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

*/
int interpgsl_nan (int narrays, int nxin, int nxout,
               double *xin, double *yin, double *xout, double *yout,
               int loop_xin, int loop_xout,
               double d_below, double d_above,
               const gsl_interp_type *type)                          {

  int i, j, n, start, end, step; 
  double *local_xin, *local_xout, *local_yin;
  double xmin, ymin, xmax, ymax;

  // Check the order of the x coordinates
  // If the order seems to be reversed, then flip the sign of the coordinates
  if (nxin > 1 && xin[0] > xin[1]) { 
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

  gsl_interp *interp;
  gsl_interp_accel *acc = gsl_interp_accel_alloc();

  for (n = 0; n < narrays; n++) {
    // Search for nans, resort if necessary
    for (i = start, j = 0; i != end; i += step) {
      if isfinite(yin[i]) {
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

    interp = gsl_interp_alloc (type, j);
    gsl_interp_init (interp, local_xin, local_yin, j);

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

}
