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
