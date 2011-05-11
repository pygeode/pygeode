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
