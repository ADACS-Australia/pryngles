#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Maximum number of tabulated points dderiv() and get_spline_derivs() can
// handle (size of their scratch arrays), and maximum number of Stokes
// elements reflection() supports (size of Bplus).
#define MAX_SPLINE_N 1000
#define MAX_NSTOKES 4
#define MAX_NMAT (MAX_NSTOKES - 1)

#define REFLECTION_TOLERANCE 1.0e-6

/*
 *----------------------------------------------------------------------------
 *     Spline interpolation routine from Press et al. (1986, p.88).
 *
 *     Given arrays x and y of length n containing a tabulated function,
 *     i.e. y(i)=f(x(i)), with x(1)<x(2)<...<x(n), and given values yp1
 *     and ypn for the first derivative of the interpolating function at
 *     points 1 and n respectively, this routine returns an array ddy of
 *     length n which contains the second derivatives of the interpola-
 *     ting function at the tabulated points x(i).
 *
 *     If yp1 and/or yp2 are equal to 1x10^30 or larger, the routine is
 *     signalled to set the corresponding boundary condition for a natu-
 *     ral spline, with zero second derivative on that boundary.
 *
 *     n is the number of elements in x and y
 *
 *     Returns 0 on success, -1 if n < 2 or n > MAX_SPLINE_N.
 *----------------------------------------------------------------------------
 */
int dderiv(const double x[], const double y[], int n, double ddy[]) {
  double u[MAX_SPLINE_N];
  double sig, p;

  if (n < 2 || n > MAX_SPLINE_N) {
    fprintf(stderr, "Error in spline: n=%d outside [2, %d].\n", n,
            MAX_SPLINE_N);
    return -1;
  }

  ddy[0] = 0;
  u[0] = 0;

  for (int i = 1; i < n - 1; i++) {
    sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    p = sig * ddy[i - 1] + 2;
    ddy[i] = (sig - 1) / p;
    u[i] = (6 *
                ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
                 (y[i] - y[i - 1]) / (x[i] - x[i - 1])) /
                (x[i + 1] - x[i - 1]) -
            sig * u[i - 1]) /
           p;
  }

  double qn = 0;
  double un = 0;
  ddy[n - 1] = (un - qn * u[n - 2]) / (qn * ddy[n - 2] + 1);

  for (int k = n - 2; k >= 0; k--) {
    ddy[k] = ddy[k] * ddy[k + 1] + u[k];
  }

  return 0;
}

/*
 * Precompute the spline second-derivative tables used by reflection().
 * They depend only on xmu and fou, so call this once after reading the
 * Fourier file. All arrays are flat, shape (nfou, nmat, nmugs, nmugs);
 * the three outputs are allocated by the caller.
 *
 *   ddf_j     d2f/dj2        (along the last axis)
 *   ddf_i     d2f/di2        (along the second-to-last axis)
 *   ddddf_ij  d2(ddf_i)/dj2
 *
 * Returns 0 on success, -1 on invalid sizes.
 */
int get_spline_derivs(int nmat, int nmugs, int nfou, const double *xmu,
                      const double *fou_flat, double *ddf_j_flat,
                      double *ddf_i_flat, double *ddddf_ij_flat) {
  if (nmat < 1 || nmat > MAX_NMAT) {
    fprintf(stderr, "Error in get_spline_derivs: nmat=%d outside [1, %d].\n",
            nmat, MAX_NMAT);
    return -1;
  }
  if (nmugs < 2 || nmugs > MAX_SPLINE_N) {
    fprintf(stderr, "Error in get_spline_derivs: nmugs=%d outside [2, %d].\n",
            nmugs, MAX_SPLINE_N);
    return -1;
  }
  if (nfou < 1) {
    fprintf(stderr, "Error in get_spline_derivs: nfou=%d is < 1.\n", nfou);
    return -1;
  }

  const double (*fou)[nmat][nmugs][nmugs] =
      (const double (*)[nmat][nmugs][nmugs])fou_flat;
  double (*ddf_j)[nmat][nmugs][nmugs] =
      (double (*)[nmat][nmugs][nmugs])ddf_j_flat;
  double (*ddf_i)[nmat][nmugs][nmugs] =
      (double (*)[nmat][nmugs][nmugs])ddf_i_flat;
  double (*ddddf_ij)[nmat][nmugs][nmugs] =
      (double (*)[nmat][nmugs][nmugs])ddddf_ij_flat;

  double f_col[MAX_SPLINE_N], ddf_col[MAX_SPLINE_N];

  // nmugs was validated above, so dderiv() cannot fail here.
  for (int m = 0; m < nfou; m++) {
    for (int k = 0; k < nmat; k++) {
      // d2f/dj2 along each (contiguous) row
      for (int i = 0; i < nmugs; i++)
        dderiv(xmu, fou[m][k][i], nmugs, ddf_j[m][k][i]);

      // d2f/di2 along each column
      for (int j = 0; j < nmugs; j++) {
        for (int i = 0; i < nmugs; i++)
          f_col[i] = fou[m][k][i][j];
        dderiv(xmu, f_col, nmugs, ddf_col);
        for (int i = 0; i < nmugs; i++)
          ddf_i[m][k][i][j] = ddf_col[i];
      }

      // d2(d2f/di2)/dj2
      for (int i = 0; i < nmugs; i++)
        dderiv(xmu, ddf_i[m][k][i], nmugs, ddddf_ij[m][k][i]);
    }
  }

  return 0;
}
