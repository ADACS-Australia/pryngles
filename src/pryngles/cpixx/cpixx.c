//////////////////////////////////////////////////////////////
// DEPENDENCIES
//////////////////////////////////////////////////////////////
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//////////////////////////////////////////////////////////////
// MACROS
//////////////////////////////////////////////////////////////
#define MAX_FOU 1000
#define MAX_STRING 1000
#define VERBOSITY 0

// Maximum number of tabulated points spline() can handle (size of its
// scratch array), and maximum number of Stokes elements reflection() supports
// (size of Bplus).
#define MAX_SPLINE_N 1000
#define MAX_NMAT 4

//////////////////////////////////////////////////////////////
// ROUTINES
//////////////////////////////////////////////////////////////
struct FourierCoefficients {
  int nmat, nmugs, nfou;
  double *xmu;
  double ***rfou;
  double ***rtra;
};

/*
 * Checked calloc wrapper for the zeros_* allocators.
 *
 * Takes an int count (the dimension type used throughout this file) and
 * converts it to size_t in one place, which keeps -Wsign-conversion quiet.
 * Non-positive counts are clamped to 1 so a negative value can never wrap
 * into a huge allocation request. Aborts on allocation failure rather than
 * returning NULL, since callers write to the result immediately.
 *
 * Memory is zero-initialised by calloc, so callers need no fill loop.
 */
static void *xcalloc(int n, size_t size) {
  if (n <= 0)
    n = 1; /* avoid calloc(0) ambiguity */
  void *p = calloc((size_t)n, size);
  if (!p) {
    fprintf(stderr, "calloc failed\n");
    abort();
  }
  return p;
}

double *zeros_vector(int n) {
  return (double *)xcalloc(n, sizeof(double)); /* calloc already zeroes */
}

double **zeros_matrix(int n, int m) {
  double **M = (double **)xcalloc(n, sizeof(double *));
  for (int i = 0; i < n; i++)
    M[i] = zeros_vector(m);
  return M;
}

double ***zeros_cube(int n, int m, int p) {
  double ***C = (double ***)xcalloc(n, sizeof(double **));
  for (int i = 0; i < n; i++)
    C[i] = zeros_matrix(m, p);
  return C;
}

// Free a matrix allocated with zeros_matrix(n, m)
void free_matrix(double **M, int n) {
  if (!M)
    return;
  for (int i = 0; i < n; i++)
    free(M[i]);
  free(M);
}

// Free a cube allocated with zeros_cube(n, m, p)
void free_cube(double ***C, int n, int m) {
  if (!C)
    return;
  for (int i = 0; i < n; i++)
    free_matrix(C[i], m);
  free(C);
}

/*
 *----------------------------------------------------------------------------
 *     Spline interpolation routine from Press et al. (1986, p.88).
 *
 *     Given arrays x and y of length n containing a tabulated function,
 *     i.e. y(i)=f(x(i)), with x(1)<x(2)<...<x(n), and given values yp1
 *     and ypn for the first derivative of the interpolating function at
 *     points 1 and n respectively, this routine returns an array y2 of
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
int spline(double x[], double y[], int n, double y2[]) {
  int i, k;
  double u[MAX_SPLINE_N];
  double sig, p, qn, un;

  if (n < 2 || n > MAX_SPLINE_N) {
    fprintf(stderr, "Error in spline: n=%d outside [2, %d].\n", n,
            MAX_SPLINE_N);
    return -1;
  }

  y2[0] = 0;
  u[0] = 0;

  for (i = 1; i < n - 1; i++) {
    sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    p = sig * y2[i - 1] + 2;
    y2[i] = (sig - 1) / p;
    u[i] = (6 *
                ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
                 (y[i] - y[i - 1]) / (x[i] - x[i - 1])) /
                (x[i + 1] - x[i - 1]) -
            sig * u[i - 1]) /
           p;
  }

  qn = 0;
  un = 0;
  y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1);

  for (k = n - 2; k >= 0; k--) {
    y2[k] = y2[k] * y2[k + 1] + u[k];
  }

  return 0;
}

int bisect(double xa[], int n, double x) {
  int k;
  int klo = 0;
  int khi = n - 1;

  while ((khi - klo) > 1) {
    k = (klo + khi) / 2; // No need to floor() here, since it's integer division
    if (xa[k] > x)
      khi = k;
    else
      klo = k;
  }

  return klo;
}

/*
 * Find the bracketing indices and interpolation coefficients for x in xa.
 *
 * Returns 0 on success, -1 if n < 2 or the bracketing interval is
 * degenerate (xa not strictly increasing). On failure the outputs are
 * left untouched.
 */
int spline_coefficients(double xa[], int n, double x, int *klo_out,
                        int *khi_out, double *a_out, double *b_out,
                        double *h_out) {
  if (n < 2) {
    fprintf(stderr, "Error in spline_coefficients: n=%d < 2.\n", n);
    return -1;
  }

  // Find indices to interpolate between
  int klo = bisect(xa, n, x);
  int khi = klo + 1;

  double h = xa[khi] - xa[klo];

  if (fabs(h) < 1e-10) {
    fprintf(stderr, "Error in spline_coefficients: bad xa input.\n");
    return -1;
  }

  // Coefficients
  *a_out = (xa[khi] - x) / h;
  *b_out = (x - xa[klo]) / h;
  *h_out = h;

  // Indices
  *klo_out = klo;
  *khi_out = khi;

  return 0;
}

/*
 *----------------------------------------------------------------------------
 *     Spline interpolation routine from Press et al. (1986, p.88).
 *
 *     Given the arrays xa and ya of length n, which tabulate a function
 *     (with the xa(i)'s in order), and given the array y2a, which is
 *     the output from SPLINE above, and given a value of x, this
 *     routine returns a cubic-spline interpolated value y.
 *
 *     NOTE: for computational efficiency, we pre-compute lower and upper
 *     indices as well as the spline coefficients (above), since they don't
 *     change for an individual pixel/spangle.
 *----------------------------------------------------------------------------
 */
double splint(double ya[], double y2a[], int klo, int khi, double a, double b,
              double h) {
  double y =
      a * ya[klo] + b * ya[khi] +
      ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6;
  return y;
}

//////////////////////////////////////////////////////////////
// PHYSICAL ROUTINES
//////////////////////////////////////////////////////////////
/*
 *----------------------------------------------------------------------------
 *     Read a Fourier coefficients file and calculate the Stokes vector
 *     for a given geometry.
 *
 *     It is assumed that the Stokes vector of the incoming sunlight
 *     is [1,0,0,0], with the flux measured perpendicular to the direction
 *     of incidence equal to pi.
 *
 *     phi and beta are assumed to be in radian
 *     theta0 and theta are given as cos(theta0) and cos(theta) respectively
 *
 *     Returns 0 on success, -1 on invalid sizes (nmat outside [1, MAX_NMAT],
 *     nmugs outside [2, MAX_SPLINE_N]) or a degenerate xmu grid. On failure
 *     Sarr is not (fully) written.
 *
 *     Sarr must have npix rows of at least nmat + 1 elements.
 *
 *     Author: Daphne M. Stam
 *     Date: September 2022
 *----------------------------------------------------------------------------
 */
int reflection(struct FourierCoefficients F, int qreflection, int npix,
               double *phi, double *beta, double *theta0, double *theta,
               double *apix, double **Sarr) {
  // Declarations
  int i, j, k, m, n;
  double be, SvR2, SvR3, P;
  double mu, mu0, muold = 1, mu0old = 1;
  int ki;
  double fac;
  int ret = 0;

  // Read sizes
  int nmat = F.nmat;
  int nmugs = F.nmugs;
  int nfou = F.nfou;

  // Validate sizes before allocating anything: Bplus has MAX_NMAT elements
  // and spline() has a fixed-size scratch array.
  if (nmat < 1 || nmat > MAX_NMAT) {
    fprintf(stderr, "Error in reflection: nmat=%d outside [1, %d].\n", nmat,
            MAX_NMAT);
    return -1;
  }
  if (nmugs < 2 || nmugs > MAX_SPLINE_N) {
    fprintf(stderr, "Error in reflection: nmugs=%d outside [2, %d].\n", nmugs,
            MAX_SPLINE_N);
    return -1;
  }

  double ***rf, ***rfsec;
  double **RM, **rfmu0;
  double *rfsecmu0;
  double *SvR, *rf3save;
  double Bplus[MAX_NMAT];

  // Allocate dynamically temporal matrices
  // Cubes
  rf = zeros_cube(nmat, nmugs, nmugs);
  rfsec = zeros_cube(nmat, nmugs, nmugs);
  // Matrices
  rfmu0 = zeros_matrix(nmat, nmugs);
  RM = zeros_matrix(npix, nmat);
  // Vectos nmugs
  rfsecmu0 = zeros_vector(nmugs);
  // Vectos nmats
  rf3save = zeros_vector(nmat);
  SvR = zeros_vector(nmat);

  // Loop over the Fourier coefficients:
  for (m = 0; m < nfou; m++) {

    fac = 1.0;
    if (m == 0)
      fac = 0.5;

    // Initialize the interpolation matrix for the current fourier coefficient:
    for (j = 0; j < nmugs; j++) {

      for (k = 0; k < nmat; k++) {
        ki = j * nmat + k;

        for (n = 0; n < nmugs; n++) {
          rf[k][j][n] = qreflection ? F.rfou[ki][n][m] : F.rtra[ki][n][m];
        }

        // Use slice rf(k,j,:), write directly into corresponding rfsec row.
        // nmugs was validated above, so spline() cannot fail here.
        spline(F.xmu, rf[k][j], nmugs, rfsec[k][j]);

      } // End k

    } // End j

    /*
     *----------------------------------------------------------------------------
     *     Loop over the pixels:
     *       If the input angles are (very) similar to a previously calculated
     * case use those values. To obtain obtain the fourier coefficient at
     * (mu,mu0) spline has to be called a second time.
     *----------------------------------------------------------------------------
     */
    for (i = 0; i < npix; i++) {
      mu = theta[i];
      mu0 = theta0[i];
      Bplus[0] = cos(m * phi[i]);
      Bplus[1] = cos(m * phi[i]);
      Bplus[2] = sin(m * phi[i]);
      Bplus[3] = sin(m * phi[i]);

      if ((i > 0) && ((fabs(mu - muold) < 1e-6)) &&
          (fabs(mu0 - mu0old) < 1e-6)) {

        for (k = 0; k < nmat; k++)
          RM[i][k] = RM[i][k] + 2 * Bplus[k] * fac * rf3save[k];

      } else {

        // Get indices and coefficients to work with (at mu0)
        int klo, khi;
        double a, b, h;
        if (spline_coefficients(F.xmu, nmugs, mu0, &klo, &khi, &a, &b, &h)) {
          ret = -1;
          goto cleanup;
        }

        for (j = 0; j < nmugs; j++) {
          for (k = 0; k < nmat; k++) {
            // Use rf(k,j,:) and rfsec(k,j,:) slices directly, and write
            // straight into rmu0(k,j,:) Do the spline interpolation between klo
            // and khi, given the coefficients a,b,h
            rfmu0[k][j] = splint(rf[k][j], rfsec[k][j], klo, khi, a, b, h);
          }
        }

        // Get indices and coefficients to work with (now at mu)
        if (spline_coefficients(F.xmu, nmugs, mu, &klo, &khi, &a, &b, &h)) {
          ret = -1;
          goto cleanup;
        }

        for (k = 0; k < nmat; k++) {
          spline(F.xmu, rfmu0[k], nmugs, rfsecmu0);
          // Do the spline interpolation between klo and khi, given the
          // coefficients a,b,h
          rf3save[k] = splint(rfmu0[k], rfsecmu0, klo, khi, a, b, h);
          muold = mu;
          mu0old = mu0;
          RM[i][k] = RM[i][k] + 2 * Bplus[k] * fac * rf3save[k];
        }
      } // End else
    } // End i (pix)
  } // End loop fourier coefficients

  // Loop again over the pixels to rotate Stokes vector:
  for (i = 0; i < npix; i++) {
    mu = theta[i];
    mu0 = theta0[i];

    // Calculate the locally reflected Stokes vector:
    for (k = 0; k < nmat; k++)
      SvR[k] = mu0 * RM[i][k];

    // Rotate Stokes elements Q and U to the actual reference plane:
    be = 2 * beta[i];
    SvR2 = cos(be) * SvR[1] + sin(be) * SvR[2];
    SvR3 = -sin(be) * SvR[1] + cos(be) * SvR[2];
    SvR[1] = SvR2;
    SvR[2] = SvR3;

    // Compute the local degree of polarisation P:
    if (fabs(SvR[0]) < 1e-6)
      P = 0;
    else if (fabs(SvR[2]) < 1e-6)
      P = -SvR[1] / SvR[0];
    else
      P = sqrt(SvR[1] * SvR[1] + SvR[2] * SvR[2]) / SvR[0];
    if (fabs(P) < 1e-6)
      P = 0;

    /*
     *----------------------------------------------------------------------------
     *       Add the Stokes elements of the pixel to an array:
     *       Multiply with mu and the actual pixel area to obtain stokes
     * elements
     *----------------------------------------------------------------------------
     */
    for (k = 0; k < nmat; k++) {
      Sarr[i][k] = SvR[k] * mu * apix[i];
    }

    // The value of the degree of polarization
    Sarr[i][nmat] = P;
  } // End i (pix)

cleanup:
  // Free temporal arrays (every level, bottom-up)
  free_cube(rf, nmat, nmugs);
  free_cube(rfsec, nmat, nmugs);
  // Matrices
  free_matrix(rfmu0, nmat);
  free_matrix(RM, npix);
  // Vectos nmugs
  free(rfsecmu0);
  // Vectos nmats
  free(rf3save);
  free(SvR);

  return ret;
}

size_t fourier_coefficients_size(void) {
  return sizeof(struct FourierCoefficients);
}
