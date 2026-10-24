#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Maximum number of tabulated points dderiv() can handle (size of its
// scratch array), and maximum number of Stokes elements reflection() supports
// (size of Bplus).
#define MAX_SPLINE_N 1000
#define MAX_NSTOKES 4
#define MAX_NMAT (MAX_NSTOKES - 1)

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

static void *zeros1d(int n) { return xcalloc(n, sizeof(double)); }

static void *zeros2d(int n, int m) {
  return xcalloc(n, (size_t)m * sizeof(double));
}

static void *zeros3d(int n, int m, int k) {
  return xcalloc(n, (size_t)m * (size_t)k * sizeof(double));
}

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

int bisect(const double x[], int n, double xi) {
  int k;
  int klo = 0;
  int khi = n - 1;

  while ((khi - klo) > 1) {
    k = (klo + khi) / 2; // No need to floor() here, since it's integer division
    if (x[k] > xi)
      khi = k;
    else
      klo = k;
  }

  return klo;
}

/*
 * Find the bracketing indices and interpolation coefficients for xi in x.
 *
 * Returns 0 on success, -1 if n < 2 or the bracketing interval is
 * degenerate (x not strictly increasing). On failure the outputs are
 * left untouched.
 */
int spline_coefficients(const double x[], int n, double xi, int *klo_out,
                        int *khi_out, double *a_out, double *b_out,
                        double *h_out) {
  if (n < 2) {
    fprintf(stderr, "Error in spline_coefficients: n=%d < 2.\n", n);
    return -1;
  }

  // Find indices to interpolate between
  int klo = bisect(x, n, xi);
  int khi = klo + 1;

  double h = x[khi] - x[klo];

  if (fabs(h) < 1e-10) {
    fprintf(stderr, "Error in spline_coefficients: bad x input.\n");
    return -1;
  }

  // Coefficients
  *a_out = (x[khi] - xi) / h;
  *b_out = (xi - x[klo]) / h;
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
 *     Given the arrays x and y of length n, which tabulate a function
 *     (with the x(i)'s in order), and given the array ddy, which is
 *     the output from SPLINE above, and given a specific xi, this
 *     routine returns a cubic-spline interpolated value yi.
 *
 *     NOTE: for computational efficiency, we pre-compute lower and upper
 *     indices as well as the spline coefficients (above), since they don't
 *     change for an individual pixel/spangle.
 *----------------------------------------------------------------------------
 */
double interp(const double y[], const double ddy[], int klo, int khi, double a,
              double b, double h) {
  double yi =
      a * y[klo] + b * y[khi] +
      ((a * a * a - a) * ddy[klo] + (b * b * b - b) * ddy[khi]) * (h * h) / 6;
  return yi;
}

/*
 *----------------------------------------------------------------------------
 *     Calculate the Stokes vector for a given geometry (given a table of
 *     Fourier coefficients read from a file).
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
 *     stokes is not (fully) written.
 *
 *     stokes is a flat row-major buffer of npix * (nmat + 1) doubles: row n
 *     holds the nmat Stokes elements of pixel n followed by the degree of
 *     polarisation.
 *
 *----------------------------------------------------------------------------
 */
int reflection(int nmat, int nmugs, int nfou, const double *xmu,
               const double fou[nfou][nmat][nmugs][nmugs], int npix,
               const double phi[npix], const double beta[npix],
               const double theta0[npix], const double theta[npix],
               const double apix[npix], double stokes[npix][nmat + 1]) {

  double muold = 1, mu0old = 1;
  int ret = 0;
  double Bplus[MAX_NSTOKES];

  // Validate sizes before allocating anything

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
  if (npix < 1) {
    fprintf(stderr, "Error in reflection: npix=%d is < 1.\n", npix);
    return -1;
  }
  if (nfou < 1) {
    fprintf(stderr, "Error in reflection: nfou=%d is < 1.\n", nfou);
    return -1;
  }

  // Allocate temporary arrays with VLA pointers,
  // using calloc (so that they're on the heap)
  double (*ddf)[nmugs][nmugs] = zeros3d(nmat, nmugs, nmugs);
  double (*f_mu0)[nmugs] = zeros2d(nmat, nmugs);
  double (*RM)[nmat] = zeros2d(npix, nmat);
  double *ddf_mu0 = zeros1d(nmugs);

  // nmat is small, these can go on the stack
  double f_mu0_mu[MAX_NSTOKES] = {0};
  double SvR[MAX_NSTOKES] = {0};

  // Loop over the Fourier coefficients:
  for (int m = 0; m < nfou; m++) {

    double fac = 1.0;
    if (m == 0)
      fac = 0.5;

    // Initialize the interpolation matrix for the current fourier coefficient:
    for (int i = 0; i < nmugs; i++) {
      for (int k = 0; k < nmat; k++) {
        // Row fou[m][k][i][:] is contiguous: no copy needed.
        // Write directly into corresponding ddf row.
        // nmugs was validated above, so dderiv() cannot fail here.
        dderiv(xmu, fou[m][k][i], nmugs, ddf[k][i]);
      }
    }

    /*
     *----------------------------------------------------------------------------
     *     Loop over the pixels:
     *     If the input angles are (very) similar to a previously calculated
     *     case use those values. To obtain obtain the fourier coefficient at
     *     (mu,mu0) spline has to be called a second time.
     *----------------------------------------------------------------------------
     */
    for (int n = 0; n < npix; n++) {
      double mu = theta[n];
      double mu0 = theta0[n];
      double cosmphi = cos(m * phi[n]);
      double sinmphi = sin(m * phi[n]);
      Bplus[0] = cosmphi;
      Bplus[1] = cosmphi;
      Bplus[2] = sinmphi;
      Bplus[3] = sinmphi;

      if ((n <= 0) || (fabs(mu - muold) > 1e-6) ||
          (fabs(mu0 - mu0old) > 1e-6)) {
        // Interpolate the fourier terms at j=mu0 and i=mu, otherwise skip and
        // reuse previous cached value

        // Get indices and coefficients to work with (at mu0)
        int klo, khi;
        double a, b, h;
        if (spline_coefficients(xmu, nmugs, mu0, &klo, &khi, &a, &b, &h)) {
          ret = -1;
          goto cleanup;
        }

        for (int i = 0; i < nmugs; i++) {
          for (int k = 0; k < nmat; k++) {
            // Use fou(m,k,i,:) and ddf(k,i,:) slices directly, and write
            // straight into rmu0(k,i,:) Do the spline interpolation between klo
            // and khi, given the coefficients a,b,h
            f_mu0[k][i] = interp(fou[m][k][i], ddf[k][i], klo, khi, a, b, h);
          }
        }

        // Get indices and coefficients to work with (now at mu)
        if (spline_coefficients(xmu, nmugs, mu, &klo, &khi, &a, &b, &h)) {
          ret = -1;
          goto cleanup;
        }

        for (int k = 0; k < nmat; k++) {
          dderiv(xmu, f_mu0[k], nmugs, ddf_mu0);
          // Do the spline interpolation between klo and khi, given the
          // coefficients a,b,h
          f_mu0_mu[k] = interp(f_mu0[k], ddf_mu0, klo, khi, a, b, h);
          muold = mu;
          mu0old = mu0;
        }
      }

      // Compute the reflection matrix at pixel n and stokes parameter k,
      // summing over fourier terms m
      for (int k = 0; k < nmat; k++) {
        RM[n][k] = RM[n][k] + 2 * Bplus[k] * fac * f_mu0_mu[k];
      }

    } // End loop over pixels
  } // End loop over fourier coefficients

  // Loop again over the pixels to rotate the Stokes vector
  for (int n = 0; n < npix; n++) {
    double mu = theta[n];
    double mu0 = theta0[n];

    // Calculate the locally reflected Stokes vector:
    for (int k = 0; k < nmat; k++)
      SvR[k] = mu0 * RM[n][k];

    // Rotate Stokes elements Q and U to the actual reference plane.
    // Note: does this implicitly assume nmat=3 ?
    double cosb2 = cos(2 * beta[n]);
    double sinb2 = sin(2 * beta[n]);
    double SvR2 = cosb2 * SvR[1] + sinb2 * SvR[2];
    double SvR3 = -sinb2 * SvR[1] + cosb2 * SvR[2];
    SvR[1] = SvR2;
    SvR[2] = SvR3;

    // Compute the local degree of polarisation P:
    double P;
    if (fabs(SvR[0]) < 1e-6)
      P = 0;
    else if (fabs(SvR[2]) < 1e-6)
      P = -SvR[1] / SvR[0];
    else
      P = sqrt(SvR[1] * SvR[1] + SvR[2] * SvR[2]) / SvR[0];
    if (fabs(P) < 1e-6)
      P = 0;

    // Multiply with mu and the actual pixel area to obtain the stokes
    // element k for each pixel n.
    for (int k = 0; k < nmat; k++) {
      stokes[n][k] = SvR[k] * mu * apix[n];
    }

    // The value of the degree of polarization (last index)
    stokes[n][nmat] = P;

  } // End loop over pixels

cleanup:
  // Free temporary arrays
  free(ddf);
  free(f_mu0);
  free(RM);
  free(ddf_mu0);

  return ret;
}
