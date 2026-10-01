##################################################################
#                                                                #
# .#####...#####...##..##..##..##...####...##......######...####..#
# .##..##..##..##...####...###.##..##......##......##......##.....#
# .#####...#####.....##....##.###..##.###..##......####.....####..#
# .##......##..##....##....##..##..##..##..##......##..........##.#
# .##......##..##....##....##..##...####...######..######...####..#
# ................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################

# --------------------------------------------------
# External required packages
# --------------------------------------------------


import matplotlib.pyplot as plt

# Plotting in 3d
import numpy as np
from matplotlib.patches import Circle

from pryngles import *


# --------------------------------------------------
# Class Sampler
# --------------------------------------------------
class Sampler(PrynglesCommon):
    r"""
    Class for generating evenly distributed points on various geometric shapes
    using the Fibonacci spiral algorithm. It allows for customization of the
    number of points, random seed for reproducibility, and loading/saving of samples
    to binary files. Presets for common geometries like spheres, circles, and rings are also available.

    Parameters
    ----------------
    N : `int`
        Number of points to generate the surface sampling | **Default** = 1000

    seed : `integer`
        Value of the integer seed of random number generation (if 0 no random seed is set).
        If a non-zero seed is used the position of the points for a given geometry will be
        always the same | **Default** = 0

    filename : `str`
        To initialize and load object from a binary file.
        The binary file should be previously prepared with the method :data:`~ __init__.PrynglesCommon.save_to` of the class.
        | **Default =** None

    preset : `tuple`
        To load a preset sample from disk of the type preset.
        Preset should have two components:

        - **geometry** (`str`)
          Possible values are from :any:`consts.SAMPLE_SHAPES`

        - **geometry_args** (`dict : dict(ri = 0)`)
          Arguments of the routine to generate sample from preset.
          Inner radius (``ri``) of the ring when ``geometry = "ring"``

    Attributes
    ----------
    dim : `int`.
        Dimension of sampling.

    ss, pp : `numpy.array`.
        (N x 3) Position of the points in cartesian and spherical coordinates :math:`(r, θ, \phi)`,
        where :math:`θ` is azimutal angle (angle with respecto to x-axis) and :math:`\phi` is "elevation"
        (complement of polar angle).

    purged : `bool`
        ¿Is the sample purged?. Purge is the process by which points too close are
        removed until the average distance between points is similar | **Default =** `False`

    dmin, dmed, dmax : `float`
        Minimum, median and maximum distance between points.

    ds : `numpy.array`
        Distances to the nearest neighbor for all the points.

    dran : `float`
        Range of distances between points (``dmax - dmin``). While the smaller this quantity the better.

    dstar : `float`
        As measure of distances :math:`\sqrt{N}d_{med}`. Typically this value is between 2.4 and 3.4
        (see source http://extremelearning.com.au/evenly-distributing-points-on-a-sphere)


    Returns
    -------
    :
        output : :data:`~ sampler.Sampler`
            Sampler object containing the initialize attribute to compute the surface sampling.

    Raises
    ------
    ValueError
        if ``geometry`` parameter is not a value from :any:`consts.SAMPLE_SHAPES`

    See Also
    --------
    save_to()
        Save object to a binary file | `filename = str`
    load_to()
        Read object from a binary file | `filename = str`
    """

    # --------------------------------------------------
    # Bassic methods
    # --------------------------------------------------
    def __init__(
        self,
        # Initialize from a file
        filename=None,
        # Initialize from a preset
        preset=None,
        # Initialize from parameters
        N=1000,
        seed=0,
    ):

        # If filename is provided load object from filename
        if filename:
            verbose(VERB_SIMPLE, f"Loading sampler from {filename}")
            self.load_from(filename)
            return

        # If preset is provided, read preset from file
        if preset:
            verbose(VERB_SIMPLE, f"Loading sampler from preset: {preset}")

            geometry = preset[0]
            geometry_args = preset[1]

            if geometry not in SAMPLER_PRESETS:
                raise ValueError(
                    f"No presets for {geometry} available.  This are the available presets: {SAMPLER_PRESETS}."
                )

            # Modify N if it is a ring
            qring = False
            if geometry == "ring":
                ri = geometry_args["ri"]
                verbose(VERB_VERIFY, f"Original ring preset {N}")
                N = int(N / (1 - ri**2))
                verbose(VERB_VERIFY, f"Computed ring preset {N}")
                geometry = "circle"
                qring = True

            # Calculate the closest Npreset
            exec(
                f"self.Npreset=SAMPLER_{geometry.upper()}_PRESETS[abs({N}-SAMPLER_{geometry.upper()}_PRESETS).argmin()]"
            )

            Npreset = self.Npreset
            filename = Misc.get_data(f"sampler_presets/sample_{geometry}_N_{Npreset}.pkl")
            verbose(VERB_SYSTEM, f"Reading preset data from {filename}")
            self.load_from(filename)
            self.Npreset = Npreset
            self.filename = filename

            # If the preset is a ring cut a hole
            if qring:
                verbose(VERB_VERIFY, f"Cutting hole with ri = {ri}")
                self._cut_hole(ri)
            return

        # Basic
        self.N = N
        self.seed = seed

        # Derivative
        self.dim = 0
        self.geometry = -1
        self.ss = None
        self.pp = None
        self.ns = None
        self.dmin = self.dmed = self.dmax = self.dran = self.dstar = 0

        # Purge
        self.purged = False

        # Plotting
        self.cargs = dict(color="k", fill=False, alpha=0.1)  # 2d plot
        self.wargs = dict(color="k", lw=0.1)  # 3d plot

        # Random seed
        self._seed_sampler()

    def _get_min_distance(self, r, rs):
        """
        Get the minimum distance from point r to points rs

        Parameter:
            r: numpy array (3):
                coordinates of the point.
            rs: numpy array (N)
                coordinates of the points.

        Return:
            dmin: float
                minimum distance.
        """
        deltas = rs - r
        dist = np.einsum("ij,ij->i", deltas, deltas)
        imin = np.argsort(dist)[1]
        return np.sqrt(dist[imin])

    def _calc_distances(self):
        """
        Calculate the minimum distances of all points in the sample.

        Update:
            ds,dmin,dmax,dmed,dran,dstar.
        """
        self.ds = np.array([self._get_min_distance(self.ss[i], self.ss) for i in range(len(self.ss))])
        self.dmin = self.ds.min()
        self.dmax = self.ds.max()
        self.dmed = np.median(self.ds)
        self.dran = self.dmax - self.dmin
        self.dstar = np.sqrt(self.N) * self.dmed

        # Area of each spangle
        self.aes = self.A / self.N

        # Effective diameter
        self.deff = 2 * (self.aes / np.pi) ** 0.5

    def _seed_sampler(self):
        if self.seed:
            np.random.seed(self.seed)

    # --------------------------------------------------
    # Tested methods from module file sampler
    # --------------------------------------------------

    def plot(self, spangled=dict(), **args):
        """
        Method to visualize a plot of the sample generated.

        Parameters
        -----------------------
        spangled : `dict`
            Scatter plotting options

        Examples
        ----------------
        >>> # Creating Sampler object
        >>> sp = pr.Sampler(N = 1000)
        >>>
        >>> # Generate sample in a unitary circle
        >>> sp.gen_circle()
        >>>
        >>> # Plot the sample
        >>> sp.plot()
        >>> sp.ax.set_title(f"Spangler, N = {sp.N}", fontsize = 10)
        >>> sp.fig.tight_layout()

        .. image:: images/sampler.png
            :align: center
            :width: 600px
        """
        sargs = dict(c="k", s=1.5)
        sargs.update(args)

        # Spangles options
        if spangled:
            defaults = dict(alpha=0.3)
            defaults.update(spangled)

        if self.dim == 2:
            self.fig, self.ax = plt.subplots()
            self.ax.scatter(self.ss[:, 0], self.ss[:, 1], **sargs)
            self.ax.add_patch(plt.Circle((0, 0), 1, **self.cargs))
            self.ax.set_aspect("equal")
            if spangled:
                for isp in range(self.N):
                    c = Circle(self.ss[isp], self.deff / 2, **defaults)
                    self.ax.add_patch(c)

        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.scatter(self.ss[:, 0], self.ss[:, 1], self.ss[:, 2], **sargs)
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            self.ax.plot_wireframe(x, y, z, **self.wargs)

            if spangled:
                for isp in range(self.N):
                    Plot.circle3d(self.ax, self.ss[isp], self.deff / 2, zDir=self.ss[isp], **defaults)

            self.ax.set_box_aspect([1, 1, 1])

        self.fig.tight_layout()

    def gen_circle(self, perturbation=1, boundary=2):
        """Sample points in fibonacci spiral on the unit circle

        Parameters
        ----------------
        perturbation : `int`
            Type of perturbation (0 normal perturbation, 1 random perturbation)
            | **Default** = `1`
        boundary : `int`
            Type of boundary (0 jagged, >1 smooth)
            | **Default** = `2`

        Examples
        -----------------
        >>> # Modify some properties of plot
        >>> sp.plot(spangled=dict(color='r'))
        >>>
        >>> # Show some quantities of the sample, See Attributes of Sampler Class
        >>> sp.ax.set_title(f"N = {sp.N}, dmed = {sp.dmed:.4f}, deff = {sp.deff:.4f}", fontsize = 10)
        >>> sp.fig.tight_layout()

        .. image:: images/sampler_gen_circle.png
            :align: center
            :width: 600px
        """
        self._seed_sampler()
        self.geometry = SAMPLER_GEOMETRY_CIRCLE

        # Unitary radius
        self.R = 1
        self.ri = 0  # For compatibility with ring

        # Total area
        self.A = np.pi * self.R**2

        shift = 1.0 if perturbation == 0 else self.N * np.random.random()

        ga = np.pi * (3.0 - np.sqrt(5.0))

        # Boundary points
        np_boundary = round(boundary * np.sqrt(self.N))

        self.dim = 2
        self.ss = np.zeros((self.N, 3))
        self.pp = np.zeros((self.N, 3))
        self.ns = np.zeros((self.N, 3))
        j = 0
        for i in range(self.N):
            if i > self.N - (np_boundary + 1):
                r = 1.0
            else:
                r = np.sqrt((i + 0.5) / (self.N - 0.5 * (np_boundary + 1)))
            phi = ga * (i + shift)
            self.ss[j, :] = np.array([r * np.cos(phi), r * np.sin(phi), 0])
            self.ns[j, :] = np.array([0, 0, 1])
            self.pp[j, :] = np.array([r, np.mod(phi, 2 * np.pi), 0])
            j += 1

        # Distances
        self._calc_distances()

    def _cut_hole(self, ri=0):
        """
        Cut a hole in the data (this applies to circle sampling)
        """
        # Purge points
        cond = ~((self.pp[:, 1] > 0) & (self.pp[:, 0] <= ri))
        self.pp = self.pp[cond]
        self.ss = self.ss[cond]
        self.ns = self.ns[cond]
        self.N = len(self.pp)

        # Correct area
        self.A = self.A - np.pi * ri**2
        self.ri = ri

        # Distances
        self._calc_distances()

    def gen_ring(self, ri=0.5, perturbation=1, boundary=2):
        """Sample points in fibonacci spiral on the unit circle, but including an inner gap (as in ring)

        Parameters
        --------------
        ri: `float`
            Inner radius of the ring | **default** = `0.5`
        perturbation : `int`
            Type of perturbation (0 normal perturbation, 1 random perturbation)
            | **Default** = `1`
        boundary : `int`
            Type of boundary (0 jagged, >1 smooth)
            | **Default** = `2`

        Example
        -------------
        >>> # We can modify our sampling
        >>> sp.N=500
        >>>
        >>> # Generate a sampling for a ring
        >>> sp.gen_ring(0.5,boundary=1)
        >>>
        >>> # Let's plot it!!
        >>> sp.plot(c = 'r',s = 0.5,spangled = dict(color = 'r',alpha = 0.1))
        >>> sp.ax.set_title(f"N = {sp.N}, dmin = {sp.dmin:.2e}, dmax = {sp.dmax:.2e}", fontsize = 8);
        >>> sp.fig.tight_layout()

        .. image:: images/sampler_gen_ring.png
            :align: center
            :width: 600px
        """
        if self.N < SAMPLER_MIN_RING:
            raise ValueError(
                f"The number of points for a ring shouldn't be lower than {SAMPLER_MIN_RING}.  You provided {self.N}"
            )

        if ri >= 1:
            raise ValueError(f"The radius of the inner ring should be less than 1 (r ={ri} provided)")

        # Compute effective number
        self.N = int(self.N / (1 - ri**2))
        self.gen_circle(perturbation, boundary)

        # Cut hole
        self._cut_hole(ri)

    def gen_sphere(self, perturbation=1):
        """Sample points in the unit sphere following fibonacci spiral

        Parameters
        --------------
        perturbation : `int`
            type of perturbation (0 normal perturbation, 1 random perturbation)
            | **default** = `1`

        Examples
        -------------
        >>> # If you want that the spangle position not be different each time use
        >>> sp = pr.Sampler(N = 1000, seed = 1)
        >>>
        >>> # Create a sphere sampling
        >>> sp.gen_sphere()
        >>>
        >>> # And plot it
        >>> sp.plot()
        >>> sp.ax.set_title(f"N = {sp.N}, dmin = {sp.dmin:.4f}, dmax = {sp.dmax:.4f}", fontsize = 8);
        >>> sp.fig.tight_layout()

        .. image:: images/sampler_gen_sphere.png
            :align: center
            :width: 600px
        """
        self._seed_sampler()
        self.geometry = SAMPLER_GEOMETRY_SPHERE

        # Unitary radius
        self.R = 1

        # Total area of the Sample
        self.A = 4 * np.pi * self.R**2

        shift = 1.0 if perturbation == 0 else self.N * np.random.random()

        ga = np.pi * (3.0 - np.sqrt(5.0))
        offset = 2.0 / self.N

        self.dim = 3
        self.ss = np.zeros((self.N, self.dim))
        self.pp = np.zeros((self.N, self.dim))
        self.ns = np.zeros((self.N, self.dim))
        j = 0
        for i in range(self.N):
            phi = ga * ((i + shift) % self.N)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = ((i + 0.5) * offset) - 1.0
            sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
            theta = np.arccos(cos_theta)
            self.ss[j, :] = np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta])
            self.ns[j, :] = self.ss[j, :].copy()
            self.pp[j, :] = np.array([1, np.mod(phi, 2 * np.pi), np.pi / 2 - theta])
            j += 1

        # Distances
        self._calc_distances()

    def purge_sample(self, tol=0.5):
        """
        Purge sample, ie. remove points close than a given threshold.

        Optional parameters:
            tol: distance to purge, ie. if dmin<tol*dmed then purge, float

        Update:
            ss, pp, N, _purge
        """
        if self.purged:
            verbose(VERB_SIMPLE, "Already purged.")
            return 0
        else:
            verbose(VERB_SIMPLE, "Purging sample")

        self.purged = True
        purge = True
        while purge:
            self._calc_distances()
            if self.dmin < tol * self.dmed:
                ipurge = np.argsort(self.ds)[0]
                self.ss = np.delete(self.ss, ipurge, 0)
                self.ns = np.delete(self.ns, ipurge, 0)
                self.pp = np.delete(self.pp, ipurge, 0)
                self.N -= 1
                self.purged = True
                purge = True
            else:
                purge = False

    def update_normals(self, ss):
        """Update normal vectors according to geometry

        Parameters:
            ss: array (Nx3):
                Cartesian coordinates of points.

        Return:
            ns: array (Nx3):
                Normals to geometry.
        """
        if self.geometry == SAMPLER_GEOMETRY_CIRCLE:
            ns = np.array([[0, 0, 1]] * self.N)
        elif self.geometry == SAMPLER_GEOMETRY_SPHERE:
            ns = ss / np.linalg.norm(ss, axis=-1)[:, np.newaxis]
        return ns
