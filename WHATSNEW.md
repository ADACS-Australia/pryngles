# Pryngles

## PlanetaRY spaNGLES

`Pryngles` is changing rapidly.  Here you will find a (non-exhaustive)
list of the features introduced in each version family of the package.

## What's new

- **1.0.x versions**:

  - **Performance — spangle data stored as numpy arrays**: light curves are 1.7x faster for WASP-43b-like systems
    with polarization (11.9 s → 7.1 s for 11 time steps) and 7.7x faster with diffuse reflection (5.7 s → 0.7 s);
    spangling is 2x faster. The remaining time with polarization is mostly the Stokes reflection kernel.
  - **API change — `Spangler.data` / `System.data` are now snapshots**: the data is stored as one numpy array per
    column, and `.data` builds a new `pandas.DataFrame` on every access. Modifying the returned DataFrame no longer
    changes the spangler; assign a whole DataFrame (`sg.data = df`) to replace the data. `System.data` is read-only.
    Spanglers and systems saved with earlier versions are converted when loaded.
  - **API change — hidden/transit state columns**: the string columns `hidden_by_*`, `transit_over_*` and
    `string_int` are replaced by integer bitmasks `hidden_mask_*` / `transit_mask_*` (one bit per body, see
    `Spangler.body_bit`) and `transit_rho_*` (distance to the center of the transited body). For example,
    spangles hidden from the observer by the ring: `(sys.data.hidden_mask_obs & sys.sg.body_bit("Ring")) != 0`.
  - **Column dtypes**: coordinate, cosine and Stokes columns are `float64` from construction, and `beta_loc` is
    now a `float64` column (previously `object`).
  - **Benchmarks**: new `benchmarks/bench_lightcurve.py` times the regression-test systems phase by phase (see
    `benchmarks/README.md`).
  - `System` interface is now capable to flux modelling for arbitrary configurations and parameters
  - Methods `System.update_DiffuseReflection` and `System.update_Transit` are adapted from `RingedPlanet` 
    interface to compute Diffuse Reflection and Transit effects.
  - `System.update_ThermalEmission` introduced as a new feature to calculate the emmited flux
    from a planetary surface based on temperature distribution models on tidally locked exoplanets based on [`SPIDERMAN` code](https://spiderman.readthedocs.io/)
  - `System.compute_lightcurve` is now included to facilitate the user's workflow for simulations
  - Synthetic signal generation is now possible based on Poisson stocastic photon count
  - Spangle computation is nearly 50% percent faster. Optimization was achieved by vectorized some calculus
  - Transit light curves: fixed flux accumulation to ensure non-flat transit depths in recent `pandas` versions.
  - Reflected-light curves: fixed incident/reflected flux updates to ensure diffuse reflection contributions are applied correctly.
  - Polarimetric light curves: support for polarized flux and degree of polarization in `System.compute_lightcurve(effects=["polarization"])`.
  - **Documentation — `DESCRIPTION.md`**: new engineering- and science-oriented description of the codebase added at the repository root. Covers the core modeling concept (spangles), both interfaces (`System` and legacy `RingedPlanet`), the full computation pipeline, module-by-module package structure, performance notes, and external links (GitHub, PyPI, ReadTheDocs, ASCL). Fully hyperlinked with raw GitHub image URLs.
  - **Documentation — `README.md`**: added a complete `System` (official) interface quickstart section showing how to build a Star/Planet/Ring hierarchy, initialize the simulation, compute a polarimetric light curve, and extract body-wise contributions from pandas DataFrames. Gallery images embedded inline. Legacy interface heading typo fixed.
  - **Tutorials — new `tutorials/Quickstart-System.ipynb`**: step-by-step quickstart notebook using the modern `System` interface. Covers body creation, observer setup, orbital dynamics, spangling, flux/polarization light-curve computation, and pandas-based extraction of Planet and Ring contributions.
  - **Tutorials — `tutorials/Quickstart.ipynb` renamed to `tutorials/Quickstart-RingedPlanet.ipynb`**: clarifies that this notebook uses the legacy `RingedPlanet` interface. All internal and external references updated accordingly.
  - **Tutorials — Colab support**: all three notebooks in `tutorials/` now include an "Open in Colab" badge and a conditional `pip install pryngles` cell.
  - **Gallery — new images**: added gallery images for the `System` interface quickstart and the WASP-43b tutorial workflow.

- **0.9.x versions**:

  - Pryngles data repository included.
  - A major issue in the units was corrected when initializing RP with
    the system interface.
  - Major upgrade in the optical model of the legacy module.
  - Reflection is now calculated using cpixx extension.
  - Method `updateReflection` introduced to calculate the flux
    reflected on a realistic surface (atmospheric atmosphere or
    ring/disk).
  - Scattering is calculated from Fourier coefficients as calculated
    by PyMieDap.
  - Solved some problems with Pandas in the new versions.
  - New scatterer module.
  - Scatterers introduced: LambertianGraySurface,
    LambertianGrayAtmosphere, BlackBody, Neutral
  - Sizes of the spangles in the plot2d routine are now calculated in
    a smarter way taking into account the scale of the plot.
  - Multiple light-sources are now included into the package: you can
    simulate a binary system with its respective planets, rings and disks.
  - Source code of the package is distributed in a clean form (using
    xconvert to convert Developer Jupyter Notebooks to python code).
  - Illumination from multiple sources of light (separated).
  - Brand new modular structure of the package.
  - Arbitrary number of bodies.
  - Illuminations and visibility computed numerically.
  - N-body problem orbital calculations (Rebound).
  - Spangles info stored in Pandas DataFrames.
  - Azimuths and further information on spangles.
  - Many modules with a more friendly interface.
  - Contribution and extension made easy.
  - Realistic previsualization.

- **0.7.x versions**:

  - Add Center displacement to Spangler.
  - Improve performance of matrix-vector transformation.
  - Join spanglers methods able to produce complex sets of objects.
  - Separation of Sampler and Spangler class.
  - All changes from the 0.6.1.x tests were assumed.
  - The code was refactored to make it more modular.
  - In the new version of the code we have made public a complete
    notebook illustrating the use of the package for producing
    light-curves of scientific uses
    (pryngles-examples-exploration.ipynb).
  - New class, `Spangler`, intented to sample with a Fibonacci
    distribution of points the surface of spheres and disks.  In the
    `RingedPlanet` interface of `Pryngles` the `Spangler` class correspond to
    the class `Sample`.
  - The spangler class have been implemented to a point of creating
    multiple spanglers in a single one.
  - We add the capability to preview spangles with the Spangler methods.
  - New classes introduced: Spangle.
  - New methods introduced in body class: spangle_body.
  - A new tutorial for developers have been added.
   
- **0.6.x versions**:

  - 0.6.0 is the official release version, after paper acceptance and
    arXiv submission.
  - File `version.py` included.
  - Link to quickstart tutorial in Google Colab, updated.
  - Updated information about paper in the arXiv and ACL code.

- **0.5.x versions**:
  
  - Preview method plotRingedPlanet modified to work under Google Colab.
  - Physical and astronomical constants included.
  - A new tutorial was included.
  - A major update in the classes to create and populate planetary
    system.

- **0.4.x versions**:

  - A new model to create and populate planetary system has been
    implemented.

- **0.3.x versions**:

  - A water mark with version number included.
  - Version is now available in the __version__ variable.
  - Scattering formulae tested and verified.
  - Package has been compared against similar packages (good
    agreement) but disclaimer has been done.
  - New version number scheme: 0.x.y (x-major, y-minor release),
    0.x.y.z (z test version).
  - Major corrections in diffuse formulae.

- **0.2.1.x versions**:

  - Tutorial is now working in Google Colab.
  - References were corrected.
  - The home url was set as the PyPI web page.
  - Non-linear (4th order) limb darkening included.
  - Added the class `Extra`.
  - Function to draw logo: `drawPryngles`.
  - Added function `prynglesMark`.
  - Now `__version__` variable is available.

- **0.2.0.x versions**:

  - First official version of the package.


### Test versions

These are the improvements coming in the next releases of the package.
Improvements in the 0.6.1.x test versions will be released in the
0.7.x official versions.

- **0.7.3.x versions**:

  - We have greatly improved the algorithms to compute intersections.

- **0.6.1.x versions**:

  - This is branch `refactor`.
  - We have refactor the package to make it much more modular.
  - The previous version (`RingedPlanet` interface) have been preserved in the
    `legacy` module.

- **0.7.0.x versions**:

  - New.

