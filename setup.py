##################################################################
#                                                                #
# #####...#####...##..##..##..##...####...##......######...####..#
# ##..##..##..##...####...###.##..##......##......##......##.....#
# #####...#####.....##....##.###..##.###..##......####.....####..#
# ##......##..##....##....##..##..##..##..##......##..........##.#
# ##......##..##....##....##..##...####...######..######...####..#
# ...............................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    # ######################################################################
    # BASIC DESCRIPTION
    # ######################################################################
    name='pryngles',
    author="Jorge I. Zuluaga, Sebastian Numpaque, Allard Veenstra, Jaime A. Alvarado-Montes, Mario Sucerquia",
    author_email="jorge.zuluaga@udea.edu.co",
    description="PlanetaRY spanGLES: general photometry of planets, rings and shadows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/pryngles",
    keywords='astronomy exoplanets planetary-rings',
    license='MIT',

    # ######################################################################
    # CLASSIFIER
    # ######################################################################
    # For Development status see: https://martin-thoma.com/software-development-stages/
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        ],
    version='1.0.2',

    # ######################################################################
    # FILES
    # ######################################################################
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    
    # ######################################################################
    # EXTENSIONS
    # ######################################################################
    ext_modules=[
        setuptools.Extension(name="pryngles.cpixx",
                  sources=["src/pryngles/cpixx/cpixx.c",
                           ]),
    ],
    
    # ######################################################################
    # ENTRY POINTS
    # ######################################################################
    entry_points={
        'console_scripts': ['install=pryngles.install:main'],
    },

    # ######################################################################
    # TESTS
    # ######################################################################
    test_suite='nose.collector',
    tests_require=['nose'],

    # ######################################################################
    # DEPENDENCIES
    # ######################################################################
    install_requires=['rebound>=4,<5','scipy','ipython',
	              'matplotlib','tqdm','dill','astropy',
	              'spiceypy','cmasher','pandas','celluloid',
	              'sigfig','anytree','ipywidgets','gdown'],
    python_requires=">=3.12",

    # ######################################################################
    # OPTIONS
    # ######################################################################
    include_package_data=True,
    package_data={"": ["data/*.*", "data/sampler_presets/*.*", "tests/*.*"]},
)
