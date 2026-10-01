# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src/pryngles'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pryngles'
copyright = '2026, Jorge I. Zuluaga, Sebastian Numpaque, Allard Veenstra, Jaime A. Alvarado-Montes, Mario Sucerquia'
author = 'Zuluaga et al.'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_mdinclude', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode']

napoleon_numpy_docstring = True
napoleon_google_docstring = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygment_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_theme_options = {
    "logo": {
        "image_light": "pryngles_logo.png",
        "image_dark": "pryngles_logo.png",
        "text": f"{project} {release} Documentation",
    },
    "external_links": [
    {"name": "Source", "url": "https://github.com/seap-udea/pryngles"},
  ]
    }