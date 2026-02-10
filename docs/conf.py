# docs/conf.py
import os
import sys
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.abspath(".."))

project = "tsseg"
copyright = "2024-2026, Félix Chavelli"
author = "Félix Chavelli"
try:
    release = version("tsseg")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",           # Google/NumPy docstrings
    "sphinx.ext.intersphinx",        # cross-project links
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
html_theme = "sphinx_rtd_theme"
html_theme_options = {"collapse_navigation": False}
html_logo = "_static/logo.png"
html_static_path = ["_static"]

# Optionally track the RTD version in the footer
html_context = {"display_github": True, "github_user": "fchavelli", "github_repo": "tsseg"}

# Allow Markdown documents
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = [".rst", ".md"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/modules.rst",
]

# Mock optional dependencies so autodoc can import modules without installing
# heavy libraries on the documentation builder.
autodoc_mock_imports = [
    "torch",
    "networkx",
    "tensorflow",
    "keras_tcn",
    "stumpy",
    "tslearn",
    "prophet",
    "pmdarima",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "aeon": ("https://www.aeon-toolkit.org/en/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]