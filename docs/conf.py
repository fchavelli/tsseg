# docs/conf.py
import os
import sys
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

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
    "sphinx.ext.napoleon",  # Google/NumPy docstrings
    "sphinx.ext.intersphinx",  # cross-project links
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "tsseg_taxonomy",  # docs/_ext: the interactive taxonomy page
]

autosummary_generate = True

# -- HTML output -----------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "tsseg"
html_favicon = "_static/img/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["css/tsseg.css"]
html_js_files = [("js/tsseg.js", {"defer": "defer"})]
html_show_sourcelink = False

html_theme_options = {
    "logo": {
        "text": "tsseg",
        "image_light": "_static/img/logo-mark.svg",
        "image_dark": "_static/img/logo-mark-dark.svg",
        "alt_text": "tsseg - Home",
    },
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "Live demo on Hugging Face Spaces",
            "url": "https://huggingface.co/spaces/fchavelli/tsseg",
            "icon": "fa-solid fa-play",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/tsseg/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/fchavelli/tsseg",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_prev_next": False,
    "navigation_with_keys": False,
    "pygments_light_style": "github-light",
    "pygments_dark_style": "github-dark",
    "secondary_sidebar_items": {
        # taxonomy-link renders on the detector pages only
        "**": ["page-toc", "taxonomy-link", "edit-this-page"],
        "index": [],
    },
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
}

# Pages without the left sidebar: the landing page and the taxonomy
html_sidebars = {"index": [], "taxonomy": [], "changelog": []}

html_context = {
    "github_user": "fchavelli",
    "github_repo": "tsseg",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}

# Copy code blocks without their prompts
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Allow Markdown documents
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = [".rst", ".md"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Mock optional dependencies so autodoc can import modules without installing
# heavy libraries on the documentation builder.
autodoc_mock_imports = [
    "claspy",
    "keras",
    "keras_tcn",
    "networkx",
    "numba",
    "pmdarima",
    "prophet",
    "stumpy",
    "tcn",
    "tensorflow",
    "torch",
    "tsfresh",
    "tsseg_tirex",
    "tslearn",
    "workflows",
]

# Bibliography-style references (.. [1]) in docstrings are not cited inline;
# suppress the corresponding Sphinx warning.
suppress_warnings = ["ref.footnote"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "aeon": ("https://www.aeon-toolkit.org/en/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
