# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add project root to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Kosmic Lab"
copyright = "2025, Kosmic Lab Team"
author = "Kosmic Lab Team"
release = "1.0.0"
version = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx.ext.todo",  # Support for todo items
    "sphinx.ext.coverage",  # Documentation coverage checker
    "sphinx.ext.mathjax",  # Math formula support
    "sphinx.ext.githubpages",  # GitHub Pages support
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "all"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Templates path
templates_path = ["_templates"]

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pygments style for syntax highlighting
pygments_style = "sphinx"

# Todo extension
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # ReadTheDocs theme
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "display_version": True,
}

html_static_path = ["_static"]
html_css_files = []

html_logo = None
html_favicon = None

# Output file base name
htmlhelp_basename = "KosmicLabdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

latex_documents = [
    (
        master_doc,
        "KosmicLab.tex",
        "Kosmic Lab Documentation",
        "Kosmic Lab Team",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "kosmiclab", "Kosmic Lab Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "KosmicLab",
        "Kosmic Lab Documentation",
        author,
        "KosmicLab",
        "AI-Accelerated Consciousness Research Platform",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets)
html_context = {
    "display_github": True,
    "github_user": "Luminous-Dynamics",
    "github_repo": "kosmic-lab",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
