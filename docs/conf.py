# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'quantium'
copyright = '2025, Parneet Sidhu'
author = 'Parneet Sidhu'
html_title = 'Quantium Docs'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    "myst_parser"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_logo = "_static/quantium_logo_light2.png"

html_theme_options = {
    # Handy keyboard nav (j/k) through the sidebar
    "navigation_with_keys": True,

    'navigation_depth': -1,

    "show_toc_level": 2,

    "collapse_navigation": False,  # Prevent collapsing of navigation
}


html_static_path = ['_static']
html_css_files = ['custom.css']


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}