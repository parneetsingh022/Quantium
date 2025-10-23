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

html_theme = "furo"

html_theme_options = {
    # Show project name at top of the sidebar
    "sidebar_hide_name": False,

    # Handy keyboard nav (j/k) through the sidebar
    "navigation_with_keys": True,

    # Buttons above the content (shows "Edit on GitHub", source link)
    "top_of_page_buttons": [
        "edit",  # existing "Edit on GitHub"
        "view"  # existing "View source"
    ],

    # Repo info for the "Edit on GitHub" button
    "source_repository": "https://github.com/parneetsingh022/quantium",
    "source_branch": "main",
    "source_directory": "docs/",

    

    # Theme colors
    "light_css_variables": {
        "color-brand-primary": "#2980b9",
        "color-brand-content": "#1f4e79",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9b59b6",
        "color-brand-content": "#bfb3ff",
    },
}


html_static_path = ['_static']
html_css_files = ['custom.css']


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}