"""Sphinx configuration for the quchip documentation site."""

import quchip

project = "quchip"
author = "Ibraheem AlYousef"
copyright = "2026, Ibraheem AlYousef"
version = quchip.__version__
release = quchip.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "qutip": ("https://qutip.readthedocs.io/en/latest/", None),
}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "attrs_inline"]
myst_heading_anchors = 4

# The API is documented both at its public re-export location (`quchip`) and at
# the defining module; the resulting duplicate-index and ambiguous-reference
# warnings are inherent to that layout. Docstring RST nits (docutils) and
# repository-relative source links inside included markdown (myst.xref_missing)
# are cosmetic on the rendered pages.
suppress_warnings = ["docutils", "ref.python", "myst.xref_missing", "ref.ref", "ref.footnote"]

# -- HTML --------------------------------------------------------------------

html_theme = "furo"
html_title = f"quchip {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.png"
# Colors come from the quchip identity system (see the wordmark assets):
# ink #16181c / paper #fafbfc, #f2f4f6 / accent #c92f33 in light mode;
# text #f2f4f6 / background #14161a / coral #ef6661, hover #ff8a80 in dark mode.
html_theme_options = {
    "light_logo": "quchip-wordmark-light.png",
    "dark_logo": "quchip-wordmark-dark.png",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#c92f33",
        "color-brand-content": "#c92f33",
        "color-link": "#c92f33",
        "color-link--visited": "#c92f33",
        "color-link--visited--hover": "#c92f33",
        "color-foreground-primary": "#16181c",
        "color-background-primary": "#fafbfc",
        "color-background-secondary": "#f2f4f6",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ef6661",
        "color-brand-content": "#ef6661",
        "color-link": "#ef6661",
        "color-link--hover": "#ff8a80",
        "color-link--visited": "#ef6661",
        "color-link--visited--hover": "#ff8a80",
        "color-foreground-primary": "#f2f4f6",
        "color-background-primary": "#14161a",
        "color-background-secondary": "#101216",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/quchip/quchip",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
    ],
}
html_copy_source = False
html_show_sphinx = False
