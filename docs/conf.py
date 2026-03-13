import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project   = 'WetlandMapper'
author    = 'Manudeo Singh'
copyright = '2026, Manudeo Singh'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
}

autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'show-inheritance': True,
}

templates_path   = ['_templates']
exclude_patterns = ['_build']

html_theme = 'sphinx_rtd_theme'
