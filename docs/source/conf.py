"""
Конфигурация Sphinx для Crypto Analytics Platform
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('../..'))

project = 'Crypto Analytics Platform'
author = 'Романов Е.В.'
copyright = f'{datetime.now().year}, {author}'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.mermaid',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autosummary_generate = True
autoclass_content = 'both'
add_module_names = False

language = 'ru'

exclude_patterns = []
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True
}
