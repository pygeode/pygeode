# Allow the use of plugins from other directories in $PYTHONPATH

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
del extend_path

# Include packages via the entry_points mechanism of pkg_resources.
import pkg_resources
for ep in pkg_resources.iter_entry_points('pygeode.plugins'):
  ep.load()
  del ep
del pkg_resources
