# Allow the use of plugins from other directories in $PYTHONPATH

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
del extend_path
