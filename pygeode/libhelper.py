# Helper functions for accessing libraries

def get_library_paths():
  ''' Returns list of paths in which pygeode looks for libraries. '''
  import os, sys
  from os import path
  import pygeode

  # Search in PyGeode directory
  paths = [p for p in pygeode.__path__]
  # Search relative to python install prefix
  paths.append(path.join(path.sep, sys.prefix, 'lib', 'pygeode'))
  paths.append(path.join(path.sep, sys.prefix, 'lib'))
  # Search in linux system paths
  paths.append(path.join(path.sep, 'usr', 'lib', 'pygeode'))
  paths.append(path.join(path.sep, 'usr', 'local', 'lib', 'pygeode'))
  # Search LD_LIBRARY_PATH
  paths += [d for d in os.environ.get('LD_LIBRARY_PATH','').split(':')]
  return paths

# Get a library name
# (extends ctypes.util.find_library to include PyGeode-specific paths)
def find_library (name):
  ''' Searches for a library in the list of default paths. Returns the path
  of the first instance found if one is found, otherwise returns None. '''
  from ctypes.util import find_library
  from glob import glob
  import os
  from os import path
  import pygeode

  if '/' in name:
    dir, name = name.rsplit('/',1)
  else:
    dir = ''

  exts = ['.so', '.dll', '.dylib']
  paths = get_library_paths()

  # Return first match
  for libpath in paths:
    for e in exts:
      libnames = glob(path.join(libpath, dir, 'lib' + name + e))
      if len(libnames) > 0: return libnames[0]

  # Search using ctypes find_library
  libname = find_library(name)
  if libname is not None: return libname

  return None

# old code - but might need later?
#  # Try using LD_LIBRARY_PATH???
#  # Why doesn't ctypes respect LD_LIBRARY_PATH?  Why must we force it like this??
#  if not exists(name):
#    dir=os.environ.get('LD_LIBRARY_PATH','')
#    libname = dir + '/lib' + name + '.so'
#    if exists(libname): name = libname


# Load a dynamic library
# Global = Use a global name space so inter-library dependencies can be resolved
def load_lib (name, Global=False):
# {{{
  from ctypes import CDLL, RTLD_GLOBAL, RTLD_LOCAL
  import os

#  # If we have a library in the current directory, prepend it with './'
#  if exists(name) and '/' not in name: name = './' + name
  libname = find_library (name)
  assert libname is not None, "can't find library '%s'"%name
#  print "debug: '%s' => '%s'"%(name, libname)

  mode = RTLD_GLOBAL if Global==True else RTLD_LOCAL
  return CDLL(libname, mode=mode)
# }}}
