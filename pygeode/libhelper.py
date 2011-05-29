# miscellaneous stuff that's in development

# Get a library name
# (extends ctypes.util.find_library to include PyGeode-specific paths)
def find_library (name):
  from ctypes.util import find_library
  from glob import glob
  from os import path
  import pygeode

  if '/' in name:
    dir, name = name.rsplit('/',1)
  else:
    dir = ''

  # Search in PyGeode directory
  for libpath in pygeode.__path__:
    libnames = glob(path.join(libpath, dir, 'lib'+name+'.so'))
    if len(libnames) > 0: return libnames[0]
    libnames = glob(path.join(libpath, dir, 'lib'+name+'.dll'))
    if len(libnames) > 0: return libnames[0]

  # Search Linux paths (for installed version)
  libnames = glob(path.join('usr', 'lib', 'pygeode', dir, 'lib'+name+'.so'))
  if len(libnames) > 0: return libnames[0]
  libnames = glob(path.join('usr', 'local', 'lib', dir, 'pygeode', 'lib'+name+'.so'))
  if len(libnames) > 0: return libnames[0]

  # Search in the default system path
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
