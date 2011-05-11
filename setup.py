#!/usr/bin/env python

from distutils.core import setup, Extension

# PyGeode installation script

#extargs = dict(extra_compile_args=['-std=c99'])

setup (	name="pygeode",
	version="0.6.0",
        author="Peter Hitchcock, Andre Erler, Mike Neish",
        author_email="",
        url="http://sparc01.atmosp.physics.utoronto.ca/pygeode/docs",
	requires=['numpy','matplotlib','progressbar'], # NOTE: distutils doesn't ever check this!
	packages=["pygeode", "pygeode.formats", "pygeode.server", "pygeode.plugins"]
)

"""
        ext_modules=[	Extension('pygeode/libquadrule', ['pygeode/quadrule.c']),
			Extension('pygeode/libtools', ['pygeode/tools.c'], **extargs),
			Extension('pygeode/libinterp', ['pygeode/interp.c'], libraries=['gsl'], **extargs),
			Extension('pygeode/libeof', ['pygeode/eof.c'], **extargs),
			Extension('pygeode/libsvd', ['pygeode/svd.c'], **extargs),
			Extension('pygeode/libtimeaxis', ['pygeode/timeaxis.c'], **extargs),
			Extension('pygeode/formats/libgrib', ['pygeode/formats/grib.c'], **extargs),
			Extension('pygeode/formats/libopendap', ['pygeode/formats/opendap.c'], **extargs),
	],
"""
