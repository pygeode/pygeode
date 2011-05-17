#!/usr/bin/env python

from distutils.core import setup

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

