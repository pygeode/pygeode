#!/usr/bin/env python

from distutils.core import setup

# PyGeode installation script

setup (	name="pygeode",
	version="0.6.1",
        author="Peter Hitchcock, Andre Erler, Mike Neish",
        author_email="",
        url="http://sparc01.atmosp.physics.utoronto.ca/pygeode/docs",
	requires=['numpy','matplotlib','progressbar'], # NOTE: distutils doesn't ever check this!
        # Note: When building Windows version, pre-compile the libraries
        # in the 'pygeode' subdirectory.
	package_data={'pygeode': ['*.dll'], 'pygeode.formats': ['*.dll']},
	packages=["pygeode", "pygeode.formats", "pygeode.server", "pygeode.plugins"]
)

