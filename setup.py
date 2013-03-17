#!/usr/bin/env python

from distutils.core import setup, Extension

interpcore = Extension ('pygeode.interpcore', sources=['pygeode/interp.c'], libraries=['gsl','gslcblas'])
timeaxiscore = Extension ('pygeode.timeaxiscore', sources=['pygeode/timeaxis.c'], extra_compile_args=['-std=c99'])
quadrulepy = Extension ('pygeode.quadrulepy', sources=['pygeode/quadrule.c','pygeode/quadrulepy.c'])
toolscore = Extension ('pygeode.toolscore', sources=['pygeode/tools.c'], extra_compile_args=['-std=c99'])
svdcore = Extension ('pygeode.svdcore', sources=['pygeode/svd.c'], extra_compile_args=['-std=c99'])

# PyGeode installation script

setup (	name="pygeode",
	version="0.7.0",
        author="Peter Hitchcock, Andre Erler, Mike Neish",
        author_email="",
        url="http://sparc01.atmosp.physics.utoronto.ca/pygeode/docs",
	requires=['numpy','matplotlib','progressbar'], # NOTE: distutils doesn't ever check this!
        # Note: When building Windows version, pre-compile the libraries
        # in the 'pygeode' subdirectory.
	package_data={'pygeode': ['*.dll'], 'pygeode.formats': ['*.dll']},
	packages=["pygeode", "pygeode.formats", "pygeode.server", "pygeode.plugins"],
	ext_modules=[interpcore, timeaxiscore, quadrulepy, toolscore, svdcore]
)

