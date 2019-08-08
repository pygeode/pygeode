#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel lapack-devel gsl-devel netcdf

# Compile wheels
for PYBIN in /opt/python/{cp27,cp3[56]}*/bin; do
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel --no-deps /io/ -w wheelhouse/
done
for PYBIN in /opt/python/cp3[7-9]*/bin; do
    "${PYBIN}/pip" install -r /io/dev-requirements-3.7.txt
    "${PYBIN}/pip" wheel --no-deps /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
#for PYBIN in /opt/python/*/bin/; do
#    "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
#    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
#done

