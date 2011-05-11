build:
	python setup.py build

install:
	# Install the python scripts
	python setup.py install --prefix=$(DESTDIR)/usr/local
	# Compile/Install the shared libraries
	@(cd pygeode; $(MAKE))
	./cp_libs.sh
clean:
	rm -Rf build/
