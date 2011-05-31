buildlibs:
	@(cd pygeode; $(MAKE))

build:
	python setup.py build

install: buildlibs
	# Install the python scripts
	python setup.py install --prefix=$(DESTDIR)/usr/local
	# Install the shared libraries (compiled in buildlibs rule)
	./cp_libs.sh
clean:
	rm -Rf build/
	@(cd pygeode; $(MAKE) clean)
