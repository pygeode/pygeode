test:
	env PYTHONPATH=$(PWD)/../ nosetests
	@(cd issues; $(MAKE) test)

clean:
	rm -f *.pyc *.nc
	@(cd issues; $(MAKE) clean)
