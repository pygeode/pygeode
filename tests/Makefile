test:
	env PYTHONPATH=$(PWD)/../ nosetests
	@(cd issues; $(MAKE) test)

test3:
	env PYTHONPATH=$(PWD)/../ nosetests3
	@(cd issues; $(MAKE) test3)

clean:
	rm -f *.pyc *.nc
	@(cd issues; $(MAKE) clean)
