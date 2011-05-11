all: $(PROGS) $(addsuffix .dir, $(SUBDIRS)) $(addprefix lib, $(addsuffix .so, $(LIBS)))

CC = gcc
FC = gfortran
CFLAGS += -std=c99 -fPIC -g -fbounds-check -fopenmp
FFLAGS += -fPIC -g -fbounds-check -fopenmp
LDLIBS += -lm

%: %.c
	$(CC) $(CFLAGS) $(LDLIBS) -o $@ $^

lib%.so: %.c
	$(CC) $(CFLAGS) $(LDLIBS) -shared -Wl,-soname,$(basename $^) -o $@ $^

lib%.so: %.f
	$(FC) $(FFLAGS) $(LDLIBS) -shared -Wl,-soname,$(basename $^) -o $@ $^


%.dir:
	@(cd $(basename $@); $(MAKE);)

%.clean:
	@(cd $(basename $@); $(MAKE) clean;)

clean: $(addsuffix .clean, $(SUBDIRS))
	rm -f $(PROGS) *.o *.so *.pyc


