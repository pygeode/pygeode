
# Platform-specific settings
ifeq ($(MAKE),mingw32-make)
  # assume Windows
  PLATFORM = WINDOWS
  RM = del
  LIBEXT = dll
else
  # assume Linux
  PLATFORM = LINUX
  RM = rm -f
  LIBEXT = so
endif

all: $(PROGS) $(addsuffix .dir, $(SUBDIRS)) $(addprefix lib, $(addsuffix .$(LIBEXT), $(LIBS)))

# Intel
#CC = icc
#FC = ifort
#CFLAGS += -O3 -std=c99 -fPIC -D$(PLATFORM)
#FFLAGS += -O3 -fPIC
# GCC
CC = gcc
FC = gfortran
CFLAGS += -std=c99 -fPIC -g -fbounds-check -Wall -D$(PLATFORM)
FFLAGS += -fPIC -g -fbounds-check -Wall
LDLIBS += -lm
SHARED = -shared -Wl,-soname,$(basename $<)

%: %.c
	$(CC) $(CFLAGS) $(LDLIBS) -o $@ $^

lib%.$(LIBEXT): %.c
	$(CC) $(CFLAGS) $(LDLIBS) $(SHARED) -o $@ $^

lib%.$(LIBEXT): %.f
	$(FC) $(FFLAGS) $(LDLIBS) $(SHARED) -o $@ $^


%.dir:
	@(cd $(basename $@) && $(MAKE))

%.clean:
	@(cd $(basename $@) && $(MAKE) clean)

clean: $(addsuffix .clean, $(SUBDIRS))
	@( $(RM) $(PROGS) *.o *.$(LIBEXT) *.pyc && exit 0 )

# Windows kludges
# (You must grab these libraries from somewhere else beforehand)
ifeq ($(PLATFORM),WINDOWS)
libinterp.$(LIBEXT): interp.c libgsl.dll libgslcblas.dll
libeof.$(LIBEXT): eof.c liblapack.dll libblas.dll
endif
