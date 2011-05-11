LIBS = grib opendap index
PROGS = cccma4

include rules.mk

cccma4 : cccma4.c index.c
	$(CC) $^ -o $@ $(CFLAGS) -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE


