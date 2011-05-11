#ifndef __INDEX_H

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>


typedef uint8_t byte;
typedef int64_t Offset;

// 8-bit I/O
uint8_t read8 (FILE *f);
void write8 (FILE *f, uint8_t x);

// 16-bit I/O
uint16_t get16 (byte *b);
void put16 (byte *b, uint16_t x);
uint16_t read16 (FILE *f);
void write16 (FILE *f, uint16_t x);

// 24-bit I/O
uint32_t get24 (byte *b);
void put24 (byte *b, uint32_t x);
uint32_t read24 (FILE *f);
void write24 (FILE *f, uint32_t x);

// 32-bit I/O
uint32_t get32 (byte *b);
void put32 (byte *b, uint32_t x);
uint32_t read32 (FILE *f);
void write32 (FILE *f, uint32_t x);

// 40-bit I/O (should handle most reasonable file offsets)
uint64_t get40 (byte *b);
void put40 (byte *b, uint64_t x);
uint64_t read40 (FILE *f);
void write40 (FILE *f, uint64_t x);

// 64-bit I/O
uint64_t get64 (byte *b);
void put64 (byte *b, uint64_t x);
uint64_t read64 (FILE *f);
void write64 (FILE *f, uint64_t x);

// Write a positive integer, using just enough bytes to cover a particular range
void writeX (FILE *f, uint64_t x, uint64_t max);

// Read a positive integer, with a size determined by the maximum value given
uint64_t readX (FILE *f, uint64_t max);


// Function type for printing information
typedef void(PrintFunc)(byte *);

// Functions for reading/writing a coordinate
typedef void(CoordIOFunc)(FILE*, byte**, int, int);

// A pattern for the sequence of offsets for a variable
typedef struct {
  int count;
  Offset *vals;
  int *shifts;
  int *jumps;
} OffsetPattern;

// An index
typedef struct {
  int nvars;  // # of variables
  int varidsize;  // # of bytes in a varid
  byte **varids;
  int vardescsize;  // # of bytes in a variable description
  byte **vardescs;
  int ndims;  // # of dimensions (fixed)
  int *dimidsize;  // # of bytes for each dimid
  CoordIOFunc **read_coord;  // functions to read/write coordinates
  CoordIOFunc **write_coord;
  int **dimlengths;  // lengths along each dimension
  byte ****coords;  // coordinate values

  // Offsets (either explicit list or set of diffs)
  int *noffsets;
  Offset **offsets;  // offsets into the file

//  int *patt_count;
//  Offset *patt_first;
//  Offset **patt_vals;
//  int **patt_shifts;
//  int **patt_jumps;

  OffsetPattern **patterns;
  Offset *first;  // First offset to start applying the pattern

  // Temporary bookkeeping arrays (while constructing the index)
  int *_nrecs;  // number of records read so far (per var)
  int ***_coords;  // coordinate indices
  Offset **_offsets; // offsets (in record order)

  // Auxiliary functions for displaying the data
  PrintFunc *printvar;

} Index;




// Inititialize an index
// (The Index itself must be allocated before calling this routine)
void init_Index (Index *ind);

// set the # of bytes in a varid
void set_vartype (Index *ind, int varidsize, int vardescsize, PrintFunc *pf);

// number of array elements to resize by
// (reduces the number of realloc calls, at the expense of more memory usage)
#define REALLOC_SIZE 100

// Add a dimension - define how many bytes it takes
void set_dimtype (Index *ind, int dimidsize, CoordIOFunc *r, CoordIOFunc *w);

// Free the internal arrays of an Index
// (Doesn't free the Index itself)
void free_Index (Index *ind);

// Add a record to the index
void add_record (Index *ind, Offset offset, byte *varid, byte *vardesc, ...);

// Maximum number of superimposed patterns before we give up
#define MAX_PATTERN_SIZE 10

// Minimum number of offsets to bother writing a pattern
#define MIN_SIZE_FOR_PATTERN 4

// Put the offsets into an array, organized by dimensions
void finalize_Index (Index *ind);

// Save an index to disk
void save_Index (Index *ind, char *filename);

// Read an index from disk
// (the index must be initialized beforehand)
void load_Index (Index *ind, char *filename);

// Get an offset, given the indices along all the dimensions
Offset get_offset (Index *ind, int v, ...);


/*
// Print an index (assuming it's fully initialized)
void print_Index (Index *ind);
*/


#define __INDEX_H
#endif
