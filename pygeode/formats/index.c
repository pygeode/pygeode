#include "index.h"

/******
  Index generator

  Collects file offsets for record-based file formats, allows random access.

******/

// 8-bit I/O
uint8_t read8 (FILE *f) {
  uint8_t b[1];
  fread (b, 1, 1, f);
  return *b;
}
void write8 (FILE *f, uint8_t x) {
  uint8_t b[1];
  *b = x;
  fwrite (b, 1, 1, f);  
}

// 16-bit I/O
uint16_t get16 (byte *b) {
  return (((uint16_t)(b[0]))<<8) | (((uint16_t)(b[1]))<<0);
}
void put16 (byte *b, uint16_t x) {
  b[1] = x & 0xff; x>>=8;
  b[0] = x & 0xff; x>>=8;
}
uint16_t read16 (FILE *f) {
  byte b[2];
  fread (b, 1, 2, f);
  return get16 (b);
}
void write16 (FILE *f, uint16_t x) {
  byte b[2];
  put16 (b, x);
  fwrite (b, 1, 2, f);
}

// 24-bit I/O
uint32_t get24 (byte *b) {
  return (((uint32_t)(b[0]))<<16) | (((uint32_t)(b[1]))<<8) | (((uint32_t)(b[2]))<<0);
}
void put24 (byte *b, uint32_t x) {
  b[2] = x & 0xff; x>>=8;
  b[1] = x & 0xff; x>>=8;
  b[0] = x & 0xff; x>>=8;
}
uint32_t read24 (FILE *f) {
  byte b[3];
  fread (b, 1, 3, f);
  return get24 (b);
}
void write24 (FILE *f, uint32_t x) {
  byte b[3];
  put24 (b, x);
  fwrite (b, 1, 3, f);
}


// 32-bit I/O
uint32_t get32 (byte *b) {
  return (((uint32_t)(b[0]))<<24) | (((uint32_t)(b[1]))<<16) | (((uint32_t)(b[2]))<<8) | (((uint32_t)(b[3]))<<0);
}
void put32 (byte *b, uint32_t x) {
  b[3] = x & 0xff; x>>=8;
  b[2] = x & 0xff; x>>=8;
  b[1] = x & 0xff; x>>=8;
  b[0] = x & 0xff; x>>=8;
}
uint32_t read32 (FILE *f) {
  byte b[4];
  fread (b, 1, 4, f);
  return get32 (b);
}
void write32 (FILE *f, uint32_t x) {
  byte b[4];
  put32 (b, x);
  fwrite (b, 1, 4, f);
}

// 40-bit I/O (should handle most reasonable file offsets)
uint64_t get40 (byte *b) {
  return (((uint64_t)(b[0]))<<32) | (((uint64_t)(b[1]))<<24) | (((uint64_t)(b[2]))<<16) |
         (((uint64_t)(b[3]))<<8)  | (((uint64_t)(b[4]))<<0);
}
void put40 (byte *b, uint64_t x) {
  b[4] = x & 0xff; x>>=8;
  b[3] = x & 0xff; x>>=8;
  b[2] = x & 0xff; x>>=8;
  b[1] = x & 0xff; x>>=8;
  b[0] = x & 0xff; x>>=8;
}
uint64_t read40 (FILE *f) {
  byte b[5];
  fread (b, 1, 5, f);
  return get40 (b);
}
void write40 (FILE *f, uint64_t x) {
  byte b[5];
  put40 (b, x);
  fwrite (b, 1, 5, f);
}

// 64-bit I/O
uint64_t get64 (byte *b) {
  return (((uint64_t)(b[0]))<<56) | (((uint64_t)(b[1]))<<48) | (((uint64_t)(b[2]))<<40) | 
         (((uint64_t)(b[3]))<<32) | (((uint64_t)(b[4]))<<24) | (((uint64_t)(b[5]))<<16) |
         (((uint64_t)(b[6]))<<8)  | (((uint64_t)(b[7]))<<0);
}
void put64 (byte *b, uint64_t x) {
  b[7] = x & 0xff; x>>=8;
  b[6] = x & 0xff; x>>=8;
  b[5] = x & 0xff; x>>=8;
  b[4] = x & 0xff; x>>=8;
  b[3] = x & 0xff; x>>=8;
  b[2] = x & 0xff; x>>=8;
  b[1] = x & 0xff; x>>=8;
  b[0] = x & 0xff; x>>=8;
}
uint64_t read64 (FILE *f) {
  byte b[8];
  fread (b, 1, 8, f);
  return get64 (b);
}
void write64 (FILE *f, uint64_t x) {
  byte b[8];
  put64 (b, x);
  fwrite (b, 1, 8, f);
}


// Write a positive integer, using just enough bytes to cover a particular range
void writeX (FILE *f, uint64_t x, uint64_t max) {
  assert (x <= max);
  if (max <= 0xFF) return write8 (f, x);
  if (max <= 0xFFFF) return write16 (f, x);
  if (max <= 0xFFFFFF) return write24 (f, x);
  if (max <= 0xFFFFFFFF) return write32 (f, x);
  if (max <= 0xFFFFFFFFFF) return write40 (f, x);
  return write64 (f, x);
}

// Read a positive integer, with a size determined by the maximum value given
uint64_t readX (FILE *f, uint64_t max) {
  if (max <= 0xFF) return read8 (f);
  if (max <= 0xFFFF) return read16 (f);
  if (max <= 0xFFFFFF) return read24 (f);
  if (max <= 0xFFFFFFFF) return read32(f);
  if (max <= 0xFFFFFFFFFF) return read40(f);
  return read64 (f);
}



// Inititialize an index
// (The Index itself must be allocated before calling this routine)
void init_Index (Index *ind) {
  ind->nvars = 0;
  ind->varidsize = 0;
  ind->varids = NULL;
  ind->vardescsize = 0;
  ind->vardescs = NULL;
  ind->ndims = 0;
  ind->dimlengths = NULL;
  ind->dimidsize = NULL;
  ind->coords = NULL;
  ind->noffsets = 0;
  ind->offsets = NULL;
  ind->patterns = NULL;
  ind->first = NULL;

  ind->_nrecs = 0;
  ind->_coords = NULL;
  ind->_offsets = NULL;

  ind->printvar = NULL;
  ind->read_coord = NULL;
  ind->write_coord = NULL;
}

// set the # of bytes in a varid
void set_vartype (Index *ind, int varidsize, int vardescsize, PrintFunc *pf) {
  assert (ind->nvars == 0);
  assert (varidsize > 0);
  assert (vardescsize >= 0);
  ind->varidsize = varidsize;
  ind->vardescsize = vardescsize;
  ind->printvar = pf;
}

// Add a dimension - define how many bytes it takes
void set_dimtype (Index *ind, int dimidsize, CoordIOFunc *r, CoordIOFunc *w) {
  assert (ind->nvars == 0);
  int d = ind->ndims;
  if (d % REALLOC_SIZE == 0) {
    int newsize = d+REALLOC_SIZE;
    ind->dimidsize = realloc(ind->dimidsize, newsize*sizeof(int));
    ind->read_coord = realloc(ind->read_coord, newsize*sizeof(CoordIOFunc*));
    ind->write_coord = realloc(ind->write_coord, newsize*sizeof(CoordIOFunc*));
  }
  assert (dimidsize > 0);
  ind->dimidsize[d] = dimidsize;
  ind->read_coord[d] = r;
  ind->write_coord[d] = w;
  ind->ndims++;
}

// Free the internal arrays of an Index
// (Doesn't free the Index itself)
void free_Index (Index *ind) {
  for (int v = 0; v < ind->nvars; v++) {
    for (int d = 0; d < ind->ndims; d++) {
      byte **coords = ind->coords[v][d];
      // Was this already cleared?
      if (coords == NULL) continue;
      for (int k = 0; k < ind->dimlengths[v][d]; k++) {
        free (ind->coords[v][d][k]);
      }
      // Free the coordinate arrays
      free (coords);
      // Other variables may be sharing this coord array, so set them to NULL
      // (also null out this one too)
      for (int v2 = v; v2 < ind->nvars; v2++) {
        if (ind->coords[v2][d] == coords) ind->coords[v2][d] = NULL;
      }
    }
    free (ind->varids[v]);
    // free the var descriptions
    if (ind->vardescs[v] != NULL) {
      byte * vardesc = ind->vardescs[v];
      free (vardesc);
      for (int v2 = v; v2 < ind->nvars; v2++) {
        if (ind->vardescs[v2] == vardesc) ind->vardescs[v2] = NULL;
      }
    }
    free (ind->dimlengths[v]);
    free (ind->coords[v]);
    // Free either the explicit offsets or the patterns
    if (ind->offsets[v] != NULL) free (ind->offsets[v]);
    else if (ind->patterns[v] != NULL) {  // free the pattern (if isn't not already freed)
      OffsetPattern *pat = ind->patterns[v];
      assert (pat != NULL);
      assert (pat->vals != NULL);
      free (pat->vals);
      assert (pat->jumps != NULL);
      free (pat->jumps);
      assert (pat->shifts != NULL);
      free (pat->shifts);
      free (pat);
      // If any other vars were sharing this pattern, set it to NULL
      for (int v2 = v; v2 < ind->nvars; v2++) {
        if (ind->patterns[v2] == pat) ind->patterns[v2] = NULL;
      }
    }
  }
  if (ind->varids     != NULL) free (ind->varids);
  if (ind->vardescs   != NULL) free (ind->vardescs);
  if (ind->dimlengths != NULL) free (ind->dimlengths);
  if (ind->dimidsize  != NULL) free (ind->dimidsize);
  if (ind->coords     != NULL) free (ind->coords);
  if (ind->noffsets   != NULL) free (ind->noffsets);
  if (ind->offsets    != NULL) free (ind->offsets);
  if (ind->first      != NULL) free (ind->first);

  if (ind->patterns   != NULL) free (ind->patterns);

  if (ind->read_coord != NULL) free (ind->read_coord);
  if (ind->write_coord != NULL) free (ind->write_coord);

  init_Index(ind);  // set everything to NULL / 0
}

// Add a record to the index
void add_record (Index *ind, Offset offset, byte *varid, byte *vardesc, ...) {
  int ndims = ind->ndims;
  int varidsize = ind->varidsize;
  int vardescsize = ind->vardescsize;
  byte *dimids[ndims];
  // Get the dimension ids
  va_list ap;
  va_start (ap, vardesc);
  for (int i = 0; i < ndims; i++) dimids[i] = va_arg (ap, byte*);
  va_end (ap);

  // Check if we have this variable
  int v = 0;
  for (v = 0; v < ind->nvars; v++) {
    if (memcmp(varid, ind->varids[v], varidsize) == 0)
      if (memcmp(vardesc, ind->vardescs[v], vardescsize) == 0) break;
  }
  // Add a new variable?
  if (v == ind->nvars) {
    // Do we need to allocate more space?
    if (v % REALLOC_SIZE == 0) {
//      printf ("realloc\n");
      int newsize = v+REALLOC_SIZE;
      ind->varids = realloc(ind->varids, newsize*sizeof(byte*));
      ind->vardescs = realloc(ind->vardescs, newsize*sizeof(byte*));
      ind->dimlengths = realloc(ind->dimlengths, newsize*sizeof(int*));
      ind->coords = realloc(ind->coords, newsize*sizeof(byte***));

      ind->_nrecs = realloc(ind->_nrecs, newsize*sizeof(int));
      ind->_coords = realloc(ind->_coords, newsize*sizeof(int**));
      ind->_offsets = realloc(ind->_offsets, newsize*sizeof(Offset*));
    }
    ind->varids[v] = malloc(varidsize);
    ind->vardescs[v] = malloc(vardescsize);
    memcpy (ind->varids[v], varid, varidsize);
    memcpy (ind->vardescs[v], vardesc, vardescsize);
    ind->dimlengths[v] = malloc(ndims*sizeof(int));
    for (int i = 0; i < ndims; i++) ind->dimlengths[v][i] = 0;
    ind->coords[v] = malloc(ndims*sizeof(byte**));
    for (int i = 0; i < ndims; i++) ind->coords[v][i] = NULL;

    ind->_nrecs[v] = 0;
    ind->_coords[v] = NULL;
    ind->_offsets[v] = NULL;

    ind->nvars++;
  }

  // Check the dimensions
  int ids[ndims];
  for (int i = 0; i < ndims; i++) {
    // Checking dimension i
    int j;
    for (j = 0; j < ind->dimlengths[v][i]; j++) {
      if (memcmp(dimids[i], ind->coords[v][i][j], ind->dimidsize[i]) == 0) break;
    }
    // Add new dimension?
    if (j == ind->dimlengths[v][i]) {
      // Do we need to allocate more space?
      if (j % REALLOC_SIZE == 0) {
        int newsize = j+REALLOC_SIZE;
        ind->coords[v][i] = realloc(ind->coords[v][i], newsize*sizeof(byte*));
      }
      ind->coords[v][i][j] = malloc(ind->dimidsize[i]);
      memcpy (ind->coords[v][i][j], dimids[i], ind->dimidsize[i]);
      ind->dimlengths[v][i]++;
    }
    // Keep track of this id (we'll use it later)
    ids[i] = j;
  }

  // Keep track of the offsets

  int rec = ind->_nrecs[v];
  // Do we need to allocate more space?
  if (rec % REALLOC_SIZE == 0) {
    int newsize = rec + REALLOC_SIZE;
    ind->_coords[v] = realloc(ind->_coords[v], newsize*sizeof(int*));
    ind->_offsets[v] = realloc(ind->_offsets[v], newsize*sizeof(Offset));
  }
  ind->_coords[v][rec] = malloc(ndims*sizeof(int));
  for (int i = 0; i < ndims; i++) ind->_coords[v][rec][i] = ids[i];
  ind->_offsets[v][rec] = offset;

  ind->_nrecs[v]++;
}


// Put the offsets into an array, organized by dimensions
void finalize_Index (Index *ind) {
  int nvars = ind->nvars;
  int ndims = ind->ndims;
  assert (ind->offsets == NULL);
  ind->noffsets = malloc(nvars*sizeof(int));
  ind->offsets = malloc(nvars*sizeof(Offset*));

  ind->patterns = malloc(nvars*sizeof(OffsetPattern*));
  ind->first = malloc(nvars*sizeof(Offset));

  for (int v = 0; v < nvars; v++) {
    // Determine total allocation size for this var
    int size = 1;
    for (int d = 0; d < ndims; d++) size *= ind->dimlengths[v][d];
    ind->offsets[v] = malloc(size * sizeof(Offset));
    // Guard against missing records (offset of -1 means that record isn't available)
    for (int j = 0; j < size; j++) ind->offsets[v][j] = -1;

    // Fill in the offsets!
    for (int r = 0; r < ind->_nrecs[v]; r++) {
      // Figure out the index of where to put this offset in the array
      int i = 0;
      for (int d = 0; d < ndims; d++) i = i * ind->dimlengths[v][d] + ind->_coords[v][r][d];
      assert (ind->_offsets[v][r] >= 0);  // make sure all expected records actually exist
                                          // will fail if, for example, there's a level missing for one timestep
      ind->offsets[v][i] = ind->_offsets[v][r];
    }
    ind->noffsets[v] = size;

/*
    // Initialize the pattern arrays
    ind->patterns[v].count = -1;
    ind->patterns[v].first = -1;
    ind->patterns[v].vals = NULL;
    ind->patterns[v].shifts = NULL;
    ind->patterns[v].jumps = NULL;
*/

    ind->patterns[v] = NULL;

    // Try representing the offsets as a few jumps
    Offset *x = malloc(size*sizeof(Offset));
    do {
      // Abort if there's only a few offsets
      if (size < MIN_SIZE_FOR_PATTERN) break;

      // Calculate diffs
      int first = ind->offsets[v][0];
//      printf ("first: %d\n", first);
      int diffsize = size-1;
      for (int i = 0; i < diffsize; i++) {
        x[i] = ind->offsets[v][i+1] - ind->offsets[v][i];
//        printf ("delta %d\n", (int)x[i]);
      }
      // Abort if we have any negative diffs
      {
        int neg = 0;
        for (int i = 0; i < diffsize; i++) if (x[i] < 0) neg = 1;
        if (neg == 1) break;
      }

      // Loop until there are no non-zero elements left
      int npatt = 0;
      Offset vals[MAX_PATTERN_SIZE];
      int shifts[MAX_PATTERN_SIZE];
      int jumps[MAX_PATTERN_SIZE];

      while (npatt < MAX_PATTERN_SIZE) {

        // Find a minimum (non-zero) element
        int imin;
        // Start by finding *any* non-zero element
        for (imin = 0; imin < diffsize; imin++) if (x[imin] > 0) break;
        // If all elements are zero, then we're done!
        if (imin == diffsize) break;
        // Now find the smallest one
        for (int i = 0; i < diffsize; i++) if (x[i] > 0 && x[i] < x[imin]) imin = i;
        Offset min = x[imin];

        // Find the shortest jump interval that satisfies the min
        int jump, shift;
        for (jump = 1; jump <= diffsize; jump++) {
          int valid = 1;
          shift = imin % jump;
          for (int k = shift; k < diffsize; k+=jump) if (x[k] == 0) valid = 0;
          if (valid == 0) continue;
          // Apply this jump interval
//          printf ("+%d *%d: %d\n", imin, jump, (int)min);
          for (int k = shift; k < diffsize; k+=jump) x[k] -= min;
          break;  // found a valid jump
        }
        // Store this pattern
        vals[npatt] = min;
        shifts[npatt] = shift;
        jumps[npatt] = jump;

        npatt++;
      }

//      printf ("npatt: %d\n", npatt);
      // Abort if there was no real pattern found
      assert (npatt <= diffsize);
      if (npatt == diffsize || npatt == MAX_PATTERN_SIZE) {
//        printf ("no pattern found\n");
        break;
      }

      for (int i = 0; i < diffsize; i++) assert (x[i] == 0);

      // Store this information
      ind->first[v] = first;
      ind->patterns[v] = malloc(sizeof(OffsetPattern));
      OffsetPattern *pat = ind->patterns[v];
      pat->count = npatt;
      pat->vals = malloc(npatt*sizeof(Offset));
      pat->jumps = malloc(npatt*sizeof(int));
      pat->shifts = malloc(npatt*sizeof(int));
      for (int i = 0; i < npatt; i++) {
        pat->vals[i] = vals[i];
        pat->jumps[i] = jumps[i];
        pat->shifts[i] = shifts[i];
      }

      // We don't need the explicit offset array anymore
      free(ind->offsets[v]);
      ind->offsets[v] = NULL;

    } while (0);
    free (x);

  }

  // Free the bookkeeping arrays
  for (int v = 0; v < nvars; v++) {
    for (int r = 0; r < ind->_nrecs[v]; r++) {
      free (ind->_coords[v][r]);
    }
    free (ind->_coords[v]);
    free (ind->_offsets[v]);
  }
  free (ind->_nrecs);
  free (ind->_coords);
  free (ind->_offsets);


  // Check coords for redundancy
  // (Only need one copy of each unique coord array)
  for (int d = 0; d < ind->ndims; d++) {
//    printf ("uniquify: looking at dimension %d\n", d);
    int nunique = 0;
    byte **coords[ind->nvars];
    int lengths[ind->nvars];
    for (int i = 0; i < ind->nvars; i++) coords[i] = NULL;

    // Iterate through all coord arrays for this dimension
    for (int v = 0; v < ind->nvars; v++) {
      // Check if we have this one
      int match = 0;
      int imatch = -1;  // index of the match
      for (int i = 0; i < nunique; i++) {
        // Check the lengths
        if (ind->dimlengths[v][d] != lengths[i]) {match = 0; continue; }

        // Check the elements
        match = 1;
        for (int k = 0; k < ind->dimlengths[v][d]; k++) {
          if (memcmp (ind->coords[v][d][k], coords[i][k], ind->dimidsize[d]) !=0){
            match = 0;
            break;
          }
        }
        // found a match, no need to check other unique coords
        if (match == 1) {imatch = i; break; }
      }

      // If we have a match to an existing unique coord array, then use that instead
      if (match == 1) {
//        printf ("match on var %d to unique array %d\n", v, imatch);
        assert (ind->dimlengths[v][d] == lengths[imatch]); // sanity check
        for (int k = 0; k < ind->dimlengths[v][d]; k++) free (ind->coords[v][d][k]);
        free (ind->coords[v][d]);
        ind->coords[v][d] = coords[imatch];
      }
      else {
        // Add a new unique coord array
//        printf ("adding new unique array %d\n", nunique);
        coords[nunique] = ind->coords[v][d];
        lengths[nunique] = ind->dimlengths[v][d];
        nunique++;
      }
    }

  }

  // Check var descriptors for redundancy
  for (int v = 0; v < ind->nvars; v++) {
    for (int v2 = 0; v2 < v; v2++) {
      if (memcmp(ind->vardescs[v], ind->vardescs[v2], ind->vardescsize) == 0) {
//        printf ("same vardesc found\n");
        free (ind->vardescs[v]);
        ind->vardescs[v] = ind->vardescs[v2];
        break;  // found a replacemet for this var descriptor
      }
    }
  }

  // Check patterns for redundancy
  for (int v = 0; v < ind->nvars; v++) {
    if (ind->offsets[v] != NULL) continue;
    int v2;
    for (v2 = 0; v2 < v; v2++) {
      if (ind->offsets[v2] != NULL) continue;
      OffsetPattern *pat = ind->patterns[v];
      OffsetPattern *pat2 = ind->patterns[v2];
      if (pat->count != pat2->count) continue;
      int i;
      for (i = 0; i < pat->count; i++) {
        if (pat->vals[i] != pat2->vals[i]) break;
        if (pat->shifts[i] != pat2->shifts[i]) break;
        if (pat->jumps[i] != pat2->jumps[i]) break;
      }
      if (i == pat->count) break;  // match?
    }
    // no unique match?
    if (v2 == v) continue;
    printf ("varid %d has same offset pattern as varid %d\n", v, v2);
    OffsetPattern *pat = ind->patterns[v];
    free (pat->vals);
    free (pat->shifts);
    free (pat->jumps);
    free (pat);
    ind->patterns[v] = ind->patterns[v2];
  }

}

// Save an index to disk
void save_Index (Index *ind, char *filename) {
  FILE *f = fopen(filename, "wb");
  assert (f != NULL);

  char header[] = "PyGeode Index v1";
  fwrite (header, 1, strlen(header), f);

  // number of dimensions
  write32(f, ind->ndims);

  // store information on unique coordinate arrays
  byte **coords[ind->ndims][ind->nvars];
  int nunique_coords[ind->ndims];
  for (int d = 0; d < ind->ndims; d++)
    for (int v = 0; v < ind->nvars; v++)
      coords[d][v] = NULL;  // just in case - I'd rather get a segfault if there's a bug than have a garbage address that might 'work'

  // for each dimension,
  for (int d = 0; d < ind->ndims; d++) {
    // size of a coordinate value
    write32(f, ind->dimidsize[d]);

    // number of unique coordinate arrays
    int nunique = 0;
    int lengths[ind->nvars];
    for (int v = 0; v < ind->nvars; v++) {
      int i;
      for (i = 0; i < nunique; i++) if (ind->coords[v][d] == coords[d][i]) break;
      // found another unique array?
      if (i == nunique) {
        coords[d][nunique] = ind->coords[v][d];
        lengths[nunique] = ind->dimlengths[v][d];
        nunique++;
      }
    }

    // Write out this information
    write32 (f, nunique);
    for (int i = 0; i < nunique; i++) {
      write32 (f, lengths[i]);
      // Write raw values?
      if (ind->write_coord[d] == NULL || lengths[i] == 1) {
        for (int k = 0; k < lengths[i]; k++) {
          fwrite (coords[d][i][k], 1, ind->dimidsize[d], f);
        }
      }
      // Write encoded values?
      else ind->write_coord[d](f, coords[d][i], lengths[i], ind->dimidsize[d]);
    }

    nunique_coords[d] = nunique;
  }

  // size of a var description
  write32(f, ind->vardescsize);

  byte *vardescs[ind->nvars];
  int nunique_vardescs;

  // Write out the unique variable descriptions
  {
    int nunique = 0;
    for (int v = 0; v < ind->nvars; v++) {
      int i;
      for (i = 0; i < nunique; i++) if (ind->vardescs[v] == vardescs[i]) break;
      // found another unique description?
      if (i == nunique) {
        vardescs[nunique] = ind->vardescs[v];
        nunique++;
      }
    }
//    printf ("# unique descriptions: %d\n", nunique);
    write32(f, nunique);
    for (int i = 0; i < nunique; i++) {
      fwrite (vardescs[i], 1, ind->vardescsize, f);
    }
    nunique_vardescs = nunique;
  }

  // Maximum offset
  Offset max_offset = 0;
  for (int v = 0; v < ind->nvars; v++) {
    if (ind->offsets[v] == NULL) {
      if (ind->first[v] > max_offset) max_offset = ind->first[v];
      OffsetPattern *pat = ind->patterns[v];
      for (int i = 0; i < pat->count; i++) {
       if (pat->vals[i] > max_offset) max_offset = pat->vals[i];
      }
    }
    else {
      for (int i = 0; i < ind->noffsets[v]; i++) {
        if (ind->offsets[v][i] > max_offset) max_offset = ind->offsets[v][i];
      }
    }
  }

  write64(f, max_offset);

  // Unique patterns
  OffsetPattern *patterns[ind->nvars];
  int nunique_patterns;
  {
    int nunique = 0;
    for (int v = 0; v < ind->nvars; v++) {
      if (ind->offsets[v] != NULL) continue;
      int i;
      for (i = 0; i < nunique; i++) if (ind->patterns[v] == patterns[i]) break;
      if (i == nunique) {
        patterns[i] = ind->patterns[v];
        nunique++;
      }
    }

    write32 (f, nunique);
    for (int i = 0; i < nunique; i++) {
      writeX (f, patterns[i]->count, 254);  //  assume we'll never have more than 254 patterns
      for (int j = 0; j < patterns[i]->count; j++) {
        writeX (f, patterns[i]->vals[j], max_offset);
//        write64 (f, patterns[i]->vals[j]);
        write32 (f, patterns[i]->jumps[j]);
        writeX (f, patterns[i]->shifts[j], patterns[i]->jumps[j]);
      }

    }

    nunique_patterns = nunique;
  }

  printf ("%d unique patterns:\n", nunique_patterns);
  for (int i = 0; i < nunique_patterns; i++) {
    printf ("pattern %d:\n", i);
    for (int j = 0; j < patterns[i]->count; j++)
      printf ("%d: %8d %8d *%lld\n", j, patterns[i]->shifts[j], patterns[i]->jumps[j], (long long)(patterns[i]->vals[j]));
    printf ("\n");
  }

  // size of a var id
  write32(f, ind->varidsize);

  // number of vars
  write32(f, ind->nvars);

  // write out all variable descriptors and offset info
  for (int v = 0; v < ind->nvars; v++) {
    fwrite (ind->varids[v], 1, ind->varidsize, f);
    // var description #
    int i;
    for (i = 0; i < nunique_vardescs; i++) {
      if (ind->vardescs[v] == vardescs[i]) {
        writeX (f, i, nunique_vardescs);
        break;
      }
    }
    assert (i < nunique_vardescs);
    // dimension info
    for (int d = 0; d < ind->ndims; d++) {
      for (i = 0; i < nunique_coords[d]; i++) {
        if (ind->coords[v][d] == coords[d][i]) {
          writeX (f, i, nunique_coords[d]);
          break;
        }
      }
      assert (i < nunique_coords[d]);
    }
    // write the offsets
    // (either explicit or pattern based)
    int size = ind->noffsets[v];
    if (ind->offsets[v] != NULL) {
//      printf ("using explicit offsets\n");
//      writeX (f, size, size);
      write8 (f, 0xFF);
      for (int i = 0; i < size; i++) writeX (f, ind->offsets[v][i], max_offset);
    }
    else {
      OffsetPattern *pat = ind->patterns[v];
//      writeX (f, 0, size);
      for (int i = 0; i < nunique_patterns; i++) {
        if (pat == patterns[i]) {
          write8 (f, i);
          break;
        }
      }
      writeX (f, ind->first[v], max_offset);
//      writeX (f, pat->count, 255);  //  assume we'll never have more than 255 patterns
//      for (int i = 0; i < pat->count; i++) {
//        writeX (f, pat->vals[i], max_diff);
//        writeX (f, pat->jumps[i], size);
//        writeX (f, pat->shifts[i], pat->jumps[i]);
//      }
    }
  }


  fclose(f);
}


// Read an index from disk
// (the index must be initialized beforehand)
void load_Index (Index *ind, char *filename) {
  FILE *f = fopen(filename, "rb");
  assert (f != NULL);

  char header[] = "PyGeode Index v1";
  char header_check[strlen(header)+1];
  fread (header_check, 1, strlen(header), f);
  header_check[strlen(header)] = 0;
  assert (strcmp(header_check,header) == 0);

  // number of dimensions
  int ndims = read32 (f);
  assert (ndims == ind->ndims);

  // Unique coordinate arrays
  byte ***coords[ndims];
  int nunique_coords[ndims];
  int *lengths[ndims];

  // for each dimension,
  for (int d = 0; d < ndims; d++) {
    // size of a coordinate value
    int dimidsize = read32(f);
    assert (dimidsize == ind->dimidsize[d]);
    assert (dimidsize > 0);
    int nunique = read32(f);
    assert (nunique >= 0);
    coords[d] = malloc(nunique*sizeof(byte**));
    lengths[d] = malloc(nunique*sizeof(int));

    for (int i = 0; i < nunique; i++) {
      int length = read32(f);
      assert (length > 0);
      coords[d][i] = malloc(length * sizeof(byte*));
      lengths[d][i] = length;
      for (int k = 0; k < length; k++) {
        coords[d][i][k] = malloc(dimidsize);
      }
      // Read raw values?
      if (ind->read_coord[d] == NULL || length == 1) {
        for (int k = 0; k < length; k++) {
          fread (coords[d][i][k], 1, dimidsize, f);
        }
      }
      // Read encoded values?
      else ind->read_coord[d](f, coords[d][i], length, dimidsize);
    }

    nunique_coords[d] = nunique;
  }

  // size of a var description
  int vardescsize = read32(f);
  assert (vardescsize == ind->vardescsize);

  // Unique var descriptions
  int nunique_vardescs = read32(f);
  byte *vardescs[nunique_vardescs];
  {
    int nunique = nunique_vardescs;
    assert (nunique > 0);
    for (int i = 0; i < nunique; i++) {
      vardescs[i] = malloc(vardescsize);
      fread (vardescs[i], 1, vardescsize, f);
    }
  }


  // Maximum offset
  Offset max_offset = read64(f);

  // offset patterns
  int nunique_patterns = read32 (f);
  OffsetPattern *patterns[nunique_patterns];
  {
    int nunique = nunique_patterns;
    for (int i = 0; i < nunique; i++) {
      patterns[i] = malloc(sizeof(OffsetPattern));
      patterns[i]->count = readX (f, 255);  //  assume we'll never have more than 255 patterns
      int count = patterns[i]->count;
      patterns[i]->vals = malloc(count*sizeof(Offset));
      patterns[i]->jumps = malloc(count*sizeof(int));
      patterns[i]->shifts = malloc(count*sizeof(int));
      for (int j = 0; j < count; j++) {
        patterns[i]->vals[j] = readX (f, max_offset);
//        patterns[i]->vals[j] = read64 (f);
        patterns[i]->jumps[j] = read32 (f);
        patterns[i]->shifts[j] = readX (f, patterns[i]->jumps[j]);
      }
    }
  }

  // size of a var id
  int varidsize = read32(f);
  assert (varidsize == ind->varidsize);

  // number of vars
  int nvars = read32(f);
  ind->nvars = nvars;
  assert (nvars > 0);


  ind->varids = malloc(nvars*sizeof(byte*));
  ind->vardescs = malloc(nvars*sizeof(byte*));
  ind->dimlengths = malloc(nvars*sizeof(int*));
  ind->coords = malloc(nvars*sizeof(byte***));
  ind->noffsets = malloc(nvars*sizeof(int));
  ind->offsets = malloc(nvars*sizeof(Offset*));
  ind->patterns = malloc(nvars*sizeof(OffsetPattern*));
  ind->first = malloc(nvars*sizeof(Offset));

  // read info on each var
  for (int v = 0; v < nvars; v++) {
    // Var id
    ind->varids[v] = malloc(varidsize);
    fread(ind->varids[v], 1, varidsize, f);
    // Var descriptor
    int i = readX(f, nunique_vardescs);    
    ind->vardescs[v] = vardescs[i];

    // coordinate arrays
    ind->coords[v] = malloc(ndims*sizeof(byte**));
    ind->dimlengths[v] = malloc(ndims*sizeof(int));
    for (int d = 0; d < ndims; d++) {
      int i = readX(f, nunique_coords[d]);
      ind->coords[v][d] = coords[d][i];
      ind->dimlengths[v][d] = lengths[d][i];
    }

    // offsets
    // (either explicit or pattern based)
    int size = 1;
    for (int d = 0; d < ndims; d++) size *= ind->dimlengths[v][d];
    ind->noffsets[v] = size;
    int ipat = read8(f);
    assert (ipat >= 0);
    // no offset pattern? (explicit offsets)
    if (ipat == 0xFF) {
      ind->offsets[v] = malloc(size * sizeof(Offset));
      // Don't need patterns for this var
      ind->patterns[v] = NULL;
      for (int i = 0; i < size; i++) ind->offsets[v][i] = readX(f, max_offset);
    }
    // offset pattern?
    else {
      assert (ipat < nunique_patterns);
      ind->offsets[v] = NULL;
      ind->first[v] = readX(f, max_offset);
      ind->patterns[v] = patterns[ipat];
/*
      ind->patterns[v] = malloc(sizeof(OffsetPattern));
      OffsetPattern *pat = ind->patterns[v];
      pat->count = readX (f, 255);  // assume we'll never have more than 255 patterns
      pat->vals = malloc(pat->count*sizeof(Offset));
      pat->shifts = malloc(pat->count*sizeof(int));
      pat->jumps = malloc(pat->count*sizeof(int));
      for (int i = 0; i < pat->count; i++) {
        pat->vals[i] = readX (f, max_diff);
        pat->jumps[i] = readX (f, size);
        pat->shifts[i] = readX (f, pat->jumps[i]);
      }
*/

    }


  }

  // Free local arrays
  for (int d = 0; d < ndims; d++) {
    free (coords[d]);
    free (lengths[d]);
  }
  fclose (f);
}

// Get an offset, given the indices along all the dimensions
Offset get_offset (Index *ind, int v, ...) {
  va_list ap;
  va_start (ap, v);

  assert (v >= 0);
  assert (v < ind->nvars);

  int i = 0;
  for (int d = 0; d < ind->ndims; d++) {
    int x = va_arg(ap, int);
    assert (x >= 0);
    assert (x < ind->dimlengths[v][d]);
    i = i * ind->dimlengths[v][d] + x;
  }
  va_end (ap);

//  printf ("i: %d\n", i);

  // Do we have explicit offsets?
  if (ind->offsets[v] != NULL) return ind->offsets[v][i];

  OffsetPattern *pat = ind->patterns[v];

  // Otherwise, apply the patterns
  assert (pat->count >= 0);
  assert (pat->vals != NULL);
  assert (pat->jumps != NULL);
  assert (pat->shifts != NULL);

  Offset offset = ind->first[v];

//  printf ("%lld...\n", (long long)offset);

  for (int j = 0; j < pat->count; j++) {
    Offset val = pat->vals[j];
    int jump = pat->jumps[j];
    int shift = pat->shifts[j];

//    printf ("+ %lld * ( (%d - %d - 1) / %d + 1)\n", (long long)val, i, shift, jump);
    if (i - 1 - shift < 0) continue;
    offset += val * ( (i - 1 - shift) / jump + 1);  // i-1 because we want index into diff array
                                                    // +1 because we want # of landing points, not # of jumps
  }

  return offset;
}

/*
// Print an index (assuming it's fully initialized)
void print_Index (Index *ind) {
  for (int v = 0; v < ind->nvars; v++) {
    if (ind->printvar != NULL) ind->printvar(ind->varids[v]);
    else {
      printf ("[");
      for (int j = 0; j < ind->varidsize; j++) printf ("%d ", ind->varids[v][j]);
      printf ("]");
    }
    printf (":\n");
    for (int d = 0; d < ind->ndims; d++) {
      for (int j = 0; j < ind->dimlengths[v][d]; j++) {
        if (ind->printcoord[d] != NULL)  ind->printcoord[d](ind->coords[v][d][j]);
        else {
          printf ("    [");
          for (int k = 0; k < ind->dimidsize[d]; k++) printf ("%d ", ind->coords[v][d][j][k]);
          printf ("] ");
        }
      }
      printf ("\n");
    }
  }
}
*/

/*
void print_my_var (byte *b) {
  printf ("<%d:%d:%d>", b[0], b[1], b[2]);
}

void setup_index (Index *ind) {
  init_Index (ind);
  set_vartype (ind, 3, 1, print_my_var);
  set_dimtype (ind, 2, NULL);
  set_dimtype (ind, 4, NULL);
}

// test it out!
int main (int argc, char *argv[]) {
  int ndims = 2;
  byte A1[] = {3,4};
  byte A2[] = {5,6};
  byte A1x[] = {3,4};
  byte A2x[] = {5,7};
  byte B1[] = {1,2,3,4};
  byte B2[] = {1,3,5,7};
  byte B3[] = {2,4,6,8};
  byte varid1[] = {3,1,4};
  byte varid2[] = {2,7,1};
  byte desc[] = {100};
  byte desc2[] = {101};

  Index ind;

  setup_index (&ind);

  add_record (&ind, 10, varid1, desc, A1, B1);
  add_record (&ind, 20, varid1, desc, A1, B2);
  add_record (&ind, 35, varid1, desc, A1, B3);
  add_record (&ind, 110, varid1, desc, A2, B1);
  add_record (&ind, 120, varid1, desc, A2, B2);
  add_record (&ind, 135, varid1, desc, A2, B3);

  add_record (&ind, 210, varid2, desc, A1x, B1);
  add_record (&ind, 220, varid2, desc, A1x, B2);
  add_record (&ind, 235, varid2, desc, A1x, B3);
  add_record (&ind, 310, varid2, desc, A2x, B1);
  add_record (&ind, 320, varid2, desc, A2x, B2);
  add_record (&ind, 335, varid2, desc, A2x, B3);

  finalize_Index (&ind);

  print_Index (&ind);
  printf ("%lld\n", (long long)get_offset (&ind, 0, 0, 0));
  printf ("%lld\n", (long long)get_offset (&ind, 0, 0, 1));
  printf ("%lld\n", (long long)get_offset (&ind, 0, 0, 2));
  printf ("%lld\n", (long long)get_offset (&ind, 0, 1, 0));
  printf ("%lld\n", (long long)get_offset (&ind, 0, 1, 1));
  printf ("%lld\n", (long long)get_offset (&ind, 0, 1, 2));

  save_Index(&ind, "myindex");
  free_Index (&ind);
  setup_index (&ind);
  load_Index(&ind, "myindex");
  free_Index (&ind);
}
*/
