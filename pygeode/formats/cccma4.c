#include <stdio.h>
#include "index.h"


// Integration of general binary record indexer with cccma files


//typedef byte HEADER[64];
typedef struct {
  byte kind[8];
  byte time[8];
  byte name[8];
  byte level[8];
  byte ilg[8];
  byte ilat[8];
  byte khem[8];
  byte pack[8];
} HEADER;

typedef byte TIMESTEP[8];
typedef byte LEVEL[8];
typedef byte VARID[8];
typedef struct {
  byte kind[8];
  byte ilg[8];
  byte ilat[8];
  byte khem[8];
  byte superlabel[80];
} VARDESC;

void print_name (byte *name) {
  printf ("%c%c%c%c", name[0], name[1], name[2], name[3]);
}

void print_time (byte *time) {
  int64_t x = get64(time);
  printf ("%10lld", (long long)x);
}

void print_level (byte *level) {
  int64_t x = get64(level);
  printf ("%10lld", (long long)x);
}

// Write out timesteps in an efficient way
void write_timesteps (FILE *f, byte **b, int n, int fieldsize) {
  assert (fieldsize == 8);
  assert (n > 1);
  // Decode the timesteps
  int64_t t[n];
  for (int i = 0; i < n; i++) t[i] = get64(b[i]);
  // Get the deltas
  int64_t dt[n-1];
  for (int i = 0; i < n-1; i++) dt[i] = t[i+1] - t[i];
  int64_t dt0 = dt[0];
  int i;
  for (i = 0; i < n-1; i++) if (dt[i] != dt0) break;
  // Non-uniform timesteps?
  if (i < n-1) {
    printf ("writing non-uniform timesteps\n");
    fputc (0x00, f);
    for (int i = 0; i < n; i++) fwrite (b[i], 1, 8, f);
  }
  // Uniform timesteps?
  else {
    printf ("writing uniform timesteps\n");
    fputc (0x01, f);
    printf ("t[0]: %lld  dt0: %lld\n", (long long)t[0], (long long)dt0);
    write64(f,t[0]);
    write64(f,dt0);
  }
}
// Read timesteps
void read_timesteps (FILE *f, byte **b, int n, int fieldsize) {
  assert (fieldsize == 8);
  assert (n > 1);
  if (n == 0) return;
  // Non-uniform timesteps?
  if (fgetc(f) == 0x00) {
    printf ("reading non-uniform timesteps\n");
    for (int i = 0; i < n; i++) fread (b[i], 1, 8, f);
  }
  // Uniform timesteps?
  else {
    printf ("reading uniform timesteps\n");
    int64_t t[n];
    t[0] = read64(f);
    printf ("read t[0]: %lld\n", (long long)(t[0]));
    int64_t dt = read64(f);
    printf ("read dt: %lld\n", (long long)dt);
    for (int i = 1; i < n; i++) t[i] = t[i-1] + dt;
    printf ("??: ");
    for (int i = 0; i < n; i++) printf ("%8lld", (long long)(t[i]));
    printf ("\n");
    for (int i = 0; i < n; i++) put64(b[i], t[i]);
  }
}

// Write levels
void write_levels (FILE *f, byte **b, int n, int fieldsize) {
  assert (fieldsize == 8);
  assert (n > 0);
  // Get the integer codes
  int64_t z[n];
  for (int i = 0; i < n; i++) z[i] = get64(b[i]);
  // Find max/min values
  int64_t max = z[0], min = z[0];
  for (int i = 0; i < n; i++) {
    if (z[i] > max) max = z[i];
    if (z[i] < min) min = z[i];
  }

  // 8-bit packing?
  if (min >= -128 && max <= 127) {
    printf ("writing 8-bit levels\n");
    fputc (1, f);
    for (int i = 0; i < n; i++) write8 (f, z[i]);
  }
  // 16-bit packing?
  else if (min >= -32768 && max <= 32767) {
    printf ("writing 16-bit levels\n");
    fputc (2, f);
    for (int i = 0; i < n; i++) write16 (f, z[i]);
  }
//  // 24-bit packing?
//  else if (min >= -(1<<23) && max < (1<<23)) {
//    printf ("writing 24-bit levels\n");
//    fputc (3, f);
//    for (int i = 0; i < n; i++) write24 (f, z[i]);
//  }
  // 32-bit packing?
  else if (min >= -(1LL<<31) && max < (1LL<<31)) {
    printf ("writing 32-bit levels\n");
    fputc (4, f);
    for (int i = 0; i < n; i++) write32 (f, z[i]);
  }
  // 64-bit packing?
  else {
    printf ("writing 64-bit levels\n");
    fputc (8, f);
    for (int i = 0; i < n; i++) write64 (f, z[i]);
  }
}

// Read levels
void read_levels (FILE *f, byte **b, int n, int fieldsize) {
  assert (fieldsize == 8);
  assert (n > 0);
  int nbytes = fgetc (f);
  int64_t z[n];
  // 8-bit packing?
  if (nbytes == 1) {
    printf ("reading 8-bit levels\n");
    for (int i = 0; i < n; i++) z[i] = (int8_t)read8(f);
  }
  // 16-bit packing?
  else if (nbytes == 2) {
    printf ("reading 16-bit levels\n");
    for (int i = 0; i < n; i++) z[i] = (int16_t)read16(f);
  }
//  // 24-bit packing?
//  else if (nbytes == 3) {
//    printf ("reading 24-bit levels\n");
//    for (int i = 0; i < n; i++) z[i] = read24(f);
//  }
  // 32-bit packing?
  else if (nbytes == 4) {
    printf ("reading 32-bit levels\n");
    for (int i = 0; i < n; i++) z[i] = (int32_t)read32(f);
  }
  // 64-bit packing?
  else {
    assert (nbytes == 8);
    printf ("reading 64-bit levels\n");
    for (int i = 0; i < n; i++) z[i] = (int64_t)read64(f);
  }
  for (int i = 0; i < n; i++) put64(b[i], z[i]);
}


void init (Index *ind) {
  init_Index (ind);
  set_vartype (ind, sizeof(VARID), sizeof(VARDESC), print_name);
  set_dimtype (ind, sizeof(TIMESTEP), read_timesteps, write_timesteps);
//  set_dimtype (ind, sizeof(TIMESTEP), NULL, NULL);
  set_dimtype (ind, sizeof(LEVEL), read_levels, write_levels);
//  set_dimtype (ind, sizeof(LEVEL), NULL, NULL);
}

int main (int argc, char *argv[]) {
  Index ind;
  init (&ind);

//  FILE *f = fopen ("/data/neishm/pygeode/mm_t31ref2d4_010_m01_ss", "rb");
//  FILE *f = fopen ("/home/neish/devel/pygeode/data/mm_t31ref2d4_010_m01_ss", "rb");
  FILE *f = fopen ("/home/mike/work/devel/pygeode/data/mm_t31ref2d4_010_m01_ss", "rb");
  assert (f != NULL);

  HEADER header;

  byte superlabel[80];
  memset (superlabel, ' ', 80);

  while (1) {
    Offset offset = ftello64(f);
    int size = read32(f);
    if (feof(f)) break;
    assert (size == sizeof(HEADER));
    fread (&header, sizeof(HEADER), 1, f);
    assert (read32(f) == sizeof(HEADER));

    size = read32(f);

    // Superlabel?
    if (strncmp(header.name, "LABL    ", 8) == 0) {
      fread (superlabel, 1, 80, f);
    }
    // Regular var?
    else {
      fseeko64 (f, size, SEEK_CUR);

      // Construct the relevant arrays
      VARID varid;
      memcpy (varid, header.name, 8);
      VARDESC vardesc;
      memcpy (vardesc.kind, header.kind, 8);
      memcpy (vardesc.ilg, header.ilg, 8);
      memcpy (vardesc.ilat, header.ilat, 8);
      memcpy (vardesc.khem, header.khem, 8);
      memcpy (vardesc.superlabel, superlabel, 80);
      TIMESTEP time;
      memcpy (time, header.time, sizeof(TIMESTEP));
      LEVEL level;
      memcpy (level, header.level, sizeof(LEVEL));
      assert (sizeof(vardesc) == 4*8 + 80);
      add_record (&ind, offset, varid, (byte*)&vardesc, time, level);
    }
    assert (read32(f) == size);

  }

  finalize_Index (&ind);
  fclose(f);

//  save_Index (&ind, "/data/neishm/pygeode/mm_t31ref2d4_010_m01_ss.index4");
//  save_Index (&ind, "/home/neish/devel/pygeode/data/mm_t31ref2d4_010_m01_ss.index4");
  save_Index (&ind, "/home/mike/work/devel/pygeode/data/mm_t31ref2d4_010_m01_ss.index4");


/*
  for (int v = 0; v < ind.nvars; v++) {
    print_name(ind.varids[v]);
    printf (":\n");
    if (ind.offsets[v] == NULL) {
      OffsetPattern *pat = ind.patterns[v];
      printf ("first = %lld\n", (long long)(ind.first[v]));
      for (int i = 0; i < pat->count; i++) {
        printf ("%d: %8d %8d *%lld\n", i, pat->shifts[i], pat->jumps[i], (long long)(pat->vals[i]));
      }
    }
  }
*/
//  print_Index (&ind);

  free_Index(&ind);

  init(&ind);
//  load_Index (&ind, "/data/neishm/pygeode/mm_t31ref2d4_010_m01_ss.index4");
//  load_Index (&ind, "/home/neish/devel/pygeode/data/mm_t31ref2d4_010_m01_ss.index4");
  load_Index (&ind, "/home/mike/work/devel/pygeode/data/mm_t31ref2d4_010_m01_ss.index4");

  int varid = -1;
  for (int v = 0; v < ind.nvars; v++)
    if (strncmp(ind.varids[v], "TEMP", 4) == 0) {varid = v; break; }
  assert (varid >= 0);

  printf ("temperature varid: %d\n", varid);

  // Test temperature offsets
  long long toff;
  toff = get_offset (&ind, varid, 0, 0);
  printf ("first: %llx\n", toff);
  toff = get_offset (&ind, varid, 0, 70);
  printf ("surf: %llx\n", toff);
  toff = get_offset (&ind, varid, 1, 0);
  printf ("time2: %llx\n", toff);
  printf ("timesteps: ");
  for (int i = 0; i < ind.dimlengths[varid][0]; i++) {
    printf ("%lld ", (long long)get64(ind.coords[varid][0][i]));
  }
  printf ("\n");
  printf ("levels: ");
  for (int i = 0; i < ind.dimlengths[varid][1]; i++) {
    printf ("%lld ", (long long)get64(ind.coords[varid][1][i]));
  }
  printf ("\n");

  free_Index (&ind);
  return 0;
}
