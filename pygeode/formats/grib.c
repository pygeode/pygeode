#include <Python.h>
#include <numpy/arrayobject.h>

#include "grib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#ifdef WINDOWS
  #include <malloc.h>
  // Note: this may cause issues with files > 2GB on Windows!
  #define fseeko fseek
  #define ftello ftell
#else
  #include <alloca.h>
#endif

// Decode a floating-point value from the special encoding used by GRIB.
// Guess they didn't have IEEE standards when this format was created.
float decode_float (unsigned char x[4]) {
  float y;
  static double scale = 0;
  if (scale == 0) scale = pow(2,-24);  // only do this once
  unsigned int b = (x[1]<<16) + (x[2]<<8) + x[3];
  signed char a = (x[0] & 127) - 64;
  y = b * pow(16, a) * scale;
  if (x[0]&128) y *= -1;
//  printf ("decode float: %03d,%03d,%03d,%03d -> %g\n", x[0], x[1], x[2], x[3], y);
  return y;
}

// Find the next record header.
// Look for the work 'GRIB', ignore anything else before that.
// (Some files contain some metadata before the record starts)
int read_head (FILE *f) {
  int c;
  int match = 0;
  while (1) {
    c = getc(f); if (c == EOF) break;
    if (c == 'G') match = 1;
    else if (c == 'R' && match == 1) match++;
    else if (c == 'I' && match == 2) match++;
    else if (c == 'B' && match == 3) match++;
    else match = 0;
    if (match == 4) break;
  }
  if (c == EOF) return 0;
  return 1;
}

int decode_IS (IS_raw *raw, IS *is) {
  //strcpy (is->grib, "GRIB");
  is->length = (raw->length[0]<<16) + (raw->length[1]<<8) + raw->length[2];
  is->edition = raw->edition[0];
  assert (is->edition == 1);
  return 0;
}
int read_IS (FILE *f, IS *is) {
  IS_raw raw;
  int n = fread (&raw, sizeof(IS_raw), 1, f);
  if (n != 1) return -1;
  decode_IS (&raw, is);
  return 0;
}
void print_IS (IS *x) {
  printf ("Indicator section {\n");
  //printf ("  GRIB    = '%s'\n", x->grib);
  printf ("  length  = %i\n", x->length);
  printf ("  edition = %i\n", x->edition);
  printf ("}\n");
}

int decode_PDS (PDS_raw *raw, PDS *pds) {
  pds->length = (raw->length[0]<<16) + (raw->length[1]<<8) + raw->length[0];
  if (pds->length == 0) {
//    fprintf (stderr, "Warning: length of product definition section is unset.  Setting default of 28.\n");
    pds->length = 28;
  }
  pds->table_version = raw->table_version[0];
  pds->center_id     = raw->center_id[0];
  pds->process_id    = raw->process_id[0];
  pds->grid_id       = raw->grid_id[0];
  pds->gds_bms_flag  = raw->gds_bms_flag[0];
  pds->param         = raw->param[0];
  pds->level_type    = raw->level_type[0];
  pds->level         = (raw->level[0]<<8) + raw->level[1];
  pds->year          = raw->year[0];
  pds->month         = raw->month[0];
  pds->day           = raw->day[0];
  pds->hour          = raw->hour[0];
  pds->minute        = raw->minute[0];
  pds->forecast_time_unit = raw->forecast_time_unit[0];
  pds->p1            = raw->p1[0];
  pds->p2            = raw->p2[0];
  pds->time_range    = raw->time_range[0];
  pds->num_in_avg    = (raw->num_in_avg[0]<<8) + raw->num_in_avg[1];
  pds->num_missing_avg = raw->num_missing_avg[0];
  pds->century       = raw->century[0];
  pds->subcenter_id  = raw->subcenter_id[0];
  pds->scale_factor  = (raw->scale_factor[0]<<8) + raw->scale_factor[1];
  if (raw->scale_factor[0] & 128) pds->scale_factor = (pds->scale_factor^0x8000) * -1;

  // Check that GDS is set and BMS is unset
  if ((!pds->gds_bms_flag) & 128) {
    fprintf (stderr, "read_PDS: Error: GDS not available.\n");
    return 1;
  }
  if (pds->gds_bms_flag & 64) {
    fprintf (stderr, "read_PDS: Error: can't handle BMS yet.\n");
    return 1;
  }

  //TODO: skip to end of section
  return 0;
}
int read_PDS (FILE *f, PDS *pds) {
  PDS_raw raw;
  int n = fread (&raw, sizeof(PDS_raw), 1, f);
  if (n != 1) return -1;
  decode_PDS (&raw, pds);
  return 0;
}
void print_PDS (PDS *x) {
  printf ("Product Definition Section {\n");
  printf ("  length        = %i\n", x->length);
  printf ("  table_version = %i\n", x->table_version);
  printf ("  center_id     = %i\n", x->center_id);
  printf ("  process_id    = %i\n", x->process_id);
  printf ("  grid_id       = %i\n", x->grid_id);
  printf ("  gds_bms_flag  = %i\n", x->gds_bms_flag);
  printf ("  param         = %i\n", x->param);
  printf ("  level_type    = %i\n", x->level_type);
  printf ("  level         = %i\n", x->level);
  printf ("  year          = %i\n", x->year);
  printf ("  month         = %i\n", x->month);
  printf ("  day           = %i\n", x->day);
  printf ("  hour          = %i\n", x->hour);
  printf ("  minute        = %i\n", x->minute);
  printf ("  forecast_time_unit = %i\n", x->forecast_time_unit);
  printf ("  p1            = %i\n", x->p1);
  printf ("  p2            = %i\n", x->p2);
  printf ("  time_range    = %i\n", x->time_range);
  printf ("  num_in_avg    = %i\n", x->num_in_avg);
  printf ("  num_missing_avg = %i\n", x->num_missing_avg);
  printf ("  century       = %i\n", x->century);
  printf ("  subcenter_id  = %i\n", x->subcenter_id);
  printf ("  scale_factor  = %i\n", x->scale_factor);
  printf ("}\n");
}
int encode_PDS (PDS *pds, PDS_raw *raw) {
  raw->length[0] = pds->length>>16;
  raw->length[1] = pds->length>>8;
  raw->length[2] = pds->length;
  raw->table_version[0] = pds->table_version;
  raw->center_id[0]     = pds->center_id;
  raw->process_id[0]    = pds->process_id;
  raw->grid_id[0]       = pds->grid_id;
  raw->gds_bms_flag[0]  = pds->gds_bms_flag;
  raw->param[0]         = pds->param;
  raw->level_type[0]    = pds->level_type;
  raw->level[0]         = pds->level >> 8;
  raw->level[1]         = pds->level;
  raw->year[0]          = pds->year;
  raw->month[0]         = pds->month;
  raw->day[0]           = pds->day;
  raw->hour[0]          = pds->hour;
  raw->minute[0]        = pds->minute;
  raw->forecast_time_unit[0] = pds->forecast_time_unit;
  raw->p1[0]            = pds->p1;
  raw->p2 [0]           = pds->p2;
  raw->time_range[0]    = pds->time_range;
  raw->num_in_avg[0]    = pds->num_in_avg >> 8;
  raw->num_in_avg[1]    = pds->num_in_avg;
  raw->num_missing_avg[0] = pds->num_missing_avg;
  raw->century[0]       = pds->century;
  raw->subcenter_id[0]  = pds->subcenter_id;
  int scale_factor = pds->scale_factor;
  if (scale_factor < 0) printf ("warning: untested negative scale factor\n");
  if (scale_factor < 0) scale_factor = (-scale_factor) ^ 0x8000;
  raw->scale_factor[0]  = scale_factor >> 8;
  raw->scale_factor[1]  = scale_factor;
  return 0;
}
int write_PDS (PDS *pds, FILE *f) {
  PDS_raw raw;
  encode_PDS (pds, &raw);
  fwrite (&raw, sizeof(PDS_raw), 1, f);
  return 0;
}

int decode_GDS (GDS_raw *raw, GDS *gds) {
  gds->length = (raw->length[0]<<16) + (raw->length[1]<<8) + raw->length[2];
  gds->nv        = raw->nv[0];
  gds->pv_pl     = raw->pv_pl[0];
  gds->grid_type = raw->grid_type[0];
  gds->ni        = (raw->ni[0]<<8) + raw->ni[1];
  gds->nj        = (raw->nj[0]<<8) + raw->nj[1];
  gds->la1       = (raw->la1[0]<<16) + (raw->la1[1]<<8) + raw->la1[2];
  gds->lo1       = (raw->lo1[0]<<16) + (raw->lo1[1]<<8) + raw->lo1[2];
  gds->res_flags = raw->res_flags[0];
  gds->la2       = (raw->la2[0]<<16) + (raw->la2[1]<<8) + raw->la2[2];
  gds->lo2       = (raw->lo2[0]<<16) + (raw->lo2[1]<<8) + raw->lo2[2];
  gds->di        = (raw->di[0]<<8) + raw->di[1];
  gds->dj        = (raw->dj[0]<<8) + raw->dj[1];
  gds->scan_flag = raw->scan_flag[0];
  if (gds->la1 & bit23) gds->la1 = -(gds->la1^bit23);
  if (gds->lo1 & bit23) gds->lo1 = -(gds->lo1^bit23);
  if (gds->la2 & bit23) gds->la2 = -(gds->la2^bit23);
  if (gds->lo2 & bit23) gds->lo2 = -(gds->lo2^bit23);
//  if (gds->scan_flag & 128) gds->di *= -1;
//  if (gds->scan_flag & 64) gds->dj *= -1;
//  // Hack: if la1 > 0, assume a negative increment for dj
//  if (gds->la1 > 0) printf ("hacking dj to be positive (currently = %d)\n", gds->dj);
//  if (gds->la1 > 0) gds->dj = -abs(gds->dj);

  if (gds->grid_type != 0 && gds->grid_type != 4) {
    fprintf (stderr, "Error: can only handle lat/lon grids right now.\n");
    fprintf (stderr, "(expected data type 0 or 4, found type %i)\n", gds->grid_type);
    return 1;

  }
  //for (int i = 0; i < gds->nv; i++) gds->v[i] = decode_float(raw->v[i]);

  return 0;
}
int read_GDS (FILE *f, GDS *gds) {
  GDS_raw raw;
  int n = fread (&raw, sizeof(GDS_raw), 1, f);
  if (n != 1) return -1;
  decode_GDS (&raw, gds);
  return 0;
}
void print_GDS (GDS *x) {
  printf ("Grid Description Section {\n");
  printf ("  length    = %i\n", x->length);
  printf ("  nv        = %i\n", x->nv);
  printf ("  pv_pl     = %i\n", x->pv_pl);
  printf ("  grid_type = %i\n", x->grid_type);
  printf ("  ni        = %i\n", x->ni);
  printf ("  nj        = %i\n", x->nj);
  printf ("  la1       = %i\n", x->la1);
  printf ("  lo1       = %i\n", x->lo1);
  printf ("  res_flags = %i\n", x->res_flags);
  printf ("  la2       = %i\n", x->la2);
  printf ("  lo2       = %i\n", x->lo2);
  printf ("  di        = %i\n", x->di);
  printf ("  dj        = %i\n", x->dj);
  printf ("  scan_flag = %i\n", x->scan_flag);
  printf ("  V         =\n");
  //for (int i = 0; i < x->nv; i++) printf ("    %i %g\n", i, x->v[i]);
  printf ("}\n");
}
int encode_GDS (GDS *gds, GDS_raw *raw) {

  raw->length[0]  = gds->length >> 16;
  raw->length[1]  = gds->length >> 8;
  raw->length[2]  = gds->length;

  raw->nv[0]        = gds->nv;
  raw->pv_pl[0]     = gds->pv_pl;
  raw->grid_type[0] = gds->grid_type;

  raw->ni[0] = gds->ni >> 8;
  raw->ni[1] = gds->ni;
  raw->nj[0] = gds->nj >> 8;
  raw->nj[1] = gds->nj;

//  int di = gds->di, dj = gds->dj;
//  if (gds->scan_flag & 128) di *= -1;
//  if (gds->scan_flag & 64) dj *= -1;
  int la1 = gds->la1, la2 = gds->la2, 
      lo1 = gds->lo1, lo2 = gds->lo2;
  if (la1 < 0) la1 = (-la1) ^ 0x800000;
  if (lo1 < 0) lo1 = (-lo1) ^ 0x800000;
  if (la2 < 0) la2 = (-la2) ^ 0x800000;
  if (lo2 < 0) lo2 = (-lo2) ^ 0x800000;

  raw->la1[0] = la1 >> 16;
  raw->la1[1] = la1 >> 8;
  raw->la1[2] = la1;
  raw->lo1[0] = lo1 >> 16;
  raw->lo1[1] = lo1 >> 8;
  raw->lo1[2] = lo1;
  raw->res_flags[0] = gds->res_flags;
  raw->la2[0] = la2 >> 16;
  raw->la2[1] = la2 >> 8;
  raw->la2[2] = la2;
  raw->lo2[0] = lo2 >> 16;
  raw->lo2[1] = lo2 >> 8;
  raw->lo2[2] = lo2;
  raw->di[0] = gds->di >> 8;
  raw->di[1] = gds->di;
  raw->dj[0] = gds->dj >> 8;
  raw->dj[1] = gds->dj;
  raw->scan_flag[0] = gds->scan_flag;
  raw->reserved[0] = raw->reserved[1] = raw->reserved[2] = raw->reserved[3] = 0;

  return 0;
}

// Write a raw GDS to file
int write_GDS (GDS *gds, FILE *f) {
  GDS_raw raw;
  encode_GDS (gds, &raw);
  fwrite (&raw, sizeof(GDS_raw), 1, f);
  return 0;
}

int decode_BDS (BDS_raw *raw, BDS *bds) {
  bds->length = (raw->length[0]<<16) + (raw->length[1]<<8) + raw->length[2];
  bds->flag = raw->flag[0]>>4;
  bds->unused_bits = raw->flag[0] & 15;
  bds->e = ((raw->e[0]&127)<<8) + raw->e[1];
  if (raw->e[0]&128) bds->e *= -1;
  bds->r = decode_float (raw->r);
  bds->nbits = raw->nbits[0];

  if (bds->flag != 0) {
    fprintf (stderr, "decode_BDS: Error: can't handle any special data flags yet.\n");
    return 1;
  }
  return 0;
}
int read_BDS (FILE *f, BDS *bds) {
  BDS_raw raw;
  int n = fread (&raw, sizeof(BDS_raw), 1, f);
  if (n != 1) return -1;
  decode_BDS (&raw, bds);
  return 0;
}

/*
void print_BDS (BDS *x) {
  printf ("Binary Data Section {\n");
  printf ("  length      = %i\n", x->length);
  printf ("  flag        = %i\n", x->flag);
  printf ("  unused_bits = %i\n", x->unused_bits);
  printf ("  e           = %i\n", x->e); 
  printf ("  r           = %g\n", x->r);
  printf ("  nbits       = %i\n", x->nbits);
//  printf ("  data sample(1-100):");
//  int i;
//  for (i = 0; i < 100; i++) printf ("%i ", x->idata[i]);
//  printf ("\n");
  printf ("}\n");
}
*/

// Read a record terminator ('7777')
int read_EOR (FILE *f) {
  char data[4];
  int n = fread (data, sizeof(char)*4, 1, f);
  if (n != 1) return -1;
  if (strncmp(data, "7777", 4)!=0) {
    fprintf (stderr, "read_EOR: Error: bad record terminator\n");
    return 1;
  }
  return 0;
}



// Skip over the BDS section
int skip_BDS (FILE *f) {
  BDS_raw raw;
  int n = fread (&raw, sizeof(BDS_raw), 1, f);
  if (n != 1) return -1;
  int length = (raw.length[0]<<16) + (raw.length[1]<<8) + raw.length[2];
  //printf ("BDS length = %d, nbits = %d\n", length, raw.nbits[0]);
  fseeko (f, length - sizeof(BDS_raw), SEEK_CUR);
  return 0;

}

#define o(x) ((long long)(o[x]))
long long decode_offset(unsigned char *o) {
  return (o(0)<<56) | (o(1)<<48) | (o(2)<<40) | (o(3)<<32)
       | (o(4)<<24) | (o(5)<<16) | (o(6)<<8) | (o(7));
//  printf ("offset: %d,%d,%d,%d,%d,%d,%d,%d -> %lld\n",
//           o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7],
//           out);
}
#undef o

// Variable code (table #, param #, and level type)
typedef unsigned char Code[4];
#define samecode(x,y) (x[0] == y[0] && x[1] == y[1] && x[2] == y[2] && x[3] == y[3])
#define setcode(x,y) do{x[0] = y[0]; x[1] = y[1]; x[2] = y[2]; x[3] = y[3];} while(0)
#define splitcode(x) x[0],x[1],x[2],x[3]

// Level code
typedef unsigned char Level[2];
#define samelevel(x,y) (x[0] == y[0] && x[1] == y[1])
#define setlevel(x,y) do{x[0] = y[0]; x[1] = y[1];} while(0)
#define splitlevel(x) x[0],x[1]

// Time code
typedef unsigned char Time[13];
#define sametime(x,y) (memcmp(x,y,sizeof(Time))==0)
#define settime(x,y) memcpy(x,y,sizeof(Time))
#define splittime(x) (x[11]-1)*100+x[0], x[1], x[2], x[3], x[4]


/*******************************************************
/  make an index file
*******************************************************/
#define MAX_LEVELS 255
int make_index (char *gribfile, char *indexfile) {

  FILE *f = fopen(gribfile, "rb");
  if (f == NULL) {
    fprintf (stderr, "make_index: Error: can't open file '%s'\n", gribfile);
    return 1;
  }

  // Linked list of times
  typedef struct TimeLL_ {
    Time time;
    long long offset[MAX_LEVELS];
    struct TimeLL_ *next;
  } TimeLL;

  // Linked list of variables
  typedef struct VarLL_ {
    Code code;
    int nt; int nz;
    TimeLL *t0, *tlast;
    Level z[MAX_LEVELS];
    unsigned char pv[2][MAX_LEVELS][4];
    PDS pds;
    GDS gds;
    struct VarLL_ *next;
  } VarLL;

  VarLL *first = NULL, *last = NULL;
  int nvars = 0;

  // Get all variables
  while (read_head(f)) {
    long long offset = ftello(f)-4; // Recorded offset is right before 'GRIB' sequence
    IS is;
    PDS pds;
    GDS gds;

    read_IS(f,&is);
    read_PDS(f, &pds);
    read_GDS(f, &gds);
    //fseeko (f, 4*gds.nv[0], SEEK_CUR);
    unsigned char pv[2][MAX_LEVELS][4];
    int n;
    n = fread (&(pv[0][0][0]), 4, gds.nv/2, f);
    if (n != gds.nv/2) return -1;
    n = fread (&(pv[1][0][0]), 4, gds.nv/2, f);
    if (n != gds.nv/2) return -1;
    skip_BDS(f);
    read_EOR(f);

    // Get the variable code
    Code c = {pds.center_id, pds.table_version, pds.param, pds.level_type};
    VarLL *v;
    for (v = first; v != NULL; v = v->next) if (samecode(v->code, c)) break;
    // New var?
    if (v == NULL) {
      nvars++;
      v = alloca(sizeof(VarLL));
      setcode(v->code,c);
      // PDS & GDS (with levels & time info stripped out)
//      memcpy(&(v->pds), &pds, sizeof(PDS));
      v->pds = pds;
      // Strip out level and timestep info
//      memset(&(v->pds.level), 0, 15);
      #define p v->pds
      p.level = p.year = p.month = p.day = p.hour = p.minute      = 
                p.forecast_time_unit = p.p1 = p.p2 = p.time_range =
                p.num_in_avg = p.num_missing_avg = p.century      = 0;
      #undef p
//      memcpy(&(v->gds), &gds, sizeof(GDS_raw));
      v->gds = gds;
      v->nt = v->nz = 0;
      v->t0 = v->tlast = NULL;
      // Store pv values
      memcpy(&(v->pv[0][0][0]), &(pv[0][0][0]), 4*gds.nv/2);
      memcpy(&(v->pv[1][0][0]), &(pv[1][0][0]), 4*gds.nv/2);
      v->next = NULL;
      if (last != NULL) last->next = v;
      else first = v;
      last = v;      
    }
    // Get the time
    Time time = { pds.year, pds.month, pds.day, pds.hour,
                  pds.minute, pds.forecast_time_unit, pds.p1,
                  pds.p2, pds.time_range, pds.num_in_avg,
                  pds.num_missing_avg, pds.century };
    TimeLL *t;
    for (t = v->t0; t != NULL; t = t->next) if (sametime(t,time)) break;
    // New time?
    if (t == NULL) {
      v->nt++;
      t = alloca(sizeof(TimeLL));
      settime(t, time);
      memset(t->offset, 0, sizeof(long long)*v->nz);
      t->next = NULL;
      if (v->tlast != NULL) v->tlast->next = t;
      else v->t0 = t;
      v->tlast = t;
    }
    // Get the level
//    Level l = { pds.level[0], pds.level[1] };
    Level l = { pds.level>>8, pds.level&255 };
    int zi;
    for (zi = 0; zi < v->nz; zi++) if (samelevel(v->z[zi],l)) break;
    // New level?
    if (zi == v->nz) {
      v->nz++;
      setlevel(v->z[zi],l);
    }

    // Get the offset
    t->offset[zi] = offset;
  }

  fclose(f);

/*
  for (VarLL *v = first; v != NULL; v = v->next) {
    printf ("%03d,%03d,%03d,%03d:\n  ", splitcode(v->code));
    for (TimeLL *t = v->t0; t != NULL; t = t->next) {
      printf ("%4d-%02d-%02d,%02d:%02d:\n", splittime(t->time));
      for (int zi = 0; zi < v->nz; zi++) {
       printf ("   %03d,%03d: %lld\n", splitlevel(v->z[zi]),t->offset[zi]);
      }
      printf ("\n");
    }
    printf ("\n");
  }
*/


  /***********************************************
  / Now, we can collect this stuff into an index
  ***********************************************/
  f = fopen (indexfile, "wb");
  if (f == NULL) {
    fprintf (stderr, "make_index: Error: can't open file '%s'\n", indexfile);
    return 1;
  }

  assert (nvars < 256);  // for simplicity
  putc ((unsigned char)(nvars), f);

  // Write variable info
  for (VarLL *v = first; v != NULL; v = v->next) {
    // Write the code
    fwrite (v->code, 4, 1, f);
    // Write the PDS
//    fwrite (&(v->pds), sizeof(PDS_raw), 1, f);
    write_PDS (&(v->pds), f);
    // Write the GDS
//    fwrite (&(v->gds), sizeof(GDS_raw), 1, f);
    write_GDS (&(v->gds), f);
    // nt / nz
    assert (v->nt < 256);
    assert (v->nz < 256);
    putc (v->nt, f);
    putc (v->nz, f);
    // Write the pv values
    // nv is stored in the GDS above, but write it again just for the hell of it.
    putc (v->gds.nv, f);
    fwrite (&(v->pv[0][0][0]), 4, v->gds.nv/2, f);
    fwrite (&(v->pv[1][0][0]), 4, v->gds.nv/2, f);
  }
  // Write levels, timesteps, and offsets
  for (VarLL *v = first; v != NULL; v = v->next) {
    fwrite (v->z, 2, v->nz, f);
    for (TimeLL *t = v->t0; t != NULL; t = t->next) {
      fwrite (t->time, sizeof(Time), 1, f);
      // Write offsets
      for (int i = 0; i < v->nz; i++) {
        long long o = t->offset[i];
//        unsigned char c[8];
//        c[0] = o>>56; c[1] = o>>48; c[2] = o>>40; c[3] = o>>32;
//        c[4] = o>>24; c[5] = o>>16; c[6] = o>>8; c[7] = o;
        putc(o>>56,f); putc(o>>48,f); putc(o>>40,f); putc(o>>32,f);
        putc(o>>24,f); putc(o>>16,f); putc(o>>8,f); putc(o,f);
//        fwrite (c, 8, 1, f);
      }
    }
  }
  

  fclose(f);


  return 0;
}

// For easy storage of variables
typedef struct {
  Code code;
  PDS pds;
  GDS gds;
  int nt, nz;
  Time *time;
  Level *level;
  int nv;  // For eta levels
  float *a;
  float *b;
  long long **offset;
} Var;

typedef struct {
  int nvars;
  Var *var;
} Index;

/****************************************************************************
/ Read an index from a file
****************************************************************************/
int read_Index (char *indexfile, Index **index) {
  FILE *f = fopen (indexfile, "rb");
  if (f == NULL) {
    fprintf (stderr, "read_Index: Error: can't open file '%s'\n", indexfile);
    return 1;
  }

  // Read variable info
  unsigned char nvars = getc(f);
  //printf ("nvars = %d\n", nvars);

  int nrecs = 0, nt = 0, nz = 0;

  Var *var = malloc(sizeof(Var)*nvars);
  for (int i = 0; i < nvars; i++) {
    Var *v = var+i;
    // Read the code
    fread (v->code, 4, 1, f);
    // Read the PDS
    read_PDS (f, &(v->pds));
    // Read the GDS
    read_GDS (f, &(v->gds));
    // nt / nz
    v->nt = getc (f);
    v->nz = getc (f);

    nt += v->nt;
    nz += v->nz;
    nrecs += (v->nt * v->nz);

    // Read the pv values
    v->nv = getc (f);
    // Note: length is scaled by 2 here (since we're separating a and b)
    if (v->nv % 2 != 0) {
      fprintf (stderr, "read_index: nv parity error\n");
      return -1;
    }
    //assert (v->nv % 2 == 0);
    v->nv /= 2;
    if (v->nv > 0) {
      int nv = v->nv;
      unsigned char a[nv][4];
      unsigned char b[nv][4];
      fread (&(a[0][0]), 4, nv, f);
      fread (&(b[0][0]), 4, nv, f);
      v->a = malloc(sizeof(float)*nv);
      v->b = malloc(sizeof(float)*nv);
      for (int j = 0; j < nv; j++) {
        v->a[j] = decode_float(a[j]);
        v->b[j] = decode_float(b[j]);
      }
      /*
      if (i == 0) {
        printf ("n\ta\tb\n");
        for (int j = 0; j < k; j++) {
          printf ("%d\t%f\t%f\n", j, v->a[j], v->b[j]);
        }
      }
      */
    } else {
      v->a = NULL;
      v->b = NULL;
    }


/*
    printf ("var = %d:\n", i);
    print_PDS (&(v->pds));
    print_GDS (&(v->gds));
    printf ("nt = %d, nz = %d\n", v->nt, v->nz);
*/
  }

  //printf ("total nt = %d, nrecs = %d\n", nt, nrecs);

  // Allocate space for the timesteps, levels, and offsets
  Level *level = malloc(sizeof(Level)*nz);
  Time *time = malloc(sizeof(Time)*nt);
  long long **offset_ = malloc(sizeof(long long*) * nt);
  long long *offset = malloc(sizeof(long long) * nrecs);
//  float *a = malloc(sizeof(float) * nrecs);
//  float *b = malloc(sizeof(float) * nrecs);

  // Read in the data
  for (int i = 0; i < nvars; i++) {
    Var *v = var+i;
    // point to the allocated space
    v->time = time;
    v->level = level;
    v->offset = offset_;
    for (int j = 0; j < v->nt; j++) v->offset[j] = offset + j*v->nz;
    // Load the data from file
    fread (v->level, 2, v->nz, f);
    for (int j = 0; j < v->nt; j++) {
      fread (&(v->time[j]), sizeof(Time), 1, f);
      unsigned char o[8*v->nz];
      fread (o, 8, v->nz, f);
      for (int k = 0; k < v->nz; k++) {
        //printf ("offset: %d,%d,%d,%d,%d,%d,%d,%d\n", o[8*k], o[8*k+1], o[8*k+2], o[8*k+3], o[8*k+4], o[8*k+5], o[8*k+6], o[8*k+7]);
        v->offset[j][k] = decode_offset(o+8*k);
      }
    }
    // move the global pointers to the next available space
    level += v->nz;
    time += v->nt;
    offset_ += v->nt;
    offset += v->nt * v->nz;
  }
/*
  for (int i = 0; i < nvars; i++) {
    Var *v = var+i;
    printf ("var %d (%03d+%03d+%03d+%03d):\n", i, splitcode(v->code));
    for (int j = 0; j < v->nt; j++) {
      printf (" %4d-%02d-%02d %02d:%02d:\n", splittime(v->time[j]));
      for (int k = 0; k < v->nz; k++) {
        printf ("  level %d: %lld\n", v->level[k][1], v->offset[j][k]);
      }
    }
  }
*/
  fclose(f);

  *index = malloc(sizeof(Index));
  (*index)->nvars = nvars;
  (*index)->var = var;

  return 0;
}

int free_Index (Index **index) {
  assert (index != NULL);
  assert (*index != NULL);
  free ((*index)->var[0].offset[0]);
  free ((*index)->var[0].offset);
  free ((*index)->var[0].time);
  free ((*index)->var[0].level);

  // free a & b coefficients
  for (int i = 0; i < (*index)->nvars; i++) {
    if ((*index)->var[i].nv == 0) continue;
    free ((*index)->var[i].a);
    free ((*index)->var[i].b);
  }
  free ((*index)->var);
  free (*index);
  *index = NULL;
  return 0;
}

// Similar to above, but is compatible with Python CObjects
void destroy_Index (PyObject *index) {
  Index *junk = (Index*)PyCapsule_GetPointer(index,NULL);
  free_Index(&junk);
}

int get_nvars (Index *index) {
  return index->nvars;
}

/*
// Return an array of pointers to the variables
int get_vars (Index *index, Var **var) {
  *var = index->var;
  return 0;
}
*/

// Open a file
int open_grib (char *filename, FILE **f) {
  assert (f != NULL);
  *f = fopen(filename, "rb");
  return 0;
}
// Close a file
int close_grib (FILE **f) {
  assert (f != NULL);
  assert (*f != NULL);
  fclose (*f);
  *f = NULL;
  return 0;
}

// Similar to above, but compatible with Python CObjects
void destroy_grib (PyObject *f) {
  FILE *junk = (FILE*)PyCapsule_GetPointer(f,NULL);
  close_grib (&junk);
}

// Get info about a variable
int get_varcode (Index *index, int v, int *center, int *table, int *varid, int *level_type) {
  assert (index != NULL);
  assert (v >= 0 && v < index->nvars);
  unsigned char *code = index->var[v].code;
  *center = code[0];
  *table = code[1];
  *varid = code[2];
  *level_type = code[3];
  return 0;
}

int get_var_nt (Index *index, int v) {
  assert (index != NULL);
  assert (v >= 0 && v < index->nvars);
  return index->var[v].nt;
}
int get_var_t (Index *index, int v, int *y, int *m, int *d, int *H, int *M) {
  assert (index != NULL);
  assert (v >= 0 && v < index->nvars);
  assert (y != NULL && m != NULL && d != NULL && H != NULL && M != NULL);
  int nt = index->var[v].nt;
  for (int i = 0; i < nt; i++) {
    unsigned char *time = index->var[v].time[i];
    y[i] = (time[11]-1)*100 + time[0];
    m[i] = time[1];
    d[i] = time[2];
    H[i] = time[3];
    M[i] = time[4];
  }
  return 0;
}
int get_var_nz (Index *index, int v) {
  assert (index != NULL);
  assert (v >= 0 && v < index->nvars);
  return index->var[v].nz;
}
int get_var_z (Index *index, int v, int *l1, int *l2) {
  assert (index != NULL);
  assert (v >= 0 && v < index->nvars);
  assert (l1 != NULL && l2 != NULL);
  int nz = index->var[v].nz;
  for (int i = 0; i < nz; i++) {
    unsigned char *level = index->var[v].level[i];
    l1[i] = level[0];
    l2[i] = level[1];
  }
  return 0;
}
int get_var_neta (Index *index, int v) {
  assert (index != NULL);
  assert (v >= 0 && v < index->nvars);
  return index->var[v].nv;
}
int get_var_eta (Index *index, int v, float *a, float *b) {
  assert (index != NULL);
  assert (v >= 0 && v < index->nvars);
  assert (a != NULL && b != NULL);
  Var *var = &(index->var[v]);
  for (int i = 0; i < var->nv; i++) {
    a[i] = var->a[i];
    b[i] = var->b[i];
  }
  return 0;
}
int get_grid (Index *index, int v, int *type, int *ni, int *nj, 
              int *la1, int *lo1, int *la2, int *lo2) {
  assert (index != NULL);
  assert (v >= 0 && v < index->nvars);
  assert (type!=NULL && ni!=NULL && nj!=NULL && la1!=NULL && lo1!=NULL && la2!=NULL && lo2!=NULL);
  GDS *g = &(index->var[v].gds);
  *type = g->grid_type;
  *ni = g->ni;
  *nj = g->nj;
  *la1 = g->la1;
  *lo1 = g->lo1;
  *la2 = g->la2;
  *lo2 = g->lo2;
  return 0;
}


/****************************************************************************
/ Read some data
****************************************************************************/

int read_data (FILE *f, Index *index, int v, int t, int z, double *out) {
  assert (f != NULL);
  assert (index != NULL);
  assert (out != NULL);
  int nt = index->var[v].nt;
  int nz = index->var[v].nz;
  assert (0 <= t && t < nt);
  assert (0 <= z && z < nz);

  int size = index->var[v].gds.ni * index->var[v].gds.nj;

  long long offset = index->var[v].offset[t][z];

  fseeko (f, offset, SEEK_SET);
  read_head(f);
  IS is;
  PDS pds;
  GDS gds;
  BDS bds;

  read_IS(f,&is);
  read_PDS(f, &pds);
//  print_PDS(&pds);
  read_GDS(f, &gds);
  fseeko (f, 4*gds.nv, SEEK_CUR);

  read_BDS(f, &bds);

  double e = pow(2, bds.e);
  double r = bds.r;
  double D = pow(10,-pds.scale_factor);

  //printf ("D = %d,  e = %g, r = %g\n", pds.scale_factor, e, r);

  int pack_size = bds.length - sizeof(BDS_raw);
  // Allocate space for the temporary data
  unsigned char *packed = malloc(pack_size);
  fread (packed, 1, pack_size, f);
  // Translate the packed values
  char nbits = bds.nbits;
//  printf ("nbits: %d\n", nbits);
  for (int i = 0; i < size; i++) {
    int p = (i*nbits)/8;
    unsigned int d  = (packed[p]<<24) + (packed[p+1]<<16) + (packed[p+2]<<8) + packed[p+3];
    char leftpad = (i*nbits)%8;
    char rightpad = 32 - nbits - leftpad;
    d <<= leftpad; d >>= leftpad;
    d >>= rightpad;
    out[i] = (r + d * e) * D;
  }
  free (packed);
  
  read_EOR(f);

//  out += size*sizeof(double);

  return 0;
}

// Extend the read_data routine to handle multiple timesteps and levels,
// and allow the lat & lon dimensions to be subsetted.
int read_data_loop (FILE *f, Index *index, int v, int nt, int *t, 
                    int nz, int *z, int ny, int *y, int nx, int *x, double *out) {
  GDS *gds = &(index->var[v].gds);
  int ni = gds->ni, nj = gds->nj;
  // Allocate space for a single, complete record
  double *record = malloc(sizeof(double) * ni * nj);
  double *o = out;
  // Size of a (reduced) record
  int recsize = nx * ny;
  for (int ti = 0; ti < nt; ti++) {
    for (int zi = 0; zi < nz; zi++) {
      // Read the record
      read_data (f, index, v, t[ti], z[zi], record);
      // Put into the output
      for (int yi = 0; yi < ny; yi++) {
        int y_ = y[yi];
        for (int xi = 0; xi < nx; xi++) {
          o[yi*nx+xi] = record[y_*ni + x[xi]];
        }
      }
      // Next record
      o += recsize;
    }
  }
  // Free all temporary space
  free (record);
  return 0;
}


// Python wrapper

static PyObject *gribcore_make_index (PyObject *self, PyObject *args) {
  char *gribfile, *indexfile;
  if (!PyArg_ParseTuple(args, "ss", &gribfile, &indexfile)) return NULL;
  int ret = make_index (gribfile, indexfile);
  if (ret != 0) return NULL;
  Py_RETURN_NONE;
}

static PyObject *gribcore_read_Index (PyObject *self, PyObject *args) {
  char *indexfile;
  Index *index = NULL;
  if (!PyArg_ParseTuple(args, "s", &indexfile)) return NULL;
  int ret = read_Index (indexfile, &index);
  if (ret != 0) return NULL;
  return PyCapsule_New(index, NULL, destroy_Index);
}

static PyObject *gribcore_open_grib (PyObject *self, PyObject *args) {
  char *filename;
  FILE *f;
  if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;
  int ret = open_grib (filename, &f);
  if (ret != 0) return NULL;
  return PyCapsule_New(f, NULL, destroy_grib);
}

static PyObject *gribcore_get_varcode (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  int v;
  int center, table, varid, level_type;
  if (!PyArg_ParseTuple(args, "O!i", &PyCapsule_Type, &index_obj, &v)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  int ret = get_varcode (index, v, &center, &table, &varid, &level_type);
  if (ret != 0) return NULL;
  return Py_BuildValue("(i,i,i,i)", center, table, varid, level_type);
}

static PyObject *gribcore_get_var_nt (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  int v, nt;
  if (!PyArg_ParseTuple(args, "O!i", &PyCapsule_Type, &index_obj, &v)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  nt = get_var_nt (index, v);
  return Py_BuildValue("i", nt);
}

static PyObject *gribcore_get_var_t (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  int v;
  npy_intp nt;
  PyArrayObject *y, *m, *d, *H, *M;
  if (!PyArg_ParseTuple(args, "O!i", &PyCapsule_Type, &index_obj, &v)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  nt = get_var_nt (index, v);
  y = (PyArrayObject*)PyArray_SimpleNew(1,&nt,NPY_INT32);
  m = (PyArrayObject*)PyArray_SimpleNew(1,&nt,NPY_INT32);
  d = (PyArrayObject*)PyArray_SimpleNew(1,&nt,NPY_INT32);
  H = (PyArrayObject*)PyArray_SimpleNew(1,&nt,NPY_INT32);
  M = (PyArrayObject*)PyArray_SimpleNew(1,&nt,NPY_INT32);
  int ret = get_var_t (index, v, (int*)y->data, (int*)m->data, (int*)d->data, (int*)H->data, (int*)M->data);
  if (ret != 0) return NULL;
  PyObject *tuple = Py_BuildValue("(O,O,O,O,O)", y, m, d, H, M);
  Py_DECREF (y);
  Py_DECREF (m);
  Py_DECREF (d);
  Py_DECREF (H);
  Py_DECREF (M);
  return tuple;
}

static PyObject *gribcore_get_var_nz (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  int v, nz;
  if (!PyArg_ParseTuple(args, "O!i", &PyCapsule_Type, &index_obj, &v)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  nz = get_var_nz (index, v);
  return Py_BuildValue("i", nz);
}

static PyObject *gribcore_get_var_z (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  int v;
  npy_intp nz;
  PyArrayObject *l1, *l2;
  if (!PyArg_ParseTuple(args, "O!i", &PyCapsule_Type, &index_obj, &v)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  nz = get_var_nz (index, v);
  l1 = (PyArrayObject*)PyArray_SimpleNew(1,&nz,NPY_INT32);
  l2 = (PyArrayObject*)PyArray_SimpleNew(1,&nz,NPY_INT32);
  int ret = get_var_z (index, v, (int*)l1->data, (int*)l2->data);
  if (ret != 0) return NULL;
  PyObject *tuple = Py_BuildValue("(O,O)", l1, l2);
  Py_DECREF (l1);
  Py_DECREF (l2);
  return tuple;
}

static PyObject *gribcore_get_var_neta (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  int v, neta;
  if (!PyArg_ParseTuple(args, "O!i", &PyCapsule_Type, &index_obj, &v)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  neta = get_var_neta (index, v);
  return Py_BuildValue("i", neta);
}

static PyObject *gribcore_get_var_eta (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  int v;
  npy_intp neta;
  PyArrayObject *a, *b;
  if (!PyArg_ParseTuple(args, "O!i", &PyCapsule_Type, &index_obj, &v)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  neta = get_var_neta (index, v);
  a = (PyArrayObject*)PyArray_SimpleNew(1,&neta,NPY_FLOAT32);
  b = (PyArrayObject*)PyArray_SimpleNew(1,&neta,NPY_FLOAT32);
  int ret = get_var_eta (index, v, (float*)a->data, (float*)b->data);
  if (ret != 0) return NULL;
  PyObject *tuple = Py_BuildValue("(O,O)", a, b);
  Py_DECREF (a);
  Py_DECREF (b);
  return tuple;
}

static PyObject *gribcore_get_grid (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  int v, type, ni, nj, la1, lo1, la2, lo2;
  if (!PyArg_ParseTuple(args, "O!i", &PyCapsule_Type, &index_obj, &v)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  int ret = get_grid (index, v, &type, &ni, &nj, &la1, &lo1, &la2, &lo2);
  if (ret != 0) return NULL;
  return Py_BuildValue("(i,i,i,i,i,i,i)", type, ni, nj, la1, lo1, la2, lo2);
}

static PyObject *gribcore_get_nvars (PyObject *self, PyObject *args) {
  PyObject *index_obj;
  Index *index;
  if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &index_obj)) return NULL;
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);
  int nvars = get_nvars(index);
  return Py_BuildValue("i", nvars);
}

static PyObject *gribcore_read_data_loop (PyObject *self, PyObject *args) {
  PyObject *f_obj, *index_obj;
  FILE *f;
  Index *index;
  int v, nt, nz, ny, nx;
  PyArrayObject *t, *z, *y, *x, *out;
  // Assuming the arrays are of the right type and contiguous
  if (!PyArg_ParseTuple(args, "O!O!iiO!iO!iO!iO!O!", &PyCapsule_Type, &f_obj, &PyCapsule_Type, &index_obj, &v, &nt, &PyArray_Type, &t, &nz, &PyArray_Type, &z, &ny, &PyArray_Type, &y, &nx, &PyArray_Type, &x, &PyArray_Type, &out)) return NULL;
  f = (FILE*)PyCapsule_New(f_obj,NULL,NULL);
  index = (Index*)PyCapsule_New(index_obj,NULL,NULL);

  int ret = read_data_loop (f, index, v, nt, (int*)t->data, 
    nz, (int*)z->data, ny, (int*)y->data, nx, (int*)x->data,
    (double*)out->data);
  if (ret != 0) return NULL;
  Py_RETURN_NONE;
}


static PyMethodDef GribMethods[] = {
  {"make_index", gribcore_make_index, METH_VARARGS, ""},
  {"read_Index", gribcore_read_Index, METH_VARARGS, ""},
  {"open_grib", gribcore_open_grib, METH_VARARGS, ""},
  {"get_varcode", gribcore_get_varcode, METH_VARARGS, ""},
  {"get_var_nt", gribcore_get_var_nt, METH_VARARGS, ""},
  {"get_var_t", gribcore_get_var_t, METH_VARARGS, ""},
  {"get_var_nz", gribcore_get_var_nz, METH_VARARGS, ""},
  {"get_var_z", gribcore_get_var_z, METH_VARARGS, ""},
  {"get_var_neta", gribcore_get_var_neta, METH_VARARGS, ""},
  {"get_var_eta", gribcore_get_var_eta, METH_VARARGS, ""},
  {"get_grid", gribcore_get_grid, METH_VARARGS, ""},
  {"get_nvars", gribcore_get_nvars, METH_VARARGS, ""},
  {"read_data_loop", gribcore_read_data_loop, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "gribcore",          /* m_name */
        NULL,                /* m_doc */
        -1,                  /* m_size */
        GribMethods,         /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

static PyObject *
moduleinit(void)
{
    PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("gribcore", GribMethods);
#endif

    import_array();

    return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initgribcore(void)
    {
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_gribcore(void)
    {
        return moduleinit();
    }
#endif

