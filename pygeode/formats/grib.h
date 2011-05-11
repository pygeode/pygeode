//#define __USE_LARGEFILE64
#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE

#include <stdio.h>

#define bit23 8388608


typedef struct {
  //unsigned char grib[4];
  unsigned char length[3];
  unsigned char edition[1];
} IS_raw;

// Indicator Section
typedef struct {
  //char grib[5];
  unsigned int length;
  unsigned char edition;
} IS;

typedef struct {
  unsigned char length[3];
  unsigned char table_version[1];
  unsigned char center_id[1];
  unsigned char process_id[1];
  unsigned char grid_id[1];
  unsigned char gds_bms_flag[1];
  unsigned char param[1];
  unsigned char level_type[1];
  unsigned char level[2];
  unsigned char year[1];
  unsigned char month[1];
  unsigned char day[1];
  unsigned char hour[1];
  unsigned char minute[1];
  unsigned char forecast_time_unit[1];
  unsigned char p1[1];
  unsigned char p2[1];
  unsigned char time_range[1];
  unsigned char num_in_avg[2];
  unsigned char num_missing_avg[1];
  unsigned char century[1];
  unsigned char subcenter_id[1];
  unsigned char scale_factor[2];
} PDS_raw;

// Product Definition Section
// TODO: handle extended PDS (for NCEP ensembles)
typedef struct {
  unsigned int length;
  unsigned char table_version;
  unsigned char center_id;
  unsigned char process_id;
  unsigned char grid_id;
  unsigned char gds_bms_flag;
  unsigned char param;
  unsigned char level_type;
  unsigned short level;
  unsigned char year;
  unsigned char month;
  unsigned char day;
  unsigned char hour;
  unsigned char minute;
  unsigned char forecast_time_unit;
  unsigned char p1;
  unsigned char p2;
  unsigned char time_range;
  unsigned short num_in_avg;
  unsigned char num_missing_avg;
  unsigned char century;
  unsigned char subcenter_id;
  unsigned short scale_factor;
} PDS;

typedef struct {
  unsigned char length[3];
  unsigned char nv[1];
  unsigned char pv_pl[1];
  unsigned char grid_type[1];
// Only lat/lon grids supported here
  unsigned char ni[2];
  unsigned char nj[2];
  unsigned char la1[3];
  unsigned char lo1[3];
  unsigned char res_flags[1];
  unsigned char la2[3];
  unsigned char lo2[3];
  unsigned char di[2];
  unsigned char dj[2];
  unsigned char scan_flag[1];
  unsigned char reserved[4];
  //char v[MAX_NV][4];
} GDS_raw;

// Grid Description Section
typedef struct {
  unsigned int length;
  unsigned char nv;
  unsigned char pv_pl;
  unsigned char grid_type;
// Only lat/lon grids supported here
  unsigned short ni;
  unsigned short nj;
  signed int la1;
  signed int lo1;
  unsigned char res_flags;
  signed int la2;
  signed int lo2;
  signed short di;
  signed short dj;
  unsigned char scan_flag;
// insert PV here
  //float v[MAX_NV];
} GDS;


// Bit Map Section
//TODO
typedef struct {
  unsigned char length[3];
  unsigned char num_unused_bits[1];
  unsigned char source[2];
} BMS;

typedef struct {
  unsigned char length[3];
  unsigned char flag[1];
  unsigned char e[2];
  unsigned char r[4];
  unsigned char nbits[1];
// data must be read in seperately
} BDS_raw;

// Binary Data Section
typedef struct {
  unsigned int length;
  unsigned char flag;
  unsigned char unused_bits;
  signed short e;
  double r;
  unsigned char nbits;
// data must be read in seperately
  //unsigned int *idata;
  //float *fdata;
} BDS;


