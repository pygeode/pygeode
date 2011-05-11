int str2int8 (unsigned char *str, unsigned char *x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = str[i];
  }
  return 0;
}


// Convert a string to an array (assume it's encoded as network endian (big endian)
int str2int32 (unsigned char *str, unsigned int *x, int n) {
  for (int i = 0; i < n; i++) {
    int j = 4*i;
    x[i] = str[j] * (1<<24) + str[j+1] * (1<<16) + str[j+2] * (1<<8) + str[j+3];
  }
  return 0;
}

int str2int64 (unsigned char *str, unsigned long long *x, int n) {
  for (int i = 0; i < n; i++) {
    int j = 8*i;
    x[i] = str[j] * (1LL<<56) + str[j+1] * (1LL<<48) + str[j+2] * (1LL<<40) + str[j+3] * (1LL<<32)
         + str[j+4] * (1LL<<24) + str[j+5] * (1LL<<16) + str[j+6] * (1LL<<8) + str[j+7];
  }
  return 0;
}


int int8toStr (unsigned char *x, unsigned char *str, int n) {
  for (int i = 0; i < n; i++) {
    str[i] = x[i];
  }
  return 0;
}

// Convert an array to a string (big endian encoding)
int int32toStr (unsigned int *x, unsigned char *str, int n) {
  for (int i = 0; i < n; i++) {
    int j = 4*i;
    str[j]   = x[i]>>24;
    str[j+1] = x[i]>>16;
    str[j+2] = x[i]>>8;
    str[j+3] = x[i];
  }
  return 0;
}

int int64toStr (unsigned long long *x, unsigned char *str, int n) {
  for (int i = 0; i < n; i++) {
    int j = 8*i;
    str[j]   = x[i]>>56;
    str[j+1] = x[i]>>48;
    str[j+2] = x[i]>>40;
    str[j+3] = x[i]>>32;
    str[j+4] = x[i]>>24;
    str[j+5] = x[i]>>16;
    str[j+6] = x[i]>>8;
    str[j+7] = x[i];
  }
  return 0;
}
