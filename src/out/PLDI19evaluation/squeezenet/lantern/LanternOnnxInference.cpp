#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <math.h>
#include <memory>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cblas.h>
#include <algorithm>
#include <numeric>

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

long fsize(int fd) {
  struct stat stat;
  int res = fstat(fd, &stat);
  return stat.st_size;
}

int printll(char *s) {
  while (*s != '\n' && *s != ',' && *s != '\t') {
    putchar(*s++);
  }
  return 0;
}

long hash(char *str0, int len) {
  unsigned char *str = (unsigned char *)str0;
  unsigned long hash = 5381;
  int c;

  while ((c = *str++) && len--)
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}

long HEAP_SIZE_CPU = 1073741826; // 1048576; // 536870912; // 268435456; // 2097152; 1610612739; // 4294967304; //
void *mallocBase = calloc(HEAP_SIZE_CPU, 1);
void *mallocAddr = mallocBase;
void *waterMark = mallocBase;
void *myMalloc(size_t bytes) {
  void *res = mallocAddr;
  mallocAddr = (void *)((char *)mallocAddr + bytes);
  if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE_CPU)
    fprintf(stderr, "CPU memory breached limit of HEAP_SIZE_CPU\n");
  return res;
}

long HEAP_SIZE = 8589934608; //  4294967304; // this is for GPU

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
  long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
  result->tv_sec = diff / 1000000;
  result->tv_usec = diff % 1000000;
  return (diff < 0);
}



void Snippet(char *);

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<> d{0, 0.01};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("usage: query <filename>\n");
    return 0;
  }
  Snippet(argv[1]);
  return 0;
}

/*****************************************
  Emitting C Generated Code                  
*******************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
void Snippet(char*  x0) {
// Backend setup.
srand(42);
struct timeval begin_0, end_0, diff_0;
gettimeofday(&begin_0, NULL);
int32_t x5 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
int64_t x6 = fsize(x5);
int64_t x8 = x6 / 3073LL;
int32_t x9 = (int32_t)x8;
int32_t x10 = x9 * 3072;
float* x11 = (float*)myMalloc(x10 * sizeof(float));;
int* x12 = (int32_t*)myMalloc(x9 * sizeof(int32_t));;
char* x7 = (char*)mmap(0, x6, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x5, 0);
for(int x14=0; x14 < x9; x14++) {
int32_t x15 = x14 * 3073;
char x16 = x7[x15];
int32_t x17 = (int32_t)(unsigned char)x16;
x12[x14] = x17;
int32_t x23 = x15 + 1;
int32_t x21 = x14 * 3072;
for(int x20=0; x20 < 3072; x20++) {
int32_t x24 = x23 + x20;
char x25 = x7[x24];
int32_t x22 = x21 + x20;
float x26 = (float)(unsigned char)x25;
float x27 = x26 / 255.0f;
x11[x22] = x27;

}

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x35 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x36 = (float)x35;
float x37 = x36 / 1000000.0f;
printf("Data reading in %lf sec\n",x37);
int64_t x95 = (long)mallocAddr;
// inferencing loop starts here
int32_t x103 = x9 / 64;
int32_t x110 = 31 / 1;
int32_t x111 = x110 + 1;
int32_t x115 = 6144 * x111;
int32_t x116 = x115 * x111;
int32_t x112 = x111 * x111;
int32_t x39 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/squeezenet/squeezenetCifar10.onnx.bin",0);
int64_t x40 = fsize(x39);
float* x41 = (float*)mmap(0, x40, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x39, 0);
float* x85 = x41+2592;
int32_t x137 = 1728 * x112;
int32_t x113 = 96 * x112;
int32_t x135 = 27 * x112;
float* x75 = x41+0;
int32_t x114 = 64 * x113;
bool x237 = x111 >= 2;
bool x238;
if (x237) {
x238 = x237;
} else {
x238 = false;
}
int32_t x243 = x111 - 2;
int32_t x244 = x243 / 2;
int32_t x245 = x244 + 1;
int32_t x249 = 6144 * x245;
int32_t x250 = x249 * x245;
int32_t x246 = x245 * x245;
int32_t x247 = 96 * x246;
int32_t x248 = 64 * x247;
int32_t x336 = 2 * x111;
int32_t x346 = x244 / 1;
int32_t x347 = x346 + 1;
int32_t x351 = 1024 * x347;
int32_t x352 = x351 * x347;
int32_t x348 = x347 * x347;
float* x50 = x41+4224;
int32_t x372 = 6144 * x348;
int32_t x349 = 16 * x348;
int32_t x370 = 96 * x348;
float* x92 = x41+2688;
int32_t x350 = 64 * x349;
int32_t x427 = x346 / 1;
int32_t x428 = x427 + 1;
int32_t x432 = 4096 * x428;
int32_t x433 = x432 * x428;
int32_t x429 = x428 * x428;
float* x73 = x41+5264;
int32_t x452 = 1024 * x429;
int32_t x430 = 64 * x429;
int32_t x450 = 16 * x429;
float* x66 = x41+4240;
int32_t x431 = 64 * x430;
int32_t x507 = x347 + 2;
int32_t x508 = x507 - 3;
int32_t x509 = x508 / 1;
int32_t x510 = x509 + 1;
int32_t x514 = 4096 * x510;
int32_t x515 = x514 * x510;
int32_t x511 = x510 * x510;
float* x47 = x41+14544;
int32_t x534 = 9216 * x511;
int32_t x512 = 64 * x511;
int32_t x532 = 144 * x511;
float* x89 = x41+5328;
int32_t x513 = 64 * x512;
bool x634 = true || false;
bool x636;
if (x634) {
bool x635 = true || true;
x636 = x635;
} else {
x636 = false;
}
bool x639;
if (x636) {
bool x637 = x510 == x428;
bool x638 = x637 || false;
x639 = x638;
} else {
x639 = false;
}
bool x640;
if (x639) {
bool x637 = x510 == x428;
bool x638 = x637 || false;
x640 = x638;
} else {
x640 = false;
}
int32_t x649 = 8192 * x428;
int32_t x650 = x649 * x428;
int32_t x676 = x427 / 1;
int32_t x677 = x676 + 1;
int32_t x681 = 1024 * x677;
int32_t x682 = x681 * x677;
int32_t x678 = x677 * x677;
float* x67 = x41+16656;
int32_t x701 = 8192 * x678;
int32_t x647 = 128 * x429;
int32_t x679 = 16 * x678;
int32_t x699 = 128 * x678;
float* x54 = x41+14608;
int32_t x680 = 64 * x679;
int32_t x757 = x676 / 1;
int32_t x758 = x757 + 1;
int32_t x762 = 4096 * x758;
int32_t x763 = x762 * x758;
int32_t x759 = x758 * x758;
float* x45 = x41+17696;
int32_t x782 = 1024 * x759;
int32_t x760 = 64 * x759;
int32_t x780 = 16 * x759;
float* x53 = x41+16672;
int32_t x761 = 64 * x760;
int32_t x837 = x677 + 2;
int32_t x838 = x837 - 3;
int32_t x839 = x838 / 1;
int32_t x840 = x839 + 1;
int32_t x844 = 4096 * x840;
int32_t x845 = x844 * x840;
int32_t x841 = x840 * x840;
float* x79 = x41+26976;
int32_t x864 = 9216 * x841;
int32_t x842 = 64 * x841;
int32_t x862 = 144 * x841;
float* x61 = x41+17760;
int32_t x843 = 64 * x842;
bool x965;
if (x636) {
bool x963 = x840 == x758;
bool x964 = x963 || false;
x965 = x964;
} else {
x965 = false;
}
bool x966;
if (x965) {
bool x963 = x840 == x758;
bool x964 = x963 || false;
x966 = x964;
} else {
x966 = false;
}
int32_t x975 = 8192 * x758;
int32_t x976 = x975 * x758;
int32_t x1002 = x757 / 1;
int32_t x1003 = x1002 + 1;
int32_t x1007 = 2048 * x1003;
int32_t x1008 = x1007 * x1003;
int32_t x1004 = x1003 * x1003;
float* x65 = x41+31136;
int32_t x1028 = 8192 * x1004;
int32_t x973 = 128 * x759;
int32_t x1005 = 32 * x1004;
int32_t x1026 = 128 * x1004;
float* x52 = x41+27040;
int32_t x1006 = 64 * x1005;
int32_t x1083 = x1002 / 1;
int32_t x1084 = x1083 + 1;
int32_t x1088 = 8192 * x1084;
int32_t x1089 = x1088 * x1084;
int32_t x1085 = x1084 * x1084;
float* x87 = x41+35264;
int32_t x1108 = 2048 * x1085;
int32_t x1086 = 128 * x1085;
int32_t x1106 = 32 * x1085;
float* x77 = x41+31168;
int32_t x1087 = 64 * x1086;
int32_t x1163 = x1003 + 2;
int32_t x1164 = x1163 - 3;
int32_t x1165 = x1164 / 1;
int32_t x1166 = x1165 + 1;
int32_t x1170 = 8192 * x1166;
int32_t x1171 = x1170 * x1166;
int32_t x1167 = x1166 * x1166;
float* x83 = x41+72256;
int32_t x1190 = 18432 * x1167;
int32_t x1168 = 128 * x1167;
int32_t x1188 = 288 * x1167;
float* x48 = x41+35392;
int32_t x1169 = 64 * x1168;
bool x1292;
if (x636) {
bool x1290 = x1166 == x1084;
bool x1291 = x1290 || false;
x1292 = x1291;
} else {
x1292 = false;
}
bool x1293;
if (x1292) {
bool x1290 = x1166 == x1084;
bool x1291 = x1290 || false;
x1293 = x1291;
} else {
x1293 = false;
}
int32_t x1302 = 16384 * x1084;
int32_t x1303 = x1302 * x1084;
bool x1329 = x1084 >= 2;
bool x1330;
if (x1329) {
x1330 = x1329;
} else {
x1330 = false;
}
int32_t x1335 = x1084 - 2;
int32_t x1336 = x1335 / 2;
int32_t x1337 = x1336 + 1;
int32_t x1341 = 16384 * x1337;
int32_t x1342 = x1341 * x1337;
int32_t x1338 = x1337 * x1337;
int32_t x1339 = 256 * x1338;
int32_t x1340 = 64 * x1339;
int32_t x1300 = 256 * x1085;
int32_t x1429 = 2 * x1084;
int32_t x1439 = x1336 / 1;
int32_t x1440 = x1439 + 1;
int32_t x1444 = 2048 * x1440;
int32_t x1445 = x1444 * x1440;
int32_t x1441 = x1440 * x1440;
float* x57 = x41+80576;
int32_t x1464 = 16384 * x1441;
int32_t x1442 = 32 * x1441;
int32_t x1462 = 256 * x1441;
float* x69 = x41+72384;
int32_t x1443 = 64 * x1442;
int32_t x1519 = x1439 / 1;
int32_t x1520 = x1519 + 1;
int32_t x1524 = 8192 * x1520;
int32_t x1525 = x1524 * x1520;
int32_t x1521 = x1520 * x1520;
float* x63 = x41+84704;
int32_t x1544 = 2048 * x1521;
int32_t x1522 = 128 * x1521;
int32_t x1542 = 32 * x1521;
float* x49 = x41+80608;
int32_t x1523 = 64 * x1522;
int32_t x1599 = x1440 + 2;
int32_t x1600 = x1599 - 3;
int32_t x1601 = x1600 / 1;
int32_t x1602 = x1601 + 1;
int32_t x1606 = 8192 * x1602;
int32_t x1607 = x1606 * x1602;
int32_t x1603 = x1602 * x1602;
float* x58 = x41+121696;
int32_t x1626 = 18432 * x1603;
int32_t x1604 = 128 * x1603;
int32_t x1624 = 288 * x1603;
float* x78 = x41+84832;
int32_t x1605 = 64 * x1604;
bool x1727;
if (x636) {
bool x1725 = x1602 == x1520;
bool x1726 = x1725 || false;
x1727 = x1726;
} else {
x1727 = false;
}
bool x1728;
if (x1727) {
bool x1725 = x1602 == x1520;
bool x1726 = x1725 || false;
x1728 = x1726;
} else {
x1728 = false;
}
int32_t x1737 = 16384 * x1520;
int32_t x1738 = x1737 * x1520;
int32_t x1764 = x1519 / 1;
int32_t x1765 = x1764 + 1;
int32_t x1769 = 3072 * x1765;
int32_t x1770 = x1769 * x1765;
int32_t x1766 = x1765 * x1765;
float* x94 = x41+134112;
int32_t x1790 = 16384 * x1766;
int32_t x1735 = 256 * x1521;
int32_t x1767 = 48 * x1766;
int32_t x1788 = 256 * x1766;
float* x84 = x41+121824;
int32_t x1768 = 64 * x1767;
int32_t x1845 = x1764 / 1;
int32_t x1846 = x1845 + 1;
int32_t x1850 = 12288 * x1846;
int32_t x1851 = x1850 * x1846;
int32_t x1847 = x1846 * x1846;
float* x88 = x41+143376;
int32_t x1871 = 3072 * x1847;
int32_t x1848 = 192 * x1847;
int32_t x1869 = 48 * x1847;
float* x90 = x41+134160;
int32_t x1849 = 64 * x1848;
int32_t x1926 = x1765 + 2;
int32_t x1927 = x1926 - 3;
int32_t x1928 = x1927 / 1;
int32_t x1929 = x1928 + 1;
int32_t x1933 = 12288 * x1929;
int32_t x1934 = x1933 * x1929;
int32_t x1930 = x1929 * x1929;
float* x71 = x41+226512;
int32_t x1953 = 27648 * x1930;
int32_t x1931 = 192 * x1930;
int32_t x1951 = 432 * x1930;
float* x81 = x41+143568;
int32_t x1932 = 64 * x1931;
bool x2055;
if (x636) {
bool x2053 = x1929 == x1846;
bool x2054 = x2053 || false;
x2055 = x2054;
} else {
x2055 = false;
}
bool x2056;
if (x2055) {
bool x2053 = x1929 == x1846;
bool x2054 = x2053 || false;
x2056 = x2054;
} else {
x2056 = false;
}
int32_t x2065 = 24576 * x1846;
int32_t x2066 = x2065 * x1846;
int32_t x2092 = x1845 / 1;
int32_t x2093 = x2092 + 1;
int32_t x2097 = 3072 * x2093;
int32_t x2098 = x2097 * x2093;
int32_t x2094 = x2093 * x2093;
float* x44 = x41+245136;
int32_t x2117 = 24576 * x2094;
int32_t x2063 = 384 * x1847;
int32_t x2095 = 48 * x2094;
int32_t x2115 = 384 * x2094;
float* x56 = x41+226704;
int32_t x2096 = 64 * x2095;
int32_t x2173 = x2092 / 1;
int32_t x2174 = x2173 + 1;
int32_t x2178 = 12288 * x2174;
int32_t x2179 = x2178 * x2174;
int32_t x2175 = x2174 * x2174;
float* x74 = x41+254400;
int32_t x2198 = 3072 * x2175;
int32_t x2176 = 192 * x2175;
int32_t x2196 = 48 * x2175;
float* x64 = x41+245184;
int32_t x2177 = 64 * x2176;
int32_t x2253 = x2093 + 2;
int32_t x2254 = x2253 - 3;
int32_t x2255 = x2254 / 1;
int32_t x2256 = x2255 + 1;
int32_t x2260 = 12288 * x2256;
int32_t x2261 = x2260 * x2256;
int32_t x2257 = x2256 * x2256;
float* x86 = x41+337536;
int32_t x2280 = 27648 * x2257;
int32_t x2258 = 192 * x2257;
int32_t x2278 = 432 * x2257;
float* x60 = x41+254592;
int32_t x2259 = 64 * x2258;
bool x2381;
if (x636) {
bool x2379 = x2256 == x2174;
bool x2380 = x2379 || false;
x2381 = x2380;
} else {
x2381 = false;
}
bool x2382;
if (x2381) {
bool x2379 = x2256 == x2174;
bool x2380 = x2379 || false;
x2382 = x2380;
} else {
x2382 = false;
}
int32_t x2391 = 24576 * x2174;
int32_t x2392 = x2391 * x2174;
int32_t x2418 = x2173 / 1;
int32_t x2419 = x2418 + 1;
int32_t x2423 = 4096 * x2419;
int32_t x2424 = x2423 * x2419;
int32_t x2420 = x2419 * x2419;
float* x51 = x41+362304;
int32_t x2443 = 24576 * x2420;
int32_t x2389 = 384 * x2175;
int32_t x2421 = 64 * x2420;
int32_t x2441 = 384 * x2420;
float* x76 = x41+337728;
int32_t x2422 = 64 * x2421;
int32_t x2498 = x2418 / 1;
int32_t x2499 = x2498 + 1;
int32_t x2503 = 16384 * x2499;
int32_t x2504 = x2503 * x2499;
int32_t x2500 = x2499 * x2499;
float* x82 = x41+378752;
int32_t x2523 = 4096 * x2500;
int32_t x2501 = 256 * x2500;
int32_t x2521 = 64 * x2500;
float* x91 = x41+362368;
int32_t x2502 = 64 * x2501;
int32_t x2578 = x2419 + 2;
int32_t x2579 = x2578 - 3;
int32_t x2580 = x2579 / 1;
int32_t x2581 = x2580 + 1;
int32_t x2585 = 16384 * x2581;
int32_t x2586 = x2585 * x2581;
int32_t x2582 = x2581 * x2581;
float* x55 = x41+526464;
int32_t x2605 = 36864 * x2582;
int32_t x2583 = 256 * x2582;
int32_t x2603 = 576 * x2582;
float* x70 = x41+379008;
int32_t x2584 = 64 * x2583;
bool x2707;
if (x636) {
bool x2705 = x2581 == x2499;
bool x2706 = x2705 || false;
x2707 = x2706;
} else {
x2707 = false;
}
bool x2708;
if (x2707) {
bool x2705 = x2581 == x2499;
bool x2706 = x2705 || false;
x2708 = x2706;
} else {
x2708 = false;
}
int32_t x2717 = 32768 * x2499;
int32_t x2718 = x2717 * x2499;
bool x2744 = x2499 >= 2;
bool x2745;
if (x2744) {
x2745 = x2744;
} else {
x2745 = false;
}
int32_t x2750 = x2499 - 2;
int32_t x2751 = x2750 / 2;
int32_t x2752 = x2751 + 1;
int32_t x2756 = 32768 * x2752;
int32_t x2757 = x2756 * x2752;
int32_t x2753 = x2752 * x2752;
int32_t x2754 = 512 * x2753;
int32_t x2755 = 64 * x2754;
int32_t x2715 = 512 * x2500;
int32_t x2844 = 2 * x2499;
int32_t x2854 = x2751 / 1;
int32_t x2855 = x2854 + 1;
int32_t x2859 = 4096 * x2855;
int32_t x2860 = x2859 * x2855;
int32_t x2856 = x2855 * x2855;
float* x62 = x41+559488;
int32_t x2879 = 32768 * x2856;
int32_t x2857 = 64 * x2856;
int32_t x2877 = 512 * x2856;
float* x43 = x41+526720;
int32_t x2858 = 64 * x2857;
int32_t x2934 = x2854 / 1;
int32_t x2935 = x2934 + 1;
int32_t x2939 = 16384 * x2935;
int32_t x2940 = x2939 * x2935;
int32_t x2936 = x2935 * x2935;
float* x68 = x41+575936;
int32_t x2959 = 4096 * x2936;
int32_t x2937 = 256 * x2936;
int32_t x2957 = 64 * x2936;
float* x80 = x41+559552;
int32_t x2938 = 64 * x2937;
int32_t x3014 = x2855 + 2;
int32_t x3015 = x3014 - 3;
int32_t x3016 = x3015 / 1;
int32_t x3017 = x3016 + 1;
int32_t x3021 = 16384 * x3017;
int32_t x3022 = x3021 * x3017;
int32_t x3018 = x3017 * x3017;
float* x59 = x41+723648;
int32_t x3041 = 36864 * x3018;
int32_t x3019 = 256 * x3018;
int32_t x3039 = 576 * x3018;
float* x72 = x41+576192;
int32_t x3020 = 64 * x3019;
bool x3142;
if (x636) {
bool x3140 = x3017 == x2935;
bool x3141 = x3140 || false;
x3142 = x3141;
} else {
x3142 = false;
}
bool x3143;
if (x3142) {
bool x3140 = x3017 == x2935;
bool x3141 = x3140 || false;
x3143 = x3141;
} else {
x3143 = false;
}
int32_t x3152 = 32768 * x2935;
int32_t x3153 = x3152 * x2935;
int32_t x3179 = x2935 - 4;
int32_t x3180 = x3179 / 1;
int32_t x3181 = x3180 + 1;
int32_t x3185 = 640 * x3181;
int32_t x3186 = x3185 * x3181;
int32_t x3182 = x3181 * x3181;
float* x93 = x41+805824;
int32_t x3206 = 524288 * x3182;
int32_t x3150 = 512 * x2936;
int32_t x3183 = 10 * x3182;
int32_t x3204 = 8192 * x3182;
float* x46 = x41+723904;
int32_t x3184 = 64 * x3183;
int64_t x3282 = (int64_t)x9;
for(int x98=0; x98 < 4; x98++) {
struct timeval begin_1, end_1, diff_1;
int32_t x100 = x98 + 1;
printf("Start inferencing epoch %d\n",x100);
gettimeofday(&begin_1, NULL);
for(int x105=0; x105 < x103; x105++) {
int32_t x106 = x105 * 64;
int32_t x107 = x106 * 3072;
float* x108 = x11+x107;
int* x109 = x12+x106;
float* x117 = (float*)myMalloc(x116 * sizeof(float));;
int32_t x118 = 0;
for(int x120=0; x120 < 64; x120++) {
for(int x122=0; x122 < 96; x122++) {
for(int x124=0; x124 < x112; x124++) {
int32_t x125 = x118;
float x126 = x85[x122];
x117[x125] = x126;
x118 += 1;

}

}

}
float* x138 = (float*)myMalloc(x137 * sizeof(float));;
for(int x139=0; x139 < 64; x139++) {
int32_t x140 = x139 * 3072;
float* x141 = x108+x140;
int32_t x142 = x139 * x113;
float* x143 = x117+x142;
int32_t x144 = x139 * x135;
float* x145 = x138+x144;
for(int x147=0; x147 < 27; x147++) {
int32_t x148 = x147 / 9;
int32_t x152 = x148 * 3;
int32_t x153 = x152 * 3;
int32_t x154 = x153 * x111;
int32_t x155 = x154 * x111;
int32_t x149 = x147 % 9;
int32_t x150 = x149 / 3;
int32_t x156 = x150 * 3;
int32_t x157 = x156 * x111;
int32_t x158 = x157 * x111;
int32_t x159 = x155 + x158;
int32_t x151 = x149 % 3;
int32_t x160 = x151 * x111;
int32_t x161 = x160 * x111;
int32_t x162 = x159 + x161;
float* x163 = x145+x162;
int32_t x164 = x148 * 32;
int32_t x165 = x164 * 32;
float* x166 = x141+x165;
int32_t x179 = 1 - x151;
bool x180 = x179 > 0;
int32_t x181;
if (x180) {
x181 = x179;
} else {
x181 = 0;
}
int32_t x182 = 3 - x151;
int32_t x183 = x182 - 1;
int32_t x184 = 1 - x183;
bool x185 = x184 > 0;
int32_t x186;
if (x185) {
x186 = x184;
} else {
x186 = 0;
}
int32_t x187 = x111 - x186;
int32_t x188 = x187 - x181;
bool x189 = x188 <= 0;
bool x193 = x181 > 0;
int32_t x178 = -1 + x151;
bool x206 = x186 > 0;
for(int x168=0; x168 < x111; x168++) {
int32_t x169 = x168 - 1;
int32_t x170 = x169 + x150;
bool x171 = x170 < 0;
bool x172 = x170 >= 32;
bool x173 = x171 || x172;
if (x173) {
int32_t x174 = x168 * x111;
float* x175 = x163+x174;
memset(x175, 0, 4 * x111);;
} else {
if (x189) {
int32_t x174 = x168 * x111;
float* x190 = x163+x174;
memset(x190, 0, 4 * x111);;
} else {
int32_t x174 = x168 * x111;
if (x193) {
float* x194 = x163+x174;
memset(x194, 0, 4 * x181);;
} else {
}
// may have segfault here
int32_t x199 = x174 + x181;
float* x200 = x163+x199;
int32_t x201 = x170 * 32;
int32_t x202 = x201 + x178;
int32_t x203 = x202 + x181;
float* x204 = x166+x203;
memcpy(x200, x204, 4 * x188);;
if (x206) {
int32_t x207 = x174 + x111;
int32_t x208 = x207 - x186;
float* x209 = x163+x208;
memset(x209, 0, 4 * x186);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 96,x112,27,1,x75,27,x145,x112,1,x143,x112);

}
float* x224 = (float*)myMalloc(x114 * sizeof(float));;
for(int x226=0; x226 < x114; x226++) {
float x227 = x117[x226];
bool x228 = x227 < 0.0f;
if (x228) {
x224[x226] = 0.0f;
} else {
float x231 = x117[x226];
x224[x226] = x231;
}

}
if (x238) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(64) x Const(96) x Sym(111) x Sym(111)|(2,2)");
}
float* x251 = (float*)myMalloc(x250 * sizeof(float));;
for(int x253=0; x253 < x250; x253++) {
x251[x253] = -3.4028235E38f;

}
int* x257 = (int32_t*)myMalloc(x248 * sizeof(int32_t));;
for(int x258=0; x258 < 64; x258++) {
int32_t x259 = x258 * x113;
float* x260 = x224+x259;
int32_t x261 = x258 * x247;
float* x262 = x251+x261;
int* x263 = x257+x261;
int32_t x264 = 0;
int32_t x265 = 0;
for(int x266=0; x266 < 96; x266++) {
int32_t x267 = x264;
int32_t x268 = x267;
int32_t x269 = x265;
int32_t x270 = x269;
for(int x272=0; x272 < x245; x272++) {
int32_t x273 = x268;
int32_t x274 = x273;
int32_t x275 = x270;
int32_t x276 = x275;
for(int x277=0; x277 < x245; x277++) {
int32_t x278 = x276;
int32_t x279 = x278;
int32_t x280 = x279;
int32_t x281 = x280;
int32_t x282 = x281;
float x283 = x260[x282];
int32_t x284 = x274;
float x285 = x262[x284];
bool x286 = x283 > x285;
if (x286) {
float x287 = x260[x282];
x262[x284] = x287;
int32_t x289 = x282 + x259;
x263[x284] = x289;
} else {
}
x281 += 1;
int32_t x294 = x281;
float x295 = x260[x294];
float x296 = x262[x284];
bool x297 = x295 > x296;
if (x297) {
float x298 = x260[x294];
x262[x284] = x298;
int32_t x300 = x294 + x259;
x263[x284] = x300;
} else {
}
x281 += 1;
x279 += x111;
int32_t x306 = x279;
int32_t x307 = x306;
int32_t x308 = x307;
float x309 = x260[x308];
float x310 = x262[x284];
bool x311 = x309 > x310;
if (x311) {
float x312 = x260[x308];
x262[x284] = x312;
int32_t x314 = x308 + x259;
x263[x284] = x314;
} else {
}
x307 += 1;
int32_t x319 = x307;
float x320 = x260[x319];
float x321 = x262[x284];
bool x322 = x320 > x321;
if (x322) {
float x323 = x260[x319];
x262[x284] = x323;
int32_t x325 = x319 + x259;
x263[x284] = x325;
} else {
}
x307 += 1;
x279 += x111;
x274 += 1;
x276 += 2;

}
x268 += x245;
x270 += x336;

}
x264 += x246;
x265 += x112;

}

}
float* x353 = (float*)myMalloc(x352 * sizeof(float));;
int32_t x354 = 0;
for(int x355=0; x355 < 64; x355++) {
for(int x357=0; x357 < 16; x357++) {
for(int x359=0; x359 < x348; x359++) {
int32_t x360 = x354;
float x361 = x50[x357];
x353[x360] = x361;
x354 += 1;

}

}

}
float* x373 = (float*)myMalloc(x372 * sizeof(float));;
for(int x374=0; x374 < 64; x374++) {
int32_t x375 = x374 * x247;
float* x376 = x251+x375;
int32_t x377 = x374 * x349;
float* x378 = x353+x377;
int32_t x379 = x374 * x370;
float* x380 = x373+x379;
for(int x381=0; x381 < 96; x381++) {
int32_t x382 = x381 / 1;
int32_t x386 = x382 * x347;
int32_t x387 = x386 * x347;
int32_t x383 = x381 % 1;
int32_t x384 = x383 / 1;
int32_t x388 = x384 * x347;
int32_t x389 = x388 * x347;
int32_t x390 = x387 + x389;
int32_t x385 = x383 % 1;
int32_t x391 = x385 * x347;
int32_t x392 = x391 * x347;
int32_t x393 = x390 + x392;
float* x394 = x380+x393;
int32_t x395 = x382 * x245;
int32_t x396 = x395 * x245;
float* x397 = x376+x396;
for(int x399=0; x399 < x347; x399++) {
int32_t x401 = x399 * x347;
float* x402 = x394+x401;
int32_t x400 = x399 + x384;
int32_t x403 = x400 * x245;
int32_t x404 = x403 + x385;
float* x405 = x397+x404;
memcpy(x402, x405, 4 * x347);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 16,x348,96,1,x92,96,x380,x348,1,x378,x348);

}
float* x414 = (float*)myMalloc(x350 * sizeof(float));;
for(int x416=0; x416 < x350; x416++) {
float x417 = x353[x416];
bool x418 = x417 < 0.0f;
if (x418) {
x414[x416] = 0.0f;
} else {
float x421 = x353[x416];
x414[x416] = x421;
}

}
float* x434 = (float*)myMalloc(x433 * sizeof(float));;
int32_t x435 = 0;
for(int x436=0; x436 < 64; x436++) {
for(int x437=0; x437 < 64; x437++) {
for(int x439=0; x439 < x429; x439++) {
int32_t x440 = x435;
float x441 = x73[x437];
x434[x440] = x441;
x435 += 1;

}

}

}
float* x453 = (float*)myMalloc(x452 * sizeof(float));;
for(int x454=0; x454 < 64; x454++) {
int32_t x455 = x454 * x349;
float* x456 = x414+x455;
int32_t x457 = x454 * x430;
float* x458 = x434+x457;
int32_t x459 = x454 * x450;
float* x460 = x453+x459;
for(int x461=0; x461 < 16; x461++) {
int32_t x462 = x461 / 1;
int32_t x466 = x462 * x428;
int32_t x467 = x466 * x428;
int32_t x463 = x461 % 1;
int32_t x464 = x463 / 1;
int32_t x468 = x464 * x428;
int32_t x469 = x468 * x428;
int32_t x470 = x467 + x469;
int32_t x465 = x463 % 1;
int32_t x471 = x465 * x428;
int32_t x472 = x471 * x428;
int32_t x473 = x470 + x472;
float* x474 = x460+x473;
int32_t x475 = x462 * x347;
int32_t x476 = x475 * x347;
float* x477 = x456+x476;
for(int x479=0; x479 < x428; x479++) {
int32_t x481 = x479 * x428;
float* x482 = x474+x481;
int32_t x480 = x479 + x464;
int32_t x483 = x480 * x347;
int32_t x484 = x483 + x465;
float* x485 = x477+x484;
memcpy(x482, x485, 4 * x428);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x429,16,1,x66,16,x460,x429,1,x458,x429);

}
float* x494 = (float*)myMalloc(x431 * sizeof(float));;
for(int x496=0; x496 < x431; x496++) {
float x497 = x434[x496];
bool x498 = x497 < 0.0f;
if (x498) {
x494[x496] = 0.0f;
} else {
float x501 = x434[x496];
x494[x496] = x501;
}

}
float* x516 = (float*)myMalloc(x515 * sizeof(float));;
int32_t x517 = 0;
for(int x518=0; x518 < 64; x518++) {
for(int x519=0; x519 < 64; x519++) {
for(int x521=0; x521 < x511; x521++) {
int32_t x522 = x517;
float x523 = x47[x519];
x516[x522] = x523;
x517 += 1;

}

}

}
float* x535 = (float*)myMalloc(x534 * sizeof(float));;
for(int x536=0; x536 < 64; x536++) {
int32_t x537 = x536 * x349;
float* x538 = x414+x537;
int32_t x539 = x536 * x512;
float* x540 = x516+x539;
int32_t x541 = x536 * x532;
float* x542 = x535+x541;
for(int x544=0; x544 < 144; x544++) {
int32_t x545 = x544 / 9;
int32_t x549 = x545 * 3;
int32_t x550 = x549 * 3;
int32_t x551 = x550 * x510;
int32_t x552 = x551 * x510;
int32_t x546 = x544 % 9;
int32_t x547 = x546 / 3;
int32_t x553 = x547 * 3;
int32_t x554 = x553 * x510;
int32_t x555 = x554 * x510;
int32_t x556 = x552 + x555;
int32_t x548 = x546 % 3;
int32_t x557 = x548 * x510;
int32_t x558 = x557 * x510;
int32_t x559 = x556 + x558;
float* x560 = x542+x559;
int32_t x561 = x545 * x347;
int32_t x562 = x561 * x347;
float* x563 = x538+x562;
int32_t x576 = 1 - x548;
bool x577 = x576 > 0;
int32_t x578;
if (x577) {
x578 = x576;
} else {
x578 = 0;
}
int32_t x579 = 3 - x548;
int32_t x580 = x579 - 1;
int32_t x581 = 1 - x580;
bool x582 = x581 > 0;
int32_t x583;
if (x582) {
x583 = x581;
} else {
x583 = 0;
}
int32_t x584 = x510 - x583;
int32_t x585 = x584 - x578;
bool x586 = x585 <= 0;
bool x590 = x578 > 0;
int32_t x575 = -1 + x548;
bool x603 = x583 > 0;
for(int x565=0; x565 < x510; x565++) {
int32_t x566 = x565 - 1;
int32_t x567 = x566 + x547;
bool x568 = x567 < 0;
bool x569 = x567 >= x347;
bool x570 = x568 || x569;
if (x570) {
int32_t x571 = x565 * x510;
float* x572 = x560+x571;
memset(x572, 0, 4 * x510);;
} else {
if (x586) {
int32_t x571 = x565 * x510;
float* x587 = x560+x571;
memset(x587, 0, 4 * x510);;
} else {
int32_t x571 = x565 * x510;
if (x590) {
float* x591 = x560+x571;
memset(x591, 0, 4 * x578);;
} else {
}
// may have segfault here
int32_t x596 = x571 + x578;
float* x597 = x560+x596;
int32_t x598 = x567 * x347;
int32_t x599 = x598 + x575;
int32_t x600 = x599 + x578;
float* x601 = x563+x600;
memcpy(x597, x601, 4 * x585);;
if (x603) {
int32_t x604 = x571 + x510;
int32_t x605 = x604 - x583;
float* x606 = x560+x605;
memset(x606, 0, 4 * x583);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x511,144,1,x89,144,x542,x511,1,x540,x511);

}
float* x621 = (float*)myMalloc(x513 * sizeof(float));;
for(int x623=0; x623 < x513; x623++) {
float x624 = x516[x623];
bool x625 = x624 < 0.0f;
if (x625) {
x621[x623] = 0.0f;
} else {
float x628 = x516[x623];
x621[x623] = x628;
}

}
if (x640) {
} else {
printf("all dimensions except the concatenation dimension should be the same\n");
assert(false && "");
}
// back prop for concat
float* x651 = (float*)myMalloc(x650 * sizeof(float));;
int32_t x652 = 0;
for(int x653=0; x653 < 64; x653++) {
int32_t x654 = x653 * x430;
float* x655 = x494+x654;
for(int x657=0; x657 < x430; x657++) {
int32_t x658 = x652;
float x659 = x655[x657];
x651[x658] = x659;
x652 += 1;

}
int32_t x664 = x653 * x512;
float* x665 = x621+x664;
for(int x667=0; x667 < x512; x667++) {
int32_t x668 = x652;
float x669 = x665[x667];
x651[x668] = x669;
x652 += 1;

}

}
float* x683 = (float*)myMalloc(x682 * sizeof(float));;
int32_t x684 = 0;
for(int x685=0; x685 < 64; x685++) {
for(int x686=0; x686 < 16; x686++) {
for(int x688=0; x688 < x678; x688++) {
int32_t x689 = x684;
float x690 = x67[x686];
x683[x689] = x690;
x684 += 1;

}

}

}
float* x702 = (float*)myMalloc(x701 * sizeof(float));;
for(int x703=0; x703 < 64; x703++) {
int32_t x704 = x703 * x647;
float* x705 = x651+x704;
int32_t x706 = x703 * x679;
float* x707 = x683+x706;
int32_t x708 = x703 * x699;
float* x709 = x702+x708;
for(int x711=0; x711 < 128; x711++) {
int32_t x712 = x711 / 1;
int32_t x716 = x712 * x677;
int32_t x717 = x716 * x677;
int32_t x713 = x711 % 1;
int32_t x714 = x713 / 1;
int32_t x718 = x714 * x677;
int32_t x719 = x718 * x677;
int32_t x720 = x717 + x719;
int32_t x715 = x713 % 1;
int32_t x721 = x715 * x677;
int32_t x722 = x721 * x677;
int32_t x723 = x720 + x722;
float* x724 = x709+x723;
int32_t x725 = x712 * x428;
int32_t x726 = x725 * x428;
float* x727 = x705+x726;
for(int x729=0; x729 < x677; x729++) {
int32_t x731 = x729 * x677;
float* x732 = x724+x731;
int32_t x730 = x729 + x714;
int32_t x733 = x730 * x428;
int32_t x734 = x733 + x715;
float* x735 = x727+x734;
memcpy(x732, x735, 4 * x677);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 16,x678,128,1,x54,128,x709,x678,1,x707,x678);

}
float* x744 = (float*)myMalloc(x680 * sizeof(float));;
for(int x746=0; x746 < x680; x746++) {
float x747 = x683[x746];
bool x748 = x747 < 0.0f;
if (x748) {
x744[x746] = 0.0f;
} else {
float x751 = x683[x746];
x744[x746] = x751;
}

}
float* x764 = (float*)myMalloc(x763 * sizeof(float));;
int32_t x765 = 0;
for(int x766=0; x766 < 64; x766++) {
for(int x767=0; x767 < 64; x767++) {
for(int x769=0; x769 < x759; x769++) {
int32_t x770 = x765;
float x771 = x45[x767];
x764[x770] = x771;
x765 += 1;

}

}

}
float* x783 = (float*)myMalloc(x782 * sizeof(float));;
for(int x784=0; x784 < 64; x784++) {
int32_t x785 = x784 * x679;
float* x786 = x744+x785;
int32_t x787 = x784 * x760;
float* x788 = x764+x787;
int32_t x789 = x784 * x780;
float* x790 = x783+x789;
for(int x791=0; x791 < 16; x791++) {
int32_t x792 = x791 / 1;
int32_t x796 = x792 * x758;
int32_t x797 = x796 * x758;
int32_t x793 = x791 % 1;
int32_t x794 = x793 / 1;
int32_t x798 = x794 * x758;
int32_t x799 = x798 * x758;
int32_t x800 = x797 + x799;
int32_t x795 = x793 % 1;
int32_t x801 = x795 * x758;
int32_t x802 = x801 * x758;
int32_t x803 = x800 + x802;
float* x804 = x790+x803;
int32_t x805 = x792 * x677;
int32_t x806 = x805 * x677;
float* x807 = x786+x806;
for(int x809=0; x809 < x758; x809++) {
int32_t x811 = x809 * x758;
float* x812 = x804+x811;
int32_t x810 = x809 + x794;
int32_t x813 = x810 * x677;
int32_t x814 = x813 + x795;
float* x815 = x807+x814;
memcpy(x812, x815, 4 * x758);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x759,16,1,x53,16,x790,x759,1,x788,x759);

}
float* x824 = (float*)myMalloc(x761 * sizeof(float));;
for(int x826=0; x826 < x761; x826++) {
float x827 = x764[x826];
bool x828 = x827 < 0.0f;
if (x828) {
x824[x826] = 0.0f;
} else {
float x831 = x764[x826];
x824[x826] = x831;
}

}
float* x846 = (float*)myMalloc(x845 * sizeof(float));;
int32_t x847 = 0;
for(int x848=0; x848 < 64; x848++) {
for(int x849=0; x849 < 64; x849++) {
for(int x851=0; x851 < x841; x851++) {
int32_t x852 = x847;
float x853 = x79[x849];
x846[x852] = x853;
x847 += 1;

}

}

}
float* x865 = (float*)myMalloc(x864 * sizeof(float));;
for(int x866=0; x866 < 64; x866++) {
int32_t x867 = x866 * x679;
float* x868 = x744+x867;
int32_t x869 = x866 * x842;
float* x870 = x846+x869;
int32_t x871 = x866 * x862;
float* x872 = x865+x871;
for(int x873=0; x873 < 144; x873++) {
int32_t x874 = x873 / 9;
int32_t x878 = x874 * 3;
int32_t x879 = x878 * 3;
int32_t x880 = x879 * x840;
int32_t x881 = x880 * x840;
int32_t x875 = x873 % 9;
int32_t x876 = x875 / 3;
int32_t x882 = x876 * 3;
int32_t x883 = x882 * x840;
int32_t x884 = x883 * x840;
int32_t x885 = x881 + x884;
int32_t x877 = x875 % 3;
int32_t x886 = x877 * x840;
int32_t x887 = x886 * x840;
int32_t x888 = x885 + x887;
float* x889 = x872+x888;
int32_t x890 = x874 * x677;
int32_t x891 = x890 * x677;
float* x892 = x868+x891;
int32_t x905 = 1 - x877;
bool x906 = x905 > 0;
int32_t x907;
if (x906) {
x907 = x905;
} else {
x907 = 0;
}
int32_t x908 = 3 - x877;
int32_t x909 = x908 - 1;
int32_t x910 = 1 - x909;
bool x911 = x910 > 0;
int32_t x912;
if (x911) {
x912 = x910;
} else {
x912 = 0;
}
int32_t x913 = x840 - x912;
int32_t x914 = x913 - x907;
bool x915 = x914 <= 0;
bool x919 = x907 > 0;
int32_t x904 = -1 + x877;
bool x932 = x912 > 0;
for(int x894=0; x894 < x840; x894++) {
int32_t x895 = x894 - 1;
int32_t x896 = x895 + x876;
bool x897 = x896 < 0;
bool x898 = x896 >= x677;
bool x899 = x897 || x898;
if (x899) {
int32_t x900 = x894 * x840;
float* x901 = x889+x900;
memset(x901, 0, 4 * x840);;
} else {
if (x915) {
int32_t x900 = x894 * x840;
float* x916 = x889+x900;
memset(x916, 0, 4 * x840);;
} else {
int32_t x900 = x894 * x840;
if (x919) {
float* x920 = x889+x900;
memset(x920, 0, 4 * x907);;
} else {
}
// may have segfault here
int32_t x925 = x900 + x907;
float* x926 = x889+x925;
int32_t x927 = x896 * x677;
int32_t x928 = x927 + x904;
int32_t x929 = x928 + x907;
float* x930 = x892+x929;
memcpy(x926, x930, 4 * x914);;
if (x932) {
int32_t x933 = x900 + x840;
int32_t x934 = x933 - x912;
float* x935 = x889+x934;
memset(x935, 0, 4 * x912);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x841,144,1,x61,144,x872,x841,1,x870,x841);

}
float* x950 = (float*)myMalloc(x843 * sizeof(float));;
for(int x952=0; x952 < x843; x952++) {
float x953 = x846[x952];
bool x954 = x953 < 0.0f;
if (x954) {
x950[x952] = 0.0f;
} else {
float x957 = x846[x952];
x950[x952] = x957;
}

}
if (x966) {
} else {
printf("all dimensions except the concatenation dimension should be the same\n");
assert(false && "");
}
// back prop for concat
float* x977 = (float*)myMalloc(x976 * sizeof(float));;
int32_t x978 = 0;
for(int x979=0; x979 < 64; x979++) {
int32_t x980 = x979 * x760;
float* x981 = x824+x980;
for(int x983=0; x983 < x760; x983++) {
int32_t x984 = x978;
float x985 = x981[x983];
x977[x984] = x985;
x978 += 1;

}
int32_t x990 = x979 * x842;
float* x991 = x950+x990;
for(int x993=0; x993 < x842; x993++) {
int32_t x994 = x978;
float x995 = x991[x993];
x977[x994] = x995;
x978 += 1;

}

}
float* x1009 = (float*)myMalloc(x1008 * sizeof(float));;
int32_t x1010 = 0;
for(int x1011=0; x1011 < 64; x1011++) {
for(int x1013=0; x1013 < 32; x1013++) {
for(int x1015=0; x1015 < x1004; x1015++) {
int32_t x1016 = x1010;
float x1017 = x65[x1013];
x1009[x1016] = x1017;
x1010 += 1;

}

}

}
float* x1029 = (float*)myMalloc(x1028 * sizeof(float));;
for(int x1030=0; x1030 < 64; x1030++) {
int32_t x1031 = x1030 * x973;
float* x1032 = x977+x1031;
int32_t x1033 = x1030 * x1005;
float* x1034 = x1009+x1033;
int32_t x1035 = x1030 * x1026;
float* x1036 = x1029+x1035;
for(int x1037=0; x1037 < 128; x1037++) {
int32_t x1038 = x1037 / 1;
int32_t x1042 = x1038 * x1003;
int32_t x1043 = x1042 * x1003;
int32_t x1039 = x1037 % 1;
int32_t x1040 = x1039 / 1;
int32_t x1044 = x1040 * x1003;
int32_t x1045 = x1044 * x1003;
int32_t x1046 = x1043 + x1045;
int32_t x1041 = x1039 % 1;
int32_t x1047 = x1041 * x1003;
int32_t x1048 = x1047 * x1003;
int32_t x1049 = x1046 + x1048;
float* x1050 = x1036+x1049;
int32_t x1051 = x1038 * x758;
int32_t x1052 = x1051 * x758;
float* x1053 = x1032+x1052;
for(int x1055=0; x1055 < x1003; x1055++) {
int32_t x1057 = x1055 * x1003;
float* x1058 = x1050+x1057;
int32_t x1056 = x1055 + x1040;
int32_t x1059 = x1056 * x758;
int32_t x1060 = x1059 + x1041;
float* x1061 = x1053+x1060;
memcpy(x1058, x1061, 4 * x1003);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 32,x1004,128,1,x52,128,x1036,x1004,1,x1034,x1004);

}
float* x1070 = (float*)myMalloc(x1006 * sizeof(float));;
for(int x1072=0; x1072 < x1006; x1072++) {
float x1073 = x1009[x1072];
bool x1074 = x1073 < 0.0f;
if (x1074) {
x1070[x1072] = 0.0f;
} else {
float x1077 = x1009[x1072];
x1070[x1072] = x1077;
}

}
float* x1090 = (float*)myMalloc(x1089 * sizeof(float));;
int32_t x1091 = 0;
for(int x1092=0; x1092 < 64; x1092++) {
for(int x1093=0; x1093 < 128; x1093++) {
for(int x1095=0; x1095 < x1085; x1095++) {
int32_t x1096 = x1091;
float x1097 = x87[x1093];
x1090[x1096] = x1097;
x1091 += 1;

}

}

}
float* x1109 = (float*)myMalloc(x1108 * sizeof(float));;
for(int x1110=0; x1110 < 64; x1110++) {
int32_t x1111 = x1110 * x1005;
float* x1112 = x1070+x1111;
int32_t x1113 = x1110 * x1086;
float* x1114 = x1090+x1113;
int32_t x1115 = x1110 * x1106;
float* x1116 = x1109+x1115;
for(int x1117=0; x1117 < 32; x1117++) {
int32_t x1118 = x1117 / 1;
int32_t x1122 = x1118 * x1084;
int32_t x1123 = x1122 * x1084;
int32_t x1119 = x1117 % 1;
int32_t x1120 = x1119 / 1;
int32_t x1124 = x1120 * x1084;
int32_t x1125 = x1124 * x1084;
int32_t x1126 = x1123 + x1125;
int32_t x1121 = x1119 % 1;
int32_t x1127 = x1121 * x1084;
int32_t x1128 = x1127 * x1084;
int32_t x1129 = x1126 + x1128;
float* x1130 = x1116+x1129;
int32_t x1131 = x1118 * x1003;
int32_t x1132 = x1131 * x1003;
float* x1133 = x1112+x1132;
for(int x1135=0; x1135 < x1084; x1135++) {
int32_t x1137 = x1135 * x1084;
float* x1138 = x1130+x1137;
int32_t x1136 = x1135 + x1120;
int32_t x1139 = x1136 * x1003;
int32_t x1140 = x1139 + x1121;
float* x1141 = x1133+x1140;
memcpy(x1138, x1141, 4 * x1084);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1085,32,1,x77,32,x1116,x1085,1,x1114,x1085);

}
float* x1150 = (float*)myMalloc(x1087 * sizeof(float));;
for(int x1152=0; x1152 < x1087; x1152++) {
float x1153 = x1090[x1152];
bool x1154 = x1153 < 0.0f;
if (x1154) {
x1150[x1152] = 0.0f;
} else {
float x1157 = x1090[x1152];
x1150[x1152] = x1157;
}

}
float* x1172 = (float*)myMalloc(x1171 * sizeof(float));;
int32_t x1173 = 0;
for(int x1174=0; x1174 < 64; x1174++) {
for(int x1175=0; x1175 < 128; x1175++) {
for(int x1177=0; x1177 < x1167; x1177++) {
int32_t x1178 = x1173;
float x1179 = x83[x1175];
x1172[x1178] = x1179;
x1173 += 1;

}

}

}
float* x1191 = (float*)myMalloc(x1190 * sizeof(float));;
for(int x1192=0; x1192 < 64; x1192++) {
int32_t x1193 = x1192 * x1005;
float* x1194 = x1070+x1193;
int32_t x1195 = x1192 * x1168;
float* x1196 = x1172+x1195;
int32_t x1197 = x1192 * x1188;
float* x1198 = x1191+x1197;
for(int x1200=0; x1200 < 288; x1200++) {
int32_t x1201 = x1200 / 9;
int32_t x1205 = x1201 * 3;
int32_t x1206 = x1205 * 3;
int32_t x1207 = x1206 * x1166;
int32_t x1208 = x1207 * x1166;
int32_t x1202 = x1200 % 9;
int32_t x1203 = x1202 / 3;
int32_t x1209 = x1203 * 3;
int32_t x1210 = x1209 * x1166;
int32_t x1211 = x1210 * x1166;
int32_t x1212 = x1208 + x1211;
int32_t x1204 = x1202 % 3;
int32_t x1213 = x1204 * x1166;
int32_t x1214 = x1213 * x1166;
int32_t x1215 = x1212 + x1214;
float* x1216 = x1198+x1215;
int32_t x1217 = x1201 * x1003;
int32_t x1218 = x1217 * x1003;
float* x1219 = x1194+x1218;
int32_t x1232 = 1 - x1204;
bool x1233 = x1232 > 0;
int32_t x1234;
if (x1233) {
x1234 = x1232;
} else {
x1234 = 0;
}
int32_t x1235 = 3 - x1204;
int32_t x1236 = x1235 - 1;
int32_t x1237 = 1 - x1236;
bool x1238 = x1237 > 0;
int32_t x1239;
if (x1238) {
x1239 = x1237;
} else {
x1239 = 0;
}
int32_t x1240 = x1166 - x1239;
int32_t x1241 = x1240 - x1234;
bool x1242 = x1241 <= 0;
bool x1246 = x1234 > 0;
int32_t x1231 = -1 + x1204;
bool x1259 = x1239 > 0;
for(int x1221=0; x1221 < x1166; x1221++) {
int32_t x1222 = x1221 - 1;
int32_t x1223 = x1222 + x1203;
bool x1224 = x1223 < 0;
bool x1225 = x1223 >= x1003;
bool x1226 = x1224 || x1225;
if (x1226) {
int32_t x1227 = x1221 * x1166;
float* x1228 = x1216+x1227;
memset(x1228, 0, 4 * x1166);;
} else {
if (x1242) {
int32_t x1227 = x1221 * x1166;
float* x1243 = x1216+x1227;
memset(x1243, 0, 4 * x1166);;
} else {
int32_t x1227 = x1221 * x1166;
if (x1246) {
float* x1247 = x1216+x1227;
memset(x1247, 0, 4 * x1234);;
} else {
}
// may have segfault here
int32_t x1252 = x1227 + x1234;
float* x1253 = x1216+x1252;
int32_t x1254 = x1223 * x1003;
int32_t x1255 = x1254 + x1231;
int32_t x1256 = x1255 + x1234;
float* x1257 = x1219+x1256;
memcpy(x1253, x1257, 4 * x1241);;
if (x1259) {
int32_t x1260 = x1227 + x1166;
int32_t x1261 = x1260 - x1239;
float* x1262 = x1216+x1261;
memset(x1262, 0, 4 * x1239);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1167,288,1,x48,288,x1198,x1167,1,x1196,x1167);

}
float* x1277 = (float*)myMalloc(x1169 * sizeof(float));;
for(int x1279=0; x1279 < x1169; x1279++) {
float x1280 = x1172[x1279];
bool x1281 = x1280 < 0.0f;
if (x1281) {
x1277[x1279] = 0.0f;
} else {
float x1284 = x1172[x1279];
x1277[x1279] = x1284;
}

}
if (x1293) {
} else {
printf("all dimensions except the concatenation dimension should be the same\n");
assert(false && "");
}
// back prop for concat
float* x1304 = (float*)myMalloc(x1303 * sizeof(float));;
int32_t x1305 = 0;
for(int x1306=0; x1306 < 64; x1306++) {
int32_t x1307 = x1306 * x1086;
float* x1308 = x1150+x1307;
for(int x1310=0; x1310 < x1086; x1310++) {
int32_t x1311 = x1305;
float x1312 = x1308[x1310];
x1304[x1311] = x1312;
x1305 += 1;

}
int32_t x1317 = x1306 * x1168;
float* x1318 = x1277+x1317;
for(int x1320=0; x1320 < x1168; x1320++) {
int32_t x1321 = x1305;
float x1322 = x1318[x1320];
x1304[x1321] = x1322;
x1305 += 1;

}

}
if (x1330) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(64) x Const(256) x Sym(1084) x Sym(1084)|(2,2)");
}
float* x1343 = (float*)myMalloc(x1342 * sizeof(float));;
for(int x1345=0; x1345 < x1342; x1345++) {
x1343[x1345] = -3.4028235E38f;

}
int* x1349 = (int32_t*)myMalloc(x1340 * sizeof(int32_t));;
for(int x1350=0; x1350 < 64; x1350++) {
int32_t x1351 = x1350 * x1300;
float* x1352 = x1304+x1351;
int32_t x1353 = x1350 * x1339;
float* x1354 = x1343+x1353;
int* x1355 = x1349+x1353;
int32_t x1356 = 0;
int32_t x1357 = 0;
for(int x1359=0; x1359 < 256; x1359++) {
int32_t x1360 = x1356;
int32_t x1361 = x1360;
int32_t x1362 = x1357;
int32_t x1363 = x1362;
for(int x1365=0; x1365 < x1337; x1365++) {
int32_t x1366 = x1361;
int32_t x1367 = x1366;
int32_t x1368 = x1363;
int32_t x1369 = x1368;
for(int x1370=0; x1370 < x1337; x1370++) {
int32_t x1371 = x1369;
int32_t x1372 = x1371;
int32_t x1373 = x1372;
int32_t x1374 = x1373;
int32_t x1375 = x1374;
float x1376 = x1352[x1375];
int32_t x1377 = x1367;
float x1378 = x1354[x1377];
bool x1379 = x1376 > x1378;
if (x1379) {
float x1380 = x1352[x1375];
x1354[x1377] = x1380;
int32_t x1382 = x1375 + x1351;
x1355[x1377] = x1382;
} else {
}
x1374 += 1;
int32_t x1387 = x1374;
float x1388 = x1352[x1387];
float x1389 = x1354[x1377];
bool x1390 = x1388 > x1389;
if (x1390) {
float x1391 = x1352[x1387];
x1354[x1377] = x1391;
int32_t x1393 = x1387 + x1351;
x1355[x1377] = x1393;
} else {
}
x1374 += 1;
x1372 += x1084;
int32_t x1399 = x1372;
int32_t x1400 = x1399;
int32_t x1401 = x1400;
float x1402 = x1352[x1401];
float x1403 = x1354[x1377];
bool x1404 = x1402 > x1403;
if (x1404) {
float x1405 = x1352[x1401];
x1354[x1377] = x1405;
int32_t x1407 = x1401 + x1351;
x1355[x1377] = x1407;
} else {
}
x1400 += 1;
int32_t x1412 = x1400;
float x1413 = x1352[x1412];
float x1414 = x1354[x1377];
bool x1415 = x1413 > x1414;
if (x1415) {
float x1416 = x1352[x1412];
x1354[x1377] = x1416;
int32_t x1418 = x1412 + x1351;
x1355[x1377] = x1418;
} else {
}
x1400 += 1;
x1372 += x1084;
x1367 += 1;
x1369 += 2;

}
x1361 += x1337;
x1363 += x1429;

}
x1356 += x1338;
x1357 += x1085;

}

}
float* x1446 = (float*)myMalloc(x1445 * sizeof(float));;
int32_t x1447 = 0;
for(int x1448=0; x1448 < 64; x1448++) {
for(int x1449=0; x1449 < 32; x1449++) {
for(int x1451=0; x1451 < x1441; x1451++) {
int32_t x1452 = x1447;
float x1453 = x57[x1449];
x1446[x1452] = x1453;
x1447 += 1;

}

}

}
float* x1465 = (float*)myMalloc(x1464 * sizeof(float));;
for(int x1466=0; x1466 < 64; x1466++) {
int32_t x1467 = x1466 * x1339;
float* x1468 = x1343+x1467;
int32_t x1469 = x1466 * x1442;
float* x1470 = x1446+x1469;
int32_t x1471 = x1466 * x1462;
float* x1472 = x1465+x1471;
for(int x1473=0; x1473 < 256; x1473++) {
int32_t x1474 = x1473 / 1;
int32_t x1478 = x1474 * x1440;
int32_t x1479 = x1478 * x1440;
int32_t x1475 = x1473 % 1;
int32_t x1476 = x1475 / 1;
int32_t x1480 = x1476 * x1440;
int32_t x1481 = x1480 * x1440;
int32_t x1482 = x1479 + x1481;
int32_t x1477 = x1475 % 1;
int32_t x1483 = x1477 * x1440;
int32_t x1484 = x1483 * x1440;
int32_t x1485 = x1482 + x1484;
float* x1486 = x1472+x1485;
int32_t x1487 = x1474 * x1337;
int32_t x1488 = x1487 * x1337;
float* x1489 = x1468+x1488;
for(int x1491=0; x1491 < x1440; x1491++) {
int32_t x1493 = x1491 * x1440;
float* x1494 = x1486+x1493;
int32_t x1492 = x1491 + x1476;
int32_t x1495 = x1492 * x1337;
int32_t x1496 = x1495 + x1477;
float* x1497 = x1489+x1496;
memcpy(x1494, x1497, 4 * x1440);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 32,x1441,256,1,x69,256,x1472,x1441,1,x1470,x1441);

}
float* x1506 = (float*)myMalloc(x1443 * sizeof(float));;
for(int x1508=0; x1508 < x1443; x1508++) {
float x1509 = x1446[x1508];
bool x1510 = x1509 < 0.0f;
if (x1510) {
x1506[x1508] = 0.0f;
} else {
float x1513 = x1446[x1508];
x1506[x1508] = x1513;
}

}
float* x1526 = (float*)myMalloc(x1525 * sizeof(float));;
int32_t x1527 = 0;
for(int x1528=0; x1528 < 64; x1528++) {
for(int x1529=0; x1529 < 128; x1529++) {
for(int x1531=0; x1531 < x1521; x1531++) {
int32_t x1532 = x1527;
float x1533 = x63[x1529];
x1526[x1532] = x1533;
x1527 += 1;

}

}

}
float* x1545 = (float*)myMalloc(x1544 * sizeof(float));;
for(int x1546=0; x1546 < 64; x1546++) {
int32_t x1547 = x1546 * x1442;
float* x1548 = x1506+x1547;
int32_t x1549 = x1546 * x1522;
float* x1550 = x1526+x1549;
int32_t x1551 = x1546 * x1542;
float* x1552 = x1545+x1551;
for(int x1553=0; x1553 < 32; x1553++) {
int32_t x1554 = x1553 / 1;
int32_t x1558 = x1554 * x1520;
int32_t x1559 = x1558 * x1520;
int32_t x1555 = x1553 % 1;
int32_t x1556 = x1555 / 1;
int32_t x1560 = x1556 * x1520;
int32_t x1561 = x1560 * x1520;
int32_t x1562 = x1559 + x1561;
int32_t x1557 = x1555 % 1;
int32_t x1563 = x1557 * x1520;
int32_t x1564 = x1563 * x1520;
int32_t x1565 = x1562 + x1564;
float* x1566 = x1552+x1565;
int32_t x1567 = x1554 * x1440;
int32_t x1568 = x1567 * x1440;
float* x1569 = x1548+x1568;
for(int x1571=0; x1571 < x1520; x1571++) {
int32_t x1573 = x1571 * x1520;
float* x1574 = x1566+x1573;
int32_t x1572 = x1571 + x1556;
int32_t x1575 = x1572 * x1440;
int32_t x1576 = x1575 + x1557;
float* x1577 = x1569+x1576;
memcpy(x1574, x1577, 4 * x1520);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1521,32,1,x49,32,x1552,x1521,1,x1550,x1521);

}
float* x1586 = (float*)myMalloc(x1523 * sizeof(float));;
for(int x1588=0; x1588 < x1523; x1588++) {
float x1589 = x1526[x1588];
bool x1590 = x1589 < 0.0f;
if (x1590) {
x1586[x1588] = 0.0f;
} else {
float x1593 = x1526[x1588];
x1586[x1588] = x1593;
}

}
float* x1608 = (float*)myMalloc(x1607 * sizeof(float));;
int32_t x1609 = 0;
for(int x1610=0; x1610 < 64; x1610++) {
for(int x1611=0; x1611 < 128; x1611++) {
for(int x1613=0; x1613 < x1603; x1613++) {
int32_t x1614 = x1609;
float x1615 = x58[x1611];
x1608[x1614] = x1615;
x1609 += 1;

}

}

}
float* x1627 = (float*)myMalloc(x1626 * sizeof(float));;
for(int x1628=0; x1628 < 64; x1628++) {
int32_t x1629 = x1628 * x1442;
float* x1630 = x1506+x1629;
int32_t x1631 = x1628 * x1604;
float* x1632 = x1608+x1631;
int32_t x1633 = x1628 * x1624;
float* x1634 = x1627+x1633;
for(int x1635=0; x1635 < 288; x1635++) {
int32_t x1636 = x1635 / 9;
int32_t x1640 = x1636 * 3;
int32_t x1641 = x1640 * 3;
int32_t x1642 = x1641 * x1602;
int32_t x1643 = x1642 * x1602;
int32_t x1637 = x1635 % 9;
int32_t x1638 = x1637 / 3;
int32_t x1644 = x1638 * 3;
int32_t x1645 = x1644 * x1602;
int32_t x1646 = x1645 * x1602;
int32_t x1647 = x1643 + x1646;
int32_t x1639 = x1637 % 3;
int32_t x1648 = x1639 * x1602;
int32_t x1649 = x1648 * x1602;
int32_t x1650 = x1647 + x1649;
float* x1651 = x1634+x1650;
int32_t x1652 = x1636 * x1440;
int32_t x1653 = x1652 * x1440;
float* x1654 = x1630+x1653;
int32_t x1667 = 1 - x1639;
bool x1668 = x1667 > 0;
int32_t x1669;
if (x1668) {
x1669 = x1667;
} else {
x1669 = 0;
}
int32_t x1670 = 3 - x1639;
int32_t x1671 = x1670 - 1;
int32_t x1672 = 1 - x1671;
bool x1673 = x1672 > 0;
int32_t x1674;
if (x1673) {
x1674 = x1672;
} else {
x1674 = 0;
}
int32_t x1675 = x1602 - x1674;
int32_t x1676 = x1675 - x1669;
bool x1677 = x1676 <= 0;
bool x1681 = x1669 > 0;
int32_t x1666 = -1 + x1639;
bool x1694 = x1674 > 0;
for(int x1656=0; x1656 < x1602; x1656++) {
int32_t x1657 = x1656 - 1;
int32_t x1658 = x1657 + x1638;
bool x1659 = x1658 < 0;
bool x1660 = x1658 >= x1440;
bool x1661 = x1659 || x1660;
if (x1661) {
int32_t x1662 = x1656 * x1602;
float* x1663 = x1651+x1662;
memset(x1663, 0, 4 * x1602);;
} else {
if (x1677) {
int32_t x1662 = x1656 * x1602;
float* x1678 = x1651+x1662;
memset(x1678, 0, 4 * x1602);;
} else {
int32_t x1662 = x1656 * x1602;
if (x1681) {
float* x1682 = x1651+x1662;
memset(x1682, 0, 4 * x1669);;
} else {
}
// may have segfault here
int32_t x1687 = x1662 + x1669;
float* x1688 = x1651+x1687;
int32_t x1689 = x1658 * x1440;
int32_t x1690 = x1689 + x1666;
int32_t x1691 = x1690 + x1669;
float* x1692 = x1654+x1691;
memcpy(x1688, x1692, 4 * x1676);;
if (x1694) {
int32_t x1695 = x1662 + x1602;
int32_t x1696 = x1695 - x1674;
float* x1697 = x1651+x1696;
memset(x1697, 0, 4 * x1674);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1603,288,1,x78,288,x1634,x1603,1,x1632,x1603);

}
float* x1712 = (float*)myMalloc(x1605 * sizeof(float));;
for(int x1714=0; x1714 < x1605; x1714++) {
float x1715 = x1608[x1714];
bool x1716 = x1715 < 0.0f;
if (x1716) {
x1712[x1714] = 0.0f;
} else {
float x1719 = x1608[x1714];
x1712[x1714] = x1719;
}

}
if (x1728) {
} else {
printf("all dimensions except the concatenation dimension should be the same\n");
assert(false && "");
}
// back prop for concat
float* x1739 = (float*)myMalloc(x1738 * sizeof(float));;
int32_t x1740 = 0;
for(int x1741=0; x1741 < 64; x1741++) {
int32_t x1742 = x1741 * x1522;
float* x1743 = x1586+x1742;
for(int x1745=0; x1745 < x1522; x1745++) {
int32_t x1746 = x1740;
float x1747 = x1743[x1745];
x1739[x1746] = x1747;
x1740 += 1;

}
int32_t x1752 = x1741 * x1604;
float* x1753 = x1712+x1752;
for(int x1755=0; x1755 < x1604; x1755++) {
int32_t x1756 = x1740;
float x1757 = x1753[x1755];
x1739[x1756] = x1757;
x1740 += 1;

}

}
float* x1771 = (float*)myMalloc(x1770 * sizeof(float));;
int32_t x1772 = 0;
for(int x1773=0; x1773 < 64; x1773++) {
for(int x1775=0; x1775 < 48; x1775++) {
for(int x1777=0; x1777 < x1766; x1777++) {
int32_t x1778 = x1772;
float x1779 = x94[x1775];
x1771[x1778] = x1779;
x1772 += 1;

}

}

}
float* x1791 = (float*)myMalloc(x1790 * sizeof(float));;
for(int x1792=0; x1792 < 64; x1792++) {
int32_t x1793 = x1792 * x1735;
float* x1794 = x1739+x1793;
int32_t x1795 = x1792 * x1767;
float* x1796 = x1771+x1795;
int32_t x1797 = x1792 * x1788;
float* x1798 = x1791+x1797;
for(int x1799=0; x1799 < 256; x1799++) {
int32_t x1800 = x1799 / 1;
int32_t x1804 = x1800 * x1765;
int32_t x1805 = x1804 * x1765;
int32_t x1801 = x1799 % 1;
int32_t x1802 = x1801 / 1;
int32_t x1806 = x1802 * x1765;
int32_t x1807 = x1806 * x1765;
int32_t x1808 = x1805 + x1807;
int32_t x1803 = x1801 % 1;
int32_t x1809 = x1803 * x1765;
int32_t x1810 = x1809 * x1765;
int32_t x1811 = x1808 + x1810;
float* x1812 = x1798+x1811;
int32_t x1813 = x1800 * x1520;
int32_t x1814 = x1813 * x1520;
float* x1815 = x1794+x1814;
for(int x1817=0; x1817 < x1765; x1817++) {
int32_t x1819 = x1817 * x1765;
float* x1820 = x1812+x1819;
int32_t x1818 = x1817 + x1802;
int32_t x1821 = x1818 * x1520;
int32_t x1822 = x1821 + x1803;
float* x1823 = x1815+x1822;
memcpy(x1820, x1823, 4 * x1765);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 48,x1766,256,1,x84,256,x1798,x1766,1,x1796,x1766);

}
float* x1832 = (float*)myMalloc(x1768 * sizeof(float));;
for(int x1834=0; x1834 < x1768; x1834++) {
float x1835 = x1771[x1834];
bool x1836 = x1835 < 0.0f;
if (x1836) {
x1832[x1834] = 0.0f;
} else {
float x1839 = x1771[x1834];
x1832[x1834] = x1839;
}

}
float* x1852 = (float*)myMalloc(x1851 * sizeof(float));;
int32_t x1853 = 0;
for(int x1854=0; x1854 < 64; x1854++) {
for(int x1856=0; x1856 < 192; x1856++) {
for(int x1858=0; x1858 < x1847; x1858++) {
int32_t x1859 = x1853;
float x1860 = x88[x1856];
x1852[x1859] = x1860;
x1853 += 1;

}

}

}
float* x1872 = (float*)myMalloc(x1871 * sizeof(float));;
for(int x1873=0; x1873 < 64; x1873++) {
int32_t x1874 = x1873 * x1767;
float* x1875 = x1832+x1874;
int32_t x1876 = x1873 * x1848;
float* x1877 = x1852+x1876;
int32_t x1878 = x1873 * x1869;
float* x1879 = x1872+x1878;
for(int x1880=0; x1880 < 48; x1880++) {
int32_t x1881 = x1880 / 1;
int32_t x1885 = x1881 * x1846;
int32_t x1886 = x1885 * x1846;
int32_t x1882 = x1880 % 1;
int32_t x1883 = x1882 / 1;
int32_t x1887 = x1883 * x1846;
int32_t x1888 = x1887 * x1846;
int32_t x1889 = x1886 + x1888;
int32_t x1884 = x1882 % 1;
int32_t x1890 = x1884 * x1846;
int32_t x1891 = x1890 * x1846;
int32_t x1892 = x1889 + x1891;
float* x1893 = x1879+x1892;
int32_t x1894 = x1881 * x1765;
int32_t x1895 = x1894 * x1765;
float* x1896 = x1875+x1895;
for(int x1898=0; x1898 < x1846; x1898++) {
int32_t x1900 = x1898 * x1846;
float* x1901 = x1893+x1900;
int32_t x1899 = x1898 + x1883;
int32_t x1902 = x1899 * x1765;
int32_t x1903 = x1902 + x1884;
float* x1904 = x1896+x1903;
memcpy(x1901, x1904, 4 * x1846);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 192,x1847,48,1,x90,48,x1879,x1847,1,x1877,x1847);

}
float* x1913 = (float*)myMalloc(x1849 * sizeof(float));;
for(int x1915=0; x1915 < x1849; x1915++) {
float x1916 = x1852[x1915];
bool x1917 = x1916 < 0.0f;
if (x1917) {
x1913[x1915] = 0.0f;
} else {
float x1920 = x1852[x1915];
x1913[x1915] = x1920;
}

}
float* x1935 = (float*)myMalloc(x1934 * sizeof(float));;
int32_t x1936 = 0;
for(int x1937=0; x1937 < 64; x1937++) {
for(int x1938=0; x1938 < 192; x1938++) {
for(int x1940=0; x1940 < x1930; x1940++) {
int32_t x1941 = x1936;
float x1942 = x71[x1938];
x1935[x1941] = x1942;
x1936 += 1;

}

}

}
float* x1954 = (float*)myMalloc(x1953 * sizeof(float));;
for(int x1955=0; x1955 < 64; x1955++) {
int32_t x1956 = x1955 * x1767;
float* x1957 = x1832+x1956;
int32_t x1958 = x1955 * x1931;
float* x1959 = x1935+x1958;
int32_t x1960 = x1955 * x1951;
float* x1961 = x1954+x1960;
for(int x1963=0; x1963 < 432; x1963++) {
int32_t x1964 = x1963 / 9;
int32_t x1968 = x1964 * 3;
int32_t x1969 = x1968 * 3;
int32_t x1970 = x1969 * x1929;
int32_t x1971 = x1970 * x1929;
int32_t x1965 = x1963 % 9;
int32_t x1966 = x1965 / 3;
int32_t x1972 = x1966 * 3;
int32_t x1973 = x1972 * x1929;
int32_t x1974 = x1973 * x1929;
int32_t x1975 = x1971 + x1974;
int32_t x1967 = x1965 % 3;
int32_t x1976 = x1967 * x1929;
int32_t x1977 = x1976 * x1929;
int32_t x1978 = x1975 + x1977;
float* x1979 = x1961+x1978;
int32_t x1980 = x1964 * x1765;
int32_t x1981 = x1980 * x1765;
float* x1982 = x1957+x1981;
int32_t x1995 = 1 - x1967;
bool x1996 = x1995 > 0;
int32_t x1997;
if (x1996) {
x1997 = x1995;
} else {
x1997 = 0;
}
int32_t x1998 = 3 - x1967;
int32_t x1999 = x1998 - 1;
int32_t x2000 = 1 - x1999;
bool x2001 = x2000 > 0;
int32_t x2002;
if (x2001) {
x2002 = x2000;
} else {
x2002 = 0;
}
int32_t x2003 = x1929 - x2002;
int32_t x2004 = x2003 - x1997;
bool x2005 = x2004 <= 0;
bool x2009 = x1997 > 0;
int32_t x1994 = -1 + x1967;
bool x2022 = x2002 > 0;
for(int x1984=0; x1984 < x1929; x1984++) {
int32_t x1985 = x1984 - 1;
int32_t x1986 = x1985 + x1966;
bool x1987 = x1986 < 0;
bool x1988 = x1986 >= x1765;
bool x1989 = x1987 || x1988;
if (x1989) {
int32_t x1990 = x1984 * x1929;
float* x1991 = x1979+x1990;
memset(x1991, 0, 4 * x1929);;
} else {
if (x2005) {
int32_t x1990 = x1984 * x1929;
float* x2006 = x1979+x1990;
memset(x2006, 0, 4 * x1929);;
} else {
int32_t x1990 = x1984 * x1929;
if (x2009) {
float* x2010 = x1979+x1990;
memset(x2010, 0, 4 * x1997);;
} else {
}
// may have segfault here
int32_t x2015 = x1990 + x1997;
float* x2016 = x1979+x2015;
int32_t x2017 = x1986 * x1765;
int32_t x2018 = x2017 + x1994;
int32_t x2019 = x2018 + x1997;
float* x2020 = x1982+x2019;
memcpy(x2016, x2020, 4 * x2004);;
if (x2022) {
int32_t x2023 = x1990 + x1929;
int32_t x2024 = x2023 - x2002;
float* x2025 = x1979+x2024;
memset(x2025, 0, 4 * x2002);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 192,x1930,432,1,x81,432,x1961,x1930,1,x1959,x1930);

}
float* x2040 = (float*)myMalloc(x1932 * sizeof(float));;
for(int x2042=0; x2042 < x1932; x2042++) {
float x2043 = x1935[x2042];
bool x2044 = x2043 < 0.0f;
if (x2044) {
x2040[x2042] = 0.0f;
} else {
float x2047 = x1935[x2042];
x2040[x2042] = x2047;
}

}
if (x2056) {
} else {
printf("all dimensions except the concatenation dimension should be the same\n");
assert(false && "");
}
// back prop for concat
float* x2067 = (float*)myMalloc(x2066 * sizeof(float));;
int32_t x2068 = 0;
for(int x2069=0; x2069 < 64; x2069++) {
int32_t x2070 = x2069 * x1848;
float* x2071 = x1913+x2070;
for(int x2073=0; x2073 < x1848; x2073++) {
int32_t x2074 = x2068;
float x2075 = x2071[x2073];
x2067[x2074] = x2075;
x2068 += 1;

}
int32_t x2080 = x2069 * x1931;
float* x2081 = x2040+x2080;
for(int x2083=0; x2083 < x1931; x2083++) {
int32_t x2084 = x2068;
float x2085 = x2081[x2083];
x2067[x2084] = x2085;
x2068 += 1;

}

}
float* x2099 = (float*)myMalloc(x2098 * sizeof(float));;
int32_t x2100 = 0;
for(int x2101=0; x2101 < 64; x2101++) {
for(int x2102=0; x2102 < 48; x2102++) {
for(int x2104=0; x2104 < x2094; x2104++) {
int32_t x2105 = x2100;
float x2106 = x44[x2102];
x2099[x2105] = x2106;
x2100 += 1;

}

}

}
float* x2118 = (float*)myMalloc(x2117 * sizeof(float));;
for(int x2119=0; x2119 < 64; x2119++) {
int32_t x2120 = x2119 * x2063;
float* x2121 = x2067+x2120;
int32_t x2122 = x2119 * x2095;
float* x2123 = x2099+x2122;
int32_t x2124 = x2119 * x2115;
float* x2125 = x2118+x2124;
for(int x2127=0; x2127 < 384; x2127++) {
int32_t x2128 = x2127 / 1;
int32_t x2132 = x2128 * x2093;
int32_t x2133 = x2132 * x2093;
int32_t x2129 = x2127 % 1;
int32_t x2130 = x2129 / 1;
int32_t x2134 = x2130 * x2093;
int32_t x2135 = x2134 * x2093;
int32_t x2136 = x2133 + x2135;
int32_t x2131 = x2129 % 1;
int32_t x2137 = x2131 * x2093;
int32_t x2138 = x2137 * x2093;
int32_t x2139 = x2136 + x2138;
float* x2140 = x2125+x2139;
int32_t x2141 = x2128 * x1846;
int32_t x2142 = x2141 * x1846;
float* x2143 = x2121+x2142;
for(int x2145=0; x2145 < x2093; x2145++) {
int32_t x2147 = x2145 * x2093;
float* x2148 = x2140+x2147;
int32_t x2146 = x2145 + x2130;
int32_t x2149 = x2146 * x1846;
int32_t x2150 = x2149 + x2131;
float* x2151 = x2143+x2150;
memcpy(x2148, x2151, 4 * x2093);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 48,x2094,384,1,x56,384,x2125,x2094,1,x2123,x2094);

}
float* x2160 = (float*)myMalloc(x2096 * sizeof(float));;
for(int x2162=0; x2162 < x2096; x2162++) {
float x2163 = x2099[x2162];
bool x2164 = x2163 < 0.0f;
if (x2164) {
x2160[x2162] = 0.0f;
} else {
float x2167 = x2099[x2162];
x2160[x2162] = x2167;
}

}
float* x2180 = (float*)myMalloc(x2179 * sizeof(float));;
int32_t x2181 = 0;
for(int x2182=0; x2182 < 64; x2182++) {
for(int x2183=0; x2183 < 192; x2183++) {
for(int x2185=0; x2185 < x2175; x2185++) {
int32_t x2186 = x2181;
float x2187 = x74[x2183];
x2180[x2186] = x2187;
x2181 += 1;

}

}

}
float* x2199 = (float*)myMalloc(x2198 * sizeof(float));;
for(int x2200=0; x2200 < 64; x2200++) {
int32_t x2201 = x2200 * x2095;
float* x2202 = x2160+x2201;
int32_t x2203 = x2200 * x2176;
float* x2204 = x2180+x2203;
int32_t x2205 = x2200 * x2196;
float* x2206 = x2199+x2205;
for(int x2207=0; x2207 < 48; x2207++) {
int32_t x2208 = x2207 / 1;
int32_t x2212 = x2208 * x2174;
int32_t x2213 = x2212 * x2174;
int32_t x2209 = x2207 % 1;
int32_t x2210 = x2209 / 1;
int32_t x2214 = x2210 * x2174;
int32_t x2215 = x2214 * x2174;
int32_t x2216 = x2213 + x2215;
int32_t x2211 = x2209 % 1;
int32_t x2217 = x2211 * x2174;
int32_t x2218 = x2217 * x2174;
int32_t x2219 = x2216 + x2218;
float* x2220 = x2206+x2219;
int32_t x2221 = x2208 * x2093;
int32_t x2222 = x2221 * x2093;
float* x2223 = x2202+x2222;
for(int x2225=0; x2225 < x2174; x2225++) {
int32_t x2227 = x2225 * x2174;
float* x2228 = x2220+x2227;
int32_t x2226 = x2225 + x2210;
int32_t x2229 = x2226 * x2093;
int32_t x2230 = x2229 + x2211;
float* x2231 = x2223+x2230;
memcpy(x2228, x2231, 4 * x2174);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 192,x2175,48,1,x64,48,x2206,x2175,1,x2204,x2175);

}
float* x2240 = (float*)myMalloc(x2177 * sizeof(float));;
for(int x2242=0; x2242 < x2177; x2242++) {
float x2243 = x2180[x2242];
bool x2244 = x2243 < 0.0f;
if (x2244) {
x2240[x2242] = 0.0f;
} else {
float x2247 = x2180[x2242];
x2240[x2242] = x2247;
}

}
float* x2262 = (float*)myMalloc(x2261 * sizeof(float));;
int32_t x2263 = 0;
for(int x2264=0; x2264 < 64; x2264++) {
for(int x2265=0; x2265 < 192; x2265++) {
for(int x2267=0; x2267 < x2257; x2267++) {
int32_t x2268 = x2263;
float x2269 = x86[x2265];
x2262[x2268] = x2269;
x2263 += 1;

}

}

}
float* x2281 = (float*)myMalloc(x2280 * sizeof(float));;
for(int x2282=0; x2282 < 64; x2282++) {
int32_t x2283 = x2282 * x2095;
float* x2284 = x2160+x2283;
int32_t x2285 = x2282 * x2258;
float* x2286 = x2262+x2285;
int32_t x2287 = x2282 * x2278;
float* x2288 = x2281+x2287;
for(int x2289=0; x2289 < 432; x2289++) {
int32_t x2290 = x2289 / 9;
int32_t x2294 = x2290 * 3;
int32_t x2295 = x2294 * 3;
int32_t x2296 = x2295 * x2256;
int32_t x2297 = x2296 * x2256;
int32_t x2291 = x2289 % 9;
int32_t x2292 = x2291 / 3;
int32_t x2298 = x2292 * 3;
int32_t x2299 = x2298 * x2256;
int32_t x2300 = x2299 * x2256;
int32_t x2301 = x2297 + x2300;
int32_t x2293 = x2291 % 3;
int32_t x2302 = x2293 * x2256;
int32_t x2303 = x2302 * x2256;
int32_t x2304 = x2301 + x2303;
float* x2305 = x2288+x2304;
int32_t x2306 = x2290 * x2093;
int32_t x2307 = x2306 * x2093;
float* x2308 = x2284+x2307;
int32_t x2321 = 1 - x2293;
bool x2322 = x2321 > 0;
int32_t x2323;
if (x2322) {
x2323 = x2321;
} else {
x2323 = 0;
}
int32_t x2324 = 3 - x2293;
int32_t x2325 = x2324 - 1;
int32_t x2326 = 1 - x2325;
bool x2327 = x2326 > 0;
int32_t x2328;
if (x2327) {
x2328 = x2326;
} else {
x2328 = 0;
}
int32_t x2329 = x2256 - x2328;
int32_t x2330 = x2329 - x2323;
bool x2331 = x2330 <= 0;
bool x2335 = x2323 > 0;
int32_t x2320 = -1 + x2293;
bool x2348 = x2328 > 0;
for(int x2310=0; x2310 < x2256; x2310++) {
int32_t x2311 = x2310 - 1;
int32_t x2312 = x2311 + x2292;
bool x2313 = x2312 < 0;
bool x2314 = x2312 >= x2093;
bool x2315 = x2313 || x2314;
if (x2315) {
int32_t x2316 = x2310 * x2256;
float* x2317 = x2305+x2316;
memset(x2317, 0, 4 * x2256);;
} else {
if (x2331) {
int32_t x2316 = x2310 * x2256;
float* x2332 = x2305+x2316;
memset(x2332, 0, 4 * x2256);;
} else {
int32_t x2316 = x2310 * x2256;
if (x2335) {
float* x2336 = x2305+x2316;
memset(x2336, 0, 4 * x2323);;
} else {
}
// may have segfault here
int32_t x2341 = x2316 + x2323;
float* x2342 = x2305+x2341;
int32_t x2343 = x2312 * x2093;
int32_t x2344 = x2343 + x2320;
int32_t x2345 = x2344 + x2323;
float* x2346 = x2308+x2345;
memcpy(x2342, x2346, 4 * x2330);;
if (x2348) {
int32_t x2349 = x2316 + x2256;
int32_t x2350 = x2349 - x2328;
float* x2351 = x2305+x2350;
memset(x2351, 0, 4 * x2328);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 192,x2257,432,1,x60,432,x2288,x2257,1,x2286,x2257);

}
float* x2366 = (float*)myMalloc(x2259 * sizeof(float));;
for(int x2368=0; x2368 < x2259; x2368++) {
float x2369 = x2262[x2368];
bool x2370 = x2369 < 0.0f;
if (x2370) {
x2366[x2368] = 0.0f;
} else {
float x2373 = x2262[x2368];
x2366[x2368] = x2373;
}

}
if (x2382) {
} else {
printf("all dimensions except the concatenation dimension should be the same\n");
assert(false && "");
}
// back prop for concat
float* x2393 = (float*)myMalloc(x2392 * sizeof(float));;
int32_t x2394 = 0;
for(int x2395=0; x2395 < 64; x2395++) {
int32_t x2396 = x2395 * x2176;
float* x2397 = x2240+x2396;
for(int x2399=0; x2399 < x2176; x2399++) {
int32_t x2400 = x2394;
float x2401 = x2397[x2399];
x2393[x2400] = x2401;
x2394 += 1;

}
int32_t x2406 = x2395 * x2258;
float* x2407 = x2366+x2406;
for(int x2409=0; x2409 < x2258; x2409++) {
int32_t x2410 = x2394;
float x2411 = x2407[x2409];
x2393[x2410] = x2411;
x2394 += 1;

}

}
float* x2425 = (float*)myMalloc(x2424 * sizeof(float));;
int32_t x2426 = 0;
for(int x2427=0; x2427 < 64; x2427++) {
for(int x2428=0; x2428 < 64; x2428++) {
for(int x2430=0; x2430 < x2420; x2430++) {
int32_t x2431 = x2426;
float x2432 = x51[x2428];
x2425[x2431] = x2432;
x2426 += 1;

}

}

}
float* x2444 = (float*)myMalloc(x2443 * sizeof(float));;
for(int x2445=0; x2445 < 64; x2445++) {
int32_t x2446 = x2445 * x2389;
float* x2447 = x2393+x2446;
int32_t x2448 = x2445 * x2421;
float* x2449 = x2425+x2448;
int32_t x2450 = x2445 * x2441;
float* x2451 = x2444+x2450;
for(int x2452=0; x2452 < 384; x2452++) {
int32_t x2453 = x2452 / 1;
int32_t x2457 = x2453 * x2419;
int32_t x2458 = x2457 * x2419;
int32_t x2454 = x2452 % 1;
int32_t x2455 = x2454 / 1;
int32_t x2459 = x2455 * x2419;
int32_t x2460 = x2459 * x2419;
int32_t x2461 = x2458 + x2460;
int32_t x2456 = x2454 % 1;
int32_t x2462 = x2456 * x2419;
int32_t x2463 = x2462 * x2419;
int32_t x2464 = x2461 + x2463;
float* x2465 = x2451+x2464;
int32_t x2466 = x2453 * x2174;
int32_t x2467 = x2466 * x2174;
float* x2468 = x2447+x2467;
for(int x2470=0; x2470 < x2419; x2470++) {
int32_t x2472 = x2470 * x2419;
float* x2473 = x2465+x2472;
int32_t x2471 = x2470 + x2455;
int32_t x2474 = x2471 * x2174;
int32_t x2475 = x2474 + x2456;
float* x2476 = x2468+x2475;
memcpy(x2473, x2476, 4 * x2419);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x2420,384,1,x76,384,x2451,x2420,1,x2449,x2420);

}
float* x2485 = (float*)myMalloc(x2422 * sizeof(float));;
for(int x2487=0; x2487 < x2422; x2487++) {
float x2488 = x2425[x2487];
bool x2489 = x2488 < 0.0f;
if (x2489) {
x2485[x2487] = 0.0f;
} else {
float x2492 = x2425[x2487];
x2485[x2487] = x2492;
}

}
float* x2505 = (float*)myMalloc(x2504 * sizeof(float));;
int32_t x2506 = 0;
for(int x2507=0; x2507 < 64; x2507++) {
for(int x2508=0; x2508 < 256; x2508++) {
for(int x2510=0; x2510 < x2500; x2510++) {
int32_t x2511 = x2506;
float x2512 = x82[x2508];
x2505[x2511] = x2512;
x2506 += 1;

}

}

}
float* x2524 = (float*)myMalloc(x2523 * sizeof(float));;
for(int x2525=0; x2525 < 64; x2525++) {
int32_t x2526 = x2525 * x2421;
float* x2527 = x2485+x2526;
int32_t x2528 = x2525 * x2501;
float* x2529 = x2505+x2528;
int32_t x2530 = x2525 * x2521;
float* x2531 = x2524+x2530;
for(int x2532=0; x2532 < 64; x2532++) {
int32_t x2533 = x2532 / 1;
int32_t x2537 = x2533 * x2499;
int32_t x2538 = x2537 * x2499;
int32_t x2534 = x2532 % 1;
int32_t x2535 = x2534 / 1;
int32_t x2539 = x2535 * x2499;
int32_t x2540 = x2539 * x2499;
int32_t x2541 = x2538 + x2540;
int32_t x2536 = x2534 % 1;
int32_t x2542 = x2536 * x2499;
int32_t x2543 = x2542 * x2499;
int32_t x2544 = x2541 + x2543;
float* x2545 = x2531+x2544;
int32_t x2546 = x2533 * x2419;
int32_t x2547 = x2546 * x2419;
float* x2548 = x2527+x2547;
for(int x2550=0; x2550 < x2499; x2550++) {
int32_t x2552 = x2550 * x2499;
float* x2553 = x2545+x2552;
int32_t x2551 = x2550 + x2535;
int32_t x2554 = x2551 * x2419;
int32_t x2555 = x2554 + x2536;
float* x2556 = x2548+x2555;
memcpy(x2553, x2556, 4 * x2499);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x2500,64,1,x91,64,x2531,x2500,1,x2529,x2500);

}
float* x2565 = (float*)myMalloc(x2502 * sizeof(float));;
for(int x2567=0; x2567 < x2502; x2567++) {
float x2568 = x2505[x2567];
bool x2569 = x2568 < 0.0f;
if (x2569) {
x2565[x2567] = 0.0f;
} else {
float x2572 = x2505[x2567];
x2565[x2567] = x2572;
}

}
float* x2587 = (float*)myMalloc(x2586 * sizeof(float));;
int32_t x2588 = 0;
for(int x2589=0; x2589 < 64; x2589++) {
for(int x2590=0; x2590 < 256; x2590++) {
for(int x2592=0; x2592 < x2582; x2592++) {
int32_t x2593 = x2588;
float x2594 = x55[x2590];
x2587[x2593] = x2594;
x2588 += 1;

}

}

}
float* x2606 = (float*)myMalloc(x2605 * sizeof(float));;
for(int x2607=0; x2607 < 64; x2607++) {
int32_t x2608 = x2607 * x2421;
float* x2609 = x2485+x2608;
int32_t x2610 = x2607 * x2583;
float* x2611 = x2587+x2610;
int32_t x2612 = x2607 * x2603;
float* x2613 = x2606+x2612;
for(int x2615=0; x2615 < 576; x2615++) {
int32_t x2616 = x2615 / 9;
int32_t x2620 = x2616 * 3;
int32_t x2621 = x2620 * 3;
int32_t x2622 = x2621 * x2581;
int32_t x2623 = x2622 * x2581;
int32_t x2617 = x2615 % 9;
int32_t x2618 = x2617 / 3;
int32_t x2624 = x2618 * 3;
int32_t x2625 = x2624 * x2581;
int32_t x2626 = x2625 * x2581;
int32_t x2627 = x2623 + x2626;
int32_t x2619 = x2617 % 3;
int32_t x2628 = x2619 * x2581;
int32_t x2629 = x2628 * x2581;
int32_t x2630 = x2627 + x2629;
float* x2631 = x2613+x2630;
int32_t x2632 = x2616 * x2419;
int32_t x2633 = x2632 * x2419;
float* x2634 = x2609+x2633;
int32_t x2647 = 1 - x2619;
bool x2648 = x2647 > 0;
int32_t x2649;
if (x2648) {
x2649 = x2647;
} else {
x2649 = 0;
}
int32_t x2650 = 3 - x2619;
int32_t x2651 = x2650 - 1;
int32_t x2652 = 1 - x2651;
bool x2653 = x2652 > 0;
int32_t x2654;
if (x2653) {
x2654 = x2652;
} else {
x2654 = 0;
}
int32_t x2655 = x2581 - x2654;
int32_t x2656 = x2655 - x2649;
bool x2657 = x2656 <= 0;
bool x2661 = x2649 > 0;
int32_t x2646 = -1 + x2619;
bool x2674 = x2654 > 0;
for(int x2636=0; x2636 < x2581; x2636++) {
int32_t x2637 = x2636 - 1;
int32_t x2638 = x2637 + x2618;
bool x2639 = x2638 < 0;
bool x2640 = x2638 >= x2419;
bool x2641 = x2639 || x2640;
if (x2641) {
int32_t x2642 = x2636 * x2581;
float* x2643 = x2631+x2642;
memset(x2643, 0, 4 * x2581);;
} else {
if (x2657) {
int32_t x2642 = x2636 * x2581;
float* x2658 = x2631+x2642;
memset(x2658, 0, 4 * x2581);;
} else {
int32_t x2642 = x2636 * x2581;
if (x2661) {
float* x2662 = x2631+x2642;
memset(x2662, 0, 4 * x2649);;
} else {
}
// may have segfault here
int32_t x2667 = x2642 + x2649;
float* x2668 = x2631+x2667;
int32_t x2669 = x2638 * x2419;
int32_t x2670 = x2669 + x2646;
int32_t x2671 = x2670 + x2649;
float* x2672 = x2634+x2671;
memcpy(x2668, x2672, 4 * x2656);;
if (x2674) {
int32_t x2675 = x2642 + x2581;
int32_t x2676 = x2675 - x2654;
float* x2677 = x2631+x2676;
memset(x2677, 0, 4 * x2654);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x2582,576,1,x70,576,x2613,x2582,1,x2611,x2582);

}
float* x2692 = (float*)myMalloc(x2584 * sizeof(float));;
for(int x2694=0; x2694 < x2584; x2694++) {
float x2695 = x2587[x2694];
bool x2696 = x2695 < 0.0f;
if (x2696) {
x2692[x2694] = 0.0f;
} else {
float x2699 = x2587[x2694];
x2692[x2694] = x2699;
}

}
if (x2708) {
} else {
printf("all dimensions except the concatenation dimension should be the same\n");
assert(false && "");
}
// back prop for concat
float* x2719 = (float*)myMalloc(x2718 * sizeof(float));;
int32_t x2720 = 0;
for(int x2721=0; x2721 < 64; x2721++) {
int32_t x2722 = x2721 * x2501;
float* x2723 = x2565+x2722;
for(int x2725=0; x2725 < x2501; x2725++) {
int32_t x2726 = x2720;
float x2727 = x2723[x2725];
x2719[x2726] = x2727;
x2720 += 1;

}
int32_t x2732 = x2721 * x2583;
float* x2733 = x2692+x2732;
for(int x2735=0; x2735 < x2583; x2735++) {
int32_t x2736 = x2720;
float x2737 = x2733[x2735];
x2719[x2736] = x2737;
x2720 += 1;

}

}
if (x2745) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(64) x Const(512) x Sym(2499) x Sym(2499)|(2,2)");
}
float* x2758 = (float*)myMalloc(x2757 * sizeof(float));;
for(int x2760=0; x2760 < x2757; x2760++) {
x2758[x2760] = -3.4028235E38f;

}
int* x2764 = (int32_t*)myMalloc(x2755 * sizeof(int32_t));;
for(int x2765=0; x2765 < 64; x2765++) {
int32_t x2766 = x2765 * x2715;
float* x2767 = x2719+x2766;
int32_t x2768 = x2765 * x2754;
float* x2769 = x2758+x2768;
int* x2770 = x2764+x2768;
int32_t x2771 = 0;
int32_t x2772 = 0;
for(int x2774=0; x2774 < 512; x2774++) {
int32_t x2775 = x2771;
int32_t x2776 = x2775;
int32_t x2777 = x2772;
int32_t x2778 = x2777;
for(int x2780=0; x2780 < x2752; x2780++) {
int32_t x2781 = x2776;
int32_t x2782 = x2781;
int32_t x2783 = x2778;
int32_t x2784 = x2783;
for(int x2785=0; x2785 < x2752; x2785++) {
int32_t x2786 = x2784;
int32_t x2787 = x2786;
int32_t x2788 = x2787;
int32_t x2789 = x2788;
int32_t x2790 = x2789;
float x2791 = x2767[x2790];
int32_t x2792 = x2782;
float x2793 = x2769[x2792];
bool x2794 = x2791 > x2793;
if (x2794) {
float x2795 = x2767[x2790];
x2769[x2792] = x2795;
int32_t x2797 = x2790 + x2766;
x2770[x2792] = x2797;
} else {
}
x2789 += 1;
int32_t x2802 = x2789;
float x2803 = x2767[x2802];
float x2804 = x2769[x2792];
bool x2805 = x2803 > x2804;
if (x2805) {
float x2806 = x2767[x2802];
x2769[x2792] = x2806;
int32_t x2808 = x2802 + x2766;
x2770[x2792] = x2808;
} else {
}
x2789 += 1;
x2787 += x2499;
int32_t x2814 = x2787;
int32_t x2815 = x2814;
int32_t x2816 = x2815;
float x2817 = x2767[x2816];
float x2818 = x2769[x2792];
bool x2819 = x2817 > x2818;
if (x2819) {
float x2820 = x2767[x2816];
x2769[x2792] = x2820;
int32_t x2822 = x2816 + x2766;
x2770[x2792] = x2822;
} else {
}
x2815 += 1;
int32_t x2827 = x2815;
float x2828 = x2767[x2827];
float x2829 = x2769[x2792];
bool x2830 = x2828 > x2829;
if (x2830) {
float x2831 = x2767[x2827];
x2769[x2792] = x2831;
int32_t x2833 = x2827 + x2766;
x2770[x2792] = x2833;
} else {
}
x2815 += 1;
x2787 += x2499;
x2782 += 1;
x2784 += 2;

}
x2776 += x2752;
x2778 += x2844;

}
x2771 += x2753;
x2772 += x2500;

}

}
float* x2861 = (float*)myMalloc(x2860 * sizeof(float));;
int32_t x2862 = 0;
for(int x2863=0; x2863 < 64; x2863++) {
for(int x2864=0; x2864 < 64; x2864++) {
for(int x2866=0; x2866 < x2856; x2866++) {
int32_t x2867 = x2862;
float x2868 = x62[x2864];
x2861[x2867] = x2868;
x2862 += 1;

}

}

}
float* x2880 = (float*)myMalloc(x2879 * sizeof(float));;
for(int x2881=0; x2881 < 64; x2881++) {
int32_t x2882 = x2881 * x2754;
float* x2883 = x2758+x2882;
int32_t x2884 = x2881 * x2857;
float* x2885 = x2861+x2884;
int32_t x2886 = x2881 * x2877;
float* x2887 = x2880+x2886;
for(int x2888=0; x2888 < 512; x2888++) {
int32_t x2889 = x2888 / 1;
int32_t x2893 = x2889 * x2855;
int32_t x2894 = x2893 * x2855;
int32_t x2890 = x2888 % 1;
int32_t x2891 = x2890 / 1;
int32_t x2895 = x2891 * x2855;
int32_t x2896 = x2895 * x2855;
int32_t x2897 = x2894 + x2896;
int32_t x2892 = x2890 % 1;
int32_t x2898 = x2892 * x2855;
int32_t x2899 = x2898 * x2855;
int32_t x2900 = x2897 + x2899;
float* x2901 = x2887+x2900;
int32_t x2902 = x2889 * x2752;
int32_t x2903 = x2902 * x2752;
float* x2904 = x2883+x2903;
for(int x2906=0; x2906 < x2855; x2906++) {
int32_t x2908 = x2906 * x2855;
float* x2909 = x2901+x2908;
int32_t x2907 = x2906 + x2891;
int32_t x2910 = x2907 * x2752;
int32_t x2911 = x2910 + x2892;
float* x2912 = x2904+x2911;
memcpy(x2909, x2912, 4 * x2855);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x2856,512,1,x43,512,x2887,x2856,1,x2885,x2856);

}
float* x2921 = (float*)myMalloc(x2858 * sizeof(float));;
for(int x2923=0; x2923 < x2858; x2923++) {
float x2924 = x2861[x2923];
bool x2925 = x2924 < 0.0f;
if (x2925) {
x2921[x2923] = 0.0f;
} else {
float x2928 = x2861[x2923];
x2921[x2923] = x2928;
}

}
float* x2941 = (float*)myMalloc(x2940 * sizeof(float));;
int32_t x2942 = 0;
for(int x2943=0; x2943 < 64; x2943++) {
for(int x2944=0; x2944 < 256; x2944++) {
for(int x2946=0; x2946 < x2936; x2946++) {
int32_t x2947 = x2942;
float x2948 = x68[x2944];
x2941[x2947] = x2948;
x2942 += 1;

}

}

}
float* x2960 = (float*)myMalloc(x2959 * sizeof(float));;
for(int x2961=0; x2961 < 64; x2961++) {
int32_t x2962 = x2961 * x2857;
float* x2963 = x2921+x2962;
int32_t x2964 = x2961 * x2937;
float* x2965 = x2941+x2964;
int32_t x2966 = x2961 * x2957;
float* x2967 = x2960+x2966;
for(int x2968=0; x2968 < 64; x2968++) {
int32_t x2969 = x2968 / 1;
int32_t x2973 = x2969 * x2935;
int32_t x2974 = x2973 * x2935;
int32_t x2970 = x2968 % 1;
int32_t x2971 = x2970 / 1;
int32_t x2975 = x2971 * x2935;
int32_t x2976 = x2975 * x2935;
int32_t x2977 = x2974 + x2976;
int32_t x2972 = x2970 % 1;
int32_t x2978 = x2972 * x2935;
int32_t x2979 = x2978 * x2935;
int32_t x2980 = x2977 + x2979;
float* x2981 = x2967+x2980;
int32_t x2982 = x2969 * x2855;
int32_t x2983 = x2982 * x2855;
float* x2984 = x2963+x2983;
for(int x2986=0; x2986 < x2935; x2986++) {
int32_t x2988 = x2986 * x2935;
float* x2989 = x2981+x2988;
int32_t x2987 = x2986 + x2971;
int32_t x2990 = x2987 * x2855;
int32_t x2991 = x2990 + x2972;
float* x2992 = x2984+x2991;
memcpy(x2989, x2992, 4 * x2935);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x2936,64,1,x80,64,x2967,x2936,1,x2965,x2936);

}
float* x3001 = (float*)myMalloc(x2938 * sizeof(float));;
for(int x3003=0; x3003 < x2938; x3003++) {
float x3004 = x2941[x3003];
bool x3005 = x3004 < 0.0f;
if (x3005) {
x3001[x3003] = 0.0f;
} else {
float x3008 = x2941[x3003];
x3001[x3003] = x3008;
}

}
float* x3023 = (float*)myMalloc(x3022 * sizeof(float));;
int32_t x3024 = 0;
for(int x3025=0; x3025 < 64; x3025++) {
for(int x3026=0; x3026 < 256; x3026++) {
for(int x3028=0; x3028 < x3018; x3028++) {
int32_t x3029 = x3024;
float x3030 = x59[x3026];
x3023[x3029] = x3030;
x3024 += 1;

}

}

}
float* x3042 = (float*)myMalloc(x3041 * sizeof(float));;
for(int x3043=0; x3043 < 64; x3043++) {
int32_t x3044 = x3043 * x2857;
float* x3045 = x2921+x3044;
int32_t x3046 = x3043 * x3019;
float* x3047 = x3023+x3046;
int32_t x3048 = x3043 * x3039;
float* x3049 = x3042+x3048;
for(int x3050=0; x3050 < 576; x3050++) {
int32_t x3051 = x3050 / 9;
int32_t x3055 = x3051 * 3;
int32_t x3056 = x3055 * 3;
int32_t x3057 = x3056 * x3017;
int32_t x3058 = x3057 * x3017;
int32_t x3052 = x3050 % 9;
int32_t x3053 = x3052 / 3;
int32_t x3059 = x3053 * 3;
int32_t x3060 = x3059 * x3017;
int32_t x3061 = x3060 * x3017;
int32_t x3062 = x3058 + x3061;
int32_t x3054 = x3052 % 3;
int32_t x3063 = x3054 * x3017;
int32_t x3064 = x3063 * x3017;
int32_t x3065 = x3062 + x3064;
float* x3066 = x3049+x3065;
int32_t x3067 = x3051 * x2855;
int32_t x3068 = x3067 * x2855;
float* x3069 = x3045+x3068;
int32_t x3082 = 1 - x3054;
bool x3083 = x3082 > 0;
int32_t x3084;
if (x3083) {
x3084 = x3082;
} else {
x3084 = 0;
}
int32_t x3085 = 3 - x3054;
int32_t x3086 = x3085 - 1;
int32_t x3087 = 1 - x3086;
bool x3088 = x3087 > 0;
int32_t x3089;
if (x3088) {
x3089 = x3087;
} else {
x3089 = 0;
}
int32_t x3090 = x3017 - x3089;
int32_t x3091 = x3090 - x3084;
bool x3092 = x3091 <= 0;
bool x3096 = x3084 > 0;
int32_t x3081 = -1 + x3054;
bool x3109 = x3089 > 0;
for(int x3071=0; x3071 < x3017; x3071++) {
int32_t x3072 = x3071 - 1;
int32_t x3073 = x3072 + x3053;
bool x3074 = x3073 < 0;
bool x3075 = x3073 >= x2855;
bool x3076 = x3074 || x3075;
if (x3076) {
int32_t x3077 = x3071 * x3017;
float* x3078 = x3066+x3077;
memset(x3078, 0, 4 * x3017);;
} else {
if (x3092) {
int32_t x3077 = x3071 * x3017;
float* x3093 = x3066+x3077;
memset(x3093, 0, 4 * x3017);;
} else {
int32_t x3077 = x3071 * x3017;
if (x3096) {
float* x3097 = x3066+x3077;
memset(x3097, 0, 4 * x3084);;
} else {
}
// may have segfault here
int32_t x3102 = x3077 + x3084;
float* x3103 = x3066+x3102;
int32_t x3104 = x3073 * x2855;
int32_t x3105 = x3104 + x3081;
int32_t x3106 = x3105 + x3084;
float* x3107 = x3069+x3106;
memcpy(x3103, x3107, 4 * x3091);;
if (x3109) {
int32_t x3110 = x3077 + x3017;
int32_t x3111 = x3110 - x3089;
float* x3112 = x3066+x3111;
memset(x3112, 0, 4 * x3089);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x3018,576,1,x72,576,x3049,x3018,1,x3047,x3018);

}
float* x3127 = (float*)myMalloc(x3020 * sizeof(float));;
for(int x3129=0; x3129 < x3020; x3129++) {
float x3130 = x3023[x3129];
bool x3131 = x3130 < 0.0f;
if (x3131) {
x3127[x3129] = 0.0f;
} else {
float x3134 = x3023[x3129];
x3127[x3129] = x3134;
}

}
if (x3143) {
} else {
printf("all dimensions except the concatenation dimension should be the same\n");
assert(false && "");
}
// back prop for concat
float* x3154 = (float*)myMalloc(x3153 * sizeof(float));;
int32_t x3155 = 0;
for(int x3156=0; x3156 < 64; x3156++) {
int32_t x3157 = x3156 * x2937;
float* x3158 = x3001+x3157;
for(int x3160=0; x3160 < x2937; x3160++) {
int32_t x3161 = x3155;
float x3162 = x3158[x3160];
x3154[x3161] = x3162;
x3155 += 1;

}
int32_t x3167 = x3156 * x3019;
float* x3168 = x3127+x3167;
for(int x3170=0; x3170 < x3019; x3170++) {
int32_t x3171 = x3155;
float x3172 = x3168[x3170];
x3154[x3171] = x3172;
x3155 += 1;

}

}
float* x3187 = (float*)myMalloc(x3186 * sizeof(float));;
int32_t x3188 = 0;
for(int x3189=0; x3189 < 64; x3189++) {
for(int x3191=0; x3191 < 10; x3191++) {
for(int x3193=0; x3193 < x3182; x3193++) {
int32_t x3194 = x3188;
float x3195 = x93[x3191];
x3187[x3194] = x3195;
x3188 += 1;

}

}

}
float* x3207 = (float*)myMalloc(x3206 * sizeof(float));;
for(int x3208=0; x3208 < 64; x3208++) {
int32_t x3209 = x3208 * x3150;
float* x3210 = x3154+x3209;
int32_t x3211 = x3208 * x3183;
float* x3212 = x3187+x3211;
int32_t x3213 = x3208 * x3204;
float* x3214 = x3207+x3213;
for(int x3216=0; x3216 < 8192; x3216++) {
int32_t x3217 = x3216 / 16;
int32_t x3221 = x3217 * 4;
int32_t x3222 = x3221 * 4;
int32_t x3223 = x3222 * x3181;
int32_t x3224 = x3223 * x3181;
int32_t x3218 = x3216 % 16;
int32_t x3219 = x3218 / 4;
int32_t x3225 = x3219 * 4;
int32_t x3226 = x3225 * x3181;
int32_t x3227 = x3226 * x3181;
int32_t x3228 = x3224 + x3227;
int32_t x3220 = x3218 % 4;
int32_t x3229 = x3220 * x3181;
int32_t x3230 = x3229 * x3181;
int32_t x3231 = x3228 + x3230;
float* x3232 = x3214+x3231;
int32_t x3233 = x3217 * x2935;
int32_t x3234 = x3233 * x2935;
float* x3235 = x3210+x3234;
for(int x3237=0; x3237 < x3181; x3237++) {
int32_t x3239 = x3237 * x3181;
float* x3240 = x3232+x3239;
int32_t x3238 = x3237 + x3219;
int32_t x3241 = x3238 * x2935;
int32_t x3242 = x3241 + x3220;
float* x3243 = x3235+x3242;
memcpy(x3240, x3243, 4 * x3181);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 10,x3182,8192,1,x46,8192,x3214,x3182,1,x3212,x3182);

}
int32_t x3252 = 0;
int32_t x3253 = 1;
x3253 *= 64;
x3253 *= 10;
int32_t x3256 = x3252;
bool x3257 = x3256 >= 2;
if (x3257) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3263 = x3256 == 0;
if (x3263) {
int32_t x3264 = x3253;
bool x3265 = x3264 == x3184;
if (x3265) {
} else {
assert(false && "must same size!!");
}
} else {
}
int64_t x3272 = (long)mallocAddr;
int64_t x3273 = x3272 - x95;
memset((void*)x95, 0, x3273);
mallocAddr = (void*)x95;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x3280 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x3281 = x3280 / 1000LL;
int64_t x3283 = x3280 / x3282;
printf("Inferencing completed in %ldms (%ld us/images)\n",x3281,x3283);

}
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

