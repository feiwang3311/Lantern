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
int32_t x273 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
int64_t x274 = fsize(x273);
int64_t x276 = x274 / 3073LL;
int32_t x277 = (int32_t)x276;
int32_t x278 = x277 * 3072;
float* x279 = (float*)myMalloc(x278 * sizeof(float));;
int* x280 = (int32_t*)myMalloc(x277 * sizeof(int32_t));;
char* x275 = (char*)mmap(0, x274, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x273, 0);
for(int x282=0; x282 < x277; x282++) {
int32_t x283 = x282 * 3073;
char x284 = x275[x283];
int32_t x285 = (int32_t)(unsigned char)x284;
x280[x282] = x285;
int32_t x291 = x283 + 1;
int32_t x289 = x282 * 3072;
for(int x288=0; x288 < 3072; x288++) {
int32_t x292 = x291 + x288;
char x293 = x275[x292];
int32_t x290 = x289 + x288;
float x294 = (float)(unsigned char)x293;
float x295 = x294 / 255.0f;
x279[x290] = x295;

}

}
int32_t x301 = x277 / 64;
int32_t x332 = 31 / 1;
int32_t x333 = x332 + 1;
int32_t x337 = 4096 * x333;
int32_t x338 = x337 * x333;
int32_t x334 = x333 * x333;
int32_t x342 = 1728 * x334;
int32_t x335 = 64 * x334;
int32_t x340 = 27 * x334;
int32_t x3 = open("/home/fei/bitbucket/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
int64_t x4 = fsize(x3);
float* x5 = (float*)mmap(0, x4, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x3, 0);
float* x152 = x5+0;
bool x459 = x333 == 1;
bool x460 = x459 || true;
bool x461 = x460 || x459;
bool x454 = true || false;
bool x471 = x333 <= 1;
int32_t x472;
if (x471) {
x472 = 1;
} else {
x472 = x333;
}
int32_t x478 = x472 * x472;
int32_t x482;
if (x459) {
x482 = 0;
} else {
x482 = x333;
}
int32_t x483;
if (x459) {
x483 = 0;
} else {
x483 = 1;
}
float* x40 = x5+1856;
float* x110 = x5+1920;
bool x562 = x472 == 1;
bool x563 = x562 || true;
bool x564 = x563 || x562;
bool x574 = x472 <= 1;
int32_t x575;
if (x574) {
x575 = 1;
} else {
x575 = x472;
}
int32_t x581 = x575 * x575;
int32_t x586;
if (x562) {
x586 = 0;
} else {
x586 = x472;
}
int32_t x587;
if (x562) {
x587 = 0;
} else {
x587 = 1;
}
bool x650 = x575 == 1;
bool x651 = x650 || true;
bool x652 = x651 || x650;
bool x662 = x575 <= 1;
int32_t x663;
if (x662) {
x663 = 1;
} else {
x663 = x575;
}
int32_t x669 = x663 * x663;
int32_t x674;
if (x650) {
x674 = 0;
} else {
x674 = x575;
}
int32_t x675;
if (x650) {
x675 = 0;
} else {
x675 = 1;
}
float* x206 = x5+1728;
bool x738 = x663 == 1;
bool x739 = x738 || true;
bool x740 = x739 || x738;
bool x750 = x663 <= 1;
int32_t x751;
if (x750) {
x751 = 1;
} else {
x751 = x663;
}
int32_t x757 = x751 * x751;
int32_t x762;
if (x738) {
x762 = 0;
} else {
x762 = x663;
}
int32_t x763;
if (x738) {
x763 = 0;
} else {
x763 = 1;
}
float* x251 = x5+1792;
bool x810 = x751 >= 2;
bool x811;
if (x810) {
x811 = x810;
} else {
x811 = false;
}
int32_t x816 = x751 - 2;
int32_t x817 = x816 / 2;
int32_t x818 = x817 + 1;
int32_t x819 = x818 * x818;
int32_t x910 = 2 * x751;
int32_t x920 = x817 / 1;
int32_t x921 = x920 + 1;
int32_t x925 = 4096 * x921;
int32_t x926 = x925 * x921;
int32_t x922 = x921 * x921;
int32_t x923 = 64 * x922;
float* x233 = x5+1984;
bool x999 = x921 == 1;
bool x1000 = x999 || true;
bool x1001 = x1000 || x999;
bool x1011 = x921 <= 1;
int32_t x1012;
if (x1011) {
x1012 = 1;
} else {
x1012 = x921;
}
int32_t x1018 = x1012 * x1012;
int32_t x1022;
if (x999) {
x1022 = 0;
} else {
x1022 = x921;
}
int32_t x1023;
if (x999) {
x1023 = 0;
} else {
x1023 = 1;
}
float* x114 = x5+6208;
float* x51 = x5+6272;
bool x1102 = x1012 == 1;
bool x1103 = x1102 || true;
bool x1104 = x1103 || x1102;
bool x1114 = x1012 <= 1;
int32_t x1115;
if (x1114) {
x1115 = 1;
} else {
x1115 = x1012;
}
int32_t x1121 = x1115 * x1115;
int32_t x1126;
if (x1102) {
x1126 = 0;
} else {
x1126 = x1012;
}
int32_t x1127;
if (x1102) {
x1127 = 0;
} else {
x1127 = 1;
}
bool x1190 = x1115 == 1;
bool x1191 = x1190 || true;
bool x1192 = x1191 || x1190;
bool x1202 = x1115 <= 1;
int32_t x1203;
if (x1202) {
x1203 = 1;
} else {
x1203 = x1115;
}
int32_t x1209 = x1203 * x1203;
int32_t x1214;
if (x1190) {
x1214 = 0;
} else {
x1214 = x1115;
}
int32_t x1215;
if (x1190) {
x1215 = 0;
} else {
x1215 = 1;
}
float* x26 = x5+6080;
bool x1278 = x1203 == 1;
bool x1279 = x1278 || true;
bool x1280 = x1279 || x1278;
bool x1290 = x1203 <= 1;
int32_t x1291;
if (x1290) {
x1291 = 1;
} else {
x1291 = x1203;
}
int32_t x1297 = x1291 * x1291;
int32_t x1302;
if (x1278) {
x1302 = 0;
} else {
x1302 = x1203;
}
int32_t x1303;
if (x1278) {
x1303 = 0;
} else {
x1303 = 1;
}
float* x53 = x5+6144;
int32_t x1350 = x1291 + 2;
int32_t x1351 = x1350 - 3;
int32_t x1352 = x1351 / 1;
int32_t x1353 = x1352 + 1;
int32_t x1357 = 4096 * x1353;
int32_t x1358 = x1357 * x1353;
int32_t x1354 = x1353 * x1353;
int32_t x1355 = 64 * x1354;
float* x90 = x5+6336;
bool x1480 = x1353 == 1;
bool x1481 = x1480 || true;
bool x1482 = x1481 || x1480;
bool x1492 = x1353 <= 1;
int32_t x1493;
if (x1492) {
x1493 = 1;
} else {
x1493 = x1353;
}
int32_t x1499 = x1493 * x1493;
int32_t x1503;
if (x1480) {
x1503 = 0;
} else {
x1503 = x1353;
}
int32_t x1504;
if (x1480) {
x1504 = 0;
} else {
x1504 = 1;
}
float* x105 = x5+43328;
float* x158 = x5+43392;
bool x1583 = x1493 == 1;
bool x1584 = x1583 || true;
bool x1585 = x1584 || x1583;
bool x1595 = x1493 <= 1;
int32_t x1596;
if (x1595) {
x1596 = 1;
} else {
x1596 = x1493;
}
int32_t x1602 = x1596 * x1596;
int32_t x1607;
if (x1583) {
x1607 = 0;
} else {
x1607 = x1493;
}
int32_t x1608;
if (x1583) {
x1608 = 0;
} else {
x1608 = 1;
}
bool x1671 = x1596 == 1;
bool x1672 = x1671 || true;
bool x1673 = x1672 || x1671;
bool x1683 = x1596 <= 1;
int32_t x1684;
if (x1683) {
x1684 = 1;
} else {
x1684 = x1596;
}
int32_t x1690 = x1684 * x1684;
int32_t x1695;
if (x1671) {
x1695 = 0;
} else {
x1695 = x1596;
}
int32_t x1696;
if (x1671) {
x1696 = 0;
} else {
x1696 = 1;
}
float* x164 = x5+43200;
bool x1759 = x1684 == 1;
bool x1760 = x1759 || true;
bool x1761 = x1760 || x1759;
bool x1771 = x1684 <= 1;
int32_t x1772;
if (x1771) {
x1772 = 1;
} else {
x1772 = x1684;
}
int32_t x1778 = x1772 * x1772;
int32_t x1783;
if (x1759) {
x1783 = 0;
} else {
x1783 = x1684;
}
int32_t x1784;
if (x1759) {
x1784 = 0;
} else {
x1784 = 1;
}
float* x49 = x5+43264;
int32_t x1831 = x1772 - 1;
int32_t x1832 = x1831 / 1;
int32_t x1833 = x1832 + 1;
int32_t x1837 = 16384 * x1833;
int32_t x1838 = x1837 * x1833;
int32_t x1834 = x1833 * x1833;
int32_t x1835 = 256 * x1834;
float* x32 = x5+43456;
bool x1912 = x1833 == 1;
bool x1913 = x1912 || true;
bool x1914 = x1913 || x1912;
bool x1924 = x1833 <= 1;
int32_t x1925;
if (x1924) {
x1925 = 1;
} else {
x1925 = x1833;
}
int32_t x1931 = x1925 * x1925;
int32_t x1935;
if (x1912) {
x1935 = 0;
} else {
x1935 = x1833;
}
int32_t x1936;
if (x1912) {
x1936 = 0;
} else {
x1936 = 1;
}
float* x71 = x5+60352;
float* x36 = x5+60608;
bool x2016 = x1925 == 1;
bool x2017 = x2016 || true;
bool x2018 = x2017 || x2016;
bool x2028 = x1925 <= 1;
int32_t x2029;
if (x2028) {
x2029 = 1;
} else {
x2029 = x1925;
}
int32_t x2035 = x2029 * x2029;
int32_t x2040;
if (x2016) {
x2040 = 0;
} else {
x2040 = x1925;
}
int32_t x2041;
if (x2016) {
x2041 = 0;
} else {
x2041 = 1;
}
bool x2104 = x2029 == 1;
bool x2105 = x2104 || true;
bool x2106 = x2105 || x2104;
bool x2116 = x2029 <= 1;
int32_t x2117;
if (x2116) {
x2117 = 1;
} else {
x2117 = x2029;
}
int32_t x2123 = x2117 * x2117;
int32_t x2128;
if (x2104) {
x2128 = 0;
} else {
x2128 = x2029;
}
int32_t x2129;
if (x2104) {
x2129 = 0;
} else {
x2129 = 1;
}
float* x199 = x5+59840;
bool x2192 = x2117 == 1;
bool x2193 = x2192 || true;
bool x2194 = x2193 || x2192;
bool x2204 = x2117 <= 1;
int32_t x2205;
if (x2204) {
x2205 = 1;
} else {
x2205 = x2117;
}
int32_t x2211 = x2205 * x2205;
int32_t x2216;
if (x2192) {
x2216 = 0;
} else {
x2216 = x2117;
}
int32_t x2217;
if (x2192) {
x2217 = 0;
} else {
x2217 = 1;
}
float* x126 = x5+60096;
int32_t x2253 = 16384 * x921;
int32_t x2254 = x2253 * x921;
int32_t x2251 = 256 * x922;
float* x162 = x5+60864;
float* x264 = x5+77760;
float* x243 = x5+78016;
float* x76 = x5+77248;
float* x203 = x5+77504;
bool x2626 = x2205 == 1;
bool x2627 = x1291 == 1;
bool x2628 = x2626 || x2627;
bool x2629 = x2205 == x1291;
bool x2630 = x2628 || x2629;
bool x2640 = x2205 <= x1291;
int32_t x2641;
if (x2640) {
x2641 = x1291;
} else {
x2641 = x2205;
}
int32_t x2656;
if (x2626) {
x2656 = 0;
} else {
x2656 = x2205;
}
int32_t x2657;
if (x2626) {
x2657 = 0;
} else {
x2657 = 1;
}
int32_t x2659;
if (x2627) {
x2659 = 0;
} else {
x2659 = x1291;
}
int32_t x2660;
if (x2627) {
x2660 = 0;
} else {
x2660 = 1;
}
int32_t x2706 = x2205 - 1;
int32_t x2707 = x2706 / 1;
int32_t x2708 = x2707 + 1;
int32_t x2712 = 4096 * x2708;
int32_t x2713 = x2712 * x2708;
int32_t x2709 = x2708 * x2708;
int32_t x2710 = 64 * x2709;
float* x171 = x5+78272;
bool x2787 = x2708 == 1;
bool x2788 = x2787 || true;
bool x2789 = x2788 || x2787;
bool x2799 = x2708 <= 1;
int32_t x2800;
if (x2799) {
x2800 = 1;
} else {
x2800 = x2708;
}
int32_t x2806 = x2800 * x2800;
int32_t x2810;
if (x2787) {
x2810 = 0;
} else {
x2810 = x2708;
}
int32_t x2811;
if (x2787) {
x2811 = 0;
} else {
x2811 = 1;
}
float* x10 = x5+94784;
float* x102 = x5+94848;
bool x2890 = x2800 == 1;
bool x2891 = x2890 || true;
bool x2892 = x2891 || x2890;
bool x2902 = x2800 <= 1;
int32_t x2903;
if (x2902) {
x2903 = 1;
} else {
x2903 = x2800;
}
int32_t x2909 = x2903 * x2903;
int32_t x2914;
if (x2890) {
x2914 = 0;
} else {
x2914 = x2800;
}
int32_t x2915;
if (x2890) {
x2915 = 0;
} else {
x2915 = 1;
}
bool x2978 = x2903 == 1;
bool x2979 = x2978 || true;
bool x2980 = x2979 || x2978;
bool x2990 = x2903 <= 1;
int32_t x2991;
if (x2990) {
x2991 = 1;
} else {
x2991 = x2903;
}
int32_t x2997 = x2991 * x2991;
int32_t x3002;
if (x2978) {
x3002 = 0;
} else {
x3002 = x2903;
}
int32_t x3003;
if (x2978) {
x3003 = 0;
} else {
x3003 = 1;
}
float* x142 = x5+94656;
bool x3066 = x2991 == 1;
bool x3067 = x3066 || true;
bool x3068 = x3067 || x3066;
bool x3078 = x2991 <= 1;
int32_t x3079;
if (x3078) {
x3079 = 1;
} else {
x3079 = x2991;
}
int32_t x3085 = x3079 * x3079;
int32_t x3090;
if (x3066) {
x3090 = 0;
} else {
x3090 = x2991;
}
int32_t x3091;
if (x3066) {
x3091 = 0;
} else {
x3091 = 1;
}
float* x60 = x5+94720;
int32_t x3138 = x3079 + 2;
int32_t x3139 = x3138 - 3;
int32_t x3140 = x3139 / 1;
int32_t x3141 = x3140 + 1;
int32_t x3145 = 4096 * x3141;
int32_t x3146 = x3145 * x3141;
int32_t x3142 = x3141 * x3141;
int32_t x3143 = 64 * x3142;
float* x83 = x5+94912;
bool x3268 = x3141 == 1;
bool x3269 = x3268 || true;
bool x3270 = x3269 || x3268;
bool x3280 = x3141 <= 1;
int32_t x3281;
if (x3280) {
x3281 = 1;
} else {
x3281 = x3141;
}
int32_t x3287 = x3281 * x3281;
int32_t x3291;
if (x3268) {
x3291 = 0;
} else {
x3291 = x3141;
}
int32_t x3292;
if (x3268) {
x3292 = 0;
} else {
x3292 = 1;
}
float* x44 = x5+131904;
float* x244 = x5+131968;
bool x3371 = x3281 == 1;
bool x3372 = x3371 || true;
bool x3373 = x3372 || x3371;
bool x3383 = x3281 <= 1;
int32_t x3384;
if (x3383) {
x3384 = 1;
} else {
x3384 = x3281;
}
int32_t x3390 = x3384 * x3384;
int32_t x3395;
if (x3371) {
x3395 = 0;
} else {
x3395 = x3281;
}
int32_t x3396;
if (x3371) {
x3396 = 0;
} else {
x3396 = 1;
}
bool x3459 = x3384 == 1;
bool x3460 = x3459 || true;
bool x3461 = x3460 || x3459;
bool x3471 = x3384 <= 1;
int32_t x3472;
if (x3471) {
x3472 = 1;
} else {
x3472 = x3384;
}
int32_t x3478 = x3472 * x3472;
int32_t x3483;
if (x3459) {
x3483 = 0;
} else {
x3483 = x3384;
}
int32_t x3484;
if (x3459) {
x3484 = 0;
} else {
x3484 = 1;
}
float* x208 = x5+131776;
bool x3547 = x3472 == 1;
bool x3548 = x3547 || true;
bool x3549 = x3548 || x3547;
bool x3559 = x3472 <= 1;
int32_t x3560;
if (x3559) {
x3560 = 1;
} else {
x3560 = x3472;
}
int32_t x3566 = x3560 * x3560;
int32_t x3571;
if (x3547) {
x3571 = 0;
} else {
x3571 = x3472;
}
int32_t x3572;
if (x3547) {
x3572 = 0;
} else {
x3572 = 1;
}
float* x153 = x5+131840;
int32_t x3619 = x3560 - 1;
int32_t x3620 = x3619 / 1;
int32_t x3621 = x3620 + 1;
int32_t x3625 = 16384 * x3621;
int32_t x3626 = x3625 * x3621;
int32_t x3622 = x3621 * x3621;
int32_t x3623 = 256 * x3622;
float* x130 = x5+132032;
bool x3700 = x3621 == 1;
bool x3701 = x3700 || true;
bool x3702 = x3701 || x3700;
bool x3712 = x3621 <= 1;
int32_t x3713;
if (x3712) {
x3713 = 1;
} else {
x3713 = x3621;
}
int32_t x3719 = x3713 * x3713;
int32_t x3723;
if (x3700) {
x3723 = 0;
} else {
x3723 = x3621;
}
int32_t x3724;
if (x3700) {
x3724 = 0;
} else {
x3724 = 1;
}
float* x91 = x5+148928;
float* x166 = x5+149184;
bool x3803 = x3713 == 1;
bool x3804 = x3803 || true;
bool x3805 = x3804 || x3803;
bool x3815 = x3713 <= 1;
int32_t x3816;
if (x3815) {
x3816 = 1;
} else {
x3816 = x3713;
}
int32_t x3822 = x3816 * x3816;
int32_t x3827;
if (x3803) {
x3827 = 0;
} else {
x3827 = x3713;
}
int32_t x3828;
if (x3803) {
x3828 = 0;
} else {
x3828 = 1;
}
bool x3891 = x3816 == 1;
bool x3892 = x3891 || true;
bool x3893 = x3892 || x3891;
bool x3903 = x3816 <= 1;
int32_t x3904;
if (x3903) {
x3904 = 1;
} else {
x3904 = x3816;
}
int32_t x3910 = x3904 * x3904;
int32_t x3915;
if (x3891) {
x3915 = 0;
} else {
x3915 = x3816;
}
int32_t x3916;
if (x3891) {
x3916 = 0;
} else {
x3916 = 1;
}
float* x58 = x5+148416;
bool x3979 = x3904 == 1;
bool x3980 = x3979 || true;
bool x3981 = x3980 || x3979;
bool x3991 = x3904 <= 1;
int32_t x3992;
if (x3991) {
x3992 = 1;
} else {
x3992 = x3904;
}
int32_t x3998 = x3992 * x3992;
int32_t x4003;
if (x3979) {
x4003 = 0;
} else {
x4003 = x3904;
}
int32_t x4004;
if (x3979) {
x4004 = 0;
} else {
x4004 = 1;
}
float* x7 = x5+148672;
bool x4042 = x3992 == 1;
bool x4043 = x4042 || x2626;
bool x4044 = x3992 == x2205;
bool x4045 = x4043 || x4044;
bool x4055 = x3992 <= x2205;
int32_t x4056;
if (x4055) {
x4056 = x2205;
} else {
x4056 = x3992;
}
int32_t x4071;
if (x4042) {
x4071 = 0;
} else {
x4071 = x3992;
}
int32_t x4072;
if (x4042) {
x4072 = 0;
} else {
x4072 = 1;
}
int32_t x4118 = x3992 - 1;
int32_t x4119 = x4118 / 1;
int32_t x4120 = x4119 + 1;
int32_t x4124 = 4096 * x4120;
int32_t x4125 = x4124 * x4120;
int32_t x4121 = x4120 * x4120;
int32_t x4122 = 64 * x4121;
float* x150 = x5+149440;
bool x4199 = x4120 == 1;
bool x4200 = x4199 || true;
bool x4201 = x4200 || x4199;
bool x4211 = x4120 <= 1;
int32_t x4212;
if (x4211) {
x4212 = 1;
} else {
x4212 = x4120;
}
int32_t x4218 = x4212 * x4212;
int32_t x4222;
if (x4199) {
x4222 = 0;
} else {
x4222 = x4120;
}
int32_t x4223;
if (x4199) {
x4223 = 0;
} else {
x4223 = 1;
}
float* x257 = x5+165952;
float* x187 = x5+166016;
bool x4302 = x4212 == 1;
bool x4303 = x4302 || true;
bool x4304 = x4303 || x4302;
bool x4314 = x4212 <= 1;
int32_t x4315;
if (x4314) {
x4315 = 1;
} else {
x4315 = x4212;
}
int32_t x4321 = x4315 * x4315;
int32_t x4326;
if (x4302) {
x4326 = 0;
} else {
x4326 = x4212;
}
int32_t x4327;
if (x4302) {
x4327 = 0;
} else {
x4327 = 1;
}
bool x4390 = x4315 == 1;
bool x4391 = x4390 || true;
bool x4392 = x4391 || x4390;
bool x4402 = x4315 <= 1;
int32_t x4403;
if (x4402) {
x4403 = 1;
} else {
x4403 = x4315;
}
int32_t x4409 = x4403 * x4403;
int32_t x4414;
if (x4390) {
x4414 = 0;
} else {
x4414 = x4315;
}
int32_t x4415;
if (x4390) {
x4415 = 0;
} else {
x4415 = 1;
}
float* x81 = x5+165824;
bool x4478 = x4403 == 1;
bool x4479 = x4478 || true;
bool x4480 = x4479 || x4478;
bool x4490 = x4403 <= 1;
int32_t x4491;
if (x4490) {
x4491 = 1;
} else {
x4491 = x4403;
}
int32_t x4497 = x4491 * x4491;
int32_t x4502;
if (x4478) {
x4502 = 0;
} else {
x4502 = x4403;
}
int32_t x4503;
if (x4478) {
x4503 = 0;
} else {
x4503 = 1;
}
float* x24 = x5+165888;
int32_t x4550 = x4491 + 2;
int32_t x4551 = x4550 - 3;
int32_t x4552 = x4551 / 1;
int32_t x4553 = x4552 + 1;
int32_t x4557 = 4096 * x4553;
int32_t x4558 = x4557 * x4553;
int32_t x4554 = x4553 * x4553;
int32_t x4555 = 64 * x4554;
float* x73 = x5+166080;
bool x4680 = x4553 == 1;
bool x4681 = x4680 || true;
bool x4682 = x4681 || x4680;
bool x4692 = x4553 <= 1;
int32_t x4693;
if (x4692) {
x4693 = 1;
} else {
x4693 = x4553;
}
int32_t x4699 = x4693 * x4693;
int32_t x4703;
if (x4680) {
x4703 = 0;
} else {
x4703 = x4553;
}
int32_t x4704;
if (x4680) {
x4704 = 0;
} else {
x4704 = 1;
}
float* x179 = x5+203072;
float* x118 = x5+203136;
bool x4783 = x4693 == 1;
bool x4784 = x4783 || true;
bool x4785 = x4784 || x4783;
bool x4795 = x4693 <= 1;
int32_t x4796;
if (x4795) {
x4796 = 1;
} else {
x4796 = x4693;
}
int32_t x4802 = x4796 * x4796;
int32_t x4807;
if (x4783) {
x4807 = 0;
} else {
x4807 = x4693;
}
int32_t x4808;
if (x4783) {
x4808 = 0;
} else {
x4808 = 1;
}
bool x4871 = x4796 == 1;
bool x4872 = x4871 || true;
bool x4873 = x4872 || x4871;
bool x4883 = x4796 <= 1;
int32_t x4884;
if (x4883) {
x4884 = 1;
} else {
x4884 = x4796;
}
int32_t x4890 = x4884 * x4884;
int32_t x4895;
if (x4871) {
x4895 = 0;
} else {
x4895 = x4796;
}
int32_t x4896;
if (x4871) {
x4896 = 0;
} else {
x4896 = 1;
}
float* x72 = x5+202944;
bool x4959 = x4884 == 1;
bool x4960 = x4959 || true;
bool x4961 = x4960 || x4959;
bool x4971 = x4884 <= 1;
int32_t x4972;
if (x4971) {
x4972 = 1;
} else {
x4972 = x4884;
}
int32_t x4978 = x4972 * x4972;
int32_t x4983;
if (x4959) {
x4983 = 0;
} else {
x4983 = x4884;
}
int32_t x4984;
if (x4959) {
x4984 = 0;
} else {
x4984 = 1;
}
float* x135 = x5+203008;
int32_t x5031 = x4972 - 1;
int32_t x5032 = x5031 / 1;
int32_t x5033 = x5032 + 1;
int32_t x5037 = 16384 * x5033;
int32_t x5038 = x5037 * x5033;
int32_t x5034 = x5033 * x5033;
int32_t x5035 = 256 * x5034;
float* x87 = x5+203200;
bool x5112 = x5033 == 1;
bool x5113 = x5112 || true;
bool x5114 = x5113 || x5112;
bool x5124 = x5033 <= 1;
int32_t x5125;
if (x5124) {
x5125 = 1;
} else {
x5125 = x5033;
}
int32_t x5131 = x5125 * x5125;
int32_t x5135;
if (x5112) {
x5135 = 0;
} else {
x5135 = x5033;
}
int32_t x5136;
if (x5112) {
x5136 = 0;
} else {
x5136 = 1;
}
float* x184 = x5+220096;
float* x133 = x5+220352;
bool x5215 = x5125 == 1;
bool x5216 = x5215 || true;
bool x5217 = x5216 || x5215;
bool x5227 = x5125 <= 1;
int32_t x5228;
if (x5227) {
x5228 = 1;
} else {
x5228 = x5125;
}
int32_t x5234 = x5228 * x5228;
int32_t x5239;
if (x5215) {
x5239 = 0;
} else {
x5239 = x5125;
}
int32_t x5240;
if (x5215) {
x5240 = 0;
} else {
x5240 = 1;
}
bool x5303 = x5228 == 1;
bool x5304 = x5303 || true;
bool x5305 = x5304 || x5303;
bool x5315 = x5228 <= 1;
int32_t x5316;
if (x5315) {
x5316 = 1;
} else {
x5316 = x5228;
}
int32_t x5322 = x5316 * x5316;
int32_t x5327;
if (x5303) {
x5327 = 0;
} else {
x5327 = x5228;
}
int32_t x5328;
if (x5303) {
x5328 = 0;
} else {
x5328 = 1;
}
float* x37 = x5+219584;
bool x5391 = x5316 == 1;
bool x5392 = x5391 || true;
bool x5393 = x5392 || x5391;
bool x5403 = x5316 <= 1;
int32_t x5404;
if (x5403) {
x5404 = 1;
} else {
x5404 = x5316;
}
int32_t x5410 = x5404 * x5404;
int32_t x5415;
if (x5391) {
x5415 = 0;
} else {
x5415 = x5316;
}
int32_t x5416;
if (x5391) {
x5416 = 0;
} else {
x5416 = 1;
}
float* x247 = x5+219840;
bool x5454 = x5404 == 1;
bool x5455 = x5454 || x4042;
bool x5456 = x5404 == x3992;
bool x5457 = x5455 || x5456;
bool x5467 = x5404 <= x3992;
int32_t x5468;
if (x5467) {
x5468 = x3992;
} else {
x5468 = x5404;
}
int32_t x5483;
if (x5454) {
x5483 = 0;
} else {
x5483 = x5404;
}
int32_t x5484;
if (x5454) {
x5484 = 0;
} else {
x5484 = 1;
}
int32_t x5530 = x5404 - 1;
int32_t x5531 = x5530 / 1;
int32_t x5532 = x5531 + 1;
int32_t x5536 = 8192 * x5532;
int32_t x5537 = x5536 * x5532;
int32_t x5533 = x5532 * x5532;
int32_t x5534 = 128 * x5533;
float* x11 = x5+220608;
bool x5611 = x5532 == 1;
bool x5612 = x5611 || true;
bool x5613 = x5612 || x5611;
bool x5623 = x5532 <= 1;
int32_t x5624;
if (x5623) {
x5624 = 1;
} else {
x5624 = x5532;
}
int32_t x5630 = x5624 * x5624;
int32_t x5634;
if (x5611) {
x5634 = 0;
} else {
x5634 = x5532;
}
int32_t x5635;
if (x5611) {
x5635 = 0;
} else {
x5635 = 1;
}
float* x204 = x5+253632;
float* x134 = x5+253760;
bool x5715 = x5624 == 1;
bool x5716 = x5715 || true;
bool x5717 = x5716 || x5715;
bool x5727 = x5624 <= 1;
int32_t x5728;
if (x5727) {
x5728 = 1;
} else {
x5728 = x5624;
}
int32_t x5734 = x5728 * x5728;
int32_t x5739;
if (x5715) {
x5739 = 0;
} else {
x5739 = x5624;
}
int32_t x5740;
if (x5715) {
x5740 = 0;
} else {
x5740 = 1;
}
bool x5803 = x5728 == 1;
bool x5804 = x5803 || true;
bool x5805 = x5804 || x5803;
bool x5815 = x5728 <= 1;
int32_t x5816;
if (x5815) {
x5816 = 1;
} else {
x5816 = x5728;
}
int32_t x5822 = x5816 * x5816;
int32_t x5827;
if (x5803) {
x5827 = 0;
} else {
x5827 = x5728;
}
int32_t x5828;
if (x5803) {
x5828 = 0;
} else {
x5828 = 1;
}
float* x84 = x5+253376;
bool x5891 = x5816 == 1;
bool x5892 = x5891 || true;
bool x5893 = x5892 || x5891;
bool x5903 = x5816 <= 1;
int32_t x5904;
if (x5903) {
x5904 = 1;
} else {
x5904 = x5816;
}
int32_t x5910 = x5904 * x5904;
int32_t x5915;
if (x5891) {
x5915 = 0;
} else {
x5915 = x5816;
}
int32_t x5916;
if (x5891) {
x5916 = 0;
} else {
x5916 = 1;
}
float* x172 = x5+253504;
int32_t x5963 = x5904 + 2;
int32_t x5964 = x5963 - 3;
int32_t x5965 = x5964 / 2;
int32_t x5966 = x5965 + 1;
int32_t x5970 = 8192 * x5966;
int32_t x5971 = x5970 * x5966;
int32_t x5967 = x5966 * x5966;
int32_t x5968 = 128 * x5967;
float* x27 = x5+253888;
bool x6077 = x5966 == 1;
bool x6078 = x6077 || true;
bool x6079 = x6078 || x6077;
bool x6089 = x5966 <= 1;
int32_t x6090;
if (x6089) {
x6090 = 1;
} else {
x6090 = x5966;
}
int32_t x6096 = x6090 * x6090;
int32_t x6100;
if (x6077) {
x6100 = 0;
} else {
x6100 = x5966;
}
int32_t x6101;
if (x6077) {
x6101 = 0;
} else {
x6101 = 1;
}
float* x128 = x5+401600;
float* x43 = x5+401728;
bool x6180 = x6090 == 1;
bool x6181 = x6180 || true;
bool x6182 = x6181 || x6180;
bool x6192 = x6090 <= 1;
int32_t x6193;
if (x6192) {
x6193 = 1;
} else {
x6193 = x6090;
}
int32_t x6199 = x6193 * x6193;
int32_t x6204;
if (x6180) {
x6204 = 0;
} else {
x6204 = x6090;
}
int32_t x6205;
if (x6180) {
x6205 = 0;
} else {
x6205 = 1;
}
bool x6268 = x6193 == 1;
bool x6269 = x6268 || true;
bool x6270 = x6269 || x6268;
bool x6280 = x6193 <= 1;
int32_t x6281;
if (x6280) {
x6281 = 1;
} else {
x6281 = x6193;
}
int32_t x6287 = x6281 * x6281;
int32_t x6292;
if (x6268) {
x6292 = 0;
} else {
x6292 = x6193;
}
int32_t x6293;
if (x6268) {
x6293 = 0;
} else {
x6293 = 1;
}
float* x252 = x5+401344;
bool x6356 = x6281 == 1;
bool x6357 = x6356 || true;
bool x6358 = x6357 || x6356;
bool x6368 = x6281 <= 1;
int32_t x6369;
if (x6368) {
x6369 = 1;
} else {
x6369 = x6281;
}
int32_t x6375 = x6369 * x6369;
int32_t x6380;
if (x6356) {
x6380 = 0;
} else {
x6380 = x6281;
}
int32_t x6381;
if (x6356) {
x6381 = 0;
} else {
x6381 = 1;
}
float* x190 = x5+401472;
int32_t x6428 = x6369 - 1;
int32_t x6429 = x6428 / 1;
int32_t x6430 = x6429 + 1;
int32_t x6434 = 32768 * x6430;
int32_t x6435 = x6434 * x6430;
int32_t x6431 = x6430 * x6430;
int32_t x6432 = 512 * x6431;
float* x106 = x5+401856;
bool x6509 = x6430 == 1;
bool x6510 = x6509 || true;
bool x6511 = x6510 || x6509;
bool x6521 = x6430 <= 1;
int32_t x6522;
if (x6521) {
x6522 = 1;
} else {
x6522 = x6430;
}
int32_t x6528 = x6522 * x6522;
int32_t x6532;
if (x6509) {
x6532 = 0;
} else {
x6532 = x6430;
}
int32_t x6533;
if (x6509) {
x6533 = 0;
} else {
x6533 = 1;
}
float* x149 = x5+468416;
float* x101 = x5+468928;
bool x6613 = x6522 == 1;
bool x6614 = x6613 || true;
bool x6615 = x6614 || x6613;
bool x6625 = x6522 <= 1;
int32_t x6626;
if (x6625) {
x6626 = 1;
} else {
x6626 = x6522;
}
int32_t x6632 = x6626 * x6626;
int32_t x6637;
if (x6613) {
x6637 = 0;
} else {
x6637 = x6522;
}
int32_t x6638;
if (x6613) {
x6638 = 0;
} else {
x6638 = 1;
}
bool x6701 = x6626 == 1;
bool x6702 = x6701 || true;
bool x6703 = x6702 || x6701;
bool x6713 = x6626 <= 1;
int32_t x6714;
if (x6713) {
x6714 = 1;
} else {
x6714 = x6626;
}
int32_t x6720 = x6714 * x6714;
int32_t x6725;
if (x6701) {
x6725 = 0;
} else {
x6725 = x6626;
}
int32_t x6726;
if (x6701) {
x6726 = 0;
} else {
x6726 = 1;
}
float* x145 = x5+467392;
bool x6789 = x6714 == 1;
bool x6790 = x6789 || true;
bool x6791 = x6790 || x6789;
bool x6801 = x6714 <= 1;
int32_t x6802;
if (x6801) {
x6802 = 1;
} else {
x6802 = x6714;
}
int32_t x6808 = x6802 * x6802;
int32_t x6813;
if (x6789) {
x6813 = 0;
} else {
x6813 = x6714;
}
int32_t x6814;
if (x6789) {
x6814 = 0;
} else {
x6814 = 1;
}
float* x210 = x5+467904;
int32_t x6848 = x5530 / 2;
int32_t x6849 = x6848 + 1;
int32_t x6853 = 32768 * x6849;
int32_t x6854 = x6853 * x6849;
int32_t x6850 = x6849 * x6849;
int32_t x6851 = 512 * x6850;
float* x258 = x5+469440;
bool x6934 = x6849 == 1;
bool x6935 = x6934 || true;
bool x6936 = x6935 || x6934;
bool x6946 = x6849 <= 1;
int32_t x6947;
if (x6946) {
x6947 = 1;
} else {
x6947 = x6849;
}
int32_t x6953 = x6947 * x6947;
int32_t x6957;
if (x6934) {
x6957 = 0;
} else {
x6957 = x6849;
}
int32_t x6958;
if (x6934) {
x6958 = 0;
} else {
x6958 = 1;
}
float* x42 = x5+601536;
float* x23 = x5+602048;
bool x7037 = x6947 == 1;
bool x7038 = x7037 || true;
bool x7039 = x7038 || x7037;
bool x7049 = x6947 <= 1;
int32_t x7050;
if (x7049) {
x7050 = 1;
} else {
x7050 = x6947;
}
int32_t x7056 = x7050 * x7050;
int32_t x7061;
if (x7037) {
x7061 = 0;
} else {
x7061 = x6947;
}
int32_t x7062;
if (x7037) {
x7062 = 0;
} else {
x7062 = 1;
}
bool x7125 = x7050 == 1;
bool x7126 = x7125 || true;
bool x7127 = x7126 || x7125;
bool x7137 = x7050 <= 1;
int32_t x7138;
if (x7137) {
x7138 = 1;
} else {
x7138 = x7050;
}
int32_t x7144 = x7138 * x7138;
int32_t x7149;
if (x7125) {
x7149 = 0;
} else {
x7149 = x7050;
}
int32_t x7150;
if (x7125) {
x7150 = 0;
} else {
x7150 = 1;
}
float* x207 = x5+600512;
bool x7213 = x7138 == 1;
bool x7214 = x7213 || true;
bool x7215 = x7214 || x7213;
bool x7225 = x7138 <= 1;
int32_t x7226;
if (x7225) {
x7226 = 1;
} else {
x7226 = x7138;
}
int32_t x7232 = x7226 * x7226;
int32_t x7237;
if (x7213) {
x7237 = 0;
} else {
x7237 = x7138;
}
int32_t x7238;
if (x7213) {
x7238 = 0;
} else {
x7238 = 1;
}
float* x119 = x5+601024;
bool x7277 = x6802 == 1;
bool x7278 = x7226 == 1;
bool x7279 = x7277 || x7278;
bool x7280 = x6802 == x7226;
bool x7281 = x7279 || x7280;
bool x7291 = x6802 <= x7226;
int32_t x7292;
if (x7291) {
x7292 = x7226;
} else {
x7292 = x6802;
}
int32_t x7307;
if (x7277) {
x7307 = 0;
} else {
x7307 = x6802;
}
int32_t x7308;
if (x7277) {
x7308 = 0;
} else {
x7308 = 1;
}
int32_t x7310;
if (x7278) {
x7310 = 0;
} else {
x7310 = x7226;
}
int32_t x7311;
if (x7278) {
x7311 = 0;
} else {
x7311 = 1;
}
int32_t x7357 = x6802 - 1;
int32_t x7358 = x7357 / 1;
int32_t x7359 = x7358 + 1;
int32_t x7363 = 8192 * x7359;
int32_t x7364 = x7363 * x7359;
int32_t x7360 = x7359 * x7359;
int32_t x7361 = 128 * x7360;
float* x256 = x5+602560;
bool x7438 = x7359 == 1;
bool x7439 = x7438 || true;
bool x7440 = x7439 || x7438;
bool x7450 = x7359 <= 1;
int32_t x7451;
if (x7450) {
x7451 = 1;
} else {
x7451 = x7359;
}
int32_t x7457 = x7451 * x7451;
int32_t x7461;
if (x7438) {
x7461 = 0;
} else {
x7461 = x7359;
}
int32_t x7462;
if (x7438) {
x7462 = 0;
} else {
x7462 = 1;
}
float* x100 = x5+668352;
float* x177 = x5+668480;
bool x7541 = x7451 == 1;
bool x7542 = x7541 || true;
bool x7543 = x7542 || x7541;
bool x7553 = x7451 <= 1;
int32_t x7554;
if (x7553) {
x7554 = 1;
} else {
x7554 = x7451;
}
int32_t x7560 = x7554 * x7554;
int32_t x7565;
if (x7541) {
x7565 = 0;
} else {
x7565 = x7451;
}
int32_t x7566;
if (x7541) {
x7566 = 0;
} else {
x7566 = 1;
}
bool x7629 = x7554 == 1;
bool x7630 = x7629 || true;
bool x7631 = x7630 || x7629;
bool x7641 = x7554 <= 1;
int32_t x7642;
if (x7641) {
x7642 = 1;
} else {
x7642 = x7554;
}
int32_t x7648 = x7642 * x7642;
int32_t x7653;
if (x7629) {
x7653 = 0;
} else {
x7653 = x7554;
}
int32_t x7654;
if (x7629) {
x7654 = 0;
} else {
x7654 = 1;
}
float* x222 = x5+668096;
bool x7717 = x7642 == 1;
bool x7718 = x7717 || true;
bool x7719 = x7718 || x7717;
bool x7729 = x7642 <= 1;
int32_t x7730;
if (x7729) {
x7730 = 1;
} else {
x7730 = x7642;
}
int32_t x7736 = x7730 * x7730;
int32_t x7741;
if (x7717) {
x7741 = 0;
} else {
x7741 = x7642;
}
int32_t x7742;
if (x7717) {
x7742 = 0;
} else {
x7742 = 1;
}
float* x17 = x5+668224;
int32_t x7789 = x7730 + 2;
int32_t x7790 = x7789 - 3;
int32_t x7791 = x7790 / 1;
int32_t x7792 = x7791 + 1;
int32_t x7796 = 8192 * x7792;
int32_t x7797 = x7796 * x7792;
int32_t x7793 = x7792 * x7792;
int32_t x7794 = 128 * x7793;
float* x235 = x5+668608;
bool x7919 = x7792 == 1;
bool x7920 = x7919 || true;
bool x7921 = x7920 || x7919;
bool x7931 = x7792 <= 1;
int32_t x7932;
if (x7931) {
x7932 = 1;
} else {
x7932 = x7792;
}
int32_t x7938 = x7932 * x7932;
int32_t x7942;
if (x7919) {
x7942 = 0;
} else {
x7942 = x7792;
}
int32_t x7943;
if (x7919) {
x7943 = 0;
} else {
x7943 = 1;
}
float* x35 = x5+816320;
float* x225 = x5+816448;
bool x8022 = x7932 == 1;
bool x8023 = x8022 || true;
bool x8024 = x8023 || x8022;
bool x8034 = x7932 <= 1;
int32_t x8035;
if (x8034) {
x8035 = 1;
} else {
x8035 = x7932;
}
int32_t x8041 = x8035 * x8035;
int32_t x8046;
if (x8022) {
x8046 = 0;
} else {
x8046 = x7932;
}
int32_t x8047;
if (x8022) {
x8047 = 0;
} else {
x8047 = 1;
}
bool x8110 = x8035 == 1;
bool x8111 = x8110 || true;
bool x8112 = x8111 || x8110;
bool x8122 = x8035 <= 1;
int32_t x8123;
if (x8122) {
x8123 = 1;
} else {
x8123 = x8035;
}
int32_t x8129 = x8123 * x8123;
int32_t x8134;
if (x8110) {
x8134 = 0;
} else {
x8134 = x8035;
}
int32_t x8135;
if (x8110) {
x8135 = 0;
} else {
x8135 = 1;
}
float* x8 = x5+816064;
bool x8198 = x8123 == 1;
bool x8199 = x8198 || true;
bool x8200 = x8199 || x8198;
bool x8210 = x8123 <= 1;
int32_t x8211;
if (x8210) {
x8211 = 1;
} else {
x8211 = x8123;
}
int32_t x8217 = x8211 * x8211;
int32_t x8222;
if (x8198) {
x8222 = 0;
} else {
x8222 = x8123;
}
int32_t x8223;
if (x8198) {
x8223 = 0;
} else {
x8223 = 1;
}
float* x95 = x5+816192;
int32_t x8270 = x8211 - 1;
int32_t x8271 = x8270 / 1;
int32_t x8272 = x8271 + 1;
int32_t x8276 = 32768 * x8272;
int32_t x8277 = x8276 * x8272;
int32_t x8273 = x8272 * x8272;
int32_t x8274 = 512 * x8273;
float* x111 = x5+816576;
bool x8351 = x8272 == 1;
bool x8352 = x8351 || true;
bool x8353 = x8352 || x8351;
bool x8363 = x8272 <= 1;
int32_t x8364;
if (x8363) {
x8364 = 1;
} else {
x8364 = x8272;
}
int32_t x8370 = x8364 * x8364;
int32_t x8374;
if (x8351) {
x8374 = 0;
} else {
x8374 = x8272;
}
int32_t x8375;
if (x8351) {
x8375 = 0;
} else {
x8375 = 1;
}
float* x147 = x5+883136;
float* x88 = x5+883648;
bool x8454 = x8364 == 1;
bool x8455 = x8454 || true;
bool x8456 = x8455 || x8454;
bool x8466 = x8364 <= 1;
int32_t x8467;
if (x8466) {
x8467 = 1;
} else {
x8467 = x8364;
}
int32_t x8473 = x8467 * x8467;
int32_t x8478;
if (x8454) {
x8478 = 0;
} else {
x8478 = x8364;
}
int32_t x8479;
if (x8454) {
x8479 = 0;
} else {
x8479 = 1;
}
bool x8542 = x8467 == 1;
bool x8543 = x8542 || true;
bool x8544 = x8543 || x8542;
bool x8554 = x8467 <= 1;
int32_t x8555;
if (x8554) {
x8555 = 1;
} else {
x8555 = x8467;
}
int32_t x8561 = x8555 * x8555;
int32_t x8566;
if (x8542) {
x8566 = 0;
} else {
x8566 = x8467;
}
int32_t x8567;
if (x8542) {
x8567 = 0;
} else {
x8567 = 1;
}
float* x52 = x5+882112;
bool x8630 = x8555 == 1;
bool x8631 = x8630 || true;
bool x8632 = x8631 || x8630;
bool x8642 = x8555 <= 1;
int32_t x8643;
if (x8642) {
x8643 = 1;
} else {
x8643 = x8555;
}
int32_t x8649 = x8643 * x8643;
int32_t x8654;
if (x8630) {
x8654 = 0;
} else {
x8654 = x8555;
}
int32_t x8655;
if (x8630) {
x8655 = 0;
} else {
x8655 = 1;
}
float* x246 = x5+882624;
bool x8693 = x8643 == 1;
bool x8694 = x8693 || x7277;
bool x8695 = x8643 == x6802;
bool x8696 = x8694 || x8695;
bool x8706 = x8643 <= x6802;
int32_t x8707;
if (x8706) {
x8707 = x6802;
} else {
x8707 = x8643;
}
int32_t x8722;
if (x8693) {
x8722 = 0;
} else {
x8722 = x8643;
}
int32_t x8723;
if (x8693) {
x8723 = 0;
} else {
x8723 = 1;
}
int32_t x8769 = x8643 - 1;
int32_t x8770 = x8769 / 1;
int32_t x8771 = x8770 + 1;
int32_t x8775 = 8192 * x8771;
int32_t x8776 = x8775 * x8771;
int32_t x8772 = x8771 * x8771;
int32_t x8773 = 128 * x8772;
float* x196 = x5+884160;
bool x8850 = x8771 == 1;
bool x8851 = x8850 || true;
bool x8852 = x8851 || x8850;
bool x8862 = x8771 <= 1;
int32_t x8863;
if (x8862) {
x8863 = 1;
} else {
x8863 = x8771;
}
int32_t x8869 = x8863 * x8863;
int32_t x8873;
if (x8850) {
x8873 = 0;
} else {
x8873 = x8771;
}
int32_t x8874;
if (x8850) {
x8874 = 0;
} else {
x8874 = 1;
}
float* x112 = x5+949952;
float* x9 = x5+950080;
bool x8953 = x8863 == 1;
bool x8954 = x8953 || true;
bool x8955 = x8954 || x8953;
bool x8965 = x8863 <= 1;
int32_t x8966;
if (x8965) {
x8966 = 1;
} else {
x8966 = x8863;
}
int32_t x8972 = x8966 * x8966;
int32_t x8977;
if (x8953) {
x8977 = 0;
} else {
x8977 = x8863;
}
int32_t x8978;
if (x8953) {
x8978 = 0;
} else {
x8978 = 1;
}
bool x9041 = x8966 == 1;
bool x9042 = x9041 || true;
bool x9043 = x9042 || x9041;
bool x9053 = x8966 <= 1;
int32_t x9054;
if (x9053) {
x9054 = 1;
} else {
x9054 = x8966;
}
int32_t x9060 = x9054 * x9054;
int32_t x9065;
if (x9041) {
x9065 = 0;
} else {
x9065 = x8966;
}
int32_t x9066;
if (x9041) {
x9066 = 0;
} else {
x9066 = 1;
}
float* x45 = x5+949696;
bool x9129 = x9054 == 1;
bool x9130 = x9129 || true;
bool x9131 = x9130 || x9129;
bool x9141 = x9054 <= 1;
int32_t x9142;
if (x9141) {
x9142 = 1;
} else {
x9142 = x9054;
}
int32_t x9148 = x9142 * x9142;
int32_t x9153;
if (x9129) {
x9153 = 0;
} else {
x9153 = x9054;
}
int32_t x9154;
if (x9129) {
x9154 = 0;
} else {
x9154 = 1;
}
float* x170 = x5+949824;
int32_t x9201 = x9142 + 2;
int32_t x9202 = x9201 - 3;
int32_t x9203 = x9202 / 1;
int32_t x9204 = x9203 + 1;
int32_t x9208 = 8192 * x9204;
int32_t x9209 = x9208 * x9204;
int32_t x9205 = x9204 * x9204;
int32_t x9206 = 128 * x9205;
float* x191 = x5+950208;
bool x9331 = x9204 == 1;
bool x9332 = x9331 || true;
bool x9333 = x9332 || x9331;
bool x9343 = x9204 <= 1;
int32_t x9344;
if (x9343) {
x9344 = 1;
} else {
x9344 = x9204;
}
int32_t x9350 = x9344 * x9344;
int32_t x9354;
if (x9331) {
x9354 = 0;
} else {
x9354 = x9204;
}
int32_t x9355;
if (x9331) {
x9355 = 0;
} else {
x9355 = 1;
}
float* x217 = x5+1097920;
float* x266 = x5+1098048;
bool x9434 = x9344 == 1;
bool x9435 = x9434 || true;
bool x9436 = x9435 || x9434;
bool x9446 = x9344 <= 1;
int32_t x9447;
if (x9446) {
x9447 = 1;
} else {
x9447 = x9344;
}
int32_t x9453 = x9447 * x9447;
int32_t x9458;
if (x9434) {
x9458 = 0;
} else {
x9458 = x9344;
}
int32_t x9459;
if (x9434) {
x9459 = 0;
} else {
x9459 = 1;
}
bool x9522 = x9447 == 1;
bool x9523 = x9522 || true;
bool x9524 = x9523 || x9522;
bool x9534 = x9447 <= 1;
int32_t x9535;
if (x9534) {
x9535 = 1;
} else {
x9535 = x9447;
}
int32_t x9541 = x9535 * x9535;
int32_t x9546;
if (x9522) {
x9546 = 0;
} else {
x9546 = x9447;
}
int32_t x9547;
if (x9522) {
x9547 = 0;
} else {
x9547 = 1;
}
float* x127 = x5+1097664;
bool x9610 = x9535 == 1;
bool x9611 = x9610 || true;
bool x9612 = x9611 || x9610;
bool x9622 = x9535 <= 1;
int32_t x9623;
if (x9622) {
x9623 = 1;
} else {
x9623 = x9535;
}
int32_t x9629 = x9623 * x9623;
int32_t x9634;
if (x9610) {
x9634 = 0;
} else {
x9634 = x9535;
}
int32_t x9635;
if (x9610) {
x9635 = 0;
} else {
x9635 = 1;
}
float* x61 = x5+1097792;
int32_t x9682 = x9623 - 1;
int32_t x9683 = x9682 / 1;
int32_t x9684 = x9683 + 1;
int32_t x9688 = 32768 * x9684;
int32_t x9689 = x9688 * x9684;
int32_t x9685 = x9684 * x9684;
int32_t x9686 = 512 * x9685;
float* x41 = x5+1098176;
bool x9763 = x9684 == 1;
bool x9764 = x9763 || true;
bool x9765 = x9764 || x9763;
bool x9775 = x9684 <= 1;
int32_t x9776;
if (x9775) {
x9776 = 1;
} else {
x9776 = x9684;
}
int32_t x9782 = x9776 * x9776;
int32_t x9786;
if (x9763) {
x9786 = 0;
} else {
x9786 = x9684;
}
int32_t x9787;
if (x9763) {
x9787 = 0;
} else {
x9787 = 1;
}
float* x25 = x5+1164736;
float* x223 = x5+1165248;
bool x9866 = x9776 == 1;
bool x9867 = x9866 || true;
bool x9868 = x9867 || x9866;
bool x9878 = x9776 <= 1;
int32_t x9879;
if (x9878) {
x9879 = 1;
} else {
x9879 = x9776;
}
int32_t x9885 = x9879 * x9879;
int32_t x9890;
if (x9866) {
x9890 = 0;
} else {
x9890 = x9776;
}
int32_t x9891;
if (x9866) {
x9891 = 0;
} else {
x9891 = 1;
}
bool x9954 = x9879 == 1;
bool x9955 = x9954 || true;
bool x9956 = x9955 || x9954;
bool x9966 = x9879 <= 1;
int32_t x9967;
if (x9966) {
x9967 = 1;
} else {
x9967 = x9879;
}
int32_t x9973 = x9967 * x9967;
int32_t x9978;
if (x9954) {
x9978 = 0;
} else {
x9978 = x9879;
}
int32_t x9979;
if (x9954) {
x9979 = 0;
} else {
x9979 = 1;
}
float* x167 = x5+1163712;
bool x10042 = x9967 == 1;
bool x10043 = x10042 || true;
bool x10044 = x10043 || x10042;
bool x10054 = x9967 <= 1;
int32_t x10055;
if (x10054) {
x10055 = 1;
} else {
x10055 = x9967;
}
int32_t x10061 = x10055 * x10055;
int32_t x10066;
if (x10042) {
x10066 = 0;
} else {
x10066 = x9967;
}
int32_t x10067;
if (x10042) {
x10067 = 0;
} else {
x10067 = 1;
}
float* x82 = x5+1164224;
bool x10105 = x10055 == 1;
bool x10106 = x10105 || x8693;
bool x10107 = x10055 == x8643;
bool x10108 = x10106 || x10107;
bool x10118 = x10055 <= x8643;
int32_t x10119;
if (x10118) {
x10119 = x8643;
} else {
x10119 = x10055;
}
int32_t x10134;
if (x10105) {
x10134 = 0;
} else {
x10134 = x10055;
}
int32_t x10135;
if (x10105) {
x10135 = 0;
} else {
x10135 = 1;
}
int32_t x10181 = x10055 - 1;
int32_t x10182 = x10181 / 1;
int32_t x10183 = x10182 + 1;
int32_t x10187 = 8192 * x10183;
int32_t x10188 = x10187 * x10183;
int32_t x10184 = x10183 * x10183;
int32_t x10185 = 128 * x10184;
float* x132 = x5+1165760;
bool x10262 = x10183 == 1;
bool x10263 = x10262 || true;
bool x10264 = x10263 || x10262;
bool x10274 = x10183 <= 1;
int32_t x10275;
if (x10274) {
x10275 = 1;
} else {
x10275 = x10183;
}
int32_t x10281 = x10275 * x10275;
int32_t x10285;
if (x10262) {
x10285 = 0;
} else {
x10285 = x10183;
}
int32_t x10286;
if (x10262) {
x10286 = 0;
} else {
x10286 = 1;
}
float* x236 = x5+1231552;
float* x261 = x5+1231680;
bool x10365 = x10275 == 1;
bool x10366 = x10365 || true;
bool x10367 = x10366 || x10365;
bool x10377 = x10275 <= 1;
int32_t x10378;
if (x10377) {
x10378 = 1;
} else {
x10378 = x10275;
}
int32_t x10384 = x10378 * x10378;
int32_t x10389;
if (x10365) {
x10389 = 0;
} else {
x10389 = x10275;
}
int32_t x10390;
if (x10365) {
x10390 = 0;
} else {
x10390 = 1;
}
bool x10453 = x10378 == 1;
bool x10454 = x10453 || true;
bool x10455 = x10454 || x10453;
bool x10465 = x10378 <= 1;
int32_t x10466;
if (x10465) {
x10466 = 1;
} else {
x10466 = x10378;
}
int32_t x10472 = x10466 * x10466;
int32_t x10477;
if (x10453) {
x10477 = 0;
} else {
x10477 = x10378;
}
int32_t x10478;
if (x10453) {
x10478 = 0;
} else {
x10478 = 1;
}
float* x39 = x5+1231296;
bool x10541 = x10466 == 1;
bool x10542 = x10541 || true;
bool x10543 = x10542 || x10541;
bool x10553 = x10466 <= 1;
int32_t x10554;
if (x10553) {
x10554 = 1;
} else {
x10554 = x10466;
}
int32_t x10560 = x10554 * x10554;
int32_t x10565;
if (x10541) {
x10565 = 0;
} else {
x10565 = x10466;
}
int32_t x10566;
if (x10541) {
x10566 = 0;
} else {
x10566 = 1;
}
float* x242 = x5+1231424;
int32_t x10613 = x10554 + 2;
int32_t x10614 = x10613 - 3;
int32_t x10615 = x10614 / 1;
int32_t x10616 = x10615 + 1;
int32_t x10620 = 8192 * x10616;
int32_t x10621 = x10620 * x10616;
int32_t x10617 = x10616 * x10616;
int32_t x10618 = 128 * x10617;
float* x165 = x5+1231808;
bool x10743 = x10616 == 1;
bool x10744 = x10743 || true;
bool x10745 = x10744 || x10743;
bool x10755 = x10616 <= 1;
int32_t x10756;
if (x10755) {
x10756 = 1;
} else {
x10756 = x10616;
}
int32_t x10762 = x10756 * x10756;
int32_t x10766;
if (x10743) {
x10766 = 0;
} else {
x10766 = x10616;
}
int32_t x10767;
if (x10743) {
x10767 = 0;
} else {
x10767 = 1;
}
float* x268 = x5+1379520;
float* x148 = x5+1379648;
bool x10846 = x10756 == 1;
bool x10847 = x10846 || true;
bool x10848 = x10847 || x10846;
bool x10858 = x10756 <= 1;
int32_t x10859;
if (x10858) {
x10859 = 1;
} else {
x10859 = x10756;
}
int32_t x10865 = x10859 * x10859;
int32_t x10870;
if (x10846) {
x10870 = 0;
} else {
x10870 = x10756;
}
int32_t x10871;
if (x10846) {
x10871 = 0;
} else {
x10871 = 1;
}
bool x10934 = x10859 == 1;
bool x10935 = x10934 || true;
bool x10936 = x10935 || x10934;
bool x10946 = x10859 <= 1;
int32_t x10947;
if (x10946) {
x10947 = 1;
} else {
x10947 = x10859;
}
int32_t x10953 = x10947 * x10947;
int32_t x10958;
if (x10934) {
x10958 = 0;
} else {
x10958 = x10859;
}
int32_t x10959;
if (x10934) {
x10959 = 0;
} else {
x10959 = 1;
}
float* x79 = x5+1379264;
bool x11022 = x10947 == 1;
bool x11023 = x11022 || true;
bool x11024 = x11023 || x11022;
bool x11034 = x10947 <= 1;
int32_t x11035;
if (x11034) {
x11035 = 1;
} else {
x11035 = x10947;
}
int32_t x11041 = x11035 * x11035;
int32_t x11046;
if (x11022) {
x11046 = 0;
} else {
x11046 = x10947;
}
int32_t x11047;
if (x11022) {
x11047 = 0;
} else {
x11047 = 1;
}
float* x38 = x5+1379392;
int32_t x11094 = x11035 - 1;
int32_t x11095 = x11094 / 1;
int32_t x11096 = x11095 + 1;
int32_t x11100 = 32768 * x11096;
int32_t x11101 = x11100 * x11096;
int32_t x11097 = x11096 * x11096;
int32_t x11098 = 512 * x11097;
float* x55 = x5+1379776;
bool x11175 = x11096 == 1;
bool x11176 = x11175 || true;
bool x11177 = x11176 || x11175;
bool x11187 = x11096 <= 1;
int32_t x11188;
if (x11187) {
x11188 = 1;
} else {
x11188 = x11096;
}
int32_t x11194 = x11188 * x11188;
int32_t x11198;
if (x11175) {
x11198 = 0;
} else {
x11198 = x11096;
}
int32_t x11199;
if (x11175) {
x11199 = 0;
} else {
x11199 = 1;
}
float* x19 = x5+1446336;
float* x234 = x5+1446848;
bool x11278 = x11188 == 1;
bool x11279 = x11278 || true;
bool x11280 = x11279 || x11278;
bool x11290 = x11188 <= 1;
int32_t x11291;
if (x11290) {
x11291 = 1;
} else {
x11291 = x11188;
}
int32_t x11297 = x11291 * x11291;
int32_t x11302;
if (x11278) {
x11302 = 0;
} else {
x11302 = x11188;
}
int32_t x11303;
if (x11278) {
x11303 = 0;
} else {
x11303 = 1;
}
bool x11366 = x11291 == 1;
bool x11367 = x11366 || true;
bool x11368 = x11367 || x11366;
bool x11378 = x11291 <= 1;
int32_t x11379;
if (x11378) {
x11379 = 1;
} else {
x11379 = x11291;
}
int32_t x11385 = x11379 * x11379;
int32_t x11390;
if (x11366) {
x11390 = 0;
} else {
x11390 = x11291;
}
int32_t x11391;
if (x11366) {
x11391 = 0;
} else {
x11391 = 1;
}
float* x156 = x5+1445312;
bool x11454 = x11379 == 1;
bool x11455 = x11454 || true;
bool x11456 = x11455 || x11454;
bool x11466 = x11379 <= 1;
int32_t x11467;
if (x11466) {
x11467 = 1;
} else {
x11467 = x11379;
}
int32_t x11473 = x11467 * x11467;
int32_t x11478;
if (x11454) {
x11478 = 0;
} else {
x11478 = x11379;
}
int32_t x11479;
if (x11454) {
x11479 = 0;
} else {
x11479 = 1;
}
float* x54 = x5+1445824;
bool x11517 = x11467 == 1;
bool x11518 = x11517 || x10105;
bool x11519 = x11467 == x10055;
bool x11520 = x11518 || x11519;
bool x11530 = x11467 <= x10055;
int32_t x11531;
if (x11530) {
x11531 = x10055;
} else {
x11531 = x11467;
}
int32_t x11546;
if (x11517) {
x11546 = 0;
} else {
x11546 = x11467;
}
int32_t x11547;
if (x11517) {
x11547 = 0;
} else {
x11547 = 1;
}
int32_t x11593 = x11467 - 1;
int32_t x11594 = x11593 / 1;
int32_t x11595 = x11594 + 1;
int32_t x11599 = 16384 * x11595;
int32_t x11600 = x11599 * x11595;
int32_t x11596 = x11595 * x11595;
int32_t x11597 = 256 * x11596;
float* x180 = x5+1447360;
bool x11674 = x11595 == 1;
bool x11675 = x11674 || true;
bool x11676 = x11675 || x11674;
bool x11686 = x11595 <= 1;
int32_t x11687;
if (x11686) {
x11687 = 1;
} else {
x11687 = x11595;
}
int32_t x11693 = x11687 * x11687;
int32_t x11697;
if (x11674) {
x11697 = 0;
} else {
x11697 = x11595;
}
int32_t x11698;
if (x11674) {
x11698 = 0;
} else {
x11698 = 1;
}
float* x131 = x5+1578944;
float* x198 = x5+1579200;
bool x11777 = x11687 == 1;
bool x11778 = x11777 || true;
bool x11779 = x11778 || x11777;
bool x11789 = x11687 <= 1;
int32_t x11790;
if (x11789) {
x11790 = 1;
} else {
x11790 = x11687;
}
int32_t x11796 = x11790 * x11790;
int32_t x11801;
if (x11777) {
x11801 = 0;
} else {
x11801 = x11687;
}
int32_t x11802;
if (x11777) {
x11802 = 0;
} else {
x11802 = 1;
}
bool x11865 = x11790 == 1;
bool x11866 = x11865 || true;
bool x11867 = x11866 || x11865;
bool x11877 = x11790 <= 1;
int32_t x11878;
if (x11877) {
x11878 = 1;
} else {
x11878 = x11790;
}
int32_t x11884 = x11878 * x11878;
int32_t x11889;
if (x11865) {
x11889 = 0;
} else {
x11889 = x11790;
}
int32_t x11890;
if (x11865) {
x11890 = 0;
} else {
x11890 = 1;
}
float* x270 = x5+1578432;
bool x11953 = x11878 == 1;
bool x11954 = x11953 || true;
bool x11955 = x11954 || x11953;
bool x11965 = x11878 <= 1;
int32_t x11966;
if (x11965) {
x11966 = 1;
} else {
x11966 = x11878;
}
int32_t x11972 = x11966 * x11966;
int32_t x11977;
if (x11953) {
x11977 = 0;
} else {
x11977 = x11878;
}
int32_t x11978;
if (x11953) {
x11978 = 0;
} else {
x11978 = 1;
}
float* x21 = x5+1578688;
int32_t x12025 = x11966 + 2;
int32_t x12026 = x12025 - 3;
int32_t x12027 = x12026 / 2;
int32_t x12028 = x12027 + 1;
int32_t x12032 = 16384 * x12028;
int32_t x12033 = x12032 * x12028;
int32_t x12029 = x12028 * x12028;
int32_t x12030 = 256 * x12029;
float* x175 = x5+1579456;
bool x12139 = x12028 == 1;
bool x12140 = x12139 || true;
bool x12141 = x12140 || x12139;
bool x12151 = x12028 <= 1;
int32_t x12152;
if (x12151) {
x12152 = 1;
} else {
x12152 = x12028;
}
int32_t x12158 = x12152 * x12152;
int32_t x12162;
if (x12139) {
x12162 = 0;
} else {
x12162 = x12028;
}
int32_t x12163;
if (x12139) {
x12163 = 0;
} else {
x12163 = 1;
}
float* x229 = x5+2169792;
float* x99 = x5+2170048;
bool x12242 = x12152 == 1;
bool x12243 = x12242 || true;
bool x12244 = x12243 || x12242;
bool x12254 = x12152 <= 1;
int32_t x12255;
if (x12254) {
x12255 = 1;
} else {
x12255 = x12152;
}
int32_t x12261 = x12255 * x12255;
int32_t x12266;
if (x12242) {
x12266 = 0;
} else {
x12266 = x12152;
}
int32_t x12267;
if (x12242) {
x12267 = 0;
} else {
x12267 = 1;
}
bool x12330 = x12255 == 1;
bool x12331 = x12330 || true;
bool x12332 = x12331 || x12330;
bool x12342 = x12255 <= 1;
int32_t x12343;
if (x12342) {
x12343 = 1;
} else {
x12343 = x12255;
}
int32_t x12349 = x12343 * x12343;
int32_t x12354;
if (x12330) {
x12354 = 0;
} else {
x12354 = x12255;
}
int32_t x12355;
if (x12330) {
x12355 = 0;
} else {
x12355 = 1;
}
float* x108 = x5+2169280;
bool x12418 = x12343 == 1;
bool x12419 = x12418 || true;
bool x12420 = x12419 || x12418;
bool x12430 = x12343 <= 1;
int32_t x12431;
if (x12430) {
x12431 = 1;
} else {
x12431 = x12343;
}
int32_t x12437 = x12431 * x12431;
int32_t x12442;
if (x12418) {
x12442 = 0;
} else {
x12442 = x12343;
}
int32_t x12443;
if (x12418) {
x12443 = 0;
} else {
x12443 = 1;
}
float* x16 = x5+2169536;
int32_t x12490 = x12431 - 1;
int32_t x12491 = x12490 / 1;
int32_t x12492 = x12491 + 1;
int32_t x12496 = 65536 * x12492;
int32_t x12497 = x12496 * x12492;
int32_t x12493 = x12492 * x12492;
int32_t x12494 = 1024 * x12493;
float* x269 = x5+2170304;
bool x12571 = x12492 == 1;
bool x12572 = x12571 || true;
bool x12573 = x12572 || x12571;
bool x12583 = x12492 <= 1;
int32_t x12584;
if (x12583) {
x12584 = 1;
} else {
x12584 = x12492;
}
int32_t x12590 = x12584 * x12584;
int32_t x12594;
if (x12571) {
x12594 = 0;
} else {
x12594 = x12492;
}
int32_t x12595;
if (x12571) {
x12595 = 0;
} else {
x12595 = 1;
}
float* x216 = x5+2434496;
float* x267 = x5+2435520;
bool x12675 = x12584 == 1;
bool x12676 = x12675 || true;
bool x12677 = x12676 || x12675;
bool x12687 = x12584 <= 1;
int32_t x12688;
if (x12687) {
x12688 = 1;
} else {
x12688 = x12584;
}
int32_t x12694 = x12688 * x12688;
int32_t x12699;
if (x12675) {
x12699 = 0;
} else {
x12699 = x12584;
}
int32_t x12700;
if (x12675) {
x12700 = 0;
} else {
x12700 = 1;
}
bool x12763 = x12688 == 1;
bool x12764 = x12763 || true;
bool x12765 = x12764 || x12763;
bool x12775 = x12688 <= 1;
int32_t x12776;
if (x12775) {
x12776 = 1;
} else {
x12776 = x12688;
}
int32_t x12782 = x12776 * x12776;
int32_t x12787;
if (x12763) {
x12787 = 0;
} else {
x12787 = x12688;
}
int32_t x12788;
if (x12763) {
x12788 = 0;
} else {
x12788 = 1;
}
float* x18 = x5+2432448;
bool x12851 = x12776 == 1;
bool x12852 = x12851 || true;
bool x12853 = x12852 || x12851;
bool x12863 = x12776 <= 1;
int32_t x12864;
if (x12863) {
x12864 = 1;
} else {
x12864 = x12776;
}
int32_t x12870 = x12864 * x12864;
int32_t x12875;
if (x12851) {
x12875 = 0;
} else {
x12875 = x12776;
}
int32_t x12876;
if (x12851) {
x12876 = 0;
} else {
x12876 = 1;
}
float* x117 = x5+2433472;
int32_t x12910 = x11593 / 2;
int32_t x12911 = x12910 + 1;
int32_t x12915 = 65536 * x12911;
int32_t x12916 = x12915 * x12911;
int32_t x12912 = x12911 * x12911;
int32_t x12913 = 1024 * x12912;
float* x75 = x5+2436544;
bool x12996 = x12911 == 1;
bool x12997 = x12996 || true;
bool x12998 = x12997 || x12996;
bool x13008 = x12911 <= 1;
int32_t x13009;
if (x13008) {
x13009 = 1;
} else {
x13009 = x12911;
}
int32_t x13015 = x13009 * x13009;
int32_t x13019;
if (x12996) {
x13019 = 0;
} else {
x13019 = x12911;
}
int32_t x13020;
if (x12996) {
x13020 = 0;
} else {
x13020 = 1;
}
float* x86 = x5+2962880;
float* x211 = x5+2963904;
bool x13099 = x13009 == 1;
bool x13100 = x13099 || true;
bool x13101 = x13100 || x13099;
bool x13111 = x13009 <= 1;
int32_t x13112;
if (x13111) {
x13112 = 1;
} else {
x13112 = x13009;
}
int32_t x13118 = x13112 * x13112;
int32_t x13123;
if (x13099) {
x13123 = 0;
} else {
x13123 = x13009;
}
int32_t x13124;
if (x13099) {
x13124 = 0;
} else {
x13124 = 1;
}
bool x13187 = x13112 == 1;
bool x13188 = x13187 || true;
bool x13189 = x13188 || x13187;
bool x13199 = x13112 <= 1;
int32_t x13200;
if (x13199) {
x13200 = 1;
} else {
x13200 = x13112;
}
int32_t x13206 = x13200 * x13200;
int32_t x13211;
if (x13187) {
x13211 = 0;
} else {
x13211 = x13112;
}
int32_t x13212;
if (x13187) {
x13212 = 0;
} else {
x13212 = 1;
}
float* x29 = x5+2960832;
bool x13275 = x13200 == 1;
bool x13276 = x13275 || true;
bool x13277 = x13276 || x13275;
bool x13287 = x13200 <= 1;
int32_t x13288;
if (x13287) {
x13288 = 1;
} else {
x13288 = x13200;
}
int32_t x13294 = x13288 * x13288;
int32_t x13299;
if (x13275) {
x13299 = 0;
} else {
x13299 = x13200;
}
int32_t x13300;
if (x13275) {
x13300 = 0;
} else {
x13300 = 1;
}
float* x220 = x5+2961856;
bool x13339 = x12864 == 1;
bool x13340 = x13288 == 1;
bool x13341 = x13339 || x13340;
bool x13342 = x12864 == x13288;
bool x13343 = x13341 || x13342;
bool x13353 = x12864 <= x13288;
int32_t x13354;
if (x13353) {
x13354 = x13288;
} else {
x13354 = x12864;
}
int32_t x13369;
if (x13339) {
x13369 = 0;
} else {
x13369 = x12864;
}
int32_t x13370;
if (x13339) {
x13370 = 0;
} else {
x13370 = 1;
}
int32_t x13372;
if (x13340) {
x13372 = 0;
} else {
x13372 = x13288;
}
int32_t x13373;
if (x13340) {
x13373 = 0;
} else {
x13373 = 1;
}
int32_t x13419 = x12864 - 1;
int32_t x13420 = x13419 / 1;
int32_t x13421 = x13420 + 1;
int32_t x13425 = 16384 * x13421;
int32_t x13426 = x13425 * x13421;
int32_t x13422 = x13421 * x13421;
int32_t x13423 = 256 * x13422;
float* x13 = x5+2964928;
bool x13500 = x13421 == 1;
bool x13501 = x13500 || true;
bool x13502 = x13501 || x13500;
bool x13512 = x13421 <= 1;
int32_t x13513;
if (x13512) {
x13513 = 1;
} else {
x13513 = x13421;
}
int32_t x13519 = x13513 * x13513;
int32_t x13523;
if (x13500) {
x13523 = 0;
} else {
x13523 = x13421;
}
int32_t x13524;
if (x13500) {
x13524 = 0;
} else {
x13524 = 1;
}
float* x259 = x5+3227584;
float* x157 = x5+3227840;
bool x13603 = x13513 == 1;
bool x13604 = x13603 || true;
bool x13605 = x13604 || x13603;
bool x13615 = x13513 <= 1;
int32_t x13616;
if (x13615) {
x13616 = 1;
} else {
x13616 = x13513;
}
int32_t x13622 = x13616 * x13616;
int32_t x13627;
if (x13603) {
x13627 = 0;
} else {
x13627 = x13513;
}
int32_t x13628;
if (x13603) {
x13628 = 0;
} else {
x13628 = 1;
}
bool x13691 = x13616 == 1;
bool x13692 = x13691 || true;
bool x13693 = x13692 || x13691;
bool x13703 = x13616 <= 1;
int32_t x13704;
if (x13703) {
x13704 = 1;
} else {
x13704 = x13616;
}
int32_t x13710 = x13704 * x13704;
int32_t x13715;
if (x13691) {
x13715 = 0;
} else {
x13715 = x13616;
}
int32_t x13716;
if (x13691) {
x13716 = 0;
} else {
x13716 = 1;
}
float* x30 = x5+3227072;
bool x13779 = x13704 == 1;
bool x13780 = x13779 || true;
bool x13781 = x13780 || x13779;
bool x13791 = x13704 <= 1;
int32_t x13792;
if (x13791) {
x13792 = 1;
} else {
x13792 = x13704;
}
int32_t x13798 = x13792 * x13792;
int32_t x13803;
if (x13779) {
x13803 = 0;
} else {
x13803 = x13704;
}
int32_t x13804;
if (x13779) {
x13804 = 0;
} else {
x13804 = 1;
}
float* x219 = x5+3227328;
int32_t x13851 = x13792 + 2;
int32_t x13852 = x13851 - 3;
int32_t x13853 = x13852 / 1;
int32_t x13854 = x13853 + 1;
int32_t x13858 = 16384 * x13854;
int32_t x13859 = x13858 * x13854;
int32_t x13855 = x13854 * x13854;
int32_t x13856 = 256 * x13855;
float* x31 = x5+3228096;
bool x13981 = x13854 == 1;
bool x13982 = x13981 || true;
bool x13983 = x13982 || x13981;
bool x13993 = x13854 <= 1;
int32_t x13994;
if (x13993) {
x13994 = 1;
} else {
x13994 = x13854;
}
int32_t x14000 = x13994 * x13994;
int32_t x14004;
if (x13981) {
x14004 = 0;
} else {
x14004 = x13854;
}
int32_t x14005;
if (x13981) {
x14005 = 0;
} else {
x14005 = 1;
}
float* x200 = x5+3818432;
float* x237 = x5+3818688;
bool x14084 = x13994 == 1;
bool x14085 = x14084 || true;
bool x14086 = x14085 || x14084;
bool x14096 = x13994 <= 1;
int32_t x14097;
if (x14096) {
x14097 = 1;
} else {
x14097 = x13994;
}
int32_t x14103 = x14097 * x14097;
int32_t x14108;
if (x14084) {
x14108 = 0;
} else {
x14108 = x13994;
}
int32_t x14109;
if (x14084) {
x14109 = 0;
} else {
x14109 = 1;
}
bool x14172 = x14097 == 1;
bool x14173 = x14172 || true;
bool x14174 = x14173 || x14172;
bool x14184 = x14097 <= 1;
int32_t x14185;
if (x14184) {
x14185 = 1;
} else {
x14185 = x14097;
}
int32_t x14191 = x14185 * x14185;
int32_t x14196;
if (x14172) {
x14196 = 0;
} else {
x14196 = x14097;
}
int32_t x14197;
if (x14172) {
x14197 = 0;
} else {
x14197 = 1;
}
float* x271 = x5+3817920;
bool x14260 = x14185 == 1;
bool x14261 = x14260 || true;
bool x14262 = x14261 || x14260;
bool x14272 = x14185 <= 1;
int32_t x14273;
if (x14272) {
x14273 = 1;
} else {
x14273 = x14185;
}
int32_t x14279 = x14273 * x14273;
int32_t x14284;
if (x14260) {
x14284 = 0;
} else {
x14284 = x14185;
}
int32_t x14285;
if (x14260) {
x14285 = 0;
} else {
x14285 = 1;
}
float* x96 = x5+3818176;
int32_t x14332 = x14273 - 1;
int32_t x14333 = x14332 / 1;
int32_t x14334 = x14333 + 1;
int32_t x14338 = 65536 * x14334;
int32_t x14339 = x14338 * x14334;
int32_t x14335 = x14334 * x14334;
int32_t x14336 = 1024 * x14335;
float* x56 = x5+3818944;
bool x14413 = x14334 == 1;
bool x14414 = x14413 || true;
bool x14415 = x14414 || x14413;
bool x14425 = x14334 <= 1;
int32_t x14426;
if (x14425) {
x14426 = 1;
} else {
x14426 = x14334;
}
int32_t x14432 = x14426 * x14426;
int32_t x14436;
if (x14413) {
x14436 = 0;
} else {
x14436 = x14334;
}
int32_t x14437;
if (x14413) {
x14437 = 0;
} else {
x14437 = 1;
}
float* x182 = x5+4083136;
float* x143 = x5+4084160;
bool x14516 = x14426 == 1;
bool x14517 = x14516 || true;
bool x14518 = x14517 || x14516;
bool x14528 = x14426 <= 1;
int32_t x14529;
if (x14528) {
x14529 = 1;
} else {
x14529 = x14426;
}
int32_t x14535 = x14529 * x14529;
int32_t x14540;
if (x14516) {
x14540 = 0;
} else {
x14540 = x14426;
}
int32_t x14541;
if (x14516) {
x14541 = 0;
} else {
x14541 = 1;
}
bool x14604 = x14529 == 1;
bool x14605 = x14604 || true;
bool x14606 = x14605 || x14604;
bool x14616 = x14529 <= 1;
int32_t x14617;
if (x14616) {
x14617 = 1;
} else {
x14617 = x14529;
}
int32_t x14623 = x14617 * x14617;
int32_t x14628;
if (x14604) {
x14628 = 0;
} else {
x14628 = x14529;
}
int32_t x14629;
if (x14604) {
x14629 = 0;
} else {
x14629 = 1;
}
float* x20 = x5+4081088;
bool x14692 = x14617 == 1;
bool x14693 = x14692 || true;
bool x14694 = x14693 || x14692;
bool x14704 = x14617 <= 1;
int32_t x14705;
if (x14704) {
x14705 = 1;
} else {
x14705 = x14617;
}
int32_t x14711 = x14705 * x14705;
int32_t x14716;
if (x14692) {
x14716 = 0;
} else {
x14716 = x14617;
}
int32_t x14717;
if (x14692) {
x14717 = 0;
} else {
x14717 = 1;
}
float* x232 = x5+4082112;
bool x14755 = x14705 == 1;
bool x14756 = x14755 || x13339;
bool x14757 = x14705 == x12864;
bool x14758 = x14756 || x14757;
bool x14768 = x14705 <= x12864;
int32_t x14769;
if (x14768) {
x14769 = x12864;
} else {
x14769 = x14705;
}
int32_t x14784;
if (x14755) {
x14784 = 0;
} else {
x14784 = x14705;
}
int32_t x14785;
if (x14755) {
x14785 = 0;
} else {
x14785 = 1;
}
int32_t x14831 = x14705 - 1;
int32_t x14832 = x14831 / 1;
int32_t x14833 = x14832 + 1;
int32_t x14837 = 16384 * x14833;
int32_t x14838 = x14837 * x14833;
int32_t x14834 = x14833 * x14833;
int32_t x14835 = 256 * x14834;
float* x218 = x5+4085184;
bool x14912 = x14833 == 1;
bool x14913 = x14912 || true;
bool x14914 = x14913 || x14912;
bool x14924 = x14833 <= 1;
int32_t x14925;
if (x14924) {
x14925 = 1;
} else {
x14925 = x14833;
}
int32_t x14931 = x14925 * x14925;
int32_t x14935;
if (x14912) {
x14935 = 0;
} else {
x14935 = x14833;
}
int32_t x14936;
if (x14912) {
x14936 = 0;
} else {
x14936 = 1;
}
float* x178 = x5+4347840;
float* x174 = x5+4348096;
bool x15015 = x14925 == 1;
bool x15016 = x15015 || true;
bool x15017 = x15016 || x15015;
bool x15027 = x14925 <= 1;
int32_t x15028;
if (x15027) {
x15028 = 1;
} else {
x15028 = x14925;
}
int32_t x15034 = x15028 * x15028;
int32_t x15039;
if (x15015) {
x15039 = 0;
} else {
x15039 = x14925;
}
int32_t x15040;
if (x15015) {
x15040 = 0;
} else {
x15040 = 1;
}
bool x15103 = x15028 == 1;
bool x15104 = x15103 || true;
bool x15105 = x15104 || x15103;
bool x15115 = x15028 <= 1;
int32_t x15116;
if (x15115) {
x15116 = 1;
} else {
x15116 = x15028;
}
int32_t x15122 = x15116 * x15116;
int32_t x15127;
if (x15103) {
x15127 = 0;
} else {
x15127 = x15028;
}
int32_t x15128;
if (x15103) {
x15128 = 0;
} else {
x15128 = 1;
}
float* x129 = x5+4347328;
bool x15191 = x15116 == 1;
bool x15192 = x15191 || true;
bool x15193 = x15192 || x15191;
bool x15203 = x15116 <= 1;
int32_t x15204;
if (x15203) {
x15204 = 1;
} else {
x15204 = x15116;
}
int32_t x15210 = x15204 * x15204;
int32_t x15215;
if (x15191) {
x15215 = 0;
} else {
x15215 = x15116;
}
int32_t x15216;
if (x15191) {
x15216 = 0;
} else {
x15216 = 1;
}
float* x197 = x5+4347584;
int32_t x15263 = x15204 + 2;
int32_t x15264 = x15263 - 3;
int32_t x15265 = x15264 / 1;
int32_t x15266 = x15265 + 1;
int32_t x15270 = 16384 * x15266;
int32_t x15271 = x15270 * x15266;
int32_t x15267 = x15266 * x15266;
int32_t x15268 = 256 * x15267;
float* x14 = x5+4348352;
bool x15393 = x15266 == 1;
bool x15394 = x15393 || true;
bool x15395 = x15394 || x15393;
bool x15405 = x15266 <= 1;
int32_t x15406;
if (x15405) {
x15406 = 1;
} else {
x15406 = x15266;
}
int32_t x15412 = x15406 * x15406;
int32_t x15416;
if (x15393) {
x15416 = 0;
} else {
x15416 = x15266;
}
int32_t x15417;
if (x15393) {
x15417 = 0;
} else {
x15417 = 1;
}
float* x124 = x5+4938688;
float* x63 = x5+4938944;
bool x15496 = x15406 == 1;
bool x15497 = x15496 || true;
bool x15498 = x15497 || x15496;
bool x15508 = x15406 <= 1;
int32_t x15509;
if (x15508) {
x15509 = 1;
} else {
x15509 = x15406;
}
int32_t x15515 = x15509 * x15509;
int32_t x15520;
if (x15496) {
x15520 = 0;
} else {
x15520 = x15406;
}
int32_t x15521;
if (x15496) {
x15521 = 0;
} else {
x15521 = 1;
}
bool x15584 = x15509 == 1;
bool x15585 = x15584 || true;
bool x15586 = x15585 || x15584;
bool x15596 = x15509 <= 1;
int32_t x15597;
if (x15596) {
x15597 = 1;
} else {
x15597 = x15509;
}
int32_t x15603 = x15597 * x15597;
int32_t x15608;
if (x15584) {
x15608 = 0;
} else {
x15608 = x15509;
}
int32_t x15609;
if (x15584) {
x15609 = 0;
} else {
x15609 = 1;
}
float* x228 = x5+4938176;
bool x15672 = x15597 == 1;
bool x15673 = x15672 || true;
bool x15674 = x15673 || x15672;
bool x15684 = x15597 <= 1;
int32_t x15685;
if (x15684) {
x15685 = 1;
} else {
x15685 = x15597;
}
int32_t x15691 = x15685 * x15685;
int32_t x15696;
if (x15672) {
x15696 = 0;
} else {
x15696 = x15597;
}
int32_t x15697;
if (x15672) {
x15697 = 0;
} else {
x15697 = 1;
}
float* x192 = x5+4938432;
int32_t x15744 = x15685 - 1;
int32_t x15745 = x15744 / 1;
int32_t x15746 = x15745 + 1;
int32_t x15750 = 65536 * x15746;
int32_t x15751 = x15750 * x15746;
int32_t x15747 = x15746 * x15746;
int32_t x15748 = 1024 * x15747;
float* x116 = x5+4939200;
bool x15825 = x15746 == 1;
bool x15826 = x15825 || true;
bool x15827 = x15826 || x15825;
bool x15837 = x15746 <= 1;
int32_t x15838;
if (x15837) {
x15838 = 1;
} else {
x15838 = x15746;
}
int32_t x15844 = x15838 * x15838;
int32_t x15848;
if (x15825) {
x15848 = 0;
} else {
x15848 = x15746;
}
int32_t x15849;
if (x15825) {
x15849 = 0;
} else {
x15849 = 1;
}
float* x140 = x5+5203392;
float* x188 = x5+5204416;
bool x15928 = x15838 == 1;
bool x15929 = x15928 || true;
bool x15930 = x15929 || x15928;
bool x15940 = x15838 <= 1;
int32_t x15941;
if (x15940) {
x15941 = 1;
} else {
x15941 = x15838;
}
int32_t x15947 = x15941 * x15941;
int32_t x15952;
if (x15928) {
x15952 = 0;
} else {
x15952 = x15838;
}
int32_t x15953;
if (x15928) {
x15953 = 0;
} else {
x15953 = 1;
}
bool x16016 = x15941 == 1;
bool x16017 = x16016 || true;
bool x16018 = x16017 || x16016;
bool x16028 = x15941 <= 1;
int32_t x16029;
if (x16028) {
x16029 = 1;
} else {
x16029 = x15941;
}
int32_t x16035 = x16029 * x16029;
int32_t x16040;
if (x16016) {
x16040 = 0;
} else {
x16040 = x15941;
}
int32_t x16041;
if (x16016) {
x16041 = 0;
} else {
x16041 = 1;
}
float* x263 = x5+5201344;
bool x16104 = x16029 == 1;
bool x16105 = x16104 || true;
bool x16106 = x16105 || x16104;
bool x16116 = x16029 <= 1;
int32_t x16117;
if (x16116) {
x16117 = 1;
} else {
x16117 = x16029;
}
int32_t x16123 = x16117 * x16117;
int32_t x16128;
if (x16104) {
x16128 = 0;
} else {
x16128 = x16029;
}
int32_t x16129;
if (x16104) {
x16129 = 0;
} else {
x16129 = 1;
}
float* x57 = x5+5202368;
bool x16167 = x16117 == 1;
bool x16168 = x16167 || x14755;
bool x16169 = x16117 == x14705;
bool x16170 = x16168 || x16169;
bool x16180 = x16117 <= x14705;
int32_t x16181;
if (x16180) {
x16181 = x14705;
} else {
x16181 = x16117;
}
int32_t x16196;
if (x16167) {
x16196 = 0;
} else {
x16196 = x16117;
}
int32_t x16197;
if (x16167) {
x16197 = 0;
} else {
x16197 = 1;
}
int32_t x16243 = x16117 - 1;
int32_t x16244 = x16243 / 1;
int32_t x16245 = x16244 + 1;
int32_t x16249 = 16384 * x16245;
int32_t x16250 = x16249 * x16245;
int32_t x16246 = x16245 * x16245;
int32_t x16247 = 256 * x16246;
float* x6 = x5+5205440;
bool x16324 = x16245 == 1;
bool x16325 = x16324 || true;
bool x16326 = x16325 || x16324;
bool x16336 = x16245 <= 1;
int32_t x16337;
if (x16336) {
x16337 = 1;
} else {
x16337 = x16245;
}
int32_t x16343 = x16337 * x16337;
int32_t x16347;
if (x16324) {
x16347 = 0;
} else {
x16347 = x16245;
}
int32_t x16348;
if (x16324) {
x16348 = 0;
} else {
x16348 = 1;
}
float* x163 = x5+5468096;
float* x98 = x5+5468352;
bool x16427 = x16337 == 1;
bool x16428 = x16427 || true;
bool x16429 = x16428 || x16427;
bool x16439 = x16337 <= 1;
int32_t x16440;
if (x16439) {
x16440 = 1;
} else {
x16440 = x16337;
}
int32_t x16446 = x16440 * x16440;
int32_t x16451;
if (x16427) {
x16451 = 0;
} else {
x16451 = x16337;
}
int32_t x16452;
if (x16427) {
x16452 = 0;
} else {
x16452 = 1;
}
bool x16515 = x16440 == 1;
bool x16516 = x16515 || true;
bool x16517 = x16516 || x16515;
bool x16527 = x16440 <= 1;
int32_t x16528;
if (x16527) {
x16528 = 1;
} else {
x16528 = x16440;
}
int32_t x16534 = x16528 * x16528;
int32_t x16539;
if (x16515) {
x16539 = 0;
} else {
x16539 = x16440;
}
int32_t x16540;
if (x16515) {
x16540 = 0;
} else {
x16540 = 1;
}
float* x92 = x5+5467584;
bool x16603 = x16528 == 1;
bool x16604 = x16603 || true;
bool x16605 = x16604 || x16603;
bool x16615 = x16528 <= 1;
int32_t x16616;
if (x16615) {
x16616 = 1;
} else {
x16616 = x16528;
}
int32_t x16622 = x16616 * x16616;
int32_t x16627;
if (x16603) {
x16627 = 0;
} else {
x16627 = x16528;
}
int32_t x16628;
if (x16603) {
x16628 = 0;
} else {
x16628 = 1;
}
float* x241 = x5+5467840;
int32_t x16675 = x16616 + 2;
int32_t x16676 = x16675 - 3;
int32_t x16677 = x16676 / 1;
int32_t x16678 = x16677 + 1;
int32_t x16682 = 16384 * x16678;
int32_t x16683 = x16682 * x16678;
int32_t x16679 = x16678 * x16678;
int32_t x16680 = 256 * x16679;
float* x249 = x5+5468608;
bool x16805 = x16678 == 1;
bool x16806 = x16805 || true;
bool x16807 = x16806 || x16805;
bool x16817 = x16678 <= 1;
int32_t x16818;
if (x16817) {
x16818 = 1;
} else {
x16818 = x16678;
}
int32_t x16824 = x16818 * x16818;
int32_t x16828;
if (x16805) {
x16828 = 0;
} else {
x16828 = x16678;
}
int32_t x16829;
if (x16805) {
x16829 = 0;
} else {
x16829 = 1;
}
float* x186 = x5+6058944;
float* x230 = x5+6059200;
bool x16908 = x16818 == 1;
bool x16909 = x16908 || true;
bool x16910 = x16909 || x16908;
bool x16920 = x16818 <= 1;
int32_t x16921;
if (x16920) {
x16921 = 1;
} else {
x16921 = x16818;
}
int32_t x16927 = x16921 * x16921;
int32_t x16932;
if (x16908) {
x16932 = 0;
} else {
x16932 = x16818;
}
int32_t x16933;
if (x16908) {
x16933 = 0;
} else {
x16933 = 1;
}
bool x16996 = x16921 == 1;
bool x16997 = x16996 || true;
bool x16998 = x16997 || x16996;
bool x17008 = x16921 <= 1;
int32_t x17009;
if (x17008) {
x17009 = 1;
} else {
x17009 = x16921;
}
int32_t x17015 = x17009 * x17009;
int32_t x17020;
if (x16996) {
x17020 = 0;
} else {
x17020 = x16921;
}
int32_t x17021;
if (x16996) {
x17021 = 0;
} else {
x17021 = 1;
}
float* x74 = x5+6058432;
bool x17084 = x17009 == 1;
bool x17085 = x17084 || true;
bool x17086 = x17085 || x17084;
bool x17096 = x17009 <= 1;
int32_t x17097;
if (x17096) {
x17097 = 1;
} else {
x17097 = x17009;
}
int32_t x17103 = x17097 * x17097;
int32_t x17108;
if (x17084) {
x17108 = 0;
} else {
x17108 = x17009;
}
int32_t x17109;
if (x17084) {
x17109 = 0;
} else {
x17109 = 1;
}
float* x136 = x5+6058688;
int32_t x17156 = x17097 - 1;
int32_t x17157 = x17156 / 1;
int32_t x17158 = x17157 + 1;
int32_t x17162 = 65536 * x17158;
int32_t x17163 = x17162 * x17158;
int32_t x17159 = x17158 * x17158;
int32_t x17160 = 1024 * x17159;
float* x89 = x5+6059456;
bool x17237 = x17158 == 1;
bool x17238 = x17237 || true;
bool x17239 = x17238 || x17237;
bool x17249 = x17158 <= 1;
int32_t x17250;
if (x17249) {
x17250 = 1;
} else {
x17250 = x17158;
}
int32_t x17256 = x17250 * x17250;
int32_t x17260;
if (x17237) {
x17260 = 0;
} else {
x17260 = x17158;
}
int32_t x17261;
if (x17237) {
x17261 = 0;
} else {
x17261 = 1;
}
float* x231 = x5+6323648;
float* x161 = x5+6324672;
bool x17340 = x17250 == 1;
bool x17341 = x17340 || true;
bool x17342 = x17341 || x17340;
bool x17352 = x17250 <= 1;
int32_t x17353;
if (x17352) {
x17353 = 1;
} else {
x17353 = x17250;
}
int32_t x17359 = x17353 * x17353;
int32_t x17364;
if (x17340) {
x17364 = 0;
} else {
x17364 = x17250;
}
int32_t x17365;
if (x17340) {
x17365 = 0;
} else {
x17365 = 1;
}
bool x17428 = x17353 == 1;
bool x17429 = x17428 || true;
bool x17430 = x17429 || x17428;
bool x17440 = x17353 <= 1;
int32_t x17441;
if (x17440) {
x17441 = 1;
} else {
x17441 = x17353;
}
int32_t x17447 = x17441 * x17441;
int32_t x17452;
if (x17428) {
x17452 = 0;
} else {
x17452 = x17353;
}
int32_t x17453;
if (x17428) {
x17453 = 0;
} else {
x17453 = 1;
}
float* x238 = x5+6321600;
bool x17516 = x17441 == 1;
bool x17517 = x17516 || true;
bool x17518 = x17517 || x17516;
bool x17528 = x17441 <= 1;
int32_t x17529;
if (x17528) {
x17529 = 1;
} else {
x17529 = x17441;
}
int32_t x17535 = x17529 * x17529;
int32_t x17540;
if (x17516) {
x17540 = 0;
} else {
x17540 = x17441;
}
int32_t x17541;
if (x17516) {
x17541 = 0;
} else {
x17541 = 1;
}
float* x146 = x5+6322624;
bool x17579 = x17529 == 1;
bool x17580 = x17579 || x16167;
bool x17581 = x17529 == x16117;
bool x17582 = x17580 || x17581;
bool x17592 = x17529 <= x16117;
int32_t x17593;
if (x17592) {
x17593 = x16117;
} else {
x17593 = x17529;
}
int32_t x17608;
if (x17579) {
x17608 = 0;
} else {
x17608 = x17529;
}
int32_t x17609;
if (x17579) {
x17609 = 0;
} else {
x17609 = 1;
}
int32_t x17655 = x17529 - 1;
int32_t x17656 = x17655 / 1;
int32_t x17657 = x17656 + 1;
int32_t x17661 = 16384 * x17657;
int32_t x17662 = x17661 * x17657;
int32_t x17658 = x17657 * x17657;
int32_t x17659 = 256 * x17658;
float* x22 = x5+6325696;
bool x17736 = x17657 == 1;
bool x17737 = x17736 || true;
bool x17738 = x17737 || x17736;
bool x17748 = x17657 <= 1;
int32_t x17749;
if (x17748) {
x17749 = 1;
} else {
x17749 = x17657;
}
int32_t x17755 = x17749 * x17749;
int32_t x17759;
if (x17736) {
x17759 = 0;
} else {
x17759 = x17657;
}
int32_t x17760;
if (x17736) {
x17760 = 0;
} else {
x17760 = 1;
}
float* x254 = x5+6588352;
float* x69 = x5+6588608;
bool x17839 = x17749 == 1;
bool x17840 = x17839 || true;
bool x17841 = x17840 || x17839;
bool x17851 = x17749 <= 1;
int32_t x17852;
if (x17851) {
x17852 = 1;
} else {
x17852 = x17749;
}
int32_t x17858 = x17852 * x17852;
int32_t x17863;
if (x17839) {
x17863 = 0;
} else {
x17863 = x17749;
}
int32_t x17864;
if (x17839) {
x17864 = 0;
} else {
x17864 = 1;
}
bool x17927 = x17852 == 1;
bool x17928 = x17927 || true;
bool x17929 = x17928 || x17927;
bool x17939 = x17852 <= 1;
int32_t x17940;
if (x17939) {
x17940 = 1;
} else {
x17940 = x17852;
}
int32_t x17946 = x17940 * x17940;
int32_t x17951;
if (x17927) {
x17951 = 0;
} else {
x17951 = x17852;
}
int32_t x17952;
if (x17927) {
x17952 = 0;
} else {
x17952 = 1;
}
float* x77 = x5+6587840;
bool x18015 = x17940 == 1;
bool x18016 = x18015 || true;
bool x18017 = x18016 || x18015;
bool x18027 = x17940 <= 1;
int32_t x18028;
if (x18027) {
x18028 = 1;
} else {
x18028 = x17940;
}
int32_t x18034 = x18028 * x18028;
int32_t x18039;
if (x18015) {
x18039 = 0;
} else {
x18039 = x17940;
}
int32_t x18040;
if (x18015) {
x18040 = 0;
} else {
x18040 = 1;
}
float* x185 = x5+6588096;
int32_t x18087 = x18028 + 2;
int32_t x18088 = x18087 - 3;
int32_t x18089 = x18088 / 1;
int32_t x18090 = x18089 + 1;
int32_t x18094 = 16384 * x18090;
int32_t x18095 = x18094 * x18090;
int32_t x18091 = x18090 * x18090;
int32_t x18092 = 256 * x18091;
float* x262 = x5+6588864;
bool x18217 = x18090 == 1;
bool x18218 = x18217 || true;
bool x18219 = x18218 || x18217;
bool x18229 = x18090 <= 1;
int32_t x18230;
if (x18229) {
x18230 = 1;
} else {
x18230 = x18090;
}
int32_t x18236 = x18230 * x18230;
int32_t x18240;
if (x18217) {
x18240 = 0;
} else {
x18240 = x18090;
}
int32_t x18241;
if (x18217) {
x18241 = 0;
} else {
x18241 = 1;
}
float* x250 = x5+7179200;
float* x104 = x5+7179456;
bool x18320 = x18230 == 1;
bool x18321 = x18320 || true;
bool x18322 = x18321 || x18320;
bool x18332 = x18230 <= 1;
int32_t x18333;
if (x18332) {
x18333 = 1;
} else {
x18333 = x18230;
}
int32_t x18339 = x18333 * x18333;
int32_t x18344;
if (x18320) {
x18344 = 0;
} else {
x18344 = x18230;
}
int32_t x18345;
if (x18320) {
x18345 = 0;
} else {
x18345 = 1;
}
bool x18408 = x18333 == 1;
bool x18409 = x18408 || true;
bool x18410 = x18409 || x18408;
bool x18420 = x18333 <= 1;
int32_t x18421;
if (x18420) {
x18421 = 1;
} else {
x18421 = x18333;
}
int32_t x18427 = x18421 * x18421;
int32_t x18432;
if (x18408) {
x18432 = 0;
} else {
x18432 = x18333;
}
int32_t x18433;
if (x18408) {
x18433 = 0;
} else {
x18433 = 1;
}
float* x168 = x5+7178688;
bool x18496 = x18421 == 1;
bool x18497 = x18496 || true;
bool x18498 = x18497 || x18496;
bool x18508 = x18421 <= 1;
int32_t x18509;
if (x18508) {
x18509 = 1;
} else {
x18509 = x18421;
}
int32_t x18515 = x18509 * x18509;
int32_t x18520;
if (x18496) {
x18520 = 0;
} else {
x18520 = x18421;
}
int32_t x18521;
if (x18496) {
x18521 = 0;
} else {
x18521 = 1;
}
float* x109 = x5+7178944;
int32_t x18568 = x18509 - 1;
int32_t x18569 = x18568 / 1;
int32_t x18570 = x18569 + 1;
int32_t x18574 = 65536 * x18570;
int32_t x18575 = x18574 * x18570;
int32_t x18571 = x18570 * x18570;
int32_t x18572 = 1024 * x18571;
float* x221 = x5+7179712;
bool x18649 = x18570 == 1;
bool x18650 = x18649 || true;
bool x18651 = x18650 || x18649;
bool x18661 = x18570 <= 1;
int32_t x18662;
if (x18661) {
x18662 = 1;
} else {
x18662 = x18570;
}
int32_t x18668 = x18662 * x18662;
int32_t x18672;
if (x18649) {
x18672 = 0;
} else {
x18672 = x18570;
}
int32_t x18673;
if (x18649) {
x18673 = 0;
} else {
x18673 = 1;
}
float* x209 = x5+7443904;
float* x272 = x5+7444928;
bool x18752 = x18662 == 1;
bool x18753 = x18752 || true;
bool x18754 = x18753 || x18752;
bool x18764 = x18662 <= 1;
int32_t x18765;
if (x18764) {
x18765 = 1;
} else {
x18765 = x18662;
}
int32_t x18771 = x18765 * x18765;
int32_t x18776;
if (x18752) {
x18776 = 0;
} else {
x18776 = x18662;
}
int32_t x18777;
if (x18752) {
x18777 = 0;
} else {
x18777 = 1;
}
bool x18840 = x18765 == 1;
bool x18841 = x18840 || true;
bool x18842 = x18841 || x18840;
bool x18852 = x18765 <= 1;
int32_t x18853;
if (x18852) {
x18853 = 1;
} else {
x18853 = x18765;
}
int32_t x18859 = x18853 * x18853;
int32_t x18864;
if (x18840) {
x18864 = 0;
} else {
x18864 = x18765;
}
int32_t x18865;
if (x18840) {
x18865 = 0;
} else {
x18865 = 1;
}
float* x59 = x5+7441856;
bool x18928 = x18853 == 1;
bool x18929 = x18928 || true;
bool x18930 = x18929 || x18928;
bool x18940 = x18853 <= 1;
int32_t x18941;
if (x18940) {
x18941 = 1;
} else {
x18941 = x18853;
}
int32_t x18947 = x18941 * x18941;
int32_t x18952;
if (x18928) {
x18952 = 0;
} else {
x18952 = x18853;
}
int32_t x18953;
if (x18928) {
x18953 = 0;
} else {
x18953 = 1;
}
float* x120 = x5+7442880;
bool x18991 = x18941 == 1;
bool x18992 = x18991 || x17579;
bool x18993 = x18941 == x17529;
bool x18994 = x18992 || x18993;
bool x19004 = x18941 <= x17529;
int32_t x19005;
if (x19004) {
x19005 = x17529;
} else {
x19005 = x18941;
}
int32_t x19020;
if (x18991) {
x19020 = 0;
} else {
x19020 = x18941;
}
int32_t x19021;
if (x18991) {
x19021 = 0;
} else {
x19021 = 1;
}
int32_t x19067 = x18941 - 1;
int32_t x19068 = x19067 / 1;
int32_t x19069 = x19068 + 1;
int32_t x19073 = 16384 * x19069;
int32_t x19074 = x19073 * x19069;
int32_t x19070 = x19069 * x19069;
int32_t x19071 = 256 * x19070;
float* x151 = x5+7445952;
bool x19148 = x19069 == 1;
bool x19149 = x19148 || true;
bool x19150 = x19149 || x19148;
bool x19160 = x19069 <= 1;
int32_t x19161;
if (x19160) {
x19161 = 1;
} else {
x19161 = x19069;
}
int32_t x19167 = x19161 * x19161;
int32_t x19171;
if (x19148) {
x19171 = 0;
} else {
x19171 = x19069;
}
int32_t x19172;
if (x19148) {
x19172 = 0;
} else {
x19172 = 1;
}
float* x80 = x5+7708608;
float* x176 = x5+7708864;
bool x19251 = x19161 == 1;
bool x19252 = x19251 || true;
bool x19253 = x19252 || x19251;
bool x19263 = x19161 <= 1;
int32_t x19264;
if (x19263) {
x19264 = 1;
} else {
x19264 = x19161;
}
int32_t x19270 = x19264 * x19264;
int32_t x19275;
if (x19251) {
x19275 = 0;
} else {
x19275 = x19161;
}
int32_t x19276;
if (x19251) {
x19276 = 0;
} else {
x19276 = 1;
}
bool x19339 = x19264 == 1;
bool x19340 = x19339 || true;
bool x19341 = x19340 || x19339;
bool x19351 = x19264 <= 1;
int32_t x19352;
if (x19351) {
x19352 = 1;
} else {
x19352 = x19264;
}
int32_t x19358 = x19352 * x19352;
int32_t x19363;
if (x19339) {
x19363 = 0;
} else {
x19363 = x19264;
}
int32_t x19364;
if (x19339) {
x19364 = 0;
} else {
x19364 = 1;
}
float* x85 = x5+7708096;
bool x19427 = x19352 == 1;
bool x19428 = x19427 || true;
bool x19429 = x19428 || x19427;
bool x19439 = x19352 <= 1;
int32_t x19440;
if (x19439) {
x19440 = 1;
} else {
x19440 = x19352;
}
int32_t x19446 = x19440 * x19440;
int32_t x19451;
if (x19427) {
x19451 = 0;
} else {
x19451 = x19352;
}
int32_t x19452;
if (x19427) {
x19452 = 0;
} else {
x19452 = 1;
}
float* x253 = x5+7708352;
int32_t x19499 = x19440 + 2;
int32_t x19500 = x19499 - 3;
int32_t x19501 = x19500 / 1;
int32_t x19502 = x19501 + 1;
int32_t x19506 = 16384 * x19502;
int32_t x19507 = x19506 * x19502;
int32_t x19503 = x19502 * x19502;
int32_t x19504 = 256 * x19503;
float* x226 = x5+7709120;
bool x19629 = x19502 == 1;
bool x19630 = x19629 || true;
bool x19631 = x19630 || x19629;
bool x19641 = x19502 <= 1;
int32_t x19642;
if (x19641) {
x19642 = 1;
} else {
x19642 = x19502;
}
int32_t x19648 = x19642 * x19642;
int32_t x19652;
if (x19629) {
x19652 = 0;
} else {
x19652 = x19502;
}
int32_t x19653;
if (x19629) {
x19653 = 0;
} else {
x19653 = 1;
}
float* x70 = x5+8299456;
float* x240 = x5+8299712;
bool x19732 = x19642 == 1;
bool x19733 = x19732 || true;
bool x19734 = x19733 || x19732;
bool x19744 = x19642 <= 1;
int32_t x19745;
if (x19744) {
x19745 = 1;
} else {
x19745 = x19642;
}
int32_t x19751 = x19745 * x19745;
int32_t x19756;
if (x19732) {
x19756 = 0;
} else {
x19756 = x19642;
}
int32_t x19757;
if (x19732) {
x19757 = 0;
} else {
x19757 = 1;
}
bool x19820 = x19745 == 1;
bool x19821 = x19820 || true;
bool x19822 = x19821 || x19820;
bool x19832 = x19745 <= 1;
int32_t x19833;
if (x19832) {
x19833 = 1;
} else {
x19833 = x19745;
}
int32_t x19839 = x19833 * x19833;
int32_t x19844;
if (x19820) {
x19844 = 0;
} else {
x19844 = x19745;
}
int32_t x19845;
if (x19820) {
x19845 = 0;
} else {
x19845 = 1;
}
float* x141 = x5+8298944;
bool x19908 = x19833 == 1;
bool x19909 = x19908 || true;
bool x19910 = x19909 || x19908;
bool x19920 = x19833 <= 1;
int32_t x19921;
if (x19920) {
x19921 = 1;
} else {
x19921 = x19833;
}
int32_t x19927 = x19921 * x19921;
int32_t x19932;
if (x19908) {
x19932 = 0;
} else {
x19932 = x19833;
}
int32_t x19933;
if (x19908) {
x19933 = 0;
} else {
x19933 = 1;
}
float* x189 = x5+8299200;
int32_t x19980 = x19921 - 1;
int32_t x19981 = x19980 / 1;
int32_t x19982 = x19981 + 1;
int32_t x19986 = 65536 * x19982;
int32_t x19987 = x19986 * x19982;
int32_t x19983 = x19982 * x19982;
int32_t x19984 = 1024 * x19983;
float* x97 = x5+8299968;
bool x20061 = x19982 == 1;
bool x20062 = x20061 || true;
bool x20063 = x20062 || x20061;
bool x20073 = x19982 <= 1;
int32_t x20074;
if (x20073) {
x20074 = 1;
} else {
x20074 = x19982;
}
int32_t x20080 = x20074 * x20074;
int32_t x20084;
if (x20061) {
x20084 = 0;
} else {
x20084 = x19982;
}
int32_t x20085;
if (x20061) {
x20085 = 0;
} else {
x20085 = 1;
}
float* x122 = x5+8564160;
float* x183 = x5+8565184;
bool x20164 = x20074 == 1;
bool x20165 = x20164 || true;
bool x20166 = x20165 || x20164;
bool x20176 = x20074 <= 1;
int32_t x20177;
if (x20176) {
x20177 = 1;
} else {
x20177 = x20074;
}
int32_t x20183 = x20177 * x20177;
int32_t x20188;
if (x20164) {
x20188 = 0;
} else {
x20188 = x20074;
}
int32_t x20189;
if (x20164) {
x20189 = 0;
} else {
x20189 = 1;
}
bool x20252 = x20177 == 1;
bool x20253 = x20252 || true;
bool x20254 = x20253 || x20252;
bool x20264 = x20177 <= 1;
int32_t x20265;
if (x20264) {
x20265 = 1;
} else {
x20265 = x20177;
}
int32_t x20271 = x20265 * x20265;
int32_t x20276;
if (x20252) {
x20276 = 0;
} else {
x20276 = x20177;
}
int32_t x20277;
if (x20252) {
x20277 = 0;
} else {
x20277 = 1;
}
float* x248 = x5+8562112;
bool x20340 = x20265 == 1;
bool x20341 = x20340 || true;
bool x20342 = x20341 || x20340;
bool x20352 = x20265 <= 1;
int32_t x20353;
if (x20352) {
x20353 = 1;
} else {
x20353 = x20265;
}
int32_t x20359 = x20353 * x20353;
int32_t x20364;
if (x20340) {
x20364 = 0;
} else {
x20364 = x20265;
}
int32_t x20365;
if (x20340) {
x20365 = 0;
} else {
x20365 = 1;
}
float* x93 = x5+8563136;
bool x20403 = x20353 == 1;
bool x20404 = x20403 || x18991;
bool x20405 = x20353 == x18941;
bool x20406 = x20404 || x20405;
bool x20416 = x20353 <= x18941;
int32_t x20417;
if (x20416) {
x20417 = x18941;
} else {
x20417 = x20353;
}
int32_t x20432;
if (x20403) {
x20432 = 0;
} else {
x20432 = x20353;
}
int32_t x20433;
if (x20403) {
x20433 = 0;
} else {
x20433 = 1;
}
int32_t x20479 = x20353 - 1;
int32_t x20480 = x20479 / 1;
int32_t x20481 = x20480 + 1;
int32_t x20485 = 32768 * x20481;
int32_t x20486 = x20485 * x20481;
int32_t x20482 = x20481 * x20481;
int32_t x20483 = 512 * x20482;
float* x139 = x5+8566208;
bool x20560 = x20481 == 1;
bool x20561 = x20560 || true;
bool x20562 = x20561 || x20560;
bool x20572 = x20481 <= 1;
int32_t x20573;
if (x20572) {
x20573 = 1;
} else {
x20573 = x20481;
}
int32_t x20579 = x20573 * x20573;
int32_t x20583;
if (x20560) {
x20583 = 0;
} else {
x20583 = x20481;
}
int32_t x20584;
if (x20560) {
x20584 = 0;
} else {
x20584 = 1;
}
float* x67 = x5+9091520;
float* x121 = x5+9092032;
bool x20663 = x20573 == 1;
bool x20664 = x20663 || true;
bool x20665 = x20664 || x20663;
bool x20675 = x20573 <= 1;
int32_t x20676;
if (x20675) {
x20676 = 1;
} else {
x20676 = x20573;
}
int32_t x20682 = x20676 * x20676;
int32_t x20687;
if (x20663) {
x20687 = 0;
} else {
x20687 = x20573;
}
int32_t x20688;
if (x20663) {
x20688 = 0;
} else {
x20688 = 1;
}
bool x20751 = x20676 == 1;
bool x20752 = x20751 || true;
bool x20753 = x20752 || x20751;
bool x20763 = x20676 <= 1;
int32_t x20764;
if (x20763) {
x20764 = 1;
} else {
x20764 = x20676;
}
int32_t x20770 = x20764 * x20764;
int32_t x20775;
if (x20751) {
x20775 = 0;
} else {
x20775 = x20676;
}
int32_t x20776;
if (x20751) {
x20776 = 0;
} else {
x20776 = 1;
}
float* x201 = x5+9090496;
bool x20839 = x20764 == 1;
bool x20840 = x20839 || true;
bool x20841 = x20840 || x20839;
bool x20851 = x20764 <= 1;
int32_t x20852;
if (x20851) {
x20852 = 1;
} else {
x20852 = x20764;
}
int32_t x20858 = x20852 * x20852;
int32_t x20863;
if (x20839) {
x20863 = 0;
} else {
x20863 = x20764;
}
int32_t x20864;
if (x20839) {
x20864 = 0;
} else {
x20864 = 1;
}
float* x224 = x5+9091008;
int32_t x20911 = x20852 + 2;
int32_t x20912 = x20911 - 3;
int32_t x20913 = x20912 / 2;
int32_t x20914 = x20913 + 1;
int32_t x20918 = 32768 * x20914;
int32_t x20919 = x20918 * x20914;
int32_t x20915 = x20914 * x20914;
int32_t x20916 = 512 * x20915;
float* x34 = x5+9092544;
bool x21025 = x20914 == 1;
bool x21026 = x21025 || true;
bool x21027 = x21026 || x21025;
bool x21037 = x20914 <= 1;
int32_t x21038;
if (x21037) {
x21038 = 1;
} else {
x21038 = x20914;
}
int32_t x21044 = x21038 * x21038;
int32_t x21048;
if (x21025) {
x21048 = 0;
} else {
x21048 = x20914;
}
int32_t x21049;
if (x21025) {
x21049 = 0;
} else {
x21049 = 1;
}
float* x113 = x5+11452864;
float* x50 = x5+11453376;
bool x21128 = x21038 == 1;
bool x21129 = x21128 || true;
bool x21130 = x21129 || x21128;
bool x21140 = x21038 <= 1;
int32_t x21141;
if (x21140) {
x21141 = 1;
} else {
x21141 = x21038;
}
int32_t x21147 = x21141 * x21141;
int32_t x21152;
if (x21128) {
x21152 = 0;
} else {
x21152 = x21038;
}
int32_t x21153;
if (x21128) {
x21153 = 0;
} else {
x21153 = 1;
}
bool x21216 = x21141 == 1;
bool x21217 = x21216 || true;
bool x21218 = x21217 || x21216;
bool x21228 = x21141 <= 1;
int32_t x21229;
if (x21228) {
x21229 = 1;
} else {
x21229 = x21141;
}
int32_t x21235 = x21229 * x21229;
int32_t x21240;
if (x21216) {
x21240 = 0;
} else {
x21240 = x21141;
}
int32_t x21241;
if (x21216) {
x21241 = 0;
} else {
x21241 = 1;
}
float* x205 = x5+11451840;
bool x21304 = x21229 == 1;
bool x21305 = x21304 || true;
bool x21306 = x21305 || x21304;
bool x21316 = x21229 <= 1;
int32_t x21317;
if (x21316) {
x21317 = 1;
} else {
x21317 = x21229;
}
int32_t x21323 = x21317 * x21317;
int32_t x21328;
if (x21304) {
x21328 = 0;
} else {
x21328 = x21229;
}
int32_t x21329;
if (x21304) {
x21329 = 0;
} else {
x21329 = 1;
}
float* x159 = x5+11452352;
int32_t x21376 = x21317 - 1;
int32_t x21377 = x21376 / 1;
int32_t x21378 = x21377 + 1;
int32_t x21382 = 131072 * x21378;
int32_t x21383 = x21382 * x21378;
int32_t x21379 = x21378 * x21378;
int32_t x21380 = 2048 * x21379;
float* x212 = x5+11453888;
bool x21457 = x21378 == 1;
bool x21458 = x21457 || true;
bool x21459 = x21458 || x21457;
bool x21469 = x21378 <= 1;
int32_t x21470;
if (x21469) {
x21470 = 1;
} else {
x21470 = x21378;
}
int32_t x21476 = x21470 * x21470;
int32_t x21480;
if (x21457) {
x21480 = 0;
} else {
x21480 = x21378;
}
int32_t x21481;
if (x21457) {
x21481 = 0;
} else {
x21481 = 1;
}
float* x115 = x5+12506560;
float* x193 = x5+12508608;
bool x21561 = x21470 == 1;
bool x21562 = x21561 || true;
bool x21563 = x21562 || x21561;
bool x21573 = x21470 <= 1;
int32_t x21574;
if (x21573) {
x21574 = 1;
} else {
x21574 = x21470;
}
int32_t x21580 = x21574 * x21574;
int32_t x21585;
if (x21561) {
x21585 = 0;
} else {
x21585 = x21470;
}
int32_t x21586;
if (x21561) {
x21586 = 0;
} else {
x21586 = 1;
}
bool x21649 = x21574 == 1;
bool x21650 = x21649 || true;
bool x21651 = x21650 || x21649;
bool x21661 = x21574 <= 1;
int32_t x21662;
if (x21661) {
x21662 = 1;
} else {
x21662 = x21574;
}
int32_t x21668 = x21662 * x21662;
int32_t x21673;
if (x21649) {
x21673 = 0;
} else {
x21673 = x21574;
}
int32_t x21674;
if (x21649) {
x21674 = 0;
} else {
x21674 = 1;
}
float* x239 = x5+12502464;
bool x21737 = x21662 == 1;
bool x21738 = x21737 || true;
bool x21739 = x21738 || x21737;
bool x21749 = x21662 <= 1;
int32_t x21750;
if (x21749) {
x21750 = 1;
} else {
x21750 = x21662;
}
int32_t x21756 = x21750 * x21750;
int32_t x21761;
if (x21737) {
x21761 = 0;
} else {
x21761 = x21662;
}
int32_t x21762;
if (x21737) {
x21762 = 0;
} else {
x21762 = 1;
}
float* x62 = x5+12504512;
int32_t x21796 = x20479 / 2;
int32_t x21797 = x21796 + 1;
int32_t x21801 = 131072 * x21797;
int32_t x21802 = x21801 * x21797;
int32_t x21798 = x21797 * x21797;
int32_t x21799 = 2048 * x21798;
float* x214 = x5+12510656;
bool x21882 = x21797 == 1;
bool x21883 = x21882 || true;
bool x21884 = x21883 || x21882;
bool x21894 = x21797 <= 1;
int32_t x21895;
if (x21894) {
x21895 = 1;
} else {
x21895 = x21797;
}
int32_t x21901 = x21895 * x21895;
int32_t x21905;
if (x21882) {
x21905 = 0;
} else {
x21905 = x21797;
}
int32_t x21906;
if (x21882) {
x21906 = 0;
} else {
x21906 = 1;
}
float* x64 = x5+14611904;
float* x125 = x5+14613952;
bool x21985 = x21895 == 1;
bool x21986 = x21985 || true;
bool x21987 = x21986 || x21985;
bool x21997 = x21895 <= 1;
int32_t x21998;
if (x21997) {
x21998 = 1;
} else {
x21998 = x21895;
}
int32_t x22004 = x21998 * x21998;
int32_t x22009;
if (x21985) {
x22009 = 0;
} else {
x22009 = x21895;
}
int32_t x22010;
if (x21985) {
x22010 = 0;
} else {
x22010 = 1;
}
bool x22073 = x21998 == 1;
bool x22074 = x22073 || true;
bool x22075 = x22074 || x22073;
bool x22085 = x21998 <= 1;
int32_t x22086;
if (x22085) {
x22086 = 1;
} else {
x22086 = x21998;
}
int32_t x22092 = x22086 * x22086;
int32_t x22097;
if (x22073) {
x22097 = 0;
} else {
x22097 = x21998;
}
int32_t x22098;
if (x22073) {
x22098 = 0;
} else {
x22098 = 1;
}
float* x173 = x5+14607808;
bool x22161 = x22086 == 1;
bool x22162 = x22161 || true;
bool x22163 = x22162 || x22161;
bool x22173 = x22086 <= 1;
int32_t x22174;
if (x22173) {
x22174 = 1;
} else {
x22174 = x22086;
}
int32_t x22180 = x22174 * x22174;
int32_t x22185;
if (x22161) {
x22185 = 0;
} else {
x22185 = x22086;
}
int32_t x22186;
if (x22161) {
x22186 = 0;
} else {
x22186 = 1;
}
float* x107 = x5+14609856;
bool x22225 = x21750 == 1;
bool x22226 = x22174 == 1;
bool x22227 = x22225 || x22226;
bool x22228 = x21750 == x22174;
bool x22229 = x22227 || x22228;
bool x22239 = x21750 <= x22174;
int32_t x22240;
if (x22239) {
x22240 = x22174;
} else {
x22240 = x21750;
}
int32_t x22255;
if (x22225) {
x22255 = 0;
} else {
x22255 = x21750;
}
int32_t x22256;
if (x22225) {
x22256 = 0;
} else {
x22256 = 1;
}
int32_t x22258;
if (x22226) {
x22258 = 0;
} else {
x22258 = x22174;
}
int32_t x22259;
if (x22226) {
x22259 = 0;
} else {
x22259 = 1;
}
int32_t x22305 = x21750 - 1;
int32_t x22306 = x22305 / 1;
int32_t x22307 = x22306 + 1;
int32_t x22311 = 32768 * x22307;
int32_t x22312 = x22311 * x22307;
int32_t x22308 = x22307 * x22307;
int32_t x22309 = 512 * x22308;
float* x215 = x5+14616000;
bool x22386 = x22307 == 1;
bool x22387 = x22386 || true;
bool x22388 = x22387 || x22386;
bool x22398 = x22307 <= 1;
int32_t x22399;
if (x22398) {
x22399 = 1;
} else {
x22399 = x22307;
}
int32_t x22405 = x22399 * x22399;
int32_t x22409;
if (x22386) {
x22409 = 0;
} else {
x22409 = x22307;
}
int32_t x22410;
if (x22386) {
x22410 = 0;
} else {
x22410 = 1;
}
float* x154 = x5+15665600;
float* x65 = x5+15666112;
bool x22489 = x22399 == 1;
bool x22490 = x22489 || true;
bool x22491 = x22490 || x22489;
bool x22501 = x22399 <= 1;
int32_t x22502;
if (x22501) {
x22502 = 1;
} else {
x22502 = x22399;
}
int32_t x22508 = x22502 * x22502;
int32_t x22513;
if (x22489) {
x22513 = 0;
} else {
x22513 = x22399;
}
int32_t x22514;
if (x22489) {
x22514 = 0;
} else {
x22514 = 1;
}
bool x22577 = x22502 == 1;
bool x22578 = x22577 || true;
bool x22579 = x22578 || x22577;
bool x22589 = x22502 <= 1;
int32_t x22590;
if (x22589) {
x22590 = 1;
} else {
x22590 = x22502;
}
int32_t x22596 = x22590 * x22590;
int32_t x22601;
if (x22577) {
x22601 = 0;
} else {
x22601 = x22502;
}
int32_t x22602;
if (x22577) {
x22602 = 0;
} else {
x22602 = 1;
}
float* x46 = x5+15664576;
bool x22665 = x22590 == 1;
bool x22666 = x22665 || true;
bool x22667 = x22666 || x22665;
bool x22677 = x22590 <= 1;
int32_t x22678;
if (x22677) {
x22678 = 1;
} else {
x22678 = x22590;
}
int32_t x22684 = x22678 * x22678;
int32_t x22689;
if (x22665) {
x22689 = 0;
} else {
x22689 = x22590;
}
int32_t x22690;
if (x22665) {
x22690 = 0;
} else {
x22690 = 1;
}
float* x137 = x5+15665088;
int32_t x22737 = x22678 + 2;
int32_t x22738 = x22737 - 3;
int32_t x22739 = x22738 / 1;
int32_t x22740 = x22739 + 1;
int32_t x22744 = 32768 * x22740;
int32_t x22745 = x22744 * x22740;
int32_t x22741 = x22740 * x22740;
int32_t x22742 = 512 * x22741;
float* x155 = x5+15666624;
bool x22867 = x22740 == 1;
bool x22868 = x22867 || true;
bool x22869 = x22868 || x22867;
bool x22879 = x22740 <= 1;
int32_t x22880;
if (x22879) {
x22880 = 1;
} else {
x22880 = x22740;
}
int32_t x22886 = x22880 * x22880;
int32_t x22890;
if (x22867) {
x22890 = 0;
} else {
x22890 = x22740;
}
int32_t x22891;
if (x22867) {
x22891 = 0;
} else {
x22891 = 1;
}
float* x138 = x5+18026944;
float* x195 = x5+18027456;
bool x22970 = x22880 == 1;
bool x22971 = x22970 || true;
bool x22972 = x22971 || x22970;
bool x22982 = x22880 <= 1;
int32_t x22983;
if (x22982) {
x22983 = 1;
} else {
x22983 = x22880;
}
int32_t x22989 = x22983 * x22983;
int32_t x22994;
if (x22970) {
x22994 = 0;
} else {
x22994 = x22880;
}
int32_t x22995;
if (x22970) {
x22995 = 0;
} else {
x22995 = 1;
}
bool x23058 = x22983 == 1;
bool x23059 = x23058 || true;
bool x23060 = x23059 || x23058;
bool x23070 = x22983 <= 1;
int32_t x23071;
if (x23070) {
x23071 = 1;
} else {
x23071 = x22983;
}
int32_t x23077 = x23071 * x23071;
int32_t x23082;
if (x23058) {
x23082 = 0;
} else {
x23082 = x22983;
}
int32_t x23083;
if (x23058) {
x23083 = 0;
} else {
x23083 = 1;
}
float* x160 = x5+18025920;
bool x23146 = x23071 == 1;
bool x23147 = x23146 || true;
bool x23148 = x23147 || x23146;
bool x23158 = x23071 <= 1;
int32_t x23159;
if (x23158) {
x23159 = 1;
} else {
x23159 = x23071;
}
int32_t x23165 = x23159 * x23159;
int32_t x23170;
if (x23146) {
x23170 = 0;
} else {
x23170 = x23071;
}
int32_t x23171;
if (x23146) {
x23171 = 0;
} else {
x23171 = 1;
}
float* x66 = x5+18026432;
int32_t x23218 = x23159 - 1;
int32_t x23219 = x23218 / 1;
int32_t x23220 = x23219 + 1;
int32_t x23224 = 131072 * x23220;
int32_t x23225 = x23224 * x23220;
int32_t x23221 = x23220 * x23220;
int32_t x23222 = 2048 * x23221;
float* x47 = x5+18027968;
bool x23299 = x23220 == 1;
bool x23300 = x23299 || true;
bool x23301 = x23300 || x23299;
bool x23311 = x23220 <= 1;
int32_t x23312;
if (x23311) {
x23312 = 1;
} else {
x23312 = x23220;
}
int32_t x23318 = x23312 * x23312;
int32_t x23322;
if (x23299) {
x23322 = 0;
} else {
x23322 = x23220;
}
int32_t x23323;
if (x23299) {
x23323 = 0;
} else {
x23323 = 1;
}
float* x68 = x5+19080640;
float* x245 = x5+19082688;
bool x23402 = x23312 == 1;
bool x23403 = x23402 || true;
bool x23404 = x23403 || x23402;
bool x23414 = x23312 <= 1;
int32_t x23415;
if (x23414) {
x23415 = 1;
} else {
x23415 = x23312;
}
int32_t x23421 = x23415 * x23415;
int32_t x23426;
if (x23402) {
x23426 = 0;
} else {
x23426 = x23312;
}
int32_t x23427;
if (x23402) {
x23427 = 0;
} else {
x23427 = 1;
}
bool x23490 = x23415 == 1;
bool x23491 = x23490 || true;
bool x23492 = x23491 || x23490;
bool x23502 = x23415 <= 1;
int32_t x23503;
if (x23502) {
x23503 = 1;
} else {
x23503 = x23415;
}
int32_t x23509 = x23503 * x23503;
int32_t x23514;
if (x23490) {
x23514 = 0;
} else {
x23514 = x23415;
}
int32_t x23515;
if (x23490) {
x23515 = 0;
} else {
x23515 = 1;
}
float* x94 = x5+19076544;
bool x23578 = x23503 == 1;
bool x23579 = x23578 || true;
bool x23580 = x23579 || x23578;
bool x23590 = x23503 <= 1;
int32_t x23591;
if (x23590) {
x23591 = 1;
} else {
x23591 = x23503;
}
int32_t x23597 = x23591 * x23591;
int32_t x23602;
if (x23578) {
x23602 = 0;
} else {
x23602 = x23503;
}
int32_t x23603;
if (x23578) {
x23603 = 0;
} else {
x23603 = 1;
}
float* x144 = x5+19078592;
bool x23641 = x23591 == 1;
bool x23642 = x23641 || x22225;
bool x23643 = x23591 == x21750;
bool x23644 = x23642 || x23643;
bool x23654 = x23591 <= x21750;
int32_t x23655;
if (x23654) {
x23655 = x21750;
} else {
x23655 = x23591;
}
int32_t x23670;
if (x23641) {
x23670 = 0;
} else {
x23670 = x23591;
}
int32_t x23671;
if (x23641) {
x23671 = 0;
} else {
x23671 = 1;
}
int32_t x23717 = x23591 - 1;
int32_t x23718 = x23717 / 1;
int32_t x23719 = x23718 + 1;
int32_t x23723 = 32768 * x23719;
int32_t x23724 = x23723 * x23719;
int32_t x23720 = x23719 * x23719;
int32_t x23721 = 512 * x23720;
float* x265 = x5+19084736;
bool x23798 = x23719 == 1;
bool x23799 = x23798 || true;
bool x23800 = x23799 || x23798;
bool x23810 = x23719 <= 1;
int32_t x23811;
if (x23810) {
x23811 = 1;
} else {
x23811 = x23719;
}
int32_t x23817 = x23811 * x23811;
int32_t x23821;
if (x23798) {
x23821 = 0;
} else {
x23821 = x23719;
}
int32_t x23822;
if (x23798) {
x23822 = 0;
} else {
x23822 = 1;
}
float* x213 = x5+20134336;
float* x255 = x5+20134848;
bool x23901 = x23811 == 1;
bool x23902 = x23901 || true;
bool x23903 = x23902 || x23901;
bool x23913 = x23811 <= 1;
int32_t x23914;
if (x23913) {
x23914 = 1;
} else {
x23914 = x23811;
}
int32_t x23920 = x23914 * x23914;
int32_t x23925;
if (x23901) {
x23925 = 0;
} else {
x23925 = x23811;
}
int32_t x23926;
if (x23901) {
x23926 = 0;
} else {
x23926 = 1;
}
bool x23989 = x23914 == 1;
bool x23990 = x23989 || true;
bool x23991 = x23990 || x23989;
bool x24001 = x23914 <= 1;
int32_t x24002;
if (x24001) {
x24002 = 1;
} else {
x24002 = x23914;
}
int32_t x24008 = x24002 * x24002;
int32_t x24013;
if (x23989) {
x24013 = 0;
} else {
x24013 = x23914;
}
int32_t x24014;
if (x23989) {
x24014 = 0;
} else {
x24014 = 1;
}
float* x15 = x5+20133312;
bool x24077 = x24002 == 1;
bool x24078 = x24077 || true;
bool x24079 = x24078 || x24077;
bool x24089 = x24002 <= 1;
int32_t x24090;
if (x24089) {
x24090 = 1;
} else {
x24090 = x24002;
}
int32_t x24096 = x24090 * x24090;
int32_t x24101;
if (x24077) {
x24101 = 0;
} else {
x24101 = x24002;
}
int32_t x24102;
if (x24077) {
x24102 = 0;
} else {
x24102 = 1;
}
float* x78 = x5+20133824;
int32_t x24149 = x24090 + 2;
int32_t x24150 = x24149 - 3;
int32_t x24151 = x24150 / 1;
int32_t x24152 = x24151 + 1;
int32_t x24156 = 32768 * x24152;
int32_t x24157 = x24156 * x24152;
int32_t x24153 = x24152 * x24152;
int32_t x24154 = 512 * x24153;
float* x28 = x5+20135360;
bool x24279 = x24152 == 1;
bool x24280 = x24279 || true;
bool x24281 = x24280 || x24279;
bool x24291 = x24152 <= 1;
int32_t x24292;
if (x24291) {
x24292 = 1;
} else {
x24292 = x24152;
}
int32_t x24298 = x24292 * x24292;
int32_t x24302;
if (x24279) {
x24302 = 0;
} else {
x24302 = x24152;
}
int32_t x24303;
if (x24279) {
x24303 = 0;
} else {
x24303 = 1;
}
float* x12 = x5+22495680;
float* x202 = x5+22496192;
bool x24382 = x24292 == 1;
bool x24383 = x24382 || true;
bool x24384 = x24383 || x24382;
bool x24394 = x24292 <= 1;
int32_t x24395;
if (x24394) {
x24395 = 1;
} else {
x24395 = x24292;
}
int32_t x24401 = x24395 * x24395;
int32_t x24406;
if (x24382) {
x24406 = 0;
} else {
x24406 = x24292;
}
int32_t x24407;
if (x24382) {
x24407 = 0;
} else {
x24407 = 1;
}
bool x24470 = x24395 == 1;
bool x24471 = x24470 || true;
bool x24472 = x24471 || x24470;
bool x24482 = x24395 <= 1;
int32_t x24483;
if (x24482) {
x24483 = 1;
} else {
x24483 = x24395;
}
int32_t x24489 = x24483 * x24483;
int32_t x24494;
if (x24470) {
x24494 = 0;
} else {
x24494 = x24395;
}
int32_t x24495;
if (x24470) {
x24495 = 0;
} else {
x24495 = 1;
}
float* x194 = x5+22494656;
bool x24558 = x24483 == 1;
bool x24559 = x24558 || true;
bool x24560 = x24559 || x24558;
bool x24570 = x24483 <= 1;
int32_t x24571;
if (x24570) {
x24571 = 1;
} else {
x24571 = x24483;
}
int32_t x24577 = x24571 * x24571;
int32_t x24582;
if (x24558) {
x24582 = 0;
} else {
x24582 = x24483;
}
int32_t x24583;
if (x24558) {
x24583 = 0;
} else {
x24583 = 1;
}
float* x169 = x5+22495168;
int32_t x24630 = x24571 - 1;
int32_t x24631 = x24630 / 1;
int32_t x24632 = x24631 + 1;
int32_t x24636 = 131072 * x24632;
int32_t x24637 = x24636 * x24632;
int32_t x24633 = x24632 * x24632;
int32_t x24634 = 2048 * x24633;
float* x33 = x5+22496704;
bool x24711 = x24632 == 1;
bool x24712 = x24711 || true;
bool x24713 = x24712 || x24711;
bool x24723 = x24632 <= 1;
int32_t x24724;
if (x24723) {
x24724 = 1;
} else {
x24724 = x24632;
}
int32_t x24730 = x24724 * x24724;
int32_t x24734;
if (x24711) {
x24734 = 0;
} else {
x24734 = x24632;
}
int32_t x24735;
if (x24711) {
x24735 = 0;
} else {
x24735 = 1;
}
float* x260 = x5+23549376;
float* x123 = x5+23551424;
bool x24814 = x24724 == 1;
bool x24815 = x24814 || true;
bool x24816 = x24815 || x24814;
bool x24826 = x24724 <= 1;
int32_t x24827;
if (x24826) {
x24827 = 1;
} else {
x24827 = x24724;
}
int32_t x24833 = x24827 * x24827;
int32_t x24838;
if (x24814) {
x24838 = 0;
} else {
x24838 = x24724;
}
int32_t x24839;
if (x24814) {
x24839 = 0;
} else {
x24839 = 1;
}
bool x24902 = x24827 == 1;
bool x24903 = x24902 || true;
bool x24904 = x24903 || x24902;
bool x24914 = x24827 <= 1;
int32_t x24915;
if (x24914) {
x24915 = 1;
} else {
x24915 = x24827;
}
int32_t x24921 = x24915 * x24915;
int32_t x24926;
if (x24902) {
x24926 = 0;
} else {
x24926 = x24827;
}
int32_t x24927;
if (x24902) {
x24927 = 0;
} else {
x24927 = 1;
}
float* x103 = x5+23545280;
bool x24990 = x24915 == 1;
bool x24991 = x24990 || true;
bool x24992 = x24991 || x24990;
bool x25002 = x24915 <= 1;
int32_t x25003;
if (x25002) {
x25003 = 1;
} else {
x25003 = x24915;
}
int32_t x25009 = x25003 * x25003;
int32_t x25014;
if (x24990) {
x25014 = 0;
} else {
x25014 = x24915;
}
int32_t x25015;
if (x24990) {
x25015 = 0;
} else {
x25015 = 1;
}
float* x181 = x5+23547328;
bool x25053 = x25003 == 1;
bool x25054 = x25053 || x23641;
bool x25055 = x25003 == x23591;
bool x25056 = x25054 || x25055;
bool x25066 = x25003 <= x23591;
int32_t x25067;
if (x25066) {
x25067 = x23591;
} else {
x25067 = x25003;
}
int32_t x25082;
if (x25053) {
x25082 = 0;
} else {
x25082 = x25003;
}
int32_t x25083;
if (x25053) {
x25083 = 0;
} else {
x25083 = 1;
}
bool x25129 = x25003 >= 2;
bool x25130;
if (x25129) {
x25130 = x25129;
} else {
x25130 = false;
}
int32_t x25135 = x25003 - 2;
int32_t x25136 = x25135 / 1;
int32_t x25137 = x25136 + 1;
int32_t x25138 = x25137 * x25137;
float* x227 = x5+23553472;
float* x48 = x5+23573952;
for(int x303=0; x303 < x301; x303++) {
int32_t x304 = x303 * 64;
int32_t x305 = x304 * 3072;
float* x306 = x279+x305;
int* x307 = x280+x304;
printf("input (size Const(64) x Const(3) x Const(32) x Const(32))\n");
float x309 = 0.0f;
for(int x311=0; x311 < 196608; x311++) {
float x312 = x309;
float x314 = x306[x311];
float x313 = fabs(x312);
float x315 = fabs(x314);
bool x316 = x313 > x315;
float x319;
if (x316) {
x319 = x312;
} else {
float x317 = x306[x311];
x319 = x317;
}
x309 = x319;

}
float x323 = x309;
printf("Max Abs: %.5f || ",x323);
for(int x326=0; x326 < 10; x326++) {
float x327 = x306[x326];
printf("%.5f ",x327);

}
printf("\n");
float* x339 = (float*)myMalloc(x338 * sizeof(float));;
float* x343 = (float*)myMalloc(x342 * sizeof(float));;
for(int x345=0; x345 < 64; x345++) {
int32_t x346 = x345 * 3072;
float* x347 = x306+x346;
int32_t x348 = x345 * x335;
float* x349 = x339+x348;
int32_t x350 = x345 * x340;
float* x351 = x343+x350;
for(int x353=0; x353 < 27; x353++) {
int32_t x354 = x353 / 9;
int32_t x358 = x354 * 3;
int32_t x359 = x358 * 3;
int32_t x360 = x359 * x333;
int32_t x361 = x360 * x333;
int32_t x355 = x353 % 9;
int32_t x356 = x355 / 3;
int32_t x362 = x356 * 3;
int32_t x363 = x362 * x333;
int32_t x364 = x363 * x333;
int32_t x365 = x361 + x364;
int32_t x357 = x355 % 3;
int32_t x366 = x357 * x333;
int32_t x367 = x366 * x333;
int32_t x368 = x365 + x367;
float* x369 = x351+x368;
int32_t x370 = x354 * 32;
int32_t x371 = x370 * 32;
float* x372 = x347+x371;
int32_t x385 = 1 - x357;
bool x386 = x385 > 0;
int32_t x387;
if (x386) {
x387 = x385;
} else {
x387 = 0;
}
int32_t x388 = 3 - x357;
int32_t x389 = x388 - 1;
int32_t x390 = 1 - x389;
bool x391 = x390 > 0;
int32_t x392;
if (x391) {
x392 = x390;
} else {
x392 = 0;
}
int32_t x393 = x333 - x392;
int32_t x394 = x393 - x387;
bool x395 = x394 <= 0;
bool x399 = x387 > 0;
int32_t x384 = -1 + x357;
bool x412 = x392 > 0;
for(int x374=0; x374 < x333; x374++) {
int32_t x375 = x374 - 1;
int32_t x376 = x375 + x356;
bool x377 = x376 < 0;
bool x378 = x376 >= 32;
bool x379 = x377 || x378;
if (x379) {
int32_t x380 = x374 * x333;
float* x381 = x369+x380;
memset(x381, 0, 4 * x333);;
} else {
if (x395) {
int32_t x380 = x374 * x333;
float* x396 = x369+x380;
memset(x396, 0, 4 * x333);;
} else {
int32_t x380 = x374 * x333;
if (x399) {
float* x400 = x369+x380;
memset(x400, 0, 4 * x387);;
} else {
}
// may have segfault here
int32_t x405 = x380 + x387;
float* x406 = x369+x405;
int32_t x407 = x376 * 32;
int32_t x408 = x407 + x384;
int32_t x409 = x408 + x387;
float* x410 = x372+x409;
memcpy(x406, x410, 4 * x394);;
if (x412) {
int32_t x413 = x380 + x333;
int32_t x414 = x413 - x392;
float* x415 = x369+x414;
memset(x415, 0, 4 * x392);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x334,27,1,x152,27,x351,x334,1,x349,x334);

}
int32_t x430 = 0;
int32_t x431 = 1;
x431 *= 1;
x430 += 1;
x431 *= 1;
x431 *= 1;
int32_t x436 = x430;
bool x437 = x436 >= 2;
if (x437) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x443 = x436 == 0;
if (x443) {
int32_t x444 = x431;
bool x445 = x444 == 64;
if (x445) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x452 = x431;
int32_t x453 = 64 / x452;
bool x455 = x453 == 1;
bool x458;
if (x454) {
bool x456 = 64 == x453;
bool x457 = x455 || x456;
x458 = x457;
} else {
x458 = false;
}
bool x462;
if (x458) {
x462 = x461;
} else {
x462 = false;
}
bool x463;
if (x462) {
x463 = x461;
} else {
x463 = false;
}
if (x463) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x333,x333,1,x453,1,1);
assert(false && "");
}
bool x469 = 64 <= x453;
int32_t x470;
if (x469) {
x470 = x453;
} else {
x470 = 64;
}
int32_t x479 = x470 * x478;
int32_t x480 = 64 * x479;
float* x481 = (float*)myMalloc(x480 * sizeof(float));;
int32_t x484;
if (x455) {
x484 = 0;
} else {
x484 = 1;
}
for(int x485=0; x485 < 64; x485++) {
int32_t x497 = x335 * x485;
int32_t x491 = x479 * x485;
for(int x487=0; x487 < x470; x487++) {
int32_t x498 = x334 * x487;
int32_t x499 = x497 + x498;
int32_t x504 = x484 * x487;
int32_t x493 = x478 * x487;
for(int x489=0; x489 < x472; x489++) {
int32_t x500 = x482 * x489;
int32_t x501 = x499 + x500;
int32_t x495 = x472 * x489;
for(int x490=0; x490 < x472; x490++) {
int32_t x502 = x483 * x490;
int32_t x503 = x501 + x502;
float x505 = x339[x503];
float x506 = x40[x504];
int32_t x492 = x490 + x491;
int32_t x494 = x492 + x493;
int32_t x496 = x494 + x495;
float x507 = x505 - x506;
x481[x496] = x507;

}

}

}

}
float* x517 = (float*)myMalloc(64 * sizeof(float));;
for(int x518=0; x518 < 64; x518++) {
float x519 = x110[x518];
float x520 = x519 + 1.0E-5f;
x517[x518] = x520;

}
float* x524 = (float*)myMalloc(64 * sizeof(float));;
for(int x525=0; x525 < 64; x525++) {
float x526 = x517[x525];
double x527 = (double)x526;
double x528 = sqrt(x527);
float x529 = (float)x528;
x524[x525] = x529;

}
int32_t x533 = 0;
int32_t x534 = 1;
x534 *= 1;
x533 += 1;
x534 *= 1;
x534 *= 1;
int32_t x539 = x533;
bool x540 = x539 >= 2;
if (x540) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x545 = x539 == 0;
if (x545) {
int32_t x546 = x534;
bool x547 = x546 == 64;
if (x547) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x554 = x534;
bool x556 = x470 == 1;
int32_t x555 = 64 / x554;
bool x557 = x555 == 1;
bool x561;
if (x454) {
bool x558 = x556 || x557;
bool x559 = x470 == x555;
bool x560 = x558 || x559;
x561 = x560;
} else {
x561 = false;
}
bool x565;
if (x561) {
x565 = x564;
} else {
x565 = false;
}
bool x566;
if (x565) {
x566 = x564;
} else {
x566 = false;
}
if (x566) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x470,x472,x472,1,x555,1,1);
assert(false && "");
}
bool x572 = x470 <= x555;
int32_t x573;
if (x572) {
x573 = x555;
} else {
x573 = x470;
}
int32_t x582 = x573 * x581;
int32_t x583 = 64 * x582;
float* x584 = (float*)myMalloc(x583 * sizeof(float));;
int32_t x585;
if (x556) {
x585 = 0;
} else {
x585 = x478;
}
int32_t x588;
if (x557) {
x588 = 0;
} else {
x588 = 1;
}
for(int x589=0; x589 < 64; x589++) {
int32_t x601 = x479 * x589;
int32_t x595 = x582 * x589;
for(int x591=0; x591 < x573; x591++) {
int32_t x602 = x585 * x591;
int32_t x603 = x601 + x602;
int32_t x608 = x588 * x591;
int32_t x597 = x581 * x591;
for(int x593=0; x593 < x575; x593++) {
int32_t x604 = x586 * x593;
int32_t x605 = x603 + x604;
int32_t x599 = x575 * x593;
for(int x594=0; x594 < x575; x594++) {
int32_t x606 = x587 * x594;
int32_t x607 = x605 + x606;
float x609 = x481[x607];
float x610 = x524[x608];
int32_t x596 = x594 + x595;
int32_t x598 = x596 + x597;
int32_t x600 = x598 + x599;
float x611 = x609 / x610;
x584[x600] = x611;

}

}

}

}
int32_t x621 = 0;
int32_t x622 = 1;
x622 *= 1;
x621 += 1;
x622 *= 1;
x622 *= 1;
int32_t x627 = x621;
bool x628 = x627 >= 2;
if (x628) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x633 = x627 == 0;
if (x633) {
int32_t x634 = x622;
bool x635 = x634 == 64;
if (x635) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x642 = x622;
bool x644 = x573 == 1;
int32_t x643 = 64 / x642;
bool x645 = x643 == 1;
bool x649;
if (x454) {
bool x646 = x644 || x645;
bool x647 = x573 == x643;
bool x648 = x646 || x647;
x649 = x648;
} else {
x649 = false;
}
bool x653;
if (x649) {
x653 = x652;
} else {
x653 = false;
}
bool x654;
if (x653) {
x654 = x652;
} else {
x654 = false;
}
if (x654) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x573,x575,x575,1,x643,1,1);
assert(false && "");
}
bool x660 = x573 <= x643;
int32_t x661;
if (x660) {
x661 = x643;
} else {
x661 = x573;
}
int32_t x670 = x661 * x669;
int32_t x671 = 64 * x670;
float* x672 = (float*)myMalloc(x671 * sizeof(float));;
int32_t x673;
if (x644) {
x673 = 0;
} else {
x673 = x581;
}
int32_t x676;
if (x645) {
x676 = 0;
} else {
x676 = 1;
}
for(int x677=0; x677 < 64; x677++) {
int32_t x689 = x582 * x677;
int32_t x683 = x670 * x677;
for(int x679=0; x679 < x661; x679++) {
int32_t x690 = x673 * x679;
int32_t x691 = x689 + x690;
int32_t x696 = x676 * x679;
int32_t x685 = x669 * x679;
for(int x681=0; x681 < x663; x681++) {
int32_t x692 = x674 * x681;
int32_t x693 = x691 + x692;
int32_t x687 = x663 * x681;
for(int x682=0; x682 < x663; x682++) {
int32_t x694 = x675 * x682;
int32_t x695 = x693 + x694;
float x697 = x584[x695];
float x698 = x206[x696];
int32_t x684 = x682 + x683;
int32_t x686 = x684 + x685;
int32_t x688 = x686 + x687;
float x699 = x697 * x698;
x672[x688] = x699;

}

}

}

}
int32_t x709 = 0;
int32_t x710 = 1;
x710 *= 1;
x709 += 1;
x710 *= 1;
x710 *= 1;
int32_t x715 = x709;
bool x716 = x715 >= 2;
if (x716) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x721 = x715 == 0;
if (x721) {
int32_t x722 = x710;
bool x723 = x722 == 64;
if (x723) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x730 = x710;
bool x732 = x661 == 1;
int32_t x731 = 64 / x730;
bool x733 = x731 == 1;
bool x737;
if (x454) {
bool x734 = x732 || x733;
bool x735 = x661 == x731;
bool x736 = x734 || x735;
x737 = x736;
} else {
x737 = false;
}
bool x741;
if (x737) {
x741 = x740;
} else {
x741 = false;
}
bool x742;
if (x741) {
x742 = x740;
} else {
x742 = false;
}
if (x742) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x661,x663,x663,1,x731,1,1);
assert(false && "");
}
bool x748 = x661 <= x731;
int32_t x749;
if (x748) {
x749 = x731;
} else {
x749 = x661;
}
int32_t x758 = x749 * x757;
int32_t x759 = 64 * x758;
float* x760 = (float*)myMalloc(x759 * sizeof(float));;
int32_t x761;
if (x732) {
x761 = 0;
} else {
x761 = x669;
}
int32_t x764;
if (x733) {
x764 = 0;
} else {
x764 = 1;
}
for(int x765=0; x765 < 64; x765++) {
int32_t x777 = x670 * x765;
int32_t x771 = x758 * x765;
for(int x767=0; x767 < x749; x767++) {
int32_t x778 = x761 * x767;
int32_t x779 = x777 + x778;
int32_t x784 = x764 * x767;
int32_t x773 = x757 * x767;
for(int x769=0; x769 < x751; x769++) {
int32_t x780 = x762 * x769;
int32_t x781 = x779 + x780;
int32_t x775 = x751 * x769;
for(int x770=0; x770 < x751; x770++) {
int32_t x782 = x763 * x770;
int32_t x783 = x781 + x782;
float x785 = x672[x783];
float x786 = x251[x784];
int32_t x772 = x770 + x771;
int32_t x774 = x772 + x773;
int32_t x776 = x774 + x775;
float x787 = x785 + x786;
x760[x776] = x787;

}

}

}

}
float* x797 = (float*)myMalloc(x759 * sizeof(float));;
for(int x799=0; x799 < x759; x799++) {
float x800 = x760[x799];
bool x801 = x800 < 0.0f;
if (x801) {
x797[x799] = 0.0f;
} else {
float x804 = x760[x799];
x797[x799] = x804;
}

}
if (x811) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(64) x Sym(749) x Sym(751) x Sym(751)|(2,2)");
}
int32_t x822 = 64 * x749;
int32_t x823 = x822 * x818;
int32_t x824 = x823 * x818;
float* x825 = (float*)myMalloc(x824 * sizeof(float));;
for(int x827=0; x827 < x824; x827++) {
x825[x827] = -3.4028235E38f;

}
int32_t x820 = x749 * x819;
int32_t x821 = 64 * x820;
int* x831 = (int32_t*)myMalloc(x821 * sizeof(int32_t));;
for(int x832=0; x832 < 64; x832++) {
int32_t x833 = x832 * x758;
float* x834 = x797+x833;
int32_t x835 = x832 * x820;
float* x836 = x825+x835;
int* x837 = x831+x835;
int32_t x838 = 0;
int32_t x839 = 0;
for(int x840=0; x840 < x749; x840++) {
int32_t x841 = x838;
int32_t x842 = x841;
int32_t x843 = x839;
int32_t x844 = x843;
for(int x846=0; x846 < x818; x846++) {
int32_t x847 = x842;
int32_t x848 = x847;
int32_t x849 = x844;
int32_t x850 = x849;
for(int x851=0; x851 < x818; x851++) {
int32_t x852 = x850;
int32_t x853 = x852;
int32_t x854 = x853;
int32_t x855 = x854;
int32_t x856 = x855;
float x857 = x834[x856];
int32_t x858 = x848;
float x859 = x836[x858];
bool x860 = x857 > x859;
if (x860) {
float x861 = x834[x856];
x836[x858] = x861;
int32_t x863 = x856 + x833;
x837[x858] = x863;
} else {
}
x855 += 1;
int32_t x868 = x855;
float x869 = x834[x868];
float x870 = x836[x858];
bool x871 = x869 > x870;
if (x871) {
float x872 = x834[x868];
x836[x858] = x872;
int32_t x874 = x868 + x833;
x837[x858] = x874;
} else {
}
x855 += 1;
x853 += x751;
int32_t x880 = x853;
int32_t x881 = x880;
int32_t x882 = x881;
float x883 = x834[x882];
float x884 = x836[x858];
bool x885 = x883 > x884;
if (x885) {
float x886 = x834[x882];
x836[x858] = x886;
int32_t x888 = x882 + x833;
x837[x858] = x888;
} else {
}
x881 += 1;
int32_t x893 = x881;
float x894 = x834[x893];
float x895 = x836[x858];
bool x896 = x894 > x895;
if (x896) {
float x897 = x834[x893];
x836[x858] = x897;
int32_t x899 = x893 + x833;
x837[x858] = x899;
} else {
}
x881 += 1;
x853 += x751;
x848 += 1;
x850 += 2;

}
x842 += x818;
x844 += x910;

}
x838 += x819;
x839 += x757;

}

}
float* x927 = (float*)myMalloc(x926 * sizeof(float));;
int32_t x930 = x822 * x922;
float* x931 = (float*)myMalloc(x930 * sizeof(float));;
int32_t x928 = x749 * x922;
for(int x932=0; x932 < 64; x932++) {
int32_t x933 = x932 * x820;
float* x934 = x825+x933;
int32_t x935 = x932 * x923;
float* x936 = x927+x935;
int32_t x937 = x932 * x928;
float* x938 = x931+x937;
for(int x939=0; x939 < x749; x939++) {
int32_t x940 = x939 / 1;
int32_t x944 = x940 * x921;
int32_t x945 = x944 * x921;
int32_t x941 = x939 % 1;
int32_t x942 = x941 / 1;
int32_t x946 = x942 * x921;
int32_t x947 = x946 * x921;
int32_t x948 = x945 + x947;
int32_t x943 = x941 % 1;
int32_t x949 = x943 * x921;
int32_t x950 = x949 * x921;
int32_t x951 = x948 + x950;
float* x952 = x938+x951;
int32_t x953 = x940 * x818;
int32_t x954 = x953 * x818;
float* x955 = x934+x954;
for(int x957=0; x957 < x921; x957++) {
int32_t x959 = x957 * x921;
float* x960 = x952+x959;
int32_t x958 = x957 + x942;
int32_t x961 = x958 * x818;
int32_t x962 = x961 + x943;
float* x963 = x955+x962;
memcpy(x960, x963, 4 * x921);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x922,x749,1,x233,x749,x938,x922,1,x936,x922);

}
int32_t x972 = 0;
int32_t x973 = 1;
x973 *= 1;
x972 += 1;
x973 *= 1;
x973 *= 1;
int32_t x978 = x972;
bool x979 = x978 >= 2;
if (x979) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x984 = x978 == 0;
if (x984) {
int32_t x985 = x973;
bool x986 = x985 == 64;
if (x986) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x993 = x973;
int32_t x994 = 64 / x993;
bool x995 = x994 == 1;
bool x998;
if (x454) {
bool x996 = 64 == x994;
bool x997 = x995 || x996;
x998 = x997;
} else {
x998 = false;
}
bool x1002;
if (x998) {
x1002 = x1001;
} else {
x1002 = false;
}
bool x1003;
if (x1002) {
x1003 = x1001;
} else {
x1003 = false;
}
if (x1003) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x921,x921,1,x994,1,1);
assert(false && "");
}
bool x1009 = 64 <= x994;
int32_t x1010;
if (x1009) {
x1010 = x994;
} else {
x1010 = 64;
}
int32_t x1019 = x1010 * x1018;
int32_t x1020 = 64 * x1019;
float* x1021 = (float*)myMalloc(x1020 * sizeof(float));;
int32_t x1024;
if (x995) {
x1024 = 0;
} else {
x1024 = 1;
}
for(int x1025=0; x1025 < 64; x1025++) {
int32_t x1037 = x923 * x1025;
int32_t x1031 = x1019 * x1025;
for(int x1027=0; x1027 < x1010; x1027++) {
int32_t x1038 = x922 * x1027;
int32_t x1039 = x1037 + x1038;
int32_t x1044 = x1024 * x1027;
int32_t x1033 = x1018 * x1027;
for(int x1029=0; x1029 < x1012; x1029++) {
int32_t x1040 = x1022 * x1029;
int32_t x1041 = x1039 + x1040;
int32_t x1035 = x1012 * x1029;
for(int x1030=0; x1030 < x1012; x1030++) {
int32_t x1042 = x1023 * x1030;
int32_t x1043 = x1041 + x1042;
float x1045 = x927[x1043];
float x1046 = x114[x1044];
int32_t x1032 = x1030 + x1031;
int32_t x1034 = x1032 + x1033;
int32_t x1036 = x1034 + x1035;
float x1047 = x1045 - x1046;
x1021[x1036] = x1047;

}

}

}

}
float* x1057 = (float*)myMalloc(64 * sizeof(float));;
for(int x1058=0; x1058 < 64; x1058++) {
float x1059 = x51[x1058];
float x1060 = x1059 + 1.0E-5f;
x1057[x1058] = x1060;

}
float* x1064 = (float*)myMalloc(64 * sizeof(float));;
for(int x1065=0; x1065 < 64; x1065++) {
float x1066 = x1057[x1065];
double x1067 = (double)x1066;
double x1068 = sqrt(x1067);
float x1069 = (float)x1068;
x1064[x1065] = x1069;

}
int32_t x1073 = 0;
int32_t x1074 = 1;
x1074 *= 1;
x1073 += 1;
x1074 *= 1;
x1074 *= 1;
int32_t x1079 = x1073;
bool x1080 = x1079 >= 2;
if (x1080) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1085 = x1079 == 0;
if (x1085) {
int32_t x1086 = x1074;
bool x1087 = x1086 == 64;
if (x1087) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1094 = x1074;
bool x1096 = x1010 == 1;
int32_t x1095 = 64 / x1094;
bool x1097 = x1095 == 1;
bool x1101;
if (x454) {
bool x1098 = x1096 || x1097;
bool x1099 = x1010 == x1095;
bool x1100 = x1098 || x1099;
x1101 = x1100;
} else {
x1101 = false;
}
bool x1105;
if (x1101) {
x1105 = x1104;
} else {
x1105 = false;
}
bool x1106;
if (x1105) {
x1106 = x1104;
} else {
x1106 = false;
}
if (x1106) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1010,x1012,x1012,1,x1095,1,1);
assert(false && "");
}
bool x1112 = x1010 <= x1095;
int32_t x1113;
if (x1112) {
x1113 = x1095;
} else {
x1113 = x1010;
}
int32_t x1122 = x1113 * x1121;
int32_t x1123 = 64 * x1122;
float* x1124 = (float*)myMalloc(x1123 * sizeof(float));;
int32_t x1125;
if (x1096) {
x1125 = 0;
} else {
x1125 = x1018;
}
int32_t x1128;
if (x1097) {
x1128 = 0;
} else {
x1128 = 1;
}
for(int x1129=0; x1129 < 64; x1129++) {
int32_t x1141 = x1019 * x1129;
int32_t x1135 = x1122 * x1129;
for(int x1131=0; x1131 < x1113; x1131++) {
int32_t x1142 = x1125 * x1131;
int32_t x1143 = x1141 + x1142;
int32_t x1148 = x1128 * x1131;
int32_t x1137 = x1121 * x1131;
for(int x1133=0; x1133 < x1115; x1133++) {
int32_t x1144 = x1126 * x1133;
int32_t x1145 = x1143 + x1144;
int32_t x1139 = x1115 * x1133;
for(int x1134=0; x1134 < x1115; x1134++) {
int32_t x1146 = x1127 * x1134;
int32_t x1147 = x1145 + x1146;
float x1149 = x1021[x1147];
float x1150 = x1064[x1148];
int32_t x1136 = x1134 + x1135;
int32_t x1138 = x1136 + x1137;
int32_t x1140 = x1138 + x1139;
float x1151 = x1149 / x1150;
x1124[x1140] = x1151;

}

}

}

}
int32_t x1161 = 0;
int32_t x1162 = 1;
x1162 *= 1;
x1161 += 1;
x1162 *= 1;
x1162 *= 1;
int32_t x1167 = x1161;
bool x1168 = x1167 >= 2;
if (x1168) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1173 = x1167 == 0;
if (x1173) {
int32_t x1174 = x1162;
bool x1175 = x1174 == 64;
if (x1175) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1182 = x1162;
bool x1184 = x1113 == 1;
int32_t x1183 = 64 / x1182;
bool x1185 = x1183 == 1;
bool x1189;
if (x454) {
bool x1186 = x1184 || x1185;
bool x1187 = x1113 == x1183;
bool x1188 = x1186 || x1187;
x1189 = x1188;
} else {
x1189 = false;
}
bool x1193;
if (x1189) {
x1193 = x1192;
} else {
x1193 = false;
}
bool x1194;
if (x1193) {
x1194 = x1192;
} else {
x1194 = false;
}
if (x1194) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1113,x1115,x1115,1,x1183,1,1);
assert(false && "");
}
bool x1200 = x1113 <= x1183;
int32_t x1201;
if (x1200) {
x1201 = x1183;
} else {
x1201 = x1113;
}
int32_t x1210 = x1201 * x1209;
int32_t x1211 = 64 * x1210;
float* x1212 = (float*)myMalloc(x1211 * sizeof(float));;
int32_t x1213;
if (x1184) {
x1213 = 0;
} else {
x1213 = x1121;
}
int32_t x1216;
if (x1185) {
x1216 = 0;
} else {
x1216 = 1;
}
for(int x1217=0; x1217 < 64; x1217++) {
int32_t x1229 = x1122 * x1217;
int32_t x1223 = x1210 * x1217;
for(int x1219=0; x1219 < x1201; x1219++) {
int32_t x1230 = x1213 * x1219;
int32_t x1231 = x1229 + x1230;
int32_t x1236 = x1216 * x1219;
int32_t x1225 = x1209 * x1219;
for(int x1221=0; x1221 < x1203; x1221++) {
int32_t x1232 = x1214 * x1221;
int32_t x1233 = x1231 + x1232;
int32_t x1227 = x1203 * x1221;
for(int x1222=0; x1222 < x1203; x1222++) {
int32_t x1234 = x1215 * x1222;
int32_t x1235 = x1233 + x1234;
float x1237 = x1124[x1235];
float x1238 = x26[x1236];
int32_t x1224 = x1222 + x1223;
int32_t x1226 = x1224 + x1225;
int32_t x1228 = x1226 + x1227;
float x1239 = x1237 * x1238;
x1212[x1228] = x1239;

}

}

}

}
int32_t x1249 = 0;
int32_t x1250 = 1;
x1250 *= 1;
x1249 += 1;
x1250 *= 1;
x1250 *= 1;
int32_t x1255 = x1249;
bool x1256 = x1255 >= 2;
if (x1256) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1261 = x1255 == 0;
if (x1261) {
int32_t x1262 = x1250;
bool x1263 = x1262 == 64;
if (x1263) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1270 = x1250;
bool x1272 = x1201 == 1;
int32_t x1271 = 64 / x1270;
bool x1273 = x1271 == 1;
bool x1277;
if (x454) {
bool x1274 = x1272 || x1273;
bool x1275 = x1201 == x1271;
bool x1276 = x1274 || x1275;
x1277 = x1276;
} else {
x1277 = false;
}
bool x1281;
if (x1277) {
x1281 = x1280;
} else {
x1281 = false;
}
bool x1282;
if (x1281) {
x1282 = x1280;
} else {
x1282 = false;
}
if (x1282) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1201,x1203,x1203,1,x1271,1,1);
assert(false && "");
}
bool x1288 = x1201 <= x1271;
int32_t x1289;
if (x1288) {
x1289 = x1271;
} else {
x1289 = x1201;
}
int32_t x1298 = x1289 * x1297;
int32_t x1299 = 64 * x1298;
float* x1300 = (float*)myMalloc(x1299 * sizeof(float));;
int32_t x1301;
if (x1272) {
x1301 = 0;
} else {
x1301 = x1209;
}
int32_t x1304;
if (x1273) {
x1304 = 0;
} else {
x1304 = 1;
}
for(int x1305=0; x1305 < 64; x1305++) {
int32_t x1317 = x1210 * x1305;
int32_t x1311 = x1298 * x1305;
for(int x1307=0; x1307 < x1289; x1307++) {
int32_t x1318 = x1301 * x1307;
int32_t x1319 = x1317 + x1318;
int32_t x1324 = x1304 * x1307;
int32_t x1313 = x1297 * x1307;
for(int x1309=0; x1309 < x1291; x1309++) {
int32_t x1320 = x1302 * x1309;
int32_t x1321 = x1319 + x1320;
int32_t x1315 = x1291 * x1309;
for(int x1310=0; x1310 < x1291; x1310++) {
int32_t x1322 = x1303 * x1310;
int32_t x1323 = x1321 + x1322;
float x1325 = x1212[x1323];
float x1326 = x53[x1324];
int32_t x1312 = x1310 + x1311;
int32_t x1314 = x1312 + x1313;
int32_t x1316 = x1314 + x1315;
float x1327 = x1325 + x1326;
x1300[x1316] = x1327;

}

}

}

}
float* x1337 = (float*)myMalloc(x1299 * sizeof(float));;
for(int x1339=0; x1339 < x1299; x1339++) {
float x1340 = x1300[x1339];
bool x1341 = x1340 < 0.0f;
if (x1341) {
x1337[x1339] = 0.0f;
} else {
float x1344 = x1300[x1339];
x1337[x1339] = x1344;
}

}
float* x1359 = (float*)myMalloc(x1358 * sizeof(float));;
int32_t x1360 = 9 * x1289;
int32_t x1363 = 64 * x1360;
int32_t x1364 = x1363 * x1354;
float* x1365 = (float*)myMalloc(x1364 * sizeof(float));;
int32_t x1361 = x1360 * x1354;
int32_t x1373 = x1289 * 3;
int32_t x1374 = x1373 * 3;
for(int x1366=0; x1366 < 64; x1366++) {
int32_t x1367 = x1366 * x1298;
float* x1368 = x1337+x1367;
int32_t x1369 = x1366 * x1355;
float* x1370 = x1359+x1369;
int32_t x1371 = x1366 * x1361;
float* x1372 = x1365+x1371;
for(int x1376=0; x1376 < x1374; x1376++) {
int32_t x1377 = x1376 / 9;
int32_t x1381 = x1377 * 3;
int32_t x1382 = x1381 * 3;
int32_t x1383 = x1382 * x1353;
int32_t x1384 = x1383 * x1353;
int32_t x1378 = x1376 % 9;
int32_t x1379 = x1378 / 3;
int32_t x1385 = x1379 * 3;
int32_t x1386 = x1385 * x1353;
int32_t x1387 = x1386 * x1353;
int32_t x1388 = x1384 + x1387;
int32_t x1380 = x1378 % 3;
int32_t x1389 = x1380 * x1353;
int32_t x1390 = x1389 * x1353;
int32_t x1391 = x1388 + x1390;
float* x1392 = x1372+x1391;
int32_t x1393 = x1377 * x1291;
int32_t x1394 = x1393 * x1291;
float* x1395 = x1368+x1394;
int32_t x1408 = 1 - x1380;
bool x1409 = x1408 > 0;
int32_t x1410;
if (x1409) {
x1410 = x1408;
} else {
x1410 = 0;
}
int32_t x1411 = 3 - x1380;
int32_t x1412 = x1411 - 1;
int32_t x1413 = 1 - x1412;
bool x1414 = x1413 > 0;
int32_t x1415;
if (x1414) {
x1415 = x1413;
} else {
x1415 = 0;
}
int32_t x1416 = x1353 - x1415;
int32_t x1417 = x1416 - x1410;
bool x1418 = x1417 <= 0;
bool x1422 = x1410 > 0;
int32_t x1407 = -1 + x1380;
bool x1435 = x1415 > 0;
for(int x1397=0; x1397 < x1353; x1397++) {
int32_t x1398 = x1397 - 1;
int32_t x1399 = x1398 + x1379;
bool x1400 = x1399 < 0;
bool x1401 = x1399 >= x1291;
bool x1402 = x1400 || x1401;
if (x1402) {
int32_t x1403 = x1397 * x1353;
float* x1404 = x1392+x1403;
memset(x1404, 0, 4 * x1353);;
} else {
if (x1418) {
int32_t x1403 = x1397 * x1353;
float* x1419 = x1392+x1403;
memset(x1419, 0, 4 * x1353);;
} else {
int32_t x1403 = x1397 * x1353;
if (x1422) {
float* x1423 = x1392+x1403;
memset(x1423, 0, 4 * x1410);;
} else {
}
// may have segfault here
int32_t x1428 = x1403 + x1410;
float* x1429 = x1392+x1428;
int32_t x1430 = x1399 * x1291;
int32_t x1431 = x1430 + x1407;
int32_t x1432 = x1431 + x1410;
float* x1433 = x1395+x1432;
memcpy(x1429, x1433, 4 * x1417);;
if (x1435) {
int32_t x1436 = x1403 + x1353;
int32_t x1437 = x1436 - x1415;
float* x1438 = x1392+x1437;
memset(x1438, 0, 4 * x1415);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1354,x1360,1,x90,x1360,x1372,x1354,1,x1370,x1354);

}
int32_t x1453 = 0;
int32_t x1454 = 1;
x1454 *= 1;
x1453 += 1;
x1454 *= 1;
x1454 *= 1;
int32_t x1459 = x1453;
bool x1460 = x1459 >= 2;
if (x1460) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1465 = x1459 == 0;
if (x1465) {
int32_t x1466 = x1454;
bool x1467 = x1466 == 64;
if (x1467) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1474 = x1454;
int32_t x1475 = 64 / x1474;
bool x1476 = x1475 == 1;
bool x1479;
if (x454) {
bool x1477 = 64 == x1475;
bool x1478 = x1476 || x1477;
x1479 = x1478;
} else {
x1479 = false;
}
bool x1483;
if (x1479) {
x1483 = x1482;
} else {
x1483 = false;
}
bool x1484;
if (x1483) {
x1484 = x1482;
} else {
x1484 = false;
}
if (x1484) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x1353,x1353,1,x1475,1,1);
assert(false && "");
}
bool x1490 = 64 <= x1475;
int32_t x1491;
if (x1490) {
x1491 = x1475;
} else {
x1491 = 64;
}
int32_t x1500 = x1491 * x1499;
int32_t x1501 = 64 * x1500;
float* x1502 = (float*)myMalloc(x1501 * sizeof(float));;
int32_t x1505;
if (x1476) {
x1505 = 0;
} else {
x1505 = 1;
}
for(int x1506=0; x1506 < 64; x1506++) {
int32_t x1518 = x1355 * x1506;
int32_t x1512 = x1500 * x1506;
for(int x1508=0; x1508 < x1491; x1508++) {
int32_t x1519 = x1354 * x1508;
int32_t x1520 = x1518 + x1519;
int32_t x1525 = x1505 * x1508;
int32_t x1514 = x1499 * x1508;
for(int x1510=0; x1510 < x1493; x1510++) {
int32_t x1521 = x1503 * x1510;
int32_t x1522 = x1520 + x1521;
int32_t x1516 = x1493 * x1510;
for(int x1511=0; x1511 < x1493; x1511++) {
int32_t x1523 = x1504 * x1511;
int32_t x1524 = x1522 + x1523;
float x1526 = x1359[x1524];
float x1527 = x105[x1525];
int32_t x1513 = x1511 + x1512;
int32_t x1515 = x1513 + x1514;
int32_t x1517 = x1515 + x1516;
float x1528 = x1526 - x1527;
x1502[x1517] = x1528;

}

}

}

}
float* x1538 = (float*)myMalloc(64 * sizeof(float));;
for(int x1539=0; x1539 < 64; x1539++) {
float x1540 = x158[x1539];
float x1541 = x1540 + 1.0E-5f;
x1538[x1539] = x1541;

}
float* x1545 = (float*)myMalloc(64 * sizeof(float));;
for(int x1546=0; x1546 < 64; x1546++) {
float x1547 = x1538[x1546];
double x1548 = (double)x1547;
double x1549 = sqrt(x1548);
float x1550 = (float)x1549;
x1545[x1546] = x1550;

}
int32_t x1554 = 0;
int32_t x1555 = 1;
x1555 *= 1;
x1554 += 1;
x1555 *= 1;
x1555 *= 1;
int32_t x1560 = x1554;
bool x1561 = x1560 >= 2;
if (x1561) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1566 = x1560 == 0;
if (x1566) {
int32_t x1567 = x1555;
bool x1568 = x1567 == 64;
if (x1568) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1575 = x1555;
bool x1577 = x1491 == 1;
int32_t x1576 = 64 / x1575;
bool x1578 = x1576 == 1;
bool x1582;
if (x454) {
bool x1579 = x1577 || x1578;
bool x1580 = x1491 == x1576;
bool x1581 = x1579 || x1580;
x1582 = x1581;
} else {
x1582 = false;
}
bool x1586;
if (x1582) {
x1586 = x1585;
} else {
x1586 = false;
}
bool x1587;
if (x1586) {
x1587 = x1585;
} else {
x1587 = false;
}
if (x1587) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1491,x1493,x1493,1,x1576,1,1);
assert(false && "");
}
bool x1593 = x1491 <= x1576;
int32_t x1594;
if (x1593) {
x1594 = x1576;
} else {
x1594 = x1491;
}
int32_t x1603 = x1594 * x1602;
int32_t x1604 = 64 * x1603;
float* x1605 = (float*)myMalloc(x1604 * sizeof(float));;
int32_t x1606;
if (x1577) {
x1606 = 0;
} else {
x1606 = x1499;
}
int32_t x1609;
if (x1578) {
x1609 = 0;
} else {
x1609 = 1;
}
for(int x1610=0; x1610 < 64; x1610++) {
int32_t x1622 = x1500 * x1610;
int32_t x1616 = x1603 * x1610;
for(int x1612=0; x1612 < x1594; x1612++) {
int32_t x1623 = x1606 * x1612;
int32_t x1624 = x1622 + x1623;
int32_t x1629 = x1609 * x1612;
int32_t x1618 = x1602 * x1612;
for(int x1614=0; x1614 < x1596; x1614++) {
int32_t x1625 = x1607 * x1614;
int32_t x1626 = x1624 + x1625;
int32_t x1620 = x1596 * x1614;
for(int x1615=0; x1615 < x1596; x1615++) {
int32_t x1627 = x1608 * x1615;
int32_t x1628 = x1626 + x1627;
float x1630 = x1502[x1628];
float x1631 = x1545[x1629];
int32_t x1617 = x1615 + x1616;
int32_t x1619 = x1617 + x1618;
int32_t x1621 = x1619 + x1620;
float x1632 = x1630 / x1631;
x1605[x1621] = x1632;

}

}

}

}
int32_t x1642 = 0;
int32_t x1643 = 1;
x1643 *= 1;
x1642 += 1;
x1643 *= 1;
x1643 *= 1;
int32_t x1648 = x1642;
bool x1649 = x1648 >= 2;
if (x1649) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1654 = x1648 == 0;
if (x1654) {
int32_t x1655 = x1643;
bool x1656 = x1655 == 64;
if (x1656) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1663 = x1643;
bool x1665 = x1594 == 1;
int32_t x1664 = 64 / x1663;
bool x1666 = x1664 == 1;
bool x1670;
if (x454) {
bool x1667 = x1665 || x1666;
bool x1668 = x1594 == x1664;
bool x1669 = x1667 || x1668;
x1670 = x1669;
} else {
x1670 = false;
}
bool x1674;
if (x1670) {
x1674 = x1673;
} else {
x1674 = false;
}
bool x1675;
if (x1674) {
x1675 = x1673;
} else {
x1675 = false;
}
if (x1675) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1594,x1596,x1596,1,x1664,1,1);
assert(false && "");
}
bool x1681 = x1594 <= x1664;
int32_t x1682;
if (x1681) {
x1682 = x1664;
} else {
x1682 = x1594;
}
int32_t x1691 = x1682 * x1690;
int32_t x1692 = 64 * x1691;
float* x1693 = (float*)myMalloc(x1692 * sizeof(float));;
int32_t x1694;
if (x1665) {
x1694 = 0;
} else {
x1694 = x1602;
}
int32_t x1697;
if (x1666) {
x1697 = 0;
} else {
x1697 = 1;
}
for(int x1698=0; x1698 < 64; x1698++) {
int32_t x1710 = x1603 * x1698;
int32_t x1704 = x1691 * x1698;
for(int x1700=0; x1700 < x1682; x1700++) {
int32_t x1711 = x1694 * x1700;
int32_t x1712 = x1710 + x1711;
int32_t x1717 = x1697 * x1700;
int32_t x1706 = x1690 * x1700;
for(int x1702=0; x1702 < x1684; x1702++) {
int32_t x1713 = x1695 * x1702;
int32_t x1714 = x1712 + x1713;
int32_t x1708 = x1684 * x1702;
for(int x1703=0; x1703 < x1684; x1703++) {
int32_t x1715 = x1696 * x1703;
int32_t x1716 = x1714 + x1715;
float x1718 = x1605[x1716];
float x1719 = x164[x1717];
int32_t x1705 = x1703 + x1704;
int32_t x1707 = x1705 + x1706;
int32_t x1709 = x1707 + x1708;
float x1720 = x1718 * x1719;
x1693[x1709] = x1720;

}

}

}

}
int32_t x1730 = 0;
int32_t x1731 = 1;
x1731 *= 1;
x1730 += 1;
x1731 *= 1;
x1731 *= 1;
int32_t x1736 = x1730;
bool x1737 = x1736 >= 2;
if (x1737) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1742 = x1736 == 0;
if (x1742) {
int32_t x1743 = x1731;
bool x1744 = x1743 == 64;
if (x1744) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1751 = x1731;
bool x1753 = x1682 == 1;
int32_t x1752 = 64 / x1751;
bool x1754 = x1752 == 1;
bool x1758;
if (x454) {
bool x1755 = x1753 || x1754;
bool x1756 = x1682 == x1752;
bool x1757 = x1755 || x1756;
x1758 = x1757;
} else {
x1758 = false;
}
bool x1762;
if (x1758) {
x1762 = x1761;
} else {
x1762 = false;
}
bool x1763;
if (x1762) {
x1763 = x1761;
} else {
x1763 = false;
}
if (x1763) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1682,x1684,x1684,1,x1752,1,1);
assert(false && "");
}
bool x1769 = x1682 <= x1752;
int32_t x1770;
if (x1769) {
x1770 = x1752;
} else {
x1770 = x1682;
}
int32_t x1779 = x1770 * x1778;
int32_t x1780 = 64 * x1779;
float* x1781 = (float*)myMalloc(x1780 * sizeof(float));;
int32_t x1782;
if (x1753) {
x1782 = 0;
} else {
x1782 = x1690;
}
int32_t x1785;
if (x1754) {
x1785 = 0;
} else {
x1785 = 1;
}
for(int x1786=0; x1786 < 64; x1786++) {
int32_t x1798 = x1691 * x1786;
int32_t x1792 = x1779 * x1786;
for(int x1788=0; x1788 < x1770; x1788++) {
int32_t x1799 = x1782 * x1788;
int32_t x1800 = x1798 + x1799;
int32_t x1805 = x1785 * x1788;
int32_t x1794 = x1778 * x1788;
for(int x1790=0; x1790 < x1772; x1790++) {
int32_t x1801 = x1783 * x1790;
int32_t x1802 = x1800 + x1801;
int32_t x1796 = x1772 * x1790;
for(int x1791=0; x1791 < x1772; x1791++) {
int32_t x1803 = x1784 * x1791;
int32_t x1804 = x1802 + x1803;
float x1806 = x1693[x1804];
float x1807 = x49[x1805];
int32_t x1793 = x1791 + x1792;
int32_t x1795 = x1793 + x1794;
int32_t x1797 = x1795 + x1796;
float x1808 = x1806 + x1807;
x1781[x1797] = x1808;

}

}

}

}
float* x1818 = (float*)myMalloc(x1780 * sizeof(float));;
for(int x1820=0; x1820 < x1780; x1820++) {
float x1821 = x1781[x1820];
bool x1822 = x1821 < 0.0f;
if (x1822) {
x1818[x1820] = 0.0f;
} else {
float x1825 = x1781[x1820];
x1818[x1820] = x1825;
}

}
float* x1839 = (float*)myMalloc(x1838 * sizeof(float));;
int32_t x1842 = 64 * x1770;
int32_t x1843 = x1842 * x1834;
float* x1844 = (float*)myMalloc(x1843 * sizeof(float));;
int32_t x1840 = x1770 * x1834;
for(int x1845=0; x1845 < 64; x1845++) {
int32_t x1846 = x1845 * x1779;
float* x1847 = x1818+x1846;
int32_t x1848 = x1845 * x1835;
float* x1849 = x1839+x1848;
int32_t x1850 = x1845 * x1840;
float* x1851 = x1844+x1850;
for(int x1852=0; x1852 < x1770; x1852++) {
int32_t x1853 = x1852 / 1;
int32_t x1857 = x1853 * x1833;
int32_t x1858 = x1857 * x1833;
int32_t x1854 = x1852 % 1;
int32_t x1855 = x1854 / 1;
int32_t x1859 = x1855 * x1833;
int32_t x1860 = x1859 * x1833;
int32_t x1861 = x1858 + x1860;
int32_t x1856 = x1854 % 1;
int32_t x1862 = x1856 * x1833;
int32_t x1863 = x1862 * x1833;
int32_t x1864 = x1861 + x1863;
float* x1865 = x1851+x1864;
int32_t x1866 = x1853 * x1772;
int32_t x1867 = x1866 * x1772;
float* x1868 = x1847+x1867;
for(int x1870=0; x1870 < x1833; x1870++) {
int32_t x1872 = x1870 * x1833;
float* x1873 = x1865+x1872;
int32_t x1871 = x1870 + x1855;
int32_t x1874 = x1871 * x1772;
int32_t x1875 = x1874 + x1856;
float* x1876 = x1868+x1875;
memcpy(x1873, x1876, 4 * x1833);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1834,x1770,1,x32,x1770,x1851,x1834,1,x1849,x1834);

}
int32_t x1885 = 0;
int32_t x1886 = 1;
x1886 *= 1;
x1885 += 1;
x1886 *= 1;
x1886 *= 1;
int32_t x1891 = x1885;
bool x1892 = x1891 >= 2;
if (x1892) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1897 = x1891 == 0;
if (x1897) {
int32_t x1898 = x1886;
bool x1899 = x1898 == 256;
if (x1899) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1906 = x1886;
int32_t x1907 = 256 / x1906;
bool x1908 = x1907 == 1;
bool x1911;
if (x454) {
bool x1909 = 256 == x1907;
bool x1910 = x1908 || x1909;
x1911 = x1910;
} else {
x1911 = false;
}
bool x1915;
if (x1911) {
x1915 = x1914;
} else {
x1915 = false;
}
bool x1916;
if (x1915) {
x1916 = x1914;
} else {
x1916 = false;
}
if (x1916) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x1833,x1833,1,x1907,1,1);
assert(false && "");
}
bool x1922 = 256 <= x1907;
int32_t x1923;
if (x1922) {
x1923 = x1907;
} else {
x1923 = 256;
}
int32_t x1932 = x1923 * x1931;
int32_t x1933 = 64 * x1932;
float* x1934 = (float*)myMalloc(x1933 * sizeof(float));;
int32_t x1937;
if (x1908) {
x1937 = 0;
} else {
x1937 = 1;
}
for(int x1938=0; x1938 < 64; x1938++) {
int32_t x1950 = x1835 * x1938;
int32_t x1944 = x1932 * x1938;
for(int x1940=0; x1940 < x1923; x1940++) {
int32_t x1951 = x1834 * x1940;
int32_t x1952 = x1950 + x1951;
int32_t x1957 = x1937 * x1940;
int32_t x1946 = x1931 * x1940;
for(int x1942=0; x1942 < x1925; x1942++) {
int32_t x1953 = x1935 * x1942;
int32_t x1954 = x1952 + x1953;
int32_t x1948 = x1925 * x1942;
for(int x1943=0; x1943 < x1925; x1943++) {
int32_t x1955 = x1936 * x1943;
int32_t x1956 = x1954 + x1955;
float x1958 = x1839[x1956];
float x1959 = x71[x1957];
int32_t x1945 = x1943 + x1944;
int32_t x1947 = x1945 + x1946;
int32_t x1949 = x1947 + x1948;
float x1960 = x1958 - x1959;
x1934[x1949] = x1960;

}

}

}

}
float* x1970 = (float*)myMalloc(256 * sizeof(float));;
for(int x1972=0; x1972 < 256; x1972++) {
float x1973 = x36[x1972];
float x1974 = x1973 + 1.0E-5f;
x1970[x1972] = x1974;

}
float* x1978 = (float*)myMalloc(256 * sizeof(float));;
for(int x1979=0; x1979 < 256; x1979++) {
float x1980 = x1970[x1979];
double x1981 = (double)x1980;
double x1982 = sqrt(x1981);
float x1983 = (float)x1982;
x1978[x1979] = x1983;

}
int32_t x1987 = 0;
int32_t x1988 = 1;
x1988 *= 1;
x1987 += 1;
x1988 *= 1;
x1988 *= 1;
int32_t x1993 = x1987;
bool x1994 = x1993 >= 2;
if (x1994) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1999 = x1993 == 0;
if (x1999) {
int32_t x2000 = x1988;
bool x2001 = x2000 == 256;
if (x2001) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2008 = x1988;
bool x2010 = x1923 == 1;
int32_t x2009 = 256 / x2008;
bool x2011 = x2009 == 1;
bool x2015;
if (x454) {
bool x2012 = x2010 || x2011;
bool x2013 = x1923 == x2009;
bool x2014 = x2012 || x2013;
x2015 = x2014;
} else {
x2015 = false;
}
bool x2019;
if (x2015) {
x2019 = x2018;
} else {
x2019 = false;
}
bool x2020;
if (x2019) {
x2020 = x2018;
} else {
x2020 = false;
}
if (x2020) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1923,x1925,x1925,1,x2009,1,1);
assert(false && "");
}
bool x2026 = x1923 <= x2009;
int32_t x2027;
if (x2026) {
x2027 = x2009;
} else {
x2027 = x1923;
}
int32_t x2036 = x2027 * x2035;
int32_t x2037 = 64 * x2036;
float* x2038 = (float*)myMalloc(x2037 * sizeof(float));;
int32_t x2039;
if (x2010) {
x2039 = 0;
} else {
x2039 = x1931;
}
int32_t x2042;
if (x2011) {
x2042 = 0;
} else {
x2042 = 1;
}
for(int x2043=0; x2043 < 64; x2043++) {
int32_t x2055 = x1932 * x2043;
int32_t x2049 = x2036 * x2043;
for(int x2045=0; x2045 < x2027; x2045++) {
int32_t x2056 = x2039 * x2045;
int32_t x2057 = x2055 + x2056;
int32_t x2062 = x2042 * x2045;
int32_t x2051 = x2035 * x2045;
for(int x2047=0; x2047 < x2029; x2047++) {
int32_t x2058 = x2040 * x2047;
int32_t x2059 = x2057 + x2058;
int32_t x2053 = x2029 * x2047;
for(int x2048=0; x2048 < x2029; x2048++) {
int32_t x2060 = x2041 * x2048;
int32_t x2061 = x2059 + x2060;
float x2063 = x1934[x2061];
float x2064 = x1978[x2062];
int32_t x2050 = x2048 + x2049;
int32_t x2052 = x2050 + x2051;
int32_t x2054 = x2052 + x2053;
float x2065 = x2063 / x2064;
x2038[x2054] = x2065;

}

}

}

}
int32_t x2075 = 0;
int32_t x2076 = 1;
x2076 *= 1;
x2075 += 1;
x2076 *= 1;
x2076 *= 1;
int32_t x2081 = x2075;
bool x2082 = x2081 >= 2;
if (x2082) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2087 = x2081 == 0;
if (x2087) {
int32_t x2088 = x2076;
bool x2089 = x2088 == 256;
if (x2089) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2096 = x2076;
bool x2098 = x2027 == 1;
int32_t x2097 = 256 / x2096;
bool x2099 = x2097 == 1;
bool x2103;
if (x454) {
bool x2100 = x2098 || x2099;
bool x2101 = x2027 == x2097;
bool x2102 = x2100 || x2101;
x2103 = x2102;
} else {
x2103 = false;
}
bool x2107;
if (x2103) {
x2107 = x2106;
} else {
x2107 = false;
}
bool x2108;
if (x2107) {
x2108 = x2106;
} else {
x2108 = false;
}
if (x2108) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2027,x2029,x2029,1,x2097,1,1);
assert(false && "");
}
bool x2114 = x2027 <= x2097;
int32_t x2115;
if (x2114) {
x2115 = x2097;
} else {
x2115 = x2027;
}
int32_t x2124 = x2115 * x2123;
int32_t x2125 = 64 * x2124;
float* x2126 = (float*)myMalloc(x2125 * sizeof(float));;
int32_t x2127;
if (x2098) {
x2127 = 0;
} else {
x2127 = x2035;
}
int32_t x2130;
if (x2099) {
x2130 = 0;
} else {
x2130 = 1;
}
for(int x2131=0; x2131 < 64; x2131++) {
int32_t x2143 = x2036 * x2131;
int32_t x2137 = x2124 * x2131;
for(int x2133=0; x2133 < x2115; x2133++) {
int32_t x2144 = x2127 * x2133;
int32_t x2145 = x2143 + x2144;
int32_t x2150 = x2130 * x2133;
int32_t x2139 = x2123 * x2133;
for(int x2135=0; x2135 < x2117; x2135++) {
int32_t x2146 = x2128 * x2135;
int32_t x2147 = x2145 + x2146;
int32_t x2141 = x2117 * x2135;
for(int x2136=0; x2136 < x2117; x2136++) {
int32_t x2148 = x2129 * x2136;
int32_t x2149 = x2147 + x2148;
float x2151 = x2038[x2149];
float x2152 = x199[x2150];
int32_t x2138 = x2136 + x2137;
int32_t x2140 = x2138 + x2139;
int32_t x2142 = x2140 + x2141;
float x2153 = x2151 * x2152;
x2126[x2142] = x2153;

}

}

}

}
int32_t x2163 = 0;
int32_t x2164 = 1;
x2164 *= 1;
x2163 += 1;
x2164 *= 1;
x2164 *= 1;
int32_t x2169 = x2163;
bool x2170 = x2169 >= 2;
if (x2170) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2175 = x2169 == 0;
if (x2175) {
int32_t x2176 = x2164;
bool x2177 = x2176 == 256;
if (x2177) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2184 = x2164;
bool x2186 = x2115 == 1;
int32_t x2185 = 256 / x2184;
bool x2187 = x2185 == 1;
bool x2191;
if (x454) {
bool x2188 = x2186 || x2187;
bool x2189 = x2115 == x2185;
bool x2190 = x2188 || x2189;
x2191 = x2190;
} else {
x2191 = false;
}
bool x2195;
if (x2191) {
x2195 = x2194;
} else {
x2195 = false;
}
bool x2196;
if (x2195) {
x2196 = x2194;
} else {
x2196 = false;
}
if (x2196) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2115,x2117,x2117,1,x2185,1,1);
assert(false && "");
}
bool x2202 = x2115 <= x2185;
int32_t x2203;
if (x2202) {
x2203 = x2185;
} else {
x2203 = x2115;
}
int32_t x2212 = x2203 * x2211;
int32_t x2213 = 64 * x2212;
float* x2214 = (float*)myMalloc(x2213 * sizeof(float));;
int32_t x2215;
if (x2186) {
x2215 = 0;
} else {
x2215 = x2123;
}
int32_t x2218;
if (x2187) {
x2218 = 0;
} else {
x2218 = 1;
}
for(int x2219=0; x2219 < 64; x2219++) {
int32_t x2231 = x2124 * x2219;
int32_t x2225 = x2212 * x2219;
for(int x2221=0; x2221 < x2203; x2221++) {
int32_t x2232 = x2215 * x2221;
int32_t x2233 = x2231 + x2232;
int32_t x2238 = x2218 * x2221;
int32_t x2227 = x2211 * x2221;
for(int x2223=0; x2223 < x2205; x2223++) {
int32_t x2234 = x2216 * x2223;
int32_t x2235 = x2233 + x2234;
int32_t x2229 = x2205 * x2223;
for(int x2224=0; x2224 < x2205; x2224++) {
int32_t x2236 = x2217 * x2224;
int32_t x2237 = x2235 + x2236;
float x2239 = x2126[x2237];
float x2240 = x126[x2238];
int32_t x2226 = x2224 + x2225;
int32_t x2228 = x2226 + x2227;
int32_t x2230 = x2228 + x2229;
float x2241 = x2239 + x2240;
x2214[x2230] = x2241;

}

}

}

}
float* x2255 = (float*)myMalloc(x2254 * sizeof(float));;
float* x2256 = (float*)myMalloc(x930 * sizeof(float));;
for(int x2257=0; x2257 < 64; x2257++) {
int32_t x2258 = x2257 * x820;
float* x2259 = x825+x2258;
int32_t x2260 = x2257 * x2251;
float* x2261 = x2255+x2260;
int32_t x2262 = x2257 * x928;
float* x2263 = x2256+x2262;
for(int x2264=0; x2264 < x749; x2264++) {
int32_t x2265 = x2264 / 1;
int32_t x2269 = x2265 * x921;
int32_t x2270 = x2269 * x921;
int32_t x2266 = x2264 % 1;
int32_t x2267 = x2266 / 1;
int32_t x2271 = x2267 * x921;
int32_t x2272 = x2271 * x921;
int32_t x2273 = x2270 + x2272;
int32_t x2268 = x2266 % 1;
int32_t x2274 = x2268 * x921;
int32_t x2275 = x2274 * x921;
int32_t x2276 = x2273 + x2275;
float* x2277 = x2263+x2276;
int32_t x2278 = x2265 * x818;
int32_t x2279 = x2278 * x818;
float* x2280 = x2259+x2279;
for(int x2281=0; x2281 < x921; x2281++) {
int32_t x2283 = x2281 * x921;
float* x2284 = x2277+x2283;
int32_t x2282 = x2281 + x2267;
int32_t x2285 = x2282 * x818;
int32_t x2286 = x2285 + x2268;
float* x2287 = x2280+x2286;
memcpy(x2284, x2287, 4 * x921);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x922,x749,1,x162,x749,x2263,x922,1,x2261,x922);

}
int32_t x2296 = 0;
int32_t x2297 = 1;
x2297 *= 1;
x2296 += 1;
x2297 *= 1;
x2297 *= 1;
int32_t x2302 = x2296;
bool x2303 = x2302 >= 2;
if (x2303) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2308 = x2302 == 0;
if (x2308) {
int32_t x2309 = x2297;
bool x2310 = x2309 == 256;
if (x2310) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2317 = x2297;
int32_t x2318 = 256 / x2317;
bool x2319 = x2318 == 1;
bool x2322;
if (x454) {
bool x2320 = 256 == x2318;
bool x2321 = x2319 || x2320;
x2322 = x2321;
} else {
x2322 = false;
}
bool x2323;
if (x2322) {
x2323 = x1001;
} else {
x2323 = false;
}
bool x2324;
if (x2323) {
x2324 = x1001;
} else {
x2324 = false;
}
if (x2324) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x921,x921,1,x2318,1,1);
assert(false && "");
}
bool x2330 = 256 <= x2318;
int32_t x2331;
if (x2330) {
x2331 = x2318;
} else {
x2331 = 256;
}
int32_t x2336 = x2331 * x1018;
int32_t x2337 = 64 * x2336;
float* x2338 = (float*)myMalloc(x2337 * sizeof(float));;
int32_t x2339;
if (x2319) {
x2339 = 0;
} else {
x2339 = 1;
}
for(int x2340=0; x2340 < 64; x2340++) {
int32_t x2351 = x2251 * x2340;
int32_t x2345 = x2336 * x2340;
for(int x2342=0; x2342 < x2331; x2342++) {
int32_t x2352 = x922 * x2342;
int32_t x2353 = x2351 + x2352;
int32_t x2358 = x2339 * x2342;
int32_t x2347 = x1018 * x2342;
for(int x2343=0; x2343 < x1012; x2343++) {
int32_t x2354 = x1022 * x2343;
int32_t x2355 = x2353 + x2354;
int32_t x2349 = x1012 * x2343;
for(int x2344=0; x2344 < x1012; x2344++) {
int32_t x2356 = x1023 * x2344;
int32_t x2357 = x2355 + x2356;
float x2359 = x2255[x2357];
float x2360 = x264[x2358];
int32_t x2346 = x2344 + x2345;
int32_t x2348 = x2346 + x2347;
int32_t x2350 = x2348 + x2349;
float x2361 = x2359 - x2360;
x2338[x2350] = x2361;

}

}

}

}
float* x2371 = (float*)myMalloc(256 * sizeof(float));;
for(int x2372=0; x2372 < 256; x2372++) {
float x2373 = x243[x2372];
float x2374 = x2373 + 1.0E-5f;
x2371[x2372] = x2374;

}
float* x2378 = (float*)myMalloc(256 * sizeof(float));;
for(int x2379=0; x2379 < 256; x2379++) {
float x2380 = x2371[x2379];
double x2381 = (double)x2380;
double x2382 = sqrt(x2381);
float x2383 = (float)x2382;
x2378[x2379] = x2383;

}
int32_t x2387 = 0;
int32_t x2388 = 1;
x2388 *= 1;
x2387 += 1;
x2388 *= 1;
x2388 *= 1;
int32_t x2393 = x2387;
bool x2394 = x2393 >= 2;
if (x2394) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2399 = x2393 == 0;
if (x2399) {
int32_t x2400 = x2388;
bool x2401 = x2400 == 256;
if (x2401) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2408 = x2388;
bool x2410 = x2331 == 1;
int32_t x2409 = 256 / x2408;
bool x2411 = x2409 == 1;
bool x2415;
if (x454) {
bool x2412 = x2410 || x2411;
bool x2413 = x2331 == x2409;
bool x2414 = x2412 || x2413;
x2415 = x2414;
} else {
x2415 = false;
}
bool x2416;
if (x2415) {
x2416 = x1104;
} else {
x2416 = false;
}
bool x2417;
if (x2416) {
x2417 = x1104;
} else {
x2417 = false;
}
if (x2417) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2331,x1012,x1012,1,x2409,1,1);
assert(false && "");
}
bool x2423 = x2331 <= x2409;
int32_t x2424;
if (x2423) {
x2424 = x2409;
} else {
x2424 = x2331;
}
int32_t x2429 = x2424 * x1121;
int32_t x2430 = 64 * x2429;
float* x2431 = (float*)myMalloc(x2430 * sizeof(float));;
int32_t x2432;
if (x2410) {
x2432 = 0;
} else {
x2432 = x1018;
}
int32_t x2433;
if (x2411) {
x2433 = 0;
} else {
x2433 = 1;
}
for(int x2434=0; x2434 < 64; x2434++) {
int32_t x2445 = x2336 * x2434;
int32_t x2439 = x2429 * x2434;
for(int x2436=0; x2436 < x2424; x2436++) {
int32_t x2446 = x2432 * x2436;
int32_t x2447 = x2445 + x2446;
int32_t x2452 = x2433 * x2436;
int32_t x2441 = x1121 * x2436;
for(int x2437=0; x2437 < x1115; x2437++) {
int32_t x2448 = x1126 * x2437;
int32_t x2449 = x2447 + x2448;
int32_t x2443 = x1115 * x2437;
for(int x2438=0; x2438 < x1115; x2438++) {
int32_t x2450 = x1127 * x2438;
int32_t x2451 = x2449 + x2450;
float x2453 = x2338[x2451];
float x2454 = x2378[x2452];
int32_t x2440 = x2438 + x2439;
int32_t x2442 = x2440 + x2441;
int32_t x2444 = x2442 + x2443;
float x2455 = x2453 / x2454;
x2431[x2444] = x2455;

}

}

}

}
int32_t x2465 = 0;
int32_t x2466 = 1;
x2466 *= 1;
x2465 += 1;
x2466 *= 1;
x2466 *= 1;
int32_t x2471 = x2465;
bool x2472 = x2471 >= 2;
if (x2472) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2477 = x2471 == 0;
if (x2477) {
int32_t x2478 = x2466;
bool x2479 = x2478 == 256;
if (x2479) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2486 = x2466;
bool x2488 = x2424 == 1;
int32_t x2487 = 256 / x2486;
bool x2489 = x2487 == 1;
bool x2493;
if (x454) {
bool x2490 = x2488 || x2489;
bool x2491 = x2424 == x2487;
bool x2492 = x2490 || x2491;
x2493 = x2492;
} else {
x2493 = false;
}
bool x2494;
if (x2493) {
x2494 = x1192;
} else {
x2494 = false;
}
bool x2495;
if (x2494) {
x2495 = x1192;
} else {
x2495 = false;
}
if (x2495) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2424,x1115,x1115,1,x2487,1,1);
assert(false && "");
}
bool x2501 = x2424 <= x2487;
int32_t x2502;
if (x2501) {
x2502 = x2487;
} else {
x2502 = x2424;
}
int32_t x2507 = x2502 * x1209;
int32_t x2508 = 64 * x2507;
float* x2509 = (float*)myMalloc(x2508 * sizeof(float));;
int32_t x2510;
if (x2488) {
x2510 = 0;
} else {
x2510 = x1121;
}
int32_t x2511;
if (x2489) {
x2511 = 0;
} else {
x2511 = 1;
}
for(int x2512=0; x2512 < 64; x2512++) {
int32_t x2523 = x2429 * x2512;
int32_t x2517 = x2507 * x2512;
for(int x2514=0; x2514 < x2502; x2514++) {
int32_t x2524 = x2510 * x2514;
int32_t x2525 = x2523 + x2524;
int32_t x2530 = x2511 * x2514;
int32_t x2519 = x1209 * x2514;
for(int x2515=0; x2515 < x1203; x2515++) {
int32_t x2526 = x1214 * x2515;
int32_t x2527 = x2525 + x2526;
int32_t x2521 = x1203 * x2515;
for(int x2516=0; x2516 < x1203; x2516++) {
int32_t x2528 = x1215 * x2516;
int32_t x2529 = x2527 + x2528;
float x2531 = x2431[x2529];
float x2532 = x76[x2530];
int32_t x2518 = x2516 + x2517;
int32_t x2520 = x2518 + x2519;
int32_t x2522 = x2520 + x2521;
float x2533 = x2531 * x2532;
x2509[x2522] = x2533;

}

}

}

}
int32_t x2543 = 0;
int32_t x2544 = 1;
x2544 *= 1;
x2543 += 1;
x2544 *= 1;
x2544 *= 1;
int32_t x2549 = x2543;
bool x2550 = x2549 >= 2;
if (x2550) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2555 = x2549 == 0;
if (x2555) {
int32_t x2556 = x2544;
bool x2557 = x2556 == 256;
if (x2557) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2564 = x2544;
bool x2566 = x2502 == 1;
int32_t x2565 = 256 / x2564;
bool x2567 = x2565 == 1;
bool x2571;
if (x454) {
bool x2568 = x2566 || x2567;
bool x2569 = x2502 == x2565;
bool x2570 = x2568 || x2569;
x2571 = x2570;
} else {
x2571 = false;
}
bool x2572;
if (x2571) {
x2572 = x1280;
} else {
x2572 = false;
}
bool x2573;
if (x2572) {
x2573 = x1280;
} else {
x2573 = false;
}
if (x2573) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2502,x1203,x1203,1,x2565,1,1);
assert(false && "");
}
bool x2579 = x2502 <= x2565;
int32_t x2580;
if (x2579) {
x2580 = x2565;
} else {
x2580 = x2502;
}
int32_t x2585 = x2580 * x1297;
int32_t x2586 = 64 * x2585;
float* x2587 = (float*)myMalloc(x2586 * sizeof(float));;
int32_t x2588;
if (x2566) {
x2588 = 0;
} else {
x2588 = x1209;
}
int32_t x2589;
if (x2567) {
x2589 = 0;
} else {
x2589 = 1;
}
for(int x2590=0; x2590 < 64; x2590++) {
int32_t x2601 = x2507 * x2590;
int32_t x2595 = x2585 * x2590;
for(int x2592=0; x2592 < x2580; x2592++) {
int32_t x2602 = x2588 * x2592;
int32_t x2603 = x2601 + x2602;
int32_t x2608 = x2589 * x2592;
int32_t x2597 = x1297 * x2592;
for(int x2593=0; x2593 < x1291; x2593++) {
int32_t x2604 = x1302 * x2593;
int32_t x2605 = x2603 + x2604;
int32_t x2599 = x1291 * x2593;
for(int x2594=0; x2594 < x1291; x2594++) {
int32_t x2606 = x1303 * x2594;
int32_t x2607 = x2605 + x2606;
float x2609 = x2509[x2607];
float x2610 = x203[x2608];
int32_t x2596 = x2594 + x2595;
int32_t x2598 = x2596 + x2597;
int32_t x2600 = x2598 + x2599;
float x2611 = x2609 + x2610;
x2587[x2600] = x2611;

}

}

}

}
bool x2621 = x2203 == 1;
bool x2622 = x2580 == 1;
bool x2623 = x2621 || x2622;
bool x2624 = x2203 == x2580;
bool x2625 = x2623 || x2624;
bool x2631;
if (x2625) {
x2631 = x2630;
} else {
x2631 = false;
}
bool x2632;
if (x2631) {
x2632 = x2630;
} else {
x2632 = false;
}
if (x2632) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2203,x2205,x2205,64,x2580,x1291,x1291);
assert(false && "");
}
bool x2638 = x2203 <= x2580;
int32_t x2639;
if (x2638) {
x2639 = x2580;
} else {
x2639 = x2203;
}
int32_t x2655;
if (x2621) {
x2655 = 0;
} else {
x2655 = x2211;
}
int32_t x2658;
if (x2622) {
x2658 = 0;
} else {
x2658 = x1297;
}
for(int x2661=0; x2661 < 64; x2661++) {
int32_t x2667 = x2212 * x2661;
int32_t x2674 = x2585 * x2661;
for(int x2663=0; x2663 < x2639; x2663++) {
int32_t x2668 = x2655 * x2663;
int32_t x2669 = x2667 + x2668;
int32_t x2675 = x2658 * x2663;
int32_t x2676 = x2674 + x2675;
for(int x2665=0; x2665 < x2641; x2665++) {
int32_t x2670 = x2656 * x2665;
int32_t x2671 = x2669 + x2670;
int32_t x2677 = x2659 * x2665;
int32_t x2678 = x2676 + x2677;
for(int x2666=0; x2666 < x2641; x2666++) {
int32_t x2672 = x2657 * x2666;
int32_t x2673 = x2671 + x2672;
float x2681 = x2214[x2673];
int32_t x2679 = x2660 * x2666;
int32_t x2680 = x2678 + x2679;
float x2682 = x2587[x2680];
float x2683 = x2681 + x2682;
x2214[x2673] = x2683;

}

}

}

}
float* x2693 = (float*)myMalloc(x2213 * sizeof(float));;
for(int x2695=0; x2695 < x2213; x2695++) {
float x2696 = x2214[x2695];
bool x2697 = x2696 < 0.0f;
if (x2697) {
x2693[x2695] = 0.0f;
} else {
float x2700 = x2214[x2695];
x2693[x2695] = x2700;
}

}
float* x2714 = (float*)myMalloc(x2713 * sizeof(float));;
int32_t x2717 = 64 * x2203;
int32_t x2718 = x2717 * x2709;
float* x2719 = (float*)myMalloc(x2718 * sizeof(float));;
int32_t x2715 = x2203 * x2709;
for(int x2720=0; x2720 < 64; x2720++) {
int32_t x2721 = x2720 * x2212;
float* x2722 = x2693+x2721;
int32_t x2723 = x2720 * x2710;
float* x2724 = x2714+x2723;
int32_t x2725 = x2720 * x2715;
float* x2726 = x2719+x2725;
for(int x2727=0; x2727 < x2203; x2727++) {
int32_t x2728 = x2727 / 1;
int32_t x2732 = x2728 * x2708;
int32_t x2733 = x2732 * x2708;
int32_t x2729 = x2727 % 1;
int32_t x2730 = x2729 / 1;
int32_t x2734 = x2730 * x2708;
int32_t x2735 = x2734 * x2708;
int32_t x2736 = x2733 + x2735;
int32_t x2731 = x2729 % 1;
int32_t x2737 = x2731 * x2708;
int32_t x2738 = x2737 * x2708;
int32_t x2739 = x2736 + x2738;
float* x2740 = x2726+x2739;
int32_t x2741 = x2728 * x2205;
int32_t x2742 = x2741 * x2205;
float* x2743 = x2722+x2742;
for(int x2745=0; x2745 < x2708; x2745++) {
int32_t x2747 = x2745 * x2708;
float* x2748 = x2740+x2747;
int32_t x2746 = x2745 + x2730;
int32_t x2749 = x2746 * x2205;
int32_t x2750 = x2749 + x2731;
float* x2751 = x2743+x2750;
memcpy(x2748, x2751, 4 * x2708);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x2709,x2203,1,x171,x2203,x2726,x2709,1,x2724,x2709);

}
int32_t x2760 = 0;
int32_t x2761 = 1;
x2761 *= 1;
x2760 += 1;
x2761 *= 1;
x2761 *= 1;
int32_t x2766 = x2760;
bool x2767 = x2766 >= 2;
if (x2767) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2772 = x2766 == 0;
if (x2772) {
int32_t x2773 = x2761;
bool x2774 = x2773 == 64;
if (x2774) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2781 = x2761;
int32_t x2782 = 64 / x2781;
bool x2783 = x2782 == 1;
bool x2786;
if (x454) {
bool x2784 = 64 == x2782;
bool x2785 = x2783 || x2784;
x2786 = x2785;
} else {
x2786 = false;
}
bool x2790;
if (x2786) {
x2790 = x2789;
} else {
x2790 = false;
}
bool x2791;
if (x2790) {
x2791 = x2789;
} else {
x2791 = false;
}
if (x2791) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x2708,x2708,1,x2782,1,1);
assert(false && "");
}
bool x2797 = 64 <= x2782;
int32_t x2798;
if (x2797) {
x2798 = x2782;
} else {
x2798 = 64;
}
int32_t x2807 = x2798 * x2806;
int32_t x2808 = 64 * x2807;
float* x2809 = (float*)myMalloc(x2808 * sizeof(float));;
int32_t x2812;
if (x2783) {
x2812 = 0;
} else {
x2812 = 1;
}
for(int x2813=0; x2813 < 64; x2813++) {
int32_t x2825 = x2710 * x2813;
int32_t x2819 = x2807 * x2813;
for(int x2815=0; x2815 < x2798; x2815++) {
int32_t x2826 = x2709 * x2815;
int32_t x2827 = x2825 + x2826;
int32_t x2832 = x2812 * x2815;
int32_t x2821 = x2806 * x2815;
for(int x2817=0; x2817 < x2800; x2817++) {
int32_t x2828 = x2810 * x2817;
int32_t x2829 = x2827 + x2828;
int32_t x2823 = x2800 * x2817;
for(int x2818=0; x2818 < x2800; x2818++) {
int32_t x2830 = x2811 * x2818;
int32_t x2831 = x2829 + x2830;
float x2833 = x2714[x2831];
float x2834 = x10[x2832];
int32_t x2820 = x2818 + x2819;
int32_t x2822 = x2820 + x2821;
int32_t x2824 = x2822 + x2823;
float x2835 = x2833 - x2834;
x2809[x2824] = x2835;

}

}

}

}
float* x2845 = (float*)myMalloc(64 * sizeof(float));;
for(int x2846=0; x2846 < 64; x2846++) {
float x2847 = x102[x2846];
float x2848 = x2847 + 1.0E-5f;
x2845[x2846] = x2848;

}
float* x2852 = (float*)myMalloc(64 * sizeof(float));;
for(int x2853=0; x2853 < 64; x2853++) {
float x2854 = x2845[x2853];
double x2855 = (double)x2854;
double x2856 = sqrt(x2855);
float x2857 = (float)x2856;
x2852[x2853] = x2857;

}
int32_t x2861 = 0;
int32_t x2862 = 1;
x2862 *= 1;
x2861 += 1;
x2862 *= 1;
x2862 *= 1;
int32_t x2867 = x2861;
bool x2868 = x2867 >= 2;
if (x2868) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2873 = x2867 == 0;
if (x2873) {
int32_t x2874 = x2862;
bool x2875 = x2874 == 64;
if (x2875) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2882 = x2862;
bool x2884 = x2798 == 1;
int32_t x2883 = 64 / x2882;
bool x2885 = x2883 == 1;
bool x2889;
if (x454) {
bool x2886 = x2884 || x2885;
bool x2887 = x2798 == x2883;
bool x2888 = x2886 || x2887;
x2889 = x2888;
} else {
x2889 = false;
}
bool x2893;
if (x2889) {
x2893 = x2892;
} else {
x2893 = false;
}
bool x2894;
if (x2893) {
x2894 = x2892;
} else {
x2894 = false;
}
if (x2894) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2798,x2800,x2800,1,x2883,1,1);
assert(false && "");
}
bool x2900 = x2798 <= x2883;
int32_t x2901;
if (x2900) {
x2901 = x2883;
} else {
x2901 = x2798;
}
int32_t x2910 = x2901 * x2909;
int32_t x2911 = 64 * x2910;
float* x2912 = (float*)myMalloc(x2911 * sizeof(float));;
int32_t x2913;
if (x2884) {
x2913 = 0;
} else {
x2913 = x2806;
}
int32_t x2916;
if (x2885) {
x2916 = 0;
} else {
x2916 = 1;
}
for(int x2917=0; x2917 < 64; x2917++) {
int32_t x2929 = x2807 * x2917;
int32_t x2923 = x2910 * x2917;
for(int x2919=0; x2919 < x2901; x2919++) {
int32_t x2930 = x2913 * x2919;
int32_t x2931 = x2929 + x2930;
int32_t x2936 = x2916 * x2919;
int32_t x2925 = x2909 * x2919;
for(int x2921=0; x2921 < x2903; x2921++) {
int32_t x2932 = x2914 * x2921;
int32_t x2933 = x2931 + x2932;
int32_t x2927 = x2903 * x2921;
for(int x2922=0; x2922 < x2903; x2922++) {
int32_t x2934 = x2915 * x2922;
int32_t x2935 = x2933 + x2934;
float x2937 = x2809[x2935];
float x2938 = x2852[x2936];
int32_t x2924 = x2922 + x2923;
int32_t x2926 = x2924 + x2925;
int32_t x2928 = x2926 + x2927;
float x2939 = x2937 / x2938;
x2912[x2928] = x2939;

}

}

}

}
int32_t x2949 = 0;
int32_t x2950 = 1;
x2950 *= 1;
x2949 += 1;
x2950 *= 1;
x2950 *= 1;
int32_t x2955 = x2949;
bool x2956 = x2955 >= 2;
if (x2956) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2961 = x2955 == 0;
if (x2961) {
int32_t x2962 = x2950;
bool x2963 = x2962 == 64;
if (x2963) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2970 = x2950;
bool x2972 = x2901 == 1;
int32_t x2971 = 64 / x2970;
bool x2973 = x2971 == 1;
bool x2977;
if (x454) {
bool x2974 = x2972 || x2973;
bool x2975 = x2901 == x2971;
bool x2976 = x2974 || x2975;
x2977 = x2976;
} else {
x2977 = false;
}
bool x2981;
if (x2977) {
x2981 = x2980;
} else {
x2981 = false;
}
bool x2982;
if (x2981) {
x2982 = x2980;
} else {
x2982 = false;
}
if (x2982) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2901,x2903,x2903,1,x2971,1,1);
assert(false && "");
}
bool x2988 = x2901 <= x2971;
int32_t x2989;
if (x2988) {
x2989 = x2971;
} else {
x2989 = x2901;
}
int32_t x2998 = x2989 * x2997;
int32_t x2999 = 64 * x2998;
float* x3000 = (float*)myMalloc(x2999 * sizeof(float));;
int32_t x3001;
if (x2972) {
x3001 = 0;
} else {
x3001 = x2909;
}
int32_t x3004;
if (x2973) {
x3004 = 0;
} else {
x3004 = 1;
}
for(int x3005=0; x3005 < 64; x3005++) {
int32_t x3017 = x2910 * x3005;
int32_t x3011 = x2998 * x3005;
for(int x3007=0; x3007 < x2989; x3007++) {
int32_t x3018 = x3001 * x3007;
int32_t x3019 = x3017 + x3018;
int32_t x3024 = x3004 * x3007;
int32_t x3013 = x2997 * x3007;
for(int x3009=0; x3009 < x2991; x3009++) {
int32_t x3020 = x3002 * x3009;
int32_t x3021 = x3019 + x3020;
int32_t x3015 = x2991 * x3009;
for(int x3010=0; x3010 < x2991; x3010++) {
int32_t x3022 = x3003 * x3010;
int32_t x3023 = x3021 + x3022;
float x3025 = x2912[x3023];
float x3026 = x142[x3024];
int32_t x3012 = x3010 + x3011;
int32_t x3014 = x3012 + x3013;
int32_t x3016 = x3014 + x3015;
float x3027 = x3025 * x3026;
x3000[x3016] = x3027;

}

}

}

}
int32_t x3037 = 0;
int32_t x3038 = 1;
x3038 *= 1;
x3037 += 1;
x3038 *= 1;
x3038 *= 1;
int32_t x3043 = x3037;
bool x3044 = x3043 >= 2;
if (x3044) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3049 = x3043 == 0;
if (x3049) {
int32_t x3050 = x3038;
bool x3051 = x3050 == 64;
if (x3051) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3058 = x3038;
bool x3060 = x2989 == 1;
int32_t x3059 = 64 / x3058;
bool x3061 = x3059 == 1;
bool x3065;
if (x454) {
bool x3062 = x3060 || x3061;
bool x3063 = x2989 == x3059;
bool x3064 = x3062 || x3063;
x3065 = x3064;
} else {
x3065 = false;
}
bool x3069;
if (x3065) {
x3069 = x3068;
} else {
x3069 = false;
}
bool x3070;
if (x3069) {
x3070 = x3068;
} else {
x3070 = false;
}
if (x3070) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2989,x2991,x2991,1,x3059,1,1);
assert(false && "");
}
bool x3076 = x2989 <= x3059;
int32_t x3077;
if (x3076) {
x3077 = x3059;
} else {
x3077 = x2989;
}
int32_t x3086 = x3077 * x3085;
int32_t x3087 = 64 * x3086;
float* x3088 = (float*)myMalloc(x3087 * sizeof(float));;
int32_t x3089;
if (x3060) {
x3089 = 0;
} else {
x3089 = x2997;
}
int32_t x3092;
if (x3061) {
x3092 = 0;
} else {
x3092 = 1;
}
for(int x3093=0; x3093 < 64; x3093++) {
int32_t x3105 = x2998 * x3093;
int32_t x3099 = x3086 * x3093;
for(int x3095=0; x3095 < x3077; x3095++) {
int32_t x3106 = x3089 * x3095;
int32_t x3107 = x3105 + x3106;
int32_t x3112 = x3092 * x3095;
int32_t x3101 = x3085 * x3095;
for(int x3097=0; x3097 < x3079; x3097++) {
int32_t x3108 = x3090 * x3097;
int32_t x3109 = x3107 + x3108;
int32_t x3103 = x3079 * x3097;
for(int x3098=0; x3098 < x3079; x3098++) {
int32_t x3110 = x3091 * x3098;
int32_t x3111 = x3109 + x3110;
float x3113 = x3000[x3111];
float x3114 = x60[x3112];
int32_t x3100 = x3098 + x3099;
int32_t x3102 = x3100 + x3101;
int32_t x3104 = x3102 + x3103;
float x3115 = x3113 + x3114;
x3088[x3104] = x3115;

}

}

}

}
float* x3125 = (float*)myMalloc(x3087 * sizeof(float));;
for(int x3127=0; x3127 < x3087; x3127++) {
float x3128 = x3088[x3127];
bool x3129 = x3128 < 0.0f;
if (x3129) {
x3125[x3127] = 0.0f;
} else {
float x3132 = x3088[x3127];
x3125[x3127] = x3132;
}

}
float* x3147 = (float*)myMalloc(x3146 * sizeof(float));;
int32_t x3148 = 9 * x3077;
int32_t x3151 = 64 * x3148;
int32_t x3152 = x3151 * x3142;
float* x3153 = (float*)myMalloc(x3152 * sizeof(float));;
int32_t x3149 = x3148 * x3142;
int32_t x3161 = x3077 * 3;
int32_t x3162 = x3161 * 3;
for(int x3154=0; x3154 < 64; x3154++) {
int32_t x3155 = x3154 * x3086;
float* x3156 = x3125+x3155;
int32_t x3157 = x3154 * x3143;
float* x3158 = x3147+x3157;
int32_t x3159 = x3154 * x3149;
float* x3160 = x3153+x3159;
for(int x3164=0; x3164 < x3162; x3164++) {
int32_t x3165 = x3164 / 9;
int32_t x3169 = x3165 * 3;
int32_t x3170 = x3169 * 3;
int32_t x3171 = x3170 * x3141;
int32_t x3172 = x3171 * x3141;
int32_t x3166 = x3164 % 9;
int32_t x3167 = x3166 / 3;
int32_t x3173 = x3167 * 3;
int32_t x3174 = x3173 * x3141;
int32_t x3175 = x3174 * x3141;
int32_t x3176 = x3172 + x3175;
int32_t x3168 = x3166 % 3;
int32_t x3177 = x3168 * x3141;
int32_t x3178 = x3177 * x3141;
int32_t x3179 = x3176 + x3178;
float* x3180 = x3160+x3179;
int32_t x3181 = x3165 * x3079;
int32_t x3182 = x3181 * x3079;
float* x3183 = x3156+x3182;
int32_t x3196 = 1 - x3168;
bool x3197 = x3196 > 0;
int32_t x3198;
if (x3197) {
x3198 = x3196;
} else {
x3198 = 0;
}
int32_t x3199 = 3 - x3168;
int32_t x3200 = x3199 - 1;
int32_t x3201 = 1 - x3200;
bool x3202 = x3201 > 0;
int32_t x3203;
if (x3202) {
x3203 = x3201;
} else {
x3203 = 0;
}
int32_t x3204 = x3141 - x3203;
int32_t x3205 = x3204 - x3198;
bool x3206 = x3205 <= 0;
bool x3210 = x3198 > 0;
int32_t x3195 = -1 + x3168;
bool x3223 = x3203 > 0;
for(int x3185=0; x3185 < x3141; x3185++) {
int32_t x3186 = x3185 - 1;
int32_t x3187 = x3186 + x3167;
bool x3188 = x3187 < 0;
bool x3189 = x3187 >= x3079;
bool x3190 = x3188 || x3189;
if (x3190) {
int32_t x3191 = x3185 * x3141;
float* x3192 = x3180+x3191;
memset(x3192, 0, 4 * x3141);;
} else {
if (x3206) {
int32_t x3191 = x3185 * x3141;
float* x3207 = x3180+x3191;
memset(x3207, 0, 4 * x3141);;
} else {
int32_t x3191 = x3185 * x3141;
if (x3210) {
float* x3211 = x3180+x3191;
memset(x3211, 0, 4 * x3198);;
} else {
}
// may have segfault here
int32_t x3216 = x3191 + x3198;
float* x3217 = x3180+x3216;
int32_t x3218 = x3187 * x3079;
int32_t x3219 = x3218 + x3195;
int32_t x3220 = x3219 + x3198;
float* x3221 = x3183+x3220;
memcpy(x3217, x3221, 4 * x3205);;
if (x3223) {
int32_t x3224 = x3191 + x3141;
int32_t x3225 = x3224 - x3203;
float* x3226 = x3180+x3225;
memset(x3226, 0, 4 * x3203);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x3142,x3148,1,x83,x3148,x3160,x3142,1,x3158,x3142);

}
int32_t x3241 = 0;
int32_t x3242 = 1;
x3242 *= 1;
x3241 += 1;
x3242 *= 1;
x3242 *= 1;
int32_t x3247 = x3241;
bool x3248 = x3247 >= 2;
if (x3248) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3253 = x3247 == 0;
if (x3253) {
int32_t x3254 = x3242;
bool x3255 = x3254 == 64;
if (x3255) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3262 = x3242;
int32_t x3263 = 64 / x3262;
bool x3264 = x3263 == 1;
bool x3267;
if (x454) {
bool x3265 = 64 == x3263;
bool x3266 = x3264 || x3265;
x3267 = x3266;
} else {
x3267 = false;
}
bool x3271;
if (x3267) {
x3271 = x3270;
} else {
x3271 = false;
}
bool x3272;
if (x3271) {
x3272 = x3270;
} else {
x3272 = false;
}
if (x3272) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x3141,x3141,1,x3263,1,1);
assert(false && "");
}
bool x3278 = 64 <= x3263;
int32_t x3279;
if (x3278) {
x3279 = x3263;
} else {
x3279 = 64;
}
int32_t x3288 = x3279 * x3287;
int32_t x3289 = 64 * x3288;
float* x3290 = (float*)myMalloc(x3289 * sizeof(float));;
int32_t x3293;
if (x3264) {
x3293 = 0;
} else {
x3293 = 1;
}
for(int x3294=0; x3294 < 64; x3294++) {
int32_t x3306 = x3143 * x3294;
int32_t x3300 = x3288 * x3294;
for(int x3296=0; x3296 < x3279; x3296++) {
int32_t x3307 = x3142 * x3296;
int32_t x3308 = x3306 + x3307;
int32_t x3313 = x3293 * x3296;
int32_t x3302 = x3287 * x3296;
for(int x3298=0; x3298 < x3281; x3298++) {
int32_t x3309 = x3291 * x3298;
int32_t x3310 = x3308 + x3309;
int32_t x3304 = x3281 * x3298;
for(int x3299=0; x3299 < x3281; x3299++) {
int32_t x3311 = x3292 * x3299;
int32_t x3312 = x3310 + x3311;
float x3314 = x3147[x3312];
float x3315 = x44[x3313];
int32_t x3301 = x3299 + x3300;
int32_t x3303 = x3301 + x3302;
int32_t x3305 = x3303 + x3304;
float x3316 = x3314 - x3315;
x3290[x3305] = x3316;

}

}

}

}
float* x3326 = (float*)myMalloc(64 * sizeof(float));;
for(int x3327=0; x3327 < 64; x3327++) {
float x3328 = x244[x3327];
float x3329 = x3328 + 1.0E-5f;
x3326[x3327] = x3329;

}
float* x3333 = (float*)myMalloc(64 * sizeof(float));;
for(int x3334=0; x3334 < 64; x3334++) {
float x3335 = x3326[x3334];
double x3336 = (double)x3335;
double x3337 = sqrt(x3336);
float x3338 = (float)x3337;
x3333[x3334] = x3338;

}
int32_t x3342 = 0;
int32_t x3343 = 1;
x3343 *= 1;
x3342 += 1;
x3343 *= 1;
x3343 *= 1;
int32_t x3348 = x3342;
bool x3349 = x3348 >= 2;
if (x3349) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3354 = x3348 == 0;
if (x3354) {
int32_t x3355 = x3343;
bool x3356 = x3355 == 64;
if (x3356) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3363 = x3343;
bool x3365 = x3279 == 1;
int32_t x3364 = 64 / x3363;
bool x3366 = x3364 == 1;
bool x3370;
if (x454) {
bool x3367 = x3365 || x3366;
bool x3368 = x3279 == x3364;
bool x3369 = x3367 || x3368;
x3370 = x3369;
} else {
x3370 = false;
}
bool x3374;
if (x3370) {
x3374 = x3373;
} else {
x3374 = false;
}
bool x3375;
if (x3374) {
x3375 = x3373;
} else {
x3375 = false;
}
if (x3375) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3279,x3281,x3281,1,x3364,1,1);
assert(false && "");
}
bool x3381 = x3279 <= x3364;
int32_t x3382;
if (x3381) {
x3382 = x3364;
} else {
x3382 = x3279;
}
int32_t x3391 = x3382 * x3390;
int32_t x3392 = 64 * x3391;
float* x3393 = (float*)myMalloc(x3392 * sizeof(float));;
int32_t x3394;
if (x3365) {
x3394 = 0;
} else {
x3394 = x3287;
}
int32_t x3397;
if (x3366) {
x3397 = 0;
} else {
x3397 = 1;
}
for(int x3398=0; x3398 < 64; x3398++) {
int32_t x3410 = x3288 * x3398;
int32_t x3404 = x3391 * x3398;
for(int x3400=0; x3400 < x3382; x3400++) {
int32_t x3411 = x3394 * x3400;
int32_t x3412 = x3410 + x3411;
int32_t x3417 = x3397 * x3400;
int32_t x3406 = x3390 * x3400;
for(int x3402=0; x3402 < x3384; x3402++) {
int32_t x3413 = x3395 * x3402;
int32_t x3414 = x3412 + x3413;
int32_t x3408 = x3384 * x3402;
for(int x3403=0; x3403 < x3384; x3403++) {
int32_t x3415 = x3396 * x3403;
int32_t x3416 = x3414 + x3415;
float x3418 = x3290[x3416];
float x3419 = x3333[x3417];
int32_t x3405 = x3403 + x3404;
int32_t x3407 = x3405 + x3406;
int32_t x3409 = x3407 + x3408;
float x3420 = x3418 / x3419;
x3393[x3409] = x3420;

}

}

}

}
int32_t x3430 = 0;
int32_t x3431 = 1;
x3431 *= 1;
x3430 += 1;
x3431 *= 1;
x3431 *= 1;
int32_t x3436 = x3430;
bool x3437 = x3436 >= 2;
if (x3437) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3442 = x3436 == 0;
if (x3442) {
int32_t x3443 = x3431;
bool x3444 = x3443 == 64;
if (x3444) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3451 = x3431;
bool x3453 = x3382 == 1;
int32_t x3452 = 64 / x3451;
bool x3454 = x3452 == 1;
bool x3458;
if (x454) {
bool x3455 = x3453 || x3454;
bool x3456 = x3382 == x3452;
bool x3457 = x3455 || x3456;
x3458 = x3457;
} else {
x3458 = false;
}
bool x3462;
if (x3458) {
x3462 = x3461;
} else {
x3462 = false;
}
bool x3463;
if (x3462) {
x3463 = x3461;
} else {
x3463 = false;
}
if (x3463) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3382,x3384,x3384,1,x3452,1,1);
assert(false && "");
}
bool x3469 = x3382 <= x3452;
int32_t x3470;
if (x3469) {
x3470 = x3452;
} else {
x3470 = x3382;
}
int32_t x3479 = x3470 * x3478;
int32_t x3480 = 64 * x3479;
float* x3481 = (float*)myMalloc(x3480 * sizeof(float));;
int32_t x3482;
if (x3453) {
x3482 = 0;
} else {
x3482 = x3390;
}
int32_t x3485;
if (x3454) {
x3485 = 0;
} else {
x3485 = 1;
}
for(int x3486=0; x3486 < 64; x3486++) {
int32_t x3498 = x3391 * x3486;
int32_t x3492 = x3479 * x3486;
for(int x3488=0; x3488 < x3470; x3488++) {
int32_t x3499 = x3482 * x3488;
int32_t x3500 = x3498 + x3499;
int32_t x3505 = x3485 * x3488;
int32_t x3494 = x3478 * x3488;
for(int x3490=0; x3490 < x3472; x3490++) {
int32_t x3501 = x3483 * x3490;
int32_t x3502 = x3500 + x3501;
int32_t x3496 = x3472 * x3490;
for(int x3491=0; x3491 < x3472; x3491++) {
int32_t x3503 = x3484 * x3491;
int32_t x3504 = x3502 + x3503;
float x3506 = x3393[x3504];
float x3507 = x208[x3505];
int32_t x3493 = x3491 + x3492;
int32_t x3495 = x3493 + x3494;
int32_t x3497 = x3495 + x3496;
float x3508 = x3506 * x3507;
x3481[x3497] = x3508;

}

}

}

}
int32_t x3518 = 0;
int32_t x3519 = 1;
x3519 *= 1;
x3518 += 1;
x3519 *= 1;
x3519 *= 1;
int32_t x3524 = x3518;
bool x3525 = x3524 >= 2;
if (x3525) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3530 = x3524 == 0;
if (x3530) {
int32_t x3531 = x3519;
bool x3532 = x3531 == 64;
if (x3532) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3539 = x3519;
bool x3541 = x3470 == 1;
int32_t x3540 = 64 / x3539;
bool x3542 = x3540 == 1;
bool x3546;
if (x454) {
bool x3543 = x3541 || x3542;
bool x3544 = x3470 == x3540;
bool x3545 = x3543 || x3544;
x3546 = x3545;
} else {
x3546 = false;
}
bool x3550;
if (x3546) {
x3550 = x3549;
} else {
x3550 = false;
}
bool x3551;
if (x3550) {
x3551 = x3549;
} else {
x3551 = false;
}
if (x3551) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3470,x3472,x3472,1,x3540,1,1);
assert(false && "");
}
bool x3557 = x3470 <= x3540;
int32_t x3558;
if (x3557) {
x3558 = x3540;
} else {
x3558 = x3470;
}
int32_t x3567 = x3558 * x3566;
int32_t x3568 = 64 * x3567;
float* x3569 = (float*)myMalloc(x3568 * sizeof(float));;
int32_t x3570;
if (x3541) {
x3570 = 0;
} else {
x3570 = x3478;
}
int32_t x3573;
if (x3542) {
x3573 = 0;
} else {
x3573 = 1;
}
for(int x3574=0; x3574 < 64; x3574++) {
int32_t x3586 = x3479 * x3574;
int32_t x3580 = x3567 * x3574;
for(int x3576=0; x3576 < x3558; x3576++) {
int32_t x3587 = x3570 * x3576;
int32_t x3588 = x3586 + x3587;
int32_t x3593 = x3573 * x3576;
int32_t x3582 = x3566 * x3576;
for(int x3578=0; x3578 < x3560; x3578++) {
int32_t x3589 = x3571 * x3578;
int32_t x3590 = x3588 + x3589;
int32_t x3584 = x3560 * x3578;
for(int x3579=0; x3579 < x3560; x3579++) {
int32_t x3591 = x3572 * x3579;
int32_t x3592 = x3590 + x3591;
float x3594 = x3481[x3592];
float x3595 = x153[x3593];
int32_t x3581 = x3579 + x3580;
int32_t x3583 = x3581 + x3582;
int32_t x3585 = x3583 + x3584;
float x3596 = x3594 + x3595;
x3569[x3585] = x3596;

}

}

}

}
float* x3606 = (float*)myMalloc(x3568 * sizeof(float));;
for(int x3608=0; x3608 < x3568; x3608++) {
float x3609 = x3569[x3608];
bool x3610 = x3609 < 0.0f;
if (x3610) {
x3606[x3608] = 0.0f;
} else {
float x3613 = x3569[x3608];
x3606[x3608] = x3613;
}

}
float* x3627 = (float*)myMalloc(x3626 * sizeof(float));;
int32_t x3630 = 64 * x3558;
int32_t x3631 = x3630 * x3622;
float* x3632 = (float*)myMalloc(x3631 * sizeof(float));;
int32_t x3628 = x3558 * x3622;
for(int x3633=0; x3633 < 64; x3633++) {
int32_t x3634 = x3633 * x3567;
float* x3635 = x3606+x3634;
int32_t x3636 = x3633 * x3623;
float* x3637 = x3627+x3636;
int32_t x3638 = x3633 * x3628;
float* x3639 = x3632+x3638;
for(int x3640=0; x3640 < x3558; x3640++) {
int32_t x3641 = x3640 / 1;
int32_t x3645 = x3641 * x3621;
int32_t x3646 = x3645 * x3621;
int32_t x3642 = x3640 % 1;
int32_t x3643 = x3642 / 1;
int32_t x3647 = x3643 * x3621;
int32_t x3648 = x3647 * x3621;
int32_t x3649 = x3646 + x3648;
int32_t x3644 = x3642 % 1;
int32_t x3650 = x3644 * x3621;
int32_t x3651 = x3650 * x3621;
int32_t x3652 = x3649 + x3651;
float* x3653 = x3639+x3652;
int32_t x3654 = x3641 * x3560;
int32_t x3655 = x3654 * x3560;
float* x3656 = x3635+x3655;
for(int x3658=0; x3658 < x3621; x3658++) {
int32_t x3660 = x3658 * x3621;
float* x3661 = x3653+x3660;
int32_t x3659 = x3658 + x3643;
int32_t x3662 = x3659 * x3560;
int32_t x3663 = x3662 + x3644;
float* x3664 = x3656+x3663;
memcpy(x3661, x3664, 4 * x3621);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x3622,x3558,1,x130,x3558,x3639,x3622,1,x3637,x3622);

}
int32_t x3673 = 0;
int32_t x3674 = 1;
x3674 *= 1;
x3673 += 1;
x3674 *= 1;
x3674 *= 1;
int32_t x3679 = x3673;
bool x3680 = x3679 >= 2;
if (x3680) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3685 = x3679 == 0;
if (x3685) {
int32_t x3686 = x3674;
bool x3687 = x3686 == 256;
if (x3687) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3694 = x3674;
int32_t x3695 = 256 / x3694;
bool x3696 = x3695 == 1;
bool x3699;
if (x454) {
bool x3697 = 256 == x3695;
bool x3698 = x3696 || x3697;
x3699 = x3698;
} else {
x3699 = false;
}
bool x3703;
if (x3699) {
x3703 = x3702;
} else {
x3703 = false;
}
bool x3704;
if (x3703) {
x3704 = x3702;
} else {
x3704 = false;
}
if (x3704) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x3621,x3621,1,x3695,1,1);
assert(false && "");
}
bool x3710 = 256 <= x3695;
int32_t x3711;
if (x3710) {
x3711 = x3695;
} else {
x3711 = 256;
}
int32_t x3720 = x3711 * x3719;
int32_t x3721 = 64 * x3720;
float* x3722 = (float*)myMalloc(x3721 * sizeof(float));;
int32_t x3725;
if (x3696) {
x3725 = 0;
} else {
x3725 = 1;
}
for(int x3726=0; x3726 < 64; x3726++) {
int32_t x3738 = x3623 * x3726;
int32_t x3732 = x3720 * x3726;
for(int x3728=0; x3728 < x3711; x3728++) {
int32_t x3739 = x3622 * x3728;
int32_t x3740 = x3738 + x3739;
int32_t x3745 = x3725 * x3728;
int32_t x3734 = x3719 * x3728;
for(int x3730=0; x3730 < x3713; x3730++) {
int32_t x3741 = x3723 * x3730;
int32_t x3742 = x3740 + x3741;
int32_t x3736 = x3713 * x3730;
for(int x3731=0; x3731 < x3713; x3731++) {
int32_t x3743 = x3724 * x3731;
int32_t x3744 = x3742 + x3743;
float x3746 = x3627[x3744];
float x3747 = x91[x3745];
int32_t x3733 = x3731 + x3732;
int32_t x3735 = x3733 + x3734;
int32_t x3737 = x3735 + x3736;
float x3748 = x3746 - x3747;
x3722[x3737] = x3748;

}

}

}

}
float* x3758 = (float*)myMalloc(256 * sizeof(float));;
for(int x3759=0; x3759 < 256; x3759++) {
float x3760 = x166[x3759];
float x3761 = x3760 + 1.0E-5f;
x3758[x3759] = x3761;

}
float* x3765 = (float*)myMalloc(256 * sizeof(float));;
for(int x3766=0; x3766 < 256; x3766++) {
float x3767 = x3758[x3766];
double x3768 = (double)x3767;
double x3769 = sqrt(x3768);
float x3770 = (float)x3769;
x3765[x3766] = x3770;

}
int32_t x3774 = 0;
int32_t x3775 = 1;
x3775 *= 1;
x3774 += 1;
x3775 *= 1;
x3775 *= 1;
int32_t x3780 = x3774;
bool x3781 = x3780 >= 2;
if (x3781) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3786 = x3780 == 0;
if (x3786) {
int32_t x3787 = x3775;
bool x3788 = x3787 == 256;
if (x3788) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3795 = x3775;
bool x3797 = x3711 == 1;
int32_t x3796 = 256 / x3795;
bool x3798 = x3796 == 1;
bool x3802;
if (x454) {
bool x3799 = x3797 || x3798;
bool x3800 = x3711 == x3796;
bool x3801 = x3799 || x3800;
x3802 = x3801;
} else {
x3802 = false;
}
bool x3806;
if (x3802) {
x3806 = x3805;
} else {
x3806 = false;
}
bool x3807;
if (x3806) {
x3807 = x3805;
} else {
x3807 = false;
}
if (x3807) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3711,x3713,x3713,1,x3796,1,1);
assert(false && "");
}
bool x3813 = x3711 <= x3796;
int32_t x3814;
if (x3813) {
x3814 = x3796;
} else {
x3814 = x3711;
}
int32_t x3823 = x3814 * x3822;
int32_t x3824 = 64 * x3823;
float* x3825 = (float*)myMalloc(x3824 * sizeof(float));;
int32_t x3826;
if (x3797) {
x3826 = 0;
} else {
x3826 = x3719;
}
int32_t x3829;
if (x3798) {
x3829 = 0;
} else {
x3829 = 1;
}
for(int x3830=0; x3830 < 64; x3830++) {
int32_t x3842 = x3720 * x3830;
int32_t x3836 = x3823 * x3830;
for(int x3832=0; x3832 < x3814; x3832++) {
int32_t x3843 = x3826 * x3832;
int32_t x3844 = x3842 + x3843;
int32_t x3849 = x3829 * x3832;
int32_t x3838 = x3822 * x3832;
for(int x3834=0; x3834 < x3816; x3834++) {
int32_t x3845 = x3827 * x3834;
int32_t x3846 = x3844 + x3845;
int32_t x3840 = x3816 * x3834;
for(int x3835=0; x3835 < x3816; x3835++) {
int32_t x3847 = x3828 * x3835;
int32_t x3848 = x3846 + x3847;
float x3850 = x3722[x3848];
float x3851 = x3765[x3849];
int32_t x3837 = x3835 + x3836;
int32_t x3839 = x3837 + x3838;
int32_t x3841 = x3839 + x3840;
float x3852 = x3850 / x3851;
x3825[x3841] = x3852;

}

}

}

}
int32_t x3862 = 0;
int32_t x3863 = 1;
x3863 *= 1;
x3862 += 1;
x3863 *= 1;
x3863 *= 1;
int32_t x3868 = x3862;
bool x3869 = x3868 >= 2;
if (x3869) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3874 = x3868 == 0;
if (x3874) {
int32_t x3875 = x3863;
bool x3876 = x3875 == 256;
if (x3876) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3883 = x3863;
bool x3885 = x3814 == 1;
int32_t x3884 = 256 / x3883;
bool x3886 = x3884 == 1;
bool x3890;
if (x454) {
bool x3887 = x3885 || x3886;
bool x3888 = x3814 == x3884;
bool x3889 = x3887 || x3888;
x3890 = x3889;
} else {
x3890 = false;
}
bool x3894;
if (x3890) {
x3894 = x3893;
} else {
x3894 = false;
}
bool x3895;
if (x3894) {
x3895 = x3893;
} else {
x3895 = false;
}
if (x3895) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3814,x3816,x3816,1,x3884,1,1);
assert(false && "");
}
bool x3901 = x3814 <= x3884;
int32_t x3902;
if (x3901) {
x3902 = x3884;
} else {
x3902 = x3814;
}
int32_t x3911 = x3902 * x3910;
int32_t x3912 = 64 * x3911;
float* x3913 = (float*)myMalloc(x3912 * sizeof(float));;
int32_t x3914;
if (x3885) {
x3914 = 0;
} else {
x3914 = x3822;
}
int32_t x3917;
if (x3886) {
x3917 = 0;
} else {
x3917 = 1;
}
for(int x3918=0; x3918 < 64; x3918++) {
int32_t x3930 = x3823 * x3918;
int32_t x3924 = x3911 * x3918;
for(int x3920=0; x3920 < x3902; x3920++) {
int32_t x3931 = x3914 * x3920;
int32_t x3932 = x3930 + x3931;
int32_t x3937 = x3917 * x3920;
int32_t x3926 = x3910 * x3920;
for(int x3922=0; x3922 < x3904; x3922++) {
int32_t x3933 = x3915 * x3922;
int32_t x3934 = x3932 + x3933;
int32_t x3928 = x3904 * x3922;
for(int x3923=0; x3923 < x3904; x3923++) {
int32_t x3935 = x3916 * x3923;
int32_t x3936 = x3934 + x3935;
float x3938 = x3825[x3936];
float x3939 = x58[x3937];
int32_t x3925 = x3923 + x3924;
int32_t x3927 = x3925 + x3926;
int32_t x3929 = x3927 + x3928;
float x3940 = x3938 * x3939;
x3913[x3929] = x3940;

}

}

}

}
int32_t x3950 = 0;
int32_t x3951 = 1;
x3951 *= 1;
x3950 += 1;
x3951 *= 1;
x3951 *= 1;
int32_t x3956 = x3950;
bool x3957 = x3956 >= 2;
if (x3957) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3962 = x3956 == 0;
if (x3962) {
int32_t x3963 = x3951;
bool x3964 = x3963 == 256;
if (x3964) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3971 = x3951;
bool x3973 = x3902 == 1;
int32_t x3972 = 256 / x3971;
bool x3974 = x3972 == 1;
bool x3978;
if (x454) {
bool x3975 = x3973 || x3974;
bool x3976 = x3902 == x3972;
bool x3977 = x3975 || x3976;
x3978 = x3977;
} else {
x3978 = false;
}
bool x3982;
if (x3978) {
x3982 = x3981;
} else {
x3982 = false;
}
bool x3983;
if (x3982) {
x3983 = x3981;
} else {
x3983 = false;
}
if (x3983) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3902,x3904,x3904,1,x3972,1,1);
assert(false && "");
}
bool x3989 = x3902 <= x3972;
int32_t x3990;
if (x3989) {
x3990 = x3972;
} else {
x3990 = x3902;
}
int32_t x3999 = x3990 * x3998;
int32_t x4000 = 64 * x3999;
float* x4001 = (float*)myMalloc(x4000 * sizeof(float));;
int32_t x4002;
if (x3973) {
x4002 = 0;
} else {
x4002 = x3910;
}
int32_t x4005;
if (x3974) {
x4005 = 0;
} else {
x4005 = 1;
}
for(int x4006=0; x4006 < 64; x4006++) {
int32_t x4018 = x3911 * x4006;
int32_t x4012 = x3999 * x4006;
for(int x4008=0; x4008 < x3990; x4008++) {
int32_t x4019 = x4002 * x4008;
int32_t x4020 = x4018 + x4019;
int32_t x4025 = x4005 * x4008;
int32_t x4014 = x3998 * x4008;
for(int x4010=0; x4010 < x3992; x4010++) {
int32_t x4021 = x4003 * x4010;
int32_t x4022 = x4020 + x4021;
int32_t x4016 = x3992 * x4010;
for(int x4011=0; x4011 < x3992; x4011++) {
int32_t x4023 = x4004 * x4011;
int32_t x4024 = x4022 + x4023;
float x4026 = x3913[x4024];
float x4027 = x7[x4025];
int32_t x4013 = x4011 + x4012;
int32_t x4015 = x4013 + x4014;
int32_t x4017 = x4015 + x4016;
float x4028 = x4026 + x4027;
x4001[x4017] = x4028;

}

}

}

}
bool x4038 = x3990 == 1;
bool x4039 = x4038 || x2621;
bool x4040 = x3990 == x2203;
bool x4041 = x4039 || x4040;
bool x4046;
if (x4041) {
x4046 = x4045;
} else {
x4046 = false;
}
bool x4047;
if (x4046) {
x4047 = x4045;
} else {
x4047 = false;
}
if (x4047) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3990,x3992,x3992,64,x2203,x2205,x2205);
assert(false && "");
}
bool x4053 = x3990 <= x2203;
int32_t x4054;
if (x4053) {
x4054 = x2203;
} else {
x4054 = x3990;
}
int32_t x4070;
if (x4038) {
x4070 = 0;
} else {
x4070 = x3998;
}
for(int x4073=0; x4073 < 64; x4073++) {
int32_t x4079 = x3999 * x4073;
int32_t x4086 = x2212 * x4073;
for(int x4075=0; x4075 < x4054; x4075++) {
int32_t x4080 = x4070 * x4075;
int32_t x4081 = x4079 + x4080;
int32_t x4087 = x2655 * x4075;
int32_t x4088 = x4086 + x4087;
for(int x4077=0; x4077 < x4056; x4077++) {
int32_t x4082 = x4071 * x4077;
int32_t x4083 = x4081 + x4082;
int32_t x4089 = x2656 * x4077;
int32_t x4090 = x4088 + x4089;
for(int x4078=0; x4078 < x4056; x4078++) {
int32_t x4084 = x4072 * x4078;
int32_t x4085 = x4083 + x4084;
float x4093 = x4001[x4085];
int32_t x4091 = x2657 * x4078;
int32_t x4092 = x4090 + x4091;
float x4094 = x2693[x4092];
float x4095 = x4093 + x4094;
x4001[x4085] = x4095;

}

}

}

}
float* x4105 = (float*)myMalloc(x4000 * sizeof(float));;
for(int x4107=0; x4107 < x4000; x4107++) {
float x4108 = x4001[x4107];
bool x4109 = x4108 < 0.0f;
if (x4109) {
x4105[x4107] = 0.0f;
} else {
float x4112 = x4001[x4107];
x4105[x4107] = x4112;
}

}
float* x4126 = (float*)myMalloc(x4125 * sizeof(float));;
int32_t x4129 = 64 * x3990;
int32_t x4130 = x4129 * x4121;
float* x4131 = (float*)myMalloc(x4130 * sizeof(float));;
int32_t x4127 = x3990 * x4121;
for(int x4132=0; x4132 < 64; x4132++) {
int32_t x4133 = x4132 * x3999;
float* x4134 = x4105+x4133;
int32_t x4135 = x4132 * x4122;
float* x4136 = x4126+x4135;
int32_t x4137 = x4132 * x4127;
float* x4138 = x4131+x4137;
for(int x4139=0; x4139 < x3990; x4139++) {
int32_t x4140 = x4139 / 1;
int32_t x4144 = x4140 * x4120;
int32_t x4145 = x4144 * x4120;
int32_t x4141 = x4139 % 1;
int32_t x4142 = x4141 / 1;
int32_t x4146 = x4142 * x4120;
int32_t x4147 = x4146 * x4120;
int32_t x4148 = x4145 + x4147;
int32_t x4143 = x4141 % 1;
int32_t x4149 = x4143 * x4120;
int32_t x4150 = x4149 * x4120;
int32_t x4151 = x4148 + x4150;
float* x4152 = x4138+x4151;
int32_t x4153 = x4140 * x3992;
int32_t x4154 = x4153 * x3992;
float* x4155 = x4134+x4154;
for(int x4157=0; x4157 < x4120; x4157++) {
int32_t x4159 = x4157 * x4120;
float* x4160 = x4152+x4159;
int32_t x4158 = x4157 + x4142;
int32_t x4161 = x4158 * x3992;
int32_t x4162 = x4161 + x4143;
float* x4163 = x4155+x4162;
memcpy(x4160, x4163, 4 * x4120);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x4121,x3990,1,x150,x3990,x4138,x4121,1,x4136,x4121);

}
int32_t x4172 = 0;
int32_t x4173 = 1;
x4173 *= 1;
x4172 += 1;
x4173 *= 1;
x4173 *= 1;
int32_t x4178 = x4172;
bool x4179 = x4178 >= 2;
if (x4179) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4184 = x4178 == 0;
if (x4184) {
int32_t x4185 = x4173;
bool x4186 = x4185 == 64;
if (x4186) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4193 = x4173;
int32_t x4194 = 64 / x4193;
bool x4195 = x4194 == 1;
bool x4198;
if (x454) {
bool x4196 = 64 == x4194;
bool x4197 = x4195 || x4196;
x4198 = x4197;
} else {
x4198 = false;
}
bool x4202;
if (x4198) {
x4202 = x4201;
} else {
x4202 = false;
}
bool x4203;
if (x4202) {
x4203 = x4201;
} else {
x4203 = false;
}
if (x4203) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x4120,x4120,1,x4194,1,1);
assert(false && "");
}
bool x4209 = 64 <= x4194;
int32_t x4210;
if (x4209) {
x4210 = x4194;
} else {
x4210 = 64;
}
int32_t x4219 = x4210 * x4218;
int32_t x4220 = 64 * x4219;
float* x4221 = (float*)myMalloc(x4220 * sizeof(float));;
int32_t x4224;
if (x4195) {
x4224 = 0;
} else {
x4224 = 1;
}
for(int x4225=0; x4225 < 64; x4225++) {
int32_t x4237 = x4122 * x4225;
int32_t x4231 = x4219 * x4225;
for(int x4227=0; x4227 < x4210; x4227++) {
int32_t x4238 = x4121 * x4227;
int32_t x4239 = x4237 + x4238;
int32_t x4244 = x4224 * x4227;
int32_t x4233 = x4218 * x4227;
for(int x4229=0; x4229 < x4212; x4229++) {
int32_t x4240 = x4222 * x4229;
int32_t x4241 = x4239 + x4240;
int32_t x4235 = x4212 * x4229;
for(int x4230=0; x4230 < x4212; x4230++) {
int32_t x4242 = x4223 * x4230;
int32_t x4243 = x4241 + x4242;
float x4245 = x4126[x4243];
float x4246 = x257[x4244];
int32_t x4232 = x4230 + x4231;
int32_t x4234 = x4232 + x4233;
int32_t x4236 = x4234 + x4235;
float x4247 = x4245 - x4246;
x4221[x4236] = x4247;

}

}

}

}
float* x4257 = (float*)myMalloc(64 * sizeof(float));;
for(int x4258=0; x4258 < 64; x4258++) {
float x4259 = x187[x4258];
float x4260 = x4259 + 1.0E-5f;
x4257[x4258] = x4260;

}
float* x4264 = (float*)myMalloc(64 * sizeof(float));;
for(int x4265=0; x4265 < 64; x4265++) {
float x4266 = x4257[x4265];
double x4267 = (double)x4266;
double x4268 = sqrt(x4267);
float x4269 = (float)x4268;
x4264[x4265] = x4269;

}
int32_t x4273 = 0;
int32_t x4274 = 1;
x4274 *= 1;
x4273 += 1;
x4274 *= 1;
x4274 *= 1;
int32_t x4279 = x4273;
bool x4280 = x4279 >= 2;
if (x4280) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4285 = x4279 == 0;
if (x4285) {
int32_t x4286 = x4274;
bool x4287 = x4286 == 64;
if (x4287) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4294 = x4274;
bool x4296 = x4210 == 1;
int32_t x4295 = 64 / x4294;
bool x4297 = x4295 == 1;
bool x4301;
if (x454) {
bool x4298 = x4296 || x4297;
bool x4299 = x4210 == x4295;
bool x4300 = x4298 || x4299;
x4301 = x4300;
} else {
x4301 = false;
}
bool x4305;
if (x4301) {
x4305 = x4304;
} else {
x4305 = false;
}
bool x4306;
if (x4305) {
x4306 = x4304;
} else {
x4306 = false;
}
if (x4306) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4210,x4212,x4212,1,x4295,1,1);
assert(false && "");
}
bool x4312 = x4210 <= x4295;
int32_t x4313;
if (x4312) {
x4313 = x4295;
} else {
x4313 = x4210;
}
int32_t x4322 = x4313 * x4321;
int32_t x4323 = 64 * x4322;
float* x4324 = (float*)myMalloc(x4323 * sizeof(float));;
int32_t x4325;
if (x4296) {
x4325 = 0;
} else {
x4325 = x4218;
}
int32_t x4328;
if (x4297) {
x4328 = 0;
} else {
x4328 = 1;
}
for(int x4329=0; x4329 < 64; x4329++) {
int32_t x4341 = x4219 * x4329;
int32_t x4335 = x4322 * x4329;
for(int x4331=0; x4331 < x4313; x4331++) {
int32_t x4342 = x4325 * x4331;
int32_t x4343 = x4341 + x4342;
int32_t x4348 = x4328 * x4331;
int32_t x4337 = x4321 * x4331;
for(int x4333=0; x4333 < x4315; x4333++) {
int32_t x4344 = x4326 * x4333;
int32_t x4345 = x4343 + x4344;
int32_t x4339 = x4315 * x4333;
for(int x4334=0; x4334 < x4315; x4334++) {
int32_t x4346 = x4327 * x4334;
int32_t x4347 = x4345 + x4346;
float x4349 = x4221[x4347];
float x4350 = x4264[x4348];
int32_t x4336 = x4334 + x4335;
int32_t x4338 = x4336 + x4337;
int32_t x4340 = x4338 + x4339;
float x4351 = x4349 / x4350;
x4324[x4340] = x4351;

}

}

}

}
int32_t x4361 = 0;
int32_t x4362 = 1;
x4362 *= 1;
x4361 += 1;
x4362 *= 1;
x4362 *= 1;
int32_t x4367 = x4361;
bool x4368 = x4367 >= 2;
if (x4368) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4373 = x4367 == 0;
if (x4373) {
int32_t x4374 = x4362;
bool x4375 = x4374 == 64;
if (x4375) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4382 = x4362;
bool x4384 = x4313 == 1;
int32_t x4383 = 64 / x4382;
bool x4385 = x4383 == 1;
bool x4389;
if (x454) {
bool x4386 = x4384 || x4385;
bool x4387 = x4313 == x4383;
bool x4388 = x4386 || x4387;
x4389 = x4388;
} else {
x4389 = false;
}
bool x4393;
if (x4389) {
x4393 = x4392;
} else {
x4393 = false;
}
bool x4394;
if (x4393) {
x4394 = x4392;
} else {
x4394 = false;
}
if (x4394) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4313,x4315,x4315,1,x4383,1,1);
assert(false && "");
}
bool x4400 = x4313 <= x4383;
int32_t x4401;
if (x4400) {
x4401 = x4383;
} else {
x4401 = x4313;
}
int32_t x4410 = x4401 * x4409;
int32_t x4411 = 64 * x4410;
float* x4412 = (float*)myMalloc(x4411 * sizeof(float));;
int32_t x4413;
if (x4384) {
x4413 = 0;
} else {
x4413 = x4321;
}
int32_t x4416;
if (x4385) {
x4416 = 0;
} else {
x4416 = 1;
}
for(int x4417=0; x4417 < 64; x4417++) {
int32_t x4429 = x4322 * x4417;
int32_t x4423 = x4410 * x4417;
for(int x4419=0; x4419 < x4401; x4419++) {
int32_t x4430 = x4413 * x4419;
int32_t x4431 = x4429 + x4430;
int32_t x4436 = x4416 * x4419;
int32_t x4425 = x4409 * x4419;
for(int x4421=0; x4421 < x4403; x4421++) {
int32_t x4432 = x4414 * x4421;
int32_t x4433 = x4431 + x4432;
int32_t x4427 = x4403 * x4421;
for(int x4422=0; x4422 < x4403; x4422++) {
int32_t x4434 = x4415 * x4422;
int32_t x4435 = x4433 + x4434;
float x4437 = x4324[x4435];
float x4438 = x81[x4436];
int32_t x4424 = x4422 + x4423;
int32_t x4426 = x4424 + x4425;
int32_t x4428 = x4426 + x4427;
float x4439 = x4437 * x4438;
x4412[x4428] = x4439;

}

}

}

}
int32_t x4449 = 0;
int32_t x4450 = 1;
x4450 *= 1;
x4449 += 1;
x4450 *= 1;
x4450 *= 1;
int32_t x4455 = x4449;
bool x4456 = x4455 >= 2;
if (x4456) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4461 = x4455 == 0;
if (x4461) {
int32_t x4462 = x4450;
bool x4463 = x4462 == 64;
if (x4463) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4470 = x4450;
bool x4472 = x4401 == 1;
int32_t x4471 = 64 / x4470;
bool x4473 = x4471 == 1;
bool x4477;
if (x454) {
bool x4474 = x4472 || x4473;
bool x4475 = x4401 == x4471;
bool x4476 = x4474 || x4475;
x4477 = x4476;
} else {
x4477 = false;
}
bool x4481;
if (x4477) {
x4481 = x4480;
} else {
x4481 = false;
}
bool x4482;
if (x4481) {
x4482 = x4480;
} else {
x4482 = false;
}
if (x4482) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4401,x4403,x4403,1,x4471,1,1);
assert(false && "");
}
bool x4488 = x4401 <= x4471;
int32_t x4489;
if (x4488) {
x4489 = x4471;
} else {
x4489 = x4401;
}
int32_t x4498 = x4489 * x4497;
int32_t x4499 = 64 * x4498;
float* x4500 = (float*)myMalloc(x4499 * sizeof(float));;
int32_t x4501;
if (x4472) {
x4501 = 0;
} else {
x4501 = x4409;
}
int32_t x4504;
if (x4473) {
x4504 = 0;
} else {
x4504 = 1;
}
for(int x4505=0; x4505 < 64; x4505++) {
int32_t x4517 = x4410 * x4505;
int32_t x4511 = x4498 * x4505;
for(int x4507=0; x4507 < x4489; x4507++) {
int32_t x4518 = x4501 * x4507;
int32_t x4519 = x4517 + x4518;
int32_t x4524 = x4504 * x4507;
int32_t x4513 = x4497 * x4507;
for(int x4509=0; x4509 < x4491; x4509++) {
int32_t x4520 = x4502 * x4509;
int32_t x4521 = x4519 + x4520;
int32_t x4515 = x4491 * x4509;
for(int x4510=0; x4510 < x4491; x4510++) {
int32_t x4522 = x4503 * x4510;
int32_t x4523 = x4521 + x4522;
float x4525 = x4412[x4523];
float x4526 = x24[x4524];
int32_t x4512 = x4510 + x4511;
int32_t x4514 = x4512 + x4513;
int32_t x4516 = x4514 + x4515;
float x4527 = x4525 + x4526;
x4500[x4516] = x4527;

}

}

}

}
float* x4537 = (float*)myMalloc(x4499 * sizeof(float));;
for(int x4539=0; x4539 < x4499; x4539++) {
float x4540 = x4500[x4539];
bool x4541 = x4540 < 0.0f;
if (x4541) {
x4537[x4539] = 0.0f;
} else {
float x4544 = x4500[x4539];
x4537[x4539] = x4544;
}

}
float* x4559 = (float*)myMalloc(x4558 * sizeof(float));;
int32_t x4560 = 9 * x4489;
int32_t x4563 = 64 * x4560;
int32_t x4564 = x4563 * x4554;
float* x4565 = (float*)myMalloc(x4564 * sizeof(float));;
int32_t x4561 = x4560 * x4554;
int32_t x4573 = x4489 * 3;
int32_t x4574 = x4573 * 3;
for(int x4566=0; x4566 < 64; x4566++) {
int32_t x4567 = x4566 * x4498;
float* x4568 = x4537+x4567;
int32_t x4569 = x4566 * x4555;
float* x4570 = x4559+x4569;
int32_t x4571 = x4566 * x4561;
float* x4572 = x4565+x4571;
for(int x4576=0; x4576 < x4574; x4576++) {
int32_t x4577 = x4576 / 9;
int32_t x4581 = x4577 * 3;
int32_t x4582 = x4581 * 3;
int32_t x4583 = x4582 * x4553;
int32_t x4584 = x4583 * x4553;
int32_t x4578 = x4576 % 9;
int32_t x4579 = x4578 / 3;
int32_t x4585 = x4579 * 3;
int32_t x4586 = x4585 * x4553;
int32_t x4587 = x4586 * x4553;
int32_t x4588 = x4584 + x4587;
int32_t x4580 = x4578 % 3;
int32_t x4589 = x4580 * x4553;
int32_t x4590 = x4589 * x4553;
int32_t x4591 = x4588 + x4590;
float* x4592 = x4572+x4591;
int32_t x4593 = x4577 * x4491;
int32_t x4594 = x4593 * x4491;
float* x4595 = x4568+x4594;
int32_t x4608 = 1 - x4580;
bool x4609 = x4608 > 0;
int32_t x4610;
if (x4609) {
x4610 = x4608;
} else {
x4610 = 0;
}
int32_t x4611 = 3 - x4580;
int32_t x4612 = x4611 - 1;
int32_t x4613 = 1 - x4612;
bool x4614 = x4613 > 0;
int32_t x4615;
if (x4614) {
x4615 = x4613;
} else {
x4615 = 0;
}
int32_t x4616 = x4553 - x4615;
int32_t x4617 = x4616 - x4610;
bool x4618 = x4617 <= 0;
bool x4622 = x4610 > 0;
int32_t x4607 = -1 + x4580;
bool x4635 = x4615 > 0;
for(int x4597=0; x4597 < x4553; x4597++) {
int32_t x4598 = x4597 - 1;
int32_t x4599 = x4598 + x4579;
bool x4600 = x4599 < 0;
bool x4601 = x4599 >= x4491;
bool x4602 = x4600 || x4601;
if (x4602) {
int32_t x4603 = x4597 * x4553;
float* x4604 = x4592+x4603;
memset(x4604, 0, 4 * x4553);;
} else {
if (x4618) {
int32_t x4603 = x4597 * x4553;
float* x4619 = x4592+x4603;
memset(x4619, 0, 4 * x4553);;
} else {
int32_t x4603 = x4597 * x4553;
if (x4622) {
float* x4623 = x4592+x4603;
memset(x4623, 0, 4 * x4610);;
} else {
}
// may have segfault here
int32_t x4628 = x4603 + x4610;
float* x4629 = x4592+x4628;
int32_t x4630 = x4599 * x4491;
int32_t x4631 = x4630 + x4607;
int32_t x4632 = x4631 + x4610;
float* x4633 = x4595+x4632;
memcpy(x4629, x4633, 4 * x4617);;
if (x4635) {
int32_t x4636 = x4603 + x4553;
int32_t x4637 = x4636 - x4615;
float* x4638 = x4592+x4637;
memset(x4638, 0, 4 * x4615);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x4554,x4560,1,x73,x4560,x4572,x4554,1,x4570,x4554);

}
int32_t x4653 = 0;
int32_t x4654 = 1;
x4654 *= 1;
x4653 += 1;
x4654 *= 1;
x4654 *= 1;
int32_t x4659 = x4653;
bool x4660 = x4659 >= 2;
if (x4660) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4665 = x4659 == 0;
if (x4665) {
int32_t x4666 = x4654;
bool x4667 = x4666 == 64;
if (x4667) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4674 = x4654;
int32_t x4675 = 64 / x4674;
bool x4676 = x4675 == 1;
bool x4679;
if (x454) {
bool x4677 = 64 == x4675;
bool x4678 = x4676 || x4677;
x4679 = x4678;
} else {
x4679 = false;
}
bool x4683;
if (x4679) {
x4683 = x4682;
} else {
x4683 = false;
}
bool x4684;
if (x4683) {
x4684 = x4682;
} else {
x4684 = false;
}
if (x4684) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x4553,x4553,1,x4675,1,1);
assert(false && "");
}
bool x4690 = 64 <= x4675;
int32_t x4691;
if (x4690) {
x4691 = x4675;
} else {
x4691 = 64;
}
int32_t x4700 = x4691 * x4699;
int32_t x4701 = 64 * x4700;
float* x4702 = (float*)myMalloc(x4701 * sizeof(float));;
int32_t x4705;
if (x4676) {
x4705 = 0;
} else {
x4705 = 1;
}
for(int x4706=0; x4706 < 64; x4706++) {
int32_t x4718 = x4555 * x4706;
int32_t x4712 = x4700 * x4706;
for(int x4708=0; x4708 < x4691; x4708++) {
int32_t x4719 = x4554 * x4708;
int32_t x4720 = x4718 + x4719;
int32_t x4725 = x4705 * x4708;
int32_t x4714 = x4699 * x4708;
for(int x4710=0; x4710 < x4693; x4710++) {
int32_t x4721 = x4703 * x4710;
int32_t x4722 = x4720 + x4721;
int32_t x4716 = x4693 * x4710;
for(int x4711=0; x4711 < x4693; x4711++) {
int32_t x4723 = x4704 * x4711;
int32_t x4724 = x4722 + x4723;
float x4726 = x4559[x4724];
float x4727 = x179[x4725];
int32_t x4713 = x4711 + x4712;
int32_t x4715 = x4713 + x4714;
int32_t x4717 = x4715 + x4716;
float x4728 = x4726 - x4727;
x4702[x4717] = x4728;

}

}

}

}
float* x4738 = (float*)myMalloc(64 * sizeof(float));;
for(int x4739=0; x4739 < 64; x4739++) {
float x4740 = x118[x4739];
float x4741 = x4740 + 1.0E-5f;
x4738[x4739] = x4741;

}
float* x4745 = (float*)myMalloc(64 * sizeof(float));;
for(int x4746=0; x4746 < 64; x4746++) {
float x4747 = x4738[x4746];
double x4748 = (double)x4747;
double x4749 = sqrt(x4748);
float x4750 = (float)x4749;
x4745[x4746] = x4750;

}
int32_t x4754 = 0;
int32_t x4755 = 1;
x4755 *= 1;
x4754 += 1;
x4755 *= 1;
x4755 *= 1;
int32_t x4760 = x4754;
bool x4761 = x4760 >= 2;
if (x4761) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4766 = x4760 == 0;
if (x4766) {
int32_t x4767 = x4755;
bool x4768 = x4767 == 64;
if (x4768) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4775 = x4755;
bool x4777 = x4691 == 1;
int32_t x4776 = 64 / x4775;
bool x4778 = x4776 == 1;
bool x4782;
if (x454) {
bool x4779 = x4777 || x4778;
bool x4780 = x4691 == x4776;
bool x4781 = x4779 || x4780;
x4782 = x4781;
} else {
x4782 = false;
}
bool x4786;
if (x4782) {
x4786 = x4785;
} else {
x4786 = false;
}
bool x4787;
if (x4786) {
x4787 = x4785;
} else {
x4787 = false;
}
if (x4787) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4691,x4693,x4693,1,x4776,1,1);
assert(false && "");
}
bool x4793 = x4691 <= x4776;
int32_t x4794;
if (x4793) {
x4794 = x4776;
} else {
x4794 = x4691;
}
int32_t x4803 = x4794 * x4802;
int32_t x4804 = 64 * x4803;
float* x4805 = (float*)myMalloc(x4804 * sizeof(float));;
int32_t x4806;
if (x4777) {
x4806 = 0;
} else {
x4806 = x4699;
}
int32_t x4809;
if (x4778) {
x4809 = 0;
} else {
x4809 = 1;
}
for(int x4810=0; x4810 < 64; x4810++) {
int32_t x4822 = x4700 * x4810;
int32_t x4816 = x4803 * x4810;
for(int x4812=0; x4812 < x4794; x4812++) {
int32_t x4823 = x4806 * x4812;
int32_t x4824 = x4822 + x4823;
int32_t x4829 = x4809 * x4812;
int32_t x4818 = x4802 * x4812;
for(int x4814=0; x4814 < x4796; x4814++) {
int32_t x4825 = x4807 * x4814;
int32_t x4826 = x4824 + x4825;
int32_t x4820 = x4796 * x4814;
for(int x4815=0; x4815 < x4796; x4815++) {
int32_t x4827 = x4808 * x4815;
int32_t x4828 = x4826 + x4827;
float x4830 = x4702[x4828];
float x4831 = x4745[x4829];
int32_t x4817 = x4815 + x4816;
int32_t x4819 = x4817 + x4818;
int32_t x4821 = x4819 + x4820;
float x4832 = x4830 / x4831;
x4805[x4821] = x4832;

}

}

}

}
int32_t x4842 = 0;
int32_t x4843 = 1;
x4843 *= 1;
x4842 += 1;
x4843 *= 1;
x4843 *= 1;
int32_t x4848 = x4842;
bool x4849 = x4848 >= 2;
if (x4849) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4854 = x4848 == 0;
if (x4854) {
int32_t x4855 = x4843;
bool x4856 = x4855 == 64;
if (x4856) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4863 = x4843;
bool x4865 = x4794 == 1;
int32_t x4864 = 64 / x4863;
bool x4866 = x4864 == 1;
bool x4870;
if (x454) {
bool x4867 = x4865 || x4866;
bool x4868 = x4794 == x4864;
bool x4869 = x4867 || x4868;
x4870 = x4869;
} else {
x4870 = false;
}
bool x4874;
if (x4870) {
x4874 = x4873;
} else {
x4874 = false;
}
bool x4875;
if (x4874) {
x4875 = x4873;
} else {
x4875 = false;
}
if (x4875) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4794,x4796,x4796,1,x4864,1,1);
assert(false && "");
}
bool x4881 = x4794 <= x4864;
int32_t x4882;
if (x4881) {
x4882 = x4864;
} else {
x4882 = x4794;
}
int32_t x4891 = x4882 * x4890;
int32_t x4892 = 64 * x4891;
float* x4893 = (float*)myMalloc(x4892 * sizeof(float));;
int32_t x4894;
if (x4865) {
x4894 = 0;
} else {
x4894 = x4802;
}
int32_t x4897;
if (x4866) {
x4897 = 0;
} else {
x4897 = 1;
}
for(int x4898=0; x4898 < 64; x4898++) {
int32_t x4910 = x4803 * x4898;
int32_t x4904 = x4891 * x4898;
for(int x4900=0; x4900 < x4882; x4900++) {
int32_t x4911 = x4894 * x4900;
int32_t x4912 = x4910 + x4911;
int32_t x4917 = x4897 * x4900;
int32_t x4906 = x4890 * x4900;
for(int x4902=0; x4902 < x4884; x4902++) {
int32_t x4913 = x4895 * x4902;
int32_t x4914 = x4912 + x4913;
int32_t x4908 = x4884 * x4902;
for(int x4903=0; x4903 < x4884; x4903++) {
int32_t x4915 = x4896 * x4903;
int32_t x4916 = x4914 + x4915;
float x4918 = x4805[x4916];
float x4919 = x72[x4917];
int32_t x4905 = x4903 + x4904;
int32_t x4907 = x4905 + x4906;
int32_t x4909 = x4907 + x4908;
float x4920 = x4918 * x4919;
x4893[x4909] = x4920;

}

}

}

}
int32_t x4930 = 0;
int32_t x4931 = 1;
x4931 *= 1;
x4930 += 1;
x4931 *= 1;
x4931 *= 1;
int32_t x4936 = x4930;
bool x4937 = x4936 >= 2;
if (x4937) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4942 = x4936 == 0;
if (x4942) {
int32_t x4943 = x4931;
bool x4944 = x4943 == 64;
if (x4944) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4951 = x4931;
bool x4953 = x4882 == 1;
int32_t x4952 = 64 / x4951;
bool x4954 = x4952 == 1;
bool x4958;
if (x454) {
bool x4955 = x4953 || x4954;
bool x4956 = x4882 == x4952;
bool x4957 = x4955 || x4956;
x4958 = x4957;
} else {
x4958 = false;
}
bool x4962;
if (x4958) {
x4962 = x4961;
} else {
x4962 = false;
}
bool x4963;
if (x4962) {
x4963 = x4961;
} else {
x4963 = false;
}
if (x4963) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4882,x4884,x4884,1,x4952,1,1);
assert(false && "");
}
bool x4969 = x4882 <= x4952;
int32_t x4970;
if (x4969) {
x4970 = x4952;
} else {
x4970 = x4882;
}
int32_t x4979 = x4970 * x4978;
int32_t x4980 = 64 * x4979;
float* x4981 = (float*)myMalloc(x4980 * sizeof(float));;
int32_t x4982;
if (x4953) {
x4982 = 0;
} else {
x4982 = x4890;
}
int32_t x4985;
if (x4954) {
x4985 = 0;
} else {
x4985 = 1;
}
for(int x4986=0; x4986 < 64; x4986++) {
int32_t x4998 = x4891 * x4986;
int32_t x4992 = x4979 * x4986;
for(int x4988=0; x4988 < x4970; x4988++) {
int32_t x4999 = x4982 * x4988;
int32_t x5000 = x4998 + x4999;
int32_t x5005 = x4985 * x4988;
int32_t x4994 = x4978 * x4988;
for(int x4990=0; x4990 < x4972; x4990++) {
int32_t x5001 = x4983 * x4990;
int32_t x5002 = x5000 + x5001;
int32_t x4996 = x4972 * x4990;
for(int x4991=0; x4991 < x4972; x4991++) {
int32_t x5003 = x4984 * x4991;
int32_t x5004 = x5002 + x5003;
float x5006 = x4893[x5004];
float x5007 = x135[x5005];
int32_t x4993 = x4991 + x4992;
int32_t x4995 = x4993 + x4994;
int32_t x4997 = x4995 + x4996;
float x5008 = x5006 + x5007;
x4981[x4997] = x5008;

}

}

}

}
float* x5018 = (float*)myMalloc(x4980 * sizeof(float));;
for(int x5020=0; x5020 < x4980; x5020++) {
float x5021 = x4981[x5020];
bool x5022 = x5021 < 0.0f;
if (x5022) {
x5018[x5020] = 0.0f;
} else {
float x5025 = x4981[x5020];
x5018[x5020] = x5025;
}

}
float* x5039 = (float*)myMalloc(x5038 * sizeof(float));;
int32_t x5042 = 64 * x4970;
int32_t x5043 = x5042 * x5034;
float* x5044 = (float*)myMalloc(x5043 * sizeof(float));;
int32_t x5040 = x4970 * x5034;
for(int x5045=0; x5045 < 64; x5045++) {
int32_t x5046 = x5045 * x4979;
float* x5047 = x5018+x5046;
int32_t x5048 = x5045 * x5035;
float* x5049 = x5039+x5048;
int32_t x5050 = x5045 * x5040;
float* x5051 = x5044+x5050;
for(int x5052=0; x5052 < x4970; x5052++) {
int32_t x5053 = x5052 / 1;
int32_t x5057 = x5053 * x5033;
int32_t x5058 = x5057 * x5033;
int32_t x5054 = x5052 % 1;
int32_t x5055 = x5054 / 1;
int32_t x5059 = x5055 * x5033;
int32_t x5060 = x5059 * x5033;
int32_t x5061 = x5058 + x5060;
int32_t x5056 = x5054 % 1;
int32_t x5062 = x5056 * x5033;
int32_t x5063 = x5062 * x5033;
int32_t x5064 = x5061 + x5063;
float* x5065 = x5051+x5064;
int32_t x5066 = x5053 * x4972;
int32_t x5067 = x5066 * x4972;
float* x5068 = x5047+x5067;
for(int x5070=0; x5070 < x5033; x5070++) {
int32_t x5072 = x5070 * x5033;
float* x5073 = x5065+x5072;
int32_t x5071 = x5070 + x5055;
int32_t x5074 = x5071 * x4972;
int32_t x5075 = x5074 + x5056;
float* x5076 = x5068+x5075;
memcpy(x5073, x5076, 4 * x5033);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x5034,x4970,1,x87,x4970,x5051,x5034,1,x5049,x5034);

}
int32_t x5085 = 0;
int32_t x5086 = 1;
x5086 *= 1;
x5085 += 1;
x5086 *= 1;
x5086 *= 1;
int32_t x5091 = x5085;
bool x5092 = x5091 >= 2;
if (x5092) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5097 = x5091 == 0;
if (x5097) {
int32_t x5098 = x5086;
bool x5099 = x5098 == 256;
if (x5099) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5106 = x5086;
int32_t x5107 = 256 / x5106;
bool x5108 = x5107 == 1;
bool x5111;
if (x454) {
bool x5109 = 256 == x5107;
bool x5110 = x5108 || x5109;
x5111 = x5110;
} else {
x5111 = false;
}
bool x5115;
if (x5111) {
x5115 = x5114;
} else {
x5115 = false;
}
bool x5116;
if (x5115) {
x5116 = x5114;
} else {
x5116 = false;
}
if (x5116) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x5033,x5033,1,x5107,1,1);
assert(false && "");
}
bool x5122 = 256 <= x5107;
int32_t x5123;
if (x5122) {
x5123 = x5107;
} else {
x5123 = 256;
}
int32_t x5132 = x5123 * x5131;
int32_t x5133 = 64 * x5132;
float* x5134 = (float*)myMalloc(x5133 * sizeof(float));;
int32_t x5137;
if (x5108) {
x5137 = 0;
} else {
x5137 = 1;
}
for(int x5138=0; x5138 < 64; x5138++) {
int32_t x5150 = x5035 * x5138;
int32_t x5144 = x5132 * x5138;
for(int x5140=0; x5140 < x5123; x5140++) {
int32_t x5151 = x5034 * x5140;
int32_t x5152 = x5150 + x5151;
int32_t x5157 = x5137 * x5140;
int32_t x5146 = x5131 * x5140;
for(int x5142=0; x5142 < x5125; x5142++) {
int32_t x5153 = x5135 * x5142;
int32_t x5154 = x5152 + x5153;
int32_t x5148 = x5125 * x5142;
for(int x5143=0; x5143 < x5125; x5143++) {
int32_t x5155 = x5136 * x5143;
int32_t x5156 = x5154 + x5155;
float x5158 = x5039[x5156];
float x5159 = x184[x5157];
int32_t x5145 = x5143 + x5144;
int32_t x5147 = x5145 + x5146;
int32_t x5149 = x5147 + x5148;
float x5160 = x5158 - x5159;
x5134[x5149] = x5160;

}

}

}

}
float* x5170 = (float*)myMalloc(256 * sizeof(float));;
for(int x5171=0; x5171 < 256; x5171++) {
float x5172 = x133[x5171];
float x5173 = x5172 + 1.0E-5f;
x5170[x5171] = x5173;

}
float* x5177 = (float*)myMalloc(256 * sizeof(float));;
for(int x5178=0; x5178 < 256; x5178++) {
float x5179 = x5170[x5178];
double x5180 = (double)x5179;
double x5181 = sqrt(x5180);
float x5182 = (float)x5181;
x5177[x5178] = x5182;

}
int32_t x5186 = 0;
int32_t x5187 = 1;
x5187 *= 1;
x5186 += 1;
x5187 *= 1;
x5187 *= 1;
int32_t x5192 = x5186;
bool x5193 = x5192 >= 2;
if (x5193) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5198 = x5192 == 0;
if (x5198) {
int32_t x5199 = x5187;
bool x5200 = x5199 == 256;
if (x5200) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5207 = x5187;
bool x5209 = x5123 == 1;
int32_t x5208 = 256 / x5207;
bool x5210 = x5208 == 1;
bool x5214;
if (x454) {
bool x5211 = x5209 || x5210;
bool x5212 = x5123 == x5208;
bool x5213 = x5211 || x5212;
x5214 = x5213;
} else {
x5214 = false;
}
bool x5218;
if (x5214) {
x5218 = x5217;
} else {
x5218 = false;
}
bool x5219;
if (x5218) {
x5219 = x5217;
} else {
x5219 = false;
}
if (x5219) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5123,x5125,x5125,1,x5208,1,1);
assert(false && "");
}
bool x5225 = x5123 <= x5208;
int32_t x5226;
if (x5225) {
x5226 = x5208;
} else {
x5226 = x5123;
}
int32_t x5235 = x5226 * x5234;
int32_t x5236 = 64 * x5235;
float* x5237 = (float*)myMalloc(x5236 * sizeof(float));;
int32_t x5238;
if (x5209) {
x5238 = 0;
} else {
x5238 = x5131;
}
int32_t x5241;
if (x5210) {
x5241 = 0;
} else {
x5241 = 1;
}
for(int x5242=0; x5242 < 64; x5242++) {
int32_t x5254 = x5132 * x5242;
int32_t x5248 = x5235 * x5242;
for(int x5244=0; x5244 < x5226; x5244++) {
int32_t x5255 = x5238 * x5244;
int32_t x5256 = x5254 + x5255;
int32_t x5261 = x5241 * x5244;
int32_t x5250 = x5234 * x5244;
for(int x5246=0; x5246 < x5228; x5246++) {
int32_t x5257 = x5239 * x5246;
int32_t x5258 = x5256 + x5257;
int32_t x5252 = x5228 * x5246;
for(int x5247=0; x5247 < x5228; x5247++) {
int32_t x5259 = x5240 * x5247;
int32_t x5260 = x5258 + x5259;
float x5262 = x5134[x5260];
float x5263 = x5177[x5261];
int32_t x5249 = x5247 + x5248;
int32_t x5251 = x5249 + x5250;
int32_t x5253 = x5251 + x5252;
float x5264 = x5262 / x5263;
x5237[x5253] = x5264;

}

}

}

}
int32_t x5274 = 0;
int32_t x5275 = 1;
x5275 *= 1;
x5274 += 1;
x5275 *= 1;
x5275 *= 1;
int32_t x5280 = x5274;
bool x5281 = x5280 >= 2;
if (x5281) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5286 = x5280 == 0;
if (x5286) {
int32_t x5287 = x5275;
bool x5288 = x5287 == 256;
if (x5288) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5295 = x5275;
bool x5297 = x5226 == 1;
int32_t x5296 = 256 / x5295;
bool x5298 = x5296 == 1;
bool x5302;
if (x454) {
bool x5299 = x5297 || x5298;
bool x5300 = x5226 == x5296;
bool x5301 = x5299 || x5300;
x5302 = x5301;
} else {
x5302 = false;
}
bool x5306;
if (x5302) {
x5306 = x5305;
} else {
x5306 = false;
}
bool x5307;
if (x5306) {
x5307 = x5305;
} else {
x5307 = false;
}
if (x5307) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5226,x5228,x5228,1,x5296,1,1);
assert(false && "");
}
bool x5313 = x5226 <= x5296;
int32_t x5314;
if (x5313) {
x5314 = x5296;
} else {
x5314 = x5226;
}
int32_t x5323 = x5314 * x5322;
int32_t x5324 = 64 * x5323;
float* x5325 = (float*)myMalloc(x5324 * sizeof(float));;
int32_t x5326;
if (x5297) {
x5326 = 0;
} else {
x5326 = x5234;
}
int32_t x5329;
if (x5298) {
x5329 = 0;
} else {
x5329 = 1;
}
for(int x5330=0; x5330 < 64; x5330++) {
int32_t x5342 = x5235 * x5330;
int32_t x5336 = x5323 * x5330;
for(int x5332=0; x5332 < x5314; x5332++) {
int32_t x5343 = x5326 * x5332;
int32_t x5344 = x5342 + x5343;
int32_t x5349 = x5329 * x5332;
int32_t x5338 = x5322 * x5332;
for(int x5334=0; x5334 < x5316; x5334++) {
int32_t x5345 = x5327 * x5334;
int32_t x5346 = x5344 + x5345;
int32_t x5340 = x5316 * x5334;
for(int x5335=0; x5335 < x5316; x5335++) {
int32_t x5347 = x5328 * x5335;
int32_t x5348 = x5346 + x5347;
float x5350 = x5237[x5348];
float x5351 = x37[x5349];
int32_t x5337 = x5335 + x5336;
int32_t x5339 = x5337 + x5338;
int32_t x5341 = x5339 + x5340;
float x5352 = x5350 * x5351;
x5325[x5341] = x5352;

}

}

}

}
int32_t x5362 = 0;
int32_t x5363 = 1;
x5363 *= 1;
x5362 += 1;
x5363 *= 1;
x5363 *= 1;
int32_t x5368 = x5362;
bool x5369 = x5368 >= 2;
if (x5369) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5374 = x5368 == 0;
if (x5374) {
int32_t x5375 = x5363;
bool x5376 = x5375 == 256;
if (x5376) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5383 = x5363;
bool x5385 = x5314 == 1;
int32_t x5384 = 256 / x5383;
bool x5386 = x5384 == 1;
bool x5390;
if (x454) {
bool x5387 = x5385 || x5386;
bool x5388 = x5314 == x5384;
bool x5389 = x5387 || x5388;
x5390 = x5389;
} else {
x5390 = false;
}
bool x5394;
if (x5390) {
x5394 = x5393;
} else {
x5394 = false;
}
bool x5395;
if (x5394) {
x5395 = x5393;
} else {
x5395 = false;
}
if (x5395) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5314,x5316,x5316,1,x5384,1,1);
assert(false && "");
}
bool x5401 = x5314 <= x5384;
int32_t x5402;
if (x5401) {
x5402 = x5384;
} else {
x5402 = x5314;
}
int32_t x5411 = x5402 * x5410;
int32_t x5412 = 64 * x5411;
float* x5413 = (float*)myMalloc(x5412 * sizeof(float));;
int32_t x5414;
if (x5385) {
x5414 = 0;
} else {
x5414 = x5322;
}
int32_t x5417;
if (x5386) {
x5417 = 0;
} else {
x5417 = 1;
}
for(int x5418=0; x5418 < 64; x5418++) {
int32_t x5430 = x5323 * x5418;
int32_t x5424 = x5411 * x5418;
for(int x5420=0; x5420 < x5402; x5420++) {
int32_t x5431 = x5414 * x5420;
int32_t x5432 = x5430 + x5431;
int32_t x5437 = x5417 * x5420;
int32_t x5426 = x5410 * x5420;
for(int x5422=0; x5422 < x5404; x5422++) {
int32_t x5433 = x5415 * x5422;
int32_t x5434 = x5432 + x5433;
int32_t x5428 = x5404 * x5422;
for(int x5423=0; x5423 < x5404; x5423++) {
int32_t x5435 = x5416 * x5423;
int32_t x5436 = x5434 + x5435;
float x5438 = x5325[x5436];
float x5439 = x247[x5437];
int32_t x5425 = x5423 + x5424;
int32_t x5427 = x5425 + x5426;
int32_t x5429 = x5427 + x5428;
float x5440 = x5438 + x5439;
x5413[x5429] = x5440;

}

}

}

}
bool x5450 = x5402 == 1;
bool x5451 = x5450 || x4038;
bool x5452 = x5402 == x3990;
bool x5453 = x5451 || x5452;
bool x5458;
if (x5453) {
x5458 = x5457;
} else {
x5458 = false;
}
bool x5459;
if (x5458) {
x5459 = x5457;
} else {
x5459 = false;
}
if (x5459) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5402,x5404,x5404,64,x3990,x3992,x3992);
assert(false && "");
}
bool x5465 = x5402 <= x3990;
int32_t x5466;
if (x5465) {
x5466 = x3990;
} else {
x5466 = x5402;
}
int32_t x5482;
if (x5450) {
x5482 = 0;
} else {
x5482 = x5410;
}
for(int x5485=0; x5485 < 64; x5485++) {
int32_t x5491 = x5411 * x5485;
int32_t x5498 = x3999 * x5485;
for(int x5487=0; x5487 < x5466; x5487++) {
int32_t x5492 = x5482 * x5487;
int32_t x5493 = x5491 + x5492;
int32_t x5499 = x4070 * x5487;
int32_t x5500 = x5498 + x5499;
for(int x5489=0; x5489 < x5468; x5489++) {
int32_t x5494 = x5483 * x5489;
int32_t x5495 = x5493 + x5494;
int32_t x5501 = x4071 * x5489;
int32_t x5502 = x5500 + x5501;
for(int x5490=0; x5490 < x5468; x5490++) {
int32_t x5496 = x5484 * x5490;
int32_t x5497 = x5495 + x5496;
float x5505 = x5413[x5497];
int32_t x5503 = x4072 * x5490;
int32_t x5504 = x5502 + x5503;
float x5506 = x4105[x5504];
float x5507 = x5505 + x5506;
x5413[x5497] = x5507;

}

}

}

}
float* x5517 = (float*)myMalloc(x5412 * sizeof(float));;
for(int x5519=0; x5519 < x5412; x5519++) {
float x5520 = x5413[x5519];
bool x5521 = x5520 < 0.0f;
if (x5521) {
x5517[x5519] = 0.0f;
} else {
float x5524 = x5413[x5519];
x5517[x5519] = x5524;
}

}
float* x5538 = (float*)myMalloc(x5537 * sizeof(float));;
int32_t x5541 = 64 * x5402;
int32_t x5542 = x5541 * x5533;
float* x5543 = (float*)myMalloc(x5542 * sizeof(float));;
int32_t x5539 = x5402 * x5533;
for(int x5544=0; x5544 < 64; x5544++) {
int32_t x5545 = x5544 * x5411;
float* x5546 = x5517+x5545;
int32_t x5547 = x5544 * x5534;
float* x5548 = x5538+x5547;
int32_t x5549 = x5544 * x5539;
float* x5550 = x5543+x5549;
for(int x5551=0; x5551 < x5402; x5551++) {
int32_t x5552 = x5551 / 1;
int32_t x5556 = x5552 * x5532;
int32_t x5557 = x5556 * x5532;
int32_t x5553 = x5551 % 1;
int32_t x5554 = x5553 / 1;
int32_t x5558 = x5554 * x5532;
int32_t x5559 = x5558 * x5532;
int32_t x5560 = x5557 + x5559;
int32_t x5555 = x5553 % 1;
int32_t x5561 = x5555 * x5532;
int32_t x5562 = x5561 * x5532;
int32_t x5563 = x5560 + x5562;
float* x5564 = x5550+x5563;
int32_t x5565 = x5552 * x5404;
int32_t x5566 = x5565 * x5404;
float* x5567 = x5546+x5566;
for(int x5569=0; x5569 < x5532; x5569++) {
int32_t x5571 = x5569 * x5532;
float* x5572 = x5564+x5571;
int32_t x5570 = x5569 + x5554;
int32_t x5573 = x5570 * x5404;
int32_t x5574 = x5573 + x5555;
float* x5575 = x5567+x5574;
memcpy(x5572, x5575, 4 * x5532);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x5533,x5402,1,x11,x5402,x5550,x5533,1,x5548,x5533);

}
int32_t x5584 = 0;
int32_t x5585 = 1;
x5585 *= 1;
x5584 += 1;
x5585 *= 1;
x5585 *= 1;
int32_t x5590 = x5584;
bool x5591 = x5590 >= 2;
if (x5591) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5596 = x5590 == 0;
if (x5596) {
int32_t x5597 = x5585;
bool x5598 = x5597 == 128;
if (x5598) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5605 = x5585;
int32_t x5606 = 128 / x5605;
bool x5607 = x5606 == 1;
bool x5610;
if (x454) {
bool x5608 = 128 == x5606;
bool x5609 = x5607 || x5608;
x5610 = x5609;
} else {
x5610 = false;
}
bool x5614;
if (x5610) {
x5614 = x5613;
} else {
x5614 = false;
}
bool x5615;
if (x5614) {
x5615 = x5613;
} else {
x5615 = false;
}
if (x5615) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x5532,x5532,1,x5606,1,1);
assert(false && "");
}
bool x5621 = 128 <= x5606;
int32_t x5622;
if (x5621) {
x5622 = x5606;
} else {
x5622 = 128;
}
int32_t x5631 = x5622 * x5630;
int32_t x5632 = 64 * x5631;
float* x5633 = (float*)myMalloc(x5632 * sizeof(float));;
int32_t x5636;
if (x5607) {
x5636 = 0;
} else {
x5636 = 1;
}
for(int x5637=0; x5637 < 64; x5637++) {
int32_t x5649 = x5534 * x5637;
int32_t x5643 = x5631 * x5637;
for(int x5639=0; x5639 < x5622; x5639++) {
int32_t x5650 = x5533 * x5639;
int32_t x5651 = x5649 + x5650;
int32_t x5656 = x5636 * x5639;
int32_t x5645 = x5630 * x5639;
for(int x5641=0; x5641 < x5624; x5641++) {
int32_t x5652 = x5634 * x5641;
int32_t x5653 = x5651 + x5652;
int32_t x5647 = x5624 * x5641;
for(int x5642=0; x5642 < x5624; x5642++) {
int32_t x5654 = x5635 * x5642;
int32_t x5655 = x5653 + x5654;
float x5657 = x5538[x5655];
float x5658 = x204[x5656];
int32_t x5644 = x5642 + x5643;
int32_t x5646 = x5644 + x5645;
int32_t x5648 = x5646 + x5647;
float x5659 = x5657 - x5658;
x5633[x5648] = x5659;

}

}

}

}
float* x5669 = (float*)myMalloc(128 * sizeof(float));;
for(int x5671=0; x5671 < 128; x5671++) {
float x5672 = x134[x5671];
float x5673 = x5672 + 1.0E-5f;
x5669[x5671] = x5673;

}
float* x5677 = (float*)myMalloc(128 * sizeof(float));;
for(int x5678=0; x5678 < 128; x5678++) {
float x5679 = x5669[x5678];
double x5680 = (double)x5679;
double x5681 = sqrt(x5680);
float x5682 = (float)x5681;
x5677[x5678] = x5682;

}
int32_t x5686 = 0;
int32_t x5687 = 1;
x5687 *= 1;
x5686 += 1;
x5687 *= 1;
x5687 *= 1;
int32_t x5692 = x5686;
bool x5693 = x5692 >= 2;
if (x5693) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5698 = x5692 == 0;
if (x5698) {
int32_t x5699 = x5687;
bool x5700 = x5699 == 128;
if (x5700) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5707 = x5687;
bool x5709 = x5622 == 1;
int32_t x5708 = 128 / x5707;
bool x5710 = x5708 == 1;
bool x5714;
if (x454) {
bool x5711 = x5709 || x5710;
bool x5712 = x5622 == x5708;
bool x5713 = x5711 || x5712;
x5714 = x5713;
} else {
x5714 = false;
}
bool x5718;
if (x5714) {
x5718 = x5717;
} else {
x5718 = false;
}
bool x5719;
if (x5718) {
x5719 = x5717;
} else {
x5719 = false;
}
if (x5719) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5622,x5624,x5624,1,x5708,1,1);
assert(false && "");
}
bool x5725 = x5622 <= x5708;
int32_t x5726;
if (x5725) {
x5726 = x5708;
} else {
x5726 = x5622;
}
int32_t x5735 = x5726 * x5734;
int32_t x5736 = 64 * x5735;
float* x5737 = (float*)myMalloc(x5736 * sizeof(float));;
int32_t x5738;
if (x5709) {
x5738 = 0;
} else {
x5738 = x5630;
}
int32_t x5741;
if (x5710) {
x5741 = 0;
} else {
x5741 = 1;
}
for(int x5742=0; x5742 < 64; x5742++) {
int32_t x5754 = x5631 * x5742;
int32_t x5748 = x5735 * x5742;
for(int x5744=0; x5744 < x5726; x5744++) {
int32_t x5755 = x5738 * x5744;
int32_t x5756 = x5754 + x5755;
int32_t x5761 = x5741 * x5744;
int32_t x5750 = x5734 * x5744;
for(int x5746=0; x5746 < x5728; x5746++) {
int32_t x5757 = x5739 * x5746;
int32_t x5758 = x5756 + x5757;
int32_t x5752 = x5728 * x5746;
for(int x5747=0; x5747 < x5728; x5747++) {
int32_t x5759 = x5740 * x5747;
int32_t x5760 = x5758 + x5759;
float x5762 = x5633[x5760];
float x5763 = x5677[x5761];
int32_t x5749 = x5747 + x5748;
int32_t x5751 = x5749 + x5750;
int32_t x5753 = x5751 + x5752;
float x5764 = x5762 / x5763;
x5737[x5753] = x5764;

}

}

}

}
int32_t x5774 = 0;
int32_t x5775 = 1;
x5775 *= 1;
x5774 += 1;
x5775 *= 1;
x5775 *= 1;
int32_t x5780 = x5774;
bool x5781 = x5780 >= 2;
if (x5781) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5786 = x5780 == 0;
if (x5786) {
int32_t x5787 = x5775;
bool x5788 = x5787 == 128;
if (x5788) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5795 = x5775;
bool x5797 = x5726 == 1;
int32_t x5796 = 128 / x5795;
bool x5798 = x5796 == 1;
bool x5802;
if (x454) {
bool x5799 = x5797 || x5798;
bool x5800 = x5726 == x5796;
bool x5801 = x5799 || x5800;
x5802 = x5801;
} else {
x5802 = false;
}
bool x5806;
if (x5802) {
x5806 = x5805;
} else {
x5806 = false;
}
bool x5807;
if (x5806) {
x5807 = x5805;
} else {
x5807 = false;
}
if (x5807) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5726,x5728,x5728,1,x5796,1,1);
assert(false && "");
}
bool x5813 = x5726 <= x5796;
int32_t x5814;
if (x5813) {
x5814 = x5796;
} else {
x5814 = x5726;
}
int32_t x5823 = x5814 * x5822;
int32_t x5824 = 64 * x5823;
float* x5825 = (float*)myMalloc(x5824 * sizeof(float));;
int32_t x5826;
if (x5797) {
x5826 = 0;
} else {
x5826 = x5734;
}
int32_t x5829;
if (x5798) {
x5829 = 0;
} else {
x5829 = 1;
}
for(int x5830=0; x5830 < 64; x5830++) {
int32_t x5842 = x5735 * x5830;
int32_t x5836 = x5823 * x5830;
for(int x5832=0; x5832 < x5814; x5832++) {
int32_t x5843 = x5826 * x5832;
int32_t x5844 = x5842 + x5843;
int32_t x5849 = x5829 * x5832;
int32_t x5838 = x5822 * x5832;
for(int x5834=0; x5834 < x5816; x5834++) {
int32_t x5845 = x5827 * x5834;
int32_t x5846 = x5844 + x5845;
int32_t x5840 = x5816 * x5834;
for(int x5835=0; x5835 < x5816; x5835++) {
int32_t x5847 = x5828 * x5835;
int32_t x5848 = x5846 + x5847;
float x5850 = x5737[x5848];
float x5851 = x84[x5849];
int32_t x5837 = x5835 + x5836;
int32_t x5839 = x5837 + x5838;
int32_t x5841 = x5839 + x5840;
float x5852 = x5850 * x5851;
x5825[x5841] = x5852;

}

}

}

}
int32_t x5862 = 0;
int32_t x5863 = 1;
x5863 *= 1;
x5862 += 1;
x5863 *= 1;
x5863 *= 1;
int32_t x5868 = x5862;
bool x5869 = x5868 >= 2;
if (x5869) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5874 = x5868 == 0;
if (x5874) {
int32_t x5875 = x5863;
bool x5876 = x5875 == 128;
if (x5876) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5883 = x5863;
bool x5885 = x5814 == 1;
int32_t x5884 = 128 / x5883;
bool x5886 = x5884 == 1;
bool x5890;
if (x454) {
bool x5887 = x5885 || x5886;
bool x5888 = x5814 == x5884;
bool x5889 = x5887 || x5888;
x5890 = x5889;
} else {
x5890 = false;
}
bool x5894;
if (x5890) {
x5894 = x5893;
} else {
x5894 = false;
}
bool x5895;
if (x5894) {
x5895 = x5893;
} else {
x5895 = false;
}
if (x5895) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5814,x5816,x5816,1,x5884,1,1);
assert(false && "");
}
bool x5901 = x5814 <= x5884;
int32_t x5902;
if (x5901) {
x5902 = x5884;
} else {
x5902 = x5814;
}
int32_t x5911 = x5902 * x5910;
int32_t x5912 = 64 * x5911;
float* x5913 = (float*)myMalloc(x5912 * sizeof(float));;
int32_t x5914;
if (x5885) {
x5914 = 0;
} else {
x5914 = x5822;
}
int32_t x5917;
if (x5886) {
x5917 = 0;
} else {
x5917 = 1;
}
for(int x5918=0; x5918 < 64; x5918++) {
int32_t x5930 = x5823 * x5918;
int32_t x5924 = x5911 * x5918;
for(int x5920=0; x5920 < x5902; x5920++) {
int32_t x5931 = x5914 * x5920;
int32_t x5932 = x5930 + x5931;
int32_t x5937 = x5917 * x5920;
int32_t x5926 = x5910 * x5920;
for(int x5922=0; x5922 < x5904; x5922++) {
int32_t x5933 = x5915 * x5922;
int32_t x5934 = x5932 + x5933;
int32_t x5928 = x5904 * x5922;
for(int x5923=0; x5923 < x5904; x5923++) {
int32_t x5935 = x5916 * x5923;
int32_t x5936 = x5934 + x5935;
float x5938 = x5825[x5936];
float x5939 = x172[x5937];
int32_t x5925 = x5923 + x5924;
int32_t x5927 = x5925 + x5926;
int32_t x5929 = x5927 + x5928;
float x5940 = x5938 + x5939;
x5913[x5929] = x5940;

}

}

}

}
float* x5950 = (float*)myMalloc(x5912 * sizeof(float));;
for(int x5952=0; x5952 < x5912; x5952++) {
float x5953 = x5913[x5952];
bool x5954 = x5953 < 0.0f;
if (x5954) {
x5950[x5952] = 0.0f;
} else {
float x5957 = x5913[x5952];
x5950[x5952] = x5957;
}

}
float* x5972 = (float*)myMalloc(x5971 * sizeof(float));;
int32_t x5973 = 9 * x5902;
int32_t x5976 = 64 * x5973;
int32_t x5977 = x5976 * x5967;
float* x5978 = (float*)myMalloc(x5977 * sizeof(float));;
int32_t x5974 = x5973 * x5967;
int32_t x5986 = x5902 * 3;
int32_t x5987 = x5986 * 3;
for(int x5979=0; x5979 < 64; x5979++) {
int32_t x5980 = x5979 * x5911;
float* x5981 = x5950+x5980;
int32_t x5982 = x5979 * x5968;
float* x5983 = x5972+x5982;
int32_t x5984 = x5979 * x5974;
float* x5985 = x5978+x5984;
for(int x5989=0; x5989 < x5987; x5989++) {
int32_t x5990 = x5989 / 9;
int32_t x5994 = x5990 * 3;
int32_t x5995 = x5994 * 3;
int32_t x5996 = x5995 * x5966;
int32_t x5997 = x5996 * x5966;
int32_t x5991 = x5989 % 9;
int32_t x5992 = x5991 / 3;
int32_t x5998 = x5992 * 3;
int32_t x5999 = x5998 * x5966;
int32_t x6000 = x5999 * x5966;
int32_t x6001 = x5997 + x6000;
int32_t x5993 = x5991 % 3;
int32_t x6002 = x5993 * x5966;
int32_t x6003 = x6002 * x5966;
int32_t x6004 = x6001 + x6003;
float* x6005 = x5985+x6004;
int32_t x6006 = x5990 * x5904;
int32_t x6007 = x6006 * x5904;
float* x6008 = x5981+x6007;
for(int x6010=0; x6010 < x5966; x6010++) {
int32_t x6011 = x6010 * 2;
int32_t x6012 = x6011 - 1;
int32_t x6013 = x6012 + x5992;
bool x6014 = x6013 < 0;
bool x6015 = x6013 >= x5904;
bool x6016 = x6014 || x6015;
if (x6016) {
int32_t x6017 = x6010 * x5966;
float* x6018 = x6005+x6017;
memset(x6018, 0, 4 * x5966);;
} else {
int32_t x6017 = x6010 * x5966;
int32_t x6033 = x6013 * x5904;
for(int x6021=0; x6021 < x5966; x6021++) {
int32_t x6022 = x6021 * 2;
int32_t x6023 = x6022 - 1;
int32_t x6024 = x6023 + x5993;
bool x6025 = x6024 < 0;
bool x6026 = x6024 >= x5904;
bool x6027 = x6025 || x6026;
if (x6027) {
int32_t x6028 = x6017 + x6021;
float* x6029 = x6005+x6028;
memset(x6029, 0, 4 * 1);;
} else {
int32_t x6028 = x6017 + x6021;
float* x6032 = x6005+x6028;
int32_t x6034 = x6033 + x6024;
float* x6035 = x6008+x6034;
memcpy(x6032, x6035, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x5967,x5973,1,x27,x5973,x5985,x5967,1,x5983,x5967);

}
int32_t x6050 = 0;
int32_t x6051 = 1;
x6051 *= 1;
x6050 += 1;
x6051 *= 1;
x6051 *= 1;
int32_t x6056 = x6050;
bool x6057 = x6056 >= 2;
if (x6057) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6062 = x6056 == 0;
if (x6062) {
int32_t x6063 = x6051;
bool x6064 = x6063 == 128;
if (x6064) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6071 = x6051;
int32_t x6072 = 128 / x6071;
bool x6073 = x6072 == 1;
bool x6076;
if (x454) {
bool x6074 = 128 == x6072;
bool x6075 = x6073 || x6074;
x6076 = x6075;
} else {
x6076 = false;
}
bool x6080;
if (x6076) {
x6080 = x6079;
} else {
x6080 = false;
}
bool x6081;
if (x6080) {
x6081 = x6079;
} else {
x6081 = false;
}
if (x6081) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x5966,x5966,1,x6072,1,1);
assert(false && "");
}
bool x6087 = 128 <= x6072;
int32_t x6088;
if (x6087) {
x6088 = x6072;
} else {
x6088 = 128;
}
int32_t x6097 = x6088 * x6096;
int32_t x6098 = 64 * x6097;
float* x6099 = (float*)myMalloc(x6098 * sizeof(float));;
int32_t x6102;
if (x6073) {
x6102 = 0;
} else {
x6102 = 1;
}
for(int x6103=0; x6103 < 64; x6103++) {
int32_t x6115 = x5968 * x6103;
int32_t x6109 = x6097 * x6103;
for(int x6105=0; x6105 < x6088; x6105++) {
int32_t x6116 = x5967 * x6105;
int32_t x6117 = x6115 + x6116;
int32_t x6122 = x6102 * x6105;
int32_t x6111 = x6096 * x6105;
for(int x6107=0; x6107 < x6090; x6107++) {
int32_t x6118 = x6100 * x6107;
int32_t x6119 = x6117 + x6118;
int32_t x6113 = x6090 * x6107;
for(int x6108=0; x6108 < x6090; x6108++) {
int32_t x6120 = x6101 * x6108;
int32_t x6121 = x6119 + x6120;
float x6123 = x5972[x6121];
float x6124 = x128[x6122];
int32_t x6110 = x6108 + x6109;
int32_t x6112 = x6110 + x6111;
int32_t x6114 = x6112 + x6113;
float x6125 = x6123 - x6124;
x6099[x6114] = x6125;

}

}

}

}
float* x6135 = (float*)myMalloc(128 * sizeof(float));;
for(int x6136=0; x6136 < 128; x6136++) {
float x6137 = x43[x6136];
float x6138 = x6137 + 1.0E-5f;
x6135[x6136] = x6138;

}
float* x6142 = (float*)myMalloc(128 * sizeof(float));;
for(int x6143=0; x6143 < 128; x6143++) {
float x6144 = x6135[x6143];
double x6145 = (double)x6144;
double x6146 = sqrt(x6145);
float x6147 = (float)x6146;
x6142[x6143] = x6147;

}
int32_t x6151 = 0;
int32_t x6152 = 1;
x6152 *= 1;
x6151 += 1;
x6152 *= 1;
x6152 *= 1;
int32_t x6157 = x6151;
bool x6158 = x6157 >= 2;
if (x6158) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6163 = x6157 == 0;
if (x6163) {
int32_t x6164 = x6152;
bool x6165 = x6164 == 128;
if (x6165) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6172 = x6152;
bool x6174 = x6088 == 1;
int32_t x6173 = 128 / x6172;
bool x6175 = x6173 == 1;
bool x6179;
if (x454) {
bool x6176 = x6174 || x6175;
bool x6177 = x6088 == x6173;
bool x6178 = x6176 || x6177;
x6179 = x6178;
} else {
x6179 = false;
}
bool x6183;
if (x6179) {
x6183 = x6182;
} else {
x6183 = false;
}
bool x6184;
if (x6183) {
x6184 = x6182;
} else {
x6184 = false;
}
if (x6184) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6088,x6090,x6090,1,x6173,1,1);
assert(false && "");
}
bool x6190 = x6088 <= x6173;
int32_t x6191;
if (x6190) {
x6191 = x6173;
} else {
x6191 = x6088;
}
int32_t x6200 = x6191 * x6199;
int32_t x6201 = 64 * x6200;
float* x6202 = (float*)myMalloc(x6201 * sizeof(float));;
int32_t x6203;
if (x6174) {
x6203 = 0;
} else {
x6203 = x6096;
}
int32_t x6206;
if (x6175) {
x6206 = 0;
} else {
x6206 = 1;
}
for(int x6207=0; x6207 < 64; x6207++) {
int32_t x6219 = x6097 * x6207;
int32_t x6213 = x6200 * x6207;
for(int x6209=0; x6209 < x6191; x6209++) {
int32_t x6220 = x6203 * x6209;
int32_t x6221 = x6219 + x6220;
int32_t x6226 = x6206 * x6209;
int32_t x6215 = x6199 * x6209;
for(int x6211=0; x6211 < x6193; x6211++) {
int32_t x6222 = x6204 * x6211;
int32_t x6223 = x6221 + x6222;
int32_t x6217 = x6193 * x6211;
for(int x6212=0; x6212 < x6193; x6212++) {
int32_t x6224 = x6205 * x6212;
int32_t x6225 = x6223 + x6224;
float x6227 = x6099[x6225];
float x6228 = x6142[x6226];
int32_t x6214 = x6212 + x6213;
int32_t x6216 = x6214 + x6215;
int32_t x6218 = x6216 + x6217;
float x6229 = x6227 / x6228;
x6202[x6218] = x6229;

}

}

}

}
int32_t x6239 = 0;
int32_t x6240 = 1;
x6240 *= 1;
x6239 += 1;
x6240 *= 1;
x6240 *= 1;
int32_t x6245 = x6239;
bool x6246 = x6245 >= 2;
if (x6246) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6251 = x6245 == 0;
if (x6251) {
int32_t x6252 = x6240;
bool x6253 = x6252 == 128;
if (x6253) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6260 = x6240;
bool x6262 = x6191 == 1;
int32_t x6261 = 128 / x6260;
bool x6263 = x6261 == 1;
bool x6267;
if (x454) {
bool x6264 = x6262 || x6263;
bool x6265 = x6191 == x6261;
bool x6266 = x6264 || x6265;
x6267 = x6266;
} else {
x6267 = false;
}
bool x6271;
if (x6267) {
x6271 = x6270;
} else {
x6271 = false;
}
bool x6272;
if (x6271) {
x6272 = x6270;
} else {
x6272 = false;
}
if (x6272) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6191,x6193,x6193,1,x6261,1,1);
assert(false && "");
}
bool x6278 = x6191 <= x6261;
int32_t x6279;
if (x6278) {
x6279 = x6261;
} else {
x6279 = x6191;
}
int32_t x6288 = x6279 * x6287;
int32_t x6289 = 64 * x6288;
float* x6290 = (float*)myMalloc(x6289 * sizeof(float));;
int32_t x6291;
if (x6262) {
x6291 = 0;
} else {
x6291 = x6199;
}
int32_t x6294;
if (x6263) {
x6294 = 0;
} else {
x6294 = 1;
}
for(int x6295=0; x6295 < 64; x6295++) {
int32_t x6307 = x6200 * x6295;
int32_t x6301 = x6288 * x6295;
for(int x6297=0; x6297 < x6279; x6297++) {
int32_t x6308 = x6291 * x6297;
int32_t x6309 = x6307 + x6308;
int32_t x6314 = x6294 * x6297;
int32_t x6303 = x6287 * x6297;
for(int x6299=0; x6299 < x6281; x6299++) {
int32_t x6310 = x6292 * x6299;
int32_t x6311 = x6309 + x6310;
int32_t x6305 = x6281 * x6299;
for(int x6300=0; x6300 < x6281; x6300++) {
int32_t x6312 = x6293 * x6300;
int32_t x6313 = x6311 + x6312;
float x6315 = x6202[x6313];
float x6316 = x252[x6314];
int32_t x6302 = x6300 + x6301;
int32_t x6304 = x6302 + x6303;
int32_t x6306 = x6304 + x6305;
float x6317 = x6315 * x6316;
x6290[x6306] = x6317;

}

}

}

}
int32_t x6327 = 0;
int32_t x6328 = 1;
x6328 *= 1;
x6327 += 1;
x6328 *= 1;
x6328 *= 1;
int32_t x6333 = x6327;
bool x6334 = x6333 >= 2;
if (x6334) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6339 = x6333 == 0;
if (x6339) {
int32_t x6340 = x6328;
bool x6341 = x6340 == 128;
if (x6341) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6348 = x6328;
bool x6350 = x6279 == 1;
int32_t x6349 = 128 / x6348;
bool x6351 = x6349 == 1;
bool x6355;
if (x454) {
bool x6352 = x6350 || x6351;
bool x6353 = x6279 == x6349;
bool x6354 = x6352 || x6353;
x6355 = x6354;
} else {
x6355 = false;
}
bool x6359;
if (x6355) {
x6359 = x6358;
} else {
x6359 = false;
}
bool x6360;
if (x6359) {
x6360 = x6358;
} else {
x6360 = false;
}
if (x6360) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6279,x6281,x6281,1,x6349,1,1);
assert(false && "");
}
bool x6366 = x6279 <= x6349;
int32_t x6367;
if (x6366) {
x6367 = x6349;
} else {
x6367 = x6279;
}
int32_t x6376 = x6367 * x6375;
int32_t x6377 = 64 * x6376;
float* x6378 = (float*)myMalloc(x6377 * sizeof(float));;
int32_t x6379;
if (x6350) {
x6379 = 0;
} else {
x6379 = x6287;
}
int32_t x6382;
if (x6351) {
x6382 = 0;
} else {
x6382 = 1;
}
for(int x6383=0; x6383 < 64; x6383++) {
int32_t x6395 = x6288 * x6383;
int32_t x6389 = x6376 * x6383;
for(int x6385=0; x6385 < x6367; x6385++) {
int32_t x6396 = x6379 * x6385;
int32_t x6397 = x6395 + x6396;
int32_t x6402 = x6382 * x6385;
int32_t x6391 = x6375 * x6385;
for(int x6387=0; x6387 < x6369; x6387++) {
int32_t x6398 = x6380 * x6387;
int32_t x6399 = x6397 + x6398;
int32_t x6393 = x6369 * x6387;
for(int x6388=0; x6388 < x6369; x6388++) {
int32_t x6400 = x6381 * x6388;
int32_t x6401 = x6399 + x6400;
float x6403 = x6290[x6401];
float x6404 = x190[x6402];
int32_t x6390 = x6388 + x6389;
int32_t x6392 = x6390 + x6391;
int32_t x6394 = x6392 + x6393;
float x6405 = x6403 + x6404;
x6378[x6394] = x6405;

}

}

}

}
float* x6415 = (float*)myMalloc(x6377 * sizeof(float));;
for(int x6417=0; x6417 < x6377; x6417++) {
float x6418 = x6378[x6417];
bool x6419 = x6418 < 0.0f;
if (x6419) {
x6415[x6417] = 0.0f;
} else {
float x6422 = x6378[x6417];
x6415[x6417] = x6422;
}

}
float* x6436 = (float*)myMalloc(x6435 * sizeof(float));;
int32_t x6439 = 64 * x6367;
int32_t x6440 = x6439 * x6431;
float* x6441 = (float*)myMalloc(x6440 * sizeof(float));;
int32_t x6437 = x6367 * x6431;
for(int x6442=0; x6442 < 64; x6442++) {
int32_t x6443 = x6442 * x6376;
float* x6444 = x6415+x6443;
int32_t x6445 = x6442 * x6432;
float* x6446 = x6436+x6445;
int32_t x6447 = x6442 * x6437;
float* x6448 = x6441+x6447;
for(int x6449=0; x6449 < x6367; x6449++) {
int32_t x6450 = x6449 / 1;
int32_t x6454 = x6450 * x6430;
int32_t x6455 = x6454 * x6430;
int32_t x6451 = x6449 % 1;
int32_t x6452 = x6451 / 1;
int32_t x6456 = x6452 * x6430;
int32_t x6457 = x6456 * x6430;
int32_t x6458 = x6455 + x6457;
int32_t x6453 = x6451 % 1;
int32_t x6459 = x6453 * x6430;
int32_t x6460 = x6459 * x6430;
int32_t x6461 = x6458 + x6460;
float* x6462 = x6448+x6461;
int32_t x6463 = x6450 * x6369;
int32_t x6464 = x6463 * x6369;
float* x6465 = x6444+x6464;
for(int x6467=0; x6467 < x6430; x6467++) {
int32_t x6469 = x6467 * x6430;
float* x6470 = x6462+x6469;
int32_t x6468 = x6467 + x6452;
int32_t x6471 = x6468 * x6369;
int32_t x6472 = x6471 + x6453;
float* x6473 = x6465+x6472;
memcpy(x6470, x6473, 4 * x6430);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x6431,x6367,1,x106,x6367,x6448,x6431,1,x6446,x6431);

}
int32_t x6482 = 0;
int32_t x6483 = 1;
x6483 *= 1;
x6482 += 1;
x6483 *= 1;
x6483 *= 1;
int32_t x6488 = x6482;
bool x6489 = x6488 >= 2;
if (x6489) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6494 = x6488 == 0;
if (x6494) {
int32_t x6495 = x6483;
bool x6496 = x6495 == 512;
if (x6496) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6503 = x6483;
int32_t x6504 = 512 / x6503;
bool x6505 = x6504 == 1;
bool x6508;
if (x454) {
bool x6506 = 512 == x6504;
bool x6507 = x6505 || x6506;
x6508 = x6507;
} else {
x6508 = false;
}
bool x6512;
if (x6508) {
x6512 = x6511;
} else {
x6512 = false;
}
bool x6513;
if (x6512) {
x6513 = x6511;
} else {
x6513 = false;
}
if (x6513) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x6430,x6430,1,x6504,1,1);
assert(false && "");
}
bool x6519 = 512 <= x6504;
int32_t x6520;
if (x6519) {
x6520 = x6504;
} else {
x6520 = 512;
}
int32_t x6529 = x6520 * x6528;
int32_t x6530 = 64 * x6529;
float* x6531 = (float*)myMalloc(x6530 * sizeof(float));;
int32_t x6534;
if (x6505) {
x6534 = 0;
} else {
x6534 = 1;
}
for(int x6535=0; x6535 < 64; x6535++) {
int32_t x6547 = x6432 * x6535;
int32_t x6541 = x6529 * x6535;
for(int x6537=0; x6537 < x6520; x6537++) {
int32_t x6548 = x6431 * x6537;
int32_t x6549 = x6547 + x6548;
int32_t x6554 = x6534 * x6537;
int32_t x6543 = x6528 * x6537;
for(int x6539=0; x6539 < x6522; x6539++) {
int32_t x6550 = x6532 * x6539;
int32_t x6551 = x6549 + x6550;
int32_t x6545 = x6522 * x6539;
for(int x6540=0; x6540 < x6522; x6540++) {
int32_t x6552 = x6533 * x6540;
int32_t x6553 = x6551 + x6552;
float x6555 = x6436[x6553];
float x6556 = x149[x6554];
int32_t x6542 = x6540 + x6541;
int32_t x6544 = x6542 + x6543;
int32_t x6546 = x6544 + x6545;
float x6557 = x6555 - x6556;
x6531[x6546] = x6557;

}

}

}

}
float* x6567 = (float*)myMalloc(512 * sizeof(float));;
for(int x6569=0; x6569 < 512; x6569++) {
float x6570 = x101[x6569];
float x6571 = x6570 + 1.0E-5f;
x6567[x6569] = x6571;

}
float* x6575 = (float*)myMalloc(512 * sizeof(float));;
for(int x6576=0; x6576 < 512; x6576++) {
float x6577 = x6567[x6576];
double x6578 = (double)x6577;
double x6579 = sqrt(x6578);
float x6580 = (float)x6579;
x6575[x6576] = x6580;

}
int32_t x6584 = 0;
int32_t x6585 = 1;
x6585 *= 1;
x6584 += 1;
x6585 *= 1;
x6585 *= 1;
int32_t x6590 = x6584;
bool x6591 = x6590 >= 2;
if (x6591) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6596 = x6590 == 0;
if (x6596) {
int32_t x6597 = x6585;
bool x6598 = x6597 == 512;
if (x6598) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6605 = x6585;
bool x6607 = x6520 == 1;
int32_t x6606 = 512 / x6605;
bool x6608 = x6606 == 1;
bool x6612;
if (x454) {
bool x6609 = x6607 || x6608;
bool x6610 = x6520 == x6606;
bool x6611 = x6609 || x6610;
x6612 = x6611;
} else {
x6612 = false;
}
bool x6616;
if (x6612) {
x6616 = x6615;
} else {
x6616 = false;
}
bool x6617;
if (x6616) {
x6617 = x6615;
} else {
x6617 = false;
}
if (x6617) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6520,x6522,x6522,1,x6606,1,1);
assert(false && "");
}
bool x6623 = x6520 <= x6606;
int32_t x6624;
if (x6623) {
x6624 = x6606;
} else {
x6624 = x6520;
}
int32_t x6633 = x6624 * x6632;
int32_t x6634 = 64 * x6633;
float* x6635 = (float*)myMalloc(x6634 * sizeof(float));;
int32_t x6636;
if (x6607) {
x6636 = 0;
} else {
x6636 = x6528;
}
int32_t x6639;
if (x6608) {
x6639 = 0;
} else {
x6639 = 1;
}
for(int x6640=0; x6640 < 64; x6640++) {
int32_t x6652 = x6529 * x6640;
int32_t x6646 = x6633 * x6640;
for(int x6642=0; x6642 < x6624; x6642++) {
int32_t x6653 = x6636 * x6642;
int32_t x6654 = x6652 + x6653;
int32_t x6659 = x6639 * x6642;
int32_t x6648 = x6632 * x6642;
for(int x6644=0; x6644 < x6626; x6644++) {
int32_t x6655 = x6637 * x6644;
int32_t x6656 = x6654 + x6655;
int32_t x6650 = x6626 * x6644;
for(int x6645=0; x6645 < x6626; x6645++) {
int32_t x6657 = x6638 * x6645;
int32_t x6658 = x6656 + x6657;
float x6660 = x6531[x6658];
float x6661 = x6575[x6659];
int32_t x6647 = x6645 + x6646;
int32_t x6649 = x6647 + x6648;
int32_t x6651 = x6649 + x6650;
float x6662 = x6660 / x6661;
x6635[x6651] = x6662;

}

}

}

}
int32_t x6672 = 0;
int32_t x6673 = 1;
x6673 *= 1;
x6672 += 1;
x6673 *= 1;
x6673 *= 1;
int32_t x6678 = x6672;
bool x6679 = x6678 >= 2;
if (x6679) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6684 = x6678 == 0;
if (x6684) {
int32_t x6685 = x6673;
bool x6686 = x6685 == 512;
if (x6686) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6693 = x6673;
bool x6695 = x6624 == 1;
int32_t x6694 = 512 / x6693;
bool x6696 = x6694 == 1;
bool x6700;
if (x454) {
bool x6697 = x6695 || x6696;
bool x6698 = x6624 == x6694;
bool x6699 = x6697 || x6698;
x6700 = x6699;
} else {
x6700 = false;
}
bool x6704;
if (x6700) {
x6704 = x6703;
} else {
x6704 = false;
}
bool x6705;
if (x6704) {
x6705 = x6703;
} else {
x6705 = false;
}
if (x6705) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6624,x6626,x6626,1,x6694,1,1);
assert(false && "");
}
bool x6711 = x6624 <= x6694;
int32_t x6712;
if (x6711) {
x6712 = x6694;
} else {
x6712 = x6624;
}
int32_t x6721 = x6712 * x6720;
int32_t x6722 = 64 * x6721;
float* x6723 = (float*)myMalloc(x6722 * sizeof(float));;
int32_t x6724;
if (x6695) {
x6724 = 0;
} else {
x6724 = x6632;
}
int32_t x6727;
if (x6696) {
x6727 = 0;
} else {
x6727 = 1;
}
for(int x6728=0; x6728 < 64; x6728++) {
int32_t x6740 = x6633 * x6728;
int32_t x6734 = x6721 * x6728;
for(int x6730=0; x6730 < x6712; x6730++) {
int32_t x6741 = x6724 * x6730;
int32_t x6742 = x6740 + x6741;
int32_t x6747 = x6727 * x6730;
int32_t x6736 = x6720 * x6730;
for(int x6732=0; x6732 < x6714; x6732++) {
int32_t x6743 = x6725 * x6732;
int32_t x6744 = x6742 + x6743;
int32_t x6738 = x6714 * x6732;
for(int x6733=0; x6733 < x6714; x6733++) {
int32_t x6745 = x6726 * x6733;
int32_t x6746 = x6744 + x6745;
float x6748 = x6635[x6746];
float x6749 = x145[x6747];
int32_t x6735 = x6733 + x6734;
int32_t x6737 = x6735 + x6736;
int32_t x6739 = x6737 + x6738;
float x6750 = x6748 * x6749;
x6723[x6739] = x6750;

}

}

}

}
int32_t x6760 = 0;
int32_t x6761 = 1;
x6761 *= 1;
x6760 += 1;
x6761 *= 1;
x6761 *= 1;
int32_t x6766 = x6760;
bool x6767 = x6766 >= 2;
if (x6767) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6772 = x6766 == 0;
if (x6772) {
int32_t x6773 = x6761;
bool x6774 = x6773 == 512;
if (x6774) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6781 = x6761;
bool x6783 = x6712 == 1;
int32_t x6782 = 512 / x6781;
bool x6784 = x6782 == 1;
bool x6788;
if (x454) {
bool x6785 = x6783 || x6784;
bool x6786 = x6712 == x6782;
bool x6787 = x6785 || x6786;
x6788 = x6787;
} else {
x6788 = false;
}
bool x6792;
if (x6788) {
x6792 = x6791;
} else {
x6792 = false;
}
bool x6793;
if (x6792) {
x6793 = x6791;
} else {
x6793 = false;
}
if (x6793) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6712,x6714,x6714,1,x6782,1,1);
assert(false && "");
}
bool x6799 = x6712 <= x6782;
int32_t x6800;
if (x6799) {
x6800 = x6782;
} else {
x6800 = x6712;
}
int32_t x6809 = x6800 * x6808;
int32_t x6810 = 64 * x6809;
float* x6811 = (float*)myMalloc(x6810 * sizeof(float));;
int32_t x6812;
if (x6783) {
x6812 = 0;
} else {
x6812 = x6720;
}
int32_t x6815;
if (x6784) {
x6815 = 0;
} else {
x6815 = 1;
}
for(int x6816=0; x6816 < 64; x6816++) {
int32_t x6828 = x6721 * x6816;
int32_t x6822 = x6809 * x6816;
for(int x6818=0; x6818 < x6800; x6818++) {
int32_t x6829 = x6812 * x6818;
int32_t x6830 = x6828 + x6829;
int32_t x6835 = x6815 * x6818;
int32_t x6824 = x6808 * x6818;
for(int x6820=0; x6820 < x6802; x6820++) {
int32_t x6831 = x6813 * x6820;
int32_t x6832 = x6830 + x6831;
int32_t x6826 = x6802 * x6820;
for(int x6821=0; x6821 < x6802; x6821++) {
int32_t x6833 = x6814 * x6821;
int32_t x6834 = x6832 + x6833;
float x6836 = x6723[x6834];
float x6837 = x210[x6835];
int32_t x6823 = x6821 + x6822;
int32_t x6825 = x6823 + x6824;
int32_t x6827 = x6825 + x6826;
float x6838 = x6836 + x6837;
x6811[x6827] = x6838;

}

}

}

}
float* x6855 = (float*)myMalloc(x6854 * sizeof(float));;
int32_t x6858 = x5541 * x6850;
float* x6859 = (float*)myMalloc(x6858 * sizeof(float));;
int32_t x6856 = x5402 * x6850;
for(int x6860=0; x6860 < 64; x6860++) {
int32_t x6861 = x6860 * x5411;
float* x6862 = x5517+x6861;
int32_t x6863 = x6860 * x6851;
float* x6864 = x6855+x6863;
int32_t x6865 = x6860 * x6856;
float* x6866 = x6859+x6865;
for(int x6867=0; x6867 < x5402; x6867++) {
int32_t x6868 = x6867 / 1;
int32_t x6872 = x6868 * x6849;
int32_t x6873 = x6872 * x6849;
int32_t x6869 = x6867 % 1;
int32_t x6870 = x6869 / 1;
int32_t x6874 = x6870 * x6849;
int32_t x6875 = x6874 * x6849;
int32_t x6876 = x6873 + x6875;
int32_t x6871 = x6869 % 1;
int32_t x6877 = x6871 * x6849;
int32_t x6878 = x6877 * x6849;
int32_t x6879 = x6876 + x6878;
float* x6880 = x6866+x6879;
int32_t x6881 = x6868 * x5404;
int32_t x6882 = x6881 * x5404;
float* x6883 = x6862+x6882;
for(int x6885=0; x6885 < x6849; x6885++) {
int32_t x6889 = x6885 * x6849;
int32_t x6886 = x6885 * 2;
int32_t x6887 = x6886 + x6870;
int32_t x6892 = x6887 * x5404;
int32_t x6893 = x6892 + x6871;
for(int x6888=0; x6888 < x6849; x6888++) {
int32_t x6890 = x6889 + x6888;
float* x6891 = x6880+x6890;
int32_t x6894 = x6888 * 2;
int32_t x6895 = x6893 + x6894;
float* x6896 = x6883+x6895;
memcpy(x6891, x6896, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x6850,x5402,1,x258,x5402,x6866,x6850,1,x6864,x6850);

}
int32_t x6907 = 0;
int32_t x6908 = 1;
x6908 *= 1;
x6907 += 1;
x6908 *= 1;
x6908 *= 1;
int32_t x6913 = x6907;
bool x6914 = x6913 >= 2;
if (x6914) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6919 = x6913 == 0;
if (x6919) {
int32_t x6920 = x6908;
bool x6921 = x6920 == 512;
if (x6921) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6928 = x6908;
int32_t x6929 = 512 / x6928;
bool x6930 = x6929 == 1;
bool x6933;
if (x454) {
bool x6931 = 512 == x6929;
bool x6932 = x6930 || x6931;
x6933 = x6932;
} else {
x6933 = false;
}
bool x6937;
if (x6933) {
x6937 = x6936;
} else {
x6937 = false;
}
bool x6938;
if (x6937) {
x6938 = x6936;
} else {
x6938 = false;
}
if (x6938) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x6849,x6849,1,x6929,1,1);
assert(false && "");
}
bool x6944 = 512 <= x6929;
int32_t x6945;
if (x6944) {
x6945 = x6929;
} else {
x6945 = 512;
}
int32_t x6954 = x6945 * x6953;
int32_t x6955 = 64 * x6954;
float* x6956 = (float*)myMalloc(x6955 * sizeof(float));;
int32_t x6959;
if (x6930) {
x6959 = 0;
} else {
x6959 = 1;
}
for(int x6960=0; x6960 < 64; x6960++) {
int32_t x6972 = x6851 * x6960;
int32_t x6966 = x6954 * x6960;
for(int x6962=0; x6962 < x6945; x6962++) {
int32_t x6973 = x6850 * x6962;
int32_t x6974 = x6972 + x6973;
int32_t x6979 = x6959 * x6962;
int32_t x6968 = x6953 * x6962;
for(int x6964=0; x6964 < x6947; x6964++) {
int32_t x6975 = x6957 * x6964;
int32_t x6976 = x6974 + x6975;
int32_t x6970 = x6947 * x6964;
for(int x6965=0; x6965 < x6947; x6965++) {
int32_t x6977 = x6958 * x6965;
int32_t x6978 = x6976 + x6977;
float x6980 = x6855[x6978];
float x6981 = x42[x6979];
int32_t x6967 = x6965 + x6966;
int32_t x6969 = x6967 + x6968;
int32_t x6971 = x6969 + x6970;
float x6982 = x6980 - x6981;
x6956[x6971] = x6982;

}

}

}

}
float* x6992 = (float*)myMalloc(512 * sizeof(float));;
for(int x6993=0; x6993 < 512; x6993++) {
float x6994 = x23[x6993];
float x6995 = x6994 + 1.0E-5f;
x6992[x6993] = x6995;

}
float* x6999 = (float*)myMalloc(512 * sizeof(float));;
for(int x7000=0; x7000 < 512; x7000++) {
float x7001 = x6992[x7000];
double x7002 = (double)x7001;
double x7003 = sqrt(x7002);
float x7004 = (float)x7003;
x6999[x7000] = x7004;

}
int32_t x7008 = 0;
int32_t x7009 = 1;
x7009 *= 1;
x7008 += 1;
x7009 *= 1;
x7009 *= 1;
int32_t x7014 = x7008;
bool x7015 = x7014 >= 2;
if (x7015) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7020 = x7014 == 0;
if (x7020) {
int32_t x7021 = x7009;
bool x7022 = x7021 == 512;
if (x7022) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7029 = x7009;
bool x7031 = x6945 == 1;
int32_t x7030 = 512 / x7029;
bool x7032 = x7030 == 1;
bool x7036;
if (x454) {
bool x7033 = x7031 || x7032;
bool x7034 = x6945 == x7030;
bool x7035 = x7033 || x7034;
x7036 = x7035;
} else {
x7036 = false;
}
bool x7040;
if (x7036) {
x7040 = x7039;
} else {
x7040 = false;
}
bool x7041;
if (x7040) {
x7041 = x7039;
} else {
x7041 = false;
}
if (x7041) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6945,x6947,x6947,1,x7030,1,1);
assert(false && "");
}
bool x7047 = x6945 <= x7030;
int32_t x7048;
if (x7047) {
x7048 = x7030;
} else {
x7048 = x6945;
}
int32_t x7057 = x7048 * x7056;
int32_t x7058 = 64 * x7057;
float* x7059 = (float*)myMalloc(x7058 * sizeof(float));;
int32_t x7060;
if (x7031) {
x7060 = 0;
} else {
x7060 = x6953;
}
int32_t x7063;
if (x7032) {
x7063 = 0;
} else {
x7063 = 1;
}
for(int x7064=0; x7064 < 64; x7064++) {
int32_t x7076 = x6954 * x7064;
int32_t x7070 = x7057 * x7064;
for(int x7066=0; x7066 < x7048; x7066++) {
int32_t x7077 = x7060 * x7066;
int32_t x7078 = x7076 + x7077;
int32_t x7083 = x7063 * x7066;
int32_t x7072 = x7056 * x7066;
for(int x7068=0; x7068 < x7050; x7068++) {
int32_t x7079 = x7061 * x7068;
int32_t x7080 = x7078 + x7079;
int32_t x7074 = x7050 * x7068;
for(int x7069=0; x7069 < x7050; x7069++) {
int32_t x7081 = x7062 * x7069;
int32_t x7082 = x7080 + x7081;
float x7084 = x6956[x7082];
float x7085 = x6999[x7083];
int32_t x7071 = x7069 + x7070;
int32_t x7073 = x7071 + x7072;
int32_t x7075 = x7073 + x7074;
float x7086 = x7084 / x7085;
x7059[x7075] = x7086;

}

}

}

}
int32_t x7096 = 0;
int32_t x7097 = 1;
x7097 *= 1;
x7096 += 1;
x7097 *= 1;
x7097 *= 1;
int32_t x7102 = x7096;
bool x7103 = x7102 >= 2;
if (x7103) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7108 = x7102 == 0;
if (x7108) {
int32_t x7109 = x7097;
bool x7110 = x7109 == 512;
if (x7110) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7117 = x7097;
bool x7119 = x7048 == 1;
int32_t x7118 = 512 / x7117;
bool x7120 = x7118 == 1;
bool x7124;
if (x454) {
bool x7121 = x7119 || x7120;
bool x7122 = x7048 == x7118;
bool x7123 = x7121 || x7122;
x7124 = x7123;
} else {
x7124 = false;
}
bool x7128;
if (x7124) {
x7128 = x7127;
} else {
x7128 = false;
}
bool x7129;
if (x7128) {
x7129 = x7127;
} else {
x7129 = false;
}
if (x7129) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7048,x7050,x7050,1,x7118,1,1);
assert(false && "");
}
bool x7135 = x7048 <= x7118;
int32_t x7136;
if (x7135) {
x7136 = x7118;
} else {
x7136 = x7048;
}
int32_t x7145 = x7136 * x7144;
int32_t x7146 = 64 * x7145;
float* x7147 = (float*)myMalloc(x7146 * sizeof(float));;
int32_t x7148;
if (x7119) {
x7148 = 0;
} else {
x7148 = x7056;
}
int32_t x7151;
if (x7120) {
x7151 = 0;
} else {
x7151 = 1;
}
for(int x7152=0; x7152 < 64; x7152++) {
int32_t x7164 = x7057 * x7152;
int32_t x7158 = x7145 * x7152;
for(int x7154=0; x7154 < x7136; x7154++) {
int32_t x7165 = x7148 * x7154;
int32_t x7166 = x7164 + x7165;
int32_t x7171 = x7151 * x7154;
int32_t x7160 = x7144 * x7154;
for(int x7156=0; x7156 < x7138; x7156++) {
int32_t x7167 = x7149 * x7156;
int32_t x7168 = x7166 + x7167;
int32_t x7162 = x7138 * x7156;
for(int x7157=0; x7157 < x7138; x7157++) {
int32_t x7169 = x7150 * x7157;
int32_t x7170 = x7168 + x7169;
float x7172 = x7059[x7170];
float x7173 = x207[x7171];
int32_t x7159 = x7157 + x7158;
int32_t x7161 = x7159 + x7160;
int32_t x7163 = x7161 + x7162;
float x7174 = x7172 * x7173;
x7147[x7163] = x7174;

}

}

}

}
int32_t x7184 = 0;
int32_t x7185 = 1;
x7185 *= 1;
x7184 += 1;
x7185 *= 1;
x7185 *= 1;
int32_t x7190 = x7184;
bool x7191 = x7190 >= 2;
if (x7191) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7196 = x7190 == 0;
if (x7196) {
int32_t x7197 = x7185;
bool x7198 = x7197 == 512;
if (x7198) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7205 = x7185;
bool x7207 = x7136 == 1;
int32_t x7206 = 512 / x7205;
bool x7208 = x7206 == 1;
bool x7212;
if (x454) {
bool x7209 = x7207 || x7208;
bool x7210 = x7136 == x7206;
bool x7211 = x7209 || x7210;
x7212 = x7211;
} else {
x7212 = false;
}
bool x7216;
if (x7212) {
x7216 = x7215;
} else {
x7216 = false;
}
bool x7217;
if (x7216) {
x7217 = x7215;
} else {
x7217 = false;
}
if (x7217) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7136,x7138,x7138,1,x7206,1,1);
assert(false && "");
}
bool x7223 = x7136 <= x7206;
int32_t x7224;
if (x7223) {
x7224 = x7206;
} else {
x7224 = x7136;
}
int32_t x7233 = x7224 * x7232;
int32_t x7234 = 64 * x7233;
float* x7235 = (float*)myMalloc(x7234 * sizeof(float));;
int32_t x7236;
if (x7207) {
x7236 = 0;
} else {
x7236 = x7144;
}
int32_t x7239;
if (x7208) {
x7239 = 0;
} else {
x7239 = 1;
}
for(int x7240=0; x7240 < 64; x7240++) {
int32_t x7252 = x7145 * x7240;
int32_t x7246 = x7233 * x7240;
for(int x7242=0; x7242 < x7224; x7242++) {
int32_t x7253 = x7236 * x7242;
int32_t x7254 = x7252 + x7253;
int32_t x7259 = x7239 * x7242;
int32_t x7248 = x7232 * x7242;
for(int x7244=0; x7244 < x7226; x7244++) {
int32_t x7255 = x7237 * x7244;
int32_t x7256 = x7254 + x7255;
int32_t x7250 = x7226 * x7244;
for(int x7245=0; x7245 < x7226; x7245++) {
int32_t x7257 = x7238 * x7245;
int32_t x7258 = x7256 + x7257;
float x7260 = x7147[x7258];
float x7261 = x119[x7259];
int32_t x7247 = x7245 + x7246;
int32_t x7249 = x7247 + x7248;
int32_t x7251 = x7249 + x7250;
float x7262 = x7260 + x7261;
x7235[x7251] = x7262;

}

}

}

}
bool x7272 = x6800 == 1;
bool x7273 = x7224 == 1;
bool x7274 = x7272 || x7273;
bool x7275 = x6800 == x7224;
bool x7276 = x7274 || x7275;
bool x7282;
if (x7276) {
x7282 = x7281;
} else {
x7282 = false;
}
bool x7283;
if (x7282) {
x7283 = x7281;
} else {
x7283 = false;
}
if (x7283) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6800,x6802,x6802,64,x7224,x7226,x7226);
assert(false && "");
}
bool x7289 = x6800 <= x7224;
int32_t x7290;
if (x7289) {
x7290 = x7224;
} else {
x7290 = x6800;
}
int32_t x7306;
if (x7272) {
x7306 = 0;
} else {
x7306 = x6808;
}
int32_t x7309;
if (x7273) {
x7309 = 0;
} else {
x7309 = x7232;
}
for(int x7312=0; x7312 < 64; x7312++) {
int32_t x7318 = x6809 * x7312;
int32_t x7325 = x7233 * x7312;
for(int x7314=0; x7314 < x7290; x7314++) {
int32_t x7319 = x7306 * x7314;
int32_t x7320 = x7318 + x7319;
int32_t x7326 = x7309 * x7314;
int32_t x7327 = x7325 + x7326;
for(int x7316=0; x7316 < x7292; x7316++) {
int32_t x7321 = x7307 * x7316;
int32_t x7322 = x7320 + x7321;
int32_t x7328 = x7310 * x7316;
int32_t x7329 = x7327 + x7328;
for(int x7317=0; x7317 < x7292; x7317++) {
int32_t x7323 = x7308 * x7317;
int32_t x7324 = x7322 + x7323;
float x7332 = x6811[x7324];
int32_t x7330 = x7311 * x7317;
int32_t x7331 = x7329 + x7330;
float x7333 = x7235[x7331];
float x7334 = x7332 + x7333;
x6811[x7324] = x7334;

}

}

}

}
float* x7344 = (float*)myMalloc(x6810 * sizeof(float));;
for(int x7346=0; x7346 < x6810; x7346++) {
float x7347 = x6811[x7346];
bool x7348 = x7347 < 0.0f;
if (x7348) {
x7344[x7346] = 0.0f;
} else {
float x7351 = x6811[x7346];
x7344[x7346] = x7351;
}

}
float* x7365 = (float*)myMalloc(x7364 * sizeof(float));;
int32_t x7368 = 64 * x6800;
int32_t x7369 = x7368 * x7360;
float* x7370 = (float*)myMalloc(x7369 * sizeof(float));;
int32_t x7366 = x6800 * x7360;
for(int x7371=0; x7371 < 64; x7371++) {
int32_t x7372 = x7371 * x6809;
float* x7373 = x7344+x7372;
int32_t x7374 = x7371 * x7361;
float* x7375 = x7365+x7374;
int32_t x7376 = x7371 * x7366;
float* x7377 = x7370+x7376;
for(int x7378=0; x7378 < x6800; x7378++) {
int32_t x7379 = x7378 / 1;
int32_t x7383 = x7379 * x7359;
int32_t x7384 = x7383 * x7359;
int32_t x7380 = x7378 % 1;
int32_t x7381 = x7380 / 1;
int32_t x7385 = x7381 * x7359;
int32_t x7386 = x7385 * x7359;
int32_t x7387 = x7384 + x7386;
int32_t x7382 = x7380 % 1;
int32_t x7388 = x7382 * x7359;
int32_t x7389 = x7388 * x7359;
int32_t x7390 = x7387 + x7389;
float* x7391 = x7377+x7390;
int32_t x7392 = x7379 * x6802;
int32_t x7393 = x7392 * x6802;
float* x7394 = x7373+x7393;
for(int x7396=0; x7396 < x7359; x7396++) {
int32_t x7398 = x7396 * x7359;
float* x7399 = x7391+x7398;
int32_t x7397 = x7396 + x7381;
int32_t x7400 = x7397 * x6802;
int32_t x7401 = x7400 + x7382;
float* x7402 = x7394+x7401;
memcpy(x7399, x7402, 4 * x7359);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x7360,x6800,1,x256,x6800,x7377,x7360,1,x7375,x7360);

}
int32_t x7411 = 0;
int32_t x7412 = 1;
x7412 *= 1;
x7411 += 1;
x7412 *= 1;
x7412 *= 1;
int32_t x7417 = x7411;
bool x7418 = x7417 >= 2;
if (x7418) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7423 = x7417 == 0;
if (x7423) {
int32_t x7424 = x7412;
bool x7425 = x7424 == 128;
if (x7425) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7432 = x7412;
int32_t x7433 = 128 / x7432;
bool x7434 = x7433 == 1;
bool x7437;
if (x454) {
bool x7435 = 128 == x7433;
bool x7436 = x7434 || x7435;
x7437 = x7436;
} else {
x7437 = false;
}
bool x7441;
if (x7437) {
x7441 = x7440;
} else {
x7441 = false;
}
bool x7442;
if (x7441) {
x7442 = x7440;
} else {
x7442 = false;
}
if (x7442) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x7359,x7359,1,x7433,1,1);
assert(false && "");
}
bool x7448 = 128 <= x7433;
int32_t x7449;
if (x7448) {
x7449 = x7433;
} else {
x7449 = 128;
}
int32_t x7458 = x7449 * x7457;
int32_t x7459 = 64 * x7458;
float* x7460 = (float*)myMalloc(x7459 * sizeof(float));;
int32_t x7463;
if (x7434) {
x7463 = 0;
} else {
x7463 = 1;
}
for(int x7464=0; x7464 < 64; x7464++) {
int32_t x7476 = x7361 * x7464;
int32_t x7470 = x7458 * x7464;
for(int x7466=0; x7466 < x7449; x7466++) {
int32_t x7477 = x7360 * x7466;
int32_t x7478 = x7476 + x7477;
int32_t x7483 = x7463 * x7466;
int32_t x7472 = x7457 * x7466;
for(int x7468=0; x7468 < x7451; x7468++) {
int32_t x7479 = x7461 * x7468;
int32_t x7480 = x7478 + x7479;
int32_t x7474 = x7451 * x7468;
for(int x7469=0; x7469 < x7451; x7469++) {
int32_t x7481 = x7462 * x7469;
int32_t x7482 = x7480 + x7481;
float x7484 = x7365[x7482];
float x7485 = x100[x7483];
int32_t x7471 = x7469 + x7470;
int32_t x7473 = x7471 + x7472;
int32_t x7475 = x7473 + x7474;
float x7486 = x7484 - x7485;
x7460[x7475] = x7486;

}

}

}

}
float* x7496 = (float*)myMalloc(128 * sizeof(float));;
for(int x7497=0; x7497 < 128; x7497++) {
float x7498 = x177[x7497];
float x7499 = x7498 + 1.0E-5f;
x7496[x7497] = x7499;

}
float* x7503 = (float*)myMalloc(128 * sizeof(float));;
for(int x7504=0; x7504 < 128; x7504++) {
float x7505 = x7496[x7504];
double x7506 = (double)x7505;
double x7507 = sqrt(x7506);
float x7508 = (float)x7507;
x7503[x7504] = x7508;

}
int32_t x7512 = 0;
int32_t x7513 = 1;
x7513 *= 1;
x7512 += 1;
x7513 *= 1;
x7513 *= 1;
int32_t x7518 = x7512;
bool x7519 = x7518 >= 2;
if (x7519) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7524 = x7518 == 0;
if (x7524) {
int32_t x7525 = x7513;
bool x7526 = x7525 == 128;
if (x7526) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7533 = x7513;
bool x7535 = x7449 == 1;
int32_t x7534 = 128 / x7533;
bool x7536 = x7534 == 1;
bool x7540;
if (x454) {
bool x7537 = x7535 || x7536;
bool x7538 = x7449 == x7534;
bool x7539 = x7537 || x7538;
x7540 = x7539;
} else {
x7540 = false;
}
bool x7544;
if (x7540) {
x7544 = x7543;
} else {
x7544 = false;
}
bool x7545;
if (x7544) {
x7545 = x7543;
} else {
x7545 = false;
}
if (x7545) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7449,x7451,x7451,1,x7534,1,1);
assert(false && "");
}
bool x7551 = x7449 <= x7534;
int32_t x7552;
if (x7551) {
x7552 = x7534;
} else {
x7552 = x7449;
}
int32_t x7561 = x7552 * x7560;
int32_t x7562 = 64 * x7561;
float* x7563 = (float*)myMalloc(x7562 * sizeof(float));;
int32_t x7564;
if (x7535) {
x7564 = 0;
} else {
x7564 = x7457;
}
int32_t x7567;
if (x7536) {
x7567 = 0;
} else {
x7567 = 1;
}
for(int x7568=0; x7568 < 64; x7568++) {
int32_t x7580 = x7458 * x7568;
int32_t x7574 = x7561 * x7568;
for(int x7570=0; x7570 < x7552; x7570++) {
int32_t x7581 = x7564 * x7570;
int32_t x7582 = x7580 + x7581;
int32_t x7587 = x7567 * x7570;
int32_t x7576 = x7560 * x7570;
for(int x7572=0; x7572 < x7554; x7572++) {
int32_t x7583 = x7565 * x7572;
int32_t x7584 = x7582 + x7583;
int32_t x7578 = x7554 * x7572;
for(int x7573=0; x7573 < x7554; x7573++) {
int32_t x7585 = x7566 * x7573;
int32_t x7586 = x7584 + x7585;
float x7588 = x7460[x7586];
float x7589 = x7503[x7587];
int32_t x7575 = x7573 + x7574;
int32_t x7577 = x7575 + x7576;
int32_t x7579 = x7577 + x7578;
float x7590 = x7588 / x7589;
x7563[x7579] = x7590;

}

}

}

}
int32_t x7600 = 0;
int32_t x7601 = 1;
x7601 *= 1;
x7600 += 1;
x7601 *= 1;
x7601 *= 1;
int32_t x7606 = x7600;
bool x7607 = x7606 >= 2;
if (x7607) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7612 = x7606 == 0;
if (x7612) {
int32_t x7613 = x7601;
bool x7614 = x7613 == 128;
if (x7614) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7621 = x7601;
bool x7623 = x7552 == 1;
int32_t x7622 = 128 / x7621;
bool x7624 = x7622 == 1;
bool x7628;
if (x454) {
bool x7625 = x7623 || x7624;
bool x7626 = x7552 == x7622;
bool x7627 = x7625 || x7626;
x7628 = x7627;
} else {
x7628 = false;
}
bool x7632;
if (x7628) {
x7632 = x7631;
} else {
x7632 = false;
}
bool x7633;
if (x7632) {
x7633 = x7631;
} else {
x7633 = false;
}
if (x7633) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7552,x7554,x7554,1,x7622,1,1);
assert(false && "");
}
bool x7639 = x7552 <= x7622;
int32_t x7640;
if (x7639) {
x7640 = x7622;
} else {
x7640 = x7552;
}
int32_t x7649 = x7640 * x7648;
int32_t x7650 = 64 * x7649;
float* x7651 = (float*)myMalloc(x7650 * sizeof(float));;
int32_t x7652;
if (x7623) {
x7652 = 0;
} else {
x7652 = x7560;
}
int32_t x7655;
if (x7624) {
x7655 = 0;
} else {
x7655 = 1;
}
for(int x7656=0; x7656 < 64; x7656++) {
int32_t x7668 = x7561 * x7656;
int32_t x7662 = x7649 * x7656;
for(int x7658=0; x7658 < x7640; x7658++) {
int32_t x7669 = x7652 * x7658;
int32_t x7670 = x7668 + x7669;
int32_t x7675 = x7655 * x7658;
int32_t x7664 = x7648 * x7658;
for(int x7660=0; x7660 < x7642; x7660++) {
int32_t x7671 = x7653 * x7660;
int32_t x7672 = x7670 + x7671;
int32_t x7666 = x7642 * x7660;
for(int x7661=0; x7661 < x7642; x7661++) {
int32_t x7673 = x7654 * x7661;
int32_t x7674 = x7672 + x7673;
float x7676 = x7563[x7674];
float x7677 = x222[x7675];
int32_t x7663 = x7661 + x7662;
int32_t x7665 = x7663 + x7664;
int32_t x7667 = x7665 + x7666;
float x7678 = x7676 * x7677;
x7651[x7667] = x7678;

}

}

}

}
int32_t x7688 = 0;
int32_t x7689 = 1;
x7689 *= 1;
x7688 += 1;
x7689 *= 1;
x7689 *= 1;
int32_t x7694 = x7688;
bool x7695 = x7694 >= 2;
if (x7695) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7700 = x7694 == 0;
if (x7700) {
int32_t x7701 = x7689;
bool x7702 = x7701 == 128;
if (x7702) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7709 = x7689;
bool x7711 = x7640 == 1;
int32_t x7710 = 128 / x7709;
bool x7712 = x7710 == 1;
bool x7716;
if (x454) {
bool x7713 = x7711 || x7712;
bool x7714 = x7640 == x7710;
bool x7715 = x7713 || x7714;
x7716 = x7715;
} else {
x7716 = false;
}
bool x7720;
if (x7716) {
x7720 = x7719;
} else {
x7720 = false;
}
bool x7721;
if (x7720) {
x7721 = x7719;
} else {
x7721 = false;
}
if (x7721) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7640,x7642,x7642,1,x7710,1,1);
assert(false && "");
}
bool x7727 = x7640 <= x7710;
int32_t x7728;
if (x7727) {
x7728 = x7710;
} else {
x7728 = x7640;
}
int32_t x7737 = x7728 * x7736;
int32_t x7738 = 64 * x7737;
float* x7739 = (float*)myMalloc(x7738 * sizeof(float));;
int32_t x7740;
if (x7711) {
x7740 = 0;
} else {
x7740 = x7648;
}
int32_t x7743;
if (x7712) {
x7743 = 0;
} else {
x7743 = 1;
}
for(int x7744=0; x7744 < 64; x7744++) {
int32_t x7756 = x7649 * x7744;
int32_t x7750 = x7737 * x7744;
for(int x7746=0; x7746 < x7728; x7746++) {
int32_t x7757 = x7740 * x7746;
int32_t x7758 = x7756 + x7757;
int32_t x7763 = x7743 * x7746;
int32_t x7752 = x7736 * x7746;
for(int x7748=0; x7748 < x7730; x7748++) {
int32_t x7759 = x7741 * x7748;
int32_t x7760 = x7758 + x7759;
int32_t x7754 = x7730 * x7748;
for(int x7749=0; x7749 < x7730; x7749++) {
int32_t x7761 = x7742 * x7749;
int32_t x7762 = x7760 + x7761;
float x7764 = x7651[x7762];
float x7765 = x17[x7763];
int32_t x7751 = x7749 + x7750;
int32_t x7753 = x7751 + x7752;
int32_t x7755 = x7753 + x7754;
float x7766 = x7764 + x7765;
x7739[x7755] = x7766;

}

}

}

}
float* x7776 = (float*)myMalloc(x7738 * sizeof(float));;
for(int x7778=0; x7778 < x7738; x7778++) {
float x7779 = x7739[x7778];
bool x7780 = x7779 < 0.0f;
if (x7780) {
x7776[x7778] = 0.0f;
} else {
float x7783 = x7739[x7778];
x7776[x7778] = x7783;
}

}
float* x7798 = (float*)myMalloc(x7797 * sizeof(float));;
int32_t x7799 = 9 * x7728;
int32_t x7802 = 64 * x7799;
int32_t x7803 = x7802 * x7793;
float* x7804 = (float*)myMalloc(x7803 * sizeof(float));;
int32_t x7800 = x7799 * x7793;
int32_t x7812 = x7728 * 3;
int32_t x7813 = x7812 * 3;
for(int x7805=0; x7805 < 64; x7805++) {
int32_t x7806 = x7805 * x7737;
float* x7807 = x7776+x7806;
int32_t x7808 = x7805 * x7794;
float* x7809 = x7798+x7808;
int32_t x7810 = x7805 * x7800;
float* x7811 = x7804+x7810;
for(int x7815=0; x7815 < x7813; x7815++) {
int32_t x7816 = x7815 / 9;
int32_t x7820 = x7816 * 3;
int32_t x7821 = x7820 * 3;
int32_t x7822 = x7821 * x7792;
int32_t x7823 = x7822 * x7792;
int32_t x7817 = x7815 % 9;
int32_t x7818 = x7817 / 3;
int32_t x7824 = x7818 * 3;
int32_t x7825 = x7824 * x7792;
int32_t x7826 = x7825 * x7792;
int32_t x7827 = x7823 + x7826;
int32_t x7819 = x7817 % 3;
int32_t x7828 = x7819 * x7792;
int32_t x7829 = x7828 * x7792;
int32_t x7830 = x7827 + x7829;
float* x7831 = x7811+x7830;
int32_t x7832 = x7816 * x7730;
int32_t x7833 = x7832 * x7730;
float* x7834 = x7807+x7833;
int32_t x7847 = 1 - x7819;
bool x7848 = x7847 > 0;
int32_t x7849;
if (x7848) {
x7849 = x7847;
} else {
x7849 = 0;
}
int32_t x7850 = 3 - x7819;
int32_t x7851 = x7850 - 1;
int32_t x7852 = 1 - x7851;
bool x7853 = x7852 > 0;
int32_t x7854;
if (x7853) {
x7854 = x7852;
} else {
x7854 = 0;
}
int32_t x7855 = x7792 - x7854;
int32_t x7856 = x7855 - x7849;
bool x7857 = x7856 <= 0;
bool x7861 = x7849 > 0;
int32_t x7846 = -1 + x7819;
bool x7874 = x7854 > 0;
for(int x7836=0; x7836 < x7792; x7836++) {
int32_t x7837 = x7836 - 1;
int32_t x7838 = x7837 + x7818;
bool x7839 = x7838 < 0;
bool x7840 = x7838 >= x7730;
bool x7841 = x7839 || x7840;
if (x7841) {
int32_t x7842 = x7836 * x7792;
float* x7843 = x7831+x7842;
memset(x7843, 0, 4 * x7792);;
} else {
if (x7857) {
int32_t x7842 = x7836 * x7792;
float* x7858 = x7831+x7842;
memset(x7858, 0, 4 * x7792);;
} else {
int32_t x7842 = x7836 * x7792;
if (x7861) {
float* x7862 = x7831+x7842;
memset(x7862, 0, 4 * x7849);;
} else {
}
// may have segfault here
int32_t x7867 = x7842 + x7849;
float* x7868 = x7831+x7867;
int32_t x7869 = x7838 * x7730;
int32_t x7870 = x7869 + x7846;
int32_t x7871 = x7870 + x7849;
float* x7872 = x7834+x7871;
memcpy(x7868, x7872, 4 * x7856);;
if (x7874) {
int32_t x7875 = x7842 + x7792;
int32_t x7876 = x7875 - x7854;
float* x7877 = x7831+x7876;
memset(x7877, 0, 4 * x7854);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x7793,x7799,1,x235,x7799,x7811,x7793,1,x7809,x7793);

}
int32_t x7892 = 0;
int32_t x7893 = 1;
x7893 *= 1;
x7892 += 1;
x7893 *= 1;
x7893 *= 1;
int32_t x7898 = x7892;
bool x7899 = x7898 >= 2;
if (x7899) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7904 = x7898 == 0;
if (x7904) {
int32_t x7905 = x7893;
bool x7906 = x7905 == 128;
if (x7906) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7913 = x7893;
int32_t x7914 = 128 / x7913;
bool x7915 = x7914 == 1;
bool x7918;
if (x454) {
bool x7916 = 128 == x7914;
bool x7917 = x7915 || x7916;
x7918 = x7917;
} else {
x7918 = false;
}
bool x7922;
if (x7918) {
x7922 = x7921;
} else {
x7922 = false;
}
bool x7923;
if (x7922) {
x7923 = x7921;
} else {
x7923 = false;
}
if (x7923) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x7792,x7792,1,x7914,1,1);
assert(false && "");
}
bool x7929 = 128 <= x7914;
int32_t x7930;
if (x7929) {
x7930 = x7914;
} else {
x7930 = 128;
}
int32_t x7939 = x7930 * x7938;
int32_t x7940 = 64 * x7939;
float* x7941 = (float*)myMalloc(x7940 * sizeof(float));;
int32_t x7944;
if (x7915) {
x7944 = 0;
} else {
x7944 = 1;
}
for(int x7945=0; x7945 < 64; x7945++) {
int32_t x7957 = x7794 * x7945;
int32_t x7951 = x7939 * x7945;
for(int x7947=0; x7947 < x7930; x7947++) {
int32_t x7958 = x7793 * x7947;
int32_t x7959 = x7957 + x7958;
int32_t x7964 = x7944 * x7947;
int32_t x7953 = x7938 * x7947;
for(int x7949=0; x7949 < x7932; x7949++) {
int32_t x7960 = x7942 * x7949;
int32_t x7961 = x7959 + x7960;
int32_t x7955 = x7932 * x7949;
for(int x7950=0; x7950 < x7932; x7950++) {
int32_t x7962 = x7943 * x7950;
int32_t x7963 = x7961 + x7962;
float x7965 = x7798[x7963];
float x7966 = x35[x7964];
int32_t x7952 = x7950 + x7951;
int32_t x7954 = x7952 + x7953;
int32_t x7956 = x7954 + x7955;
float x7967 = x7965 - x7966;
x7941[x7956] = x7967;

}

}

}

}
float* x7977 = (float*)myMalloc(128 * sizeof(float));;
for(int x7978=0; x7978 < 128; x7978++) {
float x7979 = x225[x7978];
float x7980 = x7979 + 1.0E-5f;
x7977[x7978] = x7980;

}
float* x7984 = (float*)myMalloc(128 * sizeof(float));;
for(int x7985=0; x7985 < 128; x7985++) {
float x7986 = x7977[x7985];
double x7987 = (double)x7986;
double x7988 = sqrt(x7987);
float x7989 = (float)x7988;
x7984[x7985] = x7989;

}
int32_t x7993 = 0;
int32_t x7994 = 1;
x7994 *= 1;
x7993 += 1;
x7994 *= 1;
x7994 *= 1;
int32_t x7999 = x7993;
bool x8000 = x7999 >= 2;
if (x8000) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8005 = x7999 == 0;
if (x8005) {
int32_t x8006 = x7994;
bool x8007 = x8006 == 128;
if (x8007) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8014 = x7994;
bool x8016 = x7930 == 1;
int32_t x8015 = 128 / x8014;
bool x8017 = x8015 == 1;
bool x8021;
if (x454) {
bool x8018 = x8016 || x8017;
bool x8019 = x7930 == x8015;
bool x8020 = x8018 || x8019;
x8021 = x8020;
} else {
x8021 = false;
}
bool x8025;
if (x8021) {
x8025 = x8024;
} else {
x8025 = false;
}
bool x8026;
if (x8025) {
x8026 = x8024;
} else {
x8026 = false;
}
if (x8026) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7930,x7932,x7932,1,x8015,1,1);
assert(false && "");
}
bool x8032 = x7930 <= x8015;
int32_t x8033;
if (x8032) {
x8033 = x8015;
} else {
x8033 = x7930;
}
int32_t x8042 = x8033 * x8041;
int32_t x8043 = 64 * x8042;
float* x8044 = (float*)myMalloc(x8043 * sizeof(float));;
int32_t x8045;
if (x8016) {
x8045 = 0;
} else {
x8045 = x7938;
}
int32_t x8048;
if (x8017) {
x8048 = 0;
} else {
x8048 = 1;
}
for(int x8049=0; x8049 < 64; x8049++) {
int32_t x8061 = x7939 * x8049;
int32_t x8055 = x8042 * x8049;
for(int x8051=0; x8051 < x8033; x8051++) {
int32_t x8062 = x8045 * x8051;
int32_t x8063 = x8061 + x8062;
int32_t x8068 = x8048 * x8051;
int32_t x8057 = x8041 * x8051;
for(int x8053=0; x8053 < x8035; x8053++) {
int32_t x8064 = x8046 * x8053;
int32_t x8065 = x8063 + x8064;
int32_t x8059 = x8035 * x8053;
for(int x8054=0; x8054 < x8035; x8054++) {
int32_t x8066 = x8047 * x8054;
int32_t x8067 = x8065 + x8066;
float x8069 = x7941[x8067];
float x8070 = x7984[x8068];
int32_t x8056 = x8054 + x8055;
int32_t x8058 = x8056 + x8057;
int32_t x8060 = x8058 + x8059;
float x8071 = x8069 / x8070;
x8044[x8060] = x8071;

}

}

}

}
int32_t x8081 = 0;
int32_t x8082 = 1;
x8082 *= 1;
x8081 += 1;
x8082 *= 1;
x8082 *= 1;
int32_t x8087 = x8081;
bool x8088 = x8087 >= 2;
if (x8088) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8093 = x8087 == 0;
if (x8093) {
int32_t x8094 = x8082;
bool x8095 = x8094 == 128;
if (x8095) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8102 = x8082;
bool x8104 = x8033 == 1;
int32_t x8103 = 128 / x8102;
bool x8105 = x8103 == 1;
bool x8109;
if (x454) {
bool x8106 = x8104 || x8105;
bool x8107 = x8033 == x8103;
bool x8108 = x8106 || x8107;
x8109 = x8108;
} else {
x8109 = false;
}
bool x8113;
if (x8109) {
x8113 = x8112;
} else {
x8113 = false;
}
bool x8114;
if (x8113) {
x8114 = x8112;
} else {
x8114 = false;
}
if (x8114) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8033,x8035,x8035,1,x8103,1,1);
assert(false && "");
}
bool x8120 = x8033 <= x8103;
int32_t x8121;
if (x8120) {
x8121 = x8103;
} else {
x8121 = x8033;
}
int32_t x8130 = x8121 * x8129;
int32_t x8131 = 64 * x8130;
float* x8132 = (float*)myMalloc(x8131 * sizeof(float));;
int32_t x8133;
if (x8104) {
x8133 = 0;
} else {
x8133 = x8041;
}
int32_t x8136;
if (x8105) {
x8136 = 0;
} else {
x8136 = 1;
}
for(int x8137=0; x8137 < 64; x8137++) {
int32_t x8149 = x8042 * x8137;
int32_t x8143 = x8130 * x8137;
for(int x8139=0; x8139 < x8121; x8139++) {
int32_t x8150 = x8133 * x8139;
int32_t x8151 = x8149 + x8150;
int32_t x8156 = x8136 * x8139;
int32_t x8145 = x8129 * x8139;
for(int x8141=0; x8141 < x8123; x8141++) {
int32_t x8152 = x8134 * x8141;
int32_t x8153 = x8151 + x8152;
int32_t x8147 = x8123 * x8141;
for(int x8142=0; x8142 < x8123; x8142++) {
int32_t x8154 = x8135 * x8142;
int32_t x8155 = x8153 + x8154;
float x8157 = x8044[x8155];
float x8158 = x8[x8156];
int32_t x8144 = x8142 + x8143;
int32_t x8146 = x8144 + x8145;
int32_t x8148 = x8146 + x8147;
float x8159 = x8157 * x8158;
x8132[x8148] = x8159;

}

}

}

}
int32_t x8169 = 0;
int32_t x8170 = 1;
x8170 *= 1;
x8169 += 1;
x8170 *= 1;
x8170 *= 1;
int32_t x8175 = x8169;
bool x8176 = x8175 >= 2;
if (x8176) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8181 = x8175 == 0;
if (x8181) {
int32_t x8182 = x8170;
bool x8183 = x8182 == 128;
if (x8183) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8190 = x8170;
bool x8192 = x8121 == 1;
int32_t x8191 = 128 / x8190;
bool x8193 = x8191 == 1;
bool x8197;
if (x454) {
bool x8194 = x8192 || x8193;
bool x8195 = x8121 == x8191;
bool x8196 = x8194 || x8195;
x8197 = x8196;
} else {
x8197 = false;
}
bool x8201;
if (x8197) {
x8201 = x8200;
} else {
x8201 = false;
}
bool x8202;
if (x8201) {
x8202 = x8200;
} else {
x8202 = false;
}
if (x8202) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8121,x8123,x8123,1,x8191,1,1);
assert(false && "");
}
bool x8208 = x8121 <= x8191;
int32_t x8209;
if (x8208) {
x8209 = x8191;
} else {
x8209 = x8121;
}
int32_t x8218 = x8209 * x8217;
int32_t x8219 = 64 * x8218;
float* x8220 = (float*)myMalloc(x8219 * sizeof(float));;
int32_t x8221;
if (x8192) {
x8221 = 0;
} else {
x8221 = x8129;
}
int32_t x8224;
if (x8193) {
x8224 = 0;
} else {
x8224 = 1;
}
for(int x8225=0; x8225 < 64; x8225++) {
int32_t x8237 = x8130 * x8225;
int32_t x8231 = x8218 * x8225;
for(int x8227=0; x8227 < x8209; x8227++) {
int32_t x8238 = x8221 * x8227;
int32_t x8239 = x8237 + x8238;
int32_t x8244 = x8224 * x8227;
int32_t x8233 = x8217 * x8227;
for(int x8229=0; x8229 < x8211; x8229++) {
int32_t x8240 = x8222 * x8229;
int32_t x8241 = x8239 + x8240;
int32_t x8235 = x8211 * x8229;
for(int x8230=0; x8230 < x8211; x8230++) {
int32_t x8242 = x8223 * x8230;
int32_t x8243 = x8241 + x8242;
float x8245 = x8132[x8243];
float x8246 = x95[x8244];
int32_t x8232 = x8230 + x8231;
int32_t x8234 = x8232 + x8233;
int32_t x8236 = x8234 + x8235;
float x8247 = x8245 + x8246;
x8220[x8236] = x8247;

}

}

}

}
float* x8257 = (float*)myMalloc(x8219 * sizeof(float));;
for(int x8259=0; x8259 < x8219; x8259++) {
float x8260 = x8220[x8259];
bool x8261 = x8260 < 0.0f;
if (x8261) {
x8257[x8259] = 0.0f;
} else {
float x8264 = x8220[x8259];
x8257[x8259] = x8264;
}

}
float* x8278 = (float*)myMalloc(x8277 * sizeof(float));;
int32_t x8281 = 64 * x8209;
int32_t x8282 = x8281 * x8273;
float* x8283 = (float*)myMalloc(x8282 * sizeof(float));;
int32_t x8279 = x8209 * x8273;
for(int x8284=0; x8284 < 64; x8284++) {
int32_t x8285 = x8284 * x8218;
float* x8286 = x8257+x8285;
int32_t x8287 = x8284 * x8274;
float* x8288 = x8278+x8287;
int32_t x8289 = x8284 * x8279;
float* x8290 = x8283+x8289;
for(int x8291=0; x8291 < x8209; x8291++) {
int32_t x8292 = x8291 / 1;
int32_t x8296 = x8292 * x8272;
int32_t x8297 = x8296 * x8272;
int32_t x8293 = x8291 % 1;
int32_t x8294 = x8293 / 1;
int32_t x8298 = x8294 * x8272;
int32_t x8299 = x8298 * x8272;
int32_t x8300 = x8297 + x8299;
int32_t x8295 = x8293 % 1;
int32_t x8301 = x8295 * x8272;
int32_t x8302 = x8301 * x8272;
int32_t x8303 = x8300 + x8302;
float* x8304 = x8290+x8303;
int32_t x8305 = x8292 * x8211;
int32_t x8306 = x8305 * x8211;
float* x8307 = x8286+x8306;
for(int x8309=0; x8309 < x8272; x8309++) {
int32_t x8311 = x8309 * x8272;
float* x8312 = x8304+x8311;
int32_t x8310 = x8309 + x8294;
int32_t x8313 = x8310 * x8211;
int32_t x8314 = x8313 + x8295;
float* x8315 = x8307+x8314;
memcpy(x8312, x8315, 4 * x8272);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x8273,x8209,1,x111,x8209,x8290,x8273,1,x8288,x8273);

}
int32_t x8324 = 0;
int32_t x8325 = 1;
x8325 *= 1;
x8324 += 1;
x8325 *= 1;
x8325 *= 1;
int32_t x8330 = x8324;
bool x8331 = x8330 >= 2;
if (x8331) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8336 = x8330 == 0;
if (x8336) {
int32_t x8337 = x8325;
bool x8338 = x8337 == 512;
if (x8338) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8345 = x8325;
int32_t x8346 = 512 / x8345;
bool x8347 = x8346 == 1;
bool x8350;
if (x454) {
bool x8348 = 512 == x8346;
bool x8349 = x8347 || x8348;
x8350 = x8349;
} else {
x8350 = false;
}
bool x8354;
if (x8350) {
x8354 = x8353;
} else {
x8354 = false;
}
bool x8355;
if (x8354) {
x8355 = x8353;
} else {
x8355 = false;
}
if (x8355) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x8272,x8272,1,x8346,1,1);
assert(false && "");
}
bool x8361 = 512 <= x8346;
int32_t x8362;
if (x8361) {
x8362 = x8346;
} else {
x8362 = 512;
}
int32_t x8371 = x8362 * x8370;
int32_t x8372 = 64 * x8371;
float* x8373 = (float*)myMalloc(x8372 * sizeof(float));;
int32_t x8376;
if (x8347) {
x8376 = 0;
} else {
x8376 = 1;
}
for(int x8377=0; x8377 < 64; x8377++) {
int32_t x8389 = x8274 * x8377;
int32_t x8383 = x8371 * x8377;
for(int x8379=0; x8379 < x8362; x8379++) {
int32_t x8390 = x8273 * x8379;
int32_t x8391 = x8389 + x8390;
int32_t x8396 = x8376 * x8379;
int32_t x8385 = x8370 * x8379;
for(int x8381=0; x8381 < x8364; x8381++) {
int32_t x8392 = x8374 * x8381;
int32_t x8393 = x8391 + x8392;
int32_t x8387 = x8364 * x8381;
for(int x8382=0; x8382 < x8364; x8382++) {
int32_t x8394 = x8375 * x8382;
int32_t x8395 = x8393 + x8394;
float x8397 = x8278[x8395];
float x8398 = x147[x8396];
int32_t x8384 = x8382 + x8383;
int32_t x8386 = x8384 + x8385;
int32_t x8388 = x8386 + x8387;
float x8399 = x8397 - x8398;
x8373[x8388] = x8399;

}

}

}

}
float* x8409 = (float*)myMalloc(512 * sizeof(float));;
for(int x8410=0; x8410 < 512; x8410++) {
float x8411 = x88[x8410];
float x8412 = x8411 + 1.0E-5f;
x8409[x8410] = x8412;

}
float* x8416 = (float*)myMalloc(512 * sizeof(float));;
for(int x8417=0; x8417 < 512; x8417++) {
float x8418 = x8409[x8417];
double x8419 = (double)x8418;
double x8420 = sqrt(x8419);
float x8421 = (float)x8420;
x8416[x8417] = x8421;

}
int32_t x8425 = 0;
int32_t x8426 = 1;
x8426 *= 1;
x8425 += 1;
x8426 *= 1;
x8426 *= 1;
int32_t x8431 = x8425;
bool x8432 = x8431 >= 2;
if (x8432) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8437 = x8431 == 0;
if (x8437) {
int32_t x8438 = x8426;
bool x8439 = x8438 == 512;
if (x8439) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8446 = x8426;
bool x8448 = x8362 == 1;
int32_t x8447 = 512 / x8446;
bool x8449 = x8447 == 1;
bool x8453;
if (x454) {
bool x8450 = x8448 || x8449;
bool x8451 = x8362 == x8447;
bool x8452 = x8450 || x8451;
x8453 = x8452;
} else {
x8453 = false;
}
bool x8457;
if (x8453) {
x8457 = x8456;
} else {
x8457 = false;
}
bool x8458;
if (x8457) {
x8458 = x8456;
} else {
x8458 = false;
}
if (x8458) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8362,x8364,x8364,1,x8447,1,1);
assert(false && "");
}
bool x8464 = x8362 <= x8447;
int32_t x8465;
if (x8464) {
x8465 = x8447;
} else {
x8465 = x8362;
}
int32_t x8474 = x8465 * x8473;
int32_t x8475 = 64 * x8474;
float* x8476 = (float*)myMalloc(x8475 * sizeof(float));;
int32_t x8477;
if (x8448) {
x8477 = 0;
} else {
x8477 = x8370;
}
int32_t x8480;
if (x8449) {
x8480 = 0;
} else {
x8480 = 1;
}
for(int x8481=0; x8481 < 64; x8481++) {
int32_t x8493 = x8371 * x8481;
int32_t x8487 = x8474 * x8481;
for(int x8483=0; x8483 < x8465; x8483++) {
int32_t x8494 = x8477 * x8483;
int32_t x8495 = x8493 + x8494;
int32_t x8500 = x8480 * x8483;
int32_t x8489 = x8473 * x8483;
for(int x8485=0; x8485 < x8467; x8485++) {
int32_t x8496 = x8478 * x8485;
int32_t x8497 = x8495 + x8496;
int32_t x8491 = x8467 * x8485;
for(int x8486=0; x8486 < x8467; x8486++) {
int32_t x8498 = x8479 * x8486;
int32_t x8499 = x8497 + x8498;
float x8501 = x8373[x8499];
float x8502 = x8416[x8500];
int32_t x8488 = x8486 + x8487;
int32_t x8490 = x8488 + x8489;
int32_t x8492 = x8490 + x8491;
float x8503 = x8501 / x8502;
x8476[x8492] = x8503;

}

}

}

}
int32_t x8513 = 0;
int32_t x8514 = 1;
x8514 *= 1;
x8513 += 1;
x8514 *= 1;
x8514 *= 1;
int32_t x8519 = x8513;
bool x8520 = x8519 >= 2;
if (x8520) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8525 = x8519 == 0;
if (x8525) {
int32_t x8526 = x8514;
bool x8527 = x8526 == 512;
if (x8527) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8534 = x8514;
bool x8536 = x8465 == 1;
int32_t x8535 = 512 / x8534;
bool x8537 = x8535 == 1;
bool x8541;
if (x454) {
bool x8538 = x8536 || x8537;
bool x8539 = x8465 == x8535;
bool x8540 = x8538 || x8539;
x8541 = x8540;
} else {
x8541 = false;
}
bool x8545;
if (x8541) {
x8545 = x8544;
} else {
x8545 = false;
}
bool x8546;
if (x8545) {
x8546 = x8544;
} else {
x8546 = false;
}
if (x8546) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8465,x8467,x8467,1,x8535,1,1);
assert(false && "");
}
bool x8552 = x8465 <= x8535;
int32_t x8553;
if (x8552) {
x8553 = x8535;
} else {
x8553 = x8465;
}
int32_t x8562 = x8553 * x8561;
int32_t x8563 = 64 * x8562;
float* x8564 = (float*)myMalloc(x8563 * sizeof(float));;
int32_t x8565;
if (x8536) {
x8565 = 0;
} else {
x8565 = x8473;
}
int32_t x8568;
if (x8537) {
x8568 = 0;
} else {
x8568 = 1;
}
for(int x8569=0; x8569 < 64; x8569++) {
int32_t x8581 = x8474 * x8569;
int32_t x8575 = x8562 * x8569;
for(int x8571=0; x8571 < x8553; x8571++) {
int32_t x8582 = x8565 * x8571;
int32_t x8583 = x8581 + x8582;
int32_t x8588 = x8568 * x8571;
int32_t x8577 = x8561 * x8571;
for(int x8573=0; x8573 < x8555; x8573++) {
int32_t x8584 = x8566 * x8573;
int32_t x8585 = x8583 + x8584;
int32_t x8579 = x8555 * x8573;
for(int x8574=0; x8574 < x8555; x8574++) {
int32_t x8586 = x8567 * x8574;
int32_t x8587 = x8585 + x8586;
float x8589 = x8476[x8587];
float x8590 = x52[x8588];
int32_t x8576 = x8574 + x8575;
int32_t x8578 = x8576 + x8577;
int32_t x8580 = x8578 + x8579;
float x8591 = x8589 * x8590;
x8564[x8580] = x8591;

}

}

}

}
int32_t x8601 = 0;
int32_t x8602 = 1;
x8602 *= 1;
x8601 += 1;
x8602 *= 1;
x8602 *= 1;
int32_t x8607 = x8601;
bool x8608 = x8607 >= 2;
if (x8608) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8613 = x8607 == 0;
if (x8613) {
int32_t x8614 = x8602;
bool x8615 = x8614 == 512;
if (x8615) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8622 = x8602;
bool x8624 = x8553 == 1;
int32_t x8623 = 512 / x8622;
bool x8625 = x8623 == 1;
bool x8629;
if (x454) {
bool x8626 = x8624 || x8625;
bool x8627 = x8553 == x8623;
bool x8628 = x8626 || x8627;
x8629 = x8628;
} else {
x8629 = false;
}
bool x8633;
if (x8629) {
x8633 = x8632;
} else {
x8633 = false;
}
bool x8634;
if (x8633) {
x8634 = x8632;
} else {
x8634 = false;
}
if (x8634) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8553,x8555,x8555,1,x8623,1,1);
assert(false && "");
}
bool x8640 = x8553 <= x8623;
int32_t x8641;
if (x8640) {
x8641 = x8623;
} else {
x8641 = x8553;
}
int32_t x8650 = x8641 * x8649;
int32_t x8651 = 64 * x8650;
float* x8652 = (float*)myMalloc(x8651 * sizeof(float));;
int32_t x8653;
if (x8624) {
x8653 = 0;
} else {
x8653 = x8561;
}
int32_t x8656;
if (x8625) {
x8656 = 0;
} else {
x8656 = 1;
}
for(int x8657=0; x8657 < 64; x8657++) {
int32_t x8669 = x8562 * x8657;
int32_t x8663 = x8650 * x8657;
for(int x8659=0; x8659 < x8641; x8659++) {
int32_t x8670 = x8653 * x8659;
int32_t x8671 = x8669 + x8670;
int32_t x8676 = x8656 * x8659;
int32_t x8665 = x8649 * x8659;
for(int x8661=0; x8661 < x8643; x8661++) {
int32_t x8672 = x8654 * x8661;
int32_t x8673 = x8671 + x8672;
int32_t x8667 = x8643 * x8661;
for(int x8662=0; x8662 < x8643; x8662++) {
int32_t x8674 = x8655 * x8662;
int32_t x8675 = x8673 + x8674;
float x8677 = x8564[x8675];
float x8678 = x246[x8676];
int32_t x8664 = x8662 + x8663;
int32_t x8666 = x8664 + x8665;
int32_t x8668 = x8666 + x8667;
float x8679 = x8677 + x8678;
x8652[x8668] = x8679;

}

}

}

}
bool x8689 = x8641 == 1;
bool x8690 = x8689 || x7272;
bool x8691 = x8641 == x6800;
bool x8692 = x8690 || x8691;
bool x8697;
if (x8692) {
x8697 = x8696;
} else {
x8697 = false;
}
bool x8698;
if (x8697) {
x8698 = x8696;
} else {
x8698 = false;
}
if (x8698) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8641,x8643,x8643,64,x6800,x6802,x6802);
assert(false && "");
}
bool x8704 = x8641 <= x6800;
int32_t x8705;
if (x8704) {
x8705 = x6800;
} else {
x8705 = x8641;
}
int32_t x8721;
if (x8689) {
x8721 = 0;
} else {
x8721 = x8649;
}
for(int x8724=0; x8724 < 64; x8724++) {
int32_t x8730 = x8650 * x8724;
int32_t x8737 = x6809 * x8724;
for(int x8726=0; x8726 < x8705; x8726++) {
int32_t x8731 = x8721 * x8726;
int32_t x8732 = x8730 + x8731;
int32_t x8738 = x7306 * x8726;
int32_t x8739 = x8737 + x8738;
for(int x8728=0; x8728 < x8707; x8728++) {
int32_t x8733 = x8722 * x8728;
int32_t x8734 = x8732 + x8733;
int32_t x8740 = x7307 * x8728;
int32_t x8741 = x8739 + x8740;
for(int x8729=0; x8729 < x8707; x8729++) {
int32_t x8735 = x8723 * x8729;
int32_t x8736 = x8734 + x8735;
float x8744 = x8652[x8736];
int32_t x8742 = x7308 * x8729;
int32_t x8743 = x8741 + x8742;
float x8745 = x7344[x8743];
float x8746 = x8744 + x8745;
x8652[x8736] = x8746;

}

}

}

}
float* x8756 = (float*)myMalloc(x8651 * sizeof(float));;
for(int x8758=0; x8758 < x8651; x8758++) {
float x8759 = x8652[x8758];
bool x8760 = x8759 < 0.0f;
if (x8760) {
x8756[x8758] = 0.0f;
} else {
float x8763 = x8652[x8758];
x8756[x8758] = x8763;
}

}
float* x8777 = (float*)myMalloc(x8776 * sizeof(float));;
int32_t x8780 = 64 * x8641;
int32_t x8781 = x8780 * x8772;
float* x8782 = (float*)myMalloc(x8781 * sizeof(float));;
int32_t x8778 = x8641 * x8772;
for(int x8783=0; x8783 < 64; x8783++) {
int32_t x8784 = x8783 * x8650;
float* x8785 = x8756+x8784;
int32_t x8786 = x8783 * x8773;
float* x8787 = x8777+x8786;
int32_t x8788 = x8783 * x8778;
float* x8789 = x8782+x8788;
for(int x8790=0; x8790 < x8641; x8790++) {
int32_t x8791 = x8790 / 1;
int32_t x8795 = x8791 * x8771;
int32_t x8796 = x8795 * x8771;
int32_t x8792 = x8790 % 1;
int32_t x8793 = x8792 / 1;
int32_t x8797 = x8793 * x8771;
int32_t x8798 = x8797 * x8771;
int32_t x8799 = x8796 + x8798;
int32_t x8794 = x8792 % 1;
int32_t x8800 = x8794 * x8771;
int32_t x8801 = x8800 * x8771;
int32_t x8802 = x8799 + x8801;
float* x8803 = x8789+x8802;
int32_t x8804 = x8791 * x8643;
int32_t x8805 = x8804 * x8643;
float* x8806 = x8785+x8805;
for(int x8808=0; x8808 < x8771; x8808++) {
int32_t x8810 = x8808 * x8771;
float* x8811 = x8803+x8810;
int32_t x8809 = x8808 + x8793;
int32_t x8812 = x8809 * x8643;
int32_t x8813 = x8812 + x8794;
float* x8814 = x8806+x8813;
memcpy(x8811, x8814, 4 * x8771);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x8772,x8641,1,x196,x8641,x8789,x8772,1,x8787,x8772);

}
int32_t x8823 = 0;
int32_t x8824 = 1;
x8824 *= 1;
x8823 += 1;
x8824 *= 1;
x8824 *= 1;
int32_t x8829 = x8823;
bool x8830 = x8829 >= 2;
if (x8830) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8835 = x8829 == 0;
if (x8835) {
int32_t x8836 = x8824;
bool x8837 = x8836 == 128;
if (x8837) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8844 = x8824;
int32_t x8845 = 128 / x8844;
bool x8846 = x8845 == 1;
bool x8849;
if (x454) {
bool x8847 = 128 == x8845;
bool x8848 = x8846 || x8847;
x8849 = x8848;
} else {
x8849 = false;
}
bool x8853;
if (x8849) {
x8853 = x8852;
} else {
x8853 = false;
}
bool x8854;
if (x8853) {
x8854 = x8852;
} else {
x8854 = false;
}
if (x8854) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x8771,x8771,1,x8845,1,1);
assert(false && "");
}
bool x8860 = 128 <= x8845;
int32_t x8861;
if (x8860) {
x8861 = x8845;
} else {
x8861 = 128;
}
int32_t x8870 = x8861 * x8869;
int32_t x8871 = 64 * x8870;
float* x8872 = (float*)myMalloc(x8871 * sizeof(float));;
int32_t x8875;
if (x8846) {
x8875 = 0;
} else {
x8875 = 1;
}
for(int x8876=0; x8876 < 64; x8876++) {
int32_t x8888 = x8773 * x8876;
int32_t x8882 = x8870 * x8876;
for(int x8878=0; x8878 < x8861; x8878++) {
int32_t x8889 = x8772 * x8878;
int32_t x8890 = x8888 + x8889;
int32_t x8895 = x8875 * x8878;
int32_t x8884 = x8869 * x8878;
for(int x8880=0; x8880 < x8863; x8880++) {
int32_t x8891 = x8873 * x8880;
int32_t x8892 = x8890 + x8891;
int32_t x8886 = x8863 * x8880;
for(int x8881=0; x8881 < x8863; x8881++) {
int32_t x8893 = x8874 * x8881;
int32_t x8894 = x8892 + x8893;
float x8896 = x8777[x8894];
float x8897 = x112[x8895];
int32_t x8883 = x8881 + x8882;
int32_t x8885 = x8883 + x8884;
int32_t x8887 = x8885 + x8886;
float x8898 = x8896 - x8897;
x8872[x8887] = x8898;

}

}

}

}
float* x8908 = (float*)myMalloc(128 * sizeof(float));;
for(int x8909=0; x8909 < 128; x8909++) {
float x8910 = x9[x8909];
float x8911 = x8910 + 1.0E-5f;
x8908[x8909] = x8911;

}
float* x8915 = (float*)myMalloc(128 * sizeof(float));;
for(int x8916=0; x8916 < 128; x8916++) {
float x8917 = x8908[x8916];
double x8918 = (double)x8917;
double x8919 = sqrt(x8918);
float x8920 = (float)x8919;
x8915[x8916] = x8920;

}
int32_t x8924 = 0;
int32_t x8925 = 1;
x8925 *= 1;
x8924 += 1;
x8925 *= 1;
x8925 *= 1;
int32_t x8930 = x8924;
bool x8931 = x8930 >= 2;
if (x8931) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8936 = x8930 == 0;
if (x8936) {
int32_t x8937 = x8925;
bool x8938 = x8937 == 128;
if (x8938) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8945 = x8925;
bool x8947 = x8861 == 1;
int32_t x8946 = 128 / x8945;
bool x8948 = x8946 == 1;
bool x8952;
if (x454) {
bool x8949 = x8947 || x8948;
bool x8950 = x8861 == x8946;
bool x8951 = x8949 || x8950;
x8952 = x8951;
} else {
x8952 = false;
}
bool x8956;
if (x8952) {
x8956 = x8955;
} else {
x8956 = false;
}
bool x8957;
if (x8956) {
x8957 = x8955;
} else {
x8957 = false;
}
if (x8957) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8861,x8863,x8863,1,x8946,1,1);
assert(false && "");
}
bool x8963 = x8861 <= x8946;
int32_t x8964;
if (x8963) {
x8964 = x8946;
} else {
x8964 = x8861;
}
int32_t x8973 = x8964 * x8972;
int32_t x8974 = 64 * x8973;
float* x8975 = (float*)myMalloc(x8974 * sizeof(float));;
int32_t x8976;
if (x8947) {
x8976 = 0;
} else {
x8976 = x8869;
}
int32_t x8979;
if (x8948) {
x8979 = 0;
} else {
x8979 = 1;
}
for(int x8980=0; x8980 < 64; x8980++) {
int32_t x8992 = x8870 * x8980;
int32_t x8986 = x8973 * x8980;
for(int x8982=0; x8982 < x8964; x8982++) {
int32_t x8993 = x8976 * x8982;
int32_t x8994 = x8992 + x8993;
int32_t x8999 = x8979 * x8982;
int32_t x8988 = x8972 * x8982;
for(int x8984=0; x8984 < x8966; x8984++) {
int32_t x8995 = x8977 * x8984;
int32_t x8996 = x8994 + x8995;
int32_t x8990 = x8966 * x8984;
for(int x8985=0; x8985 < x8966; x8985++) {
int32_t x8997 = x8978 * x8985;
int32_t x8998 = x8996 + x8997;
float x9000 = x8872[x8998];
float x9001 = x8915[x8999];
int32_t x8987 = x8985 + x8986;
int32_t x8989 = x8987 + x8988;
int32_t x8991 = x8989 + x8990;
float x9002 = x9000 / x9001;
x8975[x8991] = x9002;

}

}

}

}
int32_t x9012 = 0;
int32_t x9013 = 1;
x9013 *= 1;
x9012 += 1;
x9013 *= 1;
x9013 *= 1;
int32_t x9018 = x9012;
bool x9019 = x9018 >= 2;
if (x9019) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9024 = x9018 == 0;
if (x9024) {
int32_t x9025 = x9013;
bool x9026 = x9025 == 128;
if (x9026) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9033 = x9013;
bool x9035 = x8964 == 1;
int32_t x9034 = 128 / x9033;
bool x9036 = x9034 == 1;
bool x9040;
if (x454) {
bool x9037 = x9035 || x9036;
bool x9038 = x8964 == x9034;
bool x9039 = x9037 || x9038;
x9040 = x9039;
} else {
x9040 = false;
}
bool x9044;
if (x9040) {
x9044 = x9043;
} else {
x9044 = false;
}
bool x9045;
if (x9044) {
x9045 = x9043;
} else {
x9045 = false;
}
if (x9045) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8964,x8966,x8966,1,x9034,1,1);
assert(false && "");
}
bool x9051 = x8964 <= x9034;
int32_t x9052;
if (x9051) {
x9052 = x9034;
} else {
x9052 = x8964;
}
int32_t x9061 = x9052 * x9060;
int32_t x9062 = 64 * x9061;
float* x9063 = (float*)myMalloc(x9062 * sizeof(float));;
int32_t x9064;
if (x9035) {
x9064 = 0;
} else {
x9064 = x8972;
}
int32_t x9067;
if (x9036) {
x9067 = 0;
} else {
x9067 = 1;
}
for(int x9068=0; x9068 < 64; x9068++) {
int32_t x9080 = x8973 * x9068;
int32_t x9074 = x9061 * x9068;
for(int x9070=0; x9070 < x9052; x9070++) {
int32_t x9081 = x9064 * x9070;
int32_t x9082 = x9080 + x9081;
int32_t x9087 = x9067 * x9070;
int32_t x9076 = x9060 * x9070;
for(int x9072=0; x9072 < x9054; x9072++) {
int32_t x9083 = x9065 * x9072;
int32_t x9084 = x9082 + x9083;
int32_t x9078 = x9054 * x9072;
for(int x9073=0; x9073 < x9054; x9073++) {
int32_t x9085 = x9066 * x9073;
int32_t x9086 = x9084 + x9085;
float x9088 = x8975[x9086];
float x9089 = x45[x9087];
int32_t x9075 = x9073 + x9074;
int32_t x9077 = x9075 + x9076;
int32_t x9079 = x9077 + x9078;
float x9090 = x9088 * x9089;
x9063[x9079] = x9090;

}

}

}

}
int32_t x9100 = 0;
int32_t x9101 = 1;
x9101 *= 1;
x9100 += 1;
x9101 *= 1;
x9101 *= 1;
int32_t x9106 = x9100;
bool x9107 = x9106 >= 2;
if (x9107) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9112 = x9106 == 0;
if (x9112) {
int32_t x9113 = x9101;
bool x9114 = x9113 == 128;
if (x9114) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9121 = x9101;
bool x9123 = x9052 == 1;
int32_t x9122 = 128 / x9121;
bool x9124 = x9122 == 1;
bool x9128;
if (x454) {
bool x9125 = x9123 || x9124;
bool x9126 = x9052 == x9122;
bool x9127 = x9125 || x9126;
x9128 = x9127;
} else {
x9128 = false;
}
bool x9132;
if (x9128) {
x9132 = x9131;
} else {
x9132 = false;
}
bool x9133;
if (x9132) {
x9133 = x9131;
} else {
x9133 = false;
}
if (x9133) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9052,x9054,x9054,1,x9122,1,1);
assert(false && "");
}
bool x9139 = x9052 <= x9122;
int32_t x9140;
if (x9139) {
x9140 = x9122;
} else {
x9140 = x9052;
}
int32_t x9149 = x9140 * x9148;
int32_t x9150 = 64 * x9149;
float* x9151 = (float*)myMalloc(x9150 * sizeof(float));;
int32_t x9152;
if (x9123) {
x9152 = 0;
} else {
x9152 = x9060;
}
int32_t x9155;
if (x9124) {
x9155 = 0;
} else {
x9155 = 1;
}
for(int x9156=0; x9156 < 64; x9156++) {
int32_t x9168 = x9061 * x9156;
int32_t x9162 = x9149 * x9156;
for(int x9158=0; x9158 < x9140; x9158++) {
int32_t x9169 = x9152 * x9158;
int32_t x9170 = x9168 + x9169;
int32_t x9175 = x9155 * x9158;
int32_t x9164 = x9148 * x9158;
for(int x9160=0; x9160 < x9142; x9160++) {
int32_t x9171 = x9153 * x9160;
int32_t x9172 = x9170 + x9171;
int32_t x9166 = x9142 * x9160;
for(int x9161=0; x9161 < x9142; x9161++) {
int32_t x9173 = x9154 * x9161;
int32_t x9174 = x9172 + x9173;
float x9176 = x9063[x9174];
float x9177 = x170[x9175];
int32_t x9163 = x9161 + x9162;
int32_t x9165 = x9163 + x9164;
int32_t x9167 = x9165 + x9166;
float x9178 = x9176 + x9177;
x9151[x9167] = x9178;

}

}

}

}
float* x9188 = (float*)myMalloc(x9150 * sizeof(float));;
for(int x9190=0; x9190 < x9150; x9190++) {
float x9191 = x9151[x9190];
bool x9192 = x9191 < 0.0f;
if (x9192) {
x9188[x9190] = 0.0f;
} else {
float x9195 = x9151[x9190];
x9188[x9190] = x9195;
}

}
float* x9210 = (float*)myMalloc(x9209 * sizeof(float));;
int32_t x9211 = 9 * x9140;
int32_t x9214 = 64 * x9211;
int32_t x9215 = x9214 * x9205;
float* x9216 = (float*)myMalloc(x9215 * sizeof(float));;
int32_t x9212 = x9211 * x9205;
int32_t x9224 = x9140 * 3;
int32_t x9225 = x9224 * 3;
for(int x9217=0; x9217 < 64; x9217++) {
int32_t x9218 = x9217 * x9149;
float* x9219 = x9188+x9218;
int32_t x9220 = x9217 * x9206;
float* x9221 = x9210+x9220;
int32_t x9222 = x9217 * x9212;
float* x9223 = x9216+x9222;
for(int x9227=0; x9227 < x9225; x9227++) {
int32_t x9228 = x9227 / 9;
int32_t x9232 = x9228 * 3;
int32_t x9233 = x9232 * 3;
int32_t x9234 = x9233 * x9204;
int32_t x9235 = x9234 * x9204;
int32_t x9229 = x9227 % 9;
int32_t x9230 = x9229 / 3;
int32_t x9236 = x9230 * 3;
int32_t x9237 = x9236 * x9204;
int32_t x9238 = x9237 * x9204;
int32_t x9239 = x9235 + x9238;
int32_t x9231 = x9229 % 3;
int32_t x9240 = x9231 * x9204;
int32_t x9241 = x9240 * x9204;
int32_t x9242 = x9239 + x9241;
float* x9243 = x9223+x9242;
int32_t x9244 = x9228 * x9142;
int32_t x9245 = x9244 * x9142;
float* x9246 = x9219+x9245;
int32_t x9259 = 1 - x9231;
bool x9260 = x9259 > 0;
int32_t x9261;
if (x9260) {
x9261 = x9259;
} else {
x9261 = 0;
}
int32_t x9262 = 3 - x9231;
int32_t x9263 = x9262 - 1;
int32_t x9264 = 1 - x9263;
bool x9265 = x9264 > 0;
int32_t x9266;
if (x9265) {
x9266 = x9264;
} else {
x9266 = 0;
}
int32_t x9267 = x9204 - x9266;
int32_t x9268 = x9267 - x9261;
bool x9269 = x9268 <= 0;
bool x9273 = x9261 > 0;
int32_t x9258 = -1 + x9231;
bool x9286 = x9266 > 0;
for(int x9248=0; x9248 < x9204; x9248++) {
int32_t x9249 = x9248 - 1;
int32_t x9250 = x9249 + x9230;
bool x9251 = x9250 < 0;
bool x9252 = x9250 >= x9142;
bool x9253 = x9251 || x9252;
if (x9253) {
int32_t x9254 = x9248 * x9204;
float* x9255 = x9243+x9254;
memset(x9255, 0, 4 * x9204);;
} else {
if (x9269) {
int32_t x9254 = x9248 * x9204;
float* x9270 = x9243+x9254;
memset(x9270, 0, 4 * x9204);;
} else {
int32_t x9254 = x9248 * x9204;
if (x9273) {
float* x9274 = x9243+x9254;
memset(x9274, 0, 4 * x9261);;
} else {
}
// may have segfault here
int32_t x9279 = x9254 + x9261;
float* x9280 = x9243+x9279;
int32_t x9281 = x9250 * x9142;
int32_t x9282 = x9281 + x9258;
int32_t x9283 = x9282 + x9261;
float* x9284 = x9246+x9283;
memcpy(x9280, x9284, 4 * x9268);;
if (x9286) {
int32_t x9287 = x9254 + x9204;
int32_t x9288 = x9287 - x9266;
float* x9289 = x9243+x9288;
memset(x9289, 0, 4 * x9266);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x9205,x9211,1,x191,x9211,x9223,x9205,1,x9221,x9205);

}
int32_t x9304 = 0;
int32_t x9305 = 1;
x9305 *= 1;
x9304 += 1;
x9305 *= 1;
x9305 *= 1;
int32_t x9310 = x9304;
bool x9311 = x9310 >= 2;
if (x9311) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9316 = x9310 == 0;
if (x9316) {
int32_t x9317 = x9305;
bool x9318 = x9317 == 128;
if (x9318) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9325 = x9305;
int32_t x9326 = 128 / x9325;
bool x9327 = x9326 == 1;
bool x9330;
if (x454) {
bool x9328 = 128 == x9326;
bool x9329 = x9327 || x9328;
x9330 = x9329;
} else {
x9330 = false;
}
bool x9334;
if (x9330) {
x9334 = x9333;
} else {
x9334 = false;
}
bool x9335;
if (x9334) {
x9335 = x9333;
} else {
x9335 = false;
}
if (x9335) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x9204,x9204,1,x9326,1,1);
assert(false && "");
}
bool x9341 = 128 <= x9326;
int32_t x9342;
if (x9341) {
x9342 = x9326;
} else {
x9342 = 128;
}
int32_t x9351 = x9342 * x9350;
int32_t x9352 = 64 * x9351;
float* x9353 = (float*)myMalloc(x9352 * sizeof(float));;
int32_t x9356;
if (x9327) {
x9356 = 0;
} else {
x9356 = 1;
}
for(int x9357=0; x9357 < 64; x9357++) {
int32_t x9369 = x9206 * x9357;
int32_t x9363 = x9351 * x9357;
for(int x9359=0; x9359 < x9342; x9359++) {
int32_t x9370 = x9205 * x9359;
int32_t x9371 = x9369 + x9370;
int32_t x9376 = x9356 * x9359;
int32_t x9365 = x9350 * x9359;
for(int x9361=0; x9361 < x9344; x9361++) {
int32_t x9372 = x9354 * x9361;
int32_t x9373 = x9371 + x9372;
int32_t x9367 = x9344 * x9361;
for(int x9362=0; x9362 < x9344; x9362++) {
int32_t x9374 = x9355 * x9362;
int32_t x9375 = x9373 + x9374;
float x9377 = x9210[x9375];
float x9378 = x217[x9376];
int32_t x9364 = x9362 + x9363;
int32_t x9366 = x9364 + x9365;
int32_t x9368 = x9366 + x9367;
float x9379 = x9377 - x9378;
x9353[x9368] = x9379;

}

}

}

}
float* x9389 = (float*)myMalloc(128 * sizeof(float));;
for(int x9390=0; x9390 < 128; x9390++) {
float x9391 = x266[x9390];
float x9392 = x9391 + 1.0E-5f;
x9389[x9390] = x9392;

}
float* x9396 = (float*)myMalloc(128 * sizeof(float));;
for(int x9397=0; x9397 < 128; x9397++) {
float x9398 = x9389[x9397];
double x9399 = (double)x9398;
double x9400 = sqrt(x9399);
float x9401 = (float)x9400;
x9396[x9397] = x9401;

}
int32_t x9405 = 0;
int32_t x9406 = 1;
x9406 *= 1;
x9405 += 1;
x9406 *= 1;
x9406 *= 1;
int32_t x9411 = x9405;
bool x9412 = x9411 >= 2;
if (x9412) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9417 = x9411 == 0;
if (x9417) {
int32_t x9418 = x9406;
bool x9419 = x9418 == 128;
if (x9419) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9426 = x9406;
bool x9428 = x9342 == 1;
int32_t x9427 = 128 / x9426;
bool x9429 = x9427 == 1;
bool x9433;
if (x454) {
bool x9430 = x9428 || x9429;
bool x9431 = x9342 == x9427;
bool x9432 = x9430 || x9431;
x9433 = x9432;
} else {
x9433 = false;
}
bool x9437;
if (x9433) {
x9437 = x9436;
} else {
x9437 = false;
}
bool x9438;
if (x9437) {
x9438 = x9436;
} else {
x9438 = false;
}
if (x9438) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9342,x9344,x9344,1,x9427,1,1);
assert(false && "");
}
bool x9444 = x9342 <= x9427;
int32_t x9445;
if (x9444) {
x9445 = x9427;
} else {
x9445 = x9342;
}
int32_t x9454 = x9445 * x9453;
int32_t x9455 = 64 * x9454;
float* x9456 = (float*)myMalloc(x9455 * sizeof(float));;
int32_t x9457;
if (x9428) {
x9457 = 0;
} else {
x9457 = x9350;
}
int32_t x9460;
if (x9429) {
x9460 = 0;
} else {
x9460 = 1;
}
for(int x9461=0; x9461 < 64; x9461++) {
int32_t x9473 = x9351 * x9461;
int32_t x9467 = x9454 * x9461;
for(int x9463=0; x9463 < x9445; x9463++) {
int32_t x9474 = x9457 * x9463;
int32_t x9475 = x9473 + x9474;
int32_t x9480 = x9460 * x9463;
int32_t x9469 = x9453 * x9463;
for(int x9465=0; x9465 < x9447; x9465++) {
int32_t x9476 = x9458 * x9465;
int32_t x9477 = x9475 + x9476;
int32_t x9471 = x9447 * x9465;
for(int x9466=0; x9466 < x9447; x9466++) {
int32_t x9478 = x9459 * x9466;
int32_t x9479 = x9477 + x9478;
float x9481 = x9353[x9479];
float x9482 = x9396[x9480];
int32_t x9468 = x9466 + x9467;
int32_t x9470 = x9468 + x9469;
int32_t x9472 = x9470 + x9471;
float x9483 = x9481 / x9482;
x9456[x9472] = x9483;

}

}

}

}
int32_t x9493 = 0;
int32_t x9494 = 1;
x9494 *= 1;
x9493 += 1;
x9494 *= 1;
x9494 *= 1;
int32_t x9499 = x9493;
bool x9500 = x9499 >= 2;
if (x9500) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9505 = x9499 == 0;
if (x9505) {
int32_t x9506 = x9494;
bool x9507 = x9506 == 128;
if (x9507) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9514 = x9494;
bool x9516 = x9445 == 1;
int32_t x9515 = 128 / x9514;
bool x9517 = x9515 == 1;
bool x9521;
if (x454) {
bool x9518 = x9516 || x9517;
bool x9519 = x9445 == x9515;
bool x9520 = x9518 || x9519;
x9521 = x9520;
} else {
x9521 = false;
}
bool x9525;
if (x9521) {
x9525 = x9524;
} else {
x9525 = false;
}
bool x9526;
if (x9525) {
x9526 = x9524;
} else {
x9526 = false;
}
if (x9526) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9445,x9447,x9447,1,x9515,1,1);
assert(false && "");
}
bool x9532 = x9445 <= x9515;
int32_t x9533;
if (x9532) {
x9533 = x9515;
} else {
x9533 = x9445;
}
int32_t x9542 = x9533 * x9541;
int32_t x9543 = 64 * x9542;
float* x9544 = (float*)myMalloc(x9543 * sizeof(float));;
int32_t x9545;
if (x9516) {
x9545 = 0;
} else {
x9545 = x9453;
}
int32_t x9548;
if (x9517) {
x9548 = 0;
} else {
x9548 = 1;
}
for(int x9549=0; x9549 < 64; x9549++) {
int32_t x9561 = x9454 * x9549;
int32_t x9555 = x9542 * x9549;
for(int x9551=0; x9551 < x9533; x9551++) {
int32_t x9562 = x9545 * x9551;
int32_t x9563 = x9561 + x9562;
int32_t x9568 = x9548 * x9551;
int32_t x9557 = x9541 * x9551;
for(int x9553=0; x9553 < x9535; x9553++) {
int32_t x9564 = x9546 * x9553;
int32_t x9565 = x9563 + x9564;
int32_t x9559 = x9535 * x9553;
for(int x9554=0; x9554 < x9535; x9554++) {
int32_t x9566 = x9547 * x9554;
int32_t x9567 = x9565 + x9566;
float x9569 = x9456[x9567];
float x9570 = x127[x9568];
int32_t x9556 = x9554 + x9555;
int32_t x9558 = x9556 + x9557;
int32_t x9560 = x9558 + x9559;
float x9571 = x9569 * x9570;
x9544[x9560] = x9571;

}

}

}

}
int32_t x9581 = 0;
int32_t x9582 = 1;
x9582 *= 1;
x9581 += 1;
x9582 *= 1;
x9582 *= 1;
int32_t x9587 = x9581;
bool x9588 = x9587 >= 2;
if (x9588) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9593 = x9587 == 0;
if (x9593) {
int32_t x9594 = x9582;
bool x9595 = x9594 == 128;
if (x9595) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9602 = x9582;
bool x9604 = x9533 == 1;
int32_t x9603 = 128 / x9602;
bool x9605 = x9603 == 1;
bool x9609;
if (x454) {
bool x9606 = x9604 || x9605;
bool x9607 = x9533 == x9603;
bool x9608 = x9606 || x9607;
x9609 = x9608;
} else {
x9609 = false;
}
bool x9613;
if (x9609) {
x9613 = x9612;
} else {
x9613 = false;
}
bool x9614;
if (x9613) {
x9614 = x9612;
} else {
x9614 = false;
}
if (x9614) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9533,x9535,x9535,1,x9603,1,1);
assert(false && "");
}
bool x9620 = x9533 <= x9603;
int32_t x9621;
if (x9620) {
x9621 = x9603;
} else {
x9621 = x9533;
}
int32_t x9630 = x9621 * x9629;
int32_t x9631 = 64 * x9630;
float* x9632 = (float*)myMalloc(x9631 * sizeof(float));;
int32_t x9633;
if (x9604) {
x9633 = 0;
} else {
x9633 = x9541;
}
int32_t x9636;
if (x9605) {
x9636 = 0;
} else {
x9636 = 1;
}
for(int x9637=0; x9637 < 64; x9637++) {
int32_t x9649 = x9542 * x9637;
int32_t x9643 = x9630 * x9637;
for(int x9639=0; x9639 < x9621; x9639++) {
int32_t x9650 = x9633 * x9639;
int32_t x9651 = x9649 + x9650;
int32_t x9656 = x9636 * x9639;
int32_t x9645 = x9629 * x9639;
for(int x9641=0; x9641 < x9623; x9641++) {
int32_t x9652 = x9634 * x9641;
int32_t x9653 = x9651 + x9652;
int32_t x9647 = x9623 * x9641;
for(int x9642=0; x9642 < x9623; x9642++) {
int32_t x9654 = x9635 * x9642;
int32_t x9655 = x9653 + x9654;
float x9657 = x9544[x9655];
float x9658 = x61[x9656];
int32_t x9644 = x9642 + x9643;
int32_t x9646 = x9644 + x9645;
int32_t x9648 = x9646 + x9647;
float x9659 = x9657 + x9658;
x9632[x9648] = x9659;

}

}

}

}
float* x9669 = (float*)myMalloc(x9631 * sizeof(float));;
for(int x9671=0; x9671 < x9631; x9671++) {
float x9672 = x9632[x9671];
bool x9673 = x9672 < 0.0f;
if (x9673) {
x9669[x9671] = 0.0f;
} else {
float x9676 = x9632[x9671];
x9669[x9671] = x9676;
}

}
float* x9690 = (float*)myMalloc(x9689 * sizeof(float));;
int32_t x9693 = 64 * x9621;
int32_t x9694 = x9693 * x9685;
float* x9695 = (float*)myMalloc(x9694 * sizeof(float));;
int32_t x9691 = x9621 * x9685;
for(int x9696=0; x9696 < 64; x9696++) {
int32_t x9697 = x9696 * x9630;
float* x9698 = x9669+x9697;
int32_t x9699 = x9696 * x9686;
float* x9700 = x9690+x9699;
int32_t x9701 = x9696 * x9691;
float* x9702 = x9695+x9701;
for(int x9703=0; x9703 < x9621; x9703++) {
int32_t x9704 = x9703 / 1;
int32_t x9708 = x9704 * x9684;
int32_t x9709 = x9708 * x9684;
int32_t x9705 = x9703 % 1;
int32_t x9706 = x9705 / 1;
int32_t x9710 = x9706 * x9684;
int32_t x9711 = x9710 * x9684;
int32_t x9712 = x9709 + x9711;
int32_t x9707 = x9705 % 1;
int32_t x9713 = x9707 * x9684;
int32_t x9714 = x9713 * x9684;
int32_t x9715 = x9712 + x9714;
float* x9716 = x9702+x9715;
int32_t x9717 = x9704 * x9623;
int32_t x9718 = x9717 * x9623;
float* x9719 = x9698+x9718;
for(int x9721=0; x9721 < x9684; x9721++) {
int32_t x9723 = x9721 * x9684;
float* x9724 = x9716+x9723;
int32_t x9722 = x9721 + x9706;
int32_t x9725 = x9722 * x9623;
int32_t x9726 = x9725 + x9707;
float* x9727 = x9719+x9726;
memcpy(x9724, x9727, 4 * x9684);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x9685,x9621,1,x41,x9621,x9702,x9685,1,x9700,x9685);

}
int32_t x9736 = 0;
int32_t x9737 = 1;
x9737 *= 1;
x9736 += 1;
x9737 *= 1;
x9737 *= 1;
int32_t x9742 = x9736;
bool x9743 = x9742 >= 2;
if (x9743) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9748 = x9742 == 0;
if (x9748) {
int32_t x9749 = x9737;
bool x9750 = x9749 == 512;
if (x9750) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9757 = x9737;
int32_t x9758 = 512 / x9757;
bool x9759 = x9758 == 1;
bool x9762;
if (x454) {
bool x9760 = 512 == x9758;
bool x9761 = x9759 || x9760;
x9762 = x9761;
} else {
x9762 = false;
}
bool x9766;
if (x9762) {
x9766 = x9765;
} else {
x9766 = false;
}
bool x9767;
if (x9766) {
x9767 = x9765;
} else {
x9767 = false;
}
if (x9767) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x9684,x9684,1,x9758,1,1);
assert(false && "");
}
bool x9773 = 512 <= x9758;
int32_t x9774;
if (x9773) {
x9774 = x9758;
} else {
x9774 = 512;
}
int32_t x9783 = x9774 * x9782;
int32_t x9784 = 64 * x9783;
float* x9785 = (float*)myMalloc(x9784 * sizeof(float));;
int32_t x9788;
if (x9759) {
x9788 = 0;
} else {
x9788 = 1;
}
for(int x9789=0; x9789 < 64; x9789++) {
int32_t x9801 = x9686 * x9789;
int32_t x9795 = x9783 * x9789;
for(int x9791=0; x9791 < x9774; x9791++) {
int32_t x9802 = x9685 * x9791;
int32_t x9803 = x9801 + x9802;
int32_t x9808 = x9788 * x9791;
int32_t x9797 = x9782 * x9791;
for(int x9793=0; x9793 < x9776; x9793++) {
int32_t x9804 = x9786 * x9793;
int32_t x9805 = x9803 + x9804;
int32_t x9799 = x9776 * x9793;
for(int x9794=0; x9794 < x9776; x9794++) {
int32_t x9806 = x9787 * x9794;
int32_t x9807 = x9805 + x9806;
float x9809 = x9690[x9807];
float x9810 = x25[x9808];
int32_t x9796 = x9794 + x9795;
int32_t x9798 = x9796 + x9797;
int32_t x9800 = x9798 + x9799;
float x9811 = x9809 - x9810;
x9785[x9800] = x9811;

}

}

}

}
float* x9821 = (float*)myMalloc(512 * sizeof(float));;
for(int x9822=0; x9822 < 512; x9822++) {
float x9823 = x223[x9822];
float x9824 = x9823 + 1.0E-5f;
x9821[x9822] = x9824;

}
float* x9828 = (float*)myMalloc(512 * sizeof(float));;
for(int x9829=0; x9829 < 512; x9829++) {
float x9830 = x9821[x9829];
double x9831 = (double)x9830;
double x9832 = sqrt(x9831);
float x9833 = (float)x9832;
x9828[x9829] = x9833;

}
int32_t x9837 = 0;
int32_t x9838 = 1;
x9838 *= 1;
x9837 += 1;
x9838 *= 1;
x9838 *= 1;
int32_t x9843 = x9837;
bool x9844 = x9843 >= 2;
if (x9844) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9849 = x9843 == 0;
if (x9849) {
int32_t x9850 = x9838;
bool x9851 = x9850 == 512;
if (x9851) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9858 = x9838;
bool x9860 = x9774 == 1;
int32_t x9859 = 512 / x9858;
bool x9861 = x9859 == 1;
bool x9865;
if (x454) {
bool x9862 = x9860 || x9861;
bool x9863 = x9774 == x9859;
bool x9864 = x9862 || x9863;
x9865 = x9864;
} else {
x9865 = false;
}
bool x9869;
if (x9865) {
x9869 = x9868;
} else {
x9869 = false;
}
bool x9870;
if (x9869) {
x9870 = x9868;
} else {
x9870 = false;
}
if (x9870) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9774,x9776,x9776,1,x9859,1,1);
assert(false && "");
}
bool x9876 = x9774 <= x9859;
int32_t x9877;
if (x9876) {
x9877 = x9859;
} else {
x9877 = x9774;
}
int32_t x9886 = x9877 * x9885;
int32_t x9887 = 64 * x9886;
float* x9888 = (float*)myMalloc(x9887 * sizeof(float));;
int32_t x9889;
if (x9860) {
x9889 = 0;
} else {
x9889 = x9782;
}
int32_t x9892;
if (x9861) {
x9892 = 0;
} else {
x9892 = 1;
}
for(int x9893=0; x9893 < 64; x9893++) {
int32_t x9905 = x9783 * x9893;
int32_t x9899 = x9886 * x9893;
for(int x9895=0; x9895 < x9877; x9895++) {
int32_t x9906 = x9889 * x9895;
int32_t x9907 = x9905 + x9906;
int32_t x9912 = x9892 * x9895;
int32_t x9901 = x9885 * x9895;
for(int x9897=0; x9897 < x9879; x9897++) {
int32_t x9908 = x9890 * x9897;
int32_t x9909 = x9907 + x9908;
int32_t x9903 = x9879 * x9897;
for(int x9898=0; x9898 < x9879; x9898++) {
int32_t x9910 = x9891 * x9898;
int32_t x9911 = x9909 + x9910;
float x9913 = x9785[x9911];
float x9914 = x9828[x9912];
int32_t x9900 = x9898 + x9899;
int32_t x9902 = x9900 + x9901;
int32_t x9904 = x9902 + x9903;
float x9915 = x9913 / x9914;
x9888[x9904] = x9915;

}

}

}

}
int32_t x9925 = 0;
int32_t x9926 = 1;
x9926 *= 1;
x9925 += 1;
x9926 *= 1;
x9926 *= 1;
int32_t x9931 = x9925;
bool x9932 = x9931 >= 2;
if (x9932) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9937 = x9931 == 0;
if (x9937) {
int32_t x9938 = x9926;
bool x9939 = x9938 == 512;
if (x9939) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9946 = x9926;
bool x9948 = x9877 == 1;
int32_t x9947 = 512 / x9946;
bool x9949 = x9947 == 1;
bool x9953;
if (x454) {
bool x9950 = x9948 || x9949;
bool x9951 = x9877 == x9947;
bool x9952 = x9950 || x9951;
x9953 = x9952;
} else {
x9953 = false;
}
bool x9957;
if (x9953) {
x9957 = x9956;
} else {
x9957 = false;
}
bool x9958;
if (x9957) {
x9958 = x9956;
} else {
x9958 = false;
}
if (x9958) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9877,x9879,x9879,1,x9947,1,1);
assert(false && "");
}
bool x9964 = x9877 <= x9947;
int32_t x9965;
if (x9964) {
x9965 = x9947;
} else {
x9965 = x9877;
}
int32_t x9974 = x9965 * x9973;
int32_t x9975 = 64 * x9974;
float* x9976 = (float*)myMalloc(x9975 * sizeof(float));;
int32_t x9977;
if (x9948) {
x9977 = 0;
} else {
x9977 = x9885;
}
int32_t x9980;
if (x9949) {
x9980 = 0;
} else {
x9980 = 1;
}
for(int x9981=0; x9981 < 64; x9981++) {
int32_t x9993 = x9886 * x9981;
int32_t x9987 = x9974 * x9981;
for(int x9983=0; x9983 < x9965; x9983++) {
int32_t x9994 = x9977 * x9983;
int32_t x9995 = x9993 + x9994;
int32_t x10000 = x9980 * x9983;
int32_t x9989 = x9973 * x9983;
for(int x9985=0; x9985 < x9967; x9985++) {
int32_t x9996 = x9978 * x9985;
int32_t x9997 = x9995 + x9996;
int32_t x9991 = x9967 * x9985;
for(int x9986=0; x9986 < x9967; x9986++) {
int32_t x9998 = x9979 * x9986;
int32_t x9999 = x9997 + x9998;
float x10001 = x9888[x9999];
float x10002 = x167[x10000];
int32_t x9988 = x9986 + x9987;
int32_t x9990 = x9988 + x9989;
int32_t x9992 = x9990 + x9991;
float x10003 = x10001 * x10002;
x9976[x9992] = x10003;

}

}

}

}
int32_t x10013 = 0;
int32_t x10014 = 1;
x10014 *= 1;
x10013 += 1;
x10014 *= 1;
x10014 *= 1;
int32_t x10019 = x10013;
bool x10020 = x10019 >= 2;
if (x10020) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10025 = x10019 == 0;
if (x10025) {
int32_t x10026 = x10014;
bool x10027 = x10026 == 512;
if (x10027) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10034 = x10014;
bool x10036 = x9965 == 1;
int32_t x10035 = 512 / x10034;
bool x10037 = x10035 == 1;
bool x10041;
if (x454) {
bool x10038 = x10036 || x10037;
bool x10039 = x9965 == x10035;
bool x10040 = x10038 || x10039;
x10041 = x10040;
} else {
x10041 = false;
}
bool x10045;
if (x10041) {
x10045 = x10044;
} else {
x10045 = false;
}
bool x10046;
if (x10045) {
x10046 = x10044;
} else {
x10046 = false;
}
if (x10046) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9965,x9967,x9967,1,x10035,1,1);
assert(false && "");
}
bool x10052 = x9965 <= x10035;
int32_t x10053;
if (x10052) {
x10053 = x10035;
} else {
x10053 = x9965;
}
int32_t x10062 = x10053 * x10061;
int32_t x10063 = 64 * x10062;
float* x10064 = (float*)myMalloc(x10063 * sizeof(float));;
int32_t x10065;
if (x10036) {
x10065 = 0;
} else {
x10065 = x9973;
}
int32_t x10068;
if (x10037) {
x10068 = 0;
} else {
x10068 = 1;
}
for(int x10069=0; x10069 < 64; x10069++) {
int32_t x10081 = x9974 * x10069;
int32_t x10075 = x10062 * x10069;
for(int x10071=0; x10071 < x10053; x10071++) {
int32_t x10082 = x10065 * x10071;
int32_t x10083 = x10081 + x10082;
int32_t x10088 = x10068 * x10071;
int32_t x10077 = x10061 * x10071;
for(int x10073=0; x10073 < x10055; x10073++) {
int32_t x10084 = x10066 * x10073;
int32_t x10085 = x10083 + x10084;
int32_t x10079 = x10055 * x10073;
for(int x10074=0; x10074 < x10055; x10074++) {
int32_t x10086 = x10067 * x10074;
int32_t x10087 = x10085 + x10086;
float x10089 = x9976[x10087];
float x10090 = x82[x10088];
int32_t x10076 = x10074 + x10075;
int32_t x10078 = x10076 + x10077;
int32_t x10080 = x10078 + x10079;
float x10091 = x10089 + x10090;
x10064[x10080] = x10091;

}

}

}

}
bool x10101 = x10053 == 1;
bool x10102 = x10101 || x8689;
bool x10103 = x10053 == x8641;
bool x10104 = x10102 || x10103;
bool x10109;
if (x10104) {
x10109 = x10108;
} else {
x10109 = false;
}
bool x10110;
if (x10109) {
x10110 = x10108;
} else {
x10110 = false;
}
if (x10110) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10053,x10055,x10055,64,x8641,x8643,x8643);
assert(false && "");
}
bool x10116 = x10053 <= x8641;
int32_t x10117;
if (x10116) {
x10117 = x8641;
} else {
x10117 = x10053;
}
int32_t x10133;
if (x10101) {
x10133 = 0;
} else {
x10133 = x10061;
}
for(int x10136=0; x10136 < 64; x10136++) {
int32_t x10142 = x10062 * x10136;
int32_t x10149 = x8650 * x10136;
for(int x10138=0; x10138 < x10117; x10138++) {
int32_t x10143 = x10133 * x10138;
int32_t x10144 = x10142 + x10143;
int32_t x10150 = x8721 * x10138;
int32_t x10151 = x10149 + x10150;
for(int x10140=0; x10140 < x10119; x10140++) {
int32_t x10145 = x10134 * x10140;
int32_t x10146 = x10144 + x10145;
int32_t x10152 = x8722 * x10140;
int32_t x10153 = x10151 + x10152;
for(int x10141=0; x10141 < x10119; x10141++) {
int32_t x10147 = x10135 * x10141;
int32_t x10148 = x10146 + x10147;
float x10156 = x10064[x10148];
int32_t x10154 = x8723 * x10141;
int32_t x10155 = x10153 + x10154;
float x10157 = x8756[x10155];
float x10158 = x10156 + x10157;
x10064[x10148] = x10158;

}

}

}

}
float* x10168 = (float*)myMalloc(x10063 * sizeof(float));;
for(int x10170=0; x10170 < x10063; x10170++) {
float x10171 = x10064[x10170];
bool x10172 = x10171 < 0.0f;
if (x10172) {
x10168[x10170] = 0.0f;
} else {
float x10175 = x10064[x10170];
x10168[x10170] = x10175;
}

}
float* x10189 = (float*)myMalloc(x10188 * sizeof(float));;
int32_t x10192 = 64 * x10053;
int32_t x10193 = x10192 * x10184;
float* x10194 = (float*)myMalloc(x10193 * sizeof(float));;
int32_t x10190 = x10053 * x10184;
for(int x10195=0; x10195 < 64; x10195++) {
int32_t x10196 = x10195 * x10062;
float* x10197 = x10168+x10196;
int32_t x10198 = x10195 * x10185;
float* x10199 = x10189+x10198;
int32_t x10200 = x10195 * x10190;
float* x10201 = x10194+x10200;
for(int x10202=0; x10202 < x10053; x10202++) {
int32_t x10203 = x10202 / 1;
int32_t x10207 = x10203 * x10183;
int32_t x10208 = x10207 * x10183;
int32_t x10204 = x10202 % 1;
int32_t x10205 = x10204 / 1;
int32_t x10209 = x10205 * x10183;
int32_t x10210 = x10209 * x10183;
int32_t x10211 = x10208 + x10210;
int32_t x10206 = x10204 % 1;
int32_t x10212 = x10206 * x10183;
int32_t x10213 = x10212 * x10183;
int32_t x10214 = x10211 + x10213;
float* x10215 = x10201+x10214;
int32_t x10216 = x10203 * x10055;
int32_t x10217 = x10216 * x10055;
float* x10218 = x10197+x10217;
for(int x10220=0; x10220 < x10183; x10220++) {
int32_t x10222 = x10220 * x10183;
float* x10223 = x10215+x10222;
int32_t x10221 = x10220 + x10205;
int32_t x10224 = x10221 * x10055;
int32_t x10225 = x10224 + x10206;
float* x10226 = x10218+x10225;
memcpy(x10223, x10226, 4 * x10183);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x10184,x10053,1,x132,x10053,x10201,x10184,1,x10199,x10184);

}
int32_t x10235 = 0;
int32_t x10236 = 1;
x10236 *= 1;
x10235 += 1;
x10236 *= 1;
x10236 *= 1;
int32_t x10241 = x10235;
bool x10242 = x10241 >= 2;
if (x10242) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10247 = x10241 == 0;
if (x10247) {
int32_t x10248 = x10236;
bool x10249 = x10248 == 128;
if (x10249) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10256 = x10236;
int32_t x10257 = 128 / x10256;
bool x10258 = x10257 == 1;
bool x10261;
if (x454) {
bool x10259 = 128 == x10257;
bool x10260 = x10258 || x10259;
x10261 = x10260;
} else {
x10261 = false;
}
bool x10265;
if (x10261) {
x10265 = x10264;
} else {
x10265 = false;
}
bool x10266;
if (x10265) {
x10266 = x10264;
} else {
x10266 = false;
}
if (x10266) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x10183,x10183,1,x10257,1,1);
assert(false && "");
}
bool x10272 = 128 <= x10257;
int32_t x10273;
if (x10272) {
x10273 = x10257;
} else {
x10273 = 128;
}
int32_t x10282 = x10273 * x10281;
int32_t x10283 = 64 * x10282;
float* x10284 = (float*)myMalloc(x10283 * sizeof(float));;
int32_t x10287;
if (x10258) {
x10287 = 0;
} else {
x10287 = 1;
}
for(int x10288=0; x10288 < 64; x10288++) {
int32_t x10300 = x10185 * x10288;
int32_t x10294 = x10282 * x10288;
for(int x10290=0; x10290 < x10273; x10290++) {
int32_t x10301 = x10184 * x10290;
int32_t x10302 = x10300 + x10301;
int32_t x10307 = x10287 * x10290;
int32_t x10296 = x10281 * x10290;
for(int x10292=0; x10292 < x10275; x10292++) {
int32_t x10303 = x10285 * x10292;
int32_t x10304 = x10302 + x10303;
int32_t x10298 = x10275 * x10292;
for(int x10293=0; x10293 < x10275; x10293++) {
int32_t x10305 = x10286 * x10293;
int32_t x10306 = x10304 + x10305;
float x10308 = x10189[x10306];
float x10309 = x236[x10307];
int32_t x10295 = x10293 + x10294;
int32_t x10297 = x10295 + x10296;
int32_t x10299 = x10297 + x10298;
float x10310 = x10308 - x10309;
x10284[x10299] = x10310;

}

}

}

}
float* x10320 = (float*)myMalloc(128 * sizeof(float));;
for(int x10321=0; x10321 < 128; x10321++) {
float x10322 = x261[x10321];
float x10323 = x10322 + 1.0E-5f;
x10320[x10321] = x10323;

}
float* x10327 = (float*)myMalloc(128 * sizeof(float));;
for(int x10328=0; x10328 < 128; x10328++) {
float x10329 = x10320[x10328];
double x10330 = (double)x10329;
double x10331 = sqrt(x10330);
float x10332 = (float)x10331;
x10327[x10328] = x10332;

}
int32_t x10336 = 0;
int32_t x10337 = 1;
x10337 *= 1;
x10336 += 1;
x10337 *= 1;
x10337 *= 1;
int32_t x10342 = x10336;
bool x10343 = x10342 >= 2;
if (x10343) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10348 = x10342 == 0;
if (x10348) {
int32_t x10349 = x10337;
bool x10350 = x10349 == 128;
if (x10350) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10357 = x10337;
bool x10359 = x10273 == 1;
int32_t x10358 = 128 / x10357;
bool x10360 = x10358 == 1;
bool x10364;
if (x454) {
bool x10361 = x10359 || x10360;
bool x10362 = x10273 == x10358;
bool x10363 = x10361 || x10362;
x10364 = x10363;
} else {
x10364 = false;
}
bool x10368;
if (x10364) {
x10368 = x10367;
} else {
x10368 = false;
}
bool x10369;
if (x10368) {
x10369 = x10367;
} else {
x10369 = false;
}
if (x10369) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10273,x10275,x10275,1,x10358,1,1);
assert(false && "");
}
bool x10375 = x10273 <= x10358;
int32_t x10376;
if (x10375) {
x10376 = x10358;
} else {
x10376 = x10273;
}
int32_t x10385 = x10376 * x10384;
int32_t x10386 = 64 * x10385;
float* x10387 = (float*)myMalloc(x10386 * sizeof(float));;
int32_t x10388;
if (x10359) {
x10388 = 0;
} else {
x10388 = x10281;
}
int32_t x10391;
if (x10360) {
x10391 = 0;
} else {
x10391 = 1;
}
for(int x10392=0; x10392 < 64; x10392++) {
int32_t x10404 = x10282 * x10392;
int32_t x10398 = x10385 * x10392;
for(int x10394=0; x10394 < x10376; x10394++) {
int32_t x10405 = x10388 * x10394;
int32_t x10406 = x10404 + x10405;
int32_t x10411 = x10391 * x10394;
int32_t x10400 = x10384 * x10394;
for(int x10396=0; x10396 < x10378; x10396++) {
int32_t x10407 = x10389 * x10396;
int32_t x10408 = x10406 + x10407;
int32_t x10402 = x10378 * x10396;
for(int x10397=0; x10397 < x10378; x10397++) {
int32_t x10409 = x10390 * x10397;
int32_t x10410 = x10408 + x10409;
float x10412 = x10284[x10410];
float x10413 = x10327[x10411];
int32_t x10399 = x10397 + x10398;
int32_t x10401 = x10399 + x10400;
int32_t x10403 = x10401 + x10402;
float x10414 = x10412 / x10413;
x10387[x10403] = x10414;

}

}

}

}
int32_t x10424 = 0;
int32_t x10425 = 1;
x10425 *= 1;
x10424 += 1;
x10425 *= 1;
x10425 *= 1;
int32_t x10430 = x10424;
bool x10431 = x10430 >= 2;
if (x10431) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10436 = x10430 == 0;
if (x10436) {
int32_t x10437 = x10425;
bool x10438 = x10437 == 128;
if (x10438) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10445 = x10425;
bool x10447 = x10376 == 1;
int32_t x10446 = 128 / x10445;
bool x10448 = x10446 == 1;
bool x10452;
if (x454) {
bool x10449 = x10447 || x10448;
bool x10450 = x10376 == x10446;
bool x10451 = x10449 || x10450;
x10452 = x10451;
} else {
x10452 = false;
}
bool x10456;
if (x10452) {
x10456 = x10455;
} else {
x10456 = false;
}
bool x10457;
if (x10456) {
x10457 = x10455;
} else {
x10457 = false;
}
if (x10457) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10376,x10378,x10378,1,x10446,1,1);
assert(false && "");
}
bool x10463 = x10376 <= x10446;
int32_t x10464;
if (x10463) {
x10464 = x10446;
} else {
x10464 = x10376;
}
int32_t x10473 = x10464 * x10472;
int32_t x10474 = 64 * x10473;
float* x10475 = (float*)myMalloc(x10474 * sizeof(float));;
int32_t x10476;
if (x10447) {
x10476 = 0;
} else {
x10476 = x10384;
}
int32_t x10479;
if (x10448) {
x10479 = 0;
} else {
x10479 = 1;
}
for(int x10480=0; x10480 < 64; x10480++) {
int32_t x10492 = x10385 * x10480;
int32_t x10486 = x10473 * x10480;
for(int x10482=0; x10482 < x10464; x10482++) {
int32_t x10493 = x10476 * x10482;
int32_t x10494 = x10492 + x10493;
int32_t x10499 = x10479 * x10482;
int32_t x10488 = x10472 * x10482;
for(int x10484=0; x10484 < x10466; x10484++) {
int32_t x10495 = x10477 * x10484;
int32_t x10496 = x10494 + x10495;
int32_t x10490 = x10466 * x10484;
for(int x10485=0; x10485 < x10466; x10485++) {
int32_t x10497 = x10478 * x10485;
int32_t x10498 = x10496 + x10497;
float x10500 = x10387[x10498];
float x10501 = x39[x10499];
int32_t x10487 = x10485 + x10486;
int32_t x10489 = x10487 + x10488;
int32_t x10491 = x10489 + x10490;
float x10502 = x10500 * x10501;
x10475[x10491] = x10502;

}

}

}

}
int32_t x10512 = 0;
int32_t x10513 = 1;
x10513 *= 1;
x10512 += 1;
x10513 *= 1;
x10513 *= 1;
int32_t x10518 = x10512;
bool x10519 = x10518 >= 2;
if (x10519) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10524 = x10518 == 0;
if (x10524) {
int32_t x10525 = x10513;
bool x10526 = x10525 == 128;
if (x10526) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10533 = x10513;
bool x10535 = x10464 == 1;
int32_t x10534 = 128 / x10533;
bool x10536 = x10534 == 1;
bool x10540;
if (x454) {
bool x10537 = x10535 || x10536;
bool x10538 = x10464 == x10534;
bool x10539 = x10537 || x10538;
x10540 = x10539;
} else {
x10540 = false;
}
bool x10544;
if (x10540) {
x10544 = x10543;
} else {
x10544 = false;
}
bool x10545;
if (x10544) {
x10545 = x10543;
} else {
x10545 = false;
}
if (x10545) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10464,x10466,x10466,1,x10534,1,1);
assert(false && "");
}
bool x10551 = x10464 <= x10534;
int32_t x10552;
if (x10551) {
x10552 = x10534;
} else {
x10552 = x10464;
}
int32_t x10561 = x10552 * x10560;
int32_t x10562 = 64 * x10561;
float* x10563 = (float*)myMalloc(x10562 * sizeof(float));;
int32_t x10564;
if (x10535) {
x10564 = 0;
} else {
x10564 = x10472;
}
int32_t x10567;
if (x10536) {
x10567 = 0;
} else {
x10567 = 1;
}
for(int x10568=0; x10568 < 64; x10568++) {
int32_t x10580 = x10473 * x10568;
int32_t x10574 = x10561 * x10568;
for(int x10570=0; x10570 < x10552; x10570++) {
int32_t x10581 = x10564 * x10570;
int32_t x10582 = x10580 + x10581;
int32_t x10587 = x10567 * x10570;
int32_t x10576 = x10560 * x10570;
for(int x10572=0; x10572 < x10554; x10572++) {
int32_t x10583 = x10565 * x10572;
int32_t x10584 = x10582 + x10583;
int32_t x10578 = x10554 * x10572;
for(int x10573=0; x10573 < x10554; x10573++) {
int32_t x10585 = x10566 * x10573;
int32_t x10586 = x10584 + x10585;
float x10588 = x10475[x10586];
float x10589 = x242[x10587];
int32_t x10575 = x10573 + x10574;
int32_t x10577 = x10575 + x10576;
int32_t x10579 = x10577 + x10578;
float x10590 = x10588 + x10589;
x10563[x10579] = x10590;

}

}

}

}
float* x10600 = (float*)myMalloc(x10562 * sizeof(float));;
for(int x10602=0; x10602 < x10562; x10602++) {
float x10603 = x10563[x10602];
bool x10604 = x10603 < 0.0f;
if (x10604) {
x10600[x10602] = 0.0f;
} else {
float x10607 = x10563[x10602];
x10600[x10602] = x10607;
}

}
float* x10622 = (float*)myMalloc(x10621 * sizeof(float));;
int32_t x10623 = 9 * x10552;
int32_t x10626 = 64 * x10623;
int32_t x10627 = x10626 * x10617;
float* x10628 = (float*)myMalloc(x10627 * sizeof(float));;
int32_t x10624 = x10623 * x10617;
int32_t x10636 = x10552 * 3;
int32_t x10637 = x10636 * 3;
for(int x10629=0; x10629 < 64; x10629++) {
int32_t x10630 = x10629 * x10561;
float* x10631 = x10600+x10630;
int32_t x10632 = x10629 * x10618;
float* x10633 = x10622+x10632;
int32_t x10634 = x10629 * x10624;
float* x10635 = x10628+x10634;
for(int x10639=0; x10639 < x10637; x10639++) {
int32_t x10640 = x10639 / 9;
int32_t x10644 = x10640 * 3;
int32_t x10645 = x10644 * 3;
int32_t x10646 = x10645 * x10616;
int32_t x10647 = x10646 * x10616;
int32_t x10641 = x10639 % 9;
int32_t x10642 = x10641 / 3;
int32_t x10648 = x10642 * 3;
int32_t x10649 = x10648 * x10616;
int32_t x10650 = x10649 * x10616;
int32_t x10651 = x10647 + x10650;
int32_t x10643 = x10641 % 3;
int32_t x10652 = x10643 * x10616;
int32_t x10653 = x10652 * x10616;
int32_t x10654 = x10651 + x10653;
float* x10655 = x10635+x10654;
int32_t x10656 = x10640 * x10554;
int32_t x10657 = x10656 * x10554;
float* x10658 = x10631+x10657;
int32_t x10671 = 1 - x10643;
bool x10672 = x10671 > 0;
int32_t x10673;
if (x10672) {
x10673 = x10671;
} else {
x10673 = 0;
}
int32_t x10674 = 3 - x10643;
int32_t x10675 = x10674 - 1;
int32_t x10676 = 1 - x10675;
bool x10677 = x10676 > 0;
int32_t x10678;
if (x10677) {
x10678 = x10676;
} else {
x10678 = 0;
}
int32_t x10679 = x10616 - x10678;
int32_t x10680 = x10679 - x10673;
bool x10681 = x10680 <= 0;
bool x10685 = x10673 > 0;
int32_t x10670 = -1 + x10643;
bool x10698 = x10678 > 0;
for(int x10660=0; x10660 < x10616; x10660++) {
int32_t x10661 = x10660 - 1;
int32_t x10662 = x10661 + x10642;
bool x10663 = x10662 < 0;
bool x10664 = x10662 >= x10554;
bool x10665 = x10663 || x10664;
if (x10665) {
int32_t x10666 = x10660 * x10616;
float* x10667 = x10655+x10666;
memset(x10667, 0, 4 * x10616);;
} else {
if (x10681) {
int32_t x10666 = x10660 * x10616;
float* x10682 = x10655+x10666;
memset(x10682, 0, 4 * x10616);;
} else {
int32_t x10666 = x10660 * x10616;
if (x10685) {
float* x10686 = x10655+x10666;
memset(x10686, 0, 4 * x10673);;
} else {
}
// may have segfault here
int32_t x10691 = x10666 + x10673;
float* x10692 = x10655+x10691;
int32_t x10693 = x10662 * x10554;
int32_t x10694 = x10693 + x10670;
int32_t x10695 = x10694 + x10673;
float* x10696 = x10658+x10695;
memcpy(x10692, x10696, 4 * x10680);;
if (x10698) {
int32_t x10699 = x10666 + x10616;
int32_t x10700 = x10699 - x10678;
float* x10701 = x10655+x10700;
memset(x10701, 0, 4 * x10678);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x10617,x10623,1,x165,x10623,x10635,x10617,1,x10633,x10617);

}
int32_t x10716 = 0;
int32_t x10717 = 1;
x10717 *= 1;
x10716 += 1;
x10717 *= 1;
x10717 *= 1;
int32_t x10722 = x10716;
bool x10723 = x10722 >= 2;
if (x10723) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10728 = x10722 == 0;
if (x10728) {
int32_t x10729 = x10717;
bool x10730 = x10729 == 128;
if (x10730) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10737 = x10717;
int32_t x10738 = 128 / x10737;
bool x10739 = x10738 == 1;
bool x10742;
if (x454) {
bool x10740 = 128 == x10738;
bool x10741 = x10739 || x10740;
x10742 = x10741;
} else {
x10742 = false;
}
bool x10746;
if (x10742) {
x10746 = x10745;
} else {
x10746 = false;
}
bool x10747;
if (x10746) {
x10747 = x10745;
} else {
x10747 = false;
}
if (x10747) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x10616,x10616,1,x10738,1,1);
assert(false && "");
}
bool x10753 = 128 <= x10738;
int32_t x10754;
if (x10753) {
x10754 = x10738;
} else {
x10754 = 128;
}
int32_t x10763 = x10754 * x10762;
int32_t x10764 = 64 * x10763;
float* x10765 = (float*)myMalloc(x10764 * sizeof(float));;
int32_t x10768;
if (x10739) {
x10768 = 0;
} else {
x10768 = 1;
}
for(int x10769=0; x10769 < 64; x10769++) {
int32_t x10781 = x10618 * x10769;
int32_t x10775 = x10763 * x10769;
for(int x10771=0; x10771 < x10754; x10771++) {
int32_t x10782 = x10617 * x10771;
int32_t x10783 = x10781 + x10782;
int32_t x10788 = x10768 * x10771;
int32_t x10777 = x10762 * x10771;
for(int x10773=0; x10773 < x10756; x10773++) {
int32_t x10784 = x10766 * x10773;
int32_t x10785 = x10783 + x10784;
int32_t x10779 = x10756 * x10773;
for(int x10774=0; x10774 < x10756; x10774++) {
int32_t x10786 = x10767 * x10774;
int32_t x10787 = x10785 + x10786;
float x10789 = x10622[x10787];
float x10790 = x268[x10788];
int32_t x10776 = x10774 + x10775;
int32_t x10778 = x10776 + x10777;
int32_t x10780 = x10778 + x10779;
float x10791 = x10789 - x10790;
x10765[x10780] = x10791;

}

}

}

}
float* x10801 = (float*)myMalloc(128 * sizeof(float));;
for(int x10802=0; x10802 < 128; x10802++) {
float x10803 = x148[x10802];
float x10804 = x10803 + 1.0E-5f;
x10801[x10802] = x10804;

}
float* x10808 = (float*)myMalloc(128 * sizeof(float));;
for(int x10809=0; x10809 < 128; x10809++) {
float x10810 = x10801[x10809];
double x10811 = (double)x10810;
double x10812 = sqrt(x10811);
float x10813 = (float)x10812;
x10808[x10809] = x10813;

}
int32_t x10817 = 0;
int32_t x10818 = 1;
x10818 *= 1;
x10817 += 1;
x10818 *= 1;
x10818 *= 1;
int32_t x10823 = x10817;
bool x10824 = x10823 >= 2;
if (x10824) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10829 = x10823 == 0;
if (x10829) {
int32_t x10830 = x10818;
bool x10831 = x10830 == 128;
if (x10831) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10838 = x10818;
bool x10840 = x10754 == 1;
int32_t x10839 = 128 / x10838;
bool x10841 = x10839 == 1;
bool x10845;
if (x454) {
bool x10842 = x10840 || x10841;
bool x10843 = x10754 == x10839;
bool x10844 = x10842 || x10843;
x10845 = x10844;
} else {
x10845 = false;
}
bool x10849;
if (x10845) {
x10849 = x10848;
} else {
x10849 = false;
}
bool x10850;
if (x10849) {
x10850 = x10848;
} else {
x10850 = false;
}
if (x10850) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10754,x10756,x10756,1,x10839,1,1);
assert(false && "");
}
bool x10856 = x10754 <= x10839;
int32_t x10857;
if (x10856) {
x10857 = x10839;
} else {
x10857 = x10754;
}
int32_t x10866 = x10857 * x10865;
int32_t x10867 = 64 * x10866;
float* x10868 = (float*)myMalloc(x10867 * sizeof(float));;
int32_t x10869;
if (x10840) {
x10869 = 0;
} else {
x10869 = x10762;
}
int32_t x10872;
if (x10841) {
x10872 = 0;
} else {
x10872 = 1;
}
for(int x10873=0; x10873 < 64; x10873++) {
int32_t x10885 = x10763 * x10873;
int32_t x10879 = x10866 * x10873;
for(int x10875=0; x10875 < x10857; x10875++) {
int32_t x10886 = x10869 * x10875;
int32_t x10887 = x10885 + x10886;
int32_t x10892 = x10872 * x10875;
int32_t x10881 = x10865 * x10875;
for(int x10877=0; x10877 < x10859; x10877++) {
int32_t x10888 = x10870 * x10877;
int32_t x10889 = x10887 + x10888;
int32_t x10883 = x10859 * x10877;
for(int x10878=0; x10878 < x10859; x10878++) {
int32_t x10890 = x10871 * x10878;
int32_t x10891 = x10889 + x10890;
float x10893 = x10765[x10891];
float x10894 = x10808[x10892];
int32_t x10880 = x10878 + x10879;
int32_t x10882 = x10880 + x10881;
int32_t x10884 = x10882 + x10883;
float x10895 = x10893 / x10894;
x10868[x10884] = x10895;

}

}

}

}
int32_t x10905 = 0;
int32_t x10906 = 1;
x10906 *= 1;
x10905 += 1;
x10906 *= 1;
x10906 *= 1;
int32_t x10911 = x10905;
bool x10912 = x10911 >= 2;
if (x10912) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10917 = x10911 == 0;
if (x10917) {
int32_t x10918 = x10906;
bool x10919 = x10918 == 128;
if (x10919) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10926 = x10906;
bool x10928 = x10857 == 1;
int32_t x10927 = 128 / x10926;
bool x10929 = x10927 == 1;
bool x10933;
if (x454) {
bool x10930 = x10928 || x10929;
bool x10931 = x10857 == x10927;
bool x10932 = x10930 || x10931;
x10933 = x10932;
} else {
x10933 = false;
}
bool x10937;
if (x10933) {
x10937 = x10936;
} else {
x10937 = false;
}
bool x10938;
if (x10937) {
x10938 = x10936;
} else {
x10938 = false;
}
if (x10938) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10857,x10859,x10859,1,x10927,1,1);
assert(false && "");
}
bool x10944 = x10857 <= x10927;
int32_t x10945;
if (x10944) {
x10945 = x10927;
} else {
x10945 = x10857;
}
int32_t x10954 = x10945 * x10953;
int32_t x10955 = 64 * x10954;
float* x10956 = (float*)myMalloc(x10955 * sizeof(float));;
int32_t x10957;
if (x10928) {
x10957 = 0;
} else {
x10957 = x10865;
}
int32_t x10960;
if (x10929) {
x10960 = 0;
} else {
x10960 = 1;
}
for(int x10961=0; x10961 < 64; x10961++) {
int32_t x10973 = x10866 * x10961;
int32_t x10967 = x10954 * x10961;
for(int x10963=0; x10963 < x10945; x10963++) {
int32_t x10974 = x10957 * x10963;
int32_t x10975 = x10973 + x10974;
int32_t x10980 = x10960 * x10963;
int32_t x10969 = x10953 * x10963;
for(int x10965=0; x10965 < x10947; x10965++) {
int32_t x10976 = x10958 * x10965;
int32_t x10977 = x10975 + x10976;
int32_t x10971 = x10947 * x10965;
for(int x10966=0; x10966 < x10947; x10966++) {
int32_t x10978 = x10959 * x10966;
int32_t x10979 = x10977 + x10978;
float x10981 = x10868[x10979];
float x10982 = x79[x10980];
int32_t x10968 = x10966 + x10967;
int32_t x10970 = x10968 + x10969;
int32_t x10972 = x10970 + x10971;
float x10983 = x10981 * x10982;
x10956[x10972] = x10983;

}

}

}

}
int32_t x10993 = 0;
int32_t x10994 = 1;
x10994 *= 1;
x10993 += 1;
x10994 *= 1;
x10994 *= 1;
int32_t x10999 = x10993;
bool x11000 = x10999 >= 2;
if (x11000) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11005 = x10999 == 0;
if (x11005) {
int32_t x11006 = x10994;
bool x11007 = x11006 == 128;
if (x11007) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11014 = x10994;
bool x11016 = x10945 == 1;
int32_t x11015 = 128 / x11014;
bool x11017 = x11015 == 1;
bool x11021;
if (x454) {
bool x11018 = x11016 || x11017;
bool x11019 = x10945 == x11015;
bool x11020 = x11018 || x11019;
x11021 = x11020;
} else {
x11021 = false;
}
bool x11025;
if (x11021) {
x11025 = x11024;
} else {
x11025 = false;
}
bool x11026;
if (x11025) {
x11026 = x11024;
} else {
x11026 = false;
}
if (x11026) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10945,x10947,x10947,1,x11015,1,1);
assert(false && "");
}
bool x11032 = x10945 <= x11015;
int32_t x11033;
if (x11032) {
x11033 = x11015;
} else {
x11033 = x10945;
}
int32_t x11042 = x11033 * x11041;
int32_t x11043 = 64 * x11042;
float* x11044 = (float*)myMalloc(x11043 * sizeof(float));;
int32_t x11045;
if (x11016) {
x11045 = 0;
} else {
x11045 = x10953;
}
int32_t x11048;
if (x11017) {
x11048 = 0;
} else {
x11048 = 1;
}
for(int x11049=0; x11049 < 64; x11049++) {
int32_t x11061 = x10954 * x11049;
int32_t x11055 = x11042 * x11049;
for(int x11051=0; x11051 < x11033; x11051++) {
int32_t x11062 = x11045 * x11051;
int32_t x11063 = x11061 + x11062;
int32_t x11068 = x11048 * x11051;
int32_t x11057 = x11041 * x11051;
for(int x11053=0; x11053 < x11035; x11053++) {
int32_t x11064 = x11046 * x11053;
int32_t x11065 = x11063 + x11064;
int32_t x11059 = x11035 * x11053;
for(int x11054=0; x11054 < x11035; x11054++) {
int32_t x11066 = x11047 * x11054;
int32_t x11067 = x11065 + x11066;
float x11069 = x10956[x11067];
float x11070 = x38[x11068];
int32_t x11056 = x11054 + x11055;
int32_t x11058 = x11056 + x11057;
int32_t x11060 = x11058 + x11059;
float x11071 = x11069 + x11070;
x11044[x11060] = x11071;

}

}

}

}
float* x11081 = (float*)myMalloc(x11043 * sizeof(float));;
for(int x11083=0; x11083 < x11043; x11083++) {
float x11084 = x11044[x11083];
bool x11085 = x11084 < 0.0f;
if (x11085) {
x11081[x11083] = 0.0f;
} else {
float x11088 = x11044[x11083];
x11081[x11083] = x11088;
}

}
float* x11102 = (float*)myMalloc(x11101 * sizeof(float));;
int32_t x11105 = 64 * x11033;
int32_t x11106 = x11105 * x11097;
float* x11107 = (float*)myMalloc(x11106 * sizeof(float));;
int32_t x11103 = x11033 * x11097;
for(int x11108=0; x11108 < 64; x11108++) {
int32_t x11109 = x11108 * x11042;
float* x11110 = x11081+x11109;
int32_t x11111 = x11108 * x11098;
float* x11112 = x11102+x11111;
int32_t x11113 = x11108 * x11103;
float* x11114 = x11107+x11113;
for(int x11115=0; x11115 < x11033; x11115++) {
int32_t x11116 = x11115 / 1;
int32_t x11120 = x11116 * x11096;
int32_t x11121 = x11120 * x11096;
int32_t x11117 = x11115 % 1;
int32_t x11118 = x11117 / 1;
int32_t x11122 = x11118 * x11096;
int32_t x11123 = x11122 * x11096;
int32_t x11124 = x11121 + x11123;
int32_t x11119 = x11117 % 1;
int32_t x11125 = x11119 * x11096;
int32_t x11126 = x11125 * x11096;
int32_t x11127 = x11124 + x11126;
float* x11128 = x11114+x11127;
int32_t x11129 = x11116 * x11035;
int32_t x11130 = x11129 * x11035;
float* x11131 = x11110+x11130;
for(int x11133=0; x11133 < x11096; x11133++) {
int32_t x11135 = x11133 * x11096;
float* x11136 = x11128+x11135;
int32_t x11134 = x11133 + x11118;
int32_t x11137 = x11134 * x11035;
int32_t x11138 = x11137 + x11119;
float* x11139 = x11131+x11138;
memcpy(x11136, x11139, 4 * x11096);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x11097,x11033,1,x55,x11033,x11114,x11097,1,x11112,x11097);

}
int32_t x11148 = 0;
int32_t x11149 = 1;
x11149 *= 1;
x11148 += 1;
x11149 *= 1;
x11149 *= 1;
int32_t x11154 = x11148;
bool x11155 = x11154 >= 2;
if (x11155) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11160 = x11154 == 0;
if (x11160) {
int32_t x11161 = x11149;
bool x11162 = x11161 == 512;
if (x11162) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11169 = x11149;
int32_t x11170 = 512 / x11169;
bool x11171 = x11170 == 1;
bool x11174;
if (x454) {
bool x11172 = 512 == x11170;
bool x11173 = x11171 || x11172;
x11174 = x11173;
} else {
x11174 = false;
}
bool x11178;
if (x11174) {
x11178 = x11177;
} else {
x11178 = false;
}
bool x11179;
if (x11178) {
x11179 = x11177;
} else {
x11179 = false;
}
if (x11179) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x11096,x11096,1,x11170,1,1);
assert(false && "");
}
bool x11185 = 512 <= x11170;
int32_t x11186;
if (x11185) {
x11186 = x11170;
} else {
x11186 = 512;
}
int32_t x11195 = x11186 * x11194;
int32_t x11196 = 64 * x11195;
float* x11197 = (float*)myMalloc(x11196 * sizeof(float));;
int32_t x11200;
if (x11171) {
x11200 = 0;
} else {
x11200 = 1;
}
for(int x11201=0; x11201 < 64; x11201++) {
int32_t x11213 = x11098 * x11201;
int32_t x11207 = x11195 * x11201;
for(int x11203=0; x11203 < x11186; x11203++) {
int32_t x11214 = x11097 * x11203;
int32_t x11215 = x11213 + x11214;
int32_t x11220 = x11200 * x11203;
int32_t x11209 = x11194 * x11203;
for(int x11205=0; x11205 < x11188; x11205++) {
int32_t x11216 = x11198 * x11205;
int32_t x11217 = x11215 + x11216;
int32_t x11211 = x11188 * x11205;
for(int x11206=0; x11206 < x11188; x11206++) {
int32_t x11218 = x11199 * x11206;
int32_t x11219 = x11217 + x11218;
float x11221 = x11102[x11219];
float x11222 = x19[x11220];
int32_t x11208 = x11206 + x11207;
int32_t x11210 = x11208 + x11209;
int32_t x11212 = x11210 + x11211;
float x11223 = x11221 - x11222;
x11197[x11212] = x11223;

}

}

}

}
float* x11233 = (float*)myMalloc(512 * sizeof(float));;
for(int x11234=0; x11234 < 512; x11234++) {
float x11235 = x234[x11234];
float x11236 = x11235 + 1.0E-5f;
x11233[x11234] = x11236;

}
float* x11240 = (float*)myMalloc(512 * sizeof(float));;
for(int x11241=0; x11241 < 512; x11241++) {
float x11242 = x11233[x11241];
double x11243 = (double)x11242;
double x11244 = sqrt(x11243);
float x11245 = (float)x11244;
x11240[x11241] = x11245;

}
int32_t x11249 = 0;
int32_t x11250 = 1;
x11250 *= 1;
x11249 += 1;
x11250 *= 1;
x11250 *= 1;
int32_t x11255 = x11249;
bool x11256 = x11255 >= 2;
if (x11256) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11261 = x11255 == 0;
if (x11261) {
int32_t x11262 = x11250;
bool x11263 = x11262 == 512;
if (x11263) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11270 = x11250;
bool x11272 = x11186 == 1;
int32_t x11271 = 512 / x11270;
bool x11273 = x11271 == 1;
bool x11277;
if (x454) {
bool x11274 = x11272 || x11273;
bool x11275 = x11186 == x11271;
bool x11276 = x11274 || x11275;
x11277 = x11276;
} else {
x11277 = false;
}
bool x11281;
if (x11277) {
x11281 = x11280;
} else {
x11281 = false;
}
bool x11282;
if (x11281) {
x11282 = x11280;
} else {
x11282 = false;
}
if (x11282) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11186,x11188,x11188,1,x11271,1,1);
assert(false && "");
}
bool x11288 = x11186 <= x11271;
int32_t x11289;
if (x11288) {
x11289 = x11271;
} else {
x11289 = x11186;
}
int32_t x11298 = x11289 * x11297;
int32_t x11299 = 64 * x11298;
float* x11300 = (float*)myMalloc(x11299 * sizeof(float));;
int32_t x11301;
if (x11272) {
x11301 = 0;
} else {
x11301 = x11194;
}
int32_t x11304;
if (x11273) {
x11304 = 0;
} else {
x11304 = 1;
}
for(int x11305=0; x11305 < 64; x11305++) {
int32_t x11317 = x11195 * x11305;
int32_t x11311 = x11298 * x11305;
for(int x11307=0; x11307 < x11289; x11307++) {
int32_t x11318 = x11301 * x11307;
int32_t x11319 = x11317 + x11318;
int32_t x11324 = x11304 * x11307;
int32_t x11313 = x11297 * x11307;
for(int x11309=0; x11309 < x11291; x11309++) {
int32_t x11320 = x11302 * x11309;
int32_t x11321 = x11319 + x11320;
int32_t x11315 = x11291 * x11309;
for(int x11310=0; x11310 < x11291; x11310++) {
int32_t x11322 = x11303 * x11310;
int32_t x11323 = x11321 + x11322;
float x11325 = x11197[x11323];
float x11326 = x11240[x11324];
int32_t x11312 = x11310 + x11311;
int32_t x11314 = x11312 + x11313;
int32_t x11316 = x11314 + x11315;
float x11327 = x11325 / x11326;
x11300[x11316] = x11327;

}

}

}

}
int32_t x11337 = 0;
int32_t x11338 = 1;
x11338 *= 1;
x11337 += 1;
x11338 *= 1;
x11338 *= 1;
int32_t x11343 = x11337;
bool x11344 = x11343 >= 2;
if (x11344) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11349 = x11343 == 0;
if (x11349) {
int32_t x11350 = x11338;
bool x11351 = x11350 == 512;
if (x11351) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11358 = x11338;
bool x11360 = x11289 == 1;
int32_t x11359 = 512 / x11358;
bool x11361 = x11359 == 1;
bool x11365;
if (x454) {
bool x11362 = x11360 || x11361;
bool x11363 = x11289 == x11359;
bool x11364 = x11362 || x11363;
x11365 = x11364;
} else {
x11365 = false;
}
bool x11369;
if (x11365) {
x11369 = x11368;
} else {
x11369 = false;
}
bool x11370;
if (x11369) {
x11370 = x11368;
} else {
x11370 = false;
}
if (x11370) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11289,x11291,x11291,1,x11359,1,1);
assert(false && "");
}
bool x11376 = x11289 <= x11359;
int32_t x11377;
if (x11376) {
x11377 = x11359;
} else {
x11377 = x11289;
}
int32_t x11386 = x11377 * x11385;
int32_t x11387 = 64 * x11386;
float* x11388 = (float*)myMalloc(x11387 * sizeof(float));;
int32_t x11389;
if (x11360) {
x11389 = 0;
} else {
x11389 = x11297;
}
int32_t x11392;
if (x11361) {
x11392 = 0;
} else {
x11392 = 1;
}
for(int x11393=0; x11393 < 64; x11393++) {
int32_t x11405 = x11298 * x11393;
int32_t x11399 = x11386 * x11393;
for(int x11395=0; x11395 < x11377; x11395++) {
int32_t x11406 = x11389 * x11395;
int32_t x11407 = x11405 + x11406;
int32_t x11412 = x11392 * x11395;
int32_t x11401 = x11385 * x11395;
for(int x11397=0; x11397 < x11379; x11397++) {
int32_t x11408 = x11390 * x11397;
int32_t x11409 = x11407 + x11408;
int32_t x11403 = x11379 * x11397;
for(int x11398=0; x11398 < x11379; x11398++) {
int32_t x11410 = x11391 * x11398;
int32_t x11411 = x11409 + x11410;
float x11413 = x11300[x11411];
float x11414 = x156[x11412];
int32_t x11400 = x11398 + x11399;
int32_t x11402 = x11400 + x11401;
int32_t x11404 = x11402 + x11403;
float x11415 = x11413 * x11414;
x11388[x11404] = x11415;

}

}

}

}
int32_t x11425 = 0;
int32_t x11426 = 1;
x11426 *= 1;
x11425 += 1;
x11426 *= 1;
x11426 *= 1;
int32_t x11431 = x11425;
bool x11432 = x11431 >= 2;
if (x11432) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11437 = x11431 == 0;
if (x11437) {
int32_t x11438 = x11426;
bool x11439 = x11438 == 512;
if (x11439) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11446 = x11426;
bool x11448 = x11377 == 1;
int32_t x11447 = 512 / x11446;
bool x11449 = x11447 == 1;
bool x11453;
if (x454) {
bool x11450 = x11448 || x11449;
bool x11451 = x11377 == x11447;
bool x11452 = x11450 || x11451;
x11453 = x11452;
} else {
x11453 = false;
}
bool x11457;
if (x11453) {
x11457 = x11456;
} else {
x11457 = false;
}
bool x11458;
if (x11457) {
x11458 = x11456;
} else {
x11458 = false;
}
if (x11458) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11377,x11379,x11379,1,x11447,1,1);
assert(false && "");
}
bool x11464 = x11377 <= x11447;
int32_t x11465;
if (x11464) {
x11465 = x11447;
} else {
x11465 = x11377;
}
int32_t x11474 = x11465 * x11473;
int32_t x11475 = 64 * x11474;
float* x11476 = (float*)myMalloc(x11475 * sizeof(float));;
int32_t x11477;
if (x11448) {
x11477 = 0;
} else {
x11477 = x11385;
}
int32_t x11480;
if (x11449) {
x11480 = 0;
} else {
x11480 = 1;
}
for(int x11481=0; x11481 < 64; x11481++) {
int32_t x11493 = x11386 * x11481;
int32_t x11487 = x11474 * x11481;
for(int x11483=0; x11483 < x11465; x11483++) {
int32_t x11494 = x11477 * x11483;
int32_t x11495 = x11493 + x11494;
int32_t x11500 = x11480 * x11483;
int32_t x11489 = x11473 * x11483;
for(int x11485=0; x11485 < x11467; x11485++) {
int32_t x11496 = x11478 * x11485;
int32_t x11497 = x11495 + x11496;
int32_t x11491 = x11467 * x11485;
for(int x11486=0; x11486 < x11467; x11486++) {
int32_t x11498 = x11479 * x11486;
int32_t x11499 = x11497 + x11498;
float x11501 = x11388[x11499];
float x11502 = x54[x11500];
int32_t x11488 = x11486 + x11487;
int32_t x11490 = x11488 + x11489;
int32_t x11492 = x11490 + x11491;
float x11503 = x11501 + x11502;
x11476[x11492] = x11503;

}

}

}

}
bool x11513 = x11465 == 1;
bool x11514 = x11513 || x10101;
bool x11515 = x11465 == x10053;
bool x11516 = x11514 || x11515;
bool x11521;
if (x11516) {
x11521 = x11520;
} else {
x11521 = false;
}
bool x11522;
if (x11521) {
x11522 = x11520;
} else {
x11522 = false;
}
if (x11522) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11465,x11467,x11467,64,x10053,x10055,x10055);
assert(false && "");
}
bool x11528 = x11465 <= x10053;
int32_t x11529;
if (x11528) {
x11529 = x10053;
} else {
x11529 = x11465;
}
int32_t x11545;
if (x11513) {
x11545 = 0;
} else {
x11545 = x11473;
}
for(int x11548=0; x11548 < 64; x11548++) {
int32_t x11554 = x11474 * x11548;
int32_t x11561 = x10062 * x11548;
for(int x11550=0; x11550 < x11529; x11550++) {
int32_t x11555 = x11545 * x11550;
int32_t x11556 = x11554 + x11555;
int32_t x11562 = x10133 * x11550;
int32_t x11563 = x11561 + x11562;
for(int x11552=0; x11552 < x11531; x11552++) {
int32_t x11557 = x11546 * x11552;
int32_t x11558 = x11556 + x11557;
int32_t x11564 = x10134 * x11552;
int32_t x11565 = x11563 + x11564;
for(int x11553=0; x11553 < x11531; x11553++) {
int32_t x11559 = x11547 * x11553;
int32_t x11560 = x11558 + x11559;
float x11568 = x11476[x11560];
int32_t x11566 = x10135 * x11553;
int32_t x11567 = x11565 + x11566;
float x11569 = x10168[x11567];
float x11570 = x11568 + x11569;
x11476[x11560] = x11570;

}

}

}

}
float* x11580 = (float*)myMalloc(x11475 * sizeof(float));;
for(int x11582=0; x11582 < x11475; x11582++) {
float x11583 = x11476[x11582];
bool x11584 = x11583 < 0.0f;
if (x11584) {
x11580[x11582] = 0.0f;
} else {
float x11587 = x11476[x11582];
x11580[x11582] = x11587;
}

}
float* x11601 = (float*)myMalloc(x11600 * sizeof(float));;
int32_t x11604 = 64 * x11465;
int32_t x11605 = x11604 * x11596;
float* x11606 = (float*)myMalloc(x11605 * sizeof(float));;
int32_t x11602 = x11465 * x11596;
for(int x11607=0; x11607 < 64; x11607++) {
int32_t x11608 = x11607 * x11474;
float* x11609 = x11580+x11608;
int32_t x11610 = x11607 * x11597;
float* x11611 = x11601+x11610;
int32_t x11612 = x11607 * x11602;
float* x11613 = x11606+x11612;
for(int x11614=0; x11614 < x11465; x11614++) {
int32_t x11615 = x11614 / 1;
int32_t x11619 = x11615 * x11595;
int32_t x11620 = x11619 * x11595;
int32_t x11616 = x11614 % 1;
int32_t x11617 = x11616 / 1;
int32_t x11621 = x11617 * x11595;
int32_t x11622 = x11621 * x11595;
int32_t x11623 = x11620 + x11622;
int32_t x11618 = x11616 % 1;
int32_t x11624 = x11618 * x11595;
int32_t x11625 = x11624 * x11595;
int32_t x11626 = x11623 + x11625;
float* x11627 = x11613+x11626;
int32_t x11628 = x11615 * x11467;
int32_t x11629 = x11628 * x11467;
float* x11630 = x11609+x11629;
for(int x11632=0; x11632 < x11595; x11632++) {
int32_t x11634 = x11632 * x11595;
float* x11635 = x11627+x11634;
int32_t x11633 = x11632 + x11617;
int32_t x11636 = x11633 * x11467;
int32_t x11637 = x11636 + x11618;
float* x11638 = x11630+x11637;
memcpy(x11635, x11638, 4 * x11595);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x11596,x11465,1,x180,x11465,x11613,x11596,1,x11611,x11596);

}
int32_t x11647 = 0;
int32_t x11648 = 1;
x11648 *= 1;
x11647 += 1;
x11648 *= 1;
x11648 *= 1;
int32_t x11653 = x11647;
bool x11654 = x11653 >= 2;
if (x11654) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11659 = x11653 == 0;
if (x11659) {
int32_t x11660 = x11648;
bool x11661 = x11660 == 256;
if (x11661) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11668 = x11648;
int32_t x11669 = 256 / x11668;
bool x11670 = x11669 == 1;
bool x11673;
if (x454) {
bool x11671 = 256 == x11669;
bool x11672 = x11670 || x11671;
x11673 = x11672;
} else {
x11673 = false;
}
bool x11677;
if (x11673) {
x11677 = x11676;
} else {
x11677 = false;
}
bool x11678;
if (x11677) {
x11678 = x11676;
} else {
x11678 = false;
}
if (x11678) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x11595,x11595,1,x11669,1,1);
assert(false && "");
}
bool x11684 = 256 <= x11669;
int32_t x11685;
if (x11684) {
x11685 = x11669;
} else {
x11685 = 256;
}
int32_t x11694 = x11685 * x11693;
int32_t x11695 = 64 * x11694;
float* x11696 = (float*)myMalloc(x11695 * sizeof(float));;
int32_t x11699;
if (x11670) {
x11699 = 0;
} else {
x11699 = 1;
}
for(int x11700=0; x11700 < 64; x11700++) {
int32_t x11712 = x11597 * x11700;
int32_t x11706 = x11694 * x11700;
for(int x11702=0; x11702 < x11685; x11702++) {
int32_t x11713 = x11596 * x11702;
int32_t x11714 = x11712 + x11713;
int32_t x11719 = x11699 * x11702;
int32_t x11708 = x11693 * x11702;
for(int x11704=0; x11704 < x11687; x11704++) {
int32_t x11715 = x11697 * x11704;
int32_t x11716 = x11714 + x11715;
int32_t x11710 = x11687 * x11704;
for(int x11705=0; x11705 < x11687; x11705++) {
int32_t x11717 = x11698 * x11705;
int32_t x11718 = x11716 + x11717;
float x11720 = x11601[x11718];
float x11721 = x131[x11719];
int32_t x11707 = x11705 + x11706;
int32_t x11709 = x11707 + x11708;
int32_t x11711 = x11709 + x11710;
float x11722 = x11720 - x11721;
x11696[x11711] = x11722;

}

}

}

}
float* x11732 = (float*)myMalloc(256 * sizeof(float));;
for(int x11733=0; x11733 < 256; x11733++) {
float x11734 = x198[x11733];
float x11735 = x11734 + 1.0E-5f;
x11732[x11733] = x11735;

}
float* x11739 = (float*)myMalloc(256 * sizeof(float));;
for(int x11740=0; x11740 < 256; x11740++) {
float x11741 = x11732[x11740];
double x11742 = (double)x11741;
double x11743 = sqrt(x11742);
float x11744 = (float)x11743;
x11739[x11740] = x11744;

}
int32_t x11748 = 0;
int32_t x11749 = 1;
x11749 *= 1;
x11748 += 1;
x11749 *= 1;
x11749 *= 1;
int32_t x11754 = x11748;
bool x11755 = x11754 >= 2;
if (x11755) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11760 = x11754 == 0;
if (x11760) {
int32_t x11761 = x11749;
bool x11762 = x11761 == 256;
if (x11762) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11769 = x11749;
bool x11771 = x11685 == 1;
int32_t x11770 = 256 / x11769;
bool x11772 = x11770 == 1;
bool x11776;
if (x454) {
bool x11773 = x11771 || x11772;
bool x11774 = x11685 == x11770;
bool x11775 = x11773 || x11774;
x11776 = x11775;
} else {
x11776 = false;
}
bool x11780;
if (x11776) {
x11780 = x11779;
} else {
x11780 = false;
}
bool x11781;
if (x11780) {
x11781 = x11779;
} else {
x11781 = false;
}
if (x11781) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11685,x11687,x11687,1,x11770,1,1);
assert(false && "");
}
bool x11787 = x11685 <= x11770;
int32_t x11788;
if (x11787) {
x11788 = x11770;
} else {
x11788 = x11685;
}
int32_t x11797 = x11788 * x11796;
int32_t x11798 = 64 * x11797;
float* x11799 = (float*)myMalloc(x11798 * sizeof(float));;
int32_t x11800;
if (x11771) {
x11800 = 0;
} else {
x11800 = x11693;
}
int32_t x11803;
if (x11772) {
x11803 = 0;
} else {
x11803 = 1;
}
for(int x11804=0; x11804 < 64; x11804++) {
int32_t x11816 = x11694 * x11804;
int32_t x11810 = x11797 * x11804;
for(int x11806=0; x11806 < x11788; x11806++) {
int32_t x11817 = x11800 * x11806;
int32_t x11818 = x11816 + x11817;
int32_t x11823 = x11803 * x11806;
int32_t x11812 = x11796 * x11806;
for(int x11808=0; x11808 < x11790; x11808++) {
int32_t x11819 = x11801 * x11808;
int32_t x11820 = x11818 + x11819;
int32_t x11814 = x11790 * x11808;
for(int x11809=0; x11809 < x11790; x11809++) {
int32_t x11821 = x11802 * x11809;
int32_t x11822 = x11820 + x11821;
float x11824 = x11696[x11822];
float x11825 = x11739[x11823];
int32_t x11811 = x11809 + x11810;
int32_t x11813 = x11811 + x11812;
int32_t x11815 = x11813 + x11814;
float x11826 = x11824 / x11825;
x11799[x11815] = x11826;

}

}

}

}
int32_t x11836 = 0;
int32_t x11837 = 1;
x11837 *= 1;
x11836 += 1;
x11837 *= 1;
x11837 *= 1;
int32_t x11842 = x11836;
bool x11843 = x11842 >= 2;
if (x11843) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11848 = x11842 == 0;
if (x11848) {
int32_t x11849 = x11837;
bool x11850 = x11849 == 256;
if (x11850) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11857 = x11837;
bool x11859 = x11788 == 1;
int32_t x11858 = 256 / x11857;
bool x11860 = x11858 == 1;
bool x11864;
if (x454) {
bool x11861 = x11859 || x11860;
bool x11862 = x11788 == x11858;
bool x11863 = x11861 || x11862;
x11864 = x11863;
} else {
x11864 = false;
}
bool x11868;
if (x11864) {
x11868 = x11867;
} else {
x11868 = false;
}
bool x11869;
if (x11868) {
x11869 = x11867;
} else {
x11869 = false;
}
if (x11869) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11788,x11790,x11790,1,x11858,1,1);
assert(false && "");
}
bool x11875 = x11788 <= x11858;
int32_t x11876;
if (x11875) {
x11876 = x11858;
} else {
x11876 = x11788;
}
int32_t x11885 = x11876 * x11884;
int32_t x11886 = 64 * x11885;
float* x11887 = (float*)myMalloc(x11886 * sizeof(float));;
int32_t x11888;
if (x11859) {
x11888 = 0;
} else {
x11888 = x11796;
}
int32_t x11891;
if (x11860) {
x11891 = 0;
} else {
x11891 = 1;
}
for(int x11892=0; x11892 < 64; x11892++) {
int32_t x11904 = x11797 * x11892;
int32_t x11898 = x11885 * x11892;
for(int x11894=0; x11894 < x11876; x11894++) {
int32_t x11905 = x11888 * x11894;
int32_t x11906 = x11904 + x11905;
int32_t x11911 = x11891 * x11894;
int32_t x11900 = x11884 * x11894;
for(int x11896=0; x11896 < x11878; x11896++) {
int32_t x11907 = x11889 * x11896;
int32_t x11908 = x11906 + x11907;
int32_t x11902 = x11878 * x11896;
for(int x11897=0; x11897 < x11878; x11897++) {
int32_t x11909 = x11890 * x11897;
int32_t x11910 = x11908 + x11909;
float x11912 = x11799[x11910];
float x11913 = x270[x11911];
int32_t x11899 = x11897 + x11898;
int32_t x11901 = x11899 + x11900;
int32_t x11903 = x11901 + x11902;
float x11914 = x11912 * x11913;
x11887[x11903] = x11914;

}

}

}

}
int32_t x11924 = 0;
int32_t x11925 = 1;
x11925 *= 1;
x11924 += 1;
x11925 *= 1;
x11925 *= 1;
int32_t x11930 = x11924;
bool x11931 = x11930 >= 2;
if (x11931) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11936 = x11930 == 0;
if (x11936) {
int32_t x11937 = x11925;
bool x11938 = x11937 == 256;
if (x11938) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11945 = x11925;
bool x11947 = x11876 == 1;
int32_t x11946 = 256 / x11945;
bool x11948 = x11946 == 1;
bool x11952;
if (x454) {
bool x11949 = x11947 || x11948;
bool x11950 = x11876 == x11946;
bool x11951 = x11949 || x11950;
x11952 = x11951;
} else {
x11952 = false;
}
bool x11956;
if (x11952) {
x11956 = x11955;
} else {
x11956 = false;
}
bool x11957;
if (x11956) {
x11957 = x11955;
} else {
x11957 = false;
}
if (x11957) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11876,x11878,x11878,1,x11946,1,1);
assert(false && "");
}
bool x11963 = x11876 <= x11946;
int32_t x11964;
if (x11963) {
x11964 = x11946;
} else {
x11964 = x11876;
}
int32_t x11973 = x11964 * x11972;
int32_t x11974 = 64 * x11973;
float* x11975 = (float*)myMalloc(x11974 * sizeof(float));;
int32_t x11976;
if (x11947) {
x11976 = 0;
} else {
x11976 = x11884;
}
int32_t x11979;
if (x11948) {
x11979 = 0;
} else {
x11979 = 1;
}
for(int x11980=0; x11980 < 64; x11980++) {
int32_t x11992 = x11885 * x11980;
int32_t x11986 = x11973 * x11980;
for(int x11982=0; x11982 < x11964; x11982++) {
int32_t x11993 = x11976 * x11982;
int32_t x11994 = x11992 + x11993;
int32_t x11999 = x11979 * x11982;
int32_t x11988 = x11972 * x11982;
for(int x11984=0; x11984 < x11966; x11984++) {
int32_t x11995 = x11977 * x11984;
int32_t x11996 = x11994 + x11995;
int32_t x11990 = x11966 * x11984;
for(int x11985=0; x11985 < x11966; x11985++) {
int32_t x11997 = x11978 * x11985;
int32_t x11998 = x11996 + x11997;
float x12000 = x11887[x11998];
float x12001 = x21[x11999];
int32_t x11987 = x11985 + x11986;
int32_t x11989 = x11987 + x11988;
int32_t x11991 = x11989 + x11990;
float x12002 = x12000 + x12001;
x11975[x11991] = x12002;

}

}

}

}
float* x12012 = (float*)myMalloc(x11974 * sizeof(float));;
for(int x12014=0; x12014 < x11974; x12014++) {
float x12015 = x11975[x12014];
bool x12016 = x12015 < 0.0f;
if (x12016) {
x12012[x12014] = 0.0f;
} else {
float x12019 = x11975[x12014];
x12012[x12014] = x12019;
}

}
float* x12034 = (float*)myMalloc(x12033 * sizeof(float));;
int32_t x12035 = 9 * x11964;
int32_t x12038 = 64 * x12035;
int32_t x12039 = x12038 * x12029;
float* x12040 = (float*)myMalloc(x12039 * sizeof(float));;
int32_t x12036 = x12035 * x12029;
int32_t x12048 = x11964 * 3;
int32_t x12049 = x12048 * 3;
for(int x12041=0; x12041 < 64; x12041++) {
int32_t x12042 = x12041 * x11973;
float* x12043 = x12012+x12042;
int32_t x12044 = x12041 * x12030;
float* x12045 = x12034+x12044;
int32_t x12046 = x12041 * x12036;
float* x12047 = x12040+x12046;
for(int x12051=0; x12051 < x12049; x12051++) {
int32_t x12052 = x12051 / 9;
int32_t x12056 = x12052 * 3;
int32_t x12057 = x12056 * 3;
int32_t x12058 = x12057 * x12028;
int32_t x12059 = x12058 * x12028;
int32_t x12053 = x12051 % 9;
int32_t x12054 = x12053 / 3;
int32_t x12060 = x12054 * 3;
int32_t x12061 = x12060 * x12028;
int32_t x12062 = x12061 * x12028;
int32_t x12063 = x12059 + x12062;
int32_t x12055 = x12053 % 3;
int32_t x12064 = x12055 * x12028;
int32_t x12065 = x12064 * x12028;
int32_t x12066 = x12063 + x12065;
float* x12067 = x12047+x12066;
int32_t x12068 = x12052 * x11966;
int32_t x12069 = x12068 * x11966;
float* x12070 = x12043+x12069;
for(int x12072=0; x12072 < x12028; x12072++) {
int32_t x12073 = x12072 * 2;
int32_t x12074 = x12073 - 1;
int32_t x12075 = x12074 + x12054;
bool x12076 = x12075 < 0;
bool x12077 = x12075 >= x11966;
bool x12078 = x12076 || x12077;
if (x12078) {
int32_t x12079 = x12072 * x12028;
float* x12080 = x12067+x12079;
memset(x12080, 0, 4 * x12028);;
} else {
int32_t x12079 = x12072 * x12028;
int32_t x12095 = x12075 * x11966;
for(int x12083=0; x12083 < x12028; x12083++) {
int32_t x12084 = x12083 * 2;
int32_t x12085 = x12084 - 1;
int32_t x12086 = x12085 + x12055;
bool x12087 = x12086 < 0;
bool x12088 = x12086 >= x11966;
bool x12089 = x12087 || x12088;
if (x12089) {
int32_t x12090 = x12079 + x12083;
float* x12091 = x12067+x12090;
memset(x12091, 0, 4 * 1);;
} else {
int32_t x12090 = x12079 + x12083;
float* x12094 = x12067+x12090;
int32_t x12096 = x12095 + x12086;
float* x12097 = x12070+x12096;
memcpy(x12094, x12097, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x12029,x12035,1,x175,x12035,x12047,x12029,1,x12045,x12029);

}
int32_t x12112 = 0;
int32_t x12113 = 1;
x12113 *= 1;
x12112 += 1;
x12113 *= 1;
x12113 *= 1;
int32_t x12118 = x12112;
bool x12119 = x12118 >= 2;
if (x12119) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12124 = x12118 == 0;
if (x12124) {
int32_t x12125 = x12113;
bool x12126 = x12125 == 256;
if (x12126) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12133 = x12113;
int32_t x12134 = 256 / x12133;
bool x12135 = x12134 == 1;
bool x12138;
if (x454) {
bool x12136 = 256 == x12134;
bool x12137 = x12135 || x12136;
x12138 = x12137;
} else {
x12138 = false;
}
bool x12142;
if (x12138) {
x12142 = x12141;
} else {
x12142 = false;
}
bool x12143;
if (x12142) {
x12143 = x12141;
} else {
x12143 = false;
}
if (x12143) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x12028,x12028,1,x12134,1,1);
assert(false && "");
}
bool x12149 = 256 <= x12134;
int32_t x12150;
if (x12149) {
x12150 = x12134;
} else {
x12150 = 256;
}
int32_t x12159 = x12150 * x12158;
int32_t x12160 = 64 * x12159;
float* x12161 = (float*)myMalloc(x12160 * sizeof(float));;
int32_t x12164;
if (x12135) {
x12164 = 0;
} else {
x12164 = 1;
}
for(int x12165=0; x12165 < 64; x12165++) {
int32_t x12177 = x12030 * x12165;
int32_t x12171 = x12159 * x12165;
for(int x12167=0; x12167 < x12150; x12167++) {
int32_t x12178 = x12029 * x12167;
int32_t x12179 = x12177 + x12178;
int32_t x12184 = x12164 * x12167;
int32_t x12173 = x12158 * x12167;
for(int x12169=0; x12169 < x12152; x12169++) {
int32_t x12180 = x12162 * x12169;
int32_t x12181 = x12179 + x12180;
int32_t x12175 = x12152 * x12169;
for(int x12170=0; x12170 < x12152; x12170++) {
int32_t x12182 = x12163 * x12170;
int32_t x12183 = x12181 + x12182;
float x12185 = x12034[x12183];
float x12186 = x229[x12184];
int32_t x12172 = x12170 + x12171;
int32_t x12174 = x12172 + x12173;
int32_t x12176 = x12174 + x12175;
float x12187 = x12185 - x12186;
x12161[x12176] = x12187;

}

}

}

}
float* x12197 = (float*)myMalloc(256 * sizeof(float));;
for(int x12198=0; x12198 < 256; x12198++) {
float x12199 = x99[x12198];
float x12200 = x12199 + 1.0E-5f;
x12197[x12198] = x12200;

}
float* x12204 = (float*)myMalloc(256 * sizeof(float));;
for(int x12205=0; x12205 < 256; x12205++) {
float x12206 = x12197[x12205];
double x12207 = (double)x12206;
double x12208 = sqrt(x12207);
float x12209 = (float)x12208;
x12204[x12205] = x12209;

}
int32_t x12213 = 0;
int32_t x12214 = 1;
x12214 *= 1;
x12213 += 1;
x12214 *= 1;
x12214 *= 1;
int32_t x12219 = x12213;
bool x12220 = x12219 >= 2;
if (x12220) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12225 = x12219 == 0;
if (x12225) {
int32_t x12226 = x12214;
bool x12227 = x12226 == 256;
if (x12227) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12234 = x12214;
bool x12236 = x12150 == 1;
int32_t x12235 = 256 / x12234;
bool x12237 = x12235 == 1;
bool x12241;
if (x454) {
bool x12238 = x12236 || x12237;
bool x12239 = x12150 == x12235;
bool x12240 = x12238 || x12239;
x12241 = x12240;
} else {
x12241 = false;
}
bool x12245;
if (x12241) {
x12245 = x12244;
} else {
x12245 = false;
}
bool x12246;
if (x12245) {
x12246 = x12244;
} else {
x12246 = false;
}
if (x12246) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12150,x12152,x12152,1,x12235,1,1);
assert(false && "");
}
bool x12252 = x12150 <= x12235;
int32_t x12253;
if (x12252) {
x12253 = x12235;
} else {
x12253 = x12150;
}
int32_t x12262 = x12253 * x12261;
int32_t x12263 = 64 * x12262;
float* x12264 = (float*)myMalloc(x12263 * sizeof(float));;
int32_t x12265;
if (x12236) {
x12265 = 0;
} else {
x12265 = x12158;
}
int32_t x12268;
if (x12237) {
x12268 = 0;
} else {
x12268 = 1;
}
for(int x12269=0; x12269 < 64; x12269++) {
int32_t x12281 = x12159 * x12269;
int32_t x12275 = x12262 * x12269;
for(int x12271=0; x12271 < x12253; x12271++) {
int32_t x12282 = x12265 * x12271;
int32_t x12283 = x12281 + x12282;
int32_t x12288 = x12268 * x12271;
int32_t x12277 = x12261 * x12271;
for(int x12273=0; x12273 < x12255; x12273++) {
int32_t x12284 = x12266 * x12273;
int32_t x12285 = x12283 + x12284;
int32_t x12279 = x12255 * x12273;
for(int x12274=0; x12274 < x12255; x12274++) {
int32_t x12286 = x12267 * x12274;
int32_t x12287 = x12285 + x12286;
float x12289 = x12161[x12287];
float x12290 = x12204[x12288];
int32_t x12276 = x12274 + x12275;
int32_t x12278 = x12276 + x12277;
int32_t x12280 = x12278 + x12279;
float x12291 = x12289 / x12290;
x12264[x12280] = x12291;

}

}

}

}
int32_t x12301 = 0;
int32_t x12302 = 1;
x12302 *= 1;
x12301 += 1;
x12302 *= 1;
x12302 *= 1;
int32_t x12307 = x12301;
bool x12308 = x12307 >= 2;
if (x12308) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12313 = x12307 == 0;
if (x12313) {
int32_t x12314 = x12302;
bool x12315 = x12314 == 256;
if (x12315) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12322 = x12302;
bool x12324 = x12253 == 1;
int32_t x12323 = 256 / x12322;
bool x12325 = x12323 == 1;
bool x12329;
if (x454) {
bool x12326 = x12324 || x12325;
bool x12327 = x12253 == x12323;
bool x12328 = x12326 || x12327;
x12329 = x12328;
} else {
x12329 = false;
}
bool x12333;
if (x12329) {
x12333 = x12332;
} else {
x12333 = false;
}
bool x12334;
if (x12333) {
x12334 = x12332;
} else {
x12334 = false;
}
if (x12334) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12253,x12255,x12255,1,x12323,1,1);
assert(false && "");
}
bool x12340 = x12253 <= x12323;
int32_t x12341;
if (x12340) {
x12341 = x12323;
} else {
x12341 = x12253;
}
int32_t x12350 = x12341 * x12349;
int32_t x12351 = 64 * x12350;
float* x12352 = (float*)myMalloc(x12351 * sizeof(float));;
int32_t x12353;
if (x12324) {
x12353 = 0;
} else {
x12353 = x12261;
}
int32_t x12356;
if (x12325) {
x12356 = 0;
} else {
x12356 = 1;
}
for(int x12357=0; x12357 < 64; x12357++) {
int32_t x12369 = x12262 * x12357;
int32_t x12363 = x12350 * x12357;
for(int x12359=0; x12359 < x12341; x12359++) {
int32_t x12370 = x12353 * x12359;
int32_t x12371 = x12369 + x12370;
int32_t x12376 = x12356 * x12359;
int32_t x12365 = x12349 * x12359;
for(int x12361=0; x12361 < x12343; x12361++) {
int32_t x12372 = x12354 * x12361;
int32_t x12373 = x12371 + x12372;
int32_t x12367 = x12343 * x12361;
for(int x12362=0; x12362 < x12343; x12362++) {
int32_t x12374 = x12355 * x12362;
int32_t x12375 = x12373 + x12374;
float x12377 = x12264[x12375];
float x12378 = x108[x12376];
int32_t x12364 = x12362 + x12363;
int32_t x12366 = x12364 + x12365;
int32_t x12368 = x12366 + x12367;
float x12379 = x12377 * x12378;
x12352[x12368] = x12379;

}

}

}

}
int32_t x12389 = 0;
int32_t x12390 = 1;
x12390 *= 1;
x12389 += 1;
x12390 *= 1;
x12390 *= 1;
int32_t x12395 = x12389;
bool x12396 = x12395 >= 2;
if (x12396) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12401 = x12395 == 0;
if (x12401) {
int32_t x12402 = x12390;
bool x12403 = x12402 == 256;
if (x12403) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12410 = x12390;
bool x12412 = x12341 == 1;
int32_t x12411 = 256 / x12410;
bool x12413 = x12411 == 1;
bool x12417;
if (x454) {
bool x12414 = x12412 || x12413;
bool x12415 = x12341 == x12411;
bool x12416 = x12414 || x12415;
x12417 = x12416;
} else {
x12417 = false;
}
bool x12421;
if (x12417) {
x12421 = x12420;
} else {
x12421 = false;
}
bool x12422;
if (x12421) {
x12422 = x12420;
} else {
x12422 = false;
}
if (x12422) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12341,x12343,x12343,1,x12411,1,1);
assert(false && "");
}
bool x12428 = x12341 <= x12411;
int32_t x12429;
if (x12428) {
x12429 = x12411;
} else {
x12429 = x12341;
}
int32_t x12438 = x12429 * x12437;
int32_t x12439 = 64 * x12438;
float* x12440 = (float*)myMalloc(x12439 * sizeof(float));;
int32_t x12441;
if (x12412) {
x12441 = 0;
} else {
x12441 = x12349;
}
int32_t x12444;
if (x12413) {
x12444 = 0;
} else {
x12444 = 1;
}
for(int x12445=0; x12445 < 64; x12445++) {
int32_t x12457 = x12350 * x12445;
int32_t x12451 = x12438 * x12445;
for(int x12447=0; x12447 < x12429; x12447++) {
int32_t x12458 = x12441 * x12447;
int32_t x12459 = x12457 + x12458;
int32_t x12464 = x12444 * x12447;
int32_t x12453 = x12437 * x12447;
for(int x12449=0; x12449 < x12431; x12449++) {
int32_t x12460 = x12442 * x12449;
int32_t x12461 = x12459 + x12460;
int32_t x12455 = x12431 * x12449;
for(int x12450=0; x12450 < x12431; x12450++) {
int32_t x12462 = x12443 * x12450;
int32_t x12463 = x12461 + x12462;
float x12465 = x12352[x12463];
float x12466 = x16[x12464];
int32_t x12452 = x12450 + x12451;
int32_t x12454 = x12452 + x12453;
int32_t x12456 = x12454 + x12455;
float x12467 = x12465 + x12466;
x12440[x12456] = x12467;

}

}

}

}
float* x12477 = (float*)myMalloc(x12439 * sizeof(float));;
for(int x12479=0; x12479 < x12439; x12479++) {
float x12480 = x12440[x12479];
bool x12481 = x12480 < 0.0f;
if (x12481) {
x12477[x12479] = 0.0f;
} else {
float x12484 = x12440[x12479];
x12477[x12479] = x12484;
}

}
float* x12498 = (float*)myMalloc(x12497 * sizeof(float));;
int32_t x12501 = 64 * x12429;
int32_t x12502 = x12501 * x12493;
float* x12503 = (float*)myMalloc(x12502 * sizeof(float));;
int32_t x12499 = x12429 * x12493;
for(int x12504=0; x12504 < 64; x12504++) {
int32_t x12505 = x12504 * x12438;
float* x12506 = x12477+x12505;
int32_t x12507 = x12504 * x12494;
float* x12508 = x12498+x12507;
int32_t x12509 = x12504 * x12499;
float* x12510 = x12503+x12509;
for(int x12511=0; x12511 < x12429; x12511++) {
int32_t x12512 = x12511 / 1;
int32_t x12516 = x12512 * x12492;
int32_t x12517 = x12516 * x12492;
int32_t x12513 = x12511 % 1;
int32_t x12514 = x12513 / 1;
int32_t x12518 = x12514 * x12492;
int32_t x12519 = x12518 * x12492;
int32_t x12520 = x12517 + x12519;
int32_t x12515 = x12513 % 1;
int32_t x12521 = x12515 * x12492;
int32_t x12522 = x12521 * x12492;
int32_t x12523 = x12520 + x12522;
float* x12524 = x12510+x12523;
int32_t x12525 = x12512 * x12431;
int32_t x12526 = x12525 * x12431;
float* x12527 = x12506+x12526;
for(int x12529=0; x12529 < x12492; x12529++) {
int32_t x12531 = x12529 * x12492;
float* x12532 = x12524+x12531;
int32_t x12530 = x12529 + x12514;
int32_t x12533 = x12530 * x12431;
int32_t x12534 = x12533 + x12515;
float* x12535 = x12527+x12534;
memcpy(x12532, x12535, 4 * x12492);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x12493,x12429,1,x269,x12429,x12510,x12493,1,x12508,x12493);

}
int32_t x12544 = 0;
int32_t x12545 = 1;
x12545 *= 1;
x12544 += 1;
x12545 *= 1;
x12545 *= 1;
int32_t x12550 = x12544;
bool x12551 = x12550 >= 2;
if (x12551) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12556 = x12550 == 0;
if (x12556) {
int32_t x12557 = x12545;
bool x12558 = x12557 == 1024;
if (x12558) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12565 = x12545;
int32_t x12566 = 1024 / x12565;
bool x12567 = x12566 == 1;
bool x12570;
if (x454) {
bool x12568 = 1024 == x12566;
bool x12569 = x12567 || x12568;
x12570 = x12569;
} else {
x12570 = false;
}
bool x12574;
if (x12570) {
x12574 = x12573;
} else {
x12574 = false;
}
bool x12575;
if (x12574) {
x12575 = x12573;
} else {
x12575 = false;
}
if (x12575) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x12492,x12492,1,x12566,1,1);
assert(false && "");
}
bool x12581 = 1024 <= x12566;
int32_t x12582;
if (x12581) {
x12582 = x12566;
} else {
x12582 = 1024;
}
int32_t x12591 = x12582 * x12590;
int32_t x12592 = 64 * x12591;
float* x12593 = (float*)myMalloc(x12592 * sizeof(float));;
int32_t x12596;
if (x12567) {
x12596 = 0;
} else {
x12596 = 1;
}
for(int x12597=0; x12597 < 64; x12597++) {
int32_t x12609 = x12494 * x12597;
int32_t x12603 = x12591 * x12597;
for(int x12599=0; x12599 < x12582; x12599++) {
int32_t x12610 = x12493 * x12599;
int32_t x12611 = x12609 + x12610;
int32_t x12616 = x12596 * x12599;
int32_t x12605 = x12590 * x12599;
for(int x12601=0; x12601 < x12584; x12601++) {
int32_t x12612 = x12594 * x12601;
int32_t x12613 = x12611 + x12612;
int32_t x12607 = x12584 * x12601;
for(int x12602=0; x12602 < x12584; x12602++) {
int32_t x12614 = x12595 * x12602;
int32_t x12615 = x12613 + x12614;
float x12617 = x12498[x12615];
float x12618 = x216[x12616];
int32_t x12604 = x12602 + x12603;
int32_t x12606 = x12604 + x12605;
int32_t x12608 = x12606 + x12607;
float x12619 = x12617 - x12618;
x12593[x12608] = x12619;

}

}

}

}
float* x12629 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12631=0; x12631 < 1024; x12631++) {
float x12632 = x267[x12631];
float x12633 = x12632 + 1.0E-5f;
x12629[x12631] = x12633;

}
float* x12637 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12638=0; x12638 < 1024; x12638++) {
float x12639 = x12629[x12638];
double x12640 = (double)x12639;
double x12641 = sqrt(x12640);
float x12642 = (float)x12641;
x12637[x12638] = x12642;

}
int32_t x12646 = 0;
int32_t x12647 = 1;
x12647 *= 1;
x12646 += 1;
x12647 *= 1;
x12647 *= 1;
int32_t x12652 = x12646;
bool x12653 = x12652 >= 2;
if (x12653) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12658 = x12652 == 0;
if (x12658) {
int32_t x12659 = x12647;
bool x12660 = x12659 == 1024;
if (x12660) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12667 = x12647;
bool x12669 = x12582 == 1;
int32_t x12668 = 1024 / x12667;
bool x12670 = x12668 == 1;
bool x12674;
if (x454) {
bool x12671 = x12669 || x12670;
bool x12672 = x12582 == x12668;
bool x12673 = x12671 || x12672;
x12674 = x12673;
} else {
x12674 = false;
}
bool x12678;
if (x12674) {
x12678 = x12677;
} else {
x12678 = false;
}
bool x12679;
if (x12678) {
x12679 = x12677;
} else {
x12679 = false;
}
if (x12679) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12582,x12584,x12584,1,x12668,1,1);
assert(false && "");
}
bool x12685 = x12582 <= x12668;
int32_t x12686;
if (x12685) {
x12686 = x12668;
} else {
x12686 = x12582;
}
int32_t x12695 = x12686 * x12694;
int32_t x12696 = 64 * x12695;
float* x12697 = (float*)myMalloc(x12696 * sizeof(float));;
int32_t x12698;
if (x12669) {
x12698 = 0;
} else {
x12698 = x12590;
}
int32_t x12701;
if (x12670) {
x12701 = 0;
} else {
x12701 = 1;
}
for(int x12702=0; x12702 < 64; x12702++) {
int32_t x12714 = x12591 * x12702;
int32_t x12708 = x12695 * x12702;
for(int x12704=0; x12704 < x12686; x12704++) {
int32_t x12715 = x12698 * x12704;
int32_t x12716 = x12714 + x12715;
int32_t x12721 = x12701 * x12704;
int32_t x12710 = x12694 * x12704;
for(int x12706=0; x12706 < x12688; x12706++) {
int32_t x12717 = x12699 * x12706;
int32_t x12718 = x12716 + x12717;
int32_t x12712 = x12688 * x12706;
for(int x12707=0; x12707 < x12688; x12707++) {
int32_t x12719 = x12700 * x12707;
int32_t x12720 = x12718 + x12719;
float x12722 = x12593[x12720];
float x12723 = x12637[x12721];
int32_t x12709 = x12707 + x12708;
int32_t x12711 = x12709 + x12710;
int32_t x12713 = x12711 + x12712;
float x12724 = x12722 / x12723;
x12697[x12713] = x12724;

}

}

}

}
int32_t x12734 = 0;
int32_t x12735 = 1;
x12735 *= 1;
x12734 += 1;
x12735 *= 1;
x12735 *= 1;
int32_t x12740 = x12734;
bool x12741 = x12740 >= 2;
if (x12741) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12746 = x12740 == 0;
if (x12746) {
int32_t x12747 = x12735;
bool x12748 = x12747 == 1024;
if (x12748) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12755 = x12735;
bool x12757 = x12686 == 1;
int32_t x12756 = 1024 / x12755;
bool x12758 = x12756 == 1;
bool x12762;
if (x454) {
bool x12759 = x12757 || x12758;
bool x12760 = x12686 == x12756;
bool x12761 = x12759 || x12760;
x12762 = x12761;
} else {
x12762 = false;
}
bool x12766;
if (x12762) {
x12766 = x12765;
} else {
x12766 = false;
}
bool x12767;
if (x12766) {
x12767 = x12765;
} else {
x12767 = false;
}
if (x12767) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12686,x12688,x12688,1,x12756,1,1);
assert(false && "");
}
bool x12773 = x12686 <= x12756;
int32_t x12774;
if (x12773) {
x12774 = x12756;
} else {
x12774 = x12686;
}
int32_t x12783 = x12774 * x12782;
int32_t x12784 = 64 * x12783;
float* x12785 = (float*)myMalloc(x12784 * sizeof(float));;
int32_t x12786;
if (x12757) {
x12786 = 0;
} else {
x12786 = x12694;
}
int32_t x12789;
if (x12758) {
x12789 = 0;
} else {
x12789 = 1;
}
for(int x12790=0; x12790 < 64; x12790++) {
int32_t x12802 = x12695 * x12790;
int32_t x12796 = x12783 * x12790;
for(int x12792=0; x12792 < x12774; x12792++) {
int32_t x12803 = x12786 * x12792;
int32_t x12804 = x12802 + x12803;
int32_t x12809 = x12789 * x12792;
int32_t x12798 = x12782 * x12792;
for(int x12794=0; x12794 < x12776; x12794++) {
int32_t x12805 = x12787 * x12794;
int32_t x12806 = x12804 + x12805;
int32_t x12800 = x12776 * x12794;
for(int x12795=0; x12795 < x12776; x12795++) {
int32_t x12807 = x12788 * x12795;
int32_t x12808 = x12806 + x12807;
float x12810 = x12697[x12808];
float x12811 = x18[x12809];
int32_t x12797 = x12795 + x12796;
int32_t x12799 = x12797 + x12798;
int32_t x12801 = x12799 + x12800;
float x12812 = x12810 * x12811;
x12785[x12801] = x12812;

}

}

}

}
int32_t x12822 = 0;
int32_t x12823 = 1;
x12823 *= 1;
x12822 += 1;
x12823 *= 1;
x12823 *= 1;
int32_t x12828 = x12822;
bool x12829 = x12828 >= 2;
if (x12829) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12834 = x12828 == 0;
if (x12834) {
int32_t x12835 = x12823;
bool x12836 = x12835 == 1024;
if (x12836) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12843 = x12823;
bool x12845 = x12774 == 1;
int32_t x12844 = 1024 / x12843;
bool x12846 = x12844 == 1;
bool x12850;
if (x454) {
bool x12847 = x12845 || x12846;
bool x12848 = x12774 == x12844;
bool x12849 = x12847 || x12848;
x12850 = x12849;
} else {
x12850 = false;
}
bool x12854;
if (x12850) {
x12854 = x12853;
} else {
x12854 = false;
}
bool x12855;
if (x12854) {
x12855 = x12853;
} else {
x12855 = false;
}
if (x12855) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12774,x12776,x12776,1,x12844,1,1);
assert(false && "");
}
bool x12861 = x12774 <= x12844;
int32_t x12862;
if (x12861) {
x12862 = x12844;
} else {
x12862 = x12774;
}
int32_t x12871 = x12862 * x12870;
int32_t x12872 = 64 * x12871;
float* x12873 = (float*)myMalloc(x12872 * sizeof(float));;
int32_t x12874;
if (x12845) {
x12874 = 0;
} else {
x12874 = x12782;
}
int32_t x12877;
if (x12846) {
x12877 = 0;
} else {
x12877 = 1;
}
for(int x12878=0; x12878 < 64; x12878++) {
int32_t x12890 = x12783 * x12878;
int32_t x12884 = x12871 * x12878;
for(int x12880=0; x12880 < x12862; x12880++) {
int32_t x12891 = x12874 * x12880;
int32_t x12892 = x12890 + x12891;
int32_t x12897 = x12877 * x12880;
int32_t x12886 = x12870 * x12880;
for(int x12882=0; x12882 < x12864; x12882++) {
int32_t x12893 = x12875 * x12882;
int32_t x12894 = x12892 + x12893;
int32_t x12888 = x12864 * x12882;
for(int x12883=0; x12883 < x12864; x12883++) {
int32_t x12895 = x12876 * x12883;
int32_t x12896 = x12894 + x12895;
float x12898 = x12785[x12896];
float x12899 = x117[x12897];
int32_t x12885 = x12883 + x12884;
int32_t x12887 = x12885 + x12886;
int32_t x12889 = x12887 + x12888;
float x12900 = x12898 + x12899;
x12873[x12889] = x12900;

}

}

}

}
float* x12917 = (float*)myMalloc(x12916 * sizeof(float));;
int32_t x12920 = x11604 * x12912;
float* x12921 = (float*)myMalloc(x12920 * sizeof(float));;
int32_t x12918 = x11465 * x12912;
for(int x12922=0; x12922 < 64; x12922++) {
int32_t x12923 = x12922 * x11474;
float* x12924 = x11580+x12923;
int32_t x12925 = x12922 * x12913;
float* x12926 = x12917+x12925;
int32_t x12927 = x12922 * x12918;
float* x12928 = x12921+x12927;
for(int x12929=0; x12929 < x11465; x12929++) {
int32_t x12930 = x12929 / 1;
int32_t x12934 = x12930 * x12911;
int32_t x12935 = x12934 * x12911;
int32_t x12931 = x12929 % 1;
int32_t x12932 = x12931 / 1;
int32_t x12936 = x12932 * x12911;
int32_t x12937 = x12936 * x12911;
int32_t x12938 = x12935 + x12937;
int32_t x12933 = x12931 % 1;
int32_t x12939 = x12933 * x12911;
int32_t x12940 = x12939 * x12911;
int32_t x12941 = x12938 + x12940;
float* x12942 = x12928+x12941;
int32_t x12943 = x12930 * x11467;
int32_t x12944 = x12943 * x11467;
float* x12945 = x12924+x12944;
for(int x12947=0; x12947 < x12911; x12947++) {
int32_t x12951 = x12947 * x12911;
int32_t x12948 = x12947 * 2;
int32_t x12949 = x12948 + x12932;
int32_t x12954 = x12949 * x11467;
int32_t x12955 = x12954 + x12933;
for(int x12950=0; x12950 < x12911; x12950++) {
int32_t x12952 = x12951 + x12950;
float* x12953 = x12942+x12952;
int32_t x12956 = x12950 * 2;
int32_t x12957 = x12955 + x12956;
float* x12958 = x12945+x12957;
memcpy(x12953, x12958, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x12912,x11465,1,x75,x11465,x12928,x12912,1,x12926,x12912);

}
int32_t x12969 = 0;
int32_t x12970 = 1;
x12970 *= 1;
x12969 += 1;
x12970 *= 1;
x12970 *= 1;
int32_t x12975 = x12969;
bool x12976 = x12975 >= 2;
if (x12976) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12981 = x12975 == 0;
if (x12981) {
int32_t x12982 = x12970;
bool x12983 = x12982 == 1024;
if (x12983) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12990 = x12970;
int32_t x12991 = 1024 / x12990;
bool x12992 = x12991 == 1;
bool x12995;
if (x454) {
bool x12993 = 1024 == x12991;
bool x12994 = x12992 || x12993;
x12995 = x12994;
} else {
x12995 = false;
}
bool x12999;
if (x12995) {
x12999 = x12998;
} else {
x12999 = false;
}
bool x13000;
if (x12999) {
x13000 = x12998;
} else {
x13000 = false;
}
if (x13000) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x12911,x12911,1,x12991,1,1);
assert(false && "");
}
bool x13006 = 1024 <= x12991;
int32_t x13007;
if (x13006) {
x13007 = x12991;
} else {
x13007 = 1024;
}
int32_t x13016 = x13007 * x13015;
int32_t x13017 = 64 * x13016;
float* x13018 = (float*)myMalloc(x13017 * sizeof(float));;
int32_t x13021;
if (x12992) {
x13021 = 0;
} else {
x13021 = 1;
}
for(int x13022=0; x13022 < 64; x13022++) {
int32_t x13034 = x12913 * x13022;
int32_t x13028 = x13016 * x13022;
for(int x13024=0; x13024 < x13007; x13024++) {
int32_t x13035 = x12912 * x13024;
int32_t x13036 = x13034 + x13035;
int32_t x13041 = x13021 * x13024;
int32_t x13030 = x13015 * x13024;
for(int x13026=0; x13026 < x13009; x13026++) {
int32_t x13037 = x13019 * x13026;
int32_t x13038 = x13036 + x13037;
int32_t x13032 = x13009 * x13026;
for(int x13027=0; x13027 < x13009; x13027++) {
int32_t x13039 = x13020 * x13027;
int32_t x13040 = x13038 + x13039;
float x13042 = x12917[x13040];
float x13043 = x86[x13041];
int32_t x13029 = x13027 + x13028;
int32_t x13031 = x13029 + x13030;
int32_t x13033 = x13031 + x13032;
float x13044 = x13042 - x13043;
x13018[x13033] = x13044;

}

}

}

}
float* x13054 = (float*)myMalloc(1024 * sizeof(float));;
for(int x13055=0; x13055 < 1024; x13055++) {
float x13056 = x211[x13055];
float x13057 = x13056 + 1.0E-5f;
x13054[x13055] = x13057;

}
float* x13061 = (float*)myMalloc(1024 * sizeof(float));;
for(int x13062=0; x13062 < 1024; x13062++) {
float x13063 = x13054[x13062];
double x13064 = (double)x13063;
double x13065 = sqrt(x13064);
float x13066 = (float)x13065;
x13061[x13062] = x13066;

}
int32_t x13070 = 0;
int32_t x13071 = 1;
x13071 *= 1;
x13070 += 1;
x13071 *= 1;
x13071 *= 1;
int32_t x13076 = x13070;
bool x13077 = x13076 >= 2;
if (x13077) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13082 = x13076 == 0;
if (x13082) {
int32_t x13083 = x13071;
bool x13084 = x13083 == 1024;
if (x13084) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13091 = x13071;
bool x13093 = x13007 == 1;
int32_t x13092 = 1024 / x13091;
bool x13094 = x13092 == 1;
bool x13098;
if (x454) {
bool x13095 = x13093 || x13094;
bool x13096 = x13007 == x13092;
bool x13097 = x13095 || x13096;
x13098 = x13097;
} else {
x13098 = false;
}
bool x13102;
if (x13098) {
x13102 = x13101;
} else {
x13102 = false;
}
bool x13103;
if (x13102) {
x13103 = x13101;
} else {
x13103 = false;
}
if (x13103) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13007,x13009,x13009,1,x13092,1,1);
assert(false && "");
}
bool x13109 = x13007 <= x13092;
int32_t x13110;
if (x13109) {
x13110 = x13092;
} else {
x13110 = x13007;
}
int32_t x13119 = x13110 * x13118;
int32_t x13120 = 64 * x13119;
float* x13121 = (float*)myMalloc(x13120 * sizeof(float));;
int32_t x13122;
if (x13093) {
x13122 = 0;
} else {
x13122 = x13015;
}
int32_t x13125;
if (x13094) {
x13125 = 0;
} else {
x13125 = 1;
}
for(int x13126=0; x13126 < 64; x13126++) {
int32_t x13138 = x13016 * x13126;
int32_t x13132 = x13119 * x13126;
for(int x13128=0; x13128 < x13110; x13128++) {
int32_t x13139 = x13122 * x13128;
int32_t x13140 = x13138 + x13139;
int32_t x13145 = x13125 * x13128;
int32_t x13134 = x13118 * x13128;
for(int x13130=0; x13130 < x13112; x13130++) {
int32_t x13141 = x13123 * x13130;
int32_t x13142 = x13140 + x13141;
int32_t x13136 = x13112 * x13130;
for(int x13131=0; x13131 < x13112; x13131++) {
int32_t x13143 = x13124 * x13131;
int32_t x13144 = x13142 + x13143;
float x13146 = x13018[x13144];
float x13147 = x13061[x13145];
int32_t x13133 = x13131 + x13132;
int32_t x13135 = x13133 + x13134;
int32_t x13137 = x13135 + x13136;
float x13148 = x13146 / x13147;
x13121[x13137] = x13148;

}

}

}

}
int32_t x13158 = 0;
int32_t x13159 = 1;
x13159 *= 1;
x13158 += 1;
x13159 *= 1;
x13159 *= 1;
int32_t x13164 = x13158;
bool x13165 = x13164 >= 2;
if (x13165) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13170 = x13164 == 0;
if (x13170) {
int32_t x13171 = x13159;
bool x13172 = x13171 == 1024;
if (x13172) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13179 = x13159;
bool x13181 = x13110 == 1;
int32_t x13180 = 1024 / x13179;
bool x13182 = x13180 == 1;
bool x13186;
if (x454) {
bool x13183 = x13181 || x13182;
bool x13184 = x13110 == x13180;
bool x13185 = x13183 || x13184;
x13186 = x13185;
} else {
x13186 = false;
}
bool x13190;
if (x13186) {
x13190 = x13189;
} else {
x13190 = false;
}
bool x13191;
if (x13190) {
x13191 = x13189;
} else {
x13191 = false;
}
if (x13191) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13110,x13112,x13112,1,x13180,1,1);
assert(false && "");
}
bool x13197 = x13110 <= x13180;
int32_t x13198;
if (x13197) {
x13198 = x13180;
} else {
x13198 = x13110;
}
int32_t x13207 = x13198 * x13206;
int32_t x13208 = 64 * x13207;
float* x13209 = (float*)myMalloc(x13208 * sizeof(float));;
int32_t x13210;
if (x13181) {
x13210 = 0;
} else {
x13210 = x13118;
}
int32_t x13213;
if (x13182) {
x13213 = 0;
} else {
x13213 = 1;
}
for(int x13214=0; x13214 < 64; x13214++) {
int32_t x13226 = x13119 * x13214;
int32_t x13220 = x13207 * x13214;
for(int x13216=0; x13216 < x13198; x13216++) {
int32_t x13227 = x13210 * x13216;
int32_t x13228 = x13226 + x13227;
int32_t x13233 = x13213 * x13216;
int32_t x13222 = x13206 * x13216;
for(int x13218=0; x13218 < x13200; x13218++) {
int32_t x13229 = x13211 * x13218;
int32_t x13230 = x13228 + x13229;
int32_t x13224 = x13200 * x13218;
for(int x13219=0; x13219 < x13200; x13219++) {
int32_t x13231 = x13212 * x13219;
int32_t x13232 = x13230 + x13231;
float x13234 = x13121[x13232];
float x13235 = x29[x13233];
int32_t x13221 = x13219 + x13220;
int32_t x13223 = x13221 + x13222;
int32_t x13225 = x13223 + x13224;
float x13236 = x13234 * x13235;
x13209[x13225] = x13236;

}

}

}

}
int32_t x13246 = 0;
int32_t x13247 = 1;
x13247 *= 1;
x13246 += 1;
x13247 *= 1;
x13247 *= 1;
int32_t x13252 = x13246;
bool x13253 = x13252 >= 2;
if (x13253) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13258 = x13252 == 0;
if (x13258) {
int32_t x13259 = x13247;
bool x13260 = x13259 == 1024;
if (x13260) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13267 = x13247;
bool x13269 = x13198 == 1;
int32_t x13268 = 1024 / x13267;
bool x13270 = x13268 == 1;
bool x13274;
if (x454) {
bool x13271 = x13269 || x13270;
bool x13272 = x13198 == x13268;
bool x13273 = x13271 || x13272;
x13274 = x13273;
} else {
x13274 = false;
}
bool x13278;
if (x13274) {
x13278 = x13277;
} else {
x13278 = false;
}
bool x13279;
if (x13278) {
x13279 = x13277;
} else {
x13279 = false;
}
if (x13279) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13198,x13200,x13200,1,x13268,1,1);
assert(false && "");
}
bool x13285 = x13198 <= x13268;
int32_t x13286;
if (x13285) {
x13286 = x13268;
} else {
x13286 = x13198;
}
int32_t x13295 = x13286 * x13294;
int32_t x13296 = 64 * x13295;
float* x13297 = (float*)myMalloc(x13296 * sizeof(float));;
int32_t x13298;
if (x13269) {
x13298 = 0;
} else {
x13298 = x13206;
}
int32_t x13301;
if (x13270) {
x13301 = 0;
} else {
x13301 = 1;
}
for(int x13302=0; x13302 < 64; x13302++) {
int32_t x13314 = x13207 * x13302;
int32_t x13308 = x13295 * x13302;
for(int x13304=0; x13304 < x13286; x13304++) {
int32_t x13315 = x13298 * x13304;
int32_t x13316 = x13314 + x13315;
int32_t x13321 = x13301 * x13304;
int32_t x13310 = x13294 * x13304;
for(int x13306=0; x13306 < x13288; x13306++) {
int32_t x13317 = x13299 * x13306;
int32_t x13318 = x13316 + x13317;
int32_t x13312 = x13288 * x13306;
for(int x13307=0; x13307 < x13288; x13307++) {
int32_t x13319 = x13300 * x13307;
int32_t x13320 = x13318 + x13319;
float x13322 = x13209[x13320];
float x13323 = x220[x13321];
int32_t x13309 = x13307 + x13308;
int32_t x13311 = x13309 + x13310;
int32_t x13313 = x13311 + x13312;
float x13324 = x13322 + x13323;
x13297[x13313] = x13324;

}

}

}

}
bool x13334 = x12862 == 1;
bool x13335 = x13286 == 1;
bool x13336 = x13334 || x13335;
bool x13337 = x12862 == x13286;
bool x13338 = x13336 || x13337;
bool x13344;
if (x13338) {
x13344 = x13343;
} else {
x13344 = false;
}
bool x13345;
if (x13344) {
x13345 = x13343;
} else {
x13345 = false;
}
if (x13345) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12862,x12864,x12864,64,x13286,x13288,x13288);
assert(false && "");
}
bool x13351 = x12862 <= x13286;
int32_t x13352;
if (x13351) {
x13352 = x13286;
} else {
x13352 = x12862;
}
int32_t x13368;
if (x13334) {
x13368 = 0;
} else {
x13368 = x12870;
}
int32_t x13371;
if (x13335) {
x13371 = 0;
} else {
x13371 = x13294;
}
for(int x13374=0; x13374 < 64; x13374++) {
int32_t x13380 = x12871 * x13374;
int32_t x13387 = x13295 * x13374;
for(int x13376=0; x13376 < x13352; x13376++) {
int32_t x13381 = x13368 * x13376;
int32_t x13382 = x13380 + x13381;
int32_t x13388 = x13371 * x13376;
int32_t x13389 = x13387 + x13388;
for(int x13378=0; x13378 < x13354; x13378++) {
int32_t x13383 = x13369 * x13378;
int32_t x13384 = x13382 + x13383;
int32_t x13390 = x13372 * x13378;
int32_t x13391 = x13389 + x13390;
for(int x13379=0; x13379 < x13354; x13379++) {
int32_t x13385 = x13370 * x13379;
int32_t x13386 = x13384 + x13385;
float x13394 = x12873[x13386];
int32_t x13392 = x13373 * x13379;
int32_t x13393 = x13391 + x13392;
float x13395 = x13297[x13393];
float x13396 = x13394 + x13395;
x12873[x13386] = x13396;

}

}

}

}
float* x13406 = (float*)myMalloc(x12872 * sizeof(float));;
for(int x13408=0; x13408 < x12872; x13408++) {
float x13409 = x12873[x13408];
bool x13410 = x13409 < 0.0f;
if (x13410) {
x13406[x13408] = 0.0f;
} else {
float x13413 = x12873[x13408];
x13406[x13408] = x13413;
}

}
float* x13427 = (float*)myMalloc(x13426 * sizeof(float));;
int32_t x13430 = 64 * x12862;
int32_t x13431 = x13430 * x13422;
float* x13432 = (float*)myMalloc(x13431 * sizeof(float));;
int32_t x13428 = x12862 * x13422;
for(int x13433=0; x13433 < 64; x13433++) {
int32_t x13434 = x13433 * x12871;
float* x13435 = x13406+x13434;
int32_t x13436 = x13433 * x13423;
float* x13437 = x13427+x13436;
int32_t x13438 = x13433 * x13428;
float* x13439 = x13432+x13438;
for(int x13440=0; x13440 < x12862; x13440++) {
int32_t x13441 = x13440 / 1;
int32_t x13445 = x13441 * x13421;
int32_t x13446 = x13445 * x13421;
int32_t x13442 = x13440 % 1;
int32_t x13443 = x13442 / 1;
int32_t x13447 = x13443 * x13421;
int32_t x13448 = x13447 * x13421;
int32_t x13449 = x13446 + x13448;
int32_t x13444 = x13442 % 1;
int32_t x13450 = x13444 * x13421;
int32_t x13451 = x13450 * x13421;
int32_t x13452 = x13449 + x13451;
float* x13453 = x13439+x13452;
int32_t x13454 = x13441 * x12864;
int32_t x13455 = x13454 * x12864;
float* x13456 = x13435+x13455;
for(int x13458=0; x13458 < x13421; x13458++) {
int32_t x13460 = x13458 * x13421;
float* x13461 = x13453+x13460;
int32_t x13459 = x13458 + x13443;
int32_t x13462 = x13459 * x12864;
int32_t x13463 = x13462 + x13444;
float* x13464 = x13456+x13463;
memcpy(x13461, x13464, 4 * x13421);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x13422,x12862,1,x13,x12862,x13439,x13422,1,x13437,x13422);

}
int32_t x13473 = 0;
int32_t x13474 = 1;
x13474 *= 1;
x13473 += 1;
x13474 *= 1;
x13474 *= 1;
int32_t x13479 = x13473;
bool x13480 = x13479 >= 2;
if (x13480) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13485 = x13479 == 0;
if (x13485) {
int32_t x13486 = x13474;
bool x13487 = x13486 == 256;
if (x13487) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13494 = x13474;
int32_t x13495 = 256 / x13494;
bool x13496 = x13495 == 1;
bool x13499;
if (x454) {
bool x13497 = 256 == x13495;
bool x13498 = x13496 || x13497;
x13499 = x13498;
} else {
x13499 = false;
}
bool x13503;
if (x13499) {
x13503 = x13502;
} else {
x13503 = false;
}
bool x13504;
if (x13503) {
x13504 = x13502;
} else {
x13504 = false;
}
if (x13504) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x13421,x13421,1,x13495,1,1);
assert(false && "");
}
bool x13510 = 256 <= x13495;
int32_t x13511;
if (x13510) {
x13511 = x13495;
} else {
x13511 = 256;
}
int32_t x13520 = x13511 * x13519;
int32_t x13521 = 64 * x13520;
float* x13522 = (float*)myMalloc(x13521 * sizeof(float));;
int32_t x13525;
if (x13496) {
x13525 = 0;
} else {
x13525 = 1;
}
for(int x13526=0; x13526 < 64; x13526++) {
int32_t x13538 = x13423 * x13526;
int32_t x13532 = x13520 * x13526;
for(int x13528=0; x13528 < x13511; x13528++) {
int32_t x13539 = x13422 * x13528;
int32_t x13540 = x13538 + x13539;
int32_t x13545 = x13525 * x13528;
int32_t x13534 = x13519 * x13528;
for(int x13530=0; x13530 < x13513; x13530++) {
int32_t x13541 = x13523 * x13530;
int32_t x13542 = x13540 + x13541;
int32_t x13536 = x13513 * x13530;
for(int x13531=0; x13531 < x13513; x13531++) {
int32_t x13543 = x13524 * x13531;
int32_t x13544 = x13542 + x13543;
float x13546 = x13427[x13544];
float x13547 = x259[x13545];
int32_t x13533 = x13531 + x13532;
int32_t x13535 = x13533 + x13534;
int32_t x13537 = x13535 + x13536;
float x13548 = x13546 - x13547;
x13522[x13537] = x13548;

}

}

}

}
float* x13558 = (float*)myMalloc(256 * sizeof(float));;
for(int x13559=0; x13559 < 256; x13559++) {
float x13560 = x157[x13559];
float x13561 = x13560 + 1.0E-5f;
x13558[x13559] = x13561;

}
float* x13565 = (float*)myMalloc(256 * sizeof(float));;
for(int x13566=0; x13566 < 256; x13566++) {
float x13567 = x13558[x13566];
double x13568 = (double)x13567;
double x13569 = sqrt(x13568);
float x13570 = (float)x13569;
x13565[x13566] = x13570;

}
int32_t x13574 = 0;
int32_t x13575 = 1;
x13575 *= 1;
x13574 += 1;
x13575 *= 1;
x13575 *= 1;
int32_t x13580 = x13574;
bool x13581 = x13580 >= 2;
if (x13581) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13586 = x13580 == 0;
if (x13586) {
int32_t x13587 = x13575;
bool x13588 = x13587 == 256;
if (x13588) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13595 = x13575;
bool x13597 = x13511 == 1;
int32_t x13596 = 256 / x13595;
bool x13598 = x13596 == 1;
bool x13602;
if (x454) {
bool x13599 = x13597 || x13598;
bool x13600 = x13511 == x13596;
bool x13601 = x13599 || x13600;
x13602 = x13601;
} else {
x13602 = false;
}
bool x13606;
if (x13602) {
x13606 = x13605;
} else {
x13606 = false;
}
bool x13607;
if (x13606) {
x13607 = x13605;
} else {
x13607 = false;
}
if (x13607) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13511,x13513,x13513,1,x13596,1,1);
assert(false && "");
}
bool x13613 = x13511 <= x13596;
int32_t x13614;
if (x13613) {
x13614 = x13596;
} else {
x13614 = x13511;
}
int32_t x13623 = x13614 * x13622;
int32_t x13624 = 64 * x13623;
float* x13625 = (float*)myMalloc(x13624 * sizeof(float));;
int32_t x13626;
if (x13597) {
x13626 = 0;
} else {
x13626 = x13519;
}
int32_t x13629;
if (x13598) {
x13629 = 0;
} else {
x13629 = 1;
}
for(int x13630=0; x13630 < 64; x13630++) {
int32_t x13642 = x13520 * x13630;
int32_t x13636 = x13623 * x13630;
for(int x13632=0; x13632 < x13614; x13632++) {
int32_t x13643 = x13626 * x13632;
int32_t x13644 = x13642 + x13643;
int32_t x13649 = x13629 * x13632;
int32_t x13638 = x13622 * x13632;
for(int x13634=0; x13634 < x13616; x13634++) {
int32_t x13645 = x13627 * x13634;
int32_t x13646 = x13644 + x13645;
int32_t x13640 = x13616 * x13634;
for(int x13635=0; x13635 < x13616; x13635++) {
int32_t x13647 = x13628 * x13635;
int32_t x13648 = x13646 + x13647;
float x13650 = x13522[x13648];
float x13651 = x13565[x13649];
int32_t x13637 = x13635 + x13636;
int32_t x13639 = x13637 + x13638;
int32_t x13641 = x13639 + x13640;
float x13652 = x13650 / x13651;
x13625[x13641] = x13652;

}

}

}

}
int32_t x13662 = 0;
int32_t x13663 = 1;
x13663 *= 1;
x13662 += 1;
x13663 *= 1;
x13663 *= 1;
int32_t x13668 = x13662;
bool x13669 = x13668 >= 2;
if (x13669) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13674 = x13668 == 0;
if (x13674) {
int32_t x13675 = x13663;
bool x13676 = x13675 == 256;
if (x13676) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13683 = x13663;
bool x13685 = x13614 == 1;
int32_t x13684 = 256 / x13683;
bool x13686 = x13684 == 1;
bool x13690;
if (x454) {
bool x13687 = x13685 || x13686;
bool x13688 = x13614 == x13684;
bool x13689 = x13687 || x13688;
x13690 = x13689;
} else {
x13690 = false;
}
bool x13694;
if (x13690) {
x13694 = x13693;
} else {
x13694 = false;
}
bool x13695;
if (x13694) {
x13695 = x13693;
} else {
x13695 = false;
}
if (x13695) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13614,x13616,x13616,1,x13684,1,1);
assert(false && "");
}
bool x13701 = x13614 <= x13684;
int32_t x13702;
if (x13701) {
x13702 = x13684;
} else {
x13702 = x13614;
}
int32_t x13711 = x13702 * x13710;
int32_t x13712 = 64 * x13711;
float* x13713 = (float*)myMalloc(x13712 * sizeof(float));;
int32_t x13714;
if (x13685) {
x13714 = 0;
} else {
x13714 = x13622;
}
int32_t x13717;
if (x13686) {
x13717 = 0;
} else {
x13717 = 1;
}
for(int x13718=0; x13718 < 64; x13718++) {
int32_t x13730 = x13623 * x13718;
int32_t x13724 = x13711 * x13718;
for(int x13720=0; x13720 < x13702; x13720++) {
int32_t x13731 = x13714 * x13720;
int32_t x13732 = x13730 + x13731;
int32_t x13737 = x13717 * x13720;
int32_t x13726 = x13710 * x13720;
for(int x13722=0; x13722 < x13704; x13722++) {
int32_t x13733 = x13715 * x13722;
int32_t x13734 = x13732 + x13733;
int32_t x13728 = x13704 * x13722;
for(int x13723=0; x13723 < x13704; x13723++) {
int32_t x13735 = x13716 * x13723;
int32_t x13736 = x13734 + x13735;
float x13738 = x13625[x13736];
float x13739 = x30[x13737];
int32_t x13725 = x13723 + x13724;
int32_t x13727 = x13725 + x13726;
int32_t x13729 = x13727 + x13728;
float x13740 = x13738 * x13739;
x13713[x13729] = x13740;

}

}

}

}
int32_t x13750 = 0;
int32_t x13751 = 1;
x13751 *= 1;
x13750 += 1;
x13751 *= 1;
x13751 *= 1;
int32_t x13756 = x13750;
bool x13757 = x13756 >= 2;
if (x13757) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13762 = x13756 == 0;
if (x13762) {
int32_t x13763 = x13751;
bool x13764 = x13763 == 256;
if (x13764) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13771 = x13751;
bool x13773 = x13702 == 1;
int32_t x13772 = 256 / x13771;
bool x13774 = x13772 == 1;
bool x13778;
if (x454) {
bool x13775 = x13773 || x13774;
bool x13776 = x13702 == x13772;
bool x13777 = x13775 || x13776;
x13778 = x13777;
} else {
x13778 = false;
}
bool x13782;
if (x13778) {
x13782 = x13781;
} else {
x13782 = false;
}
bool x13783;
if (x13782) {
x13783 = x13781;
} else {
x13783 = false;
}
if (x13783) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13702,x13704,x13704,1,x13772,1,1);
assert(false && "");
}
bool x13789 = x13702 <= x13772;
int32_t x13790;
if (x13789) {
x13790 = x13772;
} else {
x13790 = x13702;
}
int32_t x13799 = x13790 * x13798;
int32_t x13800 = 64 * x13799;
float* x13801 = (float*)myMalloc(x13800 * sizeof(float));;
int32_t x13802;
if (x13773) {
x13802 = 0;
} else {
x13802 = x13710;
}
int32_t x13805;
if (x13774) {
x13805 = 0;
} else {
x13805 = 1;
}
for(int x13806=0; x13806 < 64; x13806++) {
int32_t x13818 = x13711 * x13806;
int32_t x13812 = x13799 * x13806;
for(int x13808=0; x13808 < x13790; x13808++) {
int32_t x13819 = x13802 * x13808;
int32_t x13820 = x13818 + x13819;
int32_t x13825 = x13805 * x13808;
int32_t x13814 = x13798 * x13808;
for(int x13810=0; x13810 < x13792; x13810++) {
int32_t x13821 = x13803 * x13810;
int32_t x13822 = x13820 + x13821;
int32_t x13816 = x13792 * x13810;
for(int x13811=0; x13811 < x13792; x13811++) {
int32_t x13823 = x13804 * x13811;
int32_t x13824 = x13822 + x13823;
float x13826 = x13713[x13824];
float x13827 = x219[x13825];
int32_t x13813 = x13811 + x13812;
int32_t x13815 = x13813 + x13814;
int32_t x13817 = x13815 + x13816;
float x13828 = x13826 + x13827;
x13801[x13817] = x13828;

}

}

}

}
float* x13838 = (float*)myMalloc(x13800 * sizeof(float));;
for(int x13840=0; x13840 < x13800; x13840++) {
float x13841 = x13801[x13840];
bool x13842 = x13841 < 0.0f;
if (x13842) {
x13838[x13840] = 0.0f;
} else {
float x13845 = x13801[x13840];
x13838[x13840] = x13845;
}

}
float* x13860 = (float*)myMalloc(x13859 * sizeof(float));;
int32_t x13861 = 9 * x13790;
int32_t x13864 = 64 * x13861;
int32_t x13865 = x13864 * x13855;
float* x13866 = (float*)myMalloc(x13865 * sizeof(float));;
int32_t x13862 = x13861 * x13855;
int32_t x13874 = x13790 * 3;
int32_t x13875 = x13874 * 3;
for(int x13867=0; x13867 < 64; x13867++) {
int32_t x13868 = x13867 * x13799;
float* x13869 = x13838+x13868;
int32_t x13870 = x13867 * x13856;
float* x13871 = x13860+x13870;
int32_t x13872 = x13867 * x13862;
float* x13873 = x13866+x13872;
for(int x13877=0; x13877 < x13875; x13877++) {
int32_t x13878 = x13877 / 9;
int32_t x13882 = x13878 * 3;
int32_t x13883 = x13882 * 3;
int32_t x13884 = x13883 * x13854;
int32_t x13885 = x13884 * x13854;
int32_t x13879 = x13877 % 9;
int32_t x13880 = x13879 / 3;
int32_t x13886 = x13880 * 3;
int32_t x13887 = x13886 * x13854;
int32_t x13888 = x13887 * x13854;
int32_t x13889 = x13885 + x13888;
int32_t x13881 = x13879 % 3;
int32_t x13890 = x13881 * x13854;
int32_t x13891 = x13890 * x13854;
int32_t x13892 = x13889 + x13891;
float* x13893 = x13873+x13892;
int32_t x13894 = x13878 * x13792;
int32_t x13895 = x13894 * x13792;
float* x13896 = x13869+x13895;
int32_t x13909 = 1 - x13881;
bool x13910 = x13909 > 0;
int32_t x13911;
if (x13910) {
x13911 = x13909;
} else {
x13911 = 0;
}
int32_t x13912 = 3 - x13881;
int32_t x13913 = x13912 - 1;
int32_t x13914 = 1 - x13913;
bool x13915 = x13914 > 0;
int32_t x13916;
if (x13915) {
x13916 = x13914;
} else {
x13916 = 0;
}
int32_t x13917 = x13854 - x13916;
int32_t x13918 = x13917 - x13911;
bool x13919 = x13918 <= 0;
bool x13923 = x13911 > 0;
int32_t x13908 = -1 + x13881;
bool x13936 = x13916 > 0;
for(int x13898=0; x13898 < x13854; x13898++) {
int32_t x13899 = x13898 - 1;
int32_t x13900 = x13899 + x13880;
bool x13901 = x13900 < 0;
bool x13902 = x13900 >= x13792;
bool x13903 = x13901 || x13902;
if (x13903) {
int32_t x13904 = x13898 * x13854;
float* x13905 = x13893+x13904;
memset(x13905, 0, 4 * x13854);;
} else {
if (x13919) {
int32_t x13904 = x13898 * x13854;
float* x13920 = x13893+x13904;
memset(x13920, 0, 4 * x13854);;
} else {
int32_t x13904 = x13898 * x13854;
if (x13923) {
float* x13924 = x13893+x13904;
memset(x13924, 0, 4 * x13911);;
} else {
}
// may have segfault here
int32_t x13929 = x13904 + x13911;
float* x13930 = x13893+x13929;
int32_t x13931 = x13900 * x13792;
int32_t x13932 = x13931 + x13908;
int32_t x13933 = x13932 + x13911;
float* x13934 = x13896+x13933;
memcpy(x13930, x13934, 4 * x13918);;
if (x13936) {
int32_t x13937 = x13904 + x13854;
int32_t x13938 = x13937 - x13916;
float* x13939 = x13893+x13938;
memset(x13939, 0, 4 * x13916);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x13855,x13861,1,x31,x13861,x13873,x13855,1,x13871,x13855);

}
int32_t x13954 = 0;
int32_t x13955 = 1;
x13955 *= 1;
x13954 += 1;
x13955 *= 1;
x13955 *= 1;
int32_t x13960 = x13954;
bool x13961 = x13960 >= 2;
if (x13961) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13966 = x13960 == 0;
if (x13966) {
int32_t x13967 = x13955;
bool x13968 = x13967 == 256;
if (x13968) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13975 = x13955;
int32_t x13976 = 256 / x13975;
bool x13977 = x13976 == 1;
bool x13980;
if (x454) {
bool x13978 = 256 == x13976;
bool x13979 = x13977 || x13978;
x13980 = x13979;
} else {
x13980 = false;
}
bool x13984;
if (x13980) {
x13984 = x13983;
} else {
x13984 = false;
}
bool x13985;
if (x13984) {
x13985 = x13983;
} else {
x13985 = false;
}
if (x13985) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x13854,x13854,1,x13976,1,1);
assert(false && "");
}
bool x13991 = 256 <= x13976;
int32_t x13992;
if (x13991) {
x13992 = x13976;
} else {
x13992 = 256;
}
int32_t x14001 = x13992 * x14000;
int32_t x14002 = 64 * x14001;
float* x14003 = (float*)myMalloc(x14002 * sizeof(float));;
int32_t x14006;
if (x13977) {
x14006 = 0;
} else {
x14006 = 1;
}
for(int x14007=0; x14007 < 64; x14007++) {
int32_t x14019 = x13856 * x14007;
int32_t x14013 = x14001 * x14007;
for(int x14009=0; x14009 < x13992; x14009++) {
int32_t x14020 = x13855 * x14009;
int32_t x14021 = x14019 + x14020;
int32_t x14026 = x14006 * x14009;
int32_t x14015 = x14000 * x14009;
for(int x14011=0; x14011 < x13994; x14011++) {
int32_t x14022 = x14004 * x14011;
int32_t x14023 = x14021 + x14022;
int32_t x14017 = x13994 * x14011;
for(int x14012=0; x14012 < x13994; x14012++) {
int32_t x14024 = x14005 * x14012;
int32_t x14025 = x14023 + x14024;
float x14027 = x13860[x14025];
float x14028 = x200[x14026];
int32_t x14014 = x14012 + x14013;
int32_t x14016 = x14014 + x14015;
int32_t x14018 = x14016 + x14017;
float x14029 = x14027 - x14028;
x14003[x14018] = x14029;

}

}

}

}
float* x14039 = (float*)myMalloc(256 * sizeof(float));;
for(int x14040=0; x14040 < 256; x14040++) {
float x14041 = x237[x14040];
float x14042 = x14041 + 1.0E-5f;
x14039[x14040] = x14042;

}
float* x14046 = (float*)myMalloc(256 * sizeof(float));;
for(int x14047=0; x14047 < 256; x14047++) {
float x14048 = x14039[x14047];
double x14049 = (double)x14048;
double x14050 = sqrt(x14049);
float x14051 = (float)x14050;
x14046[x14047] = x14051;

}
int32_t x14055 = 0;
int32_t x14056 = 1;
x14056 *= 1;
x14055 += 1;
x14056 *= 1;
x14056 *= 1;
int32_t x14061 = x14055;
bool x14062 = x14061 >= 2;
if (x14062) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14067 = x14061 == 0;
if (x14067) {
int32_t x14068 = x14056;
bool x14069 = x14068 == 256;
if (x14069) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14076 = x14056;
bool x14078 = x13992 == 1;
int32_t x14077 = 256 / x14076;
bool x14079 = x14077 == 1;
bool x14083;
if (x454) {
bool x14080 = x14078 || x14079;
bool x14081 = x13992 == x14077;
bool x14082 = x14080 || x14081;
x14083 = x14082;
} else {
x14083 = false;
}
bool x14087;
if (x14083) {
x14087 = x14086;
} else {
x14087 = false;
}
bool x14088;
if (x14087) {
x14088 = x14086;
} else {
x14088 = false;
}
if (x14088) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13992,x13994,x13994,1,x14077,1,1);
assert(false && "");
}
bool x14094 = x13992 <= x14077;
int32_t x14095;
if (x14094) {
x14095 = x14077;
} else {
x14095 = x13992;
}
int32_t x14104 = x14095 * x14103;
int32_t x14105 = 64 * x14104;
float* x14106 = (float*)myMalloc(x14105 * sizeof(float));;
int32_t x14107;
if (x14078) {
x14107 = 0;
} else {
x14107 = x14000;
}
int32_t x14110;
if (x14079) {
x14110 = 0;
} else {
x14110 = 1;
}
for(int x14111=0; x14111 < 64; x14111++) {
int32_t x14123 = x14001 * x14111;
int32_t x14117 = x14104 * x14111;
for(int x14113=0; x14113 < x14095; x14113++) {
int32_t x14124 = x14107 * x14113;
int32_t x14125 = x14123 + x14124;
int32_t x14130 = x14110 * x14113;
int32_t x14119 = x14103 * x14113;
for(int x14115=0; x14115 < x14097; x14115++) {
int32_t x14126 = x14108 * x14115;
int32_t x14127 = x14125 + x14126;
int32_t x14121 = x14097 * x14115;
for(int x14116=0; x14116 < x14097; x14116++) {
int32_t x14128 = x14109 * x14116;
int32_t x14129 = x14127 + x14128;
float x14131 = x14003[x14129];
float x14132 = x14046[x14130];
int32_t x14118 = x14116 + x14117;
int32_t x14120 = x14118 + x14119;
int32_t x14122 = x14120 + x14121;
float x14133 = x14131 / x14132;
x14106[x14122] = x14133;

}

}

}

}
int32_t x14143 = 0;
int32_t x14144 = 1;
x14144 *= 1;
x14143 += 1;
x14144 *= 1;
x14144 *= 1;
int32_t x14149 = x14143;
bool x14150 = x14149 >= 2;
if (x14150) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14155 = x14149 == 0;
if (x14155) {
int32_t x14156 = x14144;
bool x14157 = x14156 == 256;
if (x14157) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14164 = x14144;
bool x14166 = x14095 == 1;
int32_t x14165 = 256 / x14164;
bool x14167 = x14165 == 1;
bool x14171;
if (x454) {
bool x14168 = x14166 || x14167;
bool x14169 = x14095 == x14165;
bool x14170 = x14168 || x14169;
x14171 = x14170;
} else {
x14171 = false;
}
bool x14175;
if (x14171) {
x14175 = x14174;
} else {
x14175 = false;
}
bool x14176;
if (x14175) {
x14176 = x14174;
} else {
x14176 = false;
}
if (x14176) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14095,x14097,x14097,1,x14165,1,1);
assert(false && "");
}
bool x14182 = x14095 <= x14165;
int32_t x14183;
if (x14182) {
x14183 = x14165;
} else {
x14183 = x14095;
}
int32_t x14192 = x14183 * x14191;
int32_t x14193 = 64 * x14192;
float* x14194 = (float*)myMalloc(x14193 * sizeof(float));;
int32_t x14195;
if (x14166) {
x14195 = 0;
} else {
x14195 = x14103;
}
int32_t x14198;
if (x14167) {
x14198 = 0;
} else {
x14198 = 1;
}
for(int x14199=0; x14199 < 64; x14199++) {
int32_t x14211 = x14104 * x14199;
int32_t x14205 = x14192 * x14199;
for(int x14201=0; x14201 < x14183; x14201++) {
int32_t x14212 = x14195 * x14201;
int32_t x14213 = x14211 + x14212;
int32_t x14218 = x14198 * x14201;
int32_t x14207 = x14191 * x14201;
for(int x14203=0; x14203 < x14185; x14203++) {
int32_t x14214 = x14196 * x14203;
int32_t x14215 = x14213 + x14214;
int32_t x14209 = x14185 * x14203;
for(int x14204=0; x14204 < x14185; x14204++) {
int32_t x14216 = x14197 * x14204;
int32_t x14217 = x14215 + x14216;
float x14219 = x14106[x14217];
float x14220 = x271[x14218];
int32_t x14206 = x14204 + x14205;
int32_t x14208 = x14206 + x14207;
int32_t x14210 = x14208 + x14209;
float x14221 = x14219 * x14220;
x14194[x14210] = x14221;

}

}

}

}
int32_t x14231 = 0;
int32_t x14232 = 1;
x14232 *= 1;
x14231 += 1;
x14232 *= 1;
x14232 *= 1;
int32_t x14237 = x14231;
bool x14238 = x14237 >= 2;
if (x14238) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14243 = x14237 == 0;
if (x14243) {
int32_t x14244 = x14232;
bool x14245 = x14244 == 256;
if (x14245) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14252 = x14232;
bool x14254 = x14183 == 1;
int32_t x14253 = 256 / x14252;
bool x14255 = x14253 == 1;
bool x14259;
if (x454) {
bool x14256 = x14254 || x14255;
bool x14257 = x14183 == x14253;
bool x14258 = x14256 || x14257;
x14259 = x14258;
} else {
x14259 = false;
}
bool x14263;
if (x14259) {
x14263 = x14262;
} else {
x14263 = false;
}
bool x14264;
if (x14263) {
x14264 = x14262;
} else {
x14264 = false;
}
if (x14264) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14183,x14185,x14185,1,x14253,1,1);
assert(false && "");
}
bool x14270 = x14183 <= x14253;
int32_t x14271;
if (x14270) {
x14271 = x14253;
} else {
x14271 = x14183;
}
int32_t x14280 = x14271 * x14279;
int32_t x14281 = 64 * x14280;
float* x14282 = (float*)myMalloc(x14281 * sizeof(float));;
int32_t x14283;
if (x14254) {
x14283 = 0;
} else {
x14283 = x14191;
}
int32_t x14286;
if (x14255) {
x14286 = 0;
} else {
x14286 = 1;
}
for(int x14287=0; x14287 < 64; x14287++) {
int32_t x14299 = x14192 * x14287;
int32_t x14293 = x14280 * x14287;
for(int x14289=0; x14289 < x14271; x14289++) {
int32_t x14300 = x14283 * x14289;
int32_t x14301 = x14299 + x14300;
int32_t x14306 = x14286 * x14289;
int32_t x14295 = x14279 * x14289;
for(int x14291=0; x14291 < x14273; x14291++) {
int32_t x14302 = x14284 * x14291;
int32_t x14303 = x14301 + x14302;
int32_t x14297 = x14273 * x14291;
for(int x14292=0; x14292 < x14273; x14292++) {
int32_t x14304 = x14285 * x14292;
int32_t x14305 = x14303 + x14304;
float x14307 = x14194[x14305];
float x14308 = x96[x14306];
int32_t x14294 = x14292 + x14293;
int32_t x14296 = x14294 + x14295;
int32_t x14298 = x14296 + x14297;
float x14309 = x14307 + x14308;
x14282[x14298] = x14309;

}

}

}

}
float* x14319 = (float*)myMalloc(x14281 * sizeof(float));;
for(int x14321=0; x14321 < x14281; x14321++) {
float x14322 = x14282[x14321];
bool x14323 = x14322 < 0.0f;
if (x14323) {
x14319[x14321] = 0.0f;
} else {
float x14326 = x14282[x14321];
x14319[x14321] = x14326;
}

}
float* x14340 = (float*)myMalloc(x14339 * sizeof(float));;
int32_t x14343 = 64 * x14271;
int32_t x14344 = x14343 * x14335;
float* x14345 = (float*)myMalloc(x14344 * sizeof(float));;
int32_t x14341 = x14271 * x14335;
for(int x14346=0; x14346 < 64; x14346++) {
int32_t x14347 = x14346 * x14280;
float* x14348 = x14319+x14347;
int32_t x14349 = x14346 * x14336;
float* x14350 = x14340+x14349;
int32_t x14351 = x14346 * x14341;
float* x14352 = x14345+x14351;
for(int x14353=0; x14353 < x14271; x14353++) {
int32_t x14354 = x14353 / 1;
int32_t x14358 = x14354 * x14334;
int32_t x14359 = x14358 * x14334;
int32_t x14355 = x14353 % 1;
int32_t x14356 = x14355 / 1;
int32_t x14360 = x14356 * x14334;
int32_t x14361 = x14360 * x14334;
int32_t x14362 = x14359 + x14361;
int32_t x14357 = x14355 % 1;
int32_t x14363 = x14357 * x14334;
int32_t x14364 = x14363 * x14334;
int32_t x14365 = x14362 + x14364;
float* x14366 = x14352+x14365;
int32_t x14367 = x14354 * x14273;
int32_t x14368 = x14367 * x14273;
float* x14369 = x14348+x14368;
for(int x14371=0; x14371 < x14334; x14371++) {
int32_t x14373 = x14371 * x14334;
float* x14374 = x14366+x14373;
int32_t x14372 = x14371 + x14356;
int32_t x14375 = x14372 * x14273;
int32_t x14376 = x14375 + x14357;
float* x14377 = x14369+x14376;
memcpy(x14374, x14377, 4 * x14334);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x14335,x14271,1,x56,x14271,x14352,x14335,1,x14350,x14335);

}
int32_t x14386 = 0;
int32_t x14387 = 1;
x14387 *= 1;
x14386 += 1;
x14387 *= 1;
x14387 *= 1;
int32_t x14392 = x14386;
bool x14393 = x14392 >= 2;
if (x14393) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14398 = x14392 == 0;
if (x14398) {
int32_t x14399 = x14387;
bool x14400 = x14399 == 1024;
if (x14400) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14407 = x14387;
int32_t x14408 = 1024 / x14407;
bool x14409 = x14408 == 1;
bool x14412;
if (x454) {
bool x14410 = 1024 == x14408;
bool x14411 = x14409 || x14410;
x14412 = x14411;
} else {
x14412 = false;
}
bool x14416;
if (x14412) {
x14416 = x14415;
} else {
x14416 = false;
}
bool x14417;
if (x14416) {
x14417 = x14415;
} else {
x14417 = false;
}
if (x14417) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x14334,x14334,1,x14408,1,1);
assert(false && "");
}
bool x14423 = 1024 <= x14408;
int32_t x14424;
if (x14423) {
x14424 = x14408;
} else {
x14424 = 1024;
}
int32_t x14433 = x14424 * x14432;
int32_t x14434 = 64 * x14433;
float* x14435 = (float*)myMalloc(x14434 * sizeof(float));;
int32_t x14438;
if (x14409) {
x14438 = 0;
} else {
x14438 = 1;
}
for(int x14439=0; x14439 < 64; x14439++) {
int32_t x14451 = x14336 * x14439;
int32_t x14445 = x14433 * x14439;
for(int x14441=0; x14441 < x14424; x14441++) {
int32_t x14452 = x14335 * x14441;
int32_t x14453 = x14451 + x14452;
int32_t x14458 = x14438 * x14441;
int32_t x14447 = x14432 * x14441;
for(int x14443=0; x14443 < x14426; x14443++) {
int32_t x14454 = x14436 * x14443;
int32_t x14455 = x14453 + x14454;
int32_t x14449 = x14426 * x14443;
for(int x14444=0; x14444 < x14426; x14444++) {
int32_t x14456 = x14437 * x14444;
int32_t x14457 = x14455 + x14456;
float x14459 = x14340[x14457];
float x14460 = x182[x14458];
int32_t x14446 = x14444 + x14445;
int32_t x14448 = x14446 + x14447;
int32_t x14450 = x14448 + x14449;
float x14461 = x14459 - x14460;
x14435[x14450] = x14461;

}

}

}

}
float* x14471 = (float*)myMalloc(1024 * sizeof(float));;
for(int x14472=0; x14472 < 1024; x14472++) {
float x14473 = x143[x14472];
float x14474 = x14473 + 1.0E-5f;
x14471[x14472] = x14474;

}
float* x14478 = (float*)myMalloc(1024 * sizeof(float));;
for(int x14479=0; x14479 < 1024; x14479++) {
float x14480 = x14471[x14479];
double x14481 = (double)x14480;
double x14482 = sqrt(x14481);
float x14483 = (float)x14482;
x14478[x14479] = x14483;

}
int32_t x14487 = 0;
int32_t x14488 = 1;
x14488 *= 1;
x14487 += 1;
x14488 *= 1;
x14488 *= 1;
int32_t x14493 = x14487;
bool x14494 = x14493 >= 2;
if (x14494) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14499 = x14493 == 0;
if (x14499) {
int32_t x14500 = x14488;
bool x14501 = x14500 == 1024;
if (x14501) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14508 = x14488;
bool x14510 = x14424 == 1;
int32_t x14509 = 1024 / x14508;
bool x14511 = x14509 == 1;
bool x14515;
if (x454) {
bool x14512 = x14510 || x14511;
bool x14513 = x14424 == x14509;
bool x14514 = x14512 || x14513;
x14515 = x14514;
} else {
x14515 = false;
}
bool x14519;
if (x14515) {
x14519 = x14518;
} else {
x14519 = false;
}
bool x14520;
if (x14519) {
x14520 = x14518;
} else {
x14520 = false;
}
if (x14520) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14424,x14426,x14426,1,x14509,1,1);
assert(false && "");
}
bool x14526 = x14424 <= x14509;
int32_t x14527;
if (x14526) {
x14527 = x14509;
} else {
x14527 = x14424;
}
int32_t x14536 = x14527 * x14535;
int32_t x14537 = 64 * x14536;
float* x14538 = (float*)myMalloc(x14537 * sizeof(float));;
int32_t x14539;
if (x14510) {
x14539 = 0;
} else {
x14539 = x14432;
}
int32_t x14542;
if (x14511) {
x14542 = 0;
} else {
x14542 = 1;
}
for(int x14543=0; x14543 < 64; x14543++) {
int32_t x14555 = x14433 * x14543;
int32_t x14549 = x14536 * x14543;
for(int x14545=0; x14545 < x14527; x14545++) {
int32_t x14556 = x14539 * x14545;
int32_t x14557 = x14555 + x14556;
int32_t x14562 = x14542 * x14545;
int32_t x14551 = x14535 * x14545;
for(int x14547=0; x14547 < x14529; x14547++) {
int32_t x14558 = x14540 * x14547;
int32_t x14559 = x14557 + x14558;
int32_t x14553 = x14529 * x14547;
for(int x14548=0; x14548 < x14529; x14548++) {
int32_t x14560 = x14541 * x14548;
int32_t x14561 = x14559 + x14560;
float x14563 = x14435[x14561];
float x14564 = x14478[x14562];
int32_t x14550 = x14548 + x14549;
int32_t x14552 = x14550 + x14551;
int32_t x14554 = x14552 + x14553;
float x14565 = x14563 / x14564;
x14538[x14554] = x14565;

}

}

}

}
int32_t x14575 = 0;
int32_t x14576 = 1;
x14576 *= 1;
x14575 += 1;
x14576 *= 1;
x14576 *= 1;
int32_t x14581 = x14575;
bool x14582 = x14581 >= 2;
if (x14582) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14587 = x14581 == 0;
if (x14587) {
int32_t x14588 = x14576;
bool x14589 = x14588 == 1024;
if (x14589) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14596 = x14576;
bool x14598 = x14527 == 1;
int32_t x14597 = 1024 / x14596;
bool x14599 = x14597 == 1;
bool x14603;
if (x454) {
bool x14600 = x14598 || x14599;
bool x14601 = x14527 == x14597;
bool x14602 = x14600 || x14601;
x14603 = x14602;
} else {
x14603 = false;
}
bool x14607;
if (x14603) {
x14607 = x14606;
} else {
x14607 = false;
}
bool x14608;
if (x14607) {
x14608 = x14606;
} else {
x14608 = false;
}
if (x14608) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14527,x14529,x14529,1,x14597,1,1);
assert(false && "");
}
bool x14614 = x14527 <= x14597;
int32_t x14615;
if (x14614) {
x14615 = x14597;
} else {
x14615 = x14527;
}
int32_t x14624 = x14615 * x14623;
int32_t x14625 = 64 * x14624;
float* x14626 = (float*)myMalloc(x14625 * sizeof(float));;
int32_t x14627;
if (x14598) {
x14627 = 0;
} else {
x14627 = x14535;
}
int32_t x14630;
if (x14599) {
x14630 = 0;
} else {
x14630 = 1;
}
for(int x14631=0; x14631 < 64; x14631++) {
int32_t x14643 = x14536 * x14631;
int32_t x14637 = x14624 * x14631;
for(int x14633=0; x14633 < x14615; x14633++) {
int32_t x14644 = x14627 * x14633;
int32_t x14645 = x14643 + x14644;
int32_t x14650 = x14630 * x14633;
int32_t x14639 = x14623 * x14633;
for(int x14635=0; x14635 < x14617; x14635++) {
int32_t x14646 = x14628 * x14635;
int32_t x14647 = x14645 + x14646;
int32_t x14641 = x14617 * x14635;
for(int x14636=0; x14636 < x14617; x14636++) {
int32_t x14648 = x14629 * x14636;
int32_t x14649 = x14647 + x14648;
float x14651 = x14538[x14649];
float x14652 = x20[x14650];
int32_t x14638 = x14636 + x14637;
int32_t x14640 = x14638 + x14639;
int32_t x14642 = x14640 + x14641;
float x14653 = x14651 * x14652;
x14626[x14642] = x14653;

}

}

}

}
int32_t x14663 = 0;
int32_t x14664 = 1;
x14664 *= 1;
x14663 += 1;
x14664 *= 1;
x14664 *= 1;
int32_t x14669 = x14663;
bool x14670 = x14669 >= 2;
if (x14670) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14675 = x14669 == 0;
if (x14675) {
int32_t x14676 = x14664;
bool x14677 = x14676 == 1024;
if (x14677) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14684 = x14664;
bool x14686 = x14615 == 1;
int32_t x14685 = 1024 / x14684;
bool x14687 = x14685 == 1;
bool x14691;
if (x454) {
bool x14688 = x14686 || x14687;
bool x14689 = x14615 == x14685;
bool x14690 = x14688 || x14689;
x14691 = x14690;
} else {
x14691 = false;
}
bool x14695;
if (x14691) {
x14695 = x14694;
} else {
x14695 = false;
}
bool x14696;
if (x14695) {
x14696 = x14694;
} else {
x14696 = false;
}
if (x14696) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14615,x14617,x14617,1,x14685,1,1);
assert(false && "");
}
bool x14702 = x14615 <= x14685;
int32_t x14703;
if (x14702) {
x14703 = x14685;
} else {
x14703 = x14615;
}
int32_t x14712 = x14703 * x14711;
int32_t x14713 = 64 * x14712;
float* x14714 = (float*)myMalloc(x14713 * sizeof(float));;
int32_t x14715;
if (x14686) {
x14715 = 0;
} else {
x14715 = x14623;
}
int32_t x14718;
if (x14687) {
x14718 = 0;
} else {
x14718 = 1;
}
for(int x14719=0; x14719 < 64; x14719++) {
int32_t x14731 = x14624 * x14719;
int32_t x14725 = x14712 * x14719;
for(int x14721=0; x14721 < x14703; x14721++) {
int32_t x14732 = x14715 * x14721;
int32_t x14733 = x14731 + x14732;
int32_t x14738 = x14718 * x14721;
int32_t x14727 = x14711 * x14721;
for(int x14723=0; x14723 < x14705; x14723++) {
int32_t x14734 = x14716 * x14723;
int32_t x14735 = x14733 + x14734;
int32_t x14729 = x14705 * x14723;
for(int x14724=0; x14724 < x14705; x14724++) {
int32_t x14736 = x14717 * x14724;
int32_t x14737 = x14735 + x14736;
float x14739 = x14626[x14737];
float x14740 = x232[x14738];
int32_t x14726 = x14724 + x14725;
int32_t x14728 = x14726 + x14727;
int32_t x14730 = x14728 + x14729;
float x14741 = x14739 + x14740;
x14714[x14730] = x14741;

}

}

}

}
bool x14751 = x14703 == 1;
bool x14752 = x14751 || x13334;
bool x14753 = x14703 == x12862;
bool x14754 = x14752 || x14753;
bool x14759;
if (x14754) {
x14759 = x14758;
} else {
x14759 = false;
}
bool x14760;
if (x14759) {
x14760 = x14758;
} else {
x14760 = false;
}
if (x14760) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14703,x14705,x14705,64,x12862,x12864,x12864);
assert(false && "");
}
bool x14766 = x14703 <= x12862;
int32_t x14767;
if (x14766) {
x14767 = x12862;
} else {
x14767 = x14703;
}
int32_t x14783;
if (x14751) {
x14783 = 0;
} else {
x14783 = x14711;
}
for(int x14786=0; x14786 < 64; x14786++) {
int32_t x14792 = x14712 * x14786;
int32_t x14799 = x12871 * x14786;
for(int x14788=0; x14788 < x14767; x14788++) {
int32_t x14793 = x14783 * x14788;
int32_t x14794 = x14792 + x14793;
int32_t x14800 = x13368 * x14788;
int32_t x14801 = x14799 + x14800;
for(int x14790=0; x14790 < x14769; x14790++) {
int32_t x14795 = x14784 * x14790;
int32_t x14796 = x14794 + x14795;
int32_t x14802 = x13369 * x14790;
int32_t x14803 = x14801 + x14802;
for(int x14791=0; x14791 < x14769; x14791++) {
int32_t x14797 = x14785 * x14791;
int32_t x14798 = x14796 + x14797;
float x14806 = x14714[x14798];
int32_t x14804 = x13370 * x14791;
int32_t x14805 = x14803 + x14804;
float x14807 = x13406[x14805];
float x14808 = x14806 + x14807;
x14714[x14798] = x14808;

}

}

}

}
float* x14818 = (float*)myMalloc(x14713 * sizeof(float));;
for(int x14820=0; x14820 < x14713; x14820++) {
float x14821 = x14714[x14820];
bool x14822 = x14821 < 0.0f;
if (x14822) {
x14818[x14820] = 0.0f;
} else {
float x14825 = x14714[x14820];
x14818[x14820] = x14825;
}

}
float* x14839 = (float*)myMalloc(x14838 * sizeof(float));;
int32_t x14842 = 64 * x14703;
int32_t x14843 = x14842 * x14834;
float* x14844 = (float*)myMalloc(x14843 * sizeof(float));;
int32_t x14840 = x14703 * x14834;
for(int x14845=0; x14845 < 64; x14845++) {
int32_t x14846 = x14845 * x14712;
float* x14847 = x14818+x14846;
int32_t x14848 = x14845 * x14835;
float* x14849 = x14839+x14848;
int32_t x14850 = x14845 * x14840;
float* x14851 = x14844+x14850;
for(int x14852=0; x14852 < x14703; x14852++) {
int32_t x14853 = x14852 / 1;
int32_t x14857 = x14853 * x14833;
int32_t x14858 = x14857 * x14833;
int32_t x14854 = x14852 % 1;
int32_t x14855 = x14854 / 1;
int32_t x14859 = x14855 * x14833;
int32_t x14860 = x14859 * x14833;
int32_t x14861 = x14858 + x14860;
int32_t x14856 = x14854 % 1;
int32_t x14862 = x14856 * x14833;
int32_t x14863 = x14862 * x14833;
int32_t x14864 = x14861 + x14863;
float* x14865 = x14851+x14864;
int32_t x14866 = x14853 * x14705;
int32_t x14867 = x14866 * x14705;
float* x14868 = x14847+x14867;
for(int x14870=0; x14870 < x14833; x14870++) {
int32_t x14872 = x14870 * x14833;
float* x14873 = x14865+x14872;
int32_t x14871 = x14870 + x14855;
int32_t x14874 = x14871 * x14705;
int32_t x14875 = x14874 + x14856;
float* x14876 = x14868+x14875;
memcpy(x14873, x14876, 4 * x14833);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x14834,x14703,1,x218,x14703,x14851,x14834,1,x14849,x14834);

}
int32_t x14885 = 0;
int32_t x14886 = 1;
x14886 *= 1;
x14885 += 1;
x14886 *= 1;
x14886 *= 1;
int32_t x14891 = x14885;
bool x14892 = x14891 >= 2;
if (x14892) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14897 = x14891 == 0;
if (x14897) {
int32_t x14898 = x14886;
bool x14899 = x14898 == 256;
if (x14899) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14906 = x14886;
int32_t x14907 = 256 / x14906;
bool x14908 = x14907 == 1;
bool x14911;
if (x454) {
bool x14909 = 256 == x14907;
bool x14910 = x14908 || x14909;
x14911 = x14910;
} else {
x14911 = false;
}
bool x14915;
if (x14911) {
x14915 = x14914;
} else {
x14915 = false;
}
bool x14916;
if (x14915) {
x14916 = x14914;
} else {
x14916 = false;
}
if (x14916) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x14833,x14833,1,x14907,1,1);
assert(false && "");
}
bool x14922 = 256 <= x14907;
int32_t x14923;
if (x14922) {
x14923 = x14907;
} else {
x14923 = 256;
}
int32_t x14932 = x14923 * x14931;
int32_t x14933 = 64 * x14932;
float* x14934 = (float*)myMalloc(x14933 * sizeof(float));;
int32_t x14937;
if (x14908) {
x14937 = 0;
} else {
x14937 = 1;
}
for(int x14938=0; x14938 < 64; x14938++) {
int32_t x14950 = x14835 * x14938;
int32_t x14944 = x14932 * x14938;
for(int x14940=0; x14940 < x14923; x14940++) {
int32_t x14951 = x14834 * x14940;
int32_t x14952 = x14950 + x14951;
int32_t x14957 = x14937 * x14940;
int32_t x14946 = x14931 * x14940;
for(int x14942=0; x14942 < x14925; x14942++) {
int32_t x14953 = x14935 * x14942;
int32_t x14954 = x14952 + x14953;
int32_t x14948 = x14925 * x14942;
for(int x14943=0; x14943 < x14925; x14943++) {
int32_t x14955 = x14936 * x14943;
int32_t x14956 = x14954 + x14955;
float x14958 = x14839[x14956];
float x14959 = x178[x14957];
int32_t x14945 = x14943 + x14944;
int32_t x14947 = x14945 + x14946;
int32_t x14949 = x14947 + x14948;
float x14960 = x14958 - x14959;
x14934[x14949] = x14960;

}

}

}

}
float* x14970 = (float*)myMalloc(256 * sizeof(float));;
for(int x14971=0; x14971 < 256; x14971++) {
float x14972 = x174[x14971];
float x14973 = x14972 + 1.0E-5f;
x14970[x14971] = x14973;

}
float* x14977 = (float*)myMalloc(256 * sizeof(float));;
for(int x14978=0; x14978 < 256; x14978++) {
float x14979 = x14970[x14978];
double x14980 = (double)x14979;
double x14981 = sqrt(x14980);
float x14982 = (float)x14981;
x14977[x14978] = x14982;

}
int32_t x14986 = 0;
int32_t x14987 = 1;
x14987 *= 1;
x14986 += 1;
x14987 *= 1;
x14987 *= 1;
int32_t x14992 = x14986;
bool x14993 = x14992 >= 2;
if (x14993) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14998 = x14992 == 0;
if (x14998) {
int32_t x14999 = x14987;
bool x15000 = x14999 == 256;
if (x15000) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15007 = x14987;
bool x15009 = x14923 == 1;
int32_t x15008 = 256 / x15007;
bool x15010 = x15008 == 1;
bool x15014;
if (x454) {
bool x15011 = x15009 || x15010;
bool x15012 = x14923 == x15008;
bool x15013 = x15011 || x15012;
x15014 = x15013;
} else {
x15014 = false;
}
bool x15018;
if (x15014) {
x15018 = x15017;
} else {
x15018 = false;
}
bool x15019;
if (x15018) {
x15019 = x15017;
} else {
x15019 = false;
}
if (x15019) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14923,x14925,x14925,1,x15008,1,1);
assert(false && "");
}
bool x15025 = x14923 <= x15008;
int32_t x15026;
if (x15025) {
x15026 = x15008;
} else {
x15026 = x14923;
}
int32_t x15035 = x15026 * x15034;
int32_t x15036 = 64 * x15035;
float* x15037 = (float*)myMalloc(x15036 * sizeof(float));;
int32_t x15038;
if (x15009) {
x15038 = 0;
} else {
x15038 = x14931;
}
int32_t x15041;
if (x15010) {
x15041 = 0;
} else {
x15041 = 1;
}
for(int x15042=0; x15042 < 64; x15042++) {
int32_t x15054 = x14932 * x15042;
int32_t x15048 = x15035 * x15042;
for(int x15044=0; x15044 < x15026; x15044++) {
int32_t x15055 = x15038 * x15044;
int32_t x15056 = x15054 + x15055;
int32_t x15061 = x15041 * x15044;
int32_t x15050 = x15034 * x15044;
for(int x15046=0; x15046 < x15028; x15046++) {
int32_t x15057 = x15039 * x15046;
int32_t x15058 = x15056 + x15057;
int32_t x15052 = x15028 * x15046;
for(int x15047=0; x15047 < x15028; x15047++) {
int32_t x15059 = x15040 * x15047;
int32_t x15060 = x15058 + x15059;
float x15062 = x14934[x15060];
float x15063 = x14977[x15061];
int32_t x15049 = x15047 + x15048;
int32_t x15051 = x15049 + x15050;
int32_t x15053 = x15051 + x15052;
float x15064 = x15062 / x15063;
x15037[x15053] = x15064;

}

}

}

}
int32_t x15074 = 0;
int32_t x15075 = 1;
x15075 *= 1;
x15074 += 1;
x15075 *= 1;
x15075 *= 1;
int32_t x15080 = x15074;
bool x15081 = x15080 >= 2;
if (x15081) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15086 = x15080 == 0;
if (x15086) {
int32_t x15087 = x15075;
bool x15088 = x15087 == 256;
if (x15088) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15095 = x15075;
bool x15097 = x15026 == 1;
int32_t x15096 = 256 / x15095;
bool x15098 = x15096 == 1;
bool x15102;
if (x454) {
bool x15099 = x15097 || x15098;
bool x15100 = x15026 == x15096;
bool x15101 = x15099 || x15100;
x15102 = x15101;
} else {
x15102 = false;
}
bool x15106;
if (x15102) {
x15106 = x15105;
} else {
x15106 = false;
}
bool x15107;
if (x15106) {
x15107 = x15105;
} else {
x15107 = false;
}
if (x15107) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15026,x15028,x15028,1,x15096,1,1);
assert(false && "");
}
bool x15113 = x15026 <= x15096;
int32_t x15114;
if (x15113) {
x15114 = x15096;
} else {
x15114 = x15026;
}
int32_t x15123 = x15114 * x15122;
int32_t x15124 = 64 * x15123;
float* x15125 = (float*)myMalloc(x15124 * sizeof(float));;
int32_t x15126;
if (x15097) {
x15126 = 0;
} else {
x15126 = x15034;
}
int32_t x15129;
if (x15098) {
x15129 = 0;
} else {
x15129 = 1;
}
for(int x15130=0; x15130 < 64; x15130++) {
int32_t x15142 = x15035 * x15130;
int32_t x15136 = x15123 * x15130;
for(int x15132=0; x15132 < x15114; x15132++) {
int32_t x15143 = x15126 * x15132;
int32_t x15144 = x15142 + x15143;
int32_t x15149 = x15129 * x15132;
int32_t x15138 = x15122 * x15132;
for(int x15134=0; x15134 < x15116; x15134++) {
int32_t x15145 = x15127 * x15134;
int32_t x15146 = x15144 + x15145;
int32_t x15140 = x15116 * x15134;
for(int x15135=0; x15135 < x15116; x15135++) {
int32_t x15147 = x15128 * x15135;
int32_t x15148 = x15146 + x15147;
float x15150 = x15037[x15148];
float x15151 = x129[x15149];
int32_t x15137 = x15135 + x15136;
int32_t x15139 = x15137 + x15138;
int32_t x15141 = x15139 + x15140;
float x15152 = x15150 * x15151;
x15125[x15141] = x15152;

}

}

}

}
int32_t x15162 = 0;
int32_t x15163 = 1;
x15163 *= 1;
x15162 += 1;
x15163 *= 1;
x15163 *= 1;
int32_t x15168 = x15162;
bool x15169 = x15168 >= 2;
if (x15169) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15174 = x15168 == 0;
if (x15174) {
int32_t x15175 = x15163;
bool x15176 = x15175 == 256;
if (x15176) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15183 = x15163;
bool x15185 = x15114 == 1;
int32_t x15184 = 256 / x15183;
bool x15186 = x15184 == 1;
bool x15190;
if (x454) {
bool x15187 = x15185 || x15186;
bool x15188 = x15114 == x15184;
bool x15189 = x15187 || x15188;
x15190 = x15189;
} else {
x15190 = false;
}
bool x15194;
if (x15190) {
x15194 = x15193;
} else {
x15194 = false;
}
bool x15195;
if (x15194) {
x15195 = x15193;
} else {
x15195 = false;
}
if (x15195) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15114,x15116,x15116,1,x15184,1,1);
assert(false && "");
}
bool x15201 = x15114 <= x15184;
int32_t x15202;
if (x15201) {
x15202 = x15184;
} else {
x15202 = x15114;
}
int32_t x15211 = x15202 * x15210;
int32_t x15212 = 64 * x15211;
float* x15213 = (float*)myMalloc(x15212 * sizeof(float));;
int32_t x15214;
if (x15185) {
x15214 = 0;
} else {
x15214 = x15122;
}
int32_t x15217;
if (x15186) {
x15217 = 0;
} else {
x15217 = 1;
}
for(int x15218=0; x15218 < 64; x15218++) {
int32_t x15230 = x15123 * x15218;
int32_t x15224 = x15211 * x15218;
for(int x15220=0; x15220 < x15202; x15220++) {
int32_t x15231 = x15214 * x15220;
int32_t x15232 = x15230 + x15231;
int32_t x15237 = x15217 * x15220;
int32_t x15226 = x15210 * x15220;
for(int x15222=0; x15222 < x15204; x15222++) {
int32_t x15233 = x15215 * x15222;
int32_t x15234 = x15232 + x15233;
int32_t x15228 = x15204 * x15222;
for(int x15223=0; x15223 < x15204; x15223++) {
int32_t x15235 = x15216 * x15223;
int32_t x15236 = x15234 + x15235;
float x15238 = x15125[x15236];
float x15239 = x197[x15237];
int32_t x15225 = x15223 + x15224;
int32_t x15227 = x15225 + x15226;
int32_t x15229 = x15227 + x15228;
float x15240 = x15238 + x15239;
x15213[x15229] = x15240;

}

}

}

}
float* x15250 = (float*)myMalloc(x15212 * sizeof(float));;
for(int x15252=0; x15252 < x15212; x15252++) {
float x15253 = x15213[x15252];
bool x15254 = x15253 < 0.0f;
if (x15254) {
x15250[x15252] = 0.0f;
} else {
float x15257 = x15213[x15252];
x15250[x15252] = x15257;
}

}
float* x15272 = (float*)myMalloc(x15271 * sizeof(float));;
int32_t x15273 = 9 * x15202;
int32_t x15276 = 64 * x15273;
int32_t x15277 = x15276 * x15267;
float* x15278 = (float*)myMalloc(x15277 * sizeof(float));;
int32_t x15274 = x15273 * x15267;
int32_t x15286 = x15202 * 3;
int32_t x15287 = x15286 * 3;
for(int x15279=0; x15279 < 64; x15279++) {
int32_t x15280 = x15279 * x15211;
float* x15281 = x15250+x15280;
int32_t x15282 = x15279 * x15268;
float* x15283 = x15272+x15282;
int32_t x15284 = x15279 * x15274;
float* x15285 = x15278+x15284;
for(int x15289=0; x15289 < x15287; x15289++) {
int32_t x15290 = x15289 / 9;
int32_t x15294 = x15290 * 3;
int32_t x15295 = x15294 * 3;
int32_t x15296 = x15295 * x15266;
int32_t x15297 = x15296 * x15266;
int32_t x15291 = x15289 % 9;
int32_t x15292 = x15291 / 3;
int32_t x15298 = x15292 * 3;
int32_t x15299 = x15298 * x15266;
int32_t x15300 = x15299 * x15266;
int32_t x15301 = x15297 + x15300;
int32_t x15293 = x15291 % 3;
int32_t x15302 = x15293 * x15266;
int32_t x15303 = x15302 * x15266;
int32_t x15304 = x15301 + x15303;
float* x15305 = x15285+x15304;
int32_t x15306 = x15290 * x15204;
int32_t x15307 = x15306 * x15204;
float* x15308 = x15281+x15307;
int32_t x15321 = 1 - x15293;
bool x15322 = x15321 > 0;
int32_t x15323;
if (x15322) {
x15323 = x15321;
} else {
x15323 = 0;
}
int32_t x15324 = 3 - x15293;
int32_t x15325 = x15324 - 1;
int32_t x15326 = 1 - x15325;
bool x15327 = x15326 > 0;
int32_t x15328;
if (x15327) {
x15328 = x15326;
} else {
x15328 = 0;
}
int32_t x15329 = x15266 - x15328;
int32_t x15330 = x15329 - x15323;
bool x15331 = x15330 <= 0;
bool x15335 = x15323 > 0;
int32_t x15320 = -1 + x15293;
bool x15348 = x15328 > 0;
for(int x15310=0; x15310 < x15266; x15310++) {
int32_t x15311 = x15310 - 1;
int32_t x15312 = x15311 + x15292;
bool x15313 = x15312 < 0;
bool x15314 = x15312 >= x15204;
bool x15315 = x15313 || x15314;
if (x15315) {
int32_t x15316 = x15310 * x15266;
float* x15317 = x15305+x15316;
memset(x15317, 0, 4 * x15266);;
} else {
if (x15331) {
int32_t x15316 = x15310 * x15266;
float* x15332 = x15305+x15316;
memset(x15332, 0, 4 * x15266);;
} else {
int32_t x15316 = x15310 * x15266;
if (x15335) {
float* x15336 = x15305+x15316;
memset(x15336, 0, 4 * x15323);;
} else {
}
// may have segfault here
int32_t x15341 = x15316 + x15323;
float* x15342 = x15305+x15341;
int32_t x15343 = x15312 * x15204;
int32_t x15344 = x15343 + x15320;
int32_t x15345 = x15344 + x15323;
float* x15346 = x15308+x15345;
memcpy(x15342, x15346, 4 * x15330);;
if (x15348) {
int32_t x15349 = x15316 + x15266;
int32_t x15350 = x15349 - x15328;
float* x15351 = x15305+x15350;
memset(x15351, 0, 4 * x15328);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x15267,x15273,1,x14,x15273,x15285,x15267,1,x15283,x15267);

}
int32_t x15366 = 0;
int32_t x15367 = 1;
x15367 *= 1;
x15366 += 1;
x15367 *= 1;
x15367 *= 1;
int32_t x15372 = x15366;
bool x15373 = x15372 >= 2;
if (x15373) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15378 = x15372 == 0;
if (x15378) {
int32_t x15379 = x15367;
bool x15380 = x15379 == 256;
if (x15380) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15387 = x15367;
int32_t x15388 = 256 / x15387;
bool x15389 = x15388 == 1;
bool x15392;
if (x454) {
bool x15390 = 256 == x15388;
bool x15391 = x15389 || x15390;
x15392 = x15391;
} else {
x15392 = false;
}
bool x15396;
if (x15392) {
x15396 = x15395;
} else {
x15396 = false;
}
bool x15397;
if (x15396) {
x15397 = x15395;
} else {
x15397 = false;
}
if (x15397) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x15266,x15266,1,x15388,1,1);
assert(false && "");
}
bool x15403 = 256 <= x15388;
int32_t x15404;
if (x15403) {
x15404 = x15388;
} else {
x15404 = 256;
}
int32_t x15413 = x15404 * x15412;
int32_t x15414 = 64 * x15413;
float* x15415 = (float*)myMalloc(x15414 * sizeof(float));;
int32_t x15418;
if (x15389) {
x15418 = 0;
} else {
x15418 = 1;
}
for(int x15419=0; x15419 < 64; x15419++) {
int32_t x15431 = x15268 * x15419;
int32_t x15425 = x15413 * x15419;
for(int x15421=0; x15421 < x15404; x15421++) {
int32_t x15432 = x15267 * x15421;
int32_t x15433 = x15431 + x15432;
int32_t x15438 = x15418 * x15421;
int32_t x15427 = x15412 * x15421;
for(int x15423=0; x15423 < x15406; x15423++) {
int32_t x15434 = x15416 * x15423;
int32_t x15435 = x15433 + x15434;
int32_t x15429 = x15406 * x15423;
for(int x15424=0; x15424 < x15406; x15424++) {
int32_t x15436 = x15417 * x15424;
int32_t x15437 = x15435 + x15436;
float x15439 = x15272[x15437];
float x15440 = x124[x15438];
int32_t x15426 = x15424 + x15425;
int32_t x15428 = x15426 + x15427;
int32_t x15430 = x15428 + x15429;
float x15441 = x15439 - x15440;
x15415[x15430] = x15441;

}

}

}

}
float* x15451 = (float*)myMalloc(256 * sizeof(float));;
for(int x15452=0; x15452 < 256; x15452++) {
float x15453 = x63[x15452];
float x15454 = x15453 + 1.0E-5f;
x15451[x15452] = x15454;

}
float* x15458 = (float*)myMalloc(256 * sizeof(float));;
for(int x15459=0; x15459 < 256; x15459++) {
float x15460 = x15451[x15459];
double x15461 = (double)x15460;
double x15462 = sqrt(x15461);
float x15463 = (float)x15462;
x15458[x15459] = x15463;

}
int32_t x15467 = 0;
int32_t x15468 = 1;
x15468 *= 1;
x15467 += 1;
x15468 *= 1;
x15468 *= 1;
int32_t x15473 = x15467;
bool x15474 = x15473 >= 2;
if (x15474) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15479 = x15473 == 0;
if (x15479) {
int32_t x15480 = x15468;
bool x15481 = x15480 == 256;
if (x15481) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15488 = x15468;
bool x15490 = x15404 == 1;
int32_t x15489 = 256 / x15488;
bool x15491 = x15489 == 1;
bool x15495;
if (x454) {
bool x15492 = x15490 || x15491;
bool x15493 = x15404 == x15489;
bool x15494 = x15492 || x15493;
x15495 = x15494;
} else {
x15495 = false;
}
bool x15499;
if (x15495) {
x15499 = x15498;
} else {
x15499 = false;
}
bool x15500;
if (x15499) {
x15500 = x15498;
} else {
x15500 = false;
}
if (x15500) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15404,x15406,x15406,1,x15489,1,1);
assert(false && "");
}
bool x15506 = x15404 <= x15489;
int32_t x15507;
if (x15506) {
x15507 = x15489;
} else {
x15507 = x15404;
}
int32_t x15516 = x15507 * x15515;
int32_t x15517 = 64 * x15516;
float* x15518 = (float*)myMalloc(x15517 * sizeof(float));;
int32_t x15519;
if (x15490) {
x15519 = 0;
} else {
x15519 = x15412;
}
int32_t x15522;
if (x15491) {
x15522 = 0;
} else {
x15522 = 1;
}
for(int x15523=0; x15523 < 64; x15523++) {
int32_t x15535 = x15413 * x15523;
int32_t x15529 = x15516 * x15523;
for(int x15525=0; x15525 < x15507; x15525++) {
int32_t x15536 = x15519 * x15525;
int32_t x15537 = x15535 + x15536;
int32_t x15542 = x15522 * x15525;
int32_t x15531 = x15515 * x15525;
for(int x15527=0; x15527 < x15509; x15527++) {
int32_t x15538 = x15520 * x15527;
int32_t x15539 = x15537 + x15538;
int32_t x15533 = x15509 * x15527;
for(int x15528=0; x15528 < x15509; x15528++) {
int32_t x15540 = x15521 * x15528;
int32_t x15541 = x15539 + x15540;
float x15543 = x15415[x15541];
float x15544 = x15458[x15542];
int32_t x15530 = x15528 + x15529;
int32_t x15532 = x15530 + x15531;
int32_t x15534 = x15532 + x15533;
float x15545 = x15543 / x15544;
x15518[x15534] = x15545;

}

}

}

}
int32_t x15555 = 0;
int32_t x15556 = 1;
x15556 *= 1;
x15555 += 1;
x15556 *= 1;
x15556 *= 1;
int32_t x15561 = x15555;
bool x15562 = x15561 >= 2;
if (x15562) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15567 = x15561 == 0;
if (x15567) {
int32_t x15568 = x15556;
bool x15569 = x15568 == 256;
if (x15569) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15576 = x15556;
bool x15578 = x15507 == 1;
int32_t x15577 = 256 / x15576;
bool x15579 = x15577 == 1;
bool x15583;
if (x454) {
bool x15580 = x15578 || x15579;
bool x15581 = x15507 == x15577;
bool x15582 = x15580 || x15581;
x15583 = x15582;
} else {
x15583 = false;
}
bool x15587;
if (x15583) {
x15587 = x15586;
} else {
x15587 = false;
}
bool x15588;
if (x15587) {
x15588 = x15586;
} else {
x15588 = false;
}
if (x15588) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15507,x15509,x15509,1,x15577,1,1);
assert(false && "");
}
bool x15594 = x15507 <= x15577;
int32_t x15595;
if (x15594) {
x15595 = x15577;
} else {
x15595 = x15507;
}
int32_t x15604 = x15595 * x15603;
int32_t x15605 = 64 * x15604;
float* x15606 = (float*)myMalloc(x15605 * sizeof(float));;
int32_t x15607;
if (x15578) {
x15607 = 0;
} else {
x15607 = x15515;
}
int32_t x15610;
if (x15579) {
x15610 = 0;
} else {
x15610 = 1;
}
for(int x15611=0; x15611 < 64; x15611++) {
int32_t x15623 = x15516 * x15611;
int32_t x15617 = x15604 * x15611;
for(int x15613=0; x15613 < x15595; x15613++) {
int32_t x15624 = x15607 * x15613;
int32_t x15625 = x15623 + x15624;
int32_t x15630 = x15610 * x15613;
int32_t x15619 = x15603 * x15613;
for(int x15615=0; x15615 < x15597; x15615++) {
int32_t x15626 = x15608 * x15615;
int32_t x15627 = x15625 + x15626;
int32_t x15621 = x15597 * x15615;
for(int x15616=0; x15616 < x15597; x15616++) {
int32_t x15628 = x15609 * x15616;
int32_t x15629 = x15627 + x15628;
float x15631 = x15518[x15629];
float x15632 = x228[x15630];
int32_t x15618 = x15616 + x15617;
int32_t x15620 = x15618 + x15619;
int32_t x15622 = x15620 + x15621;
float x15633 = x15631 * x15632;
x15606[x15622] = x15633;

}

}

}

}
int32_t x15643 = 0;
int32_t x15644 = 1;
x15644 *= 1;
x15643 += 1;
x15644 *= 1;
x15644 *= 1;
int32_t x15649 = x15643;
bool x15650 = x15649 >= 2;
if (x15650) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15655 = x15649 == 0;
if (x15655) {
int32_t x15656 = x15644;
bool x15657 = x15656 == 256;
if (x15657) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15664 = x15644;
bool x15666 = x15595 == 1;
int32_t x15665 = 256 / x15664;
bool x15667 = x15665 == 1;
bool x15671;
if (x454) {
bool x15668 = x15666 || x15667;
bool x15669 = x15595 == x15665;
bool x15670 = x15668 || x15669;
x15671 = x15670;
} else {
x15671 = false;
}
bool x15675;
if (x15671) {
x15675 = x15674;
} else {
x15675 = false;
}
bool x15676;
if (x15675) {
x15676 = x15674;
} else {
x15676 = false;
}
if (x15676) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15595,x15597,x15597,1,x15665,1,1);
assert(false && "");
}
bool x15682 = x15595 <= x15665;
int32_t x15683;
if (x15682) {
x15683 = x15665;
} else {
x15683 = x15595;
}
int32_t x15692 = x15683 * x15691;
int32_t x15693 = 64 * x15692;
float* x15694 = (float*)myMalloc(x15693 * sizeof(float));;
int32_t x15695;
if (x15666) {
x15695 = 0;
} else {
x15695 = x15603;
}
int32_t x15698;
if (x15667) {
x15698 = 0;
} else {
x15698 = 1;
}
for(int x15699=0; x15699 < 64; x15699++) {
int32_t x15711 = x15604 * x15699;
int32_t x15705 = x15692 * x15699;
for(int x15701=0; x15701 < x15683; x15701++) {
int32_t x15712 = x15695 * x15701;
int32_t x15713 = x15711 + x15712;
int32_t x15718 = x15698 * x15701;
int32_t x15707 = x15691 * x15701;
for(int x15703=0; x15703 < x15685; x15703++) {
int32_t x15714 = x15696 * x15703;
int32_t x15715 = x15713 + x15714;
int32_t x15709 = x15685 * x15703;
for(int x15704=0; x15704 < x15685; x15704++) {
int32_t x15716 = x15697 * x15704;
int32_t x15717 = x15715 + x15716;
float x15719 = x15606[x15717];
float x15720 = x192[x15718];
int32_t x15706 = x15704 + x15705;
int32_t x15708 = x15706 + x15707;
int32_t x15710 = x15708 + x15709;
float x15721 = x15719 + x15720;
x15694[x15710] = x15721;

}

}

}

}
float* x15731 = (float*)myMalloc(x15693 * sizeof(float));;
for(int x15733=0; x15733 < x15693; x15733++) {
float x15734 = x15694[x15733];
bool x15735 = x15734 < 0.0f;
if (x15735) {
x15731[x15733] = 0.0f;
} else {
float x15738 = x15694[x15733];
x15731[x15733] = x15738;
}

}
float* x15752 = (float*)myMalloc(x15751 * sizeof(float));;
int32_t x15755 = 64 * x15683;
int32_t x15756 = x15755 * x15747;
float* x15757 = (float*)myMalloc(x15756 * sizeof(float));;
int32_t x15753 = x15683 * x15747;
for(int x15758=0; x15758 < 64; x15758++) {
int32_t x15759 = x15758 * x15692;
float* x15760 = x15731+x15759;
int32_t x15761 = x15758 * x15748;
float* x15762 = x15752+x15761;
int32_t x15763 = x15758 * x15753;
float* x15764 = x15757+x15763;
for(int x15765=0; x15765 < x15683; x15765++) {
int32_t x15766 = x15765 / 1;
int32_t x15770 = x15766 * x15746;
int32_t x15771 = x15770 * x15746;
int32_t x15767 = x15765 % 1;
int32_t x15768 = x15767 / 1;
int32_t x15772 = x15768 * x15746;
int32_t x15773 = x15772 * x15746;
int32_t x15774 = x15771 + x15773;
int32_t x15769 = x15767 % 1;
int32_t x15775 = x15769 * x15746;
int32_t x15776 = x15775 * x15746;
int32_t x15777 = x15774 + x15776;
float* x15778 = x15764+x15777;
int32_t x15779 = x15766 * x15685;
int32_t x15780 = x15779 * x15685;
float* x15781 = x15760+x15780;
for(int x15783=0; x15783 < x15746; x15783++) {
int32_t x15785 = x15783 * x15746;
float* x15786 = x15778+x15785;
int32_t x15784 = x15783 + x15768;
int32_t x15787 = x15784 * x15685;
int32_t x15788 = x15787 + x15769;
float* x15789 = x15781+x15788;
memcpy(x15786, x15789, 4 * x15746);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x15747,x15683,1,x116,x15683,x15764,x15747,1,x15762,x15747);

}
int32_t x15798 = 0;
int32_t x15799 = 1;
x15799 *= 1;
x15798 += 1;
x15799 *= 1;
x15799 *= 1;
int32_t x15804 = x15798;
bool x15805 = x15804 >= 2;
if (x15805) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15810 = x15804 == 0;
if (x15810) {
int32_t x15811 = x15799;
bool x15812 = x15811 == 1024;
if (x15812) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15819 = x15799;
int32_t x15820 = 1024 / x15819;
bool x15821 = x15820 == 1;
bool x15824;
if (x454) {
bool x15822 = 1024 == x15820;
bool x15823 = x15821 || x15822;
x15824 = x15823;
} else {
x15824 = false;
}
bool x15828;
if (x15824) {
x15828 = x15827;
} else {
x15828 = false;
}
bool x15829;
if (x15828) {
x15829 = x15827;
} else {
x15829 = false;
}
if (x15829) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x15746,x15746,1,x15820,1,1);
assert(false && "");
}
bool x15835 = 1024 <= x15820;
int32_t x15836;
if (x15835) {
x15836 = x15820;
} else {
x15836 = 1024;
}
int32_t x15845 = x15836 * x15844;
int32_t x15846 = 64 * x15845;
float* x15847 = (float*)myMalloc(x15846 * sizeof(float));;
int32_t x15850;
if (x15821) {
x15850 = 0;
} else {
x15850 = 1;
}
for(int x15851=0; x15851 < 64; x15851++) {
int32_t x15863 = x15748 * x15851;
int32_t x15857 = x15845 * x15851;
for(int x15853=0; x15853 < x15836; x15853++) {
int32_t x15864 = x15747 * x15853;
int32_t x15865 = x15863 + x15864;
int32_t x15870 = x15850 * x15853;
int32_t x15859 = x15844 * x15853;
for(int x15855=0; x15855 < x15838; x15855++) {
int32_t x15866 = x15848 * x15855;
int32_t x15867 = x15865 + x15866;
int32_t x15861 = x15838 * x15855;
for(int x15856=0; x15856 < x15838; x15856++) {
int32_t x15868 = x15849 * x15856;
int32_t x15869 = x15867 + x15868;
float x15871 = x15752[x15869];
float x15872 = x140[x15870];
int32_t x15858 = x15856 + x15857;
int32_t x15860 = x15858 + x15859;
int32_t x15862 = x15860 + x15861;
float x15873 = x15871 - x15872;
x15847[x15862] = x15873;

}

}

}

}
float* x15883 = (float*)myMalloc(1024 * sizeof(float));;
for(int x15884=0; x15884 < 1024; x15884++) {
float x15885 = x188[x15884];
float x15886 = x15885 + 1.0E-5f;
x15883[x15884] = x15886;

}
float* x15890 = (float*)myMalloc(1024 * sizeof(float));;
for(int x15891=0; x15891 < 1024; x15891++) {
float x15892 = x15883[x15891];
double x15893 = (double)x15892;
double x15894 = sqrt(x15893);
float x15895 = (float)x15894;
x15890[x15891] = x15895;

}
int32_t x15899 = 0;
int32_t x15900 = 1;
x15900 *= 1;
x15899 += 1;
x15900 *= 1;
x15900 *= 1;
int32_t x15905 = x15899;
bool x15906 = x15905 >= 2;
if (x15906) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15911 = x15905 == 0;
if (x15911) {
int32_t x15912 = x15900;
bool x15913 = x15912 == 1024;
if (x15913) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15920 = x15900;
bool x15922 = x15836 == 1;
int32_t x15921 = 1024 / x15920;
bool x15923 = x15921 == 1;
bool x15927;
if (x454) {
bool x15924 = x15922 || x15923;
bool x15925 = x15836 == x15921;
bool x15926 = x15924 || x15925;
x15927 = x15926;
} else {
x15927 = false;
}
bool x15931;
if (x15927) {
x15931 = x15930;
} else {
x15931 = false;
}
bool x15932;
if (x15931) {
x15932 = x15930;
} else {
x15932 = false;
}
if (x15932) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15836,x15838,x15838,1,x15921,1,1);
assert(false && "");
}
bool x15938 = x15836 <= x15921;
int32_t x15939;
if (x15938) {
x15939 = x15921;
} else {
x15939 = x15836;
}
int32_t x15948 = x15939 * x15947;
int32_t x15949 = 64 * x15948;
float* x15950 = (float*)myMalloc(x15949 * sizeof(float));;
int32_t x15951;
if (x15922) {
x15951 = 0;
} else {
x15951 = x15844;
}
int32_t x15954;
if (x15923) {
x15954 = 0;
} else {
x15954 = 1;
}
for(int x15955=0; x15955 < 64; x15955++) {
int32_t x15967 = x15845 * x15955;
int32_t x15961 = x15948 * x15955;
for(int x15957=0; x15957 < x15939; x15957++) {
int32_t x15968 = x15951 * x15957;
int32_t x15969 = x15967 + x15968;
int32_t x15974 = x15954 * x15957;
int32_t x15963 = x15947 * x15957;
for(int x15959=0; x15959 < x15941; x15959++) {
int32_t x15970 = x15952 * x15959;
int32_t x15971 = x15969 + x15970;
int32_t x15965 = x15941 * x15959;
for(int x15960=0; x15960 < x15941; x15960++) {
int32_t x15972 = x15953 * x15960;
int32_t x15973 = x15971 + x15972;
float x15975 = x15847[x15973];
float x15976 = x15890[x15974];
int32_t x15962 = x15960 + x15961;
int32_t x15964 = x15962 + x15963;
int32_t x15966 = x15964 + x15965;
float x15977 = x15975 / x15976;
x15950[x15966] = x15977;

}

}

}

}
int32_t x15987 = 0;
int32_t x15988 = 1;
x15988 *= 1;
x15987 += 1;
x15988 *= 1;
x15988 *= 1;
int32_t x15993 = x15987;
bool x15994 = x15993 >= 2;
if (x15994) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15999 = x15993 == 0;
if (x15999) {
int32_t x16000 = x15988;
bool x16001 = x16000 == 1024;
if (x16001) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16008 = x15988;
bool x16010 = x15939 == 1;
int32_t x16009 = 1024 / x16008;
bool x16011 = x16009 == 1;
bool x16015;
if (x454) {
bool x16012 = x16010 || x16011;
bool x16013 = x15939 == x16009;
bool x16014 = x16012 || x16013;
x16015 = x16014;
} else {
x16015 = false;
}
bool x16019;
if (x16015) {
x16019 = x16018;
} else {
x16019 = false;
}
bool x16020;
if (x16019) {
x16020 = x16018;
} else {
x16020 = false;
}
if (x16020) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15939,x15941,x15941,1,x16009,1,1);
assert(false && "");
}
bool x16026 = x15939 <= x16009;
int32_t x16027;
if (x16026) {
x16027 = x16009;
} else {
x16027 = x15939;
}
int32_t x16036 = x16027 * x16035;
int32_t x16037 = 64 * x16036;
float* x16038 = (float*)myMalloc(x16037 * sizeof(float));;
int32_t x16039;
if (x16010) {
x16039 = 0;
} else {
x16039 = x15947;
}
int32_t x16042;
if (x16011) {
x16042 = 0;
} else {
x16042 = 1;
}
for(int x16043=0; x16043 < 64; x16043++) {
int32_t x16055 = x15948 * x16043;
int32_t x16049 = x16036 * x16043;
for(int x16045=0; x16045 < x16027; x16045++) {
int32_t x16056 = x16039 * x16045;
int32_t x16057 = x16055 + x16056;
int32_t x16062 = x16042 * x16045;
int32_t x16051 = x16035 * x16045;
for(int x16047=0; x16047 < x16029; x16047++) {
int32_t x16058 = x16040 * x16047;
int32_t x16059 = x16057 + x16058;
int32_t x16053 = x16029 * x16047;
for(int x16048=0; x16048 < x16029; x16048++) {
int32_t x16060 = x16041 * x16048;
int32_t x16061 = x16059 + x16060;
float x16063 = x15950[x16061];
float x16064 = x263[x16062];
int32_t x16050 = x16048 + x16049;
int32_t x16052 = x16050 + x16051;
int32_t x16054 = x16052 + x16053;
float x16065 = x16063 * x16064;
x16038[x16054] = x16065;

}

}

}

}
int32_t x16075 = 0;
int32_t x16076 = 1;
x16076 *= 1;
x16075 += 1;
x16076 *= 1;
x16076 *= 1;
int32_t x16081 = x16075;
bool x16082 = x16081 >= 2;
if (x16082) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16087 = x16081 == 0;
if (x16087) {
int32_t x16088 = x16076;
bool x16089 = x16088 == 1024;
if (x16089) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16096 = x16076;
bool x16098 = x16027 == 1;
int32_t x16097 = 1024 / x16096;
bool x16099 = x16097 == 1;
bool x16103;
if (x454) {
bool x16100 = x16098 || x16099;
bool x16101 = x16027 == x16097;
bool x16102 = x16100 || x16101;
x16103 = x16102;
} else {
x16103 = false;
}
bool x16107;
if (x16103) {
x16107 = x16106;
} else {
x16107 = false;
}
bool x16108;
if (x16107) {
x16108 = x16106;
} else {
x16108 = false;
}
if (x16108) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16027,x16029,x16029,1,x16097,1,1);
assert(false && "");
}
bool x16114 = x16027 <= x16097;
int32_t x16115;
if (x16114) {
x16115 = x16097;
} else {
x16115 = x16027;
}
int32_t x16124 = x16115 * x16123;
int32_t x16125 = 64 * x16124;
float* x16126 = (float*)myMalloc(x16125 * sizeof(float));;
int32_t x16127;
if (x16098) {
x16127 = 0;
} else {
x16127 = x16035;
}
int32_t x16130;
if (x16099) {
x16130 = 0;
} else {
x16130 = 1;
}
for(int x16131=0; x16131 < 64; x16131++) {
int32_t x16143 = x16036 * x16131;
int32_t x16137 = x16124 * x16131;
for(int x16133=0; x16133 < x16115; x16133++) {
int32_t x16144 = x16127 * x16133;
int32_t x16145 = x16143 + x16144;
int32_t x16150 = x16130 * x16133;
int32_t x16139 = x16123 * x16133;
for(int x16135=0; x16135 < x16117; x16135++) {
int32_t x16146 = x16128 * x16135;
int32_t x16147 = x16145 + x16146;
int32_t x16141 = x16117 * x16135;
for(int x16136=0; x16136 < x16117; x16136++) {
int32_t x16148 = x16129 * x16136;
int32_t x16149 = x16147 + x16148;
float x16151 = x16038[x16149];
float x16152 = x57[x16150];
int32_t x16138 = x16136 + x16137;
int32_t x16140 = x16138 + x16139;
int32_t x16142 = x16140 + x16141;
float x16153 = x16151 + x16152;
x16126[x16142] = x16153;

}

}

}

}
bool x16163 = x16115 == 1;
bool x16164 = x16163 || x14751;
bool x16165 = x16115 == x14703;
bool x16166 = x16164 || x16165;
bool x16171;
if (x16166) {
x16171 = x16170;
} else {
x16171 = false;
}
bool x16172;
if (x16171) {
x16172 = x16170;
} else {
x16172 = false;
}
if (x16172) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16115,x16117,x16117,64,x14703,x14705,x14705);
assert(false && "");
}
bool x16178 = x16115 <= x14703;
int32_t x16179;
if (x16178) {
x16179 = x14703;
} else {
x16179 = x16115;
}
int32_t x16195;
if (x16163) {
x16195 = 0;
} else {
x16195 = x16123;
}
for(int x16198=0; x16198 < 64; x16198++) {
int32_t x16204 = x16124 * x16198;
int32_t x16211 = x14712 * x16198;
for(int x16200=0; x16200 < x16179; x16200++) {
int32_t x16205 = x16195 * x16200;
int32_t x16206 = x16204 + x16205;
int32_t x16212 = x14783 * x16200;
int32_t x16213 = x16211 + x16212;
for(int x16202=0; x16202 < x16181; x16202++) {
int32_t x16207 = x16196 * x16202;
int32_t x16208 = x16206 + x16207;
int32_t x16214 = x14784 * x16202;
int32_t x16215 = x16213 + x16214;
for(int x16203=0; x16203 < x16181; x16203++) {
int32_t x16209 = x16197 * x16203;
int32_t x16210 = x16208 + x16209;
float x16218 = x16126[x16210];
int32_t x16216 = x14785 * x16203;
int32_t x16217 = x16215 + x16216;
float x16219 = x14818[x16217];
float x16220 = x16218 + x16219;
x16126[x16210] = x16220;

}

}

}

}
float* x16230 = (float*)myMalloc(x16125 * sizeof(float));;
for(int x16232=0; x16232 < x16125; x16232++) {
float x16233 = x16126[x16232];
bool x16234 = x16233 < 0.0f;
if (x16234) {
x16230[x16232] = 0.0f;
} else {
float x16237 = x16126[x16232];
x16230[x16232] = x16237;
}

}
float* x16251 = (float*)myMalloc(x16250 * sizeof(float));;
int32_t x16254 = 64 * x16115;
int32_t x16255 = x16254 * x16246;
float* x16256 = (float*)myMalloc(x16255 * sizeof(float));;
int32_t x16252 = x16115 * x16246;
for(int x16257=0; x16257 < 64; x16257++) {
int32_t x16258 = x16257 * x16124;
float* x16259 = x16230+x16258;
int32_t x16260 = x16257 * x16247;
float* x16261 = x16251+x16260;
int32_t x16262 = x16257 * x16252;
float* x16263 = x16256+x16262;
for(int x16264=0; x16264 < x16115; x16264++) {
int32_t x16265 = x16264 / 1;
int32_t x16269 = x16265 * x16245;
int32_t x16270 = x16269 * x16245;
int32_t x16266 = x16264 % 1;
int32_t x16267 = x16266 / 1;
int32_t x16271 = x16267 * x16245;
int32_t x16272 = x16271 * x16245;
int32_t x16273 = x16270 + x16272;
int32_t x16268 = x16266 % 1;
int32_t x16274 = x16268 * x16245;
int32_t x16275 = x16274 * x16245;
int32_t x16276 = x16273 + x16275;
float* x16277 = x16263+x16276;
int32_t x16278 = x16265 * x16117;
int32_t x16279 = x16278 * x16117;
float* x16280 = x16259+x16279;
for(int x16282=0; x16282 < x16245; x16282++) {
int32_t x16284 = x16282 * x16245;
float* x16285 = x16277+x16284;
int32_t x16283 = x16282 + x16267;
int32_t x16286 = x16283 * x16117;
int32_t x16287 = x16286 + x16268;
float* x16288 = x16280+x16287;
memcpy(x16285, x16288, 4 * x16245);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x16246,x16115,1,x6,x16115,x16263,x16246,1,x16261,x16246);

}
int32_t x16297 = 0;
int32_t x16298 = 1;
x16298 *= 1;
x16297 += 1;
x16298 *= 1;
x16298 *= 1;
int32_t x16303 = x16297;
bool x16304 = x16303 >= 2;
if (x16304) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16309 = x16303 == 0;
if (x16309) {
int32_t x16310 = x16298;
bool x16311 = x16310 == 256;
if (x16311) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16318 = x16298;
int32_t x16319 = 256 / x16318;
bool x16320 = x16319 == 1;
bool x16323;
if (x454) {
bool x16321 = 256 == x16319;
bool x16322 = x16320 || x16321;
x16323 = x16322;
} else {
x16323 = false;
}
bool x16327;
if (x16323) {
x16327 = x16326;
} else {
x16327 = false;
}
bool x16328;
if (x16327) {
x16328 = x16326;
} else {
x16328 = false;
}
if (x16328) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x16245,x16245,1,x16319,1,1);
assert(false && "");
}
bool x16334 = 256 <= x16319;
int32_t x16335;
if (x16334) {
x16335 = x16319;
} else {
x16335 = 256;
}
int32_t x16344 = x16335 * x16343;
int32_t x16345 = 64 * x16344;
float* x16346 = (float*)myMalloc(x16345 * sizeof(float));;
int32_t x16349;
if (x16320) {
x16349 = 0;
} else {
x16349 = 1;
}
for(int x16350=0; x16350 < 64; x16350++) {
int32_t x16362 = x16247 * x16350;
int32_t x16356 = x16344 * x16350;
for(int x16352=0; x16352 < x16335; x16352++) {
int32_t x16363 = x16246 * x16352;
int32_t x16364 = x16362 + x16363;
int32_t x16369 = x16349 * x16352;
int32_t x16358 = x16343 * x16352;
for(int x16354=0; x16354 < x16337; x16354++) {
int32_t x16365 = x16347 * x16354;
int32_t x16366 = x16364 + x16365;
int32_t x16360 = x16337 * x16354;
for(int x16355=0; x16355 < x16337; x16355++) {
int32_t x16367 = x16348 * x16355;
int32_t x16368 = x16366 + x16367;
float x16370 = x16251[x16368];
float x16371 = x163[x16369];
int32_t x16357 = x16355 + x16356;
int32_t x16359 = x16357 + x16358;
int32_t x16361 = x16359 + x16360;
float x16372 = x16370 - x16371;
x16346[x16361] = x16372;

}

}

}

}
float* x16382 = (float*)myMalloc(256 * sizeof(float));;
for(int x16383=0; x16383 < 256; x16383++) {
float x16384 = x98[x16383];
float x16385 = x16384 + 1.0E-5f;
x16382[x16383] = x16385;

}
float* x16389 = (float*)myMalloc(256 * sizeof(float));;
for(int x16390=0; x16390 < 256; x16390++) {
float x16391 = x16382[x16390];
double x16392 = (double)x16391;
double x16393 = sqrt(x16392);
float x16394 = (float)x16393;
x16389[x16390] = x16394;

}
int32_t x16398 = 0;
int32_t x16399 = 1;
x16399 *= 1;
x16398 += 1;
x16399 *= 1;
x16399 *= 1;
int32_t x16404 = x16398;
bool x16405 = x16404 >= 2;
if (x16405) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16410 = x16404 == 0;
if (x16410) {
int32_t x16411 = x16399;
bool x16412 = x16411 == 256;
if (x16412) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16419 = x16399;
bool x16421 = x16335 == 1;
int32_t x16420 = 256 / x16419;
bool x16422 = x16420 == 1;
bool x16426;
if (x454) {
bool x16423 = x16421 || x16422;
bool x16424 = x16335 == x16420;
bool x16425 = x16423 || x16424;
x16426 = x16425;
} else {
x16426 = false;
}
bool x16430;
if (x16426) {
x16430 = x16429;
} else {
x16430 = false;
}
bool x16431;
if (x16430) {
x16431 = x16429;
} else {
x16431 = false;
}
if (x16431) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16335,x16337,x16337,1,x16420,1,1);
assert(false && "");
}
bool x16437 = x16335 <= x16420;
int32_t x16438;
if (x16437) {
x16438 = x16420;
} else {
x16438 = x16335;
}
int32_t x16447 = x16438 * x16446;
int32_t x16448 = 64 * x16447;
float* x16449 = (float*)myMalloc(x16448 * sizeof(float));;
int32_t x16450;
if (x16421) {
x16450 = 0;
} else {
x16450 = x16343;
}
int32_t x16453;
if (x16422) {
x16453 = 0;
} else {
x16453 = 1;
}
for(int x16454=0; x16454 < 64; x16454++) {
int32_t x16466 = x16344 * x16454;
int32_t x16460 = x16447 * x16454;
for(int x16456=0; x16456 < x16438; x16456++) {
int32_t x16467 = x16450 * x16456;
int32_t x16468 = x16466 + x16467;
int32_t x16473 = x16453 * x16456;
int32_t x16462 = x16446 * x16456;
for(int x16458=0; x16458 < x16440; x16458++) {
int32_t x16469 = x16451 * x16458;
int32_t x16470 = x16468 + x16469;
int32_t x16464 = x16440 * x16458;
for(int x16459=0; x16459 < x16440; x16459++) {
int32_t x16471 = x16452 * x16459;
int32_t x16472 = x16470 + x16471;
float x16474 = x16346[x16472];
float x16475 = x16389[x16473];
int32_t x16461 = x16459 + x16460;
int32_t x16463 = x16461 + x16462;
int32_t x16465 = x16463 + x16464;
float x16476 = x16474 / x16475;
x16449[x16465] = x16476;

}

}

}

}
int32_t x16486 = 0;
int32_t x16487 = 1;
x16487 *= 1;
x16486 += 1;
x16487 *= 1;
x16487 *= 1;
int32_t x16492 = x16486;
bool x16493 = x16492 >= 2;
if (x16493) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16498 = x16492 == 0;
if (x16498) {
int32_t x16499 = x16487;
bool x16500 = x16499 == 256;
if (x16500) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16507 = x16487;
bool x16509 = x16438 == 1;
int32_t x16508 = 256 / x16507;
bool x16510 = x16508 == 1;
bool x16514;
if (x454) {
bool x16511 = x16509 || x16510;
bool x16512 = x16438 == x16508;
bool x16513 = x16511 || x16512;
x16514 = x16513;
} else {
x16514 = false;
}
bool x16518;
if (x16514) {
x16518 = x16517;
} else {
x16518 = false;
}
bool x16519;
if (x16518) {
x16519 = x16517;
} else {
x16519 = false;
}
if (x16519) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16438,x16440,x16440,1,x16508,1,1);
assert(false && "");
}
bool x16525 = x16438 <= x16508;
int32_t x16526;
if (x16525) {
x16526 = x16508;
} else {
x16526 = x16438;
}
int32_t x16535 = x16526 * x16534;
int32_t x16536 = 64 * x16535;
float* x16537 = (float*)myMalloc(x16536 * sizeof(float));;
int32_t x16538;
if (x16509) {
x16538 = 0;
} else {
x16538 = x16446;
}
int32_t x16541;
if (x16510) {
x16541 = 0;
} else {
x16541 = 1;
}
for(int x16542=0; x16542 < 64; x16542++) {
int32_t x16554 = x16447 * x16542;
int32_t x16548 = x16535 * x16542;
for(int x16544=0; x16544 < x16526; x16544++) {
int32_t x16555 = x16538 * x16544;
int32_t x16556 = x16554 + x16555;
int32_t x16561 = x16541 * x16544;
int32_t x16550 = x16534 * x16544;
for(int x16546=0; x16546 < x16528; x16546++) {
int32_t x16557 = x16539 * x16546;
int32_t x16558 = x16556 + x16557;
int32_t x16552 = x16528 * x16546;
for(int x16547=0; x16547 < x16528; x16547++) {
int32_t x16559 = x16540 * x16547;
int32_t x16560 = x16558 + x16559;
float x16562 = x16449[x16560];
float x16563 = x92[x16561];
int32_t x16549 = x16547 + x16548;
int32_t x16551 = x16549 + x16550;
int32_t x16553 = x16551 + x16552;
float x16564 = x16562 * x16563;
x16537[x16553] = x16564;

}

}

}

}
int32_t x16574 = 0;
int32_t x16575 = 1;
x16575 *= 1;
x16574 += 1;
x16575 *= 1;
x16575 *= 1;
int32_t x16580 = x16574;
bool x16581 = x16580 >= 2;
if (x16581) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16586 = x16580 == 0;
if (x16586) {
int32_t x16587 = x16575;
bool x16588 = x16587 == 256;
if (x16588) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16595 = x16575;
bool x16597 = x16526 == 1;
int32_t x16596 = 256 / x16595;
bool x16598 = x16596 == 1;
bool x16602;
if (x454) {
bool x16599 = x16597 || x16598;
bool x16600 = x16526 == x16596;
bool x16601 = x16599 || x16600;
x16602 = x16601;
} else {
x16602 = false;
}
bool x16606;
if (x16602) {
x16606 = x16605;
} else {
x16606 = false;
}
bool x16607;
if (x16606) {
x16607 = x16605;
} else {
x16607 = false;
}
if (x16607) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16526,x16528,x16528,1,x16596,1,1);
assert(false && "");
}
bool x16613 = x16526 <= x16596;
int32_t x16614;
if (x16613) {
x16614 = x16596;
} else {
x16614 = x16526;
}
int32_t x16623 = x16614 * x16622;
int32_t x16624 = 64 * x16623;
float* x16625 = (float*)myMalloc(x16624 * sizeof(float));;
int32_t x16626;
if (x16597) {
x16626 = 0;
} else {
x16626 = x16534;
}
int32_t x16629;
if (x16598) {
x16629 = 0;
} else {
x16629 = 1;
}
for(int x16630=0; x16630 < 64; x16630++) {
int32_t x16642 = x16535 * x16630;
int32_t x16636 = x16623 * x16630;
for(int x16632=0; x16632 < x16614; x16632++) {
int32_t x16643 = x16626 * x16632;
int32_t x16644 = x16642 + x16643;
int32_t x16649 = x16629 * x16632;
int32_t x16638 = x16622 * x16632;
for(int x16634=0; x16634 < x16616; x16634++) {
int32_t x16645 = x16627 * x16634;
int32_t x16646 = x16644 + x16645;
int32_t x16640 = x16616 * x16634;
for(int x16635=0; x16635 < x16616; x16635++) {
int32_t x16647 = x16628 * x16635;
int32_t x16648 = x16646 + x16647;
float x16650 = x16537[x16648];
float x16651 = x241[x16649];
int32_t x16637 = x16635 + x16636;
int32_t x16639 = x16637 + x16638;
int32_t x16641 = x16639 + x16640;
float x16652 = x16650 + x16651;
x16625[x16641] = x16652;

}

}

}

}
float* x16662 = (float*)myMalloc(x16624 * sizeof(float));;
for(int x16664=0; x16664 < x16624; x16664++) {
float x16665 = x16625[x16664];
bool x16666 = x16665 < 0.0f;
if (x16666) {
x16662[x16664] = 0.0f;
} else {
float x16669 = x16625[x16664];
x16662[x16664] = x16669;
}

}
float* x16684 = (float*)myMalloc(x16683 * sizeof(float));;
int32_t x16685 = 9 * x16614;
int32_t x16688 = 64 * x16685;
int32_t x16689 = x16688 * x16679;
float* x16690 = (float*)myMalloc(x16689 * sizeof(float));;
int32_t x16686 = x16685 * x16679;
int32_t x16698 = x16614 * 3;
int32_t x16699 = x16698 * 3;
for(int x16691=0; x16691 < 64; x16691++) {
int32_t x16692 = x16691 * x16623;
float* x16693 = x16662+x16692;
int32_t x16694 = x16691 * x16680;
float* x16695 = x16684+x16694;
int32_t x16696 = x16691 * x16686;
float* x16697 = x16690+x16696;
for(int x16701=0; x16701 < x16699; x16701++) {
int32_t x16702 = x16701 / 9;
int32_t x16706 = x16702 * 3;
int32_t x16707 = x16706 * 3;
int32_t x16708 = x16707 * x16678;
int32_t x16709 = x16708 * x16678;
int32_t x16703 = x16701 % 9;
int32_t x16704 = x16703 / 3;
int32_t x16710 = x16704 * 3;
int32_t x16711 = x16710 * x16678;
int32_t x16712 = x16711 * x16678;
int32_t x16713 = x16709 + x16712;
int32_t x16705 = x16703 % 3;
int32_t x16714 = x16705 * x16678;
int32_t x16715 = x16714 * x16678;
int32_t x16716 = x16713 + x16715;
float* x16717 = x16697+x16716;
int32_t x16718 = x16702 * x16616;
int32_t x16719 = x16718 * x16616;
float* x16720 = x16693+x16719;
int32_t x16733 = 1 - x16705;
bool x16734 = x16733 > 0;
int32_t x16735;
if (x16734) {
x16735 = x16733;
} else {
x16735 = 0;
}
int32_t x16736 = 3 - x16705;
int32_t x16737 = x16736 - 1;
int32_t x16738 = 1 - x16737;
bool x16739 = x16738 > 0;
int32_t x16740;
if (x16739) {
x16740 = x16738;
} else {
x16740 = 0;
}
int32_t x16741 = x16678 - x16740;
int32_t x16742 = x16741 - x16735;
bool x16743 = x16742 <= 0;
bool x16747 = x16735 > 0;
int32_t x16732 = -1 + x16705;
bool x16760 = x16740 > 0;
for(int x16722=0; x16722 < x16678; x16722++) {
int32_t x16723 = x16722 - 1;
int32_t x16724 = x16723 + x16704;
bool x16725 = x16724 < 0;
bool x16726 = x16724 >= x16616;
bool x16727 = x16725 || x16726;
if (x16727) {
int32_t x16728 = x16722 * x16678;
float* x16729 = x16717+x16728;
memset(x16729, 0, 4 * x16678);;
} else {
if (x16743) {
int32_t x16728 = x16722 * x16678;
float* x16744 = x16717+x16728;
memset(x16744, 0, 4 * x16678);;
} else {
int32_t x16728 = x16722 * x16678;
if (x16747) {
float* x16748 = x16717+x16728;
memset(x16748, 0, 4 * x16735);;
} else {
}
// may have segfault here
int32_t x16753 = x16728 + x16735;
float* x16754 = x16717+x16753;
int32_t x16755 = x16724 * x16616;
int32_t x16756 = x16755 + x16732;
int32_t x16757 = x16756 + x16735;
float* x16758 = x16720+x16757;
memcpy(x16754, x16758, 4 * x16742);;
if (x16760) {
int32_t x16761 = x16728 + x16678;
int32_t x16762 = x16761 - x16740;
float* x16763 = x16717+x16762;
memset(x16763, 0, 4 * x16740);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x16679,x16685,1,x249,x16685,x16697,x16679,1,x16695,x16679);

}
int32_t x16778 = 0;
int32_t x16779 = 1;
x16779 *= 1;
x16778 += 1;
x16779 *= 1;
x16779 *= 1;
int32_t x16784 = x16778;
bool x16785 = x16784 >= 2;
if (x16785) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16790 = x16784 == 0;
if (x16790) {
int32_t x16791 = x16779;
bool x16792 = x16791 == 256;
if (x16792) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16799 = x16779;
int32_t x16800 = 256 / x16799;
bool x16801 = x16800 == 1;
bool x16804;
if (x454) {
bool x16802 = 256 == x16800;
bool x16803 = x16801 || x16802;
x16804 = x16803;
} else {
x16804 = false;
}
bool x16808;
if (x16804) {
x16808 = x16807;
} else {
x16808 = false;
}
bool x16809;
if (x16808) {
x16809 = x16807;
} else {
x16809 = false;
}
if (x16809) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x16678,x16678,1,x16800,1,1);
assert(false && "");
}
bool x16815 = 256 <= x16800;
int32_t x16816;
if (x16815) {
x16816 = x16800;
} else {
x16816 = 256;
}
int32_t x16825 = x16816 * x16824;
int32_t x16826 = 64 * x16825;
float* x16827 = (float*)myMalloc(x16826 * sizeof(float));;
int32_t x16830;
if (x16801) {
x16830 = 0;
} else {
x16830 = 1;
}
for(int x16831=0; x16831 < 64; x16831++) {
int32_t x16843 = x16680 * x16831;
int32_t x16837 = x16825 * x16831;
for(int x16833=0; x16833 < x16816; x16833++) {
int32_t x16844 = x16679 * x16833;
int32_t x16845 = x16843 + x16844;
int32_t x16850 = x16830 * x16833;
int32_t x16839 = x16824 * x16833;
for(int x16835=0; x16835 < x16818; x16835++) {
int32_t x16846 = x16828 * x16835;
int32_t x16847 = x16845 + x16846;
int32_t x16841 = x16818 * x16835;
for(int x16836=0; x16836 < x16818; x16836++) {
int32_t x16848 = x16829 * x16836;
int32_t x16849 = x16847 + x16848;
float x16851 = x16684[x16849];
float x16852 = x186[x16850];
int32_t x16838 = x16836 + x16837;
int32_t x16840 = x16838 + x16839;
int32_t x16842 = x16840 + x16841;
float x16853 = x16851 - x16852;
x16827[x16842] = x16853;

}

}

}

}
float* x16863 = (float*)myMalloc(256 * sizeof(float));;
for(int x16864=0; x16864 < 256; x16864++) {
float x16865 = x230[x16864];
float x16866 = x16865 + 1.0E-5f;
x16863[x16864] = x16866;

}
float* x16870 = (float*)myMalloc(256 * sizeof(float));;
for(int x16871=0; x16871 < 256; x16871++) {
float x16872 = x16863[x16871];
double x16873 = (double)x16872;
double x16874 = sqrt(x16873);
float x16875 = (float)x16874;
x16870[x16871] = x16875;

}
int32_t x16879 = 0;
int32_t x16880 = 1;
x16880 *= 1;
x16879 += 1;
x16880 *= 1;
x16880 *= 1;
int32_t x16885 = x16879;
bool x16886 = x16885 >= 2;
if (x16886) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16891 = x16885 == 0;
if (x16891) {
int32_t x16892 = x16880;
bool x16893 = x16892 == 256;
if (x16893) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16900 = x16880;
bool x16902 = x16816 == 1;
int32_t x16901 = 256 / x16900;
bool x16903 = x16901 == 1;
bool x16907;
if (x454) {
bool x16904 = x16902 || x16903;
bool x16905 = x16816 == x16901;
bool x16906 = x16904 || x16905;
x16907 = x16906;
} else {
x16907 = false;
}
bool x16911;
if (x16907) {
x16911 = x16910;
} else {
x16911 = false;
}
bool x16912;
if (x16911) {
x16912 = x16910;
} else {
x16912 = false;
}
if (x16912) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16816,x16818,x16818,1,x16901,1,1);
assert(false && "");
}
bool x16918 = x16816 <= x16901;
int32_t x16919;
if (x16918) {
x16919 = x16901;
} else {
x16919 = x16816;
}
int32_t x16928 = x16919 * x16927;
int32_t x16929 = 64 * x16928;
float* x16930 = (float*)myMalloc(x16929 * sizeof(float));;
int32_t x16931;
if (x16902) {
x16931 = 0;
} else {
x16931 = x16824;
}
int32_t x16934;
if (x16903) {
x16934 = 0;
} else {
x16934 = 1;
}
for(int x16935=0; x16935 < 64; x16935++) {
int32_t x16947 = x16825 * x16935;
int32_t x16941 = x16928 * x16935;
for(int x16937=0; x16937 < x16919; x16937++) {
int32_t x16948 = x16931 * x16937;
int32_t x16949 = x16947 + x16948;
int32_t x16954 = x16934 * x16937;
int32_t x16943 = x16927 * x16937;
for(int x16939=0; x16939 < x16921; x16939++) {
int32_t x16950 = x16932 * x16939;
int32_t x16951 = x16949 + x16950;
int32_t x16945 = x16921 * x16939;
for(int x16940=0; x16940 < x16921; x16940++) {
int32_t x16952 = x16933 * x16940;
int32_t x16953 = x16951 + x16952;
float x16955 = x16827[x16953];
float x16956 = x16870[x16954];
int32_t x16942 = x16940 + x16941;
int32_t x16944 = x16942 + x16943;
int32_t x16946 = x16944 + x16945;
float x16957 = x16955 / x16956;
x16930[x16946] = x16957;

}

}

}

}
int32_t x16967 = 0;
int32_t x16968 = 1;
x16968 *= 1;
x16967 += 1;
x16968 *= 1;
x16968 *= 1;
int32_t x16973 = x16967;
bool x16974 = x16973 >= 2;
if (x16974) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16979 = x16973 == 0;
if (x16979) {
int32_t x16980 = x16968;
bool x16981 = x16980 == 256;
if (x16981) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16988 = x16968;
bool x16990 = x16919 == 1;
int32_t x16989 = 256 / x16988;
bool x16991 = x16989 == 1;
bool x16995;
if (x454) {
bool x16992 = x16990 || x16991;
bool x16993 = x16919 == x16989;
bool x16994 = x16992 || x16993;
x16995 = x16994;
} else {
x16995 = false;
}
bool x16999;
if (x16995) {
x16999 = x16998;
} else {
x16999 = false;
}
bool x17000;
if (x16999) {
x17000 = x16998;
} else {
x17000 = false;
}
if (x17000) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16919,x16921,x16921,1,x16989,1,1);
assert(false && "");
}
bool x17006 = x16919 <= x16989;
int32_t x17007;
if (x17006) {
x17007 = x16989;
} else {
x17007 = x16919;
}
int32_t x17016 = x17007 * x17015;
int32_t x17017 = 64 * x17016;
float* x17018 = (float*)myMalloc(x17017 * sizeof(float));;
int32_t x17019;
if (x16990) {
x17019 = 0;
} else {
x17019 = x16927;
}
int32_t x17022;
if (x16991) {
x17022 = 0;
} else {
x17022 = 1;
}
for(int x17023=0; x17023 < 64; x17023++) {
int32_t x17035 = x16928 * x17023;
int32_t x17029 = x17016 * x17023;
for(int x17025=0; x17025 < x17007; x17025++) {
int32_t x17036 = x17019 * x17025;
int32_t x17037 = x17035 + x17036;
int32_t x17042 = x17022 * x17025;
int32_t x17031 = x17015 * x17025;
for(int x17027=0; x17027 < x17009; x17027++) {
int32_t x17038 = x17020 * x17027;
int32_t x17039 = x17037 + x17038;
int32_t x17033 = x17009 * x17027;
for(int x17028=0; x17028 < x17009; x17028++) {
int32_t x17040 = x17021 * x17028;
int32_t x17041 = x17039 + x17040;
float x17043 = x16930[x17041];
float x17044 = x74[x17042];
int32_t x17030 = x17028 + x17029;
int32_t x17032 = x17030 + x17031;
int32_t x17034 = x17032 + x17033;
float x17045 = x17043 * x17044;
x17018[x17034] = x17045;

}

}

}

}
int32_t x17055 = 0;
int32_t x17056 = 1;
x17056 *= 1;
x17055 += 1;
x17056 *= 1;
x17056 *= 1;
int32_t x17061 = x17055;
bool x17062 = x17061 >= 2;
if (x17062) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17067 = x17061 == 0;
if (x17067) {
int32_t x17068 = x17056;
bool x17069 = x17068 == 256;
if (x17069) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17076 = x17056;
bool x17078 = x17007 == 1;
int32_t x17077 = 256 / x17076;
bool x17079 = x17077 == 1;
bool x17083;
if (x454) {
bool x17080 = x17078 || x17079;
bool x17081 = x17007 == x17077;
bool x17082 = x17080 || x17081;
x17083 = x17082;
} else {
x17083 = false;
}
bool x17087;
if (x17083) {
x17087 = x17086;
} else {
x17087 = false;
}
bool x17088;
if (x17087) {
x17088 = x17086;
} else {
x17088 = false;
}
if (x17088) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17007,x17009,x17009,1,x17077,1,1);
assert(false && "");
}
bool x17094 = x17007 <= x17077;
int32_t x17095;
if (x17094) {
x17095 = x17077;
} else {
x17095 = x17007;
}
int32_t x17104 = x17095 * x17103;
int32_t x17105 = 64 * x17104;
float* x17106 = (float*)myMalloc(x17105 * sizeof(float));;
int32_t x17107;
if (x17078) {
x17107 = 0;
} else {
x17107 = x17015;
}
int32_t x17110;
if (x17079) {
x17110 = 0;
} else {
x17110 = 1;
}
for(int x17111=0; x17111 < 64; x17111++) {
int32_t x17123 = x17016 * x17111;
int32_t x17117 = x17104 * x17111;
for(int x17113=0; x17113 < x17095; x17113++) {
int32_t x17124 = x17107 * x17113;
int32_t x17125 = x17123 + x17124;
int32_t x17130 = x17110 * x17113;
int32_t x17119 = x17103 * x17113;
for(int x17115=0; x17115 < x17097; x17115++) {
int32_t x17126 = x17108 * x17115;
int32_t x17127 = x17125 + x17126;
int32_t x17121 = x17097 * x17115;
for(int x17116=0; x17116 < x17097; x17116++) {
int32_t x17128 = x17109 * x17116;
int32_t x17129 = x17127 + x17128;
float x17131 = x17018[x17129];
float x17132 = x136[x17130];
int32_t x17118 = x17116 + x17117;
int32_t x17120 = x17118 + x17119;
int32_t x17122 = x17120 + x17121;
float x17133 = x17131 + x17132;
x17106[x17122] = x17133;

}

}

}

}
float* x17143 = (float*)myMalloc(x17105 * sizeof(float));;
for(int x17145=0; x17145 < x17105; x17145++) {
float x17146 = x17106[x17145];
bool x17147 = x17146 < 0.0f;
if (x17147) {
x17143[x17145] = 0.0f;
} else {
float x17150 = x17106[x17145];
x17143[x17145] = x17150;
}

}
float* x17164 = (float*)myMalloc(x17163 * sizeof(float));;
int32_t x17167 = 64 * x17095;
int32_t x17168 = x17167 * x17159;
float* x17169 = (float*)myMalloc(x17168 * sizeof(float));;
int32_t x17165 = x17095 * x17159;
for(int x17170=0; x17170 < 64; x17170++) {
int32_t x17171 = x17170 * x17104;
float* x17172 = x17143+x17171;
int32_t x17173 = x17170 * x17160;
float* x17174 = x17164+x17173;
int32_t x17175 = x17170 * x17165;
float* x17176 = x17169+x17175;
for(int x17177=0; x17177 < x17095; x17177++) {
int32_t x17178 = x17177 / 1;
int32_t x17182 = x17178 * x17158;
int32_t x17183 = x17182 * x17158;
int32_t x17179 = x17177 % 1;
int32_t x17180 = x17179 / 1;
int32_t x17184 = x17180 * x17158;
int32_t x17185 = x17184 * x17158;
int32_t x17186 = x17183 + x17185;
int32_t x17181 = x17179 % 1;
int32_t x17187 = x17181 * x17158;
int32_t x17188 = x17187 * x17158;
int32_t x17189 = x17186 + x17188;
float* x17190 = x17176+x17189;
int32_t x17191 = x17178 * x17097;
int32_t x17192 = x17191 * x17097;
float* x17193 = x17172+x17192;
for(int x17195=0; x17195 < x17158; x17195++) {
int32_t x17197 = x17195 * x17158;
float* x17198 = x17190+x17197;
int32_t x17196 = x17195 + x17180;
int32_t x17199 = x17196 * x17097;
int32_t x17200 = x17199 + x17181;
float* x17201 = x17193+x17200;
memcpy(x17198, x17201, 4 * x17158);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x17159,x17095,1,x89,x17095,x17176,x17159,1,x17174,x17159);

}
int32_t x17210 = 0;
int32_t x17211 = 1;
x17211 *= 1;
x17210 += 1;
x17211 *= 1;
x17211 *= 1;
int32_t x17216 = x17210;
bool x17217 = x17216 >= 2;
if (x17217) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17222 = x17216 == 0;
if (x17222) {
int32_t x17223 = x17211;
bool x17224 = x17223 == 1024;
if (x17224) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17231 = x17211;
int32_t x17232 = 1024 / x17231;
bool x17233 = x17232 == 1;
bool x17236;
if (x454) {
bool x17234 = 1024 == x17232;
bool x17235 = x17233 || x17234;
x17236 = x17235;
} else {
x17236 = false;
}
bool x17240;
if (x17236) {
x17240 = x17239;
} else {
x17240 = false;
}
bool x17241;
if (x17240) {
x17241 = x17239;
} else {
x17241 = false;
}
if (x17241) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x17158,x17158,1,x17232,1,1);
assert(false && "");
}
bool x17247 = 1024 <= x17232;
int32_t x17248;
if (x17247) {
x17248 = x17232;
} else {
x17248 = 1024;
}
int32_t x17257 = x17248 * x17256;
int32_t x17258 = 64 * x17257;
float* x17259 = (float*)myMalloc(x17258 * sizeof(float));;
int32_t x17262;
if (x17233) {
x17262 = 0;
} else {
x17262 = 1;
}
for(int x17263=0; x17263 < 64; x17263++) {
int32_t x17275 = x17160 * x17263;
int32_t x17269 = x17257 * x17263;
for(int x17265=0; x17265 < x17248; x17265++) {
int32_t x17276 = x17159 * x17265;
int32_t x17277 = x17275 + x17276;
int32_t x17282 = x17262 * x17265;
int32_t x17271 = x17256 * x17265;
for(int x17267=0; x17267 < x17250; x17267++) {
int32_t x17278 = x17260 * x17267;
int32_t x17279 = x17277 + x17278;
int32_t x17273 = x17250 * x17267;
for(int x17268=0; x17268 < x17250; x17268++) {
int32_t x17280 = x17261 * x17268;
int32_t x17281 = x17279 + x17280;
float x17283 = x17164[x17281];
float x17284 = x231[x17282];
int32_t x17270 = x17268 + x17269;
int32_t x17272 = x17270 + x17271;
int32_t x17274 = x17272 + x17273;
float x17285 = x17283 - x17284;
x17259[x17274] = x17285;

}

}

}

}
float* x17295 = (float*)myMalloc(1024 * sizeof(float));;
for(int x17296=0; x17296 < 1024; x17296++) {
float x17297 = x161[x17296];
float x17298 = x17297 + 1.0E-5f;
x17295[x17296] = x17298;

}
float* x17302 = (float*)myMalloc(1024 * sizeof(float));;
for(int x17303=0; x17303 < 1024; x17303++) {
float x17304 = x17295[x17303];
double x17305 = (double)x17304;
double x17306 = sqrt(x17305);
float x17307 = (float)x17306;
x17302[x17303] = x17307;

}
int32_t x17311 = 0;
int32_t x17312 = 1;
x17312 *= 1;
x17311 += 1;
x17312 *= 1;
x17312 *= 1;
int32_t x17317 = x17311;
bool x17318 = x17317 >= 2;
if (x17318) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17323 = x17317 == 0;
if (x17323) {
int32_t x17324 = x17312;
bool x17325 = x17324 == 1024;
if (x17325) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17332 = x17312;
bool x17334 = x17248 == 1;
int32_t x17333 = 1024 / x17332;
bool x17335 = x17333 == 1;
bool x17339;
if (x454) {
bool x17336 = x17334 || x17335;
bool x17337 = x17248 == x17333;
bool x17338 = x17336 || x17337;
x17339 = x17338;
} else {
x17339 = false;
}
bool x17343;
if (x17339) {
x17343 = x17342;
} else {
x17343 = false;
}
bool x17344;
if (x17343) {
x17344 = x17342;
} else {
x17344 = false;
}
if (x17344) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17248,x17250,x17250,1,x17333,1,1);
assert(false && "");
}
bool x17350 = x17248 <= x17333;
int32_t x17351;
if (x17350) {
x17351 = x17333;
} else {
x17351 = x17248;
}
int32_t x17360 = x17351 * x17359;
int32_t x17361 = 64 * x17360;
float* x17362 = (float*)myMalloc(x17361 * sizeof(float));;
int32_t x17363;
if (x17334) {
x17363 = 0;
} else {
x17363 = x17256;
}
int32_t x17366;
if (x17335) {
x17366 = 0;
} else {
x17366 = 1;
}
for(int x17367=0; x17367 < 64; x17367++) {
int32_t x17379 = x17257 * x17367;
int32_t x17373 = x17360 * x17367;
for(int x17369=0; x17369 < x17351; x17369++) {
int32_t x17380 = x17363 * x17369;
int32_t x17381 = x17379 + x17380;
int32_t x17386 = x17366 * x17369;
int32_t x17375 = x17359 * x17369;
for(int x17371=0; x17371 < x17353; x17371++) {
int32_t x17382 = x17364 * x17371;
int32_t x17383 = x17381 + x17382;
int32_t x17377 = x17353 * x17371;
for(int x17372=0; x17372 < x17353; x17372++) {
int32_t x17384 = x17365 * x17372;
int32_t x17385 = x17383 + x17384;
float x17387 = x17259[x17385];
float x17388 = x17302[x17386];
int32_t x17374 = x17372 + x17373;
int32_t x17376 = x17374 + x17375;
int32_t x17378 = x17376 + x17377;
float x17389 = x17387 / x17388;
x17362[x17378] = x17389;

}

}

}

}
int32_t x17399 = 0;
int32_t x17400 = 1;
x17400 *= 1;
x17399 += 1;
x17400 *= 1;
x17400 *= 1;
int32_t x17405 = x17399;
bool x17406 = x17405 >= 2;
if (x17406) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17411 = x17405 == 0;
if (x17411) {
int32_t x17412 = x17400;
bool x17413 = x17412 == 1024;
if (x17413) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17420 = x17400;
bool x17422 = x17351 == 1;
int32_t x17421 = 1024 / x17420;
bool x17423 = x17421 == 1;
bool x17427;
if (x454) {
bool x17424 = x17422 || x17423;
bool x17425 = x17351 == x17421;
bool x17426 = x17424 || x17425;
x17427 = x17426;
} else {
x17427 = false;
}
bool x17431;
if (x17427) {
x17431 = x17430;
} else {
x17431 = false;
}
bool x17432;
if (x17431) {
x17432 = x17430;
} else {
x17432 = false;
}
if (x17432) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17351,x17353,x17353,1,x17421,1,1);
assert(false && "");
}
bool x17438 = x17351 <= x17421;
int32_t x17439;
if (x17438) {
x17439 = x17421;
} else {
x17439 = x17351;
}
int32_t x17448 = x17439 * x17447;
int32_t x17449 = 64 * x17448;
float* x17450 = (float*)myMalloc(x17449 * sizeof(float));;
int32_t x17451;
if (x17422) {
x17451 = 0;
} else {
x17451 = x17359;
}
int32_t x17454;
if (x17423) {
x17454 = 0;
} else {
x17454 = 1;
}
for(int x17455=0; x17455 < 64; x17455++) {
int32_t x17467 = x17360 * x17455;
int32_t x17461 = x17448 * x17455;
for(int x17457=0; x17457 < x17439; x17457++) {
int32_t x17468 = x17451 * x17457;
int32_t x17469 = x17467 + x17468;
int32_t x17474 = x17454 * x17457;
int32_t x17463 = x17447 * x17457;
for(int x17459=0; x17459 < x17441; x17459++) {
int32_t x17470 = x17452 * x17459;
int32_t x17471 = x17469 + x17470;
int32_t x17465 = x17441 * x17459;
for(int x17460=0; x17460 < x17441; x17460++) {
int32_t x17472 = x17453 * x17460;
int32_t x17473 = x17471 + x17472;
float x17475 = x17362[x17473];
float x17476 = x238[x17474];
int32_t x17462 = x17460 + x17461;
int32_t x17464 = x17462 + x17463;
int32_t x17466 = x17464 + x17465;
float x17477 = x17475 * x17476;
x17450[x17466] = x17477;

}

}

}

}
int32_t x17487 = 0;
int32_t x17488 = 1;
x17488 *= 1;
x17487 += 1;
x17488 *= 1;
x17488 *= 1;
int32_t x17493 = x17487;
bool x17494 = x17493 >= 2;
if (x17494) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17499 = x17493 == 0;
if (x17499) {
int32_t x17500 = x17488;
bool x17501 = x17500 == 1024;
if (x17501) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17508 = x17488;
bool x17510 = x17439 == 1;
int32_t x17509 = 1024 / x17508;
bool x17511 = x17509 == 1;
bool x17515;
if (x454) {
bool x17512 = x17510 || x17511;
bool x17513 = x17439 == x17509;
bool x17514 = x17512 || x17513;
x17515 = x17514;
} else {
x17515 = false;
}
bool x17519;
if (x17515) {
x17519 = x17518;
} else {
x17519 = false;
}
bool x17520;
if (x17519) {
x17520 = x17518;
} else {
x17520 = false;
}
if (x17520) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17439,x17441,x17441,1,x17509,1,1);
assert(false && "");
}
bool x17526 = x17439 <= x17509;
int32_t x17527;
if (x17526) {
x17527 = x17509;
} else {
x17527 = x17439;
}
int32_t x17536 = x17527 * x17535;
int32_t x17537 = 64 * x17536;
float* x17538 = (float*)myMalloc(x17537 * sizeof(float));;
int32_t x17539;
if (x17510) {
x17539 = 0;
} else {
x17539 = x17447;
}
int32_t x17542;
if (x17511) {
x17542 = 0;
} else {
x17542 = 1;
}
for(int x17543=0; x17543 < 64; x17543++) {
int32_t x17555 = x17448 * x17543;
int32_t x17549 = x17536 * x17543;
for(int x17545=0; x17545 < x17527; x17545++) {
int32_t x17556 = x17539 * x17545;
int32_t x17557 = x17555 + x17556;
int32_t x17562 = x17542 * x17545;
int32_t x17551 = x17535 * x17545;
for(int x17547=0; x17547 < x17529; x17547++) {
int32_t x17558 = x17540 * x17547;
int32_t x17559 = x17557 + x17558;
int32_t x17553 = x17529 * x17547;
for(int x17548=0; x17548 < x17529; x17548++) {
int32_t x17560 = x17541 * x17548;
int32_t x17561 = x17559 + x17560;
float x17563 = x17450[x17561];
float x17564 = x146[x17562];
int32_t x17550 = x17548 + x17549;
int32_t x17552 = x17550 + x17551;
int32_t x17554 = x17552 + x17553;
float x17565 = x17563 + x17564;
x17538[x17554] = x17565;

}

}

}

}
bool x17575 = x17527 == 1;
bool x17576 = x17575 || x16163;
bool x17577 = x17527 == x16115;
bool x17578 = x17576 || x17577;
bool x17583;
if (x17578) {
x17583 = x17582;
} else {
x17583 = false;
}
bool x17584;
if (x17583) {
x17584 = x17582;
} else {
x17584 = false;
}
if (x17584) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17527,x17529,x17529,64,x16115,x16117,x16117);
assert(false && "");
}
bool x17590 = x17527 <= x16115;
int32_t x17591;
if (x17590) {
x17591 = x16115;
} else {
x17591 = x17527;
}
int32_t x17607;
if (x17575) {
x17607 = 0;
} else {
x17607 = x17535;
}
for(int x17610=0; x17610 < 64; x17610++) {
int32_t x17616 = x17536 * x17610;
int32_t x17623 = x16124 * x17610;
for(int x17612=0; x17612 < x17591; x17612++) {
int32_t x17617 = x17607 * x17612;
int32_t x17618 = x17616 + x17617;
int32_t x17624 = x16195 * x17612;
int32_t x17625 = x17623 + x17624;
for(int x17614=0; x17614 < x17593; x17614++) {
int32_t x17619 = x17608 * x17614;
int32_t x17620 = x17618 + x17619;
int32_t x17626 = x16196 * x17614;
int32_t x17627 = x17625 + x17626;
for(int x17615=0; x17615 < x17593; x17615++) {
int32_t x17621 = x17609 * x17615;
int32_t x17622 = x17620 + x17621;
float x17630 = x17538[x17622];
int32_t x17628 = x16197 * x17615;
int32_t x17629 = x17627 + x17628;
float x17631 = x16230[x17629];
float x17632 = x17630 + x17631;
x17538[x17622] = x17632;

}

}

}

}
float* x17642 = (float*)myMalloc(x17537 * sizeof(float));;
for(int x17644=0; x17644 < x17537; x17644++) {
float x17645 = x17538[x17644];
bool x17646 = x17645 < 0.0f;
if (x17646) {
x17642[x17644] = 0.0f;
} else {
float x17649 = x17538[x17644];
x17642[x17644] = x17649;
}

}
float* x17663 = (float*)myMalloc(x17662 * sizeof(float));;
int32_t x17666 = 64 * x17527;
int32_t x17667 = x17666 * x17658;
float* x17668 = (float*)myMalloc(x17667 * sizeof(float));;
int32_t x17664 = x17527 * x17658;
for(int x17669=0; x17669 < 64; x17669++) {
int32_t x17670 = x17669 * x17536;
float* x17671 = x17642+x17670;
int32_t x17672 = x17669 * x17659;
float* x17673 = x17663+x17672;
int32_t x17674 = x17669 * x17664;
float* x17675 = x17668+x17674;
for(int x17676=0; x17676 < x17527; x17676++) {
int32_t x17677 = x17676 / 1;
int32_t x17681 = x17677 * x17657;
int32_t x17682 = x17681 * x17657;
int32_t x17678 = x17676 % 1;
int32_t x17679 = x17678 / 1;
int32_t x17683 = x17679 * x17657;
int32_t x17684 = x17683 * x17657;
int32_t x17685 = x17682 + x17684;
int32_t x17680 = x17678 % 1;
int32_t x17686 = x17680 * x17657;
int32_t x17687 = x17686 * x17657;
int32_t x17688 = x17685 + x17687;
float* x17689 = x17675+x17688;
int32_t x17690 = x17677 * x17529;
int32_t x17691 = x17690 * x17529;
float* x17692 = x17671+x17691;
for(int x17694=0; x17694 < x17657; x17694++) {
int32_t x17696 = x17694 * x17657;
float* x17697 = x17689+x17696;
int32_t x17695 = x17694 + x17679;
int32_t x17698 = x17695 * x17529;
int32_t x17699 = x17698 + x17680;
float* x17700 = x17692+x17699;
memcpy(x17697, x17700, 4 * x17657);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x17658,x17527,1,x22,x17527,x17675,x17658,1,x17673,x17658);

}
int32_t x17709 = 0;
int32_t x17710 = 1;
x17710 *= 1;
x17709 += 1;
x17710 *= 1;
x17710 *= 1;
int32_t x17715 = x17709;
bool x17716 = x17715 >= 2;
if (x17716) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17721 = x17715 == 0;
if (x17721) {
int32_t x17722 = x17710;
bool x17723 = x17722 == 256;
if (x17723) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17730 = x17710;
int32_t x17731 = 256 / x17730;
bool x17732 = x17731 == 1;
bool x17735;
if (x454) {
bool x17733 = 256 == x17731;
bool x17734 = x17732 || x17733;
x17735 = x17734;
} else {
x17735 = false;
}
bool x17739;
if (x17735) {
x17739 = x17738;
} else {
x17739 = false;
}
bool x17740;
if (x17739) {
x17740 = x17738;
} else {
x17740 = false;
}
if (x17740) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x17657,x17657,1,x17731,1,1);
assert(false && "");
}
bool x17746 = 256 <= x17731;
int32_t x17747;
if (x17746) {
x17747 = x17731;
} else {
x17747 = 256;
}
int32_t x17756 = x17747 * x17755;
int32_t x17757 = 64 * x17756;
float* x17758 = (float*)myMalloc(x17757 * sizeof(float));;
int32_t x17761;
if (x17732) {
x17761 = 0;
} else {
x17761 = 1;
}
for(int x17762=0; x17762 < 64; x17762++) {
int32_t x17774 = x17659 * x17762;
int32_t x17768 = x17756 * x17762;
for(int x17764=0; x17764 < x17747; x17764++) {
int32_t x17775 = x17658 * x17764;
int32_t x17776 = x17774 + x17775;
int32_t x17781 = x17761 * x17764;
int32_t x17770 = x17755 * x17764;
for(int x17766=0; x17766 < x17749; x17766++) {
int32_t x17777 = x17759 * x17766;
int32_t x17778 = x17776 + x17777;
int32_t x17772 = x17749 * x17766;
for(int x17767=0; x17767 < x17749; x17767++) {
int32_t x17779 = x17760 * x17767;
int32_t x17780 = x17778 + x17779;
float x17782 = x17663[x17780];
float x17783 = x254[x17781];
int32_t x17769 = x17767 + x17768;
int32_t x17771 = x17769 + x17770;
int32_t x17773 = x17771 + x17772;
float x17784 = x17782 - x17783;
x17758[x17773] = x17784;

}

}

}

}
float* x17794 = (float*)myMalloc(256 * sizeof(float));;
for(int x17795=0; x17795 < 256; x17795++) {
float x17796 = x69[x17795];
float x17797 = x17796 + 1.0E-5f;
x17794[x17795] = x17797;

}
float* x17801 = (float*)myMalloc(256 * sizeof(float));;
for(int x17802=0; x17802 < 256; x17802++) {
float x17803 = x17794[x17802];
double x17804 = (double)x17803;
double x17805 = sqrt(x17804);
float x17806 = (float)x17805;
x17801[x17802] = x17806;

}
int32_t x17810 = 0;
int32_t x17811 = 1;
x17811 *= 1;
x17810 += 1;
x17811 *= 1;
x17811 *= 1;
int32_t x17816 = x17810;
bool x17817 = x17816 >= 2;
if (x17817) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17822 = x17816 == 0;
if (x17822) {
int32_t x17823 = x17811;
bool x17824 = x17823 == 256;
if (x17824) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17831 = x17811;
bool x17833 = x17747 == 1;
int32_t x17832 = 256 / x17831;
bool x17834 = x17832 == 1;
bool x17838;
if (x454) {
bool x17835 = x17833 || x17834;
bool x17836 = x17747 == x17832;
bool x17837 = x17835 || x17836;
x17838 = x17837;
} else {
x17838 = false;
}
bool x17842;
if (x17838) {
x17842 = x17841;
} else {
x17842 = false;
}
bool x17843;
if (x17842) {
x17843 = x17841;
} else {
x17843 = false;
}
if (x17843) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17747,x17749,x17749,1,x17832,1,1);
assert(false && "");
}
bool x17849 = x17747 <= x17832;
int32_t x17850;
if (x17849) {
x17850 = x17832;
} else {
x17850 = x17747;
}
int32_t x17859 = x17850 * x17858;
int32_t x17860 = 64 * x17859;
float* x17861 = (float*)myMalloc(x17860 * sizeof(float));;
int32_t x17862;
if (x17833) {
x17862 = 0;
} else {
x17862 = x17755;
}
int32_t x17865;
if (x17834) {
x17865 = 0;
} else {
x17865 = 1;
}
for(int x17866=0; x17866 < 64; x17866++) {
int32_t x17878 = x17756 * x17866;
int32_t x17872 = x17859 * x17866;
for(int x17868=0; x17868 < x17850; x17868++) {
int32_t x17879 = x17862 * x17868;
int32_t x17880 = x17878 + x17879;
int32_t x17885 = x17865 * x17868;
int32_t x17874 = x17858 * x17868;
for(int x17870=0; x17870 < x17852; x17870++) {
int32_t x17881 = x17863 * x17870;
int32_t x17882 = x17880 + x17881;
int32_t x17876 = x17852 * x17870;
for(int x17871=0; x17871 < x17852; x17871++) {
int32_t x17883 = x17864 * x17871;
int32_t x17884 = x17882 + x17883;
float x17886 = x17758[x17884];
float x17887 = x17801[x17885];
int32_t x17873 = x17871 + x17872;
int32_t x17875 = x17873 + x17874;
int32_t x17877 = x17875 + x17876;
float x17888 = x17886 / x17887;
x17861[x17877] = x17888;

}

}

}

}
int32_t x17898 = 0;
int32_t x17899 = 1;
x17899 *= 1;
x17898 += 1;
x17899 *= 1;
x17899 *= 1;
int32_t x17904 = x17898;
bool x17905 = x17904 >= 2;
if (x17905) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17910 = x17904 == 0;
if (x17910) {
int32_t x17911 = x17899;
bool x17912 = x17911 == 256;
if (x17912) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17919 = x17899;
bool x17921 = x17850 == 1;
int32_t x17920 = 256 / x17919;
bool x17922 = x17920 == 1;
bool x17926;
if (x454) {
bool x17923 = x17921 || x17922;
bool x17924 = x17850 == x17920;
bool x17925 = x17923 || x17924;
x17926 = x17925;
} else {
x17926 = false;
}
bool x17930;
if (x17926) {
x17930 = x17929;
} else {
x17930 = false;
}
bool x17931;
if (x17930) {
x17931 = x17929;
} else {
x17931 = false;
}
if (x17931) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17850,x17852,x17852,1,x17920,1,1);
assert(false && "");
}
bool x17937 = x17850 <= x17920;
int32_t x17938;
if (x17937) {
x17938 = x17920;
} else {
x17938 = x17850;
}
int32_t x17947 = x17938 * x17946;
int32_t x17948 = 64 * x17947;
float* x17949 = (float*)myMalloc(x17948 * sizeof(float));;
int32_t x17950;
if (x17921) {
x17950 = 0;
} else {
x17950 = x17858;
}
int32_t x17953;
if (x17922) {
x17953 = 0;
} else {
x17953 = 1;
}
for(int x17954=0; x17954 < 64; x17954++) {
int32_t x17966 = x17859 * x17954;
int32_t x17960 = x17947 * x17954;
for(int x17956=0; x17956 < x17938; x17956++) {
int32_t x17967 = x17950 * x17956;
int32_t x17968 = x17966 + x17967;
int32_t x17973 = x17953 * x17956;
int32_t x17962 = x17946 * x17956;
for(int x17958=0; x17958 < x17940; x17958++) {
int32_t x17969 = x17951 * x17958;
int32_t x17970 = x17968 + x17969;
int32_t x17964 = x17940 * x17958;
for(int x17959=0; x17959 < x17940; x17959++) {
int32_t x17971 = x17952 * x17959;
int32_t x17972 = x17970 + x17971;
float x17974 = x17861[x17972];
float x17975 = x77[x17973];
int32_t x17961 = x17959 + x17960;
int32_t x17963 = x17961 + x17962;
int32_t x17965 = x17963 + x17964;
float x17976 = x17974 * x17975;
x17949[x17965] = x17976;

}

}

}

}
int32_t x17986 = 0;
int32_t x17987 = 1;
x17987 *= 1;
x17986 += 1;
x17987 *= 1;
x17987 *= 1;
int32_t x17992 = x17986;
bool x17993 = x17992 >= 2;
if (x17993) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17998 = x17992 == 0;
if (x17998) {
int32_t x17999 = x17987;
bool x18000 = x17999 == 256;
if (x18000) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18007 = x17987;
bool x18009 = x17938 == 1;
int32_t x18008 = 256 / x18007;
bool x18010 = x18008 == 1;
bool x18014;
if (x454) {
bool x18011 = x18009 || x18010;
bool x18012 = x17938 == x18008;
bool x18013 = x18011 || x18012;
x18014 = x18013;
} else {
x18014 = false;
}
bool x18018;
if (x18014) {
x18018 = x18017;
} else {
x18018 = false;
}
bool x18019;
if (x18018) {
x18019 = x18017;
} else {
x18019 = false;
}
if (x18019) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17938,x17940,x17940,1,x18008,1,1);
assert(false && "");
}
bool x18025 = x17938 <= x18008;
int32_t x18026;
if (x18025) {
x18026 = x18008;
} else {
x18026 = x17938;
}
int32_t x18035 = x18026 * x18034;
int32_t x18036 = 64 * x18035;
float* x18037 = (float*)myMalloc(x18036 * sizeof(float));;
int32_t x18038;
if (x18009) {
x18038 = 0;
} else {
x18038 = x17946;
}
int32_t x18041;
if (x18010) {
x18041 = 0;
} else {
x18041 = 1;
}
for(int x18042=0; x18042 < 64; x18042++) {
int32_t x18054 = x17947 * x18042;
int32_t x18048 = x18035 * x18042;
for(int x18044=0; x18044 < x18026; x18044++) {
int32_t x18055 = x18038 * x18044;
int32_t x18056 = x18054 + x18055;
int32_t x18061 = x18041 * x18044;
int32_t x18050 = x18034 * x18044;
for(int x18046=0; x18046 < x18028; x18046++) {
int32_t x18057 = x18039 * x18046;
int32_t x18058 = x18056 + x18057;
int32_t x18052 = x18028 * x18046;
for(int x18047=0; x18047 < x18028; x18047++) {
int32_t x18059 = x18040 * x18047;
int32_t x18060 = x18058 + x18059;
float x18062 = x17949[x18060];
float x18063 = x185[x18061];
int32_t x18049 = x18047 + x18048;
int32_t x18051 = x18049 + x18050;
int32_t x18053 = x18051 + x18052;
float x18064 = x18062 + x18063;
x18037[x18053] = x18064;

}

}

}

}
float* x18074 = (float*)myMalloc(x18036 * sizeof(float));;
for(int x18076=0; x18076 < x18036; x18076++) {
float x18077 = x18037[x18076];
bool x18078 = x18077 < 0.0f;
if (x18078) {
x18074[x18076] = 0.0f;
} else {
float x18081 = x18037[x18076];
x18074[x18076] = x18081;
}

}
float* x18096 = (float*)myMalloc(x18095 * sizeof(float));;
int32_t x18097 = 9 * x18026;
int32_t x18100 = 64 * x18097;
int32_t x18101 = x18100 * x18091;
float* x18102 = (float*)myMalloc(x18101 * sizeof(float));;
int32_t x18098 = x18097 * x18091;
int32_t x18110 = x18026 * 3;
int32_t x18111 = x18110 * 3;
for(int x18103=0; x18103 < 64; x18103++) {
int32_t x18104 = x18103 * x18035;
float* x18105 = x18074+x18104;
int32_t x18106 = x18103 * x18092;
float* x18107 = x18096+x18106;
int32_t x18108 = x18103 * x18098;
float* x18109 = x18102+x18108;
for(int x18113=0; x18113 < x18111; x18113++) {
int32_t x18114 = x18113 / 9;
int32_t x18118 = x18114 * 3;
int32_t x18119 = x18118 * 3;
int32_t x18120 = x18119 * x18090;
int32_t x18121 = x18120 * x18090;
int32_t x18115 = x18113 % 9;
int32_t x18116 = x18115 / 3;
int32_t x18122 = x18116 * 3;
int32_t x18123 = x18122 * x18090;
int32_t x18124 = x18123 * x18090;
int32_t x18125 = x18121 + x18124;
int32_t x18117 = x18115 % 3;
int32_t x18126 = x18117 * x18090;
int32_t x18127 = x18126 * x18090;
int32_t x18128 = x18125 + x18127;
float* x18129 = x18109+x18128;
int32_t x18130 = x18114 * x18028;
int32_t x18131 = x18130 * x18028;
float* x18132 = x18105+x18131;
int32_t x18145 = 1 - x18117;
bool x18146 = x18145 > 0;
int32_t x18147;
if (x18146) {
x18147 = x18145;
} else {
x18147 = 0;
}
int32_t x18148 = 3 - x18117;
int32_t x18149 = x18148 - 1;
int32_t x18150 = 1 - x18149;
bool x18151 = x18150 > 0;
int32_t x18152;
if (x18151) {
x18152 = x18150;
} else {
x18152 = 0;
}
int32_t x18153 = x18090 - x18152;
int32_t x18154 = x18153 - x18147;
bool x18155 = x18154 <= 0;
bool x18159 = x18147 > 0;
int32_t x18144 = -1 + x18117;
bool x18172 = x18152 > 0;
for(int x18134=0; x18134 < x18090; x18134++) {
int32_t x18135 = x18134 - 1;
int32_t x18136 = x18135 + x18116;
bool x18137 = x18136 < 0;
bool x18138 = x18136 >= x18028;
bool x18139 = x18137 || x18138;
if (x18139) {
int32_t x18140 = x18134 * x18090;
float* x18141 = x18129+x18140;
memset(x18141, 0, 4 * x18090);;
} else {
if (x18155) {
int32_t x18140 = x18134 * x18090;
float* x18156 = x18129+x18140;
memset(x18156, 0, 4 * x18090);;
} else {
int32_t x18140 = x18134 * x18090;
if (x18159) {
float* x18160 = x18129+x18140;
memset(x18160, 0, 4 * x18147);;
} else {
}
// may have segfault here
int32_t x18165 = x18140 + x18147;
float* x18166 = x18129+x18165;
int32_t x18167 = x18136 * x18028;
int32_t x18168 = x18167 + x18144;
int32_t x18169 = x18168 + x18147;
float* x18170 = x18132+x18169;
memcpy(x18166, x18170, 4 * x18154);;
if (x18172) {
int32_t x18173 = x18140 + x18090;
int32_t x18174 = x18173 - x18152;
float* x18175 = x18129+x18174;
memset(x18175, 0, 4 * x18152);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x18091,x18097,1,x262,x18097,x18109,x18091,1,x18107,x18091);

}
int32_t x18190 = 0;
int32_t x18191 = 1;
x18191 *= 1;
x18190 += 1;
x18191 *= 1;
x18191 *= 1;
int32_t x18196 = x18190;
bool x18197 = x18196 >= 2;
if (x18197) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18202 = x18196 == 0;
if (x18202) {
int32_t x18203 = x18191;
bool x18204 = x18203 == 256;
if (x18204) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18211 = x18191;
int32_t x18212 = 256 / x18211;
bool x18213 = x18212 == 1;
bool x18216;
if (x454) {
bool x18214 = 256 == x18212;
bool x18215 = x18213 || x18214;
x18216 = x18215;
} else {
x18216 = false;
}
bool x18220;
if (x18216) {
x18220 = x18219;
} else {
x18220 = false;
}
bool x18221;
if (x18220) {
x18221 = x18219;
} else {
x18221 = false;
}
if (x18221) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x18090,x18090,1,x18212,1,1);
assert(false && "");
}
bool x18227 = 256 <= x18212;
int32_t x18228;
if (x18227) {
x18228 = x18212;
} else {
x18228 = 256;
}
int32_t x18237 = x18228 * x18236;
int32_t x18238 = 64 * x18237;
float* x18239 = (float*)myMalloc(x18238 * sizeof(float));;
int32_t x18242;
if (x18213) {
x18242 = 0;
} else {
x18242 = 1;
}
for(int x18243=0; x18243 < 64; x18243++) {
int32_t x18255 = x18092 * x18243;
int32_t x18249 = x18237 * x18243;
for(int x18245=0; x18245 < x18228; x18245++) {
int32_t x18256 = x18091 * x18245;
int32_t x18257 = x18255 + x18256;
int32_t x18262 = x18242 * x18245;
int32_t x18251 = x18236 * x18245;
for(int x18247=0; x18247 < x18230; x18247++) {
int32_t x18258 = x18240 * x18247;
int32_t x18259 = x18257 + x18258;
int32_t x18253 = x18230 * x18247;
for(int x18248=0; x18248 < x18230; x18248++) {
int32_t x18260 = x18241 * x18248;
int32_t x18261 = x18259 + x18260;
float x18263 = x18096[x18261];
float x18264 = x250[x18262];
int32_t x18250 = x18248 + x18249;
int32_t x18252 = x18250 + x18251;
int32_t x18254 = x18252 + x18253;
float x18265 = x18263 - x18264;
x18239[x18254] = x18265;

}

}

}

}
float* x18275 = (float*)myMalloc(256 * sizeof(float));;
for(int x18276=0; x18276 < 256; x18276++) {
float x18277 = x104[x18276];
float x18278 = x18277 + 1.0E-5f;
x18275[x18276] = x18278;

}
float* x18282 = (float*)myMalloc(256 * sizeof(float));;
for(int x18283=0; x18283 < 256; x18283++) {
float x18284 = x18275[x18283];
double x18285 = (double)x18284;
double x18286 = sqrt(x18285);
float x18287 = (float)x18286;
x18282[x18283] = x18287;

}
int32_t x18291 = 0;
int32_t x18292 = 1;
x18292 *= 1;
x18291 += 1;
x18292 *= 1;
x18292 *= 1;
int32_t x18297 = x18291;
bool x18298 = x18297 >= 2;
if (x18298) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18303 = x18297 == 0;
if (x18303) {
int32_t x18304 = x18292;
bool x18305 = x18304 == 256;
if (x18305) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18312 = x18292;
bool x18314 = x18228 == 1;
int32_t x18313 = 256 / x18312;
bool x18315 = x18313 == 1;
bool x18319;
if (x454) {
bool x18316 = x18314 || x18315;
bool x18317 = x18228 == x18313;
bool x18318 = x18316 || x18317;
x18319 = x18318;
} else {
x18319 = false;
}
bool x18323;
if (x18319) {
x18323 = x18322;
} else {
x18323 = false;
}
bool x18324;
if (x18323) {
x18324 = x18322;
} else {
x18324 = false;
}
if (x18324) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18228,x18230,x18230,1,x18313,1,1);
assert(false && "");
}
bool x18330 = x18228 <= x18313;
int32_t x18331;
if (x18330) {
x18331 = x18313;
} else {
x18331 = x18228;
}
int32_t x18340 = x18331 * x18339;
int32_t x18341 = 64 * x18340;
float* x18342 = (float*)myMalloc(x18341 * sizeof(float));;
int32_t x18343;
if (x18314) {
x18343 = 0;
} else {
x18343 = x18236;
}
int32_t x18346;
if (x18315) {
x18346 = 0;
} else {
x18346 = 1;
}
for(int x18347=0; x18347 < 64; x18347++) {
int32_t x18359 = x18237 * x18347;
int32_t x18353 = x18340 * x18347;
for(int x18349=0; x18349 < x18331; x18349++) {
int32_t x18360 = x18343 * x18349;
int32_t x18361 = x18359 + x18360;
int32_t x18366 = x18346 * x18349;
int32_t x18355 = x18339 * x18349;
for(int x18351=0; x18351 < x18333; x18351++) {
int32_t x18362 = x18344 * x18351;
int32_t x18363 = x18361 + x18362;
int32_t x18357 = x18333 * x18351;
for(int x18352=0; x18352 < x18333; x18352++) {
int32_t x18364 = x18345 * x18352;
int32_t x18365 = x18363 + x18364;
float x18367 = x18239[x18365];
float x18368 = x18282[x18366];
int32_t x18354 = x18352 + x18353;
int32_t x18356 = x18354 + x18355;
int32_t x18358 = x18356 + x18357;
float x18369 = x18367 / x18368;
x18342[x18358] = x18369;

}

}

}

}
int32_t x18379 = 0;
int32_t x18380 = 1;
x18380 *= 1;
x18379 += 1;
x18380 *= 1;
x18380 *= 1;
int32_t x18385 = x18379;
bool x18386 = x18385 >= 2;
if (x18386) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18391 = x18385 == 0;
if (x18391) {
int32_t x18392 = x18380;
bool x18393 = x18392 == 256;
if (x18393) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18400 = x18380;
bool x18402 = x18331 == 1;
int32_t x18401 = 256 / x18400;
bool x18403 = x18401 == 1;
bool x18407;
if (x454) {
bool x18404 = x18402 || x18403;
bool x18405 = x18331 == x18401;
bool x18406 = x18404 || x18405;
x18407 = x18406;
} else {
x18407 = false;
}
bool x18411;
if (x18407) {
x18411 = x18410;
} else {
x18411 = false;
}
bool x18412;
if (x18411) {
x18412 = x18410;
} else {
x18412 = false;
}
if (x18412) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18331,x18333,x18333,1,x18401,1,1);
assert(false && "");
}
bool x18418 = x18331 <= x18401;
int32_t x18419;
if (x18418) {
x18419 = x18401;
} else {
x18419 = x18331;
}
int32_t x18428 = x18419 * x18427;
int32_t x18429 = 64 * x18428;
float* x18430 = (float*)myMalloc(x18429 * sizeof(float));;
int32_t x18431;
if (x18402) {
x18431 = 0;
} else {
x18431 = x18339;
}
int32_t x18434;
if (x18403) {
x18434 = 0;
} else {
x18434 = 1;
}
for(int x18435=0; x18435 < 64; x18435++) {
int32_t x18447 = x18340 * x18435;
int32_t x18441 = x18428 * x18435;
for(int x18437=0; x18437 < x18419; x18437++) {
int32_t x18448 = x18431 * x18437;
int32_t x18449 = x18447 + x18448;
int32_t x18454 = x18434 * x18437;
int32_t x18443 = x18427 * x18437;
for(int x18439=0; x18439 < x18421; x18439++) {
int32_t x18450 = x18432 * x18439;
int32_t x18451 = x18449 + x18450;
int32_t x18445 = x18421 * x18439;
for(int x18440=0; x18440 < x18421; x18440++) {
int32_t x18452 = x18433 * x18440;
int32_t x18453 = x18451 + x18452;
float x18455 = x18342[x18453];
float x18456 = x168[x18454];
int32_t x18442 = x18440 + x18441;
int32_t x18444 = x18442 + x18443;
int32_t x18446 = x18444 + x18445;
float x18457 = x18455 * x18456;
x18430[x18446] = x18457;

}

}

}

}
int32_t x18467 = 0;
int32_t x18468 = 1;
x18468 *= 1;
x18467 += 1;
x18468 *= 1;
x18468 *= 1;
int32_t x18473 = x18467;
bool x18474 = x18473 >= 2;
if (x18474) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18479 = x18473 == 0;
if (x18479) {
int32_t x18480 = x18468;
bool x18481 = x18480 == 256;
if (x18481) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18488 = x18468;
bool x18490 = x18419 == 1;
int32_t x18489 = 256 / x18488;
bool x18491 = x18489 == 1;
bool x18495;
if (x454) {
bool x18492 = x18490 || x18491;
bool x18493 = x18419 == x18489;
bool x18494 = x18492 || x18493;
x18495 = x18494;
} else {
x18495 = false;
}
bool x18499;
if (x18495) {
x18499 = x18498;
} else {
x18499 = false;
}
bool x18500;
if (x18499) {
x18500 = x18498;
} else {
x18500 = false;
}
if (x18500) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18419,x18421,x18421,1,x18489,1,1);
assert(false && "");
}
bool x18506 = x18419 <= x18489;
int32_t x18507;
if (x18506) {
x18507 = x18489;
} else {
x18507 = x18419;
}
int32_t x18516 = x18507 * x18515;
int32_t x18517 = 64 * x18516;
float* x18518 = (float*)myMalloc(x18517 * sizeof(float));;
int32_t x18519;
if (x18490) {
x18519 = 0;
} else {
x18519 = x18427;
}
int32_t x18522;
if (x18491) {
x18522 = 0;
} else {
x18522 = 1;
}
for(int x18523=0; x18523 < 64; x18523++) {
int32_t x18535 = x18428 * x18523;
int32_t x18529 = x18516 * x18523;
for(int x18525=0; x18525 < x18507; x18525++) {
int32_t x18536 = x18519 * x18525;
int32_t x18537 = x18535 + x18536;
int32_t x18542 = x18522 * x18525;
int32_t x18531 = x18515 * x18525;
for(int x18527=0; x18527 < x18509; x18527++) {
int32_t x18538 = x18520 * x18527;
int32_t x18539 = x18537 + x18538;
int32_t x18533 = x18509 * x18527;
for(int x18528=0; x18528 < x18509; x18528++) {
int32_t x18540 = x18521 * x18528;
int32_t x18541 = x18539 + x18540;
float x18543 = x18430[x18541];
float x18544 = x109[x18542];
int32_t x18530 = x18528 + x18529;
int32_t x18532 = x18530 + x18531;
int32_t x18534 = x18532 + x18533;
float x18545 = x18543 + x18544;
x18518[x18534] = x18545;

}

}

}

}
float* x18555 = (float*)myMalloc(x18517 * sizeof(float));;
for(int x18557=0; x18557 < x18517; x18557++) {
float x18558 = x18518[x18557];
bool x18559 = x18558 < 0.0f;
if (x18559) {
x18555[x18557] = 0.0f;
} else {
float x18562 = x18518[x18557];
x18555[x18557] = x18562;
}

}
float* x18576 = (float*)myMalloc(x18575 * sizeof(float));;
int32_t x18579 = 64 * x18507;
int32_t x18580 = x18579 * x18571;
float* x18581 = (float*)myMalloc(x18580 * sizeof(float));;
int32_t x18577 = x18507 * x18571;
for(int x18582=0; x18582 < 64; x18582++) {
int32_t x18583 = x18582 * x18516;
float* x18584 = x18555+x18583;
int32_t x18585 = x18582 * x18572;
float* x18586 = x18576+x18585;
int32_t x18587 = x18582 * x18577;
float* x18588 = x18581+x18587;
for(int x18589=0; x18589 < x18507; x18589++) {
int32_t x18590 = x18589 / 1;
int32_t x18594 = x18590 * x18570;
int32_t x18595 = x18594 * x18570;
int32_t x18591 = x18589 % 1;
int32_t x18592 = x18591 / 1;
int32_t x18596 = x18592 * x18570;
int32_t x18597 = x18596 * x18570;
int32_t x18598 = x18595 + x18597;
int32_t x18593 = x18591 % 1;
int32_t x18599 = x18593 * x18570;
int32_t x18600 = x18599 * x18570;
int32_t x18601 = x18598 + x18600;
float* x18602 = x18588+x18601;
int32_t x18603 = x18590 * x18509;
int32_t x18604 = x18603 * x18509;
float* x18605 = x18584+x18604;
for(int x18607=0; x18607 < x18570; x18607++) {
int32_t x18609 = x18607 * x18570;
float* x18610 = x18602+x18609;
int32_t x18608 = x18607 + x18592;
int32_t x18611 = x18608 * x18509;
int32_t x18612 = x18611 + x18593;
float* x18613 = x18605+x18612;
memcpy(x18610, x18613, 4 * x18570);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x18571,x18507,1,x221,x18507,x18588,x18571,1,x18586,x18571);

}
int32_t x18622 = 0;
int32_t x18623 = 1;
x18623 *= 1;
x18622 += 1;
x18623 *= 1;
x18623 *= 1;
int32_t x18628 = x18622;
bool x18629 = x18628 >= 2;
if (x18629) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18634 = x18628 == 0;
if (x18634) {
int32_t x18635 = x18623;
bool x18636 = x18635 == 1024;
if (x18636) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18643 = x18623;
int32_t x18644 = 1024 / x18643;
bool x18645 = x18644 == 1;
bool x18648;
if (x454) {
bool x18646 = 1024 == x18644;
bool x18647 = x18645 || x18646;
x18648 = x18647;
} else {
x18648 = false;
}
bool x18652;
if (x18648) {
x18652 = x18651;
} else {
x18652 = false;
}
bool x18653;
if (x18652) {
x18653 = x18651;
} else {
x18653 = false;
}
if (x18653) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x18570,x18570,1,x18644,1,1);
assert(false && "");
}
bool x18659 = 1024 <= x18644;
int32_t x18660;
if (x18659) {
x18660 = x18644;
} else {
x18660 = 1024;
}
int32_t x18669 = x18660 * x18668;
int32_t x18670 = 64 * x18669;
float* x18671 = (float*)myMalloc(x18670 * sizeof(float));;
int32_t x18674;
if (x18645) {
x18674 = 0;
} else {
x18674 = 1;
}
for(int x18675=0; x18675 < 64; x18675++) {
int32_t x18687 = x18572 * x18675;
int32_t x18681 = x18669 * x18675;
for(int x18677=0; x18677 < x18660; x18677++) {
int32_t x18688 = x18571 * x18677;
int32_t x18689 = x18687 + x18688;
int32_t x18694 = x18674 * x18677;
int32_t x18683 = x18668 * x18677;
for(int x18679=0; x18679 < x18662; x18679++) {
int32_t x18690 = x18672 * x18679;
int32_t x18691 = x18689 + x18690;
int32_t x18685 = x18662 * x18679;
for(int x18680=0; x18680 < x18662; x18680++) {
int32_t x18692 = x18673 * x18680;
int32_t x18693 = x18691 + x18692;
float x18695 = x18576[x18693];
float x18696 = x209[x18694];
int32_t x18682 = x18680 + x18681;
int32_t x18684 = x18682 + x18683;
int32_t x18686 = x18684 + x18685;
float x18697 = x18695 - x18696;
x18671[x18686] = x18697;

}

}

}

}
float* x18707 = (float*)myMalloc(1024 * sizeof(float));;
for(int x18708=0; x18708 < 1024; x18708++) {
float x18709 = x272[x18708];
float x18710 = x18709 + 1.0E-5f;
x18707[x18708] = x18710;

}
float* x18714 = (float*)myMalloc(1024 * sizeof(float));;
for(int x18715=0; x18715 < 1024; x18715++) {
float x18716 = x18707[x18715];
double x18717 = (double)x18716;
double x18718 = sqrt(x18717);
float x18719 = (float)x18718;
x18714[x18715] = x18719;

}
int32_t x18723 = 0;
int32_t x18724 = 1;
x18724 *= 1;
x18723 += 1;
x18724 *= 1;
x18724 *= 1;
int32_t x18729 = x18723;
bool x18730 = x18729 >= 2;
if (x18730) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18735 = x18729 == 0;
if (x18735) {
int32_t x18736 = x18724;
bool x18737 = x18736 == 1024;
if (x18737) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18744 = x18724;
bool x18746 = x18660 == 1;
int32_t x18745 = 1024 / x18744;
bool x18747 = x18745 == 1;
bool x18751;
if (x454) {
bool x18748 = x18746 || x18747;
bool x18749 = x18660 == x18745;
bool x18750 = x18748 || x18749;
x18751 = x18750;
} else {
x18751 = false;
}
bool x18755;
if (x18751) {
x18755 = x18754;
} else {
x18755 = false;
}
bool x18756;
if (x18755) {
x18756 = x18754;
} else {
x18756 = false;
}
if (x18756) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18660,x18662,x18662,1,x18745,1,1);
assert(false && "");
}
bool x18762 = x18660 <= x18745;
int32_t x18763;
if (x18762) {
x18763 = x18745;
} else {
x18763 = x18660;
}
int32_t x18772 = x18763 * x18771;
int32_t x18773 = 64 * x18772;
float* x18774 = (float*)myMalloc(x18773 * sizeof(float));;
int32_t x18775;
if (x18746) {
x18775 = 0;
} else {
x18775 = x18668;
}
int32_t x18778;
if (x18747) {
x18778 = 0;
} else {
x18778 = 1;
}
for(int x18779=0; x18779 < 64; x18779++) {
int32_t x18791 = x18669 * x18779;
int32_t x18785 = x18772 * x18779;
for(int x18781=0; x18781 < x18763; x18781++) {
int32_t x18792 = x18775 * x18781;
int32_t x18793 = x18791 + x18792;
int32_t x18798 = x18778 * x18781;
int32_t x18787 = x18771 * x18781;
for(int x18783=0; x18783 < x18765; x18783++) {
int32_t x18794 = x18776 * x18783;
int32_t x18795 = x18793 + x18794;
int32_t x18789 = x18765 * x18783;
for(int x18784=0; x18784 < x18765; x18784++) {
int32_t x18796 = x18777 * x18784;
int32_t x18797 = x18795 + x18796;
float x18799 = x18671[x18797];
float x18800 = x18714[x18798];
int32_t x18786 = x18784 + x18785;
int32_t x18788 = x18786 + x18787;
int32_t x18790 = x18788 + x18789;
float x18801 = x18799 / x18800;
x18774[x18790] = x18801;

}

}

}

}
int32_t x18811 = 0;
int32_t x18812 = 1;
x18812 *= 1;
x18811 += 1;
x18812 *= 1;
x18812 *= 1;
int32_t x18817 = x18811;
bool x18818 = x18817 >= 2;
if (x18818) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18823 = x18817 == 0;
if (x18823) {
int32_t x18824 = x18812;
bool x18825 = x18824 == 1024;
if (x18825) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18832 = x18812;
bool x18834 = x18763 == 1;
int32_t x18833 = 1024 / x18832;
bool x18835 = x18833 == 1;
bool x18839;
if (x454) {
bool x18836 = x18834 || x18835;
bool x18837 = x18763 == x18833;
bool x18838 = x18836 || x18837;
x18839 = x18838;
} else {
x18839 = false;
}
bool x18843;
if (x18839) {
x18843 = x18842;
} else {
x18843 = false;
}
bool x18844;
if (x18843) {
x18844 = x18842;
} else {
x18844 = false;
}
if (x18844) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18763,x18765,x18765,1,x18833,1,1);
assert(false && "");
}
bool x18850 = x18763 <= x18833;
int32_t x18851;
if (x18850) {
x18851 = x18833;
} else {
x18851 = x18763;
}
int32_t x18860 = x18851 * x18859;
int32_t x18861 = 64 * x18860;
float* x18862 = (float*)myMalloc(x18861 * sizeof(float));;
int32_t x18863;
if (x18834) {
x18863 = 0;
} else {
x18863 = x18771;
}
int32_t x18866;
if (x18835) {
x18866 = 0;
} else {
x18866 = 1;
}
for(int x18867=0; x18867 < 64; x18867++) {
int32_t x18879 = x18772 * x18867;
int32_t x18873 = x18860 * x18867;
for(int x18869=0; x18869 < x18851; x18869++) {
int32_t x18880 = x18863 * x18869;
int32_t x18881 = x18879 + x18880;
int32_t x18886 = x18866 * x18869;
int32_t x18875 = x18859 * x18869;
for(int x18871=0; x18871 < x18853; x18871++) {
int32_t x18882 = x18864 * x18871;
int32_t x18883 = x18881 + x18882;
int32_t x18877 = x18853 * x18871;
for(int x18872=0; x18872 < x18853; x18872++) {
int32_t x18884 = x18865 * x18872;
int32_t x18885 = x18883 + x18884;
float x18887 = x18774[x18885];
float x18888 = x59[x18886];
int32_t x18874 = x18872 + x18873;
int32_t x18876 = x18874 + x18875;
int32_t x18878 = x18876 + x18877;
float x18889 = x18887 * x18888;
x18862[x18878] = x18889;

}

}

}

}
int32_t x18899 = 0;
int32_t x18900 = 1;
x18900 *= 1;
x18899 += 1;
x18900 *= 1;
x18900 *= 1;
int32_t x18905 = x18899;
bool x18906 = x18905 >= 2;
if (x18906) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18911 = x18905 == 0;
if (x18911) {
int32_t x18912 = x18900;
bool x18913 = x18912 == 1024;
if (x18913) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18920 = x18900;
bool x18922 = x18851 == 1;
int32_t x18921 = 1024 / x18920;
bool x18923 = x18921 == 1;
bool x18927;
if (x454) {
bool x18924 = x18922 || x18923;
bool x18925 = x18851 == x18921;
bool x18926 = x18924 || x18925;
x18927 = x18926;
} else {
x18927 = false;
}
bool x18931;
if (x18927) {
x18931 = x18930;
} else {
x18931 = false;
}
bool x18932;
if (x18931) {
x18932 = x18930;
} else {
x18932 = false;
}
if (x18932) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18851,x18853,x18853,1,x18921,1,1);
assert(false && "");
}
bool x18938 = x18851 <= x18921;
int32_t x18939;
if (x18938) {
x18939 = x18921;
} else {
x18939 = x18851;
}
int32_t x18948 = x18939 * x18947;
int32_t x18949 = 64 * x18948;
float* x18950 = (float*)myMalloc(x18949 * sizeof(float));;
int32_t x18951;
if (x18922) {
x18951 = 0;
} else {
x18951 = x18859;
}
int32_t x18954;
if (x18923) {
x18954 = 0;
} else {
x18954 = 1;
}
for(int x18955=0; x18955 < 64; x18955++) {
int32_t x18967 = x18860 * x18955;
int32_t x18961 = x18948 * x18955;
for(int x18957=0; x18957 < x18939; x18957++) {
int32_t x18968 = x18951 * x18957;
int32_t x18969 = x18967 + x18968;
int32_t x18974 = x18954 * x18957;
int32_t x18963 = x18947 * x18957;
for(int x18959=0; x18959 < x18941; x18959++) {
int32_t x18970 = x18952 * x18959;
int32_t x18971 = x18969 + x18970;
int32_t x18965 = x18941 * x18959;
for(int x18960=0; x18960 < x18941; x18960++) {
int32_t x18972 = x18953 * x18960;
int32_t x18973 = x18971 + x18972;
float x18975 = x18862[x18973];
float x18976 = x120[x18974];
int32_t x18962 = x18960 + x18961;
int32_t x18964 = x18962 + x18963;
int32_t x18966 = x18964 + x18965;
float x18977 = x18975 + x18976;
x18950[x18966] = x18977;

}

}

}

}
bool x18987 = x18939 == 1;
bool x18988 = x18987 || x17575;
bool x18989 = x18939 == x17527;
bool x18990 = x18988 || x18989;
bool x18995;
if (x18990) {
x18995 = x18994;
} else {
x18995 = false;
}
bool x18996;
if (x18995) {
x18996 = x18994;
} else {
x18996 = false;
}
if (x18996) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18939,x18941,x18941,64,x17527,x17529,x17529);
assert(false && "");
}
bool x19002 = x18939 <= x17527;
int32_t x19003;
if (x19002) {
x19003 = x17527;
} else {
x19003 = x18939;
}
int32_t x19019;
if (x18987) {
x19019 = 0;
} else {
x19019 = x18947;
}
for(int x19022=0; x19022 < 64; x19022++) {
int32_t x19028 = x18948 * x19022;
int32_t x19035 = x17536 * x19022;
for(int x19024=0; x19024 < x19003; x19024++) {
int32_t x19029 = x19019 * x19024;
int32_t x19030 = x19028 + x19029;
int32_t x19036 = x17607 * x19024;
int32_t x19037 = x19035 + x19036;
for(int x19026=0; x19026 < x19005; x19026++) {
int32_t x19031 = x19020 * x19026;
int32_t x19032 = x19030 + x19031;
int32_t x19038 = x17608 * x19026;
int32_t x19039 = x19037 + x19038;
for(int x19027=0; x19027 < x19005; x19027++) {
int32_t x19033 = x19021 * x19027;
int32_t x19034 = x19032 + x19033;
float x19042 = x18950[x19034];
int32_t x19040 = x17609 * x19027;
int32_t x19041 = x19039 + x19040;
float x19043 = x17642[x19041];
float x19044 = x19042 + x19043;
x18950[x19034] = x19044;

}

}

}

}
float* x19054 = (float*)myMalloc(x18949 * sizeof(float));;
for(int x19056=0; x19056 < x18949; x19056++) {
float x19057 = x18950[x19056];
bool x19058 = x19057 < 0.0f;
if (x19058) {
x19054[x19056] = 0.0f;
} else {
float x19061 = x18950[x19056];
x19054[x19056] = x19061;
}

}
float* x19075 = (float*)myMalloc(x19074 * sizeof(float));;
int32_t x19078 = 64 * x18939;
int32_t x19079 = x19078 * x19070;
float* x19080 = (float*)myMalloc(x19079 * sizeof(float));;
int32_t x19076 = x18939 * x19070;
for(int x19081=0; x19081 < 64; x19081++) {
int32_t x19082 = x19081 * x18948;
float* x19083 = x19054+x19082;
int32_t x19084 = x19081 * x19071;
float* x19085 = x19075+x19084;
int32_t x19086 = x19081 * x19076;
float* x19087 = x19080+x19086;
for(int x19088=0; x19088 < x18939; x19088++) {
int32_t x19089 = x19088 / 1;
int32_t x19093 = x19089 * x19069;
int32_t x19094 = x19093 * x19069;
int32_t x19090 = x19088 % 1;
int32_t x19091 = x19090 / 1;
int32_t x19095 = x19091 * x19069;
int32_t x19096 = x19095 * x19069;
int32_t x19097 = x19094 + x19096;
int32_t x19092 = x19090 % 1;
int32_t x19098 = x19092 * x19069;
int32_t x19099 = x19098 * x19069;
int32_t x19100 = x19097 + x19099;
float* x19101 = x19087+x19100;
int32_t x19102 = x19089 * x18941;
int32_t x19103 = x19102 * x18941;
float* x19104 = x19083+x19103;
for(int x19106=0; x19106 < x19069; x19106++) {
int32_t x19108 = x19106 * x19069;
float* x19109 = x19101+x19108;
int32_t x19107 = x19106 + x19091;
int32_t x19110 = x19107 * x18941;
int32_t x19111 = x19110 + x19092;
float* x19112 = x19104+x19111;
memcpy(x19109, x19112, 4 * x19069);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x19070,x18939,1,x151,x18939,x19087,x19070,1,x19085,x19070);

}
int32_t x19121 = 0;
int32_t x19122 = 1;
x19122 *= 1;
x19121 += 1;
x19122 *= 1;
x19122 *= 1;
int32_t x19127 = x19121;
bool x19128 = x19127 >= 2;
if (x19128) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19133 = x19127 == 0;
if (x19133) {
int32_t x19134 = x19122;
bool x19135 = x19134 == 256;
if (x19135) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19142 = x19122;
int32_t x19143 = 256 / x19142;
bool x19144 = x19143 == 1;
bool x19147;
if (x454) {
bool x19145 = 256 == x19143;
bool x19146 = x19144 || x19145;
x19147 = x19146;
} else {
x19147 = false;
}
bool x19151;
if (x19147) {
x19151 = x19150;
} else {
x19151 = false;
}
bool x19152;
if (x19151) {
x19152 = x19150;
} else {
x19152 = false;
}
if (x19152) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x19069,x19069,1,x19143,1,1);
assert(false && "");
}
bool x19158 = 256 <= x19143;
int32_t x19159;
if (x19158) {
x19159 = x19143;
} else {
x19159 = 256;
}
int32_t x19168 = x19159 * x19167;
int32_t x19169 = 64 * x19168;
float* x19170 = (float*)myMalloc(x19169 * sizeof(float));;
int32_t x19173;
if (x19144) {
x19173 = 0;
} else {
x19173 = 1;
}
for(int x19174=0; x19174 < 64; x19174++) {
int32_t x19186 = x19071 * x19174;
int32_t x19180 = x19168 * x19174;
for(int x19176=0; x19176 < x19159; x19176++) {
int32_t x19187 = x19070 * x19176;
int32_t x19188 = x19186 + x19187;
int32_t x19193 = x19173 * x19176;
int32_t x19182 = x19167 * x19176;
for(int x19178=0; x19178 < x19161; x19178++) {
int32_t x19189 = x19171 * x19178;
int32_t x19190 = x19188 + x19189;
int32_t x19184 = x19161 * x19178;
for(int x19179=0; x19179 < x19161; x19179++) {
int32_t x19191 = x19172 * x19179;
int32_t x19192 = x19190 + x19191;
float x19194 = x19075[x19192];
float x19195 = x80[x19193];
int32_t x19181 = x19179 + x19180;
int32_t x19183 = x19181 + x19182;
int32_t x19185 = x19183 + x19184;
float x19196 = x19194 - x19195;
x19170[x19185] = x19196;

}

}

}

}
float* x19206 = (float*)myMalloc(256 * sizeof(float));;
for(int x19207=0; x19207 < 256; x19207++) {
float x19208 = x176[x19207];
float x19209 = x19208 + 1.0E-5f;
x19206[x19207] = x19209;

}
float* x19213 = (float*)myMalloc(256 * sizeof(float));;
for(int x19214=0; x19214 < 256; x19214++) {
float x19215 = x19206[x19214];
double x19216 = (double)x19215;
double x19217 = sqrt(x19216);
float x19218 = (float)x19217;
x19213[x19214] = x19218;

}
int32_t x19222 = 0;
int32_t x19223 = 1;
x19223 *= 1;
x19222 += 1;
x19223 *= 1;
x19223 *= 1;
int32_t x19228 = x19222;
bool x19229 = x19228 >= 2;
if (x19229) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19234 = x19228 == 0;
if (x19234) {
int32_t x19235 = x19223;
bool x19236 = x19235 == 256;
if (x19236) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19243 = x19223;
bool x19245 = x19159 == 1;
int32_t x19244 = 256 / x19243;
bool x19246 = x19244 == 1;
bool x19250;
if (x454) {
bool x19247 = x19245 || x19246;
bool x19248 = x19159 == x19244;
bool x19249 = x19247 || x19248;
x19250 = x19249;
} else {
x19250 = false;
}
bool x19254;
if (x19250) {
x19254 = x19253;
} else {
x19254 = false;
}
bool x19255;
if (x19254) {
x19255 = x19253;
} else {
x19255 = false;
}
if (x19255) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19159,x19161,x19161,1,x19244,1,1);
assert(false && "");
}
bool x19261 = x19159 <= x19244;
int32_t x19262;
if (x19261) {
x19262 = x19244;
} else {
x19262 = x19159;
}
int32_t x19271 = x19262 * x19270;
int32_t x19272 = 64 * x19271;
float* x19273 = (float*)myMalloc(x19272 * sizeof(float));;
int32_t x19274;
if (x19245) {
x19274 = 0;
} else {
x19274 = x19167;
}
int32_t x19277;
if (x19246) {
x19277 = 0;
} else {
x19277 = 1;
}
for(int x19278=0; x19278 < 64; x19278++) {
int32_t x19290 = x19168 * x19278;
int32_t x19284 = x19271 * x19278;
for(int x19280=0; x19280 < x19262; x19280++) {
int32_t x19291 = x19274 * x19280;
int32_t x19292 = x19290 + x19291;
int32_t x19297 = x19277 * x19280;
int32_t x19286 = x19270 * x19280;
for(int x19282=0; x19282 < x19264; x19282++) {
int32_t x19293 = x19275 * x19282;
int32_t x19294 = x19292 + x19293;
int32_t x19288 = x19264 * x19282;
for(int x19283=0; x19283 < x19264; x19283++) {
int32_t x19295 = x19276 * x19283;
int32_t x19296 = x19294 + x19295;
float x19298 = x19170[x19296];
float x19299 = x19213[x19297];
int32_t x19285 = x19283 + x19284;
int32_t x19287 = x19285 + x19286;
int32_t x19289 = x19287 + x19288;
float x19300 = x19298 / x19299;
x19273[x19289] = x19300;

}

}

}

}
int32_t x19310 = 0;
int32_t x19311 = 1;
x19311 *= 1;
x19310 += 1;
x19311 *= 1;
x19311 *= 1;
int32_t x19316 = x19310;
bool x19317 = x19316 >= 2;
if (x19317) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19322 = x19316 == 0;
if (x19322) {
int32_t x19323 = x19311;
bool x19324 = x19323 == 256;
if (x19324) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19331 = x19311;
bool x19333 = x19262 == 1;
int32_t x19332 = 256 / x19331;
bool x19334 = x19332 == 1;
bool x19338;
if (x454) {
bool x19335 = x19333 || x19334;
bool x19336 = x19262 == x19332;
bool x19337 = x19335 || x19336;
x19338 = x19337;
} else {
x19338 = false;
}
bool x19342;
if (x19338) {
x19342 = x19341;
} else {
x19342 = false;
}
bool x19343;
if (x19342) {
x19343 = x19341;
} else {
x19343 = false;
}
if (x19343) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19262,x19264,x19264,1,x19332,1,1);
assert(false && "");
}
bool x19349 = x19262 <= x19332;
int32_t x19350;
if (x19349) {
x19350 = x19332;
} else {
x19350 = x19262;
}
int32_t x19359 = x19350 * x19358;
int32_t x19360 = 64 * x19359;
float* x19361 = (float*)myMalloc(x19360 * sizeof(float));;
int32_t x19362;
if (x19333) {
x19362 = 0;
} else {
x19362 = x19270;
}
int32_t x19365;
if (x19334) {
x19365 = 0;
} else {
x19365 = 1;
}
for(int x19366=0; x19366 < 64; x19366++) {
int32_t x19378 = x19271 * x19366;
int32_t x19372 = x19359 * x19366;
for(int x19368=0; x19368 < x19350; x19368++) {
int32_t x19379 = x19362 * x19368;
int32_t x19380 = x19378 + x19379;
int32_t x19385 = x19365 * x19368;
int32_t x19374 = x19358 * x19368;
for(int x19370=0; x19370 < x19352; x19370++) {
int32_t x19381 = x19363 * x19370;
int32_t x19382 = x19380 + x19381;
int32_t x19376 = x19352 * x19370;
for(int x19371=0; x19371 < x19352; x19371++) {
int32_t x19383 = x19364 * x19371;
int32_t x19384 = x19382 + x19383;
float x19386 = x19273[x19384];
float x19387 = x85[x19385];
int32_t x19373 = x19371 + x19372;
int32_t x19375 = x19373 + x19374;
int32_t x19377 = x19375 + x19376;
float x19388 = x19386 * x19387;
x19361[x19377] = x19388;

}

}

}

}
int32_t x19398 = 0;
int32_t x19399 = 1;
x19399 *= 1;
x19398 += 1;
x19399 *= 1;
x19399 *= 1;
int32_t x19404 = x19398;
bool x19405 = x19404 >= 2;
if (x19405) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19410 = x19404 == 0;
if (x19410) {
int32_t x19411 = x19399;
bool x19412 = x19411 == 256;
if (x19412) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19419 = x19399;
bool x19421 = x19350 == 1;
int32_t x19420 = 256 / x19419;
bool x19422 = x19420 == 1;
bool x19426;
if (x454) {
bool x19423 = x19421 || x19422;
bool x19424 = x19350 == x19420;
bool x19425 = x19423 || x19424;
x19426 = x19425;
} else {
x19426 = false;
}
bool x19430;
if (x19426) {
x19430 = x19429;
} else {
x19430 = false;
}
bool x19431;
if (x19430) {
x19431 = x19429;
} else {
x19431 = false;
}
if (x19431) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19350,x19352,x19352,1,x19420,1,1);
assert(false && "");
}
bool x19437 = x19350 <= x19420;
int32_t x19438;
if (x19437) {
x19438 = x19420;
} else {
x19438 = x19350;
}
int32_t x19447 = x19438 * x19446;
int32_t x19448 = 64 * x19447;
float* x19449 = (float*)myMalloc(x19448 * sizeof(float));;
int32_t x19450;
if (x19421) {
x19450 = 0;
} else {
x19450 = x19358;
}
int32_t x19453;
if (x19422) {
x19453 = 0;
} else {
x19453 = 1;
}
for(int x19454=0; x19454 < 64; x19454++) {
int32_t x19466 = x19359 * x19454;
int32_t x19460 = x19447 * x19454;
for(int x19456=0; x19456 < x19438; x19456++) {
int32_t x19467 = x19450 * x19456;
int32_t x19468 = x19466 + x19467;
int32_t x19473 = x19453 * x19456;
int32_t x19462 = x19446 * x19456;
for(int x19458=0; x19458 < x19440; x19458++) {
int32_t x19469 = x19451 * x19458;
int32_t x19470 = x19468 + x19469;
int32_t x19464 = x19440 * x19458;
for(int x19459=0; x19459 < x19440; x19459++) {
int32_t x19471 = x19452 * x19459;
int32_t x19472 = x19470 + x19471;
float x19474 = x19361[x19472];
float x19475 = x253[x19473];
int32_t x19461 = x19459 + x19460;
int32_t x19463 = x19461 + x19462;
int32_t x19465 = x19463 + x19464;
float x19476 = x19474 + x19475;
x19449[x19465] = x19476;

}

}

}

}
float* x19486 = (float*)myMalloc(x19448 * sizeof(float));;
for(int x19488=0; x19488 < x19448; x19488++) {
float x19489 = x19449[x19488];
bool x19490 = x19489 < 0.0f;
if (x19490) {
x19486[x19488] = 0.0f;
} else {
float x19493 = x19449[x19488];
x19486[x19488] = x19493;
}

}
float* x19508 = (float*)myMalloc(x19507 * sizeof(float));;
int32_t x19509 = 9 * x19438;
int32_t x19512 = 64 * x19509;
int32_t x19513 = x19512 * x19503;
float* x19514 = (float*)myMalloc(x19513 * sizeof(float));;
int32_t x19510 = x19509 * x19503;
int32_t x19522 = x19438 * 3;
int32_t x19523 = x19522 * 3;
for(int x19515=0; x19515 < 64; x19515++) {
int32_t x19516 = x19515 * x19447;
float* x19517 = x19486+x19516;
int32_t x19518 = x19515 * x19504;
float* x19519 = x19508+x19518;
int32_t x19520 = x19515 * x19510;
float* x19521 = x19514+x19520;
for(int x19525=0; x19525 < x19523; x19525++) {
int32_t x19526 = x19525 / 9;
int32_t x19530 = x19526 * 3;
int32_t x19531 = x19530 * 3;
int32_t x19532 = x19531 * x19502;
int32_t x19533 = x19532 * x19502;
int32_t x19527 = x19525 % 9;
int32_t x19528 = x19527 / 3;
int32_t x19534 = x19528 * 3;
int32_t x19535 = x19534 * x19502;
int32_t x19536 = x19535 * x19502;
int32_t x19537 = x19533 + x19536;
int32_t x19529 = x19527 % 3;
int32_t x19538 = x19529 * x19502;
int32_t x19539 = x19538 * x19502;
int32_t x19540 = x19537 + x19539;
float* x19541 = x19521+x19540;
int32_t x19542 = x19526 * x19440;
int32_t x19543 = x19542 * x19440;
float* x19544 = x19517+x19543;
int32_t x19557 = 1 - x19529;
bool x19558 = x19557 > 0;
int32_t x19559;
if (x19558) {
x19559 = x19557;
} else {
x19559 = 0;
}
int32_t x19560 = 3 - x19529;
int32_t x19561 = x19560 - 1;
int32_t x19562 = 1 - x19561;
bool x19563 = x19562 > 0;
int32_t x19564;
if (x19563) {
x19564 = x19562;
} else {
x19564 = 0;
}
int32_t x19565 = x19502 - x19564;
int32_t x19566 = x19565 - x19559;
bool x19567 = x19566 <= 0;
bool x19571 = x19559 > 0;
int32_t x19556 = -1 + x19529;
bool x19584 = x19564 > 0;
for(int x19546=0; x19546 < x19502; x19546++) {
int32_t x19547 = x19546 - 1;
int32_t x19548 = x19547 + x19528;
bool x19549 = x19548 < 0;
bool x19550 = x19548 >= x19440;
bool x19551 = x19549 || x19550;
if (x19551) {
int32_t x19552 = x19546 * x19502;
float* x19553 = x19541+x19552;
memset(x19553, 0, 4 * x19502);;
} else {
if (x19567) {
int32_t x19552 = x19546 * x19502;
float* x19568 = x19541+x19552;
memset(x19568, 0, 4 * x19502);;
} else {
int32_t x19552 = x19546 * x19502;
if (x19571) {
float* x19572 = x19541+x19552;
memset(x19572, 0, 4 * x19559);;
} else {
}
// may have segfault here
int32_t x19577 = x19552 + x19559;
float* x19578 = x19541+x19577;
int32_t x19579 = x19548 * x19440;
int32_t x19580 = x19579 + x19556;
int32_t x19581 = x19580 + x19559;
float* x19582 = x19544+x19581;
memcpy(x19578, x19582, 4 * x19566);;
if (x19584) {
int32_t x19585 = x19552 + x19502;
int32_t x19586 = x19585 - x19564;
float* x19587 = x19541+x19586;
memset(x19587, 0, 4 * x19564);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x19503,x19509,1,x226,x19509,x19521,x19503,1,x19519,x19503);

}
int32_t x19602 = 0;
int32_t x19603 = 1;
x19603 *= 1;
x19602 += 1;
x19603 *= 1;
x19603 *= 1;
int32_t x19608 = x19602;
bool x19609 = x19608 >= 2;
if (x19609) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19614 = x19608 == 0;
if (x19614) {
int32_t x19615 = x19603;
bool x19616 = x19615 == 256;
if (x19616) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19623 = x19603;
int32_t x19624 = 256 / x19623;
bool x19625 = x19624 == 1;
bool x19628;
if (x454) {
bool x19626 = 256 == x19624;
bool x19627 = x19625 || x19626;
x19628 = x19627;
} else {
x19628 = false;
}
bool x19632;
if (x19628) {
x19632 = x19631;
} else {
x19632 = false;
}
bool x19633;
if (x19632) {
x19633 = x19631;
} else {
x19633 = false;
}
if (x19633) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x19502,x19502,1,x19624,1,1);
assert(false && "");
}
bool x19639 = 256 <= x19624;
int32_t x19640;
if (x19639) {
x19640 = x19624;
} else {
x19640 = 256;
}
int32_t x19649 = x19640 * x19648;
int32_t x19650 = 64 * x19649;
float* x19651 = (float*)myMalloc(x19650 * sizeof(float));;
int32_t x19654;
if (x19625) {
x19654 = 0;
} else {
x19654 = 1;
}
for(int x19655=0; x19655 < 64; x19655++) {
int32_t x19667 = x19504 * x19655;
int32_t x19661 = x19649 * x19655;
for(int x19657=0; x19657 < x19640; x19657++) {
int32_t x19668 = x19503 * x19657;
int32_t x19669 = x19667 + x19668;
int32_t x19674 = x19654 * x19657;
int32_t x19663 = x19648 * x19657;
for(int x19659=0; x19659 < x19642; x19659++) {
int32_t x19670 = x19652 * x19659;
int32_t x19671 = x19669 + x19670;
int32_t x19665 = x19642 * x19659;
for(int x19660=0; x19660 < x19642; x19660++) {
int32_t x19672 = x19653 * x19660;
int32_t x19673 = x19671 + x19672;
float x19675 = x19508[x19673];
float x19676 = x70[x19674];
int32_t x19662 = x19660 + x19661;
int32_t x19664 = x19662 + x19663;
int32_t x19666 = x19664 + x19665;
float x19677 = x19675 - x19676;
x19651[x19666] = x19677;

}

}

}

}
float* x19687 = (float*)myMalloc(256 * sizeof(float));;
for(int x19688=0; x19688 < 256; x19688++) {
float x19689 = x240[x19688];
float x19690 = x19689 + 1.0E-5f;
x19687[x19688] = x19690;

}
float* x19694 = (float*)myMalloc(256 * sizeof(float));;
for(int x19695=0; x19695 < 256; x19695++) {
float x19696 = x19687[x19695];
double x19697 = (double)x19696;
double x19698 = sqrt(x19697);
float x19699 = (float)x19698;
x19694[x19695] = x19699;

}
int32_t x19703 = 0;
int32_t x19704 = 1;
x19704 *= 1;
x19703 += 1;
x19704 *= 1;
x19704 *= 1;
int32_t x19709 = x19703;
bool x19710 = x19709 >= 2;
if (x19710) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19715 = x19709 == 0;
if (x19715) {
int32_t x19716 = x19704;
bool x19717 = x19716 == 256;
if (x19717) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19724 = x19704;
bool x19726 = x19640 == 1;
int32_t x19725 = 256 / x19724;
bool x19727 = x19725 == 1;
bool x19731;
if (x454) {
bool x19728 = x19726 || x19727;
bool x19729 = x19640 == x19725;
bool x19730 = x19728 || x19729;
x19731 = x19730;
} else {
x19731 = false;
}
bool x19735;
if (x19731) {
x19735 = x19734;
} else {
x19735 = false;
}
bool x19736;
if (x19735) {
x19736 = x19734;
} else {
x19736 = false;
}
if (x19736) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19640,x19642,x19642,1,x19725,1,1);
assert(false && "");
}
bool x19742 = x19640 <= x19725;
int32_t x19743;
if (x19742) {
x19743 = x19725;
} else {
x19743 = x19640;
}
int32_t x19752 = x19743 * x19751;
int32_t x19753 = 64 * x19752;
float* x19754 = (float*)myMalloc(x19753 * sizeof(float));;
int32_t x19755;
if (x19726) {
x19755 = 0;
} else {
x19755 = x19648;
}
int32_t x19758;
if (x19727) {
x19758 = 0;
} else {
x19758 = 1;
}
for(int x19759=0; x19759 < 64; x19759++) {
int32_t x19771 = x19649 * x19759;
int32_t x19765 = x19752 * x19759;
for(int x19761=0; x19761 < x19743; x19761++) {
int32_t x19772 = x19755 * x19761;
int32_t x19773 = x19771 + x19772;
int32_t x19778 = x19758 * x19761;
int32_t x19767 = x19751 * x19761;
for(int x19763=0; x19763 < x19745; x19763++) {
int32_t x19774 = x19756 * x19763;
int32_t x19775 = x19773 + x19774;
int32_t x19769 = x19745 * x19763;
for(int x19764=0; x19764 < x19745; x19764++) {
int32_t x19776 = x19757 * x19764;
int32_t x19777 = x19775 + x19776;
float x19779 = x19651[x19777];
float x19780 = x19694[x19778];
int32_t x19766 = x19764 + x19765;
int32_t x19768 = x19766 + x19767;
int32_t x19770 = x19768 + x19769;
float x19781 = x19779 / x19780;
x19754[x19770] = x19781;

}

}

}

}
int32_t x19791 = 0;
int32_t x19792 = 1;
x19792 *= 1;
x19791 += 1;
x19792 *= 1;
x19792 *= 1;
int32_t x19797 = x19791;
bool x19798 = x19797 >= 2;
if (x19798) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19803 = x19797 == 0;
if (x19803) {
int32_t x19804 = x19792;
bool x19805 = x19804 == 256;
if (x19805) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19812 = x19792;
bool x19814 = x19743 == 1;
int32_t x19813 = 256 / x19812;
bool x19815 = x19813 == 1;
bool x19819;
if (x454) {
bool x19816 = x19814 || x19815;
bool x19817 = x19743 == x19813;
bool x19818 = x19816 || x19817;
x19819 = x19818;
} else {
x19819 = false;
}
bool x19823;
if (x19819) {
x19823 = x19822;
} else {
x19823 = false;
}
bool x19824;
if (x19823) {
x19824 = x19822;
} else {
x19824 = false;
}
if (x19824) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19743,x19745,x19745,1,x19813,1,1);
assert(false && "");
}
bool x19830 = x19743 <= x19813;
int32_t x19831;
if (x19830) {
x19831 = x19813;
} else {
x19831 = x19743;
}
int32_t x19840 = x19831 * x19839;
int32_t x19841 = 64 * x19840;
float* x19842 = (float*)myMalloc(x19841 * sizeof(float));;
int32_t x19843;
if (x19814) {
x19843 = 0;
} else {
x19843 = x19751;
}
int32_t x19846;
if (x19815) {
x19846 = 0;
} else {
x19846 = 1;
}
for(int x19847=0; x19847 < 64; x19847++) {
int32_t x19859 = x19752 * x19847;
int32_t x19853 = x19840 * x19847;
for(int x19849=0; x19849 < x19831; x19849++) {
int32_t x19860 = x19843 * x19849;
int32_t x19861 = x19859 + x19860;
int32_t x19866 = x19846 * x19849;
int32_t x19855 = x19839 * x19849;
for(int x19851=0; x19851 < x19833; x19851++) {
int32_t x19862 = x19844 * x19851;
int32_t x19863 = x19861 + x19862;
int32_t x19857 = x19833 * x19851;
for(int x19852=0; x19852 < x19833; x19852++) {
int32_t x19864 = x19845 * x19852;
int32_t x19865 = x19863 + x19864;
float x19867 = x19754[x19865];
float x19868 = x141[x19866];
int32_t x19854 = x19852 + x19853;
int32_t x19856 = x19854 + x19855;
int32_t x19858 = x19856 + x19857;
float x19869 = x19867 * x19868;
x19842[x19858] = x19869;

}

}

}

}
int32_t x19879 = 0;
int32_t x19880 = 1;
x19880 *= 1;
x19879 += 1;
x19880 *= 1;
x19880 *= 1;
int32_t x19885 = x19879;
bool x19886 = x19885 >= 2;
if (x19886) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19891 = x19885 == 0;
if (x19891) {
int32_t x19892 = x19880;
bool x19893 = x19892 == 256;
if (x19893) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19900 = x19880;
bool x19902 = x19831 == 1;
int32_t x19901 = 256 / x19900;
bool x19903 = x19901 == 1;
bool x19907;
if (x454) {
bool x19904 = x19902 || x19903;
bool x19905 = x19831 == x19901;
bool x19906 = x19904 || x19905;
x19907 = x19906;
} else {
x19907 = false;
}
bool x19911;
if (x19907) {
x19911 = x19910;
} else {
x19911 = false;
}
bool x19912;
if (x19911) {
x19912 = x19910;
} else {
x19912 = false;
}
if (x19912) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19831,x19833,x19833,1,x19901,1,1);
assert(false && "");
}
bool x19918 = x19831 <= x19901;
int32_t x19919;
if (x19918) {
x19919 = x19901;
} else {
x19919 = x19831;
}
int32_t x19928 = x19919 * x19927;
int32_t x19929 = 64 * x19928;
float* x19930 = (float*)myMalloc(x19929 * sizeof(float));;
int32_t x19931;
if (x19902) {
x19931 = 0;
} else {
x19931 = x19839;
}
int32_t x19934;
if (x19903) {
x19934 = 0;
} else {
x19934 = 1;
}
for(int x19935=0; x19935 < 64; x19935++) {
int32_t x19947 = x19840 * x19935;
int32_t x19941 = x19928 * x19935;
for(int x19937=0; x19937 < x19919; x19937++) {
int32_t x19948 = x19931 * x19937;
int32_t x19949 = x19947 + x19948;
int32_t x19954 = x19934 * x19937;
int32_t x19943 = x19927 * x19937;
for(int x19939=0; x19939 < x19921; x19939++) {
int32_t x19950 = x19932 * x19939;
int32_t x19951 = x19949 + x19950;
int32_t x19945 = x19921 * x19939;
for(int x19940=0; x19940 < x19921; x19940++) {
int32_t x19952 = x19933 * x19940;
int32_t x19953 = x19951 + x19952;
float x19955 = x19842[x19953];
float x19956 = x189[x19954];
int32_t x19942 = x19940 + x19941;
int32_t x19944 = x19942 + x19943;
int32_t x19946 = x19944 + x19945;
float x19957 = x19955 + x19956;
x19930[x19946] = x19957;

}

}

}

}
float* x19967 = (float*)myMalloc(x19929 * sizeof(float));;
for(int x19969=0; x19969 < x19929; x19969++) {
float x19970 = x19930[x19969];
bool x19971 = x19970 < 0.0f;
if (x19971) {
x19967[x19969] = 0.0f;
} else {
float x19974 = x19930[x19969];
x19967[x19969] = x19974;
}

}
float* x19988 = (float*)myMalloc(x19987 * sizeof(float));;
int32_t x19991 = 64 * x19919;
int32_t x19992 = x19991 * x19983;
float* x19993 = (float*)myMalloc(x19992 * sizeof(float));;
int32_t x19989 = x19919 * x19983;
for(int x19994=0; x19994 < 64; x19994++) {
int32_t x19995 = x19994 * x19928;
float* x19996 = x19967+x19995;
int32_t x19997 = x19994 * x19984;
float* x19998 = x19988+x19997;
int32_t x19999 = x19994 * x19989;
float* x20000 = x19993+x19999;
for(int x20001=0; x20001 < x19919; x20001++) {
int32_t x20002 = x20001 / 1;
int32_t x20006 = x20002 * x19982;
int32_t x20007 = x20006 * x19982;
int32_t x20003 = x20001 % 1;
int32_t x20004 = x20003 / 1;
int32_t x20008 = x20004 * x19982;
int32_t x20009 = x20008 * x19982;
int32_t x20010 = x20007 + x20009;
int32_t x20005 = x20003 % 1;
int32_t x20011 = x20005 * x19982;
int32_t x20012 = x20011 * x19982;
int32_t x20013 = x20010 + x20012;
float* x20014 = x20000+x20013;
int32_t x20015 = x20002 * x19921;
int32_t x20016 = x20015 * x19921;
float* x20017 = x19996+x20016;
for(int x20019=0; x20019 < x19982; x20019++) {
int32_t x20021 = x20019 * x19982;
float* x20022 = x20014+x20021;
int32_t x20020 = x20019 + x20004;
int32_t x20023 = x20020 * x19921;
int32_t x20024 = x20023 + x20005;
float* x20025 = x20017+x20024;
memcpy(x20022, x20025, 4 * x19982);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x19983,x19919,1,x97,x19919,x20000,x19983,1,x19998,x19983);

}
int32_t x20034 = 0;
int32_t x20035 = 1;
x20035 *= 1;
x20034 += 1;
x20035 *= 1;
x20035 *= 1;
int32_t x20040 = x20034;
bool x20041 = x20040 >= 2;
if (x20041) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20046 = x20040 == 0;
if (x20046) {
int32_t x20047 = x20035;
bool x20048 = x20047 == 1024;
if (x20048) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20055 = x20035;
int32_t x20056 = 1024 / x20055;
bool x20057 = x20056 == 1;
bool x20060;
if (x454) {
bool x20058 = 1024 == x20056;
bool x20059 = x20057 || x20058;
x20060 = x20059;
} else {
x20060 = false;
}
bool x20064;
if (x20060) {
x20064 = x20063;
} else {
x20064 = false;
}
bool x20065;
if (x20064) {
x20065 = x20063;
} else {
x20065 = false;
}
if (x20065) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x19982,x19982,1,x20056,1,1);
assert(false && "");
}
bool x20071 = 1024 <= x20056;
int32_t x20072;
if (x20071) {
x20072 = x20056;
} else {
x20072 = 1024;
}
int32_t x20081 = x20072 * x20080;
int32_t x20082 = 64 * x20081;
float* x20083 = (float*)myMalloc(x20082 * sizeof(float));;
int32_t x20086;
if (x20057) {
x20086 = 0;
} else {
x20086 = 1;
}
for(int x20087=0; x20087 < 64; x20087++) {
int32_t x20099 = x19984 * x20087;
int32_t x20093 = x20081 * x20087;
for(int x20089=0; x20089 < x20072; x20089++) {
int32_t x20100 = x19983 * x20089;
int32_t x20101 = x20099 + x20100;
int32_t x20106 = x20086 * x20089;
int32_t x20095 = x20080 * x20089;
for(int x20091=0; x20091 < x20074; x20091++) {
int32_t x20102 = x20084 * x20091;
int32_t x20103 = x20101 + x20102;
int32_t x20097 = x20074 * x20091;
for(int x20092=0; x20092 < x20074; x20092++) {
int32_t x20104 = x20085 * x20092;
int32_t x20105 = x20103 + x20104;
float x20107 = x19988[x20105];
float x20108 = x122[x20106];
int32_t x20094 = x20092 + x20093;
int32_t x20096 = x20094 + x20095;
int32_t x20098 = x20096 + x20097;
float x20109 = x20107 - x20108;
x20083[x20098] = x20109;

}

}

}

}
float* x20119 = (float*)myMalloc(1024 * sizeof(float));;
for(int x20120=0; x20120 < 1024; x20120++) {
float x20121 = x183[x20120];
float x20122 = x20121 + 1.0E-5f;
x20119[x20120] = x20122;

}
float* x20126 = (float*)myMalloc(1024 * sizeof(float));;
for(int x20127=0; x20127 < 1024; x20127++) {
float x20128 = x20119[x20127];
double x20129 = (double)x20128;
double x20130 = sqrt(x20129);
float x20131 = (float)x20130;
x20126[x20127] = x20131;

}
int32_t x20135 = 0;
int32_t x20136 = 1;
x20136 *= 1;
x20135 += 1;
x20136 *= 1;
x20136 *= 1;
int32_t x20141 = x20135;
bool x20142 = x20141 >= 2;
if (x20142) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20147 = x20141 == 0;
if (x20147) {
int32_t x20148 = x20136;
bool x20149 = x20148 == 1024;
if (x20149) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20156 = x20136;
bool x20158 = x20072 == 1;
int32_t x20157 = 1024 / x20156;
bool x20159 = x20157 == 1;
bool x20163;
if (x454) {
bool x20160 = x20158 || x20159;
bool x20161 = x20072 == x20157;
bool x20162 = x20160 || x20161;
x20163 = x20162;
} else {
x20163 = false;
}
bool x20167;
if (x20163) {
x20167 = x20166;
} else {
x20167 = false;
}
bool x20168;
if (x20167) {
x20168 = x20166;
} else {
x20168 = false;
}
if (x20168) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20072,x20074,x20074,1,x20157,1,1);
assert(false && "");
}
bool x20174 = x20072 <= x20157;
int32_t x20175;
if (x20174) {
x20175 = x20157;
} else {
x20175 = x20072;
}
int32_t x20184 = x20175 * x20183;
int32_t x20185 = 64 * x20184;
float* x20186 = (float*)myMalloc(x20185 * sizeof(float));;
int32_t x20187;
if (x20158) {
x20187 = 0;
} else {
x20187 = x20080;
}
int32_t x20190;
if (x20159) {
x20190 = 0;
} else {
x20190 = 1;
}
for(int x20191=0; x20191 < 64; x20191++) {
int32_t x20203 = x20081 * x20191;
int32_t x20197 = x20184 * x20191;
for(int x20193=0; x20193 < x20175; x20193++) {
int32_t x20204 = x20187 * x20193;
int32_t x20205 = x20203 + x20204;
int32_t x20210 = x20190 * x20193;
int32_t x20199 = x20183 * x20193;
for(int x20195=0; x20195 < x20177; x20195++) {
int32_t x20206 = x20188 * x20195;
int32_t x20207 = x20205 + x20206;
int32_t x20201 = x20177 * x20195;
for(int x20196=0; x20196 < x20177; x20196++) {
int32_t x20208 = x20189 * x20196;
int32_t x20209 = x20207 + x20208;
float x20211 = x20083[x20209];
float x20212 = x20126[x20210];
int32_t x20198 = x20196 + x20197;
int32_t x20200 = x20198 + x20199;
int32_t x20202 = x20200 + x20201;
float x20213 = x20211 / x20212;
x20186[x20202] = x20213;

}

}

}

}
int32_t x20223 = 0;
int32_t x20224 = 1;
x20224 *= 1;
x20223 += 1;
x20224 *= 1;
x20224 *= 1;
int32_t x20229 = x20223;
bool x20230 = x20229 >= 2;
if (x20230) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20235 = x20229 == 0;
if (x20235) {
int32_t x20236 = x20224;
bool x20237 = x20236 == 1024;
if (x20237) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20244 = x20224;
bool x20246 = x20175 == 1;
int32_t x20245 = 1024 / x20244;
bool x20247 = x20245 == 1;
bool x20251;
if (x454) {
bool x20248 = x20246 || x20247;
bool x20249 = x20175 == x20245;
bool x20250 = x20248 || x20249;
x20251 = x20250;
} else {
x20251 = false;
}
bool x20255;
if (x20251) {
x20255 = x20254;
} else {
x20255 = false;
}
bool x20256;
if (x20255) {
x20256 = x20254;
} else {
x20256 = false;
}
if (x20256) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20175,x20177,x20177,1,x20245,1,1);
assert(false && "");
}
bool x20262 = x20175 <= x20245;
int32_t x20263;
if (x20262) {
x20263 = x20245;
} else {
x20263 = x20175;
}
int32_t x20272 = x20263 * x20271;
int32_t x20273 = 64 * x20272;
float* x20274 = (float*)myMalloc(x20273 * sizeof(float));;
int32_t x20275;
if (x20246) {
x20275 = 0;
} else {
x20275 = x20183;
}
int32_t x20278;
if (x20247) {
x20278 = 0;
} else {
x20278 = 1;
}
for(int x20279=0; x20279 < 64; x20279++) {
int32_t x20291 = x20184 * x20279;
int32_t x20285 = x20272 * x20279;
for(int x20281=0; x20281 < x20263; x20281++) {
int32_t x20292 = x20275 * x20281;
int32_t x20293 = x20291 + x20292;
int32_t x20298 = x20278 * x20281;
int32_t x20287 = x20271 * x20281;
for(int x20283=0; x20283 < x20265; x20283++) {
int32_t x20294 = x20276 * x20283;
int32_t x20295 = x20293 + x20294;
int32_t x20289 = x20265 * x20283;
for(int x20284=0; x20284 < x20265; x20284++) {
int32_t x20296 = x20277 * x20284;
int32_t x20297 = x20295 + x20296;
float x20299 = x20186[x20297];
float x20300 = x248[x20298];
int32_t x20286 = x20284 + x20285;
int32_t x20288 = x20286 + x20287;
int32_t x20290 = x20288 + x20289;
float x20301 = x20299 * x20300;
x20274[x20290] = x20301;

}

}

}

}
int32_t x20311 = 0;
int32_t x20312 = 1;
x20312 *= 1;
x20311 += 1;
x20312 *= 1;
x20312 *= 1;
int32_t x20317 = x20311;
bool x20318 = x20317 >= 2;
if (x20318) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20323 = x20317 == 0;
if (x20323) {
int32_t x20324 = x20312;
bool x20325 = x20324 == 1024;
if (x20325) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20332 = x20312;
bool x20334 = x20263 == 1;
int32_t x20333 = 1024 / x20332;
bool x20335 = x20333 == 1;
bool x20339;
if (x454) {
bool x20336 = x20334 || x20335;
bool x20337 = x20263 == x20333;
bool x20338 = x20336 || x20337;
x20339 = x20338;
} else {
x20339 = false;
}
bool x20343;
if (x20339) {
x20343 = x20342;
} else {
x20343 = false;
}
bool x20344;
if (x20343) {
x20344 = x20342;
} else {
x20344 = false;
}
if (x20344) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20263,x20265,x20265,1,x20333,1,1);
assert(false && "");
}
bool x20350 = x20263 <= x20333;
int32_t x20351;
if (x20350) {
x20351 = x20333;
} else {
x20351 = x20263;
}
int32_t x20360 = x20351 * x20359;
int32_t x20361 = 64 * x20360;
float* x20362 = (float*)myMalloc(x20361 * sizeof(float));;
int32_t x20363;
if (x20334) {
x20363 = 0;
} else {
x20363 = x20271;
}
int32_t x20366;
if (x20335) {
x20366 = 0;
} else {
x20366 = 1;
}
for(int x20367=0; x20367 < 64; x20367++) {
int32_t x20379 = x20272 * x20367;
int32_t x20373 = x20360 * x20367;
for(int x20369=0; x20369 < x20351; x20369++) {
int32_t x20380 = x20363 * x20369;
int32_t x20381 = x20379 + x20380;
int32_t x20386 = x20366 * x20369;
int32_t x20375 = x20359 * x20369;
for(int x20371=0; x20371 < x20353; x20371++) {
int32_t x20382 = x20364 * x20371;
int32_t x20383 = x20381 + x20382;
int32_t x20377 = x20353 * x20371;
for(int x20372=0; x20372 < x20353; x20372++) {
int32_t x20384 = x20365 * x20372;
int32_t x20385 = x20383 + x20384;
float x20387 = x20274[x20385];
float x20388 = x93[x20386];
int32_t x20374 = x20372 + x20373;
int32_t x20376 = x20374 + x20375;
int32_t x20378 = x20376 + x20377;
float x20389 = x20387 + x20388;
x20362[x20378] = x20389;

}

}

}

}
bool x20399 = x20351 == 1;
bool x20400 = x20399 || x18987;
bool x20401 = x20351 == x18939;
bool x20402 = x20400 || x20401;
bool x20407;
if (x20402) {
x20407 = x20406;
} else {
x20407 = false;
}
bool x20408;
if (x20407) {
x20408 = x20406;
} else {
x20408 = false;
}
if (x20408) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20351,x20353,x20353,64,x18939,x18941,x18941);
assert(false && "");
}
bool x20414 = x20351 <= x18939;
int32_t x20415;
if (x20414) {
x20415 = x18939;
} else {
x20415 = x20351;
}
int32_t x20431;
if (x20399) {
x20431 = 0;
} else {
x20431 = x20359;
}
for(int x20434=0; x20434 < 64; x20434++) {
int32_t x20440 = x20360 * x20434;
int32_t x20447 = x18948 * x20434;
for(int x20436=0; x20436 < x20415; x20436++) {
int32_t x20441 = x20431 * x20436;
int32_t x20442 = x20440 + x20441;
int32_t x20448 = x19019 * x20436;
int32_t x20449 = x20447 + x20448;
for(int x20438=0; x20438 < x20417; x20438++) {
int32_t x20443 = x20432 * x20438;
int32_t x20444 = x20442 + x20443;
int32_t x20450 = x19020 * x20438;
int32_t x20451 = x20449 + x20450;
for(int x20439=0; x20439 < x20417; x20439++) {
int32_t x20445 = x20433 * x20439;
int32_t x20446 = x20444 + x20445;
float x20454 = x20362[x20446];
int32_t x20452 = x19021 * x20439;
int32_t x20453 = x20451 + x20452;
float x20455 = x19054[x20453];
float x20456 = x20454 + x20455;
x20362[x20446] = x20456;

}

}

}

}
float* x20466 = (float*)myMalloc(x20361 * sizeof(float));;
for(int x20468=0; x20468 < x20361; x20468++) {
float x20469 = x20362[x20468];
bool x20470 = x20469 < 0.0f;
if (x20470) {
x20466[x20468] = 0.0f;
} else {
float x20473 = x20362[x20468];
x20466[x20468] = x20473;
}

}
float* x20487 = (float*)myMalloc(x20486 * sizeof(float));;
int32_t x20490 = 64 * x20351;
int32_t x20491 = x20490 * x20482;
float* x20492 = (float*)myMalloc(x20491 * sizeof(float));;
int32_t x20488 = x20351 * x20482;
for(int x20493=0; x20493 < 64; x20493++) {
int32_t x20494 = x20493 * x20360;
float* x20495 = x20466+x20494;
int32_t x20496 = x20493 * x20483;
float* x20497 = x20487+x20496;
int32_t x20498 = x20493 * x20488;
float* x20499 = x20492+x20498;
for(int x20500=0; x20500 < x20351; x20500++) {
int32_t x20501 = x20500 / 1;
int32_t x20505 = x20501 * x20481;
int32_t x20506 = x20505 * x20481;
int32_t x20502 = x20500 % 1;
int32_t x20503 = x20502 / 1;
int32_t x20507 = x20503 * x20481;
int32_t x20508 = x20507 * x20481;
int32_t x20509 = x20506 + x20508;
int32_t x20504 = x20502 % 1;
int32_t x20510 = x20504 * x20481;
int32_t x20511 = x20510 * x20481;
int32_t x20512 = x20509 + x20511;
float* x20513 = x20499+x20512;
int32_t x20514 = x20501 * x20353;
int32_t x20515 = x20514 * x20353;
float* x20516 = x20495+x20515;
for(int x20518=0; x20518 < x20481; x20518++) {
int32_t x20520 = x20518 * x20481;
float* x20521 = x20513+x20520;
int32_t x20519 = x20518 + x20503;
int32_t x20522 = x20519 * x20353;
int32_t x20523 = x20522 + x20504;
float* x20524 = x20516+x20523;
memcpy(x20521, x20524, 4 * x20481);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x20482,x20351,1,x139,x20351,x20499,x20482,1,x20497,x20482);

}
int32_t x20533 = 0;
int32_t x20534 = 1;
x20534 *= 1;
x20533 += 1;
x20534 *= 1;
x20534 *= 1;
int32_t x20539 = x20533;
bool x20540 = x20539 >= 2;
if (x20540) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20545 = x20539 == 0;
if (x20545) {
int32_t x20546 = x20534;
bool x20547 = x20546 == 512;
if (x20547) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20554 = x20534;
int32_t x20555 = 512 / x20554;
bool x20556 = x20555 == 1;
bool x20559;
if (x454) {
bool x20557 = 512 == x20555;
bool x20558 = x20556 || x20557;
x20559 = x20558;
} else {
x20559 = false;
}
bool x20563;
if (x20559) {
x20563 = x20562;
} else {
x20563 = false;
}
bool x20564;
if (x20563) {
x20564 = x20562;
} else {
x20564 = false;
}
if (x20564) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x20481,x20481,1,x20555,1,1);
assert(false && "");
}
bool x20570 = 512 <= x20555;
int32_t x20571;
if (x20570) {
x20571 = x20555;
} else {
x20571 = 512;
}
int32_t x20580 = x20571 * x20579;
int32_t x20581 = 64 * x20580;
float* x20582 = (float*)myMalloc(x20581 * sizeof(float));;
int32_t x20585;
if (x20556) {
x20585 = 0;
} else {
x20585 = 1;
}
for(int x20586=0; x20586 < 64; x20586++) {
int32_t x20598 = x20483 * x20586;
int32_t x20592 = x20580 * x20586;
for(int x20588=0; x20588 < x20571; x20588++) {
int32_t x20599 = x20482 * x20588;
int32_t x20600 = x20598 + x20599;
int32_t x20605 = x20585 * x20588;
int32_t x20594 = x20579 * x20588;
for(int x20590=0; x20590 < x20573; x20590++) {
int32_t x20601 = x20583 * x20590;
int32_t x20602 = x20600 + x20601;
int32_t x20596 = x20573 * x20590;
for(int x20591=0; x20591 < x20573; x20591++) {
int32_t x20603 = x20584 * x20591;
int32_t x20604 = x20602 + x20603;
float x20606 = x20487[x20604];
float x20607 = x67[x20605];
int32_t x20593 = x20591 + x20592;
int32_t x20595 = x20593 + x20594;
int32_t x20597 = x20595 + x20596;
float x20608 = x20606 - x20607;
x20582[x20597] = x20608;

}

}

}

}
float* x20618 = (float*)myMalloc(512 * sizeof(float));;
for(int x20619=0; x20619 < 512; x20619++) {
float x20620 = x121[x20619];
float x20621 = x20620 + 1.0E-5f;
x20618[x20619] = x20621;

}
float* x20625 = (float*)myMalloc(512 * sizeof(float));;
for(int x20626=0; x20626 < 512; x20626++) {
float x20627 = x20618[x20626];
double x20628 = (double)x20627;
double x20629 = sqrt(x20628);
float x20630 = (float)x20629;
x20625[x20626] = x20630;

}
int32_t x20634 = 0;
int32_t x20635 = 1;
x20635 *= 1;
x20634 += 1;
x20635 *= 1;
x20635 *= 1;
int32_t x20640 = x20634;
bool x20641 = x20640 >= 2;
if (x20641) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20646 = x20640 == 0;
if (x20646) {
int32_t x20647 = x20635;
bool x20648 = x20647 == 512;
if (x20648) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20655 = x20635;
bool x20657 = x20571 == 1;
int32_t x20656 = 512 / x20655;
bool x20658 = x20656 == 1;
bool x20662;
if (x454) {
bool x20659 = x20657 || x20658;
bool x20660 = x20571 == x20656;
bool x20661 = x20659 || x20660;
x20662 = x20661;
} else {
x20662 = false;
}
bool x20666;
if (x20662) {
x20666 = x20665;
} else {
x20666 = false;
}
bool x20667;
if (x20666) {
x20667 = x20665;
} else {
x20667 = false;
}
if (x20667) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20571,x20573,x20573,1,x20656,1,1);
assert(false && "");
}
bool x20673 = x20571 <= x20656;
int32_t x20674;
if (x20673) {
x20674 = x20656;
} else {
x20674 = x20571;
}
int32_t x20683 = x20674 * x20682;
int32_t x20684 = 64 * x20683;
float* x20685 = (float*)myMalloc(x20684 * sizeof(float));;
int32_t x20686;
if (x20657) {
x20686 = 0;
} else {
x20686 = x20579;
}
int32_t x20689;
if (x20658) {
x20689 = 0;
} else {
x20689 = 1;
}
for(int x20690=0; x20690 < 64; x20690++) {
int32_t x20702 = x20580 * x20690;
int32_t x20696 = x20683 * x20690;
for(int x20692=0; x20692 < x20674; x20692++) {
int32_t x20703 = x20686 * x20692;
int32_t x20704 = x20702 + x20703;
int32_t x20709 = x20689 * x20692;
int32_t x20698 = x20682 * x20692;
for(int x20694=0; x20694 < x20676; x20694++) {
int32_t x20705 = x20687 * x20694;
int32_t x20706 = x20704 + x20705;
int32_t x20700 = x20676 * x20694;
for(int x20695=0; x20695 < x20676; x20695++) {
int32_t x20707 = x20688 * x20695;
int32_t x20708 = x20706 + x20707;
float x20710 = x20582[x20708];
float x20711 = x20625[x20709];
int32_t x20697 = x20695 + x20696;
int32_t x20699 = x20697 + x20698;
int32_t x20701 = x20699 + x20700;
float x20712 = x20710 / x20711;
x20685[x20701] = x20712;

}

}

}

}
int32_t x20722 = 0;
int32_t x20723 = 1;
x20723 *= 1;
x20722 += 1;
x20723 *= 1;
x20723 *= 1;
int32_t x20728 = x20722;
bool x20729 = x20728 >= 2;
if (x20729) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20734 = x20728 == 0;
if (x20734) {
int32_t x20735 = x20723;
bool x20736 = x20735 == 512;
if (x20736) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20743 = x20723;
bool x20745 = x20674 == 1;
int32_t x20744 = 512 / x20743;
bool x20746 = x20744 == 1;
bool x20750;
if (x454) {
bool x20747 = x20745 || x20746;
bool x20748 = x20674 == x20744;
bool x20749 = x20747 || x20748;
x20750 = x20749;
} else {
x20750 = false;
}
bool x20754;
if (x20750) {
x20754 = x20753;
} else {
x20754 = false;
}
bool x20755;
if (x20754) {
x20755 = x20753;
} else {
x20755 = false;
}
if (x20755) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20674,x20676,x20676,1,x20744,1,1);
assert(false && "");
}
bool x20761 = x20674 <= x20744;
int32_t x20762;
if (x20761) {
x20762 = x20744;
} else {
x20762 = x20674;
}
int32_t x20771 = x20762 * x20770;
int32_t x20772 = 64 * x20771;
float* x20773 = (float*)myMalloc(x20772 * sizeof(float));;
int32_t x20774;
if (x20745) {
x20774 = 0;
} else {
x20774 = x20682;
}
int32_t x20777;
if (x20746) {
x20777 = 0;
} else {
x20777 = 1;
}
for(int x20778=0; x20778 < 64; x20778++) {
int32_t x20790 = x20683 * x20778;
int32_t x20784 = x20771 * x20778;
for(int x20780=0; x20780 < x20762; x20780++) {
int32_t x20791 = x20774 * x20780;
int32_t x20792 = x20790 + x20791;
int32_t x20797 = x20777 * x20780;
int32_t x20786 = x20770 * x20780;
for(int x20782=0; x20782 < x20764; x20782++) {
int32_t x20793 = x20775 * x20782;
int32_t x20794 = x20792 + x20793;
int32_t x20788 = x20764 * x20782;
for(int x20783=0; x20783 < x20764; x20783++) {
int32_t x20795 = x20776 * x20783;
int32_t x20796 = x20794 + x20795;
float x20798 = x20685[x20796];
float x20799 = x201[x20797];
int32_t x20785 = x20783 + x20784;
int32_t x20787 = x20785 + x20786;
int32_t x20789 = x20787 + x20788;
float x20800 = x20798 * x20799;
x20773[x20789] = x20800;

}

}

}

}
int32_t x20810 = 0;
int32_t x20811 = 1;
x20811 *= 1;
x20810 += 1;
x20811 *= 1;
x20811 *= 1;
int32_t x20816 = x20810;
bool x20817 = x20816 >= 2;
if (x20817) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20822 = x20816 == 0;
if (x20822) {
int32_t x20823 = x20811;
bool x20824 = x20823 == 512;
if (x20824) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20831 = x20811;
bool x20833 = x20762 == 1;
int32_t x20832 = 512 / x20831;
bool x20834 = x20832 == 1;
bool x20838;
if (x454) {
bool x20835 = x20833 || x20834;
bool x20836 = x20762 == x20832;
bool x20837 = x20835 || x20836;
x20838 = x20837;
} else {
x20838 = false;
}
bool x20842;
if (x20838) {
x20842 = x20841;
} else {
x20842 = false;
}
bool x20843;
if (x20842) {
x20843 = x20841;
} else {
x20843 = false;
}
if (x20843) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20762,x20764,x20764,1,x20832,1,1);
assert(false && "");
}
bool x20849 = x20762 <= x20832;
int32_t x20850;
if (x20849) {
x20850 = x20832;
} else {
x20850 = x20762;
}
int32_t x20859 = x20850 * x20858;
int32_t x20860 = 64 * x20859;
float* x20861 = (float*)myMalloc(x20860 * sizeof(float));;
int32_t x20862;
if (x20833) {
x20862 = 0;
} else {
x20862 = x20770;
}
int32_t x20865;
if (x20834) {
x20865 = 0;
} else {
x20865 = 1;
}
for(int x20866=0; x20866 < 64; x20866++) {
int32_t x20878 = x20771 * x20866;
int32_t x20872 = x20859 * x20866;
for(int x20868=0; x20868 < x20850; x20868++) {
int32_t x20879 = x20862 * x20868;
int32_t x20880 = x20878 + x20879;
int32_t x20885 = x20865 * x20868;
int32_t x20874 = x20858 * x20868;
for(int x20870=0; x20870 < x20852; x20870++) {
int32_t x20881 = x20863 * x20870;
int32_t x20882 = x20880 + x20881;
int32_t x20876 = x20852 * x20870;
for(int x20871=0; x20871 < x20852; x20871++) {
int32_t x20883 = x20864 * x20871;
int32_t x20884 = x20882 + x20883;
float x20886 = x20773[x20884];
float x20887 = x224[x20885];
int32_t x20873 = x20871 + x20872;
int32_t x20875 = x20873 + x20874;
int32_t x20877 = x20875 + x20876;
float x20888 = x20886 + x20887;
x20861[x20877] = x20888;

}

}

}

}
float* x20898 = (float*)myMalloc(x20860 * sizeof(float));;
for(int x20900=0; x20900 < x20860; x20900++) {
float x20901 = x20861[x20900];
bool x20902 = x20901 < 0.0f;
if (x20902) {
x20898[x20900] = 0.0f;
} else {
float x20905 = x20861[x20900];
x20898[x20900] = x20905;
}

}
float* x20920 = (float*)myMalloc(x20919 * sizeof(float));;
int32_t x20921 = 9 * x20850;
int32_t x20924 = 64 * x20921;
int32_t x20925 = x20924 * x20915;
float* x20926 = (float*)myMalloc(x20925 * sizeof(float));;
int32_t x20922 = x20921 * x20915;
int32_t x20934 = x20850 * 3;
int32_t x20935 = x20934 * 3;
for(int x20927=0; x20927 < 64; x20927++) {
int32_t x20928 = x20927 * x20859;
float* x20929 = x20898+x20928;
int32_t x20930 = x20927 * x20916;
float* x20931 = x20920+x20930;
int32_t x20932 = x20927 * x20922;
float* x20933 = x20926+x20932;
for(int x20937=0; x20937 < x20935; x20937++) {
int32_t x20938 = x20937 / 9;
int32_t x20942 = x20938 * 3;
int32_t x20943 = x20942 * 3;
int32_t x20944 = x20943 * x20914;
int32_t x20945 = x20944 * x20914;
int32_t x20939 = x20937 % 9;
int32_t x20940 = x20939 / 3;
int32_t x20946 = x20940 * 3;
int32_t x20947 = x20946 * x20914;
int32_t x20948 = x20947 * x20914;
int32_t x20949 = x20945 + x20948;
int32_t x20941 = x20939 % 3;
int32_t x20950 = x20941 * x20914;
int32_t x20951 = x20950 * x20914;
int32_t x20952 = x20949 + x20951;
float* x20953 = x20933+x20952;
int32_t x20954 = x20938 * x20852;
int32_t x20955 = x20954 * x20852;
float* x20956 = x20929+x20955;
for(int x20958=0; x20958 < x20914; x20958++) {
int32_t x20959 = x20958 * 2;
int32_t x20960 = x20959 - 1;
int32_t x20961 = x20960 + x20940;
bool x20962 = x20961 < 0;
bool x20963 = x20961 >= x20852;
bool x20964 = x20962 || x20963;
if (x20964) {
int32_t x20965 = x20958 * x20914;
float* x20966 = x20953+x20965;
memset(x20966, 0, 4 * x20914);;
} else {
int32_t x20965 = x20958 * x20914;
int32_t x20981 = x20961 * x20852;
for(int x20969=0; x20969 < x20914; x20969++) {
int32_t x20970 = x20969 * 2;
int32_t x20971 = x20970 - 1;
int32_t x20972 = x20971 + x20941;
bool x20973 = x20972 < 0;
bool x20974 = x20972 >= x20852;
bool x20975 = x20973 || x20974;
if (x20975) {
int32_t x20976 = x20965 + x20969;
float* x20977 = x20953+x20976;
memset(x20977, 0, 4 * 1);;
} else {
int32_t x20976 = x20965 + x20969;
float* x20980 = x20953+x20976;
int32_t x20982 = x20981 + x20972;
float* x20983 = x20956+x20982;
memcpy(x20980, x20983, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x20915,x20921,1,x34,x20921,x20933,x20915,1,x20931,x20915);

}
int32_t x20998 = 0;
int32_t x20999 = 1;
x20999 *= 1;
x20998 += 1;
x20999 *= 1;
x20999 *= 1;
int32_t x21004 = x20998;
bool x21005 = x21004 >= 2;
if (x21005) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21010 = x21004 == 0;
if (x21010) {
int32_t x21011 = x20999;
bool x21012 = x21011 == 512;
if (x21012) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21019 = x20999;
int32_t x21020 = 512 / x21019;
bool x21021 = x21020 == 1;
bool x21024;
if (x454) {
bool x21022 = 512 == x21020;
bool x21023 = x21021 || x21022;
x21024 = x21023;
} else {
x21024 = false;
}
bool x21028;
if (x21024) {
x21028 = x21027;
} else {
x21028 = false;
}
bool x21029;
if (x21028) {
x21029 = x21027;
} else {
x21029 = false;
}
if (x21029) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x20914,x20914,1,x21020,1,1);
assert(false && "");
}
bool x21035 = 512 <= x21020;
int32_t x21036;
if (x21035) {
x21036 = x21020;
} else {
x21036 = 512;
}
int32_t x21045 = x21036 * x21044;
int32_t x21046 = 64 * x21045;
float* x21047 = (float*)myMalloc(x21046 * sizeof(float));;
int32_t x21050;
if (x21021) {
x21050 = 0;
} else {
x21050 = 1;
}
for(int x21051=0; x21051 < 64; x21051++) {
int32_t x21063 = x20916 * x21051;
int32_t x21057 = x21045 * x21051;
for(int x21053=0; x21053 < x21036; x21053++) {
int32_t x21064 = x20915 * x21053;
int32_t x21065 = x21063 + x21064;
int32_t x21070 = x21050 * x21053;
int32_t x21059 = x21044 * x21053;
for(int x21055=0; x21055 < x21038; x21055++) {
int32_t x21066 = x21048 * x21055;
int32_t x21067 = x21065 + x21066;
int32_t x21061 = x21038 * x21055;
for(int x21056=0; x21056 < x21038; x21056++) {
int32_t x21068 = x21049 * x21056;
int32_t x21069 = x21067 + x21068;
float x21071 = x20920[x21069];
float x21072 = x113[x21070];
int32_t x21058 = x21056 + x21057;
int32_t x21060 = x21058 + x21059;
int32_t x21062 = x21060 + x21061;
float x21073 = x21071 - x21072;
x21047[x21062] = x21073;

}

}

}

}
float* x21083 = (float*)myMalloc(512 * sizeof(float));;
for(int x21084=0; x21084 < 512; x21084++) {
float x21085 = x50[x21084];
float x21086 = x21085 + 1.0E-5f;
x21083[x21084] = x21086;

}
float* x21090 = (float*)myMalloc(512 * sizeof(float));;
for(int x21091=0; x21091 < 512; x21091++) {
float x21092 = x21083[x21091];
double x21093 = (double)x21092;
double x21094 = sqrt(x21093);
float x21095 = (float)x21094;
x21090[x21091] = x21095;

}
int32_t x21099 = 0;
int32_t x21100 = 1;
x21100 *= 1;
x21099 += 1;
x21100 *= 1;
x21100 *= 1;
int32_t x21105 = x21099;
bool x21106 = x21105 >= 2;
if (x21106) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21111 = x21105 == 0;
if (x21111) {
int32_t x21112 = x21100;
bool x21113 = x21112 == 512;
if (x21113) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21120 = x21100;
bool x21122 = x21036 == 1;
int32_t x21121 = 512 / x21120;
bool x21123 = x21121 == 1;
bool x21127;
if (x454) {
bool x21124 = x21122 || x21123;
bool x21125 = x21036 == x21121;
bool x21126 = x21124 || x21125;
x21127 = x21126;
} else {
x21127 = false;
}
bool x21131;
if (x21127) {
x21131 = x21130;
} else {
x21131 = false;
}
bool x21132;
if (x21131) {
x21132 = x21130;
} else {
x21132 = false;
}
if (x21132) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21036,x21038,x21038,1,x21121,1,1);
assert(false && "");
}
bool x21138 = x21036 <= x21121;
int32_t x21139;
if (x21138) {
x21139 = x21121;
} else {
x21139 = x21036;
}
int32_t x21148 = x21139 * x21147;
int32_t x21149 = 64 * x21148;
float* x21150 = (float*)myMalloc(x21149 * sizeof(float));;
int32_t x21151;
if (x21122) {
x21151 = 0;
} else {
x21151 = x21044;
}
int32_t x21154;
if (x21123) {
x21154 = 0;
} else {
x21154 = 1;
}
for(int x21155=0; x21155 < 64; x21155++) {
int32_t x21167 = x21045 * x21155;
int32_t x21161 = x21148 * x21155;
for(int x21157=0; x21157 < x21139; x21157++) {
int32_t x21168 = x21151 * x21157;
int32_t x21169 = x21167 + x21168;
int32_t x21174 = x21154 * x21157;
int32_t x21163 = x21147 * x21157;
for(int x21159=0; x21159 < x21141; x21159++) {
int32_t x21170 = x21152 * x21159;
int32_t x21171 = x21169 + x21170;
int32_t x21165 = x21141 * x21159;
for(int x21160=0; x21160 < x21141; x21160++) {
int32_t x21172 = x21153 * x21160;
int32_t x21173 = x21171 + x21172;
float x21175 = x21047[x21173];
float x21176 = x21090[x21174];
int32_t x21162 = x21160 + x21161;
int32_t x21164 = x21162 + x21163;
int32_t x21166 = x21164 + x21165;
float x21177 = x21175 / x21176;
x21150[x21166] = x21177;

}

}

}

}
int32_t x21187 = 0;
int32_t x21188 = 1;
x21188 *= 1;
x21187 += 1;
x21188 *= 1;
x21188 *= 1;
int32_t x21193 = x21187;
bool x21194 = x21193 >= 2;
if (x21194) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21199 = x21193 == 0;
if (x21199) {
int32_t x21200 = x21188;
bool x21201 = x21200 == 512;
if (x21201) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21208 = x21188;
bool x21210 = x21139 == 1;
int32_t x21209 = 512 / x21208;
bool x21211 = x21209 == 1;
bool x21215;
if (x454) {
bool x21212 = x21210 || x21211;
bool x21213 = x21139 == x21209;
bool x21214 = x21212 || x21213;
x21215 = x21214;
} else {
x21215 = false;
}
bool x21219;
if (x21215) {
x21219 = x21218;
} else {
x21219 = false;
}
bool x21220;
if (x21219) {
x21220 = x21218;
} else {
x21220 = false;
}
if (x21220) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21139,x21141,x21141,1,x21209,1,1);
assert(false && "");
}
bool x21226 = x21139 <= x21209;
int32_t x21227;
if (x21226) {
x21227 = x21209;
} else {
x21227 = x21139;
}
int32_t x21236 = x21227 * x21235;
int32_t x21237 = 64 * x21236;
float* x21238 = (float*)myMalloc(x21237 * sizeof(float));;
int32_t x21239;
if (x21210) {
x21239 = 0;
} else {
x21239 = x21147;
}
int32_t x21242;
if (x21211) {
x21242 = 0;
} else {
x21242 = 1;
}
for(int x21243=0; x21243 < 64; x21243++) {
int32_t x21255 = x21148 * x21243;
int32_t x21249 = x21236 * x21243;
for(int x21245=0; x21245 < x21227; x21245++) {
int32_t x21256 = x21239 * x21245;
int32_t x21257 = x21255 + x21256;
int32_t x21262 = x21242 * x21245;
int32_t x21251 = x21235 * x21245;
for(int x21247=0; x21247 < x21229; x21247++) {
int32_t x21258 = x21240 * x21247;
int32_t x21259 = x21257 + x21258;
int32_t x21253 = x21229 * x21247;
for(int x21248=0; x21248 < x21229; x21248++) {
int32_t x21260 = x21241 * x21248;
int32_t x21261 = x21259 + x21260;
float x21263 = x21150[x21261];
float x21264 = x205[x21262];
int32_t x21250 = x21248 + x21249;
int32_t x21252 = x21250 + x21251;
int32_t x21254 = x21252 + x21253;
float x21265 = x21263 * x21264;
x21238[x21254] = x21265;

}

}

}

}
int32_t x21275 = 0;
int32_t x21276 = 1;
x21276 *= 1;
x21275 += 1;
x21276 *= 1;
x21276 *= 1;
int32_t x21281 = x21275;
bool x21282 = x21281 >= 2;
if (x21282) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21287 = x21281 == 0;
if (x21287) {
int32_t x21288 = x21276;
bool x21289 = x21288 == 512;
if (x21289) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21296 = x21276;
bool x21298 = x21227 == 1;
int32_t x21297 = 512 / x21296;
bool x21299 = x21297 == 1;
bool x21303;
if (x454) {
bool x21300 = x21298 || x21299;
bool x21301 = x21227 == x21297;
bool x21302 = x21300 || x21301;
x21303 = x21302;
} else {
x21303 = false;
}
bool x21307;
if (x21303) {
x21307 = x21306;
} else {
x21307 = false;
}
bool x21308;
if (x21307) {
x21308 = x21306;
} else {
x21308 = false;
}
if (x21308) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21227,x21229,x21229,1,x21297,1,1);
assert(false && "");
}
bool x21314 = x21227 <= x21297;
int32_t x21315;
if (x21314) {
x21315 = x21297;
} else {
x21315 = x21227;
}
int32_t x21324 = x21315 * x21323;
int32_t x21325 = 64 * x21324;
float* x21326 = (float*)myMalloc(x21325 * sizeof(float));;
int32_t x21327;
if (x21298) {
x21327 = 0;
} else {
x21327 = x21235;
}
int32_t x21330;
if (x21299) {
x21330 = 0;
} else {
x21330 = 1;
}
for(int x21331=0; x21331 < 64; x21331++) {
int32_t x21343 = x21236 * x21331;
int32_t x21337 = x21324 * x21331;
for(int x21333=0; x21333 < x21315; x21333++) {
int32_t x21344 = x21327 * x21333;
int32_t x21345 = x21343 + x21344;
int32_t x21350 = x21330 * x21333;
int32_t x21339 = x21323 * x21333;
for(int x21335=0; x21335 < x21317; x21335++) {
int32_t x21346 = x21328 * x21335;
int32_t x21347 = x21345 + x21346;
int32_t x21341 = x21317 * x21335;
for(int x21336=0; x21336 < x21317; x21336++) {
int32_t x21348 = x21329 * x21336;
int32_t x21349 = x21347 + x21348;
float x21351 = x21238[x21349];
float x21352 = x159[x21350];
int32_t x21338 = x21336 + x21337;
int32_t x21340 = x21338 + x21339;
int32_t x21342 = x21340 + x21341;
float x21353 = x21351 + x21352;
x21326[x21342] = x21353;

}

}

}

}
float* x21363 = (float*)myMalloc(x21325 * sizeof(float));;
for(int x21365=0; x21365 < x21325; x21365++) {
float x21366 = x21326[x21365];
bool x21367 = x21366 < 0.0f;
if (x21367) {
x21363[x21365] = 0.0f;
} else {
float x21370 = x21326[x21365];
x21363[x21365] = x21370;
}

}
float* x21384 = (float*)myMalloc(x21383 * sizeof(float));;
int32_t x21387 = 64 * x21315;
int32_t x21388 = x21387 * x21379;
float* x21389 = (float*)myMalloc(x21388 * sizeof(float));;
int32_t x21385 = x21315 * x21379;
for(int x21390=0; x21390 < 64; x21390++) {
int32_t x21391 = x21390 * x21324;
float* x21392 = x21363+x21391;
int32_t x21393 = x21390 * x21380;
float* x21394 = x21384+x21393;
int32_t x21395 = x21390 * x21385;
float* x21396 = x21389+x21395;
for(int x21397=0; x21397 < x21315; x21397++) {
int32_t x21398 = x21397 / 1;
int32_t x21402 = x21398 * x21378;
int32_t x21403 = x21402 * x21378;
int32_t x21399 = x21397 % 1;
int32_t x21400 = x21399 / 1;
int32_t x21404 = x21400 * x21378;
int32_t x21405 = x21404 * x21378;
int32_t x21406 = x21403 + x21405;
int32_t x21401 = x21399 % 1;
int32_t x21407 = x21401 * x21378;
int32_t x21408 = x21407 * x21378;
int32_t x21409 = x21406 + x21408;
float* x21410 = x21396+x21409;
int32_t x21411 = x21398 * x21317;
int32_t x21412 = x21411 * x21317;
float* x21413 = x21392+x21412;
for(int x21415=0; x21415 < x21378; x21415++) {
int32_t x21417 = x21415 * x21378;
float* x21418 = x21410+x21417;
int32_t x21416 = x21415 + x21400;
int32_t x21419 = x21416 * x21317;
int32_t x21420 = x21419 + x21401;
float* x21421 = x21413+x21420;
memcpy(x21418, x21421, 4 * x21378);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x21379,x21315,1,x212,x21315,x21396,x21379,1,x21394,x21379);

}
int32_t x21430 = 0;
int32_t x21431 = 1;
x21431 *= 1;
x21430 += 1;
x21431 *= 1;
x21431 *= 1;
int32_t x21436 = x21430;
bool x21437 = x21436 >= 2;
if (x21437) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21442 = x21436 == 0;
if (x21442) {
int32_t x21443 = x21431;
bool x21444 = x21443 == 2048;
if (x21444) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21451 = x21431;
int32_t x21452 = 2048 / x21451;
bool x21453 = x21452 == 1;
bool x21456;
if (x454) {
bool x21454 = 2048 == x21452;
bool x21455 = x21453 || x21454;
x21456 = x21455;
} else {
x21456 = false;
}
bool x21460;
if (x21456) {
x21460 = x21459;
} else {
x21460 = false;
}
bool x21461;
if (x21460) {
x21461 = x21459;
} else {
x21461 = false;
}
if (x21461) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x21378,x21378,1,x21452,1,1);
assert(false && "");
}
bool x21467 = 2048 <= x21452;
int32_t x21468;
if (x21467) {
x21468 = x21452;
} else {
x21468 = 2048;
}
int32_t x21477 = x21468 * x21476;
int32_t x21478 = 64 * x21477;
float* x21479 = (float*)myMalloc(x21478 * sizeof(float));;
int32_t x21482;
if (x21453) {
x21482 = 0;
} else {
x21482 = 1;
}
for(int x21483=0; x21483 < 64; x21483++) {
int32_t x21495 = x21380 * x21483;
int32_t x21489 = x21477 * x21483;
for(int x21485=0; x21485 < x21468; x21485++) {
int32_t x21496 = x21379 * x21485;
int32_t x21497 = x21495 + x21496;
int32_t x21502 = x21482 * x21485;
int32_t x21491 = x21476 * x21485;
for(int x21487=0; x21487 < x21470; x21487++) {
int32_t x21498 = x21480 * x21487;
int32_t x21499 = x21497 + x21498;
int32_t x21493 = x21470 * x21487;
for(int x21488=0; x21488 < x21470; x21488++) {
int32_t x21500 = x21481 * x21488;
int32_t x21501 = x21499 + x21500;
float x21503 = x21384[x21501];
float x21504 = x115[x21502];
int32_t x21490 = x21488 + x21489;
int32_t x21492 = x21490 + x21491;
int32_t x21494 = x21492 + x21493;
float x21505 = x21503 - x21504;
x21479[x21494] = x21505;

}

}

}

}
float* x21515 = (float*)myMalloc(2048 * sizeof(float));;
for(int x21517=0; x21517 < 2048; x21517++) {
float x21518 = x193[x21517];
float x21519 = x21518 + 1.0E-5f;
x21515[x21517] = x21519;

}
float* x21523 = (float*)myMalloc(2048 * sizeof(float));;
for(int x21524=0; x21524 < 2048; x21524++) {
float x21525 = x21515[x21524];
double x21526 = (double)x21525;
double x21527 = sqrt(x21526);
float x21528 = (float)x21527;
x21523[x21524] = x21528;

}
int32_t x21532 = 0;
int32_t x21533 = 1;
x21533 *= 1;
x21532 += 1;
x21533 *= 1;
x21533 *= 1;
int32_t x21538 = x21532;
bool x21539 = x21538 >= 2;
if (x21539) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21544 = x21538 == 0;
if (x21544) {
int32_t x21545 = x21533;
bool x21546 = x21545 == 2048;
if (x21546) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21553 = x21533;
bool x21555 = x21468 == 1;
int32_t x21554 = 2048 / x21553;
bool x21556 = x21554 == 1;
bool x21560;
if (x454) {
bool x21557 = x21555 || x21556;
bool x21558 = x21468 == x21554;
bool x21559 = x21557 || x21558;
x21560 = x21559;
} else {
x21560 = false;
}
bool x21564;
if (x21560) {
x21564 = x21563;
} else {
x21564 = false;
}
bool x21565;
if (x21564) {
x21565 = x21563;
} else {
x21565 = false;
}
if (x21565) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21468,x21470,x21470,1,x21554,1,1);
assert(false && "");
}
bool x21571 = x21468 <= x21554;
int32_t x21572;
if (x21571) {
x21572 = x21554;
} else {
x21572 = x21468;
}
int32_t x21581 = x21572 * x21580;
int32_t x21582 = 64 * x21581;
float* x21583 = (float*)myMalloc(x21582 * sizeof(float));;
int32_t x21584;
if (x21555) {
x21584 = 0;
} else {
x21584 = x21476;
}
int32_t x21587;
if (x21556) {
x21587 = 0;
} else {
x21587 = 1;
}
for(int x21588=0; x21588 < 64; x21588++) {
int32_t x21600 = x21477 * x21588;
int32_t x21594 = x21581 * x21588;
for(int x21590=0; x21590 < x21572; x21590++) {
int32_t x21601 = x21584 * x21590;
int32_t x21602 = x21600 + x21601;
int32_t x21607 = x21587 * x21590;
int32_t x21596 = x21580 * x21590;
for(int x21592=0; x21592 < x21574; x21592++) {
int32_t x21603 = x21585 * x21592;
int32_t x21604 = x21602 + x21603;
int32_t x21598 = x21574 * x21592;
for(int x21593=0; x21593 < x21574; x21593++) {
int32_t x21605 = x21586 * x21593;
int32_t x21606 = x21604 + x21605;
float x21608 = x21479[x21606];
float x21609 = x21523[x21607];
int32_t x21595 = x21593 + x21594;
int32_t x21597 = x21595 + x21596;
int32_t x21599 = x21597 + x21598;
float x21610 = x21608 / x21609;
x21583[x21599] = x21610;

}

}

}

}
int32_t x21620 = 0;
int32_t x21621 = 1;
x21621 *= 1;
x21620 += 1;
x21621 *= 1;
x21621 *= 1;
int32_t x21626 = x21620;
bool x21627 = x21626 >= 2;
if (x21627) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21632 = x21626 == 0;
if (x21632) {
int32_t x21633 = x21621;
bool x21634 = x21633 == 2048;
if (x21634) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21641 = x21621;
bool x21643 = x21572 == 1;
int32_t x21642 = 2048 / x21641;
bool x21644 = x21642 == 1;
bool x21648;
if (x454) {
bool x21645 = x21643 || x21644;
bool x21646 = x21572 == x21642;
bool x21647 = x21645 || x21646;
x21648 = x21647;
} else {
x21648 = false;
}
bool x21652;
if (x21648) {
x21652 = x21651;
} else {
x21652 = false;
}
bool x21653;
if (x21652) {
x21653 = x21651;
} else {
x21653 = false;
}
if (x21653) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21572,x21574,x21574,1,x21642,1,1);
assert(false && "");
}
bool x21659 = x21572 <= x21642;
int32_t x21660;
if (x21659) {
x21660 = x21642;
} else {
x21660 = x21572;
}
int32_t x21669 = x21660 * x21668;
int32_t x21670 = 64 * x21669;
float* x21671 = (float*)myMalloc(x21670 * sizeof(float));;
int32_t x21672;
if (x21643) {
x21672 = 0;
} else {
x21672 = x21580;
}
int32_t x21675;
if (x21644) {
x21675 = 0;
} else {
x21675 = 1;
}
for(int x21676=0; x21676 < 64; x21676++) {
int32_t x21688 = x21581 * x21676;
int32_t x21682 = x21669 * x21676;
for(int x21678=0; x21678 < x21660; x21678++) {
int32_t x21689 = x21672 * x21678;
int32_t x21690 = x21688 + x21689;
int32_t x21695 = x21675 * x21678;
int32_t x21684 = x21668 * x21678;
for(int x21680=0; x21680 < x21662; x21680++) {
int32_t x21691 = x21673 * x21680;
int32_t x21692 = x21690 + x21691;
int32_t x21686 = x21662 * x21680;
for(int x21681=0; x21681 < x21662; x21681++) {
int32_t x21693 = x21674 * x21681;
int32_t x21694 = x21692 + x21693;
float x21696 = x21583[x21694];
float x21697 = x239[x21695];
int32_t x21683 = x21681 + x21682;
int32_t x21685 = x21683 + x21684;
int32_t x21687 = x21685 + x21686;
float x21698 = x21696 * x21697;
x21671[x21687] = x21698;

}

}

}

}
int32_t x21708 = 0;
int32_t x21709 = 1;
x21709 *= 1;
x21708 += 1;
x21709 *= 1;
x21709 *= 1;
int32_t x21714 = x21708;
bool x21715 = x21714 >= 2;
if (x21715) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21720 = x21714 == 0;
if (x21720) {
int32_t x21721 = x21709;
bool x21722 = x21721 == 2048;
if (x21722) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21729 = x21709;
bool x21731 = x21660 == 1;
int32_t x21730 = 2048 / x21729;
bool x21732 = x21730 == 1;
bool x21736;
if (x454) {
bool x21733 = x21731 || x21732;
bool x21734 = x21660 == x21730;
bool x21735 = x21733 || x21734;
x21736 = x21735;
} else {
x21736 = false;
}
bool x21740;
if (x21736) {
x21740 = x21739;
} else {
x21740 = false;
}
bool x21741;
if (x21740) {
x21741 = x21739;
} else {
x21741 = false;
}
if (x21741) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21660,x21662,x21662,1,x21730,1,1);
assert(false && "");
}
bool x21747 = x21660 <= x21730;
int32_t x21748;
if (x21747) {
x21748 = x21730;
} else {
x21748 = x21660;
}
int32_t x21757 = x21748 * x21756;
int32_t x21758 = 64 * x21757;
float* x21759 = (float*)myMalloc(x21758 * sizeof(float));;
int32_t x21760;
if (x21731) {
x21760 = 0;
} else {
x21760 = x21668;
}
int32_t x21763;
if (x21732) {
x21763 = 0;
} else {
x21763 = 1;
}
for(int x21764=0; x21764 < 64; x21764++) {
int32_t x21776 = x21669 * x21764;
int32_t x21770 = x21757 * x21764;
for(int x21766=0; x21766 < x21748; x21766++) {
int32_t x21777 = x21760 * x21766;
int32_t x21778 = x21776 + x21777;
int32_t x21783 = x21763 * x21766;
int32_t x21772 = x21756 * x21766;
for(int x21768=0; x21768 < x21750; x21768++) {
int32_t x21779 = x21761 * x21768;
int32_t x21780 = x21778 + x21779;
int32_t x21774 = x21750 * x21768;
for(int x21769=0; x21769 < x21750; x21769++) {
int32_t x21781 = x21762 * x21769;
int32_t x21782 = x21780 + x21781;
float x21784 = x21671[x21782];
float x21785 = x62[x21783];
int32_t x21771 = x21769 + x21770;
int32_t x21773 = x21771 + x21772;
int32_t x21775 = x21773 + x21774;
float x21786 = x21784 + x21785;
x21759[x21775] = x21786;

}

}

}

}
float* x21803 = (float*)myMalloc(x21802 * sizeof(float));;
int32_t x21806 = x20490 * x21798;
float* x21807 = (float*)myMalloc(x21806 * sizeof(float));;
int32_t x21804 = x20351 * x21798;
for(int x21808=0; x21808 < 64; x21808++) {
int32_t x21809 = x21808 * x20360;
float* x21810 = x20466+x21809;
int32_t x21811 = x21808 * x21799;
float* x21812 = x21803+x21811;
int32_t x21813 = x21808 * x21804;
float* x21814 = x21807+x21813;
for(int x21815=0; x21815 < x20351; x21815++) {
int32_t x21816 = x21815 / 1;
int32_t x21820 = x21816 * x21797;
int32_t x21821 = x21820 * x21797;
int32_t x21817 = x21815 % 1;
int32_t x21818 = x21817 / 1;
int32_t x21822 = x21818 * x21797;
int32_t x21823 = x21822 * x21797;
int32_t x21824 = x21821 + x21823;
int32_t x21819 = x21817 % 1;
int32_t x21825 = x21819 * x21797;
int32_t x21826 = x21825 * x21797;
int32_t x21827 = x21824 + x21826;
float* x21828 = x21814+x21827;
int32_t x21829 = x21816 * x20353;
int32_t x21830 = x21829 * x20353;
float* x21831 = x21810+x21830;
for(int x21833=0; x21833 < x21797; x21833++) {
int32_t x21837 = x21833 * x21797;
int32_t x21834 = x21833 * 2;
int32_t x21835 = x21834 + x21818;
int32_t x21840 = x21835 * x20353;
int32_t x21841 = x21840 + x21819;
for(int x21836=0; x21836 < x21797; x21836++) {
int32_t x21838 = x21837 + x21836;
float* x21839 = x21828+x21838;
int32_t x21842 = x21836 * 2;
int32_t x21843 = x21841 + x21842;
float* x21844 = x21831+x21843;
memcpy(x21839, x21844, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x21798,x20351,1,x214,x20351,x21814,x21798,1,x21812,x21798);

}
int32_t x21855 = 0;
int32_t x21856 = 1;
x21856 *= 1;
x21855 += 1;
x21856 *= 1;
x21856 *= 1;
int32_t x21861 = x21855;
bool x21862 = x21861 >= 2;
if (x21862) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21867 = x21861 == 0;
if (x21867) {
int32_t x21868 = x21856;
bool x21869 = x21868 == 2048;
if (x21869) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21876 = x21856;
int32_t x21877 = 2048 / x21876;
bool x21878 = x21877 == 1;
bool x21881;
if (x454) {
bool x21879 = 2048 == x21877;
bool x21880 = x21878 || x21879;
x21881 = x21880;
} else {
x21881 = false;
}
bool x21885;
if (x21881) {
x21885 = x21884;
} else {
x21885 = false;
}
bool x21886;
if (x21885) {
x21886 = x21884;
} else {
x21886 = false;
}
if (x21886) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x21797,x21797,1,x21877,1,1);
assert(false && "");
}
bool x21892 = 2048 <= x21877;
int32_t x21893;
if (x21892) {
x21893 = x21877;
} else {
x21893 = 2048;
}
int32_t x21902 = x21893 * x21901;
int32_t x21903 = 64 * x21902;
float* x21904 = (float*)myMalloc(x21903 * sizeof(float));;
int32_t x21907;
if (x21878) {
x21907 = 0;
} else {
x21907 = 1;
}
for(int x21908=0; x21908 < 64; x21908++) {
int32_t x21920 = x21799 * x21908;
int32_t x21914 = x21902 * x21908;
for(int x21910=0; x21910 < x21893; x21910++) {
int32_t x21921 = x21798 * x21910;
int32_t x21922 = x21920 + x21921;
int32_t x21927 = x21907 * x21910;
int32_t x21916 = x21901 * x21910;
for(int x21912=0; x21912 < x21895; x21912++) {
int32_t x21923 = x21905 * x21912;
int32_t x21924 = x21922 + x21923;
int32_t x21918 = x21895 * x21912;
for(int x21913=0; x21913 < x21895; x21913++) {
int32_t x21925 = x21906 * x21913;
int32_t x21926 = x21924 + x21925;
float x21928 = x21803[x21926];
float x21929 = x64[x21927];
int32_t x21915 = x21913 + x21914;
int32_t x21917 = x21915 + x21916;
int32_t x21919 = x21917 + x21918;
float x21930 = x21928 - x21929;
x21904[x21919] = x21930;

}

}

}

}
float* x21940 = (float*)myMalloc(2048 * sizeof(float));;
for(int x21941=0; x21941 < 2048; x21941++) {
float x21942 = x125[x21941];
float x21943 = x21942 + 1.0E-5f;
x21940[x21941] = x21943;

}
float* x21947 = (float*)myMalloc(2048 * sizeof(float));;
for(int x21948=0; x21948 < 2048; x21948++) {
float x21949 = x21940[x21948];
double x21950 = (double)x21949;
double x21951 = sqrt(x21950);
float x21952 = (float)x21951;
x21947[x21948] = x21952;

}
int32_t x21956 = 0;
int32_t x21957 = 1;
x21957 *= 1;
x21956 += 1;
x21957 *= 1;
x21957 *= 1;
int32_t x21962 = x21956;
bool x21963 = x21962 >= 2;
if (x21963) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21968 = x21962 == 0;
if (x21968) {
int32_t x21969 = x21957;
bool x21970 = x21969 == 2048;
if (x21970) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21977 = x21957;
bool x21979 = x21893 == 1;
int32_t x21978 = 2048 / x21977;
bool x21980 = x21978 == 1;
bool x21984;
if (x454) {
bool x21981 = x21979 || x21980;
bool x21982 = x21893 == x21978;
bool x21983 = x21981 || x21982;
x21984 = x21983;
} else {
x21984 = false;
}
bool x21988;
if (x21984) {
x21988 = x21987;
} else {
x21988 = false;
}
bool x21989;
if (x21988) {
x21989 = x21987;
} else {
x21989 = false;
}
if (x21989) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21893,x21895,x21895,1,x21978,1,1);
assert(false && "");
}
bool x21995 = x21893 <= x21978;
int32_t x21996;
if (x21995) {
x21996 = x21978;
} else {
x21996 = x21893;
}
int32_t x22005 = x21996 * x22004;
int32_t x22006 = 64 * x22005;
float* x22007 = (float*)myMalloc(x22006 * sizeof(float));;
int32_t x22008;
if (x21979) {
x22008 = 0;
} else {
x22008 = x21901;
}
int32_t x22011;
if (x21980) {
x22011 = 0;
} else {
x22011 = 1;
}
for(int x22012=0; x22012 < 64; x22012++) {
int32_t x22024 = x21902 * x22012;
int32_t x22018 = x22005 * x22012;
for(int x22014=0; x22014 < x21996; x22014++) {
int32_t x22025 = x22008 * x22014;
int32_t x22026 = x22024 + x22025;
int32_t x22031 = x22011 * x22014;
int32_t x22020 = x22004 * x22014;
for(int x22016=0; x22016 < x21998; x22016++) {
int32_t x22027 = x22009 * x22016;
int32_t x22028 = x22026 + x22027;
int32_t x22022 = x21998 * x22016;
for(int x22017=0; x22017 < x21998; x22017++) {
int32_t x22029 = x22010 * x22017;
int32_t x22030 = x22028 + x22029;
float x22032 = x21904[x22030];
float x22033 = x21947[x22031];
int32_t x22019 = x22017 + x22018;
int32_t x22021 = x22019 + x22020;
int32_t x22023 = x22021 + x22022;
float x22034 = x22032 / x22033;
x22007[x22023] = x22034;

}

}

}

}
int32_t x22044 = 0;
int32_t x22045 = 1;
x22045 *= 1;
x22044 += 1;
x22045 *= 1;
x22045 *= 1;
int32_t x22050 = x22044;
bool x22051 = x22050 >= 2;
if (x22051) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22056 = x22050 == 0;
if (x22056) {
int32_t x22057 = x22045;
bool x22058 = x22057 == 2048;
if (x22058) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22065 = x22045;
bool x22067 = x21996 == 1;
int32_t x22066 = 2048 / x22065;
bool x22068 = x22066 == 1;
bool x22072;
if (x454) {
bool x22069 = x22067 || x22068;
bool x22070 = x21996 == x22066;
bool x22071 = x22069 || x22070;
x22072 = x22071;
} else {
x22072 = false;
}
bool x22076;
if (x22072) {
x22076 = x22075;
} else {
x22076 = false;
}
bool x22077;
if (x22076) {
x22077 = x22075;
} else {
x22077 = false;
}
if (x22077) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21996,x21998,x21998,1,x22066,1,1);
assert(false && "");
}
bool x22083 = x21996 <= x22066;
int32_t x22084;
if (x22083) {
x22084 = x22066;
} else {
x22084 = x21996;
}
int32_t x22093 = x22084 * x22092;
int32_t x22094 = 64 * x22093;
float* x22095 = (float*)myMalloc(x22094 * sizeof(float));;
int32_t x22096;
if (x22067) {
x22096 = 0;
} else {
x22096 = x22004;
}
int32_t x22099;
if (x22068) {
x22099 = 0;
} else {
x22099 = 1;
}
for(int x22100=0; x22100 < 64; x22100++) {
int32_t x22112 = x22005 * x22100;
int32_t x22106 = x22093 * x22100;
for(int x22102=0; x22102 < x22084; x22102++) {
int32_t x22113 = x22096 * x22102;
int32_t x22114 = x22112 + x22113;
int32_t x22119 = x22099 * x22102;
int32_t x22108 = x22092 * x22102;
for(int x22104=0; x22104 < x22086; x22104++) {
int32_t x22115 = x22097 * x22104;
int32_t x22116 = x22114 + x22115;
int32_t x22110 = x22086 * x22104;
for(int x22105=0; x22105 < x22086; x22105++) {
int32_t x22117 = x22098 * x22105;
int32_t x22118 = x22116 + x22117;
float x22120 = x22007[x22118];
float x22121 = x173[x22119];
int32_t x22107 = x22105 + x22106;
int32_t x22109 = x22107 + x22108;
int32_t x22111 = x22109 + x22110;
float x22122 = x22120 * x22121;
x22095[x22111] = x22122;

}

}

}

}
int32_t x22132 = 0;
int32_t x22133 = 1;
x22133 *= 1;
x22132 += 1;
x22133 *= 1;
x22133 *= 1;
int32_t x22138 = x22132;
bool x22139 = x22138 >= 2;
if (x22139) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22144 = x22138 == 0;
if (x22144) {
int32_t x22145 = x22133;
bool x22146 = x22145 == 2048;
if (x22146) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22153 = x22133;
bool x22155 = x22084 == 1;
int32_t x22154 = 2048 / x22153;
bool x22156 = x22154 == 1;
bool x22160;
if (x454) {
bool x22157 = x22155 || x22156;
bool x22158 = x22084 == x22154;
bool x22159 = x22157 || x22158;
x22160 = x22159;
} else {
x22160 = false;
}
bool x22164;
if (x22160) {
x22164 = x22163;
} else {
x22164 = false;
}
bool x22165;
if (x22164) {
x22165 = x22163;
} else {
x22165 = false;
}
if (x22165) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22084,x22086,x22086,1,x22154,1,1);
assert(false && "");
}
bool x22171 = x22084 <= x22154;
int32_t x22172;
if (x22171) {
x22172 = x22154;
} else {
x22172 = x22084;
}
int32_t x22181 = x22172 * x22180;
int32_t x22182 = 64 * x22181;
float* x22183 = (float*)myMalloc(x22182 * sizeof(float));;
int32_t x22184;
if (x22155) {
x22184 = 0;
} else {
x22184 = x22092;
}
int32_t x22187;
if (x22156) {
x22187 = 0;
} else {
x22187 = 1;
}
for(int x22188=0; x22188 < 64; x22188++) {
int32_t x22200 = x22093 * x22188;
int32_t x22194 = x22181 * x22188;
for(int x22190=0; x22190 < x22172; x22190++) {
int32_t x22201 = x22184 * x22190;
int32_t x22202 = x22200 + x22201;
int32_t x22207 = x22187 * x22190;
int32_t x22196 = x22180 * x22190;
for(int x22192=0; x22192 < x22174; x22192++) {
int32_t x22203 = x22185 * x22192;
int32_t x22204 = x22202 + x22203;
int32_t x22198 = x22174 * x22192;
for(int x22193=0; x22193 < x22174; x22193++) {
int32_t x22205 = x22186 * x22193;
int32_t x22206 = x22204 + x22205;
float x22208 = x22095[x22206];
float x22209 = x107[x22207];
int32_t x22195 = x22193 + x22194;
int32_t x22197 = x22195 + x22196;
int32_t x22199 = x22197 + x22198;
float x22210 = x22208 + x22209;
x22183[x22199] = x22210;

}

}

}

}
bool x22220 = x21748 == 1;
bool x22221 = x22172 == 1;
bool x22222 = x22220 || x22221;
bool x22223 = x21748 == x22172;
bool x22224 = x22222 || x22223;
bool x22230;
if (x22224) {
x22230 = x22229;
} else {
x22230 = false;
}
bool x22231;
if (x22230) {
x22231 = x22229;
} else {
x22231 = false;
}
if (x22231) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21748,x21750,x21750,64,x22172,x22174,x22174);
assert(false && "");
}
bool x22237 = x21748 <= x22172;
int32_t x22238;
if (x22237) {
x22238 = x22172;
} else {
x22238 = x21748;
}
int32_t x22254;
if (x22220) {
x22254 = 0;
} else {
x22254 = x21756;
}
int32_t x22257;
if (x22221) {
x22257 = 0;
} else {
x22257 = x22180;
}
for(int x22260=0; x22260 < 64; x22260++) {
int32_t x22266 = x21757 * x22260;
int32_t x22273 = x22181 * x22260;
for(int x22262=0; x22262 < x22238; x22262++) {
int32_t x22267 = x22254 * x22262;
int32_t x22268 = x22266 + x22267;
int32_t x22274 = x22257 * x22262;
int32_t x22275 = x22273 + x22274;
for(int x22264=0; x22264 < x22240; x22264++) {
int32_t x22269 = x22255 * x22264;
int32_t x22270 = x22268 + x22269;
int32_t x22276 = x22258 * x22264;
int32_t x22277 = x22275 + x22276;
for(int x22265=0; x22265 < x22240; x22265++) {
int32_t x22271 = x22256 * x22265;
int32_t x22272 = x22270 + x22271;
float x22280 = x21759[x22272];
int32_t x22278 = x22259 * x22265;
int32_t x22279 = x22277 + x22278;
float x22281 = x22183[x22279];
float x22282 = x22280 + x22281;
x21759[x22272] = x22282;

}

}

}

}
float* x22292 = (float*)myMalloc(x21758 * sizeof(float));;
for(int x22294=0; x22294 < x21758; x22294++) {
float x22295 = x21759[x22294];
bool x22296 = x22295 < 0.0f;
if (x22296) {
x22292[x22294] = 0.0f;
} else {
float x22299 = x21759[x22294];
x22292[x22294] = x22299;
}

}
float* x22313 = (float*)myMalloc(x22312 * sizeof(float));;
int32_t x22316 = 64 * x21748;
int32_t x22317 = x22316 * x22308;
float* x22318 = (float*)myMalloc(x22317 * sizeof(float));;
int32_t x22314 = x21748 * x22308;
for(int x22319=0; x22319 < 64; x22319++) {
int32_t x22320 = x22319 * x21757;
float* x22321 = x22292+x22320;
int32_t x22322 = x22319 * x22309;
float* x22323 = x22313+x22322;
int32_t x22324 = x22319 * x22314;
float* x22325 = x22318+x22324;
for(int x22326=0; x22326 < x21748; x22326++) {
int32_t x22327 = x22326 / 1;
int32_t x22331 = x22327 * x22307;
int32_t x22332 = x22331 * x22307;
int32_t x22328 = x22326 % 1;
int32_t x22329 = x22328 / 1;
int32_t x22333 = x22329 * x22307;
int32_t x22334 = x22333 * x22307;
int32_t x22335 = x22332 + x22334;
int32_t x22330 = x22328 % 1;
int32_t x22336 = x22330 * x22307;
int32_t x22337 = x22336 * x22307;
int32_t x22338 = x22335 + x22337;
float* x22339 = x22325+x22338;
int32_t x22340 = x22327 * x21750;
int32_t x22341 = x22340 * x21750;
float* x22342 = x22321+x22341;
for(int x22344=0; x22344 < x22307; x22344++) {
int32_t x22346 = x22344 * x22307;
float* x22347 = x22339+x22346;
int32_t x22345 = x22344 + x22329;
int32_t x22348 = x22345 * x21750;
int32_t x22349 = x22348 + x22330;
float* x22350 = x22342+x22349;
memcpy(x22347, x22350, 4 * x22307);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x22308,x21748,1,x215,x21748,x22325,x22308,1,x22323,x22308);

}
int32_t x22359 = 0;
int32_t x22360 = 1;
x22360 *= 1;
x22359 += 1;
x22360 *= 1;
x22360 *= 1;
int32_t x22365 = x22359;
bool x22366 = x22365 >= 2;
if (x22366) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22371 = x22365 == 0;
if (x22371) {
int32_t x22372 = x22360;
bool x22373 = x22372 == 512;
if (x22373) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22380 = x22360;
int32_t x22381 = 512 / x22380;
bool x22382 = x22381 == 1;
bool x22385;
if (x454) {
bool x22383 = 512 == x22381;
bool x22384 = x22382 || x22383;
x22385 = x22384;
} else {
x22385 = false;
}
bool x22389;
if (x22385) {
x22389 = x22388;
} else {
x22389 = false;
}
bool x22390;
if (x22389) {
x22390 = x22388;
} else {
x22390 = false;
}
if (x22390) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x22307,x22307,1,x22381,1,1);
assert(false && "");
}
bool x22396 = 512 <= x22381;
int32_t x22397;
if (x22396) {
x22397 = x22381;
} else {
x22397 = 512;
}
int32_t x22406 = x22397 * x22405;
int32_t x22407 = 64 * x22406;
float* x22408 = (float*)myMalloc(x22407 * sizeof(float));;
int32_t x22411;
if (x22382) {
x22411 = 0;
} else {
x22411 = 1;
}
for(int x22412=0; x22412 < 64; x22412++) {
int32_t x22424 = x22309 * x22412;
int32_t x22418 = x22406 * x22412;
for(int x22414=0; x22414 < x22397; x22414++) {
int32_t x22425 = x22308 * x22414;
int32_t x22426 = x22424 + x22425;
int32_t x22431 = x22411 * x22414;
int32_t x22420 = x22405 * x22414;
for(int x22416=0; x22416 < x22399; x22416++) {
int32_t x22427 = x22409 * x22416;
int32_t x22428 = x22426 + x22427;
int32_t x22422 = x22399 * x22416;
for(int x22417=0; x22417 < x22399; x22417++) {
int32_t x22429 = x22410 * x22417;
int32_t x22430 = x22428 + x22429;
float x22432 = x22313[x22430];
float x22433 = x154[x22431];
int32_t x22419 = x22417 + x22418;
int32_t x22421 = x22419 + x22420;
int32_t x22423 = x22421 + x22422;
float x22434 = x22432 - x22433;
x22408[x22423] = x22434;

}

}

}

}
float* x22444 = (float*)myMalloc(512 * sizeof(float));;
for(int x22445=0; x22445 < 512; x22445++) {
float x22446 = x65[x22445];
float x22447 = x22446 + 1.0E-5f;
x22444[x22445] = x22447;

}
float* x22451 = (float*)myMalloc(512 * sizeof(float));;
for(int x22452=0; x22452 < 512; x22452++) {
float x22453 = x22444[x22452];
double x22454 = (double)x22453;
double x22455 = sqrt(x22454);
float x22456 = (float)x22455;
x22451[x22452] = x22456;

}
int32_t x22460 = 0;
int32_t x22461 = 1;
x22461 *= 1;
x22460 += 1;
x22461 *= 1;
x22461 *= 1;
int32_t x22466 = x22460;
bool x22467 = x22466 >= 2;
if (x22467) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22472 = x22466 == 0;
if (x22472) {
int32_t x22473 = x22461;
bool x22474 = x22473 == 512;
if (x22474) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22481 = x22461;
bool x22483 = x22397 == 1;
int32_t x22482 = 512 / x22481;
bool x22484 = x22482 == 1;
bool x22488;
if (x454) {
bool x22485 = x22483 || x22484;
bool x22486 = x22397 == x22482;
bool x22487 = x22485 || x22486;
x22488 = x22487;
} else {
x22488 = false;
}
bool x22492;
if (x22488) {
x22492 = x22491;
} else {
x22492 = false;
}
bool x22493;
if (x22492) {
x22493 = x22491;
} else {
x22493 = false;
}
if (x22493) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22397,x22399,x22399,1,x22482,1,1);
assert(false && "");
}
bool x22499 = x22397 <= x22482;
int32_t x22500;
if (x22499) {
x22500 = x22482;
} else {
x22500 = x22397;
}
int32_t x22509 = x22500 * x22508;
int32_t x22510 = 64 * x22509;
float* x22511 = (float*)myMalloc(x22510 * sizeof(float));;
int32_t x22512;
if (x22483) {
x22512 = 0;
} else {
x22512 = x22405;
}
int32_t x22515;
if (x22484) {
x22515 = 0;
} else {
x22515 = 1;
}
for(int x22516=0; x22516 < 64; x22516++) {
int32_t x22528 = x22406 * x22516;
int32_t x22522 = x22509 * x22516;
for(int x22518=0; x22518 < x22500; x22518++) {
int32_t x22529 = x22512 * x22518;
int32_t x22530 = x22528 + x22529;
int32_t x22535 = x22515 * x22518;
int32_t x22524 = x22508 * x22518;
for(int x22520=0; x22520 < x22502; x22520++) {
int32_t x22531 = x22513 * x22520;
int32_t x22532 = x22530 + x22531;
int32_t x22526 = x22502 * x22520;
for(int x22521=0; x22521 < x22502; x22521++) {
int32_t x22533 = x22514 * x22521;
int32_t x22534 = x22532 + x22533;
float x22536 = x22408[x22534];
float x22537 = x22451[x22535];
int32_t x22523 = x22521 + x22522;
int32_t x22525 = x22523 + x22524;
int32_t x22527 = x22525 + x22526;
float x22538 = x22536 / x22537;
x22511[x22527] = x22538;

}

}

}

}
int32_t x22548 = 0;
int32_t x22549 = 1;
x22549 *= 1;
x22548 += 1;
x22549 *= 1;
x22549 *= 1;
int32_t x22554 = x22548;
bool x22555 = x22554 >= 2;
if (x22555) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22560 = x22554 == 0;
if (x22560) {
int32_t x22561 = x22549;
bool x22562 = x22561 == 512;
if (x22562) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22569 = x22549;
bool x22571 = x22500 == 1;
int32_t x22570 = 512 / x22569;
bool x22572 = x22570 == 1;
bool x22576;
if (x454) {
bool x22573 = x22571 || x22572;
bool x22574 = x22500 == x22570;
bool x22575 = x22573 || x22574;
x22576 = x22575;
} else {
x22576 = false;
}
bool x22580;
if (x22576) {
x22580 = x22579;
} else {
x22580 = false;
}
bool x22581;
if (x22580) {
x22581 = x22579;
} else {
x22581 = false;
}
if (x22581) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22500,x22502,x22502,1,x22570,1,1);
assert(false && "");
}
bool x22587 = x22500 <= x22570;
int32_t x22588;
if (x22587) {
x22588 = x22570;
} else {
x22588 = x22500;
}
int32_t x22597 = x22588 * x22596;
int32_t x22598 = 64 * x22597;
float* x22599 = (float*)myMalloc(x22598 * sizeof(float));;
int32_t x22600;
if (x22571) {
x22600 = 0;
} else {
x22600 = x22508;
}
int32_t x22603;
if (x22572) {
x22603 = 0;
} else {
x22603 = 1;
}
for(int x22604=0; x22604 < 64; x22604++) {
int32_t x22616 = x22509 * x22604;
int32_t x22610 = x22597 * x22604;
for(int x22606=0; x22606 < x22588; x22606++) {
int32_t x22617 = x22600 * x22606;
int32_t x22618 = x22616 + x22617;
int32_t x22623 = x22603 * x22606;
int32_t x22612 = x22596 * x22606;
for(int x22608=0; x22608 < x22590; x22608++) {
int32_t x22619 = x22601 * x22608;
int32_t x22620 = x22618 + x22619;
int32_t x22614 = x22590 * x22608;
for(int x22609=0; x22609 < x22590; x22609++) {
int32_t x22621 = x22602 * x22609;
int32_t x22622 = x22620 + x22621;
float x22624 = x22511[x22622];
float x22625 = x46[x22623];
int32_t x22611 = x22609 + x22610;
int32_t x22613 = x22611 + x22612;
int32_t x22615 = x22613 + x22614;
float x22626 = x22624 * x22625;
x22599[x22615] = x22626;

}

}

}

}
int32_t x22636 = 0;
int32_t x22637 = 1;
x22637 *= 1;
x22636 += 1;
x22637 *= 1;
x22637 *= 1;
int32_t x22642 = x22636;
bool x22643 = x22642 >= 2;
if (x22643) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22648 = x22642 == 0;
if (x22648) {
int32_t x22649 = x22637;
bool x22650 = x22649 == 512;
if (x22650) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22657 = x22637;
bool x22659 = x22588 == 1;
int32_t x22658 = 512 / x22657;
bool x22660 = x22658 == 1;
bool x22664;
if (x454) {
bool x22661 = x22659 || x22660;
bool x22662 = x22588 == x22658;
bool x22663 = x22661 || x22662;
x22664 = x22663;
} else {
x22664 = false;
}
bool x22668;
if (x22664) {
x22668 = x22667;
} else {
x22668 = false;
}
bool x22669;
if (x22668) {
x22669 = x22667;
} else {
x22669 = false;
}
if (x22669) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22588,x22590,x22590,1,x22658,1,1);
assert(false && "");
}
bool x22675 = x22588 <= x22658;
int32_t x22676;
if (x22675) {
x22676 = x22658;
} else {
x22676 = x22588;
}
int32_t x22685 = x22676 * x22684;
int32_t x22686 = 64 * x22685;
float* x22687 = (float*)myMalloc(x22686 * sizeof(float));;
int32_t x22688;
if (x22659) {
x22688 = 0;
} else {
x22688 = x22596;
}
int32_t x22691;
if (x22660) {
x22691 = 0;
} else {
x22691 = 1;
}
for(int x22692=0; x22692 < 64; x22692++) {
int32_t x22704 = x22597 * x22692;
int32_t x22698 = x22685 * x22692;
for(int x22694=0; x22694 < x22676; x22694++) {
int32_t x22705 = x22688 * x22694;
int32_t x22706 = x22704 + x22705;
int32_t x22711 = x22691 * x22694;
int32_t x22700 = x22684 * x22694;
for(int x22696=0; x22696 < x22678; x22696++) {
int32_t x22707 = x22689 * x22696;
int32_t x22708 = x22706 + x22707;
int32_t x22702 = x22678 * x22696;
for(int x22697=0; x22697 < x22678; x22697++) {
int32_t x22709 = x22690 * x22697;
int32_t x22710 = x22708 + x22709;
float x22712 = x22599[x22710];
float x22713 = x137[x22711];
int32_t x22699 = x22697 + x22698;
int32_t x22701 = x22699 + x22700;
int32_t x22703 = x22701 + x22702;
float x22714 = x22712 + x22713;
x22687[x22703] = x22714;

}

}

}

}
float* x22724 = (float*)myMalloc(x22686 * sizeof(float));;
for(int x22726=0; x22726 < x22686; x22726++) {
float x22727 = x22687[x22726];
bool x22728 = x22727 < 0.0f;
if (x22728) {
x22724[x22726] = 0.0f;
} else {
float x22731 = x22687[x22726];
x22724[x22726] = x22731;
}

}
float* x22746 = (float*)myMalloc(x22745 * sizeof(float));;
int32_t x22747 = 9 * x22676;
int32_t x22750 = 64 * x22747;
int32_t x22751 = x22750 * x22741;
float* x22752 = (float*)myMalloc(x22751 * sizeof(float));;
int32_t x22748 = x22747 * x22741;
int32_t x22760 = x22676 * 3;
int32_t x22761 = x22760 * 3;
for(int x22753=0; x22753 < 64; x22753++) {
int32_t x22754 = x22753 * x22685;
float* x22755 = x22724+x22754;
int32_t x22756 = x22753 * x22742;
float* x22757 = x22746+x22756;
int32_t x22758 = x22753 * x22748;
float* x22759 = x22752+x22758;
for(int x22763=0; x22763 < x22761; x22763++) {
int32_t x22764 = x22763 / 9;
int32_t x22768 = x22764 * 3;
int32_t x22769 = x22768 * 3;
int32_t x22770 = x22769 * x22740;
int32_t x22771 = x22770 * x22740;
int32_t x22765 = x22763 % 9;
int32_t x22766 = x22765 / 3;
int32_t x22772 = x22766 * 3;
int32_t x22773 = x22772 * x22740;
int32_t x22774 = x22773 * x22740;
int32_t x22775 = x22771 + x22774;
int32_t x22767 = x22765 % 3;
int32_t x22776 = x22767 * x22740;
int32_t x22777 = x22776 * x22740;
int32_t x22778 = x22775 + x22777;
float* x22779 = x22759+x22778;
int32_t x22780 = x22764 * x22678;
int32_t x22781 = x22780 * x22678;
float* x22782 = x22755+x22781;
int32_t x22795 = 1 - x22767;
bool x22796 = x22795 > 0;
int32_t x22797;
if (x22796) {
x22797 = x22795;
} else {
x22797 = 0;
}
int32_t x22798 = 3 - x22767;
int32_t x22799 = x22798 - 1;
int32_t x22800 = 1 - x22799;
bool x22801 = x22800 > 0;
int32_t x22802;
if (x22801) {
x22802 = x22800;
} else {
x22802 = 0;
}
int32_t x22803 = x22740 - x22802;
int32_t x22804 = x22803 - x22797;
bool x22805 = x22804 <= 0;
bool x22809 = x22797 > 0;
int32_t x22794 = -1 + x22767;
bool x22822 = x22802 > 0;
for(int x22784=0; x22784 < x22740; x22784++) {
int32_t x22785 = x22784 - 1;
int32_t x22786 = x22785 + x22766;
bool x22787 = x22786 < 0;
bool x22788 = x22786 >= x22678;
bool x22789 = x22787 || x22788;
if (x22789) {
int32_t x22790 = x22784 * x22740;
float* x22791 = x22779+x22790;
memset(x22791, 0, 4 * x22740);;
} else {
if (x22805) {
int32_t x22790 = x22784 * x22740;
float* x22806 = x22779+x22790;
memset(x22806, 0, 4 * x22740);;
} else {
int32_t x22790 = x22784 * x22740;
if (x22809) {
float* x22810 = x22779+x22790;
memset(x22810, 0, 4 * x22797);;
} else {
}
// may have segfault here
int32_t x22815 = x22790 + x22797;
float* x22816 = x22779+x22815;
int32_t x22817 = x22786 * x22678;
int32_t x22818 = x22817 + x22794;
int32_t x22819 = x22818 + x22797;
float* x22820 = x22782+x22819;
memcpy(x22816, x22820, 4 * x22804);;
if (x22822) {
int32_t x22823 = x22790 + x22740;
int32_t x22824 = x22823 - x22802;
float* x22825 = x22779+x22824;
memset(x22825, 0, 4 * x22802);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x22741,x22747,1,x155,x22747,x22759,x22741,1,x22757,x22741);

}
int32_t x22840 = 0;
int32_t x22841 = 1;
x22841 *= 1;
x22840 += 1;
x22841 *= 1;
x22841 *= 1;
int32_t x22846 = x22840;
bool x22847 = x22846 >= 2;
if (x22847) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22852 = x22846 == 0;
if (x22852) {
int32_t x22853 = x22841;
bool x22854 = x22853 == 512;
if (x22854) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22861 = x22841;
int32_t x22862 = 512 / x22861;
bool x22863 = x22862 == 1;
bool x22866;
if (x454) {
bool x22864 = 512 == x22862;
bool x22865 = x22863 || x22864;
x22866 = x22865;
} else {
x22866 = false;
}
bool x22870;
if (x22866) {
x22870 = x22869;
} else {
x22870 = false;
}
bool x22871;
if (x22870) {
x22871 = x22869;
} else {
x22871 = false;
}
if (x22871) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x22740,x22740,1,x22862,1,1);
assert(false && "");
}
bool x22877 = 512 <= x22862;
int32_t x22878;
if (x22877) {
x22878 = x22862;
} else {
x22878 = 512;
}
int32_t x22887 = x22878 * x22886;
int32_t x22888 = 64 * x22887;
float* x22889 = (float*)myMalloc(x22888 * sizeof(float));;
int32_t x22892;
if (x22863) {
x22892 = 0;
} else {
x22892 = 1;
}
for(int x22893=0; x22893 < 64; x22893++) {
int32_t x22905 = x22742 * x22893;
int32_t x22899 = x22887 * x22893;
for(int x22895=0; x22895 < x22878; x22895++) {
int32_t x22906 = x22741 * x22895;
int32_t x22907 = x22905 + x22906;
int32_t x22912 = x22892 * x22895;
int32_t x22901 = x22886 * x22895;
for(int x22897=0; x22897 < x22880; x22897++) {
int32_t x22908 = x22890 * x22897;
int32_t x22909 = x22907 + x22908;
int32_t x22903 = x22880 * x22897;
for(int x22898=0; x22898 < x22880; x22898++) {
int32_t x22910 = x22891 * x22898;
int32_t x22911 = x22909 + x22910;
float x22913 = x22746[x22911];
float x22914 = x138[x22912];
int32_t x22900 = x22898 + x22899;
int32_t x22902 = x22900 + x22901;
int32_t x22904 = x22902 + x22903;
float x22915 = x22913 - x22914;
x22889[x22904] = x22915;

}

}

}

}
float* x22925 = (float*)myMalloc(512 * sizeof(float));;
for(int x22926=0; x22926 < 512; x22926++) {
float x22927 = x195[x22926];
float x22928 = x22927 + 1.0E-5f;
x22925[x22926] = x22928;

}
float* x22932 = (float*)myMalloc(512 * sizeof(float));;
for(int x22933=0; x22933 < 512; x22933++) {
float x22934 = x22925[x22933];
double x22935 = (double)x22934;
double x22936 = sqrt(x22935);
float x22937 = (float)x22936;
x22932[x22933] = x22937;

}
int32_t x22941 = 0;
int32_t x22942 = 1;
x22942 *= 1;
x22941 += 1;
x22942 *= 1;
x22942 *= 1;
int32_t x22947 = x22941;
bool x22948 = x22947 >= 2;
if (x22948) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22953 = x22947 == 0;
if (x22953) {
int32_t x22954 = x22942;
bool x22955 = x22954 == 512;
if (x22955) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22962 = x22942;
bool x22964 = x22878 == 1;
int32_t x22963 = 512 / x22962;
bool x22965 = x22963 == 1;
bool x22969;
if (x454) {
bool x22966 = x22964 || x22965;
bool x22967 = x22878 == x22963;
bool x22968 = x22966 || x22967;
x22969 = x22968;
} else {
x22969 = false;
}
bool x22973;
if (x22969) {
x22973 = x22972;
} else {
x22973 = false;
}
bool x22974;
if (x22973) {
x22974 = x22972;
} else {
x22974 = false;
}
if (x22974) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22878,x22880,x22880,1,x22963,1,1);
assert(false && "");
}
bool x22980 = x22878 <= x22963;
int32_t x22981;
if (x22980) {
x22981 = x22963;
} else {
x22981 = x22878;
}
int32_t x22990 = x22981 * x22989;
int32_t x22991 = 64 * x22990;
float* x22992 = (float*)myMalloc(x22991 * sizeof(float));;
int32_t x22993;
if (x22964) {
x22993 = 0;
} else {
x22993 = x22886;
}
int32_t x22996;
if (x22965) {
x22996 = 0;
} else {
x22996 = 1;
}
for(int x22997=0; x22997 < 64; x22997++) {
int32_t x23009 = x22887 * x22997;
int32_t x23003 = x22990 * x22997;
for(int x22999=0; x22999 < x22981; x22999++) {
int32_t x23010 = x22993 * x22999;
int32_t x23011 = x23009 + x23010;
int32_t x23016 = x22996 * x22999;
int32_t x23005 = x22989 * x22999;
for(int x23001=0; x23001 < x22983; x23001++) {
int32_t x23012 = x22994 * x23001;
int32_t x23013 = x23011 + x23012;
int32_t x23007 = x22983 * x23001;
for(int x23002=0; x23002 < x22983; x23002++) {
int32_t x23014 = x22995 * x23002;
int32_t x23015 = x23013 + x23014;
float x23017 = x22889[x23015];
float x23018 = x22932[x23016];
int32_t x23004 = x23002 + x23003;
int32_t x23006 = x23004 + x23005;
int32_t x23008 = x23006 + x23007;
float x23019 = x23017 / x23018;
x22992[x23008] = x23019;

}

}

}

}
int32_t x23029 = 0;
int32_t x23030 = 1;
x23030 *= 1;
x23029 += 1;
x23030 *= 1;
x23030 *= 1;
int32_t x23035 = x23029;
bool x23036 = x23035 >= 2;
if (x23036) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23041 = x23035 == 0;
if (x23041) {
int32_t x23042 = x23030;
bool x23043 = x23042 == 512;
if (x23043) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23050 = x23030;
bool x23052 = x22981 == 1;
int32_t x23051 = 512 / x23050;
bool x23053 = x23051 == 1;
bool x23057;
if (x454) {
bool x23054 = x23052 || x23053;
bool x23055 = x22981 == x23051;
bool x23056 = x23054 || x23055;
x23057 = x23056;
} else {
x23057 = false;
}
bool x23061;
if (x23057) {
x23061 = x23060;
} else {
x23061 = false;
}
bool x23062;
if (x23061) {
x23062 = x23060;
} else {
x23062 = false;
}
if (x23062) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22981,x22983,x22983,1,x23051,1,1);
assert(false && "");
}
bool x23068 = x22981 <= x23051;
int32_t x23069;
if (x23068) {
x23069 = x23051;
} else {
x23069 = x22981;
}
int32_t x23078 = x23069 * x23077;
int32_t x23079 = 64 * x23078;
float* x23080 = (float*)myMalloc(x23079 * sizeof(float));;
int32_t x23081;
if (x23052) {
x23081 = 0;
} else {
x23081 = x22989;
}
int32_t x23084;
if (x23053) {
x23084 = 0;
} else {
x23084 = 1;
}
for(int x23085=0; x23085 < 64; x23085++) {
int32_t x23097 = x22990 * x23085;
int32_t x23091 = x23078 * x23085;
for(int x23087=0; x23087 < x23069; x23087++) {
int32_t x23098 = x23081 * x23087;
int32_t x23099 = x23097 + x23098;
int32_t x23104 = x23084 * x23087;
int32_t x23093 = x23077 * x23087;
for(int x23089=0; x23089 < x23071; x23089++) {
int32_t x23100 = x23082 * x23089;
int32_t x23101 = x23099 + x23100;
int32_t x23095 = x23071 * x23089;
for(int x23090=0; x23090 < x23071; x23090++) {
int32_t x23102 = x23083 * x23090;
int32_t x23103 = x23101 + x23102;
float x23105 = x22992[x23103];
float x23106 = x160[x23104];
int32_t x23092 = x23090 + x23091;
int32_t x23094 = x23092 + x23093;
int32_t x23096 = x23094 + x23095;
float x23107 = x23105 * x23106;
x23080[x23096] = x23107;

}

}

}

}
int32_t x23117 = 0;
int32_t x23118 = 1;
x23118 *= 1;
x23117 += 1;
x23118 *= 1;
x23118 *= 1;
int32_t x23123 = x23117;
bool x23124 = x23123 >= 2;
if (x23124) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23129 = x23123 == 0;
if (x23129) {
int32_t x23130 = x23118;
bool x23131 = x23130 == 512;
if (x23131) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23138 = x23118;
bool x23140 = x23069 == 1;
int32_t x23139 = 512 / x23138;
bool x23141 = x23139 == 1;
bool x23145;
if (x454) {
bool x23142 = x23140 || x23141;
bool x23143 = x23069 == x23139;
bool x23144 = x23142 || x23143;
x23145 = x23144;
} else {
x23145 = false;
}
bool x23149;
if (x23145) {
x23149 = x23148;
} else {
x23149 = false;
}
bool x23150;
if (x23149) {
x23150 = x23148;
} else {
x23150 = false;
}
if (x23150) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23069,x23071,x23071,1,x23139,1,1);
assert(false && "");
}
bool x23156 = x23069 <= x23139;
int32_t x23157;
if (x23156) {
x23157 = x23139;
} else {
x23157 = x23069;
}
int32_t x23166 = x23157 * x23165;
int32_t x23167 = 64 * x23166;
float* x23168 = (float*)myMalloc(x23167 * sizeof(float));;
int32_t x23169;
if (x23140) {
x23169 = 0;
} else {
x23169 = x23077;
}
int32_t x23172;
if (x23141) {
x23172 = 0;
} else {
x23172 = 1;
}
for(int x23173=0; x23173 < 64; x23173++) {
int32_t x23185 = x23078 * x23173;
int32_t x23179 = x23166 * x23173;
for(int x23175=0; x23175 < x23157; x23175++) {
int32_t x23186 = x23169 * x23175;
int32_t x23187 = x23185 + x23186;
int32_t x23192 = x23172 * x23175;
int32_t x23181 = x23165 * x23175;
for(int x23177=0; x23177 < x23159; x23177++) {
int32_t x23188 = x23170 * x23177;
int32_t x23189 = x23187 + x23188;
int32_t x23183 = x23159 * x23177;
for(int x23178=0; x23178 < x23159; x23178++) {
int32_t x23190 = x23171 * x23178;
int32_t x23191 = x23189 + x23190;
float x23193 = x23080[x23191];
float x23194 = x66[x23192];
int32_t x23180 = x23178 + x23179;
int32_t x23182 = x23180 + x23181;
int32_t x23184 = x23182 + x23183;
float x23195 = x23193 + x23194;
x23168[x23184] = x23195;

}

}

}

}
float* x23205 = (float*)myMalloc(x23167 * sizeof(float));;
for(int x23207=0; x23207 < x23167; x23207++) {
float x23208 = x23168[x23207];
bool x23209 = x23208 < 0.0f;
if (x23209) {
x23205[x23207] = 0.0f;
} else {
float x23212 = x23168[x23207];
x23205[x23207] = x23212;
}

}
float* x23226 = (float*)myMalloc(x23225 * sizeof(float));;
int32_t x23229 = 64 * x23157;
int32_t x23230 = x23229 * x23221;
float* x23231 = (float*)myMalloc(x23230 * sizeof(float));;
int32_t x23227 = x23157 * x23221;
for(int x23232=0; x23232 < 64; x23232++) {
int32_t x23233 = x23232 * x23166;
float* x23234 = x23205+x23233;
int32_t x23235 = x23232 * x23222;
float* x23236 = x23226+x23235;
int32_t x23237 = x23232 * x23227;
float* x23238 = x23231+x23237;
for(int x23239=0; x23239 < x23157; x23239++) {
int32_t x23240 = x23239 / 1;
int32_t x23244 = x23240 * x23220;
int32_t x23245 = x23244 * x23220;
int32_t x23241 = x23239 % 1;
int32_t x23242 = x23241 / 1;
int32_t x23246 = x23242 * x23220;
int32_t x23247 = x23246 * x23220;
int32_t x23248 = x23245 + x23247;
int32_t x23243 = x23241 % 1;
int32_t x23249 = x23243 * x23220;
int32_t x23250 = x23249 * x23220;
int32_t x23251 = x23248 + x23250;
float* x23252 = x23238+x23251;
int32_t x23253 = x23240 * x23159;
int32_t x23254 = x23253 * x23159;
float* x23255 = x23234+x23254;
for(int x23257=0; x23257 < x23220; x23257++) {
int32_t x23259 = x23257 * x23220;
float* x23260 = x23252+x23259;
int32_t x23258 = x23257 + x23242;
int32_t x23261 = x23258 * x23159;
int32_t x23262 = x23261 + x23243;
float* x23263 = x23255+x23262;
memcpy(x23260, x23263, 4 * x23220);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x23221,x23157,1,x47,x23157,x23238,x23221,1,x23236,x23221);

}
int32_t x23272 = 0;
int32_t x23273 = 1;
x23273 *= 1;
x23272 += 1;
x23273 *= 1;
x23273 *= 1;
int32_t x23278 = x23272;
bool x23279 = x23278 >= 2;
if (x23279) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23284 = x23278 == 0;
if (x23284) {
int32_t x23285 = x23273;
bool x23286 = x23285 == 2048;
if (x23286) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23293 = x23273;
int32_t x23294 = 2048 / x23293;
bool x23295 = x23294 == 1;
bool x23298;
if (x454) {
bool x23296 = 2048 == x23294;
bool x23297 = x23295 || x23296;
x23298 = x23297;
} else {
x23298 = false;
}
bool x23302;
if (x23298) {
x23302 = x23301;
} else {
x23302 = false;
}
bool x23303;
if (x23302) {
x23303 = x23301;
} else {
x23303 = false;
}
if (x23303) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x23220,x23220,1,x23294,1,1);
assert(false && "");
}
bool x23309 = 2048 <= x23294;
int32_t x23310;
if (x23309) {
x23310 = x23294;
} else {
x23310 = 2048;
}
int32_t x23319 = x23310 * x23318;
int32_t x23320 = 64 * x23319;
float* x23321 = (float*)myMalloc(x23320 * sizeof(float));;
int32_t x23324;
if (x23295) {
x23324 = 0;
} else {
x23324 = 1;
}
for(int x23325=0; x23325 < 64; x23325++) {
int32_t x23337 = x23222 * x23325;
int32_t x23331 = x23319 * x23325;
for(int x23327=0; x23327 < x23310; x23327++) {
int32_t x23338 = x23221 * x23327;
int32_t x23339 = x23337 + x23338;
int32_t x23344 = x23324 * x23327;
int32_t x23333 = x23318 * x23327;
for(int x23329=0; x23329 < x23312; x23329++) {
int32_t x23340 = x23322 * x23329;
int32_t x23341 = x23339 + x23340;
int32_t x23335 = x23312 * x23329;
for(int x23330=0; x23330 < x23312; x23330++) {
int32_t x23342 = x23323 * x23330;
int32_t x23343 = x23341 + x23342;
float x23345 = x23226[x23343];
float x23346 = x68[x23344];
int32_t x23332 = x23330 + x23331;
int32_t x23334 = x23332 + x23333;
int32_t x23336 = x23334 + x23335;
float x23347 = x23345 - x23346;
x23321[x23336] = x23347;

}

}

}

}
float* x23357 = (float*)myMalloc(2048 * sizeof(float));;
for(int x23358=0; x23358 < 2048; x23358++) {
float x23359 = x245[x23358];
float x23360 = x23359 + 1.0E-5f;
x23357[x23358] = x23360;

}
float* x23364 = (float*)myMalloc(2048 * sizeof(float));;
for(int x23365=0; x23365 < 2048; x23365++) {
float x23366 = x23357[x23365];
double x23367 = (double)x23366;
double x23368 = sqrt(x23367);
float x23369 = (float)x23368;
x23364[x23365] = x23369;

}
int32_t x23373 = 0;
int32_t x23374 = 1;
x23374 *= 1;
x23373 += 1;
x23374 *= 1;
x23374 *= 1;
int32_t x23379 = x23373;
bool x23380 = x23379 >= 2;
if (x23380) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23385 = x23379 == 0;
if (x23385) {
int32_t x23386 = x23374;
bool x23387 = x23386 == 2048;
if (x23387) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23394 = x23374;
bool x23396 = x23310 == 1;
int32_t x23395 = 2048 / x23394;
bool x23397 = x23395 == 1;
bool x23401;
if (x454) {
bool x23398 = x23396 || x23397;
bool x23399 = x23310 == x23395;
bool x23400 = x23398 || x23399;
x23401 = x23400;
} else {
x23401 = false;
}
bool x23405;
if (x23401) {
x23405 = x23404;
} else {
x23405 = false;
}
bool x23406;
if (x23405) {
x23406 = x23404;
} else {
x23406 = false;
}
if (x23406) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23310,x23312,x23312,1,x23395,1,1);
assert(false && "");
}
bool x23412 = x23310 <= x23395;
int32_t x23413;
if (x23412) {
x23413 = x23395;
} else {
x23413 = x23310;
}
int32_t x23422 = x23413 * x23421;
int32_t x23423 = 64 * x23422;
float* x23424 = (float*)myMalloc(x23423 * sizeof(float));;
int32_t x23425;
if (x23396) {
x23425 = 0;
} else {
x23425 = x23318;
}
int32_t x23428;
if (x23397) {
x23428 = 0;
} else {
x23428 = 1;
}
for(int x23429=0; x23429 < 64; x23429++) {
int32_t x23441 = x23319 * x23429;
int32_t x23435 = x23422 * x23429;
for(int x23431=0; x23431 < x23413; x23431++) {
int32_t x23442 = x23425 * x23431;
int32_t x23443 = x23441 + x23442;
int32_t x23448 = x23428 * x23431;
int32_t x23437 = x23421 * x23431;
for(int x23433=0; x23433 < x23415; x23433++) {
int32_t x23444 = x23426 * x23433;
int32_t x23445 = x23443 + x23444;
int32_t x23439 = x23415 * x23433;
for(int x23434=0; x23434 < x23415; x23434++) {
int32_t x23446 = x23427 * x23434;
int32_t x23447 = x23445 + x23446;
float x23449 = x23321[x23447];
float x23450 = x23364[x23448];
int32_t x23436 = x23434 + x23435;
int32_t x23438 = x23436 + x23437;
int32_t x23440 = x23438 + x23439;
float x23451 = x23449 / x23450;
x23424[x23440] = x23451;

}

}

}

}
int32_t x23461 = 0;
int32_t x23462 = 1;
x23462 *= 1;
x23461 += 1;
x23462 *= 1;
x23462 *= 1;
int32_t x23467 = x23461;
bool x23468 = x23467 >= 2;
if (x23468) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23473 = x23467 == 0;
if (x23473) {
int32_t x23474 = x23462;
bool x23475 = x23474 == 2048;
if (x23475) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23482 = x23462;
bool x23484 = x23413 == 1;
int32_t x23483 = 2048 / x23482;
bool x23485 = x23483 == 1;
bool x23489;
if (x454) {
bool x23486 = x23484 || x23485;
bool x23487 = x23413 == x23483;
bool x23488 = x23486 || x23487;
x23489 = x23488;
} else {
x23489 = false;
}
bool x23493;
if (x23489) {
x23493 = x23492;
} else {
x23493 = false;
}
bool x23494;
if (x23493) {
x23494 = x23492;
} else {
x23494 = false;
}
if (x23494) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23413,x23415,x23415,1,x23483,1,1);
assert(false && "");
}
bool x23500 = x23413 <= x23483;
int32_t x23501;
if (x23500) {
x23501 = x23483;
} else {
x23501 = x23413;
}
int32_t x23510 = x23501 * x23509;
int32_t x23511 = 64 * x23510;
float* x23512 = (float*)myMalloc(x23511 * sizeof(float));;
int32_t x23513;
if (x23484) {
x23513 = 0;
} else {
x23513 = x23421;
}
int32_t x23516;
if (x23485) {
x23516 = 0;
} else {
x23516 = 1;
}
for(int x23517=0; x23517 < 64; x23517++) {
int32_t x23529 = x23422 * x23517;
int32_t x23523 = x23510 * x23517;
for(int x23519=0; x23519 < x23501; x23519++) {
int32_t x23530 = x23513 * x23519;
int32_t x23531 = x23529 + x23530;
int32_t x23536 = x23516 * x23519;
int32_t x23525 = x23509 * x23519;
for(int x23521=0; x23521 < x23503; x23521++) {
int32_t x23532 = x23514 * x23521;
int32_t x23533 = x23531 + x23532;
int32_t x23527 = x23503 * x23521;
for(int x23522=0; x23522 < x23503; x23522++) {
int32_t x23534 = x23515 * x23522;
int32_t x23535 = x23533 + x23534;
float x23537 = x23424[x23535];
float x23538 = x94[x23536];
int32_t x23524 = x23522 + x23523;
int32_t x23526 = x23524 + x23525;
int32_t x23528 = x23526 + x23527;
float x23539 = x23537 * x23538;
x23512[x23528] = x23539;

}

}

}

}
int32_t x23549 = 0;
int32_t x23550 = 1;
x23550 *= 1;
x23549 += 1;
x23550 *= 1;
x23550 *= 1;
int32_t x23555 = x23549;
bool x23556 = x23555 >= 2;
if (x23556) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23561 = x23555 == 0;
if (x23561) {
int32_t x23562 = x23550;
bool x23563 = x23562 == 2048;
if (x23563) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23570 = x23550;
bool x23572 = x23501 == 1;
int32_t x23571 = 2048 / x23570;
bool x23573 = x23571 == 1;
bool x23577;
if (x454) {
bool x23574 = x23572 || x23573;
bool x23575 = x23501 == x23571;
bool x23576 = x23574 || x23575;
x23577 = x23576;
} else {
x23577 = false;
}
bool x23581;
if (x23577) {
x23581 = x23580;
} else {
x23581 = false;
}
bool x23582;
if (x23581) {
x23582 = x23580;
} else {
x23582 = false;
}
if (x23582) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23501,x23503,x23503,1,x23571,1,1);
assert(false && "");
}
bool x23588 = x23501 <= x23571;
int32_t x23589;
if (x23588) {
x23589 = x23571;
} else {
x23589 = x23501;
}
int32_t x23598 = x23589 * x23597;
int32_t x23599 = 64 * x23598;
float* x23600 = (float*)myMalloc(x23599 * sizeof(float));;
int32_t x23601;
if (x23572) {
x23601 = 0;
} else {
x23601 = x23509;
}
int32_t x23604;
if (x23573) {
x23604 = 0;
} else {
x23604 = 1;
}
for(int x23605=0; x23605 < 64; x23605++) {
int32_t x23617 = x23510 * x23605;
int32_t x23611 = x23598 * x23605;
for(int x23607=0; x23607 < x23589; x23607++) {
int32_t x23618 = x23601 * x23607;
int32_t x23619 = x23617 + x23618;
int32_t x23624 = x23604 * x23607;
int32_t x23613 = x23597 * x23607;
for(int x23609=0; x23609 < x23591; x23609++) {
int32_t x23620 = x23602 * x23609;
int32_t x23621 = x23619 + x23620;
int32_t x23615 = x23591 * x23609;
for(int x23610=0; x23610 < x23591; x23610++) {
int32_t x23622 = x23603 * x23610;
int32_t x23623 = x23621 + x23622;
float x23625 = x23512[x23623];
float x23626 = x144[x23624];
int32_t x23612 = x23610 + x23611;
int32_t x23614 = x23612 + x23613;
int32_t x23616 = x23614 + x23615;
float x23627 = x23625 + x23626;
x23600[x23616] = x23627;

}

}

}

}
bool x23637 = x23589 == 1;
bool x23638 = x23637 || x22220;
bool x23639 = x23589 == x21748;
bool x23640 = x23638 || x23639;
bool x23645;
if (x23640) {
x23645 = x23644;
} else {
x23645 = false;
}
bool x23646;
if (x23645) {
x23646 = x23644;
} else {
x23646 = false;
}
if (x23646) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23589,x23591,x23591,64,x21748,x21750,x21750);
assert(false && "");
}
bool x23652 = x23589 <= x21748;
int32_t x23653;
if (x23652) {
x23653 = x21748;
} else {
x23653 = x23589;
}
int32_t x23669;
if (x23637) {
x23669 = 0;
} else {
x23669 = x23597;
}
for(int x23672=0; x23672 < 64; x23672++) {
int32_t x23678 = x23598 * x23672;
int32_t x23685 = x21757 * x23672;
for(int x23674=0; x23674 < x23653; x23674++) {
int32_t x23679 = x23669 * x23674;
int32_t x23680 = x23678 + x23679;
int32_t x23686 = x22254 * x23674;
int32_t x23687 = x23685 + x23686;
for(int x23676=0; x23676 < x23655; x23676++) {
int32_t x23681 = x23670 * x23676;
int32_t x23682 = x23680 + x23681;
int32_t x23688 = x22255 * x23676;
int32_t x23689 = x23687 + x23688;
for(int x23677=0; x23677 < x23655; x23677++) {
int32_t x23683 = x23671 * x23677;
int32_t x23684 = x23682 + x23683;
float x23692 = x23600[x23684];
int32_t x23690 = x22256 * x23677;
int32_t x23691 = x23689 + x23690;
float x23693 = x22292[x23691];
float x23694 = x23692 + x23693;
x23600[x23684] = x23694;

}

}

}

}
float* x23704 = (float*)myMalloc(x23599 * sizeof(float));;
for(int x23706=0; x23706 < x23599; x23706++) {
float x23707 = x23600[x23706];
bool x23708 = x23707 < 0.0f;
if (x23708) {
x23704[x23706] = 0.0f;
} else {
float x23711 = x23600[x23706];
x23704[x23706] = x23711;
}

}
float* x23725 = (float*)myMalloc(x23724 * sizeof(float));;
int32_t x23728 = 64 * x23589;
int32_t x23729 = x23728 * x23720;
float* x23730 = (float*)myMalloc(x23729 * sizeof(float));;
int32_t x23726 = x23589 * x23720;
for(int x23731=0; x23731 < 64; x23731++) {
int32_t x23732 = x23731 * x23598;
float* x23733 = x23704+x23732;
int32_t x23734 = x23731 * x23721;
float* x23735 = x23725+x23734;
int32_t x23736 = x23731 * x23726;
float* x23737 = x23730+x23736;
for(int x23738=0; x23738 < x23589; x23738++) {
int32_t x23739 = x23738 / 1;
int32_t x23743 = x23739 * x23719;
int32_t x23744 = x23743 * x23719;
int32_t x23740 = x23738 % 1;
int32_t x23741 = x23740 / 1;
int32_t x23745 = x23741 * x23719;
int32_t x23746 = x23745 * x23719;
int32_t x23747 = x23744 + x23746;
int32_t x23742 = x23740 % 1;
int32_t x23748 = x23742 * x23719;
int32_t x23749 = x23748 * x23719;
int32_t x23750 = x23747 + x23749;
float* x23751 = x23737+x23750;
int32_t x23752 = x23739 * x23591;
int32_t x23753 = x23752 * x23591;
float* x23754 = x23733+x23753;
for(int x23756=0; x23756 < x23719; x23756++) {
int32_t x23758 = x23756 * x23719;
float* x23759 = x23751+x23758;
int32_t x23757 = x23756 + x23741;
int32_t x23760 = x23757 * x23591;
int32_t x23761 = x23760 + x23742;
float* x23762 = x23754+x23761;
memcpy(x23759, x23762, 4 * x23719);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x23720,x23589,1,x265,x23589,x23737,x23720,1,x23735,x23720);

}
int32_t x23771 = 0;
int32_t x23772 = 1;
x23772 *= 1;
x23771 += 1;
x23772 *= 1;
x23772 *= 1;
int32_t x23777 = x23771;
bool x23778 = x23777 >= 2;
if (x23778) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23783 = x23777 == 0;
if (x23783) {
int32_t x23784 = x23772;
bool x23785 = x23784 == 512;
if (x23785) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23792 = x23772;
int32_t x23793 = 512 / x23792;
bool x23794 = x23793 == 1;
bool x23797;
if (x454) {
bool x23795 = 512 == x23793;
bool x23796 = x23794 || x23795;
x23797 = x23796;
} else {
x23797 = false;
}
bool x23801;
if (x23797) {
x23801 = x23800;
} else {
x23801 = false;
}
bool x23802;
if (x23801) {
x23802 = x23800;
} else {
x23802 = false;
}
if (x23802) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x23719,x23719,1,x23793,1,1);
assert(false && "");
}
bool x23808 = 512 <= x23793;
int32_t x23809;
if (x23808) {
x23809 = x23793;
} else {
x23809 = 512;
}
int32_t x23818 = x23809 * x23817;
int32_t x23819 = 64 * x23818;
float* x23820 = (float*)myMalloc(x23819 * sizeof(float));;
int32_t x23823;
if (x23794) {
x23823 = 0;
} else {
x23823 = 1;
}
for(int x23824=0; x23824 < 64; x23824++) {
int32_t x23836 = x23721 * x23824;
int32_t x23830 = x23818 * x23824;
for(int x23826=0; x23826 < x23809; x23826++) {
int32_t x23837 = x23720 * x23826;
int32_t x23838 = x23836 + x23837;
int32_t x23843 = x23823 * x23826;
int32_t x23832 = x23817 * x23826;
for(int x23828=0; x23828 < x23811; x23828++) {
int32_t x23839 = x23821 * x23828;
int32_t x23840 = x23838 + x23839;
int32_t x23834 = x23811 * x23828;
for(int x23829=0; x23829 < x23811; x23829++) {
int32_t x23841 = x23822 * x23829;
int32_t x23842 = x23840 + x23841;
float x23844 = x23725[x23842];
float x23845 = x213[x23843];
int32_t x23831 = x23829 + x23830;
int32_t x23833 = x23831 + x23832;
int32_t x23835 = x23833 + x23834;
float x23846 = x23844 - x23845;
x23820[x23835] = x23846;

}

}

}

}
float* x23856 = (float*)myMalloc(512 * sizeof(float));;
for(int x23857=0; x23857 < 512; x23857++) {
float x23858 = x255[x23857];
float x23859 = x23858 + 1.0E-5f;
x23856[x23857] = x23859;

}
float* x23863 = (float*)myMalloc(512 * sizeof(float));;
for(int x23864=0; x23864 < 512; x23864++) {
float x23865 = x23856[x23864];
double x23866 = (double)x23865;
double x23867 = sqrt(x23866);
float x23868 = (float)x23867;
x23863[x23864] = x23868;

}
int32_t x23872 = 0;
int32_t x23873 = 1;
x23873 *= 1;
x23872 += 1;
x23873 *= 1;
x23873 *= 1;
int32_t x23878 = x23872;
bool x23879 = x23878 >= 2;
if (x23879) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23884 = x23878 == 0;
if (x23884) {
int32_t x23885 = x23873;
bool x23886 = x23885 == 512;
if (x23886) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23893 = x23873;
bool x23895 = x23809 == 1;
int32_t x23894 = 512 / x23893;
bool x23896 = x23894 == 1;
bool x23900;
if (x454) {
bool x23897 = x23895 || x23896;
bool x23898 = x23809 == x23894;
bool x23899 = x23897 || x23898;
x23900 = x23899;
} else {
x23900 = false;
}
bool x23904;
if (x23900) {
x23904 = x23903;
} else {
x23904 = false;
}
bool x23905;
if (x23904) {
x23905 = x23903;
} else {
x23905 = false;
}
if (x23905) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23809,x23811,x23811,1,x23894,1,1);
assert(false && "");
}
bool x23911 = x23809 <= x23894;
int32_t x23912;
if (x23911) {
x23912 = x23894;
} else {
x23912 = x23809;
}
int32_t x23921 = x23912 * x23920;
int32_t x23922 = 64 * x23921;
float* x23923 = (float*)myMalloc(x23922 * sizeof(float));;
int32_t x23924;
if (x23895) {
x23924 = 0;
} else {
x23924 = x23817;
}
int32_t x23927;
if (x23896) {
x23927 = 0;
} else {
x23927 = 1;
}
for(int x23928=0; x23928 < 64; x23928++) {
int32_t x23940 = x23818 * x23928;
int32_t x23934 = x23921 * x23928;
for(int x23930=0; x23930 < x23912; x23930++) {
int32_t x23941 = x23924 * x23930;
int32_t x23942 = x23940 + x23941;
int32_t x23947 = x23927 * x23930;
int32_t x23936 = x23920 * x23930;
for(int x23932=0; x23932 < x23914; x23932++) {
int32_t x23943 = x23925 * x23932;
int32_t x23944 = x23942 + x23943;
int32_t x23938 = x23914 * x23932;
for(int x23933=0; x23933 < x23914; x23933++) {
int32_t x23945 = x23926 * x23933;
int32_t x23946 = x23944 + x23945;
float x23948 = x23820[x23946];
float x23949 = x23863[x23947];
int32_t x23935 = x23933 + x23934;
int32_t x23937 = x23935 + x23936;
int32_t x23939 = x23937 + x23938;
float x23950 = x23948 / x23949;
x23923[x23939] = x23950;

}

}

}

}
int32_t x23960 = 0;
int32_t x23961 = 1;
x23961 *= 1;
x23960 += 1;
x23961 *= 1;
x23961 *= 1;
int32_t x23966 = x23960;
bool x23967 = x23966 >= 2;
if (x23967) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23972 = x23966 == 0;
if (x23972) {
int32_t x23973 = x23961;
bool x23974 = x23973 == 512;
if (x23974) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23981 = x23961;
bool x23983 = x23912 == 1;
int32_t x23982 = 512 / x23981;
bool x23984 = x23982 == 1;
bool x23988;
if (x454) {
bool x23985 = x23983 || x23984;
bool x23986 = x23912 == x23982;
bool x23987 = x23985 || x23986;
x23988 = x23987;
} else {
x23988 = false;
}
bool x23992;
if (x23988) {
x23992 = x23991;
} else {
x23992 = false;
}
bool x23993;
if (x23992) {
x23993 = x23991;
} else {
x23993 = false;
}
if (x23993) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23912,x23914,x23914,1,x23982,1,1);
assert(false && "");
}
bool x23999 = x23912 <= x23982;
int32_t x24000;
if (x23999) {
x24000 = x23982;
} else {
x24000 = x23912;
}
int32_t x24009 = x24000 * x24008;
int32_t x24010 = 64 * x24009;
float* x24011 = (float*)myMalloc(x24010 * sizeof(float));;
int32_t x24012;
if (x23983) {
x24012 = 0;
} else {
x24012 = x23920;
}
int32_t x24015;
if (x23984) {
x24015 = 0;
} else {
x24015 = 1;
}
for(int x24016=0; x24016 < 64; x24016++) {
int32_t x24028 = x23921 * x24016;
int32_t x24022 = x24009 * x24016;
for(int x24018=0; x24018 < x24000; x24018++) {
int32_t x24029 = x24012 * x24018;
int32_t x24030 = x24028 + x24029;
int32_t x24035 = x24015 * x24018;
int32_t x24024 = x24008 * x24018;
for(int x24020=0; x24020 < x24002; x24020++) {
int32_t x24031 = x24013 * x24020;
int32_t x24032 = x24030 + x24031;
int32_t x24026 = x24002 * x24020;
for(int x24021=0; x24021 < x24002; x24021++) {
int32_t x24033 = x24014 * x24021;
int32_t x24034 = x24032 + x24033;
float x24036 = x23923[x24034];
float x24037 = x15[x24035];
int32_t x24023 = x24021 + x24022;
int32_t x24025 = x24023 + x24024;
int32_t x24027 = x24025 + x24026;
float x24038 = x24036 * x24037;
x24011[x24027] = x24038;

}

}

}

}
int32_t x24048 = 0;
int32_t x24049 = 1;
x24049 *= 1;
x24048 += 1;
x24049 *= 1;
x24049 *= 1;
int32_t x24054 = x24048;
bool x24055 = x24054 >= 2;
if (x24055) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24060 = x24054 == 0;
if (x24060) {
int32_t x24061 = x24049;
bool x24062 = x24061 == 512;
if (x24062) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24069 = x24049;
bool x24071 = x24000 == 1;
int32_t x24070 = 512 / x24069;
bool x24072 = x24070 == 1;
bool x24076;
if (x454) {
bool x24073 = x24071 || x24072;
bool x24074 = x24000 == x24070;
bool x24075 = x24073 || x24074;
x24076 = x24075;
} else {
x24076 = false;
}
bool x24080;
if (x24076) {
x24080 = x24079;
} else {
x24080 = false;
}
bool x24081;
if (x24080) {
x24081 = x24079;
} else {
x24081 = false;
}
if (x24081) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24000,x24002,x24002,1,x24070,1,1);
assert(false && "");
}
bool x24087 = x24000 <= x24070;
int32_t x24088;
if (x24087) {
x24088 = x24070;
} else {
x24088 = x24000;
}
int32_t x24097 = x24088 * x24096;
int32_t x24098 = 64 * x24097;
float* x24099 = (float*)myMalloc(x24098 * sizeof(float));;
int32_t x24100;
if (x24071) {
x24100 = 0;
} else {
x24100 = x24008;
}
int32_t x24103;
if (x24072) {
x24103 = 0;
} else {
x24103 = 1;
}
for(int x24104=0; x24104 < 64; x24104++) {
int32_t x24116 = x24009 * x24104;
int32_t x24110 = x24097 * x24104;
for(int x24106=0; x24106 < x24088; x24106++) {
int32_t x24117 = x24100 * x24106;
int32_t x24118 = x24116 + x24117;
int32_t x24123 = x24103 * x24106;
int32_t x24112 = x24096 * x24106;
for(int x24108=0; x24108 < x24090; x24108++) {
int32_t x24119 = x24101 * x24108;
int32_t x24120 = x24118 + x24119;
int32_t x24114 = x24090 * x24108;
for(int x24109=0; x24109 < x24090; x24109++) {
int32_t x24121 = x24102 * x24109;
int32_t x24122 = x24120 + x24121;
float x24124 = x24011[x24122];
float x24125 = x78[x24123];
int32_t x24111 = x24109 + x24110;
int32_t x24113 = x24111 + x24112;
int32_t x24115 = x24113 + x24114;
float x24126 = x24124 + x24125;
x24099[x24115] = x24126;

}

}

}

}
float* x24136 = (float*)myMalloc(x24098 * sizeof(float));;
for(int x24138=0; x24138 < x24098; x24138++) {
float x24139 = x24099[x24138];
bool x24140 = x24139 < 0.0f;
if (x24140) {
x24136[x24138] = 0.0f;
} else {
float x24143 = x24099[x24138];
x24136[x24138] = x24143;
}

}
float* x24158 = (float*)myMalloc(x24157 * sizeof(float));;
int32_t x24159 = 9 * x24088;
int32_t x24162 = 64 * x24159;
int32_t x24163 = x24162 * x24153;
float* x24164 = (float*)myMalloc(x24163 * sizeof(float));;
int32_t x24160 = x24159 * x24153;
int32_t x24172 = x24088 * 3;
int32_t x24173 = x24172 * 3;
for(int x24165=0; x24165 < 64; x24165++) {
int32_t x24166 = x24165 * x24097;
float* x24167 = x24136+x24166;
int32_t x24168 = x24165 * x24154;
float* x24169 = x24158+x24168;
int32_t x24170 = x24165 * x24160;
float* x24171 = x24164+x24170;
for(int x24175=0; x24175 < x24173; x24175++) {
int32_t x24176 = x24175 / 9;
int32_t x24180 = x24176 * 3;
int32_t x24181 = x24180 * 3;
int32_t x24182 = x24181 * x24152;
int32_t x24183 = x24182 * x24152;
int32_t x24177 = x24175 % 9;
int32_t x24178 = x24177 / 3;
int32_t x24184 = x24178 * 3;
int32_t x24185 = x24184 * x24152;
int32_t x24186 = x24185 * x24152;
int32_t x24187 = x24183 + x24186;
int32_t x24179 = x24177 % 3;
int32_t x24188 = x24179 * x24152;
int32_t x24189 = x24188 * x24152;
int32_t x24190 = x24187 + x24189;
float* x24191 = x24171+x24190;
int32_t x24192 = x24176 * x24090;
int32_t x24193 = x24192 * x24090;
float* x24194 = x24167+x24193;
int32_t x24207 = 1 - x24179;
bool x24208 = x24207 > 0;
int32_t x24209;
if (x24208) {
x24209 = x24207;
} else {
x24209 = 0;
}
int32_t x24210 = 3 - x24179;
int32_t x24211 = x24210 - 1;
int32_t x24212 = 1 - x24211;
bool x24213 = x24212 > 0;
int32_t x24214;
if (x24213) {
x24214 = x24212;
} else {
x24214 = 0;
}
int32_t x24215 = x24152 - x24214;
int32_t x24216 = x24215 - x24209;
bool x24217 = x24216 <= 0;
bool x24221 = x24209 > 0;
int32_t x24206 = -1 + x24179;
bool x24234 = x24214 > 0;
for(int x24196=0; x24196 < x24152; x24196++) {
int32_t x24197 = x24196 - 1;
int32_t x24198 = x24197 + x24178;
bool x24199 = x24198 < 0;
bool x24200 = x24198 >= x24090;
bool x24201 = x24199 || x24200;
if (x24201) {
int32_t x24202 = x24196 * x24152;
float* x24203 = x24191+x24202;
memset(x24203, 0, 4 * x24152);;
} else {
if (x24217) {
int32_t x24202 = x24196 * x24152;
float* x24218 = x24191+x24202;
memset(x24218, 0, 4 * x24152);;
} else {
int32_t x24202 = x24196 * x24152;
if (x24221) {
float* x24222 = x24191+x24202;
memset(x24222, 0, 4 * x24209);;
} else {
}
// may have segfault here
int32_t x24227 = x24202 + x24209;
float* x24228 = x24191+x24227;
int32_t x24229 = x24198 * x24090;
int32_t x24230 = x24229 + x24206;
int32_t x24231 = x24230 + x24209;
float* x24232 = x24194+x24231;
memcpy(x24228, x24232, 4 * x24216);;
if (x24234) {
int32_t x24235 = x24202 + x24152;
int32_t x24236 = x24235 - x24214;
float* x24237 = x24191+x24236;
memset(x24237, 0, 4 * x24214);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x24153,x24159,1,x28,x24159,x24171,x24153,1,x24169,x24153);

}
int32_t x24252 = 0;
int32_t x24253 = 1;
x24253 *= 1;
x24252 += 1;
x24253 *= 1;
x24253 *= 1;
int32_t x24258 = x24252;
bool x24259 = x24258 >= 2;
if (x24259) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24264 = x24258 == 0;
if (x24264) {
int32_t x24265 = x24253;
bool x24266 = x24265 == 512;
if (x24266) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24273 = x24253;
int32_t x24274 = 512 / x24273;
bool x24275 = x24274 == 1;
bool x24278;
if (x454) {
bool x24276 = 512 == x24274;
bool x24277 = x24275 || x24276;
x24278 = x24277;
} else {
x24278 = false;
}
bool x24282;
if (x24278) {
x24282 = x24281;
} else {
x24282 = false;
}
bool x24283;
if (x24282) {
x24283 = x24281;
} else {
x24283 = false;
}
if (x24283) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x24152,x24152,1,x24274,1,1);
assert(false && "");
}
bool x24289 = 512 <= x24274;
int32_t x24290;
if (x24289) {
x24290 = x24274;
} else {
x24290 = 512;
}
int32_t x24299 = x24290 * x24298;
int32_t x24300 = 64 * x24299;
float* x24301 = (float*)myMalloc(x24300 * sizeof(float));;
int32_t x24304;
if (x24275) {
x24304 = 0;
} else {
x24304 = 1;
}
for(int x24305=0; x24305 < 64; x24305++) {
int32_t x24317 = x24154 * x24305;
int32_t x24311 = x24299 * x24305;
for(int x24307=0; x24307 < x24290; x24307++) {
int32_t x24318 = x24153 * x24307;
int32_t x24319 = x24317 + x24318;
int32_t x24324 = x24304 * x24307;
int32_t x24313 = x24298 * x24307;
for(int x24309=0; x24309 < x24292; x24309++) {
int32_t x24320 = x24302 * x24309;
int32_t x24321 = x24319 + x24320;
int32_t x24315 = x24292 * x24309;
for(int x24310=0; x24310 < x24292; x24310++) {
int32_t x24322 = x24303 * x24310;
int32_t x24323 = x24321 + x24322;
float x24325 = x24158[x24323];
float x24326 = x12[x24324];
int32_t x24312 = x24310 + x24311;
int32_t x24314 = x24312 + x24313;
int32_t x24316 = x24314 + x24315;
float x24327 = x24325 - x24326;
x24301[x24316] = x24327;

}

}

}

}
float* x24337 = (float*)myMalloc(512 * sizeof(float));;
for(int x24338=0; x24338 < 512; x24338++) {
float x24339 = x202[x24338];
float x24340 = x24339 + 1.0E-5f;
x24337[x24338] = x24340;

}
float* x24344 = (float*)myMalloc(512 * sizeof(float));;
for(int x24345=0; x24345 < 512; x24345++) {
float x24346 = x24337[x24345];
double x24347 = (double)x24346;
double x24348 = sqrt(x24347);
float x24349 = (float)x24348;
x24344[x24345] = x24349;

}
int32_t x24353 = 0;
int32_t x24354 = 1;
x24354 *= 1;
x24353 += 1;
x24354 *= 1;
x24354 *= 1;
int32_t x24359 = x24353;
bool x24360 = x24359 >= 2;
if (x24360) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24365 = x24359 == 0;
if (x24365) {
int32_t x24366 = x24354;
bool x24367 = x24366 == 512;
if (x24367) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24374 = x24354;
bool x24376 = x24290 == 1;
int32_t x24375 = 512 / x24374;
bool x24377 = x24375 == 1;
bool x24381;
if (x454) {
bool x24378 = x24376 || x24377;
bool x24379 = x24290 == x24375;
bool x24380 = x24378 || x24379;
x24381 = x24380;
} else {
x24381 = false;
}
bool x24385;
if (x24381) {
x24385 = x24384;
} else {
x24385 = false;
}
bool x24386;
if (x24385) {
x24386 = x24384;
} else {
x24386 = false;
}
if (x24386) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24290,x24292,x24292,1,x24375,1,1);
assert(false && "");
}
bool x24392 = x24290 <= x24375;
int32_t x24393;
if (x24392) {
x24393 = x24375;
} else {
x24393 = x24290;
}
int32_t x24402 = x24393 * x24401;
int32_t x24403 = 64 * x24402;
float* x24404 = (float*)myMalloc(x24403 * sizeof(float));;
int32_t x24405;
if (x24376) {
x24405 = 0;
} else {
x24405 = x24298;
}
int32_t x24408;
if (x24377) {
x24408 = 0;
} else {
x24408 = 1;
}
for(int x24409=0; x24409 < 64; x24409++) {
int32_t x24421 = x24299 * x24409;
int32_t x24415 = x24402 * x24409;
for(int x24411=0; x24411 < x24393; x24411++) {
int32_t x24422 = x24405 * x24411;
int32_t x24423 = x24421 + x24422;
int32_t x24428 = x24408 * x24411;
int32_t x24417 = x24401 * x24411;
for(int x24413=0; x24413 < x24395; x24413++) {
int32_t x24424 = x24406 * x24413;
int32_t x24425 = x24423 + x24424;
int32_t x24419 = x24395 * x24413;
for(int x24414=0; x24414 < x24395; x24414++) {
int32_t x24426 = x24407 * x24414;
int32_t x24427 = x24425 + x24426;
float x24429 = x24301[x24427];
float x24430 = x24344[x24428];
int32_t x24416 = x24414 + x24415;
int32_t x24418 = x24416 + x24417;
int32_t x24420 = x24418 + x24419;
float x24431 = x24429 / x24430;
x24404[x24420] = x24431;

}

}

}

}
int32_t x24441 = 0;
int32_t x24442 = 1;
x24442 *= 1;
x24441 += 1;
x24442 *= 1;
x24442 *= 1;
int32_t x24447 = x24441;
bool x24448 = x24447 >= 2;
if (x24448) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24453 = x24447 == 0;
if (x24453) {
int32_t x24454 = x24442;
bool x24455 = x24454 == 512;
if (x24455) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24462 = x24442;
bool x24464 = x24393 == 1;
int32_t x24463 = 512 / x24462;
bool x24465 = x24463 == 1;
bool x24469;
if (x454) {
bool x24466 = x24464 || x24465;
bool x24467 = x24393 == x24463;
bool x24468 = x24466 || x24467;
x24469 = x24468;
} else {
x24469 = false;
}
bool x24473;
if (x24469) {
x24473 = x24472;
} else {
x24473 = false;
}
bool x24474;
if (x24473) {
x24474 = x24472;
} else {
x24474 = false;
}
if (x24474) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24393,x24395,x24395,1,x24463,1,1);
assert(false && "");
}
bool x24480 = x24393 <= x24463;
int32_t x24481;
if (x24480) {
x24481 = x24463;
} else {
x24481 = x24393;
}
int32_t x24490 = x24481 * x24489;
int32_t x24491 = 64 * x24490;
float* x24492 = (float*)myMalloc(x24491 * sizeof(float));;
int32_t x24493;
if (x24464) {
x24493 = 0;
} else {
x24493 = x24401;
}
int32_t x24496;
if (x24465) {
x24496 = 0;
} else {
x24496 = 1;
}
for(int x24497=0; x24497 < 64; x24497++) {
int32_t x24509 = x24402 * x24497;
int32_t x24503 = x24490 * x24497;
for(int x24499=0; x24499 < x24481; x24499++) {
int32_t x24510 = x24493 * x24499;
int32_t x24511 = x24509 + x24510;
int32_t x24516 = x24496 * x24499;
int32_t x24505 = x24489 * x24499;
for(int x24501=0; x24501 < x24483; x24501++) {
int32_t x24512 = x24494 * x24501;
int32_t x24513 = x24511 + x24512;
int32_t x24507 = x24483 * x24501;
for(int x24502=0; x24502 < x24483; x24502++) {
int32_t x24514 = x24495 * x24502;
int32_t x24515 = x24513 + x24514;
float x24517 = x24404[x24515];
float x24518 = x194[x24516];
int32_t x24504 = x24502 + x24503;
int32_t x24506 = x24504 + x24505;
int32_t x24508 = x24506 + x24507;
float x24519 = x24517 * x24518;
x24492[x24508] = x24519;

}

}

}

}
int32_t x24529 = 0;
int32_t x24530 = 1;
x24530 *= 1;
x24529 += 1;
x24530 *= 1;
x24530 *= 1;
int32_t x24535 = x24529;
bool x24536 = x24535 >= 2;
if (x24536) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24541 = x24535 == 0;
if (x24541) {
int32_t x24542 = x24530;
bool x24543 = x24542 == 512;
if (x24543) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24550 = x24530;
bool x24552 = x24481 == 1;
int32_t x24551 = 512 / x24550;
bool x24553 = x24551 == 1;
bool x24557;
if (x454) {
bool x24554 = x24552 || x24553;
bool x24555 = x24481 == x24551;
bool x24556 = x24554 || x24555;
x24557 = x24556;
} else {
x24557 = false;
}
bool x24561;
if (x24557) {
x24561 = x24560;
} else {
x24561 = false;
}
bool x24562;
if (x24561) {
x24562 = x24560;
} else {
x24562 = false;
}
if (x24562) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24481,x24483,x24483,1,x24551,1,1);
assert(false && "");
}
bool x24568 = x24481 <= x24551;
int32_t x24569;
if (x24568) {
x24569 = x24551;
} else {
x24569 = x24481;
}
int32_t x24578 = x24569 * x24577;
int32_t x24579 = 64 * x24578;
float* x24580 = (float*)myMalloc(x24579 * sizeof(float));;
int32_t x24581;
if (x24552) {
x24581 = 0;
} else {
x24581 = x24489;
}
int32_t x24584;
if (x24553) {
x24584 = 0;
} else {
x24584 = 1;
}
for(int x24585=0; x24585 < 64; x24585++) {
int32_t x24597 = x24490 * x24585;
int32_t x24591 = x24578 * x24585;
for(int x24587=0; x24587 < x24569; x24587++) {
int32_t x24598 = x24581 * x24587;
int32_t x24599 = x24597 + x24598;
int32_t x24604 = x24584 * x24587;
int32_t x24593 = x24577 * x24587;
for(int x24589=0; x24589 < x24571; x24589++) {
int32_t x24600 = x24582 * x24589;
int32_t x24601 = x24599 + x24600;
int32_t x24595 = x24571 * x24589;
for(int x24590=0; x24590 < x24571; x24590++) {
int32_t x24602 = x24583 * x24590;
int32_t x24603 = x24601 + x24602;
float x24605 = x24492[x24603];
float x24606 = x169[x24604];
int32_t x24592 = x24590 + x24591;
int32_t x24594 = x24592 + x24593;
int32_t x24596 = x24594 + x24595;
float x24607 = x24605 + x24606;
x24580[x24596] = x24607;

}

}

}

}
float* x24617 = (float*)myMalloc(x24579 * sizeof(float));;
for(int x24619=0; x24619 < x24579; x24619++) {
float x24620 = x24580[x24619];
bool x24621 = x24620 < 0.0f;
if (x24621) {
x24617[x24619] = 0.0f;
} else {
float x24624 = x24580[x24619];
x24617[x24619] = x24624;
}

}
float* x24638 = (float*)myMalloc(x24637 * sizeof(float));;
int32_t x24641 = 64 * x24569;
int32_t x24642 = x24641 * x24633;
float* x24643 = (float*)myMalloc(x24642 * sizeof(float));;
int32_t x24639 = x24569 * x24633;
for(int x24644=0; x24644 < 64; x24644++) {
int32_t x24645 = x24644 * x24578;
float* x24646 = x24617+x24645;
int32_t x24647 = x24644 * x24634;
float* x24648 = x24638+x24647;
int32_t x24649 = x24644 * x24639;
float* x24650 = x24643+x24649;
for(int x24651=0; x24651 < x24569; x24651++) {
int32_t x24652 = x24651 / 1;
int32_t x24656 = x24652 * x24632;
int32_t x24657 = x24656 * x24632;
int32_t x24653 = x24651 % 1;
int32_t x24654 = x24653 / 1;
int32_t x24658 = x24654 * x24632;
int32_t x24659 = x24658 * x24632;
int32_t x24660 = x24657 + x24659;
int32_t x24655 = x24653 % 1;
int32_t x24661 = x24655 * x24632;
int32_t x24662 = x24661 * x24632;
int32_t x24663 = x24660 + x24662;
float* x24664 = x24650+x24663;
int32_t x24665 = x24652 * x24571;
int32_t x24666 = x24665 * x24571;
float* x24667 = x24646+x24666;
for(int x24669=0; x24669 < x24632; x24669++) {
int32_t x24671 = x24669 * x24632;
float* x24672 = x24664+x24671;
int32_t x24670 = x24669 + x24654;
int32_t x24673 = x24670 * x24571;
int32_t x24674 = x24673 + x24655;
float* x24675 = x24667+x24674;
memcpy(x24672, x24675, 4 * x24632);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x24633,x24569,1,x33,x24569,x24650,x24633,1,x24648,x24633);

}
int32_t x24684 = 0;
int32_t x24685 = 1;
x24685 *= 1;
x24684 += 1;
x24685 *= 1;
x24685 *= 1;
int32_t x24690 = x24684;
bool x24691 = x24690 >= 2;
if (x24691) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24696 = x24690 == 0;
if (x24696) {
int32_t x24697 = x24685;
bool x24698 = x24697 == 2048;
if (x24698) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24705 = x24685;
int32_t x24706 = 2048 / x24705;
bool x24707 = x24706 == 1;
bool x24710;
if (x454) {
bool x24708 = 2048 == x24706;
bool x24709 = x24707 || x24708;
x24710 = x24709;
} else {
x24710 = false;
}
bool x24714;
if (x24710) {
x24714 = x24713;
} else {
x24714 = false;
}
bool x24715;
if (x24714) {
x24715 = x24713;
} else {
x24715 = false;
}
if (x24715) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x24632,x24632,1,x24706,1,1);
assert(false && "");
}
bool x24721 = 2048 <= x24706;
int32_t x24722;
if (x24721) {
x24722 = x24706;
} else {
x24722 = 2048;
}
int32_t x24731 = x24722 * x24730;
int32_t x24732 = 64 * x24731;
float* x24733 = (float*)myMalloc(x24732 * sizeof(float));;
int32_t x24736;
if (x24707) {
x24736 = 0;
} else {
x24736 = 1;
}
for(int x24737=0; x24737 < 64; x24737++) {
int32_t x24749 = x24634 * x24737;
int32_t x24743 = x24731 * x24737;
for(int x24739=0; x24739 < x24722; x24739++) {
int32_t x24750 = x24633 * x24739;
int32_t x24751 = x24749 + x24750;
int32_t x24756 = x24736 * x24739;
int32_t x24745 = x24730 * x24739;
for(int x24741=0; x24741 < x24724; x24741++) {
int32_t x24752 = x24734 * x24741;
int32_t x24753 = x24751 + x24752;
int32_t x24747 = x24724 * x24741;
for(int x24742=0; x24742 < x24724; x24742++) {
int32_t x24754 = x24735 * x24742;
int32_t x24755 = x24753 + x24754;
float x24757 = x24638[x24755];
float x24758 = x260[x24756];
int32_t x24744 = x24742 + x24743;
int32_t x24746 = x24744 + x24745;
int32_t x24748 = x24746 + x24747;
float x24759 = x24757 - x24758;
x24733[x24748] = x24759;

}

}

}

}
float* x24769 = (float*)myMalloc(2048 * sizeof(float));;
for(int x24770=0; x24770 < 2048; x24770++) {
float x24771 = x123[x24770];
float x24772 = x24771 + 1.0E-5f;
x24769[x24770] = x24772;

}
float* x24776 = (float*)myMalloc(2048 * sizeof(float));;
for(int x24777=0; x24777 < 2048; x24777++) {
float x24778 = x24769[x24777];
double x24779 = (double)x24778;
double x24780 = sqrt(x24779);
float x24781 = (float)x24780;
x24776[x24777] = x24781;

}
int32_t x24785 = 0;
int32_t x24786 = 1;
x24786 *= 1;
x24785 += 1;
x24786 *= 1;
x24786 *= 1;
int32_t x24791 = x24785;
bool x24792 = x24791 >= 2;
if (x24792) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24797 = x24791 == 0;
if (x24797) {
int32_t x24798 = x24786;
bool x24799 = x24798 == 2048;
if (x24799) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24806 = x24786;
bool x24808 = x24722 == 1;
int32_t x24807 = 2048 / x24806;
bool x24809 = x24807 == 1;
bool x24813;
if (x454) {
bool x24810 = x24808 || x24809;
bool x24811 = x24722 == x24807;
bool x24812 = x24810 || x24811;
x24813 = x24812;
} else {
x24813 = false;
}
bool x24817;
if (x24813) {
x24817 = x24816;
} else {
x24817 = false;
}
bool x24818;
if (x24817) {
x24818 = x24816;
} else {
x24818 = false;
}
if (x24818) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24722,x24724,x24724,1,x24807,1,1);
assert(false && "");
}
bool x24824 = x24722 <= x24807;
int32_t x24825;
if (x24824) {
x24825 = x24807;
} else {
x24825 = x24722;
}
int32_t x24834 = x24825 * x24833;
int32_t x24835 = 64 * x24834;
float* x24836 = (float*)myMalloc(x24835 * sizeof(float));;
int32_t x24837;
if (x24808) {
x24837 = 0;
} else {
x24837 = x24730;
}
int32_t x24840;
if (x24809) {
x24840 = 0;
} else {
x24840 = 1;
}
for(int x24841=0; x24841 < 64; x24841++) {
int32_t x24853 = x24731 * x24841;
int32_t x24847 = x24834 * x24841;
for(int x24843=0; x24843 < x24825; x24843++) {
int32_t x24854 = x24837 * x24843;
int32_t x24855 = x24853 + x24854;
int32_t x24860 = x24840 * x24843;
int32_t x24849 = x24833 * x24843;
for(int x24845=0; x24845 < x24827; x24845++) {
int32_t x24856 = x24838 * x24845;
int32_t x24857 = x24855 + x24856;
int32_t x24851 = x24827 * x24845;
for(int x24846=0; x24846 < x24827; x24846++) {
int32_t x24858 = x24839 * x24846;
int32_t x24859 = x24857 + x24858;
float x24861 = x24733[x24859];
float x24862 = x24776[x24860];
int32_t x24848 = x24846 + x24847;
int32_t x24850 = x24848 + x24849;
int32_t x24852 = x24850 + x24851;
float x24863 = x24861 / x24862;
x24836[x24852] = x24863;

}

}

}

}
int32_t x24873 = 0;
int32_t x24874 = 1;
x24874 *= 1;
x24873 += 1;
x24874 *= 1;
x24874 *= 1;
int32_t x24879 = x24873;
bool x24880 = x24879 >= 2;
if (x24880) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24885 = x24879 == 0;
if (x24885) {
int32_t x24886 = x24874;
bool x24887 = x24886 == 2048;
if (x24887) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24894 = x24874;
bool x24896 = x24825 == 1;
int32_t x24895 = 2048 / x24894;
bool x24897 = x24895 == 1;
bool x24901;
if (x454) {
bool x24898 = x24896 || x24897;
bool x24899 = x24825 == x24895;
bool x24900 = x24898 || x24899;
x24901 = x24900;
} else {
x24901 = false;
}
bool x24905;
if (x24901) {
x24905 = x24904;
} else {
x24905 = false;
}
bool x24906;
if (x24905) {
x24906 = x24904;
} else {
x24906 = false;
}
if (x24906) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24825,x24827,x24827,1,x24895,1,1);
assert(false && "");
}
bool x24912 = x24825 <= x24895;
int32_t x24913;
if (x24912) {
x24913 = x24895;
} else {
x24913 = x24825;
}
int32_t x24922 = x24913 * x24921;
int32_t x24923 = 64 * x24922;
float* x24924 = (float*)myMalloc(x24923 * sizeof(float));;
int32_t x24925;
if (x24896) {
x24925 = 0;
} else {
x24925 = x24833;
}
int32_t x24928;
if (x24897) {
x24928 = 0;
} else {
x24928 = 1;
}
for(int x24929=0; x24929 < 64; x24929++) {
int32_t x24941 = x24834 * x24929;
int32_t x24935 = x24922 * x24929;
for(int x24931=0; x24931 < x24913; x24931++) {
int32_t x24942 = x24925 * x24931;
int32_t x24943 = x24941 + x24942;
int32_t x24948 = x24928 * x24931;
int32_t x24937 = x24921 * x24931;
for(int x24933=0; x24933 < x24915; x24933++) {
int32_t x24944 = x24926 * x24933;
int32_t x24945 = x24943 + x24944;
int32_t x24939 = x24915 * x24933;
for(int x24934=0; x24934 < x24915; x24934++) {
int32_t x24946 = x24927 * x24934;
int32_t x24947 = x24945 + x24946;
float x24949 = x24836[x24947];
float x24950 = x103[x24948];
int32_t x24936 = x24934 + x24935;
int32_t x24938 = x24936 + x24937;
int32_t x24940 = x24938 + x24939;
float x24951 = x24949 * x24950;
x24924[x24940] = x24951;

}

}

}

}
int32_t x24961 = 0;
int32_t x24962 = 1;
x24962 *= 1;
x24961 += 1;
x24962 *= 1;
x24962 *= 1;
int32_t x24967 = x24961;
bool x24968 = x24967 >= 2;
if (x24968) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24973 = x24967 == 0;
if (x24973) {
int32_t x24974 = x24962;
bool x24975 = x24974 == 2048;
if (x24975) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24982 = x24962;
bool x24984 = x24913 == 1;
int32_t x24983 = 2048 / x24982;
bool x24985 = x24983 == 1;
bool x24989;
if (x454) {
bool x24986 = x24984 || x24985;
bool x24987 = x24913 == x24983;
bool x24988 = x24986 || x24987;
x24989 = x24988;
} else {
x24989 = false;
}
bool x24993;
if (x24989) {
x24993 = x24992;
} else {
x24993 = false;
}
bool x24994;
if (x24993) {
x24994 = x24992;
} else {
x24994 = false;
}
if (x24994) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24913,x24915,x24915,1,x24983,1,1);
assert(false && "");
}
bool x25000 = x24913 <= x24983;
int32_t x25001;
if (x25000) {
x25001 = x24983;
} else {
x25001 = x24913;
}
int32_t x25010 = x25001 * x25009;
int32_t x25011 = 64 * x25010;
float* x25012 = (float*)myMalloc(x25011 * sizeof(float));;
int32_t x25013;
if (x24984) {
x25013 = 0;
} else {
x25013 = x24921;
}
int32_t x25016;
if (x24985) {
x25016 = 0;
} else {
x25016 = 1;
}
for(int x25017=0; x25017 < 64; x25017++) {
int32_t x25029 = x24922 * x25017;
int32_t x25023 = x25010 * x25017;
for(int x25019=0; x25019 < x25001; x25019++) {
int32_t x25030 = x25013 * x25019;
int32_t x25031 = x25029 + x25030;
int32_t x25036 = x25016 * x25019;
int32_t x25025 = x25009 * x25019;
for(int x25021=0; x25021 < x25003; x25021++) {
int32_t x25032 = x25014 * x25021;
int32_t x25033 = x25031 + x25032;
int32_t x25027 = x25003 * x25021;
for(int x25022=0; x25022 < x25003; x25022++) {
int32_t x25034 = x25015 * x25022;
int32_t x25035 = x25033 + x25034;
float x25037 = x24924[x25035];
float x25038 = x181[x25036];
int32_t x25024 = x25022 + x25023;
int32_t x25026 = x25024 + x25025;
int32_t x25028 = x25026 + x25027;
float x25039 = x25037 + x25038;
x25012[x25028] = x25039;

}

}

}

}
bool x25049 = x25001 == 1;
bool x25050 = x25049 || x23637;
bool x25051 = x25001 == x23589;
bool x25052 = x25050 || x25051;
bool x25057;
if (x25052) {
x25057 = x25056;
} else {
x25057 = false;
}
bool x25058;
if (x25057) {
x25058 = x25056;
} else {
x25058 = false;
}
if (x25058) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x25001,x25003,x25003,64,x23589,x23591,x23591);
assert(false && "");
}
bool x25064 = x25001 <= x23589;
int32_t x25065;
if (x25064) {
x25065 = x23589;
} else {
x25065 = x25001;
}
int32_t x25081;
if (x25049) {
x25081 = 0;
} else {
x25081 = x25009;
}
for(int x25084=0; x25084 < 64; x25084++) {
int32_t x25090 = x25010 * x25084;
int32_t x25097 = x23598 * x25084;
for(int x25086=0; x25086 < x25065; x25086++) {
int32_t x25091 = x25081 * x25086;
int32_t x25092 = x25090 + x25091;
int32_t x25098 = x23669 * x25086;
int32_t x25099 = x25097 + x25098;
for(int x25088=0; x25088 < x25067; x25088++) {
int32_t x25093 = x25082 * x25088;
int32_t x25094 = x25092 + x25093;
int32_t x25100 = x23670 * x25088;
int32_t x25101 = x25099 + x25100;
for(int x25089=0; x25089 < x25067; x25089++) {
int32_t x25095 = x25083 * x25089;
int32_t x25096 = x25094 + x25095;
float x25104 = x25012[x25096];
int32_t x25102 = x23671 * x25089;
int32_t x25103 = x25101 + x25102;
float x25105 = x23704[x25103];
float x25106 = x25104 + x25105;
x25012[x25096] = x25106;

}

}

}

}
float* x25116 = (float*)myMalloc(x25011 * sizeof(float));;
for(int x25118=0; x25118 < x25011; x25118++) {
float x25119 = x25012[x25118];
bool x25120 = x25119 < 0.0f;
if (x25120) {
x25116[x25118] = 0.0f;
} else {
float x25123 = x25012[x25118];
x25116[x25118] = x25123;
}

}
if (x25130) {
} else {
assert(false && "Image too small for averagePool_batch:  x Const(64) x Sym(25001) x Sym(25003) x Sym(25003)|(2,2)");
}
int32_t x25141 = 64 * x25001;
int32_t x25142 = x25141 * x25137;
int32_t x25143 = x25142 * x25137;
float* x25144 = (float*)myMalloc(x25143 * sizeof(float));;
int32_t x25139 = x25001 * x25138;
for(int x25145=0; x25145 < 64; x25145++) {
int32_t x25146 = x25145 * x25010;
float* x25147 = x25116+x25146;
int32_t x25148 = x25145 * x25139;
float* x25149 = x25144+x25148;
for(int x25150=0; x25150 < x25001; x25150++) {
int32_t x25158 = x25150 * x25009;
int32_t x25154 = x25150 * x25138;
for(int x25152=0; x25152 < x25137; x25152++) {
int32_t x25159 = x25152 * x25003;
int32_t x25160 = x25158 + x25159;
int32_t x25155 = x25152 * x25137;
int32_t x25156 = x25154 + x25155;
for(int x25153=0; x25153 < x25137; x25153++) {
float x25162 = 0.0f;
int32_t x25161 = x25160 + x25153;
float x25163 = x25147[x25161];
x25162 += x25163;
int32_t x25165 = x25161 + 1;
float x25166 = x25147[x25165];
x25162 += x25166;
int32_t x25168 = x25161 + x25003;
float x25169 = x25147[x25168];
x25162 += x25169;
int32_t x25171 = x25168 + 1;
float x25172 = x25147[x25171];
x25162 += x25172;
float x25174 = x25162;
int32_t x25157 = x25156 + x25153;
float x25175 = x25174 / 4.0f;
x25149[x25157] = x25175;

}

}

}

}
int32_t x25185 = 0;
int32_t x25186 = 1;
x25186 *= 64;
x25185 += 1;
int32_t x25189 = x25185;
bool x25190 = x25189 >= 2;
if (x25190) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x25195 = x25189 == 0;
int32_t x25140 = 64 * x25139;
if (x25195) {
int32_t x25196 = x25186;
bool x25197 = x25196 == x25140;
if (x25197) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x25204 = x25186;
// gemm: List(Const(64), Sym(25205)), Vector(Const(10), Const(2048))
assert(false && "ERROR not specified");
float* x25209 = (float*)myMalloc(640 * sizeof(float));;
int32_t x25205 = x25140 / x25204;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 64,10,x25205,1.0,x25144,x25205,x227,x25205,0,x25209,10);
for(int x25211=0; x25211 < 64; x25211++) {
int32_t x25213 = 10 * x25211;
for(int x25212=0; x25212 < 10; x25212++) {
int32_t x25214 = x25213 + x25212;
float x25215 = x25209[x25214];
float x25216 = x48[x25212];
float x25217 = x25215 + x25216;
x25209[x25214] = x25217;

}

}
printf("output (size Const(64) x Const(10))\n");
float x25224 = 0.0f;
for(int x25226=0; x25226 < 640; x25226++) {
float x25227 = x25224;
float x25229 = x25209[x25226];
float x25228 = fabs(x25227);
float x25230 = fabs(x25229);
bool x25231 = x25228 > x25230;
float x25234;
if (x25231) {
x25234 = x25227;
} else {
float x25232 = x25209[x25226];
x25234 = x25232;
}
x25224 = x25234;

}
float x25238 = x25224;
printf("Max Abs: %.5f || ",x25238);
for(int x25240=0; x25240 < 10; x25240++) {
float x25241 = x25209[x25240];
printf("%.5f ",x25241);

}
printf("\n");
assert(false && "stop");

}
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

