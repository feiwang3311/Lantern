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
int32_t x473 = x472 * x472;
int32_t x477;
if (x459) {
x477 = 0;
} else {
x477 = x333;
}
int32_t x478;
if (x459) {
x478 = 0;
} else {
x478 = 1;
}
float* x40 = x5+1856;
float* x110 = x5+1920;
bool x557 = x472 == 1;
bool x558 = x557 || true;
bool x559 = x558 || x557;
bool x569 = x472 <= 1;
int32_t x570;
if (x569) {
x570 = 1;
} else {
x570 = x472;
}
int32_t x571 = x570 * x570;
int32_t x576;
if (x557) {
x576 = 0;
} else {
x576 = x472;
}
int32_t x577;
if (x557) {
x577 = 0;
} else {
x577 = 1;
}
bool x640 = x570 == 1;
bool x641 = x640 || true;
bool x642 = x641 || x640;
bool x652 = x570 <= 1;
int32_t x653;
if (x652) {
x653 = 1;
} else {
x653 = x570;
}
int32_t x654 = x653 * x653;
int32_t x659;
if (x640) {
x659 = 0;
} else {
x659 = x570;
}
int32_t x660;
if (x640) {
x660 = 0;
} else {
x660 = 1;
}
float* x206 = x5+1728;
bool x723 = x653 == 1;
bool x724 = x723 || true;
bool x725 = x724 || x723;
bool x735 = x653 <= 1;
int32_t x736;
if (x735) {
x736 = 1;
} else {
x736 = x653;
}
int32_t x737 = x736 * x736;
int32_t x742;
if (x723) {
x742 = 0;
} else {
x742 = x653;
}
int32_t x743;
if (x723) {
x743 = 0;
} else {
x743 = 1;
}
float* x251 = x5+1792;
bool x790 = x736 >= 2;
bool x791;
if (x790) {
x791 = x790;
} else {
x791 = false;
}
int32_t x796 = x736 - 2;
int32_t x797 = x796 / 2;
int32_t x798 = x797 + 1;
int32_t x799 = x798 * x798;
int32_t x890 = 2 * x736;
int32_t x900 = x797 / 1;
int32_t x901 = x900 + 1;
int32_t x905 = 4096 * x901;
int32_t x906 = x905 * x901;
int32_t x902 = x901 * x901;
int32_t x903 = 64 * x902;
float* x233 = x5+1984;
bool x979 = x901 == 1;
bool x980 = x979 || true;
bool x981 = x980 || x979;
bool x991 = x901 <= 1;
int32_t x992;
if (x991) {
x992 = 1;
} else {
x992 = x901;
}
int32_t x993 = x992 * x992;
int32_t x997;
if (x979) {
x997 = 0;
} else {
x997 = x901;
}
int32_t x998;
if (x979) {
x998 = 0;
} else {
x998 = 1;
}
float* x114 = x5+6208;
float* x51 = x5+6272;
bool x1077 = x992 == 1;
bool x1078 = x1077 || true;
bool x1079 = x1078 || x1077;
bool x1089 = x992 <= 1;
int32_t x1090;
if (x1089) {
x1090 = 1;
} else {
x1090 = x992;
}
int32_t x1091 = x1090 * x1090;
int32_t x1096;
if (x1077) {
x1096 = 0;
} else {
x1096 = x992;
}
int32_t x1097;
if (x1077) {
x1097 = 0;
} else {
x1097 = 1;
}
bool x1160 = x1090 == 1;
bool x1161 = x1160 || true;
bool x1162 = x1161 || x1160;
bool x1172 = x1090 <= 1;
int32_t x1173;
if (x1172) {
x1173 = 1;
} else {
x1173 = x1090;
}
int32_t x1174 = x1173 * x1173;
int32_t x1179;
if (x1160) {
x1179 = 0;
} else {
x1179 = x1090;
}
int32_t x1180;
if (x1160) {
x1180 = 0;
} else {
x1180 = 1;
}
float* x26 = x5+6080;
bool x1243 = x1173 == 1;
bool x1244 = x1243 || true;
bool x1245 = x1244 || x1243;
bool x1255 = x1173 <= 1;
int32_t x1256;
if (x1255) {
x1256 = 1;
} else {
x1256 = x1173;
}
int32_t x1257 = x1256 * x1256;
int32_t x1262;
if (x1243) {
x1262 = 0;
} else {
x1262 = x1173;
}
int32_t x1263;
if (x1243) {
x1263 = 0;
} else {
x1263 = 1;
}
float* x53 = x5+6144;
int32_t x1310 = x1256 + 2;
int32_t x1311 = x1310 - 3;
int32_t x1312 = x1311 / 1;
int32_t x1313 = x1312 + 1;
int32_t x1317 = 4096 * x1313;
int32_t x1318 = x1317 * x1313;
int32_t x1314 = x1313 * x1313;
int32_t x1315 = 64 * x1314;
float* x90 = x5+6336;
bool x1440 = x1313 == 1;
bool x1441 = x1440 || true;
bool x1442 = x1441 || x1440;
bool x1452 = x1313 <= 1;
int32_t x1453;
if (x1452) {
x1453 = 1;
} else {
x1453 = x1313;
}
int32_t x1454 = x1453 * x1453;
int32_t x1458;
if (x1440) {
x1458 = 0;
} else {
x1458 = x1313;
}
int32_t x1459;
if (x1440) {
x1459 = 0;
} else {
x1459 = 1;
}
float* x105 = x5+43328;
float* x158 = x5+43392;
bool x1538 = x1453 == 1;
bool x1539 = x1538 || true;
bool x1540 = x1539 || x1538;
bool x1550 = x1453 <= 1;
int32_t x1551;
if (x1550) {
x1551 = 1;
} else {
x1551 = x1453;
}
int32_t x1552 = x1551 * x1551;
int32_t x1557;
if (x1538) {
x1557 = 0;
} else {
x1557 = x1453;
}
int32_t x1558;
if (x1538) {
x1558 = 0;
} else {
x1558 = 1;
}
bool x1621 = x1551 == 1;
bool x1622 = x1621 || true;
bool x1623 = x1622 || x1621;
bool x1633 = x1551 <= 1;
int32_t x1634;
if (x1633) {
x1634 = 1;
} else {
x1634 = x1551;
}
int32_t x1635 = x1634 * x1634;
int32_t x1640;
if (x1621) {
x1640 = 0;
} else {
x1640 = x1551;
}
int32_t x1641;
if (x1621) {
x1641 = 0;
} else {
x1641 = 1;
}
float* x164 = x5+43200;
bool x1704 = x1634 == 1;
bool x1705 = x1704 || true;
bool x1706 = x1705 || x1704;
bool x1716 = x1634 <= 1;
int32_t x1717;
if (x1716) {
x1717 = 1;
} else {
x1717 = x1634;
}
int32_t x1718 = x1717 * x1717;
int32_t x1723;
if (x1704) {
x1723 = 0;
} else {
x1723 = x1634;
}
int32_t x1724;
if (x1704) {
x1724 = 0;
} else {
x1724 = 1;
}
float* x49 = x5+43264;
int32_t x1771 = x1717 - 1;
int32_t x1772 = x1771 / 1;
int32_t x1773 = x1772 + 1;
int32_t x1777 = 16384 * x1773;
int32_t x1778 = x1777 * x1773;
int32_t x1774 = x1773 * x1773;
int32_t x1775 = 256 * x1774;
float* x32 = x5+43456;
bool x1852 = x1773 == 1;
bool x1853 = x1852 || true;
bool x1854 = x1853 || x1852;
bool x1864 = x1773 <= 1;
int32_t x1865;
if (x1864) {
x1865 = 1;
} else {
x1865 = x1773;
}
int32_t x1866 = x1865 * x1865;
int32_t x1870;
if (x1852) {
x1870 = 0;
} else {
x1870 = x1773;
}
int32_t x1871;
if (x1852) {
x1871 = 0;
} else {
x1871 = 1;
}
float* x71 = x5+60352;
float* x36 = x5+60608;
bool x1951 = x1865 == 1;
bool x1952 = x1951 || true;
bool x1953 = x1952 || x1951;
bool x1963 = x1865 <= 1;
int32_t x1964;
if (x1963) {
x1964 = 1;
} else {
x1964 = x1865;
}
int32_t x1965 = x1964 * x1964;
int32_t x1970;
if (x1951) {
x1970 = 0;
} else {
x1970 = x1865;
}
int32_t x1971;
if (x1951) {
x1971 = 0;
} else {
x1971 = 1;
}
bool x2034 = x1964 == 1;
bool x2035 = x2034 || true;
bool x2036 = x2035 || x2034;
bool x2046 = x1964 <= 1;
int32_t x2047;
if (x2046) {
x2047 = 1;
} else {
x2047 = x1964;
}
int32_t x2048 = x2047 * x2047;
int32_t x2053;
if (x2034) {
x2053 = 0;
} else {
x2053 = x1964;
}
int32_t x2054;
if (x2034) {
x2054 = 0;
} else {
x2054 = 1;
}
float* x199 = x5+59840;
bool x2117 = x2047 == 1;
bool x2118 = x2117 || true;
bool x2119 = x2118 || x2117;
bool x2129 = x2047 <= 1;
int32_t x2130;
if (x2129) {
x2130 = 1;
} else {
x2130 = x2047;
}
int32_t x2131 = x2130 * x2130;
int32_t x2136;
if (x2117) {
x2136 = 0;
} else {
x2136 = x2047;
}
int32_t x2137;
if (x2117) {
x2137 = 0;
} else {
x2137 = 1;
}
float* x126 = x5+60096;
int32_t x2173 = 16384 * x901;
int32_t x2174 = x2173 * x901;
int32_t x2171 = 256 * x902;
float* x162 = x5+60864;
float* x264 = x5+77760;
float* x243 = x5+78016;
float* x76 = x5+77248;
float* x203 = x5+77504;
bool x2530 = x2130 == 1;
bool x2531 = x1256 == 1;
bool x2532 = x2530 || x2531;
bool x2533 = x2130 == x1256;
bool x2534 = x2532 || x2533;
bool x2544 = x2130 <= x1256;
int32_t x2545;
if (x2544) {
x2545 = x1256;
} else {
x2545 = x2130;
}
int32_t x2550;
if (x2530) {
x2550 = 0;
} else {
x2550 = x2130;
}
int32_t x2551;
if (x2530) {
x2551 = 0;
} else {
x2551 = 1;
}
int32_t x2553;
if (x2531) {
x2553 = 0;
} else {
x2553 = x1256;
}
int32_t x2554;
if (x2531) {
x2554 = 0;
} else {
x2554 = 1;
}
int32_t x2600 = x2130 - 1;
int32_t x2601 = x2600 / 1;
int32_t x2602 = x2601 + 1;
int32_t x2606 = 4096 * x2602;
int32_t x2607 = x2606 * x2602;
int32_t x2603 = x2602 * x2602;
int32_t x2604 = 64 * x2603;
float* x171 = x5+78272;
bool x2681 = x2602 == 1;
bool x2682 = x2681 || true;
bool x2683 = x2682 || x2681;
bool x2693 = x2602 <= 1;
int32_t x2694;
if (x2693) {
x2694 = 1;
} else {
x2694 = x2602;
}
int32_t x2695 = x2694 * x2694;
int32_t x2699;
if (x2681) {
x2699 = 0;
} else {
x2699 = x2602;
}
int32_t x2700;
if (x2681) {
x2700 = 0;
} else {
x2700 = 1;
}
float* x10 = x5+94784;
float* x102 = x5+94848;
bool x2779 = x2694 == 1;
bool x2780 = x2779 || true;
bool x2781 = x2780 || x2779;
bool x2791 = x2694 <= 1;
int32_t x2792;
if (x2791) {
x2792 = 1;
} else {
x2792 = x2694;
}
int32_t x2793 = x2792 * x2792;
int32_t x2798;
if (x2779) {
x2798 = 0;
} else {
x2798 = x2694;
}
int32_t x2799;
if (x2779) {
x2799 = 0;
} else {
x2799 = 1;
}
bool x2862 = x2792 == 1;
bool x2863 = x2862 || true;
bool x2864 = x2863 || x2862;
bool x2874 = x2792 <= 1;
int32_t x2875;
if (x2874) {
x2875 = 1;
} else {
x2875 = x2792;
}
int32_t x2876 = x2875 * x2875;
int32_t x2881;
if (x2862) {
x2881 = 0;
} else {
x2881 = x2792;
}
int32_t x2882;
if (x2862) {
x2882 = 0;
} else {
x2882 = 1;
}
float* x142 = x5+94656;
bool x2945 = x2875 == 1;
bool x2946 = x2945 || true;
bool x2947 = x2946 || x2945;
bool x2957 = x2875 <= 1;
int32_t x2958;
if (x2957) {
x2958 = 1;
} else {
x2958 = x2875;
}
int32_t x2959 = x2958 * x2958;
int32_t x2964;
if (x2945) {
x2964 = 0;
} else {
x2964 = x2875;
}
int32_t x2965;
if (x2945) {
x2965 = 0;
} else {
x2965 = 1;
}
float* x60 = x5+94720;
int32_t x3012 = x2958 + 2;
int32_t x3013 = x3012 - 3;
int32_t x3014 = x3013 / 1;
int32_t x3015 = x3014 + 1;
int32_t x3019 = 4096 * x3015;
int32_t x3020 = x3019 * x3015;
int32_t x3016 = x3015 * x3015;
int32_t x3017 = 64 * x3016;
float* x83 = x5+94912;
bool x3142 = x3015 == 1;
bool x3143 = x3142 || true;
bool x3144 = x3143 || x3142;
bool x3154 = x3015 <= 1;
int32_t x3155;
if (x3154) {
x3155 = 1;
} else {
x3155 = x3015;
}
int32_t x3156 = x3155 * x3155;
int32_t x3160;
if (x3142) {
x3160 = 0;
} else {
x3160 = x3015;
}
int32_t x3161;
if (x3142) {
x3161 = 0;
} else {
x3161 = 1;
}
float* x44 = x5+131904;
float* x244 = x5+131968;
bool x3240 = x3155 == 1;
bool x3241 = x3240 || true;
bool x3242 = x3241 || x3240;
bool x3252 = x3155 <= 1;
int32_t x3253;
if (x3252) {
x3253 = 1;
} else {
x3253 = x3155;
}
int32_t x3254 = x3253 * x3253;
int32_t x3259;
if (x3240) {
x3259 = 0;
} else {
x3259 = x3155;
}
int32_t x3260;
if (x3240) {
x3260 = 0;
} else {
x3260 = 1;
}
bool x3323 = x3253 == 1;
bool x3324 = x3323 || true;
bool x3325 = x3324 || x3323;
bool x3335 = x3253 <= 1;
int32_t x3336;
if (x3335) {
x3336 = 1;
} else {
x3336 = x3253;
}
int32_t x3337 = x3336 * x3336;
int32_t x3342;
if (x3323) {
x3342 = 0;
} else {
x3342 = x3253;
}
int32_t x3343;
if (x3323) {
x3343 = 0;
} else {
x3343 = 1;
}
float* x208 = x5+131776;
bool x3406 = x3336 == 1;
bool x3407 = x3406 || true;
bool x3408 = x3407 || x3406;
bool x3418 = x3336 <= 1;
int32_t x3419;
if (x3418) {
x3419 = 1;
} else {
x3419 = x3336;
}
int32_t x3420 = x3419 * x3419;
int32_t x3425;
if (x3406) {
x3425 = 0;
} else {
x3425 = x3336;
}
int32_t x3426;
if (x3406) {
x3426 = 0;
} else {
x3426 = 1;
}
float* x153 = x5+131840;
int32_t x3473 = x3419 - 1;
int32_t x3474 = x3473 / 1;
int32_t x3475 = x3474 + 1;
int32_t x3479 = 16384 * x3475;
int32_t x3480 = x3479 * x3475;
int32_t x3476 = x3475 * x3475;
int32_t x3477 = 256 * x3476;
float* x130 = x5+132032;
bool x3554 = x3475 == 1;
bool x3555 = x3554 || true;
bool x3556 = x3555 || x3554;
bool x3566 = x3475 <= 1;
int32_t x3567;
if (x3566) {
x3567 = 1;
} else {
x3567 = x3475;
}
int32_t x3568 = x3567 * x3567;
int32_t x3572;
if (x3554) {
x3572 = 0;
} else {
x3572 = x3475;
}
int32_t x3573;
if (x3554) {
x3573 = 0;
} else {
x3573 = 1;
}
float* x91 = x5+148928;
float* x166 = x5+149184;
bool x3652 = x3567 == 1;
bool x3653 = x3652 || true;
bool x3654 = x3653 || x3652;
bool x3664 = x3567 <= 1;
int32_t x3665;
if (x3664) {
x3665 = 1;
} else {
x3665 = x3567;
}
int32_t x3666 = x3665 * x3665;
int32_t x3671;
if (x3652) {
x3671 = 0;
} else {
x3671 = x3567;
}
int32_t x3672;
if (x3652) {
x3672 = 0;
} else {
x3672 = 1;
}
bool x3735 = x3665 == 1;
bool x3736 = x3735 || true;
bool x3737 = x3736 || x3735;
bool x3747 = x3665 <= 1;
int32_t x3748;
if (x3747) {
x3748 = 1;
} else {
x3748 = x3665;
}
int32_t x3749 = x3748 * x3748;
int32_t x3754;
if (x3735) {
x3754 = 0;
} else {
x3754 = x3665;
}
int32_t x3755;
if (x3735) {
x3755 = 0;
} else {
x3755 = 1;
}
float* x58 = x5+148416;
bool x3818 = x3748 == 1;
bool x3819 = x3818 || true;
bool x3820 = x3819 || x3818;
bool x3830 = x3748 <= 1;
int32_t x3831;
if (x3830) {
x3831 = 1;
} else {
x3831 = x3748;
}
int32_t x3832 = x3831 * x3831;
int32_t x3837;
if (x3818) {
x3837 = 0;
} else {
x3837 = x3748;
}
int32_t x3838;
if (x3818) {
x3838 = 0;
} else {
x3838 = 1;
}
float* x7 = x5+148672;
bool x3876 = x3831 == 1;
bool x3877 = x3876 || x2530;
bool x3878 = x3831 == x2130;
bool x3879 = x3877 || x3878;
bool x3889 = x3831 <= x2130;
int32_t x3890;
if (x3889) {
x3890 = x2130;
} else {
x3890 = x3831;
}
int32_t x3895;
if (x3876) {
x3895 = 0;
} else {
x3895 = x3831;
}
int32_t x3896;
if (x3876) {
x3896 = 0;
} else {
x3896 = 1;
}
int32_t x3942 = x3831 - 1;
int32_t x3943 = x3942 / 1;
int32_t x3944 = x3943 + 1;
int32_t x3948 = 4096 * x3944;
int32_t x3949 = x3948 * x3944;
int32_t x3945 = x3944 * x3944;
int32_t x3946 = 64 * x3945;
float* x150 = x5+149440;
bool x4023 = x3944 == 1;
bool x4024 = x4023 || true;
bool x4025 = x4024 || x4023;
bool x4035 = x3944 <= 1;
int32_t x4036;
if (x4035) {
x4036 = 1;
} else {
x4036 = x3944;
}
int32_t x4037 = x4036 * x4036;
int32_t x4041;
if (x4023) {
x4041 = 0;
} else {
x4041 = x3944;
}
int32_t x4042;
if (x4023) {
x4042 = 0;
} else {
x4042 = 1;
}
float* x257 = x5+165952;
float* x187 = x5+166016;
bool x4121 = x4036 == 1;
bool x4122 = x4121 || true;
bool x4123 = x4122 || x4121;
bool x4133 = x4036 <= 1;
int32_t x4134;
if (x4133) {
x4134 = 1;
} else {
x4134 = x4036;
}
int32_t x4135 = x4134 * x4134;
int32_t x4140;
if (x4121) {
x4140 = 0;
} else {
x4140 = x4036;
}
int32_t x4141;
if (x4121) {
x4141 = 0;
} else {
x4141 = 1;
}
bool x4204 = x4134 == 1;
bool x4205 = x4204 || true;
bool x4206 = x4205 || x4204;
bool x4216 = x4134 <= 1;
int32_t x4217;
if (x4216) {
x4217 = 1;
} else {
x4217 = x4134;
}
int32_t x4218 = x4217 * x4217;
int32_t x4223;
if (x4204) {
x4223 = 0;
} else {
x4223 = x4134;
}
int32_t x4224;
if (x4204) {
x4224 = 0;
} else {
x4224 = 1;
}
float* x81 = x5+165824;
bool x4287 = x4217 == 1;
bool x4288 = x4287 || true;
bool x4289 = x4288 || x4287;
bool x4299 = x4217 <= 1;
int32_t x4300;
if (x4299) {
x4300 = 1;
} else {
x4300 = x4217;
}
int32_t x4301 = x4300 * x4300;
int32_t x4306;
if (x4287) {
x4306 = 0;
} else {
x4306 = x4217;
}
int32_t x4307;
if (x4287) {
x4307 = 0;
} else {
x4307 = 1;
}
float* x24 = x5+165888;
int32_t x4354 = x4300 + 2;
int32_t x4355 = x4354 - 3;
int32_t x4356 = x4355 / 1;
int32_t x4357 = x4356 + 1;
int32_t x4361 = 4096 * x4357;
int32_t x4362 = x4361 * x4357;
int32_t x4358 = x4357 * x4357;
int32_t x4359 = 64 * x4358;
float* x73 = x5+166080;
bool x4484 = x4357 == 1;
bool x4485 = x4484 || true;
bool x4486 = x4485 || x4484;
bool x4496 = x4357 <= 1;
int32_t x4497;
if (x4496) {
x4497 = 1;
} else {
x4497 = x4357;
}
int32_t x4498 = x4497 * x4497;
int32_t x4502;
if (x4484) {
x4502 = 0;
} else {
x4502 = x4357;
}
int32_t x4503;
if (x4484) {
x4503 = 0;
} else {
x4503 = 1;
}
float* x179 = x5+203072;
float* x118 = x5+203136;
bool x4582 = x4497 == 1;
bool x4583 = x4582 || true;
bool x4584 = x4583 || x4582;
bool x4594 = x4497 <= 1;
int32_t x4595;
if (x4594) {
x4595 = 1;
} else {
x4595 = x4497;
}
int32_t x4596 = x4595 * x4595;
int32_t x4601;
if (x4582) {
x4601 = 0;
} else {
x4601 = x4497;
}
int32_t x4602;
if (x4582) {
x4602 = 0;
} else {
x4602 = 1;
}
bool x4665 = x4595 == 1;
bool x4666 = x4665 || true;
bool x4667 = x4666 || x4665;
bool x4677 = x4595 <= 1;
int32_t x4678;
if (x4677) {
x4678 = 1;
} else {
x4678 = x4595;
}
int32_t x4679 = x4678 * x4678;
int32_t x4684;
if (x4665) {
x4684 = 0;
} else {
x4684 = x4595;
}
int32_t x4685;
if (x4665) {
x4685 = 0;
} else {
x4685 = 1;
}
float* x72 = x5+202944;
bool x4748 = x4678 == 1;
bool x4749 = x4748 || true;
bool x4750 = x4749 || x4748;
bool x4760 = x4678 <= 1;
int32_t x4761;
if (x4760) {
x4761 = 1;
} else {
x4761 = x4678;
}
int32_t x4762 = x4761 * x4761;
int32_t x4767;
if (x4748) {
x4767 = 0;
} else {
x4767 = x4678;
}
int32_t x4768;
if (x4748) {
x4768 = 0;
} else {
x4768 = 1;
}
float* x135 = x5+203008;
int32_t x4815 = x4761 - 1;
int32_t x4816 = x4815 / 1;
int32_t x4817 = x4816 + 1;
int32_t x4821 = 16384 * x4817;
int32_t x4822 = x4821 * x4817;
int32_t x4818 = x4817 * x4817;
int32_t x4819 = 256 * x4818;
float* x87 = x5+203200;
bool x4896 = x4817 == 1;
bool x4897 = x4896 || true;
bool x4898 = x4897 || x4896;
bool x4908 = x4817 <= 1;
int32_t x4909;
if (x4908) {
x4909 = 1;
} else {
x4909 = x4817;
}
int32_t x4910 = x4909 * x4909;
int32_t x4914;
if (x4896) {
x4914 = 0;
} else {
x4914 = x4817;
}
int32_t x4915;
if (x4896) {
x4915 = 0;
} else {
x4915 = 1;
}
float* x184 = x5+220096;
float* x133 = x5+220352;
bool x4994 = x4909 == 1;
bool x4995 = x4994 || true;
bool x4996 = x4995 || x4994;
bool x5006 = x4909 <= 1;
int32_t x5007;
if (x5006) {
x5007 = 1;
} else {
x5007 = x4909;
}
int32_t x5008 = x5007 * x5007;
int32_t x5013;
if (x4994) {
x5013 = 0;
} else {
x5013 = x4909;
}
int32_t x5014;
if (x4994) {
x5014 = 0;
} else {
x5014 = 1;
}
bool x5077 = x5007 == 1;
bool x5078 = x5077 || true;
bool x5079 = x5078 || x5077;
bool x5089 = x5007 <= 1;
int32_t x5090;
if (x5089) {
x5090 = 1;
} else {
x5090 = x5007;
}
int32_t x5091 = x5090 * x5090;
int32_t x5096;
if (x5077) {
x5096 = 0;
} else {
x5096 = x5007;
}
int32_t x5097;
if (x5077) {
x5097 = 0;
} else {
x5097 = 1;
}
float* x37 = x5+219584;
bool x5160 = x5090 == 1;
bool x5161 = x5160 || true;
bool x5162 = x5161 || x5160;
bool x5172 = x5090 <= 1;
int32_t x5173;
if (x5172) {
x5173 = 1;
} else {
x5173 = x5090;
}
int32_t x5174 = x5173 * x5173;
int32_t x5179;
if (x5160) {
x5179 = 0;
} else {
x5179 = x5090;
}
int32_t x5180;
if (x5160) {
x5180 = 0;
} else {
x5180 = 1;
}
float* x247 = x5+219840;
bool x5218 = x5173 == 1;
bool x5219 = x5218 || x3876;
bool x5220 = x5173 == x3831;
bool x5221 = x5219 || x5220;
bool x5231 = x5173 <= x3831;
int32_t x5232;
if (x5231) {
x5232 = x3831;
} else {
x5232 = x5173;
}
int32_t x5237;
if (x5218) {
x5237 = 0;
} else {
x5237 = x5173;
}
int32_t x5238;
if (x5218) {
x5238 = 0;
} else {
x5238 = 1;
}
int32_t x5284 = x5173 - 1;
int32_t x5285 = x5284 / 1;
int32_t x5286 = x5285 + 1;
int32_t x5290 = 8192 * x5286;
int32_t x5291 = x5290 * x5286;
int32_t x5287 = x5286 * x5286;
int32_t x5288 = 128 * x5287;
float* x11 = x5+220608;
bool x5365 = x5286 == 1;
bool x5366 = x5365 || true;
bool x5367 = x5366 || x5365;
bool x5377 = x5286 <= 1;
int32_t x5378;
if (x5377) {
x5378 = 1;
} else {
x5378 = x5286;
}
int32_t x5379 = x5378 * x5378;
int32_t x5383;
if (x5365) {
x5383 = 0;
} else {
x5383 = x5286;
}
int32_t x5384;
if (x5365) {
x5384 = 0;
} else {
x5384 = 1;
}
float* x204 = x5+253632;
float* x134 = x5+253760;
bool x5464 = x5378 == 1;
bool x5465 = x5464 || true;
bool x5466 = x5465 || x5464;
bool x5476 = x5378 <= 1;
int32_t x5477;
if (x5476) {
x5477 = 1;
} else {
x5477 = x5378;
}
int32_t x5478 = x5477 * x5477;
int32_t x5483;
if (x5464) {
x5483 = 0;
} else {
x5483 = x5378;
}
int32_t x5484;
if (x5464) {
x5484 = 0;
} else {
x5484 = 1;
}
bool x5547 = x5477 == 1;
bool x5548 = x5547 || true;
bool x5549 = x5548 || x5547;
bool x5559 = x5477 <= 1;
int32_t x5560;
if (x5559) {
x5560 = 1;
} else {
x5560 = x5477;
}
int32_t x5561 = x5560 * x5560;
int32_t x5566;
if (x5547) {
x5566 = 0;
} else {
x5566 = x5477;
}
int32_t x5567;
if (x5547) {
x5567 = 0;
} else {
x5567 = 1;
}
float* x84 = x5+253376;
bool x5630 = x5560 == 1;
bool x5631 = x5630 || true;
bool x5632 = x5631 || x5630;
bool x5642 = x5560 <= 1;
int32_t x5643;
if (x5642) {
x5643 = 1;
} else {
x5643 = x5560;
}
int32_t x5644 = x5643 * x5643;
int32_t x5649;
if (x5630) {
x5649 = 0;
} else {
x5649 = x5560;
}
int32_t x5650;
if (x5630) {
x5650 = 0;
} else {
x5650 = 1;
}
float* x172 = x5+253504;
int32_t x5697 = x5643 + 2;
int32_t x5698 = x5697 - 3;
int32_t x5699 = x5698 / 2;
int32_t x5700 = x5699 + 1;
int32_t x5704 = 8192 * x5700;
int32_t x5705 = x5704 * x5700;
int32_t x5701 = x5700 * x5700;
int32_t x5702 = 128 * x5701;
float* x27 = x5+253888;
bool x5811 = x5700 == 1;
bool x5812 = x5811 || true;
bool x5813 = x5812 || x5811;
bool x5823 = x5700 <= 1;
int32_t x5824;
if (x5823) {
x5824 = 1;
} else {
x5824 = x5700;
}
int32_t x5825 = x5824 * x5824;
int32_t x5829;
if (x5811) {
x5829 = 0;
} else {
x5829 = x5700;
}
int32_t x5830;
if (x5811) {
x5830 = 0;
} else {
x5830 = 1;
}
float* x128 = x5+401600;
float* x43 = x5+401728;
bool x5909 = x5824 == 1;
bool x5910 = x5909 || true;
bool x5911 = x5910 || x5909;
bool x5921 = x5824 <= 1;
int32_t x5922;
if (x5921) {
x5922 = 1;
} else {
x5922 = x5824;
}
int32_t x5923 = x5922 * x5922;
int32_t x5928;
if (x5909) {
x5928 = 0;
} else {
x5928 = x5824;
}
int32_t x5929;
if (x5909) {
x5929 = 0;
} else {
x5929 = 1;
}
bool x5992 = x5922 == 1;
bool x5993 = x5992 || true;
bool x5994 = x5993 || x5992;
bool x6004 = x5922 <= 1;
int32_t x6005;
if (x6004) {
x6005 = 1;
} else {
x6005 = x5922;
}
int32_t x6006 = x6005 * x6005;
int32_t x6011;
if (x5992) {
x6011 = 0;
} else {
x6011 = x5922;
}
int32_t x6012;
if (x5992) {
x6012 = 0;
} else {
x6012 = 1;
}
float* x252 = x5+401344;
bool x6075 = x6005 == 1;
bool x6076 = x6075 || true;
bool x6077 = x6076 || x6075;
bool x6087 = x6005 <= 1;
int32_t x6088;
if (x6087) {
x6088 = 1;
} else {
x6088 = x6005;
}
int32_t x6089 = x6088 * x6088;
int32_t x6094;
if (x6075) {
x6094 = 0;
} else {
x6094 = x6005;
}
int32_t x6095;
if (x6075) {
x6095 = 0;
} else {
x6095 = 1;
}
float* x190 = x5+401472;
int32_t x6142 = x6088 - 1;
int32_t x6143 = x6142 / 1;
int32_t x6144 = x6143 + 1;
int32_t x6148 = 32768 * x6144;
int32_t x6149 = x6148 * x6144;
int32_t x6145 = x6144 * x6144;
int32_t x6146 = 512 * x6145;
float* x106 = x5+401856;
bool x6223 = x6144 == 1;
bool x6224 = x6223 || true;
bool x6225 = x6224 || x6223;
bool x6235 = x6144 <= 1;
int32_t x6236;
if (x6235) {
x6236 = 1;
} else {
x6236 = x6144;
}
int32_t x6237 = x6236 * x6236;
int32_t x6241;
if (x6223) {
x6241 = 0;
} else {
x6241 = x6144;
}
int32_t x6242;
if (x6223) {
x6242 = 0;
} else {
x6242 = 1;
}
float* x149 = x5+468416;
float* x101 = x5+468928;
bool x6322 = x6236 == 1;
bool x6323 = x6322 || true;
bool x6324 = x6323 || x6322;
bool x6334 = x6236 <= 1;
int32_t x6335;
if (x6334) {
x6335 = 1;
} else {
x6335 = x6236;
}
int32_t x6336 = x6335 * x6335;
int32_t x6341;
if (x6322) {
x6341 = 0;
} else {
x6341 = x6236;
}
int32_t x6342;
if (x6322) {
x6342 = 0;
} else {
x6342 = 1;
}
bool x6405 = x6335 == 1;
bool x6406 = x6405 || true;
bool x6407 = x6406 || x6405;
bool x6417 = x6335 <= 1;
int32_t x6418;
if (x6417) {
x6418 = 1;
} else {
x6418 = x6335;
}
int32_t x6419 = x6418 * x6418;
int32_t x6424;
if (x6405) {
x6424 = 0;
} else {
x6424 = x6335;
}
int32_t x6425;
if (x6405) {
x6425 = 0;
} else {
x6425 = 1;
}
float* x145 = x5+467392;
bool x6488 = x6418 == 1;
bool x6489 = x6488 || true;
bool x6490 = x6489 || x6488;
bool x6500 = x6418 <= 1;
int32_t x6501;
if (x6500) {
x6501 = 1;
} else {
x6501 = x6418;
}
int32_t x6502 = x6501 * x6501;
int32_t x6507;
if (x6488) {
x6507 = 0;
} else {
x6507 = x6418;
}
int32_t x6508;
if (x6488) {
x6508 = 0;
} else {
x6508 = 1;
}
float* x210 = x5+467904;
int32_t x6542 = x5284 / 2;
int32_t x6543 = x6542 + 1;
int32_t x6547 = 32768 * x6543;
int32_t x6548 = x6547 * x6543;
int32_t x6544 = x6543 * x6543;
int32_t x6545 = 512 * x6544;
float* x258 = x5+469440;
bool x6628 = x6543 == 1;
bool x6629 = x6628 || true;
bool x6630 = x6629 || x6628;
bool x6640 = x6543 <= 1;
int32_t x6641;
if (x6640) {
x6641 = 1;
} else {
x6641 = x6543;
}
int32_t x6642 = x6641 * x6641;
int32_t x6646;
if (x6628) {
x6646 = 0;
} else {
x6646 = x6543;
}
int32_t x6647;
if (x6628) {
x6647 = 0;
} else {
x6647 = 1;
}
float* x42 = x5+601536;
float* x23 = x5+602048;
bool x6726 = x6641 == 1;
bool x6727 = x6726 || true;
bool x6728 = x6727 || x6726;
bool x6738 = x6641 <= 1;
int32_t x6739;
if (x6738) {
x6739 = 1;
} else {
x6739 = x6641;
}
int32_t x6740 = x6739 * x6739;
int32_t x6745;
if (x6726) {
x6745 = 0;
} else {
x6745 = x6641;
}
int32_t x6746;
if (x6726) {
x6746 = 0;
} else {
x6746 = 1;
}
bool x6809 = x6739 == 1;
bool x6810 = x6809 || true;
bool x6811 = x6810 || x6809;
bool x6821 = x6739 <= 1;
int32_t x6822;
if (x6821) {
x6822 = 1;
} else {
x6822 = x6739;
}
int32_t x6823 = x6822 * x6822;
int32_t x6828;
if (x6809) {
x6828 = 0;
} else {
x6828 = x6739;
}
int32_t x6829;
if (x6809) {
x6829 = 0;
} else {
x6829 = 1;
}
float* x207 = x5+600512;
bool x6892 = x6822 == 1;
bool x6893 = x6892 || true;
bool x6894 = x6893 || x6892;
bool x6904 = x6822 <= 1;
int32_t x6905;
if (x6904) {
x6905 = 1;
} else {
x6905 = x6822;
}
int32_t x6906 = x6905 * x6905;
int32_t x6911;
if (x6892) {
x6911 = 0;
} else {
x6911 = x6822;
}
int32_t x6912;
if (x6892) {
x6912 = 0;
} else {
x6912 = 1;
}
float* x119 = x5+601024;
bool x6951 = x6501 == 1;
bool x6952 = x6905 == 1;
bool x6953 = x6951 || x6952;
bool x6954 = x6501 == x6905;
bool x6955 = x6953 || x6954;
bool x6965 = x6501 <= x6905;
int32_t x6966;
if (x6965) {
x6966 = x6905;
} else {
x6966 = x6501;
}
int32_t x6971;
if (x6951) {
x6971 = 0;
} else {
x6971 = x6501;
}
int32_t x6972;
if (x6951) {
x6972 = 0;
} else {
x6972 = 1;
}
int32_t x6974;
if (x6952) {
x6974 = 0;
} else {
x6974 = x6905;
}
int32_t x6975;
if (x6952) {
x6975 = 0;
} else {
x6975 = 1;
}
int32_t x7021 = x6501 - 1;
int32_t x7022 = x7021 / 1;
int32_t x7023 = x7022 + 1;
int32_t x7027 = 8192 * x7023;
int32_t x7028 = x7027 * x7023;
int32_t x7024 = x7023 * x7023;
int32_t x7025 = 128 * x7024;
float* x256 = x5+602560;
bool x7102 = x7023 == 1;
bool x7103 = x7102 || true;
bool x7104 = x7103 || x7102;
bool x7114 = x7023 <= 1;
int32_t x7115;
if (x7114) {
x7115 = 1;
} else {
x7115 = x7023;
}
int32_t x7116 = x7115 * x7115;
int32_t x7120;
if (x7102) {
x7120 = 0;
} else {
x7120 = x7023;
}
int32_t x7121;
if (x7102) {
x7121 = 0;
} else {
x7121 = 1;
}
float* x100 = x5+668352;
float* x177 = x5+668480;
bool x7200 = x7115 == 1;
bool x7201 = x7200 || true;
bool x7202 = x7201 || x7200;
bool x7212 = x7115 <= 1;
int32_t x7213;
if (x7212) {
x7213 = 1;
} else {
x7213 = x7115;
}
int32_t x7214 = x7213 * x7213;
int32_t x7219;
if (x7200) {
x7219 = 0;
} else {
x7219 = x7115;
}
int32_t x7220;
if (x7200) {
x7220 = 0;
} else {
x7220 = 1;
}
bool x7283 = x7213 == 1;
bool x7284 = x7283 || true;
bool x7285 = x7284 || x7283;
bool x7295 = x7213 <= 1;
int32_t x7296;
if (x7295) {
x7296 = 1;
} else {
x7296 = x7213;
}
int32_t x7297 = x7296 * x7296;
int32_t x7302;
if (x7283) {
x7302 = 0;
} else {
x7302 = x7213;
}
int32_t x7303;
if (x7283) {
x7303 = 0;
} else {
x7303 = 1;
}
float* x222 = x5+668096;
bool x7366 = x7296 == 1;
bool x7367 = x7366 || true;
bool x7368 = x7367 || x7366;
bool x7378 = x7296 <= 1;
int32_t x7379;
if (x7378) {
x7379 = 1;
} else {
x7379 = x7296;
}
int32_t x7380 = x7379 * x7379;
int32_t x7385;
if (x7366) {
x7385 = 0;
} else {
x7385 = x7296;
}
int32_t x7386;
if (x7366) {
x7386 = 0;
} else {
x7386 = 1;
}
float* x17 = x5+668224;
int32_t x7433 = x7379 + 2;
int32_t x7434 = x7433 - 3;
int32_t x7435 = x7434 / 1;
int32_t x7436 = x7435 + 1;
int32_t x7440 = 8192 * x7436;
int32_t x7441 = x7440 * x7436;
int32_t x7437 = x7436 * x7436;
int32_t x7438 = 128 * x7437;
float* x235 = x5+668608;
bool x7563 = x7436 == 1;
bool x7564 = x7563 || true;
bool x7565 = x7564 || x7563;
bool x7575 = x7436 <= 1;
int32_t x7576;
if (x7575) {
x7576 = 1;
} else {
x7576 = x7436;
}
int32_t x7577 = x7576 * x7576;
int32_t x7581;
if (x7563) {
x7581 = 0;
} else {
x7581 = x7436;
}
int32_t x7582;
if (x7563) {
x7582 = 0;
} else {
x7582 = 1;
}
float* x35 = x5+816320;
float* x225 = x5+816448;
bool x7661 = x7576 == 1;
bool x7662 = x7661 || true;
bool x7663 = x7662 || x7661;
bool x7673 = x7576 <= 1;
int32_t x7674;
if (x7673) {
x7674 = 1;
} else {
x7674 = x7576;
}
int32_t x7675 = x7674 * x7674;
int32_t x7680;
if (x7661) {
x7680 = 0;
} else {
x7680 = x7576;
}
int32_t x7681;
if (x7661) {
x7681 = 0;
} else {
x7681 = 1;
}
bool x7744 = x7674 == 1;
bool x7745 = x7744 || true;
bool x7746 = x7745 || x7744;
bool x7756 = x7674 <= 1;
int32_t x7757;
if (x7756) {
x7757 = 1;
} else {
x7757 = x7674;
}
int32_t x7758 = x7757 * x7757;
int32_t x7763;
if (x7744) {
x7763 = 0;
} else {
x7763 = x7674;
}
int32_t x7764;
if (x7744) {
x7764 = 0;
} else {
x7764 = 1;
}
float* x8 = x5+816064;
bool x7827 = x7757 == 1;
bool x7828 = x7827 || true;
bool x7829 = x7828 || x7827;
bool x7839 = x7757 <= 1;
int32_t x7840;
if (x7839) {
x7840 = 1;
} else {
x7840 = x7757;
}
int32_t x7841 = x7840 * x7840;
int32_t x7846;
if (x7827) {
x7846 = 0;
} else {
x7846 = x7757;
}
int32_t x7847;
if (x7827) {
x7847 = 0;
} else {
x7847 = 1;
}
float* x95 = x5+816192;
int32_t x7894 = x7840 - 1;
int32_t x7895 = x7894 / 1;
int32_t x7896 = x7895 + 1;
int32_t x7900 = 32768 * x7896;
int32_t x7901 = x7900 * x7896;
int32_t x7897 = x7896 * x7896;
int32_t x7898 = 512 * x7897;
float* x111 = x5+816576;
bool x7975 = x7896 == 1;
bool x7976 = x7975 || true;
bool x7977 = x7976 || x7975;
bool x7987 = x7896 <= 1;
int32_t x7988;
if (x7987) {
x7988 = 1;
} else {
x7988 = x7896;
}
int32_t x7989 = x7988 * x7988;
int32_t x7993;
if (x7975) {
x7993 = 0;
} else {
x7993 = x7896;
}
int32_t x7994;
if (x7975) {
x7994 = 0;
} else {
x7994 = 1;
}
float* x147 = x5+883136;
float* x88 = x5+883648;
bool x8073 = x7988 == 1;
bool x8074 = x8073 || true;
bool x8075 = x8074 || x8073;
bool x8085 = x7988 <= 1;
int32_t x8086;
if (x8085) {
x8086 = 1;
} else {
x8086 = x7988;
}
int32_t x8087 = x8086 * x8086;
int32_t x8092;
if (x8073) {
x8092 = 0;
} else {
x8092 = x7988;
}
int32_t x8093;
if (x8073) {
x8093 = 0;
} else {
x8093 = 1;
}
bool x8156 = x8086 == 1;
bool x8157 = x8156 || true;
bool x8158 = x8157 || x8156;
bool x8168 = x8086 <= 1;
int32_t x8169;
if (x8168) {
x8169 = 1;
} else {
x8169 = x8086;
}
int32_t x8170 = x8169 * x8169;
int32_t x8175;
if (x8156) {
x8175 = 0;
} else {
x8175 = x8086;
}
int32_t x8176;
if (x8156) {
x8176 = 0;
} else {
x8176 = 1;
}
float* x52 = x5+882112;
bool x8239 = x8169 == 1;
bool x8240 = x8239 || true;
bool x8241 = x8240 || x8239;
bool x8251 = x8169 <= 1;
int32_t x8252;
if (x8251) {
x8252 = 1;
} else {
x8252 = x8169;
}
int32_t x8253 = x8252 * x8252;
int32_t x8258;
if (x8239) {
x8258 = 0;
} else {
x8258 = x8169;
}
int32_t x8259;
if (x8239) {
x8259 = 0;
} else {
x8259 = 1;
}
float* x246 = x5+882624;
bool x8297 = x8252 == 1;
bool x8298 = x8297 || x6951;
bool x8299 = x8252 == x6501;
bool x8300 = x8298 || x8299;
bool x8310 = x8252 <= x6501;
int32_t x8311;
if (x8310) {
x8311 = x6501;
} else {
x8311 = x8252;
}
int32_t x8316;
if (x8297) {
x8316 = 0;
} else {
x8316 = x8252;
}
int32_t x8317;
if (x8297) {
x8317 = 0;
} else {
x8317 = 1;
}
int32_t x8363 = x8252 - 1;
int32_t x8364 = x8363 / 1;
int32_t x8365 = x8364 + 1;
int32_t x8369 = 8192 * x8365;
int32_t x8370 = x8369 * x8365;
int32_t x8366 = x8365 * x8365;
int32_t x8367 = 128 * x8366;
float* x196 = x5+884160;
bool x8444 = x8365 == 1;
bool x8445 = x8444 || true;
bool x8446 = x8445 || x8444;
bool x8456 = x8365 <= 1;
int32_t x8457;
if (x8456) {
x8457 = 1;
} else {
x8457 = x8365;
}
int32_t x8458 = x8457 * x8457;
int32_t x8462;
if (x8444) {
x8462 = 0;
} else {
x8462 = x8365;
}
int32_t x8463;
if (x8444) {
x8463 = 0;
} else {
x8463 = 1;
}
float* x112 = x5+949952;
float* x9 = x5+950080;
bool x8542 = x8457 == 1;
bool x8543 = x8542 || true;
bool x8544 = x8543 || x8542;
bool x8554 = x8457 <= 1;
int32_t x8555;
if (x8554) {
x8555 = 1;
} else {
x8555 = x8457;
}
int32_t x8556 = x8555 * x8555;
int32_t x8561;
if (x8542) {
x8561 = 0;
} else {
x8561 = x8457;
}
int32_t x8562;
if (x8542) {
x8562 = 0;
} else {
x8562 = 1;
}
bool x8625 = x8555 == 1;
bool x8626 = x8625 || true;
bool x8627 = x8626 || x8625;
bool x8637 = x8555 <= 1;
int32_t x8638;
if (x8637) {
x8638 = 1;
} else {
x8638 = x8555;
}
int32_t x8639 = x8638 * x8638;
int32_t x8644;
if (x8625) {
x8644 = 0;
} else {
x8644 = x8555;
}
int32_t x8645;
if (x8625) {
x8645 = 0;
} else {
x8645 = 1;
}
float* x45 = x5+949696;
bool x8708 = x8638 == 1;
bool x8709 = x8708 || true;
bool x8710 = x8709 || x8708;
bool x8720 = x8638 <= 1;
int32_t x8721;
if (x8720) {
x8721 = 1;
} else {
x8721 = x8638;
}
int32_t x8722 = x8721 * x8721;
int32_t x8727;
if (x8708) {
x8727 = 0;
} else {
x8727 = x8638;
}
int32_t x8728;
if (x8708) {
x8728 = 0;
} else {
x8728 = 1;
}
float* x170 = x5+949824;
int32_t x8775 = x8721 + 2;
int32_t x8776 = x8775 - 3;
int32_t x8777 = x8776 / 1;
int32_t x8778 = x8777 + 1;
int32_t x8782 = 8192 * x8778;
int32_t x8783 = x8782 * x8778;
int32_t x8779 = x8778 * x8778;
int32_t x8780 = 128 * x8779;
float* x191 = x5+950208;
bool x8905 = x8778 == 1;
bool x8906 = x8905 || true;
bool x8907 = x8906 || x8905;
bool x8917 = x8778 <= 1;
int32_t x8918;
if (x8917) {
x8918 = 1;
} else {
x8918 = x8778;
}
int32_t x8919 = x8918 * x8918;
int32_t x8923;
if (x8905) {
x8923 = 0;
} else {
x8923 = x8778;
}
int32_t x8924;
if (x8905) {
x8924 = 0;
} else {
x8924 = 1;
}
float* x217 = x5+1097920;
float* x266 = x5+1098048;
bool x9003 = x8918 == 1;
bool x9004 = x9003 || true;
bool x9005 = x9004 || x9003;
bool x9015 = x8918 <= 1;
int32_t x9016;
if (x9015) {
x9016 = 1;
} else {
x9016 = x8918;
}
int32_t x9017 = x9016 * x9016;
int32_t x9022;
if (x9003) {
x9022 = 0;
} else {
x9022 = x8918;
}
int32_t x9023;
if (x9003) {
x9023 = 0;
} else {
x9023 = 1;
}
bool x9086 = x9016 == 1;
bool x9087 = x9086 || true;
bool x9088 = x9087 || x9086;
bool x9098 = x9016 <= 1;
int32_t x9099;
if (x9098) {
x9099 = 1;
} else {
x9099 = x9016;
}
int32_t x9100 = x9099 * x9099;
int32_t x9105;
if (x9086) {
x9105 = 0;
} else {
x9105 = x9016;
}
int32_t x9106;
if (x9086) {
x9106 = 0;
} else {
x9106 = 1;
}
float* x127 = x5+1097664;
bool x9169 = x9099 == 1;
bool x9170 = x9169 || true;
bool x9171 = x9170 || x9169;
bool x9181 = x9099 <= 1;
int32_t x9182;
if (x9181) {
x9182 = 1;
} else {
x9182 = x9099;
}
int32_t x9183 = x9182 * x9182;
int32_t x9188;
if (x9169) {
x9188 = 0;
} else {
x9188 = x9099;
}
int32_t x9189;
if (x9169) {
x9189 = 0;
} else {
x9189 = 1;
}
float* x61 = x5+1097792;
int32_t x9236 = x9182 - 1;
int32_t x9237 = x9236 / 1;
int32_t x9238 = x9237 + 1;
int32_t x9242 = 32768 * x9238;
int32_t x9243 = x9242 * x9238;
int32_t x9239 = x9238 * x9238;
int32_t x9240 = 512 * x9239;
float* x41 = x5+1098176;
bool x9317 = x9238 == 1;
bool x9318 = x9317 || true;
bool x9319 = x9318 || x9317;
bool x9329 = x9238 <= 1;
int32_t x9330;
if (x9329) {
x9330 = 1;
} else {
x9330 = x9238;
}
int32_t x9331 = x9330 * x9330;
int32_t x9335;
if (x9317) {
x9335 = 0;
} else {
x9335 = x9238;
}
int32_t x9336;
if (x9317) {
x9336 = 0;
} else {
x9336 = 1;
}
float* x25 = x5+1164736;
float* x223 = x5+1165248;
bool x9415 = x9330 == 1;
bool x9416 = x9415 || true;
bool x9417 = x9416 || x9415;
bool x9427 = x9330 <= 1;
int32_t x9428;
if (x9427) {
x9428 = 1;
} else {
x9428 = x9330;
}
int32_t x9429 = x9428 * x9428;
int32_t x9434;
if (x9415) {
x9434 = 0;
} else {
x9434 = x9330;
}
int32_t x9435;
if (x9415) {
x9435 = 0;
} else {
x9435 = 1;
}
bool x9498 = x9428 == 1;
bool x9499 = x9498 || true;
bool x9500 = x9499 || x9498;
bool x9510 = x9428 <= 1;
int32_t x9511;
if (x9510) {
x9511 = 1;
} else {
x9511 = x9428;
}
int32_t x9512 = x9511 * x9511;
int32_t x9517;
if (x9498) {
x9517 = 0;
} else {
x9517 = x9428;
}
int32_t x9518;
if (x9498) {
x9518 = 0;
} else {
x9518 = 1;
}
float* x167 = x5+1163712;
bool x9581 = x9511 == 1;
bool x9582 = x9581 || true;
bool x9583 = x9582 || x9581;
bool x9593 = x9511 <= 1;
int32_t x9594;
if (x9593) {
x9594 = 1;
} else {
x9594 = x9511;
}
int32_t x9595 = x9594 * x9594;
int32_t x9600;
if (x9581) {
x9600 = 0;
} else {
x9600 = x9511;
}
int32_t x9601;
if (x9581) {
x9601 = 0;
} else {
x9601 = 1;
}
float* x82 = x5+1164224;
bool x9639 = x9594 == 1;
bool x9640 = x9639 || x8297;
bool x9641 = x9594 == x8252;
bool x9642 = x9640 || x9641;
bool x9652 = x9594 <= x8252;
int32_t x9653;
if (x9652) {
x9653 = x8252;
} else {
x9653 = x9594;
}
int32_t x9658;
if (x9639) {
x9658 = 0;
} else {
x9658 = x9594;
}
int32_t x9659;
if (x9639) {
x9659 = 0;
} else {
x9659 = 1;
}
int32_t x9705 = x9594 - 1;
int32_t x9706 = x9705 / 1;
int32_t x9707 = x9706 + 1;
int32_t x9711 = 8192 * x9707;
int32_t x9712 = x9711 * x9707;
int32_t x9708 = x9707 * x9707;
int32_t x9709 = 128 * x9708;
float* x132 = x5+1165760;
bool x9786 = x9707 == 1;
bool x9787 = x9786 || true;
bool x9788 = x9787 || x9786;
bool x9798 = x9707 <= 1;
int32_t x9799;
if (x9798) {
x9799 = 1;
} else {
x9799 = x9707;
}
int32_t x9800 = x9799 * x9799;
int32_t x9804;
if (x9786) {
x9804 = 0;
} else {
x9804 = x9707;
}
int32_t x9805;
if (x9786) {
x9805 = 0;
} else {
x9805 = 1;
}
float* x236 = x5+1231552;
float* x261 = x5+1231680;
bool x9884 = x9799 == 1;
bool x9885 = x9884 || true;
bool x9886 = x9885 || x9884;
bool x9896 = x9799 <= 1;
int32_t x9897;
if (x9896) {
x9897 = 1;
} else {
x9897 = x9799;
}
int32_t x9898 = x9897 * x9897;
int32_t x9903;
if (x9884) {
x9903 = 0;
} else {
x9903 = x9799;
}
int32_t x9904;
if (x9884) {
x9904 = 0;
} else {
x9904 = 1;
}
bool x9967 = x9897 == 1;
bool x9968 = x9967 || true;
bool x9969 = x9968 || x9967;
bool x9979 = x9897 <= 1;
int32_t x9980;
if (x9979) {
x9980 = 1;
} else {
x9980 = x9897;
}
int32_t x9981 = x9980 * x9980;
int32_t x9986;
if (x9967) {
x9986 = 0;
} else {
x9986 = x9897;
}
int32_t x9987;
if (x9967) {
x9987 = 0;
} else {
x9987 = 1;
}
float* x39 = x5+1231296;
bool x10050 = x9980 == 1;
bool x10051 = x10050 || true;
bool x10052 = x10051 || x10050;
bool x10062 = x9980 <= 1;
int32_t x10063;
if (x10062) {
x10063 = 1;
} else {
x10063 = x9980;
}
int32_t x10064 = x10063 * x10063;
int32_t x10069;
if (x10050) {
x10069 = 0;
} else {
x10069 = x9980;
}
int32_t x10070;
if (x10050) {
x10070 = 0;
} else {
x10070 = 1;
}
float* x242 = x5+1231424;
int32_t x10117 = x10063 + 2;
int32_t x10118 = x10117 - 3;
int32_t x10119 = x10118 / 1;
int32_t x10120 = x10119 + 1;
int32_t x10124 = 8192 * x10120;
int32_t x10125 = x10124 * x10120;
int32_t x10121 = x10120 * x10120;
int32_t x10122 = 128 * x10121;
float* x165 = x5+1231808;
bool x10247 = x10120 == 1;
bool x10248 = x10247 || true;
bool x10249 = x10248 || x10247;
bool x10259 = x10120 <= 1;
int32_t x10260;
if (x10259) {
x10260 = 1;
} else {
x10260 = x10120;
}
int32_t x10261 = x10260 * x10260;
int32_t x10265;
if (x10247) {
x10265 = 0;
} else {
x10265 = x10120;
}
int32_t x10266;
if (x10247) {
x10266 = 0;
} else {
x10266 = 1;
}
float* x268 = x5+1379520;
float* x148 = x5+1379648;
bool x10345 = x10260 == 1;
bool x10346 = x10345 || true;
bool x10347 = x10346 || x10345;
bool x10357 = x10260 <= 1;
int32_t x10358;
if (x10357) {
x10358 = 1;
} else {
x10358 = x10260;
}
int32_t x10359 = x10358 * x10358;
int32_t x10364;
if (x10345) {
x10364 = 0;
} else {
x10364 = x10260;
}
int32_t x10365;
if (x10345) {
x10365 = 0;
} else {
x10365 = 1;
}
bool x10428 = x10358 == 1;
bool x10429 = x10428 || true;
bool x10430 = x10429 || x10428;
bool x10440 = x10358 <= 1;
int32_t x10441;
if (x10440) {
x10441 = 1;
} else {
x10441 = x10358;
}
int32_t x10442 = x10441 * x10441;
int32_t x10447;
if (x10428) {
x10447 = 0;
} else {
x10447 = x10358;
}
int32_t x10448;
if (x10428) {
x10448 = 0;
} else {
x10448 = 1;
}
float* x79 = x5+1379264;
bool x10511 = x10441 == 1;
bool x10512 = x10511 || true;
bool x10513 = x10512 || x10511;
bool x10523 = x10441 <= 1;
int32_t x10524;
if (x10523) {
x10524 = 1;
} else {
x10524 = x10441;
}
int32_t x10525 = x10524 * x10524;
int32_t x10530;
if (x10511) {
x10530 = 0;
} else {
x10530 = x10441;
}
int32_t x10531;
if (x10511) {
x10531 = 0;
} else {
x10531 = 1;
}
float* x38 = x5+1379392;
int32_t x10578 = x10524 - 1;
int32_t x10579 = x10578 / 1;
int32_t x10580 = x10579 + 1;
int32_t x10584 = 32768 * x10580;
int32_t x10585 = x10584 * x10580;
int32_t x10581 = x10580 * x10580;
int32_t x10582 = 512 * x10581;
float* x55 = x5+1379776;
bool x10659 = x10580 == 1;
bool x10660 = x10659 || true;
bool x10661 = x10660 || x10659;
bool x10671 = x10580 <= 1;
int32_t x10672;
if (x10671) {
x10672 = 1;
} else {
x10672 = x10580;
}
int32_t x10673 = x10672 * x10672;
int32_t x10677;
if (x10659) {
x10677 = 0;
} else {
x10677 = x10580;
}
int32_t x10678;
if (x10659) {
x10678 = 0;
} else {
x10678 = 1;
}
float* x19 = x5+1446336;
float* x234 = x5+1446848;
bool x10757 = x10672 == 1;
bool x10758 = x10757 || true;
bool x10759 = x10758 || x10757;
bool x10769 = x10672 <= 1;
int32_t x10770;
if (x10769) {
x10770 = 1;
} else {
x10770 = x10672;
}
int32_t x10771 = x10770 * x10770;
int32_t x10776;
if (x10757) {
x10776 = 0;
} else {
x10776 = x10672;
}
int32_t x10777;
if (x10757) {
x10777 = 0;
} else {
x10777 = 1;
}
bool x10840 = x10770 == 1;
bool x10841 = x10840 || true;
bool x10842 = x10841 || x10840;
bool x10852 = x10770 <= 1;
int32_t x10853;
if (x10852) {
x10853 = 1;
} else {
x10853 = x10770;
}
int32_t x10854 = x10853 * x10853;
int32_t x10859;
if (x10840) {
x10859 = 0;
} else {
x10859 = x10770;
}
int32_t x10860;
if (x10840) {
x10860 = 0;
} else {
x10860 = 1;
}
float* x156 = x5+1445312;
bool x10923 = x10853 == 1;
bool x10924 = x10923 || true;
bool x10925 = x10924 || x10923;
bool x10935 = x10853 <= 1;
int32_t x10936;
if (x10935) {
x10936 = 1;
} else {
x10936 = x10853;
}
int32_t x10937 = x10936 * x10936;
int32_t x10942;
if (x10923) {
x10942 = 0;
} else {
x10942 = x10853;
}
int32_t x10943;
if (x10923) {
x10943 = 0;
} else {
x10943 = 1;
}
float* x54 = x5+1445824;
bool x10981 = x10936 == 1;
bool x10982 = x10981 || x9639;
bool x10983 = x10936 == x9594;
bool x10984 = x10982 || x10983;
bool x10994 = x10936 <= x9594;
int32_t x10995;
if (x10994) {
x10995 = x9594;
} else {
x10995 = x10936;
}
int32_t x11000;
if (x10981) {
x11000 = 0;
} else {
x11000 = x10936;
}
int32_t x11001;
if (x10981) {
x11001 = 0;
} else {
x11001 = 1;
}
int32_t x11047 = x10936 - 1;
int32_t x11048 = x11047 / 1;
int32_t x11049 = x11048 + 1;
int32_t x11053 = 16384 * x11049;
int32_t x11054 = x11053 * x11049;
int32_t x11050 = x11049 * x11049;
int32_t x11051 = 256 * x11050;
float* x180 = x5+1447360;
bool x11128 = x11049 == 1;
bool x11129 = x11128 || true;
bool x11130 = x11129 || x11128;
bool x11140 = x11049 <= 1;
int32_t x11141;
if (x11140) {
x11141 = 1;
} else {
x11141 = x11049;
}
int32_t x11142 = x11141 * x11141;
int32_t x11146;
if (x11128) {
x11146 = 0;
} else {
x11146 = x11049;
}
int32_t x11147;
if (x11128) {
x11147 = 0;
} else {
x11147 = 1;
}
float* x131 = x5+1578944;
float* x198 = x5+1579200;
bool x11226 = x11141 == 1;
bool x11227 = x11226 || true;
bool x11228 = x11227 || x11226;
bool x11238 = x11141 <= 1;
int32_t x11239;
if (x11238) {
x11239 = 1;
} else {
x11239 = x11141;
}
int32_t x11240 = x11239 * x11239;
int32_t x11245;
if (x11226) {
x11245 = 0;
} else {
x11245 = x11141;
}
int32_t x11246;
if (x11226) {
x11246 = 0;
} else {
x11246 = 1;
}
bool x11309 = x11239 == 1;
bool x11310 = x11309 || true;
bool x11311 = x11310 || x11309;
bool x11321 = x11239 <= 1;
int32_t x11322;
if (x11321) {
x11322 = 1;
} else {
x11322 = x11239;
}
int32_t x11323 = x11322 * x11322;
int32_t x11328;
if (x11309) {
x11328 = 0;
} else {
x11328 = x11239;
}
int32_t x11329;
if (x11309) {
x11329 = 0;
} else {
x11329 = 1;
}
float* x270 = x5+1578432;
bool x11392 = x11322 == 1;
bool x11393 = x11392 || true;
bool x11394 = x11393 || x11392;
bool x11404 = x11322 <= 1;
int32_t x11405;
if (x11404) {
x11405 = 1;
} else {
x11405 = x11322;
}
int32_t x11406 = x11405 * x11405;
int32_t x11411;
if (x11392) {
x11411 = 0;
} else {
x11411 = x11322;
}
int32_t x11412;
if (x11392) {
x11412 = 0;
} else {
x11412 = 1;
}
float* x21 = x5+1578688;
int32_t x11459 = x11405 + 2;
int32_t x11460 = x11459 - 3;
int32_t x11461 = x11460 / 2;
int32_t x11462 = x11461 + 1;
int32_t x11466 = 16384 * x11462;
int32_t x11467 = x11466 * x11462;
int32_t x11463 = x11462 * x11462;
int32_t x11464 = 256 * x11463;
float* x175 = x5+1579456;
bool x11573 = x11462 == 1;
bool x11574 = x11573 || true;
bool x11575 = x11574 || x11573;
bool x11585 = x11462 <= 1;
int32_t x11586;
if (x11585) {
x11586 = 1;
} else {
x11586 = x11462;
}
int32_t x11587 = x11586 * x11586;
int32_t x11591;
if (x11573) {
x11591 = 0;
} else {
x11591 = x11462;
}
int32_t x11592;
if (x11573) {
x11592 = 0;
} else {
x11592 = 1;
}
float* x229 = x5+2169792;
float* x99 = x5+2170048;
bool x11671 = x11586 == 1;
bool x11672 = x11671 || true;
bool x11673 = x11672 || x11671;
bool x11683 = x11586 <= 1;
int32_t x11684;
if (x11683) {
x11684 = 1;
} else {
x11684 = x11586;
}
int32_t x11685 = x11684 * x11684;
int32_t x11690;
if (x11671) {
x11690 = 0;
} else {
x11690 = x11586;
}
int32_t x11691;
if (x11671) {
x11691 = 0;
} else {
x11691 = 1;
}
bool x11754 = x11684 == 1;
bool x11755 = x11754 || true;
bool x11756 = x11755 || x11754;
bool x11766 = x11684 <= 1;
int32_t x11767;
if (x11766) {
x11767 = 1;
} else {
x11767 = x11684;
}
int32_t x11768 = x11767 * x11767;
int32_t x11773;
if (x11754) {
x11773 = 0;
} else {
x11773 = x11684;
}
int32_t x11774;
if (x11754) {
x11774 = 0;
} else {
x11774 = 1;
}
float* x108 = x5+2169280;
bool x11837 = x11767 == 1;
bool x11838 = x11837 || true;
bool x11839 = x11838 || x11837;
bool x11849 = x11767 <= 1;
int32_t x11850;
if (x11849) {
x11850 = 1;
} else {
x11850 = x11767;
}
int32_t x11851 = x11850 * x11850;
int32_t x11856;
if (x11837) {
x11856 = 0;
} else {
x11856 = x11767;
}
int32_t x11857;
if (x11837) {
x11857 = 0;
} else {
x11857 = 1;
}
float* x16 = x5+2169536;
int32_t x11904 = x11850 - 1;
int32_t x11905 = x11904 / 1;
int32_t x11906 = x11905 + 1;
int32_t x11910 = 65536 * x11906;
int32_t x11911 = x11910 * x11906;
int32_t x11907 = x11906 * x11906;
int32_t x11908 = 1024 * x11907;
float* x269 = x5+2170304;
bool x11985 = x11906 == 1;
bool x11986 = x11985 || true;
bool x11987 = x11986 || x11985;
bool x11997 = x11906 <= 1;
int32_t x11998;
if (x11997) {
x11998 = 1;
} else {
x11998 = x11906;
}
int32_t x11999 = x11998 * x11998;
int32_t x12003;
if (x11985) {
x12003 = 0;
} else {
x12003 = x11906;
}
int32_t x12004;
if (x11985) {
x12004 = 0;
} else {
x12004 = 1;
}
float* x216 = x5+2434496;
float* x267 = x5+2435520;
bool x12084 = x11998 == 1;
bool x12085 = x12084 || true;
bool x12086 = x12085 || x12084;
bool x12096 = x11998 <= 1;
int32_t x12097;
if (x12096) {
x12097 = 1;
} else {
x12097 = x11998;
}
int32_t x12098 = x12097 * x12097;
int32_t x12103;
if (x12084) {
x12103 = 0;
} else {
x12103 = x11998;
}
int32_t x12104;
if (x12084) {
x12104 = 0;
} else {
x12104 = 1;
}
bool x12167 = x12097 == 1;
bool x12168 = x12167 || true;
bool x12169 = x12168 || x12167;
bool x12179 = x12097 <= 1;
int32_t x12180;
if (x12179) {
x12180 = 1;
} else {
x12180 = x12097;
}
int32_t x12181 = x12180 * x12180;
int32_t x12186;
if (x12167) {
x12186 = 0;
} else {
x12186 = x12097;
}
int32_t x12187;
if (x12167) {
x12187 = 0;
} else {
x12187 = 1;
}
float* x18 = x5+2432448;
bool x12250 = x12180 == 1;
bool x12251 = x12250 || true;
bool x12252 = x12251 || x12250;
bool x12262 = x12180 <= 1;
int32_t x12263;
if (x12262) {
x12263 = 1;
} else {
x12263 = x12180;
}
int32_t x12264 = x12263 * x12263;
int32_t x12269;
if (x12250) {
x12269 = 0;
} else {
x12269 = x12180;
}
int32_t x12270;
if (x12250) {
x12270 = 0;
} else {
x12270 = 1;
}
float* x117 = x5+2433472;
int32_t x12304 = x11047 / 2;
int32_t x12305 = x12304 + 1;
int32_t x12309 = 65536 * x12305;
int32_t x12310 = x12309 * x12305;
int32_t x12306 = x12305 * x12305;
int32_t x12307 = 1024 * x12306;
float* x75 = x5+2436544;
bool x12390 = x12305 == 1;
bool x12391 = x12390 || true;
bool x12392 = x12391 || x12390;
bool x12402 = x12305 <= 1;
int32_t x12403;
if (x12402) {
x12403 = 1;
} else {
x12403 = x12305;
}
int32_t x12404 = x12403 * x12403;
int32_t x12408;
if (x12390) {
x12408 = 0;
} else {
x12408 = x12305;
}
int32_t x12409;
if (x12390) {
x12409 = 0;
} else {
x12409 = 1;
}
float* x86 = x5+2962880;
float* x211 = x5+2963904;
bool x12488 = x12403 == 1;
bool x12489 = x12488 || true;
bool x12490 = x12489 || x12488;
bool x12500 = x12403 <= 1;
int32_t x12501;
if (x12500) {
x12501 = 1;
} else {
x12501 = x12403;
}
int32_t x12502 = x12501 * x12501;
int32_t x12507;
if (x12488) {
x12507 = 0;
} else {
x12507 = x12403;
}
int32_t x12508;
if (x12488) {
x12508 = 0;
} else {
x12508 = 1;
}
bool x12571 = x12501 == 1;
bool x12572 = x12571 || true;
bool x12573 = x12572 || x12571;
bool x12583 = x12501 <= 1;
int32_t x12584;
if (x12583) {
x12584 = 1;
} else {
x12584 = x12501;
}
int32_t x12585 = x12584 * x12584;
int32_t x12590;
if (x12571) {
x12590 = 0;
} else {
x12590 = x12501;
}
int32_t x12591;
if (x12571) {
x12591 = 0;
} else {
x12591 = 1;
}
float* x29 = x5+2960832;
bool x12654 = x12584 == 1;
bool x12655 = x12654 || true;
bool x12656 = x12655 || x12654;
bool x12666 = x12584 <= 1;
int32_t x12667;
if (x12666) {
x12667 = 1;
} else {
x12667 = x12584;
}
int32_t x12668 = x12667 * x12667;
int32_t x12673;
if (x12654) {
x12673 = 0;
} else {
x12673 = x12584;
}
int32_t x12674;
if (x12654) {
x12674 = 0;
} else {
x12674 = 1;
}
float* x220 = x5+2961856;
bool x12713 = x12263 == 1;
bool x12714 = x12667 == 1;
bool x12715 = x12713 || x12714;
bool x12716 = x12263 == x12667;
bool x12717 = x12715 || x12716;
bool x12727 = x12263 <= x12667;
int32_t x12728;
if (x12727) {
x12728 = x12667;
} else {
x12728 = x12263;
}
int32_t x12733;
if (x12713) {
x12733 = 0;
} else {
x12733 = x12263;
}
int32_t x12734;
if (x12713) {
x12734 = 0;
} else {
x12734 = 1;
}
int32_t x12736;
if (x12714) {
x12736 = 0;
} else {
x12736 = x12667;
}
int32_t x12737;
if (x12714) {
x12737 = 0;
} else {
x12737 = 1;
}
int32_t x12783 = x12263 - 1;
int32_t x12784 = x12783 / 1;
int32_t x12785 = x12784 + 1;
int32_t x12789 = 16384 * x12785;
int32_t x12790 = x12789 * x12785;
int32_t x12786 = x12785 * x12785;
int32_t x12787 = 256 * x12786;
float* x13 = x5+2964928;
bool x12864 = x12785 == 1;
bool x12865 = x12864 || true;
bool x12866 = x12865 || x12864;
bool x12876 = x12785 <= 1;
int32_t x12877;
if (x12876) {
x12877 = 1;
} else {
x12877 = x12785;
}
int32_t x12878 = x12877 * x12877;
int32_t x12882;
if (x12864) {
x12882 = 0;
} else {
x12882 = x12785;
}
int32_t x12883;
if (x12864) {
x12883 = 0;
} else {
x12883 = 1;
}
float* x259 = x5+3227584;
float* x157 = x5+3227840;
bool x12962 = x12877 == 1;
bool x12963 = x12962 || true;
bool x12964 = x12963 || x12962;
bool x12974 = x12877 <= 1;
int32_t x12975;
if (x12974) {
x12975 = 1;
} else {
x12975 = x12877;
}
int32_t x12976 = x12975 * x12975;
int32_t x12981;
if (x12962) {
x12981 = 0;
} else {
x12981 = x12877;
}
int32_t x12982;
if (x12962) {
x12982 = 0;
} else {
x12982 = 1;
}
bool x13045 = x12975 == 1;
bool x13046 = x13045 || true;
bool x13047 = x13046 || x13045;
bool x13057 = x12975 <= 1;
int32_t x13058;
if (x13057) {
x13058 = 1;
} else {
x13058 = x12975;
}
int32_t x13059 = x13058 * x13058;
int32_t x13064;
if (x13045) {
x13064 = 0;
} else {
x13064 = x12975;
}
int32_t x13065;
if (x13045) {
x13065 = 0;
} else {
x13065 = 1;
}
float* x30 = x5+3227072;
bool x13128 = x13058 == 1;
bool x13129 = x13128 || true;
bool x13130 = x13129 || x13128;
bool x13140 = x13058 <= 1;
int32_t x13141;
if (x13140) {
x13141 = 1;
} else {
x13141 = x13058;
}
int32_t x13142 = x13141 * x13141;
int32_t x13147;
if (x13128) {
x13147 = 0;
} else {
x13147 = x13058;
}
int32_t x13148;
if (x13128) {
x13148 = 0;
} else {
x13148 = 1;
}
float* x219 = x5+3227328;
int32_t x13195 = x13141 + 2;
int32_t x13196 = x13195 - 3;
int32_t x13197 = x13196 / 1;
int32_t x13198 = x13197 + 1;
int32_t x13202 = 16384 * x13198;
int32_t x13203 = x13202 * x13198;
int32_t x13199 = x13198 * x13198;
int32_t x13200 = 256 * x13199;
float* x31 = x5+3228096;
bool x13325 = x13198 == 1;
bool x13326 = x13325 || true;
bool x13327 = x13326 || x13325;
bool x13337 = x13198 <= 1;
int32_t x13338;
if (x13337) {
x13338 = 1;
} else {
x13338 = x13198;
}
int32_t x13339 = x13338 * x13338;
int32_t x13343;
if (x13325) {
x13343 = 0;
} else {
x13343 = x13198;
}
int32_t x13344;
if (x13325) {
x13344 = 0;
} else {
x13344 = 1;
}
float* x200 = x5+3818432;
float* x237 = x5+3818688;
bool x13423 = x13338 == 1;
bool x13424 = x13423 || true;
bool x13425 = x13424 || x13423;
bool x13435 = x13338 <= 1;
int32_t x13436;
if (x13435) {
x13436 = 1;
} else {
x13436 = x13338;
}
int32_t x13437 = x13436 * x13436;
int32_t x13442;
if (x13423) {
x13442 = 0;
} else {
x13442 = x13338;
}
int32_t x13443;
if (x13423) {
x13443 = 0;
} else {
x13443 = 1;
}
bool x13506 = x13436 == 1;
bool x13507 = x13506 || true;
bool x13508 = x13507 || x13506;
bool x13518 = x13436 <= 1;
int32_t x13519;
if (x13518) {
x13519 = 1;
} else {
x13519 = x13436;
}
int32_t x13520 = x13519 * x13519;
int32_t x13525;
if (x13506) {
x13525 = 0;
} else {
x13525 = x13436;
}
int32_t x13526;
if (x13506) {
x13526 = 0;
} else {
x13526 = 1;
}
float* x271 = x5+3817920;
bool x13589 = x13519 == 1;
bool x13590 = x13589 || true;
bool x13591 = x13590 || x13589;
bool x13601 = x13519 <= 1;
int32_t x13602;
if (x13601) {
x13602 = 1;
} else {
x13602 = x13519;
}
int32_t x13603 = x13602 * x13602;
int32_t x13608;
if (x13589) {
x13608 = 0;
} else {
x13608 = x13519;
}
int32_t x13609;
if (x13589) {
x13609 = 0;
} else {
x13609 = 1;
}
float* x96 = x5+3818176;
int32_t x13656 = x13602 - 1;
int32_t x13657 = x13656 / 1;
int32_t x13658 = x13657 + 1;
int32_t x13662 = 65536 * x13658;
int32_t x13663 = x13662 * x13658;
int32_t x13659 = x13658 * x13658;
int32_t x13660 = 1024 * x13659;
float* x56 = x5+3818944;
bool x13737 = x13658 == 1;
bool x13738 = x13737 || true;
bool x13739 = x13738 || x13737;
bool x13749 = x13658 <= 1;
int32_t x13750;
if (x13749) {
x13750 = 1;
} else {
x13750 = x13658;
}
int32_t x13751 = x13750 * x13750;
int32_t x13755;
if (x13737) {
x13755 = 0;
} else {
x13755 = x13658;
}
int32_t x13756;
if (x13737) {
x13756 = 0;
} else {
x13756 = 1;
}
float* x182 = x5+4083136;
float* x143 = x5+4084160;
bool x13835 = x13750 == 1;
bool x13836 = x13835 || true;
bool x13837 = x13836 || x13835;
bool x13847 = x13750 <= 1;
int32_t x13848;
if (x13847) {
x13848 = 1;
} else {
x13848 = x13750;
}
int32_t x13849 = x13848 * x13848;
int32_t x13854;
if (x13835) {
x13854 = 0;
} else {
x13854 = x13750;
}
int32_t x13855;
if (x13835) {
x13855 = 0;
} else {
x13855 = 1;
}
bool x13918 = x13848 == 1;
bool x13919 = x13918 || true;
bool x13920 = x13919 || x13918;
bool x13930 = x13848 <= 1;
int32_t x13931;
if (x13930) {
x13931 = 1;
} else {
x13931 = x13848;
}
int32_t x13932 = x13931 * x13931;
int32_t x13937;
if (x13918) {
x13937 = 0;
} else {
x13937 = x13848;
}
int32_t x13938;
if (x13918) {
x13938 = 0;
} else {
x13938 = 1;
}
float* x20 = x5+4081088;
bool x14001 = x13931 == 1;
bool x14002 = x14001 || true;
bool x14003 = x14002 || x14001;
bool x14013 = x13931 <= 1;
int32_t x14014;
if (x14013) {
x14014 = 1;
} else {
x14014 = x13931;
}
int32_t x14015 = x14014 * x14014;
int32_t x14020;
if (x14001) {
x14020 = 0;
} else {
x14020 = x13931;
}
int32_t x14021;
if (x14001) {
x14021 = 0;
} else {
x14021 = 1;
}
float* x232 = x5+4082112;
bool x14059 = x14014 == 1;
bool x14060 = x14059 || x12713;
bool x14061 = x14014 == x12263;
bool x14062 = x14060 || x14061;
bool x14072 = x14014 <= x12263;
int32_t x14073;
if (x14072) {
x14073 = x12263;
} else {
x14073 = x14014;
}
int32_t x14078;
if (x14059) {
x14078 = 0;
} else {
x14078 = x14014;
}
int32_t x14079;
if (x14059) {
x14079 = 0;
} else {
x14079 = 1;
}
int32_t x14125 = x14014 - 1;
int32_t x14126 = x14125 / 1;
int32_t x14127 = x14126 + 1;
int32_t x14131 = 16384 * x14127;
int32_t x14132 = x14131 * x14127;
int32_t x14128 = x14127 * x14127;
int32_t x14129 = 256 * x14128;
float* x218 = x5+4085184;
bool x14206 = x14127 == 1;
bool x14207 = x14206 || true;
bool x14208 = x14207 || x14206;
bool x14218 = x14127 <= 1;
int32_t x14219;
if (x14218) {
x14219 = 1;
} else {
x14219 = x14127;
}
int32_t x14220 = x14219 * x14219;
int32_t x14224;
if (x14206) {
x14224 = 0;
} else {
x14224 = x14127;
}
int32_t x14225;
if (x14206) {
x14225 = 0;
} else {
x14225 = 1;
}
float* x178 = x5+4347840;
float* x174 = x5+4348096;
bool x14304 = x14219 == 1;
bool x14305 = x14304 || true;
bool x14306 = x14305 || x14304;
bool x14316 = x14219 <= 1;
int32_t x14317;
if (x14316) {
x14317 = 1;
} else {
x14317 = x14219;
}
int32_t x14318 = x14317 * x14317;
int32_t x14323;
if (x14304) {
x14323 = 0;
} else {
x14323 = x14219;
}
int32_t x14324;
if (x14304) {
x14324 = 0;
} else {
x14324 = 1;
}
bool x14387 = x14317 == 1;
bool x14388 = x14387 || true;
bool x14389 = x14388 || x14387;
bool x14399 = x14317 <= 1;
int32_t x14400;
if (x14399) {
x14400 = 1;
} else {
x14400 = x14317;
}
int32_t x14401 = x14400 * x14400;
int32_t x14406;
if (x14387) {
x14406 = 0;
} else {
x14406 = x14317;
}
int32_t x14407;
if (x14387) {
x14407 = 0;
} else {
x14407 = 1;
}
float* x129 = x5+4347328;
bool x14470 = x14400 == 1;
bool x14471 = x14470 || true;
bool x14472 = x14471 || x14470;
bool x14482 = x14400 <= 1;
int32_t x14483;
if (x14482) {
x14483 = 1;
} else {
x14483 = x14400;
}
int32_t x14484 = x14483 * x14483;
int32_t x14489;
if (x14470) {
x14489 = 0;
} else {
x14489 = x14400;
}
int32_t x14490;
if (x14470) {
x14490 = 0;
} else {
x14490 = 1;
}
float* x197 = x5+4347584;
int32_t x14537 = x14483 + 2;
int32_t x14538 = x14537 - 3;
int32_t x14539 = x14538 / 1;
int32_t x14540 = x14539 + 1;
int32_t x14544 = 16384 * x14540;
int32_t x14545 = x14544 * x14540;
int32_t x14541 = x14540 * x14540;
int32_t x14542 = 256 * x14541;
float* x14 = x5+4348352;
bool x14667 = x14540 == 1;
bool x14668 = x14667 || true;
bool x14669 = x14668 || x14667;
bool x14679 = x14540 <= 1;
int32_t x14680;
if (x14679) {
x14680 = 1;
} else {
x14680 = x14540;
}
int32_t x14681 = x14680 * x14680;
int32_t x14685;
if (x14667) {
x14685 = 0;
} else {
x14685 = x14540;
}
int32_t x14686;
if (x14667) {
x14686 = 0;
} else {
x14686 = 1;
}
float* x124 = x5+4938688;
float* x63 = x5+4938944;
bool x14765 = x14680 == 1;
bool x14766 = x14765 || true;
bool x14767 = x14766 || x14765;
bool x14777 = x14680 <= 1;
int32_t x14778;
if (x14777) {
x14778 = 1;
} else {
x14778 = x14680;
}
int32_t x14779 = x14778 * x14778;
int32_t x14784;
if (x14765) {
x14784 = 0;
} else {
x14784 = x14680;
}
int32_t x14785;
if (x14765) {
x14785 = 0;
} else {
x14785 = 1;
}
bool x14848 = x14778 == 1;
bool x14849 = x14848 || true;
bool x14850 = x14849 || x14848;
bool x14860 = x14778 <= 1;
int32_t x14861;
if (x14860) {
x14861 = 1;
} else {
x14861 = x14778;
}
int32_t x14862 = x14861 * x14861;
int32_t x14867;
if (x14848) {
x14867 = 0;
} else {
x14867 = x14778;
}
int32_t x14868;
if (x14848) {
x14868 = 0;
} else {
x14868 = 1;
}
float* x228 = x5+4938176;
bool x14931 = x14861 == 1;
bool x14932 = x14931 || true;
bool x14933 = x14932 || x14931;
bool x14943 = x14861 <= 1;
int32_t x14944;
if (x14943) {
x14944 = 1;
} else {
x14944 = x14861;
}
int32_t x14945 = x14944 * x14944;
int32_t x14950;
if (x14931) {
x14950 = 0;
} else {
x14950 = x14861;
}
int32_t x14951;
if (x14931) {
x14951 = 0;
} else {
x14951 = 1;
}
float* x192 = x5+4938432;
int32_t x14998 = x14944 - 1;
int32_t x14999 = x14998 / 1;
int32_t x15000 = x14999 + 1;
int32_t x15004 = 65536 * x15000;
int32_t x15005 = x15004 * x15000;
int32_t x15001 = x15000 * x15000;
int32_t x15002 = 1024 * x15001;
float* x116 = x5+4939200;
bool x15079 = x15000 == 1;
bool x15080 = x15079 || true;
bool x15081 = x15080 || x15079;
bool x15091 = x15000 <= 1;
int32_t x15092;
if (x15091) {
x15092 = 1;
} else {
x15092 = x15000;
}
int32_t x15093 = x15092 * x15092;
int32_t x15097;
if (x15079) {
x15097 = 0;
} else {
x15097 = x15000;
}
int32_t x15098;
if (x15079) {
x15098 = 0;
} else {
x15098 = 1;
}
float* x140 = x5+5203392;
float* x188 = x5+5204416;
bool x15177 = x15092 == 1;
bool x15178 = x15177 || true;
bool x15179 = x15178 || x15177;
bool x15189 = x15092 <= 1;
int32_t x15190;
if (x15189) {
x15190 = 1;
} else {
x15190 = x15092;
}
int32_t x15191 = x15190 * x15190;
int32_t x15196;
if (x15177) {
x15196 = 0;
} else {
x15196 = x15092;
}
int32_t x15197;
if (x15177) {
x15197 = 0;
} else {
x15197 = 1;
}
bool x15260 = x15190 == 1;
bool x15261 = x15260 || true;
bool x15262 = x15261 || x15260;
bool x15272 = x15190 <= 1;
int32_t x15273;
if (x15272) {
x15273 = 1;
} else {
x15273 = x15190;
}
int32_t x15274 = x15273 * x15273;
int32_t x15279;
if (x15260) {
x15279 = 0;
} else {
x15279 = x15190;
}
int32_t x15280;
if (x15260) {
x15280 = 0;
} else {
x15280 = 1;
}
float* x263 = x5+5201344;
bool x15343 = x15273 == 1;
bool x15344 = x15343 || true;
bool x15345 = x15344 || x15343;
bool x15355 = x15273 <= 1;
int32_t x15356;
if (x15355) {
x15356 = 1;
} else {
x15356 = x15273;
}
int32_t x15357 = x15356 * x15356;
int32_t x15362;
if (x15343) {
x15362 = 0;
} else {
x15362 = x15273;
}
int32_t x15363;
if (x15343) {
x15363 = 0;
} else {
x15363 = 1;
}
float* x57 = x5+5202368;
bool x15401 = x15356 == 1;
bool x15402 = x15401 || x14059;
bool x15403 = x15356 == x14014;
bool x15404 = x15402 || x15403;
bool x15414 = x15356 <= x14014;
int32_t x15415;
if (x15414) {
x15415 = x14014;
} else {
x15415 = x15356;
}
int32_t x15420;
if (x15401) {
x15420 = 0;
} else {
x15420 = x15356;
}
int32_t x15421;
if (x15401) {
x15421 = 0;
} else {
x15421 = 1;
}
int32_t x15467 = x15356 - 1;
int32_t x15468 = x15467 / 1;
int32_t x15469 = x15468 + 1;
int32_t x15473 = 16384 * x15469;
int32_t x15474 = x15473 * x15469;
int32_t x15470 = x15469 * x15469;
int32_t x15471 = 256 * x15470;
float* x6 = x5+5205440;
bool x15548 = x15469 == 1;
bool x15549 = x15548 || true;
bool x15550 = x15549 || x15548;
bool x15560 = x15469 <= 1;
int32_t x15561;
if (x15560) {
x15561 = 1;
} else {
x15561 = x15469;
}
int32_t x15562 = x15561 * x15561;
int32_t x15566;
if (x15548) {
x15566 = 0;
} else {
x15566 = x15469;
}
int32_t x15567;
if (x15548) {
x15567 = 0;
} else {
x15567 = 1;
}
float* x163 = x5+5468096;
float* x98 = x5+5468352;
bool x15646 = x15561 == 1;
bool x15647 = x15646 || true;
bool x15648 = x15647 || x15646;
bool x15658 = x15561 <= 1;
int32_t x15659;
if (x15658) {
x15659 = 1;
} else {
x15659 = x15561;
}
int32_t x15660 = x15659 * x15659;
int32_t x15665;
if (x15646) {
x15665 = 0;
} else {
x15665 = x15561;
}
int32_t x15666;
if (x15646) {
x15666 = 0;
} else {
x15666 = 1;
}
bool x15729 = x15659 == 1;
bool x15730 = x15729 || true;
bool x15731 = x15730 || x15729;
bool x15741 = x15659 <= 1;
int32_t x15742;
if (x15741) {
x15742 = 1;
} else {
x15742 = x15659;
}
int32_t x15743 = x15742 * x15742;
int32_t x15748;
if (x15729) {
x15748 = 0;
} else {
x15748 = x15659;
}
int32_t x15749;
if (x15729) {
x15749 = 0;
} else {
x15749 = 1;
}
float* x92 = x5+5467584;
bool x15812 = x15742 == 1;
bool x15813 = x15812 || true;
bool x15814 = x15813 || x15812;
bool x15824 = x15742 <= 1;
int32_t x15825;
if (x15824) {
x15825 = 1;
} else {
x15825 = x15742;
}
int32_t x15826 = x15825 * x15825;
int32_t x15831;
if (x15812) {
x15831 = 0;
} else {
x15831 = x15742;
}
int32_t x15832;
if (x15812) {
x15832 = 0;
} else {
x15832 = 1;
}
float* x241 = x5+5467840;
int32_t x15879 = x15825 + 2;
int32_t x15880 = x15879 - 3;
int32_t x15881 = x15880 / 1;
int32_t x15882 = x15881 + 1;
int32_t x15886 = 16384 * x15882;
int32_t x15887 = x15886 * x15882;
int32_t x15883 = x15882 * x15882;
int32_t x15884 = 256 * x15883;
float* x249 = x5+5468608;
bool x16009 = x15882 == 1;
bool x16010 = x16009 || true;
bool x16011 = x16010 || x16009;
bool x16021 = x15882 <= 1;
int32_t x16022;
if (x16021) {
x16022 = 1;
} else {
x16022 = x15882;
}
int32_t x16023 = x16022 * x16022;
int32_t x16027;
if (x16009) {
x16027 = 0;
} else {
x16027 = x15882;
}
int32_t x16028;
if (x16009) {
x16028 = 0;
} else {
x16028 = 1;
}
float* x186 = x5+6058944;
float* x230 = x5+6059200;
bool x16107 = x16022 == 1;
bool x16108 = x16107 || true;
bool x16109 = x16108 || x16107;
bool x16119 = x16022 <= 1;
int32_t x16120;
if (x16119) {
x16120 = 1;
} else {
x16120 = x16022;
}
int32_t x16121 = x16120 * x16120;
int32_t x16126;
if (x16107) {
x16126 = 0;
} else {
x16126 = x16022;
}
int32_t x16127;
if (x16107) {
x16127 = 0;
} else {
x16127 = 1;
}
bool x16190 = x16120 == 1;
bool x16191 = x16190 || true;
bool x16192 = x16191 || x16190;
bool x16202 = x16120 <= 1;
int32_t x16203;
if (x16202) {
x16203 = 1;
} else {
x16203 = x16120;
}
int32_t x16204 = x16203 * x16203;
int32_t x16209;
if (x16190) {
x16209 = 0;
} else {
x16209 = x16120;
}
int32_t x16210;
if (x16190) {
x16210 = 0;
} else {
x16210 = 1;
}
float* x74 = x5+6058432;
bool x16273 = x16203 == 1;
bool x16274 = x16273 || true;
bool x16275 = x16274 || x16273;
bool x16285 = x16203 <= 1;
int32_t x16286;
if (x16285) {
x16286 = 1;
} else {
x16286 = x16203;
}
int32_t x16287 = x16286 * x16286;
int32_t x16292;
if (x16273) {
x16292 = 0;
} else {
x16292 = x16203;
}
int32_t x16293;
if (x16273) {
x16293 = 0;
} else {
x16293 = 1;
}
float* x136 = x5+6058688;
int32_t x16340 = x16286 - 1;
int32_t x16341 = x16340 / 1;
int32_t x16342 = x16341 + 1;
int32_t x16346 = 65536 * x16342;
int32_t x16347 = x16346 * x16342;
int32_t x16343 = x16342 * x16342;
int32_t x16344 = 1024 * x16343;
float* x89 = x5+6059456;
bool x16421 = x16342 == 1;
bool x16422 = x16421 || true;
bool x16423 = x16422 || x16421;
bool x16433 = x16342 <= 1;
int32_t x16434;
if (x16433) {
x16434 = 1;
} else {
x16434 = x16342;
}
int32_t x16435 = x16434 * x16434;
int32_t x16439;
if (x16421) {
x16439 = 0;
} else {
x16439 = x16342;
}
int32_t x16440;
if (x16421) {
x16440 = 0;
} else {
x16440 = 1;
}
float* x231 = x5+6323648;
float* x161 = x5+6324672;
bool x16519 = x16434 == 1;
bool x16520 = x16519 || true;
bool x16521 = x16520 || x16519;
bool x16531 = x16434 <= 1;
int32_t x16532;
if (x16531) {
x16532 = 1;
} else {
x16532 = x16434;
}
int32_t x16533 = x16532 * x16532;
int32_t x16538;
if (x16519) {
x16538 = 0;
} else {
x16538 = x16434;
}
int32_t x16539;
if (x16519) {
x16539 = 0;
} else {
x16539 = 1;
}
bool x16602 = x16532 == 1;
bool x16603 = x16602 || true;
bool x16604 = x16603 || x16602;
bool x16614 = x16532 <= 1;
int32_t x16615;
if (x16614) {
x16615 = 1;
} else {
x16615 = x16532;
}
int32_t x16616 = x16615 * x16615;
int32_t x16621;
if (x16602) {
x16621 = 0;
} else {
x16621 = x16532;
}
int32_t x16622;
if (x16602) {
x16622 = 0;
} else {
x16622 = 1;
}
float* x238 = x5+6321600;
bool x16685 = x16615 == 1;
bool x16686 = x16685 || true;
bool x16687 = x16686 || x16685;
bool x16697 = x16615 <= 1;
int32_t x16698;
if (x16697) {
x16698 = 1;
} else {
x16698 = x16615;
}
int32_t x16699 = x16698 * x16698;
int32_t x16704;
if (x16685) {
x16704 = 0;
} else {
x16704 = x16615;
}
int32_t x16705;
if (x16685) {
x16705 = 0;
} else {
x16705 = 1;
}
float* x146 = x5+6322624;
bool x16743 = x16698 == 1;
bool x16744 = x16743 || x15401;
bool x16745 = x16698 == x15356;
bool x16746 = x16744 || x16745;
bool x16756 = x16698 <= x15356;
int32_t x16757;
if (x16756) {
x16757 = x15356;
} else {
x16757 = x16698;
}
int32_t x16762;
if (x16743) {
x16762 = 0;
} else {
x16762 = x16698;
}
int32_t x16763;
if (x16743) {
x16763 = 0;
} else {
x16763 = 1;
}
int32_t x16809 = x16698 - 1;
int32_t x16810 = x16809 / 1;
int32_t x16811 = x16810 + 1;
int32_t x16815 = 16384 * x16811;
int32_t x16816 = x16815 * x16811;
int32_t x16812 = x16811 * x16811;
int32_t x16813 = 256 * x16812;
float* x22 = x5+6325696;
bool x16890 = x16811 == 1;
bool x16891 = x16890 || true;
bool x16892 = x16891 || x16890;
bool x16902 = x16811 <= 1;
int32_t x16903;
if (x16902) {
x16903 = 1;
} else {
x16903 = x16811;
}
int32_t x16904 = x16903 * x16903;
int32_t x16908;
if (x16890) {
x16908 = 0;
} else {
x16908 = x16811;
}
int32_t x16909;
if (x16890) {
x16909 = 0;
} else {
x16909 = 1;
}
float* x254 = x5+6588352;
float* x69 = x5+6588608;
bool x16988 = x16903 == 1;
bool x16989 = x16988 || true;
bool x16990 = x16989 || x16988;
bool x17000 = x16903 <= 1;
int32_t x17001;
if (x17000) {
x17001 = 1;
} else {
x17001 = x16903;
}
int32_t x17002 = x17001 * x17001;
int32_t x17007;
if (x16988) {
x17007 = 0;
} else {
x17007 = x16903;
}
int32_t x17008;
if (x16988) {
x17008 = 0;
} else {
x17008 = 1;
}
bool x17071 = x17001 == 1;
bool x17072 = x17071 || true;
bool x17073 = x17072 || x17071;
bool x17083 = x17001 <= 1;
int32_t x17084;
if (x17083) {
x17084 = 1;
} else {
x17084 = x17001;
}
int32_t x17085 = x17084 * x17084;
int32_t x17090;
if (x17071) {
x17090 = 0;
} else {
x17090 = x17001;
}
int32_t x17091;
if (x17071) {
x17091 = 0;
} else {
x17091 = 1;
}
float* x77 = x5+6587840;
bool x17154 = x17084 == 1;
bool x17155 = x17154 || true;
bool x17156 = x17155 || x17154;
bool x17166 = x17084 <= 1;
int32_t x17167;
if (x17166) {
x17167 = 1;
} else {
x17167 = x17084;
}
int32_t x17168 = x17167 * x17167;
int32_t x17173;
if (x17154) {
x17173 = 0;
} else {
x17173 = x17084;
}
int32_t x17174;
if (x17154) {
x17174 = 0;
} else {
x17174 = 1;
}
float* x185 = x5+6588096;
int32_t x17221 = x17167 + 2;
int32_t x17222 = x17221 - 3;
int32_t x17223 = x17222 / 1;
int32_t x17224 = x17223 + 1;
int32_t x17228 = 16384 * x17224;
int32_t x17229 = x17228 * x17224;
int32_t x17225 = x17224 * x17224;
int32_t x17226 = 256 * x17225;
float* x262 = x5+6588864;
bool x17351 = x17224 == 1;
bool x17352 = x17351 || true;
bool x17353 = x17352 || x17351;
bool x17363 = x17224 <= 1;
int32_t x17364;
if (x17363) {
x17364 = 1;
} else {
x17364 = x17224;
}
int32_t x17365 = x17364 * x17364;
int32_t x17369;
if (x17351) {
x17369 = 0;
} else {
x17369 = x17224;
}
int32_t x17370;
if (x17351) {
x17370 = 0;
} else {
x17370 = 1;
}
float* x250 = x5+7179200;
float* x104 = x5+7179456;
bool x17449 = x17364 == 1;
bool x17450 = x17449 || true;
bool x17451 = x17450 || x17449;
bool x17461 = x17364 <= 1;
int32_t x17462;
if (x17461) {
x17462 = 1;
} else {
x17462 = x17364;
}
int32_t x17463 = x17462 * x17462;
int32_t x17468;
if (x17449) {
x17468 = 0;
} else {
x17468 = x17364;
}
int32_t x17469;
if (x17449) {
x17469 = 0;
} else {
x17469 = 1;
}
bool x17532 = x17462 == 1;
bool x17533 = x17532 || true;
bool x17534 = x17533 || x17532;
bool x17544 = x17462 <= 1;
int32_t x17545;
if (x17544) {
x17545 = 1;
} else {
x17545 = x17462;
}
int32_t x17546 = x17545 * x17545;
int32_t x17551;
if (x17532) {
x17551 = 0;
} else {
x17551 = x17462;
}
int32_t x17552;
if (x17532) {
x17552 = 0;
} else {
x17552 = 1;
}
float* x168 = x5+7178688;
bool x17615 = x17545 == 1;
bool x17616 = x17615 || true;
bool x17617 = x17616 || x17615;
bool x17627 = x17545 <= 1;
int32_t x17628;
if (x17627) {
x17628 = 1;
} else {
x17628 = x17545;
}
int32_t x17629 = x17628 * x17628;
int32_t x17634;
if (x17615) {
x17634 = 0;
} else {
x17634 = x17545;
}
int32_t x17635;
if (x17615) {
x17635 = 0;
} else {
x17635 = 1;
}
float* x109 = x5+7178944;
int32_t x17682 = x17628 - 1;
int32_t x17683 = x17682 / 1;
int32_t x17684 = x17683 + 1;
int32_t x17688 = 65536 * x17684;
int32_t x17689 = x17688 * x17684;
int32_t x17685 = x17684 * x17684;
int32_t x17686 = 1024 * x17685;
float* x221 = x5+7179712;
bool x17763 = x17684 == 1;
bool x17764 = x17763 || true;
bool x17765 = x17764 || x17763;
bool x17775 = x17684 <= 1;
int32_t x17776;
if (x17775) {
x17776 = 1;
} else {
x17776 = x17684;
}
int32_t x17777 = x17776 * x17776;
int32_t x17781;
if (x17763) {
x17781 = 0;
} else {
x17781 = x17684;
}
int32_t x17782;
if (x17763) {
x17782 = 0;
} else {
x17782 = 1;
}
float* x209 = x5+7443904;
float* x272 = x5+7444928;
bool x17861 = x17776 == 1;
bool x17862 = x17861 || true;
bool x17863 = x17862 || x17861;
bool x17873 = x17776 <= 1;
int32_t x17874;
if (x17873) {
x17874 = 1;
} else {
x17874 = x17776;
}
int32_t x17875 = x17874 * x17874;
int32_t x17880;
if (x17861) {
x17880 = 0;
} else {
x17880 = x17776;
}
int32_t x17881;
if (x17861) {
x17881 = 0;
} else {
x17881 = 1;
}
bool x17944 = x17874 == 1;
bool x17945 = x17944 || true;
bool x17946 = x17945 || x17944;
bool x17956 = x17874 <= 1;
int32_t x17957;
if (x17956) {
x17957 = 1;
} else {
x17957 = x17874;
}
int32_t x17958 = x17957 * x17957;
int32_t x17963;
if (x17944) {
x17963 = 0;
} else {
x17963 = x17874;
}
int32_t x17964;
if (x17944) {
x17964 = 0;
} else {
x17964 = 1;
}
float* x59 = x5+7441856;
bool x18027 = x17957 == 1;
bool x18028 = x18027 || true;
bool x18029 = x18028 || x18027;
bool x18039 = x17957 <= 1;
int32_t x18040;
if (x18039) {
x18040 = 1;
} else {
x18040 = x17957;
}
int32_t x18041 = x18040 * x18040;
int32_t x18046;
if (x18027) {
x18046 = 0;
} else {
x18046 = x17957;
}
int32_t x18047;
if (x18027) {
x18047 = 0;
} else {
x18047 = 1;
}
float* x120 = x5+7442880;
bool x18085 = x18040 == 1;
bool x18086 = x18085 || x16743;
bool x18087 = x18040 == x16698;
bool x18088 = x18086 || x18087;
bool x18098 = x18040 <= x16698;
int32_t x18099;
if (x18098) {
x18099 = x16698;
} else {
x18099 = x18040;
}
int32_t x18104;
if (x18085) {
x18104 = 0;
} else {
x18104 = x18040;
}
int32_t x18105;
if (x18085) {
x18105 = 0;
} else {
x18105 = 1;
}
int32_t x18151 = x18040 - 1;
int32_t x18152 = x18151 / 1;
int32_t x18153 = x18152 + 1;
int32_t x18157 = 16384 * x18153;
int32_t x18158 = x18157 * x18153;
int32_t x18154 = x18153 * x18153;
int32_t x18155 = 256 * x18154;
float* x151 = x5+7445952;
bool x18232 = x18153 == 1;
bool x18233 = x18232 || true;
bool x18234 = x18233 || x18232;
bool x18244 = x18153 <= 1;
int32_t x18245;
if (x18244) {
x18245 = 1;
} else {
x18245 = x18153;
}
int32_t x18246 = x18245 * x18245;
int32_t x18250;
if (x18232) {
x18250 = 0;
} else {
x18250 = x18153;
}
int32_t x18251;
if (x18232) {
x18251 = 0;
} else {
x18251 = 1;
}
float* x80 = x5+7708608;
float* x176 = x5+7708864;
bool x18330 = x18245 == 1;
bool x18331 = x18330 || true;
bool x18332 = x18331 || x18330;
bool x18342 = x18245 <= 1;
int32_t x18343;
if (x18342) {
x18343 = 1;
} else {
x18343 = x18245;
}
int32_t x18344 = x18343 * x18343;
int32_t x18349;
if (x18330) {
x18349 = 0;
} else {
x18349 = x18245;
}
int32_t x18350;
if (x18330) {
x18350 = 0;
} else {
x18350 = 1;
}
bool x18413 = x18343 == 1;
bool x18414 = x18413 || true;
bool x18415 = x18414 || x18413;
bool x18425 = x18343 <= 1;
int32_t x18426;
if (x18425) {
x18426 = 1;
} else {
x18426 = x18343;
}
int32_t x18427 = x18426 * x18426;
int32_t x18432;
if (x18413) {
x18432 = 0;
} else {
x18432 = x18343;
}
int32_t x18433;
if (x18413) {
x18433 = 0;
} else {
x18433 = 1;
}
float* x85 = x5+7708096;
bool x18496 = x18426 == 1;
bool x18497 = x18496 || true;
bool x18498 = x18497 || x18496;
bool x18508 = x18426 <= 1;
int32_t x18509;
if (x18508) {
x18509 = 1;
} else {
x18509 = x18426;
}
int32_t x18510 = x18509 * x18509;
int32_t x18515;
if (x18496) {
x18515 = 0;
} else {
x18515 = x18426;
}
int32_t x18516;
if (x18496) {
x18516 = 0;
} else {
x18516 = 1;
}
float* x253 = x5+7708352;
int32_t x18563 = x18509 + 2;
int32_t x18564 = x18563 - 3;
int32_t x18565 = x18564 / 1;
int32_t x18566 = x18565 + 1;
int32_t x18570 = 16384 * x18566;
int32_t x18571 = x18570 * x18566;
int32_t x18567 = x18566 * x18566;
int32_t x18568 = 256 * x18567;
float* x226 = x5+7709120;
bool x18693 = x18566 == 1;
bool x18694 = x18693 || true;
bool x18695 = x18694 || x18693;
bool x18705 = x18566 <= 1;
int32_t x18706;
if (x18705) {
x18706 = 1;
} else {
x18706 = x18566;
}
int32_t x18707 = x18706 * x18706;
int32_t x18711;
if (x18693) {
x18711 = 0;
} else {
x18711 = x18566;
}
int32_t x18712;
if (x18693) {
x18712 = 0;
} else {
x18712 = 1;
}
float* x70 = x5+8299456;
float* x240 = x5+8299712;
bool x18791 = x18706 == 1;
bool x18792 = x18791 || true;
bool x18793 = x18792 || x18791;
bool x18803 = x18706 <= 1;
int32_t x18804;
if (x18803) {
x18804 = 1;
} else {
x18804 = x18706;
}
int32_t x18805 = x18804 * x18804;
int32_t x18810;
if (x18791) {
x18810 = 0;
} else {
x18810 = x18706;
}
int32_t x18811;
if (x18791) {
x18811 = 0;
} else {
x18811 = 1;
}
bool x18874 = x18804 == 1;
bool x18875 = x18874 || true;
bool x18876 = x18875 || x18874;
bool x18886 = x18804 <= 1;
int32_t x18887;
if (x18886) {
x18887 = 1;
} else {
x18887 = x18804;
}
int32_t x18888 = x18887 * x18887;
int32_t x18893;
if (x18874) {
x18893 = 0;
} else {
x18893 = x18804;
}
int32_t x18894;
if (x18874) {
x18894 = 0;
} else {
x18894 = 1;
}
float* x141 = x5+8298944;
bool x18957 = x18887 == 1;
bool x18958 = x18957 || true;
bool x18959 = x18958 || x18957;
bool x18969 = x18887 <= 1;
int32_t x18970;
if (x18969) {
x18970 = 1;
} else {
x18970 = x18887;
}
int32_t x18971 = x18970 * x18970;
int32_t x18976;
if (x18957) {
x18976 = 0;
} else {
x18976 = x18887;
}
int32_t x18977;
if (x18957) {
x18977 = 0;
} else {
x18977 = 1;
}
float* x189 = x5+8299200;
int32_t x19024 = x18970 - 1;
int32_t x19025 = x19024 / 1;
int32_t x19026 = x19025 + 1;
int32_t x19030 = 65536 * x19026;
int32_t x19031 = x19030 * x19026;
int32_t x19027 = x19026 * x19026;
int32_t x19028 = 1024 * x19027;
float* x97 = x5+8299968;
bool x19105 = x19026 == 1;
bool x19106 = x19105 || true;
bool x19107 = x19106 || x19105;
bool x19117 = x19026 <= 1;
int32_t x19118;
if (x19117) {
x19118 = 1;
} else {
x19118 = x19026;
}
int32_t x19119 = x19118 * x19118;
int32_t x19123;
if (x19105) {
x19123 = 0;
} else {
x19123 = x19026;
}
int32_t x19124;
if (x19105) {
x19124 = 0;
} else {
x19124 = 1;
}
float* x122 = x5+8564160;
float* x183 = x5+8565184;
bool x19203 = x19118 == 1;
bool x19204 = x19203 || true;
bool x19205 = x19204 || x19203;
bool x19215 = x19118 <= 1;
int32_t x19216;
if (x19215) {
x19216 = 1;
} else {
x19216 = x19118;
}
int32_t x19217 = x19216 * x19216;
int32_t x19222;
if (x19203) {
x19222 = 0;
} else {
x19222 = x19118;
}
int32_t x19223;
if (x19203) {
x19223 = 0;
} else {
x19223 = 1;
}
bool x19286 = x19216 == 1;
bool x19287 = x19286 || true;
bool x19288 = x19287 || x19286;
bool x19298 = x19216 <= 1;
int32_t x19299;
if (x19298) {
x19299 = 1;
} else {
x19299 = x19216;
}
int32_t x19300 = x19299 * x19299;
int32_t x19305;
if (x19286) {
x19305 = 0;
} else {
x19305 = x19216;
}
int32_t x19306;
if (x19286) {
x19306 = 0;
} else {
x19306 = 1;
}
float* x248 = x5+8562112;
bool x19369 = x19299 == 1;
bool x19370 = x19369 || true;
bool x19371 = x19370 || x19369;
bool x19381 = x19299 <= 1;
int32_t x19382;
if (x19381) {
x19382 = 1;
} else {
x19382 = x19299;
}
int32_t x19383 = x19382 * x19382;
int32_t x19388;
if (x19369) {
x19388 = 0;
} else {
x19388 = x19299;
}
int32_t x19389;
if (x19369) {
x19389 = 0;
} else {
x19389 = 1;
}
float* x93 = x5+8563136;
bool x19427 = x19382 == 1;
bool x19428 = x19427 || x18085;
bool x19429 = x19382 == x18040;
bool x19430 = x19428 || x19429;
bool x19440 = x19382 <= x18040;
int32_t x19441;
if (x19440) {
x19441 = x18040;
} else {
x19441 = x19382;
}
int32_t x19446;
if (x19427) {
x19446 = 0;
} else {
x19446 = x19382;
}
int32_t x19447;
if (x19427) {
x19447 = 0;
} else {
x19447 = 1;
}
int32_t x19493 = x19382 - 1;
int32_t x19494 = x19493 / 1;
int32_t x19495 = x19494 + 1;
int32_t x19499 = 32768 * x19495;
int32_t x19500 = x19499 * x19495;
int32_t x19496 = x19495 * x19495;
int32_t x19497 = 512 * x19496;
float* x139 = x5+8566208;
bool x19574 = x19495 == 1;
bool x19575 = x19574 || true;
bool x19576 = x19575 || x19574;
bool x19586 = x19495 <= 1;
int32_t x19587;
if (x19586) {
x19587 = 1;
} else {
x19587 = x19495;
}
int32_t x19588 = x19587 * x19587;
int32_t x19592;
if (x19574) {
x19592 = 0;
} else {
x19592 = x19495;
}
int32_t x19593;
if (x19574) {
x19593 = 0;
} else {
x19593 = 1;
}
float* x67 = x5+9091520;
float* x121 = x5+9092032;
bool x19672 = x19587 == 1;
bool x19673 = x19672 || true;
bool x19674 = x19673 || x19672;
bool x19684 = x19587 <= 1;
int32_t x19685;
if (x19684) {
x19685 = 1;
} else {
x19685 = x19587;
}
int32_t x19686 = x19685 * x19685;
int32_t x19691;
if (x19672) {
x19691 = 0;
} else {
x19691 = x19587;
}
int32_t x19692;
if (x19672) {
x19692 = 0;
} else {
x19692 = 1;
}
bool x19755 = x19685 == 1;
bool x19756 = x19755 || true;
bool x19757 = x19756 || x19755;
bool x19767 = x19685 <= 1;
int32_t x19768;
if (x19767) {
x19768 = 1;
} else {
x19768 = x19685;
}
int32_t x19769 = x19768 * x19768;
int32_t x19774;
if (x19755) {
x19774 = 0;
} else {
x19774 = x19685;
}
int32_t x19775;
if (x19755) {
x19775 = 0;
} else {
x19775 = 1;
}
float* x201 = x5+9090496;
bool x19838 = x19768 == 1;
bool x19839 = x19838 || true;
bool x19840 = x19839 || x19838;
bool x19850 = x19768 <= 1;
int32_t x19851;
if (x19850) {
x19851 = 1;
} else {
x19851 = x19768;
}
int32_t x19852 = x19851 * x19851;
int32_t x19857;
if (x19838) {
x19857 = 0;
} else {
x19857 = x19768;
}
int32_t x19858;
if (x19838) {
x19858 = 0;
} else {
x19858 = 1;
}
float* x224 = x5+9091008;
int32_t x19905 = x19851 + 2;
int32_t x19906 = x19905 - 3;
int32_t x19907 = x19906 / 2;
int32_t x19908 = x19907 + 1;
int32_t x19912 = 32768 * x19908;
int32_t x19913 = x19912 * x19908;
int32_t x19909 = x19908 * x19908;
int32_t x19910 = 512 * x19909;
float* x34 = x5+9092544;
bool x20019 = x19908 == 1;
bool x20020 = x20019 || true;
bool x20021 = x20020 || x20019;
bool x20031 = x19908 <= 1;
int32_t x20032;
if (x20031) {
x20032 = 1;
} else {
x20032 = x19908;
}
int32_t x20033 = x20032 * x20032;
int32_t x20037;
if (x20019) {
x20037 = 0;
} else {
x20037 = x19908;
}
int32_t x20038;
if (x20019) {
x20038 = 0;
} else {
x20038 = 1;
}
float* x113 = x5+11452864;
float* x50 = x5+11453376;
bool x20117 = x20032 == 1;
bool x20118 = x20117 || true;
bool x20119 = x20118 || x20117;
bool x20129 = x20032 <= 1;
int32_t x20130;
if (x20129) {
x20130 = 1;
} else {
x20130 = x20032;
}
int32_t x20131 = x20130 * x20130;
int32_t x20136;
if (x20117) {
x20136 = 0;
} else {
x20136 = x20032;
}
int32_t x20137;
if (x20117) {
x20137 = 0;
} else {
x20137 = 1;
}
bool x20200 = x20130 == 1;
bool x20201 = x20200 || true;
bool x20202 = x20201 || x20200;
bool x20212 = x20130 <= 1;
int32_t x20213;
if (x20212) {
x20213 = 1;
} else {
x20213 = x20130;
}
int32_t x20214 = x20213 * x20213;
int32_t x20219;
if (x20200) {
x20219 = 0;
} else {
x20219 = x20130;
}
int32_t x20220;
if (x20200) {
x20220 = 0;
} else {
x20220 = 1;
}
float* x205 = x5+11451840;
bool x20283 = x20213 == 1;
bool x20284 = x20283 || true;
bool x20285 = x20284 || x20283;
bool x20295 = x20213 <= 1;
int32_t x20296;
if (x20295) {
x20296 = 1;
} else {
x20296 = x20213;
}
int32_t x20297 = x20296 * x20296;
int32_t x20302;
if (x20283) {
x20302 = 0;
} else {
x20302 = x20213;
}
int32_t x20303;
if (x20283) {
x20303 = 0;
} else {
x20303 = 1;
}
float* x159 = x5+11452352;
int32_t x20350 = x20296 - 1;
int32_t x20351 = x20350 / 1;
int32_t x20352 = x20351 + 1;
int32_t x20356 = 131072 * x20352;
int32_t x20357 = x20356 * x20352;
int32_t x20353 = x20352 * x20352;
int32_t x20354 = 2048 * x20353;
float* x212 = x5+11453888;
bool x20431 = x20352 == 1;
bool x20432 = x20431 || true;
bool x20433 = x20432 || x20431;
bool x20443 = x20352 <= 1;
int32_t x20444;
if (x20443) {
x20444 = 1;
} else {
x20444 = x20352;
}
int32_t x20445 = x20444 * x20444;
int32_t x20449;
if (x20431) {
x20449 = 0;
} else {
x20449 = x20352;
}
int32_t x20450;
if (x20431) {
x20450 = 0;
} else {
x20450 = 1;
}
float* x115 = x5+12506560;
float* x193 = x5+12508608;
bool x20530 = x20444 == 1;
bool x20531 = x20530 || true;
bool x20532 = x20531 || x20530;
bool x20542 = x20444 <= 1;
int32_t x20543;
if (x20542) {
x20543 = 1;
} else {
x20543 = x20444;
}
int32_t x20544 = x20543 * x20543;
int32_t x20549;
if (x20530) {
x20549 = 0;
} else {
x20549 = x20444;
}
int32_t x20550;
if (x20530) {
x20550 = 0;
} else {
x20550 = 1;
}
bool x20613 = x20543 == 1;
bool x20614 = x20613 || true;
bool x20615 = x20614 || x20613;
bool x20625 = x20543 <= 1;
int32_t x20626;
if (x20625) {
x20626 = 1;
} else {
x20626 = x20543;
}
int32_t x20627 = x20626 * x20626;
int32_t x20632;
if (x20613) {
x20632 = 0;
} else {
x20632 = x20543;
}
int32_t x20633;
if (x20613) {
x20633 = 0;
} else {
x20633 = 1;
}
float* x239 = x5+12502464;
bool x20696 = x20626 == 1;
bool x20697 = x20696 || true;
bool x20698 = x20697 || x20696;
bool x20708 = x20626 <= 1;
int32_t x20709;
if (x20708) {
x20709 = 1;
} else {
x20709 = x20626;
}
int32_t x20710 = x20709 * x20709;
int32_t x20715;
if (x20696) {
x20715 = 0;
} else {
x20715 = x20626;
}
int32_t x20716;
if (x20696) {
x20716 = 0;
} else {
x20716 = 1;
}
float* x62 = x5+12504512;
int32_t x20750 = x19493 / 2;
int32_t x20751 = x20750 + 1;
int32_t x20755 = 131072 * x20751;
int32_t x20756 = x20755 * x20751;
int32_t x20752 = x20751 * x20751;
int32_t x20753 = 2048 * x20752;
float* x214 = x5+12510656;
bool x20836 = x20751 == 1;
bool x20837 = x20836 || true;
bool x20838 = x20837 || x20836;
bool x20848 = x20751 <= 1;
int32_t x20849;
if (x20848) {
x20849 = 1;
} else {
x20849 = x20751;
}
int32_t x20850 = x20849 * x20849;
int32_t x20854;
if (x20836) {
x20854 = 0;
} else {
x20854 = x20751;
}
int32_t x20855;
if (x20836) {
x20855 = 0;
} else {
x20855 = 1;
}
float* x64 = x5+14611904;
float* x125 = x5+14613952;
bool x20934 = x20849 == 1;
bool x20935 = x20934 || true;
bool x20936 = x20935 || x20934;
bool x20946 = x20849 <= 1;
int32_t x20947;
if (x20946) {
x20947 = 1;
} else {
x20947 = x20849;
}
int32_t x20948 = x20947 * x20947;
int32_t x20953;
if (x20934) {
x20953 = 0;
} else {
x20953 = x20849;
}
int32_t x20954;
if (x20934) {
x20954 = 0;
} else {
x20954 = 1;
}
bool x21017 = x20947 == 1;
bool x21018 = x21017 || true;
bool x21019 = x21018 || x21017;
bool x21029 = x20947 <= 1;
int32_t x21030;
if (x21029) {
x21030 = 1;
} else {
x21030 = x20947;
}
int32_t x21031 = x21030 * x21030;
int32_t x21036;
if (x21017) {
x21036 = 0;
} else {
x21036 = x20947;
}
int32_t x21037;
if (x21017) {
x21037 = 0;
} else {
x21037 = 1;
}
float* x173 = x5+14607808;
bool x21100 = x21030 == 1;
bool x21101 = x21100 || true;
bool x21102 = x21101 || x21100;
bool x21112 = x21030 <= 1;
int32_t x21113;
if (x21112) {
x21113 = 1;
} else {
x21113 = x21030;
}
int32_t x21114 = x21113 * x21113;
int32_t x21119;
if (x21100) {
x21119 = 0;
} else {
x21119 = x21030;
}
int32_t x21120;
if (x21100) {
x21120 = 0;
} else {
x21120 = 1;
}
float* x107 = x5+14609856;
bool x21159 = x20709 == 1;
bool x21160 = x21113 == 1;
bool x21161 = x21159 || x21160;
bool x21162 = x20709 == x21113;
bool x21163 = x21161 || x21162;
bool x21173 = x20709 <= x21113;
int32_t x21174;
if (x21173) {
x21174 = x21113;
} else {
x21174 = x20709;
}
int32_t x21179;
if (x21159) {
x21179 = 0;
} else {
x21179 = x20709;
}
int32_t x21180;
if (x21159) {
x21180 = 0;
} else {
x21180 = 1;
}
int32_t x21182;
if (x21160) {
x21182 = 0;
} else {
x21182 = x21113;
}
int32_t x21183;
if (x21160) {
x21183 = 0;
} else {
x21183 = 1;
}
int32_t x21229 = x20709 - 1;
int32_t x21230 = x21229 / 1;
int32_t x21231 = x21230 + 1;
int32_t x21235 = 32768 * x21231;
int32_t x21236 = x21235 * x21231;
int32_t x21232 = x21231 * x21231;
int32_t x21233 = 512 * x21232;
float* x215 = x5+14616000;
bool x21310 = x21231 == 1;
bool x21311 = x21310 || true;
bool x21312 = x21311 || x21310;
bool x21322 = x21231 <= 1;
int32_t x21323;
if (x21322) {
x21323 = 1;
} else {
x21323 = x21231;
}
int32_t x21324 = x21323 * x21323;
int32_t x21328;
if (x21310) {
x21328 = 0;
} else {
x21328 = x21231;
}
int32_t x21329;
if (x21310) {
x21329 = 0;
} else {
x21329 = 1;
}
float* x154 = x5+15665600;
float* x65 = x5+15666112;
bool x21408 = x21323 == 1;
bool x21409 = x21408 || true;
bool x21410 = x21409 || x21408;
bool x21420 = x21323 <= 1;
int32_t x21421;
if (x21420) {
x21421 = 1;
} else {
x21421 = x21323;
}
int32_t x21422 = x21421 * x21421;
int32_t x21427;
if (x21408) {
x21427 = 0;
} else {
x21427 = x21323;
}
int32_t x21428;
if (x21408) {
x21428 = 0;
} else {
x21428 = 1;
}
bool x21491 = x21421 == 1;
bool x21492 = x21491 || true;
bool x21493 = x21492 || x21491;
bool x21503 = x21421 <= 1;
int32_t x21504;
if (x21503) {
x21504 = 1;
} else {
x21504 = x21421;
}
int32_t x21505 = x21504 * x21504;
int32_t x21510;
if (x21491) {
x21510 = 0;
} else {
x21510 = x21421;
}
int32_t x21511;
if (x21491) {
x21511 = 0;
} else {
x21511 = 1;
}
float* x46 = x5+15664576;
bool x21574 = x21504 == 1;
bool x21575 = x21574 || true;
bool x21576 = x21575 || x21574;
bool x21586 = x21504 <= 1;
int32_t x21587;
if (x21586) {
x21587 = 1;
} else {
x21587 = x21504;
}
int32_t x21588 = x21587 * x21587;
int32_t x21593;
if (x21574) {
x21593 = 0;
} else {
x21593 = x21504;
}
int32_t x21594;
if (x21574) {
x21594 = 0;
} else {
x21594 = 1;
}
float* x137 = x5+15665088;
int32_t x21641 = x21587 + 2;
int32_t x21642 = x21641 - 3;
int32_t x21643 = x21642 / 1;
int32_t x21644 = x21643 + 1;
int32_t x21648 = 32768 * x21644;
int32_t x21649 = x21648 * x21644;
int32_t x21645 = x21644 * x21644;
int32_t x21646 = 512 * x21645;
float* x155 = x5+15666624;
bool x21771 = x21644 == 1;
bool x21772 = x21771 || true;
bool x21773 = x21772 || x21771;
bool x21783 = x21644 <= 1;
int32_t x21784;
if (x21783) {
x21784 = 1;
} else {
x21784 = x21644;
}
int32_t x21785 = x21784 * x21784;
int32_t x21789;
if (x21771) {
x21789 = 0;
} else {
x21789 = x21644;
}
int32_t x21790;
if (x21771) {
x21790 = 0;
} else {
x21790 = 1;
}
float* x138 = x5+18026944;
float* x195 = x5+18027456;
bool x21869 = x21784 == 1;
bool x21870 = x21869 || true;
bool x21871 = x21870 || x21869;
bool x21881 = x21784 <= 1;
int32_t x21882;
if (x21881) {
x21882 = 1;
} else {
x21882 = x21784;
}
int32_t x21883 = x21882 * x21882;
int32_t x21888;
if (x21869) {
x21888 = 0;
} else {
x21888 = x21784;
}
int32_t x21889;
if (x21869) {
x21889 = 0;
} else {
x21889 = 1;
}
bool x21952 = x21882 == 1;
bool x21953 = x21952 || true;
bool x21954 = x21953 || x21952;
bool x21964 = x21882 <= 1;
int32_t x21965;
if (x21964) {
x21965 = 1;
} else {
x21965 = x21882;
}
int32_t x21966 = x21965 * x21965;
int32_t x21971;
if (x21952) {
x21971 = 0;
} else {
x21971 = x21882;
}
int32_t x21972;
if (x21952) {
x21972 = 0;
} else {
x21972 = 1;
}
float* x160 = x5+18025920;
bool x22035 = x21965 == 1;
bool x22036 = x22035 || true;
bool x22037 = x22036 || x22035;
bool x22047 = x21965 <= 1;
int32_t x22048;
if (x22047) {
x22048 = 1;
} else {
x22048 = x21965;
}
int32_t x22049 = x22048 * x22048;
int32_t x22054;
if (x22035) {
x22054 = 0;
} else {
x22054 = x21965;
}
int32_t x22055;
if (x22035) {
x22055 = 0;
} else {
x22055 = 1;
}
float* x66 = x5+18026432;
int32_t x22102 = x22048 - 1;
int32_t x22103 = x22102 / 1;
int32_t x22104 = x22103 + 1;
int32_t x22108 = 131072 * x22104;
int32_t x22109 = x22108 * x22104;
int32_t x22105 = x22104 * x22104;
int32_t x22106 = 2048 * x22105;
float* x47 = x5+18027968;
bool x22183 = x22104 == 1;
bool x22184 = x22183 || true;
bool x22185 = x22184 || x22183;
bool x22195 = x22104 <= 1;
int32_t x22196;
if (x22195) {
x22196 = 1;
} else {
x22196 = x22104;
}
int32_t x22197 = x22196 * x22196;
int32_t x22201;
if (x22183) {
x22201 = 0;
} else {
x22201 = x22104;
}
int32_t x22202;
if (x22183) {
x22202 = 0;
} else {
x22202 = 1;
}
float* x68 = x5+19080640;
float* x245 = x5+19082688;
bool x22281 = x22196 == 1;
bool x22282 = x22281 || true;
bool x22283 = x22282 || x22281;
bool x22293 = x22196 <= 1;
int32_t x22294;
if (x22293) {
x22294 = 1;
} else {
x22294 = x22196;
}
int32_t x22295 = x22294 * x22294;
int32_t x22300;
if (x22281) {
x22300 = 0;
} else {
x22300 = x22196;
}
int32_t x22301;
if (x22281) {
x22301 = 0;
} else {
x22301 = 1;
}
bool x22364 = x22294 == 1;
bool x22365 = x22364 || true;
bool x22366 = x22365 || x22364;
bool x22376 = x22294 <= 1;
int32_t x22377;
if (x22376) {
x22377 = 1;
} else {
x22377 = x22294;
}
int32_t x22378 = x22377 * x22377;
int32_t x22383;
if (x22364) {
x22383 = 0;
} else {
x22383 = x22294;
}
int32_t x22384;
if (x22364) {
x22384 = 0;
} else {
x22384 = 1;
}
float* x94 = x5+19076544;
bool x22447 = x22377 == 1;
bool x22448 = x22447 || true;
bool x22449 = x22448 || x22447;
bool x22459 = x22377 <= 1;
int32_t x22460;
if (x22459) {
x22460 = 1;
} else {
x22460 = x22377;
}
int32_t x22461 = x22460 * x22460;
int32_t x22466;
if (x22447) {
x22466 = 0;
} else {
x22466 = x22377;
}
int32_t x22467;
if (x22447) {
x22467 = 0;
} else {
x22467 = 1;
}
float* x144 = x5+19078592;
bool x22505 = x22460 == 1;
bool x22506 = x22505 || x21159;
bool x22507 = x22460 == x20709;
bool x22508 = x22506 || x22507;
bool x22518 = x22460 <= x20709;
int32_t x22519;
if (x22518) {
x22519 = x20709;
} else {
x22519 = x22460;
}
int32_t x22524;
if (x22505) {
x22524 = 0;
} else {
x22524 = x22460;
}
int32_t x22525;
if (x22505) {
x22525 = 0;
} else {
x22525 = 1;
}
int32_t x22571 = x22460 - 1;
int32_t x22572 = x22571 / 1;
int32_t x22573 = x22572 + 1;
int32_t x22577 = 32768 * x22573;
int32_t x22578 = x22577 * x22573;
int32_t x22574 = x22573 * x22573;
int32_t x22575 = 512 * x22574;
float* x265 = x5+19084736;
bool x22652 = x22573 == 1;
bool x22653 = x22652 || true;
bool x22654 = x22653 || x22652;
bool x22664 = x22573 <= 1;
int32_t x22665;
if (x22664) {
x22665 = 1;
} else {
x22665 = x22573;
}
int32_t x22666 = x22665 * x22665;
int32_t x22670;
if (x22652) {
x22670 = 0;
} else {
x22670 = x22573;
}
int32_t x22671;
if (x22652) {
x22671 = 0;
} else {
x22671 = 1;
}
float* x213 = x5+20134336;
float* x255 = x5+20134848;
bool x22750 = x22665 == 1;
bool x22751 = x22750 || true;
bool x22752 = x22751 || x22750;
bool x22762 = x22665 <= 1;
int32_t x22763;
if (x22762) {
x22763 = 1;
} else {
x22763 = x22665;
}
int32_t x22764 = x22763 * x22763;
int32_t x22769;
if (x22750) {
x22769 = 0;
} else {
x22769 = x22665;
}
int32_t x22770;
if (x22750) {
x22770 = 0;
} else {
x22770 = 1;
}
bool x22833 = x22763 == 1;
bool x22834 = x22833 || true;
bool x22835 = x22834 || x22833;
bool x22845 = x22763 <= 1;
int32_t x22846;
if (x22845) {
x22846 = 1;
} else {
x22846 = x22763;
}
int32_t x22847 = x22846 * x22846;
int32_t x22852;
if (x22833) {
x22852 = 0;
} else {
x22852 = x22763;
}
int32_t x22853;
if (x22833) {
x22853 = 0;
} else {
x22853 = 1;
}
float* x15 = x5+20133312;
bool x22916 = x22846 == 1;
bool x22917 = x22916 || true;
bool x22918 = x22917 || x22916;
bool x22928 = x22846 <= 1;
int32_t x22929;
if (x22928) {
x22929 = 1;
} else {
x22929 = x22846;
}
int32_t x22930 = x22929 * x22929;
int32_t x22935;
if (x22916) {
x22935 = 0;
} else {
x22935 = x22846;
}
int32_t x22936;
if (x22916) {
x22936 = 0;
} else {
x22936 = 1;
}
float* x78 = x5+20133824;
int32_t x22983 = x22929 + 2;
int32_t x22984 = x22983 - 3;
int32_t x22985 = x22984 / 1;
int32_t x22986 = x22985 + 1;
int32_t x22990 = 32768 * x22986;
int32_t x22991 = x22990 * x22986;
int32_t x22987 = x22986 * x22986;
int32_t x22988 = 512 * x22987;
float* x28 = x5+20135360;
bool x23113 = x22986 == 1;
bool x23114 = x23113 || true;
bool x23115 = x23114 || x23113;
bool x23125 = x22986 <= 1;
int32_t x23126;
if (x23125) {
x23126 = 1;
} else {
x23126 = x22986;
}
int32_t x23127 = x23126 * x23126;
int32_t x23131;
if (x23113) {
x23131 = 0;
} else {
x23131 = x22986;
}
int32_t x23132;
if (x23113) {
x23132 = 0;
} else {
x23132 = 1;
}
float* x12 = x5+22495680;
float* x202 = x5+22496192;
bool x23211 = x23126 == 1;
bool x23212 = x23211 || true;
bool x23213 = x23212 || x23211;
bool x23223 = x23126 <= 1;
int32_t x23224;
if (x23223) {
x23224 = 1;
} else {
x23224 = x23126;
}
int32_t x23225 = x23224 * x23224;
int32_t x23230;
if (x23211) {
x23230 = 0;
} else {
x23230 = x23126;
}
int32_t x23231;
if (x23211) {
x23231 = 0;
} else {
x23231 = 1;
}
bool x23294 = x23224 == 1;
bool x23295 = x23294 || true;
bool x23296 = x23295 || x23294;
bool x23306 = x23224 <= 1;
int32_t x23307;
if (x23306) {
x23307 = 1;
} else {
x23307 = x23224;
}
int32_t x23308 = x23307 * x23307;
int32_t x23313;
if (x23294) {
x23313 = 0;
} else {
x23313 = x23224;
}
int32_t x23314;
if (x23294) {
x23314 = 0;
} else {
x23314 = 1;
}
float* x194 = x5+22494656;
bool x23377 = x23307 == 1;
bool x23378 = x23377 || true;
bool x23379 = x23378 || x23377;
bool x23389 = x23307 <= 1;
int32_t x23390;
if (x23389) {
x23390 = 1;
} else {
x23390 = x23307;
}
int32_t x23391 = x23390 * x23390;
int32_t x23396;
if (x23377) {
x23396 = 0;
} else {
x23396 = x23307;
}
int32_t x23397;
if (x23377) {
x23397 = 0;
} else {
x23397 = 1;
}
float* x169 = x5+22495168;
int32_t x23444 = x23390 - 1;
int32_t x23445 = x23444 / 1;
int32_t x23446 = x23445 + 1;
int32_t x23450 = 131072 * x23446;
int32_t x23451 = x23450 * x23446;
int32_t x23447 = x23446 * x23446;
int32_t x23448 = 2048 * x23447;
float* x33 = x5+22496704;
bool x23525 = x23446 == 1;
bool x23526 = x23525 || true;
bool x23527 = x23526 || x23525;
bool x23537 = x23446 <= 1;
int32_t x23538;
if (x23537) {
x23538 = 1;
} else {
x23538 = x23446;
}
int32_t x23539 = x23538 * x23538;
int32_t x23543;
if (x23525) {
x23543 = 0;
} else {
x23543 = x23446;
}
int32_t x23544;
if (x23525) {
x23544 = 0;
} else {
x23544 = 1;
}
float* x260 = x5+23549376;
float* x123 = x5+23551424;
bool x23623 = x23538 == 1;
bool x23624 = x23623 || true;
bool x23625 = x23624 || x23623;
bool x23635 = x23538 <= 1;
int32_t x23636;
if (x23635) {
x23636 = 1;
} else {
x23636 = x23538;
}
int32_t x23637 = x23636 * x23636;
int32_t x23642;
if (x23623) {
x23642 = 0;
} else {
x23642 = x23538;
}
int32_t x23643;
if (x23623) {
x23643 = 0;
} else {
x23643 = 1;
}
bool x23706 = x23636 == 1;
bool x23707 = x23706 || true;
bool x23708 = x23707 || x23706;
bool x23718 = x23636 <= 1;
int32_t x23719;
if (x23718) {
x23719 = 1;
} else {
x23719 = x23636;
}
int32_t x23720 = x23719 * x23719;
int32_t x23725;
if (x23706) {
x23725 = 0;
} else {
x23725 = x23636;
}
int32_t x23726;
if (x23706) {
x23726 = 0;
} else {
x23726 = 1;
}
float* x103 = x5+23545280;
bool x23789 = x23719 == 1;
bool x23790 = x23789 || true;
bool x23791 = x23790 || x23789;
bool x23801 = x23719 <= 1;
int32_t x23802;
if (x23801) {
x23802 = 1;
} else {
x23802 = x23719;
}
int32_t x23803 = x23802 * x23802;
int32_t x23808;
if (x23789) {
x23808 = 0;
} else {
x23808 = x23719;
}
int32_t x23809;
if (x23789) {
x23809 = 0;
} else {
x23809 = 1;
}
float* x181 = x5+23547328;
bool x23847 = x23802 == 1;
bool x23848 = x23847 || x22505;
bool x23849 = x23802 == x22460;
bool x23850 = x23848 || x23849;
bool x23860 = x23802 <= x22460;
int32_t x23861;
if (x23860) {
x23861 = x22460;
} else {
x23861 = x23802;
}
int32_t x23866;
if (x23847) {
x23866 = 0;
} else {
x23866 = x23802;
}
int32_t x23867;
if (x23847) {
x23867 = 0;
} else {
x23867 = 1;
}
bool x23913 = x23802 >= 2;
bool x23914;
if (x23913) {
x23914 = x23913;
} else {
x23914 = false;
}
int32_t x23919 = x23802 - 2;
int32_t x23920 = x23919 / 1;
int32_t x23921 = x23920 + 1;
int32_t x23922 = x23921 * x23921;
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
int32_t x474 = x470 * x473;
int32_t x475 = 64 * x474;
float* x476 = (float*)myMalloc(x475 * sizeof(float));;
int32_t x479;
if (x455) {
x479 = 0;
} else {
x479 = 1;
}
for(int x480=0; x480 < 64; x480++) {
int32_t x492 = x335 * x480;
int32_t x486 = x474 * x480;
for(int x482=0; x482 < x470; x482++) {
int32_t x493 = x334 * x482;
int32_t x494 = x492 + x493;
int32_t x499 = x479 * x482;
int32_t x488 = x473 * x482;
for(int x484=0; x484 < x472; x484++) {
int32_t x495 = x477 * x484;
int32_t x496 = x494 + x495;
int32_t x490 = x472 * x484;
for(int x485=0; x485 < x472; x485++) {
int32_t x497 = x478 * x485;
int32_t x498 = x496 + x497;
float x500 = x339[x498];
float x501 = x40[x499];
int32_t x487 = x485 + x486;
int32_t x489 = x487 + x488;
int32_t x491 = x489 + x490;
float x502 = x500 - x501;
x476[x491] = x502;

}

}

}

}
float* x512 = (float*)myMalloc(64 * sizeof(float));;
for(int x513=0; x513 < 64; x513++) {
float x514 = x110[x513];
float x515 = x514 + 1.0E-5f;
x512[x513] = x515;

}
float* x519 = (float*)myMalloc(64 * sizeof(float));;
for(int x520=0; x520 < 64; x520++) {
float x521 = x512[x520];
double x522 = (double)x521;
double x523 = sqrt(x522);
float x524 = (float)x523;
x519[x520] = x524;

}
int32_t x528 = 0;
int32_t x529 = 1;
x529 *= 1;
x528 += 1;
x529 *= 1;
x529 *= 1;
int32_t x534 = x528;
bool x535 = x534 >= 2;
if (x535) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x540 = x534 == 0;
if (x540) {
int32_t x541 = x529;
bool x542 = x541 == 64;
if (x542) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x549 = x529;
bool x551 = x470 == 1;
int32_t x550 = 64 / x549;
bool x552 = x550 == 1;
bool x556;
if (x454) {
bool x553 = x551 || x552;
bool x554 = x470 == x550;
bool x555 = x553 || x554;
x556 = x555;
} else {
x556 = false;
}
bool x560;
if (x556) {
x560 = x559;
} else {
x560 = false;
}
bool x561;
if (x560) {
x561 = x559;
} else {
x561 = false;
}
if (x561) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x470,x472,x472,1,x550,1,1);
assert(false && "");
}
bool x567 = x470 <= x550;
int32_t x568;
if (x567) {
x568 = x550;
} else {
x568 = x470;
}
int32_t x572 = x568 * x571;
int32_t x573 = 64 * x572;
float* x574 = (float*)myMalloc(x573 * sizeof(float));;
int32_t x575;
if (x551) {
x575 = 0;
} else {
x575 = x473;
}
int32_t x578;
if (x552) {
x578 = 0;
} else {
x578 = 1;
}
for(int x579=0; x579 < 64; x579++) {
int32_t x591 = x474 * x579;
int32_t x585 = x572 * x579;
for(int x581=0; x581 < x568; x581++) {
int32_t x592 = x575 * x581;
int32_t x593 = x591 + x592;
int32_t x598 = x578 * x581;
int32_t x587 = x571 * x581;
for(int x583=0; x583 < x570; x583++) {
int32_t x594 = x576 * x583;
int32_t x595 = x593 + x594;
int32_t x589 = x570 * x583;
for(int x584=0; x584 < x570; x584++) {
int32_t x596 = x577 * x584;
int32_t x597 = x595 + x596;
float x599 = x476[x597];
float x600 = x519[x598];
int32_t x586 = x584 + x585;
int32_t x588 = x586 + x587;
int32_t x590 = x588 + x589;
float x601 = x599 / x600;
x574[x590] = x601;

}

}

}

}
int32_t x611 = 0;
int32_t x612 = 1;
x612 *= 1;
x611 += 1;
x612 *= 1;
x612 *= 1;
int32_t x617 = x611;
bool x618 = x617 >= 2;
if (x618) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x623 = x617 == 0;
if (x623) {
int32_t x624 = x612;
bool x625 = x624 == 64;
if (x625) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x632 = x612;
bool x634 = x568 == 1;
int32_t x633 = 64 / x632;
bool x635 = x633 == 1;
bool x639;
if (x454) {
bool x636 = x634 || x635;
bool x637 = x568 == x633;
bool x638 = x636 || x637;
x639 = x638;
} else {
x639 = false;
}
bool x643;
if (x639) {
x643 = x642;
} else {
x643 = false;
}
bool x644;
if (x643) {
x644 = x642;
} else {
x644 = false;
}
if (x644) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x568,x570,x570,1,x633,1,1);
assert(false && "");
}
bool x650 = x568 <= x633;
int32_t x651;
if (x650) {
x651 = x633;
} else {
x651 = x568;
}
int32_t x655 = x651 * x654;
int32_t x656 = 64 * x655;
float* x657 = (float*)myMalloc(x656 * sizeof(float));;
int32_t x658;
if (x634) {
x658 = 0;
} else {
x658 = x571;
}
int32_t x661;
if (x635) {
x661 = 0;
} else {
x661 = 1;
}
for(int x662=0; x662 < 64; x662++) {
int32_t x674 = x572 * x662;
int32_t x668 = x655 * x662;
for(int x664=0; x664 < x651; x664++) {
int32_t x675 = x658 * x664;
int32_t x676 = x674 + x675;
int32_t x681 = x661 * x664;
int32_t x670 = x654 * x664;
for(int x666=0; x666 < x653; x666++) {
int32_t x677 = x659 * x666;
int32_t x678 = x676 + x677;
int32_t x672 = x653 * x666;
for(int x667=0; x667 < x653; x667++) {
int32_t x679 = x660 * x667;
int32_t x680 = x678 + x679;
float x682 = x574[x680];
float x683 = x206[x681];
int32_t x669 = x667 + x668;
int32_t x671 = x669 + x670;
int32_t x673 = x671 + x672;
float x684 = x682 * x683;
x657[x673] = x684;

}

}

}

}
int32_t x694 = 0;
int32_t x695 = 1;
x695 *= 1;
x694 += 1;
x695 *= 1;
x695 *= 1;
int32_t x700 = x694;
bool x701 = x700 >= 2;
if (x701) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x706 = x700 == 0;
if (x706) {
int32_t x707 = x695;
bool x708 = x707 == 64;
if (x708) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x715 = x695;
bool x717 = x651 == 1;
int32_t x716 = 64 / x715;
bool x718 = x716 == 1;
bool x722;
if (x454) {
bool x719 = x717 || x718;
bool x720 = x651 == x716;
bool x721 = x719 || x720;
x722 = x721;
} else {
x722 = false;
}
bool x726;
if (x722) {
x726 = x725;
} else {
x726 = false;
}
bool x727;
if (x726) {
x727 = x725;
} else {
x727 = false;
}
if (x727) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x651,x653,x653,1,x716,1,1);
assert(false && "");
}
bool x733 = x651 <= x716;
int32_t x734;
if (x733) {
x734 = x716;
} else {
x734 = x651;
}
int32_t x738 = x734 * x737;
int32_t x739 = 64 * x738;
float* x740 = (float*)myMalloc(x739 * sizeof(float));;
int32_t x741;
if (x717) {
x741 = 0;
} else {
x741 = x654;
}
int32_t x744;
if (x718) {
x744 = 0;
} else {
x744 = 1;
}
for(int x745=0; x745 < 64; x745++) {
int32_t x757 = x655 * x745;
int32_t x751 = x738 * x745;
for(int x747=0; x747 < x734; x747++) {
int32_t x758 = x741 * x747;
int32_t x759 = x757 + x758;
int32_t x764 = x744 * x747;
int32_t x753 = x737 * x747;
for(int x749=0; x749 < x736; x749++) {
int32_t x760 = x742 * x749;
int32_t x761 = x759 + x760;
int32_t x755 = x736 * x749;
for(int x750=0; x750 < x736; x750++) {
int32_t x762 = x743 * x750;
int32_t x763 = x761 + x762;
float x765 = x657[x763];
float x766 = x251[x764];
int32_t x752 = x750 + x751;
int32_t x754 = x752 + x753;
int32_t x756 = x754 + x755;
float x767 = x765 + x766;
x740[x756] = x767;

}

}

}

}
float* x777 = (float*)myMalloc(x739 * sizeof(float));;
for(int x779=0; x779 < x739; x779++) {
float x780 = x740[x779];
bool x781 = x780 < 0.0f;
if (x781) {
x777[x779] = 0.0f;
} else {
float x784 = x740[x779];
x777[x779] = x784;
}

}
if (x791) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(64) x Sym(734) x Sym(736) x Sym(736)|(2,2)");
}
int32_t x802 = 64 * x734;
int32_t x803 = x802 * x798;
int32_t x804 = x803 * x798;
float* x805 = (float*)myMalloc(x804 * sizeof(float));;
for(int x807=0; x807 < x804; x807++) {
x805[x807] = -3.4028235E38f;

}
int32_t x800 = x734 * x799;
int32_t x801 = 64 * x800;
int* x811 = (int32_t*)myMalloc(x801 * sizeof(int32_t));;
for(int x812=0; x812 < 64; x812++) {
int32_t x813 = x812 * x738;
float* x814 = x777+x813;
int32_t x815 = x812 * x800;
float* x816 = x805+x815;
int* x817 = x811+x815;
int32_t x818 = 0;
int32_t x819 = 0;
for(int x820=0; x820 < x734; x820++) {
int32_t x821 = x818;
int32_t x822 = x821;
int32_t x823 = x819;
int32_t x824 = x823;
for(int x826=0; x826 < x798; x826++) {
int32_t x827 = x822;
int32_t x828 = x827;
int32_t x829 = x824;
int32_t x830 = x829;
for(int x831=0; x831 < x798; x831++) {
int32_t x832 = x830;
int32_t x833 = x832;
int32_t x834 = x833;
int32_t x835 = x834;
int32_t x836 = x835;
float x837 = x814[x836];
int32_t x838 = x828;
float x839 = x816[x838];
bool x840 = x837 > x839;
if (x840) {
float x841 = x814[x836];
x816[x838] = x841;
int32_t x843 = x836 + x813;
x817[x838] = x843;
} else {
}
x835 += 1;
int32_t x848 = x835;
float x849 = x814[x848];
float x850 = x816[x838];
bool x851 = x849 > x850;
if (x851) {
float x852 = x814[x848];
x816[x838] = x852;
int32_t x854 = x848 + x813;
x817[x838] = x854;
} else {
}
x835 += 1;
x833 += x736;
int32_t x860 = x833;
int32_t x861 = x860;
int32_t x862 = x861;
float x863 = x814[x862];
float x864 = x816[x838];
bool x865 = x863 > x864;
if (x865) {
float x866 = x814[x862];
x816[x838] = x866;
int32_t x868 = x862 + x813;
x817[x838] = x868;
} else {
}
x861 += 1;
int32_t x873 = x861;
float x874 = x814[x873];
float x875 = x816[x838];
bool x876 = x874 > x875;
if (x876) {
float x877 = x814[x873];
x816[x838] = x877;
int32_t x879 = x873 + x813;
x817[x838] = x879;
} else {
}
x861 += 1;
x833 += x736;
x828 += 1;
x830 += 2;

}
x822 += x798;
x824 += x890;

}
x818 += x799;
x819 += x737;

}

}
float* x907 = (float*)myMalloc(x906 * sizeof(float));;
int32_t x910 = x802 * x902;
float* x911 = (float*)myMalloc(x910 * sizeof(float));;
int32_t x908 = x734 * x902;
for(int x912=0; x912 < 64; x912++) {
int32_t x913 = x912 * x800;
float* x914 = x805+x913;
int32_t x915 = x912 * x903;
float* x916 = x907+x915;
int32_t x917 = x912 * x908;
float* x918 = x911+x917;
for(int x919=0; x919 < x734; x919++) {
int32_t x920 = x919 / 1;
int32_t x924 = x920 * x901;
int32_t x925 = x924 * x901;
int32_t x921 = x919 % 1;
int32_t x922 = x921 / 1;
int32_t x926 = x922 * x901;
int32_t x927 = x926 * x901;
int32_t x928 = x925 + x927;
int32_t x923 = x921 % 1;
int32_t x929 = x923 * x901;
int32_t x930 = x929 * x901;
int32_t x931 = x928 + x930;
float* x932 = x918+x931;
int32_t x933 = x920 * x798;
int32_t x934 = x933 * x798;
float* x935 = x914+x934;
for(int x937=0; x937 < x901; x937++) {
int32_t x939 = x937 * x901;
float* x940 = x932+x939;
int32_t x938 = x937 + x922;
int32_t x941 = x938 * x798;
int32_t x942 = x941 + x923;
float* x943 = x935+x942;
memcpy(x940, x943, 4 * x901);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x902,x734,1,x233,x734,x918,x902,1,x916,x902);

}
int32_t x952 = 0;
int32_t x953 = 1;
x953 *= 1;
x952 += 1;
x953 *= 1;
x953 *= 1;
int32_t x958 = x952;
bool x959 = x958 >= 2;
if (x959) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x964 = x958 == 0;
if (x964) {
int32_t x965 = x953;
bool x966 = x965 == 64;
if (x966) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x973 = x953;
int32_t x974 = 64 / x973;
bool x975 = x974 == 1;
bool x978;
if (x454) {
bool x976 = 64 == x974;
bool x977 = x975 || x976;
x978 = x977;
} else {
x978 = false;
}
bool x982;
if (x978) {
x982 = x981;
} else {
x982 = false;
}
bool x983;
if (x982) {
x983 = x981;
} else {
x983 = false;
}
if (x983) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x901,x901,1,x974,1,1);
assert(false && "");
}
bool x989 = 64 <= x974;
int32_t x990;
if (x989) {
x990 = x974;
} else {
x990 = 64;
}
int32_t x994 = x990 * x993;
int32_t x995 = 64 * x994;
float* x996 = (float*)myMalloc(x995 * sizeof(float));;
int32_t x999;
if (x975) {
x999 = 0;
} else {
x999 = 1;
}
for(int x1000=0; x1000 < 64; x1000++) {
int32_t x1012 = x903 * x1000;
int32_t x1006 = x994 * x1000;
for(int x1002=0; x1002 < x990; x1002++) {
int32_t x1013 = x902 * x1002;
int32_t x1014 = x1012 + x1013;
int32_t x1019 = x999 * x1002;
int32_t x1008 = x993 * x1002;
for(int x1004=0; x1004 < x992; x1004++) {
int32_t x1015 = x997 * x1004;
int32_t x1016 = x1014 + x1015;
int32_t x1010 = x992 * x1004;
for(int x1005=0; x1005 < x992; x1005++) {
int32_t x1017 = x998 * x1005;
int32_t x1018 = x1016 + x1017;
float x1020 = x907[x1018];
float x1021 = x114[x1019];
int32_t x1007 = x1005 + x1006;
int32_t x1009 = x1007 + x1008;
int32_t x1011 = x1009 + x1010;
float x1022 = x1020 - x1021;
x996[x1011] = x1022;

}

}

}

}
float* x1032 = (float*)myMalloc(64 * sizeof(float));;
for(int x1033=0; x1033 < 64; x1033++) {
float x1034 = x51[x1033];
float x1035 = x1034 + 1.0E-5f;
x1032[x1033] = x1035;

}
float* x1039 = (float*)myMalloc(64 * sizeof(float));;
for(int x1040=0; x1040 < 64; x1040++) {
float x1041 = x1032[x1040];
double x1042 = (double)x1041;
double x1043 = sqrt(x1042);
float x1044 = (float)x1043;
x1039[x1040] = x1044;

}
int32_t x1048 = 0;
int32_t x1049 = 1;
x1049 *= 1;
x1048 += 1;
x1049 *= 1;
x1049 *= 1;
int32_t x1054 = x1048;
bool x1055 = x1054 >= 2;
if (x1055) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1060 = x1054 == 0;
if (x1060) {
int32_t x1061 = x1049;
bool x1062 = x1061 == 64;
if (x1062) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1069 = x1049;
bool x1071 = x990 == 1;
int32_t x1070 = 64 / x1069;
bool x1072 = x1070 == 1;
bool x1076;
if (x454) {
bool x1073 = x1071 || x1072;
bool x1074 = x990 == x1070;
bool x1075 = x1073 || x1074;
x1076 = x1075;
} else {
x1076 = false;
}
bool x1080;
if (x1076) {
x1080 = x1079;
} else {
x1080 = false;
}
bool x1081;
if (x1080) {
x1081 = x1079;
} else {
x1081 = false;
}
if (x1081) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x990,x992,x992,1,x1070,1,1);
assert(false && "");
}
bool x1087 = x990 <= x1070;
int32_t x1088;
if (x1087) {
x1088 = x1070;
} else {
x1088 = x990;
}
int32_t x1092 = x1088 * x1091;
int32_t x1093 = 64 * x1092;
float* x1094 = (float*)myMalloc(x1093 * sizeof(float));;
int32_t x1095;
if (x1071) {
x1095 = 0;
} else {
x1095 = x993;
}
int32_t x1098;
if (x1072) {
x1098 = 0;
} else {
x1098 = 1;
}
for(int x1099=0; x1099 < 64; x1099++) {
int32_t x1111 = x994 * x1099;
int32_t x1105 = x1092 * x1099;
for(int x1101=0; x1101 < x1088; x1101++) {
int32_t x1112 = x1095 * x1101;
int32_t x1113 = x1111 + x1112;
int32_t x1118 = x1098 * x1101;
int32_t x1107 = x1091 * x1101;
for(int x1103=0; x1103 < x1090; x1103++) {
int32_t x1114 = x1096 * x1103;
int32_t x1115 = x1113 + x1114;
int32_t x1109 = x1090 * x1103;
for(int x1104=0; x1104 < x1090; x1104++) {
int32_t x1116 = x1097 * x1104;
int32_t x1117 = x1115 + x1116;
float x1119 = x996[x1117];
float x1120 = x1039[x1118];
int32_t x1106 = x1104 + x1105;
int32_t x1108 = x1106 + x1107;
int32_t x1110 = x1108 + x1109;
float x1121 = x1119 / x1120;
x1094[x1110] = x1121;

}

}

}

}
int32_t x1131 = 0;
int32_t x1132 = 1;
x1132 *= 1;
x1131 += 1;
x1132 *= 1;
x1132 *= 1;
int32_t x1137 = x1131;
bool x1138 = x1137 >= 2;
if (x1138) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1143 = x1137 == 0;
if (x1143) {
int32_t x1144 = x1132;
bool x1145 = x1144 == 64;
if (x1145) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1152 = x1132;
bool x1154 = x1088 == 1;
int32_t x1153 = 64 / x1152;
bool x1155 = x1153 == 1;
bool x1159;
if (x454) {
bool x1156 = x1154 || x1155;
bool x1157 = x1088 == x1153;
bool x1158 = x1156 || x1157;
x1159 = x1158;
} else {
x1159 = false;
}
bool x1163;
if (x1159) {
x1163 = x1162;
} else {
x1163 = false;
}
bool x1164;
if (x1163) {
x1164 = x1162;
} else {
x1164 = false;
}
if (x1164) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1088,x1090,x1090,1,x1153,1,1);
assert(false && "");
}
bool x1170 = x1088 <= x1153;
int32_t x1171;
if (x1170) {
x1171 = x1153;
} else {
x1171 = x1088;
}
int32_t x1175 = x1171 * x1174;
int32_t x1176 = 64 * x1175;
float* x1177 = (float*)myMalloc(x1176 * sizeof(float));;
int32_t x1178;
if (x1154) {
x1178 = 0;
} else {
x1178 = x1091;
}
int32_t x1181;
if (x1155) {
x1181 = 0;
} else {
x1181 = 1;
}
for(int x1182=0; x1182 < 64; x1182++) {
int32_t x1194 = x1092 * x1182;
int32_t x1188 = x1175 * x1182;
for(int x1184=0; x1184 < x1171; x1184++) {
int32_t x1195 = x1178 * x1184;
int32_t x1196 = x1194 + x1195;
int32_t x1201 = x1181 * x1184;
int32_t x1190 = x1174 * x1184;
for(int x1186=0; x1186 < x1173; x1186++) {
int32_t x1197 = x1179 * x1186;
int32_t x1198 = x1196 + x1197;
int32_t x1192 = x1173 * x1186;
for(int x1187=0; x1187 < x1173; x1187++) {
int32_t x1199 = x1180 * x1187;
int32_t x1200 = x1198 + x1199;
float x1202 = x1094[x1200];
float x1203 = x26[x1201];
int32_t x1189 = x1187 + x1188;
int32_t x1191 = x1189 + x1190;
int32_t x1193 = x1191 + x1192;
float x1204 = x1202 * x1203;
x1177[x1193] = x1204;

}

}

}

}
int32_t x1214 = 0;
int32_t x1215 = 1;
x1215 *= 1;
x1214 += 1;
x1215 *= 1;
x1215 *= 1;
int32_t x1220 = x1214;
bool x1221 = x1220 >= 2;
if (x1221) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1226 = x1220 == 0;
if (x1226) {
int32_t x1227 = x1215;
bool x1228 = x1227 == 64;
if (x1228) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1235 = x1215;
bool x1237 = x1171 == 1;
int32_t x1236 = 64 / x1235;
bool x1238 = x1236 == 1;
bool x1242;
if (x454) {
bool x1239 = x1237 || x1238;
bool x1240 = x1171 == x1236;
bool x1241 = x1239 || x1240;
x1242 = x1241;
} else {
x1242 = false;
}
bool x1246;
if (x1242) {
x1246 = x1245;
} else {
x1246 = false;
}
bool x1247;
if (x1246) {
x1247 = x1245;
} else {
x1247 = false;
}
if (x1247) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1171,x1173,x1173,1,x1236,1,1);
assert(false && "");
}
bool x1253 = x1171 <= x1236;
int32_t x1254;
if (x1253) {
x1254 = x1236;
} else {
x1254 = x1171;
}
int32_t x1258 = x1254 * x1257;
int32_t x1259 = 64 * x1258;
float* x1260 = (float*)myMalloc(x1259 * sizeof(float));;
int32_t x1261;
if (x1237) {
x1261 = 0;
} else {
x1261 = x1174;
}
int32_t x1264;
if (x1238) {
x1264 = 0;
} else {
x1264 = 1;
}
for(int x1265=0; x1265 < 64; x1265++) {
int32_t x1277 = x1175 * x1265;
int32_t x1271 = x1258 * x1265;
for(int x1267=0; x1267 < x1254; x1267++) {
int32_t x1278 = x1261 * x1267;
int32_t x1279 = x1277 + x1278;
int32_t x1284 = x1264 * x1267;
int32_t x1273 = x1257 * x1267;
for(int x1269=0; x1269 < x1256; x1269++) {
int32_t x1280 = x1262 * x1269;
int32_t x1281 = x1279 + x1280;
int32_t x1275 = x1256 * x1269;
for(int x1270=0; x1270 < x1256; x1270++) {
int32_t x1282 = x1263 * x1270;
int32_t x1283 = x1281 + x1282;
float x1285 = x1177[x1283];
float x1286 = x53[x1284];
int32_t x1272 = x1270 + x1271;
int32_t x1274 = x1272 + x1273;
int32_t x1276 = x1274 + x1275;
float x1287 = x1285 + x1286;
x1260[x1276] = x1287;

}

}

}

}
float* x1297 = (float*)myMalloc(x1259 * sizeof(float));;
for(int x1299=0; x1299 < x1259; x1299++) {
float x1300 = x1260[x1299];
bool x1301 = x1300 < 0.0f;
if (x1301) {
x1297[x1299] = 0.0f;
} else {
float x1304 = x1260[x1299];
x1297[x1299] = x1304;
}

}
float* x1319 = (float*)myMalloc(x1318 * sizeof(float));;
int32_t x1320 = 9 * x1254;
int32_t x1323 = 64 * x1320;
int32_t x1324 = x1323 * x1314;
float* x1325 = (float*)myMalloc(x1324 * sizeof(float));;
int32_t x1321 = x1320 * x1314;
int32_t x1333 = x1254 * 3;
int32_t x1334 = x1333 * 3;
for(int x1326=0; x1326 < 64; x1326++) {
int32_t x1327 = x1326 * x1258;
float* x1328 = x1297+x1327;
int32_t x1329 = x1326 * x1315;
float* x1330 = x1319+x1329;
int32_t x1331 = x1326 * x1321;
float* x1332 = x1325+x1331;
for(int x1336=0; x1336 < x1334; x1336++) {
int32_t x1337 = x1336 / 9;
int32_t x1341 = x1337 * 3;
int32_t x1342 = x1341 * 3;
int32_t x1343 = x1342 * x1313;
int32_t x1344 = x1343 * x1313;
int32_t x1338 = x1336 % 9;
int32_t x1339 = x1338 / 3;
int32_t x1345 = x1339 * 3;
int32_t x1346 = x1345 * x1313;
int32_t x1347 = x1346 * x1313;
int32_t x1348 = x1344 + x1347;
int32_t x1340 = x1338 % 3;
int32_t x1349 = x1340 * x1313;
int32_t x1350 = x1349 * x1313;
int32_t x1351 = x1348 + x1350;
float* x1352 = x1332+x1351;
int32_t x1353 = x1337 * x1256;
int32_t x1354 = x1353 * x1256;
float* x1355 = x1328+x1354;
int32_t x1368 = 1 - x1340;
bool x1369 = x1368 > 0;
int32_t x1370;
if (x1369) {
x1370 = x1368;
} else {
x1370 = 0;
}
int32_t x1371 = 3 - x1340;
int32_t x1372 = x1371 - 1;
int32_t x1373 = 1 - x1372;
bool x1374 = x1373 > 0;
int32_t x1375;
if (x1374) {
x1375 = x1373;
} else {
x1375 = 0;
}
int32_t x1376 = x1313 - x1375;
int32_t x1377 = x1376 - x1370;
bool x1378 = x1377 <= 0;
bool x1382 = x1370 > 0;
int32_t x1367 = -1 + x1340;
bool x1395 = x1375 > 0;
for(int x1357=0; x1357 < x1313; x1357++) {
int32_t x1358 = x1357 - 1;
int32_t x1359 = x1358 + x1339;
bool x1360 = x1359 < 0;
bool x1361 = x1359 >= x1256;
bool x1362 = x1360 || x1361;
if (x1362) {
int32_t x1363 = x1357 * x1313;
float* x1364 = x1352+x1363;
memset(x1364, 0, 4 * x1313);;
} else {
if (x1378) {
int32_t x1363 = x1357 * x1313;
float* x1379 = x1352+x1363;
memset(x1379, 0, 4 * x1313);;
} else {
int32_t x1363 = x1357 * x1313;
if (x1382) {
float* x1383 = x1352+x1363;
memset(x1383, 0, 4 * x1370);;
} else {
}
// may have segfault here
int32_t x1388 = x1363 + x1370;
float* x1389 = x1352+x1388;
int32_t x1390 = x1359 * x1256;
int32_t x1391 = x1390 + x1367;
int32_t x1392 = x1391 + x1370;
float* x1393 = x1355+x1392;
memcpy(x1389, x1393, 4 * x1377);;
if (x1395) {
int32_t x1396 = x1363 + x1313;
int32_t x1397 = x1396 - x1375;
float* x1398 = x1352+x1397;
memset(x1398, 0, 4 * x1375);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1314,x1320,1,x90,x1320,x1332,x1314,1,x1330,x1314);

}
int32_t x1413 = 0;
int32_t x1414 = 1;
x1414 *= 1;
x1413 += 1;
x1414 *= 1;
x1414 *= 1;
int32_t x1419 = x1413;
bool x1420 = x1419 >= 2;
if (x1420) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1425 = x1419 == 0;
if (x1425) {
int32_t x1426 = x1414;
bool x1427 = x1426 == 64;
if (x1427) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1434 = x1414;
int32_t x1435 = 64 / x1434;
bool x1436 = x1435 == 1;
bool x1439;
if (x454) {
bool x1437 = 64 == x1435;
bool x1438 = x1436 || x1437;
x1439 = x1438;
} else {
x1439 = false;
}
bool x1443;
if (x1439) {
x1443 = x1442;
} else {
x1443 = false;
}
bool x1444;
if (x1443) {
x1444 = x1442;
} else {
x1444 = false;
}
if (x1444) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x1313,x1313,1,x1435,1,1);
assert(false && "");
}
bool x1450 = 64 <= x1435;
int32_t x1451;
if (x1450) {
x1451 = x1435;
} else {
x1451 = 64;
}
int32_t x1455 = x1451 * x1454;
int32_t x1456 = 64 * x1455;
float* x1457 = (float*)myMalloc(x1456 * sizeof(float));;
int32_t x1460;
if (x1436) {
x1460 = 0;
} else {
x1460 = 1;
}
for(int x1461=0; x1461 < 64; x1461++) {
int32_t x1473 = x1315 * x1461;
int32_t x1467 = x1455 * x1461;
for(int x1463=0; x1463 < x1451; x1463++) {
int32_t x1474 = x1314 * x1463;
int32_t x1475 = x1473 + x1474;
int32_t x1480 = x1460 * x1463;
int32_t x1469 = x1454 * x1463;
for(int x1465=0; x1465 < x1453; x1465++) {
int32_t x1476 = x1458 * x1465;
int32_t x1477 = x1475 + x1476;
int32_t x1471 = x1453 * x1465;
for(int x1466=0; x1466 < x1453; x1466++) {
int32_t x1478 = x1459 * x1466;
int32_t x1479 = x1477 + x1478;
float x1481 = x1319[x1479];
float x1482 = x105[x1480];
int32_t x1468 = x1466 + x1467;
int32_t x1470 = x1468 + x1469;
int32_t x1472 = x1470 + x1471;
float x1483 = x1481 - x1482;
x1457[x1472] = x1483;

}

}

}

}
float* x1493 = (float*)myMalloc(64 * sizeof(float));;
for(int x1494=0; x1494 < 64; x1494++) {
float x1495 = x158[x1494];
float x1496 = x1495 + 1.0E-5f;
x1493[x1494] = x1496;

}
float* x1500 = (float*)myMalloc(64 * sizeof(float));;
for(int x1501=0; x1501 < 64; x1501++) {
float x1502 = x1493[x1501];
double x1503 = (double)x1502;
double x1504 = sqrt(x1503);
float x1505 = (float)x1504;
x1500[x1501] = x1505;

}
int32_t x1509 = 0;
int32_t x1510 = 1;
x1510 *= 1;
x1509 += 1;
x1510 *= 1;
x1510 *= 1;
int32_t x1515 = x1509;
bool x1516 = x1515 >= 2;
if (x1516) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1521 = x1515 == 0;
if (x1521) {
int32_t x1522 = x1510;
bool x1523 = x1522 == 64;
if (x1523) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1530 = x1510;
bool x1532 = x1451 == 1;
int32_t x1531 = 64 / x1530;
bool x1533 = x1531 == 1;
bool x1537;
if (x454) {
bool x1534 = x1532 || x1533;
bool x1535 = x1451 == x1531;
bool x1536 = x1534 || x1535;
x1537 = x1536;
} else {
x1537 = false;
}
bool x1541;
if (x1537) {
x1541 = x1540;
} else {
x1541 = false;
}
bool x1542;
if (x1541) {
x1542 = x1540;
} else {
x1542 = false;
}
if (x1542) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1451,x1453,x1453,1,x1531,1,1);
assert(false && "");
}
bool x1548 = x1451 <= x1531;
int32_t x1549;
if (x1548) {
x1549 = x1531;
} else {
x1549 = x1451;
}
int32_t x1553 = x1549 * x1552;
int32_t x1554 = 64 * x1553;
float* x1555 = (float*)myMalloc(x1554 * sizeof(float));;
int32_t x1556;
if (x1532) {
x1556 = 0;
} else {
x1556 = x1454;
}
int32_t x1559;
if (x1533) {
x1559 = 0;
} else {
x1559 = 1;
}
for(int x1560=0; x1560 < 64; x1560++) {
int32_t x1572 = x1455 * x1560;
int32_t x1566 = x1553 * x1560;
for(int x1562=0; x1562 < x1549; x1562++) {
int32_t x1573 = x1556 * x1562;
int32_t x1574 = x1572 + x1573;
int32_t x1579 = x1559 * x1562;
int32_t x1568 = x1552 * x1562;
for(int x1564=0; x1564 < x1551; x1564++) {
int32_t x1575 = x1557 * x1564;
int32_t x1576 = x1574 + x1575;
int32_t x1570 = x1551 * x1564;
for(int x1565=0; x1565 < x1551; x1565++) {
int32_t x1577 = x1558 * x1565;
int32_t x1578 = x1576 + x1577;
float x1580 = x1457[x1578];
float x1581 = x1500[x1579];
int32_t x1567 = x1565 + x1566;
int32_t x1569 = x1567 + x1568;
int32_t x1571 = x1569 + x1570;
float x1582 = x1580 / x1581;
x1555[x1571] = x1582;

}

}

}

}
int32_t x1592 = 0;
int32_t x1593 = 1;
x1593 *= 1;
x1592 += 1;
x1593 *= 1;
x1593 *= 1;
int32_t x1598 = x1592;
bool x1599 = x1598 >= 2;
if (x1599) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1604 = x1598 == 0;
if (x1604) {
int32_t x1605 = x1593;
bool x1606 = x1605 == 64;
if (x1606) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1613 = x1593;
bool x1615 = x1549 == 1;
int32_t x1614 = 64 / x1613;
bool x1616 = x1614 == 1;
bool x1620;
if (x454) {
bool x1617 = x1615 || x1616;
bool x1618 = x1549 == x1614;
bool x1619 = x1617 || x1618;
x1620 = x1619;
} else {
x1620 = false;
}
bool x1624;
if (x1620) {
x1624 = x1623;
} else {
x1624 = false;
}
bool x1625;
if (x1624) {
x1625 = x1623;
} else {
x1625 = false;
}
if (x1625) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1549,x1551,x1551,1,x1614,1,1);
assert(false && "");
}
bool x1631 = x1549 <= x1614;
int32_t x1632;
if (x1631) {
x1632 = x1614;
} else {
x1632 = x1549;
}
int32_t x1636 = x1632 * x1635;
int32_t x1637 = 64 * x1636;
float* x1638 = (float*)myMalloc(x1637 * sizeof(float));;
int32_t x1639;
if (x1615) {
x1639 = 0;
} else {
x1639 = x1552;
}
int32_t x1642;
if (x1616) {
x1642 = 0;
} else {
x1642 = 1;
}
for(int x1643=0; x1643 < 64; x1643++) {
int32_t x1655 = x1553 * x1643;
int32_t x1649 = x1636 * x1643;
for(int x1645=0; x1645 < x1632; x1645++) {
int32_t x1656 = x1639 * x1645;
int32_t x1657 = x1655 + x1656;
int32_t x1662 = x1642 * x1645;
int32_t x1651 = x1635 * x1645;
for(int x1647=0; x1647 < x1634; x1647++) {
int32_t x1658 = x1640 * x1647;
int32_t x1659 = x1657 + x1658;
int32_t x1653 = x1634 * x1647;
for(int x1648=0; x1648 < x1634; x1648++) {
int32_t x1660 = x1641 * x1648;
int32_t x1661 = x1659 + x1660;
float x1663 = x1555[x1661];
float x1664 = x164[x1662];
int32_t x1650 = x1648 + x1649;
int32_t x1652 = x1650 + x1651;
int32_t x1654 = x1652 + x1653;
float x1665 = x1663 * x1664;
x1638[x1654] = x1665;

}

}

}

}
int32_t x1675 = 0;
int32_t x1676 = 1;
x1676 *= 1;
x1675 += 1;
x1676 *= 1;
x1676 *= 1;
int32_t x1681 = x1675;
bool x1682 = x1681 >= 2;
if (x1682) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1687 = x1681 == 0;
if (x1687) {
int32_t x1688 = x1676;
bool x1689 = x1688 == 64;
if (x1689) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1696 = x1676;
bool x1698 = x1632 == 1;
int32_t x1697 = 64 / x1696;
bool x1699 = x1697 == 1;
bool x1703;
if (x454) {
bool x1700 = x1698 || x1699;
bool x1701 = x1632 == x1697;
bool x1702 = x1700 || x1701;
x1703 = x1702;
} else {
x1703 = false;
}
bool x1707;
if (x1703) {
x1707 = x1706;
} else {
x1707 = false;
}
bool x1708;
if (x1707) {
x1708 = x1706;
} else {
x1708 = false;
}
if (x1708) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1632,x1634,x1634,1,x1697,1,1);
assert(false && "");
}
bool x1714 = x1632 <= x1697;
int32_t x1715;
if (x1714) {
x1715 = x1697;
} else {
x1715 = x1632;
}
int32_t x1719 = x1715 * x1718;
int32_t x1720 = 64 * x1719;
float* x1721 = (float*)myMalloc(x1720 * sizeof(float));;
int32_t x1722;
if (x1698) {
x1722 = 0;
} else {
x1722 = x1635;
}
int32_t x1725;
if (x1699) {
x1725 = 0;
} else {
x1725 = 1;
}
for(int x1726=0; x1726 < 64; x1726++) {
int32_t x1738 = x1636 * x1726;
int32_t x1732 = x1719 * x1726;
for(int x1728=0; x1728 < x1715; x1728++) {
int32_t x1739 = x1722 * x1728;
int32_t x1740 = x1738 + x1739;
int32_t x1745 = x1725 * x1728;
int32_t x1734 = x1718 * x1728;
for(int x1730=0; x1730 < x1717; x1730++) {
int32_t x1741 = x1723 * x1730;
int32_t x1742 = x1740 + x1741;
int32_t x1736 = x1717 * x1730;
for(int x1731=0; x1731 < x1717; x1731++) {
int32_t x1743 = x1724 * x1731;
int32_t x1744 = x1742 + x1743;
float x1746 = x1638[x1744];
float x1747 = x49[x1745];
int32_t x1733 = x1731 + x1732;
int32_t x1735 = x1733 + x1734;
int32_t x1737 = x1735 + x1736;
float x1748 = x1746 + x1747;
x1721[x1737] = x1748;

}

}

}

}
float* x1758 = (float*)myMalloc(x1720 * sizeof(float));;
for(int x1760=0; x1760 < x1720; x1760++) {
float x1761 = x1721[x1760];
bool x1762 = x1761 < 0.0f;
if (x1762) {
x1758[x1760] = 0.0f;
} else {
float x1765 = x1721[x1760];
x1758[x1760] = x1765;
}

}
float* x1779 = (float*)myMalloc(x1778 * sizeof(float));;
int32_t x1782 = 64 * x1715;
int32_t x1783 = x1782 * x1774;
float* x1784 = (float*)myMalloc(x1783 * sizeof(float));;
int32_t x1780 = x1715 * x1774;
for(int x1785=0; x1785 < 64; x1785++) {
int32_t x1786 = x1785 * x1719;
float* x1787 = x1758+x1786;
int32_t x1788 = x1785 * x1775;
float* x1789 = x1779+x1788;
int32_t x1790 = x1785 * x1780;
float* x1791 = x1784+x1790;
for(int x1792=0; x1792 < x1715; x1792++) {
int32_t x1793 = x1792 / 1;
int32_t x1797 = x1793 * x1773;
int32_t x1798 = x1797 * x1773;
int32_t x1794 = x1792 % 1;
int32_t x1795 = x1794 / 1;
int32_t x1799 = x1795 * x1773;
int32_t x1800 = x1799 * x1773;
int32_t x1801 = x1798 + x1800;
int32_t x1796 = x1794 % 1;
int32_t x1802 = x1796 * x1773;
int32_t x1803 = x1802 * x1773;
int32_t x1804 = x1801 + x1803;
float* x1805 = x1791+x1804;
int32_t x1806 = x1793 * x1717;
int32_t x1807 = x1806 * x1717;
float* x1808 = x1787+x1807;
for(int x1810=0; x1810 < x1773; x1810++) {
int32_t x1812 = x1810 * x1773;
float* x1813 = x1805+x1812;
int32_t x1811 = x1810 + x1795;
int32_t x1814 = x1811 * x1717;
int32_t x1815 = x1814 + x1796;
float* x1816 = x1808+x1815;
memcpy(x1813, x1816, 4 * x1773);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1774,x1715,1,x32,x1715,x1791,x1774,1,x1789,x1774);

}
int32_t x1825 = 0;
int32_t x1826 = 1;
x1826 *= 1;
x1825 += 1;
x1826 *= 1;
x1826 *= 1;
int32_t x1831 = x1825;
bool x1832 = x1831 >= 2;
if (x1832) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1837 = x1831 == 0;
if (x1837) {
int32_t x1838 = x1826;
bool x1839 = x1838 == 256;
if (x1839) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1846 = x1826;
int32_t x1847 = 256 / x1846;
bool x1848 = x1847 == 1;
bool x1851;
if (x454) {
bool x1849 = 256 == x1847;
bool x1850 = x1848 || x1849;
x1851 = x1850;
} else {
x1851 = false;
}
bool x1855;
if (x1851) {
x1855 = x1854;
} else {
x1855 = false;
}
bool x1856;
if (x1855) {
x1856 = x1854;
} else {
x1856 = false;
}
if (x1856) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x1773,x1773,1,x1847,1,1);
assert(false && "");
}
bool x1862 = 256 <= x1847;
int32_t x1863;
if (x1862) {
x1863 = x1847;
} else {
x1863 = 256;
}
int32_t x1867 = x1863 * x1866;
int32_t x1868 = 64 * x1867;
float* x1869 = (float*)myMalloc(x1868 * sizeof(float));;
int32_t x1872;
if (x1848) {
x1872 = 0;
} else {
x1872 = 1;
}
for(int x1873=0; x1873 < 64; x1873++) {
int32_t x1885 = x1775 * x1873;
int32_t x1879 = x1867 * x1873;
for(int x1875=0; x1875 < x1863; x1875++) {
int32_t x1886 = x1774 * x1875;
int32_t x1887 = x1885 + x1886;
int32_t x1892 = x1872 * x1875;
int32_t x1881 = x1866 * x1875;
for(int x1877=0; x1877 < x1865; x1877++) {
int32_t x1888 = x1870 * x1877;
int32_t x1889 = x1887 + x1888;
int32_t x1883 = x1865 * x1877;
for(int x1878=0; x1878 < x1865; x1878++) {
int32_t x1890 = x1871 * x1878;
int32_t x1891 = x1889 + x1890;
float x1893 = x1779[x1891];
float x1894 = x71[x1892];
int32_t x1880 = x1878 + x1879;
int32_t x1882 = x1880 + x1881;
int32_t x1884 = x1882 + x1883;
float x1895 = x1893 - x1894;
x1869[x1884] = x1895;

}

}

}

}
float* x1905 = (float*)myMalloc(256 * sizeof(float));;
for(int x1907=0; x1907 < 256; x1907++) {
float x1908 = x36[x1907];
float x1909 = x1908 + 1.0E-5f;
x1905[x1907] = x1909;

}
float* x1913 = (float*)myMalloc(256 * sizeof(float));;
for(int x1914=0; x1914 < 256; x1914++) {
float x1915 = x1905[x1914];
double x1916 = (double)x1915;
double x1917 = sqrt(x1916);
float x1918 = (float)x1917;
x1913[x1914] = x1918;

}
int32_t x1922 = 0;
int32_t x1923 = 1;
x1923 *= 1;
x1922 += 1;
x1923 *= 1;
x1923 *= 1;
int32_t x1928 = x1922;
bool x1929 = x1928 >= 2;
if (x1929) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1934 = x1928 == 0;
if (x1934) {
int32_t x1935 = x1923;
bool x1936 = x1935 == 256;
if (x1936) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1943 = x1923;
bool x1945 = x1863 == 1;
int32_t x1944 = 256 / x1943;
bool x1946 = x1944 == 1;
bool x1950;
if (x454) {
bool x1947 = x1945 || x1946;
bool x1948 = x1863 == x1944;
bool x1949 = x1947 || x1948;
x1950 = x1949;
} else {
x1950 = false;
}
bool x1954;
if (x1950) {
x1954 = x1953;
} else {
x1954 = false;
}
bool x1955;
if (x1954) {
x1955 = x1953;
} else {
x1955 = false;
}
if (x1955) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1863,x1865,x1865,1,x1944,1,1);
assert(false && "");
}
bool x1961 = x1863 <= x1944;
int32_t x1962;
if (x1961) {
x1962 = x1944;
} else {
x1962 = x1863;
}
int32_t x1966 = x1962 * x1965;
int32_t x1967 = 64 * x1966;
float* x1968 = (float*)myMalloc(x1967 * sizeof(float));;
int32_t x1969;
if (x1945) {
x1969 = 0;
} else {
x1969 = x1866;
}
int32_t x1972;
if (x1946) {
x1972 = 0;
} else {
x1972 = 1;
}
for(int x1973=0; x1973 < 64; x1973++) {
int32_t x1985 = x1867 * x1973;
int32_t x1979 = x1966 * x1973;
for(int x1975=0; x1975 < x1962; x1975++) {
int32_t x1986 = x1969 * x1975;
int32_t x1987 = x1985 + x1986;
int32_t x1992 = x1972 * x1975;
int32_t x1981 = x1965 * x1975;
for(int x1977=0; x1977 < x1964; x1977++) {
int32_t x1988 = x1970 * x1977;
int32_t x1989 = x1987 + x1988;
int32_t x1983 = x1964 * x1977;
for(int x1978=0; x1978 < x1964; x1978++) {
int32_t x1990 = x1971 * x1978;
int32_t x1991 = x1989 + x1990;
float x1993 = x1869[x1991];
float x1994 = x1913[x1992];
int32_t x1980 = x1978 + x1979;
int32_t x1982 = x1980 + x1981;
int32_t x1984 = x1982 + x1983;
float x1995 = x1993 / x1994;
x1968[x1984] = x1995;

}

}

}

}
int32_t x2005 = 0;
int32_t x2006 = 1;
x2006 *= 1;
x2005 += 1;
x2006 *= 1;
x2006 *= 1;
int32_t x2011 = x2005;
bool x2012 = x2011 >= 2;
if (x2012) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2017 = x2011 == 0;
if (x2017) {
int32_t x2018 = x2006;
bool x2019 = x2018 == 256;
if (x2019) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2026 = x2006;
bool x2028 = x1962 == 1;
int32_t x2027 = 256 / x2026;
bool x2029 = x2027 == 1;
bool x2033;
if (x454) {
bool x2030 = x2028 || x2029;
bool x2031 = x1962 == x2027;
bool x2032 = x2030 || x2031;
x2033 = x2032;
} else {
x2033 = false;
}
bool x2037;
if (x2033) {
x2037 = x2036;
} else {
x2037 = false;
}
bool x2038;
if (x2037) {
x2038 = x2036;
} else {
x2038 = false;
}
if (x2038) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1962,x1964,x1964,1,x2027,1,1);
assert(false && "");
}
bool x2044 = x1962 <= x2027;
int32_t x2045;
if (x2044) {
x2045 = x2027;
} else {
x2045 = x1962;
}
int32_t x2049 = x2045 * x2048;
int32_t x2050 = 64 * x2049;
float* x2051 = (float*)myMalloc(x2050 * sizeof(float));;
int32_t x2052;
if (x2028) {
x2052 = 0;
} else {
x2052 = x1965;
}
int32_t x2055;
if (x2029) {
x2055 = 0;
} else {
x2055 = 1;
}
for(int x2056=0; x2056 < 64; x2056++) {
int32_t x2068 = x1966 * x2056;
int32_t x2062 = x2049 * x2056;
for(int x2058=0; x2058 < x2045; x2058++) {
int32_t x2069 = x2052 * x2058;
int32_t x2070 = x2068 + x2069;
int32_t x2075 = x2055 * x2058;
int32_t x2064 = x2048 * x2058;
for(int x2060=0; x2060 < x2047; x2060++) {
int32_t x2071 = x2053 * x2060;
int32_t x2072 = x2070 + x2071;
int32_t x2066 = x2047 * x2060;
for(int x2061=0; x2061 < x2047; x2061++) {
int32_t x2073 = x2054 * x2061;
int32_t x2074 = x2072 + x2073;
float x2076 = x1968[x2074];
float x2077 = x199[x2075];
int32_t x2063 = x2061 + x2062;
int32_t x2065 = x2063 + x2064;
int32_t x2067 = x2065 + x2066;
float x2078 = x2076 * x2077;
x2051[x2067] = x2078;

}

}

}

}
int32_t x2088 = 0;
int32_t x2089 = 1;
x2089 *= 1;
x2088 += 1;
x2089 *= 1;
x2089 *= 1;
int32_t x2094 = x2088;
bool x2095 = x2094 >= 2;
if (x2095) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2100 = x2094 == 0;
if (x2100) {
int32_t x2101 = x2089;
bool x2102 = x2101 == 256;
if (x2102) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2109 = x2089;
bool x2111 = x2045 == 1;
int32_t x2110 = 256 / x2109;
bool x2112 = x2110 == 1;
bool x2116;
if (x454) {
bool x2113 = x2111 || x2112;
bool x2114 = x2045 == x2110;
bool x2115 = x2113 || x2114;
x2116 = x2115;
} else {
x2116 = false;
}
bool x2120;
if (x2116) {
x2120 = x2119;
} else {
x2120 = false;
}
bool x2121;
if (x2120) {
x2121 = x2119;
} else {
x2121 = false;
}
if (x2121) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2045,x2047,x2047,1,x2110,1,1);
assert(false && "");
}
bool x2127 = x2045 <= x2110;
int32_t x2128;
if (x2127) {
x2128 = x2110;
} else {
x2128 = x2045;
}
int32_t x2132 = x2128 * x2131;
int32_t x2133 = 64 * x2132;
float* x2134 = (float*)myMalloc(x2133 * sizeof(float));;
int32_t x2135;
if (x2111) {
x2135 = 0;
} else {
x2135 = x2048;
}
int32_t x2138;
if (x2112) {
x2138 = 0;
} else {
x2138 = 1;
}
for(int x2139=0; x2139 < 64; x2139++) {
int32_t x2151 = x2049 * x2139;
int32_t x2145 = x2132 * x2139;
for(int x2141=0; x2141 < x2128; x2141++) {
int32_t x2152 = x2135 * x2141;
int32_t x2153 = x2151 + x2152;
int32_t x2158 = x2138 * x2141;
int32_t x2147 = x2131 * x2141;
for(int x2143=0; x2143 < x2130; x2143++) {
int32_t x2154 = x2136 * x2143;
int32_t x2155 = x2153 + x2154;
int32_t x2149 = x2130 * x2143;
for(int x2144=0; x2144 < x2130; x2144++) {
int32_t x2156 = x2137 * x2144;
int32_t x2157 = x2155 + x2156;
float x2159 = x2051[x2157];
float x2160 = x126[x2158];
int32_t x2146 = x2144 + x2145;
int32_t x2148 = x2146 + x2147;
int32_t x2150 = x2148 + x2149;
float x2161 = x2159 + x2160;
x2134[x2150] = x2161;

}

}

}

}
float* x2175 = (float*)myMalloc(x2174 * sizeof(float));;
float* x2176 = (float*)myMalloc(x910 * sizeof(float));;
for(int x2177=0; x2177 < 64; x2177++) {
int32_t x2178 = x2177 * x800;
float* x2179 = x805+x2178;
int32_t x2180 = x2177 * x2171;
float* x2181 = x2175+x2180;
int32_t x2182 = x2177 * x908;
float* x2183 = x2176+x2182;
for(int x2184=0; x2184 < x734; x2184++) {
int32_t x2185 = x2184 / 1;
int32_t x2189 = x2185 * x901;
int32_t x2190 = x2189 * x901;
int32_t x2186 = x2184 % 1;
int32_t x2187 = x2186 / 1;
int32_t x2191 = x2187 * x901;
int32_t x2192 = x2191 * x901;
int32_t x2193 = x2190 + x2192;
int32_t x2188 = x2186 % 1;
int32_t x2194 = x2188 * x901;
int32_t x2195 = x2194 * x901;
int32_t x2196 = x2193 + x2195;
float* x2197 = x2183+x2196;
int32_t x2198 = x2185 * x798;
int32_t x2199 = x2198 * x798;
float* x2200 = x2179+x2199;
for(int x2201=0; x2201 < x901; x2201++) {
int32_t x2203 = x2201 * x901;
float* x2204 = x2197+x2203;
int32_t x2202 = x2201 + x2187;
int32_t x2205 = x2202 * x798;
int32_t x2206 = x2205 + x2188;
float* x2207 = x2200+x2206;
memcpy(x2204, x2207, 4 * x901);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x902,x734,1,x162,x734,x2183,x902,1,x2181,x902);

}
int32_t x2216 = 0;
int32_t x2217 = 1;
x2217 *= 1;
x2216 += 1;
x2217 *= 1;
x2217 *= 1;
int32_t x2222 = x2216;
bool x2223 = x2222 >= 2;
if (x2223) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2228 = x2222 == 0;
if (x2228) {
int32_t x2229 = x2217;
bool x2230 = x2229 == 256;
if (x2230) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2237 = x2217;
int32_t x2238 = 256 / x2237;
bool x2239 = x2238 == 1;
bool x2242;
if (x454) {
bool x2240 = 256 == x2238;
bool x2241 = x2239 || x2240;
x2242 = x2241;
} else {
x2242 = false;
}
bool x2243;
if (x2242) {
x2243 = x981;
} else {
x2243 = false;
}
bool x2244;
if (x2243) {
x2244 = x981;
} else {
x2244 = false;
}
if (x2244) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x901,x901,1,x2238,1,1);
assert(false && "");
}
bool x2250 = 256 <= x2238;
int32_t x2251;
if (x2250) {
x2251 = x2238;
} else {
x2251 = 256;
}
int32_t x2252 = x2251 * x993;
int32_t x2253 = 64 * x2252;
float* x2254 = (float*)myMalloc(x2253 * sizeof(float));;
int32_t x2255;
if (x2239) {
x2255 = 0;
} else {
x2255 = 1;
}
for(int x2256=0; x2256 < 64; x2256++) {
int32_t x2267 = x2171 * x2256;
int32_t x2261 = x2252 * x2256;
for(int x2258=0; x2258 < x2251; x2258++) {
int32_t x2268 = x902 * x2258;
int32_t x2269 = x2267 + x2268;
int32_t x2274 = x2255 * x2258;
int32_t x2263 = x993 * x2258;
for(int x2259=0; x2259 < x992; x2259++) {
int32_t x2270 = x997 * x2259;
int32_t x2271 = x2269 + x2270;
int32_t x2265 = x992 * x2259;
for(int x2260=0; x2260 < x992; x2260++) {
int32_t x2272 = x998 * x2260;
int32_t x2273 = x2271 + x2272;
float x2275 = x2175[x2273];
float x2276 = x264[x2274];
int32_t x2262 = x2260 + x2261;
int32_t x2264 = x2262 + x2263;
int32_t x2266 = x2264 + x2265;
float x2277 = x2275 - x2276;
x2254[x2266] = x2277;

}

}

}

}
float* x2287 = (float*)myMalloc(256 * sizeof(float));;
for(int x2288=0; x2288 < 256; x2288++) {
float x2289 = x243[x2288];
float x2290 = x2289 + 1.0E-5f;
x2287[x2288] = x2290;

}
float* x2294 = (float*)myMalloc(256 * sizeof(float));;
for(int x2295=0; x2295 < 256; x2295++) {
float x2296 = x2287[x2295];
double x2297 = (double)x2296;
double x2298 = sqrt(x2297);
float x2299 = (float)x2298;
x2294[x2295] = x2299;

}
int32_t x2303 = 0;
int32_t x2304 = 1;
x2304 *= 1;
x2303 += 1;
x2304 *= 1;
x2304 *= 1;
int32_t x2309 = x2303;
bool x2310 = x2309 >= 2;
if (x2310) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2315 = x2309 == 0;
if (x2315) {
int32_t x2316 = x2304;
bool x2317 = x2316 == 256;
if (x2317) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2324 = x2304;
bool x2326 = x2251 == 1;
int32_t x2325 = 256 / x2324;
bool x2327 = x2325 == 1;
bool x2331;
if (x454) {
bool x2328 = x2326 || x2327;
bool x2329 = x2251 == x2325;
bool x2330 = x2328 || x2329;
x2331 = x2330;
} else {
x2331 = false;
}
bool x2332;
if (x2331) {
x2332 = x1079;
} else {
x2332 = false;
}
bool x2333;
if (x2332) {
x2333 = x1079;
} else {
x2333 = false;
}
if (x2333) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2251,x992,x992,1,x2325,1,1);
assert(false && "");
}
bool x2339 = x2251 <= x2325;
int32_t x2340;
if (x2339) {
x2340 = x2325;
} else {
x2340 = x2251;
}
int32_t x2341 = x2340 * x1091;
int32_t x2342 = 64 * x2341;
float* x2343 = (float*)myMalloc(x2342 * sizeof(float));;
int32_t x2344;
if (x2326) {
x2344 = 0;
} else {
x2344 = x993;
}
int32_t x2345;
if (x2327) {
x2345 = 0;
} else {
x2345 = 1;
}
for(int x2346=0; x2346 < 64; x2346++) {
int32_t x2357 = x2252 * x2346;
int32_t x2351 = x2341 * x2346;
for(int x2348=0; x2348 < x2340; x2348++) {
int32_t x2358 = x2344 * x2348;
int32_t x2359 = x2357 + x2358;
int32_t x2364 = x2345 * x2348;
int32_t x2353 = x1091 * x2348;
for(int x2349=0; x2349 < x1090; x2349++) {
int32_t x2360 = x1096 * x2349;
int32_t x2361 = x2359 + x2360;
int32_t x2355 = x1090 * x2349;
for(int x2350=0; x2350 < x1090; x2350++) {
int32_t x2362 = x1097 * x2350;
int32_t x2363 = x2361 + x2362;
float x2365 = x2254[x2363];
float x2366 = x2294[x2364];
int32_t x2352 = x2350 + x2351;
int32_t x2354 = x2352 + x2353;
int32_t x2356 = x2354 + x2355;
float x2367 = x2365 / x2366;
x2343[x2356] = x2367;

}

}

}

}
int32_t x2377 = 0;
int32_t x2378 = 1;
x2378 *= 1;
x2377 += 1;
x2378 *= 1;
x2378 *= 1;
int32_t x2383 = x2377;
bool x2384 = x2383 >= 2;
if (x2384) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2389 = x2383 == 0;
if (x2389) {
int32_t x2390 = x2378;
bool x2391 = x2390 == 256;
if (x2391) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2398 = x2378;
bool x2400 = x2340 == 1;
int32_t x2399 = 256 / x2398;
bool x2401 = x2399 == 1;
bool x2405;
if (x454) {
bool x2402 = x2400 || x2401;
bool x2403 = x2340 == x2399;
bool x2404 = x2402 || x2403;
x2405 = x2404;
} else {
x2405 = false;
}
bool x2406;
if (x2405) {
x2406 = x1162;
} else {
x2406 = false;
}
bool x2407;
if (x2406) {
x2407 = x1162;
} else {
x2407 = false;
}
if (x2407) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2340,x1090,x1090,1,x2399,1,1);
assert(false && "");
}
bool x2413 = x2340 <= x2399;
int32_t x2414;
if (x2413) {
x2414 = x2399;
} else {
x2414 = x2340;
}
int32_t x2415 = x2414 * x1174;
int32_t x2416 = 64 * x2415;
float* x2417 = (float*)myMalloc(x2416 * sizeof(float));;
int32_t x2418;
if (x2400) {
x2418 = 0;
} else {
x2418 = x1091;
}
int32_t x2419;
if (x2401) {
x2419 = 0;
} else {
x2419 = 1;
}
for(int x2420=0; x2420 < 64; x2420++) {
int32_t x2431 = x2341 * x2420;
int32_t x2425 = x2415 * x2420;
for(int x2422=0; x2422 < x2414; x2422++) {
int32_t x2432 = x2418 * x2422;
int32_t x2433 = x2431 + x2432;
int32_t x2438 = x2419 * x2422;
int32_t x2427 = x1174 * x2422;
for(int x2423=0; x2423 < x1173; x2423++) {
int32_t x2434 = x1179 * x2423;
int32_t x2435 = x2433 + x2434;
int32_t x2429 = x1173 * x2423;
for(int x2424=0; x2424 < x1173; x2424++) {
int32_t x2436 = x1180 * x2424;
int32_t x2437 = x2435 + x2436;
float x2439 = x2343[x2437];
float x2440 = x76[x2438];
int32_t x2426 = x2424 + x2425;
int32_t x2428 = x2426 + x2427;
int32_t x2430 = x2428 + x2429;
float x2441 = x2439 * x2440;
x2417[x2430] = x2441;

}

}

}

}
int32_t x2451 = 0;
int32_t x2452 = 1;
x2452 *= 1;
x2451 += 1;
x2452 *= 1;
x2452 *= 1;
int32_t x2457 = x2451;
bool x2458 = x2457 >= 2;
if (x2458) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2463 = x2457 == 0;
if (x2463) {
int32_t x2464 = x2452;
bool x2465 = x2464 == 256;
if (x2465) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2472 = x2452;
bool x2474 = x2414 == 1;
int32_t x2473 = 256 / x2472;
bool x2475 = x2473 == 1;
bool x2479;
if (x454) {
bool x2476 = x2474 || x2475;
bool x2477 = x2414 == x2473;
bool x2478 = x2476 || x2477;
x2479 = x2478;
} else {
x2479 = false;
}
bool x2480;
if (x2479) {
x2480 = x1245;
} else {
x2480 = false;
}
bool x2481;
if (x2480) {
x2481 = x1245;
} else {
x2481 = false;
}
if (x2481) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2414,x1173,x1173,1,x2473,1,1);
assert(false && "");
}
bool x2487 = x2414 <= x2473;
int32_t x2488;
if (x2487) {
x2488 = x2473;
} else {
x2488 = x2414;
}
int32_t x2489 = x2488 * x1257;
int32_t x2490 = 64 * x2489;
float* x2491 = (float*)myMalloc(x2490 * sizeof(float));;
int32_t x2492;
if (x2474) {
x2492 = 0;
} else {
x2492 = x1174;
}
int32_t x2493;
if (x2475) {
x2493 = 0;
} else {
x2493 = 1;
}
for(int x2494=0; x2494 < 64; x2494++) {
int32_t x2505 = x2415 * x2494;
int32_t x2499 = x2489 * x2494;
for(int x2496=0; x2496 < x2488; x2496++) {
int32_t x2506 = x2492 * x2496;
int32_t x2507 = x2505 + x2506;
int32_t x2512 = x2493 * x2496;
int32_t x2501 = x1257 * x2496;
for(int x2497=0; x2497 < x1256; x2497++) {
int32_t x2508 = x1262 * x2497;
int32_t x2509 = x2507 + x2508;
int32_t x2503 = x1256 * x2497;
for(int x2498=0; x2498 < x1256; x2498++) {
int32_t x2510 = x1263 * x2498;
int32_t x2511 = x2509 + x2510;
float x2513 = x2417[x2511];
float x2514 = x203[x2512];
int32_t x2500 = x2498 + x2499;
int32_t x2502 = x2500 + x2501;
int32_t x2504 = x2502 + x2503;
float x2515 = x2513 + x2514;
x2491[x2504] = x2515;

}

}

}

}
bool x2525 = x2128 == 1;
bool x2526 = x2488 == 1;
bool x2527 = x2525 || x2526;
bool x2528 = x2128 == x2488;
bool x2529 = x2527 || x2528;
bool x2535;
if (x2529) {
x2535 = x2534;
} else {
x2535 = false;
}
bool x2536;
if (x2535) {
x2536 = x2534;
} else {
x2536 = false;
}
if (x2536) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2128,x2130,x2130,64,x2488,x1256,x1256);
assert(false && "");
}
bool x2542 = x2128 <= x2488;
int32_t x2543;
if (x2542) {
x2543 = x2488;
} else {
x2543 = x2128;
}
int32_t x2549;
if (x2525) {
x2549 = 0;
} else {
x2549 = x2131;
}
int32_t x2552;
if (x2526) {
x2552 = 0;
} else {
x2552 = x1257;
}
for(int x2555=0; x2555 < 64; x2555++) {
int32_t x2561 = x2132 * x2555;
int32_t x2568 = x2489 * x2555;
for(int x2557=0; x2557 < x2543; x2557++) {
int32_t x2562 = x2549 * x2557;
int32_t x2563 = x2561 + x2562;
int32_t x2569 = x2552 * x2557;
int32_t x2570 = x2568 + x2569;
for(int x2559=0; x2559 < x2545; x2559++) {
int32_t x2564 = x2550 * x2559;
int32_t x2565 = x2563 + x2564;
int32_t x2571 = x2553 * x2559;
int32_t x2572 = x2570 + x2571;
for(int x2560=0; x2560 < x2545; x2560++) {
int32_t x2566 = x2551 * x2560;
int32_t x2567 = x2565 + x2566;
float x2575 = x2134[x2567];
int32_t x2573 = x2554 * x2560;
int32_t x2574 = x2572 + x2573;
float x2576 = x2491[x2574];
float x2577 = x2575 + x2576;
x2134[x2567] = x2577;

}

}

}

}
float* x2587 = (float*)myMalloc(x2133 * sizeof(float));;
for(int x2589=0; x2589 < x2133; x2589++) {
float x2590 = x2134[x2589];
bool x2591 = x2590 < 0.0f;
if (x2591) {
x2587[x2589] = 0.0f;
} else {
float x2594 = x2134[x2589];
x2587[x2589] = x2594;
}

}
float* x2608 = (float*)myMalloc(x2607 * sizeof(float));;
int32_t x2611 = 64 * x2128;
int32_t x2612 = x2611 * x2603;
float* x2613 = (float*)myMalloc(x2612 * sizeof(float));;
int32_t x2609 = x2128 * x2603;
for(int x2614=0; x2614 < 64; x2614++) {
int32_t x2615 = x2614 * x2132;
float* x2616 = x2587+x2615;
int32_t x2617 = x2614 * x2604;
float* x2618 = x2608+x2617;
int32_t x2619 = x2614 * x2609;
float* x2620 = x2613+x2619;
for(int x2621=0; x2621 < x2128; x2621++) {
int32_t x2622 = x2621 / 1;
int32_t x2626 = x2622 * x2602;
int32_t x2627 = x2626 * x2602;
int32_t x2623 = x2621 % 1;
int32_t x2624 = x2623 / 1;
int32_t x2628 = x2624 * x2602;
int32_t x2629 = x2628 * x2602;
int32_t x2630 = x2627 + x2629;
int32_t x2625 = x2623 % 1;
int32_t x2631 = x2625 * x2602;
int32_t x2632 = x2631 * x2602;
int32_t x2633 = x2630 + x2632;
float* x2634 = x2620+x2633;
int32_t x2635 = x2622 * x2130;
int32_t x2636 = x2635 * x2130;
float* x2637 = x2616+x2636;
for(int x2639=0; x2639 < x2602; x2639++) {
int32_t x2641 = x2639 * x2602;
float* x2642 = x2634+x2641;
int32_t x2640 = x2639 + x2624;
int32_t x2643 = x2640 * x2130;
int32_t x2644 = x2643 + x2625;
float* x2645 = x2637+x2644;
memcpy(x2642, x2645, 4 * x2602);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x2603,x2128,1,x171,x2128,x2620,x2603,1,x2618,x2603);

}
int32_t x2654 = 0;
int32_t x2655 = 1;
x2655 *= 1;
x2654 += 1;
x2655 *= 1;
x2655 *= 1;
int32_t x2660 = x2654;
bool x2661 = x2660 >= 2;
if (x2661) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2666 = x2660 == 0;
if (x2666) {
int32_t x2667 = x2655;
bool x2668 = x2667 == 64;
if (x2668) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2675 = x2655;
int32_t x2676 = 64 / x2675;
bool x2677 = x2676 == 1;
bool x2680;
if (x454) {
bool x2678 = 64 == x2676;
bool x2679 = x2677 || x2678;
x2680 = x2679;
} else {
x2680 = false;
}
bool x2684;
if (x2680) {
x2684 = x2683;
} else {
x2684 = false;
}
bool x2685;
if (x2684) {
x2685 = x2683;
} else {
x2685 = false;
}
if (x2685) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x2602,x2602,1,x2676,1,1);
assert(false && "");
}
bool x2691 = 64 <= x2676;
int32_t x2692;
if (x2691) {
x2692 = x2676;
} else {
x2692 = 64;
}
int32_t x2696 = x2692 * x2695;
int32_t x2697 = 64 * x2696;
float* x2698 = (float*)myMalloc(x2697 * sizeof(float));;
int32_t x2701;
if (x2677) {
x2701 = 0;
} else {
x2701 = 1;
}
for(int x2702=0; x2702 < 64; x2702++) {
int32_t x2714 = x2604 * x2702;
int32_t x2708 = x2696 * x2702;
for(int x2704=0; x2704 < x2692; x2704++) {
int32_t x2715 = x2603 * x2704;
int32_t x2716 = x2714 + x2715;
int32_t x2721 = x2701 * x2704;
int32_t x2710 = x2695 * x2704;
for(int x2706=0; x2706 < x2694; x2706++) {
int32_t x2717 = x2699 * x2706;
int32_t x2718 = x2716 + x2717;
int32_t x2712 = x2694 * x2706;
for(int x2707=0; x2707 < x2694; x2707++) {
int32_t x2719 = x2700 * x2707;
int32_t x2720 = x2718 + x2719;
float x2722 = x2608[x2720];
float x2723 = x10[x2721];
int32_t x2709 = x2707 + x2708;
int32_t x2711 = x2709 + x2710;
int32_t x2713 = x2711 + x2712;
float x2724 = x2722 - x2723;
x2698[x2713] = x2724;

}

}

}

}
float* x2734 = (float*)myMalloc(64 * sizeof(float));;
for(int x2735=0; x2735 < 64; x2735++) {
float x2736 = x102[x2735];
float x2737 = x2736 + 1.0E-5f;
x2734[x2735] = x2737;

}
float* x2741 = (float*)myMalloc(64 * sizeof(float));;
for(int x2742=0; x2742 < 64; x2742++) {
float x2743 = x2734[x2742];
double x2744 = (double)x2743;
double x2745 = sqrt(x2744);
float x2746 = (float)x2745;
x2741[x2742] = x2746;

}
int32_t x2750 = 0;
int32_t x2751 = 1;
x2751 *= 1;
x2750 += 1;
x2751 *= 1;
x2751 *= 1;
int32_t x2756 = x2750;
bool x2757 = x2756 >= 2;
if (x2757) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2762 = x2756 == 0;
if (x2762) {
int32_t x2763 = x2751;
bool x2764 = x2763 == 64;
if (x2764) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2771 = x2751;
bool x2773 = x2692 == 1;
int32_t x2772 = 64 / x2771;
bool x2774 = x2772 == 1;
bool x2778;
if (x454) {
bool x2775 = x2773 || x2774;
bool x2776 = x2692 == x2772;
bool x2777 = x2775 || x2776;
x2778 = x2777;
} else {
x2778 = false;
}
bool x2782;
if (x2778) {
x2782 = x2781;
} else {
x2782 = false;
}
bool x2783;
if (x2782) {
x2783 = x2781;
} else {
x2783 = false;
}
if (x2783) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2692,x2694,x2694,1,x2772,1,1);
assert(false && "");
}
bool x2789 = x2692 <= x2772;
int32_t x2790;
if (x2789) {
x2790 = x2772;
} else {
x2790 = x2692;
}
int32_t x2794 = x2790 * x2793;
int32_t x2795 = 64 * x2794;
float* x2796 = (float*)myMalloc(x2795 * sizeof(float));;
int32_t x2797;
if (x2773) {
x2797 = 0;
} else {
x2797 = x2695;
}
int32_t x2800;
if (x2774) {
x2800 = 0;
} else {
x2800 = 1;
}
for(int x2801=0; x2801 < 64; x2801++) {
int32_t x2813 = x2696 * x2801;
int32_t x2807 = x2794 * x2801;
for(int x2803=0; x2803 < x2790; x2803++) {
int32_t x2814 = x2797 * x2803;
int32_t x2815 = x2813 + x2814;
int32_t x2820 = x2800 * x2803;
int32_t x2809 = x2793 * x2803;
for(int x2805=0; x2805 < x2792; x2805++) {
int32_t x2816 = x2798 * x2805;
int32_t x2817 = x2815 + x2816;
int32_t x2811 = x2792 * x2805;
for(int x2806=0; x2806 < x2792; x2806++) {
int32_t x2818 = x2799 * x2806;
int32_t x2819 = x2817 + x2818;
float x2821 = x2698[x2819];
float x2822 = x2741[x2820];
int32_t x2808 = x2806 + x2807;
int32_t x2810 = x2808 + x2809;
int32_t x2812 = x2810 + x2811;
float x2823 = x2821 / x2822;
x2796[x2812] = x2823;

}

}

}

}
int32_t x2833 = 0;
int32_t x2834 = 1;
x2834 *= 1;
x2833 += 1;
x2834 *= 1;
x2834 *= 1;
int32_t x2839 = x2833;
bool x2840 = x2839 >= 2;
if (x2840) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2845 = x2839 == 0;
if (x2845) {
int32_t x2846 = x2834;
bool x2847 = x2846 == 64;
if (x2847) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2854 = x2834;
bool x2856 = x2790 == 1;
int32_t x2855 = 64 / x2854;
bool x2857 = x2855 == 1;
bool x2861;
if (x454) {
bool x2858 = x2856 || x2857;
bool x2859 = x2790 == x2855;
bool x2860 = x2858 || x2859;
x2861 = x2860;
} else {
x2861 = false;
}
bool x2865;
if (x2861) {
x2865 = x2864;
} else {
x2865 = false;
}
bool x2866;
if (x2865) {
x2866 = x2864;
} else {
x2866 = false;
}
if (x2866) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2790,x2792,x2792,1,x2855,1,1);
assert(false && "");
}
bool x2872 = x2790 <= x2855;
int32_t x2873;
if (x2872) {
x2873 = x2855;
} else {
x2873 = x2790;
}
int32_t x2877 = x2873 * x2876;
int32_t x2878 = 64 * x2877;
float* x2879 = (float*)myMalloc(x2878 * sizeof(float));;
int32_t x2880;
if (x2856) {
x2880 = 0;
} else {
x2880 = x2793;
}
int32_t x2883;
if (x2857) {
x2883 = 0;
} else {
x2883 = 1;
}
for(int x2884=0; x2884 < 64; x2884++) {
int32_t x2896 = x2794 * x2884;
int32_t x2890 = x2877 * x2884;
for(int x2886=0; x2886 < x2873; x2886++) {
int32_t x2897 = x2880 * x2886;
int32_t x2898 = x2896 + x2897;
int32_t x2903 = x2883 * x2886;
int32_t x2892 = x2876 * x2886;
for(int x2888=0; x2888 < x2875; x2888++) {
int32_t x2899 = x2881 * x2888;
int32_t x2900 = x2898 + x2899;
int32_t x2894 = x2875 * x2888;
for(int x2889=0; x2889 < x2875; x2889++) {
int32_t x2901 = x2882 * x2889;
int32_t x2902 = x2900 + x2901;
float x2904 = x2796[x2902];
float x2905 = x142[x2903];
int32_t x2891 = x2889 + x2890;
int32_t x2893 = x2891 + x2892;
int32_t x2895 = x2893 + x2894;
float x2906 = x2904 * x2905;
x2879[x2895] = x2906;

}

}

}

}
int32_t x2916 = 0;
int32_t x2917 = 1;
x2917 *= 1;
x2916 += 1;
x2917 *= 1;
x2917 *= 1;
int32_t x2922 = x2916;
bool x2923 = x2922 >= 2;
if (x2923) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2928 = x2922 == 0;
if (x2928) {
int32_t x2929 = x2917;
bool x2930 = x2929 == 64;
if (x2930) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2937 = x2917;
bool x2939 = x2873 == 1;
int32_t x2938 = 64 / x2937;
bool x2940 = x2938 == 1;
bool x2944;
if (x454) {
bool x2941 = x2939 || x2940;
bool x2942 = x2873 == x2938;
bool x2943 = x2941 || x2942;
x2944 = x2943;
} else {
x2944 = false;
}
bool x2948;
if (x2944) {
x2948 = x2947;
} else {
x2948 = false;
}
bool x2949;
if (x2948) {
x2949 = x2947;
} else {
x2949 = false;
}
if (x2949) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2873,x2875,x2875,1,x2938,1,1);
assert(false && "");
}
bool x2955 = x2873 <= x2938;
int32_t x2956;
if (x2955) {
x2956 = x2938;
} else {
x2956 = x2873;
}
int32_t x2960 = x2956 * x2959;
int32_t x2961 = 64 * x2960;
float* x2962 = (float*)myMalloc(x2961 * sizeof(float));;
int32_t x2963;
if (x2939) {
x2963 = 0;
} else {
x2963 = x2876;
}
int32_t x2966;
if (x2940) {
x2966 = 0;
} else {
x2966 = 1;
}
for(int x2967=0; x2967 < 64; x2967++) {
int32_t x2979 = x2877 * x2967;
int32_t x2973 = x2960 * x2967;
for(int x2969=0; x2969 < x2956; x2969++) {
int32_t x2980 = x2963 * x2969;
int32_t x2981 = x2979 + x2980;
int32_t x2986 = x2966 * x2969;
int32_t x2975 = x2959 * x2969;
for(int x2971=0; x2971 < x2958; x2971++) {
int32_t x2982 = x2964 * x2971;
int32_t x2983 = x2981 + x2982;
int32_t x2977 = x2958 * x2971;
for(int x2972=0; x2972 < x2958; x2972++) {
int32_t x2984 = x2965 * x2972;
int32_t x2985 = x2983 + x2984;
float x2987 = x2879[x2985];
float x2988 = x60[x2986];
int32_t x2974 = x2972 + x2973;
int32_t x2976 = x2974 + x2975;
int32_t x2978 = x2976 + x2977;
float x2989 = x2987 + x2988;
x2962[x2978] = x2989;

}

}

}

}
float* x2999 = (float*)myMalloc(x2961 * sizeof(float));;
for(int x3001=0; x3001 < x2961; x3001++) {
float x3002 = x2962[x3001];
bool x3003 = x3002 < 0.0f;
if (x3003) {
x2999[x3001] = 0.0f;
} else {
float x3006 = x2962[x3001];
x2999[x3001] = x3006;
}

}
float* x3021 = (float*)myMalloc(x3020 * sizeof(float));;
int32_t x3022 = 9 * x2956;
int32_t x3025 = 64 * x3022;
int32_t x3026 = x3025 * x3016;
float* x3027 = (float*)myMalloc(x3026 * sizeof(float));;
int32_t x3023 = x3022 * x3016;
int32_t x3035 = x2956 * 3;
int32_t x3036 = x3035 * 3;
for(int x3028=0; x3028 < 64; x3028++) {
int32_t x3029 = x3028 * x2960;
float* x3030 = x2999+x3029;
int32_t x3031 = x3028 * x3017;
float* x3032 = x3021+x3031;
int32_t x3033 = x3028 * x3023;
float* x3034 = x3027+x3033;
for(int x3038=0; x3038 < x3036; x3038++) {
int32_t x3039 = x3038 / 9;
int32_t x3043 = x3039 * 3;
int32_t x3044 = x3043 * 3;
int32_t x3045 = x3044 * x3015;
int32_t x3046 = x3045 * x3015;
int32_t x3040 = x3038 % 9;
int32_t x3041 = x3040 / 3;
int32_t x3047 = x3041 * 3;
int32_t x3048 = x3047 * x3015;
int32_t x3049 = x3048 * x3015;
int32_t x3050 = x3046 + x3049;
int32_t x3042 = x3040 % 3;
int32_t x3051 = x3042 * x3015;
int32_t x3052 = x3051 * x3015;
int32_t x3053 = x3050 + x3052;
float* x3054 = x3034+x3053;
int32_t x3055 = x3039 * x2958;
int32_t x3056 = x3055 * x2958;
float* x3057 = x3030+x3056;
int32_t x3070 = 1 - x3042;
bool x3071 = x3070 > 0;
int32_t x3072;
if (x3071) {
x3072 = x3070;
} else {
x3072 = 0;
}
int32_t x3073 = 3 - x3042;
int32_t x3074 = x3073 - 1;
int32_t x3075 = 1 - x3074;
bool x3076 = x3075 > 0;
int32_t x3077;
if (x3076) {
x3077 = x3075;
} else {
x3077 = 0;
}
int32_t x3078 = x3015 - x3077;
int32_t x3079 = x3078 - x3072;
bool x3080 = x3079 <= 0;
bool x3084 = x3072 > 0;
int32_t x3069 = -1 + x3042;
bool x3097 = x3077 > 0;
for(int x3059=0; x3059 < x3015; x3059++) {
int32_t x3060 = x3059 - 1;
int32_t x3061 = x3060 + x3041;
bool x3062 = x3061 < 0;
bool x3063 = x3061 >= x2958;
bool x3064 = x3062 || x3063;
if (x3064) {
int32_t x3065 = x3059 * x3015;
float* x3066 = x3054+x3065;
memset(x3066, 0, 4 * x3015);;
} else {
if (x3080) {
int32_t x3065 = x3059 * x3015;
float* x3081 = x3054+x3065;
memset(x3081, 0, 4 * x3015);;
} else {
int32_t x3065 = x3059 * x3015;
if (x3084) {
float* x3085 = x3054+x3065;
memset(x3085, 0, 4 * x3072);;
} else {
}
// may have segfault here
int32_t x3090 = x3065 + x3072;
float* x3091 = x3054+x3090;
int32_t x3092 = x3061 * x2958;
int32_t x3093 = x3092 + x3069;
int32_t x3094 = x3093 + x3072;
float* x3095 = x3057+x3094;
memcpy(x3091, x3095, 4 * x3079);;
if (x3097) {
int32_t x3098 = x3065 + x3015;
int32_t x3099 = x3098 - x3077;
float* x3100 = x3054+x3099;
memset(x3100, 0, 4 * x3077);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x3016,x3022,1,x83,x3022,x3034,x3016,1,x3032,x3016);

}
int32_t x3115 = 0;
int32_t x3116 = 1;
x3116 *= 1;
x3115 += 1;
x3116 *= 1;
x3116 *= 1;
int32_t x3121 = x3115;
bool x3122 = x3121 >= 2;
if (x3122) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3127 = x3121 == 0;
if (x3127) {
int32_t x3128 = x3116;
bool x3129 = x3128 == 64;
if (x3129) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3136 = x3116;
int32_t x3137 = 64 / x3136;
bool x3138 = x3137 == 1;
bool x3141;
if (x454) {
bool x3139 = 64 == x3137;
bool x3140 = x3138 || x3139;
x3141 = x3140;
} else {
x3141 = false;
}
bool x3145;
if (x3141) {
x3145 = x3144;
} else {
x3145 = false;
}
bool x3146;
if (x3145) {
x3146 = x3144;
} else {
x3146 = false;
}
if (x3146) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x3015,x3015,1,x3137,1,1);
assert(false && "");
}
bool x3152 = 64 <= x3137;
int32_t x3153;
if (x3152) {
x3153 = x3137;
} else {
x3153 = 64;
}
int32_t x3157 = x3153 * x3156;
int32_t x3158 = 64 * x3157;
float* x3159 = (float*)myMalloc(x3158 * sizeof(float));;
int32_t x3162;
if (x3138) {
x3162 = 0;
} else {
x3162 = 1;
}
for(int x3163=0; x3163 < 64; x3163++) {
int32_t x3175 = x3017 * x3163;
int32_t x3169 = x3157 * x3163;
for(int x3165=0; x3165 < x3153; x3165++) {
int32_t x3176 = x3016 * x3165;
int32_t x3177 = x3175 + x3176;
int32_t x3182 = x3162 * x3165;
int32_t x3171 = x3156 * x3165;
for(int x3167=0; x3167 < x3155; x3167++) {
int32_t x3178 = x3160 * x3167;
int32_t x3179 = x3177 + x3178;
int32_t x3173 = x3155 * x3167;
for(int x3168=0; x3168 < x3155; x3168++) {
int32_t x3180 = x3161 * x3168;
int32_t x3181 = x3179 + x3180;
float x3183 = x3021[x3181];
float x3184 = x44[x3182];
int32_t x3170 = x3168 + x3169;
int32_t x3172 = x3170 + x3171;
int32_t x3174 = x3172 + x3173;
float x3185 = x3183 - x3184;
x3159[x3174] = x3185;

}

}

}

}
float* x3195 = (float*)myMalloc(64 * sizeof(float));;
for(int x3196=0; x3196 < 64; x3196++) {
float x3197 = x244[x3196];
float x3198 = x3197 + 1.0E-5f;
x3195[x3196] = x3198;

}
float* x3202 = (float*)myMalloc(64 * sizeof(float));;
for(int x3203=0; x3203 < 64; x3203++) {
float x3204 = x3195[x3203];
double x3205 = (double)x3204;
double x3206 = sqrt(x3205);
float x3207 = (float)x3206;
x3202[x3203] = x3207;

}
int32_t x3211 = 0;
int32_t x3212 = 1;
x3212 *= 1;
x3211 += 1;
x3212 *= 1;
x3212 *= 1;
int32_t x3217 = x3211;
bool x3218 = x3217 >= 2;
if (x3218) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3223 = x3217 == 0;
if (x3223) {
int32_t x3224 = x3212;
bool x3225 = x3224 == 64;
if (x3225) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3232 = x3212;
bool x3234 = x3153 == 1;
int32_t x3233 = 64 / x3232;
bool x3235 = x3233 == 1;
bool x3239;
if (x454) {
bool x3236 = x3234 || x3235;
bool x3237 = x3153 == x3233;
bool x3238 = x3236 || x3237;
x3239 = x3238;
} else {
x3239 = false;
}
bool x3243;
if (x3239) {
x3243 = x3242;
} else {
x3243 = false;
}
bool x3244;
if (x3243) {
x3244 = x3242;
} else {
x3244 = false;
}
if (x3244) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3153,x3155,x3155,1,x3233,1,1);
assert(false && "");
}
bool x3250 = x3153 <= x3233;
int32_t x3251;
if (x3250) {
x3251 = x3233;
} else {
x3251 = x3153;
}
int32_t x3255 = x3251 * x3254;
int32_t x3256 = 64 * x3255;
float* x3257 = (float*)myMalloc(x3256 * sizeof(float));;
int32_t x3258;
if (x3234) {
x3258 = 0;
} else {
x3258 = x3156;
}
int32_t x3261;
if (x3235) {
x3261 = 0;
} else {
x3261 = 1;
}
for(int x3262=0; x3262 < 64; x3262++) {
int32_t x3274 = x3157 * x3262;
int32_t x3268 = x3255 * x3262;
for(int x3264=0; x3264 < x3251; x3264++) {
int32_t x3275 = x3258 * x3264;
int32_t x3276 = x3274 + x3275;
int32_t x3281 = x3261 * x3264;
int32_t x3270 = x3254 * x3264;
for(int x3266=0; x3266 < x3253; x3266++) {
int32_t x3277 = x3259 * x3266;
int32_t x3278 = x3276 + x3277;
int32_t x3272 = x3253 * x3266;
for(int x3267=0; x3267 < x3253; x3267++) {
int32_t x3279 = x3260 * x3267;
int32_t x3280 = x3278 + x3279;
float x3282 = x3159[x3280];
float x3283 = x3202[x3281];
int32_t x3269 = x3267 + x3268;
int32_t x3271 = x3269 + x3270;
int32_t x3273 = x3271 + x3272;
float x3284 = x3282 / x3283;
x3257[x3273] = x3284;

}

}

}

}
int32_t x3294 = 0;
int32_t x3295 = 1;
x3295 *= 1;
x3294 += 1;
x3295 *= 1;
x3295 *= 1;
int32_t x3300 = x3294;
bool x3301 = x3300 >= 2;
if (x3301) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3306 = x3300 == 0;
if (x3306) {
int32_t x3307 = x3295;
bool x3308 = x3307 == 64;
if (x3308) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3315 = x3295;
bool x3317 = x3251 == 1;
int32_t x3316 = 64 / x3315;
bool x3318 = x3316 == 1;
bool x3322;
if (x454) {
bool x3319 = x3317 || x3318;
bool x3320 = x3251 == x3316;
bool x3321 = x3319 || x3320;
x3322 = x3321;
} else {
x3322 = false;
}
bool x3326;
if (x3322) {
x3326 = x3325;
} else {
x3326 = false;
}
bool x3327;
if (x3326) {
x3327 = x3325;
} else {
x3327 = false;
}
if (x3327) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3251,x3253,x3253,1,x3316,1,1);
assert(false && "");
}
bool x3333 = x3251 <= x3316;
int32_t x3334;
if (x3333) {
x3334 = x3316;
} else {
x3334 = x3251;
}
int32_t x3338 = x3334 * x3337;
int32_t x3339 = 64 * x3338;
float* x3340 = (float*)myMalloc(x3339 * sizeof(float));;
int32_t x3341;
if (x3317) {
x3341 = 0;
} else {
x3341 = x3254;
}
int32_t x3344;
if (x3318) {
x3344 = 0;
} else {
x3344 = 1;
}
for(int x3345=0; x3345 < 64; x3345++) {
int32_t x3357 = x3255 * x3345;
int32_t x3351 = x3338 * x3345;
for(int x3347=0; x3347 < x3334; x3347++) {
int32_t x3358 = x3341 * x3347;
int32_t x3359 = x3357 + x3358;
int32_t x3364 = x3344 * x3347;
int32_t x3353 = x3337 * x3347;
for(int x3349=0; x3349 < x3336; x3349++) {
int32_t x3360 = x3342 * x3349;
int32_t x3361 = x3359 + x3360;
int32_t x3355 = x3336 * x3349;
for(int x3350=0; x3350 < x3336; x3350++) {
int32_t x3362 = x3343 * x3350;
int32_t x3363 = x3361 + x3362;
float x3365 = x3257[x3363];
float x3366 = x208[x3364];
int32_t x3352 = x3350 + x3351;
int32_t x3354 = x3352 + x3353;
int32_t x3356 = x3354 + x3355;
float x3367 = x3365 * x3366;
x3340[x3356] = x3367;

}

}

}

}
int32_t x3377 = 0;
int32_t x3378 = 1;
x3378 *= 1;
x3377 += 1;
x3378 *= 1;
x3378 *= 1;
int32_t x3383 = x3377;
bool x3384 = x3383 >= 2;
if (x3384) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3389 = x3383 == 0;
if (x3389) {
int32_t x3390 = x3378;
bool x3391 = x3390 == 64;
if (x3391) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3398 = x3378;
bool x3400 = x3334 == 1;
int32_t x3399 = 64 / x3398;
bool x3401 = x3399 == 1;
bool x3405;
if (x454) {
bool x3402 = x3400 || x3401;
bool x3403 = x3334 == x3399;
bool x3404 = x3402 || x3403;
x3405 = x3404;
} else {
x3405 = false;
}
bool x3409;
if (x3405) {
x3409 = x3408;
} else {
x3409 = false;
}
bool x3410;
if (x3409) {
x3410 = x3408;
} else {
x3410 = false;
}
if (x3410) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3334,x3336,x3336,1,x3399,1,1);
assert(false && "");
}
bool x3416 = x3334 <= x3399;
int32_t x3417;
if (x3416) {
x3417 = x3399;
} else {
x3417 = x3334;
}
int32_t x3421 = x3417 * x3420;
int32_t x3422 = 64 * x3421;
float* x3423 = (float*)myMalloc(x3422 * sizeof(float));;
int32_t x3424;
if (x3400) {
x3424 = 0;
} else {
x3424 = x3337;
}
int32_t x3427;
if (x3401) {
x3427 = 0;
} else {
x3427 = 1;
}
for(int x3428=0; x3428 < 64; x3428++) {
int32_t x3440 = x3338 * x3428;
int32_t x3434 = x3421 * x3428;
for(int x3430=0; x3430 < x3417; x3430++) {
int32_t x3441 = x3424 * x3430;
int32_t x3442 = x3440 + x3441;
int32_t x3447 = x3427 * x3430;
int32_t x3436 = x3420 * x3430;
for(int x3432=0; x3432 < x3419; x3432++) {
int32_t x3443 = x3425 * x3432;
int32_t x3444 = x3442 + x3443;
int32_t x3438 = x3419 * x3432;
for(int x3433=0; x3433 < x3419; x3433++) {
int32_t x3445 = x3426 * x3433;
int32_t x3446 = x3444 + x3445;
float x3448 = x3340[x3446];
float x3449 = x153[x3447];
int32_t x3435 = x3433 + x3434;
int32_t x3437 = x3435 + x3436;
int32_t x3439 = x3437 + x3438;
float x3450 = x3448 + x3449;
x3423[x3439] = x3450;

}

}

}

}
float* x3460 = (float*)myMalloc(x3422 * sizeof(float));;
for(int x3462=0; x3462 < x3422; x3462++) {
float x3463 = x3423[x3462];
bool x3464 = x3463 < 0.0f;
if (x3464) {
x3460[x3462] = 0.0f;
} else {
float x3467 = x3423[x3462];
x3460[x3462] = x3467;
}

}
float* x3481 = (float*)myMalloc(x3480 * sizeof(float));;
int32_t x3484 = 64 * x3417;
int32_t x3485 = x3484 * x3476;
float* x3486 = (float*)myMalloc(x3485 * sizeof(float));;
int32_t x3482 = x3417 * x3476;
for(int x3487=0; x3487 < 64; x3487++) {
int32_t x3488 = x3487 * x3421;
float* x3489 = x3460+x3488;
int32_t x3490 = x3487 * x3477;
float* x3491 = x3481+x3490;
int32_t x3492 = x3487 * x3482;
float* x3493 = x3486+x3492;
for(int x3494=0; x3494 < x3417; x3494++) {
int32_t x3495 = x3494 / 1;
int32_t x3499 = x3495 * x3475;
int32_t x3500 = x3499 * x3475;
int32_t x3496 = x3494 % 1;
int32_t x3497 = x3496 / 1;
int32_t x3501 = x3497 * x3475;
int32_t x3502 = x3501 * x3475;
int32_t x3503 = x3500 + x3502;
int32_t x3498 = x3496 % 1;
int32_t x3504 = x3498 * x3475;
int32_t x3505 = x3504 * x3475;
int32_t x3506 = x3503 + x3505;
float* x3507 = x3493+x3506;
int32_t x3508 = x3495 * x3419;
int32_t x3509 = x3508 * x3419;
float* x3510 = x3489+x3509;
for(int x3512=0; x3512 < x3475; x3512++) {
int32_t x3514 = x3512 * x3475;
float* x3515 = x3507+x3514;
int32_t x3513 = x3512 + x3497;
int32_t x3516 = x3513 * x3419;
int32_t x3517 = x3516 + x3498;
float* x3518 = x3510+x3517;
memcpy(x3515, x3518, 4 * x3475);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x3476,x3417,1,x130,x3417,x3493,x3476,1,x3491,x3476);

}
int32_t x3527 = 0;
int32_t x3528 = 1;
x3528 *= 1;
x3527 += 1;
x3528 *= 1;
x3528 *= 1;
int32_t x3533 = x3527;
bool x3534 = x3533 >= 2;
if (x3534) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3539 = x3533 == 0;
if (x3539) {
int32_t x3540 = x3528;
bool x3541 = x3540 == 256;
if (x3541) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3548 = x3528;
int32_t x3549 = 256 / x3548;
bool x3550 = x3549 == 1;
bool x3553;
if (x454) {
bool x3551 = 256 == x3549;
bool x3552 = x3550 || x3551;
x3553 = x3552;
} else {
x3553 = false;
}
bool x3557;
if (x3553) {
x3557 = x3556;
} else {
x3557 = false;
}
bool x3558;
if (x3557) {
x3558 = x3556;
} else {
x3558 = false;
}
if (x3558) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x3475,x3475,1,x3549,1,1);
assert(false && "");
}
bool x3564 = 256 <= x3549;
int32_t x3565;
if (x3564) {
x3565 = x3549;
} else {
x3565 = 256;
}
int32_t x3569 = x3565 * x3568;
int32_t x3570 = 64 * x3569;
float* x3571 = (float*)myMalloc(x3570 * sizeof(float));;
int32_t x3574;
if (x3550) {
x3574 = 0;
} else {
x3574 = 1;
}
for(int x3575=0; x3575 < 64; x3575++) {
int32_t x3587 = x3477 * x3575;
int32_t x3581 = x3569 * x3575;
for(int x3577=0; x3577 < x3565; x3577++) {
int32_t x3588 = x3476 * x3577;
int32_t x3589 = x3587 + x3588;
int32_t x3594 = x3574 * x3577;
int32_t x3583 = x3568 * x3577;
for(int x3579=0; x3579 < x3567; x3579++) {
int32_t x3590 = x3572 * x3579;
int32_t x3591 = x3589 + x3590;
int32_t x3585 = x3567 * x3579;
for(int x3580=0; x3580 < x3567; x3580++) {
int32_t x3592 = x3573 * x3580;
int32_t x3593 = x3591 + x3592;
float x3595 = x3481[x3593];
float x3596 = x91[x3594];
int32_t x3582 = x3580 + x3581;
int32_t x3584 = x3582 + x3583;
int32_t x3586 = x3584 + x3585;
float x3597 = x3595 - x3596;
x3571[x3586] = x3597;

}

}

}

}
float* x3607 = (float*)myMalloc(256 * sizeof(float));;
for(int x3608=0; x3608 < 256; x3608++) {
float x3609 = x166[x3608];
float x3610 = x3609 + 1.0E-5f;
x3607[x3608] = x3610;

}
float* x3614 = (float*)myMalloc(256 * sizeof(float));;
for(int x3615=0; x3615 < 256; x3615++) {
float x3616 = x3607[x3615];
double x3617 = (double)x3616;
double x3618 = sqrt(x3617);
float x3619 = (float)x3618;
x3614[x3615] = x3619;

}
int32_t x3623 = 0;
int32_t x3624 = 1;
x3624 *= 1;
x3623 += 1;
x3624 *= 1;
x3624 *= 1;
int32_t x3629 = x3623;
bool x3630 = x3629 >= 2;
if (x3630) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3635 = x3629 == 0;
if (x3635) {
int32_t x3636 = x3624;
bool x3637 = x3636 == 256;
if (x3637) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3644 = x3624;
bool x3646 = x3565 == 1;
int32_t x3645 = 256 / x3644;
bool x3647 = x3645 == 1;
bool x3651;
if (x454) {
bool x3648 = x3646 || x3647;
bool x3649 = x3565 == x3645;
bool x3650 = x3648 || x3649;
x3651 = x3650;
} else {
x3651 = false;
}
bool x3655;
if (x3651) {
x3655 = x3654;
} else {
x3655 = false;
}
bool x3656;
if (x3655) {
x3656 = x3654;
} else {
x3656 = false;
}
if (x3656) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3565,x3567,x3567,1,x3645,1,1);
assert(false && "");
}
bool x3662 = x3565 <= x3645;
int32_t x3663;
if (x3662) {
x3663 = x3645;
} else {
x3663 = x3565;
}
int32_t x3667 = x3663 * x3666;
int32_t x3668 = 64 * x3667;
float* x3669 = (float*)myMalloc(x3668 * sizeof(float));;
int32_t x3670;
if (x3646) {
x3670 = 0;
} else {
x3670 = x3568;
}
int32_t x3673;
if (x3647) {
x3673 = 0;
} else {
x3673 = 1;
}
for(int x3674=0; x3674 < 64; x3674++) {
int32_t x3686 = x3569 * x3674;
int32_t x3680 = x3667 * x3674;
for(int x3676=0; x3676 < x3663; x3676++) {
int32_t x3687 = x3670 * x3676;
int32_t x3688 = x3686 + x3687;
int32_t x3693 = x3673 * x3676;
int32_t x3682 = x3666 * x3676;
for(int x3678=0; x3678 < x3665; x3678++) {
int32_t x3689 = x3671 * x3678;
int32_t x3690 = x3688 + x3689;
int32_t x3684 = x3665 * x3678;
for(int x3679=0; x3679 < x3665; x3679++) {
int32_t x3691 = x3672 * x3679;
int32_t x3692 = x3690 + x3691;
float x3694 = x3571[x3692];
float x3695 = x3614[x3693];
int32_t x3681 = x3679 + x3680;
int32_t x3683 = x3681 + x3682;
int32_t x3685 = x3683 + x3684;
float x3696 = x3694 / x3695;
x3669[x3685] = x3696;

}

}

}

}
int32_t x3706 = 0;
int32_t x3707 = 1;
x3707 *= 1;
x3706 += 1;
x3707 *= 1;
x3707 *= 1;
int32_t x3712 = x3706;
bool x3713 = x3712 >= 2;
if (x3713) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3718 = x3712 == 0;
if (x3718) {
int32_t x3719 = x3707;
bool x3720 = x3719 == 256;
if (x3720) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3727 = x3707;
bool x3729 = x3663 == 1;
int32_t x3728 = 256 / x3727;
bool x3730 = x3728 == 1;
bool x3734;
if (x454) {
bool x3731 = x3729 || x3730;
bool x3732 = x3663 == x3728;
bool x3733 = x3731 || x3732;
x3734 = x3733;
} else {
x3734 = false;
}
bool x3738;
if (x3734) {
x3738 = x3737;
} else {
x3738 = false;
}
bool x3739;
if (x3738) {
x3739 = x3737;
} else {
x3739 = false;
}
if (x3739) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3663,x3665,x3665,1,x3728,1,1);
assert(false && "");
}
bool x3745 = x3663 <= x3728;
int32_t x3746;
if (x3745) {
x3746 = x3728;
} else {
x3746 = x3663;
}
int32_t x3750 = x3746 * x3749;
int32_t x3751 = 64 * x3750;
float* x3752 = (float*)myMalloc(x3751 * sizeof(float));;
int32_t x3753;
if (x3729) {
x3753 = 0;
} else {
x3753 = x3666;
}
int32_t x3756;
if (x3730) {
x3756 = 0;
} else {
x3756 = 1;
}
for(int x3757=0; x3757 < 64; x3757++) {
int32_t x3769 = x3667 * x3757;
int32_t x3763 = x3750 * x3757;
for(int x3759=0; x3759 < x3746; x3759++) {
int32_t x3770 = x3753 * x3759;
int32_t x3771 = x3769 + x3770;
int32_t x3776 = x3756 * x3759;
int32_t x3765 = x3749 * x3759;
for(int x3761=0; x3761 < x3748; x3761++) {
int32_t x3772 = x3754 * x3761;
int32_t x3773 = x3771 + x3772;
int32_t x3767 = x3748 * x3761;
for(int x3762=0; x3762 < x3748; x3762++) {
int32_t x3774 = x3755 * x3762;
int32_t x3775 = x3773 + x3774;
float x3777 = x3669[x3775];
float x3778 = x58[x3776];
int32_t x3764 = x3762 + x3763;
int32_t x3766 = x3764 + x3765;
int32_t x3768 = x3766 + x3767;
float x3779 = x3777 * x3778;
x3752[x3768] = x3779;

}

}

}

}
int32_t x3789 = 0;
int32_t x3790 = 1;
x3790 *= 1;
x3789 += 1;
x3790 *= 1;
x3790 *= 1;
int32_t x3795 = x3789;
bool x3796 = x3795 >= 2;
if (x3796) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3801 = x3795 == 0;
if (x3801) {
int32_t x3802 = x3790;
bool x3803 = x3802 == 256;
if (x3803) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3810 = x3790;
bool x3812 = x3746 == 1;
int32_t x3811 = 256 / x3810;
bool x3813 = x3811 == 1;
bool x3817;
if (x454) {
bool x3814 = x3812 || x3813;
bool x3815 = x3746 == x3811;
bool x3816 = x3814 || x3815;
x3817 = x3816;
} else {
x3817 = false;
}
bool x3821;
if (x3817) {
x3821 = x3820;
} else {
x3821 = false;
}
bool x3822;
if (x3821) {
x3822 = x3820;
} else {
x3822 = false;
}
if (x3822) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3746,x3748,x3748,1,x3811,1,1);
assert(false && "");
}
bool x3828 = x3746 <= x3811;
int32_t x3829;
if (x3828) {
x3829 = x3811;
} else {
x3829 = x3746;
}
int32_t x3833 = x3829 * x3832;
int32_t x3834 = 64 * x3833;
float* x3835 = (float*)myMalloc(x3834 * sizeof(float));;
int32_t x3836;
if (x3812) {
x3836 = 0;
} else {
x3836 = x3749;
}
int32_t x3839;
if (x3813) {
x3839 = 0;
} else {
x3839 = 1;
}
for(int x3840=0; x3840 < 64; x3840++) {
int32_t x3852 = x3750 * x3840;
int32_t x3846 = x3833 * x3840;
for(int x3842=0; x3842 < x3829; x3842++) {
int32_t x3853 = x3836 * x3842;
int32_t x3854 = x3852 + x3853;
int32_t x3859 = x3839 * x3842;
int32_t x3848 = x3832 * x3842;
for(int x3844=0; x3844 < x3831; x3844++) {
int32_t x3855 = x3837 * x3844;
int32_t x3856 = x3854 + x3855;
int32_t x3850 = x3831 * x3844;
for(int x3845=0; x3845 < x3831; x3845++) {
int32_t x3857 = x3838 * x3845;
int32_t x3858 = x3856 + x3857;
float x3860 = x3752[x3858];
float x3861 = x7[x3859];
int32_t x3847 = x3845 + x3846;
int32_t x3849 = x3847 + x3848;
int32_t x3851 = x3849 + x3850;
float x3862 = x3860 + x3861;
x3835[x3851] = x3862;

}

}

}

}
bool x3872 = x3829 == 1;
bool x3873 = x3872 || x2525;
bool x3874 = x3829 == x2128;
bool x3875 = x3873 || x3874;
bool x3880;
if (x3875) {
x3880 = x3879;
} else {
x3880 = false;
}
bool x3881;
if (x3880) {
x3881 = x3879;
} else {
x3881 = false;
}
if (x3881) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3829,x3831,x3831,64,x2128,x2130,x2130);
assert(false && "");
}
bool x3887 = x3829 <= x2128;
int32_t x3888;
if (x3887) {
x3888 = x2128;
} else {
x3888 = x3829;
}
int32_t x3894;
if (x3872) {
x3894 = 0;
} else {
x3894 = x3832;
}
for(int x3897=0; x3897 < 64; x3897++) {
int32_t x3903 = x3833 * x3897;
int32_t x3910 = x2132 * x3897;
for(int x3899=0; x3899 < x3888; x3899++) {
int32_t x3904 = x3894 * x3899;
int32_t x3905 = x3903 + x3904;
int32_t x3911 = x2549 * x3899;
int32_t x3912 = x3910 + x3911;
for(int x3901=0; x3901 < x3890; x3901++) {
int32_t x3906 = x3895 * x3901;
int32_t x3907 = x3905 + x3906;
int32_t x3913 = x2550 * x3901;
int32_t x3914 = x3912 + x3913;
for(int x3902=0; x3902 < x3890; x3902++) {
int32_t x3908 = x3896 * x3902;
int32_t x3909 = x3907 + x3908;
float x3917 = x3835[x3909];
int32_t x3915 = x2551 * x3902;
int32_t x3916 = x3914 + x3915;
float x3918 = x2587[x3916];
float x3919 = x3917 + x3918;
x3835[x3909] = x3919;

}

}

}

}
float* x3929 = (float*)myMalloc(x3834 * sizeof(float));;
for(int x3931=0; x3931 < x3834; x3931++) {
float x3932 = x3835[x3931];
bool x3933 = x3932 < 0.0f;
if (x3933) {
x3929[x3931] = 0.0f;
} else {
float x3936 = x3835[x3931];
x3929[x3931] = x3936;
}

}
float* x3950 = (float*)myMalloc(x3949 * sizeof(float));;
int32_t x3953 = 64 * x3829;
int32_t x3954 = x3953 * x3945;
float* x3955 = (float*)myMalloc(x3954 * sizeof(float));;
int32_t x3951 = x3829 * x3945;
for(int x3956=0; x3956 < 64; x3956++) {
int32_t x3957 = x3956 * x3833;
float* x3958 = x3929+x3957;
int32_t x3959 = x3956 * x3946;
float* x3960 = x3950+x3959;
int32_t x3961 = x3956 * x3951;
float* x3962 = x3955+x3961;
for(int x3963=0; x3963 < x3829; x3963++) {
int32_t x3964 = x3963 / 1;
int32_t x3968 = x3964 * x3944;
int32_t x3969 = x3968 * x3944;
int32_t x3965 = x3963 % 1;
int32_t x3966 = x3965 / 1;
int32_t x3970 = x3966 * x3944;
int32_t x3971 = x3970 * x3944;
int32_t x3972 = x3969 + x3971;
int32_t x3967 = x3965 % 1;
int32_t x3973 = x3967 * x3944;
int32_t x3974 = x3973 * x3944;
int32_t x3975 = x3972 + x3974;
float* x3976 = x3962+x3975;
int32_t x3977 = x3964 * x3831;
int32_t x3978 = x3977 * x3831;
float* x3979 = x3958+x3978;
for(int x3981=0; x3981 < x3944; x3981++) {
int32_t x3983 = x3981 * x3944;
float* x3984 = x3976+x3983;
int32_t x3982 = x3981 + x3966;
int32_t x3985 = x3982 * x3831;
int32_t x3986 = x3985 + x3967;
float* x3987 = x3979+x3986;
memcpy(x3984, x3987, 4 * x3944);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x3945,x3829,1,x150,x3829,x3962,x3945,1,x3960,x3945);

}
int32_t x3996 = 0;
int32_t x3997 = 1;
x3997 *= 1;
x3996 += 1;
x3997 *= 1;
x3997 *= 1;
int32_t x4002 = x3996;
bool x4003 = x4002 >= 2;
if (x4003) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4008 = x4002 == 0;
if (x4008) {
int32_t x4009 = x3997;
bool x4010 = x4009 == 64;
if (x4010) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4017 = x3997;
int32_t x4018 = 64 / x4017;
bool x4019 = x4018 == 1;
bool x4022;
if (x454) {
bool x4020 = 64 == x4018;
bool x4021 = x4019 || x4020;
x4022 = x4021;
} else {
x4022 = false;
}
bool x4026;
if (x4022) {
x4026 = x4025;
} else {
x4026 = false;
}
bool x4027;
if (x4026) {
x4027 = x4025;
} else {
x4027 = false;
}
if (x4027) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x3944,x3944,1,x4018,1,1);
assert(false && "");
}
bool x4033 = 64 <= x4018;
int32_t x4034;
if (x4033) {
x4034 = x4018;
} else {
x4034 = 64;
}
int32_t x4038 = x4034 * x4037;
int32_t x4039 = 64 * x4038;
float* x4040 = (float*)myMalloc(x4039 * sizeof(float));;
int32_t x4043;
if (x4019) {
x4043 = 0;
} else {
x4043 = 1;
}
for(int x4044=0; x4044 < 64; x4044++) {
int32_t x4056 = x3946 * x4044;
int32_t x4050 = x4038 * x4044;
for(int x4046=0; x4046 < x4034; x4046++) {
int32_t x4057 = x3945 * x4046;
int32_t x4058 = x4056 + x4057;
int32_t x4063 = x4043 * x4046;
int32_t x4052 = x4037 * x4046;
for(int x4048=0; x4048 < x4036; x4048++) {
int32_t x4059 = x4041 * x4048;
int32_t x4060 = x4058 + x4059;
int32_t x4054 = x4036 * x4048;
for(int x4049=0; x4049 < x4036; x4049++) {
int32_t x4061 = x4042 * x4049;
int32_t x4062 = x4060 + x4061;
float x4064 = x3950[x4062];
float x4065 = x257[x4063];
int32_t x4051 = x4049 + x4050;
int32_t x4053 = x4051 + x4052;
int32_t x4055 = x4053 + x4054;
float x4066 = x4064 - x4065;
x4040[x4055] = x4066;

}

}

}

}
float* x4076 = (float*)myMalloc(64 * sizeof(float));;
for(int x4077=0; x4077 < 64; x4077++) {
float x4078 = x187[x4077];
float x4079 = x4078 + 1.0E-5f;
x4076[x4077] = x4079;

}
float* x4083 = (float*)myMalloc(64 * sizeof(float));;
for(int x4084=0; x4084 < 64; x4084++) {
float x4085 = x4076[x4084];
double x4086 = (double)x4085;
double x4087 = sqrt(x4086);
float x4088 = (float)x4087;
x4083[x4084] = x4088;

}
int32_t x4092 = 0;
int32_t x4093 = 1;
x4093 *= 1;
x4092 += 1;
x4093 *= 1;
x4093 *= 1;
int32_t x4098 = x4092;
bool x4099 = x4098 >= 2;
if (x4099) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4104 = x4098 == 0;
if (x4104) {
int32_t x4105 = x4093;
bool x4106 = x4105 == 64;
if (x4106) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4113 = x4093;
bool x4115 = x4034 == 1;
int32_t x4114 = 64 / x4113;
bool x4116 = x4114 == 1;
bool x4120;
if (x454) {
bool x4117 = x4115 || x4116;
bool x4118 = x4034 == x4114;
bool x4119 = x4117 || x4118;
x4120 = x4119;
} else {
x4120 = false;
}
bool x4124;
if (x4120) {
x4124 = x4123;
} else {
x4124 = false;
}
bool x4125;
if (x4124) {
x4125 = x4123;
} else {
x4125 = false;
}
if (x4125) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4034,x4036,x4036,1,x4114,1,1);
assert(false && "");
}
bool x4131 = x4034 <= x4114;
int32_t x4132;
if (x4131) {
x4132 = x4114;
} else {
x4132 = x4034;
}
int32_t x4136 = x4132 * x4135;
int32_t x4137 = 64 * x4136;
float* x4138 = (float*)myMalloc(x4137 * sizeof(float));;
int32_t x4139;
if (x4115) {
x4139 = 0;
} else {
x4139 = x4037;
}
int32_t x4142;
if (x4116) {
x4142 = 0;
} else {
x4142 = 1;
}
for(int x4143=0; x4143 < 64; x4143++) {
int32_t x4155 = x4038 * x4143;
int32_t x4149 = x4136 * x4143;
for(int x4145=0; x4145 < x4132; x4145++) {
int32_t x4156 = x4139 * x4145;
int32_t x4157 = x4155 + x4156;
int32_t x4162 = x4142 * x4145;
int32_t x4151 = x4135 * x4145;
for(int x4147=0; x4147 < x4134; x4147++) {
int32_t x4158 = x4140 * x4147;
int32_t x4159 = x4157 + x4158;
int32_t x4153 = x4134 * x4147;
for(int x4148=0; x4148 < x4134; x4148++) {
int32_t x4160 = x4141 * x4148;
int32_t x4161 = x4159 + x4160;
float x4163 = x4040[x4161];
float x4164 = x4083[x4162];
int32_t x4150 = x4148 + x4149;
int32_t x4152 = x4150 + x4151;
int32_t x4154 = x4152 + x4153;
float x4165 = x4163 / x4164;
x4138[x4154] = x4165;

}

}

}

}
int32_t x4175 = 0;
int32_t x4176 = 1;
x4176 *= 1;
x4175 += 1;
x4176 *= 1;
x4176 *= 1;
int32_t x4181 = x4175;
bool x4182 = x4181 >= 2;
if (x4182) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4187 = x4181 == 0;
if (x4187) {
int32_t x4188 = x4176;
bool x4189 = x4188 == 64;
if (x4189) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4196 = x4176;
bool x4198 = x4132 == 1;
int32_t x4197 = 64 / x4196;
bool x4199 = x4197 == 1;
bool x4203;
if (x454) {
bool x4200 = x4198 || x4199;
bool x4201 = x4132 == x4197;
bool x4202 = x4200 || x4201;
x4203 = x4202;
} else {
x4203 = false;
}
bool x4207;
if (x4203) {
x4207 = x4206;
} else {
x4207 = false;
}
bool x4208;
if (x4207) {
x4208 = x4206;
} else {
x4208 = false;
}
if (x4208) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4132,x4134,x4134,1,x4197,1,1);
assert(false && "");
}
bool x4214 = x4132 <= x4197;
int32_t x4215;
if (x4214) {
x4215 = x4197;
} else {
x4215 = x4132;
}
int32_t x4219 = x4215 * x4218;
int32_t x4220 = 64 * x4219;
float* x4221 = (float*)myMalloc(x4220 * sizeof(float));;
int32_t x4222;
if (x4198) {
x4222 = 0;
} else {
x4222 = x4135;
}
int32_t x4225;
if (x4199) {
x4225 = 0;
} else {
x4225 = 1;
}
for(int x4226=0; x4226 < 64; x4226++) {
int32_t x4238 = x4136 * x4226;
int32_t x4232 = x4219 * x4226;
for(int x4228=0; x4228 < x4215; x4228++) {
int32_t x4239 = x4222 * x4228;
int32_t x4240 = x4238 + x4239;
int32_t x4245 = x4225 * x4228;
int32_t x4234 = x4218 * x4228;
for(int x4230=0; x4230 < x4217; x4230++) {
int32_t x4241 = x4223 * x4230;
int32_t x4242 = x4240 + x4241;
int32_t x4236 = x4217 * x4230;
for(int x4231=0; x4231 < x4217; x4231++) {
int32_t x4243 = x4224 * x4231;
int32_t x4244 = x4242 + x4243;
float x4246 = x4138[x4244];
float x4247 = x81[x4245];
int32_t x4233 = x4231 + x4232;
int32_t x4235 = x4233 + x4234;
int32_t x4237 = x4235 + x4236;
float x4248 = x4246 * x4247;
x4221[x4237] = x4248;

}

}

}

}
int32_t x4258 = 0;
int32_t x4259 = 1;
x4259 *= 1;
x4258 += 1;
x4259 *= 1;
x4259 *= 1;
int32_t x4264 = x4258;
bool x4265 = x4264 >= 2;
if (x4265) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4270 = x4264 == 0;
if (x4270) {
int32_t x4271 = x4259;
bool x4272 = x4271 == 64;
if (x4272) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4279 = x4259;
bool x4281 = x4215 == 1;
int32_t x4280 = 64 / x4279;
bool x4282 = x4280 == 1;
bool x4286;
if (x454) {
bool x4283 = x4281 || x4282;
bool x4284 = x4215 == x4280;
bool x4285 = x4283 || x4284;
x4286 = x4285;
} else {
x4286 = false;
}
bool x4290;
if (x4286) {
x4290 = x4289;
} else {
x4290 = false;
}
bool x4291;
if (x4290) {
x4291 = x4289;
} else {
x4291 = false;
}
if (x4291) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4215,x4217,x4217,1,x4280,1,1);
assert(false && "");
}
bool x4297 = x4215 <= x4280;
int32_t x4298;
if (x4297) {
x4298 = x4280;
} else {
x4298 = x4215;
}
int32_t x4302 = x4298 * x4301;
int32_t x4303 = 64 * x4302;
float* x4304 = (float*)myMalloc(x4303 * sizeof(float));;
int32_t x4305;
if (x4281) {
x4305 = 0;
} else {
x4305 = x4218;
}
int32_t x4308;
if (x4282) {
x4308 = 0;
} else {
x4308 = 1;
}
for(int x4309=0; x4309 < 64; x4309++) {
int32_t x4321 = x4219 * x4309;
int32_t x4315 = x4302 * x4309;
for(int x4311=0; x4311 < x4298; x4311++) {
int32_t x4322 = x4305 * x4311;
int32_t x4323 = x4321 + x4322;
int32_t x4328 = x4308 * x4311;
int32_t x4317 = x4301 * x4311;
for(int x4313=0; x4313 < x4300; x4313++) {
int32_t x4324 = x4306 * x4313;
int32_t x4325 = x4323 + x4324;
int32_t x4319 = x4300 * x4313;
for(int x4314=0; x4314 < x4300; x4314++) {
int32_t x4326 = x4307 * x4314;
int32_t x4327 = x4325 + x4326;
float x4329 = x4221[x4327];
float x4330 = x24[x4328];
int32_t x4316 = x4314 + x4315;
int32_t x4318 = x4316 + x4317;
int32_t x4320 = x4318 + x4319;
float x4331 = x4329 + x4330;
x4304[x4320] = x4331;

}

}

}

}
float* x4341 = (float*)myMalloc(x4303 * sizeof(float));;
for(int x4343=0; x4343 < x4303; x4343++) {
float x4344 = x4304[x4343];
bool x4345 = x4344 < 0.0f;
if (x4345) {
x4341[x4343] = 0.0f;
} else {
float x4348 = x4304[x4343];
x4341[x4343] = x4348;
}

}
float* x4363 = (float*)myMalloc(x4362 * sizeof(float));;
int32_t x4364 = 9 * x4298;
int32_t x4367 = 64 * x4364;
int32_t x4368 = x4367 * x4358;
float* x4369 = (float*)myMalloc(x4368 * sizeof(float));;
int32_t x4365 = x4364 * x4358;
int32_t x4377 = x4298 * 3;
int32_t x4378 = x4377 * 3;
for(int x4370=0; x4370 < 64; x4370++) {
int32_t x4371 = x4370 * x4302;
float* x4372 = x4341+x4371;
int32_t x4373 = x4370 * x4359;
float* x4374 = x4363+x4373;
int32_t x4375 = x4370 * x4365;
float* x4376 = x4369+x4375;
for(int x4380=0; x4380 < x4378; x4380++) {
int32_t x4381 = x4380 / 9;
int32_t x4385 = x4381 * 3;
int32_t x4386 = x4385 * 3;
int32_t x4387 = x4386 * x4357;
int32_t x4388 = x4387 * x4357;
int32_t x4382 = x4380 % 9;
int32_t x4383 = x4382 / 3;
int32_t x4389 = x4383 * 3;
int32_t x4390 = x4389 * x4357;
int32_t x4391 = x4390 * x4357;
int32_t x4392 = x4388 + x4391;
int32_t x4384 = x4382 % 3;
int32_t x4393 = x4384 * x4357;
int32_t x4394 = x4393 * x4357;
int32_t x4395 = x4392 + x4394;
float* x4396 = x4376+x4395;
int32_t x4397 = x4381 * x4300;
int32_t x4398 = x4397 * x4300;
float* x4399 = x4372+x4398;
int32_t x4412 = 1 - x4384;
bool x4413 = x4412 > 0;
int32_t x4414;
if (x4413) {
x4414 = x4412;
} else {
x4414 = 0;
}
int32_t x4415 = 3 - x4384;
int32_t x4416 = x4415 - 1;
int32_t x4417 = 1 - x4416;
bool x4418 = x4417 > 0;
int32_t x4419;
if (x4418) {
x4419 = x4417;
} else {
x4419 = 0;
}
int32_t x4420 = x4357 - x4419;
int32_t x4421 = x4420 - x4414;
bool x4422 = x4421 <= 0;
bool x4426 = x4414 > 0;
int32_t x4411 = -1 + x4384;
bool x4439 = x4419 > 0;
for(int x4401=0; x4401 < x4357; x4401++) {
int32_t x4402 = x4401 - 1;
int32_t x4403 = x4402 + x4383;
bool x4404 = x4403 < 0;
bool x4405 = x4403 >= x4300;
bool x4406 = x4404 || x4405;
if (x4406) {
int32_t x4407 = x4401 * x4357;
float* x4408 = x4396+x4407;
memset(x4408, 0, 4 * x4357);;
} else {
if (x4422) {
int32_t x4407 = x4401 * x4357;
float* x4423 = x4396+x4407;
memset(x4423, 0, 4 * x4357);;
} else {
int32_t x4407 = x4401 * x4357;
if (x4426) {
float* x4427 = x4396+x4407;
memset(x4427, 0, 4 * x4414);;
} else {
}
// may have segfault here
int32_t x4432 = x4407 + x4414;
float* x4433 = x4396+x4432;
int32_t x4434 = x4403 * x4300;
int32_t x4435 = x4434 + x4411;
int32_t x4436 = x4435 + x4414;
float* x4437 = x4399+x4436;
memcpy(x4433, x4437, 4 * x4421);;
if (x4439) {
int32_t x4440 = x4407 + x4357;
int32_t x4441 = x4440 - x4419;
float* x4442 = x4396+x4441;
memset(x4442, 0, 4 * x4419);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x4358,x4364,1,x73,x4364,x4376,x4358,1,x4374,x4358);

}
int32_t x4457 = 0;
int32_t x4458 = 1;
x4458 *= 1;
x4457 += 1;
x4458 *= 1;
x4458 *= 1;
int32_t x4463 = x4457;
bool x4464 = x4463 >= 2;
if (x4464) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4469 = x4463 == 0;
if (x4469) {
int32_t x4470 = x4458;
bool x4471 = x4470 == 64;
if (x4471) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4478 = x4458;
int32_t x4479 = 64 / x4478;
bool x4480 = x4479 == 1;
bool x4483;
if (x454) {
bool x4481 = 64 == x4479;
bool x4482 = x4480 || x4481;
x4483 = x4482;
} else {
x4483 = false;
}
bool x4487;
if (x4483) {
x4487 = x4486;
} else {
x4487 = false;
}
bool x4488;
if (x4487) {
x4488 = x4486;
} else {
x4488 = false;
}
if (x4488) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x4357,x4357,1,x4479,1,1);
assert(false && "");
}
bool x4494 = 64 <= x4479;
int32_t x4495;
if (x4494) {
x4495 = x4479;
} else {
x4495 = 64;
}
int32_t x4499 = x4495 * x4498;
int32_t x4500 = 64 * x4499;
float* x4501 = (float*)myMalloc(x4500 * sizeof(float));;
int32_t x4504;
if (x4480) {
x4504 = 0;
} else {
x4504 = 1;
}
for(int x4505=0; x4505 < 64; x4505++) {
int32_t x4517 = x4359 * x4505;
int32_t x4511 = x4499 * x4505;
for(int x4507=0; x4507 < x4495; x4507++) {
int32_t x4518 = x4358 * x4507;
int32_t x4519 = x4517 + x4518;
int32_t x4524 = x4504 * x4507;
int32_t x4513 = x4498 * x4507;
for(int x4509=0; x4509 < x4497; x4509++) {
int32_t x4520 = x4502 * x4509;
int32_t x4521 = x4519 + x4520;
int32_t x4515 = x4497 * x4509;
for(int x4510=0; x4510 < x4497; x4510++) {
int32_t x4522 = x4503 * x4510;
int32_t x4523 = x4521 + x4522;
float x4525 = x4363[x4523];
float x4526 = x179[x4524];
int32_t x4512 = x4510 + x4511;
int32_t x4514 = x4512 + x4513;
int32_t x4516 = x4514 + x4515;
float x4527 = x4525 - x4526;
x4501[x4516] = x4527;

}

}

}

}
float* x4537 = (float*)myMalloc(64 * sizeof(float));;
for(int x4538=0; x4538 < 64; x4538++) {
float x4539 = x118[x4538];
float x4540 = x4539 + 1.0E-5f;
x4537[x4538] = x4540;

}
float* x4544 = (float*)myMalloc(64 * sizeof(float));;
for(int x4545=0; x4545 < 64; x4545++) {
float x4546 = x4537[x4545];
double x4547 = (double)x4546;
double x4548 = sqrt(x4547);
float x4549 = (float)x4548;
x4544[x4545] = x4549;

}
int32_t x4553 = 0;
int32_t x4554 = 1;
x4554 *= 1;
x4553 += 1;
x4554 *= 1;
x4554 *= 1;
int32_t x4559 = x4553;
bool x4560 = x4559 >= 2;
if (x4560) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4565 = x4559 == 0;
if (x4565) {
int32_t x4566 = x4554;
bool x4567 = x4566 == 64;
if (x4567) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4574 = x4554;
bool x4576 = x4495 == 1;
int32_t x4575 = 64 / x4574;
bool x4577 = x4575 == 1;
bool x4581;
if (x454) {
bool x4578 = x4576 || x4577;
bool x4579 = x4495 == x4575;
bool x4580 = x4578 || x4579;
x4581 = x4580;
} else {
x4581 = false;
}
bool x4585;
if (x4581) {
x4585 = x4584;
} else {
x4585 = false;
}
bool x4586;
if (x4585) {
x4586 = x4584;
} else {
x4586 = false;
}
if (x4586) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4495,x4497,x4497,1,x4575,1,1);
assert(false && "");
}
bool x4592 = x4495 <= x4575;
int32_t x4593;
if (x4592) {
x4593 = x4575;
} else {
x4593 = x4495;
}
int32_t x4597 = x4593 * x4596;
int32_t x4598 = 64 * x4597;
float* x4599 = (float*)myMalloc(x4598 * sizeof(float));;
int32_t x4600;
if (x4576) {
x4600 = 0;
} else {
x4600 = x4498;
}
int32_t x4603;
if (x4577) {
x4603 = 0;
} else {
x4603 = 1;
}
for(int x4604=0; x4604 < 64; x4604++) {
int32_t x4616 = x4499 * x4604;
int32_t x4610 = x4597 * x4604;
for(int x4606=0; x4606 < x4593; x4606++) {
int32_t x4617 = x4600 * x4606;
int32_t x4618 = x4616 + x4617;
int32_t x4623 = x4603 * x4606;
int32_t x4612 = x4596 * x4606;
for(int x4608=0; x4608 < x4595; x4608++) {
int32_t x4619 = x4601 * x4608;
int32_t x4620 = x4618 + x4619;
int32_t x4614 = x4595 * x4608;
for(int x4609=0; x4609 < x4595; x4609++) {
int32_t x4621 = x4602 * x4609;
int32_t x4622 = x4620 + x4621;
float x4624 = x4501[x4622];
float x4625 = x4544[x4623];
int32_t x4611 = x4609 + x4610;
int32_t x4613 = x4611 + x4612;
int32_t x4615 = x4613 + x4614;
float x4626 = x4624 / x4625;
x4599[x4615] = x4626;

}

}

}

}
int32_t x4636 = 0;
int32_t x4637 = 1;
x4637 *= 1;
x4636 += 1;
x4637 *= 1;
x4637 *= 1;
int32_t x4642 = x4636;
bool x4643 = x4642 >= 2;
if (x4643) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4648 = x4642 == 0;
if (x4648) {
int32_t x4649 = x4637;
bool x4650 = x4649 == 64;
if (x4650) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4657 = x4637;
bool x4659 = x4593 == 1;
int32_t x4658 = 64 / x4657;
bool x4660 = x4658 == 1;
bool x4664;
if (x454) {
bool x4661 = x4659 || x4660;
bool x4662 = x4593 == x4658;
bool x4663 = x4661 || x4662;
x4664 = x4663;
} else {
x4664 = false;
}
bool x4668;
if (x4664) {
x4668 = x4667;
} else {
x4668 = false;
}
bool x4669;
if (x4668) {
x4669 = x4667;
} else {
x4669 = false;
}
if (x4669) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4593,x4595,x4595,1,x4658,1,1);
assert(false && "");
}
bool x4675 = x4593 <= x4658;
int32_t x4676;
if (x4675) {
x4676 = x4658;
} else {
x4676 = x4593;
}
int32_t x4680 = x4676 * x4679;
int32_t x4681 = 64 * x4680;
float* x4682 = (float*)myMalloc(x4681 * sizeof(float));;
int32_t x4683;
if (x4659) {
x4683 = 0;
} else {
x4683 = x4596;
}
int32_t x4686;
if (x4660) {
x4686 = 0;
} else {
x4686 = 1;
}
for(int x4687=0; x4687 < 64; x4687++) {
int32_t x4699 = x4597 * x4687;
int32_t x4693 = x4680 * x4687;
for(int x4689=0; x4689 < x4676; x4689++) {
int32_t x4700 = x4683 * x4689;
int32_t x4701 = x4699 + x4700;
int32_t x4706 = x4686 * x4689;
int32_t x4695 = x4679 * x4689;
for(int x4691=0; x4691 < x4678; x4691++) {
int32_t x4702 = x4684 * x4691;
int32_t x4703 = x4701 + x4702;
int32_t x4697 = x4678 * x4691;
for(int x4692=0; x4692 < x4678; x4692++) {
int32_t x4704 = x4685 * x4692;
int32_t x4705 = x4703 + x4704;
float x4707 = x4599[x4705];
float x4708 = x72[x4706];
int32_t x4694 = x4692 + x4693;
int32_t x4696 = x4694 + x4695;
int32_t x4698 = x4696 + x4697;
float x4709 = x4707 * x4708;
x4682[x4698] = x4709;

}

}

}

}
int32_t x4719 = 0;
int32_t x4720 = 1;
x4720 *= 1;
x4719 += 1;
x4720 *= 1;
x4720 *= 1;
int32_t x4725 = x4719;
bool x4726 = x4725 >= 2;
if (x4726) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4731 = x4725 == 0;
if (x4731) {
int32_t x4732 = x4720;
bool x4733 = x4732 == 64;
if (x4733) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4740 = x4720;
bool x4742 = x4676 == 1;
int32_t x4741 = 64 / x4740;
bool x4743 = x4741 == 1;
bool x4747;
if (x454) {
bool x4744 = x4742 || x4743;
bool x4745 = x4676 == x4741;
bool x4746 = x4744 || x4745;
x4747 = x4746;
} else {
x4747 = false;
}
bool x4751;
if (x4747) {
x4751 = x4750;
} else {
x4751 = false;
}
bool x4752;
if (x4751) {
x4752 = x4750;
} else {
x4752 = false;
}
if (x4752) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4676,x4678,x4678,1,x4741,1,1);
assert(false && "");
}
bool x4758 = x4676 <= x4741;
int32_t x4759;
if (x4758) {
x4759 = x4741;
} else {
x4759 = x4676;
}
int32_t x4763 = x4759 * x4762;
int32_t x4764 = 64 * x4763;
float* x4765 = (float*)myMalloc(x4764 * sizeof(float));;
int32_t x4766;
if (x4742) {
x4766 = 0;
} else {
x4766 = x4679;
}
int32_t x4769;
if (x4743) {
x4769 = 0;
} else {
x4769 = 1;
}
for(int x4770=0; x4770 < 64; x4770++) {
int32_t x4782 = x4680 * x4770;
int32_t x4776 = x4763 * x4770;
for(int x4772=0; x4772 < x4759; x4772++) {
int32_t x4783 = x4766 * x4772;
int32_t x4784 = x4782 + x4783;
int32_t x4789 = x4769 * x4772;
int32_t x4778 = x4762 * x4772;
for(int x4774=0; x4774 < x4761; x4774++) {
int32_t x4785 = x4767 * x4774;
int32_t x4786 = x4784 + x4785;
int32_t x4780 = x4761 * x4774;
for(int x4775=0; x4775 < x4761; x4775++) {
int32_t x4787 = x4768 * x4775;
int32_t x4788 = x4786 + x4787;
float x4790 = x4682[x4788];
float x4791 = x135[x4789];
int32_t x4777 = x4775 + x4776;
int32_t x4779 = x4777 + x4778;
int32_t x4781 = x4779 + x4780;
float x4792 = x4790 + x4791;
x4765[x4781] = x4792;

}

}

}

}
float* x4802 = (float*)myMalloc(x4764 * sizeof(float));;
for(int x4804=0; x4804 < x4764; x4804++) {
float x4805 = x4765[x4804];
bool x4806 = x4805 < 0.0f;
if (x4806) {
x4802[x4804] = 0.0f;
} else {
float x4809 = x4765[x4804];
x4802[x4804] = x4809;
}

}
float* x4823 = (float*)myMalloc(x4822 * sizeof(float));;
int32_t x4826 = 64 * x4759;
int32_t x4827 = x4826 * x4818;
float* x4828 = (float*)myMalloc(x4827 * sizeof(float));;
int32_t x4824 = x4759 * x4818;
for(int x4829=0; x4829 < 64; x4829++) {
int32_t x4830 = x4829 * x4763;
float* x4831 = x4802+x4830;
int32_t x4832 = x4829 * x4819;
float* x4833 = x4823+x4832;
int32_t x4834 = x4829 * x4824;
float* x4835 = x4828+x4834;
for(int x4836=0; x4836 < x4759; x4836++) {
int32_t x4837 = x4836 / 1;
int32_t x4841 = x4837 * x4817;
int32_t x4842 = x4841 * x4817;
int32_t x4838 = x4836 % 1;
int32_t x4839 = x4838 / 1;
int32_t x4843 = x4839 * x4817;
int32_t x4844 = x4843 * x4817;
int32_t x4845 = x4842 + x4844;
int32_t x4840 = x4838 % 1;
int32_t x4846 = x4840 * x4817;
int32_t x4847 = x4846 * x4817;
int32_t x4848 = x4845 + x4847;
float* x4849 = x4835+x4848;
int32_t x4850 = x4837 * x4761;
int32_t x4851 = x4850 * x4761;
float* x4852 = x4831+x4851;
for(int x4854=0; x4854 < x4817; x4854++) {
int32_t x4856 = x4854 * x4817;
float* x4857 = x4849+x4856;
int32_t x4855 = x4854 + x4839;
int32_t x4858 = x4855 * x4761;
int32_t x4859 = x4858 + x4840;
float* x4860 = x4852+x4859;
memcpy(x4857, x4860, 4 * x4817);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x4818,x4759,1,x87,x4759,x4835,x4818,1,x4833,x4818);

}
int32_t x4869 = 0;
int32_t x4870 = 1;
x4870 *= 1;
x4869 += 1;
x4870 *= 1;
x4870 *= 1;
int32_t x4875 = x4869;
bool x4876 = x4875 >= 2;
if (x4876) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4881 = x4875 == 0;
if (x4881) {
int32_t x4882 = x4870;
bool x4883 = x4882 == 256;
if (x4883) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4890 = x4870;
int32_t x4891 = 256 / x4890;
bool x4892 = x4891 == 1;
bool x4895;
if (x454) {
bool x4893 = 256 == x4891;
bool x4894 = x4892 || x4893;
x4895 = x4894;
} else {
x4895 = false;
}
bool x4899;
if (x4895) {
x4899 = x4898;
} else {
x4899 = false;
}
bool x4900;
if (x4899) {
x4900 = x4898;
} else {
x4900 = false;
}
if (x4900) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x4817,x4817,1,x4891,1,1);
assert(false && "");
}
bool x4906 = 256 <= x4891;
int32_t x4907;
if (x4906) {
x4907 = x4891;
} else {
x4907 = 256;
}
int32_t x4911 = x4907 * x4910;
int32_t x4912 = 64 * x4911;
float* x4913 = (float*)myMalloc(x4912 * sizeof(float));;
int32_t x4916;
if (x4892) {
x4916 = 0;
} else {
x4916 = 1;
}
for(int x4917=0; x4917 < 64; x4917++) {
int32_t x4929 = x4819 * x4917;
int32_t x4923 = x4911 * x4917;
for(int x4919=0; x4919 < x4907; x4919++) {
int32_t x4930 = x4818 * x4919;
int32_t x4931 = x4929 + x4930;
int32_t x4936 = x4916 * x4919;
int32_t x4925 = x4910 * x4919;
for(int x4921=0; x4921 < x4909; x4921++) {
int32_t x4932 = x4914 * x4921;
int32_t x4933 = x4931 + x4932;
int32_t x4927 = x4909 * x4921;
for(int x4922=0; x4922 < x4909; x4922++) {
int32_t x4934 = x4915 * x4922;
int32_t x4935 = x4933 + x4934;
float x4937 = x4823[x4935];
float x4938 = x184[x4936];
int32_t x4924 = x4922 + x4923;
int32_t x4926 = x4924 + x4925;
int32_t x4928 = x4926 + x4927;
float x4939 = x4937 - x4938;
x4913[x4928] = x4939;

}

}

}

}
float* x4949 = (float*)myMalloc(256 * sizeof(float));;
for(int x4950=0; x4950 < 256; x4950++) {
float x4951 = x133[x4950];
float x4952 = x4951 + 1.0E-5f;
x4949[x4950] = x4952;

}
float* x4956 = (float*)myMalloc(256 * sizeof(float));;
for(int x4957=0; x4957 < 256; x4957++) {
float x4958 = x4949[x4957];
double x4959 = (double)x4958;
double x4960 = sqrt(x4959);
float x4961 = (float)x4960;
x4956[x4957] = x4961;

}
int32_t x4965 = 0;
int32_t x4966 = 1;
x4966 *= 1;
x4965 += 1;
x4966 *= 1;
x4966 *= 1;
int32_t x4971 = x4965;
bool x4972 = x4971 >= 2;
if (x4972) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4977 = x4971 == 0;
if (x4977) {
int32_t x4978 = x4966;
bool x4979 = x4978 == 256;
if (x4979) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4986 = x4966;
bool x4988 = x4907 == 1;
int32_t x4987 = 256 / x4986;
bool x4989 = x4987 == 1;
bool x4993;
if (x454) {
bool x4990 = x4988 || x4989;
bool x4991 = x4907 == x4987;
bool x4992 = x4990 || x4991;
x4993 = x4992;
} else {
x4993 = false;
}
bool x4997;
if (x4993) {
x4997 = x4996;
} else {
x4997 = false;
}
bool x4998;
if (x4997) {
x4998 = x4996;
} else {
x4998 = false;
}
if (x4998) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4907,x4909,x4909,1,x4987,1,1);
assert(false && "");
}
bool x5004 = x4907 <= x4987;
int32_t x5005;
if (x5004) {
x5005 = x4987;
} else {
x5005 = x4907;
}
int32_t x5009 = x5005 * x5008;
int32_t x5010 = 64 * x5009;
float* x5011 = (float*)myMalloc(x5010 * sizeof(float));;
int32_t x5012;
if (x4988) {
x5012 = 0;
} else {
x5012 = x4910;
}
int32_t x5015;
if (x4989) {
x5015 = 0;
} else {
x5015 = 1;
}
for(int x5016=0; x5016 < 64; x5016++) {
int32_t x5028 = x4911 * x5016;
int32_t x5022 = x5009 * x5016;
for(int x5018=0; x5018 < x5005; x5018++) {
int32_t x5029 = x5012 * x5018;
int32_t x5030 = x5028 + x5029;
int32_t x5035 = x5015 * x5018;
int32_t x5024 = x5008 * x5018;
for(int x5020=0; x5020 < x5007; x5020++) {
int32_t x5031 = x5013 * x5020;
int32_t x5032 = x5030 + x5031;
int32_t x5026 = x5007 * x5020;
for(int x5021=0; x5021 < x5007; x5021++) {
int32_t x5033 = x5014 * x5021;
int32_t x5034 = x5032 + x5033;
float x5036 = x4913[x5034];
float x5037 = x4956[x5035];
int32_t x5023 = x5021 + x5022;
int32_t x5025 = x5023 + x5024;
int32_t x5027 = x5025 + x5026;
float x5038 = x5036 / x5037;
x5011[x5027] = x5038;

}

}

}

}
int32_t x5048 = 0;
int32_t x5049 = 1;
x5049 *= 1;
x5048 += 1;
x5049 *= 1;
x5049 *= 1;
int32_t x5054 = x5048;
bool x5055 = x5054 >= 2;
if (x5055) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5060 = x5054 == 0;
if (x5060) {
int32_t x5061 = x5049;
bool x5062 = x5061 == 256;
if (x5062) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5069 = x5049;
bool x5071 = x5005 == 1;
int32_t x5070 = 256 / x5069;
bool x5072 = x5070 == 1;
bool x5076;
if (x454) {
bool x5073 = x5071 || x5072;
bool x5074 = x5005 == x5070;
bool x5075 = x5073 || x5074;
x5076 = x5075;
} else {
x5076 = false;
}
bool x5080;
if (x5076) {
x5080 = x5079;
} else {
x5080 = false;
}
bool x5081;
if (x5080) {
x5081 = x5079;
} else {
x5081 = false;
}
if (x5081) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5005,x5007,x5007,1,x5070,1,1);
assert(false && "");
}
bool x5087 = x5005 <= x5070;
int32_t x5088;
if (x5087) {
x5088 = x5070;
} else {
x5088 = x5005;
}
int32_t x5092 = x5088 * x5091;
int32_t x5093 = 64 * x5092;
float* x5094 = (float*)myMalloc(x5093 * sizeof(float));;
int32_t x5095;
if (x5071) {
x5095 = 0;
} else {
x5095 = x5008;
}
int32_t x5098;
if (x5072) {
x5098 = 0;
} else {
x5098 = 1;
}
for(int x5099=0; x5099 < 64; x5099++) {
int32_t x5111 = x5009 * x5099;
int32_t x5105 = x5092 * x5099;
for(int x5101=0; x5101 < x5088; x5101++) {
int32_t x5112 = x5095 * x5101;
int32_t x5113 = x5111 + x5112;
int32_t x5118 = x5098 * x5101;
int32_t x5107 = x5091 * x5101;
for(int x5103=0; x5103 < x5090; x5103++) {
int32_t x5114 = x5096 * x5103;
int32_t x5115 = x5113 + x5114;
int32_t x5109 = x5090 * x5103;
for(int x5104=0; x5104 < x5090; x5104++) {
int32_t x5116 = x5097 * x5104;
int32_t x5117 = x5115 + x5116;
float x5119 = x5011[x5117];
float x5120 = x37[x5118];
int32_t x5106 = x5104 + x5105;
int32_t x5108 = x5106 + x5107;
int32_t x5110 = x5108 + x5109;
float x5121 = x5119 * x5120;
x5094[x5110] = x5121;

}

}

}

}
int32_t x5131 = 0;
int32_t x5132 = 1;
x5132 *= 1;
x5131 += 1;
x5132 *= 1;
x5132 *= 1;
int32_t x5137 = x5131;
bool x5138 = x5137 >= 2;
if (x5138) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5143 = x5137 == 0;
if (x5143) {
int32_t x5144 = x5132;
bool x5145 = x5144 == 256;
if (x5145) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5152 = x5132;
bool x5154 = x5088 == 1;
int32_t x5153 = 256 / x5152;
bool x5155 = x5153 == 1;
bool x5159;
if (x454) {
bool x5156 = x5154 || x5155;
bool x5157 = x5088 == x5153;
bool x5158 = x5156 || x5157;
x5159 = x5158;
} else {
x5159 = false;
}
bool x5163;
if (x5159) {
x5163 = x5162;
} else {
x5163 = false;
}
bool x5164;
if (x5163) {
x5164 = x5162;
} else {
x5164 = false;
}
if (x5164) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5088,x5090,x5090,1,x5153,1,1);
assert(false && "");
}
bool x5170 = x5088 <= x5153;
int32_t x5171;
if (x5170) {
x5171 = x5153;
} else {
x5171 = x5088;
}
int32_t x5175 = x5171 * x5174;
int32_t x5176 = 64 * x5175;
float* x5177 = (float*)myMalloc(x5176 * sizeof(float));;
int32_t x5178;
if (x5154) {
x5178 = 0;
} else {
x5178 = x5091;
}
int32_t x5181;
if (x5155) {
x5181 = 0;
} else {
x5181 = 1;
}
for(int x5182=0; x5182 < 64; x5182++) {
int32_t x5194 = x5092 * x5182;
int32_t x5188 = x5175 * x5182;
for(int x5184=0; x5184 < x5171; x5184++) {
int32_t x5195 = x5178 * x5184;
int32_t x5196 = x5194 + x5195;
int32_t x5201 = x5181 * x5184;
int32_t x5190 = x5174 * x5184;
for(int x5186=0; x5186 < x5173; x5186++) {
int32_t x5197 = x5179 * x5186;
int32_t x5198 = x5196 + x5197;
int32_t x5192 = x5173 * x5186;
for(int x5187=0; x5187 < x5173; x5187++) {
int32_t x5199 = x5180 * x5187;
int32_t x5200 = x5198 + x5199;
float x5202 = x5094[x5200];
float x5203 = x247[x5201];
int32_t x5189 = x5187 + x5188;
int32_t x5191 = x5189 + x5190;
int32_t x5193 = x5191 + x5192;
float x5204 = x5202 + x5203;
x5177[x5193] = x5204;

}

}

}

}
bool x5214 = x5171 == 1;
bool x5215 = x5214 || x3872;
bool x5216 = x5171 == x3829;
bool x5217 = x5215 || x5216;
bool x5222;
if (x5217) {
x5222 = x5221;
} else {
x5222 = false;
}
bool x5223;
if (x5222) {
x5223 = x5221;
} else {
x5223 = false;
}
if (x5223) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5171,x5173,x5173,64,x3829,x3831,x3831);
assert(false && "");
}
bool x5229 = x5171 <= x3829;
int32_t x5230;
if (x5229) {
x5230 = x3829;
} else {
x5230 = x5171;
}
int32_t x5236;
if (x5214) {
x5236 = 0;
} else {
x5236 = x5174;
}
for(int x5239=0; x5239 < 64; x5239++) {
int32_t x5245 = x5175 * x5239;
int32_t x5252 = x3833 * x5239;
for(int x5241=0; x5241 < x5230; x5241++) {
int32_t x5246 = x5236 * x5241;
int32_t x5247 = x5245 + x5246;
int32_t x5253 = x3894 * x5241;
int32_t x5254 = x5252 + x5253;
for(int x5243=0; x5243 < x5232; x5243++) {
int32_t x5248 = x5237 * x5243;
int32_t x5249 = x5247 + x5248;
int32_t x5255 = x3895 * x5243;
int32_t x5256 = x5254 + x5255;
for(int x5244=0; x5244 < x5232; x5244++) {
int32_t x5250 = x5238 * x5244;
int32_t x5251 = x5249 + x5250;
float x5259 = x5177[x5251];
int32_t x5257 = x3896 * x5244;
int32_t x5258 = x5256 + x5257;
float x5260 = x3929[x5258];
float x5261 = x5259 + x5260;
x5177[x5251] = x5261;

}

}

}

}
float* x5271 = (float*)myMalloc(x5176 * sizeof(float));;
for(int x5273=0; x5273 < x5176; x5273++) {
float x5274 = x5177[x5273];
bool x5275 = x5274 < 0.0f;
if (x5275) {
x5271[x5273] = 0.0f;
} else {
float x5278 = x5177[x5273];
x5271[x5273] = x5278;
}

}
float* x5292 = (float*)myMalloc(x5291 * sizeof(float));;
int32_t x5295 = 64 * x5171;
int32_t x5296 = x5295 * x5287;
float* x5297 = (float*)myMalloc(x5296 * sizeof(float));;
int32_t x5293 = x5171 * x5287;
for(int x5298=0; x5298 < 64; x5298++) {
int32_t x5299 = x5298 * x5175;
float* x5300 = x5271+x5299;
int32_t x5301 = x5298 * x5288;
float* x5302 = x5292+x5301;
int32_t x5303 = x5298 * x5293;
float* x5304 = x5297+x5303;
for(int x5305=0; x5305 < x5171; x5305++) {
int32_t x5306 = x5305 / 1;
int32_t x5310 = x5306 * x5286;
int32_t x5311 = x5310 * x5286;
int32_t x5307 = x5305 % 1;
int32_t x5308 = x5307 / 1;
int32_t x5312 = x5308 * x5286;
int32_t x5313 = x5312 * x5286;
int32_t x5314 = x5311 + x5313;
int32_t x5309 = x5307 % 1;
int32_t x5315 = x5309 * x5286;
int32_t x5316 = x5315 * x5286;
int32_t x5317 = x5314 + x5316;
float* x5318 = x5304+x5317;
int32_t x5319 = x5306 * x5173;
int32_t x5320 = x5319 * x5173;
float* x5321 = x5300+x5320;
for(int x5323=0; x5323 < x5286; x5323++) {
int32_t x5325 = x5323 * x5286;
float* x5326 = x5318+x5325;
int32_t x5324 = x5323 + x5308;
int32_t x5327 = x5324 * x5173;
int32_t x5328 = x5327 + x5309;
float* x5329 = x5321+x5328;
memcpy(x5326, x5329, 4 * x5286);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x5287,x5171,1,x11,x5171,x5304,x5287,1,x5302,x5287);

}
int32_t x5338 = 0;
int32_t x5339 = 1;
x5339 *= 1;
x5338 += 1;
x5339 *= 1;
x5339 *= 1;
int32_t x5344 = x5338;
bool x5345 = x5344 >= 2;
if (x5345) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5350 = x5344 == 0;
if (x5350) {
int32_t x5351 = x5339;
bool x5352 = x5351 == 128;
if (x5352) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5359 = x5339;
int32_t x5360 = 128 / x5359;
bool x5361 = x5360 == 1;
bool x5364;
if (x454) {
bool x5362 = 128 == x5360;
bool x5363 = x5361 || x5362;
x5364 = x5363;
} else {
x5364 = false;
}
bool x5368;
if (x5364) {
x5368 = x5367;
} else {
x5368 = false;
}
bool x5369;
if (x5368) {
x5369 = x5367;
} else {
x5369 = false;
}
if (x5369) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x5286,x5286,1,x5360,1,1);
assert(false && "");
}
bool x5375 = 128 <= x5360;
int32_t x5376;
if (x5375) {
x5376 = x5360;
} else {
x5376 = 128;
}
int32_t x5380 = x5376 * x5379;
int32_t x5381 = 64 * x5380;
float* x5382 = (float*)myMalloc(x5381 * sizeof(float));;
int32_t x5385;
if (x5361) {
x5385 = 0;
} else {
x5385 = 1;
}
for(int x5386=0; x5386 < 64; x5386++) {
int32_t x5398 = x5288 * x5386;
int32_t x5392 = x5380 * x5386;
for(int x5388=0; x5388 < x5376; x5388++) {
int32_t x5399 = x5287 * x5388;
int32_t x5400 = x5398 + x5399;
int32_t x5405 = x5385 * x5388;
int32_t x5394 = x5379 * x5388;
for(int x5390=0; x5390 < x5378; x5390++) {
int32_t x5401 = x5383 * x5390;
int32_t x5402 = x5400 + x5401;
int32_t x5396 = x5378 * x5390;
for(int x5391=0; x5391 < x5378; x5391++) {
int32_t x5403 = x5384 * x5391;
int32_t x5404 = x5402 + x5403;
float x5406 = x5292[x5404];
float x5407 = x204[x5405];
int32_t x5393 = x5391 + x5392;
int32_t x5395 = x5393 + x5394;
int32_t x5397 = x5395 + x5396;
float x5408 = x5406 - x5407;
x5382[x5397] = x5408;

}

}

}

}
float* x5418 = (float*)myMalloc(128 * sizeof(float));;
for(int x5420=0; x5420 < 128; x5420++) {
float x5421 = x134[x5420];
float x5422 = x5421 + 1.0E-5f;
x5418[x5420] = x5422;

}
float* x5426 = (float*)myMalloc(128 * sizeof(float));;
for(int x5427=0; x5427 < 128; x5427++) {
float x5428 = x5418[x5427];
double x5429 = (double)x5428;
double x5430 = sqrt(x5429);
float x5431 = (float)x5430;
x5426[x5427] = x5431;

}
int32_t x5435 = 0;
int32_t x5436 = 1;
x5436 *= 1;
x5435 += 1;
x5436 *= 1;
x5436 *= 1;
int32_t x5441 = x5435;
bool x5442 = x5441 >= 2;
if (x5442) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5447 = x5441 == 0;
if (x5447) {
int32_t x5448 = x5436;
bool x5449 = x5448 == 128;
if (x5449) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5456 = x5436;
bool x5458 = x5376 == 1;
int32_t x5457 = 128 / x5456;
bool x5459 = x5457 == 1;
bool x5463;
if (x454) {
bool x5460 = x5458 || x5459;
bool x5461 = x5376 == x5457;
bool x5462 = x5460 || x5461;
x5463 = x5462;
} else {
x5463 = false;
}
bool x5467;
if (x5463) {
x5467 = x5466;
} else {
x5467 = false;
}
bool x5468;
if (x5467) {
x5468 = x5466;
} else {
x5468 = false;
}
if (x5468) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5376,x5378,x5378,1,x5457,1,1);
assert(false && "");
}
bool x5474 = x5376 <= x5457;
int32_t x5475;
if (x5474) {
x5475 = x5457;
} else {
x5475 = x5376;
}
int32_t x5479 = x5475 * x5478;
int32_t x5480 = 64 * x5479;
float* x5481 = (float*)myMalloc(x5480 * sizeof(float));;
int32_t x5482;
if (x5458) {
x5482 = 0;
} else {
x5482 = x5379;
}
int32_t x5485;
if (x5459) {
x5485 = 0;
} else {
x5485 = 1;
}
for(int x5486=0; x5486 < 64; x5486++) {
int32_t x5498 = x5380 * x5486;
int32_t x5492 = x5479 * x5486;
for(int x5488=0; x5488 < x5475; x5488++) {
int32_t x5499 = x5482 * x5488;
int32_t x5500 = x5498 + x5499;
int32_t x5505 = x5485 * x5488;
int32_t x5494 = x5478 * x5488;
for(int x5490=0; x5490 < x5477; x5490++) {
int32_t x5501 = x5483 * x5490;
int32_t x5502 = x5500 + x5501;
int32_t x5496 = x5477 * x5490;
for(int x5491=0; x5491 < x5477; x5491++) {
int32_t x5503 = x5484 * x5491;
int32_t x5504 = x5502 + x5503;
float x5506 = x5382[x5504];
float x5507 = x5426[x5505];
int32_t x5493 = x5491 + x5492;
int32_t x5495 = x5493 + x5494;
int32_t x5497 = x5495 + x5496;
float x5508 = x5506 / x5507;
x5481[x5497] = x5508;

}

}

}

}
int32_t x5518 = 0;
int32_t x5519 = 1;
x5519 *= 1;
x5518 += 1;
x5519 *= 1;
x5519 *= 1;
int32_t x5524 = x5518;
bool x5525 = x5524 >= 2;
if (x5525) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5530 = x5524 == 0;
if (x5530) {
int32_t x5531 = x5519;
bool x5532 = x5531 == 128;
if (x5532) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5539 = x5519;
bool x5541 = x5475 == 1;
int32_t x5540 = 128 / x5539;
bool x5542 = x5540 == 1;
bool x5546;
if (x454) {
bool x5543 = x5541 || x5542;
bool x5544 = x5475 == x5540;
bool x5545 = x5543 || x5544;
x5546 = x5545;
} else {
x5546 = false;
}
bool x5550;
if (x5546) {
x5550 = x5549;
} else {
x5550 = false;
}
bool x5551;
if (x5550) {
x5551 = x5549;
} else {
x5551 = false;
}
if (x5551) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5475,x5477,x5477,1,x5540,1,1);
assert(false && "");
}
bool x5557 = x5475 <= x5540;
int32_t x5558;
if (x5557) {
x5558 = x5540;
} else {
x5558 = x5475;
}
int32_t x5562 = x5558 * x5561;
int32_t x5563 = 64 * x5562;
float* x5564 = (float*)myMalloc(x5563 * sizeof(float));;
int32_t x5565;
if (x5541) {
x5565 = 0;
} else {
x5565 = x5478;
}
int32_t x5568;
if (x5542) {
x5568 = 0;
} else {
x5568 = 1;
}
for(int x5569=0; x5569 < 64; x5569++) {
int32_t x5581 = x5479 * x5569;
int32_t x5575 = x5562 * x5569;
for(int x5571=0; x5571 < x5558; x5571++) {
int32_t x5582 = x5565 * x5571;
int32_t x5583 = x5581 + x5582;
int32_t x5588 = x5568 * x5571;
int32_t x5577 = x5561 * x5571;
for(int x5573=0; x5573 < x5560; x5573++) {
int32_t x5584 = x5566 * x5573;
int32_t x5585 = x5583 + x5584;
int32_t x5579 = x5560 * x5573;
for(int x5574=0; x5574 < x5560; x5574++) {
int32_t x5586 = x5567 * x5574;
int32_t x5587 = x5585 + x5586;
float x5589 = x5481[x5587];
float x5590 = x84[x5588];
int32_t x5576 = x5574 + x5575;
int32_t x5578 = x5576 + x5577;
int32_t x5580 = x5578 + x5579;
float x5591 = x5589 * x5590;
x5564[x5580] = x5591;

}

}

}

}
int32_t x5601 = 0;
int32_t x5602 = 1;
x5602 *= 1;
x5601 += 1;
x5602 *= 1;
x5602 *= 1;
int32_t x5607 = x5601;
bool x5608 = x5607 >= 2;
if (x5608) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5613 = x5607 == 0;
if (x5613) {
int32_t x5614 = x5602;
bool x5615 = x5614 == 128;
if (x5615) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5622 = x5602;
bool x5624 = x5558 == 1;
int32_t x5623 = 128 / x5622;
bool x5625 = x5623 == 1;
bool x5629;
if (x454) {
bool x5626 = x5624 || x5625;
bool x5627 = x5558 == x5623;
bool x5628 = x5626 || x5627;
x5629 = x5628;
} else {
x5629 = false;
}
bool x5633;
if (x5629) {
x5633 = x5632;
} else {
x5633 = false;
}
bool x5634;
if (x5633) {
x5634 = x5632;
} else {
x5634 = false;
}
if (x5634) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5558,x5560,x5560,1,x5623,1,1);
assert(false && "");
}
bool x5640 = x5558 <= x5623;
int32_t x5641;
if (x5640) {
x5641 = x5623;
} else {
x5641 = x5558;
}
int32_t x5645 = x5641 * x5644;
int32_t x5646 = 64 * x5645;
float* x5647 = (float*)myMalloc(x5646 * sizeof(float));;
int32_t x5648;
if (x5624) {
x5648 = 0;
} else {
x5648 = x5561;
}
int32_t x5651;
if (x5625) {
x5651 = 0;
} else {
x5651 = 1;
}
for(int x5652=0; x5652 < 64; x5652++) {
int32_t x5664 = x5562 * x5652;
int32_t x5658 = x5645 * x5652;
for(int x5654=0; x5654 < x5641; x5654++) {
int32_t x5665 = x5648 * x5654;
int32_t x5666 = x5664 + x5665;
int32_t x5671 = x5651 * x5654;
int32_t x5660 = x5644 * x5654;
for(int x5656=0; x5656 < x5643; x5656++) {
int32_t x5667 = x5649 * x5656;
int32_t x5668 = x5666 + x5667;
int32_t x5662 = x5643 * x5656;
for(int x5657=0; x5657 < x5643; x5657++) {
int32_t x5669 = x5650 * x5657;
int32_t x5670 = x5668 + x5669;
float x5672 = x5564[x5670];
float x5673 = x172[x5671];
int32_t x5659 = x5657 + x5658;
int32_t x5661 = x5659 + x5660;
int32_t x5663 = x5661 + x5662;
float x5674 = x5672 + x5673;
x5647[x5663] = x5674;

}

}

}

}
float* x5684 = (float*)myMalloc(x5646 * sizeof(float));;
for(int x5686=0; x5686 < x5646; x5686++) {
float x5687 = x5647[x5686];
bool x5688 = x5687 < 0.0f;
if (x5688) {
x5684[x5686] = 0.0f;
} else {
float x5691 = x5647[x5686];
x5684[x5686] = x5691;
}

}
float* x5706 = (float*)myMalloc(x5705 * sizeof(float));;
int32_t x5707 = 9 * x5641;
int32_t x5710 = 64 * x5707;
int32_t x5711 = x5710 * x5701;
float* x5712 = (float*)myMalloc(x5711 * sizeof(float));;
int32_t x5708 = x5707 * x5701;
int32_t x5720 = x5641 * 3;
int32_t x5721 = x5720 * 3;
for(int x5713=0; x5713 < 64; x5713++) {
int32_t x5714 = x5713 * x5645;
float* x5715 = x5684+x5714;
int32_t x5716 = x5713 * x5702;
float* x5717 = x5706+x5716;
int32_t x5718 = x5713 * x5708;
float* x5719 = x5712+x5718;
for(int x5723=0; x5723 < x5721; x5723++) {
int32_t x5724 = x5723 / 9;
int32_t x5728 = x5724 * 3;
int32_t x5729 = x5728 * 3;
int32_t x5730 = x5729 * x5700;
int32_t x5731 = x5730 * x5700;
int32_t x5725 = x5723 % 9;
int32_t x5726 = x5725 / 3;
int32_t x5732 = x5726 * 3;
int32_t x5733 = x5732 * x5700;
int32_t x5734 = x5733 * x5700;
int32_t x5735 = x5731 + x5734;
int32_t x5727 = x5725 % 3;
int32_t x5736 = x5727 * x5700;
int32_t x5737 = x5736 * x5700;
int32_t x5738 = x5735 + x5737;
float* x5739 = x5719+x5738;
int32_t x5740 = x5724 * x5643;
int32_t x5741 = x5740 * x5643;
float* x5742 = x5715+x5741;
for(int x5744=0; x5744 < x5700; x5744++) {
int32_t x5745 = x5744 * 2;
int32_t x5746 = x5745 - 1;
int32_t x5747 = x5746 + x5726;
bool x5748 = x5747 < 0;
bool x5749 = x5747 >= x5643;
bool x5750 = x5748 || x5749;
if (x5750) {
int32_t x5751 = x5744 * x5700;
float* x5752 = x5739+x5751;
memset(x5752, 0, 4 * x5700);;
} else {
int32_t x5751 = x5744 * x5700;
int32_t x5767 = x5747 * x5643;
for(int x5755=0; x5755 < x5700; x5755++) {
int32_t x5756 = x5755 * 2;
int32_t x5757 = x5756 - 1;
int32_t x5758 = x5757 + x5727;
bool x5759 = x5758 < 0;
bool x5760 = x5758 >= x5643;
bool x5761 = x5759 || x5760;
if (x5761) {
int32_t x5762 = x5751 + x5755;
float* x5763 = x5739+x5762;
memset(x5763, 0, 4 * 1);;
} else {
int32_t x5762 = x5751 + x5755;
float* x5766 = x5739+x5762;
int32_t x5768 = x5767 + x5758;
float* x5769 = x5742+x5768;
memcpy(x5766, x5769, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x5701,x5707,1,x27,x5707,x5719,x5701,1,x5717,x5701);

}
int32_t x5784 = 0;
int32_t x5785 = 1;
x5785 *= 1;
x5784 += 1;
x5785 *= 1;
x5785 *= 1;
int32_t x5790 = x5784;
bool x5791 = x5790 >= 2;
if (x5791) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5796 = x5790 == 0;
if (x5796) {
int32_t x5797 = x5785;
bool x5798 = x5797 == 128;
if (x5798) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5805 = x5785;
int32_t x5806 = 128 / x5805;
bool x5807 = x5806 == 1;
bool x5810;
if (x454) {
bool x5808 = 128 == x5806;
bool x5809 = x5807 || x5808;
x5810 = x5809;
} else {
x5810 = false;
}
bool x5814;
if (x5810) {
x5814 = x5813;
} else {
x5814 = false;
}
bool x5815;
if (x5814) {
x5815 = x5813;
} else {
x5815 = false;
}
if (x5815) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x5700,x5700,1,x5806,1,1);
assert(false && "");
}
bool x5821 = 128 <= x5806;
int32_t x5822;
if (x5821) {
x5822 = x5806;
} else {
x5822 = 128;
}
int32_t x5826 = x5822 * x5825;
int32_t x5827 = 64 * x5826;
float* x5828 = (float*)myMalloc(x5827 * sizeof(float));;
int32_t x5831;
if (x5807) {
x5831 = 0;
} else {
x5831 = 1;
}
for(int x5832=0; x5832 < 64; x5832++) {
int32_t x5844 = x5702 * x5832;
int32_t x5838 = x5826 * x5832;
for(int x5834=0; x5834 < x5822; x5834++) {
int32_t x5845 = x5701 * x5834;
int32_t x5846 = x5844 + x5845;
int32_t x5851 = x5831 * x5834;
int32_t x5840 = x5825 * x5834;
for(int x5836=0; x5836 < x5824; x5836++) {
int32_t x5847 = x5829 * x5836;
int32_t x5848 = x5846 + x5847;
int32_t x5842 = x5824 * x5836;
for(int x5837=0; x5837 < x5824; x5837++) {
int32_t x5849 = x5830 * x5837;
int32_t x5850 = x5848 + x5849;
float x5852 = x5706[x5850];
float x5853 = x128[x5851];
int32_t x5839 = x5837 + x5838;
int32_t x5841 = x5839 + x5840;
int32_t x5843 = x5841 + x5842;
float x5854 = x5852 - x5853;
x5828[x5843] = x5854;

}

}

}

}
float* x5864 = (float*)myMalloc(128 * sizeof(float));;
for(int x5865=0; x5865 < 128; x5865++) {
float x5866 = x43[x5865];
float x5867 = x5866 + 1.0E-5f;
x5864[x5865] = x5867;

}
float* x5871 = (float*)myMalloc(128 * sizeof(float));;
for(int x5872=0; x5872 < 128; x5872++) {
float x5873 = x5864[x5872];
double x5874 = (double)x5873;
double x5875 = sqrt(x5874);
float x5876 = (float)x5875;
x5871[x5872] = x5876;

}
int32_t x5880 = 0;
int32_t x5881 = 1;
x5881 *= 1;
x5880 += 1;
x5881 *= 1;
x5881 *= 1;
int32_t x5886 = x5880;
bool x5887 = x5886 >= 2;
if (x5887) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5892 = x5886 == 0;
if (x5892) {
int32_t x5893 = x5881;
bool x5894 = x5893 == 128;
if (x5894) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5901 = x5881;
bool x5903 = x5822 == 1;
int32_t x5902 = 128 / x5901;
bool x5904 = x5902 == 1;
bool x5908;
if (x454) {
bool x5905 = x5903 || x5904;
bool x5906 = x5822 == x5902;
bool x5907 = x5905 || x5906;
x5908 = x5907;
} else {
x5908 = false;
}
bool x5912;
if (x5908) {
x5912 = x5911;
} else {
x5912 = false;
}
bool x5913;
if (x5912) {
x5913 = x5911;
} else {
x5913 = false;
}
if (x5913) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5822,x5824,x5824,1,x5902,1,1);
assert(false && "");
}
bool x5919 = x5822 <= x5902;
int32_t x5920;
if (x5919) {
x5920 = x5902;
} else {
x5920 = x5822;
}
int32_t x5924 = x5920 * x5923;
int32_t x5925 = 64 * x5924;
float* x5926 = (float*)myMalloc(x5925 * sizeof(float));;
int32_t x5927;
if (x5903) {
x5927 = 0;
} else {
x5927 = x5825;
}
int32_t x5930;
if (x5904) {
x5930 = 0;
} else {
x5930 = 1;
}
for(int x5931=0; x5931 < 64; x5931++) {
int32_t x5943 = x5826 * x5931;
int32_t x5937 = x5924 * x5931;
for(int x5933=0; x5933 < x5920; x5933++) {
int32_t x5944 = x5927 * x5933;
int32_t x5945 = x5943 + x5944;
int32_t x5950 = x5930 * x5933;
int32_t x5939 = x5923 * x5933;
for(int x5935=0; x5935 < x5922; x5935++) {
int32_t x5946 = x5928 * x5935;
int32_t x5947 = x5945 + x5946;
int32_t x5941 = x5922 * x5935;
for(int x5936=0; x5936 < x5922; x5936++) {
int32_t x5948 = x5929 * x5936;
int32_t x5949 = x5947 + x5948;
float x5951 = x5828[x5949];
float x5952 = x5871[x5950];
int32_t x5938 = x5936 + x5937;
int32_t x5940 = x5938 + x5939;
int32_t x5942 = x5940 + x5941;
float x5953 = x5951 / x5952;
x5926[x5942] = x5953;

}

}

}

}
int32_t x5963 = 0;
int32_t x5964 = 1;
x5964 *= 1;
x5963 += 1;
x5964 *= 1;
x5964 *= 1;
int32_t x5969 = x5963;
bool x5970 = x5969 >= 2;
if (x5970) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5975 = x5969 == 0;
if (x5975) {
int32_t x5976 = x5964;
bool x5977 = x5976 == 128;
if (x5977) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5984 = x5964;
bool x5986 = x5920 == 1;
int32_t x5985 = 128 / x5984;
bool x5987 = x5985 == 1;
bool x5991;
if (x454) {
bool x5988 = x5986 || x5987;
bool x5989 = x5920 == x5985;
bool x5990 = x5988 || x5989;
x5991 = x5990;
} else {
x5991 = false;
}
bool x5995;
if (x5991) {
x5995 = x5994;
} else {
x5995 = false;
}
bool x5996;
if (x5995) {
x5996 = x5994;
} else {
x5996 = false;
}
if (x5996) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5920,x5922,x5922,1,x5985,1,1);
assert(false && "");
}
bool x6002 = x5920 <= x5985;
int32_t x6003;
if (x6002) {
x6003 = x5985;
} else {
x6003 = x5920;
}
int32_t x6007 = x6003 * x6006;
int32_t x6008 = 64 * x6007;
float* x6009 = (float*)myMalloc(x6008 * sizeof(float));;
int32_t x6010;
if (x5986) {
x6010 = 0;
} else {
x6010 = x5923;
}
int32_t x6013;
if (x5987) {
x6013 = 0;
} else {
x6013 = 1;
}
for(int x6014=0; x6014 < 64; x6014++) {
int32_t x6026 = x5924 * x6014;
int32_t x6020 = x6007 * x6014;
for(int x6016=0; x6016 < x6003; x6016++) {
int32_t x6027 = x6010 * x6016;
int32_t x6028 = x6026 + x6027;
int32_t x6033 = x6013 * x6016;
int32_t x6022 = x6006 * x6016;
for(int x6018=0; x6018 < x6005; x6018++) {
int32_t x6029 = x6011 * x6018;
int32_t x6030 = x6028 + x6029;
int32_t x6024 = x6005 * x6018;
for(int x6019=0; x6019 < x6005; x6019++) {
int32_t x6031 = x6012 * x6019;
int32_t x6032 = x6030 + x6031;
float x6034 = x5926[x6032];
float x6035 = x252[x6033];
int32_t x6021 = x6019 + x6020;
int32_t x6023 = x6021 + x6022;
int32_t x6025 = x6023 + x6024;
float x6036 = x6034 * x6035;
x6009[x6025] = x6036;

}

}

}

}
int32_t x6046 = 0;
int32_t x6047 = 1;
x6047 *= 1;
x6046 += 1;
x6047 *= 1;
x6047 *= 1;
int32_t x6052 = x6046;
bool x6053 = x6052 >= 2;
if (x6053) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6058 = x6052 == 0;
if (x6058) {
int32_t x6059 = x6047;
bool x6060 = x6059 == 128;
if (x6060) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6067 = x6047;
bool x6069 = x6003 == 1;
int32_t x6068 = 128 / x6067;
bool x6070 = x6068 == 1;
bool x6074;
if (x454) {
bool x6071 = x6069 || x6070;
bool x6072 = x6003 == x6068;
bool x6073 = x6071 || x6072;
x6074 = x6073;
} else {
x6074 = false;
}
bool x6078;
if (x6074) {
x6078 = x6077;
} else {
x6078 = false;
}
bool x6079;
if (x6078) {
x6079 = x6077;
} else {
x6079 = false;
}
if (x6079) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6003,x6005,x6005,1,x6068,1,1);
assert(false && "");
}
bool x6085 = x6003 <= x6068;
int32_t x6086;
if (x6085) {
x6086 = x6068;
} else {
x6086 = x6003;
}
int32_t x6090 = x6086 * x6089;
int32_t x6091 = 64 * x6090;
float* x6092 = (float*)myMalloc(x6091 * sizeof(float));;
int32_t x6093;
if (x6069) {
x6093 = 0;
} else {
x6093 = x6006;
}
int32_t x6096;
if (x6070) {
x6096 = 0;
} else {
x6096 = 1;
}
for(int x6097=0; x6097 < 64; x6097++) {
int32_t x6109 = x6007 * x6097;
int32_t x6103 = x6090 * x6097;
for(int x6099=0; x6099 < x6086; x6099++) {
int32_t x6110 = x6093 * x6099;
int32_t x6111 = x6109 + x6110;
int32_t x6116 = x6096 * x6099;
int32_t x6105 = x6089 * x6099;
for(int x6101=0; x6101 < x6088; x6101++) {
int32_t x6112 = x6094 * x6101;
int32_t x6113 = x6111 + x6112;
int32_t x6107 = x6088 * x6101;
for(int x6102=0; x6102 < x6088; x6102++) {
int32_t x6114 = x6095 * x6102;
int32_t x6115 = x6113 + x6114;
float x6117 = x6009[x6115];
float x6118 = x190[x6116];
int32_t x6104 = x6102 + x6103;
int32_t x6106 = x6104 + x6105;
int32_t x6108 = x6106 + x6107;
float x6119 = x6117 + x6118;
x6092[x6108] = x6119;

}

}

}

}
float* x6129 = (float*)myMalloc(x6091 * sizeof(float));;
for(int x6131=0; x6131 < x6091; x6131++) {
float x6132 = x6092[x6131];
bool x6133 = x6132 < 0.0f;
if (x6133) {
x6129[x6131] = 0.0f;
} else {
float x6136 = x6092[x6131];
x6129[x6131] = x6136;
}

}
float* x6150 = (float*)myMalloc(x6149 * sizeof(float));;
int32_t x6153 = 64 * x6086;
int32_t x6154 = x6153 * x6145;
float* x6155 = (float*)myMalloc(x6154 * sizeof(float));;
int32_t x6151 = x6086 * x6145;
for(int x6156=0; x6156 < 64; x6156++) {
int32_t x6157 = x6156 * x6090;
float* x6158 = x6129+x6157;
int32_t x6159 = x6156 * x6146;
float* x6160 = x6150+x6159;
int32_t x6161 = x6156 * x6151;
float* x6162 = x6155+x6161;
for(int x6163=0; x6163 < x6086; x6163++) {
int32_t x6164 = x6163 / 1;
int32_t x6168 = x6164 * x6144;
int32_t x6169 = x6168 * x6144;
int32_t x6165 = x6163 % 1;
int32_t x6166 = x6165 / 1;
int32_t x6170 = x6166 * x6144;
int32_t x6171 = x6170 * x6144;
int32_t x6172 = x6169 + x6171;
int32_t x6167 = x6165 % 1;
int32_t x6173 = x6167 * x6144;
int32_t x6174 = x6173 * x6144;
int32_t x6175 = x6172 + x6174;
float* x6176 = x6162+x6175;
int32_t x6177 = x6164 * x6088;
int32_t x6178 = x6177 * x6088;
float* x6179 = x6158+x6178;
for(int x6181=0; x6181 < x6144; x6181++) {
int32_t x6183 = x6181 * x6144;
float* x6184 = x6176+x6183;
int32_t x6182 = x6181 + x6166;
int32_t x6185 = x6182 * x6088;
int32_t x6186 = x6185 + x6167;
float* x6187 = x6179+x6186;
memcpy(x6184, x6187, 4 * x6144);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x6145,x6086,1,x106,x6086,x6162,x6145,1,x6160,x6145);

}
int32_t x6196 = 0;
int32_t x6197 = 1;
x6197 *= 1;
x6196 += 1;
x6197 *= 1;
x6197 *= 1;
int32_t x6202 = x6196;
bool x6203 = x6202 >= 2;
if (x6203) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6208 = x6202 == 0;
if (x6208) {
int32_t x6209 = x6197;
bool x6210 = x6209 == 512;
if (x6210) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6217 = x6197;
int32_t x6218 = 512 / x6217;
bool x6219 = x6218 == 1;
bool x6222;
if (x454) {
bool x6220 = 512 == x6218;
bool x6221 = x6219 || x6220;
x6222 = x6221;
} else {
x6222 = false;
}
bool x6226;
if (x6222) {
x6226 = x6225;
} else {
x6226 = false;
}
bool x6227;
if (x6226) {
x6227 = x6225;
} else {
x6227 = false;
}
if (x6227) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x6144,x6144,1,x6218,1,1);
assert(false && "");
}
bool x6233 = 512 <= x6218;
int32_t x6234;
if (x6233) {
x6234 = x6218;
} else {
x6234 = 512;
}
int32_t x6238 = x6234 * x6237;
int32_t x6239 = 64 * x6238;
float* x6240 = (float*)myMalloc(x6239 * sizeof(float));;
int32_t x6243;
if (x6219) {
x6243 = 0;
} else {
x6243 = 1;
}
for(int x6244=0; x6244 < 64; x6244++) {
int32_t x6256 = x6146 * x6244;
int32_t x6250 = x6238 * x6244;
for(int x6246=0; x6246 < x6234; x6246++) {
int32_t x6257 = x6145 * x6246;
int32_t x6258 = x6256 + x6257;
int32_t x6263 = x6243 * x6246;
int32_t x6252 = x6237 * x6246;
for(int x6248=0; x6248 < x6236; x6248++) {
int32_t x6259 = x6241 * x6248;
int32_t x6260 = x6258 + x6259;
int32_t x6254 = x6236 * x6248;
for(int x6249=0; x6249 < x6236; x6249++) {
int32_t x6261 = x6242 * x6249;
int32_t x6262 = x6260 + x6261;
float x6264 = x6150[x6262];
float x6265 = x149[x6263];
int32_t x6251 = x6249 + x6250;
int32_t x6253 = x6251 + x6252;
int32_t x6255 = x6253 + x6254;
float x6266 = x6264 - x6265;
x6240[x6255] = x6266;

}

}

}

}
float* x6276 = (float*)myMalloc(512 * sizeof(float));;
for(int x6278=0; x6278 < 512; x6278++) {
float x6279 = x101[x6278];
float x6280 = x6279 + 1.0E-5f;
x6276[x6278] = x6280;

}
float* x6284 = (float*)myMalloc(512 * sizeof(float));;
for(int x6285=0; x6285 < 512; x6285++) {
float x6286 = x6276[x6285];
double x6287 = (double)x6286;
double x6288 = sqrt(x6287);
float x6289 = (float)x6288;
x6284[x6285] = x6289;

}
int32_t x6293 = 0;
int32_t x6294 = 1;
x6294 *= 1;
x6293 += 1;
x6294 *= 1;
x6294 *= 1;
int32_t x6299 = x6293;
bool x6300 = x6299 >= 2;
if (x6300) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6305 = x6299 == 0;
if (x6305) {
int32_t x6306 = x6294;
bool x6307 = x6306 == 512;
if (x6307) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6314 = x6294;
bool x6316 = x6234 == 1;
int32_t x6315 = 512 / x6314;
bool x6317 = x6315 == 1;
bool x6321;
if (x454) {
bool x6318 = x6316 || x6317;
bool x6319 = x6234 == x6315;
bool x6320 = x6318 || x6319;
x6321 = x6320;
} else {
x6321 = false;
}
bool x6325;
if (x6321) {
x6325 = x6324;
} else {
x6325 = false;
}
bool x6326;
if (x6325) {
x6326 = x6324;
} else {
x6326 = false;
}
if (x6326) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6234,x6236,x6236,1,x6315,1,1);
assert(false && "");
}
bool x6332 = x6234 <= x6315;
int32_t x6333;
if (x6332) {
x6333 = x6315;
} else {
x6333 = x6234;
}
int32_t x6337 = x6333 * x6336;
int32_t x6338 = 64 * x6337;
float* x6339 = (float*)myMalloc(x6338 * sizeof(float));;
int32_t x6340;
if (x6316) {
x6340 = 0;
} else {
x6340 = x6237;
}
int32_t x6343;
if (x6317) {
x6343 = 0;
} else {
x6343 = 1;
}
for(int x6344=0; x6344 < 64; x6344++) {
int32_t x6356 = x6238 * x6344;
int32_t x6350 = x6337 * x6344;
for(int x6346=0; x6346 < x6333; x6346++) {
int32_t x6357 = x6340 * x6346;
int32_t x6358 = x6356 + x6357;
int32_t x6363 = x6343 * x6346;
int32_t x6352 = x6336 * x6346;
for(int x6348=0; x6348 < x6335; x6348++) {
int32_t x6359 = x6341 * x6348;
int32_t x6360 = x6358 + x6359;
int32_t x6354 = x6335 * x6348;
for(int x6349=0; x6349 < x6335; x6349++) {
int32_t x6361 = x6342 * x6349;
int32_t x6362 = x6360 + x6361;
float x6364 = x6240[x6362];
float x6365 = x6284[x6363];
int32_t x6351 = x6349 + x6350;
int32_t x6353 = x6351 + x6352;
int32_t x6355 = x6353 + x6354;
float x6366 = x6364 / x6365;
x6339[x6355] = x6366;

}

}

}

}
int32_t x6376 = 0;
int32_t x6377 = 1;
x6377 *= 1;
x6376 += 1;
x6377 *= 1;
x6377 *= 1;
int32_t x6382 = x6376;
bool x6383 = x6382 >= 2;
if (x6383) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6388 = x6382 == 0;
if (x6388) {
int32_t x6389 = x6377;
bool x6390 = x6389 == 512;
if (x6390) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6397 = x6377;
bool x6399 = x6333 == 1;
int32_t x6398 = 512 / x6397;
bool x6400 = x6398 == 1;
bool x6404;
if (x454) {
bool x6401 = x6399 || x6400;
bool x6402 = x6333 == x6398;
bool x6403 = x6401 || x6402;
x6404 = x6403;
} else {
x6404 = false;
}
bool x6408;
if (x6404) {
x6408 = x6407;
} else {
x6408 = false;
}
bool x6409;
if (x6408) {
x6409 = x6407;
} else {
x6409 = false;
}
if (x6409) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6333,x6335,x6335,1,x6398,1,1);
assert(false && "");
}
bool x6415 = x6333 <= x6398;
int32_t x6416;
if (x6415) {
x6416 = x6398;
} else {
x6416 = x6333;
}
int32_t x6420 = x6416 * x6419;
int32_t x6421 = 64 * x6420;
float* x6422 = (float*)myMalloc(x6421 * sizeof(float));;
int32_t x6423;
if (x6399) {
x6423 = 0;
} else {
x6423 = x6336;
}
int32_t x6426;
if (x6400) {
x6426 = 0;
} else {
x6426 = 1;
}
for(int x6427=0; x6427 < 64; x6427++) {
int32_t x6439 = x6337 * x6427;
int32_t x6433 = x6420 * x6427;
for(int x6429=0; x6429 < x6416; x6429++) {
int32_t x6440 = x6423 * x6429;
int32_t x6441 = x6439 + x6440;
int32_t x6446 = x6426 * x6429;
int32_t x6435 = x6419 * x6429;
for(int x6431=0; x6431 < x6418; x6431++) {
int32_t x6442 = x6424 * x6431;
int32_t x6443 = x6441 + x6442;
int32_t x6437 = x6418 * x6431;
for(int x6432=0; x6432 < x6418; x6432++) {
int32_t x6444 = x6425 * x6432;
int32_t x6445 = x6443 + x6444;
float x6447 = x6339[x6445];
float x6448 = x145[x6446];
int32_t x6434 = x6432 + x6433;
int32_t x6436 = x6434 + x6435;
int32_t x6438 = x6436 + x6437;
float x6449 = x6447 * x6448;
x6422[x6438] = x6449;

}

}

}

}
int32_t x6459 = 0;
int32_t x6460 = 1;
x6460 *= 1;
x6459 += 1;
x6460 *= 1;
x6460 *= 1;
int32_t x6465 = x6459;
bool x6466 = x6465 >= 2;
if (x6466) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6471 = x6465 == 0;
if (x6471) {
int32_t x6472 = x6460;
bool x6473 = x6472 == 512;
if (x6473) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6480 = x6460;
bool x6482 = x6416 == 1;
int32_t x6481 = 512 / x6480;
bool x6483 = x6481 == 1;
bool x6487;
if (x454) {
bool x6484 = x6482 || x6483;
bool x6485 = x6416 == x6481;
bool x6486 = x6484 || x6485;
x6487 = x6486;
} else {
x6487 = false;
}
bool x6491;
if (x6487) {
x6491 = x6490;
} else {
x6491 = false;
}
bool x6492;
if (x6491) {
x6492 = x6490;
} else {
x6492 = false;
}
if (x6492) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6416,x6418,x6418,1,x6481,1,1);
assert(false && "");
}
bool x6498 = x6416 <= x6481;
int32_t x6499;
if (x6498) {
x6499 = x6481;
} else {
x6499 = x6416;
}
int32_t x6503 = x6499 * x6502;
int32_t x6504 = 64 * x6503;
float* x6505 = (float*)myMalloc(x6504 * sizeof(float));;
int32_t x6506;
if (x6482) {
x6506 = 0;
} else {
x6506 = x6419;
}
int32_t x6509;
if (x6483) {
x6509 = 0;
} else {
x6509 = 1;
}
for(int x6510=0; x6510 < 64; x6510++) {
int32_t x6522 = x6420 * x6510;
int32_t x6516 = x6503 * x6510;
for(int x6512=0; x6512 < x6499; x6512++) {
int32_t x6523 = x6506 * x6512;
int32_t x6524 = x6522 + x6523;
int32_t x6529 = x6509 * x6512;
int32_t x6518 = x6502 * x6512;
for(int x6514=0; x6514 < x6501; x6514++) {
int32_t x6525 = x6507 * x6514;
int32_t x6526 = x6524 + x6525;
int32_t x6520 = x6501 * x6514;
for(int x6515=0; x6515 < x6501; x6515++) {
int32_t x6527 = x6508 * x6515;
int32_t x6528 = x6526 + x6527;
float x6530 = x6422[x6528];
float x6531 = x210[x6529];
int32_t x6517 = x6515 + x6516;
int32_t x6519 = x6517 + x6518;
int32_t x6521 = x6519 + x6520;
float x6532 = x6530 + x6531;
x6505[x6521] = x6532;

}

}

}

}
float* x6549 = (float*)myMalloc(x6548 * sizeof(float));;
int32_t x6552 = x5295 * x6544;
float* x6553 = (float*)myMalloc(x6552 * sizeof(float));;
int32_t x6550 = x5171 * x6544;
for(int x6554=0; x6554 < 64; x6554++) {
int32_t x6555 = x6554 * x5175;
float* x6556 = x5271+x6555;
int32_t x6557 = x6554 * x6545;
float* x6558 = x6549+x6557;
int32_t x6559 = x6554 * x6550;
float* x6560 = x6553+x6559;
for(int x6561=0; x6561 < x5171; x6561++) {
int32_t x6562 = x6561 / 1;
int32_t x6566 = x6562 * x6543;
int32_t x6567 = x6566 * x6543;
int32_t x6563 = x6561 % 1;
int32_t x6564 = x6563 / 1;
int32_t x6568 = x6564 * x6543;
int32_t x6569 = x6568 * x6543;
int32_t x6570 = x6567 + x6569;
int32_t x6565 = x6563 % 1;
int32_t x6571 = x6565 * x6543;
int32_t x6572 = x6571 * x6543;
int32_t x6573 = x6570 + x6572;
float* x6574 = x6560+x6573;
int32_t x6575 = x6562 * x5173;
int32_t x6576 = x6575 * x5173;
float* x6577 = x6556+x6576;
for(int x6579=0; x6579 < x6543; x6579++) {
int32_t x6583 = x6579 * x6543;
int32_t x6580 = x6579 * 2;
int32_t x6581 = x6580 + x6564;
int32_t x6586 = x6581 * x5173;
int32_t x6587 = x6586 + x6565;
for(int x6582=0; x6582 < x6543; x6582++) {
int32_t x6584 = x6583 + x6582;
float* x6585 = x6574+x6584;
int32_t x6588 = x6582 * 2;
int32_t x6589 = x6587 + x6588;
float* x6590 = x6577+x6589;
memcpy(x6585, x6590, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x6544,x5171,1,x258,x5171,x6560,x6544,1,x6558,x6544);

}
int32_t x6601 = 0;
int32_t x6602 = 1;
x6602 *= 1;
x6601 += 1;
x6602 *= 1;
x6602 *= 1;
int32_t x6607 = x6601;
bool x6608 = x6607 >= 2;
if (x6608) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6613 = x6607 == 0;
if (x6613) {
int32_t x6614 = x6602;
bool x6615 = x6614 == 512;
if (x6615) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6622 = x6602;
int32_t x6623 = 512 / x6622;
bool x6624 = x6623 == 1;
bool x6627;
if (x454) {
bool x6625 = 512 == x6623;
bool x6626 = x6624 || x6625;
x6627 = x6626;
} else {
x6627 = false;
}
bool x6631;
if (x6627) {
x6631 = x6630;
} else {
x6631 = false;
}
bool x6632;
if (x6631) {
x6632 = x6630;
} else {
x6632 = false;
}
if (x6632) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x6543,x6543,1,x6623,1,1);
assert(false && "");
}
bool x6638 = 512 <= x6623;
int32_t x6639;
if (x6638) {
x6639 = x6623;
} else {
x6639 = 512;
}
int32_t x6643 = x6639 * x6642;
int32_t x6644 = 64 * x6643;
float* x6645 = (float*)myMalloc(x6644 * sizeof(float));;
int32_t x6648;
if (x6624) {
x6648 = 0;
} else {
x6648 = 1;
}
for(int x6649=0; x6649 < 64; x6649++) {
int32_t x6661 = x6545 * x6649;
int32_t x6655 = x6643 * x6649;
for(int x6651=0; x6651 < x6639; x6651++) {
int32_t x6662 = x6544 * x6651;
int32_t x6663 = x6661 + x6662;
int32_t x6668 = x6648 * x6651;
int32_t x6657 = x6642 * x6651;
for(int x6653=0; x6653 < x6641; x6653++) {
int32_t x6664 = x6646 * x6653;
int32_t x6665 = x6663 + x6664;
int32_t x6659 = x6641 * x6653;
for(int x6654=0; x6654 < x6641; x6654++) {
int32_t x6666 = x6647 * x6654;
int32_t x6667 = x6665 + x6666;
float x6669 = x6549[x6667];
float x6670 = x42[x6668];
int32_t x6656 = x6654 + x6655;
int32_t x6658 = x6656 + x6657;
int32_t x6660 = x6658 + x6659;
float x6671 = x6669 - x6670;
x6645[x6660] = x6671;

}

}

}

}
float* x6681 = (float*)myMalloc(512 * sizeof(float));;
for(int x6682=0; x6682 < 512; x6682++) {
float x6683 = x23[x6682];
float x6684 = x6683 + 1.0E-5f;
x6681[x6682] = x6684;

}
float* x6688 = (float*)myMalloc(512 * sizeof(float));;
for(int x6689=0; x6689 < 512; x6689++) {
float x6690 = x6681[x6689];
double x6691 = (double)x6690;
double x6692 = sqrt(x6691);
float x6693 = (float)x6692;
x6688[x6689] = x6693;

}
int32_t x6697 = 0;
int32_t x6698 = 1;
x6698 *= 1;
x6697 += 1;
x6698 *= 1;
x6698 *= 1;
int32_t x6703 = x6697;
bool x6704 = x6703 >= 2;
if (x6704) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6709 = x6703 == 0;
if (x6709) {
int32_t x6710 = x6698;
bool x6711 = x6710 == 512;
if (x6711) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6718 = x6698;
bool x6720 = x6639 == 1;
int32_t x6719 = 512 / x6718;
bool x6721 = x6719 == 1;
bool x6725;
if (x454) {
bool x6722 = x6720 || x6721;
bool x6723 = x6639 == x6719;
bool x6724 = x6722 || x6723;
x6725 = x6724;
} else {
x6725 = false;
}
bool x6729;
if (x6725) {
x6729 = x6728;
} else {
x6729 = false;
}
bool x6730;
if (x6729) {
x6730 = x6728;
} else {
x6730 = false;
}
if (x6730) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6639,x6641,x6641,1,x6719,1,1);
assert(false && "");
}
bool x6736 = x6639 <= x6719;
int32_t x6737;
if (x6736) {
x6737 = x6719;
} else {
x6737 = x6639;
}
int32_t x6741 = x6737 * x6740;
int32_t x6742 = 64 * x6741;
float* x6743 = (float*)myMalloc(x6742 * sizeof(float));;
int32_t x6744;
if (x6720) {
x6744 = 0;
} else {
x6744 = x6642;
}
int32_t x6747;
if (x6721) {
x6747 = 0;
} else {
x6747 = 1;
}
for(int x6748=0; x6748 < 64; x6748++) {
int32_t x6760 = x6643 * x6748;
int32_t x6754 = x6741 * x6748;
for(int x6750=0; x6750 < x6737; x6750++) {
int32_t x6761 = x6744 * x6750;
int32_t x6762 = x6760 + x6761;
int32_t x6767 = x6747 * x6750;
int32_t x6756 = x6740 * x6750;
for(int x6752=0; x6752 < x6739; x6752++) {
int32_t x6763 = x6745 * x6752;
int32_t x6764 = x6762 + x6763;
int32_t x6758 = x6739 * x6752;
for(int x6753=0; x6753 < x6739; x6753++) {
int32_t x6765 = x6746 * x6753;
int32_t x6766 = x6764 + x6765;
float x6768 = x6645[x6766];
float x6769 = x6688[x6767];
int32_t x6755 = x6753 + x6754;
int32_t x6757 = x6755 + x6756;
int32_t x6759 = x6757 + x6758;
float x6770 = x6768 / x6769;
x6743[x6759] = x6770;

}

}

}

}
int32_t x6780 = 0;
int32_t x6781 = 1;
x6781 *= 1;
x6780 += 1;
x6781 *= 1;
x6781 *= 1;
int32_t x6786 = x6780;
bool x6787 = x6786 >= 2;
if (x6787) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6792 = x6786 == 0;
if (x6792) {
int32_t x6793 = x6781;
bool x6794 = x6793 == 512;
if (x6794) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6801 = x6781;
bool x6803 = x6737 == 1;
int32_t x6802 = 512 / x6801;
bool x6804 = x6802 == 1;
bool x6808;
if (x454) {
bool x6805 = x6803 || x6804;
bool x6806 = x6737 == x6802;
bool x6807 = x6805 || x6806;
x6808 = x6807;
} else {
x6808 = false;
}
bool x6812;
if (x6808) {
x6812 = x6811;
} else {
x6812 = false;
}
bool x6813;
if (x6812) {
x6813 = x6811;
} else {
x6813 = false;
}
if (x6813) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6737,x6739,x6739,1,x6802,1,1);
assert(false && "");
}
bool x6819 = x6737 <= x6802;
int32_t x6820;
if (x6819) {
x6820 = x6802;
} else {
x6820 = x6737;
}
int32_t x6824 = x6820 * x6823;
int32_t x6825 = 64 * x6824;
float* x6826 = (float*)myMalloc(x6825 * sizeof(float));;
int32_t x6827;
if (x6803) {
x6827 = 0;
} else {
x6827 = x6740;
}
int32_t x6830;
if (x6804) {
x6830 = 0;
} else {
x6830 = 1;
}
for(int x6831=0; x6831 < 64; x6831++) {
int32_t x6843 = x6741 * x6831;
int32_t x6837 = x6824 * x6831;
for(int x6833=0; x6833 < x6820; x6833++) {
int32_t x6844 = x6827 * x6833;
int32_t x6845 = x6843 + x6844;
int32_t x6850 = x6830 * x6833;
int32_t x6839 = x6823 * x6833;
for(int x6835=0; x6835 < x6822; x6835++) {
int32_t x6846 = x6828 * x6835;
int32_t x6847 = x6845 + x6846;
int32_t x6841 = x6822 * x6835;
for(int x6836=0; x6836 < x6822; x6836++) {
int32_t x6848 = x6829 * x6836;
int32_t x6849 = x6847 + x6848;
float x6851 = x6743[x6849];
float x6852 = x207[x6850];
int32_t x6838 = x6836 + x6837;
int32_t x6840 = x6838 + x6839;
int32_t x6842 = x6840 + x6841;
float x6853 = x6851 * x6852;
x6826[x6842] = x6853;

}

}

}

}
int32_t x6863 = 0;
int32_t x6864 = 1;
x6864 *= 1;
x6863 += 1;
x6864 *= 1;
x6864 *= 1;
int32_t x6869 = x6863;
bool x6870 = x6869 >= 2;
if (x6870) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6875 = x6869 == 0;
if (x6875) {
int32_t x6876 = x6864;
bool x6877 = x6876 == 512;
if (x6877) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6884 = x6864;
bool x6886 = x6820 == 1;
int32_t x6885 = 512 / x6884;
bool x6887 = x6885 == 1;
bool x6891;
if (x454) {
bool x6888 = x6886 || x6887;
bool x6889 = x6820 == x6885;
bool x6890 = x6888 || x6889;
x6891 = x6890;
} else {
x6891 = false;
}
bool x6895;
if (x6891) {
x6895 = x6894;
} else {
x6895 = false;
}
bool x6896;
if (x6895) {
x6896 = x6894;
} else {
x6896 = false;
}
if (x6896) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6820,x6822,x6822,1,x6885,1,1);
assert(false && "");
}
bool x6902 = x6820 <= x6885;
int32_t x6903;
if (x6902) {
x6903 = x6885;
} else {
x6903 = x6820;
}
int32_t x6907 = x6903 * x6906;
int32_t x6908 = 64 * x6907;
float* x6909 = (float*)myMalloc(x6908 * sizeof(float));;
int32_t x6910;
if (x6886) {
x6910 = 0;
} else {
x6910 = x6823;
}
int32_t x6913;
if (x6887) {
x6913 = 0;
} else {
x6913 = 1;
}
for(int x6914=0; x6914 < 64; x6914++) {
int32_t x6926 = x6824 * x6914;
int32_t x6920 = x6907 * x6914;
for(int x6916=0; x6916 < x6903; x6916++) {
int32_t x6927 = x6910 * x6916;
int32_t x6928 = x6926 + x6927;
int32_t x6933 = x6913 * x6916;
int32_t x6922 = x6906 * x6916;
for(int x6918=0; x6918 < x6905; x6918++) {
int32_t x6929 = x6911 * x6918;
int32_t x6930 = x6928 + x6929;
int32_t x6924 = x6905 * x6918;
for(int x6919=0; x6919 < x6905; x6919++) {
int32_t x6931 = x6912 * x6919;
int32_t x6932 = x6930 + x6931;
float x6934 = x6826[x6932];
float x6935 = x119[x6933];
int32_t x6921 = x6919 + x6920;
int32_t x6923 = x6921 + x6922;
int32_t x6925 = x6923 + x6924;
float x6936 = x6934 + x6935;
x6909[x6925] = x6936;

}

}

}

}
bool x6946 = x6499 == 1;
bool x6947 = x6903 == 1;
bool x6948 = x6946 || x6947;
bool x6949 = x6499 == x6903;
bool x6950 = x6948 || x6949;
bool x6956;
if (x6950) {
x6956 = x6955;
} else {
x6956 = false;
}
bool x6957;
if (x6956) {
x6957 = x6955;
} else {
x6957 = false;
}
if (x6957) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6499,x6501,x6501,64,x6903,x6905,x6905);
assert(false && "");
}
bool x6963 = x6499 <= x6903;
int32_t x6964;
if (x6963) {
x6964 = x6903;
} else {
x6964 = x6499;
}
int32_t x6970;
if (x6946) {
x6970 = 0;
} else {
x6970 = x6502;
}
int32_t x6973;
if (x6947) {
x6973 = 0;
} else {
x6973 = x6906;
}
for(int x6976=0; x6976 < 64; x6976++) {
int32_t x6982 = x6503 * x6976;
int32_t x6989 = x6907 * x6976;
for(int x6978=0; x6978 < x6964; x6978++) {
int32_t x6983 = x6970 * x6978;
int32_t x6984 = x6982 + x6983;
int32_t x6990 = x6973 * x6978;
int32_t x6991 = x6989 + x6990;
for(int x6980=0; x6980 < x6966; x6980++) {
int32_t x6985 = x6971 * x6980;
int32_t x6986 = x6984 + x6985;
int32_t x6992 = x6974 * x6980;
int32_t x6993 = x6991 + x6992;
for(int x6981=0; x6981 < x6966; x6981++) {
int32_t x6987 = x6972 * x6981;
int32_t x6988 = x6986 + x6987;
float x6996 = x6505[x6988];
int32_t x6994 = x6975 * x6981;
int32_t x6995 = x6993 + x6994;
float x6997 = x6909[x6995];
float x6998 = x6996 + x6997;
x6505[x6988] = x6998;

}

}

}

}
float* x7008 = (float*)myMalloc(x6504 * sizeof(float));;
for(int x7010=0; x7010 < x6504; x7010++) {
float x7011 = x6505[x7010];
bool x7012 = x7011 < 0.0f;
if (x7012) {
x7008[x7010] = 0.0f;
} else {
float x7015 = x6505[x7010];
x7008[x7010] = x7015;
}

}
float* x7029 = (float*)myMalloc(x7028 * sizeof(float));;
int32_t x7032 = 64 * x6499;
int32_t x7033 = x7032 * x7024;
float* x7034 = (float*)myMalloc(x7033 * sizeof(float));;
int32_t x7030 = x6499 * x7024;
for(int x7035=0; x7035 < 64; x7035++) {
int32_t x7036 = x7035 * x6503;
float* x7037 = x7008+x7036;
int32_t x7038 = x7035 * x7025;
float* x7039 = x7029+x7038;
int32_t x7040 = x7035 * x7030;
float* x7041 = x7034+x7040;
for(int x7042=0; x7042 < x6499; x7042++) {
int32_t x7043 = x7042 / 1;
int32_t x7047 = x7043 * x7023;
int32_t x7048 = x7047 * x7023;
int32_t x7044 = x7042 % 1;
int32_t x7045 = x7044 / 1;
int32_t x7049 = x7045 * x7023;
int32_t x7050 = x7049 * x7023;
int32_t x7051 = x7048 + x7050;
int32_t x7046 = x7044 % 1;
int32_t x7052 = x7046 * x7023;
int32_t x7053 = x7052 * x7023;
int32_t x7054 = x7051 + x7053;
float* x7055 = x7041+x7054;
int32_t x7056 = x7043 * x6501;
int32_t x7057 = x7056 * x6501;
float* x7058 = x7037+x7057;
for(int x7060=0; x7060 < x7023; x7060++) {
int32_t x7062 = x7060 * x7023;
float* x7063 = x7055+x7062;
int32_t x7061 = x7060 + x7045;
int32_t x7064 = x7061 * x6501;
int32_t x7065 = x7064 + x7046;
float* x7066 = x7058+x7065;
memcpy(x7063, x7066, 4 * x7023);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x7024,x6499,1,x256,x6499,x7041,x7024,1,x7039,x7024);

}
int32_t x7075 = 0;
int32_t x7076 = 1;
x7076 *= 1;
x7075 += 1;
x7076 *= 1;
x7076 *= 1;
int32_t x7081 = x7075;
bool x7082 = x7081 >= 2;
if (x7082) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7087 = x7081 == 0;
if (x7087) {
int32_t x7088 = x7076;
bool x7089 = x7088 == 128;
if (x7089) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7096 = x7076;
int32_t x7097 = 128 / x7096;
bool x7098 = x7097 == 1;
bool x7101;
if (x454) {
bool x7099 = 128 == x7097;
bool x7100 = x7098 || x7099;
x7101 = x7100;
} else {
x7101 = false;
}
bool x7105;
if (x7101) {
x7105 = x7104;
} else {
x7105 = false;
}
bool x7106;
if (x7105) {
x7106 = x7104;
} else {
x7106 = false;
}
if (x7106) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x7023,x7023,1,x7097,1,1);
assert(false && "");
}
bool x7112 = 128 <= x7097;
int32_t x7113;
if (x7112) {
x7113 = x7097;
} else {
x7113 = 128;
}
int32_t x7117 = x7113 * x7116;
int32_t x7118 = 64 * x7117;
float* x7119 = (float*)myMalloc(x7118 * sizeof(float));;
int32_t x7122;
if (x7098) {
x7122 = 0;
} else {
x7122 = 1;
}
for(int x7123=0; x7123 < 64; x7123++) {
int32_t x7135 = x7025 * x7123;
int32_t x7129 = x7117 * x7123;
for(int x7125=0; x7125 < x7113; x7125++) {
int32_t x7136 = x7024 * x7125;
int32_t x7137 = x7135 + x7136;
int32_t x7142 = x7122 * x7125;
int32_t x7131 = x7116 * x7125;
for(int x7127=0; x7127 < x7115; x7127++) {
int32_t x7138 = x7120 * x7127;
int32_t x7139 = x7137 + x7138;
int32_t x7133 = x7115 * x7127;
for(int x7128=0; x7128 < x7115; x7128++) {
int32_t x7140 = x7121 * x7128;
int32_t x7141 = x7139 + x7140;
float x7143 = x7029[x7141];
float x7144 = x100[x7142];
int32_t x7130 = x7128 + x7129;
int32_t x7132 = x7130 + x7131;
int32_t x7134 = x7132 + x7133;
float x7145 = x7143 - x7144;
x7119[x7134] = x7145;

}

}

}

}
float* x7155 = (float*)myMalloc(128 * sizeof(float));;
for(int x7156=0; x7156 < 128; x7156++) {
float x7157 = x177[x7156];
float x7158 = x7157 + 1.0E-5f;
x7155[x7156] = x7158;

}
float* x7162 = (float*)myMalloc(128 * sizeof(float));;
for(int x7163=0; x7163 < 128; x7163++) {
float x7164 = x7155[x7163];
double x7165 = (double)x7164;
double x7166 = sqrt(x7165);
float x7167 = (float)x7166;
x7162[x7163] = x7167;

}
int32_t x7171 = 0;
int32_t x7172 = 1;
x7172 *= 1;
x7171 += 1;
x7172 *= 1;
x7172 *= 1;
int32_t x7177 = x7171;
bool x7178 = x7177 >= 2;
if (x7178) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7183 = x7177 == 0;
if (x7183) {
int32_t x7184 = x7172;
bool x7185 = x7184 == 128;
if (x7185) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7192 = x7172;
bool x7194 = x7113 == 1;
int32_t x7193 = 128 / x7192;
bool x7195 = x7193 == 1;
bool x7199;
if (x454) {
bool x7196 = x7194 || x7195;
bool x7197 = x7113 == x7193;
bool x7198 = x7196 || x7197;
x7199 = x7198;
} else {
x7199 = false;
}
bool x7203;
if (x7199) {
x7203 = x7202;
} else {
x7203 = false;
}
bool x7204;
if (x7203) {
x7204 = x7202;
} else {
x7204 = false;
}
if (x7204) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7113,x7115,x7115,1,x7193,1,1);
assert(false && "");
}
bool x7210 = x7113 <= x7193;
int32_t x7211;
if (x7210) {
x7211 = x7193;
} else {
x7211 = x7113;
}
int32_t x7215 = x7211 * x7214;
int32_t x7216 = 64 * x7215;
float* x7217 = (float*)myMalloc(x7216 * sizeof(float));;
int32_t x7218;
if (x7194) {
x7218 = 0;
} else {
x7218 = x7116;
}
int32_t x7221;
if (x7195) {
x7221 = 0;
} else {
x7221 = 1;
}
for(int x7222=0; x7222 < 64; x7222++) {
int32_t x7234 = x7117 * x7222;
int32_t x7228 = x7215 * x7222;
for(int x7224=0; x7224 < x7211; x7224++) {
int32_t x7235 = x7218 * x7224;
int32_t x7236 = x7234 + x7235;
int32_t x7241 = x7221 * x7224;
int32_t x7230 = x7214 * x7224;
for(int x7226=0; x7226 < x7213; x7226++) {
int32_t x7237 = x7219 * x7226;
int32_t x7238 = x7236 + x7237;
int32_t x7232 = x7213 * x7226;
for(int x7227=0; x7227 < x7213; x7227++) {
int32_t x7239 = x7220 * x7227;
int32_t x7240 = x7238 + x7239;
float x7242 = x7119[x7240];
float x7243 = x7162[x7241];
int32_t x7229 = x7227 + x7228;
int32_t x7231 = x7229 + x7230;
int32_t x7233 = x7231 + x7232;
float x7244 = x7242 / x7243;
x7217[x7233] = x7244;

}

}

}

}
int32_t x7254 = 0;
int32_t x7255 = 1;
x7255 *= 1;
x7254 += 1;
x7255 *= 1;
x7255 *= 1;
int32_t x7260 = x7254;
bool x7261 = x7260 >= 2;
if (x7261) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7266 = x7260 == 0;
if (x7266) {
int32_t x7267 = x7255;
bool x7268 = x7267 == 128;
if (x7268) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7275 = x7255;
bool x7277 = x7211 == 1;
int32_t x7276 = 128 / x7275;
bool x7278 = x7276 == 1;
bool x7282;
if (x454) {
bool x7279 = x7277 || x7278;
bool x7280 = x7211 == x7276;
bool x7281 = x7279 || x7280;
x7282 = x7281;
} else {
x7282 = false;
}
bool x7286;
if (x7282) {
x7286 = x7285;
} else {
x7286 = false;
}
bool x7287;
if (x7286) {
x7287 = x7285;
} else {
x7287 = false;
}
if (x7287) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7211,x7213,x7213,1,x7276,1,1);
assert(false && "");
}
bool x7293 = x7211 <= x7276;
int32_t x7294;
if (x7293) {
x7294 = x7276;
} else {
x7294 = x7211;
}
int32_t x7298 = x7294 * x7297;
int32_t x7299 = 64 * x7298;
float* x7300 = (float*)myMalloc(x7299 * sizeof(float));;
int32_t x7301;
if (x7277) {
x7301 = 0;
} else {
x7301 = x7214;
}
int32_t x7304;
if (x7278) {
x7304 = 0;
} else {
x7304 = 1;
}
for(int x7305=0; x7305 < 64; x7305++) {
int32_t x7317 = x7215 * x7305;
int32_t x7311 = x7298 * x7305;
for(int x7307=0; x7307 < x7294; x7307++) {
int32_t x7318 = x7301 * x7307;
int32_t x7319 = x7317 + x7318;
int32_t x7324 = x7304 * x7307;
int32_t x7313 = x7297 * x7307;
for(int x7309=0; x7309 < x7296; x7309++) {
int32_t x7320 = x7302 * x7309;
int32_t x7321 = x7319 + x7320;
int32_t x7315 = x7296 * x7309;
for(int x7310=0; x7310 < x7296; x7310++) {
int32_t x7322 = x7303 * x7310;
int32_t x7323 = x7321 + x7322;
float x7325 = x7217[x7323];
float x7326 = x222[x7324];
int32_t x7312 = x7310 + x7311;
int32_t x7314 = x7312 + x7313;
int32_t x7316 = x7314 + x7315;
float x7327 = x7325 * x7326;
x7300[x7316] = x7327;

}

}

}

}
int32_t x7337 = 0;
int32_t x7338 = 1;
x7338 *= 1;
x7337 += 1;
x7338 *= 1;
x7338 *= 1;
int32_t x7343 = x7337;
bool x7344 = x7343 >= 2;
if (x7344) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7349 = x7343 == 0;
if (x7349) {
int32_t x7350 = x7338;
bool x7351 = x7350 == 128;
if (x7351) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7358 = x7338;
bool x7360 = x7294 == 1;
int32_t x7359 = 128 / x7358;
bool x7361 = x7359 == 1;
bool x7365;
if (x454) {
bool x7362 = x7360 || x7361;
bool x7363 = x7294 == x7359;
bool x7364 = x7362 || x7363;
x7365 = x7364;
} else {
x7365 = false;
}
bool x7369;
if (x7365) {
x7369 = x7368;
} else {
x7369 = false;
}
bool x7370;
if (x7369) {
x7370 = x7368;
} else {
x7370 = false;
}
if (x7370) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7294,x7296,x7296,1,x7359,1,1);
assert(false && "");
}
bool x7376 = x7294 <= x7359;
int32_t x7377;
if (x7376) {
x7377 = x7359;
} else {
x7377 = x7294;
}
int32_t x7381 = x7377 * x7380;
int32_t x7382 = 64 * x7381;
float* x7383 = (float*)myMalloc(x7382 * sizeof(float));;
int32_t x7384;
if (x7360) {
x7384 = 0;
} else {
x7384 = x7297;
}
int32_t x7387;
if (x7361) {
x7387 = 0;
} else {
x7387 = 1;
}
for(int x7388=0; x7388 < 64; x7388++) {
int32_t x7400 = x7298 * x7388;
int32_t x7394 = x7381 * x7388;
for(int x7390=0; x7390 < x7377; x7390++) {
int32_t x7401 = x7384 * x7390;
int32_t x7402 = x7400 + x7401;
int32_t x7407 = x7387 * x7390;
int32_t x7396 = x7380 * x7390;
for(int x7392=0; x7392 < x7379; x7392++) {
int32_t x7403 = x7385 * x7392;
int32_t x7404 = x7402 + x7403;
int32_t x7398 = x7379 * x7392;
for(int x7393=0; x7393 < x7379; x7393++) {
int32_t x7405 = x7386 * x7393;
int32_t x7406 = x7404 + x7405;
float x7408 = x7300[x7406];
float x7409 = x17[x7407];
int32_t x7395 = x7393 + x7394;
int32_t x7397 = x7395 + x7396;
int32_t x7399 = x7397 + x7398;
float x7410 = x7408 + x7409;
x7383[x7399] = x7410;

}

}

}

}
float* x7420 = (float*)myMalloc(x7382 * sizeof(float));;
for(int x7422=0; x7422 < x7382; x7422++) {
float x7423 = x7383[x7422];
bool x7424 = x7423 < 0.0f;
if (x7424) {
x7420[x7422] = 0.0f;
} else {
float x7427 = x7383[x7422];
x7420[x7422] = x7427;
}

}
float* x7442 = (float*)myMalloc(x7441 * sizeof(float));;
int32_t x7443 = 9 * x7377;
int32_t x7446 = 64 * x7443;
int32_t x7447 = x7446 * x7437;
float* x7448 = (float*)myMalloc(x7447 * sizeof(float));;
int32_t x7444 = x7443 * x7437;
int32_t x7456 = x7377 * 3;
int32_t x7457 = x7456 * 3;
for(int x7449=0; x7449 < 64; x7449++) {
int32_t x7450 = x7449 * x7381;
float* x7451 = x7420+x7450;
int32_t x7452 = x7449 * x7438;
float* x7453 = x7442+x7452;
int32_t x7454 = x7449 * x7444;
float* x7455 = x7448+x7454;
for(int x7459=0; x7459 < x7457; x7459++) {
int32_t x7460 = x7459 / 9;
int32_t x7464 = x7460 * 3;
int32_t x7465 = x7464 * 3;
int32_t x7466 = x7465 * x7436;
int32_t x7467 = x7466 * x7436;
int32_t x7461 = x7459 % 9;
int32_t x7462 = x7461 / 3;
int32_t x7468 = x7462 * 3;
int32_t x7469 = x7468 * x7436;
int32_t x7470 = x7469 * x7436;
int32_t x7471 = x7467 + x7470;
int32_t x7463 = x7461 % 3;
int32_t x7472 = x7463 * x7436;
int32_t x7473 = x7472 * x7436;
int32_t x7474 = x7471 + x7473;
float* x7475 = x7455+x7474;
int32_t x7476 = x7460 * x7379;
int32_t x7477 = x7476 * x7379;
float* x7478 = x7451+x7477;
int32_t x7491 = 1 - x7463;
bool x7492 = x7491 > 0;
int32_t x7493;
if (x7492) {
x7493 = x7491;
} else {
x7493 = 0;
}
int32_t x7494 = 3 - x7463;
int32_t x7495 = x7494 - 1;
int32_t x7496 = 1 - x7495;
bool x7497 = x7496 > 0;
int32_t x7498;
if (x7497) {
x7498 = x7496;
} else {
x7498 = 0;
}
int32_t x7499 = x7436 - x7498;
int32_t x7500 = x7499 - x7493;
bool x7501 = x7500 <= 0;
bool x7505 = x7493 > 0;
int32_t x7490 = -1 + x7463;
bool x7518 = x7498 > 0;
for(int x7480=0; x7480 < x7436; x7480++) {
int32_t x7481 = x7480 - 1;
int32_t x7482 = x7481 + x7462;
bool x7483 = x7482 < 0;
bool x7484 = x7482 >= x7379;
bool x7485 = x7483 || x7484;
if (x7485) {
int32_t x7486 = x7480 * x7436;
float* x7487 = x7475+x7486;
memset(x7487, 0, 4 * x7436);;
} else {
if (x7501) {
int32_t x7486 = x7480 * x7436;
float* x7502 = x7475+x7486;
memset(x7502, 0, 4 * x7436);;
} else {
int32_t x7486 = x7480 * x7436;
if (x7505) {
float* x7506 = x7475+x7486;
memset(x7506, 0, 4 * x7493);;
} else {
}
// may have segfault here
int32_t x7511 = x7486 + x7493;
float* x7512 = x7475+x7511;
int32_t x7513 = x7482 * x7379;
int32_t x7514 = x7513 + x7490;
int32_t x7515 = x7514 + x7493;
float* x7516 = x7478+x7515;
memcpy(x7512, x7516, 4 * x7500);;
if (x7518) {
int32_t x7519 = x7486 + x7436;
int32_t x7520 = x7519 - x7498;
float* x7521 = x7475+x7520;
memset(x7521, 0, 4 * x7498);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x7437,x7443,1,x235,x7443,x7455,x7437,1,x7453,x7437);

}
int32_t x7536 = 0;
int32_t x7537 = 1;
x7537 *= 1;
x7536 += 1;
x7537 *= 1;
x7537 *= 1;
int32_t x7542 = x7536;
bool x7543 = x7542 >= 2;
if (x7543) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7548 = x7542 == 0;
if (x7548) {
int32_t x7549 = x7537;
bool x7550 = x7549 == 128;
if (x7550) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7557 = x7537;
int32_t x7558 = 128 / x7557;
bool x7559 = x7558 == 1;
bool x7562;
if (x454) {
bool x7560 = 128 == x7558;
bool x7561 = x7559 || x7560;
x7562 = x7561;
} else {
x7562 = false;
}
bool x7566;
if (x7562) {
x7566 = x7565;
} else {
x7566 = false;
}
bool x7567;
if (x7566) {
x7567 = x7565;
} else {
x7567 = false;
}
if (x7567) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x7436,x7436,1,x7558,1,1);
assert(false && "");
}
bool x7573 = 128 <= x7558;
int32_t x7574;
if (x7573) {
x7574 = x7558;
} else {
x7574 = 128;
}
int32_t x7578 = x7574 * x7577;
int32_t x7579 = 64 * x7578;
float* x7580 = (float*)myMalloc(x7579 * sizeof(float));;
int32_t x7583;
if (x7559) {
x7583 = 0;
} else {
x7583 = 1;
}
for(int x7584=0; x7584 < 64; x7584++) {
int32_t x7596 = x7438 * x7584;
int32_t x7590 = x7578 * x7584;
for(int x7586=0; x7586 < x7574; x7586++) {
int32_t x7597 = x7437 * x7586;
int32_t x7598 = x7596 + x7597;
int32_t x7603 = x7583 * x7586;
int32_t x7592 = x7577 * x7586;
for(int x7588=0; x7588 < x7576; x7588++) {
int32_t x7599 = x7581 * x7588;
int32_t x7600 = x7598 + x7599;
int32_t x7594 = x7576 * x7588;
for(int x7589=0; x7589 < x7576; x7589++) {
int32_t x7601 = x7582 * x7589;
int32_t x7602 = x7600 + x7601;
float x7604 = x7442[x7602];
float x7605 = x35[x7603];
int32_t x7591 = x7589 + x7590;
int32_t x7593 = x7591 + x7592;
int32_t x7595 = x7593 + x7594;
float x7606 = x7604 - x7605;
x7580[x7595] = x7606;

}

}

}

}
float* x7616 = (float*)myMalloc(128 * sizeof(float));;
for(int x7617=0; x7617 < 128; x7617++) {
float x7618 = x225[x7617];
float x7619 = x7618 + 1.0E-5f;
x7616[x7617] = x7619;

}
float* x7623 = (float*)myMalloc(128 * sizeof(float));;
for(int x7624=0; x7624 < 128; x7624++) {
float x7625 = x7616[x7624];
double x7626 = (double)x7625;
double x7627 = sqrt(x7626);
float x7628 = (float)x7627;
x7623[x7624] = x7628;

}
int32_t x7632 = 0;
int32_t x7633 = 1;
x7633 *= 1;
x7632 += 1;
x7633 *= 1;
x7633 *= 1;
int32_t x7638 = x7632;
bool x7639 = x7638 >= 2;
if (x7639) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7644 = x7638 == 0;
if (x7644) {
int32_t x7645 = x7633;
bool x7646 = x7645 == 128;
if (x7646) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7653 = x7633;
bool x7655 = x7574 == 1;
int32_t x7654 = 128 / x7653;
bool x7656 = x7654 == 1;
bool x7660;
if (x454) {
bool x7657 = x7655 || x7656;
bool x7658 = x7574 == x7654;
bool x7659 = x7657 || x7658;
x7660 = x7659;
} else {
x7660 = false;
}
bool x7664;
if (x7660) {
x7664 = x7663;
} else {
x7664 = false;
}
bool x7665;
if (x7664) {
x7665 = x7663;
} else {
x7665 = false;
}
if (x7665) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7574,x7576,x7576,1,x7654,1,1);
assert(false && "");
}
bool x7671 = x7574 <= x7654;
int32_t x7672;
if (x7671) {
x7672 = x7654;
} else {
x7672 = x7574;
}
int32_t x7676 = x7672 * x7675;
int32_t x7677 = 64 * x7676;
float* x7678 = (float*)myMalloc(x7677 * sizeof(float));;
int32_t x7679;
if (x7655) {
x7679 = 0;
} else {
x7679 = x7577;
}
int32_t x7682;
if (x7656) {
x7682 = 0;
} else {
x7682 = 1;
}
for(int x7683=0; x7683 < 64; x7683++) {
int32_t x7695 = x7578 * x7683;
int32_t x7689 = x7676 * x7683;
for(int x7685=0; x7685 < x7672; x7685++) {
int32_t x7696 = x7679 * x7685;
int32_t x7697 = x7695 + x7696;
int32_t x7702 = x7682 * x7685;
int32_t x7691 = x7675 * x7685;
for(int x7687=0; x7687 < x7674; x7687++) {
int32_t x7698 = x7680 * x7687;
int32_t x7699 = x7697 + x7698;
int32_t x7693 = x7674 * x7687;
for(int x7688=0; x7688 < x7674; x7688++) {
int32_t x7700 = x7681 * x7688;
int32_t x7701 = x7699 + x7700;
float x7703 = x7580[x7701];
float x7704 = x7623[x7702];
int32_t x7690 = x7688 + x7689;
int32_t x7692 = x7690 + x7691;
int32_t x7694 = x7692 + x7693;
float x7705 = x7703 / x7704;
x7678[x7694] = x7705;

}

}

}

}
int32_t x7715 = 0;
int32_t x7716 = 1;
x7716 *= 1;
x7715 += 1;
x7716 *= 1;
x7716 *= 1;
int32_t x7721 = x7715;
bool x7722 = x7721 >= 2;
if (x7722) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7727 = x7721 == 0;
if (x7727) {
int32_t x7728 = x7716;
bool x7729 = x7728 == 128;
if (x7729) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7736 = x7716;
bool x7738 = x7672 == 1;
int32_t x7737 = 128 / x7736;
bool x7739 = x7737 == 1;
bool x7743;
if (x454) {
bool x7740 = x7738 || x7739;
bool x7741 = x7672 == x7737;
bool x7742 = x7740 || x7741;
x7743 = x7742;
} else {
x7743 = false;
}
bool x7747;
if (x7743) {
x7747 = x7746;
} else {
x7747 = false;
}
bool x7748;
if (x7747) {
x7748 = x7746;
} else {
x7748 = false;
}
if (x7748) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7672,x7674,x7674,1,x7737,1,1);
assert(false && "");
}
bool x7754 = x7672 <= x7737;
int32_t x7755;
if (x7754) {
x7755 = x7737;
} else {
x7755 = x7672;
}
int32_t x7759 = x7755 * x7758;
int32_t x7760 = 64 * x7759;
float* x7761 = (float*)myMalloc(x7760 * sizeof(float));;
int32_t x7762;
if (x7738) {
x7762 = 0;
} else {
x7762 = x7675;
}
int32_t x7765;
if (x7739) {
x7765 = 0;
} else {
x7765 = 1;
}
for(int x7766=0; x7766 < 64; x7766++) {
int32_t x7778 = x7676 * x7766;
int32_t x7772 = x7759 * x7766;
for(int x7768=0; x7768 < x7755; x7768++) {
int32_t x7779 = x7762 * x7768;
int32_t x7780 = x7778 + x7779;
int32_t x7785 = x7765 * x7768;
int32_t x7774 = x7758 * x7768;
for(int x7770=0; x7770 < x7757; x7770++) {
int32_t x7781 = x7763 * x7770;
int32_t x7782 = x7780 + x7781;
int32_t x7776 = x7757 * x7770;
for(int x7771=0; x7771 < x7757; x7771++) {
int32_t x7783 = x7764 * x7771;
int32_t x7784 = x7782 + x7783;
float x7786 = x7678[x7784];
float x7787 = x8[x7785];
int32_t x7773 = x7771 + x7772;
int32_t x7775 = x7773 + x7774;
int32_t x7777 = x7775 + x7776;
float x7788 = x7786 * x7787;
x7761[x7777] = x7788;

}

}

}

}
int32_t x7798 = 0;
int32_t x7799 = 1;
x7799 *= 1;
x7798 += 1;
x7799 *= 1;
x7799 *= 1;
int32_t x7804 = x7798;
bool x7805 = x7804 >= 2;
if (x7805) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7810 = x7804 == 0;
if (x7810) {
int32_t x7811 = x7799;
bool x7812 = x7811 == 128;
if (x7812) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7819 = x7799;
bool x7821 = x7755 == 1;
int32_t x7820 = 128 / x7819;
bool x7822 = x7820 == 1;
bool x7826;
if (x454) {
bool x7823 = x7821 || x7822;
bool x7824 = x7755 == x7820;
bool x7825 = x7823 || x7824;
x7826 = x7825;
} else {
x7826 = false;
}
bool x7830;
if (x7826) {
x7830 = x7829;
} else {
x7830 = false;
}
bool x7831;
if (x7830) {
x7831 = x7829;
} else {
x7831 = false;
}
if (x7831) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7755,x7757,x7757,1,x7820,1,1);
assert(false && "");
}
bool x7837 = x7755 <= x7820;
int32_t x7838;
if (x7837) {
x7838 = x7820;
} else {
x7838 = x7755;
}
int32_t x7842 = x7838 * x7841;
int32_t x7843 = 64 * x7842;
float* x7844 = (float*)myMalloc(x7843 * sizeof(float));;
int32_t x7845;
if (x7821) {
x7845 = 0;
} else {
x7845 = x7758;
}
int32_t x7848;
if (x7822) {
x7848 = 0;
} else {
x7848 = 1;
}
for(int x7849=0; x7849 < 64; x7849++) {
int32_t x7861 = x7759 * x7849;
int32_t x7855 = x7842 * x7849;
for(int x7851=0; x7851 < x7838; x7851++) {
int32_t x7862 = x7845 * x7851;
int32_t x7863 = x7861 + x7862;
int32_t x7868 = x7848 * x7851;
int32_t x7857 = x7841 * x7851;
for(int x7853=0; x7853 < x7840; x7853++) {
int32_t x7864 = x7846 * x7853;
int32_t x7865 = x7863 + x7864;
int32_t x7859 = x7840 * x7853;
for(int x7854=0; x7854 < x7840; x7854++) {
int32_t x7866 = x7847 * x7854;
int32_t x7867 = x7865 + x7866;
float x7869 = x7761[x7867];
float x7870 = x95[x7868];
int32_t x7856 = x7854 + x7855;
int32_t x7858 = x7856 + x7857;
int32_t x7860 = x7858 + x7859;
float x7871 = x7869 + x7870;
x7844[x7860] = x7871;

}

}

}

}
float* x7881 = (float*)myMalloc(x7843 * sizeof(float));;
for(int x7883=0; x7883 < x7843; x7883++) {
float x7884 = x7844[x7883];
bool x7885 = x7884 < 0.0f;
if (x7885) {
x7881[x7883] = 0.0f;
} else {
float x7888 = x7844[x7883];
x7881[x7883] = x7888;
}

}
float* x7902 = (float*)myMalloc(x7901 * sizeof(float));;
int32_t x7905 = 64 * x7838;
int32_t x7906 = x7905 * x7897;
float* x7907 = (float*)myMalloc(x7906 * sizeof(float));;
int32_t x7903 = x7838 * x7897;
for(int x7908=0; x7908 < 64; x7908++) {
int32_t x7909 = x7908 * x7842;
float* x7910 = x7881+x7909;
int32_t x7911 = x7908 * x7898;
float* x7912 = x7902+x7911;
int32_t x7913 = x7908 * x7903;
float* x7914 = x7907+x7913;
for(int x7915=0; x7915 < x7838; x7915++) {
int32_t x7916 = x7915 / 1;
int32_t x7920 = x7916 * x7896;
int32_t x7921 = x7920 * x7896;
int32_t x7917 = x7915 % 1;
int32_t x7918 = x7917 / 1;
int32_t x7922 = x7918 * x7896;
int32_t x7923 = x7922 * x7896;
int32_t x7924 = x7921 + x7923;
int32_t x7919 = x7917 % 1;
int32_t x7925 = x7919 * x7896;
int32_t x7926 = x7925 * x7896;
int32_t x7927 = x7924 + x7926;
float* x7928 = x7914+x7927;
int32_t x7929 = x7916 * x7840;
int32_t x7930 = x7929 * x7840;
float* x7931 = x7910+x7930;
for(int x7933=0; x7933 < x7896; x7933++) {
int32_t x7935 = x7933 * x7896;
float* x7936 = x7928+x7935;
int32_t x7934 = x7933 + x7918;
int32_t x7937 = x7934 * x7840;
int32_t x7938 = x7937 + x7919;
float* x7939 = x7931+x7938;
memcpy(x7936, x7939, 4 * x7896);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x7897,x7838,1,x111,x7838,x7914,x7897,1,x7912,x7897);

}
int32_t x7948 = 0;
int32_t x7949 = 1;
x7949 *= 1;
x7948 += 1;
x7949 *= 1;
x7949 *= 1;
int32_t x7954 = x7948;
bool x7955 = x7954 >= 2;
if (x7955) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7960 = x7954 == 0;
if (x7960) {
int32_t x7961 = x7949;
bool x7962 = x7961 == 512;
if (x7962) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7969 = x7949;
int32_t x7970 = 512 / x7969;
bool x7971 = x7970 == 1;
bool x7974;
if (x454) {
bool x7972 = 512 == x7970;
bool x7973 = x7971 || x7972;
x7974 = x7973;
} else {
x7974 = false;
}
bool x7978;
if (x7974) {
x7978 = x7977;
} else {
x7978 = false;
}
bool x7979;
if (x7978) {
x7979 = x7977;
} else {
x7979 = false;
}
if (x7979) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x7896,x7896,1,x7970,1,1);
assert(false && "");
}
bool x7985 = 512 <= x7970;
int32_t x7986;
if (x7985) {
x7986 = x7970;
} else {
x7986 = 512;
}
int32_t x7990 = x7986 * x7989;
int32_t x7991 = 64 * x7990;
float* x7992 = (float*)myMalloc(x7991 * sizeof(float));;
int32_t x7995;
if (x7971) {
x7995 = 0;
} else {
x7995 = 1;
}
for(int x7996=0; x7996 < 64; x7996++) {
int32_t x8008 = x7898 * x7996;
int32_t x8002 = x7990 * x7996;
for(int x7998=0; x7998 < x7986; x7998++) {
int32_t x8009 = x7897 * x7998;
int32_t x8010 = x8008 + x8009;
int32_t x8015 = x7995 * x7998;
int32_t x8004 = x7989 * x7998;
for(int x8000=0; x8000 < x7988; x8000++) {
int32_t x8011 = x7993 * x8000;
int32_t x8012 = x8010 + x8011;
int32_t x8006 = x7988 * x8000;
for(int x8001=0; x8001 < x7988; x8001++) {
int32_t x8013 = x7994 * x8001;
int32_t x8014 = x8012 + x8013;
float x8016 = x7902[x8014];
float x8017 = x147[x8015];
int32_t x8003 = x8001 + x8002;
int32_t x8005 = x8003 + x8004;
int32_t x8007 = x8005 + x8006;
float x8018 = x8016 - x8017;
x7992[x8007] = x8018;

}

}

}

}
float* x8028 = (float*)myMalloc(512 * sizeof(float));;
for(int x8029=0; x8029 < 512; x8029++) {
float x8030 = x88[x8029];
float x8031 = x8030 + 1.0E-5f;
x8028[x8029] = x8031;

}
float* x8035 = (float*)myMalloc(512 * sizeof(float));;
for(int x8036=0; x8036 < 512; x8036++) {
float x8037 = x8028[x8036];
double x8038 = (double)x8037;
double x8039 = sqrt(x8038);
float x8040 = (float)x8039;
x8035[x8036] = x8040;

}
int32_t x8044 = 0;
int32_t x8045 = 1;
x8045 *= 1;
x8044 += 1;
x8045 *= 1;
x8045 *= 1;
int32_t x8050 = x8044;
bool x8051 = x8050 >= 2;
if (x8051) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8056 = x8050 == 0;
if (x8056) {
int32_t x8057 = x8045;
bool x8058 = x8057 == 512;
if (x8058) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8065 = x8045;
bool x8067 = x7986 == 1;
int32_t x8066 = 512 / x8065;
bool x8068 = x8066 == 1;
bool x8072;
if (x454) {
bool x8069 = x8067 || x8068;
bool x8070 = x7986 == x8066;
bool x8071 = x8069 || x8070;
x8072 = x8071;
} else {
x8072 = false;
}
bool x8076;
if (x8072) {
x8076 = x8075;
} else {
x8076 = false;
}
bool x8077;
if (x8076) {
x8077 = x8075;
} else {
x8077 = false;
}
if (x8077) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7986,x7988,x7988,1,x8066,1,1);
assert(false && "");
}
bool x8083 = x7986 <= x8066;
int32_t x8084;
if (x8083) {
x8084 = x8066;
} else {
x8084 = x7986;
}
int32_t x8088 = x8084 * x8087;
int32_t x8089 = 64 * x8088;
float* x8090 = (float*)myMalloc(x8089 * sizeof(float));;
int32_t x8091;
if (x8067) {
x8091 = 0;
} else {
x8091 = x7989;
}
int32_t x8094;
if (x8068) {
x8094 = 0;
} else {
x8094 = 1;
}
for(int x8095=0; x8095 < 64; x8095++) {
int32_t x8107 = x7990 * x8095;
int32_t x8101 = x8088 * x8095;
for(int x8097=0; x8097 < x8084; x8097++) {
int32_t x8108 = x8091 * x8097;
int32_t x8109 = x8107 + x8108;
int32_t x8114 = x8094 * x8097;
int32_t x8103 = x8087 * x8097;
for(int x8099=0; x8099 < x8086; x8099++) {
int32_t x8110 = x8092 * x8099;
int32_t x8111 = x8109 + x8110;
int32_t x8105 = x8086 * x8099;
for(int x8100=0; x8100 < x8086; x8100++) {
int32_t x8112 = x8093 * x8100;
int32_t x8113 = x8111 + x8112;
float x8115 = x7992[x8113];
float x8116 = x8035[x8114];
int32_t x8102 = x8100 + x8101;
int32_t x8104 = x8102 + x8103;
int32_t x8106 = x8104 + x8105;
float x8117 = x8115 / x8116;
x8090[x8106] = x8117;

}

}

}

}
int32_t x8127 = 0;
int32_t x8128 = 1;
x8128 *= 1;
x8127 += 1;
x8128 *= 1;
x8128 *= 1;
int32_t x8133 = x8127;
bool x8134 = x8133 >= 2;
if (x8134) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8139 = x8133 == 0;
if (x8139) {
int32_t x8140 = x8128;
bool x8141 = x8140 == 512;
if (x8141) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8148 = x8128;
bool x8150 = x8084 == 1;
int32_t x8149 = 512 / x8148;
bool x8151 = x8149 == 1;
bool x8155;
if (x454) {
bool x8152 = x8150 || x8151;
bool x8153 = x8084 == x8149;
bool x8154 = x8152 || x8153;
x8155 = x8154;
} else {
x8155 = false;
}
bool x8159;
if (x8155) {
x8159 = x8158;
} else {
x8159 = false;
}
bool x8160;
if (x8159) {
x8160 = x8158;
} else {
x8160 = false;
}
if (x8160) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8084,x8086,x8086,1,x8149,1,1);
assert(false && "");
}
bool x8166 = x8084 <= x8149;
int32_t x8167;
if (x8166) {
x8167 = x8149;
} else {
x8167 = x8084;
}
int32_t x8171 = x8167 * x8170;
int32_t x8172 = 64 * x8171;
float* x8173 = (float*)myMalloc(x8172 * sizeof(float));;
int32_t x8174;
if (x8150) {
x8174 = 0;
} else {
x8174 = x8087;
}
int32_t x8177;
if (x8151) {
x8177 = 0;
} else {
x8177 = 1;
}
for(int x8178=0; x8178 < 64; x8178++) {
int32_t x8190 = x8088 * x8178;
int32_t x8184 = x8171 * x8178;
for(int x8180=0; x8180 < x8167; x8180++) {
int32_t x8191 = x8174 * x8180;
int32_t x8192 = x8190 + x8191;
int32_t x8197 = x8177 * x8180;
int32_t x8186 = x8170 * x8180;
for(int x8182=0; x8182 < x8169; x8182++) {
int32_t x8193 = x8175 * x8182;
int32_t x8194 = x8192 + x8193;
int32_t x8188 = x8169 * x8182;
for(int x8183=0; x8183 < x8169; x8183++) {
int32_t x8195 = x8176 * x8183;
int32_t x8196 = x8194 + x8195;
float x8198 = x8090[x8196];
float x8199 = x52[x8197];
int32_t x8185 = x8183 + x8184;
int32_t x8187 = x8185 + x8186;
int32_t x8189 = x8187 + x8188;
float x8200 = x8198 * x8199;
x8173[x8189] = x8200;

}

}

}

}
int32_t x8210 = 0;
int32_t x8211 = 1;
x8211 *= 1;
x8210 += 1;
x8211 *= 1;
x8211 *= 1;
int32_t x8216 = x8210;
bool x8217 = x8216 >= 2;
if (x8217) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8222 = x8216 == 0;
if (x8222) {
int32_t x8223 = x8211;
bool x8224 = x8223 == 512;
if (x8224) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8231 = x8211;
bool x8233 = x8167 == 1;
int32_t x8232 = 512 / x8231;
bool x8234 = x8232 == 1;
bool x8238;
if (x454) {
bool x8235 = x8233 || x8234;
bool x8236 = x8167 == x8232;
bool x8237 = x8235 || x8236;
x8238 = x8237;
} else {
x8238 = false;
}
bool x8242;
if (x8238) {
x8242 = x8241;
} else {
x8242 = false;
}
bool x8243;
if (x8242) {
x8243 = x8241;
} else {
x8243 = false;
}
if (x8243) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8167,x8169,x8169,1,x8232,1,1);
assert(false && "");
}
bool x8249 = x8167 <= x8232;
int32_t x8250;
if (x8249) {
x8250 = x8232;
} else {
x8250 = x8167;
}
int32_t x8254 = x8250 * x8253;
int32_t x8255 = 64 * x8254;
float* x8256 = (float*)myMalloc(x8255 * sizeof(float));;
int32_t x8257;
if (x8233) {
x8257 = 0;
} else {
x8257 = x8170;
}
int32_t x8260;
if (x8234) {
x8260 = 0;
} else {
x8260 = 1;
}
for(int x8261=0; x8261 < 64; x8261++) {
int32_t x8273 = x8171 * x8261;
int32_t x8267 = x8254 * x8261;
for(int x8263=0; x8263 < x8250; x8263++) {
int32_t x8274 = x8257 * x8263;
int32_t x8275 = x8273 + x8274;
int32_t x8280 = x8260 * x8263;
int32_t x8269 = x8253 * x8263;
for(int x8265=0; x8265 < x8252; x8265++) {
int32_t x8276 = x8258 * x8265;
int32_t x8277 = x8275 + x8276;
int32_t x8271 = x8252 * x8265;
for(int x8266=0; x8266 < x8252; x8266++) {
int32_t x8278 = x8259 * x8266;
int32_t x8279 = x8277 + x8278;
float x8281 = x8173[x8279];
float x8282 = x246[x8280];
int32_t x8268 = x8266 + x8267;
int32_t x8270 = x8268 + x8269;
int32_t x8272 = x8270 + x8271;
float x8283 = x8281 + x8282;
x8256[x8272] = x8283;

}

}

}

}
bool x8293 = x8250 == 1;
bool x8294 = x8293 || x6946;
bool x8295 = x8250 == x6499;
bool x8296 = x8294 || x8295;
bool x8301;
if (x8296) {
x8301 = x8300;
} else {
x8301 = false;
}
bool x8302;
if (x8301) {
x8302 = x8300;
} else {
x8302 = false;
}
if (x8302) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8250,x8252,x8252,64,x6499,x6501,x6501);
assert(false && "");
}
bool x8308 = x8250 <= x6499;
int32_t x8309;
if (x8308) {
x8309 = x6499;
} else {
x8309 = x8250;
}
int32_t x8315;
if (x8293) {
x8315 = 0;
} else {
x8315 = x8253;
}
for(int x8318=0; x8318 < 64; x8318++) {
int32_t x8324 = x8254 * x8318;
int32_t x8331 = x6503 * x8318;
for(int x8320=0; x8320 < x8309; x8320++) {
int32_t x8325 = x8315 * x8320;
int32_t x8326 = x8324 + x8325;
int32_t x8332 = x6970 * x8320;
int32_t x8333 = x8331 + x8332;
for(int x8322=0; x8322 < x8311; x8322++) {
int32_t x8327 = x8316 * x8322;
int32_t x8328 = x8326 + x8327;
int32_t x8334 = x6971 * x8322;
int32_t x8335 = x8333 + x8334;
for(int x8323=0; x8323 < x8311; x8323++) {
int32_t x8329 = x8317 * x8323;
int32_t x8330 = x8328 + x8329;
float x8338 = x8256[x8330];
int32_t x8336 = x6972 * x8323;
int32_t x8337 = x8335 + x8336;
float x8339 = x7008[x8337];
float x8340 = x8338 + x8339;
x8256[x8330] = x8340;

}

}

}

}
float* x8350 = (float*)myMalloc(x8255 * sizeof(float));;
for(int x8352=0; x8352 < x8255; x8352++) {
float x8353 = x8256[x8352];
bool x8354 = x8353 < 0.0f;
if (x8354) {
x8350[x8352] = 0.0f;
} else {
float x8357 = x8256[x8352];
x8350[x8352] = x8357;
}

}
float* x8371 = (float*)myMalloc(x8370 * sizeof(float));;
int32_t x8374 = 64 * x8250;
int32_t x8375 = x8374 * x8366;
float* x8376 = (float*)myMalloc(x8375 * sizeof(float));;
int32_t x8372 = x8250 * x8366;
for(int x8377=0; x8377 < 64; x8377++) {
int32_t x8378 = x8377 * x8254;
float* x8379 = x8350+x8378;
int32_t x8380 = x8377 * x8367;
float* x8381 = x8371+x8380;
int32_t x8382 = x8377 * x8372;
float* x8383 = x8376+x8382;
for(int x8384=0; x8384 < x8250; x8384++) {
int32_t x8385 = x8384 / 1;
int32_t x8389 = x8385 * x8365;
int32_t x8390 = x8389 * x8365;
int32_t x8386 = x8384 % 1;
int32_t x8387 = x8386 / 1;
int32_t x8391 = x8387 * x8365;
int32_t x8392 = x8391 * x8365;
int32_t x8393 = x8390 + x8392;
int32_t x8388 = x8386 % 1;
int32_t x8394 = x8388 * x8365;
int32_t x8395 = x8394 * x8365;
int32_t x8396 = x8393 + x8395;
float* x8397 = x8383+x8396;
int32_t x8398 = x8385 * x8252;
int32_t x8399 = x8398 * x8252;
float* x8400 = x8379+x8399;
for(int x8402=0; x8402 < x8365; x8402++) {
int32_t x8404 = x8402 * x8365;
float* x8405 = x8397+x8404;
int32_t x8403 = x8402 + x8387;
int32_t x8406 = x8403 * x8252;
int32_t x8407 = x8406 + x8388;
float* x8408 = x8400+x8407;
memcpy(x8405, x8408, 4 * x8365);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x8366,x8250,1,x196,x8250,x8383,x8366,1,x8381,x8366);

}
int32_t x8417 = 0;
int32_t x8418 = 1;
x8418 *= 1;
x8417 += 1;
x8418 *= 1;
x8418 *= 1;
int32_t x8423 = x8417;
bool x8424 = x8423 >= 2;
if (x8424) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8429 = x8423 == 0;
if (x8429) {
int32_t x8430 = x8418;
bool x8431 = x8430 == 128;
if (x8431) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8438 = x8418;
int32_t x8439 = 128 / x8438;
bool x8440 = x8439 == 1;
bool x8443;
if (x454) {
bool x8441 = 128 == x8439;
bool x8442 = x8440 || x8441;
x8443 = x8442;
} else {
x8443 = false;
}
bool x8447;
if (x8443) {
x8447 = x8446;
} else {
x8447 = false;
}
bool x8448;
if (x8447) {
x8448 = x8446;
} else {
x8448 = false;
}
if (x8448) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x8365,x8365,1,x8439,1,1);
assert(false && "");
}
bool x8454 = 128 <= x8439;
int32_t x8455;
if (x8454) {
x8455 = x8439;
} else {
x8455 = 128;
}
int32_t x8459 = x8455 * x8458;
int32_t x8460 = 64 * x8459;
float* x8461 = (float*)myMalloc(x8460 * sizeof(float));;
int32_t x8464;
if (x8440) {
x8464 = 0;
} else {
x8464 = 1;
}
for(int x8465=0; x8465 < 64; x8465++) {
int32_t x8477 = x8367 * x8465;
int32_t x8471 = x8459 * x8465;
for(int x8467=0; x8467 < x8455; x8467++) {
int32_t x8478 = x8366 * x8467;
int32_t x8479 = x8477 + x8478;
int32_t x8484 = x8464 * x8467;
int32_t x8473 = x8458 * x8467;
for(int x8469=0; x8469 < x8457; x8469++) {
int32_t x8480 = x8462 * x8469;
int32_t x8481 = x8479 + x8480;
int32_t x8475 = x8457 * x8469;
for(int x8470=0; x8470 < x8457; x8470++) {
int32_t x8482 = x8463 * x8470;
int32_t x8483 = x8481 + x8482;
float x8485 = x8371[x8483];
float x8486 = x112[x8484];
int32_t x8472 = x8470 + x8471;
int32_t x8474 = x8472 + x8473;
int32_t x8476 = x8474 + x8475;
float x8487 = x8485 - x8486;
x8461[x8476] = x8487;

}

}

}

}
float* x8497 = (float*)myMalloc(128 * sizeof(float));;
for(int x8498=0; x8498 < 128; x8498++) {
float x8499 = x9[x8498];
float x8500 = x8499 + 1.0E-5f;
x8497[x8498] = x8500;

}
float* x8504 = (float*)myMalloc(128 * sizeof(float));;
for(int x8505=0; x8505 < 128; x8505++) {
float x8506 = x8497[x8505];
double x8507 = (double)x8506;
double x8508 = sqrt(x8507);
float x8509 = (float)x8508;
x8504[x8505] = x8509;

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
bool x8527 = x8526 == 128;
if (x8527) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8534 = x8514;
bool x8536 = x8455 == 1;
int32_t x8535 = 128 / x8534;
bool x8537 = x8535 == 1;
bool x8541;
if (x454) {
bool x8538 = x8536 || x8537;
bool x8539 = x8455 == x8535;
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
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8455,x8457,x8457,1,x8535,1,1);
assert(false && "");
}
bool x8552 = x8455 <= x8535;
int32_t x8553;
if (x8552) {
x8553 = x8535;
} else {
x8553 = x8455;
}
int32_t x8557 = x8553 * x8556;
int32_t x8558 = 64 * x8557;
float* x8559 = (float*)myMalloc(x8558 * sizeof(float));;
int32_t x8560;
if (x8536) {
x8560 = 0;
} else {
x8560 = x8458;
}
int32_t x8563;
if (x8537) {
x8563 = 0;
} else {
x8563 = 1;
}
for(int x8564=0; x8564 < 64; x8564++) {
int32_t x8576 = x8459 * x8564;
int32_t x8570 = x8557 * x8564;
for(int x8566=0; x8566 < x8553; x8566++) {
int32_t x8577 = x8560 * x8566;
int32_t x8578 = x8576 + x8577;
int32_t x8583 = x8563 * x8566;
int32_t x8572 = x8556 * x8566;
for(int x8568=0; x8568 < x8555; x8568++) {
int32_t x8579 = x8561 * x8568;
int32_t x8580 = x8578 + x8579;
int32_t x8574 = x8555 * x8568;
for(int x8569=0; x8569 < x8555; x8569++) {
int32_t x8581 = x8562 * x8569;
int32_t x8582 = x8580 + x8581;
float x8584 = x8461[x8582];
float x8585 = x8504[x8583];
int32_t x8571 = x8569 + x8570;
int32_t x8573 = x8571 + x8572;
int32_t x8575 = x8573 + x8574;
float x8586 = x8584 / x8585;
x8559[x8575] = x8586;

}

}

}

}
int32_t x8596 = 0;
int32_t x8597 = 1;
x8597 *= 1;
x8596 += 1;
x8597 *= 1;
x8597 *= 1;
int32_t x8602 = x8596;
bool x8603 = x8602 >= 2;
if (x8603) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8608 = x8602 == 0;
if (x8608) {
int32_t x8609 = x8597;
bool x8610 = x8609 == 128;
if (x8610) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8617 = x8597;
bool x8619 = x8553 == 1;
int32_t x8618 = 128 / x8617;
bool x8620 = x8618 == 1;
bool x8624;
if (x454) {
bool x8621 = x8619 || x8620;
bool x8622 = x8553 == x8618;
bool x8623 = x8621 || x8622;
x8624 = x8623;
} else {
x8624 = false;
}
bool x8628;
if (x8624) {
x8628 = x8627;
} else {
x8628 = false;
}
bool x8629;
if (x8628) {
x8629 = x8627;
} else {
x8629 = false;
}
if (x8629) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8553,x8555,x8555,1,x8618,1,1);
assert(false && "");
}
bool x8635 = x8553 <= x8618;
int32_t x8636;
if (x8635) {
x8636 = x8618;
} else {
x8636 = x8553;
}
int32_t x8640 = x8636 * x8639;
int32_t x8641 = 64 * x8640;
float* x8642 = (float*)myMalloc(x8641 * sizeof(float));;
int32_t x8643;
if (x8619) {
x8643 = 0;
} else {
x8643 = x8556;
}
int32_t x8646;
if (x8620) {
x8646 = 0;
} else {
x8646 = 1;
}
for(int x8647=0; x8647 < 64; x8647++) {
int32_t x8659 = x8557 * x8647;
int32_t x8653 = x8640 * x8647;
for(int x8649=0; x8649 < x8636; x8649++) {
int32_t x8660 = x8643 * x8649;
int32_t x8661 = x8659 + x8660;
int32_t x8666 = x8646 * x8649;
int32_t x8655 = x8639 * x8649;
for(int x8651=0; x8651 < x8638; x8651++) {
int32_t x8662 = x8644 * x8651;
int32_t x8663 = x8661 + x8662;
int32_t x8657 = x8638 * x8651;
for(int x8652=0; x8652 < x8638; x8652++) {
int32_t x8664 = x8645 * x8652;
int32_t x8665 = x8663 + x8664;
float x8667 = x8559[x8665];
float x8668 = x45[x8666];
int32_t x8654 = x8652 + x8653;
int32_t x8656 = x8654 + x8655;
int32_t x8658 = x8656 + x8657;
float x8669 = x8667 * x8668;
x8642[x8658] = x8669;

}

}

}

}
int32_t x8679 = 0;
int32_t x8680 = 1;
x8680 *= 1;
x8679 += 1;
x8680 *= 1;
x8680 *= 1;
int32_t x8685 = x8679;
bool x8686 = x8685 >= 2;
if (x8686) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8691 = x8685 == 0;
if (x8691) {
int32_t x8692 = x8680;
bool x8693 = x8692 == 128;
if (x8693) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8700 = x8680;
bool x8702 = x8636 == 1;
int32_t x8701 = 128 / x8700;
bool x8703 = x8701 == 1;
bool x8707;
if (x454) {
bool x8704 = x8702 || x8703;
bool x8705 = x8636 == x8701;
bool x8706 = x8704 || x8705;
x8707 = x8706;
} else {
x8707 = false;
}
bool x8711;
if (x8707) {
x8711 = x8710;
} else {
x8711 = false;
}
bool x8712;
if (x8711) {
x8712 = x8710;
} else {
x8712 = false;
}
if (x8712) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8636,x8638,x8638,1,x8701,1,1);
assert(false && "");
}
bool x8718 = x8636 <= x8701;
int32_t x8719;
if (x8718) {
x8719 = x8701;
} else {
x8719 = x8636;
}
int32_t x8723 = x8719 * x8722;
int32_t x8724 = 64 * x8723;
float* x8725 = (float*)myMalloc(x8724 * sizeof(float));;
int32_t x8726;
if (x8702) {
x8726 = 0;
} else {
x8726 = x8639;
}
int32_t x8729;
if (x8703) {
x8729 = 0;
} else {
x8729 = 1;
}
for(int x8730=0; x8730 < 64; x8730++) {
int32_t x8742 = x8640 * x8730;
int32_t x8736 = x8723 * x8730;
for(int x8732=0; x8732 < x8719; x8732++) {
int32_t x8743 = x8726 * x8732;
int32_t x8744 = x8742 + x8743;
int32_t x8749 = x8729 * x8732;
int32_t x8738 = x8722 * x8732;
for(int x8734=0; x8734 < x8721; x8734++) {
int32_t x8745 = x8727 * x8734;
int32_t x8746 = x8744 + x8745;
int32_t x8740 = x8721 * x8734;
for(int x8735=0; x8735 < x8721; x8735++) {
int32_t x8747 = x8728 * x8735;
int32_t x8748 = x8746 + x8747;
float x8750 = x8642[x8748];
float x8751 = x170[x8749];
int32_t x8737 = x8735 + x8736;
int32_t x8739 = x8737 + x8738;
int32_t x8741 = x8739 + x8740;
float x8752 = x8750 + x8751;
x8725[x8741] = x8752;

}

}

}

}
float* x8762 = (float*)myMalloc(x8724 * sizeof(float));;
for(int x8764=0; x8764 < x8724; x8764++) {
float x8765 = x8725[x8764];
bool x8766 = x8765 < 0.0f;
if (x8766) {
x8762[x8764] = 0.0f;
} else {
float x8769 = x8725[x8764];
x8762[x8764] = x8769;
}

}
float* x8784 = (float*)myMalloc(x8783 * sizeof(float));;
int32_t x8785 = 9 * x8719;
int32_t x8788 = 64 * x8785;
int32_t x8789 = x8788 * x8779;
float* x8790 = (float*)myMalloc(x8789 * sizeof(float));;
int32_t x8786 = x8785 * x8779;
int32_t x8798 = x8719 * 3;
int32_t x8799 = x8798 * 3;
for(int x8791=0; x8791 < 64; x8791++) {
int32_t x8792 = x8791 * x8723;
float* x8793 = x8762+x8792;
int32_t x8794 = x8791 * x8780;
float* x8795 = x8784+x8794;
int32_t x8796 = x8791 * x8786;
float* x8797 = x8790+x8796;
for(int x8801=0; x8801 < x8799; x8801++) {
int32_t x8802 = x8801 / 9;
int32_t x8806 = x8802 * 3;
int32_t x8807 = x8806 * 3;
int32_t x8808 = x8807 * x8778;
int32_t x8809 = x8808 * x8778;
int32_t x8803 = x8801 % 9;
int32_t x8804 = x8803 / 3;
int32_t x8810 = x8804 * 3;
int32_t x8811 = x8810 * x8778;
int32_t x8812 = x8811 * x8778;
int32_t x8813 = x8809 + x8812;
int32_t x8805 = x8803 % 3;
int32_t x8814 = x8805 * x8778;
int32_t x8815 = x8814 * x8778;
int32_t x8816 = x8813 + x8815;
float* x8817 = x8797+x8816;
int32_t x8818 = x8802 * x8721;
int32_t x8819 = x8818 * x8721;
float* x8820 = x8793+x8819;
int32_t x8833 = 1 - x8805;
bool x8834 = x8833 > 0;
int32_t x8835;
if (x8834) {
x8835 = x8833;
} else {
x8835 = 0;
}
int32_t x8836 = 3 - x8805;
int32_t x8837 = x8836 - 1;
int32_t x8838 = 1 - x8837;
bool x8839 = x8838 > 0;
int32_t x8840;
if (x8839) {
x8840 = x8838;
} else {
x8840 = 0;
}
int32_t x8841 = x8778 - x8840;
int32_t x8842 = x8841 - x8835;
bool x8843 = x8842 <= 0;
bool x8847 = x8835 > 0;
int32_t x8832 = -1 + x8805;
bool x8860 = x8840 > 0;
for(int x8822=0; x8822 < x8778; x8822++) {
int32_t x8823 = x8822 - 1;
int32_t x8824 = x8823 + x8804;
bool x8825 = x8824 < 0;
bool x8826 = x8824 >= x8721;
bool x8827 = x8825 || x8826;
if (x8827) {
int32_t x8828 = x8822 * x8778;
float* x8829 = x8817+x8828;
memset(x8829, 0, 4 * x8778);;
} else {
if (x8843) {
int32_t x8828 = x8822 * x8778;
float* x8844 = x8817+x8828;
memset(x8844, 0, 4 * x8778);;
} else {
int32_t x8828 = x8822 * x8778;
if (x8847) {
float* x8848 = x8817+x8828;
memset(x8848, 0, 4 * x8835);;
} else {
}
// may have segfault here
int32_t x8853 = x8828 + x8835;
float* x8854 = x8817+x8853;
int32_t x8855 = x8824 * x8721;
int32_t x8856 = x8855 + x8832;
int32_t x8857 = x8856 + x8835;
float* x8858 = x8820+x8857;
memcpy(x8854, x8858, 4 * x8842);;
if (x8860) {
int32_t x8861 = x8828 + x8778;
int32_t x8862 = x8861 - x8840;
float* x8863 = x8817+x8862;
memset(x8863, 0, 4 * x8840);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x8779,x8785,1,x191,x8785,x8797,x8779,1,x8795,x8779);

}
int32_t x8878 = 0;
int32_t x8879 = 1;
x8879 *= 1;
x8878 += 1;
x8879 *= 1;
x8879 *= 1;
int32_t x8884 = x8878;
bool x8885 = x8884 >= 2;
if (x8885) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8890 = x8884 == 0;
if (x8890) {
int32_t x8891 = x8879;
bool x8892 = x8891 == 128;
if (x8892) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8899 = x8879;
int32_t x8900 = 128 / x8899;
bool x8901 = x8900 == 1;
bool x8904;
if (x454) {
bool x8902 = 128 == x8900;
bool x8903 = x8901 || x8902;
x8904 = x8903;
} else {
x8904 = false;
}
bool x8908;
if (x8904) {
x8908 = x8907;
} else {
x8908 = false;
}
bool x8909;
if (x8908) {
x8909 = x8907;
} else {
x8909 = false;
}
if (x8909) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x8778,x8778,1,x8900,1,1);
assert(false && "");
}
bool x8915 = 128 <= x8900;
int32_t x8916;
if (x8915) {
x8916 = x8900;
} else {
x8916 = 128;
}
int32_t x8920 = x8916 * x8919;
int32_t x8921 = 64 * x8920;
float* x8922 = (float*)myMalloc(x8921 * sizeof(float));;
int32_t x8925;
if (x8901) {
x8925 = 0;
} else {
x8925 = 1;
}
for(int x8926=0; x8926 < 64; x8926++) {
int32_t x8938 = x8780 * x8926;
int32_t x8932 = x8920 * x8926;
for(int x8928=0; x8928 < x8916; x8928++) {
int32_t x8939 = x8779 * x8928;
int32_t x8940 = x8938 + x8939;
int32_t x8945 = x8925 * x8928;
int32_t x8934 = x8919 * x8928;
for(int x8930=0; x8930 < x8918; x8930++) {
int32_t x8941 = x8923 * x8930;
int32_t x8942 = x8940 + x8941;
int32_t x8936 = x8918 * x8930;
for(int x8931=0; x8931 < x8918; x8931++) {
int32_t x8943 = x8924 * x8931;
int32_t x8944 = x8942 + x8943;
float x8946 = x8784[x8944];
float x8947 = x217[x8945];
int32_t x8933 = x8931 + x8932;
int32_t x8935 = x8933 + x8934;
int32_t x8937 = x8935 + x8936;
float x8948 = x8946 - x8947;
x8922[x8937] = x8948;

}

}

}

}
float* x8958 = (float*)myMalloc(128 * sizeof(float));;
for(int x8959=0; x8959 < 128; x8959++) {
float x8960 = x266[x8959];
float x8961 = x8960 + 1.0E-5f;
x8958[x8959] = x8961;

}
float* x8965 = (float*)myMalloc(128 * sizeof(float));;
for(int x8966=0; x8966 < 128; x8966++) {
float x8967 = x8958[x8966];
double x8968 = (double)x8967;
double x8969 = sqrt(x8968);
float x8970 = (float)x8969;
x8965[x8966] = x8970;

}
int32_t x8974 = 0;
int32_t x8975 = 1;
x8975 *= 1;
x8974 += 1;
x8975 *= 1;
x8975 *= 1;
int32_t x8980 = x8974;
bool x8981 = x8980 >= 2;
if (x8981) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8986 = x8980 == 0;
if (x8986) {
int32_t x8987 = x8975;
bool x8988 = x8987 == 128;
if (x8988) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8995 = x8975;
bool x8997 = x8916 == 1;
int32_t x8996 = 128 / x8995;
bool x8998 = x8996 == 1;
bool x9002;
if (x454) {
bool x8999 = x8997 || x8998;
bool x9000 = x8916 == x8996;
bool x9001 = x8999 || x9000;
x9002 = x9001;
} else {
x9002 = false;
}
bool x9006;
if (x9002) {
x9006 = x9005;
} else {
x9006 = false;
}
bool x9007;
if (x9006) {
x9007 = x9005;
} else {
x9007 = false;
}
if (x9007) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8916,x8918,x8918,1,x8996,1,1);
assert(false && "");
}
bool x9013 = x8916 <= x8996;
int32_t x9014;
if (x9013) {
x9014 = x8996;
} else {
x9014 = x8916;
}
int32_t x9018 = x9014 * x9017;
int32_t x9019 = 64 * x9018;
float* x9020 = (float*)myMalloc(x9019 * sizeof(float));;
int32_t x9021;
if (x8997) {
x9021 = 0;
} else {
x9021 = x8919;
}
int32_t x9024;
if (x8998) {
x9024 = 0;
} else {
x9024 = 1;
}
for(int x9025=0; x9025 < 64; x9025++) {
int32_t x9037 = x8920 * x9025;
int32_t x9031 = x9018 * x9025;
for(int x9027=0; x9027 < x9014; x9027++) {
int32_t x9038 = x9021 * x9027;
int32_t x9039 = x9037 + x9038;
int32_t x9044 = x9024 * x9027;
int32_t x9033 = x9017 * x9027;
for(int x9029=0; x9029 < x9016; x9029++) {
int32_t x9040 = x9022 * x9029;
int32_t x9041 = x9039 + x9040;
int32_t x9035 = x9016 * x9029;
for(int x9030=0; x9030 < x9016; x9030++) {
int32_t x9042 = x9023 * x9030;
int32_t x9043 = x9041 + x9042;
float x9045 = x8922[x9043];
float x9046 = x8965[x9044];
int32_t x9032 = x9030 + x9031;
int32_t x9034 = x9032 + x9033;
int32_t x9036 = x9034 + x9035;
float x9047 = x9045 / x9046;
x9020[x9036] = x9047;

}

}

}

}
int32_t x9057 = 0;
int32_t x9058 = 1;
x9058 *= 1;
x9057 += 1;
x9058 *= 1;
x9058 *= 1;
int32_t x9063 = x9057;
bool x9064 = x9063 >= 2;
if (x9064) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9069 = x9063 == 0;
if (x9069) {
int32_t x9070 = x9058;
bool x9071 = x9070 == 128;
if (x9071) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9078 = x9058;
bool x9080 = x9014 == 1;
int32_t x9079 = 128 / x9078;
bool x9081 = x9079 == 1;
bool x9085;
if (x454) {
bool x9082 = x9080 || x9081;
bool x9083 = x9014 == x9079;
bool x9084 = x9082 || x9083;
x9085 = x9084;
} else {
x9085 = false;
}
bool x9089;
if (x9085) {
x9089 = x9088;
} else {
x9089 = false;
}
bool x9090;
if (x9089) {
x9090 = x9088;
} else {
x9090 = false;
}
if (x9090) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9014,x9016,x9016,1,x9079,1,1);
assert(false && "");
}
bool x9096 = x9014 <= x9079;
int32_t x9097;
if (x9096) {
x9097 = x9079;
} else {
x9097 = x9014;
}
int32_t x9101 = x9097 * x9100;
int32_t x9102 = 64 * x9101;
float* x9103 = (float*)myMalloc(x9102 * sizeof(float));;
int32_t x9104;
if (x9080) {
x9104 = 0;
} else {
x9104 = x9017;
}
int32_t x9107;
if (x9081) {
x9107 = 0;
} else {
x9107 = 1;
}
for(int x9108=0; x9108 < 64; x9108++) {
int32_t x9120 = x9018 * x9108;
int32_t x9114 = x9101 * x9108;
for(int x9110=0; x9110 < x9097; x9110++) {
int32_t x9121 = x9104 * x9110;
int32_t x9122 = x9120 + x9121;
int32_t x9127 = x9107 * x9110;
int32_t x9116 = x9100 * x9110;
for(int x9112=0; x9112 < x9099; x9112++) {
int32_t x9123 = x9105 * x9112;
int32_t x9124 = x9122 + x9123;
int32_t x9118 = x9099 * x9112;
for(int x9113=0; x9113 < x9099; x9113++) {
int32_t x9125 = x9106 * x9113;
int32_t x9126 = x9124 + x9125;
float x9128 = x9020[x9126];
float x9129 = x127[x9127];
int32_t x9115 = x9113 + x9114;
int32_t x9117 = x9115 + x9116;
int32_t x9119 = x9117 + x9118;
float x9130 = x9128 * x9129;
x9103[x9119] = x9130;

}

}

}

}
int32_t x9140 = 0;
int32_t x9141 = 1;
x9141 *= 1;
x9140 += 1;
x9141 *= 1;
x9141 *= 1;
int32_t x9146 = x9140;
bool x9147 = x9146 >= 2;
if (x9147) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9152 = x9146 == 0;
if (x9152) {
int32_t x9153 = x9141;
bool x9154 = x9153 == 128;
if (x9154) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9161 = x9141;
bool x9163 = x9097 == 1;
int32_t x9162 = 128 / x9161;
bool x9164 = x9162 == 1;
bool x9168;
if (x454) {
bool x9165 = x9163 || x9164;
bool x9166 = x9097 == x9162;
bool x9167 = x9165 || x9166;
x9168 = x9167;
} else {
x9168 = false;
}
bool x9172;
if (x9168) {
x9172 = x9171;
} else {
x9172 = false;
}
bool x9173;
if (x9172) {
x9173 = x9171;
} else {
x9173 = false;
}
if (x9173) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9097,x9099,x9099,1,x9162,1,1);
assert(false && "");
}
bool x9179 = x9097 <= x9162;
int32_t x9180;
if (x9179) {
x9180 = x9162;
} else {
x9180 = x9097;
}
int32_t x9184 = x9180 * x9183;
int32_t x9185 = 64 * x9184;
float* x9186 = (float*)myMalloc(x9185 * sizeof(float));;
int32_t x9187;
if (x9163) {
x9187 = 0;
} else {
x9187 = x9100;
}
int32_t x9190;
if (x9164) {
x9190 = 0;
} else {
x9190 = 1;
}
for(int x9191=0; x9191 < 64; x9191++) {
int32_t x9203 = x9101 * x9191;
int32_t x9197 = x9184 * x9191;
for(int x9193=0; x9193 < x9180; x9193++) {
int32_t x9204 = x9187 * x9193;
int32_t x9205 = x9203 + x9204;
int32_t x9210 = x9190 * x9193;
int32_t x9199 = x9183 * x9193;
for(int x9195=0; x9195 < x9182; x9195++) {
int32_t x9206 = x9188 * x9195;
int32_t x9207 = x9205 + x9206;
int32_t x9201 = x9182 * x9195;
for(int x9196=0; x9196 < x9182; x9196++) {
int32_t x9208 = x9189 * x9196;
int32_t x9209 = x9207 + x9208;
float x9211 = x9103[x9209];
float x9212 = x61[x9210];
int32_t x9198 = x9196 + x9197;
int32_t x9200 = x9198 + x9199;
int32_t x9202 = x9200 + x9201;
float x9213 = x9211 + x9212;
x9186[x9202] = x9213;

}

}

}

}
float* x9223 = (float*)myMalloc(x9185 * sizeof(float));;
for(int x9225=0; x9225 < x9185; x9225++) {
float x9226 = x9186[x9225];
bool x9227 = x9226 < 0.0f;
if (x9227) {
x9223[x9225] = 0.0f;
} else {
float x9230 = x9186[x9225];
x9223[x9225] = x9230;
}

}
float* x9244 = (float*)myMalloc(x9243 * sizeof(float));;
int32_t x9247 = 64 * x9180;
int32_t x9248 = x9247 * x9239;
float* x9249 = (float*)myMalloc(x9248 * sizeof(float));;
int32_t x9245 = x9180 * x9239;
for(int x9250=0; x9250 < 64; x9250++) {
int32_t x9251 = x9250 * x9184;
float* x9252 = x9223+x9251;
int32_t x9253 = x9250 * x9240;
float* x9254 = x9244+x9253;
int32_t x9255 = x9250 * x9245;
float* x9256 = x9249+x9255;
for(int x9257=0; x9257 < x9180; x9257++) {
int32_t x9258 = x9257 / 1;
int32_t x9262 = x9258 * x9238;
int32_t x9263 = x9262 * x9238;
int32_t x9259 = x9257 % 1;
int32_t x9260 = x9259 / 1;
int32_t x9264 = x9260 * x9238;
int32_t x9265 = x9264 * x9238;
int32_t x9266 = x9263 + x9265;
int32_t x9261 = x9259 % 1;
int32_t x9267 = x9261 * x9238;
int32_t x9268 = x9267 * x9238;
int32_t x9269 = x9266 + x9268;
float* x9270 = x9256+x9269;
int32_t x9271 = x9258 * x9182;
int32_t x9272 = x9271 * x9182;
float* x9273 = x9252+x9272;
for(int x9275=0; x9275 < x9238; x9275++) {
int32_t x9277 = x9275 * x9238;
float* x9278 = x9270+x9277;
int32_t x9276 = x9275 + x9260;
int32_t x9279 = x9276 * x9182;
int32_t x9280 = x9279 + x9261;
float* x9281 = x9273+x9280;
memcpy(x9278, x9281, 4 * x9238);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x9239,x9180,1,x41,x9180,x9256,x9239,1,x9254,x9239);

}
int32_t x9290 = 0;
int32_t x9291 = 1;
x9291 *= 1;
x9290 += 1;
x9291 *= 1;
x9291 *= 1;
int32_t x9296 = x9290;
bool x9297 = x9296 >= 2;
if (x9297) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9302 = x9296 == 0;
if (x9302) {
int32_t x9303 = x9291;
bool x9304 = x9303 == 512;
if (x9304) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9311 = x9291;
int32_t x9312 = 512 / x9311;
bool x9313 = x9312 == 1;
bool x9316;
if (x454) {
bool x9314 = 512 == x9312;
bool x9315 = x9313 || x9314;
x9316 = x9315;
} else {
x9316 = false;
}
bool x9320;
if (x9316) {
x9320 = x9319;
} else {
x9320 = false;
}
bool x9321;
if (x9320) {
x9321 = x9319;
} else {
x9321 = false;
}
if (x9321) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x9238,x9238,1,x9312,1,1);
assert(false && "");
}
bool x9327 = 512 <= x9312;
int32_t x9328;
if (x9327) {
x9328 = x9312;
} else {
x9328 = 512;
}
int32_t x9332 = x9328 * x9331;
int32_t x9333 = 64 * x9332;
float* x9334 = (float*)myMalloc(x9333 * sizeof(float));;
int32_t x9337;
if (x9313) {
x9337 = 0;
} else {
x9337 = 1;
}
for(int x9338=0; x9338 < 64; x9338++) {
int32_t x9350 = x9240 * x9338;
int32_t x9344 = x9332 * x9338;
for(int x9340=0; x9340 < x9328; x9340++) {
int32_t x9351 = x9239 * x9340;
int32_t x9352 = x9350 + x9351;
int32_t x9357 = x9337 * x9340;
int32_t x9346 = x9331 * x9340;
for(int x9342=0; x9342 < x9330; x9342++) {
int32_t x9353 = x9335 * x9342;
int32_t x9354 = x9352 + x9353;
int32_t x9348 = x9330 * x9342;
for(int x9343=0; x9343 < x9330; x9343++) {
int32_t x9355 = x9336 * x9343;
int32_t x9356 = x9354 + x9355;
float x9358 = x9244[x9356];
float x9359 = x25[x9357];
int32_t x9345 = x9343 + x9344;
int32_t x9347 = x9345 + x9346;
int32_t x9349 = x9347 + x9348;
float x9360 = x9358 - x9359;
x9334[x9349] = x9360;

}

}

}

}
float* x9370 = (float*)myMalloc(512 * sizeof(float));;
for(int x9371=0; x9371 < 512; x9371++) {
float x9372 = x223[x9371];
float x9373 = x9372 + 1.0E-5f;
x9370[x9371] = x9373;

}
float* x9377 = (float*)myMalloc(512 * sizeof(float));;
for(int x9378=0; x9378 < 512; x9378++) {
float x9379 = x9370[x9378];
double x9380 = (double)x9379;
double x9381 = sqrt(x9380);
float x9382 = (float)x9381;
x9377[x9378] = x9382;

}
int32_t x9386 = 0;
int32_t x9387 = 1;
x9387 *= 1;
x9386 += 1;
x9387 *= 1;
x9387 *= 1;
int32_t x9392 = x9386;
bool x9393 = x9392 >= 2;
if (x9393) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9398 = x9392 == 0;
if (x9398) {
int32_t x9399 = x9387;
bool x9400 = x9399 == 512;
if (x9400) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9407 = x9387;
bool x9409 = x9328 == 1;
int32_t x9408 = 512 / x9407;
bool x9410 = x9408 == 1;
bool x9414;
if (x454) {
bool x9411 = x9409 || x9410;
bool x9412 = x9328 == x9408;
bool x9413 = x9411 || x9412;
x9414 = x9413;
} else {
x9414 = false;
}
bool x9418;
if (x9414) {
x9418 = x9417;
} else {
x9418 = false;
}
bool x9419;
if (x9418) {
x9419 = x9417;
} else {
x9419 = false;
}
if (x9419) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9328,x9330,x9330,1,x9408,1,1);
assert(false && "");
}
bool x9425 = x9328 <= x9408;
int32_t x9426;
if (x9425) {
x9426 = x9408;
} else {
x9426 = x9328;
}
int32_t x9430 = x9426 * x9429;
int32_t x9431 = 64 * x9430;
float* x9432 = (float*)myMalloc(x9431 * sizeof(float));;
int32_t x9433;
if (x9409) {
x9433 = 0;
} else {
x9433 = x9331;
}
int32_t x9436;
if (x9410) {
x9436 = 0;
} else {
x9436 = 1;
}
for(int x9437=0; x9437 < 64; x9437++) {
int32_t x9449 = x9332 * x9437;
int32_t x9443 = x9430 * x9437;
for(int x9439=0; x9439 < x9426; x9439++) {
int32_t x9450 = x9433 * x9439;
int32_t x9451 = x9449 + x9450;
int32_t x9456 = x9436 * x9439;
int32_t x9445 = x9429 * x9439;
for(int x9441=0; x9441 < x9428; x9441++) {
int32_t x9452 = x9434 * x9441;
int32_t x9453 = x9451 + x9452;
int32_t x9447 = x9428 * x9441;
for(int x9442=0; x9442 < x9428; x9442++) {
int32_t x9454 = x9435 * x9442;
int32_t x9455 = x9453 + x9454;
float x9457 = x9334[x9455];
float x9458 = x9377[x9456];
int32_t x9444 = x9442 + x9443;
int32_t x9446 = x9444 + x9445;
int32_t x9448 = x9446 + x9447;
float x9459 = x9457 / x9458;
x9432[x9448] = x9459;

}

}

}

}
int32_t x9469 = 0;
int32_t x9470 = 1;
x9470 *= 1;
x9469 += 1;
x9470 *= 1;
x9470 *= 1;
int32_t x9475 = x9469;
bool x9476 = x9475 >= 2;
if (x9476) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9481 = x9475 == 0;
if (x9481) {
int32_t x9482 = x9470;
bool x9483 = x9482 == 512;
if (x9483) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9490 = x9470;
bool x9492 = x9426 == 1;
int32_t x9491 = 512 / x9490;
bool x9493 = x9491 == 1;
bool x9497;
if (x454) {
bool x9494 = x9492 || x9493;
bool x9495 = x9426 == x9491;
bool x9496 = x9494 || x9495;
x9497 = x9496;
} else {
x9497 = false;
}
bool x9501;
if (x9497) {
x9501 = x9500;
} else {
x9501 = false;
}
bool x9502;
if (x9501) {
x9502 = x9500;
} else {
x9502 = false;
}
if (x9502) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9426,x9428,x9428,1,x9491,1,1);
assert(false && "");
}
bool x9508 = x9426 <= x9491;
int32_t x9509;
if (x9508) {
x9509 = x9491;
} else {
x9509 = x9426;
}
int32_t x9513 = x9509 * x9512;
int32_t x9514 = 64 * x9513;
float* x9515 = (float*)myMalloc(x9514 * sizeof(float));;
int32_t x9516;
if (x9492) {
x9516 = 0;
} else {
x9516 = x9429;
}
int32_t x9519;
if (x9493) {
x9519 = 0;
} else {
x9519 = 1;
}
for(int x9520=0; x9520 < 64; x9520++) {
int32_t x9532 = x9430 * x9520;
int32_t x9526 = x9513 * x9520;
for(int x9522=0; x9522 < x9509; x9522++) {
int32_t x9533 = x9516 * x9522;
int32_t x9534 = x9532 + x9533;
int32_t x9539 = x9519 * x9522;
int32_t x9528 = x9512 * x9522;
for(int x9524=0; x9524 < x9511; x9524++) {
int32_t x9535 = x9517 * x9524;
int32_t x9536 = x9534 + x9535;
int32_t x9530 = x9511 * x9524;
for(int x9525=0; x9525 < x9511; x9525++) {
int32_t x9537 = x9518 * x9525;
int32_t x9538 = x9536 + x9537;
float x9540 = x9432[x9538];
float x9541 = x167[x9539];
int32_t x9527 = x9525 + x9526;
int32_t x9529 = x9527 + x9528;
int32_t x9531 = x9529 + x9530;
float x9542 = x9540 * x9541;
x9515[x9531] = x9542;

}

}

}

}
int32_t x9552 = 0;
int32_t x9553 = 1;
x9553 *= 1;
x9552 += 1;
x9553 *= 1;
x9553 *= 1;
int32_t x9558 = x9552;
bool x9559 = x9558 >= 2;
if (x9559) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9564 = x9558 == 0;
if (x9564) {
int32_t x9565 = x9553;
bool x9566 = x9565 == 512;
if (x9566) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9573 = x9553;
bool x9575 = x9509 == 1;
int32_t x9574 = 512 / x9573;
bool x9576 = x9574 == 1;
bool x9580;
if (x454) {
bool x9577 = x9575 || x9576;
bool x9578 = x9509 == x9574;
bool x9579 = x9577 || x9578;
x9580 = x9579;
} else {
x9580 = false;
}
bool x9584;
if (x9580) {
x9584 = x9583;
} else {
x9584 = false;
}
bool x9585;
if (x9584) {
x9585 = x9583;
} else {
x9585 = false;
}
if (x9585) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9509,x9511,x9511,1,x9574,1,1);
assert(false && "");
}
bool x9591 = x9509 <= x9574;
int32_t x9592;
if (x9591) {
x9592 = x9574;
} else {
x9592 = x9509;
}
int32_t x9596 = x9592 * x9595;
int32_t x9597 = 64 * x9596;
float* x9598 = (float*)myMalloc(x9597 * sizeof(float));;
int32_t x9599;
if (x9575) {
x9599 = 0;
} else {
x9599 = x9512;
}
int32_t x9602;
if (x9576) {
x9602 = 0;
} else {
x9602 = 1;
}
for(int x9603=0; x9603 < 64; x9603++) {
int32_t x9615 = x9513 * x9603;
int32_t x9609 = x9596 * x9603;
for(int x9605=0; x9605 < x9592; x9605++) {
int32_t x9616 = x9599 * x9605;
int32_t x9617 = x9615 + x9616;
int32_t x9622 = x9602 * x9605;
int32_t x9611 = x9595 * x9605;
for(int x9607=0; x9607 < x9594; x9607++) {
int32_t x9618 = x9600 * x9607;
int32_t x9619 = x9617 + x9618;
int32_t x9613 = x9594 * x9607;
for(int x9608=0; x9608 < x9594; x9608++) {
int32_t x9620 = x9601 * x9608;
int32_t x9621 = x9619 + x9620;
float x9623 = x9515[x9621];
float x9624 = x82[x9622];
int32_t x9610 = x9608 + x9609;
int32_t x9612 = x9610 + x9611;
int32_t x9614 = x9612 + x9613;
float x9625 = x9623 + x9624;
x9598[x9614] = x9625;

}

}

}

}
bool x9635 = x9592 == 1;
bool x9636 = x9635 || x8293;
bool x9637 = x9592 == x8250;
bool x9638 = x9636 || x9637;
bool x9643;
if (x9638) {
x9643 = x9642;
} else {
x9643 = false;
}
bool x9644;
if (x9643) {
x9644 = x9642;
} else {
x9644 = false;
}
if (x9644) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9592,x9594,x9594,64,x8250,x8252,x8252);
assert(false && "");
}
bool x9650 = x9592 <= x8250;
int32_t x9651;
if (x9650) {
x9651 = x8250;
} else {
x9651 = x9592;
}
int32_t x9657;
if (x9635) {
x9657 = 0;
} else {
x9657 = x9595;
}
for(int x9660=0; x9660 < 64; x9660++) {
int32_t x9666 = x9596 * x9660;
int32_t x9673 = x8254 * x9660;
for(int x9662=0; x9662 < x9651; x9662++) {
int32_t x9667 = x9657 * x9662;
int32_t x9668 = x9666 + x9667;
int32_t x9674 = x8315 * x9662;
int32_t x9675 = x9673 + x9674;
for(int x9664=0; x9664 < x9653; x9664++) {
int32_t x9669 = x9658 * x9664;
int32_t x9670 = x9668 + x9669;
int32_t x9676 = x8316 * x9664;
int32_t x9677 = x9675 + x9676;
for(int x9665=0; x9665 < x9653; x9665++) {
int32_t x9671 = x9659 * x9665;
int32_t x9672 = x9670 + x9671;
float x9680 = x9598[x9672];
int32_t x9678 = x8317 * x9665;
int32_t x9679 = x9677 + x9678;
float x9681 = x8350[x9679];
float x9682 = x9680 + x9681;
x9598[x9672] = x9682;

}

}

}

}
float* x9692 = (float*)myMalloc(x9597 * sizeof(float));;
for(int x9694=0; x9694 < x9597; x9694++) {
float x9695 = x9598[x9694];
bool x9696 = x9695 < 0.0f;
if (x9696) {
x9692[x9694] = 0.0f;
} else {
float x9699 = x9598[x9694];
x9692[x9694] = x9699;
}

}
float* x9713 = (float*)myMalloc(x9712 * sizeof(float));;
int32_t x9716 = 64 * x9592;
int32_t x9717 = x9716 * x9708;
float* x9718 = (float*)myMalloc(x9717 * sizeof(float));;
int32_t x9714 = x9592 * x9708;
for(int x9719=0; x9719 < 64; x9719++) {
int32_t x9720 = x9719 * x9596;
float* x9721 = x9692+x9720;
int32_t x9722 = x9719 * x9709;
float* x9723 = x9713+x9722;
int32_t x9724 = x9719 * x9714;
float* x9725 = x9718+x9724;
for(int x9726=0; x9726 < x9592; x9726++) {
int32_t x9727 = x9726 / 1;
int32_t x9731 = x9727 * x9707;
int32_t x9732 = x9731 * x9707;
int32_t x9728 = x9726 % 1;
int32_t x9729 = x9728 / 1;
int32_t x9733 = x9729 * x9707;
int32_t x9734 = x9733 * x9707;
int32_t x9735 = x9732 + x9734;
int32_t x9730 = x9728 % 1;
int32_t x9736 = x9730 * x9707;
int32_t x9737 = x9736 * x9707;
int32_t x9738 = x9735 + x9737;
float* x9739 = x9725+x9738;
int32_t x9740 = x9727 * x9594;
int32_t x9741 = x9740 * x9594;
float* x9742 = x9721+x9741;
for(int x9744=0; x9744 < x9707; x9744++) {
int32_t x9746 = x9744 * x9707;
float* x9747 = x9739+x9746;
int32_t x9745 = x9744 + x9729;
int32_t x9748 = x9745 * x9594;
int32_t x9749 = x9748 + x9730;
float* x9750 = x9742+x9749;
memcpy(x9747, x9750, 4 * x9707);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x9708,x9592,1,x132,x9592,x9725,x9708,1,x9723,x9708);

}
int32_t x9759 = 0;
int32_t x9760 = 1;
x9760 *= 1;
x9759 += 1;
x9760 *= 1;
x9760 *= 1;
int32_t x9765 = x9759;
bool x9766 = x9765 >= 2;
if (x9766) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9771 = x9765 == 0;
if (x9771) {
int32_t x9772 = x9760;
bool x9773 = x9772 == 128;
if (x9773) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9780 = x9760;
int32_t x9781 = 128 / x9780;
bool x9782 = x9781 == 1;
bool x9785;
if (x454) {
bool x9783 = 128 == x9781;
bool x9784 = x9782 || x9783;
x9785 = x9784;
} else {
x9785 = false;
}
bool x9789;
if (x9785) {
x9789 = x9788;
} else {
x9789 = false;
}
bool x9790;
if (x9789) {
x9790 = x9788;
} else {
x9790 = false;
}
if (x9790) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x9707,x9707,1,x9781,1,1);
assert(false && "");
}
bool x9796 = 128 <= x9781;
int32_t x9797;
if (x9796) {
x9797 = x9781;
} else {
x9797 = 128;
}
int32_t x9801 = x9797 * x9800;
int32_t x9802 = 64 * x9801;
float* x9803 = (float*)myMalloc(x9802 * sizeof(float));;
int32_t x9806;
if (x9782) {
x9806 = 0;
} else {
x9806 = 1;
}
for(int x9807=0; x9807 < 64; x9807++) {
int32_t x9819 = x9709 * x9807;
int32_t x9813 = x9801 * x9807;
for(int x9809=0; x9809 < x9797; x9809++) {
int32_t x9820 = x9708 * x9809;
int32_t x9821 = x9819 + x9820;
int32_t x9826 = x9806 * x9809;
int32_t x9815 = x9800 * x9809;
for(int x9811=0; x9811 < x9799; x9811++) {
int32_t x9822 = x9804 * x9811;
int32_t x9823 = x9821 + x9822;
int32_t x9817 = x9799 * x9811;
for(int x9812=0; x9812 < x9799; x9812++) {
int32_t x9824 = x9805 * x9812;
int32_t x9825 = x9823 + x9824;
float x9827 = x9713[x9825];
float x9828 = x236[x9826];
int32_t x9814 = x9812 + x9813;
int32_t x9816 = x9814 + x9815;
int32_t x9818 = x9816 + x9817;
float x9829 = x9827 - x9828;
x9803[x9818] = x9829;

}

}

}

}
float* x9839 = (float*)myMalloc(128 * sizeof(float));;
for(int x9840=0; x9840 < 128; x9840++) {
float x9841 = x261[x9840];
float x9842 = x9841 + 1.0E-5f;
x9839[x9840] = x9842;

}
float* x9846 = (float*)myMalloc(128 * sizeof(float));;
for(int x9847=0; x9847 < 128; x9847++) {
float x9848 = x9839[x9847];
double x9849 = (double)x9848;
double x9850 = sqrt(x9849);
float x9851 = (float)x9850;
x9846[x9847] = x9851;

}
int32_t x9855 = 0;
int32_t x9856 = 1;
x9856 *= 1;
x9855 += 1;
x9856 *= 1;
x9856 *= 1;
int32_t x9861 = x9855;
bool x9862 = x9861 >= 2;
if (x9862) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9867 = x9861 == 0;
if (x9867) {
int32_t x9868 = x9856;
bool x9869 = x9868 == 128;
if (x9869) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9876 = x9856;
bool x9878 = x9797 == 1;
int32_t x9877 = 128 / x9876;
bool x9879 = x9877 == 1;
bool x9883;
if (x454) {
bool x9880 = x9878 || x9879;
bool x9881 = x9797 == x9877;
bool x9882 = x9880 || x9881;
x9883 = x9882;
} else {
x9883 = false;
}
bool x9887;
if (x9883) {
x9887 = x9886;
} else {
x9887 = false;
}
bool x9888;
if (x9887) {
x9888 = x9886;
} else {
x9888 = false;
}
if (x9888) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9797,x9799,x9799,1,x9877,1,1);
assert(false && "");
}
bool x9894 = x9797 <= x9877;
int32_t x9895;
if (x9894) {
x9895 = x9877;
} else {
x9895 = x9797;
}
int32_t x9899 = x9895 * x9898;
int32_t x9900 = 64 * x9899;
float* x9901 = (float*)myMalloc(x9900 * sizeof(float));;
int32_t x9902;
if (x9878) {
x9902 = 0;
} else {
x9902 = x9800;
}
int32_t x9905;
if (x9879) {
x9905 = 0;
} else {
x9905 = 1;
}
for(int x9906=0; x9906 < 64; x9906++) {
int32_t x9918 = x9801 * x9906;
int32_t x9912 = x9899 * x9906;
for(int x9908=0; x9908 < x9895; x9908++) {
int32_t x9919 = x9902 * x9908;
int32_t x9920 = x9918 + x9919;
int32_t x9925 = x9905 * x9908;
int32_t x9914 = x9898 * x9908;
for(int x9910=0; x9910 < x9897; x9910++) {
int32_t x9921 = x9903 * x9910;
int32_t x9922 = x9920 + x9921;
int32_t x9916 = x9897 * x9910;
for(int x9911=0; x9911 < x9897; x9911++) {
int32_t x9923 = x9904 * x9911;
int32_t x9924 = x9922 + x9923;
float x9926 = x9803[x9924];
float x9927 = x9846[x9925];
int32_t x9913 = x9911 + x9912;
int32_t x9915 = x9913 + x9914;
int32_t x9917 = x9915 + x9916;
float x9928 = x9926 / x9927;
x9901[x9917] = x9928;

}

}

}

}
int32_t x9938 = 0;
int32_t x9939 = 1;
x9939 *= 1;
x9938 += 1;
x9939 *= 1;
x9939 *= 1;
int32_t x9944 = x9938;
bool x9945 = x9944 >= 2;
if (x9945) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9950 = x9944 == 0;
if (x9950) {
int32_t x9951 = x9939;
bool x9952 = x9951 == 128;
if (x9952) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9959 = x9939;
bool x9961 = x9895 == 1;
int32_t x9960 = 128 / x9959;
bool x9962 = x9960 == 1;
bool x9966;
if (x454) {
bool x9963 = x9961 || x9962;
bool x9964 = x9895 == x9960;
bool x9965 = x9963 || x9964;
x9966 = x9965;
} else {
x9966 = false;
}
bool x9970;
if (x9966) {
x9970 = x9969;
} else {
x9970 = false;
}
bool x9971;
if (x9970) {
x9971 = x9969;
} else {
x9971 = false;
}
if (x9971) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9895,x9897,x9897,1,x9960,1,1);
assert(false && "");
}
bool x9977 = x9895 <= x9960;
int32_t x9978;
if (x9977) {
x9978 = x9960;
} else {
x9978 = x9895;
}
int32_t x9982 = x9978 * x9981;
int32_t x9983 = 64 * x9982;
float* x9984 = (float*)myMalloc(x9983 * sizeof(float));;
int32_t x9985;
if (x9961) {
x9985 = 0;
} else {
x9985 = x9898;
}
int32_t x9988;
if (x9962) {
x9988 = 0;
} else {
x9988 = 1;
}
for(int x9989=0; x9989 < 64; x9989++) {
int32_t x10001 = x9899 * x9989;
int32_t x9995 = x9982 * x9989;
for(int x9991=0; x9991 < x9978; x9991++) {
int32_t x10002 = x9985 * x9991;
int32_t x10003 = x10001 + x10002;
int32_t x10008 = x9988 * x9991;
int32_t x9997 = x9981 * x9991;
for(int x9993=0; x9993 < x9980; x9993++) {
int32_t x10004 = x9986 * x9993;
int32_t x10005 = x10003 + x10004;
int32_t x9999 = x9980 * x9993;
for(int x9994=0; x9994 < x9980; x9994++) {
int32_t x10006 = x9987 * x9994;
int32_t x10007 = x10005 + x10006;
float x10009 = x9901[x10007];
float x10010 = x39[x10008];
int32_t x9996 = x9994 + x9995;
int32_t x9998 = x9996 + x9997;
int32_t x10000 = x9998 + x9999;
float x10011 = x10009 * x10010;
x9984[x10000] = x10011;

}

}

}

}
int32_t x10021 = 0;
int32_t x10022 = 1;
x10022 *= 1;
x10021 += 1;
x10022 *= 1;
x10022 *= 1;
int32_t x10027 = x10021;
bool x10028 = x10027 >= 2;
if (x10028) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10033 = x10027 == 0;
if (x10033) {
int32_t x10034 = x10022;
bool x10035 = x10034 == 128;
if (x10035) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10042 = x10022;
bool x10044 = x9978 == 1;
int32_t x10043 = 128 / x10042;
bool x10045 = x10043 == 1;
bool x10049;
if (x454) {
bool x10046 = x10044 || x10045;
bool x10047 = x9978 == x10043;
bool x10048 = x10046 || x10047;
x10049 = x10048;
} else {
x10049 = false;
}
bool x10053;
if (x10049) {
x10053 = x10052;
} else {
x10053 = false;
}
bool x10054;
if (x10053) {
x10054 = x10052;
} else {
x10054 = false;
}
if (x10054) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9978,x9980,x9980,1,x10043,1,1);
assert(false && "");
}
bool x10060 = x9978 <= x10043;
int32_t x10061;
if (x10060) {
x10061 = x10043;
} else {
x10061 = x9978;
}
int32_t x10065 = x10061 * x10064;
int32_t x10066 = 64 * x10065;
float* x10067 = (float*)myMalloc(x10066 * sizeof(float));;
int32_t x10068;
if (x10044) {
x10068 = 0;
} else {
x10068 = x9981;
}
int32_t x10071;
if (x10045) {
x10071 = 0;
} else {
x10071 = 1;
}
for(int x10072=0; x10072 < 64; x10072++) {
int32_t x10084 = x9982 * x10072;
int32_t x10078 = x10065 * x10072;
for(int x10074=0; x10074 < x10061; x10074++) {
int32_t x10085 = x10068 * x10074;
int32_t x10086 = x10084 + x10085;
int32_t x10091 = x10071 * x10074;
int32_t x10080 = x10064 * x10074;
for(int x10076=0; x10076 < x10063; x10076++) {
int32_t x10087 = x10069 * x10076;
int32_t x10088 = x10086 + x10087;
int32_t x10082 = x10063 * x10076;
for(int x10077=0; x10077 < x10063; x10077++) {
int32_t x10089 = x10070 * x10077;
int32_t x10090 = x10088 + x10089;
float x10092 = x9984[x10090];
float x10093 = x242[x10091];
int32_t x10079 = x10077 + x10078;
int32_t x10081 = x10079 + x10080;
int32_t x10083 = x10081 + x10082;
float x10094 = x10092 + x10093;
x10067[x10083] = x10094;

}

}

}

}
float* x10104 = (float*)myMalloc(x10066 * sizeof(float));;
for(int x10106=0; x10106 < x10066; x10106++) {
float x10107 = x10067[x10106];
bool x10108 = x10107 < 0.0f;
if (x10108) {
x10104[x10106] = 0.0f;
} else {
float x10111 = x10067[x10106];
x10104[x10106] = x10111;
}

}
float* x10126 = (float*)myMalloc(x10125 * sizeof(float));;
int32_t x10127 = 9 * x10061;
int32_t x10130 = 64 * x10127;
int32_t x10131 = x10130 * x10121;
float* x10132 = (float*)myMalloc(x10131 * sizeof(float));;
int32_t x10128 = x10127 * x10121;
int32_t x10140 = x10061 * 3;
int32_t x10141 = x10140 * 3;
for(int x10133=0; x10133 < 64; x10133++) {
int32_t x10134 = x10133 * x10065;
float* x10135 = x10104+x10134;
int32_t x10136 = x10133 * x10122;
float* x10137 = x10126+x10136;
int32_t x10138 = x10133 * x10128;
float* x10139 = x10132+x10138;
for(int x10143=0; x10143 < x10141; x10143++) {
int32_t x10144 = x10143 / 9;
int32_t x10148 = x10144 * 3;
int32_t x10149 = x10148 * 3;
int32_t x10150 = x10149 * x10120;
int32_t x10151 = x10150 * x10120;
int32_t x10145 = x10143 % 9;
int32_t x10146 = x10145 / 3;
int32_t x10152 = x10146 * 3;
int32_t x10153 = x10152 * x10120;
int32_t x10154 = x10153 * x10120;
int32_t x10155 = x10151 + x10154;
int32_t x10147 = x10145 % 3;
int32_t x10156 = x10147 * x10120;
int32_t x10157 = x10156 * x10120;
int32_t x10158 = x10155 + x10157;
float* x10159 = x10139+x10158;
int32_t x10160 = x10144 * x10063;
int32_t x10161 = x10160 * x10063;
float* x10162 = x10135+x10161;
int32_t x10175 = 1 - x10147;
bool x10176 = x10175 > 0;
int32_t x10177;
if (x10176) {
x10177 = x10175;
} else {
x10177 = 0;
}
int32_t x10178 = 3 - x10147;
int32_t x10179 = x10178 - 1;
int32_t x10180 = 1 - x10179;
bool x10181 = x10180 > 0;
int32_t x10182;
if (x10181) {
x10182 = x10180;
} else {
x10182 = 0;
}
int32_t x10183 = x10120 - x10182;
int32_t x10184 = x10183 - x10177;
bool x10185 = x10184 <= 0;
bool x10189 = x10177 > 0;
int32_t x10174 = -1 + x10147;
bool x10202 = x10182 > 0;
for(int x10164=0; x10164 < x10120; x10164++) {
int32_t x10165 = x10164 - 1;
int32_t x10166 = x10165 + x10146;
bool x10167 = x10166 < 0;
bool x10168 = x10166 >= x10063;
bool x10169 = x10167 || x10168;
if (x10169) {
int32_t x10170 = x10164 * x10120;
float* x10171 = x10159+x10170;
memset(x10171, 0, 4 * x10120);;
} else {
if (x10185) {
int32_t x10170 = x10164 * x10120;
float* x10186 = x10159+x10170;
memset(x10186, 0, 4 * x10120);;
} else {
int32_t x10170 = x10164 * x10120;
if (x10189) {
float* x10190 = x10159+x10170;
memset(x10190, 0, 4 * x10177);;
} else {
}
// may have segfault here
int32_t x10195 = x10170 + x10177;
float* x10196 = x10159+x10195;
int32_t x10197 = x10166 * x10063;
int32_t x10198 = x10197 + x10174;
int32_t x10199 = x10198 + x10177;
float* x10200 = x10162+x10199;
memcpy(x10196, x10200, 4 * x10184);;
if (x10202) {
int32_t x10203 = x10170 + x10120;
int32_t x10204 = x10203 - x10182;
float* x10205 = x10159+x10204;
memset(x10205, 0, 4 * x10182);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x10121,x10127,1,x165,x10127,x10139,x10121,1,x10137,x10121);

}
int32_t x10220 = 0;
int32_t x10221 = 1;
x10221 *= 1;
x10220 += 1;
x10221 *= 1;
x10221 *= 1;
int32_t x10226 = x10220;
bool x10227 = x10226 >= 2;
if (x10227) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10232 = x10226 == 0;
if (x10232) {
int32_t x10233 = x10221;
bool x10234 = x10233 == 128;
if (x10234) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10241 = x10221;
int32_t x10242 = 128 / x10241;
bool x10243 = x10242 == 1;
bool x10246;
if (x454) {
bool x10244 = 128 == x10242;
bool x10245 = x10243 || x10244;
x10246 = x10245;
} else {
x10246 = false;
}
bool x10250;
if (x10246) {
x10250 = x10249;
} else {
x10250 = false;
}
bool x10251;
if (x10250) {
x10251 = x10249;
} else {
x10251 = false;
}
if (x10251) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x10120,x10120,1,x10242,1,1);
assert(false && "");
}
bool x10257 = 128 <= x10242;
int32_t x10258;
if (x10257) {
x10258 = x10242;
} else {
x10258 = 128;
}
int32_t x10262 = x10258 * x10261;
int32_t x10263 = 64 * x10262;
float* x10264 = (float*)myMalloc(x10263 * sizeof(float));;
int32_t x10267;
if (x10243) {
x10267 = 0;
} else {
x10267 = 1;
}
for(int x10268=0; x10268 < 64; x10268++) {
int32_t x10280 = x10122 * x10268;
int32_t x10274 = x10262 * x10268;
for(int x10270=0; x10270 < x10258; x10270++) {
int32_t x10281 = x10121 * x10270;
int32_t x10282 = x10280 + x10281;
int32_t x10287 = x10267 * x10270;
int32_t x10276 = x10261 * x10270;
for(int x10272=0; x10272 < x10260; x10272++) {
int32_t x10283 = x10265 * x10272;
int32_t x10284 = x10282 + x10283;
int32_t x10278 = x10260 * x10272;
for(int x10273=0; x10273 < x10260; x10273++) {
int32_t x10285 = x10266 * x10273;
int32_t x10286 = x10284 + x10285;
float x10288 = x10126[x10286];
float x10289 = x268[x10287];
int32_t x10275 = x10273 + x10274;
int32_t x10277 = x10275 + x10276;
int32_t x10279 = x10277 + x10278;
float x10290 = x10288 - x10289;
x10264[x10279] = x10290;

}

}

}

}
float* x10300 = (float*)myMalloc(128 * sizeof(float));;
for(int x10301=0; x10301 < 128; x10301++) {
float x10302 = x148[x10301];
float x10303 = x10302 + 1.0E-5f;
x10300[x10301] = x10303;

}
float* x10307 = (float*)myMalloc(128 * sizeof(float));;
for(int x10308=0; x10308 < 128; x10308++) {
float x10309 = x10300[x10308];
double x10310 = (double)x10309;
double x10311 = sqrt(x10310);
float x10312 = (float)x10311;
x10307[x10308] = x10312;

}
int32_t x10316 = 0;
int32_t x10317 = 1;
x10317 *= 1;
x10316 += 1;
x10317 *= 1;
x10317 *= 1;
int32_t x10322 = x10316;
bool x10323 = x10322 >= 2;
if (x10323) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10328 = x10322 == 0;
if (x10328) {
int32_t x10329 = x10317;
bool x10330 = x10329 == 128;
if (x10330) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10337 = x10317;
bool x10339 = x10258 == 1;
int32_t x10338 = 128 / x10337;
bool x10340 = x10338 == 1;
bool x10344;
if (x454) {
bool x10341 = x10339 || x10340;
bool x10342 = x10258 == x10338;
bool x10343 = x10341 || x10342;
x10344 = x10343;
} else {
x10344 = false;
}
bool x10348;
if (x10344) {
x10348 = x10347;
} else {
x10348 = false;
}
bool x10349;
if (x10348) {
x10349 = x10347;
} else {
x10349 = false;
}
if (x10349) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10258,x10260,x10260,1,x10338,1,1);
assert(false && "");
}
bool x10355 = x10258 <= x10338;
int32_t x10356;
if (x10355) {
x10356 = x10338;
} else {
x10356 = x10258;
}
int32_t x10360 = x10356 * x10359;
int32_t x10361 = 64 * x10360;
float* x10362 = (float*)myMalloc(x10361 * sizeof(float));;
int32_t x10363;
if (x10339) {
x10363 = 0;
} else {
x10363 = x10261;
}
int32_t x10366;
if (x10340) {
x10366 = 0;
} else {
x10366 = 1;
}
for(int x10367=0; x10367 < 64; x10367++) {
int32_t x10379 = x10262 * x10367;
int32_t x10373 = x10360 * x10367;
for(int x10369=0; x10369 < x10356; x10369++) {
int32_t x10380 = x10363 * x10369;
int32_t x10381 = x10379 + x10380;
int32_t x10386 = x10366 * x10369;
int32_t x10375 = x10359 * x10369;
for(int x10371=0; x10371 < x10358; x10371++) {
int32_t x10382 = x10364 * x10371;
int32_t x10383 = x10381 + x10382;
int32_t x10377 = x10358 * x10371;
for(int x10372=0; x10372 < x10358; x10372++) {
int32_t x10384 = x10365 * x10372;
int32_t x10385 = x10383 + x10384;
float x10387 = x10264[x10385];
float x10388 = x10307[x10386];
int32_t x10374 = x10372 + x10373;
int32_t x10376 = x10374 + x10375;
int32_t x10378 = x10376 + x10377;
float x10389 = x10387 / x10388;
x10362[x10378] = x10389;

}

}

}

}
int32_t x10399 = 0;
int32_t x10400 = 1;
x10400 *= 1;
x10399 += 1;
x10400 *= 1;
x10400 *= 1;
int32_t x10405 = x10399;
bool x10406 = x10405 >= 2;
if (x10406) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10411 = x10405 == 0;
if (x10411) {
int32_t x10412 = x10400;
bool x10413 = x10412 == 128;
if (x10413) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10420 = x10400;
bool x10422 = x10356 == 1;
int32_t x10421 = 128 / x10420;
bool x10423 = x10421 == 1;
bool x10427;
if (x454) {
bool x10424 = x10422 || x10423;
bool x10425 = x10356 == x10421;
bool x10426 = x10424 || x10425;
x10427 = x10426;
} else {
x10427 = false;
}
bool x10431;
if (x10427) {
x10431 = x10430;
} else {
x10431 = false;
}
bool x10432;
if (x10431) {
x10432 = x10430;
} else {
x10432 = false;
}
if (x10432) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10356,x10358,x10358,1,x10421,1,1);
assert(false && "");
}
bool x10438 = x10356 <= x10421;
int32_t x10439;
if (x10438) {
x10439 = x10421;
} else {
x10439 = x10356;
}
int32_t x10443 = x10439 * x10442;
int32_t x10444 = 64 * x10443;
float* x10445 = (float*)myMalloc(x10444 * sizeof(float));;
int32_t x10446;
if (x10422) {
x10446 = 0;
} else {
x10446 = x10359;
}
int32_t x10449;
if (x10423) {
x10449 = 0;
} else {
x10449 = 1;
}
for(int x10450=0; x10450 < 64; x10450++) {
int32_t x10462 = x10360 * x10450;
int32_t x10456 = x10443 * x10450;
for(int x10452=0; x10452 < x10439; x10452++) {
int32_t x10463 = x10446 * x10452;
int32_t x10464 = x10462 + x10463;
int32_t x10469 = x10449 * x10452;
int32_t x10458 = x10442 * x10452;
for(int x10454=0; x10454 < x10441; x10454++) {
int32_t x10465 = x10447 * x10454;
int32_t x10466 = x10464 + x10465;
int32_t x10460 = x10441 * x10454;
for(int x10455=0; x10455 < x10441; x10455++) {
int32_t x10467 = x10448 * x10455;
int32_t x10468 = x10466 + x10467;
float x10470 = x10362[x10468];
float x10471 = x79[x10469];
int32_t x10457 = x10455 + x10456;
int32_t x10459 = x10457 + x10458;
int32_t x10461 = x10459 + x10460;
float x10472 = x10470 * x10471;
x10445[x10461] = x10472;

}

}

}

}
int32_t x10482 = 0;
int32_t x10483 = 1;
x10483 *= 1;
x10482 += 1;
x10483 *= 1;
x10483 *= 1;
int32_t x10488 = x10482;
bool x10489 = x10488 >= 2;
if (x10489) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10494 = x10488 == 0;
if (x10494) {
int32_t x10495 = x10483;
bool x10496 = x10495 == 128;
if (x10496) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10503 = x10483;
bool x10505 = x10439 == 1;
int32_t x10504 = 128 / x10503;
bool x10506 = x10504 == 1;
bool x10510;
if (x454) {
bool x10507 = x10505 || x10506;
bool x10508 = x10439 == x10504;
bool x10509 = x10507 || x10508;
x10510 = x10509;
} else {
x10510 = false;
}
bool x10514;
if (x10510) {
x10514 = x10513;
} else {
x10514 = false;
}
bool x10515;
if (x10514) {
x10515 = x10513;
} else {
x10515 = false;
}
if (x10515) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10439,x10441,x10441,1,x10504,1,1);
assert(false && "");
}
bool x10521 = x10439 <= x10504;
int32_t x10522;
if (x10521) {
x10522 = x10504;
} else {
x10522 = x10439;
}
int32_t x10526 = x10522 * x10525;
int32_t x10527 = 64 * x10526;
float* x10528 = (float*)myMalloc(x10527 * sizeof(float));;
int32_t x10529;
if (x10505) {
x10529 = 0;
} else {
x10529 = x10442;
}
int32_t x10532;
if (x10506) {
x10532 = 0;
} else {
x10532 = 1;
}
for(int x10533=0; x10533 < 64; x10533++) {
int32_t x10545 = x10443 * x10533;
int32_t x10539 = x10526 * x10533;
for(int x10535=0; x10535 < x10522; x10535++) {
int32_t x10546 = x10529 * x10535;
int32_t x10547 = x10545 + x10546;
int32_t x10552 = x10532 * x10535;
int32_t x10541 = x10525 * x10535;
for(int x10537=0; x10537 < x10524; x10537++) {
int32_t x10548 = x10530 * x10537;
int32_t x10549 = x10547 + x10548;
int32_t x10543 = x10524 * x10537;
for(int x10538=0; x10538 < x10524; x10538++) {
int32_t x10550 = x10531 * x10538;
int32_t x10551 = x10549 + x10550;
float x10553 = x10445[x10551];
float x10554 = x38[x10552];
int32_t x10540 = x10538 + x10539;
int32_t x10542 = x10540 + x10541;
int32_t x10544 = x10542 + x10543;
float x10555 = x10553 + x10554;
x10528[x10544] = x10555;

}

}

}

}
float* x10565 = (float*)myMalloc(x10527 * sizeof(float));;
for(int x10567=0; x10567 < x10527; x10567++) {
float x10568 = x10528[x10567];
bool x10569 = x10568 < 0.0f;
if (x10569) {
x10565[x10567] = 0.0f;
} else {
float x10572 = x10528[x10567];
x10565[x10567] = x10572;
}

}
float* x10586 = (float*)myMalloc(x10585 * sizeof(float));;
int32_t x10589 = 64 * x10522;
int32_t x10590 = x10589 * x10581;
float* x10591 = (float*)myMalloc(x10590 * sizeof(float));;
int32_t x10587 = x10522 * x10581;
for(int x10592=0; x10592 < 64; x10592++) {
int32_t x10593 = x10592 * x10526;
float* x10594 = x10565+x10593;
int32_t x10595 = x10592 * x10582;
float* x10596 = x10586+x10595;
int32_t x10597 = x10592 * x10587;
float* x10598 = x10591+x10597;
for(int x10599=0; x10599 < x10522; x10599++) {
int32_t x10600 = x10599 / 1;
int32_t x10604 = x10600 * x10580;
int32_t x10605 = x10604 * x10580;
int32_t x10601 = x10599 % 1;
int32_t x10602 = x10601 / 1;
int32_t x10606 = x10602 * x10580;
int32_t x10607 = x10606 * x10580;
int32_t x10608 = x10605 + x10607;
int32_t x10603 = x10601 % 1;
int32_t x10609 = x10603 * x10580;
int32_t x10610 = x10609 * x10580;
int32_t x10611 = x10608 + x10610;
float* x10612 = x10598+x10611;
int32_t x10613 = x10600 * x10524;
int32_t x10614 = x10613 * x10524;
float* x10615 = x10594+x10614;
for(int x10617=0; x10617 < x10580; x10617++) {
int32_t x10619 = x10617 * x10580;
float* x10620 = x10612+x10619;
int32_t x10618 = x10617 + x10602;
int32_t x10621 = x10618 * x10524;
int32_t x10622 = x10621 + x10603;
float* x10623 = x10615+x10622;
memcpy(x10620, x10623, 4 * x10580);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x10581,x10522,1,x55,x10522,x10598,x10581,1,x10596,x10581);

}
int32_t x10632 = 0;
int32_t x10633 = 1;
x10633 *= 1;
x10632 += 1;
x10633 *= 1;
x10633 *= 1;
int32_t x10638 = x10632;
bool x10639 = x10638 >= 2;
if (x10639) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10644 = x10638 == 0;
if (x10644) {
int32_t x10645 = x10633;
bool x10646 = x10645 == 512;
if (x10646) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10653 = x10633;
int32_t x10654 = 512 / x10653;
bool x10655 = x10654 == 1;
bool x10658;
if (x454) {
bool x10656 = 512 == x10654;
bool x10657 = x10655 || x10656;
x10658 = x10657;
} else {
x10658 = false;
}
bool x10662;
if (x10658) {
x10662 = x10661;
} else {
x10662 = false;
}
bool x10663;
if (x10662) {
x10663 = x10661;
} else {
x10663 = false;
}
if (x10663) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x10580,x10580,1,x10654,1,1);
assert(false && "");
}
bool x10669 = 512 <= x10654;
int32_t x10670;
if (x10669) {
x10670 = x10654;
} else {
x10670 = 512;
}
int32_t x10674 = x10670 * x10673;
int32_t x10675 = 64 * x10674;
float* x10676 = (float*)myMalloc(x10675 * sizeof(float));;
int32_t x10679;
if (x10655) {
x10679 = 0;
} else {
x10679 = 1;
}
for(int x10680=0; x10680 < 64; x10680++) {
int32_t x10692 = x10582 * x10680;
int32_t x10686 = x10674 * x10680;
for(int x10682=0; x10682 < x10670; x10682++) {
int32_t x10693 = x10581 * x10682;
int32_t x10694 = x10692 + x10693;
int32_t x10699 = x10679 * x10682;
int32_t x10688 = x10673 * x10682;
for(int x10684=0; x10684 < x10672; x10684++) {
int32_t x10695 = x10677 * x10684;
int32_t x10696 = x10694 + x10695;
int32_t x10690 = x10672 * x10684;
for(int x10685=0; x10685 < x10672; x10685++) {
int32_t x10697 = x10678 * x10685;
int32_t x10698 = x10696 + x10697;
float x10700 = x10586[x10698];
float x10701 = x19[x10699];
int32_t x10687 = x10685 + x10686;
int32_t x10689 = x10687 + x10688;
int32_t x10691 = x10689 + x10690;
float x10702 = x10700 - x10701;
x10676[x10691] = x10702;

}

}

}

}
float* x10712 = (float*)myMalloc(512 * sizeof(float));;
for(int x10713=0; x10713 < 512; x10713++) {
float x10714 = x234[x10713];
float x10715 = x10714 + 1.0E-5f;
x10712[x10713] = x10715;

}
float* x10719 = (float*)myMalloc(512 * sizeof(float));;
for(int x10720=0; x10720 < 512; x10720++) {
float x10721 = x10712[x10720];
double x10722 = (double)x10721;
double x10723 = sqrt(x10722);
float x10724 = (float)x10723;
x10719[x10720] = x10724;

}
int32_t x10728 = 0;
int32_t x10729 = 1;
x10729 *= 1;
x10728 += 1;
x10729 *= 1;
x10729 *= 1;
int32_t x10734 = x10728;
bool x10735 = x10734 >= 2;
if (x10735) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10740 = x10734 == 0;
if (x10740) {
int32_t x10741 = x10729;
bool x10742 = x10741 == 512;
if (x10742) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10749 = x10729;
bool x10751 = x10670 == 1;
int32_t x10750 = 512 / x10749;
bool x10752 = x10750 == 1;
bool x10756;
if (x454) {
bool x10753 = x10751 || x10752;
bool x10754 = x10670 == x10750;
bool x10755 = x10753 || x10754;
x10756 = x10755;
} else {
x10756 = false;
}
bool x10760;
if (x10756) {
x10760 = x10759;
} else {
x10760 = false;
}
bool x10761;
if (x10760) {
x10761 = x10759;
} else {
x10761 = false;
}
if (x10761) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10670,x10672,x10672,1,x10750,1,1);
assert(false && "");
}
bool x10767 = x10670 <= x10750;
int32_t x10768;
if (x10767) {
x10768 = x10750;
} else {
x10768 = x10670;
}
int32_t x10772 = x10768 * x10771;
int32_t x10773 = 64 * x10772;
float* x10774 = (float*)myMalloc(x10773 * sizeof(float));;
int32_t x10775;
if (x10751) {
x10775 = 0;
} else {
x10775 = x10673;
}
int32_t x10778;
if (x10752) {
x10778 = 0;
} else {
x10778 = 1;
}
for(int x10779=0; x10779 < 64; x10779++) {
int32_t x10791 = x10674 * x10779;
int32_t x10785 = x10772 * x10779;
for(int x10781=0; x10781 < x10768; x10781++) {
int32_t x10792 = x10775 * x10781;
int32_t x10793 = x10791 + x10792;
int32_t x10798 = x10778 * x10781;
int32_t x10787 = x10771 * x10781;
for(int x10783=0; x10783 < x10770; x10783++) {
int32_t x10794 = x10776 * x10783;
int32_t x10795 = x10793 + x10794;
int32_t x10789 = x10770 * x10783;
for(int x10784=0; x10784 < x10770; x10784++) {
int32_t x10796 = x10777 * x10784;
int32_t x10797 = x10795 + x10796;
float x10799 = x10676[x10797];
float x10800 = x10719[x10798];
int32_t x10786 = x10784 + x10785;
int32_t x10788 = x10786 + x10787;
int32_t x10790 = x10788 + x10789;
float x10801 = x10799 / x10800;
x10774[x10790] = x10801;

}

}

}

}
int32_t x10811 = 0;
int32_t x10812 = 1;
x10812 *= 1;
x10811 += 1;
x10812 *= 1;
x10812 *= 1;
int32_t x10817 = x10811;
bool x10818 = x10817 >= 2;
if (x10818) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10823 = x10817 == 0;
if (x10823) {
int32_t x10824 = x10812;
bool x10825 = x10824 == 512;
if (x10825) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10832 = x10812;
bool x10834 = x10768 == 1;
int32_t x10833 = 512 / x10832;
bool x10835 = x10833 == 1;
bool x10839;
if (x454) {
bool x10836 = x10834 || x10835;
bool x10837 = x10768 == x10833;
bool x10838 = x10836 || x10837;
x10839 = x10838;
} else {
x10839 = false;
}
bool x10843;
if (x10839) {
x10843 = x10842;
} else {
x10843 = false;
}
bool x10844;
if (x10843) {
x10844 = x10842;
} else {
x10844 = false;
}
if (x10844) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10768,x10770,x10770,1,x10833,1,1);
assert(false && "");
}
bool x10850 = x10768 <= x10833;
int32_t x10851;
if (x10850) {
x10851 = x10833;
} else {
x10851 = x10768;
}
int32_t x10855 = x10851 * x10854;
int32_t x10856 = 64 * x10855;
float* x10857 = (float*)myMalloc(x10856 * sizeof(float));;
int32_t x10858;
if (x10834) {
x10858 = 0;
} else {
x10858 = x10771;
}
int32_t x10861;
if (x10835) {
x10861 = 0;
} else {
x10861 = 1;
}
for(int x10862=0; x10862 < 64; x10862++) {
int32_t x10874 = x10772 * x10862;
int32_t x10868 = x10855 * x10862;
for(int x10864=0; x10864 < x10851; x10864++) {
int32_t x10875 = x10858 * x10864;
int32_t x10876 = x10874 + x10875;
int32_t x10881 = x10861 * x10864;
int32_t x10870 = x10854 * x10864;
for(int x10866=0; x10866 < x10853; x10866++) {
int32_t x10877 = x10859 * x10866;
int32_t x10878 = x10876 + x10877;
int32_t x10872 = x10853 * x10866;
for(int x10867=0; x10867 < x10853; x10867++) {
int32_t x10879 = x10860 * x10867;
int32_t x10880 = x10878 + x10879;
float x10882 = x10774[x10880];
float x10883 = x156[x10881];
int32_t x10869 = x10867 + x10868;
int32_t x10871 = x10869 + x10870;
int32_t x10873 = x10871 + x10872;
float x10884 = x10882 * x10883;
x10857[x10873] = x10884;

}

}

}

}
int32_t x10894 = 0;
int32_t x10895 = 1;
x10895 *= 1;
x10894 += 1;
x10895 *= 1;
x10895 *= 1;
int32_t x10900 = x10894;
bool x10901 = x10900 >= 2;
if (x10901) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10906 = x10900 == 0;
if (x10906) {
int32_t x10907 = x10895;
bool x10908 = x10907 == 512;
if (x10908) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10915 = x10895;
bool x10917 = x10851 == 1;
int32_t x10916 = 512 / x10915;
bool x10918 = x10916 == 1;
bool x10922;
if (x454) {
bool x10919 = x10917 || x10918;
bool x10920 = x10851 == x10916;
bool x10921 = x10919 || x10920;
x10922 = x10921;
} else {
x10922 = false;
}
bool x10926;
if (x10922) {
x10926 = x10925;
} else {
x10926 = false;
}
bool x10927;
if (x10926) {
x10927 = x10925;
} else {
x10927 = false;
}
if (x10927) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10851,x10853,x10853,1,x10916,1,1);
assert(false && "");
}
bool x10933 = x10851 <= x10916;
int32_t x10934;
if (x10933) {
x10934 = x10916;
} else {
x10934 = x10851;
}
int32_t x10938 = x10934 * x10937;
int32_t x10939 = 64 * x10938;
float* x10940 = (float*)myMalloc(x10939 * sizeof(float));;
int32_t x10941;
if (x10917) {
x10941 = 0;
} else {
x10941 = x10854;
}
int32_t x10944;
if (x10918) {
x10944 = 0;
} else {
x10944 = 1;
}
for(int x10945=0; x10945 < 64; x10945++) {
int32_t x10957 = x10855 * x10945;
int32_t x10951 = x10938 * x10945;
for(int x10947=0; x10947 < x10934; x10947++) {
int32_t x10958 = x10941 * x10947;
int32_t x10959 = x10957 + x10958;
int32_t x10964 = x10944 * x10947;
int32_t x10953 = x10937 * x10947;
for(int x10949=0; x10949 < x10936; x10949++) {
int32_t x10960 = x10942 * x10949;
int32_t x10961 = x10959 + x10960;
int32_t x10955 = x10936 * x10949;
for(int x10950=0; x10950 < x10936; x10950++) {
int32_t x10962 = x10943 * x10950;
int32_t x10963 = x10961 + x10962;
float x10965 = x10857[x10963];
float x10966 = x54[x10964];
int32_t x10952 = x10950 + x10951;
int32_t x10954 = x10952 + x10953;
int32_t x10956 = x10954 + x10955;
float x10967 = x10965 + x10966;
x10940[x10956] = x10967;

}

}

}

}
bool x10977 = x10934 == 1;
bool x10978 = x10977 || x9635;
bool x10979 = x10934 == x9592;
bool x10980 = x10978 || x10979;
bool x10985;
if (x10980) {
x10985 = x10984;
} else {
x10985 = false;
}
bool x10986;
if (x10985) {
x10986 = x10984;
} else {
x10986 = false;
}
if (x10986) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10934,x10936,x10936,64,x9592,x9594,x9594);
assert(false && "");
}
bool x10992 = x10934 <= x9592;
int32_t x10993;
if (x10992) {
x10993 = x9592;
} else {
x10993 = x10934;
}
int32_t x10999;
if (x10977) {
x10999 = 0;
} else {
x10999 = x10937;
}
for(int x11002=0; x11002 < 64; x11002++) {
int32_t x11008 = x10938 * x11002;
int32_t x11015 = x9596 * x11002;
for(int x11004=0; x11004 < x10993; x11004++) {
int32_t x11009 = x10999 * x11004;
int32_t x11010 = x11008 + x11009;
int32_t x11016 = x9657 * x11004;
int32_t x11017 = x11015 + x11016;
for(int x11006=0; x11006 < x10995; x11006++) {
int32_t x11011 = x11000 * x11006;
int32_t x11012 = x11010 + x11011;
int32_t x11018 = x9658 * x11006;
int32_t x11019 = x11017 + x11018;
for(int x11007=0; x11007 < x10995; x11007++) {
int32_t x11013 = x11001 * x11007;
int32_t x11014 = x11012 + x11013;
float x11022 = x10940[x11014];
int32_t x11020 = x9659 * x11007;
int32_t x11021 = x11019 + x11020;
float x11023 = x9692[x11021];
float x11024 = x11022 + x11023;
x10940[x11014] = x11024;

}

}

}

}
float* x11034 = (float*)myMalloc(x10939 * sizeof(float));;
for(int x11036=0; x11036 < x10939; x11036++) {
float x11037 = x10940[x11036];
bool x11038 = x11037 < 0.0f;
if (x11038) {
x11034[x11036] = 0.0f;
} else {
float x11041 = x10940[x11036];
x11034[x11036] = x11041;
}

}
float* x11055 = (float*)myMalloc(x11054 * sizeof(float));;
int32_t x11058 = 64 * x10934;
int32_t x11059 = x11058 * x11050;
float* x11060 = (float*)myMalloc(x11059 * sizeof(float));;
int32_t x11056 = x10934 * x11050;
for(int x11061=0; x11061 < 64; x11061++) {
int32_t x11062 = x11061 * x10938;
float* x11063 = x11034+x11062;
int32_t x11064 = x11061 * x11051;
float* x11065 = x11055+x11064;
int32_t x11066 = x11061 * x11056;
float* x11067 = x11060+x11066;
for(int x11068=0; x11068 < x10934; x11068++) {
int32_t x11069 = x11068 / 1;
int32_t x11073 = x11069 * x11049;
int32_t x11074 = x11073 * x11049;
int32_t x11070 = x11068 % 1;
int32_t x11071 = x11070 / 1;
int32_t x11075 = x11071 * x11049;
int32_t x11076 = x11075 * x11049;
int32_t x11077 = x11074 + x11076;
int32_t x11072 = x11070 % 1;
int32_t x11078 = x11072 * x11049;
int32_t x11079 = x11078 * x11049;
int32_t x11080 = x11077 + x11079;
float* x11081 = x11067+x11080;
int32_t x11082 = x11069 * x10936;
int32_t x11083 = x11082 * x10936;
float* x11084 = x11063+x11083;
for(int x11086=0; x11086 < x11049; x11086++) {
int32_t x11088 = x11086 * x11049;
float* x11089 = x11081+x11088;
int32_t x11087 = x11086 + x11071;
int32_t x11090 = x11087 * x10936;
int32_t x11091 = x11090 + x11072;
float* x11092 = x11084+x11091;
memcpy(x11089, x11092, 4 * x11049);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x11050,x10934,1,x180,x10934,x11067,x11050,1,x11065,x11050);

}
int32_t x11101 = 0;
int32_t x11102 = 1;
x11102 *= 1;
x11101 += 1;
x11102 *= 1;
x11102 *= 1;
int32_t x11107 = x11101;
bool x11108 = x11107 >= 2;
if (x11108) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11113 = x11107 == 0;
if (x11113) {
int32_t x11114 = x11102;
bool x11115 = x11114 == 256;
if (x11115) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11122 = x11102;
int32_t x11123 = 256 / x11122;
bool x11124 = x11123 == 1;
bool x11127;
if (x454) {
bool x11125 = 256 == x11123;
bool x11126 = x11124 || x11125;
x11127 = x11126;
} else {
x11127 = false;
}
bool x11131;
if (x11127) {
x11131 = x11130;
} else {
x11131 = false;
}
bool x11132;
if (x11131) {
x11132 = x11130;
} else {
x11132 = false;
}
if (x11132) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x11049,x11049,1,x11123,1,1);
assert(false && "");
}
bool x11138 = 256 <= x11123;
int32_t x11139;
if (x11138) {
x11139 = x11123;
} else {
x11139 = 256;
}
int32_t x11143 = x11139 * x11142;
int32_t x11144 = 64 * x11143;
float* x11145 = (float*)myMalloc(x11144 * sizeof(float));;
int32_t x11148;
if (x11124) {
x11148 = 0;
} else {
x11148 = 1;
}
for(int x11149=0; x11149 < 64; x11149++) {
int32_t x11161 = x11051 * x11149;
int32_t x11155 = x11143 * x11149;
for(int x11151=0; x11151 < x11139; x11151++) {
int32_t x11162 = x11050 * x11151;
int32_t x11163 = x11161 + x11162;
int32_t x11168 = x11148 * x11151;
int32_t x11157 = x11142 * x11151;
for(int x11153=0; x11153 < x11141; x11153++) {
int32_t x11164 = x11146 * x11153;
int32_t x11165 = x11163 + x11164;
int32_t x11159 = x11141 * x11153;
for(int x11154=0; x11154 < x11141; x11154++) {
int32_t x11166 = x11147 * x11154;
int32_t x11167 = x11165 + x11166;
float x11169 = x11055[x11167];
float x11170 = x131[x11168];
int32_t x11156 = x11154 + x11155;
int32_t x11158 = x11156 + x11157;
int32_t x11160 = x11158 + x11159;
float x11171 = x11169 - x11170;
x11145[x11160] = x11171;

}

}

}

}
float* x11181 = (float*)myMalloc(256 * sizeof(float));;
for(int x11182=0; x11182 < 256; x11182++) {
float x11183 = x198[x11182];
float x11184 = x11183 + 1.0E-5f;
x11181[x11182] = x11184;

}
float* x11188 = (float*)myMalloc(256 * sizeof(float));;
for(int x11189=0; x11189 < 256; x11189++) {
float x11190 = x11181[x11189];
double x11191 = (double)x11190;
double x11192 = sqrt(x11191);
float x11193 = (float)x11192;
x11188[x11189] = x11193;

}
int32_t x11197 = 0;
int32_t x11198 = 1;
x11198 *= 1;
x11197 += 1;
x11198 *= 1;
x11198 *= 1;
int32_t x11203 = x11197;
bool x11204 = x11203 >= 2;
if (x11204) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11209 = x11203 == 0;
if (x11209) {
int32_t x11210 = x11198;
bool x11211 = x11210 == 256;
if (x11211) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11218 = x11198;
bool x11220 = x11139 == 1;
int32_t x11219 = 256 / x11218;
bool x11221 = x11219 == 1;
bool x11225;
if (x454) {
bool x11222 = x11220 || x11221;
bool x11223 = x11139 == x11219;
bool x11224 = x11222 || x11223;
x11225 = x11224;
} else {
x11225 = false;
}
bool x11229;
if (x11225) {
x11229 = x11228;
} else {
x11229 = false;
}
bool x11230;
if (x11229) {
x11230 = x11228;
} else {
x11230 = false;
}
if (x11230) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11139,x11141,x11141,1,x11219,1,1);
assert(false && "");
}
bool x11236 = x11139 <= x11219;
int32_t x11237;
if (x11236) {
x11237 = x11219;
} else {
x11237 = x11139;
}
int32_t x11241 = x11237 * x11240;
int32_t x11242 = 64 * x11241;
float* x11243 = (float*)myMalloc(x11242 * sizeof(float));;
int32_t x11244;
if (x11220) {
x11244 = 0;
} else {
x11244 = x11142;
}
int32_t x11247;
if (x11221) {
x11247 = 0;
} else {
x11247 = 1;
}
for(int x11248=0; x11248 < 64; x11248++) {
int32_t x11260 = x11143 * x11248;
int32_t x11254 = x11241 * x11248;
for(int x11250=0; x11250 < x11237; x11250++) {
int32_t x11261 = x11244 * x11250;
int32_t x11262 = x11260 + x11261;
int32_t x11267 = x11247 * x11250;
int32_t x11256 = x11240 * x11250;
for(int x11252=0; x11252 < x11239; x11252++) {
int32_t x11263 = x11245 * x11252;
int32_t x11264 = x11262 + x11263;
int32_t x11258 = x11239 * x11252;
for(int x11253=0; x11253 < x11239; x11253++) {
int32_t x11265 = x11246 * x11253;
int32_t x11266 = x11264 + x11265;
float x11268 = x11145[x11266];
float x11269 = x11188[x11267];
int32_t x11255 = x11253 + x11254;
int32_t x11257 = x11255 + x11256;
int32_t x11259 = x11257 + x11258;
float x11270 = x11268 / x11269;
x11243[x11259] = x11270;

}

}

}

}
int32_t x11280 = 0;
int32_t x11281 = 1;
x11281 *= 1;
x11280 += 1;
x11281 *= 1;
x11281 *= 1;
int32_t x11286 = x11280;
bool x11287 = x11286 >= 2;
if (x11287) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11292 = x11286 == 0;
if (x11292) {
int32_t x11293 = x11281;
bool x11294 = x11293 == 256;
if (x11294) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11301 = x11281;
bool x11303 = x11237 == 1;
int32_t x11302 = 256 / x11301;
bool x11304 = x11302 == 1;
bool x11308;
if (x454) {
bool x11305 = x11303 || x11304;
bool x11306 = x11237 == x11302;
bool x11307 = x11305 || x11306;
x11308 = x11307;
} else {
x11308 = false;
}
bool x11312;
if (x11308) {
x11312 = x11311;
} else {
x11312 = false;
}
bool x11313;
if (x11312) {
x11313 = x11311;
} else {
x11313 = false;
}
if (x11313) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11237,x11239,x11239,1,x11302,1,1);
assert(false && "");
}
bool x11319 = x11237 <= x11302;
int32_t x11320;
if (x11319) {
x11320 = x11302;
} else {
x11320 = x11237;
}
int32_t x11324 = x11320 * x11323;
int32_t x11325 = 64 * x11324;
float* x11326 = (float*)myMalloc(x11325 * sizeof(float));;
int32_t x11327;
if (x11303) {
x11327 = 0;
} else {
x11327 = x11240;
}
int32_t x11330;
if (x11304) {
x11330 = 0;
} else {
x11330 = 1;
}
for(int x11331=0; x11331 < 64; x11331++) {
int32_t x11343 = x11241 * x11331;
int32_t x11337 = x11324 * x11331;
for(int x11333=0; x11333 < x11320; x11333++) {
int32_t x11344 = x11327 * x11333;
int32_t x11345 = x11343 + x11344;
int32_t x11350 = x11330 * x11333;
int32_t x11339 = x11323 * x11333;
for(int x11335=0; x11335 < x11322; x11335++) {
int32_t x11346 = x11328 * x11335;
int32_t x11347 = x11345 + x11346;
int32_t x11341 = x11322 * x11335;
for(int x11336=0; x11336 < x11322; x11336++) {
int32_t x11348 = x11329 * x11336;
int32_t x11349 = x11347 + x11348;
float x11351 = x11243[x11349];
float x11352 = x270[x11350];
int32_t x11338 = x11336 + x11337;
int32_t x11340 = x11338 + x11339;
int32_t x11342 = x11340 + x11341;
float x11353 = x11351 * x11352;
x11326[x11342] = x11353;

}

}

}

}
int32_t x11363 = 0;
int32_t x11364 = 1;
x11364 *= 1;
x11363 += 1;
x11364 *= 1;
x11364 *= 1;
int32_t x11369 = x11363;
bool x11370 = x11369 >= 2;
if (x11370) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11375 = x11369 == 0;
if (x11375) {
int32_t x11376 = x11364;
bool x11377 = x11376 == 256;
if (x11377) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11384 = x11364;
bool x11386 = x11320 == 1;
int32_t x11385 = 256 / x11384;
bool x11387 = x11385 == 1;
bool x11391;
if (x454) {
bool x11388 = x11386 || x11387;
bool x11389 = x11320 == x11385;
bool x11390 = x11388 || x11389;
x11391 = x11390;
} else {
x11391 = false;
}
bool x11395;
if (x11391) {
x11395 = x11394;
} else {
x11395 = false;
}
bool x11396;
if (x11395) {
x11396 = x11394;
} else {
x11396 = false;
}
if (x11396) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11320,x11322,x11322,1,x11385,1,1);
assert(false && "");
}
bool x11402 = x11320 <= x11385;
int32_t x11403;
if (x11402) {
x11403 = x11385;
} else {
x11403 = x11320;
}
int32_t x11407 = x11403 * x11406;
int32_t x11408 = 64 * x11407;
float* x11409 = (float*)myMalloc(x11408 * sizeof(float));;
int32_t x11410;
if (x11386) {
x11410 = 0;
} else {
x11410 = x11323;
}
int32_t x11413;
if (x11387) {
x11413 = 0;
} else {
x11413 = 1;
}
for(int x11414=0; x11414 < 64; x11414++) {
int32_t x11426 = x11324 * x11414;
int32_t x11420 = x11407 * x11414;
for(int x11416=0; x11416 < x11403; x11416++) {
int32_t x11427 = x11410 * x11416;
int32_t x11428 = x11426 + x11427;
int32_t x11433 = x11413 * x11416;
int32_t x11422 = x11406 * x11416;
for(int x11418=0; x11418 < x11405; x11418++) {
int32_t x11429 = x11411 * x11418;
int32_t x11430 = x11428 + x11429;
int32_t x11424 = x11405 * x11418;
for(int x11419=0; x11419 < x11405; x11419++) {
int32_t x11431 = x11412 * x11419;
int32_t x11432 = x11430 + x11431;
float x11434 = x11326[x11432];
float x11435 = x21[x11433];
int32_t x11421 = x11419 + x11420;
int32_t x11423 = x11421 + x11422;
int32_t x11425 = x11423 + x11424;
float x11436 = x11434 + x11435;
x11409[x11425] = x11436;

}

}

}

}
float* x11446 = (float*)myMalloc(x11408 * sizeof(float));;
for(int x11448=0; x11448 < x11408; x11448++) {
float x11449 = x11409[x11448];
bool x11450 = x11449 < 0.0f;
if (x11450) {
x11446[x11448] = 0.0f;
} else {
float x11453 = x11409[x11448];
x11446[x11448] = x11453;
}

}
float* x11468 = (float*)myMalloc(x11467 * sizeof(float));;
int32_t x11469 = 9 * x11403;
int32_t x11472 = 64 * x11469;
int32_t x11473 = x11472 * x11463;
float* x11474 = (float*)myMalloc(x11473 * sizeof(float));;
int32_t x11470 = x11469 * x11463;
int32_t x11482 = x11403 * 3;
int32_t x11483 = x11482 * 3;
for(int x11475=0; x11475 < 64; x11475++) {
int32_t x11476 = x11475 * x11407;
float* x11477 = x11446+x11476;
int32_t x11478 = x11475 * x11464;
float* x11479 = x11468+x11478;
int32_t x11480 = x11475 * x11470;
float* x11481 = x11474+x11480;
for(int x11485=0; x11485 < x11483; x11485++) {
int32_t x11486 = x11485 / 9;
int32_t x11490 = x11486 * 3;
int32_t x11491 = x11490 * 3;
int32_t x11492 = x11491 * x11462;
int32_t x11493 = x11492 * x11462;
int32_t x11487 = x11485 % 9;
int32_t x11488 = x11487 / 3;
int32_t x11494 = x11488 * 3;
int32_t x11495 = x11494 * x11462;
int32_t x11496 = x11495 * x11462;
int32_t x11497 = x11493 + x11496;
int32_t x11489 = x11487 % 3;
int32_t x11498 = x11489 * x11462;
int32_t x11499 = x11498 * x11462;
int32_t x11500 = x11497 + x11499;
float* x11501 = x11481+x11500;
int32_t x11502 = x11486 * x11405;
int32_t x11503 = x11502 * x11405;
float* x11504 = x11477+x11503;
for(int x11506=0; x11506 < x11462; x11506++) {
int32_t x11507 = x11506 * 2;
int32_t x11508 = x11507 - 1;
int32_t x11509 = x11508 + x11488;
bool x11510 = x11509 < 0;
bool x11511 = x11509 >= x11405;
bool x11512 = x11510 || x11511;
if (x11512) {
int32_t x11513 = x11506 * x11462;
float* x11514 = x11501+x11513;
memset(x11514, 0, 4 * x11462);;
} else {
int32_t x11513 = x11506 * x11462;
int32_t x11529 = x11509 * x11405;
for(int x11517=0; x11517 < x11462; x11517++) {
int32_t x11518 = x11517 * 2;
int32_t x11519 = x11518 - 1;
int32_t x11520 = x11519 + x11489;
bool x11521 = x11520 < 0;
bool x11522 = x11520 >= x11405;
bool x11523 = x11521 || x11522;
if (x11523) {
int32_t x11524 = x11513 + x11517;
float* x11525 = x11501+x11524;
memset(x11525, 0, 4 * 1);;
} else {
int32_t x11524 = x11513 + x11517;
float* x11528 = x11501+x11524;
int32_t x11530 = x11529 + x11520;
float* x11531 = x11504+x11530;
memcpy(x11528, x11531, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x11463,x11469,1,x175,x11469,x11481,x11463,1,x11479,x11463);

}
int32_t x11546 = 0;
int32_t x11547 = 1;
x11547 *= 1;
x11546 += 1;
x11547 *= 1;
x11547 *= 1;
int32_t x11552 = x11546;
bool x11553 = x11552 >= 2;
if (x11553) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11558 = x11552 == 0;
if (x11558) {
int32_t x11559 = x11547;
bool x11560 = x11559 == 256;
if (x11560) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11567 = x11547;
int32_t x11568 = 256 / x11567;
bool x11569 = x11568 == 1;
bool x11572;
if (x454) {
bool x11570 = 256 == x11568;
bool x11571 = x11569 || x11570;
x11572 = x11571;
} else {
x11572 = false;
}
bool x11576;
if (x11572) {
x11576 = x11575;
} else {
x11576 = false;
}
bool x11577;
if (x11576) {
x11577 = x11575;
} else {
x11577 = false;
}
if (x11577) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x11462,x11462,1,x11568,1,1);
assert(false && "");
}
bool x11583 = 256 <= x11568;
int32_t x11584;
if (x11583) {
x11584 = x11568;
} else {
x11584 = 256;
}
int32_t x11588 = x11584 * x11587;
int32_t x11589 = 64 * x11588;
float* x11590 = (float*)myMalloc(x11589 * sizeof(float));;
int32_t x11593;
if (x11569) {
x11593 = 0;
} else {
x11593 = 1;
}
for(int x11594=0; x11594 < 64; x11594++) {
int32_t x11606 = x11464 * x11594;
int32_t x11600 = x11588 * x11594;
for(int x11596=0; x11596 < x11584; x11596++) {
int32_t x11607 = x11463 * x11596;
int32_t x11608 = x11606 + x11607;
int32_t x11613 = x11593 * x11596;
int32_t x11602 = x11587 * x11596;
for(int x11598=0; x11598 < x11586; x11598++) {
int32_t x11609 = x11591 * x11598;
int32_t x11610 = x11608 + x11609;
int32_t x11604 = x11586 * x11598;
for(int x11599=0; x11599 < x11586; x11599++) {
int32_t x11611 = x11592 * x11599;
int32_t x11612 = x11610 + x11611;
float x11614 = x11468[x11612];
float x11615 = x229[x11613];
int32_t x11601 = x11599 + x11600;
int32_t x11603 = x11601 + x11602;
int32_t x11605 = x11603 + x11604;
float x11616 = x11614 - x11615;
x11590[x11605] = x11616;

}

}

}

}
float* x11626 = (float*)myMalloc(256 * sizeof(float));;
for(int x11627=0; x11627 < 256; x11627++) {
float x11628 = x99[x11627];
float x11629 = x11628 + 1.0E-5f;
x11626[x11627] = x11629;

}
float* x11633 = (float*)myMalloc(256 * sizeof(float));;
for(int x11634=0; x11634 < 256; x11634++) {
float x11635 = x11626[x11634];
double x11636 = (double)x11635;
double x11637 = sqrt(x11636);
float x11638 = (float)x11637;
x11633[x11634] = x11638;

}
int32_t x11642 = 0;
int32_t x11643 = 1;
x11643 *= 1;
x11642 += 1;
x11643 *= 1;
x11643 *= 1;
int32_t x11648 = x11642;
bool x11649 = x11648 >= 2;
if (x11649) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11654 = x11648 == 0;
if (x11654) {
int32_t x11655 = x11643;
bool x11656 = x11655 == 256;
if (x11656) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11663 = x11643;
bool x11665 = x11584 == 1;
int32_t x11664 = 256 / x11663;
bool x11666 = x11664 == 1;
bool x11670;
if (x454) {
bool x11667 = x11665 || x11666;
bool x11668 = x11584 == x11664;
bool x11669 = x11667 || x11668;
x11670 = x11669;
} else {
x11670 = false;
}
bool x11674;
if (x11670) {
x11674 = x11673;
} else {
x11674 = false;
}
bool x11675;
if (x11674) {
x11675 = x11673;
} else {
x11675 = false;
}
if (x11675) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11584,x11586,x11586,1,x11664,1,1);
assert(false && "");
}
bool x11681 = x11584 <= x11664;
int32_t x11682;
if (x11681) {
x11682 = x11664;
} else {
x11682 = x11584;
}
int32_t x11686 = x11682 * x11685;
int32_t x11687 = 64 * x11686;
float* x11688 = (float*)myMalloc(x11687 * sizeof(float));;
int32_t x11689;
if (x11665) {
x11689 = 0;
} else {
x11689 = x11587;
}
int32_t x11692;
if (x11666) {
x11692 = 0;
} else {
x11692 = 1;
}
for(int x11693=0; x11693 < 64; x11693++) {
int32_t x11705 = x11588 * x11693;
int32_t x11699 = x11686 * x11693;
for(int x11695=0; x11695 < x11682; x11695++) {
int32_t x11706 = x11689 * x11695;
int32_t x11707 = x11705 + x11706;
int32_t x11712 = x11692 * x11695;
int32_t x11701 = x11685 * x11695;
for(int x11697=0; x11697 < x11684; x11697++) {
int32_t x11708 = x11690 * x11697;
int32_t x11709 = x11707 + x11708;
int32_t x11703 = x11684 * x11697;
for(int x11698=0; x11698 < x11684; x11698++) {
int32_t x11710 = x11691 * x11698;
int32_t x11711 = x11709 + x11710;
float x11713 = x11590[x11711];
float x11714 = x11633[x11712];
int32_t x11700 = x11698 + x11699;
int32_t x11702 = x11700 + x11701;
int32_t x11704 = x11702 + x11703;
float x11715 = x11713 / x11714;
x11688[x11704] = x11715;

}

}

}

}
int32_t x11725 = 0;
int32_t x11726 = 1;
x11726 *= 1;
x11725 += 1;
x11726 *= 1;
x11726 *= 1;
int32_t x11731 = x11725;
bool x11732 = x11731 >= 2;
if (x11732) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11737 = x11731 == 0;
if (x11737) {
int32_t x11738 = x11726;
bool x11739 = x11738 == 256;
if (x11739) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11746 = x11726;
bool x11748 = x11682 == 1;
int32_t x11747 = 256 / x11746;
bool x11749 = x11747 == 1;
bool x11753;
if (x454) {
bool x11750 = x11748 || x11749;
bool x11751 = x11682 == x11747;
bool x11752 = x11750 || x11751;
x11753 = x11752;
} else {
x11753 = false;
}
bool x11757;
if (x11753) {
x11757 = x11756;
} else {
x11757 = false;
}
bool x11758;
if (x11757) {
x11758 = x11756;
} else {
x11758 = false;
}
if (x11758) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11682,x11684,x11684,1,x11747,1,1);
assert(false && "");
}
bool x11764 = x11682 <= x11747;
int32_t x11765;
if (x11764) {
x11765 = x11747;
} else {
x11765 = x11682;
}
int32_t x11769 = x11765 * x11768;
int32_t x11770 = 64 * x11769;
float* x11771 = (float*)myMalloc(x11770 * sizeof(float));;
int32_t x11772;
if (x11748) {
x11772 = 0;
} else {
x11772 = x11685;
}
int32_t x11775;
if (x11749) {
x11775 = 0;
} else {
x11775 = 1;
}
for(int x11776=0; x11776 < 64; x11776++) {
int32_t x11788 = x11686 * x11776;
int32_t x11782 = x11769 * x11776;
for(int x11778=0; x11778 < x11765; x11778++) {
int32_t x11789 = x11772 * x11778;
int32_t x11790 = x11788 + x11789;
int32_t x11795 = x11775 * x11778;
int32_t x11784 = x11768 * x11778;
for(int x11780=0; x11780 < x11767; x11780++) {
int32_t x11791 = x11773 * x11780;
int32_t x11792 = x11790 + x11791;
int32_t x11786 = x11767 * x11780;
for(int x11781=0; x11781 < x11767; x11781++) {
int32_t x11793 = x11774 * x11781;
int32_t x11794 = x11792 + x11793;
float x11796 = x11688[x11794];
float x11797 = x108[x11795];
int32_t x11783 = x11781 + x11782;
int32_t x11785 = x11783 + x11784;
int32_t x11787 = x11785 + x11786;
float x11798 = x11796 * x11797;
x11771[x11787] = x11798;

}

}

}

}
int32_t x11808 = 0;
int32_t x11809 = 1;
x11809 *= 1;
x11808 += 1;
x11809 *= 1;
x11809 *= 1;
int32_t x11814 = x11808;
bool x11815 = x11814 >= 2;
if (x11815) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11820 = x11814 == 0;
if (x11820) {
int32_t x11821 = x11809;
bool x11822 = x11821 == 256;
if (x11822) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11829 = x11809;
bool x11831 = x11765 == 1;
int32_t x11830 = 256 / x11829;
bool x11832 = x11830 == 1;
bool x11836;
if (x454) {
bool x11833 = x11831 || x11832;
bool x11834 = x11765 == x11830;
bool x11835 = x11833 || x11834;
x11836 = x11835;
} else {
x11836 = false;
}
bool x11840;
if (x11836) {
x11840 = x11839;
} else {
x11840 = false;
}
bool x11841;
if (x11840) {
x11841 = x11839;
} else {
x11841 = false;
}
if (x11841) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11765,x11767,x11767,1,x11830,1,1);
assert(false && "");
}
bool x11847 = x11765 <= x11830;
int32_t x11848;
if (x11847) {
x11848 = x11830;
} else {
x11848 = x11765;
}
int32_t x11852 = x11848 * x11851;
int32_t x11853 = 64 * x11852;
float* x11854 = (float*)myMalloc(x11853 * sizeof(float));;
int32_t x11855;
if (x11831) {
x11855 = 0;
} else {
x11855 = x11768;
}
int32_t x11858;
if (x11832) {
x11858 = 0;
} else {
x11858 = 1;
}
for(int x11859=0; x11859 < 64; x11859++) {
int32_t x11871 = x11769 * x11859;
int32_t x11865 = x11852 * x11859;
for(int x11861=0; x11861 < x11848; x11861++) {
int32_t x11872 = x11855 * x11861;
int32_t x11873 = x11871 + x11872;
int32_t x11878 = x11858 * x11861;
int32_t x11867 = x11851 * x11861;
for(int x11863=0; x11863 < x11850; x11863++) {
int32_t x11874 = x11856 * x11863;
int32_t x11875 = x11873 + x11874;
int32_t x11869 = x11850 * x11863;
for(int x11864=0; x11864 < x11850; x11864++) {
int32_t x11876 = x11857 * x11864;
int32_t x11877 = x11875 + x11876;
float x11879 = x11771[x11877];
float x11880 = x16[x11878];
int32_t x11866 = x11864 + x11865;
int32_t x11868 = x11866 + x11867;
int32_t x11870 = x11868 + x11869;
float x11881 = x11879 + x11880;
x11854[x11870] = x11881;

}

}

}

}
float* x11891 = (float*)myMalloc(x11853 * sizeof(float));;
for(int x11893=0; x11893 < x11853; x11893++) {
float x11894 = x11854[x11893];
bool x11895 = x11894 < 0.0f;
if (x11895) {
x11891[x11893] = 0.0f;
} else {
float x11898 = x11854[x11893];
x11891[x11893] = x11898;
}

}
float* x11912 = (float*)myMalloc(x11911 * sizeof(float));;
int32_t x11915 = 64 * x11848;
int32_t x11916 = x11915 * x11907;
float* x11917 = (float*)myMalloc(x11916 * sizeof(float));;
int32_t x11913 = x11848 * x11907;
for(int x11918=0; x11918 < 64; x11918++) {
int32_t x11919 = x11918 * x11852;
float* x11920 = x11891+x11919;
int32_t x11921 = x11918 * x11908;
float* x11922 = x11912+x11921;
int32_t x11923 = x11918 * x11913;
float* x11924 = x11917+x11923;
for(int x11925=0; x11925 < x11848; x11925++) {
int32_t x11926 = x11925 / 1;
int32_t x11930 = x11926 * x11906;
int32_t x11931 = x11930 * x11906;
int32_t x11927 = x11925 % 1;
int32_t x11928 = x11927 / 1;
int32_t x11932 = x11928 * x11906;
int32_t x11933 = x11932 * x11906;
int32_t x11934 = x11931 + x11933;
int32_t x11929 = x11927 % 1;
int32_t x11935 = x11929 * x11906;
int32_t x11936 = x11935 * x11906;
int32_t x11937 = x11934 + x11936;
float* x11938 = x11924+x11937;
int32_t x11939 = x11926 * x11850;
int32_t x11940 = x11939 * x11850;
float* x11941 = x11920+x11940;
for(int x11943=0; x11943 < x11906; x11943++) {
int32_t x11945 = x11943 * x11906;
float* x11946 = x11938+x11945;
int32_t x11944 = x11943 + x11928;
int32_t x11947 = x11944 * x11850;
int32_t x11948 = x11947 + x11929;
float* x11949 = x11941+x11948;
memcpy(x11946, x11949, 4 * x11906);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x11907,x11848,1,x269,x11848,x11924,x11907,1,x11922,x11907);

}
int32_t x11958 = 0;
int32_t x11959 = 1;
x11959 *= 1;
x11958 += 1;
x11959 *= 1;
x11959 *= 1;
int32_t x11964 = x11958;
bool x11965 = x11964 >= 2;
if (x11965) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11970 = x11964 == 0;
if (x11970) {
int32_t x11971 = x11959;
bool x11972 = x11971 == 1024;
if (x11972) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11979 = x11959;
int32_t x11980 = 1024 / x11979;
bool x11981 = x11980 == 1;
bool x11984;
if (x454) {
bool x11982 = 1024 == x11980;
bool x11983 = x11981 || x11982;
x11984 = x11983;
} else {
x11984 = false;
}
bool x11988;
if (x11984) {
x11988 = x11987;
} else {
x11988 = false;
}
bool x11989;
if (x11988) {
x11989 = x11987;
} else {
x11989 = false;
}
if (x11989) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x11906,x11906,1,x11980,1,1);
assert(false && "");
}
bool x11995 = 1024 <= x11980;
int32_t x11996;
if (x11995) {
x11996 = x11980;
} else {
x11996 = 1024;
}
int32_t x12000 = x11996 * x11999;
int32_t x12001 = 64 * x12000;
float* x12002 = (float*)myMalloc(x12001 * sizeof(float));;
int32_t x12005;
if (x11981) {
x12005 = 0;
} else {
x12005 = 1;
}
for(int x12006=0; x12006 < 64; x12006++) {
int32_t x12018 = x11908 * x12006;
int32_t x12012 = x12000 * x12006;
for(int x12008=0; x12008 < x11996; x12008++) {
int32_t x12019 = x11907 * x12008;
int32_t x12020 = x12018 + x12019;
int32_t x12025 = x12005 * x12008;
int32_t x12014 = x11999 * x12008;
for(int x12010=0; x12010 < x11998; x12010++) {
int32_t x12021 = x12003 * x12010;
int32_t x12022 = x12020 + x12021;
int32_t x12016 = x11998 * x12010;
for(int x12011=0; x12011 < x11998; x12011++) {
int32_t x12023 = x12004 * x12011;
int32_t x12024 = x12022 + x12023;
float x12026 = x11912[x12024];
float x12027 = x216[x12025];
int32_t x12013 = x12011 + x12012;
int32_t x12015 = x12013 + x12014;
int32_t x12017 = x12015 + x12016;
float x12028 = x12026 - x12027;
x12002[x12017] = x12028;

}

}

}

}
float* x12038 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12040=0; x12040 < 1024; x12040++) {
float x12041 = x267[x12040];
float x12042 = x12041 + 1.0E-5f;
x12038[x12040] = x12042;

}
float* x12046 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12047=0; x12047 < 1024; x12047++) {
float x12048 = x12038[x12047];
double x12049 = (double)x12048;
double x12050 = sqrt(x12049);
float x12051 = (float)x12050;
x12046[x12047] = x12051;

}
int32_t x12055 = 0;
int32_t x12056 = 1;
x12056 *= 1;
x12055 += 1;
x12056 *= 1;
x12056 *= 1;
int32_t x12061 = x12055;
bool x12062 = x12061 >= 2;
if (x12062) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12067 = x12061 == 0;
if (x12067) {
int32_t x12068 = x12056;
bool x12069 = x12068 == 1024;
if (x12069) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12076 = x12056;
bool x12078 = x11996 == 1;
int32_t x12077 = 1024 / x12076;
bool x12079 = x12077 == 1;
bool x12083;
if (x454) {
bool x12080 = x12078 || x12079;
bool x12081 = x11996 == x12077;
bool x12082 = x12080 || x12081;
x12083 = x12082;
} else {
x12083 = false;
}
bool x12087;
if (x12083) {
x12087 = x12086;
} else {
x12087 = false;
}
bool x12088;
if (x12087) {
x12088 = x12086;
} else {
x12088 = false;
}
if (x12088) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11996,x11998,x11998,1,x12077,1,1);
assert(false && "");
}
bool x12094 = x11996 <= x12077;
int32_t x12095;
if (x12094) {
x12095 = x12077;
} else {
x12095 = x11996;
}
int32_t x12099 = x12095 * x12098;
int32_t x12100 = 64 * x12099;
float* x12101 = (float*)myMalloc(x12100 * sizeof(float));;
int32_t x12102;
if (x12078) {
x12102 = 0;
} else {
x12102 = x11999;
}
int32_t x12105;
if (x12079) {
x12105 = 0;
} else {
x12105 = 1;
}
for(int x12106=0; x12106 < 64; x12106++) {
int32_t x12118 = x12000 * x12106;
int32_t x12112 = x12099 * x12106;
for(int x12108=0; x12108 < x12095; x12108++) {
int32_t x12119 = x12102 * x12108;
int32_t x12120 = x12118 + x12119;
int32_t x12125 = x12105 * x12108;
int32_t x12114 = x12098 * x12108;
for(int x12110=0; x12110 < x12097; x12110++) {
int32_t x12121 = x12103 * x12110;
int32_t x12122 = x12120 + x12121;
int32_t x12116 = x12097 * x12110;
for(int x12111=0; x12111 < x12097; x12111++) {
int32_t x12123 = x12104 * x12111;
int32_t x12124 = x12122 + x12123;
float x12126 = x12002[x12124];
float x12127 = x12046[x12125];
int32_t x12113 = x12111 + x12112;
int32_t x12115 = x12113 + x12114;
int32_t x12117 = x12115 + x12116;
float x12128 = x12126 / x12127;
x12101[x12117] = x12128;

}

}

}

}
int32_t x12138 = 0;
int32_t x12139 = 1;
x12139 *= 1;
x12138 += 1;
x12139 *= 1;
x12139 *= 1;
int32_t x12144 = x12138;
bool x12145 = x12144 >= 2;
if (x12145) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12150 = x12144 == 0;
if (x12150) {
int32_t x12151 = x12139;
bool x12152 = x12151 == 1024;
if (x12152) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12159 = x12139;
bool x12161 = x12095 == 1;
int32_t x12160 = 1024 / x12159;
bool x12162 = x12160 == 1;
bool x12166;
if (x454) {
bool x12163 = x12161 || x12162;
bool x12164 = x12095 == x12160;
bool x12165 = x12163 || x12164;
x12166 = x12165;
} else {
x12166 = false;
}
bool x12170;
if (x12166) {
x12170 = x12169;
} else {
x12170 = false;
}
bool x12171;
if (x12170) {
x12171 = x12169;
} else {
x12171 = false;
}
if (x12171) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12095,x12097,x12097,1,x12160,1,1);
assert(false && "");
}
bool x12177 = x12095 <= x12160;
int32_t x12178;
if (x12177) {
x12178 = x12160;
} else {
x12178 = x12095;
}
int32_t x12182 = x12178 * x12181;
int32_t x12183 = 64 * x12182;
float* x12184 = (float*)myMalloc(x12183 * sizeof(float));;
int32_t x12185;
if (x12161) {
x12185 = 0;
} else {
x12185 = x12098;
}
int32_t x12188;
if (x12162) {
x12188 = 0;
} else {
x12188 = 1;
}
for(int x12189=0; x12189 < 64; x12189++) {
int32_t x12201 = x12099 * x12189;
int32_t x12195 = x12182 * x12189;
for(int x12191=0; x12191 < x12178; x12191++) {
int32_t x12202 = x12185 * x12191;
int32_t x12203 = x12201 + x12202;
int32_t x12208 = x12188 * x12191;
int32_t x12197 = x12181 * x12191;
for(int x12193=0; x12193 < x12180; x12193++) {
int32_t x12204 = x12186 * x12193;
int32_t x12205 = x12203 + x12204;
int32_t x12199 = x12180 * x12193;
for(int x12194=0; x12194 < x12180; x12194++) {
int32_t x12206 = x12187 * x12194;
int32_t x12207 = x12205 + x12206;
float x12209 = x12101[x12207];
float x12210 = x18[x12208];
int32_t x12196 = x12194 + x12195;
int32_t x12198 = x12196 + x12197;
int32_t x12200 = x12198 + x12199;
float x12211 = x12209 * x12210;
x12184[x12200] = x12211;

}

}

}

}
int32_t x12221 = 0;
int32_t x12222 = 1;
x12222 *= 1;
x12221 += 1;
x12222 *= 1;
x12222 *= 1;
int32_t x12227 = x12221;
bool x12228 = x12227 >= 2;
if (x12228) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12233 = x12227 == 0;
if (x12233) {
int32_t x12234 = x12222;
bool x12235 = x12234 == 1024;
if (x12235) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12242 = x12222;
bool x12244 = x12178 == 1;
int32_t x12243 = 1024 / x12242;
bool x12245 = x12243 == 1;
bool x12249;
if (x454) {
bool x12246 = x12244 || x12245;
bool x12247 = x12178 == x12243;
bool x12248 = x12246 || x12247;
x12249 = x12248;
} else {
x12249 = false;
}
bool x12253;
if (x12249) {
x12253 = x12252;
} else {
x12253 = false;
}
bool x12254;
if (x12253) {
x12254 = x12252;
} else {
x12254 = false;
}
if (x12254) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12178,x12180,x12180,1,x12243,1,1);
assert(false && "");
}
bool x12260 = x12178 <= x12243;
int32_t x12261;
if (x12260) {
x12261 = x12243;
} else {
x12261 = x12178;
}
int32_t x12265 = x12261 * x12264;
int32_t x12266 = 64 * x12265;
float* x12267 = (float*)myMalloc(x12266 * sizeof(float));;
int32_t x12268;
if (x12244) {
x12268 = 0;
} else {
x12268 = x12181;
}
int32_t x12271;
if (x12245) {
x12271 = 0;
} else {
x12271 = 1;
}
for(int x12272=0; x12272 < 64; x12272++) {
int32_t x12284 = x12182 * x12272;
int32_t x12278 = x12265 * x12272;
for(int x12274=0; x12274 < x12261; x12274++) {
int32_t x12285 = x12268 * x12274;
int32_t x12286 = x12284 + x12285;
int32_t x12291 = x12271 * x12274;
int32_t x12280 = x12264 * x12274;
for(int x12276=0; x12276 < x12263; x12276++) {
int32_t x12287 = x12269 * x12276;
int32_t x12288 = x12286 + x12287;
int32_t x12282 = x12263 * x12276;
for(int x12277=0; x12277 < x12263; x12277++) {
int32_t x12289 = x12270 * x12277;
int32_t x12290 = x12288 + x12289;
float x12292 = x12184[x12290];
float x12293 = x117[x12291];
int32_t x12279 = x12277 + x12278;
int32_t x12281 = x12279 + x12280;
int32_t x12283 = x12281 + x12282;
float x12294 = x12292 + x12293;
x12267[x12283] = x12294;

}

}

}

}
float* x12311 = (float*)myMalloc(x12310 * sizeof(float));;
int32_t x12314 = x11058 * x12306;
float* x12315 = (float*)myMalloc(x12314 * sizeof(float));;
int32_t x12312 = x10934 * x12306;
for(int x12316=0; x12316 < 64; x12316++) {
int32_t x12317 = x12316 * x10938;
float* x12318 = x11034+x12317;
int32_t x12319 = x12316 * x12307;
float* x12320 = x12311+x12319;
int32_t x12321 = x12316 * x12312;
float* x12322 = x12315+x12321;
for(int x12323=0; x12323 < x10934; x12323++) {
int32_t x12324 = x12323 / 1;
int32_t x12328 = x12324 * x12305;
int32_t x12329 = x12328 * x12305;
int32_t x12325 = x12323 % 1;
int32_t x12326 = x12325 / 1;
int32_t x12330 = x12326 * x12305;
int32_t x12331 = x12330 * x12305;
int32_t x12332 = x12329 + x12331;
int32_t x12327 = x12325 % 1;
int32_t x12333 = x12327 * x12305;
int32_t x12334 = x12333 * x12305;
int32_t x12335 = x12332 + x12334;
float* x12336 = x12322+x12335;
int32_t x12337 = x12324 * x10936;
int32_t x12338 = x12337 * x10936;
float* x12339 = x12318+x12338;
for(int x12341=0; x12341 < x12305; x12341++) {
int32_t x12345 = x12341 * x12305;
int32_t x12342 = x12341 * 2;
int32_t x12343 = x12342 + x12326;
int32_t x12348 = x12343 * x10936;
int32_t x12349 = x12348 + x12327;
for(int x12344=0; x12344 < x12305; x12344++) {
int32_t x12346 = x12345 + x12344;
float* x12347 = x12336+x12346;
int32_t x12350 = x12344 * 2;
int32_t x12351 = x12349 + x12350;
float* x12352 = x12339+x12351;
memcpy(x12347, x12352, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x12306,x10934,1,x75,x10934,x12322,x12306,1,x12320,x12306);

}
int32_t x12363 = 0;
int32_t x12364 = 1;
x12364 *= 1;
x12363 += 1;
x12364 *= 1;
x12364 *= 1;
int32_t x12369 = x12363;
bool x12370 = x12369 >= 2;
if (x12370) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12375 = x12369 == 0;
if (x12375) {
int32_t x12376 = x12364;
bool x12377 = x12376 == 1024;
if (x12377) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12384 = x12364;
int32_t x12385 = 1024 / x12384;
bool x12386 = x12385 == 1;
bool x12389;
if (x454) {
bool x12387 = 1024 == x12385;
bool x12388 = x12386 || x12387;
x12389 = x12388;
} else {
x12389 = false;
}
bool x12393;
if (x12389) {
x12393 = x12392;
} else {
x12393 = false;
}
bool x12394;
if (x12393) {
x12394 = x12392;
} else {
x12394 = false;
}
if (x12394) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x12305,x12305,1,x12385,1,1);
assert(false && "");
}
bool x12400 = 1024 <= x12385;
int32_t x12401;
if (x12400) {
x12401 = x12385;
} else {
x12401 = 1024;
}
int32_t x12405 = x12401 * x12404;
int32_t x12406 = 64 * x12405;
float* x12407 = (float*)myMalloc(x12406 * sizeof(float));;
int32_t x12410;
if (x12386) {
x12410 = 0;
} else {
x12410 = 1;
}
for(int x12411=0; x12411 < 64; x12411++) {
int32_t x12423 = x12307 * x12411;
int32_t x12417 = x12405 * x12411;
for(int x12413=0; x12413 < x12401; x12413++) {
int32_t x12424 = x12306 * x12413;
int32_t x12425 = x12423 + x12424;
int32_t x12430 = x12410 * x12413;
int32_t x12419 = x12404 * x12413;
for(int x12415=0; x12415 < x12403; x12415++) {
int32_t x12426 = x12408 * x12415;
int32_t x12427 = x12425 + x12426;
int32_t x12421 = x12403 * x12415;
for(int x12416=0; x12416 < x12403; x12416++) {
int32_t x12428 = x12409 * x12416;
int32_t x12429 = x12427 + x12428;
float x12431 = x12311[x12429];
float x12432 = x86[x12430];
int32_t x12418 = x12416 + x12417;
int32_t x12420 = x12418 + x12419;
int32_t x12422 = x12420 + x12421;
float x12433 = x12431 - x12432;
x12407[x12422] = x12433;

}

}

}

}
float* x12443 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12444=0; x12444 < 1024; x12444++) {
float x12445 = x211[x12444];
float x12446 = x12445 + 1.0E-5f;
x12443[x12444] = x12446;

}
float* x12450 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12451=0; x12451 < 1024; x12451++) {
float x12452 = x12443[x12451];
double x12453 = (double)x12452;
double x12454 = sqrt(x12453);
float x12455 = (float)x12454;
x12450[x12451] = x12455;

}
int32_t x12459 = 0;
int32_t x12460 = 1;
x12460 *= 1;
x12459 += 1;
x12460 *= 1;
x12460 *= 1;
int32_t x12465 = x12459;
bool x12466 = x12465 >= 2;
if (x12466) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12471 = x12465 == 0;
if (x12471) {
int32_t x12472 = x12460;
bool x12473 = x12472 == 1024;
if (x12473) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12480 = x12460;
bool x12482 = x12401 == 1;
int32_t x12481 = 1024 / x12480;
bool x12483 = x12481 == 1;
bool x12487;
if (x454) {
bool x12484 = x12482 || x12483;
bool x12485 = x12401 == x12481;
bool x12486 = x12484 || x12485;
x12487 = x12486;
} else {
x12487 = false;
}
bool x12491;
if (x12487) {
x12491 = x12490;
} else {
x12491 = false;
}
bool x12492;
if (x12491) {
x12492 = x12490;
} else {
x12492 = false;
}
if (x12492) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12401,x12403,x12403,1,x12481,1,1);
assert(false && "");
}
bool x12498 = x12401 <= x12481;
int32_t x12499;
if (x12498) {
x12499 = x12481;
} else {
x12499 = x12401;
}
int32_t x12503 = x12499 * x12502;
int32_t x12504 = 64 * x12503;
float* x12505 = (float*)myMalloc(x12504 * sizeof(float));;
int32_t x12506;
if (x12482) {
x12506 = 0;
} else {
x12506 = x12404;
}
int32_t x12509;
if (x12483) {
x12509 = 0;
} else {
x12509 = 1;
}
for(int x12510=0; x12510 < 64; x12510++) {
int32_t x12522 = x12405 * x12510;
int32_t x12516 = x12503 * x12510;
for(int x12512=0; x12512 < x12499; x12512++) {
int32_t x12523 = x12506 * x12512;
int32_t x12524 = x12522 + x12523;
int32_t x12529 = x12509 * x12512;
int32_t x12518 = x12502 * x12512;
for(int x12514=0; x12514 < x12501; x12514++) {
int32_t x12525 = x12507 * x12514;
int32_t x12526 = x12524 + x12525;
int32_t x12520 = x12501 * x12514;
for(int x12515=0; x12515 < x12501; x12515++) {
int32_t x12527 = x12508 * x12515;
int32_t x12528 = x12526 + x12527;
float x12530 = x12407[x12528];
float x12531 = x12450[x12529];
int32_t x12517 = x12515 + x12516;
int32_t x12519 = x12517 + x12518;
int32_t x12521 = x12519 + x12520;
float x12532 = x12530 / x12531;
x12505[x12521] = x12532;

}

}

}

}
int32_t x12542 = 0;
int32_t x12543 = 1;
x12543 *= 1;
x12542 += 1;
x12543 *= 1;
x12543 *= 1;
int32_t x12548 = x12542;
bool x12549 = x12548 >= 2;
if (x12549) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12554 = x12548 == 0;
if (x12554) {
int32_t x12555 = x12543;
bool x12556 = x12555 == 1024;
if (x12556) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12563 = x12543;
bool x12565 = x12499 == 1;
int32_t x12564 = 1024 / x12563;
bool x12566 = x12564 == 1;
bool x12570;
if (x454) {
bool x12567 = x12565 || x12566;
bool x12568 = x12499 == x12564;
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
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12499,x12501,x12501,1,x12564,1,1);
assert(false && "");
}
bool x12581 = x12499 <= x12564;
int32_t x12582;
if (x12581) {
x12582 = x12564;
} else {
x12582 = x12499;
}
int32_t x12586 = x12582 * x12585;
int32_t x12587 = 64 * x12586;
float* x12588 = (float*)myMalloc(x12587 * sizeof(float));;
int32_t x12589;
if (x12565) {
x12589 = 0;
} else {
x12589 = x12502;
}
int32_t x12592;
if (x12566) {
x12592 = 0;
} else {
x12592 = 1;
}
for(int x12593=0; x12593 < 64; x12593++) {
int32_t x12605 = x12503 * x12593;
int32_t x12599 = x12586 * x12593;
for(int x12595=0; x12595 < x12582; x12595++) {
int32_t x12606 = x12589 * x12595;
int32_t x12607 = x12605 + x12606;
int32_t x12612 = x12592 * x12595;
int32_t x12601 = x12585 * x12595;
for(int x12597=0; x12597 < x12584; x12597++) {
int32_t x12608 = x12590 * x12597;
int32_t x12609 = x12607 + x12608;
int32_t x12603 = x12584 * x12597;
for(int x12598=0; x12598 < x12584; x12598++) {
int32_t x12610 = x12591 * x12598;
int32_t x12611 = x12609 + x12610;
float x12613 = x12505[x12611];
float x12614 = x29[x12612];
int32_t x12600 = x12598 + x12599;
int32_t x12602 = x12600 + x12601;
int32_t x12604 = x12602 + x12603;
float x12615 = x12613 * x12614;
x12588[x12604] = x12615;

}

}

}

}
int32_t x12625 = 0;
int32_t x12626 = 1;
x12626 *= 1;
x12625 += 1;
x12626 *= 1;
x12626 *= 1;
int32_t x12631 = x12625;
bool x12632 = x12631 >= 2;
if (x12632) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12637 = x12631 == 0;
if (x12637) {
int32_t x12638 = x12626;
bool x12639 = x12638 == 1024;
if (x12639) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12646 = x12626;
bool x12648 = x12582 == 1;
int32_t x12647 = 1024 / x12646;
bool x12649 = x12647 == 1;
bool x12653;
if (x454) {
bool x12650 = x12648 || x12649;
bool x12651 = x12582 == x12647;
bool x12652 = x12650 || x12651;
x12653 = x12652;
} else {
x12653 = false;
}
bool x12657;
if (x12653) {
x12657 = x12656;
} else {
x12657 = false;
}
bool x12658;
if (x12657) {
x12658 = x12656;
} else {
x12658 = false;
}
if (x12658) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12582,x12584,x12584,1,x12647,1,1);
assert(false && "");
}
bool x12664 = x12582 <= x12647;
int32_t x12665;
if (x12664) {
x12665 = x12647;
} else {
x12665 = x12582;
}
int32_t x12669 = x12665 * x12668;
int32_t x12670 = 64 * x12669;
float* x12671 = (float*)myMalloc(x12670 * sizeof(float));;
int32_t x12672;
if (x12648) {
x12672 = 0;
} else {
x12672 = x12585;
}
int32_t x12675;
if (x12649) {
x12675 = 0;
} else {
x12675 = 1;
}
for(int x12676=0; x12676 < 64; x12676++) {
int32_t x12688 = x12586 * x12676;
int32_t x12682 = x12669 * x12676;
for(int x12678=0; x12678 < x12665; x12678++) {
int32_t x12689 = x12672 * x12678;
int32_t x12690 = x12688 + x12689;
int32_t x12695 = x12675 * x12678;
int32_t x12684 = x12668 * x12678;
for(int x12680=0; x12680 < x12667; x12680++) {
int32_t x12691 = x12673 * x12680;
int32_t x12692 = x12690 + x12691;
int32_t x12686 = x12667 * x12680;
for(int x12681=0; x12681 < x12667; x12681++) {
int32_t x12693 = x12674 * x12681;
int32_t x12694 = x12692 + x12693;
float x12696 = x12588[x12694];
float x12697 = x220[x12695];
int32_t x12683 = x12681 + x12682;
int32_t x12685 = x12683 + x12684;
int32_t x12687 = x12685 + x12686;
float x12698 = x12696 + x12697;
x12671[x12687] = x12698;

}

}

}

}
bool x12708 = x12261 == 1;
bool x12709 = x12665 == 1;
bool x12710 = x12708 || x12709;
bool x12711 = x12261 == x12665;
bool x12712 = x12710 || x12711;
bool x12718;
if (x12712) {
x12718 = x12717;
} else {
x12718 = false;
}
bool x12719;
if (x12718) {
x12719 = x12717;
} else {
x12719 = false;
}
if (x12719) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12261,x12263,x12263,64,x12665,x12667,x12667);
assert(false && "");
}
bool x12725 = x12261 <= x12665;
int32_t x12726;
if (x12725) {
x12726 = x12665;
} else {
x12726 = x12261;
}
int32_t x12732;
if (x12708) {
x12732 = 0;
} else {
x12732 = x12264;
}
int32_t x12735;
if (x12709) {
x12735 = 0;
} else {
x12735 = x12668;
}
for(int x12738=0; x12738 < 64; x12738++) {
int32_t x12744 = x12265 * x12738;
int32_t x12751 = x12669 * x12738;
for(int x12740=0; x12740 < x12726; x12740++) {
int32_t x12745 = x12732 * x12740;
int32_t x12746 = x12744 + x12745;
int32_t x12752 = x12735 * x12740;
int32_t x12753 = x12751 + x12752;
for(int x12742=0; x12742 < x12728; x12742++) {
int32_t x12747 = x12733 * x12742;
int32_t x12748 = x12746 + x12747;
int32_t x12754 = x12736 * x12742;
int32_t x12755 = x12753 + x12754;
for(int x12743=0; x12743 < x12728; x12743++) {
int32_t x12749 = x12734 * x12743;
int32_t x12750 = x12748 + x12749;
float x12758 = x12267[x12750];
int32_t x12756 = x12737 * x12743;
int32_t x12757 = x12755 + x12756;
float x12759 = x12671[x12757];
float x12760 = x12758 + x12759;
x12267[x12750] = x12760;

}

}

}

}
float* x12770 = (float*)myMalloc(x12266 * sizeof(float));;
for(int x12772=0; x12772 < x12266; x12772++) {
float x12773 = x12267[x12772];
bool x12774 = x12773 < 0.0f;
if (x12774) {
x12770[x12772] = 0.0f;
} else {
float x12777 = x12267[x12772];
x12770[x12772] = x12777;
}

}
float* x12791 = (float*)myMalloc(x12790 * sizeof(float));;
int32_t x12794 = 64 * x12261;
int32_t x12795 = x12794 * x12786;
float* x12796 = (float*)myMalloc(x12795 * sizeof(float));;
int32_t x12792 = x12261 * x12786;
for(int x12797=0; x12797 < 64; x12797++) {
int32_t x12798 = x12797 * x12265;
float* x12799 = x12770+x12798;
int32_t x12800 = x12797 * x12787;
float* x12801 = x12791+x12800;
int32_t x12802 = x12797 * x12792;
float* x12803 = x12796+x12802;
for(int x12804=0; x12804 < x12261; x12804++) {
int32_t x12805 = x12804 / 1;
int32_t x12809 = x12805 * x12785;
int32_t x12810 = x12809 * x12785;
int32_t x12806 = x12804 % 1;
int32_t x12807 = x12806 / 1;
int32_t x12811 = x12807 * x12785;
int32_t x12812 = x12811 * x12785;
int32_t x12813 = x12810 + x12812;
int32_t x12808 = x12806 % 1;
int32_t x12814 = x12808 * x12785;
int32_t x12815 = x12814 * x12785;
int32_t x12816 = x12813 + x12815;
float* x12817 = x12803+x12816;
int32_t x12818 = x12805 * x12263;
int32_t x12819 = x12818 * x12263;
float* x12820 = x12799+x12819;
for(int x12822=0; x12822 < x12785; x12822++) {
int32_t x12824 = x12822 * x12785;
float* x12825 = x12817+x12824;
int32_t x12823 = x12822 + x12807;
int32_t x12826 = x12823 * x12263;
int32_t x12827 = x12826 + x12808;
float* x12828 = x12820+x12827;
memcpy(x12825, x12828, 4 * x12785);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x12786,x12261,1,x13,x12261,x12803,x12786,1,x12801,x12786);

}
int32_t x12837 = 0;
int32_t x12838 = 1;
x12838 *= 1;
x12837 += 1;
x12838 *= 1;
x12838 *= 1;
int32_t x12843 = x12837;
bool x12844 = x12843 >= 2;
if (x12844) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12849 = x12843 == 0;
if (x12849) {
int32_t x12850 = x12838;
bool x12851 = x12850 == 256;
if (x12851) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12858 = x12838;
int32_t x12859 = 256 / x12858;
bool x12860 = x12859 == 1;
bool x12863;
if (x454) {
bool x12861 = 256 == x12859;
bool x12862 = x12860 || x12861;
x12863 = x12862;
} else {
x12863 = false;
}
bool x12867;
if (x12863) {
x12867 = x12866;
} else {
x12867 = false;
}
bool x12868;
if (x12867) {
x12868 = x12866;
} else {
x12868 = false;
}
if (x12868) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x12785,x12785,1,x12859,1,1);
assert(false && "");
}
bool x12874 = 256 <= x12859;
int32_t x12875;
if (x12874) {
x12875 = x12859;
} else {
x12875 = 256;
}
int32_t x12879 = x12875 * x12878;
int32_t x12880 = 64 * x12879;
float* x12881 = (float*)myMalloc(x12880 * sizeof(float));;
int32_t x12884;
if (x12860) {
x12884 = 0;
} else {
x12884 = 1;
}
for(int x12885=0; x12885 < 64; x12885++) {
int32_t x12897 = x12787 * x12885;
int32_t x12891 = x12879 * x12885;
for(int x12887=0; x12887 < x12875; x12887++) {
int32_t x12898 = x12786 * x12887;
int32_t x12899 = x12897 + x12898;
int32_t x12904 = x12884 * x12887;
int32_t x12893 = x12878 * x12887;
for(int x12889=0; x12889 < x12877; x12889++) {
int32_t x12900 = x12882 * x12889;
int32_t x12901 = x12899 + x12900;
int32_t x12895 = x12877 * x12889;
for(int x12890=0; x12890 < x12877; x12890++) {
int32_t x12902 = x12883 * x12890;
int32_t x12903 = x12901 + x12902;
float x12905 = x12791[x12903];
float x12906 = x259[x12904];
int32_t x12892 = x12890 + x12891;
int32_t x12894 = x12892 + x12893;
int32_t x12896 = x12894 + x12895;
float x12907 = x12905 - x12906;
x12881[x12896] = x12907;

}

}

}

}
float* x12917 = (float*)myMalloc(256 * sizeof(float));;
for(int x12918=0; x12918 < 256; x12918++) {
float x12919 = x157[x12918];
float x12920 = x12919 + 1.0E-5f;
x12917[x12918] = x12920;

}
float* x12924 = (float*)myMalloc(256 * sizeof(float));;
for(int x12925=0; x12925 < 256; x12925++) {
float x12926 = x12917[x12925];
double x12927 = (double)x12926;
double x12928 = sqrt(x12927);
float x12929 = (float)x12928;
x12924[x12925] = x12929;

}
int32_t x12933 = 0;
int32_t x12934 = 1;
x12934 *= 1;
x12933 += 1;
x12934 *= 1;
x12934 *= 1;
int32_t x12939 = x12933;
bool x12940 = x12939 >= 2;
if (x12940) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12945 = x12939 == 0;
if (x12945) {
int32_t x12946 = x12934;
bool x12947 = x12946 == 256;
if (x12947) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12954 = x12934;
bool x12956 = x12875 == 1;
int32_t x12955 = 256 / x12954;
bool x12957 = x12955 == 1;
bool x12961;
if (x454) {
bool x12958 = x12956 || x12957;
bool x12959 = x12875 == x12955;
bool x12960 = x12958 || x12959;
x12961 = x12960;
} else {
x12961 = false;
}
bool x12965;
if (x12961) {
x12965 = x12964;
} else {
x12965 = false;
}
bool x12966;
if (x12965) {
x12966 = x12964;
} else {
x12966 = false;
}
if (x12966) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12875,x12877,x12877,1,x12955,1,1);
assert(false && "");
}
bool x12972 = x12875 <= x12955;
int32_t x12973;
if (x12972) {
x12973 = x12955;
} else {
x12973 = x12875;
}
int32_t x12977 = x12973 * x12976;
int32_t x12978 = 64 * x12977;
float* x12979 = (float*)myMalloc(x12978 * sizeof(float));;
int32_t x12980;
if (x12956) {
x12980 = 0;
} else {
x12980 = x12878;
}
int32_t x12983;
if (x12957) {
x12983 = 0;
} else {
x12983 = 1;
}
for(int x12984=0; x12984 < 64; x12984++) {
int32_t x12996 = x12879 * x12984;
int32_t x12990 = x12977 * x12984;
for(int x12986=0; x12986 < x12973; x12986++) {
int32_t x12997 = x12980 * x12986;
int32_t x12998 = x12996 + x12997;
int32_t x13003 = x12983 * x12986;
int32_t x12992 = x12976 * x12986;
for(int x12988=0; x12988 < x12975; x12988++) {
int32_t x12999 = x12981 * x12988;
int32_t x13000 = x12998 + x12999;
int32_t x12994 = x12975 * x12988;
for(int x12989=0; x12989 < x12975; x12989++) {
int32_t x13001 = x12982 * x12989;
int32_t x13002 = x13000 + x13001;
float x13004 = x12881[x13002];
float x13005 = x12924[x13003];
int32_t x12991 = x12989 + x12990;
int32_t x12993 = x12991 + x12992;
int32_t x12995 = x12993 + x12994;
float x13006 = x13004 / x13005;
x12979[x12995] = x13006;

}

}

}

}
int32_t x13016 = 0;
int32_t x13017 = 1;
x13017 *= 1;
x13016 += 1;
x13017 *= 1;
x13017 *= 1;
int32_t x13022 = x13016;
bool x13023 = x13022 >= 2;
if (x13023) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13028 = x13022 == 0;
if (x13028) {
int32_t x13029 = x13017;
bool x13030 = x13029 == 256;
if (x13030) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13037 = x13017;
bool x13039 = x12973 == 1;
int32_t x13038 = 256 / x13037;
bool x13040 = x13038 == 1;
bool x13044;
if (x454) {
bool x13041 = x13039 || x13040;
bool x13042 = x12973 == x13038;
bool x13043 = x13041 || x13042;
x13044 = x13043;
} else {
x13044 = false;
}
bool x13048;
if (x13044) {
x13048 = x13047;
} else {
x13048 = false;
}
bool x13049;
if (x13048) {
x13049 = x13047;
} else {
x13049 = false;
}
if (x13049) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12973,x12975,x12975,1,x13038,1,1);
assert(false && "");
}
bool x13055 = x12973 <= x13038;
int32_t x13056;
if (x13055) {
x13056 = x13038;
} else {
x13056 = x12973;
}
int32_t x13060 = x13056 * x13059;
int32_t x13061 = 64 * x13060;
float* x13062 = (float*)myMalloc(x13061 * sizeof(float));;
int32_t x13063;
if (x13039) {
x13063 = 0;
} else {
x13063 = x12976;
}
int32_t x13066;
if (x13040) {
x13066 = 0;
} else {
x13066 = 1;
}
for(int x13067=0; x13067 < 64; x13067++) {
int32_t x13079 = x12977 * x13067;
int32_t x13073 = x13060 * x13067;
for(int x13069=0; x13069 < x13056; x13069++) {
int32_t x13080 = x13063 * x13069;
int32_t x13081 = x13079 + x13080;
int32_t x13086 = x13066 * x13069;
int32_t x13075 = x13059 * x13069;
for(int x13071=0; x13071 < x13058; x13071++) {
int32_t x13082 = x13064 * x13071;
int32_t x13083 = x13081 + x13082;
int32_t x13077 = x13058 * x13071;
for(int x13072=0; x13072 < x13058; x13072++) {
int32_t x13084 = x13065 * x13072;
int32_t x13085 = x13083 + x13084;
float x13087 = x12979[x13085];
float x13088 = x30[x13086];
int32_t x13074 = x13072 + x13073;
int32_t x13076 = x13074 + x13075;
int32_t x13078 = x13076 + x13077;
float x13089 = x13087 * x13088;
x13062[x13078] = x13089;

}

}

}

}
int32_t x13099 = 0;
int32_t x13100 = 1;
x13100 *= 1;
x13099 += 1;
x13100 *= 1;
x13100 *= 1;
int32_t x13105 = x13099;
bool x13106 = x13105 >= 2;
if (x13106) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13111 = x13105 == 0;
if (x13111) {
int32_t x13112 = x13100;
bool x13113 = x13112 == 256;
if (x13113) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13120 = x13100;
bool x13122 = x13056 == 1;
int32_t x13121 = 256 / x13120;
bool x13123 = x13121 == 1;
bool x13127;
if (x454) {
bool x13124 = x13122 || x13123;
bool x13125 = x13056 == x13121;
bool x13126 = x13124 || x13125;
x13127 = x13126;
} else {
x13127 = false;
}
bool x13131;
if (x13127) {
x13131 = x13130;
} else {
x13131 = false;
}
bool x13132;
if (x13131) {
x13132 = x13130;
} else {
x13132 = false;
}
if (x13132) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13056,x13058,x13058,1,x13121,1,1);
assert(false && "");
}
bool x13138 = x13056 <= x13121;
int32_t x13139;
if (x13138) {
x13139 = x13121;
} else {
x13139 = x13056;
}
int32_t x13143 = x13139 * x13142;
int32_t x13144 = 64 * x13143;
float* x13145 = (float*)myMalloc(x13144 * sizeof(float));;
int32_t x13146;
if (x13122) {
x13146 = 0;
} else {
x13146 = x13059;
}
int32_t x13149;
if (x13123) {
x13149 = 0;
} else {
x13149 = 1;
}
for(int x13150=0; x13150 < 64; x13150++) {
int32_t x13162 = x13060 * x13150;
int32_t x13156 = x13143 * x13150;
for(int x13152=0; x13152 < x13139; x13152++) {
int32_t x13163 = x13146 * x13152;
int32_t x13164 = x13162 + x13163;
int32_t x13169 = x13149 * x13152;
int32_t x13158 = x13142 * x13152;
for(int x13154=0; x13154 < x13141; x13154++) {
int32_t x13165 = x13147 * x13154;
int32_t x13166 = x13164 + x13165;
int32_t x13160 = x13141 * x13154;
for(int x13155=0; x13155 < x13141; x13155++) {
int32_t x13167 = x13148 * x13155;
int32_t x13168 = x13166 + x13167;
float x13170 = x13062[x13168];
float x13171 = x219[x13169];
int32_t x13157 = x13155 + x13156;
int32_t x13159 = x13157 + x13158;
int32_t x13161 = x13159 + x13160;
float x13172 = x13170 + x13171;
x13145[x13161] = x13172;

}

}

}

}
float* x13182 = (float*)myMalloc(x13144 * sizeof(float));;
for(int x13184=0; x13184 < x13144; x13184++) {
float x13185 = x13145[x13184];
bool x13186 = x13185 < 0.0f;
if (x13186) {
x13182[x13184] = 0.0f;
} else {
float x13189 = x13145[x13184];
x13182[x13184] = x13189;
}

}
float* x13204 = (float*)myMalloc(x13203 * sizeof(float));;
int32_t x13205 = 9 * x13139;
int32_t x13208 = 64 * x13205;
int32_t x13209 = x13208 * x13199;
float* x13210 = (float*)myMalloc(x13209 * sizeof(float));;
int32_t x13206 = x13205 * x13199;
int32_t x13218 = x13139 * 3;
int32_t x13219 = x13218 * 3;
for(int x13211=0; x13211 < 64; x13211++) {
int32_t x13212 = x13211 * x13143;
float* x13213 = x13182+x13212;
int32_t x13214 = x13211 * x13200;
float* x13215 = x13204+x13214;
int32_t x13216 = x13211 * x13206;
float* x13217 = x13210+x13216;
for(int x13221=0; x13221 < x13219; x13221++) {
int32_t x13222 = x13221 / 9;
int32_t x13226 = x13222 * 3;
int32_t x13227 = x13226 * 3;
int32_t x13228 = x13227 * x13198;
int32_t x13229 = x13228 * x13198;
int32_t x13223 = x13221 % 9;
int32_t x13224 = x13223 / 3;
int32_t x13230 = x13224 * 3;
int32_t x13231 = x13230 * x13198;
int32_t x13232 = x13231 * x13198;
int32_t x13233 = x13229 + x13232;
int32_t x13225 = x13223 % 3;
int32_t x13234 = x13225 * x13198;
int32_t x13235 = x13234 * x13198;
int32_t x13236 = x13233 + x13235;
float* x13237 = x13217+x13236;
int32_t x13238 = x13222 * x13141;
int32_t x13239 = x13238 * x13141;
float* x13240 = x13213+x13239;
int32_t x13253 = 1 - x13225;
bool x13254 = x13253 > 0;
int32_t x13255;
if (x13254) {
x13255 = x13253;
} else {
x13255 = 0;
}
int32_t x13256 = 3 - x13225;
int32_t x13257 = x13256 - 1;
int32_t x13258 = 1 - x13257;
bool x13259 = x13258 > 0;
int32_t x13260;
if (x13259) {
x13260 = x13258;
} else {
x13260 = 0;
}
int32_t x13261 = x13198 - x13260;
int32_t x13262 = x13261 - x13255;
bool x13263 = x13262 <= 0;
bool x13267 = x13255 > 0;
int32_t x13252 = -1 + x13225;
bool x13280 = x13260 > 0;
for(int x13242=0; x13242 < x13198; x13242++) {
int32_t x13243 = x13242 - 1;
int32_t x13244 = x13243 + x13224;
bool x13245 = x13244 < 0;
bool x13246 = x13244 >= x13141;
bool x13247 = x13245 || x13246;
if (x13247) {
int32_t x13248 = x13242 * x13198;
float* x13249 = x13237+x13248;
memset(x13249, 0, 4 * x13198);;
} else {
if (x13263) {
int32_t x13248 = x13242 * x13198;
float* x13264 = x13237+x13248;
memset(x13264, 0, 4 * x13198);;
} else {
int32_t x13248 = x13242 * x13198;
if (x13267) {
float* x13268 = x13237+x13248;
memset(x13268, 0, 4 * x13255);;
} else {
}
// may have segfault here
int32_t x13273 = x13248 + x13255;
float* x13274 = x13237+x13273;
int32_t x13275 = x13244 * x13141;
int32_t x13276 = x13275 + x13252;
int32_t x13277 = x13276 + x13255;
float* x13278 = x13240+x13277;
memcpy(x13274, x13278, 4 * x13262);;
if (x13280) {
int32_t x13281 = x13248 + x13198;
int32_t x13282 = x13281 - x13260;
float* x13283 = x13237+x13282;
memset(x13283, 0, 4 * x13260);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x13199,x13205,1,x31,x13205,x13217,x13199,1,x13215,x13199);

}
int32_t x13298 = 0;
int32_t x13299 = 1;
x13299 *= 1;
x13298 += 1;
x13299 *= 1;
x13299 *= 1;
int32_t x13304 = x13298;
bool x13305 = x13304 >= 2;
if (x13305) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13310 = x13304 == 0;
if (x13310) {
int32_t x13311 = x13299;
bool x13312 = x13311 == 256;
if (x13312) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13319 = x13299;
int32_t x13320 = 256 / x13319;
bool x13321 = x13320 == 1;
bool x13324;
if (x454) {
bool x13322 = 256 == x13320;
bool x13323 = x13321 || x13322;
x13324 = x13323;
} else {
x13324 = false;
}
bool x13328;
if (x13324) {
x13328 = x13327;
} else {
x13328 = false;
}
bool x13329;
if (x13328) {
x13329 = x13327;
} else {
x13329 = false;
}
if (x13329) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x13198,x13198,1,x13320,1,1);
assert(false && "");
}
bool x13335 = 256 <= x13320;
int32_t x13336;
if (x13335) {
x13336 = x13320;
} else {
x13336 = 256;
}
int32_t x13340 = x13336 * x13339;
int32_t x13341 = 64 * x13340;
float* x13342 = (float*)myMalloc(x13341 * sizeof(float));;
int32_t x13345;
if (x13321) {
x13345 = 0;
} else {
x13345 = 1;
}
for(int x13346=0; x13346 < 64; x13346++) {
int32_t x13358 = x13200 * x13346;
int32_t x13352 = x13340 * x13346;
for(int x13348=0; x13348 < x13336; x13348++) {
int32_t x13359 = x13199 * x13348;
int32_t x13360 = x13358 + x13359;
int32_t x13365 = x13345 * x13348;
int32_t x13354 = x13339 * x13348;
for(int x13350=0; x13350 < x13338; x13350++) {
int32_t x13361 = x13343 * x13350;
int32_t x13362 = x13360 + x13361;
int32_t x13356 = x13338 * x13350;
for(int x13351=0; x13351 < x13338; x13351++) {
int32_t x13363 = x13344 * x13351;
int32_t x13364 = x13362 + x13363;
float x13366 = x13204[x13364];
float x13367 = x200[x13365];
int32_t x13353 = x13351 + x13352;
int32_t x13355 = x13353 + x13354;
int32_t x13357 = x13355 + x13356;
float x13368 = x13366 - x13367;
x13342[x13357] = x13368;

}

}

}

}
float* x13378 = (float*)myMalloc(256 * sizeof(float));;
for(int x13379=0; x13379 < 256; x13379++) {
float x13380 = x237[x13379];
float x13381 = x13380 + 1.0E-5f;
x13378[x13379] = x13381;

}
float* x13385 = (float*)myMalloc(256 * sizeof(float));;
for(int x13386=0; x13386 < 256; x13386++) {
float x13387 = x13378[x13386];
double x13388 = (double)x13387;
double x13389 = sqrt(x13388);
float x13390 = (float)x13389;
x13385[x13386] = x13390;

}
int32_t x13394 = 0;
int32_t x13395 = 1;
x13395 *= 1;
x13394 += 1;
x13395 *= 1;
x13395 *= 1;
int32_t x13400 = x13394;
bool x13401 = x13400 >= 2;
if (x13401) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13406 = x13400 == 0;
if (x13406) {
int32_t x13407 = x13395;
bool x13408 = x13407 == 256;
if (x13408) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13415 = x13395;
bool x13417 = x13336 == 1;
int32_t x13416 = 256 / x13415;
bool x13418 = x13416 == 1;
bool x13422;
if (x454) {
bool x13419 = x13417 || x13418;
bool x13420 = x13336 == x13416;
bool x13421 = x13419 || x13420;
x13422 = x13421;
} else {
x13422 = false;
}
bool x13426;
if (x13422) {
x13426 = x13425;
} else {
x13426 = false;
}
bool x13427;
if (x13426) {
x13427 = x13425;
} else {
x13427 = false;
}
if (x13427) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13336,x13338,x13338,1,x13416,1,1);
assert(false && "");
}
bool x13433 = x13336 <= x13416;
int32_t x13434;
if (x13433) {
x13434 = x13416;
} else {
x13434 = x13336;
}
int32_t x13438 = x13434 * x13437;
int32_t x13439 = 64 * x13438;
float* x13440 = (float*)myMalloc(x13439 * sizeof(float));;
int32_t x13441;
if (x13417) {
x13441 = 0;
} else {
x13441 = x13339;
}
int32_t x13444;
if (x13418) {
x13444 = 0;
} else {
x13444 = 1;
}
for(int x13445=0; x13445 < 64; x13445++) {
int32_t x13457 = x13340 * x13445;
int32_t x13451 = x13438 * x13445;
for(int x13447=0; x13447 < x13434; x13447++) {
int32_t x13458 = x13441 * x13447;
int32_t x13459 = x13457 + x13458;
int32_t x13464 = x13444 * x13447;
int32_t x13453 = x13437 * x13447;
for(int x13449=0; x13449 < x13436; x13449++) {
int32_t x13460 = x13442 * x13449;
int32_t x13461 = x13459 + x13460;
int32_t x13455 = x13436 * x13449;
for(int x13450=0; x13450 < x13436; x13450++) {
int32_t x13462 = x13443 * x13450;
int32_t x13463 = x13461 + x13462;
float x13465 = x13342[x13463];
float x13466 = x13385[x13464];
int32_t x13452 = x13450 + x13451;
int32_t x13454 = x13452 + x13453;
int32_t x13456 = x13454 + x13455;
float x13467 = x13465 / x13466;
x13440[x13456] = x13467;

}

}

}

}
int32_t x13477 = 0;
int32_t x13478 = 1;
x13478 *= 1;
x13477 += 1;
x13478 *= 1;
x13478 *= 1;
int32_t x13483 = x13477;
bool x13484 = x13483 >= 2;
if (x13484) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13489 = x13483 == 0;
if (x13489) {
int32_t x13490 = x13478;
bool x13491 = x13490 == 256;
if (x13491) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13498 = x13478;
bool x13500 = x13434 == 1;
int32_t x13499 = 256 / x13498;
bool x13501 = x13499 == 1;
bool x13505;
if (x454) {
bool x13502 = x13500 || x13501;
bool x13503 = x13434 == x13499;
bool x13504 = x13502 || x13503;
x13505 = x13504;
} else {
x13505 = false;
}
bool x13509;
if (x13505) {
x13509 = x13508;
} else {
x13509 = false;
}
bool x13510;
if (x13509) {
x13510 = x13508;
} else {
x13510 = false;
}
if (x13510) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13434,x13436,x13436,1,x13499,1,1);
assert(false && "");
}
bool x13516 = x13434 <= x13499;
int32_t x13517;
if (x13516) {
x13517 = x13499;
} else {
x13517 = x13434;
}
int32_t x13521 = x13517 * x13520;
int32_t x13522 = 64 * x13521;
float* x13523 = (float*)myMalloc(x13522 * sizeof(float));;
int32_t x13524;
if (x13500) {
x13524 = 0;
} else {
x13524 = x13437;
}
int32_t x13527;
if (x13501) {
x13527 = 0;
} else {
x13527 = 1;
}
for(int x13528=0; x13528 < 64; x13528++) {
int32_t x13540 = x13438 * x13528;
int32_t x13534 = x13521 * x13528;
for(int x13530=0; x13530 < x13517; x13530++) {
int32_t x13541 = x13524 * x13530;
int32_t x13542 = x13540 + x13541;
int32_t x13547 = x13527 * x13530;
int32_t x13536 = x13520 * x13530;
for(int x13532=0; x13532 < x13519; x13532++) {
int32_t x13543 = x13525 * x13532;
int32_t x13544 = x13542 + x13543;
int32_t x13538 = x13519 * x13532;
for(int x13533=0; x13533 < x13519; x13533++) {
int32_t x13545 = x13526 * x13533;
int32_t x13546 = x13544 + x13545;
float x13548 = x13440[x13546];
float x13549 = x271[x13547];
int32_t x13535 = x13533 + x13534;
int32_t x13537 = x13535 + x13536;
int32_t x13539 = x13537 + x13538;
float x13550 = x13548 * x13549;
x13523[x13539] = x13550;

}

}

}

}
int32_t x13560 = 0;
int32_t x13561 = 1;
x13561 *= 1;
x13560 += 1;
x13561 *= 1;
x13561 *= 1;
int32_t x13566 = x13560;
bool x13567 = x13566 >= 2;
if (x13567) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13572 = x13566 == 0;
if (x13572) {
int32_t x13573 = x13561;
bool x13574 = x13573 == 256;
if (x13574) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13581 = x13561;
bool x13583 = x13517 == 1;
int32_t x13582 = 256 / x13581;
bool x13584 = x13582 == 1;
bool x13588;
if (x454) {
bool x13585 = x13583 || x13584;
bool x13586 = x13517 == x13582;
bool x13587 = x13585 || x13586;
x13588 = x13587;
} else {
x13588 = false;
}
bool x13592;
if (x13588) {
x13592 = x13591;
} else {
x13592 = false;
}
bool x13593;
if (x13592) {
x13593 = x13591;
} else {
x13593 = false;
}
if (x13593) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13517,x13519,x13519,1,x13582,1,1);
assert(false && "");
}
bool x13599 = x13517 <= x13582;
int32_t x13600;
if (x13599) {
x13600 = x13582;
} else {
x13600 = x13517;
}
int32_t x13604 = x13600 * x13603;
int32_t x13605 = 64 * x13604;
float* x13606 = (float*)myMalloc(x13605 * sizeof(float));;
int32_t x13607;
if (x13583) {
x13607 = 0;
} else {
x13607 = x13520;
}
int32_t x13610;
if (x13584) {
x13610 = 0;
} else {
x13610 = 1;
}
for(int x13611=0; x13611 < 64; x13611++) {
int32_t x13623 = x13521 * x13611;
int32_t x13617 = x13604 * x13611;
for(int x13613=0; x13613 < x13600; x13613++) {
int32_t x13624 = x13607 * x13613;
int32_t x13625 = x13623 + x13624;
int32_t x13630 = x13610 * x13613;
int32_t x13619 = x13603 * x13613;
for(int x13615=0; x13615 < x13602; x13615++) {
int32_t x13626 = x13608 * x13615;
int32_t x13627 = x13625 + x13626;
int32_t x13621 = x13602 * x13615;
for(int x13616=0; x13616 < x13602; x13616++) {
int32_t x13628 = x13609 * x13616;
int32_t x13629 = x13627 + x13628;
float x13631 = x13523[x13629];
float x13632 = x96[x13630];
int32_t x13618 = x13616 + x13617;
int32_t x13620 = x13618 + x13619;
int32_t x13622 = x13620 + x13621;
float x13633 = x13631 + x13632;
x13606[x13622] = x13633;

}

}

}

}
float* x13643 = (float*)myMalloc(x13605 * sizeof(float));;
for(int x13645=0; x13645 < x13605; x13645++) {
float x13646 = x13606[x13645];
bool x13647 = x13646 < 0.0f;
if (x13647) {
x13643[x13645] = 0.0f;
} else {
float x13650 = x13606[x13645];
x13643[x13645] = x13650;
}

}
float* x13664 = (float*)myMalloc(x13663 * sizeof(float));;
int32_t x13667 = 64 * x13600;
int32_t x13668 = x13667 * x13659;
float* x13669 = (float*)myMalloc(x13668 * sizeof(float));;
int32_t x13665 = x13600 * x13659;
for(int x13670=0; x13670 < 64; x13670++) {
int32_t x13671 = x13670 * x13604;
float* x13672 = x13643+x13671;
int32_t x13673 = x13670 * x13660;
float* x13674 = x13664+x13673;
int32_t x13675 = x13670 * x13665;
float* x13676 = x13669+x13675;
for(int x13677=0; x13677 < x13600; x13677++) {
int32_t x13678 = x13677 / 1;
int32_t x13682 = x13678 * x13658;
int32_t x13683 = x13682 * x13658;
int32_t x13679 = x13677 % 1;
int32_t x13680 = x13679 / 1;
int32_t x13684 = x13680 * x13658;
int32_t x13685 = x13684 * x13658;
int32_t x13686 = x13683 + x13685;
int32_t x13681 = x13679 % 1;
int32_t x13687 = x13681 * x13658;
int32_t x13688 = x13687 * x13658;
int32_t x13689 = x13686 + x13688;
float* x13690 = x13676+x13689;
int32_t x13691 = x13678 * x13602;
int32_t x13692 = x13691 * x13602;
float* x13693 = x13672+x13692;
for(int x13695=0; x13695 < x13658; x13695++) {
int32_t x13697 = x13695 * x13658;
float* x13698 = x13690+x13697;
int32_t x13696 = x13695 + x13680;
int32_t x13699 = x13696 * x13602;
int32_t x13700 = x13699 + x13681;
float* x13701 = x13693+x13700;
memcpy(x13698, x13701, 4 * x13658);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x13659,x13600,1,x56,x13600,x13676,x13659,1,x13674,x13659);

}
int32_t x13710 = 0;
int32_t x13711 = 1;
x13711 *= 1;
x13710 += 1;
x13711 *= 1;
x13711 *= 1;
int32_t x13716 = x13710;
bool x13717 = x13716 >= 2;
if (x13717) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13722 = x13716 == 0;
if (x13722) {
int32_t x13723 = x13711;
bool x13724 = x13723 == 1024;
if (x13724) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13731 = x13711;
int32_t x13732 = 1024 / x13731;
bool x13733 = x13732 == 1;
bool x13736;
if (x454) {
bool x13734 = 1024 == x13732;
bool x13735 = x13733 || x13734;
x13736 = x13735;
} else {
x13736 = false;
}
bool x13740;
if (x13736) {
x13740 = x13739;
} else {
x13740 = false;
}
bool x13741;
if (x13740) {
x13741 = x13739;
} else {
x13741 = false;
}
if (x13741) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x13658,x13658,1,x13732,1,1);
assert(false && "");
}
bool x13747 = 1024 <= x13732;
int32_t x13748;
if (x13747) {
x13748 = x13732;
} else {
x13748 = 1024;
}
int32_t x13752 = x13748 * x13751;
int32_t x13753 = 64 * x13752;
float* x13754 = (float*)myMalloc(x13753 * sizeof(float));;
int32_t x13757;
if (x13733) {
x13757 = 0;
} else {
x13757 = 1;
}
for(int x13758=0; x13758 < 64; x13758++) {
int32_t x13770 = x13660 * x13758;
int32_t x13764 = x13752 * x13758;
for(int x13760=0; x13760 < x13748; x13760++) {
int32_t x13771 = x13659 * x13760;
int32_t x13772 = x13770 + x13771;
int32_t x13777 = x13757 * x13760;
int32_t x13766 = x13751 * x13760;
for(int x13762=0; x13762 < x13750; x13762++) {
int32_t x13773 = x13755 * x13762;
int32_t x13774 = x13772 + x13773;
int32_t x13768 = x13750 * x13762;
for(int x13763=0; x13763 < x13750; x13763++) {
int32_t x13775 = x13756 * x13763;
int32_t x13776 = x13774 + x13775;
float x13778 = x13664[x13776];
float x13779 = x182[x13777];
int32_t x13765 = x13763 + x13764;
int32_t x13767 = x13765 + x13766;
int32_t x13769 = x13767 + x13768;
float x13780 = x13778 - x13779;
x13754[x13769] = x13780;

}

}

}

}
float* x13790 = (float*)myMalloc(1024 * sizeof(float));;
for(int x13791=0; x13791 < 1024; x13791++) {
float x13792 = x143[x13791];
float x13793 = x13792 + 1.0E-5f;
x13790[x13791] = x13793;

}
float* x13797 = (float*)myMalloc(1024 * sizeof(float));;
for(int x13798=0; x13798 < 1024; x13798++) {
float x13799 = x13790[x13798];
double x13800 = (double)x13799;
double x13801 = sqrt(x13800);
float x13802 = (float)x13801;
x13797[x13798] = x13802;

}
int32_t x13806 = 0;
int32_t x13807 = 1;
x13807 *= 1;
x13806 += 1;
x13807 *= 1;
x13807 *= 1;
int32_t x13812 = x13806;
bool x13813 = x13812 >= 2;
if (x13813) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13818 = x13812 == 0;
if (x13818) {
int32_t x13819 = x13807;
bool x13820 = x13819 == 1024;
if (x13820) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13827 = x13807;
bool x13829 = x13748 == 1;
int32_t x13828 = 1024 / x13827;
bool x13830 = x13828 == 1;
bool x13834;
if (x454) {
bool x13831 = x13829 || x13830;
bool x13832 = x13748 == x13828;
bool x13833 = x13831 || x13832;
x13834 = x13833;
} else {
x13834 = false;
}
bool x13838;
if (x13834) {
x13838 = x13837;
} else {
x13838 = false;
}
bool x13839;
if (x13838) {
x13839 = x13837;
} else {
x13839 = false;
}
if (x13839) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13748,x13750,x13750,1,x13828,1,1);
assert(false && "");
}
bool x13845 = x13748 <= x13828;
int32_t x13846;
if (x13845) {
x13846 = x13828;
} else {
x13846 = x13748;
}
int32_t x13850 = x13846 * x13849;
int32_t x13851 = 64 * x13850;
float* x13852 = (float*)myMalloc(x13851 * sizeof(float));;
int32_t x13853;
if (x13829) {
x13853 = 0;
} else {
x13853 = x13751;
}
int32_t x13856;
if (x13830) {
x13856 = 0;
} else {
x13856 = 1;
}
for(int x13857=0; x13857 < 64; x13857++) {
int32_t x13869 = x13752 * x13857;
int32_t x13863 = x13850 * x13857;
for(int x13859=0; x13859 < x13846; x13859++) {
int32_t x13870 = x13853 * x13859;
int32_t x13871 = x13869 + x13870;
int32_t x13876 = x13856 * x13859;
int32_t x13865 = x13849 * x13859;
for(int x13861=0; x13861 < x13848; x13861++) {
int32_t x13872 = x13854 * x13861;
int32_t x13873 = x13871 + x13872;
int32_t x13867 = x13848 * x13861;
for(int x13862=0; x13862 < x13848; x13862++) {
int32_t x13874 = x13855 * x13862;
int32_t x13875 = x13873 + x13874;
float x13877 = x13754[x13875];
float x13878 = x13797[x13876];
int32_t x13864 = x13862 + x13863;
int32_t x13866 = x13864 + x13865;
int32_t x13868 = x13866 + x13867;
float x13879 = x13877 / x13878;
x13852[x13868] = x13879;

}

}

}

}
int32_t x13889 = 0;
int32_t x13890 = 1;
x13890 *= 1;
x13889 += 1;
x13890 *= 1;
x13890 *= 1;
int32_t x13895 = x13889;
bool x13896 = x13895 >= 2;
if (x13896) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13901 = x13895 == 0;
if (x13901) {
int32_t x13902 = x13890;
bool x13903 = x13902 == 1024;
if (x13903) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13910 = x13890;
bool x13912 = x13846 == 1;
int32_t x13911 = 1024 / x13910;
bool x13913 = x13911 == 1;
bool x13917;
if (x454) {
bool x13914 = x13912 || x13913;
bool x13915 = x13846 == x13911;
bool x13916 = x13914 || x13915;
x13917 = x13916;
} else {
x13917 = false;
}
bool x13921;
if (x13917) {
x13921 = x13920;
} else {
x13921 = false;
}
bool x13922;
if (x13921) {
x13922 = x13920;
} else {
x13922 = false;
}
if (x13922) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13846,x13848,x13848,1,x13911,1,1);
assert(false && "");
}
bool x13928 = x13846 <= x13911;
int32_t x13929;
if (x13928) {
x13929 = x13911;
} else {
x13929 = x13846;
}
int32_t x13933 = x13929 * x13932;
int32_t x13934 = 64 * x13933;
float* x13935 = (float*)myMalloc(x13934 * sizeof(float));;
int32_t x13936;
if (x13912) {
x13936 = 0;
} else {
x13936 = x13849;
}
int32_t x13939;
if (x13913) {
x13939 = 0;
} else {
x13939 = 1;
}
for(int x13940=0; x13940 < 64; x13940++) {
int32_t x13952 = x13850 * x13940;
int32_t x13946 = x13933 * x13940;
for(int x13942=0; x13942 < x13929; x13942++) {
int32_t x13953 = x13936 * x13942;
int32_t x13954 = x13952 + x13953;
int32_t x13959 = x13939 * x13942;
int32_t x13948 = x13932 * x13942;
for(int x13944=0; x13944 < x13931; x13944++) {
int32_t x13955 = x13937 * x13944;
int32_t x13956 = x13954 + x13955;
int32_t x13950 = x13931 * x13944;
for(int x13945=0; x13945 < x13931; x13945++) {
int32_t x13957 = x13938 * x13945;
int32_t x13958 = x13956 + x13957;
float x13960 = x13852[x13958];
float x13961 = x20[x13959];
int32_t x13947 = x13945 + x13946;
int32_t x13949 = x13947 + x13948;
int32_t x13951 = x13949 + x13950;
float x13962 = x13960 * x13961;
x13935[x13951] = x13962;

}

}

}

}
int32_t x13972 = 0;
int32_t x13973 = 1;
x13973 *= 1;
x13972 += 1;
x13973 *= 1;
x13973 *= 1;
int32_t x13978 = x13972;
bool x13979 = x13978 >= 2;
if (x13979) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13984 = x13978 == 0;
if (x13984) {
int32_t x13985 = x13973;
bool x13986 = x13985 == 1024;
if (x13986) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13993 = x13973;
bool x13995 = x13929 == 1;
int32_t x13994 = 1024 / x13993;
bool x13996 = x13994 == 1;
bool x14000;
if (x454) {
bool x13997 = x13995 || x13996;
bool x13998 = x13929 == x13994;
bool x13999 = x13997 || x13998;
x14000 = x13999;
} else {
x14000 = false;
}
bool x14004;
if (x14000) {
x14004 = x14003;
} else {
x14004 = false;
}
bool x14005;
if (x14004) {
x14005 = x14003;
} else {
x14005 = false;
}
if (x14005) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13929,x13931,x13931,1,x13994,1,1);
assert(false && "");
}
bool x14011 = x13929 <= x13994;
int32_t x14012;
if (x14011) {
x14012 = x13994;
} else {
x14012 = x13929;
}
int32_t x14016 = x14012 * x14015;
int32_t x14017 = 64 * x14016;
float* x14018 = (float*)myMalloc(x14017 * sizeof(float));;
int32_t x14019;
if (x13995) {
x14019 = 0;
} else {
x14019 = x13932;
}
int32_t x14022;
if (x13996) {
x14022 = 0;
} else {
x14022 = 1;
}
for(int x14023=0; x14023 < 64; x14023++) {
int32_t x14035 = x13933 * x14023;
int32_t x14029 = x14016 * x14023;
for(int x14025=0; x14025 < x14012; x14025++) {
int32_t x14036 = x14019 * x14025;
int32_t x14037 = x14035 + x14036;
int32_t x14042 = x14022 * x14025;
int32_t x14031 = x14015 * x14025;
for(int x14027=0; x14027 < x14014; x14027++) {
int32_t x14038 = x14020 * x14027;
int32_t x14039 = x14037 + x14038;
int32_t x14033 = x14014 * x14027;
for(int x14028=0; x14028 < x14014; x14028++) {
int32_t x14040 = x14021 * x14028;
int32_t x14041 = x14039 + x14040;
float x14043 = x13935[x14041];
float x14044 = x232[x14042];
int32_t x14030 = x14028 + x14029;
int32_t x14032 = x14030 + x14031;
int32_t x14034 = x14032 + x14033;
float x14045 = x14043 + x14044;
x14018[x14034] = x14045;

}

}

}

}
bool x14055 = x14012 == 1;
bool x14056 = x14055 || x12708;
bool x14057 = x14012 == x12261;
bool x14058 = x14056 || x14057;
bool x14063;
if (x14058) {
x14063 = x14062;
} else {
x14063 = false;
}
bool x14064;
if (x14063) {
x14064 = x14062;
} else {
x14064 = false;
}
if (x14064) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14012,x14014,x14014,64,x12261,x12263,x12263);
assert(false && "");
}
bool x14070 = x14012 <= x12261;
int32_t x14071;
if (x14070) {
x14071 = x12261;
} else {
x14071 = x14012;
}
int32_t x14077;
if (x14055) {
x14077 = 0;
} else {
x14077 = x14015;
}
for(int x14080=0; x14080 < 64; x14080++) {
int32_t x14086 = x14016 * x14080;
int32_t x14093 = x12265 * x14080;
for(int x14082=0; x14082 < x14071; x14082++) {
int32_t x14087 = x14077 * x14082;
int32_t x14088 = x14086 + x14087;
int32_t x14094 = x12732 * x14082;
int32_t x14095 = x14093 + x14094;
for(int x14084=0; x14084 < x14073; x14084++) {
int32_t x14089 = x14078 * x14084;
int32_t x14090 = x14088 + x14089;
int32_t x14096 = x12733 * x14084;
int32_t x14097 = x14095 + x14096;
for(int x14085=0; x14085 < x14073; x14085++) {
int32_t x14091 = x14079 * x14085;
int32_t x14092 = x14090 + x14091;
float x14100 = x14018[x14092];
int32_t x14098 = x12734 * x14085;
int32_t x14099 = x14097 + x14098;
float x14101 = x12770[x14099];
float x14102 = x14100 + x14101;
x14018[x14092] = x14102;

}

}

}

}
float* x14112 = (float*)myMalloc(x14017 * sizeof(float));;
for(int x14114=0; x14114 < x14017; x14114++) {
float x14115 = x14018[x14114];
bool x14116 = x14115 < 0.0f;
if (x14116) {
x14112[x14114] = 0.0f;
} else {
float x14119 = x14018[x14114];
x14112[x14114] = x14119;
}

}
float* x14133 = (float*)myMalloc(x14132 * sizeof(float));;
int32_t x14136 = 64 * x14012;
int32_t x14137 = x14136 * x14128;
float* x14138 = (float*)myMalloc(x14137 * sizeof(float));;
int32_t x14134 = x14012 * x14128;
for(int x14139=0; x14139 < 64; x14139++) {
int32_t x14140 = x14139 * x14016;
float* x14141 = x14112+x14140;
int32_t x14142 = x14139 * x14129;
float* x14143 = x14133+x14142;
int32_t x14144 = x14139 * x14134;
float* x14145 = x14138+x14144;
for(int x14146=0; x14146 < x14012; x14146++) {
int32_t x14147 = x14146 / 1;
int32_t x14151 = x14147 * x14127;
int32_t x14152 = x14151 * x14127;
int32_t x14148 = x14146 % 1;
int32_t x14149 = x14148 / 1;
int32_t x14153 = x14149 * x14127;
int32_t x14154 = x14153 * x14127;
int32_t x14155 = x14152 + x14154;
int32_t x14150 = x14148 % 1;
int32_t x14156 = x14150 * x14127;
int32_t x14157 = x14156 * x14127;
int32_t x14158 = x14155 + x14157;
float* x14159 = x14145+x14158;
int32_t x14160 = x14147 * x14014;
int32_t x14161 = x14160 * x14014;
float* x14162 = x14141+x14161;
for(int x14164=0; x14164 < x14127; x14164++) {
int32_t x14166 = x14164 * x14127;
float* x14167 = x14159+x14166;
int32_t x14165 = x14164 + x14149;
int32_t x14168 = x14165 * x14014;
int32_t x14169 = x14168 + x14150;
float* x14170 = x14162+x14169;
memcpy(x14167, x14170, 4 * x14127);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x14128,x14012,1,x218,x14012,x14145,x14128,1,x14143,x14128);

}
int32_t x14179 = 0;
int32_t x14180 = 1;
x14180 *= 1;
x14179 += 1;
x14180 *= 1;
x14180 *= 1;
int32_t x14185 = x14179;
bool x14186 = x14185 >= 2;
if (x14186) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14191 = x14185 == 0;
if (x14191) {
int32_t x14192 = x14180;
bool x14193 = x14192 == 256;
if (x14193) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14200 = x14180;
int32_t x14201 = 256 / x14200;
bool x14202 = x14201 == 1;
bool x14205;
if (x454) {
bool x14203 = 256 == x14201;
bool x14204 = x14202 || x14203;
x14205 = x14204;
} else {
x14205 = false;
}
bool x14209;
if (x14205) {
x14209 = x14208;
} else {
x14209 = false;
}
bool x14210;
if (x14209) {
x14210 = x14208;
} else {
x14210 = false;
}
if (x14210) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x14127,x14127,1,x14201,1,1);
assert(false && "");
}
bool x14216 = 256 <= x14201;
int32_t x14217;
if (x14216) {
x14217 = x14201;
} else {
x14217 = 256;
}
int32_t x14221 = x14217 * x14220;
int32_t x14222 = 64 * x14221;
float* x14223 = (float*)myMalloc(x14222 * sizeof(float));;
int32_t x14226;
if (x14202) {
x14226 = 0;
} else {
x14226 = 1;
}
for(int x14227=0; x14227 < 64; x14227++) {
int32_t x14239 = x14129 * x14227;
int32_t x14233 = x14221 * x14227;
for(int x14229=0; x14229 < x14217; x14229++) {
int32_t x14240 = x14128 * x14229;
int32_t x14241 = x14239 + x14240;
int32_t x14246 = x14226 * x14229;
int32_t x14235 = x14220 * x14229;
for(int x14231=0; x14231 < x14219; x14231++) {
int32_t x14242 = x14224 * x14231;
int32_t x14243 = x14241 + x14242;
int32_t x14237 = x14219 * x14231;
for(int x14232=0; x14232 < x14219; x14232++) {
int32_t x14244 = x14225 * x14232;
int32_t x14245 = x14243 + x14244;
float x14247 = x14133[x14245];
float x14248 = x178[x14246];
int32_t x14234 = x14232 + x14233;
int32_t x14236 = x14234 + x14235;
int32_t x14238 = x14236 + x14237;
float x14249 = x14247 - x14248;
x14223[x14238] = x14249;

}

}

}

}
float* x14259 = (float*)myMalloc(256 * sizeof(float));;
for(int x14260=0; x14260 < 256; x14260++) {
float x14261 = x174[x14260];
float x14262 = x14261 + 1.0E-5f;
x14259[x14260] = x14262;

}
float* x14266 = (float*)myMalloc(256 * sizeof(float));;
for(int x14267=0; x14267 < 256; x14267++) {
float x14268 = x14259[x14267];
double x14269 = (double)x14268;
double x14270 = sqrt(x14269);
float x14271 = (float)x14270;
x14266[x14267] = x14271;

}
int32_t x14275 = 0;
int32_t x14276 = 1;
x14276 *= 1;
x14275 += 1;
x14276 *= 1;
x14276 *= 1;
int32_t x14281 = x14275;
bool x14282 = x14281 >= 2;
if (x14282) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14287 = x14281 == 0;
if (x14287) {
int32_t x14288 = x14276;
bool x14289 = x14288 == 256;
if (x14289) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14296 = x14276;
bool x14298 = x14217 == 1;
int32_t x14297 = 256 / x14296;
bool x14299 = x14297 == 1;
bool x14303;
if (x454) {
bool x14300 = x14298 || x14299;
bool x14301 = x14217 == x14297;
bool x14302 = x14300 || x14301;
x14303 = x14302;
} else {
x14303 = false;
}
bool x14307;
if (x14303) {
x14307 = x14306;
} else {
x14307 = false;
}
bool x14308;
if (x14307) {
x14308 = x14306;
} else {
x14308 = false;
}
if (x14308) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14217,x14219,x14219,1,x14297,1,1);
assert(false && "");
}
bool x14314 = x14217 <= x14297;
int32_t x14315;
if (x14314) {
x14315 = x14297;
} else {
x14315 = x14217;
}
int32_t x14319 = x14315 * x14318;
int32_t x14320 = 64 * x14319;
float* x14321 = (float*)myMalloc(x14320 * sizeof(float));;
int32_t x14322;
if (x14298) {
x14322 = 0;
} else {
x14322 = x14220;
}
int32_t x14325;
if (x14299) {
x14325 = 0;
} else {
x14325 = 1;
}
for(int x14326=0; x14326 < 64; x14326++) {
int32_t x14338 = x14221 * x14326;
int32_t x14332 = x14319 * x14326;
for(int x14328=0; x14328 < x14315; x14328++) {
int32_t x14339 = x14322 * x14328;
int32_t x14340 = x14338 + x14339;
int32_t x14345 = x14325 * x14328;
int32_t x14334 = x14318 * x14328;
for(int x14330=0; x14330 < x14317; x14330++) {
int32_t x14341 = x14323 * x14330;
int32_t x14342 = x14340 + x14341;
int32_t x14336 = x14317 * x14330;
for(int x14331=0; x14331 < x14317; x14331++) {
int32_t x14343 = x14324 * x14331;
int32_t x14344 = x14342 + x14343;
float x14346 = x14223[x14344];
float x14347 = x14266[x14345];
int32_t x14333 = x14331 + x14332;
int32_t x14335 = x14333 + x14334;
int32_t x14337 = x14335 + x14336;
float x14348 = x14346 / x14347;
x14321[x14337] = x14348;

}

}

}

}
int32_t x14358 = 0;
int32_t x14359 = 1;
x14359 *= 1;
x14358 += 1;
x14359 *= 1;
x14359 *= 1;
int32_t x14364 = x14358;
bool x14365 = x14364 >= 2;
if (x14365) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14370 = x14364 == 0;
if (x14370) {
int32_t x14371 = x14359;
bool x14372 = x14371 == 256;
if (x14372) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14379 = x14359;
bool x14381 = x14315 == 1;
int32_t x14380 = 256 / x14379;
bool x14382 = x14380 == 1;
bool x14386;
if (x454) {
bool x14383 = x14381 || x14382;
bool x14384 = x14315 == x14380;
bool x14385 = x14383 || x14384;
x14386 = x14385;
} else {
x14386 = false;
}
bool x14390;
if (x14386) {
x14390 = x14389;
} else {
x14390 = false;
}
bool x14391;
if (x14390) {
x14391 = x14389;
} else {
x14391 = false;
}
if (x14391) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14315,x14317,x14317,1,x14380,1,1);
assert(false && "");
}
bool x14397 = x14315 <= x14380;
int32_t x14398;
if (x14397) {
x14398 = x14380;
} else {
x14398 = x14315;
}
int32_t x14402 = x14398 * x14401;
int32_t x14403 = 64 * x14402;
float* x14404 = (float*)myMalloc(x14403 * sizeof(float));;
int32_t x14405;
if (x14381) {
x14405 = 0;
} else {
x14405 = x14318;
}
int32_t x14408;
if (x14382) {
x14408 = 0;
} else {
x14408 = 1;
}
for(int x14409=0; x14409 < 64; x14409++) {
int32_t x14421 = x14319 * x14409;
int32_t x14415 = x14402 * x14409;
for(int x14411=0; x14411 < x14398; x14411++) {
int32_t x14422 = x14405 * x14411;
int32_t x14423 = x14421 + x14422;
int32_t x14428 = x14408 * x14411;
int32_t x14417 = x14401 * x14411;
for(int x14413=0; x14413 < x14400; x14413++) {
int32_t x14424 = x14406 * x14413;
int32_t x14425 = x14423 + x14424;
int32_t x14419 = x14400 * x14413;
for(int x14414=0; x14414 < x14400; x14414++) {
int32_t x14426 = x14407 * x14414;
int32_t x14427 = x14425 + x14426;
float x14429 = x14321[x14427];
float x14430 = x129[x14428];
int32_t x14416 = x14414 + x14415;
int32_t x14418 = x14416 + x14417;
int32_t x14420 = x14418 + x14419;
float x14431 = x14429 * x14430;
x14404[x14420] = x14431;

}

}

}

}
int32_t x14441 = 0;
int32_t x14442 = 1;
x14442 *= 1;
x14441 += 1;
x14442 *= 1;
x14442 *= 1;
int32_t x14447 = x14441;
bool x14448 = x14447 >= 2;
if (x14448) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14453 = x14447 == 0;
if (x14453) {
int32_t x14454 = x14442;
bool x14455 = x14454 == 256;
if (x14455) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14462 = x14442;
bool x14464 = x14398 == 1;
int32_t x14463 = 256 / x14462;
bool x14465 = x14463 == 1;
bool x14469;
if (x454) {
bool x14466 = x14464 || x14465;
bool x14467 = x14398 == x14463;
bool x14468 = x14466 || x14467;
x14469 = x14468;
} else {
x14469 = false;
}
bool x14473;
if (x14469) {
x14473 = x14472;
} else {
x14473 = false;
}
bool x14474;
if (x14473) {
x14474 = x14472;
} else {
x14474 = false;
}
if (x14474) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14398,x14400,x14400,1,x14463,1,1);
assert(false && "");
}
bool x14480 = x14398 <= x14463;
int32_t x14481;
if (x14480) {
x14481 = x14463;
} else {
x14481 = x14398;
}
int32_t x14485 = x14481 * x14484;
int32_t x14486 = 64 * x14485;
float* x14487 = (float*)myMalloc(x14486 * sizeof(float));;
int32_t x14488;
if (x14464) {
x14488 = 0;
} else {
x14488 = x14401;
}
int32_t x14491;
if (x14465) {
x14491 = 0;
} else {
x14491 = 1;
}
for(int x14492=0; x14492 < 64; x14492++) {
int32_t x14504 = x14402 * x14492;
int32_t x14498 = x14485 * x14492;
for(int x14494=0; x14494 < x14481; x14494++) {
int32_t x14505 = x14488 * x14494;
int32_t x14506 = x14504 + x14505;
int32_t x14511 = x14491 * x14494;
int32_t x14500 = x14484 * x14494;
for(int x14496=0; x14496 < x14483; x14496++) {
int32_t x14507 = x14489 * x14496;
int32_t x14508 = x14506 + x14507;
int32_t x14502 = x14483 * x14496;
for(int x14497=0; x14497 < x14483; x14497++) {
int32_t x14509 = x14490 * x14497;
int32_t x14510 = x14508 + x14509;
float x14512 = x14404[x14510];
float x14513 = x197[x14511];
int32_t x14499 = x14497 + x14498;
int32_t x14501 = x14499 + x14500;
int32_t x14503 = x14501 + x14502;
float x14514 = x14512 + x14513;
x14487[x14503] = x14514;

}

}

}

}
float* x14524 = (float*)myMalloc(x14486 * sizeof(float));;
for(int x14526=0; x14526 < x14486; x14526++) {
float x14527 = x14487[x14526];
bool x14528 = x14527 < 0.0f;
if (x14528) {
x14524[x14526] = 0.0f;
} else {
float x14531 = x14487[x14526];
x14524[x14526] = x14531;
}

}
float* x14546 = (float*)myMalloc(x14545 * sizeof(float));;
int32_t x14547 = 9 * x14481;
int32_t x14550 = 64 * x14547;
int32_t x14551 = x14550 * x14541;
float* x14552 = (float*)myMalloc(x14551 * sizeof(float));;
int32_t x14548 = x14547 * x14541;
int32_t x14560 = x14481 * 3;
int32_t x14561 = x14560 * 3;
for(int x14553=0; x14553 < 64; x14553++) {
int32_t x14554 = x14553 * x14485;
float* x14555 = x14524+x14554;
int32_t x14556 = x14553 * x14542;
float* x14557 = x14546+x14556;
int32_t x14558 = x14553 * x14548;
float* x14559 = x14552+x14558;
for(int x14563=0; x14563 < x14561; x14563++) {
int32_t x14564 = x14563 / 9;
int32_t x14568 = x14564 * 3;
int32_t x14569 = x14568 * 3;
int32_t x14570 = x14569 * x14540;
int32_t x14571 = x14570 * x14540;
int32_t x14565 = x14563 % 9;
int32_t x14566 = x14565 / 3;
int32_t x14572 = x14566 * 3;
int32_t x14573 = x14572 * x14540;
int32_t x14574 = x14573 * x14540;
int32_t x14575 = x14571 + x14574;
int32_t x14567 = x14565 % 3;
int32_t x14576 = x14567 * x14540;
int32_t x14577 = x14576 * x14540;
int32_t x14578 = x14575 + x14577;
float* x14579 = x14559+x14578;
int32_t x14580 = x14564 * x14483;
int32_t x14581 = x14580 * x14483;
float* x14582 = x14555+x14581;
int32_t x14595 = 1 - x14567;
bool x14596 = x14595 > 0;
int32_t x14597;
if (x14596) {
x14597 = x14595;
} else {
x14597 = 0;
}
int32_t x14598 = 3 - x14567;
int32_t x14599 = x14598 - 1;
int32_t x14600 = 1 - x14599;
bool x14601 = x14600 > 0;
int32_t x14602;
if (x14601) {
x14602 = x14600;
} else {
x14602 = 0;
}
int32_t x14603 = x14540 - x14602;
int32_t x14604 = x14603 - x14597;
bool x14605 = x14604 <= 0;
bool x14609 = x14597 > 0;
int32_t x14594 = -1 + x14567;
bool x14622 = x14602 > 0;
for(int x14584=0; x14584 < x14540; x14584++) {
int32_t x14585 = x14584 - 1;
int32_t x14586 = x14585 + x14566;
bool x14587 = x14586 < 0;
bool x14588 = x14586 >= x14483;
bool x14589 = x14587 || x14588;
if (x14589) {
int32_t x14590 = x14584 * x14540;
float* x14591 = x14579+x14590;
memset(x14591, 0, 4 * x14540);;
} else {
if (x14605) {
int32_t x14590 = x14584 * x14540;
float* x14606 = x14579+x14590;
memset(x14606, 0, 4 * x14540);;
} else {
int32_t x14590 = x14584 * x14540;
if (x14609) {
float* x14610 = x14579+x14590;
memset(x14610, 0, 4 * x14597);;
} else {
}
// may have segfault here
int32_t x14615 = x14590 + x14597;
float* x14616 = x14579+x14615;
int32_t x14617 = x14586 * x14483;
int32_t x14618 = x14617 + x14594;
int32_t x14619 = x14618 + x14597;
float* x14620 = x14582+x14619;
memcpy(x14616, x14620, 4 * x14604);;
if (x14622) {
int32_t x14623 = x14590 + x14540;
int32_t x14624 = x14623 - x14602;
float* x14625 = x14579+x14624;
memset(x14625, 0, 4 * x14602);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x14541,x14547,1,x14,x14547,x14559,x14541,1,x14557,x14541);

}
int32_t x14640 = 0;
int32_t x14641 = 1;
x14641 *= 1;
x14640 += 1;
x14641 *= 1;
x14641 *= 1;
int32_t x14646 = x14640;
bool x14647 = x14646 >= 2;
if (x14647) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14652 = x14646 == 0;
if (x14652) {
int32_t x14653 = x14641;
bool x14654 = x14653 == 256;
if (x14654) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14661 = x14641;
int32_t x14662 = 256 / x14661;
bool x14663 = x14662 == 1;
bool x14666;
if (x454) {
bool x14664 = 256 == x14662;
bool x14665 = x14663 || x14664;
x14666 = x14665;
} else {
x14666 = false;
}
bool x14670;
if (x14666) {
x14670 = x14669;
} else {
x14670 = false;
}
bool x14671;
if (x14670) {
x14671 = x14669;
} else {
x14671 = false;
}
if (x14671) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x14540,x14540,1,x14662,1,1);
assert(false && "");
}
bool x14677 = 256 <= x14662;
int32_t x14678;
if (x14677) {
x14678 = x14662;
} else {
x14678 = 256;
}
int32_t x14682 = x14678 * x14681;
int32_t x14683 = 64 * x14682;
float* x14684 = (float*)myMalloc(x14683 * sizeof(float));;
int32_t x14687;
if (x14663) {
x14687 = 0;
} else {
x14687 = 1;
}
for(int x14688=0; x14688 < 64; x14688++) {
int32_t x14700 = x14542 * x14688;
int32_t x14694 = x14682 * x14688;
for(int x14690=0; x14690 < x14678; x14690++) {
int32_t x14701 = x14541 * x14690;
int32_t x14702 = x14700 + x14701;
int32_t x14707 = x14687 * x14690;
int32_t x14696 = x14681 * x14690;
for(int x14692=0; x14692 < x14680; x14692++) {
int32_t x14703 = x14685 * x14692;
int32_t x14704 = x14702 + x14703;
int32_t x14698 = x14680 * x14692;
for(int x14693=0; x14693 < x14680; x14693++) {
int32_t x14705 = x14686 * x14693;
int32_t x14706 = x14704 + x14705;
float x14708 = x14546[x14706];
float x14709 = x124[x14707];
int32_t x14695 = x14693 + x14694;
int32_t x14697 = x14695 + x14696;
int32_t x14699 = x14697 + x14698;
float x14710 = x14708 - x14709;
x14684[x14699] = x14710;

}

}

}

}
float* x14720 = (float*)myMalloc(256 * sizeof(float));;
for(int x14721=0; x14721 < 256; x14721++) {
float x14722 = x63[x14721];
float x14723 = x14722 + 1.0E-5f;
x14720[x14721] = x14723;

}
float* x14727 = (float*)myMalloc(256 * sizeof(float));;
for(int x14728=0; x14728 < 256; x14728++) {
float x14729 = x14720[x14728];
double x14730 = (double)x14729;
double x14731 = sqrt(x14730);
float x14732 = (float)x14731;
x14727[x14728] = x14732;

}
int32_t x14736 = 0;
int32_t x14737 = 1;
x14737 *= 1;
x14736 += 1;
x14737 *= 1;
x14737 *= 1;
int32_t x14742 = x14736;
bool x14743 = x14742 >= 2;
if (x14743) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14748 = x14742 == 0;
if (x14748) {
int32_t x14749 = x14737;
bool x14750 = x14749 == 256;
if (x14750) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14757 = x14737;
bool x14759 = x14678 == 1;
int32_t x14758 = 256 / x14757;
bool x14760 = x14758 == 1;
bool x14764;
if (x454) {
bool x14761 = x14759 || x14760;
bool x14762 = x14678 == x14758;
bool x14763 = x14761 || x14762;
x14764 = x14763;
} else {
x14764 = false;
}
bool x14768;
if (x14764) {
x14768 = x14767;
} else {
x14768 = false;
}
bool x14769;
if (x14768) {
x14769 = x14767;
} else {
x14769 = false;
}
if (x14769) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14678,x14680,x14680,1,x14758,1,1);
assert(false && "");
}
bool x14775 = x14678 <= x14758;
int32_t x14776;
if (x14775) {
x14776 = x14758;
} else {
x14776 = x14678;
}
int32_t x14780 = x14776 * x14779;
int32_t x14781 = 64 * x14780;
float* x14782 = (float*)myMalloc(x14781 * sizeof(float));;
int32_t x14783;
if (x14759) {
x14783 = 0;
} else {
x14783 = x14681;
}
int32_t x14786;
if (x14760) {
x14786 = 0;
} else {
x14786 = 1;
}
for(int x14787=0; x14787 < 64; x14787++) {
int32_t x14799 = x14682 * x14787;
int32_t x14793 = x14780 * x14787;
for(int x14789=0; x14789 < x14776; x14789++) {
int32_t x14800 = x14783 * x14789;
int32_t x14801 = x14799 + x14800;
int32_t x14806 = x14786 * x14789;
int32_t x14795 = x14779 * x14789;
for(int x14791=0; x14791 < x14778; x14791++) {
int32_t x14802 = x14784 * x14791;
int32_t x14803 = x14801 + x14802;
int32_t x14797 = x14778 * x14791;
for(int x14792=0; x14792 < x14778; x14792++) {
int32_t x14804 = x14785 * x14792;
int32_t x14805 = x14803 + x14804;
float x14807 = x14684[x14805];
float x14808 = x14727[x14806];
int32_t x14794 = x14792 + x14793;
int32_t x14796 = x14794 + x14795;
int32_t x14798 = x14796 + x14797;
float x14809 = x14807 / x14808;
x14782[x14798] = x14809;

}

}

}

}
int32_t x14819 = 0;
int32_t x14820 = 1;
x14820 *= 1;
x14819 += 1;
x14820 *= 1;
x14820 *= 1;
int32_t x14825 = x14819;
bool x14826 = x14825 >= 2;
if (x14826) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14831 = x14825 == 0;
if (x14831) {
int32_t x14832 = x14820;
bool x14833 = x14832 == 256;
if (x14833) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14840 = x14820;
bool x14842 = x14776 == 1;
int32_t x14841 = 256 / x14840;
bool x14843 = x14841 == 1;
bool x14847;
if (x454) {
bool x14844 = x14842 || x14843;
bool x14845 = x14776 == x14841;
bool x14846 = x14844 || x14845;
x14847 = x14846;
} else {
x14847 = false;
}
bool x14851;
if (x14847) {
x14851 = x14850;
} else {
x14851 = false;
}
bool x14852;
if (x14851) {
x14852 = x14850;
} else {
x14852 = false;
}
if (x14852) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14776,x14778,x14778,1,x14841,1,1);
assert(false && "");
}
bool x14858 = x14776 <= x14841;
int32_t x14859;
if (x14858) {
x14859 = x14841;
} else {
x14859 = x14776;
}
int32_t x14863 = x14859 * x14862;
int32_t x14864 = 64 * x14863;
float* x14865 = (float*)myMalloc(x14864 * sizeof(float));;
int32_t x14866;
if (x14842) {
x14866 = 0;
} else {
x14866 = x14779;
}
int32_t x14869;
if (x14843) {
x14869 = 0;
} else {
x14869 = 1;
}
for(int x14870=0; x14870 < 64; x14870++) {
int32_t x14882 = x14780 * x14870;
int32_t x14876 = x14863 * x14870;
for(int x14872=0; x14872 < x14859; x14872++) {
int32_t x14883 = x14866 * x14872;
int32_t x14884 = x14882 + x14883;
int32_t x14889 = x14869 * x14872;
int32_t x14878 = x14862 * x14872;
for(int x14874=0; x14874 < x14861; x14874++) {
int32_t x14885 = x14867 * x14874;
int32_t x14886 = x14884 + x14885;
int32_t x14880 = x14861 * x14874;
for(int x14875=0; x14875 < x14861; x14875++) {
int32_t x14887 = x14868 * x14875;
int32_t x14888 = x14886 + x14887;
float x14890 = x14782[x14888];
float x14891 = x228[x14889];
int32_t x14877 = x14875 + x14876;
int32_t x14879 = x14877 + x14878;
int32_t x14881 = x14879 + x14880;
float x14892 = x14890 * x14891;
x14865[x14881] = x14892;

}

}

}

}
int32_t x14902 = 0;
int32_t x14903 = 1;
x14903 *= 1;
x14902 += 1;
x14903 *= 1;
x14903 *= 1;
int32_t x14908 = x14902;
bool x14909 = x14908 >= 2;
if (x14909) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14914 = x14908 == 0;
if (x14914) {
int32_t x14915 = x14903;
bool x14916 = x14915 == 256;
if (x14916) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14923 = x14903;
bool x14925 = x14859 == 1;
int32_t x14924 = 256 / x14923;
bool x14926 = x14924 == 1;
bool x14930;
if (x454) {
bool x14927 = x14925 || x14926;
bool x14928 = x14859 == x14924;
bool x14929 = x14927 || x14928;
x14930 = x14929;
} else {
x14930 = false;
}
bool x14934;
if (x14930) {
x14934 = x14933;
} else {
x14934 = false;
}
bool x14935;
if (x14934) {
x14935 = x14933;
} else {
x14935 = false;
}
if (x14935) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14859,x14861,x14861,1,x14924,1,1);
assert(false && "");
}
bool x14941 = x14859 <= x14924;
int32_t x14942;
if (x14941) {
x14942 = x14924;
} else {
x14942 = x14859;
}
int32_t x14946 = x14942 * x14945;
int32_t x14947 = 64 * x14946;
float* x14948 = (float*)myMalloc(x14947 * sizeof(float));;
int32_t x14949;
if (x14925) {
x14949 = 0;
} else {
x14949 = x14862;
}
int32_t x14952;
if (x14926) {
x14952 = 0;
} else {
x14952 = 1;
}
for(int x14953=0; x14953 < 64; x14953++) {
int32_t x14965 = x14863 * x14953;
int32_t x14959 = x14946 * x14953;
for(int x14955=0; x14955 < x14942; x14955++) {
int32_t x14966 = x14949 * x14955;
int32_t x14967 = x14965 + x14966;
int32_t x14972 = x14952 * x14955;
int32_t x14961 = x14945 * x14955;
for(int x14957=0; x14957 < x14944; x14957++) {
int32_t x14968 = x14950 * x14957;
int32_t x14969 = x14967 + x14968;
int32_t x14963 = x14944 * x14957;
for(int x14958=0; x14958 < x14944; x14958++) {
int32_t x14970 = x14951 * x14958;
int32_t x14971 = x14969 + x14970;
float x14973 = x14865[x14971];
float x14974 = x192[x14972];
int32_t x14960 = x14958 + x14959;
int32_t x14962 = x14960 + x14961;
int32_t x14964 = x14962 + x14963;
float x14975 = x14973 + x14974;
x14948[x14964] = x14975;

}

}

}

}
float* x14985 = (float*)myMalloc(x14947 * sizeof(float));;
for(int x14987=0; x14987 < x14947; x14987++) {
float x14988 = x14948[x14987];
bool x14989 = x14988 < 0.0f;
if (x14989) {
x14985[x14987] = 0.0f;
} else {
float x14992 = x14948[x14987];
x14985[x14987] = x14992;
}

}
float* x15006 = (float*)myMalloc(x15005 * sizeof(float));;
int32_t x15009 = 64 * x14942;
int32_t x15010 = x15009 * x15001;
float* x15011 = (float*)myMalloc(x15010 * sizeof(float));;
int32_t x15007 = x14942 * x15001;
for(int x15012=0; x15012 < 64; x15012++) {
int32_t x15013 = x15012 * x14946;
float* x15014 = x14985+x15013;
int32_t x15015 = x15012 * x15002;
float* x15016 = x15006+x15015;
int32_t x15017 = x15012 * x15007;
float* x15018 = x15011+x15017;
for(int x15019=0; x15019 < x14942; x15019++) {
int32_t x15020 = x15019 / 1;
int32_t x15024 = x15020 * x15000;
int32_t x15025 = x15024 * x15000;
int32_t x15021 = x15019 % 1;
int32_t x15022 = x15021 / 1;
int32_t x15026 = x15022 * x15000;
int32_t x15027 = x15026 * x15000;
int32_t x15028 = x15025 + x15027;
int32_t x15023 = x15021 % 1;
int32_t x15029 = x15023 * x15000;
int32_t x15030 = x15029 * x15000;
int32_t x15031 = x15028 + x15030;
float* x15032 = x15018+x15031;
int32_t x15033 = x15020 * x14944;
int32_t x15034 = x15033 * x14944;
float* x15035 = x15014+x15034;
for(int x15037=0; x15037 < x15000; x15037++) {
int32_t x15039 = x15037 * x15000;
float* x15040 = x15032+x15039;
int32_t x15038 = x15037 + x15022;
int32_t x15041 = x15038 * x14944;
int32_t x15042 = x15041 + x15023;
float* x15043 = x15035+x15042;
memcpy(x15040, x15043, 4 * x15000);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x15001,x14942,1,x116,x14942,x15018,x15001,1,x15016,x15001);

}
int32_t x15052 = 0;
int32_t x15053 = 1;
x15053 *= 1;
x15052 += 1;
x15053 *= 1;
x15053 *= 1;
int32_t x15058 = x15052;
bool x15059 = x15058 >= 2;
if (x15059) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15064 = x15058 == 0;
if (x15064) {
int32_t x15065 = x15053;
bool x15066 = x15065 == 1024;
if (x15066) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15073 = x15053;
int32_t x15074 = 1024 / x15073;
bool x15075 = x15074 == 1;
bool x15078;
if (x454) {
bool x15076 = 1024 == x15074;
bool x15077 = x15075 || x15076;
x15078 = x15077;
} else {
x15078 = false;
}
bool x15082;
if (x15078) {
x15082 = x15081;
} else {
x15082 = false;
}
bool x15083;
if (x15082) {
x15083 = x15081;
} else {
x15083 = false;
}
if (x15083) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x15000,x15000,1,x15074,1,1);
assert(false && "");
}
bool x15089 = 1024 <= x15074;
int32_t x15090;
if (x15089) {
x15090 = x15074;
} else {
x15090 = 1024;
}
int32_t x15094 = x15090 * x15093;
int32_t x15095 = 64 * x15094;
float* x15096 = (float*)myMalloc(x15095 * sizeof(float));;
int32_t x15099;
if (x15075) {
x15099 = 0;
} else {
x15099 = 1;
}
for(int x15100=0; x15100 < 64; x15100++) {
int32_t x15112 = x15002 * x15100;
int32_t x15106 = x15094 * x15100;
for(int x15102=0; x15102 < x15090; x15102++) {
int32_t x15113 = x15001 * x15102;
int32_t x15114 = x15112 + x15113;
int32_t x15119 = x15099 * x15102;
int32_t x15108 = x15093 * x15102;
for(int x15104=0; x15104 < x15092; x15104++) {
int32_t x15115 = x15097 * x15104;
int32_t x15116 = x15114 + x15115;
int32_t x15110 = x15092 * x15104;
for(int x15105=0; x15105 < x15092; x15105++) {
int32_t x15117 = x15098 * x15105;
int32_t x15118 = x15116 + x15117;
float x15120 = x15006[x15118];
float x15121 = x140[x15119];
int32_t x15107 = x15105 + x15106;
int32_t x15109 = x15107 + x15108;
int32_t x15111 = x15109 + x15110;
float x15122 = x15120 - x15121;
x15096[x15111] = x15122;

}

}

}

}
float* x15132 = (float*)myMalloc(1024 * sizeof(float));;
for(int x15133=0; x15133 < 1024; x15133++) {
float x15134 = x188[x15133];
float x15135 = x15134 + 1.0E-5f;
x15132[x15133] = x15135;

}
float* x15139 = (float*)myMalloc(1024 * sizeof(float));;
for(int x15140=0; x15140 < 1024; x15140++) {
float x15141 = x15132[x15140];
double x15142 = (double)x15141;
double x15143 = sqrt(x15142);
float x15144 = (float)x15143;
x15139[x15140] = x15144;

}
int32_t x15148 = 0;
int32_t x15149 = 1;
x15149 *= 1;
x15148 += 1;
x15149 *= 1;
x15149 *= 1;
int32_t x15154 = x15148;
bool x15155 = x15154 >= 2;
if (x15155) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15160 = x15154 == 0;
if (x15160) {
int32_t x15161 = x15149;
bool x15162 = x15161 == 1024;
if (x15162) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15169 = x15149;
bool x15171 = x15090 == 1;
int32_t x15170 = 1024 / x15169;
bool x15172 = x15170 == 1;
bool x15176;
if (x454) {
bool x15173 = x15171 || x15172;
bool x15174 = x15090 == x15170;
bool x15175 = x15173 || x15174;
x15176 = x15175;
} else {
x15176 = false;
}
bool x15180;
if (x15176) {
x15180 = x15179;
} else {
x15180 = false;
}
bool x15181;
if (x15180) {
x15181 = x15179;
} else {
x15181 = false;
}
if (x15181) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15090,x15092,x15092,1,x15170,1,1);
assert(false && "");
}
bool x15187 = x15090 <= x15170;
int32_t x15188;
if (x15187) {
x15188 = x15170;
} else {
x15188 = x15090;
}
int32_t x15192 = x15188 * x15191;
int32_t x15193 = 64 * x15192;
float* x15194 = (float*)myMalloc(x15193 * sizeof(float));;
int32_t x15195;
if (x15171) {
x15195 = 0;
} else {
x15195 = x15093;
}
int32_t x15198;
if (x15172) {
x15198 = 0;
} else {
x15198 = 1;
}
for(int x15199=0; x15199 < 64; x15199++) {
int32_t x15211 = x15094 * x15199;
int32_t x15205 = x15192 * x15199;
for(int x15201=0; x15201 < x15188; x15201++) {
int32_t x15212 = x15195 * x15201;
int32_t x15213 = x15211 + x15212;
int32_t x15218 = x15198 * x15201;
int32_t x15207 = x15191 * x15201;
for(int x15203=0; x15203 < x15190; x15203++) {
int32_t x15214 = x15196 * x15203;
int32_t x15215 = x15213 + x15214;
int32_t x15209 = x15190 * x15203;
for(int x15204=0; x15204 < x15190; x15204++) {
int32_t x15216 = x15197 * x15204;
int32_t x15217 = x15215 + x15216;
float x15219 = x15096[x15217];
float x15220 = x15139[x15218];
int32_t x15206 = x15204 + x15205;
int32_t x15208 = x15206 + x15207;
int32_t x15210 = x15208 + x15209;
float x15221 = x15219 / x15220;
x15194[x15210] = x15221;

}

}

}

}
int32_t x15231 = 0;
int32_t x15232 = 1;
x15232 *= 1;
x15231 += 1;
x15232 *= 1;
x15232 *= 1;
int32_t x15237 = x15231;
bool x15238 = x15237 >= 2;
if (x15238) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15243 = x15237 == 0;
if (x15243) {
int32_t x15244 = x15232;
bool x15245 = x15244 == 1024;
if (x15245) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15252 = x15232;
bool x15254 = x15188 == 1;
int32_t x15253 = 1024 / x15252;
bool x15255 = x15253 == 1;
bool x15259;
if (x454) {
bool x15256 = x15254 || x15255;
bool x15257 = x15188 == x15253;
bool x15258 = x15256 || x15257;
x15259 = x15258;
} else {
x15259 = false;
}
bool x15263;
if (x15259) {
x15263 = x15262;
} else {
x15263 = false;
}
bool x15264;
if (x15263) {
x15264 = x15262;
} else {
x15264 = false;
}
if (x15264) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15188,x15190,x15190,1,x15253,1,1);
assert(false && "");
}
bool x15270 = x15188 <= x15253;
int32_t x15271;
if (x15270) {
x15271 = x15253;
} else {
x15271 = x15188;
}
int32_t x15275 = x15271 * x15274;
int32_t x15276 = 64 * x15275;
float* x15277 = (float*)myMalloc(x15276 * sizeof(float));;
int32_t x15278;
if (x15254) {
x15278 = 0;
} else {
x15278 = x15191;
}
int32_t x15281;
if (x15255) {
x15281 = 0;
} else {
x15281 = 1;
}
for(int x15282=0; x15282 < 64; x15282++) {
int32_t x15294 = x15192 * x15282;
int32_t x15288 = x15275 * x15282;
for(int x15284=0; x15284 < x15271; x15284++) {
int32_t x15295 = x15278 * x15284;
int32_t x15296 = x15294 + x15295;
int32_t x15301 = x15281 * x15284;
int32_t x15290 = x15274 * x15284;
for(int x15286=0; x15286 < x15273; x15286++) {
int32_t x15297 = x15279 * x15286;
int32_t x15298 = x15296 + x15297;
int32_t x15292 = x15273 * x15286;
for(int x15287=0; x15287 < x15273; x15287++) {
int32_t x15299 = x15280 * x15287;
int32_t x15300 = x15298 + x15299;
float x15302 = x15194[x15300];
float x15303 = x263[x15301];
int32_t x15289 = x15287 + x15288;
int32_t x15291 = x15289 + x15290;
int32_t x15293 = x15291 + x15292;
float x15304 = x15302 * x15303;
x15277[x15293] = x15304;

}

}

}

}
int32_t x15314 = 0;
int32_t x15315 = 1;
x15315 *= 1;
x15314 += 1;
x15315 *= 1;
x15315 *= 1;
int32_t x15320 = x15314;
bool x15321 = x15320 >= 2;
if (x15321) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15326 = x15320 == 0;
if (x15326) {
int32_t x15327 = x15315;
bool x15328 = x15327 == 1024;
if (x15328) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15335 = x15315;
bool x15337 = x15271 == 1;
int32_t x15336 = 1024 / x15335;
bool x15338 = x15336 == 1;
bool x15342;
if (x454) {
bool x15339 = x15337 || x15338;
bool x15340 = x15271 == x15336;
bool x15341 = x15339 || x15340;
x15342 = x15341;
} else {
x15342 = false;
}
bool x15346;
if (x15342) {
x15346 = x15345;
} else {
x15346 = false;
}
bool x15347;
if (x15346) {
x15347 = x15345;
} else {
x15347 = false;
}
if (x15347) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15271,x15273,x15273,1,x15336,1,1);
assert(false && "");
}
bool x15353 = x15271 <= x15336;
int32_t x15354;
if (x15353) {
x15354 = x15336;
} else {
x15354 = x15271;
}
int32_t x15358 = x15354 * x15357;
int32_t x15359 = 64 * x15358;
float* x15360 = (float*)myMalloc(x15359 * sizeof(float));;
int32_t x15361;
if (x15337) {
x15361 = 0;
} else {
x15361 = x15274;
}
int32_t x15364;
if (x15338) {
x15364 = 0;
} else {
x15364 = 1;
}
for(int x15365=0; x15365 < 64; x15365++) {
int32_t x15377 = x15275 * x15365;
int32_t x15371 = x15358 * x15365;
for(int x15367=0; x15367 < x15354; x15367++) {
int32_t x15378 = x15361 * x15367;
int32_t x15379 = x15377 + x15378;
int32_t x15384 = x15364 * x15367;
int32_t x15373 = x15357 * x15367;
for(int x15369=0; x15369 < x15356; x15369++) {
int32_t x15380 = x15362 * x15369;
int32_t x15381 = x15379 + x15380;
int32_t x15375 = x15356 * x15369;
for(int x15370=0; x15370 < x15356; x15370++) {
int32_t x15382 = x15363 * x15370;
int32_t x15383 = x15381 + x15382;
float x15385 = x15277[x15383];
float x15386 = x57[x15384];
int32_t x15372 = x15370 + x15371;
int32_t x15374 = x15372 + x15373;
int32_t x15376 = x15374 + x15375;
float x15387 = x15385 + x15386;
x15360[x15376] = x15387;

}

}

}

}
bool x15397 = x15354 == 1;
bool x15398 = x15397 || x14055;
bool x15399 = x15354 == x14012;
bool x15400 = x15398 || x15399;
bool x15405;
if (x15400) {
x15405 = x15404;
} else {
x15405 = false;
}
bool x15406;
if (x15405) {
x15406 = x15404;
} else {
x15406 = false;
}
if (x15406) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15354,x15356,x15356,64,x14012,x14014,x14014);
assert(false && "");
}
bool x15412 = x15354 <= x14012;
int32_t x15413;
if (x15412) {
x15413 = x14012;
} else {
x15413 = x15354;
}
int32_t x15419;
if (x15397) {
x15419 = 0;
} else {
x15419 = x15357;
}
for(int x15422=0; x15422 < 64; x15422++) {
int32_t x15428 = x15358 * x15422;
int32_t x15435 = x14016 * x15422;
for(int x15424=0; x15424 < x15413; x15424++) {
int32_t x15429 = x15419 * x15424;
int32_t x15430 = x15428 + x15429;
int32_t x15436 = x14077 * x15424;
int32_t x15437 = x15435 + x15436;
for(int x15426=0; x15426 < x15415; x15426++) {
int32_t x15431 = x15420 * x15426;
int32_t x15432 = x15430 + x15431;
int32_t x15438 = x14078 * x15426;
int32_t x15439 = x15437 + x15438;
for(int x15427=0; x15427 < x15415; x15427++) {
int32_t x15433 = x15421 * x15427;
int32_t x15434 = x15432 + x15433;
float x15442 = x15360[x15434];
int32_t x15440 = x14079 * x15427;
int32_t x15441 = x15439 + x15440;
float x15443 = x14112[x15441];
float x15444 = x15442 + x15443;
x15360[x15434] = x15444;

}

}

}

}
float* x15454 = (float*)myMalloc(x15359 * sizeof(float));;
for(int x15456=0; x15456 < x15359; x15456++) {
float x15457 = x15360[x15456];
bool x15458 = x15457 < 0.0f;
if (x15458) {
x15454[x15456] = 0.0f;
} else {
float x15461 = x15360[x15456];
x15454[x15456] = x15461;
}

}
float* x15475 = (float*)myMalloc(x15474 * sizeof(float));;
int32_t x15478 = 64 * x15354;
int32_t x15479 = x15478 * x15470;
float* x15480 = (float*)myMalloc(x15479 * sizeof(float));;
int32_t x15476 = x15354 * x15470;
for(int x15481=0; x15481 < 64; x15481++) {
int32_t x15482 = x15481 * x15358;
float* x15483 = x15454+x15482;
int32_t x15484 = x15481 * x15471;
float* x15485 = x15475+x15484;
int32_t x15486 = x15481 * x15476;
float* x15487 = x15480+x15486;
for(int x15488=0; x15488 < x15354; x15488++) {
int32_t x15489 = x15488 / 1;
int32_t x15493 = x15489 * x15469;
int32_t x15494 = x15493 * x15469;
int32_t x15490 = x15488 % 1;
int32_t x15491 = x15490 / 1;
int32_t x15495 = x15491 * x15469;
int32_t x15496 = x15495 * x15469;
int32_t x15497 = x15494 + x15496;
int32_t x15492 = x15490 % 1;
int32_t x15498 = x15492 * x15469;
int32_t x15499 = x15498 * x15469;
int32_t x15500 = x15497 + x15499;
float* x15501 = x15487+x15500;
int32_t x15502 = x15489 * x15356;
int32_t x15503 = x15502 * x15356;
float* x15504 = x15483+x15503;
for(int x15506=0; x15506 < x15469; x15506++) {
int32_t x15508 = x15506 * x15469;
float* x15509 = x15501+x15508;
int32_t x15507 = x15506 + x15491;
int32_t x15510 = x15507 * x15356;
int32_t x15511 = x15510 + x15492;
float* x15512 = x15504+x15511;
memcpy(x15509, x15512, 4 * x15469);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x15470,x15354,1,x6,x15354,x15487,x15470,1,x15485,x15470);

}
int32_t x15521 = 0;
int32_t x15522 = 1;
x15522 *= 1;
x15521 += 1;
x15522 *= 1;
x15522 *= 1;
int32_t x15527 = x15521;
bool x15528 = x15527 >= 2;
if (x15528) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15533 = x15527 == 0;
if (x15533) {
int32_t x15534 = x15522;
bool x15535 = x15534 == 256;
if (x15535) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15542 = x15522;
int32_t x15543 = 256 / x15542;
bool x15544 = x15543 == 1;
bool x15547;
if (x454) {
bool x15545 = 256 == x15543;
bool x15546 = x15544 || x15545;
x15547 = x15546;
} else {
x15547 = false;
}
bool x15551;
if (x15547) {
x15551 = x15550;
} else {
x15551 = false;
}
bool x15552;
if (x15551) {
x15552 = x15550;
} else {
x15552 = false;
}
if (x15552) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x15469,x15469,1,x15543,1,1);
assert(false && "");
}
bool x15558 = 256 <= x15543;
int32_t x15559;
if (x15558) {
x15559 = x15543;
} else {
x15559 = 256;
}
int32_t x15563 = x15559 * x15562;
int32_t x15564 = 64 * x15563;
float* x15565 = (float*)myMalloc(x15564 * sizeof(float));;
int32_t x15568;
if (x15544) {
x15568 = 0;
} else {
x15568 = 1;
}
for(int x15569=0; x15569 < 64; x15569++) {
int32_t x15581 = x15471 * x15569;
int32_t x15575 = x15563 * x15569;
for(int x15571=0; x15571 < x15559; x15571++) {
int32_t x15582 = x15470 * x15571;
int32_t x15583 = x15581 + x15582;
int32_t x15588 = x15568 * x15571;
int32_t x15577 = x15562 * x15571;
for(int x15573=0; x15573 < x15561; x15573++) {
int32_t x15584 = x15566 * x15573;
int32_t x15585 = x15583 + x15584;
int32_t x15579 = x15561 * x15573;
for(int x15574=0; x15574 < x15561; x15574++) {
int32_t x15586 = x15567 * x15574;
int32_t x15587 = x15585 + x15586;
float x15589 = x15475[x15587];
float x15590 = x163[x15588];
int32_t x15576 = x15574 + x15575;
int32_t x15578 = x15576 + x15577;
int32_t x15580 = x15578 + x15579;
float x15591 = x15589 - x15590;
x15565[x15580] = x15591;

}

}

}

}
float* x15601 = (float*)myMalloc(256 * sizeof(float));;
for(int x15602=0; x15602 < 256; x15602++) {
float x15603 = x98[x15602];
float x15604 = x15603 + 1.0E-5f;
x15601[x15602] = x15604;

}
float* x15608 = (float*)myMalloc(256 * sizeof(float));;
for(int x15609=0; x15609 < 256; x15609++) {
float x15610 = x15601[x15609];
double x15611 = (double)x15610;
double x15612 = sqrt(x15611);
float x15613 = (float)x15612;
x15608[x15609] = x15613;

}
int32_t x15617 = 0;
int32_t x15618 = 1;
x15618 *= 1;
x15617 += 1;
x15618 *= 1;
x15618 *= 1;
int32_t x15623 = x15617;
bool x15624 = x15623 >= 2;
if (x15624) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15629 = x15623 == 0;
if (x15629) {
int32_t x15630 = x15618;
bool x15631 = x15630 == 256;
if (x15631) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15638 = x15618;
bool x15640 = x15559 == 1;
int32_t x15639 = 256 / x15638;
bool x15641 = x15639 == 1;
bool x15645;
if (x454) {
bool x15642 = x15640 || x15641;
bool x15643 = x15559 == x15639;
bool x15644 = x15642 || x15643;
x15645 = x15644;
} else {
x15645 = false;
}
bool x15649;
if (x15645) {
x15649 = x15648;
} else {
x15649 = false;
}
bool x15650;
if (x15649) {
x15650 = x15648;
} else {
x15650 = false;
}
if (x15650) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15559,x15561,x15561,1,x15639,1,1);
assert(false && "");
}
bool x15656 = x15559 <= x15639;
int32_t x15657;
if (x15656) {
x15657 = x15639;
} else {
x15657 = x15559;
}
int32_t x15661 = x15657 * x15660;
int32_t x15662 = 64 * x15661;
float* x15663 = (float*)myMalloc(x15662 * sizeof(float));;
int32_t x15664;
if (x15640) {
x15664 = 0;
} else {
x15664 = x15562;
}
int32_t x15667;
if (x15641) {
x15667 = 0;
} else {
x15667 = 1;
}
for(int x15668=0; x15668 < 64; x15668++) {
int32_t x15680 = x15563 * x15668;
int32_t x15674 = x15661 * x15668;
for(int x15670=0; x15670 < x15657; x15670++) {
int32_t x15681 = x15664 * x15670;
int32_t x15682 = x15680 + x15681;
int32_t x15687 = x15667 * x15670;
int32_t x15676 = x15660 * x15670;
for(int x15672=0; x15672 < x15659; x15672++) {
int32_t x15683 = x15665 * x15672;
int32_t x15684 = x15682 + x15683;
int32_t x15678 = x15659 * x15672;
for(int x15673=0; x15673 < x15659; x15673++) {
int32_t x15685 = x15666 * x15673;
int32_t x15686 = x15684 + x15685;
float x15688 = x15565[x15686];
float x15689 = x15608[x15687];
int32_t x15675 = x15673 + x15674;
int32_t x15677 = x15675 + x15676;
int32_t x15679 = x15677 + x15678;
float x15690 = x15688 / x15689;
x15663[x15679] = x15690;

}

}

}

}
int32_t x15700 = 0;
int32_t x15701 = 1;
x15701 *= 1;
x15700 += 1;
x15701 *= 1;
x15701 *= 1;
int32_t x15706 = x15700;
bool x15707 = x15706 >= 2;
if (x15707) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15712 = x15706 == 0;
if (x15712) {
int32_t x15713 = x15701;
bool x15714 = x15713 == 256;
if (x15714) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15721 = x15701;
bool x15723 = x15657 == 1;
int32_t x15722 = 256 / x15721;
bool x15724 = x15722 == 1;
bool x15728;
if (x454) {
bool x15725 = x15723 || x15724;
bool x15726 = x15657 == x15722;
bool x15727 = x15725 || x15726;
x15728 = x15727;
} else {
x15728 = false;
}
bool x15732;
if (x15728) {
x15732 = x15731;
} else {
x15732 = false;
}
bool x15733;
if (x15732) {
x15733 = x15731;
} else {
x15733 = false;
}
if (x15733) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15657,x15659,x15659,1,x15722,1,1);
assert(false && "");
}
bool x15739 = x15657 <= x15722;
int32_t x15740;
if (x15739) {
x15740 = x15722;
} else {
x15740 = x15657;
}
int32_t x15744 = x15740 * x15743;
int32_t x15745 = 64 * x15744;
float* x15746 = (float*)myMalloc(x15745 * sizeof(float));;
int32_t x15747;
if (x15723) {
x15747 = 0;
} else {
x15747 = x15660;
}
int32_t x15750;
if (x15724) {
x15750 = 0;
} else {
x15750 = 1;
}
for(int x15751=0; x15751 < 64; x15751++) {
int32_t x15763 = x15661 * x15751;
int32_t x15757 = x15744 * x15751;
for(int x15753=0; x15753 < x15740; x15753++) {
int32_t x15764 = x15747 * x15753;
int32_t x15765 = x15763 + x15764;
int32_t x15770 = x15750 * x15753;
int32_t x15759 = x15743 * x15753;
for(int x15755=0; x15755 < x15742; x15755++) {
int32_t x15766 = x15748 * x15755;
int32_t x15767 = x15765 + x15766;
int32_t x15761 = x15742 * x15755;
for(int x15756=0; x15756 < x15742; x15756++) {
int32_t x15768 = x15749 * x15756;
int32_t x15769 = x15767 + x15768;
float x15771 = x15663[x15769];
float x15772 = x92[x15770];
int32_t x15758 = x15756 + x15757;
int32_t x15760 = x15758 + x15759;
int32_t x15762 = x15760 + x15761;
float x15773 = x15771 * x15772;
x15746[x15762] = x15773;

}

}

}

}
int32_t x15783 = 0;
int32_t x15784 = 1;
x15784 *= 1;
x15783 += 1;
x15784 *= 1;
x15784 *= 1;
int32_t x15789 = x15783;
bool x15790 = x15789 >= 2;
if (x15790) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15795 = x15789 == 0;
if (x15795) {
int32_t x15796 = x15784;
bool x15797 = x15796 == 256;
if (x15797) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15804 = x15784;
bool x15806 = x15740 == 1;
int32_t x15805 = 256 / x15804;
bool x15807 = x15805 == 1;
bool x15811;
if (x454) {
bool x15808 = x15806 || x15807;
bool x15809 = x15740 == x15805;
bool x15810 = x15808 || x15809;
x15811 = x15810;
} else {
x15811 = false;
}
bool x15815;
if (x15811) {
x15815 = x15814;
} else {
x15815 = false;
}
bool x15816;
if (x15815) {
x15816 = x15814;
} else {
x15816 = false;
}
if (x15816) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15740,x15742,x15742,1,x15805,1,1);
assert(false && "");
}
bool x15822 = x15740 <= x15805;
int32_t x15823;
if (x15822) {
x15823 = x15805;
} else {
x15823 = x15740;
}
int32_t x15827 = x15823 * x15826;
int32_t x15828 = 64 * x15827;
float* x15829 = (float*)myMalloc(x15828 * sizeof(float));;
int32_t x15830;
if (x15806) {
x15830 = 0;
} else {
x15830 = x15743;
}
int32_t x15833;
if (x15807) {
x15833 = 0;
} else {
x15833 = 1;
}
for(int x15834=0; x15834 < 64; x15834++) {
int32_t x15846 = x15744 * x15834;
int32_t x15840 = x15827 * x15834;
for(int x15836=0; x15836 < x15823; x15836++) {
int32_t x15847 = x15830 * x15836;
int32_t x15848 = x15846 + x15847;
int32_t x15853 = x15833 * x15836;
int32_t x15842 = x15826 * x15836;
for(int x15838=0; x15838 < x15825; x15838++) {
int32_t x15849 = x15831 * x15838;
int32_t x15850 = x15848 + x15849;
int32_t x15844 = x15825 * x15838;
for(int x15839=0; x15839 < x15825; x15839++) {
int32_t x15851 = x15832 * x15839;
int32_t x15852 = x15850 + x15851;
float x15854 = x15746[x15852];
float x15855 = x241[x15853];
int32_t x15841 = x15839 + x15840;
int32_t x15843 = x15841 + x15842;
int32_t x15845 = x15843 + x15844;
float x15856 = x15854 + x15855;
x15829[x15845] = x15856;

}

}

}

}
float* x15866 = (float*)myMalloc(x15828 * sizeof(float));;
for(int x15868=0; x15868 < x15828; x15868++) {
float x15869 = x15829[x15868];
bool x15870 = x15869 < 0.0f;
if (x15870) {
x15866[x15868] = 0.0f;
} else {
float x15873 = x15829[x15868];
x15866[x15868] = x15873;
}

}
float* x15888 = (float*)myMalloc(x15887 * sizeof(float));;
int32_t x15889 = 9 * x15823;
int32_t x15892 = 64 * x15889;
int32_t x15893 = x15892 * x15883;
float* x15894 = (float*)myMalloc(x15893 * sizeof(float));;
int32_t x15890 = x15889 * x15883;
int32_t x15902 = x15823 * 3;
int32_t x15903 = x15902 * 3;
for(int x15895=0; x15895 < 64; x15895++) {
int32_t x15896 = x15895 * x15827;
float* x15897 = x15866+x15896;
int32_t x15898 = x15895 * x15884;
float* x15899 = x15888+x15898;
int32_t x15900 = x15895 * x15890;
float* x15901 = x15894+x15900;
for(int x15905=0; x15905 < x15903; x15905++) {
int32_t x15906 = x15905 / 9;
int32_t x15910 = x15906 * 3;
int32_t x15911 = x15910 * 3;
int32_t x15912 = x15911 * x15882;
int32_t x15913 = x15912 * x15882;
int32_t x15907 = x15905 % 9;
int32_t x15908 = x15907 / 3;
int32_t x15914 = x15908 * 3;
int32_t x15915 = x15914 * x15882;
int32_t x15916 = x15915 * x15882;
int32_t x15917 = x15913 + x15916;
int32_t x15909 = x15907 % 3;
int32_t x15918 = x15909 * x15882;
int32_t x15919 = x15918 * x15882;
int32_t x15920 = x15917 + x15919;
float* x15921 = x15901+x15920;
int32_t x15922 = x15906 * x15825;
int32_t x15923 = x15922 * x15825;
float* x15924 = x15897+x15923;
int32_t x15937 = 1 - x15909;
bool x15938 = x15937 > 0;
int32_t x15939;
if (x15938) {
x15939 = x15937;
} else {
x15939 = 0;
}
int32_t x15940 = 3 - x15909;
int32_t x15941 = x15940 - 1;
int32_t x15942 = 1 - x15941;
bool x15943 = x15942 > 0;
int32_t x15944;
if (x15943) {
x15944 = x15942;
} else {
x15944 = 0;
}
int32_t x15945 = x15882 - x15944;
int32_t x15946 = x15945 - x15939;
bool x15947 = x15946 <= 0;
bool x15951 = x15939 > 0;
int32_t x15936 = -1 + x15909;
bool x15964 = x15944 > 0;
for(int x15926=0; x15926 < x15882; x15926++) {
int32_t x15927 = x15926 - 1;
int32_t x15928 = x15927 + x15908;
bool x15929 = x15928 < 0;
bool x15930 = x15928 >= x15825;
bool x15931 = x15929 || x15930;
if (x15931) {
int32_t x15932 = x15926 * x15882;
float* x15933 = x15921+x15932;
memset(x15933, 0, 4 * x15882);;
} else {
if (x15947) {
int32_t x15932 = x15926 * x15882;
float* x15948 = x15921+x15932;
memset(x15948, 0, 4 * x15882);;
} else {
int32_t x15932 = x15926 * x15882;
if (x15951) {
float* x15952 = x15921+x15932;
memset(x15952, 0, 4 * x15939);;
} else {
}
// may have segfault here
int32_t x15957 = x15932 + x15939;
float* x15958 = x15921+x15957;
int32_t x15959 = x15928 * x15825;
int32_t x15960 = x15959 + x15936;
int32_t x15961 = x15960 + x15939;
float* x15962 = x15924+x15961;
memcpy(x15958, x15962, 4 * x15946);;
if (x15964) {
int32_t x15965 = x15932 + x15882;
int32_t x15966 = x15965 - x15944;
float* x15967 = x15921+x15966;
memset(x15967, 0, 4 * x15944);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x15883,x15889,1,x249,x15889,x15901,x15883,1,x15899,x15883);

}
int32_t x15982 = 0;
int32_t x15983 = 1;
x15983 *= 1;
x15982 += 1;
x15983 *= 1;
x15983 *= 1;
int32_t x15988 = x15982;
bool x15989 = x15988 >= 2;
if (x15989) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15994 = x15988 == 0;
if (x15994) {
int32_t x15995 = x15983;
bool x15996 = x15995 == 256;
if (x15996) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16003 = x15983;
int32_t x16004 = 256 / x16003;
bool x16005 = x16004 == 1;
bool x16008;
if (x454) {
bool x16006 = 256 == x16004;
bool x16007 = x16005 || x16006;
x16008 = x16007;
} else {
x16008 = false;
}
bool x16012;
if (x16008) {
x16012 = x16011;
} else {
x16012 = false;
}
bool x16013;
if (x16012) {
x16013 = x16011;
} else {
x16013 = false;
}
if (x16013) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x15882,x15882,1,x16004,1,1);
assert(false && "");
}
bool x16019 = 256 <= x16004;
int32_t x16020;
if (x16019) {
x16020 = x16004;
} else {
x16020 = 256;
}
int32_t x16024 = x16020 * x16023;
int32_t x16025 = 64 * x16024;
float* x16026 = (float*)myMalloc(x16025 * sizeof(float));;
int32_t x16029;
if (x16005) {
x16029 = 0;
} else {
x16029 = 1;
}
for(int x16030=0; x16030 < 64; x16030++) {
int32_t x16042 = x15884 * x16030;
int32_t x16036 = x16024 * x16030;
for(int x16032=0; x16032 < x16020; x16032++) {
int32_t x16043 = x15883 * x16032;
int32_t x16044 = x16042 + x16043;
int32_t x16049 = x16029 * x16032;
int32_t x16038 = x16023 * x16032;
for(int x16034=0; x16034 < x16022; x16034++) {
int32_t x16045 = x16027 * x16034;
int32_t x16046 = x16044 + x16045;
int32_t x16040 = x16022 * x16034;
for(int x16035=0; x16035 < x16022; x16035++) {
int32_t x16047 = x16028 * x16035;
int32_t x16048 = x16046 + x16047;
float x16050 = x15888[x16048];
float x16051 = x186[x16049];
int32_t x16037 = x16035 + x16036;
int32_t x16039 = x16037 + x16038;
int32_t x16041 = x16039 + x16040;
float x16052 = x16050 - x16051;
x16026[x16041] = x16052;

}

}

}

}
float* x16062 = (float*)myMalloc(256 * sizeof(float));;
for(int x16063=0; x16063 < 256; x16063++) {
float x16064 = x230[x16063];
float x16065 = x16064 + 1.0E-5f;
x16062[x16063] = x16065;

}
float* x16069 = (float*)myMalloc(256 * sizeof(float));;
for(int x16070=0; x16070 < 256; x16070++) {
float x16071 = x16062[x16070];
double x16072 = (double)x16071;
double x16073 = sqrt(x16072);
float x16074 = (float)x16073;
x16069[x16070] = x16074;

}
int32_t x16078 = 0;
int32_t x16079 = 1;
x16079 *= 1;
x16078 += 1;
x16079 *= 1;
x16079 *= 1;
int32_t x16084 = x16078;
bool x16085 = x16084 >= 2;
if (x16085) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16090 = x16084 == 0;
if (x16090) {
int32_t x16091 = x16079;
bool x16092 = x16091 == 256;
if (x16092) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16099 = x16079;
bool x16101 = x16020 == 1;
int32_t x16100 = 256 / x16099;
bool x16102 = x16100 == 1;
bool x16106;
if (x454) {
bool x16103 = x16101 || x16102;
bool x16104 = x16020 == x16100;
bool x16105 = x16103 || x16104;
x16106 = x16105;
} else {
x16106 = false;
}
bool x16110;
if (x16106) {
x16110 = x16109;
} else {
x16110 = false;
}
bool x16111;
if (x16110) {
x16111 = x16109;
} else {
x16111 = false;
}
if (x16111) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16020,x16022,x16022,1,x16100,1,1);
assert(false && "");
}
bool x16117 = x16020 <= x16100;
int32_t x16118;
if (x16117) {
x16118 = x16100;
} else {
x16118 = x16020;
}
int32_t x16122 = x16118 * x16121;
int32_t x16123 = 64 * x16122;
float* x16124 = (float*)myMalloc(x16123 * sizeof(float));;
int32_t x16125;
if (x16101) {
x16125 = 0;
} else {
x16125 = x16023;
}
int32_t x16128;
if (x16102) {
x16128 = 0;
} else {
x16128 = 1;
}
for(int x16129=0; x16129 < 64; x16129++) {
int32_t x16141 = x16024 * x16129;
int32_t x16135 = x16122 * x16129;
for(int x16131=0; x16131 < x16118; x16131++) {
int32_t x16142 = x16125 * x16131;
int32_t x16143 = x16141 + x16142;
int32_t x16148 = x16128 * x16131;
int32_t x16137 = x16121 * x16131;
for(int x16133=0; x16133 < x16120; x16133++) {
int32_t x16144 = x16126 * x16133;
int32_t x16145 = x16143 + x16144;
int32_t x16139 = x16120 * x16133;
for(int x16134=0; x16134 < x16120; x16134++) {
int32_t x16146 = x16127 * x16134;
int32_t x16147 = x16145 + x16146;
float x16149 = x16026[x16147];
float x16150 = x16069[x16148];
int32_t x16136 = x16134 + x16135;
int32_t x16138 = x16136 + x16137;
int32_t x16140 = x16138 + x16139;
float x16151 = x16149 / x16150;
x16124[x16140] = x16151;

}

}

}

}
int32_t x16161 = 0;
int32_t x16162 = 1;
x16162 *= 1;
x16161 += 1;
x16162 *= 1;
x16162 *= 1;
int32_t x16167 = x16161;
bool x16168 = x16167 >= 2;
if (x16168) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16173 = x16167 == 0;
if (x16173) {
int32_t x16174 = x16162;
bool x16175 = x16174 == 256;
if (x16175) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16182 = x16162;
bool x16184 = x16118 == 1;
int32_t x16183 = 256 / x16182;
bool x16185 = x16183 == 1;
bool x16189;
if (x454) {
bool x16186 = x16184 || x16185;
bool x16187 = x16118 == x16183;
bool x16188 = x16186 || x16187;
x16189 = x16188;
} else {
x16189 = false;
}
bool x16193;
if (x16189) {
x16193 = x16192;
} else {
x16193 = false;
}
bool x16194;
if (x16193) {
x16194 = x16192;
} else {
x16194 = false;
}
if (x16194) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16118,x16120,x16120,1,x16183,1,1);
assert(false && "");
}
bool x16200 = x16118 <= x16183;
int32_t x16201;
if (x16200) {
x16201 = x16183;
} else {
x16201 = x16118;
}
int32_t x16205 = x16201 * x16204;
int32_t x16206 = 64 * x16205;
float* x16207 = (float*)myMalloc(x16206 * sizeof(float));;
int32_t x16208;
if (x16184) {
x16208 = 0;
} else {
x16208 = x16121;
}
int32_t x16211;
if (x16185) {
x16211 = 0;
} else {
x16211 = 1;
}
for(int x16212=0; x16212 < 64; x16212++) {
int32_t x16224 = x16122 * x16212;
int32_t x16218 = x16205 * x16212;
for(int x16214=0; x16214 < x16201; x16214++) {
int32_t x16225 = x16208 * x16214;
int32_t x16226 = x16224 + x16225;
int32_t x16231 = x16211 * x16214;
int32_t x16220 = x16204 * x16214;
for(int x16216=0; x16216 < x16203; x16216++) {
int32_t x16227 = x16209 * x16216;
int32_t x16228 = x16226 + x16227;
int32_t x16222 = x16203 * x16216;
for(int x16217=0; x16217 < x16203; x16217++) {
int32_t x16229 = x16210 * x16217;
int32_t x16230 = x16228 + x16229;
float x16232 = x16124[x16230];
float x16233 = x74[x16231];
int32_t x16219 = x16217 + x16218;
int32_t x16221 = x16219 + x16220;
int32_t x16223 = x16221 + x16222;
float x16234 = x16232 * x16233;
x16207[x16223] = x16234;

}

}

}

}
int32_t x16244 = 0;
int32_t x16245 = 1;
x16245 *= 1;
x16244 += 1;
x16245 *= 1;
x16245 *= 1;
int32_t x16250 = x16244;
bool x16251 = x16250 >= 2;
if (x16251) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16256 = x16250 == 0;
if (x16256) {
int32_t x16257 = x16245;
bool x16258 = x16257 == 256;
if (x16258) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16265 = x16245;
bool x16267 = x16201 == 1;
int32_t x16266 = 256 / x16265;
bool x16268 = x16266 == 1;
bool x16272;
if (x454) {
bool x16269 = x16267 || x16268;
bool x16270 = x16201 == x16266;
bool x16271 = x16269 || x16270;
x16272 = x16271;
} else {
x16272 = false;
}
bool x16276;
if (x16272) {
x16276 = x16275;
} else {
x16276 = false;
}
bool x16277;
if (x16276) {
x16277 = x16275;
} else {
x16277 = false;
}
if (x16277) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16201,x16203,x16203,1,x16266,1,1);
assert(false && "");
}
bool x16283 = x16201 <= x16266;
int32_t x16284;
if (x16283) {
x16284 = x16266;
} else {
x16284 = x16201;
}
int32_t x16288 = x16284 * x16287;
int32_t x16289 = 64 * x16288;
float* x16290 = (float*)myMalloc(x16289 * sizeof(float));;
int32_t x16291;
if (x16267) {
x16291 = 0;
} else {
x16291 = x16204;
}
int32_t x16294;
if (x16268) {
x16294 = 0;
} else {
x16294 = 1;
}
for(int x16295=0; x16295 < 64; x16295++) {
int32_t x16307 = x16205 * x16295;
int32_t x16301 = x16288 * x16295;
for(int x16297=0; x16297 < x16284; x16297++) {
int32_t x16308 = x16291 * x16297;
int32_t x16309 = x16307 + x16308;
int32_t x16314 = x16294 * x16297;
int32_t x16303 = x16287 * x16297;
for(int x16299=0; x16299 < x16286; x16299++) {
int32_t x16310 = x16292 * x16299;
int32_t x16311 = x16309 + x16310;
int32_t x16305 = x16286 * x16299;
for(int x16300=0; x16300 < x16286; x16300++) {
int32_t x16312 = x16293 * x16300;
int32_t x16313 = x16311 + x16312;
float x16315 = x16207[x16313];
float x16316 = x136[x16314];
int32_t x16302 = x16300 + x16301;
int32_t x16304 = x16302 + x16303;
int32_t x16306 = x16304 + x16305;
float x16317 = x16315 + x16316;
x16290[x16306] = x16317;

}

}

}

}
float* x16327 = (float*)myMalloc(x16289 * sizeof(float));;
for(int x16329=0; x16329 < x16289; x16329++) {
float x16330 = x16290[x16329];
bool x16331 = x16330 < 0.0f;
if (x16331) {
x16327[x16329] = 0.0f;
} else {
float x16334 = x16290[x16329];
x16327[x16329] = x16334;
}

}
float* x16348 = (float*)myMalloc(x16347 * sizeof(float));;
int32_t x16351 = 64 * x16284;
int32_t x16352 = x16351 * x16343;
float* x16353 = (float*)myMalloc(x16352 * sizeof(float));;
int32_t x16349 = x16284 * x16343;
for(int x16354=0; x16354 < 64; x16354++) {
int32_t x16355 = x16354 * x16288;
float* x16356 = x16327+x16355;
int32_t x16357 = x16354 * x16344;
float* x16358 = x16348+x16357;
int32_t x16359 = x16354 * x16349;
float* x16360 = x16353+x16359;
for(int x16361=0; x16361 < x16284; x16361++) {
int32_t x16362 = x16361 / 1;
int32_t x16366 = x16362 * x16342;
int32_t x16367 = x16366 * x16342;
int32_t x16363 = x16361 % 1;
int32_t x16364 = x16363 / 1;
int32_t x16368 = x16364 * x16342;
int32_t x16369 = x16368 * x16342;
int32_t x16370 = x16367 + x16369;
int32_t x16365 = x16363 % 1;
int32_t x16371 = x16365 * x16342;
int32_t x16372 = x16371 * x16342;
int32_t x16373 = x16370 + x16372;
float* x16374 = x16360+x16373;
int32_t x16375 = x16362 * x16286;
int32_t x16376 = x16375 * x16286;
float* x16377 = x16356+x16376;
for(int x16379=0; x16379 < x16342; x16379++) {
int32_t x16381 = x16379 * x16342;
float* x16382 = x16374+x16381;
int32_t x16380 = x16379 + x16364;
int32_t x16383 = x16380 * x16286;
int32_t x16384 = x16383 + x16365;
float* x16385 = x16377+x16384;
memcpy(x16382, x16385, 4 * x16342);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x16343,x16284,1,x89,x16284,x16360,x16343,1,x16358,x16343);

}
int32_t x16394 = 0;
int32_t x16395 = 1;
x16395 *= 1;
x16394 += 1;
x16395 *= 1;
x16395 *= 1;
int32_t x16400 = x16394;
bool x16401 = x16400 >= 2;
if (x16401) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16406 = x16400 == 0;
if (x16406) {
int32_t x16407 = x16395;
bool x16408 = x16407 == 1024;
if (x16408) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16415 = x16395;
int32_t x16416 = 1024 / x16415;
bool x16417 = x16416 == 1;
bool x16420;
if (x454) {
bool x16418 = 1024 == x16416;
bool x16419 = x16417 || x16418;
x16420 = x16419;
} else {
x16420 = false;
}
bool x16424;
if (x16420) {
x16424 = x16423;
} else {
x16424 = false;
}
bool x16425;
if (x16424) {
x16425 = x16423;
} else {
x16425 = false;
}
if (x16425) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x16342,x16342,1,x16416,1,1);
assert(false && "");
}
bool x16431 = 1024 <= x16416;
int32_t x16432;
if (x16431) {
x16432 = x16416;
} else {
x16432 = 1024;
}
int32_t x16436 = x16432 * x16435;
int32_t x16437 = 64 * x16436;
float* x16438 = (float*)myMalloc(x16437 * sizeof(float));;
int32_t x16441;
if (x16417) {
x16441 = 0;
} else {
x16441 = 1;
}
for(int x16442=0; x16442 < 64; x16442++) {
int32_t x16454 = x16344 * x16442;
int32_t x16448 = x16436 * x16442;
for(int x16444=0; x16444 < x16432; x16444++) {
int32_t x16455 = x16343 * x16444;
int32_t x16456 = x16454 + x16455;
int32_t x16461 = x16441 * x16444;
int32_t x16450 = x16435 * x16444;
for(int x16446=0; x16446 < x16434; x16446++) {
int32_t x16457 = x16439 * x16446;
int32_t x16458 = x16456 + x16457;
int32_t x16452 = x16434 * x16446;
for(int x16447=0; x16447 < x16434; x16447++) {
int32_t x16459 = x16440 * x16447;
int32_t x16460 = x16458 + x16459;
float x16462 = x16348[x16460];
float x16463 = x231[x16461];
int32_t x16449 = x16447 + x16448;
int32_t x16451 = x16449 + x16450;
int32_t x16453 = x16451 + x16452;
float x16464 = x16462 - x16463;
x16438[x16453] = x16464;

}

}

}

}
float* x16474 = (float*)myMalloc(1024 * sizeof(float));;
for(int x16475=0; x16475 < 1024; x16475++) {
float x16476 = x161[x16475];
float x16477 = x16476 + 1.0E-5f;
x16474[x16475] = x16477;

}
float* x16481 = (float*)myMalloc(1024 * sizeof(float));;
for(int x16482=0; x16482 < 1024; x16482++) {
float x16483 = x16474[x16482];
double x16484 = (double)x16483;
double x16485 = sqrt(x16484);
float x16486 = (float)x16485;
x16481[x16482] = x16486;

}
int32_t x16490 = 0;
int32_t x16491 = 1;
x16491 *= 1;
x16490 += 1;
x16491 *= 1;
x16491 *= 1;
int32_t x16496 = x16490;
bool x16497 = x16496 >= 2;
if (x16497) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16502 = x16496 == 0;
if (x16502) {
int32_t x16503 = x16491;
bool x16504 = x16503 == 1024;
if (x16504) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16511 = x16491;
bool x16513 = x16432 == 1;
int32_t x16512 = 1024 / x16511;
bool x16514 = x16512 == 1;
bool x16518;
if (x454) {
bool x16515 = x16513 || x16514;
bool x16516 = x16432 == x16512;
bool x16517 = x16515 || x16516;
x16518 = x16517;
} else {
x16518 = false;
}
bool x16522;
if (x16518) {
x16522 = x16521;
} else {
x16522 = false;
}
bool x16523;
if (x16522) {
x16523 = x16521;
} else {
x16523 = false;
}
if (x16523) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16432,x16434,x16434,1,x16512,1,1);
assert(false && "");
}
bool x16529 = x16432 <= x16512;
int32_t x16530;
if (x16529) {
x16530 = x16512;
} else {
x16530 = x16432;
}
int32_t x16534 = x16530 * x16533;
int32_t x16535 = 64 * x16534;
float* x16536 = (float*)myMalloc(x16535 * sizeof(float));;
int32_t x16537;
if (x16513) {
x16537 = 0;
} else {
x16537 = x16435;
}
int32_t x16540;
if (x16514) {
x16540 = 0;
} else {
x16540 = 1;
}
for(int x16541=0; x16541 < 64; x16541++) {
int32_t x16553 = x16436 * x16541;
int32_t x16547 = x16534 * x16541;
for(int x16543=0; x16543 < x16530; x16543++) {
int32_t x16554 = x16537 * x16543;
int32_t x16555 = x16553 + x16554;
int32_t x16560 = x16540 * x16543;
int32_t x16549 = x16533 * x16543;
for(int x16545=0; x16545 < x16532; x16545++) {
int32_t x16556 = x16538 * x16545;
int32_t x16557 = x16555 + x16556;
int32_t x16551 = x16532 * x16545;
for(int x16546=0; x16546 < x16532; x16546++) {
int32_t x16558 = x16539 * x16546;
int32_t x16559 = x16557 + x16558;
float x16561 = x16438[x16559];
float x16562 = x16481[x16560];
int32_t x16548 = x16546 + x16547;
int32_t x16550 = x16548 + x16549;
int32_t x16552 = x16550 + x16551;
float x16563 = x16561 / x16562;
x16536[x16552] = x16563;

}

}

}

}
int32_t x16573 = 0;
int32_t x16574 = 1;
x16574 *= 1;
x16573 += 1;
x16574 *= 1;
x16574 *= 1;
int32_t x16579 = x16573;
bool x16580 = x16579 >= 2;
if (x16580) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16585 = x16579 == 0;
if (x16585) {
int32_t x16586 = x16574;
bool x16587 = x16586 == 1024;
if (x16587) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16594 = x16574;
bool x16596 = x16530 == 1;
int32_t x16595 = 1024 / x16594;
bool x16597 = x16595 == 1;
bool x16601;
if (x454) {
bool x16598 = x16596 || x16597;
bool x16599 = x16530 == x16595;
bool x16600 = x16598 || x16599;
x16601 = x16600;
} else {
x16601 = false;
}
bool x16605;
if (x16601) {
x16605 = x16604;
} else {
x16605 = false;
}
bool x16606;
if (x16605) {
x16606 = x16604;
} else {
x16606 = false;
}
if (x16606) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16530,x16532,x16532,1,x16595,1,1);
assert(false && "");
}
bool x16612 = x16530 <= x16595;
int32_t x16613;
if (x16612) {
x16613 = x16595;
} else {
x16613 = x16530;
}
int32_t x16617 = x16613 * x16616;
int32_t x16618 = 64 * x16617;
float* x16619 = (float*)myMalloc(x16618 * sizeof(float));;
int32_t x16620;
if (x16596) {
x16620 = 0;
} else {
x16620 = x16533;
}
int32_t x16623;
if (x16597) {
x16623 = 0;
} else {
x16623 = 1;
}
for(int x16624=0; x16624 < 64; x16624++) {
int32_t x16636 = x16534 * x16624;
int32_t x16630 = x16617 * x16624;
for(int x16626=0; x16626 < x16613; x16626++) {
int32_t x16637 = x16620 * x16626;
int32_t x16638 = x16636 + x16637;
int32_t x16643 = x16623 * x16626;
int32_t x16632 = x16616 * x16626;
for(int x16628=0; x16628 < x16615; x16628++) {
int32_t x16639 = x16621 * x16628;
int32_t x16640 = x16638 + x16639;
int32_t x16634 = x16615 * x16628;
for(int x16629=0; x16629 < x16615; x16629++) {
int32_t x16641 = x16622 * x16629;
int32_t x16642 = x16640 + x16641;
float x16644 = x16536[x16642];
float x16645 = x238[x16643];
int32_t x16631 = x16629 + x16630;
int32_t x16633 = x16631 + x16632;
int32_t x16635 = x16633 + x16634;
float x16646 = x16644 * x16645;
x16619[x16635] = x16646;

}

}

}

}
int32_t x16656 = 0;
int32_t x16657 = 1;
x16657 *= 1;
x16656 += 1;
x16657 *= 1;
x16657 *= 1;
int32_t x16662 = x16656;
bool x16663 = x16662 >= 2;
if (x16663) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16668 = x16662 == 0;
if (x16668) {
int32_t x16669 = x16657;
bool x16670 = x16669 == 1024;
if (x16670) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16677 = x16657;
bool x16679 = x16613 == 1;
int32_t x16678 = 1024 / x16677;
bool x16680 = x16678 == 1;
bool x16684;
if (x454) {
bool x16681 = x16679 || x16680;
bool x16682 = x16613 == x16678;
bool x16683 = x16681 || x16682;
x16684 = x16683;
} else {
x16684 = false;
}
bool x16688;
if (x16684) {
x16688 = x16687;
} else {
x16688 = false;
}
bool x16689;
if (x16688) {
x16689 = x16687;
} else {
x16689 = false;
}
if (x16689) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16613,x16615,x16615,1,x16678,1,1);
assert(false && "");
}
bool x16695 = x16613 <= x16678;
int32_t x16696;
if (x16695) {
x16696 = x16678;
} else {
x16696 = x16613;
}
int32_t x16700 = x16696 * x16699;
int32_t x16701 = 64 * x16700;
float* x16702 = (float*)myMalloc(x16701 * sizeof(float));;
int32_t x16703;
if (x16679) {
x16703 = 0;
} else {
x16703 = x16616;
}
int32_t x16706;
if (x16680) {
x16706 = 0;
} else {
x16706 = 1;
}
for(int x16707=0; x16707 < 64; x16707++) {
int32_t x16719 = x16617 * x16707;
int32_t x16713 = x16700 * x16707;
for(int x16709=0; x16709 < x16696; x16709++) {
int32_t x16720 = x16703 * x16709;
int32_t x16721 = x16719 + x16720;
int32_t x16726 = x16706 * x16709;
int32_t x16715 = x16699 * x16709;
for(int x16711=0; x16711 < x16698; x16711++) {
int32_t x16722 = x16704 * x16711;
int32_t x16723 = x16721 + x16722;
int32_t x16717 = x16698 * x16711;
for(int x16712=0; x16712 < x16698; x16712++) {
int32_t x16724 = x16705 * x16712;
int32_t x16725 = x16723 + x16724;
float x16727 = x16619[x16725];
float x16728 = x146[x16726];
int32_t x16714 = x16712 + x16713;
int32_t x16716 = x16714 + x16715;
int32_t x16718 = x16716 + x16717;
float x16729 = x16727 + x16728;
x16702[x16718] = x16729;

}

}

}

}
bool x16739 = x16696 == 1;
bool x16740 = x16739 || x15397;
bool x16741 = x16696 == x15354;
bool x16742 = x16740 || x16741;
bool x16747;
if (x16742) {
x16747 = x16746;
} else {
x16747 = false;
}
bool x16748;
if (x16747) {
x16748 = x16746;
} else {
x16748 = false;
}
if (x16748) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16696,x16698,x16698,64,x15354,x15356,x15356);
assert(false && "");
}
bool x16754 = x16696 <= x15354;
int32_t x16755;
if (x16754) {
x16755 = x15354;
} else {
x16755 = x16696;
}
int32_t x16761;
if (x16739) {
x16761 = 0;
} else {
x16761 = x16699;
}
for(int x16764=0; x16764 < 64; x16764++) {
int32_t x16770 = x16700 * x16764;
int32_t x16777 = x15358 * x16764;
for(int x16766=0; x16766 < x16755; x16766++) {
int32_t x16771 = x16761 * x16766;
int32_t x16772 = x16770 + x16771;
int32_t x16778 = x15419 * x16766;
int32_t x16779 = x16777 + x16778;
for(int x16768=0; x16768 < x16757; x16768++) {
int32_t x16773 = x16762 * x16768;
int32_t x16774 = x16772 + x16773;
int32_t x16780 = x15420 * x16768;
int32_t x16781 = x16779 + x16780;
for(int x16769=0; x16769 < x16757; x16769++) {
int32_t x16775 = x16763 * x16769;
int32_t x16776 = x16774 + x16775;
float x16784 = x16702[x16776];
int32_t x16782 = x15421 * x16769;
int32_t x16783 = x16781 + x16782;
float x16785 = x15454[x16783];
float x16786 = x16784 + x16785;
x16702[x16776] = x16786;

}

}

}

}
float* x16796 = (float*)myMalloc(x16701 * sizeof(float));;
for(int x16798=0; x16798 < x16701; x16798++) {
float x16799 = x16702[x16798];
bool x16800 = x16799 < 0.0f;
if (x16800) {
x16796[x16798] = 0.0f;
} else {
float x16803 = x16702[x16798];
x16796[x16798] = x16803;
}

}
float* x16817 = (float*)myMalloc(x16816 * sizeof(float));;
int32_t x16820 = 64 * x16696;
int32_t x16821 = x16820 * x16812;
float* x16822 = (float*)myMalloc(x16821 * sizeof(float));;
int32_t x16818 = x16696 * x16812;
for(int x16823=0; x16823 < 64; x16823++) {
int32_t x16824 = x16823 * x16700;
float* x16825 = x16796+x16824;
int32_t x16826 = x16823 * x16813;
float* x16827 = x16817+x16826;
int32_t x16828 = x16823 * x16818;
float* x16829 = x16822+x16828;
for(int x16830=0; x16830 < x16696; x16830++) {
int32_t x16831 = x16830 / 1;
int32_t x16835 = x16831 * x16811;
int32_t x16836 = x16835 * x16811;
int32_t x16832 = x16830 % 1;
int32_t x16833 = x16832 / 1;
int32_t x16837 = x16833 * x16811;
int32_t x16838 = x16837 * x16811;
int32_t x16839 = x16836 + x16838;
int32_t x16834 = x16832 % 1;
int32_t x16840 = x16834 * x16811;
int32_t x16841 = x16840 * x16811;
int32_t x16842 = x16839 + x16841;
float* x16843 = x16829+x16842;
int32_t x16844 = x16831 * x16698;
int32_t x16845 = x16844 * x16698;
float* x16846 = x16825+x16845;
for(int x16848=0; x16848 < x16811; x16848++) {
int32_t x16850 = x16848 * x16811;
float* x16851 = x16843+x16850;
int32_t x16849 = x16848 + x16833;
int32_t x16852 = x16849 * x16698;
int32_t x16853 = x16852 + x16834;
float* x16854 = x16846+x16853;
memcpy(x16851, x16854, 4 * x16811);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x16812,x16696,1,x22,x16696,x16829,x16812,1,x16827,x16812);

}
int32_t x16863 = 0;
int32_t x16864 = 1;
x16864 *= 1;
x16863 += 1;
x16864 *= 1;
x16864 *= 1;
int32_t x16869 = x16863;
bool x16870 = x16869 >= 2;
if (x16870) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16875 = x16869 == 0;
if (x16875) {
int32_t x16876 = x16864;
bool x16877 = x16876 == 256;
if (x16877) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16884 = x16864;
int32_t x16885 = 256 / x16884;
bool x16886 = x16885 == 1;
bool x16889;
if (x454) {
bool x16887 = 256 == x16885;
bool x16888 = x16886 || x16887;
x16889 = x16888;
} else {
x16889 = false;
}
bool x16893;
if (x16889) {
x16893 = x16892;
} else {
x16893 = false;
}
bool x16894;
if (x16893) {
x16894 = x16892;
} else {
x16894 = false;
}
if (x16894) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x16811,x16811,1,x16885,1,1);
assert(false && "");
}
bool x16900 = 256 <= x16885;
int32_t x16901;
if (x16900) {
x16901 = x16885;
} else {
x16901 = 256;
}
int32_t x16905 = x16901 * x16904;
int32_t x16906 = 64 * x16905;
float* x16907 = (float*)myMalloc(x16906 * sizeof(float));;
int32_t x16910;
if (x16886) {
x16910 = 0;
} else {
x16910 = 1;
}
for(int x16911=0; x16911 < 64; x16911++) {
int32_t x16923 = x16813 * x16911;
int32_t x16917 = x16905 * x16911;
for(int x16913=0; x16913 < x16901; x16913++) {
int32_t x16924 = x16812 * x16913;
int32_t x16925 = x16923 + x16924;
int32_t x16930 = x16910 * x16913;
int32_t x16919 = x16904 * x16913;
for(int x16915=0; x16915 < x16903; x16915++) {
int32_t x16926 = x16908 * x16915;
int32_t x16927 = x16925 + x16926;
int32_t x16921 = x16903 * x16915;
for(int x16916=0; x16916 < x16903; x16916++) {
int32_t x16928 = x16909 * x16916;
int32_t x16929 = x16927 + x16928;
float x16931 = x16817[x16929];
float x16932 = x254[x16930];
int32_t x16918 = x16916 + x16917;
int32_t x16920 = x16918 + x16919;
int32_t x16922 = x16920 + x16921;
float x16933 = x16931 - x16932;
x16907[x16922] = x16933;

}

}

}

}
float* x16943 = (float*)myMalloc(256 * sizeof(float));;
for(int x16944=0; x16944 < 256; x16944++) {
float x16945 = x69[x16944];
float x16946 = x16945 + 1.0E-5f;
x16943[x16944] = x16946;

}
float* x16950 = (float*)myMalloc(256 * sizeof(float));;
for(int x16951=0; x16951 < 256; x16951++) {
float x16952 = x16943[x16951];
double x16953 = (double)x16952;
double x16954 = sqrt(x16953);
float x16955 = (float)x16954;
x16950[x16951] = x16955;

}
int32_t x16959 = 0;
int32_t x16960 = 1;
x16960 *= 1;
x16959 += 1;
x16960 *= 1;
x16960 *= 1;
int32_t x16965 = x16959;
bool x16966 = x16965 >= 2;
if (x16966) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16971 = x16965 == 0;
if (x16971) {
int32_t x16972 = x16960;
bool x16973 = x16972 == 256;
if (x16973) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16980 = x16960;
bool x16982 = x16901 == 1;
int32_t x16981 = 256 / x16980;
bool x16983 = x16981 == 1;
bool x16987;
if (x454) {
bool x16984 = x16982 || x16983;
bool x16985 = x16901 == x16981;
bool x16986 = x16984 || x16985;
x16987 = x16986;
} else {
x16987 = false;
}
bool x16991;
if (x16987) {
x16991 = x16990;
} else {
x16991 = false;
}
bool x16992;
if (x16991) {
x16992 = x16990;
} else {
x16992 = false;
}
if (x16992) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16901,x16903,x16903,1,x16981,1,1);
assert(false && "");
}
bool x16998 = x16901 <= x16981;
int32_t x16999;
if (x16998) {
x16999 = x16981;
} else {
x16999 = x16901;
}
int32_t x17003 = x16999 * x17002;
int32_t x17004 = 64 * x17003;
float* x17005 = (float*)myMalloc(x17004 * sizeof(float));;
int32_t x17006;
if (x16982) {
x17006 = 0;
} else {
x17006 = x16904;
}
int32_t x17009;
if (x16983) {
x17009 = 0;
} else {
x17009 = 1;
}
for(int x17010=0; x17010 < 64; x17010++) {
int32_t x17022 = x16905 * x17010;
int32_t x17016 = x17003 * x17010;
for(int x17012=0; x17012 < x16999; x17012++) {
int32_t x17023 = x17006 * x17012;
int32_t x17024 = x17022 + x17023;
int32_t x17029 = x17009 * x17012;
int32_t x17018 = x17002 * x17012;
for(int x17014=0; x17014 < x17001; x17014++) {
int32_t x17025 = x17007 * x17014;
int32_t x17026 = x17024 + x17025;
int32_t x17020 = x17001 * x17014;
for(int x17015=0; x17015 < x17001; x17015++) {
int32_t x17027 = x17008 * x17015;
int32_t x17028 = x17026 + x17027;
float x17030 = x16907[x17028];
float x17031 = x16950[x17029];
int32_t x17017 = x17015 + x17016;
int32_t x17019 = x17017 + x17018;
int32_t x17021 = x17019 + x17020;
float x17032 = x17030 / x17031;
x17005[x17021] = x17032;

}

}

}

}
int32_t x17042 = 0;
int32_t x17043 = 1;
x17043 *= 1;
x17042 += 1;
x17043 *= 1;
x17043 *= 1;
int32_t x17048 = x17042;
bool x17049 = x17048 >= 2;
if (x17049) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17054 = x17048 == 0;
if (x17054) {
int32_t x17055 = x17043;
bool x17056 = x17055 == 256;
if (x17056) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17063 = x17043;
bool x17065 = x16999 == 1;
int32_t x17064 = 256 / x17063;
bool x17066 = x17064 == 1;
bool x17070;
if (x454) {
bool x17067 = x17065 || x17066;
bool x17068 = x16999 == x17064;
bool x17069 = x17067 || x17068;
x17070 = x17069;
} else {
x17070 = false;
}
bool x17074;
if (x17070) {
x17074 = x17073;
} else {
x17074 = false;
}
bool x17075;
if (x17074) {
x17075 = x17073;
} else {
x17075 = false;
}
if (x17075) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16999,x17001,x17001,1,x17064,1,1);
assert(false && "");
}
bool x17081 = x16999 <= x17064;
int32_t x17082;
if (x17081) {
x17082 = x17064;
} else {
x17082 = x16999;
}
int32_t x17086 = x17082 * x17085;
int32_t x17087 = 64 * x17086;
float* x17088 = (float*)myMalloc(x17087 * sizeof(float));;
int32_t x17089;
if (x17065) {
x17089 = 0;
} else {
x17089 = x17002;
}
int32_t x17092;
if (x17066) {
x17092 = 0;
} else {
x17092 = 1;
}
for(int x17093=0; x17093 < 64; x17093++) {
int32_t x17105 = x17003 * x17093;
int32_t x17099 = x17086 * x17093;
for(int x17095=0; x17095 < x17082; x17095++) {
int32_t x17106 = x17089 * x17095;
int32_t x17107 = x17105 + x17106;
int32_t x17112 = x17092 * x17095;
int32_t x17101 = x17085 * x17095;
for(int x17097=0; x17097 < x17084; x17097++) {
int32_t x17108 = x17090 * x17097;
int32_t x17109 = x17107 + x17108;
int32_t x17103 = x17084 * x17097;
for(int x17098=0; x17098 < x17084; x17098++) {
int32_t x17110 = x17091 * x17098;
int32_t x17111 = x17109 + x17110;
float x17113 = x17005[x17111];
float x17114 = x77[x17112];
int32_t x17100 = x17098 + x17099;
int32_t x17102 = x17100 + x17101;
int32_t x17104 = x17102 + x17103;
float x17115 = x17113 * x17114;
x17088[x17104] = x17115;

}

}

}

}
int32_t x17125 = 0;
int32_t x17126 = 1;
x17126 *= 1;
x17125 += 1;
x17126 *= 1;
x17126 *= 1;
int32_t x17131 = x17125;
bool x17132 = x17131 >= 2;
if (x17132) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17137 = x17131 == 0;
if (x17137) {
int32_t x17138 = x17126;
bool x17139 = x17138 == 256;
if (x17139) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17146 = x17126;
bool x17148 = x17082 == 1;
int32_t x17147 = 256 / x17146;
bool x17149 = x17147 == 1;
bool x17153;
if (x454) {
bool x17150 = x17148 || x17149;
bool x17151 = x17082 == x17147;
bool x17152 = x17150 || x17151;
x17153 = x17152;
} else {
x17153 = false;
}
bool x17157;
if (x17153) {
x17157 = x17156;
} else {
x17157 = false;
}
bool x17158;
if (x17157) {
x17158 = x17156;
} else {
x17158 = false;
}
if (x17158) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17082,x17084,x17084,1,x17147,1,1);
assert(false && "");
}
bool x17164 = x17082 <= x17147;
int32_t x17165;
if (x17164) {
x17165 = x17147;
} else {
x17165 = x17082;
}
int32_t x17169 = x17165 * x17168;
int32_t x17170 = 64 * x17169;
float* x17171 = (float*)myMalloc(x17170 * sizeof(float));;
int32_t x17172;
if (x17148) {
x17172 = 0;
} else {
x17172 = x17085;
}
int32_t x17175;
if (x17149) {
x17175 = 0;
} else {
x17175 = 1;
}
for(int x17176=0; x17176 < 64; x17176++) {
int32_t x17188 = x17086 * x17176;
int32_t x17182 = x17169 * x17176;
for(int x17178=0; x17178 < x17165; x17178++) {
int32_t x17189 = x17172 * x17178;
int32_t x17190 = x17188 + x17189;
int32_t x17195 = x17175 * x17178;
int32_t x17184 = x17168 * x17178;
for(int x17180=0; x17180 < x17167; x17180++) {
int32_t x17191 = x17173 * x17180;
int32_t x17192 = x17190 + x17191;
int32_t x17186 = x17167 * x17180;
for(int x17181=0; x17181 < x17167; x17181++) {
int32_t x17193 = x17174 * x17181;
int32_t x17194 = x17192 + x17193;
float x17196 = x17088[x17194];
float x17197 = x185[x17195];
int32_t x17183 = x17181 + x17182;
int32_t x17185 = x17183 + x17184;
int32_t x17187 = x17185 + x17186;
float x17198 = x17196 + x17197;
x17171[x17187] = x17198;

}

}

}

}
float* x17208 = (float*)myMalloc(x17170 * sizeof(float));;
for(int x17210=0; x17210 < x17170; x17210++) {
float x17211 = x17171[x17210];
bool x17212 = x17211 < 0.0f;
if (x17212) {
x17208[x17210] = 0.0f;
} else {
float x17215 = x17171[x17210];
x17208[x17210] = x17215;
}

}
float* x17230 = (float*)myMalloc(x17229 * sizeof(float));;
int32_t x17231 = 9 * x17165;
int32_t x17234 = 64 * x17231;
int32_t x17235 = x17234 * x17225;
float* x17236 = (float*)myMalloc(x17235 * sizeof(float));;
int32_t x17232 = x17231 * x17225;
int32_t x17244 = x17165 * 3;
int32_t x17245 = x17244 * 3;
for(int x17237=0; x17237 < 64; x17237++) {
int32_t x17238 = x17237 * x17169;
float* x17239 = x17208+x17238;
int32_t x17240 = x17237 * x17226;
float* x17241 = x17230+x17240;
int32_t x17242 = x17237 * x17232;
float* x17243 = x17236+x17242;
for(int x17247=0; x17247 < x17245; x17247++) {
int32_t x17248 = x17247 / 9;
int32_t x17252 = x17248 * 3;
int32_t x17253 = x17252 * 3;
int32_t x17254 = x17253 * x17224;
int32_t x17255 = x17254 * x17224;
int32_t x17249 = x17247 % 9;
int32_t x17250 = x17249 / 3;
int32_t x17256 = x17250 * 3;
int32_t x17257 = x17256 * x17224;
int32_t x17258 = x17257 * x17224;
int32_t x17259 = x17255 + x17258;
int32_t x17251 = x17249 % 3;
int32_t x17260 = x17251 * x17224;
int32_t x17261 = x17260 * x17224;
int32_t x17262 = x17259 + x17261;
float* x17263 = x17243+x17262;
int32_t x17264 = x17248 * x17167;
int32_t x17265 = x17264 * x17167;
float* x17266 = x17239+x17265;
int32_t x17279 = 1 - x17251;
bool x17280 = x17279 > 0;
int32_t x17281;
if (x17280) {
x17281 = x17279;
} else {
x17281 = 0;
}
int32_t x17282 = 3 - x17251;
int32_t x17283 = x17282 - 1;
int32_t x17284 = 1 - x17283;
bool x17285 = x17284 > 0;
int32_t x17286;
if (x17285) {
x17286 = x17284;
} else {
x17286 = 0;
}
int32_t x17287 = x17224 - x17286;
int32_t x17288 = x17287 - x17281;
bool x17289 = x17288 <= 0;
bool x17293 = x17281 > 0;
int32_t x17278 = -1 + x17251;
bool x17306 = x17286 > 0;
for(int x17268=0; x17268 < x17224; x17268++) {
int32_t x17269 = x17268 - 1;
int32_t x17270 = x17269 + x17250;
bool x17271 = x17270 < 0;
bool x17272 = x17270 >= x17167;
bool x17273 = x17271 || x17272;
if (x17273) {
int32_t x17274 = x17268 * x17224;
float* x17275 = x17263+x17274;
memset(x17275, 0, 4 * x17224);;
} else {
if (x17289) {
int32_t x17274 = x17268 * x17224;
float* x17290 = x17263+x17274;
memset(x17290, 0, 4 * x17224);;
} else {
int32_t x17274 = x17268 * x17224;
if (x17293) {
float* x17294 = x17263+x17274;
memset(x17294, 0, 4 * x17281);;
} else {
}
// may have segfault here
int32_t x17299 = x17274 + x17281;
float* x17300 = x17263+x17299;
int32_t x17301 = x17270 * x17167;
int32_t x17302 = x17301 + x17278;
int32_t x17303 = x17302 + x17281;
float* x17304 = x17266+x17303;
memcpy(x17300, x17304, 4 * x17288);;
if (x17306) {
int32_t x17307 = x17274 + x17224;
int32_t x17308 = x17307 - x17286;
float* x17309 = x17263+x17308;
memset(x17309, 0, 4 * x17286);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x17225,x17231,1,x262,x17231,x17243,x17225,1,x17241,x17225);

}
int32_t x17324 = 0;
int32_t x17325 = 1;
x17325 *= 1;
x17324 += 1;
x17325 *= 1;
x17325 *= 1;
int32_t x17330 = x17324;
bool x17331 = x17330 >= 2;
if (x17331) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17336 = x17330 == 0;
if (x17336) {
int32_t x17337 = x17325;
bool x17338 = x17337 == 256;
if (x17338) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17345 = x17325;
int32_t x17346 = 256 / x17345;
bool x17347 = x17346 == 1;
bool x17350;
if (x454) {
bool x17348 = 256 == x17346;
bool x17349 = x17347 || x17348;
x17350 = x17349;
} else {
x17350 = false;
}
bool x17354;
if (x17350) {
x17354 = x17353;
} else {
x17354 = false;
}
bool x17355;
if (x17354) {
x17355 = x17353;
} else {
x17355 = false;
}
if (x17355) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x17224,x17224,1,x17346,1,1);
assert(false && "");
}
bool x17361 = 256 <= x17346;
int32_t x17362;
if (x17361) {
x17362 = x17346;
} else {
x17362 = 256;
}
int32_t x17366 = x17362 * x17365;
int32_t x17367 = 64 * x17366;
float* x17368 = (float*)myMalloc(x17367 * sizeof(float));;
int32_t x17371;
if (x17347) {
x17371 = 0;
} else {
x17371 = 1;
}
for(int x17372=0; x17372 < 64; x17372++) {
int32_t x17384 = x17226 * x17372;
int32_t x17378 = x17366 * x17372;
for(int x17374=0; x17374 < x17362; x17374++) {
int32_t x17385 = x17225 * x17374;
int32_t x17386 = x17384 + x17385;
int32_t x17391 = x17371 * x17374;
int32_t x17380 = x17365 * x17374;
for(int x17376=0; x17376 < x17364; x17376++) {
int32_t x17387 = x17369 * x17376;
int32_t x17388 = x17386 + x17387;
int32_t x17382 = x17364 * x17376;
for(int x17377=0; x17377 < x17364; x17377++) {
int32_t x17389 = x17370 * x17377;
int32_t x17390 = x17388 + x17389;
float x17392 = x17230[x17390];
float x17393 = x250[x17391];
int32_t x17379 = x17377 + x17378;
int32_t x17381 = x17379 + x17380;
int32_t x17383 = x17381 + x17382;
float x17394 = x17392 - x17393;
x17368[x17383] = x17394;

}

}

}

}
float* x17404 = (float*)myMalloc(256 * sizeof(float));;
for(int x17405=0; x17405 < 256; x17405++) {
float x17406 = x104[x17405];
float x17407 = x17406 + 1.0E-5f;
x17404[x17405] = x17407;

}
float* x17411 = (float*)myMalloc(256 * sizeof(float));;
for(int x17412=0; x17412 < 256; x17412++) {
float x17413 = x17404[x17412];
double x17414 = (double)x17413;
double x17415 = sqrt(x17414);
float x17416 = (float)x17415;
x17411[x17412] = x17416;

}
int32_t x17420 = 0;
int32_t x17421 = 1;
x17421 *= 1;
x17420 += 1;
x17421 *= 1;
x17421 *= 1;
int32_t x17426 = x17420;
bool x17427 = x17426 >= 2;
if (x17427) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17432 = x17426 == 0;
if (x17432) {
int32_t x17433 = x17421;
bool x17434 = x17433 == 256;
if (x17434) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17441 = x17421;
bool x17443 = x17362 == 1;
int32_t x17442 = 256 / x17441;
bool x17444 = x17442 == 1;
bool x17448;
if (x454) {
bool x17445 = x17443 || x17444;
bool x17446 = x17362 == x17442;
bool x17447 = x17445 || x17446;
x17448 = x17447;
} else {
x17448 = false;
}
bool x17452;
if (x17448) {
x17452 = x17451;
} else {
x17452 = false;
}
bool x17453;
if (x17452) {
x17453 = x17451;
} else {
x17453 = false;
}
if (x17453) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17362,x17364,x17364,1,x17442,1,1);
assert(false && "");
}
bool x17459 = x17362 <= x17442;
int32_t x17460;
if (x17459) {
x17460 = x17442;
} else {
x17460 = x17362;
}
int32_t x17464 = x17460 * x17463;
int32_t x17465 = 64 * x17464;
float* x17466 = (float*)myMalloc(x17465 * sizeof(float));;
int32_t x17467;
if (x17443) {
x17467 = 0;
} else {
x17467 = x17365;
}
int32_t x17470;
if (x17444) {
x17470 = 0;
} else {
x17470 = 1;
}
for(int x17471=0; x17471 < 64; x17471++) {
int32_t x17483 = x17366 * x17471;
int32_t x17477 = x17464 * x17471;
for(int x17473=0; x17473 < x17460; x17473++) {
int32_t x17484 = x17467 * x17473;
int32_t x17485 = x17483 + x17484;
int32_t x17490 = x17470 * x17473;
int32_t x17479 = x17463 * x17473;
for(int x17475=0; x17475 < x17462; x17475++) {
int32_t x17486 = x17468 * x17475;
int32_t x17487 = x17485 + x17486;
int32_t x17481 = x17462 * x17475;
for(int x17476=0; x17476 < x17462; x17476++) {
int32_t x17488 = x17469 * x17476;
int32_t x17489 = x17487 + x17488;
float x17491 = x17368[x17489];
float x17492 = x17411[x17490];
int32_t x17478 = x17476 + x17477;
int32_t x17480 = x17478 + x17479;
int32_t x17482 = x17480 + x17481;
float x17493 = x17491 / x17492;
x17466[x17482] = x17493;

}

}

}

}
int32_t x17503 = 0;
int32_t x17504 = 1;
x17504 *= 1;
x17503 += 1;
x17504 *= 1;
x17504 *= 1;
int32_t x17509 = x17503;
bool x17510 = x17509 >= 2;
if (x17510) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17515 = x17509 == 0;
if (x17515) {
int32_t x17516 = x17504;
bool x17517 = x17516 == 256;
if (x17517) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17524 = x17504;
bool x17526 = x17460 == 1;
int32_t x17525 = 256 / x17524;
bool x17527 = x17525 == 1;
bool x17531;
if (x454) {
bool x17528 = x17526 || x17527;
bool x17529 = x17460 == x17525;
bool x17530 = x17528 || x17529;
x17531 = x17530;
} else {
x17531 = false;
}
bool x17535;
if (x17531) {
x17535 = x17534;
} else {
x17535 = false;
}
bool x17536;
if (x17535) {
x17536 = x17534;
} else {
x17536 = false;
}
if (x17536) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17460,x17462,x17462,1,x17525,1,1);
assert(false && "");
}
bool x17542 = x17460 <= x17525;
int32_t x17543;
if (x17542) {
x17543 = x17525;
} else {
x17543 = x17460;
}
int32_t x17547 = x17543 * x17546;
int32_t x17548 = 64 * x17547;
float* x17549 = (float*)myMalloc(x17548 * sizeof(float));;
int32_t x17550;
if (x17526) {
x17550 = 0;
} else {
x17550 = x17463;
}
int32_t x17553;
if (x17527) {
x17553 = 0;
} else {
x17553 = 1;
}
for(int x17554=0; x17554 < 64; x17554++) {
int32_t x17566 = x17464 * x17554;
int32_t x17560 = x17547 * x17554;
for(int x17556=0; x17556 < x17543; x17556++) {
int32_t x17567 = x17550 * x17556;
int32_t x17568 = x17566 + x17567;
int32_t x17573 = x17553 * x17556;
int32_t x17562 = x17546 * x17556;
for(int x17558=0; x17558 < x17545; x17558++) {
int32_t x17569 = x17551 * x17558;
int32_t x17570 = x17568 + x17569;
int32_t x17564 = x17545 * x17558;
for(int x17559=0; x17559 < x17545; x17559++) {
int32_t x17571 = x17552 * x17559;
int32_t x17572 = x17570 + x17571;
float x17574 = x17466[x17572];
float x17575 = x168[x17573];
int32_t x17561 = x17559 + x17560;
int32_t x17563 = x17561 + x17562;
int32_t x17565 = x17563 + x17564;
float x17576 = x17574 * x17575;
x17549[x17565] = x17576;

}

}

}

}
int32_t x17586 = 0;
int32_t x17587 = 1;
x17587 *= 1;
x17586 += 1;
x17587 *= 1;
x17587 *= 1;
int32_t x17592 = x17586;
bool x17593 = x17592 >= 2;
if (x17593) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17598 = x17592 == 0;
if (x17598) {
int32_t x17599 = x17587;
bool x17600 = x17599 == 256;
if (x17600) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17607 = x17587;
bool x17609 = x17543 == 1;
int32_t x17608 = 256 / x17607;
bool x17610 = x17608 == 1;
bool x17614;
if (x454) {
bool x17611 = x17609 || x17610;
bool x17612 = x17543 == x17608;
bool x17613 = x17611 || x17612;
x17614 = x17613;
} else {
x17614 = false;
}
bool x17618;
if (x17614) {
x17618 = x17617;
} else {
x17618 = false;
}
bool x17619;
if (x17618) {
x17619 = x17617;
} else {
x17619 = false;
}
if (x17619) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17543,x17545,x17545,1,x17608,1,1);
assert(false && "");
}
bool x17625 = x17543 <= x17608;
int32_t x17626;
if (x17625) {
x17626 = x17608;
} else {
x17626 = x17543;
}
int32_t x17630 = x17626 * x17629;
int32_t x17631 = 64 * x17630;
float* x17632 = (float*)myMalloc(x17631 * sizeof(float));;
int32_t x17633;
if (x17609) {
x17633 = 0;
} else {
x17633 = x17546;
}
int32_t x17636;
if (x17610) {
x17636 = 0;
} else {
x17636 = 1;
}
for(int x17637=0; x17637 < 64; x17637++) {
int32_t x17649 = x17547 * x17637;
int32_t x17643 = x17630 * x17637;
for(int x17639=0; x17639 < x17626; x17639++) {
int32_t x17650 = x17633 * x17639;
int32_t x17651 = x17649 + x17650;
int32_t x17656 = x17636 * x17639;
int32_t x17645 = x17629 * x17639;
for(int x17641=0; x17641 < x17628; x17641++) {
int32_t x17652 = x17634 * x17641;
int32_t x17653 = x17651 + x17652;
int32_t x17647 = x17628 * x17641;
for(int x17642=0; x17642 < x17628; x17642++) {
int32_t x17654 = x17635 * x17642;
int32_t x17655 = x17653 + x17654;
float x17657 = x17549[x17655];
float x17658 = x109[x17656];
int32_t x17644 = x17642 + x17643;
int32_t x17646 = x17644 + x17645;
int32_t x17648 = x17646 + x17647;
float x17659 = x17657 + x17658;
x17632[x17648] = x17659;

}

}

}

}
float* x17669 = (float*)myMalloc(x17631 * sizeof(float));;
for(int x17671=0; x17671 < x17631; x17671++) {
float x17672 = x17632[x17671];
bool x17673 = x17672 < 0.0f;
if (x17673) {
x17669[x17671] = 0.0f;
} else {
float x17676 = x17632[x17671];
x17669[x17671] = x17676;
}

}
float* x17690 = (float*)myMalloc(x17689 * sizeof(float));;
int32_t x17693 = 64 * x17626;
int32_t x17694 = x17693 * x17685;
float* x17695 = (float*)myMalloc(x17694 * sizeof(float));;
int32_t x17691 = x17626 * x17685;
for(int x17696=0; x17696 < 64; x17696++) {
int32_t x17697 = x17696 * x17630;
float* x17698 = x17669+x17697;
int32_t x17699 = x17696 * x17686;
float* x17700 = x17690+x17699;
int32_t x17701 = x17696 * x17691;
float* x17702 = x17695+x17701;
for(int x17703=0; x17703 < x17626; x17703++) {
int32_t x17704 = x17703 / 1;
int32_t x17708 = x17704 * x17684;
int32_t x17709 = x17708 * x17684;
int32_t x17705 = x17703 % 1;
int32_t x17706 = x17705 / 1;
int32_t x17710 = x17706 * x17684;
int32_t x17711 = x17710 * x17684;
int32_t x17712 = x17709 + x17711;
int32_t x17707 = x17705 % 1;
int32_t x17713 = x17707 * x17684;
int32_t x17714 = x17713 * x17684;
int32_t x17715 = x17712 + x17714;
float* x17716 = x17702+x17715;
int32_t x17717 = x17704 * x17628;
int32_t x17718 = x17717 * x17628;
float* x17719 = x17698+x17718;
for(int x17721=0; x17721 < x17684; x17721++) {
int32_t x17723 = x17721 * x17684;
float* x17724 = x17716+x17723;
int32_t x17722 = x17721 + x17706;
int32_t x17725 = x17722 * x17628;
int32_t x17726 = x17725 + x17707;
float* x17727 = x17719+x17726;
memcpy(x17724, x17727, 4 * x17684);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x17685,x17626,1,x221,x17626,x17702,x17685,1,x17700,x17685);

}
int32_t x17736 = 0;
int32_t x17737 = 1;
x17737 *= 1;
x17736 += 1;
x17737 *= 1;
x17737 *= 1;
int32_t x17742 = x17736;
bool x17743 = x17742 >= 2;
if (x17743) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17748 = x17742 == 0;
if (x17748) {
int32_t x17749 = x17737;
bool x17750 = x17749 == 1024;
if (x17750) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17757 = x17737;
int32_t x17758 = 1024 / x17757;
bool x17759 = x17758 == 1;
bool x17762;
if (x454) {
bool x17760 = 1024 == x17758;
bool x17761 = x17759 || x17760;
x17762 = x17761;
} else {
x17762 = false;
}
bool x17766;
if (x17762) {
x17766 = x17765;
} else {
x17766 = false;
}
bool x17767;
if (x17766) {
x17767 = x17765;
} else {
x17767 = false;
}
if (x17767) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x17684,x17684,1,x17758,1,1);
assert(false && "");
}
bool x17773 = 1024 <= x17758;
int32_t x17774;
if (x17773) {
x17774 = x17758;
} else {
x17774 = 1024;
}
int32_t x17778 = x17774 * x17777;
int32_t x17779 = 64 * x17778;
float* x17780 = (float*)myMalloc(x17779 * sizeof(float));;
int32_t x17783;
if (x17759) {
x17783 = 0;
} else {
x17783 = 1;
}
for(int x17784=0; x17784 < 64; x17784++) {
int32_t x17796 = x17686 * x17784;
int32_t x17790 = x17778 * x17784;
for(int x17786=0; x17786 < x17774; x17786++) {
int32_t x17797 = x17685 * x17786;
int32_t x17798 = x17796 + x17797;
int32_t x17803 = x17783 * x17786;
int32_t x17792 = x17777 * x17786;
for(int x17788=0; x17788 < x17776; x17788++) {
int32_t x17799 = x17781 * x17788;
int32_t x17800 = x17798 + x17799;
int32_t x17794 = x17776 * x17788;
for(int x17789=0; x17789 < x17776; x17789++) {
int32_t x17801 = x17782 * x17789;
int32_t x17802 = x17800 + x17801;
float x17804 = x17690[x17802];
float x17805 = x209[x17803];
int32_t x17791 = x17789 + x17790;
int32_t x17793 = x17791 + x17792;
int32_t x17795 = x17793 + x17794;
float x17806 = x17804 - x17805;
x17780[x17795] = x17806;

}

}

}

}
float* x17816 = (float*)myMalloc(1024 * sizeof(float));;
for(int x17817=0; x17817 < 1024; x17817++) {
float x17818 = x272[x17817];
float x17819 = x17818 + 1.0E-5f;
x17816[x17817] = x17819;

}
float* x17823 = (float*)myMalloc(1024 * sizeof(float));;
for(int x17824=0; x17824 < 1024; x17824++) {
float x17825 = x17816[x17824];
double x17826 = (double)x17825;
double x17827 = sqrt(x17826);
float x17828 = (float)x17827;
x17823[x17824] = x17828;

}
int32_t x17832 = 0;
int32_t x17833 = 1;
x17833 *= 1;
x17832 += 1;
x17833 *= 1;
x17833 *= 1;
int32_t x17838 = x17832;
bool x17839 = x17838 >= 2;
if (x17839) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17844 = x17838 == 0;
if (x17844) {
int32_t x17845 = x17833;
bool x17846 = x17845 == 1024;
if (x17846) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17853 = x17833;
bool x17855 = x17774 == 1;
int32_t x17854 = 1024 / x17853;
bool x17856 = x17854 == 1;
bool x17860;
if (x454) {
bool x17857 = x17855 || x17856;
bool x17858 = x17774 == x17854;
bool x17859 = x17857 || x17858;
x17860 = x17859;
} else {
x17860 = false;
}
bool x17864;
if (x17860) {
x17864 = x17863;
} else {
x17864 = false;
}
bool x17865;
if (x17864) {
x17865 = x17863;
} else {
x17865 = false;
}
if (x17865) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17774,x17776,x17776,1,x17854,1,1);
assert(false && "");
}
bool x17871 = x17774 <= x17854;
int32_t x17872;
if (x17871) {
x17872 = x17854;
} else {
x17872 = x17774;
}
int32_t x17876 = x17872 * x17875;
int32_t x17877 = 64 * x17876;
float* x17878 = (float*)myMalloc(x17877 * sizeof(float));;
int32_t x17879;
if (x17855) {
x17879 = 0;
} else {
x17879 = x17777;
}
int32_t x17882;
if (x17856) {
x17882 = 0;
} else {
x17882 = 1;
}
for(int x17883=0; x17883 < 64; x17883++) {
int32_t x17895 = x17778 * x17883;
int32_t x17889 = x17876 * x17883;
for(int x17885=0; x17885 < x17872; x17885++) {
int32_t x17896 = x17879 * x17885;
int32_t x17897 = x17895 + x17896;
int32_t x17902 = x17882 * x17885;
int32_t x17891 = x17875 * x17885;
for(int x17887=0; x17887 < x17874; x17887++) {
int32_t x17898 = x17880 * x17887;
int32_t x17899 = x17897 + x17898;
int32_t x17893 = x17874 * x17887;
for(int x17888=0; x17888 < x17874; x17888++) {
int32_t x17900 = x17881 * x17888;
int32_t x17901 = x17899 + x17900;
float x17903 = x17780[x17901];
float x17904 = x17823[x17902];
int32_t x17890 = x17888 + x17889;
int32_t x17892 = x17890 + x17891;
int32_t x17894 = x17892 + x17893;
float x17905 = x17903 / x17904;
x17878[x17894] = x17905;

}

}

}

}
int32_t x17915 = 0;
int32_t x17916 = 1;
x17916 *= 1;
x17915 += 1;
x17916 *= 1;
x17916 *= 1;
int32_t x17921 = x17915;
bool x17922 = x17921 >= 2;
if (x17922) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17927 = x17921 == 0;
if (x17927) {
int32_t x17928 = x17916;
bool x17929 = x17928 == 1024;
if (x17929) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17936 = x17916;
bool x17938 = x17872 == 1;
int32_t x17937 = 1024 / x17936;
bool x17939 = x17937 == 1;
bool x17943;
if (x454) {
bool x17940 = x17938 || x17939;
bool x17941 = x17872 == x17937;
bool x17942 = x17940 || x17941;
x17943 = x17942;
} else {
x17943 = false;
}
bool x17947;
if (x17943) {
x17947 = x17946;
} else {
x17947 = false;
}
bool x17948;
if (x17947) {
x17948 = x17946;
} else {
x17948 = false;
}
if (x17948) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17872,x17874,x17874,1,x17937,1,1);
assert(false && "");
}
bool x17954 = x17872 <= x17937;
int32_t x17955;
if (x17954) {
x17955 = x17937;
} else {
x17955 = x17872;
}
int32_t x17959 = x17955 * x17958;
int32_t x17960 = 64 * x17959;
float* x17961 = (float*)myMalloc(x17960 * sizeof(float));;
int32_t x17962;
if (x17938) {
x17962 = 0;
} else {
x17962 = x17875;
}
int32_t x17965;
if (x17939) {
x17965 = 0;
} else {
x17965 = 1;
}
for(int x17966=0; x17966 < 64; x17966++) {
int32_t x17978 = x17876 * x17966;
int32_t x17972 = x17959 * x17966;
for(int x17968=0; x17968 < x17955; x17968++) {
int32_t x17979 = x17962 * x17968;
int32_t x17980 = x17978 + x17979;
int32_t x17985 = x17965 * x17968;
int32_t x17974 = x17958 * x17968;
for(int x17970=0; x17970 < x17957; x17970++) {
int32_t x17981 = x17963 * x17970;
int32_t x17982 = x17980 + x17981;
int32_t x17976 = x17957 * x17970;
for(int x17971=0; x17971 < x17957; x17971++) {
int32_t x17983 = x17964 * x17971;
int32_t x17984 = x17982 + x17983;
float x17986 = x17878[x17984];
float x17987 = x59[x17985];
int32_t x17973 = x17971 + x17972;
int32_t x17975 = x17973 + x17974;
int32_t x17977 = x17975 + x17976;
float x17988 = x17986 * x17987;
x17961[x17977] = x17988;

}

}

}

}
int32_t x17998 = 0;
int32_t x17999 = 1;
x17999 *= 1;
x17998 += 1;
x17999 *= 1;
x17999 *= 1;
int32_t x18004 = x17998;
bool x18005 = x18004 >= 2;
if (x18005) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18010 = x18004 == 0;
if (x18010) {
int32_t x18011 = x17999;
bool x18012 = x18011 == 1024;
if (x18012) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18019 = x17999;
bool x18021 = x17955 == 1;
int32_t x18020 = 1024 / x18019;
bool x18022 = x18020 == 1;
bool x18026;
if (x454) {
bool x18023 = x18021 || x18022;
bool x18024 = x17955 == x18020;
bool x18025 = x18023 || x18024;
x18026 = x18025;
} else {
x18026 = false;
}
bool x18030;
if (x18026) {
x18030 = x18029;
} else {
x18030 = false;
}
bool x18031;
if (x18030) {
x18031 = x18029;
} else {
x18031 = false;
}
if (x18031) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17955,x17957,x17957,1,x18020,1,1);
assert(false && "");
}
bool x18037 = x17955 <= x18020;
int32_t x18038;
if (x18037) {
x18038 = x18020;
} else {
x18038 = x17955;
}
int32_t x18042 = x18038 * x18041;
int32_t x18043 = 64 * x18042;
float* x18044 = (float*)myMalloc(x18043 * sizeof(float));;
int32_t x18045;
if (x18021) {
x18045 = 0;
} else {
x18045 = x17958;
}
int32_t x18048;
if (x18022) {
x18048 = 0;
} else {
x18048 = 1;
}
for(int x18049=0; x18049 < 64; x18049++) {
int32_t x18061 = x17959 * x18049;
int32_t x18055 = x18042 * x18049;
for(int x18051=0; x18051 < x18038; x18051++) {
int32_t x18062 = x18045 * x18051;
int32_t x18063 = x18061 + x18062;
int32_t x18068 = x18048 * x18051;
int32_t x18057 = x18041 * x18051;
for(int x18053=0; x18053 < x18040; x18053++) {
int32_t x18064 = x18046 * x18053;
int32_t x18065 = x18063 + x18064;
int32_t x18059 = x18040 * x18053;
for(int x18054=0; x18054 < x18040; x18054++) {
int32_t x18066 = x18047 * x18054;
int32_t x18067 = x18065 + x18066;
float x18069 = x17961[x18067];
float x18070 = x120[x18068];
int32_t x18056 = x18054 + x18055;
int32_t x18058 = x18056 + x18057;
int32_t x18060 = x18058 + x18059;
float x18071 = x18069 + x18070;
x18044[x18060] = x18071;

}

}

}

}
bool x18081 = x18038 == 1;
bool x18082 = x18081 || x16739;
bool x18083 = x18038 == x16696;
bool x18084 = x18082 || x18083;
bool x18089;
if (x18084) {
x18089 = x18088;
} else {
x18089 = false;
}
bool x18090;
if (x18089) {
x18090 = x18088;
} else {
x18090 = false;
}
if (x18090) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18038,x18040,x18040,64,x16696,x16698,x16698);
assert(false && "");
}
bool x18096 = x18038 <= x16696;
int32_t x18097;
if (x18096) {
x18097 = x16696;
} else {
x18097 = x18038;
}
int32_t x18103;
if (x18081) {
x18103 = 0;
} else {
x18103 = x18041;
}
for(int x18106=0; x18106 < 64; x18106++) {
int32_t x18112 = x18042 * x18106;
int32_t x18119 = x16700 * x18106;
for(int x18108=0; x18108 < x18097; x18108++) {
int32_t x18113 = x18103 * x18108;
int32_t x18114 = x18112 + x18113;
int32_t x18120 = x16761 * x18108;
int32_t x18121 = x18119 + x18120;
for(int x18110=0; x18110 < x18099; x18110++) {
int32_t x18115 = x18104 * x18110;
int32_t x18116 = x18114 + x18115;
int32_t x18122 = x16762 * x18110;
int32_t x18123 = x18121 + x18122;
for(int x18111=0; x18111 < x18099; x18111++) {
int32_t x18117 = x18105 * x18111;
int32_t x18118 = x18116 + x18117;
float x18126 = x18044[x18118];
int32_t x18124 = x16763 * x18111;
int32_t x18125 = x18123 + x18124;
float x18127 = x16796[x18125];
float x18128 = x18126 + x18127;
x18044[x18118] = x18128;

}

}

}

}
float* x18138 = (float*)myMalloc(x18043 * sizeof(float));;
for(int x18140=0; x18140 < x18043; x18140++) {
float x18141 = x18044[x18140];
bool x18142 = x18141 < 0.0f;
if (x18142) {
x18138[x18140] = 0.0f;
} else {
float x18145 = x18044[x18140];
x18138[x18140] = x18145;
}

}
float* x18159 = (float*)myMalloc(x18158 * sizeof(float));;
int32_t x18162 = 64 * x18038;
int32_t x18163 = x18162 * x18154;
float* x18164 = (float*)myMalloc(x18163 * sizeof(float));;
int32_t x18160 = x18038 * x18154;
for(int x18165=0; x18165 < 64; x18165++) {
int32_t x18166 = x18165 * x18042;
float* x18167 = x18138+x18166;
int32_t x18168 = x18165 * x18155;
float* x18169 = x18159+x18168;
int32_t x18170 = x18165 * x18160;
float* x18171 = x18164+x18170;
for(int x18172=0; x18172 < x18038; x18172++) {
int32_t x18173 = x18172 / 1;
int32_t x18177 = x18173 * x18153;
int32_t x18178 = x18177 * x18153;
int32_t x18174 = x18172 % 1;
int32_t x18175 = x18174 / 1;
int32_t x18179 = x18175 * x18153;
int32_t x18180 = x18179 * x18153;
int32_t x18181 = x18178 + x18180;
int32_t x18176 = x18174 % 1;
int32_t x18182 = x18176 * x18153;
int32_t x18183 = x18182 * x18153;
int32_t x18184 = x18181 + x18183;
float* x18185 = x18171+x18184;
int32_t x18186 = x18173 * x18040;
int32_t x18187 = x18186 * x18040;
float* x18188 = x18167+x18187;
for(int x18190=0; x18190 < x18153; x18190++) {
int32_t x18192 = x18190 * x18153;
float* x18193 = x18185+x18192;
int32_t x18191 = x18190 + x18175;
int32_t x18194 = x18191 * x18040;
int32_t x18195 = x18194 + x18176;
float* x18196 = x18188+x18195;
memcpy(x18193, x18196, 4 * x18153);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x18154,x18038,1,x151,x18038,x18171,x18154,1,x18169,x18154);

}
int32_t x18205 = 0;
int32_t x18206 = 1;
x18206 *= 1;
x18205 += 1;
x18206 *= 1;
x18206 *= 1;
int32_t x18211 = x18205;
bool x18212 = x18211 >= 2;
if (x18212) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18217 = x18211 == 0;
if (x18217) {
int32_t x18218 = x18206;
bool x18219 = x18218 == 256;
if (x18219) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18226 = x18206;
int32_t x18227 = 256 / x18226;
bool x18228 = x18227 == 1;
bool x18231;
if (x454) {
bool x18229 = 256 == x18227;
bool x18230 = x18228 || x18229;
x18231 = x18230;
} else {
x18231 = false;
}
bool x18235;
if (x18231) {
x18235 = x18234;
} else {
x18235 = false;
}
bool x18236;
if (x18235) {
x18236 = x18234;
} else {
x18236 = false;
}
if (x18236) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x18153,x18153,1,x18227,1,1);
assert(false && "");
}
bool x18242 = 256 <= x18227;
int32_t x18243;
if (x18242) {
x18243 = x18227;
} else {
x18243 = 256;
}
int32_t x18247 = x18243 * x18246;
int32_t x18248 = 64 * x18247;
float* x18249 = (float*)myMalloc(x18248 * sizeof(float));;
int32_t x18252;
if (x18228) {
x18252 = 0;
} else {
x18252 = 1;
}
for(int x18253=0; x18253 < 64; x18253++) {
int32_t x18265 = x18155 * x18253;
int32_t x18259 = x18247 * x18253;
for(int x18255=0; x18255 < x18243; x18255++) {
int32_t x18266 = x18154 * x18255;
int32_t x18267 = x18265 + x18266;
int32_t x18272 = x18252 * x18255;
int32_t x18261 = x18246 * x18255;
for(int x18257=0; x18257 < x18245; x18257++) {
int32_t x18268 = x18250 * x18257;
int32_t x18269 = x18267 + x18268;
int32_t x18263 = x18245 * x18257;
for(int x18258=0; x18258 < x18245; x18258++) {
int32_t x18270 = x18251 * x18258;
int32_t x18271 = x18269 + x18270;
float x18273 = x18159[x18271];
float x18274 = x80[x18272];
int32_t x18260 = x18258 + x18259;
int32_t x18262 = x18260 + x18261;
int32_t x18264 = x18262 + x18263;
float x18275 = x18273 - x18274;
x18249[x18264] = x18275;

}

}

}

}
float* x18285 = (float*)myMalloc(256 * sizeof(float));;
for(int x18286=0; x18286 < 256; x18286++) {
float x18287 = x176[x18286];
float x18288 = x18287 + 1.0E-5f;
x18285[x18286] = x18288;

}
float* x18292 = (float*)myMalloc(256 * sizeof(float));;
for(int x18293=0; x18293 < 256; x18293++) {
float x18294 = x18285[x18293];
double x18295 = (double)x18294;
double x18296 = sqrt(x18295);
float x18297 = (float)x18296;
x18292[x18293] = x18297;

}
int32_t x18301 = 0;
int32_t x18302 = 1;
x18302 *= 1;
x18301 += 1;
x18302 *= 1;
x18302 *= 1;
int32_t x18307 = x18301;
bool x18308 = x18307 >= 2;
if (x18308) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18313 = x18307 == 0;
if (x18313) {
int32_t x18314 = x18302;
bool x18315 = x18314 == 256;
if (x18315) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18322 = x18302;
bool x18324 = x18243 == 1;
int32_t x18323 = 256 / x18322;
bool x18325 = x18323 == 1;
bool x18329;
if (x454) {
bool x18326 = x18324 || x18325;
bool x18327 = x18243 == x18323;
bool x18328 = x18326 || x18327;
x18329 = x18328;
} else {
x18329 = false;
}
bool x18333;
if (x18329) {
x18333 = x18332;
} else {
x18333 = false;
}
bool x18334;
if (x18333) {
x18334 = x18332;
} else {
x18334 = false;
}
if (x18334) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18243,x18245,x18245,1,x18323,1,1);
assert(false && "");
}
bool x18340 = x18243 <= x18323;
int32_t x18341;
if (x18340) {
x18341 = x18323;
} else {
x18341 = x18243;
}
int32_t x18345 = x18341 * x18344;
int32_t x18346 = 64 * x18345;
float* x18347 = (float*)myMalloc(x18346 * sizeof(float));;
int32_t x18348;
if (x18324) {
x18348 = 0;
} else {
x18348 = x18246;
}
int32_t x18351;
if (x18325) {
x18351 = 0;
} else {
x18351 = 1;
}
for(int x18352=0; x18352 < 64; x18352++) {
int32_t x18364 = x18247 * x18352;
int32_t x18358 = x18345 * x18352;
for(int x18354=0; x18354 < x18341; x18354++) {
int32_t x18365 = x18348 * x18354;
int32_t x18366 = x18364 + x18365;
int32_t x18371 = x18351 * x18354;
int32_t x18360 = x18344 * x18354;
for(int x18356=0; x18356 < x18343; x18356++) {
int32_t x18367 = x18349 * x18356;
int32_t x18368 = x18366 + x18367;
int32_t x18362 = x18343 * x18356;
for(int x18357=0; x18357 < x18343; x18357++) {
int32_t x18369 = x18350 * x18357;
int32_t x18370 = x18368 + x18369;
float x18372 = x18249[x18370];
float x18373 = x18292[x18371];
int32_t x18359 = x18357 + x18358;
int32_t x18361 = x18359 + x18360;
int32_t x18363 = x18361 + x18362;
float x18374 = x18372 / x18373;
x18347[x18363] = x18374;

}

}

}

}
int32_t x18384 = 0;
int32_t x18385 = 1;
x18385 *= 1;
x18384 += 1;
x18385 *= 1;
x18385 *= 1;
int32_t x18390 = x18384;
bool x18391 = x18390 >= 2;
if (x18391) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18396 = x18390 == 0;
if (x18396) {
int32_t x18397 = x18385;
bool x18398 = x18397 == 256;
if (x18398) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18405 = x18385;
bool x18407 = x18341 == 1;
int32_t x18406 = 256 / x18405;
bool x18408 = x18406 == 1;
bool x18412;
if (x454) {
bool x18409 = x18407 || x18408;
bool x18410 = x18341 == x18406;
bool x18411 = x18409 || x18410;
x18412 = x18411;
} else {
x18412 = false;
}
bool x18416;
if (x18412) {
x18416 = x18415;
} else {
x18416 = false;
}
bool x18417;
if (x18416) {
x18417 = x18415;
} else {
x18417 = false;
}
if (x18417) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18341,x18343,x18343,1,x18406,1,1);
assert(false && "");
}
bool x18423 = x18341 <= x18406;
int32_t x18424;
if (x18423) {
x18424 = x18406;
} else {
x18424 = x18341;
}
int32_t x18428 = x18424 * x18427;
int32_t x18429 = 64 * x18428;
float* x18430 = (float*)myMalloc(x18429 * sizeof(float));;
int32_t x18431;
if (x18407) {
x18431 = 0;
} else {
x18431 = x18344;
}
int32_t x18434;
if (x18408) {
x18434 = 0;
} else {
x18434 = 1;
}
for(int x18435=0; x18435 < 64; x18435++) {
int32_t x18447 = x18345 * x18435;
int32_t x18441 = x18428 * x18435;
for(int x18437=0; x18437 < x18424; x18437++) {
int32_t x18448 = x18431 * x18437;
int32_t x18449 = x18447 + x18448;
int32_t x18454 = x18434 * x18437;
int32_t x18443 = x18427 * x18437;
for(int x18439=0; x18439 < x18426; x18439++) {
int32_t x18450 = x18432 * x18439;
int32_t x18451 = x18449 + x18450;
int32_t x18445 = x18426 * x18439;
for(int x18440=0; x18440 < x18426; x18440++) {
int32_t x18452 = x18433 * x18440;
int32_t x18453 = x18451 + x18452;
float x18455 = x18347[x18453];
float x18456 = x85[x18454];
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
bool x18490 = x18424 == 1;
int32_t x18489 = 256 / x18488;
bool x18491 = x18489 == 1;
bool x18495;
if (x454) {
bool x18492 = x18490 || x18491;
bool x18493 = x18424 == x18489;
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
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18424,x18426,x18426,1,x18489,1,1);
assert(false && "");
}
bool x18506 = x18424 <= x18489;
int32_t x18507;
if (x18506) {
x18507 = x18489;
} else {
x18507 = x18424;
}
int32_t x18511 = x18507 * x18510;
int32_t x18512 = 64 * x18511;
float* x18513 = (float*)myMalloc(x18512 * sizeof(float));;
int32_t x18514;
if (x18490) {
x18514 = 0;
} else {
x18514 = x18427;
}
int32_t x18517;
if (x18491) {
x18517 = 0;
} else {
x18517 = 1;
}
for(int x18518=0; x18518 < 64; x18518++) {
int32_t x18530 = x18428 * x18518;
int32_t x18524 = x18511 * x18518;
for(int x18520=0; x18520 < x18507; x18520++) {
int32_t x18531 = x18514 * x18520;
int32_t x18532 = x18530 + x18531;
int32_t x18537 = x18517 * x18520;
int32_t x18526 = x18510 * x18520;
for(int x18522=0; x18522 < x18509; x18522++) {
int32_t x18533 = x18515 * x18522;
int32_t x18534 = x18532 + x18533;
int32_t x18528 = x18509 * x18522;
for(int x18523=0; x18523 < x18509; x18523++) {
int32_t x18535 = x18516 * x18523;
int32_t x18536 = x18534 + x18535;
float x18538 = x18430[x18536];
float x18539 = x253[x18537];
int32_t x18525 = x18523 + x18524;
int32_t x18527 = x18525 + x18526;
int32_t x18529 = x18527 + x18528;
float x18540 = x18538 + x18539;
x18513[x18529] = x18540;

}

}

}

}
float* x18550 = (float*)myMalloc(x18512 * sizeof(float));;
for(int x18552=0; x18552 < x18512; x18552++) {
float x18553 = x18513[x18552];
bool x18554 = x18553 < 0.0f;
if (x18554) {
x18550[x18552] = 0.0f;
} else {
float x18557 = x18513[x18552];
x18550[x18552] = x18557;
}

}
float* x18572 = (float*)myMalloc(x18571 * sizeof(float));;
int32_t x18573 = 9 * x18507;
int32_t x18576 = 64 * x18573;
int32_t x18577 = x18576 * x18567;
float* x18578 = (float*)myMalloc(x18577 * sizeof(float));;
int32_t x18574 = x18573 * x18567;
int32_t x18586 = x18507 * 3;
int32_t x18587 = x18586 * 3;
for(int x18579=0; x18579 < 64; x18579++) {
int32_t x18580 = x18579 * x18511;
float* x18581 = x18550+x18580;
int32_t x18582 = x18579 * x18568;
float* x18583 = x18572+x18582;
int32_t x18584 = x18579 * x18574;
float* x18585 = x18578+x18584;
for(int x18589=0; x18589 < x18587; x18589++) {
int32_t x18590 = x18589 / 9;
int32_t x18594 = x18590 * 3;
int32_t x18595 = x18594 * 3;
int32_t x18596 = x18595 * x18566;
int32_t x18597 = x18596 * x18566;
int32_t x18591 = x18589 % 9;
int32_t x18592 = x18591 / 3;
int32_t x18598 = x18592 * 3;
int32_t x18599 = x18598 * x18566;
int32_t x18600 = x18599 * x18566;
int32_t x18601 = x18597 + x18600;
int32_t x18593 = x18591 % 3;
int32_t x18602 = x18593 * x18566;
int32_t x18603 = x18602 * x18566;
int32_t x18604 = x18601 + x18603;
float* x18605 = x18585+x18604;
int32_t x18606 = x18590 * x18509;
int32_t x18607 = x18606 * x18509;
float* x18608 = x18581+x18607;
int32_t x18621 = 1 - x18593;
bool x18622 = x18621 > 0;
int32_t x18623;
if (x18622) {
x18623 = x18621;
} else {
x18623 = 0;
}
int32_t x18624 = 3 - x18593;
int32_t x18625 = x18624 - 1;
int32_t x18626 = 1 - x18625;
bool x18627 = x18626 > 0;
int32_t x18628;
if (x18627) {
x18628 = x18626;
} else {
x18628 = 0;
}
int32_t x18629 = x18566 - x18628;
int32_t x18630 = x18629 - x18623;
bool x18631 = x18630 <= 0;
bool x18635 = x18623 > 0;
int32_t x18620 = -1 + x18593;
bool x18648 = x18628 > 0;
for(int x18610=0; x18610 < x18566; x18610++) {
int32_t x18611 = x18610 - 1;
int32_t x18612 = x18611 + x18592;
bool x18613 = x18612 < 0;
bool x18614 = x18612 >= x18509;
bool x18615 = x18613 || x18614;
if (x18615) {
int32_t x18616 = x18610 * x18566;
float* x18617 = x18605+x18616;
memset(x18617, 0, 4 * x18566);;
} else {
if (x18631) {
int32_t x18616 = x18610 * x18566;
float* x18632 = x18605+x18616;
memset(x18632, 0, 4 * x18566);;
} else {
int32_t x18616 = x18610 * x18566;
if (x18635) {
float* x18636 = x18605+x18616;
memset(x18636, 0, 4 * x18623);;
} else {
}
// may have segfault here
int32_t x18641 = x18616 + x18623;
float* x18642 = x18605+x18641;
int32_t x18643 = x18612 * x18509;
int32_t x18644 = x18643 + x18620;
int32_t x18645 = x18644 + x18623;
float* x18646 = x18608+x18645;
memcpy(x18642, x18646, 4 * x18630);;
if (x18648) {
int32_t x18649 = x18616 + x18566;
int32_t x18650 = x18649 - x18628;
float* x18651 = x18605+x18650;
memset(x18651, 0, 4 * x18628);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x18567,x18573,1,x226,x18573,x18585,x18567,1,x18583,x18567);

}
int32_t x18666 = 0;
int32_t x18667 = 1;
x18667 *= 1;
x18666 += 1;
x18667 *= 1;
x18667 *= 1;
int32_t x18672 = x18666;
bool x18673 = x18672 >= 2;
if (x18673) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18678 = x18672 == 0;
if (x18678) {
int32_t x18679 = x18667;
bool x18680 = x18679 == 256;
if (x18680) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18687 = x18667;
int32_t x18688 = 256 / x18687;
bool x18689 = x18688 == 1;
bool x18692;
if (x454) {
bool x18690 = 256 == x18688;
bool x18691 = x18689 || x18690;
x18692 = x18691;
} else {
x18692 = false;
}
bool x18696;
if (x18692) {
x18696 = x18695;
} else {
x18696 = false;
}
bool x18697;
if (x18696) {
x18697 = x18695;
} else {
x18697 = false;
}
if (x18697) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x18566,x18566,1,x18688,1,1);
assert(false && "");
}
bool x18703 = 256 <= x18688;
int32_t x18704;
if (x18703) {
x18704 = x18688;
} else {
x18704 = 256;
}
int32_t x18708 = x18704 * x18707;
int32_t x18709 = 64 * x18708;
float* x18710 = (float*)myMalloc(x18709 * sizeof(float));;
int32_t x18713;
if (x18689) {
x18713 = 0;
} else {
x18713 = 1;
}
for(int x18714=0; x18714 < 64; x18714++) {
int32_t x18726 = x18568 * x18714;
int32_t x18720 = x18708 * x18714;
for(int x18716=0; x18716 < x18704; x18716++) {
int32_t x18727 = x18567 * x18716;
int32_t x18728 = x18726 + x18727;
int32_t x18733 = x18713 * x18716;
int32_t x18722 = x18707 * x18716;
for(int x18718=0; x18718 < x18706; x18718++) {
int32_t x18729 = x18711 * x18718;
int32_t x18730 = x18728 + x18729;
int32_t x18724 = x18706 * x18718;
for(int x18719=0; x18719 < x18706; x18719++) {
int32_t x18731 = x18712 * x18719;
int32_t x18732 = x18730 + x18731;
float x18734 = x18572[x18732];
float x18735 = x70[x18733];
int32_t x18721 = x18719 + x18720;
int32_t x18723 = x18721 + x18722;
int32_t x18725 = x18723 + x18724;
float x18736 = x18734 - x18735;
x18710[x18725] = x18736;

}

}

}

}
float* x18746 = (float*)myMalloc(256 * sizeof(float));;
for(int x18747=0; x18747 < 256; x18747++) {
float x18748 = x240[x18747];
float x18749 = x18748 + 1.0E-5f;
x18746[x18747] = x18749;

}
float* x18753 = (float*)myMalloc(256 * sizeof(float));;
for(int x18754=0; x18754 < 256; x18754++) {
float x18755 = x18746[x18754];
double x18756 = (double)x18755;
double x18757 = sqrt(x18756);
float x18758 = (float)x18757;
x18753[x18754] = x18758;

}
int32_t x18762 = 0;
int32_t x18763 = 1;
x18763 *= 1;
x18762 += 1;
x18763 *= 1;
x18763 *= 1;
int32_t x18768 = x18762;
bool x18769 = x18768 >= 2;
if (x18769) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18774 = x18768 == 0;
if (x18774) {
int32_t x18775 = x18763;
bool x18776 = x18775 == 256;
if (x18776) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18783 = x18763;
bool x18785 = x18704 == 1;
int32_t x18784 = 256 / x18783;
bool x18786 = x18784 == 1;
bool x18790;
if (x454) {
bool x18787 = x18785 || x18786;
bool x18788 = x18704 == x18784;
bool x18789 = x18787 || x18788;
x18790 = x18789;
} else {
x18790 = false;
}
bool x18794;
if (x18790) {
x18794 = x18793;
} else {
x18794 = false;
}
bool x18795;
if (x18794) {
x18795 = x18793;
} else {
x18795 = false;
}
if (x18795) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18704,x18706,x18706,1,x18784,1,1);
assert(false && "");
}
bool x18801 = x18704 <= x18784;
int32_t x18802;
if (x18801) {
x18802 = x18784;
} else {
x18802 = x18704;
}
int32_t x18806 = x18802 * x18805;
int32_t x18807 = 64 * x18806;
float* x18808 = (float*)myMalloc(x18807 * sizeof(float));;
int32_t x18809;
if (x18785) {
x18809 = 0;
} else {
x18809 = x18707;
}
int32_t x18812;
if (x18786) {
x18812 = 0;
} else {
x18812 = 1;
}
for(int x18813=0; x18813 < 64; x18813++) {
int32_t x18825 = x18708 * x18813;
int32_t x18819 = x18806 * x18813;
for(int x18815=0; x18815 < x18802; x18815++) {
int32_t x18826 = x18809 * x18815;
int32_t x18827 = x18825 + x18826;
int32_t x18832 = x18812 * x18815;
int32_t x18821 = x18805 * x18815;
for(int x18817=0; x18817 < x18804; x18817++) {
int32_t x18828 = x18810 * x18817;
int32_t x18829 = x18827 + x18828;
int32_t x18823 = x18804 * x18817;
for(int x18818=0; x18818 < x18804; x18818++) {
int32_t x18830 = x18811 * x18818;
int32_t x18831 = x18829 + x18830;
float x18833 = x18710[x18831];
float x18834 = x18753[x18832];
int32_t x18820 = x18818 + x18819;
int32_t x18822 = x18820 + x18821;
int32_t x18824 = x18822 + x18823;
float x18835 = x18833 / x18834;
x18808[x18824] = x18835;

}

}

}

}
int32_t x18845 = 0;
int32_t x18846 = 1;
x18846 *= 1;
x18845 += 1;
x18846 *= 1;
x18846 *= 1;
int32_t x18851 = x18845;
bool x18852 = x18851 >= 2;
if (x18852) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18857 = x18851 == 0;
if (x18857) {
int32_t x18858 = x18846;
bool x18859 = x18858 == 256;
if (x18859) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18866 = x18846;
bool x18868 = x18802 == 1;
int32_t x18867 = 256 / x18866;
bool x18869 = x18867 == 1;
bool x18873;
if (x454) {
bool x18870 = x18868 || x18869;
bool x18871 = x18802 == x18867;
bool x18872 = x18870 || x18871;
x18873 = x18872;
} else {
x18873 = false;
}
bool x18877;
if (x18873) {
x18877 = x18876;
} else {
x18877 = false;
}
bool x18878;
if (x18877) {
x18878 = x18876;
} else {
x18878 = false;
}
if (x18878) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18802,x18804,x18804,1,x18867,1,1);
assert(false && "");
}
bool x18884 = x18802 <= x18867;
int32_t x18885;
if (x18884) {
x18885 = x18867;
} else {
x18885 = x18802;
}
int32_t x18889 = x18885 * x18888;
int32_t x18890 = 64 * x18889;
float* x18891 = (float*)myMalloc(x18890 * sizeof(float));;
int32_t x18892;
if (x18868) {
x18892 = 0;
} else {
x18892 = x18805;
}
int32_t x18895;
if (x18869) {
x18895 = 0;
} else {
x18895 = 1;
}
for(int x18896=0; x18896 < 64; x18896++) {
int32_t x18908 = x18806 * x18896;
int32_t x18902 = x18889 * x18896;
for(int x18898=0; x18898 < x18885; x18898++) {
int32_t x18909 = x18892 * x18898;
int32_t x18910 = x18908 + x18909;
int32_t x18915 = x18895 * x18898;
int32_t x18904 = x18888 * x18898;
for(int x18900=0; x18900 < x18887; x18900++) {
int32_t x18911 = x18893 * x18900;
int32_t x18912 = x18910 + x18911;
int32_t x18906 = x18887 * x18900;
for(int x18901=0; x18901 < x18887; x18901++) {
int32_t x18913 = x18894 * x18901;
int32_t x18914 = x18912 + x18913;
float x18916 = x18808[x18914];
float x18917 = x141[x18915];
int32_t x18903 = x18901 + x18902;
int32_t x18905 = x18903 + x18904;
int32_t x18907 = x18905 + x18906;
float x18918 = x18916 * x18917;
x18891[x18907] = x18918;

}

}

}

}
int32_t x18928 = 0;
int32_t x18929 = 1;
x18929 *= 1;
x18928 += 1;
x18929 *= 1;
x18929 *= 1;
int32_t x18934 = x18928;
bool x18935 = x18934 >= 2;
if (x18935) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18940 = x18934 == 0;
if (x18940) {
int32_t x18941 = x18929;
bool x18942 = x18941 == 256;
if (x18942) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18949 = x18929;
bool x18951 = x18885 == 1;
int32_t x18950 = 256 / x18949;
bool x18952 = x18950 == 1;
bool x18956;
if (x454) {
bool x18953 = x18951 || x18952;
bool x18954 = x18885 == x18950;
bool x18955 = x18953 || x18954;
x18956 = x18955;
} else {
x18956 = false;
}
bool x18960;
if (x18956) {
x18960 = x18959;
} else {
x18960 = false;
}
bool x18961;
if (x18960) {
x18961 = x18959;
} else {
x18961 = false;
}
if (x18961) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18885,x18887,x18887,1,x18950,1,1);
assert(false && "");
}
bool x18967 = x18885 <= x18950;
int32_t x18968;
if (x18967) {
x18968 = x18950;
} else {
x18968 = x18885;
}
int32_t x18972 = x18968 * x18971;
int32_t x18973 = 64 * x18972;
float* x18974 = (float*)myMalloc(x18973 * sizeof(float));;
int32_t x18975;
if (x18951) {
x18975 = 0;
} else {
x18975 = x18888;
}
int32_t x18978;
if (x18952) {
x18978 = 0;
} else {
x18978 = 1;
}
for(int x18979=0; x18979 < 64; x18979++) {
int32_t x18991 = x18889 * x18979;
int32_t x18985 = x18972 * x18979;
for(int x18981=0; x18981 < x18968; x18981++) {
int32_t x18992 = x18975 * x18981;
int32_t x18993 = x18991 + x18992;
int32_t x18998 = x18978 * x18981;
int32_t x18987 = x18971 * x18981;
for(int x18983=0; x18983 < x18970; x18983++) {
int32_t x18994 = x18976 * x18983;
int32_t x18995 = x18993 + x18994;
int32_t x18989 = x18970 * x18983;
for(int x18984=0; x18984 < x18970; x18984++) {
int32_t x18996 = x18977 * x18984;
int32_t x18997 = x18995 + x18996;
float x18999 = x18891[x18997];
float x19000 = x189[x18998];
int32_t x18986 = x18984 + x18985;
int32_t x18988 = x18986 + x18987;
int32_t x18990 = x18988 + x18989;
float x19001 = x18999 + x19000;
x18974[x18990] = x19001;

}

}

}

}
float* x19011 = (float*)myMalloc(x18973 * sizeof(float));;
for(int x19013=0; x19013 < x18973; x19013++) {
float x19014 = x18974[x19013];
bool x19015 = x19014 < 0.0f;
if (x19015) {
x19011[x19013] = 0.0f;
} else {
float x19018 = x18974[x19013];
x19011[x19013] = x19018;
}

}
float* x19032 = (float*)myMalloc(x19031 * sizeof(float));;
int32_t x19035 = 64 * x18968;
int32_t x19036 = x19035 * x19027;
float* x19037 = (float*)myMalloc(x19036 * sizeof(float));;
int32_t x19033 = x18968 * x19027;
for(int x19038=0; x19038 < 64; x19038++) {
int32_t x19039 = x19038 * x18972;
float* x19040 = x19011+x19039;
int32_t x19041 = x19038 * x19028;
float* x19042 = x19032+x19041;
int32_t x19043 = x19038 * x19033;
float* x19044 = x19037+x19043;
for(int x19045=0; x19045 < x18968; x19045++) {
int32_t x19046 = x19045 / 1;
int32_t x19050 = x19046 * x19026;
int32_t x19051 = x19050 * x19026;
int32_t x19047 = x19045 % 1;
int32_t x19048 = x19047 / 1;
int32_t x19052 = x19048 * x19026;
int32_t x19053 = x19052 * x19026;
int32_t x19054 = x19051 + x19053;
int32_t x19049 = x19047 % 1;
int32_t x19055 = x19049 * x19026;
int32_t x19056 = x19055 * x19026;
int32_t x19057 = x19054 + x19056;
float* x19058 = x19044+x19057;
int32_t x19059 = x19046 * x18970;
int32_t x19060 = x19059 * x18970;
float* x19061 = x19040+x19060;
for(int x19063=0; x19063 < x19026; x19063++) {
int32_t x19065 = x19063 * x19026;
float* x19066 = x19058+x19065;
int32_t x19064 = x19063 + x19048;
int32_t x19067 = x19064 * x18970;
int32_t x19068 = x19067 + x19049;
float* x19069 = x19061+x19068;
memcpy(x19066, x19069, 4 * x19026);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x19027,x18968,1,x97,x18968,x19044,x19027,1,x19042,x19027);

}
int32_t x19078 = 0;
int32_t x19079 = 1;
x19079 *= 1;
x19078 += 1;
x19079 *= 1;
x19079 *= 1;
int32_t x19084 = x19078;
bool x19085 = x19084 >= 2;
if (x19085) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19090 = x19084 == 0;
if (x19090) {
int32_t x19091 = x19079;
bool x19092 = x19091 == 1024;
if (x19092) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19099 = x19079;
int32_t x19100 = 1024 / x19099;
bool x19101 = x19100 == 1;
bool x19104;
if (x454) {
bool x19102 = 1024 == x19100;
bool x19103 = x19101 || x19102;
x19104 = x19103;
} else {
x19104 = false;
}
bool x19108;
if (x19104) {
x19108 = x19107;
} else {
x19108 = false;
}
bool x19109;
if (x19108) {
x19109 = x19107;
} else {
x19109 = false;
}
if (x19109) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x19026,x19026,1,x19100,1,1);
assert(false && "");
}
bool x19115 = 1024 <= x19100;
int32_t x19116;
if (x19115) {
x19116 = x19100;
} else {
x19116 = 1024;
}
int32_t x19120 = x19116 * x19119;
int32_t x19121 = 64 * x19120;
float* x19122 = (float*)myMalloc(x19121 * sizeof(float));;
int32_t x19125;
if (x19101) {
x19125 = 0;
} else {
x19125 = 1;
}
for(int x19126=0; x19126 < 64; x19126++) {
int32_t x19138 = x19028 * x19126;
int32_t x19132 = x19120 * x19126;
for(int x19128=0; x19128 < x19116; x19128++) {
int32_t x19139 = x19027 * x19128;
int32_t x19140 = x19138 + x19139;
int32_t x19145 = x19125 * x19128;
int32_t x19134 = x19119 * x19128;
for(int x19130=0; x19130 < x19118; x19130++) {
int32_t x19141 = x19123 * x19130;
int32_t x19142 = x19140 + x19141;
int32_t x19136 = x19118 * x19130;
for(int x19131=0; x19131 < x19118; x19131++) {
int32_t x19143 = x19124 * x19131;
int32_t x19144 = x19142 + x19143;
float x19146 = x19032[x19144];
float x19147 = x122[x19145];
int32_t x19133 = x19131 + x19132;
int32_t x19135 = x19133 + x19134;
int32_t x19137 = x19135 + x19136;
float x19148 = x19146 - x19147;
x19122[x19137] = x19148;

}

}

}

}
float* x19158 = (float*)myMalloc(1024 * sizeof(float));;
for(int x19159=0; x19159 < 1024; x19159++) {
float x19160 = x183[x19159];
float x19161 = x19160 + 1.0E-5f;
x19158[x19159] = x19161;

}
float* x19165 = (float*)myMalloc(1024 * sizeof(float));;
for(int x19166=0; x19166 < 1024; x19166++) {
float x19167 = x19158[x19166];
double x19168 = (double)x19167;
double x19169 = sqrt(x19168);
float x19170 = (float)x19169;
x19165[x19166] = x19170;

}
int32_t x19174 = 0;
int32_t x19175 = 1;
x19175 *= 1;
x19174 += 1;
x19175 *= 1;
x19175 *= 1;
int32_t x19180 = x19174;
bool x19181 = x19180 >= 2;
if (x19181) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19186 = x19180 == 0;
if (x19186) {
int32_t x19187 = x19175;
bool x19188 = x19187 == 1024;
if (x19188) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19195 = x19175;
bool x19197 = x19116 == 1;
int32_t x19196 = 1024 / x19195;
bool x19198 = x19196 == 1;
bool x19202;
if (x454) {
bool x19199 = x19197 || x19198;
bool x19200 = x19116 == x19196;
bool x19201 = x19199 || x19200;
x19202 = x19201;
} else {
x19202 = false;
}
bool x19206;
if (x19202) {
x19206 = x19205;
} else {
x19206 = false;
}
bool x19207;
if (x19206) {
x19207 = x19205;
} else {
x19207 = false;
}
if (x19207) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19116,x19118,x19118,1,x19196,1,1);
assert(false && "");
}
bool x19213 = x19116 <= x19196;
int32_t x19214;
if (x19213) {
x19214 = x19196;
} else {
x19214 = x19116;
}
int32_t x19218 = x19214 * x19217;
int32_t x19219 = 64 * x19218;
float* x19220 = (float*)myMalloc(x19219 * sizeof(float));;
int32_t x19221;
if (x19197) {
x19221 = 0;
} else {
x19221 = x19119;
}
int32_t x19224;
if (x19198) {
x19224 = 0;
} else {
x19224 = 1;
}
for(int x19225=0; x19225 < 64; x19225++) {
int32_t x19237 = x19120 * x19225;
int32_t x19231 = x19218 * x19225;
for(int x19227=0; x19227 < x19214; x19227++) {
int32_t x19238 = x19221 * x19227;
int32_t x19239 = x19237 + x19238;
int32_t x19244 = x19224 * x19227;
int32_t x19233 = x19217 * x19227;
for(int x19229=0; x19229 < x19216; x19229++) {
int32_t x19240 = x19222 * x19229;
int32_t x19241 = x19239 + x19240;
int32_t x19235 = x19216 * x19229;
for(int x19230=0; x19230 < x19216; x19230++) {
int32_t x19242 = x19223 * x19230;
int32_t x19243 = x19241 + x19242;
float x19245 = x19122[x19243];
float x19246 = x19165[x19244];
int32_t x19232 = x19230 + x19231;
int32_t x19234 = x19232 + x19233;
int32_t x19236 = x19234 + x19235;
float x19247 = x19245 / x19246;
x19220[x19236] = x19247;

}

}

}

}
int32_t x19257 = 0;
int32_t x19258 = 1;
x19258 *= 1;
x19257 += 1;
x19258 *= 1;
x19258 *= 1;
int32_t x19263 = x19257;
bool x19264 = x19263 >= 2;
if (x19264) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19269 = x19263 == 0;
if (x19269) {
int32_t x19270 = x19258;
bool x19271 = x19270 == 1024;
if (x19271) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19278 = x19258;
bool x19280 = x19214 == 1;
int32_t x19279 = 1024 / x19278;
bool x19281 = x19279 == 1;
bool x19285;
if (x454) {
bool x19282 = x19280 || x19281;
bool x19283 = x19214 == x19279;
bool x19284 = x19282 || x19283;
x19285 = x19284;
} else {
x19285 = false;
}
bool x19289;
if (x19285) {
x19289 = x19288;
} else {
x19289 = false;
}
bool x19290;
if (x19289) {
x19290 = x19288;
} else {
x19290 = false;
}
if (x19290) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19214,x19216,x19216,1,x19279,1,1);
assert(false && "");
}
bool x19296 = x19214 <= x19279;
int32_t x19297;
if (x19296) {
x19297 = x19279;
} else {
x19297 = x19214;
}
int32_t x19301 = x19297 * x19300;
int32_t x19302 = 64 * x19301;
float* x19303 = (float*)myMalloc(x19302 * sizeof(float));;
int32_t x19304;
if (x19280) {
x19304 = 0;
} else {
x19304 = x19217;
}
int32_t x19307;
if (x19281) {
x19307 = 0;
} else {
x19307 = 1;
}
for(int x19308=0; x19308 < 64; x19308++) {
int32_t x19320 = x19218 * x19308;
int32_t x19314 = x19301 * x19308;
for(int x19310=0; x19310 < x19297; x19310++) {
int32_t x19321 = x19304 * x19310;
int32_t x19322 = x19320 + x19321;
int32_t x19327 = x19307 * x19310;
int32_t x19316 = x19300 * x19310;
for(int x19312=0; x19312 < x19299; x19312++) {
int32_t x19323 = x19305 * x19312;
int32_t x19324 = x19322 + x19323;
int32_t x19318 = x19299 * x19312;
for(int x19313=0; x19313 < x19299; x19313++) {
int32_t x19325 = x19306 * x19313;
int32_t x19326 = x19324 + x19325;
float x19328 = x19220[x19326];
float x19329 = x248[x19327];
int32_t x19315 = x19313 + x19314;
int32_t x19317 = x19315 + x19316;
int32_t x19319 = x19317 + x19318;
float x19330 = x19328 * x19329;
x19303[x19319] = x19330;

}

}

}

}
int32_t x19340 = 0;
int32_t x19341 = 1;
x19341 *= 1;
x19340 += 1;
x19341 *= 1;
x19341 *= 1;
int32_t x19346 = x19340;
bool x19347 = x19346 >= 2;
if (x19347) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19352 = x19346 == 0;
if (x19352) {
int32_t x19353 = x19341;
bool x19354 = x19353 == 1024;
if (x19354) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19361 = x19341;
bool x19363 = x19297 == 1;
int32_t x19362 = 1024 / x19361;
bool x19364 = x19362 == 1;
bool x19368;
if (x454) {
bool x19365 = x19363 || x19364;
bool x19366 = x19297 == x19362;
bool x19367 = x19365 || x19366;
x19368 = x19367;
} else {
x19368 = false;
}
bool x19372;
if (x19368) {
x19372 = x19371;
} else {
x19372 = false;
}
bool x19373;
if (x19372) {
x19373 = x19371;
} else {
x19373 = false;
}
if (x19373) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19297,x19299,x19299,1,x19362,1,1);
assert(false && "");
}
bool x19379 = x19297 <= x19362;
int32_t x19380;
if (x19379) {
x19380 = x19362;
} else {
x19380 = x19297;
}
int32_t x19384 = x19380 * x19383;
int32_t x19385 = 64 * x19384;
float* x19386 = (float*)myMalloc(x19385 * sizeof(float));;
int32_t x19387;
if (x19363) {
x19387 = 0;
} else {
x19387 = x19300;
}
int32_t x19390;
if (x19364) {
x19390 = 0;
} else {
x19390 = 1;
}
for(int x19391=0; x19391 < 64; x19391++) {
int32_t x19403 = x19301 * x19391;
int32_t x19397 = x19384 * x19391;
for(int x19393=0; x19393 < x19380; x19393++) {
int32_t x19404 = x19387 * x19393;
int32_t x19405 = x19403 + x19404;
int32_t x19410 = x19390 * x19393;
int32_t x19399 = x19383 * x19393;
for(int x19395=0; x19395 < x19382; x19395++) {
int32_t x19406 = x19388 * x19395;
int32_t x19407 = x19405 + x19406;
int32_t x19401 = x19382 * x19395;
for(int x19396=0; x19396 < x19382; x19396++) {
int32_t x19408 = x19389 * x19396;
int32_t x19409 = x19407 + x19408;
float x19411 = x19303[x19409];
float x19412 = x93[x19410];
int32_t x19398 = x19396 + x19397;
int32_t x19400 = x19398 + x19399;
int32_t x19402 = x19400 + x19401;
float x19413 = x19411 + x19412;
x19386[x19402] = x19413;

}

}

}

}
bool x19423 = x19380 == 1;
bool x19424 = x19423 || x18081;
bool x19425 = x19380 == x18038;
bool x19426 = x19424 || x19425;
bool x19431;
if (x19426) {
x19431 = x19430;
} else {
x19431 = false;
}
bool x19432;
if (x19431) {
x19432 = x19430;
} else {
x19432 = false;
}
if (x19432) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19380,x19382,x19382,64,x18038,x18040,x18040);
assert(false && "");
}
bool x19438 = x19380 <= x18038;
int32_t x19439;
if (x19438) {
x19439 = x18038;
} else {
x19439 = x19380;
}
int32_t x19445;
if (x19423) {
x19445 = 0;
} else {
x19445 = x19383;
}
for(int x19448=0; x19448 < 64; x19448++) {
int32_t x19454 = x19384 * x19448;
int32_t x19461 = x18042 * x19448;
for(int x19450=0; x19450 < x19439; x19450++) {
int32_t x19455 = x19445 * x19450;
int32_t x19456 = x19454 + x19455;
int32_t x19462 = x18103 * x19450;
int32_t x19463 = x19461 + x19462;
for(int x19452=0; x19452 < x19441; x19452++) {
int32_t x19457 = x19446 * x19452;
int32_t x19458 = x19456 + x19457;
int32_t x19464 = x18104 * x19452;
int32_t x19465 = x19463 + x19464;
for(int x19453=0; x19453 < x19441; x19453++) {
int32_t x19459 = x19447 * x19453;
int32_t x19460 = x19458 + x19459;
float x19468 = x19386[x19460];
int32_t x19466 = x18105 * x19453;
int32_t x19467 = x19465 + x19466;
float x19469 = x18138[x19467];
float x19470 = x19468 + x19469;
x19386[x19460] = x19470;

}

}

}

}
float* x19480 = (float*)myMalloc(x19385 * sizeof(float));;
for(int x19482=0; x19482 < x19385; x19482++) {
float x19483 = x19386[x19482];
bool x19484 = x19483 < 0.0f;
if (x19484) {
x19480[x19482] = 0.0f;
} else {
float x19487 = x19386[x19482];
x19480[x19482] = x19487;
}

}
float* x19501 = (float*)myMalloc(x19500 * sizeof(float));;
int32_t x19504 = 64 * x19380;
int32_t x19505 = x19504 * x19496;
float* x19506 = (float*)myMalloc(x19505 * sizeof(float));;
int32_t x19502 = x19380 * x19496;
for(int x19507=0; x19507 < 64; x19507++) {
int32_t x19508 = x19507 * x19384;
float* x19509 = x19480+x19508;
int32_t x19510 = x19507 * x19497;
float* x19511 = x19501+x19510;
int32_t x19512 = x19507 * x19502;
float* x19513 = x19506+x19512;
for(int x19514=0; x19514 < x19380; x19514++) {
int32_t x19515 = x19514 / 1;
int32_t x19519 = x19515 * x19495;
int32_t x19520 = x19519 * x19495;
int32_t x19516 = x19514 % 1;
int32_t x19517 = x19516 / 1;
int32_t x19521 = x19517 * x19495;
int32_t x19522 = x19521 * x19495;
int32_t x19523 = x19520 + x19522;
int32_t x19518 = x19516 % 1;
int32_t x19524 = x19518 * x19495;
int32_t x19525 = x19524 * x19495;
int32_t x19526 = x19523 + x19525;
float* x19527 = x19513+x19526;
int32_t x19528 = x19515 * x19382;
int32_t x19529 = x19528 * x19382;
float* x19530 = x19509+x19529;
for(int x19532=0; x19532 < x19495; x19532++) {
int32_t x19534 = x19532 * x19495;
float* x19535 = x19527+x19534;
int32_t x19533 = x19532 + x19517;
int32_t x19536 = x19533 * x19382;
int32_t x19537 = x19536 + x19518;
float* x19538 = x19530+x19537;
memcpy(x19535, x19538, 4 * x19495);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x19496,x19380,1,x139,x19380,x19513,x19496,1,x19511,x19496);

}
int32_t x19547 = 0;
int32_t x19548 = 1;
x19548 *= 1;
x19547 += 1;
x19548 *= 1;
x19548 *= 1;
int32_t x19553 = x19547;
bool x19554 = x19553 >= 2;
if (x19554) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19559 = x19553 == 0;
if (x19559) {
int32_t x19560 = x19548;
bool x19561 = x19560 == 512;
if (x19561) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19568 = x19548;
int32_t x19569 = 512 / x19568;
bool x19570 = x19569 == 1;
bool x19573;
if (x454) {
bool x19571 = 512 == x19569;
bool x19572 = x19570 || x19571;
x19573 = x19572;
} else {
x19573 = false;
}
bool x19577;
if (x19573) {
x19577 = x19576;
} else {
x19577 = false;
}
bool x19578;
if (x19577) {
x19578 = x19576;
} else {
x19578 = false;
}
if (x19578) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x19495,x19495,1,x19569,1,1);
assert(false && "");
}
bool x19584 = 512 <= x19569;
int32_t x19585;
if (x19584) {
x19585 = x19569;
} else {
x19585 = 512;
}
int32_t x19589 = x19585 * x19588;
int32_t x19590 = 64 * x19589;
float* x19591 = (float*)myMalloc(x19590 * sizeof(float));;
int32_t x19594;
if (x19570) {
x19594 = 0;
} else {
x19594 = 1;
}
for(int x19595=0; x19595 < 64; x19595++) {
int32_t x19607 = x19497 * x19595;
int32_t x19601 = x19589 * x19595;
for(int x19597=0; x19597 < x19585; x19597++) {
int32_t x19608 = x19496 * x19597;
int32_t x19609 = x19607 + x19608;
int32_t x19614 = x19594 * x19597;
int32_t x19603 = x19588 * x19597;
for(int x19599=0; x19599 < x19587; x19599++) {
int32_t x19610 = x19592 * x19599;
int32_t x19611 = x19609 + x19610;
int32_t x19605 = x19587 * x19599;
for(int x19600=0; x19600 < x19587; x19600++) {
int32_t x19612 = x19593 * x19600;
int32_t x19613 = x19611 + x19612;
float x19615 = x19501[x19613];
float x19616 = x67[x19614];
int32_t x19602 = x19600 + x19601;
int32_t x19604 = x19602 + x19603;
int32_t x19606 = x19604 + x19605;
float x19617 = x19615 - x19616;
x19591[x19606] = x19617;

}

}

}

}
float* x19627 = (float*)myMalloc(512 * sizeof(float));;
for(int x19628=0; x19628 < 512; x19628++) {
float x19629 = x121[x19628];
float x19630 = x19629 + 1.0E-5f;
x19627[x19628] = x19630;

}
float* x19634 = (float*)myMalloc(512 * sizeof(float));;
for(int x19635=0; x19635 < 512; x19635++) {
float x19636 = x19627[x19635];
double x19637 = (double)x19636;
double x19638 = sqrt(x19637);
float x19639 = (float)x19638;
x19634[x19635] = x19639;

}
int32_t x19643 = 0;
int32_t x19644 = 1;
x19644 *= 1;
x19643 += 1;
x19644 *= 1;
x19644 *= 1;
int32_t x19649 = x19643;
bool x19650 = x19649 >= 2;
if (x19650) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19655 = x19649 == 0;
if (x19655) {
int32_t x19656 = x19644;
bool x19657 = x19656 == 512;
if (x19657) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19664 = x19644;
bool x19666 = x19585 == 1;
int32_t x19665 = 512 / x19664;
bool x19667 = x19665 == 1;
bool x19671;
if (x454) {
bool x19668 = x19666 || x19667;
bool x19669 = x19585 == x19665;
bool x19670 = x19668 || x19669;
x19671 = x19670;
} else {
x19671 = false;
}
bool x19675;
if (x19671) {
x19675 = x19674;
} else {
x19675 = false;
}
bool x19676;
if (x19675) {
x19676 = x19674;
} else {
x19676 = false;
}
if (x19676) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19585,x19587,x19587,1,x19665,1,1);
assert(false && "");
}
bool x19682 = x19585 <= x19665;
int32_t x19683;
if (x19682) {
x19683 = x19665;
} else {
x19683 = x19585;
}
int32_t x19687 = x19683 * x19686;
int32_t x19688 = 64 * x19687;
float* x19689 = (float*)myMalloc(x19688 * sizeof(float));;
int32_t x19690;
if (x19666) {
x19690 = 0;
} else {
x19690 = x19588;
}
int32_t x19693;
if (x19667) {
x19693 = 0;
} else {
x19693 = 1;
}
for(int x19694=0; x19694 < 64; x19694++) {
int32_t x19706 = x19589 * x19694;
int32_t x19700 = x19687 * x19694;
for(int x19696=0; x19696 < x19683; x19696++) {
int32_t x19707 = x19690 * x19696;
int32_t x19708 = x19706 + x19707;
int32_t x19713 = x19693 * x19696;
int32_t x19702 = x19686 * x19696;
for(int x19698=0; x19698 < x19685; x19698++) {
int32_t x19709 = x19691 * x19698;
int32_t x19710 = x19708 + x19709;
int32_t x19704 = x19685 * x19698;
for(int x19699=0; x19699 < x19685; x19699++) {
int32_t x19711 = x19692 * x19699;
int32_t x19712 = x19710 + x19711;
float x19714 = x19591[x19712];
float x19715 = x19634[x19713];
int32_t x19701 = x19699 + x19700;
int32_t x19703 = x19701 + x19702;
int32_t x19705 = x19703 + x19704;
float x19716 = x19714 / x19715;
x19689[x19705] = x19716;

}

}

}

}
int32_t x19726 = 0;
int32_t x19727 = 1;
x19727 *= 1;
x19726 += 1;
x19727 *= 1;
x19727 *= 1;
int32_t x19732 = x19726;
bool x19733 = x19732 >= 2;
if (x19733) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19738 = x19732 == 0;
if (x19738) {
int32_t x19739 = x19727;
bool x19740 = x19739 == 512;
if (x19740) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19747 = x19727;
bool x19749 = x19683 == 1;
int32_t x19748 = 512 / x19747;
bool x19750 = x19748 == 1;
bool x19754;
if (x454) {
bool x19751 = x19749 || x19750;
bool x19752 = x19683 == x19748;
bool x19753 = x19751 || x19752;
x19754 = x19753;
} else {
x19754 = false;
}
bool x19758;
if (x19754) {
x19758 = x19757;
} else {
x19758 = false;
}
bool x19759;
if (x19758) {
x19759 = x19757;
} else {
x19759 = false;
}
if (x19759) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19683,x19685,x19685,1,x19748,1,1);
assert(false && "");
}
bool x19765 = x19683 <= x19748;
int32_t x19766;
if (x19765) {
x19766 = x19748;
} else {
x19766 = x19683;
}
int32_t x19770 = x19766 * x19769;
int32_t x19771 = 64 * x19770;
float* x19772 = (float*)myMalloc(x19771 * sizeof(float));;
int32_t x19773;
if (x19749) {
x19773 = 0;
} else {
x19773 = x19686;
}
int32_t x19776;
if (x19750) {
x19776 = 0;
} else {
x19776 = 1;
}
for(int x19777=0; x19777 < 64; x19777++) {
int32_t x19789 = x19687 * x19777;
int32_t x19783 = x19770 * x19777;
for(int x19779=0; x19779 < x19766; x19779++) {
int32_t x19790 = x19773 * x19779;
int32_t x19791 = x19789 + x19790;
int32_t x19796 = x19776 * x19779;
int32_t x19785 = x19769 * x19779;
for(int x19781=0; x19781 < x19768; x19781++) {
int32_t x19792 = x19774 * x19781;
int32_t x19793 = x19791 + x19792;
int32_t x19787 = x19768 * x19781;
for(int x19782=0; x19782 < x19768; x19782++) {
int32_t x19794 = x19775 * x19782;
int32_t x19795 = x19793 + x19794;
float x19797 = x19689[x19795];
float x19798 = x201[x19796];
int32_t x19784 = x19782 + x19783;
int32_t x19786 = x19784 + x19785;
int32_t x19788 = x19786 + x19787;
float x19799 = x19797 * x19798;
x19772[x19788] = x19799;

}

}

}

}
int32_t x19809 = 0;
int32_t x19810 = 1;
x19810 *= 1;
x19809 += 1;
x19810 *= 1;
x19810 *= 1;
int32_t x19815 = x19809;
bool x19816 = x19815 >= 2;
if (x19816) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19821 = x19815 == 0;
if (x19821) {
int32_t x19822 = x19810;
bool x19823 = x19822 == 512;
if (x19823) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19830 = x19810;
bool x19832 = x19766 == 1;
int32_t x19831 = 512 / x19830;
bool x19833 = x19831 == 1;
bool x19837;
if (x454) {
bool x19834 = x19832 || x19833;
bool x19835 = x19766 == x19831;
bool x19836 = x19834 || x19835;
x19837 = x19836;
} else {
x19837 = false;
}
bool x19841;
if (x19837) {
x19841 = x19840;
} else {
x19841 = false;
}
bool x19842;
if (x19841) {
x19842 = x19840;
} else {
x19842 = false;
}
if (x19842) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19766,x19768,x19768,1,x19831,1,1);
assert(false && "");
}
bool x19848 = x19766 <= x19831;
int32_t x19849;
if (x19848) {
x19849 = x19831;
} else {
x19849 = x19766;
}
int32_t x19853 = x19849 * x19852;
int32_t x19854 = 64 * x19853;
float* x19855 = (float*)myMalloc(x19854 * sizeof(float));;
int32_t x19856;
if (x19832) {
x19856 = 0;
} else {
x19856 = x19769;
}
int32_t x19859;
if (x19833) {
x19859 = 0;
} else {
x19859 = 1;
}
for(int x19860=0; x19860 < 64; x19860++) {
int32_t x19872 = x19770 * x19860;
int32_t x19866 = x19853 * x19860;
for(int x19862=0; x19862 < x19849; x19862++) {
int32_t x19873 = x19856 * x19862;
int32_t x19874 = x19872 + x19873;
int32_t x19879 = x19859 * x19862;
int32_t x19868 = x19852 * x19862;
for(int x19864=0; x19864 < x19851; x19864++) {
int32_t x19875 = x19857 * x19864;
int32_t x19876 = x19874 + x19875;
int32_t x19870 = x19851 * x19864;
for(int x19865=0; x19865 < x19851; x19865++) {
int32_t x19877 = x19858 * x19865;
int32_t x19878 = x19876 + x19877;
float x19880 = x19772[x19878];
float x19881 = x224[x19879];
int32_t x19867 = x19865 + x19866;
int32_t x19869 = x19867 + x19868;
int32_t x19871 = x19869 + x19870;
float x19882 = x19880 + x19881;
x19855[x19871] = x19882;

}

}

}

}
float* x19892 = (float*)myMalloc(x19854 * sizeof(float));;
for(int x19894=0; x19894 < x19854; x19894++) {
float x19895 = x19855[x19894];
bool x19896 = x19895 < 0.0f;
if (x19896) {
x19892[x19894] = 0.0f;
} else {
float x19899 = x19855[x19894];
x19892[x19894] = x19899;
}

}
float* x19914 = (float*)myMalloc(x19913 * sizeof(float));;
int32_t x19915 = 9 * x19849;
int32_t x19918 = 64 * x19915;
int32_t x19919 = x19918 * x19909;
float* x19920 = (float*)myMalloc(x19919 * sizeof(float));;
int32_t x19916 = x19915 * x19909;
int32_t x19928 = x19849 * 3;
int32_t x19929 = x19928 * 3;
for(int x19921=0; x19921 < 64; x19921++) {
int32_t x19922 = x19921 * x19853;
float* x19923 = x19892+x19922;
int32_t x19924 = x19921 * x19910;
float* x19925 = x19914+x19924;
int32_t x19926 = x19921 * x19916;
float* x19927 = x19920+x19926;
for(int x19931=0; x19931 < x19929; x19931++) {
int32_t x19932 = x19931 / 9;
int32_t x19936 = x19932 * 3;
int32_t x19937 = x19936 * 3;
int32_t x19938 = x19937 * x19908;
int32_t x19939 = x19938 * x19908;
int32_t x19933 = x19931 % 9;
int32_t x19934 = x19933 / 3;
int32_t x19940 = x19934 * 3;
int32_t x19941 = x19940 * x19908;
int32_t x19942 = x19941 * x19908;
int32_t x19943 = x19939 + x19942;
int32_t x19935 = x19933 % 3;
int32_t x19944 = x19935 * x19908;
int32_t x19945 = x19944 * x19908;
int32_t x19946 = x19943 + x19945;
float* x19947 = x19927+x19946;
int32_t x19948 = x19932 * x19851;
int32_t x19949 = x19948 * x19851;
float* x19950 = x19923+x19949;
for(int x19952=0; x19952 < x19908; x19952++) {
int32_t x19953 = x19952 * 2;
int32_t x19954 = x19953 - 1;
int32_t x19955 = x19954 + x19934;
bool x19956 = x19955 < 0;
bool x19957 = x19955 >= x19851;
bool x19958 = x19956 || x19957;
if (x19958) {
int32_t x19959 = x19952 * x19908;
float* x19960 = x19947+x19959;
memset(x19960, 0, 4 * x19908);;
} else {
int32_t x19959 = x19952 * x19908;
int32_t x19975 = x19955 * x19851;
for(int x19963=0; x19963 < x19908; x19963++) {
int32_t x19964 = x19963 * 2;
int32_t x19965 = x19964 - 1;
int32_t x19966 = x19965 + x19935;
bool x19967 = x19966 < 0;
bool x19968 = x19966 >= x19851;
bool x19969 = x19967 || x19968;
if (x19969) {
int32_t x19970 = x19959 + x19963;
float* x19971 = x19947+x19970;
memset(x19971, 0, 4 * 1);;
} else {
int32_t x19970 = x19959 + x19963;
float* x19974 = x19947+x19970;
int32_t x19976 = x19975 + x19966;
float* x19977 = x19950+x19976;
memcpy(x19974, x19977, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x19909,x19915,1,x34,x19915,x19927,x19909,1,x19925,x19909);

}
int32_t x19992 = 0;
int32_t x19993 = 1;
x19993 *= 1;
x19992 += 1;
x19993 *= 1;
x19993 *= 1;
int32_t x19998 = x19992;
bool x19999 = x19998 >= 2;
if (x19999) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20004 = x19998 == 0;
if (x20004) {
int32_t x20005 = x19993;
bool x20006 = x20005 == 512;
if (x20006) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20013 = x19993;
int32_t x20014 = 512 / x20013;
bool x20015 = x20014 == 1;
bool x20018;
if (x454) {
bool x20016 = 512 == x20014;
bool x20017 = x20015 || x20016;
x20018 = x20017;
} else {
x20018 = false;
}
bool x20022;
if (x20018) {
x20022 = x20021;
} else {
x20022 = false;
}
bool x20023;
if (x20022) {
x20023 = x20021;
} else {
x20023 = false;
}
if (x20023) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x19908,x19908,1,x20014,1,1);
assert(false && "");
}
bool x20029 = 512 <= x20014;
int32_t x20030;
if (x20029) {
x20030 = x20014;
} else {
x20030 = 512;
}
int32_t x20034 = x20030 * x20033;
int32_t x20035 = 64 * x20034;
float* x20036 = (float*)myMalloc(x20035 * sizeof(float));;
int32_t x20039;
if (x20015) {
x20039 = 0;
} else {
x20039 = 1;
}
for(int x20040=0; x20040 < 64; x20040++) {
int32_t x20052 = x19910 * x20040;
int32_t x20046 = x20034 * x20040;
for(int x20042=0; x20042 < x20030; x20042++) {
int32_t x20053 = x19909 * x20042;
int32_t x20054 = x20052 + x20053;
int32_t x20059 = x20039 * x20042;
int32_t x20048 = x20033 * x20042;
for(int x20044=0; x20044 < x20032; x20044++) {
int32_t x20055 = x20037 * x20044;
int32_t x20056 = x20054 + x20055;
int32_t x20050 = x20032 * x20044;
for(int x20045=0; x20045 < x20032; x20045++) {
int32_t x20057 = x20038 * x20045;
int32_t x20058 = x20056 + x20057;
float x20060 = x19914[x20058];
float x20061 = x113[x20059];
int32_t x20047 = x20045 + x20046;
int32_t x20049 = x20047 + x20048;
int32_t x20051 = x20049 + x20050;
float x20062 = x20060 - x20061;
x20036[x20051] = x20062;

}

}

}

}
float* x20072 = (float*)myMalloc(512 * sizeof(float));;
for(int x20073=0; x20073 < 512; x20073++) {
float x20074 = x50[x20073];
float x20075 = x20074 + 1.0E-5f;
x20072[x20073] = x20075;

}
float* x20079 = (float*)myMalloc(512 * sizeof(float));;
for(int x20080=0; x20080 < 512; x20080++) {
float x20081 = x20072[x20080];
double x20082 = (double)x20081;
double x20083 = sqrt(x20082);
float x20084 = (float)x20083;
x20079[x20080] = x20084;

}
int32_t x20088 = 0;
int32_t x20089 = 1;
x20089 *= 1;
x20088 += 1;
x20089 *= 1;
x20089 *= 1;
int32_t x20094 = x20088;
bool x20095 = x20094 >= 2;
if (x20095) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20100 = x20094 == 0;
if (x20100) {
int32_t x20101 = x20089;
bool x20102 = x20101 == 512;
if (x20102) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20109 = x20089;
bool x20111 = x20030 == 1;
int32_t x20110 = 512 / x20109;
bool x20112 = x20110 == 1;
bool x20116;
if (x454) {
bool x20113 = x20111 || x20112;
bool x20114 = x20030 == x20110;
bool x20115 = x20113 || x20114;
x20116 = x20115;
} else {
x20116 = false;
}
bool x20120;
if (x20116) {
x20120 = x20119;
} else {
x20120 = false;
}
bool x20121;
if (x20120) {
x20121 = x20119;
} else {
x20121 = false;
}
if (x20121) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20030,x20032,x20032,1,x20110,1,1);
assert(false && "");
}
bool x20127 = x20030 <= x20110;
int32_t x20128;
if (x20127) {
x20128 = x20110;
} else {
x20128 = x20030;
}
int32_t x20132 = x20128 * x20131;
int32_t x20133 = 64 * x20132;
float* x20134 = (float*)myMalloc(x20133 * sizeof(float));;
int32_t x20135;
if (x20111) {
x20135 = 0;
} else {
x20135 = x20033;
}
int32_t x20138;
if (x20112) {
x20138 = 0;
} else {
x20138 = 1;
}
for(int x20139=0; x20139 < 64; x20139++) {
int32_t x20151 = x20034 * x20139;
int32_t x20145 = x20132 * x20139;
for(int x20141=0; x20141 < x20128; x20141++) {
int32_t x20152 = x20135 * x20141;
int32_t x20153 = x20151 + x20152;
int32_t x20158 = x20138 * x20141;
int32_t x20147 = x20131 * x20141;
for(int x20143=0; x20143 < x20130; x20143++) {
int32_t x20154 = x20136 * x20143;
int32_t x20155 = x20153 + x20154;
int32_t x20149 = x20130 * x20143;
for(int x20144=0; x20144 < x20130; x20144++) {
int32_t x20156 = x20137 * x20144;
int32_t x20157 = x20155 + x20156;
float x20159 = x20036[x20157];
float x20160 = x20079[x20158];
int32_t x20146 = x20144 + x20145;
int32_t x20148 = x20146 + x20147;
int32_t x20150 = x20148 + x20149;
float x20161 = x20159 / x20160;
x20134[x20150] = x20161;

}

}

}

}
int32_t x20171 = 0;
int32_t x20172 = 1;
x20172 *= 1;
x20171 += 1;
x20172 *= 1;
x20172 *= 1;
int32_t x20177 = x20171;
bool x20178 = x20177 >= 2;
if (x20178) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20183 = x20177 == 0;
if (x20183) {
int32_t x20184 = x20172;
bool x20185 = x20184 == 512;
if (x20185) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20192 = x20172;
bool x20194 = x20128 == 1;
int32_t x20193 = 512 / x20192;
bool x20195 = x20193 == 1;
bool x20199;
if (x454) {
bool x20196 = x20194 || x20195;
bool x20197 = x20128 == x20193;
bool x20198 = x20196 || x20197;
x20199 = x20198;
} else {
x20199 = false;
}
bool x20203;
if (x20199) {
x20203 = x20202;
} else {
x20203 = false;
}
bool x20204;
if (x20203) {
x20204 = x20202;
} else {
x20204 = false;
}
if (x20204) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20128,x20130,x20130,1,x20193,1,1);
assert(false && "");
}
bool x20210 = x20128 <= x20193;
int32_t x20211;
if (x20210) {
x20211 = x20193;
} else {
x20211 = x20128;
}
int32_t x20215 = x20211 * x20214;
int32_t x20216 = 64 * x20215;
float* x20217 = (float*)myMalloc(x20216 * sizeof(float));;
int32_t x20218;
if (x20194) {
x20218 = 0;
} else {
x20218 = x20131;
}
int32_t x20221;
if (x20195) {
x20221 = 0;
} else {
x20221 = 1;
}
for(int x20222=0; x20222 < 64; x20222++) {
int32_t x20234 = x20132 * x20222;
int32_t x20228 = x20215 * x20222;
for(int x20224=0; x20224 < x20211; x20224++) {
int32_t x20235 = x20218 * x20224;
int32_t x20236 = x20234 + x20235;
int32_t x20241 = x20221 * x20224;
int32_t x20230 = x20214 * x20224;
for(int x20226=0; x20226 < x20213; x20226++) {
int32_t x20237 = x20219 * x20226;
int32_t x20238 = x20236 + x20237;
int32_t x20232 = x20213 * x20226;
for(int x20227=0; x20227 < x20213; x20227++) {
int32_t x20239 = x20220 * x20227;
int32_t x20240 = x20238 + x20239;
float x20242 = x20134[x20240];
float x20243 = x205[x20241];
int32_t x20229 = x20227 + x20228;
int32_t x20231 = x20229 + x20230;
int32_t x20233 = x20231 + x20232;
float x20244 = x20242 * x20243;
x20217[x20233] = x20244;

}

}

}

}
int32_t x20254 = 0;
int32_t x20255 = 1;
x20255 *= 1;
x20254 += 1;
x20255 *= 1;
x20255 *= 1;
int32_t x20260 = x20254;
bool x20261 = x20260 >= 2;
if (x20261) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20266 = x20260 == 0;
if (x20266) {
int32_t x20267 = x20255;
bool x20268 = x20267 == 512;
if (x20268) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20275 = x20255;
bool x20277 = x20211 == 1;
int32_t x20276 = 512 / x20275;
bool x20278 = x20276 == 1;
bool x20282;
if (x454) {
bool x20279 = x20277 || x20278;
bool x20280 = x20211 == x20276;
bool x20281 = x20279 || x20280;
x20282 = x20281;
} else {
x20282 = false;
}
bool x20286;
if (x20282) {
x20286 = x20285;
} else {
x20286 = false;
}
bool x20287;
if (x20286) {
x20287 = x20285;
} else {
x20287 = false;
}
if (x20287) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20211,x20213,x20213,1,x20276,1,1);
assert(false && "");
}
bool x20293 = x20211 <= x20276;
int32_t x20294;
if (x20293) {
x20294 = x20276;
} else {
x20294 = x20211;
}
int32_t x20298 = x20294 * x20297;
int32_t x20299 = 64 * x20298;
float* x20300 = (float*)myMalloc(x20299 * sizeof(float));;
int32_t x20301;
if (x20277) {
x20301 = 0;
} else {
x20301 = x20214;
}
int32_t x20304;
if (x20278) {
x20304 = 0;
} else {
x20304 = 1;
}
for(int x20305=0; x20305 < 64; x20305++) {
int32_t x20317 = x20215 * x20305;
int32_t x20311 = x20298 * x20305;
for(int x20307=0; x20307 < x20294; x20307++) {
int32_t x20318 = x20301 * x20307;
int32_t x20319 = x20317 + x20318;
int32_t x20324 = x20304 * x20307;
int32_t x20313 = x20297 * x20307;
for(int x20309=0; x20309 < x20296; x20309++) {
int32_t x20320 = x20302 * x20309;
int32_t x20321 = x20319 + x20320;
int32_t x20315 = x20296 * x20309;
for(int x20310=0; x20310 < x20296; x20310++) {
int32_t x20322 = x20303 * x20310;
int32_t x20323 = x20321 + x20322;
float x20325 = x20217[x20323];
float x20326 = x159[x20324];
int32_t x20312 = x20310 + x20311;
int32_t x20314 = x20312 + x20313;
int32_t x20316 = x20314 + x20315;
float x20327 = x20325 + x20326;
x20300[x20316] = x20327;

}

}

}

}
float* x20337 = (float*)myMalloc(x20299 * sizeof(float));;
for(int x20339=0; x20339 < x20299; x20339++) {
float x20340 = x20300[x20339];
bool x20341 = x20340 < 0.0f;
if (x20341) {
x20337[x20339] = 0.0f;
} else {
float x20344 = x20300[x20339];
x20337[x20339] = x20344;
}

}
float* x20358 = (float*)myMalloc(x20357 * sizeof(float));;
int32_t x20361 = 64 * x20294;
int32_t x20362 = x20361 * x20353;
float* x20363 = (float*)myMalloc(x20362 * sizeof(float));;
int32_t x20359 = x20294 * x20353;
for(int x20364=0; x20364 < 64; x20364++) {
int32_t x20365 = x20364 * x20298;
float* x20366 = x20337+x20365;
int32_t x20367 = x20364 * x20354;
float* x20368 = x20358+x20367;
int32_t x20369 = x20364 * x20359;
float* x20370 = x20363+x20369;
for(int x20371=0; x20371 < x20294; x20371++) {
int32_t x20372 = x20371 / 1;
int32_t x20376 = x20372 * x20352;
int32_t x20377 = x20376 * x20352;
int32_t x20373 = x20371 % 1;
int32_t x20374 = x20373 / 1;
int32_t x20378 = x20374 * x20352;
int32_t x20379 = x20378 * x20352;
int32_t x20380 = x20377 + x20379;
int32_t x20375 = x20373 % 1;
int32_t x20381 = x20375 * x20352;
int32_t x20382 = x20381 * x20352;
int32_t x20383 = x20380 + x20382;
float* x20384 = x20370+x20383;
int32_t x20385 = x20372 * x20296;
int32_t x20386 = x20385 * x20296;
float* x20387 = x20366+x20386;
for(int x20389=0; x20389 < x20352; x20389++) {
int32_t x20391 = x20389 * x20352;
float* x20392 = x20384+x20391;
int32_t x20390 = x20389 + x20374;
int32_t x20393 = x20390 * x20296;
int32_t x20394 = x20393 + x20375;
float* x20395 = x20387+x20394;
memcpy(x20392, x20395, 4 * x20352);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x20353,x20294,1,x212,x20294,x20370,x20353,1,x20368,x20353);

}
int32_t x20404 = 0;
int32_t x20405 = 1;
x20405 *= 1;
x20404 += 1;
x20405 *= 1;
x20405 *= 1;
int32_t x20410 = x20404;
bool x20411 = x20410 >= 2;
if (x20411) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20416 = x20410 == 0;
if (x20416) {
int32_t x20417 = x20405;
bool x20418 = x20417 == 2048;
if (x20418) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20425 = x20405;
int32_t x20426 = 2048 / x20425;
bool x20427 = x20426 == 1;
bool x20430;
if (x454) {
bool x20428 = 2048 == x20426;
bool x20429 = x20427 || x20428;
x20430 = x20429;
} else {
x20430 = false;
}
bool x20434;
if (x20430) {
x20434 = x20433;
} else {
x20434 = false;
}
bool x20435;
if (x20434) {
x20435 = x20433;
} else {
x20435 = false;
}
if (x20435) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x20352,x20352,1,x20426,1,1);
assert(false && "");
}
bool x20441 = 2048 <= x20426;
int32_t x20442;
if (x20441) {
x20442 = x20426;
} else {
x20442 = 2048;
}
int32_t x20446 = x20442 * x20445;
int32_t x20447 = 64 * x20446;
float* x20448 = (float*)myMalloc(x20447 * sizeof(float));;
int32_t x20451;
if (x20427) {
x20451 = 0;
} else {
x20451 = 1;
}
for(int x20452=0; x20452 < 64; x20452++) {
int32_t x20464 = x20354 * x20452;
int32_t x20458 = x20446 * x20452;
for(int x20454=0; x20454 < x20442; x20454++) {
int32_t x20465 = x20353 * x20454;
int32_t x20466 = x20464 + x20465;
int32_t x20471 = x20451 * x20454;
int32_t x20460 = x20445 * x20454;
for(int x20456=0; x20456 < x20444; x20456++) {
int32_t x20467 = x20449 * x20456;
int32_t x20468 = x20466 + x20467;
int32_t x20462 = x20444 * x20456;
for(int x20457=0; x20457 < x20444; x20457++) {
int32_t x20469 = x20450 * x20457;
int32_t x20470 = x20468 + x20469;
float x20472 = x20358[x20470];
float x20473 = x115[x20471];
int32_t x20459 = x20457 + x20458;
int32_t x20461 = x20459 + x20460;
int32_t x20463 = x20461 + x20462;
float x20474 = x20472 - x20473;
x20448[x20463] = x20474;

}

}

}

}
float* x20484 = (float*)myMalloc(2048 * sizeof(float));;
for(int x20486=0; x20486 < 2048; x20486++) {
float x20487 = x193[x20486];
float x20488 = x20487 + 1.0E-5f;
x20484[x20486] = x20488;

}
float* x20492 = (float*)myMalloc(2048 * sizeof(float));;
for(int x20493=0; x20493 < 2048; x20493++) {
float x20494 = x20484[x20493];
double x20495 = (double)x20494;
double x20496 = sqrt(x20495);
float x20497 = (float)x20496;
x20492[x20493] = x20497;

}
int32_t x20501 = 0;
int32_t x20502 = 1;
x20502 *= 1;
x20501 += 1;
x20502 *= 1;
x20502 *= 1;
int32_t x20507 = x20501;
bool x20508 = x20507 >= 2;
if (x20508) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20513 = x20507 == 0;
if (x20513) {
int32_t x20514 = x20502;
bool x20515 = x20514 == 2048;
if (x20515) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20522 = x20502;
bool x20524 = x20442 == 1;
int32_t x20523 = 2048 / x20522;
bool x20525 = x20523 == 1;
bool x20529;
if (x454) {
bool x20526 = x20524 || x20525;
bool x20527 = x20442 == x20523;
bool x20528 = x20526 || x20527;
x20529 = x20528;
} else {
x20529 = false;
}
bool x20533;
if (x20529) {
x20533 = x20532;
} else {
x20533 = false;
}
bool x20534;
if (x20533) {
x20534 = x20532;
} else {
x20534 = false;
}
if (x20534) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20442,x20444,x20444,1,x20523,1,1);
assert(false && "");
}
bool x20540 = x20442 <= x20523;
int32_t x20541;
if (x20540) {
x20541 = x20523;
} else {
x20541 = x20442;
}
int32_t x20545 = x20541 * x20544;
int32_t x20546 = 64 * x20545;
float* x20547 = (float*)myMalloc(x20546 * sizeof(float));;
int32_t x20548;
if (x20524) {
x20548 = 0;
} else {
x20548 = x20445;
}
int32_t x20551;
if (x20525) {
x20551 = 0;
} else {
x20551 = 1;
}
for(int x20552=0; x20552 < 64; x20552++) {
int32_t x20564 = x20446 * x20552;
int32_t x20558 = x20545 * x20552;
for(int x20554=0; x20554 < x20541; x20554++) {
int32_t x20565 = x20548 * x20554;
int32_t x20566 = x20564 + x20565;
int32_t x20571 = x20551 * x20554;
int32_t x20560 = x20544 * x20554;
for(int x20556=0; x20556 < x20543; x20556++) {
int32_t x20567 = x20549 * x20556;
int32_t x20568 = x20566 + x20567;
int32_t x20562 = x20543 * x20556;
for(int x20557=0; x20557 < x20543; x20557++) {
int32_t x20569 = x20550 * x20557;
int32_t x20570 = x20568 + x20569;
float x20572 = x20448[x20570];
float x20573 = x20492[x20571];
int32_t x20559 = x20557 + x20558;
int32_t x20561 = x20559 + x20560;
int32_t x20563 = x20561 + x20562;
float x20574 = x20572 / x20573;
x20547[x20563] = x20574;

}

}

}

}
int32_t x20584 = 0;
int32_t x20585 = 1;
x20585 *= 1;
x20584 += 1;
x20585 *= 1;
x20585 *= 1;
int32_t x20590 = x20584;
bool x20591 = x20590 >= 2;
if (x20591) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20596 = x20590 == 0;
if (x20596) {
int32_t x20597 = x20585;
bool x20598 = x20597 == 2048;
if (x20598) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20605 = x20585;
bool x20607 = x20541 == 1;
int32_t x20606 = 2048 / x20605;
bool x20608 = x20606 == 1;
bool x20612;
if (x454) {
bool x20609 = x20607 || x20608;
bool x20610 = x20541 == x20606;
bool x20611 = x20609 || x20610;
x20612 = x20611;
} else {
x20612 = false;
}
bool x20616;
if (x20612) {
x20616 = x20615;
} else {
x20616 = false;
}
bool x20617;
if (x20616) {
x20617 = x20615;
} else {
x20617 = false;
}
if (x20617) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20541,x20543,x20543,1,x20606,1,1);
assert(false && "");
}
bool x20623 = x20541 <= x20606;
int32_t x20624;
if (x20623) {
x20624 = x20606;
} else {
x20624 = x20541;
}
int32_t x20628 = x20624 * x20627;
int32_t x20629 = 64 * x20628;
float* x20630 = (float*)myMalloc(x20629 * sizeof(float));;
int32_t x20631;
if (x20607) {
x20631 = 0;
} else {
x20631 = x20544;
}
int32_t x20634;
if (x20608) {
x20634 = 0;
} else {
x20634 = 1;
}
for(int x20635=0; x20635 < 64; x20635++) {
int32_t x20647 = x20545 * x20635;
int32_t x20641 = x20628 * x20635;
for(int x20637=0; x20637 < x20624; x20637++) {
int32_t x20648 = x20631 * x20637;
int32_t x20649 = x20647 + x20648;
int32_t x20654 = x20634 * x20637;
int32_t x20643 = x20627 * x20637;
for(int x20639=0; x20639 < x20626; x20639++) {
int32_t x20650 = x20632 * x20639;
int32_t x20651 = x20649 + x20650;
int32_t x20645 = x20626 * x20639;
for(int x20640=0; x20640 < x20626; x20640++) {
int32_t x20652 = x20633 * x20640;
int32_t x20653 = x20651 + x20652;
float x20655 = x20547[x20653];
float x20656 = x239[x20654];
int32_t x20642 = x20640 + x20641;
int32_t x20644 = x20642 + x20643;
int32_t x20646 = x20644 + x20645;
float x20657 = x20655 * x20656;
x20630[x20646] = x20657;

}

}

}

}
int32_t x20667 = 0;
int32_t x20668 = 1;
x20668 *= 1;
x20667 += 1;
x20668 *= 1;
x20668 *= 1;
int32_t x20673 = x20667;
bool x20674 = x20673 >= 2;
if (x20674) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20679 = x20673 == 0;
if (x20679) {
int32_t x20680 = x20668;
bool x20681 = x20680 == 2048;
if (x20681) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20688 = x20668;
bool x20690 = x20624 == 1;
int32_t x20689 = 2048 / x20688;
bool x20691 = x20689 == 1;
bool x20695;
if (x454) {
bool x20692 = x20690 || x20691;
bool x20693 = x20624 == x20689;
bool x20694 = x20692 || x20693;
x20695 = x20694;
} else {
x20695 = false;
}
bool x20699;
if (x20695) {
x20699 = x20698;
} else {
x20699 = false;
}
bool x20700;
if (x20699) {
x20700 = x20698;
} else {
x20700 = false;
}
if (x20700) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20624,x20626,x20626,1,x20689,1,1);
assert(false && "");
}
bool x20706 = x20624 <= x20689;
int32_t x20707;
if (x20706) {
x20707 = x20689;
} else {
x20707 = x20624;
}
int32_t x20711 = x20707 * x20710;
int32_t x20712 = 64 * x20711;
float* x20713 = (float*)myMalloc(x20712 * sizeof(float));;
int32_t x20714;
if (x20690) {
x20714 = 0;
} else {
x20714 = x20627;
}
int32_t x20717;
if (x20691) {
x20717 = 0;
} else {
x20717 = 1;
}
for(int x20718=0; x20718 < 64; x20718++) {
int32_t x20730 = x20628 * x20718;
int32_t x20724 = x20711 * x20718;
for(int x20720=0; x20720 < x20707; x20720++) {
int32_t x20731 = x20714 * x20720;
int32_t x20732 = x20730 + x20731;
int32_t x20737 = x20717 * x20720;
int32_t x20726 = x20710 * x20720;
for(int x20722=0; x20722 < x20709; x20722++) {
int32_t x20733 = x20715 * x20722;
int32_t x20734 = x20732 + x20733;
int32_t x20728 = x20709 * x20722;
for(int x20723=0; x20723 < x20709; x20723++) {
int32_t x20735 = x20716 * x20723;
int32_t x20736 = x20734 + x20735;
float x20738 = x20630[x20736];
float x20739 = x62[x20737];
int32_t x20725 = x20723 + x20724;
int32_t x20727 = x20725 + x20726;
int32_t x20729 = x20727 + x20728;
float x20740 = x20738 + x20739;
x20713[x20729] = x20740;

}

}

}

}
float* x20757 = (float*)myMalloc(x20756 * sizeof(float));;
int32_t x20760 = x19504 * x20752;
float* x20761 = (float*)myMalloc(x20760 * sizeof(float));;
int32_t x20758 = x19380 * x20752;
for(int x20762=0; x20762 < 64; x20762++) {
int32_t x20763 = x20762 * x19384;
float* x20764 = x19480+x20763;
int32_t x20765 = x20762 * x20753;
float* x20766 = x20757+x20765;
int32_t x20767 = x20762 * x20758;
float* x20768 = x20761+x20767;
for(int x20769=0; x20769 < x19380; x20769++) {
int32_t x20770 = x20769 / 1;
int32_t x20774 = x20770 * x20751;
int32_t x20775 = x20774 * x20751;
int32_t x20771 = x20769 % 1;
int32_t x20772 = x20771 / 1;
int32_t x20776 = x20772 * x20751;
int32_t x20777 = x20776 * x20751;
int32_t x20778 = x20775 + x20777;
int32_t x20773 = x20771 % 1;
int32_t x20779 = x20773 * x20751;
int32_t x20780 = x20779 * x20751;
int32_t x20781 = x20778 + x20780;
float* x20782 = x20768+x20781;
int32_t x20783 = x20770 * x19382;
int32_t x20784 = x20783 * x19382;
float* x20785 = x20764+x20784;
for(int x20787=0; x20787 < x20751; x20787++) {
int32_t x20791 = x20787 * x20751;
int32_t x20788 = x20787 * 2;
int32_t x20789 = x20788 + x20772;
int32_t x20794 = x20789 * x19382;
int32_t x20795 = x20794 + x20773;
for(int x20790=0; x20790 < x20751; x20790++) {
int32_t x20792 = x20791 + x20790;
float* x20793 = x20782+x20792;
int32_t x20796 = x20790 * 2;
int32_t x20797 = x20795 + x20796;
float* x20798 = x20785+x20797;
memcpy(x20793, x20798, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x20752,x19380,1,x214,x19380,x20768,x20752,1,x20766,x20752);

}
int32_t x20809 = 0;
int32_t x20810 = 1;
x20810 *= 1;
x20809 += 1;
x20810 *= 1;
x20810 *= 1;
int32_t x20815 = x20809;
bool x20816 = x20815 >= 2;
if (x20816) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20821 = x20815 == 0;
if (x20821) {
int32_t x20822 = x20810;
bool x20823 = x20822 == 2048;
if (x20823) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20830 = x20810;
int32_t x20831 = 2048 / x20830;
bool x20832 = x20831 == 1;
bool x20835;
if (x454) {
bool x20833 = 2048 == x20831;
bool x20834 = x20832 || x20833;
x20835 = x20834;
} else {
x20835 = false;
}
bool x20839;
if (x20835) {
x20839 = x20838;
} else {
x20839 = false;
}
bool x20840;
if (x20839) {
x20840 = x20838;
} else {
x20840 = false;
}
if (x20840) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x20751,x20751,1,x20831,1,1);
assert(false && "");
}
bool x20846 = 2048 <= x20831;
int32_t x20847;
if (x20846) {
x20847 = x20831;
} else {
x20847 = 2048;
}
int32_t x20851 = x20847 * x20850;
int32_t x20852 = 64 * x20851;
float* x20853 = (float*)myMalloc(x20852 * sizeof(float));;
int32_t x20856;
if (x20832) {
x20856 = 0;
} else {
x20856 = 1;
}
for(int x20857=0; x20857 < 64; x20857++) {
int32_t x20869 = x20753 * x20857;
int32_t x20863 = x20851 * x20857;
for(int x20859=0; x20859 < x20847; x20859++) {
int32_t x20870 = x20752 * x20859;
int32_t x20871 = x20869 + x20870;
int32_t x20876 = x20856 * x20859;
int32_t x20865 = x20850 * x20859;
for(int x20861=0; x20861 < x20849; x20861++) {
int32_t x20872 = x20854 * x20861;
int32_t x20873 = x20871 + x20872;
int32_t x20867 = x20849 * x20861;
for(int x20862=0; x20862 < x20849; x20862++) {
int32_t x20874 = x20855 * x20862;
int32_t x20875 = x20873 + x20874;
float x20877 = x20757[x20875];
float x20878 = x64[x20876];
int32_t x20864 = x20862 + x20863;
int32_t x20866 = x20864 + x20865;
int32_t x20868 = x20866 + x20867;
float x20879 = x20877 - x20878;
x20853[x20868] = x20879;

}

}

}

}
float* x20889 = (float*)myMalloc(2048 * sizeof(float));;
for(int x20890=0; x20890 < 2048; x20890++) {
float x20891 = x125[x20890];
float x20892 = x20891 + 1.0E-5f;
x20889[x20890] = x20892;

}
float* x20896 = (float*)myMalloc(2048 * sizeof(float));;
for(int x20897=0; x20897 < 2048; x20897++) {
float x20898 = x20889[x20897];
double x20899 = (double)x20898;
double x20900 = sqrt(x20899);
float x20901 = (float)x20900;
x20896[x20897] = x20901;

}
int32_t x20905 = 0;
int32_t x20906 = 1;
x20906 *= 1;
x20905 += 1;
x20906 *= 1;
x20906 *= 1;
int32_t x20911 = x20905;
bool x20912 = x20911 >= 2;
if (x20912) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20917 = x20911 == 0;
if (x20917) {
int32_t x20918 = x20906;
bool x20919 = x20918 == 2048;
if (x20919) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20926 = x20906;
bool x20928 = x20847 == 1;
int32_t x20927 = 2048 / x20926;
bool x20929 = x20927 == 1;
bool x20933;
if (x454) {
bool x20930 = x20928 || x20929;
bool x20931 = x20847 == x20927;
bool x20932 = x20930 || x20931;
x20933 = x20932;
} else {
x20933 = false;
}
bool x20937;
if (x20933) {
x20937 = x20936;
} else {
x20937 = false;
}
bool x20938;
if (x20937) {
x20938 = x20936;
} else {
x20938 = false;
}
if (x20938) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20847,x20849,x20849,1,x20927,1,1);
assert(false && "");
}
bool x20944 = x20847 <= x20927;
int32_t x20945;
if (x20944) {
x20945 = x20927;
} else {
x20945 = x20847;
}
int32_t x20949 = x20945 * x20948;
int32_t x20950 = 64 * x20949;
float* x20951 = (float*)myMalloc(x20950 * sizeof(float));;
int32_t x20952;
if (x20928) {
x20952 = 0;
} else {
x20952 = x20850;
}
int32_t x20955;
if (x20929) {
x20955 = 0;
} else {
x20955 = 1;
}
for(int x20956=0; x20956 < 64; x20956++) {
int32_t x20968 = x20851 * x20956;
int32_t x20962 = x20949 * x20956;
for(int x20958=0; x20958 < x20945; x20958++) {
int32_t x20969 = x20952 * x20958;
int32_t x20970 = x20968 + x20969;
int32_t x20975 = x20955 * x20958;
int32_t x20964 = x20948 * x20958;
for(int x20960=0; x20960 < x20947; x20960++) {
int32_t x20971 = x20953 * x20960;
int32_t x20972 = x20970 + x20971;
int32_t x20966 = x20947 * x20960;
for(int x20961=0; x20961 < x20947; x20961++) {
int32_t x20973 = x20954 * x20961;
int32_t x20974 = x20972 + x20973;
float x20976 = x20853[x20974];
float x20977 = x20896[x20975];
int32_t x20963 = x20961 + x20962;
int32_t x20965 = x20963 + x20964;
int32_t x20967 = x20965 + x20966;
float x20978 = x20976 / x20977;
x20951[x20967] = x20978;

}

}

}

}
int32_t x20988 = 0;
int32_t x20989 = 1;
x20989 *= 1;
x20988 += 1;
x20989 *= 1;
x20989 *= 1;
int32_t x20994 = x20988;
bool x20995 = x20994 >= 2;
if (x20995) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21000 = x20994 == 0;
if (x21000) {
int32_t x21001 = x20989;
bool x21002 = x21001 == 2048;
if (x21002) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21009 = x20989;
bool x21011 = x20945 == 1;
int32_t x21010 = 2048 / x21009;
bool x21012 = x21010 == 1;
bool x21016;
if (x454) {
bool x21013 = x21011 || x21012;
bool x21014 = x20945 == x21010;
bool x21015 = x21013 || x21014;
x21016 = x21015;
} else {
x21016 = false;
}
bool x21020;
if (x21016) {
x21020 = x21019;
} else {
x21020 = false;
}
bool x21021;
if (x21020) {
x21021 = x21019;
} else {
x21021 = false;
}
if (x21021) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20945,x20947,x20947,1,x21010,1,1);
assert(false && "");
}
bool x21027 = x20945 <= x21010;
int32_t x21028;
if (x21027) {
x21028 = x21010;
} else {
x21028 = x20945;
}
int32_t x21032 = x21028 * x21031;
int32_t x21033 = 64 * x21032;
float* x21034 = (float*)myMalloc(x21033 * sizeof(float));;
int32_t x21035;
if (x21011) {
x21035 = 0;
} else {
x21035 = x20948;
}
int32_t x21038;
if (x21012) {
x21038 = 0;
} else {
x21038 = 1;
}
for(int x21039=0; x21039 < 64; x21039++) {
int32_t x21051 = x20949 * x21039;
int32_t x21045 = x21032 * x21039;
for(int x21041=0; x21041 < x21028; x21041++) {
int32_t x21052 = x21035 * x21041;
int32_t x21053 = x21051 + x21052;
int32_t x21058 = x21038 * x21041;
int32_t x21047 = x21031 * x21041;
for(int x21043=0; x21043 < x21030; x21043++) {
int32_t x21054 = x21036 * x21043;
int32_t x21055 = x21053 + x21054;
int32_t x21049 = x21030 * x21043;
for(int x21044=0; x21044 < x21030; x21044++) {
int32_t x21056 = x21037 * x21044;
int32_t x21057 = x21055 + x21056;
float x21059 = x20951[x21057];
float x21060 = x173[x21058];
int32_t x21046 = x21044 + x21045;
int32_t x21048 = x21046 + x21047;
int32_t x21050 = x21048 + x21049;
float x21061 = x21059 * x21060;
x21034[x21050] = x21061;

}

}

}

}
int32_t x21071 = 0;
int32_t x21072 = 1;
x21072 *= 1;
x21071 += 1;
x21072 *= 1;
x21072 *= 1;
int32_t x21077 = x21071;
bool x21078 = x21077 >= 2;
if (x21078) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21083 = x21077 == 0;
if (x21083) {
int32_t x21084 = x21072;
bool x21085 = x21084 == 2048;
if (x21085) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21092 = x21072;
bool x21094 = x21028 == 1;
int32_t x21093 = 2048 / x21092;
bool x21095 = x21093 == 1;
bool x21099;
if (x454) {
bool x21096 = x21094 || x21095;
bool x21097 = x21028 == x21093;
bool x21098 = x21096 || x21097;
x21099 = x21098;
} else {
x21099 = false;
}
bool x21103;
if (x21099) {
x21103 = x21102;
} else {
x21103 = false;
}
bool x21104;
if (x21103) {
x21104 = x21102;
} else {
x21104 = false;
}
if (x21104) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21028,x21030,x21030,1,x21093,1,1);
assert(false && "");
}
bool x21110 = x21028 <= x21093;
int32_t x21111;
if (x21110) {
x21111 = x21093;
} else {
x21111 = x21028;
}
int32_t x21115 = x21111 * x21114;
int32_t x21116 = 64 * x21115;
float* x21117 = (float*)myMalloc(x21116 * sizeof(float));;
int32_t x21118;
if (x21094) {
x21118 = 0;
} else {
x21118 = x21031;
}
int32_t x21121;
if (x21095) {
x21121 = 0;
} else {
x21121 = 1;
}
for(int x21122=0; x21122 < 64; x21122++) {
int32_t x21134 = x21032 * x21122;
int32_t x21128 = x21115 * x21122;
for(int x21124=0; x21124 < x21111; x21124++) {
int32_t x21135 = x21118 * x21124;
int32_t x21136 = x21134 + x21135;
int32_t x21141 = x21121 * x21124;
int32_t x21130 = x21114 * x21124;
for(int x21126=0; x21126 < x21113; x21126++) {
int32_t x21137 = x21119 * x21126;
int32_t x21138 = x21136 + x21137;
int32_t x21132 = x21113 * x21126;
for(int x21127=0; x21127 < x21113; x21127++) {
int32_t x21139 = x21120 * x21127;
int32_t x21140 = x21138 + x21139;
float x21142 = x21034[x21140];
float x21143 = x107[x21141];
int32_t x21129 = x21127 + x21128;
int32_t x21131 = x21129 + x21130;
int32_t x21133 = x21131 + x21132;
float x21144 = x21142 + x21143;
x21117[x21133] = x21144;

}

}

}

}
bool x21154 = x20707 == 1;
bool x21155 = x21111 == 1;
bool x21156 = x21154 || x21155;
bool x21157 = x20707 == x21111;
bool x21158 = x21156 || x21157;
bool x21164;
if (x21158) {
x21164 = x21163;
} else {
x21164 = false;
}
bool x21165;
if (x21164) {
x21165 = x21163;
} else {
x21165 = false;
}
if (x21165) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20707,x20709,x20709,64,x21111,x21113,x21113);
assert(false && "");
}
bool x21171 = x20707 <= x21111;
int32_t x21172;
if (x21171) {
x21172 = x21111;
} else {
x21172 = x20707;
}
int32_t x21178;
if (x21154) {
x21178 = 0;
} else {
x21178 = x20710;
}
int32_t x21181;
if (x21155) {
x21181 = 0;
} else {
x21181 = x21114;
}
for(int x21184=0; x21184 < 64; x21184++) {
int32_t x21190 = x20711 * x21184;
int32_t x21197 = x21115 * x21184;
for(int x21186=0; x21186 < x21172; x21186++) {
int32_t x21191 = x21178 * x21186;
int32_t x21192 = x21190 + x21191;
int32_t x21198 = x21181 * x21186;
int32_t x21199 = x21197 + x21198;
for(int x21188=0; x21188 < x21174; x21188++) {
int32_t x21193 = x21179 * x21188;
int32_t x21194 = x21192 + x21193;
int32_t x21200 = x21182 * x21188;
int32_t x21201 = x21199 + x21200;
for(int x21189=0; x21189 < x21174; x21189++) {
int32_t x21195 = x21180 * x21189;
int32_t x21196 = x21194 + x21195;
float x21204 = x20713[x21196];
int32_t x21202 = x21183 * x21189;
int32_t x21203 = x21201 + x21202;
float x21205 = x21117[x21203];
float x21206 = x21204 + x21205;
x20713[x21196] = x21206;

}

}

}

}
float* x21216 = (float*)myMalloc(x20712 * sizeof(float));;
for(int x21218=0; x21218 < x20712; x21218++) {
float x21219 = x20713[x21218];
bool x21220 = x21219 < 0.0f;
if (x21220) {
x21216[x21218] = 0.0f;
} else {
float x21223 = x20713[x21218];
x21216[x21218] = x21223;
}

}
float* x21237 = (float*)myMalloc(x21236 * sizeof(float));;
int32_t x21240 = 64 * x20707;
int32_t x21241 = x21240 * x21232;
float* x21242 = (float*)myMalloc(x21241 * sizeof(float));;
int32_t x21238 = x20707 * x21232;
for(int x21243=0; x21243 < 64; x21243++) {
int32_t x21244 = x21243 * x20711;
float* x21245 = x21216+x21244;
int32_t x21246 = x21243 * x21233;
float* x21247 = x21237+x21246;
int32_t x21248 = x21243 * x21238;
float* x21249 = x21242+x21248;
for(int x21250=0; x21250 < x20707; x21250++) {
int32_t x21251 = x21250 / 1;
int32_t x21255 = x21251 * x21231;
int32_t x21256 = x21255 * x21231;
int32_t x21252 = x21250 % 1;
int32_t x21253 = x21252 / 1;
int32_t x21257 = x21253 * x21231;
int32_t x21258 = x21257 * x21231;
int32_t x21259 = x21256 + x21258;
int32_t x21254 = x21252 % 1;
int32_t x21260 = x21254 * x21231;
int32_t x21261 = x21260 * x21231;
int32_t x21262 = x21259 + x21261;
float* x21263 = x21249+x21262;
int32_t x21264 = x21251 * x20709;
int32_t x21265 = x21264 * x20709;
float* x21266 = x21245+x21265;
for(int x21268=0; x21268 < x21231; x21268++) {
int32_t x21270 = x21268 * x21231;
float* x21271 = x21263+x21270;
int32_t x21269 = x21268 + x21253;
int32_t x21272 = x21269 * x20709;
int32_t x21273 = x21272 + x21254;
float* x21274 = x21266+x21273;
memcpy(x21271, x21274, 4 * x21231);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x21232,x20707,1,x215,x20707,x21249,x21232,1,x21247,x21232);

}
int32_t x21283 = 0;
int32_t x21284 = 1;
x21284 *= 1;
x21283 += 1;
x21284 *= 1;
x21284 *= 1;
int32_t x21289 = x21283;
bool x21290 = x21289 >= 2;
if (x21290) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21295 = x21289 == 0;
if (x21295) {
int32_t x21296 = x21284;
bool x21297 = x21296 == 512;
if (x21297) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21304 = x21284;
int32_t x21305 = 512 / x21304;
bool x21306 = x21305 == 1;
bool x21309;
if (x454) {
bool x21307 = 512 == x21305;
bool x21308 = x21306 || x21307;
x21309 = x21308;
} else {
x21309 = false;
}
bool x21313;
if (x21309) {
x21313 = x21312;
} else {
x21313 = false;
}
bool x21314;
if (x21313) {
x21314 = x21312;
} else {
x21314 = false;
}
if (x21314) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x21231,x21231,1,x21305,1,1);
assert(false && "");
}
bool x21320 = 512 <= x21305;
int32_t x21321;
if (x21320) {
x21321 = x21305;
} else {
x21321 = 512;
}
int32_t x21325 = x21321 * x21324;
int32_t x21326 = 64 * x21325;
float* x21327 = (float*)myMalloc(x21326 * sizeof(float));;
int32_t x21330;
if (x21306) {
x21330 = 0;
} else {
x21330 = 1;
}
for(int x21331=0; x21331 < 64; x21331++) {
int32_t x21343 = x21233 * x21331;
int32_t x21337 = x21325 * x21331;
for(int x21333=0; x21333 < x21321; x21333++) {
int32_t x21344 = x21232 * x21333;
int32_t x21345 = x21343 + x21344;
int32_t x21350 = x21330 * x21333;
int32_t x21339 = x21324 * x21333;
for(int x21335=0; x21335 < x21323; x21335++) {
int32_t x21346 = x21328 * x21335;
int32_t x21347 = x21345 + x21346;
int32_t x21341 = x21323 * x21335;
for(int x21336=0; x21336 < x21323; x21336++) {
int32_t x21348 = x21329 * x21336;
int32_t x21349 = x21347 + x21348;
float x21351 = x21237[x21349];
float x21352 = x154[x21350];
int32_t x21338 = x21336 + x21337;
int32_t x21340 = x21338 + x21339;
int32_t x21342 = x21340 + x21341;
float x21353 = x21351 - x21352;
x21327[x21342] = x21353;

}

}

}

}
float* x21363 = (float*)myMalloc(512 * sizeof(float));;
for(int x21364=0; x21364 < 512; x21364++) {
float x21365 = x65[x21364];
float x21366 = x21365 + 1.0E-5f;
x21363[x21364] = x21366;

}
float* x21370 = (float*)myMalloc(512 * sizeof(float));;
for(int x21371=0; x21371 < 512; x21371++) {
float x21372 = x21363[x21371];
double x21373 = (double)x21372;
double x21374 = sqrt(x21373);
float x21375 = (float)x21374;
x21370[x21371] = x21375;

}
int32_t x21379 = 0;
int32_t x21380 = 1;
x21380 *= 1;
x21379 += 1;
x21380 *= 1;
x21380 *= 1;
int32_t x21385 = x21379;
bool x21386 = x21385 >= 2;
if (x21386) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21391 = x21385 == 0;
if (x21391) {
int32_t x21392 = x21380;
bool x21393 = x21392 == 512;
if (x21393) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21400 = x21380;
bool x21402 = x21321 == 1;
int32_t x21401 = 512 / x21400;
bool x21403 = x21401 == 1;
bool x21407;
if (x454) {
bool x21404 = x21402 || x21403;
bool x21405 = x21321 == x21401;
bool x21406 = x21404 || x21405;
x21407 = x21406;
} else {
x21407 = false;
}
bool x21411;
if (x21407) {
x21411 = x21410;
} else {
x21411 = false;
}
bool x21412;
if (x21411) {
x21412 = x21410;
} else {
x21412 = false;
}
if (x21412) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21321,x21323,x21323,1,x21401,1,1);
assert(false && "");
}
bool x21418 = x21321 <= x21401;
int32_t x21419;
if (x21418) {
x21419 = x21401;
} else {
x21419 = x21321;
}
int32_t x21423 = x21419 * x21422;
int32_t x21424 = 64 * x21423;
float* x21425 = (float*)myMalloc(x21424 * sizeof(float));;
int32_t x21426;
if (x21402) {
x21426 = 0;
} else {
x21426 = x21324;
}
int32_t x21429;
if (x21403) {
x21429 = 0;
} else {
x21429 = 1;
}
for(int x21430=0; x21430 < 64; x21430++) {
int32_t x21442 = x21325 * x21430;
int32_t x21436 = x21423 * x21430;
for(int x21432=0; x21432 < x21419; x21432++) {
int32_t x21443 = x21426 * x21432;
int32_t x21444 = x21442 + x21443;
int32_t x21449 = x21429 * x21432;
int32_t x21438 = x21422 * x21432;
for(int x21434=0; x21434 < x21421; x21434++) {
int32_t x21445 = x21427 * x21434;
int32_t x21446 = x21444 + x21445;
int32_t x21440 = x21421 * x21434;
for(int x21435=0; x21435 < x21421; x21435++) {
int32_t x21447 = x21428 * x21435;
int32_t x21448 = x21446 + x21447;
float x21450 = x21327[x21448];
float x21451 = x21370[x21449];
int32_t x21437 = x21435 + x21436;
int32_t x21439 = x21437 + x21438;
int32_t x21441 = x21439 + x21440;
float x21452 = x21450 / x21451;
x21425[x21441] = x21452;

}

}

}

}
int32_t x21462 = 0;
int32_t x21463 = 1;
x21463 *= 1;
x21462 += 1;
x21463 *= 1;
x21463 *= 1;
int32_t x21468 = x21462;
bool x21469 = x21468 >= 2;
if (x21469) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21474 = x21468 == 0;
if (x21474) {
int32_t x21475 = x21463;
bool x21476 = x21475 == 512;
if (x21476) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21483 = x21463;
bool x21485 = x21419 == 1;
int32_t x21484 = 512 / x21483;
bool x21486 = x21484 == 1;
bool x21490;
if (x454) {
bool x21487 = x21485 || x21486;
bool x21488 = x21419 == x21484;
bool x21489 = x21487 || x21488;
x21490 = x21489;
} else {
x21490 = false;
}
bool x21494;
if (x21490) {
x21494 = x21493;
} else {
x21494 = false;
}
bool x21495;
if (x21494) {
x21495 = x21493;
} else {
x21495 = false;
}
if (x21495) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21419,x21421,x21421,1,x21484,1,1);
assert(false && "");
}
bool x21501 = x21419 <= x21484;
int32_t x21502;
if (x21501) {
x21502 = x21484;
} else {
x21502 = x21419;
}
int32_t x21506 = x21502 * x21505;
int32_t x21507 = 64 * x21506;
float* x21508 = (float*)myMalloc(x21507 * sizeof(float));;
int32_t x21509;
if (x21485) {
x21509 = 0;
} else {
x21509 = x21422;
}
int32_t x21512;
if (x21486) {
x21512 = 0;
} else {
x21512 = 1;
}
for(int x21513=0; x21513 < 64; x21513++) {
int32_t x21525 = x21423 * x21513;
int32_t x21519 = x21506 * x21513;
for(int x21515=0; x21515 < x21502; x21515++) {
int32_t x21526 = x21509 * x21515;
int32_t x21527 = x21525 + x21526;
int32_t x21532 = x21512 * x21515;
int32_t x21521 = x21505 * x21515;
for(int x21517=0; x21517 < x21504; x21517++) {
int32_t x21528 = x21510 * x21517;
int32_t x21529 = x21527 + x21528;
int32_t x21523 = x21504 * x21517;
for(int x21518=0; x21518 < x21504; x21518++) {
int32_t x21530 = x21511 * x21518;
int32_t x21531 = x21529 + x21530;
float x21533 = x21425[x21531];
float x21534 = x46[x21532];
int32_t x21520 = x21518 + x21519;
int32_t x21522 = x21520 + x21521;
int32_t x21524 = x21522 + x21523;
float x21535 = x21533 * x21534;
x21508[x21524] = x21535;

}

}

}

}
int32_t x21545 = 0;
int32_t x21546 = 1;
x21546 *= 1;
x21545 += 1;
x21546 *= 1;
x21546 *= 1;
int32_t x21551 = x21545;
bool x21552 = x21551 >= 2;
if (x21552) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21557 = x21551 == 0;
if (x21557) {
int32_t x21558 = x21546;
bool x21559 = x21558 == 512;
if (x21559) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21566 = x21546;
bool x21568 = x21502 == 1;
int32_t x21567 = 512 / x21566;
bool x21569 = x21567 == 1;
bool x21573;
if (x454) {
bool x21570 = x21568 || x21569;
bool x21571 = x21502 == x21567;
bool x21572 = x21570 || x21571;
x21573 = x21572;
} else {
x21573 = false;
}
bool x21577;
if (x21573) {
x21577 = x21576;
} else {
x21577 = false;
}
bool x21578;
if (x21577) {
x21578 = x21576;
} else {
x21578 = false;
}
if (x21578) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21502,x21504,x21504,1,x21567,1,1);
assert(false && "");
}
bool x21584 = x21502 <= x21567;
int32_t x21585;
if (x21584) {
x21585 = x21567;
} else {
x21585 = x21502;
}
int32_t x21589 = x21585 * x21588;
int32_t x21590 = 64 * x21589;
float* x21591 = (float*)myMalloc(x21590 * sizeof(float));;
int32_t x21592;
if (x21568) {
x21592 = 0;
} else {
x21592 = x21505;
}
int32_t x21595;
if (x21569) {
x21595 = 0;
} else {
x21595 = 1;
}
for(int x21596=0; x21596 < 64; x21596++) {
int32_t x21608 = x21506 * x21596;
int32_t x21602 = x21589 * x21596;
for(int x21598=0; x21598 < x21585; x21598++) {
int32_t x21609 = x21592 * x21598;
int32_t x21610 = x21608 + x21609;
int32_t x21615 = x21595 * x21598;
int32_t x21604 = x21588 * x21598;
for(int x21600=0; x21600 < x21587; x21600++) {
int32_t x21611 = x21593 * x21600;
int32_t x21612 = x21610 + x21611;
int32_t x21606 = x21587 * x21600;
for(int x21601=0; x21601 < x21587; x21601++) {
int32_t x21613 = x21594 * x21601;
int32_t x21614 = x21612 + x21613;
float x21616 = x21508[x21614];
float x21617 = x137[x21615];
int32_t x21603 = x21601 + x21602;
int32_t x21605 = x21603 + x21604;
int32_t x21607 = x21605 + x21606;
float x21618 = x21616 + x21617;
x21591[x21607] = x21618;

}

}

}

}
float* x21628 = (float*)myMalloc(x21590 * sizeof(float));;
for(int x21630=0; x21630 < x21590; x21630++) {
float x21631 = x21591[x21630];
bool x21632 = x21631 < 0.0f;
if (x21632) {
x21628[x21630] = 0.0f;
} else {
float x21635 = x21591[x21630];
x21628[x21630] = x21635;
}

}
float* x21650 = (float*)myMalloc(x21649 * sizeof(float));;
int32_t x21651 = 9 * x21585;
int32_t x21654 = 64 * x21651;
int32_t x21655 = x21654 * x21645;
float* x21656 = (float*)myMalloc(x21655 * sizeof(float));;
int32_t x21652 = x21651 * x21645;
int32_t x21664 = x21585 * 3;
int32_t x21665 = x21664 * 3;
for(int x21657=0; x21657 < 64; x21657++) {
int32_t x21658 = x21657 * x21589;
float* x21659 = x21628+x21658;
int32_t x21660 = x21657 * x21646;
float* x21661 = x21650+x21660;
int32_t x21662 = x21657 * x21652;
float* x21663 = x21656+x21662;
for(int x21667=0; x21667 < x21665; x21667++) {
int32_t x21668 = x21667 / 9;
int32_t x21672 = x21668 * 3;
int32_t x21673 = x21672 * 3;
int32_t x21674 = x21673 * x21644;
int32_t x21675 = x21674 * x21644;
int32_t x21669 = x21667 % 9;
int32_t x21670 = x21669 / 3;
int32_t x21676 = x21670 * 3;
int32_t x21677 = x21676 * x21644;
int32_t x21678 = x21677 * x21644;
int32_t x21679 = x21675 + x21678;
int32_t x21671 = x21669 % 3;
int32_t x21680 = x21671 * x21644;
int32_t x21681 = x21680 * x21644;
int32_t x21682 = x21679 + x21681;
float* x21683 = x21663+x21682;
int32_t x21684 = x21668 * x21587;
int32_t x21685 = x21684 * x21587;
float* x21686 = x21659+x21685;
int32_t x21699 = 1 - x21671;
bool x21700 = x21699 > 0;
int32_t x21701;
if (x21700) {
x21701 = x21699;
} else {
x21701 = 0;
}
int32_t x21702 = 3 - x21671;
int32_t x21703 = x21702 - 1;
int32_t x21704 = 1 - x21703;
bool x21705 = x21704 > 0;
int32_t x21706;
if (x21705) {
x21706 = x21704;
} else {
x21706 = 0;
}
int32_t x21707 = x21644 - x21706;
int32_t x21708 = x21707 - x21701;
bool x21709 = x21708 <= 0;
bool x21713 = x21701 > 0;
int32_t x21698 = -1 + x21671;
bool x21726 = x21706 > 0;
for(int x21688=0; x21688 < x21644; x21688++) {
int32_t x21689 = x21688 - 1;
int32_t x21690 = x21689 + x21670;
bool x21691 = x21690 < 0;
bool x21692 = x21690 >= x21587;
bool x21693 = x21691 || x21692;
if (x21693) {
int32_t x21694 = x21688 * x21644;
float* x21695 = x21683+x21694;
memset(x21695, 0, 4 * x21644);;
} else {
if (x21709) {
int32_t x21694 = x21688 * x21644;
float* x21710 = x21683+x21694;
memset(x21710, 0, 4 * x21644);;
} else {
int32_t x21694 = x21688 * x21644;
if (x21713) {
float* x21714 = x21683+x21694;
memset(x21714, 0, 4 * x21701);;
} else {
}
// may have segfault here
int32_t x21719 = x21694 + x21701;
float* x21720 = x21683+x21719;
int32_t x21721 = x21690 * x21587;
int32_t x21722 = x21721 + x21698;
int32_t x21723 = x21722 + x21701;
float* x21724 = x21686+x21723;
memcpy(x21720, x21724, 4 * x21708);;
if (x21726) {
int32_t x21727 = x21694 + x21644;
int32_t x21728 = x21727 - x21706;
float* x21729 = x21683+x21728;
memset(x21729, 0, 4 * x21706);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x21645,x21651,1,x155,x21651,x21663,x21645,1,x21661,x21645);

}
int32_t x21744 = 0;
int32_t x21745 = 1;
x21745 *= 1;
x21744 += 1;
x21745 *= 1;
x21745 *= 1;
int32_t x21750 = x21744;
bool x21751 = x21750 >= 2;
if (x21751) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21756 = x21750 == 0;
if (x21756) {
int32_t x21757 = x21745;
bool x21758 = x21757 == 512;
if (x21758) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21765 = x21745;
int32_t x21766 = 512 / x21765;
bool x21767 = x21766 == 1;
bool x21770;
if (x454) {
bool x21768 = 512 == x21766;
bool x21769 = x21767 || x21768;
x21770 = x21769;
} else {
x21770 = false;
}
bool x21774;
if (x21770) {
x21774 = x21773;
} else {
x21774 = false;
}
bool x21775;
if (x21774) {
x21775 = x21773;
} else {
x21775 = false;
}
if (x21775) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x21644,x21644,1,x21766,1,1);
assert(false && "");
}
bool x21781 = 512 <= x21766;
int32_t x21782;
if (x21781) {
x21782 = x21766;
} else {
x21782 = 512;
}
int32_t x21786 = x21782 * x21785;
int32_t x21787 = 64 * x21786;
float* x21788 = (float*)myMalloc(x21787 * sizeof(float));;
int32_t x21791;
if (x21767) {
x21791 = 0;
} else {
x21791 = 1;
}
for(int x21792=0; x21792 < 64; x21792++) {
int32_t x21804 = x21646 * x21792;
int32_t x21798 = x21786 * x21792;
for(int x21794=0; x21794 < x21782; x21794++) {
int32_t x21805 = x21645 * x21794;
int32_t x21806 = x21804 + x21805;
int32_t x21811 = x21791 * x21794;
int32_t x21800 = x21785 * x21794;
for(int x21796=0; x21796 < x21784; x21796++) {
int32_t x21807 = x21789 * x21796;
int32_t x21808 = x21806 + x21807;
int32_t x21802 = x21784 * x21796;
for(int x21797=0; x21797 < x21784; x21797++) {
int32_t x21809 = x21790 * x21797;
int32_t x21810 = x21808 + x21809;
float x21812 = x21650[x21810];
float x21813 = x138[x21811];
int32_t x21799 = x21797 + x21798;
int32_t x21801 = x21799 + x21800;
int32_t x21803 = x21801 + x21802;
float x21814 = x21812 - x21813;
x21788[x21803] = x21814;

}

}

}

}
float* x21824 = (float*)myMalloc(512 * sizeof(float));;
for(int x21825=0; x21825 < 512; x21825++) {
float x21826 = x195[x21825];
float x21827 = x21826 + 1.0E-5f;
x21824[x21825] = x21827;

}
float* x21831 = (float*)myMalloc(512 * sizeof(float));;
for(int x21832=0; x21832 < 512; x21832++) {
float x21833 = x21824[x21832];
double x21834 = (double)x21833;
double x21835 = sqrt(x21834);
float x21836 = (float)x21835;
x21831[x21832] = x21836;

}
int32_t x21840 = 0;
int32_t x21841 = 1;
x21841 *= 1;
x21840 += 1;
x21841 *= 1;
x21841 *= 1;
int32_t x21846 = x21840;
bool x21847 = x21846 >= 2;
if (x21847) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21852 = x21846 == 0;
if (x21852) {
int32_t x21853 = x21841;
bool x21854 = x21853 == 512;
if (x21854) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21861 = x21841;
bool x21863 = x21782 == 1;
int32_t x21862 = 512 / x21861;
bool x21864 = x21862 == 1;
bool x21868;
if (x454) {
bool x21865 = x21863 || x21864;
bool x21866 = x21782 == x21862;
bool x21867 = x21865 || x21866;
x21868 = x21867;
} else {
x21868 = false;
}
bool x21872;
if (x21868) {
x21872 = x21871;
} else {
x21872 = false;
}
bool x21873;
if (x21872) {
x21873 = x21871;
} else {
x21873 = false;
}
if (x21873) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21782,x21784,x21784,1,x21862,1,1);
assert(false && "");
}
bool x21879 = x21782 <= x21862;
int32_t x21880;
if (x21879) {
x21880 = x21862;
} else {
x21880 = x21782;
}
int32_t x21884 = x21880 * x21883;
int32_t x21885 = 64 * x21884;
float* x21886 = (float*)myMalloc(x21885 * sizeof(float));;
int32_t x21887;
if (x21863) {
x21887 = 0;
} else {
x21887 = x21785;
}
int32_t x21890;
if (x21864) {
x21890 = 0;
} else {
x21890 = 1;
}
for(int x21891=0; x21891 < 64; x21891++) {
int32_t x21903 = x21786 * x21891;
int32_t x21897 = x21884 * x21891;
for(int x21893=0; x21893 < x21880; x21893++) {
int32_t x21904 = x21887 * x21893;
int32_t x21905 = x21903 + x21904;
int32_t x21910 = x21890 * x21893;
int32_t x21899 = x21883 * x21893;
for(int x21895=0; x21895 < x21882; x21895++) {
int32_t x21906 = x21888 * x21895;
int32_t x21907 = x21905 + x21906;
int32_t x21901 = x21882 * x21895;
for(int x21896=0; x21896 < x21882; x21896++) {
int32_t x21908 = x21889 * x21896;
int32_t x21909 = x21907 + x21908;
float x21911 = x21788[x21909];
float x21912 = x21831[x21910];
int32_t x21898 = x21896 + x21897;
int32_t x21900 = x21898 + x21899;
int32_t x21902 = x21900 + x21901;
float x21913 = x21911 / x21912;
x21886[x21902] = x21913;

}

}

}

}
int32_t x21923 = 0;
int32_t x21924 = 1;
x21924 *= 1;
x21923 += 1;
x21924 *= 1;
x21924 *= 1;
int32_t x21929 = x21923;
bool x21930 = x21929 >= 2;
if (x21930) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21935 = x21929 == 0;
if (x21935) {
int32_t x21936 = x21924;
bool x21937 = x21936 == 512;
if (x21937) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21944 = x21924;
bool x21946 = x21880 == 1;
int32_t x21945 = 512 / x21944;
bool x21947 = x21945 == 1;
bool x21951;
if (x454) {
bool x21948 = x21946 || x21947;
bool x21949 = x21880 == x21945;
bool x21950 = x21948 || x21949;
x21951 = x21950;
} else {
x21951 = false;
}
bool x21955;
if (x21951) {
x21955 = x21954;
} else {
x21955 = false;
}
bool x21956;
if (x21955) {
x21956 = x21954;
} else {
x21956 = false;
}
if (x21956) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21880,x21882,x21882,1,x21945,1,1);
assert(false && "");
}
bool x21962 = x21880 <= x21945;
int32_t x21963;
if (x21962) {
x21963 = x21945;
} else {
x21963 = x21880;
}
int32_t x21967 = x21963 * x21966;
int32_t x21968 = 64 * x21967;
float* x21969 = (float*)myMalloc(x21968 * sizeof(float));;
int32_t x21970;
if (x21946) {
x21970 = 0;
} else {
x21970 = x21883;
}
int32_t x21973;
if (x21947) {
x21973 = 0;
} else {
x21973 = 1;
}
for(int x21974=0; x21974 < 64; x21974++) {
int32_t x21986 = x21884 * x21974;
int32_t x21980 = x21967 * x21974;
for(int x21976=0; x21976 < x21963; x21976++) {
int32_t x21987 = x21970 * x21976;
int32_t x21988 = x21986 + x21987;
int32_t x21993 = x21973 * x21976;
int32_t x21982 = x21966 * x21976;
for(int x21978=0; x21978 < x21965; x21978++) {
int32_t x21989 = x21971 * x21978;
int32_t x21990 = x21988 + x21989;
int32_t x21984 = x21965 * x21978;
for(int x21979=0; x21979 < x21965; x21979++) {
int32_t x21991 = x21972 * x21979;
int32_t x21992 = x21990 + x21991;
float x21994 = x21886[x21992];
float x21995 = x160[x21993];
int32_t x21981 = x21979 + x21980;
int32_t x21983 = x21981 + x21982;
int32_t x21985 = x21983 + x21984;
float x21996 = x21994 * x21995;
x21969[x21985] = x21996;

}

}

}

}
int32_t x22006 = 0;
int32_t x22007 = 1;
x22007 *= 1;
x22006 += 1;
x22007 *= 1;
x22007 *= 1;
int32_t x22012 = x22006;
bool x22013 = x22012 >= 2;
if (x22013) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22018 = x22012 == 0;
if (x22018) {
int32_t x22019 = x22007;
bool x22020 = x22019 == 512;
if (x22020) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22027 = x22007;
bool x22029 = x21963 == 1;
int32_t x22028 = 512 / x22027;
bool x22030 = x22028 == 1;
bool x22034;
if (x454) {
bool x22031 = x22029 || x22030;
bool x22032 = x21963 == x22028;
bool x22033 = x22031 || x22032;
x22034 = x22033;
} else {
x22034 = false;
}
bool x22038;
if (x22034) {
x22038 = x22037;
} else {
x22038 = false;
}
bool x22039;
if (x22038) {
x22039 = x22037;
} else {
x22039 = false;
}
if (x22039) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21963,x21965,x21965,1,x22028,1,1);
assert(false && "");
}
bool x22045 = x21963 <= x22028;
int32_t x22046;
if (x22045) {
x22046 = x22028;
} else {
x22046 = x21963;
}
int32_t x22050 = x22046 * x22049;
int32_t x22051 = 64 * x22050;
float* x22052 = (float*)myMalloc(x22051 * sizeof(float));;
int32_t x22053;
if (x22029) {
x22053 = 0;
} else {
x22053 = x21966;
}
int32_t x22056;
if (x22030) {
x22056 = 0;
} else {
x22056 = 1;
}
for(int x22057=0; x22057 < 64; x22057++) {
int32_t x22069 = x21967 * x22057;
int32_t x22063 = x22050 * x22057;
for(int x22059=0; x22059 < x22046; x22059++) {
int32_t x22070 = x22053 * x22059;
int32_t x22071 = x22069 + x22070;
int32_t x22076 = x22056 * x22059;
int32_t x22065 = x22049 * x22059;
for(int x22061=0; x22061 < x22048; x22061++) {
int32_t x22072 = x22054 * x22061;
int32_t x22073 = x22071 + x22072;
int32_t x22067 = x22048 * x22061;
for(int x22062=0; x22062 < x22048; x22062++) {
int32_t x22074 = x22055 * x22062;
int32_t x22075 = x22073 + x22074;
float x22077 = x21969[x22075];
float x22078 = x66[x22076];
int32_t x22064 = x22062 + x22063;
int32_t x22066 = x22064 + x22065;
int32_t x22068 = x22066 + x22067;
float x22079 = x22077 + x22078;
x22052[x22068] = x22079;

}

}

}

}
float* x22089 = (float*)myMalloc(x22051 * sizeof(float));;
for(int x22091=0; x22091 < x22051; x22091++) {
float x22092 = x22052[x22091];
bool x22093 = x22092 < 0.0f;
if (x22093) {
x22089[x22091] = 0.0f;
} else {
float x22096 = x22052[x22091];
x22089[x22091] = x22096;
}

}
float* x22110 = (float*)myMalloc(x22109 * sizeof(float));;
int32_t x22113 = 64 * x22046;
int32_t x22114 = x22113 * x22105;
float* x22115 = (float*)myMalloc(x22114 * sizeof(float));;
int32_t x22111 = x22046 * x22105;
for(int x22116=0; x22116 < 64; x22116++) {
int32_t x22117 = x22116 * x22050;
float* x22118 = x22089+x22117;
int32_t x22119 = x22116 * x22106;
float* x22120 = x22110+x22119;
int32_t x22121 = x22116 * x22111;
float* x22122 = x22115+x22121;
for(int x22123=0; x22123 < x22046; x22123++) {
int32_t x22124 = x22123 / 1;
int32_t x22128 = x22124 * x22104;
int32_t x22129 = x22128 * x22104;
int32_t x22125 = x22123 % 1;
int32_t x22126 = x22125 / 1;
int32_t x22130 = x22126 * x22104;
int32_t x22131 = x22130 * x22104;
int32_t x22132 = x22129 + x22131;
int32_t x22127 = x22125 % 1;
int32_t x22133 = x22127 * x22104;
int32_t x22134 = x22133 * x22104;
int32_t x22135 = x22132 + x22134;
float* x22136 = x22122+x22135;
int32_t x22137 = x22124 * x22048;
int32_t x22138 = x22137 * x22048;
float* x22139 = x22118+x22138;
for(int x22141=0; x22141 < x22104; x22141++) {
int32_t x22143 = x22141 * x22104;
float* x22144 = x22136+x22143;
int32_t x22142 = x22141 + x22126;
int32_t x22145 = x22142 * x22048;
int32_t x22146 = x22145 + x22127;
float* x22147 = x22139+x22146;
memcpy(x22144, x22147, 4 * x22104);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x22105,x22046,1,x47,x22046,x22122,x22105,1,x22120,x22105);

}
int32_t x22156 = 0;
int32_t x22157 = 1;
x22157 *= 1;
x22156 += 1;
x22157 *= 1;
x22157 *= 1;
int32_t x22162 = x22156;
bool x22163 = x22162 >= 2;
if (x22163) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22168 = x22162 == 0;
if (x22168) {
int32_t x22169 = x22157;
bool x22170 = x22169 == 2048;
if (x22170) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22177 = x22157;
int32_t x22178 = 2048 / x22177;
bool x22179 = x22178 == 1;
bool x22182;
if (x454) {
bool x22180 = 2048 == x22178;
bool x22181 = x22179 || x22180;
x22182 = x22181;
} else {
x22182 = false;
}
bool x22186;
if (x22182) {
x22186 = x22185;
} else {
x22186 = false;
}
bool x22187;
if (x22186) {
x22187 = x22185;
} else {
x22187 = false;
}
if (x22187) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x22104,x22104,1,x22178,1,1);
assert(false && "");
}
bool x22193 = 2048 <= x22178;
int32_t x22194;
if (x22193) {
x22194 = x22178;
} else {
x22194 = 2048;
}
int32_t x22198 = x22194 * x22197;
int32_t x22199 = 64 * x22198;
float* x22200 = (float*)myMalloc(x22199 * sizeof(float));;
int32_t x22203;
if (x22179) {
x22203 = 0;
} else {
x22203 = 1;
}
for(int x22204=0; x22204 < 64; x22204++) {
int32_t x22216 = x22106 * x22204;
int32_t x22210 = x22198 * x22204;
for(int x22206=0; x22206 < x22194; x22206++) {
int32_t x22217 = x22105 * x22206;
int32_t x22218 = x22216 + x22217;
int32_t x22223 = x22203 * x22206;
int32_t x22212 = x22197 * x22206;
for(int x22208=0; x22208 < x22196; x22208++) {
int32_t x22219 = x22201 * x22208;
int32_t x22220 = x22218 + x22219;
int32_t x22214 = x22196 * x22208;
for(int x22209=0; x22209 < x22196; x22209++) {
int32_t x22221 = x22202 * x22209;
int32_t x22222 = x22220 + x22221;
float x22224 = x22110[x22222];
float x22225 = x68[x22223];
int32_t x22211 = x22209 + x22210;
int32_t x22213 = x22211 + x22212;
int32_t x22215 = x22213 + x22214;
float x22226 = x22224 - x22225;
x22200[x22215] = x22226;

}

}

}

}
float* x22236 = (float*)myMalloc(2048 * sizeof(float));;
for(int x22237=0; x22237 < 2048; x22237++) {
float x22238 = x245[x22237];
float x22239 = x22238 + 1.0E-5f;
x22236[x22237] = x22239;

}
float* x22243 = (float*)myMalloc(2048 * sizeof(float));;
for(int x22244=0; x22244 < 2048; x22244++) {
float x22245 = x22236[x22244];
double x22246 = (double)x22245;
double x22247 = sqrt(x22246);
float x22248 = (float)x22247;
x22243[x22244] = x22248;

}
int32_t x22252 = 0;
int32_t x22253 = 1;
x22253 *= 1;
x22252 += 1;
x22253 *= 1;
x22253 *= 1;
int32_t x22258 = x22252;
bool x22259 = x22258 >= 2;
if (x22259) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22264 = x22258 == 0;
if (x22264) {
int32_t x22265 = x22253;
bool x22266 = x22265 == 2048;
if (x22266) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22273 = x22253;
bool x22275 = x22194 == 1;
int32_t x22274 = 2048 / x22273;
bool x22276 = x22274 == 1;
bool x22280;
if (x454) {
bool x22277 = x22275 || x22276;
bool x22278 = x22194 == x22274;
bool x22279 = x22277 || x22278;
x22280 = x22279;
} else {
x22280 = false;
}
bool x22284;
if (x22280) {
x22284 = x22283;
} else {
x22284 = false;
}
bool x22285;
if (x22284) {
x22285 = x22283;
} else {
x22285 = false;
}
if (x22285) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22194,x22196,x22196,1,x22274,1,1);
assert(false && "");
}
bool x22291 = x22194 <= x22274;
int32_t x22292;
if (x22291) {
x22292 = x22274;
} else {
x22292 = x22194;
}
int32_t x22296 = x22292 * x22295;
int32_t x22297 = 64 * x22296;
float* x22298 = (float*)myMalloc(x22297 * sizeof(float));;
int32_t x22299;
if (x22275) {
x22299 = 0;
} else {
x22299 = x22197;
}
int32_t x22302;
if (x22276) {
x22302 = 0;
} else {
x22302 = 1;
}
for(int x22303=0; x22303 < 64; x22303++) {
int32_t x22315 = x22198 * x22303;
int32_t x22309 = x22296 * x22303;
for(int x22305=0; x22305 < x22292; x22305++) {
int32_t x22316 = x22299 * x22305;
int32_t x22317 = x22315 + x22316;
int32_t x22322 = x22302 * x22305;
int32_t x22311 = x22295 * x22305;
for(int x22307=0; x22307 < x22294; x22307++) {
int32_t x22318 = x22300 * x22307;
int32_t x22319 = x22317 + x22318;
int32_t x22313 = x22294 * x22307;
for(int x22308=0; x22308 < x22294; x22308++) {
int32_t x22320 = x22301 * x22308;
int32_t x22321 = x22319 + x22320;
float x22323 = x22200[x22321];
float x22324 = x22243[x22322];
int32_t x22310 = x22308 + x22309;
int32_t x22312 = x22310 + x22311;
int32_t x22314 = x22312 + x22313;
float x22325 = x22323 / x22324;
x22298[x22314] = x22325;

}

}

}

}
int32_t x22335 = 0;
int32_t x22336 = 1;
x22336 *= 1;
x22335 += 1;
x22336 *= 1;
x22336 *= 1;
int32_t x22341 = x22335;
bool x22342 = x22341 >= 2;
if (x22342) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22347 = x22341 == 0;
if (x22347) {
int32_t x22348 = x22336;
bool x22349 = x22348 == 2048;
if (x22349) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22356 = x22336;
bool x22358 = x22292 == 1;
int32_t x22357 = 2048 / x22356;
bool x22359 = x22357 == 1;
bool x22363;
if (x454) {
bool x22360 = x22358 || x22359;
bool x22361 = x22292 == x22357;
bool x22362 = x22360 || x22361;
x22363 = x22362;
} else {
x22363 = false;
}
bool x22367;
if (x22363) {
x22367 = x22366;
} else {
x22367 = false;
}
bool x22368;
if (x22367) {
x22368 = x22366;
} else {
x22368 = false;
}
if (x22368) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22292,x22294,x22294,1,x22357,1,1);
assert(false && "");
}
bool x22374 = x22292 <= x22357;
int32_t x22375;
if (x22374) {
x22375 = x22357;
} else {
x22375 = x22292;
}
int32_t x22379 = x22375 * x22378;
int32_t x22380 = 64 * x22379;
float* x22381 = (float*)myMalloc(x22380 * sizeof(float));;
int32_t x22382;
if (x22358) {
x22382 = 0;
} else {
x22382 = x22295;
}
int32_t x22385;
if (x22359) {
x22385 = 0;
} else {
x22385 = 1;
}
for(int x22386=0; x22386 < 64; x22386++) {
int32_t x22398 = x22296 * x22386;
int32_t x22392 = x22379 * x22386;
for(int x22388=0; x22388 < x22375; x22388++) {
int32_t x22399 = x22382 * x22388;
int32_t x22400 = x22398 + x22399;
int32_t x22405 = x22385 * x22388;
int32_t x22394 = x22378 * x22388;
for(int x22390=0; x22390 < x22377; x22390++) {
int32_t x22401 = x22383 * x22390;
int32_t x22402 = x22400 + x22401;
int32_t x22396 = x22377 * x22390;
for(int x22391=0; x22391 < x22377; x22391++) {
int32_t x22403 = x22384 * x22391;
int32_t x22404 = x22402 + x22403;
float x22406 = x22298[x22404];
float x22407 = x94[x22405];
int32_t x22393 = x22391 + x22392;
int32_t x22395 = x22393 + x22394;
int32_t x22397 = x22395 + x22396;
float x22408 = x22406 * x22407;
x22381[x22397] = x22408;

}

}

}

}
int32_t x22418 = 0;
int32_t x22419 = 1;
x22419 *= 1;
x22418 += 1;
x22419 *= 1;
x22419 *= 1;
int32_t x22424 = x22418;
bool x22425 = x22424 >= 2;
if (x22425) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22430 = x22424 == 0;
if (x22430) {
int32_t x22431 = x22419;
bool x22432 = x22431 == 2048;
if (x22432) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22439 = x22419;
bool x22441 = x22375 == 1;
int32_t x22440 = 2048 / x22439;
bool x22442 = x22440 == 1;
bool x22446;
if (x454) {
bool x22443 = x22441 || x22442;
bool x22444 = x22375 == x22440;
bool x22445 = x22443 || x22444;
x22446 = x22445;
} else {
x22446 = false;
}
bool x22450;
if (x22446) {
x22450 = x22449;
} else {
x22450 = false;
}
bool x22451;
if (x22450) {
x22451 = x22449;
} else {
x22451 = false;
}
if (x22451) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22375,x22377,x22377,1,x22440,1,1);
assert(false && "");
}
bool x22457 = x22375 <= x22440;
int32_t x22458;
if (x22457) {
x22458 = x22440;
} else {
x22458 = x22375;
}
int32_t x22462 = x22458 * x22461;
int32_t x22463 = 64 * x22462;
float* x22464 = (float*)myMalloc(x22463 * sizeof(float));;
int32_t x22465;
if (x22441) {
x22465 = 0;
} else {
x22465 = x22378;
}
int32_t x22468;
if (x22442) {
x22468 = 0;
} else {
x22468 = 1;
}
for(int x22469=0; x22469 < 64; x22469++) {
int32_t x22481 = x22379 * x22469;
int32_t x22475 = x22462 * x22469;
for(int x22471=0; x22471 < x22458; x22471++) {
int32_t x22482 = x22465 * x22471;
int32_t x22483 = x22481 + x22482;
int32_t x22488 = x22468 * x22471;
int32_t x22477 = x22461 * x22471;
for(int x22473=0; x22473 < x22460; x22473++) {
int32_t x22484 = x22466 * x22473;
int32_t x22485 = x22483 + x22484;
int32_t x22479 = x22460 * x22473;
for(int x22474=0; x22474 < x22460; x22474++) {
int32_t x22486 = x22467 * x22474;
int32_t x22487 = x22485 + x22486;
float x22489 = x22381[x22487];
float x22490 = x144[x22488];
int32_t x22476 = x22474 + x22475;
int32_t x22478 = x22476 + x22477;
int32_t x22480 = x22478 + x22479;
float x22491 = x22489 + x22490;
x22464[x22480] = x22491;

}

}

}

}
bool x22501 = x22458 == 1;
bool x22502 = x22501 || x21154;
bool x22503 = x22458 == x20707;
bool x22504 = x22502 || x22503;
bool x22509;
if (x22504) {
x22509 = x22508;
} else {
x22509 = false;
}
bool x22510;
if (x22509) {
x22510 = x22508;
} else {
x22510 = false;
}
if (x22510) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22458,x22460,x22460,64,x20707,x20709,x20709);
assert(false && "");
}
bool x22516 = x22458 <= x20707;
int32_t x22517;
if (x22516) {
x22517 = x20707;
} else {
x22517 = x22458;
}
int32_t x22523;
if (x22501) {
x22523 = 0;
} else {
x22523 = x22461;
}
for(int x22526=0; x22526 < 64; x22526++) {
int32_t x22532 = x22462 * x22526;
int32_t x22539 = x20711 * x22526;
for(int x22528=0; x22528 < x22517; x22528++) {
int32_t x22533 = x22523 * x22528;
int32_t x22534 = x22532 + x22533;
int32_t x22540 = x21178 * x22528;
int32_t x22541 = x22539 + x22540;
for(int x22530=0; x22530 < x22519; x22530++) {
int32_t x22535 = x22524 * x22530;
int32_t x22536 = x22534 + x22535;
int32_t x22542 = x21179 * x22530;
int32_t x22543 = x22541 + x22542;
for(int x22531=0; x22531 < x22519; x22531++) {
int32_t x22537 = x22525 * x22531;
int32_t x22538 = x22536 + x22537;
float x22546 = x22464[x22538];
int32_t x22544 = x21180 * x22531;
int32_t x22545 = x22543 + x22544;
float x22547 = x21216[x22545];
float x22548 = x22546 + x22547;
x22464[x22538] = x22548;

}

}

}

}
float* x22558 = (float*)myMalloc(x22463 * sizeof(float));;
for(int x22560=0; x22560 < x22463; x22560++) {
float x22561 = x22464[x22560];
bool x22562 = x22561 < 0.0f;
if (x22562) {
x22558[x22560] = 0.0f;
} else {
float x22565 = x22464[x22560];
x22558[x22560] = x22565;
}

}
float* x22579 = (float*)myMalloc(x22578 * sizeof(float));;
int32_t x22582 = 64 * x22458;
int32_t x22583 = x22582 * x22574;
float* x22584 = (float*)myMalloc(x22583 * sizeof(float));;
int32_t x22580 = x22458 * x22574;
for(int x22585=0; x22585 < 64; x22585++) {
int32_t x22586 = x22585 * x22462;
float* x22587 = x22558+x22586;
int32_t x22588 = x22585 * x22575;
float* x22589 = x22579+x22588;
int32_t x22590 = x22585 * x22580;
float* x22591 = x22584+x22590;
for(int x22592=0; x22592 < x22458; x22592++) {
int32_t x22593 = x22592 / 1;
int32_t x22597 = x22593 * x22573;
int32_t x22598 = x22597 * x22573;
int32_t x22594 = x22592 % 1;
int32_t x22595 = x22594 / 1;
int32_t x22599 = x22595 * x22573;
int32_t x22600 = x22599 * x22573;
int32_t x22601 = x22598 + x22600;
int32_t x22596 = x22594 % 1;
int32_t x22602 = x22596 * x22573;
int32_t x22603 = x22602 * x22573;
int32_t x22604 = x22601 + x22603;
float* x22605 = x22591+x22604;
int32_t x22606 = x22593 * x22460;
int32_t x22607 = x22606 * x22460;
float* x22608 = x22587+x22607;
for(int x22610=0; x22610 < x22573; x22610++) {
int32_t x22612 = x22610 * x22573;
float* x22613 = x22605+x22612;
int32_t x22611 = x22610 + x22595;
int32_t x22614 = x22611 * x22460;
int32_t x22615 = x22614 + x22596;
float* x22616 = x22608+x22615;
memcpy(x22613, x22616, 4 * x22573);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x22574,x22458,1,x265,x22458,x22591,x22574,1,x22589,x22574);

}
int32_t x22625 = 0;
int32_t x22626 = 1;
x22626 *= 1;
x22625 += 1;
x22626 *= 1;
x22626 *= 1;
int32_t x22631 = x22625;
bool x22632 = x22631 >= 2;
if (x22632) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22637 = x22631 == 0;
if (x22637) {
int32_t x22638 = x22626;
bool x22639 = x22638 == 512;
if (x22639) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22646 = x22626;
int32_t x22647 = 512 / x22646;
bool x22648 = x22647 == 1;
bool x22651;
if (x454) {
bool x22649 = 512 == x22647;
bool x22650 = x22648 || x22649;
x22651 = x22650;
} else {
x22651 = false;
}
bool x22655;
if (x22651) {
x22655 = x22654;
} else {
x22655 = false;
}
bool x22656;
if (x22655) {
x22656 = x22654;
} else {
x22656 = false;
}
if (x22656) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x22573,x22573,1,x22647,1,1);
assert(false && "");
}
bool x22662 = 512 <= x22647;
int32_t x22663;
if (x22662) {
x22663 = x22647;
} else {
x22663 = 512;
}
int32_t x22667 = x22663 * x22666;
int32_t x22668 = 64 * x22667;
float* x22669 = (float*)myMalloc(x22668 * sizeof(float));;
int32_t x22672;
if (x22648) {
x22672 = 0;
} else {
x22672 = 1;
}
for(int x22673=0; x22673 < 64; x22673++) {
int32_t x22685 = x22575 * x22673;
int32_t x22679 = x22667 * x22673;
for(int x22675=0; x22675 < x22663; x22675++) {
int32_t x22686 = x22574 * x22675;
int32_t x22687 = x22685 + x22686;
int32_t x22692 = x22672 * x22675;
int32_t x22681 = x22666 * x22675;
for(int x22677=0; x22677 < x22665; x22677++) {
int32_t x22688 = x22670 * x22677;
int32_t x22689 = x22687 + x22688;
int32_t x22683 = x22665 * x22677;
for(int x22678=0; x22678 < x22665; x22678++) {
int32_t x22690 = x22671 * x22678;
int32_t x22691 = x22689 + x22690;
float x22693 = x22579[x22691];
float x22694 = x213[x22692];
int32_t x22680 = x22678 + x22679;
int32_t x22682 = x22680 + x22681;
int32_t x22684 = x22682 + x22683;
float x22695 = x22693 - x22694;
x22669[x22684] = x22695;

}

}

}

}
float* x22705 = (float*)myMalloc(512 * sizeof(float));;
for(int x22706=0; x22706 < 512; x22706++) {
float x22707 = x255[x22706];
float x22708 = x22707 + 1.0E-5f;
x22705[x22706] = x22708;

}
float* x22712 = (float*)myMalloc(512 * sizeof(float));;
for(int x22713=0; x22713 < 512; x22713++) {
float x22714 = x22705[x22713];
double x22715 = (double)x22714;
double x22716 = sqrt(x22715);
float x22717 = (float)x22716;
x22712[x22713] = x22717;

}
int32_t x22721 = 0;
int32_t x22722 = 1;
x22722 *= 1;
x22721 += 1;
x22722 *= 1;
x22722 *= 1;
int32_t x22727 = x22721;
bool x22728 = x22727 >= 2;
if (x22728) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22733 = x22727 == 0;
if (x22733) {
int32_t x22734 = x22722;
bool x22735 = x22734 == 512;
if (x22735) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22742 = x22722;
bool x22744 = x22663 == 1;
int32_t x22743 = 512 / x22742;
bool x22745 = x22743 == 1;
bool x22749;
if (x454) {
bool x22746 = x22744 || x22745;
bool x22747 = x22663 == x22743;
bool x22748 = x22746 || x22747;
x22749 = x22748;
} else {
x22749 = false;
}
bool x22753;
if (x22749) {
x22753 = x22752;
} else {
x22753 = false;
}
bool x22754;
if (x22753) {
x22754 = x22752;
} else {
x22754 = false;
}
if (x22754) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22663,x22665,x22665,1,x22743,1,1);
assert(false && "");
}
bool x22760 = x22663 <= x22743;
int32_t x22761;
if (x22760) {
x22761 = x22743;
} else {
x22761 = x22663;
}
int32_t x22765 = x22761 * x22764;
int32_t x22766 = 64 * x22765;
float* x22767 = (float*)myMalloc(x22766 * sizeof(float));;
int32_t x22768;
if (x22744) {
x22768 = 0;
} else {
x22768 = x22666;
}
int32_t x22771;
if (x22745) {
x22771 = 0;
} else {
x22771 = 1;
}
for(int x22772=0; x22772 < 64; x22772++) {
int32_t x22784 = x22667 * x22772;
int32_t x22778 = x22765 * x22772;
for(int x22774=0; x22774 < x22761; x22774++) {
int32_t x22785 = x22768 * x22774;
int32_t x22786 = x22784 + x22785;
int32_t x22791 = x22771 * x22774;
int32_t x22780 = x22764 * x22774;
for(int x22776=0; x22776 < x22763; x22776++) {
int32_t x22787 = x22769 * x22776;
int32_t x22788 = x22786 + x22787;
int32_t x22782 = x22763 * x22776;
for(int x22777=0; x22777 < x22763; x22777++) {
int32_t x22789 = x22770 * x22777;
int32_t x22790 = x22788 + x22789;
float x22792 = x22669[x22790];
float x22793 = x22712[x22791];
int32_t x22779 = x22777 + x22778;
int32_t x22781 = x22779 + x22780;
int32_t x22783 = x22781 + x22782;
float x22794 = x22792 / x22793;
x22767[x22783] = x22794;

}

}

}

}
int32_t x22804 = 0;
int32_t x22805 = 1;
x22805 *= 1;
x22804 += 1;
x22805 *= 1;
x22805 *= 1;
int32_t x22810 = x22804;
bool x22811 = x22810 >= 2;
if (x22811) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22816 = x22810 == 0;
if (x22816) {
int32_t x22817 = x22805;
bool x22818 = x22817 == 512;
if (x22818) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22825 = x22805;
bool x22827 = x22761 == 1;
int32_t x22826 = 512 / x22825;
bool x22828 = x22826 == 1;
bool x22832;
if (x454) {
bool x22829 = x22827 || x22828;
bool x22830 = x22761 == x22826;
bool x22831 = x22829 || x22830;
x22832 = x22831;
} else {
x22832 = false;
}
bool x22836;
if (x22832) {
x22836 = x22835;
} else {
x22836 = false;
}
bool x22837;
if (x22836) {
x22837 = x22835;
} else {
x22837 = false;
}
if (x22837) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22761,x22763,x22763,1,x22826,1,1);
assert(false && "");
}
bool x22843 = x22761 <= x22826;
int32_t x22844;
if (x22843) {
x22844 = x22826;
} else {
x22844 = x22761;
}
int32_t x22848 = x22844 * x22847;
int32_t x22849 = 64 * x22848;
float* x22850 = (float*)myMalloc(x22849 * sizeof(float));;
int32_t x22851;
if (x22827) {
x22851 = 0;
} else {
x22851 = x22764;
}
int32_t x22854;
if (x22828) {
x22854 = 0;
} else {
x22854 = 1;
}
for(int x22855=0; x22855 < 64; x22855++) {
int32_t x22867 = x22765 * x22855;
int32_t x22861 = x22848 * x22855;
for(int x22857=0; x22857 < x22844; x22857++) {
int32_t x22868 = x22851 * x22857;
int32_t x22869 = x22867 + x22868;
int32_t x22874 = x22854 * x22857;
int32_t x22863 = x22847 * x22857;
for(int x22859=0; x22859 < x22846; x22859++) {
int32_t x22870 = x22852 * x22859;
int32_t x22871 = x22869 + x22870;
int32_t x22865 = x22846 * x22859;
for(int x22860=0; x22860 < x22846; x22860++) {
int32_t x22872 = x22853 * x22860;
int32_t x22873 = x22871 + x22872;
float x22875 = x22767[x22873];
float x22876 = x15[x22874];
int32_t x22862 = x22860 + x22861;
int32_t x22864 = x22862 + x22863;
int32_t x22866 = x22864 + x22865;
float x22877 = x22875 * x22876;
x22850[x22866] = x22877;

}

}

}

}
int32_t x22887 = 0;
int32_t x22888 = 1;
x22888 *= 1;
x22887 += 1;
x22888 *= 1;
x22888 *= 1;
int32_t x22893 = x22887;
bool x22894 = x22893 >= 2;
if (x22894) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22899 = x22893 == 0;
if (x22899) {
int32_t x22900 = x22888;
bool x22901 = x22900 == 512;
if (x22901) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22908 = x22888;
bool x22910 = x22844 == 1;
int32_t x22909 = 512 / x22908;
bool x22911 = x22909 == 1;
bool x22915;
if (x454) {
bool x22912 = x22910 || x22911;
bool x22913 = x22844 == x22909;
bool x22914 = x22912 || x22913;
x22915 = x22914;
} else {
x22915 = false;
}
bool x22919;
if (x22915) {
x22919 = x22918;
} else {
x22919 = false;
}
bool x22920;
if (x22919) {
x22920 = x22918;
} else {
x22920 = false;
}
if (x22920) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22844,x22846,x22846,1,x22909,1,1);
assert(false && "");
}
bool x22926 = x22844 <= x22909;
int32_t x22927;
if (x22926) {
x22927 = x22909;
} else {
x22927 = x22844;
}
int32_t x22931 = x22927 * x22930;
int32_t x22932 = 64 * x22931;
float* x22933 = (float*)myMalloc(x22932 * sizeof(float));;
int32_t x22934;
if (x22910) {
x22934 = 0;
} else {
x22934 = x22847;
}
int32_t x22937;
if (x22911) {
x22937 = 0;
} else {
x22937 = 1;
}
for(int x22938=0; x22938 < 64; x22938++) {
int32_t x22950 = x22848 * x22938;
int32_t x22944 = x22931 * x22938;
for(int x22940=0; x22940 < x22927; x22940++) {
int32_t x22951 = x22934 * x22940;
int32_t x22952 = x22950 + x22951;
int32_t x22957 = x22937 * x22940;
int32_t x22946 = x22930 * x22940;
for(int x22942=0; x22942 < x22929; x22942++) {
int32_t x22953 = x22935 * x22942;
int32_t x22954 = x22952 + x22953;
int32_t x22948 = x22929 * x22942;
for(int x22943=0; x22943 < x22929; x22943++) {
int32_t x22955 = x22936 * x22943;
int32_t x22956 = x22954 + x22955;
float x22958 = x22850[x22956];
float x22959 = x78[x22957];
int32_t x22945 = x22943 + x22944;
int32_t x22947 = x22945 + x22946;
int32_t x22949 = x22947 + x22948;
float x22960 = x22958 + x22959;
x22933[x22949] = x22960;

}

}

}

}
float* x22970 = (float*)myMalloc(x22932 * sizeof(float));;
for(int x22972=0; x22972 < x22932; x22972++) {
float x22973 = x22933[x22972];
bool x22974 = x22973 < 0.0f;
if (x22974) {
x22970[x22972] = 0.0f;
} else {
float x22977 = x22933[x22972];
x22970[x22972] = x22977;
}

}
float* x22992 = (float*)myMalloc(x22991 * sizeof(float));;
int32_t x22993 = 9 * x22927;
int32_t x22996 = 64 * x22993;
int32_t x22997 = x22996 * x22987;
float* x22998 = (float*)myMalloc(x22997 * sizeof(float));;
int32_t x22994 = x22993 * x22987;
int32_t x23006 = x22927 * 3;
int32_t x23007 = x23006 * 3;
for(int x22999=0; x22999 < 64; x22999++) {
int32_t x23000 = x22999 * x22931;
float* x23001 = x22970+x23000;
int32_t x23002 = x22999 * x22988;
float* x23003 = x22992+x23002;
int32_t x23004 = x22999 * x22994;
float* x23005 = x22998+x23004;
for(int x23009=0; x23009 < x23007; x23009++) {
int32_t x23010 = x23009 / 9;
int32_t x23014 = x23010 * 3;
int32_t x23015 = x23014 * 3;
int32_t x23016 = x23015 * x22986;
int32_t x23017 = x23016 * x22986;
int32_t x23011 = x23009 % 9;
int32_t x23012 = x23011 / 3;
int32_t x23018 = x23012 * 3;
int32_t x23019 = x23018 * x22986;
int32_t x23020 = x23019 * x22986;
int32_t x23021 = x23017 + x23020;
int32_t x23013 = x23011 % 3;
int32_t x23022 = x23013 * x22986;
int32_t x23023 = x23022 * x22986;
int32_t x23024 = x23021 + x23023;
float* x23025 = x23005+x23024;
int32_t x23026 = x23010 * x22929;
int32_t x23027 = x23026 * x22929;
float* x23028 = x23001+x23027;
int32_t x23041 = 1 - x23013;
bool x23042 = x23041 > 0;
int32_t x23043;
if (x23042) {
x23043 = x23041;
} else {
x23043 = 0;
}
int32_t x23044 = 3 - x23013;
int32_t x23045 = x23044 - 1;
int32_t x23046 = 1 - x23045;
bool x23047 = x23046 > 0;
int32_t x23048;
if (x23047) {
x23048 = x23046;
} else {
x23048 = 0;
}
int32_t x23049 = x22986 - x23048;
int32_t x23050 = x23049 - x23043;
bool x23051 = x23050 <= 0;
bool x23055 = x23043 > 0;
int32_t x23040 = -1 + x23013;
bool x23068 = x23048 > 0;
for(int x23030=0; x23030 < x22986; x23030++) {
int32_t x23031 = x23030 - 1;
int32_t x23032 = x23031 + x23012;
bool x23033 = x23032 < 0;
bool x23034 = x23032 >= x22929;
bool x23035 = x23033 || x23034;
if (x23035) {
int32_t x23036 = x23030 * x22986;
float* x23037 = x23025+x23036;
memset(x23037, 0, 4 * x22986);;
} else {
if (x23051) {
int32_t x23036 = x23030 * x22986;
float* x23052 = x23025+x23036;
memset(x23052, 0, 4 * x22986);;
} else {
int32_t x23036 = x23030 * x22986;
if (x23055) {
float* x23056 = x23025+x23036;
memset(x23056, 0, 4 * x23043);;
} else {
}
// may have segfault here
int32_t x23061 = x23036 + x23043;
float* x23062 = x23025+x23061;
int32_t x23063 = x23032 * x22929;
int32_t x23064 = x23063 + x23040;
int32_t x23065 = x23064 + x23043;
float* x23066 = x23028+x23065;
memcpy(x23062, x23066, 4 * x23050);;
if (x23068) {
int32_t x23069 = x23036 + x22986;
int32_t x23070 = x23069 - x23048;
float* x23071 = x23025+x23070;
memset(x23071, 0, 4 * x23048);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x22987,x22993,1,x28,x22993,x23005,x22987,1,x23003,x22987);

}
int32_t x23086 = 0;
int32_t x23087 = 1;
x23087 *= 1;
x23086 += 1;
x23087 *= 1;
x23087 *= 1;
int32_t x23092 = x23086;
bool x23093 = x23092 >= 2;
if (x23093) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23098 = x23092 == 0;
if (x23098) {
int32_t x23099 = x23087;
bool x23100 = x23099 == 512;
if (x23100) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23107 = x23087;
int32_t x23108 = 512 / x23107;
bool x23109 = x23108 == 1;
bool x23112;
if (x454) {
bool x23110 = 512 == x23108;
bool x23111 = x23109 || x23110;
x23112 = x23111;
} else {
x23112 = false;
}
bool x23116;
if (x23112) {
x23116 = x23115;
} else {
x23116 = false;
}
bool x23117;
if (x23116) {
x23117 = x23115;
} else {
x23117 = false;
}
if (x23117) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x22986,x22986,1,x23108,1,1);
assert(false && "");
}
bool x23123 = 512 <= x23108;
int32_t x23124;
if (x23123) {
x23124 = x23108;
} else {
x23124 = 512;
}
int32_t x23128 = x23124 * x23127;
int32_t x23129 = 64 * x23128;
float* x23130 = (float*)myMalloc(x23129 * sizeof(float));;
int32_t x23133;
if (x23109) {
x23133 = 0;
} else {
x23133 = 1;
}
for(int x23134=0; x23134 < 64; x23134++) {
int32_t x23146 = x22988 * x23134;
int32_t x23140 = x23128 * x23134;
for(int x23136=0; x23136 < x23124; x23136++) {
int32_t x23147 = x22987 * x23136;
int32_t x23148 = x23146 + x23147;
int32_t x23153 = x23133 * x23136;
int32_t x23142 = x23127 * x23136;
for(int x23138=0; x23138 < x23126; x23138++) {
int32_t x23149 = x23131 * x23138;
int32_t x23150 = x23148 + x23149;
int32_t x23144 = x23126 * x23138;
for(int x23139=0; x23139 < x23126; x23139++) {
int32_t x23151 = x23132 * x23139;
int32_t x23152 = x23150 + x23151;
float x23154 = x22992[x23152];
float x23155 = x12[x23153];
int32_t x23141 = x23139 + x23140;
int32_t x23143 = x23141 + x23142;
int32_t x23145 = x23143 + x23144;
float x23156 = x23154 - x23155;
x23130[x23145] = x23156;

}

}

}

}
float* x23166 = (float*)myMalloc(512 * sizeof(float));;
for(int x23167=0; x23167 < 512; x23167++) {
float x23168 = x202[x23167];
float x23169 = x23168 + 1.0E-5f;
x23166[x23167] = x23169;

}
float* x23173 = (float*)myMalloc(512 * sizeof(float));;
for(int x23174=0; x23174 < 512; x23174++) {
float x23175 = x23166[x23174];
double x23176 = (double)x23175;
double x23177 = sqrt(x23176);
float x23178 = (float)x23177;
x23173[x23174] = x23178;

}
int32_t x23182 = 0;
int32_t x23183 = 1;
x23183 *= 1;
x23182 += 1;
x23183 *= 1;
x23183 *= 1;
int32_t x23188 = x23182;
bool x23189 = x23188 >= 2;
if (x23189) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23194 = x23188 == 0;
if (x23194) {
int32_t x23195 = x23183;
bool x23196 = x23195 == 512;
if (x23196) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23203 = x23183;
bool x23205 = x23124 == 1;
int32_t x23204 = 512 / x23203;
bool x23206 = x23204 == 1;
bool x23210;
if (x454) {
bool x23207 = x23205 || x23206;
bool x23208 = x23124 == x23204;
bool x23209 = x23207 || x23208;
x23210 = x23209;
} else {
x23210 = false;
}
bool x23214;
if (x23210) {
x23214 = x23213;
} else {
x23214 = false;
}
bool x23215;
if (x23214) {
x23215 = x23213;
} else {
x23215 = false;
}
if (x23215) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23124,x23126,x23126,1,x23204,1,1);
assert(false && "");
}
bool x23221 = x23124 <= x23204;
int32_t x23222;
if (x23221) {
x23222 = x23204;
} else {
x23222 = x23124;
}
int32_t x23226 = x23222 * x23225;
int32_t x23227 = 64 * x23226;
float* x23228 = (float*)myMalloc(x23227 * sizeof(float));;
int32_t x23229;
if (x23205) {
x23229 = 0;
} else {
x23229 = x23127;
}
int32_t x23232;
if (x23206) {
x23232 = 0;
} else {
x23232 = 1;
}
for(int x23233=0; x23233 < 64; x23233++) {
int32_t x23245 = x23128 * x23233;
int32_t x23239 = x23226 * x23233;
for(int x23235=0; x23235 < x23222; x23235++) {
int32_t x23246 = x23229 * x23235;
int32_t x23247 = x23245 + x23246;
int32_t x23252 = x23232 * x23235;
int32_t x23241 = x23225 * x23235;
for(int x23237=0; x23237 < x23224; x23237++) {
int32_t x23248 = x23230 * x23237;
int32_t x23249 = x23247 + x23248;
int32_t x23243 = x23224 * x23237;
for(int x23238=0; x23238 < x23224; x23238++) {
int32_t x23250 = x23231 * x23238;
int32_t x23251 = x23249 + x23250;
float x23253 = x23130[x23251];
float x23254 = x23173[x23252];
int32_t x23240 = x23238 + x23239;
int32_t x23242 = x23240 + x23241;
int32_t x23244 = x23242 + x23243;
float x23255 = x23253 / x23254;
x23228[x23244] = x23255;

}

}

}

}
int32_t x23265 = 0;
int32_t x23266 = 1;
x23266 *= 1;
x23265 += 1;
x23266 *= 1;
x23266 *= 1;
int32_t x23271 = x23265;
bool x23272 = x23271 >= 2;
if (x23272) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23277 = x23271 == 0;
if (x23277) {
int32_t x23278 = x23266;
bool x23279 = x23278 == 512;
if (x23279) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23286 = x23266;
bool x23288 = x23222 == 1;
int32_t x23287 = 512 / x23286;
bool x23289 = x23287 == 1;
bool x23293;
if (x454) {
bool x23290 = x23288 || x23289;
bool x23291 = x23222 == x23287;
bool x23292 = x23290 || x23291;
x23293 = x23292;
} else {
x23293 = false;
}
bool x23297;
if (x23293) {
x23297 = x23296;
} else {
x23297 = false;
}
bool x23298;
if (x23297) {
x23298 = x23296;
} else {
x23298 = false;
}
if (x23298) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23222,x23224,x23224,1,x23287,1,1);
assert(false && "");
}
bool x23304 = x23222 <= x23287;
int32_t x23305;
if (x23304) {
x23305 = x23287;
} else {
x23305 = x23222;
}
int32_t x23309 = x23305 * x23308;
int32_t x23310 = 64 * x23309;
float* x23311 = (float*)myMalloc(x23310 * sizeof(float));;
int32_t x23312;
if (x23288) {
x23312 = 0;
} else {
x23312 = x23225;
}
int32_t x23315;
if (x23289) {
x23315 = 0;
} else {
x23315 = 1;
}
for(int x23316=0; x23316 < 64; x23316++) {
int32_t x23328 = x23226 * x23316;
int32_t x23322 = x23309 * x23316;
for(int x23318=0; x23318 < x23305; x23318++) {
int32_t x23329 = x23312 * x23318;
int32_t x23330 = x23328 + x23329;
int32_t x23335 = x23315 * x23318;
int32_t x23324 = x23308 * x23318;
for(int x23320=0; x23320 < x23307; x23320++) {
int32_t x23331 = x23313 * x23320;
int32_t x23332 = x23330 + x23331;
int32_t x23326 = x23307 * x23320;
for(int x23321=0; x23321 < x23307; x23321++) {
int32_t x23333 = x23314 * x23321;
int32_t x23334 = x23332 + x23333;
float x23336 = x23228[x23334];
float x23337 = x194[x23335];
int32_t x23323 = x23321 + x23322;
int32_t x23325 = x23323 + x23324;
int32_t x23327 = x23325 + x23326;
float x23338 = x23336 * x23337;
x23311[x23327] = x23338;

}

}

}

}
int32_t x23348 = 0;
int32_t x23349 = 1;
x23349 *= 1;
x23348 += 1;
x23349 *= 1;
x23349 *= 1;
int32_t x23354 = x23348;
bool x23355 = x23354 >= 2;
if (x23355) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23360 = x23354 == 0;
if (x23360) {
int32_t x23361 = x23349;
bool x23362 = x23361 == 512;
if (x23362) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23369 = x23349;
bool x23371 = x23305 == 1;
int32_t x23370 = 512 / x23369;
bool x23372 = x23370 == 1;
bool x23376;
if (x454) {
bool x23373 = x23371 || x23372;
bool x23374 = x23305 == x23370;
bool x23375 = x23373 || x23374;
x23376 = x23375;
} else {
x23376 = false;
}
bool x23380;
if (x23376) {
x23380 = x23379;
} else {
x23380 = false;
}
bool x23381;
if (x23380) {
x23381 = x23379;
} else {
x23381 = false;
}
if (x23381) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23305,x23307,x23307,1,x23370,1,1);
assert(false && "");
}
bool x23387 = x23305 <= x23370;
int32_t x23388;
if (x23387) {
x23388 = x23370;
} else {
x23388 = x23305;
}
int32_t x23392 = x23388 * x23391;
int32_t x23393 = 64 * x23392;
float* x23394 = (float*)myMalloc(x23393 * sizeof(float));;
int32_t x23395;
if (x23371) {
x23395 = 0;
} else {
x23395 = x23308;
}
int32_t x23398;
if (x23372) {
x23398 = 0;
} else {
x23398 = 1;
}
for(int x23399=0; x23399 < 64; x23399++) {
int32_t x23411 = x23309 * x23399;
int32_t x23405 = x23392 * x23399;
for(int x23401=0; x23401 < x23388; x23401++) {
int32_t x23412 = x23395 * x23401;
int32_t x23413 = x23411 + x23412;
int32_t x23418 = x23398 * x23401;
int32_t x23407 = x23391 * x23401;
for(int x23403=0; x23403 < x23390; x23403++) {
int32_t x23414 = x23396 * x23403;
int32_t x23415 = x23413 + x23414;
int32_t x23409 = x23390 * x23403;
for(int x23404=0; x23404 < x23390; x23404++) {
int32_t x23416 = x23397 * x23404;
int32_t x23417 = x23415 + x23416;
float x23419 = x23311[x23417];
float x23420 = x169[x23418];
int32_t x23406 = x23404 + x23405;
int32_t x23408 = x23406 + x23407;
int32_t x23410 = x23408 + x23409;
float x23421 = x23419 + x23420;
x23394[x23410] = x23421;

}

}

}

}
float* x23431 = (float*)myMalloc(x23393 * sizeof(float));;
for(int x23433=0; x23433 < x23393; x23433++) {
float x23434 = x23394[x23433];
bool x23435 = x23434 < 0.0f;
if (x23435) {
x23431[x23433] = 0.0f;
} else {
float x23438 = x23394[x23433];
x23431[x23433] = x23438;
}

}
float* x23452 = (float*)myMalloc(x23451 * sizeof(float));;
int32_t x23455 = 64 * x23388;
int32_t x23456 = x23455 * x23447;
float* x23457 = (float*)myMalloc(x23456 * sizeof(float));;
int32_t x23453 = x23388 * x23447;
for(int x23458=0; x23458 < 64; x23458++) {
int32_t x23459 = x23458 * x23392;
float* x23460 = x23431+x23459;
int32_t x23461 = x23458 * x23448;
float* x23462 = x23452+x23461;
int32_t x23463 = x23458 * x23453;
float* x23464 = x23457+x23463;
for(int x23465=0; x23465 < x23388; x23465++) {
int32_t x23466 = x23465 / 1;
int32_t x23470 = x23466 * x23446;
int32_t x23471 = x23470 * x23446;
int32_t x23467 = x23465 % 1;
int32_t x23468 = x23467 / 1;
int32_t x23472 = x23468 * x23446;
int32_t x23473 = x23472 * x23446;
int32_t x23474 = x23471 + x23473;
int32_t x23469 = x23467 % 1;
int32_t x23475 = x23469 * x23446;
int32_t x23476 = x23475 * x23446;
int32_t x23477 = x23474 + x23476;
float* x23478 = x23464+x23477;
int32_t x23479 = x23466 * x23390;
int32_t x23480 = x23479 * x23390;
float* x23481 = x23460+x23480;
for(int x23483=0; x23483 < x23446; x23483++) {
int32_t x23485 = x23483 * x23446;
float* x23486 = x23478+x23485;
int32_t x23484 = x23483 + x23468;
int32_t x23487 = x23484 * x23390;
int32_t x23488 = x23487 + x23469;
float* x23489 = x23481+x23488;
memcpy(x23486, x23489, 4 * x23446);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x23447,x23388,1,x33,x23388,x23464,x23447,1,x23462,x23447);

}
int32_t x23498 = 0;
int32_t x23499 = 1;
x23499 *= 1;
x23498 += 1;
x23499 *= 1;
x23499 *= 1;
int32_t x23504 = x23498;
bool x23505 = x23504 >= 2;
if (x23505) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23510 = x23504 == 0;
if (x23510) {
int32_t x23511 = x23499;
bool x23512 = x23511 == 2048;
if (x23512) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23519 = x23499;
int32_t x23520 = 2048 / x23519;
bool x23521 = x23520 == 1;
bool x23524;
if (x454) {
bool x23522 = 2048 == x23520;
bool x23523 = x23521 || x23522;
x23524 = x23523;
} else {
x23524 = false;
}
bool x23528;
if (x23524) {
x23528 = x23527;
} else {
x23528 = false;
}
bool x23529;
if (x23528) {
x23529 = x23527;
} else {
x23529 = false;
}
if (x23529) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x23446,x23446,1,x23520,1,1);
assert(false && "");
}
bool x23535 = 2048 <= x23520;
int32_t x23536;
if (x23535) {
x23536 = x23520;
} else {
x23536 = 2048;
}
int32_t x23540 = x23536 * x23539;
int32_t x23541 = 64 * x23540;
float* x23542 = (float*)myMalloc(x23541 * sizeof(float));;
int32_t x23545;
if (x23521) {
x23545 = 0;
} else {
x23545 = 1;
}
for(int x23546=0; x23546 < 64; x23546++) {
int32_t x23558 = x23448 * x23546;
int32_t x23552 = x23540 * x23546;
for(int x23548=0; x23548 < x23536; x23548++) {
int32_t x23559 = x23447 * x23548;
int32_t x23560 = x23558 + x23559;
int32_t x23565 = x23545 * x23548;
int32_t x23554 = x23539 * x23548;
for(int x23550=0; x23550 < x23538; x23550++) {
int32_t x23561 = x23543 * x23550;
int32_t x23562 = x23560 + x23561;
int32_t x23556 = x23538 * x23550;
for(int x23551=0; x23551 < x23538; x23551++) {
int32_t x23563 = x23544 * x23551;
int32_t x23564 = x23562 + x23563;
float x23566 = x23452[x23564];
float x23567 = x260[x23565];
int32_t x23553 = x23551 + x23552;
int32_t x23555 = x23553 + x23554;
int32_t x23557 = x23555 + x23556;
float x23568 = x23566 - x23567;
x23542[x23557] = x23568;

}

}

}

}
float* x23578 = (float*)myMalloc(2048 * sizeof(float));;
for(int x23579=0; x23579 < 2048; x23579++) {
float x23580 = x123[x23579];
float x23581 = x23580 + 1.0E-5f;
x23578[x23579] = x23581;

}
float* x23585 = (float*)myMalloc(2048 * sizeof(float));;
for(int x23586=0; x23586 < 2048; x23586++) {
float x23587 = x23578[x23586];
double x23588 = (double)x23587;
double x23589 = sqrt(x23588);
float x23590 = (float)x23589;
x23585[x23586] = x23590;

}
int32_t x23594 = 0;
int32_t x23595 = 1;
x23595 *= 1;
x23594 += 1;
x23595 *= 1;
x23595 *= 1;
int32_t x23600 = x23594;
bool x23601 = x23600 >= 2;
if (x23601) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23606 = x23600 == 0;
if (x23606) {
int32_t x23607 = x23595;
bool x23608 = x23607 == 2048;
if (x23608) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23615 = x23595;
bool x23617 = x23536 == 1;
int32_t x23616 = 2048 / x23615;
bool x23618 = x23616 == 1;
bool x23622;
if (x454) {
bool x23619 = x23617 || x23618;
bool x23620 = x23536 == x23616;
bool x23621 = x23619 || x23620;
x23622 = x23621;
} else {
x23622 = false;
}
bool x23626;
if (x23622) {
x23626 = x23625;
} else {
x23626 = false;
}
bool x23627;
if (x23626) {
x23627 = x23625;
} else {
x23627 = false;
}
if (x23627) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23536,x23538,x23538,1,x23616,1,1);
assert(false && "");
}
bool x23633 = x23536 <= x23616;
int32_t x23634;
if (x23633) {
x23634 = x23616;
} else {
x23634 = x23536;
}
int32_t x23638 = x23634 * x23637;
int32_t x23639 = 64 * x23638;
float* x23640 = (float*)myMalloc(x23639 * sizeof(float));;
int32_t x23641;
if (x23617) {
x23641 = 0;
} else {
x23641 = x23539;
}
int32_t x23644;
if (x23618) {
x23644 = 0;
} else {
x23644 = 1;
}
for(int x23645=0; x23645 < 64; x23645++) {
int32_t x23657 = x23540 * x23645;
int32_t x23651 = x23638 * x23645;
for(int x23647=0; x23647 < x23634; x23647++) {
int32_t x23658 = x23641 * x23647;
int32_t x23659 = x23657 + x23658;
int32_t x23664 = x23644 * x23647;
int32_t x23653 = x23637 * x23647;
for(int x23649=0; x23649 < x23636; x23649++) {
int32_t x23660 = x23642 * x23649;
int32_t x23661 = x23659 + x23660;
int32_t x23655 = x23636 * x23649;
for(int x23650=0; x23650 < x23636; x23650++) {
int32_t x23662 = x23643 * x23650;
int32_t x23663 = x23661 + x23662;
float x23665 = x23542[x23663];
float x23666 = x23585[x23664];
int32_t x23652 = x23650 + x23651;
int32_t x23654 = x23652 + x23653;
int32_t x23656 = x23654 + x23655;
float x23667 = x23665 / x23666;
x23640[x23656] = x23667;

}

}

}

}
int32_t x23677 = 0;
int32_t x23678 = 1;
x23678 *= 1;
x23677 += 1;
x23678 *= 1;
x23678 *= 1;
int32_t x23683 = x23677;
bool x23684 = x23683 >= 2;
if (x23684) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23689 = x23683 == 0;
if (x23689) {
int32_t x23690 = x23678;
bool x23691 = x23690 == 2048;
if (x23691) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23698 = x23678;
bool x23700 = x23634 == 1;
int32_t x23699 = 2048 / x23698;
bool x23701 = x23699 == 1;
bool x23705;
if (x454) {
bool x23702 = x23700 || x23701;
bool x23703 = x23634 == x23699;
bool x23704 = x23702 || x23703;
x23705 = x23704;
} else {
x23705 = false;
}
bool x23709;
if (x23705) {
x23709 = x23708;
} else {
x23709 = false;
}
bool x23710;
if (x23709) {
x23710 = x23708;
} else {
x23710 = false;
}
if (x23710) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23634,x23636,x23636,1,x23699,1,1);
assert(false && "");
}
bool x23716 = x23634 <= x23699;
int32_t x23717;
if (x23716) {
x23717 = x23699;
} else {
x23717 = x23634;
}
int32_t x23721 = x23717 * x23720;
int32_t x23722 = 64 * x23721;
float* x23723 = (float*)myMalloc(x23722 * sizeof(float));;
int32_t x23724;
if (x23700) {
x23724 = 0;
} else {
x23724 = x23637;
}
int32_t x23727;
if (x23701) {
x23727 = 0;
} else {
x23727 = 1;
}
for(int x23728=0; x23728 < 64; x23728++) {
int32_t x23740 = x23638 * x23728;
int32_t x23734 = x23721 * x23728;
for(int x23730=0; x23730 < x23717; x23730++) {
int32_t x23741 = x23724 * x23730;
int32_t x23742 = x23740 + x23741;
int32_t x23747 = x23727 * x23730;
int32_t x23736 = x23720 * x23730;
for(int x23732=0; x23732 < x23719; x23732++) {
int32_t x23743 = x23725 * x23732;
int32_t x23744 = x23742 + x23743;
int32_t x23738 = x23719 * x23732;
for(int x23733=0; x23733 < x23719; x23733++) {
int32_t x23745 = x23726 * x23733;
int32_t x23746 = x23744 + x23745;
float x23748 = x23640[x23746];
float x23749 = x103[x23747];
int32_t x23735 = x23733 + x23734;
int32_t x23737 = x23735 + x23736;
int32_t x23739 = x23737 + x23738;
float x23750 = x23748 * x23749;
x23723[x23739] = x23750;

}

}

}

}
int32_t x23760 = 0;
int32_t x23761 = 1;
x23761 *= 1;
x23760 += 1;
x23761 *= 1;
x23761 *= 1;
int32_t x23766 = x23760;
bool x23767 = x23766 >= 2;
if (x23767) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23772 = x23766 == 0;
if (x23772) {
int32_t x23773 = x23761;
bool x23774 = x23773 == 2048;
if (x23774) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23781 = x23761;
bool x23783 = x23717 == 1;
int32_t x23782 = 2048 / x23781;
bool x23784 = x23782 == 1;
bool x23788;
if (x454) {
bool x23785 = x23783 || x23784;
bool x23786 = x23717 == x23782;
bool x23787 = x23785 || x23786;
x23788 = x23787;
} else {
x23788 = false;
}
bool x23792;
if (x23788) {
x23792 = x23791;
} else {
x23792 = false;
}
bool x23793;
if (x23792) {
x23793 = x23791;
} else {
x23793 = false;
}
if (x23793) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23717,x23719,x23719,1,x23782,1,1);
assert(false && "");
}
bool x23799 = x23717 <= x23782;
int32_t x23800;
if (x23799) {
x23800 = x23782;
} else {
x23800 = x23717;
}
int32_t x23804 = x23800 * x23803;
int32_t x23805 = 64 * x23804;
float* x23806 = (float*)myMalloc(x23805 * sizeof(float));;
int32_t x23807;
if (x23783) {
x23807 = 0;
} else {
x23807 = x23720;
}
int32_t x23810;
if (x23784) {
x23810 = 0;
} else {
x23810 = 1;
}
for(int x23811=0; x23811 < 64; x23811++) {
int32_t x23823 = x23721 * x23811;
int32_t x23817 = x23804 * x23811;
for(int x23813=0; x23813 < x23800; x23813++) {
int32_t x23824 = x23807 * x23813;
int32_t x23825 = x23823 + x23824;
int32_t x23830 = x23810 * x23813;
int32_t x23819 = x23803 * x23813;
for(int x23815=0; x23815 < x23802; x23815++) {
int32_t x23826 = x23808 * x23815;
int32_t x23827 = x23825 + x23826;
int32_t x23821 = x23802 * x23815;
for(int x23816=0; x23816 < x23802; x23816++) {
int32_t x23828 = x23809 * x23816;
int32_t x23829 = x23827 + x23828;
float x23831 = x23723[x23829];
float x23832 = x181[x23830];
int32_t x23818 = x23816 + x23817;
int32_t x23820 = x23818 + x23819;
int32_t x23822 = x23820 + x23821;
float x23833 = x23831 + x23832;
x23806[x23822] = x23833;

}

}

}

}
bool x23843 = x23800 == 1;
bool x23844 = x23843 || x22501;
bool x23845 = x23800 == x22458;
bool x23846 = x23844 || x23845;
bool x23851;
if (x23846) {
x23851 = x23850;
} else {
x23851 = false;
}
bool x23852;
if (x23851) {
x23852 = x23850;
} else {
x23852 = false;
}
if (x23852) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23800,x23802,x23802,64,x22458,x22460,x22460);
assert(false && "");
}
bool x23858 = x23800 <= x22458;
int32_t x23859;
if (x23858) {
x23859 = x22458;
} else {
x23859 = x23800;
}
int32_t x23865;
if (x23843) {
x23865 = 0;
} else {
x23865 = x23803;
}
for(int x23868=0; x23868 < 64; x23868++) {
int32_t x23874 = x23804 * x23868;
int32_t x23881 = x22462 * x23868;
for(int x23870=0; x23870 < x23859; x23870++) {
int32_t x23875 = x23865 * x23870;
int32_t x23876 = x23874 + x23875;
int32_t x23882 = x22523 * x23870;
int32_t x23883 = x23881 + x23882;
for(int x23872=0; x23872 < x23861; x23872++) {
int32_t x23877 = x23866 * x23872;
int32_t x23878 = x23876 + x23877;
int32_t x23884 = x22524 * x23872;
int32_t x23885 = x23883 + x23884;
for(int x23873=0; x23873 < x23861; x23873++) {
int32_t x23879 = x23867 * x23873;
int32_t x23880 = x23878 + x23879;
float x23888 = x23806[x23880];
int32_t x23886 = x22525 * x23873;
int32_t x23887 = x23885 + x23886;
float x23889 = x22558[x23887];
float x23890 = x23888 + x23889;
x23806[x23880] = x23890;

}

}

}

}
float* x23900 = (float*)myMalloc(x23805 * sizeof(float));;
for(int x23902=0; x23902 < x23805; x23902++) {
float x23903 = x23806[x23902];
bool x23904 = x23903 < 0.0f;
if (x23904) {
x23900[x23902] = 0.0f;
} else {
float x23907 = x23806[x23902];
x23900[x23902] = x23907;
}

}
if (x23914) {
} else {
assert(false && "Image too small for averagePool_batch:  x Const(64) x Sym(23800) x Sym(23802) x Sym(23802)|(2,2)");
}
int32_t x23925 = 64 * x23800;
int32_t x23926 = x23925 * x23921;
int32_t x23927 = x23926 * x23921;
float* x23928 = (float*)myMalloc(x23927 * sizeof(float));;
int32_t x23923 = x23800 * x23922;
for(int x23929=0; x23929 < 64; x23929++) {
int32_t x23930 = x23929 * x23804;
float* x23931 = x23900+x23930;
int32_t x23932 = x23929 * x23923;
float* x23933 = x23928+x23932;
for(int x23934=0; x23934 < x23800; x23934++) {
int32_t x23942 = x23934 * x23803;
int32_t x23938 = x23934 * x23922;
for(int x23936=0; x23936 < x23921; x23936++) {
int32_t x23943 = x23936 * x23802;
int32_t x23944 = x23942 + x23943;
int32_t x23939 = x23936 * x23921;
int32_t x23940 = x23938 + x23939;
for(int x23937=0; x23937 < x23921; x23937++) {
float x23946 = 0.0f;
int32_t x23945 = x23944 + x23937;
float x23947 = x23931[x23945];
x23946 += x23947;
int32_t x23949 = x23945 + 1;
float x23950 = x23931[x23949];
x23946 += x23950;
int32_t x23952 = x23945 + x23802;
float x23953 = x23931[x23952];
x23946 += x23953;
int32_t x23955 = x23952 + 1;
float x23956 = x23931[x23955];
x23946 += x23956;
float x23958 = x23946;
int32_t x23941 = x23940 + x23937;
float x23959 = x23958 / 4.0f;
x23933[x23941] = x23959;

}

}

}

}
int32_t x23969 = 0;
int32_t x23970 = 1;
x23970 *= 64;
x23969 += 1;
int32_t x23973 = x23969;
bool x23974 = x23973 >= 2;
if (x23974) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23979 = x23973 == 0;
int32_t x23924 = 64 * x23923;
if (x23979) {
int32_t x23980 = x23970;
bool x23981 = x23980 == x23924;
if (x23981) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23988 = x23970;
// gemm: List(Const(64), Sym(23989)), Vector(Const(10), Const(2048))
assert(false && "ERROR not specified");
float* x23993 = (float*)myMalloc(640 * sizeof(float));;
int32_t x23989 = x23924 / x23988;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 64,10,x23989,1.0,x23928,x23989,x227,x23989,0,x23993,10);
for(int x23995=0; x23995 < 64; x23995++) {
int32_t x23997 = 10 * x23995;
for(int x23996=0; x23996 < 10; x23996++) {
int32_t x23998 = x23997 + x23996;
float x23999 = x23993[x23998];
float x24000 = x48[x23996];
float x24001 = x23999 + x24000;
x23993[x23998] = x24001;

}

}
printf("output (size Const(64) x Const(10))\n");
float x24008 = 0.0f;
for(int x24010=0; x24010 < 640; x24010++) {
float x24011 = x24008;
float x24013 = x23993[x24010];
float x24012 = fabs(x24011);
float x24014 = fabs(x24013);
bool x24015 = x24012 > x24014;
float x24018;
if (x24015) {
x24018 = x24011;
} else {
float x24016 = x23993[x24010];
x24018 = x24016;
}
x24008 = x24018;

}
float x24022 = x24008;
printf("Max Abs: %.5f || ",x24022);
for(int x24024=0; x24024 < 10; x24024++) {
float x24025 = x23993[x24024];
printf("%.5f ",x24025);

}
printf("\n");
assert(false && "stop");

}
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

