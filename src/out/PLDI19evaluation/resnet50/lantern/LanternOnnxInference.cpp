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
int32_t x330 = 31 / 1;
int32_t x331 = x330 + 1;
int32_t x335 = 4096 * x331;
int32_t x336 = x335 * x331;
int32_t x332 = x331 * x331;
int32_t x340 = 1728 * x332;
int32_t x333 = 64 * x332;
int32_t x338 = 27 * x332;
int32_t x3 = open("/home/fei/bitbucket/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
int64_t x4 = fsize(x3);
float* x5 = (float*)mmap(0, x4, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x3, 0);
float* x152 = x5+0;
bool x457 = x331 == 1;
bool x458 = x457 || true;
bool x459 = x458 || x457;
bool x452 = true || false;
bool x469 = x331 <= 1;
int32_t x470;
if (x469) {
x470 = 1;
} else {
x470 = x331;
}
bool x475 = x470 > 0;
int32_t x471 = x470 * x470;
float* x40 = x5+1856;
bool x519 = x331 > 1;
float* x110 = x5+1920;
bool x588 = x470 == 1;
bool x589 = x588 || true;
bool x590 = x589 || x588;
bool x600 = x470 <= 1;
int32_t x601;
if (x600) {
x601 = 1;
} else {
x601 = x470;
}
bool x606 = x601 > 0;
int32_t x602 = x601 * x601;
bool x650 = x470 > 1;
bool x706 = x601 == 1;
bool x707 = x706 || true;
bool x708 = x707 || x706;
bool x718 = x601 <= 1;
int32_t x719;
if (x718) {
x719 = 1;
} else {
x719 = x601;
}
bool x724 = x719 > 0;
int32_t x720 = x719 * x719;
float* x206 = x5+1728;
bool x768 = x601 > 1;
bool x824 = x719 == 1;
bool x825 = x824 || true;
bool x826 = x825 || x824;
bool x836 = x719 <= 1;
int32_t x837;
if (x836) {
x837 = 1;
} else {
x837 = x719;
}
bool x842 = x837 > 0;
int32_t x838 = x837 * x837;
float* x251 = x5+1792;
bool x886 = x719 > 1;
bool x926 = x837 >= 2;
bool x927;
if (x926) {
x927 = x926;
} else {
x927 = false;
}
int32_t x932 = x837 - 2;
int32_t x933 = x932 / 2;
int32_t x934 = x933 + 1;
int32_t x935 = x934 * x934;
int32_t x1026 = 2 * x837;
int32_t x1036 = x933 / 1;
int32_t x1037 = x1036 + 1;
int32_t x1041 = 4096 * x1037;
int32_t x1042 = x1041 * x1037;
int32_t x1038 = x1037 * x1037;
int32_t x1039 = 64 * x1038;
float* x233 = x5+1984;
bool x1115 = x1037 == 1;
bool x1116 = x1115 || true;
bool x1117 = x1116 || x1115;
bool x1127 = x1037 <= 1;
int32_t x1128;
if (x1127) {
x1128 = 1;
} else {
x1128 = x1037;
}
bool x1133 = x1128 > 0;
int32_t x1129 = x1128 * x1128;
float* x114 = x5+6208;
bool x1177 = x1037 > 1;
float* x51 = x5+6272;
bool x1246 = x1128 == 1;
bool x1247 = x1246 || true;
bool x1248 = x1247 || x1246;
bool x1258 = x1128 <= 1;
int32_t x1259;
if (x1258) {
x1259 = 1;
} else {
x1259 = x1128;
}
bool x1264 = x1259 > 0;
int32_t x1260 = x1259 * x1259;
bool x1308 = x1128 > 1;
bool x1364 = x1259 == 1;
bool x1365 = x1364 || true;
bool x1366 = x1365 || x1364;
bool x1376 = x1259 <= 1;
int32_t x1377;
if (x1376) {
x1377 = 1;
} else {
x1377 = x1259;
}
bool x1382 = x1377 > 0;
int32_t x1378 = x1377 * x1377;
float* x26 = x5+6080;
bool x1426 = x1259 > 1;
bool x1482 = x1377 == 1;
bool x1483 = x1482 || true;
bool x1484 = x1483 || x1482;
bool x1494 = x1377 <= 1;
int32_t x1495;
if (x1494) {
x1495 = 1;
} else {
x1495 = x1377;
}
bool x1500 = x1495 > 0;
int32_t x1496 = x1495 * x1495;
float* x53 = x5+6144;
bool x1544 = x1377 > 1;
int32_t x1584 = x1495 + 2;
int32_t x1585 = x1584 - 3;
int32_t x1586 = x1585 / 1;
int32_t x1587 = x1586 + 1;
int32_t x1591 = 4096 * x1587;
int32_t x1592 = x1591 * x1587;
int32_t x1588 = x1587 * x1587;
int32_t x1589 = 64 * x1588;
float* x90 = x5+6336;
bool x1714 = x1587 == 1;
bool x1715 = x1714 || true;
bool x1716 = x1715 || x1714;
bool x1726 = x1587 <= 1;
int32_t x1727;
if (x1726) {
x1727 = 1;
} else {
x1727 = x1587;
}
bool x1732 = x1727 > 0;
int32_t x1728 = x1727 * x1727;
float* x105 = x5+43328;
bool x1776 = x1587 > 1;
float* x158 = x5+43392;
bool x1845 = x1727 == 1;
bool x1846 = x1845 || true;
bool x1847 = x1846 || x1845;
bool x1857 = x1727 <= 1;
int32_t x1858;
if (x1857) {
x1858 = 1;
} else {
x1858 = x1727;
}
bool x1863 = x1858 > 0;
int32_t x1859 = x1858 * x1858;
bool x1907 = x1727 > 1;
bool x1963 = x1858 == 1;
bool x1964 = x1963 || true;
bool x1965 = x1964 || x1963;
bool x1975 = x1858 <= 1;
int32_t x1976;
if (x1975) {
x1976 = 1;
} else {
x1976 = x1858;
}
bool x1981 = x1976 > 0;
int32_t x1977 = x1976 * x1976;
float* x164 = x5+43200;
bool x2025 = x1858 > 1;
bool x2081 = x1976 == 1;
bool x2082 = x2081 || true;
bool x2083 = x2082 || x2081;
bool x2093 = x1976 <= 1;
int32_t x2094;
if (x2093) {
x2094 = 1;
} else {
x2094 = x1976;
}
bool x2099 = x2094 > 0;
int32_t x2095 = x2094 * x2094;
float* x49 = x5+43264;
bool x2143 = x1976 > 1;
int32_t x2183 = x2094 - 1;
int32_t x2184 = x2183 / 1;
int32_t x2185 = x2184 + 1;
int32_t x2189 = 16384 * x2185;
int32_t x2190 = x2189 * x2185;
int32_t x2186 = x2185 * x2185;
int32_t x2187 = 256 * x2186;
float* x32 = x5+43456;
bool x2264 = x2185 == 1;
bool x2265 = x2264 || true;
bool x2266 = x2265 || x2264;
bool x2276 = x2185 <= 1;
int32_t x2277;
if (x2276) {
x2277 = 1;
} else {
x2277 = x2185;
}
bool x2282 = x2277 > 0;
int32_t x2278 = x2277 * x2277;
float* x71 = x5+60352;
bool x2326 = x2185 > 1;
float* x36 = x5+60608;
bool x2396 = x2277 == 1;
bool x2397 = x2396 || true;
bool x2398 = x2397 || x2396;
bool x2408 = x2277 <= 1;
int32_t x2409;
if (x2408) {
x2409 = 1;
} else {
x2409 = x2277;
}
bool x2414 = x2409 > 0;
int32_t x2410 = x2409 * x2409;
bool x2458 = x2277 > 1;
bool x2514 = x2409 == 1;
bool x2515 = x2514 || true;
bool x2516 = x2515 || x2514;
bool x2526 = x2409 <= 1;
int32_t x2527;
if (x2526) {
x2527 = 1;
} else {
x2527 = x2409;
}
bool x2532 = x2527 > 0;
int32_t x2528 = x2527 * x2527;
float* x199 = x5+59840;
bool x2576 = x2409 > 1;
bool x2632 = x2527 == 1;
bool x2633 = x2632 || true;
bool x2634 = x2633 || x2632;
bool x2644 = x2527 <= 1;
int32_t x2645;
if (x2644) {
x2645 = 1;
} else {
x2645 = x2527;
}
bool x2650 = x2645 > 0;
int32_t x2646 = x2645 * x2645;
float* x126 = x5+60096;
bool x2694 = x2527 > 1;
int32_t x2723 = 16384 * x1037;
int32_t x2724 = x2723 * x1037;
int32_t x2721 = 256 * x1038;
float* x162 = x5+60864;
float* x264 = x5+77760;
float* x243 = x5+78016;
float* x76 = x5+77248;
float* x203 = x5+77504;
bool x3218 = x2645 == 1;
bool x3219 = x1495 == 1;
bool x3220 = x3218 || x3219;
bool x3221 = x2645 == x1495;
bool x3222 = x3220 || x3221;
bool x3232 = x2645 <= x1495;
int32_t x3233;
if (x3232) {
x3233 = x1495;
} else {
x3233 = x2645;
}
bool x3271 = x2645 > 1;
bool x3275 = x1495 > 1;
int32_t x3234 = x3233 * x3233;
int32_t x3319 = x2645 - 1;
int32_t x3320 = x3319 / 1;
int32_t x3321 = x3320 + 1;
int32_t x3325 = 4096 * x3321;
int32_t x3326 = x3325 * x3321;
int32_t x3322 = x3321 * x3321;
int32_t x3323 = 64 * x3322;
float* x171 = x5+78272;
bool x3400 = x3321 == 1;
bool x3401 = x3400 || true;
bool x3402 = x3401 || x3400;
bool x3412 = x3321 <= 1;
int32_t x3413;
if (x3412) {
x3413 = 1;
} else {
x3413 = x3321;
}
bool x3418 = x3413 > 0;
int32_t x3414 = x3413 * x3413;
float* x10 = x5+94784;
bool x3462 = x3321 > 1;
float* x102 = x5+94848;
bool x3531 = x3413 == 1;
bool x3532 = x3531 || true;
bool x3533 = x3532 || x3531;
bool x3543 = x3413 <= 1;
int32_t x3544;
if (x3543) {
x3544 = 1;
} else {
x3544 = x3413;
}
bool x3549 = x3544 > 0;
int32_t x3545 = x3544 * x3544;
bool x3593 = x3413 > 1;
bool x3649 = x3544 == 1;
bool x3650 = x3649 || true;
bool x3651 = x3650 || x3649;
bool x3661 = x3544 <= 1;
int32_t x3662;
if (x3661) {
x3662 = 1;
} else {
x3662 = x3544;
}
bool x3667 = x3662 > 0;
int32_t x3663 = x3662 * x3662;
float* x142 = x5+94656;
bool x3711 = x3544 > 1;
bool x3767 = x3662 == 1;
bool x3768 = x3767 || true;
bool x3769 = x3768 || x3767;
bool x3779 = x3662 <= 1;
int32_t x3780;
if (x3779) {
x3780 = 1;
} else {
x3780 = x3662;
}
bool x3785 = x3780 > 0;
int32_t x3781 = x3780 * x3780;
float* x60 = x5+94720;
bool x3829 = x3662 > 1;
int32_t x3869 = x3780 + 2;
int32_t x3870 = x3869 - 3;
int32_t x3871 = x3870 / 1;
int32_t x3872 = x3871 + 1;
int32_t x3876 = 4096 * x3872;
int32_t x3877 = x3876 * x3872;
int32_t x3873 = x3872 * x3872;
int32_t x3874 = 64 * x3873;
float* x83 = x5+94912;
bool x3999 = x3872 == 1;
bool x4000 = x3999 || true;
bool x4001 = x4000 || x3999;
bool x4011 = x3872 <= 1;
int32_t x4012;
if (x4011) {
x4012 = 1;
} else {
x4012 = x3872;
}
bool x4017 = x4012 > 0;
int32_t x4013 = x4012 * x4012;
float* x44 = x5+131904;
bool x4061 = x3872 > 1;
float* x244 = x5+131968;
bool x4130 = x4012 == 1;
bool x4131 = x4130 || true;
bool x4132 = x4131 || x4130;
bool x4142 = x4012 <= 1;
int32_t x4143;
if (x4142) {
x4143 = 1;
} else {
x4143 = x4012;
}
bool x4148 = x4143 > 0;
int32_t x4144 = x4143 * x4143;
bool x4192 = x4012 > 1;
bool x4248 = x4143 == 1;
bool x4249 = x4248 || true;
bool x4250 = x4249 || x4248;
bool x4260 = x4143 <= 1;
int32_t x4261;
if (x4260) {
x4261 = 1;
} else {
x4261 = x4143;
}
bool x4266 = x4261 > 0;
int32_t x4262 = x4261 * x4261;
float* x208 = x5+131776;
bool x4310 = x4143 > 1;
bool x4366 = x4261 == 1;
bool x4367 = x4366 || true;
bool x4368 = x4367 || x4366;
bool x4378 = x4261 <= 1;
int32_t x4379;
if (x4378) {
x4379 = 1;
} else {
x4379 = x4261;
}
bool x4384 = x4379 > 0;
int32_t x4380 = x4379 * x4379;
float* x153 = x5+131840;
bool x4428 = x4261 > 1;
int32_t x4468 = x4379 - 1;
int32_t x4469 = x4468 / 1;
int32_t x4470 = x4469 + 1;
int32_t x4474 = 16384 * x4470;
int32_t x4475 = x4474 * x4470;
int32_t x4471 = x4470 * x4470;
int32_t x4472 = 256 * x4471;
float* x130 = x5+132032;
bool x4549 = x4470 == 1;
bool x4550 = x4549 || true;
bool x4551 = x4550 || x4549;
bool x4561 = x4470 <= 1;
int32_t x4562;
if (x4561) {
x4562 = 1;
} else {
x4562 = x4470;
}
bool x4567 = x4562 > 0;
int32_t x4563 = x4562 * x4562;
float* x91 = x5+148928;
bool x4611 = x4470 > 1;
float* x166 = x5+149184;
bool x4680 = x4562 == 1;
bool x4681 = x4680 || true;
bool x4682 = x4681 || x4680;
bool x4692 = x4562 <= 1;
int32_t x4693;
if (x4692) {
x4693 = 1;
} else {
x4693 = x4562;
}
bool x4698 = x4693 > 0;
int32_t x4694 = x4693 * x4693;
bool x4742 = x4562 > 1;
bool x4798 = x4693 == 1;
bool x4799 = x4798 || true;
bool x4800 = x4799 || x4798;
bool x4810 = x4693 <= 1;
int32_t x4811;
if (x4810) {
x4811 = 1;
} else {
x4811 = x4693;
}
bool x4816 = x4811 > 0;
int32_t x4812 = x4811 * x4811;
float* x58 = x5+148416;
bool x4860 = x4693 > 1;
bool x4916 = x4811 == 1;
bool x4917 = x4916 || true;
bool x4918 = x4917 || x4916;
bool x4928 = x4811 <= 1;
int32_t x4929;
if (x4928) {
x4929 = 1;
} else {
x4929 = x4811;
}
bool x4934 = x4929 > 0;
int32_t x4930 = x4929 * x4929;
float* x7 = x5+148672;
bool x4978 = x4811 > 1;
bool x5009 = x4929 == 1;
bool x5010 = x5009 || x3218;
bool x5011 = x4929 == x2645;
bool x5012 = x5010 || x5011;
bool x5022 = x4929 <= x2645;
int32_t x5023;
if (x5022) {
x5023 = x2645;
} else {
x5023 = x4929;
}
bool x5061 = x4929 > 1;
int32_t x5024 = x5023 * x5023;
int32_t x5107 = x4929 - 1;
int32_t x5108 = x5107 / 1;
int32_t x5109 = x5108 + 1;
int32_t x5113 = 4096 * x5109;
int32_t x5114 = x5113 * x5109;
int32_t x5110 = x5109 * x5109;
int32_t x5111 = 64 * x5110;
float* x150 = x5+149440;
bool x5188 = x5109 == 1;
bool x5189 = x5188 || true;
bool x5190 = x5189 || x5188;
bool x5200 = x5109 <= 1;
int32_t x5201;
if (x5200) {
x5201 = 1;
} else {
x5201 = x5109;
}
bool x5206 = x5201 > 0;
int32_t x5202 = x5201 * x5201;
float* x257 = x5+165952;
bool x5250 = x5109 > 1;
float* x187 = x5+166016;
bool x5319 = x5201 == 1;
bool x5320 = x5319 || true;
bool x5321 = x5320 || x5319;
bool x5331 = x5201 <= 1;
int32_t x5332;
if (x5331) {
x5332 = 1;
} else {
x5332 = x5201;
}
bool x5337 = x5332 > 0;
int32_t x5333 = x5332 * x5332;
bool x5381 = x5201 > 1;
bool x5437 = x5332 == 1;
bool x5438 = x5437 || true;
bool x5439 = x5438 || x5437;
bool x5449 = x5332 <= 1;
int32_t x5450;
if (x5449) {
x5450 = 1;
} else {
x5450 = x5332;
}
bool x5455 = x5450 > 0;
int32_t x5451 = x5450 * x5450;
float* x81 = x5+165824;
bool x5499 = x5332 > 1;
bool x5555 = x5450 == 1;
bool x5556 = x5555 || true;
bool x5557 = x5556 || x5555;
bool x5567 = x5450 <= 1;
int32_t x5568;
if (x5567) {
x5568 = 1;
} else {
x5568 = x5450;
}
bool x5573 = x5568 > 0;
int32_t x5569 = x5568 * x5568;
float* x24 = x5+165888;
bool x5617 = x5450 > 1;
int32_t x5657 = x5568 + 2;
int32_t x5658 = x5657 - 3;
int32_t x5659 = x5658 / 1;
int32_t x5660 = x5659 + 1;
int32_t x5664 = 4096 * x5660;
int32_t x5665 = x5664 * x5660;
int32_t x5661 = x5660 * x5660;
int32_t x5662 = 64 * x5661;
float* x73 = x5+166080;
bool x5787 = x5660 == 1;
bool x5788 = x5787 || true;
bool x5789 = x5788 || x5787;
bool x5799 = x5660 <= 1;
int32_t x5800;
if (x5799) {
x5800 = 1;
} else {
x5800 = x5660;
}
bool x5805 = x5800 > 0;
int32_t x5801 = x5800 * x5800;
float* x179 = x5+203072;
bool x5849 = x5660 > 1;
float* x118 = x5+203136;
bool x5918 = x5800 == 1;
bool x5919 = x5918 || true;
bool x5920 = x5919 || x5918;
bool x5930 = x5800 <= 1;
int32_t x5931;
if (x5930) {
x5931 = 1;
} else {
x5931 = x5800;
}
bool x5936 = x5931 > 0;
int32_t x5932 = x5931 * x5931;
bool x5980 = x5800 > 1;
bool x6036 = x5931 == 1;
bool x6037 = x6036 || true;
bool x6038 = x6037 || x6036;
bool x6048 = x5931 <= 1;
int32_t x6049;
if (x6048) {
x6049 = 1;
} else {
x6049 = x5931;
}
bool x6054 = x6049 > 0;
int32_t x6050 = x6049 * x6049;
float* x72 = x5+202944;
bool x6098 = x5931 > 1;
bool x6154 = x6049 == 1;
bool x6155 = x6154 || true;
bool x6156 = x6155 || x6154;
bool x6166 = x6049 <= 1;
int32_t x6167;
if (x6166) {
x6167 = 1;
} else {
x6167 = x6049;
}
bool x6172 = x6167 > 0;
int32_t x6168 = x6167 * x6167;
float* x135 = x5+203008;
bool x6216 = x6049 > 1;
int32_t x6256 = x6167 - 1;
int32_t x6257 = x6256 / 1;
int32_t x6258 = x6257 + 1;
int32_t x6262 = 16384 * x6258;
int32_t x6263 = x6262 * x6258;
int32_t x6259 = x6258 * x6258;
int32_t x6260 = 256 * x6259;
float* x87 = x5+203200;
bool x6337 = x6258 == 1;
bool x6338 = x6337 || true;
bool x6339 = x6338 || x6337;
bool x6349 = x6258 <= 1;
int32_t x6350;
if (x6349) {
x6350 = 1;
} else {
x6350 = x6258;
}
bool x6355 = x6350 > 0;
int32_t x6351 = x6350 * x6350;
float* x184 = x5+220096;
bool x6399 = x6258 > 1;
float* x133 = x5+220352;
bool x6468 = x6350 == 1;
bool x6469 = x6468 || true;
bool x6470 = x6469 || x6468;
bool x6480 = x6350 <= 1;
int32_t x6481;
if (x6480) {
x6481 = 1;
} else {
x6481 = x6350;
}
bool x6486 = x6481 > 0;
int32_t x6482 = x6481 * x6481;
bool x6530 = x6350 > 1;
bool x6586 = x6481 == 1;
bool x6587 = x6586 || true;
bool x6588 = x6587 || x6586;
bool x6598 = x6481 <= 1;
int32_t x6599;
if (x6598) {
x6599 = 1;
} else {
x6599 = x6481;
}
bool x6604 = x6599 > 0;
int32_t x6600 = x6599 * x6599;
float* x37 = x5+219584;
bool x6648 = x6481 > 1;
bool x6704 = x6599 == 1;
bool x6705 = x6704 || true;
bool x6706 = x6705 || x6704;
bool x6716 = x6599 <= 1;
int32_t x6717;
if (x6716) {
x6717 = 1;
} else {
x6717 = x6599;
}
bool x6722 = x6717 > 0;
int32_t x6718 = x6717 * x6717;
float* x247 = x5+219840;
bool x6766 = x6599 > 1;
bool x6797 = x6717 == 1;
bool x6798 = x6797 || x5009;
bool x6799 = x6717 == x4929;
bool x6800 = x6798 || x6799;
bool x6810 = x6717 <= x4929;
int32_t x6811;
if (x6810) {
x6811 = x4929;
} else {
x6811 = x6717;
}
bool x6849 = x6717 > 1;
int32_t x6812 = x6811 * x6811;
int32_t x6895 = x6717 - 1;
int32_t x6896 = x6895 / 1;
int32_t x6897 = x6896 + 1;
int32_t x6901 = 8192 * x6897;
int32_t x6902 = x6901 * x6897;
int32_t x6898 = x6897 * x6897;
int32_t x6899 = 128 * x6898;
float* x11 = x5+220608;
bool x6976 = x6897 == 1;
bool x6977 = x6976 || true;
bool x6978 = x6977 || x6976;
bool x6988 = x6897 <= 1;
int32_t x6989;
if (x6988) {
x6989 = 1;
} else {
x6989 = x6897;
}
bool x6994 = x6989 > 0;
int32_t x6990 = x6989 * x6989;
float* x204 = x5+253632;
bool x7038 = x6897 > 1;
float* x134 = x5+253760;
bool x7108 = x6989 == 1;
bool x7109 = x7108 || true;
bool x7110 = x7109 || x7108;
bool x7120 = x6989 <= 1;
int32_t x7121;
if (x7120) {
x7121 = 1;
} else {
x7121 = x6989;
}
bool x7126 = x7121 > 0;
int32_t x7122 = x7121 * x7121;
bool x7170 = x6989 > 1;
bool x7226 = x7121 == 1;
bool x7227 = x7226 || true;
bool x7228 = x7227 || x7226;
bool x7238 = x7121 <= 1;
int32_t x7239;
if (x7238) {
x7239 = 1;
} else {
x7239 = x7121;
}
bool x7244 = x7239 > 0;
int32_t x7240 = x7239 * x7239;
float* x84 = x5+253376;
bool x7288 = x7121 > 1;
bool x7344 = x7239 == 1;
bool x7345 = x7344 || true;
bool x7346 = x7345 || x7344;
bool x7356 = x7239 <= 1;
int32_t x7357;
if (x7356) {
x7357 = 1;
} else {
x7357 = x7239;
}
bool x7362 = x7357 > 0;
int32_t x7358 = x7357 * x7357;
float* x172 = x5+253504;
bool x7406 = x7239 > 1;
int32_t x7446 = x7357 + 2;
int32_t x7447 = x7446 - 3;
int32_t x7448 = x7447 / 2;
int32_t x7449 = x7448 + 1;
int32_t x7453 = 8192 * x7449;
int32_t x7454 = x7453 * x7449;
int32_t x7450 = x7449 * x7449;
int32_t x7451 = 128 * x7450;
float* x27 = x5+253888;
bool x7560 = x7449 == 1;
bool x7561 = x7560 || true;
bool x7562 = x7561 || x7560;
bool x7572 = x7449 <= 1;
int32_t x7573;
if (x7572) {
x7573 = 1;
} else {
x7573 = x7449;
}
bool x7578 = x7573 > 0;
int32_t x7574 = x7573 * x7573;
float* x128 = x5+401600;
bool x7622 = x7449 > 1;
float* x43 = x5+401728;
bool x7691 = x7573 == 1;
bool x7692 = x7691 || true;
bool x7693 = x7692 || x7691;
bool x7703 = x7573 <= 1;
int32_t x7704;
if (x7703) {
x7704 = 1;
} else {
x7704 = x7573;
}
bool x7709 = x7704 > 0;
int32_t x7705 = x7704 * x7704;
bool x7753 = x7573 > 1;
bool x7809 = x7704 == 1;
bool x7810 = x7809 || true;
bool x7811 = x7810 || x7809;
bool x7821 = x7704 <= 1;
int32_t x7822;
if (x7821) {
x7822 = 1;
} else {
x7822 = x7704;
}
bool x7827 = x7822 > 0;
int32_t x7823 = x7822 * x7822;
float* x252 = x5+401344;
bool x7871 = x7704 > 1;
bool x7927 = x7822 == 1;
bool x7928 = x7927 || true;
bool x7929 = x7928 || x7927;
bool x7939 = x7822 <= 1;
int32_t x7940;
if (x7939) {
x7940 = 1;
} else {
x7940 = x7822;
}
bool x7945 = x7940 > 0;
int32_t x7941 = x7940 * x7940;
float* x190 = x5+401472;
bool x7989 = x7822 > 1;
int32_t x8029 = x7940 - 1;
int32_t x8030 = x8029 / 1;
int32_t x8031 = x8030 + 1;
int32_t x8035 = 32768 * x8031;
int32_t x8036 = x8035 * x8031;
int32_t x8032 = x8031 * x8031;
int32_t x8033 = 512 * x8032;
float* x106 = x5+401856;
bool x8110 = x8031 == 1;
bool x8111 = x8110 || true;
bool x8112 = x8111 || x8110;
bool x8122 = x8031 <= 1;
int32_t x8123;
if (x8122) {
x8123 = 1;
} else {
x8123 = x8031;
}
bool x8128 = x8123 > 0;
int32_t x8124 = x8123 * x8123;
float* x149 = x5+468416;
bool x8172 = x8031 > 1;
float* x101 = x5+468928;
bool x8242 = x8123 == 1;
bool x8243 = x8242 || true;
bool x8244 = x8243 || x8242;
bool x8254 = x8123 <= 1;
int32_t x8255;
if (x8254) {
x8255 = 1;
} else {
x8255 = x8123;
}
bool x8260 = x8255 > 0;
int32_t x8256 = x8255 * x8255;
bool x8304 = x8123 > 1;
bool x8360 = x8255 == 1;
bool x8361 = x8360 || true;
bool x8362 = x8361 || x8360;
bool x8372 = x8255 <= 1;
int32_t x8373;
if (x8372) {
x8373 = 1;
} else {
x8373 = x8255;
}
bool x8378 = x8373 > 0;
int32_t x8374 = x8373 * x8373;
float* x145 = x5+467392;
bool x8422 = x8255 > 1;
bool x8478 = x8373 == 1;
bool x8479 = x8478 || true;
bool x8480 = x8479 || x8478;
bool x8490 = x8373 <= 1;
int32_t x8491;
if (x8490) {
x8491 = 1;
} else {
x8491 = x8373;
}
bool x8496 = x8491 > 0;
int32_t x8492 = x8491 * x8491;
float* x210 = x5+467904;
bool x8540 = x8373 > 1;
int32_t x8567 = x6895 / 2;
int32_t x8568 = x8567 + 1;
int32_t x8572 = 32768 * x8568;
int32_t x8573 = x8572 * x8568;
int32_t x8569 = x8568 * x8568;
int32_t x8570 = 512 * x8569;
float* x258 = x5+469440;
bool x8653 = x8568 == 1;
bool x8654 = x8653 || true;
bool x8655 = x8654 || x8653;
bool x8665 = x8568 <= 1;
int32_t x8666;
if (x8665) {
x8666 = 1;
} else {
x8666 = x8568;
}
bool x8671 = x8666 > 0;
int32_t x8667 = x8666 * x8666;
float* x42 = x5+601536;
bool x8715 = x8568 > 1;
float* x23 = x5+602048;
bool x8784 = x8666 == 1;
bool x8785 = x8784 || true;
bool x8786 = x8785 || x8784;
bool x8796 = x8666 <= 1;
int32_t x8797;
if (x8796) {
x8797 = 1;
} else {
x8797 = x8666;
}
bool x8802 = x8797 > 0;
int32_t x8798 = x8797 * x8797;
bool x8846 = x8666 > 1;
bool x8902 = x8797 == 1;
bool x8903 = x8902 || true;
bool x8904 = x8903 || x8902;
bool x8914 = x8797 <= 1;
int32_t x8915;
if (x8914) {
x8915 = 1;
} else {
x8915 = x8797;
}
bool x8920 = x8915 > 0;
int32_t x8916 = x8915 * x8915;
float* x207 = x5+600512;
bool x8964 = x8797 > 1;
bool x9020 = x8915 == 1;
bool x9021 = x9020 || true;
bool x9022 = x9021 || x9020;
bool x9032 = x8915 <= 1;
int32_t x9033;
if (x9032) {
x9033 = 1;
} else {
x9033 = x8915;
}
bool x9038 = x9033 > 0;
int32_t x9034 = x9033 * x9033;
float* x119 = x5+601024;
bool x9082 = x8915 > 1;
bool x9114 = x8491 == 1;
bool x9115 = x9033 == 1;
bool x9116 = x9114 || x9115;
bool x9117 = x8491 == x9033;
bool x9118 = x9116 || x9117;
bool x9128 = x8491 <= x9033;
int32_t x9129;
if (x9128) {
x9129 = x9033;
} else {
x9129 = x8491;
}
bool x9167 = x8491 > 1;
bool x9171 = x9033 > 1;
int32_t x9130 = x9129 * x9129;
int32_t x9215 = x8491 - 1;
int32_t x9216 = x9215 / 1;
int32_t x9217 = x9216 + 1;
int32_t x9221 = 8192 * x9217;
int32_t x9222 = x9221 * x9217;
int32_t x9218 = x9217 * x9217;
int32_t x9219 = 128 * x9218;
float* x256 = x5+602560;
bool x9296 = x9217 == 1;
bool x9297 = x9296 || true;
bool x9298 = x9297 || x9296;
bool x9308 = x9217 <= 1;
int32_t x9309;
if (x9308) {
x9309 = 1;
} else {
x9309 = x9217;
}
bool x9314 = x9309 > 0;
int32_t x9310 = x9309 * x9309;
float* x100 = x5+668352;
bool x9358 = x9217 > 1;
float* x177 = x5+668480;
bool x9427 = x9309 == 1;
bool x9428 = x9427 || true;
bool x9429 = x9428 || x9427;
bool x9439 = x9309 <= 1;
int32_t x9440;
if (x9439) {
x9440 = 1;
} else {
x9440 = x9309;
}
bool x9445 = x9440 > 0;
int32_t x9441 = x9440 * x9440;
bool x9489 = x9309 > 1;
bool x9545 = x9440 == 1;
bool x9546 = x9545 || true;
bool x9547 = x9546 || x9545;
bool x9557 = x9440 <= 1;
int32_t x9558;
if (x9557) {
x9558 = 1;
} else {
x9558 = x9440;
}
bool x9563 = x9558 > 0;
int32_t x9559 = x9558 * x9558;
float* x222 = x5+668096;
bool x9607 = x9440 > 1;
bool x9663 = x9558 == 1;
bool x9664 = x9663 || true;
bool x9665 = x9664 || x9663;
bool x9675 = x9558 <= 1;
int32_t x9676;
if (x9675) {
x9676 = 1;
} else {
x9676 = x9558;
}
bool x9681 = x9676 > 0;
int32_t x9677 = x9676 * x9676;
float* x17 = x5+668224;
bool x9725 = x9558 > 1;
int32_t x9765 = x9676 + 2;
int32_t x9766 = x9765 - 3;
int32_t x9767 = x9766 / 1;
int32_t x9768 = x9767 + 1;
int32_t x9772 = 8192 * x9768;
int32_t x9773 = x9772 * x9768;
int32_t x9769 = x9768 * x9768;
int32_t x9770 = 128 * x9769;
float* x235 = x5+668608;
bool x9895 = x9768 == 1;
bool x9896 = x9895 || true;
bool x9897 = x9896 || x9895;
bool x9907 = x9768 <= 1;
int32_t x9908;
if (x9907) {
x9908 = 1;
} else {
x9908 = x9768;
}
bool x9913 = x9908 > 0;
int32_t x9909 = x9908 * x9908;
float* x35 = x5+816320;
bool x9957 = x9768 > 1;
float* x225 = x5+816448;
bool x10026 = x9908 == 1;
bool x10027 = x10026 || true;
bool x10028 = x10027 || x10026;
bool x10038 = x9908 <= 1;
int32_t x10039;
if (x10038) {
x10039 = 1;
} else {
x10039 = x9908;
}
bool x10044 = x10039 > 0;
int32_t x10040 = x10039 * x10039;
bool x10088 = x9908 > 1;
bool x10144 = x10039 == 1;
bool x10145 = x10144 || true;
bool x10146 = x10145 || x10144;
bool x10156 = x10039 <= 1;
int32_t x10157;
if (x10156) {
x10157 = 1;
} else {
x10157 = x10039;
}
bool x10162 = x10157 > 0;
int32_t x10158 = x10157 * x10157;
float* x8 = x5+816064;
bool x10206 = x10039 > 1;
bool x10262 = x10157 == 1;
bool x10263 = x10262 || true;
bool x10264 = x10263 || x10262;
bool x10274 = x10157 <= 1;
int32_t x10275;
if (x10274) {
x10275 = 1;
} else {
x10275 = x10157;
}
bool x10280 = x10275 > 0;
int32_t x10276 = x10275 * x10275;
float* x95 = x5+816192;
bool x10324 = x10157 > 1;
int32_t x10364 = x10275 - 1;
int32_t x10365 = x10364 / 1;
int32_t x10366 = x10365 + 1;
int32_t x10370 = 32768 * x10366;
int32_t x10371 = x10370 * x10366;
int32_t x10367 = x10366 * x10366;
int32_t x10368 = 512 * x10367;
float* x111 = x5+816576;
bool x10445 = x10366 == 1;
bool x10446 = x10445 || true;
bool x10447 = x10446 || x10445;
bool x10457 = x10366 <= 1;
int32_t x10458;
if (x10457) {
x10458 = 1;
} else {
x10458 = x10366;
}
bool x10463 = x10458 > 0;
int32_t x10459 = x10458 * x10458;
float* x147 = x5+883136;
bool x10507 = x10366 > 1;
float* x88 = x5+883648;
bool x10576 = x10458 == 1;
bool x10577 = x10576 || true;
bool x10578 = x10577 || x10576;
bool x10588 = x10458 <= 1;
int32_t x10589;
if (x10588) {
x10589 = 1;
} else {
x10589 = x10458;
}
bool x10594 = x10589 > 0;
int32_t x10590 = x10589 * x10589;
bool x10638 = x10458 > 1;
bool x10694 = x10589 == 1;
bool x10695 = x10694 || true;
bool x10696 = x10695 || x10694;
bool x10706 = x10589 <= 1;
int32_t x10707;
if (x10706) {
x10707 = 1;
} else {
x10707 = x10589;
}
bool x10712 = x10707 > 0;
int32_t x10708 = x10707 * x10707;
float* x52 = x5+882112;
bool x10756 = x10589 > 1;
bool x10812 = x10707 == 1;
bool x10813 = x10812 || true;
bool x10814 = x10813 || x10812;
bool x10824 = x10707 <= 1;
int32_t x10825;
if (x10824) {
x10825 = 1;
} else {
x10825 = x10707;
}
bool x10830 = x10825 > 0;
int32_t x10826 = x10825 * x10825;
float* x246 = x5+882624;
bool x10874 = x10707 > 1;
bool x10905 = x10825 == 1;
bool x10906 = x10905 || x9114;
bool x10907 = x10825 == x8491;
bool x10908 = x10906 || x10907;
bool x10918 = x10825 <= x8491;
int32_t x10919;
if (x10918) {
x10919 = x8491;
} else {
x10919 = x10825;
}
bool x10957 = x10825 > 1;
int32_t x10920 = x10919 * x10919;
int32_t x11003 = x10825 - 1;
int32_t x11004 = x11003 / 1;
int32_t x11005 = x11004 + 1;
int32_t x11009 = 8192 * x11005;
int32_t x11010 = x11009 * x11005;
int32_t x11006 = x11005 * x11005;
int32_t x11007 = 128 * x11006;
float* x196 = x5+884160;
bool x11084 = x11005 == 1;
bool x11085 = x11084 || true;
bool x11086 = x11085 || x11084;
bool x11096 = x11005 <= 1;
int32_t x11097;
if (x11096) {
x11097 = 1;
} else {
x11097 = x11005;
}
bool x11102 = x11097 > 0;
int32_t x11098 = x11097 * x11097;
float* x112 = x5+949952;
bool x11146 = x11005 > 1;
float* x9 = x5+950080;
bool x11215 = x11097 == 1;
bool x11216 = x11215 || true;
bool x11217 = x11216 || x11215;
bool x11227 = x11097 <= 1;
int32_t x11228;
if (x11227) {
x11228 = 1;
} else {
x11228 = x11097;
}
bool x11233 = x11228 > 0;
int32_t x11229 = x11228 * x11228;
bool x11277 = x11097 > 1;
bool x11333 = x11228 == 1;
bool x11334 = x11333 || true;
bool x11335 = x11334 || x11333;
bool x11345 = x11228 <= 1;
int32_t x11346;
if (x11345) {
x11346 = 1;
} else {
x11346 = x11228;
}
bool x11351 = x11346 > 0;
int32_t x11347 = x11346 * x11346;
float* x45 = x5+949696;
bool x11395 = x11228 > 1;
bool x11451 = x11346 == 1;
bool x11452 = x11451 || true;
bool x11453 = x11452 || x11451;
bool x11463 = x11346 <= 1;
int32_t x11464;
if (x11463) {
x11464 = 1;
} else {
x11464 = x11346;
}
bool x11469 = x11464 > 0;
int32_t x11465 = x11464 * x11464;
float* x170 = x5+949824;
bool x11513 = x11346 > 1;
int32_t x11553 = x11464 + 2;
int32_t x11554 = x11553 - 3;
int32_t x11555 = x11554 / 1;
int32_t x11556 = x11555 + 1;
int32_t x11560 = 8192 * x11556;
int32_t x11561 = x11560 * x11556;
int32_t x11557 = x11556 * x11556;
int32_t x11558 = 128 * x11557;
float* x191 = x5+950208;
bool x11683 = x11556 == 1;
bool x11684 = x11683 || true;
bool x11685 = x11684 || x11683;
bool x11695 = x11556 <= 1;
int32_t x11696;
if (x11695) {
x11696 = 1;
} else {
x11696 = x11556;
}
bool x11701 = x11696 > 0;
int32_t x11697 = x11696 * x11696;
float* x217 = x5+1097920;
bool x11745 = x11556 > 1;
float* x266 = x5+1098048;
bool x11814 = x11696 == 1;
bool x11815 = x11814 || true;
bool x11816 = x11815 || x11814;
bool x11826 = x11696 <= 1;
int32_t x11827;
if (x11826) {
x11827 = 1;
} else {
x11827 = x11696;
}
bool x11832 = x11827 > 0;
int32_t x11828 = x11827 * x11827;
bool x11876 = x11696 > 1;
bool x11932 = x11827 == 1;
bool x11933 = x11932 || true;
bool x11934 = x11933 || x11932;
bool x11944 = x11827 <= 1;
int32_t x11945;
if (x11944) {
x11945 = 1;
} else {
x11945 = x11827;
}
bool x11950 = x11945 > 0;
int32_t x11946 = x11945 * x11945;
float* x127 = x5+1097664;
bool x11994 = x11827 > 1;
bool x12050 = x11945 == 1;
bool x12051 = x12050 || true;
bool x12052 = x12051 || x12050;
bool x12062 = x11945 <= 1;
int32_t x12063;
if (x12062) {
x12063 = 1;
} else {
x12063 = x11945;
}
bool x12068 = x12063 > 0;
int32_t x12064 = x12063 * x12063;
float* x61 = x5+1097792;
bool x12112 = x11945 > 1;
int32_t x12152 = x12063 - 1;
int32_t x12153 = x12152 / 1;
int32_t x12154 = x12153 + 1;
int32_t x12158 = 32768 * x12154;
int32_t x12159 = x12158 * x12154;
int32_t x12155 = x12154 * x12154;
int32_t x12156 = 512 * x12155;
float* x41 = x5+1098176;
bool x12233 = x12154 == 1;
bool x12234 = x12233 || true;
bool x12235 = x12234 || x12233;
bool x12245 = x12154 <= 1;
int32_t x12246;
if (x12245) {
x12246 = 1;
} else {
x12246 = x12154;
}
bool x12251 = x12246 > 0;
int32_t x12247 = x12246 * x12246;
float* x25 = x5+1164736;
bool x12295 = x12154 > 1;
float* x223 = x5+1165248;
bool x12364 = x12246 == 1;
bool x12365 = x12364 || true;
bool x12366 = x12365 || x12364;
bool x12376 = x12246 <= 1;
int32_t x12377;
if (x12376) {
x12377 = 1;
} else {
x12377 = x12246;
}
bool x12382 = x12377 > 0;
int32_t x12378 = x12377 * x12377;
bool x12426 = x12246 > 1;
bool x12482 = x12377 == 1;
bool x12483 = x12482 || true;
bool x12484 = x12483 || x12482;
bool x12494 = x12377 <= 1;
int32_t x12495;
if (x12494) {
x12495 = 1;
} else {
x12495 = x12377;
}
bool x12500 = x12495 > 0;
int32_t x12496 = x12495 * x12495;
float* x167 = x5+1163712;
bool x12544 = x12377 > 1;
bool x12600 = x12495 == 1;
bool x12601 = x12600 || true;
bool x12602 = x12601 || x12600;
bool x12612 = x12495 <= 1;
int32_t x12613;
if (x12612) {
x12613 = 1;
} else {
x12613 = x12495;
}
bool x12618 = x12613 > 0;
int32_t x12614 = x12613 * x12613;
float* x82 = x5+1164224;
bool x12662 = x12495 > 1;
bool x12693 = x12613 == 1;
bool x12694 = x12693 || x10905;
bool x12695 = x12613 == x10825;
bool x12696 = x12694 || x12695;
bool x12706 = x12613 <= x10825;
int32_t x12707;
if (x12706) {
x12707 = x10825;
} else {
x12707 = x12613;
}
bool x12745 = x12613 > 1;
int32_t x12708 = x12707 * x12707;
int32_t x12791 = x12613 - 1;
int32_t x12792 = x12791 / 1;
int32_t x12793 = x12792 + 1;
int32_t x12797 = 8192 * x12793;
int32_t x12798 = x12797 * x12793;
int32_t x12794 = x12793 * x12793;
int32_t x12795 = 128 * x12794;
float* x132 = x5+1165760;
bool x12872 = x12793 == 1;
bool x12873 = x12872 || true;
bool x12874 = x12873 || x12872;
bool x12884 = x12793 <= 1;
int32_t x12885;
if (x12884) {
x12885 = 1;
} else {
x12885 = x12793;
}
bool x12890 = x12885 > 0;
int32_t x12886 = x12885 * x12885;
float* x236 = x5+1231552;
bool x12934 = x12793 > 1;
float* x261 = x5+1231680;
bool x13003 = x12885 == 1;
bool x13004 = x13003 || true;
bool x13005 = x13004 || x13003;
bool x13015 = x12885 <= 1;
int32_t x13016;
if (x13015) {
x13016 = 1;
} else {
x13016 = x12885;
}
bool x13021 = x13016 > 0;
int32_t x13017 = x13016 * x13016;
bool x13065 = x12885 > 1;
bool x13121 = x13016 == 1;
bool x13122 = x13121 || true;
bool x13123 = x13122 || x13121;
bool x13133 = x13016 <= 1;
int32_t x13134;
if (x13133) {
x13134 = 1;
} else {
x13134 = x13016;
}
bool x13139 = x13134 > 0;
int32_t x13135 = x13134 * x13134;
float* x39 = x5+1231296;
bool x13183 = x13016 > 1;
bool x13239 = x13134 == 1;
bool x13240 = x13239 || true;
bool x13241 = x13240 || x13239;
bool x13251 = x13134 <= 1;
int32_t x13252;
if (x13251) {
x13252 = 1;
} else {
x13252 = x13134;
}
bool x13257 = x13252 > 0;
int32_t x13253 = x13252 * x13252;
float* x242 = x5+1231424;
bool x13301 = x13134 > 1;
int32_t x13341 = x13252 + 2;
int32_t x13342 = x13341 - 3;
int32_t x13343 = x13342 / 1;
int32_t x13344 = x13343 + 1;
int32_t x13348 = 8192 * x13344;
int32_t x13349 = x13348 * x13344;
int32_t x13345 = x13344 * x13344;
int32_t x13346 = 128 * x13345;
float* x165 = x5+1231808;
bool x13471 = x13344 == 1;
bool x13472 = x13471 || true;
bool x13473 = x13472 || x13471;
bool x13483 = x13344 <= 1;
int32_t x13484;
if (x13483) {
x13484 = 1;
} else {
x13484 = x13344;
}
bool x13489 = x13484 > 0;
int32_t x13485 = x13484 * x13484;
float* x268 = x5+1379520;
bool x13533 = x13344 > 1;
float* x148 = x5+1379648;
bool x13602 = x13484 == 1;
bool x13603 = x13602 || true;
bool x13604 = x13603 || x13602;
bool x13614 = x13484 <= 1;
int32_t x13615;
if (x13614) {
x13615 = 1;
} else {
x13615 = x13484;
}
bool x13620 = x13615 > 0;
int32_t x13616 = x13615 * x13615;
bool x13664 = x13484 > 1;
bool x13720 = x13615 == 1;
bool x13721 = x13720 || true;
bool x13722 = x13721 || x13720;
bool x13732 = x13615 <= 1;
int32_t x13733;
if (x13732) {
x13733 = 1;
} else {
x13733 = x13615;
}
bool x13738 = x13733 > 0;
int32_t x13734 = x13733 * x13733;
float* x79 = x5+1379264;
bool x13782 = x13615 > 1;
bool x13838 = x13733 == 1;
bool x13839 = x13838 || true;
bool x13840 = x13839 || x13838;
bool x13850 = x13733 <= 1;
int32_t x13851;
if (x13850) {
x13851 = 1;
} else {
x13851 = x13733;
}
bool x13856 = x13851 > 0;
int32_t x13852 = x13851 * x13851;
float* x38 = x5+1379392;
bool x13900 = x13733 > 1;
int32_t x13940 = x13851 - 1;
int32_t x13941 = x13940 / 1;
int32_t x13942 = x13941 + 1;
int32_t x13946 = 32768 * x13942;
int32_t x13947 = x13946 * x13942;
int32_t x13943 = x13942 * x13942;
int32_t x13944 = 512 * x13943;
float* x55 = x5+1379776;
bool x14021 = x13942 == 1;
bool x14022 = x14021 || true;
bool x14023 = x14022 || x14021;
bool x14033 = x13942 <= 1;
int32_t x14034;
if (x14033) {
x14034 = 1;
} else {
x14034 = x13942;
}
bool x14039 = x14034 > 0;
int32_t x14035 = x14034 * x14034;
float* x19 = x5+1446336;
bool x14083 = x13942 > 1;
float* x234 = x5+1446848;
bool x14152 = x14034 == 1;
bool x14153 = x14152 || true;
bool x14154 = x14153 || x14152;
bool x14164 = x14034 <= 1;
int32_t x14165;
if (x14164) {
x14165 = 1;
} else {
x14165 = x14034;
}
bool x14170 = x14165 > 0;
int32_t x14166 = x14165 * x14165;
bool x14214 = x14034 > 1;
bool x14270 = x14165 == 1;
bool x14271 = x14270 || true;
bool x14272 = x14271 || x14270;
bool x14282 = x14165 <= 1;
int32_t x14283;
if (x14282) {
x14283 = 1;
} else {
x14283 = x14165;
}
bool x14288 = x14283 > 0;
int32_t x14284 = x14283 * x14283;
float* x156 = x5+1445312;
bool x14332 = x14165 > 1;
bool x14388 = x14283 == 1;
bool x14389 = x14388 || true;
bool x14390 = x14389 || x14388;
bool x14400 = x14283 <= 1;
int32_t x14401;
if (x14400) {
x14401 = 1;
} else {
x14401 = x14283;
}
bool x14406 = x14401 > 0;
int32_t x14402 = x14401 * x14401;
float* x54 = x5+1445824;
bool x14450 = x14283 > 1;
bool x14481 = x14401 == 1;
bool x14482 = x14481 || x12693;
bool x14483 = x14401 == x12613;
bool x14484 = x14482 || x14483;
bool x14494 = x14401 <= x12613;
int32_t x14495;
if (x14494) {
x14495 = x12613;
} else {
x14495 = x14401;
}
bool x14533 = x14401 > 1;
int32_t x14496 = x14495 * x14495;
int32_t x14579 = x14401 - 1;
int32_t x14580 = x14579 / 1;
int32_t x14581 = x14580 + 1;
int32_t x14585 = 16384 * x14581;
int32_t x14586 = x14585 * x14581;
int32_t x14582 = x14581 * x14581;
int32_t x14583 = 256 * x14582;
float* x180 = x5+1447360;
bool x14660 = x14581 == 1;
bool x14661 = x14660 || true;
bool x14662 = x14661 || x14660;
bool x14672 = x14581 <= 1;
int32_t x14673;
if (x14672) {
x14673 = 1;
} else {
x14673 = x14581;
}
bool x14678 = x14673 > 0;
int32_t x14674 = x14673 * x14673;
float* x131 = x5+1578944;
bool x14722 = x14581 > 1;
float* x198 = x5+1579200;
bool x14791 = x14673 == 1;
bool x14792 = x14791 || true;
bool x14793 = x14792 || x14791;
bool x14803 = x14673 <= 1;
int32_t x14804;
if (x14803) {
x14804 = 1;
} else {
x14804 = x14673;
}
bool x14809 = x14804 > 0;
int32_t x14805 = x14804 * x14804;
bool x14853 = x14673 > 1;
bool x14909 = x14804 == 1;
bool x14910 = x14909 || true;
bool x14911 = x14910 || x14909;
bool x14921 = x14804 <= 1;
int32_t x14922;
if (x14921) {
x14922 = 1;
} else {
x14922 = x14804;
}
bool x14927 = x14922 > 0;
int32_t x14923 = x14922 * x14922;
float* x270 = x5+1578432;
bool x14971 = x14804 > 1;
bool x15027 = x14922 == 1;
bool x15028 = x15027 || true;
bool x15029 = x15028 || x15027;
bool x15039 = x14922 <= 1;
int32_t x15040;
if (x15039) {
x15040 = 1;
} else {
x15040 = x14922;
}
bool x15045 = x15040 > 0;
int32_t x15041 = x15040 * x15040;
float* x21 = x5+1578688;
bool x15089 = x14922 > 1;
int32_t x15129 = x15040 + 2;
int32_t x15130 = x15129 - 3;
int32_t x15131 = x15130 / 2;
int32_t x15132 = x15131 + 1;
int32_t x15136 = 16384 * x15132;
int32_t x15137 = x15136 * x15132;
int32_t x15133 = x15132 * x15132;
int32_t x15134 = 256 * x15133;
float* x175 = x5+1579456;
bool x15243 = x15132 == 1;
bool x15244 = x15243 || true;
bool x15245 = x15244 || x15243;
bool x15255 = x15132 <= 1;
int32_t x15256;
if (x15255) {
x15256 = 1;
} else {
x15256 = x15132;
}
bool x15261 = x15256 > 0;
int32_t x15257 = x15256 * x15256;
float* x229 = x5+2169792;
bool x15305 = x15132 > 1;
float* x99 = x5+2170048;
bool x15374 = x15256 == 1;
bool x15375 = x15374 || true;
bool x15376 = x15375 || x15374;
bool x15386 = x15256 <= 1;
int32_t x15387;
if (x15386) {
x15387 = 1;
} else {
x15387 = x15256;
}
bool x15392 = x15387 > 0;
int32_t x15388 = x15387 * x15387;
bool x15436 = x15256 > 1;
bool x15492 = x15387 == 1;
bool x15493 = x15492 || true;
bool x15494 = x15493 || x15492;
bool x15504 = x15387 <= 1;
int32_t x15505;
if (x15504) {
x15505 = 1;
} else {
x15505 = x15387;
}
bool x15510 = x15505 > 0;
int32_t x15506 = x15505 * x15505;
float* x108 = x5+2169280;
bool x15554 = x15387 > 1;
bool x15610 = x15505 == 1;
bool x15611 = x15610 || true;
bool x15612 = x15611 || x15610;
bool x15622 = x15505 <= 1;
int32_t x15623;
if (x15622) {
x15623 = 1;
} else {
x15623 = x15505;
}
bool x15628 = x15623 > 0;
int32_t x15624 = x15623 * x15623;
float* x16 = x5+2169536;
bool x15672 = x15505 > 1;
int32_t x15712 = x15623 - 1;
int32_t x15713 = x15712 / 1;
int32_t x15714 = x15713 + 1;
int32_t x15718 = 65536 * x15714;
int32_t x15719 = x15718 * x15714;
int32_t x15715 = x15714 * x15714;
int32_t x15716 = 1024 * x15715;
float* x269 = x5+2170304;
bool x15793 = x15714 == 1;
bool x15794 = x15793 || true;
bool x15795 = x15794 || x15793;
bool x15805 = x15714 <= 1;
int32_t x15806;
if (x15805) {
x15806 = 1;
} else {
x15806 = x15714;
}
bool x15811 = x15806 > 0;
int32_t x15807 = x15806 * x15806;
float* x216 = x5+2434496;
bool x15855 = x15714 > 1;
float* x267 = x5+2435520;
bool x15925 = x15806 == 1;
bool x15926 = x15925 || true;
bool x15927 = x15926 || x15925;
bool x15937 = x15806 <= 1;
int32_t x15938;
if (x15937) {
x15938 = 1;
} else {
x15938 = x15806;
}
bool x15943 = x15938 > 0;
int32_t x15939 = x15938 * x15938;
bool x15987 = x15806 > 1;
bool x16043 = x15938 == 1;
bool x16044 = x16043 || true;
bool x16045 = x16044 || x16043;
bool x16055 = x15938 <= 1;
int32_t x16056;
if (x16055) {
x16056 = 1;
} else {
x16056 = x15938;
}
bool x16061 = x16056 > 0;
int32_t x16057 = x16056 * x16056;
float* x18 = x5+2432448;
bool x16105 = x15938 > 1;
bool x16161 = x16056 == 1;
bool x16162 = x16161 || true;
bool x16163 = x16162 || x16161;
bool x16173 = x16056 <= 1;
int32_t x16174;
if (x16173) {
x16174 = 1;
} else {
x16174 = x16056;
}
bool x16179 = x16174 > 0;
int32_t x16175 = x16174 * x16174;
float* x117 = x5+2433472;
bool x16223 = x16056 > 1;
int32_t x16250 = x14579 / 2;
int32_t x16251 = x16250 + 1;
int32_t x16255 = 65536 * x16251;
int32_t x16256 = x16255 * x16251;
int32_t x16252 = x16251 * x16251;
int32_t x16253 = 1024 * x16252;
float* x75 = x5+2436544;
bool x16336 = x16251 == 1;
bool x16337 = x16336 || true;
bool x16338 = x16337 || x16336;
bool x16348 = x16251 <= 1;
int32_t x16349;
if (x16348) {
x16349 = 1;
} else {
x16349 = x16251;
}
bool x16354 = x16349 > 0;
int32_t x16350 = x16349 * x16349;
float* x86 = x5+2962880;
bool x16398 = x16251 > 1;
float* x211 = x5+2963904;
bool x16467 = x16349 == 1;
bool x16468 = x16467 || true;
bool x16469 = x16468 || x16467;
bool x16479 = x16349 <= 1;
int32_t x16480;
if (x16479) {
x16480 = 1;
} else {
x16480 = x16349;
}
bool x16485 = x16480 > 0;
int32_t x16481 = x16480 * x16480;
bool x16529 = x16349 > 1;
bool x16585 = x16480 == 1;
bool x16586 = x16585 || true;
bool x16587 = x16586 || x16585;
bool x16597 = x16480 <= 1;
int32_t x16598;
if (x16597) {
x16598 = 1;
} else {
x16598 = x16480;
}
bool x16603 = x16598 > 0;
int32_t x16599 = x16598 * x16598;
float* x29 = x5+2960832;
bool x16647 = x16480 > 1;
bool x16703 = x16598 == 1;
bool x16704 = x16703 || true;
bool x16705 = x16704 || x16703;
bool x16715 = x16598 <= 1;
int32_t x16716;
if (x16715) {
x16716 = 1;
} else {
x16716 = x16598;
}
bool x16721 = x16716 > 0;
int32_t x16717 = x16716 * x16716;
float* x220 = x5+2961856;
bool x16765 = x16598 > 1;
bool x16797 = x16174 == 1;
bool x16798 = x16716 == 1;
bool x16799 = x16797 || x16798;
bool x16800 = x16174 == x16716;
bool x16801 = x16799 || x16800;
bool x16811 = x16174 <= x16716;
int32_t x16812;
if (x16811) {
x16812 = x16716;
} else {
x16812 = x16174;
}
bool x16850 = x16174 > 1;
bool x16854 = x16716 > 1;
int32_t x16813 = x16812 * x16812;
int32_t x16898 = x16174 - 1;
int32_t x16899 = x16898 / 1;
int32_t x16900 = x16899 + 1;
int32_t x16904 = 16384 * x16900;
int32_t x16905 = x16904 * x16900;
int32_t x16901 = x16900 * x16900;
int32_t x16902 = 256 * x16901;
float* x13 = x5+2964928;
bool x16979 = x16900 == 1;
bool x16980 = x16979 || true;
bool x16981 = x16980 || x16979;
bool x16991 = x16900 <= 1;
int32_t x16992;
if (x16991) {
x16992 = 1;
} else {
x16992 = x16900;
}
bool x16997 = x16992 > 0;
int32_t x16993 = x16992 * x16992;
float* x259 = x5+3227584;
bool x17041 = x16900 > 1;
float* x157 = x5+3227840;
bool x17110 = x16992 == 1;
bool x17111 = x17110 || true;
bool x17112 = x17111 || x17110;
bool x17122 = x16992 <= 1;
int32_t x17123;
if (x17122) {
x17123 = 1;
} else {
x17123 = x16992;
}
bool x17128 = x17123 > 0;
int32_t x17124 = x17123 * x17123;
bool x17172 = x16992 > 1;
bool x17228 = x17123 == 1;
bool x17229 = x17228 || true;
bool x17230 = x17229 || x17228;
bool x17240 = x17123 <= 1;
int32_t x17241;
if (x17240) {
x17241 = 1;
} else {
x17241 = x17123;
}
bool x17246 = x17241 > 0;
int32_t x17242 = x17241 * x17241;
float* x30 = x5+3227072;
bool x17290 = x17123 > 1;
bool x17346 = x17241 == 1;
bool x17347 = x17346 || true;
bool x17348 = x17347 || x17346;
bool x17358 = x17241 <= 1;
int32_t x17359;
if (x17358) {
x17359 = 1;
} else {
x17359 = x17241;
}
bool x17364 = x17359 > 0;
int32_t x17360 = x17359 * x17359;
float* x219 = x5+3227328;
bool x17408 = x17241 > 1;
int32_t x17448 = x17359 + 2;
int32_t x17449 = x17448 - 3;
int32_t x17450 = x17449 / 1;
int32_t x17451 = x17450 + 1;
int32_t x17455 = 16384 * x17451;
int32_t x17456 = x17455 * x17451;
int32_t x17452 = x17451 * x17451;
int32_t x17453 = 256 * x17452;
float* x31 = x5+3228096;
bool x17578 = x17451 == 1;
bool x17579 = x17578 || true;
bool x17580 = x17579 || x17578;
bool x17590 = x17451 <= 1;
int32_t x17591;
if (x17590) {
x17591 = 1;
} else {
x17591 = x17451;
}
bool x17596 = x17591 > 0;
int32_t x17592 = x17591 * x17591;
float* x200 = x5+3818432;
bool x17640 = x17451 > 1;
float* x237 = x5+3818688;
bool x17709 = x17591 == 1;
bool x17710 = x17709 || true;
bool x17711 = x17710 || x17709;
bool x17721 = x17591 <= 1;
int32_t x17722;
if (x17721) {
x17722 = 1;
} else {
x17722 = x17591;
}
bool x17727 = x17722 > 0;
int32_t x17723 = x17722 * x17722;
bool x17771 = x17591 > 1;
bool x17827 = x17722 == 1;
bool x17828 = x17827 || true;
bool x17829 = x17828 || x17827;
bool x17839 = x17722 <= 1;
int32_t x17840;
if (x17839) {
x17840 = 1;
} else {
x17840 = x17722;
}
bool x17845 = x17840 > 0;
int32_t x17841 = x17840 * x17840;
float* x271 = x5+3817920;
bool x17889 = x17722 > 1;
bool x17945 = x17840 == 1;
bool x17946 = x17945 || true;
bool x17947 = x17946 || x17945;
bool x17957 = x17840 <= 1;
int32_t x17958;
if (x17957) {
x17958 = 1;
} else {
x17958 = x17840;
}
bool x17963 = x17958 > 0;
int32_t x17959 = x17958 * x17958;
float* x96 = x5+3818176;
bool x18007 = x17840 > 1;
int32_t x18047 = x17958 - 1;
int32_t x18048 = x18047 / 1;
int32_t x18049 = x18048 + 1;
int32_t x18053 = 65536 * x18049;
int32_t x18054 = x18053 * x18049;
int32_t x18050 = x18049 * x18049;
int32_t x18051 = 1024 * x18050;
float* x56 = x5+3818944;
bool x18128 = x18049 == 1;
bool x18129 = x18128 || true;
bool x18130 = x18129 || x18128;
bool x18140 = x18049 <= 1;
int32_t x18141;
if (x18140) {
x18141 = 1;
} else {
x18141 = x18049;
}
bool x18146 = x18141 > 0;
int32_t x18142 = x18141 * x18141;
float* x182 = x5+4083136;
bool x18190 = x18049 > 1;
float* x143 = x5+4084160;
bool x18259 = x18141 == 1;
bool x18260 = x18259 || true;
bool x18261 = x18260 || x18259;
bool x18271 = x18141 <= 1;
int32_t x18272;
if (x18271) {
x18272 = 1;
} else {
x18272 = x18141;
}
bool x18277 = x18272 > 0;
int32_t x18273 = x18272 * x18272;
bool x18321 = x18141 > 1;
bool x18377 = x18272 == 1;
bool x18378 = x18377 || true;
bool x18379 = x18378 || x18377;
bool x18389 = x18272 <= 1;
int32_t x18390;
if (x18389) {
x18390 = 1;
} else {
x18390 = x18272;
}
bool x18395 = x18390 > 0;
int32_t x18391 = x18390 * x18390;
float* x20 = x5+4081088;
bool x18439 = x18272 > 1;
bool x18495 = x18390 == 1;
bool x18496 = x18495 || true;
bool x18497 = x18496 || x18495;
bool x18507 = x18390 <= 1;
int32_t x18508;
if (x18507) {
x18508 = 1;
} else {
x18508 = x18390;
}
bool x18513 = x18508 > 0;
int32_t x18509 = x18508 * x18508;
float* x232 = x5+4082112;
bool x18557 = x18390 > 1;
bool x18588 = x18508 == 1;
bool x18589 = x18588 || x16797;
bool x18590 = x18508 == x16174;
bool x18591 = x18589 || x18590;
bool x18601 = x18508 <= x16174;
int32_t x18602;
if (x18601) {
x18602 = x16174;
} else {
x18602 = x18508;
}
bool x18640 = x18508 > 1;
int32_t x18603 = x18602 * x18602;
int32_t x18686 = x18508 - 1;
int32_t x18687 = x18686 / 1;
int32_t x18688 = x18687 + 1;
int32_t x18692 = 16384 * x18688;
int32_t x18693 = x18692 * x18688;
int32_t x18689 = x18688 * x18688;
int32_t x18690 = 256 * x18689;
float* x218 = x5+4085184;
bool x18767 = x18688 == 1;
bool x18768 = x18767 || true;
bool x18769 = x18768 || x18767;
bool x18779 = x18688 <= 1;
int32_t x18780;
if (x18779) {
x18780 = 1;
} else {
x18780 = x18688;
}
bool x18785 = x18780 > 0;
int32_t x18781 = x18780 * x18780;
float* x178 = x5+4347840;
bool x18829 = x18688 > 1;
float* x174 = x5+4348096;
bool x18898 = x18780 == 1;
bool x18899 = x18898 || true;
bool x18900 = x18899 || x18898;
bool x18910 = x18780 <= 1;
int32_t x18911;
if (x18910) {
x18911 = 1;
} else {
x18911 = x18780;
}
bool x18916 = x18911 > 0;
int32_t x18912 = x18911 * x18911;
bool x18960 = x18780 > 1;
bool x19016 = x18911 == 1;
bool x19017 = x19016 || true;
bool x19018 = x19017 || x19016;
bool x19028 = x18911 <= 1;
int32_t x19029;
if (x19028) {
x19029 = 1;
} else {
x19029 = x18911;
}
bool x19034 = x19029 > 0;
int32_t x19030 = x19029 * x19029;
float* x129 = x5+4347328;
bool x19078 = x18911 > 1;
bool x19134 = x19029 == 1;
bool x19135 = x19134 || true;
bool x19136 = x19135 || x19134;
bool x19146 = x19029 <= 1;
int32_t x19147;
if (x19146) {
x19147 = 1;
} else {
x19147 = x19029;
}
bool x19152 = x19147 > 0;
int32_t x19148 = x19147 * x19147;
float* x197 = x5+4347584;
bool x19196 = x19029 > 1;
int32_t x19236 = x19147 + 2;
int32_t x19237 = x19236 - 3;
int32_t x19238 = x19237 / 1;
int32_t x19239 = x19238 + 1;
int32_t x19243 = 16384 * x19239;
int32_t x19244 = x19243 * x19239;
int32_t x19240 = x19239 * x19239;
int32_t x19241 = 256 * x19240;
float* x14 = x5+4348352;
bool x19366 = x19239 == 1;
bool x19367 = x19366 || true;
bool x19368 = x19367 || x19366;
bool x19378 = x19239 <= 1;
int32_t x19379;
if (x19378) {
x19379 = 1;
} else {
x19379 = x19239;
}
bool x19384 = x19379 > 0;
int32_t x19380 = x19379 * x19379;
float* x124 = x5+4938688;
bool x19428 = x19239 > 1;
float* x63 = x5+4938944;
bool x19497 = x19379 == 1;
bool x19498 = x19497 || true;
bool x19499 = x19498 || x19497;
bool x19509 = x19379 <= 1;
int32_t x19510;
if (x19509) {
x19510 = 1;
} else {
x19510 = x19379;
}
bool x19515 = x19510 > 0;
int32_t x19511 = x19510 * x19510;
bool x19559 = x19379 > 1;
bool x19615 = x19510 == 1;
bool x19616 = x19615 || true;
bool x19617 = x19616 || x19615;
bool x19627 = x19510 <= 1;
int32_t x19628;
if (x19627) {
x19628 = 1;
} else {
x19628 = x19510;
}
bool x19633 = x19628 > 0;
int32_t x19629 = x19628 * x19628;
float* x228 = x5+4938176;
bool x19677 = x19510 > 1;
bool x19733 = x19628 == 1;
bool x19734 = x19733 || true;
bool x19735 = x19734 || x19733;
bool x19745 = x19628 <= 1;
int32_t x19746;
if (x19745) {
x19746 = 1;
} else {
x19746 = x19628;
}
bool x19751 = x19746 > 0;
int32_t x19747 = x19746 * x19746;
float* x192 = x5+4938432;
bool x19795 = x19628 > 1;
int32_t x19835 = x19746 - 1;
int32_t x19836 = x19835 / 1;
int32_t x19837 = x19836 + 1;
int32_t x19841 = 65536 * x19837;
int32_t x19842 = x19841 * x19837;
int32_t x19838 = x19837 * x19837;
int32_t x19839 = 1024 * x19838;
float* x116 = x5+4939200;
bool x19916 = x19837 == 1;
bool x19917 = x19916 || true;
bool x19918 = x19917 || x19916;
bool x19928 = x19837 <= 1;
int32_t x19929;
if (x19928) {
x19929 = 1;
} else {
x19929 = x19837;
}
bool x19934 = x19929 > 0;
int32_t x19930 = x19929 * x19929;
float* x140 = x5+5203392;
bool x19978 = x19837 > 1;
float* x188 = x5+5204416;
bool x20047 = x19929 == 1;
bool x20048 = x20047 || true;
bool x20049 = x20048 || x20047;
bool x20059 = x19929 <= 1;
int32_t x20060;
if (x20059) {
x20060 = 1;
} else {
x20060 = x19929;
}
bool x20065 = x20060 > 0;
int32_t x20061 = x20060 * x20060;
bool x20109 = x19929 > 1;
bool x20165 = x20060 == 1;
bool x20166 = x20165 || true;
bool x20167 = x20166 || x20165;
bool x20177 = x20060 <= 1;
int32_t x20178;
if (x20177) {
x20178 = 1;
} else {
x20178 = x20060;
}
bool x20183 = x20178 > 0;
int32_t x20179 = x20178 * x20178;
float* x263 = x5+5201344;
bool x20227 = x20060 > 1;
bool x20283 = x20178 == 1;
bool x20284 = x20283 || true;
bool x20285 = x20284 || x20283;
bool x20295 = x20178 <= 1;
int32_t x20296;
if (x20295) {
x20296 = 1;
} else {
x20296 = x20178;
}
bool x20301 = x20296 > 0;
int32_t x20297 = x20296 * x20296;
float* x57 = x5+5202368;
bool x20345 = x20178 > 1;
bool x20376 = x20296 == 1;
bool x20377 = x20376 || x18588;
bool x20378 = x20296 == x18508;
bool x20379 = x20377 || x20378;
bool x20389 = x20296 <= x18508;
int32_t x20390;
if (x20389) {
x20390 = x18508;
} else {
x20390 = x20296;
}
bool x20428 = x20296 > 1;
int32_t x20391 = x20390 * x20390;
int32_t x20474 = x20296 - 1;
int32_t x20475 = x20474 / 1;
int32_t x20476 = x20475 + 1;
int32_t x20480 = 16384 * x20476;
int32_t x20481 = x20480 * x20476;
int32_t x20477 = x20476 * x20476;
int32_t x20478 = 256 * x20477;
float* x6 = x5+5205440;
bool x20555 = x20476 == 1;
bool x20556 = x20555 || true;
bool x20557 = x20556 || x20555;
bool x20567 = x20476 <= 1;
int32_t x20568;
if (x20567) {
x20568 = 1;
} else {
x20568 = x20476;
}
bool x20573 = x20568 > 0;
int32_t x20569 = x20568 * x20568;
float* x163 = x5+5468096;
bool x20617 = x20476 > 1;
float* x98 = x5+5468352;
bool x20686 = x20568 == 1;
bool x20687 = x20686 || true;
bool x20688 = x20687 || x20686;
bool x20698 = x20568 <= 1;
int32_t x20699;
if (x20698) {
x20699 = 1;
} else {
x20699 = x20568;
}
bool x20704 = x20699 > 0;
int32_t x20700 = x20699 * x20699;
bool x20748 = x20568 > 1;
bool x20804 = x20699 == 1;
bool x20805 = x20804 || true;
bool x20806 = x20805 || x20804;
bool x20816 = x20699 <= 1;
int32_t x20817;
if (x20816) {
x20817 = 1;
} else {
x20817 = x20699;
}
bool x20822 = x20817 > 0;
int32_t x20818 = x20817 * x20817;
float* x92 = x5+5467584;
bool x20866 = x20699 > 1;
bool x20922 = x20817 == 1;
bool x20923 = x20922 || true;
bool x20924 = x20923 || x20922;
bool x20934 = x20817 <= 1;
int32_t x20935;
if (x20934) {
x20935 = 1;
} else {
x20935 = x20817;
}
bool x20940 = x20935 > 0;
int32_t x20936 = x20935 * x20935;
float* x241 = x5+5467840;
bool x20984 = x20817 > 1;
int32_t x21024 = x20935 + 2;
int32_t x21025 = x21024 - 3;
int32_t x21026 = x21025 / 1;
int32_t x21027 = x21026 + 1;
int32_t x21031 = 16384 * x21027;
int32_t x21032 = x21031 * x21027;
int32_t x21028 = x21027 * x21027;
int32_t x21029 = 256 * x21028;
float* x249 = x5+5468608;
bool x21154 = x21027 == 1;
bool x21155 = x21154 || true;
bool x21156 = x21155 || x21154;
bool x21166 = x21027 <= 1;
int32_t x21167;
if (x21166) {
x21167 = 1;
} else {
x21167 = x21027;
}
bool x21172 = x21167 > 0;
int32_t x21168 = x21167 * x21167;
float* x186 = x5+6058944;
bool x21216 = x21027 > 1;
float* x230 = x5+6059200;
bool x21285 = x21167 == 1;
bool x21286 = x21285 || true;
bool x21287 = x21286 || x21285;
bool x21297 = x21167 <= 1;
int32_t x21298;
if (x21297) {
x21298 = 1;
} else {
x21298 = x21167;
}
bool x21303 = x21298 > 0;
int32_t x21299 = x21298 * x21298;
bool x21347 = x21167 > 1;
bool x21403 = x21298 == 1;
bool x21404 = x21403 || true;
bool x21405 = x21404 || x21403;
bool x21415 = x21298 <= 1;
int32_t x21416;
if (x21415) {
x21416 = 1;
} else {
x21416 = x21298;
}
bool x21421 = x21416 > 0;
int32_t x21417 = x21416 * x21416;
float* x74 = x5+6058432;
bool x21465 = x21298 > 1;
bool x21521 = x21416 == 1;
bool x21522 = x21521 || true;
bool x21523 = x21522 || x21521;
bool x21533 = x21416 <= 1;
int32_t x21534;
if (x21533) {
x21534 = 1;
} else {
x21534 = x21416;
}
bool x21539 = x21534 > 0;
int32_t x21535 = x21534 * x21534;
float* x136 = x5+6058688;
bool x21583 = x21416 > 1;
int32_t x21623 = x21534 - 1;
int32_t x21624 = x21623 / 1;
int32_t x21625 = x21624 + 1;
int32_t x21629 = 65536 * x21625;
int32_t x21630 = x21629 * x21625;
int32_t x21626 = x21625 * x21625;
int32_t x21627 = 1024 * x21626;
float* x89 = x5+6059456;
bool x21704 = x21625 == 1;
bool x21705 = x21704 || true;
bool x21706 = x21705 || x21704;
bool x21716 = x21625 <= 1;
int32_t x21717;
if (x21716) {
x21717 = 1;
} else {
x21717 = x21625;
}
bool x21722 = x21717 > 0;
int32_t x21718 = x21717 * x21717;
float* x231 = x5+6323648;
bool x21766 = x21625 > 1;
float* x161 = x5+6324672;
bool x21835 = x21717 == 1;
bool x21836 = x21835 || true;
bool x21837 = x21836 || x21835;
bool x21847 = x21717 <= 1;
int32_t x21848;
if (x21847) {
x21848 = 1;
} else {
x21848 = x21717;
}
bool x21853 = x21848 > 0;
int32_t x21849 = x21848 * x21848;
bool x21897 = x21717 > 1;
bool x21953 = x21848 == 1;
bool x21954 = x21953 || true;
bool x21955 = x21954 || x21953;
bool x21965 = x21848 <= 1;
int32_t x21966;
if (x21965) {
x21966 = 1;
} else {
x21966 = x21848;
}
bool x21971 = x21966 > 0;
int32_t x21967 = x21966 * x21966;
float* x238 = x5+6321600;
bool x22015 = x21848 > 1;
bool x22071 = x21966 == 1;
bool x22072 = x22071 || true;
bool x22073 = x22072 || x22071;
bool x22083 = x21966 <= 1;
int32_t x22084;
if (x22083) {
x22084 = 1;
} else {
x22084 = x21966;
}
bool x22089 = x22084 > 0;
int32_t x22085 = x22084 * x22084;
float* x146 = x5+6322624;
bool x22133 = x21966 > 1;
bool x22164 = x22084 == 1;
bool x22165 = x22164 || x20376;
bool x22166 = x22084 == x20296;
bool x22167 = x22165 || x22166;
bool x22177 = x22084 <= x20296;
int32_t x22178;
if (x22177) {
x22178 = x20296;
} else {
x22178 = x22084;
}
bool x22216 = x22084 > 1;
int32_t x22179 = x22178 * x22178;
int32_t x22262 = x22084 - 1;
int32_t x22263 = x22262 / 1;
int32_t x22264 = x22263 + 1;
int32_t x22268 = 16384 * x22264;
int32_t x22269 = x22268 * x22264;
int32_t x22265 = x22264 * x22264;
int32_t x22266 = 256 * x22265;
float* x22 = x5+6325696;
bool x22343 = x22264 == 1;
bool x22344 = x22343 || true;
bool x22345 = x22344 || x22343;
bool x22355 = x22264 <= 1;
int32_t x22356;
if (x22355) {
x22356 = 1;
} else {
x22356 = x22264;
}
bool x22361 = x22356 > 0;
int32_t x22357 = x22356 * x22356;
float* x254 = x5+6588352;
bool x22405 = x22264 > 1;
float* x69 = x5+6588608;
bool x22474 = x22356 == 1;
bool x22475 = x22474 || true;
bool x22476 = x22475 || x22474;
bool x22486 = x22356 <= 1;
int32_t x22487;
if (x22486) {
x22487 = 1;
} else {
x22487 = x22356;
}
bool x22492 = x22487 > 0;
int32_t x22488 = x22487 * x22487;
bool x22536 = x22356 > 1;
bool x22592 = x22487 == 1;
bool x22593 = x22592 || true;
bool x22594 = x22593 || x22592;
bool x22604 = x22487 <= 1;
int32_t x22605;
if (x22604) {
x22605 = 1;
} else {
x22605 = x22487;
}
bool x22610 = x22605 > 0;
int32_t x22606 = x22605 * x22605;
float* x77 = x5+6587840;
bool x22654 = x22487 > 1;
bool x22710 = x22605 == 1;
bool x22711 = x22710 || true;
bool x22712 = x22711 || x22710;
bool x22722 = x22605 <= 1;
int32_t x22723;
if (x22722) {
x22723 = 1;
} else {
x22723 = x22605;
}
bool x22728 = x22723 > 0;
int32_t x22724 = x22723 * x22723;
float* x185 = x5+6588096;
bool x22772 = x22605 > 1;
int32_t x22812 = x22723 + 2;
int32_t x22813 = x22812 - 3;
int32_t x22814 = x22813 / 1;
int32_t x22815 = x22814 + 1;
int32_t x22819 = 16384 * x22815;
int32_t x22820 = x22819 * x22815;
int32_t x22816 = x22815 * x22815;
int32_t x22817 = 256 * x22816;
float* x262 = x5+6588864;
bool x22942 = x22815 == 1;
bool x22943 = x22942 || true;
bool x22944 = x22943 || x22942;
bool x22954 = x22815 <= 1;
int32_t x22955;
if (x22954) {
x22955 = 1;
} else {
x22955 = x22815;
}
bool x22960 = x22955 > 0;
int32_t x22956 = x22955 * x22955;
float* x250 = x5+7179200;
bool x23004 = x22815 > 1;
float* x104 = x5+7179456;
bool x23073 = x22955 == 1;
bool x23074 = x23073 || true;
bool x23075 = x23074 || x23073;
bool x23085 = x22955 <= 1;
int32_t x23086;
if (x23085) {
x23086 = 1;
} else {
x23086 = x22955;
}
bool x23091 = x23086 > 0;
int32_t x23087 = x23086 * x23086;
bool x23135 = x22955 > 1;
bool x23191 = x23086 == 1;
bool x23192 = x23191 || true;
bool x23193 = x23192 || x23191;
bool x23203 = x23086 <= 1;
int32_t x23204;
if (x23203) {
x23204 = 1;
} else {
x23204 = x23086;
}
bool x23209 = x23204 > 0;
int32_t x23205 = x23204 * x23204;
float* x168 = x5+7178688;
bool x23253 = x23086 > 1;
bool x23309 = x23204 == 1;
bool x23310 = x23309 || true;
bool x23311 = x23310 || x23309;
bool x23321 = x23204 <= 1;
int32_t x23322;
if (x23321) {
x23322 = 1;
} else {
x23322 = x23204;
}
bool x23327 = x23322 > 0;
int32_t x23323 = x23322 * x23322;
float* x109 = x5+7178944;
bool x23371 = x23204 > 1;
int32_t x23411 = x23322 - 1;
int32_t x23412 = x23411 / 1;
int32_t x23413 = x23412 + 1;
int32_t x23417 = 65536 * x23413;
int32_t x23418 = x23417 * x23413;
int32_t x23414 = x23413 * x23413;
int32_t x23415 = 1024 * x23414;
float* x221 = x5+7179712;
bool x23492 = x23413 == 1;
bool x23493 = x23492 || true;
bool x23494 = x23493 || x23492;
bool x23504 = x23413 <= 1;
int32_t x23505;
if (x23504) {
x23505 = 1;
} else {
x23505 = x23413;
}
bool x23510 = x23505 > 0;
int32_t x23506 = x23505 * x23505;
float* x209 = x5+7443904;
bool x23554 = x23413 > 1;
float* x272 = x5+7444928;
bool x23623 = x23505 == 1;
bool x23624 = x23623 || true;
bool x23625 = x23624 || x23623;
bool x23635 = x23505 <= 1;
int32_t x23636;
if (x23635) {
x23636 = 1;
} else {
x23636 = x23505;
}
bool x23641 = x23636 > 0;
int32_t x23637 = x23636 * x23636;
bool x23685 = x23505 > 1;
bool x23741 = x23636 == 1;
bool x23742 = x23741 || true;
bool x23743 = x23742 || x23741;
bool x23753 = x23636 <= 1;
int32_t x23754;
if (x23753) {
x23754 = 1;
} else {
x23754 = x23636;
}
bool x23759 = x23754 > 0;
int32_t x23755 = x23754 * x23754;
float* x59 = x5+7441856;
bool x23803 = x23636 > 1;
bool x23859 = x23754 == 1;
bool x23860 = x23859 || true;
bool x23861 = x23860 || x23859;
bool x23871 = x23754 <= 1;
int32_t x23872;
if (x23871) {
x23872 = 1;
} else {
x23872 = x23754;
}
bool x23877 = x23872 > 0;
int32_t x23873 = x23872 * x23872;
float* x120 = x5+7442880;
bool x23921 = x23754 > 1;
bool x23952 = x23872 == 1;
bool x23953 = x23952 || x22164;
bool x23954 = x23872 == x22084;
bool x23955 = x23953 || x23954;
bool x23965 = x23872 <= x22084;
int32_t x23966;
if (x23965) {
x23966 = x22084;
} else {
x23966 = x23872;
}
bool x24004 = x23872 > 1;
int32_t x23967 = x23966 * x23966;
int32_t x24050 = x23872 - 1;
int32_t x24051 = x24050 / 1;
int32_t x24052 = x24051 + 1;
int32_t x24056 = 16384 * x24052;
int32_t x24057 = x24056 * x24052;
int32_t x24053 = x24052 * x24052;
int32_t x24054 = 256 * x24053;
float* x151 = x5+7445952;
bool x24131 = x24052 == 1;
bool x24132 = x24131 || true;
bool x24133 = x24132 || x24131;
bool x24143 = x24052 <= 1;
int32_t x24144;
if (x24143) {
x24144 = 1;
} else {
x24144 = x24052;
}
bool x24149 = x24144 > 0;
int32_t x24145 = x24144 * x24144;
float* x80 = x5+7708608;
bool x24193 = x24052 > 1;
float* x176 = x5+7708864;
bool x24262 = x24144 == 1;
bool x24263 = x24262 || true;
bool x24264 = x24263 || x24262;
bool x24274 = x24144 <= 1;
int32_t x24275;
if (x24274) {
x24275 = 1;
} else {
x24275 = x24144;
}
bool x24280 = x24275 > 0;
int32_t x24276 = x24275 * x24275;
bool x24324 = x24144 > 1;
bool x24380 = x24275 == 1;
bool x24381 = x24380 || true;
bool x24382 = x24381 || x24380;
bool x24392 = x24275 <= 1;
int32_t x24393;
if (x24392) {
x24393 = 1;
} else {
x24393 = x24275;
}
bool x24398 = x24393 > 0;
int32_t x24394 = x24393 * x24393;
float* x85 = x5+7708096;
bool x24442 = x24275 > 1;
bool x24498 = x24393 == 1;
bool x24499 = x24498 || true;
bool x24500 = x24499 || x24498;
bool x24510 = x24393 <= 1;
int32_t x24511;
if (x24510) {
x24511 = 1;
} else {
x24511 = x24393;
}
bool x24516 = x24511 > 0;
int32_t x24512 = x24511 * x24511;
float* x253 = x5+7708352;
bool x24560 = x24393 > 1;
int32_t x24600 = x24511 + 2;
int32_t x24601 = x24600 - 3;
int32_t x24602 = x24601 / 1;
int32_t x24603 = x24602 + 1;
int32_t x24607 = 16384 * x24603;
int32_t x24608 = x24607 * x24603;
int32_t x24604 = x24603 * x24603;
int32_t x24605 = 256 * x24604;
float* x226 = x5+7709120;
bool x24730 = x24603 == 1;
bool x24731 = x24730 || true;
bool x24732 = x24731 || x24730;
bool x24742 = x24603 <= 1;
int32_t x24743;
if (x24742) {
x24743 = 1;
} else {
x24743 = x24603;
}
bool x24748 = x24743 > 0;
int32_t x24744 = x24743 * x24743;
float* x70 = x5+8299456;
bool x24792 = x24603 > 1;
float* x240 = x5+8299712;
bool x24861 = x24743 == 1;
bool x24862 = x24861 || true;
bool x24863 = x24862 || x24861;
bool x24873 = x24743 <= 1;
int32_t x24874;
if (x24873) {
x24874 = 1;
} else {
x24874 = x24743;
}
bool x24879 = x24874 > 0;
int32_t x24875 = x24874 * x24874;
bool x24923 = x24743 > 1;
bool x24979 = x24874 == 1;
bool x24980 = x24979 || true;
bool x24981 = x24980 || x24979;
bool x24991 = x24874 <= 1;
int32_t x24992;
if (x24991) {
x24992 = 1;
} else {
x24992 = x24874;
}
bool x24997 = x24992 > 0;
int32_t x24993 = x24992 * x24992;
float* x141 = x5+8298944;
bool x25041 = x24874 > 1;
bool x25097 = x24992 == 1;
bool x25098 = x25097 || true;
bool x25099 = x25098 || x25097;
bool x25109 = x24992 <= 1;
int32_t x25110;
if (x25109) {
x25110 = 1;
} else {
x25110 = x24992;
}
bool x25115 = x25110 > 0;
int32_t x25111 = x25110 * x25110;
float* x189 = x5+8299200;
bool x25159 = x24992 > 1;
int32_t x25199 = x25110 - 1;
int32_t x25200 = x25199 / 1;
int32_t x25201 = x25200 + 1;
int32_t x25205 = 65536 * x25201;
int32_t x25206 = x25205 * x25201;
int32_t x25202 = x25201 * x25201;
int32_t x25203 = 1024 * x25202;
float* x97 = x5+8299968;
bool x25280 = x25201 == 1;
bool x25281 = x25280 || true;
bool x25282 = x25281 || x25280;
bool x25292 = x25201 <= 1;
int32_t x25293;
if (x25292) {
x25293 = 1;
} else {
x25293 = x25201;
}
bool x25298 = x25293 > 0;
int32_t x25294 = x25293 * x25293;
float* x122 = x5+8564160;
bool x25342 = x25201 > 1;
float* x183 = x5+8565184;
bool x25411 = x25293 == 1;
bool x25412 = x25411 || true;
bool x25413 = x25412 || x25411;
bool x25423 = x25293 <= 1;
int32_t x25424;
if (x25423) {
x25424 = 1;
} else {
x25424 = x25293;
}
bool x25429 = x25424 > 0;
int32_t x25425 = x25424 * x25424;
bool x25473 = x25293 > 1;
bool x25529 = x25424 == 1;
bool x25530 = x25529 || true;
bool x25531 = x25530 || x25529;
bool x25541 = x25424 <= 1;
int32_t x25542;
if (x25541) {
x25542 = 1;
} else {
x25542 = x25424;
}
bool x25547 = x25542 > 0;
int32_t x25543 = x25542 * x25542;
float* x248 = x5+8562112;
bool x25591 = x25424 > 1;
bool x25647 = x25542 == 1;
bool x25648 = x25647 || true;
bool x25649 = x25648 || x25647;
bool x25659 = x25542 <= 1;
int32_t x25660;
if (x25659) {
x25660 = 1;
} else {
x25660 = x25542;
}
bool x25665 = x25660 > 0;
int32_t x25661 = x25660 * x25660;
float* x93 = x5+8563136;
bool x25709 = x25542 > 1;
bool x25740 = x25660 == 1;
bool x25741 = x25740 || x23952;
bool x25742 = x25660 == x23872;
bool x25743 = x25741 || x25742;
bool x25753 = x25660 <= x23872;
int32_t x25754;
if (x25753) {
x25754 = x23872;
} else {
x25754 = x25660;
}
bool x25792 = x25660 > 1;
int32_t x25755 = x25754 * x25754;
int32_t x25838 = x25660 - 1;
int32_t x25839 = x25838 / 1;
int32_t x25840 = x25839 + 1;
int32_t x25844 = 32768 * x25840;
int32_t x25845 = x25844 * x25840;
int32_t x25841 = x25840 * x25840;
int32_t x25842 = 512 * x25841;
float* x139 = x5+8566208;
bool x25919 = x25840 == 1;
bool x25920 = x25919 || true;
bool x25921 = x25920 || x25919;
bool x25931 = x25840 <= 1;
int32_t x25932;
if (x25931) {
x25932 = 1;
} else {
x25932 = x25840;
}
bool x25937 = x25932 > 0;
int32_t x25933 = x25932 * x25932;
float* x67 = x5+9091520;
bool x25981 = x25840 > 1;
float* x121 = x5+9092032;
bool x26050 = x25932 == 1;
bool x26051 = x26050 || true;
bool x26052 = x26051 || x26050;
bool x26062 = x25932 <= 1;
int32_t x26063;
if (x26062) {
x26063 = 1;
} else {
x26063 = x25932;
}
bool x26068 = x26063 > 0;
int32_t x26064 = x26063 * x26063;
bool x26112 = x25932 > 1;
bool x26168 = x26063 == 1;
bool x26169 = x26168 || true;
bool x26170 = x26169 || x26168;
bool x26180 = x26063 <= 1;
int32_t x26181;
if (x26180) {
x26181 = 1;
} else {
x26181 = x26063;
}
bool x26186 = x26181 > 0;
int32_t x26182 = x26181 * x26181;
float* x201 = x5+9090496;
bool x26230 = x26063 > 1;
bool x26286 = x26181 == 1;
bool x26287 = x26286 || true;
bool x26288 = x26287 || x26286;
bool x26298 = x26181 <= 1;
int32_t x26299;
if (x26298) {
x26299 = 1;
} else {
x26299 = x26181;
}
bool x26304 = x26299 > 0;
int32_t x26300 = x26299 * x26299;
float* x224 = x5+9091008;
bool x26348 = x26181 > 1;
int32_t x26388 = x26299 + 2;
int32_t x26389 = x26388 - 3;
int32_t x26390 = x26389 / 2;
int32_t x26391 = x26390 + 1;
int32_t x26395 = 32768 * x26391;
int32_t x26396 = x26395 * x26391;
int32_t x26392 = x26391 * x26391;
int32_t x26393 = 512 * x26392;
float* x34 = x5+9092544;
bool x26502 = x26391 == 1;
bool x26503 = x26502 || true;
bool x26504 = x26503 || x26502;
bool x26514 = x26391 <= 1;
int32_t x26515;
if (x26514) {
x26515 = 1;
} else {
x26515 = x26391;
}
bool x26520 = x26515 > 0;
int32_t x26516 = x26515 * x26515;
float* x113 = x5+11452864;
bool x26564 = x26391 > 1;
float* x50 = x5+11453376;
bool x26633 = x26515 == 1;
bool x26634 = x26633 || true;
bool x26635 = x26634 || x26633;
bool x26645 = x26515 <= 1;
int32_t x26646;
if (x26645) {
x26646 = 1;
} else {
x26646 = x26515;
}
bool x26651 = x26646 > 0;
int32_t x26647 = x26646 * x26646;
bool x26695 = x26515 > 1;
bool x26751 = x26646 == 1;
bool x26752 = x26751 || true;
bool x26753 = x26752 || x26751;
bool x26763 = x26646 <= 1;
int32_t x26764;
if (x26763) {
x26764 = 1;
} else {
x26764 = x26646;
}
bool x26769 = x26764 > 0;
int32_t x26765 = x26764 * x26764;
float* x205 = x5+11451840;
bool x26813 = x26646 > 1;
bool x26869 = x26764 == 1;
bool x26870 = x26869 || true;
bool x26871 = x26870 || x26869;
bool x26881 = x26764 <= 1;
int32_t x26882;
if (x26881) {
x26882 = 1;
} else {
x26882 = x26764;
}
bool x26887 = x26882 > 0;
int32_t x26883 = x26882 * x26882;
float* x159 = x5+11452352;
bool x26931 = x26764 > 1;
int32_t x26971 = x26882 - 1;
int32_t x26972 = x26971 / 1;
int32_t x26973 = x26972 + 1;
int32_t x26977 = 131072 * x26973;
int32_t x26978 = x26977 * x26973;
int32_t x26974 = x26973 * x26973;
int32_t x26975 = 2048 * x26974;
float* x212 = x5+11453888;
bool x27052 = x26973 == 1;
bool x27053 = x27052 || true;
bool x27054 = x27053 || x27052;
bool x27064 = x26973 <= 1;
int32_t x27065;
if (x27064) {
x27065 = 1;
} else {
x27065 = x26973;
}
bool x27070 = x27065 > 0;
int32_t x27066 = x27065 * x27065;
float* x115 = x5+12506560;
bool x27114 = x26973 > 1;
float* x193 = x5+12508608;
bool x27184 = x27065 == 1;
bool x27185 = x27184 || true;
bool x27186 = x27185 || x27184;
bool x27196 = x27065 <= 1;
int32_t x27197;
if (x27196) {
x27197 = 1;
} else {
x27197 = x27065;
}
bool x27202 = x27197 > 0;
int32_t x27198 = x27197 * x27197;
bool x27246 = x27065 > 1;
bool x27302 = x27197 == 1;
bool x27303 = x27302 || true;
bool x27304 = x27303 || x27302;
bool x27314 = x27197 <= 1;
int32_t x27315;
if (x27314) {
x27315 = 1;
} else {
x27315 = x27197;
}
bool x27320 = x27315 > 0;
int32_t x27316 = x27315 * x27315;
float* x239 = x5+12502464;
bool x27364 = x27197 > 1;
bool x27420 = x27315 == 1;
bool x27421 = x27420 || true;
bool x27422 = x27421 || x27420;
bool x27432 = x27315 <= 1;
int32_t x27433;
if (x27432) {
x27433 = 1;
} else {
x27433 = x27315;
}
bool x27438 = x27433 > 0;
int32_t x27434 = x27433 * x27433;
float* x62 = x5+12504512;
bool x27482 = x27315 > 1;
int32_t x27509 = x25838 / 2;
int32_t x27510 = x27509 + 1;
int32_t x27514 = 131072 * x27510;
int32_t x27515 = x27514 * x27510;
int32_t x27511 = x27510 * x27510;
int32_t x27512 = 2048 * x27511;
float* x214 = x5+12510656;
bool x27595 = x27510 == 1;
bool x27596 = x27595 || true;
bool x27597 = x27596 || x27595;
bool x27607 = x27510 <= 1;
int32_t x27608;
if (x27607) {
x27608 = 1;
} else {
x27608 = x27510;
}
bool x27613 = x27608 > 0;
int32_t x27609 = x27608 * x27608;
float* x64 = x5+14611904;
bool x27657 = x27510 > 1;
float* x125 = x5+14613952;
bool x27726 = x27608 == 1;
bool x27727 = x27726 || true;
bool x27728 = x27727 || x27726;
bool x27738 = x27608 <= 1;
int32_t x27739;
if (x27738) {
x27739 = 1;
} else {
x27739 = x27608;
}
bool x27744 = x27739 > 0;
int32_t x27740 = x27739 * x27739;
bool x27788 = x27608 > 1;
bool x27844 = x27739 == 1;
bool x27845 = x27844 || true;
bool x27846 = x27845 || x27844;
bool x27856 = x27739 <= 1;
int32_t x27857;
if (x27856) {
x27857 = 1;
} else {
x27857 = x27739;
}
bool x27862 = x27857 > 0;
int32_t x27858 = x27857 * x27857;
float* x173 = x5+14607808;
bool x27906 = x27739 > 1;
bool x27962 = x27857 == 1;
bool x27963 = x27962 || true;
bool x27964 = x27963 || x27962;
bool x27974 = x27857 <= 1;
int32_t x27975;
if (x27974) {
x27975 = 1;
} else {
x27975 = x27857;
}
bool x27980 = x27975 > 0;
int32_t x27976 = x27975 * x27975;
float* x107 = x5+14609856;
bool x28024 = x27857 > 1;
bool x28056 = x27433 == 1;
bool x28057 = x27975 == 1;
bool x28058 = x28056 || x28057;
bool x28059 = x27433 == x27975;
bool x28060 = x28058 || x28059;
bool x28070 = x27433 <= x27975;
int32_t x28071;
if (x28070) {
x28071 = x27975;
} else {
x28071 = x27433;
}
bool x28109 = x27433 > 1;
bool x28113 = x27975 > 1;
int32_t x28072 = x28071 * x28071;
int32_t x28157 = x27433 - 1;
int32_t x28158 = x28157 / 1;
int32_t x28159 = x28158 + 1;
int32_t x28163 = 32768 * x28159;
int32_t x28164 = x28163 * x28159;
int32_t x28160 = x28159 * x28159;
int32_t x28161 = 512 * x28160;
float* x215 = x5+14616000;
bool x28238 = x28159 == 1;
bool x28239 = x28238 || true;
bool x28240 = x28239 || x28238;
bool x28250 = x28159 <= 1;
int32_t x28251;
if (x28250) {
x28251 = 1;
} else {
x28251 = x28159;
}
bool x28256 = x28251 > 0;
int32_t x28252 = x28251 * x28251;
float* x154 = x5+15665600;
bool x28300 = x28159 > 1;
float* x65 = x5+15666112;
bool x28369 = x28251 == 1;
bool x28370 = x28369 || true;
bool x28371 = x28370 || x28369;
bool x28381 = x28251 <= 1;
int32_t x28382;
if (x28381) {
x28382 = 1;
} else {
x28382 = x28251;
}
bool x28387 = x28382 > 0;
int32_t x28383 = x28382 * x28382;
bool x28431 = x28251 > 1;
bool x28487 = x28382 == 1;
bool x28488 = x28487 || true;
bool x28489 = x28488 || x28487;
bool x28499 = x28382 <= 1;
int32_t x28500;
if (x28499) {
x28500 = 1;
} else {
x28500 = x28382;
}
bool x28505 = x28500 > 0;
int32_t x28501 = x28500 * x28500;
float* x46 = x5+15664576;
bool x28549 = x28382 > 1;
bool x28605 = x28500 == 1;
bool x28606 = x28605 || true;
bool x28607 = x28606 || x28605;
bool x28617 = x28500 <= 1;
int32_t x28618;
if (x28617) {
x28618 = 1;
} else {
x28618 = x28500;
}
bool x28623 = x28618 > 0;
int32_t x28619 = x28618 * x28618;
float* x137 = x5+15665088;
bool x28667 = x28500 > 1;
int32_t x28707 = x28618 + 2;
int32_t x28708 = x28707 - 3;
int32_t x28709 = x28708 / 1;
int32_t x28710 = x28709 + 1;
int32_t x28714 = 32768 * x28710;
int32_t x28715 = x28714 * x28710;
int32_t x28711 = x28710 * x28710;
int32_t x28712 = 512 * x28711;
float* x155 = x5+15666624;
bool x28837 = x28710 == 1;
bool x28838 = x28837 || true;
bool x28839 = x28838 || x28837;
bool x28849 = x28710 <= 1;
int32_t x28850;
if (x28849) {
x28850 = 1;
} else {
x28850 = x28710;
}
bool x28855 = x28850 > 0;
int32_t x28851 = x28850 * x28850;
float* x138 = x5+18026944;
bool x28899 = x28710 > 1;
float* x195 = x5+18027456;
bool x28968 = x28850 == 1;
bool x28969 = x28968 || true;
bool x28970 = x28969 || x28968;
bool x28980 = x28850 <= 1;
int32_t x28981;
if (x28980) {
x28981 = 1;
} else {
x28981 = x28850;
}
bool x28986 = x28981 > 0;
int32_t x28982 = x28981 * x28981;
bool x29030 = x28850 > 1;
bool x29086 = x28981 == 1;
bool x29087 = x29086 || true;
bool x29088 = x29087 || x29086;
bool x29098 = x28981 <= 1;
int32_t x29099;
if (x29098) {
x29099 = 1;
} else {
x29099 = x28981;
}
bool x29104 = x29099 > 0;
int32_t x29100 = x29099 * x29099;
float* x160 = x5+18025920;
bool x29148 = x28981 > 1;
bool x29204 = x29099 == 1;
bool x29205 = x29204 || true;
bool x29206 = x29205 || x29204;
bool x29216 = x29099 <= 1;
int32_t x29217;
if (x29216) {
x29217 = 1;
} else {
x29217 = x29099;
}
bool x29222 = x29217 > 0;
int32_t x29218 = x29217 * x29217;
float* x66 = x5+18026432;
bool x29266 = x29099 > 1;
int32_t x29306 = x29217 - 1;
int32_t x29307 = x29306 / 1;
int32_t x29308 = x29307 + 1;
int32_t x29312 = 131072 * x29308;
int32_t x29313 = x29312 * x29308;
int32_t x29309 = x29308 * x29308;
int32_t x29310 = 2048 * x29309;
float* x47 = x5+18027968;
bool x29387 = x29308 == 1;
bool x29388 = x29387 || true;
bool x29389 = x29388 || x29387;
bool x29399 = x29308 <= 1;
int32_t x29400;
if (x29399) {
x29400 = 1;
} else {
x29400 = x29308;
}
bool x29405 = x29400 > 0;
int32_t x29401 = x29400 * x29400;
float* x68 = x5+19080640;
bool x29449 = x29308 > 1;
float* x245 = x5+19082688;
bool x29518 = x29400 == 1;
bool x29519 = x29518 || true;
bool x29520 = x29519 || x29518;
bool x29530 = x29400 <= 1;
int32_t x29531;
if (x29530) {
x29531 = 1;
} else {
x29531 = x29400;
}
bool x29536 = x29531 > 0;
int32_t x29532 = x29531 * x29531;
bool x29580 = x29400 > 1;
bool x29636 = x29531 == 1;
bool x29637 = x29636 || true;
bool x29638 = x29637 || x29636;
bool x29648 = x29531 <= 1;
int32_t x29649;
if (x29648) {
x29649 = 1;
} else {
x29649 = x29531;
}
bool x29654 = x29649 > 0;
int32_t x29650 = x29649 * x29649;
float* x94 = x5+19076544;
bool x29698 = x29531 > 1;
bool x29754 = x29649 == 1;
bool x29755 = x29754 || true;
bool x29756 = x29755 || x29754;
bool x29766 = x29649 <= 1;
int32_t x29767;
if (x29766) {
x29767 = 1;
} else {
x29767 = x29649;
}
bool x29772 = x29767 > 0;
int32_t x29768 = x29767 * x29767;
float* x144 = x5+19078592;
bool x29816 = x29649 > 1;
bool x29847 = x29767 == 1;
bool x29848 = x29847 || x28056;
bool x29849 = x29767 == x27433;
bool x29850 = x29848 || x29849;
bool x29860 = x29767 <= x27433;
int32_t x29861;
if (x29860) {
x29861 = x27433;
} else {
x29861 = x29767;
}
bool x29899 = x29767 > 1;
int32_t x29862 = x29861 * x29861;
int32_t x29945 = x29767 - 1;
int32_t x29946 = x29945 / 1;
int32_t x29947 = x29946 + 1;
int32_t x29951 = 32768 * x29947;
int32_t x29952 = x29951 * x29947;
int32_t x29948 = x29947 * x29947;
int32_t x29949 = 512 * x29948;
float* x265 = x5+19084736;
bool x30026 = x29947 == 1;
bool x30027 = x30026 || true;
bool x30028 = x30027 || x30026;
bool x30038 = x29947 <= 1;
int32_t x30039;
if (x30038) {
x30039 = 1;
} else {
x30039 = x29947;
}
bool x30044 = x30039 > 0;
int32_t x30040 = x30039 * x30039;
float* x213 = x5+20134336;
bool x30088 = x29947 > 1;
float* x255 = x5+20134848;
bool x30157 = x30039 == 1;
bool x30158 = x30157 || true;
bool x30159 = x30158 || x30157;
bool x30169 = x30039 <= 1;
int32_t x30170;
if (x30169) {
x30170 = 1;
} else {
x30170 = x30039;
}
bool x30175 = x30170 > 0;
int32_t x30171 = x30170 * x30170;
bool x30219 = x30039 > 1;
bool x30275 = x30170 == 1;
bool x30276 = x30275 || true;
bool x30277 = x30276 || x30275;
bool x30287 = x30170 <= 1;
int32_t x30288;
if (x30287) {
x30288 = 1;
} else {
x30288 = x30170;
}
bool x30293 = x30288 > 0;
int32_t x30289 = x30288 * x30288;
float* x15 = x5+20133312;
bool x30337 = x30170 > 1;
bool x30393 = x30288 == 1;
bool x30394 = x30393 || true;
bool x30395 = x30394 || x30393;
bool x30405 = x30288 <= 1;
int32_t x30406;
if (x30405) {
x30406 = 1;
} else {
x30406 = x30288;
}
bool x30411 = x30406 > 0;
int32_t x30407 = x30406 * x30406;
float* x78 = x5+20133824;
bool x30455 = x30288 > 1;
int32_t x30495 = x30406 + 2;
int32_t x30496 = x30495 - 3;
int32_t x30497 = x30496 / 1;
int32_t x30498 = x30497 + 1;
int32_t x30502 = 32768 * x30498;
int32_t x30503 = x30502 * x30498;
int32_t x30499 = x30498 * x30498;
int32_t x30500 = 512 * x30499;
float* x28 = x5+20135360;
bool x30625 = x30498 == 1;
bool x30626 = x30625 || true;
bool x30627 = x30626 || x30625;
bool x30637 = x30498 <= 1;
int32_t x30638;
if (x30637) {
x30638 = 1;
} else {
x30638 = x30498;
}
bool x30643 = x30638 > 0;
int32_t x30639 = x30638 * x30638;
float* x12 = x5+22495680;
bool x30687 = x30498 > 1;
float* x202 = x5+22496192;
bool x30756 = x30638 == 1;
bool x30757 = x30756 || true;
bool x30758 = x30757 || x30756;
bool x30768 = x30638 <= 1;
int32_t x30769;
if (x30768) {
x30769 = 1;
} else {
x30769 = x30638;
}
bool x30774 = x30769 > 0;
int32_t x30770 = x30769 * x30769;
bool x30818 = x30638 > 1;
bool x30874 = x30769 == 1;
bool x30875 = x30874 || true;
bool x30876 = x30875 || x30874;
bool x30886 = x30769 <= 1;
int32_t x30887;
if (x30886) {
x30887 = 1;
} else {
x30887 = x30769;
}
bool x30892 = x30887 > 0;
int32_t x30888 = x30887 * x30887;
float* x194 = x5+22494656;
bool x30936 = x30769 > 1;
bool x30992 = x30887 == 1;
bool x30993 = x30992 || true;
bool x30994 = x30993 || x30992;
bool x31004 = x30887 <= 1;
int32_t x31005;
if (x31004) {
x31005 = 1;
} else {
x31005 = x30887;
}
bool x31010 = x31005 > 0;
int32_t x31006 = x31005 * x31005;
float* x169 = x5+22495168;
bool x31054 = x30887 > 1;
int32_t x31094 = x31005 - 1;
int32_t x31095 = x31094 / 1;
int32_t x31096 = x31095 + 1;
int32_t x31100 = 131072 * x31096;
int32_t x31101 = x31100 * x31096;
int32_t x31097 = x31096 * x31096;
int32_t x31098 = 2048 * x31097;
float* x33 = x5+22496704;
bool x31175 = x31096 == 1;
bool x31176 = x31175 || true;
bool x31177 = x31176 || x31175;
bool x31187 = x31096 <= 1;
int32_t x31188;
if (x31187) {
x31188 = 1;
} else {
x31188 = x31096;
}
bool x31193 = x31188 > 0;
int32_t x31189 = x31188 * x31188;
float* x260 = x5+23549376;
bool x31237 = x31096 > 1;
float* x123 = x5+23551424;
bool x31306 = x31188 == 1;
bool x31307 = x31306 || true;
bool x31308 = x31307 || x31306;
bool x31318 = x31188 <= 1;
int32_t x31319;
if (x31318) {
x31319 = 1;
} else {
x31319 = x31188;
}
bool x31324 = x31319 > 0;
int32_t x31320 = x31319 * x31319;
bool x31368 = x31188 > 1;
bool x31424 = x31319 == 1;
bool x31425 = x31424 || true;
bool x31426 = x31425 || x31424;
bool x31436 = x31319 <= 1;
int32_t x31437;
if (x31436) {
x31437 = 1;
} else {
x31437 = x31319;
}
bool x31442 = x31437 > 0;
int32_t x31438 = x31437 * x31437;
float* x103 = x5+23545280;
bool x31486 = x31319 > 1;
bool x31542 = x31437 == 1;
bool x31543 = x31542 || true;
bool x31544 = x31543 || x31542;
bool x31554 = x31437 <= 1;
int32_t x31555;
if (x31554) {
x31555 = 1;
} else {
x31555 = x31437;
}
bool x31560 = x31555 > 0;
int32_t x31556 = x31555 * x31555;
float* x181 = x5+23547328;
bool x31604 = x31437 > 1;
bool x31635 = x31555 == 1;
bool x31636 = x31635 || x29847;
bool x31637 = x31555 == x29767;
bool x31638 = x31636 || x31637;
bool x31648 = x31555 <= x29767;
int32_t x31649;
if (x31648) {
x31649 = x29767;
} else {
x31649 = x31555;
}
bool x31687 = x31555 > 1;
int32_t x31650 = x31649 * x31649;
bool x31733 = x31555 >= 2;
bool x31734;
if (x31733) {
x31734 = x31733;
} else {
x31734 = false;
}
int32_t x31739 = x31555 - 2;
int32_t x31740 = x31739 / 1;
int32_t x31741 = x31740 + 1;
int32_t x31742 = x31741 * x31741;
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
float x313 = x306[x311];
float x314 = fabs(x313);
float x315 = fabs(x312);
bool x316 = x314 > x315;
float x317;
if (x316) {
x317 = x313;
} else {
x317 = x312;
}
x309 = x317;

}
float x321 = x309;
printf("Max Abs: %.5f || ",x321);
for(int x324=0; x324 < 10; x324++) {
float x325 = x306[x324];
printf("%.5f ",x325);

}
printf("\n");
float* x337 = (float*)myMalloc(x336 * sizeof(float));;
float* x341 = (float*)myMalloc(x340 * sizeof(float));;
for(int x343=0; x343 < 64; x343++) {
int32_t x344 = x343 * 3072;
float* x345 = x306+x344;
int32_t x346 = x343 * x333;
float* x347 = x337+x346;
int32_t x348 = x343 * x338;
float* x349 = x341+x348;
for(int x351=0; x351 < 27; x351++) {
int32_t x352 = x351 / 9;
int32_t x356 = x352 * 3;
int32_t x357 = x356 * 3;
int32_t x358 = x357 * x331;
int32_t x359 = x358 * x331;
int32_t x353 = x351 % 9;
int32_t x354 = x353 / 3;
int32_t x360 = x354 * 3;
int32_t x361 = x360 * x331;
int32_t x362 = x361 * x331;
int32_t x363 = x359 + x362;
int32_t x355 = x353 % 3;
int32_t x364 = x355 * x331;
int32_t x365 = x364 * x331;
int32_t x366 = x363 + x365;
float* x367 = x349+x366;
int32_t x368 = x352 * 32;
int32_t x369 = x368 * 32;
float* x370 = x345+x369;
int32_t x383 = 1 - x355;
bool x384 = x383 > 0;
int32_t x385;
if (x384) {
x385 = x383;
} else {
x385 = 0;
}
int32_t x386 = 3 - x355;
int32_t x387 = x386 - 1;
int32_t x388 = 1 - x387;
bool x389 = x388 > 0;
int32_t x390;
if (x389) {
x390 = x388;
} else {
x390 = 0;
}
int32_t x391 = x331 - x390;
int32_t x392 = x391 - x385;
bool x393 = x392 <= 0;
bool x397 = x385 > 0;
int32_t x382 = -1 + x355;
bool x410 = x390 > 0;
for(int x372=0; x372 < x331; x372++) {
int32_t x373 = x372 - 1;
int32_t x374 = x373 + x354;
bool x375 = x374 < 0;
bool x376 = x374 >= 32;
bool x377 = x375 || x376;
if (x377) {
int32_t x378 = x372 * x331;
float* x379 = x367+x378;
memset(x379, 0, 4 * x331);;
} else {
if (x393) {
int32_t x378 = x372 * x331;
float* x394 = x367+x378;
memset(x394, 0, 4 * x331);;
} else {
int32_t x378 = x372 * x331;
if (x397) {
float* x398 = x367+x378;
memset(x398, 0, 4 * x385);;
} else {
}
// may have segfault here
int32_t x403 = x378 + x385;
float* x404 = x367+x403;
int32_t x405 = x374 * 32;
int32_t x406 = x405 + x382;
int32_t x407 = x406 + x385;
float* x408 = x370+x407;
memcpy(x404, x408, 4 * x392);;
if (x410) {
int32_t x411 = x378 + x331;
int32_t x412 = x411 - x390;
float* x413 = x367+x412;
memset(x413, 0, 4 * x390);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x332,27,1,x152,27,x349,x332,1,x347,x332);

}
int32_t x428 = 0;
int32_t x429 = 1;
x429 *= 1;
x428 += 1;
x429 *= 1;
x429 *= 1;
int32_t x434 = x428;
bool x435 = x434 >= 2;
if (x435) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x441 = x434 == 0;
if (x441) {
int32_t x442 = x429;
bool x443 = x442 == 64;
if (x443) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x450 = x429;
int32_t x451 = 64 / x450;
bool x456;
if (x452) {
bool x453 = x451 == 1;
bool x454 = 64 == x451;
bool x455 = x453 || x454;
x456 = x455;
} else {
x456 = false;
}
bool x460;
if (x456) {
x460 = x459;
} else {
x460 = false;
}
bool x461;
if (x460) {
x461 = x459;
} else {
x461 = false;
}
if (x461) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x331,x331,1,x451,1,1);
assert(false && "");
}
bool x467 = 64 <= x451;
int32_t x468;
if (x467) {
x468 = x451;
} else {
x468 = 64;
}
bool x474 = x468 > 0;
bool x476;
if (x474) {
x476 = x475;
} else {
x476 = false;
}
bool x477;
if (x476) {
x477 = x475;
} else {
x477 = false;
}
if (x477) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(64) x Sym(331) x Sym(331)"," x Const(1) x Sym(451) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x472 = x468 * x471;
int32_t x473 = 64 * x472;
float* x483 = (float*)myMalloc(x473 * sizeof(float));;
int32_t x484 = 0;
int32_t x485 = 0;
int32_t x486 = 0;
bool x533 = x451 > 1;
for(int x487=0; x487 < 64; x487++) {
int32_t x488 = x485;
int32_t x489 = x486;
int32_t x490 = x484;
int32_t x491 = x490;
int32_t x492 = x488;
int32_t x493 = x489;
for(int x495=0; x495 < x468; x495++) {
int32_t x496 = x492;
int32_t x497 = x493;
int32_t x498 = x491;
int32_t x499 = x498;
int32_t x500 = x496;
int32_t x501 = x497;
for(int x503=0; x503 < x470; x503++) {
int32_t x504 = x500;
int32_t x505 = x501;
int32_t x506 = x499;
int32_t x507 = x506;
int32_t x508 = x504;
int32_t x509 = x505;
for(int x510=0; x510 < x470; x510++) {
int32_t x511 = x507;
int32_t x512 = x508;
float x513 = x337[x512];
int32_t x514 = x509;
float x515 = x40[x514];
float x516 = x513 - x515;
x483[x511] = x516;
x507 += 1;
if (x519) {
x508 += 1;
} else {
}

}
x499 += x470;
if (x519) {
x500 += x331;
} else {
}

}
x491 += x471;
x492 += x332;
if (x533) {
x493 += 1;
} else {
}

}
x484 += x472;
x485 += x333;

}
float* x543 = (float*)myMalloc(64 * sizeof(float));;
for(int x544=0; x544 < 64; x544++) {
float x545 = x110[x544];
float x546 = x545 + 1.0E-5f;
x543[x544] = x546;

}
float* x550 = (float*)myMalloc(64 * sizeof(float));;
for(int x551=0; x551 < 64; x551++) {
float x552 = x543[x551];
double x553 = (double)x552;
double x554 = sqrt(x553);
float x555 = (float)x554;
x550[x551] = x555;

}
int32_t x559 = 0;
int32_t x560 = 1;
x560 *= 1;
x559 += 1;
x560 *= 1;
x560 *= 1;
int32_t x565 = x559;
bool x566 = x565 >= 2;
if (x566) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x571 = x565 == 0;
if (x571) {
int32_t x572 = x560;
bool x573 = x572 == 64;
if (x573) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x580 = x560;
int32_t x581 = 64 / x580;
bool x587;
if (x452) {
bool x582 = x468 == 1;
bool x583 = x581 == 1;
bool x584 = x582 || x583;
bool x585 = x468 == x581;
bool x586 = x584 || x585;
x587 = x586;
} else {
x587 = false;
}
bool x591;
if (x587) {
x591 = x590;
} else {
x591 = false;
}
bool x592;
if (x591) {
x592 = x590;
} else {
x592 = false;
}
if (x592) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x468,x470,x470,1,x581,1,1);
assert(false && "");
}
bool x598 = x468 <= x581;
int32_t x599;
if (x598) {
x599 = x581;
} else {
x599 = x468;
}
bool x605 = x599 > 0;
bool x607;
if (x605) {
x607 = x606;
} else {
x607 = false;
}
bool x608;
if (x607) {
x608 = x606;
} else {
x608 = false;
}
if (x608) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(468) x Sym(470) x Sym(470)"," x Const(1) x Sym(581) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x603 = x599 * x602;
int32_t x604 = 64 * x603;
float* x614 = (float*)myMalloc(x604 * sizeof(float));;
int32_t x615 = 0;
int32_t x616 = 0;
int32_t x617 = 0;
bool x663 = x468 > 1;
bool x667 = x581 > 1;
for(int x618=0; x618 < 64; x618++) {
int32_t x619 = x616;
int32_t x620 = x617;
int32_t x621 = x615;
int32_t x622 = x621;
int32_t x623 = x619;
int32_t x624 = x620;
for(int x626=0; x626 < x599; x626++) {
int32_t x627 = x623;
int32_t x628 = x624;
int32_t x629 = x622;
int32_t x630 = x629;
int32_t x631 = x627;
int32_t x632 = x628;
for(int x634=0; x634 < x601; x634++) {
int32_t x635 = x631;
int32_t x636 = x632;
int32_t x637 = x630;
int32_t x638 = x637;
int32_t x639 = x635;
int32_t x640 = x636;
for(int x641=0; x641 < x601; x641++) {
int32_t x642 = x638;
int32_t x643 = x639;
float x644 = x483[x643];
int32_t x645 = x640;
float x646 = x550[x645];
float x647 = x644 / x646;
x614[x642] = x647;
x638 += 1;
if (x650) {
x639 += 1;
} else {
}

}
x630 += x601;
if (x650) {
x631 += x470;
} else {
}

}
x622 += x602;
if (x663) {
x623 += x471;
} else {
}
if (x667) {
x624 += 1;
} else {
}

}
x615 += x603;
x616 += x472;

}
int32_t x677 = 0;
int32_t x678 = 1;
x678 *= 1;
x677 += 1;
x678 *= 1;
x678 *= 1;
int32_t x683 = x677;
bool x684 = x683 >= 2;
if (x684) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x689 = x683 == 0;
if (x689) {
int32_t x690 = x678;
bool x691 = x690 == 64;
if (x691) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x698 = x678;
int32_t x699 = 64 / x698;
bool x705;
if (x452) {
bool x700 = x599 == 1;
bool x701 = x699 == 1;
bool x702 = x700 || x701;
bool x703 = x599 == x699;
bool x704 = x702 || x703;
x705 = x704;
} else {
x705 = false;
}
bool x709;
if (x705) {
x709 = x708;
} else {
x709 = false;
}
bool x710;
if (x709) {
x710 = x708;
} else {
x710 = false;
}
if (x710) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x599,x601,x601,1,x699,1,1);
assert(false && "");
}
bool x716 = x599 <= x699;
int32_t x717;
if (x716) {
x717 = x699;
} else {
x717 = x599;
}
bool x723 = x717 > 0;
bool x725;
if (x723) {
x725 = x724;
} else {
x725 = false;
}
bool x726;
if (x725) {
x726 = x724;
} else {
x726 = false;
}
if (x726) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(599) x Sym(601) x Sym(601)"," x Const(1) x Sym(699) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x721 = x717 * x720;
int32_t x722 = 64 * x721;
float* x732 = (float*)myMalloc(x722 * sizeof(float));;
int32_t x733 = 0;
int32_t x734 = 0;
int32_t x735 = 0;
bool x781 = x599 > 1;
bool x785 = x699 > 1;
for(int x736=0; x736 < 64; x736++) {
int32_t x737 = x734;
int32_t x738 = x735;
int32_t x739 = x733;
int32_t x740 = x739;
int32_t x741 = x737;
int32_t x742 = x738;
for(int x744=0; x744 < x717; x744++) {
int32_t x745 = x741;
int32_t x746 = x742;
int32_t x747 = x740;
int32_t x748 = x747;
int32_t x749 = x745;
int32_t x750 = x746;
for(int x752=0; x752 < x719; x752++) {
int32_t x753 = x749;
int32_t x754 = x750;
int32_t x755 = x748;
int32_t x756 = x755;
int32_t x757 = x753;
int32_t x758 = x754;
for(int x759=0; x759 < x719; x759++) {
int32_t x760 = x756;
int32_t x761 = x757;
float x762 = x614[x761];
int32_t x763 = x758;
float x764 = x206[x763];
float x765 = x762 * x764;
x732[x760] = x765;
x756 += 1;
if (x768) {
x757 += 1;
} else {
}

}
x748 += x719;
if (x768) {
x749 += x601;
} else {
}

}
x740 += x720;
if (x781) {
x741 += x602;
} else {
}
if (x785) {
x742 += 1;
} else {
}

}
x733 += x721;
x734 += x603;

}
int32_t x795 = 0;
int32_t x796 = 1;
x796 *= 1;
x795 += 1;
x796 *= 1;
x796 *= 1;
int32_t x801 = x795;
bool x802 = x801 >= 2;
if (x802) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x807 = x801 == 0;
if (x807) {
int32_t x808 = x796;
bool x809 = x808 == 64;
if (x809) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x816 = x796;
int32_t x817 = 64 / x816;
bool x823;
if (x452) {
bool x818 = x717 == 1;
bool x819 = x817 == 1;
bool x820 = x818 || x819;
bool x821 = x717 == x817;
bool x822 = x820 || x821;
x823 = x822;
} else {
x823 = false;
}
bool x827;
if (x823) {
x827 = x826;
} else {
x827 = false;
}
bool x828;
if (x827) {
x828 = x826;
} else {
x828 = false;
}
if (x828) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x717,x719,x719,1,x817,1,1);
assert(false && "");
}
bool x834 = x717 <= x817;
int32_t x835;
if (x834) {
x835 = x817;
} else {
x835 = x717;
}
bool x841 = x835 > 0;
bool x843;
if (x841) {
x843 = x842;
} else {
x843 = false;
}
bool x844;
if (x843) {
x844 = x842;
} else {
x844 = false;
}
if (x844) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(717) x Sym(719) x Sym(719)"," x Const(1) x Sym(817) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x839 = x835 * x838;
int32_t x840 = 64 * x839;
float* x850 = (float*)myMalloc(x840 * sizeof(float));;
int32_t x851 = 0;
int32_t x852 = 0;
int32_t x853 = 0;
bool x899 = x717 > 1;
bool x903 = x817 > 1;
for(int x854=0; x854 < 64; x854++) {
int32_t x855 = x852;
int32_t x856 = x853;
int32_t x857 = x851;
int32_t x858 = x857;
int32_t x859 = x855;
int32_t x860 = x856;
for(int x862=0; x862 < x835; x862++) {
int32_t x863 = x859;
int32_t x864 = x860;
int32_t x865 = x858;
int32_t x866 = x865;
int32_t x867 = x863;
int32_t x868 = x864;
for(int x870=0; x870 < x837; x870++) {
int32_t x871 = x867;
int32_t x872 = x868;
int32_t x873 = x866;
int32_t x874 = x873;
int32_t x875 = x871;
int32_t x876 = x872;
for(int x877=0; x877 < x837; x877++) {
int32_t x878 = x874;
int32_t x879 = x875;
float x880 = x732[x879];
int32_t x881 = x876;
float x882 = x251[x881];
float x883 = x880 + x882;
x850[x878] = x883;
x874 += 1;
if (x886) {
x875 += 1;
} else {
}

}
x866 += x837;
if (x886) {
x867 += x719;
} else {
}

}
x858 += x838;
if (x899) {
x859 += x720;
} else {
}
if (x903) {
x860 += 1;
} else {
}

}
x851 += x839;
x852 += x721;

}
float* x913 = (float*)myMalloc(x840 * sizeof(float));;
for(int x915=0; x915 < x840; x915++) {
float x916 = x850[x915];
bool x917 = x916 < 0.0f;
if (x917) {
x913[x915] = 0.0f;
} else {
float x920 = x850[x915];
x913[x915] = x920;
}

}
if (x927) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(64) x Sym(835) x Sym(837) x Sym(837)|(2,2)");
}
int32_t x938 = 64 * x835;
int32_t x939 = x938 * x934;
int32_t x940 = x939 * x934;
float* x941 = (float*)myMalloc(x940 * sizeof(float));;
for(int x943=0; x943 < x940; x943++) {
x941[x943] = -3.4028235E38f;

}
int32_t x936 = x835 * x935;
int32_t x937 = 64 * x936;
int* x947 = (int32_t*)myMalloc(x937 * sizeof(int32_t));;
for(int x948=0; x948 < 64; x948++) {
int32_t x949 = x948 * x839;
float* x950 = x913+x949;
int32_t x951 = x948 * x936;
float* x952 = x941+x951;
int* x953 = x947+x951;
int32_t x954 = 0;
int32_t x955 = 0;
for(int x956=0; x956 < x835; x956++) {
int32_t x957 = x954;
int32_t x958 = x957;
int32_t x959 = x955;
int32_t x960 = x959;
for(int x962=0; x962 < x934; x962++) {
int32_t x963 = x958;
int32_t x964 = x963;
int32_t x965 = x960;
int32_t x966 = x965;
for(int x967=0; x967 < x934; x967++) {
int32_t x968 = x966;
int32_t x969 = x968;
int32_t x970 = x969;
int32_t x971 = x970;
int32_t x972 = x971;
float x973 = x950[x972];
int32_t x974 = x964;
float x975 = x952[x974];
bool x976 = x973 > x975;
if (x976) {
float x977 = x950[x972];
x952[x974] = x977;
int32_t x979 = x972 + x949;
x953[x974] = x979;
} else {
}
x971 += 1;
int32_t x984 = x971;
float x985 = x950[x984];
float x986 = x952[x974];
bool x987 = x985 > x986;
if (x987) {
float x988 = x950[x984];
x952[x974] = x988;
int32_t x990 = x984 + x949;
x953[x974] = x990;
} else {
}
x971 += 1;
x969 += x837;
int32_t x996 = x969;
int32_t x997 = x996;
int32_t x998 = x997;
float x999 = x950[x998];
float x1000 = x952[x974];
bool x1001 = x999 > x1000;
if (x1001) {
float x1002 = x950[x998];
x952[x974] = x1002;
int32_t x1004 = x998 + x949;
x953[x974] = x1004;
} else {
}
x997 += 1;
int32_t x1009 = x997;
float x1010 = x950[x1009];
float x1011 = x952[x974];
bool x1012 = x1010 > x1011;
if (x1012) {
float x1013 = x950[x1009];
x952[x974] = x1013;
int32_t x1015 = x1009 + x949;
x953[x974] = x1015;
} else {
}
x997 += 1;
x969 += x837;
x964 += 1;
x966 += 2;

}
x958 += x934;
x960 += x1026;

}
x954 += x935;
x955 += x838;

}

}
float* x1043 = (float*)myMalloc(x1042 * sizeof(float));;
int32_t x1046 = x938 * x1038;
float* x1047 = (float*)myMalloc(x1046 * sizeof(float));;
int32_t x1044 = x835 * x1038;
for(int x1048=0; x1048 < 64; x1048++) {
int32_t x1049 = x1048 * x936;
float* x1050 = x941+x1049;
int32_t x1051 = x1048 * x1039;
float* x1052 = x1043+x1051;
int32_t x1053 = x1048 * x1044;
float* x1054 = x1047+x1053;
for(int x1055=0; x1055 < x835; x1055++) {
int32_t x1056 = x1055 / 1;
int32_t x1060 = x1056 * x1037;
int32_t x1061 = x1060 * x1037;
int32_t x1057 = x1055 % 1;
int32_t x1058 = x1057 / 1;
int32_t x1062 = x1058 * x1037;
int32_t x1063 = x1062 * x1037;
int32_t x1064 = x1061 + x1063;
int32_t x1059 = x1057 % 1;
int32_t x1065 = x1059 * x1037;
int32_t x1066 = x1065 * x1037;
int32_t x1067 = x1064 + x1066;
float* x1068 = x1054+x1067;
int32_t x1069 = x1056 * x934;
int32_t x1070 = x1069 * x934;
float* x1071 = x1050+x1070;
for(int x1073=0; x1073 < x1037; x1073++) {
int32_t x1075 = x1073 * x1037;
float* x1076 = x1068+x1075;
int32_t x1074 = x1073 + x1058;
int32_t x1077 = x1074 * x934;
int32_t x1078 = x1077 + x1059;
float* x1079 = x1071+x1078;
memcpy(x1076, x1079, 4 * x1037);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1038,x835,1,x233,x835,x1054,x1038,1,x1052,x1038);

}
int32_t x1088 = 0;
int32_t x1089 = 1;
x1089 *= 1;
x1088 += 1;
x1089 *= 1;
x1089 *= 1;
int32_t x1094 = x1088;
bool x1095 = x1094 >= 2;
if (x1095) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1100 = x1094 == 0;
if (x1100) {
int32_t x1101 = x1089;
bool x1102 = x1101 == 64;
if (x1102) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1109 = x1089;
int32_t x1110 = 64 / x1109;
bool x1114;
if (x452) {
bool x1111 = x1110 == 1;
bool x1112 = 64 == x1110;
bool x1113 = x1111 || x1112;
x1114 = x1113;
} else {
x1114 = false;
}
bool x1118;
if (x1114) {
x1118 = x1117;
} else {
x1118 = false;
}
bool x1119;
if (x1118) {
x1119 = x1117;
} else {
x1119 = false;
}
if (x1119) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x1037,x1037,1,x1110,1,1);
assert(false && "");
}
bool x1125 = 64 <= x1110;
int32_t x1126;
if (x1125) {
x1126 = x1110;
} else {
x1126 = 64;
}
bool x1132 = x1126 > 0;
bool x1134;
if (x1132) {
x1134 = x1133;
} else {
x1134 = false;
}
bool x1135;
if (x1134) {
x1135 = x1133;
} else {
x1135 = false;
}
if (x1135) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(64) x Sym(1037) x Sym(1037)"," x Const(1) x Sym(1110) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x1130 = x1126 * x1129;
int32_t x1131 = 64 * x1130;
float* x1141 = (float*)myMalloc(x1131 * sizeof(float));;
int32_t x1142 = 0;
int32_t x1143 = 0;
int32_t x1144 = 0;
bool x1191 = x1110 > 1;
for(int x1145=0; x1145 < 64; x1145++) {
int32_t x1146 = x1143;
int32_t x1147 = x1144;
int32_t x1148 = x1142;
int32_t x1149 = x1148;
int32_t x1150 = x1146;
int32_t x1151 = x1147;
for(int x1153=0; x1153 < x1126; x1153++) {
int32_t x1154 = x1150;
int32_t x1155 = x1151;
int32_t x1156 = x1149;
int32_t x1157 = x1156;
int32_t x1158 = x1154;
int32_t x1159 = x1155;
for(int x1161=0; x1161 < x1128; x1161++) {
int32_t x1162 = x1158;
int32_t x1163 = x1159;
int32_t x1164 = x1157;
int32_t x1165 = x1164;
int32_t x1166 = x1162;
int32_t x1167 = x1163;
for(int x1168=0; x1168 < x1128; x1168++) {
int32_t x1169 = x1165;
int32_t x1170 = x1166;
float x1171 = x1043[x1170];
int32_t x1172 = x1167;
float x1173 = x114[x1172];
float x1174 = x1171 - x1173;
x1141[x1169] = x1174;
x1165 += 1;
if (x1177) {
x1166 += 1;
} else {
}

}
x1157 += x1128;
if (x1177) {
x1158 += x1037;
} else {
}

}
x1149 += x1129;
x1150 += x1038;
if (x1191) {
x1151 += 1;
} else {
}

}
x1142 += x1130;
x1143 += x1039;

}
float* x1201 = (float*)myMalloc(64 * sizeof(float));;
for(int x1202=0; x1202 < 64; x1202++) {
float x1203 = x51[x1202];
float x1204 = x1203 + 1.0E-5f;
x1201[x1202] = x1204;

}
float* x1208 = (float*)myMalloc(64 * sizeof(float));;
for(int x1209=0; x1209 < 64; x1209++) {
float x1210 = x1201[x1209];
double x1211 = (double)x1210;
double x1212 = sqrt(x1211);
float x1213 = (float)x1212;
x1208[x1209] = x1213;

}
int32_t x1217 = 0;
int32_t x1218 = 1;
x1218 *= 1;
x1217 += 1;
x1218 *= 1;
x1218 *= 1;
int32_t x1223 = x1217;
bool x1224 = x1223 >= 2;
if (x1224) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1229 = x1223 == 0;
if (x1229) {
int32_t x1230 = x1218;
bool x1231 = x1230 == 64;
if (x1231) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1238 = x1218;
int32_t x1239 = 64 / x1238;
bool x1245;
if (x452) {
bool x1240 = x1126 == 1;
bool x1241 = x1239 == 1;
bool x1242 = x1240 || x1241;
bool x1243 = x1126 == x1239;
bool x1244 = x1242 || x1243;
x1245 = x1244;
} else {
x1245 = false;
}
bool x1249;
if (x1245) {
x1249 = x1248;
} else {
x1249 = false;
}
bool x1250;
if (x1249) {
x1250 = x1248;
} else {
x1250 = false;
}
if (x1250) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1126,x1128,x1128,1,x1239,1,1);
assert(false && "");
}
bool x1256 = x1126 <= x1239;
int32_t x1257;
if (x1256) {
x1257 = x1239;
} else {
x1257 = x1126;
}
bool x1263 = x1257 > 0;
bool x1265;
if (x1263) {
x1265 = x1264;
} else {
x1265 = false;
}
bool x1266;
if (x1265) {
x1266 = x1264;
} else {
x1266 = false;
}
if (x1266) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(1126) x Sym(1128) x Sym(1128)"," x Const(1) x Sym(1239) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x1261 = x1257 * x1260;
int32_t x1262 = 64 * x1261;
float* x1272 = (float*)myMalloc(x1262 * sizeof(float));;
int32_t x1273 = 0;
int32_t x1274 = 0;
int32_t x1275 = 0;
bool x1321 = x1126 > 1;
bool x1325 = x1239 > 1;
for(int x1276=0; x1276 < 64; x1276++) {
int32_t x1277 = x1274;
int32_t x1278 = x1275;
int32_t x1279 = x1273;
int32_t x1280 = x1279;
int32_t x1281 = x1277;
int32_t x1282 = x1278;
for(int x1284=0; x1284 < x1257; x1284++) {
int32_t x1285 = x1281;
int32_t x1286 = x1282;
int32_t x1287 = x1280;
int32_t x1288 = x1287;
int32_t x1289 = x1285;
int32_t x1290 = x1286;
for(int x1292=0; x1292 < x1259; x1292++) {
int32_t x1293 = x1289;
int32_t x1294 = x1290;
int32_t x1295 = x1288;
int32_t x1296 = x1295;
int32_t x1297 = x1293;
int32_t x1298 = x1294;
for(int x1299=0; x1299 < x1259; x1299++) {
int32_t x1300 = x1296;
int32_t x1301 = x1297;
float x1302 = x1141[x1301];
int32_t x1303 = x1298;
float x1304 = x1208[x1303];
float x1305 = x1302 / x1304;
x1272[x1300] = x1305;
x1296 += 1;
if (x1308) {
x1297 += 1;
} else {
}

}
x1288 += x1259;
if (x1308) {
x1289 += x1128;
} else {
}

}
x1280 += x1260;
if (x1321) {
x1281 += x1129;
} else {
}
if (x1325) {
x1282 += 1;
} else {
}

}
x1273 += x1261;
x1274 += x1130;

}
int32_t x1335 = 0;
int32_t x1336 = 1;
x1336 *= 1;
x1335 += 1;
x1336 *= 1;
x1336 *= 1;
int32_t x1341 = x1335;
bool x1342 = x1341 >= 2;
if (x1342) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1347 = x1341 == 0;
if (x1347) {
int32_t x1348 = x1336;
bool x1349 = x1348 == 64;
if (x1349) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1356 = x1336;
int32_t x1357 = 64 / x1356;
bool x1363;
if (x452) {
bool x1358 = x1257 == 1;
bool x1359 = x1357 == 1;
bool x1360 = x1358 || x1359;
bool x1361 = x1257 == x1357;
bool x1362 = x1360 || x1361;
x1363 = x1362;
} else {
x1363 = false;
}
bool x1367;
if (x1363) {
x1367 = x1366;
} else {
x1367 = false;
}
bool x1368;
if (x1367) {
x1368 = x1366;
} else {
x1368 = false;
}
if (x1368) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1257,x1259,x1259,1,x1357,1,1);
assert(false && "");
}
bool x1374 = x1257 <= x1357;
int32_t x1375;
if (x1374) {
x1375 = x1357;
} else {
x1375 = x1257;
}
bool x1381 = x1375 > 0;
bool x1383;
if (x1381) {
x1383 = x1382;
} else {
x1383 = false;
}
bool x1384;
if (x1383) {
x1384 = x1382;
} else {
x1384 = false;
}
if (x1384) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(1257) x Sym(1259) x Sym(1259)"," x Const(1) x Sym(1357) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x1379 = x1375 * x1378;
int32_t x1380 = 64 * x1379;
float* x1390 = (float*)myMalloc(x1380 * sizeof(float));;
int32_t x1391 = 0;
int32_t x1392 = 0;
int32_t x1393 = 0;
bool x1439 = x1257 > 1;
bool x1443 = x1357 > 1;
for(int x1394=0; x1394 < 64; x1394++) {
int32_t x1395 = x1392;
int32_t x1396 = x1393;
int32_t x1397 = x1391;
int32_t x1398 = x1397;
int32_t x1399 = x1395;
int32_t x1400 = x1396;
for(int x1402=0; x1402 < x1375; x1402++) {
int32_t x1403 = x1399;
int32_t x1404 = x1400;
int32_t x1405 = x1398;
int32_t x1406 = x1405;
int32_t x1407 = x1403;
int32_t x1408 = x1404;
for(int x1410=0; x1410 < x1377; x1410++) {
int32_t x1411 = x1407;
int32_t x1412 = x1408;
int32_t x1413 = x1406;
int32_t x1414 = x1413;
int32_t x1415 = x1411;
int32_t x1416 = x1412;
for(int x1417=0; x1417 < x1377; x1417++) {
int32_t x1418 = x1414;
int32_t x1419 = x1415;
float x1420 = x1272[x1419];
int32_t x1421 = x1416;
float x1422 = x26[x1421];
float x1423 = x1420 * x1422;
x1390[x1418] = x1423;
x1414 += 1;
if (x1426) {
x1415 += 1;
} else {
}

}
x1406 += x1377;
if (x1426) {
x1407 += x1259;
} else {
}

}
x1398 += x1378;
if (x1439) {
x1399 += x1260;
} else {
}
if (x1443) {
x1400 += 1;
} else {
}

}
x1391 += x1379;
x1392 += x1261;

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
bool x1481;
if (x452) {
bool x1476 = x1375 == 1;
bool x1477 = x1475 == 1;
bool x1478 = x1476 || x1477;
bool x1479 = x1375 == x1475;
bool x1480 = x1478 || x1479;
x1481 = x1480;
} else {
x1481 = false;
}
bool x1485;
if (x1481) {
x1485 = x1484;
} else {
x1485 = false;
}
bool x1486;
if (x1485) {
x1486 = x1484;
} else {
x1486 = false;
}
if (x1486) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1375,x1377,x1377,1,x1475,1,1);
assert(false && "");
}
bool x1492 = x1375 <= x1475;
int32_t x1493;
if (x1492) {
x1493 = x1475;
} else {
x1493 = x1375;
}
bool x1499 = x1493 > 0;
bool x1501;
if (x1499) {
x1501 = x1500;
} else {
x1501 = false;
}
bool x1502;
if (x1501) {
x1502 = x1500;
} else {
x1502 = false;
}
if (x1502) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(1375) x Sym(1377) x Sym(1377)"," x Const(1) x Sym(1475) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x1497 = x1493 * x1496;
int32_t x1498 = 64 * x1497;
float* x1508 = (float*)myMalloc(x1498 * sizeof(float));;
int32_t x1509 = 0;
int32_t x1510 = 0;
int32_t x1511 = 0;
bool x1557 = x1375 > 1;
bool x1561 = x1475 > 1;
for(int x1512=0; x1512 < 64; x1512++) {
int32_t x1513 = x1510;
int32_t x1514 = x1511;
int32_t x1515 = x1509;
int32_t x1516 = x1515;
int32_t x1517 = x1513;
int32_t x1518 = x1514;
for(int x1520=0; x1520 < x1493; x1520++) {
int32_t x1521 = x1517;
int32_t x1522 = x1518;
int32_t x1523 = x1516;
int32_t x1524 = x1523;
int32_t x1525 = x1521;
int32_t x1526 = x1522;
for(int x1528=0; x1528 < x1495; x1528++) {
int32_t x1529 = x1525;
int32_t x1530 = x1526;
int32_t x1531 = x1524;
int32_t x1532 = x1531;
int32_t x1533 = x1529;
int32_t x1534 = x1530;
for(int x1535=0; x1535 < x1495; x1535++) {
int32_t x1536 = x1532;
int32_t x1537 = x1533;
float x1538 = x1390[x1537];
int32_t x1539 = x1534;
float x1540 = x53[x1539];
float x1541 = x1538 + x1540;
x1508[x1536] = x1541;
x1532 += 1;
if (x1544) {
x1533 += 1;
} else {
}

}
x1524 += x1495;
if (x1544) {
x1525 += x1377;
} else {
}

}
x1516 += x1496;
if (x1557) {
x1517 += x1378;
} else {
}
if (x1561) {
x1518 += 1;
} else {
}

}
x1509 += x1497;
x1510 += x1379;

}
float* x1571 = (float*)myMalloc(x1498 * sizeof(float));;
for(int x1573=0; x1573 < x1498; x1573++) {
float x1574 = x1508[x1573];
bool x1575 = x1574 < 0.0f;
if (x1575) {
x1571[x1573] = 0.0f;
} else {
float x1578 = x1508[x1573];
x1571[x1573] = x1578;
}

}
float* x1593 = (float*)myMalloc(x1592 * sizeof(float));;
int32_t x1594 = 9 * x1493;
int32_t x1597 = 64 * x1594;
int32_t x1598 = x1597 * x1588;
float* x1599 = (float*)myMalloc(x1598 * sizeof(float));;
int32_t x1595 = x1594 * x1588;
int32_t x1607 = x1493 * 3;
int32_t x1608 = x1607 * 3;
for(int x1600=0; x1600 < 64; x1600++) {
int32_t x1601 = x1600 * x1497;
float* x1602 = x1571+x1601;
int32_t x1603 = x1600 * x1589;
float* x1604 = x1593+x1603;
int32_t x1605 = x1600 * x1595;
float* x1606 = x1599+x1605;
for(int x1610=0; x1610 < x1608; x1610++) {
int32_t x1611 = x1610 / 9;
int32_t x1615 = x1611 * 3;
int32_t x1616 = x1615 * 3;
int32_t x1617 = x1616 * x1587;
int32_t x1618 = x1617 * x1587;
int32_t x1612 = x1610 % 9;
int32_t x1613 = x1612 / 3;
int32_t x1619 = x1613 * 3;
int32_t x1620 = x1619 * x1587;
int32_t x1621 = x1620 * x1587;
int32_t x1622 = x1618 + x1621;
int32_t x1614 = x1612 % 3;
int32_t x1623 = x1614 * x1587;
int32_t x1624 = x1623 * x1587;
int32_t x1625 = x1622 + x1624;
float* x1626 = x1606+x1625;
int32_t x1627 = x1611 * x1495;
int32_t x1628 = x1627 * x1495;
float* x1629 = x1602+x1628;
int32_t x1642 = 1 - x1614;
bool x1643 = x1642 > 0;
int32_t x1644;
if (x1643) {
x1644 = x1642;
} else {
x1644 = 0;
}
int32_t x1645 = 3 - x1614;
int32_t x1646 = x1645 - 1;
int32_t x1647 = 1 - x1646;
bool x1648 = x1647 > 0;
int32_t x1649;
if (x1648) {
x1649 = x1647;
} else {
x1649 = 0;
}
int32_t x1650 = x1587 - x1649;
int32_t x1651 = x1650 - x1644;
bool x1652 = x1651 <= 0;
bool x1656 = x1644 > 0;
int32_t x1641 = -1 + x1614;
bool x1669 = x1649 > 0;
for(int x1631=0; x1631 < x1587; x1631++) {
int32_t x1632 = x1631 - 1;
int32_t x1633 = x1632 + x1613;
bool x1634 = x1633 < 0;
bool x1635 = x1633 >= x1495;
bool x1636 = x1634 || x1635;
if (x1636) {
int32_t x1637 = x1631 * x1587;
float* x1638 = x1626+x1637;
memset(x1638, 0, 4 * x1587);;
} else {
if (x1652) {
int32_t x1637 = x1631 * x1587;
float* x1653 = x1626+x1637;
memset(x1653, 0, 4 * x1587);;
} else {
int32_t x1637 = x1631 * x1587;
if (x1656) {
float* x1657 = x1626+x1637;
memset(x1657, 0, 4 * x1644);;
} else {
}
// may have segfault here
int32_t x1662 = x1637 + x1644;
float* x1663 = x1626+x1662;
int32_t x1664 = x1633 * x1495;
int32_t x1665 = x1664 + x1641;
int32_t x1666 = x1665 + x1644;
float* x1667 = x1629+x1666;
memcpy(x1663, x1667, 4 * x1651);;
if (x1669) {
int32_t x1670 = x1637 + x1587;
int32_t x1671 = x1670 - x1649;
float* x1672 = x1626+x1671;
memset(x1672, 0, 4 * x1649);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1588,x1594,1,x90,x1594,x1606,x1588,1,x1604,x1588);

}
int32_t x1687 = 0;
int32_t x1688 = 1;
x1688 *= 1;
x1687 += 1;
x1688 *= 1;
x1688 *= 1;
int32_t x1693 = x1687;
bool x1694 = x1693 >= 2;
if (x1694) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1699 = x1693 == 0;
if (x1699) {
int32_t x1700 = x1688;
bool x1701 = x1700 == 64;
if (x1701) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1708 = x1688;
int32_t x1709 = 64 / x1708;
bool x1713;
if (x452) {
bool x1710 = x1709 == 1;
bool x1711 = 64 == x1709;
bool x1712 = x1710 || x1711;
x1713 = x1712;
} else {
x1713 = false;
}
bool x1717;
if (x1713) {
x1717 = x1716;
} else {
x1717 = false;
}
bool x1718;
if (x1717) {
x1718 = x1716;
} else {
x1718 = false;
}
if (x1718) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x1587,x1587,1,x1709,1,1);
assert(false && "");
}
bool x1724 = 64 <= x1709;
int32_t x1725;
if (x1724) {
x1725 = x1709;
} else {
x1725 = 64;
}
bool x1731 = x1725 > 0;
bool x1733;
if (x1731) {
x1733 = x1732;
} else {
x1733 = false;
}
bool x1734;
if (x1733) {
x1734 = x1732;
} else {
x1734 = false;
}
if (x1734) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(64) x Sym(1587) x Sym(1587)"," x Const(1) x Sym(1709) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x1729 = x1725 * x1728;
int32_t x1730 = 64 * x1729;
float* x1740 = (float*)myMalloc(x1730 * sizeof(float));;
int32_t x1741 = 0;
int32_t x1742 = 0;
int32_t x1743 = 0;
bool x1790 = x1709 > 1;
for(int x1744=0; x1744 < 64; x1744++) {
int32_t x1745 = x1742;
int32_t x1746 = x1743;
int32_t x1747 = x1741;
int32_t x1748 = x1747;
int32_t x1749 = x1745;
int32_t x1750 = x1746;
for(int x1752=0; x1752 < x1725; x1752++) {
int32_t x1753 = x1749;
int32_t x1754 = x1750;
int32_t x1755 = x1748;
int32_t x1756 = x1755;
int32_t x1757 = x1753;
int32_t x1758 = x1754;
for(int x1760=0; x1760 < x1727; x1760++) {
int32_t x1761 = x1757;
int32_t x1762 = x1758;
int32_t x1763 = x1756;
int32_t x1764 = x1763;
int32_t x1765 = x1761;
int32_t x1766 = x1762;
for(int x1767=0; x1767 < x1727; x1767++) {
int32_t x1768 = x1764;
int32_t x1769 = x1765;
float x1770 = x1593[x1769];
int32_t x1771 = x1766;
float x1772 = x105[x1771];
float x1773 = x1770 - x1772;
x1740[x1768] = x1773;
x1764 += 1;
if (x1776) {
x1765 += 1;
} else {
}

}
x1756 += x1727;
if (x1776) {
x1757 += x1587;
} else {
}

}
x1748 += x1728;
x1749 += x1588;
if (x1790) {
x1750 += 1;
} else {
}

}
x1741 += x1729;
x1742 += x1589;

}
float* x1800 = (float*)myMalloc(64 * sizeof(float));;
for(int x1801=0; x1801 < 64; x1801++) {
float x1802 = x158[x1801];
float x1803 = x1802 + 1.0E-5f;
x1800[x1801] = x1803;

}
float* x1807 = (float*)myMalloc(64 * sizeof(float));;
for(int x1808=0; x1808 < 64; x1808++) {
float x1809 = x1800[x1808];
double x1810 = (double)x1809;
double x1811 = sqrt(x1810);
float x1812 = (float)x1811;
x1807[x1808] = x1812;

}
int32_t x1816 = 0;
int32_t x1817 = 1;
x1817 *= 1;
x1816 += 1;
x1817 *= 1;
x1817 *= 1;
int32_t x1822 = x1816;
bool x1823 = x1822 >= 2;
if (x1823) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1828 = x1822 == 0;
if (x1828) {
int32_t x1829 = x1817;
bool x1830 = x1829 == 64;
if (x1830) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1837 = x1817;
int32_t x1838 = 64 / x1837;
bool x1844;
if (x452) {
bool x1839 = x1725 == 1;
bool x1840 = x1838 == 1;
bool x1841 = x1839 || x1840;
bool x1842 = x1725 == x1838;
bool x1843 = x1841 || x1842;
x1844 = x1843;
} else {
x1844 = false;
}
bool x1848;
if (x1844) {
x1848 = x1847;
} else {
x1848 = false;
}
bool x1849;
if (x1848) {
x1849 = x1847;
} else {
x1849 = false;
}
if (x1849) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1725,x1727,x1727,1,x1838,1,1);
assert(false && "");
}
bool x1855 = x1725 <= x1838;
int32_t x1856;
if (x1855) {
x1856 = x1838;
} else {
x1856 = x1725;
}
bool x1862 = x1856 > 0;
bool x1864;
if (x1862) {
x1864 = x1863;
} else {
x1864 = false;
}
bool x1865;
if (x1864) {
x1865 = x1863;
} else {
x1865 = false;
}
if (x1865) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(1725) x Sym(1727) x Sym(1727)"," x Const(1) x Sym(1838) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x1860 = x1856 * x1859;
int32_t x1861 = 64 * x1860;
float* x1871 = (float*)myMalloc(x1861 * sizeof(float));;
int32_t x1872 = 0;
int32_t x1873 = 0;
int32_t x1874 = 0;
bool x1920 = x1725 > 1;
bool x1924 = x1838 > 1;
for(int x1875=0; x1875 < 64; x1875++) {
int32_t x1876 = x1873;
int32_t x1877 = x1874;
int32_t x1878 = x1872;
int32_t x1879 = x1878;
int32_t x1880 = x1876;
int32_t x1881 = x1877;
for(int x1883=0; x1883 < x1856; x1883++) {
int32_t x1884 = x1880;
int32_t x1885 = x1881;
int32_t x1886 = x1879;
int32_t x1887 = x1886;
int32_t x1888 = x1884;
int32_t x1889 = x1885;
for(int x1891=0; x1891 < x1858; x1891++) {
int32_t x1892 = x1888;
int32_t x1893 = x1889;
int32_t x1894 = x1887;
int32_t x1895 = x1894;
int32_t x1896 = x1892;
int32_t x1897 = x1893;
for(int x1898=0; x1898 < x1858; x1898++) {
int32_t x1899 = x1895;
int32_t x1900 = x1896;
float x1901 = x1740[x1900];
int32_t x1902 = x1897;
float x1903 = x1807[x1902];
float x1904 = x1901 / x1903;
x1871[x1899] = x1904;
x1895 += 1;
if (x1907) {
x1896 += 1;
} else {
}

}
x1887 += x1858;
if (x1907) {
x1888 += x1727;
} else {
}

}
x1879 += x1859;
if (x1920) {
x1880 += x1728;
} else {
}
if (x1924) {
x1881 += 1;
} else {
}

}
x1872 += x1860;
x1873 += x1729;

}
int32_t x1934 = 0;
int32_t x1935 = 1;
x1935 *= 1;
x1934 += 1;
x1935 *= 1;
x1935 *= 1;
int32_t x1940 = x1934;
bool x1941 = x1940 >= 2;
if (x1941) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1946 = x1940 == 0;
if (x1946) {
int32_t x1947 = x1935;
bool x1948 = x1947 == 64;
if (x1948) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1955 = x1935;
int32_t x1956 = 64 / x1955;
bool x1962;
if (x452) {
bool x1957 = x1856 == 1;
bool x1958 = x1956 == 1;
bool x1959 = x1957 || x1958;
bool x1960 = x1856 == x1956;
bool x1961 = x1959 || x1960;
x1962 = x1961;
} else {
x1962 = false;
}
bool x1966;
if (x1962) {
x1966 = x1965;
} else {
x1966 = false;
}
bool x1967;
if (x1966) {
x1967 = x1965;
} else {
x1967 = false;
}
if (x1967) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1856,x1858,x1858,1,x1956,1,1);
assert(false && "");
}
bool x1973 = x1856 <= x1956;
int32_t x1974;
if (x1973) {
x1974 = x1956;
} else {
x1974 = x1856;
}
bool x1980 = x1974 > 0;
bool x1982;
if (x1980) {
x1982 = x1981;
} else {
x1982 = false;
}
bool x1983;
if (x1982) {
x1983 = x1981;
} else {
x1983 = false;
}
if (x1983) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(1856) x Sym(1858) x Sym(1858)"," x Const(1) x Sym(1956) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x1978 = x1974 * x1977;
int32_t x1979 = 64 * x1978;
float* x1989 = (float*)myMalloc(x1979 * sizeof(float));;
int32_t x1990 = 0;
int32_t x1991 = 0;
int32_t x1992 = 0;
bool x2038 = x1856 > 1;
bool x2042 = x1956 > 1;
for(int x1993=0; x1993 < 64; x1993++) {
int32_t x1994 = x1991;
int32_t x1995 = x1992;
int32_t x1996 = x1990;
int32_t x1997 = x1996;
int32_t x1998 = x1994;
int32_t x1999 = x1995;
for(int x2001=0; x2001 < x1974; x2001++) {
int32_t x2002 = x1998;
int32_t x2003 = x1999;
int32_t x2004 = x1997;
int32_t x2005 = x2004;
int32_t x2006 = x2002;
int32_t x2007 = x2003;
for(int x2009=0; x2009 < x1976; x2009++) {
int32_t x2010 = x2006;
int32_t x2011 = x2007;
int32_t x2012 = x2005;
int32_t x2013 = x2012;
int32_t x2014 = x2010;
int32_t x2015 = x2011;
for(int x2016=0; x2016 < x1976; x2016++) {
int32_t x2017 = x2013;
int32_t x2018 = x2014;
float x2019 = x1871[x2018];
int32_t x2020 = x2015;
float x2021 = x164[x2020];
float x2022 = x2019 * x2021;
x1989[x2017] = x2022;
x2013 += 1;
if (x2025) {
x2014 += 1;
} else {
}

}
x2005 += x1976;
if (x2025) {
x2006 += x1858;
} else {
}

}
x1997 += x1977;
if (x2038) {
x1998 += x1859;
} else {
}
if (x2042) {
x1999 += 1;
} else {
}

}
x1990 += x1978;
x1991 += x1860;

}
int32_t x2052 = 0;
int32_t x2053 = 1;
x2053 *= 1;
x2052 += 1;
x2053 *= 1;
x2053 *= 1;
int32_t x2058 = x2052;
bool x2059 = x2058 >= 2;
if (x2059) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2064 = x2058 == 0;
if (x2064) {
int32_t x2065 = x2053;
bool x2066 = x2065 == 64;
if (x2066) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2073 = x2053;
int32_t x2074 = 64 / x2073;
bool x2080;
if (x452) {
bool x2075 = x1974 == 1;
bool x2076 = x2074 == 1;
bool x2077 = x2075 || x2076;
bool x2078 = x1974 == x2074;
bool x2079 = x2077 || x2078;
x2080 = x2079;
} else {
x2080 = false;
}
bool x2084;
if (x2080) {
x2084 = x2083;
} else {
x2084 = false;
}
bool x2085;
if (x2084) {
x2085 = x2083;
} else {
x2085 = false;
}
if (x2085) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x1974,x1976,x1976,1,x2074,1,1);
assert(false && "");
}
bool x2091 = x1974 <= x2074;
int32_t x2092;
if (x2091) {
x2092 = x2074;
} else {
x2092 = x1974;
}
bool x2098 = x2092 > 0;
bool x2100;
if (x2098) {
x2100 = x2099;
} else {
x2100 = false;
}
bool x2101;
if (x2100) {
x2101 = x2099;
} else {
x2101 = false;
}
if (x2101) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(1974) x Sym(1976) x Sym(1976)"," x Const(1) x Sym(2074) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x2096 = x2092 * x2095;
int32_t x2097 = 64 * x2096;
float* x2107 = (float*)myMalloc(x2097 * sizeof(float));;
int32_t x2108 = 0;
int32_t x2109 = 0;
int32_t x2110 = 0;
bool x2156 = x1974 > 1;
bool x2160 = x2074 > 1;
for(int x2111=0; x2111 < 64; x2111++) {
int32_t x2112 = x2109;
int32_t x2113 = x2110;
int32_t x2114 = x2108;
int32_t x2115 = x2114;
int32_t x2116 = x2112;
int32_t x2117 = x2113;
for(int x2119=0; x2119 < x2092; x2119++) {
int32_t x2120 = x2116;
int32_t x2121 = x2117;
int32_t x2122 = x2115;
int32_t x2123 = x2122;
int32_t x2124 = x2120;
int32_t x2125 = x2121;
for(int x2127=0; x2127 < x2094; x2127++) {
int32_t x2128 = x2124;
int32_t x2129 = x2125;
int32_t x2130 = x2123;
int32_t x2131 = x2130;
int32_t x2132 = x2128;
int32_t x2133 = x2129;
for(int x2134=0; x2134 < x2094; x2134++) {
int32_t x2135 = x2131;
int32_t x2136 = x2132;
float x2137 = x1989[x2136];
int32_t x2138 = x2133;
float x2139 = x49[x2138];
float x2140 = x2137 + x2139;
x2107[x2135] = x2140;
x2131 += 1;
if (x2143) {
x2132 += 1;
} else {
}

}
x2123 += x2094;
if (x2143) {
x2124 += x1976;
} else {
}

}
x2115 += x2095;
if (x2156) {
x2116 += x1977;
} else {
}
if (x2160) {
x2117 += 1;
} else {
}

}
x2108 += x2096;
x2109 += x1978;

}
float* x2170 = (float*)myMalloc(x2097 * sizeof(float));;
for(int x2172=0; x2172 < x2097; x2172++) {
float x2173 = x2107[x2172];
bool x2174 = x2173 < 0.0f;
if (x2174) {
x2170[x2172] = 0.0f;
} else {
float x2177 = x2107[x2172];
x2170[x2172] = x2177;
}

}
float* x2191 = (float*)myMalloc(x2190 * sizeof(float));;
int32_t x2194 = 64 * x2092;
int32_t x2195 = x2194 * x2186;
float* x2196 = (float*)myMalloc(x2195 * sizeof(float));;
int32_t x2192 = x2092 * x2186;
for(int x2197=0; x2197 < 64; x2197++) {
int32_t x2198 = x2197 * x2096;
float* x2199 = x2170+x2198;
int32_t x2200 = x2197 * x2187;
float* x2201 = x2191+x2200;
int32_t x2202 = x2197 * x2192;
float* x2203 = x2196+x2202;
for(int x2204=0; x2204 < x2092; x2204++) {
int32_t x2205 = x2204 / 1;
int32_t x2209 = x2205 * x2185;
int32_t x2210 = x2209 * x2185;
int32_t x2206 = x2204 % 1;
int32_t x2207 = x2206 / 1;
int32_t x2211 = x2207 * x2185;
int32_t x2212 = x2211 * x2185;
int32_t x2213 = x2210 + x2212;
int32_t x2208 = x2206 % 1;
int32_t x2214 = x2208 * x2185;
int32_t x2215 = x2214 * x2185;
int32_t x2216 = x2213 + x2215;
float* x2217 = x2203+x2216;
int32_t x2218 = x2205 * x2094;
int32_t x2219 = x2218 * x2094;
float* x2220 = x2199+x2219;
for(int x2222=0; x2222 < x2185; x2222++) {
int32_t x2224 = x2222 * x2185;
float* x2225 = x2217+x2224;
int32_t x2223 = x2222 + x2207;
int32_t x2226 = x2223 * x2094;
int32_t x2227 = x2226 + x2208;
float* x2228 = x2220+x2227;
memcpy(x2225, x2228, 4 * x2185);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x2186,x2092,1,x32,x2092,x2203,x2186,1,x2201,x2186);

}
int32_t x2237 = 0;
int32_t x2238 = 1;
x2238 *= 1;
x2237 += 1;
x2238 *= 1;
x2238 *= 1;
int32_t x2243 = x2237;
bool x2244 = x2243 >= 2;
if (x2244) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2249 = x2243 == 0;
if (x2249) {
int32_t x2250 = x2238;
bool x2251 = x2250 == 256;
if (x2251) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2258 = x2238;
int32_t x2259 = 256 / x2258;
bool x2263;
if (x452) {
bool x2260 = x2259 == 1;
bool x2261 = 256 == x2259;
bool x2262 = x2260 || x2261;
x2263 = x2262;
} else {
x2263 = false;
}
bool x2267;
if (x2263) {
x2267 = x2266;
} else {
x2267 = false;
}
bool x2268;
if (x2267) {
x2268 = x2266;
} else {
x2268 = false;
}
if (x2268) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x2185,x2185,1,x2259,1,1);
assert(false && "");
}
bool x2274 = 256 <= x2259;
int32_t x2275;
if (x2274) {
x2275 = x2259;
} else {
x2275 = 256;
}
bool x2281 = x2275 > 0;
bool x2283;
if (x2281) {
x2283 = x2282;
} else {
x2283 = false;
}
bool x2284;
if (x2283) {
x2284 = x2282;
} else {
x2284 = false;
}
if (x2284) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(2185) x Sym(2185)"," x Const(1) x Sym(2259) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x2279 = x2275 * x2278;
int32_t x2280 = 64 * x2279;
float* x2290 = (float*)myMalloc(x2280 * sizeof(float));;
int32_t x2291 = 0;
int32_t x2292 = 0;
int32_t x2293 = 0;
bool x2340 = x2259 > 1;
for(int x2294=0; x2294 < 64; x2294++) {
int32_t x2295 = x2292;
int32_t x2296 = x2293;
int32_t x2297 = x2291;
int32_t x2298 = x2297;
int32_t x2299 = x2295;
int32_t x2300 = x2296;
for(int x2302=0; x2302 < x2275; x2302++) {
int32_t x2303 = x2299;
int32_t x2304 = x2300;
int32_t x2305 = x2298;
int32_t x2306 = x2305;
int32_t x2307 = x2303;
int32_t x2308 = x2304;
for(int x2310=0; x2310 < x2277; x2310++) {
int32_t x2311 = x2307;
int32_t x2312 = x2308;
int32_t x2313 = x2306;
int32_t x2314 = x2313;
int32_t x2315 = x2311;
int32_t x2316 = x2312;
for(int x2317=0; x2317 < x2277; x2317++) {
int32_t x2318 = x2314;
int32_t x2319 = x2315;
float x2320 = x2191[x2319];
int32_t x2321 = x2316;
float x2322 = x71[x2321];
float x2323 = x2320 - x2322;
x2290[x2318] = x2323;
x2314 += 1;
if (x2326) {
x2315 += 1;
} else {
}

}
x2306 += x2277;
if (x2326) {
x2307 += x2185;
} else {
}

}
x2298 += x2278;
x2299 += x2186;
if (x2340) {
x2300 += 1;
} else {
}

}
x2291 += x2279;
x2292 += x2187;

}
float* x2350 = (float*)myMalloc(256 * sizeof(float));;
for(int x2352=0; x2352 < 256; x2352++) {
float x2353 = x36[x2352];
float x2354 = x2353 + 1.0E-5f;
x2350[x2352] = x2354;

}
float* x2358 = (float*)myMalloc(256 * sizeof(float));;
for(int x2359=0; x2359 < 256; x2359++) {
float x2360 = x2350[x2359];
double x2361 = (double)x2360;
double x2362 = sqrt(x2361);
float x2363 = (float)x2362;
x2358[x2359] = x2363;

}
int32_t x2367 = 0;
int32_t x2368 = 1;
x2368 *= 1;
x2367 += 1;
x2368 *= 1;
x2368 *= 1;
int32_t x2373 = x2367;
bool x2374 = x2373 >= 2;
if (x2374) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2379 = x2373 == 0;
if (x2379) {
int32_t x2380 = x2368;
bool x2381 = x2380 == 256;
if (x2381) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2388 = x2368;
int32_t x2389 = 256 / x2388;
bool x2395;
if (x452) {
bool x2390 = x2275 == 1;
bool x2391 = x2389 == 1;
bool x2392 = x2390 || x2391;
bool x2393 = x2275 == x2389;
bool x2394 = x2392 || x2393;
x2395 = x2394;
} else {
x2395 = false;
}
bool x2399;
if (x2395) {
x2399 = x2398;
} else {
x2399 = false;
}
bool x2400;
if (x2399) {
x2400 = x2398;
} else {
x2400 = false;
}
if (x2400) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2275,x2277,x2277,1,x2389,1,1);
assert(false && "");
}
bool x2406 = x2275 <= x2389;
int32_t x2407;
if (x2406) {
x2407 = x2389;
} else {
x2407 = x2275;
}
bool x2413 = x2407 > 0;
bool x2415;
if (x2413) {
x2415 = x2414;
} else {
x2415 = false;
}
bool x2416;
if (x2415) {
x2416 = x2414;
} else {
x2416 = false;
}
if (x2416) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(2275) x Sym(2277) x Sym(2277)"," x Const(1) x Sym(2389) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x2411 = x2407 * x2410;
int32_t x2412 = 64 * x2411;
float* x2422 = (float*)myMalloc(x2412 * sizeof(float));;
int32_t x2423 = 0;
int32_t x2424 = 0;
int32_t x2425 = 0;
bool x2471 = x2275 > 1;
bool x2475 = x2389 > 1;
for(int x2426=0; x2426 < 64; x2426++) {
int32_t x2427 = x2424;
int32_t x2428 = x2425;
int32_t x2429 = x2423;
int32_t x2430 = x2429;
int32_t x2431 = x2427;
int32_t x2432 = x2428;
for(int x2434=0; x2434 < x2407; x2434++) {
int32_t x2435 = x2431;
int32_t x2436 = x2432;
int32_t x2437 = x2430;
int32_t x2438 = x2437;
int32_t x2439 = x2435;
int32_t x2440 = x2436;
for(int x2442=0; x2442 < x2409; x2442++) {
int32_t x2443 = x2439;
int32_t x2444 = x2440;
int32_t x2445 = x2438;
int32_t x2446 = x2445;
int32_t x2447 = x2443;
int32_t x2448 = x2444;
for(int x2449=0; x2449 < x2409; x2449++) {
int32_t x2450 = x2446;
int32_t x2451 = x2447;
float x2452 = x2290[x2451];
int32_t x2453 = x2448;
float x2454 = x2358[x2453];
float x2455 = x2452 / x2454;
x2422[x2450] = x2455;
x2446 += 1;
if (x2458) {
x2447 += 1;
} else {
}

}
x2438 += x2409;
if (x2458) {
x2439 += x2277;
} else {
}

}
x2430 += x2410;
if (x2471) {
x2431 += x2278;
} else {
}
if (x2475) {
x2432 += 1;
} else {
}

}
x2423 += x2411;
x2424 += x2279;

}
int32_t x2485 = 0;
int32_t x2486 = 1;
x2486 *= 1;
x2485 += 1;
x2486 *= 1;
x2486 *= 1;
int32_t x2491 = x2485;
bool x2492 = x2491 >= 2;
if (x2492) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2497 = x2491 == 0;
if (x2497) {
int32_t x2498 = x2486;
bool x2499 = x2498 == 256;
if (x2499) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2506 = x2486;
int32_t x2507 = 256 / x2506;
bool x2513;
if (x452) {
bool x2508 = x2407 == 1;
bool x2509 = x2507 == 1;
bool x2510 = x2508 || x2509;
bool x2511 = x2407 == x2507;
bool x2512 = x2510 || x2511;
x2513 = x2512;
} else {
x2513 = false;
}
bool x2517;
if (x2513) {
x2517 = x2516;
} else {
x2517 = false;
}
bool x2518;
if (x2517) {
x2518 = x2516;
} else {
x2518 = false;
}
if (x2518) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2407,x2409,x2409,1,x2507,1,1);
assert(false && "");
}
bool x2524 = x2407 <= x2507;
int32_t x2525;
if (x2524) {
x2525 = x2507;
} else {
x2525 = x2407;
}
bool x2531 = x2525 > 0;
bool x2533;
if (x2531) {
x2533 = x2532;
} else {
x2533 = false;
}
bool x2534;
if (x2533) {
x2534 = x2532;
} else {
x2534 = false;
}
if (x2534) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(2407) x Sym(2409) x Sym(2409)"," x Const(1) x Sym(2507) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x2529 = x2525 * x2528;
int32_t x2530 = 64 * x2529;
float* x2540 = (float*)myMalloc(x2530 * sizeof(float));;
int32_t x2541 = 0;
int32_t x2542 = 0;
int32_t x2543 = 0;
bool x2589 = x2407 > 1;
bool x2593 = x2507 > 1;
for(int x2544=0; x2544 < 64; x2544++) {
int32_t x2545 = x2542;
int32_t x2546 = x2543;
int32_t x2547 = x2541;
int32_t x2548 = x2547;
int32_t x2549 = x2545;
int32_t x2550 = x2546;
for(int x2552=0; x2552 < x2525; x2552++) {
int32_t x2553 = x2549;
int32_t x2554 = x2550;
int32_t x2555 = x2548;
int32_t x2556 = x2555;
int32_t x2557 = x2553;
int32_t x2558 = x2554;
for(int x2560=0; x2560 < x2527; x2560++) {
int32_t x2561 = x2557;
int32_t x2562 = x2558;
int32_t x2563 = x2556;
int32_t x2564 = x2563;
int32_t x2565 = x2561;
int32_t x2566 = x2562;
for(int x2567=0; x2567 < x2527; x2567++) {
int32_t x2568 = x2564;
int32_t x2569 = x2565;
float x2570 = x2422[x2569];
int32_t x2571 = x2566;
float x2572 = x199[x2571];
float x2573 = x2570 * x2572;
x2540[x2568] = x2573;
x2564 += 1;
if (x2576) {
x2565 += 1;
} else {
}

}
x2556 += x2527;
if (x2576) {
x2557 += x2409;
} else {
}

}
x2548 += x2528;
if (x2589) {
x2549 += x2410;
} else {
}
if (x2593) {
x2550 += 1;
} else {
}

}
x2541 += x2529;
x2542 += x2411;

}
int32_t x2603 = 0;
int32_t x2604 = 1;
x2604 *= 1;
x2603 += 1;
x2604 *= 1;
x2604 *= 1;
int32_t x2609 = x2603;
bool x2610 = x2609 >= 2;
if (x2610) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2615 = x2609 == 0;
if (x2615) {
int32_t x2616 = x2604;
bool x2617 = x2616 == 256;
if (x2617) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2624 = x2604;
int32_t x2625 = 256 / x2624;
bool x2631;
if (x452) {
bool x2626 = x2525 == 1;
bool x2627 = x2625 == 1;
bool x2628 = x2626 || x2627;
bool x2629 = x2525 == x2625;
bool x2630 = x2628 || x2629;
x2631 = x2630;
} else {
x2631 = false;
}
bool x2635;
if (x2631) {
x2635 = x2634;
} else {
x2635 = false;
}
bool x2636;
if (x2635) {
x2636 = x2634;
} else {
x2636 = false;
}
if (x2636) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2525,x2527,x2527,1,x2625,1,1);
assert(false && "");
}
bool x2642 = x2525 <= x2625;
int32_t x2643;
if (x2642) {
x2643 = x2625;
} else {
x2643 = x2525;
}
bool x2649 = x2643 > 0;
bool x2651;
if (x2649) {
x2651 = x2650;
} else {
x2651 = false;
}
bool x2652;
if (x2651) {
x2652 = x2650;
} else {
x2652 = false;
}
if (x2652) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(2525) x Sym(2527) x Sym(2527)"," x Const(1) x Sym(2625) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x2647 = x2643 * x2646;
int32_t x2648 = 64 * x2647;
float* x2658 = (float*)myMalloc(x2648 * sizeof(float));;
int32_t x2659 = 0;
int32_t x2660 = 0;
int32_t x2661 = 0;
bool x2707 = x2525 > 1;
bool x2711 = x2625 > 1;
for(int x2662=0; x2662 < 64; x2662++) {
int32_t x2663 = x2660;
int32_t x2664 = x2661;
int32_t x2665 = x2659;
int32_t x2666 = x2665;
int32_t x2667 = x2663;
int32_t x2668 = x2664;
for(int x2670=0; x2670 < x2643; x2670++) {
int32_t x2671 = x2667;
int32_t x2672 = x2668;
int32_t x2673 = x2666;
int32_t x2674 = x2673;
int32_t x2675 = x2671;
int32_t x2676 = x2672;
for(int x2678=0; x2678 < x2645; x2678++) {
int32_t x2679 = x2675;
int32_t x2680 = x2676;
int32_t x2681 = x2674;
int32_t x2682 = x2681;
int32_t x2683 = x2679;
int32_t x2684 = x2680;
for(int x2685=0; x2685 < x2645; x2685++) {
int32_t x2686 = x2682;
int32_t x2687 = x2683;
float x2688 = x2540[x2687];
int32_t x2689 = x2684;
float x2690 = x126[x2689];
float x2691 = x2688 + x2690;
x2658[x2686] = x2691;
x2682 += 1;
if (x2694) {
x2683 += 1;
} else {
}

}
x2674 += x2645;
if (x2694) {
x2675 += x2527;
} else {
}

}
x2666 += x2646;
if (x2707) {
x2667 += x2528;
} else {
}
if (x2711) {
x2668 += 1;
} else {
}

}
x2659 += x2647;
x2660 += x2529;

}
float* x2725 = (float*)myMalloc(x2724 * sizeof(float));;
float* x2726 = (float*)myMalloc(x1046 * sizeof(float));;
for(int x2727=0; x2727 < 64; x2727++) {
int32_t x2728 = x2727 * x936;
float* x2729 = x941+x2728;
int32_t x2730 = x2727 * x2721;
float* x2731 = x2725+x2730;
int32_t x2732 = x2727 * x1044;
float* x2733 = x2726+x2732;
for(int x2734=0; x2734 < x835; x2734++) {
int32_t x2735 = x2734 / 1;
int32_t x2739 = x2735 * x1037;
int32_t x2740 = x2739 * x1037;
int32_t x2736 = x2734 % 1;
int32_t x2737 = x2736 / 1;
int32_t x2741 = x2737 * x1037;
int32_t x2742 = x2741 * x1037;
int32_t x2743 = x2740 + x2742;
int32_t x2738 = x2736 % 1;
int32_t x2744 = x2738 * x1037;
int32_t x2745 = x2744 * x1037;
int32_t x2746 = x2743 + x2745;
float* x2747 = x2733+x2746;
int32_t x2748 = x2735 * x934;
int32_t x2749 = x2748 * x934;
float* x2750 = x2729+x2749;
for(int x2751=0; x2751 < x1037; x2751++) {
int32_t x2753 = x2751 * x1037;
float* x2754 = x2747+x2753;
int32_t x2752 = x2751 + x2737;
int32_t x2755 = x2752 * x934;
int32_t x2756 = x2755 + x2738;
float* x2757 = x2750+x2756;
memcpy(x2754, x2757, 4 * x1037);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1038,x835,1,x162,x835,x2733,x1038,1,x2731,x1038);

}
int32_t x2766 = 0;
int32_t x2767 = 1;
x2767 *= 1;
x2766 += 1;
x2767 *= 1;
x2767 *= 1;
int32_t x2772 = x2766;
bool x2773 = x2772 >= 2;
if (x2773) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2778 = x2772 == 0;
if (x2778) {
int32_t x2779 = x2767;
bool x2780 = x2779 == 256;
if (x2780) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2787 = x2767;
int32_t x2788 = 256 / x2787;
bool x2792;
if (x452) {
bool x2789 = x2788 == 1;
bool x2790 = 256 == x2788;
bool x2791 = x2789 || x2790;
x2792 = x2791;
} else {
x2792 = false;
}
bool x2793;
if (x2792) {
x2793 = x1117;
} else {
x2793 = false;
}
bool x2794;
if (x2793) {
x2794 = x1117;
} else {
x2794 = false;
}
if (x2794) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x1037,x1037,1,x2788,1,1);
assert(false && "");
}
bool x2800 = 256 <= x2788;
int32_t x2801;
if (x2800) {
x2801 = x2788;
} else {
x2801 = 256;
}
bool x2804 = x2801 > 0;
bool x2805;
if (x2804) {
x2805 = x1133;
} else {
x2805 = false;
}
bool x2806;
if (x2805) {
x2806 = x1133;
} else {
x2806 = false;
}
if (x2806) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(1037) x Sym(1037)"," x Const(1) x Sym(2788) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x2802 = x2801 * x1129;
int32_t x2803 = 64 * x2802;
float* x2812 = (float*)myMalloc(x2803 * sizeof(float));;
int32_t x2813 = 0;
int32_t x2814 = 0;
int32_t x2815 = 0;
bool x2860 = x2788 > 1;
for(int x2816=0; x2816 < 64; x2816++) {
int32_t x2817 = x2814;
int32_t x2818 = x2815;
int32_t x2819 = x2813;
int32_t x2820 = x2819;
int32_t x2821 = x2817;
int32_t x2822 = x2818;
for(int x2824=0; x2824 < x2801; x2824++) {
int32_t x2825 = x2821;
int32_t x2826 = x2822;
int32_t x2827 = x2820;
int32_t x2828 = x2827;
int32_t x2829 = x2825;
int32_t x2830 = x2826;
for(int x2831=0; x2831 < x1128; x2831++) {
int32_t x2832 = x2829;
int32_t x2833 = x2830;
int32_t x2834 = x2828;
int32_t x2835 = x2834;
int32_t x2836 = x2832;
int32_t x2837 = x2833;
for(int x2838=0; x2838 < x1128; x2838++) {
int32_t x2839 = x2835;
int32_t x2840 = x2836;
float x2841 = x2725[x2840];
int32_t x2842 = x2837;
float x2843 = x264[x2842];
float x2844 = x2841 - x2843;
x2812[x2839] = x2844;
x2835 += 1;
if (x1177) {
x2836 += 1;
} else {
}

}
x2828 += x1128;
if (x1177) {
x2829 += x1037;
} else {
}

}
x2820 += x1129;
x2821 += x1038;
if (x2860) {
x2822 += 1;
} else {
}

}
x2813 += x2802;
x2814 += x2721;

}
float* x2870 = (float*)myMalloc(256 * sizeof(float));;
for(int x2871=0; x2871 < 256; x2871++) {
float x2872 = x243[x2871];
float x2873 = x2872 + 1.0E-5f;
x2870[x2871] = x2873;

}
float* x2877 = (float*)myMalloc(256 * sizeof(float));;
for(int x2878=0; x2878 < 256; x2878++) {
float x2879 = x2870[x2878];
double x2880 = (double)x2879;
double x2881 = sqrt(x2880);
float x2882 = (float)x2881;
x2877[x2878] = x2882;

}
int32_t x2886 = 0;
int32_t x2887 = 1;
x2887 *= 1;
x2886 += 1;
x2887 *= 1;
x2887 *= 1;
int32_t x2892 = x2886;
bool x2893 = x2892 >= 2;
if (x2893) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x2898 = x2892 == 0;
if (x2898) {
int32_t x2899 = x2887;
bool x2900 = x2899 == 256;
if (x2900) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x2907 = x2887;
int32_t x2908 = 256 / x2907;
bool x2914;
if (x452) {
bool x2909 = x2801 == 1;
bool x2910 = x2908 == 1;
bool x2911 = x2909 || x2910;
bool x2912 = x2801 == x2908;
bool x2913 = x2911 || x2912;
x2914 = x2913;
} else {
x2914 = false;
}
bool x2915;
if (x2914) {
x2915 = x1248;
} else {
x2915 = false;
}
bool x2916;
if (x2915) {
x2916 = x1248;
} else {
x2916 = false;
}
if (x2916) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2801,x1128,x1128,1,x2908,1,1);
assert(false && "");
}
bool x2922 = x2801 <= x2908;
int32_t x2923;
if (x2922) {
x2923 = x2908;
} else {
x2923 = x2801;
}
bool x2926 = x2923 > 0;
bool x2927;
if (x2926) {
x2927 = x1264;
} else {
x2927 = false;
}
bool x2928;
if (x2927) {
x2928 = x1264;
} else {
x2928 = false;
}
if (x2928) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(2801) x Sym(1128) x Sym(1128)"," x Const(1) x Sym(2908) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x2924 = x2923 * x1260;
int32_t x2925 = 64 * x2924;
float* x2934 = (float*)myMalloc(x2925 * sizeof(float));;
int32_t x2935 = 0;
int32_t x2936 = 0;
int32_t x2937 = 0;
bool x2981 = x2801 > 1;
bool x2985 = x2908 > 1;
for(int x2938=0; x2938 < 64; x2938++) {
int32_t x2939 = x2936;
int32_t x2940 = x2937;
int32_t x2941 = x2935;
int32_t x2942 = x2941;
int32_t x2943 = x2939;
int32_t x2944 = x2940;
for(int x2946=0; x2946 < x2923; x2946++) {
int32_t x2947 = x2943;
int32_t x2948 = x2944;
int32_t x2949 = x2942;
int32_t x2950 = x2949;
int32_t x2951 = x2947;
int32_t x2952 = x2948;
for(int x2953=0; x2953 < x1259; x2953++) {
int32_t x2954 = x2951;
int32_t x2955 = x2952;
int32_t x2956 = x2950;
int32_t x2957 = x2956;
int32_t x2958 = x2954;
int32_t x2959 = x2955;
for(int x2960=0; x2960 < x1259; x2960++) {
int32_t x2961 = x2957;
int32_t x2962 = x2958;
float x2963 = x2812[x2962];
int32_t x2964 = x2959;
float x2965 = x2877[x2964];
float x2966 = x2963 / x2965;
x2934[x2961] = x2966;
x2957 += 1;
if (x1308) {
x2958 += 1;
} else {
}

}
x2950 += x1259;
if (x1308) {
x2951 += x1128;
} else {
}

}
x2942 += x1260;
if (x2981) {
x2943 += x1129;
} else {
}
if (x2985) {
x2944 += 1;
} else {
}

}
x2935 += x2924;
x2936 += x2802;

}
int32_t x2995 = 0;
int32_t x2996 = 1;
x2996 *= 1;
x2995 += 1;
x2996 *= 1;
x2996 *= 1;
int32_t x3001 = x2995;
bool x3002 = x3001 >= 2;
if (x3002) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3007 = x3001 == 0;
if (x3007) {
int32_t x3008 = x2996;
bool x3009 = x3008 == 256;
if (x3009) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3016 = x2996;
int32_t x3017 = 256 / x3016;
bool x3023;
if (x452) {
bool x3018 = x2923 == 1;
bool x3019 = x3017 == 1;
bool x3020 = x3018 || x3019;
bool x3021 = x2923 == x3017;
bool x3022 = x3020 || x3021;
x3023 = x3022;
} else {
x3023 = false;
}
bool x3024;
if (x3023) {
x3024 = x1366;
} else {
x3024 = false;
}
bool x3025;
if (x3024) {
x3025 = x1366;
} else {
x3025 = false;
}
if (x3025) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2923,x1259,x1259,1,x3017,1,1);
assert(false && "");
}
bool x3031 = x2923 <= x3017;
int32_t x3032;
if (x3031) {
x3032 = x3017;
} else {
x3032 = x2923;
}
bool x3035 = x3032 > 0;
bool x3036;
if (x3035) {
x3036 = x1382;
} else {
x3036 = false;
}
bool x3037;
if (x3036) {
x3037 = x1382;
} else {
x3037 = false;
}
if (x3037) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(2923) x Sym(1259) x Sym(1259)"," x Const(1) x Sym(3017) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x3033 = x3032 * x1378;
int32_t x3034 = 64 * x3033;
float* x3043 = (float*)myMalloc(x3034 * sizeof(float));;
int32_t x3044 = 0;
int32_t x3045 = 0;
int32_t x3046 = 0;
bool x3090 = x2923 > 1;
bool x3094 = x3017 > 1;
for(int x3047=0; x3047 < 64; x3047++) {
int32_t x3048 = x3045;
int32_t x3049 = x3046;
int32_t x3050 = x3044;
int32_t x3051 = x3050;
int32_t x3052 = x3048;
int32_t x3053 = x3049;
for(int x3055=0; x3055 < x3032; x3055++) {
int32_t x3056 = x3052;
int32_t x3057 = x3053;
int32_t x3058 = x3051;
int32_t x3059 = x3058;
int32_t x3060 = x3056;
int32_t x3061 = x3057;
for(int x3062=0; x3062 < x1377; x3062++) {
int32_t x3063 = x3060;
int32_t x3064 = x3061;
int32_t x3065 = x3059;
int32_t x3066 = x3065;
int32_t x3067 = x3063;
int32_t x3068 = x3064;
for(int x3069=0; x3069 < x1377; x3069++) {
int32_t x3070 = x3066;
int32_t x3071 = x3067;
float x3072 = x2934[x3071];
int32_t x3073 = x3068;
float x3074 = x76[x3073];
float x3075 = x3072 * x3074;
x3043[x3070] = x3075;
x3066 += 1;
if (x1426) {
x3067 += 1;
} else {
}

}
x3059 += x1377;
if (x1426) {
x3060 += x1259;
} else {
}

}
x3051 += x1378;
if (x3090) {
x3052 += x1260;
} else {
}
if (x3094) {
x3053 += 1;
} else {
}

}
x3044 += x3033;
x3045 += x2924;

}
int32_t x3104 = 0;
int32_t x3105 = 1;
x3105 *= 1;
x3104 += 1;
x3105 *= 1;
x3105 *= 1;
int32_t x3110 = x3104;
bool x3111 = x3110 >= 2;
if (x3111) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3116 = x3110 == 0;
if (x3116) {
int32_t x3117 = x3105;
bool x3118 = x3117 == 256;
if (x3118) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3125 = x3105;
int32_t x3126 = 256 / x3125;
bool x3132;
if (x452) {
bool x3127 = x3032 == 1;
bool x3128 = x3126 == 1;
bool x3129 = x3127 || x3128;
bool x3130 = x3032 == x3126;
bool x3131 = x3129 || x3130;
x3132 = x3131;
} else {
x3132 = false;
}
bool x3133;
if (x3132) {
x3133 = x1484;
} else {
x3133 = false;
}
bool x3134;
if (x3133) {
x3134 = x1484;
} else {
x3134 = false;
}
if (x3134) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3032,x1377,x1377,1,x3126,1,1);
assert(false && "");
}
bool x3140 = x3032 <= x3126;
int32_t x3141;
if (x3140) {
x3141 = x3126;
} else {
x3141 = x3032;
}
bool x3144 = x3141 > 0;
bool x3145;
if (x3144) {
x3145 = x1500;
} else {
x3145 = false;
}
bool x3146;
if (x3145) {
x3146 = x1500;
} else {
x3146 = false;
}
if (x3146) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(3032) x Sym(1377) x Sym(1377)"," x Const(1) x Sym(3126) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x3142 = x3141 * x1496;
int32_t x3143 = 64 * x3142;
float* x3152 = (float*)myMalloc(x3143 * sizeof(float));;
int32_t x3153 = 0;
int32_t x3154 = 0;
int32_t x3155 = 0;
bool x3199 = x3032 > 1;
bool x3203 = x3126 > 1;
for(int x3156=0; x3156 < 64; x3156++) {
int32_t x3157 = x3154;
int32_t x3158 = x3155;
int32_t x3159 = x3153;
int32_t x3160 = x3159;
int32_t x3161 = x3157;
int32_t x3162 = x3158;
for(int x3164=0; x3164 < x3141; x3164++) {
int32_t x3165 = x3161;
int32_t x3166 = x3162;
int32_t x3167 = x3160;
int32_t x3168 = x3167;
int32_t x3169 = x3165;
int32_t x3170 = x3166;
for(int x3171=0; x3171 < x1495; x3171++) {
int32_t x3172 = x3169;
int32_t x3173 = x3170;
int32_t x3174 = x3168;
int32_t x3175 = x3174;
int32_t x3176 = x3172;
int32_t x3177 = x3173;
for(int x3178=0; x3178 < x1495; x3178++) {
int32_t x3179 = x3175;
int32_t x3180 = x3176;
float x3181 = x3043[x3180];
int32_t x3182 = x3177;
float x3183 = x203[x3182];
float x3184 = x3181 + x3183;
x3152[x3179] = x3184;
x3175 += 1;
if (x1544) {
x3176 += 1;
} else {
}

}
x3168 += x1495;
if (x1544) {
x3169 += x1377;
} else {
}

}
x3160 += x1496;
if (x3199) {
x3161 += x1378;
} else {
}
if (x3203) {
x3162 += 1;
} else {
}

}
x3153 += x3142;
x3154 += x3033;

}
bool x3213 = x2643 == 1;
bool x3214 = x3141 == 1;
bool x3215 = x3213 || x3214;
bool x3216 = x2643 == x3141;
bool x3217 = x3215 || x3216;
bool x3223;
if (x3217) {
x3223 = x3222;
} else {
x3223 = false;
}
bool x3224;
if (x3223) {
x3224 = x3222;
} else {
x3224 = false;
}
if (x3224) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x2643,x2645,x2645,64,x3141,x1495,x1495);
assert(false && "");
}
int32_t x3237 = 0;
int32_t x3238 = 0;
int32_t x3239 = 0;
bool x3230 = x2643 <= x3141;
int32_t x3231;
if (x3230) {
x3231 = x3141;
} else {
x3231 = x2643;
}
bool x3291 = x2643 > 1;
bool x3295 = x3141 > 1;
int32_t x3235 = x3231 * x3234;
for(int x3240=0; x3240 < 64; x3240++) {
int32_t x3241 = x3238;
int32_t x3242 = x3239;
int32_t x3243 = x3237;
int32_t x3244 = x3243;
int32_t x3245 = x3241;
int32_t x3246 = x3242;
for(int x3248=0; x3248 < x3231; x3248++) {
int32_t x3249 = x3245;
int32_t x3250 = x3246;
int32_t x3251 = x3244;
int32_t x3252 = x3251;
int32_t x3253 = x3249;
int32_t x3254 = x3250;
for(int x3256=0; x3256 < x3233; x3256++) {
int32_t x3257 = x3253;
int32_t x3258 = x3254;
int32_t x3259 = x3252;
int32_t x3260 = x3259;
int32_t x3261 = x3257;
int32_t x3262 = x3258;
for(int x3263=0; x3263 < x3233; x3263++) {
int32_t x3264 = x3261;
float x3265 = x2658[x3264];
int32_t x3266 = x3262;
float x3267 = x3152[x3266];
float x3268 = x3265 + x3267;
x2658[x3264] = x3268;
x3260 += 1;
if (x3271) {
x3261 += 1;
} else {
}
if (x3275) {
x3262 += 1;
} else {
}

}
x3252 += x3233;
if (x3271) {
x3253 += x2645;
} else {
}
if (x3275) {
x3254 += x1495;
} else {
}

}
x3244 += x3234;
if (x3291) {
x3245 += x2646;
} else {
}
if (x3295) {
x3246 += x1496;
} else {
}

}
x3237 += x3235;
x3238 += x2647;
x3239 += x3142;

}
float* x3306 = (float*)myMalloc(x2648 * sizeof(float));;
for(int x3308=0; x3308 < x2648; x3308++) {
float x3309 = x2658[x3308];
bool x3310 = x3309 < 0.0f;
if (x3310) {
x3306[x3308] = 0.0f;
} else {
float x3313 = x2658[x3308];
x3306[x3308] = x3313;
}

}
float* x3327 = (float*)myMalloc(x3326 * sizeof(float));;
int32_t x3330 = 64 * x2643;
int32_t x3331 = x3330 * x3322;
float* x3332 = (float*)myMalloc(x3331 * sizeof(float));;
int32_t x3328 = x2643 * x3322;
for(int x3333=0; x3333 < 64; x3333++) {
int32_t x3334 = x3333 * x2647;
float* x3335 = x3306+x3334;
int32_t x3336 = x3333 * x3323;
float* x3337 = x3327+x3336;
int32_t x3338 = x3333 * x3328;
float* x3339 = x3332+x3338;
for(int x3340=0; x3340 < x2643; x3340++) {
int32_t x3341 = x3340 / 1;
int32_t x3345 = x3341 * x3321;
int32_t x3346 = x3345 * x3321;
int32_t x3342 = x3340 % 1;
int32_t x3343 = x3342 / 1;
int32_t x3347 = x3343 * x3321;
int32_t x3348 = x3347 * x3321;
int32_t x3349 = x3346 + x3348;
int32_t x3344 = x3342 % 1;
int32_t x3350 = x3344 * x3321;
int32_t x3351 = x3350 * x3321;
int32_t x3352 = x3349 + x3351;
float* x3353 = x3339+x3352;
int32_t x3354 = x3341 * x2645;
int32_t x3355 = x3354 * x2645;
float* x3356 = x3335+x3355;
for(int x3358=0; x3358 < x3321; x3358++) {
int32_t x3360 = x3358 * x3321;
float* x3361 = x3353+x3360;
int32_t x3359 = x3358 + x3343;
int32_t x3362 = x3359 * x2645;
int32_t x3363 = x3362 + x3344;
float* x3364 = x3356+x3363;
memcpy(x3361, x3364, 4 * x3321);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x3322,x2643,1,x171,x2643,x3339,x3322,1,x3337,x3322);

}
int32_t x3373 = 0;
int32_t x3374 = 1;
x3374 *= 1;
x3373 += 1;
x3374 *= 1;
x3374 *= 1;
int32_t x3379 = x3373;
bool x3380 = x3379 >= 2;
if (x3380) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3385 = x3379 == 0;
if (x3385) {
int32_t x3386 = x3374;
bool x3387 = x3386 == 64;
if (x3387) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3394 = x3374;
int32_t x3395 = 64 / x3394;
bool x3399;
if (x452) {
bool x3396 = x3395 == 1;
bool x3397 = 64 == x3395;
bool x3398 = x3396 || x3397;
x3399 = x3398;
} else {
x3399 = false;
}
bool x3403;
if (x3399) {
x3403 = x3402;
} else {
x3403 = false;
}
bool x3404;
if (x3403) {
x3404 = x3402;
} else {
x3404 = false;
}
if (x3404) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x3321,x3321,1,x3395,1,1);
assert(false && "");
}
bool x3410 = 64 <= x3395;
int32_t x3411;
if (x3410) {
x3411 = x3395;
} else {
x3411 = 64;
}
bool x3417 = x3411 > 0;
bool x3419;
if (x3417) {
x3419 = x3418;
} else {
x3419 = false;
}
bool x3420;
if (x3419) {
x3420 = x3418;
} else {
x3420 = false;
}
if (x3420) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(64) x Sym(3321) x Sym(3321)"," x Const(1) x Sym(3395) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x3415 = x3411 * x3414;
int32_t x3416 = 64 * x3415;
float* x3426 = (float*)myMalloc(x3416 * sizeof(float));;
int32_t x3427 = 0;
int32_t x3428 = 0;
int32_t x3429 = 0;
bool x3476 = x3395 > 1;
for(int x3430=0; x3430 < 64; x3430++) {
int32_t x3431 = x3428;
int32_t x3432 = x3429;
int32_t x3433 = x3427;
int32_t x3434 = x3433;
int32_t x3435 = x3431;
int32_t x3436 = x3432;
for(int x3438=0; x3438 < x3411; x3438++) {
int32_t x3439 = x3435;
int32_t x3440 = x3436;
int32_t x3441 = x3434;
int32_t x3442 = x3441;
int32_t x3443 = x3439;
int32_t x3444 = x3440;
for(int x3446=0; x3446 < x3413; x3446++) {
int32_t x3447 = x3443;
int32_t x3448 = x3444;
int32_t x3449 = x3442;
int32_t x3450 = x3449;
int32_t x3451 = x3447;
int32_t x3452 = x3448;
for(int x3453=0; x3453 < x3413; x3453++) {
int32_t x3454 = x3450;
int32_t x3455 = x3451;
float x3456 = x3327[x3455];
int32_t x3457 = x3452;
float x3458 = x10[x3457];
float x3459 = x3456 - x3458;
x3426[x3454] = x3459;
x3450 += 1;
if (x3462) {
x3451 += 1;
} else {
}

}
x3442 += x3413;
if (x3462) {
x3443 += x3321;
} else {
}

}
x3434 += x3414;
x3435 += x3322;
if (x3476) {
x3436 += 1;
} else {
}

}
x3427 += x3415;
x3428 += x3323;

}
float* x3486 = (float*)myMalloc(64 * sizeof(float));;
for(int x3487=0; x3487 < 64; x3487++) {
float x3488 = x102[x3487];
float x3489 = x3488 + 1.0E-5f;
x3486[x3487] = x3489;

}
float* x3493 = (float*)myMalloc(64 * sizeof(float));;
for(int x3494=0; x3494 < 64; x3494++) {
float x3495 = x3486[x3494];
double x3496 = (double)x3495;
double x3497 = sqrt(x3496);
float x3498 = (float)x3497;
x3493[x3494] = x3498;

}
int32_t x3502 = 0;
int32_t x3503 = 1;
x3503 *= 1;
x3502 += 1;
x3503 *= 1;
x3503 *= 1;
int32_t x3508 = x3502;
bool x3509 = x3508 >= 2;
if (x3509) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3514 = x3508 == 0;
if (x3514) {
int32_t x3515 = x3503;
bool x3516 = x3515 == 64;
if (x3516) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3523 = x3503;
int32_t x3524 = 64 / x3523;
bool x3530;
if (x452) {
bool x3525 = x3411 == 1;
bool x3526 = x3524 == 1;
bool x3527 = x3525 || x3526;
bool x3528 = x3411 == x3524;
bool x3529 = x3527 || x3528;
x3530 = x3529;
} else {
x3530 = false;
}
bool x3534;
if (x3530) {
x3534 = x3533;
} else {
x3534 = false;
}
bool x3535;
if (x3534) {
x3535 = x3533;
} else {
x3535 = false;
}
if (x3535) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3411,x3413,x3413,1,x3524,1,1);
assert(false && "");
}
bool x3541 = x3411 <= x3524;
int32_t x3542;
if (x3541) {
x3542 = x3524;
} else {
x3542 = x3411;
}
bool x3548 = x3542 > 0;
bool x3550;
if (x3548) {
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
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(3411) x Sym(3413) x Sym(3413)"," x Const(1) x Sym(3524) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x3546 = x3542 * x3545;
int32_t x3547 = 64 * x3546;
float* x3557 = (float*)myMalloc(x3547 * sizeof(float));;
int32_t x3558 = 0;
int32_t x3559 = 0;
int32_t x3560 = 0;
bool x3606 = x3411 > 1;
bool x3610 = x3524 > 1;
for(int x3561=0; x3561 < 64; x3561++) {
int32_t x3562 = x3559;
int32_t x3563 = x3560;
int32_t x3564 = x3558;
int32_t x3565 = x3564;
int32_t x3566 = x3562;
int32_t x3567 = x3563;
for(int x3569=0; x3569 < x3542; x3569++) {
int32_t x3570 = x3566;
int32_t x3571 = x3567;
int32_t x3572 = x3565;
int32_t x3573 = x3572;
int32_t x3574 = x3570;
int32_t x3575 = x3571;
for(int x3577=0; x3577 < x3544; x3577++) {
int32_t x3578 = x3574;
int32_t x3579 = x3575;
int32_t x3580 = x3573;
int32_t x3581 = x3580;
int32_t x3582 = x3578;
int32_t x3583 = x3579;
for(int x3584=0; x3584 < x3544; x3584++) {
int32_t x3585 = x3581;
int32_t x3586 = x3582;
float x3587 = x3426[x3586];
int32_t x3588 = x3583;
float x3589 = x3493[x3588];
float x3590 = x3587 / x3589;
x3557[x3585] = x3590;
x3581 += 1;
if (x3593) {
x3582 += 1;
} else {
}

}
x3573 += x3544;
if (x3593) {
x3574 += x3413;
} else {
}

}
x3565 += x3545;
if (x3606) {
x3566 += x3414;
} else {
}
if (x3610) {
x3567 += 1;
} else {
}

}
x3558 += x3546;
x3559 += x3415;

}
int32_t x3620 = 0;
int32_t x3621 = 1;
x3621 *= 1;
x3620 += 1;
x3621 *= 1;
x3621 *= 1;
int32_t x3626 = x3620;
bool x3627 = x3626 >= 2;
if (x3627) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3632 = x3626 == 0;
if (x3632) {
int32_t x3633 = x3621;
bool x3634 = x3633 == 64;
if (x3634) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3641 = x3621;
int32_t x3642 = 64 / x3641;
bool x3648;
if (x452) {
bool x3643 = x3542 == 1;
bool x3644 = x3642 == 1;
bool x3645 = x3643 || x3644;
bool x3646 = x3542 == x3642;
bool x3647 = x3645 || x3646;
x3648 = x3647;
} else {
x3648 = false;
}
bool x3652;
if (x3648) {
x3652 = x3651;
} else {
x3652 = false;
}
bool x3653;
if (x3652) {
x3653 = x3651;
} else {
x3653 = false;
}
if (x3653) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3542,x3544,x3544,1,x3642,1,1);
assert(false && "");
}
bool x3659 = x3542 <= x3642;
int32_t x3660;
if (x3659) {
x3660 = x3642;
} else {
x3660 = x3542;
}
bool x3666 = x3660 > 0;
bool x3668;
if (x3666) {
x3668 = x3667;
} else {
x3668 = false;
}
bool x3669;
if (x3668) {
x3669 = x3667;
} else {
x3669 = false;
}
if (x3669) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(3542) x Sym(3544) x Sym(3544)"," x Const(1) x Sym(3642) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x3664 = x3660 * x3663;
int32_t x3665 = 64 * x3664;
float* x3675 = (float*)myMalloc(x3665 * sizeof(float));;
int32_t x3676 = 0;
int32_t x3677 = 0;
int32_t x3678 = 0;
bool x3724 = x3542 > 1;
bool x3728 = x3642 > 1;
for(int x3679=0; x3679 < 64; x3679++) {
int32_t x3680 = x3677;
int32_t x3681 = x3678;
int32_t x3682 = x3676;
int32_t x3683 = x3682;
int32_t x3684 = x3680;
int32_t x3685 = x3681;
for(int x3687=0; x3687 < x3660; x3687++) {
int32_t x3688 = x3684;
int32_t x3689 = x3685;
int32_t x3690 = x3683;
int32_t x3691 = x3690;
int32_t x3692 = x3688;
int32_t x3693 = x3689;
for(int x3695=0; x3695 < x3662; x3695++) {
int32_t x3696 = x3692;
int32_t x3697 = x3693;
int32_t x3698 = x3691;
int32_t x3699 = x3698;
int32_t x3700 = x3696;
int32_t x3701 = x3697;
for(int x3702=0; x3702 < x3662; x3702++) {
int32_t x3703 = x3699;
int32_t x3704 = x3700;
float x3705 = x3557[x3704];
int32_t x3706 = x3701;
float x3707 = x142[x3706];
float x3708 = x3705 * x3707;
x3675[x3703] = x3708;
x3699 += 1;
if (x3711) {
x3700 += 1;
} else {
}

}
x3691 += x3662;
if (x3711) {
x3692 += x3544;
} else {
}

}
x3683 += x3663;
if (x3724) {
x3684 += x3545;
} else {
}
if (x3728) {
x3685 += 1;
} else {
}

}
x3676 += x3664;
x3677 += x3546;

}
int32_t x3738 = 0;
int32_t x3739 = 1;
x3739 *= 1;
x3738 += 1;
x3739 *= 1;
x3739 *= 1;
int32_t x3744 = x3738;
bool x3745 = x3744 >= 2;
if (x3745) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3750 = x3744 == 0;
if (x3750) {
int32_t x3751 = x3739;
bool x3752 = x3751 == 64;
if (x3752) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3759 = x3739;
int32_t x3760 = 64 / x3759;
bool x3766;
if (x452) {
bool x3761 = x3660 == 1;
bool x3762 = x3760 == 1;
bool x3763 = x3761 || x3762;
bool x3764 = x3660 == x3760;
bool x3765 = x3763 || x3764;
x3766 = x3765;
} else {
x3766 = false;
}
bool x3770;
if (x3766) {
x3770 = x3769;
} else {
x3770 = false;
}
bool x3771;
if (x3770) {
x3771 = x3769;
} else {
x3771 = false;
}
if (x3771) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x3660,x3662,x3662,1,x3760,1,1);
assert(false && "");
}
bool x3777 = x3660 <= x3760;
int32_t x3778;
if (x3777) {
x3778 = x3760;
} else {
x3778 = x3660;
}
bool x3784 = x3778 > 0;
bool x3786;
if (x3784) {
x3786 = x3785;
} else {
x3786 = false;
}
bool x3787;
if (x3786) {
x3787 = x3785;
} else {
x3787 = false;
}
if (x3787) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(3660) x Sym(3662) x Sym(3662)"," x Const(1) x Sym(3760) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x3782 = x3778 * x3781;
int32_t x3783 = 64 * x3782;
float* x3793 = (float*)myMalloc(x3783 * sizeof(float));;
int32_t x3794 = 0;
int32_t x3795 = 0;
int32_t x3796 = 0;
bool x3842 = x3660 > 1;
bool x3846 = x3760 > 1;
for(int x3797=0; x3797 < 64; x3797++) {
int32_t x3798 = x3795;
int32_t x3799 = x3796;
int32_t x3800 = x3794;
int32_t x3801 = x3800;
int32_t x3802 = x3798;
int32_t x3803 = x3799;
for(int x3805=0; x3805 < x3778; x3805++) {
int32_t x3806 = x3802;
int32_t x3807 = x3803;
int32_t x3808 = x3801;
int32_t x3809 = x3808;
int32_t x3810 = x3806;
int32_t x3811 = x3807;
for(int x3813=0; x3813 < x3780; x3813++) {
int32_t x3814 = x3810;
int32_t x3815 = x3811;
int32_t x3816 = x3809;
int32_t x3817 = x3816;
int32_t x3818 = x3814;
int32_t x3819 = x3815;
for(int x3820=0; x3820 < x3780; x3820++) {
int32_t x3821 = x3817;
int32_t x3822 = x3818;
float x3823 = x3675[x3822];
int32_t x3824 = x3819;
float x3825 = x60[x3824];
float x3826 = x3823 + x3825;
x3793[x3821] = x3826;
x3817 += 1;
if (x3829) {
x3818 += 1;
} else {
}

}
x3809 += x3780;
if (x3829) {
x3810 += x3662;
} else {
}

}
x3801 += x3781;
if (x3842) {
x3802 += x3663;
} else {
}
if (x3846) {
x3803 += 1;
} else {
}

}
x3794 += x3782;
x3795 += x3664;

}
float* x3856 = (float*)myMalloc(x3783 * sizeof(float));;
for(int x3858=0; x3858 < x3783; x3858++) {
float x3859 = x3793[x3858];
bool x3860 = x3859 < 0.0f;
if (x3860) {
x3856[x3858] = 0.0f;
} else {
float x3863 = x3793[x3858];
x3856[x3858] = x3863;
}

}
float* x3878 = (float*)myMalloc(x3877 * sizeof(float));;
int32_t x3879 = 9 * x3778;
int32_t x3882 = 64 * x3879;
int32_t x3883 = x3882 * x3873;
float* x3884 = (float*)myMalloc(x3883 * sizeof(float));;
int32_t x3880 = x3879 * x3873;
int32_t x3892 = x3778 * 3;
int32_t x3893 = x3892 * 3;
for(int x3885=0; x3885 < 64; x3885++) {
int32_t x3886 = x3885 * x3782;
float* x3887 = x3856+x3886;
int32_t x3888 = x3885 * x3874;
float* x3889 = x3878+x3888;
int32_t x3890 = x3885 * x3880;
float* x3891 = x3884+x3890;
for(int x3895=0; x3895 < x3893; x3895++) {
int32_t x3896 = x3895 / 9;
int32_t x3900 = x3896 * 3;
int32_t x3901 = x3900 * 3;
int32_t x3902 = x3901 * x3872;
int32_t x3903 = x3902 * x3872;
int32_t x3897 = x3895 % 9;
int32_t x3898 = x3897 / 3;
int32_t x3904 = x3898 * 3;
int32_t x3905 = x3904 * x3872;
int32_t x3906 = x3905 * x3872;
int32_t x3907 = x3903 + x3906;
int32_t x3899 = x3897 % 3;
int32_t x3908 = x3899 * x3872;
int32_t x3909 = x3908 * x3872;
int32_t x3910 = x3907 + x3909;
float* x3911 = x3891+x3910;
int32_t x3912 = x3896 * x3780;
int32_t x3913 = x3912 * x3780;
float* x3914 = x3887+x3913;
int32_t x3927 = 1 - x3899;
bool x3928 = x3927 > 0;
int32_t x3929;
if (x3928) {
x3929 = x3927;
} else {
x3929 = 0;
}
int32_t x3930 = 3 - x3899;
int32_t x3931 = x3930 - 1;
int32_t x3932 = 1 - x3931;
bool x3933 = x3932 > 0;
int32_t x3934;
if (x3933) {
x3934 = x3932;
} else {
x3934 = 0;
}
int32_t x3935 = x3872 - x3934;
int32_t x3936 = x3935 - x3929;
bool x3937 = x3936 <= 0;
bool x3941 = x3929 > 0;
int32_t x3926 = -1 + x3899;
bool x3954 = x3934 > 0;
for(int x3916=0; x3916 < x3872; x3916++) {
int32_t x3917 = x3916 - 1;
int32_t x3918 = x3917 + x3898;
bool x3919 = x3918 < 0;
bool x3920 = x3918 >= x3780;
bool x3921 = x3919 || x3920;
if (x3921) {
int32_t x3922 = x3916 * x3872;
float* x3923 = x3911+x3922;
memset(x3923, 0, 4 * x3872);;
} else {
if (x3937) {
int32_t x3922 = x3916 * x3872;
float* x3938 = x3911+x3922;
memset(x3938, 0, 4 * x3872);;
} else {
int32_t x3922 = x3916 * x3872;
if (x3941) {
float* x3942 = x3911+x3922;
memset(x3942, 0, 4 * x3929);;
} else {
}
// may have segfault here
int32_t x3947 = x3922 + x3929;
float* x3948 = x3911+x3947;
int32_t x3949 = x3918 * x3780;
int32_t x3950 = x3949 + x3926;
int32_t x3951 = x3950 + x3929;
float* x3952 = x3914+x3951;
memcpy(x3948, x3952, 4 * x3936);;
if (x3954) {
int32_t x3955 = x3922 + x3872;
int32_t x3956 = x3955 - x3934;
float* x3957 = x3911+x3956;
memset(x3957, 0, 4 * x3934);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x3873,x3879,1,x83,x3879,x3891,x3873,1,x3889,x3873);

}
int32_t x3972 = 0;
int32_t x3973 = 1;
x3973 *= 1;
x3972 += 1;
x3973 *= 1;
x3973 *= 1;
int32_t x3978 = x3972;
bool x3979 = x3978 >= 2;
if (x3979) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3984 = x3978 == 0;
if (x3984) {
int32_t x3985 = x3973;
bool x3986 = x3985 == 64;
if (x3986) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3993 = x3973;
int32_t x3994 = 64 / x3993;
bool x3998;
if (x452) {
bool x3995 = x3994 == 1;
bool x3996 = 64 == x3994;
bool x3997 = x3995 || x3996;
x3998 = x3997;
} else {
x3998 = false;
}
bool x4002;
if (x3998) {
x4002 = x4001;
} else {
x4002 = false;
}
bool x4003;
if (x4002) {
x4003 = x4001;
} else {
x4003 = false;
}
if (x4003) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x3872,x3872,1,x3994,1,1);
assert(false && "");
}
bool x4009 = 64 <= x3994;
int32_t x4010;
if (x4009) {
x4010 = x3994;
} else {
x4010 = 64;
}
bool x4016 = x4010 > 0;
bool x4018;
if (x4016) {
x4018 = x4017;
} else {
x4018 = false;
}
bool x4019;
if (x4018) {
x4019 = x4017;
} else {
x4019 = false;
}
if (x4019) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(64) x Sym(3872) x Sym(3872)"," x Const(1) x Sym(3994) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x4014 = x4010 * x4013;
int32_t x4015 = 64 * x4014;
float* x4025 = (float*)myMalloc(x4015 * sizeof(float));;
int32_t x4026 = 0;
int32_t x4027 = 0;
int32_t x4028 = 0;
bool x4075 = x3994 > 1;
for(int x4029=0; x4029 < 64; x4029++) {
int32_t x4030 = x4027;
int32_t x4031 = x4028;
int32_t x4032 = x4026;
int32_t x4033 = x4032;
int32_t x4034 = x4030;
int32_t x4035 = x4031;
for(int x4037=0; x4037 < x4010; x4037++) {
int32_t x4038 = x4034;
int32_t x4039 = x4035;
int32_t x4040 = x4033;
int32_t x4041 = x4040;
int32_t x4042 = x4038;
int32_t x4043 = x4039;
for(int x4045=0; x4045 < x4012; x4045++) {
int32_t x4046 = x4042;
int32_t x4047 = x4043;
int32_t x4048 = x4041;
int32_t x4049 = x4048;
int32_t x4050 = x4046;
int32_t x4051 = x4047;
for(int x4052=0; x4052 < x4012; x4052++) {
int32_t x4053 = x4049;
int32_t x4054 = x4050;
float x4055 = x3878[x4054];
int32_t x4056 = x4051;
float x4057 = x44[x4056];
float x4058 = x4055 - x4057;
x4025[x4053] = x4058;
x4049 += 1;
if (x4061) {
x4050 += 1;
} else {
}

}
x4041 += x4012;
if (x4061) {
x4042 += x3872;
} else {
}

}
x4033 += x4013;
x4034 += x3873;
if (x4075) {
x4035 += 1;
} else {
}

}
x4026 += x4014;
x4027 += x3874;

}
float* x4085 = (float*)myMalloc(64 * sizeof(float));;
for(int x4086=0; x4086 < 64; x4086++) {
float x4087 = x244[x4086];
float x4088 = x4087 + 1.0E-5f;
x4085[x4086] = x4088;

}
float* x4092 = (float*)myMalloc(64 * sizeof(float));;
for(int x4093=0; x4093 < 64; x4093++) {
float x4094 = x4085[x4093];
double x4095 = (double)x4094;
double x4096 = sqrt(x4095);
float x4097 = (float)x4096;
x4092[x4093] = x4097;

}
int32_t x4101 = 0;
int32_t x4102 = 1;
x4102 *= 1;
x4101 += 1;
x4102 *= 1;
x4102 *= 1;
int32_t x4107 = x4101;
bool x4108 = x4107 >= 2;
if (x4108) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4113 = x4107 == 0;
if (x4113) {
int32_t x4114 = x4102;
bool x4115 = x4114 == 64;
if (x4115) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4122 = x4102;
int32_t x4123 = 64 / x4122;
bool x4129;
if (x452) {
bool x4124 = x4010 == 1;
bool x4125 = x4123 == 1;
bool x4126 = x4124 || x4125;
bool x4127 = x4010 == x4123;
bool x4128 = x4126 || x4127;
x4129 = x4128;
} else {
x4129 = false;
}
bool x4133;
if (x4129) {
x4133 = x4132;
} else {
x4133 = false;
}
bool x4134;
if (x4133) {
x4134 = x4132;
} else {
x4134 = false;
}
if (x4134) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4010,x4012,x4012,1,x4123,1,1);
assert(false && "");
}
bool x4140 = x4010 <= x4123;
int32_t x4141;
if (x4140) {
x4141 = x4123;
} else {
x4141 = x4010;
}
bool x4147 = x4141 > 0;
bool x4149;
if (x4147) {
x4149 = x4148;
} else {
x4149 = false;
}
bool x4150;
if (x4149) {
x4150 = x4148;
} else {
x4150 = false;
}
if (x4150) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(4010) x Sym(4012) x Sym(4012)"," x Const(1) x Sym(4123) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x4145 = x4141 * x4144;
int32_t x4146 = 64 * x4145;
float* x4156 = (float*)myMalloc(x4146 * sizeof(float));;
int32_t x4157 = 0;
int32_t x4158 = 0;
int32_t x4159 = 0;
bool x4205 = x4010 > 1;
bool x4209 = x4123 > 1;
for(int x4160=0; x4160 < 64; x4160++) {
int32_t x4161 = x4158;
int32_t x4162 = x4159;
int32_t x4163 = x4157;
int32_t x4164 = x4163;
int32_t x4165 = x4161;
int32_t x4166 = x4162;
for(int x4168=0; x4168 < x4141; x4168++) {
int32_t x4169 = x4165;
int32_t x4170 = x4166;
int32_t x4171 = x4164;
int32_t x4172 = x4171;
int32_t x4173 = x4169;
int32_t x4174 = x4170;
for(int x4176=0; x4176 < x4143; x4176++) {
int32_t x4177 = x4173;
int32_t x4178 = x4174;
int32_t x4179 = x4172;
int32_t x4180 = x4179;
int32_t x4181 = x4177;
int32_t x4182 = x4178;
for(int x4183=0; x4183 < x4143; x4183++) {
int32_t x4184 = x4180;
int32_t x4185 = x4181;
float x4186 = x4025[x4185];
int32_t x4187 = x4182;
float x4188 = x4092[x4187];
float x4189 = x4186 / x4188;
x4156[x4184] = x4189;
x4180 += 1;
if (x4192) {
x4181 += 1;
} else {
}

}
x4172 += x4143;
if (x4192) {
x4173 += x4012;
} else {
}

}
x4164 += x4144;
if (x4205) {
x4165 += x4013;
} else {
}
if (x4209) {
x4166 += 1;
} else {
}

}
x4157 += x4145;
x4158 += x4014;

}
int32_t x4219 = 0;
int32_t x4220 = 1;
x4220 *= 1;
x4219 += 1;
x4220 *= 1;
x4220 *= 1;
int32_t x4225 = x4219;
bool x4226 = x4225 >= 2;
if (x4226) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4231 = x4225 == 0;
if (x4231) {
int32_t x4232 = x4220;
bool x4233 = x4232 == 64;
if (x4233) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4240 = x4220;
int32_t x4241 = 64 / x4240;
bool x4247;
if (x452) {
bool x4242 = x4141 == 1;
bool x4243 = x4241 == 1;
bool x4244 = x4242 || x4243;
bool x4245 = x4141 == x4241;
bool x4246 = x4244 || x4245;
x4247 = x4246;
} else {
x4247 = false;
}
bool x4251;
if (x4247) {
x4251 = x4250;
} else {
x4251 = false;
}
bool x4252;
if (x4251) {
x4252 = x4250;
} else {
x4252 = false;
}
if (x4252) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4141,x4143,x4143,1,x4241,1,1);
assert(false && "");
}
bool x4258 = x4141 <= x4241;
int32_t x4259;
if (x4258) {
x4259 = x4241;
} else {
x4259 = x4141;
}
bool x4265 = x4259 > 0;
bool x4267;
if (x4265) {
x4267 = x4266;
} else {
x4267 = false;
}
bool x4268;
if (x4267) {
x4268 = x4266;
} else {
x4268 = false;
}
if (x4268) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(4141) x Sym(4143) x Sym(4143)"," x Const(1) x Sym(4241) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x4263 = x4259 * x4262;
int32_t x4264 = 64 * x4263;
float* x4274 = (float*)myMalloc(x4264 * sizeof(float));;
int32_t x4275 = 0;
int32_t x4276 = 0;
int32_t x4277 = 0;
bool x4323 = x4141 > 1;
bool x4327 = x4241 > 1;
for(int x4278=0; x4278 < 64; x4278++) {
int32_t x4279 = x4276;
int32_t x4280 = x4277;
int32_t x4281 = x4275;
int32_t x4282 = x4281;
int32_t x4283 = x4279;
int32_t x4284 = x4280;
for(int x4286=0; x4286 < x4259; x4286++) {
int32_t x4287 = x4283;
int32_t x4288 = x4284;
int32_t x4289 = x4282;
int32_t x4290 = x4289;
int32_t x4291 = x4287;
int32_t x4292 = x4288;
for(int x4294=0; x4294 < x4261; x4294++) {
int32_t x4295 = x4291;
int32_t x4296 = x4292;
int32_t x4297 = x4290;
int32_t x4298 = x4297;
int32_t x4299 = x4295;
int32_t x4300 = x4296;
for(int x4301=0; x4301 < x4261; x4301++) {
int32_t x4302 = x4298;
int32_t x4303 = x4299;
float x4304 = x4156[x4303];
int32_t x4305 = x4300;
float x4306 = x208[x4305];
float x4307 = x4304 * x4306;
x4274[x4302] = x4307;
x4298 += 1;
if (x4310) {
x4299 += 1;
} else {
}

}
x4290 += x4261;
if (x4310) {
x4291 += x4143;
} else {
}

}
x4282 += x4262;
if (x4323) {
x4283 += x4144;
} else {
}
if (x4327) {
x4284 += 1;
} else {
}

}
x4275 += x4263;
x4276 += x4145;

}
int32_t x4337 = 0;
int32_t x4338 = 1;
x4338 *= 1;
x4337 += 1;
x4338 *= 1;
x4338 *= 1;
int32_t x4343 = x4337;
bool x4344 = x4343 >= 2;
if (x4344) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4349 = x4343 == 0;
if (x4349) {
int32_t x4350 = x4338;
bool x4351 = x4350 == 64;
if (x4351) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4358 = x4338;
int32_t x4359 = 64 / x4358;
bool x4365;
if (x452) {
bool x4360 = x4259 == 1;
bool x4361 = x4359 == 1;
bool x4362 = x4360 || x4361;
bool x4363 = x4259 == x4359;
bool x4364 = x4362 || x4363;
x4365 = x4364;
} else {
x4365 = false;
}
bool x4369;
if (x4365) {
x4369 = x4368;
} else {
x4369 = false;
}
bool x4370;
if (x4369) {
x4370 = x4368;
} else {
x4370 = false;
}
if (x4370) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4259,x4261,x4261,1,x4359,1,1);
assert(false && "");
}
bool x4376 = x4259 <= x4359;
int32_t x4377;
if (x4376) {
x4377 = x4359;
} else {
x4377 = x4259;
}
bool x4383 = x4377 > 0;
bool x4385;
if (x4383) {
x4385 = x4384;
} else {
x4385 = false;
}
bool x4386;
if (x4385) {
x4386 = x4384;
} else {
x4386 = false;
}
if (x4386) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(4259) x Sym(4261) x Sym(4261)"," x Const(1) x Sym(4359) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x4381 = x4377 * x4380;
int32_t x4382 = 64 * x4381;
float* x4392 = (float*)myMalloc(x4382 * sizeof(float));;
int32_t x4393 = 0;
int32_t x4394 = 0;
int32_t x4395 = 0;
bool x4441 = x4259 > 1;
bool x4445 = x4359 > 1;
for(int x4396=0; x4396 < 64; x4396++) {
int32_t x4397 = x4394;
int32_t x4398 = x4395;
int32_t x4399 = x4393;
int32_t x4400 = x4399;
int32_t x4401 = x4397;
int32_t x4402 = x4398;
for(int x4404=0; x4404 < x4377; x4404++) {
int32_t x4405 = x4401;
int32_t x4406 = x4402;
int32_t x4407 = x4400;
int32_t x4408 = x4407;
int32_t x4409 = x4405;
int32_t x4410 = x4406;
for(int x4412=0; x4412 < x4379; x4412++) {
int32_t x4413 = x4409;
int32_t x4414 = x4410;
int32_t x4415 = x4408;
int32_t x4416 = x4415;
int32_t x4417 = x4413;
int32_t x4418 = x4414;
for(int x4419=0; x4419 < x4379; x4419++) {
int32_t x4420 = x4416;
int32_t x4421 = x4417;
float x4422 = x4274[x4421];
int32_t x4423 = x4418;
float x4424 = x153[x4423];
float x4425 = x4422 + x4424;
x4392[x4420] = x4425;
x4416 += 1;
if (x4428) {
x4417 += 1;
} else {
}

}
x4408 += x4379;
if (x4428) {
x4409 += x4261;
} else {
}

}
x4400 += x4380;
if (x4441) {
x4401 += x4262;
} else {
}
if (x4445) {
x4402 += 1;
} else {
}

}
x4393 += x4381;
x4394 += x4263;

}
float* x4455 = (float*)myMalloc(x4382 * sizeof(float));;
for(int x4457=0; x4457 < x4382; x4457++) {
float x4458 = x4392[x4457];
bool x4459 = x4458 < 0.0f;
if (x4459) {
x4455[x4457] = 0.0f;
} else {
float x4462 = x4392[x4457];
x4455[x4457] = x4462;
}

}
float* x4476 = (float*)myMalloc(x4475 * sizeof(float));;
int32_t x4479 = 64 * x4377;
int32_t x4480 = x4479 * x4471;
float* x4481 = (float*)myMalloc(x4480 * sizeof(float));;
int32_t x4477 = x4377 * x4471;
for(int x4482=0; x4482 < 64; x4482++) {
int32_t x4483 = x4482 * x4381;
float* x4484 = x4455+x4483;
int32_t x4485 = x4482 * x4472;
float* x4486 = x4476+x4485;
int32_t x4487 = x4482 * x4477;
float* x4488 = x4481+x4487;
for(int x4489=0; x4489 < x4377; x4489++) {
int32_t x4490 = x4489 / 1;
int32_t x4494 = x4490 * x4470;
int32_t x4495 = x4494 * x4470;
int32_t x4491 = x4489 % 1;
int32_t x4492 = x4491 / 1;
int32_t x4496 = x4492 * x4470;
int32_t x4497 = x4496 * x4470;
int32_t x4498 = x4495 + x4497;
int32_t x4493 = x4491 % 1;
int32_t x4499 = x4493 * x4470;
int32_t x4500 = x4499 * x4470;
int32_t x4501 = x4498 + x4500;
float* x4502 = x4488+x4501;
int32_t x4503 = x4490 * x4379;
int32_t x4504 = x4503 * x4379;
float* x4505 = x4484+x4504;
for(int x4507=0; x4507 < x4470; x4507++) {
int32_t x4509 = x4507 * x4470;
float* x4510 = x4502+x4509;
int32_t x4508 = x4507 + x4492;
int32_t x4511 = x4508 * x4379;
int32_t x4512 = x4511 + x4493;
float* x4513 = x4505+x4512;
memcpy(x4510, x4513, 4 * x4470);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x4471,x4377,1,x130,x4377,x4488,x4471,1,x4486,x4471);

}
int32_t x4522 = 0;
int32_t x4523 = 1;
x4523 *= 1;
x4522 += 1;
x4523 *= 1;
x4523 *= 1;
int32_t x4528 = x4522;
bool x4529 = x4528 >= 2;
if (x4529) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4534 = x4528 == 0;
if (x4534) {
int32_t x4535 = x4523;
bool x4536 = x4535 == 256;
if (x4536) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4543 = x4523;
int32_t x4544 = 256 / x4543;
bool x4548;
if (x452) {
bool x4545 = x4544 == 1;
bool x4546 = 256 == x4544;
bool x4547 = x4545 || x4546;
x4548 = x4547;
} else {
x4548 = false;
}
bool x4552;
if (x4548) {
x4552 = x4551;
} else {
x4552 = false;
}
bool x4553;
if (x4552) {
x4553 = x4551;
} else {
x4553 = false;
}
if (x4553) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x4470,x4470,1,x4544,1,1);
assert(false && "");
}
bool x4559 = 256 <= x4544;
int32_t x4560;
if (x4559) {
x4560 = x4544;
} else {
x4560 = 256;
}
bool x4566 = x4560 > 0;
bool x4568;
if (x4566) {
x4568 = x4567;
} else {
x4568 = false;
}
bool x4569;
if (x4568) {
x4569 = x4567;
} else {
x4569 = false;
}
if (x4569) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(4470) x Sym(4470)"," x Const(1) x Sym(4544) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x4564 = x4560 * x4563;
int32_t x4565 = 64 * x4564;
float* x4575 = (float*)myMalloc(x4565 * sizeof(float));;
int32_t x4576 = 0;
int32_t x4577 = 0;
int32_t x4578 = 0;
bool x4625 = x4544 > 1;
for(int x4579=0; x4579 < 64; x4579++) {
int32_t x4580 = x4577;
int32_t x4581 = x4578;
int32_t x4582 = x4576;
int32_t x4583 = x4582;
int32_t x4584 = x4580;
int32_t x4585 = x4581;
for(int x4587=0; x4587 < x4560; x4587++) {
int32_t x4588 = x4584;
int32_t x4589 = x4585;
int32_t x4590 = x4583;
int32_t x4591 = x4590;
int32_t x4592 = x4588;
int32_t x4593 = x4589;
for(int x4595=0; x4595 < x4562; x4595++) {
int32_t x4596 = x4592;
int32_t x4597 = x4593;
int32_t x4598 = x4591;
int32_t x4599 = x4598;
int32_t x4600 = x4596;
int32_t x4601 = x4597;
for(int x4602=0; x4602 < x4562; x4602++) {
int32_t x4603 = x4599;
int32_t x4604 = x4600;
float x4605 = x4476[x4604];
int32_t x4606 = x4601;
float x4607 = x91[x4606];
float x4608 = x4605 - x4607;
x4575[x4603] = x4608;
x4599 += 1;
if (x4611) {
x4600 += 1;
} else {
}

}
x4591 += x4562;
if (x4611) {
x4592 += x4470;
} else {
}

}
x4583 += x4563;
x4584 += x4471;
if (x4625) {
x4585 += 1;
} else {
}

}
x4576 += x4564;
x4577 += x4472;

}
float* x4635 = (float*)myMalloc(256 * sizeof(float));;
for(int x4636=0; x4636 < 256; x4636++) {
float x4637 = x166[x4636];
float x4638 = x4637 + 1.0E-5f;
x4635[x4636] = x4638;

}
float* x4642 = (float*)myMalloc(256 * sizeof(float));;
for(int x4643=0; x4643 < 256; x4643++) {
float x4644 = x4635[x4643];
double x4645 = (double)x4644;
double x4646 = sqrt(x4645);
float x4647 = (float)x4646;
x4642[x4643] = x4647;

}
int32_t x4651 = 0;
int32_t x4652 = 1;
x4652 *= 1;
x4651 += 1;
x4652 *= 1;
x4652 *= 1;
int32_t x4657 = x4651;
bool x4658 = x4657 >= 2;
if (x4658) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4663 = x4657 == 0;
if (x4663) {
int32_t x4664 = x4652;
bool x4665 = x4664 == 256;
if (x4665) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4672 = x4652;
int32_t x4673 = 256 / x4672;
bool x4679;
if (x452) {
bool x4674 = x4560 == 1;
bool x4675 = x4673 == 1;
bool x4676 = x4674 || x4675;
bool x4677 = x4560 == x4673;
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
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4560,x4562,x4562,1,x4673,1,1);
assert(false && "");
}
bool x4690 = x4560 <= x4673;
int32_t x4691;
if (x4690) {
x4691 = x4673;
} else {
x4691 = x4560;
}
bool x4697 = x4691 > 0;
bool x4699;
if (x4697) {
x4699 = x4698;
} else {
x4699 = false;
}
bool x4700;
if (x4699) {
x4700 = x4698;
} else {
x4700 = false;
}
if (x4700) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(4560) x Sym(4562) x Sym(4562)"," x Const(1) x Sym(4673) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x4695 = x4691 * x4694;
int32_t x4696 = 64 * x4695;
float* x4706 = (float*)myMalloc(x4696 * sizeof(float));;
int32_t x4707 = 0;
int32_t x4708 = 0;
int32_t x4709 = 0;
bool x4755 = x4560 > 1;
bool x4759 = x4673 > 1;
for(int x4710=0; x4710 < 64; x4710++) {
int32_t x4711 = x4708;
int32_t x4712 = x4709;
int32_t x4713 = x4707;
int32_t x4714 = x4713;
int32_t x4715 = x4711;
int32_t x4716 = x4712;
for(int x4718=0; x4718 < x4691; x4718++) {
int32_t x4719 = x4715;
int32_t x4720 = x4716;
int32_t x4721 = x4714;
int32_t x4722 = x4721;
int32_t x4723 = x4719;
int32_t x4724 = x4720;
for(int x4726=0; x4726 < x4693; x4726++) {
int32_t x4727 = x4723;
int32_t x4728 = x4724;
int32_t x4729 = x4722;
int32_t x4730 = x4729;
int32_t x4731 = x4727;
int32_t x4732 = x4728;
for(int x4733=0; x4733 < x4693; x4733++) {
int32_t x4734 = x4730;
int32_t x4735 = x4731;
float x4736 = x4575[x4735];
int32_t x4737 = x4732;
float x4738 = x4642[x4737];
float x4739 = x4736 / x4738;
x4706[x4734] = x4739;
x4730 += 1;
if (x4742) {
x4731 += 1;
} else {
}

}
x4722 += x4693;
if (x4742) {
x4723 += x4562;
} else {
}

}
x4714 += x4694;
if (x4755) {
x4715 += x4563;
} else {
}
if (x4759) {
x4716 += 1;
} else {
}

}
x4707 += x4695;
x4708 += x4564;

}
int32_t x4769 = 0;
int32_t x4770 = 1;
x4770 *= 1;
x4769 += 1;
x4770 *= 1;
x4770 *= 1;
int32_t x4775 = x4769;
bool x4776 = x4775 >= 2;
if (x4776) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4781 = x4775 == 0;
if (x4781) {
int32_t x4782 = x4770;
bool x4783 = x4782 == 256;
if (x4783) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4790 = x4770;
int32_t x4791 = 256 / x4790;
bool x4797;
if (x452) {
bool x4792 = x4691 == 1;
bool x4793 = x4791 == 1;
bool x4794 = x4792 || x4793;
bool x4795 = x4691 == x4791;
bool x4796 = x4794 || x4795;
x4797 = x4796;
} else {
x4797 = false;
}
bool x4801;
if (x4797) {
x4801 = x4800;
} else {
x4801 = false;
}
bool x4802;
if (x4801) {
x4802 = x4800;
} else {
x4802 = false;
}
if (x4802) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4691,x4693,x4693,1,x4791,1,1);
assert(false && "");
}
bool x4808 = x4691 <= x4791;
int32_t x4809;
if (x4808) {
x4809 = x4791;
} else {
x4809 = x4691;
}
bool x4815 = x4809 > 0;
bool x4817;
if (x4815) {
x4817 = x4816;
} else {
x4817 = false;
}
bool x4818;
if (x4817) {
x4818 = x4816;
} else {
x4818 = false;
}
if (x4818) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(4691) x Sym(4693) x Sym(4693)"," x Const(1) x Sym(4791) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x4813 = x4809 * x4812;
int32_t x4814 = 64 * x4813;
float* x4824 = (float*)myMalloc(x4814 * sizeof(float));;
int32_t x4825 = 0;
int32_t x4826 = 0;
int32_t x4827 = 0;
bool x4873 = x4691 > 1;
bool x4877 = x4791 > 1;
for(int x4828=0; x4828 < 64; x4828++) {
int32_t x4829 = x4826;
int32_t x4830 = x4827;
int32_t x4831 = x4825;
int32_t x4832 = x4831;
int32_t x4833 = x4829;
int32_t x4834 = x4830;
for(int x4836=0; x4836 < x4809; x4836++) {
int32_t x4837 = x4833;
int32_t x4838 = x4834;
int32_t x4839 = x4832;
int32_t x4840 = x4839;
int32_t x4841 = x4837;
int32_t x4842 = x4838;
for(int x4844=0; x4844 < x4811; x4844++) {
int32_t x4845 = x4841;
int32_t x4846 = x4842;
int32_t x4847 = x4840;
int32_t x4848 = x4847;
int32_t x4849 = x4845;
int32_t x4850 = x4846;
for(int x4851=0; x4851 < x4811; x4851++) {
int32_t x4852 = x4848;
int32_t x4853 = x4849;
float x4854 = x4706[x4853];
int32_t x4855 = x4850;
float x4856 = x58[x4855];
float x4857 = x4854 * x4856;
x4824[x4852] = x4857;
x4848 += 1;
if (x4860) {
x4849 += 1;
} else {
}

}
x4840 += x4811;
if (x4860) {
x4841 += x4693;
} else {
}

}
x4832 += x4812;
if (x4873) {
x4833 += x4694;
} else {
}
if (x4877) {
x4834 += 1;
} else {
}

}
x4825 += x4813;
x4826 += x4695;

}
int32_t x4887 = 0;
int32_t x4888 = 1;
x4888 *= 1;
x4887 += 1;
x4888 *= 1;
x4888 *= 1;
int32_t x4893 = x4887;
bool x4894 = x4893 >= 2;
if (x4894) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x4899 = x4893 == 0;
if (x4899) {
int32_t x4900 = x4888;
bool x4901 = x4900 == 256;
if (x4901) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x4908 = x4888;
int32_t x4909 = 256 / x4908;
bool x4915;
if (x452) {
bool x4910 = x4809 == 1;
bool x4911 = x4909 == 1;
bool x4912 = x4910 || x4911;
bool x4913 = x4809 == x4909;
bool x4914 = x4912 || x4913;
x4915 = x4914;
} else {
x4915 = false;
}
bool x4919;
if (x4915) {
x4919 = x4918;
} else {
x4919 = false;
}
bool x4920;
if (x4919) {
x4920 = x4918;
} else {
x4920 = false;
}
if (x4920) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4809,x4811,x4811,1,x4909,1,1);
assert(false && "");
}
bool x4926 = x4809 <= x4909;
int32_t x4927;
if (x4926) {
x4927 = x4909;
} else {
x4927 = x4809;
}
bool x4933 = x4927 > 0;
bool x4935;
if (x4933) {
x4935 = x4934;
} else {
x4935 = false;
}
bool x4936;
if (x4935) {
x4936 = x4934;
} else {
x4936 = false;
}
if (x4936) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(4809) x Sym(4811) x Sym(4811)"," x Const(1) x Sym(4909) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x4931 = x4927 * x4930;
int32_t x4932 = 64 * x4931;
float* x4942 = (float*)myMalloc(x4932 * sizeof(float));;
int32_t x4943 = 0;
int32_t x4944 = 0;
int32_t x4945 = 0;
bool x4991 = x4809 > 1;
bool x4995 = x4909 > 1;
for(int x4946=0; x4946 < 64; x4946++) {
int32_t x4947 = x4944;
int32_t x4948 = x4945;
int32_t x4949 = x4943;
int32_t x4950 = x4949;
int32_t x4951 = x4947;
int32_t x4952 = x4948;
for(int x4954=0; x4954 < x4927; x4954++) {
int32_t x4955 = x4951;
int32_t x4956 = x4952;
int32_t x4957 = x4950;
int32_t x4958 = x4957;
int32_t x4959 = x4955;
int32_t x4960 = x4956;
for(int x4962=0; x4962 < x4929; x4962++) {
int32_t x4963 = x4959;
int32_t x4964 = x4960;
int32_t x4965 = x4958;
int32_t x4966 = x4965;
int32_t x4967 = x4963;
int32_t x4968 = x4964;
for(int x4969=0; x4969 < x4929; x4969++) {
int32_t x4970 = x4966;
int32_t x4971 = x4967;
float x4972 = x4824[x4971];
int32_t x4973 = x4968;
float x4974 = x7[x4973];
float x4975 = x4972 + x4974;
x4942[x4970] = x4975;
x4966 += 1;
if (x4978) {
x4967 += 1;
} else {
}

}
x4958 += x4929;
if (x4978) {
x4959 += x4811;
} else {
}

}
x4950 += x4930;
if (x4991) {
x4951 += x4812;
} else {
}
if (x4995) {
x4952 += 1;
} else {
}

}
x4943 += x4931;
x4944 += x4813;

}
bool x5005 = x4927 == 1;
bool x5006 = x5005 || x3213;
bool x5007 = x4927 == x2643;
bool x5008 = x5006 || x5007;
bool x5013;
if (x5008) {
x5013 = x5012;
} else {
x5013 = false;
}
bool x5014;
if (x5013) {
x5014 = x5012;
} else {
x5014 = false;
}
if (x5014) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x4927,x4929,x4929,64,x2643,x2645,x2645);
assert(false && "");
}
int32_t x5027 = 0;
int32_t x5028 = 0;
int32_t x5029 = 0;
bool x5020 = x4927 <= x2643;
int32_t x5021;
if (x5020) {
x5021 = x2643;
} else {
x5021 = x4927;
}
bool x5080 = x4927 > 1;
int32_t x5025 = x5021 * x5024;
for(int x5030=0; x5030 < 64; x5030++) {
int32_t x5031 = x5028;
int32_t x5032 = x5029;
int32_t x5033 = x5027;
int32_t x5034 = x5033;
int32_t x5035 = x5031;
int32_t x5036 = x5032;
for(int x5038=0; x5038 < x5021; x5038++) {
int32_t x5039 = x5035;
int32_t x5040 = x5036;
int32_t x5041 = x5034;
int32_t x5042 = x5041;
int32_t x5043 = x5039;
int32_t x5044 = x5040;
for(int x5046=0; x5046 < x5023; x5046++) {
int32_t x5047 = x5043;
int32_t x5048 = x5044;
int32_t x5049 = x5042;
int32_t x5050 = x5049;
int32_t x5051 = x5047;
int32_t x5052 = x5048;
for(int x5053=0; x5053 < x5023; x5053++) {
int32_t x5054 = x5051;
float x5055 = x4942[x5054];
int32_t x5056 = x5052;
float x5057 = x3306[x5056];
float x5058 = x5055 + x5057;
x4942[x5054] = x5058;
x5050 += 1;
if (x5061) {
x5051 += 1;
} else {
}
if (x3271) {
x5052 += 1;
} else {
}

}
x5042 += x5023;
if (x5061) {
x5043 += x4929;
} else {
}
if (x3271) {
x5044 += x2645;
} else {
}

}
x5034 += x5024;
if (x5080) {
x5035 += x4930;
} else {
}
if (x3291) {
x5036 += x2646;
} else {
}

}
x5027 += x5025;
x5028 += x4931;
x5029 += x2647;

}
float* x5094 = (float*)myMalloc(x4932 * sizeof(float));;
for(int x5096=0; x5096 < x4932; x5096++) {
float x5097 = x4942[x5096];
bool x5098 = x5097 < 0.0f;
if (x5098) {
x5094[x5096] = 0.0f;
} else {
float x5101 = x4942[x5096];
x5094[x5096] = x5101;
}

}
float* x5115 = (float*)myMalloc(x5114 * sizeof(float));;
int32_t x5118 = 64 * x4927;
int32_t x5119 = x5118 * x5110;
float* x5120 = (float*)myMalloc(x5119 * sizeof(float));;
int32_t x5116 = x4927 * x5110;
for(int x5121=0; x5121 < 64; x5121++) {
int32_t x5122 = x5121 * x4931;
float* x5123 = x5094+x5122;
int32_t x5124 = x5121 * x5111;
float* x5125 = x5115+x5124;
int32_t x5126 = x5121 * x5116;
float* x5127 = x5120+x5126;
for(int x5128=0; x5128 < x4927; x5128++) {
int32_t x5129 = x5128 / 1;
int32_t x5133 = x5129 * x5109;
int32_t x5134 = x5133 * x5109;
int32_t x5130 = x5128 % 1;
int32_t x5131 = x5130 / 1;
int32_t x5135 = x5131 * x5109;
int32_t x5136 = x5135 * x5109;
int32_t x5137 = x5134 + x5136;
int32_t x5132 = x5130 % 1;
int32_t x5138 = x5132 * x5109;
int32_t x5139 = x5138 * x5109;
int32_t x5140 = x5137 + x5139;
float* x5141 = x5127+x5140;
int32_t x5142 = x5129 * x4929;
int32_t x5143 = x5142 * x4929;
float* x5144 = x5123+x5143;
for(int x5146=0; x5146 < x5109; x5146++) {
int32_t x5148 = x5146 * x5109;
float* x5149 = x5141+x5148;
int32_t x5147 = x5146 + x5131;
int32_t x5150 = x5147 * x4929;
int32_t x5151 = x5150 + x5132;
float* x5152 = x5144+x5151;
memcpy(x5149, x5152, 4 * x5109);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x5110,x4927,1,x150,x4927,x5127,x5110,1,x5125,x5110);

}
int32_t x5161 = 0;
int32_t x5162 = 1;
x5162 *= 1;
x5161 += 1;
x5162 *= 1;
x5162 *= 1;
int32_t x5167 = x5161;
bool x5168 = x5167 >= 2;
if (x5168) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5173 = x5167 == 0;
if (x5173) {
int32_t x5174 = x5162;
bool x5175 = x5174 == 64;
if (x5175) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5182 = x5162;
int32_t x5183 = 64 / x5182;
bool x5187;
if (x452) {
bool x5184 = x5183 == 1;
bool x5185 = 64 == x5183;
bool x5186 = x5184 || x5185;
x5187 = x5186;
} else {
x5187 = false;
}
bool x5191;
if (x5187) {
x5191 = x5190;
} else {
x5191 = false;
}
bool x5192;
if (x5191) {
x5192 = x5190;
} else {
x5192 = false;
}
if (x5192) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x5109,x5109,1,x5183,1,1);
assert(false && "");
}
bool x5198 = 64 <= x5183;
int32_t x5199;
if (x5198) {
x5199 = x5183;
} else {
x5199 = 64;
}
bool x5205 = x5199 > 0;
bool x5207;
if (x5205) {
x5207 = x5206;
} else {
x5207 = false;
}
bool x5208;
if (x5207) {
x5208 = x5206;
} else {
x5208 = false;
}
if (x5208) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(64) x Sym(5109) x Sym(5109)"," x Const(1) x Sym(5183) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x5203 = x5199 * x5202;
int32_t x5204 = 64 * x5203;
float* x5214 = (float*)myMalloc(x5204 * sizeof(float));;
int32_t x5215 = 0;
int32_t x5216 = 0;
int32_t x5217 = 0;
bool x5264 = x5183 > 1;
for(int x5218=0; x5218 < 64; x5218++) {
int32_t x5219 = x5216;
int32_t x5220 = x5217;
int32_t x5221 = x5215;
int32_t x5222 = x5221;
int32_t x5223 = x5219;
int32_t x5224 = x5220;
for(int x5226=0; x5226 < x5199; x5226++) {
int32_t x5227 = x5223;
int32_t x5228 = x5224;
int32_t x5229 = x5222;
int32_t x5230 = x5229;
int32_t x5231 = x5227;
int32_t x5232 = x5228;
for(int x5234=0; x5234 < x5201; x5234++) {
int32_t x5235 = x5231;
int32_t x5236 = x5232;
int32_t x5237 = x5230;
int32_t x5238 = x5237;
int32_t x5239 = x5235;
int32_t x5240 = x5236;
for(int x5241=0; x5241 < x5201; x5241++) {
int32_t x5242 = x5238;
int32_t x5243 = x5239;
float x5244 = x5115[x5243];
int32_t x5245 = x5240;
float x5246 = x257[x5245];
float x5247 = x5244 - x5246;
x5214[x5242] = x5247;
x5238 += 1;
if (x5250) {
x5239 += 1;
} else {
}

}
x5230 += x5201;
if (x5250) {
x5231 += x5109;
} else {
}

}
x5222 += x5202;
x5223 += x5110;
if (x5264) {
x5224 += 1;
} else {
}

}
x5215 += x5203;
x5216 += x5111;

}
float* x5274 = (float*)myMalloc(64 * sizeof(float));;
for(int x5275=0; x5275 < 64; x5275++) {
float x5276 = x187[x5275];
float x5277 = x5276 + 1.0E-5f;
x5274[x5275] = x5277;

}
float* x5281 = (float*)myMalloc(64 * sizeof(float));;
for(int x5282=0; x5282 < 64; x5282++) {
float x5283 = x5274[x5282];
double x5284 = (double)x5283;
double x5285 = sqrt(x5284);
float x5286 = (float)x5285;
x5281[x5282] = x5286;

}
int32_t x5290 = 0;
int32_t x5291 = 1;
x5291 *= 1;
x5290 += 1;
x5291 *= 1;
x5291 *= 1;
int32_t x5296 = x5290;
bool x5297 = x5296 >= 2;
if (x5297) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5302 = x5296 == 0;
if (x5302) {
int32_t x5303 = x5291;
bool x5304 = x5303 == 64;
if (x5304) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5311 = x5291;
int32_t x5312 = 64 / x5311;
bool x5318;
if (x452) {
bool x5313 = x5199 == 1;
bool x5314 = x5312 == 1;
bool x5315 = x5313 || x5314;
bool x5316 = x5199 == x5312;
bool x5317 = x5315 || x5316;
x5318 = x5317;
} else {
x5318 = false;
}
bool x5322;
if (x5318) {
x5322 = x5321;
} else {
x5322 = false;
}
bool x5323;
if (x5322) {
x5323 = x5321;
} else {
x5323 = false;
}
if (x5323) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5199,x5201,x5201,1,x5312,1,1);
assert(false && "");
}
bool x5329 = x5199 <= x5312;
int32_t x5330;
if (x5329) {
x5330 = x5312;
} else {
x5330 = x5199;
}
bool x5336 = x5330 > 0;
bool x5338;
if (x5336) {
x5338 = x5337;
} else {
x5338 = false;
}
bool x5339;
if (x5338) {
x5339 = x5337;
} else {
x5339 = false;
}
if (x5339) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(5199) x Sym(5201) x Sym(5201)"," x Const(1) x Sym(5312) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x5334 = x5330 * x5333;
int32_t x5335 = 64 * x5334;
float* x5345 = (float*)myMalloc(x5335 * sizeof(float));;
int32_t x5346 = 0;
int32_t x5347 = 0;
int32_t x5348 = 0;
bool x5394 = x5199 > 1;
bool x5398 = x5312 > 1;
for(int x5349=0; x5349 < 64; x5349++) {
int32_t x5350 = x5347;
int32_t x5351 = x5348;
int32_t x5352 = x5346;
int32_t x5353 = x5352;
int32_t x5354 = x5350;
int32_t x5355 = x5351;
for(int x5357=0; x5357 < x5330; x5357++) {
int32_t x5358 = x5354;
int32_t x5359 = x5355;
int32_t x5360 = x5353;
int32_t x5361 = x5360;
int32_t x5362 = x5358;
int32_t x5363 = x5359;
for(int x5365=0; x5365 < x5332; x5365++) {
int32_t x5366 = x5362;
int32_t x5367 = x5363;
int32_t x5368 = x5361;
int32_t x5369 = x5368;
int32_t x5370 = x5366;
int32_t x5371 = x5367;
for(int x5372=0; x5372 < x5332; x5372++) {
int32_t x5373 = x5369;
int32_t x5374 = x5370;
float x5375 = x5214[x5374];
int32_t x5376 = x5371;
float x5377 = x5281[x5376];
float x5378 = x5375 / x5377;
x5345[x5373] = x5378;
x5369 += 1;
if (x5381) {
x5370 += 1;
} else {
}

}
x5361 += x5332;
if (x5381) {
x5362 += x5201;
} else {
}

}
x5353 += x5333;
if (x5394) {
x5354 += x5202;
} else {
}
if (x5398) {
x5355 += 1;
} else {
}

}
x5346 += x5334;
x5347 += x5203;

}
int32_t x5408 = 0;
int32_t x5409 = 1;
x5409 *= 1;
x5408 += 1;
x5409 *= 1;
x5409 *= 1;
int32_t x5414 = x5408;
bool x5415 = x5414 >= 2;
if (x5415) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5420 = x5414 == 0;
if (x5420) {
int32_t x5421 = x5409;
bool x5422 = x5421 == 64;
if (x5422) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5429 = x5409;
int32_t x5430 = 64 / x5429;
bool x5436;
if (x452) {
bool x5431 = x5330 == 1;
bool x5432 = x5430 == 1;
bool x5433 = x5431 || x5432;
bool x5434 = x5330 == x5430;
bool x5435 = x5433 || x5434;
x5436 = x5435;
} else {
x5436 = false;
}
bool x5440;
if (x5436) {
x5440 = x5439;
} else {
x5440 = false;
}
bool x5441;
if (x5440) {
x5441 = x5439;
} else {
x5441 = false;
}
if (x5441) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5330,x5332,x5332,1,x5430,1,1);
assert(false && "");
}
bool x5447 = x5330 <= x5430;
int32_t x5448;
if (x5447) {
x5448 = x5430;
} else {
x5448 = x5330;
}
bool x5454 = x5448 > 0;
bool x5456;
if (x5454) {
x5456 = x5455;
} else {
x5456 = false;
}
bool x5457;
if (x5456) {
x5457 = x5455;
} else {
x5457 = false;
}
if (x5457) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(5330) x Sym(5332) x Sym(5332)"," x Const(1) x Sym(5430) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x5452 = x5448 * x5451;
int32_t x5453 = 64 * x5452;
float* x5463 = (float*)myMalloc(x5453 * sizeof(float));;
int32_t x5464 = 0;
int32_t x5465 = 0;
int32_t x5466 = 0;
bool x5512 = x5330 > 1;
bool x5516 = x5430 > 1;
for(int x5467=0; x5467 < 64; x5467++) {
int32_t x5468 = x5465;
int32_t x5469 = x5466;
int32_t x5470 = x5464;
int32_t x5471 = x5470;
int32_t x5472 = x5468;
int32_t x5473 = x5469;
for(int x5475=0; x5475 < x5448; x5475++) {
int32_t x5476 = x5472;
int32_t x5477 = x5473;
int32_t x5478 = x5471;
int32_t x5479 = x5478;
int32_t x5480 = x5476;
int32_t x5481 = x5477;
for(int x5483=0; x5483 < x5450; x5483++) {
int32_t x5484 = x5480;
int32_t x5485 = x5481;
int32_t x5486 = x5479;
int32_t x5487 = x5486;
int32_t x5488 = x5484;
int32_t x5489 = x5485;
for(int x5490=0; x5490 < x5450; x5490++) {
int32_t x5491 = x5487;
int32_t x5492 = x5488;
float x5493 = x5345[x5492];
int32_t x5494 = x5489;
float x5495 = x81[x5494];
float x5496 = x5493 * x5495;
x5463[x5491] = x5496;
x5487 += 1;
if (x5499) {
x5488 += 1;
} else {
}

}
x5479 += x5450;
if (x5499) {
x5480 += x5332;
} else {
}

}
x5471 += x5451;
if (x5512) {
x5472 += x5333;
} else {
}
if (x5516) {
x5473 += 1;
} else {
}

}
x5464 += x5452;
x5465 += x5334;

}
int32_t x5526 = 0;
int32_t x5527 = 1;
x5527 *= 1;
x5526 += 1;
x5527 *= 1;
x5527 *= 1;
int32_t x5532 = x5526;
bool x5533 = x5532 >= 2;
if (x5533) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5538 = x5532 == 0;
if (x5538) {
int32_t x5539 = x5527;
bool x5540 = x5539 == 64;
if (x5540) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5547 = x5527;
int32_t x5548 = 64 / x5547;
bool x5554;
if (x452) {
bool x5549 = x5448 == 1;
bool x5550 = x5548 == 1;
bool x5551 = x5549 || x5550;
bool x5552 = x5448 == x5548;
bool x5553 = x5551 || x5552;
x5554 = x5553;
} else {
x5554 = false;
}
bool x5558;
if (x5554) {
x5558 = x5557;
} else {
x5558 = false;
}
bool x5559;
if (x5558) {
x5559 = x5557;
} else {
x5559 = false;
}
if (x5559) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5448,x5450,x5450,1,x5548,1,1);
assert(false && "");
}
bool x5565 = x5448 <= x5548;
int32_t x5566;
if (x5565) {
x5566 = x5548;
} else {
x5566 = x5448;
}
bool x5572 = x5566 > 0;
bool x5574;
if (x5572) {
x5574 = x5573;
} else {
x5574 = false;
}
bool x5575;
if (x5574) {
x5575 = x5573;
} else {
x5575 = false;
}
if (x5575) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(5448) x Sym(5450) x Sym(5450)"," x Const(1) x Sym(5548) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x5570 = x5566 * x5569;
int32_t x5571 = 64 * x5570;
float* x5581 = (float*)myMalloc(x5571 * sizeof(float));;
int32_t x5582 = 0;
int32_t x5583 = 0;
int32_t x5584 = 0;
bool x5630 = x5448 > 1;
bool x5634 = x5548 > 1;
for(int x5585=0; x5585 < 64; x5585++) {
int32_t x5586 = x5583;
int32_t x5587 = x5584;
int32_t x5588 = x5582;
int32_t x5589 = x5588;
int32_t x5590 = x5586;
int32_t x5591 = x5587;
for(int x5593=0; x5593 < x5566; x5593++) {
int32_t x5594 = x5590;
int32_t x5595 = x5591;
int32_t x5596 = x5589;
int32_t x5597 = x5596;
int32_t x5598 = x5594;
int32_t x5599 = x5595;
for(int x5601=0; x5601 < x5568; x5601++) {
int32_t x5602 = x5598;
int32_t x5603 = x5599;
int32_t x5604 = x5597;
int32_t x5605 = x5604;
int32_t x5606 = x5602;
int32_t x5607 = x5603;
for(int x5608=0; x5608 < x5568; x5608++) {
int32_t x5609 = x5605;
int32_t x5610 = x5606;
float x5611 = x5463[x5610];
int32_t x5612 = x5607;
float x5613 = x24[x5612];
float x5614 = x5611 + x5613;
x5581[x5609] = x5614;
x5605 += 1;
if (x5617) {
x5606 += 1;
} else {
}

}
x5597 += x5568;
if (x5617) {
x5598 += x5450;
} else {
}

}
x5589 += x5569;
if (x5630) {
x5590 += x5451;
} else {
}
if (x5634) {
x5591 += 1;
} else {
}

}
x5582 += x5570;
x5583 += x5452;

}
float* x5644 = (float*)myMalloc(x5571 * sizeof(float));;
for(int x5646=0; x5646 < x5571; x5646++) {
float x5647 = x5581[x5646];
bool x5648 = x5647 < 0.0f;
if (x5648) {
x5644[x5646] = 0.0f;
} else {
float x5651 = x5581[x5646];
x5644[x5646] = x5651;
}

}
float* x5666 = (float*)myMalloc(x5665 * sizeof(float));;
int32_t x5667 = 9 * x5566;
int32_t x5670 = 64 * x5667;
int32_t x5671 = x5670 * x5661;
float* x5672 = (float*)myMalloc(x5671 * sizeof(float));;
int32_t x5668 = x5667 * x5661;
int32_t x5680 = x5566 * 3;
int32_t x5681 = x5680 * 3;
for(int x5673=0; x5673 < 64; x5673++) {
int32_t x5674 = x5673 * x5570;
float* x5675 = x5644+x5674;
int32_t x5676 = x5673 * x5662;
float* x5677 = x5666+x5676;
int32_t x5678 = x5673 * x5668;
float* x5679 = x5672+x5678;
for(int x5683=0; x5683 < x5681; x5683++) {
int32_t x5684 = x5683 / 9;
int32_t x5688 = x5684 * 3;
int32_t x5689 = x5688 * 3;
int32_t x5690 = x5689 * x5660;
int32_t x5691 = x5690 * x5660;
int32_t x5685 = x5683 % 9;
int32_t x5686 = x5685 / 3;
int32_t x5692 = x5686 * 3;
int32_t x5693 = x5692 * x5660;
int32_t x5694 = x5693 * x5660;
int32_t x5695 = x5691 + x5694;
int32_t x5687 = x5685 % 3;
int32_t x5696 = x5687 * x5660;
int32_t x5697 = x5696 * x5660;
int32_t x5698 = x5695 + x5697;
float* x5699 = x5679+x5698;
int32_t x5700 = x5684 * x5568;
int32_t x5701 = x5700 * x5568;
float* x5702 = x5675+x5701;
int32_t x5715 = 1 - x5687;
bool x5716 = x5715 > 0;
int32_t x5717;
if (x5716) {
x5717 = x5715;
} else {
x5717 = 0;
}
int32_t x5718 = 3 - x5687;
int32_t x5719 = x5718 - 1;
int32_t x5720 = 1 - x5719;
bool x5721 = x5720 > 0;
int32_t x5722;
if (x5721) {
x5722 = x5720;
} else {
x5722 = 0;
}
int32_t x5723 = x5660 - x5722;
int32_t x5724 = x5723 - x5717;
bool x5725 = x5724 <= 0;
bool x5729 = x5717 > 0;
int32_t x5714 = -1 + x5687;
bool x5742 = x5722 > 0;
for(int x5704=0; x5704 < x5660; x5704++) {
int32_t x5705 = x5704 - 1;
int32_t x5706 = x5705 + x5686;
bool x5707 = x5706 < 0;
bool x5708 = x5706 >= x5568;
bool x5709 = x5707 || x5708;
if (x5709) {
int32_t x5710 = x5704 * x5660;
float* x5711 = x5699+x5710;
memset(x5711, 0, 4 * x5660);;
} else {
if (x5725) {
int32_t x5710 = x5704 * x5660;
float* x5726 = x5699+x5710;
memset(x5726, 0, 4 * x5660);;
} else {
int32_t x5710 = x5704 * x5660;
if (x5729) {
float* x5730 = x5699+x5710;
memset(x5730, 0, 4 * x5717);;
} else {
}
// may have segfault here
int32_t x5735 = x5710 + x5717;
float* x5736 = x5699+x5735;
int32_t x5737 = x5706 * x5568;
int32_t x5738 = x5737 + x5714;
int32_t x5739 = x5738 + x5717;
float* x5740 = x5702+x5739;
memcpy(x5736, x5740, 4 * x5724);;
if (x5742) {
int32_t x5743 = x5710 + x5660;
int32_t x5744 = x5743 - x5722;
float* x5745 = x5699+x5744;
memset(x5745, 0, 4 * x5722);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x5661,x5667,1,x73,x5667,x5679,x5661,1,x5677,x5661);

}
int32_t x5760 = 0;
int32_t x5761 = 1;
x5761 *= 1;
x5760 += 1;
x5761 *= 1;
x5761 *= 1;
int32_t x5766 = x5760;
bool x5767 = x5766 >= 2;
if (x5767) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5772 = x5766 == 0;
if (x5772) {
int32_t x5773 = x5761;
bool x5774 = x5773 == 64;
if (x5774) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5781 = x5761;
int32_t x5782 = 64 / x5781;
bool x5786;
if (x452) {
bool x5783 = x5782 == 1;
bool x5784 = 64 == x5782;
bool x5785 = x5783 || x5784;
x5786 = x5785;
} else {
x5786 = false;
}
bool x5790;
if (x5786) {
x5790 = x5789;
} else {
x5790 = false;
}
bool x5791;
if (x5790) {
x5791 = x5789;
} else {
x5791 = false;
}
if (x5791) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,64,x5660,x5660,1,x5782,1,1);
assert(false && "");
}
bool x5797 = 64 <= x5782;
int32_t x5798;
if (x5797) {
x5798 = x5782;
} else {
x5798 = 64;
}
bool x5804 = x5798 > 0;
bool x5806;
if (x5804) {
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
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(64) x Sym(5660) x Sym(5660)"," x Const(1) x Sym(5782) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x5802 = x5798 * x5801;
int32_t x5803 = 64 * x5802;
float* x5813 = (float*)myMalloc(x5803 * sizeof(float));;
int32_t x5814 = 0;
int32_t x5815 = 0;
int32_t x5816 = 0;
bool x5863 = x5782 > 1;
for(int x5817=0; x5817 < 64; x5817++) {
int32_t x5818 = x5815;
int32_t x5819 = x5816;
int32_t x5820 = x5814;
int32_t x5821 = x5820;
int32_t x5822 = x5818;
int32_t x5823 = x5819;
for(int x5825=0; x5825 < x5798; x5825++) {
int32_t x5826 = x5822;
int32_t x5827 = x5823;
int32_t x5828 = x5821;
int32_t x5829 = x5828;
int32_t x5830 = x5826;
int32_t x5831 = x5827;
for(int x5833=0; x5833 < x5800; x5833++) {
int32_t x5834 = x5830;
int32_t x5835 = x5831;
int32_t x5836 = x5829;
int32_t x5837 = x5836;
int32_t x5838 = x5834;
int32_t x5839 = x5835;
for(int x5840=0; x5840 < x5800; x5840++) {
int32_t x5841 = x5837;
int32_t x5842 = x5838;
float x5843 = x5666[x5842];
int32_t x5844 = x5839;
float x5845 = x179[x5844];
float x5846 = x5843 - x5845;
x5813[x5841] = x5846;
x5837 += 1;
if (x5849) {
x5838 += 1;
} else {
}

}
x5829 += x5800;
if (x5849) {
x5830 += x5660;
} else {
}

}
x5821 += x5801;
x5822 += x5661;
if (x5863) {
x5823 += 1;
} else {
}

}
x5814 += x5802;
x5815 += x5662;

}
float* x5873 = (float*)myMalloc(64 * sizeof(float));;
for(int x5874=0; x5874 < 64; x5874++) {
float x5875 = x118[x5874];
float x5876 = x5875 + 1.0E-5f;
x5873[x5874] = x5876;

}
float* x5880 = (float*)myMalloc(64 * sizeof(float));;
for(int x5881=0; x5881 < 64; x5881++) {
float x5882 = x5873[x5881];
double x5883 = (double)x5882;
double x5884 = sqrt(x5883);
float x5885 = (float)x5884;
x5880[x5881] = x5885;

}
int32_t x5889 = 0;
int32_t x5890 = 1;
x5890 *= 1;
x5889 += 1;
x5890 *= 1;
x5890 *= 1;
int32_t x5895 = x5889;
bool x5896 = x5895 >= 2;
if (x5896) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x5901 = x5895 == 0;
if (x5901) {
int32_t x5902 = x5890;
bool x5903 = x5902 == 64;
if (x5903) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x5910 = x5890;
int32_t x5911 = 64 / x5910;
bool x5917;
if (x452) {
bool x5912 = x5798 == 1;
bool x5913 = x5911 == 1;
bool x5914 = x5912 || x5913;
bool x5915 = x5798 == x5911;
bool x5916 = x5914 || x5915;
x5917 = x5916;
} else {
x5917 = false;
}
bool x5921;
if (x5917) {
x5921 = x5920;
} else {
x5921 = false;
}
bool x5922;
if (x5921) {
x5922 = x5920;
} else {
x5922 = false;
}
if (x5922) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5798,x5800,x5800,1,x5911,1,1);
assert(false && "");
}
bool x5928 = x5798 <= x5911;
int32_t x5929;
if (x5928) {
x5929 = x5911;
} else {
x5929 = x5798;
}
bool x5935 = x5929 > 0;
bool x5937;
if (x5935) {
x5937 = x5936;
} else {
x5937 = false;
}
bool x5938;
if (x5937) {
x5938 = x5936;
} else {
x5938 = false;
}
if (x5938) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(5798) x Sym(5800) x Sym(5800)"," x Const(1) x Sym(5911) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x5933 = x5929 * x5932;
int32_t x5934 = 64 * x5933;
float* x5944 = (float*)myMalloc(x5934 * sizeof(float));;
int32_t x5945 = 0;
int32_t x5946 = 0;
int32_t x5947 = 0;
bool x5993 = x5798 > 1;
bool x5997 = x5911 > 1;
for(int x5948=0; x5948 < 64; x5948++) {
int32_t x5949 = x5946;
int32_t x5950 = x5947;
int32_t x5951 = x5945;
int32_t x5952 = x5951;
int32_t x5953 = x5949;
int32_t x5954 = x5950;
for(int x5956=0; x5956 < x5929; x5956++) {
int32_t x5957 = x5953;
int32_t x5958 = x5954;
int32_t x5959 = x5952;
int32_t x5960 = x5959;
int32_t x5961 = x5957;
int32_t x5962 = x5958;
for(int x5964=0; x5964 < x5931; x5964++) {
int32_t x5965 = x5961;
int32_t x5966 = x5962;
int32_t x5967 = x5960;
int32_t x5968 = x5967;
int32_t x5969 = x5965;
int32_t x5970 = x5966;
for(int x5971=0; x5971 < x5931; x5971++) {
int32_t x5972 = x5968;
int32_t x5973 = x5969;
float x5974 = x5813[x5973];
int32_t x5975 = x5970;
float x5976 = x5880[x5975];
float x5977 = x5974 / x5976;
x5944[x5972] = x5977;
x5968 += 1;
if (x5980) {
x5969 += 1;
} else {
}

}
x5960 += x5931;
if (x5980) {
x5961 += x5800;
} else {
}

}
x5952 += x5932;
if (x5993) {
x5953 += x5801;
} else {
}
if (x5997) {
x5954 += 1;
} else {
}

}
x5945 += x5933;
x5946 += x5802;

}
int32_t x6007 = 0;
int32_t x6008 = 1;
x6008 *= 1;
x6007 += 1;
x6008 *= 1;
x6008 *= 1;
int32_t x6013 = x6007;
bool x6014 = x6013 >= 2;
if (x6014) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6019 = x6013 == 0;
if (x6019) {
int32_t x6020 = x6008;
bool x6021 = x6020 == 64;
if (x6021) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6028 = x6008;
int32_t x6029 = 64 / x6028;
bool x6035;
if (x452) {
bool x6030 = x5929 == 1;
bool x6031 = x6029 == 1;
bool x6032 = x6030 || x6031;
bool x6033 = x5929 == x6029;
bool x6034 = x6032 || x6033;
x6035 = x6034;
} else {
x6035 = false;
}
bool x6039;
if (x6035) {
x6039 = x6038;
} else {
x6039 = false;
}
bool x6040;
if (x6039) {
x6040 = x6038;
} else {
x6040 = false;
}
if (x6040) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x5929,x5931,x5931,1,x6029,1,1);
assert(false && "");
}
bool x6046 = x5929 <= x6029;
int32_t x6047;
if (x6046) {
x6047 = x6029;
} else {
x6047 = x5929;
}
bool x6053 = x6047 > 0;
bool x6055;
if (x6053) {
x6055 = x6054;
} else {
x6055 = false;
}
bool x6056;
if (x6055) {
x6056 = x6054;
} else {
x6056 = false;
}
if (x6056) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(5929) x Sym(5931) x Sym(5931)"," x Const(1) x Sym(6029) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x6051 = x6047 * x6050;
int32_t x6052 = 64 * x6051;
float* x6062 = (float*)myMalloc(x6052 * sizeof(float));;
int32_t x6063 = 0;
int32_t x6064 = 0;
int32_t x6065 = 0;
bool x6111 = x5929 > 1;
bool x6115 = x6029 > 1;
for(int x6066=0; x6066 < 64; x6066++) {
int32_t x6067 = x6064;
int32_t x6068 = x6065;
int32_t x6069 = x6063;
int32_t x6070 = x6069;
int32_t x6071 = x6067;
int32_t x6072 = x6068;
for(int x6074=0; x6074 < x6047; x6074++) {
int32_t x6075 = x6071;
int32_t x6076 = x6072;
int32_t x6077 = x6070;
int32_t x6078 = x6077;
int32_t x6079 = x6075;
int32_t x6080 = x6076;
for(int x6082=0; x6082 < x6049; x6082++) {
int32_t x6083 = x6079;
int32_t x6084 = x6080;
int32_t x6085 = x6078;
int32_t x6086 = x6085;
int32_t x6087 = x6083;
int32_t x6088 = x6084;
for(int x6089=0; x6089 < x6049; x6089++) {
int32_t x6090 = x6086;
int32_t x6091 = x6087;
float x6092 = x5944[x6091];
int32_t x6093 = x6088;
float x6094 = x72[x6093];
float x6095 = x6092 * x6094;
x6062[x6090] = x6095;
x6086 += 1;
if (x6098) {
x6087 += 1;
} else {
}

}
x6078 += x6049;
if (x6098) {
x6079 += x5931;
} else {
}

}
x6070 += x6050;
if (x6111) {
x6071 += x5932;
} else {
}
if (x6115) {
x6072 += 1;
} else {
}

}
x6063 += x6051;
x6064 += x5933;

}
int32_t x6125 = 0;
int32_t x6126 = 1;
x6126 *= 1;
x6125 += 1;
x6126 *= 1;
x6126 *= 1;
int32_t x6131 = x6125;
bool x6132 = x6131 >= 2;
if (x6132) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6137 = x6131 == 0;
if (x6137) {
int32_t x6138 = x6126;
bool x6139 = x6138 == 64;
if (x6139) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6146 = x6126;
int32_t x6147 = 64 / x6146;
bool x6153;
if (x452) {
bool x6148 = x6047 == 1;
bool x6149 = x6147 == 1;
bool x6150 = x6148 || x6149;
bool x6151 = x6047 == x6147;
bool x6152 = x6150 || x6151;
x6153 = x6152;
} else {
x6153 = false;
}
bool x6157;
if (x6153) {
x6157 = x6156;
} else {
x6157 = false;
}
bool x6158;
if (x6157) {
x6158 = x6156;
} else {
x6158 = false;
}
if (x6158) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6047,x6049,x6049,1,x6147,1,1);
assert(false && "");
}
bool x6164 = x6047 <= x6147;
int32_t x6165;
if (x6164) {
x6165 = x6147;
} else {
x6165 = x6047;
}
bool x6171 = x6165 > 0;
bool x6173;
if (x6171) {
x6173 = x6172;
} else {
x6173 = false;
}
bool x6174;
if (x6173) {
x6174 = x6172;
} else {
x6174 = false;
}
if (x6174) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(6047) x Sym(6049) x Sym(6049)"," x Const(1) x Sym(6147) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x6169 = x6165 * x6168;
int32_t x6170 = 64 * x6169;
float* x6180 = (float*)myMalloc(x6170 * sizeof(float));;
int32_t x6181 = 0;
int32_t x6182 = 0;
int32_t x6183 = 0;
bool x6229 = x6047 > 1;
bool x6233 = x6147 > 1;
for(int x6184=0; x6184 < 64; x6184++) {
int32_t x6185 = x6182;
int32_t x6186 = x6183;
int32_t x6187 = x6181;
int32_t x6188 = x6187;
int32_t x6189 = x6185;
int32_t x6190 = x6186;
for(int x6192=0; x6192 < x6165; x6192++) {
int32_t x6193 = x6189;
int32_t x6194 = x6190;
int32_t x6195 = x6188;
int32_t x6196 = x6195;
int32_t x6197 = x6193;
int32_t x6198 = x6194;
for(int x6200=0; x6200 < x6167; x6200++) {
int32_t x6201 = x6197;
int32_t x6202 = x6198;
int32_t x6203 = x6196;
int32_t x6204 = x6203;
int32_t x6205 = x6201;
int32_t x6206 = x6202;
for(int x6207=0; x6207 < x6167; x6207++) {
int32_t x6208 = x6204;
int32_t x6209 = x6205;
float x6210 = x6062[x6209];
int32_t x6211 = x6206;
float x6212 = x135[x6211];
float x6213 = x6210 + x6212;
x6180[x6208] = x6213;
x6204 += 1;
if (x6216) {
x6205 += 1;
} else {
}

}
x6196 += x6167;
if (x6216) {
x6197 += x6049;
} else {
}

}
x6188 += x6168;
if (x6229) {
x6189 += x6050;
} else {
}
if (x6233) {
x6190 += 1;
} else {
}

}
x6181 += x6169;
x6182 += x6051;

}
float* x6243 = (float*)myMalloc(x6170 * sizeof(float));;
for(int x6245=0; x6245 < x6170; x6245++) {
float x6246 = x6180[x6245];
bool x6247 = x6246 < 0.0f;
if (x6247) {
x6243[x6245] = 0.0f;
} else {
float x6250 = x6180[x6245];
x6243[x6245] = x6250;
}

}
float* x6264 = (float*)myMalloc(x6263 * sizeof(float));;
int32_t x6267 = 64 * x6165;
int32_t x6268 = x6267 * x6259;
float* x6269 = (float*)myMalloc(x6268 * sizeof(float));;
int32_t x6265 = x6165 * x6259;
for(int x6270=0; x6270 < 64; x6270++) {
int32_t x6271 = x6270 * x6169;
float* x6272 = x6243+x6271;
int32_t x6273 = x6270 * x6260;
float* x6274 = x6264+x6273;
int32_t x6275 = x6270 * x6265;
float* x6276 = x6269+x6275;
for(int x6277=0; x6277 < x6165; x6277++) {
int32_t x6278 = x6277 / 1;
int32_t x6282 = x6278 * x6258;
int32_t x6283 = x6282 * x6258;
int32_t x6279 = x6277 % 1;
int32_t x6280 = x6279 / 1;
int32_t x6284 = x6280 * x6258;
int32_t x6285 = x6284 * x6258;
int32_t x6286 = x6283 + x6285;
int32_t x6281 = x6279 % 1;
int32_t x6287 = x6281 * x6258;
int32_t x6288 = x6287 * x6258;
int32_t x6289 = x6286 + x6288;
float* x6290 = x6276+x6289;
int32_t x6291 = x6278 * x6167;
int32_t x6292 = x6291 * x6167;
float* x6293 = x6272+x6292;
for(int x6295=0; x6295 < x6258; x6295++) {
int32_t x6297 = x6295 * x6258;
float* x6298 = x6290+x6297;
int32_t x6296 = x6295 + x6280;
int32_t x6299 = x6296 * x6167;
int32_t x6300 = x6299 + x6281;
float* x6301 = x6293+x6300;
memcpy(x6298, x6301, 4 * x6258);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x6259,x6165,1,x87,x6165,x6276,x6259,1,x6274,x6259);

}
int32_t x6310 = 0;
int32_t x6311 = 1;
x6311 *= 1;
x6310 += 1;
x6311 *= 1;
x6311 *= 1;
int32_t x6316 = x6310;
bool x6317 = x6316 >= 2;
if (x6317) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6322 = x6316 == 0;
if (x6322) {
int32_t x6323 = x6311;
bool x6324 = x6323 == 256;
if (x6324) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6331 = x6311;
int32_t x6332 = 256 / x6331;
bool x6336;
if (x452) {
bool x6333 = x6332 == 1;
bool x6334 = 256 == x6332;
bool x6335 = x6333 || x6334;
x6336 = x6335;
} else {
x6336 = false;
}
bool x6340;
if (x6336) {
x6340 = x6339;
} else {
x6340 = false;
}
bool x6341;
if (x6340) {
x6341 = x6339;
} else {
x6341 = false;
}
if (x6341) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x6258,x6258,1,x6332,1,1);
assert(false && "");
}
bool x6347 = 256 <= x6332;
int32_t x6348;
if (x6347) {
x6348 = x6332;
} else {
x6348 = 256;
}
bool x6354 = x6348 > 0;
bool x6356;
if (x6354) {
x6356 = x6355;
} else {
x6356 = false;
}
bool x6357;
if (x6356) {
x6357 = x6355;
} else {
x6357 = false;
}
if (x6357) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(6258) x Sym(6258)"," x Const(1) x Sym(6332) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x6352 = x6348 * x6351;
int32_t x6353 = 64 * x6352;
float* x6363 = (float*)myMalloc(x6353 * sizeof(float));;
int32_t x6364 = 0;
int32_t x6365 = 0;
int32_t x6366 = 0;
bool x6413 = x6332 > 1;
for(int x6367=0; x6367 < 64; x6367++) {
int32_t x6368 = x6365;
int32_t x6369 = x6366;
int32_t x6370 = x6364;
int32_t x6371 = x6370;
int32_t x6372 = x6368;
int32_t x6373 = x6369;
for(int x6375=0; x6375 < x6348; x6375++) {
int32_t x6376 = x6372;
int32_t x6377 = x6373;
int32_t x6378 = x6371;
int32_t x6379 = x6378;
int32_t x6380 = x6376;
int32_t x6381 = x6377;
for(int x6383=0; x6383 < x6350; x6383++) {
int32_t x6384 = x6380;
int32_t x6385 = x6381;
int32_t x6386 = x6379;
int32_t x6387 = x6386;
int32_t x6388 = x6384;
int32_t x6389 = x6385;
for(int x6390=0; x6390 < x6350; x6390++) {
int32_t x6391 = x6387;
int32_t x6392 = x6388;
float x6393 = x6264[x6392];
int32_t x6394 = x6389;
float x6395 = x184[x6394];
float x6396 = x6393 - x6395;
x6363[x6391] = x6396;
x6387 += 1;
if (x6399) {
x6388 += 1;
} else {
}

}
x6379 += x6350;
if (x6399) {
x6380 += x6258;
} else {
}

}
x6371 += x6351;
x6372 += x6259;
if (x6413) {
x6373 += 1;
} else {
}

}
x6364 += x6352;
x6365 += x6260;

}
float* x6423 = (float*)myMalloc(256 * sizeof(float));;
for(int x6424=0; x6424 < 256; x6424++) {
float x6425 = x133[x6424];
float x6426 = x6425 + 1.0E-5f;
x6423[x6424] = x6426;

}
float* x6430 = (float*)myMalloc(256 * sizeof(float));;
for(int x6431=0; x6431 < 256; x6431++) {
float x6432 = x6423[x6431];
double x6433 = (double)x6432;
double x6434 = sqrt(x6433);
float x6435 = (float)x6434;
x6430[x6431] = x6435;

}
int32_t x6439 = 0;
int32_t x6440 = 1;
x6440 *= 1;
x6439 += 1;
x6440 *= 1;
x6440 *= 1;
int32_t x6445 = x6439;
bool x6446 = x6445 >= 2;
if (x6446) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6451 = x6445 == 0;
if (x6451) {
int32_t x6452 = x6440;
bool x6453 = x6452 == 256;
if (x6453) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6460 = x6440;
int32_t x6461 = 256 / x6460;
bool x6467;
if (x452) {
bool x6462 = x6348 == 1;
bool x6463 = x6461 == 1;
bool x6464 = x6462 || x6463;
bool x6465 = x6348 == x6461;
bool x6466 = x6464 || x6465;
x6467 = x6466;
} else {
x6467 = false;
}
bool x6471;
if (x6467) {
x6471 = x6470;
} else {
x6471 = false;
}
bool x6472;
if (x6471) {
x6472 = x6470;
} else {
x6472 = false;
}
if (x6472) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6348,x6350,x6350,1,x6461,1,1);
assert(false && "");
}
bool x6478 = x6348 <= x6461;
int32_t x6479;
if (x6478) {
x6479 = x6461;
} else {
x6479 = x6348;
}
bool x6485 = x6479 > 0;
bool x6487;
if (x6485) {
x6487 = x6486;
} else {
x6487 = false;
}
bool x6488;
if (x6487) {
x6488 = x6486;
} else {
x6488 = false;
}
if (x6488) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(6348) x Sym(6350) x Sym(6350)"," x Const(1) x Sym(6461) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x6483 = x6479 * x6482;
int32_t x6484 = 64 * x6483;
float* x6494 = (float*)myMalloc(x6484 * sizeof(float));;
int32_t x6495 = 0;
int32_t x6496 = 0;
int32_t x6497 = 0;
bool x6543 = x6348 > 1;
bool x6547 = x6461 > 1;
for(int x6498=0; x6498 < 64; x6498++) {
int32_t x6499 = x6496;
int32_t x6500 = x6497;
int32_t x6501 = x6495;
int32_t x6502 = x6501;
int32_t x6503 = x6499;
int32_t x6504 = x6500;
for(int x6506=0; x6506 < x6479; x6506++) {
int32_t x6507 = x6503;
int32_t x6508 = x6504;
int32_t x6509 = x6502;
int32_t x6510 = x6509;
int32_t x6511 = x6507;
int32_t x6512 = x6508;
for(int x6514=0; x6514 < x6481; x6514++) {
int32_t x6515 = x6511;
int32_t x6516 = x6512;
int32_t x6517 = x6510;
int32_t x6518 = x6517;
int32_t x6519 = x6515;
int32_t x6520 = x6516;
for(int x6521=0; x6521 < x6481; x6521++) {
int32_t x6522 = x6518;
int32_t x6523 = x6519;
float x6524 = x6363[x6523];
int32_t x6525 = x6520;
float x6526 = x6430[x6525];
float x6527 = x6524 / x6526;
x6494[x6522] = x6527;
x6518 += 1;
if (x6530) {
x6519 += 1;
} else {
}

}
x6510 += x6481;
if (x6530) {
x6511 += x6350;
} else {
}

}
x6502 += x6482;
if (x6543) {
x6503 += x6351;
} else {
}
if (x6547) {
x6504 += 1;
} else {
}

}
x6495 += x6483;
x6496 += x6352;

}
int32_t x6557 = 0;
int32_t x6558 = 1;
x6558 *= 1;
x6557 += 1;
x6558 *= 1;
x6558 *= 1;
int32_t x6563 = x6557;
bool x6564 = x6563 >= 2;
if (x6564) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6569 = x6563 == 0;
if (x6569) {
int32_t x6570 = x6558;
bool x6571 = x6570 == 256;
if (x6571) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6578 = x6558;
int32_t x6579 = 256 / x6578;
bool x6585;
if (x452) {
bool x6580 = x6479 == 1;
bool x6581 = x6579 == 1;
bool x6582 = x6580 || x6581;
bool x6583 = x6479 == x6579;
bool x6584 = x6582 || x6583;
x6585 = x6584;
} else {
x6585 = false;
}
bool x6589;
if (x6585) {
x6589 = x6588;
} else {
x6589 = false;
}
bool x6590;
if (x6589) {
x6590 = x6588;
} else {
x6590 = false;
}
if (x6590) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6479,x6481,x6481,1,x6579,1,1);
assert(false && "");
}
bool x6596 = x6479 <= x6579;
int32_t x6597;
if (x6596) {
x6597 = x6579;
} else {
x6597 = x6479;
}
bool x6603 = x6597 > 0;
bool x6605;
if (x6603) {
x6605 = x6604;
} else {
x6605 = false;
}
bool x6606;
if (x6605) {
x6606 = x6604;
} else {
x6606 = false;
}
if (x6606) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(6479) x Sym(6481) x Sym(6481)"," x Const(1) x Sym(6579) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x6601 = x6597 * x6600;
int32_t x6602 = 64 * x6601;
float* x6612 = (float*)myMalloc(x6602 * sizeof(float));;
int32_t x6613 = 0;
int32_t x6614 = 0;
int32_t x6615 = 0;
bool x6661 = x6479 > 1;
bool x6665 = x6579 > 1;
for(int x6616=0; x6616 < 64; x6616++) {
int32_t x6617 = x6614;
int32_t x6618 = x6615;
int32_t x6619 = x6613;
int32_t x6620 = x6619;
int32_t x6621 = x6617;
int32_t x6622 = x6618;
for(int x6624=0; x6624 < x6597; x6624++) {
int32_t x6625 = x6621;
int32_t x6626 = x6622;
int32_t x6627 = x6620;
int32_t x6628 = x6627;
int32_t x6629 = x6625;
int32_t x6630 = x6626;
for(int x6632=0; x6632 < x6599; x6632++) {
int32_t x6633 = x6629;
int32_t x6634 = x6630;
int32_t x6635 = x6628;
int32_t x6636 = x6635;
int32_t x6637 = x6633;
int32_t x6638 = x6634;
for(int x6639=0; x6639 < x6599; x6639++) {
int32_t x6640 = x6636;
int32_t x6641 = x6637;
float x6642 = x6494[x6641];
int32_t x6643 = x6638;
float x6644 = x37[x6643];
float x6645 = x6642 * x6644;
x6612[x6640] = x6645;
x6636 += 1;
if (x6648) {
x6637 += 1;
} else {
}

}
x6628 += x6599;
if (x6648) {
x6629 += x6481;
} else {
}

}
x6620 += x6600;
if (x6661) {
x6621 += x6482;
} else {
}
if (x6665) {
x6622 += 1;
} else {
}

}
x6613 += x6601;
x6614 += x6483;

}
int32_t x6675 = 0;
int32_t x6676 = 1;
x6676 *= 1;
x6675 += 1;
x6676 *= 1;
x6676 *= 1;
int32_t x6681 = x6675;
bool x6682 = x6681 >= 2;
if (x6682) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6687 = x6681 == 0;
if (x6687) {
int32_t x6688 = x6676;
bool x6689 = x6688 == 256;
if (x6689) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6696 = x6676;
int32_t x6697 = 256 / x6696;
bool x6703;
if (x452) {
bool x6698 = x6597 == 1;
bool x6699 = x6697 == 1;
bool x6700 = x6698 || x6699;
bool x6701 = x6597 == x6697;
bool x6702 = x6700 || x6701;
x6703 = x6702;
} else {
x6703 = false;
}
bool x6707;
if (x6703) {
x6707 = x6706;
} else {
x6707 = false;
}
bool x6708;
if (x6707) {
x6708 = x6706;
} else {
x6708 = false;
}
if (x6708) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6597,x6599,x6599,1,x6697,1,1);
assert(false && "");
}
bool x6714 = x6597 <= x6697;
int32_t x6715;
if (x6714) {
x6715 = x6697;
} else {
x6715 = x6597;
}
bool x6721 = x6715 > 0;
bool x6723;
if (x6721) {
x6723 = x6722;
} else {
x6723 = false;
}
bool x6724;
if (x6723) {
x6724 = x6722;
} else {
x6724 = false;
}
if (x6724) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(6597) x Sym(6599) x Sym(6599)"," x Const(1) x Sym(6697) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x6719 = x6715 * x6718;
int32_t x6720 = 64 * x6719;
float* x6730 = (float*)myMalloc(x6720 * sizeof(float));;
int32_t x6731 = 0;
int32_t x6732 = 0;
int32_t x6733 = 0;
bool x6779 = x6597 > 1;
bool x6783 = x6697 > 1;
for(int x6734=0; x6734 < 64; x6734++) {
int32_t x6735 = x6732;
int32_t x6736 = x6733;
int32_t x6737 = x6731;
int32_t x6738 = x6737;
int32_t x6739 = x6735;
int32_t x6740 = x6736;
for(int x6742=0; x6742 < x6715; x6742++) {
int32_t x6743 = x6739;
int32_t x6744 = x6740;
int32_t x6745 = x6738;
int32_t x6746 = x6745;
int32_t x6747 = x6743;
int32_t x6748 = x6744;
for(int x6750=0; x6750 < x6717; x6750++) {
int32_t x6751 = x6747;
int32_t x6752 = x6748;
int32_t x6753 = x6746;
int32_t x6754 = x6753;
int32_t x6755 = x6751;
int32_t x6756 = x6752;
for(int x6757=0; x6757 < x6717; x6757++) {
int32_t x6758 = x6754;
int32_t x6759 = x6755;
float x6760 = x6612[x6759];
int32_t x6761 = x6756;
float x6762 = x247[x6761];
float x6763 = x6760 + x6762;
x6730[x6758] = x6763;
x6754 += 1;
if (x6766) {
x6755 += 1;
} else {
}

}
x6746 += x6717;
if (x6766) {
x6747 += x6599;
} else {
}

}
x6738 += x6718;
if (x6779) {
x6739 += x6600;
} else {
}
if (x6783) {
x6740 += 1;
} else {
}

}
x6731 += x6719;
x6732 += x6601;

}
bool x6793 = x6715 == 1;
bool x6794 = x6793 || x5005;
bool x6795 = x6715 == x4927;
bool x6796 = x6794 || x6795;
bool x6801;
if (x6796) {
x6801 = x6800;
} else {
x6801 = false;
}
bool x6802;
if (x6801) {
x6802 = x6800;
} else {
x6802 = false;
}
if (x6802) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6715,x6717,x6717,64,x4927,x4929,x4929);
assert(false && "");
}
int32_t x6815 = 0;
int32_t x6816 = 0;
int32_t x6817 = 0;
bool x6808 = x6715 <= x4927;
int32_t x6809;
if (x6808) {
x6809 = x4927;
} else {
x6809 = x6715;
}
bool x6868 = x6715 > 1;
int32_t x6813 = x6809 * x6812;
for(int x6818=0; x6818 < 64; x6818++) {
int32_t x6819 = x6816;
int32_t x6820 = x6817;
int32_t x6821 = x6815;
int32_t x6822 = x6821;
int32_t x6823 = x6819;
int32_t x6824 = x6820;
for(int x6826=0; x6826 < x6809; x6826++) {
int32_t x6827 = x6823;
int32_t x6828 = x6824;
int32_t x6829 = x6822;
int32_t x6830 = x6829;
int32_t x6831 = x6827;
int32_t x6832 = x6828;
for(int x6834=0; x6834 < x6811; x6834++) {
int32_t x6835 = x6831;
int32_t x6836 = x6832;
int32_t x6837 = x6830;
int32_t x6838 = x6837;
int32_t x6839 = x6835;
int32_t x6840 = x6836;
for(int x6841=0; x6841 < x6811; x6841++) {
int32_t x6842 = x6839;
float x6843 = x6730[x6842];
int32_t x6844 = x6840;
float x6845 = x5094[x6844];
float x6846 = x6843 + x6845;
x6730[x6842] = x6846;
x6838 += 1;
if (x6849) {
x6839 += 1;
} else {
}
if (x5061) {
x6840 += 1;
} else {
}

}
x6830 += x6811;
if (x6849) {
x6831 += x6717;
} else {
}
if (x5061) {
x6832 += x4929;
} else {
}

}
x6822 += x6812;
if (x6868) {
x6823 += x6718;
} else {
}
if (x5080) {
x6824 += x4930;
} else {
}

}
x6815 += x6813;
x6816 += x6719;
x6817 += x4931;

}
float* x6882 = (float*)myMalloc(x6720 * sizeof(float));;
for(int x6884=0; x6884 < x6720; x6884++) {
float x6885 = x6730[x6884];
bool x6886 = x6885 < 0.0f;
if (x6886) {
x6882[x6884] = 0.0f;
} else {
float x6889 = x6730[x6884];
x6882[x6884] = x6889;
}

}
float* x6903 = (float*)myMalloc(x6902 * sizeof(float));;
int32_t x6906 = 64 * x6715;
int32_t x6907 = x6906 * x6898;
float* x6908 = (float*)myMalloc(x6907 * sizeof(float));;
int32_t x6904 = x6715 * x6898;
for(int x6909=0; x6909 < 64; x6909++) {
int32_t x6910 = x6909 * x6719;
float* x6911 = x6882+x6910;
int32_t x6912 = x6909 * x6899;
float* x6913 = x6903+x6912;
int32_t x6914 = x6909 * x6904;
float* x6915 = x6908+x6914;
for(int x6916=0; x6916 < x6715; x6916++) {
int32_t x6917 = x6916 / 1;
int32_t x6921 = x6917 * x6897;
int32_t x6922 = x6921 * x6897;
int32_t x6918 = x6916 % 1;
int32_t x6919 = x6918 / 1;
int32_t x6923 = x6919 * x6897;
int32_t x6924 = x6923 * x6897;
int32_t x6925 = x6922 + x6924;
int32_t x6920 = x6918 % 1;
int32_t x6926 = x6920 * x6897;
int32_t x6927 = x6926 * x6897;
int32_t x6928 = x6925 + x6927;
float* x6929 = x6915+x6928;
int32_t x6930 = x6917 * x6717;
int32_t x6931 = x6930 * x6717;
float* x6932 = x6911+x6931;
for(int x6934=0; x6934 < x6897; x6934++) {
int32_t x6936 = x6934 * x6897;
float* x6937 = x6929+x6936;
int32_t x6935 = x6934 + x6919;
int32_t x6938 = x6935 * x6717;
int32_t x6939 = x6938 + x6920;
float* x6940 = x6932+x6939;
memcpy(x6937, x6940, 4 * x6897);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x6898,x6715,1,x11,x6715,x6915,x6898,1,x6913,x6898);

}
int32_t x6949 = 0;
int32_t x6950 = 1;
x6950 *= 1;
x6949 += 1;
x6950 *= 1;
x6950 *= 1;
int32_t x6955 = x6949;
bool x6956 = x6955 >= 2;
if (x6956) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x6961 = x6955 == 0;
if (x6961) {
int32_t x6962 = x6950;
bool x6963 = x6962 == 128;
if (x6963) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x6970 = x6950;
int32_t x6971 = 128 / x6970;
bool x6975;
if (x452) {
bool x6972 = x6971 == 1;
bool x6973 = 128 == x6971;
bool x6974 = x6972 || x6973;
x6975 = x6974;
} else {
x6975 = false;
}
bool x6979;
if (x6975) {
x6979 = x6978;
} else {
x6979 = false;
}
bool x6980;
if (x6979) {
x6980 = x6978;
} else {
x6980 = false;
}
if (x6980) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x6897,x6897,1,x6971,1,1);
assert(false && "");
}
bool x6986 = 128 <= x6971;
int32_t x6987;
if (x6986) {
x6987 = x6971;
} else {
x6987 = 128;
}
bool x6993 = x6987 > 0;
bool x6995;
if (x6993) {
x6995 = x6994;
} else {
x6995 = false;
}
bool x6996;
if (x6995) {
x6996 = x6994;
} else {
x6996 = false;
}
if (x6996) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(128) x Sym(6897) x Sym(6897)"," x Const(1) x Sym(6971) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x6991 = x6987 * x6990;
int32_t x6992 = 64 * x6991;
float* x7002 = (float*)myMalloc(x6992 * sizeof(float));;
int32_t x7003 = 0;
int32_t x7004 = 0;
int32_t x7005 = 0;
bool x7052 = x6971 > 1;
for(int x7006=0; x7006 < 64; x7006++) {
int32_t x7007 = x7004;
int32_t x7008 = x7005;
int32_t x7009 = x7003;
int32_t x7010 = x7009;
int32_t x7011 = x7007;
int32_t x7012 = x7008;
for(int x7014=0; x7014 < x6987; x7014++) {
int32_t x7015 = x7011;
int32_t x7016 = x7012;
int32_t x7017 = x7010;
int32_t x7018 = x7017;
int32_t x7019 = x7015;
int32_t x7020 = x7016;
for(int x7022=0; x7022 < x6989; x7022++) {
int32_t x7023 = x7019;
int32_t x7024 = x7020;
int32_t x7025 = x7018;
int32_t x7026 = x7025;
int32_t x7027 = x7023;
int32_t x7028 = x7024;
for(int x7029=0; x7029 < x6989; x7029++) {
int32_t x7030 = x7026;
int32_t x7031 = x7027;
float x7032 = x6903[x7031];
int32_t x7033 = x7028;
float x7034 = x204[x7033];
float x7035 = x7032 - x7034;
x7002[x7030] = x7035;
x7026 += 1;
if (x7038) {
x7027 += 1;
} else {
}

}
x7018 += x6989;
if (x7038) {
x7019 += x6897;
} else {
}

}
x7010 += x6990;
x7011 += x6898;
if (x7052) {
x7012 += 1;
} else {
}

}
x7003 += x6991;
x7004 += x6899;

}
float* x7062 = (float*)myMalloc(128 * sizeof(float));;
for(int x7064=0; x7064 < 128; x7064++) {
float x7065 = x134[x7064];
float x7066 = x7065 + 1.0E-5f;
x7062[x7064] = x7066;

}
float* x7070 = (float*)myMalloc(128 * sizeof(float));;
for(int x7071=0; x7071 < 128; x7071++) {
float x7072 = x7062[x7071];
double x7073 = (double)x7072;
double x7074 = sqrt(x7073);
float x7075 = (float)x7074;
x7070[x7071] = x7075;

}
int32_t x7079 = 0;
int32_t x7080 = 1;
x7080 *= 1;
x7079 += 1;
x7080 *= 1;
x7080 *= 1;
int32_t x7085 = x7079;
bool x7086 = x7085 >= 2;
if (x7086) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7091 = x7085 == 0;
if (x7091) {
int32_t x7092 = x7080;
bool x7093 = x7092 == 128;
if (x7093) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7100 = x7080;
int32_t x7101 = 128 / x7100;
bool x7107;
if (x452) {
bool x7102 = x6987 == 1;
bool x7103 = x7101 == 1;
bool x7104 = x7102 || x7103;
bool x7105 = x6987 == x7101;
bool x7106 = x7104 || x7105;
x7107 = x7106;
} else {
x7107 = false;
}
bool x7111;
if (x7107) {
x7111 = x7110;
} else {
x7111 = false;
}
bool x7112;
if (x7111) {
x7112 = x7110;
} else {
x7112 = false;
}
if (x7112) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x6987,x6989,x6989,1,x7101,1,1);
assert(false && "");
}
bool x7118 = x6987 <= x7101;
int32_t x7119;
if (x7118) {
x7119 = x7101;
} else {
x7119 = x6987;
}
bool x7125 = x7119 > 0;
bool x7127;
if (x7125) {
x7127 = x7126;
} else {
x7127 = false;
}
bool x7128;
if (x7127) {
x7128 = x7126;
} else {
x7128 = false;
}
if (x7128) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(6987) x Sym(6989) x Sym(6989)"," x Const(1) x Sym(7101) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x7123 = x7119 * x7122;
int32_t x7124 = 64 * x7123;
float* x7134 = (float*)myMalloc(x7124 * sizeof(float));;
int32_t x7135 = 0;
int32_t x7136 = 0;
int32_t x7137 = 0;
bool x7183 = x6987 > 1;
bool x7187 = x7101 > 1;
for(int x7138=0; x7138 < 64; x7138++) {
int32_t x7139 = x7136;
int32_t x7140 = x7137;
int32_t x7141 = x7135;
int32_t x7142 = x7141;
int32_t x7143 = x7139;
int32_t x7144 = x7140;
for(int x7146=0; x7146 < x7119; x7146++) {
int32_t x7147 = x7143;
int32_t x7148 = x7144;
int32_t x7149 = x7142;
int32_t x7150 = x7149;
int32_t x7151 = x7147;
int32_t x7152 = x7148;
for(int x7154=0; x7154 < x7121; x7154++) {
int32_t x7155 = x7151;
int32_t x7156 = x7152;
int32_t x7157 = x7150;
int32_t x7158 = x7157;
int32_t x7159 = x7155;
int32_t x7160 = x7156;
for(int x7161=0; x7161 < x7121; x7161++) {
int32_t x7162 = x7158;
int32_t x7163 = x7159;
float x7164 = x7002[x7163];
int32_t x7165 = x7160;
float x7166 = x7070[x7165];
float x7167 = x7164 / x7166;
x7134[x7162] = x7167;
x7158 += 1;
if (x7170) {
x7159 += 1;
} else {
}

}
x7150 += x7121;
if (x7170) {
x7151 += x6989;
} else {
}

}
x7142 += x7122;
if (x7183) {
x7143 += x6990;
} else {
}
if (x7187) {
x7144 += 1;
} else {
}

}
x7135 += x7123;
x7136 += x6991;

}
int32_t x7197 = 0;
int32_t x7198 = 1;
x7198 *= 1;
x7197 += 1;
x7198 *= 1;
x7198 *= 1;
int32_t x7203 = x7197;
bool x7204 = x7203 >= 2;
if (x7204) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7209 = x7203 == 0;
if (x7209) {
int32_t x7210 = x7198;
bool x7211 = x7210 == 128;
if (x7211) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7218 = x7198;
int32_t x7219 = 128 / x7218;
bool x7225;
if (x452) {
bool x7220 = x7119 == 1;
bool x7221 = x7219 == 1;
bool x7222 = x7220 || x7221;
bool x7223 = x7119 == x7219;
bool x7224 = x7222 || x7223;
x7225 = x7224;
} else {
x7225 = false;
}
bool x7229;
if (x7225) {
x7229 = x7228;
} else {
x7229 = false;
}
bool x7230;
if (x7229) {
x7230 = x7228;
} else {
x7230 = false;
}
if (x7230) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7119,x7121,x7121,1,x7219,1,1);
assert(false && "");
}
bool x7236 = x7119 <= x7219;
int32_t x7237;
if (x7236) {
x7237 = x7219;
} else {
x7237 = x7119;
}
bool x7243 = x7237 > 0;
bool x7245;
if (x7243) {
x7245 = x7244;
} else {
x7245 = false;
}
bool x7246;
if (x7245) {
x7246 = x7244;
} else {
x7246 = false;
}
if (x7246) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(7119) x Sym(7121) x Sym(7121)"," x Const(1) x Sym(7219) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x7241 = x7237 * x7240;
int32_t x7242 = 64 * x7241;
float* x7252 = (float*)myMalloc(x7242 * sizeof(float));;
int32_t x7253 = 0;
int32_t x7254 = 0;
int32_t x7255 = 0;
bool x7301 = x7119 > 1;
bool x7305 = x7219 > 1;
for(int x7256=0; x7256 < 64; x7256++) {
int32_t x7257 = x7254;
int32_t x7258 = x7255;
int32_t x7259 = x7253;
int32_t x7260 = x7259;
int32_t x7261 = x7257;
int32_t x7262 = x7258;
for(int x7264=0; x7264 < x7237; x7264++) {
int32_t x7265 = x7261;
int32_t x7266 = x7262;
int32_t x7267 = x7260;
int32_t x7268 = x7267;
int32_t x7269 = x7265;
int32_t x7270 = x7266;
for(int x7272=0; x7272 < x7239; x7272++) {
int32_t x7273 = x7269;
int32_t x7274 = x7270;
int32_t x7275 = x7268;
int32_t x7276 = x7275;
int32_t x7277 = x7273;
int32_t x7278 = x7274;
for(int x7279=0; x7279 < x7239; x7279++) {
int32_t x7280 = x7276;
int32_t x7281 = x7277;
float x7282 = x7134[x7281];
int32_t x7283 = x7278;
float x7284 = x84[x7283];
float x7285 = x7282 * x7284;
x7252[x7280] = x7285;
x7276 += 1;
if (x7288) {
x7277 += 1;
} else {
}

}
x7268 += x7239;
if (x7288) {
x7269 += x7121;
} else {
}

}
x7260 += x7240;
if (x7301) {
x7261 += x7122;
} else {
}
if (x7305) {
x7262 += 1;
} else {
}

}
x7253 += x7241;
x7254 += x7123;

}
int32_t x7315 = 0;
int32_t x7316 = 1;
x7316 *= 1;
x7315 += 1;
x7316 *= 1;
x7316 *= 1;
int32_t x7321 = x7315;
bool x7322 = x7321 >= 2;
if (x7322) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7327 = x7321 == 0;
if (x7327) {
int32_t x7328 = x7316;
bool x7329 = x7328 == 128;
if (x7329) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7336 = x7316;
int32_t x7337 = 128 / x7336;
bool x7343;
if (x452) {
bool x7338 = x7237 == 1;
bool x7339 = x7337 == 1;
bool x7340 = x7338 || x7339;
bool x7341 = x7237 == x7337;
bool x7342 = x7340 || x7341;
x7343 = x7342;
} else {
x7343 = false;
}
bool x7347;
if (x7343) {
x7347 = x7346;
} else {
x7347 = false;
}
bool x7348;
if (x7347) {
x7348 = x7346;
} else {
x7348 = false;
}
if (x7348) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7237,x7239,x7239,1,x7337,1,1);
assert(false && "");
}
bool x7354 = x7237 <= x7337;
int32_t x7355;
if (x7354) {
x7355 = x7337;
} else {
x7355 = x7237;
}
bool x7361 = x7355 > 0;
bool x7363;
if (x7361) {
x7363 = x7362;
} else {
x7363 = false;
}
bool x7364;
if (x7363) {
x7364 = x7362;
} else {
x7364 = false;
}
if (x7364) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(7237) x Sym(7239) x Sym(7239)"," x Const(1) x Sym(7337) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x7359 = x7355 * x7358;
int32_t x7360 = 64 * x7359;
float* x7370 = (float*)myMalloc(x7360 * sizeof(float));;
int32_t x7371 = 0;
int32_t x7372 = 0;
int32_t x7373 = 0;
bool x7419 = x7237 > 1;
bool x7423 = x7337 > 1;
for(int x7374=0; x7374 < 64; x7374++) {
int32_t x7375 = x7372;
int32_t x7376 = x7373;
int32_t x7377 = x7371;
int32_t x7378 = x7377;
int32_t x7379 = x7375;
int32_t x7380 = x7376;
for(int x7382=0; x7382 < x7355; x7382++) {
int32_t x7383 = x7379;
int32_t x7384 = x7380;
int32_t x7385 = x7378;
int32_t x7386 = x7385;
int32_t x7387 = x7383;
int32_t x7388 = x7384;
for(int x7390=0; x7390 < x7357; x7390++) {
int32_t x7391 = x7387;
int32_t x7392 = x7388;
int32_t x7393 = x7386;
int32_t x7394 = x7393;
int32_t x7395 = x7391;
int32_t x7396 = x7392;
for(int x7397=0; x7397 < x7357; x7397++) {
int32_t x7398 = x7394;
int32_t x7399 = x7395;
float x7400 = x7252[x7399];
int32_t x7401 = x7396;
float x7402 = x172[x7401];
float x7403 = x7400 + x7402;
x7370[x7398] = x7403;
x7394 += 1;
if (x7406) {
x7395 += 1;
} else {
}

}
x7386 += x7357;
if (x7406) {
x7387 += x7239;
} else {
}

}
x7378 += x7358;
if (x7419) {
x7379 += x7240;
} else {
}
if (x7423) {
x7380 += 1;
} else {
}

}
x7371 += x7359;
x7372 += x7241;

}
float* x7433 = (float*)myMalloc(x7360 * sizeof(float));;
for(int x7435=0; x7435 < x7360; x7435++) {
float x7436 = x7370[x7435];
bool x7437 = x7436 < 0.0f;
if (x7437) {
x7433[x7435] = 0.0f;
} else {
float x7440 = x7370[x7435];
x7433[x7435] = x7440;
}

}
float* x7455 = (float*)myMalloc(x7454 * sizeof(float));;
int32_t x7456 = 9 * x7355;
int32_t x7459 = 64 * x7456;
int32_t x7460 = x7459 * x7450;
float* x7461 = (float*)myMalloc(x7460 * sizeof(float));;
int32_t x7457 = x7456 * x7450;
int32_t x7469 = x7355 * 3;
int32_t x7470 = x7469 * 3;
for(int x7462=0; x7462 < 64; x7462++) {
int32_t x7463 = x7462 * x7359;
float* x7464 = x7433+x7463;
int32_t x7465 = x7462 * x7451;
float* x7466 = x7455+x7465;
int32_t x7467 = x7462 * x7457;
float* x7468 = x7461+x7467;
for(int x7472=0; x7472 < x7470; x7472++) {
int32_t x7473 = x7472 / 9;
int32_t x7477 = x7473 * 3;
int32_t x7478 = x7477 * 3;
int32_t x7479 = x7478 * x7449;
int32_t x7480 = x7479 * x7449;
int32_t x7474 = x7472 % 9;
int32_t x7475 = x7474 / 3;
int32_t x7481 = x7475 * 3;
int32_t x7482 = x7481 * x7449;
int32_t x7483 = x7482 * x7449;
int32_t x7484 = x7480 + x7483;
int32_t x7476 = x7474 % 3;
int32_t x7485 = x7476 * x7449;
int32_t x7486 = x7485 * x7449;
int32_t x7487 = x7484 + x7486;
float* x7488 = x7468+x7487;
int32_t x7489 = x7473 * x7357;
int32_t x7490 = x7489 * x7357;
float* x7491 = x7464+x7490;
for(int x7493=0; x7493 < x7449; x7493++) {
int32_t x7494 = x7493 * 2;
int32_t x7495 = x7494 - 1;
int32_t x7496 = x7495 + x7475;
bool x7497 = x7496 < 0;
bool x7498 = x7496 >= x7357;
bool x7499 = x7497 || x7498;
if (x7499) {
int32_t x7500 = x7493 * x7449;
float* x7501 = x7488+x7500;
memset(x7501, 0, 4 * x7449);;
} else {
int32_t x7500 = x7493 * x7449;
int32_t x7516 = x7496 * x7357;
for(int x7504=0; x7504 < x7449; x7504++) {
int32_t x7505 = x7504 * 2;
int32_t x7506 = x7505 - 1;
int32_t x7507 = x7506 + x7476;
bool x7508 = x7507 < 0;
bool x7509 = x7507 >= x7357;
bool x7510 = x7508 || x7509;
if (x7510) {
int32_t x7511 = x7500 + x7504;
float* x7512 = x7488+x7511;
memset(x7512, 0, 4 * 1);;
} else {
int32_t x7511 = x7500 + x7504;
float* x7515 = x7488+x7511;
int32_t x7517 = x7516 + x7507;
float* x7518 = x7491+x7517;
memcpy(x7515, x7518, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x7450,x7456,1,x27,x7456,x7468,x7450,1,x7466,x7450);

}
int32_t x7533 = 0;
int32_t x7534 = 1;
x7534 *= 1;
x7533 += 1;
x7534 *= 1;
x7534 *= 1;
int32_t x7539 = x7533;
bool x7540 = x7539 >= 2;
if (x7540) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7545 = x7539 == 0;
if (x7545) {
int32_t x7546 = x7534;
bool x7547 = x7546 == 128;
if (x7547) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7554 = x7534;
int32_t x7555 = 128 / x7554;
bool x7559;
if (x452) {
bool x7556 = x7555 == 1;
bool x7557 = 128 == x7555;
bool x7558 = x7556 || x7557;
x7559 = x7558;
} else {
x7559 = false;
}
bool x7563;
if (x7559) {
x7563 = x7562;
} else {
x7563 = false;
}
bool x7564;
if (x7563) {
x7564 = x7562;
} else {
x7564 = false;
}
if (x7564) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x7449,x7449,1,x7555,1,1);
assert(false && "");
}
bool x7570 = 128 <= x7555;
int32_t x7571;
if (x7570) {
x7571 = x7555;
} else {
x7571 = 128;
}
bool x7577 = x7571 > 0;
bool x7579;
if (x7577) {
x7579 = x7578;
} else {
x7579 = false;
}
bool x7580;
if (x7579) {
x7580 = x7578;
} else {
x7580 = false;
}
if (x7580) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(128) x Sym(7449) x Sym(7449)"," x Const(1) x Sym(7555) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x7575 = x7571 * x7574;
int32_t x7576 = 64 * x7575;
float* x7586 = (float*)myMalloc(x7576 * sizeof(float));;
int32_t x7587 = 0;
int32_t x7588 = 0;
int32_t x7589 = 0;
bool x7636 = x7555 > 1;
for(int x7590=0; x7590 < 64; x7590++) {
int32_t x7591 = x7588;
int32_t x7592 = x7589;
int32_t x7593 = x7587;
int32_t x7594 = x7593;
int32_t x7595 = x7591;
int32_t x7596 = x7592;
for(int x7598=0; x7598 < x7571; x7598++) {
int32_t x7599 = x7595;
int32_t x7600 = x7596;
int32_t x7601 = x7594;
int32_t x7602 = x7601;
int32_t x7603 = x7599;
int32_t x7604 = x7600;
for(int x7606=0; x7606 < x7573; x7606++) {
int32_t x7607 = x7603;
int32_t x7608 = x7604;
int32_t x7609 = x7602;
int32_t x7610 = x7609;
int32_t x7611 = x7607;
int32_t x7612 = x7608;
for(int x7613=0; x7613 < x7573; x7613++) {
int32_t x7614 = x7610;
int32_t x7615 = x7611;
float x7616 = x7455[x7615];
int32_t x7617 = x7612;
float x7618 = x128[x7617];
float x7619 = x7616 - x7618;
x7586[x7614] = x7619;
x7610 += 1;
if (x7622) {
x7611 += 1;
} else {
}

}
x7602 += x7573;
if (x7622) {
x7603 += x7449;
} else {
}

}
x7594 += x7574;
x7595 += x7450;
if (x7636) {
x7596 += 1;
} else {
}

}
x7587 += x7575;
x7588 += x7451;

}
float* x7646 = (float*)myMalloc(128 * sizeof(float));;
for(int x7647=0; x7647 < 128; x7647++) {
float x7648 = x43[x7647];
float x7649 = x7648 + 1.0E-5f;
x7646[x7647] = x7649;

}
float* x7653 = (float*)myMalloc(128 * sizeof(float));;
for(int x7654=0; x7654 < 128; x7654++) {
float x7655 = x7646[x7654];
double x7656 = (double)x7655;
double x7657 = sqrt(x7656);
float x7658 = (float)x7657;
x7653[x7654] = x7658;

}
int32_t x7662 = 0;
int32_t x7663 = 1;
x7663 *= 1;
x7662 += 1;
x7663 *= 1;
x7663 *= 1;
int32_t x7668 = x7662;
bool x7669 = x7668 >= 2;
if (x7669) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7674 = x7668 == 0;
if (x7674) {
int32_t x7675 = x7663;
bool x7676 = x7675 == 128;
if (x7676) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7683 = x7663;
int32_t x7684 = 128 / x7683;
bool x7690;
if (x452) {
bool x7685 = x7571 == 1;
bool x7686 = x7684 == 1;
bool x7687 = x7685 || x7686;
bool x7688 = x7571 == x7684;
bool x7689 = x7687 || x7688;
x7690 = x7689;
} else {
x7690 = false;
}
bool x7694;
if (x7690) {
x7694 = x7693;
} else {
x7694 = false;
}
bool x7695;
if (x7694) {
x7695 = x7693;
} else {
x7695 = false;
}
if (x7695) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7571,x7573,x7573,1,x7684,1,1);
assert(false && "");
}
bool x7701 = x7571 <= x7684;
int32_t x7702;
if (x7701) {
x7702 = x7684;
} else {
x7702 = x7571;
}
bool x7708 = x7702 > 0;
bool x7710;
if (x7708) {
x7710 = x7709;
} else {
x7710 = false;
}
bool x7711;
if (x7710) {
x7711 = x7709;
} else {
x7711 = false;
}
if (x7711) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(7571) x Sym(7573) x Sym(7573)"," x Const(1) x Sym(7684) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x7706 = x7702 * x7705;
int32_t x7707 = 64 * x7706;
float* x7717 = (float*)myMalloc(x7707 * sizeof(float));;
int32_t x7718 = 0;
int32_t x7719 = 0;
int32_t x7720 = 0;
bool x7766 = x7571 > 1;
bool x7770 = x7684 > 1;
for(int x7721=0; x7721 < 64; x7721++) {
int32_t x7722 = x7719;
int32_t x7723 = x7720;
int32_t x7724 = x7718;
int32_t x7725 = x7724;
int32_t x7726 = x7722;
int32_t x7727 = x7723;
for(int x7729=0; x7729 < x7702; x7729++) {
int32_t x7730 = x7726;
int32_t x7731 = x7727;
int32_t x7732 = x7725;
int32_t x7733 = x7732;
int32_t x7734 = x7730;
int32_t x7735 = x7731;
for(int x7737=0; x7737 < x7704; x7737++) {
int32_t x7738 = x7734;
int32_t x7739 = x7735;
int32_t x7740 = x7733;
int32_t x7741 = x7740;
int32_t x7742 = x7738;
int32_t x7743 = x7739;
for(int x7744=0; x7744 < x7704; x7744++) {
int32_t x7745 = x7741;
int32_t x7746 = x7742;
float x7747 = x7586[x7746];
int32_t x7748 = x7743;
float x7749 = x7653[x7748];
float x7750 = x7747 / x7749;
x7717[x7745] = x7750;
x7741 += 1;
if (x7753) {
x7742 += 1;
} else {
}

}
x7733 += x7704;
if (x7753) {
x7734 += x7573;
} else {
}

}
x7725 += x7705;
if (x7766) {
x7726 += x7574;
} else {
}
if (x7770) {
x7727 += 1;
} else {
}

}
x7718 += x7706;
x7719 += x7575;

}
int32_t x7780 = 0;
int32_t x7781 = 1;
x7781 *= 1;
x7780 += 1;
x7781 *= 1;
x7781 *= 1;
int32_t x7786 = x7780;
bool x7787 = x7786 >= 2;
if (x7787) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7792 = x7786 == 0;
if (x7792) {
int32_t x7793 = x7781;
bool x7794 = x7793 == 128;
if (x7794) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7801 = x7781;
int32_t x7802 = 128 / x7801;
bool x7808;
if (x452) {
bool x7803 = x7702 == 1;
bool x7804 = x7802 == 1;
bool x7805 = x7803 || x7804;
bool x7806 = x7702 == x7802;
bool x7807 = x7805 || x7806;
x7808 = x7807;
} else {
x7808 = false;
}
bool x7812;
if (x7808) {
x7812 = x7811;
} else {
x7812 = false;
}
bool x7813;
if (x7812) {
x7813 = x7811;
} else {
x7813 = false;
}
if (x7813) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7702,x7704,x7704,1,x7802,1,1);
assert(false && "");
}
bool x7819 = x7702 <= x7802;
int32_t x7820;
if (x7819) {
x7820 = x7802;
} else {
x7820 = x7702;
}
bool x7826 = x7820 > 0;
bool x7828;
if (x7826) {
x7828 = x7827;
} else {
x7828 = false;
}
bool x7829;
if (x7828) {
x7829 = x7827;
} else {
x7829 = false;
}
if (x7829) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(7702) x Sym(7704) x Sym(7704)"," x Const(1) x Sym(7802) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x7824 = x7820 * x7823;
int32_t x7825 = 64 * x7824;
float* x7835 = (float*)myMalloc(x7825 * sizeof(float));;
int32_t x7836 = 0;
int32_t x7837 = 0;
int32_t x7838 = 0;
bool x7884 = x7702 > 1;
bool x7888 = x7802 > 1;
for(int x7839=0; x7839 < 64; x7839++) {
int32_t x7840 = x7837;
int32_t x7841 = x7838;
int32_t x7842 = x7836;
int32_t x7843 = x7842;
int32_t x7844 = x7840;
int32_t x7845 = x7841;
for(int x7847=0; x7847 < x7820; x7847++) {
int32_t x7848 = x7844;
int32_t x7849 = x7845;
int32_t x7850 = x7843;
int32_t x7851 = x7850;
int32_t x7852 = x7848;
int32_t x7853 = x7849;
for(int x7855=0; x7855 < x7822; x7855++) {
int32_t x7856 = x7852;
int32_t x7857 = x7853;
int32_t x7858 = x7851;
int32_t x7859 = x7858;
int32_t x7860 = x7856;
int32_t x7861 = x7857;
for(int x7862=0; x7862 < x7822; x7862++) {
int32_t x7863 = x7859;
int32_t x7864 = x7860;
float x7865 = x7717[x7864];
int32_t x7866 = x7861;
float x7867 = x252[x7866];
float x7868 = x7865 * x7867;
x7835[x7863] = x7868;
x7859 += 1;
if (x7871) {
x7860 += 1;
} else {
}

}
x7851 += x7822;
if (x7871) {
x7852 += x7704;
} else {
}

}
x7843 += x7823;
if (x7884) {
x7844 += x7705;
} else {
}
if (x7888) {
x7845 += 1;
} else {
}

}
x7836 += x7824;
x7837 += x7706;

}
int32_t x7898 = 0;
int32_t x7899 = 1;
x7899 *= 1;
x7898 += 1;
x7899 *= 1;
x7899 *= 1;
int32_t x7904 = x7898;
bool x7905 = x7904 >= 2;
if (x7905) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x7910 = x7904 == 0;
if (x7910) {
int32_t x7911 = x7899;
bool x7912 = x7911 == 128;
if (x7912) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x7919 = x7899;
int32_t x7920 = 128 / x7919;
bool x7926;
if (x452) {
bool x7921 = x7820 == 1;
bool x7922 = x7920 == 1;
bool x7923 = x7921 || x7922;
bool x7924 = x7820 == x7920;
bool x7925 = x7923 || x7924;
x7926 = x7925;
} else {
x7926 = false;
}
bool x7930;
if (x7926) {
x7930 = x7929;
} else {
x7930 = false;
}
bool x7931;
if (x7930) {
x7931 = x7929;
} else {
x7931 = false;
}
if (x7931) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x7820,x7822,x7822,1,x7920,1,1);
assert(false && "");
}
bool x7937 = x7820 <= x7920;
int32_t x7938;
if (x7937) {
x7938 = x7920;
} else {
x7938 = x7820;
}
bool x7944 = x7938 > 0;
bool x7946;
if (x7944) {
x7946 = x7945;
} else {
x7946 = false;
}
bool x7947;
if (x7946) {
x7947 = x7945;
} else {
x7947 = false;
}
if (x7947) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(7820) x Sym(7822) x Sym(7822)"," x Const(1) x Sym(7920) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x7942 = x7938 * x7941;
int32_t x7943 = 64 * x7942;
float* x7953 = (float*)myMalloc(x7943 * sizeof(float));;
int32_t x7954 = 0;
int32_t x7955 = 0;
int32_t x7956 = 0;
bool x8002 = x7820 > 1;
bool x8006 = x7920 > 1;
for(int x7957=0; x7957 < 64; x7957++) {
int32_t x7958 = x7955;
int32_t x7959 = x7956;
int32_t x7960 = x7954;
int32_t x7961 = x7960;
int32_t x7962 = x7958;
int32_t x7963 = x7959;
for(int x7965=0; x7965 < x7938; x7965++) {
int32_t x7966 = x7962;
int32_t x7967 = x7963;
int32_t x7968 = x7961;
int32_t x7969 = x7968;
int32_t x7970 = x7966;
int32_t x7971 = x7967;
for(int x7973=0; x7973 < x7940; x7973++) {
int32_t x7974 = x7970;
int32_t x7975 = x7971;
int32_t x7976 = x7969;
int32_t x7977 = x7976;
int32_t x7978 = x7974;
int32_t x7979 = x7975;
for(int x7980=0; x7980 < x7940; x7980++) {
int32_t x7981 = x7977;
int32_t x7982 = x7978;
float x7983 = x7835[x7982];
int32_t x7984 = x7979;
float x7985 = x190[x7984];
float x7986 = x7983 + x7985;
x7953[x7981] = x7986;
x7977 += 1;
if (x7989) {
x7978 += 1;
} else {
}

}
x7969 += x7940;
if (x7989) {
x7970 += x7822;
} else {
}

}
x7961 += x7941;
if (x8002) {
x7962 += x7823;
} else {
}
if (x8006) {
x7963 += 1;
} else {
}

}
x7954 += x7942;
x7955 += x7824;

}
float* x8016 = (float*)myMalloc(x7943 * sizeof(float));;
for(int x8018=0; x8018 < x7943; x8018++) {
float x8019 = x7953[x8018];
bool x8020 = x8019 < 0.0f;
if (x8020) {
x8016[x8018] = 0.0f;
} else {
float x8023 = x7953[x8018];
x8016[x8018] = x8023;
}

}
float* x8037 = (float*)myMalloc(x8036 * sizeof(float));;
int32_t x8040 = 64 * x7938;
int32_t x8041 = x8040 * x8032;
float* x8042 = (float*)myMalloc(x8041 * sizeof(float));;
int32_t x8038 = x7938 * x8032;
for(int x8043=0; x8043 < 64; x8043++) {
int32_t x8044 = x8043 * x7942;
float* x8045 = x8016+x8044;
int32_t x8046 = x8043 * x8033;
float* x8047 = x8037+x8046;
int32_t x8048 = x8043 * x8038;
float* x8049 = x8042+x8048;
for(int x8050=0; x8050 < x7938; x8050++) {
int32_t x8051 = x8050 / 1;
int32_t x8055 = x8051 * x8031;
int32_t x8056 = x8055 * x8031;
int32_t x8052 = x8050 % 1;
int32_t x8053 = x8052 / 1;
int32_t x8057 = x8053 * x8031;
int32_t x8058 = x8057 * x8031;
int32_t x8059 = x8056 + x8058;
int32_t x8054 = x8052 % 1;
int32_t x8060 = x8054 * x8031;
int32_t x8061 = x8060 * x8031;
int32_t x8062 = x8059 + x8061;
float* x8063 = x8049+x8062;
int32_t x8064 = x8051 * x7940;
int32_t x8065 = x8064 * x7940;
float* x8066 = x8045+x8065;
for(int x8068=0; x8068 < x8031; x8068++) {
int32_t x8070 = x8068 * x8031;
float* x8071 = x8063+x8070;
int32_t x8069 = x8068 + x8053;
int32_t x8072 = x8069 * x7940;
int32_t x8073 = x8072 + x8054;
float* x8074 = x8066+x8073;
memcpy(x8071, x8074, 4 * x8031);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x8032,x7938,1,x106,x7938,x8049,x8032,1,x8047,x8032);

}
int32_t x8083 = 0;
int32_t x8084 = 1;
x8084 *= 1;
x8083 += 1;
x8084 *= 1;
x8084 *= 1;
int32_t x8089 = x8083;
bool x8090 = x8089 >= 2;
if (x8090) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8095 = x8089 == 0;
if (x8095) {
int32_t x8096 = x8084;
bool x8097 = x8096 == 512;
if (x8097) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8104 = x8084;
int32_t x8105 = 512 / x8104;
bool x8109;
if (x452) {
bool x8106 = x8105 == 1;
bool x8107 = 512 == x8105;
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
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x8031,x8031,1,x8105,1,1);
assert(false && "");
}
bool x8120 = 512 <= x8105;
int32_t x8121;
if (x8120) {
x8121 = x8105;
} else {
x8121 = 512;
}
bool x8127 = x8121 > 0;
bool x8129;
if (x8127) {
x8129 = x8128;
} else {
x8129 = false;
}
bool x8130;
if (x8129) {
x8130 = x8128;
} else {
x8130 = false;
}
if (x8130) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(8031) x Sym(8031)"," x Const(1) x Sym(8105) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x8125 = x8121 * x8124;
int32_t x8126 = 64 * x8125;
float* x8136 = (float*)myMalloc(x8126 * sizeof(float));;
int32_t x8137 = 0;
int32_t x8138 = 0;
int32_t x8139 = 0;
bool x8186 = x8105 > 1;
for(int x8140=0; x8140 < 64; x8140++) {
int32_t x8141 = x8138;
int32_t x8142 = x8139;
int32_t x8143 = x8137;
int32_t x8144 = x8143;
int32_t x8145 = x8141;
int32_t x8146 = x8142;
for(int x8148=0; x8148 < x8121; x8148++) {
int32_t x8149 = x8145;
int32_t x8150 = x8146;
int32_t x8151 = x8144;
int32_t x8152 = x8151;
int32_t x8153 = x8149;
int32_t x8154 = x8150;
for(int x8156=0; x8156 < x8123; x8156++) {
int32_t x8157 = x8153;
int32_t x8158 = x8154;
int32_t x8159 = x8152;
int32_t x8160 = x8159;
int32_t x8161 = x8157;
int32_t x8162 = x8158;
for(int x8163=0; x8163 < x8123; x8163++) {
int32_t x8164 = x8160;
int32_t x8165 = x8161;
float x8166 = x8037[x8165];
int32_t x8167 = x8162;
float x8168 = x149[x8167];
float x8169 = x8166 - x8168;
x8136[x8164] = x8169;
x8160 += 1;
if (x8172) {
x8161 += 1;
} else {
}

}
x8152 += x8123;
if (x8172) {
x8153 += x8031;
} else {
}

}
x8144 += x8124;
x8145 += x8032;
if (x8186) {
x8146 += 1;
} else {
}

}
x8137 += x8125;
x8138 += x8033;

}
float* x8196 = (float*)myMalloc(512 * sizeof(float));;
for(int x8198=0; x8198 < 512; x8198++) {
float x8199 = x101[x8198];
float x8200 = x8199 + 1.0E-5f;
x8196[x8198] = x8200;

}
float* x8204 = (float*)myMalloc(512 * sizeof(float));;
for(int x8205=0; x8205 < 512; x8205++) {
float x8206 = x8196[x8205];
double x8207 = (double)x8206;
double x8208 = sqrt(x8207);
float x8209 = (float)x8208;
x8204[x8205] = x8209;

}
int32_t x8213 = 0;
int32_t x8214 = 1;
x8214 *= 1;
x8213 += 1;
x8214 *= 1;
x8214 *= 1;
int32_t x8219 = x8213;
bool x8220 = x8219 >= 2;
if (x8220) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8225 = x8219 == 0;
if (x8225) {
int32_t x8226 = x8214;
bool x8227 = x8226 == 512;
if (x8227) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8234 = x8214;
int32_t x8235 = 512 / x8234;
bool x8241;
if (x452) {
bool x8236 = x8121 == 1;
bool x8237 = x8235 == 1;
bool x8238 = x8236 || x8237;
bool x8239 = x8121 == x8235;
bool x8240 = x8238 || x8239;
x8241 = x8240;
} else {
x8241 = false;
}
bool x8245;
if (x8241) {
x8245 = x8244;
} else {
x8245 = false;
}
bool x8246;
if (x8245) {
x8246 = x8244;
} else {
x8246 = false;
}
if (x8246) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8121,x8123,x8123,1,x8235,1,1);
assert(false && "");
}
bool x8252 = x8121 <= x8235;
int32_t x8253;
if (x8252) {
x8253 = x8235;
} else {
x8253 = x8121;
}
bool x8259 = x8253 > 0;
bool x8261;
if (x8259) {
x8261 = x8260;
} else {
x8261 = false;
}
bool x8262;
if (x8261) {
x8262 = x8260;
} else {
x8262 = false;
}
if (x8262) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(8121) x Sym(8123) x Sym(8123)"," x Const(1) x Sym(8235) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x8257 = x8253 * x8256;
int32_t x8258 = 64 * x8257;
float* x8268 = (float*)myMalloc(x8258 * sizeof(float));;
int32_t x8269 = 0;
int32_t x8270 = 0;
int32_t x8271 = 0;
bool x8317 = x8121 > 1;
bool x8321 = x8235 > 1;
for(int x8272=0; x8272 < 64; x8272++) {
int32_t x8273 = x8270;
int32_t x8274 = x8271;
int32_t x8275 = x8269;
int32_t x8276 = x8275;
int32_t x8277 = x8273;
int32_t x8278 = x8274;
for(int x8280=0; x8280 < x8253; x8280++) {
int32_t x8281 = x8277;
int32_t x8282 = x8278;
int32_t x8283 = x8276;
int32_t x8284 = x8283;
int32_t x8285 = x8281;
int32_t x8286 = x8282;
for(int x8288=0; x8288 < x8255; x8288++) {
int32_t x8289 = x8285;
int32_t x8290 = x8286;
int32_t x8291 = x8284;
int32_t x8292 = x8291;
int32_t x8293 = x8289;
int32_t x8294 = x8290;
for(int x8295=0; x8295 < x8255; x8295++) {
int32_t x8296 = x8292;
int32_t x8297 = x8293;
float x8298 = x8136[x8297];
int32_t x8299 = x8294;
float x8300 = x8204[x8299];
float x8301 = x8298 / x8300;
x8268[x8296] = x8301;
x8292 += 1;
if (x8304) {
x8293 += 1;
} else {
}

}
x8284 += x8255;
if (x8304) {
x8285 += x8123;
} else {
}

}
x8276 += x8256;
if (x8317) {
x8277 += x8124;
} else {
}
if (x8321) {
x8278 += 1;
} else {
}

}
x8269 += x8257;
x8270 += x8125;

}
int32_t x8331 = 0;
int32_t x8332 = 1;
x8332 *= 1;
x8331 += 1;
x8332 *= 1;
x8332 *= 1;
int32_t x8337 = x8331;
bool x8338 = x8337 >= 2;
if (x8338) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8343 = x8337 == 0;
if (x8343) {
int32_t x8344 = x8332;
bool x8345 = x8344 == 512;
if (x8345) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8352 = x8332;
int32_t x8353 = 512 / x8352;
bool x8359;
if (x452) {
bool x8354 = x8253 == 1;
bool x8355 = x8353 == 1;
bool x8356 = x8354 || x8355;
bool x8357 = x8253 == x8353;
bool x8358 = x8356 || x8357;
x8359 = x8358;
} else {
x8359 = false;
}
bool x8363;
if (x8359) {
x8363 = x8362;
} else {
x8363 = false;
}
bool x8364;
if (x8363) {
x8364 = x8362;
} else {
x8364 = false;
}
if (x8364) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8253,x8255,x8255,1,x8353,1,1);
assert(false && "");
}
bool x8370 = x8253 <= x8353;
int32_t x8371;
if (x8370) {
x8371 = x8353;
} else {
x8371 = x8253;
}
bool x8377 = x8371 > 0;
bool x8379;
if (x8377) {
x8379 = x8378;
} else {
x8379 = false;
}
bool x8380;
if (x8379) {
x8380 = x8378;
} else {
x8380 = false;
}
if (x8380) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(8253) x Sym(8255) x Sym(8255)"," x Const(1) x Sym(8353) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x8375 = x8371 * x8374;
int32_t x8376 = 64 * x8375;
float* x8386 = (float*)myMalloc(x8376 * sizeof(float));;
int32_t x8387 = 0;
int32_t x8388 = 0;
int32_t x8389 = 0;
bool x8435 = x8253 > 1;
bool x8439 = x8353 > 1;
for(int x8390=0; x8390 < 64; x8390++) {
int32_t x8391 = x8388;
int32_t x8392 = x8389;
int32_t x8393 = x8387;
int32_t x8394 = x8393;
int32_t x8395 = x8391;
int32_t x8396 = x8392;
for(int x8398=0; x8398 < x8371; x8398++) {
int32_t x8399 = x8395;
int32_t x8400 = x8396;
int32_t x8401 = x8394;
int32_t x8402 = x8401;
int32_t x8403 = x8399;
int32_t x8404 = x8400;
for(int x8406=0; x8406 < x8373; x8406++) {
int32_t x8407 = x8403;
int32_t x8408 = x8404;
int32_t x8409 = x8402;
int32_t x8410 = x8409;
int32_t x8411 = x8407;
int32_t x8412 = x8408;
for(int x8413=0; x8413 < x8373; x8413++) {
int32_t x8414 = x8410;
int32_t x8415 = x8411;
float x8416 = x8268[x8415];
int32_t x8417 = x8412;
float x8418 = x145[x8417];
float x8419 = x8416 * x8418;
x8386[x8414] = x8419;
x8410 += 1;
if (x8422) {
x8411 += 1;
} else {
}

}
x8402 += x8373;
if (x8422) {
x8403 += x8255;
} else {
}

}
x8394 += x8374;
if (x8435) {
x8395 += x8256;
} else {
}
if (x8439) {
x8396 += 1;
} else {
}

}
x8387 += x8375;
x8388 += x8257;

}
int32_t x8449 = 0;
int32_t x8450 = 1;
x8450 *= 1;
x8449 += 1;
x8450 *= 1;
x8450 *= 1;
int32_t x8455 = x8449;
bool x8456 = x8455 >= 2;
if (x8456) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8461 = x8455 == 0;
if (x8461) {
int32_t x8462 = x8450;
bool x8463 = x8462 == 512;
if (x8463) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8470 = x8450;
int32_t x8471 = 512 / x8470;
bool x8477;
if (x452) {
bool x8472 = x8371 == 1;
bool x8473 = x8471 == 1;
bool x8474 = x8472 || x8473;
bool x8475 = x8371 == x8471;
bool x8476 = x8474 || x8475;
x8477 = x8476;
} else {
x8477 = false;
}
bool x8481;
if (x8477) {
x8481 = x8480;
} else {
x8481 = false;
}
bool x8482;
if (x8481) {
x8482 = x8480;
} else {
x8482 = false;
}
if (x8482) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8371,x8373,x8373,1,x8471,1,1);
assert(false && "");
}
bool x8488 = x8371 <= x8471;
int32_t x8489;
if (x8488) {
x8489 = x8471;
} else {
x8489 = x8371;
}
bool x8495 = x8489 > 0;
bool x8497;
if (x8495) {
x8497 = x8496;
} else {
x8497 = false;
}
bool x8498;
if (x8497) {
x8498 = x8496;
} else {
x8498 = false;
}
if (x8498) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(8371) x Sym(8373) x Sym(8373)"," x Const(1) x Sym(8471) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x8493 = x8489 * x8492;
int32_t x8494 = 64 * x8493;
float* x8504 = (float*)myMalloc(x8494 * sizeof(float));;
int32_t x8505 = 0;
int32_t x8506 = 0;
int32_t x8507 = 0;
bool x8553 = x8371 > 1;
bool x8557 = x8471 > 1;
for(int x8508=0; x8508 < 64; x8508++) {
int32_t x8509 = x8506;
int32_t x8510 = x8507;
int32_t x8511 = x8505;
int32_t x8512 = x8511;
int32_t x8513 = x8509;
int32_t x8514 = x8510;
for(int x8516=0; x8516 < x8489; x8516++) {
int32_t x8517 = x8513;
int32_t x8518 = x8514;
int32_t x8519 = x8512;
int32_t x8520 = x8519;
int32_t x8521 = x8517;
int32_t x8522 = x8518;
for(int x8524=0; x8524 < x8491; x8524++) {
int32_t x8525 = x8521;
int32_t x8526 = x8522;
int32_t x8527 = x8520;
int32_t x8528 = x8527;
int32_t x8529 = x8525;
int32_t x8530 = x8526;
for(int x8531=0; x8531 < x8491; x8531++) {
int32_t x8532 = x8528;
int32_t x8533 = x8529;
float x8534 = x8386[x8533];
int32_t x8535 = x8530;
float x8536 = x210[x8535];
float x8537 = x8534 + x8536;
x8504[x8532] = x8537;
x8528 += 1;
if (x8540) {
x8529 += 1;
} else {
}

}
x8520 += x8491;
if (x8540) {
x8521 += x8373;
} else {
}

}
x8512 += x8492;
if (x8553) {
x8513 += x8374;
} else {
}
if (x8557) {
x8514 += 1;
} else {
}

}
x8505 += x8493;
x8506 += x8375;

}
float* x8574 = (float*)myMalloc(x8573 * sizeof(float));;
int32_t x8577 = x6906 * x8569;
float* x8578 = (float*)myMalloc(x8577 * sizeof(float));;
int32_t x8575 = x6715 * x8569;
for(int x8579=0; x8579 < 64; x8579++) {
int32_t x8580 = x8579 * x6719;
float* x8581 = x6882+x8580;
int32_t x8582 = x8579 * x8570;
float* x8583 = x8574+x8582;
int32_t x8584 = x8579 * x8575;
float* x8585 = x8578+x8584;
for(int x8586=0; x8586 < x6715; x8586++) {
int32_t x8587 = x8586 / 1;
int32_t x8591 = x8587 * x8568;
int32_t x8592 = x8591 * x8568;
int32_t x8588 = x8586 % 1;
int32_t x8589 = x8588 / 1;
int32_t x8593 = x8589 * x8568;
int32_t x8594 = x8593 * x8568;
int32_t x8595 = x8592 + x8594;
int32_t x8590 = x8588 % 1;
int32_t x8596 = x8590 * x8568;
int32_t x8597 = x8596 * x8568;
int32_t x8598 = x8595 + x8597;
float* x8599 = x8585+x8598;
int32_t x8600 = x8587 * x6717;
int32_t x8601 = x8600 * x6717;
float* x8602 = x8581+x8601;
for(int x8604=0; x8604 < x8568; x8604++) {
int32_t x8608 = x8604 * x8568;
int32_t x8605 = x8604 * 2;
int32_t x8606 = x8605 + x8589;
int32_t x8611 = x8606 * x6717;
int32_t x8612 = x8611 + x8590;
for(int x8607=0; x8607 < x8568; x8607++) {
int32_t x8609 = x8608 + x8607;
float* x8610 = x8599+x8609;
int32_t x8613 = x8607 * 2;
int32_t x8614 = x8612 + x8613;
float* x8615 = x8602+x8614;
memcpy(x8610, x8615, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x8569,x6715,1,x258,x6715,x8585,x8569,1,x8583,x8569);

}
int32_t x8626 = 0;
int32_t x8627 = 1;
x8627 *= 1;
x8626 += 1;
x8627 *= 1;
x8627 *= 1;
int32_t x8632 = x8626;
bool x8633 = x8632 >= 2;
if (x8633) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8638 = x8632 == 0;
if (x8638) {
int32_t x8639 = x8627;
bool x8640 = x8639 == 512;
if (x8640) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8647 = x8627;
int32_t x8648 = 512 / x8647;
bool x8652;
if (x452) {
bool x8649 = x8648 == 1;
bool x8650 = 512 == x8648;
bool x8651 = x8649 || x8650;
x8652 = x8651;
} else {
x8652 = false;
}
bool x8656;
if (x8652) {
x8656 = x8655;
} else {
x8656 = false;
}
bool x8657;
if (x8656) {
x8657 = x8655;
} else {
x8657 = false;
}
if (x8657) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x8568,x8568,1,x8648,1,1);
assert(false && "");
}
bool x8663 = 512 <= x8648;
int32_t x8664;
if (x8663) {
x8664 = x8648;
} else {
x8664 = 512;
}
bool x8670 = x8664 > 0;
bool x8672;
if (x8670) {
x8672 = x8671;
} else {
x8672 = false;
}
bool x8673;
if (x8672) {
x8673 = x8671;
} else {
x8673 = false;
}
if (x8673) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(8568) x Sym(8568)"," x Const(1) x Sym(8648) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x8668 = x8664 * x8667;
int32_t x8669 = 64 * x8668;
float* x8679 = (float*)myMalloc(x8669 * sizeof(float));;
int32_t x8680 = 0;
int32_t x8681 = 0;
int32_t x8682 = 0;
bool x8729 = x8648 > 1;
for(int x8683=0; x8683 < 64; x8683++) {
int32_t x8684 = x8681;
int32_t x8685 = x8682;
int32_t x8686 = x8680;
int32_t x8687 = x8686;
int32_t x8688 = x8684;
int32_t x8689 = x8685;
for(int x8691=0; x8691 < x8664; x8691++) {
int32_t x8692 = x8688;
int32_t x8693 = x8689;
int32_t x8694 = x8687;
int32_t x8695 = x8694;
int32_t x8696 = x8692;
int32_t x8697 = x8693;
for(int x8699=0; x8699 < x8666; x8699++) {
int32_t x8700 = x8696;
int32_t x8701 = x8697;
int32_t x8702 = x8695;
int32_t x8703 = x8702;
int32_t x8704 = x8700;
int32_t x8705 = x8701;
for(int x8706=0; x8706 < x8666; x8706++) {
int32_t x8707 = x8703;
int32_t x8708 = x8704;
float x8709 = x8574[x8708];
int32_t x8710 = x8705;
float x8711 = x42[x8710];
float x8712 = x8709 - x8711;
x8679[x8707] = x8712;
x8703 += 1;
if (x8715) {
x8704 += 1;
} else {
}

}
x8695 += x8666;
if (x8715) {
x8696 += x8568;
} else {
}

}
x8687 += x8667;
x8688 += x8569;
if (x8729) {
x8689 += 1;
} else {
}

}
x8680 += x8668;
x8681 += x8570;

}
float* x8739 = (float*)myMalloc(512 * sizeof(float));;
for(int x8740=0; x8740 < 512; x8740++) {
float x8741 = x23[x8740];
float x8742 = x8741 + 1.0E-5f;
x8739[x8740] = x8742;

}
float* x8746 = (float*)myMalloc(512 * sizeof(float));;
for(int x8747=0; x8747 < 512; x8747++) {
float x8748 = x8739[x8747];
double x8749 = (double)x8748;
double x8750 = sqrt(x8749);
float x8751 = (float)x8750;
x8746[x8747] = x8751;

}
int32_t x8755 = 0;
int32_t x8756 = 1;
x8756 *= 1;
x8755 += 1;
x8756 *= 1;
x8756 *= 1;
int32_t x8761 = x8755;
bool x8762 = x8761 >= 2;
if (x8762) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8767 = x8761 == 0;
if (x8767) {
int32_t x8768 = x8756;
bool x8769 = x8768 == 512;
if (x8769) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8776 = x8756;
int32_t x8777 = 512 / x8776;
bool x8783;
if (x452) {
bool x8778 = x8664 == 1;
bool x8779 = x8777 == 1;
bool x8780 = x8778 || x8779;
bool x8781 = x8664 == x8777;
bool x8782 = x8780 || x8781;
x8783 = x8782;
} else {
x8783 = false;
}
bool x8787;
if (x8783) {
x8787 = x8786;
} else {
x8787 = false;
}
bool x8788;
if (x8787) {
x8788 = x8786;
} else {
x8788 = false;
}
if (x8788) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8664,x8666,x8666,1,x8777,1,1);
assert(false && "");
}
bool x8794 = x8664 <= x8777;
int32_t x8795;
if (x8794) {
x8795 = x8777;
} else {
x8795 = x8664;
}
bool x8801 = x8795 > 0;
bool x8803;
if (x8801) {
x8803 = x8802;
} else {
x8803 = false;
}
bool x8804;
if (x8803) {
x8804 = x8802;
} else {
x8804 = false;
}
if (x8804) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(8664) x Sym(8666) x Sym(8666)"," x Const(1) x Sym(8777) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x8799 = x8795 * x8798;
int32_t x8800 = 64 * x8799;
float* x8810 = (float*)myMalloc(x8800 * sizeof(float));;
int32_t x8811 = 0;
int32_t x8812 = 0;
int32_t x8813 = 0;
bool x8859 = x8664 > 1;
bool x8863 = x8777 > 1;
for(int x8814=0; x8814 < 64; x8814++) {
int32_t x8815 = x8812;
int32_t x8816 = x8813;
int32_t x8817 = x8811;
int32_t x8818 = x8817;
int32_t x8819 = x8815;
int32_t x8820 = x8816;
for(int x8822=0; x8822 < x8795; x8822++) {
int32_t x8823 = x8819;
int32_t x8824 = x8820;
int32_t x8825 = x8818;
int32_t x8826 = x8825;
int32_t x8827 = x8823;
int32_t x8828 = x8824;
for(int x8830=0; x8830 < x8797; x8830++) {
int32_t x8831 = x8827;
int32_t x8832 = x8828;
int32_t x8833 = x8826;
int32_t x8834 = x8833;
int32_t x8835 = x8831;
int32_t x8836 = x8832;
for(int x8837=0; x8837 < x8797; x8837++) {
int32_t x8838 = x8834;
int32_t x8839 = x8835;
float x8840 = x8679[x8839];
int32_t x8841 = x8836;
float x8842 = x8746[x8841];
float x8843 = x8840 / x8842;
x8810[x8838] = x8843;
x8834 += 1;
if (x8846) {
x8835 += 1;
} else {
}

}
x8826 += x8797;
if (x8846) {
x8827 += x8666;
} else {
}

}
x8818 += x8798;
if (x8859) {
x8819 += x8667;
} else {
}
if (x8863) {
x8820 += 1;
} else {
}

}
x8811 += x8799;
x8812 += x8668;

}
int32_t x8873 = 0;
int32_t x8874 = 1;
x8874 *= 1;
x8873 += 1;
x8874 *= 1;
x8874 *= 1;
int32_t x8879 = x8873;
bool x8880 = x8879 >= 2;
if (x8880) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x8885 = x8879 == 0;
if (x8885) {
int32_t x8886 = x8874;
bool x8887 = x8886 == 512;
if (x8887) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x8894 = x8874;
int32_t x8895 = 512 / x8894;
bool x8901;
if (x452) {
bool x8896 = x8795 == 1;
bool x8897 = x8895 == 1;
bool x8898 = x8896 || x8897;
bool x8899 = x8795 == x8895;
bool x8900 = x8898 || x8899;
x8901 = x8900;
} else {
x8901 = false;
}
bool x8905;
if (x8901) {
x8905 = x8904;
} else {
x8905 = false;
}
bool x8906;
if (x8905) {
x8906 = x8904;
} else {
x8906 = false;
}
if (x8906) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8795,x8797,x8797,1,x8895,1,1);
assert(false && "");
}
bool x8912 = x8795 <= x8895;
int32_t x8913;
if (x8912) {
x8913 = x8895;
} else {
x8913 = x8795;
}
bool x8919 = x8913 > 0;
bool x8921;
if (x8919) {
x8921 = x8920;
} else {
x8921 = false;
}
bool x8922;
if (x8921) {
x8922 = x8920;
} else {
x8922 = false;
}
if (x8922) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(8795) x Sym(8797) x Sym(8797)"," x Const(1) x Sym(8895) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x8917 = x8913 * x8916;
int32_t x8918 = 64 * x8917;
float* x8928 = (float*)myMalloc(x8918 * sizeof(float));;
int32_t x8929 = 0;
int32_t x8930 = 0;
int32_t x8931 = 0;
bool x8977 = x8795 > 1;
bool x8981 = x8895 > 1;
for(int x8932=0; x8932 < 64; x8932++) {
int32_t x8933 = x8930;
int32_t x8934 = x8931;
int32_t x8935 = x8929;
int32_t x8936 = x8935;
int32_t x8937 = x8933;
int32_t x8938 = x8934;
for(int x8940=0; x8940 < x8913; x8940++) {
int32_t x8941 = x8937;
int32_t x8942 = x8938;
int32_t x8943 = x8936;
int32_t x8944 = x8943;
int32_t x8945 = x8941;
int32_t x8946 = x8942;
for(int x8948=0; x8948 < x8915; x8948++) {
int32_t x8949 = x8945;
int32_t x8950 = x8946;
int32_t x8951 = x8944;
int32_t x8952 = x8951;
int32_t x8953 = x8949;
int32_t x8954 = x8950;
for(int x8955=0; x8955 < x8915; x8955++) {
int32_t x8956 = x8952;
int32_t x8957 = x8953;
float x8958 = x8810[x8957];
int32_t x8959 = x8954;
float x8960 = x207[x8959];
float x8961 = x8958 * x8960;
x8928[x8956] = x8961;
x8952 += 1;
if (x8964) {
x8953 += 1;
} else {
}

}
x8944 += x8915;
if (x8964) {
x8945 += x8797;
} else {
}

}
x8936 += x8916;
if (x8977) {
x8937 += x8798;
} else {
}
if (x8981) {
x8938 += 1;
} else {
}

}
x8929 += x8917;
x8930 += x8799;

}
int32_t x8991 = 0;
int32_t x8992 = 1;
x8992 *= 1;
x8991 += 1;
x8992 *= 1;
x8992 *= 1;
int32_t x8997 = x8991;
bool x8998 = x8997 >= 2;
if (x8998) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9003 = x8997 == 0;
if (x9003) {
int32_t x9004 = x8992;
bool x9005 = x9004 == 512;
if (x9005) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9012 = x8992;
int32_t x9013 = 512 / x9012;
bool x9019;
if (x452) {
bool x9014 = x8913 == 1;
bool x9015 = x9013 == 1;
bool x9016 = x9014 || x9015;
bool x9017 = x8913 == x9013;
bool x9018 = x9016 || x9017;
x9019 = x9018;
} else {
x9019 = false;
}
bool x9023;
if (x9019) {
x9023 = x9022;
} else {
x9023 = false;
}
bool x9024;
if (x9023) {
x9024 = x9022;
} else {
x9024 = false;
}
if (x9024) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8913,x8915,x8915,1,x9013,1,1);
assert(false && "");
}
bool x9030 = x8913 <= x9013;
int32_t x9031;
if (x9030) {
x9031 = x9013;
} else {
x9031 = x8913;
}
bool x9037 = x9031 > 0;
bool x9039;
if (x9037) {
x9039 = x9038;
} else {
x9039 = false;
}
bool x9040;
if (x9039) {
x9040 = x9038;
} else {
x9040 = false;
}
if (x9040) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(8913) x Sym(8915) x Sym(8915)"," x Const(1) x Sym(9013) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x9035 = x9031 * x9034;
int32_t x9036 = 64 * x9035;
float* x9046 = (float*)myMalloc(x9036 * sizeof(float));;
int32_t x9047 = 0;
int32_t x9048 = 0;
int32_t x9049 = 0;
bool x9095 = x8913 > 1;
bool x9099 = x9013 > 1;
for(int x9050=0; x9050 < 64; x9050++) {
int32_t x9051 = x9048;
int32_t x9052 = x9049;
int32_t x9053 = x9047;
int32_t x9054 = x9053;
int32_t x9055 = x9051;
int32_t x9056 = x9052;
for(int x9058=0; x9058 < x9031; x9058++) {
int32_t x9059 = x9055;
int32_t x9060 = x9056;
int32_t x9061 = x9054;
int32_t x9062 = x9061;
int32_t x9063 = x9059;
int32_t x9064 = x9060;
for(int x9066=0; x9066 < x9033; x9066++) {
int32_t x9067 = x9063;
int32_t x9068 = x9064;
int32_t x9069 = x9062;
int32_t x9070 = x9069;
int32_t x9071 = x9067;
int32_t x9072 = x9068;
for(int x9073=0; x9073 < x9033; x9073++) {
int32_t x9074 = x9070;
int32_t x9075 = x9071;
float x9076 = x8928[x9075];
int32_t x9077 = x9072;
float x9078 = x119[x9077];
float x9079 = x9076 + x9078;
x9046[x9074] = x9079;
x9070 += 1;
if (x9082) {
x9071 += 1;
} else {
}

}
x9062 += x9033;
if (x9082) {
x9063 += x8915;
} else {
}

}
x9054 += x9034;
if (x9095) {
x9055 += x8916;
} else {
}
if (x9099) {
x9056 += 1;
} else {
}

}
x9047 += x9035;
x9048 += x8917;

}
bool x9109 = x8489 == 1;
bool x9110 = x9031 == 1;
bool x9111 = x9109 || x9110;
bool x9112 = x8489 == x9031;
bool x9113 = x9111 || x9112;
bool x9119;
if (x9113) {
x9119 = x9118;
} else {
x9119 = false;
}
bool x9120;
if (x9119) {
x9120 = x9118;
} else {
x9120 = false;
}
if (x9120) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x8489,x8491,x8491,64,x9031,x9033,x9033);
assert(false && "");
}
int32_t x9133 = 0;
int32_t x9134 = 0;
int32_t x9135 = 0;
bool x9126 = x8489 <= x9031;
int32_t x9127;
if (x9126) {
x9127 = x9031;
} else {
x9127 = x8489;
}
bool x9187 = x8489 > 1;
bool x9191 = x9031 > 1;
int32_t x9131 = x9127 * x9130;
for(int x9136=0; x9136 < 64; x9136++) {
int32_t x9137 = x9134;
int32_t x9138 = x9135;
int32_t x9139 = x9133;
int32_t x9140 = x9139;
int32_t x9141 = x9137;
int32_t x9142 = x9138;
for(int x9144=0; x9144 < x9127; x9144++) {
int32_t x9145 = x9141;
int32_t x9146 = x9142;
int32_t x9147 = x9140;
int32_t x9148 = x9147;
int32_t x9149 = x9145;
int32_t x9150 = x9146;
for(int x9152=0; x9152 < x9129; x9152++) {
int32_t x9153 = x9149;
int32_t x9154 = x9150;
int32_t x9155 = x9148;
int32_t x9156 = x9155;
int32_t x9157 = x9153;
int32_t x9158 = x9154;
for(int x9159=0; x9159 < x9129; x9159++) {
int32_t x9160 = x9157;
float x9161 = x8504[x9160];
int32_t x9162 = x9158;
float x9163 = x9046[x9162];
float x9164 = x9161 + x9163;
x8504[x9160] = x9164;
x9156 += 1;
if (x9167) {
x9157 += 1;
} else {
}
if (x9171) {
x9158 += 1;
} else {
}

}
x9148 += x9129;
if (x9167) {
x9149 += x8491;
} else {
}
if (x9171) {
x9150 += x9033;
} else {
}

}
x9140 += x9130;
if (x9187) {
x9141 += x8492;
} else {
}
if (x9191) {
x9142 += x9034;
} else {
}

}
x9133 += x9131;
x9134 += x8493;
x9135 += x9035;

}
float* x9202 = (float*)myMalloc(x8494 * sizeof(float));;
for(int x9204=0; x9204 < x8494; x9204++) {
float x9205 = x8504[x9204];
bool x9206 = x9205 < 0.0f;
if (x9206) {
x9202[x9204] = 0.0f;
} else {
float x9209 = x8504[x9204];
x9202[x9204] = x9209;
}

}
float* x9223 = (float*)myMalloc(x9222 * sizeof(float));;
int32_t x9226 = 64 * x8489;
int32_t x9227 = x9226 * x9218;
float* x9228 = (float*)myMalloc(x9227 * sizeof(float));;
int32_t x9224 = x8489 * x9218;
for(int x9229=0; x9229 < 64; x9229++) {
int32_t x9230 = x9229 * x8493;
float* x9231 = x9202+x9230;
int32_t x9232 = x9229 * x9219;
float* x9233 = x9223+x9232;
int32_t x9234 = x9229 * x9224;
float* x9235 = x9228+x9234;
for(int x9236=0; x9236 < x8489; x9236++) {
int32_t x9237 = x9236 / 1;
int32_t x9241 = x9237 * x9217;
int32_t x9242 = x9241 * x9217;
int32_t x9238 = x9236 % 1;
int32_t x9239 = x9238 / 1;
int32_t x9243 = x9239 * x9217;
int32_t x9244 = x9243 * x9217;
int32_t x9245 = x9242 + x9244;
int32_t x9240 = x9238 % 1;
int32_t x9246 = x9240 * x9217;
int32_t x9247 = x9246 * x9217;
int32_t x9248 = x9245 + x9247;
float* x9249 = x9235+x9248;
int32_t x9250 = x9237 * x8491;
int32_t x9251 = x9250 * x8491;
float* x9252 = x9231+x9251;
for(int x9254=0; x9254 < x9217; x9254++) {
int32_t x9256 = x9254 * x9217;
float* x9257 = x9249+x9256;
int32_t x9255 = x9254 + x9239;
int32_t x9258 = x9255 * x8491;
int32_t x9259 = x9258 + x9240;
float* x9260 = x9252+x9259;
memcpy(x9257, x9260, 4 * x9217);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x9218,x8489,1,x256,x8489,x9235,x9218,1,x9233,x9218);

}
int32_t x9269 = 0;
int32_t x9270 = 1;
x9270 *= 1;
x9269 += 1;
x9270 *= 1;
x9270 *= 1;
int32_t x9275 = x9269;
bool x9276 = x9275 >= 2;
if (x9276) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9281 = x9275 == 0;
if (x9281) {
int32_t x9282 = x9270;
bool x9283 = x9282 == 128;
if (x9283) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9290 = x9270;
int32_t x9291 = 128 / x9290;
bool x9295;
if (x452) {
bool x9292 = x9291 == 1;
bool x9293 = 128 == x9291;
bool x9294 = x9292 || x9293;
x9295 = x9294;
} else {
x9295 = false;
}
bool x9299;
if (x9295) {
x9299 = x9298;
} else {
x9299 = false;
}
bool x9300;
if (x9299) {
x9300 = x9298;
} else {
x9300 = false;
}
if (x9300) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x9217,x9217,1,x9291,1,1);
assert(false && "");
}
bool x9306 = 128 <= x9291;
int32_t x9307;
if (x9306) {
x9307 = x9291;
} else {
x9307 = 128;
}
bool x9313 = x9307 > 0;
bool x9315;
if (x9313) {
x9315 = x9314;
} else {
x9315 = false;
}
bool x9316;
if (x9315) {
x9316 = x9314;
} else {
x9316 = false;
}
if (x9316) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(128) x Sym(9217) x Sym(9217)"," x Const(1) x Sym(9291) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x9311 = x9307 * x9310;
int32_t x9312 = 64 * x9311;
float* x9322 = (float*)myMalloc(x9312 * sizeof(float));;
int32_t x9323 = 0;
int32_t x9324 = 0;
int32_t x9325 = 0;
bool x9372 = x9291 > 1;
for(int x9326=0; x9326 < 64; x9326++) {
int32_t x9327 = x9324;
int32_t x9328 = x9325;
int32_t x9329 = x9323;
int32_t x9330 = x9329;
int32_t x9331 = x9327;
int32_t x9332 = x9328;
for(int x9334=0; x9334 < x9307; x9334++) {
int32_t x9335 = x9331;
int32_t x9336 = x9332;
int32_t x9337 = x9330;
int32_t x9338 = x9337;
int32_t x9339 = x9335;
int32_t x9340 = x9336;
for(int x9342=0; x9342 < x9309; x9342++) {
int32_t x9343 = x9339;
int32_t x9344 = x9340;
int32_t x9345 = x9338;
int32_t x9346 = x9345;
int32_t x9347 = x9343;
int32_t x9348 = x9344;
for(int x9349=0; x9349 < x9309; x9349++) {
int32_t x9350 = x9346;
int32_t x9351 = x9347;
float x9352 = x9223[x9351];
int32_t x9353 = x9348;
float x9354 = x100[x9353];
float x9355 = x9352 - x9354;
x9322[x9350] = x9355;
x9346 += 1;
if (x9358) {
x9347 += 1;
} else {
}

}
x9338 += x9309;
if (x9358) {
x9339 += x9217;
} else {
}

}
x9330 += x9310;
x9331 += x9218;
if (x9372) {
x9332 += 1;
} else {
}

}
x9323 += x9311;
x9324 += x9219;

}
float* x9382 = (float*)myMalloc(128 * sizeof(float));;
for(int x9383=0; x9383 < 128; x9383++) {
float x9384 = x177[x9383];
float x9385 = x9384 + 1.0E-5f;
x9382[x9383] = x9385;

}
float* x9389 = (float*)myMalloc(128 * sizeof(float));;
for(int x9390=0; x9390 < 128; x9390++) {
float x9391 = x9382[x9390];
double x9392 = (double)x9391;
double x9393 = sqrt(x9392);
float x9394 = (float)x9393;
x9389[x9390] = x9394;

}
int32_t x9398 = 0;
int32_t x9399 = 1;
x9399 *= 1;
x9398 += 1;
x9399 *= 1;
x9399 *= 1;
int32_t x9404 = x9398;
bool x9405 = x9404 >= 2;
if (x9405) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9410 = x9404 == 0;
if (x9410) {
int32_t x9411 = x9399;
bool x9412 = x9411 == 128;
if (x9412) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9419 = x9399;
int32_t x9420 = 128 / x9419;
bool x9426;
if (x452) {
bool x9421 = x9307 == 1;
bool x9422 = x9420 == 1;
bool x9423 = x9421 || x9422;
bool x9424 = x9307 == x9420;
bool x9425 = x9423 || x9424;
x9426 = x9425;
} else {
x9426 = false;
}
bool x9430;
if (x9426) {
x9430 = x9429;
} else {
x9430 = false;
}
bool x9431;
if (x9430) {
x9431 = x9429;
} else {
x9431 = false;
}
if (x9431) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9307,x9309,x9309,1,x9420,1,1);
assert(false && "");
}
bool x9437 = x9307 <= x9420;
int32_t x9438;
if (x9437) {
x9438 = x9420;
} else {
x9438 = x9307;
}
bool x9444 = x9438 > 0;
bool x9446;
if (x9444) {
x9446 = x9445;
} else {
x9446 = false;
}
bool x9447;
if (x9446) {
x9447 = x9445;
} else {
x9447 = false;
}
if (x9447) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(9307) x Sym(9309) x Sym(9309)"," x Const(1) x Sym(9420) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x9442 = x9438 * x9441;
int32_t x9443 = 64 * x9442;
float* x9453 = (float*)myMalloc(x9443 * sizeof(float));;
int32_t x9454 = 0;
int32_t x9455 = 0;
int32_t x9456 = 0;
bool x9502 = x9307 > 1;
bool x9506 = x9420 > 1;
for(int x9457=0; x9457 < 64; x9457++) {
int32_t x9458 = x9455;
int32_t x9459 = x9456;
int32_t x9460 = x9454;
int32_t x9461 = x9460;
int32_t x9462 = x9458;
int32_t x9463 = x9459;
for(int x9465=0; x9465 < x9438; x9465++) {
int32_t x9466 = x9462;
int32_t x9467 = x9463;
int32_t x9468 = x9461;
int32_t x9469 = x9468;
int32_t x9470 = x9466;
int32_t x9471 = x9467;
for(int x9473=0; x9473 < x9440; x9473++) {
int32_t x9474 = x9470;
int32_t x9475 = x9471;
int32_t x9476 = x9469;
int32_t x9477 = x9476;
int32_t x9478 = x9474;
int32_t x9479 = x9475;
for(int x9480=0; x9480 < x9440; x9480++) {
int32_t x9481 = x9477;
int32_t x9482 = x9478;
float x9483 = x9322[x9482];
int32_t x9484 = x9479;
float x9485 = x9389[x9484];
float x9486 = x9483 / x9485;
x9453[x9481] = x9486;
x9477 += 1;
if (x9489) {
x9478 += 1;
} else {
}

}
x9469 += x9440;
if (x9489) {
x9470 += x9309;
} else {
}

}
x9461 += x9441;
if (x9502) {
x9462 += x9310;
} else {
}
if (x9506) {
x9463 += 1;
} else {
}

}
x9454 += x9442;
x9455 += x9311;

}
int32_t x9516 = 0;
int32_t x9517 = 1;
x9517 *= 1;
x9516 += 1;
x9517 *= 1;
x9517 *= 1;
int32_t x9522 = x9516;
bool x9523 = x9522 >= 2;
if (x9523) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9528 = x9522 == 0;
if (x9528) {
int32_t x9529 = x9517;
bool x9530 = x9529 == 128;
if (x9530) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9537 = x9517;
int32_t x9538 = 128 / x9537;
bool x9544;
if (x452) {
bool x9539 = x9438 == 1;
bool x9540 = x9538 == 1;
bool x9541 = x9539 || x9540;
bool x9542 = x9438 == x9538;
bool x9543 = x9541 || x9542;
x9544 = x9543;
} else {
x9544 = false;
}
bool x9548;
if (x9544) {
x9548 = x9547;
} else {
x9548 = false;
}
bool x9549;
if (x9548) {
x9549 = x9547;
} else {
x9549 = false;
}
if (x9549) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9438,x9440,x9440,1,x9538,1,1);
assert(false && "");
}
bool x9555 = x9438 <= x9538;
int32_t x9556;
if (x9555) {
x9556 = x9538;
} else {
x9556 = x9438;
}
bool x9562 = x9556 > 0;
bool x9564;
if (x9562) {
x9564 = x9563;
} else {
x9564 = false;
}
bool x9565;
if (x9564) {
x9565 = x9563;
} else {
x9565 = false;
}
if (x9565) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(9438) x Sym(9440) x Sym(9440)"," x Const(1) x Sym(9538) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x9560 = x9556 * x9559;
int32_t x9561 = 64 * x9560;
float* x9571 = (float*)myMalloc(x9561 * sizeof(float));;
int32_t x9572 = 0;
int32_t x9573 = 0;
int32_t x9574 = 0;
bool x9620 = x9438 > 1;
bool x9624 = x9538 > 1;
for(int x9575=0; x9575 < 64; x9575++) {
int32_t x9576 = x9573;
int32_t x9577 = x9574;
int32_t x9578 = x9572;
int32_t x9579 = x9578;
int32_t x9580 = x9576;
int32_t x9581 = x9577;
for(int x9583=0; x9583 < x9556; x9583++) {
int32_t x9584 = x9580;
int32_t x9585 = x9581;
int32_t x9586 = x9579;
int32_t x9587 = x9586;
int32_t x9588 = x9584;
int32_t x9589 = x9585;
for(int x9591=0; x9591 < x9558; x9591++) {
int32_t x9592 = x9588;
int32_t x9593 = x9589;
int32_t x9594 = x9587;
int32_t x9595 = x9594;
int32_t x9596 = x9592;
int32_t x9597 = x9593;
for(int x9598=0; x9598 < x9558; x9598++) {
int32_t x9599 = x9595;
int32_t x9600 = x9596;
float x9601 = x9453[x9600];
int32_t x9602 = x9597;
float x9603 = x222[x9602];
float x9604 = x9601 * x9603;
x9571[x9599] = x9604;
x9595 += 1;
if (x9607) {
x9596 += 1;
} else {
}

}
x9587 += x9558;
if (x9607) {
x9588 += x9440;
} else {
}

}
x9579 += x9559;
if (x9620) {
x9580 += x9441;
} else {
}
if (x9624) {
x9581 += 1;
} else {
}

}
x9572 += x9560;
x9573 += x9442;

}
int32_t x9634 = 0;
int32_t x9635 = 1;
x9635 *= 1;
x9634 += 1;
x9635 *= 1;
x9635 *= 1;
int32_t x9640 = x9634;
bool x9641 = x9640 >= 2;
if (x9641) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9646 = x9640 == 0;
if (x9646) {
int32_t x9647 = x9635;
bool x9648 = x9647 == 128;
if (x9648) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9655 = x9635;
int32_t x9656 = 128 / x9655;
bool x9662;
if (x452) {
bool x9657 = x9556 == 1;
bool x9658 = x9656 == 1;
bool x9659 = x9657 || x9658;
bool x9660 = x9556 == x9656;
bool x9661 = x9659 || x9660;
x9662 = x9661;
} else {
x9662 = false;
}
bool x9666;
if (x9662) {
x9666 = x9665;
} else {
x9666 = false;
}
bool x9667;
if (x9666) {
x9667 = x9665;
} else {
x9667 = false;
}
if (x9667) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9556,x9558,x9558,1,x9656,1,1);
assert(false && "");
}
bool x9673 = x9556 <= x9656;
int32_t x9674;
if (x9673) {
x9674 = x9656;
} else {
x9674 = x9556;
}
bool x9680 = x9674 > 0;
bool x9682;
if (x9680) {
x9682 = x9681;
} else {
x9682 = false;
}
bool x9683;
if (x9682) {
x9683 = x9681;
} else {
x9683 = false;
}
if (x9683) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(9556) x Sym(9558) x Sym(9558)"," x Const(1) x Sym(9656) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x9678 = x9674 * x9677;
int32_t x9679 = 64 * x9678;
float* x9689 = (float*)myMalloc(x9679 * sizeof(float));;
int32_t x9690 = 0;
int32_t x9691 = 0;
int32_t x9692 = 0;
bool x9738 = x9556 > 1;
bool x9742 = x9656 > 1;
for(int x9693=0; x9693 < 64; x9693++) {
int32_t x9694 = x9691;
int32_t x9695 = x9692;
int32_t x9696 = x9690;
int32_t x9697 = x9696;
int32_t x9698 = x9694;
int32_t x9699 = x9695;
for(int x9701=0; x9701 < x9674; x9701++) {
int32_t x9702 = x9698;
int32_t x9703 = x9699;
int32_t x9704 = x9697;
int32_t x9705 = x9704;
int32_t x9706 = x9702;
int32_t x9707 = x9703;
for(int x9709=0; x9709 < x9676; x9709++) {
int32_t x9710 = x9706;
int32_t x9711 = x9707;
int32_t x9712 = x9705;
int32_t x9713 = x9712;
int32_t x9714 = x9710;
int32_t x9715 = x9711;
for(int x9716=0; x9716 < x9676; x9716++) {
int32_t x9717 = x9713;
int32_t x9718 = x9714;
float x9719 = x9571[x9718];
int32_t x9720 = x9715;
float x9721 = x17[x9720];
float x9722 = x9719 + x9721;
x9689[x9717] = x9722;
x9713 += 1;
if (x9725) {
x9714 += 1;
} else {
}

}
x9705 += x9676;
if (x9725) {
x9706 += x9558;
} else {
}

}
x9697 += x9677;
if (x9738) {
x9698 += x9559;
} else {
}
if (x9742) {
x9699 += 1;
} else {
}

}
x9690 += x9678;
x9691 += x9560;

}
float* x9752 = (float*)myMalloc(x9679 * sizeof(float));;
for(int x9754=0; x9754 < x9679; x9754++) {
float x9755 = x9689[x9754];
bool x9756 = x9755 < 0.0f;
if (x9756) {
x9752[x9754] = 0.0f;
} else {
float x9759 = x9689[x9754];
x9752[x9754] = x9759;
}

}
float* x9774 = (float*)myMalloc(x9773 * sizeof(float));;
int32_t x9775 = 9 * x9674;
int32_t x9778 = 64 * x9775;
int32_t x9779 = x9778 * x9769;
float* x9780 = (float*)myMalloc(x9779 * sizeof(float));;
int32_t x9776 = x9775 * x9769;
int32_t x9788 = x9674 * 3;
int32_t x9789 = x9788 * 3;
for(int x9781=0; x9781 < 64; x9781++) {
int32_t x9782 = x9781 * x9678;
float* x9783 = x9752+x9782;
int32_t x9784 = x9781 * x9770;
float* x9785 = x9774+x9784;
int32_t x9786 = x9781 * x9776;
float* x9787 = x9780+x9786;
for(int x9791=0; x9791 < x9789; x9791++) {
int32_t x9792 = x9791 / 9;
int32_t x9796 = x9792 * 3;
int32_t x9797 = x9796 * 3;
int32_t x9798 = x9797 * x9768;
int32_t x9799 = x9798 * x9768;
int32_t x9793 = x9791 % 9;
int32_t x9794 = x9793 / 3;
int32_t x9800 = x9794 * 3;
int32_t x9801 = x9800 * x9768;
int32_t x9802 = x9801 * x9768;
int32_t x9803 = x9799 + x9802;
int32_t x9795 = x9793 % 3;
int32_t x9804 = x9795 * x9768;
int32_t x9805 = x9804 * x9768;
int32_t x9806 = x9803 + x9805;
float* x9807 = x9787+x9806;
int32_t x9808 = x9792 * x9676;
int32_t x9809 = x9808 * x9676;
float* x9810 = x9783+x9809;
int32_t x9823 = 1 - x9795;
bool x9824 = x9823 > 0;
int32_t x9825;
if (x9824) {
x9825 = x9823;
} else {
x9825 = 0;
}
int32_t x9826 = 3 - x9795;
int32_t x9827 = x9826 - 1;
int32_t x9828 = 1 - x9827;
bool x9829 = x9828 > 0;
int32_t x9830;
if (x9829) {
x9830 = x9828;
} else {
x9830 = 0;
}
int32_t x9831 = x9768 - x9830;
int32_t x9832 = x9831 - x9825;
bool x9833 = x9832 <= 0;
bool x9837 = x9825 > 0;
int32_t x9822 = -1 + x9795;
bool x9850 = x9830 > 0;
for(int x9812=0; x9812 < x9768; x9812++) {
int32_t x9813 = x9812 - 1;
int32_t x9814 = x9813 + x9794;
bool x9815 = x9814 < 0;
bool x9816 = x9814 >= x9676;
bool x9817 = x9815 || x9816;
if (x9817) {
int32_t x9818 = x9812 * x9768;
float* x9819 = x9807+x9818;
memset(x9819, 0, 4 * x9768);;
} else {
if (x9833) {
int32_t x9818 = x9812 * x9768;
float* x9834 = x9807+x9818;
memset(x9834, 0, 4 * x9768);;
} else {
int32_t x9818 = x9812 * x9768;
if (x9837) {
float* x9838 = x9807+x9818;
memset(x9838, 0, 4 * x9825);;
} else {
}
// may have segfault here
int32_t x9843 = x9818 + x9825;
float* x9844 = x9807+x9843;
int32_t x9845 = x9814 * x9676;
int32_t x9846 = x9845 + x9822;
int32_t x9847 = x9846 + x9825;
float* x9848 = x9810+x9847;
memcpy(x9844, x9848, 4 * x9832);;
if (x9850) {
int32_t x9851 = x9818 + x9768;
int32_t x9852 = x9851 - x9830;
float* x9853 = x9807+x9852;
memset(x9853, 0, 4 * x9830);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x9769,x9775,1,x235,x9775,x9787,x9769,1,x9785,x9769);

}
int32_t x9868 = 0;
int32_t x9869 = 1;
x9869 *= 1;
x9868 += 1;
x9869 *= 1;
x9869 *= 1;
int32_t x9874 = x9868;
bool x9875 = x9874 >= 2;
if (x9875) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x9880 = x9874 == 0;
if (x9880) {
int32_t x9881 = x9869;
bool x9882 = x9881 == 128;
if (x9882) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x9889 = x9869;
int32_t x9890 = 128 / x9889;
bool x9894;
if (x452) {
bool x9891 = x9890 == 1;
bool x9892 = 128 == x9890;
bool x9893 = x9891 || x9892;
x9894 = x9893;
} else {
x9894 = false;
}
bool x9898;
if (x9894) {
x9898 = x9897;
} else {
x9898 = false;
}
bool x9899;
if (x9898) {
x9899 = x9897;
} else {
x9899 = false;
}
if (x9899) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x9768,x9768,1,x9890,1,1);
assert(false && "");
}
bool x9905 = 128 <= x9890;
int32_t x9906;
if (x9905) {
x9906 = x9890;
} else {
x9906 = 128;
}
bool x9912 = x9906 > 0;
bool x9914;
if (x9912) {
x9914 = x9913;
} else {
x9914 = false;
}
bool x9915;
if (x9914) {
x9915 = x9913;
} else {
x9915 = false;
}
if (x9915) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(128) x Sym(9768) x Sym(9768)"," x Const(1) x Sym(9890) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x9910 = x9906 * x9909;
int32_t x9911 = 64 * x9910;
float* x9921 = (float*)myMalloc(x9911 * sizeof(float));;
int32_t x9922 = 0;
int32_t x9923 = 0;
int32_t x9924 = 0;
bool x9971 = x9890 > 1;
for(int x9925=0; x9925 < 64; x9925++) {
int32_t x9926 = x9923;
int32_t x9927 = x9924;
int32_t x9928 = x9922;
int32_t x9929 = x9928;
int32_t x9930 = x9926;
int32_t x9931 = x9927;
for(int x9933=0; x9933 < x9906; x9933++) {
int32_t x9934 = x9930;
int32_t x9935 = x9931;
int32_t x9936 = x9929;
int32_t x9937 = x9936;
int32_t x9938 = x9934;
int32_t x9939 = x9935;
for(int x9941=0; x9941 < x9908; x9941++) {
int32_t x9942 = x9938;
int32_t x9943 = x9939;
int32_t x9944 = x9937;
int32_t x9945 = x9944;
int32_t x9946 = x9942;
int32_t x9947 = x9943;
for(int x9948=0; x9948 < x9908; x9948++) {
int32_t x9949 = x9945;
int32_t x9950 = x9946;
float x9951 = x9774[x9950];
int32_t x9952 = x9947;
float x9953 = x35[x9952];
float x9954 = x9951 - x9953;
x9921[x9949] = x9954;
x9945 += 1;
if (x9957) {
x9946 += 1;
} else {
}

}
x9937 += x9908;
if (x9957) {
x9938 += x9768;
} else {
}

}
x9929 += x9909;
x9930 += x9769;
if (x9971) {
x9931 += 1;
} else {
}

}
x9922 += x9910;
x9923 += x9770;

}
float* x9981 = (float*)myMalloc(128 * sizeof(float));;
for(int x9982=0; x9982 < 128; x9982++) {
float x9983 = x225[x9982];
float x9984 = x9983 + 1.0E-5f;
x9981[x9982] = x9984;

}
float* x9988 = (float*)myMalloc(128 * sizeof(float));;
for(int x9989=0; x9989 < 128; x9989++) {
float x9990 = x9981[x9989];
double x9991 = (double)x9990;
double x9992 = sqrt(x9991);
float x9993 = (float)x9992;
x9988[x9989] = x9993;

}
int32_t x9997 = 0;
int32_t x9998 = 1;
x9998 *= 1;
x9997 += 1;
x9998 *= 1;
x9998 *= 1;
int32_t x10003 = x9997;
bool x10004 = x10003 >= 2;
if (x10004) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10009 = x10003 == 0;
if (x10009) {
int32_t x10010 = x9998;
bool x10011 = x10010 == 128;
if (x10011) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10018 = x9998;
int32_t x10019 = 128 / x10018;
bool x10025;
if (x452) {
bool x10020 = x9906 == 1;
bool x10021 = x10019 == 1;
bool x10022 = x10020 || x10021;
bool x10023 = x9906 == x10019;
bool x10024 = x10022 || x10023;
x10025 = x10024;
} else {
x10025 = false;
}
bool x10029;
if (x10025) {
x10029 = x10028;
} else {
x10029 = false;
}
bool x10030;
if (x10029) {
x10030 = x10028;
} else {
x10030 = false;
}
if (x10030) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x9906,x9908,x9908,1,x10019,1,1);
assert(false && "");
}
bool x10036 = x9906 <= x10019;
int32_t x10037;
if (x10036) {
x10037 = x10019;
} else {
x10037 = x9906;
}
bool x10043 = x10037 > 0;
bool x10045;
if (x10043) {
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
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(9906) x Sym(9908) x Sym(9908)"," x Const(1) x Sym(10019) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x10041 = x10037 * x10040;
int32_t x10042 = 64 * x10041;
float* x10052 = (float*)myMalloc(x10042 * sizeof(float));;
int32_t x10053 = 0;
int32_t x10054 = 0;
int32_t x10055 = 0;
bool x10101 = x9906 > 1;
bool x10105 = x10019 > 1;
for(int x10056=0; x10056 < 64; x10056++) {
int32_t x10057 = x10054;
int32_t x10058 = x10055;
int32_t x10059 = x10053;
int32_t x10060 = x10059;
int32_t x10061 = x10057;
int32_t x10062 = x10058;
for(int x10064=0; x10064 < x10037; x10064++) {
int32_t x10065 = x10061;
int32_t x10066 = x10062;
int32_t x10067 = x10060;
int32_t x10068 = x10067;
int32_t x10069 = x10065;
int32_t x10070 = x10066;
for(int x10072=0; x10072 < x10039; x10072++) {
int32_t x10073 = x10069;
int32_t x10074 = x10070;
int32_t x10075 = x10068;
int32_t x10076 = x10075;
int32_t x10077 = x10073;
int32_t x10078 = x10074;
for(int x10079=0; x10079 < x10039; x10079++) {
int32_t x10080 = x10076;
int32_t x10081 = x10077;
float x10082 = x9921[x10081];
int32_t x10083 = x10078;
float x10084 = x9988[x10083];
float x10085 = x10082 / x10084;
x10052[x10080] = x10085;
x10076 += 1;
if (x10088) {
x10077 += 1;
} else {
}

}
x10068 += x10039;
if (x10088) {
x10069 += x9908;
} else {
}

}
x10060 += x10040;
if (x10101) {
x10061 += x9909;
} else {
}
if (x10105) {
x10062 += 1;
} else {
}

}
x10053 += x10041;
x10054 += x9910;

}
int32_t x10115 = 0;
int32_t x10116 = 1;
x10116 *= 1;
x10115 += 1;
x10116 *= 1;
x10116 *= 1;
int32_t x10121 = x10115;
bool x10122 = x10121 >= 2;
if (x10122) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10127 = x10121 == 0;
if (x10127) {
int32_t x10128 = x10116;
bool x10129 = x10128 == 128;
if (x10129) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10136 = x10116;
int32_t x10137 = 128 / x10136;
bool x10143;
if (x452) {
bool x10138 = x10037 == 1;
bool x10139 = x10137 == 1;
bool x10140 = x10138 || x10139;
bool x10141 = x10037 == x10137;
bool x10142 = x10140 || x10141;
x10143 = x10142;
} else {
x10143 = false;
}
bool x10147;
if (x10143) {
x10147 = x10146;
} else {
x10147 = false;
}
bool x10148;
if (x10147) {
x10148 = x10146;
} else {
x10148 = false;
}
if (x10148) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10037,x10039,x10039,1,x10137,1,1);
assert(false && "");
}
bool x10154 = x10037 <= x10137;
int32_t x10155;
if (x10154) {
x10155 = x10137;
} else {
x10155 = x10037;
}
bool x10161 = x10155 > 0;
bool x10163;
if (x10161) {
x10163 = x10162;
} else {
x10163 = false;
}
bool x10164;
if (x10163) {
x10164 = x10162;
} else {
x10164 = false;
}
if (x10164) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(10037) x Sym(10039) x Sym(10039)"," x Const(1) x Sym(10137) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x10159 = x10155 * x10158;
int32_t x10160 = 64 * x10159;
float* x10170 = (float*)myMalloc(x10160 * sizeof(float));;
int32_t x10171 = 0;
int32_t x10172 = 0;
int32_t x10173 = 0;
bool x10219 = x10037 > 1;
bool x10223 = x10137 > 1;
for(int x10174=0; x10174 < 64; x10174++) {
int32_t x10175 = x10172;
int32_t x10176 = x10173;
int32_t x10177 = x10171;
int32_t x10178 = x10177;
int32_t x10179 = x10175;
int32_t x10180 = x10176;
for(int x10182=0; x10182 < x10155; x10182++) {
int32_t x10183 = x10179;
int32_t x10184 = x10180;
int32_t x10185 = x10178;
int32_t x10186 = x10185;
int32_t x10187 = x10183;
int32_t x10188 = x10184;
for(int x10190=0; x10190 < x10157; x10190++) {
int32_t x10191 = x10187;
int32_t x10192 = x10188;
int32_t x10193 = x10186;
int32_t x10194 = x10193;
int32_t x10195 = x10191;
int32_t x10196 = x10192;
for(int x10197=0; x10197 < x10157; x10197++) {
int32_t x10198 = x10194;
int32_t x10199 = x10195;
float x10200 = x10052[x10199];
int32_t x10201 = x10196;
float x10202 = x8[x10201];
float x10203 = x10200 * x10202;
x10170[x10198] = x10203;
x10194 += 1;
if (x10206) {
x10195 += 1;
} else {
}

}
x10186 += x10157;
if (x10206) {
x10187 += x10039;
} else {
}

}
x10178 += x10158;
if (x10219) {
x10179 += x10040;
} else {
}
if (x10223) {
x10180 += 1;
} else {
}

}
x10171 += x10159;
x10172 += x10041;

}
int32_t x10233 = 0;
int32_t x10234 = 1;
x10234 *= 1;
x10233 += 1;
x10234 *= 1;
x10234 *= 1;
int32_t x10239 = x10233;
bool x10240 = x10239 >= 2;
if (x10240) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10245 = x10239 == 0;
if (x10245) {
int32_t x10246 = x10234;
bool x10247 = x10246 == 128;
if (x10247) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10254 = x10234;
int32_t x10255 = 128 / x10254;
bool x10261;
if (x452) {
bool x10256 = x10155 == 1;
bool x10257 = x10255 == 1;
bool x10258 = x10256 || x10257;
bool x10259 = x10155 == x10255;
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
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10155,x10157,x10157,1,x10255,1,1);
assert(false && "");
}
bool x10272 = x10155 <= x10255;
int32_t x10273;
if (x10272) {
x10273 = x10255;
} else {
x10273 = x10155;
}
bool x10279 = x10273 > 0;
bool x10281;
if (x10279) {
x10281 = x10280;
} else {
x10281 = false;
}
bool x10282;
if (x10281) {
x10282 = x10280;
} else {
x10282 = false;
}
if (x10282) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(10155) x Sym(10157) x Sym(10157)"," x Const(1) x Sym(10255) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x10277 = x10273 * x10276;
int32_t x10278 = 64 * x10277;
float* x10288 = (float*)myMalloc(x10278 * sizeof(float));;
int32_t x10289 = 0;
int32_t x10290 = 0;
int32_t x10291 = 0;
bool x10337 = x10155 > 1;
bool x10341 = x10255 > 1;
for(int x10292=0; x10292 < 64; x10292++) {
int32_t x10293 = x10290;
int32_t x10294 = x10291;
int32_t x10295 = x10289;
int32_t x10296 = x10295;
int32_t x10297 = x10293;
int32_t x10298 = x10294;
for(int x10300=0; x10300 < x10273; x10300++) {
int32_t x10301 = x10297;
int32_t x10302 = x10298;
int32_t x10303 = x10296;
int32_t x10304 = x10303;
int32_t x10305 = x10301;
int32_t x10306 = x10302;
for(int x10308=0; x10308 < x10275; x10308++) {
int32_t x10309 = x10305;
int32_t x10310 = x10306;
int32_t x10311 = x10304;
int32_t x10312 = x10311;
int32_t x10313 = x10309;
int32_t x10314 = x10310;
for(int x10315=0; x10315 < x10275; x10315++) {
int32_t x10316 = x10312;
int32_t x10317 = x10313;
float x10318 = x10170[x10317];
int32_t x10319 = x10314;
float x10320 = x95[x10319];
float x10321 = x10318 + x10320;
x10288[x10316] = x10321;
x10312 += 1;
if (x10324) {
x10313 += 1;
} else {
}

}
x10304 += x10275;
if (x10324) {
x10305 += x10157;
} else {
}

}
x10296 += x10276;
if (x10337) {
x10297 += x10158;
} else {
}
if (x10341) {
x10298 += 1;
} else {
}

}
x10289 += x10277;
x10290 += x10159;

}
float* x10351 = (float*)myMalloc(x10278 * sizeof(float));;
for(int x10353=0; x10353 < x10278; x10353++) {
float x10354 = x10288[x10353];
bool x10355 = x10354 < 0.0f;
if (x10355) {
x10351[x10353] = 0.0f;
} else {
float x10358 = x10288[x10353];
x10351[x10353] = x10358;
}

}
float* x10372 = (float*)myMalloc(x10371 * sizeof(float));;
int32_t x10375 = 64 * x10273;
int32_t x10376 = x10375 * x10367;
float* x10377 = (float*)myMalloc(x10376 * sizeof(float));;
int32_t x10373 = x10273 * x10367;
for(int x10378=0; x10378 < 64; x10378++) {
int32_t x10379 = x10378 * x10277;
float* x10380 = x10351+x10379;
int32_t x10381 = x10378 * x10368;
float* x10382 = x10372+x10381;
int32_t x10383 = x10378 * x10373;
float* x10384 = x10377+x10383;
for(int x10385=0; x10385 < x10273; x10385++) {
int32_t x10386 = x10385 / 1;
int32_t x10390 = x10386 * x10366;
int32_t x10391 = x10390 * x10366;
int32_t x10387 = x10385 % 1;
int32_t x10388 = x10387 / 1;
int32_t x10392 = x10388 * x10366;
int32_t x10393 = x10392 * x10366;
int32_t x10394 = x10391 + x10393;
int32_t x10389 = x10387 % 1;
int32_t x10395 = x10389 * x10366;
int32_t x10396 = x10395 * x10366;
int32_t x10397 = x10394 + x10396;
float* x10398 = x10384+x10397;
int32_t x10399 = x10386 * x10275;
int32_t x10400 = x10399 * x10275;
float* x10401 = x10380+x10400;
for(int x10403=0; x10403 < x10366; x10403++) {
int32_t x10405 = x10403 * x10366;
float* x10406 = x10398+x10405;
int32_t x10404 = x10403 + x10388;
int32_t x10407 = x10404 * x10275;
int32_t x10408 = x10407 + x10389;
float* x10409 = x10401+x10408;
memcpy(x10406, x10409, 4 * x10366);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x10367,x10273,1,x111,x10273,x10384,x10367,1,x10382,x10367);

}
int32_t x10418 = 0;
int32_t x10419 = 1;
x10419 *= 1;
x10418 += 1;
x10419 *= 1;
x10419 *= 1;
int32_t x10424 = x10418;
bool x10425 = x10424 >= 2;
if (x10425) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10430 = x10424 == 0;
if (x10430) {
int32_t x10431 = x10419;
bool x10432 = x10431 == 512;
if (x10432) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10439 = x10419;
int32_t x10440 = 512 / x10439;
bool x10444;
if (x452) {
bool x10441 = x10440 == 1;
bool x10442 = 512 == x10440;
bool x10443 = x10441 || x10442;
x10444 = x10443;
} else {
x10444 = false;
}
bool x10448;
if (x10444) {
x10448 = x10447;
} else {
x10448 = false;
}
bool x10449;
if (x10448) {
x10449 = x10447;
} else {
x10449 = false;
}
if (x10449) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x10366,x10366,1,x10440,1,1);
assert(false && "");
}
bool x10455 = 512 <= x10440;
int32_t x10456;
if (x10455) {
x10456 = x10440;
} else {
x10456 = 512;
}
bool x10462 = x10456 > 0;
bool x10464;
if (x10462) {
x10464 = x10463;
} else {
x10464 = false;
}
bool x10465;
if (x10464) {
x10465 = x10463;
} else {
x10465 = false;
}
if (x10465) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(10366) x Sym(10366)"," x Const(1) x Sym(10440) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x10460 = x10456 * x10459;
int32_t x10461 = 64 * x10460;
float* x10471 = (float*)myMalloc(x10461 * sizeof(float));;
int32_t x10472 = 0;
int32_t x10473 = 0;
int32_t x10474 = 0;
bool x10521 = x10440 > 1;
for(int x10475=0; x10475 < 64; x10475++) {
int32_t x10476 = x10473;
int32_t x10477 = x10474;
int32_t x10478 = x10472;
int32_t x10479 = x10478;
int32_t x10480 = x10476;
int32_t x10481 = x10477;
for(int x10483=0; x10483 < x10456; x10483++) {
int32_t x10484 = x10480;
int32_t x10485 = x10481;
int32_t x10486 = x10479;
int32_t x10487 = x10486;
int32_t x10488 = x10484;
int32_t x10489 = x10485;
for(int x10491=0; x10491 < x10458; x10491++) {
int32_t x10492 = x10488;
int32_t x10493 = x10489;
int32_t x10494 = x10487;
int32_t x10495 = x10494;
int32_t x10496 = x10492;
int32_t x10497 = x10493;
for(int x10498=0; x10498 < x10458; x10498++) {
int32_t x10499 = x10495;
int32_t x10500 = x10496;
float x10501 = x10372[x10500];
int32_t x10502 = x10497;
float x10503 = x147[x10502];
float x10504 = x10501 - x10503;
x10471[x10499] = x10504;
x10495 += 1;
if (x10507) {
x10496 += 1;
} else {
}

}
x10487 += x10458;
if (x10507) {
x10488 += x10366;
} else {
}

}
x10479 += x10459;
x10480 += x10367;
if (x10521) {
x10481 += 1;
} else {
}

}
x10472 += x10460;
x10473 += x10368;

}
float* x10531 = (float*)myMalloc(512 * sizeof(float));;
for(int x10532=0; x10532 < 512; x10532++) {
float x10533 = x88[x10532];
float x10534 = x10533 + 1.0E-5f;
x10531[x10532] = x10534;

}
float* x10538 = (float*)myMalloc(512 * sizeof(float));;
for(int x10539=0; x10539 < 512; x10539++) {
float x10540 = x10531[x10539];
double x10541 = (double)x10540;
double x10542 = sqrt(x10541);
float x10543 = (float)x10542;
x10538[x10539] = x10543;

}
int32_t x10547 = 0;
int32_t x10548 = 1;
x10548 *= 1;
x10547 += 1;
x10548 *= 1;
x10548 *= 1;
int32_t x10553 = x10547;
bool x10554 = x10553 >= 2;
if (x10554) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10559 = x10553 == 0;
if (x10559) {
int32_t x10560 = x10548;
bool x10561 = x10560 == 512;
if (x10561) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10568 = x10548;
int32_t x10569 = 512 / x10568;
bool x10575;
if (x452) {
bool x10570 = x10456 == 1;
bool x10571 = x10569 == 1;
bool x10572 = x10570 || x10571;
bool x10573 = x10456 == x10569;
bool x10574 = x10572 || x10573;
x10575 = x10574;
} else {
x10575 = false;
}
bool x10579;
if (x10575) {
x10579 = x10578;
} else {
x10579 = false;
}
bool x10580;
if (x10579) {
x10580 = x10578;
} else {
x10580 = false;
}
if (x10580) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10456,x10458,x10458,1,x10569,1,1);
assert(false && "");
}
bool x10586 = x10456 <= x10569;
int32_t x10587;
if (x10586) {
x10587 = x10569;
} else {
x10587 = x10456;
}
bool x10593 = x10587 > 0;
bool x10595;
if (x10593) {
x10595 = x10594;
} else {
x10595 = false;
}
bool x10596;
if (x10595) {
x10596 = x10594;
} else {
x10596 = false;
}
if (x10596) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(10456) x Sym(10458) x Sym(10458)"," x Const(1) x Sym(10569) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x10591 = x10587 * x10590;
int32_t x10592 = 64 * x10591;
float* x10602 = (float*)myMalloc(x10592 * sizeof(float));;
int32_t x10603 = 0;
int32_t x10604 = 0;
int32_t x10605 = 0;
bool x10651 = x10456 > 1;
bool x10655 = x10569 > 1;
for(int x10606=0; x10606 < 64; x10606++) {
int32_t x10607 = x10604;
int32_t x10608 = x10605;
int32_t x10609 = x10603;
int32_t x10610 = x10609;
int32_t x10611 = x10607;
int32_t x10612 = x10608;
for(int x10614=0; x10614 < x10587; x10614++) {
int32_t x10615 = x10611;
int32_t x10616 = x10612;
int32_t x10617 = x10610;
int32_t x10618 = x10617;
int32_t x10619 = x10615;
int32_t x10620 = x10616;
for(int x10622=0; x10622 < x10589; x10622++) {
int32_t x10623 = x10619;
int32_t x10624 = x10620;
int32_t x10625 = x10618;
int32_t x10626 = x10625;
int32_t x10627 = x10623;
int32_t x10628 = x10624;
for(int x10629=0; x10629 < x10589; x10629++) {
int32_t x10630 = x10626;
int32_t x10631 = x10627;
float x10632 = x10471[x10631];
int32_t x10633 = x10628;
float x10634 = x10538[x10633];
float x10635 = x10632 / x10634;
x10602[x10630] = x10635;
x10626 += 1;
if (x10638) {
x10627 += 1;
} else {
}

}
x10618 += x10589;
if (x10638) {
x10619 += x10458;
} else {
}

}
x10610 += x10590;
if (x10651) {
x10611 += x10459;
} else {
}
if (x10655) {
x10612 += 1;
} else {
}

}
x10603 += x10591;
x10604 += x10460;

}
int32_t x10665 = 0;
int32_t x10666 = 1;
x10666 *= 1;
x10665 += 1;
x10666 *= 1;
x10666 *= 1;
int32_t x10671 = x10665;
bool x10672 = x10671 >= 2;
if (x10672) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10677 = x10671 == 0;
if (x10677) {
int32_t x10678 = x10666;
bool x10679 = x10678 == 512;
if (x10679) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10686 = x10666;
int32_t x10687 = 512 / x10686;
bool x10693;
if (x452) {
bool x10688 = x10587 == 1;
bool x10689 = x10687 == 1;
bool x10690 = x10688 || x10689;
bool x10691 = x10587 == x10687;
bool x10692 = x10690 || x10691;
x10693 = x10692;
} else {
x10693 = false;
}
bool x10697;
if (x10693) {
x10697 = x10696;
} else {
x10697 = false;
}
bool x10698;
if (x10697) {
x10698 = x10696;
} else {
x10698 = false;
}
if (x10698) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10587,x10589,x10589,1,x10687,1,1);
assert(false && "");
}
bool x10704 = x10587 <= x10687;
int32_t x10705;
if (x10704) {
x10705 = x10687;
} else {
x10705 = x10587;
}
bool x10711 = x10705 > 0;
bool x10713;
if (x10711) {
x10713 = x10712;
} else {
x10713 = false;
}
bool x10714;
if (x10713) {
x10714 = x10712;
} else {
x10714 = false;
}
if (x10714) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(10587) x Sym(10589) x Sym(10589)"," x Const(1) x Sym(10687) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x10709 = x10705 * x10708;
int32_t x10710 = 64 * x10709;
float* x10720 = (float*)myMalloc(x10710 * sizeof(float));;
int32_t x10721 = 0;
int32_t x10722 = 0;
int32_t x10723 = 0;
bool x10769 = x10587 > 1;
bool x10773 = x10687 > 1;
for(int x10724=0; x10724 < 64; x10724++) {
int32_t x10725 = x10722;
int32_t x10726 = x10723;
int32_t x10727 = x10721;
int32_t x10728 = x10727;
int32_t x10729 = x10725;
int32_t x10730 = x10726;
for(int x10732=0; x10732 < x10705; x10732++) {
int32_t x10733 = x10729;
int32_t x10734 = x10730;
int32_t x10735 = x10728;
int32_t x10736 = x10735;
int32_t x10737 = x10733;
int32_t x10738 = x10734;
for(int x10740=0; x10740 < x10707; x10740++) {
int32_t x10741 = x10737;
int32_t x10742 = x10738;
int32_t x10743 = x10736;
int32_t x10744 = x10743;
int32_t x10745 = x10741;
int32_t x10746 = x10742;
for(int x10747=0; x10747 < x10707; x10747++) {
int32_t x10748 = x10744;
int32_t x10749 = x10745;
float x10750 = x10602[x10749];
int32_t x10751 = x10746;
float x10752 = x52[x10751];
float x10753 = x10750 * x10752;
x10720[x10748] = x10753;
x10744 += 1;
if (x10756) {
x10745 += 1;
} else {
}

}
x10736 += x10707;
if (x10756) {
x10737 += x10589;
} else {
}

}
x10728 += x10708;
if (x10769) {
x10729 += x10590;
} else {
}
if (x10773) {
x10730 += 1;
} else {
}

}
x10721 += x10709;
x10722 += x10591;

}
int32_t x10783 = 0;
int32_t x10784 = 1;
x10784 *= 1;
x10783 += 1;
x10784 *= 1;
x10784 *= 1;
int32_t x10789 = x10783;
bool x10790 = x10789 >= 2;
if (x10790) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x10795 = x10789 == 0;
if (x10795) {
int32_t x10796 = x10784;
bool x10797 = x10796 == 512;
if (x10797) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x10804 = x10784;
int32_t x10805 = 512 / x10804;
bool x10811;
if (x452) {
bool x10806 = x10705 == 1;
bool x10807 = x10805 == 1;
bool x10808 = x10806 || x10807;
bool x10809 = x10705 == x10805;
bool x10810 = x10808 || x10809;
x10811 = x10810;
} else {
x10811 = false;
}
bool x10815;
if (x10811) {
x10815 = x10814;
} else {
x10815 = false;
}
bool x10816;
if (x10815) {
x10816 = x10814;
} else {
x10816 = false;
}
if (x10816) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10705,x10707,x10707,1,x10805,1,1);
assert(false && "");
}
bool x10822 = x10705 <= x10805;
int32_t x10823;
if (x10822) {
x10823 = x10805;
} else {
x10823 = x10705;
}
bool x10829 = x10823 > 0;
bool x10831;
if (x10829) {
x10831 = x10830;
} else {
x10831 = false;
}
bool x10832;
if (x10831) {
x10832 = x10830;
} else {
x10832 = false;
}
if (x10832) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(10705) x Sym(10707) x Sym(10707)"," x Const(1) x Sym(10805) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x10827 = x10823 * x10826;
int32_t x10828 = 64 * x10827;
float* x10838 = (float*)myMalloc(x10828 * sizeof(float));;
int32_t x10839 = 0;
int32_t x10840 = 0;
int32_t x10841 = 0;
bool x10887 = x10705 > 1;
bool x10891 = x10805 > 1;
for(int x10842=0; x10842 < 64; x10842++) {
int32_t x10843 = x10840;
int32_t x10844 = x10841;
int32_t x10845 = x10839;
int32_t x10846 = x10845;
int32_t x10847 = x10843;
int32_t x10848 = x10844;
for(int x10850=0; x10850 < x10823; x10850++) {
int32_t x10851 = x10847;
int32_t x10852 = x10848;
int32_t x10853 = x10846;
int32_t x10854 = x10853;
int32_t x10855 = x10851;
int32_t x10856 = x10852;
for(int x10858=0; x10858 < x10825; x10858++) {
int32_t x10859 = x10855;
int32_t x10860 = x10856;
int32_t x10861 = x10854;
int32_t x10862 = x10861;
int32_t x10863 = x10859;
int32_t x10864 = x10860;
for(int x10865=0; x10865 < x10825; x10865++) {
int32_t x10866 = x10862;
int32_t x10867 = x10863;
float x10868 = x10720[x10867];
int32_t x10869 = x10864;
float x10870 = x246[x10869];
float x10871 = x10868 + x10870;
x10838[x10866] = x10871;
x10862 += 1;
if (x10874) {
x10863 += 1;
} else {
}

}
x10854 += x10825;
if (x10874) {
x10855 += x10707;
} else {
}

}
x10846 += x10826;
if (x10887) {
x10847 += x10708;
} else {
}
if (x10891) {
x10848 += 1;
} else {
}

}
x10839 += x10827;
x10840 += x10709;

}
bool x10901 = x10823 == 1;
bool x10902 = x10901 || x9109;
bool x10903 = x10823 == x8489;
bool x10904 = x10902 || x10903;
bool x10909;
if (x10904) {
x10909 = x10908;
} else {
x10909 = false;
}
bool x10910;
if (x10909) {
x10910 = x10908;
} else {
x10910 = false;
}
if (x10910) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x10823,x10825,x10825,64,x8489,x8491,x8491);
assert(false && "");
}
int32_t x10923 = 0;
int32_t x10924 = 0;
int32_t x10925 = 0;
bool x10916 = x10823 <= x8489;
int32_t x10917;
if (x10916) {
x10917 = x8489;
} else {
x10917 = x10823;
}
bool x10976 = x10823 > 1;
int32_t x10921 = x10917 * x10920;
for(int x10926=0; x10926 < 64; x10926++) {
int32_t x10927 = x10924;
int32_t x10928 = x10925;
int32_t x10929 = x10923;
int32_t x10930 = x10929;
int32_t x10931 = x10927;
int32_t x10932 = x10928;
for(int x10934=0; x10934 < x10917; x10934++) {
int32_t x10935 = x10931;
int32_t x10936 = x10932;
int32_t x10937 = x10930;
int32_t x10938 = x10937;
int32_t x10939 = x10935;
int32_t x10940 = x10936;
for(int x10942=0; x10942 < x10919; x10942++) {
int32_t x10943 = x10939;
int32_t x10944 = x10940;
int32_t x10945 = x10938;
int32_t x10946 = x10945;
int32_t x10947 = x10943;
int32_t x10948 = x10944;
for(int x10949=0; x10949 < x10919; x10949++) {
int32_t x10950 = x10947;
float x10951 = x10838[x10950];
int32_t x10952 = x10948;
float x10953 = x9202[x10952];
float x10954 = x10951 + x10953;
x10838[x10950] = x10954;
x10946 += 1;
if (x10957) {
x10947 += 1;
} else {
}
if (x9167) {
x10948 += 1;
} else {
}

}
x10938 += x10919;
if (x10957) {
x10939 += x10825;
} else {
}
if (x9167) {
x10940 += x8491;
} else {
}

}
x10930 += x10920;
if (x10976) {
x10931 += x10826;
} else {
}
if (x9187) {
x10932 += x8492;
} else {
}

}
x10923 += x10921;
x10924 += x10827;
x10925 += x8493;

}
float* x10990 = (float*)myMalloc(x10828 * sizeof(float));;
for(int x10992=0; x10992 < x10828; x10992++) {
float x10993 = x10838[x10992];
bool x10994 = x10993 < 0.0f;
if (x10994) {
x10990[x10992] = 0.0f;
} else {
float x10997 = x10838[x10992];
x10990[x10992] = x10997;
}

}
float* x11011 = (float*)myMalloc(x11010 * sizeof(float));;
int32_t x11014 = 64 * x10823;
int32_t x11015 = x11014 * x11006;
float* x11016 = (float*)myMalloc(x11015 * sizeof(float));;
int32_t x11012 = x10823 * x11006;
for(int x11017=0; x11017 < 64; x11017++) {
int32_t x11018 = x11017 * x10827;
float* x11019 = x10990+x11018;
int32_t x11020 = x11017 * x11007;
float* x11021 = x11011+x11020;
int32_t x11022 = x11017 * x11012;
float* x11023 = x11016+x11022;
for(int x11024=0; x11024 < x10823; x11024++) {
int32_t x11025 = x11024 / 1;
int32_t x11029 = x11025 * x11005;
int32_t x11030 = x11029 * x11005;
int32_t x11026 = x11024 % 1;
int32_t x11027 = x11026 / 1;
int32_t x11031 = x11027 * x11005;
int32_t x11032 = x11031 * x11005;
int32_t x11033 = x11030 + x11032;
int32_t x11028 = x11026 % 1;
int32_t x11034 = x11028 * x11005;
int32_t x11035 = x11034 * x11005;
int32_t x11036 = x11033 + x11035;
float* x11037 = x11023+x11036;
int32_t x11038 = x11025 * x10825;
int32_t x11039 = x11038 * x10825;
float* x11040 = x11019+x11039;
for(int x11042=0; x11042 < x11005; x11042++) {
int32_t x11044 = x11042 * x11005;
float* x11045 = x11037+x11044;
int32_t x11043 = x11042 + x11027;
int32_t x11046 = x11043 * x10825;
int32_t x11047 = x11046 + x11028;
float* x11048 = x11040+x11047;
memcpy(x11045, x11048, 4 * x11005);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x11006,x10823,1,x196,x10823,x11023,x11006,1,x11021,x11006);

}
int32_t x11057 = 0;
int32_t x11058 = 1;
x11058 *= 1;
x11057 += 1;
x11058 *= 1;
x11058 *= 1;
int32_t x11063 = x11057;
bool x11064 = x11063 >= 2;
if (x11064) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11069 = x11063 == 0;
if (x11069) {
int32_t x11070 = x11058;
bool x11071 = x11070 == 128;
if (x11071) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11078 = x11058;
int32_t x11079 = 128 / x11078;
bool x11083;
if (x452) {
bool x11080 = x11079 == 1;
bool x11081 = 128 == x11079;
bool x11082 = x11080 || x11081;
x11083 = x11082;
} else {
x11083 = false;
}
bool x11087;
if (x11083) {
x11087 = x11086;
} else {
x11087 = false;
}
bool x11088;
if (x11087) {
x11088 = x11086;
} else {
x11088 = false;
}
if (x11088) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x11005,x11005,1,x11079,1,1);
assert(false && "");
}
bool x11094 = 128 <= x11079;
int32_t x11095;
if (x11094) {
x11095 = x11079;
} else {
x11095 = 128;
}
bool x11101 = x11095 > 0;
bool x11103;
if (x11101) {
x11103 = x11102;
} else {
x11103 = false;
}
bool x11104;
if (x11103) {
x11104 = x11102;
} else {
x11104 = false;
}
if (x11104) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(128) x Sym(11005) x Sym(11005)"," x Const(1) x Sym(11079) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x11099 = x11095 * x11098;
int32_t x11100 = 64 * x11099;
float* x11110 = (float*)myMalloc(x11100 * sizeof(float));;
int32_t x11111 = 0;
int32_t x11112 = 0;
int32_t x11113 = 0;
bool x11160 = x11079 > 1;
for(int x11114=0; x11114 < 64; x11114++) {
int32_t x11115 = x11112;
int32_t x11116 = x11113;
int32_t x11117 = x11111;
int32_t x11118 = x11117;
int32_t x11119 = x11115;
int32_t x11120 = x11116;
for(int x11122=0; x11122 < x11095; x11122++) {
int32_t x11123 = x11119;
int32_t x11124 = x11120;
int32_t x11125 = x11118;
int32_t x11126 = x11125;
int32_t x11127 = x11123;
int32_t x11128 = x11124;
for(int x11130=0; x11130 < x11097; x11130++) {
int32_t x11131 = x11127;
int32_t x11132 = x11128;
int32_t x11133 = x11126;
int32_t x11134 = x11133;
int32_t x11135 = x11131;
int32_t x11136 = x11132;
for(int x11137=0; x11137 < x11097; x11137++) {
int32_t x11138 = x11134;
int32_t x11139 = x11135;
float x11140 = x11011[x11139];
int32_t x11141 = x11136;
float x11142 = x112[x11141];
float x11143 = x11140 - x11142;
x11110[x11138] = x11143;
x11134 += 1;
if (x11146) {
x11135 += 1;
} else {
}

}
x11126 += x11097;
if (x11146) {
x11127 += x11005;
} else {
}

}
x11118 += x11098;
x11119 += x11006;
if (x11160) {
x11120 += 1;
} else {
}

}
x11111 += x11099;
x11112 += x11007;

}
float* x11170 = (float*)myMalloc(128 * sizeof(float));;
for(int x11171=0; x11171 < 128; x11171++) {
float x11172 = x9[x11171];
float x11173 = x11172 + 1.0E-5f;
x11170[x11171] = x11173;

}
float* x11177 = (float*)myMalloc(128 * sizeof(float));;
for(int x11178=0; x11178 < 128; x11178++) {
float x11179 = x11170[x11178];
double x11180 = (double)x11179;
double x11181 = sqrt(x11180);
float x11182 = (float)x11181;
x11177[x11178] = x11182;

}
int32_t x11186 = 0;
int32_t x11187 = 1;
x11187 *= 1;
x11186 += 1;
x11187 *= 1;
x11187 *= 1;
int32_t x11192 = x11186;
bool x11193 = x11192 >= 2;
if (x11193) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11198 = x11192 == 0;
if (x11198) {
int32_t x11199 = x11187;
bool x11200 = x11199 == 128;
if (x11200) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11207 = x11187;
int32_t x11208 = 128 / x11207;
bool x11214;
if (x452) {
bool x11209 = x11095 == 1;
bool x11210 = x11208 == 1;
bool x11211 = x11209 || x11210;
bool x11212 = x11095 == x11208;
bool x11213 = x11211 || x11212;
x11214 = x11213;
} else {
x11214 = false;
}
bool x11218;
if (x11214) {
x11218 = x11217;
} else {
x11218 = false;
}
bool x11219;
if (x11218) {
x11219 = x11217;
} else {
x11219 = false;
}
if (x11219) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11095,x11097,x11097,1,x11208,1,1);
assert(false && "");
}
bool x11225 = x11095 <= x11208;
int32_t x11226;
if (x11225) {
x11226 = x11208;
} else {
x11226 = x11095;
}
bool x11232 = x11226 > 0;
bool x11234;
if (x11232) {
x11234 = x11233;
} else {
x11234 = false;
}
bool x11235;
if (x11234) {
x11235 = x11233;
} else {
x11235 = false;
}
if (x11235) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(11095) x Sym(11097) x Sym(11097)"," x Const(1) x Sym(11208) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x11230 = x11226 * x11229;
int32_t x11231 = 64 * x11230;
float* x11241 = (float*)myMalloc(x11231 * sizeof(float));;
int32_t x11242 = 0;
int32_t x11243 = 0;
int32_t x11244 = 0;
bool x11290 = x11095 > 1;
bool x11294 = x11208 > 1;
for(int x11245=0; x11245 < 64; x11245++) {
int32_t x11246 = x11243;
int32_t x11247 = x11244;
int32_t x11248 = x11242;
int32_t x11249 = x11248;
int32_t x11250 = x11246;
int32_t x11251 = x11247;
for(int x11253=0; x11253 < x11226; x11253++) {
int32_t x11254 = x11250;
int32_t x11255 = x11251;
int32_t x11256 = x11249;
int32_t x11257 = x11256;
int32_t x11258 = x11254;
int32_t x11259 = x11255;
for(int x11261=0; x11261 < x11228; x11261++) {
int32_t x11262 = x11258;
int32_t x11263 = x11259;
int32_t x11264 = x11257;
int32_t x11265 = x11264;
int32_t x11266 = x11262;
int32_t x11267 = x11263;
for(int x11268=0; x11268 < x11228; x11268++) {
int32_t x11269 = x11265;
int32_t x11270 = x11266;
float x11271 = x11110[x11270];
int32_t x11272 = x11267;
float x11273 = x11177[x11272];
float x11274 = x11271 / x11273;
x11241[x11269] = x11274;
x11265 += 1;
if (x11277) {
x11266 += 1;
} else {
}

}
x11257 += x11228;
if (x11277) {
x11258 += x11097;
} else {
}

}
x11249 += x11229;
if (x11290) {
x11250 += x11098;
} else {
}
if (x11294) {
x11251 += 1;
} else {
}

}
x11242 += x11230;
x11243 += x11099;

}
int32_t x11304 = 0;
int32_t x11305 = 1;
x11305 *= 1;
x11304 += 1;
x11305 *= 1;
x11305 *= 1;
int32_t x11310 = x11304;
bool x11311 = x11310 >= 2;
if (x11311) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11316 = x11310 == 0;
if (x11316) {
int32_t x11317 = x11305;
bool x11318 = x11317 == 128;
if (x11318) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11325 = x11305;
int32_t x11326 = 128 / x11325;
bool x11332;
if (x452) {
bool x11327 = x11226 == 1;
bool x11328 = x11326 == 1;
bool x11329 = x11327 || x11328;
bool x11330 = x11226 == x11326;
bool x11331 = x11329 || x11330;
x11332 = x11331;
} else {
x11332 = false;
}
bool x11336;
if (x11332) {
x11336 = x11335;
} else {
x11336 = false;
}
bool x11337;
if (x11336) {
x11337 = x11335;
} else {
x11337 = false;
}
if (x11337) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11226,x11228,x11228,1,x11326,1,1);
assert(false && "");
}
bool x11343 = x11226 <= x11326;
int32_t x11344;
if (x11343) {
x11344 = x11326;
} else {
x11344 = x11226;
}
bool x11350 = x11344 > 0;
bool x11352;
if (x11350) {
x11352 = x11351;
} else {
x11352 = false;
}
bool x11353;
if (x11352) {
x11353 = x11351;
} else {
x11353 = false;
}
if (x11353) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(11226) x Sym(11228) x Sym(11228)"," x Const(1) x Sym(11326) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x11348 = x11344 * x11347;
int32_t x11349 = 64 * x11348;
float* x11359 = (float*)myMalloc(x11349 * sizeof(float));;
int32_t x11360 = 0;
int32_t x11361 = 0;
int32_t x11362 = 0;
bool x11408 = x11226 > 1;
bool x11412 = x11326 > 1;
for(int x11363=0; x11363 < 64; x11363++) {
int32_t x11364 = x11361;
int32_t x11365 = x11362;
int32_t x11366 = x11360;
int32_t x11367 = x11366;
int32_t x11368 = x11364;
int32_t x11369 = x11365;
for(int x11371=0; x11371 < x11344; x11371++) {
int32_t x11372 = x11368;
int32_t x11373 = x11369;
int32_t x11374 = x11367;
int32_t x11375 = x11374;
int32_t x11376 = x11372;
int32_t x11377 = x11373;
for(int x11379=0; x11379 < x11346; x11379++) {
int32_t x11380 = x11376;
int32_t x11381 = x11377;
int32_t x11382 = x11375;
int32_t x11383 = x11382;
int32_t x11384 = x11380;
int32_t x11385 = x11381;
for(int x11386=0; x11386 < x11346; x11386++) {
int32_t x11387 = x11383;
int32_t x11388 = x11384;
float x11389 = x11241[x11388];
int32_t x11390 = x11385;
float x11391 = x45[x11390];
float x11392 = x11389 * x11391;
x11359[x11387] = x11392;
x11383 += 1;
if (x11395) {
x11384 += 1;
} else {
}

}
x11375 += x11346;
if (x11395) {
x11376 += x11228;
} else {
}

}
x11367 += x11347;
if (x11408) {
x11368 += x11229;
} else {
}
if (x11412) {
x11369 += 1;
} else {
}

}
x11360 += x11348;
x11361 += x11230;

}
int32_t x11422 = 0;
int32_t x11423 = 1;
x11423 *= 1;
x11422 += 1;
x11423 *= 1;
x11423 *= 1;
int32_t x11428 = x11422;
bool x11429 = x11428 >= 2;
if (x11429) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11434 = x11428 == 0;
if (x11434) {
int32_t x11435 = x11423;
bool x11436 = x11435 == 128;
if (x11436) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11443 = x11423;
int32_t x11444 = 128 / x11443;
bool x11450;
if (x452) {
bool x11445 = x11344 == 1;
bool x11446 = x11444 == 1;
bool x11447 = x11445 || x11446;
bool x11448 = x11344 == x11444;
bool x11449 = x11447 || x11448;
x11450 = x11449;
} else {
x11450 = false;
}
bool x11454;
if (x11450) {
x11454 = x11453;
} else {
x11454 = false;
}
bool x11455;
if (x11454) {
x11455 = x11453;
} else {
x11455 = false;
}
if (x11455) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11344,x11346,x11346,1,x11444,1,1);
assert(false && "");
}
bool x11461 = x11344 <= x11444;
int32_t x11462;
if (x11461) {
x11462 = x11444;
} else {
x11462 = x11344;
}
bool x11468 = x11462 > 0;
bool x11470;
if (x11468) {
x11470 = x11469;
} else {
x11470 = false;
}
bool x11471;
if (x11470) {
x11471 = x11469;
} else {
x11471 = false;
}
if (x11471) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(11344) x Sym(11346) x Sym(11346)"," x Const(1) x Sym(11444) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x11466 = x11462 * x11465;
int32_t x11467 = 64 * x11466;
float* x11477 = (float*)myMalloc(x11467 * sizeof(float));;
int32_t x11478 = 0;
int32_t x11479 = 0;
int32_t x11480 = 0;
bool x11526 = x11344 > 1;
bool x11530 = x11444 > 1;
for(int x11481=0; x11481 < 64; x11481++) {
int32_t x11482 = x11479;
int32_t x11483 = x11480;
int32_t x11484 = x11478;
int32_t x11485 = x11484;
int32_t x11486 = x11482;
int32_t x11487 = x11483;
for(int x11489=0; x11489 < x11462; x11489++) {
int32_t x11490 = x11486;
int32_t x11491 = x11487;
int32_t x11492 = x11485;
int32_t x11493 = x11492;
int32_t x11494 = x11490;
int32_t x11495 = x11491;
for(int x11497=0; x11497 < x11464; x11497++) {
int32_t x11498 = x11494;
int32_t x11499 = x11495;
int32_t x11500 = x11493;
int32_t x11501 = x11500;
int32_t x11502 = x11498;
int32_t x11503 = x11499;
for(int x11504=0; x11504 < x11464; x11504++) {
int32_t x11505 = x11501;
int32_t x11506 = x11502;
float x11507 = x11359[x11506];
int32_t x11508 = x11503;
float x11509 = x170[x11508];
float x11510 = x11507 + x11509;
x11477[x11505] = x11510;
x11501 += 1;
if (x11513) {
x11502 += 1;
} else {
}

}
x11493 += x11464;
if (x11513) {
x11494 += x11346;
} else {
}

}
x11485 += x11465;
if (x11526) {
x11486 += x11347;
} else {
}
if (x11530) {
x11487 += 1;
} else {
}

}
x11478 += x11466;
x11479 += x11348;

}
float* x11540 = (float*)myMalloc(x11467 * sizeof(float));;
for(int x11542=0; x11542 < x11467; x11542++) {
float x11543 = x11477[x11542];
bool x11544 = x11543 < 0.0f;
if (x11544) {
x11540[x11542] = 0.0f;
} else {
float x11547 = x11477[x11542];
x11540[x11542] = x11547;
}

}
float* x11562 = (float*)myMalloc(x11561 * sizeof(float));;
int32_t x11563 = 9 * x11462;
int32_t x11566 = 64 * x11563;
int32_t x11567 = x11566 * x11557;
float* x11568 = (float*)myMalloc(x11567 * sizeof(float));;
int32_t x11564 = x11563 * x11557;
int32_t x11576 = x11462 * 3;
int32_t x11577 = x11576 * 3;
for(int x11569=0; x11569 < 64; x11569++) {
int32_t x11570 = x11569 * x11466;
float* x11571 = x11540+x11570;
int32_t x11572 = x11569 * x11558;
float* x11573 = x11562+x11572;
int32_t x11574 = x11569 * x11564;
float* x11575 = x11568+x11574;
for(int x11579=0; x11579 < x11577; x11579++) {
int32_t x11580 = x11579 / 9;
int32_t x11584 = x11580 * 3;
int32_t x11585 = x11584 * 3;
int32_t x11586 = x11585 * x11556;
int32_t x11587 = x11586 * x11556;
int32_t x11581 = x11579 % 9;
int32_t x11582 = x11581 / 3;
int32_t x11588 = x11582 * 3;
int32_t x11589 = x11588 * x11556;
int32_t x11590 = x11589 * x11556;
int32_t x11591 = x11587 + x11590;
int32_t x11583 = x11581 % 3;
int32_t x11592 = x11583 * x11556;
int32_t x11593 = x11592 * x11556;
int32_t x11594 = x11591 + x11593;
float* x11595 = x11575+x11594;
int32_t x11596 = x11580 * x11464;
int32_t x11597 = x11596 * x11464;
float* x11598 = x11571+x11597;
int32_t x11611 = 1 - x11583;
bool x11612 = x11611 > 0;
int32_t x11613;
if (x11612) {
x11613 = x11611;
} else {
x11613 = 0;
}
int32_t x11614 = 3 - x11583;
int32_t x11615 = x11614 - 1;
int32_t x11616 = 1 - x11615;
bool x11617 = x11616 > 0;
int32_t x11618;
if (x11617) {
x11618 = x11616;
} else {
x11618 = 0;
}
int32_t x11619 = x11556 - x11618;
int32_t x11620 = x11619 - x11613;
bool x11621 = x11620 <= 0;
bool x11625 = x11613 > 0;
int32_t x11610 = -1 + x11583;
bool x11638 = x11618 > 0;
for(int x11600=0; x11600 < x11556; x11600++) {
int32_t x11601 = x11600 - 1;
int32_t x11602 = x11601 + x11582;
bool x11603 = x11602 < 0;
bool x11604 = x11602 >= x11464;
bool x11605 = x11603 || x11604;
if (x11605) {
int32_t x11606 = x11600 * x11556;
float* x11607 = x11595+x11606;
memset(x11607, 0, 4 * x11556);;
} else {
if (x11621) {
int32_t x11606 = x11600 * x11556;
float* x11622 = x11595+x11606;
memset(x11622, 0, 4 * x11556);;
} else {
int32_t x11606 = x11600 * x11556;
if (x11625) {
float* x11626 = x11595+x11606;
memset(x11626, 0, 4 * x11613);;
} else {
}
// may have segfault here
int32_t x11631 = x11606 + x11613;
float* x11632 = x11595+x11631;
int32_t x11633 = x11602 * x11464;
int32_t x11634 = x11633 + x11610;
int32_t x11635 = x11634 + x11613;
float* x11636 = x11598+x11635;
memcpy(x11632, x11636, 4 * x11620);;
if (x11638) {
int32_t x11639 = x11606 + x11556;
int32_t x11640 = x11639 - x11618;
float* x11641 = x11595+x11640;
memset(x11641, 0, 4 * x11618);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x11557,x11563,1,x191,x11563,x11575,x11557,1,x11573,x11557);

}
int32_t x11656 = 0;
int32_t x11657 = 1;
x11657 *= 1;
x11656 += 1;
x11657 *= 1;
x11657 *= 1;
int32_t x11662 = x11656;
bool x11663 = x11662 >= 2;
if (x11663) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11668 = x11662 == 0;
if (x11668) {
int32_t x11669 = x11657;
bool x11670 = x11669 == 128;
if (x11670) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11677 = x11657;
int32_t x11678 = 128 / x11677;
bool x11682;
if (x452) {
bool x11679 = x11678 == 1;
bool x11680 = 128 == x11678;
bool x11681 = x11679 || x11680;
x11682 = x11681;
} else {
x11682 = false;
}
bool x11686;
if (x11682) {
x11686 = x11685;
} else {
x11686 = false;
}
bool x11687;
if (x11686) {
x11687 = x11685;
} else {
x11687 = false;
}
if (x11687) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x11556,x11556,1,x11678,1,1);
assert(false && "");
}
bool x11693 = 128 <= x11678;
int32_t x11694;
if (x11693) {
x11694 = x11678;
} else {
x11694 = 128;
}
bool x11700 = x11694 > 0;
bool x11702;
if (x11700) {
x11702 = x11701;
} else {
x11702 = false;
}
bool x11703;
if (x11702) {
x11703 = x11701;
} else {
x11703 = false;
}
if (x11703) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(128) x Sym(11556) x Sym(11556)"," x Const(1) x Sym(11678) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x11698 = x11694 * x11697;
int32_t x11699 = 64 * x11698;
float* x11709 = (float*)myMalloc(x11699 * sizeof(float));;
int32_t x11710 = 0;
int32_t x11711 = 0;
int32_t x11712 = 0;
bool x11759 = x11678 > 1;
for(int x11713=0; x11713 < 64; x11713++) {
int32_t x11714 = x11711;
int32_t x11715 = x11712;
int32_t x11716 = x11710;
int32_t x11717 = x11716;
int32_t x11718 = x11714;
int32_t x11719 = x11715;
for(int x11721=0; x11721 < x11694; x11721++) {
int32_t x11722 = x11718;
int32_t x11723 = x11719;
int32_t x11724 = x11717;
int32_t x11725 = x11724;
int32_t x11726 = x11722;
int32_t x11727 = x11723;
for(int x11729=0; x11729 < x11696; x11729++) {
int32_t x11730 = x11726;
int32_t x11731 = x11727;
int32_t x11732 = x11725;
int32_t x11733 = x11732;
int32_t x11734 = x11730;
int32_t x11735 = x11731;
for(int x11736=0; x11736 < x11696; x11736++) {
int32_t x11737 = x11733;
int32_t x11738 = x11734;
float x11739 = x11562[x11738];
int32_t x11740 = x11735;
float x11741 = x217[x11740];
float x11742 = x11739 - x11741;
x11709[x11737] = x11742;
x11733 += 1;
if (x11745) {
x11734 += 1;
} else {
}

}
x11725 += x11696;
if (x11745) {
x11726 += x11556;
} else {
}

}
x11717 += x11697;
x11718 += x11557;
if (x11759) {
x11719 += 1;
} else {
}

}
x11710 += x11698;
x11711 += x11558;

}
float* x11769 = (float*)myMalloc(128 * sizeof(float));;
for(int x11770=0; x11770 < 128; x11770++) {
float x11771 = x266[x11770];
float x11772 = x11771 + 1.0E-5f;
x11769[x11770] = x11772;

}
float* x11776 = (float*)myMalloc(128 * sizeof(float));;
for(int x11777=0; x11777 < 128; x11777++) {
float x11778 = x11769[x11777];
double x11779 = (double)x11778;
double x11780 = sqrt(x11779);
float x11781 = (float)x11780;
x11776[x11777] = x11781;

}
int32_t x11785 = 0;
int32_t x11786 = 1;
x11786 *= 1;
x11785 += 1;
x11786 *= 1;
x11786 *= 1;
int32_t x11791 = x11785;
bool x11792 = x11791 >= 2;
if (x11792) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11797 = x11791 == 0;
if (x11797) {
int32_t x11798 = x11786;
bool x11799 = x11798 == 128;
if (x11799) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11806 = x11786;
int32_t x11807 = 128 / x11806;
bool x11813;
if (x452) {
bool x11808 = x11694 == 1;
bool x11809 = x11807 == 1;
bool x11810 = x11808 || x11809;
bool x11811 = x11694 == x11807;
bool x11812 = x11810 || x11811;
x11813 = x11812;
} else {
x11813 = false;
}
bool x11817;
if (x11813) {
x11817 = x11816;
} else {
x11817 = false;
}
bool x11818;
if (x11817) {
x11818 = x11816;
} else {
x11818 = false;
}
if (x11818) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11694,x11696,x11696,1,x11807,1,1);
assert(false && "");
}
bool x11824 = x11694 <= x11807;
int32_t x11825;
if (x11824) {
x11825 = x11807;
} else {
x11825 = x11694;
}
bool x11831 = x11825 > 0;
bool x11833;
if (x11831) {
x11833 = x11832;
} else {
x11833 = false;
}
bool x11834;
if (x11833) {
x11834 = x11832;
} else {
x11834 = false;
}
if (x11834) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(11694) x Sym(11696) x Sym(11696)"," x Const(1) x Sym(11807) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x11829 = x11825 * x11828;
int32_t x11830 = 64 * x11829;
float* x11840 = (float*)myMalloc(x11830 * sizeof(float));;
int32_t x11841 = 0;
int32_t x11842 = 0;
int32_t x11843 = 0;
bool x11889 = x11694 > 1;
bool x11893 = x11807 > 1;
for(int x11844=0; x11844 < 64; x11844++) {
int32_t x11845 = x11842;
int32_t x11846 = x11843;
int32_t x11847 = x11841;
int32_t x11848 = x11847;
int32_t x11849 = x11845;
int32_t x11850 = x11846;
for(int x11852=0; x11852 < x11825; x11852++) {
int32_t x11853 = x11849;
int32_t x11854 = x11850;
int32_t x11855 = x11848;
int32_t x11856 = x11855;
int32_t x11857 = x11853;
int32_t x11858 = x11854;
for(int x11860=0; x11860 < x11827; x11860++) {
int32_t x11861 = x11857;
int32_t x11862 = x11858;
int32_t x11863 = x11856;
int32_t x11864 = x11863;
int32_t x11865 = x11861;
int32_t x11866 = x11862;
for(int x11867=0; x11867 < x11827; x11867++) {
int32_t x11868 = x11864;
int32_t x11869 = x11865;
float x11870 = x11709[x11869];
int32_t x11871 = x11866;
float x11872 = x11776[x11871];
float x11873 = x11870 / x11872;
x11840[x11868] = x11873;
x11864 += 1;
if (x11876) {
x11865 += 1;
} else {
}

}
x11856 += x11827;
if (x11876) {
x11857 += x11696;
} else {
}

}
x11848 += x11828;
if (x11889) {
x11849 += x11697;
} else {
}
if (x11893) {
x11850 += 1;
} else {
}

}
x11841 += x11829;
x11842 += x11698;

}
int32_t x11903 = 0;
int32_t x11904 = 1;
x11904 *= 1;
x11903 += 1;
x11904 *= 1;
x11904 *= 1;
int32_t x11909 = x11903;
bool x11910 = x11909 >= 2;
if (x11910) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x11915 = x11909 == 0;
if (x11915) {
int32_t x11916 = x11904;
bool x11917 = x11916 == 128;
if (x11917) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x11924 = x11904;
int32_t x11925 = 128 / x11924;
bool x11931;
if (x452) {
bool x11926 = x11825 == 1;
bool x11927 = x11925 == 1;
bool x11928 = x11926 || x11927;
bool x11929 = x11825 == x11925;
bool x11930 = x11928 || x11929;
x11931 = x11930;
} else {
x11931 = false;
}
bool x11935;
if (x11931) {
x11935 = x11934;
} else {
x11935 = false;
}
bool x11936;
if (x11935) {
x11936 = x11934;
} else {
x11936 = false;
}
if (x11936) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11825,x11827,x11827,1,x11925,1,1);
assert(false && "");
}
bool x11942 = x11825 <= x11925;
int32_t x11943;
if (x11942) {
x11943 = x11925;
} else {
x11943 = x11825;
}
bool x11949 = x11943 > 0;
bool x11951;
if (x11949) {
x11951 = x11950;
} else {
x11951 = false;
}
bool x11952;
if (x11951) {
x11952 = x11950;
} else {
x11952 = false;
}
if (x11952) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(11825) x Sym(11827) x Sym(11827)"," x Const(1) x Sym(11925) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x11947 = x11943 * x11946;
int32_t x11948 = 64 * x11947;
float* x11958 = (float*)myMalloc(x11948 * sizeof(float));;
int32_t x11959 = 0;
int32_t x11960 = 0;
int32_t x11961 = 0;
bool x12007 = x11825 > 1;
bool x12011 = x11925 > 1;
for(int x11962=0; x11962 < 64; x11962++) {
int32_t x11963 = x11960;
int32_t x11964 = x11961;
int32_t x11965 = x11959;
int32_t x11966 = x11965;
int32_t x11967 = x11963;
int32_t x11968 = x11964;
for(int x11970=0; x11970 < x11943; x11970++) {
int32_t x11971 = x11967;
int32_t x11972 = x11968;
int32_t x11973 = x11966;
int32_t x11974 = x11973;
int32_t x11975 = x11971;
int32_t x11976 = x11972;
for(int x11978=0; x11978 < x11945; x11978++) {
int32_t x11979 = x11975;
int32_t x11980 = x11976;
int32_t x11981 = x11974;
int32_t x11982 = x11981;
int32_t x11983 = x11979;
int32_t x11984 = x11980;
for(int x11985=0; x11985 < x11945; x11985++) {
int32_t x11986 = x11982;
int32_t x11987 = x11983;
float x11988 = x11840[x11987];
int32_t x11989 = x11984;
float x11990 = x127[x11989];
float x11991 = x11988 * x11990;
x11958[x11986] = x11991;
x11982 += 1;
if (x11994) {
x11983 += 1;
} else {
}

}
x11974 += x11945;
if (x11994) {
x11975 += x11827;
} else {
}

}
x11966 += x11946;
if (x12007) {
x11967 += x11828;
} else {
}
if (x12011) {
x11968 += 1;
} else {
}

}
x11959 += x11947;
x11960 += x11829;

}
int32_t x12021 = 0;
int32_t x12022 = 1;
x12022 *= 1;
x12021 += 1;
x12022 *= 1;
x12022 *= 1;
int32_t x12027 = x12021;
bool x12028 = x12027 >= 2;
if (x12028) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12033 = x12027 == 0;
if (x12033) {
int32_t x12034 = x12022;
bool x12035 = x12034 == 128;
if (x12035) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12042 = x12022;
int32_t x12043 = 128 / x12042;
bool x12049;
if (x452) {
bool x12044 = x11943 == 1;
bool x12045 = x12043 == 1;
bool x12046 = x12044 || x12045;
bool x12047 = x11943 == x12043;
bool x12048 = x12046 || x12047;
x12049 = x12048;
} else {
x12049 = false;
}
bool x12053;
if (x12049) {
x12053 = x12052;
} else {
x12053 = false;
}
bool x12054;
if (x12053) {
x12054 = x12052;
} else {
x12054 = false;
}
if (x12054) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x11943,x11945,x11945,1,x12043,1,1);
assert(false && "");
}
bool x12060 = x11943 <= x12043;
int32_t x12061;
if (x12060) {
x12061 = x12043;
} else {
x12061 = x11943;
}
bool x12067 = x12061 > 0;
bool x12069;
if (x12067) {
x12069 = x12068;
} else {
x12069 = false;
}
bool x12070;
if (x12069) {
x12070 = x12068;
} else {
x12070 = false;
}
if (x12070) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(11943) x Sym(11945) x Sym(11945)"," x Const(1) x Sym(12043) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x12065 = x12061 * x12064;
int32_t x12066 = 64 * x12065;
float* x12076 = (float*)myMalloc(x12066 * sizeof(float));;
int32_t x12077 = 0;
int32_t x12078 = 0;
int32_t x12079 = 0;
bool x12125 = x11943 > 1;
bool x12129 = x12043 > 1;
for(int x12080=0; x12080 < 64; x12080++) {
int32_t x12081 = x12078;
int32_t x12082 = x12079;
int32_t x12083 = x12077;
int32_t x12084 = x12083;
int32_t x12085 = x12081;
int32_t x12086 = x12082;
for(int x12088=0; x12088 < x12061; x12088++) {
int32_t x12089 = x12085;
int32_t x12090 = x12086;
int32_t x12091 = x12084;
int32_t x12092 = x12091;
int32_t x12093 = x12089;
int32_t x12094 = x12090;
for(int x12096=0; x12096 < x12063; x12096++) {
int32_t x12097 = x12093;
int32_t x12098 = x12094;
int32_t x12099 = x12092;
int32_t x12100 = x12099;
int32_t x12101 = x12097;
int32_t x12102 = x12098;
for(int x12103=0; x12103 < x12063; x12103++) {
int32_t x12104 = x12100;
int32_t x12105 = x12101;
float x12106 = x11958[x12105];
int32_t x12107 = x12102;
float x12108 = x61[x12107];
float x12109 = x12106 + x12108;
x12076[x12104] = x12109;
x12100 += 1;
if (x12112) {
x12101 += 1;
} else {
}

}
x12092 += x12063;
if (x12112) {
x12093 += x11945;
} else {
}

}
x12084 += x12064;
if (x12125) {
x12085 += x11946;
} else {
}
if (x12129) {
x12086 += 1;
} else {
}

}
x12077 += x12065;
x12078 += x11947;

}
float* x12139 = (float*)myMalloc(x12066 * sizeof(float));;
for(int x12141=0; x12141 < x12066; x12141++) {
float x12142 = x12076[x12141];
bool x12143 = x12142 < 0.0f;
if (x12143) {
x12139[x12141] = 0.0f;
} else {
float x12146 = x12076[x12141];
x12139[x12141] = x12146;
}

}
float* x12160 = (float*)myMalloc(x12159 * sizeof(float));;
int32_t x12163 = 64 * x12061;
int32_t x12164 = x12163 * x12155;
float* x12165 = (float*)myMalloc(x12164 * sizeof(float));;
int32_t x12161 = x12061 * x12155;
for(int x12166=0; x12166 < 64; x12166++) {
int32_t x12167 = x12166 * x12065;
float* x12168 = x12139+x12167;
int32_t x12169 = x12166 * x12156;
float* x12170 = x12160+x12169;
int32_t x12171 = x12166 * x12161;
float* x12172 = x12165+x12171;
for(int x12173=0; x12173 < x12061; x12173++) {
int32_t x12174 = x12173 / 1;
int32_t x12178 = x12174 * x12154;
int32_t x12179 = x12178 * x12154;
int32_t x12175 = x12173 % 1;
int32_t x12176 = x12175 / 1;
int32_t x12180 = x12176 * x12154;
int32_t x12181 = x12180 * x12154;
int32_t x12182 = x12179 + x12181;
int32_t x12177 = x12175 % 1;
int32_t x12183 = x12177 * x12154;
int32_t x12184 = x12183 * x12154;
int32_t x12185 = x12182 + x12184;
float* x12186 = x12172+x12185;
int32_t x12187 = x12174 * x12063;
int32_t x12188 = x12187 * x12063;
float* x12189 = x12168+x12188;
for(int x12191=0; x12191 < x12154; x12191++) {
int32_t x12193 = x12191 * x12154;
float* x12194 = x12186+x12193;
int32_t x12192 = x12191 + x12176;
int32_t x12195 = x12192 * x12063;
int32_t x12196 = x12195 + x12177;
float* x12197 = x12189+x12196;
memcpy(x12194, x12197, 4 * x12154);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x12155,x12061,1,x41,x12061,x12172,x12155,1,x12170,x12155);

}
int32_t x12206 = 0;
int32_t x12207 = 1;
x12207 *= 1;
x12206 += 1;
x12207 *= 1;
x12207 *= 1;
int32_t x12212 = x12206;
bool x12213 = x12212 >= 2;
if (x12213) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12218 = x12212 == 0;
if (x12218) {
int32_t x12219 = x12207;
bool x12220 = x12219 == 512;
if (x12220) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12227 = x12207;
int32_t x12228 = 512 / x12227;
bool x12232;
if (x452) {
bool x12229 = x12228 == 1;
bool x12230 = 512 == x12228;
bool x12231 = x12229 || x12230;
x12232 = x12231;
} else {
x12232 = false;
}
bool x12236;
if (x12232) {
x12236 = x12235;
} else {
x12236 = false;
}
bool x12237;
if (x12236) {
x12237 = x12235;
} else {
x12237 = false;
}
if (x12237) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x12154,x12154,1,x12228,1,1);
assert(false && "");
}
bool x12243 = 512 <= x12228;
int32_t x12244;
if (x12243) {
x12244 = x12228;
} else {
x12244 = 512;
}
bool x12250 = x12244 > 0;
bool x12252;
if (x12250) {
x12252 = x12251;
} else {
x12252 = false;
}
bool x12253;
if (x12252) {
x12253 = x12251;
} else {
x12253 = false;
}
if (x12253) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(12154) x Sym(12154)"," x Const(1) x Sym(12228) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x12248 = x12244 * x12247;
int32_t x12249 = 64 * x12248;
float* x12259 = (float*)myMalloc(x12249 * sizeof(float));;
int32_t x12260 = 0;
int32_t x12261 = 0;
int32_t x12262 = 0;
bool x12309 = x12228 > 1;
for(int x12263=0; x12263 < 64; x12263++) {
int32_t x12264 = x12261;
int32_t x12265 = x12262;
int32_t x12266 = x12260;
int32_t x12267 = x12266;
int32_t x12268 = x12264;
int32_t x12269 = x12265;
for(int x12271=0; x12271 < x12244; x12271++) {
int32_t x12272 = x12268;
int32_t x12273 = x12269;
int32_t x12274 = x12267;
int32_t x12275 = x12274;
int32_t x12276 = x12272;
int32_t x12277 = x12273;
for(int x12279=0; x12279 < x12246; x12279++) {
int32_t x12280 = x12276;
int32_t x12281 = x12277;
int32_t x12282 = x12275;
int32_t x12283 = x12282;
int32_t x12284 = x12280;
int32_t x12285 = x12281;
for(int x12286=0; x12286 < x12246; x12286++) {
int32_t x12287 = x12283;
int32_t x12288 = x12284;
float x12289 = x12160[x12288];
int32_t x12290 = x12285;
float x12291 = x25[x12290];
float x12292 = x12289 - x12291;
x12259[x12287] = x12292;
x12283 += 1;
if (x12295) {
x12284 += 1;
} else {
}

}
x12275 += x12246;
if (x12295) {
x12276 += x12154;
} else {
}

}
x12267 += x12247;
x12268 += x12155;
if (x12309) {
x12269 += 1;
} else {
}

}
x12260 += x12248;
x12261 += x12156;

}
float* x12319 = (float*)myMalloc(512 * sizeof(float));;
for(int x12320=0; x12320 < 512; x12320++) {
float x12321 = x223[x12320];
float x12322 = x12321 + 1.0E-5f;
x12319[x12320] = x12322;

}
float* x12326 = (float*)myMalloc(512 * sizeof(float));;
for(int x12327=0; x12327 < 512; x12327++) {
float x12328 = x12319[x12327];
double x12329 = (double)x12328;
double x12330 = sqrt(x12329);
float x12331 = (float)x12330;
x12326[x12327] = x12331;

}
int32_t x12335 = 0;
int32_t x12336 = 1;
x12336 *= 1;
x12335 += 1;
x12336 *= 1;
x12336 *= 1;
int32_t x12341 = x12335;
bool x12342 = x12341 >= 2;
if (x12342) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12347 = x12341 == 0;
if (x12347) {
int32_t x12348 = x12336;
bool x12349 = x12348 == 512;
if (x12349) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12356 = x12336;
int32_t x12357 = 512 / x12356;
bool x12363;
if (x452) {
bool x12358 = x12244 == 1;
bool x12359 = x12357 == 1;
bool x12360 = x12358 || x12359;
bool x12361 = x12244 == x12357;
bool x12362 = x12360 || x12361;
x12363 = x12362;
} else {
x12363 = false;
}
bool x12367;
if (x12363) {
x12367 = x12366;
} else {
x12367 = false;
}
bool x12368;
if (x12367) {
x12368 = x12366;
} else {
x12368 = false;
}
if (x12368) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12244,x12246,x12246,1,x12357,1,1);
assert(false && "");
}
bool x12374 = x12244 <= x12357;
int32_t x12375;
if (x12374) {
x12375 = x12357;
} else {
x12375 = x12244;
}
bool x12381 = x12375 > 0;
bool x12383;
if (x12381) {
x12383 = x12382;
} else {
x12383 = false;
}
bool x12384;
if (x12383) {
x12384 = x12382;
} else {
x12384 = false;
}
if (x12384) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(12244) x Sym(12246) x Sym(12246)"," x Const(1) x Sym(12357) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x12379 = x12375 * x12378;
int32_t x12380 = 64 * x12379;
float* x12390 = (float*)myMalloc(x12380 * sizeof(float));;
int32_t x12391 = 0;
int32_t x12392 = 0;
int32_t x12393 = 0;
bool x12439 = x12244 > 1;
bool x12443 = x12357 > 1;
for(int x12394=0; x12394 < 64; x12394++) {
int32_t x12395 = x12392;
int32_t x12396 = x12393;
int32_t x12397 = x12391;
int32_t x12398 = x12397;
int32_t x12399 = x12395;
int32_t x12400 = x12396;
for(int x12402=0; x12402 < x12375; x12402++) {
int32_t x12403 = x12399;
int32_t x12404 = x12400;
int32_t x12405 = x12398;
int32_t x12406 = x12405;
int32_t x12407 = x12403;
int32_t x12408 = x12404;
for(int x12410=0; x12410 < x12377; x12410++) {
int32_t x12411 = x12407;
int32_t x12412 = x12408;
int32_t x12413 = x12406;
int32_t x12414 = x12413;
int32_t x12415 = x12411;
int32_t x12416 = x12412;
for(int x12417=0; x12417 < x12377; x12417++) {
int32_t x12418 = x12414;
int32_t x12419 = x12415;
float x12420 = x12259[x12419];
int32_t x12421 = x12416;
float x12422 = x12326[x12421];
float x12423 = x12420 / x12422;
x12390[x12418] = x12423;
x12414 += 1;
if (x12426) {
x12415 += 1;
} else {
}

}
x12406 += x12377;
if (x12426) {
x12407 += x12246;
} else {
}

}
x12398 += x12378;
if (x12439) {
x12399 += x12247;
} else {
}
if (x12443) {
x12400 += 1;
} else {
}

}
x12391 += x12379;
x12392 += x12248;

}
int32_t x12453 = 0;
int32_t x12454 = 1;
x12454 *= 1;
x12453 += 1;
x12454 *= 1;
x12454 *= 1;
int32_t x12459 = x12453;
bool x12460 = x12459 >= 2;
if (x12460) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12465 = x12459 == 0;
if (x12465) {
int32_t x12466 = x12454;
bool x12467 = x12466 == 512;
if (x12467) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12474 = x12454;
int32_t x12475 = 512 / x12474;
bool x12481;
if (x452) {
bool x12476 = x12375 == 1;
bool x12477 = x12475 == 1;
bool x12478 = x12476 || x12477;
bool x12479 = x12375 == x12475;
bool x12480 = x12478 || x12479;
x12481 = x12480;
} else {
x12481 = false;
}
bool x12485;
if (x12481) {
x12485 = x12484;
} else {
x12485 = false;
}
bool x12486;
if (x12485) {
x12486 = x12484;
} else {
x12486 = false;
}
if (x12486) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12375,x12377,x12377,1,x12475,1,1);
assert(false && "");
}
bool x12492 = x12375 <= x12475;
int32_t x12493;
if (x12492) {
x12493 = x12475;
} else {
x12493 = x12375;
}
bool x12499 = x12493 > 0;
bool x12501;
if (x12499) {
x12501 = x12500;
} else {
x12501 = false;
}
bool x12502;
if (x12501) {
x12502 = x12500;
} else {
x12502 = false;
}
if (x12502) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(12375) x Sym(12377) x Sym(12377)"," x Const(1) x Sym(12475) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x12497 = x12493 * x12496;
int32_t x12498 = 64 * x12497;
float* x12508 = (float*)myMalloc(x12498 * sizeof(float));;
int32_t x12509 = 0;
int32_t x12510 = 0;
int32_t x12511 = 0;
bool x12557 = x12375 > 1;
bool x12561 = x12475 > 1;
for(int x12512=0; x12512 < 64; x12512++) {
int32_t x12513 = x12510;
int32_t x12514 = x12511;
int32_t x12515 = x12509;
int32_t x12516 = x12515;
int32_t x12517 = x12513;
int32_t x12518 = x12514;
for(int x12520=0; x12520 < x12493; x12520++) {
int32_t x12521 = x12517;
int32_t x12522 = x12518;
int32_t x12523 = x12516;
int32_t x12524 = x12523;
int32_t x12525 = x12521;
int32_t x12526 = x12522;
for(int x12528=0; x12528 < x12495; x12528++) {
int32_t x12529 = x12525;
int32_t x12530 = x12526;
int32_t x12531 = x12524;
int32_t x12532 = x12531;
int32_t x12533 = x12529;
int32_t x12534 = x12530;
for(int x12535=0; x12535 < x12495; x12535++) {
int32_t x12536 = x12532;
int32_t x12537 = x12533;
float x12538 = x12390[x12537];
int32_t x12539 = x12534;
float x12540 = x167[x12539];
float x12541 = x12538 * x12540;
x12508[x12536] = x12541;
x12532 += 1;
if (x12544) {
x12533 += 1;
} else {
}

}
x12524 += x12495;
if (x12544) {
x12525 += x12377;
} else {
}

}
x12516 += x12496;
if (x12557) {
x12517 += x12378;
} else {
}
if (x12561) {
x12518 += 1;
} else {
}

}
x12509 += x12497;
x12510 += x12379;

}
int32_t x12571 = 0;
int32_t x12572 = 1;
x12572 *= 1;
x12571 += 1;
x12572 *= 1;
x12572 *= 1;
int32_t x12577 = x12571;
bool x12578 = x12577 >= 2;
if (x12578) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12583 = x12577 == 0;
if (x12583) {
int32_t x12584 = x12572;
bool x12585 = x12584 == 512;
if (x12585) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12592 = x12572;
int32_t x12593 = 512 / x12592;
bool x12599;
if (x452) {
bool x12594 = x12493 == 1;
bool x12595 = x12593 == 1;
bool x12596 = x12594 || x12595;
bool x12597 = x12493 == x12593;
bool x12598 = x12596 || x12597;
x12599 = x12598;
} else {
x12599 = false;
}
bool x12603;
if (x12599) {
x12603 = x12602;
} else {
x12603 = false;
}
bool x12604;
if (x12603) {
x12604 = x12602;
} else {
x12604 = false;
}
if (x12604) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12493,x12495,x12495,1,x12593,1,1);
assert(false && "");
}
bool x12610 = x12493 <= x12593;
int32_t x12611;
if (x12610) {
x12611 = x12593;
} else {
x12611 = x12493;
}
bool x12617 = x12611 > 0;
bool x12619;
if (x12617) {
x12619 = x12618;
} else {
x12619 = false;
}
bool x12620;
if (x12619) {
x12620 = x12618;
} else {
x12620 = false;
}
if (x12620) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(12493) x Sym(12495) x Sym(12495)"," x Const(1) x Sym(12593) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x12615 = x12611 * x12614;
int32_t x12616 = 64 * x12615;
float* x12626 = (float*)myMalloc(x12616 * sizeof(float));;
int32_t x12627 = 0;
int32_t x12628 = 0;
int32_t x12629 = 0;
bool x12675 = x12493 > 1;
bool x12679 = x12593 > 1;
for(int x12630=0; x12630 < 64; x12630++) {
int32_t x12631 = x12628;
int32_t x12632 = x12629;
int32_t x12633 = x12627;
int32_t x12634 = x12633;
int32_t x12635 = x12631;
int32_t x12636 = x12632;
for(int x12638=0; x12638 < x12611; x12638++) {
int32_t x12639 = x12635;
int32_t x12640 = x12636;
int32_t x12641 = x12634;
int32_t x12642 = x12641;
int32_t x12643 = x12639;
int32_t x12644 = x12640;
for(int x12646=0; x12646 < x12613; x12646++) {
int32_t x12647 = x12643;
int32_t x12648 = x12644;
int32_t x12649 = x12642;
int32_t x12650 = x12649;
int32_t x12651 = x12647;
int32_t x12652 = x12648;
for(int x12653=0; x12653 < x12613; x12653++) {
int32_t x12654 = x12650;
int32_t x12655 = x12651;
float x12656 = x12508[x12655];
int32_t x12657 = x12652;
float x12658 = x82[x12657];
float x12659 = x12656 + x12658;
x12626[x12654] = x12659;
x12650 += 1;
if (x12662) {
x12651 += 1;
} else {
}

}
x12642 += x12613;
if (x12662) {
x12643 += x12495;
} else {
}

}
x12634 += x12614;
if (x12675) {
x12635 += x12496;
} else {
}
if (x12679) {
x12636 += 1;
} else {
}

}
x12627 += x12615;
x12628 += x12497;

}
bool x12689 = x12611 == 1;
bool x12690 = x12689 || x10901;
bool x12691 = x12611 == x10823;
bool x12692 = x12690 || x12691;
bool x12697;
if (x12692) {
x12697 = x12696;
} else {
x12697 = false;
}
bool x12698;
if (x12697) {
x12698 = x12696;
} else {
x12698 = false;
}
if (x12698) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12611,x12613,x12613,64,x10823,x10825,x10825);
assert(false && "");
}
int32_t x12711 = 0;
int32_t x12712 = 0;
int32_t x12713 = 0;
bool x12704 = x12611 <= x10823;
int32_t x12705;
if (x12704) {
x12705 = x10823;
} else {
x12705 = x12611;
}
bool x12764 = x12611 > 1;
int32_t x12709 = x12705 * x12708;
for(int x12714=0; x12714 < 64; x12714++) {
int32_t x12715 = x12712;
int32_t x12716 = x12713;
int32_t x12717 = x12711;
int32_t x12718 = x12717;
int32_t x12719 = x12715;
int32_t x12720 = x12716;
for(int x12722=0; x12722 < x12705; x12722++) {
int32_t x12723 = x12719;
int32_t x12724 = x12720;
int32_t x12725 = x12718;
int32_t x12726 = x12725;
int32_t x12727 = x12723;
int32_t x12728 = x12724;
for(int x12730=0; x12730 < x12707; x12730++) {
int32_t x12731 = x12727;
int32_t x12732 = x12728;
int32_t x12733 = x12726;
int32_t x12734 = x12733;
int32_t x12735 = x12731;
int32_t x12736 = x12732;
for(int x12737=0; x12737 < x12707; x12737++) {
int32_t x12738 = x12735;
float x12739 = x12626[x12738];
int32_t x12740 = x12736;
float x12741 = x10990[x12740];
float x12742 = x12739 + x12741;
x12626[x12738] = x12742;
x12734 += 1;
if (x12745) {
x12735 += 1;
} else {
}
if (x10957) {
x12736 += 1;
} else {
}

}
x12726 += x12707;
if (x12745) {
x12727 += x12613;
} else {
}
if (x10957) {
x12728 += x10825;
} else {
}

}
x12718 += x12708;
if (x12764) {
x12719 += x12614;
} else {
}
if (x10976) {
x12720 += x10826;
} else {
}

}
x12711 += x12709;
x12712 += x12615;
x12713 += x10827;

}
float* x12778 = (float*)myMalloc(x12616 * sizeof(float));;
for(int x12780=0; x12780 < x12616; x12780++) {
float x12781 = x12626[x12780];
bool x12782 = x12781 < 0.0f;
if (x12782) {
x12778[x12780] = 0.0f;
} else {
float x12785 = x12626[x12780];
x12778[x12780] = x12785;
}

}
float* x12799 = (float*)myMalloc(x12798 * sizeof(float));;
int32_t x12802 = 64 * x12611;
int32_t x12803 = x12802 * x12794;
float* x12804 = (float*)myMalloc(x12803 * sizeof(float));;
int32_t x12800 = x12611 * x12794;
for(int x12805=0; x12805 < 64; x12805++) {
int32_t x12806 = x12805 * x12615;
float* x12807 = x12778+x12806;
int32_t x12808 = x12805 * x12795;
float* x12809 = x12799+x12808;
int32_t x12810 = x12805 * x12800;
float* x12811 = x12804+x12810;
for(int x12812=0; x12812 < x12611; x12812++) {
int32_t x12813 = x12812 / 1;
int32_t x12817 = x12813 * x12793;
int32_t x12818 = x12817 * x12793;
int32_t x12814 = x12812 % 1;
int32_t x12815 = x12814 / 1;
int32_t x12819 = x12815 * x12793;
int32_t x12820 = x12819 * x12793;
int32_t x12821 = x12818 + x12820;
int32_t x12816 = x12814 % 1;
int32_t x12822 = x12816 * x12793;
int32_t x12823 = x12822 * x12793;
int32_t x12824 = x12821 + x12823;
float* x12825 = x12811+x12824;
int32_t x12826 = x12813 * x12613;
int32_t x12827 = x12826 * x12613;
float* x12828 = x12807+x12827;
for(int x12830=0; x12830 < x12793; x12830++) {
int32_t x12832 = x12830 * x12793;
float* x12833 = x12825+x12832;
int32_t x12831 = x12830 + x12815;
int32_t x12834 = x12831 * x12613;
int32_t x12835 = x12834 + x12816;
float* x12836 = x12828+x12835;
memcpy(x12833, x12836, 4 * x12793);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x12794,x12611,1,x132,x12611,x12811,x12794,1,x12809,x12794);

}
int32_t x12845 = 0;
int32_t x12846 = 1;
x12846 *= 1;
x12845 += 1;
x12846 *= 1;
x12846 *= 1;
int32_t x12851 = x12845;
bool x12852 = x12851 >= 2;
if (x12852) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12857 = x12851 == 0;
if (x12857) {
int32_t x12858 = x12846;
bool x12859 = x12858 == 128;
if (x12859) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12866 = x12846;
int32_t x12867 = 128 / x12866;
bool x12871;
if (x452) {
bool x12868 = x12867 == 1;
bool x12869 = 128 == x12867;
bool x12870 = x12868 || x12869;
x12871 = x12870;
} else {
x12871 = false;
}
bool x12875;
if (x12871) {
x12875 = x12874;
} else {
x12875 = false;
}
bool x12876;
if (x12875) {
x12876 = x12874;
} else {
x12876 = false;
}
if (x12876) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x12793,x12793,1,x12867,1,1);
assert(false && "");
}
bool x12882 = 128 <= x12867;
int32_t x12883;
if (x12882) {
x12883 = x12867;
} else {
x12883 = 128;
}
bool x12889 = x12883 > 0;
bool x12891;
if (x12889) {
x12891 = x12890;
} else {
x12891 = false;
}
bool x12892;
if (x12891) {
x12892 = x12890;
} else {
x12892 = false;
}
if (x12892) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(128) x Sym(12793) x Sym(12793)"," x Const(1) x Sym(12867) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x12887 = x12883 * x12886;
int32_t x12888 = 64 * x12887;
float* x12898 = (float*)myMalloc(x12888 * sizeof(float));;
int32_t x12899 = 0;
int32_t x12900 = 0;
int32_t x12901 = 0;
bool x12948 = x12867 > 1;
for(int x12902=0; x12902 < 64; x12902++) {
int32_t x12903 = x12900;
int32_t x12904 = x12901;
int32_t x12905 = x12899;
int32_t x12906 = x12905;
int32_t x12907 = x12903;
int32_t x12908 = x12904;
for(int x12910=0; x12910 < x12883; x12910++) {
int32_t x12911 = x12907;
int32_t x12912 = x12908;
int32_t x12913 = x12906;
int32_t x12914 = x12913;
int32_t x12915 = x12911;
int32_t x12916 = x12912;
for(int x12918=0; x12918 < x12885; x12918++) {
int32_t x12919 = x12915;
int32_t x12920 = x12916;
int32_t x12921 = x12914;
int32_t x12922 = x12921;
int32_t x12923 = x12919;
int32_t x12924 = x12920;
for(int x12925=0; x12925 < x12885; x12925++) {
int32_t x12926 = x12922;
int32_t x12927 = x12923;
float x12928 = x12799[x12927];
int32_t x12929 = x12924;
float x12930 = x236[x12929];
float x12931 = x12928 - x12930;
x12898[x12926] = x12931;
x12922 += 1;
if (x12934) {
x12923 += 1;
} else {
}

}
x12914 += x12885;
if (x12934) {
x12915 += x12793;
} else {
}

}
x12906 += x12886;
x12907 += x12794;
if (x12948) {
x12908 += 1;
} else {
}

}
x12899 += x12887;
x12900 += x12795;

}
float* x12958 = (float*)myMalloc(128 * sizeof(float));;
for(int x12959=0; x12959 < 128; x12959++) {
float x12960 = x261[x12959];
float x12961 = x12960 + 1.0E-5f;
x12958[x12959] = x12961;

}
float* x12965 = (float*)myMalloc(128 * sizeof(float));;
for(int x12966=0; x12966 < 128; x12966++) {
float x12967 = x12958[x12966];
double x12968 = (double)x12967;
double x12969 = sqrt(x12968);
float x12970 = (float)x12969;
x12965[x12966] = x12970;

}
int32_t x12974 = 0;
int32_t x12975 = 1;
x12975 *= 1;
x12974 += 1;
x12975 *= 1;
x12975 *= 1;
int32_t x12980 = x12974;
bool x12981 = x12980 >= 2;
if (x12981) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x12986 = x12980 == 0;
if (x12986) {
int32_t x12987 = x12975;
bool x12988 = x12987 == 128;
if (x12988) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x12995 = x12975;
int32_t x12996 = 128 / x12995;
bool x13002;
if (x452) {
bool x12997 = x12883 == 1;
bool x12998 = x12996 == 1;
bool x12999 = x12997 || x12998;
bool x13000 = x12883 == x12996;
bool x13001 = x12999 || x13000;
x13002 = x13001;
} else {
x13002 = false;
}
bool x13006;
if (x13002) {
x13006 = x13005;
} else {
x13006 = false;
}
bool x13007;
if (x13006) {
x13007 = x13005;
} else {
x13007 = false;
}
if (x13007) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x12883,x12885,x12885,1,x12996,1,1);
assert(false && "");
}
bool x13013 = x12883 <= x12996;
int32_t x13014;
if (x13013) {
x13014 = x12996;
} else {
x13014 = x12883;
}
bool x13020 = x13014 > 0;
bool x13022;
if (x13020) {
x13022 = x13021;
} else {
x13022 = false;
}
bool x13023;
if (x13022) {
x13023 = x13021;
} else {
x13023 = false;
}
if (x13023) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(12883) x Sym(12885) x Sym(12885)"," x Const(1) x Sym(12996) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x13018 = x13014 * x13017;
int32_t x13019 = 64 * x13018;
float* x13029 = (float*)myMalloc(x13019 * sizeof(float));;
int32_t x13030 = 0;
int32_t x13031 = 0;
int32_t x13032 = 0;
bool x13078 = x12883 > 1;
bool x13082 = x12996 > 1;
for(int x13033=0; x13033 < 64; x13033++) {
int32_t x13034 = x13031;
int32_t x13035 = x13032;
int32_t x13036 = x13030;
int32_t x13037 = x13036;
int32_t x13038 = x13034;
int32_t x13039 = x13035;
for(int x13041=0; x13041 < x13014; x13041++) {
int32_t x13042 = x13038;
int32_t x13043 = x13039;
int32_t x13044 = x13037;
int32_t x13045 = x13044;
int32_t x13046 = x13042;
int32_t x13047 = x13043;
for(int x13049=0; x13049 < x13016; x13049++) {
int32_t x13050 = x13046;
int32_t x13051 = x13047;
int32_t x13052 = x13045;
int32_t x13053 = x13052;
int32_t x13054 = x13050;
int32_t x13055 = x13051;
for(int x13056=0; x13056 < x13016; x13056++) {
int32_t x13057 = x13053;
int32_t x13058 = x13054;
float x13059 = x12898[x13058];
int32_t x13060 = x13055;
float x13061 = x12965[x13060];
float x13062 = x13059 / x13061;
x13029[x13057] = x13062;
x13053 += 1;
if (x13065) {
x13054 += 1;
} else {
}

}
x13045 += x13016;
if (x13065) {
x13046 += x12885;
} else {
}

}
x13037 += x13017;
if (x13078) {
x13038 += x12886;
} else {
}
if (x13082) {
x13039 += 1;
} else {
}

}
x13030 += x13018;
x13031 += x12887;

}
int32_t x13092 = 0;
int32_t x13093 = 1;
x13093 *= 1;
x13092 += 1;
x13093 *= 1;
x13093 *= 1;
int32_t x13098 = x13092;
bool x13099 = x13098 >= 2;
if (x13099) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13104 = x13098 == 0;
if (x13104) {
int32_t x13105 = x13093;
bool x13106 = x13105 == 128;
if (x13106) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13113 = x13093;
int32_t x13114 = 128 / x13113;
bool x13120;
if (x452) {
bool x13115 = x13014 == 1;
bool x13116 = x13114 == 1;
bool x13117 = x13115 || x13116;
bool x13118 = x13014 == x13114;
bool x13119 = x13117 || x13118;
x13120 = x13119;
} else {
x13120 = false;
}
bool x13124;
if (x13120) {
x13124 = x13123;
} else {
x13124 = false;
}
bool x13125;
if (x13124) {
x13125 = x13123;
} else {
x13125 = false;
}
if (x13125) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13014,x13016,x13016,1,x13114,1,1);
assert(false && "");
}
bool x13131 = x13014 <= x13114;
int32_t x13132;
if (x13131) {
x13132 = x13114;
} else {
x13132 = x13014;
}
bool x13138 = x13132 > 0;
bool x13140;
if (x13138) {
x13140 = x13139;
} else {
x13140 = false;
}
bool x13141;
if (x13140) {
x13141 = x13139;
} else {
x13141 = false;
}
if (x13141) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(13014) x Sym(13016) x Sym(13016)"," x Const(1) x Sym(13114) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x13136 = x13132 * x13135;
int32_t x13137 = 64 * x13136;
float* x13147 = (float*)myMalloc(x13137 * sizeof(float));;
int32_t x13148 = 0;
int32_t x13149 = 0;
int32_t x13150 = 0;
bool x13196 = x13014 > 1;
bool x13200 = x13114 > 1;
for(int x13151=0; x13151 < 64; x13151++) {
int32_t x13152 = x13149;
int32_t x13153 = x13150;
int32_t x13154 = x13148;
int32_t x13155 = x13154;
int32_t x13156 = x13152;
int32_t x13157 = x13153;
for(int x13159=0; x13159 < x13132; x13159++) {
int32_t x13160 = x13156;
int32_t x13161 = x13157;
int32_t x13162 = x13155;
int32_t x13163 = x13162;
int32_t x13164 = x13160;
int32_t x13165 = x13161;
for(int x13167=0; x13167 < x13134; x13167++) {
int32_t x13168 = x13164;
int32_t x13169 = x13165;
int32_t x13170 = x13163;
int32_t x13171 = x13170;
int32_t x13172 = x13168;
int32_t x13173 = x13169;
for(int x13174=0; x13174 < x13134; x13174++) {
int32_t x13175 = x13171;
int32_t x13176 = x13172;
float x13177 = x13029[x13176];
int32_t x13178 = x13173;
float x13179 = x39[x13178];
float x13180 = x13177 * x13179;
x13147[x13175] = x13180;
x13171 += 1;
if (x13183) {
x13172 += 1;
} else {
}

}
x13163 += x13134;
if (x13183) {
x13164 += x13016;
} else {
}

}
x13155 += x13135;
if (x13196) {
x13156 += x13017;
} else {
}
if (x13200) {
x13157 += 1;
} else {
}

}
x13148 += x13136;
x13149 += x13018;

}
int32_t x13210 = 0;
int32_t x13211 = 1;
x13211 *= 1;
x13210 += 1;
x13211 *= 1;
x13211 *= 1;
int32_t x13216 = x13210;
bool x13217 = x13216 >= 2;
if (x13217) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13222 = x13216 == 0;
if (x13222) {
int32_t x13223 = x13211;
bool x13224 = x13223 == 128;
if (x13224) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13231 = x13211;
int32_t x13232 = 128 / x13231;
bool x13238;
if (x452) {
bool x13233 = x13132 == 1;
bool x13234 = x13232 == 1;
bool x13235 = x13233 || x13234;
bool x13236 = x13132 == x13232;
bool x13237 = x13235 || x13236;
x13238 = x13237;
} else {
x13238 = false;
}
bool x13242;
if (x13238) {
x13242 = x13241;
} else {
x13242 = false;
}
bool x13243;
if (x13242) {
x13243 = x13241;
} else {
x13243 = false;
}
if (x13243) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13132,x13134,x13134,1,x13232,1,1);
assert(false && "");
}
bool x13249 = x13132 <= x13232;
int32_t x13250;
if (x13249) {
x13250 = x13232;
} else {
x13250 = x13132;
}
bool x13256 = x13250 > 0;
bool x13258;
if (x13256) {
x13258 = x13257;
} else {
x13258 = false;
}
bool x13259;
if (x13258) {
x13259 = x13257;
} else {
x13259 = false;
}
if (x13259) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(13132) x Sym(13134) x Sym(13134)"," x Const(1) x Sym(13232) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x13254 = x13250 * x13253;
int32_t x13255 = 64 * x13254;
float* x13265 = (float*)myMalloc(x13255 * sizeof(float));;
int32_t x13266 = 0;
int32_t x13267 = 0;
int32_t x13268 = 0;
bool x13314 = x13132 > 1;
bool x13318 = x13232 > 1;
for(int x13269=0; x13269 < 64; x13269++) {
int32_t x13270 = x13267;
int32_t x13271 = x13268;
int32_t x13272 = x13266;
int32_t x13273 = x13272;
int32_t x13274 = x13270;
int32_t x13275 = x13271;
for(int x13277=0; x13277 < x13250; x13277++) {
int32_t x13278 = x13274;
int32_t x13279 = x13275;
int32_t x13280 = x13273;
int32_t x13281 = x13280;
int32_t x13282 = x13278;
int32_t x13283 = x13279;
for(int x13285=0; x13285 < x13252; x13285++) {
int32_t x13286 = x13282;
int32_t x13287 = x13283;
int32_t x13288 = x13281;
int32_t x13289 = x13288;
int32_t x13290 = x13286;
int32_t x13291 = x13287;
for(int x13292=0; x13292 < x13252; x13292++) {
int32_t x13293 = x13289;
int32_t x13294 = x13290;
float x13295 = x13147[x13294];
int32_t x13296 = x13291;
float x13297 = x242[x13296];
float x13298 = x13295 + x13297;
x13265[x13293] = x13298;
x13289 += 1;
if (x13301) {
x13290 += 1;
} else {
}

}
x13281 += x13252;
if (x13301) {
x13282 += x13134;
} else {
}

}
x13273 += x13253;
if (x13314) {
x13274 += x13135;
} else {
}
if (x13318) {
x13275 += 1;
} else {
}

}
x13266 += x13254;
x13267 += x13136;

}
float* x13328 = (float*)myMalloc(x13255 * sizeof(float));;
for(int x13330=0; x13330 < x13255; x13330++) {
float x13331 = x13265[x13330];
bool x13332 = x13331 < 0.0f;
if (x13332) {
x13328[x13330] = 0.0f;
} else {
float x13335 = x13265[x13330];
x13328[x13330] = x13335;
}

}
float* x13350 = (float*)myMalloc(x13349 * sizeof(float));;
int32_t x13351 = 9 * x13250;
int32_t x13354 = 64 * x13351;
int32_t x13355 = x13354 * x13345;
float* x13356 = (float*)myMalloc(x13355 * sizeof(float));;
int32_t x13352 = x13351 * x13345;
int32_t x13364 = x13250 * 3;
int32_t x13365 = x13364 * 3;
for(int x13357=0; x13357 < 64; x13357++) {
int32_t x13358 = x13357 * x13254;
float* x13359 = x13328+x13358;
int32_t x13360 = x13357 * x13346;
float* x13361 = x13350+x13360;
int32_t x13362 = x13357 * x13352;
float* x13363 = x13356+x13362;
for(int x13367=0; x13367 < x13365; x13367++) {
int32_t x13368 = x13367 / 9;
int32_t x13372 = x13368 * 3;
int32_t x13373 = x13372 * 3;
int32_t x13374 = x13373 * x13344;
int32_t x13375 = x13374 * x13344;
int32_t x13369 = x13367 % 9;
int32_t x13370 = x13369 / 3;
int32_t x13376 = x13370 * 3;
int32_t x13377 = x13376 * x13344;
int32_t x13378 = x13377 * x13344;
int32_t x13379 = x13375 + x13378;
int32_t x13371 = x13369 % 3;
int32_t x13380 = x13371 * x13344;
int32_t x13381 = x13380 * x13344;
int32_t x13382 = x13379 + x13381;
float* x13383 = x13363+x13382;
int32_t x13384 = x13368 * x13252;
int32_t x13385 = x13384 * x13252;
float* x13386 = x13359+x13385;
int32_t x13399 = 1 - x13371;
bool x13400 = x13399 > 0;
int32_t x13401;
if (x13400) {
x13401 = x13399;
} else {
x13401 = 0;
}
int32_t x13402 = 3 - x13371;
int32_t x13403 = x13402 - 1;
int32_t x13404 = 1 - x13403;
bool x13405 = x13404 > 0;
int32_t x13406;
if (x13405) {
x13406 = x13404;
} else {
x13406 = 0;
}
int32_t x13407 = x13344 - x13406;
int32_t x13408 = x13407 - x13401;
bool x13409 = x13408 <= 0;
bool x13413 = x13401 > 0;
int32_t x13398 = -1 + x13371;
bool x13426 = x13406 > 0;
for(int x13388=0; x13388 < x13344; x13388++) {
int32_t x13389 = x13388 - 1;
int32_t x13390 = x13389 + x13370;
bool x13391 = x13390 < 0;
bool x13392 = x13390 >= x13252;
bool x13393 = x13391 || x13392;
if (x13393) {
int32_t x13394 = x13388 * x13344;
float* x13395 = x13383+x13394;
memset(x13395, 0, 4 * x13344);;
} else {
if (x13409) {
int32_t x13394 = x13388 * x13344;
float* x13410 = x13383+x13394;
memset(x13410, 0, 4 * x13344);;
} else {
int32_t x13394 = x13388 * x13344;
if (x13413) {
float* x13414 = x13383+x13394;
memset(x13414, 0, 4 * x13401);;
} else {
}
// may have segfault here
int32_t x13419 = x13394 + x13401;
float* x13420 = x13383+x13419;
int32_t x13421 = x13390 * x13252;
int32_t x13422 = x13421 + x13398;
int32_t x13423 = x13422 + x13401;
float* x13424 = x13386+x13423;
memcpy(x13420, x13424, 4 * x13408);;
if (x13426) {
int32_t x13427 = x13394 + x13344;
int32_t x13428 = x13427 - x13406;
float* x13429 = x13383+x13428;
memset(x13429, 0, 4 * x13406);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x13345,x13351,1,x165,x13351,x13363,x13345,1,x13361,x13345);

}
int32_t x13444 = 0;
int32_t x13445 = 1;
x13445 *= 1;
x13444 += 1;
x13445 *= 1;
x13445 *= 1;
int32_t x13450 = x13444;
bool x13451 = x13450 >= 2;
if (x13451) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13456 = x13450 == 0;
if (x13456) {
int32_t x13457 = x13445;
bool x13458 = x13457 == 128;
if (x13458) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13465 = x13445;
int32_t x13466 = 128 / x13465;
bool x13470;
if (x452) {
bool x13467 = x13466 == 1;
bool x13468 = 128 == x13466;
bool x13469 = x13467 || x13468;
x13470 = x13469;
} else {
x13470 = false;
}
bool x13474;
if (x13470) {
x13474 = x13473;
} else {
x13474 = false;
}
bool x13475;
if (x13474) {
x13475 = x13473;
} else {
x13475 = false;
}
if (x13475) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,128,x13344,x13344,1,x13466,1,1);
assert(false && "");
}
bool x13481 = 128 <= x13466;
int32_t x13482;
if (x13481) {
x13482 = x13466;
} else {
x13482 = 128;
}
bool x13488 = x13482 > 0;
bool x13490;
if (x13488) {
x13490 = x13489;
} else {
x13490 = false;
}
bool x13491;
if (x13490) {
x13491 = x13489;
} else {
x13491 = false;
}
if (x13491) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(128) x Sym(13344) x Sym(13344)"," x Const(1) x Sym(13466) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x13486 = x13482 * x13485;
int32_t x13487 = 64 * x13486;
float* x13497 = (float*)myMalloc(x13487 * sizeof(float));;
int32_t x13498 = 0;
int32_t x13499 = 0;
int32_t x13500 = 0;
bool x13547 = x13466 > 1;
for(int x13501=0; x13501 < 64; x13501++) {
int32_t x13502 = x13499;
int32_t x13503 = x13500;
int32_t x13504 = x13498;
int32_t x13505 = x13504;
int32_t x13506 = x13502;
int32_t x13507 = x13503;
for(int x13509=0; x13509 < x13482; x13509++) {
int32_t x13510 = x13506;
int32_t x13511 = x13507;
int32_t x13512 = x13505;
int32_t x13513 = x13512;
int32_t x13514 = x13510;
int32_t x13515 = x13511;
for(int x13517=0; x13517 < x13484; x13517++) {
int32_t x13518 = x13514;
int32_t x13519 = x13515;
int32_t x13520 = x13513;
int32_t x13521 = x13520;
int32_t x13522 = x13518;
int32_t x13523 = x13519;
for(int x13524=0; x13524 < x13484; x13524++) {
int32_t x13525 = x13521;
int32_t x13526 = x13522;
float x13527 = x13350[x13526];
int32_t x13528 = x13523;
float x13529 = x268[x13528];
float x13530 = x13527 - x13529;
x13497[x13525] = x13530;
x13521 += 1;
if (x13533) {
x13522 += 1;
} else {
}

}
x13513 += x13484;
if (x13533) {
x13514 += x13344;
} else {
}

}
x13505 += x13485;
x13506 += x13345;
if (x13547) {
x13507 += 1;
} else {
}

}
x13498 += x13486;
x13499 += x13346;

}
float* x13557 = (float*)myMalloc(128 * sizeof(float));;
for(int x13558=0; x13558 < 128; x13558++) {
float x13559 = x148[x13558];
float x13560 = x13559 + 1.0E-5f;
x13557[x13558] = x13560;

}
float* x13564 = (float*)myMalloc(128 * sizeof(float));;
for(int x13565=0; x13565 < 128; x13565++) {
float x13566 = x13557[x13565];
double x13567 = (double)x13566;
double x13568 = sqrt(x13567);
float x13569 = (float)x13568;
x13564[x13565] = x13569;

}
int32_t x13573 = 0;
int32_t x13574 = 1;
x13574 *= 1;
x13573 += 1;
x13574 *= 1;
x13574 *= 1;
int32_t x13579 = x13573;
bool x13580 = x13579 >= 2;
if (x13580) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13585 = x13579 == 0;
if (x13585) {
int32_t x13586 = x13574;
bool x13587 = x13586 == 128;
if (x13587) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13594 = x13574;
int32_t x13595 = 128 / x13594;
bool x13601;
if (x452) {
bool x13596 = x13482 == 1;
bool x13597 = x13595 == 1;
bool x13598 = x13596 || x13597;
bool x13599 = x13482 == x13595;
bool x13600 = x13598 || x13599;
x13601 = x13600;
} else {
x13601 = false;
}
bool x13605;
if (x13601) {
x13605 = x13604;
} else {
x13605 = false;
}
bool x13606;
if (x13605) {
x13606 = x13604;
} else {
x13606 = false;
}
if (x13606) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13482,x13484,x13484,1,x13595,1,1);
assert(false && "");
}
bool x13612 = x13482 <= x13595;
int32_t x13613;
if (x13612) {
x13613 = x13595;
} else {
x13613 = x13482;
}
bool x13619 = x13613 > 0;
bool x13621;
if (x13619) {
x13621 = x13620;
} else {
x13621 = false;
}
bool x13622;
if (x13621) {
x13622 = x13620;
} else {
x13622 = false;
}
if (x13622) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(13482) x Sym(13484) x Sym(13484)"," x Const(1) x Sym(13595) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x13617 = x13613 * x13616;
int32_t x13618 = 64 * x13617;
float* x13628 = (float*)myMalloc(x13618 * sizeof(float));;
int32_t x13629 = 0;
int32_t x13630 = 0;
int32_t x13631 = 0;
bool x13677 = x13482 > 1;
bool x13681 = x13595 > 1;
for(int x13632=0; x13632 < 64; x13632++) {
int32_t x13633 = x13630;
int32_t x13634 = x13631;
int32_t x13635 = x13629;
int32_t x13636 = x13635;
int32_t x13637 = x13633;
int32_t x13638 = x13634;
for(int x13640=0; x13640 < x13613; x13640++) {
int32_t x13641 = x13637;
int32_t x13642 = x13638;
int32_t x13643 = x13636;
int32_t x13644 = x13643;
int32_t x13645 = x13641;
int32_t x13646 = x13642;
for(int x13648=0; x13648 < x13615; x13648++) {
int32_t x13649 = x13645;
int32_t x13650 = x13646;
int32_t x13651 = x13644;
int32_t x13652 = x13651;
int32_t x13653 = x13649;
int32_t x13654 = x13650;
for(int x13655=0; x13655 < x13615; x13655++) {
int32_t x13656 = x13652;
int32_t x13657 = x13653;
float x13658 = x13497[x13657];
int32_t x13659 = x13654;
float x13660 = x13564[x13659];
float x13661 = x13658 / x13660;
x13628[x13656] = x13661;
x13652 += 1;
if (x13664) {
x13653 += 1;
} else {
}

}
x13644 += x13615;
if (x13664) {
x13645 += x13484;
} else {
}

}
x13636 += x13616;
if (x13677) {
x13637 += x13485;
} else {
}
if (x13681) {
x13638 += 1;
} else {
}

}
x13629 += x13617;
x13630 += x13486;

}
int32_t x13691 = 0;
int32_t x13692 = 1;
x13692 *= 1;
x13691 += 1;
x13692 *= 1;
x13692 *= 1;
int32_t x13697 = x13691;
bool x13698 = x13697 >= 2;
if (x13698) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13703 = x13697 == 0;
if (x13703) {
int32_t x13704 = x13692;
bool x13705 = x13704 == 128;
if (x13705) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13712 = x13692;
int32_t x13713 = 128 / x13712;
bool x13719;
if (x452) {
bool x13714 = x13613 == 1;
bool x13715 = x13713 == 1;
bool x13716 = x13714 || x13715;
bool x13717 = x13613 == x13713;
bool x13718 = x13716 || x13717;
x13719 = x13718;
} else {
x13719 = false;
}
bool x13723;
if (x13719) {
x13723 = x13722;
} else {
x13723 = false;
}
bool x13724;
if (x13723) {
x13724 = x13722;
} else {
x13724 = false;
}
if (x13724) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13613,x13615,x13615,1,x13713,1,1);
assert(false && "");
}
bool x13730 = x13613 <= x13713;
int32_t x13731;
if (x13730) {
x13731 = x13713;
} else {
x13731 = x13613;
}
bool x13737 = x13731 > 0;
bool x13739;
if (x13737) {
x13739 = x13738;
} else {
x13739 = false;
}
bool x13740;
if (x13739) {
x13740 = x13738;
} else {
x13740 = false;
}
if (x13740) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(13613) x Sym(13615) x Sym(13615)"," x Const(1) x Sym(13713) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x13735 = x13731 * x13734;
int32_t x13736 = 64 * x13735;
float* x13746 = (float*)myMalloc(x13736 * sizeof(float));;
int32_t x13747 = 0;
int32_t x13748 = 0;
int32_t x13749 = 0;
bool x13795 = x13613 > 1;
bool x13799 = x13713 > 1;
for(int x13750=0; x13750 < 64; x13750++) {
int32_t x13751 = x13748;
int32_t x13752 = x13749;
int32_t x13753 = x13747;
int32_t x13754 = x13753;
int32_t x13755 = x13751;
int32_t x13756 = x13752;
for(int x13758=0; x13758 < x13731; x13758++) {
int32_t x13759 = x13755;
int32_t x13760 = x13756;
int32_t x13761 = x13754;
int32_t x13762 = x13761;
int32_t x13763 = x13759;
int32_t x13764 = x13760;
for(int x13766=0; x13766 < x13733; x13766++) {
int32_t x13767 = x13763;
int32_t x13768 = x13764;
int32_t x13769 = x13762;
int32_t x13770 = x13769;
int32_t x13771 = x13767;
int32_t x13772 = x13768;
for(int x13773=0; x13773 < x13733; x13773++) {
int32_t x13774 = x13770;
int32_t x13775 = x13771;
float x13776 = x13628[x13775];
int32_t x13777 = x13772;
float x13778 = x79[x13777];
float x13779 = x13776 * x13778;
x13746[x13774] = x13779;
x13770 += 1;
if (x13782) {
x13771 += 1;
} else {
}

}
x13762 += x13733;
if (x13782) {
x13763 += x13615;
} else {
}

}
x13754 += x13734;
if (x13795) {
x13755 += x13616;
} else {
}
if (x13799) {
x13756 += 1;
} else {
}

}
x13747 += x13735;
x13748 += x13617;

}
int32_t x13809 = 0;
int32_t x13810 = 1;
x13810 *= 1;
x13809 += 1;
x13810 *= 1;
x13810 *= 1;
int32_t x13815 = x13809;
bool x13816 = x13815 >= 2;
if (x13816) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x13821 = x13815 == 0;
if (x13821) {
int32_t x13822 = x13810;
bool x13823 = x13822 == 128;
if (x13823) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x13830 = x13810;
int32_t x13831 = 128 / x13830;
bool x13837;
if (x452) {
bool x13832 = x13731 == 1;
bool x13833 = x13831 == 1;
bool x13834 = x13832 || x13833;
bool x13835 = x13731 == x13831;
bool x13836 = x13834 || x13835;
x13837 = x13836;
} else {
x13837 = false;
}
bool x13841;
if (x13837) {
x13841 = x13840;
} else {
x13841 = false;
}
bool x13842;
if (x13841) {
x13842 = x13840;
} else {
x13842 = false;
}
if (x13842) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x13731,x13733,x13733,1,x13831,1,1);
assert(false && "");
}
bool x13848 = x13731 <= x13831;
int32_t x13849;
if (x13848) {
x13849 = x13831;
} else {
x13849 = x13731;
}
bool x13855 = x13849 > 0;
bool x13857;
if (x13855) {
x13857 = x13856;
} else {
x13857 = false;
}
bool x13858;
if (x13857) {
x13858 = x13856;
} else {
x13858 = false;
}
if (x13858) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(13731) x Sym(13733) x Sym(13733)"," x Const(1) x Sym(13831) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x13853 = x13849 * x13852;
int32_t x13854 = 64 * x13853;
float* x13864 = (float*)myMalloc(x13854 * sizeof(float));;
int32_t x13865 = 0;
int32_t x13866 = 0;
int32_t x13867 = 0;
bool x13913 = x13731 > 1;
bool x13917 = x13831 > 1;
for(int x13868=0; x13868 < 64; x13868++) {
int32_t x13869 = x13866;
int32_t x13870 = x13867;
int32_t x13871 = x13865;
int32_t x13872 = x13871;
int32_t x13873 = x13869;
int32_t x13874 = x13870;
for(int x13876=0; x13876 < x13849; x13876++) {
int32_t x13877 = x13873;
int32_t x13878 = x13874;
int32_t x13879 = x13872;
int32_t x13880 = x13879;
int32_t x13881 = x13877;
int32_t x13882 = x13878;
for(int x13884=0; x13884 < x13851; x13884++) {
int32_t x13885 = x13881;
int32_t x13886 = x13882;
int32_t x13887 = x13880;
int32_t x13888 = x13887;
int32_t x13889 = x13885;
int32_t x13890 = x13886;
for(int x13891=0; x13891 < x13851; x13891++) {
int32_t x13892 = x13888;
int32_t x13893 = x13889;
float x13894 = x13746[x13893];
int32_t x13895 = x13890;
float x13896 = x38[x13895];
float x13897 = x13894 + x13896;
x13864[x13892] = x13897;
x13888 += 1;
if (x13900) {
x13889 += 1;
} else {
}

}
x13880 += x13851;
if (x13900) {
x13881 += x13733;
} else {
}

}
x13872 += x13852;
if (x13913) {
x13873 += x13734;
} else {
}
if (x13917) {
x13874 += 1;
} else {
}

}
x13865 += x13853;
x13866 += x13735;

}
float* x13927 = (float*)myMalloc(x13854 * sizeof(float));;
for(int x13929=0; x13929 < x13854; x13929++) {
float x13930 = x13864[x13929];
bool x13931 = x13930 < 0.0f;
if (x13931) {
x13927[x13929] = 0.0f;
} else {
float x13934 = x13864[x13929];
x13927[x13929] = x13934;
}

}
float* x13948 = (float*)myMalloc(x13947 * sizeof(float));;
int32_t x13951 = 64 * x13849;
int32_t x13952 = x13951 * x13943;
float* x13953 = (float*)myMalloc(x13952 * sizeof(float));;
int32_t x13949 = x13849 * x13943;
for(int x13954=0; x13954 < 64; x13954++) {
int32_t x13955 = x13954 * x13853;
float* x13956 = x13927+x13955;
int32_t x13957 = x13954 * x13944;
float* x13958 = x13948+x13957;
int32_t x13959 = x13954 * x13949;
float* x13960 = x13953+x13959;
for(int x13961=0; x13961 < x13849; x13961++) {
int32_t x13962 = x13961 / 1;
int32_t x13966 = x13962 * x13942;
int32_t x13967 = x13966 * x13942;
int32_t x13963 = x13961 % 1;
int32_t x13964 = x13963 / 1;
int32_t x13968 = x13964 * x13942;
int32_t x13969 = x13968 * x13942;
int32_t x13970 = x13967 + x13969;
int32_t x13965 = x13963 % 1;
int32_t x13971 = x13965 * x13942;
int32_t x13972 = x13971 * x13942;
int32_t x13973 = x13970 + x13972;
float* x13974 = x13960+x13973;
int32_t x13975 = x13962 * x13851;
int32_t x13976 = x13975 * x13851;
float* x13977 = x13956+x13976;
for(int x13979=0; x13979 < x13942; x13979++) {
int32_t x13981 = x13979 * x13942;
float* x13982 = x13974+x13981;
int32_t x13980 = x13979 + x13964;
int32_t x13983 = x13980 * x13851;
int32_t x13984 = x13983 + x13965;
float* x13985 = x13977+x13984;
memcpy(x13982, x13985, 4 * x13942);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x13943,x13849,1,x55,x13849,x13960,x13943,1,x13958,x13943);

}
int32_t x13994 = 0;
int32_t x13995 = 1;
x13995 *= 1;
x13994 += 1;
x13995 *= 1;
x13995 *= 1;
int32_t x14000 = x13994;
bool x14001 = x14000 >= 2;
if (x14001) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14006 = x14000 == 0;
if (x14006) {
int32_t x14007 = x13995;
bool x14008 = x14007 == 512;
if (x14008) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14015 = x13995;
int32_t x14016 = 512 / x14015;
bool x14020;
if (x452) {
bool x14017 = x14016 == 1;
bool x14018 = 512 == x14016;
bool x14019 = x14017 || x14018;
x14020 = x14019;
} else {
x14020 = false;
}
bool x14024;
if (x14020) {
x14024 = x14023;
} else {
x14024 = false;
}
bool x14025;
if (x14024) {
x14025 = x14023;
} else {
x14025 = false;
}
if (x14025) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x13942,x13942,1,x14016,1,1);
assert(false && "");
}
bool x14031 = 512 <= x14016;
int32_t x14032;
if (x14031) {
x14032 = x14016;
} else {
x14032 = 512;
}
bool x14038 = x14032 > 0;
bool x14040;
if (x14038) {
x14040 = x14039;
} else {
x14040 = false;
}
bool x14041;
if (x14040) {
x14041 = x14039;
} else {
x14041 = false;
}
if (x14041) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(13942) x Sym(13942)"," x Const(1) x Sym(14016) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x14036 = x14032 * x14035;
int32_t x14037 = 64 * x14036;
float* x14047 = (float*)myMalloc(x14037 * sizeof(float));;
int32_t x14048 = 0;
int32_t x14049 = 0;
int32_t x14050 = 0;
bool x14097 = x14016 > 1;
for(int x14051=0; x14051 < 64; x14051++) {
int32_t x14052 = x14049;
int32_t x14053 = x14050;
int32_t x14054 = x14048;
int32_t x14055 = x14054;
int32_t x14056 = x14052;
int32_t x14057 = x14053;
for(int x14059=0; x14059 < x14032; x14059++) {
int32_t x14060 = x14056;
int32_t x14061 = x14057;
int32_t x14062 = x14055;
int32_t x14063 = x14062;
int32_t x14064 = x14060;
int32_t x14065 = x14061;
for(int x14067=0; x14067 < x14034; x14067++) {
int32_t x14068 = x14064;
int32_t x14069 = x14065;
int32_t x14070 = x14063;
int32_t x14071 = x14070;
int32_t x14072 = x14068;
int32_t x14073 = x14069;
for(int x14074=0; x14074 < x14034; x14074++) {
int32_t x14075 = x14071;
int32_t x14076 = x14072;
float x14077 = x13948[x14076];
int32_t x14078 = x14073;
float x14079 = x19[x14078];
float x14080 = x14077 - x14079;
x14047[x14075] = x14080;
x14071 += 1;
if (x14083) {
x14072 += 1;
} else {
}

}
x14063 += x14034;
if (x14083) {
x14064 += x13942;
} else {
}

}
x14055 += x14035;
x14056 += x13943;
if (x14097) {
x14057 += 1;
} else {
}

}
x14048 += x14036;
x14049 += x13944;

}
float* x14107 = (float*)myMalloc(512 * sizeof(float));;
for(int x14108=0; x14108 < 512; x14108++) {
float x14109 = x234[x14108];
float x14110 = x14109 + 1.0E-5f;
x14107[x14108] = x14110;

}
float* x14114 = (float*)myMalloc(512 * sizeof(float));;
for(int x14115=0; x14115 < 512; x14115++) {
float x14116 = x14107[x14115];
double x14117 = (double)x14116;
double x14118 = sqrt(x14117);
float x14119 = (float)x14118;
x14114[x14115] = x14119;

}
int32_t x14123 = 0;
int32_t x14124 = 1;
x14124 *= 1;
x14123 += 1;
x14124 *= 1;
x14124 *= 1;
int32_t x14129 = x14123;
bool x14130 = x14129 >= 2;
if (x14130) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14135 = x14129 == 0;
if (x14135) {
int32_t x14136 = x14124;
bool x14137 = x14136 == 512;
if (x14137) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14144 = x14124;
int32_t x14145 = 512 / x14144;
bool x14151;
if (x452) {
bool x14146 = x14032 == 1;
bool x14147 = x14145 == 1;
bool x14148 = x14146 || x14147;
bool x14149 = x14032 == x14145;
bool x14150 = x14148 || x14149;
x14151 = x14150;
} else {
x14151 = false;
}
bool x14155;
if (x14151) {
x14155 = x14154;
} else {
x14155 = false;
}
bool x14156;
if (x14155) {
x14156 = x14154;
} else {
x14156 = false;
}
if (x14156) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14032,x14034,x14034,1,x14145,1,1);
assert(false && "");
}
bool x14162 = x14032 <= x14145;
int32_t x14163;
if (x14162) {
x14163 = x14145;
} else {
x14163 = x14032;
}
bool x14169 = x14163 > 0;
bool x14171;
if (x14169) {
x14171 = x14170;
} else {
x14171 = false;
}
bool x14172;
if (x14171) {
x14172 = x14170;
} else {
x14172 = false;
}
if (x14172) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(14032) x Sym(14034) x Sym(14034)"," x Const(1) x Sym(14145) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x14167 = x14163 * x14166;
int32_t x14168 = 64 * x14167;
float* x14178 = (float*)myMalloc(x14168 * sizeof(float));;
int32_t x14179 = 0;
int32_t x14180 = 0;
int32_t x14181 = 0;
bool x14227 = x14032 > 1;
bool x14231 = x14145 > 1;
for(int x14182=0; x14182 < 64; x14182++) {
int32_t x14183 = x14180;
int32_t x14184 = x14181;
int32_t x14185 = x14179;
int32_t x14186 = x14185;
int32_t x14187 = x14183;
int32_t x14188 = x14184;
for(int x14190=0; x14190 < x14163; x14190++) {
int32_t x14191 = x14187;
int32_t x14192 = x14188;
int32_t x14193 = x14186;
int32_t x14194 = x14193;
int32_t x14195 = x14191;
int32_t x14196 = x14192;
for(int x14198=0; x14198 < x14165; x14198++) {
int32_t x14199 = x14195;
int32_t x14200 = x14196;
int32_t x14201 = x14194;
int32_t x14202 = x14201;
int32_t x14203 = x14199;
int32_t x14204 = x14200;
for(int x14205=0; x14205 < x14165; x14205++) {
int32_t x14206 = x14202;
int32_t x14207 = x14203;
float x14208 = x14047[x14207];
int32_t x14209 = x14204;
float x14210 = x14114[x14209];
float x14211 = x14208 / x14210;
x14178[x14206] = x14211;
x14202 += 1;
if (x14214) {
x14203 += 1;
} else {
}

}
x14194 += x14165;
if (x14214) {
x14195 += x14034;
} else {
}

}
x14186 += x14166;
if (x14227) {
x14187 += x14035;
} else {
}
if (x14231) {
x14188 += 1;
} else {
}

}
x14179 += x14167;
x14180 += x14036;

}
int32_t x14241 = 0;
int32_t x14242 = 1;
x14242 *= 1;
x14241 += 1;
x14242 *= 1;
x14242 *= 1;
int32_t x14247 = x14241;
bool x14248 = x14247 >= 2;
if (x14248) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14253 = x14247 == 0;
if (x14253) {
int32_t x14254 = x14242;
bool x14255 = x14254 == 512;
if (x14255) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14262 = x14242;
int32_t x14263 = 512 / x14262;
bool x14269;
if (x452) {
bool x14264 = x14163 == 1;
bool x14265 = x14263 == 1;
bool x14266 = x14264 || x14265;
bool x14267 = x14163 == x14263;
bool x14268 = x14266 || x14267;
x14269 = x14268;
} else {
x14269 = false;
}
bool x14273;
if (x14269) {
x14273 = x14272;
} else {
x14273 = false;
}
bool x14274;
if (x14273) {
x14274 = x14272;
} else {
x14274 = false;
}
if (x14274) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14163,x14165,x14165,1,x14263,1,1);
assert(false && "");
}
bool x14280 = x14163 <= x14263;
int32_t x14281;
if (x14280) {
x14281 = x14263;
} else {
x14281 = x14163;
}
bool x14287 = x14281 > 0;
bool x14289;
if (x14287) {
x14289 = x14288;
} else {
x14289 = false;
}
bool x14290;
if (x14289) {
x14290 = x14288;
} else {
x14290 = false;
}
if (x14290) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(14163) x Sym(14165) x Sym(14165)"," x Const(1) x Sym(14263) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x14285 = x14281 * x14284;
int32_t x14286 = 64 * x14285;
float* x14296 = (float*)myMalloc(x14286 * sizeof(float));;
int32_t x14297 = 0;
int32_t x14298 = 0;
int32_t x14299 = 0;
bool x14345 = x14163 > 1;
bool x14349 = x14263 > 1;
for(int x14300=0; x14300 < 64; x14300++) {
int32_t x14301 = x14298;
int32_t x14302 = x14299;
int32_t x14303 = x14297;
int32_t x14304 = x14303;
int32_t x14305 = x14301;
int32_t x14306 = x14302;
for(int x14308=0; x14308 < x14281; x14308++) {
int32_t x14309 = x14305;
int32_t x14310 = x14306;
int32_t x14311 = x14304;
int32_t x14312 = x14311;
int32_t x14313 = x14309;
int32_t x14314 = x14310;
for(int x14316=0; x14316 < x14283; x14316++) {
int32_t x14317 = x14313;
int32_t x14318 = x14314;
int32_t x14319 = x14312;
int32_t x14320 = x14319;
int32_t x14321 = x14317;
int32_t x14322 = x14318;
for(int x14323=0; x14323 < x14283; x14323++) {
int32_t x14324 = x14320;
int32_t x14325 = x14321;
float x14326 = x14178[x14325];
int32_t x14327 = x14322;
float x14328 = x156[x14327];
float x14329 = x14326 * x14328;
x14296[x14324] = x14329;
x14320 += 1;
if (x14332) {
x14321 += 1;
} else {
}

}
x14312 += x14283;
if (x14332) {
x14313 += x14165;
} else {
}

}
x14304 += x14284;
if (x14345) {
x14305 += x14166;
} else {
}
if (x14349) {
x14306 += 1;
} else {
}

}
x14297 += x14285;
x14298 += x14167;

}
int32_t x14359 = 0;
int32_t x14360 = 1;
x14360 *= 1;
x14359 += 1;
x14360 *= 1;
x14360 *= 1;
int32_t x14365 = x14359;
bool x14366 = x14365 >= 2;
if (x14366) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14371 = x14365 == 0;
if (x14371) {
int32_t x14372 = x14360;
bool x14373 = x14372 == 512;
if (x14373) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14380 = x14360;
int32_t x14381 = 512 / x14380;
bool x14387;
if (x452) {
bool x14382 = x14281 == 1;
bool x14383 = x14381 == 1;
bool x14384 = x14382 || x14383;
bool x14385 = x14281 == x14381;
bool x14386 = x14384 || x14385;
x14387 = x14386;
} else {
x14387 = false;
}
bool x14391;
if (x14387) {
x14391 = x14390;
} else {
x14391 = false;
}
bool x14392;
if (x14391) {
x14392 = x14390;
} else {
x14392 = false;
}
if (x14392) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14281,x14283,x14283,1,x14381,1,1);
assert(false && "");
}
bool x14398 = x14281 <= x14381;
int32_t x14399;
if (x14398) {
x14399 = x14381;
} else {
x14399 = x14281;
}
bool x14405 = x14399 > 0;
bool x14407;
if (x14405) {
x14407 = x14406;
} else {
x14407 = false;
}
bool x14408;
if (x14407) {
x14408 = x14406;
} else {
x14408 = false;
}
if (x14408) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(14281) x Sym(14283) x Sym(14283)"," x Const(1) x Sym(14381) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x14403 = x14399 * x14402;
int32_t x14404 = 64 * x14403;
float* x14414 = (float*)myMalloc(x14404 * sizeof(float));;
int32_t x14415 = 0;
int32_t x14416 = 0;
int32_t x14417 = 0;
bool x14463 = x14281 > 1;
bool x14467 = x14381 > 1;
for(int x14418=0; x14418 < 64; x14418++) {
int32_t x14419 = x14416;
int32_t x14420 = x14417;
int32_t x14421 = x14415;
int32_t x14422 = x14421;
int32_t x14423 = x14419;
int32_t x14424 = x14420;
for(int x14426=0; x14426 < x14399; x14426++) {
int32_t x14427 = x14423;
int32_t x14428 = x14424;
int32_t x14429 = x14422;
int32_t x14430 = x14429;
int32_t x14431 = x14427;
int32_t x14432 = x14428;
for(int x14434=0; x14434 < x14401; x14434++) {
int32_t x14435 = x14431;
int32_t x14436 = x14432;
int32_t x14437 = x14430;
int32_t x14438 = x14437;
int32_t x14439 = x14435;
int32_t x14440 = x14436;
for(int x14441=0; x14441 < x14401; x14441++) {
int32_t x14442 = x14438;
int32_t x14443 = x14439;
float x14444 = x14296[x14443];
int32_t x14445 = x14440;
float x14446 = x54[x14445];
float x14447 = x14444 + x14446;
x14414[x14442] = x14447;
x14438 += 1;
if (x14450) {
x14439 += 1;
} else {
}

}
x14430 += x14401;
if (x14450) {
x14431 += x14283;
} else {
}

}
x14422 += x14402;
if (x14463) {
x14423 += x14284;
} else {
}
if (x14467) {
x14424 += 1;
} else {
}

}
x14415 += x14403;
x14416 += x14285;

}
bool x14477 = x14399 == 1;
bool x14478 = x14477 || x12689;
bool x14479 = x14399 == x12611;
bool x14480 = x14478 || x14479;
bool x14485;
if (x14480) {
x14485 = x14484;
} else {
x14485 = false;
}
bool x14486;
if (x14485) {
x14486 = x14484;
} else {
x14486 = false;
}
if (x14486) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14399,x14401,x14401,64,x12611,x12613,x12613);
assert(false && "");
}
int32_t x14499 = 0;
int32_t x14500 = 0;
int32_t x14501 = 0;
bool x14492 = x14399 <= x12611;
int32_t x14493;
if (x14492) {
x14493 = x12611;
} else {
x14493 = x14399;
}
bool x14552 = x14399 > 1;
int32_t x14497 = x14493 * x14496;
for(int x14502=0; x14502 < 64; x14502++) {
int32_t x14503 = x14500;
int32_t x14504 = x14501;
int32_t x14505 = x14499;
int32_t x14506 = x14505;
int32_t x14507 = x14503;
int32_t x14508 = x14504;
for(int x14510=0; x14510 < x14493; x14510++) {
int32_t x14511 = x14507;
int32_t x14512 = x14508;
int32_t x14513 = x14506;
int32_t x14514 = x14513;
int32_t x14515 = x14511;
int32_t x14516 = x14512;
for(int x14518=0; x14518 < x14495; x14518++) {
int32_t x14519 = x14515;
int32_t x14520 = x14516;
int32_t x14521 = x14514;
int32_t x14522 = x14521;
int32_t x14523 = x14519;
int32_t x14524 = x14520;
for(int x14525=0; x14525 < x14495; x14525++) {
int32_t x14526 = x14523;
float x14527 = x14414[x14526];
int32_t x14528 = x14524;
float x14529 = x12778[x14528];
float x14530 = x14527 + x14529;
x14414[x14526] = x14530;
x14522 += 1;
if (x14533) {
x14523 += 1;
} else {
}
if (x12745) {
x14524 += 1;
} else {
}

}
x14514 += x14495;
if (x14533) {
x14515 += x14401;
} else {
}
if (x12745) {
x14516 += x12613;
} else {
}

}
x14506 += x14496;
if (x14552) {
x14507 += x14402;
} else {
}
if (x12764) {
x14508 += x12614;
} else {
}

}
x14499 += x14497;
x14500 += x14403;
x14501 += x12615;

}
float* x14566 = (float*)myMalloc(x14404 * sizeof(float));;
for(int x14568=0; x14568 < x14404; x14568++) {
float x14569 = x14414[x14568];
bool x14570 = x14569 < 0.0f;
if (x14570) {
x14566[x14568] = 0.0f;
} else {
float x14573 = x14414[x14568];
x14566[x14568] = x14573;
}

}
float* x14587 = (float*)myMalloc(x14586 * sizeof(float));;
int32_t x14590 = 64 * x14399;
int32_t x14591 = x14590 * x14582;
float* x14592 = (float*)myMalloc(x14591 * sizeof(float));;
int32_t x14588 = x14399 * x14582;
for(int x14593=0; x14593 < 64; x14593++) {
int32_t x14594 = x14593 * x14403;
float* x14595 = x14566+x14594;
int32_t x14596 = x14593 * x14583;
float* x14597 = x14587+x14596;
int32_t x14598 = x14593 * x14588;
float* x14599 = x14592+x14598;
for(int x14600=0; x14600 < x14399; x14600++) {
int32_t x14601 = x14600 / 1;
int32_t x14605 = x14601 * x14581;
int32_t x14606 = x14605 * x14581;
int32_t x14602 = x14600 % 1;
int32_t x14603 = x14602 / 1;
int32_t x14607 = x14603 * x14581;
int32_t x14608 = x14607 * x14581;
int32_t x14609 = x14606 + x14608;
int32_t x14604 = x14602 % 1;
int32_t x14610 = x14604 * x14581;
int32_t x14611 = x14610 * x14581;
int32_t x14612 = x14609 + x14611;
float* x14613 = x14599+x14612;
int32_t x14614 = x14601 * x14401;
int32_t x14615 = x14614 * x14401;
float* x14616 = x14595+x14615;
for(int x14618=0; x14618 < x14581; x14618++) {
int32_t x14620 = x14618 * x14581;
float* x14621 = x14613+x14620;
int32_t x14619 = x14618 + x14603;
int32_t x14622 = x14619 * x14401;
int32_t x14623 = x14622 + x14604;
float* x14624 = x14616+x14623;
memcpy(x14621, x14624, 4 * x14581);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x14582,x14399,1,x180,x14399,x14599,x14582,1,x14597,x14582);

}
int32_t x14633 = 0;
int32_t x14634 = 1;
x14634 *= 1;
x14633 += 1;
x14634 *= 1;
x14634 *= 1;
int32_t x14639 = x14633;
bool x14640 = x14639 >= 2;
if (x14640) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14645 = x14639 == 0;
if (x14645) {
int32_t x14646 = x14634;
bool x14647 = x14646 == 256;
if (x14647) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14654 = x14634;
int32_t x14655 = 256 / x14654;
bool x14659;
if (x452) {
bool x14656 = x14655 == 1;
bool x14657 = 256 == x14655;
bool x14658 = x14656 || x14657;
x14659 = x14658;
} else {
x14659 = false;
}
bool x14663;
if (x14659) {
x14663 = x14662;
} else {
x14663 = false;
}
bool x14664;
if (x14663) {
x14664 = x14662;
} else {
x14664 = false;
}
if (x14664) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x14581,x14581,1,x14655,1,1);
assert(false && "");
}
bool x14670 = 256 <= x14655;
int32_t x14671;
if (x14670) {
x14671 = x14655;
} else {
x14671 = 256;
}
bool x14677 = x14671 > 0;
bool x14679;
if (x14677) {
x14679 = x14678;
} else {
x14679 = false;
}
bool x14680;
if (x14679) {
x14680 = x14678;
} else {
x14680 = false;
}
if (x14680) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(14581) x Sym(14581)"," x Const(1) x Sym(14655) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x14675 = x14671 * x14674;
int32_t x14676 = 64 * x14675;
float* x14686 = (float*)myMalloc(x14676 * sizeof(float));;
int32_t x14687 = 0;
int32_t x14688 = 0;
int32_t x14689 = 0;
bool x14736 = x14655 > 1;
for(int x14690=0; x14690 < 64; x14690++) {
int32_t x14691 = x14688;
int32_t x14692 = x14689;
int32_t x14693 = x14687;
int32_t x14694 = x14693;
int32_t x14695 = x14691;
int32_t x14696 = x14692;
for(int x14698=0; x14698 < x14671; x14698++) {
int32_t x14699 = x14695;
int32_t x14700 = x14696;
int32_t x14701 = x14694;
int32_t x14702 = x14701;
int32_t x14703 = x14699;
int32_t x14704 = x14700;
for(int x14706=0; x14706 < x14673; x14706++) {
int32_t x14707 = x14703;
int32_t x14708 = x14704;
int32_t x14709 = x14702;
int32_t x14710 = x14709;
int32_t x14711 = x14707;
int32_t x14712 = x14708;
for(int x14713=0; x14713 < x14673; x14713++) {
int32_t x14714 = x14710;
int32_t x14715 = x14711;
float x14716 = x14587[x14715];
int32_t x14717 = x14712;
float x14718 = x131[x14717];
float x14719 = x14716 - x14718;
x14686[x14714] = x14719;
x14710 += 1;
if (x14722) {
x14711 += 1;
} else {
}

}
x14702 += x14673;
if (x14722) {
x14703 += x14581;
} else {
}

}
x14694 += x14674;
x14695 += x14582;
if (x14736) {
x14696 += 1;
} else {
}

}
x14687 += x14675;
x14688 += x14583;

}
float* x14746 = (float*)myMalloc(256 * sizeof(float));;
for(int x14747=0; x14747 < 256; x14747++) {
float x14748 = x198[x14747];
float x14749 = x14748 + 1.0E-5f;
x14746[x14747] = x14749;

}
float* x14753 = (float*)myMalloc(256 * sizeof(float));;
for(int x14754=0; x14754 < 256; x14754++) {
float x14755 = x14746[x14754];
double x14756 = (double)x14755;
double x14757 = sqrt(x14756);
float x14758 = (float)x14757;
x14753[x14754] = x14758;

}
int32_t x14762 = 0;
int32_t x14763 = 1;
x14763 *= 1;
x14762 += 1;
x14763 *= 1;
x14763 *= 1;
int32_t x14768 = x14762;
bool x14769 = x14768 >= 2;
if (x14769) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14774 = x14768 == 0;
if (x14774) {
int32_t x14775 = x14763;
bool x14776 = x14775 == 256;
if (x14776) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14783 = x14763;
int32_t x14784 = 256 / x14783;
bool x14790;
if (x452) {
bool x14785 = x14671 == 1;
bool x14786 = x14784 == 1;
bool x14787 = x14785 || x14786;
bool x14788 = x14671 == x14784;
bool x14789 = x14787 || x14788;
x14790 = x14789;
} else {
x14790 = false;
}
bool x14794;
if (x14790) {
x14794 = x14793;
} else {
x14794 = false;
}
bool x14795;
if (x14794) {
x14795 = x14793;
} else {
x14795 = false;
}
if (x14795) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14671,x14673,x14673,1,x14784,1,1);
assert(false && "");
}
bool x14801 = x14671 <= x14784;
int32_t x14802;
if (x14801) {
x14802 = x14784;
} else {
x14802 = x14671;
}
bool x14808 = x14802 > 0;
bool x14810;
if (x14808) {
x14810 = x14809;
} else {
x14810 = false;
}
bool x14811;
if (x14810) {
x14811 = x14809;
} else {
x14811 = false;
}
if (x14811) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(14671) x Sym(14673) x Sym(14673)"," x Const(1) x Sym(14784) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x14806 = x14802 * x14805;
int32_t x14807 = 64 * x14806;
float* x14817 = (float*)myMalloc(x14807 * sizeof(float));;
int32_t x14818 = 0;
int32_t x14819 = 0;
int32_t x14820 = 0;
bool x14866 = x14671 > 1;
bool x14870 = x14784 > 1;
for(int x14821=0; x14821 < 64; x14821++) {
int32_t x14822 = x14819;
int32_t x14823 = x14820;
int32_t x14824 = x14818;
int32_t x14825 = x14824;
int32_t x14826 = x14822;
int32_t x14827 = x14823;
for(int x14829=0; x14829 < x14802; x14829++) {
int32_t x14830 = x14826;
int32_t x14831 = x14827;
int32_t x14832 = x14825;
int32_t x14833 = x14832;
int32_t x14834 = x14830;
int32_t x14835 = x14831;
for(int x14837=0; x14837 < x14804; x14837++) {
int32_t x14838 = x14834;
int32_t x14839 = x14835;
int32_t x14840 = x14833;
int32_t x14841 = x14840;
int32_t x14842 = x14838;
int32_t x14843 = x14839;
for(int x14844=0; x14844 < x14804; x14844++) {
int32_t x14845 = x14841;
int32_t x14846 = x14842;
float x14847 = x14686[x14846];
int32_t x14848 = x14843;
float x14849 = x14753[x14848];
float x14850 = x14847 / x14849;
x14817[x14845] = x14850;
x14841 += 1;
if (x14853) {
x14842 += 1;
} else {
}

}
x14833 += x14804;
if (x14853) {
x14834 += x14673;
} else {
}

}
x14825 += x14805;
if (x14866) {
x14826 += x14674;
} else {
}
if (x14870) {
x14827 += 1;
} else {
}

}
x14818 += x14806;
x14819 += x14675;

}
int32_t x14880 = 0;
int32_t x14881 = 1;
x14881 *= 1;
x14880 += 1;
x14881 *= 1;
x14881 *= 1;
int32_t x14886 = x14880;
bool x14887 = x14886 >= 2;
if (x14887) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x14892 = x14886 == 0;
if (x14892) {
int32_t x14893 = x14881;
bool x14894 = x14893 == 256;
if (x14894) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x14901 = x14881;
int32_t x14902 = 256 / x14901;
bool x14908;
if (x452) {
bool x14903 = x14802 == 1;
bool x14904 = x14902 == 1;
bool x14905 = x14903 || x14904;
bool x14906 = x14802 == x14902;
bool x14907 = x14905 || x14906;
x14908 = x14907;
} else {
x14908 = false;
}
bool x14912;
if (x14908) {
x14912 = x14911;
} else {
x14912 = false;
}
bool x14913;
if (x14912) {
x14913 = x14911;
} else {
x14913 = false;
}
if (x14913) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14802,x14804,x14804,1,x14902,1,1);
assert(false && "");
}
bool x14919 = x14802 <= x14902;
int32_t x14920;
if (x14919) {
x14920 = x14902;
} else {
x14920 = x14802;
}
bool x14926 = x14920 > 0;
bool x14928;
if (x14926) {
x14928 = x14927;
} else {
x14928 = false;
}
bool x14929;
if (x14928) {
x14929 = x14927;
} else {
x14929 = false;
}
if (x14929) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(14802) x Sym(14804) x Sym(14804)"," x Const(1) x Sym(14902) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x14924 = x14920 * x14923;
int32_t x14925 = 64 * x14924;
float* x14935 = (float*)myMalloc(x14925 * sizeof(float));;
int32_t x14936 = 0;
int32_t x14937 = 0;
int32_t x14938 = 0;
bool x14984 = x14802 > 1;
bool x14988 = x14902 > 1;
for(int x14939=0; x14939 < 64; x14939++) {
int32_t x14940 = x14937;
int32_t x14941 = x14938;
int32_t x14942 = x14936;
int32_t x14943 = x14942;
int32_t x14944 = x14940;
int32_t x14945 = x14941;
for(int x14947=0; x14947 < x14920; x14947++) {
int32_t x14948 = x14944;
int32_t x14949 = x14945;
int32_t x14950 = x14943;
int32_t x14951 = x14950;
int32_t x14952 = x14948;
int32_t x14953 = x14949;
for(int x14955=0; x14955 < x14922; x14955++) {
int32_t x14956 = x14952;
int32_t x14957 = x14953;
int32_t x14958 = x14951;
int32_t x14959 = x14958;
int32_t x14960 = x14956;
int32_t x14961 = x14957;
for(int x14962=0; x14962 < x14922; x14962++) {
int32_t x14963 = x14959;
int32_t x14964 = x14960;
float x14965 = x14817[x14964];
int32_t x14966 = x14961;
float x14967 = x270[x14966];
float x14968 = x14965 * x14967;
x14935[x14963] = x14968;
x14959 += 1;
if (x14971) {
x14960 += 1;
} else {
}

}
x14951 += x14922;
if (x14971) {
x14952 += x14804;
} else {
}

}
x14943 += x14923;
if (x14984) {
x14944 += x14805;
} else {
}
if (x14988) {
x14945 += 1;
} else {
}

}
x14936 += x14924;
x14937 += x14806;

}
int32_t x14998 = 0;
int32_t x14999 = 1;
x14999 *= 1;
x14998 += 1;
x14999 *= 1;
x14999 *= 1;
int32_t x15004 = x14998;
bool x15005 = x15004 >= 2;
if (x15005) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15010 = x15004 == 0;
if (x15010) {
int32_t x15011 = x14999;
bool x15012 = x15011 == 256;
if (x15012) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15019 = x14999;
int32_t x15020 = 256 / x15019;
bool x15026;
if (x452) {
bool x15021 = x14920 == 1;
bool x15022 = x15020 == 1;
bool x15023 = x15021 || x15022;
bool x15024 = x14920 == x15020;
bool x15025 = x15023 || x15024;
x15026 = x15025;
} else {
x15026 = false;
}
bool x15030;
if (x15026) {
x15030 = x15029;
} else {
x15030 = false;
}
bool x15031;
if (x15030) {
x15031 = x15029;
} else {
x15031 = false;
}
if (x15031) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x14920,x14922,x14922,1,x15020,1,1);
assert(false && "");
}
bool x15037 = x14920 <= x15020;
int32_t x15038;
if (x15037) {
x15038 = x15020;
} else {
x15038 = x14920;
}
bool x15044 = x15038 > 0;
bool x15046;
if (x15044) {
x15046 = x15045;
} else {
x15046 = false;
}
bool x15047;
if (x15046) {
x15047 = x15045;
} else {
x15047 = false;
}
if (x15047) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(14920) x Sym(14922) x Sym(14922)"," x Const(1) x Sym(15020) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x15042 = x15038 * x15041;
int32_t x15043 = 64 * x15042;
float* x15053 = (float*)myMalloc(x15043 * sizeof(float));;
int32_t x15054 = 0;
int32_t x15055 = 0;
int32_t x15056 = 0;
bool x15102 = x14920 > 1;
bool x15106 = x15020 > 1;
for(int x15057=0; x15057 < 64; x15057++) {
int32_t x15058 = x15055;
int32_t x15059 = x15056;
int32_t x15060 = x15054;
int32_t x15061 = x15060;
int32_t x15062 = x15058;
int32_t x15063 = x15059;
for(int x15065=0; x15065 < x15038; x15065++) {
int32_t x15066 = x15062;
int32_t x15067 = x15063;
int32_t x15068 = x15061;
int32_t x15069 = x15068;
int32_t x15070 = x15066;
int32_t x15071 = x15067;
for(int x15073=0; x15073 < x15040; x15073++) {
int32_t x15074 = x15070;
int32_t x15075 = x15071;
int32_t x15076 = x15069;
int32_t x15077 = x15076;
int32_t x15078 = x15074;
int32_t x15079 = x15075;
for(int x15080=0; x15080 < x15040; x15080++) {
int32_t x15081 = x15077;
int32_t x15082 = x15078;
float x15083 = x14935[x15082];
int32_t x15084 = x15079;
float x15085 = x21[x15084];
float x15086 = x15083 + x15085;
x15053[x15081] = x15086;
x15077 += 1;
if (x15089) {
x15078 += 1;
} else {
}

}
x15069 += x15040;
if (x15089) {
x15070 += x14922;
} else {
}

}
x15061 += x15041;
if (x15102) {
x15062 += x14923;
} else {
}
if (x15106) {
x15063 += 1;
} else {
}

}
x15054 += x15042;
x15055 += x14924;

}
float* x15116 = (float*)myMalloc(x15043 * sizeof(float));;
for(int x15118=0; x15118 < x15043; x15118++) {
float x15119 = x15053[x15118];
bool x15120 = x15119 < 0.0f;
if (x15120) {
x15116[x15118] = 0.0f;
} else {
float x15123 = x15053[x15118];
x15116[x15118] = x15123;
}

}
float* x15138 = (float*)myMalloc(x15137 * sizeof(float));;
int32_t x15139 = 9 * x15038;
int32_t x15142 = 64 * x15139;
int32_t x15143 = x15142 * x15133;
float* x15144 = (float*)myMalloc(x15143 * sizeof(float));;
int32_t x15140 = x15139 * x15133;
int32_t x15152 = x15038 * 3;
int32_t x15153 = x15152 * 3;
for(int x15145=0; x15145 < 64; x15145++) {
int32_t x15146 = x15145 * x15042;
float* x15147 = x15116+x15146;
int32_t x15148 = x15145 * x15134;
float* x15149 = x15138+x15148;
int32_t x15150 = x15145 * x15140;
float* x15151 = x15144+x15150;
for(int x15155=0; x15155 < x15153; x15155++) {
int32_t x15156 = x15155 / 9;
int32_t x15160 = x15156 * 3;
int32_t x15161 = x15160 * 3;
int32_t x15162 = x15161 * x15132;
int32_t x15163 = x15162 * x15132;
int32_t x15157 = x15155 % 9;
int32_t x15158 = x15157 / 3;
int32_t x15164 = x15158 * 3;
int32_t x15165 = x15164 * x15132;
int32_t x15166 = x15165 * x15132;
int32_t x15167 = x15163 + x15166;
int32_t x15159 = x15157 % 3;
int32_t x15168 = x15159 * x15132;
int32_t x15169 = x15168 * x15132;
int32_t x15170 = x15167 + x15169;
float* x15171 = x15151+x15170;
int32_t x15172 = x15156 * x15040;
int32_t x15173 = x15172 * x15040;
float* x15174 = x15147+x15173;
for(int x15176=0; x15176 < x15132; x15176++) {
int32_t x15177 = x15176 * 2;
int32_t x15178 = x15177 - 1;
int32_t x15179 = x15178 + x15158;
bool x15180 = x15179 < 0;
bool x15181 = x15179 >= x15040;
bool x15182 = x15180 || x15181;
if (x15182) {
int32_t x15183 = x15176 * x15132;
float* x15184 = x15171+x15183;
memset(x15184, 0, 4 * x15132);;
} else {
int32_t x15183 = x15176 * x15132;
int32_t x15199 = x15179 * x15040;
for(int x15187=0; x15187 < x15132; x15187++) {
int32_t x15188 = x15187 * 2;
int32_t x15189 = x15188 - 1;
int32_t x15190 = x15189 + x15159;
bool x15191 = x15190 < 0;
bool x15192 = x15190 >= x15040;
bool x15193 = x15191 || x15192;
if (x15193) {
int32_t x15194 = x15183 + x15187;
float* x15195 = x15171+x15194;
memset(x15195, 0, 4 * 1);;
} else {
int32_t x15194 = x15183 + x15187;
float* x15198 = x15171+x15194;
int32_t x15200 = x15199 + x15190;
float* x15201 = x15174+x15200;
memcpy(x15198, x15201, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x15133,x15139,1,x175,x15139,x15151,x15133,1,x15149,x15133);

}
int32_t x15216 = 0;
int32_t x15217 = 1;
x15217 *= 1;
x15216 += 1;
x15217 *= 1;
x15217 *= 1;
int32_t x15222 = x15216;
bool x15223 = x15222 >= 2;
if (x15223) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15228 = x15222 == 0;
if (x15228) {
int32_t x15229 = x15217;
bool x15230 = x15229 == 256;
if (x15230) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15237 = x15217;
int32_t x15238 = 256 / x15237;
bool x15242;
if (x452) {
bool x15239 = x15238 == 1;
bool x15240 = 256 == x15238;
bool x15241 = x15239 || x15240;
x15242 = x15241;
} else {
x15242 = false;
}
bool x15246;
if (x15242) {
x15246 = x15245;
} else {
x15246 = false;
}
bool x15247;
if (x15246) {
x15247 = x15245;
} else {
x15247 = false;
}
if (x15247) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x15132,x15132,1,x15238,1,1);
assert(false && "");
}
bool x15253 = 256 <= x15238;
int32_t x15254;
if (x15253) {
x15254 = x15238;
} else {
x15254 = 256;
}
bool x15260 = x15254 > 0;
bool x15262;
if (x15260) {
x15262 = x15261;
} else {
x15262 = false;
}
bool x15263;
if (x15262) {
x15263 = x15261;
} else {
x15263 = false;
}
if (x15263) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(15132) x Sym(15132)"," x Const(1) x Sym(15238) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x15258 = x15254 * x15257;
int32_t x15259 = 64 * x15258;
float* x15269 = (float*)myMalloc(x15259 * sizeof(float));;
int32_t x15270 = 0;
int32_t x15271 = 0;
int32_t x15272 = 0;
bool x15319 = x15238 > 1;
for(int x15273=0; x15273 < 64; x15273++) {
int32_t x15274 = x15271;
int32_t x15275 = x15272;
int32_t x15276 = x15270;
int32_t x15277 = x15276;
int32_t x15278 = x15274;
int32_t x15279 = x15275;
for(int x15281=0; x15281 < x15254; x15281++) {
int32_t x15282 = x15278;
int32_t x15283 = x15279;
int32_t x15284 = x15277;
int32_t x15285 = x15284;
int32_t x15286 = x15282;
int32_t x15287 = x15283;
for(int x15289=0; x15289 < x15256; x15289++) {
int32_t x15290 = x15286;
int32_t x15291 = x15287;
int32_t x15292 = x15285;
int32_t x15293 = x15292;
int32_t x15294 = x15290;
int32_t x15295 = x15291;
for(int x15296=0; x15296 < x15256; x15296++) {
int32_t x15297 = x15293;
int32_t x15298 = x15294;
float x15299 = x15138[x15298];
int32_t x15300 = x15295;
float x15301 = x229[x15300];
float x15302 = x15299 - x15301;
x15269[x15297] = x15302;
x15293 += 1;
if (x15305) {
x15294 += 1;
} else {
}

}
x15285 += x15256;
if (x15305) {
x15286 += x15132;
} else {
}

}
x15277 += x15257;
x15278 += x15133;
if (x15319) {
x15279 += 1;
} else {
}

}
x15270 += x15258;
x15271 += x15134;

}
float* x15329 = (float*)myMalloc(256 * sizeof(float));;
for(int x15330=0; x15330 < 256; x15330++) {
float x15331 = x99[x15330];
float x15332 = x15331 + 1.0E-5f;
x15329[x15330] = x15332;

}
float* x15336 = (float*)myMalloc(256 * sizeof(float));;
for(int x15337=0; x15337 < 256; x15337++) {
float x15338 = x15329[x15337];
double x15339 = (double)x15338;
double x15340 = sqrt(x15339);
float x15341 = (float)x15340;
x15336[x15337] = x15341;

}
int32_t x15345 = 0;
int32_t x15346 = 1;
x15346 *= 1;
x15345 += 1;
x15346 *= 1;
x15346 *= 1;
int32_t x15351 = x15345;
bool x15352 = x15351 >= 2;
if (x15352) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15357 = x15351 == 0;
if (x15357) {
int32_t x15358 = x15346;
bool x15359 = x15358 == 256;
if (x15359) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15366 = x15346;
int32_t x15367 = 256 / x15366;
bool x15373;
if (x452) {
bool x15368 = x15254 == 1;
bool x15369 = x15367 == 1;
bool x15370 = x15368 || x15369;
bool x15371 = x15254 == x15367;
bool x15372 = x15370 || x15371;
x15373 = x15372;
} else {
x15373 = false;
}
bool x15377;
if (x15373) {
x15377 = x15376;
} else {
x15377 = false;
}
bool x15378;
if (x15377) {
x15378 = x15376;
} else {
x15378 = false;
}
if (x15378) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15254,x15256,x15256,1,x15367,1,1);
assert(false && "");
}
bool x15384 = x15254 <= x15367;
int32_t x15385;
if (x15384) {
x15385 = x15367;
} else {
x15385 = x15254;
}
bool x15391 = x15385 > 0;
bool x15393;
if (x15391) {
x15393 = x15392;
} else {
x15393 = false;
}
bool x15394;
if (x15393) {
x15394 = x15392;
} else {
x15394 = false;
}
if (x15394) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(15254) x Sym(15256) x Sym(15256)"," x Const(1) x Sym(15367) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x15389 = x15385 * x15388;
int32_t x15390 = 64 * x15389;
float* x15400 = (float*)myMalloc(x15390 * sizeof(float));;
int32_t x15401 = 0;
int32_t x15402 = 0;
int32_t x15403 = 0;
bool x15449 = x15254 > 1;
bool x15453 = x15367 > 1;
for(int x15404=0; x15404 < 64; x15404++) {
int32_t x15405 = x15402;
int32_t x15406 = x15403;
int32_t x15407 = x15401;
int32_t x15408 = x15407;
int32_t x15409 = x15405;
int32_t x15410 = x15406;
for(int x15412=0; x15412 < x15385; x15412++) {
int32_t x15413 = x15409;
int32_t x15414 = x15410;
int32_t x15415 = x15408;
int32_t x15416 = x15415;
int32_t x15417 = x15413;
int32_t x15418 = x15414;
for(int x15420=0; x15420 < x15387; x15420++) {
int32_t x15421 = x15417;
int32_t x15422 = x15418;
int32_t x15423 = x15416;
int32_t x15424 = x15423;
int32_t x15425 = x15421;
int32_t x15426 = x15422;
for(int x15427=0; x15427 < x15387; x15427++) {
int32_t x15428 = x15424;
int32_t x15429 = x15425;
float x15430 = x15269[x15429];
int32_t x15431 = x15426;
float x15432 = x15336[x15431];
float x15433 = x15430 / x15432;
x15400[x15428] = x15433;
x15424 += 1;
if (x15436) {
x15425 += 1;
} else {
}

}
x15416 += x15387;
if (x15436) {
x15417 += x15256;
} else {
}

}
x15408 += x15388;
if (x15449) {
x15409 += x15257;
} else {
}
if (x15453) {
x15410 += 1;
} else {
}

}
x15401 += x15389;
x15402 += x15258;

}
int32_t x15463 = 0;
int32_t x15464 = 1;
x15464 *= 1;
x15463 += 1;
x15464 *= 1;
x15464 *= 1;
int32_t x15469 = x15463;
bool x15470 = x15469 >= 2;
if (x15470) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15475 = x15469 == 0;
if (x15475) {
int32_t x15476 = x15464;
bool x15477 = x15476 == 256;
if (x15477) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15484 = x15464;
int32_t x15485 = 256 / x15484;
bool x15491;
if (x452) {
bool x15486 = x15385 == 1;
bool x15487 = x15485 == 1;
bool x15488 = x15486 || x15487;
bool x15489 = x15385 == x15485;
bool x15490 = x15488 || x15489;
x15491 = x15490;
} else {
x15491 = false;
}
bool x15495;
if (x15491) {
x15495 = x15494;
} else {
x15495 = false;
}
bool x15496;
if (x15495) {
x15496 = x15494;
} else {
x15496 = false;
}
if (x15496) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15385,x15387,x15387,1,x15485,1,1);
assert(false && "");
}
bool x15502 = x15385 <= x15485;
int32_t x15503;
if (x15502) {
x15503 = x15485;
} else {
x15503 = x15385;
}
bool x15509 = x15503 > 0;
bool x15511;
if (x15509) {
x15511 = x15510;
} else {
x15511 = false;
}
bool x15512;
if (x15511) {
x15512 = x15510;
} else {
x15512 = false;
}
if (x15512) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(15385) x Sym(15387) x Sym(15387)"," x Const(1) x Sym(15485) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x15507 = x15503 * x15506;
int32_t x15508 = 64 * x15507;
float* x15518 = (float*)myMalloc(x15508 * sizeof(float));;
int32_t x15519 = 0;
int32_t x15520 = 0;
int32_t x15521 = 0;
bool x15567 = x15385 > 1;
bool x15571 = x15485 > 1;
for(int x15522=0; x15522 < 64; x15522++) {
int32_t x15523 = x15520;
int32_t x15524 = x15521;
int32_t x15525 = x15519;
int32_t x15526 = x15525;
int32_t x15527 = x15523;
int32_t x15528 = x15524;
for(int x15530=0; x15530 < x15503; x15530++) {
int32_t x15531 = x15527;
int32_t x15532 = x15528;
int32_t x15533 = x15526;
int32_t x15534 = x15533;
int32_t x15535 = x15531;
int32_t x15536 = x15532;
for(int x15538=0; x15538 < x15505; x15538++) {
int32_t x15539 = x15535;
int32_t x15540 = x15536;
int32_t x15541 = x15534;
int32_t x15542 = x15541;
int32_t x15543 = x15539;
int32_t x15544 = x15540;
for(int x15545=0; x15545 < x15505; x15545++) {
int32_t x15546 = x15542;
int32_t x15547 = x15543;
float x15548 = x15400[x15547];
int32_t x15549 = x15544;
float x15550 = x108[x15549];
float x15551 = x15548 * x15550;
x15518[x15546] = x15551;
x15542 += 1;
if (x15554) {
x15543 += 1;
} else {
}

}
x15534 += x15505;
if (x15554) {
x15535 += x15387;
} else {
}

}
x15526 += x15506;
if (x15567) {
x15527 += x15388;
} else {
}
if (x15571) {
x15528 += 1;
} else {
}

}
x15519 += x15507;
x15520 += x15389;

}
int32_t x15581 = 0;
int32_t x15582 = 1;
x15582 *= 1;
x15581 += 1;
x15582 *= 1;
x15582 *= 1;
int32_t x15587 = x15581;
bool x15588 = x15587 >= 2;
if (x15588) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15593 = x15587 == 0;
if (x15593) {
int32_t x15594 = x15582;
bool x15595 = x15594 == 256;
if (x15595) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15602 = x15582;
int32_t x15603 = 256 / x15602;
bool x15609;
if (x452) {
bool x15604 = x15503 == 1;
bool x15605 = x15603 == 1;
bool x15606 = x15604 || x15605;
bool x15607 = x15503 == x15603;
bool x15608 = x15606 || x15607;
x15609 = x15608;
} else {
x15609 = false;
}
bool x15613;
if (x15609) {
x15613 = x15612;
} else {
x15613 = false;
}
bool x15614;
if (x15613) {
x15614 = x15612;
} else {
x15614 = false;
}
if (x15614) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15503,x15505,x15505,1,x15603,1,1);
assert(false && "");
}
bool x15620 = x15503 <= x15603;
int32_t x15621;
if (x15620) {
x15621 = x15603;
} else {
x15621 = x15503;
}
bool x15627 = x15621 > 0;
bool x15629;
if (x15627) {
x15629 = x15628;
} else {
x15629 = false;
}
bool x15630;
if (x15629) {
x15630 = x15628;
} else {
x15630 = false;
}
if (x15630) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(15503) x Sym(15505) x Sym(15505)"," x Const(1) x Sym(15603) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x15625 = x15621 * x15624;
int32_t x15626 = 64 * x15625;
float* x15636 = (float*)myMalloc(x15626 * sizeof(float));;
int32_t x15637 = 0;
int32_t x15638 = 0;
int32_t x15639 = 0;
bool x15685 = x15503 > 1;
bool x15689 = x15603 > 1;
for(int x15640=0; x15640 < 64; x15640++) {
int32_t x15641 = x15638;
int32_t x15642 = x15639;
int32_t x15643 = x15637;
int32_t x15644 = x15643;
int32_t x15645 = x15641;
int32_t x15646 = x15642;
for(int x15648=0; x15648 < x15621; x15648++) {
int32_t x15649 = x15645;
int32_t x15650 = x15646;
int32_t x15651 = x15644;
int32_t x15652 = x15651;
int32_t x15653 = x15649;
int32_t x15654 = x15650;
for(int x15656=0; x15656 < x15623; x15656++) {
int32_t x15657 = x15653;
int32_t x15658 = x15654;
int32_t x15659 = x15652;
int32_t x15660 = x15659;
int32_t x15661 = x15657;
int32_t x15662 = x15658;
for(int x15663=0; x15663 < x15623; x15663++) {
int32_t x15664 = x15660;
int32_t x15665 = x15661;
float x15666 = x15518[x15665];
int32_t x15667 = x15662;
float x15668 = x16[x15667];
float x15669 = x15666 + x15668;
x15636[x15664] = x15669;
x15660 += 1;
if (x15672) {
x15661 += 1;
} else {
}

}
x15652 += x15623;
if (x15672) {
x15653 += x15505;
} else {
}

}
x15644 += x15624;
if (x15685) {
x15645 += x15506;
} else {
}
if (x15689) {
x15646 += 1;
} else {
}

}
x15637 += x15625;
x15638 += x15507;

}
float* x15699 = (float*)myMalloc(x15626 * sizeof(float));;
for(int x15701=0; x15701 < x15626; x15701++) {
float x15702 = x15636[x15701];
bool x15703 = x15702 < 0.0f;
if (x15703) {
x15699[x15701] = 0.0f;
} else {
float x15706 = x15636[x15701];
x15699[x15701] = x15706;
}

}
float* x15720 = (float*)myMalloc(x15719 * sizeof(float));;
int32_t x15723 = 64 * x15621;
int32_t x15724 = x15723 * x15715;
float* x15725 = (float*)myMalloc(x15724 * sizeof(float));;
int32_t x15721 = x15621 * x15715;
for(int x15726=0; x15726 < 64; x15726++) {
int32_t x15727 = x15726 * x15625;
float* x15728 = x15699+x15727;
int32_t x15729 = x15726 * x15716;
float* x15730 = x15720+x15729;
int32_t x15731 = x15726 * x15721;
float* x15732 = x15725+x15731;
for(int x15733=0; x15733 < x15621; x15733++) {
int32_t x15734 = x15733 / 1;
int32_t x15738 = x15734 * x15714;
int32_t x15739 = x15738 * x15714;
int32_t x15735 = x15733 % 1;
int32_t x15736 = x15735 / 1;
int32_t x15740 = x15736 * x15714;
int32_t x15741 = x15740 * x15714;
int32_t x15742 = x15739 + x15741;
int32_t x15737 = x15735 % 1;
int32_t x15743 = x15737 * x15714;
int32_t x15744 = x15743 * x15714;
int32_t x15745 = x15742 + x15744;
float* x15746 = x15732+x15745;
int32_t x15747 = x15734 * x15623;
int32_t x15748 = x15747 * x15623;
float* x15749 = x15728+x15748;
for(int x15751=0; x15751 < x15714; x15751++) {
int32_t x15753 = x15751 * x15714;
float* x15754 = x15746+x15753;
int32_t x15752 = x15751 + x15736;
int32_t x15755 = x15752 * x15623;
int32_t x15756 = x15755 + x15737;
float* x15757 = x15749+x15756;
memcpy(x15754, x15757, 4 * x15714);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x15715,x15621,1,x269,x15621,x15732,x15715,1,x15730,x15715);

}
int32_t x15766 = 0;
int32_t x15767 = 1;
x15767 *= 1;
x15766 += 1;
x15767 *= 1;
x15767 *= 1;
int32_t x15772 = x15766;
bool x15773 = x15772 >= 2;
if (x15773) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15778 = x15772 == 0;
if (x15778) {
int32_t x15779 = x15767;
bool x15780 = x15779 == 1024;
if (x15780) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15787 = x15767;
int32_t x15788 = 1024 / x15787;
bool x15792;
if (x452) {
bool x15789 = x15788 == 1;
bool x15790 = 1024 == x15788;
bool x15791 = x15789 || x15790;
x15792 = x15791;
} else {
x15792 = false;
}
bool x15796;
if (x15792) {
x15796 = x15795;
} else {
x15796 = false;
}
bool x15797;
if (x15796) {
x15797 = x15795;
} else {
x15797 = false;
}
if (x15797) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x15714,x15714,1,x15788,1,1);
assert(false && "");
}
bool x15803 = 1024 <= x15788;
int32_t x15804;
if (x15803) {
x15804 = x15788;
} else {
x15804 = 1024;
}
bool x15810 = x15804 > 0;
bool x15812;
if (x15810) {
x15812 = x15811;
} else {
x15812 = false;
}
bool x15813;
if (x15812) {
x15813 = x15811;
} else {
x15813 = false;
}
if (x15813) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(1024) x Sym(15714) x Sym(15714)"," x Const(1) x Sym(15788) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x15808 = x15804 * x15807;
int32_t x15809 = 64 * x15808;
float* x15819 = (float*)myMalloc(x15809 * sizeof(float));;
int32_t x15820 = 0;
int32_t x15821 = 0;
int32_t x15822 = 0;
bool x15869 = x15788 > 1;
for(int x15823=0; x15823 < 64; x15823++) {
int32_t x15824 = x15821;
int32_t x15825 = x15822;
int32_t x15826 = x15820;
int32_t x15827 = x15826;
int32_t x15828 = x15824;
int32_t x15829 = x15825;
for(int x15831=0; x15831 < x15804; x15831++) {
int32_t x15832 = x15828;
int32_t x15833 = x15829;
int32_t x15834 = x15827;
int32_t x15835 = x15834;
int32_t x15836 = x15832;
int32_t x15837 = x15833;
for(int x15839=0; x15839 < x15806; x15839++) {
int32_t x15840 = x15836;
int32_t x15841 = x15837;
int32_t x15842 = x15835;
int32_t x15843 = x15842;
int32_t x15844 = x15840;
int32_t x15845 = x15841;
for(int x15846=0; x15846 < x15806; x15846++) {
int32_t x15847 = x15843;
int32_t x15848 = x15844;
float x15849 = x15720[x15848];
int32_t x15850 = x15845;
float x15851 = x216[x15850];
float x15852 = x15849 - x15851;
x15819[x15847] = x15852;
x15843 += 1;
if (x15855) {
x15844 += 1;
} else {
}

}
x15835 += x15806;
if (x15855) {
x15836 += x15714;
} else {
}

}
x15827 += x15807;
x15828 += x15715;
if (x15869) {
x15829 += 1;
} else {
}

}
x15820 += x15808;
x15821 += x15716;

}
float* x15879 = (float*)myMalloc(1024 * sizeof(float));;
for(int x15881=0; x15881 < 1024; x15881++) {
float x15882 = x267[x15881];
float x15883 = x15882 + 1.0E-5f;
x15879[x15881] = x15883;

}
float* x15887 = (float*)myMalloc(1024 * sizeof(float));;
for(int x15888=0; x15888 < 1024; x15888++) {
float x15889 = x15879[x15888];
double x15890 = (double)x15889;
double x15891 = sqrt(x15890);
float x15892 = (float)x15891;
x15887[x15888] = x15892;

}
int32_t x15896 = 0;
int32_t x15897 = 1;
x15897 *= 1;
x15896 += 1;
x15897 *= 1;
x15897 *= 1;
int32_t x15902 = x15896;
bool x15903 = x15902 >= 2;
if (x15903) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x15908 = x15902 == 0;
if (x15908) {
int32_t x15909 = x15897;
bool x15910 = x15909 == 1024;
if (x15910) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x15917 = x15897;
int32_t x15918 = 1024 / x15917;
bool x15924;
if (x452) {
bool x15919 = x15804 == 1;
bool x15920 = x15918 == 1;
bool x15921 = x15919 || x15920;
bool x15922 = x15804 == x15918;
bool x15923 = x15921 || x15922;
x15924 = x15923;
} else {
x15924 = false;
}
bool x15928;
if (x15924) {
x15928 = x15927;
} else {
x15928 = false;
}
bool x15929;
if (x15928) {
x15929 = x15927;
} else {
x15929 = false;
}
if (x15929) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15804,x15806,x15806,1,x15918,1,1);
assert(false && "");
}
bool x15935 = x15804 <= x15918;
int32_t x15936;
if (x15935) {
x15936 = x15918;
} else {
x15936 = x15804;
}
bool x15942 = x15936 > 0;
bool x15944;
if (x15942) {
x15944 = x15943;
} else {
x15944 = false;
}
bool x15945;
if (x15944) {
x15945 = x15943;
} else {
x15945 = false;
}
if (x15945) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(15804) x Sym(15806) x Sym(15806)"," x Const(1) x Sym(15918) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x15940 = x15936 * x15939;
int32_t x15941 = 64 * x15940;
float* x15951 = (float*)myMalloc(x15941 * sizeof(float));;
int32_t x15952 = 0;
int32_t x15953 = 0;
int32_t x15954 = 0;
bool x16000 = x15804 > 1;
bool x16004 = x15918 > 1;
for(int x15955=0; x15955 < 64; x15955++) {
int32_t x15956 = x15953;
int32_t x15957 = x15954;
int32_t x15958 = x15952;
int32_t x15959 = x15958;
int32_t x15960 = x15956;
int32_t x15961 = x15957;
for(int x15963=0; x15963 < x15936; x15963++) {
int32_t x15964 = x15960;
int32_t x15965 = x15961;
int32_t x15966 = x15959;
int32_t x15967 = x15966;
int32_t x15968 = x15964;
int32_t x15969 = x15965;
for(int x15971=0; x15971 < x15938; x15971++) {
int32_t x15972 = x15968;
int32_t x15973 = x15969;
int32_t x15974 = x15967;
int32_t x15975 = x15974;
int32_t x15976 = x15972;
int32_t x15977 = x15973;
for(int x15978=0; x15978 < x15938; x15978++) {
int32_t x15979 = x15975;
int32_t x15980 = x15976;
float x15981 = x15819[x15980];
int32_t x15982 = x15977;
float x15983 = x15887[x15982];
float x15984 = x15981 / x15983;
x15951[x15979] = x15984;
x15975 += 1;
if (x15987) {
x15976 += 1;
} else {
}

}
x15967 += x15938;
if (x15987) {
x15968 += x15806;
} else {
}

}
x15959 += x15939;
if (x16000) {
x15960 += x15807;
} else {
}
if (x16004) {
x15961 += 1;
} else {
}

}
x15952 += x15940;
x15953 += x15808;

}
int32_t x16014 = 0;
int32_t x16015 = 1;
x16015 *= 1;
x16014 += 1;
x16015 *= 1;
x16015 *= 1;
int32_t x16020 = x16014;
bool x16021 = x16020 >= 2;
if (x16021) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16026 = x16020 == 0;
if (x16026) {
int32_t x16027 = x16015;
bool x16028 = x16027 == 1024;
if (x16028) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16035 = x16015;
int32_t x16036 = 1024 / x16035;
bool x16042;
if (x452) {
bool x16037 = x15936 == 1;
bool x16038 = x16036 == 1;
bool x16039 = x16037 || x16038;
bool x16040 = x15936 == x16036;
bool x16041 = x16039 || x16040;
x16042 = x16041;
} else {
x16042 = false;
}
bool x16046;
if (x16042) {
x16046 = x16045;
} else {
x16046 = false;
}
bool x16047;
if (x16046) {
x16047 = x16045;
} else {
x16047 = false;
}
if (x16047) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x15936,x15938,x15938,1,x16036,1,1);
assert(false && "");
}
bool x16053 = x15936 <= x16036;
int32_t x16054;
if (x16053) {
x16054 = x16036;
} else {
x16054 = x15936;
}
bool x16060 = x16054 > 0;
bool x16062;
if (x16060) {
x16062 = x16061;
} else {
x16062 = false;
}
bool x16063;
if (x16062) {
x16063 = x16061;
} else {
x16063 = false;
}
if (x16063) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(15936) x Sym(15938) x Sym(15938)"," x Const(1) x Sym(16036) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x16058 = x16054 * x16057;
int32_t x16059 = 64 * x16058;
float* x16069 = (float*)myMalloc(x16059 * sizeof(float));;
int32_t x16070 = 0;
int32_t x16071 = 0;
int32_t x16072 = 0;
bool x16118 = x15936 > 1;
bool x16122 = x16036 > 1;
for(int x16073=0; x16073 < 64; x16073++) {
int32_t x16074 = x16071;
int32_t x16075 = x16072;
int32_t x16076 = x16070;
int32_t x16077 = x16076;
int32_t x16078 = x16074;
int32_t x16079 = x16075;
for(int x16081=0; x16081 < x16054; x16081++) {
int32_t x16082 = x16078;
int32_t x16083 = x16079;
int32_t x16084 = x16077;
int32_t x16085 = x16084;
int32_t x16086 = x16082;
int32_t x16087 = x16083;
for(int x16089=0; x16089 < x16056; x16089++) {
int32_t x16090 = x16086;
int32_t x16091 = x16087;
int32_t x16092 = x16085;
int32_t x16093 = x16092;
int32_t x16094 = x16090;
int32_t x16095 = x16091;
for(int x16096=0; x16096 < x16056; x16096++) {
int32_t x16097 = x16093;
int32_t x16098 = x16094;
float x16099 = x15951[x16098];
int32_t x16100 = x16095;
float x16101 = x18[x16100];
float x16102 = x16099 * x16101;
x16069[x16097] = x16102;
x16093 += 1;
if (x16105) {
x16094 += 1;
} else {
}

}
x16085 += x16056;
if (x16105) {
x16086 += x15938;
} else {
}

}
x16077 += x16057;
if (x16118) {
x16078 += x15939;
} else {
}
if (x16122) {
x16079 += 1;
} else {
}

}
x16070 += x16058;
x16071 += x15940;

}
int32_t x16132 = 0;
int32_t x16133 = 1;
x16133 *= 1;
x16132 += 1;
x16133 *= 1;
x16133 *= 1;
int32_t x16138 = x16132;
bool x16139 = x16138 >= 2;
if (x16139) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16144 = x16138 == 0;
if (x16144) {
int32_t x16145 = x16133;
bool x16146 = x16145 == 1024;
if (x16146) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16153 = x16133;
int32_t x16154 = 1024 / x16153;
bool x16160;
if (x452) {
bool x16155 = x16054 == 1;
bool x16156 = x16154 == 1;
bool x16157 = x16155 || x16156;
bool x16158 = x16054 == x16154;
bool x16159 = x16157 || x16158;
x16160 = x16159;
} else {
x16160 = false;
}
bool x16164;
if (x16160) {
x16164 = x16163;
} else {
x16164 = false;
}
bool x16165;
if (x16164) {
x16165 = x16163;
} else {
x16165 = false;
}
if (x16165) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16054,x16056,x16056,1,x16154,1,1);
assert(false && "");
}
bool x16171 = x16054 <= x16154;
int32_t x16172;
if (x16171) {
x16172 = x16154;
} else {
x16172 = x16054;
}
bool x16178 = x16172 > 0;
bool x16180;
if (x16178) {
x16180 = x16179;
} else {
x16180 = false;
}
bool x16181;
if (x16180) {
x16181 = x16179;
} else {
x16181 = false;
}
if (x16181) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(16054) x Sym(16056) x Sym(16056)"," x Const(1) x Sym(16154) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x16176 = x16172 * x16175;
int32_t x16177 = 64 * x16176;
float* x16187 = (float*)myMalloc(x16177 * sizeof(float));;
int32_t x16188 = 0;
int32_t x16189 = 0;
int32_t x16190 = 0;
bool x16236 = x16054 > 1;
bool x16240 = x16154 > 1;
for(int x16191=0; x16191 < 64; x16191++) {
int32_t x16192 = x16189;
int32_t x16193 = x16190;
int32_t x16194 = x16188;
int32_t x16195 = x16194;
int32_t x16196 = x16192;
int32_t x16197 = x16193;
for(int x16199=0; x16199 < x16172; x16199++) {
int32_t x16200 = x16196;
int32_t x16201 = x16197;
int32_t x16202 = x16195;
int32_t x16203 = x16202;
int32_t x16204 = x16200;
int32_t x16205 = x16201;
for(int x16207=0; x16207 < x16174; x16207++) {
int32_t x16208 = x16204;
int32_t x16209 = x16205;
int32_t x16210 = x16203;
int32_t x16211 = x16210;
int32_t x16212 = x16208;
int32_t x16213 = x16209;
for(int x16214=0; x16214 < x16174; x16214++) {
int32_t x16215 = x16211;
int32_t x16216 = x16212;
float x16217 = x16069[x16216];
int32_t x16218 = x16213;
float x16219 = x117[x16218];
float x16220 = x16217 + x16219;
x16187[x16215] = x16220;
x16211 += 1;
if (x16223) {
x16212 += 1;
} else {
}

}
x16203 += x16174;
if (x16223) {
x16204 += x16056;
} else {
}

}
x16195 += x16175;
if (x16236) {
x16196 += x16057;
} else {
}
if (x16240) {
x16197 += 1;
} else {
}

}
x16188 += x16176;
x16189 += x16058;

}
float* x16257 = (float*)myMalloc(x16256 * sizeof(float));;
int32_t x16260 = x14590 * x16252;
float* x16261 = (float*)myMalloc(x16260 * sizeof(float));;
int32_t x16258 = x14399 * x16252;
for(int x16262=0; x16262 < 64; x16262++) {
int32_t x16263 = x16262 * x14403;
float* x16264 = x14566+x16263;
int32_t x16265 = x16262 * x16253;
float* x16266 = x16257+x16265;
int32_t x16267 = x16262 * x16258;
float* x16268 = x16261+x16267;
for(int x16269=0; x16269 < x14399; x16269++) {
int32_t x16270 = x16269 / 1;
int32_t x16274 = x16270 * x16251;
int32_t x16275 = x16274 * x16251;
int32_t x16271 = x16269 % 1;
int32_t x16272 = x16271 / 1;
int32_t x16276 = x16272 * x16251;
int32_t x16277 = x16276 * x16251;
int32_t x16278 = x16275 + x16277;
int32_t x16273 = x16271 % 1;
int32_t x16279 = x16273 * x16251;
int32_t x16280 = x16279 * x16251;
int32_t x16281 = x16278 + x16280;
float* x16282 = x16268+x16281;
int32_t x16283 = x16270 * x14401;
int32_t x16284 = x16283 * x14401;
float* x16285 = x16264+x16284;
for(int x16287=0; x16287 < x16251; x16287++) {
int32_t x16291 = x16287 * x16251;
int32_t x16288 = x16287 * 2;
int32_t x16289 = x16288 + x16272;
int32_t x16294 = x16289 * x14401;
int32_t x16295 = x16294 + x16273;
for(int x16290=0; x16290 < x16251; x16290++) {
int32_t x16292 = x16291 + x16290;
float* x16293 = x16282+x16292;
int32_t x16296 = x16290 * 2;
int32_t x16297 = x16295 + x16296;
float* x16298 = x16285+x16297;
memcpy(x16293, x16298, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x16252,x14399,1,x75,x14399,x16268,x16252,1,x16266,x16252);

}
int32_t x16309 = 0;
int32_t x16310 = 1;
x16310 *= 1;
x16309 += 1;
x16310 *= 1;
x16310 *= 1;
int32_t x16315 = x16309;
bool x16316 = x16315 >= 2;
if (x16316) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16321 = x16315 == 0;
if (x16321) {
int32_t x16322 = x16310;
bool x16323 = x16322 == 1024;
if (x16323) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16330 = x16310;
int32_t x16331 = 1024 / x16330;
bool x16335;
if (x452) {
bool x16332 = x16331 == 1;
bool x16333 = 1024 == x16331;
bool x16334 = x16332 || x16333;
x16335 = x16334;
} else {
x16335 = false;
}
bool x16339;
if (x16335) {
x16339 = x16338;
} else {
x16339 = false;
}
bool x16340;
if (x16339) {
x16340 = x16338;
} else {
x16340 = false;
}
if (x16340) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x16251,x16251,1,x16331,1,1);
assert(false && "");
}
bool x16346 = 1024 <= x16331;
int32_t x16347;
if (x16346) {
x16347 = x16331;
} else {
x16347 = 1024;
}
bool x16353 = x16347 > 0;
bool x16355;
if (x16353) {
x16355 = x16354;
} else {
x16355 = false;
}
bool x16356;
if (x16355) {
x16356 = x16354;
} else {
x16356 = false;
}
if (x16356) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(1024) x Sym(16251) x Sym(16251)"," x Const(1) x Sym(16331) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x16351 = x16347 * x16350;
int32_t x16352 = 64 * x16351;
float* x16362 = (float*)myMalloc(x16352 * sizeof(float));;
int32_t x16363 = 0;
int32_t x16364 = 0;
int32_t x16365 = 0;
bool x16412 = x16331 > 1;
for(int x16366=0; x16366 < 64; x16366++) {
int32_t x16367 = x16364;
int32_t x16368 = x16365;
int32_t x16369 = x16363;
int32_t x16370 = x16369;
int32_t x16371 = x16367;
int32_t x16372 = x16368;
for(int x16374=0; x16374 < x16347; x16374++) {
int32_t x16375 = x16371;
int32_t x16376 = x16372;
int32_t x16377 = x16370;
int32_t x16378 = x16377;
int32_t x16379 = x16375;
int32_t x16380 = x16376;
for(int x16382=0; x16382 < x16349; x16382++) {
int32_t x16383 = x16379;
int32_t x16384 = x16380;
int32_t x16385 = x16378;
int32_t x16386 = x16385;
int32_t x16387 = x16383;
int32_t x16388 = x16384;
for(int x16389=0; x16389 < x16349; x16389++) {
int32_t x16390 = x16386;
int32_t x16391 = x16387;
float x16392 = x16257[x16391];
int32_t x16393 = x16388;
float x16394 = x86[x16393];
float x16395 = x16392 - x16394;
x16362[x16390] = x16395;
x16386 += 1;
if (x16398) {
x16387 += 1;
} else {
}

}
x16378 += x16349;
if (x16398) {
x16379 += x16251;
} else {
}

}
x16370 += x16350;
x16371 += x16252;
if (x16412) {
x16372 += 1;
} else {
}

}
x16363 += x16351;
x16364 += x16253;

}
float* x16422 = (float*)myMalloc(1024 * sizeof(float));;
for(int x16423=0; x16423 < 1024; x16423++) {
float x16424 = x211[x16423];
float x16425 = x16424 + 1.0E-5f;
x16422[x16423] = x16425;

}
float* x16429 = (float*)myMalloc(1024 * sizeof(float));;
for(int x16430=0; x16430 < 1024; x16430++) {
float x16431 = x16422[x16430];
double x16432 = (double)x16431;
double x16433 = sqrt(x16432);
float x16434 = (float)x16433;
x16429[x16430] = x16434;

}
int32_t x16438 = 0;
int32_t x16439 = 1;
x16439 *= 1;
x16438 += 1;
x16439 *= 1;
x16439 *= 1;
int32_t x16444 = x16438;
bool x16445 = x16444 >= 2;
if (x16445) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16450 = x16444 == 0;
if (x16450) {
int32_t x16451 = x16439;
bool x16452 = x16451 == 1024;
if (x16452) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16459 = x16439;
int32_t x16460 = 1024 / x16459;
bool x16466;
if (x452) {
bool x16461 = x16347 == 1;
bool x16462 = x16460 == 1;
bool x16463 = x16461 || x16462;
bool x16464 = x16347 == x16460;
bool x16465 = x16463 || x16464;
x16466 = x16465;
} else {
x16466 = false;
}
bool x16470;
if (x16466) {
x16470 = x16469;
} else {
x16470 = false;
}
bool x16471;
if (x16470) {
x16471 = x16469;
} else {
x16471 = false;
}
if (x16471) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16347,x16349,x16349,1,x16460,1,1);
assert(false && "");
}
bool x16477 = x16347 <= x16460;
int32_t x16478;
if (x16477) {
x16478 = x16460;
} else {
x16478 = x16347;
}
bool x16484 = x16478 > 0;
bool x16486;
if (x16484) {
x16486 = x16485;
} else {
x16486 = false;
}
bool x16487;
if (x16486) {
x16487 = x16485;
} else {
x16487 = false;
}
if (x16487) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(16347) x Sym(16349) x Sym(16349)"," x Const(1) x Sym(16460) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x16482 = x16478 * x16481;
int32_t x16483 = 64 * x16482;
float* x16493 = (float*)myMalloc(x16483 * sizeof(float));;
int32_t x16494 = 0;
int32_t x16495 = 0;
int32_t x16496 = 0;
bool x16542 = x16347 > 1;
bool x16546 = x16460 > 1;
for(int x16497=0; x16497 < 64; x16497++) {
int32_t x16498 = x16495;
int32_t x16499 = x16496;
int32_t x16500 = x16494;
int32_t x16501 = x16500;
int32_t x16502 = x16498;
int32_t x16503 = x16499;
for(int x16505=0; x16505 < x16478; x16505++) {
int32_t x16506 = x16502;
int32_t x16507 = x16503;
int32_t x16508 = x16501;
int32_t x16509 = x16508;
int32_t x16510 = x16506;
int32_t x16511 = x16507;
for(int x16513=0; x16513 < x16480; x16513++) {
int32_t x16514 = x16510;
int32_t x16515 = x16511;
int32_t x16516 = x16509;
int32_t x16517 = x16516;
int32_t x16518 = x16514;
int32_t x16519 = x16515;
for(int x16520=0; x16520 < x16480; x16520++) {
int32_t x16521 = x16517;
int32_t x16522 = x16518;
float x16523 = x16362[x16522];
int32_t x16524 = x16519;
float x16525 = x16429[x16524];
float x16526 = x16523 / x16525;
x16493[x16521] = x16526;
x16517 += 1;
if (x16529) {
x16518 += 1;
} else {
}

}
x16509 += x16480;
if (x16529) {
x16510 += x16349;
} else {
}

}
x16501 += x16481;
if (x16542) {
x16502 += x16350;
} else {
}
if (x16546) {
x16503 += 1;
} else {
}

}
x16494 += x16482;
x16495 += x16351;

}
int32_t x16556 = 0;
int32_t x16557 = 1;
x16557 *= 1;
x16556 += 1;
x16557 *= 1;
x16557 *= 1;
int32_t x16562 = x16556;
bool x16563 = x16562 >= 2;
if (x16563) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16568 = x16562 == 0;
if (x16568) {
int32_t x16569 = x16557;
bool x16570 = x16569 == 1024;
if (x16570) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16577 = x16557;
int32_t x16578 = 1024 / x16577;
bool x16584;
if (x452) {
bool x16579 = x16478 == 1;
bool x16580 = x16578 == 1;
bool x16581 = x16579 || x16580;
bool x16582 = x16478 == x16578;
bool x16583 = x16581 || x16582;
x16584 = x16583;
} else {
x16584 = false;
}
bool x16588;
if (x16584) {
x16588 = x16587;
} else {
x16588 = false;
}
bool x16589;
if (x16588) {
x16589 = x16587;
} else {
x16589 = false;
}
if (x16589) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16478,x16480,x16480,1,x16578,1,1);
assert(false && "");
}
bool x16595 = x16478 <= x16578;
int32_t x16596;
if (x16595) {
x16596 = x16578;
} else {
x16596 = x16478;
}
bool x16602 = x16596 > 0;
bool x16604;
if (x16602) {
x16604 = x16603;
} else {
x16604 = false;
}
bool x16605;
if (x16604) {
x16605 = x16603;
} else {
x16605 = false;
}
if (x16605) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(16478) x Sym(16480) x Sym(16480)"," x Const(1) x Sym(16578) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x16600 = x16596 * x16599;
int32_t x16601 = 64 * x16600;
float* x16611 = (float*)myMalloc(x16601 * sizeof(float));;
int32_t x16612 = 0;
int32_t x16613 = 0;
int32_t x16614 = 0;
bool x16660 = x16478 > 1;
bool x16664 = x16578 > 1;
for(int x16615=0; x16615 < 64; x16615++) {
int32_t x16616 = x16613;
int32_t x16617 = x16614;
int32_t x16618 = x16612;
int32_t x16619 = x16618;
int32_t x16620 = x16616;
int32_t x16621 = x16617;
for(int x16623=0; x16623 < x16596; x16623++) {
int32_t x16624 = x16620;
int32_t x16625 = x16621;
int32_t x16626 = x16619;
int32_t x16627 = x16626;
int32_t x16628 = x16624;
int32_t x16629 = x16625;
for(int x16631=0; x16631 < x16598; x16631++) {
int32_t x16632 = x16628;
int32_t x16633 = x16629;
int32_t x16634 = x16627;
int32_t x16635 = x16634;
int32_t x16636 = x16632;
int32_t x16637 = x16633;
for(int x16638=0; x16638 < x16598; x16638++) {
int32_t x16639 = x16635;
int32_t x16640 = x16636;
float x16641 = x16493[x16640];
int32_t x16642 = x16637;
float x16643 = x29[x16642];
float x16644 = x16641 * x16643;
x16611[x16639] = x16644;
x16635 += 1;
if (x16647) {
x16636 += 1;
} else {
}

}
x16627 += x16598;
if (x16647) {
x16628 += x16480;
} else {
}

}
x16619 += x16599;
if (x16660) {
x16620 += x16481;
} else {
}
if (x16664) {
x16621 += 1;
} else {
}

}
x16612 += x16600;
x16613 += x16482;

}
int32_t x16674 = 0;
int32_t x16675 = 1;
x16675 *= 1;
x16674 += 1;
x16675 *= 1;
x16675 *= 1;
int32_t x16680 = x16674;
bool x16681 = x16680 >= 2;
if (x16681) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16686 = x16680 == 0;
if (x16686) {
int32_t x16687 = x16675;
bool x16688 = x16687 == 1024;
if (x16688) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16695 = x16675;
int32_t x16696 = 1024 / x16695;
bool x16702;
if (x452) {
bool x16697 = x16596 == 1;
bool x16698 = x16696 == 1;
bool x16699 = x16697 || x16698;
bool x16700 = x16596 == x16696;
bool x16701 = x16699 || x16700;
x16702 = x16701;
} else {
x16702 = false;
}
bool x16706;
if (x16702) {
x16706 = x16705;
} else {
x16706 = false;
}
bool x16707;
if (x16706) {
x16707 = x16705;
} else {
x16707 = false;
}
if (x16707) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16596,x16598,x16598,1,x16696,1,1);
assert(false && "");
}
bool x16713 = x16596 <= x16696;
int32_t x16714;
if (x16713) {
x16714 = x16696;
} else {
x16714 = x16596;
}
bool x16720 = x16714 > 0;
bool x16722;
if (x16720) {
x16722 = x16721;
} else {
x16722 = false;
}
bool x16723;
if (x16722) {
x16723 = x16721;
} else {
x16723 = false;
}
if (x16723) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(16596) x Sym(16598) x Sym(16598)"," x Const(1) x Sym(16696) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x16718 = x16714 * x16717;
int32_t x16719 = 64 * x16718;
float* x16729 = (float*)myMalloc(x16719 * sizeof(float));;
int32_t x16730 = 0;
int32_t x16731 = 0;
int32_t x16732 = 0;
bool x16778 = x16596 > 1;
bool x16782 = x16696 > 1;
for(int x16733=0; x16733 < 64; x16733++) {
int32_t x16734 = x16731;
int32_t x16735 = x16732;
int32_t x16736 = x16730;
int32_t x16737 = x16736;
int32_t x16738 = x16734;
int32_t x16739 = x16735;
for(int x16741=0; x16741 < x16714; x16741++) {
int32_t x16742 = x16738;
int32_t x16743 = x16739;
int32_t x16744 = x16737;
int32_t x16745 = x16744;
int32_t x16746 = x16742;
int32_t x16747 = x16743;
for(int x16749=0; x16749 < x16716; x16749++) {
int32_t x16750 = x16746;
int32_t x16751 = x16747;
int32_t x16752 = x16745;
int32_t x16753 = x16752;
int32_t x16754 = x16750;
int32_t x16755 = x16751;
for(int x16756=0; x16756 < x16716; x16756++) {
int32_t x16757 = x16753;
int32_t x16758 = x16754;
float x16759 = x16611[x16758];
int32_t x16760 = x16755;
float x16761 = x220[x16760];
float x16762 = x16759 + x16761;
x16729[x16757] = x16762;
x16753 += 1;
if (x16765) {
x16754 += 1;
} else {
}

}
x16745 += x16716;
if (x16765) {
x16746 += x16598;
} else {
}

}
x16737 += x16717;
if (x16778) {
x16738 += x16599;
} else {
}
if (x16782) {
x16739 += 1;
} else {
}

}
x16730 += x16718;
x16731 += x16600;

}
bool x16792 = x16172 == 1;
bool x16793 = x16714 == 1;
bool x16794 = x16792 || x16793;
bool x16795 = x16172 == x16714;
bool x16796 = x16794 || x16795;
bool x16802;
if (x16796) {
x16802 = x16801;
} else {
x16802 = false;
}
bool x16803;
if (x16802) {
x16803 = x16801;
} else {
x16803 = false;
}
if (x16803) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16172,x16174,x16174,64,x16714,x16716,x16716);
assert(false && "");
}
int32_t x16816 = 0;
int32_t x16817 = 0;
int32_t x16818 = 0;
bool x16809 = x16172 <= x16714;
int32_t x16810;
if (x16809) {
x16810 = x16714;
} else {
x16810 = x16172;
}
bool x16870 = x16172 > 1;
bool x16874 = x16714 > 1;
int32_t x16814 = x16810 * x16813;
for(int x16819=0; x16819 < 64; x16819++) {
int32_t x16820 = x16817;
int32_t x16821 = x16818;
int32_t x16822 = x16816;
int32_t x16823 = x16822;
int32_t x16824 = x16820;
int32_t x16825 = x16821;
for(int x16827=0; x16827 < x16810; x16827++) {
int32_t x16828 = x16824;
int32_t x16829 = x16825;
int32_t x16830 = x16823;
int32_t x16831 = x16830;
int32_t x16832 = x16828;
int32_t x16833 = x16829;
for(int x16835=0; x16835 < x16812; x16835++) {
int32_t x16836 = x16832;
int32_t x16837 = x16833;
int32_t x16838 = x16831;
int32_t x16839 = x16838;
int32_t x16840 = x16836;
int32_t x16841 = x16837;
for(int x16842=0; x16842 < x16812; x16842++) {
int32_t x16843 = x16840;
float x16844 = x16187[x16843];
int32_t x16845 = x16841;
float x16846 = x16729[x16845];
float x16847 = x16844 + x16846;
x16187[x16843] = x16847;
x16839 += 1;
if (x16850) {
x16840 += 1;
} else {
}
if (x16854) {
x16841 += 1;
} else {
}

}
x16831 += x16812;
if (x16850) {
x16832 += x16174;
} else {
}
if (x16854) {
x16833 += x16716;
} else {
}

}
x16823 += x16813;
if (x16870) {
x16824 += x16175;
} else {
}
if (x16874) {
x16825 += x16717;
} else {
}

}
x16816 += x16814;
x16817 += x16176;
x16818 += x16718;

}
float* x16885 = (float*)myMalloc(x16177 * sizeof(float));;
for(int x16887=0; x16887 < x16177; x16887++) {
float x16888 = x16187[x16887];
bool x16889 = x16888 < 0.0f;
if (x16889) {
x16885[x16887] = 0.0f;
} else {
float x16892 = x16187[x16887];
x16885[x16887] = x16892;
}

}
float* x16906 = (float*)myMalloc(x16905 * sizeof(float));;
int32_t x16909 = 64 * x16172;
int32_t x16910 = x16909 * x16901;
float* x16911 = (float*)myMalloc(x16910 * sizeof(float));;
int32_t x16907 = x16172 * x16901;
for(int x16912=0; x16912 < 64; x16912++) {
int32_t x16913 = x16912 * x16176;
float* x16914 = x16885+x16913;
int32_t x16915 = x16912 * x16902;
float* x16916 = x16906+x16915;
int32_t x16917 = x16912 * x16907;
float* x16918 = x16911+x16917;
for(int x16919=0; x16919 < x16172; x16919++) {
int32_t x16920 = x16919 / 1;
int32_t x16924 = x16920 * x16900;
int32_t x16925 = x16924 * x16900;
int32_t x16921 = x16919 % 1;
int32_t x16922 = x16921 / 1;
int32_t x16926 = x16922 * x16900;
int32_t x16927 = x16926 * x16900;
int32_t x16928 = x16925 + x16927;
int32_t x16923 = x16921 % 1;
int32_t x16929 = x16923 * x16900;
int32_t x16930 = x16929 * x16900;
int32_t x16931 = x16928 + x16930;
float* x16932 = x16918+x16931;
int32_t x16933 = x16920 * x16174;
int32_t x16934 = x16933 * x16174;
float* x16935 = x16914+x16934;
for(int x16937=0; x16937 < x16900; x16937++) {
int32_t x16939 = x16937 * x16900;
float* x16940 = x16932+x16939;
int32_t x16938 = x16937 + x16922;
int32_t x16941 = x16938 * x16174;
int32_t x16942 = x16941 + x16923;
float* x16943 = x16935+x16942;
memcpy(x16940, x16943, 4 * x16900);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x16901,x16172,1,x13,x16172,x16918,x16901,1,x16916,x16901);

}
int32_t x16952 = 0;
int32_t x16953 = 1;
x16953 *= 1;
x16952 += 1;
x16953 *= 1;
x16953 *= 1;
int32_t x16958 = x16952;
bool x16959 = x16958 >= 2;
if (x16959) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x16964 = x16958 == 0;
if (x16964) {
int32_t x16965 = x16953;
bool x16966 = x16965 == 256;
if (x16966) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x16973 = x16953;
int32_t x16974 = 256 / x16973;
bool x16978;
if (x452) {
bool x16975 = x16974 == 1;
bool x16976 = 256 == x16974;
bool x16977 = x16975 || x16976;
x16978 = x16977;
} else {
x16978 = false;
}
bool x16982;
if (x16978) {
x16982 = x16981;
} else {
x16982 = false;
}
bool x16983;
if (x16982) {
x16983 = x16981;
} else {
x16983 = false;
}
if (x16983) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x16900,x16900,1,x16974,1,1);
assert(false && "");
}
bool x16989 = 256 <= x16974;
int32_t x16990;
if (x16989) {
x16990 = x16974;
} else {
x16990 = 256;
}
bool x16996 = x16990 > 0;
bool x16998;
if (x16996) {
x16998 = x16997;
} else {
x16998 = false;
}
bool x16999;
if (x16998) {
x16999 = x16997;
} else {
x16999 = false;
}
if (x16999) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(16900) x Sym(16900)"," x Const(1) x Sym(16974) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x16994 = x16990 * x16993;
int32_t x16995 = 64 * x16994;
float* x17005 = (float*)myMalloc(x16995 * sizeof(float));;
int32_t x17006 = 0;
int32_t x17007 = 0;
int32_t x17008 = 0;
bool x17055 = x16974 > 1;
for(int x17009=0; x17009 < 64; x17009++) {
int32_t x17010 = x17007;
int32_t x17011 = x17008;
int32_t x17012 = x17006;
int32_t x17013 = x17012;
int32_t x17014 = x17010;
int32_t x17015 = x17011;
for(int x17017=0; x17017 < x16990; x17017++) {
int32_t x17018 = x17014;
int32_t x17019 = x17015;
int32_t x17020 = x17013;
int32_t x17021 = x17020;
int32_t x17022 = x17018;
int32_t x17023 = x17019;
for(int x17025=0; x17025 < x16992; x17025++) {
int32_t x17026 = x17022;
int32_t x17027 = x17023;
int32_t x17028 = x17021;
int32_t x17029 = x17028;
int32_t x17030 = x17026;
int32_t x17031 = x17027;
for(int x17032=0; x17032 < x16992; x17032++) {
int32_t x17033 = x17029;
int32_t x17034 = x17030;
float x17035 = x16906[x17034];
int32_t x17036 = x17031;
float x17037 = x259[x17036];
float x17038 = x17035 - x17037;
x17005[x17033] = x17038;
x17029 += 1;
if (x17041) {
x17030 += 1;
} else {
}

}
x17021 += x16992;
if (x17041) {
x17022 += x16900;
} else {
}

}
x17013 += x16993;
x17014 += x16901;
if (x17055) {
x17015 += 1;
} else {
}

}
x17006 += x16994;
x17007 += x16902;

}
float* x17065 = (float*)myMalloc(256 * sizeof(float));;
for(int x17066=0; x17066 < 256; x17066++) {
float x17067 = x157[x17066];
float x17068 = x17067 + 1.0E-5f;
x17065[x17066] = x17068;

}
float* x17072 = (float*)myMalloc(256 * sizeof(float));;
for(int x17073=0; x17073 < 256; x17073++) {
float x17074 = x17065[x17073];
double x17075 = (double)x17074;
double x17076 = sqrt(x17075);
float x17077 = (float)x17076;
x17072[x17073] = x17077;

}
int32_t x17081 = 0;
int32_t x17082 = 1;
x17082 *= 1;
x17081 += 1;
x17082 *= 1;
x17082 *= 1;
int32_t x17087 = x17081;
bool x17088 = x17087 >= 2;
if (x17088) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17093 = x17087 == 0;
if (x17093) {
int32_t x17094 = x17082;
bool x17095 = x17094 == 256;
if (x17095) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17102 = x17082;
int32_t x17103 = 256 / x17102;
bool x17109;
if (x452) {
bool x17104 = x16990 == 1;
bool x17105 = x17103 == 1;
bool x17106 = x17104 || x17105;
bool x17107 = x16990 == x17103;
bool x17108 = x17106 || x17107;
x17109 = x17108;
} else {
x17109 = false;
}
bool x17113;
if (x17109) {
x17113 = x17112;
} else {
x17113 = false;
}
bool x17114;
if (x17113) {
x17114 = x17112;
} else {
x17114 = false;
}
if (x17114) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x16990,x16992,x16992,1,x17103,1,1);
assert(false && "");
}
bool x17120 = x16990 <= x17103;
int32_t x17121;
if (x17120) {
x17121 = x17103;
} else {
x17121 = x16990;
}
bool x17127 = x17121 > 0;
bool x17129;
if (x17127) {
x17129 = x17128;
} else {
x17129 = false;
}
bool x17130;
if (x17129) {
x17130 = x17128;
} else {
x17130 = false;
}
if (x17130) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(16990) x Sym(16992) x Sym(16992)"," x Const(1) x Sym(17103) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x17125 = x17121 * x17124;
int32_t x17126 = 64 * x17125;
float* x17136 = (float*)myMalloc(x17126 * sizeof(float));;
int32_t x17137 = 0;
int32_t x17138 = 0;
int32_t x17139 = 0;
bool x17185 = x16990 > 1;
bool x17189 = x17103 > 1;
for(int x17140=0; x17140 < 64; x17140++) {
int32_t x17141 = x17138;
int32_t x17142 = x17139;
int32_t x17143 = x17137;
int32_t x17144 = x17143;
int32_t x17145 = x17141;
int32_t x17146 = x17142;
for(int x17148=0; x17148 < x17121; x17148++) {
int32_t x17149 = x17145;
int32_t x17150 = x17146;
int32_t x17151 = x17144;
int32_t x17152 = x17151;
int32_t x17153 = x17149;
int32_t x17154 = x17150;
for(int x17156=0; x17156 < x17123; x17156++) {
int32_t x17157 = x17153;
int32_t x17158 = x17154;
int32_t x17159 = x17152;
int32_t x17160 = x17159;
int32_t x17161 = x17157;
int32_t x17162 = x17158;
for(int x17163=0; x17163 < x17123; x17163++) {
int32_t x17164 = x17160;
int32_t x17165 = x17161;
float x17166 = x17005[x17165];
int32_t x17167 = x17162;
float x17168 = x17072[x17167];
float x17169 = x17166 / x17168;
x17136[x17164] = x17169;
x17160 += 1;
if (x17172) {
x17161 += 1;
} else {
}

}
x17152 += x17123;
if (x17172) {
x17153 += x16992;
} else {
}

}
x17144 += x17124;
if (x17185) {
x17145 += x16993;
} else {
}
if (x17189) {
x17146 += 1;
} else {
}

}
x17137 += x17125;
x17138 += x16994;

}
int32_t x17199 = 0;
int32_t x17200 = 1;
x17200 *= 1;
x17199 += 1;
x17200 *= 1;
x17200 *= 1;
int32_t x17205 = x17199;
bool x17206 = x17205 >= 2;
if (x17206) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17211 = x17205 == 0;
if (x17211) {
int32_t x17212 = x17200;
bool x17213 = x17212 == 256;
if (x17213) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17220 = x17200;
int32_t x17221 = 256 / x17220;
bool x17227;
if (x452) {
bool x17222 = x17121 == 1;
bool x17223 = x17221 == 1;
bool x17224 = x17222 || x17223;
bool x17225 = x17121 == x17221;
bool x17226 = x17224 || x17225;
x17227 = x17226;
} else {
x17227 = false;
}
bool x17231;
if (x17227) {
x17231 = x17230;
} else {
x17231 = false;
}
bool x17232;
if (x17231) {
x17232 = x17230;
} else {
x17232 = false;
}
if (x17232) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17121,x17123,x17123,1,x17221,1,1);
assert(false && "");
}
bool x17238 = x17121 <= x17221;
int32_t x17239;
if (x17238) {
x17239 = x17221;
} else {
x17239 = x17121;
}
bool x17245 = x17239 > 0;
bool x17247;
if (x17245) {
x17247 = x17246;
} else {
x17247 = false;
}
bool x17248;
if (x17247) {
x17248 = x17246;
} else {
x17248 = false;
}
if (x17248) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(17121) x Sym(17123) x Sym(17123)"," x Const(1) x Sym(17221) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x17243 = x17239 * x17242;
int32_t x17244 = 64 * x17243;
float* x17254 = (float*)myMalloc(x17244 * sizeof(float));;
int32_t x17255 = 0;
int32_t x17256 = 0;
int32_t x17257 = 0;
bool x17303 = x17121 > 1;
bool x17307 = x17221 > 1;
for(int x17258=0; x17258 < 64; x17258++) {
int32_t x17259 = x17256;
int32_t x17260 = x17257;
int32_t x17261 = x17255;
int32_t x17262 = x17261;
int32_t x17263 = x17259;
int32_t x17264 = x17260;
for(int x17266=0; x17266 < x17239; x17266++) {
int32_t x17267 = x17263;
int32_t x17268 = x17264;
int32_t x17269 = x17262;
int32_t x17270 = x17269;
int32_t x17271 = x17267;
int32_t x17272 = x17268;
for(int x17274=0; x17274 < x17241; x17274++) {
int32_t x17275 = x17271;
int32_t x17276 = x17272;
int32_t x17277 = x17270;
int32_t x17278 = x17277;
int32_t x17279 = x17275;
int32_t x17280 = x17276;
for(int x17281=0; x17281 < x17241; x17281++) {
int32_t x17282 = x17278;
int32_t x17283 = x17279;
float x17284 = x17136[x17283];
int32_t x17285 = x17280;
float x17286 = x30[x17285];
float x17287 = x17284 * x17286;
x17254[x17282] = x17287;
x17278 += 1;
if (x17290) {
x17279 += 1;
} else {
}

}
x17270 += x17241;
if (x17290) {
x17271 += x17123;
} else {
}

}
x17262 += x17242;
if (x17303) {
x17263 += x17124;
} else {
}
if (x17307) {
x17264 += 1;
} else {
}

}
x17255 += x17243;
x17256 += x17125;

}
int32_t x17317 = 0;
int32_t x17318 = 1;
x17318 *= 1;
x17317 += 1;
x17318 *= 1;
x17318 *= 1;
int32_t x17323 = x17317;
bool x17324 = x17323 >= 2;
if (x17324) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17329 = x17323 == 0;
if (x17329) {
int32_t x17330 = x17318;
bool x17331 = x17330 == 256;
if (x17331) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17338 = x17318;
int32_t x17339 = 256 / x17338;
bool x17345;
if (x452) {
bool x17340 = x17239 == 1;
bool x17341 = x17339 == 1;
bool x17342 = x17340 || x17341;
bool x17343 = x17239 == x17339;
bool x17344 = x17342 || x17343;
x17345 = x17344;
} else {
x17345 = false;
}
bool x17349;
if (x17345) {
x17349 = x17348;
} else {
x17349 = false;
}
bool x17350;
if (x17349) {
x17350 = x17348;
} else {
x17350 = false;
}
if (x17350) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17239,x17241,x17241,1,x17339,1,1);
assert(false && "");
}
bool x17356 = x17239 <= x17339;
int32_t x17357;
if (x17356) {
x17357 = x17339;
} else {
x17357 = x17239;
}
bool x17363 = x17357 > 0;
bool x17365;
if (x17363) {
x17365 = x17364;
} else {
x17365 = false;
}
bool x17366;
if (x17365) {
x17366 = x17364;
} else {
x17366 = false;
}
if (x17366) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(17239) x Sym(17241) x Sym(17241)"," x Const(1) x Sym(17339) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x17361 = x17357 * x17360;
int32_t x17362 = 64 * x17361;
float* x17372 = (float*)myMalloc(x17362 * sizeof(float));;
int32_t x17373 = 0;
int32_t x17374 = 0;
int32_t x17375 = 0;
bool x17421 = x17239 > 1;
bool x17425 = x17339 > 1;
for(int x17376=0; x17376 < 64; x17376++) {
int32_t x17377 = x17374;
int32_t x17378 = x17375;
int32_t x17379 = x17373;
int32_t x17380 = x17379;
int32_t x17381 = x17377;
int32_t x17382 = x17378;
for(int x17384=0; x17384 < x17357; x17384++) {
int32_t x17385 = x17381;
int32_t x17386 = x17382;
int32_t x17387 = x17380;
int32_t x17388 = x17387;
int32_t x17389 = x17385;
int32_t x17390 = x17386;
for(int x17392=0; x17392 < x17359; x17392++) {
int32_t x17393 = x17389;
int32_t x17394 = x17390;
int32_t x17395 = x17388;
int32_t x17396 = x17395;
int32_t x17397 = x17393;
int32_t x17398 = x17394;
for(int x17399=0; x17399 < x17359; x17399++) {
int32_t x17400 = x17396;
int32_t x17401 = x17397;
float x17402 = x17254[x17401];
int32_t x17403 = x17398;
float x17404 = x219[x17403];
float x17405 = x17402 + x17404;
x17372[x17400] = x17405;
x17396 += 1;
if (x17408) {
x17397 += 1;
} else {
}

}
x17388 += x17359;
if (x17408) {
x17389 += x17241;
} else {
}

}
x17380 += x17360;
if (x17421) {
x17381 += x17242;
} else {
}
if (x17425) {
x17382 += 1;
} else {
}

}
x17373 += x17361;
x17374 += x17243;

}
float* x17435 = (float*)myMalloc(x17362 * sizeof(float));;
for(int x17437=0; x17437 < x17362; x17437++) {
float x17438 = x17372[x17437];
bool x17439 = x17438 < 0.0f;
if (x17439) {
x17435[x17437] = 0.0f;
} else {
float x17442 = x17372[x17437];
x17435[x17437] = x17442;
}

}
float* x17457 = (float*)myMalloc(x17456 * sizeof(float));;
int32_t x17458 = 9 * x17357;
int32_t x17461 = 64 * x17458;
int32_t x17462 = x17461 * x17452;
float* x17463 = (float*)myMalloc(x17462 * sizeof(float));;
int32_t x17459 = x17458 * x17452;
int32_t x17471 = x17357 * 3;
int32_t x17472 = x17471 * 3;
for(int x17464=0; x17464 < 64; x17464++) {
int32_t x17465 = x17464 * x17361;
float* x17466 = x17435+x17465;
int32_t x17467 = x17464 * x17453;
float* x17468 = x17457+x17467;
int32_t x17469 = x17464 * x17459;
float* x17470 = x17463+x17469;
for(int x17474=0; x17474 < x17472; x17474++) {
int32_t x17475 = x17474 / 9;
int32_t x17479 = x17475 * 3;
int32_t x17480 = x17479 * 3;
int32_t x17481 = x17480 * x17451;
int32_t x17482 = x17481 * x17451;
int32_t x17476 = x17474 % 9;
int32_t x17477 = x17476 / 3;
int32_t x17483 = x17477 * 3;
int32_t x17484 = x17483 * x17451;
int32_t x17485 = x17484 * x17451;
int32_t x17486 = x17482 + x17485;
int32_t x17478 = x17476 % 3;
int32_t x17487 = x17478 * x17451;
int32_t x17488 = x17487 * x17451;
int32_t x17489 = x17486 + x17488;
float* x17490 = x17470+x17489;
int32_t x17491 = x17475 * x17359;
int32_t x17492 = x17491 * x17359;
float* x17493 = x17466+x17492;
int32_t x17506 = 1 - x17478;
bool x17507 = x17506 > 0;
int32_t x17508;
if (x17507) {
x17508 = x17506;
} else {
x17508 = 0;
}
int32_t x17509 = 3 - x17478;
int32_t x17510 = x17509 - 1;
int32_t x17511 = 1 - x17510;
bool x17512 = x17511 > 0;
int32_t x17513;
if (x17512) {
x17513 = x17511;
} else {
x17513 = 0;
}
int32_t x17514 = x17451 - x17513;
int32_t x17515 = x17514 - x17508;
bool x17516 = x17515 <= 0;
bool x17520 = x17508 > 0;
int32_t x17505 = -1 + x17478;
bool x17533 = x17513 > 0;
for(int x17495=0; x17495 < x17451; x17495++) {
int32_t x17496 = x17495 - 1;
int32_t x17497 = x17496 + x17477;
bool x17498 = x17497 < 0;
bool x17499 = x17497 >= x17359;
bool x17500 = x17498 || x17499;
if (x17500) {
int32_t x17501 = x17495 * x17451;
float* x17502 = x17490+x17501;
memset(x17502, 0, 4 * x17451);;
} else {
if (x17516) {
int32_t x17501 = x17495 * x17451;
float* x17517 = x17490+x17501;
memset(x17517, 0, 4 * x17451);;
} else {
int32_t x17501 = x17495 * x17451;
if (x17520) {
float* x17521 = x17490+x17501;
memset(x17521, 0, 4 * x17508);;
} else {
}
// may have segfault here
int32_t x17526 = x17501 + x17508;
float* x17527 = x17490+x17526;
int32_t x17528 = x17497 * x17359;
int32_t x17529 = x17528 + x17505;
int32_t x17530 = x17529 + x17508;
float* x17531 = x17493+x17530;
memcpy(x17527, x17531, 4 * x17515);;
if (x17533) {
int32_t x17534 = x17501 + x17451;
int32_t x17535 = x17534 - x17513;
float* x17536 = x17490+x17535;
memset(x17536, 0, 4 * x17513);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x17452,x17458,1,x31,x17458,x17470,x17452,1,x17468,x17452);

}
int32_t x17551 = 0;
int32_t x17552 = 1;
x17552 *= 1;
x17551 += 1;
x17552 *= 1;
x17552 *= 1;
int32_t x17557 = x17551;
bool x17558 = x17557 >= 2;
if (x17558) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17563 = x17557 == 0;
if (x17563) {
int32_t x17564 = x17552;
bool x17565 = x17564 == 256;
if (x17565) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17572 = x17552;
int32_t x17573 = 256 / x17572;
bool x17577;
if (x452) {
bool x17574 = x17573 == 1;
bool x17575 = 256 == x17573;
bool x17576 = x17574 || x17575;
x17577 = x17576;
} else {
x17577 = false;
}
bool x17581;
if (x17577) {
x17581 = x17580;
} else {
x17581 = false;
}
bool x17582;
if (x17581) {
x17582 = x17580;
} else {
x17582 = false;
}
if (x17582) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x17451,x17451,1,x17573,1,1);
assert(false && "");
}
bool x17588 = 256 <= x17573;
int32_t x17589;
if (x17588) {
x17589 = x17573;
} else {
x17589 = 256;
}
bool x17595 = x17589 > 0;
bool x17597;
if (x17595) {
x17597 = x17596;
} else {
x17597 = false;
}
bool x17598;
if (x17597) {
x17598 = x17596;
} else {
x17598 = false;
}
if (x17598) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(17451) x Sym(17451)"," x Const(1) x Sym(17573) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x17593 = x17589 * x17592;
int32_t x17594 = 64 * x17593;
float* x17604 = (float*)myMalloc(x17594 * sizeof(float));;
int32_t x17605 = 0;
int32_t x17606 = 0;
int32_t x17607 = 0;
bool x17654 = x17573 > 1;
for(int x17608=0; x17608 < 64; x17608++) {
int32_t x17609 = x17606;
int32_t x17610 = x17607;
int32_t x17611 = x17605;
int32_t x17612 = x17611;
int32_t x17613 = x17609;
int32_t x17614 = x17610;
for(int x17616=0; x17616 < x17589; x17616++) {
int32_t x17617 = x17613;
int32_t x17618 = x17614;
int32_t x17619 = x17612;
int32_t x17620 = x17619;
int32_t x17621 = x17617;
int32_t x17622 = x17618;
for(int x17624=0; x17624 < x17591; x17624++) {
int32_t x17625 = x17621;
int32_t x17626 = x17622;
int32_t x17627 = x17620;
int32_t x17628 = x17627;
int32_t x17629 = x17625;
int32_t x17630 = x17626;
for(int x17631=0; x17631 < x17591; x17631++) {
int32_t x17632 = x17628;
int32_t x17633 = x17629;
float x17634 = x17457[x17633];
int32_t x17635 = x17630;
float x17636 = x200[x17635];
float x17637 = x17634 - x17636;
x17604[x17632] = x17637;
x17628 += 1;
if (x17640) {
x17629 += 1;
} else {
}

}
x17620 += x17591;
if (x17640) {
x17621 += x17451;
} else {
}

}
x17612 += x17592;
x17613 += x17452;
if (x17654) {
x17614 += 1;
} else {
}

}
x17605 += x17593;
x17606 += x17453;

}
float* x17664 = (float*)myMalloc(256 * sizeof(float));;
for(int x17665=0; x17665 < 256; x17665++) {
float x17666 = x237[x17665];
float x17667 = x17666 + 1.0E-5f;
x17664[x17665] = x17667;

}
float* x17671 = (float*)myMalloc(256 * sizeof(float));;
for(int x17672=0; x17672 < 256; x17672++) {
float x17673 = x17664[x17672];
double x17674 = (double)x17673;
double x17675 = sqrt(x17674);
float x17676 = (float)x17675;
x17671[x17672] = x17676;

}
int32_t x17680 = 0;
int32_t x17681 = 1;
x17681 *= 1;
x17680 += 1;
x17681 *= 1;
x17681 *= 1;
int32_t x17686 = x17680;
bool x17687 = x17686 >= 2;
if (x17687) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17692 = x17686 == 0;
if (x17692) {
int32_t x17693 = x17681;
bool x17694 = x17693 == 256;
if (x17694) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17701 = x17681;
int32_t x17702 = 256 / x17701;
bool x17708;
if (x452) {
bool x17703 = x17589 == 1;
bool x17704 = x17702 == 1;
bool x17705 = x17703 || x17704;
bool x17706 = x17589 == x17702;
bool x17707 = x17705 || x17706;
x17708 = x17707;
} else {
x17708 = false;
}
bool x17712;
if (x17708) {
x17712 = x17711;
} else {
x17712 = false;
}
bool x17713;
if (x17712) {
x17713 = x17711;
} else {
x17713 = false;
}
if (x17713) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17589,x17591,x17591,1,x17702,1,1);
assert(false && "");
}
bool x17719 = x17589 <= x17702;
int32_t x17720;
if (x17719) {
x17720 = x17702;
} else {
x17720 = x17589;
}
bool x17726 = x17720 > 0;
bool x17728;
if (x17726) {
x17728 = x17727;
} else {
x17728 = false;
}
bool x17729;
if (x17728) {
x17729 = x17727;
} else {
x17729 = false;
}
if (x17729) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(17589) x Sym(17591) x Sym(17591)"," x Const(1) x Sym(17702) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x17724 = x17720 * x17723;
int32_t x17725 = 64 * x17724;
float* x17735 = (float*)myMalloc(x17725 * sizeof(float));;
int32_t x17736 = 0;
int32_t x17737 = 0;
int32_t x17738 = 0;
bool x17784 = x17589 > 1;
bool x17788 = x17702 > 1;
for(int x17739=0; x17739 < 64; x17739++) {
int32_t x17740 = x17737;
int32_t x17741 = x17738;
int32_t x17742 = x17736;
int32_t x17743 = x17742;
int32_t x17744 = x17740;
int32_t x17745 = x17741;
for(int x17747=0; x17747 < x17720; x17747++) {
int32_t x17748 = x17744;
int32_t x17749 = x17745;
int32_t x17750 = x17743;
int32_t x17751 = x17750;
int32_t x17752 = x17748;
int32_t x17753 = x17749;
for(int x17755=0; x17755 < x17722; x17755++) {
int32_t x17756 = x17752;
int32_t x17757 = x17753;
int32_t x17758 = x17751;
int32_t x17759 = x17758;
int32_t x17760 = x17756;
int32_t x17761 = x17757;
for(int x17762=0; x17762 < x17722; x17762++) {
int32_t x17763 = x17759;
int32_t x17764 = x17760;
float x17765 = x17604[x17764];
int32_t x17766 = x17761;
float x17767 = x17671[x17766];
float x17768 = x17765 / x17767;
x17735[x17763] = x17768;
x17759 += 1;
if (x17771) {
x17760 += 1;
} else {
}

}
x17751 += x17722;
if (x17771) {
x17752 += x17591;
} else {
}

}
x17743 += x17723;
if (x17784) {
x17744 += x17592;
} else {
}
if (x17788) {
x17745 += 1;
} else {
}

}
x17736 += x17724;
x17737 += x17593;

}
int32_t x17798 = 0;
int32_t x17799 = 1;
x17799 *= 1;
x17798 += 1;
x17799 *= 1;
x17799 *= 1;
int32_t x17804 = x17798;
bool x17805 = x17804 >= 2;
if (x17805) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17810 = x17804 == 0;
if (x17810) {
int32_t x17811 = x17799;
bool x17812 = x17811 == 256;
if (x17812) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17819 = x17799;
int32_t x17820 = 256 / x17819;
bool x17826;
if (x452) {
bool x17821 = x17720 == 1;
bool x17822 = x17820 == 1;
bool x17823 = x17821 || x17822;
bool x17824 = x17720 == x17820;
bool x17825 = x17823 || x17824;
x17826 = x17825;
} else {
x17826 = false;
}
bool x17830;
if (x17826) {
x17830 = x17829;
} else {
x17830 = false;
}
bool x17831;
if (x17830) {
x17831 = x17829;
} else {
x17831 = false;
}
if (x17831) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17720,x17722,x17722,1,x17820,1,1);
assert(false && "");
}
bool x17837 = x17720 <= x17820;
int32_t x17838;
if (x17837) {
x17838 = x17820;
} else {
x17838 = x17720;
}
bool x17844 = x17838 > 0;
bool x17846;
if (x17844) {
x17846 = x17845;
} else {
x17846 = false;
}
bool x17847;
if (x17846) {
x17847 = x17845;
} else {
x17847 = false;
}
if (x17847) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(17720) x Sym(17722) x Sym(17722)"," x Const(1) x Sym(17820) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x17842 = x17838 * x17841;
int32_t x17843 = 64 * x17842;
float* x17853 = (float*)myMalloc(x17843 * sizeof(float));;
int32_t x17854 = 0;
int32_t x17855 = 0;
int32_t x17856 = 0;
bool x17902 = x17720 > 1;
bool x17906 = x17820 > 1;
for(int x17857=0; x17857 < 64; x17857++) {
int32_t x17858 = x17855;
int32_t x17859 = x17856;
int32_t x17860 = x17854;
int32_t x17861 = x17860;
int32_t x17862 = x17858;
int32_t x17863 = x17859;
for(int x17865=0; x17865 < x17838; x17865++) {
int32_t x17866 = x17862;
int32_t x17867 = x17863;
int32_t x17868 = x17861;
int32_t x17869 = x17868;
int32_t x17870 = x17866;
int32_t x17871 = x17867;
for(int x17873=0; x17873 < x17840; x17873++) {
int32_t x17874 = x17870;
int32_t x17875 = x17871;
int32_t x17876 = x17869;
int32_t x17877 = x17876;
int32_t x17878 = x17874;
int32_t x17879 = x17875;
for(int x17880=0; x17880 < x17840; x17880++) {
int32_t x17881 = x17877;
int32_t x17882 = x17878;
float x17883 = x17735[x17882];
int32_t x17884 = x17879;
float x17885 = x271[x17884];
float x17886 = x17883 * x17885;
x17853[x17881] = x17886;
x17877 += 1;
if (x17889) {
x17878 += 1;
} else {
}

}
x17869 += x17840;
if (x17889) {
x17870 += x17722;
} else {
}

}
x17861 += x17841;
if (x17902) {
x17862 += x17723;
} else {
}
if (x17906) {
x17863 += 1;
} else {
}

}
x17854 += x17842;
x17855 += x17724;

}
int32_t x17916 = 0;
int32_t x17917 = 1;
x17917 *= 1;
x17916 += 1;
x17917 *= 1;
x17917 *= 1;
int32_t x17922 = x17916;
bool x17923 = x17922 >= 2;
if (x17923) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x17928 = x17922 == 0;
if (x17928) {
int32_t x17929 = x17917;
bool x17930 = x17929 == 256;
if (x17930) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x17937 = x17917;
int32_t x17938 = 256 / x17937;
bool x17944;
if (x452) {
bool x17939 = x17838 == 1;
bool x17940 = x17938 == 1;
bool x17941 = x17939 || x17940;
bool x17942 = x17838 == x17938;
bool x17943 = x17941 || x17942;
x17944 = x17943;
} else {
x17944 = false;
}
bool x17948;
if (x17944) {
x17948 = x17947;
} else {
x17948 = false;
}
bool x17949;
if (x17948) {
x17949 = x17947;
} else {
x17949 = false;
}
if (x17949) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x17838,x17840,x17840,1,x17938,1,1);
assert(false && "");
}
bool x17955 = x17838 <= x17938;
int32_t x17956;
if (x17955) {
x17956 = x17938;
} else {
x17956 = x17838;
}
bool x17962 = x17956 > 0;
bool x17964;
if (x17962) {
x17964 = x17963;
} else {
x17964 = false;
}
bool x17965;
if (x17964) {
x17965 = x17963;
} else {
x17965 = false;
}
if (x17965) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(17838) x Sym(17840) x Sym(17840)"," x Const(1) x Sym(17938) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x17960 = x17956 * x17959;
int32_t x17961 = 64 * x17960;
float* x17971 = (float*)myMalloc(x17961 * sizeof(float));;
int32_t x17972 = 0;
int32_t x17973 = 0;
int32_t x17974 = 0;
bool x18020 = x17838 > 1;
bool x18024 = x17938 > 1;
for(int x17975=0; x17975 < 64; x17975++) {
int32_t x17976 = x17973;
int32_t x17977 = x17974;
int32_t x17978 = x17972;
int32_t x17979 = x17978;
int32_t x17980 = x17976;
int32_t x17981 = x17977;
for(int x17983=0; x17983 < x17956; x17983++) {
int32_t x17984 = x17980;
int32_t x17985 = x17981;
int32_t x17986 = x17979;
int32_t x17987 = x17986;
int32_t x17988 = x17984;
int32_t x17989 = x17985;
for(int x17991=0; x17991 < x17958; x17991++) {
int32_t x17992 = x17988;
int32_t x17993 = x17989;
int32_t x17994 = x17987;
int32_t x17995 = x17994;
int32_t x17996 = x17992;
int32_t x17997 = x17993;
for(int x17998=0; x17998 < x17958; x17998++) {
int32_t x17999 = x17995;
int32_t x18000 = x17996;
float x18001 = x17853[x18000];
int32_t x18002 = x17997;
float x18003 = x96[x18002];
float x18004 = x18001 + x18003;
x17971[x17999] = x18004;
x17995 += 1;
if (x18007) {
x17996 += 1;
} else {
}

}
x17987 += x17958;
if (x18007) {
x17988 += x17840;
} else {
}

}
x17979 += x17959;
if (x18020) {
x17980 += x17841;
} else {
}
if (x18024) {
x17981 += 1;
} else {
}

}
x17972 += x17960;
x17973 += x17842;

}
float* x18034 = (float*)myMalloc(x17961 * sizeof(float));;
for(int x18036=0; x18036 < x17961; x18036++) {
float x18037 = x17971[x18036];
bool x18038 = x18037 < 0.0f;
if (x18038) {
x18034[x18036] = 0.0f;
} else {
float x18041 = x17971[x18036];
x18034[x18036] = x18041;
}

}
float* x18055 = (float*)myMalloc(x18054 * sizeof(float));;
int32_t x18058 = 64 * x17956;
int32_t x18059 = x18058 * x18050;
float* x18060 = (float*)myMalloc(x18059 * sizeof(float));;
int32_t x18056 = x17956 * x18050;
for(int x18061=0; x18061 < 64; x18061++) {
int32_t x18062 = x18061 * x17960;
float* x18063 = x18034+x18062;
int32_t x18064 = x18061 * x18051;
float* x18065 = x18055+x18064;
int32_t x18066 = x18061 * x18056;
float* x18067 = x18060+x18066;
for(int x18068=0; x18068 < x17956; x18068++) {
int32_t x18069 = x18068 / 1;
int32_t x18073 = x18069 * x18049;
int32_t x18074 = x18073 * x18049;
int32_t x18070 = x18068 % 1;
int32_t x18071 = x18070 / 1;
int32_t x18075 = x18071 * x18049;
int32_t x18076 = x18075 * x18049;
int32_t x18077 = x18074 + x18076;
int32_t x18072 = x18070 % 1;
int32_t x18078 = x18072 * x18049;
int32_t x18079 = x18078 * x18049;
int32_t x18080 = x18077 + x18079;
float* x18081 = x18067+x18080;
int32_t x18082 = x18069 * x17958;
int32_t x18083 = x18082 * x17958;
float* x18084 = x18063+x18083;
for(int x18086=0; x18086 < x18049; x18086++) {
int32_t x18088 = x18086 * x18049;
float* x18089 = x18081+x18088;
int32_t x18087 = x18086 + x18071;
int32_t x18090 = x18087 * x17958;
int32_t x18091 = x18090 + x18072;
float* x18092 = x18084+x18091;
memcpy(x18089, x18092, 4 * x18049);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x18050,x17956,1,x56,x17956,x18067,x18050,1,x18065,x18050);

}
int32_t x18101 = 0;
int32_t x18102 = 1;
x18102 *= 1;
x18101 += 1;
x18102 *= 1;
x18102 *= 1;
int32_t x18107 = x18101;
bool x18108 = x18107 >= 2;
if (x18108) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18113 = x18107 == 0;
if (x18113) {
int32_t x18114 = x18102;
bool x18115 = x18114 == 1024;
if (x18115) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18122 = x18102;
int32_t x18123 = 1024 / x18122;
bool x18127;
if (x452) {
bool x18124 = x18123 == 1;
bool x18125 = 1024 == x18123;
bool x18126 = x18124 || x18125;
x18127 = x18126;
} else {
x18127 = false;
}
bool x18131;
if (x18127) {
x18131 = x18130;
} else {
x18131 = false;
}
bool x18132;
if (x18131) {
x18132 = x18130;
} else {
x18132 = false;
}
if (x18132) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x18049,x18049,1,x18123,1,1);
assert(false && "");
}
bool x18138 = 1024 <= x18123;
int32_t x18139;
if (x18138) {
x18139 = x18123;
} else {
x18139 = 1024;
}
bool x18145 = x18139 > 0;
bool x18147;
if (x18145) {
x18147 = x18146;
} else {
x18147 = false;
}
bool x18148;
if (x18147) {
x18148 = x18146;
} else {
x18148 = false;
}
if (x18148) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(1024) x Sym(18049) x Sym(18049)"," x Const(1) x Sym(18123) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x18143 = x18139 * x18142;
int32_t x18144 = 64 * x18143;
float* x18154 = (float*)myMalloc(x18144 * sizeof(float));;
int32_t x18155 = 0;
int32_t x18156 = 0;
int32_t x18157 = 0;
bool x18204 = x18123 > 1;
for(int x18158=0; x18158 < 64; x18158++) {
int32_t x18159 = x18156;
int32_t x18160 = x18157;
int32_t x18161 = x18155;
int32_t x18162 = x18161;
int32_t x18163 = x18159;
int32_t x18164 = x18160;
for(int x18166=0; x18166 < x18139; x18166++) {
int32_t x18167 = x18163;
int32_t x18168 = x18164;
int32_t x18169 = x18162;
int32_t x18170 = x18169;
int32_t x18171 = x18167;
int32_t x18172 = x18168;
for(int x18174=0; x18174 < x18141; x18174++) {
int32_t x18175 = x18171;
int32_t x18176 = x18172;
int32_t x18177 = x18170;
int32_t x18178 = x18177;
int32_t x18179 = x18175;
int32_t x18180 = x18176;
for(int x18181=0; x18181 < x18141; x18181++) {
int32_t x18182 = x18178;
int32_t x18183 = x18179;
float x18184 = x18055[x18183];
int32_t x18185 = x18180;
float x18186 = x182[x18185];
float x18187 = x18184 - x18186;
x18154[x18182] = x18187;
x18178 += 1;
if (x18190) {
x18179 += 1;
} else {
}

}
x18170 += x18141;
if (x18190) {
x18171 += x18049;
} else {
}

}
x18162 += x18142;
x18163 += x18050;
if (x18204) {
x18164 += 1;
} else {
}

}
x18155 += x18143;
x18156 += x18051;

}
float* x18214 = (float*)myMalloc(1024 * sizeof(float));;
for(int x18215=0; x18215 < 1024; x18215++) {
float x18216 = x143[x18215];
float x18217 = x18216 + 1.0E-5f;
x18214[x18215] = x18217;

}
float* x18221 = (float*)myMalloc(1024 * sizeof(float));;
for(int x18222=0; x18222 < 1024; x18222++) {
float x18223 = x18214[x18222];
double x18224 = (double)x18223;
double x18225 = sqrt(x18224);
float x18226 = (float)x18225;
x18221[x18222] = x18226;

}
int32_t x18230 = 0;
int32_t x18231 = 1;
x18231 *= 1;
x18230 += 1;
x18231 *= 1;
x18231 *= 1;
int32_t x18236 = x18230;
bool x18237 = x18236 >= 2;
if (x18237) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18242 = x18236 == 0;
if (x18242) {
int32_t x18243 = x18231;
bool x18244 = x18243 == 1024;
if (x18244) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18251 = x18231;
int32_t x18252 = 1024 / x18251;
bool x18258;
if (x452) {
bool x18253 = x18139 == 1;
bool x18254 = x18252 == 1;
bool x18255 = x18253 || x18254;
bool x18256 = x18139 == x18252;
bool x18257 = x18255 || x18256;
x18258 = x18257;
} else {
x18258 = false;
}
bool x18262;
if (x18258) {
x18262 = x18261;
} else {
x18262 = false;
}
bool x18263;
if (x18262) {
x18263 = x18261;
} else {
x18263 = false;
}
if (x18263) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18139,x18141,x18141,1,x18252,1,1);
assert(false && "");
}
bool x18269 = x18139 <= x18252;
int32_t x18270;
if (x18269) {
x18270 = x18252;
} else {
x18270 = x18139;
}
bool x18276 = x18270 > 0;
bool x18278;
if (x18276) {
x18278 = x18277;
} else {
x18278 = false;
}
bool x18279;
if (x18278) {
x18279 = x18277;
} else {
x18279 = false;
}
if (x18279) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(18139) x Sym(18141) x Sym(18141)"," x Const(1) x Sym(18252) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x18274 = x18270 * x18273;
int32_t x18275 = 64 * x18274;
float* x18285 = (float*)myMalloc(x18275 * sizeof(float));;
int32_t x18286 = 0;
int32_t x18287 = 0;
int32_t x18288 = 0;
bool x18334 = x18139 > 1;
bool x18338 = x18252 > 1;
for(int x18289=0; x18289 < 64; x18289++) {
int32_t x18290 = x18287;
int32_t x18291 = x18288;
int32_t x18292 = x18286;
int32_t x18293 = x18292;
int32_t x18294 = x18290;
int32_t x18295 = x18291;
for(int x18297=0; x18297 < x18270; x18297++) {
int32_t x18298 = x18294;
int32_t x18299 = x18295;
int32_t x18300 = x18293;
int32_t x18301 = x18300;
int32_t x18302 = x18298;
int32_t x18303 = x18299;
for(int x18305=0; x18305 < x18272; x18305++) {
int32_t x18306 = x18302;
int32_t x18307 = x18303;
int32_t x18308 = x18301;
int32_t x18309 = x18308;
int32_t x18310 = x18306;
int32_t x18311 = x18307;
for(int x18312=0; x18312 < x18272; x18312++) {
int32_t x18313 = x18309;
int32_t x18314 = x18310;
float x18315 = x18154[x18314];
int32_t x18316 = x18311;
float x18317 = x18221[x18316];
float x18318 = x18315 / x18317;
x18285[x18313] = x18318;
x18309 += 1;
if (x18321) {
x18310 += 1;
} else {
}

}
x18301 += x18272;
if (x18321) {
x18302 += x18141;
} else {
}

}
x18293 += x18273;
if (x18334) {
x18294 += x18142;
} else {
}
if (x18338) {
x18295 += 1;
} else {
}

}
x18286 += x18274;
x18287 += x18143;

}
int32_t x18348 = 0;
int32_t x18349 = 1;
x18349 *= 1;
x18348 += 1;
x18349 *= 1;
x18349 *= 1;
int32_t x18354 = x18348;
bool x18355 = x18354 >= 2;
if (x18355) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18360 = x18354 == 0;
if (x18360) {
int32_t x18361 = x18349;
bool x18362 = x18361 == 1024;
if (x18362) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18369 = x18349;
int32_t x18370 = 1024 / x18369;
bool x18376;
if (x452) {
bool x18371 = x18270 == 1;
bool x18372 = x18370 == 1;
bool x18373 = x18371 || x18372;
bool x18374 = x18270 == x18370;
bool x18375 = x18373 || x18374;
x18376 = x18375;
} else {
x18376 = false;
}
bool x18380;
if (x18376) {
x18380 = x18379;
} else {
x18380 = false;
}
bool x18381;
if (x18380) {
x18381 = x18379;
} else {
x18381 = false;
}
if (x18381) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18270,x18272,x18272,1,x18370,1,1);
assert(false && "");
}
bool x18387 = x18270 <= x18370;
int32_t x18388;
if (x18387) {
x18388 = x18370;
} else {
x18388 = x18270;
}
bool x18394 = x18388 > 0;
bool x18396;
if (x18394) {
x18396 = x18395;
} else {
x18396 = false;
}
bool x18397;
if (x18396) {
x18397 = x18395;
} else {
x18397 = false;
}
if (x18397) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(18270) x Sym(18272) x Sym(18272)"," x Const(1) x Sym(18370) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x18392 = x18388 * x18391;
int32_t x18393 = 64 * x18392;
float* x18403 = (float*)myMalloc(x18393 * sizeof(float));;
int32_t x18404 = 0;
int32_t x18405 = 0;
int32_t x18406 = 0;
bool x18452 = x18270 > 1;
bool x18456 = x18370 > 1;
for(int x18407=0; x18407 < 64; x18407++) {
int32_t x18408 = x18405;
int32_t x18409 = x18406;
int32_t x18410 = x18404;
int32_t x18411 = x18410;
int32_t x18412 = x18408;
int32_t x18413 = x18409;
for(int x18415=0; x18415 < x18388; x18415++) {
int32_t x18416 = x18412;
int32_t x18417 = x18413;
int32_t x18418 = x18411;
int32_t x18419 = x18418;
int32_t x18420 = x18416;
int32_t x18421 = x18417;
for(int x18423=0; x18423 < x18390; x18423++) {
int32_t x18424 = x18420;
int32_t x18425 = x18421;
int32_t x18426 = x18419;
int32_t x18427 = x18426;
int32_t x18428 = x18424;
int32_t x18429 = x18425;
for(int x18430=0; x18430 < x18390; x18430++) {
int32_t x18431 = x18427;
int32_t x18432 = x18428;
float x18433 = x18285[x18432];
int32_t x18434 = x18429;
float x18435 = x20[x18434];
float x18436 = x18433 * x18435;
x18403[x18431] = x18436;
x18427 += 1;
if (x18439) {
x18428 += 1;
} else {
}

}
x18419 += x18390;
if (x18439) {
x18420 += x18272;
} else {
}

}
x18411 += x18391;
if (x18452) {
x18412 += x18273;
} else {
}
if (x18456) {
x18413 += 1;
} else {
}

}
x18404 += x18392;
x18405 += x18274;

}
int32_t x18466 = 0;
int32_t x18467 = 1;
x18467 *= 1;
x18466 += 1;
x18467 *= 1;
x18467 *= 1;
int32_t x18472 = x18466;
bool x18473 = x18472 >= 2;
if (x18473) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18478 = x18472 == 0;
if (x18478) {
int32_t x18479 = x18467;
bool x18480 = x18479 == 1024;
if (x18480) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18487 = x18467;
int32_t x18488 = 1024 / x18487;
bool x18494;
if (x452) {
bool x18489 = x18388 == 1;
bool x18490 = x18488 == 1;
bool x18491 = x18489 || x18490;
bool x18492 = x18388 == x18488;
bool x18493 = x18491 || x18492;
x18494 = x18493;
} else {
x18494 = false;
}
bool x18498;
if (x18494) {
x18498 = x18497;
} else {
x18498 = false;
}
bool x18499;
if (x18498) {
x18499 = x18497;
} else {
x18499 = false;
}
if (x18499) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18388,x18390,x18390,1,x18488,1,1);
assert(false && "");
}
bool x18505 = x18388 <= x18488;
int32_t x18506;
if (x18505) {
x18506 = x18488;
} else {
x18506 = x18388;
}
bool x18512 = x18506 > 0;
bool x18514;
if (x18512) {
x18514 = x18513;
} else {
x18514 = false;
}
bool x18515;
if (x18514) {
x18515 = x18513;
} else {
x18515 = false;
}
if (x18515) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(18388) x Sym(18390) x Sym(18390)"," x Const(1) x Sym(18488) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x18510 = x18506 * x18509;
int32_t x18511 = 64 * x18510;
float* x18521 = (float*)myMalloc(x18511 * sizeof(float));;
int32_t x18522 = 0;
int32_t x18523 = 0;
int32_t x18524 = 0;
bool x18570 = x18388 > 1;
bool x18574 = x18488 > 1;
for(int x18525=0; x18525 < 64; x18525++) {
int32_t x18526 = x18523;
int32_t x18527 = x18524;
int32_t x18528 = x18522;
int32_t x18529 = x18528;
int32_t x18530 = x18526;
int32_t x18531 = x18527;
for(int x18533=0; x18533 < x18506; x18533++) {
int32_t x18534 = x18530;
int32_t x18535 = x18531;
int32_t x18536 = x18529;
int32_t x18537 = x18536;
int32_t x18538 = x18534;
int32_t x18539 = x18535;
for(int x18541=0; x18541 < x18508; x18541++) {
int32_t x18542 = x18538;
int32_t x18543 = x18539;
int32_t x18544 = x18537;
int32_t x18545 = x18544;
int32_t x18546 = x18542;
int32_t x18547 = x18543;
for(int x18548=0; x18548 < x18508; x18548++) {
int32_t x18549 = x18545;
int32_t x18550 = x18546;
float x18551 = x18403[x18550];
int32_t x18552 = x18547;
float x18553 = x232[x18552];
float x18554 = x18551 + x18553;
x18521[x18549] = x18554;
x18545 += 1;
if (x18557) {
x18546 += 1;
} else {
}

}
x18537 += x18508;
if (x18557) {
x18538 += x18390;
} else {
}

}
x18529 += x18509;
if (x18570) {
x18530 += x18391;
} else {
}
if (x18574) {
x18531 += 1;
} else {
}

}
x18522 += x18510;
x18523 += x18392;

}
bool x18584 = x18506 == 1;
bool x18585 = x18584 || x16792;
bool x18586 = x18506 == x16172;
bool x18587 = x18585 || x18586;
bool x18592;
if (x18587) {
x18592 = x18591;
} else {
x18592 = false;
}
bool x18593;
if (x18592) {
x18593 = x18591;
} else {
x18593 = false;
}
if (x18593) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18506,x18508,x18508,64,x16172,x16174,x16174);
assert(false && "");
}
int32_t x18606 = 0;
int32_t x18607 = 0;
int32_t x18608 = 0;
bool x18599 = x18506 <= x16172;
int32_t x18600;
if (x18599) {
x18600 = x16172;
} else {
x18600 = x18506;
}
bool x18659 = x18506 > 1;
int32_t x18604 = x18600 * x18603;
for(int x18609=0; x18609 < 64; x18609++) {
int32_t x18610 = x18607;
int32_t x18611 = x18608;
int32_t x18612 = x18606;
int32_t x18613 = x18612;
int32_t x18614 = x18610;
int32_t x18615 = x18611;
for(int x18617=0; x18617 < x18600; x18617++) {
int32_t x18618 = x18614;
int32_t x18619 = x18615;
int32_t x18620 = x18613;
int32_t x18621 = x18620;
int32_t x18622 = x18618;
int32_t x18623 = x18619;
for(int x18625=0; x18625 < x18602; x18625++) {
int32_t x18626 = x18622;
int32_t x18627 = x18623;
int32_t x18628 = x18621;
int32_t x18629 = x18628;
int32_t x18630 = x18626;
int32_t x18631 = x18627;
for(int x18632=0; x18632 < x18602; x18632++) {
int32_t x18633 = x18630;
float x18634 = x18521[x18633];
int32_t x18635 = x18631;
float x18636 = x16885[x18635];
float x18637 = x18634 + x18636;
x18521[x18633] = x18637;
x18629 += 1;
if (x18640) {
x18630 += 1;
} else {
}
if (x16850) {
x18631 += 1;
} else {
}

}
x18621 += x18602;
if (x18640) {
x18622 += x18508;
} else {
}
if (x16850) {
x18623 += x16174;
} else {
}

}
x18613 += x18603;
if (x18659) {
x18614 += x18509;
} else {
}
if (x16870) {
x18615 += x16175;
} else {
}

}
x18606 += x18604;
x18607 += x18510;
x18608 += x16176;

}
float* x18673 = (float*)myMalloc(x18511 * sizeof(float));;
for(int x18675=0; x18675 < x18511; x18675++) {
float x18676 = x18521[x18675];
bool x18677 = x18676 < 0.0f;
if (x18677) {
x18673[x18675] = 0.0f;
} else {
float x18680 = x18521[x18675];
x18673[x18675] = x18680;
}

}
float* x18694 = (float*)myMalloc(x18693 * sizeof(float));;
int32_t x18697 = 64 * x18506;
int32_t x18698 = x18697 * x18689;
float* x18699 = (float*)myMalloc(x18698 * sizeof(float));;
int32_t x18695 = x18506 * x18689;
for(int x18700=0; x18700 < 64; x18700++) {
int32_t x18701 = x18700 * x18510;
float* x18702 = x18673+x18701;
int32_t x18703 = x18700 * x18690;
float* x18704 = x18694+x18703;
int32_t x18705 = x18700 * x18695;
float* x18706 = x18699+x18705;
for(int x18707=0; x18707 < x18506; x18707++) {
int32_t x18708 = x18707 / 1;
int32_t x18712 = x18708 * x18688;
int32_t x18713 = x18712 * x18688;
int32_t x18709 = x18707 % 1;
int32_t x18710 = x18709 / 1;
int32_t x18714 = x18710 * x18688;
int32_t x18715 = x18714 * x18688;
int32_t x18716 = x18713 + x18715;
int32_t x18711 = x18709 % 1;
int32_t x18717 = x18711 * x18688;
int32_t x18718 = x18717 * x18688;
int32_t x18719 = x18716 + x18718;
float* x18720 = x18706+x18719;
int32_t x18721 = x18708 * x18508;
int32_t x18722 = x18721 * x18508;
float* x18723 = x18702+x18722;
for(int x18725=0; x18725 < x18688; x18725++) {
int32_t x18727 = x18725 * x18688;
float* x18728 = x18720+x18727;
int32_t x18726 = x18725 + x18710;
int32_t x18729 = x18726 * x18508;
int32_t x18730 = x18729 + x18711;
float* x18731 = x18723+x18730;
memcpy(x18728, x18731, 4 * x18688);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x18689,x18506,1,x218,x18506,x18706,x18689,1,x18704,x18689);

}
int32_t x18740 = 0;
int32_t x18741 = 1;
x18741 *= 1;
x18740 += 1;
x18741 *= 1;
x18741 *= 1;
int32_t x18746 = x18740;
bool x18747 = x18746 >= 2;
if (x18747) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18752 = x18746 == 0;
if (x18752) {
int32_t x18753 = x18741;
bool x18754 = x18753 == 256;
if (x18754) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18761 = x18741;
int32_t x18762 = 256 / x18761;
bool x18766;
if (x452) {
bool x18763 = x18762 == 1;
bool x18764 = 256 == x18762;
bool x18765 = x18763 || x18764;
x18766 = x18765;
} else {
x18766 = false;
}
bool x18770;
if (x18766) {
x18770 = x18769;
} else {
x18770 = false;
}
bool x18771;
if (x18770) {
x18771 = x18769;
} else {
x18771 = false;
}
if (x18771) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x18688,x18688,1,x18762,1,1);
assert(false && "");
}
bool x18777 = 256 <= x18762;
int32_t x18778;
if (x18777) {
x18778 = x18762;
} else {
x18778 = 256;
}
bool x18784 = x18778 > 0;
bool x18786;
if (x18784) {
x18786 = x18785;
} else {
x18786 = false;
}
bool x18787;
if (x18786) {
x18787 = x18785;
} else {
x18787 = false;
}
if (x18787) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(18688) x Sym(18688)"," x Const(1) x Sym(18762) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x18782 = x18778 * x18781;
int32_t x18783 = 64 * x18782;
float* x18793 = (float*)myMalloc(x18783 * sizeof(float));;
int32_t x18794 = 0;
int32_t x18795 = 0;
int32_t x18796 = 0;
bool x18843 = x18762 > 1;
for(int x18797=0; x18797 < 64; x18797++) {
int32_t x18798 = x18795;
int32_t x18799 = x18796;
int32_t x18800 = x18794;
int32_t x18801 = x18800;
int32_t x18802 = x18798;
int32_t x18803 = x18799;
for(int x18805=0; x18805 < x18778; x18805++) {
int32_t x18806 = x18802;
int32_t x18807 = x18803;
int32_t x18808 = x18801;
int32_t x18809 = x18808;
int32_t x18810 = x18806;
int32_t x18811 = x18807;
for(int x18813=0; x18813 < x18780; x18813++) {
int32_t x18814 = x18810;
int32_t x18815 = x18811;
int32_t x18816 = x18809;
int32_t x18817 = x18816;
int32_t x18818 = x18814;
int32_t x18819 = x18815;
for(int x18820=0; x18820 < x18780; x18820++) {
int32_t x18821 = x18817;
int32_t x18822 = x18818;
float x18823 = x18694[x18822];
int32_t x18824 = x18819;
float x18825 = x178[x18824];
float x18826 = x18823 - x18825;
x18793[x18821] = x18826;
x18817 += 1;
if (x18829) {
x18818 += 1;
} else {
}

}
x18809 += x18780;
if (x18829) {
x18810 += x18688;
} else {
}

}
x18801 += x18781;
x18802 += x18689;
if (x18843) {
x18803 += 1;
} else {
}

}
x18794 += x18782;
x18795 += x18690;

}
float* x18853 = (float*)myMalloc(256 * sizeof(float));;
for(int x18854=0; x18854 < 256; x18854++) {
float x18855 = x174[x18854];
float x18856 = x18855 + 1.0E-5f;
x18853[x18854] = x18856;

}
float* x18860 = (float*)myMalloc(256 * sizeof(float));;
for(int x18861=0; x18861 < 256; x18861++) {
float x18862 = x18853[x18861];
double x18863 = (double)x18862;
double x18864 = sqrt(x18863);
float x18865 = (float)x18864;
x18860[x18861] = x18865;

}
int32_t x18869 = 0;
int32_t x18870 = 1;
x18870 *= 1;
x18869 += 1;
x18870 *= 1;
x18870 *= 1;
int32_t x18875 = x18869;
bool x18876 = x18875 >= 2;
if (x18876) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18881 = x18875 == 0;
if (x18881) {
int32_t x18882 = x18870;
bool x18883 = x18882 == 256;
if (x18883) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x18890 = x18870;
int32_t x18891 = 256 / x18890;
bool x18897;
if (x452) {
bool x18892 = x18778 == 1;
bool x18893 = x18891 == 1;
bool x18894 = x18892 || x18893;
bool x18895 = x18778 == x18891;
bool x18896 = x18894 || x18895;
x18897 = x18896;
} else {
x18897 = false;
}
bool x18901;
if (x18897) {
x18901 = x18900;
} else {
x18901 = false;
}
bool x18902;
if (x18901) {
x18902 = x18900;
} else {
x18902 = false;
}
if (x18902) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18778,x18780,x18780,1,x18891,1,1);
assert(false && "");
}
bool x18908 = x18778 <= x18891;
int32_t x18909;
if (x18908) {
x18909 = x18891;
} else {
x18909 = x18778;
}
bool x18915 = x18909 > 0;
bool x18917;
if (x18915) {
x18917 = x18916;
} else {
x18917 = false;
}
bool x18918;
if (x18917) {
x18918 = x18916;
} else {
x18918 = false;
}
if (x18918) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(18778) x Sym(18780) x Sym(18780)"," x Const(1) x Sym(18891) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x18913 = x18909 * x18912;
int32_t x18914 = 64 * x18913;
float* x18924 = (float*)myMalloc(x18914 * sizeof(float));;
int32_t x18925 = 0;
int32_t x18926 = 0;
int32_t x18927 = 0;
bool x18973 = x18778 > 1;
bool x18977 = x18891 > 1;
for(int x18928=0; x18928 < 64; x18928++) {
int32_t x18929 = x18926;
int32_t x18930 = x18927;
int32_t x18931 = x18925;
int32_t x18932 = x18931;
int32_t x18933 = x18929;
int32_t x18934 = x18930;
for(int x18936=0; x18936 < x18909; x18936++) {
int32_t x18937 = x18933;
int32_t x18938 = x18934;
int32_t x18939 = x18932;
int32_t x18940 = x18939;
int32_t x18941 = x18937;
int32_t x18942 = x18938;
for(int x18944=0; x18944 < x18911; x18944++) {
int32_t x18945 = x18941;
int32_t x18946 = x18942;
int32_t x18947 = x18940;
int32_t x18948 = x18947;
int32_t x18949 = x18945;
int32_t x18950 = x18946;
for(int x18951=0; x18951 < x18911; x18951++) {
int32_t x18952 = x18948;
int32_t x18953 = x18949;
float x18954 = x18793[x18953];
int32_t x18955 = x18950;
float x18956 = x18860[x18955];
float x18957 = x18954 / x18956;
x18924[x18952] = x18957;
x18948 += 1;
if (x18960) {
x18949 += 1;
} else {
}

}
x18940 += x18911;
if (x18960) {
x18941 += x18780;
} else {
}

}
x18932 += x18912;
if (x18973) {
x18933 += x18781;
} else {
}
if (x18977) {
x18934 += 1;
} else {
}

}
x18925 += x18913;
x18926 += x18782;

}
int32_t x18987 = 0;
int32_t x18988 = 1;
x18988 *= 1;
x18987 += 1;
x18988 *= 1;
x18988 *= 1;
int32_t x18993 = x18987;
bool x18994 = x18993 >= 2;
if (x18994) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x18999 = x18993 == 0;
if (x18999) {
int32_t x19000 = x18988;
bool x19001 = x19000 == 256;
if (x19001) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19008 = x18988;
int32_t x19009 = 256 / x19008;
bool x19015;
if (x452) {
bool x19010 = x18909 == 1;
bool x19011 = x19009 == 1;
bool x19012 = x19010 || x19011;
bool x19013 = x18909 == x19009;
bool x19014 = x19012 || x19013;
x19015 = x19014;
} else {
x19015 = false;
}
bool x19019;
if (x19015) {
x19019 = x19018;
} else {
x19019 = false;
}
bool x19020;
if (x19019) {
x19020 = x19018;
} else {
x19020 = false;
}
if (x19020) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x18909,x18911,x18911,1,x19009,1,1);
assert(false && "");
}
bool x19026 = x18909 <= x19009;
int32_t x19027;
if (x19026) {
x19027 = x19009;
} else {
x19027 = x18909;
}
bool x19033 = x19027 > 0;
bool x19035;
if (x19033) {
x19035 = x19034;
} else {
x19035 = false;
}
bool x19036;
if (x19035) {
x19036 = x19034;
} else {
x19036 = false;
}
if (x19036) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(18909) x Sym(18911) x Sym(18911)"," x Const(1) x Sym(19009) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x19031 = x19027 * x19030;
int32_t x19032 = 64 * x19031;
float* x19042 = (float*)myMalloc(x19032 * sizeof(float));;
int32_t x19043 = 0;
int32_t x19044 = 0;
int32_t x19045 = 0;
bool x19091 = x18909 > 1;
bool x19095 = x19009 > 1;
for(int x19046=0; x19046 < 64; x19046++) {
int32_t x19047 = x19044;
int32_t x19048 = x19045;
int32_t x19049 = x19043;
int32_t x19050 = x19049;
int32_t x19051 = x19047;
int32_t x19052 = x19048;
for(int x19054=0; x19054 < x19027; x19054++) {
int32_t x19055 = x19051;
int32_t x19056 = x19052;
int32_t x19057 = x19050;
int32_t x19058 = x19057;
int32_t x19059 = x19055;
int32_t x19060 = x19056;
for(int x19062=0; x19062 < x19029; x19062++) {
int32_t x19063 = x19059;
int32_t x19064 = x19060;
int32_t x19065 = x19058;
int32_t x19066 = x19065;
int32_t x19067 = x19063;
int32_t x19068 = x19064;
for(int x19069=0; x19069 < x19029; x19069++) {
int32_t x19070 = x19066;
int32_t x19071 = x19067;
float x19072 = x18924[x19071];
int32_t x19073 = x19068;
float x19074 = x129[x19073];
float x19075 = x19072 * x19074;
x19042[x19070] = x19075;
x19066 += 1;
if (x19078) {
x19067 += 1;
} else {
}

}
x19058 += x19029;
if (x19078) {
x19059 += x18911;
} else {
}

}
x19050 += x19030;
if (x19091) {
x19051 += x18912;
} else {
}
if (x19095) {
x19052 += 1;
} else {
}

}
x19043 += x19031;
x19044 += x18913;

}
int32_t x19105 = 0;
int32_t x19106 = 1;
x19106 *= 1;
x19105 += 1;
x19106 *= 1;
x19106 *= 1;
int32_t x19111 = x19105;
bool x19112 = x19111 >= 2;
if (x19112) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19117 = x19111 == 0;
if (x19117) {
int32_t x19118 = x19106;
bool x19119 = x19118 == 256;
if (x19119) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19126 = x19106;
int32_t x19127 = 256 / x19126;
bool x19133;
if (x452) {
bool x19128 = x19027 == 1;
bool x19129 = x19127 == 1;
bool x19130 = x19128 || x19129;
bool x19131 = x19027 == x19127;
bool x19132 = x19130 || x19131;
x19133 = x19132;
} else {
x19133 = false;
}
bool x19137;
if (x19133) {
x19137 = x19136;
} else {
x19137 = false;
}
bool x19138;
if (x19137) {
x19138 = x19136;
} else {
x19138 = false;
}
if (x19138) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19027,x19029,x19029,1,x19127,1,1);
assert(false && "");
}
bool x19144 = x19027 <= x19127;
int32_t x19145;
if (x19144) {
x19145 = x19127;
} else {
x19145 = x19027;
}
bool x19151 = x19145 > 0;
bool x19153;
if (x19151) {
x19153 = x19152;
} else {
x19153 = false;
}
bool x19154;
if (x19153) {
x19154 = x19152;
} else {
x19154 = false;
}
if (x19154) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(19027) x Sym(19029) x Sym(19029)"," x Const(1) x Sym(19127) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x19149 = x19145 * x19148;
int32_t x19150 = 64 * x19149;
float* x19160 = (float*)myMalloc(x19150 * sizeof(float));;
int32_t x19161 = 0;
int32_t x19162 = 0;
int32_t x19163 = 0;
bool x19209 = x19027 > 1;
bool x19213 = x19127 > 1;
for(int x19164=0; x19164 < 64; x19164++) {
int32_t x19165 = x19162;
int32_t x19166 = x19163;
int32_t x19167 = x19161;
int32_t x19168 = x19167;
int32_t x19169 = x19165;
int32_t x19170 = x19166;
for(int x19172=0; x19172 < x19145; x19172++) {
int32_t x19173 = x19169;
int32_t x19174 = x19170;
int32_t x19175 = x19168;
int32_t x19176 = x19175;
int32_t x19177 = x19173;
int32_t x19178 = x19174;
for(int x19180=0; x19180 < x19147; x19180++) {
int32_t x19181 = x19177;
int32_t x19182 = x19178;
int32_t x19183 = x19176;
int32_t x19184 = x19183;
int32_t x19185 = x19181;
int32_t x19186 = x19182;
for(int x19187=0; x19187 < x19147; x19187++) {
int32_t x19188 = x19184;
int32_t x19189 = x19185;
float x19190 = x19042[x19189];
int32_t x19191 = x19186;
float x19192 = x197[x19191];
float x19193 = x19190 + x19192;
x19160[x19188] = x19193;
x19184 += 1;
if (x19196) {
x19185 += 1;
} else {
}

}
x19176 += x19147;
if (x19196) {
x19177 += x19029;
} else {
}

}
x19168 += x19148;
if (x19209) {
x19169 += x19030;
} else {
}
if (x19213) {
x19170 += 1;
} else {
}

}
x19161 += x19149;
x19162 += x19031;

}
float* x19223 = (float*)myMalloc(x19150 * sizeof(float));;
for(int x19225=0; x19225 < x19150; x19225++) {
float x19226 = x19160[x19225];
bool x19227 = x19226 < 0.0f;
if (x19227) {
x19223[x19225] = 0.0f;
} else {
float x19230 = x19160[x19225];
x19223[x19225] = x19230;
}

}
float* x19245 = (float*)myMalloc(x19244 * sizeof(float));;
int32_t x19246 = 9 * x19145;
int32_t x19249 = 64 * x19246;
int32_t x19250 = x19249 * x19240;
float* x19251 = (float*)myMalloc(x19250 * sizeof(float));;
int32_t x19247 = x19246 * x19240;
int32_t x19259 = x19145 * 3;
int32_t x19260 = x19259 * 3;
for(int x19252=0; x19252 < 64; x19252++) {
int32_t x19253 = x19252 * x19149;
float* x19254 = x19223+x19253;
int32_t x19255 = x19252 * x19241;
float* x19256 = x19245+x19255;
int32_t x19257 = x19252 * x19247;
float* x19258 = x19251+x19257;
for(int x19262=0; x19262 < x19260; x19262++) {
int32_t x19263 = x19262 / 9;
int32_t x19267 = x19263 * 3;
int32_t x19268 = x19267 * 3;
int32_t x19269 = x19268 * x19239;
int32_t x19270 = x19269 * x19239;
int32_t x19264 = x19262 % 9;
int32_t x19265 = x19264 / 3;
int32_t x19271 = x19265 * 3;
int32_t x19272 = x19271 * x19239;
int32_t x19273 = x19272 * x19239;
int32_t x19274 = x19270 + x19273;
int32_t x19266 = x19264 % 3;
int32_t x19275 = x19266 * x19239;
int32_t x19276 = x19275 * x19239;
int32_t x19277 = x19274 + x19276;
float* x19278 = x19258+x19277;
int32_t x19279 = x19263 * x19147;
int32_t x19280 = x19279 * x19147;
float* x19281 = x19254+x19280;
int32_t x19294 = 1 - x19266;
bool x19295 = x19294 > 0;
int32_t x19296;
if (x19295) {
x19296 = x19294;
} else {
x19296 = 0;
}
int32_t x19297 = 3 - x19266;
int32_t x19298 = x19297 - 1;
int32_t x19299 = 1 - x19298;
bool x19300 = x19299 > 0;
int32_t x19301;
if (x19300) {
x19301 = x19299;
} else {
x19301 = 0;
}
int32_t x19302 = x19239 - x19301;
int32_t x19303 = x19302 - x19296;
bool x19304 = x19303 <= 0;
bool x19308 = x19296 > 0;
int32_t x19293 = -1 + x19266;
bool x19321 = x19301 > 0;
for(int x19283=0; x19283 < x19239; x19283++) {
int32_t x19284 = x19283 - 1;
int32_t x19285 = x19284 + x19265;
bool x19286 = x19285 < 0;
bool x19287 = x19285 >= x19147;
bool x19288 = x19286 || x19287;
if (x19288) {
int32_t x19289 = x19283 * x19239;
float* x19290 = x19278+x19289;
memset(x19290, 0, 4 * x19239);;
} else {
if (x19304) {
int32_t x19289 = x19283 * x19239;
float* x19305 = x19278+x19289;
memset(x19305, 0, 4 * x19239);;
} else {
int32_t x19289 = x19283 * x19239;
if (x19308) {
float* x19309 = x19278+x19289;
memset(x19309, 0, 4 * x19296);;
} else {
}
// may have segfault here
int32_t x19314 = x19289 + x19296;
float* x19315 = x19278+x19314;
int32_t x19316 = x19285 * x19147;
int32_t x19317 = x19316 + x19293;
int32_t x19318 = x19317 + x19296;
float* x19319 = x19281+x19318;
memcpy(x19315, x19319, 4 * x19303);;
if (x19321) {
int32_t x19322 = x19289 + x19239;
int32_t x19323 = x19322 - x19301;
float* x19324 = x19278+x19323;
memset(x19324, 0, 4 * x19301);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x19240,x19246,1,x14,x19246,x19258,x19240,1,x19256,x19240);

}
int32_t x19339 = 0;
int32_t x19340 = 1;
x19340 *= 1;
x19339 += 1;
x19340 *= 1;
x19340 *= 1;
int32_t x19345 = x19339;
bool x19346 = x19345 >= 2;
if (x19346) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19351 = x19345 == 0;
if (x19351) {
int32_t x19352 = x19340;
bool x19353 = x19352 == 256;
if (x19353) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19360 = x19340;
int32_t x19361 = 256 / x19360;
bool x19365;
if (x452) {
bool x19362 = x19361 == 1;
bool x19363 = 256 == x19361;
bool x19364 = x19362 || x19363;
x19365 = x19364;
} else {
x19365 = false;
}
bool x19369;
if (x19365) {
x19369 = x19368;
} else {
x19369 = false;
}
bool x19370;
if (x19369) {
x19370 = x19368;
} else {
x19370 = false;
}
if (x19370) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x19239,x19239,1,x19361,1,1);
assert(false && "");
}
bool x19376 = 256 <= x19361;
int32_t x19377;
if (x19376) {
x19377 = x19361;
} else {
x19377 = 256;
}
bool x19383 = x19377 > 0;
bool x19385;
if (x19383) {
x19385 = x19384;
} else {
x19385 = false;
}
bool x19386;
if (x19385) {
x19386 = x19384;
} else {
x19386 = false;
}
if (x19386) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(19239) x Sym(19239)"," x Const(1) x Sym(19361) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x19381 = x19377 * x19380;
int32_t x19382 = 64 * x19381;
float* x19392 = (float*)myMalloc(x19382 * sizeof(float));;
int32_t x19393 = 0;
int32_t x19394 = 0;
int32_t x19395 = 0;
bool x19442 = x19361 > 1;
for(int x19396=0; x19396 < 64; x19396++) {
int32_t x19397 = x19394;
int32_t x19398 = x19395;
int32_t x19399 = x19393;
int32_t x19400 = x19399;
int32_t x19401 = x19397;
int32_t x19402 = x19398;
for(int x19404=0; x19404 < x19377; x19404++) {
int32_t x19405 = x19401;
int32_t x19406 = x19402;
int32_t x19407 = x19400;
int32_t x19408 = x19407;
int32_t x19409 = x19405;
int32_t x19410 = x19406;
for(int x19412=0; x19412 < x19379; x19412++) {
int32_t x19413 = x19409;
int32_t x19414 = x19410;
int32_t x19415 = x19408;
int32_t x19416 = x19415;
int32_t x19417 = x19413;
int32_t x19418 = x19414;
for(int x19419=0; x19419 < x19379; x19419++) {
int32_t x19420 = x19416;
int32_t x19421 = x19417;
float x19422 = x19245[x19421];
int32_t x19423 = x19418;
float x19424 = x124[x19423];
float x19425 = x19422 - x19424;
x19392[x19420] = x19425;
x19416 += 1;
if (x19428) {
x19417 += 1;
} else {
}

}
x19408 += x19379;
if (x19428) {
x19409 += x19239;
} else {
}

}
x19400 += x19380;
x19401 += x19240;
if (x19442) {
x19402 += 1;
} else {
}

}
x19393 += x19381;
x19394 += x19241;

}
float* x19452 = (float*)myMalloc(256 * sizeof(float));;
for(int x19453=0; x19453 < 256; x19453++) {
float x19454 = x63[x19453];
float x19455 = x19454 + 1.0E-5f;
x19452[x19453] = x19455;

}
float* x19459 = (float*)myMalloc(256 * sizeof(float));;
for(int x19460=0; x19460 < 256; x19460++) {
float x19461 = x19452[x19460];
double x19462 = (double)x19461;
double x19463 = sqrt(x19462);
float x19464 = (float)x19463;
x19459[x19460] = x19464;

}
int32_t x19468 = 0;
int32_t x19469 = 1;
x19469 *= 1;
x19468 += 1;
x19469 *= 1;
x19469 *= 1;
int32_t x19474 = x19468;
bool x19475 = x19474 >= 2;
if (x19475) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19480 = x19474 == 0;
if (x19480) {
int32_t x19481 = x19469;
bool x19482 = x19481 == 256;
if (x19482) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19489 = x19469;
int32_t x19490 = 256 / x19489;
bool x19496;
if (x452) {
bool x19491 = x19377 == 1;
bool x19492 = x19490 == 1;
bool x19493 = x19491 || x19492;
bool x19494 = x19377 == x19490;
bool x19495 = x19493 || x19494;
x19496 = x19495;
} else {
x19496 = false;
}
bool x19500;
if (x19496) {
x19500 = x19499;
} else {
x19500 = false;
}
bool x19501;
if (x19500) {
x19501 = x19499;
} else {
x19501 = false;
}
if (x19501) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19377,x19379,x19379,1,x19490,1,1);
assert(false && "");
}
bool x19507 = x19377 <= x19490;
int32_t x19508;
if (x19507) {
x19508 = x19490;
} else {
x19508 = x19377;
}
bool x19514 = x19508 > 0;
bool x19516;
if (x19514) {
x19516 = x19515;
} else {
x19516 = false;
}
bool x19517;
if (x19516) {
x19517 = x19515;
} else {
x19517 = false;
}
if (x19517) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(19377) x Sym(19379) x Sym(19379)"," x Const(1) x Sym(19490) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x19512 = x19508 * x19511;
int32_t x19513 = 64 * x19512;
float* x19523 = (float*)myMalloc(x19513 * sizeof(float));;
int32_t x19524 = 0;
int32_t x19525 = 0;
int32_t x19526 = 0;
bool x19572 = x19377 > 1;
bool x19576 = x19490 > 1;
for(int x19527=0; x19527 < 64; x19527++) {
int32_t x19528 = x19525;
int32_t x19529 = x19526;
int32_t x19530 = x19524;
int32_t x19531 = x19530;
int32_t x19532 = x19528;
int32_t x19533 = x19529;
for(int x19535=0; x19535 < x19508; x19535++) {
int32_t x19536 = x19532;
int32_t x19537 = x19533;
int32_t x19538 = x19531;
int32_t x19539 = x19538;
int32_t x19540 = x19536;
int32_t x19541 = x19537;
for(int x19543=0; x19543 < x19510; x19543++) {
int32_t x19544 = x19540;
int32_t x19545 = x19541;
int32_t x19546 = x19539;
int32_t x19547 = x19546;
int32_t x19548 = x19544;
int32_t x19549 = x19545;
for(int x19550=0; x19550 < x19510; x19550++) {
int32_t x19551 = x19547;
int32_t x19552 = x19548;
float x19553 = x19392[x19552];
int32_t x19554 = x19549;
float x19555 = x19459[x19554];
float x19556 = x19553 / x19555;
x19523[x19551] = x19556;
x19547 += 1;
if (x19559) {
x19548 += 1;
} else {
}

}
x19539 += x19510;
if (x19559) {
x19540 += x19379;
} else {
}

}
x19531 += x19511;
if (x19572) {
x19532 += x19380;
} else {
}
if (x19576) {
x19533 += 1;
} else {
}

}
x19524 += x19512;
x19525 += x19381;

}
int32_t x19586 = 0;
int32_t x19587 = 1;
x19587 *= 1;
x19586 += 1;
x19587 *= 1;
x19587 *= 1;
int32_t x19592 = x19586;
bool x19593 = x19592 >= 2;
if (x19593) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19598 = x19592 == 0;
if (x19598) {
int32_t x19599 = x19587;
bool x19600 = x19599 == 256;
if (x19600) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19607 = x19587;
int32_t x19608 = 256 / x19607;
bool x19614;
if (x452) {
bool x19609 = x19508 == 1;
bool x19610 = x19608 == 1;
bool x19611 = x19609 || x19610;
bool x19612 = x19508 == x19608;
bool x19613 = x19611 || x19612;
x19614 = x19613;
} else {
x19614 = false;
}
bool x19618;
if (x19614) {
x19618 = x19617;
} else {
x19618 = false;
}
bool x19619;
if (x19618) {
x19619 = x19617;
} else {
x19619 = false;
}
if (x19619) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19508,x19510,x19510,1,x19608,1,1);
assert(false && "");
}
bool x19625 = x19508 <= x19608;
int32_t x19626;
if (x19625) {
x19626 = x19608;
} else {
x19626 = x19508;
}
bool x19632 = x19626 > 0;
bool x19634;
if (x19632) {
x19634 = x19633;
} else {
x19634 = false;
}
bool x19635;
if (x19634) {
x19635 = x19633;
} else {
x19635 = false;
}
if (x19635) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(19508) x Sym(19510) x Sym(19510)"," x Const(1) x Sym(19608) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x19630 = x19626 * x19629;
int32_t x19631 = 64 * x19630;
float* x19641 = (float*)myMalloc(x19631 * sizeof(float));;
int32_t x19642 = 0;
int32_t x19643 = 0;
int32_t x19644 = 0;
bool x19690 = x19508 > 1;
bool x19694 = x19608 > 1;
for(int x19645=0; x19645 < 64; x19645++) {
int32_t x19646 = x19643;
int32_t x19647 = x19644;
int32_t x19648 = x19642;
int32_t x19649 = x19648;
int32_t x19650 = x19646;
int32_t x19651 = x19647;
for(int x19653=0; x19653 < x19626; x19653++) {
int32_t x19654 = x19650;
int32_t x19655 = x19651;
int32_t x19656 = x19649;
int32_t x19657 = x19656;
int32_t x19658 = x19654;
int32_t x19659 = x19655;
for(int x19661=0; x19661 < x19628; x19661++) {
int32_t x19662 = x19658;
int32_t x19663 = x19659;
int32_t x19664 = x19657;
int32_t x19665 = x19664;
int32_t x19666 = x19662;
int32_t x19667 = x19663;
for(int x19668=0; x19668 < x19628; x19668++) {
int32_t x19669 = x19665;
int32_t x19670 = x19666;
float x19671 = x19523[x19670];
int32_t x19672 = x19667;
float x19673 = x228[x19672];
float x19674 = x19671 * x19673;
x19641[x19669] = x19674;
x19665 += 1;
if (x19677) {
x19666 += 1;
} else {
}

}
x19657 += x19628;
if (x19677) {
x19658 += x19510;
} else {
}

}
x19649 += x19629;
if (x19690) {
x19650 += x19511;
} else {
}
if (x19694) {
x19651 += 1;
} else {
}

}
x19642 += x19630;
x19643 += x19512;

}
int32_t x19704 = 0;
int32_t x19705 = 1;
x19705 *= 1;
x19704 += 1;
x19705 *= 1;
x19705 *= 1;
int32_t x19710 = x19704;
bool x19711 = x19710 >= 2;
if (x19711) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19716 = x19710 == 0;
if (x19716) {
int32_t x19717 = x19705;
bool x19718 = x19717 == 256;
if (x19718) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19725 = x19705;
int32_t x19726 = 256 / x19725;
bool x19732;
if (x452) {
bool x19727 = x19626 == 1;
bool x19728 = x19726 == 1;
bool x19729 = x19727 || x19728;
bool x19730 = x19626 == x19726;
bool x19731 = x19729 || x19730;
x19732 = x19731;
} else {
x19732 = false;
}
bool x19736;
if (x19732) {
x19736 = x19735;
} else {
x19736 = false;
}
bool x19737;
if (x19736) {
x19737 = x19735;
} else {
x19737 = false;
}
if (x19737) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19626,x19628,x19628,1,x19726,1,1);
assert(false && "");
}
bool x19743 = x19626 <= x19726;
int32_t x19744;
if (x19743) {
x19744 = x19726;
} else {
x19744 = x19626;
}
bool x19750 = x19744 > 0;
bool x19752;
if (x19750) {
x19752 = x19751;
} else {
x19752 = false;
}
bool x19753;
if (x19752) {
x19753 = x19751;
} else {
x19753 = false;
}
if (x19753) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(19626) x Sym(19628) x Sym(19628)"," x Const(1) x Sym(19726) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x19748 = x19744 * x19747;
int32_t x19749 = 64 * x19748;
float* x19759 = (float*)myMalloc(x19749 * sizeof(float));;
int32_t x19760 = 0;
int32_t x19761 = 0;
int32_t x19762 = 0;
bool x19808 = x19626 > 1;
bool x19812 = x19726 > 1;
for(int x19763=0; x19763 < 64; x19763++) {
int32_t x19764 = x19761;
int32_t x19765 = x19762;
int32_t x19766 = x19760;
int32_t x19767 = x19766;
int32_t x19768 = x19764;
int32_t x19769 = x19765;
for(int x19771=0; x19771 < x19744; x19771++) {
int32_t x19772 = x19768;
int32_t x19773 = x19769;
int32_t x19774 = x19767;
int32_t x19775 = x19774;
int32_t x19776 = x19772;
int32_t x19777 = x19773;
for(int x19779=0; x19779 < x19746; x19779++) {
int32_t x19780 = x19776;
int32_t x19781 = x19777;
int32_t x19782 = x19775;
int32_t x19783 = x19782;
int32_t x19784 = x19780;
int32_t x19785 = x19781;
for(int x19786=0; x19786 < x19746; x19786++) {
int32_t x19787 = x19783;
int32_t x19788 = x19784;
float x19789 = x19641[x19788];
int32_t x19790 = x19785;
float x19791 = x192[x19790];
float x19792 = x19789 + x19791;
x19759[x19787] = x19792;
x19783 += 1;
if (x19795) {
x19784 += 1;
} else {
}

}
x19775 += x19746;
if (x19795) {
x19776 += x19628;
} else {
}

}
x19767 += x19747;
if (x19808) {
x19768 += x19629;
} else {
}
if (x19812) {
x19769 += 1;
} else {
}

}
x19760 += x19748;
x19761 += x19630;

}
float* x19822 = (float*)myMalloc(x19749 * sizeof(float));;
for(int x19824=0; x19824 < x19749; x19824++) {
float x19825 = x19759[x19824];
bool x19826 = x19825 < 0.0f;
if (x19826) {
x19822[x19824] = 0.0f;
} else {
float x19829 = x19759[x19824];
x19822[x19824] = x19829;
}

}
float* x19843 = (float*)myMalloc(x19842 * sizeof(float));;
int32_t x19846 = 64 * x19744;
int32_t x19847 = x19846 * x19838;
float* x19848 = (float*)myMalloc(x19847 * sizeof(float));;
int32_t x19844 = x19744 * x19838;
for(int x19849=0; x19849 < 64; x19849++) {
int32_t x19850 = x19849 * x19748;
float* x19851 = x19822+x19850;
int32_t x19852 = x19849 * x19839;
float* x19853 = x19843+x19852;
int32_t x19854 = x19849 * x19844;
float* x19855 = x19848+x19854;
for(int x19856=0; x19856 < x19744; x19856++) {
int32_t x19857 = x19856 / 1;
int32_t x19861 = x19857 * x19837;
int32_t x19862 = x19861 * x19837;
int32_t x19858 = x19856 % 1;
int32_t x19859 = x19858 / 1;
int32_t x19863 = x19859 * x19837;
int32_t x19864 = x19863 * x19837;
int32_t x19865 = x19862 + x19864;
int32_t x19860 = x19858 % 1;
int32_t x19866 = x19860 * x19837;
int32_t x19867 = x19866 * x19837;
int32_t x19868 = x19865 + x19867;
float* x19869 = x19855+x19868;
int32_t x19870 = x19857 * x19746;
int32_t x19871 = x19870 * x19746;
float* x19872 = x19851+x19871;
for(int x19874=0; x19874 < x19837; x19874++) {
int32_t x19876 = x19874 * x19837;
float* x19877 = x19869+x19876;
int32_t x19875 = x19874 + x19859;
int32_t x19878 = x19875 * x19746;
int32_t x19879 = x19878 + x19860;
float* x19880 = x19872+x19879;
memcpy(x19877, x19880, 4 * x19837);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x19838,x19744,1,x116,x19744,x19855,x19838,1,x19853,x19838);

}
int32_t x19889 = 0;
int32_t x19890 = 1;
x19890 *= 1;
x19889 += 1;
x19890 *= 1;
x19890 *= 1;
int32_t x19895 = x19889;
bool x19896 = x19895 >= 2;
if (x19896) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x19901 = x19895 == 0;
if (x19901) {
int32_t x19902 = x19890;
bool x19903 = x19902 == 1024;
if (x19903) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x19910 = x19890;
int32_t x19911 = 1024 / x19910;
bool x19915;
if (x452) {
bool x19912 = x19911 == 1;
bool x19913 = 1024 == x19911;
bool x19914 = x19912 || x19913;
x19915 = x19914;
} else {
x19915 = false;
}
bool x19919;
if (x19915) {
x19919 = x19918;
} else {
x19919 = false;
}
bool x19920;
if (x19919) {
x19920 = x19918;
} else {
x19920 = false;
}
if (x19920) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x19837,x19837,1,x19911,1,1);
assert(false && "");
}
bool x19926 = 1024 <= x19911;
int32_t x19927;
if (x19926) {
x19927 = x19911;
} else {
x19927 = 1024;
}
bool x19933 = x19927 > 0;
bool x19935;
if (x19933) {
x19935 = x19934;
} else {
x19935 = false;
}
bool x19936;
if (x19935) {
x19936 = x19934;
} else {
x19936 = false;
}
if (x19936) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(1024) x Sym(19837) x Sym(19837)"," x Const(1) x Sym(19911) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x19931 = x19927 * x19930;
int32_t x19932 = 64 * x19931;
float* x19942 = (float*)myMalloc(x19932 * sizeof(float));;
int32_t x19943 = 0;
int32_t x19944 = 0;
int32_t x19945 = 0;
bool x19992 = x19911 > 1;
for(int x19946=0; x19946 < 64; x19946++) {
int32_t x19947 = x19944;
int32_t x19948 = x19945;
int32_t x19949 = x19943;
int32_t x19950 = x19949;
int32_t x19951 = x19947;
int32_t x19952 = x19948;
for(int x19954=0; x19954 < x19927; x19954++) {
int32_t x19955 = x19951;
int32_t x19956 = x19952;
int32_t x19957 = x19950;
int32_t x19958 = x19957;
int32_t x19959 = x19955;
int32_t x19960 = x19956;
for(int x19962=0; x19962 < x19929; x19962++) {
int32_t x19963 = x19959;
int32_t x19964 = x19960;
int32_t x19965 = x19958;
int32_t x19966 = x19965;
int32_t x19967 = x19963;
int32_t x19968 = x19964;
for(int x19969=0; x19969 < x19929; x19969++) {
int32_t x19970 = x19966;
int32_t x19971 = x19967;
float x19972 = x19843[x19971];
int32_t x19973 = x19968;
float x19974 = x140[x19973];
float x19975 = x19972 - x19974;
x19942[x19970] = x19975;
x19966 += 1;
if (x19978) {
x19967 += 1;
} else {
}

}
x19958 += x19929;
if (x19978) {
x19959 += x19837;
} else {
}

}
x19950 += x19930;
x19951 += x19838;
if (x19992) {
x19952 += 1;
} else {
}

}
x19943 += x19931;
x19944 += x19839;

}
float* x20002 = (float*)myMalloc(1024 * sizeof(float));;
for(int x20003=0; x20003 < 1024; x20003++) {
float x20004 = x188[x20003];
float x20005 = x20004 + 1.0E-5f;
x20002[x20003] = x20005;

}
float* x20009 = (float*)myMalloc(1024 * sizeof(float));;
for(int x20010=0; x20010 < 1024; x20010++) {
float x20011 = x20002[x20010];
double x20012 = (double)x20011;
double x20013 = sqrt(x20012);
float x20014 = (float)x20013;
x20009[x20010] = x20014;

}
int32_t x20018 = 0;
int32_t x20019 = 1;
x20019 *= 1;
x20018 += 1;
x20019 *= 1;
x20019 *= 1;
int32_t x20024 = x20018;
bool x20025 = x20024 >= 2;
if (x20025) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20030 = x20024 == 0;
if (x20030) {
int32_t x20031 = x20019;
bool x20032 = x20031 == 1024;
if (x20032) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20039 = x20019;
int32_t x20040 = 1024 / x20039;
bool x20046;
if (x452) {
bool x20041 = x19927 == 1;
bool x20042 = x20040 == 1;
bool x20043 = x20041 || x20042;
bool x20044 = x19927 == x20040;
bool x20045 = x20043 || x20044;
x20046 = x20045;
} else {
x20046 = false;
}
bool x20050;
if (x20046) {
x20050 = x20049;
} else {
x20050 = false;
}
bool x20051;
if (x20050) {
x20051 = x20049;
} else {
x20051 = false;
}
if (x20051) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x19927,x19929,x19929,1,x20040,1,1);
assert(false && "");
}
bool x20057 = x19927 <= x20040;
int32_t x20058;
if (x20057) {
x20058 = x20040;
} else {
x20058 = x19927;
}
bool x20064 = x20058 > 0;
bool x20066;
if (x20064) {
x20066 = x20065;
} else {
x20066 = false;
}
bool x20067;
if (x20066) {
x20067 = x20065;
} else {
x20067 = false;
}
if (x20067) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(19927) x Sym(19929) x Sym(19929)"," x Const(1) x Sym(20040) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x20062 = x20058 * x20061;
int32_t x20063 = 64 * x20062;
float* x20073 = (float*)myMalloc(x20063 * sizeof(float));;
int32_t x20074 = 0;
int32_t x20075 = 0;
int32_t x20076 = 0;
bool x20122 = x19927 > 1;
bool x20126 = x20040 > 1;
for(int x20077=0; x20077 < 64; x20077++) {
int32_t x20078 = x20075;
int32_t x20079 = x20076;
int32_t x20080 = x20074;
int32_t x20081 = x20080;
int32_t x20082 = x20078;
int32_t x20083 = x20079;
for(int x20085=0; x20085 < x20058; x20085++) {
int32_t x20086 = x20082;
int32_t x20087 = x20083;
int32_t x20088 = x20081;
int32_t x20089 = x20088;
int32_t x20090 = x20086;
int32_t x20091 = x20087;
for(int x20093=0; x20093 < x20060; x20093++) {
int32_t x20094 = x20090;
int32_t x20095 = x20091;
int32_t x20096 = x20089;
int32_t x20097 = x20096;
int32_t x20098 = x20094;
int32_t x20099 = x20095;
for(int x20100=0; x20100 < x20060; x20100++) {
int32_t x20101 = x20097;
int32_t x20102 = x20098;
float x20103 = x19942[x20102];
int32_t x20104 = x20099;
float x20105 = x20009[x20104];
float x20106 = x20103 / x20105;
x20073[x20101] = x20106;
x20097 += 1;
if (x20109) {
x20098 += 1;
} else {
}

}
x20089 += x20060;
if (x20109) {
x20090 += x19929;
} else {
}

}
x20081 += x20061;
if (x20122) {
x20082 += x19930;
} else {
}
if (x20126) {
x20083 += 1;
} else {
}

}
x20074 += x20062;
x20075 += x19931;

}
int32_t x20136 = 0;
int32_t x20137 = 1;
x20137 *= 1;
x20136 += 1;
x20137 *= 1;
x20137 *= 1;
int32_t x20142 = x20136;
bool x20143 = x20142 >= 2;
if (x20143) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20148 = x20142 == 0;
if (x20148) {
int32_t x20149 = x20137;
bool x20150 = x20149 == 1024;
if (x20150) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20157 = x20137;
int32_t x20158 = 1024 / x20157;
bool x20164;
if (x452) {
bool x20159 = x20058 == 1;
bool x20160 = x20158 == 1;
bool x20161 = x20159 || x20160;
bool x20162 = x20058 == x20158;
bool x20163 = x20161 || x20162;
x20164 = x20163;
} else {
x20164 = false;
}
bool x20168;
if (x20164) {
x20168 = x20167;
} else {
x20168 = false;
}
bool x20169;
if (x20168) {
x20169 = x20167;
} else {
x20169 = false;
}
if (x20169) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20058,x20060,x20060,1,x20158,1,1);
assert(false && "");
}
bool x20175 = x20058 <= x20158;
int32_t x20176;
if (x20175) {
x20176 = x20158;
} else {
x20176 = x20058;
}
bool x20182 = x20176 > 0;
bool x20184;
if (x20182) {
x20184 = x20183;
} else {
x20184 = false;
}
bool x20185;
if (x20184) {
x20185 = x20183;
} else {
x20185 = false;
}
if (x20185) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(20058) x Sym(20060) x Sym(20060)"," x Const(1) x Sym(20158) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x20180 = x20176 * x20179;
int32_t x20181 = 64 * x20180;
float* x20191 = (float*)myMalloc(x20181 * sizeof(float));;
int32_t x20192 = 0;
int32_t x20193 = 0;
int32_t x20194 = 0;
bool x20240 = x20058 > 1;
bool x20244 = x20158 > 1;
for(int x20195=0; x20195 < 64; x20195++) {
int32_t x20196 = x20193;
int32_t x20197 = x20194;
int32_t x20198 = x20192;
int32_t x20199 = x20198;
int32_t x20200 = x20196;
int32_t x20201 = x20197;
for(int x20203=0; x20203 < x20176; x20203++) {
int32_t x20204 = x20200;
int32_t x20205 = x20201;
int32_t x20206 = x20199;
int32_t x20207 = x20206;
int32_t x20208 = x20204;
int32_t x20209 = x20205;
for(int x20211=0; x20211 < x20178; x20211++) {
int32_t x20212 = x20208;
int32_t x20213 = x20209;
int32_t x20214 = x20207;
int32_t x20215 = x20214;
int32_t x20216 = x20212;
int32_t x20217 = x20213;
for(int x20218=0; x20218 < x20178; x20218++) {
int32_t x20219 = x20215;
int32_t x20220 = x20216;
float x20221 = x20073[x20220];
int32_t x20222 = x20217;
float x20223 = x263[x20222];
float x20224 = x20221 * x20223;
x20191[x20219] = x20224;
x20215 += 1;
if (x20227) {
x20216 += 1;
} else {
}

}
x20207 += x20178;
if (x20227) {
x20208 += x20060;
} else {
}

}
x20199 += x20179;
if (x20240) {
x20200 += x20061;
} else {
}
if (x20244) {
x20201 += 1;
} else {
}

}
x20192 += x20180;
x20193 += x20062;

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
bool x20268 = x20267 == 1024;
if (x20268) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20275 = x20255;
int32_t x20276 = 1024 / x20275;
bool x20282;
if (x452) {
bool x20277 = x20176 == 1;
bool x20278 = x20276 == 1;
bool x20279 = x20277 || x20278;
bool x20280 = x20176 == x20276;
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
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20176,x20178,x20178,1,x20276,1,1);
assert(false && "");
}
bool x20293 = x20176 <= x20276;
int32_t x20294;
if (x20293) {
x20294 = x20276;
} else {
x20294 = x20176;
}
bool x20300 = x20294 > 0;
bool x20302;
if (x20300) {
x20302 = x20301;
} else {
x20302 = false;
}
bool x20303;
if (x20302) {
x20303 = x20301;
} else {
x20303 = false;
}
if (x20303) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(20176) x Sym(20178) x Sym(20178)"," x Const(1) x Sym(20276) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x20298 = x20294 * x20297;
int32_t x20299 = 64 * x20298;
float* x20309 = (float*)myMalloc(x20299 * sizeof(float));;
int32_t x20310 = 0;
int32_t x20311 = 0;
int32_t x20312 = 0;
bool x20358 = x20176 > 1;
bool x20362 = x20276 > 1;
for(int x20313=0; x20313 < 64; x20313++) {
int32_t x20314 = x20311;
int32_t x20315 = x20312;
int32_t x20316 = x20310;
int32_t x20317 = x20316;
int32_t x20318 = x20314;
int32_t x20319 = x20315;
for(int x20321=0; x20321 < x20294; x20321++) {
int32_t x20322 = x20318;
int32_t x20323 = x20319;
int32_t x20324 = x20317;
int32_t x20325 = x20324;
int32_t x20326 = x20322;
int32_t x20327 = x20323;
for(int x20329=0; x20329 < x20296; x20329++) {
int32_t x20330 = x20326;
int32_t x20331 = x20327;
int32_t x20332 = x20325;
int32_t x20333 = x20332;
int32_t x20334 = x20330;
int32_t x20335 = x20331;
for(int x20336=0; x20336 < x20296; x20336++) {
int32_t x20337 = x20333;
int32_t x20338 = x20334;
float x20339 = x20191[x20338];
int32_t x20340 = x20335;
float x20341 = x57[x20340];
float x20342 = x20339 + x20341;
x20309[x20337] = x20342;
x20333 += 1;
if (x20345) {
x20334 += 1;
} else {
}

}
x20325 += x20296;
if (x20345) {
x20326 += x20178;
} else {
}

}
x20317 += x20297;
if (x20358) {
x20318 += x20179;
} else {
}
if (x20362) {
x20319 += 1;
} else {
}

}
x20310 += x20298;
x20311 += x20180;

}
bool x20372 = x20294 == 1;
bool x20373 = x20372 || x18584;
bool x20374 = x20294 == x18506;
bool x20375 = x20373 || x20374;
bool x20380;
if (x20375) {
x20380 = x20379;
} else {
x20380 = false;
}
bool x20381;
if (x20380) {
x20381 = x20379;
} else {
x20381 = false;
}
if (x20381) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20294,x20296,x20296,64,x18506,x18508,x18508);
assert(false && "");
}
int32_t x20394 = 0;
int32_t x20395 = 0;
int32_t x20396 = 0;
bool x20387 = x20294 <= x18506;
int32_t x20388;
if (x20387) {
x20388 = x18506;
} else {
x20388 = x20294;
}
bool x20447 = x20294 > 1;
int32_t x20392 = x20388 * x20391;
for(int x20397=0; x20397 < 64; x20397++) {
int32_t x20398 = x20395;
int32_t x20399 = x20396;
int32_t x20400 = x20394;
int32_t x20401 = x20400;
int32_t x20402 = x20398;
int32_t x20403 = x20399;
for(int x20405=0; x20405 < x20388; x20405++) {
int32_t x20406 = x20402;
int32_t x20407 = x20403;
int32_t x20408 = x20401;
int32_t x20409 = x20408;
int32_t x20410 = x20406;
int32_t x20411 = x20407;
for(int x20413=0; x20413 < x20390; x20413++) {
int32_t x20414 = x20410;
int32_t x20415 = x20411;
int32_t x20416 = x20409;
int32_t x20417 = x20416;
int32_t x20418 = x20414;
int32_t x20419 = x20415;
for(int x20420=0; x20420 < x20390; x20420++) {
int32_t x20421 = x20418;
float x20422 = x20309[x20421];
int32_t x20423 = x20419;
float x20424 = x18673[x20423];
float x20425 = x20422 + x20424;
x20309[x20421] = x20425;
x20417 += 1;
if (x20428) {
x20418 += 1;
} else {
}
if (x18640) {
x20419 += 1;
} else {
}

}
x20409 += x20390;
if (x20428) {
x20410 += x20296;
} else {
}
if (x18640) {
x20411 += x18508;
} else {
}

}
x20401 += x20391;
if (x20447) {
x20402 += x20297;
} else {
}
if (x18659) {
x20403 += x18509;
} else {
}

}
x20394 += x20392;
x20395 += x20298;
x20396 += x18510;

}
float* x20461 = (float*)myMalloc(x20299 * sizeof(float));;
for(int x20463=0; x20463 < x20299; x20463++) {
float x20464 = x20309[x20463];
bool x20465 = x20464 < 0.0f;
if (x20465) {
x20461[x20463] = 0.0f;
} else {
float x20468 = x20309[x20463];
x20461[x20463] = x20468;
}

}
float* x20482 = (float*)myMalloc(x20481 * sizeof(float));;
int32_t x20485 = 64 * x20294;
int32_t x20486 = x20485 * x20477;
float* x20487 = (float*)myMalloc(x20486 * sizeof(float));;
int32_t x20483 = x20294 * x20477;
for(int x20488=0; x20488 < 64; x20488++) {
int32_t x20489 = x20488 * x20298;
float* x20490 = x20461+x20489;
int32_t x20491 = x20488 * x20478;
float* x20492 = x20482+x20491;
int32_t x20493 = x20488 * x20483;
float* x20494 = x20487+x20493;
for(int x20495=0; x20495 < x20294; x20495++) {
int32_t x20496 = x20495 / 1;
int32_t x20500 = x20496 * x20476;
int32_t x20501 = x20500 * x20476;
int32_t x20497 = x20495 % 1;
int32_t x20498 = x20497 / 1;
int32_t x20502 = x20498 * x20476;
int32_t x20503 = x20502 * x20476;
int32_t x20504 = x20501 + x20503;
int32_t x20499 = x20497 % 1;
int32_t x20505 = x20499 * x20476;
int32_t x20506 = x20505 * x20476;
int32_t x20507 = x20504 + x20506;
float* x20508 = x20494+x20507;
int32_t x20509 = x20496 * x20296;
int32_t x20510 = x20509 * x20296;
float* x20511 = x20490+x20510;
for(int x20513=0; x20513 < x20476; x20513++) {
int32_t x20515 = x20513 * x20476;
float* x20516 = x20508+x20515;
int32_t x20514 = x20513 + x20498;
int32_t x20517 = x20514 * x20296;
int32_t x20518 = x20517 + x20499;
float* x20519 = x20511+x20518;
memcpy(x20516, x20519, 4 * x20476);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x20477,x20294,1,x6,x20294,x20494,x20477,1,x20492,x20477);

}
int32_t x20528 = 0;
int32_t x20529 = 1;
x20529 *= 1;
x20528 += 1;
x20529 *= 1;
x20529 *= 1;
int32_t x20534 = x20528;
bool x20535 = x20534 >= 2;
if (x20535) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20540 = x20534 == 0;
if (x20540) {
int32_t x20541 = x20529;
bool x20542 = x20541 == 256;
if (x20542) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20549 = x20529;
int32_t x20550 = 256 / x20549;
bool x20554;
if (x452) {
bool x20551 = x20550 == 1;
bool x20552 = 256 == x20550;
bool x20553 = x20551 || x20552;
x20554 = x20553;
} else {
x20554 = false;
}
bool x20558;
if (x20554) {
x20558 = x20557;
} else {
x20558 = false;
}
bool x20559;
if (x20558) {
x20559 = x20557;
} else {
x20559 = false;
}
if (x20559) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x20476,x20476,1,x20550,1,1);
assert(false && "");
}
bool x20565 = 256 <= x20550;
int32_t x20566;
if (x20565) {
x20566 = x20550;
} else {
x20566 = 256;
}
bool x20572 = x20566 > 0;
bool x20574;
if (x20572) {
x20574 = x20573;
} else {
x20574 = false;
}
bool x20575;
if (x20574) {
x20575 = x20573;
} else {
x20575 = false;
}
if (x20575) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(20476) x Sym(20476)"," x Const(1) x Sym(20550) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x20570 = x20566 * x20569;
int32_t x20571 = 64 * x20570;
float* x20581 = (float*)myMalloc(x20571 * sizeof(float));;
int32_t x20582 = 0;
int32_t x20583 = 0;
int32_t x20584 = 0;
bool x20631 = x20550 > 1;
for(int x20585=0; x20585 < 64; x20585++) {
int32_t x20586 = x20583;
int32_t x20587 = x20584;
int32_t x20588 = x20582;
int32_t x20589 = x20588;
int32_t x20590 = x20586;
int32_t x20591 = x20587;
for(int x20593=0; x20593 < x20566; x20593++) {
int32_t x20594 = x20590;
int32_t x20595 = x20591;
int32_t x20596 = x20589;
int32_t x20597 = x20596;
int32_t x20598 = x20594;
int32_t x20599 = x20595;
for(int x20601=0; x20601 < x20568; x20601++) {
int32_t x20602 = x20598;
int32_t x20603 = x20599;
int32_t x20604 = x20597;
int32_t x20605 = x20604;
int32_t x20606 = x20602;
int32_t x20607 = x20603;
for(int x20608=0; x20608 < x20568; x20608++) {
int32_t x20609 = x20605;
int32_t x20610 = x20606;
float x20611 = x20482[x20610];
int32_t x20612 = x20607;
float x20613 = x163[x20612];
float x20614 = x20611 - x20613;
x20581[x20609] = x20614;
x20605 += 1;
if (x20617) {
x20606 += 1;
} else {
}

}
x20597 += x20568;
if (x20617) {
x20598 += x20476;
} else {
}

}
x20589 += x20569;
x20590 += x20477;
if (x20631) {
x20591 += 1;
} else {
}

}
x20582 += x20570;
x20583 += x20478;

}
float* x20641 = (float*)myMalloc(256 * sizeof(float));;
for(int x20642=0; x20642 < 256; x20642++) {
float x20643 = x98[x20642];
float x20644 = x20643 + 1.0E-5f;
x20641[x20642] = x20644;

}
float* x20648 = (float*)myMalloc(256 * sizeof(float));;
for(int x20649=0; x20649 < 256; x20649++) {
float x20650 = x20641[x20649];
double x20651 = (double)x20650;
double x20652 = sqrt(x20651);
float x20653 = (float)x20652;
x20648[x20649] = x20653;

}
int32_t x20657 = 0;
int32_t x20658 = 1;
x20658 *= 1;
x20657 += 1;
x20658 *= 1;
x20658 *= 1;
int32_t x20663 = x20657;
bool x20664 = x20663 >= 2;
if (x20664) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20669 = x20663 == 0;
if (x20669) {
int32_t x20670 = x20658;
bool x20671 = x20670 == 256;
if (x20671) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20678 = x20658;
int32_t x20679 = 256 / x20678;
bool x20685;
if (x452) {
bool x20680 = x20566 == 1;
bool x20681 = x20679 == 1;
bool x20682 = x20680 || x20681;
bool x20683 = x20566 == x20679;
bool x20684 = x20682 || x20683;
x20685 = x20684;
} else {
x20685 = false;
}
bool x20689;
if (x20685) {
x20689 = x20688;
} else {
x20689 = false;
}
bool x20690;
if (x20689) {
x20690 = x20688;
} else {
x20690 = false;
}
if (x20690) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20566,x20568,x20568,1,x20679,1,1);
assert(false && "");
}
bool x20696 = x20566 <= x20679;
int32_t x20697;
if (x20696) {
x20697 = x20679;
} else {
x20697 = x20566;
}
bool x20703 = x20697 > 0;
bool x20705;
if (x20703) {
x20705 = x20704;
} else {
x20705 = false;
}
bool x20706;
if (x20705) {
x20706 = x20704;
} else {
x20706 = false;
}
if (x20706) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(20566) x Sym(20568) x Sym(20568)"," x Const(1) x Sym(20679) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x20701 = x20697 * x20700;
int32_t x20702 = 64 * x20701;
float* x20712 = (float*)myMalloc(x20702 * sizeof(float));;
int32_t x20713 = 0;
int32_t x20714 = 0;
int32_t x20715 = 0;
bool x20761 = x20566 > 1;
bool x20765 = x20679 > 1;
for(int x20716=0; x20716 < 64; x20716++) {
int32_t x20717 = x20714;
int32_t x20718 = x20715;
int32_t x20719 = x20713;
int32_t x20720 = x20719;
int32_t x20721 = x20717;
int32_t x20722 = x20718;
for(int x20724=0; x20724 < x20697; x20724++) {
int32_t x20725 = x20721;
int32_t x20726 = x20722;
int32_t x20727 = x20720;
int32_t x20728 = x20727;
int32_t x20729 = x20725;
int32_t x20730 = x20726;
for(int x20732=0; x20732 < x20699; x20732++) {
int32_t x20733 = x20729;
int32_t x20734 = x20730;
int32_t x20735 = x20728;
int32_t x20736 = x20735;
int32_t x20737 = x20733;
int32_t x20738 = x20734;
for(int x20739=0; x20739 < x20699; x20739++) {
int32_t x20740 = x20736;
int32_t x20741 = x20737;
float x20742 = x20581[x20741];
int32_t x20743 = x20738;
float x20744 = x20648[x20743];
float x20745 = x20742 / x20744;
x20712[x20740] = x20745;
x20736 += 1;
if (x20748) {
x20737 += 1;
} else {
}

}
x20728 += x20699;
if (x20748) {
x20729 += x20568;
} else {
}

}
x20720 += x20700;
if (x20761) {
x20721 += x20569;
} else {
}
if (x20765) {
x20722 += 1;
} else {
}

}
x20713 += x20701;
x20714 += x20570;

}
int32_t x20775 = 0;
int32_t x20776 = 1;
x20776 *= 1;
x20775 += 1;
x20776 *= 1;
x20776 *= 1;
int32_t x20781 = x20775;
bool x20782 = x20781 >= 2;
if (x20782) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20787 = x20781 == 0;
if (x20787) {
int32_t x20788 = x20776;
bool x20789 = x20788 == 256;
if (x20789) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20796 = x20776;
int32_t x20797 = 256 / x20796;
bool x20803;
if (x452) {
bool x20798 = x20697 == 1;
bool x20799 = x20797 == 1;
bool x20800 = x20798 || x20799;
bool x20801 = x20697 == x20797;
bool x20802 = x20800 || x20801;
x20803 = x20802;
} else {
x20803 = false;
}
bool x20807;
if (x20803) {
x20807 = x20806;
} else {
x20807 = false;
}
bool x20808;
if (x20807) {
x20808 = x20806;
} else {
x20808 = false;
}
if (x20808) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20697,x20699,x20699,1,x20797,1,1);
assert(false && "");
}
bool x20814 = x20697 <= x20797;
int32_t x20815;
if (x20814) {
x20815 = x20797;
} else {
x20815 = x20697;
}
bool x20821 = x20815 > 0;
bool x20823;
if (x20821) {
x20823 = x20822;
} else {
x20823 = false;
}
bool x20824;
if (x20823) {
x20824 = x20822;
} else {
x20824 = false;
}
if (x20824) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(20697) x Sym(20699) x Sym(20699)"," x Const(1) x Sym(20797) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x20819 = x20815 * x20818;
int32_t x20820 = 64 * x20819;
float* x20830 = (float*)myMalloc(x20820 * sizeof(float));;
int32_t x20831 = 0;
int32_t x20832 = 0;
int32_t x20833 = 0;
bool x20879 = x20697 > 1;
bool x20883 = x20797 > 1;
for(int x20834=0; x20834 < 64; x20834++) {
int32_t x20835 = x20832;
int32_t x20836 = x20833;
int32_t x20837 = x20831;
int32_t x20838 = x20837;
int32_t x20839 = x20835;
int32_t x20840 = x20836;
for(int x20842=0; x20842 < x20815; x20842++) {
int32_t x20843 = x20839;
int32_t x20844 = x20840;
int32_t x20845 = x20838;
int32_t x20846 = x20845;
int32_t x20847 = x20843;
int32_t x20848 = x20844;
for(int x20850=0; x20850 < x20817; x20850++) {
int32_t x20851 = x20847;
int32_t x20852 = x20848;
int32_t x20853 = x20846;
int32_t x20854 = x20853;
int32_t x20855 = x20851;
int32_t x20856 = x20852;
for(int x20857=0; x20857 < x20817; x20857++) {
int32_t x20858 = x20854;
int32_t x20859 = x20855;
float x20860 = x20712[x20859];
int32_t x20861 = x20856;
float x20862 = x92[x20861];
float x20863 = x20860 * x20862;
x20830[x20858] = x20863;
x20854 += 1;
if (x20866) {
x20855 += 1;
} else {
}

}
x20846 += x20817;
if (x20866) {
x20847 += x20699;
} else {
}

}
x20838 += x20818;
if (x20879) {
x20839 += x20700;
} else {
}
if (x20883) {
x20840 += 1;
} else {
}

}
x20831 += x20819;
x20832 += x20701;

}
int32_t x20893 = 0;
int32_t x20894 = 1;
x20894 *= 1;
x20893 += 1;
x20894 *= 1;
x20894 *= 1;
int32_t x20899 = x20893;
bool x20900 = x20899 >= 2;
if (x20900) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x20905 = x20899 == 0;
if (x20905) {
int32_t x20906 = x20894;
bool x20907 = x20906 == 256;
if (x20907) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x20914 = x20894;
int32_t x20915 = 256 / x20914;
bool x20921;
if (x452) {
bool x20916 = x20815 == 1;
bool x20917 = x20915 == 1;
bool x20918 = x20916 || x20917;
bool x20919 = x20815 == x20915;
bool x20920 = x20918 || x20919;
x20921 = x20920;
} else {
x20921 = false;
}
bool x20925;
if (x20921) {
x20925 = x20924;
} else {
x20925 = false;
}
bool x20926;
if (x20925) {
x20926 = x20924;
} else {
x20926 = false;
}
if (x20926) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x20815,x20817,x20817,1,x20915,1,1);
assert(false && "");
}
bool x20932 = x20815 <= x20915;
int32_t x20933;
if (x20932) {
x20933 = x20915;
} else {
x20933 = x20815;
}
bool x20939 = x20933 > 0;
bool x20941;
if (x20939) {
x20941 = x20940;
} else {
x20941 = false;
}
bool x20942;
if (x20941) {
x20942 = x20940;
} else {
x20942 = false;
}
if (x20942) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(20815) x Sym(20817) x Sym(20817)"," x Const(1) x Sym(20915) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x20937 = x20933 * x20936;
int32_t x20938 = 64 * x20937;
float* x20948 = (float*)myMalloc(x20938 * sizeof(float));;
int32_t x20949 = 0;
int32_t x20950 = 0;
int32_t x20951 = 0;
bool x20997 = x20815 > 1;
bool x21001 = x20915 > 1;
for(int x20952=0; x20952 < 64; x20952++) {
int32_t x20953 = x20950;
int32_t x20954 = x20951;
int32_t x20955 = x20949;
int32_t x20956 = x20955;
int32_t x20957 = x20953;
int32_t x20958 = x20954;
for(int x20960=0; x20960 < x20933; x20960++) {
int32_t x20961 = x20957;
int32_t x20962 = x20958;
int32_t x20963 = x20956;
int32_t x20964 = x20963;
int32_t x20965 = x20961;
int32_t x20966 = x20962;
for(int x20968=0; x20968 < x20935; x20968++) {
int32_t x20969 = x20965;
int32_t x20970 = x20966;
int32_t x20971 = x20964;
int32_t x20972 = x20971;
int32_t x20973 = x20969;
int32_t x20974 = x20970;
for(int x20975=0; x20975 < x20935; x20975++) {
int32_t x20976 = x20972;
int32_t x20977 = x20973;
float x20978 = x20830[x20977];
int32_t x20979 = x20974;
float x20980 = x241[x20979];
float x20981 = x20978 + x20980;
x20948[x20976] = x20981;
x20972 += 1;
if (x20984) {
x20973 += 1;
} else {
}

}
x20964 += x20935;
if (x20984) {
x20965 += x20817;
} else {
}

}
x20956 += x20936;
if (x20997) {
x20957 += x20818;
} else {
}
if (x21001) {
x20958 += 1;
} else {
}

}
x20949 += x20937;
x20950 += x20819;

}
float* x21011 = (float*)myMalloc(x20938 * sizeof(float));;
for(int x21013=0; x21013 < x20938; x21013++) {
float x21014 = x20948[x21013];
bool x21015 = x21014 < 0.0f;
if (x21015) {
x21011[x21013] = 0.0f;
} else {
float x21018 = x20948[x21013];
x21011[x21013] = x21018;
}

}
float* x21033 = (float*)myMalloc(x21032 * sizeof(float));;
int32_t x21034 = 9 * x20933;
int32_t x21037 = 64 * x21034;
int32_t x21038 = x21037 * x21028;
float* x21039 = (float*)myMalloc(x21038 * sizeof(float));;
int32_t x21035 = x21034 * x21028;
int32_t x21047 = x20933 * 3;
int32_t x21048 = x21047 * 3;
for(int x21040=0; x21040 < 64; x21040++) {
int32_t x21041 = x21040 * x20937;
float* x21042 = x21011+x21041;
int32_t x21043 = x21040 * x21029;
float* x21044 = x21033+x21043;
int32_t x21045 = x21040 * x21035;
float* x21046 = x21039+x21045;
for(int x21050=0; x21050 < x21048; x21050++) {
int32_t x21051 = x21050 / 9;
int32_t x21055 = x21051 * 3;
int32_t x21056 = x21055 * 3;
int32_t x21057 = x21056 * x21027;
int32_t x21058 = x21057 * x21027;
int32_t x21052 = x21050 % 9;
int32_t x21053 = x21052 / 3;
int32_t x21059 = x21053 * 3;
int32_t x21060 = x21059 * x21027;
int32_t x21061 = x21060 * x21027;
int32_t x21062 = x21058 + x21061;
int32_t x21054 = x21052 % 3;
int32_t x21063 = x21054 * x21027;
int32_t x21064 = x21063 * x21027;
int32_t x21065 = x21062 + x21064;
float* x21066 = x21046+x21065;
int32_t x21067 = x21051 * x20935;
int32_t x21068 = x21067 * x20935;
float* x21069 = x21042+x21068;
int32_t x21082 = 1 - x21054;
bool x21083 = x21082 > 0;
int32_t x21084;
if (x21083) {
x21084 = x21082;
} else {
x21084 = 0;
}
int32_t x21085 = 3 - x21054;
int32_t x21086 = x21085 - 1;
int32_t x21087 = 1 - x21086;
bool x21088 = x21087 > 0;
int32_t x21089;
if (x21088) {
x21089 = x21087;
} else {
x21089 = 0;
}
int32_t x21090 = x21027 - x21089;
int32_t x21091 = x21090 - x21084;
bool x21092 = x21091 <= 0;
bool x21096 = x21084 > 0;
int32_t x21081 = -1 + x21054;
bool x21109 = x21089 > 0;
for(int x21071=0; x21071 < x21027; x21071++) {
int32_t x21072 = x21071 - 1;
int32_t x21073 = x21072 + x21053;
bool x21074 = x21073 < 0;
bool x21075 = x21073 >= x20935;
bool x21076 = x21074 || x21075;
if (x21076) {
int32_t x21077 = x21071 * x21027;
float* x21078 = x21066+x21077;
memset(x21078, 0, 4 * x21027);;
} else {
if (x21092) {
int32_t x21077 = x21071 * x21027;
float* x21093 = x21066+x21077;
memset(x21093, 0, 4 * x21027);;
} else {
int32_t x21077 = x21071 * x21027;
if (x21096) {
float* x21097 = x21066+x21077;
memset(x21097, 0, 4 * x21084);;
} else {
}
// may have segfault here
int32_t x21102 = x21077 + x21084;
float* x21103 = x21066+x21102;
int32_t x21104 = x21073 * x20935;
int32_t x21105 = x21104 + x21081;
int32_t x21106 = x21105 + x21084;
float* x21107 = x21069+x21106;
memcpy(x21103, x21107, 4 * x21091);;
if (x21109) {
int32_t x21110 = x21077 + x21027;
int32_t x21111 = x21110 - x21089;
float* x21112 = x21066+x21111;
memset(x21112, 0, 4 * x21089);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x21028,x21034,1,x249,x21034,x21046,x21028,1,x21044,x21028);

}
int32_t x21127 = 0;
int32_t x21128 = 1;
x21128 *= 1;
x21127 += 1;
x21128 *= 1;
x21128 *= 1;
int32_t x21133 = x21127;
bool x21134 = x21133 >= 2;
if (x21134) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21139 = x21133 == 0;
if (x21139) {
int32_t x21140 = x21128;
bool x21141 = x21140 == 256;
if (x21141) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21148 = x21128;
int32_t x21149 = 256 / x21148;
bool x21153;
if (x452) {
bool x21150 = x21149 == 1;
bool x21151 = 256 == x21149;
bool x21152 = x21150 || x21151;
x21153 = x21152;
} else {
x21153 = false;
}
bool x21157;
if (x21153) {
x21157 = x21156;
} else {
x21157 = false;
}
bool x21158;
if (x21157) {
x21158 = x21156;
} else {
x21158 = false;
}
if (x21158) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x21027,x21027,1,x21149,1,1);
assert(false && "");
}
bool x21164 = 256 <= x21149;
int32_t x21165;
if (x21164) {
x21165 = x21149;
} else {
x21165 = 256;
}
bool x21171 = x21165 > 0;
bool x21173;
if (x21171) {
x21173 = x21172;
} else {
x21173 = false;
}
bool x21174;
if (x21173) {
x21174 = x21172;
} else {
x21174 = false;
}
if (x21174) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(21027) x Sym(21027)"," x Const(1) x Sym(21149) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x21169 = x21165 * x21168;
int32_t x21170 = 64 * x21169;
float* x21180 = (float*)myMalloc(x21170 * sizeof(float));;
int32_t x21181 = 0;
int32_t x21182 = 0;
int32_t x21183 = 0;
bool x21230 = x21149 > 1;
for(int x21184=0; x21184 < 64; x21184++) {
int32_t x21185 = x21182;
int32_t x21186 = x21183;
int32_t x21187 = x21181;
int32_t x21188 = x21187;
int32_t x21189 = x21185;
int32_t x21190 = x21186;
for(int x21192=0; x21192 < x21165; x21192++) {
int32_t x21193 = x21189;
int32_t x21194 = x21190;
int32_t x21195 = x21188;
int32_t x21196 = x21195;
int32_t x21197 = x21193;
int32_t x21198 = x21194;
for(int x21200=0; x21200 < x21167; x21200++) {
int32_t x21201 = x21197;
int32_t x21202 = x21198;
int32_t x21203 = x21196;
int32_t x21204 = x21203;
int32_t x21205 = x21201;
int32_t x21206 = x21202;
for(int x21207=0; x21207 < x21167; x21207++) {
int32_t x21208 = x21204;
int32_t x21209 = x21205;
float x21210 = x21033[x21209];
int32_t x21211 = x21206;
float x21212 = x186[x21211];
float x21213 = x21210 - x21212;
x21180[x21208] = x21213;
x21204 += 1;
if (x21216) {
x21205 += 1;
} else {
}

}
x21196 += x21167;
if (x21216) {
x21197 += x21027;
} else {
}

}
x21188 += x21168;
x21189 += x21028;
if (x21230) {
x21190 += 1;
} else {
}

}
x21181 += x21169;
x21182 += x21029;

}
float* x21240 = (float*)myMalloc(256 * sizeof(float));;
for(int x21241=0; x21241 < 256; x21241++) {
float x21242 = x230[x21241];
float x21243 = x21242 + 1.0E-5f;
x21240[x21241] = x21243;

}
float* x21247 = (float*)myMalloc(256 * sizeof(float));;
for(int x21248=0; x21248 < 256; x21248++) {
float x21249 = x21240[x21248];
double x21250 = (double)x21249;
double x21251 = sqrt(x21250);
float x21252 = (float)x21251;
x21247[x21248] = x21252;

}
int32_t x21256 = 0;
int32_t x21257 = 1;
x21257 *= 1;
x21256 += 1;
x21257 *= 1;
x21257 *= 1;
int32_t x21262 = x21256;
bool x21263 = x21262 >= 2;
if (x21263) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21268 = x21262 == 0;
if (x21268) {
int32_t x21269 = x21257;
bool x21270 = x21269 == 256;
if (x21270) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21277 = x21257;
int32_t x21278 = 256 / x21277;
bool x21284;
if (x452) {
bool x21279 = x21165 == 1;
bool x21280 = x21278 == 1;
bool x21281 = x21279 || x21280;
bool x21282 = x21165 == x21278;
bool x21283 = x21281 || x21282;
x21284 = x21283;
} else {
x21284 = false;
}
bool x21288;
if (x21284) {
x21288 = x21287;
} else {
x21288 = false;
}
bool x21289;
if (x21288) {
x21289 = x21287;
} else {
x21289 = false;
}
if (x21289) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21165,x21167,x21167,1,x21278,1,1);
assert(false && "");
}
bool x21295 = x21165 <= x21278;
int32_t x21296;
if (x21295) {
x21296 = x21278;
} else {
x21296 = x21165;
}
bool x21302 = x21296 > 0;
bool x21304;
if (x21302) {
x21304 = x21303;
} else {
x21304 = false;
}
bool x21305;
if (x21304) {
x21305 = x21303;
} else {
x21305 = false;
}
if (x21305) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(21165) x Sym(21167) x Sym(21167)"," x Const(1) x Sym(21278) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x21300 = x21296 * x21299;
int32_t x21301 = 64 * x21300;
float* x21311 = (float*)myMalloc(x21301 * sizeof(float));;
int32_t x21312 = 0;
int32_t x21313 = 0;
int32_t x21314 = 0;
bool x21360 = x21165 > 1;
bool x21364 = x21278 > 1;
for(int x21315=0; x21315 < 64; x21315++) {
int32_t x21316 = x21313;
int32_t x21317 = x21314;
int32_t x21318 = x21312;
int32_t x21319 = x21318;
int32_t x21320 = x21316;
int32_t x21321 = x21317;
for(int x21323=0; x21323 < x21296; x21323++) {
int32_t x21324 = x21320;
int32_t x21325 = x21321;
int32_t x21326 = x21319;
int32_t x21327 = x21326;
int32_t x21328 = x21324;
int32_t x21329 = x21325;
for(int x21331=0; x21331 < x21298; x21331++) {
int32_t x21332 = x21328;
int32_t x21333 = x21329;
int32_t x21334 = x21327;
int32_t x21335 = x21334;
int32_t x21336 = x21332;
int32_t x21337 = x21333;
for(int x21338=0; x21338 < x21298; x21338++) {
int32_t x21339 = x21335;
int32_t x21340 = x21336;
float x21341 = x21180[x21340];
int32_t x21342 = x21337;
float x21343 = x21247[x21342];
float x21344 = x21341 / x21343;
x21311[x21339] = x21344;
x21335 += 1;
if (x21347) {
x21336 += 1;
} else {
}

}
x21327 += x21298;
if (x21347) {
x21328 += x21167;
} else {
}

}
x21319 += x21299;
if (x21360) {
x21320 += x21168;
} else {
}
if (x21364) {
x21321 += 1;
} else {
}

}
x21312 += x21300;
x21313 += x21169;

}
int32_t x21374 = 0;
int32_t x21375 = 1;
x21375 *= 1;
x21374 += 1;
x21375 *= 1;
x21375 *= 1;
int32_t x21380 = x21374;
bool x21381 = x21380 >= 2;
if (x21381) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21386 = x21380 == 0;
if (x21386) {
int32_t x21387 = x21375;
bool x21388 = x21387 == 256;
if (x21388) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21395 = x21375;
int32_t x21396 = 256 / x21395;
bool x21402;
if (x452) {
bool x21397 = x21296 == 1;
bool x21398 = x21396 == 1;
bool x21399 = x21397 || x21398;
bool x21400 = x21296 == x21396;
bool x21401 = x21399 || x21400;
x21402 = x21401;
} else {
x21402 = false;
}
bool x21406;
if (x21402) {
x21406 = x21405;
} else {
x21406 = false;
}
bool x21407;
if (x21406) {
x21407 = x21405;
} else {
x21407 = false;
}
if (x21407) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21296,x21298,x21298,1,x21396,1,1);
assert(false && "");
}
bool x21413 = x21296 <= x21396;
int32_t x21414;
if (x21413) {
x21414 = x21396;
} else {
x21414 = x21296;
}
bool x21420 = x21414 > 0;
bool x21422;
if (x21420) {
x21422 = x21421;
} else {
x21422 = false;
}
bool x21423;
if (x21422) {
x21423 = x21421;
} else {
x21423 = false;
}
if (x21423) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(21296) x Sym(21298) x Sym(21298)"," x Const(1) x Sym(21396) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x21418 = x21414 * x21417;
int32_t x21419 = 64 * x21418;
float* x21429 = (float*)myMalloc(x21419 * sizeof(float));;
int32_t x21430 = 0;
int32_t x21431 = 0;
int32_t x21432 = 0;
bool x21478 = x21296 > 1;
bool x21482 = x21396 > 1;
for(int x21433=0; x21433 < 64; x21433++) {
int32_t x21434 = x21431;
int32_t x21435 = x21432;
int32_t x21436 = x21430;
int32_t x21437 = x21436;
int32_t x21438 = x21434;
int32_t x21439 = x21435;
for(int x21441=0; x21441 < x21414; x21441++) {
int32_t x21442 = x21438;
int32_t x21443 = x21439;
int32_t x21444 = x21437;
int32_t x21445 = x21444;
int32_t x21446 = x21442;
int32_t x21447 = x21443;
for(int x21449=0; x21449 < x21416; x21449++) {
int32_t x21450 = x21446;
int32_t x21451 = x21447;
int32_t x21452 = x21445;
int32_t x21453 = x21452;
int32_t x21454 = x21450;
int32_t x21455 = x21451;
for(int x21456=0; x21456 < x21416; x21456++) {
int32_t x21457 = x21453;
int32_t x21458 = x21454;
float x21459 = x21311[x21458];
int32_t x21460 = x21455;
float x21461 = x74[x21460];
float x21462 = x21459 * x21461;
x21429[x21457] = x21462;
x21453 += 1;
if (x21465) {
x21454 += 1;
} else {
}

}
x21445 += x21416;
if (x21465) {
x21446 += x21298;
} else {
}

}
x21437 += x21417;
if (x21478) {
x21438 += x21299;
} else {
}
if (x21482) {
x21439 += 1;
} else {
}

}
x21430 += x21418;
x21431 += x21300;

}
int32_t x21492 = 0;
int32_t x21493 = 1;
x21493 *= 1;
x21492 += 1;
x21493 *= 1;
x21493 *= 1;
int32_t x21498 = x21492;
bool x21499 = x21498 >= 2;
if (x21499) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21504 = x21498 == 0;
if (x21504) {
int32_t x21505 = x21493;
bool x21506 = x21505 == 256;
if (x21506) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21513 = x21493;
int32_t x21514 = 256 / x21513;
bool x21520;
if (x452) {
bool x21515 = x21414 == 1;
bool x21516 = x21514 == 1;
bool x21517 = x21515 || x21516;
bool x21518 = x21414 == x21514;
bool x21519 = x21517 || x21518;
x21520 = x21519;
} else {
x21520 = false;
}
bool x21524;
if (x21520) {
x21524 = x21523;
} else {
x21524 = false;
}
bool x21525;
if (x21524) {
x21525 = x21523;
} else {
x21525 = false;
}
if (x21525) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21414,x21416,x21416,1,x21514,1,1);
assert(false && "");
}
bool x21531 = x21414 <= x21514;
int32_t x21532;
if (x21531) {
x21532 = x21514;
} else {
x21532 = x21414;
}
bool x21538 = x21532 > 0;
bool x21540;
if (x21538) {
x21540 = x21539;
} else {
x21540 = false;
}
bool x21541;
if (x21540) {
x21541 = x21539;
} else {
x21541 = false;
}
if (x21541) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(21414) x Sym(21416) x Sym(21416)"," x Const(1) x Sym(21514) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x21536 = x21532 * x21535;
int32_t x21537 = 64 * x21536;
float* x21547 = (float*)myMalloc(x21537 * sizeof(float));;
int32_t x21548 = 0;
int32_t x21549 = 0;
int32_t x21550 = 0;
bool x21596 = x21414 > 1;
bool x21600 = x21514 > 1;
for(int x21551=0; x21551 < 64; x21551++) {
int32_t x21552 = x21549;
int32_t x21553 = x21550;
int32_t x21554 = x21548;
int32_t x21555 = x21554;
int32_t x21556 = x21552;
int32_t x21557 = x21553;
for(int x21559=0; x21559 < x21532; x21559++) {
int32_t x21560 = x21556;
int32_t x21561 = x21557;
int32_t x21562 = x21555;
int32_t x21563 = x21562;
int32_t x21564 = x21560;
int32_t x21565 = x21561;
for(int x21567=0; x21567 < x21534; x21567++) {
int32_t x21568 = x21564;
int32_t x21569 = x21565;
int32_t x21570 = x21563;
int32_t x21571 = x21570;
int32_t x21572 = x21568;
int32_t x21573 = x21569;
for(int x21574=0; x21574 < x21534; x21574++) {
int32_t x21575 = x21571;
int32_t x21576 = x21572;
float x21577 = x21429[x21576];
int32_t x21578 = x21573;
float x21579 = x136[x21578];
float x21580 = x21577 + x21579;
x21547[x21575] = x21580;
x21571 += 1;
if (x21583) {
x21572 += 1;
} else {
}

}
x21563 += x21534;
if (x21583) {
x21564 += x21416;
} else {
}

}
x21555 += x21535;
if (x21596) {
x21556 += x21417;
} else {
}
if (x21600) {
x21557 += 1;
} else {
}

}
x21548 += x21536;
x21549 += x21418;

}
float* x21610 = (float*)myMalloc(x21537 * sizeof(float));;
for(int x21612=0; x21612 < x21537; x21612++) {
float x21613 = x21547[x21612];
bool x21614 = x21613 < 0.0f;
if (x21614) {
x21610[x21612] = 0.0f;
} else {
float x21617 = x21547[x21612];
x21610[x21612] = x21617;
}

}
float* x21631 = (float*)myMalloc(x21630 * sizeof(float));;
int32_t x21634 = 64 * x21532;
int32_t x21635 = x21634 * x21626;
float* x21636 = (float*)myMalloc(x21635 * sizeof(float));;
int32_t x21632 = x21532 * x21626;
for(int x21637=0; x21637 < 64; x21637++) {
int32_t x21638 = x21637 * x21536;
float* x21639 = x21610+x21638;
int32_t x21640 = x21637 * x21627;
float* x21641 = x21631+x21640;
int32_t x21642 = x21637 * x21632;
float* x21643 = x21636+x21642;
for(int x21644=0; x21644 < x21532; x21644++) {
int32_t x21645 = x21644 / 1;
int32_t x21649 = x21645 * x21625;
int32_t x21650 = x21649 * x21625;
int32_t x21646 = x21644 % 1;
int32_t x21647 = x21646 / 1;
int32_t x21651 = x21647 * x21625;
int32_t x21652 = x21651 * x21625;
int32_t x21653 = x21650 + x21652;
int32_t x21648 = x21646 % 1;
int32_t x21654 = x21648 * x21625;
int32_t x21655 = x21654 * x21625;
int32_t x21656 = x21653 + x21655;
float* x21657 = x21643+x21656;
int32_t x21658 = x21645 * x21534;
int32_t x21659 = x21658 * x21534;
float* x21660 = x21639+x21659;
for(int x21662=0; x21662 < x21625; x21662++) {
int32_t x21664 = x21662 * x21625;
float* x21665 = x21657+x21664;
int32_t x21663 = x21662 + x21647;
int32_t x21666 = x21663 * x21534;
int32_t x21667 = x21666 + x21648;
float* x21668 = x21660+x21667;
memcpy(x21665, x21668, 4 * x21625);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x21626,x21532,1,x89,x21532,x21643,x21626,1,x21641,x21626);

}
int32_t x21677 = 0;
int32_t x21678 = 1;
x21678 *= 1;
x21677 += 1;
x21678 *= 1;
x21678 *= 1;
int32_t x21683 = x21677;
bool x21684 = x21683 >= 2;
if (x21684) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21689 = x21683 == 0;
if (x21689) {
int32_t x21690 = x21678;
bool x21691 = x21690 == 1024;
if (x21691) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21698 = x21678;
int32_t x21699 = 1024 / x21698;
bool x21703;
if (x452) {
bool x21700 = x21699 == 1;
bool x21701 = 1024 == x21699;
bool x21702 = x21700 || x21701;
x21703 = x21702;
} else {
x21703 = false;
}
bool x21707;
if (x21703) {
x21707 = x21706;
} else {
x21707 = false;
}
bool x21708;
if (x21707) {
x21708 = x21706;
} else {
x21708 = false;
}
if (x21708) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x21625,x21625,1,x21699,1,1);
assert(false && "");
}
bool x21714 = 1024 <= x21699;
int32_t x21715;
if (x21714) {
x21715 = x21699;
} else {
x21715 = 1024;
}
bool x21721 = x21715 > 0;
bool x21723;
if (x21721) {
x21723 = x21722;
} else {
x21723 = false;
}
bool x21724;
if (x21723) {
x21724 = x21722;
} else {
x21724 = false;
}
if (x21724) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(1024) x Sym(21625) x Sym(21625)"," x Const(1) x Sym(21699) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x21719 = x21715 * x21718;
int32_t x21720 = 64 * x21719;
float* x21730 = (float*)myMalloc(x21720 * sizeof(float));;
int32_t x21731 = 0;
int32_t x21732 = 0;
int32_t x21733 = 0;
bool x21780 = x21699 > 1;
for(int x21734=0; x21734 < 64; x21734++) {
int32_t x21735 = x21732;
int32_t x21736 = x21733;
int32_t x21737 = x21731;
int32_t x21738 = x21737;
int32_t x21739 = x21735;
int32_t x21740 = x21736;
for(int x21742=0; x21742 < x21715; x21742++) {
int32_t x21743 = x21739;
int32_t x21744 = x21740;
int32_t x21745 = x21738;
int32_t x21746 = x21745;
int32_t x21747 = x21743;
int32_t x21748 = x21744;
for(int x21750=0; x21750 < x21717; x21750++) {
int32_t x21751 = x21747;
int32_t x21752 = x21748;
int32_t x21753 = x21746;
int32_t x21754 = x21753;
int32_t x21755 = x21751;
int32_t x21756 = x21752;
for(int x21757=0; x21757 < x21717; x21757++) {
int32_t x21758 = x21754;
int32_t x21759 = x21755;
float x21760 = x21631[x21759];
int32_t x21761 = x21756;
float x21762 = x231[x21761];
float x21763 = x21760 - x21762;
x21730[x21758] = x21763;
x21754 += 1;
if (x21766) {
x21755 += 1;
} else {
}

}
x21746 += x21717;
if (x21766) {
x21747 += x21625;
} else {
}

}
x21738 += x21718;
x21739 += x21626;
if (x21780) {
x21740 += 1;
} else {
}

}
x21731 += x21719;
x21732 += x21627;

}
float* x21790 = (float*)myMalloc(1024 * sizeof(float));;
for(int x21791=0; x21791 < 1024; x21791++) {
float x21792 = x161[x21791];
float x21793 = x21792 + 1.0E-5f;
x21790[x21791] = x21793;

}
float* x21797 = (float*)myMalloc(1024 * sizeof(float));;
for(int x21798=0; x21798 < 1024; x21798++) {
float x21799 = x21790[x21798];
double x21800 = (double)x21799;
double x21801 = sqrt(x21800);
float x21802 = (float)x21801;
x21797[x21798] = x21802;

}
int32_t x21806 = 0;
int32_t x21807 = 1;
x21807 *= 1;
x21806 += 1;
x21807 *= 1;
x21807 *= 1;
int32_t x21812 = x21806;
bool x21813 = x21812 >= 2;
if (x21813) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21818 = x21812 == 0;
if (x21818) {
int32_t x21819 = x21807;
bool x21820 = x21819 == 1024;
if (x21820) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21827 = x21807;
int32_t x21828 = 1024 / x21827;
bool x21834;
if (x452) {
bool x21829 = x21715 == 1;
bool x21830 = x21828 == 1;
bool x21831 = x21829 || x21830;
bool x21832 = x21715 == x21828;
bool x21833 = x21831 || x21832;
x21834 = x21833;
} else {
x21834 = false;
}
bool x21838;
if (x21834) {
x21838 = x21837;
} else {
x21838 = false;
}
bool x21839;
if (x21838) {
x21839 = x21837;
} else {
x21839 = false;
}
if (x21839) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21715,x21717,x21717,1,x21828,1,1);
assert(false && "");
}
bool x21845 = x21715 <= x21828;
int32_t x21846;
if (x21845) {
x21846 = x21828;
} else {
x21846 = x21715;
}
bool x21852 = x21846 > 0;
bool x21854;
if (x21852) {
x21854 = x21853;
} else {
x21854 = false;
}
bool x21855;
if (x21854) {
x21855 = x21853;
} else {
x21855 = false;
}
if (x21855) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(21715) x Sym(21717) x Sym(21717)"," x Const(1) x Sym(21828) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x21850 = x21846 * x21849;
int32_t x21851 = 64 * x21850;
float* x21861 = (float*)myMalloc(x21851 * sizeof(float));;
int32_t x21862 = 0;
int32_t x21863 = 0;
int32_t x21864 = 0;
bool x21910 = x21715 > 1;
bool x21914 = x21828 > 1;
for(int x21865=0; x21865 < 64; x21865++) {
int32_t x21866 = x21863;
int32_t x21867 = x21864;
int32_t x21868 = x21862;
int32_t x21869 = x21868;
int32_t x21870 = x21866;
int32_t x21871 = x21867;
for(int x21873=0; x21873 < x21846; x21873++) {
int32_t x21874 = x21870;
int32_t x21875 = x21871;
int32_t x21876 = x21869;
int32_t x21877 = x21876;
int32_t x21878 = x21874;
int32_t x21879 = x21875;
for(int x21881=0; x21881 < x21848; x21881++) {
int32_t x21882 = x21878;
int32_t x21883 = x21879;
int32_t x21884 = x21877;
int32_t x21885 = x21884;
int32_t x21886 = x21882;
int32_t x21887 = x21883;
for(int x21888=0; x21888 < x21848; x21888++) {
int32_t x21889 = x21885;
int32_t x21890 = x21886;
float x21891 = x21730[x21890];
int32_t x21892 = x21887;
float x21893 = x21797[x21892];
float x21894 = x21891 / x21893;
x21861[x21889] = x21894;
x21885 += 1;
if (x21897) {
x21886 += 1;
} else {
}

}
x21877 += x21848;
if (x21897) {
x21878 += x21717;
} else {
}

}
x21869 += x21849;
if (x21910) {
x21870 += x21718;
} else {
}
if (x21914) {
x21871 += 1;
} else {
}

}
x21862 += x21850;
x21863 += x21719;

}
int32_t x21924 = 0;
int32_t x21925 = 1;
x21925 *= 1;
x21924 += 1;
x21925 *= 1;
x21925 *= 1;
int32_t x21930 = x21924;
bool x21931 = x21930 >= 2;
if (x21931) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x21936 = x21930 == 0;
if (x21936) {
int32_t x21937 = x21925;
bool x21938 = x21937 == 1024;
if (x21938) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x21945 = x21925;
int32_t x21946 = 1024 / x21945;
bool x21952;
if (x452) {
bool x21947 = x21846 == 1;
bool x21948 = x21946 == 1;
bool x21949 = x21947 || x21948;
bool x21950 = x21846 == x21946;
bool x21951 = x21949 || x21950;
x21952 = x21951;
} else {
x21952 = false;
}
bool x21956;
if (x21952) {
x21956 = x21955;
} else {
x21956 = false;
}
bool x21957;
if (x21956) {
x21957 = x21955;
} else {
x21957 = false;
}
if (x21957) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21846,x21848,x21848,1,x21946,1,1);
assert(false && "");
}
bool x21963 = x21846 <= x21946;
int32_t x21964;
if (x21963) {
x21964 = x21946;
} else {
x21964 = x21846;
}
bool x21970 = x21964 > 0;
bool x21972;
if (x21970) {
x21972 = x21971;
} else {
x21972 = false;
}
bool x21973;
if (x21972) {
x21973 = x21971;
} else {
x21973 = false;
}
if (x21973) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(21846) x Sym(21848) x Sym(21848)"," x Const(1) x Sym(21946) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x21968 = x21964 * x21967;
int32_t x21969 = 64 * x21968;
float* x21979 = (float*)myMalloc(x21969 * sizeof(float));;
int32_t x21980 = 0;
int32_t x21981 = 0;
int32_t x21982 = 0;
bool x22028 = x21846 > 1;
bool x22032 = x21946 > 1;
for(int x21983=0; x21983 < 64; x21983++) {
int32_t x21984 = x21981;
int32_t x21985 = x21982;
int32_t x21986 = x21980;
int32_t x21987 = x21986;
int32_t x21988 = x21984;
int32_t x21989 = x21985;
for(int x21991=0; x21991 < x21964; x21991++) {
int32_t x21992 = x21988;
int32_t x21993 = x21989;
int32_t x21994 = x21987;
int32_t x21995 = x21994;
int32_t x21996 = x21992;
int32_t x21997 = x21993;
for(int x21999=0; x21999 < x21966; x21999++) {
int32_t x22000 = x21996;
int32_t x22001 = x21997;
int32_t x22002 = x21995;
int32_t x22003 = x22002;
int32_t x22004 = x22000;
int32_t x22005 = x22001;
for(int x22006=0; x22006 < x21966; x22006++) {
int32_t x22007 = x22003;
int32_t x22008 = x22004;
float x22009 = x21861[x22008];
int32_t x22010 = x22005;
float x22011 = x238[x22010];
float x22012 = x22009 * x22011;
x21979[x22007] = x22012;
x22003 += 1;
if (x22015) {
x22004 += 1;
} else {
}

}
x21995 += x21966;
if (x22015) {
x21996 += x21848;
} else {
}

}
x21987 += x21967;
if (x22028) {
x21988 += x21849;
} else {
}
if (x22032) {
x21989 += 1;
} else {
}

}
x21980 += x21968;
x21981 += x21850;

}
int32_t x22042 = 0;
int32_t x22043 = 1;
x22043 *= 1;
x22042 += 1;
x22043 *= 1;
x22043 *= 1;
int32_t x22048 = x22042;
bool x22049 = x22048 >= 2;
if (x22049) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22054 = x22048 == 0;
if (x22054) {
int32_t x22055 = x22043;
bool x22056 = x22055 == 1024;
if (x22056) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22063 = x22043;
int32_t x22064 = 1024 / x22063;
bool x22070;
if (x452) {
bool x22065 = x21964 == 1;
bool x22066 = x22064 == 1;
bool x22067 = x22065 || x22066;
bool x22068 = x21964 == x22064;
bool x22069 = x22067 || x22068;
x22070 = x22069;
} else {
x22070 = false;
}
bool x22074;
if (x22070) {
x22074 = x22073;
} else {
x22074 = false;
}
bool x22075;
if (x22074) {
x22075 = x22073;
} else {
x22075 = false;
}
if (x22075) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x21964,x21966,x21966,1,x22064,1,1);
assert(false && "");
}
bool x22081 = x21964 <= x22064;
int32_t x22082;
if (x22081) {
x22082 = x22064;
} else {
x22082 = x21964;
}
bool x22088 = x22082 > 0;
bool x22090;
if (x22088) {
x22090 = x22089;
} else {
x22090 = false;
}
bool x22091;
if (x22090) {
x22091 = x22089;
} else {
x22091 = false;
}
if (x22091) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(21964) x Sym(21966) x Sym(21966)"," x Const(1) x Sym(22064) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x22086 = x22082 * x22085;
int32_t x22087 = 64 * x22086;
float* x22097 = (float*)myMalloc(x22087 * sizeof(float));;
int32_t x22098 = 0;
int32_t x22099 = 0;
int32_t x22100 = 0;
bool x22146 = x21964 > 1;
bool x22150 = x22064 > 1;
for(int x22101=0; x22101 < 64; x22101++) {
int32_t x22102 = x22099;
int32_t x22103 = x22100;
int32_t x22104 = x22098;
int32_t x22105 = x22104;
int32_t x22106 = x22102;
int32_t x22107 = x22103;
for(int x22109=0; x22109 < x22082; x22109++) {
int32_t x22110 = x22106;
int32_t x22111 = x22107;
int32_t x22112 = x22105;
int32_t x22113 = x22112;
int32_t x22114 = x22110;
int32_t x22115 = x22111;
for(int x22117=0; x22117 < x22084; x22117++) {
int32_t x22118 = x22114;
int32_t x22119 = x22115;
int32_t x22120 = x22113;
int32_t x22121 = x22120;
int32_t x22122 = x22118;
int32_t x22123 = x22119;
for(int x22124=0; x22124 < x22084; x22124++) {
int32_t x22125 = x22121;
int32_t x22126 = x22122;
float x22127 = x21979[x22126];
int32_t x22128 = x22123;
float x22129 = x146[x22128];
float x22130 = x22127 + x22129;
x22097[x22125] = x22130;
x22121 += 1;
if (x22133) {
x22122 += 1;
} else {
}

}
x22113 += x22084;
if (x22133) {
x22114 += x21966;
} else {
}

}
x22105 += x22085;
if (x22146) {
x22106 += x21967;
} else {
}
if (x22150) {
x22107 += 1;
} else {
}

}
x22098 += x22086;
x22099 += x21968;

}
bool x22160 = x22082 == 1;
bool x22161 = x22160 || x20372;
bool x22162 = x22082 == x20294;
bool x22163 = x22161 || x22162;
bool x22168;
if (x22163) {
x22168 = x22167;
} else {
x22168 = false;
}
bool x22169;
if (x22168) {
x22169 = x22167;
} else {
x22169 = false;
}
if (x22169) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22082,x22084,x22084,64,x20294,x20296,x20296);
assert(false && "");
}
int32_t x22182 = 0;
int32_t x22183 = 0;
int32_t x22184 = 0;
bool x22175 = x22082 <= x20294;
int32_t x22176;
if (x22175) {
x22176 = x20294;
} else {
x22176 = x22082;
}
bool x22235 = x22082 > 1;
int32_t x22180 = x22176 * x22179;
for(int x22185=0; x22185 < 64; x22185++) {
int32_t x22186 = x22183;
int32_t x22187 = x22184;
int32_t x22188 = x22182;
int32_t x22189 = x22188;
int32_t x22190 = x22186;
int32_t x22191 = x22187;
for(int x22193=0; x22193 < x22176; x22193++) {
int32_t x22194 = x22190;
int32_t x22195 = x22191;
int32_t x22196 = x22189;
int32_t x22197 = x22196;
int32_t x22198 = x22194;
int32_t x22199 = x22195;
for(int x22201=0; x22201 < x22178; x22201++) {
int32_t x22202 = x22198;
int32_t x22203 = x22199;
int32_t x22204 = x22197;
int32_t x22205 = x22204;
int32_t x22206 = x22202;
int32_t x22207 = x22203;
for(int x22208=0; x22208 < x22178; x22208++) {
int32_t x22209 = x22206;
float x22210 = x22097[x22209];
int32_t x22211 = x22207;
float x22212 = x20461[x22211];
float x22213 = x22210 + x22212;
x22097[x22209] = x22213;
x22205 += 1;
if (x22216) {
x22206 += 1;
} else {
}
if (x20428) {
x22207 += 1;
} else {
}

}
x22197 += x22178;
if (x22216) {
x22198 += x22084;
} else {
}
if (x20428) {
x22199 += x20296;
} else {
}

}
x22189 += x22179;
if (x22235) {
x22190 += x22085;
} else {
}
if (x20447) {
x22191 += x20297;
} else {
}

}
x22182 += x22180;
x22183 += x22086;
x22184 += x20298;

}
float* x22249 = (float*)myMalloc(x22087 * sizeof(float));;
for(int x22251=0; x22251 < x22087; x22251++) {
float x22252 = x22097[x22251];
bool x22253 = x22252 < 0.0f;
if (x22253) {
x22249[x22251] = 0.0f;
} else {
float x22256 = x22097[x22251];
x22249[x22251] = x22256;
}

}
float* x22270 = (float*)myMalloc(x22269 * sizeof(float));;
int32_t x22273 = 64 * x22082;
int32_t x22274 = x22273 * x22265;
float* x22275 = (float*)myMalloc(x22274 * sizeof(float));;
int32_t x22271 = x22082 * x22265;
for(int x22276=0; x22276 < 64; x22276++) {
int32_t x22277 = x22276 * x22086;
float* x22278 = x22249+x22277;
int32_t x22279 = x22276 * x22266;
float* x22280 = x22270+x22279;
int32_t x22281 = x22276 * x22271;
float* x22282 = x22275+x22281;
for(int x22283=0; x22283 < x22082; x22283++) {
int32_t x22284 = x22283 / 1;
int32_t x22288 = x22284 * x22264;
int32_t x22289 = x22288 * x22264;
int32_t x22285 = x22283 % 1;
int32_t x22286 = x22285 / 1;
int32_t x22290 = x22286 * x22264;
int32_t x22291 = x22290 * x22264;
int32_t x22292 = x22289 + x22291;
int32_t x22287 = x22285 % 1;
int32_t x22293 = x22287 * x22264;
int32_t x22294 = x22293 * x22264;
int32_t x22295 = x22292 + x22294;
float* x22296 = x22282+x22295;
int32_t x22297 = x22284 * x22084;
int32_t x22298 = x22297 * x22084;
float* x22299 = x22278+x22298;
for(int x22301=0; x22301 < x22264; x22301++) {
int32_t x22303 = x22301 * x22264;
float* x22304 = x22296+x22303;
int32_t x22302 = x22301 + x22286;
int32_t x22305 = x22302 * x22084;
int32_t x22306 = x22305 + x22287;
float* x22307 = x22299+x22306;
memcpy(x22304, x22307, 4 * x22264);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x22265,x22082,1,x22,x22082,x22282,x22265,1,x22280,x22265);

}
int32_t x22316 = 0;
int32_t x22317 = 1;
x22317 *= 1;
x22316 += 1;
x22317 *= 1;
x22317 *= 1;
int32_t x22322 = x22316;
bool x22323 = x22322 >= 2;
if (x22323) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22328 = x22322 == 0;
if (x22328) {
int32_t x22329 = x22317;
bool x22330 = x22329 == 256;
if (x22330) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22337 = x22317;
int32_t x22338 = 256 / x22337;
bool x22342;
if (x452) {
bool x22339 = x22338 == 1;
bool x22340 = 256 == x22338;
bool x22341 = x22339 || x22340;
x22342 = x22341;
} else {
x22342 = false;
}
bool x22346;
if (x22342) {
x22346 = x22345;
} else {
x22346 = false;
}
bool x22347;
if (x22346) {
x22347 = x22345;
} else {
x22347 = false;
}
if (x22347) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x22264,x22264,1,x22338,1,1);
assert(false && "");
}
bool x22353 = 256 <= x22338;
int32_t x22354;
if (x22353) {
x22354 = x22338;
} else {
x22354 = 256;
}
bool x22360 = x22354 > 0;
bool x22362;
if (x22360) {
x22362 = x22361;
} else {
x22362 = false;
}
bool x22363;
if (x22362) {
x22363 = x22361;
} else {
x22363 = false;
}
if (x22363) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(22264) x Sym(22264)"," x Const(1) x Sym(22338) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x22358 = x22354 * x22357;
int32_t x22359 = 64 * x22358;
float* x22369 = (float*)myMalloc(x22359 * sizeof(float));;
int32_t x22370 = 0;
int32_t x22371 = 0;
int32_t x22372 = 0;
bool x22419 = x22338 > 1;
for(int x22373=0; x22373 < 64; x22373++) {
int32_t x22374 = x22371;
int32_t x22375 = x22372;
int32_t x22376 = x22370;
int32_t x22377 = x22376;
int32_t x22378 = x22374;
int32_t x22379 = x22375;
for(int x22381=0; x22381 < x22354; x22381++) {
int32_t x22382 = x22378;
int32_t x22383 = x22379;
int32_t x22384 = x22377;
int32_t x22385 = x22384;
int32_t x22386 = x22382;
int32_t x22387 = x22383;
for(int x22389=0; x22389 < x22356; x22389++) {
int32_t x22390 = x22386;
int32_t x22391 = x22387;
int32_t x22392 = x22385;
int32_t x22393 = x22392;
int32_t x22394 = x22390;
int32_t x22395 = x22391;
for(int x22396=0; x22396 < x22356; x22396++) {
int32_t x22397 = x22393;
int32_t x22398 = x22394;
float x22399 = x22270[x22398];
int32_t x22400 = x22395;
float x22401 = x254[x22400];
float x22402 = x22399 - x22401;
x22369[x22397] = x22402;
x22393 += 1;
if (x22405) {
x22394 += 1;
} else {
}

}
x22385 += x22356;
if (x22405) {
x22386 += x22264;
} else {
}

}
x22377 += x22357;
x22378 += x22265;
if (x22419) {
x22379 += 1;
} else {
}

}
x22370 += x22358;
x22371 += x22266;

}
float* x22429 = (float*)myMalloc(256 * sizeof(float));;
for(int x22430=0; x22430 < 256; x22430++) {
float x22431 = x69[x22430];
float x22432 = x22431 + 1.0E-5f;
x22429[x22430] = x22432;

}
float* x22436 = (float*)myMalloc(256 * sizeof(float));;
for(int x22437=0; x22437 < 256; x22437++) {
float x22438 = x22429[x22437];
double x22439 = (double)x22438;
double x22440 = sqrt(x22439);
float x22441 = (float)x22440;
x22436[x22437] = x22441;

}
int32_t x22445 = 0;
int32_t x22446 = 1;
x22446 *= 1;
x22445 += 1;
x22446 *= 1;
x22446 *= 1;
int32_t x22451 = x22445;
bool x22452 = x22451 >= 2;
if (x22452) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22457 = x22451 == 0;
if (x22457) {
int32_t x22458 = x22446;
bool x22459 = x22458 == 256;
if (x22459) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22466 = x22446;
int32_t x22467 = 256 / x22466;
bool x22473;
if (x452) {
bool x22468 = x22354 == 1;
bool x22469 = x22467 == 1;
bool x22470 = x22468 || x22469;
bool x22471 = x22354 == x22467;
bool x22472 = x22470 || x22471;
x22473 = x22472;
} else {
x22473 = false;
}
bool x22477;
if (x22473) {
x22477 = x22476;
} else {
x22477 = false;
}
bool x22478;
if (x22477) {
x22478 = x22476;
} else {
x22478 = false;
}
if (x22478) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22354,x22356,x22356,1,x22467,1,1);
assert(false && "");
}
bool x22484 = x22354 <= x22467;
int32_t x22485;
if (x22484) {
x22485 = x22467;
} else {
x22485 = x22354;
}
bool x22491 = x22485 > 0;
bool x22493;
if (x22491) {
x22493 = x22492;
} else {
x22493 = false;
}
bool x22494;
if (x22493) {
x22494 = x22492;
} else {
x22494 = false;
}
if (x22494) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(22354) x Sym(22356) x Sym(22356)"," x Const(1) x Sym(22467) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x22489 = x22485 * x22488;
int32_t x22490 = 64 * x22489;
float* x22500 = (float*)myMalloc(x22490 * sizeof(float));;
int32_t x22501 = 0;
int32_t x22502 = 0;
int32_t x22503 = 0;
bool x22549 = x22354 > 1;
bool x22553 = x22467 > 1;
for(int x22504=0; x22504 < 64; x22504++) {
int32_t x22505 = x22502;
int32_t x22506 = x22503;
int32_t x22507 = x22501;
int32_t x22508 = x22507;
int32_t x22509 = x22505;
int32_t x22510 = x22506;
for(int x22512=0; x22512 < x22485; x22512++) {
int32_t x22513 = x22509;
int32_t x22514 = x22510;
int32_t x22515 = x22508;
int32_t x22516 = x22515;
int32_t x22517 = x22513;
int32_t x22518 = x22514;
for(int x22520=0; x22520 < x22487; x22520++) {
int32_t x22521 = x22517;
int32_t x22522 = x22518;
int32_t x22523 = x22516;
int32_t x22524 = x22523;
int32_t x22525 = x22521;
int32_t x22526 = x22522;
for(int x22527=0; x22527 < x22487; x22527++) {
int32_t x22528 = x22524;
int32_t x22529 = x22525;
float x22530 = x22369[x22529];
int32_t x22531 = x22526;
float x22532 = x22436[x22531];
float x22533 = x22530 / x22532;
x22500[x22528] = x22533;
x22524 += 1;
if (x22536) {
x22525 += 1;
} else {
}

}
x22516 += x22487;
if (x22536) {
x22517 += x22356;
} else {
}

}
x22508 += x22488;
if (x22549) {
x22509 += x22357;
} else {
}
if (x22553) {
x22510 += 1;
} else {
}

}
x22501 += x22489;
x22502 += x22358;

}
int32_t x22563 = 0;
int32_t x22564 = 1;
x22564 *= 1;
x22563 += 1;
x22564 *= 1;
x22564 *= 1;
int32_t x22569 = x22563;
bool x22570 = x22569 >= 2;
if (x22570) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22575 = x22569 == 0;
if (x22575) {
int32_t x22576 = x22564;
bool x22577 = x22576 == 256;
if (x22577) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22584 = x22564;
int32_t x22585 = 256 / x22584;
bool x22591;
if (x452) {
bool x22586 = x22485 == 1;
bool x22587 = x22585 == 1;
bool x22588 = x22586 || x22587;
bool x22589 = x22485 == x22585;
bool x22590 = x22588 || x22589;
x22591 = x22590;
} else {
x22591 = false;
}
bool x22595;
if (x22591) {
x22595 = x22594;
} else {
x22595 = false;
}
bool x22596;
if (x22595) {
x22596 = x22594;
} else {
x22596 = false;
}
if (x22596) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22485,x22487,x22487,1,x22585,1,1);
assert(false && "");
}
bool x22602 = x22485 <= x22585;
int32_t x22603;
if (x22602) {
x22603 = x22585;
} else {
x22603 = x22485;
}
bool x22609 = x22603 > 0;
bool x22611;
if (x22609) {
x22611 = x22610;
} else {
x22611 = false;
}
bool x22612;
if (x22611) {
x22612 = x22610;
} else {
x22612 = false;
}
if (x22612) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(22485) x Sym(22487) x Sym(22487)"," x Const(1) x Sym(22585) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x22607 = x22603 * x22606;
int32_t x22608 = 64 * x22607;
float* x22618 = (float*)myMalloc(x22608 * sizeof(float));;
int32_t x22619 = 0;
int32_t x22620 = 0;
int32_t x22621 = 0;
bool x22667 = x22485 > 1;
bool x22671 = x22585 > 1;
for(int x22622=0; x22622 < 64; x22622++) {
int32_t x22623 = x22620;
int32_t x22624 = x22621;
int32_t x22625 = x22619;
int32_t x22626 = x22625;
int32_t x22627 = x22623;
int32_t x22628 = x22624;
for(int x22630=0; x22630 < x22603; x22630++) {
int32_t x22631 = x22627;
int32_t x22632 = x22628;
int32_t x22633 = x22626;
int32_t x22634 = x22633;
int32_t x22635 = x22631;
int32_t x22636 = x22632;
for(int x22638=0; x22638 < x22605; x22638++) {
int32_t x22639 = x22635;
int32_t x22640 = x22636;
int32_t x22641 = x22634;
int32_t x22642 = x22641;
int32_t x22643 = x22639;
int32_t x22644 = x22640;
for(int x22645=0; x22645 < x22605; x22645++) {
int32_t x22646 = x22642;
int32_t x22647 = x22643;
float x22648 = x22500[x22647];
int32_t x22649 = x22644;
float x22650 = x77[x22649];
float x22651 = x22648 * x22650;
x22618[x22646] = x22651;
x22642 += 1;
if (x22654) {
x22643 += 1;
} else {
}

}
x22634 += x22605;
if (x22654) {
x22635 += x22487;
} else {
}

}
x22626 += x22606;
if (x22667) {
x22627 += x22488;
} else {
}
if (x22671) {
x22628 += 1;
} else {
}

}
x22619 += x22607;
x22620 += x22489;

}
int32_t x22681 = 0;
int32_t x22682 = 1;
x22682 *= 1;
x22681 += 1;
x22682 *= 1;
x22682 *= 1;
int32_t x22687 = x22681;
bool x22688 = x22687 >= 2;
if (x22688) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22693 = x22687 == 0;
if (x22693) {
int32_t x22694 = x22682;
bool x22695 = x22694 == 256;
if (x22695) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22702 = x22682;
int32_t x22703 = 256 / x22702;
bool x22709;
if (x452) {
bool x22704 = x22603 == 1;
bool x22705 = x22703 == 1;
bool x22706 = x22704 || x22705;
bool x22707 = x22603 == x22703;
bool x22708 = x22706 || x22707;
x22709 = x22708;
} else {
x22709 = false;
}
bool x22713;
if (x22709) {
x22713 = x22712;
} else {
x22713 = false;
}
bool x22714;
if (x22713) {
x22714 = x22712;
} else {
x22714 = false;
}
if (x22714) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22603,x22605,x22605,1,x22703,1,1);
assert(false && "");
}
bool x22720 = x22603 <= x22703;
int32_t x22721;
if (x22720) {
x22721 = x22703;
} else {
x22721 = x22603;
}
bool x22727 = x22721 > 0;
bool x22729;
if (x22727) {
x22729 = x22728;
} else {
x22729 = false;
}
bool x22730;
if (x22729) {
x22730 = x22728;
} else {
x22730 = false;
}
if (x22730) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(22603) x Sym(22605) x Sym(22605)"," x Const(1) x Sym(22703) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x22725 = x22721 * x22724;
int32_t x22726 = 64 * x22725;
float* x22736 = (float*)myMalloc(x22726 * sizeof(float));;
int32_t x22737 = 0;
int32_t x22738 = 0;
int32_t x22739 = 0;
bool x22785 = x22603 > 1;
bool x22789 = x22703 > 1;
for(int x22740=0; x22740 < 64; x22740++) {
int32_t x22741 = x22738;
int32_t x22742 = x22739;
int32_t x22743 = x22737;
int32_t x22744 = x22743;
int32_t x22745 = x22741;
int32_t x22746 = x22742;
for(int x22748=0; x22748 < x22721; x22748++) {
int32_t x22749 = x22745;
int32_t x22750 = x22746;
int32_t x22751 = x22744;
int32_t x22752 = x22751;
int32_t x22753 = x22749;
int32_t x22754 = x22750;
for(int x22756=0; x22756 < x22723; x22756++) {
int32_t x22757 = x22753;
int32_t x22758 = x22754;
int32_t x22759 = x22752;
int32_t x22760 = x22759;
int32_t x22761 = x22757;
int32_t x22762 = x22758;
for(int x22763=0; x22763 < x22723; x22763++) {
int32_t x22764 = x22760;
int32_t x22765 = x22761;
float x22766 = x22618[x22765];
int32_t x22767 = x22762;
float x22768 = x185[x22767];
float x22769 = x22766 + x22768;
x22736[x22764] = x22769;
x22760 += 1;
if (x22772) {
x22761 += 1;
} else {
}

}
x22752 += x22723;
if (x22772) {
x22753 += x22605;
} else {
}

}
x22744 += x22724;
if (x22785) {
x22745 += x22606;
} else {
}
if (x22789) {
x22746 += 1;
} else {
}

}
x22737 += x22725;
x22738 += x22607;

}
float* x22799 = (float*)myMalloc(x22726 * sizeof(float));;
for(int x22801=0; x22801 < x22726; x22801++) {
float x22802 = x22736[x22801];
bool x22803 = x22802 < 0.0f;
if (x22803) {
x22799[x22801] = 0.0f;
} else {
float x22806 = x22736[x22801];
x22799[x22801] = x22806;
}

}
float* x22821 = (float*)myMalloc(x22820 * sizeof(float));;
int32_t x22822 = 9 * x22721;
int32_t x22825 = 64 * x22822;
int32_t x22826 = x22825 * x22816;
float* x22827 = (float*)myMalloc(x22826 * sizeof(float));;
int32_t x22823 = x22822 * x22816;
int32_t x22835 = x22721 * 3;
int32_t x22836 = x22835 * 3;
for(int x22828=0; x22828 < 64; x22828++) {
int32_t x22829 = x22828 * x22725;
float* x22830 = x22799+x22829;
int32_t x22831 = x22828 * x22817;
float* x22832 = x22821+x22831;
int32_t x22833 = x22828 * x22823;
float* x22834 = x22827+x22833;
for(int x22838=0; x22838 < x22836; x22838++) {
int32_t x22839 = x22838 / 9;
int32_t x22843 = x22839 * 3;
int32_t x22844 = x22843 * 3;
int32_t x22845 = x22844 * x22815;
int32_t x22846 = x22845 * x22815;
int32_t x22840 = x22838 % 9;
int32_t x22841 = x22840 / 3;
int32_t x22847 = x22841 * 3;
int32_t x22848 = x22847 * x22815;
int32_t x22849 = x22848 * x22815;
int32_t x22850 = x22846 + x22849;
int32_t x22842 = x22840 % 3;
int32_t x22851 = x22842 * x22815;
int32_t x22852 = x22851 * x22815;
int32_t x22853 = x22850 + x22852;
float* x22854 = x22834+x22853;
int32_t x22855 = x22839 * x22723;
int32_t x22856 = x22855 * x22723;
float* x22857 = x22830+x22856;
int32_t x22870 = 1 - x22842;
bool x22871 = x22870 > 0;
int32_t x22872;
if (x22871) {
x22872 = x22870;
} else {
x22872 = 0;
}
int32_t x22873 = 3 - x22842;
int32_t x22874 = x22873 - 1;
int32_t x22875 = 1 - x22874;
bool x22876 = x22875 > 0;
int32_t x22877;
if (x22876) {
x22877 = x22875;
} else {
x22877 = 0;
}
int32_t x22878 = x22815 - x22877;
int32_t x22879 = x22878 - x22872;
bool x22880 = x22879 <= 0;
bool x22884 = x22872 > 0;
int32_t x22869 = -1 + x22842;
bool x22897 = x22877 > 0;
for(int x22859=0; x22859 < x22815; x22859++) {
int32_t x22860 = x22859 - 1;
int32_t x22861 = x22860 + x22841;
bool x22862 = x22861 < 0;
bool x22863 = x22861 >= x22723;
bool x22864 = x22862 || x22863;
if (x22864) {
int32_t x22865 = x22859 * x22815;
float* x22866 = x22854+x22865;
memset(x22866, 0, 4 * x22815);;
} else {
if (x22880) {
int32_t x22865 = x22859 * x22815;
float* x22881 = x22854+x22865;
memset(x22881, 0, 4 * x22815);;
} else {
int32_t x22865 = x22859 * x22815;
if (x22884) {
float* x22885 = x22854+x22865;
memset(x22885, 0, 4 * x22872);;
} else {
}
// may have segfault here
int32_t x22890 = x22865 + x22872;
float* x22891 = x22854+x22890;
int32_t x22892 = x22861 * x22723;
int32_t x22893 = x22892 + x22869;
int32_t x22894 = x22893 + x22872;
float* x22895 = x22857+x22894;
memcpy(x22891, x22895, 4 * x22879);;
if (x22897) {
int32_t x22898 = x22865 + x22815;
int32_t x22899 = x22898 - x22877;
float* x22900 = x22854+x22899;
memset(x22900, 0, 4 * x22877);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x22816,x22822,1,x262,x22822,x22834,x22816,1,x22832,x22816);

}
int32_t x22915 = 0;
int32_t x22916 = 1;
x22916 *= 1;
x22915 += 1;
x22916 *= 1;
x22916 *= 1;
int32_t x22921 = x22915;
bool x22922 = x22921 >= 2;
if (x22922) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x22927 = x22921 == 0;
if (x22927) {
int32_t x22928 = x22916;
bool x22929 = x22928 == 256;
if (x22929) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x22936 = x22916;
int32_t x22937 = 256 / x22936;
bool x22941;
if (x452) {
bool x22938 = x22937 == 1;
bool x22939 = 256 == x22937;
bool x22940 = x22938 || x22939;
x22941 = x22940;
} else {
x22941 = false;
}
bool x22945;
if (x22941) {
x22945 = x22944;
} else {
x22945 = false;
}
bool x22946;
if (x22945) {
x22946 = x22944;
} else {
x22946 = false;
}
if (x22946) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x22815,x22815,1,x22937,1,1);
assert(false && "");
}
bool x22952 = 256 <= x22937;
int32_t x22953;
if (x22952) {
x22953 = x22937;
} else {
x22953 = 256;
}
bool x22959 = x22953 > 0;
bool x22961;
if (x22959) {
x22961 = x22960;
} else {
x22961 = false;
}
bool x22962;
if (x22961) {
x22962 = x22960;
} else {
x22962 = false;
}
if (x22962) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(22815) x Sym(22815)"," x Const(1) x Sym(22937) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x22957 = x22953 * x22956;
int32_t x22958 = 64 * x22957;
float* x22968 = (float*)myMalloc(x22958 * sizeof(float));;
int32_t x22969 = 0;
int32_t x22970 = 0;
int32_t x22971 = 0;
bool x23018 = x22937 > 1;
for(int x22972=0; x22972 < 64; x22972++) {
int32_t x22973 = x22970;
int32_t x22974 = x22971;
int32_t x22975 = x22969;
int32_t x22976 = x22975;
int32_t x22977 = x22973;
int32_t x22978 = x22974;
for(int x22980=0; x22980 < x22953; x22980++) {
int32_t x22981 = x22977;
int32_t x22982 = x22978;
int32_t x22983 = x22976;
int32_t x22984 = x22983;
int32_t x22985 = x22981;
int32_t x22986 = x22982;
for(int x22988=0; x22988 < x22955; x22988++) {
int32_t x22989 = x22985;
int32_t x22990 = x22986;
int32_t x22991 = x22984;
int32_t x22992 = x22991;
int32_t x22993 = x22989;
int32_t x22994 = x22990;
for(int x22995=0; x22995 < x22955; x22995++) {
int32_t x22996 = x22992;
int32_t x22997 = x22993;
float x22998 = x22821[x22997];
int32_t x22999 = x22994;
float x23000 = x250[x22999];
float x23001 = x22998 - x23000;
x22968[x22996] = x23001;
x22992 += 1;
if (x23004) {
x22993 += 1;
} else {
}

}
x22984 += x22955;
if (x23004) {
x22985 += x22815;
} else {
}

}
x22976 += x22956;
x22977 += x22816;
if (x23018) {
x22978 += 1;
} else {
}

}
x22969 += x22957;
x22970 += x22817;

}
float* x23028 = (float*)myMalloc(256 * sizeof(float));;
for(int x23029=0; x23029 < 256; x23029++) {
float x23030 = x104[x23029];
float x23031 = x23030 + 1.0E-5f;
x23028[x23029] = x23031;

}
float* x23035 = (float*)myMalloc(256 * sizeof(float));;
for(int x23036=0; x23036 < 256; x23036++) {
float x23037 = x23028[x23036];
double x23038 = (double)x23037;
double x23039 = sqrt(x23038);
float x23040 = (float)x23039;
x23035[x23036] = x23040;

}
int32_t x23044 = 0;
int32_t x23045 = 1;
x23045 *= 1;
x23044 += 1;
x23045 *= 1;
x23045 *= 1;
int32_t x23050 = x23044;
bool x23051 = x23050 >= 2;
if (x23051) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23056 = x23050 == 0;
if (x23056) {
int32_t x23057 = x23045;
bool x23058 = x23057 == 256;
if (x23058) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23065 = x23045;
int32_t x23066 = 256 / x23065;
bool x23072;
if (x452) {
bool x23067 = x22953 == 1;
bool x23068 = x23066 == 1;
bool x23069 = x23067 || x23068;
bool x23070 = x22953 == x23066;
bool x23071 = x23069 || x23070;
x23072 = x23071;
} else {
x23072 = false;
}
bool x23076;
if (x23072) {
x23076 = x23075;
} else {
x23076 = false;
}
bool x23077;
if (x23076) {
x23077 = x23075;
} else {
x23077 = false;
}
if (x23077) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x22953,x22955,x22955,1,x23066,1,1);
assert(false && "");
}
bool x23083 = x22953 <= x23066;
int32_t x23084;
if (x23083) {
x23084 = x23066;
} else {
x23084 = x22953;
}
bool x23090 = x23084 > 0;
bool x23092;
if (x23090) {
x23092 = x23091;
} else {
x23092 = false;
}
bool x23093;
if (x23092) {
x23093 = x23091;
} else {
x23093 = false;
}
if (x23093) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(22953) x Sym(22955) x Sym(22955)"," x Const(1) x Sym(23066) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x23088 = x23084 * x23087;
int32_t x23089 = 64 * x23088;
float* x23099 = (float*)myMalloc(x23089 * sizeof(float));;
int32_t x23100 = 0;
int32_t x23101 = 0;
int32_t x23102 = 0;
bool x23148 = x22953 > 1;
bool x23152 = x23066 > 1;
for(int x23103=0; x23103 < 64; x23103++) {
int32_t x23104 = x23101;
int32_t x23105 = x23102;
int32_t x23106 = x23100;
int32_t x23107 = x23106;
int32_t x23108 = x23104;
int32_t x23109 = x23105;
for(int x23111=0; x23111 < x23084; x23111++) {
int32_t x23112 = x23108;
int32_t x23113 = x23109;
int32_t x23114 = x23107;
int32_t x23115 = x23114;
int32_t x23116 = x23112;
int32_t x23117 = x23113;
for(int x23119=0; x23119 < x23086; x23119++) {
int32_t x23120 = x23116;
int32_t x23121 = x23117;
int32_t x23122 = x23115;
int32_t x23123 = x23122;
int32_t x23124 = x23120;
int32_t x23125 = x23121;
for(int x23126=0; x23126 < x23086; x23126++) {
int32_t x23127 = x23123;
int32_t x23128 = x23124;
float x23129 = x22968[x23128];
int32_t x23130 = x23125;
float x23131 = x23035[x23130];
float x23132 = x23129 / x23131;
x23099[x23127] = x23132;
x23123 += 1;
if (x23135) {
x23124 += 1;
} else {
}

}
x23115 += x23086;
if (x23135) {
x23116 += x22955;
} else {
}

}
x23107 += x23087;
if (x23148) {
x23108 += x22956;
} else {
}
if (x23152) {
x23109 += 1;
} else {
}

}
x23100 += x23088;
x23101 += x22957;

}
int32_t x23162 = 0;
int32_t x23163 = 1;
x23163 *= 1;
x23162 += 1;
x23163 *= 1;
x23163 *= 1;
int32_t x23168 = x23162;
bool x23169 = x23168 >= 2;
if (x23169) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23174 = x23168 == 0;
if (x23174) {
int32_t x23175 = x23163;
bool x23176 = x23175 == 256;
if (x23176) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23183 = x23163;
int32_t x23184 = 256 / x23183;
bool x23190;
if (x452) {
bool x23185 = x23084 == 1;
bool x23186 = x23184 == 1;
bool x23187 = x23185 || x23186;
bool x23188 = x23084 == x23184;
bool x23189 = x23187 || x23188;
x23190 = x23189;
} else {
x23190 = false;
}
bool x23194;
if (x23190) {
x23194 = x23193;
} else {
x23194 = false;
}
bool x23195;
if (x23194) {
x23195 = x23193;
} else {
x23195 = false;
}
if (x23195) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23084,x23086,x23086,1,x23184,1,1);
assert(false && "");
}
bool x23201 = x23084 <= x23184;
int32_t x23202;
if (x23201) {
x23202 = x23184;
} else {
x23202 = x23084;
}
bool x23208 = x23202 > 0;
bool x23210;
if (x23208) {
x23210 = x23209;
} else {
x23210 = false;
}
bool x23211;
if (x23210) {
x23211 = x23209;
} else {
x23211 = false;
}
if (x23211) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(23084) x Sym(23086) x Sym(23086)"," x Const(1) x Sym(23184) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x23206 = x23202 * x23205;
int32_t x23207 = 64 * x23206;
float* x23217 = (float*)myMalloc(x23207 * sizeof(float));;
int32_t x23218 = 0;
int32_t x23219 = 0;
int32_t x23220 = 0;
bool x23266 = x23084 > 1;
bool x23270 = x23184 > 1;
for(int x23221=0; x23221 < 64; x23221++) {
int32_t x23222 = x23219;
int32_t x23223 = x23220;
int32_t x23224 = x23218;
int32_t x23225 = x23224;
int32_t x23226 = x23222;
int32_t x23227 = x23223;
for(int x23229=0; x23229 < x23202; x23229++) {
int32_t x23230 = x23226;
int32_t x23231 = x23227;
int32_t x23232 = x23225;
int32_t x23233 = x23232;
int32_t x23234 = x23230;
int32_t x23235 = x23231;
for(int x23237=0; x23237 < x23204; x23237++) {
int32_t x23238 = x23234;
int32_t x23239 = x23235;
int32_t x23240 = x23233;
int32_t x23241 = x23240;
int32_t x23242 = x23238;
int32_t x23243 = x23239;
for(int x23244=0; x23244 < x23204; x23244++) {
int32_t x23245 = x23241;
int32_t x23246 = x23242;
float x23247 = x23099[x23246];
int32_t x23248 = x23243;
float x23249 = x168[x23248];
float x23250 = x23247 * x23249;
x23217[x23245] = x23250;
x23241 += 1;
if (x23253) {
x23242 += 1;
} else {
}

}
x23233 += x23204;
if (x23253) {
x23234 += x23086;
} else {
}

}
x23225 += x23205;
if (x23266) {
x23226 += x23087;
} else {
}
if (x23270) {
x23227 += 1;
} else {
}

}
x23218 += x23206;
x23219 += x23088;

}
int32_t x23280 = 0;
int32_t x23281 = 1;
x23281 *= 1;
x23280 += 1;
x23281 *= 1;
x23281 *= 1;
int32_t x23286 = x23280;
bool x23287 = x23286 >= 2;
if (x23287) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23292 = x23286 == 0;
if (x23292) {
int32_t x23293 = x23281;
bool x23294 = x23293 == 256;
if (x23294) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23301 = x23281;
int32_t x23302 = 256 / x23301;
bool x23308;
if (x452) {
bool x23303 = x23202 == 1;
bool x23304 = x23302 == 1;
bool x23305 = x23303 || x23304;
bool x23306 = x23202 == x23302;
bool x23307 = x23305 || x23306;
x23308 = x23307;
} else {
x23308 = false;
}
bool x23312;
if (x23308) {
x23312 = x23311;
} else {
x23312 = false;
}
bool x23313;
if (x23312) {
x23313 = x23311;
} else {
x23313 = false;
}
if (x23313) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23202,x23204,x23204,1,x23302,1,1);
assert(false && "");
}
bool x23319 = x23202 <= x23302;
int32_t x23320;
if (x23319) {
x23320 = x23302;
} else {
x23320 = x23202;
}
bool x23326 = x23320 > 0;
bool x23328;
if (x23326) {
x23328 = x23327;
} else {
x23328 = false;
}
bool x23329;
if (x23328) {
x23329 = x23327;
} else {
x23329 = false;
}
if (x23329) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(23202) x Sym(23204) x Sym(23204)"," x Const(1) x Sym(23302) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x23324 = x23320 * x23323;
int32_t x23325 = 64 * x23324;
float* x23335 = (float*)myMalloc(x23325 * sizeof(float));;
int32_t x23336 = 0;
int32_t x23337 = 0;
int32_t x23338 = 0;
bool x23384 = x23202 > 1;
bool x23388 = x23302 > 1;
for(int x23339=0; x23339 < 64; x23339++) {
int32_t x23340 = x23337;
int32_t x23341 = x23338;
int32_t x23342 = x23336;
int32_t x23343 = x23342;
int32_t x23344 = x23340;
int32_t x23345 = x23341;
for(int x23347=0; x23347 < x23320; x23347++) {
int32_t x23348 = x23344;
int32_t x23349 = x23345;
int32_t x23350 = x23343;
int32_t x23351 = x23350;
int32_t x23352 = x23348;
int32_t x23353 = x23349;
for(int x23355=0; x23355 < x23322; x23355++) {
int32_t x23356 = x23352;
int32_t x23357 = x23353;
int32_t x23358 = x23351;
int32_t x23359 = x23358;
int32_t x23360 = x23356;
int32_t x23361 = x23357;
for(int x23362=0; x23362 < x23322; x23362++) {
int32_t x23363 = x23359;
int32_t x23364 = x23360;
float x23365 = x23217[x23364];
int32_t x23366 = x23361;
float x23367 = x109[x23366];
float x23368 = x23365 + x23367;
x23335[x23363] = x23368;
x23359 += 1;
if (x23371) {
x23360 += 1;
} else {
}

}
x23351 += x23322;
if (x23371) {
x23352 += x23204;
} else {
}

}
x23343 += x23323;
if (x23384) {
x23344 += x23205;
} else {
}
if (x23388) {
x23345 += 1;
} else {
}

}
x23336 += x23324;
x23337 += x23206;

}
float* x23398 = (float*)myMalloc(x23325 * sizeof(float));;
for(int x23400=0; x23400 < x23325; x23400++) {
float x23401 = x23335[x23400];
bool x23402 = x23401 < 0.0f;
if (x23402) {
x23398[x23400] = 0.0f;
} else {
float x23405 = x23335[x23400];
x23398[x23400] = x23405;
}

}
float* x23419 = (float*)myMalloc(x23418 * sizeof(float));;
int32_t x23422 = 64 * x23320;
int32_t x23423 = x23422 * x23414;
float* x23424 = (float*)myMalloc(x23423 * sizeof(float));;
int32_t x23420 = x23320 * x23414;
for(int x23425=0; x23425 < 64; x23425++) {
int32_t x23426 = x23425 * x23324;
float* x23427 = x23398+x23426;
int32_t x23428 = x23425 * x23415;
float* x23429 = x23419+x23428;
int32_t x23430 = x23425 * x23420;
float* x23431 = x23424+x23430;
for(int x23432=0; x23432 < x23320; x23432++) {
int32_t x23433 = x23432 / 1;
int32_t x23437 = x23433 * x23413;
int32_t x23438 = x23437 * x23413;
int32_t x23434 = x23432 % 1;
int32_t x23435 = x23434 / 1;
int32_t x23439 = x23435 * x23413;
int32_t x23440 = x23439 * x23413;
int32_t x23441 = x23438 + x23440;
int32_t x23436 = x23434 % 1;
int32_t x23442 = x23436 * x23413;
int32_t x23443 = x23442 * x23413;
int32_t x23444 = x23441 + x23443;
float* x23445 = x23431+x23444;
int32_t x23446 = x23433 * x23322;
int32_t x23447 = x23446 * x23322;
float* x23448 = x23427+x23447;
for(int x23450=0; x23450 < x23413; x23450++) {
int32_t x23452 = x23450 * x23413;
float* x23453 = x23445+x23452;
int32_t x23451 = x23450 + x23435;
int32_t x23454 = x23451 * x23322;
int32_t x23455 = x23454 + x23436;
float* x23456 = x23448+x23455;
memcpy(x23453, x23456, 4 * x23413);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x23414,x23320,1,x221,x23320,x23431,x23414,1,x23429,x23414);

}
int32_t x23465 = 0;
int32_t x23466 = 1;
x23466 *= 1;
x23465 += 1;
x23466 *= 1;
x23466 *= 1;
int32_t x23471 = x23465;
bool x23472 = x23471 >= 2;
if (x23472) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23477 = x23471 == 0;
if (x23477) {
int32_t x23478 = x23466;
bool x23479 = x23478 == 1024;
if (x23479) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23486 = x23466;
int32_t x23487 = 1024 / x23486;
bool x23491;
if (x452) {
bool x23488 = x23487 == 1;
bool x23489 = 1024 == x23487;
bool x23490 = x23488 || x23489;
x23491 = x23490;
} else {
x23491 = false;
}
bool x23495;
if (x23491) {
x23495 = x23494;
} else {
x23495 = false;
}
bool x23496;
if (x23495) {
x23496 = x23494;
} else {
x23496 = false;
}
if (x23496) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x23413,x23413,1,x23487,1,1);
assert(false && "");
}
bool x23502 = 1024 <= x23487;
int32_t x23503;
if (x23502) {
x23503 = x23487;
} else {
x23503 = 1024;
}
bool x23509 = x23503 > 0;
bool x23511;
if (x23509) {
x23511 = x23510;
} else {
x23511 = false;
}
bool x23512;
if (x23511) {
x23512 = x23510;
} else {
x23512 = false;
}
if (x23512) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(1024) x Sym(23413) x Sym(23413)"," x Const(1) x Sym(23487) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x23507 = x23503 * x23506;
int32_t x23508 = 64 * x23507;
float* x23518 = (float*)myMalloc(x23508 * sizeof(float));;
int32_t x23519 = 0;
int32_t x23520 = 0;
int32_t x23521 = 0;
bool x23568 = x23487 > 1;
for(int x23522=0; x23522 < 64; x23522++) {
int32_t x23523 = x23520;
int32_t x23524 = x23521;
int32_t x23525 = x23519;
int32_t x23526 = x23525;
int32_t x23527 = x23523;
int32_t x23528 = x23524;
for(int x23530=0; x23530 < x23503; x23530++) {
int32_t x23531 = x23527;
int32_t x23532 = x23528;
int32_t x23533 = x23526;
int32_t x23534 = x23533;
int32_t x23535 = x23531;
int32_t x23536 = x23532;
for(int x23538=0; x23538 < x23505; x23538++) {
int32_t x23539 = x23535;
int32_t x23540 = x23536;
int32_t x23541 = x23534;
int32_t x23542 = x23541;
int32_t x23543 = x23539;
int32_t x23544 = x23540;
for(int x23545=0; x23545 < x23505; x23545++) {
int32_t x23546 = x23542;
int32_t x23547 = x23543;
float x23548 = x23419[x23547];
int32_t x23549 = x23544;
float x23550 = x209[x23549];
float x23551 = x23548 - x23550;
x23518[x23546] = x23551;
x23542 += 1;
if (x23554) {
x23543 += 1;
} else {
}

}
x23534 += x23505;
if (x23554) {
x23535 += x23413;
} else {
}

}
x23526 += x23506;
x23527 += x23414;
if (x23568) {
x23528 += 1;
} else {
}

}
x23519 += x23507;
x23520 += x23415;

}
float* x23578 = (float*)myMalloc(1024 * sizeof(float));;
for(int x23579=0; x23579 < 1024; x23579++) {
float x23580 = x272[x23579];
float x23581 = x23580 + 1.0E-5f;
x23578[x23579] = x23581;

}
float* x23585 = (float*)myMalloc(1024 * sizeof(float));;
for(int x23586=0; x23586 < 1024; x23586++) {
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
bool x23608 = x23607 == 1024;
if (x23608) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23615 = x23595;
int32_t x23616 = 1024 / x23615;
bool x23622;
if (x452) {
bool x23617 = x23503 == 1;
bool x23618 = x23616 == 1;
bool x23619 = x23617 || x23618;
bool x23620 = x23503 == x23616;
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
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23503,x23505,x23505,1,x23616,1,1);
assert(false && "");
}
bool x23633 = x23503 <= x23616;
int32_t x23634;
if (x23633) {
x23634 = x23616;
} else {
x23634 = x23503;
}
bool x23640 = x23634 > 0;
bool x23642;
if (x23640) {
x23642 = x23641;
} else {
x23642 = false;
}
bool x23643;
if (x23642) {
x23643 = x23641;
} else {
x23643 = false;
}
if (x23643) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(23503) x Sym(23505) x Sym(23505)"," x Const(1) x Sym(23616) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x23638 = x23634 * x23637;
int32_t x23639 = 64 * x23638;
float* x23649 = (float*)myMalloc(x23639 * sizeof(float));;
int32_t x23650 = 0;
int32_t x23651 = 0;
int32_t x23652 = 0;
bool x23698 = x23503 > 1;
bool x23702 = x23616 > 1;
for(int x23653=0; x23653 < 64; x23653++) {
int32_t x23654 = x23651;
int32_t x23655 = x23652;
int32_t x23656 = x23650;
int32_t x23657 = x23656;
int32_t x23658 = x23654;
int32_t x23659 = x23655;
for(int x23661=0; x23661 < x23634; x23661++) {
int32_t x23662 = x23658;
int32_t x23663 = x23659;
int32_t x23664 = x23657;
int32_t x23665 = x23664;
int32_t x23666 = x23662;
int32_t x23667 = x23663;
for(int x23669=0; x23669 < x23636; x23669++) {
int32_t x23670 = x23666;
int32_t x23671 = x23667;
int32_t x23672 = x23665;
int32_t x23673 = x23672;
int32_t x23674 = x23670;
int32_t x23675 = x23671;
for(int x23676=0; x23676 < x23636; x23676++) {
int32_t x23677 = x23673;
int32_t x23678 = x23674;
float x23679 = x23518[x23678];
int32_t x23680 = x23675;
float x23681 = x23585[x23680];
float x23682 = x23679 / x23681;
x23649[x23677] = x23682;
x23673 += 1;
if (x23685) {
x23674 += 1;
} else {
}

}
x23665 += x23636;
if (x23685) {
x23666 += x23505;
} else {
}

}
x23657 += x23637;
if (x23698) {
x23658 += x23506;
} else {
}
if (x23702) {
x23659 += 1;
} else {
}

}
x23650 += x23638;
x23651 += x23507;

}
int32_t x23712 = 0;
int32_t x23713 = 1;
x23713 *= 1;
x23712 += 1;
x23713 *= 1;
x23713 *= 1;
int32_t x23718 = x23712;
bool x23719 = x23718 >= 2;
if (x23719) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23724 = x23718 == 0;
if (x23724) {
int32_t x23725 = x23713;
bool x23726 = x23725 == 1024;
if (x23726) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23733 = x23713;
int32_t x23734 = 1024 / x23733;
bool x23740;
if (x452) {
bool x23735 = x23634 == 1;
bool x23736 = x23734 == 1;
bool x23737 = x23735 || x23736;
bool x23738 = x23634 == x23734;
bool x23739 = x23737 || x23738;
x23740 = x23739;
} else {
x23740 = false;
}
bool x23744;
if (x23740) {
x23744 = x23743;
} else {
x23744 = false;
}
bool x23745;
if (x23744) {
x23745 = x23743;
} else {
x23745 = false;
}
if (x23745) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23634,x23636,x23636,1,x23734,1,1);
assert(false && "");
}
bool x23751 = x23634 <= x23734;
int32_t x23752;
if (x23751) {
x23752 = x23734;
} else {
x23752 = x23634;
}
bool x23758 = x23752 > 0;
bool x23760;
if (x23758) {
x23760 = x23759;
} else {
x23760 = false;
}
bool x23761;
if (x23760) {
x23761 = x23759;
} else {
x23761 = false;
}
if (x23761) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(23634) x Sym(23636) x Sym(23636)"," x Const(1) x Sym(23734) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x23756 = x23752 * x23755;
int32_t x23757 = 64 * x23756;
float* x23767 = (float*)myMalloc(x23757 * sizeof(float));;
int32_t x23768 = 0;
int32_t x23769 = 0;
int32_t x23770 = 0;
bool x23816 = x23634 > 1;
bool x23820 = x23734 > 1;
for(int x23771=0; x23771 < 64; x23771++) {
int32_t x23772 = x23769;
int32_t x23773 = x23770;
int32_t x23774 = x23768;
int32_t x23775 = x23774;
int32_t x23776 = x23772;
int32_t x23777 = x23773;
for(int x23779=0; x23779 < x23752; x23779++) {
int32_t x23780 = x23776;
int32_t x23781 = x23777;
int32_t x23782 = x23775;
int32_t x23783 = x23782;
int32_t x23784 = x23780;
int32_t x23785 = x23781;
for(int x23787=0; x23787 < x23754; x23787++) {
int32_t x23788 = x23784;
int32_t x23789 = x23785;
int32_t x23790 = x23783;
int32_t x23791 = x23790;
int32_t x23792 = x23788;
int32_t x23793 = x23789;
for(int x23794=0; x23794 < x23754; x23794++) {
int32_t x23795 = x23791;
int32_t x23796 = x23792;
float x23797 = x23649[x23796];
int32_t x23798 = x23793;
float x23799 = x59[x23798];
float x23800 = x23797 * x23799;
x23767[x23795] = x23800;
x23791 += 1;
if (x23803) {
x23792 += 1;
} else {
}

}
x23783 += x23754;
if (x23803) {
x23784 += x23636;
} else {
}

}
x23775 += x23755;
if (x23816) {
x23776 += x23637;
} else {
}
if (x23820) {
x23777 += 1;
} else {
}

}
x23768 += x23756;
x23769 += x23638;

}
int32_t x23830 = 0;
int32_t x23831 = 1;
x23831 *= 1;
x23830 += 1;
x23831 *= 1;
x23831 *= 1;
int32_t x23836 = x23830;
bool x23837 = x23836 >= 2;
if (x23837) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x23842 = x23836 == 0;
if (x23842) {
int32_t x23843 = x23831;
bool x23844 = x23843 == 1024;
if (x23844) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x23851 = x23831;
int32_t x23852 = 1024 / x23851;
bool x23858;
if (x452) {
bool x23853 = x23752 == 1;
bool x23854 = x23852 == 1;
bool x23855 = x23853 || x23854;
bool x23856 = x23752 == x23852;
bool x23857 = x23855 || x23856;
x23858 = x23857;
} else {
x23858 = false;
}
bool x23862;
if (x23858) {
x23862 = x23861;
} else {
x23862 = false;
}
bool x23863;
if (x23862) {
x23863 = x23861;
} else {
x23863 = false;
}
if (x23863) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23752,x23754,x23754,1,x23852,1,1);
assert(false && "");
}
bool x23869 = x23752 <= x23852;
int32_t x23870;
if (x23869) {
x23870 = x23852;
} else {
x23870 = x23752;
}
bool x23876 = x23870 > 0;
bool x23878;
if (x23876) {
x23878 = x23877;
} else {
x23878 = false;
}
bool x23879;
if (x23878) {
x23879 = x23877;
} else {
x23879 = false;
}
if (x23879) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(23752) x Sym(23754) x Sym(23754)"," x Const(1) x Sym(23852) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x23874 = x23870 * x23873;
int32_t x23875 = 64 * x23874;
float* x23885 = (float*)myMalloc(x23875 * sizeof(float));;
int32_t x23886 = 0;
int32_t x23887 = 0;
int32_t x23888 = 0;
bool x23934 = x23752 > 1;
bool x23938 = x23852 > 1;
for(int x23889=0; x23889 < 64; x23889++) {
int32_t x23890 = x23887;
int32_t x23891 = x23888;
int32_t x23892 = x23886;
int32_t x23893 = x23892;
int32_t x23894 = x23890;
int32_t x23895 = x23891;
for(int x23897=0; x23897 < x23870; x23897++) {
int32_t x23898 = x23894;
int32_t x23899 = x23895;
int32_t x23900 = x23893;
int32_t x23901 = x23900;
int32_t x23902 = x23898;
int32_t x23903 = x23899;
for(int x23905=0; x23905 < x23872; x23905++) {
int32_t x23906 = x23902;
int32_t x23907 = x23903;
int32_t x23908 = x23901;
int32_t x23909 = x23908;
int32_t x23910 = x23906;
int32_t x23911 = x23907;
for(int x23912=0; x23912 < x23872; x23912++) {
int32_t x23913 = x23909;
int32_t x23914 = x23910;
float x23915 = x23767[x23914];
int32_t x23916 = x23911;
float x23917 = x120[x23916];
float x23918 = x23915 + x23917;
x23885[x23913] = x23918;
x23909 += 1;
if (x23921) {
x23910 += 1;
} else {
}

}
x23901 += x23872;
if (x23921) {
x23902 += x23754;
} else {
}

}
x23893 += x23873;
if (x23934) {
x23894 += x23755;
} else {
}
if (x23938) {
x23895 += 1;
} else {
}

}
x23886 += x23874;
x23887 += x23756;

}
bool x23948 = x23870 == 1;
bool x23949 = x23948 || x22160;
bool x23950 = x23870 == x22082;
bool x23951 = x23949 || x23950;
bool x23956;
if (x23951) {
x23956 = x23955;
} else {
x23956 = false;
}
bool x23957;
if (x23956) {
x23957 = x23955;
} else {
x23957 = false;
}
if (x23957) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x23870,x23872,x23872,64,x22082,x22084,x22084);
assert(false && "");
}
int32_t x23970 = 0;
int32_t x23971 = 0;
int32_t x23972 = 0;
bool x23963 = x23870 <= x22082;
int32_t x23964;
if (x23963) {
x23964 = x22082;
} else {
x23964 = x23870;
}
bool x24023 = x23870 > 1;
int32_t x23968 = x23964 * x23967;
for(int x23973=0; x23973 < 64; x23973++) {
int32_t x23974 = x23971;
int32_t x23975 = x23972;
int32_t x23976 = x23970;
int32_t x23977 = x23976;
int32_t x23978 = x23974;
int32_t x23979 = x23975;
for(int x23981=0; x23981 < x23964; x23981++) {
int32_t x23982 = x23978;
int32_t x23983 = x23979;
int32_t x23984 = x23977;
int32_t x23985 = x23984;
int32_t x23986 = x23982;
int32_t x23987 = x23983;
for(int x23989=0; x23989 < x23966; x23989++) {
int32_t x23990 = x23986;
int32_t x23991 = x23987;
int32_t x23992 = x23985;
int32_t x23993 = x23992;
int32_t x23994 = x23990;
int32_t x23995 = x23991;
for(int x23996=0; x23996 < x23966; x23996++) {
int32_t x23997 = x23994;
float x23998 = x23885[x23997];
int32_t x23999 = x23995;
float x24000 = x22249[x23999];
float x24001 = x23998 + x24000;
x23885[x23997] = x24001;
x23993 += 1;
if (x24004) {
x23994 += 1;
} else {
}
if (x22216) {
x23995 += 1;
} else {
}

}
x23985 += x23966;
if (x24004) {
x23986 += x23872;
} else {
}
if (x22216) {
x23987 += x22084;
} else {
}

}
x23977 += x23967;
if (x24023) {
x23978 += x23873;
} else {
}
if (x22235) {
x23979 += x22085;
} else {
}

}
x23970 += x23968;
x23971 += x23874;
x23972 += x22086;

}
float* x24037 = (float*)myMalloc(x23875 * sizeof(float));;
for(int x24039=0; x24039 < x23875; x24039++) {
float x24040 = x23885[x24039];
bool x24041 = x24040 < 0.0f;
if (x24041) {
x24037[x24039] = 0.0f;
} else {
float x24044 = x23885[x24039];
x24037[x24039] = x24044;
}

}
float* x24058 = (float*)myMalloc(x24057 * sizeof(float));;
int32_t x24061 = 64 * x23870;
int32_t x24062 = x24061 * x24053;
float* x24063 = (float*)myMalloc(x24062 * sizeof(float));;
int32_t x24059 = x23870 * x24053;
for(int x24064=0; x24064 < 64; x24064++) {
int32_t x24065 = x24064 * x23874;
float* x24066 = x24037+x24065;
int32_t x24067 = x24064 * x24054;
float* x24068 = x24058+x24067;
int32_t x24069 = x24064 * x24059;
float* x24070 = x24063+x24069;
for(int x24071=0; x24071 < x23870; x24071++) {
int32_t x24072 = x24071 / 1;
int32_t x24076 = x24072 * x24052;
int32_t x24077 = x24076 * x24052;
int32_t x24073 = x24071 % 1;
int32_t x24074 = x24073 / 1;
int32_t x24078 = x24074 * x24052;
int32_t x24079 = x24078 * x24052;
int32_t x24080 = x24077 + x24079;
int32_t x24075 = x24073 % 1;
int32_t x24081 = x24075 * x24052;
int32_t x24082 = x24081 * x24052;
int32_t x24083 = x24080 + x24082;
float* x24084 = x24070+x24083;
int32_t x24085 = x24072 * x23872;
int32_t x24086 = x24085 * x23872;
float* x24087 = x24066+x24086;
for(int x24089=0; x24089 < x24052; x24089++) {
int32_t x24091 = x24089 * x24052;
float* x24092 = x24084+x24091;
int32_t x24090 = x24089 + x24074;
int32_t x24093 = x24090 * x23872;
int32_t x24094 = x24093 + x24075;
float* x24095 = x24087+x24094;
memcpy(x24092, x24095, 4 * x24052);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x24053,x23870,1,x151,x23870,x24070,x24053,1,x24068,x24053);

}
int32_t x24104 = 0;
int32_t x24105 = 1;
x24105 *= 1;
x24104 += 1;
x24105 *= 1;
x24105 *= 1;
int32_t x24110 = x24104;
bool x24111 = x24110 >= 2;
if (x24111) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24116 = x24110 == 0;
if (x24116) {
int32_t x24117 = x24105;
bool x24118 = x24117 == 256;
if (x24118) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24125 = x24105;
int32_t x24126 = 256 / x24125;
bool x24130;
if (x452) {
bool x24127 = x24126 == 1;
bool x24128 = 256 == x24126;
bool x24129 = x24127 || x24128;
x24130 = x24129;
} else {
x24130 = false;
}
bool x24134;
if (x24130) {
x24134 = x24133;
} else {
x24134 = false;
}
bool x24135;
if (x24134) {
x24135 = x24133;
} else {
x24135 = false;
}
if (x24135) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x24052,x24052,1,x24126,1,1);
assert(false && "");
}
bool x24141 = 256 <= x24126;
int32_t x24142;
if (x24141) {
x24142 = x24126;
} else {
x24142 = 256;
}
bool x24148 = x24142 > 0;
bool x24150;
if (x24148) {
x24150 = x24149;
} else {
x24150 = false;
}
bool x24151;
if (x24150) {
x24151 = x24149;
} else {
x24151 = false;
}
if (x24151) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(24052) x Sym(24052)"," x Const(1) x Sym(24126) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x24146 = x24142 * x24145;
int32_t x24147 = 64 * x24146;
float* x24157 = (float*)myMalloc(x24147 * sizeof(float));;
int32_t x24158 = 0;
int32_t x24159 = 0;
int32_t x24160 = 0;
bool x24207 = x24126 > 1;
for(int x24161=0; x24161 < 64; x24161++) {
int32_t x24162 = x24159;
int32_t x24163 = x24160;
int32_t x24164 = x24158;
int32_t x24165 = x24164;
int32_t x24166 = x24162;
int32_t x24167 = x24163;
for(int x24169=0; x24169 < x24142; x24169++) {
int32_t x24170 = x24166;
int32_t x24171 = x24167;
int32_t x24172 = x24165;
int32_t x24173 = x24172;
int32_t x24174 = x24170;
int32_t x24175 = x24171;
for(int x24177=0; x24177 < x24144; x24177++) {
int32_t x24178 = x24174;
int32_t x24179 = x24175;
int32_t x24180 = x24173;
int32_t x24181 = x24180;
int32_t x24182 = x24178;
int32_t x24183 = x24179;
for(int x24184=0; x24184 < x24144; x24184++) {
int32_t x24185 = x24181;
int32_t x24186 = x24182;
float x24187 = x24058[x24186];
int32_t x24188 = x24183;
float x24189 = x80[x24188];
float x24190 = x24187 - x24189;
x24157[x24185] = x24190;
x24181 += 1;
if (x24193) {
x24182 += 1;
} else {
}

}
x24173 += x24144;
if (x24193) {
x24174 += x24052;
} else {
}

}
x24165 += x24145;
x24166 += x24053;
if (x24207) {
x24167 += 1;
} else {
}

}
x24158 += x24146;
x24159 += x24054;

}
float* x24217 = (float*)myMalloc(256 * sizeof(float));;
for(int x24218=0; x24218 < 256; x24218++) {
float x24219 = x176[x24218];
float x24220 = x24219 + 1.0E-5f;
x24217[x24218] = x24220;

}
float* x24224 = (float*)myMalloc(256 * sizeof(float));;
for(int x24225=0; x24225 < 256; x24225++) {
float x24226 = x24217[x24225];
double x24227 = (double)x24226;
double x24228 = sqrt(x24227);
float x24229 = (float)x24228;
x24224[x24225] = x24229;

}
int32_t x24233 = 0;
int32_t x24234 = 1;
x24234 *= 1;
x24233 += 1;
x24234 *= 1;
x24234 *= 1;
int32_t x24239 = x24233;
bool x24240 = x24239 >= 2;
if (x24240) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24245 = x24239 == 0;
if (x24245) {
int32_t x24246 = x24234;
bool x24247 = x24246 == 256;
if (x24247) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24254 = x24234;
int32_t x24255 = 256 / x24254;
bool x24261;
if (x452) {
bool x24256 = x24142 == 1;
bool x24257 = x24255 == 1;
bool x24258 = x24256 || x24257;
bool x24259 = x24142 == x24255;
bool x24260 = x24258 || x24259;
x24261 = x24260;
} else {
x24261 = false;
}
bool x24265;
if (x24261) {
x24265 = x24264;
} else {
x24265 = false;
}
bool x24266;
if (x24265) {
x24266 = x24264;
} else {
x24266 = false;
}
if (x24266) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24142,x24144,x24144,1,x24255,1,1);
assert(false && "");
}
bool x24272 = x24142 <= x24255;
int32_t x24273;
if (x24272) {
x24273 = x24255;
} else {
x24273 = x24142;
}
bool x24279 = x24273 > 0;
bool x24281;
if (x24279) {
x24281 = x24280;
} else {
x24281 = false;
}
bool x24282;
if (x24281) {
x24282 = x24280;
} else {
x24282 = false;
}
if (x24282) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(24142) x Sym(24144) x Sym(24144)"," x Const(1) x Sym(24255) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x24277 = x24273 * x24276;
int32_t x24278 = 64 * x24277;
float* x24288 = (float*)myMalloc(x24278 * sizeof(float));;
int32_t x24289 = 0;
int32_t x24290 = 0;
int32_t x24291 = 0;
bool x24337 = x24142 > 1;
bool x24341 = x24255 > 1;
for(int x24292=0; x24292 < 64; x24292++) {
int32_t x24293 = x24290;
int32_t x24294 = x24291;
int32_t x24295 = x24289;
int32_t x24296 = x24295;
int32_t x24297 = x24293;
int32_t x24298 = x24294;
for(int x24300=0; x24300 < x24273; x24300++) {
int32_t x24301 = x24297;
int32_t x24302 = x24298;
int32_t x24303 = x24296;
int32_t x24304 = x24303;
int32_t x24305 = x24301;
int32_t x24306 = x24302;
for(int x24308=0; x24308 < x24275; x24308++) {
int32_t x24309 = x24305;
int32_t x24310 = x24306;
int32_t x24311 = x24304;
int32_t x24312 = x24311;
int32_t x24313 = x24309;
int32_t x24314 = x24310;
for(int x24315=0; x24315 < x24275; x24315++) {
int32_t x24316 = x24312;
int32_t x24317 = x24313;
float x24318 = x24157[x24317];
int32_t x24319 = x24314;
float x24320 = x24224[x24319];
float x24321 = x24318 / x24320;
x24288[x24316] = x24321;
x24312 += 1;
if (x24324) {
x24313 += 1;
} else {
}

}
x24304 += x24275;
if (x24324) {
x24305 += x24144;
} else {
}

}
x24296 += x24276;
if (x24337) {
x24297 += x24145;
} else {
}
if (x24341) {
x24298 += 1;
} else {
}

}
x24289 += x24277;
x24290 += x24146;

}
int32_t x24351 = 0;
int32_t x24352 = 1;
x24352 *= 1;
x24351 += 1;
x24352 *= 1;
x24352 *= 1;
int32_t x24357 = x24351;
bool x24358 = x24357 >= 2;
if (x24358) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24363 = x24357 == 0;
if (x24363) {
int32_t x24364 = x24352;
bool x24365 = x24364 == 256;
if (x24365) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24372 = x24352;
int32_t x24373 = 256 / x24372;
bool x24379;
if (x452) {
bool x24374 = x24273 == 1;
bool x24375 = x24373 == 1;
bool x24376 = x24374 || x24375;
bool x24377 = x24273 == x24373;
bool x24378 = x24376 || x24377;
x24379 = x24378;
} else {
x24379 = false;
}
bool x24383;
if (x24379) {
x24383 = x24382;
} else {
x24383 = false;
}
bool x24384;
if (x24383) {
x24384 = x24382;
} else {
x24384 = false;
}
if (x24384) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24273,x24275,x24275,1,x24373,1,1);
assert(false && "");
}
bool x24390 = x24273 <= x24373;
int32_t x24391;
if (x24390) {
x24391 = x24373;
} else {
x24391 = x24273;
}
bool x24397 = x24391 > 0;
bool x24399;
if (x24397) {
x24399 = x24398;
} else {
x24399 = false;
}
bool x24400;
if (x24399) {
x24400 = x24398;
} else {
x24400 = false;
}
if (x24400) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(24273) x Sym(24275) x Sym(24275)"," x Const(1) x Sym(24373) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x24395 = x24391 * x24394;
int32_t x24396 = 64 * x24395;
float* x24406 = (float*)myMalloc(x24396 * sizeof(float));;
int32_t x24407 = 0;
int32_t x24408 = 0;
int32_t x24409 = 0;
bool x24455 = x24273 > 1;
bool x24459 = x24373 > 1;
for(int x24410=0; x24410 < 64; x24410++) {
int32_t x24411 = x24408;
int32_t x24412 = x24409;
int32_t x24413 = x24407;
int32_t x24414 = x24413;
int32_t x24415 = x24411;
int32_t x24416 = x24412;
for(int x24418=0; x24418 < x24391; x24418++) {
int32_t x24419 = x24415;
int32_t x24420 = x24416;
int32_t x24421 = x24414;
int32_t x24422 = x24421;
int32_t x24423 = x24419;
int32_t x24424 = x24420;
for(int x24426=0; x24426 < x24393; x24426++) {
int32_t x24427 = x24423;
int32_t x24428 = x24424;
int32_t x24429 = x24422;
int32_t x24430 = x24429;
int32_t x24431 = x24427;
int32_t x24432 = x24428;
for(int x24433=0; x24433 < x24393; x24433++) {
int32_t x24434 = x24430;
int32_t x24435 = x24431;
float x24436 = x24288[x24435];
int32_t x24437 = x24432;
float x24438 = x85[x24437];
float x24439 = x24436 * x24438;
x24406[x24434] = x24439;
x24430 += 1;
if (x24442) {
x24431 += 1;
} else {
}

}
x24422 += x24393;
if (x24442) {
x24423 += x24275;
} else {
}

}
x24414 += x24394;
if (x24455) {
x24415 += x24276;
} else {
}
if (x24459) {
x24416 += 1;
} else {
}

}
x24407 += x24395;
x24408 += x24277;

}
int32_t x24469 = 0;
int32_t x24470 = 1;
x24470 *= 1;
x24469 += 1;
x24470 *= 1;
x24470 *= 1;
int32_t x24475 = x24469;
bool x24476 = x24475 >= 2;
if (x24476) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24481 = x24475 == 0;
if (x24481) {
int32_t x24482 = x24470;
bool x24483 = x24482 == 256;
if (x24483) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24490 = x24470;
int32_t x24491 = 256 / x24490;
bool x24497;
if (x452) {
bool x24492 = x24391 == 1;
bool x24493 = x24491 == 1;
bool x24494 = x24492 || x24493;
bool x24495 = x24391 == x24491;
bool x24496 = x24494 || x24495;
x24497 = x24496;
} else {
x24497 = false;
}
bool x24501;
if (x24497) {
x24501 = x24500;
} else {
x24501 = false;
}
bool x24502;
if (x24501) {
x24502 = x24500;
} else {
x24502 = false;
}
if (x24502) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24391,x24393,x24393,1,x24491,1,1);
assert(false && "");
}
bool x24508 = x24391 <= x24491;
int32_t x24509;
if (x24508) {
x24509 = x24491;
} else {
x24509 = x24391;
}
bool x24515 = x24509 > 0;
bool x24517;
if (x24515) {
x24517 = x24516;
} else {
x24517 = false;
}
bool x24518;
if (x24517) {
x24518 = x24516;
} else {
x24518 = false;
}
if (x24518) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(24391) x Sym(24393) x Sym(24393)"," x Const(1) x Sym(24491) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x24513 = x24509 * x24512;
int32_t x24514 = 64 * x24513;
float* x24524 = (float*)myMalloc(x24514 * sizeof(float));;
int32_t x24525 = 0;
int32_t x24526 = 0;
int32_t x24527 = 0;
bool x24573 = x24391 > 1;
bool x24577 = x24491 > 1;
for(int x24528=0; x24528 < 64; x24528++) {
int32_t x24529 = x24526;
int32_t x24530 = x24527;
int32_t x24531 = x24525;
int32_t x24532 = x24531;
int32_t x24533 = x24529;
int32_t x24534 = x24530;
for(int x24536=0; x24536 < x24509; x24536++) {
int32_t x24537 = x24533;
int32_t x24538 = x24534;
int32_t x24539 = x24532;
int32_t x24540 = x24539;
int32_t x24541 = x24537;
int32_t x24542 = x24538;
for(int x24544=0; x24544 < x24511; x24544++) {
int32_t x24545 = x24541;
int32_t x24546 = x24542;
int32_t x24547 = x24540;
int32_t x24548 = x24547;
int32_t x24549 = x24545;
int32_t x24550 = x24546;
for(int x24551=0; x24551 < x24511; x24551++) {
int32_t x24552 = x24548;
int32_t x24553 = x24549;
float x24554 = x24406[x24553];
int32_t x24555 = x24550;
float x24556 = x253[x24555];
float x24557 = x24554 + x24556;
x24524[x24552] = x24557;
x24548 += 1;
if (x24560) {
x24549 += 1;
} else {
}

}
x24540 += x24511;
if (x24560) {
x24541 += x24393;
} else {
}

}
x24532 += x24512;
if (x24573) {
x24533 += x24394;
} else {
}
if (x24577) {
x24534 += 1;
} else {
}

}
x24525 += x24513;
x24526 += x24395;

}
float* x24587 = (float*)myMalloc(x24514 * sizeof(float));;
for(int x24589=0; x24589 < x24514; x24589++) {
float x24590 = x24524[x24589];
bool x24591 = x24590 < 0.0f;
if (x24591) {
x24587[x24589] = 0.0f;
} else {
float x24594 = x24524[x24589];
x24587[x24589] = x24594;
}

}
float* x24609 = (float*)myMalloc(x24608 * sizeof(float));;
int32_t x24610 = 9 * x24509;
int32_t x24613 = 64 * x24610;
int32_t x24614 = x24613 * x24604;
float* x24615 = (float*)myMalloc(x24614 * sizeof(float));;
int32_t x24611 = x24610 * x24604;
int32_t x24623 = x24509 * 3;
int32_t x24624 = x24623 * 3;
for(int x24616=0; x24616 < 64; x24616++) {
int32_t x24617 = x24616 * x24513;
float* x24618 = x24587+x24617;
int32_t x24619 = x24616 * x24605;
float* x24620 = x24609+x24619;
int32_t x24621 = x24616 * x24611;
float* x24622 = x24615+x24621;
for(int x24626=0; x24626 < x24624; x24626++) {
int32_t x24627 = x24626 / 9;
int32_t x24631 = x24627 * 3;
int32_t x24632 = x24631 * 3;
int32_t x24633 = x24632 * x24603;
int32_t x24634 = x24633 * x24603;
int32_t x24628 = x24626 % 9;
int32_t x24629 = x24628 / 3;
int32_t x24635 = x24629 * 3;
int32_t x24636 = x24635 * x24603;
int32_t x24637 = x24636 * x24603;
int32_t x24638 = x24634 + x24637;
int32_t x24630 = x24628 % 3;
int32_t x24639 = x24630 * x24603;
int32_t x24640 = x24639 * x24603;
int32_t x24641 = x24638 + x24640;
float* x24642 = x24622+x24641;
int32_t x24643 = x24627 * x24511;
int32_t x24644 = x24643 * x24511;
float* x24645 = x24618+x24644;
int32_t x24658 = 1 - x24630;
bool x24659 = x24658 > 0;
int32_t x24660;
if (x24659) {
x24660 = x24658;
} else {
x24660 = 0;
}
int32_t x24661 = 3 - x24630;
int32_t x24662 = x24661 - 1;
int32_t x24663 = 1 - x24662;
bool x24664 = x24663 > 0;
int32_t x24665;
if (x24664) {
x24665 = x24663;
} else {
x24665 = 0;
}
int32_t x24666 = x24603 - x24665;
int32_t x24667 = x24666 - x24660;
bool x24668 = x24667 <= 0;
bool x24672 = x24660 > 0;
int32_t x24657 = -1 + x24630;
bool x24685 = x24665 > 0;
for(int x24647=0; x24647 < x24603; x24647++) {
int32_t x24648 = x24647 - 1;
int32_t x24649 = x24648 + x24629;
bool x24650 = x24649 < 0;
bool x24651 = x24649 >= x24511;
bool x24652 = x24650 || x24651;
if (x24652) {
int32_t x24653 = x24647 * x24603;
float* x24654 = x24642+x24653;
memset(x24654, 0, 4 * x24603);;
} else {
if (x24668) {
int32_t x24653 = x24647 * x24603;
float* x24669 = x24642+x24653;
memset(x24669, 0, 4 * x24603);;
} else {
int32_t x24653 = x24647 * x24603;
if (x24672) {
float* x24673 = x24642+x24653;
memset(x24673, 0, 4 * x24660);;
} else {
}
// may have segfault here
int32_t x24678 = x24653 + x24660;
float* x24679 = x24642+x24678;
int32_t x24680 = x24649 * x24511;
int32_t x24681 = x24680 + x24657;
int32_t x24682 = x24681 + x24660;
float* x24683 = x24645+x24682;
memcpy(x24679, x24683, 4 * x24667);;
if (x24685) {
int32_t x24686 = x24653 + x24603;
int32_t x24687 = x24686 - x24665;
float* x24688 = x24642+x24687;
memset(x24688, 0, 4 * x24665);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x24604,x24610,1,x226,x24610,x24622,x24604,1,x24620,x24604);

}
int32_t x24703 = 0;
int32_t x24704 = 1;
x24704 *= 1;
x24703 += 1;
x24704 *= 1;
x24704 *= 1;
int32_t x24709 = x24703;
bool x24710 = x24709 >= 2;
if (x24710) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24715 = x24709 == 0;
if (x24715) {
int32_t x24716 = x24704;
bool x24717 = x24716 == 256;
if (x24717) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24724 = x24704;
int32_t x24725 = 256 / x24724;
bool x24729;
if (x452) {
bool x24726 = x24725 == 1;
bool x24727 = 256 == x24725;
bool x24728 = x24726 || x24727;
x24729 = x24728;
} else {
x24729 = false;
}
bool x24733;
if (x24729) {
x24733 = x24732;
} else {
x24733 = false;
}
bool x24734;
if (x24733) {
x24734 = x24732;
} else {
x24734 = false;
}
if (x24734) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,256,x24603,x24603,1,x24725,1,1);
assert(false && "");
}
bool x24740 = 256 <= x24725;
int32_t x24741;
if (x24740) {
x24741 = x24725;
} else {
x24741 = 256;
}
bool x24747 = x24741 > 0;
bool x24749;
if (x24747) {
x24749 = x24748;
} else {
x24749 = false;
}
bool x24750;
if (x24749) {
x24750 = x24748;
} else {
x24750 = false;
}
if (x24750) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(256) x Sym(24603) x Sym(24603)"," x Const(1) x Sym(24725) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x24745 = x24741 * x24744;
int32_t x24746 = 64 * x24745;
float* x24756 = (float*)myMalloc(x24746 * sizeof(float));;
int32_t x24757 = 0;
int32_t x24758 = 0;
int32_t x24759 = 0;
bool x24806 = x24725 > 1;
for(int x24760=0; x24760 < 64; x24760++) {
int32_t x24761 = x24758;
int32_t x24762 = x24759;
int32_t x24763 = x24757;
int32_t x24764 = x24763;
int32_t x24765 = x24761;
int32_t x24766 = x24762;
for(int x24768=0; x24768 < x24741; x24768++) {
int32_t x24769 = x24765;
int32_t x24770 = x24766;
int32_t x24771 = x24764;
int32_t x24772 = x24771;
int32_t x24773 = x24769;
int32_t x24774 = x24770;
for(int x24776=0; x24776 < x24743; x24776++) {
int32_t x24777 = x24773;
int32_t x24778 = x24774;
int32_t x24779 = x24772;
int32_t x24780 = x24779;
int32_t x24781 = x24777;
int32_t x24782 = x24778;
for(int x24783=0; x24783 < x24743; x24783++) {
int32_t x24784 = x24780;
int32_t x24785 = x24781;
float x24786 = x24609[x24785];
int32_t x24787 = x24782;
float x24788 = x70[x24787];
float x24789 = x24786 - x24788;
x24756[x24784] = x24789;
x24780 += 1;
if (x24792) {
x24781 += 1;
} else {
}

}
x24772 += x24743;
if (x24792) {
x24773 += x24603;
} else {
}

}
x24764 += x24744;
x24765 += x24604;
if (x24806) {
x24766 += 1;
} else {
}

}
x24757 += x24745;
x24758 += x24605;

}
float* x24816 = (float*)myMalloc(256 * sizeof(float));;
for(int x24817=0; x24817 < 256; x24817++) {
float x24818 = x240[x24817];
float x24819 = x24818 + 1.0E-5f;
x24816[x24817] = x24819;

}
float* x24823 = (float*)myMalloc(256 * sizeof(float));;
for(int x24824=0; x24824 < 256; x24824++) {
float x24825 = x24816[x24824];
double x24826 = (double)x24825;
double x24827 = sqrt(x24826);
float x24828 = (float)x24827;
x24823[x24824] = x24828;

}
int32_t x24832 = 0;
int32_t x24833 = 1;
x24833 *= 1;
x24832 += 1;
x24833 *= 1;
x24833 *= 1;
int32_t x24838 = x24832;
bool x24839 = x24838 >= 2;
if (x24839) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24844 = x24838 == 0;
if (x24844) {
int32_t x24845 = x24833;
bool x24846 = x24845 == 256;
if (x24846) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24853 = x24833;
int32_t x24854 = 256 / x24853;
bool x24860;
if (x452) {
bool x24855 = x24741 == 1;
bool x24856 = x24854 == 1;
bool x24857 = x24855 || x24856;
bool x24858 = x24741 == x24854;
bool x24859 = x24857 || x24858;
x24860 = x24859;
} else {
x24860 = false;
}
bool x24864;
if (x24860) {
x24864 = x24863;
} else {
x24864 = false;
}
bool x24865;
if (x24864) {
x24865 = x24863;
} else {
x24865 = false;
}
if (x24865) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24741,x24743,x24743,1,x24854,1,1);
assert(false && "");
}
bool x24871 = x24741 <= x24854;
int32_t x24872;
if (x24871) {
x24872 = x24854;
} else {
x24872 = x24741;
}
bool x24878 = x24872 > 0;
bool x24880;
if (x24878) {
x24880 = x24879;
} else {
x24880 = false;
}
bool x24881;
if (x24880) {
x24881 = x24879;
} else {
x24881 = false;
}
if (x24881) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(24741) x Sym(24743) x Sym(24743)"," x Const(1) x Sym(24854) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x24876 = x24872 * x24875;
int32_t x24877 = 64 * x24876;
float* x24887 = (float*)myMalloc(x24877 * sizeof(float));;
int32_t x24888 = 0;
int32_t x24889 = 0;
int32_t x24890 = 0;
bool x24936 = x24741 > 1;
bool x24940 = x24854 > 1;
for(int x24891=0; x24891 < 64; x24891++) {
int32_t x24892 = x24889;
int32_t x24893 = x24890;
int32_t x24894 = x24888;
int32_t x24895 = x24894;
int32_t x24896 = x24892;
int32_t x24897 = x24893;
for(int x24899=0; x24899 < x24872; x24899++) {
int32_t x24900 = x24896;
int32_t x24901 = x24897;
int32_t x24902 = x24895;
int32_t x24903 = x24902;
int32_t x24904 = x24900;
int32_t x24905 = x24901;
for(int x24907=0; x24907 < x24874; x24907++) {
int32_t x24908 = x24904;
int32_t x24909 = x24905;
int32_t x24910 = x24903;
int32_t x24911 = x24910;
int32_t x24912 = x24908;
int32_t x24913 = x24909;
for(int x24914=0; x24914 < x24874; x24914++) {
int32_t x24915 = x24911;
int32_t x24916 = x24912;
float x24917 = x24756[x24916];
int32_t x24918 = x24913;
float x24919 = x24823[x24918];
float x24920 = x24917 / x24919;
x24887[x24915] = x24920;
x24911 += 1;
if (x24923) {
x24912 += 1;
} else {
}

}
x24903 += x24874;
if (x24923) {
x24904 += x24743;
} else {
}

}
x24895 += x24875;
if (x24936) {
x24896 += x24744;
} else {
}
if (x24940) {
x24897 += 1;
} else {
}

}
x24888 += x24876;
x24889 += x24745;

}
int32_t x24950 = 0;
int32_t x24951 = 1;
x24951 *= 1;
x24950 += 1;
x24951 *= 1;
x24951 *= 1;
int32_t x24956 = x24950;
bool x24957 = x24956 >= 2;
if (x24957) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x24962 = x24956 == 0;
if (x24962) {
int32_t x24963 = x24951;
bool x24964 = x24963 == 256;
if (x24964) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x24971 = x24951;
int32_t x24972 = 256 / x24971;
bool x24978;
if (x452) {
bool x24973 = x24872 == 1;
bool x24974 = x24972 == 1;
bool x24975 = x24973 || x24974;
bool x24976 = x24872 == x24972;
bool x24977 = x24975 || x24976;
x24978 = x24977;
} else {
x24978 = false;
}
bool x24982;
if (x24978) {
x24982 = x24981;
} else {
x24982 = false;
}
bool x24983;
if (x24982) {
x24983 = x24981;
} else {
x24983 = false;
}
if (x24983) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24872,x24874,x24874,1,x24972,1,1);
assert(false && "");
}
bool x24989 = x24872 <= x24972;
int32_t x24990;
if (x24989) {
x24990 = x24972;
} else {
x24990 = x24872;
}
bool x24996 = x24990 > 0;
bool x24998;
if (x24996) {
x24998 = x24997;
} else {
x24998 = false;
}
bool x24999;
if (x24998) {
x24999 = x24997;
} else {
x24999 = false;
}
if (x24999) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(24872) x Sym(24874) x Sym(24874)"," x Const(1) x Sym(24972) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x24994 = x24990 * x24993;
int32_t x24995 = 64 * x24994;
float* x25005 = (float*)myMalloc(x24995 * sizeof(float));;
int32_t x25006 = 0;
int32_t x25007 = 0;
int32_t x25008 = 0;
bool x25054 = x24872 > 1;
bool x25058 = x24972 > 1;
for(int x25009=0; x25009 < 64; x25009++) {
int32_t x25010 = x25007;
int32_t x25011 = x25008;
int32_t x25012 = x25006;
int32_t x25013 = x25012;
int32_t x25014 = x25010;
int32_t x25015 = x25011;
for(int x25017=0; x25017 < x24990; x25017++) {
int32_t x25018 = x25014;
int32_t x25019 = x25015;
int32_t x25020 = x25013;
int32_t x25021 = x25020;
int32_t x25022 = x25018;
int32_t x25023 = x25019;
for(int x25025=0; x25025 < x24992; x25025++) {
int32_t x25026 = x25022;
int32_t x25027 = x25023;
int32_t x25028 = x25021;
int32_t x25029 = x25028;
int32_t x25030 = x25026;
int32_t x25031 = x25027;
for(int x25032=0; x25032 < x24992; x25032++) {
int32_t x25033 = x25029;
int32_t x25034 = x25030;
float x25035 = x24887[x25034];
int32_t x25036 = x25031;
float x25037 = x141[x25036];
float x25038 = x25035 * x25037;
x25005[x25033] = x25038;
x25029 += 1;
if (x25041) {
x25030 += 1;
} else {
}

}
x25021 += x24992;
if (x25041) {
x25022 += x24874;
} else {
}

}
x25013 += x24993;
if (x25054) {
x25014 += x24875;
} else {
}
if (x25058) {
x25015 += 1;
} else {
}

}
x25006 += x24994;
x25007 += x24876;

}
int32_t x25068 = 0;
int32_t x25069 = 1;
x25069 *= 1;
x25068 += 1;
x25069 *= 1;
x25069 *= 1;
int32_t x25074 = x25068;
bool x25075 = x25074 >= 2;
if (x25075) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x25080 = x25074 == 0;
if (x25080) {
int32_t x25081 = x25069;
bool x25082 = x25081 == 256;
if (x25082) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x25089 = x25069;
int32_t x25090 = 256 / x25089;
bool x25096;
if (x452) {
bool x25091 = x24990 == 1;
bool x25092 = x25090 == 1;
bool x25093 = x25091 || x25092;
bool x25094 = x24990 == x25090;
bool x25095 = x25093 || x25094;
x25096 = x25095;
} else {
x25096 = false;
}
bool x25100;
if (x25096) {
x25100 = x25099;
} else {
x25100 = false;
}
bool x25101;
if (x25100) {
x25101 = x25099;
} else {
x25101 = false;
}
if (x25101) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x24990,x24992,x24992,1,x25090,1,1);
assert(false && "");
}
bool x25107 = x24990 <= x25090;
int32_t x25108;
if (x25107) {
x25108 = x25090;
} else {
x25108 = x24990;
}
bool x25114 = x25108 > 0;
bool x25116;
if (x25114) {
x25116 = x25115;
} else {
x25116 = false;
}
bool x25117;
if (x25116) {
x25117 = x25115;
} else {
x25117 = false;
}
if (x25117) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(24990) x Sym(24992) x Sym(24992)"," x Const(1) x Sym(25090) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x25112 = x25108 * x25111;
int32_t x25113 = 64 * x25112;
float* x25123 = (float*)myMalloc(x25113 * sizeof(float));;
int32_t x25124 = 0;
int32_t x25125 = 0;
int32_t x25126 = 0;
bool x25172 = x24990 > 1;
bool x25176 = x25090 > 1;
for(int x25127=0; x25127 < 64; x25127++) {
int32_t x25128 = x25125;
int32_t x25129 = x25126;
int32_t x25130 = x25124;
int32_t x25131 = x25130;
int32_t x25132 = x25128;
int32_t x25133 = x25129;
for(int x25135=0; x25135 < x25108; x25135++) {
int32_t x25136 = x25132;
int32_t x25137 = x25133;
int32_t x25138 = x25131;
int32_t x25139 = x25138;
int32_t x25140 = x25136;
int32_t x25141 = x25137;
for(int x25143=0; x25143 < x25110; x25143++) {
int32_t x25144 = x25140;
int32_t x25145 = x25141;
int32_t x25146 = x25139;
int32_t x25147 = x25146;
int32_t x25148 = x25144;
int32_t x25149 = x25145;
for(int x25150=0; x25150 < x25110; x25150++) {
int32_t x25151 = x25147;
int32_t x25152 = x25148;
float x25153 = x25005[x25152];
int32_t x25154 = x25149;
float x25155 = x189[x25154];
float x25156 = x25153 + x25155;
x25123[x25151] = x25156;
x25147 += 1;
if (x25159) {
x25148 += 1;
} else {
}

}
x25139 += x25110;
if (x25159) {
x25140 += x24992;
} else {
}

}
x25131 += x25111;
if (x25172) {
x25132 += x24993;
} else {
}
if (x25176) {
x25133 += 1;
} else {
}

}
x25124 += x25112;
x25125 += x24994;

}
float* x25186 = (float*)myMalloc(x25113 * sizeof(float));;
for(int x25188=0; x25188 < x25113; x25188++) {
float x25189 = x25123[x25188];
bool x25190 = x25189 < 0.0f;
if (x25190) {
x25186[x25188] = 0.0f;
} else {
float x25193 = x25123[x25188];
x25186[x25188] = x25193;
}

}
float* x25207 = (float*)myMalloc(x25206 * sizeof(float));;
int32_t x25210 = 64 * x25108;
int32_t x25211 = x25210 * x25202;
float* x25212 = (float*)myMalloc(x25211 * sizeof(float));;
int32_t x25208 = x25108 * x25202;
for(int x25213=0; x25213 < 64; x25213++) {
int32_t x25214 = x25213 * x25112;
float* x25215 = x25186+x25214;
int32_t x25216 = x25213 * x25203;
float* x25217 = x25207+x25216;
int32_t x25218 = x25213 * x25208;
float* x25219 = x25212+x25218;
for(int x25220=0; x25220 < x25108; x25220++) {
int32_t x25221 = x25220 / 1;
int32_t x25225 = x25221 * x25201;
int32_t x25226 = x25225 * x25201;
int32_t x25222 = x25220 % 1;
int32_t x25223 = x25222 / 1;
int32_t x25227 = x25223 * x25201;
int32_t x25228 = x25227 * x25201;
int32_t x25229 = x25226 + x25228;
int32_t x25224 = x25222 % 1;
int32_t x25230 = x25224 * x25201;
int32_t x25231 = x25230 * x25201;
int32_t x25232 = x25229 + x25231;
float* x25233 = x25219+x25232;
int32_t x25234 = x25221 * x25110;
int32_t x25235 = x25234 * x25110;
float* x25236 = x25215+x25235;
for(int x25238=0; x25238 < x25201; x25238++) {
int32_t x25240 = x25238 * x25201;
float* x25241 = x25233+x25240;
int32_t x25239 = x25238 + x25223;
int32_t x25242 = x25239 * x25110;
int32_t x25243 = x25242 + x25224;
float* x25244 = x25236+x25243;
memcpy(x25241, x25244, 4 * x25201);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x25202,x25108,1,x97,x25108,x25219,x25202,1,x25217,x25202);

}
int32_t x25253 = 0;
int32_t x25254 = 1;
x25254 *= 1;
x25253 += 1;
x25254 *= 1;
x25254 *= 1;
int32_t x25259 = x25253;
bool x25260 = x25259 >= 2;
if (x25260) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x25265 = x25259 == 0;
if (x25265) {
int32_t x25266 = x25254;
bool x25267 = x25266 == 1024;
if (x25267) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x25274 = x25254;
int32_t x25275 = 1024 / x25274;
bool x25279;
if (x452) {
bool x25276 = x25275 == 1;
bool x25277 = 1024 == x25275;
bool x25278 = x25276 || x25277;
x25279 = x25278;
} else {
x25279 = false;
}
bool x25283;
if (x25279) {
x25283 = x25282;
} else {
x25283 = false;
}
bool x25284;
if (x25283) {
x25284 = x25282;
} else {
x25284 = false;
}
if (x25284) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,1024,x25201,x25201,1,x25275,1,1);
assert(false && "");
}
bool x25290 = 1024 <= x25275;
int32_t x25291;
if (x25290) {
x25291 = x25275;
} else {
x25291 = 1024;
}
bool x25297 = x25291 > 0;
bool x25299;
if (x25297) {
x25299 = x25298;
} else {
x25299 = false;
}
bool x25300;
if (x25299) {
x25300 = x25298;
} else {
x25300 = false;
}
if (x25300) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(1024) x Sym(25201) x Sym(25201)"," x Const(1) x Sym(25275) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x25295 = x25291 * x25294;
int32_t x25296 = 64 * x25295;
float* x25306 = (float*)myMalloc(x25296 * sizeof(float));;
int32_t x25307 = 0;
int32_t x25308 = 0;
int32_t x25309 = 0;
bool x25356 = x25275 > 1;
for(int x25310=0; x25310 < 64; x25310++) {
int32_t x25311 = x25308;
int32_t x25312 = x25309;
int32_t x25313 = x25307;
int32_t x25314 = x25313;
int32_t x25315 = x25311;
int32_t x25316 = x25312;
for(int x25318=0; x25318 < x25291; x25318++) {
int32_t x25319 = x25315;
int32_t x25320 = x25316;
int32_t x25321 = x25314;
int32_t x25322 = x25321;
int32_t x25323 = x25319;
int32_t x25324 = x25320;
for(int x25326=0; x25326 < x25293; x25326++) {
int32_t x25327 = x25323;
int32_t x25328 = x25324;
int32_t x25329 = x25322;
int32_t x25330 = x25329;
int32_t x25331 = x25327;
int32_t x25332 = x25328;
for(int x25333=0; x25333 < x25293; x25333++) {
int32_t x25334 = x25330;
int32_t x25335 = x25331;
float x25336 = x25207[x25335];
int32_t x25337 = x25332;
float x25338 = x122[x25337];
float x25339 = x25336 - x25338;
x25306[x25334] = x25339;
x25330 += 1;
if (x25342) {
x25331 += 1;
} else {
}

}
x25322 += x25293;
if (x25342) {
x25323 += x25201;
} else {
}

}
x25314 += x25294;
x25315 += x25202;
if (x25356) {
x25316 += 1;
} else {
}

}
x25307 += x25295;
x25308 += x25203;

}
float* x25366 = (float*)myMalloc(1024 * sizeof(float));;
for(int x25367=0; x25367 < 1024; x25367++) {
float x25368 = x183[x25367];
float x25369 = x25368 + 1.0E-5f;
x25366[x25367] = x25369;

}
float* x25373 = (float*)myMalloc(1024 * sizeof(float));;
for(int x25374=0; x25374 < 1024; x25374++) {
float x25375 = x25366[x25374];
double x25376 = (double)x25375;
double x25377 = sqrt(x25376);
float x25378 = (float)x25377;
x25373[x25374] = x25378;

}
int32_t x25382 = 0;
int32_t x25383 = 1;
x25383 *= 1;
x25382 += 1;
x25383 *= 1;
x25383 *= 1;
int32_t x25388 = x25382;
bool x25389 = x25388 >= 2;
if (x25389) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x25394 = x25388 == 0;
if (x25394) {
int32_t x25395 = x25383;
bool x25396 = x25395 == 1024;
if (x25396) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x25403 = x25383;
int32_t x25404 = 1024 / x25403;
bool x25410;
if (x452) {
bool x25405 = x25291 == 1;
bool x25406 = x25404 == 1;
bool x25407 = x25405 || x25406;
bool x25408 = x25291 == x25404;
bool x25409 = x25407 || x25408;
x25410 = x25409;
} else {
x25410 = false;
}
bool x25414;
if (x25410) {
x25414 = x25413;
} else {
x25414 = false;
}
bool x25415;
if (x25414) {
x25415 = x25413;
} else {
x25415 = false;
}
if (x25415) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x25291,x25293,x25293,1,x25404,1,1);
assert(false && "");
}
bool x25421 = x25291 <= x25404;
int32_t x25422;
if (x25421) {
x25422 = x25404;
} else {
x25422 = x25291;
}
bool x25428 = x25422 > 0;
bool x25430;
if (x25428) {
x25430 = x25429;
} else {
x25430 = false;
}
bool x25431;
if (x25430) {
x25431 = x25429;
} else {
x25431 = false;
}
if (x25431) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(25291) x Sym(25293) x Sym(25293)"," x Const(1) x Sym(25404) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x25426 = x25422 * x25425;
int32_t x25427 = 64 * x25426;
float* x25437 = (float*)myMalloc(x25427 * sizeof(float));;
int32_t x25438 = 0;
int32_t x25439 = 0;
int32_t x25440 = 0;
bool x25486 = x25291 > 1;
bool x25490 = x25404 > 1;
for(int x25441=0; x25441 < 64; x25441++) {
int32_t x25442 = x25439;
int32_t x25443 = x25440;
int32_t x25444 = x25438;
int32_t x25445 = x25444;
int32_t x25446 = x25442;
int32_t x25447 = x25443;
for(int x25449=0; x25449 < x25422; x25449++) {
int32_t x25450 = x25446;
int32_t x25451 = x25447;
int32_t x25452 = x25445;
int32_t x25453 = x25452;
int32_t x25454 = x25450;
int32_t x25455 = x25451;
for(int x25457=0; x25457 < x25424; x25457++) {
int32_t x25458 = x25454;
int32_t x25459 = x25455;
int32_t x25460 = x25453;
int32_t x25461 = x25460;
int32_t x25462 = x25458;
int32_t x25463 = x25459;
for(int x25464=0; x25464 < x25424; x25464++) {
int32_t x25465 = x25461;
int32_t x25466 = x25462;
float x25467 = x25306[x25466];
int32_t x25468 = x25463;
float x25469 = x25373[x25468];
float x25470 = x25467 / x25469;
x25437[x25465] = x25470;
x25461 += 1;
if (x25473) {
x25462 += 1;
} else {
}

}
x25453 += x25424;
if (x25473) {
x25454 += x25293;
} else {
}

}
x25445 += x25425;
if (x25486) {
x25446 += x25294;
} else {
}
if (x25490) {
x25447 += 1;
} else {
}

}
x25438 += x25426;
x25439 += x25295;

}
int32_t x25500 = 0;
int32_t x25501 = 1;
x25501 *= 1;
x25500 += 1;
x25501 *= 1;
x25501 *= 1;
int32_t x25506 = x25500;
bool x25507 = x25506 >= 2;
if (x25507) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x25512 = x25506 == 0;
if (x25512) {
int32_t x25513 = x25501;
bool x25514 = x25513 == 1024;
if (x25514) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x25521 = x25501;
int32_t x25522 = 1024 / x25521;
bool x25528;
if (x452) {
bool x25523 = x25422 == 1;
bool x25524 = x25522 == 1;
bool x25525 = x25523 || x25524;
bool x25526 = x25422 == x25522;
bool x25527 = x25525 || x25526;
x25528 = x25527;
} else {
x25528 = false;
}
bool x25532;
if (x25528) {
x25532 = x25531;
} else {
x25532 = false;
}
bool x25533;
if (x25532) {
x25533 = x25531;
} else {
x25533 = false;
}
if (x25533) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x25422,x25424,x25424,1,x25522,1,1);
assert(false && "");
}
bool x25539 = x25422 <= x25522;
int32_t x25540;
if (x25539) {
x25540 = x25522;
} else {
x25540 = x25422;
}
bool x25546 = x25540 > 0;
bool x25548;
if (x25546) {
x25548 = x25547;
} else {
x25548 = false;
}
bool x25549;
if (x25548) {
x25549 = x25547;
} else {
x25549 = false;
}
if (x25549) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(25422) x Sym(25424) x Sym(25424)"," x Const(1) x Sym(25522) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x25544 = x25540 * x25543;
int32_t x25545 = 64 * x25544;
float* x25555 = (float*)myMalloc(x25545 * sizeof(float));;
int32_t x25556 = 0;
int32_t x25557 = 0;
int32_t x25558 = 0;
bool x25604 = x25422 > 1;
bool x25608 = x25522 > 1;
for(int x25559=0; x25559 < 64; x25559++) {
int32_t x25560 = x25557;
int32_t x25561 = x25558;
int32_t x25562 = x25556;
int32_t x25563 = x25562;
int32_t x25564 = x25560;
int32_t x25565 = x25561;
for(int x25567=0; x25567 < x25540; x25567++) {
int32_t x25568 = x25564;
int32_t x25569 = x25565;
int32_t x25570 = x25563;
int32_t x25571 = x25570;
int32_t x25572 = x25568;
int32_t x25573 = x25569;
for(int x25575=0; x25575 < x25542; x25575++) {
int32_t x25576 = x25572;
int32_t x25577 = x25573;
int32_t x25578 = x25571;
int32_t x25579 = x25578;
int32_t x25580 = x25576;
int32_t x25581 = x25577;
for(int x25582=0; x25582 < x25542; x25582++) {
int32_t x25583 = x25579;
int32_t x25584 = x25580;
float x25585 = x25437[x25584];
int32_t x25586 = x25581;
float x25587 = x248[x25586];
float x25588 = x25585 * x25587;
x25555[x25583] = x25588;
x25579 += 1;
if (x25591) {
x25580 += 1;
} else {
}

}
x25571 += x25542;
if (x25591) {
x25572 += x25424;
} else {
}

}
x25563 += x25543;
if (x25604) {
x25564 += x25425;
} else {
}
if (x25608) {
x25565 += 1;
} else {
}

}
x25556 += x25544;
x25557 += x25426;

}
int32_t x25618 = 0;
int32_t x25619 = 1;
x25619 *= 1;
x25618 += 1;
x25619 *= 1;
x25619 *= 1;
int32_t x25624 = x25618;
bool x25625 = x25624 >= 2;
if (x25625) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x25630 = x25624 == 0;
if (x25630) {
int32_t x25631 = x25619;
bool x25632 = x25631 == 1024;
if (x25632) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x25639 = x25619;
int32_t x25640 = 1024 / x25639;
bool x25646;
if (x452) {
bool x25641 = x25540 == 1;
bool x25642 = x25640 == 1;
bool x25643 = x25641 || x25642;
bool x25644 = x25540 == x25640;
bool x25645 = x25643 || x25644;
x25646 = x25645;
} else {
x25646 = false;
}
bool x25650;
if (x25646) {
x25650 = x25649;
} else {
x25650 = false;
}
bool x25651;
if (x25650) {
x25651 = x25649;
} else {
x25651 = false;
}
if (x25651) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x25540,x25542,x25542,1,x25640,1,1);
assert(false && "");
}
bool x25657 = x25540 <= x25640;
int32_t x25658;
if (x25657) {
x25658 = x25640;
} else {
x25658 = x25540;
}
bool x25664 = x25658 > 0;
bool x25666;
if (x25664) {
x25666 = x25665;
} else {
x25666 = false;
}
bool x25667;
if (x25666) {
x25667 = x25665;
} else {
x25667 = false;
}
if (x25667) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(25540) x Sym(25542) x Sym(25542)"," x Const(1) x Sym(25640) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x25662 = x25658 * x25661;
int32_t x25663 = 64 * x25662;
float* x25673 = (float*)myMalloc(x25663 * sizeof(float));;
int32_t x25674 = 0;
int32_t x25675 = 0;
int32_t x25676 = 0;
bool x25722 = x25540 > 1;
bool x25726 = x25640 > 1;
for(int x25677=0; x25677 < 64; x25677++) {
int32_t x25678 = x25675;
int32_t x25679 = x25676;
int32_t x25680 = x25674;
int32_t x25681 = x25680;
int32_t x25682 = x25678;
int32_t x25683 = x25679;
for(int x25685=0; x25685 < x25658; x25685++) {
int32_t x25686 = x25682;
int32_t x25687 = x25683;
int32_t x25688 = x25681;
int32_t x25689 = x25688;
int32_t x25690 = x25686;
int32_t x25691 = x25687;
for(int x25693=0; x25693 < x25660; x25693++) {
int32_t x25694 = x25690;
int32_t x25695 = x25691;
int32_t x25696 = x25689;
int32_t x25697 = x25696;
int32_t x25698 = x25694;
int32_t x25699 = x25695;
for(int x25700=0; x25700 < x25660; x25700++) {
int32_t x25701 = x25697;
int32_t x25702 = x25698;
float x25703 = x25555[x25702];
int32_t x25704 = x25699;
float x25705 = x93[x25704];
float x25706 = x25703 + x25705;
x25673[x25701] = x25706;
x25697 += 1;
if (x25709) {
x25698 += 1;
} else {
}

}
x25689 += x25660;
if (x25709) {
x25690 += x25542;
} else {
}

}
x25681 += x25661;
if (x25722) {
x25682 += x25543;
} else {
}
if (x25726) {
x25683 += 1;
} else {
}

}
x25674 += x25662;
x25675 += x25544;

}
bool x25736 = x25658 == 1;
bool x25737 = x25736 || x23948;
bool x25738 = x25658 == x23870;
bool x25739 = x25737 || x25738;
bool x25744;
if (x25739) {
x25744 = x25743;
} else {
x25744 = false;
}
bool x25745;
if (x25744) {
x25745 = x25743;
} else {
x25745 = false;
}
if (x25745) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x25658,x25660,x25660,64,x23870,x23872,x23872);
assert(false && "");
}
int32_t x25758 = 0;
int32_t x25759 = 0;
int32_t x25760 = 0;
bool x25751 = x25658 <= x23870;
int32_t x25752;
if (x25751) {
x25752 = x23870;
} else {
x25752 = x25658;
}
bool x25811 = x25658 > 1;
int32_t x25756 = x25752 * x25755;
for(int x25761=0; x25761 < 64; x25761++) {
int32_t x25762 = x25759;
int32_t x25763 = x25760;
int32_t x25764 = x25758;
int32_t x25765 = x25764;
int32_t x25766 = x25762;
int32_t x25767 = x25763;
for(int x25769=0; x25769 < x25752; x25769++) {
int32_t x25770 = x25766;
int32_t x25771 = x25767;
int32_t x25772 = x25765;
int32_t x25773 = x25772;
int32_t x25774 = x25770;
int32_t x25775 = x25771;
for(int x25777=0; x25777 < x25754; x25777++) {
int32_t x25778 = x25774;
int32_t x25779 = x25775;
int32_t x25780 = x25773;
int32_t x25781 = x25780;
int32_t x25782 = x25778;
int32_t x25783 = x25779;
for(int x25784=0; x25784 < x25754; x25784++) {
int32_t x25785 = x25782;
float x25786 = x25673[x25785];
int32_t x25787 = x25783;
float x25788 = x24037[x25787];
float x25789 = x25786 + x25788;
x25673[x25785] = x25789;
x25781 += 1;
if (x25792) {
x25782 += 1;
} else {
}
if (x24004) {
x25783 += 1;
} else {
}

}
x25773 += x25754;
if (x25792) {
x25774 += x25660;
} else {
}
if (x24004) {
x25775 += x23872;
} else {
}

}
x25765 += x25755;
if (x25811) {
x25766 += x25661;
} else {
}
if (x24023) {
x25767 += x23873;
} else {
}

}
x25758 += x25756;
x25759 += x25662;
x25760 += x23874;

}
float* x25825 = (float*)myMalloc(x25663 * sizeof(float));;
for(int x25827=0; x25827 < x25663; x25827++) {
float x25828 = x25673[x25827];
bool x25829 = x25828 < 0.0f;
if (x25829) {
x25825[x25827] = 0.0f;
} else {
float x25832 = x25673[x25827];
x25825[x25827] = x25832;
}

}
float* x25846 = (float*)myMalloc(x25845 * sizeof(float));;
int32_t x25849 = 64 * x25658;
int32_t x25850 = x25849 * x25841;
float* x25851 = (float*)myMalloc(x25850 * sizeof(float));;
int32_t x25847 = x25658 * x25841;
for(int x25852=0; x25852 < 64; x25852++) {
int32_t x25853 = x25852 * x25662;
float* x25854 = x25825+x25853;
int32_t x25855 = x25852 * x25842;
float* x25856 = x25846+x25855;
int32_t x25857 = x25852 * x25847;
float* x25858 = x25851+x25857;
for(int x25859=0; x25859 < x25658; x25859++) {
int32_t x25860 = x25859 / 1;
int32_t x25864 = x25860 * x25840;
int32_t x25865 = x25864 * x25840;
int32_t x25861 = x25859 % 1;
int32_t x25862 = x25861 / 1;
int32_t x25866 = x25862 * x25840;
int32_t x25867 = x25866 * x25840;
int32_t x25868 = x25865 + x25867;
int32_t x25863 = x25861 % 1;
int32_t x25869 = x25863 * x25840;
int32_t x25870 = x25869 * x25840;
int32_t x25871 = x25868 + x25870;
float* x25872 = x25858+x25871;
int32_t x25873 = x25860 * x25660;
int32_t x25874 = x25873 * x25660;
float* x25875 = x25854+x25874;
for(int x25877=0; x25877 < x25840; x25877++) {
int32_t x25879 = x25877 * x25840;
float* x25880 = x25872+x25879;
int32_t x25878 = x25877 + x25862;
int32_t x25881 = x25878 * x25660;
int32_t x25882 = x25881 + x25863;
float* x25883 = x25875+x25882;
memcpy(x25880, x25883, 4 * x25840);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x25841,x25658,1,x139,x25658,x25858,x25841,1,x25856,x25841);

}
int32_t x25892 = 0;
int32_t x25893 = 1;
x25893 *= 1;
x25892 += 1;
x25893 *= 1;
x25893 *= 1;
int32_t x25898 = x25892;
bool x25899 = x25898 >= 2;
if (x25899) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x25904 = x25898 == 0;
if (x25904) {
int32_t x25905 = x25893;
bool x25906 = x25905 == 512;
if (x25906) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x25913 = x25893;
int32_t x25914 = 512 / x25913;
bool x25918;
if (x452) {
bool x25915 = x25914 == 1;
bool x25916 = 512 == x25914;
bool x25917 = x25915 || x25916;
x25918 = x25917;
} else {
x25918 = false;
}
bool x25922;
if (x25918) {
x25922 = x25921;
} else {
x25922 = false;
}
bool x25923;
if (x25922) {
x25923 = x25921;
} else {
x25923 = false;
}
if (x25923) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x25840,x25840,1,x25914,1,1);
assert(false && "");
}
bool x25929 = 512 <= x25914;
int32_t x25930;
if (x25929) {
x25930 = x25914;
} else {
x25930 = 512;
}
bool x25936 = x25930 > 0;
bool x25938;
if (x25936) {
x25938 = x25937;
} else {
x25938 = false;
}
bool x25939;
if (x25938) {
x25939 = x25937;
} else {
x25939 = false;
}
if (x25939) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(25840) x Sym(25840)"," x Const(1) x Sym(25914) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x25934 = x25930 * x25933;
int32_t x25935 = 64 * x25934;
float* x25945 = (float*)myMalloc(x25935 * sizeof(float));;
int32_t x25946 = 0;
int32_t x25947 = 0;
int32_t x25948 = 0;
bool x25995 = x25914 > 1;
for(int x25949=0; x25949 < 64; x25949++) {
int32_t x25950 = x25947;
int32_t x25951 = x25948;
int32_t x25952 = x25946;
int32_t x25953 = x25952;
int32_t x25954 = x25950;
int32_t x25955 = x25951;
for(int x25957=0; x25957 < x25930; x25957++) {
int32_t x25958 = x25954;
int32_t x25959 = x25955;
int32_t x25960 = x25953;
int32_t x25961 = x25960;
int32_t x25962 = x25958;
int32_t x25963 = x25959;
for(int x25965=0; x25965 < x25932; x25965++) {
int32_t x25966 = x25962;
int32_t x25967 = x25963;
int32_t x25968 = x25961;
int32_t x25969 = x25968;
int32_t x25970 = x25966;
int32_t x25971 = x25967;
for(int x25972=0; x25972 < x25932; x25972++) {
int32_t x25973 = x25969;
int32_t x25974 = x25970;
float x25975 = x25846[x25974];
int32_t x25976 = x25971;
float x25977 = x67[x25976];
float x25978 = x25975 - x25977;
x25945[x25973] = x25978;
x25969 += 1;
if (x25981) {
x25970 += 1;
} else {
}

}
x25961 += x25932;
if (x25981) {
x25962 += x25840;
} else {
}

}
x25953 += x25933;
x25954 += x25841;
if (x25995) {
x25955 += 1;
} else {
}

}
x25946 += x25934;
x25947 += x25842;

}
float* x26005 = (float*)myMalloc(512 * sizeof(float));;
for(int x26006=0; x26006 < 512; x26006++) {
float x26007 = x121[x26006];
float x26008 = x26007 + 1.0E-5f;
x26005[x26006] = x26008;

}
float* x26012 = (float*)myMalloc(512 * sizeof(float));;
for(int x26013=0; x26013 < 512; x26013++) {
float x26014 = x26005[x26013];
double x26015 = (double)x26014;
double x26016 = sqrt(x26015);
float x26017 = (float)x26016;
x26012[x26013] = x26017;

}
int32_t x26021 = 0;
int32_t x26022 = 1;
x26022 *= 1;
x26021 += 1;
x26022 *= 1;
x26022 *= 1;
int32_t x26027 = x26021;
bool x26028 = x26027 >= 2;
if (x26028) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x26033 = x26027 == 0;
if (x26033) {
int32_t x26034 = x26022;
bool x26035 = x26034 == 512;
if (x26035) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x26042 = x26022;
int32_t x26043 = 512 / x26042;
bool x26049;
if (x452) {
bool x26044 = x25930 == 1;
bool x26045 = x26043 == 1;
bool x26046 = x26044 || x26045;
bool x26047 = x25930 == x26043;
bool x26048 = x26046 || x26047;
x26049 = x26048;
} else {
x26049 = false;
}
bool x26053;
if (x26049) {
x26053 = x26052;
} else {
x26053 = false;
}
bool x26054;
if (x26053) {
x26054 = x26052;
} else {
x26054 = false;
}
if (x26054) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x25930,x25932,x25932,1,x26043,1,1);
assert(false && "");
}
bool x26060 = x25930 <= x26043;
int32_t x26061;
if (x26060) {
x26061 = x26043;
} else {
x26061 = x25930;
}
bool x26067 = x26061 > 0;
bool x26069;
if (x26067) {
x26069 = x26068;
} else {
x26069 = false;
}
bool x26070;
if (x26069) {
x26070 = x26068;
} else {
x26070 = false;
}
if (x26070) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(25930) x Sym(25932) x Sym(25932)"," x Const(1) x Sym(26043) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x26065 = x26061 * x26064;
int32_t x26066 = 64 * x26065;
float* x26076 = (float*)myMalloc(x26066 * sizeof(float));;
int32_t x26077 = 0;
int32_t x26078 = 0;
int32_t x26079 = 0;
bool x26125 = x25930 > 1;
bool x26129 = x26043 > 1;
for(int x26080=0; x26080 < 64; x26080++) {
int32_t x26081 = x26078;
int32_t x26082 = x26079;
int32_t x26083 = x26077;
int32_t x26084 = x26083;
int32_t x26085 = x26081;
int32_t x26086 = x26082;
for(int x26088=0; x26088 < x26061; x26088++) {
int32_t x26089 = x26085;
int32_t x26090 = x26086;
int32_t x26091 = x26084;
int32_t x26092 = x26091;
int32_t x26093 = x26089;
int32_t x26094 = x26090;
for(int x26096=0; x26096 < x26063; x26096++) {
int32_t x26097 = x26093;
int32_t x26098 = x26094;
int32_t x26099 = x26092;
int32_t x26100 = x26099;
int32_t x26101 = x26097;
int32_t x26102 = x26098;
for(int x26103=0; x26103 < x26063; x26103++) {
int32_t x26104 = x26100;
int32_t x26105 = x26101;
float x26106 = x25945[x26105];
int32_t x26107 = x26102;
float x26108 = x26012[x26107];
float x26109 = x26106 / x26108;
x26076[x26104] = x26109;
x26100 += 1;
if (x26112) {
x26101 += 1;
} else {
}

}
x26092 += x26063;
if (x26112) {
x26093 += x25932;
} else {
}

}
x26084 += x26064;
if (x26125) {
x26085 += x25933;
} else {
}
if (x26129) {
x26086 += 1;
} else {
}

}
x26077 += x26065;
x26078 += x25934;

}
int32_t x26139 = 0;
int32_t x26140 = 1;
x26140 *= 1;
x26139 += 1;
x26140 *= 1;
x26140 *= 1;
int32_t x26145 = x26139;
bool x26146 = x26145 >= 2;
if (x26146) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x26151 = x26145 == 0;
if (x26151) {
int32_t x26152 = x26140;
bool x26153 = x26152 == 512;
if (x26153) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x26160 = x26140;
int32_t x26161 = 512 / x26160;
bool x26167;
if (x452) {
bool x26162 = x26061 == 1;
bool x26163 = x26161 == 1;
bool x26164 = x26162 || x26163;
bool x26165 = x26061 == x26161;
bool x26166 = x26164 || x26165;
x26167 = x26166;
} else {
x26167 = false;
}
bool x26171;
if (x26167) {
x26171 = x26170;
} else {
x26171 = false;
}
bool x26172;
if (x26171) {
x26172 = x26170;
} else {
x26172 = false;
}
if (x26172) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x26061,x26063,x26063,1,x26161,1,1);
assert(false && "");
}
bool x26178 = x26061 <= x26161;
int32_t x26179;
if (x26178) {
x26179 = x26161;
} else {
x26179 = x26061;
}
bool x26185 = x26179 > 0;
bool x26187;
if (x26185) {
x26187 = x26186;
} else {
x26187 = false;
}
bool x26188;
if (x26187) {
x26188 = x26186;
} else {
x26188 = false;
}
if (x26188) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(26061) x Sym(26063) x Sym(26063)"," x Const(1) x Sym(26161) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x26183 = x26179 * x26182;
int32_t x26184 = 64 * x26183;
float* x26194 = (float*)myMalloc(x26184 * sizeof(float));;
int32_t x26195 = 0;
int32_t x26196 = 0;
int32_t x26197 = 0;
bool x26243 = x26061 > 1;
bool x26247 = x26161 > 1;
for(int x26198=0; x26198 < 64; x26198++) {
int32_t x26199 = x26196;
int32_t x26200 = x26197;
int32_t x26201 = x26195;
int32_t x26202 = x26201;
int32_t x26203 = x26199;
int32_t x26204 = x26200;
for(int x26206=0; x26206 < x26179; x26206++) {
int32_t x26207 = x26203;
int32_t x26208 = x26204;
int32_t x26209 = x26202;
int32_t x26210 = x26209;
int32_t x26211 = x26207;
int32_t x26212 = x26208;
for(int x26214=0; x26214 < x26181; x26214++) {
int32_t x26215 = x26211;
int32_t x26216 = x26212;
int32_t x26217 = x26210;
int32_t x26218 = x26217;
int32_t x26219 = x26215;
int32_t x26220 = x26216;
for(int x26221=0; x26221 < x26181; x26221++) {
int32_t x26222 = x26218;
int32_t x26223 = x26219;
float x26224 = x26076[x26223];
int32_t x26225 = x26220;
float x26226 = x201[x26225];
float x26227 = x26224 * x26226;
x26194[x26222] = x26227;
x26218 += 1;
if (x26230) {
x26219 += 1;
} else {
}

}
x26210 += x26181;
if (x26230) {
x26211 += x26063;
} else {
}

}
x26202 += x26182;
if (x26243) {
x26203 += x26064;
} else {
}
if (x26247) {
x26204 += 1;
} else {
}

}
x26195 += x26183;
x26196 += x26065;

}
int32_t x26257 = 0;
int32_t x26258 = 1;
x26258 *= 1;
x26257 += 1;
x26258 *= 1;
x26258 *= 1;
int32_t x26263 = x26257;
bool x26264 = x26263 >= 2;
if (x26264) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x26269 = x26263 == 0;
if (x26269) {
int32_t x26270 = x26258;
bool x26271 = x26270 == 512;
if (x26271) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x26278 = x26258;
int32_t x26279 = 512 / x26278;
bool x26285;
if (x452) {
bool x26280 = x26179 == 1;
bool x26281 = x26279 == 1;
bool x26282 = x26280 || x26281;
bool x26283 = x26179 == x26279;
bool x26284 = x26282 || x26283;
x26285 = x26284;
} else {
x26285 = false;
}
bool x26289;
if (x26285) {
x26289 = x26288;
} else {
x26289 = false;
}
bool x26290;
if (x26289) {
x26290 = x26288;
} else {
x26290 = false;
}
if (x26290) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x26179,x26181,x26181,1,x26279,1,1);
assert(false && "");
}
bool x26296 = x26179 <= x26279;
int32_t x26297;
if (x26296) {
x26297 = x26279;
} else {
x26297 = x26179;
}
bool x26303 = x26297 > 0;
bool x26305;
if (x26303) {
x26305 = x26304;
} else {
x26305 = false;
}
bool x26306;
if (x26305) {
x26306 = x26304;
} else {
x26306 = false;
}
if (x26306) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(26179) x Sym(26181) x Sym(26181)"," x Const(1) x Sym(26279) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x26301 = x26297 * x26300;
int32_t x26302 = 64 * x26301;
float* x26312 = (float*)myMalloc(x26302 * sizeof(float));;
int32_t x26313 = 0;
int32_t x26314 = 0;
int32_t x26315 = 0;
bool x26361 = x26179 > 1;
bool x26365 = x26279 > 1;
for(int x26316=0; x26316 < 64; x26316++) {
int32_t x26317 = x26314;
int32_t x26318 = x26315;
int32_t x26319 = x26313;
int32_t x26320 = x26319;
int32_t x26321 = x26317;
int32_t x26322 = x26318;
for(int x26324=0; x26324 < x26297; x26324++) {
int32_t x26325 = x26321;
int32_t x26326 = x26322;
int32_t x26327 = x26320;
int32_t x26328 = x26327;
int32_t x26329 = x26325;
int32_t x26330 = x26326;
for(int x26332=0; x26332 < x26299; x26332++) {
int32_t x26333 = x26329;
int32_t x26334 = x26330;
int32_t x26335 = x26328;
int32_t x26336 = x26335;
int32_t x26337 = x26333;
int32_t x26338 = x26334;
for(int x26339=0; x26339 < x26299; x26339++) {
int32_t x26340 = x26336;
int32_t x26341 = x26337;
float x26342 = x26194[x26341];
int32_t x26343 = x26338;
float x26344 = x224[x26343];
float x26345 = x26342 + x26344;
x26312[x26340] = x26345;
x26336 += 1;
if (x26348) {
x26337 += 1;
} else {
}

}
x26328 += x26299;
if (x26348) {
x26329 += x26181;
} else {
}

}
x26320 += x26300;
if (x26361) {
x26321 += x26182;
} else {
}
if (x26365) {
x26322 += 1;
} else {
}

}
x26313 += x26301;
x26314 += x26183;

}
float* x26375 = (float*)myMalloc(x26302 * sizeof(float));;
for(int x26377=0; x26377 < x26302; x26377++) {
float x26378 = x26312[x26377];
bool x26379 = x26378 < 0.0f;
if (x26379) {
x26375[x26377] = 0.0f;
} else {
float x26382 = x26312[x26377];
x26375[x26377] = x26382;
}

}
float* x26397 = (float*)myMalloc(x26396 * sizeof(float));;
int32_t x26398 = 9 * x26297;
int32_t x26401 = 64 * x26398;
int32_t x26402 = x26401 * x26392;
float* x26403 = (float*)myMalloc(x26402 * sizeof(float));;
int32_t x26399 = x26398 * x26392;
int32_t x26411 = x26297 * 3;
int32_t x26412 = x26411 * 3;
for(int x26404=0; x26404 < 64; x26404++) {
int32_t x26405 = x26404 * x26301;
float* x26406 = x26375+x26405;
int32_t x26407 = x26404 * x26393;
float* x26408 = x26397+x26407;
int32_t x26409 = x26404 * x26399;
float* x26410 = x26403+x26409;
for(int x26414=0; x26414 < x26412; x26414++) {
int32_t x26415 = x26414 / 9;
int32_t x26419 = x26415 * 3;
int32_t x26420 = x26419 * 3;
int32_t x26421 = x26420 * x26391;
int32_t x26422 = x26421 * x26391;
int32_t x26416 = x26414 % 9;
int32_t x26417 = x26416 / 3;
int32_t x26423 = x26417 * 3;
int32_t x26424 = x26423 * x26391;
int32_t x26425 = x26424 * x26391;
int32_t x26426 = x26422 + x26425;
int32_t x26418 = x26416 % 3;
int32_t x26427 = x26418 * x26391;
int32_t x26428 = x26427 * x26391;
int32_t x26429 = x26426 + x26428;
float* x26430 = x26410+x26429;
int32_t x26431 = x26415 * x26299;
int32_t x26432 = x26431 * x26299;
float* x26433 = x26406+x26432;
for(int x26435=0; x26435 < x26391; x26435++) {
int32_t x26436 = x26435 * 2;
int32_t x26437 = x26436 - 1;
int32_t x26438 = x26437 + x26417;
bool x26439 = x26438 < 0;
bool x26440 = x26438 >= x26299;
bool x26441 = x26439 || x26440;
if (x26441) {
int32_t x26442 = x26435 * x26391;
float* x26443 = x26430+x26442;
memset(x26443, 0, 4 * x26391);;
} else {
int32_t x26442 = x26435 * x26391;
int32_t x26458 = x26438 * x26299;
for(int x26446=0; x26446 < x26391; x26446++) {
int32_t x26447 = x26446 * 2;
int32_t x26448 = x26447 - 1;
int32_t x26449 = x26448 + x26418;
bool x26450 = x26449 < 0;
bool x26451 = x26449 >= x26299;
bool x26452 = x26450 || x26451;
if (x26452) {
int32_t x26453 = x26442 + x26446;
float* x26454 = x26430+x26453;
memset(x26454, 0, 4 * 1);;
} else {
int32_t x26453 = x26442 + x26446;
float* x26457 = x26430+x26453;
int32_t x26459 = x26458 + x26449;
float* x26460 = x26433+x26459;
memcpy(x26457, x26460, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x26392,x26398,1,x34,x26398,x26410,x26392,1,x26408,x26392);

}
int32_t x26475 = 0;
int32_t x26476 = 1;
x26476 *= 1;
x26475 += 1;
x26476 *= 1;
x26476 *= 1;
int32_t x26481 = x26475;
bool x26482 = x26481 >= 2;
if (x26482) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x26487 = x26481 == 0;
if (x26487) {
int32_t x26488 = x26476;
bool x26489 = x26488 == 512;
if (x26489) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x26496 = x26476;
int32_t x26497 = 512 / x26496;
bool x26501;
if (x452) {
bool x26498 = x26497 == 1;
bool x26499 = 512 == x26497;
bool x26500 = x26498 || x26499;
x26501 = x26500;
} else {
x26501 = false;
}
bool x26505;
if (x26501) {
x26505 = x26504;
} else {
x26505 = false;
}
bool x26506;
if (x26505) {
x26506 = x26504;
} else {
x26506 = false;
}
if (x26506) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x26391,x26391,1,x26497,1,1);
assert(false && "");
}
bool x26512 = 512 <= x26497;
int32_t x26513;
if (x26512) {
x26513 = x26497;
} else {
x26513 = 512;
}
bool x26519 = x26513 > 0;
bool x26521;
if (x26519) {
x26521 = x26520;
} else {
x26521 = false;
}
bool x26522;
if (x26521) {
x26522 = x26520;
} else {
x26522 = false;
}
if (x26522) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(26391) x Sym(26391)"," x Const(1) x Sym(26497) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x26517 = x26513 * x26516;
int32_t x26518 = 64 * x26517;
float* x26528 = (float*)myMalloc(x26518 * sizeof(float));;
int32_t x26529 = 0;
int32_t x26530 = 0;
int32_t x26531 = 0;
bool x26578 = x26497 > 1;
for(int x26532=0; x26532 < 64; x26532++) {
int32_t x26533 = x26530;
int32_t x26534 = x26531;
int32_t x26535 = x26529;
int32_t x26536 = x26535;
int32_t x26537 = x26533;
int32_t x26538 = x26534;
for(int x26540=0; x26540 < x26513; x26540++) {
int32_t x26541 = x26537;
int32_t x26542 = x26538;
int32_t x26543 = x26536;
int32_t x26544 = x26543;
int32_t x26545 = x26541;
int32_t x26546 = x26542;
for(int x26548=0; x26548 < x26515; x26548++) {
int32_t x26549 = x26545;
int32_t x26550 = x26546;
int32_t x26551 = x26544;
int32_t x26552 = x26551;
int32_t x26553 = x26549;
int32_t x26554 = x26550;
for(int x26555=0; x26555 < x26515; x26555++) {
int32_t x26556 = x26552;
int32_t x26557 = x26553;
float x26558 = x26397[x26557];
int32_t x26559 = x26554;
float x26560 = x113[x26559];
float x26561 = x26558 - x26560;
x26528[x26556] = x26561;
x26552 += 1;
if (x26564) {
x26553 += 1;
} else {
}

}
x26544 += x26515;
if (x26564) {
x26545 += x26391;
} else {
}

}
x26536 += x26516;
x26537 += x26392;
if (x26578) {
x26538 += 1;
} else {
}

}
x26529 += x26517;
x26530 += x26393;

}
float* x26588 = (float*)myMalloc(512 * sizeof(float));;
for(int x26589=0; x26589 < 512; x26589++) {
float x26590 = x50[x26589];
float x26591 = x26590 + 1.0E-5f;
x26588[x26589] = x26591;

}
float* x26595 = (float*)myMalloc(512 * sizeof(float));;
for(int x26596=0; x26596 < 512; x26596++) {
float x26597 = x26588[x26596];
double x26598 = (double)x26597;
double x26599 = sqrt(x26598);
float x26600 = (float)x26599;
x26595[x26596] = x26600;

}
int32_t x26604 = 0;
int32_t x26605 = 1;
x26605 *= 1;
x26604 += 1;
x26605 *= 1;
x26605 *= 1;
int32_t x26610 = x26604;
bool x26611 = x26610 >= 2;
if (x26611) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x26616 = x26610 == 0;
if (x26616) {
int32_t x26617 = x26605;
bool x26618 = x26617 == 512;
if (x26618) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x26625 = x26605;
int32_t x26626 = 512 / x26625;
bool x26632;
if (x452) {
bool x26627 = x26513 == 1;
bool x26628 = x26626 == 1;
bool x26629 = x26627 || x26628;
bool x26630 = x26513 == x26626;
bool x26631 = x26629 || x26630;
x26632 = x26631;
} else {
x26632 = false;
}
bool x26636;
if (x26632) {
x26636 = x26635;
} else {
x26636 = false;
}
bool x26637;
if (x26636) {
x26637 = x26635;
} else {
x26637 = false;
}
if (x26637) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x26513,x26515,x26515,1,x26626,1,1);
assert(false && "");
}
bool x26643 = x26513 <= x26626;
int32_t x26644;
if (x26643) {
x26644 = x26626;
} else {
x26644 = x26513;
}
bool x26650 = x26644 > 0;
bool x26652;
if (x26650) {
x26652 = x26651;
} else {
x26652 = false;
}
bool x26653;
if (x26652) {
x26653 = x26651;
} else {
x26653 = false;
}
if (x26653) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(26513) x Sym(26515) x Sym(26515)"," x Const(1) x Sym(26626) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x26648 = x26644 * x26647;
int32_t x26649 = 64 * x26648;
float* x26659 = (float*)myMalloc(x26649 * sizeof(float));;
int32_t x26660 = 0;
int32_t x26661 = 0;
int32_t x26662 = 0;
bool x26708 = x26513 > 1;
bool x26712 = x26626 > 1;
for(int x26663=0; x26663 < 64; x26663++) {
int32_t x26664 = x26661;
int32_t x26665 = x26662;
int32_t x26666 = x26660;
int32_t x26667 = x26666;
int32_t x26668 = x26664;
int32_t x26669 = x26665;
for(int x26671=0; x26671 < x26644; x26671++) {
int32_t x26672 = x26668;
int32_t x26673 = x26669;
int32_t x26674 = x26667;
int32_t x26675 = x26674;
int32_t x26676 = x26672;
int32_t x26677 = x26673;
for(int x26679=0; x26679 < x26646; x26679++) {
int32_t x26680 = x26676;
int32_t x26681 = x26677;
int32_t x26682 = x26675;
int32_t x26683 = x26682;
int32_t x26684 = x26680;
int32_t x26685 = x26681;
for(int x26686=0; x26686 < x26646; x26686++) {
int32_t x26687 = x26683;
int32_t x26688 = x26684;
float x26689 = x26528[x26688];
int32_t x26690 = x26685;
float x26691 = x26595[x26690];
float x26692 = x26689 / x26691;
x26659[x26687] = x26692;
x26683 += 1;
if (x26695) {
x26684 += 1;
} else {
}

}
x26675 += x26646;
if (x26695) {
x26676 += x26515;
} else {
}

}
x26667 += x26647;
if (x26708) {
x26668 += x26516;
} else {
}
if (x26712) {
x26669 += 1;
} else {
}

}
x26660 += x26648;
x26661 += x26517;

}
int32_t x26722 = 0;
int32_t x26723 = 1;
x26723 *= 1;
x26722 += 1;
x26723 *= 1;
x26723 *= 1;
int32_t x26728 = x26722;
bool x26729 = x26728 >= 2;
if (x26729) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x26734 = x26728 == 0;
if (x26734) {
int32_t x26735 = x26723;
bool x26736 = x26735 == 512;
if (x26736) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x26743 = x26723;
int32_t x26744 = 512 / x26743;
bool x26750;
if (x452) {
bool x26745 = x26644 == 1;
bool x26746 = x26744 == 1;
bool x26747 = x26745 || x26746;
bool x26748 = x26644 == x26744;
bool x26749 = x26747 || x26748;
x26750 = x26749;
} else {
x26750 = false;
}
bool x26754;
if (x26750) {
x26754 = x26753;
} else {
x26754 = false;
}
bool x26755;
if (x26754) {
x26755 = x26753;
} else {
x26755 = false;
}
if (x26755) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x26644,x26646,x26646,1,x26744,1,1);
assert(false && "");
}
bool x26761 = x26644 <= x26744;
int32_t x26762;
if (x26761) {
x26762 = x26744;
} else {
x26762 = x26644;
}
bool x26768 = x26762 > 0;
bool x26770;
if (x26768) {
x26770 = x26769;
} else {
x26770 = false;
}
bool x26771;
if (x26770) {
x26771 = x26769;
} else {
x26771 = false;
}
if (x26771) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(26644) x Sym(26646) x Sym(26646)"," x Const(1) x Sym(26744) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x26766 = x26762 * x26765;
int32_t x26767 = 64 * x26766;
float* x26777 = (float*)myMalloc(x26767 * sizeof(float));;
int32_t x26778 = 0;
int32_t x26779 = 0;
int32_t x26780 = 0;
bool x26826 = x26644 > 1;
bool x26830 = x26744 > 1;
for(int x26781=0; x26781 < 64; x26781++) {
int32_t x26782 = x26779;
int32_t x26783 = x26780;
int32_t x26784 = x26778;
int32_t x26785 = x26784;
int32_t x26786 = x26782;
int32_t x26787 = x26783;
for(int x26789=0; x26789 < x26762; x26789++) {
int32_t x26790 = x26786;
int32_t x26791 = x26787;
int32_t x26792 = x26785;
int32_t x26793 = x26792;
int32_t x26794 = x26790;
int32_t x26795 = x26791;
for(int x26797=0; x26797 < x26764; x26797++) {
int32_t x26798 = x26794;
int32_t x26799 = x26795;
int32_t x26800 = x26793;
int32_t x26801 = x26800;
int32_t x26802 = x26798;
int32_t x26803 = x26799;
for(int x26804=0; x26804 < x26764; x26804++) {
int32_t x26805 = x26801;
int32_t x26806 = x26802;
float x26807 = x26659[x26806];
int32_t x26808 = x26803;
float x26809 = x205[x26808];
float x26810 = x26807 * x26809;
x26777[x26805] = x26810;
x26801 += 1;
if (x26813) {
x26802 += 1;
} else {
}

}
x26793 += x26764;
if (x26813) {
x26794 += x26646;
} else {
}

}
x26785 += x26765;
if (x26826) {
x26786 += x26647;
} else {
}
if (x26830) {
x26787 += 1;
} else {
}

}
x26778 += x26766;
x26779 += x26648;

}
int32_t x26840 = 0;
int32_t x26841 = 1;
x26841 *= 1;
x26840 += 1;
x26841 *= 1;
x26841 *= 1;
int32_t x26846 = x26840;
bool x26847 = x26846 >= 2;
if (x26847) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x26852 = x26846 == 0;
if (x26852) {
int32_t x26853 = x26841;
bool x26854 = x26853 == 512;
if (x26854) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x26861 = x26841;
int32_t x26862 = 512 / x26861;
bool x26868;
if (x452) {
bool x26863 = x26762 == 1;
bool x26864 = x26862 == 1;
bool x26865 = x26863 || x26864;
bool x26866 = x26762 == x26862;
bool x26867 = x26865 || x26866;
x26868 = x26867;
} else {
x26868 = false;
}
bool x26872;
if (x26868) {
x26872 = x26871;
} else {
x26872 = false;
}
bool x26873;
if (x26872) {
x26873 = x26871;
} else {
x26873 = false;
}
if (x26873) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x26762,x26764,x26764,1,x26862,1,1);
assert(false && "");
}
bool x26879 = x26762 <= x26862;
int32_t x26880;
if (x26879) {
x26880 = x26862;
} else {
x26880 = x26762;
}
bool x26886 = x26880 > 0;
bool x26888;
if (x26886) {
x26888 = x26887;
} else {
x26888 = false;
}
bool x26889;
if (x26888) {
x26889 = x26887;
} else {
x26889 = false;
}
if (x26889) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(26762) x Sym(26764) x Sym(26764)"," x Const(1) x Sym(26862) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x26884 = x26880 * x26883;
int32_t x26885 = 64 * x26884;
float* x26895 = (float*)myMalloc(x26885 * sizeof(float));;
int32_t x26896 = 0;
int32_t x26897 = 0;
int32_t x26898 = 0;
bool x26944 = x26762 > 1;
bool x26948 = x26862 > 1;
for(int x26899=0; x26899 < 64; x26899++) {
int32_t x26900 = x26897;
int32_t x26901 = x26898;
int32_t x26902 = x26896;
int32_t x26903 = x26902;
int32_t x26904 = x26900;
int32_t x26905 = x26901;
for(int x26907=0; x26907 < x26880; x26907++) {
int32_t x26908 = x26904;
int32_t x26909 = x26905;
int32_t x26910 = x26903;
int32_t x26911 = x26910;
int32_t x26912 = x26908;
int32_t x26913 = x26909;
for(int x26915=0; x26915 < x26882; x26915++) {
int32_t x26916 = x26912;
int32_t x26917 = x26913;
int32_t x26918 = x26911;
int32_t x26919 = x26918;
int32_t x26920 = x26916;
int32_t x26921 = x26917;
for(int x26922=0; x26922 < x26882; x26922++) {
int32_t x26923 = x26919;
int32_t x26924 = x26920;
float x26925 = x26777[x26924];
int32_t x26926 = x26921;
float x26927 = x159[x26926];
float x26928 = x26925 + x26927;
x26895[x26923] = x26928;
x26919 += 1;
if (x26931) {
x26920 += 1;
} else {
}

}
x26911 += x26882;
if (x26931) {
x26912 += x26764;
} else {
}

}
x26903 += x26883;
if (x26944) {
x26904 += x26765;
} else {
}
if (x26948) {
x26905 += 1;
} else {
}

}
x26896 += x26884;
x26897 += x26766;

}
float* x26958 = (float*)myMalloc(x26885 * sizeof(float));;
for(int x26960=0; x26960 < x26885; x26960++) {
float x26961 = x26895[x26960];
bool x26962 = x26961 < 0.0f;
if (x26962) {
x26958[x26960] = 0.0f;
} else {
float x26965 = x26895[x26960];
x26958[x26960] = x26965;
}

}
float* x26979 = (float*)myMalloc(x26978 * sizeof(float));;
int32_t x26982 = 64 * x26880;
int32_t x26983 = x26982 * x26974;
float* x26984 = (float*)myMalloc(x26983 * sizeof(float));;
int32_t x26980 = x26880 * x26974;
for(int x26985=0; x26985 < 64; x26985++) {
int32_t x26986 = x26985 * x26884;
float* x26987 = x26958+x26986;
int32_t x26988 = x26985 * x26975;
float* x26989 = x26979+x26988;
int32_t x26990 = x26985 * x26980;
float* x26991 = x26984+x26990;
for(int x26992=0; x26992 < x26880; x26992++) {
int32_t x26993 = x26992 / 1;
int32_t x26997 = x26993 * x26973;
int32_t x26998 = x26997 * x26973;
int32_t x26994 = x26992 % 1;
int32_t x26995 = x26994 / 1;
int32_t x26999 = x26995 * x26973;
int32_t x27000 = x26999 * x26973;
int32_t x27001 = x26998 + x27000;
int32_t x26996 = x26994 % 1;
int32_t x27002 = x26996 * x26973;
int32_t x27003 = x27002 * x26973;
int32_t x27004 = x27001 + x27003;
float* x27005 = x26991+x27004;
int32_t x27006 = x26993 * x26882;
int32_t x27007 = x27006 * x26882;
float* x27008 = x26987+x27007;
for(int x27010=0; x27010 < x26973; x27010++) {
int32_t x27012 = x27010 * x26973;
float* x27013 = x27005+x27012;
int32_t x27011 = x27010 + x26995;
int32_t x27014 = x27011 * x26882;
int32_t x27015 = x27014 + x26996;
float* x27016 = x27008+x27015;
memcpy(x27013, x27016, 4 * x26973);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x26974,x26880,1,x212,x26880,x26991,x26974,1,x26989,x26974);

}
int32_t x27025 = 0;
int32_t x27026 = 1;
x27026 *= 1;
x27025 += 1;
x27026 *= 1;
x27026 *= 1;
int32_t x27031 = x27025;
bool x27032 = x27031 >= 2;
if (x27032) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x27037 = x27031 == 0;
if (x27037) {
int32_t x27038 = x27026;
bool x27039 = x27038 == 2048;
if (x27039) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x27046 = x27026;
int32_t x27047 = 2048 / x27046;
bool x27051;
if (x452) {
bool x27048 = x27047 == 1;
bool x27049 = 2048 == x27047;
bool x27050 = x27048 || x27049;
x27051 = x27050;
} else {
x27051 = false;
}
bool x27055;
if (x27051) {
x27055 = x27054;
} else {
x27055 = false;
}
bool x27056;
if (x27055) {
x27056 = x27054;
} else {
x27056 = false;
}
if (x27056) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x26973,x26973,1,x27047,1,1);
assert(false && "");
}
bool x27062 = 2048 <= x27047;
int32_t x27063;
if (x27062) {
x27063 = x27047;
} else {
x27063 = 2048;
}
bool x27069 = x27063 > 0;
bool x27071;
if (x27069) {
x27071 = x27070;
} else {
x27071 = false;
}
bool x27072;
if (x27071) {
x27072 = x27070;
} else {
x27072 = false;
}
if (x27072) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(2048) x Sym(26973) x Sym(26973)"," x Const(1) x Sym(27047) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x27067 = x27063 * x27066;
int32_t x27068 = 64 * x27067;
float* x27078 = (float*)myMalloc(x27068 * sizeof(float));;
int32_t x27079 = 0;
int32_t x27080 = 0;
int32_t x27081 = 0;
bool x27128 = x27047 > 1;
for(int x27082=0; x27082 < 64; x27082++) {
int32_t x27083 = x27080;
int32_t x27084 = x27081;
int32_t x27085 = x27079;
int32_t x27086 = x27085;
int32_t x27087 = x27083;
int32_t x27088 = x27084;
for(int x27090=0; x27090 < x27063; x27090++) {
int32_t x27091 = x27087;
int32_t x27092 = x27088;
int32_t x27093 = x27086;
int32_t x27094 = x27093;
int32_t x27095 = x27091;
int32_t x27096 = x27092;
for(int x27098=0; x27098 < x27065; x27098++) {
int32_t x27099 = x27095;
int32_t x27100 = x27096;
int32_t x27101 = x27094;
int32_t x27102 = x27101;
int32_t x27103 = x27099;
int32_t x27104 = x27100;
for(int x27105=0; x27105 < x27065; x27105++) {
int32_t x27106 = x27102;
int32_t x27107 = x27103;
float x27108 = x26979[x27107];
int32_t x27109 = x27104;
float x27110 = x115[x27109];
float x27111 = x27108 - x27110;
x27078[x27106] = x27111;
x27102 += 1;
if (x27114) {
x27103 += 1;
} else {
}

}
x27094 += x27065;
if (x27114) {
x27095 += x26973;
} else {
}

}
x27086 += x27066;
x27087 += x26974;
if (x27128) {
x27088 += 1;
} else {
}

}
x27079 += x27067;
x27080 += x26975;

}
float* x27138 = (float*)myMalloc(2048 * sizeof(float));;
for(int x27140=0; x27140 < 2048; x27140++) {
float x27141 = x193[x27140];
float x27142 = x27141 + 1.0E-5f;
x27138[x27140] = x27142;

}
float* x27146 = (float*)myMalloc(2048 * sizeof(float));;
for(int x27147=0; x27147 < 2048; x27147++) {
float x27148 = x27138[x27147];
double x27149 = (double)x27148;
double x27150 = sqrt(x27149);
float x27151 = (float)x27150;
x27146[x27147] = x27151;

}
int32_t x27155 = 0;
int32_t x27156 = 1;
x27156 *= 1;
x27155 += 1;
x27156 *= 1;
x27156 *= 1;
int32_t x27161 = x27155;
bool x27162 = x27161 >= 2;
if (x27162) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x27167 = x27161 == 0;
if (x27167) {
int32_t x27168 = x27156;
bool x27169 = x27168 == 2048;
if (x27169) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x27176 = x27156;
int32_t x27177 = 2048 / x27176;
bool x27183;
if (x452) {
bool x27178 = x27063 == 1;
bool x27179 = x27177 == 1;
bool x27180 = x27178 || x27179;
bool x27181 = x27063 == x27177;
bool x27182 = x27180 || x27181;
x27183 = x27182;
} else {
x27183 = false;
}
bool x27187;
if (x27183) {
x27187 = x27186;
} else {
x27187 = false;
}
bool x27188;
if (x27187) {
x27188 = x27186;
} else {
x27188 = false;
}
if (x27188) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x27063,x27065,x27065,1,x27177,1,1);
assert(false && "");
}
bool x27194 = x27063 <= x27177;
int32_t x27195;
if (x27194) {
x27195 = x27177;
} else {
x27195 = x27063;
}
bool x27201 = x27195 > 0;
bool x27203;
if (x27201) {
x27203 = x27202;
} else {
x27203 = false;
}
bool x27204;
if (x27203) {
x27204 = x27202;
} else {
x27204 = false;
}
if (x27204) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(27063) x Sym(27065) x Sym(27065)"," x Const(1) x Sym(27177) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x27199 = x27195 * x27198;
int32_t x27200 = 64 * x27199;
float* x27210 = (float*)myMalloc(x27200 * sizeof(float));;
int32_t x27211 = 0;
int32_t x27212 = 0;
int32_t x27213 = 0;
bool x27259 = x27063 > 1;
bool x27263 = x27177 > 1;
for(int x27214=0; x27214 < 64; x27214++) {
int32_t x27215 = x27212;
int32_t x27216 = x27213;
int32_t x27217 = x27211;
int32_t x27218 = x27217;
int32_t x27219 = x27215;
int32_t x27220 = x27216;
for(int x27222=0; x27222 < x27195; x27222++) {
int32_t x27223 = x27219;
int32_t x27224 = x27220;
int32_t x27225 = x27218;
int32_t x27226 = x27225;
int32_t x27227 = x27223;
int32_t x27228 = x27224;
for(int x27230=0; x27230 < x27197; x27230++) {
int32_t x27231 = x27227;
int32_t x27232 = x27228;
int32_t x27233 = x27226;
int32_t x27234 = x27233;
int32_t x27235 = x27231;
int32_t x27236 = x27232;
for(int x27237=0; x27237 < x27197; x27237++) {
int32_t x27238 = x27234;
int32_t x27239 = x27235;
float x27240 = x27078[x27239];
int32_t x27241 = x27236;
float x27242 = x27146[x27241];
float x27243 = x27240 / x27242;
x27210[x27238] = x27243;
x27234 += 1;
if (x27246) {
x27235 += 1;
} else {
}

}
x27226 += x27197;
if (x27246) {
x27227 += x27065;
} else {
}

}
x27218 += x27198;
if (x27259) {
x27219 += x27066;
} else {
}
if (x27263) {
x27220 += 1;
} else {
}

}
x27211 += x27199;
x27212 += x27067;

}
int32_t x27273 = 0;
int32_t x27274 = 1;
x27274 *= 1;
x27273 += 1;
x27274 *= 1;
x27274 *= 1;
int32_t x27279 = x27273;
bool x27280 = x27279 >= 2;
if (x27280) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x27285 = x27279 == 0;
if (x27285) {
int32_t x27286 = x27274;
bool x27287 = x27286 == 2048;
if (x27287) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x27294 = x27274;
int32_t x27295 = 2048 / x27294;
bool x27301;
if (x452) {
bool x27296 = x27195 == 1;
bool x27297 = x27295 == 1;
bool x27298 = x27296 || x27297;
bool x27299 = x27195 == x27295;
bool x27300 = x27298 || x27299;
x27301 = x27300;
} else {
x27301 = false;
}
bool x27305;
if (x27301) {
x27305 = x27304;
} else {
x27305 = false;
}
bool x27306;
if (x27305) {
x27306 = x27304;
} else {
x27306 = false;
}
if (x27306) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x27195,x27197,x27197,1,x27295,1,1);
assert(false && "");
}
bool x27312 = x27195 <= x27295;
int32_t x27313;
if (x27312) {
x27313 = x27295;
} else {
x27313 = x27195;
}
bool x27319 = x27313 > 0;
bool x27321;
if (x27319) {
x27321 = x27320;
} else {
x27321 = false;
}
bool x27322;
if (x27321) {
x27322 = x27320;
} else {
x27322 = false;
}
if (x27322) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(27195) x Sym(27197) x Sym(27197)"," x Const(1) x Sym(27295) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x27317 = x27313 * x27316;
int32_t x27318 = 64 * x27317;
float* x27328 = (float*)myMalloc(x27318 * sizeof(float));;
int32_t x27329 = 0;
int32_t x27330 = 0;
int32_t x27331 = 0;
bool x27377 = x27195 > 1;
bool x27381 = x27295 > 1;
for(int x27332=0; x27332 < 64; x27332++) {
int32_t x27333 = x27330;
int32_t x27334 = x27331;
int32_t x27335 = x27329;
int32_t x27336 = x27335;
int32_t x27337 = x27333;
int32_t x27338 = x27334;
for(int x27340=0; x27340 < x27313; x27340++) {
int32_t x27341 = x27337;
int32_t x27342 = x27338;
int32_t x27343 = x27336;
int32_t x27344 = x27343;
int32_t x27345 = x27341;
int32_t x27346 = x27342;
for(int x27348=0; x27348 < x27315; x27348++) {
int32_t x27349 = x27345;
int32_t x27350 = x27346;
int32_t x27351 = x27344;
int32_t x27352 = x27351;
int32_t x27353 = x27349;
int32_t x27354 = x27350;
for(int x27355=0; x27355 < x27315; x27355++) {
int32_t x27356 = x27352;
int32_t x27357 = x27353;
float x27358 = x27210[x27357];
int32_t x27359 = x27354;
float x27360 = x239[x27359];
float x27361 = x27358 * x27360;
x27328[x27356] = x27361;
x27352 += 1;
if (x27364) {
x27353 += 1;
} else {
}

}
x27344 += x27315;
if (x27364) {
x27345 += x27197;
} else {
}

}
x27336 += x27316;
if (x27377) {
x27337 += x27198;
} else {
}
if (x27381) {
x27338 += 1;
} else {
}

}
x27329 += x27317;
x27330 += x27199;

}
int32_t x27391 = 0;
int32_t x27392 = 1;
x27392 *= 1;
x27391 += 1;
x27392 *= 1;
x27392 *= 1;
int32_t x27397 = x27391;
bool x27398 = x27397 >= 2;
if (x27398) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x27403 = x27397 == 0;
if (x27403) {
int32_t x27404 = x27392;
bool x27405 = x27404 == 2048;
if (x27405) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x27412 = x27392;
int32_t x27413 = 2048 / x27412;
bool x27419;
if (x452) {
bool x27414 = x27313 == 1;
bool x27415 = x27413 == 1;
bool x27416 = x27414 || x27415;
bool x27417 = x27313 == x27413;
bool x27418 = x27416 || x27417;
x27419 = x27418;
} else {
x27419 = false;
}
bool x27423;
if (x27419) {
x27423 = x27422;
} else {
x27423 = false;
}
bool x27424;
if (x27423) {
x27424 = x27422;
} else {
x27424 = false;
}
if (x27424) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x27313,x27315,x27315,1,x27413,1,1);
assert(false && "");
}
bool x27430 = x27313 <= x27413;
int32_t x27431;
if (x27430) {
x27431 = x27413;
} else {
x27431 = x27313;
}
bool x27437 = x27431 > 0;
bool x27439;
if (x27437) {
x27439 = x27438;
} else {
x27439 = false;
}
bool x27440;
if (x27439) {
x27440 = x27438;
} else {
x27440 = false;
}
if (x27440) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(27313) x Sym(27315) x Sym(27315)"," x Const(1) x Sym(27413) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x27435 = x27431 * x27434;
int32_t x27436 = 64 * x27435;
float* x27446 = (float*)myMalloc(x27436 * sizeof(float));;
int32_t x27447 = 0;
int32_t x27448 = 0;
int32_t x27449 = 0;
bool x27495 = x27313 > 1;
bool x27499 = x27413 > 1;
for(int x27450=0; x27450 < 64; x27450++) {
int32_t x27451 = x27448;
int32_t x27452 = x27449;
int32_t x27453 = x27447;
int32_t x27454 = x27453;
int32_t x27455 = x27451;
int32_t x27456 = x27452;
for(int x27458=0; x27458 < x27431; x27458++) {
int32_t x27459 = x27455;
int32_t x27460 = x27456;
int32_t x27461 = x27454;
int32_t x27462 = x27461;
int32_t x27463 = x27459;
int32_t x27464 = x27460;
for(int x27466=0; x27466 < x27433; x27466++) {
int32_t x27467 = x27463;
int32_t x27468 = x27464;
int32_t x27469 = x27462;
int32_t x27470 = x27469;
int32_t x27471 = x27467;
int32_t x27472 = x27468;
for(int x27473=0; x27473 < x27433; x27473++) {
int32_t x27474 = x27470;
int32_t x27475 = x27471;
float x27476 = x27328[x27475];
int32_t x27477 = x27472;
float x27478 = x62[x27477];
float x27479 = x27476 + x27478;
x27446[x27474] = x27479;
x27470 += 1;
if (x27482) {
x27471 += 1;
} else {
}

}
x27462 += x27433;
if (x27482) {
x27463 += x27315;
} else {
}

}
x27454 += x27434;
if (x27495) {
x27455 += x27316;
} else {
}
if (x27499) {
x27456 += 1;
} else {
}

}
x27447 += x27435;
x27448 += x27317;

}
float* x27516 = (float*)myMalloc(x27515 * sizeof(float));;
int32_t x27519 = x25849 * x27511;
float* x27520 = (float*)myMalloc(x27519 * sizeof(float));;
int32_t x27517 = x25658 * x27511;
for(int x27521=0; x27521 < 64; x27521++) {
int32_t x27522 = x27521 * x25662;
float* x27523 = x25825+x27522;
int32_t x27524 = x27521 * x27512;
float* x27525 = x27516+x27524;
int32_t x27526 = x27521 * x27517;
float* x27527 = x27520+x27526;
for(int x27528=0; x27528 < x25658; x27528++) {
int32_t x27529 = x27528 / 1;
int32_t x27533 = x27529 * x27510;
int32_t x27534 = x27533 * x27510;
int32_t x27530 = x27528 % 1;
int32_t x27531 = x27530 / 1;
int32_t x27535 = x27531 * x27510;
int32_t x27536 = x27535 * x27510;
int32_t x27537 = x27534 + x27536;
int32_t x27532 = x27530 % 1;
int32_t x27538 = x27532 * x27510;
int32_t x27539 = x27538 * x27510;
int32_t x27540 = x27537 + x27539;
float* x27541 = x27527+x27540;
int32_t x27542 = x27529 * x25660;
int32_t x27543 = x27542 * x25660;
float* x27544 = x27523+x27543;
for(int x27546=0; x27546 < x27510; x27546++) {
int32_t x27550 = x27546 * x27510;
int32_t x27547 = x27546 * 2;
int32_t x27548 = x27547 + x27531;
int32_t x27553 = x27548 * x25660;
int32_t x27554 = x27553 + x27532;
for(int x27549=0; x27549 < x27510; x27549++) {
int32_t x27551 = x27550 + x27549;
float* x27552 = x27541+x27551;
int32_t x27555 = x27549 * 2;
int32_t x27556 = x27554 + x27555;
float* x27557 = x27544+x27556;
memcpy(x27552, x27557, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x27511,x25658,1,x214,x25658,x27527,x27511,1,x27525,x27511);

}
int32_t x27568 = 0;
int32_t x27569 = 1;
x27569 *= 1;
x27568 += 1;
x27569 *= 1;
x27569 *= 1;
int32_t x27574 = x27568;
bool x27575 = x27574 >= 2;
if (x27575) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x27580 = x27574 == 0;
if (x27580) {
int32_t x27581 = x27569;
bool x27582 = x27581 == 2048;
if (x27582) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x27589 = x27569;
int32_t x27590 = 2048 / x27589;
bool x27594;
if (x452) {
bool x27591 = x27590 == 1;
bool x27592 = 2048 == x27590;
bool x27593 = x27591 || x27592;
x27594 = x27593;
} else {
x27594 = false;
}
bool x27598;
if (x27594) {
x27598 = x27597;
} else {
x27598 = false;
}
bool x27599;
if (x27598) {
x27599 = x27597;
} else {
x27599 = false;
}
if (x27599) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x27510,x27510,1,x27590,1,1);
assert(false && "");
}
bool x27605 = 2048 <= x27590;
int32_t x27606;
if (x27605) {
x27606 = x27590;
} else {
x27606 = 2048;
}
bool x27612 = x27606 > 0;
bool x27614;
if (x27612) {
x27614 = x27613;
} else {
x27614 = false;
}
bool x27615;
if (x27614) {
x27615 = x27613;
} else {
x27615 = false;
}
if (x27615) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(2048) x Sym(27510) x Sym(27510)"," x Const(1) x Sym(27590) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x27610 = x27606 * x27609;
int32_t x27611 = 64 * x27610;
float* x27621 = (float*)myMalloc(x27611 * sizeof(float));;
int32_t x27622 = 0;
int32_t x27623 = 0;
int32_t x27624 = 0;
bool x27671 = x27590 > 1;
for(int x27625=0; x27625 < 64; x27625++) {
int32_t x27626 = x27623;
int32_t x27627 = x27624;
int32_t x27628 = x27622;
int32_t x27629 = x27628;
int32_t x27630 = x27626;
int32_t x27631 = x27627;
for(int x27633=0; x27633 < x27606; x27633++) {
int32_t x27634 = x27630;
int32_t x27635 = x27631;
int32_t x27636 = x27629;
int32_t x27637 = x27636;
int32_t x27638 = x27634;
int32_t x27639 = x27635;
for(int x27641=0; x27641 < x27608; x27641++) {
int32_t x27642 = x27638;
int32_t x27643 = x27639;
int32_t x27644 = x27637;
int32_t x27645 = x27644;
int32_t x27646 = x27642;
int32_t x27647 = x27643;
for(int x27648=0; x27648 < x27608; x27648++) {
int32_t x27649 = x27645;
int32_t x27650 = x27646;
float x27651 = x27516[x27650];
int32_t x27652 = x27647;
float x27653 = x64[x27652];
float x27654 = x27651 - x27653;
x27621[x27649] = x27654;
x27645 += 1;
if (x27657) {
x27646 += 1;
} else {
}

}
x27637 += x27608;
if (x27657) {
x27638 += x27510;
} else {
}

}
x27629 += x27609;
x27630 += x27511;
if (x27671) {
x27631 += 1;
} else {
}

}
x27622 += x27610;
x27623 += x27512;

}
float* x27681 = (float*)myMalloc(2048 * sizeof(float));;
for(int x27682=0; x27682 < 2048; x27682++) {
float x27683 = x125[x27682];
float x27684 = x27683 + 1.0E-5f;
x27681[x27682] = x27684;

}
float* x27688 = (float*)myMalloc(2048 * sizeof(float));;
for(int x27689=0; x27689 < 2048; x27689++) {
float x27690 = x27681[x27689];
double x27691 = (double)x27690;
double x27692 = sqrt(x27691);
float x27693 = (float)x27692;
x27688[x27689] = x27693;

}
int32_t x27697 = 0;
int32_t x27698 = 1;
x27698 *= 1;
x27697 += 1;
x27698 *= 1;
x27698 *= 1;
int32_t x27703 = x27697;
bool x27704 = x27703 >= 2;
if (x27704) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x27709 = x27703 == 0;
if (x27709) {
int32_t x27710 = x27698;
bool x27711 = x27710 == 2048;
if (x27711) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x27718 = x27698;
int32_t x27719 = 2048 / x27718;
bool x27725;
if (x452) {
bool x27720 = x27606 == 1;
bool x27721 = x27719 == 1;
bool x27722 = x27720 || x27721;
bool x27723 = x27606 == x27719;
bool x27724 = x27722 || x27723;
x27725 = x27724;
} else {
x27725 = false;
}
bool x27729;
if (x27725) {
x27729 = x27728;
} else {
x27729 = false;
}
bool x27730;
if (x27729) {
x27730 = x27728;
} else {
x27730 = false;
}
if (x27730) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x27606,x27608,x27608,1,x27719,1,1);
assert(false && "");
}
bool x27736 = x27606 <= x27719;
int32_t x27737;
if (x27736) {
x27737 = x27719;
} else {
x27737 = x27606;
}
bool x27743 = x27737 > 0;
bool x27745;
if (x27743) {
x27745 = x27744;
} else {
x27745 = false;
}
bool x27746;
if (x27745) {
x27746 = x27744;
} else {
x27746 = false;
}
if (x27746) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(27606) x Sym(27608) x Sym(27608)"," x Const(1) x Sym(27719) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x27741 = x27737 * x27740;
int32_t x27742 = 64 * x27741;
float* x27752 = (float*)myMalloc(x27742 * sizeof(float));;
int32_t x27753 = 0;
int32_t x27754 = 0;
int32_t x27755 = 0;
bool x27801 = x27606 > 1;
bool x27805 = x27719 > 1;
for(int x27756=0; x27756 < 64; x27756++) {
int32_t x27757 = x27754;
int32_t x27758 = x27755;
int32_t x27759 = x27753;
int32_t x27760 = x27759;
int32_t x27761 = x27757;
int32_t x27762 = x27758;
for(int x27764=0; x27764 < x27737; x27764++) {
int32_t x27765 = x27761;
int32_t x27766 = x27762;
int32_t x27767 = x27760;
int32_t x27768 = x27767;
int32_t x27769 = x27765;
int32_t x27770 = x27766;
for(int x27772=0; x27772 < x27739; x27772++) {
int32_t x27773 = x27769;
int32_t x27774 = x27770;
int32_t x27775 = x27768;
int32_t x27776 = x27775;
int32_t x27777 = x27773;
int32_t x27778 = x27774;
for(int x27779=0; x27779 < x27739; x27779++) {
int32_t x27780 = x27776;
int32_t x27781 = x27777;
float x27782 = x27621[x27781];
int32_t x27783 = x27778;
float x27784 = x27688[x27783];
float x27785 = x27782 / x27784;
x27752[x27780] = x27785;
x27776 += 1;
if (x27788) {
x27777 += 1;
} else {
}

}
x27768 += x27739;
if (x27788) {
x27769 += x27608;
} else {
}

}
x27760 += x27740;
if (x27801) {
x27761 += x27609;
} else {
}
if (x27805) {
x27762 += 1;
} else {
}

}
x27753 += x27741;
x27754 += x27610;

}
int32_t x27815 = 0;
int32_t x27816 = 1;
x27816 *= 1;
x27815 += 1;
x27816 *= 1;
x27816 *= 1;
int32_t x27821 = x27815;
bool x27822 = x27821 >= 2;
if (x27822) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x27827 = x27821 == 0;
if (x27827) {
int32_t x27828 = x27816;
bool x27829 = x27828 == 2048;
if (x27829) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x27836 = x27816;
int32_t x27837 = 2048 / x27836;
bool x27843;
if (x452) {
bool x27838 = x27737 == 1;
bool x27839 = x27837 == 1;
bool x27840 = x27838 || x27839;
bool x27841 = x27737 == x27837;
bool x27842 = x27840 || x27841;
x27843 = x27842;
} else {
x27843 = false;
}
bool x27847;
if (x27843) {
x27847 = x27846;
} else {
x27847 = false;
}
bool x27848;
if (x27847) {
x27848 = x27846;
} else {
x27848 = false;
}
if (x27848) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x27737,x27739,x27739,1,x27837,1,1);
assert(false && "");
}
bool x27854 = x27737 <= x27837;
int32_t x27855;
if (x27854) {
x27855 = x27837;
} else {
x27855 = x27737;
}
bool x27861 = x27855 > 0;
bool x27863;
if (x27861) {
x27863 = x27862;
} else {
x27863 = false;
}
bool x27864;
if (x27863) {
x27864 = x27862;
} else {
x27864 = false;
}
if (x27864) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(27737) x Sym(27739) x Sym(27739)"," x Const(1) x Sym(27837) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x27859 = x27855 * x27858;
int32_t x27860 = 64 * x27859;
float* x27870 = (float*)myMalloc(x27860 * sizeof(float));;
int32_t x27871 = 0;
int32_t x27872 = 0;
int32_t x27873 = 0;
bool x27919 = x27737 > 1;
bool x27923 = x27837 > 1;
for(int x27874=0; x27874 < 64; x27874++) {
int32_t x27875 = x27872;
int32_t x27876 = x27873;
int32_t x27877 = x27871;
int32_t x27878 = x27877;
int32_t x27879 = x27875;
int32_t x27880 = x27876;
for(int x27882=0; x27882 < x27855; x27882++) {
int32_t x27883 = x27879;
int32_t x27884 = x27880;
int32_t x27885 = x27878;
int32_t x27886 = x27885;
int32_t x27887 = x27883;
int32_t x27888 = x27884;
for(int x27890=0; x27890 < x27857; x27890++) {
int32_t x27891 = x27887;
int32_t x27892 = x27888;
int32_t x27893 = x27886;
int32_t x27894 = x27893;
int32_t x27895 = x27891;
int32_t x27896 = x27892;
for(int x27897=0; x27897 < x27857; x27897++) {
int32_t x27898 = x27894;
int32_t x27899 = x27895;
float x27900 = x27752[x27899];
int32_t x27901 = x27896;
float x27902 = x173[x27901];
float x27903 = x27900 * x27902;
x27870[x27898] = x27903;
x27894 += 1;
if (x27906) {
x27895 += 1;
} else {
}

}
x27886 += x27857;
if (x27906) {
x27887 += x27739;
} else {
}

}
x27878 += x27858;
if (x27919) {
x27879 += x27740;
} else {
}
if (x27923) {
x27880 += 1;
} else {
}

}
x27871 += x27859;
x27872 += x27741;

}
int32_t x27933 = 0;
int32_t x27934 = 1;
x27934 *= 1;
x27933 += 1;
x27934 *= 1;
x27934 *= 1;
int32_t x27939 = x27933;
bool x27940 = x27939 >= 2;
if (x27940) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x27945 = x27939 == 0;
if (x27945) {
int32_t x27946 = x27934;
bool x27947 = x27946 == 2048;
if (x27947) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x27954 = x27934;
int32_t x27955 = 2048 / x27954;
bool x27961;
if (x452) {
bool x27956 = x27855 == 1;
bool x27957 = x27955 == 1;
bool x27958 = x27956 || x27957;
bool x27959 = x27855 == x27955;
bool x27960 = x27958 || x27959;
x27961 = x27960;
} else {
x27961 = false;
}
bool x27965;
if (x27961) {
x27965 = x27964;
} else {
x27965 = false;
}
bool x27966;
if (x27965) {
x27966 = x27964;
} else {
x27966 = false;
}
if (x27966) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x27855,x27857,x27857,1,x27955,1,1);
assert(false && "");
}
bool x27972 = x27855 <= x27955;
int32_t x27973;
if (x27972) {
x27973 = x27955;
} else {
x27973 = x27855;
}
bool x27979 = x27973 > 0;
bool x27981;
if (x27979) {
x27981 = x27980;
} else {
x27981 = false;
}
bool x27982;
if (x27981) {
x27982 = x27980;
} else {
x27982 = false;
}
if (x27982) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(27855) x Sym(27857) x Sym(27857)"," x Const(1) x Sym(27955) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x27977 = x27973 * x27976;
int32_t x27978 = 64 * x27977;
float* x27988 = (float*)myMalloc(x27978 * sizeof(float));;
int32_t x27989 = 0;
int32_t x27990 = 0;
int32_t x27991 = 0;
bool x28037 = x27855 > 1;
bool x28041 = x27955 > 1;
for(int x27992=0; x27992 < 64; x27992++) {
int32_t x27993 = x27990;
int32_t x27994 = x27991;
int32_t x27995 = x27989;
int32_t x27996 = x27995;
int32_t x27997 = x27993;
int32_t x27998 = x27994;
for(int x28000=0; x28000 < x27973; x28000++) {
int32_t x28001 = x27997;
int32_t x28002 = x27998;
int32_t x28003 = x27996;
int32_t x28004 = x28003;
int32_t x28005 = x28001;
int32_t x28006 = x28002;
for(int x28008=0; x28008 < x27975; x28008++) {
int32_t x28009 = x28005;
int32_t x28010 = x28006;
int32_t x28011 = x28004;
int32_t x28012 = x28011;
int32_t x28013 = x28009;
int32_t x28014 = x28010;
for(int x28015=0; x28015 < x27975; x28015++) {
int32_t x28016 = x28012;
int32_t x28017 = x28013;
float x28018 = x27870[x28017];
int32_t x28019 = x28014;
float x28020 = x107[x28019];
float x28021 = x28018 + x28020;
x27988[x28016] = x28021;
x28012 += 1;
if (x28024) {
x28013 += 1;
} else {
}

}
x28004 += x27975;
if (x28024) {
x28005 += x27857;
} else {
}

}
x27996 += x27976;
if (x28037) {
x27997 += x27858;
} else {
}
if (x28041) {
x27998 += 1;
} else {
}

}
x27989 += x27977;
x27990 += x27859;

}
bool x28051 = x27431 == 1;
bool x28052 = x27973 == 1;
bool x28053 = x28051 || x28052;
bool x28054 = x27431 == x27973;
bool x28055 = x28053 || x28054;
bool x28061;
if (x28055) {
x28061 = x28060;
} else {
x28061 = false;
}
bool x28062;
if (x28061) {
x28062 = x28060;
} else {
x28062 = false;
}
if (x28062) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x27431,x27433,x27433,64,x27973,x27975,x27975);
assert(false && "");
}
int32_t x28075 = 0;
int32_t x28076 = 0;
int32_t x28077 = 0;
bool x28068 = x27431 <= x27973;
int32_t x28069;
if (x28068) {
x28069 = x27973;
} else {
x28069 = x27431;
}
bool x28129 = x27431 > 1;
bool x28133 = x27973 > 1;
int32_t x28073 = x28069 * x28072;
for(int x28078=0; x28078 < 64; x28078++) {
int32_t x28079 = x28076;
int32_t x28080 = x28077;
int32_t x28081 = x28075;
int32_t x28082 = x28081;
int32_t x28083 = x28079;
int32_t x28084 = x28080;
for(int x28086=0; x28086 < x28069; x28086++) {
int32_t x28087 = x28083;
int32_t x28088 = x28084;
int32_t x28089 = x28082;
int32_t x28090 = x28089;
int32_t x28091 = x28087;
int32_t x28092 = x28088;
for(int x28094=0; x28094 < x28071; x28094++) {
int32_t x28095 = x28091;
int32_t x28096 = x28092;
int32_t x28097 = x28090;
int32_t x28098 = x28097;
int32_t x28099 = x28095;
int32_t x28100 = x28096;
for(int x28101=0; x28101 < x28071; x28101++) {
int32_t x28102 = x28099;
float x28103 = x27446[x28102];
int32_t x28104 = x28100;
float x28105 = x27988[x28104];
float x28106 = x28103 + x28105;
x27446[x28102] = x28106;
x28098 += 1;
if (x28109) {
x28099 += 1;
} else {
}
if (x28113) {
x28100 += 1;
} else {
}

}
x28090 += x28071;
if (x28109) {
x28091 += x27433;
} else {
}
if (x28113) {
x28092 += x27975;
} else {
}

}
x28082 += x28072;
if (x28129) {
x28083 += x27434;
} else {
}
if (x28133) {
x28084 += x27976;
} else {
}

}
x28075 += x28073;
x28076 += x27435;
x28077 += x27977;

}
float* x28144 = (float*)myMalloc(x27436 * sizeof(float));;
for(int x28146=0; x28146 < x27436; x28146++) {
float x28147 = x27446[x28146];
bool x28148 = x28147 < 0.0f;
if (x28148) {
x28144[x28146] = 0.0f;
} else {
float x28151 = x27446[x28146];
x28144[x28146] = x28151;
}

}
float* x28165 = (float*)myMalloc(x28164 * sizeof(float));;
int32_t x28168 = 64 * x27431;
int32_t x28169 = x28168 * x28160;
float* x28170 = (float*)myMalloc(x28169 * sizeof(float));;
int32_t x28166 = x27431 * x28160;
for(int x28171=0; x28171 < 64; x28171++) {
int32_t x28172 = x28171 * x27435;
float* x28173 = x28144+x28172;
int32_t x28174 = x28171 * x28161;
float* x28175 = x28165+x28174;
int32_t x28176 = x28171 * x28166;
float* x28177 = x28170+x28176;
for(int x28178=0; x28178 < x27431; x28178++) {
int32_t x28179 = x28178 / 1;
int32_t x28183 = x28179 * x28159;
int32_t x28184 = x28183 * x28159;
int32_t x28180 = x28178 % 1;
int32_t x28181 = x28180 / 1;
int32_t x28185 = x28181 * x28159;
int32_t x28186 = x28185 * x28159;
int32_t x28187 = x28184 + x28186;
int32_t x28182 = x28180 % 1;
int32_t x28188 = x28182 * x28159;
int32_t x28189 = x28188 * x28159;
int32_t x28190 = x28187 + x28189;
float* x28191 = x28177+x28190;
int32_t x28192 = x28179 * x27433;
int32_t x28193 = x28192 * x27433;
float* x28194 = x28173+x28193;
for(int x28196=0; x28196 < x28159; x28196++) {
int32_t x28198 = x28196 * x28159;
float* x28199 = x28191+x28198;
int32_t x28197 = x28196 + x28181;
int32_t x28200 = x28197 * x27433;
int32_t x28201 = x28200 + x28182;
float* x28202 = x28194+x28201;
memcpy(x28199, x28202, 4 * x28159);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x28160,x27431,1,x215,x27431,x28177,x28160,1,x28175,x28160);

}
int32_t x28211 = 0;
int32_t x28212 = 1;
x28212 *= 1;
x28211 += 1;
x28212 *= 1;
x28212 *= 1;
int32_t x28217 = x28211;
bool x28218 = x28217 >= 2;
if (x28218) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x28223 = x28217 == 0;
if (x28223) {
int32_t x28224 = x28212;
bool x28225 = x28224 == 512;
if (x28225) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x28232 = x28212;
int32_t x28233 = 512 / x28232;
bool x28237;
if (x452) {
bool x28234 = x28233 == 1;
bool x28235 = 512 == x28233;
bool x28236 = x28234 || x28235;
x28237 = x28236;
} else {
x28237 = false;
}
bool x28241;
if (x28237) {
x28241 = x28240;
} else {
x28241 = false;
}
bool x28242;
if (x28241) {
x28242 = x28240;
} else {
x28242 = false;
}
if (x28242) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x28159,x28159,1,x28233,1,1);
assert(false && "");
}
bool x28248 = 512 <= x28233;
int32_t x28249;
if (x28248) {
x28249 = x28233;
} else {
x28249 = 512;
}
bool x28255 = x28249 > 0;
bool x28257;
if (x28255) {
x28257 = x28256;
} else {
x28257 = false;
}
bool x28258;
if (x28257) {
x28258 = x28256;
} else {
x28258 = false;
}
if (x28258) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(28159) x Sym(28159)"," x Const(1) x Sym(28233) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x28253 = x28249 * x28252;
int32_t x28254 = 64 * x28253;
float* x28264 = (float*)myMalloc(x28254 * sizeof(float));;
int32_t x28265 = 0;
int32_t x28266 = 0;
int32_t x28267 = 0;
bool x28314 = x28233 > 1;
for(int x28268=0; x28268 < 64; x28268++) {
int32_t x28269 = x28266;
int32_t x28270 = x28267;
int32_t x28271 = x28265;
int32_t x28272 = x28271;
int32_t x28273 = x28269;
int32_t x28274 = x28270;
for(int x28276=0; x28276 < x28249; x28276++) {
int32_t x28277 = x28273;
int32_t x28278 = x28274;
int32_t x28279 = x28272;
int32_t x28280 = x28279;
int32_t x28281 = x28277;
int32_t x28282 = x28278;
for(int x28284=0; x28284 < x28251; x28284++) {
int32_t x28285 = x28281;
int32_t x28286 = x28282;
int32_t x28287 = x28280;
int32_t x28288 = x28287;
int32_t x28289 = x28285;
int32_t x28290 = x28286;
for(int x28291=0; x28291 < x28251; x28291++) {
int32_t x28292 = x28288;
int32_t x28293 = x28289;
float x28294 = x28165[x28293];
int32_t x28295 = x28290;
float x28296 = x154[x28295];
float x28297 = x28294 - x28296;
x28264[x28292] = x28297;
x28288 += 1;
if (x28300) {
x28289 += 1;
} else {
}

}
x28280 += x28251;
if (x28300) {
x28281 += x28159;
} else {
}

}
x28272 += x28252;
x28273 += x28160;
if (x28314) {
x28274 += 1;
} else {
}

}
x28265 += x28253;
x28266 += x28161;

}
float* x28324 = (float*)myMalloc(512 * sizeof(float));;
for(int x28325=0; x28325 < 512; x28325++) {
float x28326 = x65[x28325];
float x28327 = x28326 + 1.0E-5f;
x28324[x28325] = x28327;

}
float* x28331 = (float*)myMalloc(512 * sizeof(float));;
for(int x28332=0; x28332 < 512; x28332++) {
float x28333 = x28324[x28332];
double x28334 = (double)x28333;
double x28335 = sqrt(x28334);
float x28336 = (float)x28335;
x28331[x28332] = x28336;

}
int32_t x28340 = 0;
int32_t x28341 = 1;
x28341 *= 1;
x28340 += 1;
x28341 *= 1;
x28341 *= 1;
int32_t x28346 = x28340;
bool x28347 = x28346 >= 2;
if (x28347) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x28352 = x28346 == 0;
if (x28352) {
int32_t x28353 = x28341;
bool x28354 = x28353 == 512;
if (x28354) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x28361 = x28341;
int32_t x28362 = 512 / x28361;
bool x28368;
if (x452) {
bool x28363 = x28249 == 1;
bool x28364 = x28362 == 1;
bool x28365 = x28363 || x28364;
bool x28366 = x28249 == x28362;
bool x28367 = x28365 || x28366;
x28368 = x28367;
} else {
x28368 = false;
}
bool x28372;
if (x28368) {
x28372 = x28371;
} else {
x28372 = false;
}
bool x28373;
if (x28372) {
x28373 = x28371;
} else {
x28373 = false;
}
if (x28373) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x28249,x28251,x28251,1,x28362,1,1);
assert(false && "");
}
bool x28379 = x28249 <= x28362;
int32_t x28380;
if (x28379) {
x28380 = x28362;
} else {
x28380 = x28249;
}
bool x28386 = x28380 > 0;
bool x28388;
if (x28386) {
x28388 = x28387;
} else {
x28388 = false;
}
bool x28389;
if (x28388) {
x28389 = x28387;
} else {
x28389 = false;
}
if (x28389) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(28249) x Sym(28251) x Sym(28251)"," x Const(1) x Sym(28362) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x28384 = x28380 * x28383;
int32_t x28385 = 64 * x28384;
float* x28395 = (float*)myMalloc(x28385 * sizeof(float));;
int32_t x28396 = 0;
int32_t x28397 = 0;
int32_t x28398 = 0;
bool x28444 = x28249 > 1;
bool x28448 = x28362 > 1;
for(int x28399=0; x28399 < 64; x28399++) {
int32_t x28400 = x28397;
int32_t x28401 = x28398;
int32_t x28402 = x28396;
int32_t x28403 = x28402;
int32_t x28404 = x28400;
int32_t x28405 = x28401;
for(int x28407=0; x28407 < x28380; x28407++) {
int32_t x28408 = x28404;
int32_t x28409 = x28405;
int32_t x28410 = x28403;
int32_t x28411 = x28410;
int32_t x28412 = x28408;
int32_t x28413 = x28409;
for(int x28415=0; x28415 < x28382; x28415++) {
int32_t x28416 = x28412;
int32_t x28417 = x28413;
int32_t x28418 = x28411;
int32_t x28419 = x28418;
int32_t x28420 = x28416;
int32_t x28421 = x28417;
for(int x28422=0; x28422 < x28382; x28422++) {
int32_t x28423 = x28419;
int32_t x28424 = x28420;
float x28425 = x28264[x28424];
int32_t x28426 = x28421;
float x28427 = x28331[x28426];
float x28428 = x28425 / x28427;
x28395[x28423] = x28428;
x28419 += 1;
if (x28431) {
x28420 += 1;
} else {
}

}
x28411 += x28382;
if (x28431) {
x28412 += x28251;
} else {
}

}
x28403 += x28383;
if (x28444) {
x28404 += x28252;
} else {
}
if (x28448) {
x28405 += 1;
} else {
}

}
x28396 += x28384;
x28397 += x28253;

}
int32_t x28458 = 0;
int32_t x28459 = 1;
x28459 *= 1;
x28458 += 1;
x28459 *= 1;
x28459 *= 1;
int32_t x28464 = x28458;
bool x28465 = x28464 >= 2;
if (x28465) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x28470 = x28464 == 0;
if (x28470) {
int32_t x28471 = x28459;
bool x28472 = x28471 == 512;
if (x28472) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x28479 = x28459;
int32_t x28480 = 512 / x28479;
bool x28486;
if (x452) {
bool x28481 = x28380 == 1;
bool x28482 = x28480 == 1;
bool x28483 = x28481 || x28482;
bool x28484 = x28380 == x28480;
bool x28485 = x28483 || x28484;
x28486 = x28485;
} else {
x28486 = false;
}
bool x28490;
if (x28486) {
x28490 = x28489;
} else {
x28490 = false;
}
bool x28491;
if (x28490) {
x28491 = x28489;
} else {
x28491 = false;
}
if (x28491) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x28380,x28382,x28382,1,x28480,1,1);
assert(false && "");
}
bool x28497 = x28380 <= x28480;
int32_t x28498;
if (x28497) {
x28498 = x28480;
} else {
x28498 = x28380;
}
bool x28504 = x28498 > 0;
bool x28506;
if (x28504) {
x28506 = x28505;
} else {
x28506 = false;
}
bool x28507;
if (x28506) {
x28507 = x28505;
} else {
x28507 = false;
}
if (x28507) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(28380) x Sym(28382) x Sym(28382)"," x Const(1) x Sym(28480) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x28502 = x28498 * x28501;
int32_t x28503 = 64 * x28502;
float* x28513 = (float*)myMalloc(x28503 * sizeof(float));;
int32_t x28514 = 0;
int32_t x28515 = 0;
int32_t x28516 = 0;
bool x28562 = x28380 > 1;
bool x28566 = x28480 > 1;
for(int x28517=0; x28517 < 64; x28517++) {
int32_t x28518 = x28515;
int32_t x28519 = x28516;
int32_t x28520 = x28514;
int32_t x28521 = x28520;
int32_t x28522 = x28518;
int32_t x28523 = x28519;
for(int x28525=0; x28525 < x28498; x28525++) {
int32_t x28526 = x28522;
int32_t x28527 = x28523;
int32_t x28528 = x28521;
int32_t x28529 = x28528;
int32_t x28530 = x28526;
int32_t x28531 = x28527;
for(int x28533=0; x28533 < x28500; x28533++) {
int32_t x28534 = x28530;
int32_t x28535 = x28531;
int32_t x28536 = x28529;
int32_t x28537 = x28536;
int32_t x28538 = x28534;
int32_t x28539 = x28535;
for(int x28540=0; x28540 < x28500; x28540++) {
int32_t x28541 = x28537;
int32_t x28542 = x28538;
float x28543 = x28395[x28542];
int32_t x28544 = x28539;
float x28545 = x46[x28544];
float x28546 = x28543 * x28545;
x28513[x28541] = x28546;
x28537 += 1;
if (x28549) {
x28538 += 1;
} else {
}

}
x28529 += x28500;
if (x28549) {
x28530 += x28382;
} else {
}

}
x28521 += x28501;
if (x28562) {
x28522 += x28383;
} else {
}
if (x28566) {
x28523 += 1;
} else {
}

}
x28514 += x28502;
x28515 += x28384;

}
int32_t x28576 = 0;
int32_t x28577 = 1;
x28577 *= 1;
x28576 += 1;
x28577 *= 1;
x28577 *= 1;
int32_t x28582 = x28576;
bool x28583 = x28582 >= 2;
if (x28583) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x28588 = x28582 == 0;
if (x28588) {
int32_t x28589 = x28577;
bool x28590 = x28589 == 512;
if (x28590) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x28597 = x28577;
int32_t x28598 = 512 / x28597;
bool x28604;
if (x452) {
bool x28599 = x28498 == 1;
bool x28600 = x28598 == 1;
bool x28601 = x28599 || x28600;
bool x28602 = x28498 == x28598;
bool x28603 = x28601 || x28602;
x28604 = x28603;
} else {
x28604 = false;
}
bool x28608;
if (x28604) {
x28608 = x28607;
} else {
x28608 = false;
}
bool x28609;
if (x28608) {
x28609 = x28607;
} else {
x28609 = false;
}
if (x28609) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x28498,x28500,x28500,1,x28598,1,1);
assert(false && "");
}
bool x28615 = x28498 <= x28598;
int32_t x28616;
if (x28615) {
x28616 = x28598;
} else {
x28616 = x28498;
}
bool x28622 = x28616 > 0;
bool x28624;
if (x28622) {
x28624 = x28623;
} else {
x28624 = false;
}
bool x28625;
if (x28624) {
x28625 = x28623;
} else {
x28625 = false;
}
if (x28625) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(28498) x Sym(28500) x Sym(28500)"," x Const(1) x Sym(28598) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x28620 = x28616 * x28619;
int32_t x28621 = 64 * x28620;
float* x28631 = (float*)myMalloc(x28621 * sizeof(float));;
int32_t x28632 = 0;
int32_t x28633 = 0;
int32_t x28634 = 0;
bool x28680 = x28498 > 1;
bool x28684 = x28598 > 1;
for(int x28635=0; x28635 < 64; x28635++) {
int32_t x28636 = x28633;
int32_t x28637 = x28634;
int32_t x28638 = x28632;
int32_t x28639 = x28638;
int32_t x28640 = x28636;
int32_t x28641 = x28637;
for(int x28643=0; x28643 < x28616; x28643++) {
int32_t x28644 = x28640;
int32_t x28645 = x28641;
int32_t x28646 = x28639;
int32_t x28647 = x28646;
int32_t x28648 = x28644;
int32_t x28649 = x28645;
for(int x28651=0; x28651 < x28618; x28651++) {
int32_t x28652 = x28648;
int32_t x28653 = x28649;
int32_t x28654 = x28647;
int32_t x28655 = x28654;
int32_t x28656 = x28652;
int32_t x28657 = x28653;
for(int x28658=0; x28658 < x28618; x28658++) {
int32_t x28659 = x28655;
int32_t x28660 = x28656;
float x28661 = x28513[x28660];
int32_t x28662 = x28657;
float x28663 = x137[x28662];
float x28664 = x28661 + x28663;
x28631[x28659] = x28664;
x28655 += 1;
if (x28667) {
x28656 += 1;
} else {
}

}
x28647 += x28618;
if (x28667) {
x28648 += x28500;
} else {
}

}
x28639 += x28619;
if (x28680) {
x28640 += x28501;
} else {
}
if (x28684) {
x28641 += 1;
} else {
}

}
x28632 += x28620;
x28633 += x28502;

}
float* x28694 = (float*)myMalloc(x28621 * sizeof(float));;
for(int x28696=0; x28696 < x28621; x28696++) {
float x28697 = x28631[x28696];
bool x28698 = x28697 < 0.0f;
if (x28698) {
x28694[x28696] = 0.0f;
} else {
float x28701 = x28631[x28696];
x28694[x28696] = x28701;
}

}
float* x28716 = (float*)myMalloc(x28715 * sizeof(float));;
int32_t x28717 = 9 * x28616;
int32_t x28720 = 64 * x28717;
int32_t x28721 = x28720 * x28711;
float* x28722 = (float*)myMalloc(x28721 * sizeof(float));;
int32_t x28718 = x28717 * x28711;
int32_t x28730 = x28616 * 3;
int32_t x28731 = x28730 * 3;
for(int x28723=0; x28723 < 64; x28723++) {
int32_t x28724 = x28723 * x28620;
float* x28725 = x28694+x28724;
int32_t x28726 = x28723 * x28712;
float* x28727 = x28716+x28726;
int32_t x28728 = x28723 * x28718;
float* x28729 = x28722+x28728;
for(int x28733=0; x28733 < x28731; x28733++) {
int32_t x28734 = x28733 / 9;
int32_t x28738 = x28734 * 3;
int32_t x28739 = x28738 * 3;
int32_t x28740 = x28739 * x28710;
int32_t x28741 = x28740 * x28710;
int32_t x28735 = x28733 % 9;
int32_t x28736 = x28735 / 3;
int32_t x28742 = x28736 * 3;
int32_t x28743 = x28742 * x28710;
int32_t x28744 = x28743 * x28710;
int32_t x28745 = x28741 + x28744;
int32_t x28737 = x28735 % 3;
int32_t x28746 = x28737 * x28710;
int32_t x28747 = x28746 * x28710;
int32_t x28748 = x28745 + x28747;
float* x28749 = x28729+x28748;
int32_t x28750 = x28734 * x28618;
int32_t x28751 = x28750 * x28618;
float* x28752 = x28725+x28751;
int32_t x28765 = 1 - x28737;
bool x28766 = x28765 > 0;
int32_t x28767;
if (x28766) {
x28767 = x28765;
} else {
x28767 = 0;
}
int32_t x28768 = 3 - x28737;
int32_t x28769 = x28768 - 1;
int32_t x28770 = 1 - x28769;
bool x28771 = x28770 > 0;
int32_t x28772;
if (x28771) {
x28772 = x28770;
} else {
x28772 = 0;
}
int32_t x28773 = x28710 - x28772;
int32_t x28774 = x28773 - x28767;
bool x28775 = x28774 <= 0;
bool x28779 = x28767 > 0;
int32_t x28764 = -1 + x28737;
bool x28792 = x28772 > 0;
for(int x28754=0; x28754 < x28710; x28754++) {
int32_t x28755 = x28754 - 1;
int32_t x28756 = x28755 + x28736;
bool x28757 = x28756 < 0;
bool x28758 = x28756 >= x28618;
bool x28759 = x28757 || x28758;
if (x28759) {
int32_t x28760 = x28754 * x28710;
float* x28761 = x28749+x28760;
memset(x28761, 0, 4 * x28710);;
} else {
if (x28775) {
int32_t x28760 = x28754 * x28710;
float* x28776 = x28749+x28760;
memset(x28776, 0, 4 * x28710);;
} else {
int32_t x28760 = x28754 * x28710;
if (x28779) {
float* x28780 = x28749+x28760;
memset(x28780, 0, 4 * x28767);;
} else {
}
// may have segfault here
int32_t x28785 = x28760 + x28767;
float* x28786 = x28749+x28785;
int32_t x28787 = x28756 * x28618;
int32_t x28788 = x28787 + x28764;
int32_t x28789 = x28788 + x28767;
float* x28790 = x28752+x28789;
memcpy(x28786, x28790, 4 * x28774);;
if (x28792) {
int32_t x28793 = x28760 + x28710;
int32_t x28794 = x28793 - x28772;
float* x28795 = x28749+x28794;
memset(x28795, 0, 4 * x28772);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x28711,x28717,1,x155,x28717,x28729,x28711,1,x28727,x28711);

}
int32_t x28810 = 0;
int32_t x28811 = 1;
x28811 *= 1;
x28810 += 1;
x28811 *= 1;
x28811 *= 1;
int32_t x28816 = x28810;
bool x28817 = x28816 >= 2;
if (x28817) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x28822 = x28816 == 0;
if (x28822) {
int32_t x28823 = x28811;
bool x28824 = x28823 == 512;
if (x28824) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x28831 = x28811;
int32_t x28832 = 512 / x28831;
bool x28836;
if (x452) {
bool x28833 = x28832 == 1;
bool x28834 = 512 == x28832;
bool x28835 = x28833 || x28834;
x28836 = x28835;
} else {
x28836 = false;
}
bool x28840;
if (x28836) {
x28840 = x28839;
} else {
x28840 = false;
}
bool x28841;
if (x28840) {
x28841 = x28839;
} else {
x28841 = false;
}
if (x28841) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x28710,x28710,1,x28832,1,1);
assert(false && "");
}
bool x28847 = 512 <= x28832;
int32_t x28848;
if (x28847) {
x28848 = x28832;
} else {
x28848 = 512;
}
bool x28854 = x28848 > 0;
bool x28856;
if (x28854) {
x28856 = x28855;
} else {
x28856 = false;
}
bool x28857;
if (x28856) {
x28857 = x28855;
} else {
x28857 = false;
}
if (x28857) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(28710) x Sym(28710)"," x Const(1) x Sym(28832) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x28852 = x28848 * x28851;
int32_t x28853 = 64 * x28852;
float* x28863 = (float*)myMalloc(x28853 * sizeof(float));;
int32_t x28864 = 0;
int32_t x28865 = 0;
int32_t x28866 = 0;
bool x28913 = x28832 > 1;
for(int x28867=0; x28867 < 64; x28867++) {
int32_t x28868 = x28865;
int32_t x28869 = x28866;
int32_t x28870 = x28864;
int32_t x28871 = x28870;
int32_t x28872 = x28868;
int32_t x28873 = x28869;
for(int x28875=0; x28875 < x28848; x28875++) {
int32_t x28876 = x28872;
int32_t x28877 = x28873;
int32_t x28878 = x28871;
int32_t x28879 = x28878;
int32_t x28880 = x28876;
int32_t x28881 = x28877;
for(int x28883=0; x28883 < x28850; x28883++) {
int32_t x28884 = x28880;
int32_t x28885 = x28881;
int32_t x28886 = x28879;
int32_t x28887 = x28886;
int32_t x28888 = x28884;
int32_t x28889 = x28885;
for(int x28890=0; x28890 < x28850; x28890++) {
int32_t x28891 = x28887;
int32_t x28892 = x28888;
float x28893 = x28716[x28892];
int32_t x28894 = x28889;
float x28895 = x138[x28894];
float x28896 = x28893 - x28895;
x28863[x28891] = x28896;
x28887 += 1;
if (x28899) {
x28888 += 1;
} else {
}

}
x28879 += x28850;
if (x28899) {
x28880 += x28710;
} else {
}

}
x28871 += x28851;
x28872 += x28711;
if (x28913) {
x28873 += 1;
} else {
}

}
x28864 += x28852;
x28865 += x28712;

}
float* x28923 = (float*)myMalloc(512 * sizeof(float));;
for(int x28924=0; x28924 < 512; x28924++) {
float x28925 = x195[x28924];
float x28926 = x28925 + 1.0E-5f;
x28923[x28924] = x28926;

}
float* x28930 = (float*)myMalloc(512 * sizeof(float));;
for(int x28931=0; x28931 < 512; x28931++) {
float x28932 = x28923[x28931];
double x28933 = (double)x28932;
double x28934 = sqrt(x28933);
float x28935 = (float)x28934;
x28930[x28931] = x28935;

}
int32_t x28939 = 0;
int32_t x28940 = 1;
x28940 *= 1;
x28939 += 1;
x28940 *= 1;
x28940 *= 1;
int32_t x28945 = x28939;
bool x28946 = x28945 >= 2;
if (x28946) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x28951 = x28945 == 0;
if (x28951) {
int32_t x28952 = x28940;
bool x28953 = x28952 == 512;
if (x28953) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x28960 = x28940;
int32_t x28961 = 512 / x28960;
bool x28967;
if (x452) {
bool x28962 = x28848 == 1;
bool x28963 = x28961 == 1;
bool x28964 = x28962 || x28963;
bool x28965 = x28848 == x28961;
bool x28966 = x28964 || x28965;
x28967 = x28966;
} else {
x28967 = false;
}
bool x28971;
if (x28967) {
x28971 = x28970;
} else {
x28971 = false;
}
bool x28972;
if (x28971) {
x28972 = x28970;
} else {
x28972 = false;
}
if (x28972) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x28848,x28850,x28850,1,x28961,1,1);
assert(false && "");
}
bool x28978 = x28848 <= x28961;
int32_t x28979;
if (x28978) {
x28979 = x28961;
} else {
x28979 = x28848;
}
bool x28985 = x28979 > 0;
bool x28987;
if (x28985) {
x28987 = x28986;
} else {
x28987 = false;
}
bool x28988;
if (x28987) {
x28988 = x28986;
} else {
x28988 = false;
}
if (x28988) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(28848) x Sym(28850) x Sym(28850)"," x Const(1) x Sym(28961) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x28983 = x28979 * x28982;
int32_t x28984 = 64 * x28983;
float* x28994 = (float*)myMalloc(x28984 * sizeof(float));;
int32_t x28995 = 0;
int32_t x28996 = 0;
int32_t x28997 = 0;
bool x29043 = x28848 > 1;
bool x29047 = x28961 > 1;
for(int x28998=0; x28998 < 64; x28998++) {
int32_t x28999 = x28996;
int32_t x29000 = x28997;
int32_t x29001 = x28995;
int32_t x29002 = x29001;
int32_t x29003 = x28999;
int32_t x29004 = x29000;
for(int x29006=0; x29006 < x28979; x29006++) {
int32_t x29007 = x29003;
int32_t x29008 = x29004;
int32_t x29009 = x29002;
int32_t x29010 = x29009;
int32_t x29011 = x29007;
int32_t x29012 = x29008;
for(int x29014=0; x29014 < x28981; x29014++) {
int32_t x29015 = x29011;
int32_t x29016 = x29012;
int32_t x29017 = x29010;
int32_t x29018 = x29017;
int32_t x29019 = x29015;
int32_t x29020 = x29016;
for(int x29021=0; x29021 < x28981; x29021++) {
int32_t x29022 = x29018;
int32_t x29023 = x29019;
float x29024 = x28863[x29023];
int32_t x29025 = x29020;
float x29026 = x28930[x29025];
float x29027 = x29024 / x29026;
x28994[x29022] = x29027;
x29018 += 1;
if (x29030) {
x29019 += 1;
} else {
}

}
x29010 += x28981;
if (x29030) {
x29011 += x28850;
} else {
}

}
x29002 += x28982;
if (x29043) {
x29003 += x28851;
} else {
}
if (x29047) {
x29004 += 1;
} else {
}

}
x28995 += x28983;
x28996 += x28852;

}
int32_t x29057 = 0;
int32_t x29058 = 1;
x29058 *= 1;
x29057 += 1;
x29058 *= 1;
x29058 *= 1;
int32_t x29063 = x29057;
bool x29064 = x29063 >= 2;
if (x29064) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x29069 = x29063 == 0;
if (x29069) {
int32_t x29070 = x29058;
bool x29071 = x29070 == 512;
if (x29071) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x29078 = x29058;
int32_t x29079 = 512 / x29078;
bool x29085;
if (x452) {
bool x29080 = x28979 == 1;
bool x29081 = x29079 == 1;
bool x29082 = x29080 || x29081;
bool x29083 = x28979 == x29079;
bool x29084 = x29082 || x29083;
x29085 = x29084;
} else {
x29085 = false;
}
bool x29089;
if (x29085) {
x29089 = x29088;
} else {
x29089 = false;
}
bool x29090;
if (x29089) {
x29090 = x29088;
} else {
x29090 = false;
}
if (x29090) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x28979,x28981,x28981,1,x29079,1,1);
assert(false && "");
}
bool x29096 = x28979 <= x29079;
int32_t x29097;
if (x29096) {
x29097 = x29079;
} else {
x29097 = x28979;
}
bool x29103 = x29097 > 0;
bool x29105;
if (x29103) {
x29105 = x29104;
} else {
x29105 = false;
}
bool x29106;
if (x29105) {
x29106 = x29104;
} else {
x29106 = false;
}
if (x29106) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(28979) x Sym(28981) x Sym(28981)"," x Const(1) x Sym(29079) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x29101 = x29097 * x29100;
int32_t x29102 = 64 * x29101;
float* x29112 = (float*)myMalloc(x29102 * sizeof(float));;
int32_t x29113 = 0;
int32_t x29114 = 0;
int32_t x29115 = 0;
bool x29161 = x28979 > 1;
bool x29165 = x29079 > 1;
for(int x29116=0; x29116 < 64; x29116++) {
int32_t x29117 = x29114;
int32_t x29118 = x29115;
int32_t x29119 = x29113;
int32_t x29120 = x29119;
int32_t x29121 = x29117;
int32_t x29122 = x29118;
for(int x29124=0; x29124 < x29097; x29124++) {
int32_t x29125 = x29121;
int32_t x29126 = x29122;
int32_t x29127 = x29120;
int32_t x29128 = x29127;
int32_t x29129 = x29125;
int32_t x29130 = x29126;
for(int x29132=0; x29132 < x29099; x29132++) {
int32_t x29133 = x29129;
int32_t x29134 = x29130;
int32_t x29135 = x29128;
int32_t x29136 = x29135;
int32_t x29137 = x29133;
int32_t x29138 = x29134;
for(int x29139=0; x29139 < x29099; x29139++) {
int32_t x29140 = x29136;
int32_t x29141 = x29137;
float x29142 = x28994[x29141];
int32_t x29143 = x29138;
float x29144 = x160[x29143];
float x29145 = x29142 * x29144;
x29112[x29140] = x29145;
x29136 += 1;
if (x29148) {
x29137 += 1;
} else {
}

}
x29128 += x29099;
if (x29148) {
x29129 += x28981;
} else {
}

}
x29120 += x29100;
if (x29161) {
x29121 += x28982;
} else {
}
if (x29165) {
x29122 += 1;
} else {
}

}
x29113 += x29101;
x29114 += x28983;

}
int32_t x29175 = 0;
int32_t x29176 = 1;
x29176 *= 1;
x29175 += 1;
x29176 *= 1;
x29176 *= 1;
int32_t x29181 = x29175;
bool x29182 = x29181 >= 2;
if (x29182) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x29187 = x29181 == 0;
if (x29187) {
int32_t x29188 = x29176;
bool x29189 = x29188 == 512;
if (x29189) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x29196 = x29176;
int32_t x29197 = 512 / x29196;
bool x29203;
if (x452) {
bool x29198 = x29097 == 1;
bool x29199 = x29197 == 1;
bool x29200 = x29198 || x29199;
bool x29201 = x29097 == x29197;
bool x29202 = x29200 || x29201;
x29203 = x29202;
} else {
x29203 = false;
}
bool x29207;
if (x29203) {
x29207 = x29206;
} else {
x29207 = false;
}
bool x29208;
if (x29207) {
x29208 = x29206;
} else {
x29208 = false;
}
if (x29208) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x29097,x29099,x29099,1,x29197,1,1);
assert(false && "");
}
bool x29214 = x29097 <= x29197;
int32_t x29215;
if (x29214) {
x29215 = x29197;
} else {
x29215 = x29097;
}
bool x29221 = x29215 > 0;
bool x29223;
if (x29221) {
x29223 = x29222;
} else {
x29223 = false;
}
bool x29224;
if (x29223) {
x29224 = x29222;
} else {
x29224 = false;
}
if (x29224) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(29097) x Sym(29099) x Sym(29099)"," x Const(1) x Sym(29197) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x29219 = x29215 * x29218;
int32_t x29220 = 64 * x29219;
float* x29230 = (float*)myMalloc(x29220 * sizeof(float));;
int32_t x29231 = 0;
int32_t x29232 = 0;
int32_t x29233 = 0;
bool x29279 = x29097 > 1;
bool x29283 = x29197 > 1;
for(int x29234=0; x29234 < 64; x29234++) {
int32_t x29235 = x29232;
int32_t x29236 = x29233;
int32_t x29237 = x29231;
int32_t x29238 = x29237;
int32_t x29239 = x29235;
int32_t x29240 = x29236;
for(int x29242=0; x29242 < x29215; x29242++) {
int32_t x29243 = x29239;
int32_t x29244 = x29240;
int32_t x29245 = x29238;
int32_t x29246 = x29245;
int32_t x29247 = x29243;
int32_t x29248 = x29244;
for(int x29250=0; x29250 < x29217; x29250++) {
int32_t x29251 = x29247;
int32_t x29252 = x29248;
int32_t x29253 = x29246;
int32_t x29254 = x29253;
int32_t x29255 = x29251;
int32_t x29256 = x29252;
for(int x29257=0; x29257 < x29217; x29257++) {
int32_t x29258 = x29254;
int32_t x29259 = x29255;
float x29260 = x29112[x29259];
int32_t x29261 = x29256;
float x29262 = x66[x29261];
float x29263 = x29260 + x29262;
x29230[x29258] = x29263;
x29254 += 1;
if (x29266) {
x29255 += 1;
} else {
}

}
x29246 += x29217;
if (x29266) {
x29247 += x29099;
} else {
}

}
x29238 += x29218;
if (x29279) {
x29239 += x29100;
} else {
}
if (x29283) {
x29240 += 1;
} else {
}

}
x29231 += x29219;
x29232 += x29101;

}
float* x29293 = (float*)myMalloc(x29220 * sizeof(float));;
for(int x29295=0; x29295 < x29220; x29295++) {
float x29296 = x29230[x29295];
bool x29297 = x29296 < 0.0f;
if (x29297) {
x29293[x29295] = 0.0f;
} else {
float x29300 = x29230[x29295];
x29293[x29295] = x29300;
}

}
float* x29314 = (float*)myMalloc(x29313 * sizeof(float));;
int32_t x29317 = 64 * x29215;
int32_t x29318 = x29317 * x29309;
float* x29319 = (float*)myMalloc(x29318 * sizeof(float));;
int32_t x29315 = x29215 * x29309;
for(int x29320=0; x29320 < 64; x29320++) {
int32_t x29321 = x29320 * x29219;
float* x29322 = x29293+x29321;
int32_t x29323 = x29320 * x29310;
float* x29324 = x29314+x29323;
int32_t x29325 = x29320 * x29315;
float* x29326 = x29319+x29325;
for(int x29327=0; x29327 < x29215; x29327++) {
int32_t x29328 = x29327 / 1;
int32_t x29332 = x29328 * x29308;
int32_t x29333 = x29332 * x29308;
int32_t x29329 = x29327 % 1;
int32_t x29330 = x29329 / 1;
int32_t x29334 = x29330 * x29308;
int32_t x29335 = x29334 * x29308;
int32_t x29336 = x29333 + x29335;
int32_t x29331 = x29329 % 1;
int32_t x29337 = x29331 * x29308;
int32_t x29338 = x29337 * x29308;
int32_t x29339 = x29336 + x29338;
float* x29340 = x29326+x29339;
int32_t x29341 = x29328 * x29217;
int32_t x29342 = x29341 * x29217;
float* x29343 = x29322+x29342;
for(int x29345=0; x29345 < x29308; x29345++) {
int32_t x29347 = x29345 * x29308;
float* x29348 = x29340+x29347;
int32_t x29346 = x29345 + x29330;
int32_t x29349 = x29346 * x29217;
int32_t x29350 = x29349 + x29331;
float* x29351 = x29343+x29350;
memcpy(x29348, x29351, 4 * x29308);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x29309,x29215,1,x47,x29215,x29326,x29309,1,x29324,x29309);

}
int32_t x29360 = 0;
int32_t x29361 = 1;
x29361 *= 1;
x29360 += 1;
x29361 *= 1;
x29361 *= 1;
int32_t x29366 = x29360;
bool x29367 = x29366 >= 2;
if (x29367) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x29372 = x29366 == 0;
if (x29372) {
int32_t x29373 = x29361;
bool x29374 = x29373 == 2048;
if (x29374) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x29381 = x29361;
int32_t x29382 = 2048 / x29381;
bool x29386;
if (x452) {
bool x29383 = x29382 == 1;
bool x29384 = 2048 == x29382;
bool x29385 = x29383 || x29384;
x29386 = x29385;
} else {
x29386 = false;
}
bool x29390;
if (x29386) {
x29390 = x29389;
} else {
x29390 = false;
}
bool x29391;
if (x29390) {
x29391 = x29389;
} else {
x29391 = false;
}
if (x29391) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x29308,x29308,1,x29382,1,1);
assert(false && "");
}
bool x29397 = 2048 <= x29382;
int32_t x29398;
if (x29397) {
x29398 = x29382;
} else {
x29398 = 2048;
}
bool x29404 = x29398 > 0;
bool x29406;
if (x29404) {
x29406 = x29405;
} else {
x29406 = false;
}
bool x29407;
if (x29406) {
x29407 = x29405;
} else {
x29407 = false;
}
if (x29407) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(2048) x Sym(29308) x Sym(29308)"," x Const(1) x Sym(29382) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x29402 = x29398 * x29401;
int32_t x29403 = 64 * x29402;
float* x29413 = (float*)myMalloc(x29403 * sizeof(float));;
int32_t x29414 = 0;
int32_t x29415 = 0;
int32_t x29416 = 0;
bool x29463 = x29382 > 1;
for(int x29417=0; x29417 < 64; x29417++) {
int32_t x29418 = x29415;
int32_t x29419 = x29416;
int32_t x29420 = x29414;
int32_t x29421 = x29420;
int32_t x29422 = x29418;
int32_t x29423 = x29419;
for(int x29425=0; x29425 < x29398; x29425++) {
int32_t x29426 = x29422;
int32_t x29427 = x29423;
int32_t x29428 = x29421;
int32_t x29429 = x29428;
int32_t x29430 = x29426;
int32_t x29431 = x29427;
for(int x29433=0; x29433 < x29400; x29433++) {
int32_t x29434 = x29430;
int32_t x29435 = x29431;
int32_t x29436 = x29429;
int32_t x29437 = x29436;
int32_t x29438 = x29434;
int32_t x29439 = x29435;
for(int x29440=0; x29440 < x29400; x29440++) {
int32_t x29441 = x29437;
int32_t x29442 = x29438;
float x29443 = x29314[x29442];
int32_t x29444 = x29439;
float x29445 = x68[x29444];
float x29446 = x29443 - x29445;
x29413[x29441] = x29446;
x29437 += 1;
if (x29449) {
x29438 += 1;
} else {
}

}
x29429 += x29400;
if (x29449) {
x29430 += x29308;
} else {
}

}
x29421 += x29401;
x29422 += x29309;
if (x29463) {
x29423 += 1;
} else {
}

}
x29414 += x29402;
x29415 += x29310;

}
float* x29473 = (float*)myMalloc(2048 * sizeof(float));;
for(int x29474=0; x29474 < 2048; x29474++) {
float x29475 = x245[x29474];
float x29476 = x29475 + 1.0E-5f;
x29473[x29474] = x29476;

}
float* x29480 = (float*)myMalloc(2048 * sizeof(float));;
for(int x29481=0; x29481 < 2048; x29481++) {
float x29482 = x29473[x29481];
double x29483 = (double)x29482;
double x29484 = sqrt(x29483);
float x29485 = (float)x29484;
x29480[x29481] = x29485;

}
int32_t x29489 = 0;
int32_t x29490 = 1;
x29490 *= 1;
x29489 += 1;
x29490 *= 1;
x29490 *= 1;
int32_t x29495 = x29489;
bool x29496 = x29495 >= 2;
if (x29496) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x29501 = x29495 == 0;
if (x29501) {
int32_t x29502 = x29490;
bool x29503 = x29502 == 2048;
if (x29503) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x29510 = x29490;
int32_t x29511 = 2048 / x29510;
bool x29517;
if (x452) {
bool x29512 = x29398 == 1;
bool x29513 = x29511 == 1;
bool x29514 = x29512 || x29513;
bool x29515 = x29398 == x29511;
bool x29516 = x29514 || x29515;
x29517 = x29516;
} else {
x29517 = false;
}
bool x29521;
if (x29517) {
x29521 = x29520;
} else {
x29521 = false;
}
bool x29522;
if (x29521) {
x29522 = x29520;
} else {
x29522 = false;
}
if (x29522) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x29398,x29400,x29400,1,x29511,1,1);
assert(false && "");
}
bool x29528 = x29398 <= x29511;
int32_t x29529;
if (x29528) {
x29529 = x29511;
} else {
x29529 = x29398;
}
bool x29535 = x29529 > 0;
bool x29537;
if (x29535) {
x29537 = x29536;
} else {
x29537 = false;
}
bool x29538;
if (x29537) {
x29538 = x29536;
} else {
x29538 = false;
}
if (x29538) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(29398) x Sym(29400) x Sym(29400)"," x Const(1) x Sym(29511) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x29533 = x29529 * x29532;
int32_t x29534 = 64 * x29533;
float* x29544 = (float*)myMalloc(x29534 * sizeof(float));;
int32_t x29545 = 0;
int32_t x29546 = 0;
int32_t x29547 = 0;
bool x29593 = x29398 > 1;
bool x29597 = x29511 > 1;
for(int x29548=0; x29548 < 64; x29548++) {
int32_t x29549 = x29546;
int32_t x29550 = x29547;
int32_t x29551 = x29545;
int32_t x29552 = x29551;
int32_t x29553 = x29549;
int32_t x29554 = x29550;
for(int x29556=0; x29556 < x29529; x29556++) {
int32_t x29557 = x29553;
int32_t x29558 = x29554;
int32_t x29559 = x29552;
int32_t x29560 = x29559;
int32_t x29561 = x29557;
int32_t x29562 = x29558;
for(int x29564=0; x29564 < x29531; x29564++) {
int32_t x29565 = x29561;
int32_t x29566 = x29562;
int32_t x29567 = x29560;
int32_t x29568 = x29567;
int32_t x29569 = x29565;
int32_t x29570 = x29566;
for(int x29571=0; x29571 < x29531; x29571++) {
int32_t x29572 = x29568;
int32_t x29573 = x29569;
float x29574 = x29413[x29573];
int32_t x29575 = x29570;
float x29576 = x29480[x29575];
float x29577 = x29574 / x29576;
x29544[x29572] = x29577;
x29568 += 1;
if (x29580) {
x29569 += 1;
} else {
}

}
x29560 += x29531;
if (x29580) {
x29561 += x29400;
} else {
}

}
x29552 += x29532;
if (x29593) {
x29553 += x29401;
} else {
}
if (x29597) {
x29554 += 1;
} else {
}

}
x29545 += x29533;
x29546 += x29402;

}
int32_t x29607 = 0;
int32_t x29608 = 1;
x29608 *= 1;
x29607 += 1;
x29608 *= 1;
x29608 *= 1;
int32_t x29613 = x29607;
bool x29614 = x29613 >= 2;
if (x29614) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x29619 = x29613 == 0;
if (x29619) {
int32_t x29620 = x29608;
bool x29621 = x29620 == 2048;
if (x29621) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x29628 = x29608;
int32_t x29629 = 2048 / x29628;
bool x29635;
if (x452) {
bool x29630 = x29529 == 1;
bool x29631 = x29629 == 1;
bool x29632 = x29630 || x29631;
bool x29633 = x29529 == x29629;
bool x29634 = x29632 || x29633;
x29635 = x29634;
} else {
x29635 = false;
}
bool x29639;
if (x29635) {
x29639 = x29638;
} else {
x29639 = false;
}
bool x29640;
if (x29639) {
x29640 = x29638;
} else {
x29640 = false;
}
if (x29640) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x29529,x29531,x29531,1,x29629,1,1);
assert(false && "");
}
bool x29646 = x29529 <= x29629;
int32_t x29647;
if (x29646) {
x29647 = x29629;
} else {
x29647 = x29529;
}
bool x29653 = x29647 > 0;
bool x29655;
if (x29653) {
x29655 = x29654;
} else {
x29655 = false;
}
bool x29656;
if (x29655) {
x29656 = x29654;
} else {
x29656 = false;
}
if (x29656) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(29529) x Sym(29531) x Sym(29531)"," x Const(1) x Sym(29629) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x29651 = x29647 * x29650;
int32_t x29652 = 64 * x29651;
float* x29662 = (float*)myMalloc(x29652 * sizeof(float));;
int32_t x29663 = 0;
int32_t x29664 = 0;
int32_t x29665 = 0;
bool x29711 = x29529 > 1;
bool x29715 = x29629 > 1;
for(int x29666=0; x29666 < 64; x29666++) {
int32_t x29667 = x29664;
int32_t x29668 = x29665;
int32_t x29669 = x29663;
int32_t x29670 = x29669;
int32_t x29671 = x29667;
int32_t x29672 = x29668;
for(int x29674=0; x29674 < x29647; x29674++) {
int32_t x29675 = x29671;
int32_t x29676 = x29672;
int32_t x29677 = x29670;
int32_t x29678 = x29677;
int32_t x29679 = x29675;
int32_t x29680 = x29676;
for(int x29682=0; x29682 < x29649; x29682++) {
int32_t x29683 = x29679;
int32_t x29684 = x29680;
int32_t x29685 = x29678;
int32_t x29686 = x29685;
int32_t x29687 = x29683;
int32_t x29688 = x29684;
for(int x29689=0; x29689 < x29649; x29689++) {
int32_t x29690 = x29686;
int32_t x29691 = x29687;
float x29692 = x29544[x29691];
int32_t x29693 = x29688;
float x29694 = x94[x29693];
float x29695 = x29692 * x29694;
x29662[x29690] = x29695;
x29686 += 1;
if (x29698) {
x29687 += 1;
} else {
}

}
x29678 += x29649;
if (x29698) {
x29679 += x29531;
} else {
}

}
x29670 += x29650;
if (x29711) {
x29671 += x29532;
} else {
}
if (x29715) {
x29672 += 1;
} else {
}

}
x29663 += x29651;
x29664 += x29533;

}
int32_t x29725 = 0;
int32_t x29726 = 1;
x29726 *= 1;
x29725 += 1;
x29726 *= 1;
x29726 *= 1;
int32_t x29731 = x29725;
bool x29732 = x29731 >= 2;
if (x29732) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x29737 = x29731 == 0;
if (x29737) {
int32_t x29738 = x29726;
bool x29739 = x29738 == 2048;
if (x29739) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x29746 = x29726;
int32_t x29747 = 2048 / x29746;
bool x29753;
if (x452) {
bool x29748 = x29647 == 1;
bool x29749 = x29747 == 1;
bool x29750 = x29748 || x29749;
bool x29751 = x29647 == x29747;
bool x29752 = x29750 || x29751;
x29753 = x29752;
} else {
x29753 = false;
}
bool x29757;
if (x29753) {
x29757 = x29756;
} else {
x29757 = false;
}
bool x29758;
if (x29757) {
x29758 = x29756;
} else {
x29758 = false;
}
if (x29758) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x29647,x29649,x29649,1,x29747,1,1);
assert(false && "");
}
bool x29764 = x29647 <= x29747;
int32_t x29765;
if (x29764) {
x29765 = x29747;
} else {
x29765 = x29647;
}
bool x29771 = x29765 > 0;
bool x29773;
if (x29771) {
x29773 = x29772;
} else {
x29773 = false;
}
bool x29774;
if (x29773) {
x29774 = x29772;
} else {
x29774 = false;
}
if (x29774) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(29647) x Sym(29649) x Sym(29649)"," x Const(1) x Sym(29747) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x29769 = x29765 * x29768;
int32_t x29770 = 64 * x29769;
float* x29780 = (float*)myMalloc(x29770 * sizeof(float));;
int32_t x29781 = 0;
int32_t x29782 = 0;
int32_t x29783 = 0;
bool x29829 = x29647 > 1;
bool x29833 = x29747 > 1;
for(int x29784=0; x29784 < 64; x29784++) {
int32_t x29785 = x29782;
int32_t x29786 = x29783;
int32_t x29787 = x29781;
int32_t x29788 = x29787;
int32_t x29789 = x29785;
int32_t x29790 = x29786;
for(int x29792=0; x29792 < x29765; x29792++) {
int32_t x29793 = x29789;
int32_t x29794 = x29790;
int32_t x29795 = x29788;
int32_t x29796 = x29795;
int32_t x29797 = x29793;
int32_t x29798 = x29794;
for(int x29800=0; x29800 < x29767; x29800++) {
int32_t x29801 = x29797;
int32_t x29802 = x29798;
int32_t x29803 = x29796;
int32_t x29804 = x29803;
int32_t x29805 = x29801;
int32_t x29806 = x29802;
for(int x29807=0; x29807 < x29767; x29807++) {
int32_t x29808 = x29804;
int32_t x29809 = x29805;
float x29810 = x29662[x29809];
int32_t x29811 = x29806;
float x29812 = x144[x29811];
float x29813 = x29810 + x29812;
x29780[x29808] = x29813;
x29804 += 1;
if (x29816) {
x29805 += 1;
} else {
}

}
x29796 += x29767;
if (x29816) {
x29797 += x29649;
} else {
}

}
x29788 += x29768;
if (x29829) {
x29789 += x29650;
} else {
}
if (x29833) {
x29790 += 1;
} else {
}

}
x29781 += x29769;
x29782 += x29651;

}
bool x29843 = x29765 == 1;
bool x29844 = x29843 || x28051;
bool x29845 = x29765 == x27431;
bool x29846 = x29844 || x29845;
bool x29851;
if (x29846) {
x29851 = x29850;
} else {
x29851 = false;
}
bool x29852;
if (x29851) {
x29852 = x29850;
} else {
x29852 = false;
}
if (x29852) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x29765,x29767,x29767,64,x27431,x27433,x27433);
assert(false && "");
}
int32_t x29865 = 0;
int32_t x29866 = 0;
int32_t x29867 = 0;
bool x29858 = x29765 <= x27431;
int32_t x29859;
if (x29858) {
x29859 = x27431;
} else {
x29859 = x29765;
}
bool x29918 = x29765 > 1;
int32_t x29863 = x29859 * x29862;
for(int x29868=0; x29868 < 64; x29868++) {
int32_t x29869 = x29866;
int32_t x29870 = x29867;
int32_t x29871 = x29865;
int32_t x29872 = x29871;
int32_t x29873 = x29869;
int32_t x29874 = x29870;
for(int x29876=0; x29876 < x29859; x29876++) {
int32_t x29877 = x29873;
int32_t x29878 = x29874;
int32_t x29879 = x29872;
int32_t x29880 = x29879;
int32_t x29881 = x29877;
int32_t x29882 = x29878;
for(int x29884=0; x29884 < x29861; x29884++) {
int32_t x29885 = x29881;
int32_t x29886 = x29882;
int32_t x29887 = x29880;
int32_t x29888 = x29887;
int32_t x29889 = x29885;
int32_t x29890 = x29886;
for(int x29891=0; x29891 < x29861; x29891++) {
int32_t x29892 = x29889;
float x29893 = x29780[x29892];
int32_t x29894 = x29890;
float x29895 = x28144[x29894];
float x29896 = x29893 + x29895;
x29780[x29892] = x29896;
x29888 += 1;
if (x29899) {
x29889 += 1;
} else {
}
if (x28109) {
x29890 += 1;
} else {
}

}
x29880 += x29861;
if (x29899) {
x29881 += x29767;
} else {
}
if (x28109) {
x29882 += x27433;
} else {
}

}
x29872 += x29862;
if (x29918) {
x29873 += x29768;
} else {
}
if (x28129) {
x29874 += x27434;
} else {
}

}
x29865 += x29863;
x29866 += x29769;
x29867 += x27435;

}
float* x29932 = (float*)myMalloc(x29770 * sizeof(float));;
for(int x29934=0; x29934 < x29770; x29934++) {
float x29935 = x29780[x29934];
bool x29936 = x29935 < 0.0f;
if (x29936) {
x29932[x29934] = 0.0f;
} else {
float x29939 = x29780[x29934];
x29932[x29934] = x29939;
}

}
float* x29953 = (float*)myMalloc(x29952 * sizeof(float));;
int32_t x29956 = 64 * x29765;
int32_t x29957 = x29956 * x29948;
float* x29958 = (float*)myMalloc(x29957 * sizeof(float));;
int32_t x29954 = x29765 * x29948;
for(int x29959=0; x29959 < 64; x29959++) {
int32_t x29960 = x29959 * x29769;
float* x29961 = x29932+x29960;
int32_t x29962 = x29959 * x29949;
float* x29963 = x29953+x29962;
int32_t x29964 = x29959 * x29954;
float* x29965 = x29958+x29964;
for(int x29966=0; x29966 < x29765; x29966++) {
int32_t x29967 = x29966 / 1;
int32_t x29971 = x29967 * x29947;
int32_t x29972 = x29971 * x29947;
int32_t x29968 = x29966 % 1;
int32_t x29969 = x29968 / 1;
int32_t x29973 = x29969 * x29947;
int32_t x29974 = x29973 * x29947;
int32_t x29975 = x29972 + x29974;
int32_t x29970 = x29968 % 1;
int32_t x29976 = x29970 * x29947;
int32_t x29977 = x29976 * x29947;
int32_t x29978 = x29975 + x29977;
float* x29979 = x29965+x29978;
int32_t x29980 = x29967 * x29767;
int32_t x29981 = x29980 * x29767;
float* x29982 = x29961+x29981;
for(int x29984=0; x29984 < x29947; x29984++) {
int32_t x29986 = x29984 * x29947;
float* x29987 = x29979+x29986;
int32_t x29985 = x29984 + x29969;
int32_t x29988 = x29985 * x29767;
int32_t x29989 = x29988 + x29970;
float* x29990 = x29982+x29989;
memcpy(x29987, x29990, 4 * x29947);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x29948,x29765,1,x265,x29765,x29965,x29948,1,x29963,x29948);

}
int32_t x29999 = 0;
int32_t x30000 = 1;
x30000 *= 1;
x29999 += 1;
x30000 *= 1;
x30000 *= 1;
int32_t x30005 = x29999;
bool x30006 = x30005 >= 2;
if (x30006) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x30011 = x30005 == 0;
if (x30011) {
int32_t x30012 = x30000;
bool x30013 = x30012 == 512;
if (x30013) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x30020 = x30000;
int32_t x30021 = 512 / x30020;
bool x30025;
if (x452) {
bool x30022 = x30021 == 1;
bool x30023 = 512 == x30021;
bool x30024 = x30022 || x30023;
x30025 = x30024;
} else {
x30025 = false;
}
bool x30029;
if (x30025) {
x30029 = x30028;
} else {
x30029 = false;
}
bool x30030;
if (x30029) {
x30030 = x30028;
} else {
x30030 = false;
}
if (x30030) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x29947,x29947,1,x30021,1,1);
assert(false && "");
}
bool x30036 = 512 <= x30021;
int32_t x30037;
if (x30036) {
x30037 = x30021;
} else {
x30037 = 512;
}
bool x30043 = x30037 > 0;
bool x30045;
if (x30043) {
x30045 = x30044;
} else {
x30045 = false;
}
bool x30046;
if (x30045) {
x30046 = x30044;
} else {
x30046 = false;
}
if (x30046) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(29947) x Sym(29947)"," x Const(1) x Sym(30021) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x30041 = x30037 * x30040;
int32_t x30042 = 64 * x30041;
float* x30052 = (float*)myMalloc(x30042 * sizeof(float));;
int32_t x30053 = 0;
int32_t x30054 = 0;
int32_t x30055 = 0;
bool x30102 = x30021 > 1;
for(int x30056=0; x30056 < 64; x30056++) {
int32_t x30057 = x30054;
int32_t x30058 = x30055;
int32_t x30059 = x30053;
int32_t x30060 = x30059;
int32_t x30061 = x30057;
int32_t x30062 = x30058;
for(int x30064=0; x30064 < x30037; x30064++) {
int32_t x30065 = x30061;
int32_t x30066 = x30062;
int32_t x30067 = x30060;
int32_t x30068 = x30067;
int32_t x30069 = x30065;
int32_t x30070 = x30066;
for(int x30072=0; x30072 < x30039; x30072++) {
int32_t x30073 = x30069;
int32_t x30074 = x30070;
int32_t x30075 = x30068;
int32_t x30076 = x30075;
int32_t x30077 = x30073;
int32_t x30078 = x30074;
for(int x30079=0; x30079 < x30039; x30079++) {
int32_t x30080 = x30076;
int32_t x30081 = x30077;
float x30082 = x29953[x30081];
int32_t x30083 = x30078;
float x30084 = x213[x30083];
float x30085 = x30082 - x30084;
x30052[x30080] = x30085;
x30076 += 1;
if (x30088) {
x30077 += 1;
} else {
}

}
x30068 += x30039;
if (x30088) {
x30069 += x29947;
} else {
}

}
x30060 += x30040;
x30061 += x29948;
if (x30102) {
x30062 += 1;
} else {
}

}
x30053 += x30041;
x30054 += x29949;

}
float* x30112 = (float*)myMalloc(512 * sizeof(float));;
for(int x30113=0; x30113 < 512; x30113++) {
float x30114 = x255[x30113];
float x30115 = x30114 + 1.0E-5f;
x30112[x30113] = x30115;

}
float* x30119 = (float*)myMalloc(512 * sizeof(float));;
for(int x30120=0; x30120 < 512; x30120++) {
float x30121 = x30112[x30120];
double x30122 = (double)x30121;
double x30123 = sqrt(x30122);
float x30124 = (float)x30123;
x30119[x30120] = x30124;

}
int32_t x30128 = 0;
int32_t x30129 = 1;
x30129 *= 1;
x30128 += 1;
x30129 *= 1;
x30129 *= 1;
int32_t x30134 = x30128;
bool x30135 = x30134 >= 2;
if (x30135) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x30140 = x30134 == 0;
if (x30140) {
int32_t x30141 = x30129;
bool x30142 = x30141 == 512;
if (x30142) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x30149 = x30129;
int32_t x30150 = 512 / x30149;
bool x30156;
if (x452) {
bool x30151 = x30037 == 1;
bool x30152 = x30150 == 1;
bool x30153 = x30151 || x30152;
bool x30154 = x30037 == x30150;
bool x30155 = x30153 || x30154;
x30156 = x30155;
} else {
x30156 = false;
}
bool x30160;
if (x30156) {
x30160 = x30159;
} else {
x30160 = false;
}
bool x30161;
if (x30160) {
x30161 = x30159;
} else {
x30161 = false;
}
if (x30161) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x30037,x30039,x30039,1,x30150,1,1);
assert(false && "");
}
bool x30167 = x30037 <= x30150;
int32_t x30168;
if (x30167) {
x30168 = x30150;
} else {
x30168 = x30037;
}
bool x30174 = x30168 > 0;
bool x30176;
if (x30174) {
x30176 = x30175;
} else {
x30176 = false;
}
bool x30177;
if (x30176) {
x30177 = x30175;
} else {
x30177 = false;
}
if (x30177) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(30037) x Sym(30039) x Sym(30039)"," x Const(1) x Sym(30150) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x30172 = x30168 * x30171;
int32_t x30173 = 64 * x30172;
float* x30183 = (float*)myMalloc(x30173 * sizeof(float));;
int32_t x30184 = 0;
int32_t x30185 = 0;
int32_t x30186 = 0;
bool x30232 = x30037 > 1;
bool x30236 = x30150 > 1;
for(int x30187=0; x30187 < 64; x30187++) {
int32_t x30188 = x30185;
int32_t x30189 = x30186;
int32_t x30190 = x30184;
int32_t x30191 = x30190;
int32_t x30192 = x30188;
int32_t x30193 = x30189;
for(int x30195=0; x30195 < x30168; x30195++) {
int32_t x30196 = x30192;
int32_t x30197 = x30193;
int32_t x30198 = x30191;
int32_t x30199 = x30198;
int32_t x30200 = x30196;
int32_t x30201 = x30197;
for(int x30203=0; x30203 < x30170; x30203++) {
int32_t x30204 = x30200;
int32_t x30205 = x30201;
int32_t x30206 = x30199;
int32_t x30207 = x30206;
int32_t x30208 = x30204;
int32_t x30209 = x30205;
for(int x30210=0; x30210 < x30170; x30210++) {
int32_t x30211 = x30207;
int32_t x30212 = x30208;
float x30213 = x30052[x30212];
int32_t x30214 = x30209;
float x30215 = x30119[x30214];
float x30216 = x30213 / x30215;
x30183[x30211] = x30216;
x30207 += 1;
if (x30219) {
x30208 += 1;
} else {
}

}
x30199 += x30170;
if (x30219) {
x30200 += x30039;
} else {
}

}
x30191 += x30171;
if (x30232) {
x30192 += x30040;
} else {
}
if (x30236) {
x30193 += 1;
} else {
}

}
x30184 += x30172;
x30185 += x30041;

}
int32_t x30246 = 0;
int32_t x30247 = 1;
x30247 *= 1;
x30246 += 1;
x30247 *= 1;
x30247 *= 1;
int32_t x30252 = x30246;
bool x30253 = x30252 >= 2;
if (x30253) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x30258 = x30252 == 0;
if (x30258) {
int32_t x30259 = x30247;
bool x30260 = x30259 == 512;
if (x30260) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x30267 = x30247;
int32_t x30268 = 512 / x30267;
bool x30274;
if (x452) {
bool x30269 = x30168 == 1;
bool x30270 = x30268 == 1;
bool x30271 = x30269 || x30270;
bool x30272 = x30168 == x30268;
bool x30273 = x30271 || x30272;
x30274 = x30273;
} else {
x30274 = false;
}
bool x30278;
if (x30274) {
x30278 = x30277;
} else {
x30278 = false;
}
bool x30279;
if (x30278) {
x30279 = x30277;
} else {
x30279 = false;
}
if (x30279) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x30168,x30170,x30170,1,x30268,1,1);
assert(false && "");
}
bool x30285 = x30168 <= x30268;
int32_t x30286;
if (x30285) {
x30286 = x30268;
} else {
x30286 = x30168;
}
bool x30292 = x30286 > 0;
bool x30294;
if (x30292) {
x30294 = x30293;
} else {
x30294 = false;
}
bool x30295;
if (x30294) {
x30295 = x30293;
} else {
x30295 = false;
}
if (x30295) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(30168) x Sym(30170) x Sym(30170)"," x Const(1) x Sym(30268) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x30290 = x30286 * x30289;
int32_t x30291 = 64 * x30290;
float* x30301 = (float*)myMalloc(x30291 * sizeof(float));;
int32_t x30302 = 0;
int32_t x30303 = 0;
int32_t x30304 = 0;
bool x30350 = x30168 > 1;
bool x30354 = x30268 > 1;
for(int x30305=0; x30305 < 64; x30305++) {
int32_t x30306 = x30303;
int32_t x30307 = x30304;
int32_t x30308 = x30302;
int32_t x30309 = x30308;
int32_t x30310 = x30306;
int32_t x30311 = x30307;
for(int x30313=0; x30313 < x30286; x30313++) {
int32_t x30314 = x30310;
int32_t x30315 = x30311;
int32_t x30316 = x30309;
int32_t x30317 = x30316;
int32_t x30318 = x30314;
int32_t x30319 = x30315;
for(int x30321=0; x30321 < x30288; x30321++) {
int32_t x30322 = x30318;
int32_t x30323 = x30319;
int32_t x30324 = x30317;
int32_t x30325 = x30324;
int32_t x30326 = x30322;
int32_t x30327 = x30323;
for(int x30328=0; x30328 < x30288; x30328++) {
int32_t x30329 = x30325;
int32_t x30330 = x30326;
float x30331 = x30183[x30330];
int32_t x30332 = x30327;
float x30333 = x15[x30332];
float x30334 = x30331 * x30333;
x30301[x30329] = x30334;
x30325 += 1;
if (x30337) {
x30326 += 1;
} else {
}

}
x30317 += x30288;
if (x30337) {
x30318 += x30170;
} else {
}

}
x30309 += x30289;
if (x30350) {
x30310 += x30171;
} else {
}
if (x30354) {
x30311 += 1;
} else {
}

}
x30302 += x30290;
x30303 += x30172;

}
int32_t x30364 = 0;
int32_t x30365 = 1;
x30365 *= 1;
x30364 += 1;
x30365 *= 1;
x30365 *= 1;
int32_t x30370 = x30364;
bool x30371 = x30370 >= 2;
if (x30371) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x30376 = x30370 == 0;
if (x30376) {
int32_t x30377 = x30365;
bool x30378 = x30377 == 512;
if (x30378) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x30385 = x30365;
int32_t x30386 = 512 / x30385;
bool x30392;
if (x452) {
bool x30387 = x30286 == 1;
bool x30388 = x30386 == 1;
bool x30389 = x30387 || x30388;
bool x30390 = x30286 == x30386;
bool x30391 = x30389 || x30390;
x30392 = x30391;
} else {
x30392 = false;
}
bool x30396;
if (x30392) {
x30396 = x30395;
} else {
x30396 = false;
}
bool x30397;
if (x30396) {
x30397 = x30395;
} else {
x30397 = false;
}
if (x30397) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x30286,x30288,x30288,1,x30386,1,1);
assert(false && "");
}
bool x30403 = x30286 <= x30386;
int32_t x30404;
if (x30403) {
x30404 = x30386;
} else {
x30404 = x30286;
}
bool x30410 = x30404 > 0;
bool x30412;
if (x30410) {
x30412 = x30411;
} else {
x30412 = false;
}
bool x30413;
if (x30412) {
x30413 = x30411;
} else {
x30413 = false;
}
if (x30413) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(30286) x Sym(30288) x Sym(30288)"," x Const(1) x Sym(30386) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x30408 = x30404 * x30407;
int32_t x30409 = 64 * x30408;
float* x30419 = (float*)myMalloc(x30409 * sizeof(float));;
int32_t x30420 = 0;
int32_t x30421 = 0;
int32_t x30422 = 0;
bool x30468 = x30286 > 1;
bool x30472 = x30386 > 1;
for(int x30423=0; x30423 < 64; x30423++) {
int32_t x30424 = x30421;
int32_t x30425 = x30422;
int32_t x30426 = x30420;
int32_t x30427 = x30426;
int32_t x30428 = x30424;
int32_t x30429 = x30425;
for(int x30431=0; x30431 < x30404; x30431++) {
int32_t x30432 = x30428;
int32_t x30433 = x30429;
int32_t x30434 = x30427;
int32_t x30435 = x30434;
int32_t x30436 = x30432;
int32_t x30437 = x30433;
for(int x30439=0; x30439 < x30406; x30439++) {
int32_t x30440 = x30436;
int32_t x30441 = x30437;
int32_t x30442 = x30435;
int32_t x30443 = x30442;
int32_t x30444 = x30440;
int32_t x30445 = x30441;
for(int x30446=0; x30446 < x30406; x30446++) {
int32_t x30447 = x30443;
int32_t x30448 = x30444;
float x30449 = x30301[x30448];
int32_t x30450 = x30445;
float x30451 = x78[x30450];
float x30452 = x30449 + x30451;
x30419[x30447] = x30452;
x30443 += 1;
if (x30455) {
x30444 += 1;
} else {
}

}
x30435 += x30406;
if (x30455) {
x30436 += x30288;
} else {
}

}
x30427 += x30407;
if (x30468) {
x30428 += x30289;
} else {
}
if (x30472) {
x30429 += 1;
} else {
}

}
x30420 += x30408;
x30421 += x30290;

}
float* x30482 = (float*)myMalloc(x30409 * sizeof(float));;
for(int x30484=0; x30484 < x30409; x30484++) {
float x30485 = x30419[x30484];
bool x30486 = x30485 < 0.0f;
if (x30486) {
x30482[x30484] = 0.0f;
} else {
float x30489 = x30419[x30484];
x30482[x30484] = x30489;
}

}
float* x30504 = (float*)myMalloc(x30503 * sizeof(float));;
int32_t x30505 = 9 * x30404;
int32_t x30508 = 64 * x30505;
int32_t x30509 = x30508 * x30499;
float* x30510 = (float*)myMalloc(x30509 * sizeof(float));;
int32_t x30506 = x30505 * x30499;
int32_t x30518 = x30404 * 3;
int32_t x30519 = x30518 * 3;
for(int x30511=0; x30511 < 64; x30511++) {
int32_t x30512 = x30511 * x30408;
float* x30513 = x30482+x30512;
int32_t x30514 = x30511 * x30500;
float* x30515 = x30504+x30514;
int32_t x30516 = x30511 * x30506;
float* x30517 = x30510+x30516;
for(int x30521=0; x30521 < x30519; x30521++) {
int32_t x30522 = x30521 / 9;
int32_t x30526 = x30522 * 3;
int32_t x30527 = x30526 * 3;
int32_t x30528 = x30527 * x30498;
int32_t x30529 = x30528 * x30498;
int32_t x30523 = x30521 % 9;
int32_t x30524 = x30523 / 3;
int32_t x30530 = x30524 * 3;
int32_t x30531 = x30530 * x30498;
int32_t x30532 = x30531 * x30498;
int32_t x30533 = x30529 + x30532;
int32_t x30525 = x30523 % 3;
int32_t x30534 = x30525 * x30498;
int32_t x30535 = x30534 * x30498;
int32_t x30536 = x30533 + x30535;
float* x30537 = x30517+x30536;
int32_t x30538 = x30522 * x30406;
int32_t x30539 = x30538 * x30406;
float* x30540 = x30513+x30539;
int32_t x30553 = 1 - x30525;
bool x30554 = x30553 > 0;
int32_t x30555;
if (x30554) {
x30555 = x30553;
} else {
x30555 = 0;
}
int32_t x30556 = 3 - x30525;
int32_t x30557 = x30556 - 1;
int32_t x30558 = 1 - x30557;
bool x30559 = x30558 > 0;
int32_t x30560;
if (x30559) {
x30560 = x30558;
} else {
x30560 = 0;
}
int32_t x30561 = x30498 - x30560;
int32_t x30562 = x30561 - x30555;
bool x30563 = x30562 <= 0;
bool x30567 = x30555 > 0;
int32_t x30552 = -1 + x30525;
bool x30580 = x30560 > 0;
for(int x30542=0; x30542 < x30498; x30542++) {
int32_t x30543 = x30542 - 1;
int32_t x30544 = x30543 + x30524;
bool x30545 = x30544 < 0;
bool x30546 = x30544 >= x30406;
bool x30547 = x30545 || x30546;
if (x30547) {
int32_t x30548 = x30542 * x30498;
float* x30549 = x30537+x30548;
memset(x30549, 0, 4 * x30498);;
} else {
if (x30563) {
int32_t x30548 = x30542 * x30498;
float* x30564 = x30537+x30548;
memset(x30564, 0, 4 * x30498);;
} else {
int32_t x30548 = x30542 * x30498;
if (x30567) {
float* x30568 = x30537+x30548;
memset(x30568, 0, 4 * x30555);;
} else {
}
// may have segfault here
int32_t x30573 = x30548 + x30555;
float* x30574 = x30537+x30573;
int32_t x30575 = x30544 * x30406;
int32_t x30576 = x30575 + x30552;
int32_t x30577 = x30576 + x30555;
float* x30578 = x30540+x30577;
memcpy(x30574, x30578, 4 * x30562);;
if (x30580) {
int32_t x30581 = x30548 + x30498;
int32_t x30582 = x30581 - x30560;
float* x30583 = x30537+x30582;
memset(x30583, 0, 4 * x30560);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x30499,x30505,1,x28,x30505,x30517,x30499,1,x30515,x30499);

}
int32_t x30598 = 0;
int32_t x30599 = 1;
x30599 *= 1;
x30598 += 1;
x30599 *= 1;
x30599 *= 1;
int32_t x30604 = x30598;
bool x30605 = x30604 >= 2;
if (x30605) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x30610 = x30604 == 0;
if (x30610) {
int32_t x30611 = x30599;
bool x30612 = x30611 == 512;
if (x30612) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x30619 = x30599;
int32_t x30620 = 512 / x30619;
bool x30624;
if (x452) {
bool x30621 = x30620 == 1;
bool x30622 = 512 == x30620;
bool x30623 = x30621 || x30622;
x30624 = x30623;
} else {
x30624 = false;
}
bool x30628;
if (x30624) {
x30628 = x30627;
} else {
x30628 = false;
}
bool x30629;
if (x30628) {
x30629 = x30627;
} else {
x30629 = false;
}
if (x30629) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,512,x30498,x30498,1,x30620,1,1);
assert(false && "");
}
bool x30635 = 512 <= x30620;
int32_t x30636;
if (x30635) {
x30636 = x30620;
} else {
x30636 = 512;
}
bool x30642 = x30636 > 0;
bool x30644;
if (x30642) {
x30644 = x30643;
} else {
x30644 = false;
}
bool x30645;
if (x30644) {
x30645 = x30643;
} else {
x30645 = false;
}
if (x30645) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(512) x Sym(30498) x Sym(30498)"," x Const(1) x Sym(30620) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x30640 = x30636 * x30639;
int32_t x30641 = 64 * x30640;
float* x30651 = (float*)myMalloc(x30641 * sizeof(float));;
int32_t x30652 = 0;
int32_t x30653 = 0;
int32_t x30654 = 0;
bool x30701 = x30620 > 1;
for(int x30655=0; x30655 < 64; x30655++) {
int32_t x30656 = x30653;
int32_t x30657 = x30654;
int32_t x30658 = x30652;
int32_t x30659 = x30658;
int32_t x30660 = x30656;
int32_t x30661 = x30657;
for(int x30663=0; x30663 < x30636; x30663++) {
int32_t x30664 = x30660;
int32_t x30665 = x30661;
int32_t x30666 = x30659;
int32_t x30667 = x30666;
int32_t x30668 = x30664;
int32_t x30669 = x30665;
for(int x30671=0; x30671 < x30638; x30671++) {
int32_t x30672 = x30668;
int32_t x30673 = x30669;
int32_t x30674 = x30667;
int32_t x30675 = x30674;
int32_t x30676 = x30672;
int32_t x30677 = x30673;
for(int x30678=0; x30678 < x30638; x30678++) {
int32_t x30679 = x30675;
int32_t x30680 = x30676;
float x30681 = x30504[x30680];
int32_t x30682 = x30677;
float x30683 = x12[x30682];
float x30684 = x30681 - x30683;
x30651[x30679] = x30684;
x30675 += 1;
if (x30687) {
x30676 += 1;
} else {
}

}
x30667 += x30638;
if (x30687) {
x30668 += x30498;
} else {
}

}
x30659 += x30639;
x30660 += x30499;
if (x30701) {
x30661 += 1;
} else {
}

}
x30652 += x30640;
x30653 += x30500;

}
float* x30711 = (float*)myMalloc(512 * sizeof(float));;
for(int x30712=0; x30712 < 512; x30712++) {
float x30713 = x202[x30712];
float x30714 = x30713 + 1.0E-5f;
x30711[x30712] = x30714;

}
float* x30718 = (float*)myMalloc(512 * sizeof(float));;
for(int x30719=0; x30719 < 512; x30719++) {
float x30720 = x30711[x30719];
double x30721 = (double)x30720;
double x30722 = sqrt(x30721);
float x30723 = (float)x30722;
x30718[x30719] = x30723;

}
int32_t x30727 = 0;
int32_t x30728 = 1;
x30728 *= 1;
x30727 += 1;
x30728 *= 1;
x30728 *= 1;
int32_t x30733 = x30727;
bool x30734 = x30733 >= 2;
if (x30734) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x30739 = x30733 == 0;
if (x30739) {
int32_t x30740 = x30728;
bool x30741 = x30740 == 512;
if (x30741) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x30748 = x30728;
int32_t x30749 = 512 / x30748;
bool x30755;
if (x452) {
bool x30750 = x30636 == 1;
bool x30751 = x30749 == 1;
bool x30752 = x30750 || x30751;
bool x30753 = x30636 == x30749;
bool x30754 = x30752 || x30753;
x30755 = x30754;
} else {
x30755 = false;
}
bool x30759;
if (x30755) {
x30759 = x30758;
} else {
x30759 = false;
}
bool x30760;
if (x30759) {
x30760 = x30758;
} else {
x30760 = false;
}
if (x30760) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x30636,x30638,x30638,1,x30749,1,1);
assert(false && "");
}
bool x30766 = x30636 <= x30749;
int32_t x30767;
if (x30766) {
x30767 = x30749;
} else {
x30767 = x30636;
}
bool x30773 = x30767 > 0;
bool x30775;
if (x30773) {
x30775 = x30774;
} else {
x30775 = false;
}
bool x30776;
if (x30775) {
x30776 = x30774;
} else {
x30776 = false;
}
if (x30776) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(30636) x Sym(30638) x Sym(30638)"," x Const(1) x Sym(30749) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x30771 = x30767 * x30770;
int32_t x30772 = 64 * x30771;
float* x30782 = (float*)myMalloc(x30772 * sizeof(float));;
int32_t x30783 = 0;
int32_t x30784 = 0;
int32_t x30785 = 0;
bool x30831 = x30636 > 1;
bool x30835 = x30749 > 1;
for(int x30786=0; x30786 < 64; x30786++) {
int32_t x30787 = x30784;
int32_t x30788 = x30785;
int32_t x30789 = x30783;
int32_t x30790 = x30789;
int32_t x30791 = x30787;
int32_t x30792 = x30788;
for(int x30794=0; x30794 < x30767; x30794++) {
int32_t x30795 = x30791;
int32_t x30796 = x30792;
int32_t x30797 = x30790;
int32_t x30798 = x30797;
int32_t x30799 = x30795;
int32_t x30800 = x30796;
for(int x30802=0; x30802 < x30769; x30802++) {
int32_t x30803 = x30799;
int32_t x30804 = x30800;
int32_t x30805 = x30798;
int32_t x30806 = x30805;
int32_t x30807 = x30803;
int32_t x30808 = x30804;
for(int x30809=0; x30809 < x30769; x30809++) {
int32_t x30810 = x30806;
int32_t x30811 = x30807;
float x30812 = x30651[x30811];
int32_t x30813 = x30808;
float x30814 = x30718[x30813];
float x30815 = x30812 / x30814;
x30782[x30810] = x30815;
x30806 += 1;
if (x30818) {
x30807 += 1;
} else {
}

}
x30798 += x30769;
if (x30818) {
x30799 += x30638;
} else {
}

}
x30790 += x30770;
if (x30831) {
x30791 += x30639;
} else {
}
if (x30835) {
x30792 += 1;
} else {
}

}
x30783 += x30771;
x30784 += x30640;

}
int32_t x30845 = 0;
int32_t x30846 = 1;
x30846 *= 1;
x30845 += 1;
x30846 *= 1;
x30846 *= 1;
int32_t x30851 = x30845;
bool x30852 = x30851 >= 2;
if (x30852) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x30857 = x30851 == 0;
if (x30857) {
int32_t x30858 = x30846;
bool x30859 = x30858 == 512;
if (x30859) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x30866 = x30846;
int32_t x30867 = 512 / x30866;
bool x30873;
if (x452) {
bool x30868 = x30767 == 1;
bool x30869 = x30867 == 1;
bool x30870 = x30868 || x30869;
bool x30871 = x30767 == x30867;
bool x30872 = x30870 || x30871;
x30873 = x30872;
} else {
x30873 = false;
}
bool x30877;
if (x30873) {
x30877 = x30876;
} else {
x30877 = false;
}
bool x30878;
if (x30877) {
x30878 = x30876;
} else {
x30878 = false;
}
if (x30878) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x30767,x30769,x30769,1,x30867,1,1);
assert(false && "");
}
bool x30884 = x30767 <= x30867;
int32_t x30885;
if (x30884) {
x30885 = x30867;
} else {
x30885 = x30767;
}
bool x30891 = x30885 > 0;
bool x30893;
if (x30891) {
x30893 = x30892;
} else {
x30893 = false;
}
bool x30894;
if (x30893) {
x30894 = x30892;
} else {
x30894 = false;
}
if (x30894) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(30767) x Sym(30769) x Sym(30769)"," x Const(1) x Sym(30867) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x30889 = x30885 * x30888;
int32_t x30890 = 64 * x30889;
float* x30900 = (float*)myMalloc(x30890 * sizeof(float));;
int32_t x30901 = 0;
int32_t x30902 = 0;
int32_t x30903 = 0;
bool x30949 = x30767 > 1;
bool x30953 = x30867 > 1;
for(int x30904=0; x30904 < 64; x30904++) {
int32_t x30905 = x30902;
int32_t x30906 = x30903;
int32_t x30907 = x30901;
int32_t x30908 = x30907;
int32_t x30909 = x30905;
int32_t x30910 = x30906;
for(int x30912=0; x30912 < x30885; x30912++) {
int32_t x30913 = x30909;
int32_t x30914 = x30910;
int32_t x30915 = x30908;
int32_t x30916 = x30915;
int32_t x30917 = x30913;
int32_t x30918 = x30914;
for(int x30920=0; x30920 < x30887; x30920++) {
int32_t x30921 = x30917;
int32_t x30922 = x30918;
int32_t x30923 = x30916;
int32_t x30924 = x30923;
int32_t x30925 = x30921;
int32_t x30926 = x30922;
for(int x30927=0; x30927 < x30887; x30927++) {
int32_t x30928 = x30924;
int32_t x30929 = x30925;
float x30930 = x30782[x30929];
int32_t x30931 = x30926;
float x30932 = x194[x30931];
float x30933 = x30930 * x30932;
x30900[x30928] = x30933;
x30924 += 1;
if (x30936) {
x30925 += 1;
} else {
}

}
x30916 += x30887;
if (x30936) {
x30917 += x30769;
} else {
}

}
x30908 += x30888;
if (x30949) {
x30909 += x30770;
} else {
}
if (x30953) {
x30910 += 1;
} else {
}

}
x30901 += x30889;
x30902 += x30771;

}
int32_t x30963 = 0;
int32_t x30964 = 1;
x30964 *= 1;
x30963 += 1;
x30964 *= 1;
x30964 *= 1;
int32_t x30969 = x30963;
bool x30970 = x30969 >= 2;
if (x30970) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x30975 = x30969 == 0;
if (x30975) {
int32_t x30976 = x30964;
bool x30977 = x30976 == 512;
if (x30977) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x30984 = x30964;
int32_t x30985 = 512 / x30984;
bool x30991;
if (x452) {
bool x30986 = x30885 == 1;
bool x30987 = x30985 == 1;
bool x30988 = x30986 || x30987;
bool x30989 = x30885 == x30985;
bool x30990 = x30988 || x30989;
x30991 = x30990;
} else {
x30991 = false;
}
bool x30995;
if (x30991) {
x30995 = x30994;
} else {
x30995 = false;
}
bool x30996;
if (x30995) {
x30996 = x30994;
} else {
x30996 = false;
}
if (x30996) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x30885,x30887,x30887,1,x30985,1,1);
assert(false && "");
}
bool x31002 = x30885 <= x30985;
int32_t x31003;
if (x31002) {
x31003 = x30985;
} else {
x31003 = x30885;
}
bool x31009 = x31003 > 0;
bool x31011;
if (x31009) {
x31011 = x31010;
} else {
x31011 = false;
}
bool x31012;
if (x31011) {
x31012 = x31010;
} else {
x31012 = false;
}
if (x31012) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(30885) x Sym(30887) x Sym(30887)"," x Const(1) x Sym(30985) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x31007 = x31003 * x31006;
int32_t x31008 = 64 * x31007;
float* x31018 = (float*)myMalloc(x31008 * sizeof(float));;
int32_t x31019 = 0;
int32_t x31020 = 0;
int32_t x31021 = 0;
bool x31067 = x30885 > 1;
bool x31071 = x30985 > 1;
for(int x31022=0; x31022 < 64; x31022++) {
int32_t x31023 = x31020;
int32_t x31024 = x31021;
int32_t x31025 = x31019;
int32_t x31026 = x31025;
int32_t x31027 = x31023;
int32_t x31028 = x31024;
for(int x31030=0; x31030 < x31003; x31030++) {
int32_t x31031 = x31027;
int32_t x31032 = x31028;
int32_t x31033 = x31026;
int32_t x31034 = x31033;
int32_t x31035 = x31031;
int32_t x31036 = x31032;
for(int x31038=0; x31038 < x31005; x31038++) {
int32_t x31039 = x31035;
int32_t x31040 = x31036;
int32_t x31041 = x31034;
int32_t x31042 = x31041;
int32_t x31043 = x31039;
int32_t x31044 = x31040;
for(int x31045=0; x31045 < x31005; x31045++) {
int32_t x31046 = x31042;
int32_t x31047 = x31043;
float x31048 = x30900[x31047];
int32_t x31049 = x31044;
float x31050 = x169[x31049];
float x31051 = x31048 + x31050;
x31018[x31046] = x31051;
x31042 += 1;
if (x31054) {
x31043 += 1;
} else {
}

}
x31034 += x31005;
if (x31054) {
x31035 += x30887;
} else {
}

}
x31026 += x31006;
if (x31067) {
x31027 += x30888;
} else {
}
if (x31071) {
x31028 += 1;
} else {
}

}
x31019 += x31007;
x31020 += x30889;

}
float* x31081 = (float*)myMalloc(x31008 * sizeof(float));;
for(int x31083=0; x31083 < x31008; x31083++) {
float x31084 = x31018[x31083];
bool x31085 = x31084 < 0.0f;
if (x31085) {
x31081[x31083] = 0.0f;
} else {
float x31088 = x31018[x31083];
x31081[x31083] = x31088;
}

}
float* x31102 = (float*)myMalloc(x31101 * sizeof(float));;
int32_t x31105 = 64 * x31003;
int32_t x31106 = x31105 * x31097;
float* x31107 = (float*)myMalloc(x31106 * sizeof(float));;
int32_t x31103 = x31003 * x31097;
for(int x31108=0; x31108 < 64; x31108++) {
int32_t x31109 = x31108 * x31007;
float* x31110 = x31081+x31109;
int32_t x31111 = x31108 * x31098;
float* x31112 = x31102+x31111;
int32_t x31113 = x31108 * x31103;
float* x31114 = x31107+x31113;
for(int x31115=0; x31115 < x31003; x31115++) {
int32_t x31116 = x31115 / 1;
int32_t x31120 = x31116 * x31096;
int32_t x31121 = x31120 * x31096;
int32_t x31117 = x31115 % 1;
int32_t x31118 = x31117 / 1;
int32_t x31122 = x31118 * x31096;
int32_t x31123 = x31122 * x31096;
int32_t x31124 = x31121 + x31123;
int32_t x31119 = x31117 % 1;
int32_t x31125 = x31119 * x31096;
int32_t x31126 = x31125 * x31096;
int32_t x31127 = x31124 + x31126;
float* x31128 = x31114+x31127;
int32_t x31129 = x31116 * x31005;
int32_t x31130 = x31129 * x31005;
float* x31131 = x31110+x31130;
for(int x31133=0; x31133 < x31096; x31133++) {
int32_t x31135 = x31133 * x31096;
float* x31136 = x31128+x31135;
int32_t x31134 = x31133 + x31118;
int32_t x31137 = x31134 * x31005;
int32_t x31138 = x31137 + x31119;
float* x31139 = x31131+x31138;
memcpy(x31136, x31139, 4 * x31096);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x31097,x31003,1,x33,x31003,x31114,x31097,1,x31112,x31097);

}
int32_t x31148 = 0;
int32_t x31149 = 1;
x31149 *= 1;
x31148 += 1;
x31149 *= 1;
x31149 *= 1;
int32_t x31154 = x31148;
bool x31155 = x31154 >= 2;
if (x31155) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x31160 = x31154 == 0;
if (x31160) {
int32_t x31161 = x31149;
bool x31162 = x31161 == 2048;
if (x31162) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x31169 = x31149;
int32_t x31170 = 2048 / x31169;
bool x31174;
if (x452) {
bool x31171 = x31170 == 1;
bool x31172 = 2048 == x31170;
bool x31173 = x31171 || x31172;
x31174 = x31173;
} else {
x31174 = false;
}
bool x31178;
if (x31174) {
x31178 = x31177;
} else {
x31178 = false;
}
bool x31179;
if (x31178) {
x31179 = x31177;
} else {
x31179 = false;
}
if (x31179) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,2048,x31096,x31096,1,x31170,1,1);
assert(false && "");
}
bool x31185 = 2048 <= x31170;
int32_t x31186;
if (x31185) {
x31186 = x31170;
} else {
x31186 = 2048;
}
bool x31192 = x31186 > 0;
bool x31194;
if (x31192) {
x31194 = x31193;
} else {
x31194 = false;
}
bool x31195;
if (x31194) {
x31195 = x31193;
} else {
x31195 = false;
}
if (x31195) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(2048) x Sym(31096) x Sym(31096)"," x Const(1) x Sym(31170) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x31190 = x31186 * x31189;
int32_t x31191 = 64 * x31190;
float* x31201 = (float*)myMalloc(x31191 * sizeof(float));;
int32_t x31202 = 0;
int32_t x31203 = 0;
int32_t x31204 = 0;
bool x31251 = x31170 > 1;
for(int x31205=0; x31205 < 64; x31205++) {
int32_t x31206 = x31203;
int32_t x31207 = x31204;
int32_t x31208 = x31202;
int32_t x31209 = x31208;
int32_t x31210 = x31206;
int32_t x31211 = x31207;
for(int x31213=0; x31213 < x31186; x31213++) {
int32_t x31214 = x31210;
int32_t x31215 = x31211;
int32_t x31216 = x31209;
int32_t x31217 = x31216;
int32_t x31218 = x31214;
int32_t x31219 = x31215;
for(int x31221=0; x31221 < x31188; x31221++) {
int32_t x31222 = x31218;
int32_t x31223 = x31219;
int32_t x31224 = x31217;
int32_t x31225 = x31224;
int32_t x31226 = x31222;
int32_t x31227 = x31223;
for(int x31228=0; x31228 < x31188; x31228++) {
int32_t x31229 = x31225;
int32_t x31230 = x31226;
float x31231 = x31102[x31230];
int32_t x31232 = x31227;
float x31233 = x260[x31232];
float x31234 = x31231 - x31233;
x31201[x31229] = x31234;
x31225 += 1;
if (x31237) {
x31226 += 1;
} else {
}

}
x31217 += x31188;
if (x31237) {
x31218 += x31096;
} else {
}

}
x31209 += x31189;
x31210 += x31097;
if (x31251) {
x31211 += 1;
} else {
}

}
x31202 += x31190;
x31203 += x31098;

}
float* x31261 = (float*)myMalloc(2048 * sizeof(float));;
for(int x31262=0; x31262 < 2048; x31262++) {
float x31263 = x123[x31262];
float x31264 = x31263 + 1.0E-5f;
x31261[x31262] = x31264;

}
float* x31268 = (float*)myMalloc(2048 * sizeof(float));;
for(int x31269=0; x31269 < 2048; x31269++) {
float x31270 = x31261[x31269];
double x31271 = (double)x31270;
double x31272 = sqrt(x31271);
float x31273 = (float)x31272;
x31268[x31269] = x31273;

}
int32_t x31277 = 0;
int32_t x31278 = 1;
x31278 *= 1;
x31277 += 1;
x31278 *= 1;
x31278 *= 1;
int32_t x31283 = x31277;
bool x31284 = x31283 >= 2;
if (x31284) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x31289 = x31283 == 0;
if (x31289) {
int32_t x31290 = x31278;
bool x31291 = x31290 == 2048;
if (x31291) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x31298 = x31278;
int32_t x31299 = 2048 / x31298;
bool x31305;
if (x452) {
bool x31300 = x31186 == 1;
bool x31301 = x31299 == 1;
bool x31302 = x31300 || x31301;
bool x31303 = x31186 == x31299;
bool x31304 = x31302 || x31303;
x31305 = x31304;
} else {
x31305 = false;
}
bool x31309;
if (x31305) {
x31309 = x31308;
} else {
x31309 = false;
}
bool x31310;
if (x31309) {
x31310 = x31308;
} else {
x31310 = false;
}
if (x31310) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x31186,x31188,x31188,1,x31299,1,1);
assert(false && "");
}
bool x31316 = x31186 <= x31299;
int32_t x31317;
if (x31316) {
x31317 = x31299;
} else {
x31317 = x31186;
}
bool x31323 = x31317 > 0;
bool x31325;
if (x31323) {
x31325 = x31324;
} else {
x31325 = false;
}
bool x31326;
if (x31325) {
x31326 = x31324;
} else {
x31326 = false;
}
if (x31326) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(31186) x Sym(31188) x Sym(31188)"," x Const(1) x Sym(31299) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x31321 = x31317 * x31320;
int32_t x31322 = 64 * x31321;
float* x31332 = (float*)myMalloc(x31322 * sizeof(float));;
int32_t x31333 = 0;
int32_t x31334 = 0;
int32_t x31335 = 0;
bool x31381 = x31186 > 1;
bool x31385 = x31299 > 1;
for(int x31336=0; x31336 < 64; x31336++) {
int32_t x31337 = x31334;
int32_t x31338 = x31335;
int32_t x31339 = x31333;
int32_t x31340 = x31339;
int32_t x31341 = x31337;
int32_t x31342 = x31338;
for(int x31344=0; x31344 < x31317; x31344++) {
int32_t x31345 = x31341;
int32_t x31346 = x31342;
int32_t x31347 = x31340;
int32_t x31348 = x31347;
int32_t x31349 = x31345;
int32_t x31350 = x31346;
for(int x31352=0; x31352 < x31319; x31352++) {
int32_t x31353 = x31349;
int32_t x31354 = x31350;
int32_t x31355 = x31348;
int32_t x31356 = x31355;
int32_t x31357 = x31353;
int32_t x31358 = x31354;
for(int x31359=0; x31359 < x31319; x31359++) {
int32_t x31360 = x31356;
int32_t x31361 = x31357;
float x31362 = x31201[x31361];
int32_t x31363 = x31358;
float x31364 = x31268[x31363];
float x31365 = x31362 / x31364;
x31332[x31360] = x31365;
x31356 += 1;
if (x31368) {
x31357 += 1;
} else {
}

}
x31348 += x31319;
if (x31368) {
x31349 += x31188;
} else {
}

}
x31340 += x31320;
if (x31381) {
x31341 += x31189;
} else {
}
if (x31385) {
x31342 += 1;
} else {
}

}
x31333 += x31321;
x31334 += x31190;

}
int32_t x31395 = 0;
int32_t x31396 = 1;
x31396 *= 1;
x31395 += 1;
x31396 *= 1;
x31396 *= 1;
int32_t x31401 = x31395;
bool x31402 = x31401 >= 2;
if (x31402) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x31407 = x31401 == 0;
if (x31407) {
int32_t x31408 = x31396;
bool x31409 = x31408 == 2048;
if (x31409) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x31416 = x31396;
int32_t x31417 = 2048 / x31416;
bool x31423;
if (x452) {
bool x31418 = x31317 == 1;
bool x31419 = x31417 == 1;
bool x31420 = x31418 || x31419;
bool x31421 = x31317 == x31417;
bool x31422 = x31420 || x31421;
x31423 = x31422;
} else {
x31423 = false;
}
bool x31427;
if (x31423) {
x31427 = x31426;
} else {
x31427 = false;
}
bool x31428;
if (x31427) {
x31428 = x31426;
} else {
x31428 = false;
}
if (x31428) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x31317,x31319,x31319,1,x31417,1,1);
assert(false && "");
}
bool x31434 = x31317 <= x31417;
int32_t x31435;
if (x31434) {
x31435 = x31417;
} else {
x31435 = x31317;
}
bool x31441 = x31435 > 0;
bool x31443;
if (x31441) {
x31443 = x31442;
} else {
x31443 = false;
}
bool x31444;
if (x31443) {
x31444 = x31442;
} else {
x31444 = false;
}
if (x31444) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(31317) x Sym(31319) x Sym(31319)"," x Const(1) x Sym(31417) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x31439 = x31435 * x31438;
int32_t x31440 = 64 * x31439;
float* x31450 = (float*)myMalloc(x31440 * sizeof(float));;
int32_t x31451 = 0;
int32_t x31452 = 0;
int32_t x31453 = 0;
bool x31499 = x31317 > 1;
bool x31503 = x31417 > 1;
for(int x31454=0; x31454 < 64; x31454++) {
int32_t x31455 = x31452;
int32_t x31456 = x31453;
int32_t x31457 = x31451;
int32_t x31458 = x31457;
int32_t x31459 = x31455;
int32_t x31460 = x31456;
for(int x31462=0; x31462 < x31435; x31462++) {
int32_t x31463 = x31459;
int32_t x31464 = x31460;
int32_t x31465 = x31458;
int32_t x31466 = x31465;
int32_t x31467 = x31463;
int32_t x31468 = x31464;
for(int x31470=0; x31470 < x31437; x31470++) {
int32_t x31471 = x31467;
int32_t x31472 = x31468;
int32_t x31473 = x31466;
int32_t x31474 = x31473;
int32_t x31475 = x31471;
int32_t x31476 = x31472;
for(int x31477=0; x31477 < x31437; x31477++) {
int32_t x31478 = x31474;
int32_t x31479 = x31475;
float x31480 = x31332[x31479];
int32_t x31481 = x31476;
float x31482 = x103[x31481];
float x31483 = x31480 * x31482;
x31450[x31478] = x31483;
x31474 += 1;
if (x31486) {
x31475 += 1;
} else {
}

}
x31466 += x31437;
if (x31486) {
x31467 += x31319;
} else {
}

}
x31458 += x31438;
if (x31499) {
x31459 += x31320;
} else {
}
if (x31503) {
x31460 += 1;
} else {
}

}
x31451 += x31439;
x31452 += x31321;

}
int32_t x31513 = 0;
int32_t x31514 = 1;
x31514 *= 1;
x31513 += 1;
x31514 *= 1;
x31514 *= 1;
int32_t x31519 = x31513;
bool x31520 = x31519 >= 2;
if (x31520) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x31525 = x31519 == 0;
if (x31525) {
int32_t x31526 = x31514;
bool x31527 = x31526 == 2048;
if (x31527) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x31534 = x31514;
int32_t x31535 = 2048 / x31534;
bool x31541;
if (x452) {
bool x31536 = x31435 == 1;
bool x31537 = x31535 == 1;
bool x31538 = x31536 || x31537;
bool x31539 = x31435 == x31535;
bool x31540 = x31538 || x31539;
x31541 = x31540;
} else {
x31541 = false;
}
bool x31545;
if (x31541) {
x31545 = x31544;
} else {
x31545 = false;
}
bool x31546;
if (x31545) {
x31546 = x31544;
} else {
x31546 = false;
}
if (x31546) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x31435,x31437,x31437,1,x31535,1,1);
assert(false && "");
}
bool x31552 = x31435 <= x31535;
int32_t x31553;
if (x31552) {
x31553 = x31535;
} else {
x31553 = x31435;
}
bool x31559 = x31553 > 0;
bool x31561;
if (x31559) {
x31561 = x31560;
} else {
x31561 = false;
}
bool x31562;
if (x31561) {
x31562 = x31560;
} else {
x31562 = false;
}
if (x31562) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Sym(31435) x Sym(31437) x Sym(31437)"," x Const(1) x Sym(31535) x Const(1) x Const(1)");
assert(false && "");
}
int32_t x31557 = x31553 * x31556;
int32_t x31558 = 64 * x31557;
float* x31568 = (float*)myMalloc(x31558 * sizeof(float));;
int32_t x31569 = 0;
int32_t x31570 = 0;
int32_t x31571 = 0;
bool x31617 = x31435 > 1;
bool x31621 = x31535 > 1;
for(int x31572=0; x31572 < 64; x31572++) {
int32_t x31573 = x31570;
int32_t x31574 = x31571;
int32_t x31575 = x31569;
int32_t x31576 = x31575;
int32_t x31577 = x31573;
int32_t x31578 = x31574;
for(int x31580=0; x31580 < x31553; x31580++) {
int32_t x31581 = x31577;
int32_t x31582 = x31578;
int32_t x31583 = x31576;
int32_t x31584 = x31583;
int32_t x31585 = x31581;
int32_t x31586 = x31582;
for(int x31588=0; x31588 < x31555; x31588++) {
int32_t x31589 = x31585;
int32_t x31590 = x31586;
int32_t x31591 = x31584;
int32_t x31592 = x31591;
int32_t x31593 = x31589;
int32_t x31594 = x31590;
for(int x31595=0; x31595 < x31555; x31595++) {
int32_t x31596 = x31592;
int32_t x31597 = x31593;
float x31598 = x31450[x31597];
int32_t x31599 = x31594;
float x31600 = x181[x31599];
float x31601 = x31598 + x31600;
x31568[x31596] = x31601;
x31592 += 1;
if (x31604) {
x31593 += 1;
} else {
}

}
x31584 += x31555;
if (x31604) {
x31585 += x31437;
} else {
}

}
x31576 += x31556;
if (x31617) {
x31577 += x31438;
} else {
}
if (x31621) {
x31578 += 1;
} else {
}

}
x31569 += x31557;
x31570 += x31439;

}
bool x31631 = x31553 == 1;
bool x31632 = x31631 || x29843;
bool x31633 = x31553 == x29765;
bool x31634 = x31632 || x31633;
bool x31639;
if (x31634) {
x31639 = x31638;
} else {
x31639 = false;
}
bool x31640;
if (x31639) {
x31640 = x31638;
} else {
x31640 = false;
}
if (x31640) {
} else {
printf("dimensions not compatible for broadcasting %d,%d,%d,%d, with %d,%d,%d,%d,\n",64,x31553,x31555,x31555,64,x29765,x29767,x29767);
assert(false && "");
}
int32_t x31653 = 0;
int32_t x31654 = 0;
int32_t x31655 = 0;
bool x31646 = x31553 <= x29765;
int32_t x31647;
if (x31646) {
x31647 = x29765;
} else {
x31647 = x31553;
}
bool x31706 = x31553 > 1;
int32_t x31651 = x31647 * x31650;
for(int x31656=0; x31656 < 64; x31656++) {
int32_t x31657 = x31654;
int32_t x31658 = x31655;
int32_t x31659 = x31653;
int32_t x31660 = x31659;
int32_t x31661 = x31657;
int32_t x31662 = x31658;
for(int x31664=0; x31664 < x31647; x31664++) {
int32_t x31665 = x31661;
int32_t x31666 = x31662;
int32_t x31667 = x31660;
int32_t x31668 = x31667;
int32_t x31669 = x31665;
int32_t x31670 = x31666;
for(int x31672=0; x31672 < x31649; x31672++) {
int32_t x31673 = x31669;
int32_t x31674 = x31670;
int32_t x31675 = x31668;
int32_t x31676 = x31675;
int32_t x31677 = x31673;
int32_t x31678 = x31674;
for(int x31679=0; x31679 < x31649; x31679++) {
int32_t x31680 = x31677;
float x31681 = x31568[x31680];
int32_t x31682 = x31678;
float x31683 = x29932[x31682];
float x31684 = x31681 + x31683;
x31568[x31680] = x31684;
x31676 += 1;
if (x31687) {
x31677 += 1;
} else {
}
if (x29899) {
x31678 += 1;
} else {
}

}
x31668 += x31649;
if (x31687) {
x31669 += x31555;
} else {
}
if (x29899) {
x31670 += x29767;
} else {
}

}
x31660 += x31650;
if (x31706) {
x31661 += x31556;
} else {
}
if (x29918) {
x31662 += x29768;
} else {
}

}
x31653 += x31651;
x31654 += x31557;
x31655 += x29769;

}
float* x31720 = (float*)myMalloc(x31558 * sizeof(float));;
for(int x31722=0; x31722 < x31558; x31722++) {
float x31723 = x31568[x31722];
bool x31724 = x31723 < 0.0f;
if (x31724) {
x31720[x31722] = 0.0f;
} else {
float x31727 = x31568[x31722];
x31720[x31722] = x31727;
}

}
if (x31734) {
} else {
assert(false && "Image too small for averagePool_batch:  x Const(64) x Sym(31553) x Sym(31555) x Sym(31555)|(2,2)");
}
int32_t x31745 = 64 * x31553;
int32_t x31746 = x31745 * x31741;
int32_t x31747 = x31746 * x31741;
float* x31748 = (float*)myMalloc(x31747 * sizeof(float));;
int32_t x31743 = x31553 * x31742;
for(int x31749=0; x31749 < 64; x31749++) {
int32_t x31750 = x31749 * x31557;
float* x31751 = x31720+x31750;
int32_t x31752 = x31749 * x31743;
float* x31753 = x31748+x31752;
for(int x31754=0; x31754 < x31553; x31754++) {
int32_t x31762 = x31754 * x31556;
int32_t x31758 = x31754 * x31742;
for(int x31756=0; x31756 < x31741; x31756++) {
int32_t x31763 = x31756 * x31555;
int32_t x31764 = x31762 + x31763;
int32_t x31759 = x31756 * x31741;
int32_t x31760 = x31758 + x31759;
for(int x31757=0; x31757 < x31741; x31757++) {
float x31766 = 0.0f;
int32_t x31765 = x31764 + x31757;
float x31767 = x31751[x31765];
x31766 += x31767;
int32_t x31769 = x31765 + 1;
float x31770 = x31751[x31769];
x31766 += x31770;
int32_t x31772 = x31765 + x31555;
float x31773 = x31751[x31772];
x31766 += x31773;
int32_t x31775 = x31772 + 1;
float x31776 = x31751[x31775];
x31766 += x31776;
float x31778 = x31766;
int32_t x31761 = x31760 + x31757;
float x31779 = x31778 / 4.0f;
x31753[x31761] = x31779;

}

}

}

}
int32_t x31789 = 0;
int32_t x31790 = 1;
x31790 *= 64;
x31789 += 1;
int32_t x31793 = x31789;
bool x31794 = x31793 >= 2;
if (x31794) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x31799 = x31793 == 0;
int32_t x31744 = 64 * x31743;
if (x31799) {
int32_t x31800 = x31790;
bool x31801 = x31800 == x31744;
if (x31801) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x31808 = x31790;
// gemm: List(Const(64), Sym(31809)), Vector(Const(10), Const(2048))
assert(false && "ERROR not specified");
float* x31813 = (float*)myMalloc(640 * sizeof(float));;
int32_t x31809 = x31744 / x31808;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 64,10,x31809,1.0,x31748,x31809,x227,x31809,0,x31813,10);
int32_t x31815 = 0;
int32_t x31816 = 0;
int32_t x31817 = 0;
for(int x31818=0; x31818 < 64; x31818++) {
int32_t x31819 = x31816;
int32_t x31820 = x31817;
int32_t x31821 = x31815;
int32_t x31822 = x31821;
int32_t x31823 = x31819;
int32_t x31824 = x31820;
for(int x31825=0; x31825 < 10; x31825++) {
int32_t x31826 = x31823;
float x31827 = x31813[x31826];
int32_t x31828 = x31824;
float x31829 = x48[x31828];
float x31830 = x31827 + x31829;
x31813[x31826] = x31830;
x31822 += 1;
x31823 += 1;
x31824 += 1;

}
x31815 += 10;
x31816 += 10;

}
printf("output (size Const(64) x Const(10))\n");
float x31842 = 0.0f;
for(int x31844=0; x31844 < 640; x31844++) {
float x31845 = x31842;
float x31846 = x31813[x31844];
float x31847 = fabs(x31846);
float x31848 = fabs(x31845);
bool x31849 = x31847 > x31848;
float x31850;
if (x31849) {
x31850 = x31846;
} else {
x31850 = x31845;
}
x31842 = x31850;

}
float x31854 = x31842;
printf("Max Abs: %.5f || ",x31854);
for(int x31856=0; x31856 < 10; x31856++) {
float x31857 = x31813[x31856];
printf("%.5f ",x31857);

}
printf("\n");
assert(false && "stop");

}
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

