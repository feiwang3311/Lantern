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

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

int fsize(int fd) {
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

int HEAP_SIZE = 1073741826; // 1048576; // 2147483652; // 536870912; // 268435456; // 2097152;
void *mallocBase = calloc(HEAP_SIZE, 1);
void *mallocAddr = mallocBase;
void *waterMark = mallocBase;
void *myMalloc(size_t bytes) {
  void *res = mallocAddr;
  mallocAddr = (void *)((char *)mallocAddr + bytes);
  return res;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
  long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
  result->tv_sec = diff / 1000000;
  result->tv_usec = diff % 1000000;
  return (diff < 0);
}



void Snippet(char *);

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<> d{0, 1};

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
srand(42);
struct timeval begin_0, end_0, diff_0;
gettimeofday(&begin_0, NULL);
float* x4 = (float*)myMalloc(250 * sizeof(float));;
for(int x6=0; x6 < 250; x6++) {
float x7 = (float)rand()/RAND_MAX;
float x8 = x7 - 0.5f;
float x9 = x8 * 0.2f;
x4[x6] = x9;

}
float* x13 = (float*)myMalloc(250 * sizeof(float));;
float* x14 = (float*)myMalloc(10 * sizeof(float));;
float* x15 = (float*)myMalloc(10 * sizeof(float));;
float* x16 = (float*)myMalloc(5000 * sizeof(float));;
for(int x18=0; x18 < 5000; x18++) {
float x19 = (float)rand()/RAND_MAX;
float x20 = x19 - 0.5f;
float x21 = x20 * 0.06324556f;
x16[x18] = x21;

}
float* x25 = (float*)myMalloc(5000 * sizeof(float));;
float* x26 = (float*)myMalloc(20 * sizeof(float));;
float* x27 = (float*)myMalloc(20 * sizeof(float));;
float* x28 = (float*)myMalloc(16000 * sizeof(float));;
for(int x30=0; x30 < 16000; x30++) {
float x31 = (float)rand()/RAND_MAX;
float x32 = x31 - 0.5f;
float x33 = x32 * 0.0559017f;
x28[x30] = x33;

}
float* x37 = (float*)myMalloc(16000 * sizeof(float));;
float* x38 = (float*)myMalloc(50 * sizeof(float));;
float* x39 = (float*)myMalloc(50 * sizeof(float));;
float* x40 = (float*)myMalloc(500 * sizeof(float));;
for(int x42=0; x42 < 500; x42++) {
float x43 = (float)rand()/RAND_MAX;
float x44 = x43 - 0.5f;
float x45 = x44 * 0.14142136f;
x40[x42] = x45;

}
float* x49 = (float*)myMalloc(500 * sizeof(float));;
float* x50 = (float*)myMalloc(10 * sizeof(float));;
float* x51 = (float*)myMalloc(10 * sizeof(float));;
int64_t* x52 = (int64_t*)myMalloc(2 * sizeof(int64_t));;
int64_t* x53 = (int64_t*)myMalloc(2 * sizeof(int64_t));;
int32_t x64 = 0;
int32_t x65 = x64;
int32_t x66 = x65;
int32_t x60 = open("../data/bin/mnist_train_target.bin",0);
int32_t x61 = fsize(x60);
int32_t x63 = x61 / 4;
int* x62 = (int*)mmap(0, x61, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x60, 0);
int32_t x54 = open("../data/bin/mnist_train.bin",0);
int32_t x55 = fsize(x54);
float* x56 = (float*)mmap(0, x55, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x54, 0);
for(int x68=0; x68 < x63; x68++) {
int32_t x69 = x66;
int32_t x71 = x62[x68];
float* x70 = x56+x69;
for(int x73=0; x73 < 784; x73++) {
float x74 = x70[x73];
float x75 = x74 - 0.1307f;
float x76 = x75 / 0.3081f;
x70[x73] = x76;

}
x66 += 784;

}
int32_t x83 = x66;
int64_t x57 = (int64_t)x55;
int64_t x58 = x57 / 4LL;
int32_t x59 = (int32_t)x58;
bool x84 = x83 == x59;
if (x84) {
} else {
printf("Data length doesn't match\n");
exit(0);
}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x92 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x93 = (float)x92;
float x94 = x93 / 1000000.0f;
printf("Data normalized (all prepare time) in %lf sec\n",x94);
double* x96 = (double*)myMalloc(4 * sizeof(double));;
int64_t x97 = (long)mallocAddr;
int32_t x113 = x63 / 100;
int32_t x1142 = x63 / 10;
double x1147 = (double)x63;
int64_t x1167 = (int64_t)x63;
float x1171 = (float)x63;
for(int x99=0; x99 < 4; x99++) {
struct timeval begin_1, end_1, diff_1;
int32_t x101 = 0;
int32_t x102 = x101;
int32_t x103 = x102;
float x104 = 0.0f;
float x105 = x104;
float x106 = x105;
int32_t x107 = x99 + 1;
printf("Start training epoch %d\n",x107);
gettimeofday(&begin_1, NULL);
int32_t x110 = 0;
int32_t x111 = x110;
int32_t x112 = x111;
for(int x115=0; x115 < x113; x115++) {
int32_t x116 = x112;
x103 += 100;
float* x121 = (float*)myMalloc(1 * sizeof(float));;
x121[0] = 0.0f;
float* x123 = (float*)myMalloc(1 * sizeof(float));;
x123[0] = 0.0f;
float* x125 = (float*)myMalloc(1 * sizeof(float));;
float* x126 = (float*)myMalloc(1 * sizeof(float));;
float* x127 = (float*)myMalloc(576000 * sizeof(float));;
int32_t x128 = 0;
for(int x130=0; x130 < 100; x130++) {
for(int x132=0; x132 < 10; x132++) {
for(int x134=0; x134 < 576; x134++) {
int32_t x135 = x128;
float x136 = x14[x132];
x127[x135] = x136;
x128 += 1;

}

}

}
float* x117 = x56+x116;
for(int x145=0; x145 < 100; x145++) {
int32_t x148 = x145 * 5760;
float* x149 = x127+x148;
int32_t x150 = 0;
int32_t x151 = 0;
int32_t x146 = x145 * 784;
float* x147 = x117+x146;
for(int x152=0; x152 < 10; x152++) {
int32_t x153 = x151;
int32_t x154 = x153;
int32_t x155 = 0;
int32_t x156 = x150;
float* x157 = x149+x156;
int32_t x158 = x155;
int32_t x160 = x154;
float* x161 = x4+x160;
int32_t x162 = 0;
int32_t x163 = 0;
float* x159 = x147+x158;
for(int x165=0; x165 < 24; x165++) {
int32_t x166 = x163;
int32_t x167 = x166;
for(int x168=0; x168 < 24; x168++) {
float x169 = 0.0f;
int32_t x170 = 0;
int32_t x171 = x167;
int32_t x172 = x171;
for(int x174=0; x174 < 5; x174++) {
int32_t x175 = x172;
int32_t x177 = x170;
float* x178 = x161+x177;
float* x176 = x159+x175;
for(int x179=0; x179 < 5; x179++) {
float x180 = x176[x179];
float x181 = x178[x179];
float x182 = x180 * x181;
x169 += x182;

}
x170 += 5;
x172 += 28;

}
int32_t x190 = x162;
float x191 = x157[x190];
float x192 = x169;
float x193 = x191 + x192;
x157[x190] = x193;
x162 += 1;
x167 += 1;

}
x163 += 28;

}
x154 += 25;
x155 += 784;
x151 += 25;
x150 += 576;

}

}
float* x210 = (float*)myMalloc(576000 * sizeof(float));;
float* x211 = (float*)myMalloc(576000 * sizeof(float));;
for(int x213=0; x213 < 576000; x213++) {
float x214 = x127[x213];
bool x215 = x214 < 0.0f;
if (x215) {
x211[x213] = 0.0f;
} else {
float x218 = x127[x213];
x211[x213] = x218;
}

}
float* x224 = (float*)myMalloc(576000 * sizeof(float));;
float* x225 = (float*)myMalloc(144000 * sizeof(float));;
for(int x227=0; x227 < 144000; x227++) {
x225[x227] = -3.4028235E38f;

}
int* x231 = (int32_t*)myMalloc(144000 * sizeof(int32_t));;
for(int x232=0; x232 < 100; x232++) {
int32_t x233 = x232 * 5760;
float* x234 = x211+x233;
int32_t x235 = x232 * 1440;
float* x236 = x225+x235;
int* x237 = x231+x235;
int32_t x238 = 0;
int32_t x239 = 0;
for(int x240=0; x240 < 10; x240++) {
int32_t x241 = x238;
int32_t x242 = x241;
int32_t x243 = x239;
int32_t x244 = x243;
for(int x246=0; x246 < 12; x246++) {
int32_t x247 = x242;
int32_t x248 = x247;
int32_t x249 = x244;
int32_t x250 = x249;
for(int x251=0; x251 < 12; x251++) {
int32_t x252 = x250;
int32_t x253 = x252;
int32_t x254 = x253;
int32_t x255 = x254;
int32_t x256 = x255;
float x257 = x234[x256];
int32_t x258 = x248;
float x259 = x236[x258];
bool x260 = x257 > x259;
if (x260) {
float x261 = x234[x256];
x236[x258] = x261;
int32_t x263 = x256 + x233;
x237[x258] = x263;
} else {
}
x255 += 1;
int32_t x268 = x255;
float x269 = x234[x268];
float x270 = x236[x258];
bool x271 = x269 > x270;
if (x271) {
float x272 = x234[x268];
x236[x258] = x272;
int32_t x274 = x268 + x233;
x237[x258] = x274;
} else {
}
x255 += 1;
x253 += 24;
int32_t x280 = x253;
int32_t x281 = x280;
int32_t x282 = x281;
float x283 = x234[x282];
float x284 = x236[x258];
bool x285 = x283 > x284;
if (x285) {
float x286 = x234[x282];
x236[x258] = x286;
int32_t x288 = x282 + x233;
x237[x258] = x288;
} else {
}
x281 += 1;
int32_t x293 = x281;
float x294 = x234[x293];
float x295 = x236[x258];
bool x296 = x294 > x295;
if (x296) {
float x297 = x234[x293];
x236[x258] = x297;
int32_t x299 = x293 + x233;
x237[x258] = x299;
} else {
}
x281 += 1;
x253 += 24;
x248 += 1;
x250 += 2;

}
x242 += 12;
x244 += 48;

}
x238 += 144;
x239 += 576;

}

}
float* x319 = (float*)myMalloc(144000 * sizeof(float));;
float* x320 = (float*)myMalloc(128000 * sizeof(float));;
int32_t x321 = 0;
for(int x322=0; x322 < 100; x322++) {
for(int x324=0; x324 < 20; x324++) {
for(int x326=0; x326 < 64; x326++) {
int32_t x327 = x321;
float x328 = x26[x324];
x320[x327] = x328;
x321 += 1;

}

}

}
for(int x337=0; x337 < 100; x337++) {
int32_t x338 = x337 * 1440;
float* x339 = x225+x338;
int32_t x340 = x337 * 1280;
float* x341 = x320+x340;
int32_t x342 = 0;
int32_t x343 = 0;
for(int x344=0; x344 < 20; x344++) {
int32_t x345 = x343;
int32_t x346 = x345;
int32_t x347 = 0;
int32_t x348 = x342;
float* x349 = x341+x348;
for(int x350=0; x350 < 10; x350++) {
int32_t x351 = x347;
float* x352 = x339+x351;
int32_t x353 = x346;
float* x354 = x16+x353;
int32_t x355 = 0;
int32_t x356 = 0;
for(int x358=0; x358 < 8; x358++) {
int32_t x359 = x356;
int32_t x360 = x359;
for(int x361=0; x361 < 8; x361++) {
float x362 = 0.0f;
int32_t x363 = 0;
int32_t x364 = x360;
int32_t x365 = x364;
for(int x366=0; x366 < 5; x366++) {
int32_t x367 = x365;
float* x368 = x352+x367;
int32_t x369 = x363;
float* x370 = x354+x369;
for(int x371=0; x371 < 5; x371++) {
float x372 = x368[x371];
float x373 = x370[x371];
float x374 = x372 * x373;
x362 += x374;

}
x363 += 5;
x365 += 12;

}
int32_t x382 = x355;
float x383 = x349[x382];
float x384 = x362;
float x385 = x383 + x384;
x349[x382] = x385;
x355 += 1;
x360 += 1;

}
x356 += 12;

}
x346 += 25;
x347 += 144;

}
x343 += 250;
x342 += 64;

}

}
float* x404 = (float*)myMalloc(128000 * sizeof(float));;
float* x405 = (float*)myMalloc(128000 * sizeof(float));;
for(int x407=0; x407 < 128000; x407++) {
float x408 = x320[x407];
bool x409 = x408 < 0.0f;
if (x409) {
x405[x407] = 0.0f;
} else {
float x412 = x320[x407];
x405[x407] = x412;
}

}
float* x418 = (float*)myMalloc(128000 * sizeof(float));;
float* x419 = (float*)myMalloc(32000 * sizeof(float));;
for(int x421=0; x421 < 32000; x421++) {
x419[x421] = -3.4028235E38f;

}
int* x425 = (int32_t*)myMalloc(32000 * sizeof(int32_t));;
for(int x426=0; x426 < 100; x426++) {
int32_t x427 = x426 * 1280;
float* x428 = x405+x427;
int32_t x429 = x426 * 320;
float* x430 = x419+x429;
int* x431 = x425+x429;
int32_t x432 = 0;
int32_t x433 = 0;
for(int x434=0; x434 < 20; x434++) {
int32_t x435 = x432;
int32_t x436 = x435;
int32_t x437 = x433;
int32_t x438 = x437;
for(int x439=0; x439 < 4; x439++) {
int32_t x440 = x436;
int32_t x441 = x440;
int32_t x442 = x438;
int32_t x443 = x442;
for(int x444=0; x444 < 4; x444++) {
int32_t x445 = x443;
int32_t x446 = x445;
int32_t x447 = x446;
int32_t x448 = x447;
int32_t x449 = x448;
float x450 = x428[x449];
int32_t x451 = x441;
float x452 = x430[x451];
bool x453 = x450 > x452;
if (x453) {
float x454 = x428[x449];
x430[x451] = x454;
int32_t x456 = x449 + x427;
x431[x451] = x456;
} else {
}
x448 += 1;
int32_t x461 = x448;
float x462 = x428[x461];
float x463 = x430[x451];
bool x464 = x462 > x463;
if (x464) {
float x465 = x428[x461];
x430[x451] = x465;
int32_t x467 = x461 + x427;
x431[x451] = x467;
} else {
}
x448 += 1;
x446 += 8;
int32_t x473 = x446;
int32_t x474 = x473;
int32_t x475 = x474;
float x476 = x428[x475];
float x477 = x430[x451];
bool x478 = x476 > x477;
if (x478) {
float x479 = x428[x475];
x430[x451] = x479;
int32_t x481 = x475 + x427;
x431[x451] = x481;
} else {
}
x474 += 1;
int32_t x486 = x474;
float x487 = x428[x486];
float x488 = x430[x451];
bool x489 = x487 > x488;
if (x489) {
float x490 = x428[x486];
x430[x451] = x490;
int32_t x492 = x486 + x427;
x431[x451] = x492;
} else {
}
x474 += 1;
x446 += 8;
x441 += 1;
x443 += 2;

}
x436 += 4;
x438 += 16;

}
x432 += 16;
x433 += 64;

}

}
float* x512 = (float*)myMalloc(32000 * sizeof(float));;
// dot: ArrayBuffer(100, 320), List(320, 50)
float* x514 = (float*)myMalloc(5000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 100,50,320,1,x419,320,x28,50,0,x514,50);
float* x516 = (float*)myMalloc(5000 * sizeof(float));;
float* x517 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x518 = 0;
int32_t x519 = 0;
int32_t x520 = 0;
for(int x521=0; x521 < 100; x521++) {
int32_t x522 = x519;
int32_t x523 = x520;
int32_t x524 = x518;
int32_t x525 = x524;
int32_t x526 = x522;
int32_t x527 = x523;
for(int x529=0; x529 < 50; x529++) {
int32_t x530 = x525;
int32_t x531 = x526;
float x532 = x514[x531];
int32_t x533 = x527;
float x534 = x38[x533];
float x535 = x532 + x534;
x517[x530] = x535;
x525 += 1;
x526 += 1;
x527 += 1;

}
x518 += 50;
x519 += 50;

}
float* x546 = (float*)myMalloc(5000 * sizeof(float));;
float* x547 = (float*)myMalloc(5000 * sizeof(float));;
float* x548 = (float*)myMalloc(5000 * sizeof(float));;
for(int x549=0; x549 < 5000; x549++) {
float x550 = (float)rand()/RAND_MAX;
bool x551 = x550 > 0.5f;
if (x551) {
float x552 = x517[x549];
float x553 = x552 * 2.0f;
x547[x549] = x553;
x548[x549] = 2.0f;
} else {
x547[x549] = 0.0f;
x548[x549] = 0.0f;
}

}
float* x563 = (float*)myMalloc(5000 * sizeof(float));;
// dot: List(100, 50), List(50, 10)
float* x565 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 100,10,50,1,x547,50,x40,10,0,x565,10);
float* x567 = (float*)myMalloc(1000 * sizeof(float));;
float* x568 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x569 = 0;
int32_t x570 = 0;
int32_t x571 = 0;
for(int x572=0; x572 < 100; x572++) {
int32_t x573 = x570;
int32_t x574 = x571;
int32_t x575 = x569;
int32_t x576 = x575;
int32_t x577 = x573;
int32_t x578 = x574;
for(int x579=0; x579 < 10; x579++) {
int32_t x580 = x576;
int32_t x581 = x577;
float x582 = x565[x581];
int32_t x583 = x578;
float x584 = x50[x583];
float x585 = x582 + x584;
x568[x580] = x585;
x576 += 1;
x577 += 1;
x578 += 1;

}
x569 += 10;
x570 += 10;

}
float* x596 = (float*)myMalloc(1000 * sizeof(float));;
float* x597 = (float*)myMalloc(100 * sizeof(float));;
int32_t x598 = 0;
for(int x599=0; x599 < 100; x599++) {
float x600 = -3.4028235E38f;
for(int x601=0; x601 < 10; x601++) {
int32_t x602 = x598;
float x603 = x568[x602];
float x604 = x600;
bool x605 = x603 > x604;
if (x605) {
float x606 = x568[x602];
x600 = x606;
} else {
}
x598 += 1;

}
float x613 = x600;
x597[x599] = x613;

}
float* x617 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x618 = 0;
for(int x619=0; x619 < 100; x619++) {
for(int x620=0; x620 < 10; x620++) {
int32_t x621 = x618;
float x622 = x568[x621];
float x623 = x597[x619];
float x624 = x622 - x623;
double x625 = (double)x624;
double x626 = exp(x625);
float x627 = (float)x626;
x617[x621] = x627;
x618 += 1;

}

}
float* x634 = (float*)myMalloc(100 * sizeof(float));;
for(int x635=0; x635 < 100; x635++) {
int32_t x636 = x635;
int32_t x637 = x635 * 10;
int32_t x638 = x637;
for(int x639=0; x639 < 10; x639++) {
int32_t x640 = x636;
float x641 = x634[x640];
int32_t x642 = x638;
float x643 = x617[x642];
float x644 = x641 + x643;
x634[x640] = x644;
x638 += 1;

}

}
x618 = 0;
for(int x652=0; x652 < 100; x652++) {
float x653 = x597[x652];
float x654 = x634[x652];
double x655 = (double)x654;
double x656 = log(x655);
float x657 = (float)x656;
float x658 = x653 + x657;
for(int x659=0; x659 < 10; x659++) {
int32_t x660 = x618;
float x661 = x568[x660];
float x662 = x661 - x658;
x617[x660] = x662;
x618 += 1;

}

}
float* x669 = (float*)myMalloc(1000 * sizeof(float));;
float* x670 = (float*)myMalloc(100 * sizeof(float));;
int32_t x671 = 0;
int32_t x118 = x115 * 100;
int* x119 = x62+x118;
for(int x672=0; x672 < 100; x672++) {
int32_t x673 = x671;
int32_t x674 = x119[x672];
int32_t x675 = x673 + x674;
float x676 = x617[x675];
float x677 = -1.0f * x676;
x670[x672] = x677;
x671 += 10;

}
float* x682 = (float*)myMalloc(100 * sizeof(float));;
float x683 = 0.0f;
for(int x684=0; x684 < 100; x684++) {
float x685 = x683;
float x686 = x670[x684];
float x687 = x685 + x686;
x683 = x687;

}
float x691 = x683;
float* x692 = (float*)myMalloc(1 * sizeof(float));;
x692[0] = x691;
float* x694 = (float*)myMalloc(1 * sizeof(float));;
float x695 = x694[0];
x694[0] = 1.0f;
float x697 = x692[0];
x126[0] = x697;
// += tensor of dim 0
float x700 = x694[0];
for(int x701=0; x701 < 100; x701++) {
float x702 = x682[x701];
float x703 = x702 + x700;
x682[x701] = x703;

}
int32_t x707 = 0;
for(int x708=0; x708 < 100; x708++) {
int32_t x709 = x707;
int32_t x710 = x119[x708];
int32_t x711 = x709 + x710;
float x712 = x669[x711];
float x713 = x682[x708];
float x714 = -1.0f * x713;
float x715 = x712 + x714;
x669[x711] = x715;
x707 += 10;

}
float* x720 = (float*)myMalloc(100 * sizeof(float));;
for(int x721=0; x721 < 100; x721++) {
int32_t x722 = x721;
int32_t x723 = x721 * 10;
int32_t x724 = x723;
for(int x725=0; x725 < 10; x725++) {
int32_t x726 = x722;
float x727 = x720[x726];
int32_t x728 = x724;
float x729 = x669[x728];
float x730 = x727 + x729;
x720[x726] = x730;
x724 += 1;

}

}
int32_t x737 = 0;
for(int x738=0; x738 < 100; x738++) {
for(int x739=0; x739 < 10; x739++) {
int32_t x740 = x737;
float x741 = x596[x740];
float x742 = x669[x740];
float x743 = x617[x740];
float x747 = x720[x738];
double x744 = (double)x743;
double x745 = exp(x744);
float x746 = (float)x745;
float x748 = x746 * x747;
float x749 = x742 - x748;
float x750 = x741 + x749;
x596[x740] = x750;
x737 += 1;

}

}
int32_t x757 = 0;
int32_t x758 = 0;
int32_t x759 = 0;
for(int x760=0; x760 < 100; x760++) {
int32_t x761 = x757;
int32_t x762 = x758;
int32_t x763 = x759;
int32_t x764 = x761;
int32_t x765 = x762;
int32_t x766 = x763;
for(int x767=0; x767 < 10; x767++) {
int32_t x768 = x764;
float x769 = x567[x768];
float x770 = x565[x768];
int32_t x771 = x765;
float x772 = x50[x771];
int32_t x773 = x766;
float x774 = x596[x773];
float x775 = x769 + x774;
x567[x768] = x775;
float x777 = x51[x771];
float x778 = x565[x768];
float x779 = x50[x771];
float x780 = x596[x773];
float x781 = x777 + x780;
x51[x771] = x781;
x766 += 1;
x764 += 1;
x765 += 1;

}
x759 += 10;
x757 += 10;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,50,10,1,x567,10,x40,10,1,x563,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,100,1,x547,50,x567,10,1,x49,10);
float* x794 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x795 = 0;
int32_t x796 = 0;
int32_t x797 = 0;
for(int x798=0; x798 < 100; x798++) {
int32_t x799 = x796;
int32_t x800 = x797;
int32_t x801 = x795;
int32_t x802 = x801;
int32_t x803 = x799;
int32_t x804 = x800;
for(int x805=0; x805 < 50; x805++) {
int32_t x806 = x802;
int32_t x807 = x803;
float x808 = x548[x807];
int32_t x809 = x804;
float x810 = x563[x809];
float x811 = x808 * x810;
x794[x806] = x811;
x802 += 1;
x803 += 1;
x804 += 1;

}
x795 += 50;
x796 += 50;
x797 += 50;

}
for(int x823=0; x823 < 5000; x823++) {
float x824 = x546[x823];
float x825 = x794[x823];
float x826 = x824 + x825;
x546[x823] = x826;

}
int32_t x830 = 0;
int32_t x831 = 0;
int32_t x832 = 0;
for(int x833=0; x833 < 100; x833++) {
int32_t x834 = x830;
int32_t x835 = x831;
int32_t x836 = x832;
int32_t x837 = x834;
int32_t x838 = x835;
int32_t x839 = x836;
for(int x840=0; x840 < 50; x840++) {
int32_t x841 = x837;
float x842 = x516[x841];
float x843 = x514[x841];
int32_t x844 = x838;
float x845 = x38[x844];
int32_t x846 = x839;
float x847 = x546[x846];
float x848 = x842 + x847;
x516[x841] = x848;
float x850 = x39[x844];
float x851 = x514[x841];
float x852 = x38[x844];
float x853 = x546[x846];
float x854 = x850 + x853;
x39[x844] = x854;
x839 += 1;
x837 += 1;
x838 += 1;

}
x832 += 50;
x830 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,320,50,1,x516,50,x28,50,1,x512,320);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,100,1,x419,320,x516,50,1,x37,50);
for(int x867=0; x867 < 32000; x867++) {
int32_t x868 = x425[x867];
float x869 = x512[x867];
x418[x868] = x869;

}
for(int x873=0; x873 < 128000; x873++) {
float x874 = x320[x873];
bool x875 = x874 < 0.0f;
float x878;
if (x875) {
x878 = 0.0f;
} else {
float x876 = x418[x873];
x878 = x876;
}
x404[x873] = x878;

}
for(int x882=0; x882 < 100; x882++) {
int32_t x883 = x882 * 1280;
int32_t x884 = x883;
int32_t x885 = 0;
int32_t x888 = x882 * 1440;
for(int x886=0; x886 < 20; x886++) {
float x887 = 0.0f;
int32_t x889 = x888;
for(int x890=0; x890 < 8; x890++) {
int32_t x891 = x889;
int32_t x892 = x891;
for(int x893=0; x893 < 8; x893++) {
int32_t x894 = x884;
float x895 = x404[x894];
x887 += x895;
int32_t x897 = x892;
int32_t x898 = x897;
int32_t x899 = x885;
int32_t x900 = x899;
for(int x901=0; x901 < 10; x901++) {
int32_t x902 = x898;
int32_t x903 = x902;
for(int x904=0; x904 < 5; x904++) {
for(int x905=0; x905 < 5; x905++) {
int32_t x906 = x903;
int32_t x907 = x906 + x905;
float x908 = x319[x907];
int32_t x909 = x900;
float x910 = x16[x909];
float x911 = x895 * x910;
float x912 = x908 + x911;
x319[x907] = x912;
float x914 = x25[x909];
float x915 = x225[x907];
float x916 = x895 * x915;
float x917 = x914 + x916;
x25[x909] = x917;
x900 += 1;

}
x903 += 12;

}
x898 += 144;

}
x892 += 1;
x884 += 1;

}
x889 += 12;

}
float x935 = x27[x886];
float x936 = x887;
float x937 = x935 + x936;
x27[x886] = x937;
x885 += 250;

}

}
for(int x944=0; x944 < 144000; x944++) {
int32_t x945 = x231[x944];
float x946 = x319[x944];
x224[x945] = x946;

}
for(int x950=0; x950 < 576000; x950++) {
float x951 = x127[x950];
bool x952 = x951 < 0.0f;
float x955;
if (x952) {
x955 = 0.0f;
} else {
float x953 = x224[x950];
x955 = x953;
}
x210[x950] = x955;

}
for(int x959=0; x959 < 100; x959++) {
int32_t x960 = x959 * 5760;
int32_t x961 = x960;
int32_t x962 = 0;
int32_t x965 = x959 * 784;
for(int x963=0; x963 < 10; x963++) {
float x964 = 0.0f;
int32_t x966 = x965;
for(int x967=0; x967 < 24; x967++) {
int32_t x968 = x966;
int32_t x969 = x968;
for(int x970=0; x970 < 24; x970++) {
int32_t x971 = x961;
float x972 = x210[x971];
x964 += x972;
int32_t x974 = x969;
int32_t x975 = x974;
int32_t x976 = x962;
int32_t x977 = x976;
int32_t x978 = x975;
int32_t x979 = x978;
for(int x980=0; x980 < 5; x980++) {
for(int x981=0; x981 < 5; x981++) {
int32_t x982 = x977;
float x983 = x13[x982];
int32_t x984 = x979;
int32_t x985 = x984 + x981;
float x986 = x117[x985];
float x987 = x972 * x986;
float x988 = x983 + x987;
x13[x982] = x988;
x977 += 1;

}
x979 += 28;

}
x975 += 784;
x969 += 1;
x961 += 1;

}
x966 += 28;

}
float x1004 = x15[x963];
float x1005 = x964;
float x1006 = x1004 + x1005;
x15[x963] = x1006;
x962 += 25;

}

}
float x1013 = x126[0];
x106 += x1013;
for(int x1015=0; x1015 < 5000; x1015++) {
float x1016 = x25[x1015];
float x1017 = x1016;
float x1018 = x1017;
bool x1019 = x1018 > 1000.0f;
if (x1019) {
x1017 = 1000.0f;
} else {
}
float x1023 = x1017;
bool x1024 = x1023 < -1000.0f;
if (x1024) {
x1017 = -1000.0f;
} else {
}
float x1028 = x16[x1015];
float x1029 = x1017;
float x1030 = 5.0E-4f * x1029;
float x1031 = x1028 - x1030;
x16[x1015] = x1031;
x25[x1015] = 0.0f;

}
for(int x1036=0; x1036 < 16000; x1036++) {
float x1037 = x37[x1036];
float x1038 = x1037;
float x1039 = x1038;
bool x1040 = x1039 > 1000.0f;
if (x1040) {
x1038 = 1000.0f;
} else {
}
float x1044 = x1038;
bool x1045 = x1044 < -1000.0f;
if (x1045) {
x1038 = -1000.0f;
} else {
}
float x1049 = x28[x1036];
float x1050 = x1038;
float x1051 = 5.0E-4f * x1050;
float x1052 = x1049 - x1051;
x28[x1036] = x1052;
x37[x1036] = 0.0f;

}
for(int x1057=0; x1057 < 50; x1057++) {
float x1058 = x39[x1057];
float x1059 = x1058;
float x1060 = x1059;
bool x1061 = x1060 > 1000.0f;
if (x1061) {
x1059 = 1000.0f;
} else {
}
float x1065 = x1059;
bool x1066 = x1065 < -1000.0f;
if (x1066) {
x1059 = -1000.0f;
} else {
}
float x1070 = x38[x1057];
float x1071 = x1059;
float x1072 = 5.0E-4f * x1071;
float x1073 = x1070 - x1072;
x38[x1057] = x1073;
x39[x1057] = 0.0f;

}
for(int x1078=0; x1078 < 250; x1078++) {
float x1079 = x13[x1078];
float x1080 = x1079;
float x1081 = x1080;
bool x1082 = x1081 > 1000.0f;
if (x1082) {
x1080 = 1000.0f;
} else {
}
float x1086 = x1080;
bool x1087 = x1086 < -1000.0f;
if (x1087) {
x1080 = -1000.0f;
} else {
}
float x1091 = x4[x1078];
float x1092 = x1080;
float x1093 = 5.0E-4f * x1092;
float x1094 = x1091 - x1093;
x4[x1078] = x1094;
x13[x1078] = 0.0f;

}
for(int x1099=0; x1099 < 10; x1099++) {
float x1100 = x51[x1099];
float x1101 = x1100;
float x1102 = x1101;
bool x1103 = x1102 > 1000.0f;
if (x1103) {
x1101 = 1000.0f;
} else {
}
float x1107 = x1101;
bool x1108 = x1107 < -1000.0f;
if (x1108) {
x1101 = -1000.0f;
} else {
}
float x1112 = x50[x1099];
float x1113 = x1101;
float x1114 = 5.0E-4f * x1113;
float x1115 = x1112 - x1114;
x50[x1099] = x1115;
x51[x1099] = 0.0f;

}
for(int x1120=0; x1120 < 500; x1120++) {
float x1121 = x49[x1120];
float x1122 = x1121;
float x1123 = x1122;
bool x1124 = x1123 > 1000.0f;
if (x1124) {
x1122 = 1000.0f;
} else {
}
float x1128 = x1122;
bool x1129 = x1128 < -1000.0f;
if (x1129) {
x1122 = -1000.0f;
} else {
}
float x1133 = x40[x1120];
float x1134 = x1122;
float x1135 = 5.0E-4f * x1134;
float x1136 = x1133 - x1135;
x40[x1120] = x1136;
x49[x1120] = 0.0f;

}
int32_t x1141 = x103;
int32_t x1143 = x1141 % x1142;
bool x1144 = x1143 == 0;
if (x1144) {
float x1149 = x106;
double x1145 = (double)x1141;
double x1146 = 100.0 * x1145;
double x1148 = x1146 / x1147;
float x1150 = (float)x1141;
float x1151 = x1149 / x1150;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x99,x1141,x63,x1148,x1151);
fflush(stdout);
} else {
}
int64_t x1156 = (long)mallocAddr;
int64_t x1157 = x1156 - x97;
memset((void*)x97, 0, x1157);
mallocAddr = (void*)x97;
x112 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1165 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1166 = x1165 / 1000LL;
int64_t x1168 = x1165 / x1167;
printf("Training completed in %ldms (%ld us/images)\n",x1166,x1168);
float x1170 = x106;
float x1172 = x1170 / x1171;
double x1173 = (double)x1172;
x96[x99] = x1173;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1179 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1184 = (long)fopen(x0, "w");
fprintf((FILE *)x1184, "unit: %s\n", "1 epoch");
for(int x1186=0; x1186 < 4; x1186++) {
double x1187 = x96[x1186];
fprintf((FILE *)x1184, "%lf\n", x1187);

}
float x1180 = (float)x1179;
float x1181 = x1180 / 1000000.0f;
float x1182 = x1181 - x94;
float x1183 = x1182 / 4.0f;
fprintf((FILE *)x1184, "run time: %lf %lf\n", x94, x1183);
fclose((FILE*)x1184);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

