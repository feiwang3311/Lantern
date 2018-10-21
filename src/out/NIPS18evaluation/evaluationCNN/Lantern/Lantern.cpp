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
int32_t x1138 = x63 / 10;
double x1143 = (double)x63;
int64_t x1163 = (int64_t)x63;
float x1167 = (float)x63;
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
float* x145 = (float*)myMalloc(1440000 * sizeof(float));;
float* x117 = x56+x116;
for(int x146=0; x146 < 100; x146++) {
int32_t x149 = x146 * 5760;
float* x150 = x127+x149;
int32_t x151 = x146 * 14400;
float* x152 = x145+x151;
int32_t x147 = x146 * 784;
float* x148 = x117+x147;
for(int x154=0; x154 < 25; x154++) {
int32_t x155 = x154 / 25;
int32_t x159 = x155 * 5;
int32_t x160 = x159 * 5;
int32_t x161 = x160 * 24;
int32_t x162 = x161 * 24;
int32_t x156 = x154 % 25;
int32_t x157 = x156 / 5;
int32_t x163 = x157 * 5;
int32_t x164 = x163 * 24;
int32_t x165 = x164 * 24;
int32_t x166 = x162 + x165;
int32_t x158 = x156 % 5;
int32_t x167 = x158 * 24;
int32_t x168 = x167 * 24;
int32_t x169 = x166 + x168;
float* x170 = x152+x169;
int32_t x171 = x155 * 28;
int32_t x172 = x171 * 28;
float* x173 = x148+x172;
for(int x175=0; x175 < 24; x175++) {
int32_t x177 = x175 * 24;
float* x178 = x170+x177;
int32_t x176 = x175 + x157;
int32_t x179 = x176 * 28;
int32_t x180 = x179 + x158;
float* x181 = x173+x180;
memcpy(x178, x181, 4 * 24);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 10,576,25,1,x4,25,x152,576,1,x150,576);

}
float* x190 = (float*)myMalloc(576000 * sizeof(float));;
float* x191 = (float*)myMalloc(576000 * sizeof(float));;
for(int x193=0; x193 < 576000; x193++) {
float x194 = x127[x193];
bool x195 = x194 < 0.0f;
if (x195) {
x191[x193] = 0.0f;
} else {
float x198 = x127[x193];
x191[x193] = x198;
}

}
float* x204 = (float*)myMalloc(576000 * sizeof(float));;
float* x205 = (float*)myMalloc(144000 * sizeof(float));;
for(int x207=0; x207 < 144000; x207++) {
x205[x207] = -3.4028235E38f;

}
int* x211 = (int32_t*)myMalloc(144000 * sizeof(int32_t));;
for(int x212=0; x212 < 100; x212++) {
int32_t x213 = x212 * 5760;
float* x214 = x191+x213;
int32_t x215 = x212 * 1440;
float* x216 = x205+x215;
int* x217 = x211+x215;
int32_t x218 = 0;
int32_t x219 = 0;
for(int x220=0; x220 < 10; x220++) {
int32_t x221 = x218;
int32_t x222 = x221;
int32_t x223 = x219;
int32_t x224 = x223;
for(int x226=0; x226 < 12; x226++) {
int32_t x227 = x222;
int32_t x228 = x227;
int32_t x229 = x224;
int32_t x230 = x229;
for(int x231=0; x231 < 12; x231++) {
int32_t x232 = x230;
int32_t x233 = x232;
int32_t x234 = x233;
int32_t x235 = x234;
int32_t x236 = x235;
float x237 = x214[x236];
int32_t x238 = x228;
float x239 = x216[x238];
bool x240 = x237 > x239;
if (x240) {
float x241 = x214[x236];
x216[x238] = x241;
int32_t x243 = x236 + x213;
x217[x238] = x243;
} else {
}
x235 += 1;
int32_t x248 = x235;
float x249 = x214[x248];
float x250 = x216[x238];
bool x251 = x249 > x250;
if (x251) {
float x252 = x214[x248];
x216[x238] = x252;
int32_t x254 = x248 + x213;
x217[x238] = x254;
} else {
}
x235 += 1;
x233 += 24;
int32_t x260 = x233;
int32_t x261 = x260;
int32_t x262 = x261;
float x263 = x214[x262];
float x264 = x216[x238];
bool x265 = x263 > x264;
if (x265) {
float x266 = x214[x262];
x216[x238] = x266;
int32_t x268 = x262 + x213;
x217[x238] = x268;
} else {
}
x261 += 1;
int32_t x273 = x261;
float x274 = x214[x273];
float x275 = x216[x238];
bool x276 = x274 > x275;
if (x276) {
float x277 = x214[x273];
x216[x238] = x277;
int32_t x279 = x273 + x213;
x217[x238] = x279;
} else {
}
x261 += 1;
x233 += 24;
x228 += 1;
x230 += 2;

}
x222 += 12;
x224 += 48;

}
x218 += 144;
x219 += 576;

}

}
float* x299 = (float*)myMalloc(144000 * sizeof(float));;
float* x300 = (float*)myMalloc(128000 * sizeof(float));;
int32_t x301 = 0;
for(int x302=0; x302 < 100; x302++) {
for(int x304=0; x304 < 20; x304++) {
for(int x306=0; x306 < 64; x306++) {
int32_t x307 = x301;
float x308 = x26[x304];
x300[x307] = x308;
x301 += 1;

}

}

}
float* x317 = (float*)myMalloc(1600000 * sizeof(float));;
for(int x318=0; x318 < 100; x318++) {
int32_t x319 = x318 * 1440;
float* x320 = x205+x319;
int32_t x321 = x318 * 1280;
float* x322 = x300+x321;
int32_t x323 = x318 * 16000;
float* x324 = x317+x323;
for(int x325=0; x325 < 250; x325++) {
int32_t x326 = x325 / 25;
int32_t x330 = x326 * 5;
int32_t x331 = x330 * 5;
int32_t x332 = x331 * 8;
int32_t x333 = x332 * 8;
int32_t x327 = x325 % 25;
int32_t x328 = x327 / 5;
int32_t x334 = x328 * 5;
int32_t x335 = x334 * 8;
int32_t x336 = x335 * 8;
int32_t x337 = x333 + x336;
int32_t x329 = x327 % 5;
int32_t x338 = x329 * 8;
int32_t x339 = x338 * 8;
int32_t x340 = x337 + x339;
float* x341 = x324+x340;
int32_t x342 = x326 * 12;
int32_t x343 = x342 * 12;
float* x344 = x320+x343;
for(int x346=0; x346 < 8; x346++) {
int32_t x348 = x346 * 8;
float* x349 = x341+x348;
int32_t x347 = x346 + x328;
int32_t x350 = x347 * 12;
int32_t x351 = x350 + x329;
float* x352 = x344+x351;
memcpy(x349, x352, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,64,250,1,x16,250,x324,64,1,x322,64);

}
float* x361 = (float*)myMalloc(128000 * sizeof(float));;
float* x362 = (float*)myMalloc(128000 * sizeof(float));;
for(int x364=0; x364 < 128000; x364++) {
float x365 = x300[x364];
bool x366 = x365 < 0.0f;
if (x366) {
x362[x364] = 0.0f;
} else {
float x369 = x300[x364];
x362[x364] = x369;
}

}
float* x375 = (float*)myMalloc(128000 * sizeof(float));;
float* x376 = (float*)myMalloc(32000 * sizeof(float));;
for(int x378=0; x378 < 32000; x378++) {
x376[x378] = -3.4028235E38f;

}
int* x382 = (int32_t*)myMalloc(32000 * sizeof(int32_t));;
for(int x383=0; x383 < 100; x383++) {
int32_t x384 = x383 * 1280;
float* x385 = x362+x384;
int32_t x386 = x383 * 320;
float* x387 = x376+x386;
int* x388 = x382+x386;
int32_t x389 = 0;
int32_t x390 = 0;
for(int x391=0; x391 < 20; x391++) {
int32_t x392 = x389;
int32_t x393 = x392;
int32_t x394 = x390;
int32_t x395 = x394;
for(int x396=0; x396 < 4; x396++) {
int32_t x397 = x393;
int32_t x398 = x397;
int32_t x399 = x395;
int32_t x400 = x399;
for(int x401=0; x401 < 4; x401++) {
int32_t x402 = x400;
int32_t x403 = x402;
int32_t x404 = x403;
int32_t x405 = x404;
int32_t x406 = x405;
float x407 = x385[x406];
int32_t x408 = x398;
float x409 = x387[x408];
bool x410 = x407 > x409;
if (x410) {
float x411 = x385[x406];
x387[x408] = x411;
int32_t x413 = x406 + x384;
x388[x408] = x413;
} else {
}
x405 += 1;
int32_t x418 = x405;
float x419 = x385[x418];
float x420 = x387[x408];
bool x421 = x419 > x420;
if (x421) {
float x422 = x385[x418];
x387[x408] = x422;
int32_t x424 = x418 + x384;
x388[x408] = x424;
} else {
}
x405 += 1;
x403 += 8;
int32_t x430 = x403;
int32_t x431 = x430;
int32_t x432 = x431;
float x433 = x385[x432];
float x434 = x387[x408];
bool x435 = x433 > x434;
if (x435) {
float x436 = x385[x432];
x387[x408] = x436;
int32_t x438 = x432 + x384;
x388[x408] = x438;
} else {
}
x431 += 1;
int32_t x443 = x431;
float x444 = x385[x443];
float x445 = x387[x408];
bool x446 = x444 > x445;
if (x446) {
float x447 = x385[x443];
x387[x408] = x447;
int32_t x449 = x443 + x384;
x388[x408] = x449;
} else {
}
x431 += 1;
x403 += 8;
x398 += 1;
x400 += 2;

}
x393 += 4;
x395 += 16;

}
x389 += 16;
x390 += 64;

}

}
float* x469 = (float*)myMalloc(32000 * sizeof(float));;
// dot: ArrayBuffer(100, 320), List(320, 50)
float* x471 = (float*)myMalloc(5000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 100,50,320,1,x376,320,x28,50,0,x471,50);
float* x473 = (float*)myMalloc(5000 * sizeof(float));;
float* x474 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x475 = 0;
int32_t x476 = 0;
int32_t x477 = 0;
for(int x478=0; x478 < 100; x478++) {
int32_t x479 = x476;
int32_t x480 = x477;
int32_t x481 = x475;
int32_t x482 = x481;
int32_t x483 = x479;
int32_t x484 = x480;
for(int x486=0; x486 < 50; x486++) {
int32_t x487 = x482;
int32_t x488 = x483;
float x489 = x471[x488];
int32_t x490 = x484;
float x491 = x38[x490];
float x492 = x489 + x491;
x474[x487] = x492;
x482 += 1;
x483 += 1;
x484 += 1;

}
x475 += 50;
x476 += 50;

}
float* x503 = (float*)myMalloc(5000 * sizeof(float));;
float* x504 = (float*)myMalloc(5000 * sizeof(float));;
float* x505 = (float*)myMalloc(5000 * sizeof(float));;
for(int x506=0; x506 < 5000; x506++) {
float x507 = (float)rand()/RAND_MAX;
bool x508 = x507 > 0.5f;
if (x508) {
float x509 = x474[x506];
float x510 = x509 * 2.0f;
x504[x506] = x510;
x505[x506] = 2.0f;
} else {
x504[x506] = 0.0f;
x505[x506] = 0.0f;
}

}
float* x520 = (float*)myMalloc(5000 * sizeof(float));;
// dot: List(100, 50), List(50, 10)
float* x522 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 100,10,50,1,x504,50,x40,10,0,x522,10);
float* x524 = (float*)myMalloc(1000 * sizeof(float));;
float* x525 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x526 = 0;
int32_t x527 = 0;
int32_t x528 = 0;
for(int x529=0; x529 < 100; x529++) {
int32_t x530 = x527;
int32_t x531 = x528;
int32_t x532 = x526;
int32_t x533 = x532;
int32_t x534 = x530;
int32_t x535 = x531;
for(int x536=0; x536 < 10; x536++) {
int32_t x537 = x533;
int32_t x538 = x534;
float x539 = x522[x538];
int32_t x540 = x535;
float x541 = x50[x540];
float x542 = x539 + x541;
x525[x537] = x542;
x533 += 1;
x534 += 1;
x535 += 1;

}
x526 += 10;
x527 += 10;

}
float* x553 = (float*)myMalloc(1000 * sizeof(float));;
float* x554 = (float*)myMalloc(100 * sizeof(float));;
int32_t x555 = 0;
for(int x556=0; x556 < 100; x556++) {
float x557 = -3.4028235E38f;
for(int x558=0; x558 < 10; x558++) {
int32_t x559 = x555;
float x560 = x525[x559];
float x561 = x557;
bool x562 = x560 > x561;
if (x562) {
float x563 = x525[x559];
x557 = x563;
} else {
}
x555 += 1;

}
float x570 = x557;
x554[x556] = x570;

}
float* x574 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x575 = 0;
for(int x576=0; x576 < 100; x576++) {
for(int x577=0; x577 < 10; x577++) {
int32_t x578 = x575;
float x579 = x525[x578];
float x580 = x554[x576];
float x581 = x579 - x580;
double x582 = (double)x581;
double x583 = exp(x582);
float x584 = (float)x583;
x574[x578] = x584;
x575 += 1;

}

}
float* x591 = (float*)myMalloc(100 * sizeof(float));;
for(int x592=0; x592 < 100; x592++) {
int32_t x593 = x592;
int32_t x594 = x592 * 10;
int32_t x595 = x594;
for(int x596=0; x596 < 10; x596++) {
int32_t x597 = x593;
float x598 = x591[x597];
int32_t x599 = x595;
float x600 = x574[x599];
float x601 = x598 + x600;
x591[x597] = x601;
x595 += 1;

}

}
x575 = 0;
for(int x609=0; x609 < 100; x609++) {
float x610 = x554[x609];
float x611 = x591[x609];
double x612 = (double)x611;
double x613 = log(x612);
float x614 = (float)x613;
float x615 = x610 + x614;
for(int x616=0; x616 < 10; x616++) {
int32_t x617 = x575;
float x618 = x525[x617];
float x619 = x618 - x615;
x574[x617] = x619;
x575 += 1;

}

}
float* x626 = (float*)myMalloc(1000 * sizeof(float));;
float* x627 = (float*)myMalloc(100 * sizeof(float));;
int32_t x628 = 0;
int32_t x118 = x115 * 100;
int* x119 = x62+x118;
for(int x629=0; x629 < 100; x629++) {
int32_t x630 = x628;
int32_t x631 = x119[x629];
int32_t x632 = x630 + x631;
float x633 = x574[x632];
float x634 = -1.0f * x633;
x627[x629] = x634;
x628 += 10;

}
float* x639 = (float*)myMalloc(100 * sizeof(float));;
float x640 = 0.0f;
for(int x641=0; x641 < 100; x641++) {
float x642 = x640;
float x643 = x627[x641];
float x644 = x642 + x643;
x640 = x644;

}
float x648 = x640;
float* x649 = (float*)myMalloc(1 * sizeof(float));;
x649[0] = x648;
float* x651 = (float*)myMalloc(1 * sizeof(float));;
x651[0] = 1.0f;
float x653 = x649[0];
x126[0] = x653;
// += tensor of dim 0
float x656 = x651[0];
for(int x657=0; x657 < 100; x657++) {
float x658 = x639[x657];
float x659 = x658 + x656;
x639[x657] = x659;

}
int32_t x663 = 0;
for(int x664=0; x664 < 100; x664++) {
int32_t x665 = x663;
int32_t x666 = x119[x664];
int32_t x667 = x665 + x666;
float x668 = x626[x667];
float x669 = x639[x664];
float x670 = -1.0f * x669;
float x671 = x668 + x670;
x626[x667] = x671;
x663 += 10;

}
float* x676 = (float*)myMalloc(100 * sizeof(float));;
for(int x677=0; x677 < 100; x677++) {
int32_t x678 = x677;
int32_t x679 = x677 * 10;
int32_t x680 = x679;
for(int x681=0; x681 < 10; x681++) {
int32_t x682 = x678;
float x683 = x676[x682];
int32_t x684 = x680;
float x685 = x626[x684];
float x686 = x683 + x685;
x676[x682] = x686;
x680 += 1;

}

}
int32_t x693 = 0;
for(int x694=0; x694 < 100; x694++) {
for(int x695=0; x695 < 10; x695++) {
int32_t x696 = x693;
float x697 = x553[x696];
float x698 = x626[x696];
float x699 = x574[x696];
float x703 = x676[x694];
double x700 = (double)x699;
double x701 = exp(x700);
float x702 = (float)x701;
float x704 = x702 * x703;
float x705 = x698 - x704;
float x706 = x697 + x705;
x553[x696] = x706;
x693 += 1;

}

}
int32_t x713 = 0;
int32_t x714 = 0;
int32_t x715 = 0;
for(int x716=0; x716 < 100; x716++) {
int32_t x717 = x713;
int32_t x718 = x714;
int32_t x719 = x715;
int32_t x720 = x717;
int32_t x721 = x718;
int32_t x722 = x719;
for(int x723=0; x723 < 10; x723++) {
int32_t x724 = x720;
float x725 = x524[x724];
float x726 = x522[x724];
int32_t x727 = x721;
float x728 = x50[x727];
int32_t x729 = x722;
float x730 = x553[x729];
float x731 = x725 + x730;
x524[x724] = x731;
float x733 = x51[x727];
float x734 = x522[x724];
float x735 = x50[x727];
float x736 = x553[x729];
float x737 = x733 + x736;
x51[x727] = x737;
x722 += 1;
x720 += 1;
x721 += 1;

}
x715 += 10;
x713 += 10;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,50,10,1,x524,10,x40,10,1,x520,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,100,1,x504,50,x524,10,1,x49,10);
float* x750 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x751 = 0;
int32_t x752 = 0;
int32_t x753 = 0;
for(int x754=0; x754 < 100; x754++) {
int32_t x755 = x752;
int32_t x756 = x753;
int32_t x757 = x751;
int32_t x758 = x757;
int32_t x759 = x755;
int32_t x760 = x756;
for(int x761=0; x761 < 50; x761++) {
int32_t x762 = x758;
int32_t x763 = x759;
float x764 = x505[x763];
int32_t x765 = x760;
float x766 = x520[x765];
float x767 = x764 * x766;
x750[x762] = x767;
x758 += 1;
x759 += 1;
x760 += 1;

}
x751 += 50;
x752 += 50;
x753 += 50;

}
for(int x779=0; x779 < 5000; x779++) {
float x780 = x503[x779];
float x781 = x750[x779];
float x782 = x780 + x781;
x503[x779] = x782;

}
int32_t x786 = 0;
int32_t x787 = 0;
int32_t x788 = 0;
for(int x789=0; x789 < 100; x789++) {
int32_t x790 = x786;
int32_t x791 = x787;
int32_t x792 = x788;
int32_t x793 = x790;
int32_t x794 = x791;
int32_t x795 = x792;
for(int x796=0; x796 < 50; x796++) {
int32_t x797 = x793;
float x798 = x473[x797];
float x799 = x471[x797];
int32_t x800 = x794;
float x801 = x38[x800];
int32_t x802 = x795;
float x803 = x503[x802];
float x804 = x798 + x803;
x473[x797] = x804;
float x806 = x39[x800];
float x807 = x471[x797];
float x808 = x38[x800];
float x809 = x503[x802];
float x810 = x806 + x809;
x39[x800] = x810;
x795 += 1;
x793 += 1;
x794 += 1;

}
x788 += 50;
x786 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,320,50,1,x473,50,x28,50,1,x469,320);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,100,1,x376,320,x473,50,1,x37,50);
for(int x823=0; x823 < 32000; x823++) {
int32_t x824 = x382[x823];
float x825 = x469[x823];
x375[x824] = x825;

}
for(int x829=0; x829 < 128000; x829++) {
float x830 = x300[x829];
bool x831 = x830 < 0.0f;
float x834;
if (x831) {
x834 = 0.0f;
} else {
float x832 = x375[x829];
x834 = x832;
}
x361[x829] = x834;

}
// conv2D back-propagate
float* x839 = (float*)myMalloc(1600000 * sizeof(float));;
float* x840 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x841 = 0;
for(int x842=0; x842 < 250; x842++) {
int32_t x843 = 0;
for(int x844=0; x844 < 20; x844++) {
int32_t x845 = x841;
int32_t x847 = x843;
int32_t x848 = x847 + x842;
float x849 = x16[x848];
int32_t x846 = x845 + x844;
x840[x846] = x849;
x843 += 250;

}
x841 += 20;

}
for(int x857=0; x857 < 100; x857++) {
int32_t x858 = x857 * 1440;
float* x859 = x299+x858;
int32_t x860 = x857 * 1280;
float* x861 = x361+x860;
int32_t x862 = x857 * 16000;
float* x863 = x839+x862;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 250,64,20,1,x840,20,x861,64,0,x863,64);
for(int x865=0; x865 < 10; x865++) {
int32_t x869 = x865 * 5;
int32_t x870 = x869 * 5;
int32_t x871 = x870 * 8;
int32_t x872 = x871 * 8;
int32_t x881 = x865 * 12;
int32_t x882 = x881 * 12;
for(int x867=0; x867 < 5; x867++) {
int32_t x873 = x867 * 5;
int32_t x874 = x873 * 8;
int32_t x875 = x874 * 8;
int32_t x876 = x872 + x875;
for(int x868=0; x868 < 5; x868++) {
int32_t x877 = x868 * 8;
int32_t x878 = x877 * 8;
int32_t x879 = x876 + x878;
float* x880 = x863+x879;
float* x883 = x859+x882;
for(int x884=0; x884 < 8; x884++) {
int32_t x885 = x884 + x867;
int32_t x886 = x885 * 12;
int32_t x887 = x886 + x868;
float* x888 = x883+x887;
int32_t x889 = x884 * 8;
float* x890 = x880+x889;
for(int x891=0; x891 < 8; x891++) {
float x892 = x888[x891];
float x893 = x890[x891];
float x894 = x892 + x893;
x888[x891] = x894;

}

}

}

}

}

}
for(int x908=0; x908 < 100; x908++) {
int32_t x909 = x908 * 1280;
float* x910 = x361+x909;
int32_t x911 = x908 * 16000;
float* x912 = x317+x911;
float* x913 = (float*)myMalloc(16000 * sizeof(float));;
int32_t x914 = 0;
for(int x915=0; x915 < 64; x915++) {
int32_t x916 = 0;
for(int x917=0; x917 < 250; x917++) {
int32_t x918 = x914;
int32_t x920 = x916;
int32_t x921 = x920 + x915;
float x922 = x912[x921];
int32_t x919 = x918 + x917;
x913[x919] = x922;
x916 += 64;

}
x914 += 250;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,250,64,1.0,x910,64,x913,250,1,x25,250);
for(int x931=0; x931 < 20; x931++) {
float x932 = 0.0f;
int32_t x933 = x931 * 8;
int32_t x934 = x933 * 8;
float* x935 = x910+x934;
for(int x936=0; x936 < 64; x936++) {
float x937 = x935[x936];
x932 += x937;

}
float x941 = x27[x931];
float x942 = x932;
float x943 = 1.0f * x942;
float x944 = x941 + x943;
x27[x931] = x944;

}

}
for(int x950=0; x950 < 144000; x950++) {
int32_t x951 = x211[x950];
float x952 = x299[x950];
x204[x951] = x952;

}
for(int x956=0; x956 < 576000; x956++) {
float x957 = x127[x956];
bool x958 = x957 < 0.0f;
float x961;
if (x958) {
x961 = 0.0f;
} else {
float x959 = x204[x956];
x961 = x959;
}
x190[x956] = x961;

}
// conv2D back-propagate
float* x966 = (float*)myMalloc(1440000 * sizeof(float));;
for(int x967=0; x967 < 100; x967++) {
int32_t x968 = x967 * 5760;
float* x969 = x190+x968;
int32_t x970 = x967 * 14400;
float* x971 = x145+x970;
float* x972 = (float*)myMalloc(14400 * sizeof(float));;
int32_t x973 = 0;
for(int x974=0; x974 < 576; x974++) {
int32_t x975 = 0;
for(int x976=0; x976 < 25; x976++) {
int32_t x977 = x973;
int32_t x979 = x975;
int32_t x980 = x979 + x974;
float x981 = x971[x980];
int32_t x978 = x977 + x976;
x972[x978] = x981;
x975 += 576;

}
x973 += 25;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 10,25,576,1.0,x969,576,x972,25,1,x13,25);
for(int x990=0; x990 < 10; x990++) {
float x991 = 0.0f;
int32_t x992 = x990 * 24;
int32_t x993 = x992 * 24;
float* x994 = x969+x993;
for(int x995=0; x995 < 576; x995++) {
float x996 = x994[x995];
x991 += x996;

}
float x1000 = x15[x990];
float x1001 = x991;
float x1002 = 1.0f * x1001;
float x1003 = x1000 + x1002;
x15[x990] = x1003;

}

}
float x1009 = x126[0];
x106 += x1009;
for(int x1011=0; x1011 < 5000; x1011++) {
float x1012 = x25[x1011];
float x1013 = x1012;
float x1014 = x1013;
bool x1015 = x1014 > 1000.0f;
if (x1015) {
x1013 = 1000.0f;
} else {
}
float x1019 = x1013;
bool x1020 = x1019 < -1000.0f;
if (x1020) {
x1013 = -1000.0f;
} else {
}
float x1024 = x16[x1011];
float x1025 = x1013;
float x1026 = 5.0E-4f * x1025;
float x1027 = x1024 - x1026;
x16[x1011] = x1027;
x25[x1011] = 0.0f;

}
for(int x1032=0; x1032 < 16000; x1032++) {
float x1033 = x37[x1032];
float x1034 = x1033;
float x1035 = x1034;
bool x1036 = x1035 > 1000.0f;
if (x1036) {
x1034 = 1000.0f;
} else {
}
float x1040 = x1034;
bool x1041 = x1040 < -1000.0f;
if (x1041) {
x1034 = -1000.0f;
} else {
}
float x1045 = x28[x1032];
float x1046 = x1034;
float x1047 = 5.0E-4f * x1046;
float x1048 = x1045 - x1047;
x28[x1032] = x1048;
x37[x1032] = 0.0f;

}
for(int x1053=0; x1053 < 50; x1053++) {
float x1054 = x39[x1053];
float x1055 = x1054;
float x1056 = x1055;
bool x1057 = x1056 > 1000.0f;
if (x1057) {
x1055 = 1000.0f;
} else {
}
float x1061 = x1055;
bool x1062 = x1061 < -1000.0f;
if (x1062) {
x1055 = -1000.0f;
} else {
}
float x1066 = x38[x1053];
float x1067 = x1055;
float x1068 = 5.0E-4f * x1067;
float x1069 = x1066 - x1068;
x38[x1053] = x1069;
x39[x1053] = 0.0f;

}
for(int x1074=0; x1074 < 250; x1074++) {
float x1075 = x13[x1074];
float x1076 = x1075;
float x1077 = x1076;
bool x1078 = x1077 > 1000.0f;
if (x1078) {
x1076 = 1000.0f;
} else {
}
float x1082 = x1076;
bool x1083 = x1082 < -1000.0f;
if (x1083) {
x1076 = -1000.0f;
} else {
}
float x1087 = x4[x1074];
float x1088 = x1076;
float x1089 = 5.0E-4f * x1088;
float x1090 = x1087 - x1089;
x4[x1074] = x1090;
x13[x1074] = 0.0f;

}
for(int x1095=0; x1095 < 10; x1095++) {
float x1096 = x51[x1095];
float x1097 = x1096;
float x1098 = x1097;
bool x1099 = x1098 > 1000.0f;
if (x1099) {
x1097 = 1000.0f;
} else {
}
float x1103 = x1097;
bool x1104 = x1103 < -1000.0f;
if (x1104) {
x1097 = -1000.0f;
} else {
}
float x1108 = x50[x1095];
float x1109 = x1097;
float x1110 = 5.0E-4f * x1109;
float x1111 = x1108 - x1110;
x50[x1095] = x1111;
x51[x1095] = 0.0f;

}
for(int x1116=0; x1116 < 500; x1116++) {
float x1117 = x49[x1116];
float x1118 = x1117;
float x1119 = x1118;
bool x1120 = x1119 > 1000.0f;
if (x1120) {
x1118 = 1000.0f;
} else {
}
float x1124 = x1118;
bool x1125 = x1124 < -1000.0f;
if (x1125) {
x1118 = -1000.0f;
} else {
}
float x1129 = x40[x1116];
float x1130 = x1118;
float x1131 = 5.0E-4f * x1130;
float x1132 = x1129 - x1131;
x40[x1116] = x1132;
x49[x1116] = 0.0f;

}
int32_t x1137 = x103;
int32_t x1139 = x1137 % x1138;
bool x1140 = x1139 == 0;
if (x1140) {
float x1145 = x106;
double x1141 = (double)x1137;
double x1142 = 100.0 * x1141;
double x1144 = x1142 / x1143;
float x1146 = (float)x1137;
float x1147 = x1145 / x1146;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x99,x1137,x63,x1144,x1147);
fflush(stdout);
} else {
}
int64_t x1152 = (long)mallocAddr;
int64_t x1153 = x1152 - x97;
memset((void*)x97, 0, x1153);
mallocAddr = (void*)x97;
x112 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1161 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1162 = x1161 / 1000LL;
int64_t x1164 = x1161 / x1163;
printf("Training completed in %ldms (%ld us/images)\n",x1162,x1164);
float x1166 = x106;
float x1168 = x1166 / x1167;
double x1169 = (double)x1168;
x96[x99] = x1169;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1175 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1180 = (long)fopen(x0, "w");
fprintf((FILE *)x1180, "unit: %s\n", "1 epoch");
for(int x1182=0; x1182 < 4; x1182++) {
double x1183 = x96[x1182];
fprintf((FILE *)x1180, "%lf\n", x1183);

}
float x1176 = (float)x1175;
float x1177 = x1176 / 1000000.0f;
float x1178 = x1177 - x94;
float x1179 = x1178 / 4.0f;
fprintf((FILE *)x1180, "run time: %lf %lf\n", x94, x1179);
fclose((FILE*)x1180);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

