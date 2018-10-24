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
// Backend setup.
srand(42);
struct timeval begin_0, end_0, diff_0;
gettimeofday(&begin_0, NULL);
float* x5 = (float*)myMalloc(250 * sizeof(float));;
for(int x7=0; x7 < 250; x7++) {
float x8 = (float)rand()/RAND_MAX;
float x9 = x8 - 0.5f;
float x10 = x9 * 0.2f;
x5[x7] = x10;

}
float* x14 = (float*)myMalloc(250 * sizeof(float));;
float* x15 = (float*)myMalloc(10 * sizeof(float));;
float* x16 = (float*)myMalloc(10 * sizeof(float));;
float* x17 = (float*)myMalloc(5000 * sizeof(float));;
for(int x19=0; x19 < 5000; x19++) {
float x20 = (float)rand()/RAND_MAX;
float x21 = x20 - 0.5f;
float x22 = x21 * 0.06324556f;
x17[x19] = x22;

}
float* x26 = (float*)myMalloc(5000 * sizeof(float));;
float* x27 = (float*)myMalloc(20 * sizeof(float));;
float* x28 = (float*)myMalloc(20 * sizeof(float));;
float* x29 = (float*)myMalloc(16000 * sizeof(float));;
for(int x31=0; x31 < 16000; x31++) {
float x32 = (float)rand()/RAND_MAX;
float x33 = x32 - 0.5f;
float x34 = x33 * 0.0559017f;
x29[x31] = x34;

}
float* x38 = (float*)myMalloc(16000 * sizeof(float));;
float* x39 = (float*)myMalloc(50 * sizeof(float));;
float* x40 = (float*)myMalloc(50 * sizeof(float));;
float* x41 = (float*)myMalloc(500 * sizeof(float));;
for(int x43=0; x43 < 500; x43++) {
float x44 = (float)rand()/RAND_MAX;
float x45 = x44 - 0.5f;
float x46 = x45 * 0.14142136f;
x41[x43] = x46;

}
float* x50 = (float*)myMalloc(500 * sizeof(float));;
float* x51 = (float*)myMalloc(10 * sizeof(float));;
float* x52 = (float*)myMalloc(10 * sizeof(float));;
int64_t* x53 = (int64_t*)myMalloc(2 * sizeof(int64_t));;
int64_t* x54 = (int64_t*)myMalloc(2 * sizeof(int64_t));;
int32_t x65 = 0;
int32_t x66 = x65;
int32_t x67 = x66;
int32_t x61 = open("../data/bin/mnist_train_target.bin",0);
int32_t x62 = fsize(x61);
int32_t x64 = x62 / 4;
int* x63 = (int*)mmap(0, x62, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x61, 0);
int32_t x55 = open("../data/bin/mnist_train.bin",0);
int32_t x56 = fsize(x55);
float* x57 = (float*)mmap(0, x56, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x55, 0);
for(int x69=0; x69 < x64; x69++) {
int32_t x70 = x67;
int32_t x72 = x63[x69];
float* x71 = x57+x70;
for(int x74=0; x74 < 784; x74++) {
float x75 = x71[x74];
float x76 = x75 - 0.1307f;
float x77 = x76 / 0.3081f;
x71[x74] = x77;

}
x67 += 784;

}
int32_t x84 = x67;
int64_t x58 = (int64_t)x56;
int64_t x59 = x58 / 4LL;
int32_t x60 = (int32_t)x59;
bool x85 = x84 == x60;
if (x85) {
} else {
printf("Data length doesn't match\n");
exit(0);
}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x93 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x94 = (float)x93;
float x95 = x94 / 1000000.0f;
printf("Data normalized (all prepare time) in %lf sec\n",x95);
double* x97 = (double*)myMalloc(4 * sizeof(double));;
int64_t x98 = (long)mallocAddr;
// training loop starts here
int32_t x115 = x64 / 100;
int32_t x1052 = x64 / 10;
double x1057 = (double)x64;
int64_t x1077 = (int64_t)x64;
float x1081 = (float)x64;
for(int x101=0; x101 < 4; x101++) {
struct timeval begin_1, end_1, diff_1;
int32_t x103 = 0;
int32_t x104 = x103;
int32_t x105 = x104;
float x106 = 0.0f;
float x107 = x106;
float x108 = x107;
int32_t x109 = x101 + 1;
printf("Start training epoch %d\n",x109);
gettimeofday(&begin_1, NULL);
int32_t x112 = 0;
int32_t x113 = x112;
int32_t x114 = x113;
for(int x117=0; x117 < x115; x117++) {
int32_t x118 = x114;
x105 += 100;
float* x123 = (float*)myMalloc(2 * sizeof(float));;
float* x124 = (float*)myMalloc(4 * sizeof(float));;
float* x125 = (float*)myMalloc(4 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
float* x127 = (float*)myMalloc(1 * sizeof(float));;
float* x128 = (float*)myMalloc(576000 * sizeof(float));;
int32_t x129 = 0;
for(int x131=0; x131 < 100; x131++) {
for(int x133=0; x133 < 10; x133++) {
for(int x135=0; x135 < 576; x135++) {
int32_t x136 = x129;
float x137 = x15[x133];
x128[x136] = x137;
x129 += 1;

}

}

}
float* x146 = (float*)myMalloc(1440000 * sizeof(float));;
float* x119 = x57+x118;
for(int x147=0; x147 < 100; x147++) {
int32_t x150 = x147 * 5760;
float* x151 = x128+x150;
int32_t x152 = x147 * 14400;
float* x153 = x146+x152;
int32_t x148 = x147 * 784;
float* x149 = x119+x148;
for(int x155=0; x155 < 25; x155++) {
int32_t x156 = x155 / 25;
int32_t x160 = x156 * 5;
int32_t x161 = x160 * 5;
int32_t x162 = x161 * 24;
int32_t x163 = x162 * 24;
int32_t x157 = x155 % 25;
int32_t x158 = x157 / 5;
int32_t x164 = x158 * 5;
int32_t x165 = x164 * 24;
int32_t x166 = x165 * 24;
int32_t x167 = x163 + x166;
int32_t x159 = x157 % 5;
int32_t x168 = x159 * 24;
int32_t x169 = x168 * 24;
int32_t x170 = x167 + x169;
float* x171 = x153+x170;
int32_t x172 = x156 * 28;
int32_t x173 = x172 * 28;
float* x174 = x149+x173;
for(int x176=0; x176 < 24; x176++) {
int32_t x178 = x176 * 24;
float* x179 = x171+x178;
int32_t x177 = x176 + x158;
int32_t x180 = x177 * 28;
int32_t x181 = x180 + x159;
float* x182 = x174+x181;
memcpy(x179, x182, 4 * 24);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 10,576,25,1,x5,25,x153,576,1,x151,576);

}
float* x191 = (float*)myMalloc(576000 * sizeof(float));;
float* x192 = (float*)myMalloc(576000 * sizeof(float));;
for(int x194=0; x194 < 576000; x194++) {
float x195 = x128[x194];
bool x196 = x195 < 0.0f;
if (x196) {
x192[x194] = 0.0f;
} else {
float x199 = x128[x194];
x192[x194] = x199;
}

}
float* x205 = (float*)myMalloc(576000 * sizeof(float));;
float* x206 = (float*)myMalloc(144000 * sizeof(float));;
for(int x208=0; x208 < 144000; x208++) {
x206[x208] = -3.4028235E38f;

}
int* x212 = (int32_t*)myMalloc(144000 * sizeof(int32_t));;
for(int x213=0; x213 < 100; x213++) {
int32_t x214 = x213 * 5760;
float* x215 = x192+x214;
int32_t x216 = x213 * 1440;
float* x217 = x206+x216;
int* x218 = x212+x216;
int32_t x219 = 0;
int32_t x220 = 0;
for(int x221=0; x221 < 10; x221++) {
int32_t x222 = x219;
int32_t x223 = x222;
int32_t x224 = x220;
int32_t x225 = x224;
for(int x227=0; x227 < 12; x227++) {
int32_t x228 = x223;
int32_t x229 = x228;
int32_t x230 = x225;
int32_t x231 = x230;
for(int x232=0; x232 < 12; x232++) {
int32_t x233 = x231;
int32_t x234 = x233;
int32_t x235 = x234;
int32_t x236 = x235;
int32_t x237 = x236;
float x238 = x215[x237];
int32_t x239 = x229;
float x240 = x217[x239];
bool x241 = x238 > x240;
if (x241) {
float x242 = x215[x237];
x217[x239] = x242;
int32_t x244 = x237 + x214;
x218[x239] = x244;
} else {
}
x236 += 1;
int32_t x249 = x236;
float x250 = x215[x249];
float x251 = x217[x239];
bool x252 = x250 > x251;
if (x252) {
float x253 = x215[x249];
x217[x239] = x253;
int32_t x255 = x249 + x214;
x218[x239] = x255;
} else {
}
x236 += 1;
x234 += 24;
int32_t x261 = x234;
int32_t x262 = x261;
int32_t x263 = x262;
float x264 = x215[x263];
float x265 = x217[x239];
bool x266 = x264 > x265;
if (x266) {
float x267 = x215[x263];
x217[x239] = x267;
int32_t x269 = x263 + x214;
x218[x239] = x269;
} else {
}
x262 += 1;
int32_t x274 = x262;
float x275 = x215[x274];
float x276 = x217[x239];
bool x277 = x275 > x276;
if (x277) {
float x278 = x215[x274];
x217[x239] = x278;
int32_t x280 = x274 + x214;
x218[x239] = x280;
} else {
}
x262 += 1;
x234 += 24;
x229 += 1;
x231 += 2;

}
x223 += 12;
x225 += 48;

}
x219 += 144;
x220 += 576;

}

}
float* x300 = (float*)myMalloc(144000 * sizeof(float));;
float* x301 = (float*)myMalloc(128000 * sizeof(float));;
int32_t x302 = 0;
for(int x303=0; x303 < 100; x303++) {
for(int x305=0; x305 < 20; x305++) {
for(int x307=0; x307 < 64; x307++) {
int32_t x308 = x302;
float x309 = x27[x305];
x301[x308] = x309;
x302 += 1;

}

}

}
float* x318 = (float*)myMalloc(1600000 * sizeof(float));;
for(int x319=0; x319 < 100; x319++) {
int32_t x320 = x319 * 1440;
float* x321 = x206+x320;
int32_t x322 = x319 * 1280;
float* x323 = x301+x322;
int32_t x324 = x319 * 16000;
float* x325 = x318+x324;
for(int x326=0; x326 < 250; x326++) {
int32_t x327 = x326 / 25;
int32_t x331 = x327 * 5;
int32_t x332 = x331 * 5;
int32_t x333 = x332 * 8;
int32_t x334 = x333 * 8;
int32_t x328 = x326 % 25;
int32_t x329 = x328 / 5;
int32_t x335 = x329 * 5;
int32_t x336 = x335 * 8;
int32_t x337 = x336 * 8;
int32_t x338 = x334 + x337;
int32_t x330 = x328 % 5;
int32_t x339 = x330 * 8;
int32_t x340 = x339 * 8;
int32_t x341 = x338 + x340;
float* x342 = x325+x341;
int32_t x343 = x327 * 12;
int32_t x344 = x343 * 12;
float* x345 = x321+x344;
for(int x347=0; x347 < 8; x347++) {
int32_t x349 = x347 * 8;
float* x350 = x342+x349;
int32_t x348 = x347 + x329;
int32_t x351 = x348 * 12;
int32_t x352 = x351 + x330;
float* x353 = x345+x352;
memcpy(x350, x353, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,64,250,1,x17,250,x325,64,1,x323,64);

}
float* x362 = (float*)myMalloc(128000 * sizeof(float));;
float* x363 = (float*)myMalloc(128000 * sizeof(float));;
for(int x365=0; x365 < 128000; x365++) {
float x366 = x301[x365];
bool x367 = x366 < 0.0f;
if (x367) {
x363[x365] = 0.0f;
} else {
float x370 = x301[x365];
x363[x365] = x370;
}

}
float* x376 = (float*)myMalloc(128000 * sizeof(float));;
float* x377 = (float*)myMalloc(32000 * sizeof(float));;
for(int x379=0; x379 < 32000; x379++) {
x377[x379] = -3.4028235E38f;

}
int* x383 = (int32_t*)myMalloc(32000 * sizeof(int32_t));;
for(int x384=0; x384 < 100; x384++) {
int32_t x385 = x384 * 1280;
float* x386 = x363+x385;
int32_t x387 = x384 * 320;
float* x388 = x377+x387;
int* x389 = x383+x387;
int32_t x390 = 0;
int32_t x391 = 0;
for(int x392=0; x392 < 20; x392++) {
int32_t x393 = x390;
int32_t x394 = x393;
int32_t x395 = x391;
int32_t x396 = x395;
for(int x397=0; x397 < 4; x397++) {
int32_t x398 = x394;
int32_t x399 = x398;
int32_t x400 = x396;
int32_t x401 = x400;
for(int x402=0; x402 < 4; x402++) {
int32_t x403 = x401;
int32_t x404 = x403;
int32_t x405 = x404;
int32_t x406 = x405;
int32_t x407 = x406;
float x408 = x386[x407];
int32_t x409 = x399;
float x410 = x388[x409];
bool x411 = x408 > x410;
if (x411) {
float x412 = x386[x407];
x388[x409] = x412;
int32_t x414 = x407 + x385;
x389[x409] = x414;
} else {
}
x406 += 1;
int32_t x419 = x406;
float x420 = x386[x419];
float x421 = x388[x409];
bool x422 = x420 > x421;
if (x422) {
float x423 = x386[x419];
x388[x409] = x423;
int32_t x425 = x419 + x385;
x389[x409] = x425;
} else {
}
x406 += 1;
x404 += 8;
int32_t x431 = x404;
int32_t x432 = x431;
int32_t x433 = x432;
float x434 = x386[x433];
float x435 = x388[x409];
bool x436 = x434 > x435;
if (x436) {
float x437 = x386[x433];
x388[x409] = x437;
int32_t x439 = x433 + x385;
x389[x409] = x439;
} else {
}
x432 += 1;
int32_t x444 = x432;
float x445 = x386[x444];
float x446 = x388[x409];
bool x447 = x445 > x446;
if (x447) {
float x448 = x386[x444];
x388[x409] = x448;
int32_t x450 = x444 + x385;
x389[x409] = x450;
} else {
}
x432 += 1;
x404 += 8;
x399 += 1;
x401 += 2;

}
x394 += 4;
x396 += 16;

}
x390 += 16;
x391 += 64;

}

}
float* x470 = (float*)myMalloc(32000 * sizeof(float));;
float* x471 = (float*)myMalloc(5000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 100,50,320,1,x377,320,x29,50,0,x471,50);
float* x473 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x474 = 0;
int32_t x475 = 0;
int32_t x476 = 0;
for(int x477=0; x477 < 100; x477++) {
int32_t x478 = x475;
int32_t x479 = x476;
int32_t x480 = x474;
int32_t x481 = x480;
int32_t x482 = x478;
int32_t x483 = x479;
for(int x485=0; x485 < 50; x485++) {
int32_t x486 = x482;
float x487 = x471[x486];
int32_t x488 = x483;
float x489 = x39[x488];
float x490 = x487 + x489;
x471[x486] = x490;
x481 += 1;
x482 += 1;
x483 += 1;

}
x474 += 50;
x475 += 50;

}
float* x501 = (float*)myMalloc(5000 * sizeof(float));;
float* x502 = (float*)myMalloc(5000 * sizeof(float));;
for(int x503=0; x503 < 5000; x503++) {
float x504 = (float)rand()/RAND_MAX;
bool x505 = x504 > 0.5f;
if (x505) {
float x506 = x471[x503];
float x507 = x506 * 2.0f;
x501[x503] = x507;
x502[x503] = 2.0f;
} else {
x501[x503] = 0.0f;
x502[x503] = 0.0f;
}

}
float* x517 = (float*)myMalloc(5000 * sizeof(float));;
float* x518 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 100,10,50,1,x501,50,x41,10,0,x518,10);
float* x520 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x521 = 0;
int32_t x522 = 0;
int32_t x523 = 0;
for(int x524=0; x524 < 100; x524++) {
int32_t x525 = x522;
int32_t x526 = x523;
int32_t x527 = x521;
int32_t x528 = x527;
int32_t x529 = x525;
int32_t x530 = x526;
for(int x531=0; x531 < 10; x531++) {
int32_t x532 = x529;
float x533 = x518[x532];
int32_t x534 = x530;
float x535 = x51[x534];
float x536 = x533 + x535;
x518[x532] = x536;
x528 += 1;
x529 += 1;
x530 += 1;

}
x521 += 10;
x522 += 10;

}
float* x547 = (float*)myMalloc(100 * sizeof(float));;
int32_t x548 = 0;
for(int x549=0; x549 < 100; x549++) {
float x550 = -3.4028235E38f;
for(int x551=0; x551 < 10; x551++) {
int32_t x552 = x548;
float x553 = x518[x552];
float x554 = x550;
bool x555 = x553 > x554;
if (x555) {
float x556 = x518[x552];
x550 = x556;
} else {
}
x548 += 1;

}
float x563 = x550;
x547[x549] = x563;

}
float* x567 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x568 = 0;
for(int x569=0; x569 < 100; x569++) {
for(int x570=0; x570 < 10; x570++) {
int32_t x571 = x568;
float x572 = x518[x571];
float x573 = x547[x569];
float x574 = x572 - x573;
double x575 = (double)x574;
double x576 = exp(x575);
float x577 = (float)x576;
x567[x571] = x577;
x568 += 1;

}

}
float* x584 = (float*)myMalloc(100 * sizeof(float));;
for(int x585=0; x585 < 100; x585++) {
int32_t x586 = x585;
int32_t x587 = x585 * 10;
int32_t x588 = x587;
for(int x589=0; x589 < 10; x589++) {
int32_t x590 = x586;
float x591 = x584[x590];
int32_t x592 = x588;
float x593 = x567[x592];
float x594 = x591 + x593;
x584[x590] = x594;
x588 += 1;

}

}
x568 = 0;
for(int x602=0; x602 < 100; x602++) {
float x603 = x547[x602];
float x604 = x584[x602];
double x605 = (double)x604;
double x606 = log(x605);
float x607 = (float)x606;
float x608 = x603 + x607;
for(int x609=0; x609 < 10; x609++) {
int32_t x610 = x568;
float x611 = x518[x610];
float x612 = x611 - x608;
x567[x610] = x612;
x568 += 1;

}

}
float* x619 = (float*)myMalloc(1000 * sizeof(float));;
// nllLoss forward in CPU
float* x621 = (float*)myMalloc(100 * sizeof(float));;
int32_t x622 = 0;
int32_t x120 = x117 * 100;
int* x121 = x63+x120;
for(int x623=0; x623 < 100; x623++) {
int32_t x624 = x622;
int32_t x625 = x121[x623];
int32_t x626 = x624 + x625;
float x627 = x567[x626];
float x628 = -1.0f * x627;
x621[x623] = x628;
x622 += 10;

}
float* x633 = (float*)myMalloc(100 * sizeof(float));;
// CPU sum function called here
float x635 = 0.0f;
for(int x636=0; x636 < 100; x636++) {
float x637 = x635;
float x638 = x621[x636];
float x639 = x637 + x638;
x635 = x639;

}
float x643 = x635;
float* x644 = (float*)myMalloc(1 * sizeof(float));;
x644[0] = x643;
float* x646 = (float*)myMalloc(1 * sizeof(float));;
x646[0] = 1.0f;
// backend is lantern.TensorDsl$BackendCPU@52571d55
float x649 = x644[0];
x127[0] = x649;
// 'sum' gradient.
int32_t x652 = 0;
int32_t x653 = 0;
int32_t x654 = 0;
for(int x655=0; x655 < 100; x655++) {
int32_t x656 = x653;
float x657 = x633[x656];
int32_t x658 = x654;
float x659 = x646[x658];
float x660 = x657 + x659;
x633[x656] = x660;
x652 += 1;
x653 += 1;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x668 = 0;
for(int x669=0; x669 < 100; x669++) {
int32_t x670 = x668;
int32_t x671 = x121[x669];
int32_t x672 = x670 + x671;
float x673 = x619[x672];
float x674 = x633[x669];
float x675 = -1.0f * x674;
float x676 = x673 + x675;
x619[x672] = x676;
x668 += 10;

}
float* x681 = (float*)myMalloc(100 * sizeof(float));;
for(int x682=0; x682 < 100; x682++) {
int32_t x683 = x682;
int32_t x684 = x682 * 10;
int32_t x685 = x684;
for(int x686=0; x686 < 10; x686++) {
int32_t x687 = x683;
float x688 = x681[x687];
int32_t x689 = x685;
float x690 = x619[x689];
float x691 = x688 + x690;
x681[x687] = x691;
x685 += 1;

}

}
int32_t x698 = 0;
for(int x699=0; x699 < 100; x699++) {
for(int x700=0; x700 < 10; x700++) {
int32_t x701 = x698;
float x702 = x520[x701];
float x703 = x619[x701];
float x704 = x567[x701];
float x708 = x681[x699];
double x705 = (double)x704;
double x706 = exp(x705);
float x707 = (float)x706;
float x709 = x707 * x708;
float x710 = x703 - x709;
float x711 = x702 + x710;
x520[x701] = x711;
x698 += 1;

}

}
int32_t x718 = 0;
int32_t x719 = 0;
int32_t x720 = 0;
for(int x721=0; x721 < 100; x721++) {
int32_t x722 = x719;
int32_t x723 = x720;
int32_t x724 = x718;
int32_t x725 = x724;
int32_t x726 = x722;
int32_t x727 = x723;
for(int x728=0; x728 < 10; x728++) {
int32_t x729 = x726;
float x730 = x52[x729];
int32_t x731 = x727;
float x732 = x520[x731];
float x733 = x730 + x732;
x52[x729] = x733;
x725 += 1;
x726 += 1;
x727 += 1;

}
x718 += 10;
x720 += 10;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,50,10,1,x520,10,x41,10,1,x517,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,100,1,x501,50,x520,10,1,x50,10);
float* x747 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x748 = 0;
int32_t x749 = 0;
int32_t x750 = 0;
for(int x751=0; x751 < 100; x751++) {
int32_t x752 = x749;
int32_t x753 = x750;
int32_t x754 = x748;
int32_t x755 = x754;
int32_t x756 = x752;
int32_t x757 = x753;
for(int x758=0; x758 < 50; x758++) {
int32_t x759 = x755;
int32_t x760 = x756;
float x761 = x502[x760];
int32_t x762 = x757;
float x763 = x517[x762];
float x764 = x761 * x763;
x747[x759] = x764;
x755 += 1;
x756 += 1;
x757 += 1;

}
x748 += 50;
x749 += 50;
x750 += 50;

}
int32_t x776 = 0;
int32_t x777 = 0;
int32_t x778 = 0;
for(int x779=0; x779 < 100; x779++) {
int32_t x780 = x777;
int32_t x781 = x778;
int32_t x782 = x776;
int32_t x783 = x782;
int32_t x784 = x780;
int32_t x785 = x781;
for(int x786=0; x786 < 50; x786++) {
int32_t x787 = x784;
float x788 = x473[x787];
int32_t x789 = x785;
float x790 = x747[x789];
float x791 = x788 + x790;
x473[x787] = x791;
x783 += 1;
x784 += 1;
x785 += 1;

}
x776 += 50;
x777 += 50;
x778 += 50;

}
int32_t x803 = 0;
int32_t x804 = 0;
int32_t x805 = 0;
for(int x806=0; x806 < 100; x806++) {
int32_t x807 = x804;
int32_t x808 = x805;
int32_t x809 = x803;
int32_t x810 = x809;
int32_t x811 = x807;
int32_t x812 = x808;
for(int x813=0; x813 < 50; x813++) {
int32_t x814 = x811;
float x815 = x40[x814];
int32_t x816 = x812;
float x817 = x473[x816];
float x818 = x815 + x817;
x40[x814] = x818;
x810 += 1;
x811 += 1;
x812 += 1;

}
x803 += 50;
x805 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,320,50,1,x473,50,x29,50,1,x470,320);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,100,1,x377,320,x473,50,1,x38,50);
for(int x832=0; x832 < 32000; x832++) {
int32_t x833 = x383[x832];
float x834 = x376[x833];
float x835 = x470[x832];
float x836 = x834 + x835;
x376[x833] = x836;

}
for(int x840=0; x840 < 128000; x840++) {
float x841 = x301[x840];
bool x842 = x841 < 0.0f;
float x845;
if (x842) {
x845 = 0.0f;
} else {
float x843 = x376[x840];
x845 = x843;
}
x362[x840] = x845;

}
// conv2D back-propagate
float* x850 = (float*)myMalloc(1600000 * sizeof(float));;
for(int x851=0; x851 < 100; x851++) {
int32_t x852 = x851 * 1440;
float* x853 = x300+x852;
int32_t x854 = x851 * 1280;
float* x855 = x362+x854;
int32_t x856 = x851 * 16000;
float* x857 = x850+x856;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 250,64,20,1,x17,250,x855,64,0,x857,64);
for(int x859=0; x859 < 10; x859++) {
int32_t x863 = x859 * 5;
int32_t x864 = x863 * 5;
int32_t x865 = x864 * 8;
int32_t x866 = x865 * 8;
int32_t x875 = x859 * 12;
int32_t x876 = x875 * 12;
for(int x861=0; x861 < 5; x861++) {
int32_t x867 = x861 * 5;
int32_t x868 = x867 * 8;
int32_t x869 = x868 * 8;
int32_t x870 = x866 + x869;
for(int x862=0; x862 < 5; x862++) {
int32_t x871 = x862 * 8;
int32_t x872 = x871 * 8;
int32_t x873 = x870 + x872;
float* x874 = x857+x873;
float* x877 = x853+x876;
for(int x878=0; x878 < 8; x878++) {
int32_t x879 = x878 + x861;
int32_t x880 = x879 * 12;
int32_t x881 = x880 + x862;
float* x882 = x877+x881;
int32_t x883 = x878 * 8;
float* x884 = x874+x883;
for(int x885=0; x885 < 8; x885++) {
float x886 = x882[x885];
float x887 = x884[x885];
float x888 = x886 + x887;
x882[x885] = x888;

}

}

}

}

}

}
for(int x902=0; x902 < 100; x902++) {
int32_t x903 = x902 * 1280;
float* x904 = x362+x903;
int32_t x905 = x902 * 16000;
float* x906 = x318+x905;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,250,64,1.0,x904,64,x906,64,1,x26,250);
for(int x908=0; x908 < 20; x908++) {
float x909 = 0.0f;
int32_t x910 = x908 * 8;
int32_t x911 = x910 * 8;
float* x912 = x904+x911;
for(int x913=0; x913 < 64; x913++) {
float x914 = x912[x913];
x909 += x914;

}
float x918 = x28[x908];
float x919 = x909;
float x920 = 1.0f * x919;
float x921 = x918 + x920;
x28[x908] = x921;

}

}
for(int x927=0; x927 < 144000; x927++) {
int32_t x928 = x212[x927];
float x929 = x205[x928];
float x930 = x300[x927];
float x931 = x929 + x930;
x205[x928] = x931;

}
for(int x935=0; x935 < 576000; x935++) {
float x936 = x128[x935];
bool x937 = x936 < 0.0f;
float x940;
if (x937) {
x940 = 0.0f;
} else {
float x938 = x205[x935];
x940 = x938;
}
x191[x935] = x940;

}
// conv2D back-propagate
float* x945 = (float*)myMalloc(1440000 * sizeof(float));;
for(int x946=0; x946 < 100; x946++) {
int32_t x947 = x946 * 5760;
float* x948 = x191+x947;
int32_t x949 = x946 * 14400;
float* x950 = x146+x949;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 10,25,576,1.0,x948,576,x950,576,1,x14,25);
for(int x952=0; x952 < 10; x952++) {
float x953 = 0.0f;
int32_t x954 = x952 * 24;
int32_t x955 = x954 * 24;
float* x956 = x948+x955;
for(int x957=0; x957 < 576; x957++) {
float x958 = x956[x957];
x953 += x958;

}
float x962 = x16[x952];
float x963 = x953;
float x964 = 1.0f * x963;
float x965 = x962 + x964;
x16[x952] = x965;

}

}
float x971 = x127[0];
x108 += x971;
for(int x973=0; x973 < 5000; x973++) {
float x974 = x17[x973];
float x976 = x26[x973];
float x975 = x974 * 1.0f;
float x977 = x976 * -5.0E-4f;
float x978 = x975 + x977;
x17[x973] = x978;

}
for(int x982=0; x982 < 5000; x982++) {
x26[x982] = 0.0f;

}
for(int x986=0; x986 < 16000; x986++) {
float x987 = x29[x986];
float x989 = x38[x986];
float x988 = x987 * 1.0f;
float x990 = x989 * -5.0E-4f;
float x991 = x988 + x990;
x29[x986] = x991;

}
for(int x995=0; x995 < 16000; x995++) {
x38[x995] = 0.0f;

}
for(int x999=0; x999 < 50; x999++) {
float x1000 = x39[x999];
float x1002 = x40[x999];
float x1001 = x1000 * 1.0f;
float x1003 = x1002 * -5.0E-4f;
float x1004 = x1001 + x1003;
x39[x999] = x1004;

}
for(int x1008=0; x1008 < 50; x1008++) {
x40[x1008] = 0.0f;

}
for(int x1012=0; x1012 < 250; x1012++) {
float x1013 = x5[x1012];
float x1015 = x14[x1012];
float x1014 = x1013 * 1.0f;
float x1016 = x1015 * -5.0E-4f;
float x1017 = x1014 + x1016;
x5[x1012] = x1017;

}
for(int x1021=0; x1021 < 250; x1021++) {
x14[x1021] = 0.0f;

}
for(int x1025=0; x1025 < 10; x1025++) {
float x1026 = x51[x1025];
float x1028 = x52[x1025];
float x1027 = x1026 * 1.0f;
float x1029 = x1028 * -5.0E-4f;
float x1030 = x1027 + x1029;
x51[x1025] = x1030;

}
for(int x1034=0; x1034 < 10; x1034++) {
x52[x1034] = 0.0f;

}
for(int x1038=0; x1038 < 500; x1038++) {
float x1039 = x41[x1038];
float x1041 = x50[x1038];
float x1040 = x1039 * 1.0f;
float x1042 = x1041 * -5.0E-4f;
float x1043 = x1040 + x1042;
x41[x1038] = x1043;

}
for(int x1047=0; x1047 < 500; x1047++) {
x50[x1047] = 0.0f;

}
int32_t x1051 = x105;
int32_t x1053 = x1051 % x1052;
bool x1054 = x1053 == 0;
if (x1054) {
float x1059 = x108;
double x1055 = (double)x1051;
double x1056 = 100.0 * x1055;
double x1058 = x1056 / x1057;
float x1060 = (float)x1051;
float x1061 = x1059 / x1060;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x101,x1051,x64,x1058,x1061);
fflush(stdout);
} else {
}
int64_t x1066 = (long)mallocAddr;
int64_t x1067 = x1066 - x98;
memset((void*)x98, 0, x1067);
mallocAddr = (void*)x98;
x114 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1075 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1076 = x1075 / 1000LL;
int64_t x1078 = x1075 / x1077;
printf("Training completed in %ldms (%ld us/images)\n",x1076,x1078);
float x1080 = x108;
float x1082 = x1080 / x1081;
double x1083 = (double)x1082;
x97[x101] = x1083;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1089 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1094 = (long)fopen(x0, "w");
fprintf((FILE *)x1094, "unit: %s\n", "1 epoch");
for(int x1096=0; x1096 < 4; x1096++) {
double x1097 = x97[x1096];
fprintf((FILE *)x1094, "%lf\n", x1097);

}
float x1090 = (float)x1089;
float x1091 = x1090 / 1000000.0f;
float x1092 = x1091 - x95;
float x1093 = x1092 / 4.0f;
fprintf((FILE *)x1094, "run time: %lf %lf\n", x95, x1093);
fclose((FILE*)x1094);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

