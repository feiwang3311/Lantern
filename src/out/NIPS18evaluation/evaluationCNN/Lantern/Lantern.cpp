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

int HEAP_SIZE = 1610612739; // 2147483652; // 1073741826; // 1048576; // 536870912; // 268435456; // 2097152;
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
float x10 = x9 * 0.9797959f;
x5[x7] = x10;

}
float* x14 = (float*)myMalloc(250 * sizeof(float));;
float* x15 = (float*)myMalloc(10 * sizeof(float));;
float* x16 = (float*)myMalloc(10 * sizeof(float));;
float* x17 = (float*)myMalloc(5000 * sizeof(float));;
for(int x19=0; x19 < 5000; x19++) {
float x20 = (float)rand()/RAND_MAX;
float x21 = x20 - 0.5f;
float x22 = x21 * 0.30983868f;
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
int32_t x1077 = x64 / 10;
double x1082 = (double)x64;
int64_t x1102 = (int64_t)x64;
float x1106 = (float)x64;
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
float x634 = 0.0f;
for(int x635=0; x635 < 100; x635++) {
float x636 = x634;
float x637 = x621[x635];
float x638 = x636 + x637;
x634 = x638;

}
float x642 = x634;
float* x643 = (float*)myMalloc(1 * sizeof(float));;
x643[0] = x642;
float* x645 = (float*)myMalloc(1 * sizeof(float));;
x645[0] = 1.0f;
// backend is lantern.TensorDsl$BackendCPU@82f395e
float x648 = x643[0];
x127[0] = x648;
// 'sum' gradient.
int32_t x651 = 0;
int32_t x652 = 0;
int32_t x653 = 0;
for(int x654=0; x654 < 100; x654++) {
int32_t x655 = x652;
float x656 = x633[x655];
int32_t x657 = x653;
float x658 = x645[x657];
float x659 = x656 + x658;
x633[x655] = x659;
x651 += 1;
x652 += 1;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x667 = 0;
for(int x668=0; x668 < 100; x668++) {
int32_t x669 = x667;
int32_t x670 = x121[x668];
int32_t x671 = x669 + x670;
float x672 = x619[x671];
float x673 = x633[x668];
float x674 = -1.0f * x673;
float x675 = x672 + x674;
x619[x671] = x675;
x667 += 10;

}
float* x680 = (float*)myMalloc(100 * sizeof(float));;
for(int x681=0; x681 < 100; x681++) {
int32_t x682 = x681;
int32_t x683 = x681 * 10;
int32_t x684 = x683;
for(int x685=0; x685 < 10; x685++) {
int32_t x686 = x682;
float x687 = x680[x686];
int32_t x688 = x684;
float x689 = x619[x688];
float x690 = x687 + x689;
x680[x686] = x690;
x684 += 1;

}

}
int32_t x697 = 0;
for(int x698=0; x698 < 100; x698++) {
for(int x699=0; x699 < 10; x699++) {
int32_t x700 = x697;
float x701 = x520[x700];
float x702 = x619[x700];
float x703 = x567[x700];
float x707 = x680[x698];
double x704 = (double)x703;
double x705 = exp(x704);
float x706 = (float)x705;
float x708 = x706 * x707;
float x709 = x702 - x708;
float x710 = x701 + x709;
x520[x700] = x710;
x697 += 1;

}

}
int32_t x717 = 0;
int32_t x718 = 0;
int32_t x719 = 0;
for(int x720=0; x720 < 100; x720++) {
int32_t x721 = x718;
int32_t x722 = x719;
int32_t x723 = x717;
int32_t x724 = x723;
int32_t x725 = x721;
int32_t x726 = x722;
for(int x727=0; x727 < 10; x727++) {
int32_t x728 = x725;
float x729 = x52[x728];
int32_t x730 = x726;
float x731 = x520[x730];
float x732 = x729 + x731;
x52[x728] = x732;
x724 += 1;
x725 += 1;
x726 += 1;

}
x717 += 10;
x719 += 10;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,50,10,1,x520,10,x41,10,1,x517,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,100,1,x501,50,x520,10,1,x50,10);
float* x746 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x747 = 0;
int32_t x748 = 0;
int32_t x749 = 0;
for(int x750=0; x750 < 100; x750++) {
int32_t x751 = x748;
int32_t x752 = x749;
int32_t x753 = x747;
int32_t x754 = x753;
int32_t x755 = x751;
int32_t x756 = x752;
for(int x757=0; x757 < 50; x757++) {
int32_t x758 = x754;
int32_t x759 = x755;
float x760 = x502[x759];
int32_t x761 = x756;
float x762 = x517[x761];
float x763 = x760 * x762;
x746[x758] = x763;
x754 += 1;
x755 += 1;
x756 += 1;

}
x747 += 50;
x748 += 50;
x749 += 50;

}
int32_t x775 = 0;
int32_t x776 = 0;
int32_t x777 = 0;
for(int x778=0; x778 < 100; x778++) {
int32_t x779 = x776;
int32_t x780 = x777;
int32_t x781 = x775;
int32_t x782 = x781;
int32_t x783 = x779;
int32_t x784 = x780;
for(int x785=0; x785 < 50; x785++) {
int32_t x786 = x783;
float x787 = x473[x786];
int32_t x788 = x784;
float x789 = x746[x788];
float x790 = x787 + x789;
x473[x786] = x790;
x782 += 1;
x783 += 1;
x784 += 1;

}
x775 += 50;
x776 += 50;
x777 += 50;

}
int32_t x802 = 0;
int32_t x803 = 0;
int32_t x804 = 0;
for(int x805=0; x805 < 100; x805++) {
int32_t x806 = x803;
int32_t x807 = x804;
int32_t x808 = x802;
int32_t x809 = x808;
int32_t x810 = x806;
int32_t x811 = x807;
for(int x812=0; x812 < 50; x812++) {
int32_t x813 = x810;
float x814 = x40[x813];
int32_t x815 = x811;
float x816 = x473[x815];
float x817 = x814 + x816;
x40[x813] = x817;
x809 += 1;
x810 += 1;
x811 += 1;

}
x802 += 50;
x804 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,320,50,1,x473,50,x29,50,1,x470,320);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,100,1,x377,320,x473,50,1,x38,50);
for(int x831=0; x831 < 32000; x831++) {
int32_t x832 = x383[x831];
float x833 = x376[x832];
float x834 = x470[x831];
float x835 = x833 + x834;
x376[x832] = x835;

}
for(int x839=0; x839 < 128000; x839++) {
float x840 = x301[x839];
bool x841 = x840 < 0.0f;
float x844;
if (x841) {
x844 = 0.0f;
} else {
float x842 = x376[x839];
x844 = x842;
}
x362[x839] = x844;

}
// conv2D back-propagate
float* x849 = (float*)myMalloc(1600000 * sizeof(float));;
for(int x850=0; x850 < 100; x850++) {
int32_t x851 = x850 * 1440;
float* x852 = x300+x851;
int32_t x853 = x850 * 1280;
float* x854 = x362+x853;
int32_t x855 = x850 * 16000;
float* x856 = x849+x855;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 250,64,20,1,x17,250,x854,64,0,x856,64);
for(int x858=0; x858 < 10; x858++) {
int32_t x862 = x858 * 5;
int32_t x863 = x862 * 5;
int32_t x864 = x863 * 8;
int32_t x865 = x864 * 8;
int32_t x874 = x858 * 12;
int32_t x875 = x874 * 12;
for(int x860=0; x860 < 5; x860++) {
int32_t x866 = x860 * 5;
int32_t x867 = x866 * 8;
int32_t x868 = x867 * 8;
int32_t x869 = x865 + x868;
for(int x861=0; x861 < 5; x861++) {
int32_t x870 = x861 * 8;
int32_t x871 = x870 * 8;
int32_t x872 = x869 + x871;
float* x873 = x856+x872;
float* x876 = x852+x875;
for(int x877=0; x877 < 8; x877++) {
int32_t x878 = x877 + x860;
int32_t x879 = x878 * 12;
int32_t x880 = x879 + x861;
float* x881 = x876+x880;
int32_t x882 = x877 * 8;
float* x883 = x873+x882;
for(int x884=0; x884 < 8; x884++) {
float x885 = x881[x884];
float x886 = x883[x884];
float x887 = x885 + x886;
x881[x884] = x887;

}

}

}

}

}

}
for(int x901=0; x901 < 100; x901++) {
int32_t x902 = x901 * 1280;
float* x903 = x362+x902;
int32_t x904 = x901 * 16000;
float* x905 = x318+x904;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,250,64,1.0,x903,64,x905,64,1,x26,250);
for(int x907=0; x907 < 20; x907++) {
float x908 = 0.0f;
int32_t x909 = x907 * 8;
int32_t x910 = x909 * 8;
float* x911 = x903+x910;
for(int x912=0; x912 < 64; x912++) {
float x913 = x911[x912];
x908 += x913;

}
float x917 = x28[x907];
float x918 = x908;
float x919 = 1.0f * x918;
float x920 = x917 + x919;
x28[x907] = x920;

}

}
for(int x926=0; x926 < 144000; x926++) {
int32_t x927 = x212[x926];
float x928 = x205[x927];
float x929 = x300[x926];
float x930 = x928 + x929;
x205[x927] = x930;

}
for(int x934=0; x934 < 576000; x934++) {
float x935 = x128[x934];
bool x936 = x935 < 0.0f;
float x939;
if (x936) {
x939 = 0.0f;
} else {
float x937 = x205[x934];
x939 = x937;
}
x191[x934] = x939;

}
// conv2D back-propagate
float* x944 = (float*)myMalloc(1440000 * sizeof(float));;
for(int x945=0; x945 < 100; x945++) {
int32_t x946 = x945 * 5760;
float* x947 = x191+x946;
int32_t x948 = x945 * 14400;
float* x949 = x146+x948;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 10,25,576,1.0,x947,576,x949,576,1,x14,25);
for(int x951=0; x951 < 10; x951++) {
float x952 = 0.0f;
int32_t x953 = x951 * 24;
int32_t x954 = x953 * 24;
float* x955 = x947+x954;
for(int x956=0; x956 < 576; x956++) {
float x957 = x955[x956];
x952 += x957;

}
float x961 = x16[x951];
float x962 = x952;
float x963 = 1.0f * x962;
float x964 = x961 + x963;
x16[x951] = x964;

}

}
float x970 = x127[0];
x108 += x970;
for(int x972=0; x972 < 20; x972++) {
float x973 = x27[x972];
float x975 = x28[x972];
float x974 = x973 * 1.0f;
float x976 = x975 * -5.0E-4f;
float x977 = x974 + x976;
x27[x972] = x977;

}
for(int x981=0; x981 < 20; x981++) {
x28[x981] = 0.0f;

}
for(int x985=0; x985 < 5000; x985++) {
float x986 = x17[x985];
float x988 = x26[x985];
float x987 = x986 * 1.0f;
float x989 = x988 * -5.0E-4f;
float x990 = x987 + x989;
x17[x985] = x990;

}
for(int x994=0; x994 < 5000; x994++) {
x26[x994] = 0.0f;

}
for(int x998=0; x998 < 16000; x998++) {
float x999 = x29[x998];
float x1001 = x38[x998];
float x1000 = x999 * 1.0f;
float x1002 = x1001 * -5.0E-4f;
float x1003 = x1000 + x1002;
x29[x998] = x1003;

}
for(int x1007=0; x1007 < 16000; x1007++) {
x38[x1007] = 0.0f;

}
for(int x1011=0; x1011 < 50; x1011++) {
float x1012 = x39[x1011];
float x1014 = x40[x1011];
float x1013 = x1012 * 1.0f;
float x1015 = x1014 * -5.0E-4f;
float x1016 = x1013 + x1015;
x39[x1011] = x1016;

}
for(int x1020=0; x1020 < 50; x1020++) {
x40[x1020] = 0.0f;

}
for(int x1024=0; x1024 < 10; x1024++) {
float x1025 = x15[x1024];
float x1027 = x16[x1024];
float x1026 = x1025 * 1.0f;
float x1028 = x1027 * -5.0E-4f;
float x1029 = x1026 + x1028;
x15[x1024] = x1029;

}
for(int x1033=0; x1033 < 10; x1033++) {
x16[x1033] = 0.0f;

}
for(int x1037=0; x1037 < 250; x1037++) {
float x1038 = x5[x1037];
float x1040 = x14[x1037];
float x1039 = x1038 * 1.0f;
float x1041 = x1040 * -5.0E-4f;
float x1042 = x1039 + x1041;
x5[x1037] = x1042;

}
for(int x1046=0; x1046 < 250; x1046++) {
x14[x1046] = 0.0f;

}
for(int x1050=0; x1050 < 10; x1050++) {
float x1051 = x51[x1050];
float x1053 = x52[x1050];
float x1052 = x1051 * 1.0f;
float x1054 = x1053 * -5.0E-4f;
float x1055 = x1052 + x1054;
x51[x1050] = x1055;

}
for(int x1059=0; x1059 < 10; x1059++) {
x52[x1059] = 0.0f;

}
for(int x1063=0; x1063 < 500; x1063++) {
float x1064 = x41[x1063];
float x1066 = x50[x1063];
float x1065 = x1064 * 1.0f;
float x1067 = x1066 * -5.0E-4f;
float x1068 = x1065 + x1067;
x41[x1063] = x1068;

}
for(int x1072=0; x1072 < 500; x1072++) {
x50[x1072] = 0.0f;

}
int32_t x1076 = x105;
int32_t x1078 = x1076 % x1077;
bool x1079 = x1078 == 0;
if (x1079) {
float x1084 = x108;
double x1080 = (double)x1076;
double x1081 = 100.0 * x1080;
double x1083 = x1081 / x1082;
float x1085 = (float)x1076;
float x1086 = x1084 / x1085;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x101,x1076,x64,x1083,x1086);
fflush(stdout);
} else {
}
int64_t x1091 = (long)mallocAddr;
int64_t x1092 = x1091 - x98;
memset((void*)x98, 0, x1092);
mallocAddr = (void*)x98;
x114 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1100 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1101 = x1100 / 1000LL;
int64_t x1103 = x1100 / x1102;
printf("Training completed in %ldms (%ld us/images)\n",x1101,x1103);
float x1105 = x108;
float x1107 = x1105 / x1106;
double x1108 = (double)x1107;
x97[x101] = x1108;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1114 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1119 = (long)fopen(x0, "w");
fprintf((FILE *)x1119, "unit: %s\n", "1 epoch");
for(int x1121=0; x1121 < 4; x1121++) {
double x1122 = x97[x1121];
fprintf((FILE *)x1119, "%lf\n", x1122);

}
float x1115 = (float)x1114;
float x1116 = x1115 / 1000000.0f;
float x1117 = x1116 - x95;
float x1118 = x1117 / 4.0f;
fprintf((FILE *)x1119, "run time: %lf %lf\n", x95, x1118);
fclose((FILE*)x1119);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

