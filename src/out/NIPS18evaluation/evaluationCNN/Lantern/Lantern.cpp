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
float* x6 = (float*)myMalloc(250 * sizeof(float));;
for(int x8=0; x8 < 250; x8++) {
float x9 = (float)rand()/RAND_MAX;
float x10 = x9 - 0.5f;
float x11 = x10 * 0.9797959f;
x6[x8] = x11;

}
float* x15 = (float*)myMalloc(250 * sizeof(float));;
float* x16 = (float*)myMalloc(10 * sizeof(float));;
float* x17 = (float*)myMalloc(10 * sizeof(float));;
float* x18 = (float*)myMalloc(5000 * sizeof(float));;
for(int x20=0; x20 < 5000; x20++) {
float x21 = (float)rand()/RAND_MAX;
float x22 = x21 - 0.5f;
float x23 = x22 * 0.30983868f;
x18[x20] = x23;

}
float* x27 = (float*)myMalloc(5000 * sizeof(float));;
float* x28 = (float*)myMalloc(20 * sizeof(float));;
float* x29 = (float*)myMalloc(20 * sizeof(float));;
float* x30 = (float*)myMalloc(16000 * sizeof(float));;
for(int x32=0; x32 < 16000; x32++) {
float x33 = (float)rand()/RAND_MAX;
float x34 = x33 - 0.5f;
float x35 = x34 * 0.0559017f;
x30[x32] = x35;

}
float* x39 = (float*)myMalloc(16000 * sizeof(float));;
float* x40 = (float*)myMalloc(50 * sizeof(float));;
float* x41 = (float*)myMalloc(50 * sizeof(float));;
float* x42 = (float*)myMalloc(500 * sizeof(float));;
for(int x44=0; x44 < 500; x44++) {
float x45 = (float)rand()/RAND_MAX;
float x46 = x45 - 0.5f;
float x47 = x46 * 0.14142136f;
x42[x44] = x47;

}
float* x51 = (float*)myMalloc(500 * sizeof(float));;
float* x52 = (float*)myMalloc(10 * sizeof(float));;
float* x53 = (float*)myMalloc(10 * sizeof(float));;
int64_t* x54 = (int64_t*)myMalloc(2 * sizeof(int64_t));;
int64_t* x55 = (int64_t*)myMalloc(2 * sizeof(int64_t));;
int32_t x66 = 0;
int32_t x67 = x66;
int32_t x68 = x67;
int32_t x61 = open("../data/bin/mnist_train_target.bin",0);
int64_t x62 = fsize(x61);
int32_t x64 = (int32_t)x62;
int32_t x65 = x64 / 4;
int* x63 = (int*)mmap(0, x62, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x61, 0);
int32_t x56 = open("../data/bin/mnist_train.bin",0);
int64_t x57 = fsize(x56);
float* x58 = (float*)mmap(0, x57, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x56, 0);
for(int x70=0; x70 < x65; x70++) {
int32_t x71 = x68;
int32_t x73 = x63[x70];
float* x72 = x58+x71;
for(int x75=0; x75 < 784; x75++) {
float x76 = x72[x75];
float x77 = x76 - 0.1307f;
float x78 = x77 / 0.3081f;
x72[x75] = x78;

}
x68 += 784;

}
int32_t x85 = x68;
int64_t x59 = x57 / 4LL;
int32_t x60 = (int32_t)x59;
bool x86 = x85 == x60;
if (x86) {
} else {
printf("Data length doesn't match\n\n");
assert(false && "");
}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x94 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x95 = (float)x94;
float x96 = x95 / 1000000.0f;
printf("Data normalized (all prepare time) in %lf sec\n",x96);
double* x98 = (double*)myMalloc(4 * sizeof(double));;
int64_t x99 = (long)mallocAddr;
// training loop starts here
int32_t x116 = x65 / 100;
int32_t x129 = 23 / 1;
int32_t x130 = x129 + 1;
int32_t x134 = 1000 * x130;
int32_t x135 = x134 * x130;
int32_t x131 = x130 * x130;
int32_t x156 = 2500 * x131;
int32_t x132 = 10 * x131;
int32_t x154 = 25 * x131;
int32_t x133 = 100 * x132;
bool x217 = x130 >= 2;
bool x218;
if (x217) {
x218 = x217;
} else {
x218 = false;
}
int32_t x223 = x130 - 2;
int32_t x224 = x223 / 2;
int32_t x225 = x224 + 1;
int32_t x229 = 1000 * x225;
int32_t x230 = x229 * x225;
int32_t x226 = x225 * x225;
int32_t x227 = 10 * x226;
int32_t x228 = 100 * x227;
int32_t x316 = 2 * x130;
int32_t x327 = x225 - 5;
int32_t x328 = x327 / 1;
int32_t x329 = x328 + 1;
int32_t x333 = 2000 * x329;
int32_t x334 = x333 * x329;
int32_t x330 = x329 * x329;
int32_t x354 = 25000 * x330;
int32_t x331 = 20 * x330;
int32_t x352 = 250 * x330;
int32_t x332 = 100 * x331;
bool x414 = x329 >= 2;
bool x415;
if (x414) {
x415 = x414;
} else {
x415 = false;
}
int32_t x420 = x329 - 2;
int32_t x421 = x420 / 2;
int32_t x422 = x421 + 1;
int32_t x426 = 2000 * x422;
int32_t x427 = x426 * x422;
int32_t x423 = x422 * x422;
int32_t x424 = 20 * x423;
int32_t x425 = 100 * x424;
int32_t x513 = 2 * x329;
int32_t x1233 = x65 / 10;
double x1238 = (double)x65;
int64_t x1258 = (int64_t)x65;
float x1262 = (float)x65;
for(int x102=0; x102 < 4; x102++) {
struct timeval begin_1, end_1, diff_1;
int32_t x104 = 0;
int32_t x105 = x104;
int32_t x106 = x105;
float x107 = 0.0f;
float x108 = x107;
float x109 = x108;
int32_t x110 = x102 + 1;
printf("Start training epoch %d\n",x110);
gettimeofday(&begin_1, NULL);
int32_t x113 = 0;
int32_t x114 = x113;
int32_t x115 = x114;
for(int x118=0; x118 < x116; x118++) {
int32_t x119 = x115;
x106 += 100;
float* x124 = (float*)myMalloc(2 * sizeof(float));;
float* x125 = (float*)myMalloc(4 * sizeof(float));;
float* x126 = (float*)myMalloc(4 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
float* x128 = (float*)myMalloc(1 * sizeof(float));;
float* x136 = (float*)myMalloc(x135 * sizeof(float));;
int32_t x137 = 0;
for(int x139=0; x139 < 100; x139++) {
for(int x141=0; x141 < 10; x141++) {
for(int x143=0; x143 < x131; x143++) {
int32_t x144 = x137;
float x145 = x16[x141];
x136[x144] = x145;
x137 += 1;

}

}

}
float* x157 = (float*)myMalloc(x156 * sizeof(float));;
float* x120 = x58+x119;
for(int x158=0; x158 < 100; x158++) {
int32_t x161 = x158 * x132;
float* x162 = x136+x161;
int32_t x163 = x158 * x154;
float* x164 = x157+x163;
int32_t x159 = x158 * 784;
float* x160 = x120+x159;
for(int x166=0; x166 < 25; x166++) {
int32_t x167 = x166 / 25;
int32_t x171 = x167 * 5;
int32_t x172 = x171 * 5;
int32_t x173 = x172 * x130;
int32_t x174 = x173 * x130;
int32_t x168 = x166 % 25;
int32_t x169 = x168 / 5;
int32_t x175 = x169 * 5;
int32_t x176 = x175 * x130;
int32_t x177 = x176 * x130;
int32_t x178 = x174 + x177;
int32_t x170 = x168 % 5;
int32_t x179 = x170 * x130;
int32_t x180 = x179 * x130;
int32_t x181 = x178 + x180;
float* x182 = x164+x181;
int32_t x183 = x167 * 28;
int32_t x184 = x183 * 28;
float* x185 = x160+x184;
for(int x187=0; x187 < x130; x187++) {
int32_t x189 = x187 * x130;
float* x190 = x182+x189;
int32_t x188 = x187 + x169;
int32_t x191 = x188 * 28;
int32_t x192 = x191 + x170;
float* x193 = x185+x192;
memcpy(x190, x193, 4 * x130);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 10,x131,25,1,x6,25,x164,x131,1,x162,x131);

}
float* x202 = (float*)myMalloc(x135 * sizeof(float));;
float* x203 = (float*)myMalloc(x133 * sizeof(float));;
for(int x205=0; x205 < x133; x205++) {
float x206 = x136[x205];
bool x207 = x206 < 0.0f;
if (x207) {
x203[x205] = 0.0f;
} else {
float x210 = x136[x205];
x203[x205] = x210;
}

}
float* x216 = (float*)myMalloc(x135 * sizeof(float));;
if (x218) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(100) x Const(10) x Sym(130) x Sym(130)|(2,2)");
}
float* x231 = (float*)myMalloc(x230 * sizeof(float));;
for(int x233=0; x233 < x230; x233++) {
x231[x233] = -3.4028235E38f;

}
int* x237 = (int32_t*)myMalloc(x228 * sizeof(int32_t));;
for(int x238=0; x238 < 100; x238++) {
int32_t x239 = x238 * x132;
float* x240 = x203+x239;
int32_t x241 = x238 * x227;
float* x242 = x231+x241;
int* x243 = x237+x241;
int32_t x244 = 0;
int32_t x245 = 0;
for(int x246=0; x246 < 10; x246++) {
int32_t x247 = x244;
int32_t x248 = x247;
int32_t x249 = x245;
int32_t x250 = x249;
for(int x252=0; x252 < x225; x252++) {
int32_t x253 = x248;
int32_t x254 = x253;
int32_t x255 = x250;
int32_t x256 = x255;
for(int x257=0; x257 < x225; x257++) {
int32_t x258 = x256;
int32_t x259 = x258;
int32_t x260 = x259;
int32_t x261 = x260;
int32_t x262 = x261;
float x263 = x240[x262];
int32_t x264 = x254;
float x265 = x242[x264];
bool x266 = x263 > x265;
if (x266) {
float x267 = x240[x262];
x242[x264] = x267;
int32_t x269 = x262 + x239;
x243[x264] = x269;
} else {
}
x261 += 1;
int32_t x274 = x261;
float x275 = x240[x274];
float x276 = x242[x264];
bool x277 = x275 > x276;
if (x277) {
float x278 = x240[x274];
x242[x264] = x278;
int32_t x280 = x274 + x239;
x243[x264] = x280;
} else {
}
x261 += 1;
x259 += x130;
int32_t x286 = x259;
int32_t x287 = x286;
int32_t x288 = x287;
float x289 = x240[x288];
float x290 = x242[x264];
bool x291 = x289 > x290;
if (x291) {
float x292 = x240[x288];
x242[x264] = x292;
int32_t x294 = x288 + x239;
x243[x264] = x294;
} else {
}
x287 += 1;
int32_t x299 = x287;
float x300 = x240[x299];
float x301 = x242[x264];
bool x302 = x300 > x301;
if (x302) {
float x303 = x240[x299];
x242[x264] = x303;
int32_t x305 = x299 + x239;
x243[x264] = x305;
} else {
}
x287 += 1;
x259 += x130;
x254 += 1;
x256 += 2;

}
x248 += x225;
x250 += x316;

}
x244 += x226;
x245 += x131;

}

}
float* x326 = (float*)myMalloc(x230 * sizeof(float));;
float* x335 = (float*)myMalloc(x334 * sizeof(float));;
int32_t x336 = 0;
for(int x337=0; x337 < 100; x337++) {
for(int x339=0; x339 < 20; x339++) {
for(int x341=0; x341 < x330; x341++) {
int32_t x342 = x336;
float x343 = x28[x339];
x335[x342] = x343;
x336 += 1;

}

}

}
float* x355 = (float*)myMalloc(x354 * sizeof(float));;
for(int x356=0; x356 < 100; x356++) {
int32_t x357 = x356 * x227;
float* x358 = x231+x357;
int32_t x359 = x356 * x331;
float* x360 = x335+x359;
int32_t x361 = x356 * x352;
float* x362 = x355+x361;
for(int x363=0; x363 < 250; x363++) {
int32_t x364 = x363 / 25;
int32_t x368 = x364 * 5;
int32_t x369 = x368 * 5;
int32_t x370 = x369 * x329;
int32_t x371 = x370 * x329;
int32_t x365 = x363 % 25;
int32_t x366 = x365 / 5;
int32_t x372 = x366 * 5;
int32_t x373 = x372 * x329;
int32_t x374 = x373 * x329;
int32_t x375 = x371 + x374;
int32_t x367 = x365 % 5;
int32_t x376 = x367 * x329;
int32_t x377 = x376 * x329;
int32_t x378 = x375 + x377;
float* x379 = x362+x378;
int32_t x380 = x364 * x225;
int32_t x381 = x380 * x225;
float* x382 = x358+x381;
for(int x384=0; x384 < x329; x384++) {
int32_t x386 = x384 * x329;
float* x387 = x379+x386;
int32_t x385 = x384 + x366;
int32_t x388 = x385 * x225;
int32_t x389 = x388 + x367;
float* x390 = x382+x389;
memcpy(x387, x390, 4 * x329);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,x330,250,1,x18,250,x362,x330,1,x360,x330);

}
float* x399 = (float*)myMalloc(x334 * sizeof(float));;
float* x400 = (float*)myMalloc(x332 * sizeof(float));;
for(int x402=0; x402 < x332; x402++) {
float x403 = x335[x402];
bool x404 = x403 < 0.0f;
if (x404) {
x400[x402] = 0.0f;
} else {
float x407 = x335[x402];
x400[x402] = x407;
}

}
float* x413 = (float*)myMalloc(x334 * sizeof(float));;
if (x415) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(100) x Const(20) x Sym(329) x Sym(329)|(2,2)");
}
float* x428 = (float*)myMalloc(x427 * sizeof(float));;
for(int x430=0; x430 < x427; x430++) {
x428[x430] = -3.4028235E38f;

}
int* x434 = (int32_t*)myMalloc(x425 * sizeof(int32_t));;
for(int x435=0; x435 < 100; x435++) {
int32_t x436 = x435 * x331;
float* x437 = x400+x436;
int32_t x438 = x435 * x424;
float* x439 = x428+x438;
int* x440 = x434+x438;
int32_t x441 = 0;
int32_t x442 = 0;
for(int x443=0; x443 < 20; x443++) {
int32_t x444 = x441;
int32_t x445 = x444;
int32_t x446 = x442;
int32_t x447 = x446;
for(int x449=0; x449 < x422; x449++) {
int32_t x450 = x445;
int32_t x451 = x450;
int32_t x452 = x447;
int32_t x453 = x452;
for(int x454=0; x454 < x422; x454++) {
int32_t x455 = x453;
int32_t x456 = x455;
int32_t x457 = x456;
int32_t x458 = x457;
int32_t x459 = x458;
float x460 = x437[x459];
int32_t x461 = x451;
float x462 = x439[x461];
bool x463 = x460 > x462;
if (x463) {
float x464 = x437[x459];
x439[x461] = x464;
int32_t x466 = x459 + x436;
x440[x461] = x466;
} else {
}
x458 += 1;
int32_t x471 = x458;
float x472 = x437[x471];
float x473 = x439[x461];
bool x474 = x472 > x473;
if (x474) {
float x475 = x437[x471];
x439[x461] = x475;
int32_t x477 = x471 + x436;
x440[x461] = x477;
} else {
}
x458 += 1;
x456 += x329;
int32_t x483 = x456;
int32_t x484 = x483;
int32_t x485 = x484;
float x486 = x437[x485];
float x487 = x439[x461];
bool x488 = x486 > x487;
if (x488) {
float x489 = x437[x485];
x439[x461] = x489;
int32_t x491 = x485 + x436;
x440[x461] = x491;
} else {
}
x484 += 1;
int32_t x496 = x484;
float x497 = x437[x496];
float x498 = x439[x461];
bool x499 = x497 > x498;
if (x499) {
float x500 = x437[x496];
x439[x461] = x500;
int32_t x502 = x496 + x436;
x440[x461] = x502;
} else {
}
x484 += 1;
x456 += x329;
x451 += 1;
x453 += 2;

}
x445 += x422;
x447 += x513;

}
x441 += x423;
x442 += x330;

}

}
float* x523 = (float*)myMalloc(x427 * sizeof(float));;
int32_t x524 = 0;
int32_t x525 = 1;
x524 += 1;
x525 *= 320;
int32_t x528 = x524;
bool x529 = x528 >= 2;
if (x529) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x535 = x528 == 0;
if (x535) {
int32_t x536 = x525;
bool x537 = x536 == x425;
if (x537) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x544 = x525;
int32_t x545 = x425 / x544;
int32_t x547 = x545 * 50;
float* x548 = (float*)myMalloc(x547 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x545,50,320,1,x428,320,x30,50,0,x548,50);
float* x550 = (float*)myMalloc(x547 * sizeof(float));;
int32_t x551 = 0;
int32_t x552 = 0;
int32_t x553 = 0;
bool x576 = x545 > 1;
for(int x555=0; x555 < x545; x555++) {
int32_t x556 = x552;
int32_t x557 = x553;
int32_t x558 = x551;
int32_t x559 = x558;
int32_t x560 = x556;
int32_t x561 = x557;
for(int x563=0; x563 < 50; x563++) {
int32_t x564 = x560;
float x565 = x548[x564];
int32_t x566 = x561;
float x567 = x40[x566];
float x568 = x565 + x567;
x548[x564] = x568;
x559 += 1;
x560 += 1;
x561 += 1;

}
x551 += 50;
if (x576) {
x552 += 50;
} else {
}

}
float* x582 = (float*)myMalloc(x547 * sizeof(float));;
float* x583 = (float*)myMalloc(x547 * sizeof(float));;
for(int x585=0; x585 < x547; x585++) {
float x586 = (float)rand()/RAND_MAX;
bool x587 = x586 > 0.5f;
if (x587) {
float x588 = x548[x585];
float x589 = x588 * 2.0f;
x582[x585] = x589;
x583[x585] = 2.0f;
} else {
x582[x585] = 0.0f;
x583[x585] = 0.0f;
}

}
float* x599 = (float*)myMalloc(x547 * sizeof(float));;
int32_t x600 = x545 * 10;
float* x601 = (float*)myMalloc(x600 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x545,10,50,1,x582,50,x42,10,0,x601,10);
float* x603 = (float*)myMalloc(x600 * sizeof(float));;
int32_t x604 = 0;
int32_t x605 = 0;
int32_t x606 = 0;
for(int x607=0; x607 < x545; x607++) {
int32_t x608 = x605;
int32_t x609 = x606;
int32_t x610 = x604;
int32_t x611 = x610;
int32_t x612 = x608;
int32_t x613 = x609;
for(int x614=0; x614 < 10; x614++) {
int32_t x615 = x612;
float x616 = x601[x615];
int32_t x617 = x613;
float x618 = x52[x617];
float x619 = x616 + x618;
x601[x615] = x619;
x611 += 1;
x612 += 1;
x613 += 1;

}
x604 += 10;
if (x576) {
x605 += 10;
} else {
}

}
float* x632 = (float*)myMalloc(x545 * sizeof(float));;
int32_t x633 = 0;
for(int x634=0; x634 < x545; x634++) {
float x635 = -3.4028235E38f;
for(int x636=0; x636 < 10; x636++) {
int32_t x637 = x633;
float x638 = x601[x637];
float x639 = x635;
bool x640 = x638 > x639;
if (x640) {
float x641 = x601[x637];
x635 = x641;
} else {
}
x633 += 1;

}
float x648 = x635;
x632[x634] = x648;

}
float* x652 = (float*)myMalloc(x600 * sizeof(float));;
int32_t x653 = 0;
for(int x654=0; x654 < x545; x654++) {
for(int x655=0; x655 < 10; x655++) {
int32_t x656 = x653;
float x657 = x601[x656];
float x658 = x632[x654];
float x659 = x657 - x658;
double x660 = (double)x659;
double x661 = exp(x660);
float x662 = (float)x661;
x652[x656] = x662;
x653 += 1;

}

}
float* x669 = (float*)myMalloc(x545 * sizeof(float));;
for(int x670=0; x670 < x545; x670++) {
int32_t x671 = x670;
int32_t x672 = x670 * 10;
int32_t x673 = x672;
for(int x674=0; x674 < 10; x674++) {
for(int x676=0; x676 < 1; x676++) {
int32_t x677 = x671;
int32_t x678 = x677 + x676;
float x679 = x669[x678];
int32_t x680 = x673;
int32_t x681 = x680 + x676;
float x682 = x652[x681];
float x683 = x679 + x682;
x669[x678] = x683;

}
x673 += 1;

}

}
x653 = 0;
for(int x693=0; x693 < x545; x693++) {
float x694 = x632[x693];
float x695 = x669[x693];
double x696 = (double)x695;
double x697 = log(x696);
float x698 = (float)x697;
float x699 = x694 + x698;
for(int x700=0; x700 < 10; x700++) {
int32_t x701 = x653;
float x702 = x601[x701];
float x703 = x702 - x699;
x652[x701] = x703;
x653 += 1;

}

}
float* x710 = (float*)myMalloc(x600 * sizeof(float));;
// nllLoss forward in CPU
float* x712 = (float*)myMalloc(x545 * sizeof(float));;
int32_t x713 = 0;
int32_t x121 = x118 * 100;
int* x122 = x63+x121;
for(int x714=0; x714 < x545; x714++) {
int32_t x715 = x713;
int32_t x716 = x122[x714];
int32_t x717 = x715 + x716;
float x718 = x652[x717];
float x719 = -1.0f * x718;
x712[x714] = x719;
x713 += 10;

}
float* x724 = (float*)myMalloc(x545 * sizeof(float));;
float x725 = 0.0f;
for(int x726=0; x726 < x545; x726++) {
float x727 = x725;
float x728 = x712[x726];
float x729 = x727 + x728;
x725 = x729;

}
float x733 = x725;
float* x734 = (float*)myMalloc(1 * sizeof(float));;
for(int x735=0; x735 < 1; x735++) {
x734[x735] = x733;

}
float* x739 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x741=0; x741 < 1; x741++) {
x739[x741] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@ea87b3a
for(int x746=0; x746 < 1; x746++) {
float x747 = x734[x746];
x128[x746] = x747;

}
// 'sum' gradient.
bool x752 = x545 == 1;
bool x753 = x752 || true;
bool x754 = x753 || x752;
if (x754) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",x545,1);
assert(false && "");
}
int32_t x762 = 0;
int32_t x763 = 0;
int32_t x764 = 0;
bool x760 = x545 <= 1;
int32_t x761;
if (x760) {
x761 = 1;
} else {
x761 = x545;
}
for(int x766=0; x766 < x761; x766++) {
int32_t x767 = x763;
float x768 = x724[x767];
int32_t x769 = x764;
float x770 = x739[x769];
float x771 = x768 + x770;
x724[x767] = x771;
x762 += 1;
if (x576) {
x763 += 1;
} else {
}

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x781 = 0;
for(int x782=0; x782 < x545; x782++) {
int32_t x783 = x781;
int32_t x784 = x122[x782];
int32_t x785 = x783 + x784;
float x786 = x710[x785];
float x787 = x724[x782];
float x788 = -1.0f * x787;
float x789 = x786 + x788;
x710[x785] = x789;
x781 += 10;

}
float* x794 = (float*)myMalloc(x545 * sizeof(float));;
for(int x795=0; x795 < x545; x795++) {
int32_t x796 = x795;
int32_t x797 = x795 * 10;
int32_t x798 = x797;
for(int x799=0; x799 < 10; x799++) {
for(int x800=0; x800 < 1; x800++) {
int32_t x801 = x796;
int32_t x802 = x801 + x800;
float x803 = x794[x802];
int32_t x804 = x798;
int32_t x805 = x804 + x800;
float x806 = x710[x805];
float x807 = x803 + x806;
x794[x802] = x807;

}
x798 += 1;

}

}
int32_t x816 = 0;
for(int x817=0; x817 < x545; x817++) {
for(int x818=0; x818 < 10; x818++) {
int32_t x819 = x816;
float x820 = x603[x819];
float x821 = x710[x819];
float x822 = x652[x819];
float x826 = x794[x817];
double x823 = (double)x822;
double x824 = exp(x823);
float x825 = (float)x824;
float x827 = x825 * x826;
float x828 = x821 - x827;
float x829 = x820 + x828;
x603[x819] = x829;
x816 += 1;

}

}
int32_t x836 = 0;
int32_t x837 = 0;
int32_t x838 = 0;
for(int x839=0; x839 < x545; x839++) {
int32_t x840 = x837;
int32_t x841 = x838;
int32_t x842 = x836;
int32_t x843 = x842;
int32_t x844 = x840;
int32_t x845 = x841;
for(int x846=0; x846 < 10; x846++) {
int32_t x847 = x844;
float x848 = x53[x847];
int32_t x849 = x845;
float x850 = x603[x849];
float x851 = x848 + x850;
x53[x847] = x851;
x843 += 1;
x844 += 1;
x845 += 1;

}
x836 += 10;
if (x576) {
x838 += 10;
} else {
}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x545,50,10,1,x603,10,x42,10,1,x599,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,x545,1,x582,50,x603,10,1,x51,10);
bool x867 = x752 || x752;
bool x868 = x867 || true;
bool x869;
if (x868) {
x869 = true;
} else {
x869 = false;
}
if (x869) {
} else {
printf("dimensions not compatible for broadcasting %d,%d, with %d,%d,\n",x545,50,x545,50);
assert(false && "");
}
bool x875 = x545 > 0;
bool x876;
if (x875) {
x876 = true;
} else {
x876 = false;
}
if (x876) {
} else {
printf("broadcasting dim not match %s %s\n"," x Sym(545) x Const(50)"," x Sym(545) x Const(50)");
assert(false && "");
}
float* x882 = (float*)myMalloc(x547 * sizeof(float));;
int32_t x883 = 0;
int32_t x884 = 0;
int32_t x885 = 0;
for(int x886=0; x886 < x545; x886++) {
int32_t x887 = x884;
int32_t x888 = x885;
int32_t x889 = x883;
int32_t x890 = x889;
int32_t x891 = x887;
int32_t x892 = x888;
for(int x893=0; x893 < 50; x893++) {
int32_t x894 = x890;
int32_t x895 = x891;
float x896 = x583[x895];
int32_t x897 = x892;
float x898 = x599[x897];
float x899 = x896 * x898;
x882[x894] = x899;
x890 += 1;
x891 += 1;
x892 += 1;

}
x883 += 50;
if (x576) {
x884 += 50;
} else {
}
if (x576) {
x885 += 50;
} else {
}

}
if (x869) {
} else {
printf("dimensions not compatible for broadcasting %d,%d, with %d,%d,\n",x545,50,x545,50);
assert(false && "");
}
int32_t x919 = 0;
int32_t x920 = 0;
int32_t x921 = 0;
for(int x922=0; x922 < x545; x922++) {
int32_t x923 = x920;
int32_t x924 = x921;
int32_t x925 = x919;
int32_t x926 = x925;
int32_t x927 = x923;
int32_t x928 = x924;
for(int x929=0; x929 < 50; x929++) {
int32_t x930 = x927;
float x931 = x550[x930];
int32_t x932 = x928;
float x933 = x882[x932];
float x934 = x931 + x933;
x550[x930] = x934;
x926 += 1;
x927 += 1;
x928 += 1;

}
x919 += 50;
if (x576) {
x920 += 50;
} else {
}
if (x576) {
x921 += 50;
} else {
}

}
int32_t x950 = 0;
int32_t x951 = 0;
int32_t x952 = 0;
for(int x953=0; x953 < x545; x953++) {
int32_t x954 = x951;
int32_t x955 = x952;
int32_t x956 = x950;
int32_t x957 = x956;
int32_t x958 = x954;
int32_t x959 = x955;
for(int x960=0; x960 < 50; x960++) {
int32_t x961 = x958;
float x962 = x41[x961];
int32_t x963 = x959;
float x964 = x550[x963];
float x965 = x962 + x964;
x41[x961] = x965;
x957 += 1;
x958 += 1;
x959 += 1;

}
x950 += 50;
if (x576) {
x952 += 50;
} else {
}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x545,320,50,1,x550,50,x30,50,1,x523,320);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,x545,1,x428,320,x550,50,1,x39,50);
for(int x982=0; x982 < x425; x982++) {
int32_t x983 = x434[x982];
float x984 = x413[x983];
float x985 = x523[x982];
float x986 = x984 + x985;
x413[x983] = x986;

}
for(int x990=0; x990 < x332; x990++) {
float x991 = x399[x990];
float x992 = x335[x990];
bool x993 = x992 < 0.0f;
float x996;
if (x993) {
x996 = 0.0f;
} else {
float x994 = x413[x990];
x996 = x994;
}
float x997 = x991 + x996;
x399[x990] = x997;

}
// conv2D back-propagate
float* x1002 = (float*)myMalloc(x354 * sizeof(float));;
for(int x1003=0; x1003 < 100; x1003++) {
int32_t x1004 = x1003 * x227;
float* x1005 = x326+x1004;
int32_t x1006 = x1003 * x331;
float* x1007 = x399+x1006;
int32_t x1008 = x1003 * x352;
float* x1009 = x1002+x1008;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 250,x330,20,1,x18,250,x1007,x330,0,x1009,x330);
for(int x1011=0; x1011 < 10; x1011++) {
int32_t x1015 = x1011 * 5;
int32_t x1016 = x1015 * 5;
int32_t x1017 = x1016 * x329;
int32_t x1018 = x1017 * x329;
int32_t x1027 = x1011 * x225;
int32_t x1028 = x1027 * x225;
for(int x1013=0; x1013 < 5; x1013++) {
int32_t x1019 = x1013 * 5;
int32_t x1020 = x1019 * x329;
int32_t x1021 = x1020 * x329;
int32_t x1022 = x1018 + x1021;
for(int x1014=0; x1014 < 5; x1014++) {
int32_t x1023 = x1014 * x329;
int32_t x1024 = x1023 * x329;
int32_t x1025 = x1022 + x1024;
float* x1026 = x1009+x1025;
float* x1029 = x1005+x1028;
for(int x1030=0; x1030 < x329; x1030++) {
int32_t x1031 = x1030 + x1013;
int32_t x1032 = x1031 * x225;
int32_t x1033 = x1032 + x1014;
float* x1034 = x1029+x1033;
int32_t x1035 = x1030 * x329;
float* x1036 = x1026+x1035;
for(int x1037=0; x1037 < x329; x1037++) {
float x1038 = x1034[x1037];
float x1039 = x1036[x1037];
float x1040 = x1038 + x1039;
x1034[x1037] = x1040;

}

}

}

}

}

}
for(int x1054=0; x1054 < 100; x1054++) {
int32_t x1055 = x1054 * x331;
float* x1056 = x399+x1055;
int32_t x1057 = x1054 * x352;
float* x1058 = x355+x1057;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,250,x330,1.0,x1056,x330,x1058,x330,1,x27,250);
for(int x1060=0; x1060 < 20; x1060++) {
float x1061 = 0.0f;
int32_t x1062 = x1060 * x329;
int32_t x1063 = x1062 * x329;
float* x1064 = x1056+x1063;
for(int x1065=0; x1065 < x330; x1065++) {
float x1066 = x1064[x1065];
x1061 += x1066;

}
float x1070 = x29[x1060];
float x1071 = x1061;
float x1072 = 1.0f * x1071;
float x1073 = x1070 + x1072;
x29[x1060] = x1073;

}

}
for(int x1080=0; x1080 < x228; x1080++) {
int32_t x1081 = x237[x1080];
float x1082 = x216[x1081];
float x1083 = x326[x1080];
float x1084 = x1082 + x1083;
x216[x1081] = x1084;

}
for(int x1088=0; x1088 < x133; x1088++) {
float x1089 = x202[x1088];
float x1090 = x136[x1088];
bool x1091 = x1090 < 0.0f;
float x1094;
if (x1091) {
x1094 = 0.0f;
} else {
float x1092 = x216[x1088];
x1094 = x1092;
}
float x1095 = x1089 + x1094;
x202[x1088] = x1095;

}
// conv2D back-propagate
float* x1100 = (float*)myMalloc(x156 * sizeof(float));;
for(int x1101=0; x1101 < 100; x1101++) {
int32_t x1102 = x1101 * x132;
float* x1103 = x202+x1102;
int32_t x1104 = x1101 * x154;
float* x1105 = x157+x1104;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 10,25,x131,1.0,x1103,x131,x1105,x131,1,x15,25);
for(int x1107=0; x1107 < 10; x1107++) {
float x1108 = 0.0f;
int32_t x1109 = x1107 * x130;
int32_t x1110 = x1109 * x130;
float* x1111 = x1103+x1110;
for(int x1112=0; x1112 < x131; x1112++) {
float x1113 = x1111[x1112];
x1108 += x1113;

}
float x1117 = x17[x1107];
float x1118 = x1108;
float x1119 = 1.0f * x1118;
float x1120 = x1117 + x1119;
x17[x1107] = x1120;

}

}
float x1126 = x128[0];
x109 += x1126;
for(int x1128=0; x1128 < 20; x1128++) {
float x1129 = x28[x1128];
float x1131 = x29[x1128];
float x1130 = x1129 * 1.0f;
float x1132 = x1131 * -5.0E-4f;
float x1133 = x1130 + x1132;
x28[x1128] = x1133;

}
for(int x1137=0; x1137 < 20; x1137++) {
x29[x1137] = 0.0f;

}
for(int x1141=0; x1141 < 5000; x1141++) {
float x1142 = x18[x1141];
float x1144 = x27[x1141];
float x1143 = x1142 * 1.0f;
float x1145 = x1144 * -5.0E-4f;
float x1146 = x1143 + x1145;
x18[x1141] = x1146;

}
for(int x1150=0; x1150 < 5000; x1150++) {
x27[x1150] = 0.0f;

}
for(int x1154=0; x1154 < 16000; x1154++) {
float x1155 = x30[x1154];
float x1157 = x39[x1154];
float x1156 = x1155 * 1.0f;
float x1158 = x1157 * -5.0E-4f;
float x1159 = x1156 + x1158;
x30[x1154] = x1159;

}
for(int x1163=0; x1163 < 16000; x1163++) {
x39[x1163] = 0.0f;

}
for(int x1167=0; x1167 < 50; x1167++) {
float x1168 = x40[x1167];
float x1170 = x41[x1167];
float x1169 = x1168 * 1.0f;
float x1171 = x1170 * -5.0E-4f;
float x1172 = x1169 + x1171;
x40[x1167] = x1172;

}
for(int x1176=0; x1176 < 50; x1176++) {
x41[x1176] = 0.0f;

}
for(int x1180=0; x1180 < 10; x1180++) {
float x1181 = x16[x1180];
float x1183 = x17[x1180];
float x1182 = x1181 * 1.0f;
float x1184 = x1183 * -5.0E-4f;
float x1185 = x1182 + x1184;
x16[x1180] = x1185;

}
for(int x1189=0; x1189 < 10; x1189++) {
x17[x1189] = 0.0f;

}
for(int x1193=0; x1193 < 250; x1193++) {
float x1194 = x6[x1193];
float x1196 = x15[x1193];
float x1195 = x1194 * 1.0f;
float x1197 = x1196 * -5.0E-4f;
float x1198 = x1195 + x1197;
x6[x1193] = x1198;

}
for(int x1202=0; x1202 < 250; x1202++) {
x15[x1202] = 0.0f;

}
for(int x1206=0; x1206 < 10; x1206++) {
float x1207 = x52[x1206];
float x1209 = x53[x1206];
float x1208 = x1207 * 1.0f;
float x1210 = x1209 * -5.0E-4f;
float x1211 = x1208 + x1210;
x52[x1206] = x1211;

}
for(int x1215=0; x1215 < 10; x1215++) {
x53[x1215] = 0.0f;

}
for(int x1219=0; x1219 < 500; x1219++) {
float x1220 = x42[x1219];
float x1222 = x51[x1219];
float x1221 = x1220 * 1.0f;
float x1223 = x1222 * -5.0E-4f;
float x1224 = x1221 + x1223;
x42[x1219] = x1224;

}
for(int x1228=0; x1228 < 500; x1228++) {
x51[x1228] = 0.0f;

}
int32_t x1232 = x106;
int32_t x1234 = x1232 % x1233;
bool x1235 = x1234 == 0;
if (x1235) {
float x1240 = x109;
double x1236 = (double)x1232;
double x1237 = 100.0 * x1236;
double x1239 = x1237 / x1238;
float x1241 = (float)x1232;
float x1242 = x1240 / x1241;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x102,x1232,x65,x1239,x1242);
fflush(stdout);
} else {
}
int64_t x1247 = (long)mallocAddr;
int64_t x1248 = x1247 - x99;
memset((void*)x99, 0, x1248);
mallocAddr = (void*)x99;
x115 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1256 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1257 = x1256 / 1000LL;
int64_t x1259 = x1256 / x1258;
printf("Training completed in %ldms (%ld us/images)\n",x1257,x1259);
float x1261 = x109;
float x1263 = x1261 / x1262;
double x1264 = (double)x1263;
x98[x102] = x1264;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1270 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1275 = (long)fopen(x0, "w");
fprintf((FILE *)x1275, "unit: %s\n", "1 epoch");
for(int x1277=0; x1277 < 4; x1277++) {
double x1278 = x98[x1277];
fprintf((FILE *)x1275, "%lf\n", x1278);

}
float x1271 = (float)x1270;
float x1272 = x1271 / 1000000.0f;
float x1273 = x1272 - x96;
float x1274 = x1273 / 4.0f;
fprintf((FILE *)x1275, "run time: %lf %lf\n", x96, x1274);
fclose((FILE*)x1275);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

