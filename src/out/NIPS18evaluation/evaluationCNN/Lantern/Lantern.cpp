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
int32_t x1119 = x65 / 10;
double x1124 = (double)x65;
int64_t x1144 = (int64_t)x65;
float x1148 = (float)x65;
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
bool x551 = x545 == 1;
int32_t x552;
if (x551) {
x552 = 0;
} else {
x552 = 50;
}
for(int x554=0; x554 < x545; x554++) {
int32_t x557 = x552 * x554;
for(int x556=0; x556 < 50; x556++) {
int32_t x558 = x557 + x556;
float x559 = x548[x558];
float x560 = x40[x556];
float x561 = x559 + x560;
x548[x558] = x561;

}

}
float* x567 = (float*)myMalloc(x547 * sizeof(float));;
float* x568 = (float*)myMalloc(x547 * sizeof(float));;
for(int x570=0; x570 < x547; x570++) {
float x571 = (float)rand()/RAND_MAX;
bool x572 = x571 > 0.5f;
if (x572) {
float x573 = x548[x570];
float x574 = x573 * 2.0f;
x567[x570] = x574;
x568[x570] = 2.0f;
} else {
x567[x570] = 0.0f;
x568[x570] = 0.0f;
}

}
float* x584 = (float*)myMalloc(x547 * sizeof(float));;
int32_t x585 = x545 * 10;
float* x586 = (float*)myMalloc(x585 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x545,10,50,1,x567,50,x42,10,0,x586,10);
float* x588 = (float*)myMalloc(x585 * sizeof(float));;
int32_t x589;
if (x551) {
x589 = 0;
} else {
x589 = 10;
}
for(int x590=0; x590 < x545; x590++) {
int32_t x592 = x589 * x590;
for(int x591=0; x591 < 10; x591++) {
int32_t x593 = x592 + x591;
float x594 = x586[x593];
float x595 = x52[x591];
float x596 = x594 + x595;
x586[x593] = x596;

}

}
float* x602 = (float*)myMalloc(x545 * sizeof(float));;
int32_t x603 = 0;
for(int x604=0; x604 < x545; x604++) {
float x605 = -3.4028235E38f;
for(int x606=0; x606 < 10; x606++) {
int32_t x607 = x603;
float x608 = x586[x607];
float x609 = x605;
bool x610 = x608 > x609;
if (x610) {
float x611 = x586[x607];
x605 = x611;
} else {
}
x603 += 1;

}
float x618 = x605;
x602[x604] = x618;

}
float* x622 = (float*)myMalloc(x585 * sizeof(float));;
int32_t x623 = 0;
for(int x624=0; x624 < x545; x624++) {
for(int x625=0; x625 < 10; x625++) {
int32_t x626 = x623;
float x627 = x586[x626];
float x628 = x602[x624];
float x629 = x627 - x628;
double x630 = (double)x629;
double x631 = exp(x630);
float x632 = (float)x631;
x622[x626] = x632;
x623 += 1;

}

}
float* x639 = (float*)myMalloc(x545 * sizeof(float));;
for(int x640=0; x640 < x545; x640++) {
int32_t x641 = x640;
int32_t x642 = x640 * 10;
int32_t x643 = x642;
for(int x644=0; x644 < 10; x644++) {
for(int x646=0; x646 < 1; x646++) {
int32_t x647 = x641;
int32_t x648 = x647 + x646;
float x649 = x639[x648];
int32_t x650 = x643;
int32_t x651 = x650 + x646;
float x652 = x622[x651];
float x653 = x649 + x652;
x639[x648] = x653;

}
x643 += 1;

}

}
x623 = 0;
for(int x663=0; x663 < x545; x663++) {
float x664 = x602[x663];
float x665 = x639[x663];
double x666 = (double)x665;
double x667 = log(x666);
float x668 = (float)x667;
float x669 = x664 + x668;
for(int x670=0; x670 < 10; x670++) {
int32_t x671 = x623;
float x672 = x586[x671];
float x673 = x672 - x669;
x622[x671] = x673;
x623 += 1;

}

}
float* x680 = (float*)myMalloc(x585 * sizeof(float));;
// nllLoss forward in CPU
float* x682 = (float*)myMalloc(x545 * sizeof(float));;
int32_t x683 = 0;
int32_t x121 = x118 * 100;
int* x122 = x63+x121;
for(int x684=0; x684 < x545; x684++) {
int32_t x685 = x683;
int32_t x686 = x122[x684];
int32_t x687 = x685 + x686;
float x688 = x622[x687];
float x689 = -1.0f * x688;
x682[x684] = x689;
x683 += 10;

}
float* x694 = (float*)myMalloc(x545 * sizeof(float));;
float x695 = 0.0f;
for(int x696=0; x696 < x545; x696++) {
float x697 = x695;
float x698 = x682[x696];
float x699 = x697 + x698;
x695 = x699;

}
float x703 = x695;
float* x704 = (float*)myMalloc(1 * sizeof(float));;
for(int x705=0; x705 < 1; x705++) {
x704[x705] = x703;

}
float* x709 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x711=0; x711 < 1; x711++) {
x709[x711] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@62200910
for(int x716=0; x716 < 1; x716++) {
float x717 = x704[x716];
x128[x716] = x717;

}
// 'sum' gradient.
bool x722 = x551 || true;
bool x723 = x722 || x551;
if (x723) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",x545,1);
assert(false && "");
}
bool x729 = x545 <= 1;
int32_t x730;
if (x729) {
x730 = 1;
} else {
x730 = x545;
}
int32_t x731;
if (x551) {
x731 = 0;
} else {
x731 = 1;
}
for(int x733=0; x733 < x730; x733++) {
int32_t x734 = x731 * x733;
float x735 = x694[x734];
float x736 = x709[0];
float x737 = x735 + x736;
x694[x734] = x737;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x743 = 0;
for(int x744=0; x744 < x545; x744++) {
int32_t x745 = x743;
int32_t x746 = x122[x744];
int32_t x747 = x745 + x746;
float x748 = x680[x747];
float x749 = x694[x744];
float x750 = -1.0f * x749;
float x751 = x748 + x750;
x680[x747] = x751;
x743 += 10;

}
float* x756 = (float*)myMalloc(x545 * sizeof(float));;
for(int x757=0; x757 < x545; x757++) {
int32_t x758 = x757;
int32_t x759 = x757 * 10;
int32_t x760 = x759;
for(int x761=0; x761 < 10; x761++) {
for(int x762=0; x762 < 1; x762++) {
int32_t x763 = x758;
int32_t x764 = x763 + x762;
float x765 = x756[x764];
int32_t x766 = x760;
int32_t x767 = x766 + x762;
float x768 = x680[x767];
float x769 = x765 + x768;
x756[x764] = x769;

}
x760 += 1;

}

}
int32_t x778 = 0;
for(int x779=0; x779 < x545; x779++) {
for(int x780=0; x780 < 10; x780++) {
int32_t x781 = x778;
float x782 = x588[x781];
float x783 = x680[x781];
float x784 = x622[x781];
float x788 = x756[x779];
double x785 = (double)x784;
double x786 = exp(x785);
float x787 = (float)x786;
float x789 = x787 * x788;
float x790 = x783 - x789;
float x791 = x782 + x790;
x588[x781] = x791;
x778 += 1;

}

}
for(int x798=0; x798 < x545; x798++) {
int32_t x800 = x589 * x798;
for(int x799=0; x799 < 10; x799++) {
float x802 = x53[x799];
int32_t x801 = x800 + x799;
float x803 = x588[x801];
float x804 = x802 + x803;
x53[x799] = x804;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x545,50,10,1,x588,10,x42,10,1,x584,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,x545,1,x567,50,x588,10,1,x51,10);
bool x813 = x551 || x551;
bool x814 = x813 || true;
bool x815;
if (x814) {
x815 = true;
} else {
x815 = false;
}
if (x815) {
} else {
printf("dimensions not compatible for broadcasting %d,%d, with %d,%d,\n",x545,50,x545,50);
assert(false && "");
}
float* x821 = (float*)myMalloc(x547 * sizeof(float));;
for(int x822=0; x822 < x545; x822++) {
int32_t x826 = x552 * x822;
int32_t x824 = 50 * x822;
for(int x823=0; x823 < 50; x823++) {
int32_t x827 = x826 + x823;
float x828 = x568[x827];
float x829 = x584[x827];
int32_t x825 = x823 + x824;
float x830 = x828 * x829;
x821[x825] = x830;

}

}
if (x815) {
} else {
printf("dimensions not compatible for broadcasting %d,%d, with %d,%d,\n",x545,50,x545,50);
assert(false && "");
}
for(int x840=0; x840 < x545; x840++) {
int32_t x842 = x552 * x840;
for(int x841=0; x841 < 50; x841++) {
int32_t x843 = x842 + x841;
float x844 = x550[x843];
float x845 = x821[x843];
float x846 = x844 + x845;
x550[x843] = x846;

}

}
for(int x852=0; x852 < x545; x852++) {
int32_t x854 = x552 * x852;
for(int x853=0; x853 < 50; x853++) {
float x856 = x41[x853];
int32_t x855 = x854 + x853;
float x857 = x550[x855];
float x858 = x856 + x857;
x41[x853] = x858;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x545,320,50,1,x550,50,x30,50,1,x523,320);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,x545,1,x428,320,x550,50,1,x39,50);
for(int x868=0; x868 < x425; x868++) {
int32_t x869 = x434[x868];
float x870 = x413[x869];
float x871 = x523[x868];
float x872 = x870 + x871;
x413[x869] = x872;

}
for(int x876=0; x876 < x332; x876++) {
float x877 = x399[x876];
float x878 = x335[x876];
bool x879 = x878 < 0.0f;
float x882;
if (x879) {
x882 = 0.0f;
} else {
float x880 = x413[x876];
x882 = x880;
}
float x883 = x877 + x882;
x399[x876] = x883;

}
// conv2D back-propagate
float* x888 = (float*)myMalloc(x354 * sizeof(float));;
for(int x889=0; x889 < 100; x889++) {
int32_t x890 = x889 * x227;
float* x891 = x326+x890;
int32_t x892 = x889 * x331;
float* x893 = x399+x892;
int32_t x894 = x889 * x352;
float* x895 = x888+x894;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 250,x330,20,1,x18,250,x893,x330,0,x895,x330);
for(int x897=0; x897 < 10; x897++) {
int32_t x901 = x897 * 5;
int32_t x902 = x901 * 5;
int32_t x903 = x902 * x329;
int32_t x904 = x903 * x329;
int32_t x913 = x897 * x225;
int32_t x914 = x913 * x225;
for(int x899=0; x899 < 5; x899++) {
int32_t x905 = x899 * 5;
int32_t x906 = x905 * x329;
int32_t x907 = x906 * x329;
int32_t x908 = x904 + x907;
for(int x900=0; x900 < 5; x900++) {
int32_t x909 = x900 * x329;
int32_t x910 = x909 * x329;
int32_t x911 = x908 + x910;
float* x912 = x895+x911;
float* x915 = x891+x914;
for(int x916=0; x916 < x329; x916++) {
int32_t x917 = x916 + x899;
int32_t x918 = x917 * x225;
int32_t x919 = x918 + x900;
float* x920 = x915+x919;
int32_t x921 = x916 * x329;
float* x922 = x912+x921;
for(int x923=0; x923 < x329; x923++) {
float x924 = x920[x923];
float x925 = x922[x923];
float x926 = x924 + x925;
x920[x923] = x926;

}

}

}

}

}

}
for(int x940=0; x940 < 100; x940++) {
int32_t x941 = x940 * x331;
float* x942 = x399+x941;
int32_t x943 = x940 * x352;
float* x944 = x355+x943;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,250,x330,1.0,x942,x330,x944,x330,1,x27,250);
for(int x946=0; x946 < 20; x946++) {
float x947 = 0.0f;
int32_t x948 = x946 * x329;
int32_t x949 = x948 * x329;
float* x950 = x942+x949;
for(int x951=0; x951 < x330; x951++) {
float x952 = x950[x951];
x947 += x952;

}
float x956 = x29[x946];
float x957 = x947;
float x958 = 1.0f * x957;
float x959 = x956 + x958;
x29[x946] = x959;

}

}
for(int x966=0; x966 < x228; x966++) {
int32_t x967 = x237[x966];
float x968 = x216[x967];
float x969 = x326[x966];
float x970 = x968 + x969;
x216[x967] = x970;

}
for(int x974=0; x974 < x133; x974++) {
float x975 = x202[x974];
float x976 = x136[x974];
bool x977 = x976 < 0.0f;
float x980;
if (x977) {
x980 = 0.0f;
} else {
float x978 = x216[x974];
x980 = x978;
}
float x981 = x975 + x980;
x202[x974] = x981;

}
// conv2D back-propagate
float* x986 = (float*)myMalloc(x156 * sizeof(float));;
for(int x987=0; x987 < 100; x987++) {
int32_t x988 = x987 * x132;
float* x989 = x202+x988;
int32_t x990 = x987 * x154;
float* x991 = x157+x990;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 10,25,x131,1.0,x989,x131,x991,x131,1,x15,25);
for(int x993=0; x993 < 10; x993++) {
float x994 = 0.0f;
int32_t x995 = x993 * x130;
int32_t x996 = x995 * x130;
float* x997 = x989+x996;
for(int x998=0; x998 < x131; x998++) {
float x999 = x997[x998];
x994 += x999;

}
float x1003 = x17[x993];
float x1004 = x994;
float x1005 = 1.0f * x1004;
float x1006 = x1003 + x1005;
x17[x993] = x1006;

}

}
float x1012 = x128[0];
x109 += x1012;
for(int x1014=0; x1014 < 20; x1014++) {
float x1015 = x28[x1014];
float x1017 = x29[x1014];
float x1016 = x1015 * 1.0f;
float x1018 = x1017 * -5.0E-4f;
float x1019 = x1016 + x1018;
x28[x1014] = x1019;

}
for(int x1023=0; x1023 < 20; x1023++) {
x29[x1023] = 0.0f;

}
for(int x1027=0; x1027 < 5000; x1027++) {
float x1028 = x18[x1027];
float x1030 = x27[x1027];
float x1029 = x1028 * 1.0f;
float x1031 = x1030 * -5.0E-4f;
float x1032 = x1029 + x1031;
x18[x1027] = x1032;

}
for(int x1036=0; x1036 < 5000; x1036++) {
x27[x1036] = 0.0f;

}
for(int x1040=0; x1040 < 16000; x1040++) {
float x1041 = x30[x1040];
float x1043 = x39[x1040];
float x1042 = x1041 * 1.0f;
float x1044 = x1043 * -5.0E-4f;
float x1045 = x1042 + x1044;
x30[x1040] = x1045;

}
for(int x1049=0; x1049 < 16000; x1049++) {
x39[x1049] = 0.0f;

}
for(int x1053=0; x1053 < 50; x1053++) {
float x1054 = x40[x1053];
float x1056 = x41[x1053];
float x1055 = x1054 * 1.0f;
float x1057 = x1056 * -5.0E-4f;
float x1058 = x1055 + x1057;
x40[x1053] = x1058;

}
for(int x1062=0; x1062 < 50; x1062++) {
x41[x1062] = 0.0f;

}
for(int x1066=0; x1066 < 10; x1066++) {
float x1067 = x16[x1066];
float x1069 = x17[x1066];
float x1068 = x1067 * 1.0f;
float x1070 = x1069 * -5.0E-4f;
float x1071 = x1068 + x1070;
x16[x1066] = x1071;

}
for(int x1075=0; x1075 < 10; x1075++) {
x17[x1075] = 0.0f;

}
for(int x1079=0; x1079 < 250; x1079++) {
float x1080 = x6[x1079];
float x1082 = x15[x1079];
float x1081 = x1080 * 1.0f;
float x1083 = x1082 * -5.0E-4f;
float x1084 = x1081 + x1083;
x6[x1079] = x1084;

}
for(int x1088=0; x1088 < 250; x1088++) {
x15[x1088] = 0.0f;

}
for(int x1092=0; x1092 < 10; x1092++) {
float x1093 = x52[x1092];
float x1095 = x53[x1092];
float x1094 = x1093 * 1.0f;
float x1096 = x1095 * -5.0E-4f;
float x1097 = x1094 + x1096;
x52[x1092] = x1097;

}
for(int x1101=0; x1101 < 10; x1101++) {
x53[x1101] = 0.0f;

}
for(int x1105=0; x1105 < 500; x1105++) {
float x1106 = x42[x1105];
float x1108 = x51[x1105];
float x1107 = x1106 * 1.0f;
float x1109 = x1108 * -5.0E-4f;
float x1110 = x1107 + x1109;
x42[x1105] = x1110;

}
for(int x1114=0; x1114 < 500; x1114++) {
x51[x1114] = 0.0f;

}
int32_t x1118 = x106;
int32_t x1120 = x1118 % x1119;
bool x1121 = x1120 == 0;
if (x1121) {
float x1126 = x109;
double x1122 = (double)x1118;
double x1123 = 100.0 * x1122;
double x1125 = x1123 / x1124;
float x1127 = (float)x1118;
float x1128 = x1126 / x1127;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x102,x1118,x65,x1125,x1128);
fflush(stdout);
} else {
}
int64_t x1133 = (long)mallocAddr;
int64_t x1134 = x1133 - x99;
memset((void*)x99, 0, x1134);
mallocAddr = (void*)x99;
x115 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1142 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1143 = x1142 / 1000LL;
int64_t x1145 = x1142 / x1144;
printf("Training completed in %ldms (%ld us/images)\n",x1143,x1145);
float x1147 = x109;
float x1149 = x1147 / x1148;
double x1150 = (double)x1149;
x98[x102] = x1150;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1156 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1161 = (long)fopen(x0, "w");
fprintf((FILE *)x1161, "unit: %s\n", "1 epoch");
for(int x1163=0; x1163 < 4; x1163++) {
double x1164 = x98[x1163];
fprintf((FILE *)x1161, "%lf\n", x1164);

}
float x1157 = (float)x1156;
float x1158 = x1157 / 1000000.0f;
float x1159 = x1158 - x96;
float x1160 = x1159 / 4.0f;
fprintf((FILE *)x1161, "run time: %lf %lf\n", x96, x1160);
fclose((FILE*)x1161);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

