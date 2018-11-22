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
int32_t x1126 = x65 / 10;
double x1131 = (double)x65;
int64_t x1151 = (int64_t)x65;
float x1155 = (float)x65;
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
bool x554 = x545 == 1;
int32_t x555;
if (x554) {
x555 = 0;
} else {
x555 = 50;
}
for(int x557=0; x557 < x545; x557++) {
int32_t x560 = x555 * x557;
for(int x559=0; x559 < 50; x559++) {
int32_t x561 = x560 + x559;
float x562 = x548[x561];
float x563 = x40[x559];
float x564 = x562 + x563;
x548[x561] = x564;

}

}
float* x570 = (float*)myMalloc(x547 * sizeof(float));;
float* x571 = (float*)myMalloc(x547 * sizeof(float));;
for(int x573=0; x573 < x547; x573++) {
float x574 = (float)rand()/RAND_MAX;
bool x575 = x574 > 0.5f;
if (x575) {
float x576 = x548[x573];
float x577 = x576 * 2.0f;
x570[x573] = x577;
x571[x573] = 2.0f;
} else {
x570[x573] = 0.0f;
x571[x573] = 0.0f;
}

}
float* x587 = (float*)myMalloc(x547 * sizeof(float));;
int32_t x588 = x545 * 10;
float* x589 = (float*)myMalloc(x588 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x545,10,50,1,x570,50,x42,10,0,x589,10);
float* x591 = (float*)myMalloc(x588 * sizeof(float));;
int32_t x592;
if (x554) {
x592 = 0;
} else {
x592 = 10;
}
for(int x593=0; x593 < x545; x593++) {
int32_t x595 = x592 * x593;
for(int x594=0; x594 < 10; x594++) {
int32_t x596 = x595 + x594;
float x597 = x589[x596];
float x598 = x52[x594];
float x599 = x597 + x598;
x589[x596] = x599;

}

}
float* x605 = (float*)myMalloc(x545 * sizeof(float));;
int32_t x606 = 0;
for(int x607=0; x607 < x545; x607++) {
float x608 = -3.4028235E38f;
for(int x609=0; x609 < 10; x609++) {
int32_t x610 = x606;
float x611 = x589[x610];
float x612 = x608;
bool x613 = x611 > x612;
if (x613) {
float x614 = x589[x610];
x608 = x614;
} else {
}
x606 += 1;

}
float x621 = x608;
x605[x607] = x621;

}
float* x625 = (float*)myMalloc(x588 * sizeof(float));;
int32_t x626 = 0;
for(int x627=0; x627 < x545; x627++) {
for(int x628=0; x628 < 10; x628++) {
int32_t x629 = x626;
float x630 = x589[x629];
float x631 = x605[x627];
float x632 = x630 - x631;
double x633 = (double)x632;
double x634 = exp(x633);
float x635 = (float)x634;
x625[x629] = x635;
x626 += 1;

}

}
float* x642 = (float*)myMalloc(x545 * sizeof(float));;
for(int x643=0; x643 < x545; x643++) {
int32_t x644 = x643;
int32_t x645 = x643 * 10;
int32_t x646 = x645;
for(int x647=0; x647 < 10; x647++) {
for(int x649=0; x649 < 1; x649++) {
int32_t x650 = x644;
int32_t x651 = x650 + x649;
float x652 = x642[x651];
int32_t x653 = x646;
int32_t x654 = x653 + x649;
float x655 = x625[x654];
float x656 = x652 + x655;
x642[x651] = x656;

}
x646 += 1;

}

}
x626 = 0;
for(int x666=0; x666 < x545; x666++) {
float x667 = x605[x666];
float x668 = x642[x666];
double x669 = (double)x668;
double x670 = log(x669);
float x671 = (float)x670;
float x672 = x667 + x671;
for(int x673=0; x673 < 10; x673++) {
int32_t x674 = x626;
float x675 = x589[x674];
float x676 = x675 - x672;
x625[x674] = x676;
x626 += 1;

}

}
float* x683 = (float*)myMalloc(x588 * sizeof(float));;
// nllLoss forward in CPU
float* x685 = (float*)myMalloc(x545 * sizeof(float));;
int32_t x686 = 0;
int32_t x121 = x118 * 100;
int* x122 = x63+x121;
for(int x687=0; x687 < x545; x687++) {
int32_t x688 = x686;
int32_t x689 = x122[x687];
int32_t x690 = x688 + x689;
float x691 = x625[x690];
float x692 = -1.0f * x691;
x685[x687] = x692;
x686 += 10;

}
float* x697 = (float*)myMalloc(x545 * sizeof(float));;
float x698 = 0.0f;
for(int x699=0; x699 < x545; x699++) {
float x700 = x698;
float x701 = x685[x699];
float x702 = x700 + x701;
x698 = x702;

}
float x706 = x698;
float* x707 = (float*)myMalloc(1 * sizeof(float));;
for(int x708=0; x708 < 1; x708++) {
x707[x708] = x706;

}
float* x712 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x714=0; x714 < 1; x714++) {
x712[x714] = 1.0f;

}
// backend is lantern.TensorDslCPU$BackendCPU@51b405f8
for(int x719=0; x719 < 1; x719++) {
float x720 = x707[x719];
x128[x719] = x720;

}
// 'sum' gradient.
bool x725 = x554 || true;
bool x726 = x725 || x554;
if (x726) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",x545,1);
assert(false && "");
}
bool x732 = x545 <= 1;
int32_t x733;
if (x732) {
x733 = 1;
} else {
x733 = x545;
}
int32_t x738;
if (x554) {
x738 = 0;
} else {
x738 = 1;
}
for(int x740=0; x740 < x733; x740++) {
int32_t x741 = x738 * x740;
float x742 = x697[x741];
float x743 = x712[0];
float x744 = x742 + x743;
x697[x741] = x744;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x750 = 0;
for(int x751=0; x751 < x545; x751++) {
int32_t x752 = x750;
int32_t x753 = x122[x751];
int32_t x754 = x752 + x753;
float x755 = x683[x754];
float x756 = x697[x751];
float x757 = -1.0f * x756;
float x758 = x755 + x757;
x683[x754] = x758;
x750 += 10;

}
float* x763 = (float*)myMalloc(x545 * sizeof(float));;
for(int x764=0; x764 < x545; x764++) {
int32_t x765 = x764;
int32_t x766 = x764 * 10;
int32_t x767 = x766;
for(int x768=0; x768 < 10; x768++) {
for(int x769=0; x769 < 1; x769++) {
int32_t x770 = x765;
int32_t x771 = x770 + x769;
float x772 = x763[x771];
int32_t x773 = x767;
int32_t x774 = x773 + x769;
float x775 = x683[x774];
float x776 = x772 + x775;
x763[x771] = x776;

}
x767 += 1;

}

}
int32_t x785 = 0;
for(int x786=0; x786 < x545; x786++) {
for(int x787=0; x787 < 10; x787++) {
int32_t x788 = x785;
float x789 = x591[x788];
float x790 = x683[x788];
float x791 = x625[x788];
float x795 = x763[x786];
double x792 = (double)x791;
double x793 = exp(x792);
float x794 = (float)x793;
float x796 = x794 * x795;
float x797 = x790 - x796;
float x798 = x789 + x797;
x591[x788] = x798;
x785 += 1;

}

}
for(int x805=0; x805 < x545; x805++) {
int32_t x807 = x592 * x805;
for(int x806=0; x806 < 10; x806++) {
float x809 = x53[x806];
int32_t x808 = x807 + x806;
float x810 = x591[x808];
float x811 = x809 + x810;
x53[x806] = x811;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x545,50,10,1,x591,10,x42,10,1,x587,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,x545,1,x570,50,x591,10,1,x51,10);
bool x820 = x554 || x554;
bool x821 = x820 || true;
bool x822;
if (x821) {
x822 = true;
} else {
x822 = false;
}
if (x822) {
} else {
printf("dimensions not compatible for broadcasting %d,%d, with %d,%d,\n",x545,50,x545,50);
assert(false && "");
}
float* x828 = (float*)myMalloc(x547 * sizeof(float));;
for(int x829=0; x829 < x545; x829++) {
int32_t x833 = x555 * x829;
int32_t x831 = 50 * x829;
for(int x830=0; x830 < 50; x830++) {
int32_t x834 = x833 + x830;
float x835 = x571[x834];
float x836 = x587[x834];
int32_t x832 = x830 + x831;
float x837 = x835 * x836;
x828[x832] = x837;

}

}
if (x822) {
} else {
printf("dimensions not compatible for broadcasting %d,%d, with %d,%d,\n",x545,50,x545,50);
assert(false && "");
}
for(int x847=0; x847 < x545; x847++) {
int32_t x849 = x555 * x847;
for(int x848=0; x848 < 50; x848++) {
int32_t x850 = x849 + x848;
float x851 = x550[x850];
float x852 = x828[x850];
float x853 = x851 + x852;
x550[x850] = x853;

}

}
for(int x859=0; x859 < x545; x859++) {
int32_t x861 = x555 * x859;
for(int x860=0; x860 < 50; x860++) {
float x863 = x41[x860];
int32_t x862 = x861 + x860;
float x864 = x550[x862];
float x865 = x863 + x864;
x41[x860] = x865;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x545,320,50,1,x550,50,x30,50,1,x523,320);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,x545,1,x428,320,x550,50,1,x39,50);
for(int x875=0; x875 < x425; x875++) {
int32_t x876 = x434[x875];
float x877 = x413[x876];
float x878 = x523[x875];
float x879 = x877 + x878;
x413[x876] = x879;

}
for(int x883=0; x883 < x332; x883++) {
float x884 = x399[x883];
float x885 = x335[x883];
bool x886 = x885 < 0.0f;
float x889;
if (x886) {
x889 = 0.0f;
} else {
float x887 = x413[x883];
x889 = x887;
}
float x890 = x884 + x889;
x399[x883] = x890;

}
// conv2D back-propagate
float* x895 = (float*)myMalloc(x354 * sizeof(float));;
for(int x896=0; x896 < 100; x896++) {
int32_t x897 = x896 * x227;
float* x898 = x326+x897;
int32_t x899 = x896 * x331;
float* x900 = x399+x899;
int32_t x901 = x896 * x352;
float* x902 = x895+x901;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 250,x330,20,1,x18,250,x900,x330,0,x902,x330);
for(int x904=0; x904 < 10; x904++) {
int32_t x908 = x904 * 5;
int32_t x909 = x908 * 5;
int32_t x910 = x909 * x329;
int32_t x911 = x910 * x329;
int32_t x920 = x904 * x225;
int32_t x921 = x920 * x225;
for(int x906=0; x906 < 5; x906++) {
int32_t x912 = x906 * 5;
int32_t x913 = x912 * x329;
int32_t x914 = x913 * x329;
int32_t x915 = x911 + x914;
for(int x907=0; x907 < 5; x907++) {
int32_t x916 = x907 * x329;
int32_t x917 = x916 * x329;
int32_t x918 = x915 + x917;
float* x919 = x902+x918;
float* x922 = x898+x921;
for(int x923=0; x923 < x329; x923++) {
int32_t x924 = x923 + x906;
int32_t x925 = x924 * x225;
int32_t x926 = x925 + x907;
float* x927 = x922+x926;
int32_t x928 = x923 * x329;
float* x929 = x919+x928;
for(int x930=0; x930 < x329; x930++) {
float x931 = x927[x930];
float x932 = x929[x930];
float x933 = x931 + x932;
x927[x930] = x933;

}

}

}

}

}

}
for(int x947=0; x947 < 100; x947++) {
int32_t x948 = x947 * x331;
float* x949 = x399+x948;
int32_t x950 = x947 * x352;
float* x951 = x355+x950;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,250,x330,1.0,x949,x330,x951,x330,1,x27,250);
for(int x953=0; x953 < 20; x953++) {
float x954 = 0.0f;
int32_t x955 = x953 * x329;
int32_t x956 = x955 * x329;
float* x957 = x949+x956;
for(int x958=0; x958 < x330; x958++) {
float x959 = x957[x958];
x954 += x959;

}
float x963 = x29[x953];
float x964 = x954;
float x965 = 1.0f * x964;
float x966 = x963 + x965;
x29[x953] = x966;

}

}
for(int x973=0; x973 < x228; x973++) {
int32_t x974 = x237[x973];
float x975 = x216[x974];
float x976 = x326[x973];
float x977 = x975 + x976;
x216[x974] = x977;

}
for(int x981=0; x981 < x133; x981++) {
float x982 = x202[x981];
float x983 = x136[x981];
bool x984 = x983 < 0.0f;
float x987;
if (x984) {
x987 = 0.0f;
} else {
float x985 = x216[x981];
x987 = x985;
}
float x988 = x982 + x987;
x202[x981] = x988;

}
// conv2D back-propagate
float* x993 = (float*)myMalloc(x156 * sizeof(float));;
for(int x994=0; x994 < 100; x994++) {
int32_t x995 = x994 * x132;
float* x996 = x202+x995;
int32_t x997 = x994 * x154;
float* x998 = x157+x997;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 10,25,x131,1.0,x996,x131,x998,x131,1,x15,25);
for(int x1000=0; x1000 < 10; x1000++) {
float x1001 = 0.0f;
int32_t x1002 = x1000 * x130;
int32_t x1003 = x1002 * x130;
float* x1004 = x996+x1003;
for(int x1005=0; x1005 < x131; x1005++) {
float x1006 = x1004[x1005];
x1001 += x1006;

}
float x1010 = x17[x1000];
float x1011 = x1001;
float x1012 = 1.0f * x1011;
float x1013 = x1010 + x1012;
x17[x1000] = x1013;

}

}
float x1019 = x128[0];
x109 += x1019;
for(int x1021=0; x1021 < 20; x1021++) {
float x1022 = x28[x1021];
float x1024 = x29[x1021];
float x1023 = x1022 * 1.0f;
float x1025 = x1024 * -5.0E-4f;
float x1026 = x1023 + x1025;
x28[x1021] = x1026;

}
for(int x1030=0; x1030 < 20; x1030++) {
x29[x1030] = 0.0f;

}
for(int x1034=0; x1034 < 5000; x1034++) {
float x1035 = x18[x1034];
float x1037 = x27[x1034];
float x1036 = x1035 * 1.0f;
float x1038 = x1037 * -5.0E-4f;
float x1039 = x1036 + x1038;
x18[x1034] = x1039;

}
for(int x1043=0; x1043 < 5000; x1043++) {
x27[x1043] = 0.0f;

}
for(int x1047=0; x1047 < 16000; x1047++) {
float x1048 = x30[x1047];
float x1050 = x39[x1047];
float x1049 = x1048 * 1.0f;
float x1051 = x1050 * -5.0E-4f;
float x1052 = x1049 + x1051;
x30[x1047] = x1052;

}
for(int x1056=0; x1056 < 16000; x1056++) {
x39[x1056] = 0.0f;

}
for(int x1060=0; x1060 < 50; x1060++) {
float x1061 = x40[x1060];
float x1063 = x41[x1060];
float x1062 = x1061 * 1.0f;
float x1064 = x1063 * -5.0E-4f;
float x1065 = x1062 + x1064;
x40[x1060] = x1065;

}
for(int x1069=0; x1069 < 50; x1069++) {
x41[x1069] = 0.0f;

}
for(int x1073=0; x1073 < 10; x1073++) {
float x1074 = x16[x1073];
float x1076 = x17[x1073];
float x1075 = x1074 * 1.0f;
float x1077 = x1076 * -5.0E-4f;
float x1078 = x1075 + x1077;
x16[x1073] = x1078;

}
for(int x1082=0; x1082 < 10; x1082++) {
x17[x1082] = 0.0f;

}
for(int x1086=0; x1086 < 250; x1086++) {
float x1087 = x6[x1086];
float x1089 = x15[x1086];
float x1088 = x1087 * 1.0f;
float x1090 = x1089 * -5.0E-4f;
float x1091 = x1088 + x1090;
x6[x1086] = x1091;

}
for(int x1095=0; x1095 < 250; x1095++) {
x15[x1095] = 0.0f;

}
for(int x1099=0; x1099 < 10; x1099++) {
float x1100 = x52[x1099];
float x1102 = x53[x1099];
float x1101 = x1100 * 1.0f;
float x1103 = x1102 * -5.0E-4f;
float x1104 = x1101 + x1103;
x52[x1099] = x1104;

}
for(int x1108=0; x1108 < 10; x1108++) {
x53[x1108] = 0.0f;

}
for(int x1112=0; x1112 < 500; x1112++) {
float x1113 = x42[x1112];
float x1115 = x51[x1112];
float x1114 = x1113 * 1.0f;
float x1116 = x1115 * -5.0E-4f;
float x1117 = x1114 + x1116;
x42[x1112] = x1117;

}
for(int x1121=0; x1121 < 500; x1121++) {
x51[x1121] = 0.0f;

}
int32_t x1125 = x106;
int32_t x1127 = x1125 % x1126;
bool x1128 = x1127 == 0;
if (x1128) {
float x1133 = x109;
double x1129 = (double)x1125;
double x1130 = 100.0 * x1129;
double x1132 = x1130 / x1131;
float x1134 = (float)x1125;
float x1135 = x1133 / x1134;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x102,x1125,x65,x1132,x1135);
fflush(stdout);
} else {
}
int64_t x1140 = (long)mallocAddr;
int64_t x1141 = x1140 - x99;
memset((void*)x99, 0, x1141);
mallocAddr = (void*)x99;
x115 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1149 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1150 = x1149 / 1000LL;
int64_t x1152 = x1149 / x1151;
printf("Training completed in %ldms (%ld us/images)\n",x1150,x1152);
float x1154 = x109;
float x1156 = x1154 / x1155;
double x1157 = (double)x1156;
x98[x102] = x1157;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1163 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1168 = (long)fopen(x0, "w");
fprintf((FILE *)x1168, "unit: %s\n", "1 epoch");
for(int x1170=0; x1170 < 4; x1170++) {
double x1171 = x98[x1170];
fprintf((FILE *)x1168, "%lf\n", x1171);

}
float x1164 = (float)x1163;
float x1165 = x1164 / 1000000.0f;
float x1166 = x1165 - x96;
float x1167 = x1166 / 4.0f;
fprintf((FILE *)x1168, "run time: %lf %lf\n", x96, x1167);
fclose((FILE*)x1168);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

