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
double x2 = ((double)clock() / CLOCKS_PER_SEC);
int* x3 = (int32_t*)myMalloc(1 * sizeof(int32_t));;
int64_t x4 = (long)fopen("small_glove.txt", "r");
if (fscanf((FILE *)x4,"%d", &x3[0])!=1) perror("Error reading file");
int32_t x6 = x3[0];
float** x7 = (float**)myMalloc(x6 * sizeof(float*));;
for(int x9=0; x9 < x6; x9++) {
float* x10 = (float*)myMalloc(300 * sizeof(float));;
x7[x9] = x10;
for(int x13=0; x13 < 300; x13++) {
float* x14 = x7[x9];
if (fscanf((FILE *)x4,"%f", &x14[x13])!=1) perror("Error reading file");

}

}
fclose((FILE*)x4);
int* x21 = (int32_t*)myMalloc(1 * sizeof(int32_t));;
int64_t x22 = (long)fopen("array_tree.txt", "r");
if (fscanf((FILE *)x22,"%d", &x21[0])!=1) perror("Error reading file");
int32_t x24 = x21[0];
int32_t x25 = x24 * 4;
int** x26 = (int**)myMalloc(x25 * sizeof(int*));;
int* x27 = (int32_t*)myMalloc(1 * sizeof(int32_t));;
for(int x29=0; x29 < x24; x29++) {
if (fscanf((FILE *)x22,"%d", &x27[0])!=1) perror("Error reading file");
int32_t x33 = x29 * 4;
for(int x32=0; x32 < 4; x32++) {
int32_t x35 = x27[0];
int* x36 = (int32_t*)myMalloc(x35 * sizeof(int32_t));;
int32_t x34 = x33 + x32;
x26[x34] = x36;
int32_t x38 = x27[0];
for(int x40=0; x40 < x38; x40++) {
int* x41 = x26[x34];
if (fscanf((FILE *)x22,"%d", &x41[x40])!=1) perror("Error reading file");

}

}

}
fclose((FILE*)x22);
float* x50 = (float*)myMalloc(45000 * sizeof(float));;
for(int x52=0; x52 < 45000; x52++) {
float x53 = (float)rand()/RAND_MAX;
float x54 = x53 - 0.5f;
float x55 = x54 * 0.01f;
x50[x52] = x55;

}
float* x59 = (float*)myMalloc(150 * sizeof(float));;
float* x60 = (float*)myMalloc(22500 * sizeof(float));;
for(int x62=0; x62 < 22500; x62++) {
float x63 = (float)rand()/RAND_MAX;
float x64 = x63 - 0.5f;
float x65 = x64 * 0.01f;
x60[x62] = x65;

}
float* x69 = (float*)myMalloc(22500 * sizeof(float));;
for(int x70=0; x70 < 22500; x70++) {
float x71 = (float)rand()/RAND_MAX;
float x72 = x71 - 0.5f;
float x73 = x72 * 0.01f;
x69[x70] = x73;

}
float* x77 = (float*)myMalloc(150 * sizeof(float));;
float* x78 = (float*)myMalloc(750 * sizeof(float));;
for(int x80=0; x80 < 750; x80++) {
float x81 = (float)rand()/RAND_MAX;
float x82 = x81 - 0.5f;
float x83 = x82 * 0.01f;
x78[x80] = x83;

}
float* x87 = (float*)myMalloc(5 * sizeof(float));;
float* x88 = (float*)myMalloc(45000 * sizeof(float));;
float* x89 = (float*)myMalloc(150 * sizeof(float));;
float* x90 = (float*)myMalloc(22500 * sizeof(float));;
float* x91 = (float*)myMalloc(22500 * sizeof(float));;
float* x92 = (float*)myMalloc(150 * sizeof(float));;
float* x93 = (float*)myMalloc(750 * sizeof(float));;
float* x94 = (float*)myMalloc(5 * sizeof(float));;
float* x95 = (float*)myMalloc(45000 * sizeof(float));;
float* x96 = (float*)myMalloc(150 * sizeof(float));;
float* x97 = (float*)myMalloc(22500 * sizeof(float));;
float* x98 = (float*)myMalloc(22500 * sizeof(float));;
float* x99 = (float*)myMalloc(150 * sizeof(float));;
float* x100 = (float*)myMalloc(750 * sizeof(float));;
float* x101 = (float*)myMalloc(5 * sizeof(float));;
double* x102 = (double*)myMalloc(6 * sizeof(double));;
int64_t x103 = (long)mallocAddr;
double x104 = ((double)clock() / CLOCKS_PER_SEC);
for(int x106=0; x106 < 6; x106++) {
float x107 = 0.0f;
for(int x108=0; x108 < x24; x108++) {
float* x120 = (float*)myMalloc(1 * sizeof(float));;
float* x121 = (float*)myMalloc(1 * sizeof(float));;
float* x122 = (float*)myMalloc(150 * sizeof(float));;
float* x123 = (float*)myMalloc(150 * sizeof(float));;
int32_t x109 = x108 * 4;
int* x110 = x26[x109];
int32_t x111 = x109 + 1;
int* x112 = x26[x111];
int32_t x113 = x109 + 2;
int* x114 = x26[x113];
int32_t x115 = x109 + 3;
int* x116 = x26[x115];
function<void(int32_t,function<void(float**)>,float**)> x124 = [&](int32_t x125,function<void(float**)> x126,float** x127) {
float** x130 = x127;
float* x131 = x130[0];
float* x132 = x130[1];
float* x133 = x130[2];
float* x134 = x130[3];
int32_t x128 = x125;
bool x135 = x128 >= 0;
if (x135) {
int32_t x136 = x114[x128];
float** x803 = (float**)myMalloc(4 * sizeof(float*));;
x803[0] = x120;
x803[1] = x121;
x803[2] = x122;
x803[3] = x123;
function<void(float**)> x129 = x126;
function<void(float**)> x259 = [&](float** x260) {
float* x261 = x260[0];
float* x262 = x260[1];
float* x263 = x260[2];
float* x264 = x260[3];
float** x265 = (float**)myMalloc(4 * sizeof(float*));;
x265[0] = x261;
x265[1] = x262;
x265[2] = x263;
x265[3] = x264;
x129(x265);
};
function<void(float**)> x253 = [&](float** x254) {
float* x255 = x254[0];
float* x256 = x254[1];
float* x257 = x254[2];
float* x258 = x254[3];
float** x272 = (float**)myMalloc(4 * sizeof(float*));;
x272[0] = x255;
x272[1] = x256;
x272[2] = x257;
x272[3] = x258;
x259(x272);
};
function<void(float**)> x561 = [&](float** x562) {
float* x563 = x562[0];
float* x564 = x562[1];
float* x565 = x562[2];
float* x566 = x562[3];
float** x567 = (float**)myMalloc(4 * sizeof(float*));;
x567[0] = x563;
x567[1] = x564;
x567[2] = x565;
x567[3] = x566;
x129(x567);
};
function<void(float**)> x555 = [&](float** x556) {
float* x557 = x556[0];
float* x558 = x556[1];
float* x559 = x556[2];
float* x560 = x556[3];
float** x574 = (float**)myMalloc(4 * sizeof(float*));;
x574[0] = x557;
x574[1] = x558;
x574[2] = x559;
x574[3] = x560;
x561(x574);
};
function<void(float**)> x137 = [&](float** x138) {
float* x139 = x138[0];
float* x140 = x138[1];
float* x141 = x138[2];
float* x142 = x138[3];
int32_t x143 = x116[x128];
float** x795 = (float**)myMalloc(4 * sizeof(float*));;
x795[0] = x120;
x795[1] = x121;
x795[2] = x122;
x795[3] = x123;
function<void(float**)> x144 = [&](float** x145) {
float* x146 = x145[0];
float* x147 = x145[1];
float* x148 = x145[2];
float* x149 = x145[3];
int32_t x150 = x114[x128];
bool x151 = x150 < 0;
if (x151) {
int32_t x152 = x112[x128];
float* x153 = x7[x152];
float* x154 = (float*)myMalloc(300 * sizeof(float));;
// dot: List(150, 300), WrappedArray(300)
float* x156 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x50,300,x153,1,0,x156,1);
float* x158 = (float*)myMalloc(150 * sizeof(float));;
float* x159 = (float*)myMalloc(150 * sizeof(float));;
int32_t x160 = 0;
int32_t x161 = 0;
int32_t x162 = 0;
for(int x164=0; x164 < 150; x164++) {
int32_t x165 = x160;
int32_t x166 = x161;
float x167 = x156[x166];
int32_t x168 = x162;
float x169 = x59[x168];
float x170 = x167 + x169;
x159[x165] = x170;
x160 += 1;
x161 += 1;
x162 += 1;

}
float* x177 = (float*)myMalloc(150 * sizeof(float));;
float* x178 = (float*)myMalloc(150 * sizeof(float));;
for(int x179=0; x179 < 150; x179++) {
float x180 = x159[x179];
double x181 = (double)x180;
double x182 = tanh(x181);
float x183 = (float)x182;
x178[x179] = x183;

}
float* x187 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(5, 150), List(150)
float* x189 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x78,150,x178,1,0,x189,1);
float* x191 = (float*)myMalloc(5 * sizeof(float));;
float* x192 = (float*)myMalloc(5 * sizeof(float));;
int32_t x193 = 0;
int32_t x194 = 0;
int32_t x195 = 0;
for(int x197=0; x197 < 5; x197++) {
int32_t x198 = x193;
int32_t x199 = x194;
float x200 = x189[x199];
int32_t x201 = x195;
float x202 = x87[x201];
float x203 = x200 + x202;
x192[x198] = x203;
x193 += 1;
x194 += 1;
x195 += 1;

}
float* x210 = (float*)myMalloc(5 * sizeof(float));;
float x211 = -3.4028235E38f;
for(int x212=0; x212 < 5; x212++) {
float x213 = x211;
float x214 = x192[x212];
bool x215 = x214 > x213;
float x216;
if (x215) {
x216 = x214;
} else {
x216 = x213;
}
x211 = x216;

}
float x220 = x211;
float x221 = 0.0f;
for(int x222=0; x222 < 5; x222++) {
float x223 = x221;
float x224 = x192[x222];
float x225 = x211;
float x226 = x224 - x225;
double x227 = (double)x226;
double x228 = exp(x227);
float x229 = (float)x228;
float x230 = x223 + x229;
x221 = x230;

}
float x234 = x221;
float* x239 = (float*)myMalloc(5 * sizeof(float));;
double x235 = (double)x234;
double x236 = log(x235);
float x237 = (float)x236;
float x238 = x220 + x237;
for(int x240=0; x240 < 5; x240++) {
float x241 = x192[x240];
float x242 = x241 - x238;
x239[x240] = x242;

}
float* x246 = (float*)myMalloc(5 * sizeof(float));;
int32_t x247 = x110[x128];
float x248 = x239[x247];
float* x250 = (float*)myMalloc(1 * sizeof(float));;
float x249 = -1.0f * x248;
x250[0] = x249;
float* x252 = (float*)myMalloc(1 * sizeof(float));;
float** x279 = (float**)myMalloc(4 * sizeof(float*));;
x279[0] = x250;
x279[1] = x252;
x279[2] = x178;
x279[3] = x187;
x253(x279);
float x285 = x246[x247];
float x286 = x252[0];
float x287 = -1.0f * x286;
float x288 = x285 + x287;
x246[x247] = x288;
float x290 = 0.0f;
for(int x291=0; x291 < 5; x291++) {
float x292 = x290;
float x293 = x246[x291];
float x294 = x292 + x293;
x290 = x294;

}
float x298 = x290;
float* x299 = (float*)myMalloc(1 * sizeof(float));;
x299[0] = x298;
float x301 = x299[0];
for(int x302=0; x302 < 5; x302++) {
float x303 = x210[x302];
float x304 = x246[x302];
float x305 = x239[x302];
double x306 = (double)x305;
double x307 = exp(x306);
float x308 = (float)x307;
float x309 = x308 * x301;
float x310 = x304 - x309;
float x311 = x303 + x310;
x210[x302] = x311;

}
int32_t x315 = 0;
int32_t x316 = 0;
int32_t x317 = 0;
for(int x318=0; x318 < 5; x318++) {
int32_t x319 = x315;
float x320 = x191[x319];
float x321 = x189[x319];
int32_t x322 = x316;
float x323 = x87[x322];
int32_t x324 = x317;
float x325 = x210[x324];
float x326 = x320 + x325;
x191[x319] = x326;
float x328 = x94[x322];
float x329 = x189[x319];
float x330 = x87[x322];
float x331 = x210[x324];
float x332 = x328 + x331;
x94[x322] = x332;
x317 += 1;
x315 += 1;
x316 += 1;

}
// add_cartesian
int32_t x340 = 0;
for(int x341=0; x341 < 5; x341++) {
for(int x342=0; x342 < 150; x342++) {
int32_t x343 = x340;
int32_t x344 = x343 + x342;
float x345 = x93[x344];
float x346 = x178[x342];
float x347 = x191[x341];
float x348 = x346 * x347;
float x349 = x345 + x348;
x93[x344] = x349;

}
x340 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x78,150,x191,1,1,x187,1);
for(int x357=0; x357 < 150; x357++) {
float x358 = x177[x357];
float x359 = x178[x357];
float x362 = x187[x357];
float x360 = x359 * x359;
float x361 = 1.0f - x360;
float x363 = x361 * x362;
float x364 = x358 + x363;
x177[x357] = x364;

}
int32_t x368 = 0;
int32_t x369 = 0;
int32_t x370 = 0;
for(int x371=0; x371 < 150; x371++) {
int32_t x372 = x368;
float x373 = x158[x372];
float x374 = x156[x372];
int32_t x375 = x369;
float x376 = x59[x375];
int32_t x377 = x370;
float x378 = x177[x377];
float x379 = x373 + x378;
x158[x372] = x379;
float x381 = x89[x375];
float x382 = x156[x372];
float x383 = x59[x375];
float x384 = x177[x377];
float x385 = x381 + x384;
x89[x375] = x385;
x370 += 1;
x368 += 1;
x369 += 1;

}
// add_cartesian
int32_t x393 = 0;
for(int x394=0; x394 < 150; x394++) {
for(int x395=0; x395 < 300; x395++) {
int32_t x396 = x393;
int32_t x397 = x396 + x395;
float x398 = x88[x397];
float x399 = x153[x395];
float x400 = x158[x394];
float x401 = x399 * x400;
float x402 = x398 + x401;
x88[x397] = x402;

}
x393 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x50,300,x158,1,1,x154,1);
} else {
// dot: List(150, 150), WrappedArray(150)
float* x412 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x60,150,x141,1,0,x412,1);
float* x414 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x416 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x69,150,x148,1,0,x416,1);
float* x418 = (float*)myMalloc(150 * sizeof(float));;
float* x419 = (float*)myMalloc(150 * sizeof(float));;
int32_t x420 = 0;
int32_t x421 = 0;
int32_t x422 = 0;
for(int x423=0; x423 < 150; x423++) {
int32_t x424 = x420;
int32_t x425 = x421;
float x426 = x412[x425];
int32_t x427 = x422;
float x428 = x416[x427];
float x429 = x426 + x428;
x419[x424] = x429;
x420 += 1;
x421 += 1;
x422 += 1;

}
float* x436 = (float*)myMalloc(150 * sizeof(float));;
float* x437 = (float*)myMalloc(150 * sizeof(float));;
int32_t x438 = 0;
int32_t x439 = 0;
int32_t x440 = 0;
for(int x441=0; x441 < 150; x441++) {
int32_t x442 = x438;
int32_t x443 = x439;
float x444 = x419[x443];
int32_t x445 = x440;
float x446 = x77[x445];
float x447 = x444 + x446;
x437[x442] = x447;
x438 += 1;
x439 += 1;
x440 += 1;

}
float* x454 = (float*)myMalloc(150 * sizeof(float));;
float* x455 = (float*)myMalloc(150 * sizeof(float));;
for(int x456=0; x456 < 150; x456++) {
float x457 = x437[x456];
double x458 = (double)x457;
double x459 = tanh(x458);
float x460 = (float)x459;
x455[x456] = x460;

}
float* x464 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(5, 150), List(150)
float* x466 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x78,150,x455,1,0,x466,1);
float* x468 = (float*)myMalloc(5 * sizeof(float));;
float* x469 = (float*)myMalloc(5 * sizeof(float));;
int32_t x470 = 0;
int32_t x471 = 0;
int32_t x472 = 0;
for(int x473=0; x473 < 5; x473++) {
int32_t x474 = x470;
int32_t x475 = x471;
float x476 = x466[x475];
int32_t x477 = x472;
float x478 = x87[x477];
float x479 = x476 + x478;
x469[x474] = x479;
x470 += 1;
x471 += 1;
x472 += 1;

}
float* x486 = (float*)myMalloc(5 * sizeof(float));;
float* x487 = (float*)myMalloc(1 * sizeof(float));;
int32_t x488 = 0;
int32_t x489 = 0;
int32_t x490 = 0;
int32_t x491 = x488;
int32_t x492 = x489;
float x493 = x139[x492];
int32_t x494 = x490;
float x495 = x146[x494];
float x496 = x493 + x495;
x487[x491] = x496;
x488 += 1;
float* x499 = (float*)myMalloc(1 * sizeof(float));;
float x500 = -3.4028235E38f;
for(int x501=0; x501 < 5; x501++) {
float x502 = x500;
float x503 = x469[x501];
bool x504 = x503 > x502;
float x505;
if (x504) {
x505 = x503;
} else {
x505 = x502;
}
x500 = x505;

}
float x509 = x500;
float x510 = 0.0f;
for(int x511=0; x511 < 5; x511++) {
float x512 = x510;
float x513 = x469[x511];
float x514 = x500;
float x515 = x513 - x514;
double x516 = (double)x515;
double x517 = exp(x516);
float x518 = (float)x517;
float x519 = x512 + x518;
x510 = x519;

}
float x523 = x510;
float* x528 = (float*)myMalloc(5 * sizeof(float));;
double x524 = (double)x523;
double x525 = log(x524);
float x526 = (float)x525;
float x527 = x509 + x526;
for(int x529=0; x529 < 5; x529++) {
float x530 = x469[x529];
float x531 = x530 - x527;
x528[x529] = x531;

}
float* x535 = (float*)myMalloc(5 * sizeof(float));;
int32_t x536 = x110[x128];
float x537 = x528[x536];
float* x539 = (float*)myMalloc(1 * sizeof(float));;
float x538 = -1.0f * x537;
x539[0] = x538;
float* x541 = (float*)myMalloc(1 * sizeof(float));;
float* x542 = (float*)myMalloc(1 * sizeof(float));;
int32_t x543 = 0;
int32_t x544 = 0;
int32_t x545 = 0;
int32_t x546 = x543;
int32_t x547 = x544;
float x548 = x487[x547];
int32_t x549 = x545;
float x550 = x539[x549];
float x551 = x548 + x550;
x542[x546] = x551;
x543 += 1;
float* x554 = (float*)myMalloc(1 * sizeof(float));;
float** x581 = (float**)myMalloc(4 * sizeof(float*));;
x581[0] = x542;
x581[1] = x554;
x581[2] = x455;
x581[3] = x464;
x555(x581);
int32_t x587 = 0;
int32_t x588 = 0;
int32_t x589 = 0;
int32_t x590 = x587;
float x591 = x499[x590];
float x592 = x487[x590];
int32_t x593 = x588;
float x594 = x539[x593];
int32_t x595 = x589;
float x596 = x554[x595];
float x597 = x591 + x596;
x499[x590] = x597;
float x599 = x541[x593];
float x600 = x487[x590];
float x601 = x539[x593];
float x602 = x554[x595];
float x603 = x599 + x602;
x541[x593] = x603;
x589 += 1;
float x606 = x535[x536];
float x607 = x541[0];
float x608 = -1.0f * x607;
float x609 = x606 + x608;
x535[x536] = x609;
float x611 = 0.0f;
for(int x612=0; x612 < 5; x612++) {
float x613 = x611;
float x614 = x535[x612];
float x615 = x613 + x614;
x611 = x615;

}
float x619 = x611;
float* x620 = (float*)myMalloc(1 * sizeof(float));;
x620[0] = x619;
float x622 = x620[0];
for(int x623=0; x623 < 5; x623++) {
float x624 = x486[x623];
float x625 = x535[x623];
float x626 = x528[x623];
double x627 = (double)x626;
double x628 = exp(x627);
float x629 = (float)x628;
float x630 = x629 * x622;
float x631 = x625 - x630;
float x632 = x624 + x631;
x486[x623] = x632;

}
int32_t x636 = 0;
int32_t x637 = 0;
int32_t x638 = 0;
int32_t x639 = x636;
float x640 = x140[x639];
float x641 = x139[x639];
int32_t x642 = x637;
float x643 = x146[x642];
int32_t x644 = x638;
float x645 = x499[x644];
float x646 = x640 + x645;
x140[x639] = x646;
float x648 = x147[x642];
float x649 = x139[x639];
float x650 = x146[x642];
float x651 = x499[x644];
float x652 = x648 + x651;
x147[x642] = x652;
x638 += 1;
int32_t x655 = 0;
int32_t x656 = 0;
int32_t x657 = 0;
for(int x658=0; x658 < 5; x658++) {
int32_t x659 = x655;
float x660 = x468[x659];
float x661 = x466[x659];
int32_t x662 = x656;
float x663 = x87[x662];
int32_t x664 = x657;
float x665 = x486[x664];
float x666 = x660 + x665;
x468[x659] = x666;
float x668 = x94[x662];
float x669 = x466[x659];
float x670 = x87[x662];
float x671 = x486[x664];
float x672 = x668 + x671;
x94[x662] = x672;
x657 += 1;
x655 += 1;
x656 += 1;

}
// add_cartesian
int32_t x680 = 0;
for(int x681=0; x681 < 5; x681++) {
for(int x682=0; x682 < 150; x682++) {
int32_t x683 = x680;
int32_t x684 = x683 + x682;
float x685 = x93[x684];
float x686 = x455[x682];
float x687 = x468[x681];
float x688 = x686 * x687;
float x689 = x685 + x688;
x93[x684] = x689;

}
x680 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x78,150,x468,1,1,x464,1);
for(int x697=0; x697 < 150; x697++) {
float x698 = x454[x697];
float x699 = x455[x697];
float x702 = x464[x697];
float x700 = x699 * x699;
float x701 = 1.0f - x700;
float x703 = x701 * x702;
float x704 = x698 + x703;
x454[x697] = x704;

}
int32_t x708 = 0;
int32_t x709 = 0;
int32_t x710 = 0;
for(int x711=0; x711 < 150; x711++) {
int32_t x712 = x708;
float x713 = x436[x712];
float x714 = x419[x712];
int32_t x715 = x709;
float x716 = x77[x715];
int32_t x717 = x710;
float x718 = x454[x717];
float x719 = x713 + x718;
x436[x712] = x719;
float x721 = x92[x715];
float x722 = x419[x712];
float x723 = x77[x715];
float x724 = x454[x717];
float x725 = x721 + x724;
x92[x715] = x725;
x710 += 1;
x708 += 1;
x709 += 1;

}
int32_t x732 = 0;
int32_t x733 = 0;
int32_t x734 = 0;
for(int x735=0; x735 < 150; x735++) {
int32_t x736 = x732;
float x737 = x414[x736];
float x738 = x412[x736];
int32_t x739 = x733;
float x740 = x416[x739];
int32_t x741 = x734;
float x742 = x436[x741];
float x743 = x737 + x742;
x414[x736] = x743;
float x745 = x418[x739];
float x746 = x412[x736];
float x747 = x416[x739];
float x748 = x436[x741];
float x749 = x745 + x748;
x418[x739] = x749;
x734 += 1;
x732 += 1;
x733 += 1;

}
// add_cartesian
int32_t x757 = 0;
for(int x758=0; x758 < 150; x758++) {
for(int x759=0; x759 < 150; x759++) {
int32_t x760 = x757;
int32_t x761 = x760 + x759;
float x762 = x91[x761];
float x763 = x148[x759];
float x764 = x418[x758];
float x765 = x763 * x764;
float x766 = x762 + x765;
x91[x761] = x766;

}
x757 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x69,150,x418,1,1,x149,1);
// add_cartesian
int32_t x775 = 0;
for(int x776=0; x776 < 150; x776++) {
for(int x777=0; x777 < 150; x777++) {
int32_t x778 = x775;
int32_t x779 = x778 + x777;
float x780 = x90[x779];
float x781 = x141[x777];
float x782 = x414[x776];
float x783 = x781 * x782;
float x784 = x780 + x783;
x90[x779] = x784;

}
x775 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x60,150,x414,1,1,x142,1);
}
};
x124(x143,x144,x795);
};
x124(x136,x137,x803);
} else {
float** x824 = (float**)myMalloc(4 * sizeof(float*));;
x824[0] = x120;
x824[1] = x121;
x824[2] = x122;
x824[3] = x123;
function<void(float**)> x129 = x126;
function<void(float**)> x811 = [&](float** x812) {
float* x813 = x812[0];
float* x814 = x812[1];
float* x815 = x812[2];
float* x816 = x812[3];
float** x817 = (float**)myMalloc(4 * sizeof(float*));;
x817[0] = x813;
x817[1] = x814;
x817[2] = x815;
x817[3] = x816;
x129(x817);
};
x811(x824);
}
};
float* x117 = (float*)myMalloc(1 * sizeof(float));;
float* x118 = (float*)myMalloc(1 * sizeof(float));;
float* x119 = (float*)myMalloc(1 * sizeof(float));;
float** x843 = (float**)myMalloc(4 * sizeof(float*));;
x843[0] = x120;
x843[1] = x121;
x843[2] = x122;
x843[3] = x123;
function<void(float**)> x833 = [&](float** x834) {
float* x835 = x834[0];
float* x836 = x834[1];
float* x837 = x834[2];
float* x838 = x834[3];
x836[0] = 1.0f;
float x840 = x835[0];
x119[0] = x840;
};
x124(0,x833,x843);
float x850 = x119[0];
float x851 = x107;
float x852 = (float)x108;
float x853 = x851 * x852;
int32_t x854 = x108 + 1;
float x855 = (float)x854;
float x856 = x853 / x855;
float x857 = x850 / x855;
float x858 = x856 + x857;
x107 = x858;
for(int x860=0; x860 < 45000; x860++) {
float x861 = x88[x860];
float x862 = x861;
float x863 = x95[x860];
float x864 = x862;
float x865 = x864 * x864;
float x866 = x863 + x865;
x95[x860] = x866;
float x868 = x50[x860];
float x870 = x95[x860];
float x869 = 0.05f * x864;
double x871 = (double)x870;
double x872 = x871 + 9.99999993922529E-9;
double x873 = sqrt(x872);
float x874 = (float)x873;
float x875 = x869 / x874;
float x876 = x868 - x875;
x50[x860] = x876;
x88[x860] = 0.0f;

}
for(int x881=0; x881 < 150; x881++) {
float x882 = x89[x881];
float x883 = x882;
float x884 = x96[x881];
float x885 = x883;
float x886 = x885 * x885;
float x887 = x884 + x886;
x96[x881] = x887;
float x889 = x59[x881];
float x891 = x96[x881];
float x890 = 0.05f * x885;
double x892 = (double)x891;
double x893 = x892 + 9.99999993922529E-9;
double x894 = sqrt(x893);
float x895 = (float)x894;
float x896 = x890 / x895;
float x897 = x889 - x896;
x59[x881] = x897;
x89[x881] = 0.0f;

}
for(int x902=0; x902 < 22500; x902++) {
float x903 = x90[x902];
float x904 = x903;
float x905 = x97[x902];
float x906 = x904;
float x907 = x906 * x906;
float x908 = x905 + x907;
x97[x902] = x908;
float x910 = x60[x902];
float x912 = x97[x902];
float x911 = 0.05f * x906;
double x913 = (double)x912;
double x914 = x913 + 9.99999993922529E-9;
double x915 = sqrt(x914);
float x916 = (float)x915;
float x917 = x911 / x916;
float x918 = x910 - x917;
x60[x902] = x918;
x90[x902] = 0.0f;

}
for(int x923=0; x923 < 22500; x923++) {
float x924 = x91[x923];
float x925 = x924;
float x926 = x98[x923];
float x927 = x925;
float x928 = x927 * x927;
float x929 = x926 + x928;
x98[x923] = x929;
float x931 = x69[x923];
float x933 = x98[x923];
float x932 = 0.05f * x927;
double x934 = (double)x933;
double x935 = x934 + 9.99999993922529E-9;
double x936 = sqrt(x935);
float x937 = (float)x936;
float x938 = x932 / x937;
float x939 = x931 - x938;
x69[x923] = x939;
x91[x923] = 0.0f;

}
for(int x944=0; x944 < 150; x944++) {
float x945 = x92[x944];
float x946 = x945;
float x947 = x99[x944];
float x948 = x946;
float x949 = x948 * x948;
float x950 = x947 + x949;
x99[x944] = x950;
float x952 = x77[x944];
float x954 = x99[x944];
float x953 = 0.05f * x948;
double x955 = (double)x954;
double x956 = x955 + 9.99999993922529E-9;
double x957 = sqrt(x956);
float x958 = (float)x957;
float x959 = x953 / x958;
float x960 = x952 - x959;
x77[x944] = x960;
x92[x944] = 0.0f;

}
for(int x965=0; x965 < 750; x965++) {
float x966 = x93[x965];
float x967 = x966;
float x968 = x100[x965];
float x969 = x967;
float x970 = x969 * x969;
float x971 = x968 + x970;
x100[x965] = x971;
float x973 = x78[x965];
float x975 = x100[x965];
float x974 = 0.05f * x969;
double x976 = (double)x975;
double x977 = x976 + 9.99999993922529E-9;
double x978 = sqrt(x977);
float x979 = (float)x978;
float x980 = x974 / x979;
float x981 = x973 - x980;
x78[x965] = x981;
x93[x965] = 0.0f;

}
for(int x986=0; x986 < 5; x986++) {
float x987 = x94[x986];
float x988 = x987;
float x989 = x101[x986];
float x990 = x988;
float x991 = x990 * x990;
float x992 = x989 + x991;
x101[x986] = x992;
float x994 = x87[x986];
float x996 = x101[x986];
float x995 = 0.05f * x990;
double x997 = (double)x996;
double x998 = x997 + 9.99999993922529E-9;
double x999 = sqrt(x998);
float x1000 = (float)x999;
float x1001 = x995 / x1000;
float x1002 = x994 - x1001;
x87[x986] = x1002;
x94[x986] = 0.0f;

}
int64_t x1007 = (long)mallocAddr;
int64_t x1008 = x1007 - x103;
memset((void*)x103, 0, x1008);
mallocAddr = (void*)x103;

}
float x1013 = x107;
double x1014 = (double)x1013;
x102[x106] = x1014;
double x1016 = ((double)clock() / CLOCKS_PER_SEC);
double x1017 = x1016 - x104;
printf("epoc %d, average_loss %f, time %lf\n",x106,x1013,x1017);

}
double x1021 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1025 = (long)fopen(x0, "w");
fprintf((FILE *)x1025, "unit: %s\n", "1 epoch");
for(int x1027=0; x1027 < 6; x1027++) {
double x1028 = x102[x1027];
fprintf((FILE *)x1025, "%lf\n", x1028);

}
double x1022 = x104 - x2;
double x1023 = x1021 - x104;
double x1024 = x1023 / 6.0;
fprintf((FILE *)x1025, "run time: %lf %lf\n", x1022, x1024);
fclose((FILE*)x1025);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

