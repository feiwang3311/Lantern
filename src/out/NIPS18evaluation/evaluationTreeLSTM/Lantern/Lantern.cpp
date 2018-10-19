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
float* x60 = (float*)myMalloc(45000 * sizeof(float));;
for(int x61=0; x61 < 45000; x61++) {
float x62 = (float)rand()/RAND_MAX;
float x63 = x62 - 0.5f;
float x64 = x63 * 0.01f;
x60[x61] = x64;

}
float* x68 = (float*)myMalloc(150 * sizeof(float));;
float* x69 = (float*)myMalloc(45000 * sizeof(float));;
for(int x70=0; x70 < 45000; x70++) {
float x71 = (float)rand()/RAND_MAX;
float x72 = x71 - 0.5f;
float x73 = x72 * 0.01f;
x69[x70] = x73;

}
float* x77 = (float*)myMalloc(150 * sizeof(float));;
float* x78 = (float*)myMalloc(22500 * sizeof(float));;
for(int x80=0; x80 < 22500; x80++) {
float x81 = (float)rand()/RAND_MAX;
float x82 = x81 - 0.5f;
float x83 = x82 * 0.01f;
x78[x80] = x83;

}
float* x87 = (float*)myMalloc(22500 * sizeof(float));;
for(int x88=0; x88 < 22500; x88++) {
float x89 = (float)rand()/RAND_MAX;
float x90 = x89 - 0.5f;
float x91 = x90 * 0.01f;
x87[x88] = x91;

}
float* x95 = (float*)myMalloc(150 * sizeof(float));;
float* x96 = (float*)myMalloc(22500 * sizeof(float));;
for(int x97=0; x97 < 22500; x97++) {
float x98 = (float)rand()/RAND_MAX;
float x99 = x98 - 0.5f;
float x100 = x99 * 0.01f;
x96[x97] = x100;

}
float* x104 = (float*)myMalloc(22500 * sizeof(float));;
for(int x105=0; x105 < 22500; x105++) {
float x106 = (float)rand()/RAND_MAX;
float x107 = x106 - 0.5f;
float x108 = x107 * 0.01f;
x104[x105] = x108;

}
float* x112 = (float*)myMalloc(22500 * sizeof(float));;
for(int x113=0; x113 < 22500; x113++) {
float x114 = (float)rand()/RAND_MAX;
float x115 = x114 - 0.5f;
float x116 = x115 * 0.01f;
x112[x113] = x116;

}
float* x120 = (float*)myMalloc(22500 * sizeof(float));;
for(int x121=0; x121 < 22500; x121++) {
float x122 = (float)rand()/RAND_MAX;
float x123 = x122 - 0.5f;
float x124 = x123 * 0.01f;
x120[x121] = x124;

}
float* x128 = (float*)myMalloc(150 * sizeof(float));;
float* x129 = (float*)myMalloc(22500 * sizeof(float));;
for(int x130=0; x130 < 22500; x130++) {
float x131 = (float)rand()/RAND_MAX;
float x132 = x131 - 0.5f;
float x133 = x132 * 0.01f;
x129[x130] = x133;

}
float* x137 = (float*)myMalloc(22500 * sizeof(float));;
for(int x138=0; x138 < 22500; x138++) {
float x139 = (float)rand()/RAND_MAX;
float x140 = x139 - 0.5f;
float x141 = x140 * 0.01f;
x137[x138] = x141;

}
float* x145 = (float*)myMalloc(150 * sizeof(float));;
float* x146 = (float*)myMalloc(22500 * sizeof(float));;
for(int x147=0; x147 < 22500; x147++) {
float x148 = (float)rand()/RAND_MAX;
float x149 = x148 - 0.5f;
float x150 = x149 * 0.01f;
x146[x147] = x150;

}
float* x154 = (float*)myMalloc(22500 * sizeof(float));;
for(int x155=0; x155 < 22500; x155++) {
float x156 = (float)rand()/RAND_MAX;
float x157 = x156 - 0.5f;
float x158 = x157 * 0.01f;
x154[x155] = x158;

}
float* x162 = (float*)myMalloc(150 * sizeof(float));;
float* x163 = (float*)myMalloc(750 * sizeof(float));;
for(int x165=0; x165 < 750; x165++) {
float x166 = (float)rand()/RAND_MAX;
float x167 = x166 - 0.5f;
float x168 = x167 * 0.01f;
x163[x165] = x168;

}
float* x172 = (float*)myMalloc(5 * sizeof(float));;
float* x173 = (float*)myMalloc(45000 * sizeof(float));;
float* x174 = (float*)myMalloc(150 * sizeof(float));;
float* x175 = (float*)myMalloc(45000 * sizeof(float));;
float* x176 = (float*)myMalloc(150 * sizeof(float));;
float* x177 = (float*)myMalloc(45000 * sizeof(float));;
float* x178 = (float*)myMalloc(150 * sizeof(float));;
float* x179 = (float*)myMalloc(22500 * sizeof(float));;
float* x180 = (float*)myMalloc(22500 * sizeof(float));;
float* x181 = (float*)myMalloc(150 * sizeof(float));;
float* x182 = (float*)myMalloc(22500 * sizeof(float));;
float* x183 = (float*)myMalloc(22500 * sizeof(float));;
float* x184 = (float*)myMalloc(22500 * sizeof(float));;
float* x185 = (float*)myMalloc(22500 * sizeof(float));;
float* x186 = (float*)myMalloc(150 * sizeof(float));;
float* x187 = (float*)myMalloc(22500 * sizeof(float));;
float* x188 = (float*)myMalloc(22500 * sizeof(float));;
float* x189 = (float*)myMalloc(150 * sizeof(float));;
float* x190 = (float*)myMalloc(22500 * sizeof(float));;
float* x191 = (float*)myMalloc(22500 * sizeof(float));;
float* x192 = (float*)myMalloc(150 * sizeof(float));;
float* x193 = (float*)myMalloc(750 * sizeof(float));;
float* x194 = (float*)myMalloc(5 * sizeof(float));;
float* x195 = (float*)myMalloc(45000 * sizeof(float));;
float* x196 = (float*)myMalloc(150 * sizeof(float));;
float* x197 = (float*)myMalloc(45000 * sizeof(float));;
float* x198 = (float*)myMalloc(150 * sizeof(float));;
float* x199 = (float*)myMalloc(45000 * sizeof(float));;
float* x200 = (float*)myMalloc(150 * sizeof(float));;
float* x201 = (float*)myMalloc(22500 * sizeof(float));;
float* x202 = (float*)myMalloc(22500 * sizeof(float));;
float* x203 = (float*)myMalloc(150 * sizeof(float));;
float* x204 = (float*)myMalloc(22500 * sizeof(float));;
float* x205 = (float*)myMalloc(22500 * sizeof(float));;
float* x206 = (float*)myMalloc(22500 * sizeof(float));;
float* x207 = (float*)myMalloc(22500 * sizeof(float));;
float* x208 = (float*)myMalloc(150 * sizeof(float));;
float* x209 = (float*)myMalloc(22500 * sizeof(float));;
float* x210 = (float*)myMalloc(22500 * sizeof(float));;
float* x211 = (float*)myMalloc(150 * sizeof(float));;
float* x212 = (float*)myMalloc(22500 * sizeof(float));;
float* x213 = (float*)myMalloc(22500 * sizeof(float));;
float* x214 = (float*)myMalloc(150 * sizeof(float));;
float* x215 = (float*)myMalloc(750 * sizeof(float));;
float* x216 = (float*)myMalloc(5 * sizeof(float));;
double* x217 = (double*)myMalloc(6 * sizeof(double));;
int64_t x218 = (long)mallocAddr;
double x219 = ((double)clock() / CLOCKS_PER_SEC);
for(int x221=0; x221 < 6; x221++) {
float x222 = 0.0f;
for(int x223=0; x223 < x24; x223++) {
float* x236 = (float*)myMalloc(1 * sizeof(float));;
float* x237 = (float*)myMalloc(1 * sizeof(float));;
float* x238 = (float*)myMalloc(150 * sizeof(float));;
float* x239 = (float*)myMalloc(150 * sizeof(float));;
float* x240 = (float*)myMalloc(150 * sizeof(float));;
float* x241 = (float*)myMalloc(150 * sizeof(float));;
int32_t x224 = x223 % x24;
int32_t x225 = x224 * 4;
int* x226 = x26[x225];
int32_t x227 = x225 + 1;
int* x228 = x26[x227];
int32_t x229 = x225 + 2;
int* x230 = x26[x229];
int32_t x231 = x225 + 3;
int* x232 = x26[x231];
function<void(int32_t,function<void(float**)>,float**)> x242 = [&](int32_t x243,function<void(float**)> x244,float** x245) {
float** x248 = x245;
float* x249 = x248[0];
float* x250 = x248[1];
float* x251 = x248[2];
float* x252 = x248[3];
float* x253 = x248[4];
float* x254 = x248[5];
int32_t x246 = x243;
bool x255 = x246 >= 0;
if (x255) {
int32_t x256 = x230[x246];
float** x2562 = (float**)myMalloc(6 * sizeof(float*));;
x2562[0] = x236;
x2562[1] = x237;
x2562[2] = x238;
x2562[3] = x239;
x2562[4] = x240;
x2562[5] = x241;
function<void(float**)> x247 = x244;
function<void(float**)> x561 = [&](float** x562) {
float* x563 = x562[0];
float* x564 = x562[1];
float* x565 = x562[2];
float* x566 = x562[3];
float* x567 = x562[4];
float* x568 = x562[5];
float** x569 = (float**)myMalloc(6 * sizeof(float*));;
x569[0] = x563;
x569[1] = x564;
x569[2] = x565;
x569[3] = x566;
x569[4] = x567;
x569[5] = x568;
x247(x569);
};
function<void(float**)> x553 = [&](float** x554) {
float* x555 = x554[0];
float* x556 = x554[1];
float* x557 = x554[2];
float* x558 = x554[3];
float* x559 = x554[4];
float* x560 = x554[5];
float** x578 = (float**)myMalloc(6 * sizeof(float*));;
x578[0] = x555;
x578[1] = x556;
x578[2] = x557;
x578[3] = x558;
x578[4] = x559;
x578[5] = x560;
x561(x578);
};
function<void(float**)> x1602 = [&](float** x1603) {
float* x1604 = x1603[0];
float* x1605 = x1603[1];
float* x1606 = x1603[2];
float* x1607 = x1603[3];
float* x1608 = x1603[4];
float* x1609 = x1603[5];
float** x1610 = (float**)myMalloc(6 * sizeof(float*));;
x1610[0] = x1604;
x1610[1] = x1605;
x1610[2] = x1606;
x1610[3] = x1607;
x1610[4] = x1608;
x1610[5] = x1609;
x247(x1610);
};
function<void(float**)> x1594 = [&](float** x1595) {
float* x1596 = x1595[0];
float* x1597 = x1595[1];
float* x1598 = x1595[2];
float* x1599 = x1595[3];
float* x1600 = x1595[4];
float* x1601 = x1595[5];
float** x1619 = (float**)myMalloc(6 * sizeof(float*));;
x1619[0] = x1596;
x1619[1] = x1597;
x1619[2] = x1598;
x1619[3] = x1599;
x1619[4] = x1600;
x1619[5] = x1601;
x1602(x1619);
};
function<void(float**)> x257 = [&](float** x258) {
float* x259 = x258[0];
float* x260 = x258[1];
float* x261 = x258[2];
float* x262 = x258[3];
float* x263 = x258[4];
float* x264 = x258[5];
int32_t x265 = x232[x246];
float** x2552 = (float**)myMalloc(6 * sizeof(float*));;
x2552[0] = x236;
x2552[1] = x237;
x2552[2] = x238;
x2552[3] = x239;
x2552[4] = x240;
x2552[5] = x241;
function<void(float**)> x266 = [&](float** x267) {
float* x268 = x267[0];
float* x269 = x267[1];
float* x270 = x267[2];
float* x271 = x267[3];
float* x272 = x267[4];
float* x273 = x267[5];
float* x274 = (float*)myMalloc(5 * sizeof(float));;
int32_t x275 = x226[x246];
x274[x275] = 1.0f;
float* x277 = (float*)myMalloc(5 * sizeof(float));;
int32_t x278 = x230[x246];
bool x279 = x278 < 0;
if (x279) {
int32_t x280 = x228[x246];
float* x281 = x7[x280];
float* x282 = (float*)myMalloc(300 * sizeof(float));;
// dot: List(150, 300), WrappedArray(300)
float* x284 = (float*)myMalloc(150 * sizeof(float));;
for(int x286=0; x286 < 150; x286++) {
float x287 = 0.0f;
int32_t x289 = x286 * 300;
for(int x288=0; x288 < 300; x288++) {
int32_t x290 = x289 + x288;
float x291 = x50[x290];
float x292 = x281[x288];
float x293 = x291 * x292;
x287 += x293;

}
float x297 = x287;
x284[x286] = x297;

}
float* x301 = (float*)myMalloc(150 * sizeof(float));;
float* x302 = (float*)myMalloc(150 * sizeof(float));;
int32_t x303 = 0;
int32_t x304 = 0;
int32_t x305 = 0;
for(int x306=0; x306 < 150; x306++) {
int32_t x307 = x303;
int32_t x308 = x304;
float x309 = x284[x308];
int32_t x310 = x305;
float x311 = x59[x310];
float x312 = x309 + x311;
x302[x307] = x312;
x303 += 1;
x304 += 1;
x305 += 1;

}
float* x319 = (float*)myMalloc(150 * sizeof(float));;
float* x320 = (float*)myMalloc(150 * sizeof(float));;
for(int x321=0; x321 < 150; x321++) {
float x322 = x302[x321];
float x323 = -1.0f * x322;
double x324 = (double)x323;
double x325 = exp(x324);
float x326 = (float)x325;
float x327 = x326 + 1.0f;
float x328 = 1.0f / x327;
x320[x321] = x328;

}
float* x332 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 300), WrappedArray(300)
float* x334 = (float*)myMalloc(150 * sizeof(float));;
for(int x335=0; x335 < 150; x335++) {
float x336 = 0.0f;
int32_t x338 = x335 * 300;
for(int x337=0; x337 < 300; x337++) {
int32_t x339 = x338 + x337;
float x340 = x60[x339];
float x341 = x281[x337];
float x342 = x340 * x341;
x336 += x342;

}
float x346 = x336;
x334[x335] = x346;

}
float* x350 = (float*)myMalloc(150 * sizeof(float));;
float* x351 = (float*)myMalloc(150 * sizeof(float));;
int32_t x352 = 0;
int32_t x353 = 0;
int32_t x354 = 0;
for(int x355=0; x355 < 150; x355++) {
int32_t x356 = x352;
int32_t x357 = x353;
float x358 = x334[x357];
int32_t x359 = x354;
float x360 = x68[x359];
float x361 = x358 + x360;
x351[x356] = x361;
x352 += 1;
x353 += 1;
x354 += 1;

}
float* x368 = (float*)myMalloc(150 * sizeof(float));;
float* x369 = (float*)myMalloc(150 * sizeof(float));;
for(int x370=0; x370 < 150; x370++) {
float x371 = x351[x370];
float x372 = -1.0f * x371;
double x373 = (double)x372;
double x374 = exp(x373);
float x375 = (float)x374;
float x376 = x375 + 1.0f;
float x377 = 1.0f / x376;
x369[x370] = x377;

}
float* x381 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 300), WrappedArray(300)
float* x383 = (float*)myMalloc(150 * sizeof(float));;
for(int x384=0; x384 < 150; x384++) {
float x385 = 0.0f;
int32_t x387 = x384 * 300;
for(int x386=0; x386 < 300; x386++) {
int32_t x388 = x387 + x386;
float x389 = x69[x388];
float x390 = x281[x386];
float x391 = x389 * x390;
x385 += x391;

}
float x395 = x385;
x383[x384] = x395;

}
float* x399 = (float*)myMalloc(150 * sizeof(float));;
float* x400 = (float*)myMalloc(150 * sizeof(float));;
int32_t x401 = 0;
int32_t x402 = 0;
int32_t x403 = 0;
for(int x404=0; x404 < 150; x404++) {
int32_t x405 = x401;
int32_t x406 = x402;
float x407 = x383[x406];
int32_t x408 = x403;
float x409 = x77[x408];
float x410 = x407 + x409;
x400[x405] = x410;
x401 += 1;
x402 += 1;
x403 += 1;

}
float* x417 = (float*)myMalloc(150 * sizeof(float));;
float* x418 = (float*)myMalloc(150 * sizeof(float));;
for(int x419=0; x419 < 150; x419++) {
float x420 = x400[x419];
double x421 = (double)x420;
double x422 = tanh(x421);
float x423 = (float)x422;
x418[x419] = x423;

}
float* x427 = (float*)myMalloc(150 * sizeof(float));;
float* x428 = (float*)myMalloc(150 * sizeof(float));;
int32_t x429 = 0;
int32_t x430 = 0;
int32_t x431 = 0;
for(int x432=0; x432 < 150; x432++) {
int32_t x433 = x429;
int32_t x434 = x430;
float x435 = x320[x434];
int32_t x436 = x431;
float x437 = x418[x436];
float x438 = x435 * x437;
x428[x433] = x438;
x429 += 1;
x430 += 1;
x431 += 1;

}
float* x445 = (float*)myMalloc(150 * sizeof(float));;
float* x446 = (float*)myMalloc(150 * sizeof(float));;
for(int x447=0; x447 < 150; x447++) {
float x448 = x428[x447];
double x449 = (double)x448;
double x450 = tanh(x449);
float x451 = (float)x450;
x446[x447] = x451;

}
float* x455 = (float*)myMalloc(150 * sizeof(float));;
float* x456 = (float*)myMalloc(150 * sizeof(float));;
int32_t x457 = 0;
int32_t x458 = 0;
int32_t x459 = 0;
for(int x460=0; x460 < 150; x460++) {
int32_t x461 = x457;
int32_t x462 = x458;
float x463 = x369[x462];
int32_t x464 = x459;
float x465 = x446[x464];
float x466 = x463 * x465;
x456[x461] = x466;
x457 += 1;
x458 += 1;
x459 += 1;

}
float* x473 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(5, 150), List(150)
float* x475 = (float*)myMalloc(5 * sizeof(float));;
for(int x477=0; x477 < 5; x477++) {
float x478 = 0.0f;
int32_t x480 = x477 * 150;
for(int x479=0; x479 < 150; x479++) {
int32_t x481 = x480 + x479;
float x482 = x163[x481];
float x483 = x456[x479];
float x484 = x482 * x483;
x478 += x484;

}
float x488 = x478;
x475[x477] = x488;

}
float* x492 = (float*)myMalloc(5 * sizeof(float));;
float* x493 = (float*)myMalloc(5 * sizeof(float));;
int32_t x494 = 0;
int32_t x495 = 0;
int32_t x496 = 0;
for(int x497=0; x497 < 5; x497++) {
int32_t x498 = x494;
int32_t x499 = x495;
float x500 = x475[x499];
int32_t x501 = x496;
float x502 = x172[x501];
float x503 = x500 + x502;
x493[x498] = x503;
x494 += 1;
x495 += 1;
x496 += 1;

}
float* x510 = (float*)myMalloc(5 * sizeof(float));;
float x511 = -3.4028235E38f;
for(int x512=0; x512 < 5; x512++) {
float x513 = x511;
float x514 = x493[x512];
bool x515 = x514 > x513;
float x516;
if (x515) {
x516 = x514;
} else {
x516 = x513;
}
x511 = x516;

}
float x520 = x511;
float x521 = 0.0f;
for(int x522=0; x522 < 5; x522++) {
float x523 = x521;
float x524 = x493[x522];
float x525 = x511;
float x526 = x524 - x525;
double x527 = (double)x526;
double x528 = exp(x527);
float x529 = (float)x528;
float x530 = x523 + x529;
x521 = x530;

}
float x534 = x521;
float* x539 = (float*)myMalloc(5 * sizeof(float));;
double x535 = (double)x534;
double x536 = log(x535);
float x537 = (float)x536;
float x538 = x520 + x537;
for(int x540=0; x540 < 5; x540++) {
float x541 = x493[x540];
float x542 = x541 - x538;
x539[x540] = x542;

}
float* x546 = (float*)myMalloc(5 * sizeof(float));;
int32_t x547 = x226[x246];
float x548 = x539[x547];
float* x550 = (float*)myMalloc(1 * sizeof(float));;
float x549 = -1.0f * x548;
x550[0] = x549;
float* x552 = (float*)myMalloc(1 * sizeof(float));;
float** x587 = (float**)myMalloc(6 * sizeof(float*));;
x587[0] = x550;
x587[1] = x552;
x587[2] = x456;
x587[3] = x473;
x587[4] = x428;
x587[5] = x445;
x553(x587);
float x595 = x546[x547];
float x596 = x552[0];
float x597 = -1.0f * x596;
float x598 = x595 + x597;
x546[x547] = x598;
float x600 = 0.0f;
for(int x601=0; x601 < 5; x601++) {
float x602 = x600;
float x603 = x546[x601];
float x604 = x602 + x603;
x600 = x604;

}
float x608 = x600;
float* x609 = (float*)myMalloc(1 * sizeof(float));;
x609[0] = x608;
float x611 = x609[0];
for(int x612=0; x612 < 5; x612++) {
float x613 = x510[x612];
float x614 = x546[x612];
float x615 = x539[x612];
double x616 = (double)x615;
double x617 = exp(x616);
float x618 = (float)x617;
float x619 = x618 * x611;
float x620 = x614 - x619;
float x621 = x613 + x620;
x510[x612] = x621;

}
int32_t x625 = 0;
int32_t x626 = 0;
int32_t x627 = 0;
for(int x628=0; x628 < 5; x628++) {
int32_t x629 = x625;
float x630 = x492[x629];
float x631 = x475[x629];
int32_t x632 = x626;
float x633 = x172[x632];
int32_t x634 = x627;
float x635 = x510[x634];
float x636 = x630 + x635;
x492[x629] = x636;
float x638 = x194[x632];
float x639 = x475[x629];
float x640 = x172[x632];
float x641 = x510[x634];
float x642 = x638 + x641;
x194[x632] = x642;
x627 += 1;
x625 += 1;
x626 += 1;

}
// add_cartesian
int32_t x650 = 0;
for(int x651=0; x651 < 5; x651++) {
for(int x652=0; x652 < 150; x652++) {
int32_t x653 = x650;
int32_t x654 = x653 + x652;
float x655 = x193[x654];
float x656 = x456[x652];
float x657 = x492[x651];
float x658 = x656 * x657;
float x659 = x655 + x658;
x193[x654] = x659;

}
x650 += 150;

}
int32_t x666 = 0;
for(int x667=0; x667 < 5; x667++) {
for(int x668=0; x668 < 150; x668++) {
float x669 = x473[x668];
int32_t x670 = x666;
int32_t x671 = x670 + x668;
float x672 = x163[x671];
float x673 = x492[x667];
float x674 = x672 * x673;
float x675 = x669 + x674;
x473[x668] = x675;

}
x666 += 150;

}
int32_t x682 = 0;
int32_t x683 = 0;
int32_t x684 = 0;
for(int x685=0; x685 < 150; x685++) {
int32_t x686 = x682;
float x687 = x381[x686];
float x688 = x369[x686];
int32_t x689 = x683;
float x690 = x446[x689];
int32_t x691 = x684;
float x692 = x473[x691];
float x693 = x692 * x690;
float x694 = x687 + x693;
x381[x686] = x694;
float x696 = x455[x689];
float x697 = x369[x686];
float x698 = x446[x689];
float x699 = x473[x691];
float x700 = x699 * x697;
float x701 = x696 + x700;
x455[x689] = x701;
x684 += 1;
x682 += 1;
x683 += 1;

}
for(int x708=0; x708 < 150; x708++) {
float x709 = x445[x708];
float x710 = x446[x708];
float x713 = x455[x708];
float x711 = x710 * x710;
float x712 = 1.0f - x711;
float x714 = x712 * x713;
float x715 = x709 + x714;
x445[x708] = x715;

}
int32_t x719 = 0;
int32_t x720 = 0;
int32_t x721 = 0;
for(int x722=0; x722 < 150; x722++) {
int32_t x723 = x719;
float x724 = x332[x723];
float x725 = x320[x723];
int32_t x726 = x720;
float x727 = x418[x726];
int32_t x728 = x721;
float x729 = x445[x728];
float x730 = x729 * x727;
float x731 = x724 + x730;
x332[x723] = x731;
float x733 = x427[x726];
float x734 = x320[x723];
float x735 = x418[x726];
float x736 = x445[x728];
float x737 = x736 * x734;
float x738 = x733 + x737;
x427[x726] = x738;
x721 += 1;
x719 += 1;
x720 += 1;

}
for(int x745=0; x745 < 150; x745++) {
float x746 = x417[x745];
float x747 = x418[x745];
float x750 = x427[x745];
float x748 = x747 * x747;
float x749 = 1.0f - x748;
float x751 = x749 * x750;
float x752 = x746 + x751;
x417[x745] = x752;

}
int32_t x756 = 0;
int32_t x757 = 0;
int32_t x758 = 0;
for(int x759=0; x759 < 150; x759++) {
int32_t x760 = x756;
float x761 = x399[x760];
float x762 = x383[x760];
int32_t x763 = x757;
float x764 = x77[x763];
int32_t x765 = x758;
float x766 = x417[x765];
float x767 = x761 + x766;
x399[x760] = x767;
float x769 = x178[x763];
float x770 = x383[x760];
float x771 = x77[x763];
float x772 = x417[x765];
float x773 = x769 + x772;
x178[x763] = x773;
x758 += 1;
x756 += 1;
x757 += 1;

}
// add_cartesian
int32_t x781 = 0;
for(int x782=0; x782 < 150; x782++) {
for(int x783=0; x783 < 300; x783++) {
int32_t x784 = x781;
int32_t x785 = x784 + x783;
float x786 = x177[x785];
float x787 = x281[x783];
float x788 = x399[x782];
float x789 = x787 * x788;
float x790 = x786 + x789;
x177[x785] = x790;

}
x781 += 300;

}
int32_t x797 = 0;
for(int x798=0; x798 < 150; x798++) {
for(int x799=0; x799 < 300; x799++) {
float x800 = x282[x799];
int32_t x801 = x797;
int32_t x802 = x801 + x799;
float x803 = x69[x802];
float x804 = x399[x798];
float x805 = x803 * x804;
float x806 = x800 + x805;
x282[x799] = x806;

}
x797 += 300;

}
for(int x813=0; x813 < 150; x813++) {
float x814 = x368[x813];
float x815 = x369[x813];
float x818 = x381[x813];
float x816 = 1.0f - x815;
float x817 = x816 * x815;
float x819 = x817 * x818;
float x820 = x814 + x819;
x368[x813] = x820;

}
int32_t x824 = 0;
int32_t x825 = 0;
int32_t x826 = 0;
for(int x827=0; x827 < 150; x827++) {
int32_t x828 = x824;
float x829 = x350[x828];
float x830 = x334[x828];
int32_t x831 = x825;
float x832 = x68[x831];
int32_t x833 = x826;
float x834 = x368[x833];
float x835 = x829 + x834;
x350[x828] = x835;
float x837 = x176[x831];
float x838 = x334[x828];
float x839 = x68[x831];
float x840 = x368[x833];
float x841 = x837 + x840;
x176[x831] = x841;
x826 += 1;
x824 += 1;
x825 += 1;

}
// add_cartesian
int32_t x849 = 0;
for(int x850=0; x850 < 150; x850++) {
for(int x851=0; x851 < 300; x851++) {
int32_t x852 = x849;
int32_t x853 = x852 + x851;
float x854 = x175[x853];
float x855 = x281[x851];
float x856 = x350[x850];
float x857 = x855 * x856;
float x858 = x854 + x857;
x175[x853] = x858;

}
x849 += 300;

}
int32_t x865 = 0;
for(int x866=0; x866 < 150; x866++) {
for(int x867=0; x867 < 300; x867++) {
float x868 = x282[x867];
int32_t x869 = x865;
int32_t x870 = x869 + x867;
float x871 = x60[x870];
float x872 = x350[x866];
float x873 = x871 * x872;
float x874 = x868 + x873;
x282[x867] = x874;

}
x865 += 300;

}
for(int x881=0; x881 < 150; x881++) {
float x882 = x319[x881];
float x883 = x320[x881];
float x886 = x332[x881];
float x884 = 1.0f - x883;
float x885 = x884 * x883;
float x887 = x885 * x886;
float x888 = x882 + x887;
x319[x881] = x888;

}
int32_t x892 = 0;
int32_t x893 = 0;
int32_t x894 = 0;
for(int x895=0; x895 < 150; x895++) {
int32_t x896 = x892;
float x897 = x301[x896];
float x898 = x284[x896];
int32_t x899 = x893;
float x900 = x59[x899];
int32_t x901 = x894;
float x902 = x319[x901];
float x903 = x897 + x902;
x301[x896] = x903;
float x905 = x174[x899];
float x906 = x284[x896];
float x907 = x59[x899];
float x908 = x319[x901];
float x909 = x905 + x908;
x174[x899] = x909;
x894 += 1;
x892 += 1;
x893 += 1;

}
// add_cartesian
int32_t x917 = 0;
for(int x918=0; x918 < 150; x918++) {
for(int x919=0; x919 < 300; x919++) {
int32_t x920 = x917;
int32_t x921 = x920 + x919;
float x922 = x173[x921];
float x923 = x281[x919];
float x924 = x301[x918];
float x925 = x923 * x924;
float x926 = x922 + x925;
x173[x921] = x926;

}
x917 += 300;

}
int32_t x933 = 0;
for(int x934=0; x934 < 150; x934++) {
for(int x935=0; x935 < 300; x935++) {
float x936 = x282[x935];
int32_t x937 = x933;
int32_t x938 = x937 + x935;
float x939 = x50[x938];
float x940 = x301[x934];
float x941 = x939 * x940;
float x942 = x936 + x941;
x282[x935] = x942;

}
x933 += 300;

}
} else {
// dot: List(150, 150), WrappedArray(150)
float* x951 = (float*)myMalloc(150 * sizeof(float));;
for(int x952=0; x952 < 150; x952++) {
float x953 = 0.0f;
int32_t x955 = x952 * 150;
for(int x954=0; x954 < 150; x954++) {
int32_t x956 = x955 + x954;
float x957 = x78[x956];
float x958 = x261[x954];
float x959 = x957 * x958;
x953 += x959;

}
float x963 = x953;
x951[x952] = x963;

}
float* x967 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x969 = (float*)myMalloc(150 * sizeof(float));;
for(int x970=0; x970 < 150; x970++) {
float x971 = 0.0f;
int32_t x973 = x970 * 150;
for(int x972=0; x972 < 150; x972++) {
int32_t x974 = x973 + x972;
float x975 = x87[x974];
float x976 = x270[x972];
float x977 = x975 * x976;
x971 += x977;

}
float x981 = x971;
x969[x970] = x981;

}
float* x985 = (float*)myMalloc(150 * sizeof(float));;
float* x986 = (float*)myMalloc(150 * sizeof(float));;
int32_t x987 = 0;
int32_t x988 = 0;
int32_t x989 = 0;
for(int x990=0; x990 < 150; x990++) {
int32_t x991 = x987;
int32_t x992 = x988;
float x993 = x951[x992];
int32_t x994 = x989;
float x995 = x969[x994];
float x996 = x993 + x995;
x986[x991] = x996;
x987 += 1;
x988 += 1;
x989 += 1;

}
float* x1003 = (float*)myMalloc(150 * sizeof(float));;
float* x1004 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1005 = 0;
int32_t x1006 = 0;
int32_t x1007 = 0;
for(int x1008=0; x1008 < 150; x1008++) {
int32_t x1009 = x1005;
int32_t x1010 = x1006;
float x1011 = x986[x1010];
int32_t x1012 = x1007;
float x1013 = x95[x1012];
float x1014 = x1011 + x1013;
x1004[x1009] = x1014;
x1005 += 1;
x1006 += 1;
x1007 += 1;

}
float* x1021 = (float*)myMalloc(150 * sizeof(float));;
float* x1022 = (float*)myMalloc(150 * sizeof(float));;
for(int x1023=0; x1023 < 150; x1023++) {
float x1024 = x1004[x1023];
float x1025 = -1.0f * x1024;
double x1026 = (double)x1025;
double x1027 = exp(x1026);
float x1028 = (float)x1027;
float x1029 = x1028 + 1.0f;
float x1030 = 1.0f / x1029;
x1022[x1023] = x1030;

}
float* x1034 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1036 = (float*)myMalloc(150 * sizeof(float));;
for(int x1037=0; x1037 < 150; x1037++) {
float x1038 = 0.0f;
int32_t x1040 = x1037 * 150;
for(int x1039=0; x1039 < 150; x1039++) {
int32_t x1041 = x1040 + x1039;
float x1042 = x96[x1041];
float x1043 = x261[x1039];
float x1044 = x1042 * x1043;
x1038 += x1044;

}
float x1048 = x1038;
x1036[x1037] = x1048;

}
float* x1052 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1054 = (float*)myMalloc(150 * sizeof(float));;
for(int x1055=0; x1055 < 150; x1055++) {
float x1056 = 0.0f;
int32_t x1058 = x1055 * 150;
for(int x1057=0; x1057 < 150; x1057++) {
int32_t x1059 = x1058 + x1057;
float x1060 = x104[x1059];
float x1061 = x270[x1057];
float x1062 = x1060 * x1061;
x1056 += x1062;

}
float x1066 = x1056;
x1054[x1055] = x1066;

}
float* x1070 = (float*)myMalloc(150 * sizeof(float));;
float* x1071 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1072 = 0;
int32_t x1073 = 0;
int32_t x1074 = 0;
for(int x1075=0; x1075 < 150; x1075++) {
int32_t x1076 = x1072;
int32_t x1077 = x1073;
float x1078 = x1036[x1077];
int32_t x1079 = x1074;
float x1080 = x1054[x1079];
float x1081 = x1078 + x1080;
x1071[x1076] = x1081;
x1072 += 1;
x1073 += 1;
x1074 += 1;

}
float* x1088 = (float*)myMalloc(150 * sizeof(float));;
float* x1089 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1090 = 0;
int32_t x1091 = 0;
int32_t x1092 = 0;
for(int x1093=0; x1093 < 150; x1093++) {
int32_t x1094 = x1090;
int32_t x1095 = x1091;
float x1096 = x1071[x1095];
int32_t x1097 = x1092;
float x1098 = x128[x1097];
float x1099 = x1096 + x1098;
x1089[x1094] = x1099;
x1090 += 1;
x1091 += 1;
x1092 += 1;

}
float* x1106 = (float*)myMalloc(150 * sizeof(float));;
float* x1107 = (float*)myMalloc(150 * sizeof(float));;
for(int x1108=0; x1108 < 150; x1108++) {
float x1109 = x1089[x1108];
float x1110 = -1.0f * x1109;
double x1111 = (double)x1110;
double x1112 = exp(x1111);
float x1113 = (float)x1112;
float x1114 = x1113 + 1.0f;
float x1115 = 1.0f / x1114;
x1107[x1108] = x1115;

}
float* x1119 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1121 = (float*)myMalloc(150 * sizeof(float));;
for(int x1122=0; x1122 < 150; x1122++) {
float x1123 = 0.0f;
int32_t x1125 = x1122 * 150;
for(int x1124=0; x1124 < 150; x1124++) {
int32_t x1126 = x1125 + x1124;
float x1127 = x112[x1126];
float x1128 = x261[x1124];
float x1129 = x1127 * x1128;
x1123 += x1129;

}
float x1133 = x1123;
x1121[x1122] = x1133;

}
float* x1137 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1139 = (float*)myMalloc(150 * sizeof(float));;
for(int x1140=0; x1140 < 150; x1140++) {
float x1141 = 0.0f;
int32_t x1143 = x1140 * 150;
for(int x1142=0; x1142 < 150; x1142++) {
int32_t x1144 = x1143 + x1142;
float x1145 = x120[x1144];
float x1146 = x270[x1142];
float x1147 = x1145 * x1146;
x1141 += x1147;

}
float x1151 = x1141;
x1139[x1140] = x1151;

}
float* x1155 = (float*)myMalloc(150 * sizeof(float));;
float* x1156 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1157 = 0;
int32_t x1158 = 0;
int32_t x1159 = 0;
for(int x1160=0; x1160 < 150; x1160++) {
int32_t x1161 = x1157;
int32_t x1162 = x1158;
float x1163 = x1121[x1162];
int32_t x1164 = x1159;
float x1165 = x1139[x1164];
float x1166 = x1163 + x1165;
x1156[x1161] = x1166;
x1157 += 1;
x1158 += 1;
x1159 += 1;

}
float* x1173 = (float*)myMalloc(150 * sizeof(float));;
float* x1174 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1175 = 0;
int32_t x1176 = 0;
int32_t x1177 = 0;
for(int x1178=0; x1178 < 150; x1178++) {
int32_t x1179 = x1175;
int32_t x1180 = x1176;
float x1181 = x1156[x1180];
int32_t x1182 = x1177;
float x1183 = x128[x1182];
float x1184 = x1181 + x1183;
x1174[x1179] = x1184;
x1175 += 1;
x1176 += 1;
x1177 += 1;

}
float* x1191 = (float*)myMalloc(150 * sizeof(float));;
float* x1192 = (float*)myMalloc(150 * sizeof(float));;
for(int x1193=0; x1193 < 150; x1193++) {
float x1194 = x1174[x1193];
float x1195 = -1.0f * x1194;
double x1196 = (double)x1195;
double x1197 = exp(x1196);
float x1198 = (float)x1197;
float x1199 = x1198 + 1.0f;
float x1200 = 1.0f / x1199;
x1192[x1193] = x1200;

}
float* x1204 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1206 = (float*)myMalloc(150 * sizeof(float));;
for(int x1207=0; x1207 < 150; x1207++) {
float x1208 = 0.0f;
int32_t x1210 = x1207 * 150;
for(int x1209=0; x1209 < 150; x1209++) {
int32_t x1211 = x1210 + x1209;
float x1212 = x129[x1211];
float x1213 = x261[x1209];
float x1214 = x1212 * x1213;
x1208 += x1214;

}
float x1218 = x1208;
x1206[x1207] = x1218;

}
float* x1222 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1224 = (float*)myMalloc(150 * sizeof(float));;
for(int x1225=0; x1225 < 150; x1225++) {
float x1226 = 0.0f;
int32_t x1228 = x1225 * 150;
for(int x1227=0; x1227 < 150; x1227++) {
int32_t x1229 = x1228 + x1227;
float x1230 = x137[x1229];
float x1231 = x270[x1227];
float x1232 = x1230 * x1231;
x1226 += x1232;

}
float x1236 = x1226;
x1224[x1225] = x1236;

}
float* x1240 = (float*)myMalloc(150 * sizeof(float));;
float* x1241 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1242 = 0;
int32_t x1243 = 0;
int32_t x1244 = 0;
for(int x1245=0; x1245 < 150; x1245++) {
int32_t x1246 = x1242;
int32_t x1247 = x1243;
float x1248 = x1206[x1247];
int32_t x1249 = x1244;
float x1250 = x1224[x1249];
float x1251 = x1248 + x1250;
x1241[x1246] = x1251;
x1242 += 1;
x1243 += 1;
x1244 += 1;

}
float* x1258 = (float*)myMalloc(150 * sizeof(float));;
float* x1259 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1260 = 0;
int32_t x1261 = 0;
int32_t x1262 = 0;
for(int x1263=0; x1263 < 150; x1263++) {
int32_t x1264 = x1260;
int32_t x1265 = x1261;
float x1266 = x1241[x1265];
int32_t x1267 = x1262;
float x1268 = x145[x1267];
float x1269 = x1266 + x1268;
x1259[x1264] = x1269;
x1260 += 1;
x1261 += 1;
x1262 += 1;

}
float* x1276 = (float*)myMalloc(150 * sizeof(float));;
float* x1277 = (float*)myMalloc(150 * sizeof(float));;
for(int x1278=0; x1278 < 150; x1278++) {
float x1279 = x1259[x1278];
float x1280 = -1.0f * x1279;
double x1281 = (double)x1280;
double x1282 = exp(x1281);
float x1283 = (float)x1282;
float x1284 = x1283 + 1.0f;
float x1285 = 1.0f / x1284;
x1277[x1278] = x1285;

}
float* x1289 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1291 = (float*)myMalloc(150 * sizeof(float));;
for(int x1292=0; x1292 < 150; x1292++) {
float x1293 = 0.0f;
int32_t x1295 = x1292 * 150;
for(int x1294=0; x1294 < 150; x1294++) {
int32_t x1296 = x1295 + x1294;
float x1297 = x146[x1296];
float x1298 = x261[x1294];
float x1299 = x1297 * x1298;
x1293 += x1299;

}
float x1303 = x1293;
x1291[x1292] = x1303;

}
float* x1307 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1309 = (float*)myMalloc(150 * sizeof(float));;
for(int x1310=0; x1310 < 150; x1310++) {
float x1311 = 0.0f;
int32_t x1313 = x1310 * 150;
for(int x1312=0; x1312 < 150; x1312++) {
int32_t x1314 = x1313 + x1312;
float x1315 = x154[x1314];
float x1316 = x270[x1312];
float x1317 = x1315 * x1316;
x1311 += x1317;

}
float x1321 = x1311;
x1309[x1310] = x1321;

}
float* x1325 = (float*)myMalloc(150 * sizeof(float));;
float* x1326 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1327 = 0;
int32_t x1328 = 0;
int32_t x1329 = 0;
for(int x1330=0; x1330 < 150; x1330++) {
int32_t x1331 = x1327;
int32_t x1332 = x1328;
float x1333 = x1291[x1332];
int32_t x1334 = x1329;
float x1335 = x1309[x1334];
float x1336 = x1333 + x1335;
x1326[x1331] = x1336;
x1327 += 1;
x1328 += 1;
x1329 += 1;

}
float* x1343 = (float*)myMalloc(150 * sizeof(float));;
float* x1344 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1345 = 0;
int32_t x1346 = 0;
int32_t x1347 = 0;
for(int x1348=0; x1348 < 150; x1348++) {
int32_t x1349 = x1345;
int32_t x1350 = x1346;
float x1351 = x1326[x1350];
int32_t x1352 = x1347;
float x1353 = x162[x1352];
float x1354 = x1351 + x1353;
x1344[x1349] = x1354;
x1345 += 1;
x1346 += 1;
x1347 += 1;

}
float* x1361 = (float*)myMalloc(150 * sizeof(float));;
float* x1362 = (float*)myMalloc(150 * sizeof(float));;
for(int x1363=0; x1363 < 150; x1363++) {
float x1364 = x1344[x1363];
double x1365 = (double)x1364;
double x1366 = tanh(x1365);
float x1367 = (float)x1366;
x1362[x1363] = x1367;

}
float* x1371 = (float*)myMalloc(150 * sizeof(float));;
float* x1372 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1373 = 0;
int32_t x1374 = 0;
int32_t x1375 = 0;
for(int x1376=0; x1376 < 150; x1376++) {
int32_t x1377 = x1373;
int32_t x1378 = x1374;
float x1379 = x1022[x1378];
int32_t x1380 = x1375;
float x1381 = x1362[x1380];
float x1382 = x1379 * x1381;
x1372[x1377] = x1382;
x1373 += 1;
x1374 += 1;
x1375 += 1;

}
float* x1389 = (float*)myMalloc(150 * sizeof(float));;
float* x1390 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1391 = 0;
int32_t x1392 = 0;
int32_t x1393 = 0;
for(int x1394=0; x1394 < 150; x1394++) {
int32_t x1395 = x1391;
int32_t x1396 = x1392;
float x1397 = x1107[x1396];
int32_t x1398 = x1393;
float x1399 = x263[x1398];
float x1400 = x1397 * x1399;
x1390[x1395] = x1400;
x1391 += 1;
x1392 += 1;
x1393 += 1;

}
float* x1407 = (float*)myMalloc(150 * sizeof(float));;
float* x1408 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1409 = 0;
int32_t x1410 = 0;
int32_t x1411 = 0;
for(int x1412=0; x1412 < 150; x1412++) {
int32_t x1413 = x1409;
int32_t x1414 = x1410;
float x1415 = x1372[x1414];
int32_t x1416 = x1411;
float x1417 = x1390[x1416];
float x1418 = x1415 + x1417;
x1408[x1413] = x1418;
x1409 += 1;
x1410 += 1;
x1411 += 1;

}
float* x1425 = (float*)myMalloc(150 * sizeof(float));;
float* x1426 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1427 = 0;
int32_t x1428 = 0;
int32_t x1429 = 0;
for(int x1430=0; x1430 < 150; x1430++) {
int32_t x1431 = x1427;
int32_t x1432 = x1428;
float x1433 = x1192[x1432];
int32_t x1434 = x1429;
float x1435 = x272[x1434];
float x1436 = x1433 * x1435;
x1426[x1431] = x1436;
x1427 += 1;
x1428 += 1;
x1429 += 1;

}
float* x1443 = (float*)myMalloc(150 * sizeof(float));;
float* x1444 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1445 = 0;
int32_t x1446 = 0;
int32_t x1447 = 0;
for(int x1448=0; x1448 < 150; x1448++) {
int32_t x1449 = x1445;
int32_t x1450 = x1446;
float x1451 = x1408[x1450];
int32_t x1452 = x1447;
float x1453 = x1426[x1452];
float x1454 = x1451 + x1453;
x1444[x1449] = x1454;
x1445 += 1;
x1446 += 1;
x1447 += 1;

}
float* x1461 = (float*)myMalloc(150 * sizeof(float));;
float* x1462 = (float*)myMalloc(150 * sizeof(float));;
for(int x1463=0; x1463 < 150; x1463++) {
float x1464 = x1444[x1463];
double x1465 = (double)x1464;
double x1466 = tanh(x1465);
float x1467 = (float)x1466;
x1462[x1463] = x1467;

}
float* x1471 = (float*)myMalloc(150 * sizeof(float));;
float* x1472 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1473 = 0;
int32_t x1474 = 0;
int32_t x1475 = 0;
for(int x1476=0; x1476 < 150; x1476++) {
int32_t x1477 = x1473;
int32_t x1478 = x1474;
float x1479 = x1277[x1478];
int32_t x1480 = x1475;
float x1481 = x1462[x1480];
float x1482 = x1479 * x1481;
x1472[x1477] = x1482;
x1473 += 1;
x1474 += 1;
x1475 += 1;

}
float* x1489 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(5, 150), List(150)
float* x1491 = (float*)myMalloc(5 * sizeof(float));;
for(int x1492=0; x1492 < 5; x1492++) {
float x1493 = 0.0f;
int32_t x1495 = x1492 * 150;
for(int x1494=0; x1494 < 150; x1494++) {
int32_t x1496 = x1495 + x1494;
float x1497 = x163[x1496];
float x1498 = x1472[x1494];
float x1499 = x1497 * x1498;
x1493 += x1499;

}
float x1503 = x1493;
x1491[x1492] = x1503;

}
float* x1507 = (float*)myMalloc(5 * sizeof(float));;
float* x1508 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1509 = 0;
int32_t x1510 = 0;
int32_t x1511 = 0;
for(int x1512=0; x1512 < 5; x1512++) {
int32_t x1513 = x1509;
int32_t x1514 = x1510;
float x1515 = x1491[x1514];
int32_t x1516 = x1511;
float x1517 = x172[x1516];
float x1518 = x1515 + x1517;
x1508[x1513] = x1518;
x1509 += 1;
x1510 += 1;
x1511 += 1;

}
float* x1525 = (float*)myMalloc(5 * sizeof(float));;
float* x1526 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1527 = 0;
int32_t x1528 = 0;
int32_t x1529 = 0;
int32_t x1530 = x1527;
int32_t x1531 = x1528;
float x1532 = x259[x1531];
int32_t x1533 = x1529;
float x1534 = x268[x1533];
float x1535 = x1532 + x1534;
x1526[x1530] = x1535;
x1527 += 1;
float* x1538 = (float*)myMalloc(1 * sizeof(float));;
float x1539 = -3.4028235E38f;
for(int x1540=0; x1540 < 5; x1540++) {
float x1541 = x1539;
float x1542 = x1508[x1540];
bool x1543 = x1542 > x1541;
float x1544;
if (x1543) {
x1544 = x1542;
} else {
x1544 = x1541;
}
x1539 = x1544;

}
float x1548 = x1539;
float x1549 = 0.0f;
for(int x1550=0; x1550 < 5; x1550++) {
float x1551 = x1549;
float x1552 = x1508[x1550];
float x1553 = x1539;
float x1554 = x1552 - x1553;
double x1555 = (double)x1554;
double x1556 = exp(x1555);
float x1557 = (float)x1556;
float x1558 = x1551 + x1557;
x1549 = x1558;

}
float x1562 = x1549;
float* x1567 = (float*)myMalloc(5 * sizeof(float));;
double x1563 = (double)x1562;
double x1564 = log(x1563);
float x1565 = (float)x1564;
float x1566 = x1548 + x1565;
for(int x1568=0; x1568 < 5; x1568++) {
float x1569 = x1508[x1568];
float x1570 = x1569 - x1566;
x1567[x1568] = x1570;

}
float* x1574 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1575 = x226[x246];
float x1576 = x1567[x1575];
float* x1578 = (float*)myMalloc(1 * sizeof(float));;
float x1577 = -1.0f * x1576;
x1578[0] = x1577;
float* x1580 = (float*)myMalloc(1 * sizeof(float));;
float* x1581 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1582 = 0;
int32_t x1583 = 0;
int32_t x1584 = 0;
int32_t x1585 = x1582;
int32_t x1586 = x1583;
float x1587 = x1526[x1586];
int32_t x1588 = x1584;
float x1589 = x1578[x1588];
float x1590 = x1587 + x1589;
x1581[x1585] = x1590;
x1582 += 1;
float* x1593 = (float*)myMalloc(1 * sizeof(float));;
float** x1628 = (float**)myMalloc(6 * sizeof(float*));;
x1628[0] = x1581;
x1628[1] = x1593;
x1628[2] = x1472;
x1628[3] = x1489;
x1628[4] = x1444;
x1628[5] = x1461;
x1594(x1628);
int32_t x1636 = 0;
int32_t x1637 = 0;
int32_t x1638 = 0;
int32_t x1639 = x1636;
float x1640 = x1538[x1639];
float x1641 = x1526[x1639];
int32_t x1642 = x1637;
float x1643 = x1578[x1642];
int32_t x1644 = x1638;
float x1645 = x1593[x1644];
float x1646 = x1640 + x1645;
x1538[x1639] = x1646;
float x1648 = x1580[x1642];
float x1649 = x1526[x1639];
float x1650 = x1578[x1642];
float x1651 = x1593[x1644];
float x1652 = x1648 + x1651;
x1580[x1642] = x1652;
x1638 += 1;
float x1655 = x1574[x1575];
float x1656 = x1580[0];
float x1657 = -1.0f * x1656;
float x1658 = x1655 + x1657;
x1574[x1575] = x1658;
float x1660 = 0.0f;
for(int x1661=0; x1661 < 5; x1661++) {
float x1662 = x1660;
float x1663 = x1574[x1661];
float x1664 = x1662 + x1663;
x1660 = x1664;

}
float x1668 = x1660;
float* x1669 = (float*)myMalloc(1 * sizeof(float));;
x1669[0] = x1668;
float x1671 = x1669[0];
for(int x1672=0; x1672 < 5; x1672++) {
float x1673 = x1525[x1672];
float x1674 = x1574[x1672];
float x1675 = x1567[x1672];
double x1676 = (double)x1675;
double x1677 = exp(x1676);
float x1678 = (float)x1677;
float x1679 = x1678 * x1671;
float x1680 = x1674 - x1679;
float x1681 = x1673 + x1680;
x1525[x1672] = x1681;

}
int32_t x1685 = 0;
int32_t x1686 = 0;
int32_t x1687 = 0;
int32_t x1688 = x1685;
float x1689 = x260[x1688];
float x1690 = x259[x1688];
int32_t x1691 = x1686;
float x1692 = x268[x1691];
int32_t x1693 = x1687;
float x1694 = x1538[x1693];
float x1695 = x1689 + x1694;
x260[x1688] = x1695;
float x1697 = x269[x1691];
float x1698 = x259[x1688];
float x1699 = x268[x1691];
float x1700 = x1538[x1693];
float x1701 = x1697 + x1700;
x269[x1691] = x1701;
x1687 += 1;
int32_t x1704 = 0;
int32_t x1705 = 0;
int32_t x1706 = 0;
for(int x1707=0; x1707 < 5; x1707++) {
int32_t x1708 = x1704;
float x1709 = x1507[x1708];
float x1710 = x1491[x1708];
int32_t x1711 = x1705;
float x1712 = x172[x1711];
int32_t x1713 = x1706;
float x1714 = x1525[x1713];
float x1715 = x1709 + x1714;
x1507[x1708] = x1715;
float x1717 = x194[x1711];
float x1718 = x1491[x1708];
float x1719 = x172[x1711];
float x1720 = x1525[x1713];
float x1721 = x1717 + x1720;
x194[x1711] = x1721;
x1706 += 1;
x1704 += 1;
x1705 += 1;

}
// add_cartesian
int32_t x1729 = 0;
for(int x1730=0; x1730 < 5; x1730++) {
for(int x1731=0; x1731 < 150; x1731++) {
int32_t x1732 = x1729;
int32_t x1733 = x1732 + x1731;
float x1734 = x193[x1733];
float x1735 = x1472[x1731];
float x1736 = x1507[x1730];
float x1737 = x1735 * x1736;
float x1738 = x1734 + x1737;
x193[x1733] = x1738;

}
x1729 += 150;

}
int32_t x1745 = 0;
for(int x1746=0; x1746 < 5; x1746++) {
for(int x1747=0; x1747 < 150; x1747++) {
float x1748 = x1489[x1747];
int32_t x1749 = x1745;
int32_t x1750 = x1749 + x1747;
float x1751 = x163[x1750];
float x1752 = x1507[x1746];
float x1753 = x1751 * x1752;
float x1754 = x1748 + x1753;
x1489[x1747] = x1754;

}
x1745 += 150;

}
int32_t x1761 = 0;
int32_t x1762 = 0;
int32_t x1763 = 0;
for(int x1764=0; x1764 < 150; x1764++) {
int32_t x1765 = x1761;
float x1766 = x1289[x1765];
float x1767 = x1277[x1765];
int32_t x1768 = x1762;
float x1769 = x1462[x1768];
int32_t x1770 = x1763;
float x1771 = x1489[x1770];
float x1772 = x1771 * x1769;
float x1773 = x1766 + x1772;
x1289[x1765] = x1773;
float x1775 = x1471[x1768];
float x1776 = x1277[x1765];
float x1777 = x1462[x1768];
float x1778 = x1489[x1770];
float x1779 = x1778 * x1776;
float x1780 = x1775 + x1779;
x1471[x1768] = x1780;
x1763 += 1;
x1761 += 1;
x1762 += 1;

}
for(int x1787=0; x1787 < 150; x1787++) {
float x1788 = x1461[x1787];
float x1789 = x1462[x1787];
float x1792 = x1471[x1787];
float x1790 = x1789 * x1789;
float x1791 = 1.0f - x1790;
float x1793 = x1791 * x1792;
float x1794 = x1788 + x1793;
x1461[x1787] = x1794;

}
int32_t x1798 = 0;
int32_t x1799 = 0;
int32_t x1800 = 0;
for(int x1801=0; x1801 < 150; x1801++) {
int32_t x1802 = x1798;
float x1803 = x1425[x1802];
float x1804 = x1408[x1802];
int32_t x1805 = x1799;
float x1806 = x1426[x1805];
int32_t x1807 = x1800;
float x1808 = x1461[x1807];
float x1809 = x1803 + x1808;
x1425[x1802] = x1809;
float x1811 = x1443[x1805];
float x1812 = x1408[x1802];
float x1813 = x1426[x1805];
float x1814 = x1461[x1807];
float x1815 = x1811 + x1814;
x1443[x1805] = x1815;
x1800 += 1;
x1798 += 1;
x1799 += 1;

}
int32_t x1822 = 0;
int32_t x1823 = 0;
int32_t x1824 = 0;
for(int x1825=0; x1825 < 150; x1825++) {
int32_t x1826 = x1822;
float x1827 = x1204[x1826];
float x1828 = x1192[x1826];
int32_t x1829 = x1823;
float x1830 = x272[x1829];
int32_t x1831 = x1824;
float x1832 = x1443[x1831];
float x1833 = x1832 * x1830;
float x1834 = x1827 + x1833;
x1204[x1826] = x1834;
float x1836 = x273[x1829];
float x1837 = x1192[x1826];
float x1838 = x272[x1829];
float x1839 = x1443[x1831];
float x1840 = x1839 * x1837;
float x1841 = x1836 + x1840;
x273[x1829] = x1841;
x1824 += 1;
x1822 += 1;
x1823 += 1;

}
int32_t x1848 = 0;
int32_t x1849 = 0;
int32_t x1850 = 0;
for(int x1851=0; x1851 < 150; x1851++) {
int32_t x1852 = x1848;
float x1853 = x1389[x1852];
float x1854 = x1372[x1852];
int32_t x1855 = x1849;
float x1856 = x1390[x1855];
int32_t x1857 = x1850;
float x1858 = x1425[x1857];
float x1859 = x1853 + x1858;
x1389[x1852] = x1859;
float x1861 = x1407[x1855];
float x1862 = x1372[x1852];
float x1863 = x1390[x1855];
float x1864 = x1425[x1857];
float x1865 = x1861 + x1864;
x1407[x1855] = x1865;
x1850 += 1;
x1848 += 1;
x1849 += 1;

}
int32_t x1872 = 0;
int32_t x1873 = 0;
int32_t x1874 = 0;
for(int x1875=0; x1875 < 150; x1875++) {
int32_t x1876 = x1872;
float x1877 = x1119[x1876];
float x1878 = x1107[x1876];
int32_t x1879 = x1873;
float x1880 = x263[x1879];
int32_t x1881 = x1874;
float x1882 = x1407[x1881];
float x1883 = x1882 * x1880;
float x1884 = x1877 + x1883;
x1119[x1876] = x1884;
float x1886 = x264[x1879];
float x1887 = x1107[x1876];
float x1888 = x263[x1879];
float x1889 = x1407[x1881];
float x1890 = x1889 * x1887;
float x1891 = x1886 + x1890;
x264[x1879] = x1891;
x1874 += 1;
x1872 += 1;
x1873 += 1;

}
int32_t x1898 = 0;
int32_t x1899 = 0;
int32_t x1900 = 0;
for(int x1901=0; x1901 < 150; x1901++) {
int32_t x1902 = x1898;
float x1903 = x1034[x1902];
float x1904 = x1022[x1902];
int32_t x1905 = x1899;
float x1906 = x1362[x1905];
int32_t x1907 = x1900;
float x1908 = x1389[x1907];
float x1909 = x1908 * x1906;
float x1910 = x1903 + x1909;
x1034[x1902] = x1910;
float x1912 = x1371[x1905];
float x1913 = x1022[x1902];
float x1914 = x1362[x1905];
float x1915 = x1389[x1907];
float x1916 = x1915 * x1913;
float x1917 = x1912 + x1916;
x1371[x1905] = x1917;
x1900 += 1;
x1898 += 1;
x1899 += 1;

}
for(int x1924=0; x1924 < 150; x1924++) {
float x1925 = x1361[x1924];
float x1926 = x1362[x1924];
float x1929 = x1371[x1924];
float x1927 = x1926 * x1926;
float x1928 = 1.0f - x1927;
float x1930 = x1928 * x1929;
float x1931 = x1925 + x1930;
x1361[x1924] = x1931;

}
int32_t x1935 = 0;
int32_t x1936 = 0;
int32_t x1937 = 0;
for(int x1938=0; x1938 < 150; x1938++) {
int32_t x1939 = x1935;
float x1940 = x1343[x1939];
float x1941 = x1326[x1939];
int32_t x1942 = x1936;
float x1943 = x162[x1942];
int32_t x1944 = x1937;
float x1945 = x1361[x1944];
float x1946 = x1940 + x1945;
x1343[x1939] = x1946;
float x1948 = x192[x1942];
float x1949 = x1326[x1939];
float x1950 = x162[x1942];
float x1951 = x1361[x1944];
float x1952 = x1948 + x1951;
x192[x1942] = x1952;
x1937 += 1;
x1935 += 1;
x1936 += 1;

}
int32_t x1959 = 0;
int32_t x1960 = 0;
int32_t x1961 = 0;
for(int x1962=0; x1962 < 150; x1962++) {
int32_t x1963 = x1959;
float x1964 = x1307[x1963];
float x1965 = x1291[x1963];
int32_t x1966 = x1960;
float x1967 = x1309[x1966];
int32_t x1968 = x1961;
float x1969 = x1343[x1968];
float x1970 = x1964 + x1969;
x1307[x1963] = x1970;
float x1972 = x1325[x1966];
float x1973 = x1291[x1963];
float x1974 = x1309[x1966];
float x1975 = x1343[x1968];
float x1976 = x1972 + x1975;
x1325[x1966] = x1976;
x1961 += 1;
x1959 += 1;
x1960 += 1;

}
// add_cartesian
int32_t x1984 = 0;
for(int x1985=0; x1985 < 150; x1985++) {
for(int x1986=0; x1986 < 150; x1986++) {
int32_t x1987 = x1984;
int32_t x1988 = x1987 + x1986;
float x1989 = x191[x1988];
float x1990 = x270[x1986];
float x1991 = x1325[x1985];
float x1992 = x1990 * x1991;
float x1993 = x1989 + x1992;
x191[x1988] = x1993;

}
x1984 += 150;

}
int32_t x2000 = 0;
for(int x2001=0; x2001 < 150; x2001++) {
for(int x2002=0; x2002 < 150; x2002++) {
float x2003 = x271[x2002];
int32_t x2004 = x2000;
int32_t x2005 = x2004 + x2002;
float x2006 = x154[x2005];
float x2007 = x1325[x2001];
float x2008 = x2006 * x2007;
float x2009 = x2003 + x2008;
x271[x2002] = x2009;

}
x2000 += 150;

}
// add_cartesian
int32_t x2017 = 0;
for(int x2018=0; x2018 < 150; x2018++) {
for(int x2019=0; x2019 < 150; x2019++) {
int32_t x2020 = x2017;
int32_t x2021 = x2020 + x2019;
float x2022 = x190[x2021];
float x2023 = x261[x2019];
float x2024 = x1307[x2018];
float x2025 = x2023 * x2024;
float x2026 = x2022 + x2025;
x190[x2021] = x2026;

}
x2017 += 150;

}
int32_t x2033 = 0;
for(int x2034=0; x2034 < 150; x2034++) {
for(int x2035=0; x2035 < 150; x2035++) {
float x2036 = x262[x2035];
int32_t x2037 = x2033;
int32_t x2038 = x2037 + x2035;
float x2039 = x146[x2038];
float x2040 = x1307[x2034];
float x2041 = x2039 * x2040;
float x2042 = x2036 + x2041;
x262[x2035] = x2042;

}
x2033 += 150;

}
for(int x2049=0; x2049 < 150; x2049++) {
float x2050 = x1276[x2049];
float x2051 = x1277[x2049];
float x2054 = x1289[x2049];
float x2052 = 1.0f - x2051;
float x2053 = x2052 * x2051;
float x2055 = x2053 * x2054;
float x2056 = x2050 + x2055;
x1276[x2049] = x2056;

}
int32_t x2060 = 0;
int32_t x2061 = 0;
int32_t x2062 = 0;
for(int x2063=0; x2063 < 150; x2063++) {
int32_t x2064 = x2060;
float x2065 = x1258[x2064];
float x2066 = x1241[x2064];
int32_t x2067 = x2061;
float x2068 = x145[x2067];
int32_t x2069 = x2062;
float x2070 = x1276[x2069];
float x2071 = x2065 + x2070;
x1258[x2064] = x2071;
float x2073 = x189[x2067];
float x2074 = x1241[x2064];
float x2075 = x145[x2067];
float x2076 = x1276[x2069];
float x2077 = x2073 + x2076;
x189[x2067] = x2077;
x2062 += 1;
x2060 += 1;
x2061 += 1;

}
int32_t x2084 = 0;
int32_t x2085 = 0;
int32_t x2086 = 0;
for(int x2087=0; x2087 < 150; x2087++) {
int32_t x2088 = x2084;
float x2089 = x1222[x2088];
float x2090 = x1206[x2088];
int32_t x2091 = x2085;
float x2092 = x1224[x2091];
int32_t x2093 = x2086;
float x2094 = x1258[x2093];
float x2095 = x2089 + x2094;
x1222[x2088] = x2095;
float x2097 = x1240[x2091];
float x2098 = x1206[x2088];
float x2099 = x1224[x2091];
float x2100 = x1258[x2093];
float x2101 = x2097 + x2100;
x1240[x2091] = x2101;
x2086 += 1;
x2084 += 1;
x2085 += 1;

}
// add_cartesian
int32_t x2109 = 0;
for(int x2110=0; x2110 < 150; x2110++) {
for(int x2111=0; x2111 < 150; x2111++) {
int32_t x2112 = x2109;
int32_t x2113 = x2112 + x2111;
float x2114 = x188[x2113];
float x2115 = x270[x2111];
float x2116 = x1240[x2110];
float x2117 = x2115 * x2116;
float x2118 = x2114 + x2117;
x188[x2113] = x2118;

}
x2109 += 150;

}
int32_t x2125 = 0;
for(int x2126=0; x2126 < 150; x2126++) {
for(int x2127=0; x2127 < 150; x2127++) {
float x2128 = x271[x2127];
int32_t x2129 = x2125;
int32_t x2130 = x2129 + x2127;
float x2131 = x137[x2130];
float x2132 = x1240[x2126];
float x2133 = x2131 * x2132;
float x2134 = x2128 + x2133;
x271[x2127] = x2134;

}
x2125 += 150;

}
// add_cartesian
int32_t x2142 = 0;
for(int x2143=0; x2143 < 150; x2143++) {
for(int x2144=0; x2144 < 150; x2144++) {
int32_t x2145 = x2142;
int32_t x2146 = x2145 + x2144;
float x2147 = x187[x2146];
float x2148 = x261[x2144];
float x2149 = x1222[x2143];
float x2150 = x2148 * x2149;
float x2151 = x2147 + x2150;
x187[x2146] = x2151;

}
x2142 += 150;

}
int32_t x2158 = 0;
for(int x2159=0; x2159 < 150; x2159++) {
for(int x2160=0; x2160 < 150; x2160++) {
float x2161 = x262[x2160];
int32_t x2162 = x2158;
int32_t x2163 = x2162 + x2160;
float x2164 = x129[x2163];
float x2165 = x1222[x2159];
float x2166 = x2164 * x2165;
float x2167 = x2161 + x2166;
x262[x2160] = x2167;

}
x2158 += 150;

}
for(int x2174=0; x2174 < 150; x2174++) {
float x2175 = x1191[x2174];
float x2176 = x1192[x2174];
float x2179 = x1204[x2174];
float x2177 = 1.0f - x2176;
float x2178 = x2177 * x2176;
float x2180 = x2178 * x2179;
float x2181 = x2175 + x2180;
x1191[x2174] = x2181;

}
int32_t x2185 = 0;
int32_t x2186 = 0;
int32_t x2187 = 0;
for(int x2188=0; x2188 < 150; x2188++) {
int32_t x2189 = x2185;
float x2190 = x1173[x2189];
float x2191 = x1156[x2189];
int32_t x2192 = x2186;
float x2193 = x128[x2192];
int32_t x2194 = x2187;
float x2195 = x1191[x2194];
float x2196 = x2190 + x2195;
x1173[x2189] = x2196;
float x2198 = x186[x2192];
float x2199 = x1156[x2189];
float x2200 = x128[x2192];
float x2201 = x1191[x2194];
float x2202 = x2198 + x2201;
x186[x2192] = x2202;
x2187 += 1;
x2185 += 1;
x2186 += 1;

}
int32_t x2209 = 0;
int32_t x2210 = 0;
int32_t x2211 = 0;
for(int x2212=0; x2212 < 150; x2212++) {
int32_t x2213 = x2209;
float x2214 = x1137[x2213];
float x2215 = x1121[x2213];
int32_t x2216 = x2210;
float x2217 = x1139[x2216];
int32_t x2218 = x2211;
float x2219 = x1173[x2218];
float x2220 = x2214 + x2219;
x1137[x2213] = x2220;
float x2222 = x1155[x2216];
float x2223 = x1121[x2213];
float x2224 = x1139[x2216];
float x2225 = x1173[x2218];
float x2226 = x2222 + x2225;
x1155[x2216] = x2226;
x2211 += 1;
x2209 += 1;
x2210 += 1;

}
// add_cartesian
int32_t x2234 = 0;
for(int x2235=0; x2235 < 150; x2235++) {
for(int x2236=0; x2236 < 150; x2236++) {
int32_t x2237 = x2234;
int32_t x2238 = x2237 + x2236;
float x2239 = x185[x2238];
float x2240 = x270[x2236];
float x2241 = x1155[x2235];
float x2242 = x2240 * x2241;
float x2243 = x2239 + x2242;
x185[x2238] = x2243;

}
x2234 += 150;

}
int32_t x2250 = 0;
for(int x2251=0; x2251 < 150; x2251++) {
for(int x2252=0; x2252 < 150; x2252++) {
float x2253 = x271[x2252];
int32_t x2254 = x2250;
int32_t x2255 = x2254 + x2252;
float x2256 = x120[x2255];
float x2257 = x1155[x2251];
float x2258 = x2256 * x2257;
float x2259 = x2253 + x2258;
x271[x2252] = x2259;

}
x2250 += 150;

}
// add_cartesian
int32_t x2267 = 0;
for(int x2268=0; x2268 < 150; x2268++) {
for(int x2269=0; x2269 < 150; x2269++) {
int32_t x2270 = x2267;
int32_t x2271 = x2270 + x2269;
float x2272 = x184[x2271];
float x2273 = x261[x2269];
float x2274 = x1137[x2268];
float x2275 = x2273 * x2274;
float x2276 = x2272 + x2275;
x184[x2271] = x2276;

}
x2267 += 150;

}
int32_t x2283 = 0;
for(int x2284=0; x2284 < 150; x2284++) {
for(int x2285=0; x2285 < 150; x2285++) {
float x2286 = x262[x2285];
int32_t x2287 = x2283;
int32_t x2288 = x2287 + x2285;
float x2289 = x112[x2288];
float x2290 = x1137[x2284];
float x2291 = x2289 * x2290;
float x2292 = x2286 + x2291;
x262[x2285] = x2292;

}
x2283 += 150;

}
for(int x2299=0; x2299 < 150; x2299++) {
float x2300 = x1106[x2299];
float x2301 = x1107[x2299];
float x2304 = x1119[x2299];
float x2302 = 1.0f - x2301;
float x2303 = x2302 * x2301;
float x2305 = x2303 * x2304;
float x2306 = x2300 + x2305;
x1106[x2299] = x2306;

}
int32_t x2310 = 0;
int32_t x2311 = 0;
int32_t x2312 = 0;
for(int x2313=0; x2313 < 150; x2313++) {
int32_t x2314 = x2310;
float x2315 = x1088[x2314];
float x2316 = x1071[x2314];
int32_t x2317 = x2311;
float x2318 = x128[x2317];
int32_t x2319 = x2312;
float x2320 = x1106[x2319];
float x2321 = x2315 + x2320;
x1088[x2314] = x2321;
float x2323 = x186[x2317];
float x2324 = x1071[x2314];
float x2325 = x128[x2317];
float x2326 = x1106[x2319];
float x2327 = x2323 + x2326;
x186[x2317] = x2327;
x2312 += 1;
x2310 += 1;
x2311 += 1;

}
int32_t x2334 = 0;
int32_t x2335 = 0;
int32_t x2336 = 0;
for(int x2337=0; x2337 < 150; x2337++) {
int32_t x2338 = x2334;
float x2339 = x1052[x2338];
float x2340 = x1036[x2338];
int32_t x2341 = x2335;
float x2342 = x1054[x2341];
int32_t x2343 = x2336;
float x2344 = x1088[x2343];
float x2345 = x2339 + x2344;
x1052[x2338] = x2345;
float x2347 = x1070[x2341];
float x2348 = x1036[x2338];
float x2349 = x1054[x2341];
float x2350 = x1088[x2343];
float x2351 = x2347 + x2350;
x1070[x2341] = x2351;
x2336 += 1;
x2334 += 1;
x2335 += 1;

}
// add_cartesian
int32_t x2359 = 0;
for(int x2360=0; x2360 < 150; x2360++) {
for(int x2361=0; x2361 < 150; x2361++) {
int32_t x2362 = x2359;
int32_t x2363 = x2362 + x2361;
float x2364 = x183[x2363];
float x2365 = x270[x2361];
float x2366 = x1070[x2360];
float x2367 = x2365 * x2366;
float x2368 = x2364 + x2367;
x183[x2363] = x2368;

}
x2359 += 150;

}
int32_t x2375 = 0;
for(int x2376=0; x2376 < 150; x2376++) {
for(int x2377=0; x2377 < 150; x2377++) {
float x2378 = x271[x2377];
int32_t x2379 = x2375;
int32_t x2380 = x2379 + x2377;
float x2381 = x104[x2380];
float x2382 = x1070[x2376];
float x2383 = x2381 * x2382;
float x2384 = x2378 + x2383;
x271[x2377] = x2384;

}
x2375 += 150;

}
// add_cartesian
int32_t x2392 = 0;
for(int x2393=0; x2393 < 150; x2393++) {
for(int x2394=0; x2394 < 150; x2394++) {
int32_t x2395 = x2392;
int32_t x2396 = x2395 + x2394;
float x2397 = x182[x2396];
float x2398 = x261[x2394];
float x2399 = x1052[x2393];
float x2400 = x2398 * x2399;
float x2401 = x2397 + x2400;
x182[x2396] = x2401;

}
x2392 += 150;

}
int32_t x2408 = 0;
for(int x2409=0; x2409 < 150; x2409++) {
for(int x2410=0; x2410 < 150; x2410++) {
float x2411 = x262[x2410];
int32_t x2412 = x2408;
int32_t x2413 = x2412 + x2410;
float x2414 = x96[x2413];
float x2415 = x1052[x2409];
float x2416 = x2414 * x2415;
float x2417 = x2411 + x2416;
x262[x2410] = x2417;

}
x2408 += 150;

}
for(int x2424=0; x2424 < 150; x2424++) {
float x2425 = x1021[x2424];
float x2426 = x1022[x2424];
float x2429 = x1034[x2424];
float x2427 = 1.0f - x2426;
float x2428 = x2427 * x2426;
float x2430 = x2428 * x2429;
float x2431 = x2425 + x2430;
x1021[x2424] = x2431;

}
int32_t x2435 = 0;
int32_t x2436 = 0;
int32_t x2437 = 0;
for(int x2438=0; x2438 < 150; x2438++) {
int32_t x2439 = x2435;
float x2440 = x1003[x2439];
float x2441 = x986[x2439];
int32_t x2442 = x2436;
float x2443 = x95[x2442];
int32_t x2444 = x2437;
float x2445 = x1021[x2444];
float x2446 = x2440 + x2445;
x1003[x2439] = x2446;
float x2448 = x181[x2442];
float x2449 = x986[x2439];
float x2450 = x95[x2442];
float x2451 = x1021[x2444];
float x2452 = x2448 + x2451;
x181[x2442] = x2452;
x2437 += 1;
x2435 += 1;
x2436 += 1;

}
int32_t x2459 = 0;
int32_t x2460 = 0;
int32_t x2461 = 0;
for(int x2462=0; x2462 < 150; x2462++) {
int32_t x2463 = x2459;
float x2464 = x967[x2463];
float x2465 = x951[x2463];
int32_t x2466 = x2460;
float x2467 = x969[x2466];
int32_t x2468 = x2461;
float x2469 = x1003[x2468];
float x2470 = x2464 + x2469;
x967[x2463] = x2470;
float x2472 = x985[x2466];
float x2473 = x951[x2463];
float x2474 = x969[x2466];
float x2475 = x1003[x2468];
float x2476 = x2472 + x2475;
x985[x2466] = x2476;
x2461 += 1;
x2459 += 1;
x2460 += 1;

}
// add_cartesian
int32_t x2484 = 0;
for(int x2485=0; x2485 < 150; x2485++) {
for(int x2486=0; x2486 < 150; x2486++) {
int32_t x2487 = x2484;
int32_t x2488 = x2487 + x2486;
float x2489 = x180[x2488];
float x2490 = x270[x2486];
float x2491 = x985[x2485];
float x2492 = x2490 * x2491;
float x2493 = x2489 + x2492;
x180[x2488] = x2493;

}
x2484 += 150;

}
int32_t x2500 = 0;
for(int x2501=0; x2501 < 150; x2501++) {
for(int x2502=0; x2502 < 150; x2502++) {
float x2503 = x271[x2502];
int32_t x2504 = x2500;
int32_t x2505 = x2504 + x2502;
float x2506 = x87[x2505];
float x2507 = x985[x2501];
float x2508 = x2506 * x2507;
float x2509 = x2503 + x2508;
x271[x2502] = x2509;

}
x2500 += 150;

}
// add_cartesian
int32_t x2517 = 0;
for(int x2518=0; x2518 < 150; x2518++) {
for(int x2519=0; x2519 < 150; x2519++) {
int32_t x2520 = x2517;
int32_t x2521 = x2520 + x2519;
float x2522 = x179[x2521];
float x2523 = x261[x2519];
float x2524 = x967[x2518];
float x2525 = x2523 * x2524;
float x2526 = x2522 + x2525;
x179[x2521] = x2526;

}
x2517 += 150;

}
int32_t x2533 = 0;
for(int x2534=0; x2534 < 150; x2534++) {
for(int x2535=0; x2535 < 150; x2535++) {
float x2536 = x262[x2535];
int32_t x2537 = x2533;
int32_t x2538 = x2537 + x2535;
float x2539 = x78[x2538];
float x2540 = x967[x2534];
float x2541 = x2539 * x2540;
float x2542 = x2536 + x2541;
x262[x2535] = x2542;

}
x2533 += 150;

}
}
};
x242(x265,x266,x2552);
};
x242(x256,x257,x2562);
} else {
float** x2589 = (float**)myMalloc(6 * sizeof(float*));;
x2589[0] = x236;
x2589[1] = x237;
x2589[2] = x238;
x2589[3] = x239;
x2589[4] = x240;
x2589[5] = x241;
function<void(float**)> x247 = x244;
function<void(float**)> x2572 = [&](float** x2573) {
float* x2574 = x2573[0];
float* x2575 = x2573[1];
float* x2576 = x2573[2];
float* x2577 = x2573[3];
float* x2578 = x2573[4];
float* x2579 = x2573[5];
float** x2580 = (float**)myMalloc(6 * sizeof(float*));;
x2580[0] = x2574;
x2580[1] = x2575;
x2580[2] = x2576;
x2580[3] = x2577;
x2580[4] = x2578;
x2580[5] = x2579;
x247(x2580);
};
x2572(x2589);
}
};
float* x233 = (float*)myMalloc(1 * sizeof(float));;
float* x234 = (float*)myMalloc(1 * sizeof(float));;
float* x235 = (float*)myMalloc(1 * sizeof(float));;
float** x2613 = (float**)myMalloc(6 * sizeof(float*));;
x2613[0] = x236;
x2613[1] = x237;
x2613[2] = x238;
x2613[3] = x239;
x2613[4] = x240;
x2613[5] = x241;
function<void(float**)> x2600 = [&](float** x2601) {
float* x2602 = x2601[0];
float* x2603 = x2601[1];
float* x2604 = x2601[2];
float* x2605 = x2601[3];
float* x2606 = x2601[4];
float* x2607 = x2601[5];
float x2608 = x2603[0];
x2603[0] = 1.0f;
float x2610 = x2602[0];
x235[0] = x2610;
};
x242(0,x2600,x2613);
float x2622 = x235[0];
float x2623 = x222;
float x2624 = (float)x223;
float x2625 = x2623 * x2624;
int32_t x2626 = x223 + 1;
float x2627 = (float)x2626;
float x2628 = x2625 / x2627;
float x2629 = x2622 / x2627;
float x2630 = x2628 + x2629;
x222 = x2630;
for(int x2632=0; x2632 < 45000; x2632++) {
float x2633 = x173[x2632];
float x2634 = x2633;
float x2635 = x195[x2632];
float x2636 = x2634;
float x2637 = x2636 * x2636;
float x2638 = x2635 + x2637;
x195[x2632] = x2638;
float x2640 = x50[x2632];
float x2642 = x195[x2632];
float x2641 = 0.05f * x2636;
double x2643 = (double)x2642;
double x2644 = x2643 + 9.99999993922529E-9;
double x2645 = sqrt(x2644);
float x2646 = (float)x2645;
float x2647 = x2641 / x2646;
float x2648 = x2640 - x2647;
x50[x2632] = x2648;
x173[x2632] = 0.0f;

}
for(int x2653=0; x2653 < 150; x2653++) {
float x2654 = x174[x2653];
float x2655 = x2654;
float x2656 = x196[x2653];
float x2657 = x2655;
float x2658 = x2657 * x2657;
float x2659 = x2656 + x2658;
x196[x2653] = x2659;
float x2661 = x59[x2653];
float x2663 = x196[x2653];
float x2662 = 0.05f * x2657;
double x2664 = (double)x2663;
double x2665 = x2664 + 9.99999993922529E-9;
double x2666 = sqrt(x2665);
float x2667 = (float)x2666;
float x2668 = x2662 / x2667;
float x2669 = x2661 - x2668;
x59[x2653] = x2669;
x174[x2653] = 0.0f;

}
for(int x2674=0; x2674 < 45000; x2674++) {
float x2675 = x175[x2674];
float x2676 = x2675;
float x2677 = x197[x2674];
float x2678 = x2676;
float x2679 = x2678 * x2678;
float x2680 = x2677 + x2679;
x197[x2674] = x2680;
float x2682 = x60[x2674];
float x2684 = x197[x2674];
float x2683 = 0.05f * x2678;
double x2685 = (double)x2684;
double x2686 = x2685 + 9.99999993922529E-9;
double x2687 = sqrt(x2686);
float x2688 = (float)x2687;
float x2689 = x2683 / x2688;
float x2690 = x2682 - x2689;
x60[x2674] = x2690;
x175[x2674] = 0.0f;

}
for(int x2695=0; x2695 < 150; x2695++) {
float x2696 = x176[x2695];
float x2697 = x2696;
float x2698 = x198[x2695];
float x2699 = x2697;
float x2700 = x2699 * x2699;
float x2701 = x2698 + x2700;
x198[x2695] = x2701;
float x2703 = x68[x2695];
float x2705 = x198[x2695];
float x2704 = 0.05f * x2699;
double x2706 = (double)x2705;
double x2707 = x2706 + 9.99999993922529E-9;
double x2708 = sqrt(x2707);
float x2709 = (float)x2708;
float x2710 = x2704 / x2709;
float x2711 = x2703 - x2710;
x68[x2695] = x2711;
x176[x2695] = 0.0f;

}
for(int x2716=0; x2716 < 45000; x2716++) {
float x2717 = x177[x2716];
float x2718 = x2717;
float x2719 = x199[x2716];
float x2720 = x2718;
float x2721 = x2720 * x2720;
float x2722 = x2719 + x2721;
x199[x2716] = x2722;
float x2724 = x69[x2716];
float x2726 = x199[x2716];
float x2725 = 0.05f * x2720;
double x2727 = (double)x2726;
double x2728 = x2727 + 9.99999993922529E-9;
double x2729 = sqrt(x2728);
float x2730 = (float)x2729;
float x2731 = x2725 / x2730;
float x2732 = x2724 - x2731;
x69[x2716] = x2732;
x177[x2716] = 0.0f;

}
for(int x2737=0; x2737 < 150; x2737++) {
float x2738 = x178[x2737];
float x2739 = x2738;
float x2740 = x200[x2737];
float x2741 = x2739;
float x2742 = x2741 * x2741;
float x2743 = x2740 + x2742;
x200[x2737] = x2743;
float x2745 = x77[x2737];
float x2747 = x200[x2737];
float x2746 = 0.05f * x2741;
double x2748 = (double)x2747;
double x2749 = x2748 + 9.99999993922529E-9;
double x2750 = sqrt(x2749);
float x2751 = (float)x2750;
float x2752 = x2746 / x2751;
float x2753 = x2745 - x2752;
x77[x2737] = x2753;
x178[x2737] = 0.0f;

}
for(int x2758=0; x2758 < 22500; x2758++) {
float x2759 = x179[x2758];
float x2760 = x2759;
float x2761 = x201[x2758];
float x2762 = x2760;
float x2763 = x2762 * x2762;
float x2764 = x2761 + x2763;
x201[x2758] = x2764;
float x2766 = x78[x2758];
float x2768 = x201[x2758];
float x2767 = 0.05f * x2762;
double x2769 = (double)x2768;
double x2770 = x2769 + 9.99999993922529E-9;
double x2771 = sqrt(x2770);
float x2772 = (float)x2771;
float x2773 = x2767 / x2772;
float x2774 = x2766 - x2773;
x78[x2758] = x2774;
x179[x2758] = 0.0f;

}
for(int x2779=0; x2779 < 22500; x2779++) {
float x2780 = x180[x2779];
float x2781 = x2780;
float x2782 = x202[x2779];
float x2783 = x2781;
float x2784 = x2783 * x2783;
float x2785 = x2782 + x2784;
x202[x2779] = x2785;
float x2787 = x87[x2779];
float x2789 = x202[x2779];
float x2788 = 0.05f * x2783;
double x2790 = (double)x2789;
double x2791 = x2790 + 9.99999993922529E-9;
double x2792 = sqrt(x2791);
float x2793 = (float)x2792;
float x2794 = x2788 / x2793;
float x2795 = x2787 - x2794;
x87[x2779] = x2795;
x180[x2779] = 0.0f;

}
for(int x2800=0; x2800 < 150; x2800++) {
float x2801 = x181[x2800];
float x2802 = x2801;
float x2803 = x203[x2800];
float x2804 = x2802;
float x2805 = x2804 * x2804;
float x2806 = x2803 + x2805;
x203[x2800] = x2806;
float x2808 = x95[x2800];
float x2810 = x203[x2800];
float x2809 = 0.05f * x2804;
double x2811 = (double)x2810;
double x2812 = x2811 + 9.99999993922529E-9;
double x2813 = sqrt(x2812);
float x2814 = (float)x2813;
float x2815 = x2809 / x2814;
float x2816 = x2808 - x2815;
x95[x2800] = x2816;
x181[x2800] = 0.0f;

}
for(int x2821=0; x2821 < 22500; x2821++) {
float x2822 = x182[x2821];
float x2823 = x2822;
float x2824 = x204[x2821];
float x2825 = x2823;
float x2826 = x2825 * x2825;
float x2827 = x2824 + x2826;
x204[x2821] = x2827;
float x2829 = x96[x2821];
float x2831 = x204[x2821];
float x2830 = 0.05f * x2825;
double x2832 = (double)x2831;
double x2833 = x2832 + 9.99999993922529E-9;
double x2834 = sqrt(x2833);
float x2835 = (float)x2834;
float x2836 = x2830 / x2835;
float x2837 = x2829 - x2836;
x96[x2821] = x2837;
x182[x2821] = 0.0f;

}
for(int x2842=0; x2842 < 22500; x2842++) {
float x2843 = x183[x2842];
float x2844 = x2843;
float x2845 = x205[x2842];
float x2846 = x2844;
float x2847 = x2846 * x2846;
float x2848 = x2845 + x2847;
x205[x2842] = x2848;
float x2850 = x104[x2842];
float x2852 = x205[x2842];
float x2851 = 0.05f * x2846;
double x2853 = (double)x2852;
double x2854 = x2853 + 9.99999993922529E-9;
double x2855 = sqrt(x2854);
float x2856 = (float)x2855;
float x2857 = x2851 / x2856;
float x2858 = x2850 - x2857;
x104[x2842] = x2858;
x183[x2842] = 0.0f;

}
for(int x2863=0; x2863 < 22500; x2863++) {
float x2864 = x184[x2863];
float x2865 = x2864;
float x2866 = x206[x2863];
float x2867 = x2865;
float x2868 = x2867 * x2867;
float x2869 = x2866 + x2868;
x206[x2863] = x2869;
float x2871 = x112[x2863];
float x2873 = x206[x2863];
float x2872 = 0.05f * x2867;
double x2874 = (double)x2873;
double x2875 = x2874 + 9.99999993922529E-9;
double x2876 = sqrt(x2875);
float x2877 = (float)x2876;
float x2878 = x2872 / x2877;
float x2879 = x2871 - x2878;
x112[x2863] = x2879;
x184[x2863] = 0.0f;

}
for(int x2884=0; x2884 < 22500; x2884++) {
float x2885 = x185[x2884];
float x2886 = x2885;
float x2887 = x207[x2884];
float x2888 = x2886;
float x2889 = x2888 * x2888;
float x2890 = x2887 + x2889;
x207[x2884] = x2890;
float x2892 = x120[x2884];
float x2894 = x207[x2884];
float x2893 = 0.05f * x2888;
double x2895 = (double)x2894;
double x2896 = x2895 + 9.99999993922529E-9;
double x2897 = sqrt(x2896);
float x2898 = (float)x2897;
float x2899 = x2893 / x2898;
float x2900 = x2892 - x2899;
x120[x2884] = x2900;
x185[x2884] = 0.0f;

}
for(int x2905=0; x2905 < 150; x2905++) {
float x2906 = x186[x2905];
float x2907 = x2906;
float x2908 = x208[x2905];
float x2909 = x2907;
float x2910 = x2909 * x2909;
float x2911 = x2908 + x2910;
x208[x2905] = x2911;
float x2913 = x128[x2905];
float x2915 = x208[x2905];
float x2914 = 0.05f * x2909;
double x2916 = (double)x2915;
double x2917 = x2916 + 9.99999993922529E-9;
double x2918 = sqrt(x2917);
float x2919 = (float)x2918;
float x2920 = x2914 / x2919;
float x2921 = x2913 - x2920;
x128[x2905] = x2921;
x186[x2905] = 0.0f;

}
for(int x2926=0; x2926 < 22500; x2926++) {
float x2927 = x187[x2926];
float x2928 = x2927;
float x2929 = x209[x2926];
float x2930 = x2928;
float x2931 = x2930 * x2930;
float x2932 = x2929 + x2931;
x209[x2926] = x2932;
float x2934 = x129[x2926];
float x2936 = x209[x2926];
float x2935 = 0.05f * x2930;
double x2937 = (double)x2936;
double x2938 = x2937 + 9.99999993922529E-9;
double x2939 = sqrt(x2938);
float x2940 = (float)x2939;
float x2941 = x2935 / x2940;
float x2942 = x2934 - x2941;
x129[x2926] = x2942;
x187[x2926] = 0.0f;

}
for(int x2947=0; x2947 < 22500; x2947++) {
float x2948 = x188[x2947];
float x2949 = x2948;
float x2950 = x210[x2947];
float x2951 = x2949;
float x2952 = x2951 * x2951;
float x2953 = x2950 + x2952;
x210[x2947] = x2953;
float x2955 = x137[x2947];
float x2957 = x210[x2947];
float x2956 = 0.05f * x2951;
double x2958 = (double)x2957;
double x2959 = x2958 + 9.99999993922529E-9;
double x2960 = sqrt(x2959);
float x2961 = (float)x2960;
float x2962 = x2956 / x2961;
float x2963 = x2955 - x2962;
x137[x2947] = x2963;
x188[x2947] = 0.0f;

}
for(int x2968=0; x2968 < 150; x2968++) {
float x2969 = x189[x2968];
float x2970 = x2969;
float x2971 = x211[x2968];
float x2972 = x2970;
float x2973 = x2972 * x2972;
float x2974 = x2971 + x2973;
x211[x2968] = x2974;
float x2976 = x145[x2968];
float x2978 = x211[x2968];
float x2977 = 0.05f * x2972;
double x2979 = (double)x2978;
double x2980 = x2979 + 9.99999993922529E-9;
double x2981 = sqrt(x2980);
float x2982 = (float)x2981;
float x2983 = x2977 / x2982;
float x2984 = x2976 - x2983;
x145[x2968] = x2984;
x189[x2968] = 0.0f;

}
for(int x2989=0; x2989 < 22500; x2989++) {
float x2990 = x190[x2989];
float x2991 = x2990;
float x2992 = x212[x2989];
float x2993 = x2991;
float x2994 = x2993 * x2993;
float x2995 = x2992 + x2994;
x212[x2989] = x2995;
float x2997 = x146[x2989];
float x2999 = x212[x2989];
float x2998 = 0.05f * x2993;
double x3000 = (double)x2999;
double x3001 = x3000 + 9.99999993922529E-9;
double x3002 = sqrt(x3001);
float x3003 = (float)x3002;
float x3004 = x2998 / x3003;
float x3005 = x2997 - x3004;
x146[x2989] = x3005;
x190[x2989] = 0.0f;

}
for(int x3010=0; x3010 < 22500; x3010++) {
float x3011 = x191[x3010];
float x3012 = x3011;
float x3013 = x213[x3010];
float x3014 = x3012;
float x3015 = x3014 * x3014;
float x3016 = x3013 + x3015;
x213[x3010] = x3016;
float x3018 = x154[x3010];
float x3020 = x213[x3010];
float x3019 = 0.05f * x3014;
double x3021 = (double)x3020;
double x3022 = x3021 + 9.99999993922529E-9;
double x3023 = sqrt(x3022);
float x3024 = (float)x3023;
float x3025 = x3019 / x3024;
float x3026 = x3018 - x3025;
x154[x3010] = x3026;
x191[x3010] = 0.0f;

}
for(int x3031=0; x3031 < 150; x3031++) {
float x3032 = x192[x3031];
float x3033 = x3032;
float x3034 = x214[x3031];
float x3035 = x3033;
float x3036 = x3035 * x3035;
float x3037 = x3034 + x3036;
x214[x3031] = x3037;
float x3039 = x162[x3031];
float x3041 = x214[x3031];
float x3040 = 0.05f * x3035;
double x3042 = (double)x3041;
double x3043 = x3042 + 9.99999993922529E-9;
double x3044 = sqrt(x3043);
float x3045 = (float)x3044;
float x3046 = x3040 / x3045;
float x3047 = x3039 - x3046;
x162[x3031] = x3047;
x192[x3031] = 0.0f;

}
for(int x3052=0; x3052 < 750; x3052++) {
float x3053 = x193[x3052];
float x3054 = x3053;
float x3055 = x215[x3052];
float x3056 = x3054;
float x3057 = x3056 * x3056;
float x3058 = x3055 + x3057;
x215[x3052] = x3058;
float x3060 = x163[x3052];
float x3062 = x215[x3052];
float x3061 = 0.05f * x3056;
double x3063 = (double)x3062;
double x3064 = x3063 + 9.99999993922529E-9;
double x3065 = sqrt(x3064);
float x3066 = (float)x3065;
float x3067 = x3061 / x3066;
float x3068 = x3060 - x3067;
x163[x3052] = x3068;
x193[x3052] = 0.0f;

}
for(int x3073=0; x3073 < 5; x3073++) {
float x3074 = x194[x3073];
float x3075 = x3074;
float x3076 = x216[x3073];
float x3077 = x3075;
float x3078 = x3077 * x3077;
float x3079 = x3076 + x3078;
x216[x3073] = x3079;
float x3081 = x172[x3073];
float x3083 = x216[x3073];
float x3082 = 0.05f * x3077;
double x3084 = (double)x3083;
double x3085 = x3084 + 9.99999993922529E-9;
double x3086 = sqrt(x3085);
float x3087 = (float)x3086;
float x3088 = x3082 / x3087;
float x3089 = x3081 - x3088;
x172[x3073] = x3089;
x194[x3073] = 0.0f;

}
int64_t x3094 = (long)mallocAddr;
int64_t x3095 = x3094 - x218;
memset((void*)x218, 0, x3095);
mallocAddr = (void*)x218;

}
float x3100 = x222;
double x3101 = (double)x3100;
x217[x221] = x3101;
double x3103 = ((double)clock() / CLOCKS_PER_SEC);
double x3104 = x3103 - x219;
printf("epoc %d, average_loss %f, time %lf\n",x221,x3100,x3104);

}
double x3108 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x3112 = (long)fopen(x0, "w");
fprintf((FILE *)x3112, "unit: %s\n", "1 epoch");
for(int x3114=0; x3114 < 6; x3114++) {
double x3115 = x217[x3114];
fprintf((FILE *)x3112, "%lf\n", x3115);

}
double x3109 = x219 - x2;
double x3110 = x3108 - x219;
double x3111 = x3110 / 6.0;
fprintf((FILE *)x3112, "run time: %lf %lf\n", x3109, x3111);
fclose((FILE*)x3112);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

