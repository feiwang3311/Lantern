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
float** x2112 = (float**)myMalloc(6 * sizeof(float*));;
x2112[0] = x236;
x2112[1] = x237;
x2112[2] = x238;
x2112[3] = x239;
x2112[4] = x240;
x2112[5] = x241;
function<void(float**)> x247 = x244;
function<void(float**)> x501 = [&](float** x502) {
float* x503 = x502[0];
float* x504 = x502[1];
float* x505 = x502[2];
float* x506 = x502[3];
float* x507 = x502[4];
float* x508 = x502[5];
float** x509 = (float**)myMalloc(6 * sizeof(float*));;
x509[0] = x503;
x509[1] = x504;
x509[2] = x505;
x509[3] = x506;
x509[4] = x507;
x509[5] = x508;
x247(x509);
};
function<void(float**)> x493 = [&](float** x494) {
float* x495 = x494[0];
float* x496 = x494[1];
float* x497 = x494[2];
float* x498 = x494[3];
float* x499 = x494[4];
float* x500 = x494[5];
float** x518 = (float**)myMalloc(6 * sizeof(float*));;
x518[0] = x495;
x518[1] = x496;
x518[2] = x497;
x518[3] = x498;
x518[4] = x499;
x518[5] = x500;
x501(x518);
};
function<void(float**)> x1317 = [&](float** x1318) {
float* x1319 = x1318[0];
float* x1320 = x1318[1];
float* x1321 = x1318[2];
float* x1322 = x1318[3];
float* x1323 = x1318[4];
float* x1324 = x1318[5];
float** x1325 = (float**)myMalloc(6 * sizeof(float*));;
x1325[0] = x1319;
x1325[1] = x1320;
x1325[2] = x1321;
x1325[3] = x1322;
x1325[4] = x1323;
x1325[5] = x1324;
x247(x1325);
};
function<void(float**)> x1309 = [&](float** x1310) {
float* x1311 = x1310[0];
float* x1312 = x1310[1];
float* x1313 = x1310[2];
float* x1314 = x1310[3];
float* x1315 = x1310[4];
float* x1316 = x1310[5];
float** x1334 = (float**)myMalloc(6 * sizeof(float*));;
x1334[0] = x1311;
x1334[1] = x1312;
x1334[2] = x1313;
x1334[3] = x1314;
x1334[4] = x1315;
x1334[5] = x1316;
x1317(x1334);
};
function<void(float**)> x257 = [&](float** x258) {
float* x259 = x258[0];
float* x260 = x258[1];
float* x261 = x258[2];
float* x262 = x258[3];
float* x263 = x258[4];
float* x264 = x258[5];
int32_t x265 = x232[x246];
float** x2102 = (float**)myMalloc(6 * sizeof(float*));;
x2102[0] = x236;
x2102[1] = x237;
x2102[2] = x238;
x2102[3] = x239;
x2102[4] = x240;
x2102[5] = x241;
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
float* x283 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x50,300,x281,1,0,x283,1);
float* x285 = (float*)myMalloc(150 * sizeof(float));;
float* x286 = (float*)myMalloc(150 * sizeof(float));;
int32_t x287 = 0;
int32_t x288 = 0;
int32_t x289 = 0;
for(int x291=0; x291 < 150; x291++) {
int32_t x292 = x287;
int32_t x293 = x288;
float x294 = x283[x293];
int32_t x295 = x289;
float x296 = x59[x295];
float x297 = x294 + x296;
x286[x292] = x297;
x287 += 1;
x288 += 1;
x289 += 1;

}
float* x304 = (float*)myMalloc(150 * sizeof(float));;
float* x305 = (float*)myMalloc(150 * sizeof(float));;
for(int x306=0; x306 < 150; x306++) {
float x307 = x286[x306];
float x308 = -1.0f * x307;
double x309 = (double)x308;
double x310 = exp(x309);
float x311 = (float)x310;
float x312 = x311 + 1.0f;
float x313 = 1.0f / x312;
x305[x306] = x313;

}
float* x317 = (float*)myMalloc(150 * sizeof(float));;
float* x318 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x60,300,x281,1,0,x318,1);
float* x320 = (float*)myMalloc(150 * sizeof(float));;
float* x321 = (float*)myMalloc(150 * sizeof(float));;
int32_t x322 = 0;
int32_t x323 = 0;
int32_t x324 = 0;
for(int x325=0; x325 < 150; x325++) {
int32_t x326 = x322;
int32_t x327 = x323;
float x328 = x318[x327];
int32_t x329 = x324;
float x330 = x68[x329];
float x331 = x328 + x330;
x321[x326] = x331;
x322 += 1;
x323 += 1;
x324 += 1;

}
float* x338 = (float*)myMalloc(150 * sizeof(float));;
float* x339 = (float*)myMalloc(150 * sizeof(float));;
for(int x340=0; x340 < 150; x340++) {
float x341 = x321[x340];
float x342 = -1.0f * x341;
double x343 = (double)x342;
double x344 = exp(x343);
float x345 = (float)x344;
float x346 = x345 + 1.0f;
float x347 = 1.0f / x346;
x339[x340] = x347;

}
float* x351 = (float*)myMalloc(150 * sizeof(float));;
float* x352 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x69,300,x281,1,0,x352,1);
float* x354 = (float*)myMalloc(150 * sizeof(float));;
float* x355 = (float*)myMalloc(150 * sizeof(float));;
int32_t x356 = 0;
int32_t x357 = 0;
int32_t x358 = 0;
for(int x359=0; x359 < 150; x359++) {
int32_t x360 = x356;
int32_t x361 = x357;
float x362 = x352[x361];
int32_t x363 = x358;
float x364 = x77[x363];
float x365 = x362 + x364;
x355[x360] = x365;
x356 += 1;
x357 += 1;
x358 += 1;

}
float* x372 = (float*)myMalloc(150 * sizeof(float));;
float* x373 = (float*)myMalloc(150 * sizeof(float));;
for(int x374=0; x374 < 150; x374++) {
float x375 = x355[x374];
double x376 = (double)x375;
double x377 = tanh(x376);
float x378 = (float)x377;
x373[x374] = x378;

}
float* x382 = (float*)myMalloc(150 * sizeof(float));;
float* x383 = (float*)myMalloc(150 * sizeof(float));;
int32_t x384 = 0;
int32_t x385 = 0;
int32_t x386 = 0;
for(int x387=0; x387 < 150; x387++) {
int32_t x388 = x384;
int32_t x389 = x385;
float x390 = x305[x389];
int32_t x391 = x386;
float x392 = x373[x391];
float x393 = x390 * x392;
x383[x388] = x393;
x384 += 1;
x385 += 1;
x386 += 1;

}
float* x400 = (float*)myMalloc(150 * sizeof(float));;
float* x401 = (float*)myMalloc(150 * sizeof(float));;
for(int x402=0; x402 < 150; x402++) {
float x403 = x383[x402];
double x404 = (double)x403;
double x405 = tanh(x404);
float x406 = (float)x405;
x401[x402] = x406;

}
float* x410 = (float*)myMalloc(150 * sizeof(float));;
float* x411 = (float*)myMalloc(150 * sizeof(float));;
int32_t x412 = 0;
int32_t x413 = 0;
int32_t x414 = 0;
for(int x415=0; x415 < 150; x415++) {
int32_t x416 = x412;
int32_t x417 = x413;
float x418 = x339[x417];
int32_t x419 = x414;
float x420 = x401[x419];
float x421 = x418 * x420;
x411[x416] = x421;
x412 += 1;
x413 += 1;
x414 += 1;

}
float* x428 = (float*)myMalloc(150 * sizeof(float));;
float* x429 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x163,150,x411,1,0,x429,1);
float* x431 = (float*)myMalloc(5 * sizeof(float));;
float* x432 = (float*)myMalloc(5 * sizeof(float));;
int32_t x433 = 0;
int32_t x434 = 0;
int32_t x435 = 0;
for(int x437=0; x437 < 5; x437++) {
int32_t x438 = x433;
int32_t x439 = x434;
float x440 = x429[x439];
int32_t x441 = x435;
float x442 = x172[x441];
float x443 = x440 + x442;
x432[x438] = x443;
x433 += 1;
x434 += 1;
x435 += 1;

}
float* x450 = (float*)myMalloc(5 * sizeof(float));;
float x451 = -3.4028235E38f;
for(int x452=0; x452 < 5; x452++) {
float x453 = x451;
float x454 = x432[x452];
bool x455 = x454 > x453;
float x456;
if (x455) {
x456 = x454;
} else {
x456 = x453;
}
x451 = x456;

}
float x460 = x451;
float x461 = 0.0f;
for(int x462=0; x462 < 5; x462++) {
float x463 = x461;
float x464 = x432[x462];
float x465 = x451;
float x466 = x464 - x465;
double x467 = (double)x466;
double x468 = exp(x467);
float x469 = (float)x468;
float x470 = x463 + x469;
x461 = x470;

}
float x474 = x461;
float* x479 = (float*)myMalloc(5 * sizeof(float));;
double x475 = (double)x474;
double x476 = log(x475);
float x477 = (float)x476;
float x478 = x460 + x477;
for(int x480=0; x480 < 5; x480++) {
float x481 = x432[x480];
float x482 = x481 - x478;
x479[x480] = x482;

}
float* x486 = (float*)myMalloc(5 * sizeof(float));;
int32_t x487 = x226[x246];
float x488 = x479[x487];
float* x490 = (float*)myMalloc(1 * sizeof(float));;
float x489 = -1.0f * x488;
x490[0] = x489;
float* x492 = (float*)myMalloc(1 * sizeof(float));;
float** x527 = (float**)myMalloc(6 * sizeof(float*));;
x527[0] = x490;
x527[1] = x492;
x527[2] = x411;
x527[3] = x428;
x527[4] = x383;
x527[5] = x400;
x493(x527);
float x535 = x486[x487];
float x536 = x492[0];
float x537 = -1.0f * x536;
float x538 = x535 + x537;
x486[x487] = x538;
float x540 = 0.0f;
for(int x541=0; x541 < 5; x541++) {
float x542 = x540;
float x543 = x486[x541];
float x544 = x542 + x543;
x540 = x544;

}
float x548 = x540;
float* x549 = (float*)myMalloc(1 * sizeof(float));;
x549[0] = x548;
float x551 = x549[0];
for(int x552=0; x552 < 5; x552++) {
float x553 = x450[x552];
float x554 = x486[x552];
float x555 = x479[x552];
double x556 = (double)x555;
double x557 = exp(x556);
float x558 = (float)x557;
float x559 = x558 * x551;
float x560 = x554 - x559;
float x561 = x553 + x560;
x450[x552] = x561;

}
int32_t x565 = 0;
int32_t x566 = 0;
int32_t x567 = 0;
for(int x568=0; x568 < 5; x568++) {
int32_t x569 = x565;
float x570 = x431[x569];
float x571 = x429[x569];
int32_t x572 = x566;
float x573 = x172[x572];
int32_t x574 = x567;
float x575 = x450[x574];
float x576 = x570 + x575;
x431[x569] = x576;
float x578 = x194[x572];
float x579 = x429[x569];
float x580 = x172[x572];
float x581 = x450[x574];
float x582 = x578 + x581;
x194[x572] = x582;
x567 += 1;
x565 += 1;
x566 += 1;

}
// add_cartesian
int32_t x590 = 0;
for(int x591=0; x591 < 5; x591++) {
for(int x592=0; x592 < 150; x592++) {
int32_t x593 = x590;
int32_t x594 = x593 + x592;
float x595 = x193[x594];
float x596 = x411[x592];
float x597 = x431[x591];
float x598 = x596 * x597;
float x599 = x595 + x598;
x193[x594] = x599;

}
x590 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x163,150,x431,1,1,x428,1);
int32_t x607 = 0;
int32_t x608 = 0;
int32_t x609 = 0;
for(int x610=0; x610 < 150; x610++) {
int32_t x611 = x607;
float x612 = x351[x611];
float x613 = x339[x611];
int32_t x614 = x608;
float x615 = x401[x614];
int32_t x616 = x609;
float x617 = x428[x616];
float x618 = x617 * x615;
float x619 = x612 + x618;
x351[x611] = x619;
float x621 = x410[x614];
float x622 = x339[x611];
float x623 = x401[x614];
float x624 = x428[x616];
float x625 = x624 * x622;
float x626 = x621 + x625;
x410[x614] = x626;
x609 += 1;
x607 += 1;
x608 += 1;

}
for(int x633=0; x633 < 150; x633++) {
float x634 = x400[x633];
float x635 = x401[x633];
float x638 = x410[x633];
float x636 = x635 * x635;
float x637 = 1.0f - x636;
float x639 = x637 * x638;
float x640 = x634 + x639;
x400[x633] = x640;

}
int32_t x644 = 0;
int32_t x645 = 0;
int32_t x646 = 0;
for(int x647=0; x647 < 150; x647++) {
int32_t x648 = x644;
float x649 = x317[x648];
float x650 = x305[x648];
int32_t x651 = x645;
float x652 = x373[x651];
int32_t x653 = x646;
float x654 = x400[x653];
float x655 = x654 * x652;
float x656 = x649 + x655;
x317[x648] = x656;
float x658 = x382[x651];
float x659 = x305[x648];
float x660 = x373[x651];
float x661 = x400[x653];
float x662 = x661 * x659;
float x663 = x658 + x662;
x382[x651] = x663;
x646 += 1;
x644 += 1;
x645 += 1;

}
for(int x670=0; x670 < 150; x670++) {
float x671 = x372[x670];
float x672 = x373[x670];
float x675 = x382[x670];
float x673 = x672 * x672;
float x674 = 1.0f - x673;
float x676 = x674 * x675;
float x677 = x671 + x676;
x372[x670] = x677;

}
int32_t x681 = 0;
int32_t x682 = 0;
int32_t x683 = 0;
for(int x684=0; x684 < 150; x684++) {
int32_t x685 = x681;
float x686 = x354[x685];
float x687 = x352[x685];
int32_t x688 = x682;
float x689 = x77[x688];
int32_t x690 = x683;
float x691 = x372[x690];
float x692 = x686 + x691;
x354[x685] = x692;
float x694 = x178[x688];
float x695 = x352[x685];
float x696 = x77[x688];
float x697 = x372[x690];
float x698 = x694 + x697;
x178[x688] = x698;
x683 += 1;
x681 += 1;
x682 += 1;

}
// add_cartesian
int32_t x706 = 0;
for(int x707=0; x707 < 150; x707++) {
for(int x708=0; x708 < 300; x708++) {
int32_t x709 = x706;
int32_t x710 = x709 + x708;
float x711 = x177[x710];
float x712 = x281[x708];
float x713 = x354[x707];
float x714 = x712 * x713;
float x715 = x711 + x714;
x177[x710] = x715;

}
x706 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x69,300,x354,1,1,x282,1);
for(int x723=0; x723 < 150; x723++) {
float x724 = x338[x723];
float x725 = x339[x723];
float x728 = x351[x723];
float x726 = 1.0f - x725;
float x727 = x726 * x725;
float x729 = x727 * x728;
float x730 = x724 + x729;
x338[x723] = x730;

}
int32_t x734 = 0;
int32_t x735 = 0;
int32_t x736 = 0;
for(int x737=0; x737 < 150; x737++) {
int32_t x738 = x734;
float x739 = x320[x738];
float x740 = x318[x738];
int32_t x741 = x735;
float x742 = x68[x741];
int32_t x743 = x736;
float x744 = x338[x743];
float x745 = x739 + x744;
x320[x738] = x745;
float x747 = x176[x741];
float x748 = x318[x738];
float x749 = x68[x741];
float x750 = x338[x743];
float x751 = x747 + x750;
x176[x741] = x751;
x736 += 1;
x734 += 1;
x735 += 1;

}
// add_cartesian
int32_t x759 = 0;
for(int x760=0; x760 < 150; x760++) {
for(int x761=0; x761 < 300; x761++) {
int32_t x762 = x759;
int32_t x763 = x762 + x761;
float x764 = x175[x763];
float x765 = x281[x761];
float x766 = x320[x760];
float x767 = x765 * x766;
float x768 = x764 + x767;
x175[x763] = x768;

}
x759 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x60,300,x320,1,1,x282,1);
for(int x776=0; x776 < 150; x776++) {
float x777 = x304[x776];
float x778 = x305[x776];
float x781 = x317[x776];
float x779 = 1.0f - x778;
float x780 = x779 * x778;
float x782 = x780 * x781;
float x783 = x777 + x782;
x304[x776] = x783;

}
int32_t x787 = 0;
int32_t x788 = 0;
int32_t x789 = 0;
for(int x790=0; x790 < 150; x790++) {
int32_t x791 = x787;
float x792 = x285[x791];
float x793 = x283[x791];
int32_t x794 = x788;
float x795 = x59[x794];
int32_t x796 = x789;
float x797 = x304[x796];
float x798 = x792 + x797;
x285[x791] = x798;
float x800 = x174[x794];
float x801 = x283[x791];
float x802 = x59[x794];
float x803 = x304[x796];
float x804 = x800 + x803;
x174[x794] = x804;
x789 += 1;
x787 += 1;
x788 += 1;

}
// add_cartesian
int32_t x812 = 0;
for(int x813=0; x813 < 150; x813++) {
for(int x814=0; x814 < 300; x814++) {
int32_t x815 = x812;
int32_t x816 = x815 + x814;
float x817 = x173[x816];
float x818 = x281[x814];
float x819 = x285[x813];
float x820 = x818 * x819;
float x821 = x817 + x820;
x173[x816] = x821;

}
x812 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x50,300,x285,1,1,x282,1);
} else {
float* x830 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x78,150,x261,1,0,x830,1);
float* x832 = (float*)myMalloc(150 * sizeof(float));;
float* x833 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x87,150,x270,1,0,x833,1);
float* x835 = (float*)myMalloc(150 * sizeof(float));;
float* x836 = (float*)myMalloc(150 * sizeof(float));;
int32_t x837 = 0;
int32_t x838 = 0;
int32_t x839 = 0;
for(int x840=0; x840 < 150; x840++) {
int32_t x841 = x837;
int32_t x842 = x838;
float x843 = x830[x842];
int32_t x844 = x839;
float x845 = x833[x844];
float x846 = x843 + x845;
x836[x841] = x846;
x837 += 1;
x838 += 1;
x839 += 1;

}
float* x853 = (float*)myMalloc(150 * sizeof(float));;
float* x854 = (float*)myMalloc(150 * sizeof(float));;
int32_t x855 = 0;
int32_t x856 = 0;
int32_t x857 = 0;
for(int x858=0; x858 < 150; x858++) {
int32_t x859 = x855;
int32_t x860 = x856;
float x861 = x836[x860];
int32_t x862 = x857;
float x863 = x95[x862];
float x864 = x861 + x863;
x854[x859] = x864;
x855 += 1;
x856 += 1;
x857 += 1;

}
float* x871 = (float*)myMalloc(150 * sizeof(float));;
float* x872 = (float*)myMalloc(150 * sizeof(float));;
for(int x873=0; x873 < 150; x873++) {
float x874 = x854[x873];
float x875 = -1.0f * x874;
double x876 = (double)x875;
double x877 = exp(x876);
float x878 = (float)x877;
float x879 = x878 + 1.0f;
float x880 = 1.0f / x879;
x872[x873] = x880;

}
float* x884 = (float*)myMalloc(150 * sizeof(float));;
float* x885 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x96,150,x261,1,0,x885,1);
float* x887 = (float*)myMalloc(150 * sizeof(float));;
float* x888 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x104,150,x270,1,0,x888,1);
float* x890 = (float*)myMalloc(150 * sizeof(float));;
float* x891 = (float*)myMalloc(150 * sizeof(float));;
int32_t x892 = 0;
int32_t x893 = 0;
int32_t x894 = 0;
for(int x895=0; x895 < 150; x895++) {
int32_t x896 = x892;
int32_t x897 = x893;
float x898 = x885[x897];
int32_t x899 = x894;
float x900 = x888[x899];
float x901 = x898 + x900;
x891[x896] = x901;
x892 += 1;
x893 += 1;
x894 += 1;

}
float* x908 = (float*)myMalloc(150 * sizeof(float));;
float* x909 = (float*)myMalloc(150 * sizeof(float));;
int32_t x910 = 0;
int32_t x911 = 0;
int32_t x912 = 0;
for(int x913=0; x913 < 150; x913++) {
int32_t x914 = x910;
int32_t x915 = x911;
float x916 = x891[x915];
int32_t x917 = x912;
float x918 = x128[x917];
float x919 = x916 + x918;
x909[x914] = x919;
x910 += 1;
x911 += 1;
x912 += 1;

}
float* x926 = (float*)myMalloc(150 * sizeof(float));;
float* x927 = (float*)myMalloc(150 * sizeof(float));;
for(int x928=0; x928 < 150; x928++) {
float x929 = x909[x928];
float x930 = -1.0f * x929;
double x931 = (double)x930;
double x932 = exp(x931);
float x933 = (float)x932;
float x934 = x933 + 1.0f;
float x935 = 1.0f / x934;
x927[x928] = x935;

}
float* x939 = (float*)myMalloc(150 * sizeof(float));;
float* x940 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x112,150,x261,1,0,x940,1);
float* x942 = (float*)myMalloc(150 * sizeof(float));;
float* x943 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x120,150,x270,1,0,x943,1);
float* x945 = (float*)myMalloc(150 * sizeof(float));;
float* x946 = (float*)myMalloc(150 * sizeof(float));;
int32_t x947 = 0;
int32_t x948 = 0;
int32_t x949 = 0;
for(int x950=0; x950 < 150; x950++) {
int32_t x951 = x947;
int32_t x952 = x948;
float x953 = x940[x952];
int32_t x954 = x949;
float x955 = x943[x954];
float x956 = x953 + x955;
x946[x951] = x956;
x947 += 1;
x948 += 1;
x949 += 1;

}
float* x963 = (float*)myMalloc(150 * sizeof(float));;
float* x964 = (float*)myMalloc(150 * sizeof(float));;
int32_t x965 = 0;
int32_t x966 = 0;
int32_t x967 = 0;
for(int x968=0; x968 < 150; x968++) {
int32_t x969 = x965;
int32_t x970 = x966;
float x971 = x946[x970];
int32_t x972 = x967;
float x973 = x128[x972];
float x974 = x971 + x973;
x964[x969] = x974;
x965 += 1;
x966 += 1;
x967 += 1;

}
float* x981 = (float*)myMalloc(150 * sizeof(float));;
float* x982 = (float*)myMalloc(150 * sizeof(float));;
for(int x983=0; x983 < 150; x983++) {
float x984 = x964[x983];
float x985 = -1.0f * x984;
double x986 = (double)x985;
double x987 = exp(x986);
float x988 = (float)x987;
float x989 = x988 + 1.0f;
float x990 = 1.0f / x989;
x982[x983] = x990;

}
float* x994 = (float*)myMalloc(150 * sizeof(float));;
float* x995 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x129,150,x261,1,0,x995,1);
float* x997 = (float*)myMalloc(150 * sizeof(float));;
float* x998 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x137,150,x270,1,0,x998,1);
float* x1000 = (float*)myMalloc(150 * sizeof(float));;
float* x1001 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1002 = 0;
int32_t x1003 = 0;
int32_t x1004 = 0;
for(int x1005=0; x1005 < 150; x1005++) {
int32_t x1006 = x1002;
int32_t x1007 = x1003;
float x1008 = x995[x1007];
int32_t x1009 = x1004;
float x1010 = x998[x1009];
float x1011 = x1008 + x1010;
x1001[x1006] = x1011;
x1002 += 1;
x1003 += 1;
x1004 += 1;

}
float* x1018 = (float*)myMalloc(150 * sizeof(float));;
float* x1019 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1020 = 0;
int32_t x1021 = 0;
int32_t x1022 = 0;
for(int x1023=0; x1023 < 150; x1023++) {
int32_t x1024 = x1020;
int32_t x1025 = x1021;
float x1026 = x1001[x1025];
int32_t x1027 = x1022;
float x1028 = x145[x1027];
float x1029 = x1026 + x1028;
x1019[x1024] = x1029;
x1020 += 1;
x1021 += 1;
x1022 += 1;

}
float* x1036 = (float*)myMalloc(150 * sizeof(float));;
float* x1037 = (float*)myMalloc(150 * sizeof(float));;
for(int x1038=0; x1038 < 150; x1038++) {
float x1039 = x1019[x1038];
float x1040 = -1.0f * x1039;
double x1041 = (double)x1040;
double x1042 = exp(x1041);
float x1043 = (float)x1042;
float x1044 = x1043 + 1.0f;
float x1045 = 1.0f / x1044;
x1037[x1038] = x1045;

}
float* x1049 = (float*)myMalloc(150 * sizeof(float));;
float* x1050 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x146,150,x261,1,0,x1050,1);
float* x1052 = (float*)myMalloc(150 * sizeof(float));;
float* x1053 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x154,150,x270,1,0,x1053,1);
float* x1055 = (float*)myMalloc(150 * sizeof(float));;
float* x1056 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1057 = 0;
int32_t x1058 = 0;
int32_t x1059 = 0;
for(int x1060=0; x1060 < 150; x1060++) {
int32_t x1061 = x1057;
int32_t x1062 = x1058;
float x1063 = x1050[x1062];
int32_t x1064 = x1059;
float x1065 = x1053[x1064];
float x1066 = x1063 + x1065;
x1056[x1061] = x1066;
x1057 += 1;
x1058 += 1;
x1059 += 1;

}
float* x1073 = (float*)myMalloc(150 * sizeof(float));;
float* x1074 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1075 = 0;
int32_t x1076 = 0;
int32_t x1077 = 0;
for(int x1078=0; x1078 < 150; x1078++) {
int32_t x1079 = x1075;
int32_t x1080 = x1076;
float x1081 = x1056[x1080];
int32_t x1082 = x1077;
float x1083 = x162[x1082];
float x1084 = x1081 + x1083;
x1074[x1079] = x1084;
x1075 += 1;
x1076 += 1;
x1077 += 1;

}
float* x1091 = (float*)myMalloc(150 * sizeof(float));;
float* x1092 = (float*)myMalloc(150 * sizeof(float));;
for(int x1093=0; x1093 < 150; x1093++) {
float x1094 = x1074[x1093];
double x1095 = (double)x1094;
double x1096 = tanh(x1095);
float x1097 = (float)x1096;
x1092[x1093] = x1097;

}
float* x1101 = (float*)myMalloc(150 * sizeof(float));;
float* x1102 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1103 = 0;
int32_t x1104 = 0;
int32_t x1105 = 0;
for(int x1106=0; x1106 < 150; x1106++) {
int32_t x1107 = x1103;
int32_t x1108 = x1104;
float x1109 = x872[x1108];
int32_t x1110 = x1105;
float x1111 = x1092[x1110];
float x1112 = x1109 * x1111;
x1102[x1107] = x1112;
x1103 += 1;
x1104 += 1;
x1105 += 1;

}
float* x1119 = (float*)myMalloc(150 * sizeof(float));;
float* x1120 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1121 = 0;
int32_t x1122 = 0;
int32_t x1123 = 0;
for(int x1124=0; x1124 < 150; x1124++) {
int32_t x1125 = x1121;
int32_t x1126 = x1122;
float x1127 = x927[x1126];
int32_t x1128 = x1123;
float x1129 = x263[x1128];
float x1130 = x1127 * x1129;
x1120[x1125] = x1130;
x1121 += 1;
x1122 += 1;
x1123 += 1;

}
float* x1137 = (float*)myMalloc(150 * sizeof(float));;
float* x1138 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1139 = 0;
int32_t x1140 = 0;
int32_t x1141 = 0;
for(int x1142=0; x1142 < 150; x1142++) {
int32_t x1143 = x1139;
int32_t x1144 = x1140;
float x1145 = x1102[x1144];
int32_t x1146 = x1141;
float x1147 = x1120[x1146];
float x1148 = x1145 + x1147;
x1138[x1143] = x1148;
x1139 += 1;
x1140 += 1;
x1141 += 1;

}
float* x1155 = (float*)myMalloc(150 * sizeof(float));;
float* x1156 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1157 = 0;
int32_t x1158 = 0;
int32_t x1159 = 0;
for(int x1160=0; x1160 < 150; x1160++) {
int32_t x1161 = x1157;
int32_t x1162 = x1158;
float x1163 = x982[x1162];
int32_t x1164 = x1159;
float x1165 = x272[x1164];
float x1166 = x1163 * x1165;
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
float x1181 = x1138[x1180];
int32_t x1182 = x1177;
float x1183 = x1156[x1182];
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
double x1195 = (double)x1194;
double x1196 = tanh(x1195);
float x1197 = (float)x1196;
x1192[x1193] = x1197;

}
float* x1201 = (float*)myMalloc(150 * sizeof(float));;
float* x1202 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1203 = 0;
int32_t x1204 = 0;
int32_t x1205 = 0;
for(int x1206=0; x1206 < 150; x1206++) {
int32_t x1207 = x1203;
int32_t x1208 = x1204;
float x1209 = x1037[x1208];
int32_t x1210 = x1205;
float x1211 = x1192[x1210];
float x1212 = x1209 * x1211;
x1202[x1207] = x1212;
x1203 += 1;
x1204 += 1;
x1205 += 1;

}
float* x1219 = (float*)myMalloc(150 * sizeof(float));;
float* x1220 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x163,150,x1202,1,0,x1220,1);
float* x1222 = (float*)myMalloc(5 * sizeof(float));;
float* x1223 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1224 = 0;
int32_t x1225 = 0;
int32_t x1226 = 0;
for(int x1227=0; x1227 < 5; x1227++) {
int32_t x1228 = x1224;
int32_t x1229 = x1225;
float x1230 = x1220[x1229];
int32_t x1231 = x1226;
float x1232 = x172[x1231];
float x1233 = x1230 + x1232;
x1223[x1228] = x1233;
x1224 += 1;
x1225 += 1;
x1226 += 1;

}
float* x1240 = (float*)myMalloc(5 * sizeof(float));;
float* x1241 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1242 = 0;
int32_t x1243 = 0;
int32_t x1244 = 0;
int32_t x1245 = x1242;
int32_t x1246 = x1243;
float x1247 = x259[x1246];
int32_t x1248 = x1244;
float x1249 = x268[x1248];
float x1250 = x1247 + x1249;
x1241[x1245] = x1250;
x1242 += 1;
float* x1253 = (float*)myMalloc(1 * sizeof(float));;
float x1254 = -3.4028235E38f;
for(int x1255=0; x1255 < 5; x1255++) {
float x1256 = x1254;
float x1257 = x1223[x1255];
bool x1258 = x1257 > x1256;
float x1259;
if (x1258) {
x1259 = x1257;
} else {
x1259 = x1256;
}
x1254 = x1259;

}
float x1263 = x1254;
float x1264 = 0.0f;
for(int x1265=0; x1265 < 5; x1265++) {
float x1266 = x1264;
float x1267 = x1223[x1265];
float x1268 = x1254;
float x1269 = x1267 - x1268;
double x1270 = (double)x1269;
double x1271 = exp(x1270);
float x1272 = (float)x1271;
float x1273 = x1266 + x1272;
x1264 = x1273;

}
float x1277 = x1264;
float* x1282 = (float*)myMalloc(5 * sizeof(float));;
double x1278 = (double)x1277;
double x1279 = log(x1278);
float x1280 = (float)x1279;
float x1281 = x1263 + x1280;
for(int x1283=0; x1283 < 5; x1283++) {
float x1284 = x1223[x1283];
float x1285 = x1284 - x1281;
x1282[x1283] = x1285;

}
float* x1289 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1290 = x226[x246];
float x1291 = x1282[x1290];
float* x1293 = (float*)myMalloc(1 * sizeof(float));;
float x1292 = -1.0f * x1291;
x1293[0] = x1292;
float* x1295 = (float*)myMalloc(1 * sizeof(float));;
float* x1296 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1297 = 0;
int32_t x1298 = 0;
int32_t x1299 = 0;
int32_t x1300 = x1297;
int32_t x1301 = x1298;
float x1302 = x1241[x1301];
int32_t x1303 = x1299;
float x1304 = x1293[x1303];
float x1305 = x1302 + x1304;
x1296[x1300] = x1305;
x1297 += 1;
float* x1308 = (float*)myMalloc(1 * sizeof(float));;
float** x1343 = (float**)myMalloc(6 * sizeof(float*));;
x1343[0] = x1296;
x1343[1] = x1308;
x1343[2] = x1202;
x1343[3] = x1219;
x1343[4] = x1174;
x1343[5] = x1191;
x1309(x1343);
int32_t x1351 = 0;
int32_t x1352 = 0;
int32_t x1353 = 0;
int32_t x1354 = x1351;
float x1355 = x1253[x1354];
float x1356 = x1241[x1354];
int32_t x1357 = x1352;
float x1358 = x1293[x1357];
int32_t x1359 = x1353;
float x1360 = x1308[x1359];
float x1361 = x1355 + x1360;
x1253[x1354] = x1361;
float x1363 = x1295[x1357];
float x1364 = x1241[x1354];
float x1365 = x1293[x1357];
float x1366 = x1308[x1359];
float x1367 = x1363 + x1366;
x1295[x1357] = x1367;
x1353 += 1;
float x1370 = x1289[x1290];
float x1371 = x1295[0];
float x1372 = -1.0f * x1371;
float x1373 = x1370 + x1372;
x1289[x1290] = x1373;
float x1375 = 0.0f;
for(int x1376=0; x1376 < 5; x1376++) {
float x1377 = x1375;
float x1378 = x1289[x1376];
float x1379 = x1377 + x1378;
x1375 = x1379;

}
float x1383 = x1375;
float* x1384 = (float*)myMalloc(1 * sizeof(float));;
x1384[0] = x1383;
float x1386 = x1384[0];
for(int x1387=0; x1387 < 5; x1387++) {
float x1388 = x1240[x1387];
float x1389 = x1289[x1387];
float x1390 = x1282[x1387];
double x1391 = (double)x1390;
double x1392 = exp(x1391);
float x1393 = (float)x1392;
float x1394 = x1393 * x1386;
float x1395 = x1389 - x1394;
float x1396 = x1388 + x1395;
x1240[x1387] = x1396;

}
int32_t x1400 = 0;
int32_t x1401 = 0;
int32_t x1402 = 0;
int32_t x1403 = x1400;
float x1404 = x260[x1403];
float x1405 = x259[x1403];
int32_t x1406 = x1401;
float x1407 = x268[x1406];
int32_t x1408 = x1402;
float x1409 = x1253[x1408];
float x1410 = x1404 + x1409;
x260[x1403] = x1410;
float x1412 = x269[x1406];
float x1413 = x259[x1403];
float x1414 = x268[x1406];
float x1415 = x1253[x1408];
float x1416 = x1412 + x1415;
x269[x1406] = x1416;
x1402 += 1;
int32_t x1419 = 0;
int32_t x1420 = 0;
int32_t x1421 = 0;
for(int x1422=0; x1422 < 5; x1422++) {
int32_t x1423 = x1419;
float x1424 = x1222[x1423];
float x1425 = x1220[x1423];
int32_t x1426 = x1420;
float x1427 = x172[x1426];
int32_t x1428 = x1421;
float x1429 = x1240[x1428];
float x1430 = x1424 + x1429;
x1222[x1423] = x1430;
float x1432 = x194[x1426];
float x1433 = x1220[x1423];
float x1434 = x172[x1426];
float x1435 = x1240[x1428];
float x1436 = x1432 + x1435;
x194[x1426] = x1436;
x1421 += 1;
x1419 += 1;
x1420 += 1;

}
// add_cartesian
int32_t x1444 = 0;
for(int x1445=0; x1445 < 5; x1445++) {
for(int x1446=0; x1446 < 150; x1446++) {
int32_t x1447 = x1444;
int32_t x1448 = x1447 + x1446;
float x1449 = x193[x1448];
float x1450 = x1202[x1446];
float x1451 = x1222[x1445];
float x1452 = x1450 * x1451;
float x1453 = x1449 + x1452;
x193[x1448] = x1453;

}
x1444 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x163,150,x1222,1,1,x1219,1);
int32_t x1461 = 0;
int32_t x1462 = 0;
int32_t x1463 = 0;
for(int x1464=0; x1464 < 150; x1464++) {
int32_t x1465 = x1461;
float x1466 = x1049[x1465];
float x1467 = x1037[x1465];
int32_t x1468 = x1462;
float x1469 = x1192[x1468];
int32_t x1470 = x1463;
float x1471 = x1219[x1470];
float x1472 = x1471 * x1469;
float x1473 = x1466 + x1472;
x1049[x1465] = x1473;
float x1475 = x1201[x1468];
float x1476 = x1037[x1465];
float x1477 = x1192[x1468];
float x1478 = x1219[x1470];
float x1479 = x1478 * x1476;
float x1480 = x1475 + x1479;
x1201[x1468] = x1480;
x1463 += 1;
x1461 += 1;
x1462 += 1;

}
for(int x1487=0; x1487 < 150; x1487++) {
float x1488 = x1191[x1487];
float x1489 = x1192[x1487];
float x1492 = x1201[x1487];
float x1490 = x1489 * x1489;
float x1491 = 1.0f - x1490;
float x1493 = x1491 * x1492;
float x1494 = x1488 + x1493;
x1191[x1487] = x1494;

}
int32_t x1498 = 0;
int32_t x1499 = 0;
int32_t x1500 = 0;
for(int x1501=0; x1501 < 150; x1501++) {
int32_t x1502 = x1498;
float x1503 = x1155[x1502];
float x1504 = x1138[x1502];
int32_t x1505 = x1499;
float x1506 = x1156[x1505];
int32_t x1507 = x1500;
float x1508 = x1191[x1507];
float x1509 = x1503 + x1508;
x1155[x1502] = x1509;
float x1511 = x1173[x1505];
float x1512 = x1138[x1502];
float x1513 = x1156[x1505];
float x1514 = x1191[x1507];
float x1515 = x1511 + x1514;
x1173[x1505] = x1515;
x1500 += 1;
x1498 += 1;
x1499 += 1;

}
int32_t x1522 = 0;
int32_t x1523 = 0;
int32_t x1524 = 0;
for(int x1525=0; x1525 < 150; x1525++) {
int32_t x1526 = x1522;
float x1527 = x994[x1526];
float x1528 = x982[x1526];
int32_t x1529 = x1523;
float x1530 = x272[x1529];
int32_t x1531 = x1524;
float x1532 = x1173[x1531];
float x1533 = x1532 * x1530;
float x1534 = x1527 + x1533;
x994[x1526] = x1534;
float x1536 = x273[x1529];
float x1537 = x982[x1526];
float x1538 = x272[x1529];
float x1539 = x1173[x1531];
float x1540 = x1539 * x1537;
float x1541 = x1536 + x1540;
x273[x1529] = x1541;
x1524 += 1;
x1522 += 1;
x1523 += 1;

}
int32_t x1548 = 0;
int32_t x1549 = 0;
int32_t x1550 = 0;
for(int x1551=0; x1551 < 150; x1551++) {
int32_t x1552 = x1548;
float x1553 = x1119[x1552];
float x1554 = x1102[x1552];
int32_t x1555 = x1549;
float x1556 = x1120[x1555];
int32_t x1557 = x1550;
float x1558 = x1155[x1557];
float x1559 = x1553 + x1558;
x1119[x1552] = x1559;
float x1561 = x1137[x1555];
float x1562 = x1102[x1552];
float x1563 = x1120[x1555];
float x1564 = x1155[x1557];
float x1565 = x1561 + x1564;
x1137[x1555] = x1565;
x1550 += 1;
x1548 += 1;
x1549 += 1;

}
int32_t x1572 = 0;
int32_t x1573 = 0;
int32_t x1574 = 0;
for(int x1575=0; x1575 < 150; x1575++) {
int32_t x1576 = x1572;
float x1577 = x939[x1576];
float x1578 = x927[x1576];
int32_t x1579 = x1573;
float x1580 = x263[x1579];
int32_t x1581 = x1574;
float x1582 = x1137[x1581];
float x1583 = x1582 * x1580;
float x1584 = x1577 + x1583;
x939[x1576] = x1584;
float x1586 = x264[x1579];
float x1587 = x927[x1576];
float x1588 = x263[x1579];
float x1589 = x1137[x1581];
float x1590 = x1589 * x1587;
float x1591 = x1586 + x1590;
x264[x1579] = x1591;
x1574 += 1;
x1572 += 1;
x1573 += 1;

}
int32_t x1598 = 0;
int32_t x1599 = 0;
int32_t x1600 = 0;
for(int x1601=0; x1601 < 150; x1601++) {
int32_t x1602 = x1598;
float x1603 = x884[x1602];
float x1604 = x872[x1602];
int32_t x1605 = x1599;
float x1606 = x1092[x1605];
int32_t x1607 = x1600;
float x1608 = x1119[x1607];
float x1609 = x1608 * x1606;
float x1610 = x1603 + x1609;
x884[x1602] = x1610;
float x1612 = x1101[x1605];
float x1613 = x872[x1602];
float x1614 = x1092[x1605];
float x1615 = x1119[x1607];
float x1616 = x1615 * x1613;
float x1617 = x1612 + x1616;
x1101[x1605] = x1617;
x1600 += 1;
x1598 += 1;
x1599 += 1;

}
for(int x1624=0; x1624 < 150; x1624++) {
float x1625 = x1091[x1624];
float x1626 = x1092[x1624];
float x1629 = x1101[x1624];
float x1627 = x1626 * x1626;
float x1628 = 1.0f - x1627;
float x1630 = x1628 * x1629;
float x1631 = x1625 + x1630;
x1091[x1624] = x1631;

}
int32_t x1635 = 0;
int32_t x1636 = 0;
int32_t x1637 = 0;
for(int x1638=0; x1638 < 150; x1638++) {
int32_t x1639 = x1635;
float x1640 = x1073[x1639];
float x1641 = x1056[x1639];
int32_t x1642 = x1636;
float x1643 = x162[x1642];
int32_t x1644 = x1637;
float x1645 = x1091[x1644];
float x1646 = x1640 + x1645;
x1073[x1639] = x1646;
float x1648 = x192[x1642];
float x1649 = x1056[x1639];
float x1650 = x162[x1642];
float x1651 = x1091[x1644];
float x1652 = x1648 + x1651;
x192[x1642] = x1652;
x1637 += 1;
x1635 += 1;
x1636 += 1;

}
int32_t x1659 = 0;
int32_t x1660 = 0;
int32_t x1661 = 0;
for(int x1662=0; x1662 < 150; x1662++) {
int32_t x1663 = x1659;
float x1664 = x1052[x1663];
float x1665 = x1050[x1663];
int32_t x1666 = x1660;
float x1667 = x1053[x1666];
int32_t x1668 = x1661;
float x1669 = x1073[x1668];
float x1670 = x1664 + x1669;
x1052[x1663] = x1670;
float x1672 = x1055[x1666];
float x1673 = x1050[x1663];
float x1674 = x1053[x1666];
float x1675 = x1073[x1668];
float x1676 = x1672 + x1675;
x1055[x1666] = x1676;
x1661 += 1;
x1659 += 1;
x1660 += 1;

}
// add_cartesian
int32_t x1684 = 0;
for(int x1685=0; x1685 < 150; x1685++) {
for(int x1686=0; x1686 < 150; x1686++) {
int32_t x1687 = x1684;
int32_t x1688 = x1687 + x1686;
float x1689 = x191[x1688];
float x1690 = x270[x1686];
float x1691 = x1055[x1685];
float x1692 = x1690 * x1691;
float x1693 = x1689 + x1692;
x191[x1688] = x1693;

}
x1684 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x154,150,x1055,1,1,x271,1);
// add_cartesian
int32_t x1702 = 0;
for(int x1703=0; x1703 < 150; x1703++) {
for(int x1704=0; x1704 < 150; x1704++) {
int32_t x1705 = x1702;
int32_t x1706 = x1705 + x1704;
float x1707 = x190[x1706];
float x1708 = x261[x1704];
float x1709 = x1052[x1703];
float x1710 = x1708 * x1709;
float x1711 = x1707 + x1710;
x190[x1706] = x1711;

}
x1702 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x146,150,x1052,1,1,x262,1);
for(int x1719=0; x1719 < 150; x1719++) {
float x1720 = x1036[x1719];
float x1721 = x1037[x1719];
float x1724 = x1049[x1719];
float x1722 = 1.0f - x1721;
float x1723 = x1722 * x1721;
float x1725 = x1723 * x1724;
float x1726 = x1720 + x1725;
x1036[x1719] = x1726;

}
int32_t x1730 = 0;
int32_t x1731 = 0;
int32_t x1732 = 0;
for(int x1733=0; x1733 < 150; x1733++) {
int32_t x1734 = x1730;
float x1735 = x1018[x1734];
float x1736 = x1001[x1734];
int32_t x1737 = x1731;
float x1738 = x145[x1737];
int32_t x1739 = x1732;
float x1740 = x1036[x1739];
float x1741 = x1735 + x1740;
x1018[x1734] = x1741;
float x1743 = x189[x1737];
float x1744 = x1001[x1734];
float x1745 = x145[x1737];
float x1746 = x1036[x1739];
float x1747 = x1743 + x1746;
x189[x1737] = x1747;
x1732 += 1;
x1730 += 1;
x1731 += 1;

}
int32_t x1754 = 0;
int32_t x1755 = 0;
int32_t x1756 = 0;
for(int x1757=0; x1757 < 150; x1757++) {
int32_t x1758 = x1754;
float x1759 = x997[x1758];
float x1760 = x995[x1758];
int32_t x1761 = x1755;
float x1762 = x998[x1761];
int32_t x1763 = x1756;
float x1764 = x1018[x1763];
float x1765 = x1759 + x1764;
x997[x1758] = x1765;
float x1767 = x1000[x1761];
float x1768 = x995[x1758];
float x1769 = x998[x1761];
float x1770 = x1018[x1763];
float x1771 = x1767 + x1770;
x1000[x1761] = x1771;
x1756 += 1;
x1754 += 1;
x1755 += 1;

}
// add_cartesian
int32_t x1779 = 0;
for(int x1780=0; x1780 < 150; x1780++) {
for(int x1781=0; x1781 < 150; x1781++) {
int32_t x1782 = x1779;
int32_t x1783 = x1782 + x1781;
float x1784 = x188[x1783];
float x1785 = x270[x1781];
float x1786 = x1000[x1780];
float x1787 = x1785 * x1786;
float x1788 = x1784 + x1787;
x188[x1783] = x1788;

}
x1779 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x137,150,x1000,1,1,x271,1);
// add_cartesian
int32_t x1797 = 0;
for(int x1798=0; x1798 < 150; x1798++) {
for(int x1799=0; x1799 < 150; x1799++) {
int32_t x1800 = x1797;
int32_t x1801 = x1800 + x1799;
float x1802 = x187[x1801];
float x1803 = x261[x1799];
float x1804 = x997[x1798];
float x1805 = x1803 * x1804;
float x1806 = x1802 + x1805;
x187[x1801] = x1806;

}
x1797 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x129,150,x997,1,1,x262,1);
for(int x1814=0; x1814 < 150; x1814++) {
float x1815 = x981[x1814];
float x1816 = x982[x1814];
float x1819 = x994[x1814];
float x1817 = 1.0f - x1816;
float x1818 = x1817 * x1816;
float x1820 = x1818 * x1819;
float x1821 = x1815 + x1820;
x981[x1814] = x1821;

}
int32_t x1825 = 0;
int32_t x1826 = 0;
int32_t x1827 = 0;
for(int x1828=0; x1828 < 150; x1828++) {
int32_t x1829 = x1825;
float x1830 = x963[x1829];
float x1831 = x946[x1829];
int32_t x1832 = x1826;
float x1833 = x128[x1832];
int32_t x1834 = x1827;
float x1835 = x981[x1834];
float x1836 = x1830 + x1835;
x963[x1829] = x1836;
float x1838 = x186[x1832];
float x1839 = x946[x1829];
float x1840 = x128[x1832];
float x1841 = x981[x1834];
float x1842 = x1838 + x1841;
x186[x1832] = x1842;
x1827 += 1;
x1825 += 1;
x1826 += 1;

}
int32_t x1849 = 0;
int32_t x1850 = 0;
int32_t x1851 = 0;
for(int x1852=0; x1852 < 150; x1852++) {
int32_t x1853 = x1849;
float x1854 = x942[x1853];
float x1855 = x940[x1853];
int32_t x1856 = x1850;
float x1857 = x943[x1856];
int32_t x1858 = x1851;
float x1859 = x963[x1858];
float x1860 = x1854 + x1859;
x942[x1853] = x1860;
float x1862 = x945[x1856];
float x1863 = x940[x1853];
float x1864 = x943[x1856];
float x1865 = x963[x1858];
float x1866 = x1862 + x1865;
x945[x1856] = x1866;
x1851 += 1;
x1849 += 1;
x1850 += 1;

}
// add_cartesian
int32_t x1874 = 0;
for(int x1875=0; x1875 < 150; x1875++) {
for(int x1876=0; x1876 < 150; x1876++) {
int32_t x1877 = x1874;
int32_t x1878 = x1877 + x1876;
float x1879 = x185[x1878];
float x1880 = x270[x1876];
float x1881 = x945[x1875];
float x1882 = x1880 * x1881;
float x1883 = x1879 + x1882;
x185[x1878] = x1883;

}
x1874 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x120,150,x945,1,1,x271,1);
// add_cartesian
int32_t x1892 = 0;
for(int x1893=0; x1893 < 150; x1893++) {
for(int x1894=0; x1894 < 150; x1894++) {
int32_t x1895 = x1892;
int32_t x1896 = x1895 + x1894;
float x1897 = x184[x1896];
float x1898 = x261[x1894];
float x1899 = x942[x1893];
float x1900 = x1898 * x1899;
float x1901 = x1897 + x1900;
x184[x1896] = x1901;

}
x1892 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x112,150,x942,1,1,x262,1);
for(int x1909=0; x1909 < 150; x1909++) {
float x1910 = x926[x1909];
float x1911 = x927[x1909];
float x1914 = x939[x1909];
float x1912 = 1.0f - x1911;
float x1913 = x1912 * x1911;
float x1915 = x1913 * x1914;
float x1916 = x1910 + x1915;
x926[x1909] = x1916;

}
int32_t x1920 = 0;
int32_t x1921 = 0;
int32_t x1922 = 0;
for(int x1923=0; x1923 < 150; x1923++) {
int32_t x1924 = x1920;
float x1925 = x908[x1924];
float x1926 = x891[x1924];
int32_t x1927 = x1921;
float x1928 = x128[x1927];
int32_t x1929 = x1922;
float x1930 = x926[x1929];
float x1931 = x1925 + x1930;
x908[x1924] = x1931;
float x1933 = x186[x1927];
float x1934 = x891[x1924];
float x1935 = x128[x1927];
float x1936 = x926[x1929];
float x1937 = x1933 + x1936;
x186[x1927] = x1937;
x1922 += 1;
x1920 += 1;
x1921 += 1;

}
int32_t x1944 = 0;
int32_t x1945 = 0;
int32_t x1946 = 0;
for(int x1947=0; x1947 < 150; x1947++) {
int32_t x1948 = x1944;
float x1949 = x887[x1948];
float x1950 = x885[x1948];
int32_t x1951 = x1945;
float x1952 = x888[x1951];
int32_t x1953 = x1946;
float x1954 = x908[x1953];
float x1955 = x1949 + x1954;
x887[x1948] = x1955;
float x1957 = x890[x1951];
float x1958 = x885[x1948];
float x1959 = x888[x1951];
float x1960 = x908[x1953];
float x1961 = x1957 + x1960;
x890[x1951] = x1961;
x1946 += 1;
x1944 += 1;
x1945 += 1;

}
// add_cartesian
int32_t x1969 = 0;
for(int x1970=0; x1970 < 150; x1970++) {
for(int x1971=0; x1971 < 150; x1971++) {
int32_t x1972 = x1969;
int32_t x1973 = x1972 + x1971;
float x1974 = x183[x1973];
float x1975 = x270[x1971];
float x1976 = x890[x1970];
float x1977 = x1975 * x1976;
float x1978 = x1974 + x1977;
x183[x1973] = x1978;

}
x1969 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x104,150,x890,1,1,x271,1);
// add_cartesian
int32_t x1987 = 0;
for(int x1988=0; x1988 < 150; x1988++) {
for(int x1989=0; x1989 < 150; x1989++) {
int32_t x1990 = x1987;
int32_t x1991 = x1990 + x1989;
float x1992 = x182[x1991];
float x1993 = x261[x1989];
float x1994 = x887[x1988];
float x1995 = x1993 * x1994;
float x1996 = x1992 + x1995;
x182[x1991] = x1996;

}
x1987 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x96,150,x887,1,1,x262,1);
for(int x2004=0; x2004 < 150; x2004++) {
float x2005 = x871[x2004];
float x2006 = x872[x2004];
float x2009 = x884[x2004];
float x2007 = 1.0f - x2006;
float x2008 = x2007 * x2006;
float x2010 = x2008 * x2009;
float x2011 = x2005 + x2010;
x871[x2004] = x2011;

}
int32_t x2015 = 0;
int32_t x2016 = 0;
int32_t x2017 = 0;
for(int x2018=0; x2018 < 150; x2018++) {
int32_t x2019 = x2015;
float x2020 = x853[x2019];
float x2021 = x836[x2019];
int32_t x2022 = x2016;
float x2023 = x95[x2022];
int32_t x2024 = x2017;
float x2025 = x871[x2024];
float x2026 = x2020 + x2025;
x853[x2019] = x2026;
float x2028 = x181[x2022];
float x2029 = x836[x2019];
float x2030 = x95[x2022];
float x2031 = x871[x2024];
float x2032 = x2028 + x2031;
x181[x2022] = x2032;
x2017 += 1;
x2015 += 1;
x2016 += 1;

}
int32_t x2039 = 0;
int32_t x2040 = 0;
int32_t x2041 = 0;
for(int x2042=0; x2042 < 150; x2042++) {
int32_t x2043 = x2039;
float x2044 = x832[x2043];
float x2045 = x830[x2043];
int32_t x2046 = x2040;
float x2047 = x833[x2046];
int32_t x2048 = x2041;
float x2049 = x853[x2048];
float x2050 = x2044 + x2049;
x832[x2043] = x2050;
float x2052 = x835[x2046];
float x2053 = x830[x2043];
float x2054 = x833[x2046];
float x2055 = x853[x2048];
float x2056 = x2052 + x2055;
x835[x2046] = x2056;
x2041 += 1;
x2039 += 1;
x2040 += 1;

}
// add_cartesian
int32_t x2064 = 0;
for(int x2065=0; x2065 < 150; x2065++) {
for(int x2066=0; x2066 < 150; x2066++) {
int32_t x2067 = x2064;
int32_t x2068 = x2067 + x2066;
float x2069 = x180[x2068];
float x2070 = x270[x2066];
float x2071 = x835[x2065];
float x2072 = x2070 * x2071;
float x2073 = x2069 + x2072;
x180[x2068] = x2073;

}
x2064 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x87,150,x835,1,1,x271,1);
// add_cartesian
int32_t x2082 = 0;
for(int x2083=0; x2083 < 150; x2083++) {
for(int x2084=0; x2084 < 150; x2084++) {
int32_t x2085 = x2082;
int32_t x2086 = x2085 + x2084;
float x2087 = x179[x2086];
float x2088 = x261[x2084];
float x2089 = x832[x2083];
float x2090 = x2088 * x2089;
float x2091 = x2087 + x2090;
x179[x2086] = x2091;

}
x2082 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x78,150,x832,1,1,x262,1);
}
};
x242(x265,x266,x2102);
};
x242(x256,x257,x2112);
} else {
float** x2139 = (float**)myMalloc(6 * sizeof(float*));;
x2139[0] = x236;
x2139[1] = x237;
x2139[2] = x238;
x2139[3] = x239;
x2139[4] = x240;
x2139[5] = x241;
function<void(float**)> x247 = x244;
function<void(float**)> x2122 = [&](float** x2123) {
float* x2124 = x2123[0];
float* x2125 = x2123[1];
float* x2126 = x2123[2];
float* x2127 = x2123[3];
float* x2128 = x2123[4];
float* x2129 = x2123[5];
float** x2130 = (float**)myMalloc(6 * sizeof(float*));;
x2130[0] = x2124;
x2130[1] = x2125;
x2130[2] = x2126;
x2130[3] = x2127;
x2130[4] = x2128;
x2130[5] = x2129;
x247(x2130);
};
x2122(x2139);
}
};
float* x233 = (float*)myMalloc(1 * sizeof(float));;
float* x234 = (float*)myMalloc(1 * sizeof(float));;
float* x235 = (float*)myMalloc(1 * sizeof(float));;
float** x2162 = (float**)myMalloc(6 * sizeof(float*));;
x2162[0] = x236;
x2162[1] = x237;
x2162[2] = x238;
x2162[3] = x239;
x2162[4] = x240;
x2162[5] = x241;
function<void(float**)> x2150 = [&](float** x2151) {
float* x2152 = x2151[0];
float* x2153 = x2151[1];
float* x2154 = x2151[2];
float* x2155 = x2151[3];
float* x2156 = x2151[4];
float* x2157 = x2151[5];
x2153[0] = 1.0f;
float x2159 = x2152[0];
x235[0] = x2159;
};
x242(0,x2150,x2162);
float x2171 = x235[0];
float x2172 = x222;
float x2173 = (float)x223;
float x2174 = x2172 * x2173;
int32_t x2175 = x223 + 1;
float x2176 = (float)x2175;
float x2177 = x2174 / x2176;
float x2178 = x2171 / x2176;
float x2179 = x2177 + x2178;
x222 = x2179;
for(int x2181=0; x2181 < 45000; x2181++) {
float x2182 = x173[x2181];
float x2183 = x2182;
float x2184 = x195[x2181];
float x2185 = x2183;
float x2186 = x2185 * x2185;
float x2187 = x2184 + x2186;
x195[x2181] = x2187;
float x2189 = x50[x2181];
float x2191 = x195[x2181];
float x2190 = 0.05f * x2185;
double x2192 = (double)x2191;
double x2193 = x2192 + 9.99999993922529E-9;
double x2194 = sqrt(x2193);
float x2195 = (float)x2194;
float x2196 = x2190 / x2195;
float x2197 = x2189 - x2196;
x50[x2181] = x2197;
x173[x2181] = 0.0f;

}
for(int x2202=0; x2202 < 150; x2202++) {
float x2203 = x174[x2202];
float x2204 = x2203;
float x2205 = x196[x2202];
float x2206 = x2204;
float x2207 = x2206 * x2206;
float x2208 = x2205 + x2207;
x196[x2202] = x2208;
float x2210 = x59[x2202];
float x2212 = x196[x2202];
float x2211 = 0.05f * x2206;
double x2213 = (double)x2212;
double x2214 = x2213 + 9.99999993922529E-9;
double x2215 = sqrt(x2214);
float x2216 = (float)x2215;
float x2217 = x2211 / x2216;
float x2218 = x2210 - x2217;
x59[x2202] = x2218;
x174[x2202] = 0.0f;

}
for(int x2223=0; x2223 < 45000; x2223++) {
float x2224 = x175[x2223];
float x2225 = x2224;
float x2226 = x197[x2223];
float x2227 = x2225;
float x2228 = x2227 * x2227;
float x2229 = x2226 + x2228;
x197[x2223] = x2229;
float x2231 = x60[x2223];
float x2233 = x197[x2223];
float x2232 = 0.05f * x2227;
double x2234 = (double)x2233;
double x2235 = x2234 + 9.99999993922529E-9;
double x2236 = sqrt(x2235);
float x2237 = (float)x2236;
float x2238 = x2232 / x2237;
float x2239 = x2231 - x2238;
x60[x2223] = x2239;
x175[x2223] = 0.0f;

}
for(int x2244=0; x2244 < 150; x2244++) {
float x2245 = x176[x2244];
float x2246 = x2245;
float x2247 = x198[x2244];
float x2248 = x2246;
float x2249 = x2248 * x2248;
float x2250 = x2247 + x2249;
x198[x2244] = x2250;
float x2252 = x68[x2244];
float x2254 = x198[x2244];
float x2253 = 0.05f * x2248;
double x2255 = (double)x2254;
double x2256 = x2255 + 9.99999993922529E-9;
double x2257 = sqrt(x2256);
float x2258 = (float)x2257;
float x2259 = x2253 / x2258;
float x2260 = x2252 - x2259;
x68[x2244] = x2260;
x176[x2244] = 0.0f;

}
for(int x2265=0; x2265 < 45000; x2265++) {
float x2266 = x177[x2265];
float x2267 = x2266;
float x2268 = x199[x2265];
float x2269 = x2267;
float x2270 = x2269 * x2269;
float x2271 = x2268 + x2270;
x199[x2265] = x2271;
float x2273 = x69[x2265];
float x2275 = x199[x2265];
float x2274 = 0.05f * x2269;
double x2276 = (double)x2275;
double x2277 = x2276 + 9.99999993922529E-9;
double x2278 = sqrt(x2277);
float x2279 = (float)x2278;
float x2280 = x2274 / x2279;
float x2281 = x2273 - x2280;
x69[x2265] = x2281;
x177[x2265] = 0.0f;

}
for(int x2286=0; x2286 < 150; x2286++) {
float x2287 = x178[x2286];
float x2288 = x2287;
float x2289 = x200[x2286];
float x2290 = x2288;
float x2291 = x2290 * x2290;
float x2292 = x2289 + x2291;
x200[x2286] = x2292;
float x2294 = x77[x2286];
float x2296 = x200[x2286];
float x2295 = 0.05f * x2290;
double x2297 = (double)x2296;
double x2298 = x2297 + 9.99999993922529E-9;
double x2299 = sqrt(x2298);
float x2300 = (float)x2299;
float x2301 = x2295 / x2300;
float x2302 = x2294 - x2301;
x77[x2286] = x2302;
x178[x2286] = 0.0f;

}
for(int x2307=0; x2307 < 22500; x2307++) {
float x2308 = x179[x2307];
float x2309 = x2308;
float x2310 = x201[x2307];
float x2311 = x2309;
float x2312 = x2311 * x2311;
float x2313 = x2310 + x2312;
x201[x2307] = x2313;
float x2315 = x78[x2307];
float x2317 = x201[x2307];
float x2316 = 0.05f * x2311;
double x2318 = (double)x2317;
double x2319 = x2318 + 9.99999993922529E-9;
double x2320 = sqrt(x2319);
float x2321 = (float)x2320;
float x2322 = x2316 / x2321;
float x2323 = x2315 - x2322;
x78[x2307] = x2323;
x179[x2307] = 0.0f;

}
for(int x2328=0; x2328 < 22500; x2328++) {
float x2329 = x180[x2328];
float x2330 = x2329;
float x2331 = x202[x2328];
float x2332 = x2330;
float x2333 = x2332 * x2332;
float x2334 = x2331 + x2333;
x202[x2328] = x2334;
float x2336 = x87[x2328];
float x2338 = x202[x2328];
float x2337 = 0.05f * x2332;
double x2339 = (double)x2338;
double x2340 = x2339 + 9.99999993922529E-9;
double x2341 = sqrt(x2340);
float x2342 = (float)x2341;
float x2343 = x2337 / x2342;
float x2344 = x2336 - x2343;
x87[x2328] = x2344;
x180[x2328] = 0.0f;

}
for(int x2349=0; x2349 < 150; x2349++) {
float x2350 = x181[x2349];
float x2351 = x2350;
float x2352 = x203[x2349];
float x2353 = x2351;
float x2354 = x2353 * x2353;
float x2355 = x2352 + x2354;
x203[x2349] = x2355;
float x2357 = x95[x2349];
float x2359 = x203[x2349];
float x2358 = 0.05f * x2353;
double x2360 = (double)x2359;
double x2361 = x2360 + 9.99999993922529E-9;
double x2362 = sqrt(x2361);
float x2363 = (float)x2362;
float x2364 = x2358 / x2363;
float x2365 = x2357 - x2364;
x95[x2349] = x2365;
x181[x2349] = 0.0f;

}
for(int x2370=0; x2370 < 22500; x2370++) {
float x2371 = x182[x2370];
float x2372 = x2371;
float x2373 = x204[x2370];
float x2374 = x2372;
float x2375 = x2374 * x2374;
float x2376 = x2373 + x2375;
x204[x2370] = x2376;
float x2378 = x96[x2370];
float x2380 = x204[x2370];
float x2379 = 0.05f * x2374;
double x2381 = (double)x2380;
double x2382 = x2381 + 9.99999993922529E-9;
double x2383 = sqrt(x2382);
float x2384 = (float)x2383;
float x2385 = x2379 / x2384;
float x2386 = x2378 - x2385;
x96[x2370] = x2386;
x182[x2370] = 0.0f;

}
for(int x2391=0; x2391 < 22500; x2391++) {
float x2392 = x183[x2391];
float x2393 = x2392;
float x2394 = x205[x2391];
float x2395 = x2393;
float x2396 = x2395 * x2395;
float x2397 = x2394 + x2396;
x205[x2391] = x2397;
float x2399 = x104[x2391];
float x2401 = x205[x2391];
float x2400 = 0.05f * x2395;
double x2402 = (double)x2401;
double x2403 = x2402 + 9.99999993922529E-9;
double x2404 = sqrt(x2403);
float x2405 = (float)x2404;
float x2406 = x2400 / x2405;
float x2407 = x2399 - x2406;
x104[x2391] = x2407;
x183[x2391] = 0.0f;

}
for(int x2412=0; x2412 < 22500; x2412++) {
float x2413 = x184[x2412];
float x2414 = x2413;
float x2415 = x206[x2412];
float x2416 = x2414;
float x2417 = x2416 * x2416;
float x2418 = x2415 + x2417;
x206[x2412] = x2418;
float x2420 = x112[x2412];
float x2422 = x206[x2412];
float x2421 = 0.05f * x2416;
double x2423 = (double)x2422;
double x2424 = x2423 + 9.99999993922529E-9;
double x2425 = sqrt(x2424);
float x2426 = (float)x2425;
float x2427 = x2421 / x2426;
float x2428 = x2420 - x2427;
x112[x2412] = x2428;
x184[x2412] = 0.0f;

}
for(int x2433=0; x2433 < 22500; x2433++) {
float x2434 = x185[x2433];
float x2435 = x2434;
float x2436 = x207[x2433];
float x2437 = x2435;
float x2438 = x2437 * x2437;
float x2439 = x2436 + x2438;
x207[x2433] = x2439;
float x2441 = x120[x2433];
float x2443 = x207[x2433];
float x2442 = 0.05f * x2437;
double x2444 = (double)x2443;
double x2445 = x2444 + 9.99999993922529E-9;
double x2446 = sqrt(x2445);
float x2447 = (float)x2446;
float x2448 = x2442 / x2447;
float x2449 = x2441 - x2448;
x120[x2433] = x2449;
x185[x2433] = 0.0f;

}
for(int x2454=0; x2454 < 150; x2454++) {
float x2455 = x186[x2454];
float x2456 = x2455;
float x2457 = x208[x2454];
float x2458 = x2456;
float x2459 = x2458 * x2458;
float x2460 = x2457 + x2459;
x208[x2454] = x2460;
float x2462 = x128[x2454];
float x2464 = x208[x2454];
float x2463 = 0.05f * x2458;
double x2465 = (double)x2464;
double x2466 = x2465 + 9.99999993922529E-9;
double x2467 = sqrt(x2466);
float x2468 = (float)x2467;
float x2469 = x2463 / x2468;
float x2470 = x2462 - x2469;
x128[x2454] = x2470;
x186[x2454] = 0.0f;

}
for(int x2475=0; x2475 < 22500; x2475++) {
float x2476 = x187[x2475];
float x2477 = x2476;
float x2478 = x209[x2475];
float x2479 = x2477;
float x2480 = x2479 * x2479;
float x2481 = x2478 + x2480;
x209[x2475] = x2481;
float x2483 = x129[x2475];
float x2485 = x209[x2475];
float x2484 = 0.05f * x2479;
double x2486 = (double)x2485;
double x2487 = x2486 + 9.99999993922529E-9;
double x2488 = sqrt(x2487);
float x2489 = (float)x2488;
float x2490 = x2484 / x2489;
float x2491 = x2483 - x2490;
x129[x2475] = x2491;
x187[x2475] = 0.0f;

}
for(int x2496=0; x2496 < 22500; x2496++) {
float x2497 = x188[x2496];
float x2498 = x2497;
float x2499 = x210[x2496];
float x2500 = x2498;
float x2501 = x2500 * x2500;
float x2502 = x2499 + x2501;
x210[x2496] = x2502;
float x2504 = x137[x2496];
float x2506 = x210[x2496];
float x2505 = 0.05f * x2500;
double x2507 = (double)x2506;
double x2508 = x2507 + 9.99999993922529E-9;
double x2509 = sqrt(x2508);
float x2510 = (float)x2509;
float x2511 = x2505 / x2510;
float x2512 = x2504 - x2511;
x137[x2496] = x2512;
x188[x2496] = 0.0f;

}
for(int x2517=0; x2517 < 150; x2517++) {
float x2518 = x189[x2517];
float x2519 = x2518;
float x2520 = x211[x2517];
float x2521 = x2519;
float x2522 = x2521 * x2521;
float x2523 = x2520 + x2522;
x211[x2517] = x2523;
float x2525 = x145[x2517];
float x2527 = x211[x2517];
float x2526 = 0.05f * x2521;
double x2528 = (double)x2527;
double x2529 = x2528 + 9.99999993922529E-9;
double x2530 = sqrt(x2529);
float x2531 = (float)x2530;
float x2532 = x2526 / x2531;
float x2533 = x2525 - x2532;
x145[x2517] = x2533;
x189[x2517] = 0.0f;

}
for(int x2538=0; x2538 < 22500; x2538++) {
float x2539 = x190[x2538];
float x2540 = x2539;
float x2541 = x212[x2538];
float x2542 = x2540;
float x2543 = x2542 * x2542;
float x2544 = x2541 + x2543;
x212[x2538] = x2544;
float x2546 = x146[x2538];
float x2548 = x212[x2538];
float x2547 = 0.05f * x2542;
double x2549 = (double)x2548;
double x2550 = x2549 + 9.99999993922529E-9;
double x2551 = sqrt(x2550);
float x2552 = (float)x2551;
float x2553 = x2547 / x2552;
float x2554 = x2546 - x2553;
x146[x2538] = x2554;
x190[x2538] = 0.0f;

}
for(int x2559=0; x2559 < 22500; x2559++) {
float x2560 = x191[x2559];
float x2561 = x2560;
float x2562 = x213[x2559];
float x2563 = x2561;
float x2564 = x2563 * x2563;
float x2565 = x2562 + x2564;
x213[x2559] = x2565;
float x2567 = x154[x2559];
float x2569 = x213[x2559];
float x2568 = 0.05f * x2563;
double x2570 = (double)x2569;
double x2571 = x2570 + 9.99999993922529E-9;
double x2572 = sqrt(x2571);
float x2573 = (float)x2572;
float x2574 = x2568 / x2573;
float x2575 = x2567 - x2574;
x154[x2559] = x2575;
x191[x2559] = 0.0f;

}
for(int x2580=0; x2580 < 150; x2580++) {
float x2581 = x192[x2580];
float x2582 = x2581;
float x2583 = x214[x2580];
float x2584 = x2582;
float x2585 = x2584 * x2584;
float x2586 = x2583 + x2585;
x214[x2580] = x2586;
float x2588 = x162[x2580];
float x2590 = x214[x2580];
float x2589 = 0.05f * x2584;
double x2591 = (double)x2590;
double x2592 = x2591 + 9.99999993922529E-9;
double x2593 = sqrt(x2592);
float x2594 = (float)x2593;
float x2595 = x2589 / x2594;
float x2596 = x2588 - x2595;
x162[x2580] = x2596;
x192[x2580] = 0.0f;

}
for(int x2601=0; x2601 < 750; x2601++) {
float x2602 = x193[x2601];
float x2603 = x2602;
float x2604 = x215[x2601];
float x2605 = x2603;
float x2606 = x2605 * x2605;
float x2607 = x2604 + x2606;
x215[x2601] = x2607;
float x2609 = x163[x2601];
float x2611 = x215[x2601];
float x2610 = 0.05f * x2605;
double x2612 = (double)x2611;
double x2613 = x2612 + 9.99999993922529E-9;
double x2614 = sqrt(x2613);
float x2615 = (float)x2614;
float x2616 = x2610 / x2615;
float x2617 = x2609 - x2616;
x163[x2601] = x2617;
x193[x2601] = 0.0f;

}
for(int x2622=0; x2622 < 5; x2622++) {
float x2623 = x194[x2622];
float x2624 = x2623;
float x2625 = x216[x2622];
float x2626 = x2624;
float x2627 = x2626 * x2626;
float x2628 = x2625 + x2627;
x216[x2622] = x2628;
float x2630 = x172[x2622];
float x2632 = x216[x2622];
float x2631 = 0.05f * x2626;
double x2633 = (double)x2632;
double x2634 = x2633 + 9.99999993922529E-9;
double x2635 = sqrt(x2634);
float x2636 = (float)x2635;
float x2637 = x2631 / x2636;
float x2638 = x2630 - x2637;
x172[x2622] = x2638;
x194[x2622] = 0.0f;

}
int64_t x2643 = (long)mallocAddr;
int64_t x2644 = x2643 - x218;
memset((void*)x218, 0, x2644);
mallocAddr = (void*)x218;

}
float x2649 = x222;
double x2650 = (double)x2649;
x217[x221] = x2650;
double x2652 = ((double)clock() / CLOCKS_PER_SEC);
double x2653 = x2652 - x219;
printf("epoc %d, average_loss %f, time %lf\n",x221,x2649,x2653);

}
double x2657 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2661 = (long)fopen(x0, "w");
fprintf((FILE *)x2661, "unit: %s\n", "1 epoch");
for(int x2663=0; x2663 < 6; x2663++) {
double x2664 = x217[x2663];
fprintf((FILE *)x2661, "%lf\n", x2664);

}
double x2658 = x219 - x2;
double x2659 = x2657 - x219;
double x2660 = x2659 / 6.0;
fprintf((FILE *)x2661, "run time: %lf %lf\n", x2658, x2660);
fclose((FILE*)x2661);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

