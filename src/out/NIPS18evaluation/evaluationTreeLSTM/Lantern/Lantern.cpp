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
float** x2127 = (float**)myMalloc(6 * sizeof(float*));;
x2127[0] = x236;
x2127[1] = x237;
x2127[2] = x238;
x2127[3] = x239;
x2127[4] = x240;
x2127[5] = x241;
function<void(float**)> x247 = x244;
function<void(float**)> x505 = [&](float** x506) {
float* x507 = x506[0];
float* x508 = x506[1];
float* x509 = x506[2];
float* x510 = x506[3];
float* x511 = x506[4];
float* x512 = x506[5];
float** x513 = (float**)myMalloc(6 * sizeof(float*));;
x513[0] = x507;
x513[1] = x508;
x513[2] = x509;
x513[3] = x510;
x513[4] = x511;
x513[5] = x512;
x247(x513);
};
function<void(float**)> x497 = [&](float** x498) {
float* x499 = x498[0];
float* x500 = x498[1];
float* x501 = x498[2];
float* x502 = x498[3];
float* x503 = x498[4];
float* x504 = x498[5];
float** x522 = (float**)myMalloc(6 * sizeof(float*));;
x522[0] = x499;
x522[1] = x500;
x522[2] = x501;
x522[3] = x502;
x522[4] = x503;
x522[5] = x504;
x505(x522);
};
function<void(float**)> x1332 = [&](float** x1333) {
float* x1334 = x1333[0];
float* x1335 = x1333[1];
float* x1336 = x1333[2];
float* x1337 = x1333[3];
float* x1338 = x1333[4];
float* x1339 = x1333[5];
float** x1340 = (float**)myMalloc(6 * sizeof(float*));;
x1340[0] = x1334;
x1340[1] = x1335;
x1340[2] = x1336;
x1340[3] = x1337;
x1340[4] = x1338;
x1340[5] = x1339;
x247(x1340);
};
function<void(float**)> x1324 = [&](float** x1325) {
float* x1326 = x1325[0];
float* x1327 = x1325[1];
float* x1328 = x1325[2];
float* x1329 = x1325[3];
float* x1330 = x1325[4];
float* x1331 = x1325[5];
float** x1349 = (float**)myMalloc(6 * sizeof(float*));;
x1349[0] = x1326;
x1349[1] = x1327;
x1349[2] = x1328;
x1349[3] = x1329;
x1349[4] = x1330;
x1349[5] = x1331;
x1332(x1349);
};
function<void(float**)> x257 = [&](float** x258) {
float* x259 = x258[0];
float* x260 = x258[1];
float* x261 = x258[2];
float* x262 = x258[3];
float* x263 = x258[4];
float* x264 = x258[5];
int32_t x265 = x232[x246];
float** x2117 = (float**)myMalloc(6 * sizeof(float*));;
x2117[0] = x236;
x2117[1] = x237;
x2117[2] = x238;
x2117[3] = x239;
x2117[4] = x240;
x2117[5] = x241;
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
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x50,300,x281,1,0,x284,1);
float* x286 = (float*)myMalloc(150 * sizeof(float));;
float* x287 = (float*)myMalloc(150 * sizeof(float));;
int32_t x288 = 0;
int32_t x289 = 0;
int32_t x290 = 0;
for(int x292=0; x292 < 150; x292++) {
int32_t x293 = x288;
int32_t x294 = x289;
float x295 = x284[x294];
int32_t x296 = x290;
float x297 = x59[x296];
float x298 = x295 + x297;
x287[x293] = x298;
x288 += 1;
x289 += 1;
x290 += 1;

}
float* x305 = (float*)myMalloc(150 * sizeof(float));;
float* x306 = (float*)myMalloc(150 * sizeof(float));;
for(int x307=0; x307 < 150; x307++) {
float x308 = x287[x307];
float x309 = -1.0f * x308;
double x310 = (double)x309;
double x311 = exp(x310);
float x312 = (float)x311;
float x313 = x312 + 1.0f;
float x314 = 1.0f / x313;
x306[x307] = x314;

}
float* x318 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 300), WrappedArray(300)
float* x320 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x60,300,x281,1,0,x320,1);
float* x322 = (float*)myMalloc(150 * sizeof(float));;
float* x323 = (float*)myMalloc(150 * sizeof(float));;
int32_t x324 = 0;
int32_t x325 = 0;
int32_t x326 = 0;
for(int x327=0; x327 < 150; x327++) {
int32_t x328 = x324;
int32_t x329 = x325;
float x330 = x320[x329];
int32_t x331 = x326;
float x332 = x68[x331];
float x333 = x330 + x332;
x323[x328] = x333;
x324 += 1;
x325 += 1;
x326 += 1;

}
float* x340 = (float*)myMalloc(150 * sizeof(float));;
float* x341 = (float*)myMalloc(150 * sizeof(float));;
for(int x342=0; x342 < 150; x342++) {
float x343 = x323[x342];
float x344 = -1.0f * x343;
double x345 = (double)x344;
double x346 = exp(x345);
float x347 = (float)x346;
float x348 = x347 + 1.0f;
float x349 = 1.0f / x348;
x341[x342] = x349;

}
float* x353 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 300), WrappedArray(300)
float* x355 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x69,300,x281,1,0,x355,1);
float* x357 = (float*)myMalloc(150 * sizeof(float));;
float* x358 = (float*)myMalloc(150 * sizeof(float));;
int32_t x359 = 0;
int32_t x360 = 0;
int32_t x361 = 0;
for(int x362=0; x362 < 150; x362++) {
int32_t x363 = x359;
int32_t x364 = x360;
float x365 = x355[x364];
int32_t x366 = x361;
float x367 = x77[x366];
float x368 = x365 + x367;
x358[x363] = x368;
x359 += 1;
x360 += 1;
x361 += 1;

}
float* x375 = (float*)myMalloc(150 * sizeof(float));;
float* x376 = (float*)myMalloc(150 * sizeof(float));;
for(int x377=0; x377 < 150; x377++) {
float x378 = x358[x377];
double x379 = (double)x378;
double x380 = tanh(x379);
float x381 = (float)x380;
x376[x377] = x381;

}
float* x385 = (float*)myMalloc(150 * sizeof(float));;
float* x386 = (float*)myMalloc(150 * sizeof(float));;
int32_t x387 = 0;
int32_t x388 = 0;
int32_t x389 = 0;
for(int x390=0; x390 < 150; x390++) {
int32_t x391 = x387;
int32_t x392 = x388;
float x393 = x306[x392];
int32_t x394 = x389;
float x395 = x376[x394];
float x396 = x393 * x395;
x386[x391] = x396;
x387 += 1;
x388 += 1;
x389 += 1;

}
float* x403 = (float*)myMalloc(150 * sizeof(float));;
float* x404 = (float*)myMalloc(150 * sizeof(float));;
for(int x405=0; x405 < 150; x405++) {
float x406 = x386[x405];
double x407 = (double)x406;
double x408 = tanh(x407);
float x409 = (float)x408;
x404[x405] = x409;

}
float* x413 = (float*)myMalloc(150 * sizeof(float));;
float* x414 = (float*)myMalloc(150 * sizeof(float));;
int32_t x415 = 0;
int32_t x416 = 0;
int32_t x417 = 0;
for(int x418=0; x418 < 150; x418++) {
int32_t x419 = x415;
int32_t x420 = x416;
float x421 = x341[x420];
int32_t x422 = x417;
float x423 = x404[x422];
float x424 = x421 * x423;
x414[x419] = x424;
x415 += 1;
x416 += 1;
x417 += 1;

}
float* x431 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(5, 150), List(150)
float* x433 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x163,150,x414,1,0,x433,1);
float* x435 = (float*)myMalloc(5 * sizeof(float));;
float* x436 = (float*)myMalloc(5 * sizeof(float));;
int32_t x437 = 0;
int32_t x438 = 0;
int32_t x439 = 0;
for(int x441=0; x441 < 5; x441++) {
int32_t x442 = x437;
int32_t x443 = x438;
float x444 = x433[x443];
int32_t x445 = x439;
float x446 = x172[x445];
float x447 = x444 + x446;
x436[x442] = x447;
x437 += 1;
x438 += 1;
x439 += 1;

}
float* x454 = (float*)myMalloc(5 * sizeof(float));;
float x455 = -3.4028235E38f;
for(int x456=0; x456 < 5; x456++) {
float x457 = x455;
float x458 = x436[x456];
bool x459 = x458 > x457;
float x460;
if (x459) {
x460 = x458;
} else {
x460 = x457;
}
x455 = x460;

}
float x464 = x455;
float x465 = 0.0f;
for(int x466=0; x466 < 5; x466++) {
float x467 = x465;
float x468 = x436[x466];
float x469 = x455;
float x470 = x468 - x469;
double x471 = (double)x470;
double x472 = exp(x471);
float x473 = (float)x472;
float x474 = x467 + x473;
x465 = x474;

}
float x478 = x465;
float* x483 = (float*)myMalloc(5 * sizeof(float));;
double x479 = (double)x478;
double x480 = log(x479);
float x481 = (float)x480;
float x482 = x464 + x481;
for(int x484=0; x484 < 5; x484++) {
float x485 = x436[x484];
float x486 = x485 - x482;
x483[x484] = x486;

}
float* x490 = (float*)myMalloc(5 * sizeof(float));;
int32_t x491 = x226[x246];
float x492 = x483[x491];
float* x494 = (float*)myMalloc(1 * sizeof(float));;
float x493 = -1.0f * x492;
x494[0] = x493;
float* x496 = (float*)myMalloc(1 * sizeof(float));;
float** x531 = (float**)myMalloc(6 * sizeof(float*));;
x531[0] = x494;
x531[1] = x496;
x531[2] = x414;
x531[3] = x431;
x531[4] = x386;
x531[5] = x403;
x497(x531);
float x539 = x490[x491];
float x540 = x496[0];
float x541 = -1.0f * x540;
float x542 = x539 + x541;
x490[x491] = x542;
float x544 = 0.0f;
for(int x545=0; x545 < 5; x545++) {
float x546 = x544;
float x547 = x490[x545];
float x548 = x546 + x547;
x544 = x548;

}
float x552 = x544;
float* x553 = (float*)myMalloc(1 * sizeof(float));;
x553[0] = x552;
float x555 = x553[0];
for(int x556=0; x556 < 5; x556++) {
float x557 = x454[x556];
float x558 = x490[x556];
float x559 = x483[x556];
double x560 = (double)x559;
double x561 = exp(x560);
float x562 = (float)x561;
float x563 = x562 * x555;
float x564 = x558 - x563;
float x565 = x557 + x564;
x454[x556] = x565;

}
int32_t x569 = 0;
int32_t x570 = 0;
int32_t x571 = 0;
for(int x572=0; x572 < 5; x572++) {
int32_t x573 = x569;
float x574 = x435[x573];
float x575 = x433[x573];
int32_t x576 = x570;
float x577 = x172[x576];
int32_t x578 = x571;
float x579 = x454[x578];
float x580 = x574 + x579;
x435[x573] = x580;
float x582 = x194[x576];
float x583 = x433[x573];
float x584 = x172[x576];
float x585 = x454[x578];
float x586 = x582 + x585;
x194[x576] = x586;
x571 += 1;
x569 += 1;
x570 += 1;

}
// add_cartesian
int32_t x594 = 0;
for(int x595=0; x595 < 5; x595++) {
for(int x596=0; x596 < 150; x596++) {
int32_t x597 = x594;
int32_t x598 = x597 + x596;
float x599 = x193[x598];
float x600 = x414[x596];
float x601 = x435[x595];
float x602 = x600 * x601;
float x603 = x599 + x602;
x193[x598] = x603;

}
x594 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x163,150,x435,1,1,x431,1);
int32_t x611 = 0;
int32_t x612 = 0;
int32_t x613 = 0;
for(int x614=0; x614 < 150; x614++) {
int32_t x615 = x611;
float x616 = x353[x615];
float x617 = x341[x615];
int32_t x618 = x612;
float x619 = x404[x618];
int32_t x620 = x613;
float x621 = x431[x620];
float x622 = x621 * x619;
float x623 = x616 + x622;
x353[x615] = x623;
float x625 = x413[x618];
float x626 = x341[x615];
float x627 = x404[x618];
float x628 = x431[x620];
float x629 = x628 * x626;
float x630 = x625 + x629;
x413[x618] = x630;
x613 += 1;
x611 += 1;
x612 += 1;

}
for(int x637=0; x637 < 150; x637++) {
float x638 = x403[x637];
float x639 = x404[x637];
float x642 = x413[x637];
float x640 = x639 * x639;
float x641 = 1.0f - x640;
float x643 = x641 * x642;
float x644 = x638 + x643;
x403[x637] = x644;

}
int32_t x648 = 0;
int32_t x649 = 0;
int32_t x650 = 0;
for(int x651=0; x651 < 150; x651++) {
int32_t x652 = x648;
float x653 = x318[x652];
float x654 = x306[x652];
int32_t x655 = x649;
float x656 = x376[x655];
int32_t x657 = x650;
float x658 = x403[x657];
float x659 = x658 * x656;
float x660 = x653 + x659;
x318[x652] = x660;
float x662 = x385[x655];
float x663 = x306[x652];
float x664 = x376[x655];
float x665 = x403[x657];
float x666 = x665 * x663;
float x667 = x662 + x666;
x385[x655] = x667;
x650 += 1;
x648 += 1;
x649 += 1;

}
for(int x674=0; x674 < 150; x674++) {
float x675 = x375[x674];
float x676 = x376[x674];
float x679 = x385[x674];
float x677 = x676 * x676;
float x678 = 1.0f - x677;
float x680 = x678 * x679;
float x681 = x675 + x680;
x375[x674] = x681;

}
int32_t x685 = 0;
int32_t x686 = 0;
int32_t x687 = 0;
for(int x688=0; x688 < 150; x688++) {
int32_t x689 = x685;
float x690 = x357[x689];
float x691 = x355[x689];
int32_t x692 = x686;
float x693 = x77[x692];
int32_t x694 = x687;
float x695 = x375[x694];
float x696 = x690 + x695;
x357[x689] = x696;
float x698 = x178[x692];
float x699 = x355[x689];
float x700 = x77[x692];
float x701 = x375[x694];
float x702 = x698 + x701;
x178[x692] = x702;
x687 += 1;
x685 += 1;
x686 += 1;

}
// add_cartesian
int32_t x710 = 0;
for(int x711=0; x711 < 150; x711++) {
for(int x712=0; x712 < 300; x712++) {
int32_t x713 = x710;
int32_t x714 = x713 + x712;
float x715 = x177[x714];
float x716 = x281[x712];
float x717 = x357[x711];
float x718 = x716 * x717;
float x719 = x715 + x718;
x177[x714] = x719;

}
x710 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x69,300,x357,1,1,x282,1);
for(int x727=0; x727 < 150; x727++) {
float x728 = x340[x727];
float x729 = x341[x727];
float x732 = x353[x727];
float x730 = 1.0f - x729;
float x731 = x730 * x729;
float x733 = x731 * x732;
float x734 = x728 + x733;
x340[x727] = x734;

}
int32_t x738 = 0;
int32_t x739 = 0;
int32_t x740 = 0;
for(int x741=0; x741 < 150; x741++) {
int32_t x742 = x738;
float x743 = x322[x742];
float x744 = x320[x742];
int32_t x745 = x739;
float x746 = x68[x745];
int32_t x747 = x740;
float x748 = x340[x747];
float x749 = x743 + x748;
x322[x742] = x749;
float x751 = x176[x745];
float x752 = x320[x742];
float x753 = x68[x745];
float x754 = x340[x747];
float x755 = x751 + x754;
x176[x745] = x755;
x740 += 1;
x738 += 1;
x739 += 1;

}
// add_cartesian
int32_t x763 = 0;
for(int x764=0; x764 < 150; x764++) {
for(int x765=0; x765 < 300; x765++) {
int32_t x766 = x763;
int32_t x767 = x766 + x765;
float x768 = x175[x767];
float x769 = x281[x765];
float x770 = x322[x764];
float x771 = x769 * x770;
float x772 = x768 + x771;
x175[x767] = x772;

}
x763 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x60,300,x322,1,1,x282,1);
for(int x780=0; x780 < 150; x780++) {
float x781 = x305[x780];
float x782 = x306[x780];
float x785 = x318[x780];
float x783 = 1.0f - x782;
float x784 = x783 * x782;
float x786 = x784 * x785;
float x787 = x781 + x786;
x305[x780] = x787;

}
int32_t x791 = 0;
int32_t x792 = 0;
int32_t x793 = 0;
for(int x794=0; x794 < 150; x794++) {
int32_t x795 = x791;
float x796 = x286[x795];
float x797 = x284[x795];
int32_t x798 = x792;
float x799 = x59[x798];
int32_t x800 = x793;
float x801 = x305[x800];
float x802 = x796 + x801;
x286[x795] = x802;
float x804 = x174[x798];
float x805 = x284[x795];
float x806 = x59[x798];
float x807 = x305[x800];
float x808 = x804 + x807;
x174[x798] = x808;
x793 += 1;
x791 += 1;
x792 += 1;

}
// add_cartesian
int32_t x816 = 0;
for(int x817=0; x817 < 150; x817++) {
for(int x818=0; x818 < 300; x818++) {
int32_t x819 = x816;
int32_t x820 = x819 + x818;
float x821 = x173[x820];
float x822 = x281[x818];
float x823 = x286[x817];
float x824 = x822 * x823;
float x825 = x821 + x824;
x173[x820] = x825;

}
x816 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x50,300,x286,1,1,x282,1);
} else {
// dot: List(150, 150), WrappedArray(150)
float* x835 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x78,150,x261,1,0,x835,1);
float* x837 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x839 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x87,150,x270,1,0,x839,1);
float* x841 = (float*)myMalloc(150 * sizeof(float));;
float* x842 = (float*)myMalloc(150 * sizeof(float));;
int32_t x843 = 0;
int32_t x844 = 0;
int32_t x845 = 0;
for(int x846=0; x846 < 150; x846++) {
int32_t x847 = x843;
int32_t x848 = x844;
float x849 = x835[x848];
int32_t x850 = x845;
float x851 = x839[x850];
float x852 = x849 + x851;
x842[x847] = x852;
x843 += 1;
x844 += 1;
x845 += 1;

}
float* x859 = (float*)myMalloc(150 * sizeof(float));;
float* x860 = (float*)myMalloc(150 * sizeof(float));;
int32_t x861 = 0;
int32_t x862 = 0;
int32_t x863 = 0;
for(int x864=0; x864 < 150; x864++) {
int32_t x865 = x861;
int32_t x866 = x862;
float x867 = x842[x866];
int32_t x868 = x863;
float x869 = x95[x868];
float x870 = x867 + x869;
x860[x865] = x870;
x861 += 1;
x862 += 1;
x863 += 1;

}
float* x877 = (float*)myMalloc(150 * sizeof(float));;
float* x878 = (float*)myMalloc(150 * sizeof(float));;
for(int x879=0; x879 < 150; x879++) {
float x880 = x860[x879];
float x881 = -1.0f * x880;
double x882 = (double)x881;
double x883 = exp(x882);
float x884 = (float)x883;
float x885 = x884 + 1.0f;
float x886 = 1.0f / x885;
x878[x879] = x886;

}
float* x890 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x892 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x96,150,x261,1,0,x892,1);
float* x894 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x896 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x104,150,x270,1,0,x896,1);
float* x898 = (float*)myMalloc(150 * sizeof(float));;
float* x899 = (float*)myMalloc(150 * sizeof(float));;
int32_t x900 = 0;
int32_t x901 = 0;
int32_t x902 = 0;
for(int x903=0; x903 < 150; x903++) {
int32_t x904 = x900;
int32_t x905 = x901;
float x906 = x892[x905];
int32_t x907 = x902;
float x908 = x896[x907];
float x909 = x906 + x908;
x899[x904] = x909;
x900 += 1;
x901 += 1;
x902 += 1;

}
float* x916 = (float*)myMalloc(150 * sizeof(float));;
float* x917 = (float*)myMalloc(150 * sizeof(float));;
int32_t x918 = 0;
int32_t x919 = 0;
int32_t x920 = 0;
for(int x921=0; x921 < 150; x921++) {
int32_t x922 = x918;
int32_t x923 = x919;
float x924 = x899[x923];
int32_t x925 = x920;
float x926 = x128[x925];
float x927 = x924 + x926;
x917[x922] = x927;
x918 += 1;
x919 += 1;
x920 += 1;

}
float* x934 = (float*)myMalloc(150 * sizeof(float));;
float* x935 = (float*)myMalloc(150 * sizeof(float));;
for(int x936=0; x936 < 150; x936++) {
float x937 = x917[x936];
float x938 = -1.0f * x937;
double x939 = (double)x938;
double x940 = exp(x939);
float x941 = (float)x940;
float x942 = x941 + 1.0f;
float x943 = 1.0f / x942;
x935[x936] = x943;

}
float* x947 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x949 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x112,150,x261,1,0,x949,1);
float* x951 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x953 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x120,150,x270,1,0,x953,1);
float* x955 = (float*)myMalloc(150 * sizeof(float));;
float* x956 = (float*)myMalloc(150 * sizeof(float));;
int32_t x957 = 0;
int32_t x958 = 0;
int32_t x959 = 0;
for(int x960=0; x960 < 150; x960++) {
int32_t x961 = x957;
int32_t x962 = x958;
float x963 = x949[x962];
int32_t x964 = x959;
float x965 = x953[x964];
float x966 = x963 + x965;
x956[x961] = x966;
x957 += 1;
x958 += 1;
x959 += 1;

}
float* x973 = (float*)myMalloc(150 * sizeof(float));;
float* x974 = (float*)myMalloc(150 * sizeof(float));;
int32_t x975 = 0;
int32_t x976 = 0;
int32_t x977 = 0;
for(int x978=0; x978 < 150; x978++) {
int32_t x979 = x975;
int32_t x980 = x976;
float x981 = x956[x980];
int32_t x982 = x977;
float x983 = x128[x982];
float x984 = x981 + x983;
x974[x979] = x984;
x975 += 1;
x976 += 1;
x977 += 1;

}
float* x991 = (float*)myMalloc(150 * sizeof(float));;
float* x992 = (float*)myMalloc(150 * sizeof(float));;
for(int x993=0; x993 < 150; x993++) {
float x994 = x974[x993];
float x995 = -1.0f * x994;
double x996 = (double)x995;
double x997 = exp(x996);
float x998 = (float)x997;
float x999 = x998 + 1.0f;
float x1000 = 1.0f / x999;
x992[x993] = x1000;

}
float* x1004 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1006 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x129,150,x261,1,0,x1006,1);
float* x1008 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1010 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x137,150,x270,1,0,x1010,1);
float* x1012 = (float*)myMalloc(150 * sizeof(float));;
float* x1013 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1014 = 0;
int32_t x1015 = 0;
int32_t x1016 = 0;
for(int x1017=0; x1017 < 150; x1017++) {
int32_t x1018 = x1014;
int32_t x1019 = x1015;
float x1020 = x1006[x1019];
int32_t x1021 = x1016;
float x1022 = x1010[x1021];
float x1023 = x1020 + x1022;
x1013[x1018] = x1023;
x1014 += 1;
x1015 += 1;
x1016 += 1;

}
float* x1030 = (float*)myMalloc(150 * sizeof(float));;
float* x1031 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1032 = 0;
int32_t x1033 = 0;
int32_t x1034 = 0;
for(int x1035=0; x1035 < 150; x1035++) {
int32_t x1036 = x1032;
int32_t x1037 = x1033;
float x1038 = x1013[x1037];
int32_t x1039 = x1034;
float x1040 = x145[x1039];
float x1041 = x1038 + x1040;
x1031[x1036] = x1041;
x1032 += 1;
x1033 += 1;
x1034 += 1;

}
float* x1048 = (float*)myMalloc(150 * sizeof(float));;
float* x1049 = (float*)myMalloc(150 * sizeof(float));;
for(int x1050=0; x1050 < 150; x1050++) {
float x1051 = x1031[x1050];
float x1052 = -1.0f * x1051;
double x1053 = (double)x1052;
double x1054 = exp(x1053);
float x1055 = (float)x1054;
float x1056 = x1055 + 1.0f;
float x1057 = 1.0f / x1056;
x1049[x1050] = x1057;

}
float* x1061 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1063 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x146,150,x261,1,0,x1063,1);
float* x1065 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x1067 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x154,150,x270,1,0,x1067,1);
float* x1069 = (float*)myMalloc(150 * sizeof(float));;
float* x1070 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1071 = 0;
int32_t x1072 = 0;
int32_t x1073 = 0;
for(int x1074=0; x1074 < 150; x1074++) {
int32_t x1075 = x1071;
int32_t x1076 = x1072;
float x1077 = x1063[x1076];
int32_t x1078 = x1073;
float x1079 = x1067[x1078];
float x1080 = x1077 + x1079;
x1070[x1075] = x1080;
x1071 += 1;
x1072 += 1;
x1073 += 1;

}
float* x1087 = (float*)myMalloc(150 * sizeof(float));;
float* x1088 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1089 = 0;
int32_t x1090 = 0;
int32_t x1091 = 0;
for(int x1092=0; x1092 < 150; x1092++) {
int32_t x1093 = x1089;
int32_t x1094 = x1090;
float x1095 = x1070[x1094];
int32_t x1096 = x1091;
float x1097 = x162[x1096];
float x1098 = x1095 + x1097;
x1088[x1093] = x1098;
x1089 += 1;
x1090 += 1;
x1091 += 1;

}
float* x1105 = (float*)myMalloc(150 * sizeof(float));;
float* x1106 = (float*)myMalloc(150 * sizeof(float));;
for(int x1107=0; x1107 < 150; x1107++) {
float x1108 = x1088[x1107];
double x1109 = (double)x1108;
double x1110 = tanh(x1109);
float x1111 = (float)x1110;
x1106[x1107] = x1111;

}
float* x1115 = (float*)myMalloc(150 * sizeof(float));;
float* x1116 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1117 = 0;
int32_t x1118 = 0;
int32_t x1119 = 0;
for(int x1120=0; x1120 < 150; x1120++) {
int32_t x1121 = x1117;
int32_t x1122 = x1118;
float x1123 = x878[x1122];
int32_t x1124 = x1119;
float x1125 = x1106[x1124];
float x1126 = x1123 * x1125;
x1116[x1121] = x1126;
x1117 += 1;
x1118 += 1;
x1119 += 1;

}
float* x1133 = (float*)myMalloc(150 * sizeof(float));;
float* x1134 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1135 = 0;
int32_t x1136 = 0;
int32_t x1137 = 0;
for(int x1138=0; x1138 < 150; x1138++) {
int32_t x1139 = x1135;
int32_t x1140 = x1136;
float x1141 = x935[x1140];
int32_t x1142 = x1137;
float x1143 = x263[x1142];
float x1144 = x1141 * x1143;
x1134[x1139] = x1144;
x1135 += 1;
x1136 += 1;
x1137 += 1;

}
float* x1151 = (float*)myMalloc(150 * sizeof(float));;
float* x1152 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1153 = 0;
int32_t x1154 = 0;
int32_t x1155 = 0;
for(int x1156=0; x1156 < 150; x1156++) {
int32_t x1157 = x1153;
int32_t x1158 = x1154;
float x1159 = x1116[x1158];
int32_t x1160 = x1155;
float x1161 = x1134[x1160];
float x1162 = x1159 + x1161;
x1152[x1157] = x1162;
x1153 += 1;
x1154 += 1;
x1155 += 1;

}
float* x1169 = (float*)myMalloc(150 * sizeof(float));;
float* x1170 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1171 = 0;
int32_t x1172 = 0;
int32_t x1173 = 0;
for(int x1174=0; x1174 < 150; x1174++) {
int32_t x1175 = x1171;
int32_t x1176 = x1172;
float x1177 = x992[x1176];
int32_t x1178 = x1173;
float x1179 = x272[x1178];
float x1180 = x1177 * x1179;
x1170[x1175] = x1180;
x1171 += 1;
x1172 += 1;
x1173 += 1;

}
float* x1187 = (float*)myMalloc(150 * sizeof(float));;
float* x1188 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1189 = 0;
int32_t x1190 = 0;
int32_t x1191 = 0;
for(int x1192=0; x1192 < 150; x1192++) {
int32_t x1193 = x1189;
int32_t x1194 = x1190;
float x1195 = x1152[x1194];
int32_t x1196 = x1191;
float x1197 = x1170[x1196];
float x1198 = x1195 + x1197;
x1188[x1193] = x1198;
x1189 += 1;
x1190 += 1;
x1191 += 1;

}
float* x1205 = (float*)myMalloc(150 * sizeof(float));;
float* x1206 = (float*)myMalloc(150 * sizeof(float));;
for(int x1207=0; x1207 < 150; x1207++) {
float x1208 = x1188[x1207];
double x1209 = (double)x1208;
double x1210 = tanh(x1209);
float x1211 = (float)x1210;
x1206[x1207] = x1211;

}
float* x1215 = (float*)myMalloc(150 * sizeof(float));;
float* x1216 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1217 = 0;
int32_t x1218 = 0;
int32_t x1219 = 0;
for(int x1220=0; x1220 < 150; x1220++) {
int32_t x1221 = x1217;
int32_t x1222 = x1218;
float x1223 = x1049[x1222];
int32_t x1224 = x1219;
float x1225 = x1206[x1224];
float x1226 = x1223 * x1225;
x1216[x1221] = x1226;
x1217 += 1;
x1218 += 1;
x1219 += 1;

}
float* x1233 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(5, 150), List(150)
float* x1235 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x163,150,x1216,1,0,x1235,1);
float* x1237 = (float*)myMalloc(5 * sizeof(float));;
float* x1238 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1239 = 0;
int32_t x1240 = 0;
int32_t x1241 = 0;
for(int x1242=0; x1242 < 5; x1242++) {
int32_t x1243 = x1239;
int32_t x1244 = x1240;
float x1245 = x1235[x1244];
int32_t x1246 = x1241;
float x1247 = x172[x1246];
float x1248 = x1245 + x1247;
x1238[x1243] = x1248;
x1239 += 1;
x1240 += 1;
x1241 += 1;

}
float* x1255 = (float*)myMalloc(5 * sizeof(float));;
float* x1256 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1257 = 0;
int32_t x1258 = 0;
int32_t x1259 = 0;
int32_t x1260 = x1257;
int32_t x1261 = x1258;
float x1262 = x259[x1261];
int32_t x1263 = x1259;
float x1264 = x268[x1263];
float x1265 = x1262 + x1264;
x1256[x1260] = x1265;
x1257 += 1;
float* x1268 = (float*)myMalloc(1 * sizeof(float));;
float x1269 = -3.4028235E38f;
for(int x1270=0; x1270 < 5; x1270++) {
float x1271 = x1269;
float x1272 = x1238[x1270];
bool x1273 = x1272 > x1271;
float x1274;
if (x1273) {
x1274 = x1272;
} else {
x1274 = x1271;
}
x1269 = x1274;

}
float x1278 = x1269;
float x1279 = 0.0f;
for(int x1280=0; x1280 < 5; x1280++) {
float x1281 = x1279;
float x1282 = x1238[x1280];
float x1283 = x1269;
float x1284 = x1282 - x1283;
double x1285 = (double)x1284;
double x1286 = exp(x1285);
float x1287 = (float)x1286;
float x1288 = x1281 + x1287;
x1279 = x1288;

}
float x1292 = x1279;
float* x1297 = (float*)myMalloc(5 * sizeof(float));;
double x1293 = (double)x1292;
double x1294 = log(x1293);
float x1295 = (float)x1294;
float x1296 = x1278 + x1295;
for(int x1298=0; x1298 < 5; x1298++) {
float x1299 = x1238[x1298];
float x1300 = x1299 - x1296;
x1297[x1298] = x1300;

}
float* x1304 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1305 = x226[x246];
float x1306 = x1297[x1305];
float* x1308 = (float*)myMalloc(1 * sizeof(float));;
float x1307 = -1.0f * x1306;
x1308[0] = x1307;
float* x1310 = (float*)myMalloc(1 * sizeof(float));;
float* x1311 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1312 = 0;
int32_t x1313 = 0;
int32_t x1314 = 0;
int32_t x1315 = x1312;
int32_t x1316 = x1313;
float x1317 = x1256[x1316];
int32_t x1318 = x1314;
float x1319 = x1308[x1318];
float x1320 = x1317 + x1319;
x1311[x1315] = x1320;
x1312 += 1;
float* x1323 = (float*)myMalloc(1 * sizeof(float));;
float** x1358 = (float**)myMalloc(6 * sizeof(float*));;
x1358[0] = x1311;
x1358[1] = x1323;
x1358[2] = x1216;
x1358[3] = x1233;
x1358[4] = x1188;
x1358[5] = x1205;
x1324(x1358);
int32_t x1366 = 0;
int32_t x1367 = 0;
int32_t x1368 = 0;
int32_t x1369 = x1366;
float x1370 = x1268[x1369];
float x1371 = x1256[x1369];
int32_t x1372 = x1367;
float x1373 = x1308[x1372];
int32_t x1374 = x1368;
float x1375 = x1323[x1374];
float x1376 = x1370 + x1375;
x1268[x1369] = x1376;
float x1378 = x1310[x1372];
float x1379 = x1256[x1369];
float x1380 = x1308[x1372];
float x1381 = x1323[x1374];
float x1382 = x1378 + x1381;
x1310[x1372] = x1382;
x1368 += 1;
float x1385 = x1304[x1305];
float x1386 = x1310[0];
float x1387 = -1.0f * x1386;
float x1388 = x1385 + x1387;
x1304[x1305] = x1388;
float x1390 = 0.0f;
for(int x1391=0; x1391 < 5; x1391++) {
float x1392 = x1390;
float x1393 = x1304[x1391];
float x1394 = x1392 + x1393;
x1390 = x1394;

}
float x1398 = x1390;
float* x1399 = (float*)myMalloc(1 * sizeof(float));;
x1399[0] = x1398;
float x1401 = x1399[0];
for(int x1402=0; x1402 < 5; x1402++) {
float x1403 = x1255[x1402];
float x1404 = x1304[x1402];
float x1405 = x1297[x1402];
double x1406 = (double)x1405;
double x1407 = exp(x1406);
float x1408 = (float)x1407;
float x1409 = x1408 * x1401;
float x1410 = x1404 - x1409;
float x1411 = x1403 + x1410;
x1255[x1402] = x1411;

}
int32_t x1415 = 0;
int32_t x1416 = 0;
int32_t x1417 = 0;
int32_t x1418 = x1415;
float x1419 = x260[x1418];
float x1420 = x259[x1418];
int32_t x1421 = x1416;
float x1422 = x268[x1421];
int32_t x1423 = x1417;
float x1424 = x1268[x1423];
float x1425 = x1419 + x1424;
x260[x1418] = x1425;
float x1427 = x269[x1421];
float x1428 = x259[x1418];
float x1429 = x268[x1421];
float x1430 = x1268[x1423];
float x1431 = x1427 + x1430;
x269[x1421] = x1431;
x1417 += 1;
int32_t x1434 = 0;
int32_t x1435 = 0;
int32_t x1436 = 0;
for(int x1437=0; x1437 < 5; x1437++) {
int32_t x1438 = x1434;
float x1439 = x1237[x1438];
float x1440 = x1235[x1438];
int32_t x1441 = x1435;
float x1442 = x172[x1441];
int32_t x1443 = x1436;
float x1444 = x1255[x1443];
float x1445 = x1439 + x1444;
x1237[x1438] = x1445;
float x1447 = x194[x1441];
float x1448 = x1235[x1438];
float x1449 = x172[x1441];
float x1450 = x1255[x1443];
float x1451 = x1447 + x1450;
x194[x1441] = x1451;
x1436 += 1;
x1434 += 1;
x1435 += 1;

}
// add_cartesian
int32_t x1459 = 0;
for(int x1460=0; x1460 < 5; x1460++) {
for(int x1461=0; x1461 < 150; x1461++) {
int32_t x1462 = x1459;
int32_t x1463 = x1462 + x1461;
float x1464 = x193[x1463];
float x1465 = x1216[x1461];
float x1466 = x1237[x1460];
float x1467 = x1465 * x1466;
float x1468 = x1464 + x1467;
x193[x1463] = x1468;

}
x1459 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x163,150,x1237,1,1,x1233,1);
int32_t x1476 = 0;
int32_t x1477 = 0;
int32_t x1478 = 0;
for(int x1479=0; x1479 < 150; x1479++) {
int32_t x1480 = x1476;
float x1481 = x1061[x1480];
float x1482 = x1049[x1480];
int32_t x1483 = x1477;
float x1484 = x1206[x1483];
int32_t x1485 = x1478;
float x1486 = x1233[x1485];
float x1487 = x1486 * x1484;
float x1488 = x1481 + x1487;
x1061[x1480] = x1488;
float x1490 = x1215[x1483];
float x1491 = x1049[x1480];
float x1492 = x1206[x1483];
float x1493 = x1233[x1485];
float x1494 = x1493 * x1491;
float x1495 = x1490 + x1494;
x1215[x1483] = x1495;
x1478 += 1;
x1476 += 1;
x1477 += 1;

}
for(int x1502=0; x1502 < 150; x1502++) {
float x1503 = x1205[x1502];
float x1504 = x1206[x1502];
float x1507 = x1215[x1502];
float x1505 = x1504 * x1504;
float x1506 = 1.0f - x1505;
float x1508 = x1506 * x1507;
float x1509 = x1503 + x1508;
x1205[x1502] = x1509;

}
int32_t x1513 = 0;
int32_t x1514 = 0;
int32_t x1515 = 0;
for(int x1516=0; x1516 < 150; x1516++) {
int32_t x1517 = x1513;
float x1518 = x1169[x1517];
float x1519 = x1152[x1517];
int32_t x1520 = x1514;
float x1521 = x1170[x1520];
int32_t x1522 = x1515;
float x1523 = x1205[x1522];
float x1524 = x1518 + x1523;
x1169[x1517] = x1524;
float x1526 = x1187[x1520];
float x1527 = x1152[x1517];
float x1528 = x1170[x1520];
float x1529 = x1205[x1522];
float x1530 = x1526 + x1529;
x1187[x1520] = x1530;
x1515 += 1;
x1513 += 1;
x1514 += 1;

}
int32_t x1537 = 0;
int32_t x1538 = 0;
int32_t x1539 = 0;
for(int x1540=0; x1540 < 150; x1540++) {
int32_t x1541 = x1537;
float x1542 = x1004[x1541];
float x1543 = x992[x1541];
int32_t x1544 = x1538;
float x1545 = x272[x1544];
int32_t x1546 = x1539;
float x1547 = x1187[x1546];
float x1548 = x1547 * x1545;
float x1549 = x1542 + x1548;
x1004[x1541] = x1549;
float x1551 = x273[x1544];
float x1552 = x992[x1541];
float x1553 = x272[x1544];
float x1554 = x1187[x1546];
float x1555 = x1554 * x1552;
float x1556 = x1551 + x1555;
x273[x1544] = x1556;
x1539 += 1;
x1537 += 1;
x1538 += 1;

}
int32_t x1563 = 0;
int32_t x1564 = 0;
int32_t x1565 = 0;
for(int x1566=0; x1566 < 150; x1566++) {
int32_t x1567 = x1563;
float x1568 = x1133[x1567];
float x1569 = x1116[x1567];
int32_t x1570 = x1564;
float x1571 = x1134[x1570];
int32_t x1572 = x1565;
float x1573 = x1169[x1572];
float x1574 = x1568 + x1573;
x1133[x1567] = x1574;
float x1576 = x1151[x1570];
float x1577 = x1116[x1567];
float x1578 = x1134[x1570];
float x1579 = x1169[x1572];
float x1580 = x1576 + x1579;
x1151[x1570] = x1580;
x1565 += 1;
x1563 += 1;
x1564 += 1;

}
int32_t x1587 = 0;
int32_t x1588 = 0;
int32_t x1589 = 0;
for(int x1590=0; x1590 < 150; x1590++) {
int32_t x1591 = x1587;
float x1592 = x947[x1591];
float x1593 = x935[x1591];
int32_t x1594 = x1588;
float x1595 = x263[x1594];
int32_t x1596 = x1589;
float x1597 = x1151[x1596];
float x1598 = x1597 * x1595;
float x1599 = x1592 + x1598;
x947[x1591] = x1599;
float x1601 = x264[x1594];
float x1602 = x935[x1591];
float x1603 = x263[x1594];
float x1604 = x1151[x1596];
float x1605 = x1604 * x1602;
float x1606 = x1601 + x1605;
x264[x1594] = x1606;
x1589 += 1;
x1587 += 1;
x1588 += 1;

}
int32_t x1613 = 0;
int32_t x1614 = 0;
int32_t x1615 = 0;
for(int x1616=0; x1616 < 150; x1616++) {
int32_t x1617 = x1613;
float x1618 = x890[x1617];
float x1619 = x878[x1617];
int32_t x1620 = x1614;
float x1621 = x1106[x1620];
int32_t x1622 = x1615;
float x1623 = x1133[x1622];
float x1624 = x1623 * x1621;
float x1625 = x1618 + x1624;
x890[x1617] = x1625;
float x1627 = x1115[x1620];
float x1628 = x878[x1617];
float x1629 = x1106[x1620];
float x1630 = x1133[x1622];
float x1631 = x1630 * x1628;
float x1632 = x1627 + x1631;
x1115[x1620] = x1632;
x1615 += 1;
x1613 += 1;
x1614 += 1;

}
for(int x1639=0; x1639 < 150; x1639++) {
float x1640 = x1105[x1639];
float x1641 = x1106[x1639];
float x1644 = x1115[x1639];
float x1642 = x1641 * x1641;
float x1643 = 1.0f - x1642;
float x1645 = x1643 * x1644;
float x1646 = x1640 + x1645;
x1105[x1639] = x1646;

}
int32_t x1650 = 0;
int32_t x1651 = 0;
int32_t x1652 = 0;
for(int x1653=0; x1653 < 150; x1653++) {
int32_t x1654 = x1650;
float x1655 = x1087[x1654];
float x1656 = x1070[x1654];
int32_t x1657 = x1651;
float x1658 = x162[x1657];
int32_t x1659 = x1652;
float x1660 = x1105[x1659];
float x1661 = x1655 + x1660;
x1087[x1654] = x1661;
float x1663 = x192[x1657];
float x1664 = x1070[x1654];
float x1665 = x162[x1657];
float x1666 = x1105[x1659];
float x1667 = x1663 + x1666;
x192[x1657] = x1667;
x1652 += 1;
x1650 += 1;
x1651 += 1;

}
int32_t x1674 = 0;
int32_t x1675 = 0;
int32_t x1676 = 0;
for(int x1677=0; x1677 < 150; x1677++) {
int32_t x1678 = x1674;
float x1679 = x1065[x1678];
float x1680 = x1063[x1678];
int32_t x1681 = x1675;
float x1682 = x1067[x1681];
int32_t x1683 = x1676;
float x1684 = x1087[x1683];
float x1685 = x1679 + x1684;
x1065[x1678] = x1685;
float x1687 = x1069[x1681];
float x1688 = x1063[x1678];
float x1689 = x1067[x1681];
float x1690 = x1087[x1683];
float x1691 = x1687 + x1690;
x1069[x1681] = x1691;
x1676 += 1;
x1674 += 1;
x1675 += 1;

}
// add_cartesian
int32_t x1699 = 0;
for(int x1700=0; x1700 < 150; x1700++) {
for(int x1701=0; x1701 < 150; x1701++) {
int32_t x1702 = x1699;
int32_t x1703 = x1702 + x1701;
float x1704 = x191[x1703];
float x1705 = x270[x1701];
float x1706 = x1069[x1700];
float x1707 = x1705 * x1706;
float x1708 = x1704 + x1707;
x191[x1703] = x1708;

}
x1699 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x154,150,x1069,1,1,x271,1);
// add_cartesian
int32_t x1717 = 0;
for(int x1718=0; x1718 < 150; x1718++) {
for(int x1719=0; x1719 < 150; x1719++) {
int32_t x1720 = x1717;
int32_t x1721 = x1720 + x1719;
float x1722 = x190[x1721];
float x1723 = x261[x1719];
float x1724 = x1065[x1718];
float x1725 = x1723 * x1724;
float x1726 = x1722 + x1725;
x190[x1721] = x1726;

}
x1717 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x146,150,x1065,1,1,x262,1);
for(int x1734=0; x1734 < 150; x1734++) {
float x1735 = x1048[x1734];
float x1736 = x1049[x1734];
float x1739 = x1061[x1734];
float x1737 = 1.0f - x1736;
float x1738 = x1737 * x1736;
float x1740 = x1738 * x1739;
float x1741 = x1735 + x1740;
x1048[x1734] = x1741;

}
int32_t x1745 = 0;
int32_t x1746 = 0;
int32_t x1747 = 0;
for(int x1748=0; x1748 < 150; x1748++) {
int32_t x1749 = x1745;
float x1750 = x1030[x1749];
float x1751 = x1013[x1749];
int32_t x1752 = x1746;
float x1753 = x145[x1752];
int32_t x1754 = x1747;
float x1755 = x1048[x1754];
float x1756 = x1750 + x1755;
x1030[x1749] = x1756;
float x1758 = x189[x1752];
float x1759 = x1013[x1749];
float x1760 = x145[x1752];
float x1761 = x1048[x1754];
float x1762 = x1758 + x1761;
x189[x1752] = x1762;
x1747 += 1;
x1745 += 1;
x1746 += 1;

}
int32_t x1769 = 0;
int32_t x1770 = 0;
int32_t x1771 = 0;
for(int x1772=0; x1772 < 150; x1772++) {
int32_t x1773 = x1769;
float x1774 = x1008[x1773];
float x1775 = x1006[x1773];
int32_t x1776 = x1770;
float x1777 = x1010[x1776];
int32_t x1778 = x1771;
float x1779 = x1030[x1778];
float x1780 = x1774 + x1779;
x1008[x1773] = x1780;
float x1782 = x1012[x1776];
float x1783 = x1006[x1773];
float x1784 = x1010[x1776];
float x1785 = x1030[x1778];
float x1786 = x1782 + x1785;
x1012[x1776] = x1786;
x1771 += 1;
x1769 += 1;
x1770 += 1;

}
// add_cartesian
int32_t x1794 = 0;
for(int x1795=0; x1795 < 150; x1795++) {
for(int x1796=0; x1796 < 150; x1796++) {
int32_t x1797 = x1794;
int32_t x1798 = x1797 + x1796;
float x1799 = x188[x1798];
float x1800 = x270[x1796];
float x1801 = x1012[x1795];
float x1802 = x1800 * x1801;
float x1803 = x1799 + x1802;
x188[x1798] = x1803;

}
x1794 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x137,150,x1012,1,1,x271,1);
// add_cartesian
int32_t x1812 = 0;
for(int x1813=0; x1813 < 150; x1813++) {
for(int x1814=0; x1814 < 150; x1814++) {
int32_t x1815 = x1812;
int32_t x1816 = x1815 + x1814;
float x1817 = x187[x1816];
float x1818 = x261[x1814];
float x1819 = x1008[x1813];
float x1820 = x1818 * x1819;
float x1821 = x1817 + x1820;
x187[x1816] = x1821;

}
x1812 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x129,150,x1008,1,1,x262,1);
for(int x1829=0; x1829 < 150; x1829++) {
float x1830 = x991[x1829];
float x1831 = x992[x1829];
float x1834 = x1004[x1829];
float x1832 = 1.0f - x1831;
float x1833 = x1832 * x1831;
float x1835 = x1833 * x1834;
float x1836 = x1830 + x1835;
x991[x1829] = x1836;

}
int32_t x1840 = 0;
int32_t x1841 = 0;
int32_t x1842 = 0;
for(int x1843=0; x1843 < 150; x1843++) {
int32_t x1844 = x1840;
float x1845 = x973[x1844];
float x1846 = x956[x1844];
int32_t x1847 = x1841;
float x1848 = x128[x1847];
int32_t x1849 = x1842;
float x1850 = x991[x1849];
float x1851 = x1845 + x1850;
x973[x1844] = x1851;
float x1853 = x186[x1847];
float x1854 = x956[x1844];
float x1855 = x128[x1847];
float x1856 = x991[x1849];
float x1857 = x1853 + x1856;
x186[x1847] = x1857;
x1842 += 1;
x1840 += 1;
x1841 += 1;

}
int32_t x1864 = 0;
int32_t x1865 = 0;
int32_t x1866 = 0;
for(int x1867=0; x1867 < 150; x1867++) {
int32_t x1868 = x1864;
float x1869 = x951[x1868];
float x1870 = x949[x1868];
int32_t x1871 = x1865;
float x1872 = x953[x1871];
int32_t x1873 = x1866;
float x1874 = x973[x1873];
float x1875 = x1869 + x1874;
x951[x1868] = x1875;
float x1877 = x955[x1871];
float x1878 = x949[x1868];
float x1879 = x953[x1871];
float x1880 = x973[x1873];
float x1881 = x1877 + x1880;
x955[x1871] = x1881;
x1866 += 1;
x1864 += 1;
x1865 += 1;

}
// add_cartesian
int32_t x1889 = 0;
for(int x1890=0; x1890 < 150; x1890++) {
for(int x1891=0; x1891 < 150; x1891++) {
int32_t x1892 = x1889;
int32_t x1893 = x1892 + x1891;
float x1894 = x185[x1893];
float x1895 = x270[x1891];
float x1896 = x955[x1890];
float x1897 = x1895 * x1896;
float x1898 = x1894 + x1897;
x185[x1893] = x1898;

}
x1889 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x120,150,x955,1,1,x271,1);
// add_cartesian
int32_t x1907 = 0;
for(int x1908=0; x1908 < 150; x1908++) {
for(int x1909=0; x1909 < 150; x1909++) {
int32_t x1910 = x1907;
int32_t x1911 = x1910 + x1909;
float x1912 = x184[x1911];
float x1913 = x261[x1909];
float x1914 = x951[x1908];
float x1915 = x1913 * x1914;
float x1916 = x1912 + x1915;
x184[x1911] = x1916;

}
x1907 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x112,150,x951,1,1,x262,1);
for(int x1924=0; x1924 < 150; x1924++) {
float x1925 = x934[x1924];
float x1926 = x935[x1924];
float x1929 = x947[x1924];
float x1927 = 1.0f - x1926;
float x1928 = x1927 * x1926;
float x1930 = x1928 * x1929;
float x1931 = x1925 + x1930;
x934[x1924] = x1931;

}
int32_t x1935 = 0;
int32_t x1936 = 0;
int32_t x1937 = 0;
for(int x1938=0; x1938 < 150; x1938++) {
int32_t x1939 = x1935;
float x1940 = x916[x1939];
float x1941 = x899[x1939];
int32_t x1942 = x1936;
float x1943 = x128[x1942];
int32_t x1944 = x1937;
float x1945 = x934[x1944];
float x1946 = x1940 + x1945;
x916[x1939] = x1946;
float x1948 = x186[x1942];
float x1949 = x899[x1939];
float x1950 = x128[x1942];
float x1951 = x934[x1944];
float x1952 = x1948 + x1951;
x186[x1942] = x1952;
x1937 += 1;
x1935 += 1;
x1936 += 1;

}
int32_t x1959 = 0;
int32_t x1960 = 0;
int32_t x1961 = 0;
for(int x1962=0; x1962 < 150; x1962++) {
int32_t x1963 = x1959;
float x1964 = x894[x1963];
float x1965 = x892[x1963];
int32_t x1966 = x1960;
float x1967 = x896[x1966];
int32_t x1968 = x1961;
float x1969 = x916[x1968];
float x1970 = x1964 + x1969;
x894[x1963] = x1970;
float x1972 = x898[x1966];
float x1973 = x892[x1963];
float x1974 = x896[x1966];
float x1975 = x916[x1968];
float x1976 = x1972 + x1975;
x898[x1966] = x1976;
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
float x1989 = x183[x1988];
float x1990 = x270[x1986];
float x1991 = x898[x1985];
float x1992 = x1990 * x1991;
float x1993 = x1989 + x1992;
x183[x1988] = x1993;

}
x1984 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x104,150,x898,1,1,x271,1);
// add_cartesian
int32_t x2002 = 0;
for(int x2003=0; x2003 < 150; x2003++) {
for(int x2004=0; x2004 < 150; x2004++) {
int32_t x2005 = x2002;
int32_t x2006 = x2005 + x2004;
float x2007 = x182[x2006];
float x2008 = x261[x2004];
float x2009 = x894[x2003];
float x2010 = x2008 * x2009;
float x2011 = x2007 + x2010;
x182[x2006] = x2011;

}
x2002 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x96,150,x894,1,1,x262,1);
for(int x2019=0; x2019 < 150; x2019++) {
float x2020 = x877[x2019];
float x2021 = x878[x2019];
float x2024 = x890[x2019];
float x2022 = 1.0f - x2021;
float x2023 = x2022 * x2021;
float x2025 = x2023 * x2024;
float x2026 = x2020 + x2025;
x877[x2019] = x2026;

}
int32_t x2030 = 0;
int32_t x2031 = 0;
int32_t x2032 = 0;
for(int x2033=0; x2033 < 150; x2033++) {
int32_t x2034 = x2030;
float x2035 = x859[x2034];
float x2036 = x842[x2034];
int32_t x2037 = x2031;
float x2038 = x95[x2037];
int32_t x2039 = x2032;
float x2040 = x877[x2039];
float x2041 = x2035 + x2040;
x859[x2034] = x2041;
float x2043 = x181[x2037];
float x2044 = x842[x2034];
float x2045 = x95[x2037];
float x2046 = x877[x2039];
float x2047 = x2043 + x2046;
x181[x2037] = x2047;
x2032 += 1;
x2030 += 1;
x2031 += 1;

}
int32_t x2054 = 0;
int32_t x2055 = 0;
int32_t x2056 = 0;
for(int x2057=0; x2057 < 150; x2057++) {
int32_t x2058 = x2054;
float x2059 = x837[x2058];
float x2060 = x835[x2058];
int32_t x2061 = x2055;
float x2062 = x839[x2061];
int32_t x2063 = x2056;
float x2064 = x859[x2063];
float x2065 = x2059 + x2064;
x837[x2058] = x2065;
float x2067 = x841[x2061];
float x2068 = x835[x2058];
float x2069 = x839[x2061];
float x2070 = x859[x2063];
float x2071 = x2067 + x2070;
x841[x2061] = x2071;
x2056 += 1;
x2054 += 1;
x2055 += 1;

}
// add_cartesian
int32_t x2079 = 0;
for(int x2080=0; x2080 < 150; x2080++) {
for(int x2081=0; x2081 < 150; x2081++) {
int32_t x2082 = x2079;
int32_t x2083 = x2082 + x2081;
float x2084 = x180[x2083];
float x2085 = x270[x2081];
float x2086 = x841[x2080];
float x2087 = x2085 * x2086;
float x2088 = x2084 + x2087;
x180[x2083] = x2088;

}
x2079 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x87,150,x841,1,1,x271,1);
// add_cartesian
int32_t x2097 = 0;
for(int x2098=0; x2098 < 150; x2098++) {
for(int x2099=0; x2099 < 150; x2099++) {
int32_t x2100 = x2097;
int32_t x2101 = x2100 + x2099;
float x2102 = x179[x2101];
float x2103 = x261[x2099];
float x2104 = x837[x2098];
float x2105 = x2103 * x2104;
float x2106 = x2102 + x2105;
x179[x2101] = x2106;

}
x2097 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x78,150,x837,1,1,x262,1);
}
};
x242(x265,x266,x2117);
};
x242(x256,x257,x2127);
} else {
float** x2154 = (float**)myMalloc(6 * sizeof(float*));;
x2154[0] = x236;
x2154[1] = x237;
x2154[2] = x238;
x2154[3] = x239;
x2154[4] = x240;
x2154[5] = x241;
function<void(float**)> x247 = x244;
function<void(float**)> x2137 = [&](float** x2138) {
float* x2139 = x2138[0];
float* x2140 = x2138[1];
float* x2141 = x2138[2];
float* x2142 = x2138[3];
float* x2143 = x2138[4];
float* x2144 = x2138[5];
float** x2145 = (float**)myMalloc(6 * sizeof(float*));;
x2145[0] = x2139;
x2145[1] = x2140;
x2145[2] = x2141;
x2145[3] = x2142;
x2145[4] = x2143;
x2145[5] = x2144;
x247(x2145);
};
x2137(x2154);
}
};
float* x233 = (float*)myMalloc(1 * sizeof(float));;
float* x234 = (float*)myMalloc(1 * sizeof(float));;
float* x235 = (float*)myMalloc(1 * sizeof(float));;
float** x2177 = (float**)myMalloc(6 * sizeof(float*));;
x2177[0] = x236;
x2177[1] = x237;
x2177[2] = x238;
x2177[3] = x239;
x2177[4] = x240;
x2177[5] = x241;
function<void(float**)> x2165 = [&](float** x2166) {
float* x2167 = x2166[0];
float* x2168 = x2166[1];
float* x2169 = x2166[2];
float* x2170 = x2166[3];
float* x2171 = x2166[4];
float* x2172 = x2166[5];
x2168[0] = 1.0f;
float x2174 = x2167[0];
x235[0] = x2174;
};
x242(0,x2165,x2177);
float x2186 = x235[0];
float x2187 = x222;
float x2188 = (float)x223;
float x2189 = x2187 * x2188;
int32_t x2190 = x223 + 1;
float x2191 = (float)x2190;
float x2192 = x2189 / x2191;
float x2193 = x2186 / x2191;
float x2194 = x2192 + x2193;
x222 = x2194;
for(int x2196=0; x2196 < 45000; x2196++) {
float x2197 = x173[x2196];
float x2198 = x2197;
float x2199 = x195[x2196];
float x2200 = x2198;
float x2201 = x2200 * x2200;
float x2202 = x2199 + x2201;
x195[x2196] = x2202;
float x2204 = x50[x2196];
float x2206 = x195[x2196];
float x2205 = 0.05f * x2200;
double x2207 = (double)x2206;
double x2208 = x2207 + 9.99999993922529E-9;
double x2209 = sqrt(x2208);
float x2210 = (float)x2209;
float x2211 = x2205 / x2210;
float x2212 = x2204 - x2211;
x50[x2196] = x2212;
x173[x2196] = 0.0f;

}
for(int x2217=0; x2217 < 150; x2217++) {
float x2218 = x174[x2217];
float x2219 = x2218;
float x2220 = x196[x2217];
float x2221 = x2219;
float x2222 = x2221 * x2221;
float x2223 = x2220 + x2222;
x196[x2217] = x2223;
float x2225 = x59[x2217];
float x2227 = x196[x2217];
float x2226 = 0.05f * x2221;
double x2228 = (double)x2227;
double x2229 = x2228 + 9.99999993922529E-9;
double x2230 = sqrt(x2229);
float x2231 = (float)x2230;
float x2232 = x2226 / x2231;
float x2233 = x2225 - x2232;
x59[x2217] = x2233;
x174[x2217] = 0.0f;

}
for(int x2238=0; x2238 < 45000; x2238++) {
float x2239 = x175[x2238];
float x2240 = x2239;
float x2241 = x197[x2238];
float x2242 = x2240;
float x2243 = x2242 * x2242;
float x2244 = x2241 + x2243;
x197[x2238] = x2244;
float x2246 = x60[x2238];
float x2248 = x197[x2238];
float x2247 = 0.05f * x2242;
double x2249 = (double)x2248;
double x2250 = x2249 + 9.99999993922529E-9;
double x2251 = sqrt(x2250);
float x2252 = (float)x2251;
float x2253 = x2247 / x2252;
float x2254 = x2246 - x2253;
x60[x2238] = x2254;
x175[x2238] = 0.0f;

}
for(int x2259=0; x2259 < 150; x2259++) {
float x2260 = x176[x2259];
float x2261 = x2260;
float x2262 = x198[x2259];
float x2263 = x2261;
float x2264 = x2263 * x2263;
float x2265 = x2262 + x2264;
x198[x2259] = x2265;
float x2267 = x68[x2259];
float x2269 = x198[x2259];
float x2268 = 0.05f * x2263;
double x2270 = (double)x2269;
double x2271 = x2270 + 9.99999993922529E-9;
double x2272 = sqrt(x2271);
float x2273 = (float)x2272;
float x2274 = x2268 / x2273;
float x2275 = x2267 - x2274;
x68[x2259] = x2275;
x176[x2259] = 0.0f;

}
for(int x2280=0; x2280 < 45000; x2280++) {
float x2281 = x177[x2280];
float x2282 = x2281;
float x2283 = x199[x2280];
float x2284 = x2282;
float x2285 = x2284 * x2284;
float x2286 = x2283 + x2285;
x199[x2280] = x2286;
float x2288 = x69[x2280];
float x2290 = x199[x2280];
float x2289 = 0.05f * x2284;
double x2291 = (double)x2290;
double x2292 = x2291 + 9.99999993922529E-9;
double x2293 = sqrt(x2292);
float x2294 = (float)x2293;
float x2295 = x2289 / x2294;
float x2296 = x2288 - x2295;
x69[x2280] = x2296;
x177[x2280] = 0.0f;

}
for(int x2301=0; x2301 < 150; x2301++) {
float x2302 = x178[x2301];
float x2303 = x2302;
float x2304 = x200[x2301];
float x2305 = x2303;
float x2306 = x2305 * x2305;
float x2307 = x2304 + x2306;
x200[x2301] = x2307;
float x2309 = x77[x2301];
float x2311 = x200[x2301];
float x2310 = 0.05f * x2305;
double x2312 = (double)x2311;
double x2313 = x2312 + 9.99999993922529E-9;
double x2314 = sqrt(x2313);
float x2315 = (float)x2314;
float x2316 = x2310 / x2315;
float x2317 = x2309 - x2316;
x77[x2301] = x2317;
x178[x2301] = 0.0f;

}
for(int x2322=0; x2322 < 22500; x2322++) {
float x2323 = x179[x2322];
float x2324 = x2323;
float x2325 = x201[x2322];
float x2326 = x2324;
float x2327 = x2326 * x2326;
float x2328 = x2325 + x2327;
x201[x2322] = x2328;
float x2330 = x78[x2322];
float x2332 = x201[x2322];
float x2331 = 0.05f * x2326;
double x2333 = (double)x2332;
double x2334 = x2333 + 9.99999993922529E-9;
double x2335 = sqrt(x2334);
float x2336 = (float)x2335;
float x2337 = x2331 / x2336;
float x2338 = x2330 - x2337;
x78[x2322] = x2338;
x179[x2322] = 0.0f;

}
for(int x2343=0; x2343 < 22500; x2343++) {
float x2344 = x180[x2343];
float x2345 = x2344;
float x2346 = x202[x2343];
float x2347 = x2345;
float x2348 = x2347 * x2347;
float x2349 = x2346 + x2348;
x202[x2343] = x2349;
float x2351 = x87[x2343];
float x2353 = x202[x2343];
float x2352 = 0.05f * x2347;
double x2354 = (double)x2353;
double x2355 = x2354 + 9.99999993922529E-9;
double x2356 = sqrt(x2355);
float x2357 = (float)x2356;
float x2358 = x2352 / x2357;
float x2359 = x2351 - x2358;
x87[x2343] = x2359;
x180[x2343] = 0.0f;

}
for(int x2364=0; x2364 < 150; x2364++) {
float x2365 = x181[x2364];
float x2366 = x2365;
float x2367 = x203[x2364];
float x2368 = x2366;
float x2369 = x2368 * x2368;
float x2370 = x2367 + x2369;
x203[x2364] = x2370;
float x2372 = x95[x2364];
float x2374 = x203[x2364];
float x2373 = 0.05f * x2368;
double x2375 = (double)x2374;
double x2376 = x2375 + 9.99999993922529E-9;
double x2377 = sqrt(x2376);
float x2378 = (float)x2377;
float x2379 = x2373 / x2378;
float x2380 = x2372 - x2379;
x95[x2364] = x2380;
x181[x2364] = 0.0f;

}
for(int x2385=0; x2385 < 22500; x2385++) {
float x2386 = x182[x2385];
float x2387 = x2386;
float x2388 = x204[x2385];
float x2389 = x2387;
float x2390 = x2389 * x2389;
float x2391 = x2388 + x2390;
x204[x2385] = x2391;
float x2393 = x96[x2385];
float x2395 = x204[x2385];
float x2394 = 0.05f * x2389;
double x2396 = (double)x2395;
double x2397 = x2396 + 9.99999993922529E-9;
double x2398 = sqrt(x2397);
float x2399 = (float)x2398;
float x2400 = x2394 / x2399;
float x2401 = x2393 - x2400;
x96[x2385] = x2401;
x182[x2385] = 0.0f;

}
for(int x2406=0; x2406 < 22500; x2406++) {
float x2407 = x183[x2406];
float x2408 = x2407;
float x2409 = x205[x2406];
float x2410 = x2408;
float x2411 = x2410 * x2410;
float x2412 = x2409 + x2411;
x205[x2406] = x2412;
float x2414 = x104[x2406];
float x2416 = x205[x2406];
float x2415 = 0.05f * x2410;
double x2417 = (double)x2416;
double x2418 = x2417 + 9.99999993922529E-9;
double x2419 = sqrt(x2418);
float x2420 = (float)x2419;
float x2421 = x2415 / x2420;
float x2422 = x2414 - x2421;
x104[x2406] = x2422;
x183[x2406] = 0.0f;

}
for(int x2427=0; x2427 < 22500; x2427++) {
float x2428 = x184[x2427];
float x2429 = x2428;
float x2430 = x206[x2427];
float x2431 = x2429;
float x2432 = x2431 * x2431;
float x2433 = x2430 + x2432;
x206[x2427] = x2433;
float x2435 = x112[x2427];
float x2437 = x206[x2427];
float x2436 = 0.05f * x2431;
double x2438 = (double)x2437;
double x2439 = x2438 + 9.99999993922529E-9;
double x2440 = sqrt(x2439);
float x2441 = (float)x2440;
float x2442 = x2436 / x2441;
float x2443 = x2435 - x2442;
x112[x2427] = x2443;
x184[x2427] = 0.0f;

}
for(int x2448=0; x2448 < 22500; x2448++) {
float x2449 = x185[x2448];
float x2450 = x2449;
float x2451 = x207[x2448];
float x2452 = x2450;
float x2453 = x2452 * x2452;
float x2454 = x2451 + x2453;
x207[x2448] = x2454;
float x2456 = x120[x2448];
float x2458 = x207[x2448];
float x2457 = 0.05f * x2452;
double x2459 = (double)x2458;
double x2460 = x2459 + 9.99999993922529E-9;
double x2461 = sqrt(x2460);
float x2462 = (float)x2461;
float x2463 = x2457 / x2462;
float x2464 = x2456 - x2463;
x120[x2448] = x2464;
x185[x2448] = 0.0f;

}
for(int x2469=0; x2469 < 150; x2469++) {
float x2470 = x186[x2469];
float x2471 = x2470;
float x2472 = x208[x2469];
float x2473 = x2471;
float x2474 = x2473 * x2473;
float x2475 = x2472 + x2474;
x208[x2469] = x2475;
float x2477 = x128[x2469];
float x2479 = x208[x2469];
float x2478 = 0.05f * x2473;
double x2480 = (double)x2479;
double x2481 = x2480 + 9.99999993922529E-9;
double x2482 = sqrt(x2481);
float x2483 = (float)x2482;
float x2484 = x2478 / x2483;
float x2485 = x2477 - x2484;
x128[x2469] = x2485;
x186[x2469] = 0.0f;

}
for(int x2490=0; x2490 < 22500; x2490++) {
float x2491 = x187[x2490];
float x2492 = x2491;
float x2493 = x209[x2490];
float x2494 = x2492;
float x2495 = x2494 * x2494;
float x2496 = x2493 + x2495;
x209[x2490] = x2496;
float x2498 = x129[x2490];
float x2500 = x209[x2490];
float x2499 = 0.05f * x2494;
double x2501 = (double)x2500;
double x2502 = x2501 + 9.99999993922529E-9;
double x2503 = sqrt(x2502);
float x2504 = (float)x2503;
float x2505 = x2499 / x2504;
float x2506 = x2498 - x2505;
x129[x2490] = x2506;
x187[x2490] = 0.0f;

}
for(int x2511=0; x2511 < 22500; x2511++) {
float x2512 = x188[x2511];
float x2513 = x2512;
float x2514 = x210[x2511];
float x2515 = x2513;
float x2516 = x2515 * x2515;
float x2517 = x2514 + x2516;
x210[x2511] = x2517;
float x2519 = x137[x2511];
float x2521 = x210[x2511];
float x2520 = 0.05f * x2515;
double x2522 = (double)x2521;
double x2523 = x2522 + 9.99999993922529E-9;
double x2524 = sqrt(x2523);
float x2525 = (float)x2524;
float x2526 = x2520 / x2525;
float x2527 = x2519 - x2526;
x137[x2511] = x2527;
x188[x2511] = 0.0f;

}
for(int x2532=0; x2532 < 150; x2532++) {
float x2533 = x189[x2532];
float x2534 = x2533;
float x2535 = x211[x2532];
float x2536 = x2534;
float x2537 = x2536 * x2536;
float x2538 = x2535 + x2537;
x211[x2532] = x2538;
float x2540 = x145[x2532];
float x2542 = x211[x2532];
float x2541 = 0.05f * x2536;
double x2543 = (double)x2542;
double x2544 = x2543 + 9.99999993922529E-9;
double x2545 = sqrt(x2544);
float x2546 = (float)x2545;
float x2547 = x2541 / x2546;
float x2548 = x2540 - x2547;
x145[x2532] = x2548;
x189[x2532] = 0.0f;

}
for(int x2553=0; x2553 < 22500; x2553++) {
float x2554 = x190[x2553];
float x2555 = x2554;
float x2556 = x212[x2553];
float x2557 = x2555;
float x2558 = x2557 * x2557;
float x2559 = x2556 + x2558;
x212[x2553] = x2559;
float x2561 = x146[x2553];
float x2563 = x212[x2553];
float x2562 = 0.05f * x2557;
double x2564 = (double)x2563;
double x2565 = x2564 + 9.99999993922529E-9;
double x2566 = sqrt(x2565);
float x2567 = (float)x2566;
float x2568 = x2562 / x2567;
float x2569 = x2561 - x2568;
x146[x2553] = x2569;
x190[x2553] = 0.0f;

}
for(int x2574=0; x2574 < 22500; x2574++) {
float x2575 = x191[x2574];
float x2576 = x2575;
float x2577 = x213[x2574];
float x2578 = x2576;
float x2579 = x2578 * x2578;
float x2580 = x2577 + x2579;
x213[x2574] = x2580;
float x2582 = x154[x2574];
float x2584 = x213[x2574];
float x2583 = 0.05f * x2578;
double x2585 = (double)x2584;
double x2586 = x2585 + 9.99999993922529E-9;
double x2587 = sqrt(x2586);
float x2588 = (float)x2587;
float x2589 = x2583 / x2588;
float x2590 = x2582 - x2589;
x154[x2574] = x2590;
x191[x2574] = 0.0f;

}
for(int x2595=0; x2595 < 150; x2595++) {
float x2596 = x192[x2595];
float x2597 = x2596;
float x2598 = x214[x2595];
float x2599 = x2597;
float x2600 = x2599 * x2599;
float x2601 = x2598 + x2600;
x214[x2595] = x2601;
float x2603 = x162[x2595];
float x2605 = x214[x2595];
float x2604 = 0.05f * x2599;
double x2606 = (double)x2605;
double x2607 = x2606 + 9.99999993922529E-9;
double x2608 = sqrt(x2607);
float x2609 = (float)x2608;
float x2610 = x2604 / x2609;
float x2611 = x2603 - x2610;
x162[x2595] = x2611;
x192[x2595] = 0.0f;

}
for(int x2616=0; x2616 < 750; x2616++) {
float x2617 = x193[x2616];
float x2618 = x2617;
float x2619 = x215[x2616];
float x2620 = x2618;
float x2621 = x2620 * x2620;
float x2622 = x2619 + x2621;
x215[x2616] = x2622;
float x2624 = x163[x2616];
float x2626 = x215[x2616];
float x2625 = 0.05f * x2620;
double x2627 = (double)x2626;
double x2628 = x2627 + 9.99999993922529E-9;
double x2629 = sqrt(x2628);
float x2630 = (float)x2629;
float x2631 = x2625 / x2630;
float x2632 = x2624 - x2631;
x163[x2616] = x2632;
x193[x2616] = 0.0f;

}
for(int x2637=0; x2637 < 5; x2637++) {
float x2638 = x194[x2637];
float x2639 = x2638;
float x2640 = x216[x2637];
float x2641 = x2639;
float x2642 = x2641 * x2641;
float x2643 = x2640 + x2642;
x216[x2637] = x2643;
float x2645 = x172[x2637];
float x2647 = x216[x2637];
float x2646 = 0.05f * x2641;
double x2648 = (double)x2647;
double x2649 = x2648 + 9.99999993922529E-9;
double x2650 = sqrt(x2649);
float x2651 = (float)x2650;
float x2652 = x2646 / x2651;
float x2653 = x2645 - x2652;
x172[x2637] = x2653;
x194[x2637] = 0.0f;

}
int64_t x2658 = (long)mallocAddr;
int64_t x2659 = x2658 - x218;
memset((void*)x218, 0, x2659);
mallocAddr = (void*)x218;

}
float x2664 = x222;
double x2665 = (double)x2664;
x217[x221] = x2665;
double x2667 = ((double)clock() / CLOCKS_PER_SEC);
double x2668 = x2667 - x219;
printf("epoc %d, average_loss %f, time %lf\n",x221,x2664,x2668);

}
double x2672 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2676 = (long)fopen(x0, "w");
fprintf((FILE *)x2676, "unit: %s\n", "1 epoch");
for(int x2678=0; x2678 < 6; x2678++) {
double x2679 = x217[x2678];
fprintf((FILE *)x2676, "%lf\n", x2679);

}
double x2673 = x219 - x2;
double x2674 = x2672 - x219;
double x2675 = x2674 / 6.0;
fprintf((FILE *)x2676, "run time: %lf %lf\n", x2673, x2675);
fclose((FILE*)x2676);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

