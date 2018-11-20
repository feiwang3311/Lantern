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
float* x60 = (float*)myMalloc(150 * sizeof(float));;
float* x61 = (float*)myMalloc(45000 * sizeof(float));;
for(int x62=0; x62 < 45000; x62++) {
float x63 = (float)rand()/RAND_MAX;
float x64 = x63 - 0.5f;
float x65 = x64 * 0.01f;
x61[x62] = x65;

}
float* x69 = (float*)myMalloc(150 * sizeof(float));;
float* x70 = (float*)myMalloc(45000 * sizeof(float));;
for(int x71=0; x71 < 45000; x71++) {
float x72 = (float)rand()/RAND_MAX;
float x73 = x72 - 0.5f;
float x74 = x73 * 0.01f;
x70[x71] = x74;

}
float* x78 = (float*)myMalloc(150 * sizeof(float));;
float* x79 = (float*)myMalloc(22500 * sizeof(float));;
for(int x81=0; x81 < 22500; x81++) {
float x82 = (float)rand()/RAND_MAX;
float x83 = x82 - 0.5f;
float x84 = x83 * 0.01f;
x79[x81] = x84;

}
float* x88 = (float*)myMalloc(22500 * sizeof(float));;
for(int x89=0; x89 < 22500; x89++) {
float x90 = (float)rand()/RAND_MAX;
float x91 = x90 - 0.5f;
float x92 = x91 * 0.01f;
x88[x89] = x92;

}
float* x96 = (float*)myMalloc(150 * sizeof(float));;
float* x97 = (float*)myMalloc(22500 * sizeof(float));;
for(int x98=0; x98 < 22500; x98++) {
float x99 = (float)rand()/RAND_MAX;
float x100 = x99 - 0.5f;
float x101 = x100 * 0.01f;
x97[x98] = x101;

}
float* x105 = (float*)myMalloc(22500 * sizeof(float));;
for(int x106=0; x106 < 22500; x106++) {
float x107 = (float)rand()/RAND_MAX;
float x108 = x107 - 0.5f;
float x109 = x108 * 0.01f;
x105[x106] = x109;

}
float* x113 = (float*)myMalloc(22500 * sizeof(float));;
for(int x114=0; x114 < 22500; x114++) {
float x115 = (float)rand()/RAND_MAX;
float x116 = x115 - 0.5f;
float x117 = x116 * 0.01f;
x113[x114] = x117;

}
float* x121 = (float*)myMalloc(22500 * sizeof(float));;
for(int x122=0; x122 < 22500; x122++) {
float x123 = (float)rand()/RAND_MAX;
float x124 = x123 - 0.5f;
float x125 = x124 * 0.01f;
x121[x122] = x125;

}
float* x129 = (float*)myMalloc(150 * sizeof(float));;
float* x130 = (float*)myMalloc(22500 * sizeof(float));;
for(int x131=0; x131 < 22500; x131++) {
float x132 = (float)rand()/RAND_MAX;
float x133 = x132 - 0.5f;
float x134 = x133 * 0.01f;
x130[x131] = x134;

}
float* x138 = (float*)myMalloc(22500 * sizeof(float));;
for(int x139=0; x139 < 22500; x139++) {
float x140 = (float)rand()/RAND_MAX;
float x141 = x140 - 0.5f;
float x142 = x141 * 0.01f;
x138[x139] = x142;

}
float* x146 = (float*)myMalloc(150 * sizeof(float));;
float* x147 = (float*)myMalloc(22500 * sizeof(float));;
for(int x148=0; x148 < 22500; x148++) {
float x149 = (float)rand()/RAND_MAX;
float x150 = x149 - 0.5f;
float x151 = x150 * 0.01f;
x147[x148] = x151;

}
float* x155 = (float*)myMalloc(22500 * sizeof(float));;
for(int x156=0; x156 < 22500; x156++) {
float x157 = (float)rand()/RAND_MAX;
float x158 = x157 - 0.5f;
float x159 = x158 * 0.01f;
x155[x156] = x159;

}
float* x163 = (float*)myMalloc(150 * sizeof(float));;
float* x164 = (float*)myMalloc(750 * sizeof(float));;
for(int x166=0; x166 < 750; x166++) {
float x167 = (float)rand()/RAND_MAX;
float x168 = x167 - 0.5f;
float x169 = x168 * 0.01f;
x164[x166] = x169;

}
float* x173 = (float*)myMalloc(5 * sizeof(float));;
float* x174 = (float*)myMalloc(45000 * sizeof(float));;
float* x175 = (float*)myMalloc(150 * sizeof(float));;
float* x176 = (float*)myMalloc(45000 * sizeof(float));;
float* x177 = (float*)myMalloc(150 * sizeof(float));;
float* x178 = (float*)myMalloc(45000 * sizeof(float));;
float* x179 = (float*)myMalloc(150 * sizeof(float));;
float* x180 = (float*)myMalloc(22500 * sizeof(float));;
float* x181 = (float*)myMalloc(22500 * sizeof(float));;
float* x182 = (float*)myMalloc(150 * sizeof(float));;
float* x183 = (float*)myMalloc(22500 * sizeof(float));;
float* x184 = (float*)myMalloc(22500 * sizeof(float));;
float* x185 = (float*)myMalloc(22500 * sizeof(float));;
float* x186 = (float*)myMalloc(22500 * sizeof(float));;
float* x187 = (float*)myMalloc(150 * sizeof(float));;
float* x188 = (float*)myMalloc(22500 * sizeof(float));;
float* x189 = (float*)myMalloc(22500 * sizeof(float));;
float* x190 = (float*)myMalloc(150 * sizeof(float));;
float* x191 = (float*)myMalloc(22500 * sizeof(float));;
float* x192 = (float*)myMalloc(22500 * sizeof(float));;
float* x193 = (float*)myMalloc(150 * sizeof(float));;
float* x194 = (float*)myMalloc(750 * sizeof(float));;
float* x195 = (float*)myMalloc(5 * sizeof(float));;
float* x196 = (float*)myMalloc(45000 * sizeof(float));;
float* x197 = (float*)myMalloc(150 * sizeof(float));;
float* x198 = (float*)myMalloc(45000 * sizeof(float));;
float* x199 = (float*)myMalloc(150 * sizeof(float));;
float* x200 = (float*)myMalloc(45000 * sizeof(float));;
float* x201 = (float*)myMalloc(150 * sizeof(float));;
float* x202 = (float*)myMalloc(22500 * sizeof(float));;
float* x203 = (float*)myMalloc(22500 * sizeof(float));;
float* x204 = (float*)myMalloc(150 * sizeof(float));;
float* x205 = (float*)myMalloc(22500 * sizeof(float));;
float* x206 = (float*)myMalloc(22500 * sizeof(float));;
float* x207 = (float*)myMalloc(22500 * sizeof(float));;
float* x208 = (float*)myMalloc(22500 * sizeof(float));;
float* x209 = (float*)myMalloc(150 * sizeof(float));;
float* x210 = (float*)myMalloc(22500 * sizeof(float));;
float* x211 = (float*)myMalloc(22500 * sizeof(float));;
float* x212 = (float*)myMalloc(150 * sizeof(float));;
float* x213 = (float*)myMalloc(22500 * sizeof(float));;
float* x214 = (float*)myMalloc(22500 * sizeof(float));;
float* x215 = (float*)myMalloc(150 * sizeof(float));;
float* x216 = (float*)myMalloc(750 * sizeof(float));;
float* x217 = (float*)myMalloc(5 * sizeof(float));;
double* x218 = (double*)myMalloc(6 * sizeof(double));;
int64_t x219 = (long)mallocAddr;
double x220 = ((double)clock() / CLOCKS_PER_SEC);
bool x1372 = true || true;
bool x1373 = x1372 || true;
for(int x222=0; x222 < 6; x222++) {
float x223 = 0.0f;
for(int x224=0; x224 < x24; x224++) {
float* x238 = (float*)myMalloc(1 * sizeof(float));;
float* x239 = (float*)myMalloc(1 * sizeof(float));;
float* x240 = (float*)myMalloc(150 * sizeof(float));;
float* x241 = (float*)myMalloc(150 * sizeof(float));;
float* x242 = (float*)myMalloc(150 * sizeof(float));;
float* x243 = (float*)myMalloc(150 * sizeof(float));;
int32_t x225 = x224 % x24;
int32_t x226 = x225 * 4;
int* x227 = x26[x226];
int32_t x228 = x226 + 1;
int* x229 = x26[x228];
int32_t x230 = x226 + 2;
int* x231 = x26[x230];
int32_t x232 = x226 + 3;
int* x233 = x26[x232];
function<void(int32_t,function<void(float**)>,float**)> x244 = [&](int32_t x245,function<void(float**)> x246,float** x247) {
float** x250 = x247;
float* x251 = x250[0];
float* x252 = x250[1];
float* x253 = x250[2];
float* x254 = x250[3];
float* x255 = x250[4];
float* x256 = x250[5];
int32_t x248 = x245;
bool x257 = x248 >= 0;
if (x257) {
int32_t x258 = x231[x248];
float** x2484 = (float**)myMalloc(6 * sizeof(float*));;
x2484[0] = x238;
x2484[1] = x239;
x2484[2] = x240;
x2484[3] = x241;
x2484[4] = x242;
x2484[5] = x243;
function<void(float**)> x249 = x246;
function<void(float**)> x575 = [&](float** x576) {
float* x577 = x576[0];
float* x578 = x576[1];
float* x579 = x576[2];
float* x580 = x576[3];
float* x581 = x576[4];
float* x582 = x576[5];
float** x583 = (float**)myMalloc(6 * sizeof(float*));;
x583[0] = x577;
x583[1] = x578;
x583[2] = x579;
x583[3] = x580;
x583[4] = x581;
x583[5] = x582;
x249(x583);
};
function<void(float**)> x567 = [&](float** x568) {
float* x569 = x568[0];
float* x570 = x568[1];
float* x571 = x568[2];
float* x572 = x568[3];
float* x573 = x568[4];
float* x574 = x568[5];
float** x592 = (float**)myMalloc(6 * sizeof(float*));;
x592[0] = x569;
x592[1] = x570;
x592[2] = x571;
x592[3] = x572;
x592[4] = x573;
x592[5] = x574;
x575(x592);
};
function<void(float**)> x1535 = [&](float** x1536) {
float* x1537 = x1536[0];
float* x1538 = x1536[1];
float* x1539 = x1536[2];
float* x1540 = x1536[3];
float* x1541 = x1536[4];
float* x1542 = x1536[5];
float** x1543 = (float**)myMalloc(6 * sizeof(float*));;
x1543[0] = x1537;
x1543[1] = x1538;
x1543[2] = x1539;
x1543[3] = x1540;
x1543[4] = x1541;
x1543[5] = x1542;
x249(x1543);
};
function<void(float**)> x1527 = [&](float** x1528) {
float* x1529 = x1528[0];
float* x1530 = x1528[1];
float* x1531 = x1528[2];
float* x1532 = x1528[3];
float* x1533 = x1528[4];
float* x1534 = x1528[5];
float** x1552 = (float**)myMalloc(6 * sizeof(float*));;
x1552[0] = x1529;
x1552[1] = x1530;
x1552[2] = x1531;
x1552[3] = x1532;
x1552[4] = x1533;
x1552[5] = x1534;
x1535(x1552);
};
function<void(float**)> x259 = [&](float** x260) {
float* x261 = x260[0];
float* x262 = x260[1];
float* x263 = x260[2];
float* x264 = x260[3];
float* x265 = x260[4];
float* x266 = x260[5];
int32_t x267 = x233[x248];
float** x2474 = (float**)myMalloc(6 * sizeof(float*));;
x2474[0] = x238;
x2474[1] = x239;
x2474[2] = x240;
x2474[3] = x241;
x2474[4] = x242;
x2474[5] = x243;
function<void(float**)> x268 = [&](float** x269) {
float* x270 = x269[0];
float* x271 = x269[1];
float* x272 = x269[2];
float* x273 = x269[3];
float* x274 = x269[4];
float* x275 = x269[5];
float* x276 = (float*)myMalloc(5 * sizeof(float));;
int32_t x277 = x227[x248];
x276[x277] = 1.0f;
float* x279 = (float*)myMalloc(5 * sizeof(float));;
int32_t x280 = x231[x248];
bool x281 = x280 < 0;
if (x281) {
int32_t x282 = x229[x248];
float* x283 = x7[x282];
float* x284 = (float*)myMalloc(300 * sizeof(float));;
float* x285 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x50,300,x283,1,0,x285,1);
float* x287 = (float*)myMalloc(150 * sizeof(float));;
float* x288 = (float*)myMalloc(150 * sizeof(float));;
int32_t x289 = 0;
int32_t x290 = 0;
int32_t x291 = 0;
for(int x293=0; x293 < 150; x293++) {
int32_t x294 = x289;
int32_t x295 = x290;
float x296 = x285[x295];
int32_t x297 = x291;
float x298 = x60[x297];
float x299 = x296 + x298;
x288[x294] = x299;
x289 += 1;
x290 += 1;
x291 += 1;

}
float* x306 = (float*)myMalloc(150 * sizeof(float));;
float* x307 = (float*)myMalloc(150 * sizeof(float));;
for(int x308=0; x308 < 150; x308++) {
float x309 = x288[x308];
float x310 = -1.0f * x309;
double x311 = (double)x310;
double x312 = exp(x311);
float x313 = (float)x312;
float x314 = x313 + 1.0f;
float x315 = 1.0f / x314;
x307[x308] = x315;

}
float* x319 = (float*)myMalloc(150 * sizeof(float));;
float* x320 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x61,300,x283,1,0,x320,1);
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
float x332 = x69[x331];
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
float* x354 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x70,300,x283,1,0,x354,1);
float* x356 = (float*)myMalloc(150 * sizeof(float));;
float* x357 = (float*)myMalloc(150 * sizeof(float));;
int32_t x358 = 0;
int32_t x359 = 0;
int32_t x360 = 0;
for(int x361=0; x361 < 150; x361++) {
int32_t x362 = x358;
int32_t x363 = x359;
float x364 = x354[x363];
int32_t x365 = x360;
float x366 = x78[x365];
float x367 = x364 + x366;
x357[x362] = x367;
x358 += 1;
x359 += 1;
x360 += 1;

}
float* x374 = (float*)myMalloc(150 * sizeof(float));;
float* x375 = (float*)myMalloc(150 * sizeof(float));;
for(int x376=0; x376 < 150; x376++) {
float x377 = x357[x376];
double x378 = (double)x377;
double x379 = tanh(x378);
float x380 = (float)x379;
x375[x376] = x380;

}
float* x384 = (float*)myMalloc(150 * sizeof(float));;
float* x385 = (float*)myMalloc(150 * sizeof(float));;
int32_t x386 = 0;
int32_t x387 = 0;
int32_t x388 = 0;
for(int x389=0; x389 < 150; x389++) {
int32_t x390 = x386;
int32_t x391 = x387;
float x392 = x307[x391];
int32_t x393 = x388;
float x394 = x375[x393];
float x395 = x392 * x394;
x385[x390] = x395;
x386 += 1;
x387 += 1;
x388 += 1;

}
float* x402 = (float*)myMalloc(150 * sizeof(float));;
float* x403 = (float*)myMalloc(150 * sizeof(float));;
for(int x404=0; x404 < 150; x404++) {
float x405 = x385[x404];
double x406 = (double)x405;
double x407 = tanh(x406);
float x408 = (float)x407;
x403[x404] = x408;

}
float* x412 = (float*)myMalloc(150 * sizeof(float));;
float* x413 = (float*)myMalloc(150 * sizeof(float));;
int32_t x414 = 0;
int32_t x415 = 0;
int32_t x416 = 0;
for(int x417=0; x417 < 150; x417++) {
int32_t x418 = x414;
int32_t x419 = x415;
float x420 = x341[x419];
int32_t x421 = x416;
float x422 = x403[x421];
float x423 = x420 * x422;
x413[x418] = x423;
x414 += 1;
x415 += 1;
x416 += 1;

}
float* x430 = (float*)myMalloc(150 * sizeof(float));;
float* x431 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x164,150,x413,1,0,x431,1);
float* x433 = (float*)myMalloc(5 * sizeof(float));;
float* x434 = (float*)myMalloc(5 * sizeof(float));;
int32_t x435 = 0;
int32_t x436 = 0;
int32_t x437 = 0;
for(int x439=0; x439 < 5; x439++) {
int32_t x440 = x435;
int32_t x441 = x436;
float x442 = x431[x441];
int32_t x443 = x437;
float x444 = x173[x443];
float x445 = x442 + x444;
x434[x440] = x445;
x435 += 1;
x436 += 1;
x437 += 1;

}
float* x452 = (float*)myMalloc(5 * sizeof(float));;
int32_t x453 = 0;
int32_t x454 = 1;
x454 *= 1;
x454 *= 5;
int32_t x457 = x453;
bool x458 = x457 >= 2;
if (x458) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x464 = x457 == 0;
if (x464) {
int32_t x465 = x454;
bool x466 = x465 == 5;
if (x466) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x473 = (float*)myMalloc(1 * sizeof(float));;
int32_t x474 = 0;
for(int x476=0; x476 < 1; x476++) {
float x477 = -3.4028235E38f;
for(int x478=0; x478 < 5; x478++) {
int32_t x479 = x474;
float x480 = x434[x479];
float x481 = x477;
bool x482 = x480 > x481;
if (x482) {
float x483 = x434[x479];
x477 = x483;
} else {
}
x474 += 1;

}
float x490 = x477;
x473[x476] = x490;

}
float* x494 = (float*)myMalloc(5 * sizeof(float));;
int32_t x495 = 0;
for(int x496=0; x496 < 1; x496++) {
for(int x497=0; x497 < 5; x497++) {
int32_t x498 = x495;
float x499 = x434[x498];
float x500 = x473[x496];
float x501 = x499 - x500;
double x502 = (double)x501;
double x503 = exp(x502);
float x504 = (float)x503;
x494[x498] = x504;
x495 += 1;

}

}
float* x511 = (float*)myMalloc(1 * sizeof(float));;
for(int x512=0; x512 < 1; x512++) {
int32_t x513 = x512;
int32_t x514 = x512 * 5;
int32_t x515 = x514;
for(int x516=0; x516 < 5; x516++) {
for(int x517=0; x517 < 1; x517++) {
int32_t x518 = x513;
int32_t x519 = x518 + x517;
float x520 = x511[x519];
int32_t x521 = x515;
int32_t x522 = x521 + x517;
float x523 = x494[x522];
float x524 = x520 + x523;
x511[x519] = x524;

}
x515 += 1;

}

}
x495 = 0;
for(int x534=0; x534 < 1; x534++) {
float x535 = x473[x534];
float x536 = x511[x534];
double x537 = (double)x536;
double x538 = log(x537);
float x539 = (float)x538;
float x540 = x535 + x539;
for(int x541=0; x541 < 5; x541++) {
int32_t x542 = x495;
float x543 = x434[x542];
float x544 = x543 - x540;
x494[x542] = x544;
x495 += 1;

}

}
float* x551 = (float*)myMalloc(5 * sizeof(float));;
int* x552 = x227+x248;
// nllLoss forward in CPU
float* x554 = (float*)myMalloc(1 * sizeof(float));;
int32_t x555 = 0;
for(int x556=0; x556 < 1; x556++) {
int32_t x557 = x555;
int32_t x558 = x552[x556];
int32_t x559 = x557 + x558;
float x560 = x494[x559];
float x561 = -1.0f * x560;
x554[x556] = x561;
x555 += 5;

}
float* x566 = (float*)myMalloc(1 * sizeof(float));;
float** x601 = (float**)myMalloc(6 * sizeof(float*));;
x601[0] = x554;
x601[1] = x566;
x601[2] = x413;
x601[3] = x430;
x601[4] = x385;
x601[5] = x402;
x567(x601);
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x611 = 0;
for(int x612=0; x612 < 1; x612++) {
int32_t x613 = x611;
int32_t x614 = x552[x612];
int32_t x615 = x613 + x614;
float x616 = x551[x615];
float x617 = x566[x612];
float x618 = -1.0f * x617;
float x619 = x616 + x618;
x551[x615] = x619;
x611 += 5;

}
float* x624 = (float*)myMalloc(1 * sizeof(float));;
for(int x625=0; x625 < 1; x625++) {
int32_t x626 = x625;
int32_t x627 = x625 * 5;
int32_t x628 = x627;
for(int x629=0; x629 < 5; x629++) {
for(int x630=0; x630 < 1; x630++) {
int32_t x631 = x626;
int32_t x632 = x631 + x630;
float x633 = x624[x632];
int32_t x634 = x628;
int32_t x635 = x634 + x630;
float x636 = x551[x635];
float x637 = x633 + x636;
x624[x632] = x637;

}
x628 += 1;

}

}
int32_t x646 = 0;
for(int x647=0; x647 < 1; x647++) {
for(int x648=0; x648 < 5; x648++) {
int32_t x649 = x646;
float x650 = x452[x649];
float x651 = x551[x649];
float x652 = x494[x649];
float x656 = x624[x647];
double x653 = (double)x652;
double x654 = exp(x653);
float x655 = (float)x654;
float x657 = x655 * x656;
float x658 = x651 - x657;
float x659 = x650 + x658;
x452[x649] = x659;
x646 += 1;

}

}
// back prop for + op
int32_t x667 = 0;
int32_t x668 = 0;
int32_t x669 = 0;
for(int x670=0; x670 < 5; x670++) {
int32_t x671 = x668;
float x672 = x433[x671];
int32_t x673 = x669;
float x674 = x452[x673];
float x675 = x672 + x674;
x433[x671] = x675;
x667 += 1;
x668 += 1;
x669 += 1;

}
int32_t x682 = 0;
int32_t x683 = 0;
int32_t x684 = 0;
for(int x685=0; x685 < 5; x685++) {
int32_t x686 = x683;
float x687 = x195[x686];
int32_t x688 = x684;
float x689 = x452[x688];
float x690 = x687 + x689;
x195[x686] = x690;
x682 += 1;
x683 += 1;
x684 += 1;

}
// add_cartesian
int32_t x698 = 0;
for(int x699=0; x699 < 5; x699++) {
for(int x700=0; x700 < 150; x700++) {
int32_t x701 = x698;
int32_t x702 = x701 + x700;
float x703 = x194[x702];
float x704 = x413[x700];
float x705 = x433[x699];
float x706 = x704 * x705;
float x707 = x703 + x706;
x194[x702] = x707;

}
x698 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x164,150,x433,1,1,x430,1);
// backprop for * op
int32_t x716 = 0;
int32_t x717 = 0;
int32_t x718 = 0;
for(int x719=0; x719 < 150; x719++) {
int32_t x720 = x716;
float x721 = x353[x720];
float x722 = x341[x720];
int32_t x723 = x717;
float x724 = x403[x723];
int32_t x725 = x718;
float x726 = x430[x725];
float x727 = x726 * x724;
float x728 = x721 + x727;
x353[x720] = x728;
float x730 = x412[x723];
float x731 = x341[x720];
float x732 = x403[x723];
float x733 = x430[x725];
float x734 = x733 * x731;
float x735 = x730 + x734;
x412[x723] = x735;
x718 += 1;
x716 += 1;
x717 += 1;

}
for(int x742=0; x742 < 150; x742++) {
float x743 = x402[x742];
float x744 = x403[x742];
float x747 = x412[x742];
float x745 = x744 * x744;
float x746 = 1.0f - x745;
float x748 = x746 * x747;
float x749 = x743 + x748;
x402[x742] = x749;

}
// backprop for * op
int32_t x754 = 0;
int32_t x755 = 0;
int32_t x756 = 0;
for(int x757=0; x757 < 150; x757++) {
int32_t x758 = x754;
float x759 = x319[x758];
float x760 = x307[x758];
int32_t x761 = x755;
float x762 = x375[x761];
int32_t x763 = x756;
float x764 = x402[x763];
float x765 = x764 * x762;
float x766 = x759 + x765;
x319[x758] = x766;
float x768 = x384[x761];
float x769 = x307[x758];
float x770 = x375[x761];
float x771 = x402[x763];
float x772 = x771 * x769;
float x773 = x768 + x772;
x384[x761] = x773;
x756 += 1;
x754 += 1;
x755 += 1;

}
for(int x780=0; x780 < 150; x780++) {
float x781 = x374[x780];
float x782 = x375[x780];
float x785 = x384[x780];
float x783 = x782 * x782;
float x784 = 1.0f - x783;
float x786 = x784 * x785;
float x787 = x781 + x786;
x374[x780] = x787;

}
// back prop for + op
int32_t x792 = 0;
int32_t x793 = 0;
int32_t x794 = 0;
for(int x795=0; x795 < 150; x795++) {
int32_t x796 = x793;
float x797 = x356[x796];
int32_t x798 = x794;
float x799 = x374[x798];
float x800 = x797 + x799;
x356[x796] = x800;
x792 += 1;
x793 += 1;
x794 += 1;

}
int32_t x807 = 0;
int32_t x808 = 0;
int32_t x809 = 0;
for(int x810=0; x810 < 150; x810++) {
int32_t x811 = x808;
float x812 = x179[x811];
int32_t x813 = x809;
float x814 = x374[x813];
float x815 = x812 + x814;
x179[x811] = x815;
x807 += 1;
x808 += 1;
x809 += 1;

}
// add_cartesian
int32_t x823 = 0;
for(int x824=0; x824 < 150; x824++) {
for(int x825=0; x825 < 300; x825++) {
int32_t x826 = x823;
int32_t x827 = x826 + x825;
float x828 = x178[x827];
float x829 = x283[x825];
float x830 = x356[x824];
float x831 = x829 * x830;
float x832 = x828 + x831;
x178[x827] = x832;

}
x823 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x70,300,x356,1,1,x284,1);
for(int x840=0; x840 < 150; x840++) {
float x841 = x340[x840];
float x842 = x341[x840];
float x845 = x353[x840];
float x843 = 1.0f - x842;
float x844 = x843 * x842;
float x846 = x844 * x845;
float x847 = x841 + x846;
x340[x840] = x847;

}
// back prop for + op
int32_t x852 = 0;
int32_t x853 = 0;
int32_t x854 = 0;
for(int x855=0; x855 < 150; x855++) {
int32_t x856 = x853;
float x857 = x322[x856];
int32_t x858 = x854;
float x859 = x340[x858];
float x860 = x857 + x859;
x322[x856] = x860;
x852 += 1;
x853 += 1;
x854 += 1;

}
int32_t x867 = 0;
int32_t x868 = 0;
int32_t x869 = 0;
for(int x870=0; x870 < 150; x870++) {
int32_t x871 = x868;
float x872 = x177[x871];
int32_t x873 = x869;
float x874 = x340[x873];
float x875 = x872 + x874;
x177[x871] = x875;
x867 += 1;
x868 += 1;
x869 += 1;

}
// add_cartesian
int32_t x883 = 0;
for(int x884=0; x884 < 150; x884++) {
for(int x885=0; x885 < 300; x885++) {
int32_t x886 = x883;
int32_t x887 = x886 + x885;
float x888 = x176[x887];
float x889 = x283[x885];
float x890 = x322[x884];
float x891 = x889 * x890;
float x892 = x888 + x891;
x176[x887] = x892;

}
x883 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x61,300,x322,1,1,x284,1);
for(int x900=0; x900 < 150; x900++) {
float x901 = x306[x900];
float x902 = x307[x900];
float x905 = x319[x900];
float x903 = 1.0f - x902;
float x904 = x903 * x902;
float x906 = x904 * x905;
float x907 = x901 + x906;
x306[x900] = x907;

}
// back prop for + op
int32_t x912 = 0;
int32_t x913 = 0;
int32_t x914 = 0;
for(int x915=0; x915 < 150; x915++) {
int32_t x916 = x913;
float x917 = x287[x916];
int32_t x918 = x914;
float x919 = x306[x918];
float x920 = x917 + x919;
x287[x916] = x920;
x912 += 1;
x913 += 1;
x914 += 1;

}
int32_t x927 = 0;
int32_t x928 = 0;
int32_t x929 = 0;
for(int x930=0; x930 < 150; x930++) {
int32_t x931 = x928;
float x932 = x175[x931];
int32_t x933 = x929;
float x934 = x306[x933];
float x935 = x932 + x934;
x175[x931] = x935;
x927 += 1;
x928 += 1;
x929 += 1;

}
// add_cartesian
int32_t x943 = 0;
for(int x944=0; x944 < 150; x944++) {
for(int x945=0; x945 < 300; x945++) {
int32_t x946 = x943;
int32_t x947 = x946 + x945;
float x948 = x174[x947];
float x949 = x283[x945];
float x950 = x287[x944];
float x951 = x949 * x950;
float x952 = x948 + x951;
x174[x947] = x952;

}
x943 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x50,300,x287,1,1,x284,1);
} else {
float* x961 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x79,150,x263,1,0,x961,1);
float* x963 = (float*)myMalloc(150 * sizeof(float));;
float* x964 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x88,150,x272,1,0,x964,1);
float* x966 = (float*)myMalloc(150 * sizeof(float));;
float* x967 = (float*)myMalloc(150 * sizeof(float));;
int32_t x968 = 0;
int32_t x969 = 0;
int32_t x970 = 0;
for(int x971=0; x971 < 150; x971++) {
int32_t x972 = x968;
int32_t x973 = x969;
float x974 = x961[x973];
int32_t x975 = x970;
float x976 = x964[x975];
float x977 = x974 + x976;
x967[x972] = x977;
x968 += 1;
x969 += 1;
x970 += 1;

}
float* x984 = (float*)myMalloc(150 * sizeof(float));;
float* x985 = (float*)myMalloc(150 * sizeof(float));;
int32_t x986 = 0;
int32_t x987 = 0;
int32_t x988 = 0;
for(int x989=0; x989 < 150; x989++) {
int32_t x990 = x986;
int32_t x991 = x987;
float x992 = x967[x991];
int32_t x993 = x988;
float x994 = x96[x993];
float x995 = x992 + x994;
x985[x990] = x995;
x986 += 1;
x987 += 1;
x988 += 1;

}
float* x1002 = (float*)myMalloc(150 * sizeof(float));;
float* x1003 = (float*)myMalloc(150 * sizeof(float));;
for(int x1004=0; x1004 < 150; x1004++) {
float x1005 = x985[x1004];
float x1006 = -1.0f * x1005;
double x1007 = (double)x1006;
double x1008 = exp(x1007);
float x1009 = (float)x1008;
float x1010 = x1009 + 1.0f;
float x1011 = 1.0f / x1010;
x1003[x1004] = x1011;

}
float* x1015 = (float*)myMalloc(150 * sizeof(float));;
float* x1016 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x97,150,x263,1,0,x1016,1);
float* x1018 = (float*)myMalloc(150 * sizeof(float));;
float* x1019 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x105,150,x272,1,0,x1019,1);
float* x1021 = (float*)myMalloc(150 * sizeof(float));;
float* x1022 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1023 = 0;
int32_t x1024 = 0;
int32_t x1025 = 0;
for(int x1026=0; x1026 < 150; x1026++) {
int32_t x1027 = x1023;
int32_t x1028 = x1024;
float x1029 = x1016[x1028];
int32_t x1030 = x1025;
float x1031 = x1019[x1030];
float x1032 = x1029 + x1031;
x1022[x1027] = x1032;
x1023 += 1;
x1024 += 1;
x1025 += 1;

}
float* x1039 = (float*)myMalloc(150 * sizeof(float));;
float* x1040 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1041 = 0;
int32_t x1042 = 0;
int32_t x1043 = 0;
for(int x1044=0; x1044 < 150; x1044++) {
int32_t x1045 = x1041;
int32_t x1046 = x1042;
float x1047 = x1022[x1046];
int32_t x1048 = x1043;
float x1049 = x129[x1048];
float x1050 = x1047 + x1049;
x1040[x1045] = x1050;
x1041 += 1;
x1042 += 1;
x1043 += 1;

}
float* x1057 = (float*)myMalloc(150 * sizeof(float));;
float* x1058 = (float*)myMalloc(150 * sizeof(float));;
for(int x1059=0; x1059 < 150; x1059++) {
float x1060 = x1040[x1059];
float x1061 = -1.0f * x1060;
double x1062 = (double)x1061;
double x1063 = exp(x1062);
float x1064 = (float)x1063;
float x1065 = x1064 + 1.0f;
float x1066 = 1.0f / x1065;
x1058[x1059] = x1066;

}
float* x1070 = (float*)myMalloc(150 * sizeof(float));;
float* x1071 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x113,150,x263,1,0,x1071,1);
float* x1073 = (float*)myMalloc(150 * sizeof(float));;
float* x1074 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x121,150,x272,1,0,x1074,1);
float* x1076 = (float*)myMalloc(150 * sizeof(float));;
float* x1077 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1078 = 0;
int32_t x1079 = 0;
int32_t x1080 = 0;
for(int x1081=0; x1081 < 150; x1081++) {
int32_t x1082 = x1078;
int32_t x1083 = x1079;
float x1084 = x1071[x1083];
int32_t x1085 = x1080;
float x1086 = x1074[x1085];
float x1087 = x1084 + x1086;
x1077[x1082] = x1087;
x1078 += 1;
x1079 += 1;
x1080 += 1;

}
float* x1094 = (float*)myMalloc(150 * sizeof(float));;
float* x1095 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1096 = 0;
int32_t x1097 = 0;
int32_t x1098 = 0;
for(int x1099=0; x1099 < 150; x1099++) {
int32_t x1100 = x1096;
int32_t x1101 = x1097;
float x1102 = x1077[x1101];
int32_t x1103 = x1098;
float x1104 = x129[x1103];
float x1105 = x1102 + x1104;
x1095[x1100] = x1105;
x1096 += 1;
x1097 += 1;
x1098 += 1;

}
float* x1112 = (float*)myMalloc(150 * sizeof(float));;
float* x1113 = (float*)myMalloc(150 * sizeof(float));;
for(int x1114=0; x1114 < 150; x1114++) {
float x1115 = x1095[x1114];
float x1116 = -1.0f * x1115;
double x1117 = (double)x1116;
double x1118 = exp(x1117);
float x1119 = (float)x1118;
float x1120 = x1119 + 1.0f;
float x1121 = 1.0f / x1120;
x1113[x1114] = x1121;

}
float* x1125 = (float*)myMalloc(150 * sizeof(float));;
float* x1126 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x130,150,x263,1,0,x1126,1);
float* x1128 = (float*)myMalloc(150 * sizeof(float));;
float* x1129 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x138,150,x272,1,0,x1129,1);
float* x1131 = (float*)myMalloc(150 * sizeof(float));;
float* x1132 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1133 = 0;
int32_t x1134 = 0;
int32_t x1135 = 0;
for(int x1136=0; x1136 < 150; x1136++) {
int32_t x1137 = x1133;
int32_t x1138 = x1134;
float x1139 = x1126[x1138];
int32_t x1140 = x1135;
float x1141 = x1129[x1140];
float x1142 = x1139 + x1141;
x1132[x1137] = x1142;
x1133 += 1;
x1134 += 1;
x1135 += 1;

}
float* x1149 = (float*)myMalloc(150 * sizeof(float));;
float* x1150 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1151 = 0;
int32_t x1152 = 0;
int32_t x1153 = 0;
for(int x1154=0; x1154 < 150; x1154++) {
int32_t x1155 = x1151;
int32_t x1156 = x1152;
float x1157 = x1132[x1156];
int32_t x1158 = x1153;
float x1159 = x146[x1158];
float x1160 = x1157 + x1159;
x1150[x1155] = x1160;
x1151 += 1;
x1152 += 1;
x1153 += 1;

}
float* x1167 = (float*)myMalloc(150 * sizeof(float));;
float* x1168 = (float*)myMalloc(150 * sizeof(float));;
for(int x1169=0; x1169 < 150; x1169++) {
float x1170 = x1150[x1169];
float x1171 = -1.0f * x1170;
double x1172 = (double)x1171;
double x1173 = exp(x1172);
float x1174 = (float)x1173;
float x1175 = x1174 + 1.0f;
float x1176 = 1.0f / x1175;
x1168[x1169] = x1176;

}
float* x1180 = (float*)myMalloc(150 * sizeof(float));;
float* x1181 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x147,150,x263,1,0,x1181,1);
float* x1183 = (float*)myMalloc(150 * sizeof(float));;
float* x1184 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x155,150,x272,1,0,x1184,1);
float* x1186 = (float*)myMalloc(150 * sizeof(float));;
float* x1187 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1188 = 0;
int32_t x1189 = 0;
int32_t x1190 = 0;
for(int x1191=0; x1191 < 150; x1191++) {
int32_t x1192 = x1188;
int32_t x1193 = x1189;
float x1194 = x1181[x1193];
int32_t x1195 = x1190;
float x1196 = x1184[x1195];
float x1197 = x1194 + x1196;
x1187[x1192] = x1197;
x1188 += 1;
x1189 += 1;
x1190 += 1;

}
float* x1204 = (float*)myMalloc(150 * sizeof(float));;
float* x1205 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1206 = 0;
int32_t x1207 = 0;
int32_t x1208 = 0;
for(int x1209=0; x1209 < 150; x1209++) {
int32_t x1210 = x1206;
int32_t x1211 = x1207;
float x1212 = x1187[x1211];
int32_t x1213 = x1208;
float x1214 = x163[x1213];
float x1215 = x1212 + x1214;
x1205[x1210] = x1215;
x1206 += 1;
x1207 += 1;
x1208 += 1;

}
float* x1222 = (float*)myMalloc(150 * sizeof(float));;
float* x1223 = (float*)myMalloc(150 * sizeof(float));;
for(int x1224=0; x1224 < 150; x1224++) {
float x1225 = x1205[x1224];
double x1226 = (double)x1225;
double x1227 = tanh(x1226);
float x1228 = (float)x1227;
x1223[x1224] = x1228;

}
float* x1232 = (float*)myMalloc(150 * sizeof(float));;
float* x1233 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1234 = 0;
int32_t x1235 = 0;
int32_t x1236 = 0;
for(int x1237=0; x1237 < 150; x1237++) {
int32_t x1238 = x1234;
int32_t x1239 = x1235;
float x1240 = x1003[x1239];
int32_t x1241 = x1236;
float x1242 = x1223[x1241];
float x1243 = x1240 * x1242;
x1233[x1238] = x1243;
x1234 += 1;
x1235 += 1;
x1236 += 1;

}
float* x1250 = (float*)myMalloc(150 * sizeof(float));;
float* x1251 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1252 = 0;
int32_t x1253 = 0;
int32_t x1254 = 0;
for(int x1255=0; x1255 < 150; x1255++) {
int32_t x1256 = x1252;
int32_t x1257 = x1253;
float x1258 = x1058[x1257];
int32_t x1259 = x1254;
float x1260 = x265[x1259];
float x1261 = x1258 * x1260;
x1251[x1256] = x1261;
x1252 += 1;
x1253 += 1;
x1254 += 1;

}
float* x1268 = (float*)myMalloc(150 * sizeof(float));;
float* x1269 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1270 = 0;
int32_t x1271 = 0;
int32_t x1272 = 0;
for(int x1273=0; x1273 < 150; x1273++) {
int32_t x1274 = x1270;
int32_t x1275 = x1271;
float x1276 = x1233[x1275];
int32_t x1277 = x1272;
float x1278 = x1251[x1277];
float x1279 = x1276 + x1278;
x1269[x1274] = x1279;
x1270 += 1;
x1271 += 1;
x1272 += 1;

}
float* x1286 = (float*)myMalloc(150 * sizeof(float));;
float* x1287 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1288 = 0;
int32_t x1289 = 0;
int32_t x1290 = 0;
for(int x1291=0; x1291 < 150; x1291++) {
int32_t x1292 = x1288;
int32_t x1293 = x1289;
float x1294 = x1113[x1293];
int32_t x1295 = x1290;
float x1296 = x274[x1295];
float x1297 = x1294 * x1296;
x1287[x1292] = x1297;
x1288 += 1;
x1289 += 1;
x1290 += 1;

}
float* x1304 = (float*)myMalloc(150 * sizeof(float));;
float* x1305 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1306 = 0;
int32_t x1307 = 0;
int32_t x1308 = 0;
for(int x1309=0; x1309 < 150; x1309++) {
int32_t x1310 = x1306;
int32_t x1311 = x1307;
float x1312 = x1269[x1311];
int32_t x1313 = x1308;
float x1314 = x1287[x1313];
float x1315 = x1312 + x1314;
x1305[x1310] = x1315;
x1306 += 1;
x1307 += 1;
x1308 += 1;

}
float* x1322 = (float*)myMalloc(150 * sizeof(float));;
float* x1323 = (float*)myMalloc(150 * sizeof(float));;
for(int x1324=0; x1324 < 150; x1324++) {
float x1325 = x1305[x1324];
double x1326 = (double)x1325;
double x1327 = tanh(x1326);
float x1328 = (float)x1327;
x1323[x1324] = x1328;

}
float* x1332 = (float*)myMalloc(150 * sizeof(float));;
float* x1333 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1334 = 0;
int32_t x1335 = 0;
int32_t x1336 = 0;
for(int x1337=0; x1337 < 150; x1337++) {
int32_t x1338 = x1334;
int32_t x1339 = x1335;
float x1340 = x1168[x1339];
int32_t x1341 = x1336;
float x1342 = x1323[x1341];
float x1343 = x1340 * x1342;
x1333[x1338] = x1343;
x1334 += 1;
x1335 += 1;
x1336 += 1;

}
float* x1350 = (float*)myMalloc(150 * sizeof(float));;
float* x1351 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x164,150,x1333,1,0,x1351,1);
float* x1353 = (float*)myMalloc(5 * sizeof(float));;
float* x1354 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1355 = 0;
int32_t x1356 = 0;
int32_t x1357 = 0;
for(int x1358=0; x1358 < 5; x1358++) {
int32_t x1359 = x1355;
int32_t x1360 = x1356;
float x1361 = x1351[x1360];
int32_t x1362 = x1357;
float x1363 = x173[x1362];
float x1364 = x1361 + x1363;
x1354[x1359] = x1364;
x1355 += 1;
x1356 += 1;
x1357 += 1;

}
float* x1371 = (float*)myMalloc(5 * sizeof(float));;
if (x1373) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1379 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1380 = 0;
int32_t x1381 = 0;
int32_t x1382 = 0;
for(int x1383=0; x1383 < 1; x1383++) {
int32_t x1384 = x1380;
int32_t x1385 = x1381;
float x1386 = x261[x1385];
int32_t x1387 = x1382;
float x1388 = x270[x1387];
float x1389 = x1386 + x1388;
x1379[x1384] = x1389;
x1380 += 1;

}
float* x1394 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1395 = 0;
int32_t x1396 = 1;
x1396 *= 1;
x1396 *= 5;
int32_t x1399 = x1395;
bool x1400 = x1399 >= 2;
if (x1400) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1405 = x1399 == 0;
if (x1405) {
int32_t x1406 = x1396;
bool x1407 = x1406 == 5;
if (x1407) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x1414 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1415 = 0;
for(int x1416=0; x1416 < 1; x1416++) {
float x1417 = -3.4028235E38f;
for(int x1418=0; x1418 < 5; x1418++) {
int32_t x1419 = x1415;
float x1420 = x1354[x1419];
float x1421 = x1417;
bool x1422 = x1420 > x1421;
if (x1422) {
float x1423 = x1354[x1419];
x1417 = x1423;
} else {
}
x1415 += 1;

}
float x1430 = x1417;
x1414[x1416] = x1430;

}
float* x1434 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1435 = 0;
for(int x1436=0; x1436 < 1; x1436++) {
for(int x1437=0; x1437 < 5; x1437++) {
int32_t x1438 = x1435;
float x1439 = x1354[x1438];
float x1440 = x1414[x1436];
float x1441 = x1439 - x1440;
double x1442 = (double)x1441;
double x1443 = exp(x1442);
float x1444 = (float)x1443;
x1434[x1438] = x1444;
x1435 += 1;

}

}
float* x1451 = (float*)myMalloc(1 * sizeof(float));;
for(int x1452=0; x1452 < 1; x1452++) {
int32_t x1453 = x1452;
int32_t x1454 = x1452 * 5;
int32_t x1455 = x1454;
for(int x1456=0; x1456 < 5; x1456++) {
for(int x1457=0; x1457 < 1; x1457++) {
int32_t x1458 = x1453;
int32_t x1459 = x1458 + x1457;
float x1460 = x1451[x1459];
int32_t x1461 = x1455;
int32_t x1462 = x1461 + x1457;
float x1463 = x1434[x1462];
float x1464 = x1460 + x1463;
x1451[x1459] = x1464;

}
x1455 += 1;

}

}
x1435 = 0;
for(int x1474=0; x1474 < 1; x1474++) {
float x1475 = x1414[x1474];
float x1476 = x1451[x1474];
double x1477 = (double)x1476;
double x1478 = log(x1477);
float x1479 = (float)x1478;
float x1480 = x1475 + x1479;
for(int x1481=0; x1481 < 5; x1481++) {
int32_t x1482 = x1435;
float x1483 = x1354[x1482];
float x1484 = x1483 - x1480;
x1434[x1482] = x1484;
x1435 += 1;

}

}
float* x1491 = (float*)myMalloc(5 * sizeof(float));;
int* x1492 = x227+x248;
// nllLoss forward in CPU
float* x1494 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1495 = 0;
for(int x1496=0; x1496 < 1; x1496++) {
int32_t x1497 = x1495;
int32_t x1498 = x1492[x1496];
int32_t x1499 = x1497 + x1498;
float x1500 = x1434[x1499];
float x1501 = -1.0f * x1500;
x1494[x1496] = x1501;
x1495 += 5;

}
float* x1506 = (float*)myMalloc(1 * sizeof(float));;
if (x1373) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1511 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1512 = 0;
int32_t x1513 = 0;
int32_t x1514 = 0;
for(int x1515=0; x1515 < 1; x1515++) {
int32_t x1516 = x1512;
int32_t x1517 = x1513;
float x1518 = x1379[x1517];
int32_t x1519 = x1514;
float x1520 = x1494[x1519];
float x1521 = x1518 + x1520;
x1511[x1516] = x1521;
x1512 += 1;

}
float* x1526 = (float*)myMalloc(1 * sizeof(float));;
float** x1561 = (float**)myMalloc(6 * sizeof(float*));;
x1561[0] = x1511;
x1561[1] = x1526;
x1561[2] = x1333;
x1561[3] = x1350;
x1561[4] = x1305;
x1561[5] = x1322;
x1527(x1561);
// back prop for + op
if (x1373) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x1574 = 0;
int32_t x1575 = 0;
int32_t x1576 = 0;
for(int x1577=0; x1577 < 1; x1577++) {
int32_t x1578 = x1575;
float x1579 = x1394[x1578];
int32_t x1580 = x1576;
float x1581 = x1526[x1580];
float x1582 = x1579 + x1581;
x1394[x1578] = x1582;
x1574 += 1;

}
if (x1373) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x1591 = 0;
int32_t x1592 = 0;
int32_t x1593 = 0;
for(int x1594=0; x1594 < 1; x1594++) {
int32_t x1595 = x1592;
float x1596 = x1506[x1595];
int32_t x1597 = x1593;
float x1598 = x1526[x1597];
float x1599 = x1596 + x1598;
x1506[x1595] = x1599;
x1591 += 1;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x1606 = 0;
for(int x1607=0; x1607 < 1; x1607++) {
int32_t x1608 = x1606;
int32_t x1609 = x1492[x1607];
int32_t x1610 = x1608 + x1609;
float x1611 = x1491[x1610];
float x1612 = x1506[x1607];
float x1613 = -1.0f * x1612;
float x1614 = x1611 + x1613;
x1491[x1610] = x1614;
x1606 += 5;

}
float* x1619 = (float*)myMalloc(1 * sizeof(float));;
for(int x1620=0; x1620 < 1; x1620++) {
int32_t x1621 = x1620;
int32_t x1622 = x1620 * 5;
int32_t x1623 = x1622;
for(int x1624=0; x1624 < 5; x1624++) {
for(int x1625=0; x1625 < 1; x1625++) {
int32_t x1626 = x1621;
int32_t x1627 = x1626 + x1625;
float x1628 = x1619[x1627];
int32_t x1629 = x1623;
int32_t x1630 = x1629 + x1625;
float x1631 = x1491[x1630];
float x1632 = x1628 + x1631;
x1619[x1627] = x1632;

}
x1623 += 1;

}

}
int32_t x1641 = 0;
for(int x1642=0; x1642 < 1; x1642++) {
for(int x1643=0; x1643 < 5; x1643++) {
int32_t x1644 = x1641;
float x1645 = x1371[x1644];
float x1646 = x1491[x1644];
float x1647 = x1434[x1644];
float x1651 = x1619[x1642];
double x1648 = (double)x1647;
double x1649 = exp(x1648);
float x1650 = (float)x1649;
float x1652 = x1650 * x1651;
float x1653 = x1646 - x1652;
float x1654 = x1645 + x1653;
x1371[x1644] = x1654;
x1641 += 1;

}

}
// back prop for + op
if (x1373) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x1666 = 0;
int32_t x1667 = 0;
int32_t x1668 = 0;
for(int x1669=0; x1669 < 1; x1669++) {
int32_t x1670 = x1667;
float x1671 = x262[x1670];
int32_t x1672 = x1668;
float x1673 = x1394[x1672];
float x1674 = x1671 + x1673;
x262[x1670] = x1674;
x1666 += 1;

}
if (x1373) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x1683 = 0;
int32_t x1684 = 0;
int32_t x1685 = 0;
for(int x1686=0; x1686 < 1; x1686++) {
int32_t x1687 = x1684;
float x1688 = x271[x1687];
int32_t x1689 = x1685;
float x1690 = x1394[x1689];
float x1691 = x1688 + x1690;
x271[x1687] = x1691;
x1683 += 1;

}
// back prop for + op
int32_t x1697 = 0;
int32_t x1698 = 0;
int32_t x1699 = 0;
for(int x1700=0; x1700 < 5; x1700++) {
int32_t x1701 = x1698;
float x1702 = x1353[x1701];
int32_t x1703 = x1699;
float x1704 = x1371[x1703];
float x1705 = x1702 + x1704;
x1353[x1701] = x1705;
x1697 += 1;
x1698 += 1;
x1699 += 1;

}
int32_t x1712 = 0;
int32_t x1713 = 0;
int32_t x1714 = 0;
for(int x1715=0; x1715 < 5; x1715++) {
int32_t x1716 = x1713;
float x1717 = x195[x1716];
int32_t x1718 = x1714;
float x1719 = x1371[x1718];
float x1720 = x1717 + x1719;
x195[x1716] = x1720;
x1712 += 1;
x1713 += 1;
x1714 += 1;

}
// add_cartesian
int32_t x1728 = 0;
for(int x1729=0; x1729 < 5; x1729++) {
for(int x1730=0; x1730 < 150; x1730++) {
int32_t x1731 = x1728;
int32_t x1732 = x1731 + x1730;
float x1733 = x194[x1732];
float x1734 = x1333[x1730];
float x1735 = x1353[x1729];
float x1736 = x1734 * x1735;
float x1737 = x1733 + x1736;
x194[x1732] = x1737;

}
x1728 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x164,150,x1353,1,1,x1350,1);
// backprop for * op
int32_t x1746 = 0;
int32_t x1747 = 0;
int32_t x1748 = 0;
for(int x1749=0; x1749 < 150; x1749++) {
int32_t x1750 = x1746;
float x1751 = x1180[x1750];
float x1752 = x1168[x1750];
int32_t x1753 = x1747;
float x1754 = x1323[x1753];
int32_t x1755 = x1748;
float x1756 = x1350[x1755];
float x1757 = x1756 * x1754;
float x1758 = x1751 + x1757;
x1180[x1750] = x1758;
float x1760 = x1332[x1753];
float x1761 = x1168[x1750];
float x1762 = x1323[x1753];
float x1763 = x1350[x1755];
float x1764 = x1763 * x1761;
float x1765 = x1760 + x1764;
x1332[x1753] = x1765;
x1748 += 1;
x1746 += 1;
x1747 += 1;

}
for(int x1772=0; x1772 < 150; x1772++) {
float x1773 = x1322[x1772];
float x1774 = x1323[x1772];
float x1777 = x1332[x1772];
float x1775 = x1774 * x1774;
float x1776 = 1.0f - x1775;
float x1778 = x1776 * x1777;
float x1779 = x1773 + x1778;
x1322[x1772] = x1779;

}
// back prop for + op
int32_t x1784 = 0;
int32_t x1785 = 0;
int32_t x1786 = 0;
for(int x1787=0; x1787 < 150; x1787++) {
int32_t x1788 = x1785;
float x1789 = x1286[x1788];
int32_t x1790 = x1786;
float x1791 = x1322[x1790];
float x1792 = x1789 + x1791;
x1286[x1788] = x1792;
x1784 += 1;
x1785 += 1;
x1786 += 1;

}
int32_t x1799 = 0;
int32_t x1800 = 0;
int32_t x1801 = 0;
for(int x1802=0; x1802 < 150; x1802++) {
int32_t x1803 = x1800;
float x1804 = x1304[x1803];
int32_t x1805 = x1801;
float x1806 = x1322[x1805];
float x1807 = x1804 + x1806;
x1304[x1803] = x1807;
x1799 += 1;
x1800 += 1;
x1801 += 1;

}
// backprop for * op
int32_t x1815 = 0;
int32_t x1816 = 0;
int32_t x1817 = 0;
for(int x1818=0; x1818 < 150; x1818++) {
int32_t x1819 = x1815;
float x1820 = x1125[x1819];
float x1821 = x1113[x1819];
int32_t x1822 = x1816;
float x1823 = x274[x1822];
int32_t x1824 = x1817;
float x1825 = x1304[x1824];
float x1826 = x1825 * x1823;
float x1827 = x1820 + x1826;
x1125[x1819] = x1827;
float x1829 = x275[x1822];
float x1830 = x1113[x1819];
float x1831 = x274[x1822];
float x1832 = x1304[x1824];
float x1833 = x1832 * x1830;
float x1834 = x1829 + x1833;
x275[x1822] = x1834;
x1817 += 1;
x1815 += 1;
x1816 += 1;

}
// back prop for + op
int32_t x1842 = 0;
int32_t x1843 = 0;
int32_t x1844 = 0;
for(int x1845=0; x1845 < 150; x1845++) {
int32_t x1846 = x1843;
float x1847 = x1250[x1846];
int32_t x1848 = x1844;
float x1849 = x1286[x1848];
float x1850 = x1847 + x1849;
x1250[x1846] = x1850;
x1842 += 1;
x1843 += 1;
x1844 += 1;

}
int32_t x1857 = 0;
int32_t x1858 = 0;
int32_t x1859 = 0;
for(int x1860=0; x1860 < 150; x1860++) {
int32_t x1861 = x1858;
float x1862 = x1268[x1861];
int32_t x1863 = x1859;
float x1864 = x1286[x1863];
float x1865 = x1862 + x1864;
x1268[x1861] = x1865;
x1857 += 1;
x1858 += 1;
x1859 += 1;

}
// backprop for * op
int32_t x1873 = 0;
int32_t x1874 = 0;
int32_t x1875 = 0;
for(int x1876=0; x1876 < 150; x1876++) {
int32_t x1877 = x1873;
float x1878 = x1070[x1877];
float x1879 = x1058[x1877];
int32_t x1880 = x1874;
float x1881 = x265[x1880];
int32_t x1882 = x1875;
float x1883 = x1268[x1882];
float x1884 = x1883 * x1881;
float x1885 = x1878 + x1884;
x1070[x1877] = x1885;
float x1887 = x266[x1880];
float x1888 = x1058[x1877];
float x1889 = x265[x1880];
float x1890 = x1268[x1882];
float x1891 = x1890 * x1888;
float x1892 = x1887 + x1891;
x266[x1880] = x1892;
x1875 += 1;
x1873 += 1;
x1874 += 1;

}
// backprop for * op
int32_t x1900 = 0;
int32_t x1901 = 0;
int32_t x1902 = 0;
for(int x1903=0; x1903 < 150; x1903++) {
int32_t x1904 = x1900;
float x1905 = x1015[x1904];
float x1906 = x1003[x1904];
int32_t x1907 = x1901;
float x1908 = x1223[x1907];
int32_t x1909 = x1902;
float x1910 = x1250[x1909];
float x1911 = x1910 * x1908;
float x1912 = x1905 + x1911;
x1015[x1904] = x1912;
float x1914 = x1232[x1907];
float x1915 = x1003[x1904];
float x1916 = x1223[x1907];
float x1917 = x1250[x1909];
float x1918 = x1917 * x1915;
float x1919 = x1914 + x1918;
x1232[x1907] = x1919;
x1902 += 1;
x1900 += 1;
x1901 += 1;

}
for(int x1926=0; x1926 < 150; x1926++) {
float x1927 = x1222[x1926];
float x1928 = x1223[x1926];
float x1931 = x1232[x1926];
float x1929 = x1928 * x1928;
float x1930 = 1.0f - x1929;
float x1932 = x1930 * x1931;
float x1933 = x1927 + x1932;
x1222[x1926] = x1933;

}
// back prop for + op
int32_t x1938 = 0;
int32_t x1939 = 0;
int32_t x1940 = 0;
for(int x1941=0; x1941 < 150; x1941++) {
int32_t x1942 = x1939;
float x1943 = x1204[x1942];
int32_t x1944 = x1940;
float x1945 = x1222[x1944];
float x1946 = x1943 + x1945;
x1204[x1942] = x1946;
x1938 += 1;
x1939 += 1;
x1940 += 1;

}
int32_t x1953 = 0;
int32_t x1954 = 0;
int32_t x1955 = 0;
for(int x1956=0; x1956 < 150; x1956++) {
int32_t x1957 = x1954;
float x1958 = x193[x1957];
int32_t x1959 = x1955;
float x1960 = x1222[x1959];
float x1961 = x1958 + x1960;
x193[x1957] = x1961;
x1953 += 1;
x1954 += 1;
x1955 += 1;

}
// back prop for + op
int32_t x1969 = 0;
int32_t x1970 = 0;
int32_t x1971 = 0;
for(int x1972=0; x1972 < 150; x1972++) {
int32_t x1973 = x1970;
float x1974 = x1183[x1973];
int32_t x1975 = x1971;
float x1976 = x1204[x1975];
float x1977 = x1974 + x1976;
x1183[x1973] = x1977;
x1969 += 1;
x1970 += 1;
x1971 += 1;

}
int32_t x1984 = 0;
int32_t x1985 = 0;
int32_t x1986 = 0;
for(int x1987=0; x1987 < 150; x1987++) {
int32_t x1988 = x1985;
float x1989 = x1186[x1988];
int32_t x1990 = x1986;
float x1991 = x1204[x1990];
float x1992 = x1989 + x1991;
x1186[x1988] = x1992;
x1984 += 1;
x1985 += 1;
x1986 += 1;

}
// add_cartesian
int32_t x2000 = 0;
for(int x2001=0; x2001 < 150; x2001++) {
for(int x2002=0; x2002 < 150; x2002++) {
int32_t x2003 = x2000;
int32_t x2004 = x2003 + x2002;
float x2005 = x192[x2004];
float x2006 = x272[x2002];
float x2007 = x1186[x2001];
float x2008 = x2006 * x2007;
float x2009 = x2005 + x2008;
x192[x2004] = x2009;

}
x2000 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x155,150,x1186,1,1,x273,1);
// add_cartesian
int32_t x2018 = 0;
for(int x2019=0; x2019 < 150; x2019++) {
for(int x2020=0; x2020 < 150; x2020++) {
int32_t x2021 = x2018;
int32_t x2022 = x2021 + x2020;
float x2023 = x191[x2022];
float x2024 = x263[x2020];
float x2025 = x1183[x2019];
float x2026 = x2024 * x2025;
float x2027 = x2023 + x2026;
x191[x2022] = x2027;

}
x2018 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x147,150,x1183,1,1,x264,1);
for(int x2035=0; x2035 < 150; x2035++) {
float x2036 = x1167[x2035];
float x2037 = x1168[x2035];
float x2040 = x1180[x2035];
float x2038 = 1.0f - x2037;
float x2039 = x2038 * x2037;
float x2041 = x2039 * x2040;
float x2042 = x2036 + x2041;
x1167[x2035] = x2042;

}
// back prop for + op
int32_t x2047 = 0;
int32_t x2048 = 0;
int32_t x2049 = 0;
for(int x2050=0; x2050 < 150; x2050++) {
int32_t x2051 = x2048;
float x2052 = x1149[x2051];
int32_t x2053 = x2049;
float x2054 = x1167[x2053];
float x2055 = x2052 + x2054;
x1149[x2051] = x2055;
x2047 += 1;
x2048 += 1;
x2049 += 1;

}
int32_t x2062 = 0;
int32_t x2063 = 0;
int32_t x2064 = 0;
for(int x2065=0; x2065 < 150; x2065++) {
int32_t x2066 = x2063;
float x2067 = x190[x2066];
int32_t x2068 = x2064;
float x2069 = x1167[x2068];
float x2070 = x2067 + x2069;
x190[x2066] = x2070;
x2062 += 1;
x2063 += 1;
x2064 += 1;

}
// back prop for + op
int32_t x2078 = 0;
int32_t x2079 = 0;
int32_t x2080 = 0;
for(int x2081=0; x2081 < 150; x2081++) {
int32_t x2082 = x2079;
float x2083 = x1128[x2082];
int32_t x2084 = x2080;
float x2085 = x1149[x2084];
float x2086 = x2083 + x2085;
x1128[x2082] = x2086;
x2078 += 1;
x2079 += 1;
x2080 += 1;

}
int32_t x2093 = 0;
int32_t x2094 = 0;
int32_t x2095 = 0;
for(int x2096=0; x2096 < 150; x2096++) {
int32_t x2097 = x2094;
float x2098 = x1131[x2097];
int32_t x2099 = x2095;
float x2100 = x1149[x2099];
float x2101 = x2098 + x2100;
x1131[x2097] = x2101;
x2093 += 1;
x2094 += 1;
x2095 += 1;

}
// add_cartesian
int32_t x2109 = 0;
for(int x2110=0; x2110 < 150; x2110++) {
for(int x2111=0; x2111 < 150; x2111++) {
int32_t x2112 = x2109;
int32_t x2113 = x2112 + x2111;
float x2114 = x189[x2113];
float x2115 = x272[x2111];
float x2116 = x1131[x2110];
float x2117 = x2115 * x2116;
float x2118 = x2114 + x2117;
x189[x2113] = x2118;

}
x2109 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x138,150,x1131,1,1,x273,1);
// add_cartesian
int32_t x2127 = 0;
for(int x2128=0; x2128 < 150; x2128++) {
for(int x2129=0; x2129 < 150; x2129++) {
int32_t x2130 = x2127;
int32_t x2131 = x2130 + x2129;
float x2132 = x188[x2131];
float x2133 = x263[x2129];
float x2134 = x1128[x2128];
float x2135 = x2133 * x2134;
float x2136 = x2132 + x2135;
x188[x2131] = x2136;

}
x2127 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x130,150,x1128,1,1,x264,1);
for(int x2144=0; x2144 < 150; x2144++) {
float x2145 = x1112[x2144];
float x2146 = x1113[x2144];
float x2149 = x1125[x2144];
float x2147 = 1.0f - x2146;
float x2148 = x2147 * x2146;
float x2150 = x2148 * x2149;
float x2151 = x2145 + x2150;
x1112[x2144] = x2151;

}
// back prop for + op
int32_t x2156 = 0;
int32_t x2157 = 0;
int32_t x2158 = 0;
for(int x2159=0; x2159 < 150; x2159++) {
int32_t x2160 = x2157;
float x2161 = x1094[x2160];
int32_t x2162 = x2158;
float x2163 = x1112[x2162];
float x2164 = x2161 + x2163;
x1094[x2160] = x2164;
x2156 += 1;
x2157 += 1;
x2158 += 1;

}
int32_t x2171 = 0;
int32_t x2172 = 0;
int32_t x2173 = 0;
for(int x2174=0; x2174 < 150; x2174++) {
int32_t x2175 = x2172;
float x2176 = x187[x2175];
int32_t x2177 = x2173;
float x2178 = x1112[x2177];
float x2179 = x2176 + x2178;
x187[x2175] = x2179;
x2171 += 1;
x2172 += 1;
x2173 += 1;

}
// back prop for + op
int32_t x2187 = 0;
int32_t x2188 = 0;
int32_t x2189 = 0;
for(int x2190=0; x2190 < 150; x2190++) {
int32_t x2191 = x2188;
float x2192 = x1073[x2191];
int32_t x2193 = x2189;
float x2194 = x1094[x2193];
float x2195 = x2192 + x2194;
x1073[x2191] = x2195;
x2187 += 1;
x2188 += 1;
x2189 += 1;

}
int32_t x2202 = 0;
int32_t x2203 = 0;
int32_t x2204 = 0;
for(int x2205=0; x2205 < 150; x2205++) {
int32_t x2206 = x2203;
float x2207 = x1076[x2206];
int32_t x2208 = x2204;
float x2209 = x1094[x2208];
float x2210 = x2207 + x2209;
x1076[x2206] = x2210;
x2202 += 1;
x2203 += 1;
x2204 += 1;

}
// add_cartesian
int32_t x2218 = 0;
for(int x2219=0; x2219 < 150; x2219++) {
for(int x2220=0; x2220 < 150; x2220++) {
int32_t x2221 = x2218;
int32_t x2222 = x2221 + x2220;
float x2223 = x186[x2222];
float x2224 = x272[x2220];
float x2225 = x1076[x2219];
float x2226 = x2224 * x2225;
float x2227 = x2223 + x2226;
x186[x2222] = x2227;

}
x2218 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x121,150,x1076,1,1,x273,1);
// add_cartesian
int32_t x2236 = 0;
for(int x2237=0; x2237 < 150; x2237++) {
for(int x2238=0; x2238 < 150; x2238++) {
int32_t x2239 = x2236;
int32_t x2240 = x2239 + x2238;
float x2241 = x185[x2240];
float x2242 = x263[x2238];
float x2243 = x1073[x2237];
float x2244 = x2242 * x2243;
float x2245 = x2241 + x2244;
x185[x2240] = x2245;

}
x2236 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x113,150,x1073,1,1,x264,1);
for(int x2253=0; x2253 < 150; x2253++) {
float x2254 = x1057[x2253];
float x2255 = x1058[x2253];
float x2258 = x1070[x2253];
float x2256 = 1.0f - x2255;
float x2257 = x2256 * x2255;
float x2259 = x2257 * x2258;
float x2260 = x2254 + x2259;
x1057[x2253] = x2260;

}
// back prop for + op
int32_t x2265 = 0;
int32_t x2266 = 0;
int32_t x2267 = 0;
for(int x2268=0; x2268 < 150; x2268++) {
int32_t x2269 = x2266;
float x2270 = x1039[x2269];
int32_t x2271 = x2267;
float x2272 = x1057[x2271];
float x2273 = x2270 + x2272;
x1039[x2269] = x2273;
x2265 += 1;
x2266 += 1;
x2267 += 1;

}
int32_t x2280 = 0;
int32_t x2281 = 0;
int32_t x2282 = 0;
for(int x2283=0; x2283 < 150; x2283++) {
int32_t x2284 = x2281;
float x2285 = x187[x2284];
int32_t x2286 = x2282;
float x2287 = x1057[x2286];
float x2288 = x2285 + x2287;
x187[x2284] = x2288;
x2280 += 1;
x2281 += 1;
x2282 += 1;

}
// back prop for + op
int32_t x2296 = 0;
int32_t x2297 = 0;
int32_t x2298 = 0;
for(int x2299=0; x2299 < 150; x2299++) {
int32_t x2300 = x2297;
float x2301 = x1018[x2300];
int32_t x2302 = x2298;
float x2303 = x1039[x2302];
float x2304 = x2301 + x2303;
x1018[x2300] = x2304;
x2296 += 1;
x2297 += 1;
x2298 += 1;

}
int32_t x2311 = 0;
int32_t x2312 = 0;
int32_t x2313 = 0;
for(int x2314=0; x2314 < 150; x2314++) {
int32_t x2315 = x2312;
float x2316 = x1021[x2315];
int32_t x2317 = x2313;
float x2318 = x1039[x2317];
float x2319 = x2316 + x2318;
x1021[x2315] = x2319;
x2311 += 1;
x2312 += 1;
x2313 += 1;

}
// add_cartesian
int32_t x2327 = 0;
for(int x2328=0; x2328 < 150; x2328++) {
for(int x2329=0; x2329 < 150; x2329++) {
int32_t x2330 = x2327;
int32_t x2331 = x2330 + x2329;
float x2332 = x184[x2331];
float x2333 = x272[x2329];
float x2334 = x1021[x2328];
float x2335 = x2333 * x2334;
float x2336 = x2332 + x2335;
x184[x2331] = x2336;

}
x2327 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x105,150,x1021,1,1,x273,1);
// add_cartesian
int32_t x2345 = 0;
for(int x2346=0; x2346 < 150; x2346++) {
for(int x2347=0; x2347 < 150; x2347++) {
int32_t x2348 = x2345;
int32_t x2349 = x2348 + x2347;
float x2350 = x183[x2349];
float x2351 = x263[x2347];
float x2352 = x1018[x2346];
float x2353 = x2351 * x2352;
float x2354 = x2350 + x2353;
x183[x2349] = x2354;

}
x2345 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x97,150,x1018,1,1,x264,1);
for(int x2362=0; x2362 < 150; x2362++) {
float x2363 = x1002[x2362];
float x2364 = x1003[x2362];
float x2367 = x1015[x2362];
float x2365 = 1.0f - x2364;
float x2366 = x2365 * x2364;
float x2368 = x2366 * x2367;
float x2369 = x2363 + x2368;
x1002[x2362] = x2369;

}
// back prop for + op
int32_t x2374 = 0;
int32_t x2375 = 0;
int32_t x2376 = 0;
for(int x2377=0; x2377 < 150; x2377++) {
int32_t x2378 = x2375;
float x2379 = x984[x2378];
int32_t x2380 = x2376;
float x2381 = x1002[x2380];
float x2382 = x2379 + x2381;
x984[x2378] = x2382;
x2374 += 1;
x2375 += 1;
x2376 += 1;

}
int32_t x2389 = 0;
int32_t x2390 = 0;
int32_t x2391 = 0;
for(int x2392=0; x2392 < 150; x2392++) {
int32_t x2393 = x2390;
float x2394 = x182[x2393];
int32_t x2395 = x2391;
float x2396 = x1002[x2395];
float x2397 = x2394 + x2396;
x182[x2393] = x2397;
x2389 += 1;
x2390 += 1;
x2391 += 1;

}
// back prop for + op
int32_t x2405 = 0;
int32_t x2406 = 0;
int32_t x2407 = 0;
for(int x2408=0; x2408 < 150; x2408++) {
int32_t x2409 = x2406;
float x2410 = x963[x2409];
int32_t x2411 = x2407;
float x2412 = x984[x2411];
float x2413 = x2410 + x2412;
x963[x2409] = x2413;
x2405 += 1;
x2406 += 1;
x2407 += 1;

}
int32_t x2420 = 0;
int32_t x2421 = 0;
int32_t x2422 = 0;
for(int x2423=0; x2423 < 150; x2423++) {
int32_t x2424 = x2421;
float x2425 = x966[x2424];
int32_t x2426 = x2422;
float x2427 = x984[x2426];
float x2428 = x2425 + x2427;
x966[x2424] = x2428;
x2420 += 1;
x2421 += 1;
x2422 += 1;

}
// add_cartesian
int32_t x2436 = 0;
for(int x2437=0; x2437 < 150; x2437++) {
for(int x2438=0; x2438 < 150; x2438++) {
int32_t x2439 = x2436;
int32_t x2440 = x2439 + x2438;
float x2441 = x181[x2440];
float x2442 = x272[x2438];
float x2443 = x966[x2437];
float x2444 = x2442 * x2443;
float x2445 = x2441 + x2444;
x181[x2440] = x2445;

}
x2436 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x88,150,x966,1,1,x273,1);
// add_cartesian
int32_t x2454 = 0;
for(int x2455=0; x2455 < 150; x2455++) {
for(int x2456=0; x2456 < 150; x2456++) {
int32_t x2457 = x2454;
int32_t x2458 = x2457 + x2456;
float x2459 = x180[x2458];
float x2460 = x263[x2456];
float x2461 = x963[x2455];
float x2462 = x2460 * x2461;
float x2463 = x2459 + x2462;
x180[x2458] = x2463;

}
x2454 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x79,150,x963,1,1,x264,1);
}
};
x244(x267,x268,x2474);
};
x244(x258,x259,x2484);
} else {
float** x2511 = (float**)myMalloc(6 * sizeof(float*));;
x2511[0] = x238;
x2511[1] = x239;
x2511[2] = x240;
x2511[3] = x241;
x2511[4] = x242;
x2511[5] = x243;
function<void(float**)> x249 = x246;
function<void(float**)> x2494 = [&](float** x2495) {
float* x2496 = x2495[0];
float* x2497 = x2495[1];
float* x2498 = x2495[2];
float* x2499 = x2495[3];
float* x2500 = x2495[4];
float* x2501 = x2495[5];
float** x2502 = (float**)myMalloc(6 * sizeof(float*));;
x2502[0] = x2496;
x2502[1] = x2497;
x2502[2] = x2498;
x2502[3] = x2499;
x2502[4] = x2500;
x2502[5] = x2501;
x249(x2502);
};
x2494(x2511);
}
};
float* x234 = (float*)myMalloc(1 * sizeof(float));;
float* x235 = (float*)myMalloc(1 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
float* x237 = (float*)myMalloc(1 * sizeof(float));;
float** x2542 = (float**)myMalloc(6 * sizeof(float*));;
x2542[0] = x238;
x2542[1] = x239;
x2542[2] = x240;
x2542[3] = x241;
x2542[4] = x242;
x2542[5] = x243;
function<void(float**)> x2522 = [&](float** x2523) {
float* x2524 = x2523[0];
float* x2525 = x2523[1];
float* x2526 = x2523[2];
float* x2527 = x2523[3];
float* x2528 = x2523[4];
float* x2529 = x2523[5];
// make sure the size of loss is 1
for(int x2531=0; x2531 < 1; x2531++) {
x2525[x2531] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@ce5300c
for(int x2536=0; x2536 < 1; x2536++) {
float x2537 = x2524[x2536];
x237[x2536] = x2537;

}
};
x244(0,x2522,x2542);
float x2551 = x237[0];
float x2552 = x223;
float x2553 = (float)x224;
float x2554 = x2552 * x2553;
int32_t x2555 = x224 + 1;
float x2556 = (float)x2555;
float x2557 = x2554 / x2556;
float x2558 = x2551 / x2556;
float x2559 = x2557 + x2558;
x223 = x2559;
for(int x2561=0; x2561 < 45000; x2561++) {
float x2562 = x174[x2561];
float x2563 = x2562;
float x2564 = x196[x2561];
float x2565 = x2563;
float x2566 = x2565 * x2565;
float x2567 = x2564 + x2566;
x196[x2561] = x2567;
float x2569 = x50[x2561];
float x2571 = x196[x2561];
float x2570 = 0.05f * x2565;
double x2572 = (double)x2571;
double x2573 = x2572 + 9.99999993922529E-9;
double x2574 = sqrt(x2573);
float x2575 = (float)x2574;
float x2576 = x2570 / x2575;
float x2577 = x2569 - x2576;
x50[x2561] = x2577;
x174[x2561] = 0.0f;

}
for(int x2582=0; x2582 < 150; x2582++) {
float x2583 = x175[x2582];
float x2584 = x2583;
float x2585 = x197[x2582];
float x2586 = x2584;
float x2587 = x2586 * x2586;
float x2588 = x2585 + x2587;
x197[x2582] = x2588;
float x2590 = x60[x2582];
float x2592 = x197[x2582];
float x2591 = 0.05f * x2586;
double x2593 = (double)x2592;
double x2594 = x2593 + 9.99999993922529E-9;
double x2595 = sqrt(x2594);
float x2596 = (float)x2595;
float x2597 = x2591 / x2596;
float x2598 = x2590 - x2597;
x60[x2582] = x2598;
x175[x2582] = 0.0f;

}
for(int x2603=0; x2603 < 45000; x2603++) {
float x2604 = x176[x2603];
float x2605 = x2604;
float x2606 = x198[x2603];
float x2607 = x2605;
float x2608 = x2607 * x2607;
float x2609 = x2606 + x2608;
x198[x2603] = x2609;
float x2611 = x61[x2603];
float x2613 = x198[x2603];
float x2612 = 0.05f * x2607;
double x2614 = (double)x2613;
double x2615 = x2614 + 9.99999993922529E-9;
double x2616 = sqrt(x2615);
float x2617 = (float)x2616;
float x2618 = x2612 / x2617;
float x2619 = x2611 - x2618;
x61[x2603] = x2619;
x176[x2603] = 0.0f;

}
for(int x2624=0; x2624 < 150; x2624++) {
float x2625 = x177[x2624];
float x2626 = x2625;
float x2627 = x199[x2624];
float x2628 = x2626;
float x2629 = x2628 * x2628;
float x2630 = x2627 + x2629;
x199[x2624] = x2630;
float x2632 = x69[x2624];
float x2634 = x199[x2624];
float x2633 = 0.05f * x2628;
double x2635 = (double)x2634;
double x2636 = x2635 + 9.99999993922529E-9;
double x2637 = sqrt(x2636);
float x2638 = (float)x2637;
float x2639 = x2633 / x2638;
float x2640 = x2632 - x2639;
x69[x2624] = x2640;
x177[x2624] = 0.0f;

}
for(int x2645=0; x2645 < 45000; x2645++) {
float x2646 = x178[x2645];
float x2647 = x2646;
float x2648 = x200[x2645];
float x2649 = x2647;
float x2650 = x2649 * x2649;
float x2651 = x2648 + x2650;
x200[x2645] = x2651;
float x2653 = x70[x2645];
float x2655 = x200[x2645];
float x2654 = 0.05f * x2649;
double x2656 = (double)x2655;
double x2657 = x2656 + 9.99999993922529E-9;
double x2658 = sqrt(x2657);
float x2659 = (float)x2658;
float x2660 = x2654 / x2659;
float x2661 = x2653 - x2660;
x70[x2645] = x2661;
x178[x2645] = 0.0f;

}
for(int x2666=0; x2666 < 150; x2666++) {
float x2667 = x179[x2666];
float x2668 = x2667;
float x2669 = x201[x2666];
float x2670 = x2668;
float x2671 = x2670 * x2670;
float x2672 = x2669 + x2671;
x201[x2666] = x2672;
float x2674 = x78[x2666];
float x2676 = x201[x2666];
float x2675 = 0.05f * x2670;
double x2677 = (double)x2676;
double x2678 = x2677 + 9.99999993922529E-9;
double x2679 = sqrt(x2678);
float x2680 = (float)x2679;
float x2681 = x2675 / x2680;
float x2682 = x2674 - x2681;
x78[x2666] = x2682;
x179[x2666] = 0.0f;

}
for(int x2687=0; x2687 < 22500; x2687++) {
float x2688 = x180[x2687];
float x2689 = x2688;
float x2690 = x202[x2687];
float x2691 = x2689;
float x2692 = x2691 * x2691;
float x2693 = x2690 + x2692;
x202[x2687] = x2693;
float x2695 = x79[x2687];
float x2697 = x202[x2687];
float x2696 = 0.05f * x2691;
double x2698 = (double)x2697;
double x2699 = x2698 + 9.99999993922529E-9;
double x2700 = sqrt(x2699);
float x2701 = (float)x2700;
float x2702 = x2696 / x2701;
float x2703 = x2695 - x2702;
x79[x2687] = x2703;
x180[x2687] = 0.0f;

}
for(int x2708=0; x2708 < 22500; x2708++) {
float x2709 = x181[x2708];
float x2710 = x2709;
float x2711 = x203[x2708];
float x2712 = x2710;
float x2713 = x2712 * x2712;
float x2714 = x2711 + x2713;
x203[x2708] = x2714;
float x2716 = x88[x2708];
float x2718 = x203[x2708];
float x2717 = 0.05f * x2712;
double x2719 = (double)x2718;
double x2720 = x2719 + 9.99999993922529E-9;
double x2721 = sqrt(x2720);
float x2722 = (float)x2721;
float x2723 = x2717 / x2722;
float x2724 = x2716 - x2723;
x88[x2708] = x2724;
x181[x2708] = 0.0f;

}
for(int x2729=0; x2729 < 150; x2729++) {
float x2730 = x182[x2729];
float x2731 = x2730;
float x2732 = x204[x2729];
float x2733 = x2731;
float x2734 = x2733 * x2733;
float x2735 = x2732 + x2734;
x204[x2729] = x2735;
float x2737 = x96[x2729];
float x2739 = x204[x2729];
float x2738 = 0.05f * x2733;
double x2740 = (double)x2739;
double x2741 = x2740 + 9.99999993922529E-9;
double x2742 = sqrt(x2741);
float x2743 = (float)x2742;
float x2744 = x2738 / x2743;
float x2745 = x2737 - x2744;
x96[x2729] = x2745;
x182[x2729] = 0.0f;

}
for(int x2750=0; x2750 < 22500; x2750++) {
float x2751 = x183[x2750];
float x2752 = x2751;
float x2753 = x205[x2750];
float x2754 = x2752;
float x2755 = x2754 * x2754;
float x2756 = x2753 + x2755;
x205[x2750] = x2756;
float x2758 = x97[x2750];
float x2760 = x205[x2750];
float x2759 = 0.05f * x2754;
double x2761 = (double)x2760;
double x2762 = x2761 + 9.99999993922529E-9;
double x2763 = sqrt(x2762);
float x2764 = (float)x2763;
float x2765 = x2759 / x2764;
float x2766 = x2758 - x2765;
x97[x2750] = x2766;
x183[x2750] = 0.0f;

}
for(int x2771=0; x2771 < 22500; x2771++) {
float x2772 = x184[x2771];
float x2773 = x2772;
float x2774 = x206[x2771];
float x2775 = x2773;
float x2776 = x2775 * x2775;
float x2777 = x2774 + x2776;
x206[x2771] = x2777;
float x2779 = x105[x2771];
float x2781 = x206[x2771];
float x2780 = 0.05f * x2775;
double x2782 = (double)x2781;
double x2783 = x2782 + 9.99999993922529E-9;
double x2784 = sqrt(x2783);
float x2785 = (float)x2784;
float x2786 = x2780 / x2785;
float x2787 = x2779 - x2786;
x105[x2771] = x2787;
x184[x2771] = 0.0f;

}
for(int x2792=0; x2792 < 22500; x2792++) {
float x2793 = x185[x2792];
float x2794 = x2793;
float x2795 = x207[x2792];
float x2796 = x2794;
float x2797 = x2796 * x2796;
float x2798 = x2795 + x2797;
x207[x2792] = x2798;
float x2800 = x113[x2792];
float x2802 = x207[x2792];
float x2801 = 0.05f * x2796;
double x2803 = (double)x2802;
double x2804 = x2803 + 9.99999993922529E-9;
double x2805 = sqrt(x2804);
float x2806 = (float)x2805;
float x2807 = x2801 / x2806;
float x2808 = x2800 - x2807;
x113[x2792] = x2808;
x185[x2792] = 0.0f;

}
for(int x2813=0; x2813 < 22500; x2813++) {
float x2814 = x186[x2813];
float x2815 = x2814;
float x2816 = x208[x2813];
float x2817 = x2815;
float x2818 = x2817 * x2817;
float x2819 = x2816 + x2818;
x208[x2813] = x2819;
float x2821 = x121[x2813];
float x2823 = x208[x2813];
float x2822 = 0.05f * x2817;
double x2824 = (double)x2823;
double x2825 = x2824 + 9.99999993922529E-9;
double x2826 = sqrt(x2825);
float x2827 = (float)x2826;
float x2828 = x2822 / x2827;
float x2829 = x2821 - x2828;
x121[x2813] = x2829;
x186[x2813] = 0.0f;

}
for(int x2834=0; x2834 < 150; x2834++) {
float x2835 = x187[x2834];
float x2836 = x2835;
float x2837 = x209[x2834];
float x2838 = x2836;
float x2839 = x2838 * x2838;
float x2840 = x2837 + x2839;
x209[x2834] = x2840;
float x2842 = x129[x2834];
float x2844 = x209[x2834];
float x2843 = 0.05f * x2838;
double x2845 = (double)x2844;
double x2846 = x2845 + 9.99999993922529E-9;
double x2847 = sqrt(x2846);
float x2848 = (float)x2847;
float x2849 = x2843 / x2848;
float x2850 = x2842 - x2849;
x129[x2834] = x2850;
x187[x2834] = 0.0f;

}
for(int x2855=0; x2855 < 22500; x2855++) {
float x2856 = x188[x2855];
float x2857 = x2856;
float x2858 = x210[x2855];
float x2859 = x2857;
float x2860 = x2859 * x2859;
float x2861 = x2858 + x2860;
x210[x2855] = x2861;
float x2863 = x130[x2855];
float x2865 = x210[x2855];
float x2864 = 0.05f * x2859;
double x2866 = (double)x2865;
double x2867 = x2866 + 9.99999993922529E-9;
double x2868 = sqrt(x2867);
float x2869 = (float)x2868;
float x2870 = x2864 / x2869;
float x2871 = x2863 - x2870;
x130[x2855] = x2871;
x188[x2855] = 0.0f;

}
for(int x2876=0; x2876 < 22500; x2876++) {
float x2877 = x189[x2876];
float x2878 = x2877;
float x2879 = x211[x2876];
float x2880 = x2878;
float x2881 = x2880 * x2880;
float x2882 = x2879 + x2881;
x211[x2876] = x2882;
float x2884 = x138[x2876];
float x2886 = x211[x2876];
float x2885 = 0.05f * x2880;
double x2887 = (double)x2886;
double x2888 = x2887 + 9.99999993922529E-9;
double x2889 = sqrt(x2888);
float x2890 = (float)x2889;
float x2891 = x2885 / x2890;
float x2892 = x2884 - x2891;
x138[x2876] = x2892;
x189[x2876] = 0.0f;

}
for(int x2897=0; x2897 < 150; x2897++) {
float x2898 = x190[x2897];
float x2899 = x2898;
float x2900 = x212[x2897];
float x2901 = x2899;
float x2902 = x2901 * x2901;
float x2903 = x2900 + x2902;
x212[x2897] = x2903;
float x2905 = x146[x2897];
float x2907 = x212[x2897];
float x2906 = 0.05f * x2901;
double x2908 = (double)x2907;
double x2909 = x2908 + 9.99999993922529E-9;
double x2910 = sqrt(x2909);
float x2911 = (float)x2910;
float x2912 = x2906 / x2911;
float x2913 = x2905 - x2912;
x146[x2897] = x2913;
x190[x2897] = 0.0f;

}
for(int x2918=0; x2918 < 22500; x2918++) {
float x2919 = x191[x2918];
float x2920 = x2919;
float x2921 = x213[x2918];
float x2922 = x2920;
float x2923 = x2922 * x2922;
float x2924 = x2921 + x2923;
x213[x2918] = x2924;
float x2926 = x147[x2918];
float x2928 = x213[x2918];
float x2927 = 0.05f * x2922;
double x2929 = (double)x2928;
double x2930 = x2929 + 9.99999993922529E-9;
double x2931 = sqrt(x2930);
float x2932 = (float)x2931;
float x2933 = x2927 / x2932;
float x2934 = x2926 - x2933;
x147[x2918] = x2934;
x191[x2918] = 0.0f;

}
for(int x2939=0; x2939 < 22500; x2939++) {
float x2940 = x192[x2939];
float x2941 = x2940;
float x2942 = x214[x2939];
float x2943 = x2941;
float x2944 = x2943 * x2943;
float x2945 = x2942 + x2944;
x214[x2939] = x2945;
float x2947 = x155[x2939];
float x2949 = x214[x2939];
float x2948 = 0.05f * x2943;
double x2950 = (double)x2949;
double x2951 = x2950 + 9.99999993922529E-9;
double x2952 = sqrt(x2951);
float x2953 = (float)x2952;
float x2954 = x2948 / x2953;
float x2955 = x2947 - x2954;
x155[x2939] = x2955;
x192[x2939] = 0.0f;

}
for(int x2960=0; x2960 < 150; x2960++) {
float x2961 = x193[x2960];
float x2962 = x2961;
float x2963 = x215[x2960];
float x2964 = x2962;
float x2965 = x2964 * x2964;
float x2966 = x2963 + x2965;
x215[x2960] = x2966;
float x2968 = x163[x2960];
float x2970 = x215[x2960];
float x2969 = 0.05f * x2964;
double x2971 = (double)x2970;
double x2972 = x2971 + 9.99999993922529E-9;
double x2973 = sqrt(x2972);
float x2974 = (float)x2973;
float x2975 = x2969 / x2974;
float x2976 = x2968 - x2975;
x163[x2960] = x2976;
x193[x2960] = 0.0f;

}
for(int x2981=0; x2981 < 750; x2981++) {
float x2982 = x194[x2981];
float x2983 = x2982;
float x2984 = x216[x2981];
float x2985 = x2983;
float x2986 = x2985 * x2985;
float x2987 = x2984 + x2986;
x216[x2981] = x2987;
float x2989 = x164[x2981];
float x2991 = x216[x2981];
float x2990 = 0.05f * x2985;
double x2992 = (double)x2991;
double x2993 = x2992 + 9.99999993922529E-9;
double x2994 = sqrt(x2993);
float x2995 = (float)x2994;
float x2996 = x2990 / x2995;
float x2997 = x2989 - x2996;
x164[x2981] = x2997;
x194[x2981] = 0.0f;

}
for(int x3002=0; x3002 < 5; x3002++) {
float x3003 = x195[x3002];
float x3004 = x3003;
float x3005 = x217[x3002];
float x3006 = x3004;
float x3007 = x3006 * x3006;
float x3008 = x3005 + x3007;
x217[x3002] = x3008;
float x3010 = x173[x3002];
float x3012 = x217[x3002];
float x3011 = 0.05f * x3006;
double x3013 = (double)x3012;
double x3014 = x3013 + 9.99999993922529E-9;
double x3015 = sqrt(x3014);
float x3016 = (float)x3015;
float x3017 = x3011 / x3016;
float x3018 = x3010 - x3017;
x173[x3002] = x3018;
x195[x3002] = 0.0f;

}
int64_t x3023 = (long)mallocAddr;
int64_t x3024 = x3023 - x219;
memset((void*)x219, 0, x3024);
mallocAddr = (void*)x219;

}
float x3029 = x223;
double x3030 = (double)x3029;
x218[x222] = x3030;
double x3032 = ((double)clock() / CLOCKS_PER_SEC);
double x3033 = x3032 - x220;
printf("epoc %d, average_loss %f, time %lf\n",x222,x3029,x3033);

}
double x3037 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x3041 = (long)fopen(x0, "w");
fprintf((FILE *)x3041, "unit: %s\n", "1 epoch");
for(int x3043=0; x3043 < 6; x3043++) {
double x3044 = x218[x3043];
fprintf((FILE *)x3041, "%lf\n", x3044);

}
double x3038 = x220 - x2;
double x3039 = x3037 - x220;
double x3040 = x3039 / 6.0;
fprintf((FILE *)x3041, "run time: %lf %lf\n", x3038, x3040);
fclose((FILE*)x3041);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

