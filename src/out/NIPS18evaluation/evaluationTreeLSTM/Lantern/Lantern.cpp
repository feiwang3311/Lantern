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
bool x1087 = true || true;
bool x1088 = x1087 || true;
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
float** x1924 = (float**)myMalloc(6 * sizeof(float*));;
x1924[0] = x238;
x1924[1] = x239;
x1924[2] = x240;
x1924[3] = x241;
x1924[4] = x242;
x1924[5] = x243;
function<void(float**)> x249 = x246;
function<void(float**)> x521 = [&](float** x522) {
float* x523 = x522[0];
float* x524 = x522[1];
float* x525 = x522[2];
float* x526 = x522[3];
float* x527 = x522[4];
float* x528 = x522[5];
float** x529 = (float**)myMalloc(6 * sizeof(float*));;
x529[0] = x523;
x529[1] = x524;
x529[2] = x525;
x529[3] = x526;
x529[4] = x527;
x529[5] = x528;
x249(x529);
};
function<void(float**)> x513 = [&](float** x514) {
float* x515 = x514[0];
float* x516 = x514[1];
float* x517 = x514[2];
float* x518 = x514[3];
float* x519 = x514[4];
float* x520 = x514[5];
float** x538 = (float**)myMalloc(6 * sizeof(float*));;
x538[0] = x515;
x538[1] = x516;
x538[2] = x517;
x538[3] = x518;
x538[4] = x519;
x538[5] = x520;
x521(x538);
};
function<void(float**)> x1236 = [&](float** x1237) {
float* x1238 = x1237[0];
float* x1239 = x1237[1];
float* x1240 = x1237[2];
float* x1241 = x1237[3];
float* x1242 = x1237[4];
float* x1243 = x1237[5];
float** x1244 = (float**)myMalloc(6 * sizeof(float*));;
x1244[0] = x1238;
x1244[1] = x1239;
x1244[2] = x1240;
x1244[3] = x1241;
x1244[4] = x1242;
x1244[5] = x1243;
x249(x1244);
};
function<void(float**)> x1228 = [&](float** x1229) {
float* x1230 = x1229[0];
float* x1231 = x1229[1];
float* x1232 = x1229[2];
float* x1233 = x1229[3];
float* x1234 = x1229[4];
float* x1235 = x1229[5];
float** x1253 = (float**)myMalloc(6 * sizeof(float*));;
x1253[0] = x1230;
x1253[1] = x1231;
x1253[2] = x1232;
x1253[3] = x1233;
x1253[4] = x1234;
x1253[5] = x1235;
x1236(x1253);
};
function<void(float**)> x259 = [&](float** x260) {
float* x261 = x260[0];
float* x262 = x260[1];
float* x263 = x260[2];
float* x264 = x260[3];
float* x265 = x260[4];
float* x266 = x260[5];
int32_t x267 = x233[x248];
float** x1914 = (float**)myMalloc(6 * sizeof(float*));;
x1914[0] = x238;
x1914[1] = x239;
x1914[2] = x240;
x1914[3] = x241;
x1914[4] = x242;
x1914[5] = x243;
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
for(int x290=0; x290 < 150; x290++) {
float x291 = x285[x290];
float x292 = x60[x290];
float x293 = x291 + x292;
x288[x290] = x293;

}
float* x297 = (float*)myMalloc(150 * sizeof(float));;
float* x298 = (float*)myMalloc(150 * sizeof(float));;
for(int x299=0; x299 < 150; x299++) {
float x300 = x288[x299];
float x301 = -1.0f * x300;
double x302 = (double)x301;
double x303 = exp(x302);
float x304 = (float)x303;
float x305 = x304 + 1.0f;
float x306 = 1.0f / x305;
x298[x299] = x306;

}
float* x310 = (float*)myMalloc(150 * sizeof(float));;
float* x311 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x61,300,x283,1,0,x311,1);
float* x313 = (float*)myMalloc(150 * sizeof(float));;
float* x314 = (float*)myMalloc(150 * sizeof(float));;
for(int x315=0; x315 < 150; x315++) {
float x316 = x311[x315];
float x317 = x69[x315];
float x318 = x316 + x317;
x314[x315] = x318;

}
float* x322 = (float*)myMalloc(150 * sizeof(float));;
float* x323 = (float*)myMalloc(150 * sizeof(float));;
for(int x324=0; x324 < 150; x324++) {
float x325 = x314[x324];
float x326 = -1.0f * x325;
double x327 = (double)x326;
double x328 = exp(x327);
float x329 = (float)x328;
float x330 = x329 + 1.0f;
float x331 = 1.0f / x330;
x323[x324] = x331;

}
float* x335 = (float*)myMalloc(150 * sizeof(float));;
float* x336 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x70,300,x283,1,0,x336,1);
float* x338 = (float*)myMalloc(150 * sizeof(float));;
float* x339 = (float*)myMalloc(150 * sizeof(float));;
for(int x340=0; x340 < 150; x340++) {
float x341 = x336[x340];
float x342 = x78[x340];
float x343 = x341 + x342;
x339[x340] = x343;

}
float* x347 = (float*)myMalloc(150 * sizeof(float));;
float* x348 = (float*)myMalloc(150 * sizeof(float));;
for(int x349=0; x349 < 150; x349++) {
float x350 = x339[x349];
double x351 = (double)x350;
double x352 = tanh(x351);
float x353 = (float)x352;
x348[x349] = x353;

}
float* x357 = (float*)myMalloc(150 * sizeof(float));;
float* x358 = (float*)myMalloc(150 * sizeof(float));;
for(int x359=0; x359 < 150; x359++) {
float x360 = x298[x359];
float x361 = x348[x359];
float x362 = x360 * x361;
x358[x359] = x362;

}
float* x366 = (float*)myMalloc(150 * sizeof(float));;
float* x367 = (float*)myMalloc(150 * sizeof(float));;
for(int x368=0; x368 < 150; x368++) {
float x369 = x358[x368];
double x370 = (double)x369;
double x371 = tanh(x370);
float x372 = (float)x371;
x367[x368] = x372;

}
float* x376 = (float*)myMalloc(150 * sizeof(float));;
float* x377 = (float*)myMalloc(150 * sizeof(float));;
for(int x378=0; x378 < 150; x378++) {
float x379 = x323[x378];
float x380 = x367[x378];
float x381 = x379 * x380;
x377[x378] = x381;

}
float* x385 = (float*)myMalloc(150 * sizeof(float));;
float* x386 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x164,150,x377,1,0,x386,1);
float* x388 = (float*)myMalloc(5 * sizeof(float));;
float* x389 = (float*)myMalloc(5 * sizeof(float));;
for(int x391=0; x391 < 5; x391++) {
float x392 = x386[x391];
float x393 = x173[x391];
float x394 = x392 + x393;
x389[x391] = x394;

}
float* x398 = (float*)myMalloc(5 * sizeof(float));;
int32_t x399 = 0;
int32_t x400 = 1;
x400 *= 1;
x400 *= 5;
int32_t x403 = x399;
bool x404 = x403 >= 2;
if (x404) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x410 = x403 == 0;
if (x410) {
int32_t x411 = x400;
bool x412 = x411 == 5;
if (x412) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x419 = (float*)myMalloc(1 * sizeof(float));;
int32_t x420 = 0;
for(int x422=0; x422 < 1; x422++) {
float x423 = -3.4028235E38f;
for(int x424=0; x424 < 5; x424++) {
int32_t x425 = x420;
float x426 = x389[x425];
float x427 = x423;
bool x428 = x426 > x427;
if (x428) {
float x429 = x389[x425];
x423 = x429;
} else {
}
x420 += 1;

}
float x436 = x423;
x419[x422] = x436;

}
float* x440 = (float*)myMalloc(5 * sizeof(float));;
int32_t x441 = 0;
for(int x442=0; x442 < 1; x442++) {
for(int x443=0; x443 < 5; x443++) {
int32_t x444 = x441;
float x445 = x389[x444];
float x446 = x419[x442];
float x447 = x445 - x446;
double x448 = (double)x447;
double x449 = exp(x448);
float x450 = (float)x449;
x440[x444] = x450;
x441 += 1;

}

}
float* x457 = (float*)myMalloc(1 * sizeof(float));;
for(int x458=0; x458 < 1; x458++) {
int32_t x459 = x458;
int32_t x460 = x458 * 5;
int32_t x461 = x460;
for(int x462=0; x462 < 5; x462++) {
for(int x463=0; x463 < 1; x463++) {
int32_t x464 = x459;
int32_t x465 = x464 + x463;
float x466 = x457[x465];
int32_t x467 = x461;
int32_t x468 = x467 + x463;
float x469 = x440[x468];
float x470 = x466 + x469;
x457[x465] = x470;

}
x461 += 1;

}

}
x441 = 0;
for(int x480=0; x480 < 1; x480++) {
float x481 = x419[x480];
float x482 = x457[x480];
double x483 = (double)x482;
double x484 = log(x483);
float x485 = (float)x484;
float x486 = x481 + x485;
for(int x487=0; x487 < 5; x487++) {
int32_t x488 = x441;
float x489 = x389[x488];
float x490 = x489 - x486;
x440[x488] = x490;
x441 += 1;

}

}
float* x497 = (float*)myMalloc(5 * sizeof(float));;
int* x498 = x227+x248;
// nllLoss forward in CPU
float* x500 = (float*)myMalloc(1 * sizeof(float));;
int32_t x501 = 0;
for(int x502=0; x502 < 1; x502++) {
int32_t x503 = x501;
int32_t x504 = x498[x502];
int32_t x505 = x503 + x504;
float x506 = x440[x505];
float x507 = -1.0f * x506;
x500[x502] = x507;
x501 += 5;

}
float* x512 = (float*)myMalloc(1 * sizeof(float));;
float** x547 = (float**)myMalloc(6 * sizeof(float*));;
x547[0] = x500;
x547[1] = x512;
x547[2] = x377;
x547[3] = x385;
x547[4] = x358;
x547[5] = x366;
x513(x547);
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x557 = 0;
for(int x558=0; x558 < 1; x558++) {
int32_t x559 = x557;
int32_t x560 = x498[x558];
int32_t x561 = x559 + x560;
float x562 = x497[x561];
float x563 = x512[x558];
float x564 = -1.0f * x563;
float x565 = x562 + x564;
x497[x561] = x565;
x557 += 5;

}
float* x570 = (float*)myMalloc(1 * sizeof(float));;
for(int x571=0; x571 < 1; x571++) {
int32_t x572 = x571;
int32_t x573 = x571 * 5;
int32_t x574 = x573;
for(int x575=0; x575 < 5; x575++) {
for(int x576=0; x576 < 1; x576++) {
int32_t x577 = x572;
int32_t x578 = x577 + x576;
float x579 = x570[x578];
int32_t x580 = x574;
int32_t x581 = x580 + x576;
float x582 = x497[x581];
float x583 = x579 + x582;
x570[x578] = x583;

}
x574 += 1;

}

}
int32_t x592 = 0;
for(int x593=0; x593 < 1; x593++) {
for(int x594=0; x594 < 5; x594++) {
int32_t x595 = x592;
float x596 = x398[x595];
float x597 = x497[x595];
float x598 = x440[x595];
float x602 = x570[x593];
double x599 = (double)x598;
double x600 = exp(x599);
float x601 = (float)x600;
float x603 = x601 * x602;
float x604 = x597 - x603;
float x605 = x596 + x604;
x398[x595] = x605;
x592 += 1;

}

}
// back prop for + op
for(int x613=0; x613 < 5; x613++) {
float x614 = x388[x613];
float x615 = x386[x613];
float x616 = x173[x613];
float x617 = x398[x613];
float x618 = x614 + x617;
x388[x613] = x618;
float x620 = x195[x613];
float x621 = x386[x613];
float x622 = x173[x613];
float x623 = x398[x613];
float x624 = x620 + x623;
x195[x613] = x624;

}
// add_cartesian
int32_t x629 = 0;
for(int x630=0; x630 < 5; x630++) {
for(int x631=0; x631 < 150; x631++) {
int32_t x632 = x629;
int32_t x633 = x632 + x631;
float x634 = x194[x633];
float x635 = x377[x631];
float x636 = x388[x630];
float x637 = x635 * x636;
float x638 = x634 + x637;
x194[x633] = x638;

}
x629 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x164,150,x388,1,1,x385,1);
// backprop for * op
for(int x647=0; x647 < 150; x647++) {
float x648 = x335[x647];
float x649 = x323[x647];
float x650 = x367[x647];
float x651 = x385[x647];
float x652 = x651 * x650;
float x653 = x648 + x652;
x335[x647] = x653;
float x655 = x376[x647];
float x656 = x323[x647];
float x657 = x367[x647];
float x658 = x385[x647];
float x659 = x658 * x656;
float x660 = x655 + x659;
x376[x647] = x660;

}
for(int x664=0; x664 < 150; x664++) {
float x665 = x366[x664];
float x666 = x367[x664];
float x669 = x376[x664];
float x667 = x666 * x666;
float x668 = 1.0f - x667;
float x670 = x668 * x669;
float x671 = x665 + x670;
x366[x664] = x671;

}
// backprop for * op
for(int x676=0; x676 < 150; x676++) {
float x677 = x310[x676];
float x678 = x298[x676];
float x679 = x348[x676];
float x680 = x366[x676];
float x681 = x680 * x679;
float x682 = x677 + x681;
x310[x676] = x682;
float x684 = x357[x676];
float x685 = x298[x676];
float x686 = x348[x676];
float x687 = x366[x676];
float x688 = x687 * x685;
float x689 = x684 + x688;
x357[x676] = x689;

}
for(int x693=0; x693 < 150; x693++) {
float x694 = x347[x693];
float x695 = x348[x693];
float x698 = x357[x693];
float x696 = x695 * x695;
float x697 = 1.0f - x696;
float x699 = x697 * x698;
float x700 = x694 + x699;
x347[x693] = x700;

}
// back prop for + op
for(int x705=0; x705 < 150; x705++) {
float x706 = x338[x705];
float x707 = x336[x705];
float x708 = x78[x705];
float x709 = x347[x705];
float x710 = x706 + x709;
x338[x705] = x710;
float x712 = x179[x705];
float x713 = x336[x705];
float x714 = x78[x705];
float x715 = x347[x705];
float x716 = x712 + x715;
x179[x705] = x716;

}
// add_cartesian
int32_t x721 = 0;
for(int x722=0; x722 < 150; x722++) {
for(int x723=0; x723 < 300; x723++) {
int32_t x724 = x721;
int32_t x725 = x724 + x723;
float x726 = x178[x725];
float x727 = x283[x723];
float x728 = x338[x722];
float x729 = x727 * x728;
float x730 = x726 + x729;
x178[x725] = x730;

}
x721 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x70,300,x338,1,1,x284,1);
for(int x738=0; x738 < 150; x738++) {
float x739 = x322[x738];
float x740 = x323[x738];
float x743 = x335[x738];
float x741 = 1.0f - x740;
float x742 = x741 * x740;
float x744 = x742 * x743;
float x745 = x739 + x744;
x322[x738] = x745;

}
// back prop for + op
for(int x750=0; x750 < 150; x750++) {
float x751 = x313[x750];
float x752 = x311[x750];
float x753 = x69[x750];
float x754 = x322[x750];
float x755 = x751 + x754;
x313[x750] = x755;
float x757 = x177[x750];
float x758 = x311[x750];
float x759 = x69[x750];
float x760 = x322[x750];
float x761 = x757 + x760;
x177[x750] = x761;

}
// add_cartesian
int32_t x766 = 0;
for(int x767=0; x767 < 150; x767++) {
for(int x768=0; x768 < 300; x768++) {
int32_t x769 = x766;
int32_t x770 = x769 + x768;
float x771 = x176[x770];
float x772 = x283[x768];
float x773 = x313[x767];
float x774 = x772 * x773;
float x775 = x771 + x774;
x176[x770] = x775;

}
x766 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x61,300,x313,1,1,x284,1);
for(int x783=0; x783 < 150; x783++) {
float x784 = x297[x783];
float x785 = x298[x783];
float x788 = x310[x783];
float x786 = 1.0f - x785;
float x787 = x786 * x785;
float x789 = x787 * x788;
float x790 = x784 + x789;
x297[x783] = x790;

}
// back prop for + op
for(int x795=0; x795 < 150; x795++) {
float x796 = x287[x795];
float x797 = x285[x795];
float x798 = x60[x795];
float x799 = x297[x795];
float x800 = x796 + x799;
x287[x795] = x800;
float x802 = x175[x795];
float x803 = x285[x795];
float x804 = x60[x795];
float x805 = x297[x795];
float x806 = x802 + x805;
x175[x795] = x806;

}
// add_cartesian
int32_t x811 = 0;
for(int x812=0; x812 < 150; x812++) {
for(int x813=0; x813 < 300; x813++) {
int32_t x814 = x811;
int32_t x815 = x814 + x813;
float x816 = x174[x815];
float x817 = x283[x813];
float x818 = x287[x812];
float x819 = x817 * x818;
float x820 = x816 + x819;
x174[x815] = x820;

}
x811 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x50,300,x287,1,1,x284,1);
} else {
float* x829 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x79,150,x263,1,0,x829,1);
float* x831 = (float*)myMalloc(150 * sizeof(float));;
float* x832 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x88,150,x272,1,0,x832,1);
float* x834 = (float*)myMalloc(150 * sizeof(float));;
float* x835 = (float*)myMalloc(150 * sizeof(float));;
for(int x836=0; x836 < 150; x836++) {
float x837 = x829[x836];
float x838 = x832[x836];
float x839 = x837 + x838;
x835[x836] = x839;

}
float* x843 = (float*)myMalloc(150 * sizeof(float));;
float* x844 = (float*)myMalloc(150 * sizeof(float));;
for(int x845=0; x845 < 150; x845++) {
float x846 = x835[x845];
float x847 = x96[x845];
float x848 = x846 + x847;
x844[x845] = x848;

}
float* x852 = (float*)myMalloc(150 * sizeof(float));;
float* x853 = (float*)myMalloc(150 * sizeof(float));;
for(int x854=0; x854 < 150; x854++) {
float x855 = x844[x854];
float x856 = -1.0f * x855;
double x857 = (double)x856;
double x858 = exp(x857);
float x859 = (float)x858;
float x860 = x859 + 1.0f;
float x861 = 1.0f / x860;
x853[x854] = x861;

}
float* x865 = (float*)myMalloc(150 * sizeof(float));;
float* x866 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x97,150,x263,1,0,x866,1);
float* x868 = (float*)myMalloc(150 * sizeof(float));;
float* x869 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x105,150,x272,1,0,x869,1);
float* x871 = (float*)myMalloc(150 * sizeof(float));;
float* x872 = (float*)myMalloc(150 * sizeof(float));;
for(int x873=0; x873 < 150; x873++) {
float x874 = x866[x873];
float x875 = x869[x873];
float x876 = x874 + x875;
x872[x873] = x876;

}
float* x880 = (float*)myMalloc(150 * sizeof(float));;
float* x881 = (float*)myMalloc(150 * sizeof(float));;
for(int x882=0; x882 < 150; x882++) {
float x883 = x872[x882];
float x884 = x129[x882];
float x885 = x883 + x884;
x881[x882] = x885;

}
float* x889 = (float*)myMalloc(150 * sizeof(float));;
float* x890 = (float*)myMalloc(150 * sizeof(float));;
for(int x891=0; x891 < 150; x891++) {
float x892 = x881[x891];
float x893 = -1.0f * x892;
double x894 = (double)x893;
double x895 = exp(x894);
float x896 = (float)x895;
float x897 = x896 + 1.0f;
float x898 = 1.0f / x897;
x890[x891] = x898;

}
float* x902 = (float*)myMalloc(150 * sizeof(float));;
float* x903 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x113,150,x263,1,0,x903,1);
float* x905 = (float*)myMalloc(150 * sizeof(float));;
float* x906 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x121,150,x272,1,0,x906,1);
float* x908 = (float*)myMalloc(150 * sizeof(float));;
float* x909 = (float*)myMalloc(150 * sizeof(float));;
for(int x910=0; x910 < 150; x910++) {
float x911 = x903[x910];
float x912 = x906[x910];
float x913 = x911 + x912;
x909[x910] = x913;

}
float* x917 = (float*)myMalloc(150 * sizeof(float));;
float* x918 = (float*)myMalloc(150 * sizeof(float));;
for(int x919=0; x919 < 150; x919++) {
float x920 = x909[x919];
float x921 = x129[x919];
float x922 = x920 + x921;
x918[x919] = x922;

}
float* x926 = (float*)myMalloc(150 * sizeof(float));;
float* x927 = (float*)myMalloc(150 * sizeof(float));;
for(int x928=0; x928 < 150; x928++) {
float x929 = x918[x928];
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
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x130,150,x263,1,0,x940,1);
float* x942 = (float*)myMalloc(150 * sizeof(float));;
float* x943 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x138,150,x272,1,0,x943,1);
float* x945 = (float*)myMalloc(150 * sizeof(float));;
float* x946 = (float*)myMalloc(150 * sizeof(float));;
for(int x947=0; x947 < 150; x947++) {
float x948 = x940[x947];
float x949 = x943[x947];
float x950 = x948 + x949;
x946[x947] = x950;

}
float* x954 = (float*)myMalloc(150 * sizeof(float));;
float* x955 = (float*)myMalloc(150 * sizeof(float));;
for(int x956=0; x956 < 150; x956++) {
float x957 = x946[x956];
float x958 = x146[x956];
float x959 = x957 + x958;
x955[x956] = x959;

}
float* x963 = (float*)myMalloc(150 * sizeof(float));;
float* x964 = (float*)myMalloc(150 * sizeof(float));;
for(int x965=0; x965 < 150; x965++) {
float x966 = x955[x965];
float x967 = -1.0f * x966;
double x968 = (double)x967;
double x969 = exp(x968);
float x970 = (float)x969;
float x971 = x970 + 1.0f;
float x972 = 1.0f / x971;
x964[x965] = x972;

}
float* x976 = (float*)myMalloc(150 * sizeof(float));;
float* x977 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x147,150,x263,1,0,x977,1);
float* x979 = (float*)myMalloc(150 * sizeof(float));;
float* x980 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x155,150,x272,1,0,x980,1);
float* x982 = (float*)myMalloc(150 * sizeof(float));;
float* x983 = (float*)myMalloc(150 * sizeof(float));;
for(int x984=0; x984 < 150; x984++) {
float x985 = x977[x984];
float x986 = x980[x984];
float x987 = x985 + x986;
x983[x984] = x987;

}
float* x991 = (float*)myMalloc(150 * sizeof(float));;
float* x992 = (float*)myMalloc(150 * sizeof(float));;
for(int x993=0; x993 < 150; x993++) {
float x994 = x983[x993];
float x995 = x163[x993];
float x996 = x994 + x995;
x992[x993] = x996;

}
float* x1000 = (float*)myMalloc(150 * sizeof(float));;
float* x1001 = (float*)myMalloc(150 * sizeof(float));;
for(int x1002=0; x1002 < 150; x1002++) {
float x1003 = x992[x1002];
double x1004 = (double)x1003;
double x1005 = tanh(x1004);
float x1006 = (float)x1005;
x1001[x1002] = x1006;

}
float* x1010 = (float*)myMalloc(150 * sizeof(float));;
float* x1011 = (float*)myMalloc(150 * sizeof(float));;
for(int x1012=0; x1012 < 150; x1012++) {
float x1013 = x853[x1012];
float x1014 = x1001[x1012];
float x1015 = x1013 * x1014;
x1011[x1012] = x1015;

}
float* x1019 = (float*)myMalloc(150 * sizeof(float));;
float* x1020 = (float*)myMalloc(150 * sizeof(float));;
for(int x1021=0; x1021 < 150; x1021++) {
float x1022 = x890[x1021];
float x1023 = x265[x1021];
float x1024 = x1022 * x1023;
x1020[x1021] = x1024;

}
float* x1028 = (float*)myMalloc(150 * sizeof(float));;
float* x1029 = (float*)myMalloc(150 * sizeof(float));;
for(int x1030=0; x1030 < 150; x1030++) {
float x1031 = x1011[x1030];
float x1032 = x1020[x1030];
float x1033 = x1031 + x1032;
x1029[x1030] = x1033;

}
float* x1037 = (float*)myMalloc(150 * sizeof(float));;
float* x1038 = (float*)myMalloc(150 * sizeof(float));;
for(int x1039=0; x1039 < 150; x1039++) {
float x1040 = x927[x1039];
float x1041 = x274[x1039];
float x1042 = x1040 * x1041;
x1038[x1039] = x1042;

}
float* x1046 = (float*)myMalloc(150 * sizeof(float));;
float* x1047 = (float*)myMalloc(150 * sizeof(float));;
for(int x1048=0; x1048 < 150; x1048++) {
float x1049 = x1029[x1048];
float x1050 = x1038[x1048];
float x1051 = x1049 + x1050;
x1047[x1048] = x1051;

}
float* x1055 = (float*)myMalloc(150 * sizeof(float));;
float* x1056 = (float*)myMalloc(150 * sizeof(float));;
for(int x1057=0; x1057 < 150; x1057++) {
float x1058 = x1047[x1057];
double x1059 = (double)x1058;
double x1060 = tanh(x1059);
float x1061 = (float)x1060;
x1056[x1057] = x1061;

}
float* x1065 = (float*)myMalloc(150 * sizeof(float));;
float* x1066 = (float*)myMalloc(150 * sizeof(float));;
for(int x1067=0; x1067 < 150; x1067++) {
float x1068 = x964[x1067];
float x1069 = x1056[x1067];
float x1070 = x1068 * x1069;
x1066[x1067] = x1070;

}
float* x1074 = (float*)myMalloc(150 * sizeof(float));;
float* x1075 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x164,150,x1066,1,0,x1075,1);
float* x1077 = (float*)myMalloc(5 * sizeof(float));;
float* x1078 = (float*)myMalloc(5 * sizeof(float));;
for(int x1079=0; x1079 < 5; x1079++) {
float x1080 = x1075[x1079];
float x1081 = x173[x1079];
float x1082 = x1080 + x1081;
x1078[x1079] = x1082;

}
float* x1086 = (float*)myMalloc(5 * sizeof(float));;
if (x1088) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1094 = (float*)myMalloc(1 * sizeof(float));;
for(int x1095=0; x1095 < 1; x1095++) {
float x1096 = x261[0];
float x1097 = x270[0];
float x1098 = x1096 + x1097;
x1094[x1095] = x1098;

}
float* x1102 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1103 = 0;
int32_t x1104 = 1;
x1104 *= 1;
x1104 *= 5;
int32_t x1107 = x1103;
bool x1108 = x1107 >= 2;
if (x1108) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1113 = x1107 == 0;
if (x1113) {
int32_t x1114 = x1104;
bool x1115 = x1114 == 5;
if (x1115) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x1122 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1123 = 0;
for(int x1124=0; x1124 < 1; x1124++) {
float x1125 = -3.4028235E38f;
for(int x1126=0; x1126 < 5; x1126++) {
int32_t x1127 = x1123;
float x1128 = x1078[x1127];
float x1129 = x1125;
bool x1130 = x1128 > x1129;
if (x1130) {
float x1131 = x1078[x1127];
x1125 = x1131;
} else {
}
x1123 += 1;

}
float x1138 = x1125;
x1122[x1124] = x1138;

}
float* x1142 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1143 = 0;
for(int x1144=0; x1144 < 1; x1144++) {
for(int x1145=0; x1145 < 5; x1145++) {
int32_t x1146 = x1143;
float x1147 = x1078[x1146];
float x1148 = x1122[x1144];
float x1149 = x1147 - x1148;
double x1150 = (double)x1149;
double x1151 = exp(x1150);
float x1152 = (float)x1151;
x1142[x1146] = x1152;
x1143 += 1;

}

}
float* x1159 = (float*)myMalloc(1 * sizeof(float));;
for(int x1160=0; x1160 < 1; x1160++) {
int32_t x1161 = x1160;
int32_t x1162 = x1160 * 5;
int32_t x1163 = x1162;
for(int x1164=0; x1164 < 5; x1164++) {
for(int x1165=0; x1165 < 1; x1165++) {
int32_t x1166 = x1161;
int32_t x1167 = x1166 + x1165;
float x1168 = x1159[x1167];
int32_t x1169 = x1163;
int32_t x1170 = x1169 + x1165;
float x1171 = x1142[x1170];
float x1172 = x1168 + x1171;
x1159[x1167] = x1172;

}
x1163 += 1;

}

}
x1143 = 0;
for(int x1182=0; x1182 < 1; x1182++) {
float x1183 = x1122[x1182];
float x1184 = x1159[x1182];
double x1185 = (double)x1184;
double x1186 = log(x1185);
float x1187 = (float)x1186;
float x1188 = x1183 + x1187;
for(int x1189=0; x1189 < 5; x1189++) {
int32_t x1190 = x1143;
float x1191 = x1078[x1190];
float x1192 = x1191 - x1188;
x1142[x1190] = x1192;
x1143 += 1;

}

}
float* x1199 = (float*)myMalloc(5 * sizeof(float));;
int* x1200 = x227+x248;
// nllLoss forward in CPU
float* x1202 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1203 = 0;
for(int x1204=0; x1204 < 1; x1204++) {
int32_t x1205 = x1203;
int32_t x1206 = x1200[x1204];
int32_t x1207 = x1205 + x1206;
float x1208 = x1142[x1207];
float x1209 = -1.0f * x1208;
x1202[x1204] = x1209;
x1203 += 5;

}
float* x1214 = (float*)myMalloc(1 * sizeof(float));;
if (x1088) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1219 = (float*)myMalloc(1 * sizeof(float));;
for(int x1220=0; x1220 < 1; x1220++) {
float x1221 = x1094[0];
float x1222 = x1202[0];
float x1223 = x1221 + x1222;
x1219[x1220] = x1223;

}
float* x1227 = (float*)myMalloc(1 * sizeof(float));;
float** x1262 = (float**)myMalloc(6 * sizeof(float*));;
x1262[0] = x1219;
x1262[1] = x1227;
x1262[2] = x1066;
x1262[3] = x1074;
x1262[4] = x1047;
x1262[5] = x1055;
x1228(x1262);
// back prop for + op
if (x1088) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1275=0; x1275 < 1; x1275++) {
float x1276 = x1102[0];
float x1277 = x1094[0];
float x1278 = x1202[0];
float x1279 = x1227[x1275];
float x1280 = x1276 + x1279;
x1102[0] = x1280;
float x1282 = x1214[0];
float x1283 = x1094[0];
float x1284 = x1202[0];
float x1285 = x1227[x1275];
float x1286 = x1282 + x1285;
x1214[0] = x1286;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x1292 = 0;
for(int x1293=0; x1293 < 1; x1293++) {
int32_t x1294 = x1292;
int32_t x1295 = x1200[x1293];
int32_t x1296 = x1294 + x1295;
float x1297 = x1199[x1296];
float x1298 = x1214[x1293];
float x1299 = -1.0f * x1298;
float x1300 = x1297 + x1299;
x1199[x1296] = x1300;
x1292 += 5;

}
float* x1305 = (float*)myMalloc(1 * sizeof(float));;
for(int x1306=0; x1306 < 1; x1306++) {
int32_t x1307 = x1306;
int32_t x1308 = x1306 * 5;
int32_t x1309 = x1308;
for(int x1310=0; x1310 < 5; x1310++) {
for(int x1311=0; x1311 < 1; x1311++) {
int32_t x1312 = x1307;
int32_t x1313 = x1312 + x1311;
float x1314 = x1305[x1313];
int32_t x1315 = x1309;
int32_t x1316 = x1315 + x1311;
float x1317 = x1199[x1316];
float x1318 = x1314 + x1317;
x1305[x1313] = x1318;

}
x1309 += 1;

}

}
int32_t x1327 = 0;
for(int x1328=0; x1328 < 1; x1328++) {
for(int x1329=0; x1329 < 5; x1329++) {
int32_t x1330 = x1327;
float x1331 = x1086[x1330];
float x1332 = x1199[x1330];
float x1333 = x1142[x1330];
float x1337 = x1305[x1328];
double x1334 = (double)x1333;
double x1335 = exp(x1334);
float x1336 = (float)x1335;
float x1338 = x1336 * x1337;
float x1339 = x1332 - x1338;
float x1340 = x1331 + x1339;
x1086[x1330] = x1340;
x1327 += 1;

}

}
// back prop for + op
if (x1088) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1352=0; x1352 < 1; x1352++) {
float x1353 = x262[0];
float x1354 = x261[0];
float x1355 = x270[0];
float x1356 = x1102[x1352];
float x1357 = x1353 + x1356;
x262[0] = x1357;
float x1359 = x271[0];
float x1360 = x261[0];
float x1361 = x270[0];
float x1362 = x1102[x1352];
float x1363 = x1359 + x1362;
x271[0] = x1363;

}
// back prop for + op
for(int x1368=0; x1368 < 5; x1368++) {
float x1369 = x1077[x1368];
float x1370 = x1075[x1368];
float x1371 = x173[x1368];
float x1372 = x1086[x1368];
float x1373 = x1369 + x1372;
x1077[x1368] = x1373;
float x1375 = x195[x1368];
float x1376 = x1075[x1368];
float x1377 = x173[x1368];
float x1378 = x1086[x1368];
float x1379 = x1375 + x1378;
x195[x1368] = x1379;

}
// add_cartesian
int32_t x1384 = 0;
for(int x1385=0; x1385 < 5; x1385++) {
for(int x1386=0; x1386 < 150; x1386++) {
int32_t x1387 = x1384;
int32_t x1388 = x1387 + x1386;
float x1389 = x194[x1388];
float x1390 = x1066[x1386];
float x1391 = x1077[x1385];
float x1392 = x1390 * x1391;
float x1393 = x1389 + x1392;
x194[x1388] = x1393;

}
x1384 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x164,150,x1077,1,1,x1074,1);
// backprop for * op
for(int x1402=0; x1402 < 150; x1402++) {
float x1403 = x976[x1402];
float x1404 = x964[x1402];
float x1405 = x1056[x1402];
float x1406 = x1074[x1402];
float x1407 = x1406 * x1405;
float x1408 = x1403 + x1407;
x976[x1402] = x1408;
float x1410 = x1065[x1402];
float x1411 = x964[x1402];
float x1412 = x1056[x1402];
float x1413 = x1074[x1402];
float x1414 = x1413 * x1411;
float x1415 = x1410 + x1414;
x1065[x1402] = x1415;

}
for(int x1419=0; x1419 < 150; x1419++) {
float x1420 = x1055[x1419];
float x1421 = x1056[x1419];
float x1424 = x1065[x1419];
float x1422 = x1421 * x1421;
float x1423 = 1.0f - x1422;
float x1425 = x1423 * x1424;
float x1426 = x1420 + x1425;
x1055[x1419] = x1426;

}
// back prop for + op
for(int x1431=0; x1431 < 150; x1431++) {
float x1432 = x1037[x1431];
float x1433 = x1029[x1431];
float x1434 = x1038[x1431];
float x1435 = x1055[x1431];
float x1436 = x1432 + x1435;
x1037[x1431] = x1436;
float x1438 = x1046[x1431];
float x1439 = x1029[x1431];
float x1440 = x1038[x1431];
float x1441 = x1055[x1431];
float x1442 = x1438 + x1441;
x1046[x1431] = x1442;

}
// backprop for * op
for(int x1447=0; x1447 < 150; x1447++) {
float x1448 = x939[x1447];
float x1449 = x927[x1447];
float x1450 = x274[x1447];
float x1451 = x1046[x1447];
float x1452 = x1451 * x1450;
float x1453 = x1448 + x1452;
x939[x1447] = x1453;
float x1455 = x275[x1447];
float x1456 = x927[x1447];
float x1457 = x274[x1447];
float x1458 = x1046[x1447];
float x1459 = x1458 * x1456;
float x1460 = x1455 + x1459;
x275[x1447] = x1460;

}
// back prop for + op
for(int x1465=0; x1465 < 150; x1465++) {
float x1466 = x1019[x1465];
float x1467 = x1011[x1465];
float x1468 = x1020[x1465];
float x1469 = x1037[x1465];
float x1470 = x1466 + x1469;
x1019[x1465] = x1470;
float x1472 = x1028[x1465];
float x1473 = x1011[x1465];
float x1474 = x1020[x1465];
float x1475 = x1037[x1465];
float x1476 = x1472 + x1475;
x1028[x1465] = x1476;

}
// backprop for * op
for(int x1481=0; x1481 < 150; x1481++) {
float x1482 = x902[x1481];
float x1483 = x890[x1481];
float x1484 = x265[x1481];
float x1485 = x1028[x1481];
float x1486 = x1485 * x1484;
float x1487 = x1482 + x1486;
x902[x1481] = x1487;
float x1489 = x266[x1481];
float x1490 = x890[x1481];
float x1491 = x265[x1481];
float x1492 = x1028[x1481];
float x1493 = x1492 * x1490;
float x1494 = x1489 + x1493;
x266[x1481] = x1494;

}
// backprop for * op
for(int x1499=0; x1499 < 150; x1499++) {
float x1500 = x865[x1499];
float x1501 = x853[x1499];
float x1502 = x1001[x1499];
float x1503 = x1019[x1499];
float x1504 = x1503 * x1502;
float x1505 = x1500 + x1504;
x865[x1499] = x1505;
float x1507 = x1010[x1499];
float x1508 = x853[x1499];
float x1509 = x1001[x1499];
float x1510 = x1019[x1499];
float x1511 = x1510 * x1508;
float x1512 = x1507 + x1511;
x1010[x1499] = x1512;

}
for(int x1516=0; x1516 < 150; x1516++) {
float x1517 = x1000[x1516];
float x1518 = x1001[x1516];
float x1521 = x1010[x1516];
float x1519 = x1518 * x1518;
float x1520 = 1.0f - x1519;
float x1522 = x1520 * x1521;
float x1523 = x1517 + x1522;
x1000[x1516] = x1523;

}
// back prop for + op
for(int x1528=0; x1528 < 150; x1528++) {
float x1529 = x991[x1528];
float x1530 = x983[x1528];
float x1531 = x163[x1528];
float x1532 = x1000[x1528];
float x1533 = x1529 + x1532;
x991[x1528] = x1533;
float x1535 = x193[x1528];
float x1536 = x983[x1528];
float x1537 = x163[x1528];
float x1538 = x1000[x1528];
float x1539 = x1535 + x1538;
x193[x1528] = x1539;

}
// back prop for + op
for(int x1544=0; x1544 < 150; x1544++) {
float x1545 = x979[x1544];
float x1546 = x977[x1544];
float x1547 = x980[x1544];
float x1548 = x991[x1544];
float x1549 = x1545 + x1548;
x979[x1544] = x1549;
float x1551 = x982[x1544];
float x1552 = x977[x1544];
float x1553 = x980[x1544];
float x1554 = x991[x1544];
float x1555 = x1551 + x1554;
x982[x1544] = x1555;

}
// add_cartesian
int32_t x1560 = 0;
for(int x1561=0; x1561 < 150; x1561++) {
for(int x1562=0; x1562 < 150; x1562++) {
int32_t x1563 = x1560;
int32_t x1564 = x1563 + x1562;
float x1565 = x192[x1564];
float x1566 = x272[x1562];
float x1567 = x982[x1561];
float x1568 = x1566 * x1567;
float x1569 = x1565 + x1568;
x192[x1564] = x1569;

}
x1560 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x155,150,x982,1,1,x273,1);
// add_cartesian
int32_t x1578 = 0;
for(int x1579=0; x1579 < 150; x1579++) {
for(int x1580=0; x1580 < 150; x1580++) {
int32_t x1581 = x1578;
int32_t x1582 = x1581 + x1580;
float x1583 = x191[x1582];
float x1584 = x263[x1580];
float x1585 = x979[x1579];
float x1586 = x1584 * x1585;
float x1587 = x1583 + x1586;
x191[x1582] = x1587;

}
x1578 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x147,150,x979,1,1,x264,1);
for(int x1595=0; x1595 < 150; x1595++) {
float x1596 = x963[x1595];
float x1597 = x964[x1595];
float x1600 = x976[x1595];
float x1598 = 1.0f - x1597;
float x1599 = x1598 * x1597;
float x1601 = x1599 * x1600;
float x1602 = x1596 + x1601;
x963[x1595] = x1602;

}
// back prop for + op
for(int x1607=0; x1607 < 150; x1607++) {
float x1608 = x954[x1607];
float x1609 = x946[x1607];
float x1610 = x146[x1607];
float x1611 = x963[x1607];
float x1612 = x1608 + x1611;
x954[x1607] = x1612;
float x1614 = x190[x1607];
float x1615 = x946[x1607];
float x1616 = x146[x1607];
float x1617 = x963[x1607];
float x1618 = x1614 + x1617;
x190[x1607] = x1618;

}
// back prop for + op
for(int x1623=0; x1623 < 150; x1623++) {
float x1624 = x942[x1623];
float x1625 = x940[x1623];
float x1626 = x943[x1623];
float x1627 = x954[x1623];
float x1628 = x1624 + x1627;
x942[x1623] = x1628;
float x1630 = x945[x1623];
float x1631 = x940[x1623];
float x1632 = x943[x1623];
float x1633 = x954[x1623];
float x1634 = x1630 + x1633;
x945[x1623] = x1634;

}
// add_cartesian
int32_t x1639 = 0;
for(int x1640=0; x1640 < 150; x1640++) {
for(int x1641=0; x1641 < 150; x1641++) {
int32_t x1642 = x1639;
int32_t x1643 = x1642 + x1641;
float x1644 = x189[x1643];
float x1645 = x272[x1641];
float x1646 = x945[x1640];
float x1647 = x1645 * x1646;
float x1648 = x1644 + x1647;
x189[x1643] = x1648;

}
x1639 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x138,150,x945,1,1,x273,1);
// add_cartesian
int32_t x1657 = 0;
for(int x1658=0; x1658 < 150; x1658++) {
for(int x1659=0; x1659 < 150; x1659++) {
int32_t x1660 = x1657;
int32_t x1661 = x1660 + x1659;
float x1662 = x188[x1661];
float x1663 = x263[x1659];
float x1664 = x942[x1658];
float x1665 = x1663 * x1664;
float x1666 = x1662 + x1665;
x188[x1661] = x1666;

}
x1657 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x130,150,x942,1,1,x264,1);
for(int x1674=0; x1674 < 150; x1674++) {
float x1675 = x926[x1674];
float x1676 = x927[x1674];
float x1679 = x939[x1674];
float x1677 = 1.0f - x1676;
float x1678 = x1677 * x1676;
float x1680 = x1678 * x1679;
float x1681 = x1675 + x1680;
x926[x1674] = x1681;

}
// back prop for + op
for(int x1686=0; x1686 < 150; x1686++) {
float x1687 = x917[x1686];
float x1688 = x909[x1686];
float x1689 = x129[x1686];
float x1690 = x926[x1686];
float x1691 = x1687 + x1690;
x917[x1686] = x1691;
float x1693 = x187[x1686];
float x1694 = x909[x1686];
float x1695 = x129[x1686];
float x1696 = x926[x1686];
float x1697 = x1693 + x1696;
x187[x1686] = x1697;

}
// back prop for + op
for(int x1702=0; x1702 < 150; x1702++) {
float x1703 = x905[x1702];
float x1704 = x903[x1702];
float x1705 = x906[x1702];
float x1706 = x917[x1702];
float x1707 = x1703 + x1706;
x905[x1702] = x1707;
float x1709 = x908[x1702];
float x1710 = x903[x1702];
float x1711 = x906[x1702];
float x1712 = x917[x1702];
float x1713 = x1709 + x1712;
x908[x1702] = x1713;

}
// add_cartesian
int32_t x1718 = 0;
for(int x1719=0; x1719 < 150; x1719++) {
for(int x1720=0; x1720 < 150; x1720++) {
int32_t x1721 = x1718;
int32_t x1722 = x1721 + x1720;
float x1723 = x186[x1722];
float x1724 = x272[x1720];
float x1725 = x908[x1719];
float x1726 = x1724 * x1725;
float x1727 = x1723 + x1726;
x186[x1722] = x1727;

}
x1718 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x121,150,x908,1,1,x273,1);
// add_cartesian
int32_t x1736 = 0;
for(int x1737=0; x1737 < 150; x1737++) {
for(int x1738=0; x1738 < 150; x1738++) {
int32_t x1739 = x1736;
int32_t x1740 = x1739 + x1738;
float x1741 = x185[x1740];
float x1742 = x263[x1738];
float x1743 = x905[x1737];
float x1744 = x1742 * x1743;
float x1745 = x1741 + x1744;
x185[x1740] = x1745;

}
x1736 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x113,150,x905,1,1,x264,1);
for(int x1753=0; x1753 < 150; x1753++) {
float x1754 = x889[x1753];
float x1755 = x890[x1753];
float x1758 = x902[x1753];
float x1756 = 1.0f - x1755;
float x1757 = x1756 * x1755;
float x1759 = x1757 * x1758;
float x1760 = x1754 + x1759;
x889[x1753] = x1760;

}
// back prop for + op
for(int x1765=0; x1765 < 150; x1765++) {
float x1766 = x880[x1765];
float x1767 = x872[x1765];
float x1768 = x129[x1765];
float x1769 = x889[x1765];
float x1770 = x1766 + x1769;
x880[x1765] = x1770;
float x1772 = x187[x1765];
float x1773 = x872[x1765];
float x1774 = x129[x1765];
float x1775 = x889[x1765];
float x1776 = x1772 + x1775;
x187[x1765] = x1776;

}
// back prop for + op
for(int x1781=0; x1781 < 150; x1781++) {
float x1782 = x868[x1781];
float x1783 = x866[x1781];
float x1784 = x869[x1781];
float x1785 = x880[x1781];
float x1786 = x1782 + x1785;
x868[x1781] = x1786;
float x1788 = x871[x1781];
float x1789 = x866[x1781];
float x1790 = x869[x1781];
float x1791 = x880[x1781];
float x1792 = x1788 + x1791;
x871[x1781] = x1792;

}
// add_cartesian
int32_t x1797 = 0;
for(int x1798=0; x1798 < 150; x1798++) {
for(int x1799=0; x1799 < 150; x1799++) {
int32_t x1800 = x1797;
int32_t x1801 = x1800 + x1799;
float x1802 = x184[x1801];
float x1803 = x272[x1799];
float x1804 = x871[x1798];
float x1805 = x1803 * x1804;
float x1806 = x1802 + x1805;
x184[x1801] = x1806;

}
x1797 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x105,150,x871,1,1,x273,1);
// add_cartesian
int32_t x1815 = 0;
for(int x1816=0; x1816 < 150; x1816++) {
for(int x1817=0; x1817 < 150; x1817++) {
int32_t x1818 = x1815;
int32_t x1819 = x1818 + x1817;
float x1820 = x183[x1819];
float x1821 = x263[x1817];
float x1822 = x868[x1816];
float x1823 = x1821 * x1822;
float x1824 = x1820 + x1823;
x183[x1819] = x1824;

}
x1815 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x97,150,x868,1,1,x264,1);
for(int x1832=0; x1832 < 150; x1832++) {
float x1833 = x852[x1832];
float x1834 = x853[x1832];
float x1837 = x865[x1832];
float x1835 = 1.0f - x1834;
float x1836 = x1835 * x1834;
float x1838 = x1836 * x1837;
float x1839 = x1833 + x1838;
x852[x1832] = x1839;

}
// back prop for + op
for(int x1844=0; x1844 < 150; x1844++) {
float x1845 = x843[x1844];
float x1846 = x835[x1844];
float x1847 = x96[x1844];
float x1848 = x852[x1844];
float x1849 = x1845 + x1848;
x843[x1844] = x1849;
float x1851 = x182[x1844];
float x1852 = x835[x1844];
float x1853 = x96[x1844];
float x1854 = x852[x1844];
float x1855 = x1851 + x1854;
x182[x1844] = x1855;

}
// back prop for + op
for(int x1860=0; x1860 < 150; x1860++) {
float x1861 = x831[x1860];
float x1862 = x829[x1860];
float x1863 = x832[x1860];
float x1864 = x843[x1860];
float x1865 = x1861 + x1864;
x831[x1860] = x1865;
float x1867 = x834[x1860];
float x1868 = x829[x1860];
float x1869 = x832[x1860];
float x1870 = x843[x1860];
float x1871 = x1867 + x1870;
x834[x1860] = x1871;

}
// add_cartesian
int32_t x1876 = 0;
for(int x1877=0; x1877 < 150; x1877++) {
for(int x1878=0; x1878 < 150; x1878++) {
int32_t x1879 = x1876;
int32_t x1880 = x1879 + x1878;
float x1881 = x181[x1880];
float x1882 = x272[x1878];
float x1883 = x834[x1877];
float x1884 = x1882 * x1883;
float x1885 = x1881 + x1884;
x181[x1880] = x1885;

}
x1876 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x88,150,x834,1,1,x273,1);
// add_cartesian
int32_t x1894 = 0;
for(int x1895=0; x1895 < 150; x1895++) {
for(int x1896=0; x1896 < 150; x1896++) {
int32_t x1897 = x1894;
int32_t x1898 = x1897 + x1896;
float x1899 = x180[x1898];
float x1900 = x263[x1896];
float x1901 = x831[x1895];
float x1902 = x1900 * x1901;
float x1903 = x1899 + x1902;
x180[x1898] = x1903;

}
x1894 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x79,150,x831,1,1,x264,1);
}
};
x244(x267,x268,x1914);
};
x244(x258,x259,x1924);
} else {
float** x1951 = (float**)myMalloc(6 * sizeof(float*));;
x1951[0] = x238;
x1951[1] = x239;
x1951[2] = x240;
x1951[3] = x241;
x1951[4] = x242;
x1951[5] = x243;
function<void(float**)> x249 = x246;
function<void(float**)> x1934 = [&](float** x1935) {
float* x1936 = x1935[0];
float* x1937 = x1935[1];
float* x1938 = x1935[2];
float* x1939 = x1935[3];
float* x1940 = x1935[4];
float* x1941 = x1935[5];
float** x1942 = (float**)myMalloc(6 * sizeof(float*));;
x1942[0] = x1936;
x1942[1] = x1937;
x1942[2] = x1938;
x1942[3] = x1939;
x1942[4] = x1940;
x1942[5] = x1941;
x249(x1942);
};
x1934(x1951);
}
};
float* x234 = (float*)myMalloc(1 * sizeof(float));;
float* x235 = (float*)myMalloc(1 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
float* x237 = (float*)myMalloc(1 * sizeof(float));;
float** x1982 = (float**)myMalloc(6 * sizeof(float*));;
x1982[0] = x238;
x1982[1] = x239;
x1982[2] = x240;
x1982[3] = x241;
x1982[4] = x242;
x1982[5] = x243;
function<void(float**)> x1962 = [&](float** x1963) {
float* x1964 = x1963[0];
float* x1965 = x1963[1];
float* x1966 = x1963[2];
float* x1967 = x1963[3];
float* x1968 = x1963[4];
float* x1969 = x1963[5];
// make sure the size of loss is 1
for(int x1971=0; x1971 < 1; x1971++) {
x1965[x1971] = 1.0f;

}
// backend is lantern.TensorDslCPU$BackendCPU@4ddd0a27
for(int x1976=0; x1976 < 1; x1976++) {
float x1977 = x1964[x1976];
x237[x1976] = x1977;

}
};
x244(0,x1962,x1982);
float x1991 = x237[0];
float x1992 = x223;
float x1993 = (float)x224;
float x1994 = x1992 * x1993;
int32_t x1995 = x224 + 1;
float x1996 = (float)x1995;
float x1997 = x1994 / x1996;
float x1998 = x1991 / x1996;
float x1999 = x1997 + x1998;
x223 = x1999;
for(int x2001=0; x2001 < 45000; x2001++) {
float x2002 = x174[x2001];
float x2003 = x2002;
float x2004 = x196[x2001];
float x2005 = x2003;
float x2006 = x2005 * x2005;
float x2007 = x2004 + x2006;
x196[x2001] = x2007;
float x2009 = x50[x2001];
float x2011 = x196[x2001];
float x2010 = 0.05f * x2005;
double x2012 = (double)x2011;
double x2013 = x2012 + 9.99999993922529E-9;
double x2014 = sqrt(x2013);
float x2015 = (float)x2014;
float x2016 = x2010 / x2015;
float x2017 = x2009 - x2016;
x50[x2001] = x2017;
x174[x2001] = 0.0f;

}
for(int x2022=0; x2022 < 150; x2022++) {
float x2023 = x175[x2022];
float x2024 = x2023;
float x2025 = x197[x2022];
float x2026 = x2024;
float x2027 = x2026 * x2026;
float x2028 = x2025 + x2027;
x197[x2022] = x2028;
float x2030 = x60[x2022];
float x2032 = x197[x2022];
float x2031 = 0.05f * x2026;
double x2033 = (double)x2032;
double x2034 = x2033 + 9.99999993922529E-9;
double x2035 = sqrt(x2034);
float x2036 = (float)x2035;
float x2037 = x2031 / x2036;
float x2038 = x2030 - x2037;
x60[x2022] = x2038;
x175[x2022] = 0.0f;

}
for(int x2043=0; x2043 < 45000; x2043++) {
float x2044 = x176[x2043];
float x2045 = x2044;
float x2046 = x198[x2043];
float x2047 = x2045;
float x2048 = x2047 * x2047;
float x2049 = x2046 + x2048;
x198[x2043] = x2049;
float x2051 = x61[x2043];
float x2053 = x198[x2043];
float x2052 = 0.05f * x2047;
double x2054 = (double)x2053;
double x2055 = x2054 + 9.99999993922529E-9;
double x2056 = sqrt(x2055);
float x2057 = (float)x2056;
float x2058 = x2052 / x2057;
float x2059 = x2051 - x2058;
x61[x2043] = x2059;
x176[x2043] = 0.0f;

}
for(int x2064=0; x2064 < 150; x2064++) {
float x2065 = x177[x2064];
float x2066 = x2065;
float x2067 = x199[x2064];
float x2068 = x2066;
float x2069 = x2068 * x2068;
float x2070 = x2067 + x2069;
x199[x2064] = x2070;
float x2072 = x69[x2064];
float x2074 = x199[x2064];
float x2073 = 0.05f * x2068;
double x2075 = (double)x2074;
double x2076 = x2075 + 9.99999993922529E-9;
double x2077 = sqrt(x2076);
float x2078 = (float)x2077;
float x2079 = x2073 / x2078;
float x2080 = x2072 - x2079;
x69[x2064] = x2080;
x177[x2064] = 0.0f;

}
for(int x2085=0; x2085 < 45000; x2085++) {
float x2086 = x178[x2085];
float x2087 = x2086;
float x2088 = x200[x2085];
float x2089 = x2087;
float x2090 = x2089 * x2089;
float x2091 = x2088 + x2090;
x200[x2085] = x2091;
float x2093 = x70[x2085];
float x2095 = x200[x2085];
float x2094 = 0.05f * x2089;
double x2096 = (double)x2095;
double x2097 = x2096 + 9.99999993922529E-9;
double x2098 = sqrt(x2097);
float x2099 = (float)x2098;
float x2100 = x2094 / x2099;
float x2101 = x2093 - x2100;
x70[x2085] = x2101;
x178[x2085] = 0.0f;

}
for(int x2106=0; x2106 < 150; x2106++) {
float x2107 = x179[x2106];
float x2108 = x2107;
float x2109 = x201[x2106];
float x2110 = x2108;
float x2111 = x2110 * x2110;
float x2112 = x2109 + x2111;
x201[x2106] = x2112;
float x2114 = x78[x2106];
float x2116 = x201[x2106];
float x2115 = 0.05f * x2110;
double x2117 = (double)x2116;
double x2118 = x2117 + 9.99999993922529E-9;
double x2119 = sqrt(x2118);
float x2120 = (float)x2119;
float x2121 = x2115 / x2120;
float x2122 = x2114 - x2121;
x78[x2106] = x2122;
x179[x2106] = 0.0f;

}
for(int x2127=0; x2127 < 22500; x2127++) {
float x2128 = x180[x2127];
float x2129 = x2128;
float x2130 = x202[x2127];
float x2131 = x2129;
float x2132 = x2131 * x2131;
float x2133 = x2130 + x2132;
x202[x2127] = x2133;
float x2135 = x79[x2127];
float x2137 = x202[x2127];
float x2136 = 0.05f * x2131;
double x2138 = (double)x2137;
double x2139 = x2138 + 9.99999993922529E-9;
double x2140 = sqrt(x2139);
float x2141 = (float)x2140;
float x2142 = x2136 / x2141;
float x2143 = x2135 - x2142;
x79[x2127] = x2143;
x180[x2127] = 0.0f;

}
for(int x2148=0; x2148 < 22500; x2148++) {
float x2149 = x181[x2148];
float x2150 = x2149;
float x2151 = x203[x2148];
float x2152 = x2150;
float x2153 = x2152 * x2152;
float x2154 = x2151 + x2153;
x203[x2148] = x2154;
float x2156 = x88[x2148];
float x2158 = x203[x2148];
float x2157 = 0.05f * x2152;
double x2159 = (double)x2158;
double x2160 = x2159 + 9.99999993922529E-9;
double x2161 = sqrt(x2160);
float x2162 = (float)x2161;
float x2163 = x2157 / x2162;
float x2164 = x2156 - x2163;
x88[x2148] = x2164;
x181[x2148] = 0.0f;

}
for(int x2169=0; x2169 < 150; x2169++) {
float x2170 = x182[x2169];
float x2171 = x2170;
float x2172 = x204[x2169];
float x2173 = x2171;
float x2174 = x2173 * x2173;
float x2175 = x2172 + x2174;
x204[x2169] = x2175;
float x2177 = x96[x2169];
float x2179 = x204[x2169];
float x2178 = 0.05f * x2173;
double x2180 = (double)x2179;
double x2181 = x2180 + 9.99999993922529E-9;
double x2182 = sqrt(x2181);
float x2183 = (float)x2182;
float x2184 = x2178 / x2183;
float x2185 = x2177 - x2184;
x96[x2169] = x2185;
x182[x2169] = 0.0f;

}
for(int x2190=0; x2190 < 22500; x2190++) {
float x2191 = x183[x2190];
float x2192 = x2191;
float x2193 = x205[x2190];
float x2194 = x2192;
float x2195 = x2194 * x2194;
float x2196 = x2193 + x2195;
x205[x2190] = x2196;
float x2198 = x97[x2190];
float x2200 = x205[x2190];
float x2199 = 0.05f * x2194;
double x2201 = (double)x2200;
double x2202 = x2201 + 9.99999993922529E-9;
double x2203 = sqrt(x2202);
float x2204 = (float)x2203;
float x2205 = x2199 / x2204;
float x2206 = x2198 - x2205;
x97[x2190] = x2206;
x183[x2190] = 0.0f;

}
for(int x2211=0; x2211 < 22500; x2211++) {
float x2212 = x184[x2211];
float x2213 = x2212;
float x2214 = x206[x2211];
float x2215 = x2213;
float x2216 = x2215 * x2215;
float x2217 = x2214 + x2216;
x206[x2211] = x2217;
float x2219 = x105[x2211];
float x2221 = x206[x2211];
float x2220 = 0.05f * x2215;
double x2222 = (double)x2221;
double x2223 = x2222 + 9.99999993922529E-9;
double x2224 = sqrt(x2223);
float x2225 = (float)x2224;
float x2226 = x2220 / x2225;
float x2227 = x2219 - x2226;
x105[x2211] = x2227;
x184[x2211] = 0.0f;

}
for(int x2232=0; x2232 < 22500; x2232++) {
float x2233 = x185[x2232];
float x2234 = x2233;
float x2235 = x207[x2232];
float x2236 = x2234;
float x2237 = x2236 * x2236;
float x2238 = x2235 + x2237;
x207[x2232] = x2238;
float x2240 = x113[x2232];
float x2242 = x207[x2232];
float x2241 = 0.05f * x2236;
double x2243 = (double)x2242;
double x2244 = x2243 + 9.99999993922529E-9;
double x2245 = sqrt(x2244);
float x2246 = (float)x2245;
float x2247 = x2241 / x2246;
float x2248 = x2240 - x2247;
x113[x2232] = x2248;
x185[x2232] = 0.0f;

}
for(int x2253=0; x2253 < 22500; x2253++) {
float x2254 = x186[x2253];
float x2255 = x2254;
float x2256 = x208[x2253];
float x2257 = x2255;
float x2258 = x2257 * x2257;
float x2259 = x2256 + x2258;
x208[x2253] = x2259;
float x2261 = x121[x2253];
float x2263 = x208[x2253];
float x2262 = 0.05f * x2257;
double x2264 = (double)x2263;
double x2265 = x2264 + 9.99999993922529E-9;
double x2266 = sqrt(x2265);
float x2267 = (float)x2266;
float x2268 = x2262 / x2267;
float x2269 = x2261 - x2268;
x121[x2253] = x2269;
x186[x2253] = 0.0f;

}
for(int x2274=0; x2274 < 150; x2274++) {
float x2275 = x187[x2274];
float x2276 = x2275;
float x2277 = x209[x2274];
float x2278 = x2276;
float x2279 = x2278 * x2278;
float x2280 = x2277 + x2279;
x209[x2274] = x2280;
float x2282 = x129[x2274];
float x2284 = x209[x2274];
float x2283 = 0.05f * x2278;
double x2285 = (double)x2284;
double x2286 = x2285 + 9.99999993922529E-9;
double x2287 = sqrt(x2286);
float x2288 = (float)x2287;
float x2289 = x2283 / x2288;
float x2290 = x2282 - x2289;
x129[x2274] = x2290;
x187[x2274] = 0.0f;

}
for(int x2295=0; x2295 < 22500; x2295++) {
float x2296 = x188[x2295];
float x2297 = x2296;
float x2298 = x210[x2295];
float x2299 = x2297;
float x2300 = x2299 * x2299;
float x2301 = x2298 + x2300;
x210[x2295] = x2301;
float x2303 = x130[x2295];
float x2305 = x210[x2295];
float x2304 = 0.05f * x2299;
double x2306 = (double)x2305;
double x2307 = x2306 + 9.99999993922529E-9;
double x2308 = sqrt(x2307);
float x2309 = (float)x2308;
float x2310 = x2304 / x2309;
float x2311 = x2303 - x2310;
x130[x2295] = x2311;
x188[x2295] = 0.0f;

}
for(int x2316=0; x2316 < 22500; x2316++) {
float x2317 = x189[x2316];
float x2318 = x2317;
float x2319 = x211[x2316];
float x2320 = x2318;
float x2321 = x2320 * x2320;
float x2322 = x2319 + x2321;
x211[x2316] = x2322;
float x2324 = x138[x2316];
float x2326 = x211[x2316];
float x2325 = 0.05f * x2320;
double x2327 = (double)x2326;
double x2328 = x2327 + 9.99999993922529E-9;
double x2329 = sqrt(x2328);
float x2330 = (float)x2329;
float x2331 = x2325 / x2330;
float x2332 = x2324 - x2331;
x138[x2316] = x2332;
x189[x2316] = 0.0f;

}
for(int x2337=0; x2337 < 150; x2337++) {
float x2338 = x190[x2337];
float x2339 = x2338;
float x2340 = x212[x2337];
float x2341 = x2339;
float x2342 = x2341 * x2341;
float x2343 = x2340 + x2342;
x212[x2337] = x2343;
float x2345 = x146[x2337];
float x2347 = x212[x2337];
float x2346 = 0.05f * x2341;
double x2348 = (double)x2347;
double x2349 = x2348 + 9.99999993922529E-9;
double x2350 = sqrt(x2349);
float x2351 = (float)x2350;
float x2352 = x2346 / x2351;
float x2353 = x2345 - x2352;
x146[x2337] = x2353;
x190[x2337] = 0.0f;

}
for(int x2358=0; x2358 < 22500; x2358++) {
float x2359 = x191[x2358];
float x2360 = x2359;
float x2361 = x213[x2358];
float x2362 = x2360;
float x2363 = x2362 * x2362;
float x2364 = x2361 + x2363;
x213[x2358] = x2364;
float x2366 = x147[x2358];
float x2368 = x213[x2358];
float x2367 = 0.05f * x2362;
double x2369 = (double)x2368;
double x2370 = x2369 + 9.99999993922529E-9;
double x2371 = sqrt(x2370);
float x2372 = (float)x2371;
float x2373 = x2367 / x2372;
float x2374 = x2366 - x2373;
x147[x2358] = x2374;
x191[x2358] = 0.0f;

}
for(int x2379=0; x2379 < 22500; x2379++) {
float x2380 = x192[x2379];
float x2381 = x2380;
float x2382 = x214[x2379];
float x2383 = x2381;
float x2384 = x2383 * x2383;
float x2385 = x2382 + x2384;
x214[x2379] = x2385;
float x2387 = x155[x2379];
float x2389 = x214[x2379];
float x2388 = 0.05f * x2383;
double x2390 = (double)x2389;
double x2391 = x2390 + 9.99999993922529E-9;
double x2392 = sqrt(x2391);
float x2393 = (float)x2392;
float x2394 = x2388 / x2393;
float x2395 = x2387 - x2394;
x155[x2379] = x2395;
x192[x2379] = 0.0f;

}
for(int x2400=0; x2400 < 150; x2400++) {
float x2401 = x193[x2400];
float x2402 = x2401;
float x2403 = x215[x2400];
float x2404 = x2402;
float x2405 = x2404 * x2404;
float x2406 = x2403 + x2405;
x215[x2400] = x2406;
float x2408 = x163[x2400];
float x2410 = x215[x2400];
float x2409 = 0.05f * x2404;
double x2411 = (double)x2410;
double x2412 = x2411 + 9.99999993922529E-9;
double x2413 = sqrt(x2412);
float x2414 = (float)x2413;
float x2415 = x2409 / x2414;
float x2416 = x2408 - x2415;
x163[x2400] = x2416;
x193[x2400] = 0.0f;

}
for(int x2421=0; x2421 < 750; x2421++) {
float x2422 = x194[x2421];
float x2423 = x2422;
float x2424 = x216[x2421];
float x2425 = x2423;
float x2426 = x2425 * x2425;
float x2427 = x2424 + x2426;
x216[x2421] = x2427;
float x2429 = x164[x2421];
float x2431 = x216[x2421];
float x2430 = 0.05f * x2425;
double x2432 = (double)x2431;
double x2433 = x2432 + 9.99999993922529E-9;
double x2434 = sqrt(x2433);
float x2435 = (float)x2434;
float x2436 = x2430 / x2435;
float x2437 = x2429 - x2436;
x164[x2421] = x2437;
x194[x2421] = 0.0f;

}
for(int x2442=0; x2442 < 5; x2442++) {
float x2443 = x195[x2442];
float x2444 = x2443;
float x2445 = x217[x2442];
float x2446 = x2444;
float x2447 = x2446 * x2446;
float x2448 = x2445 + x2447;
x217[x2442] = x2448;
float x2450 = x173[x2442];
float x2452 = x217[x2442];
float x2451 = 0.05f * x2446;
double x2453 = (double)x2452;
double x2454 = x2453 + 9.99999993922529E-9;
double x2455 = sqrt(x2454);
float x2456 = (float)x2455;
float x2457 = x2451 / x2456;
float x2458 = x2450 - x2457;
x173[x2442] = x2458;
x195[x2442] = 0.0f;

}
int64_t x2463 = (long)mallocAddr;
int64_t x2464 = x2463 - x219;
memset((void*)x219, 0, x2464);
mallocAddr = (void*)x219;

}
float x2469 = x223;
double x2470 = (double)x2469;
x218[x222] = x2470;
double x2472 = ((double)clock() / CLOCKS_PER_SEC);
double x2473 = x2472 - x220;
printf("epoc %d, average_loss %f, time %lf\n",x222,x2469,x2473);

}
double x2477 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2481 = (long)fopen(x0, "w");
fprintf((FILE *)x2481, "unit: %s\n", "1 epoch");
for(int x2483=0; x2483 < 6; x2483++) {
double x2484 = x218[x2483];
fprintf((FILE *)x2481, "%lf\n", x2484);

}
double x2478 = x220 - x2;
double x2479 = x2477 - x220;
double x2480 = x2479 / 6.0;
fprintf((FILE *)x2481, "run time: %lf %lf\n", x2478, x2480);
fclose((FILE*)x2481);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

