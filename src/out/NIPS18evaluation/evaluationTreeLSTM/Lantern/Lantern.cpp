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
bool x1083 = true || true;
bool x1084 = x1083 || true;
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
float** x1913 = (float**)myMalloc(6 * sizeof(float*));;
x1913[0] = x238;
x1913[1] = x239;
x1913[2] = x240;
x1913[3] = x241;
x1913[4] = x242;
x1913[5] = x243;
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
function<void(float**)> x1232 = [&](float** x1233) {
float* x1234 = x1233[0];
float* x1235 = x1233[1];
float* x1236 = x1233[2];
float* x1237 = x1233[3];
float* x1238 = x1233[4];
float* x1239 = x1233[5];
float** x1240 = (float**)myMalloc(6 * sizeof(float*));;
x1240[0] = x1234;
x1240[1] = x1235;
x1240[2] = x1236;
x1240[3] = x1237;
x1240[4] = x1238;
x1240[5] = x1239;
x249(x1240);
};
function<void(float**)> x1224 = [&](float** x1225) {
float* x1226 = x1225[0];
float* x1227 = x1225[1];
float* x1228 = x1225[2];
float* x1229 = x1225[3];
float* x1230 = x1225[4];
float* x1231 = x1225[5];
float** x1249 = (float**)myMalloc(6 * sizeof(float*));;
x1249[0] = x1226;
x1249[1] = x1227;
x1249[2] = x1228;
x1249[3] = x1229;
x1249[4] = x1230;
x1249[5] = x1231;
x1232(x1249);
};
function<void(float**)> x259 = [&](float** x260) {
float* x261 = x260[0];
float* x262 = x260[1];
float* x263 = x260[2];
float* x264 = x260[3];
float* x265 = x260[4];
float* x266 = x260[5];
int32_t x267 = x233[x248];
float** x1903 = (float**)myMalloc(6 * sizeof(float*));;
x1903[0] = x238;
x1903[1] = x239;
x1903[2] = x240;
x1903[3] = x241;
x1903[4] = x242;
x1903[5] = x243;
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
float x615 = x398[x613];
float x616 = x614 + x615;
x388[x613] = x616;

}
for(int x620=0; x620 < 5; x620++) {
float x621 = x195[x620];
float x622 = x398[x620];
float x623 = x621 + x622;
x195[x620] = x623;

}
// add_cartesian
int32_t x628 = 0;
for(int x629=0; x629 < 5; x629++) {
for(int x630=0; x630 < 150; x630++) {
int32_t x631 = x628;
int32_t x632 = x631 + x630;
float x633 = x194[x632];
float x634 = x377[x630];
float x635 = x388[x629];
float x636 = x634 * x635;
float x637 = x633 + x636;
x194[x632] = x637;

}
x628 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x164,150,x388,1,1,x385,1);
// backprop for * op
for(int x646=0; x646 < 150; x646++) {
float x647 = x335[x646];
float x648 = x323[x646];
float x649 = x367[x646];
float x650 = x385[x646];
float x651 = x650 * x649;
float x652 = x647 + x651;
x335[x646] = x652;
float x654 = x376[x646];
float x655 = x323[x646];
float x656 = x367[x646];
float x657 = x385[x646];
float x658 = x657 * x655;
float x659 = x654 + x658;
x376[x646] = x659;

}
for(int x663=0; x663 < 150; x663++) {
float x664 = x366[x663];
float x665 = x367[x663];
float x668 = x376[x663];
float x666 = x665 * x665;
float x667 = 1.0f - x666;
float x669 = x667 * x668;
float x670 = x664 + x669;
x366[x663] = x670;

}
// backprop for * op
for(int x675=0; x675 < 150; x675++) {
float x676 = x310[x675];
float x677 = x298[x675];
float x678 = x348[x675];
float x679 = x366[x675];
float x680 = x679 * x678;
float x681 = x676 + x680;
x310[x675] = x681;
float x683 = x357[x675];
float x684 = x298[x675];
float x685 = x348[x675];
float x686 = x366[x675];
float x687 = x686 * x684;
float x688 = x683 + x687;
x357[x675] = x688;

}
for(int x692=0; x692 < 150; x692++) {
float x693 = x347[x692];
float x694 = x348[x692];
float x697 = x357[x692];
float x695 = x694 * x694;
float x696 = 1.0f - x695;
float x698 = x696 * x697;
float x699 = x693 + x698;
x347[x692] = x699;

}
// back prop for + op
for(int x704=0; x704 < 150; x704++) {
float x705 = x338[x704];
float x706 = x347[x704];
float x707 = x705 + x706;
x338[x704] = x707;

}
for(int x711=0; x711 < 150; x711++) {
float x712 = x179[x711];
float x713 = x347[x711];
float x714 = x712 + x713;
x179[x711] = x714;

}
// add_cartesian
int32_t x719 = 0;
for(int x720=0; x720 < 150; x720++) {
for(int x721=0; x721 < 300; x721++) {
int32_t x722 = x719;
int32_t x723 = x722 + x721;
float x724 = x178[x723];
float x725 = x283[x721];
float x726 = x338[x720];
float x727 = x725 * x726;
float x728 = x724 + x727;
x178[x723] = x728;

}
x719 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x70,300,x338,1,1,x284,1);
for(int x736=0; x736 < 150; x736++) {
float x737 = x322[x736];
float x738 = x323[x736];
float x741 = x335[x736];
float x739 = 1.0f - x738;
float x740 = x739 * x738;
float x742 = x740 * x741;
float x743 = x737 + x742;
x322[x736] = x743;

}
// back prop for + op
for(int x748=0; x748 < 150; x748++) {
float x749 = x313[x748];
float x750 = x322[x748];
float x751 = x749 + x750;
x313[x748] = x751;

}
for(int x755=0; x755 < 150; x755++) {
float x756 = x177[x755];
float x757 = x322[x755];
float x758 = x756 + x757;
x177[x755] = x758;

}
// add_cartesian
int32_t x763 = 0;
for(int x764=0; x764 < 150; x764++) {
for(int x765=0; x765 < 300; x765++) {
int32_t x766 = x763;
int32_t x767 = x766 + x765;
float x768 = x176[x767];
float x769 = x283[x765];
float x770 = x313[x764];
float x771 = x769 * x770;
float x772 = x768 + x771;
x176[x767] = x772;

}
x763 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x61,300,x313,1,1,x284,1);
for(int x780=0; x780 < 150; x780++) {
float x781 = x297[x780];
float x782 = x298[x780];
float x785 = x310[x780];
float x783 = 1.0f - x782;
float x784 = x783 * x782;
float x786 = x784 * x785;
float x787 = x781 + x786;
x297[x780] = x787;

}
// back prop for + op
for(int x792=0; x792 < 150; x792++) {
float x793 = x287[x792];
float x794 = x297[x792];
float x795 = x793 + x794;
x287[x792] = x795;

}
for(int x799=0; x799 < 150; x799++) {
float x800 = x175[x799];
float x801 = x297[x799];
float x802 = x800 + x801;
x175[x799] = x802;

}
// add_cartesian
int32_t x807 = 0;
for(int x808=0; x808 < 150; x808++) {
for(int x809=0; x809 < 300; x809++) {
int32_t x810 = x807;
int32_t x811 = x810 + x809;
float x812 = x174[x811];
float x813 = x283[x809];
float x814 = x287[x808];
float x815 = x813 * x814;
float x816 = x812 + x815;
x174[x811] = x816;

}
x807 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x50,300,x287,1,1,x284,1);
} else {
float* x825 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x79,150,x263,1,0,x825,1);
float* x827 = (float*)myMalloc(150 * sizeof(float));;
float* x828 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x88,150,x272,1,0,x828,1);
float* x830 = (float*)myMalloc(150 * sizeof(float));;
float* x831 = (float*)myMalloc(150 * sizeof(float));;
for(int x832=0; x832 < 150; x832++) {
float x833 = x825[x832];
float x834 = x828[x832];
float x835 = x833 + x834;
x831[x832] = x835;

}
float* x839 = (float*)myMalloc(150 * sizeof(float));;
float* x840 = (float*)myMalloc(150 * sizeof(float));;
for(int x841=0; x841 < 150; x841++) {
float x842 = x831[x841];
float x843 = x96[x841];
float x844 = x842 + x843;
x840[x841] = x844;

}
float* x848 = (float*)myMalloc(150 * sizeof(float));;
float* x849 = (float*)myMalloc(150 * sizeof(float));;
for(int x850=0; x850 < 150; x850++) {
float x851 = x840[x850];
float x852 = -1.0f * x851;
double x853 = (double)x852;
double x854 = exp(x853);
float x855 = (float)x854;
float x856 = x855 + 1.0f;
float x857 = 1.0f / x856;
x849[x850] = x857;

}
float* x861 = (float*)myMalloc(150 * sizeof(float));;
float* x862 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x97,150,x263,1,0,x862,1);
float* x864 = (float*)myMalloc(150 * sizeof(float));;
float* x865 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x105,150,x272,1,0,x865,1);
float* x867 = (float*)myMalloc(150 * sizeof(float));;
float* x868 = (float*)myMalloc(150 * sizeof(float));;
for(int x869=0; x869 < 150; x869++) {
float x870 = x862[x869];
float x871 = x865[x869];
float x872 = x870 + x871;
x868[x869] = x872;

}
float* x876 = (float*)myMalloc(150 * sizeof(float));;
float* x877 = (float*)myMalloc(150 * sizeof(float));;
for(int x878=0; x878 < 150; x878++) {
float x879 = x868[x878];
float x880 = x129[x878];
float x881 = x879 + x880;
x877[x878] = x881;

}
float* x885 = (float*)myMalloc(150 * sizeof(float));;
float* x886 = (float*)myMalloc(150 * sizeof(float));;
for(int x887=0; x887 < 150; x887++) {
float x888 = x877[x887];
float x889 = -1.0f * x888;
double x890 = (double)x889;
double x891 = exp(x890);
float x892 = (float)x891;
float x893 = x892 + 1.0f;
float x894 = 1.0f / x893;
x886[x887] = x894;

}
float* x898 = (float*)myMalloc(150 * sizeof(float));;
float* x899 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x113,150,x263,1,0,x899,1);
float* x901 = (float*)myMalloc(150 * sizeof(float));;
float* x902 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x121,150,x272,1,0,x902,1);
float* x904 = (float*)myMalloc(150 * sizeof(float));;
float* x905 = (float*)myMalloc(150 * sizeof(float));;
for(int x906=0; x906 < 150; x906++) {
float x907 = x899[x906];
float x908 = x902[x906];
float x909 = x907 + x908;
x905[x906] = x909;

}
float* x913 = (float*)myMalloc(150 * sizeof(float));;
float* x914 = (float*)myMalloc(150 * sizeof(float));;
for(int x915=0; x915 < 150; x915++) {
float x916 = x905[x915];
float x917 = x129[x915];
float x918 = x916 + x917;
x914[x915] = x918;

}
float* x922 = (float*)myMalloc(150 * sizeof(float));;
float* x923 = (float*)myMalloc(150 * sizeof(float));;
for(int x924=0; x924 < 150; x924++) {
float x925 = x914[x924];
float x926 = -1.0f * x925;
double x927 = (double)x926;
double x928 = exp(x927);
float x929 = (float)x928;
float x930 = x929 + 1.0f;
float x931 = 1.0f / x930;
x923[x924] = x931;

}
float* x935 = (float*)myMalloc(150 * sizeof(float));;
float* x936 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x130,150,x263,1,0,x936,1);
float* x938 = (float*)myMalloc(150 * sizeof(float));;
float* x939 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x138,150,x272,1,0,x939,1);
float* x941 = (float*)myMalloc(150 * sizeof(float));;
float* x942 = (float*)myMalloc(150 * sizeof(float));;
for(int x943=0; x943 < 150; x943++) {
float x944 = x936[x943];
float x945 = x939[x943];
float x946 = x944 + x945;
x942[x943] = x946;

}
float* x950 = (float*)myMalloc(150 * sizeof(float));;
float* x951 = (float*)myMalloc(150 * sizeof(float));;
for(int x952=0; x952 < 150; x952++) {
float x953 = x942[x952];
float x954 = x146[x952];
float x955 = x953 + x954;
x951[x952] = x955;

}
float* x959 = (float*)myMalloc(150 * sizeof(float));;
float* x960 = (float*)myMalloc(150 * sizeof(float));;
for(int x961=0; x961 < 150; x961++) {
float x962 = x951[x961];
float x963 = -1.0f * x962;
double x964 = (double)x963;
double x965 = exp(x964);
float x966 = (float)x965;
float x967 = x966 + 1.0f;
float x968 = 1.0f / x967;
x960[x961] = x968;

}
float* x972 = (float*)myMalloc(150 * sizeof(float));;
float* x973 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x147,150,x263,1,0,x973,1);
float* x975 = (float*)myMalloc(150 * sizeof(float));;
float* x976 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x155,150,x272,1,0,x976,1);
float* x978 = (float*)myMalloc(150 * sizeof(float));;
float* x979 = (float*)myMalloc(150 * sizeof(float));;
for(int x980=0; x980 < 150; x980++) {
float x981 = x973[x980];
float x982 = x976[x980];
float x983 = x981 + x982;
x979[x980] = x983;

}
float* x987 = (float*)myMalloc(150 * sizeof(float));;
float* x988 = (float*)myMalloc(150 * sizeof(float));;
for(int x989=0; x989 < 150; x989++) {
float x990 = x979[x989];
float x991 = x163[x989];
float x992 = x990 + x991;
x988[x989] = x992;

}
float* x996 = (float*)myMalloc(150 * sizeof(float));;
float* x997 = (float*)myMalloc(150 * sizeof(float));;
for(int x998=0; x998 < 150; x998++) {
float x999 = x988[x998];
double x1000 = (double)x999;
double x1001 = tanh(x1000);
float x1002 = (float)x1001;
x997[x998] = x1002;

}
float* x1006 = (float*)myMalloc(150 * sizeof(float));;
float* x1007 = (float*)myMalloc(150 * sizeof(float));;
for(int x1008=0; x1008 < 150; x1008++) {
float x1009 = x849[x1008];
float x1010 = x997[x1008];
float x1011 = x1009 * x1010;
x1007[x1008] = x1011;

}
float* x1015 = (float*)myMalloc(150 * sizeof(float));;
float* x1016 = (float*)myMalloc(150 * sizeof(float));;
for(int x1017=0; x1017 < 150; x1017++) {
float x1018 = x886[x1017];
float x1019 = x265[x1017];
float x1020 = x1018 * x1019;
x1016[x1017] = x1020;

}
float* x1024 = (float*)myMalloc(150 * sizeof(float));;
float* x1025 = (float*)myMalloc(150 * sizeof(float));;
for(int x1026=0; x1026 < 150; x1026++) {
float x1027 = x1007[x1026];
float x1028 = x1016[x1026];
float x1029 = x1027 + x1028;
x1025[x1026] = x1029;

}
float* x1033 = (float*)myMalloc(150 * sizeof(float));;
float* x1034 = (float*)myMalloc(150 * sizeof(float));;
for(int x1035=0; x1035 < 150; x1035++) {
float x1036 = x923[x1035];
float x1037 = x274[x1035];
float x1038 = x1036 * x1037;
x1034[x1035] = x1038;

}
float* x1042 = (float*)myMalloc(150 * sizeof(float));;
float* x1043 = (float*)myMalloc(150 * sizeof(float));;
for(int x1044=0; x1044 < 150; x1044++) {
float x1045 = x1025[x1044];
float x1046 = x1034[x1044];
float x1047 = x1045 + x1046;
x1043[x1044] = x1047;

}
float* x1051 = (float*)myMalloc(150 * sizeof(float));;
float* x1052 = (float*)myMalloc(150 * sizeof(float));;
for(int x1053=0; x1053 < 150; x1053++) {
float x1054 = x1043[x1053];
double x1055 = (double)x1054;
double x1056 = tanh(x1055);
float x1057 = (float)x1056;
x1052[x1053] = x1057;

}
float* x1061 = (float*)myMalloc(150 * sizeof(float));;
float* x1062 = (float*)myMalloc(150 * sizeof(float));;
for(int x1063=0; x1063 < 150; x1063++) {
float x1064 = x960[x1063];
float x1065 = x1052[x1063];
float x1066 = x1064 * x1065;
x1062[x1063] = x1066;

}
float* x1070 = (float*)myMalloc(150 * sizeof(float));;
float* x1071 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x164,150,x1062,1,0,x1071,1);
float* x1073 = (float*)myMalloc(5 * sizeof(float));;
float* x1074 = (float*)myMalloc(5 * sizeof(float));;
for(int x1075=0; x1075 < 5; x1075++) {
float x1076 = x1071[x1075];
float x1077 = x173[x1075];
float x1078 = x1076 + x1077;
x1074[x1075] = x1078;

}
float* x1082 = (float*)myMalloc(5 * sizeof(float));;
if (x1084) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1090 = (float*)myMalloc(1 * sizeof(float));;
for(int x1091=0; x1091 < 1; x1091++) {
float x1092 = x261[0];
float x1093 = x270[0];
float x1094 = x1092 + x1093;
x1090[x1091] = x1094;

}
float* x1098 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1099 = 0;
int32_t x1100 = 1;
x1100 *= 1;
x1100 *= 5;
int32_t x1103 = x1099;
bool x1104 = x1103 >= 2;
if (x1104) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1109 = x1103 == 0;
if (x1109) {
int32_t x1110 = x1100;
bool x1111 = x1110 == 5;
if (x1111) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x1118 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1119 = 0;
for(int x1120=0; x1120 < 1; x1120++) {
float x1121 = -3.4028235E38f;
for(int x1122=0; x1122 < 5; x1122++) {
int32_t x1123 = x1119;
float x1124 = x1074[x1123];
float x1125 = x1121;
bool x1126 = x1124 > x1125;
if (x1126) {
float x1127 = x1074[x1123];
x1121 = x1127;
} else {
}
x1119 += 1;

}
float x1134 = x1121;
x1118[x1120] = x1134;

}
float* x1138 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1139 = 0;
for(int x1140=0; x1140 < 1; x1140++) {
for(int x1141=0; x1141 < 5; x1141++) {
int32_t x1142 = x1139;
float x1143 = x1074[x1142];
float x1144 = x1118[x1140];
float x1145 = x1143 - x1144;
double x1146 = (double)x1145;
double x1147 = exp(x1146);
float x1148 = (float)x1147;
x1138[x1142] = x1148;
x1139 += 1;

}

}
float* x1155 = (float*)myMalloc(1 * sizeof(float));;
for(int x1156=0; x1156 < 1; x1156++) {
int32_t x1157 = x1156;
int32_t x1158 = x1156 * 5;
int32_t x1159 = x1158;
for(int x1160=0; x1160 < 5; x1160++) {
for(int x1161=0; x1161 < 1; x1161++) {
int32_t x1162 = x1157;
int32_t x1163 = x1162 + x1161;
float x1164 = x1155[x1163];
int32_t x1165 = x1159;
int32_t x1166 = x1165 + x1161;
float x1167 = x1138[x1166];
float x1168 = x1164 + x1167;
x1155[x1163] = x1168;

}
x1159 += 1;

}

}
x1139 = 0;
for(int x1178=0; x1178 < 1; x1178++) {
float x1179 = x1118[x1178];
float x1180 = x1155[x1178];
double x1181 = (double)x1180;
double x1182 = log(x1181);
float x1183 = (float)x1182;
float x1184 = x1179 + x1183;
for(int x1185=0; x1185 < 5; x1185++) {
int32_t x1186 = x1139;
float x1187 = x1074[x1186];
float x1188 = x1187 - x1184;
x1138[x1186] = x1188;
x1139 += 1;

}

}
float* x1195 = (float*)myMalloc(5 * sizeof(float));;
int* x1196 = x227+x248;
// nllLoss forward in CPU
float* x1198 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1199 = 0;
for(int x1200=0; x1200 < 1; x1200++) {
int32_t x1201 = x1199;
int32_t x1202 = x1196[x1200];
int32_t x1203 = x1201 + x1202;
float x1204 = x1138[x1203];
float x1205 = -1.0f * x1204;
x1198[x1200] = x1205;
x1199 += 5;

}
float* x1210 = (float*)myMalloc(1 * sizeof(float));;
if (x1084) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1215 = (float*)myMalloc(1 * sizeof(float));;
for(int x1216=0; x1216 < 1; x1216++) {
float x1217 = x1090[0];
float x1218 = x1198[0];
float x1219 = x1217 + x1218;
x1215[x1216] = x1219;

}
float* x1223 = (float*)myMalloc(1 * sizeof(float));;
float** x1258 = (float**)myMalloc(6 * sizeof(float*));;
x1258[0] = x1215;
x1258[1] = x1223;
x1258[2] = x1062;
x1258[3] = x1070;
x1258[4] = x1043;
x1258[5] = x1051;
x1224(x1258);
// back prop for + op
if (x1084) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1271=0; x1271 < 1; x1271++) {
float x1272 = x1098[0];
float x1273 = x1223[0];
float x1274 = x1272 + x1273;
x1098[0] = x1274;

}
if (x1084) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1282=0; x1282 < 1; x1282++) {
float x1283 = x1210[0];
float x1284 = x1223[0];
float x1285 = x1283 + x1284;
x1210[0] = x1285;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x1291 = 0;
for(int x1292=0; x1292 < 1; x1292++) {
int32_t x1293 = x1291;
int32_t x1294 = x1196[x1292];
int32_t x1295 = x1293 + x1294;
float x1296 = x1195[x1295];
float x1297 = x1210[x1292];
float x1298 = -1.0f * x1297;
float x1299 = x1296 + x1298;
x1195[x1295] = x1299;
x1291 += 5;

}
float* x1304 = (float*)myMalloc(1 * sizeof(float));;
for(int x1305=0; x1305 < 1; x1305++) {
int32_t x1306 = x1305;
int32_t x1307 = x1305 * 5;
int32_t x1308 = x1307;
for(int x1309=0; x1309 < 5; x1309++) {
for(int x1310=0; x1310 < 1; x1310++) {
int32_t x1311 = x1306;
int32_t x1312 = x1311 + x1310;
float x1313 = x1304[x1312];
int32_t x1314 = x1308;
int32_t x1315 = x1314 + x1310;
float x1316 = x1195[x1315];
float x1317 = x1313 + x1316;
x1304[x1312] = x1317;

}
x1308 += 1;

}

}
int32_t x1326 = 0;
for(int x1327=0; x1327 < 1; x1327++) {
for(int x1328=0; x1328 < 5; x1328++) {
int32_t x1329 = x1326;
float x1330 = x1082[x1329];
float x1331 = x1195[x1329];
float x1332 = x1138[x1329];
float x1336 = x1304[x1327];
double x1333 = (double)x1332;
double x1334 = exp(x1333);
float x1335 = (float)x1334;
float x1337 = x1335 * x1336;
float x1338 = x1331 - x1337;
float x1339 = x1330 + x1338;
x1082[x1329] = x1339;
x1326 += 1;

}

}
// back prop for + op
if (x1084) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1351=0; x1351 < 1; x1351++) {
float x1352 = x262[0];
float x1353 = x1098[0];
float x1354 = x1352 + x1353;
x262[0] = x1354;

}
if (x1084) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1362=0; x1362 < 1; x1362++) {
float x1363 = x271[0];
float x1364 = x1098[0];
float x1365 = x1363 + x1364;
x271[0] = x1365;

}
// back prop for + op
for(int x1370=0; x1370 < 5; x1370++) {
float x1371 = x1073[x1370];
float x1372 = x1082[x1370];
float x1373 = x1371 + x1372;
x1073[x1370] = x1373;

}
for(int x1377=0; x1377 < 5; x1377++) {
float x1378 = x195[x1377];
float x1379 = x1082[x1377];
float x1380 = x1378 + x1379;
x195[x1377] = x1380;

}
// add_cartesian
int32_t x1385 = 0;
for(int x1386=0; x1386 < 5; x1386++) {
for(int x1387=0; x1387 < 150; x1387++) {
int32_t x1388 = x1385;
int32_t x1389 = x1388 + x1387;
float x1390 = x194[x1389];
float x1391 = x1062[x1387];
float x1392 = x1073[x1386];
float x1393 = x1391 * x1392;
float x1394 = x1390 + x1393;
x194[x1389] = x1394;

}
x1385 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x164,150,x1073,1,1,x1070,1);
// backprop for * op
for(int x1403=0; x1403 < 150; x1403++) {
float x1404 = x972[x1403];
float x1405 = x960[x1403];
float x1406 = x1052[x1403];
float x1407 = x1070[x1403];
float x1408 = x1407 * x1406;
float x1409 = x1404 + x1408;
x972[x1403] = x1409;
float x1411 = x1061[x1403];
float x1412 = x960[x1403];
float x1413 = x1052[x1403];
float x1414 = x1070[x1403];
float x1415 = x1414 * x1412;
float x1416 = x1411 + x1415;
x1061[x1403] = x1416;

}
for(int x1420=0; x1420 < 150; x1420++) {
float x1421 = x1051[x1420];
float x1422 = x1052[x1420];
float x1425 = x1061[x1420];
float x1423 = x1422 * x1422;
float x1424 = 1.0f - x1423;
float x1426 = x1424 * x1425;
float x1427 = x1421 + x1426;
x1051[x1420] = x1427;

}
// back prop for + op
for(int x1432=0; x1432 < 150; x1432++) {
float x1433 = x1033[x1432];
float x1434 = x1051[x1432];
float x1435 = x1433 + x1434;
x1033[x1432] = x1435;

}
for(int x1439=0; x1439 < 150; x1439++) {
float x1440 = x1042[x1439];
float x1441 = x1051[x1439];
float x1442 = x1440 + x1441;
x1042[x1439] = x1442;

}
// backprop for * op
for(int x1447=0; x1447 < 150; x1447++) {
float x1448 = x935[x1447];
float x1449 = x923[x1447];
float x1450 = x274[x1447];
float x1451 = x1042[x1447];
float x1452 = x1451 * x1450;
float x1453 = x1448 + x1452;
x935[x1447] = x1453;
float x1455 = x275[x1447];
float x1456 = x923[x1447];
float x1457 = x274[x1447];
float x1458 = x1042[x1447];
float x1459 = x1458 * x1456;
float x1460 = x1455 + x1459;
x275[x1447] = x1460;

}
// back prop for + op
for(int x1465=0; x1465 < 150; x1465++) {
float x1466 = x1015[x1465];
float x1467 = x1033[x1465];
float x1468 = x1466 + x1467;
x1015[x1465] = x1468;

}
for(int x1472=0; x1472 < 150; x1472++) {
float x1473 = x1024[x1472];
float x1474 = x1033[x1472];
float x1475 = x1473 + x1474;
x1024[x1472] = x1475;

}
// backprop for * op
for(int x1480=0; x1480 < 150; x1480++) {
float x1481 = x898[x1480];
float x1482 = x886[x1480];
float x1483 = x265[x1480];
float x1484 = x1024[x1480];
float x1485 = x1484 * x1483;
float x1486 = x1481 + x1485;
x898[x1480] = x1486;
float x1488 = x266[x1480];
float x1489 = x886[x1480];
float x1490 = x265[x1480];
float x1491 = x1024[x1480];
float x1492 = x1491 * x1489;
float x1493 = x1488 + x1492;
x266[x1480] = x1493;

}
// backprop for * op
for(int x1498=0; x1498 < 150; x1498++) {
float x1499 = x861[x1498];
float x1500 = x849[x1498];
float x1501 = x997[x1498];
float x1502 = x1015[x1498];
float x1503 = x1502 * x1501;
float x1504 = x1499 + x1503;
x861[x1498] = x1504;
float x1506 = x1006[x1498];
float x1507 = x849[x1498];
float x1508 = x997[x1498];
float x1509 = x1015[x1498];
float x1510 = x1509 * x1507;
float x1511 = x1506 + x1510;
x1006[x1498] = x1511;

}
for(int x1515=0; x1515 < 150; x1515++) {
float x1516 = x996[x1515];
float x1517 = x997[x1515];
float x1520 = x1006[x1515];
float x1518 = x1517 * x1517;
float x1519 = 1.0f - x1518;
float x1521 = x1519 * x1520;
float x1522 = x1516 + x1521;
x996[x1515] = x1522;

}
// back prop for + op
for(int x1527=0; x1527 < 150; x1527++) {
float x1528 = x987[x1527];
float x1529 = x996[x1527];
float x1530 = x1528 + x1529;
x987[x1527] = x1530;

}
for(int x1534=0; x1534 < 150; x1534++) {
float x1535 = x193[x1534];
float x1536 = x996[x1534];
float x1537 = x1535 + x1536;
x193[x1534] = x1537;

}
// back prop for + op
for(int x1542=0; x1542 < 150; x1542++) {
float x1543 = x975[x1542];
float x1544 = x987[x1542];
float x1545 = x1543 + x1544;
x975[x1542] = x1545;

}
for(int x1549=0; x1549 < 150; x1549++) {
float x1550 = x978[x1549];
float x1551 = x987[x1549];
float x1552 = x1550 + x1551;
x978[x1549] = x1552;

}
// add_cartesian
int32_t x1557 = 0;
for(int x1558=0; x1558 < 150; x1558++) {
for(int x1559=0; x1559 < 150; x1559++) {
int32_t x1560 = x1557;
int32_t x1561 = x1560 + x1559;
float x1562 = x192[x1561];
float x1563 = x272[x1559];
float x1564 = x978[x1558];
float x1565 = x1563 * x1564;
float x1566 = x1562 + x1565;
x192[x1561] = x1566;

}
x1557 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x155,150,x978,1,1,x273,1);
// add_cartesian
int32_t x1575 = 0;
for(int x1576=0; x1576 < 150; x1576++) {
for(int x1577=0; x1577 < 150; x1577++) {
int32_t x1578 = x1575;
int32_t x1579 = x1578 + x1577;
float x1580 = x191[x1579];
float x1581 = x263[x1577];
float x1582 = x975[x1576];
float x1583 = x1581 * x1582;
float x1584 = x1580 + x1583;
x191[x1579] = x1584;

}
x1575 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x147,150,x975,1,1,x264,1);
for(int x1592=0; x1592 < 150; x1592++) {
float x1593 = x959[x1592];
float x1594 = x960[x1592];
float x1597 = x972[x1592];
float x1595 = 1.0f - x1594;
float x1596 = x1595 * x1594;
float x1598 = x1596 * x1597;
float x1599 = x1593 + x1598;
x959[x1592] = x1599;

}
// back prop for + op
for(int x1604=0; x1604 < 150; x1604++) {
float x1605 = x950[x1604];
float x1606 = x959[x1604];
float x1607 = x1605 + x1606;
x950[x1604] = x1607;

}
for(int x1611=0; x1611 < 150; x1611++) {
float x1612 = x190[x1611];
float x1613 = x959[x1611];
float x1614 = x1612 + x1613;
x190[x1611] = x1614;

}
// back prop for + op
for(int x1619=0; x1619 < 150; x1619++) {
float x1620 = x938[x1619];
float x1621 = x950[x1619];
float x1622 = x1620 + x1621;
x938[x1619] = x1622;

}
for(int x1626=0; x1626 < 150; x1626++) {
float x1627 = x941[x1626];
float x1628 = x950[x1626];
float x1629 = x1627 + x1628;
x941[x1626] = x1629;

}
// add_cartesian
int32_t x1634 = 0;
for(int x1635=0; x1635 < 150; x1635++) {
for(int x1636=0; x1636 < 150; x1636++) {
int32_t x1637 = x1634;
int32_t x1638 = x1637 + x1636;
float x1639 = x189[x1638];
float x1640 = x272[x1636];
float x1641 = x941[x1635];
float x1642 = x1640 * x1641;
float x1643 = x1639 + x1642;
x189[x1638] = x1643;

}
x1634 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x138,150,x941,1,1,x273,1);
// add_cartesian
int32_t x1652 = 0;
for(int x1653=0; x1653 < 150; x1653++) {
for(int x1654=0; x1654 < 150; x1654++) {
int32_t x1655 = x1652;
int32_t x1656 = x1655 + x1654;
float x1657 = x188[x1656];
float x1658 = x263[x1654];
float x1659 = x938[x1653];
float x1660 = x1658 * x1659;
float x1661 = x1657 + x1660;
x188[x1656] = x1661;

}
x1652 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x130,150,x938,1,1,x264,1);
for(int x1669=0; x1669 < 150; x1669++) {
float x1670 = x922[x1669];
float x1671 = x923[x1669];
float x1674 = x935[x1669];
float x1672 = 1.0f - x1671;
float x1673 = x1672 * x1671;
float x1675 = x1673 * x1674;
float x1676 = x1670 + x1675;
x922[x1669] = x1676;

}
// back prop for + op
for(int x1681=0; x1681 < 150; x1681++) {
float x1682 = x913[x1681];
float x1683 = x922[x1681];
float x1684 = x1682 + x1683;
x913[x1681] = x1684;

}
for(int x1688=0; x1688 < 150; x1688++) {
float x1689 = x187[x1688];
float x1690 = x922[x1688];
float x1691 = x1689 + x1690;
x187[x1688] = x1691;

}
// back prop for + op
for(int x1696=0; x1696 < 150; x1696++) {
float x1697 = x901[x1696];
float x1698 = x913[x1696];
float x1699 = x1697 + x1698;
x901[x1696] = x1699;

}
for(int x1703=0; x1703 < 150; x1703++) {
float x1704 = x904[x1703];
float x1705 = x913[x1703];
float x1706 = x1704 + x1705;
x904[x1703] = x1706;

}
// add_cartesian
int32_t x1711 = 0;
for(int x1712=0; x1712 < 150; x1712++) {
for(int x1713=0; x1713 < 150; x1713++) {
int32_t x1714 = x1711;
int32_t x1715 = x1714 + x1713;
float x1716 = x186[x1715];
float x1717 = x272[x1713];
float x1718 = x904[x1712];
float x1719 = x1717 * x1718;
float x1720 = x1716 + x1719;
x186[x1715] = x1720;

}
x1711 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x121,150,x904,1,1,x273,1);
// add_cartesian
int32_t x1729 = 0;
for(int x1730=0; x1730 < 150; x1730++) {
for(int x1731=0; x1731 < 150; x1731++) {
int32_t x1732 = x1729;
int32_t x1733 = x1732 + x1731;
float x1734 = x185[x1733];
float x1735 = x263[x1731];
float x1736 = x901[x1730];
float x1737 = x1735 * x1736;
float x1738 = x1734 + x1737;
x185[x1733] = x1738;

}
x1729 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x113,150,x901,1,1,x264,1);
for(int x1746=0; x1746 < 150; x1746++) {
float x1747 = x885[x1746];
float x1748 = x886[x1746];
float x1751 = x898[x1746];
float x1749 = 1.0f - x1748;
float x1750 = x1749 * x1748;
float x1752 = x1750 * x1751;
float x1753 = x1747 + x1752;
x885[x1746] = x1753;

}
// back prop for + op
for(int x1758=0; x1758 < 150; x1758++) {
float x1759 = x876[x1758];
float x1760 = x885[x1758];
float x1761 = x1759 + x1760;
x876[x1758] = x1761;

}
for(int x1765=0; x1765 < 150; x1765++) {
float x1766 = x187[x1765];
float x1767 = x885[x1765];
float x1768 = x1766 + x1767;
x187[x1765] = x1768;

}
// back prop for + op
for(int x1773=0; x1773 < 150; x1773++) {
float x1774 = x864[x1773];
float x1775 = x876[x1773];
float x1776 = x1774 + x1775;
x864[x1773] = x1776;

}
for(int x1780=0; x1780 < 150; x1780++) {
float x1781 = x867[x1780];
float x1782 = x876[x1780];
float x1783 = x1781 + x1782;
x867[x1780] = x1783;

}
// add_cartesian
int32_t x1788 = 0;
for(int x1789=0; x1789 < 150; x1789++) {
for(int x1790=0; x1790 < 150; x1790++) {
int32_t x1791 = x1788;
int32_t x1792 = x1791 + x1790;
float x1793 = x184[x1792];
float x1794 = x272[x1790];
float x1795 = x867[x1789];
float x1796 = x1794 * x1795;
float x1797 = x1793 + x1796;
x184[x1792] = x1797;

}
x1788 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x105,150,x867,1,1,x273,1);
// add_cartesian
int32_t x1806 = 0;
for(int x1807=0; x1807 < 150; x1807++) {
for(int x1808=0; x1808 < 150; x1808++) {
int32_t x1809 = x1806;
int32_t x1810 = x1809 + x1808;
float x1811 = x183[x1810];
float x1812 = x263[x1808];
float x1813 = x864[x1807];
float x1814 = x1812 * x1813;
float x1815 = x1811 + x1814;
x183[x1810] = x1815;

}
x1806 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x97,150,x864,1,1,x264,1);
for(int x1823=0; x1823 < 150; x1823++) {
float x1824 = x848[x1823];
float x1825 = x849[x1823];
float x1828 = x861[x1823];
float x1826 = 1.0f - x1825;
float x1827 = x1826 * x1825;
float x1829 = x1827 * x1828;
float x1830 = x1824 + x1829;
x848[x1823] = x1830;

}
// back prop for + op
for(int x1835=0; x1835 < 150; x1835++) {
float x1836 = x839[x1835];
float x1837 = x848[x1835];
float x1838 = x1836 + x1837;
x839[x1835] = x1838;

}
for(int x1842=0; x1842 < 150; x1842++) {
float x1843 = x182[x1842];
float x1844 = x848[x1842];
float x1845 = x1843 + x1844;
x182[x1842] = x1845;

}
// back prop for + op
for(int x1850=0; x1850 < 150; x1850++) {
float x1851 = x827[x1850];
float x1852 = x839[x1850];
float x1853 = x1851 + x1852;
x827[x1850] = x1853;

}
for(int x1857=0; x1857 < 150; x1857++) {
float x1858 = x830[x1857];
float x1859 = x839[x1857];
float x1860 = x1858 + x1859;
x830[x1857] = x1860;

}
// add_cartesian
int32_t x1865 = 0;
for(int x1866=0; x1866 < 150; x1866++) {
for(int x1867=0; x1867 < 150; x1867++) {
int32_t x1868 = x1865;
int32_t x1869 = x1868 + x1867;
float x1870 = x181[x1869];
float x1871 = x272[x1867];
float x1872 = x830[x1866];
float x1873 = x1871 * x1872;
float x1874 = x1870 + x1873;
x181[x1869] = x1874;

}
x1865 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x88,150,x830,1,1,x273,1);
// add_cartesian
int32_t x1883 = 0;
for(int x1884=0; x1884 < 150; x1884++) {
for(int x1885=0; x1885 < 150; x1885++) {
int32_t x1886 = x1883;
int32_t x1887 = x1886 + x1885;
float x1888 = x180[x1887];
float x1889 = x263[x1885];
float x1890 = x827[x1884];
float x1891 = x1889 * x1890;
float x1892 = x1888 + x1891;
x180[x1887] = x1892;

}
x1883 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x79,150,x827,1,1,x264,1);
}
};
x244(x267,x268,x1903);
};
x244(x258,x259,x1913);
} else {
float** x1940 = (float**)myMalloc(6 * sizeof(float*));;
x1940[0] = x238;
x1940[1] = x239;
x1940[2] = x240;
x1940[3] = x241;
x1940[4] = x242;
x1940[5] = x243;
function<void(float**)> x249 = x246;
function<void(float**)> x1923 = [&](float** x1924) {
float* x1925 = x1924[0];
float* x1926 = x1924[1];
float* x1927 = x1924[2];
float* x1928 = x1924[3];
float* x1929 = x1924[4];
float* x1930 = x1924[5];
float** x1931 = (float**)myMalloc(6 * sizeof(float*));;
x1931[0] = x1925;
x1931[1] = x1926;
x1931[2] = x1927;
x1931[3] = x1928;
x1931[4] = x1929;
x1931[5] = x1930;
x249(x1931);
};
x1923(x1940);
}
};
float* x234 = (float*)myMalloc(1 * sizeof(float));;
float* x235 = (float*)myMalloc(1 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
float* x237 = (float*)myMalloc(1 * sizeof(float));;
float** x1971 = (float**)myMalloc(6 * sizeof(float*));;
x1971[0] = x238;
x1971[1] = x239;
x1971[2] = x240;
x1971[3] = x241;
x1971[4] = x242;
x1971[5] = x243;
function<void(float**)> x1951 = [&](float** x1952) {
float* x1953 = x1952[0];
float* x1954 = x1952[1];
float* x1955 = x1952[2];
float* x1956 = x1952[3];
float* x1957 = x1952[4];
float* x1958 = x1952[5];
// make sure the size of loss is 1
for(int x1960=0; x1960 < 1; x1960++) {
x1954[x1960] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@4ed2a773
for(int x1965=0; x1965 < 1; x1965++) {
float x1966 = x1953[x1965];
x237[x1965] = x1966;

}
};
x244(0,x1951,x1971);
float x1980 = x237[0];
float x1981 = x223;
float x1982 = (float)x224;
float x1983 = x1981 * x1982;
int32_t x1984 = x224 + 1;
float x1985 = (float)x1984;
float x1986 = x1983 / x1985;
float x1987 = x1980 / x1985;
float x1988 = x1986 + x1987;
x223 = x1988;
for(int x1990=0; x1990 < 45000; x1990++) {
float x1991 = x174[x1990];
float x1992 = x1991;
float x1993 = x196[x1990];
float x1994 = x1992;
float x1995 = x1994 * x1994;
float x1996 = x1993 + x1995;
x196[x1990] = x1996;
float x1998 = x50[x1990];
float x2000 = x196[x1990];
float x1999 = 0.05f * x1994;
double x2001 = (double)x2000;
double x2002 = x2001 + 9.99999993922529E-9;
double x2003 = sqrt(x2002);
float x2004 = (float)x2003;
float x2005 = x1999 / x2004;
float x2006 = x1998 - x2005;
x50[x1990] = x2006;
x174[x1990] = 0.0f;

}
for(int x2011=0; x2011 < 150; x2011++) {
float x2012 = x175[x2011];
float x2013 = x2012;
float x2014 = x197[x2011];
float x2015 = x2013;
float x2016 = x2015 * x2015;
float x2017 = x2014 + x2016;
x197[x2011] = x2017;
float x2019 = x60[x2011];
float x2021 = x197[x2011];
float x2020 = 0.05f * x2015;
double x2022 = (double)x2021;
double x2023 = x2022 + 9.99999993922529E-9;
double x2024 = sqrt(x2023);
float x2025 = (float)x2024;
float x2026 = x2020 / x2025;
float x2027 = x2019 - x2026;
x60[x2011] = x2027;
x175[x2011] = 0.0f;

}
for(int x2032=0; x2032 < 45000; x2032++) {
float x2033 = x176[x2032];
float x2034 = x2033;
float x2035 = x198[x2032];
float x2036 = x2034;
float x2037 = x2036 * x2036;
float x2038 = x2035 + x2037;
x198[x2032] = x2038;
float x2040 = x61[x2032];
float x2042 = x198[x2032];
float x2041 = 0.05f * x2036;
double x2043 = (double)x2042;
double x2044 = x2043 + 9.99999993922529E-9;
double x2045 = sqrt(x2044);
float x2046 = (float)x2045;
float x2047 = x2041 / x2046;
float x2048 = x2040 - x2047;
x61[x2032] = x2048;
x176[x2032] = 0.0f;

}
for(int x2053=0; x2053 < 150; x2053++) {
float x2054 = x177[x2053];
float x2055 = x2054;
float x2056 = x199[x2053];
float x2057 = x2055;
float x2058 = x2057 * x2057;
float x2059 = x2056 + x2058;
x199[x2053] = x2059;
float x2061 = x69[x2053];
float x2063 = x199[x2053];
float x2062 = 0.05f * x2057;
double x2064 = (double)x2063;
double x2065 = x2064 + 9.99999993922529E-9;
double x2066 = sqrt(x2065);
float x2067 = (float)x2066;
float x2068 = x2062 / x2067;
float x2069 = x2061 - x2068;
x69[x2053] = x2069;
x177[x2053] = 0.0f;

}
for(int x2074=0; x2074 < 45000; x2074++) {
float x2075 = x178[x2074];
float x2076 = x2075;
float x2077 = x200[x2074];
float x2078 = x2076;
float x2079 = x2078 * x2078;
float x2080 = x2077 + x2079;
x200[x2074] = x2080;
float x2082 = x70[x2074];
float x2084 = x200[x2074];
float x2083 = 0.05f * x2078;
double x2085 = (double)x2084;
double x2086 = x2085 + 9.99999993922529E-9;
double x2087 = sqrt(x2086);
float x2088 = (float)x2087;
float x2089 = x2083 / x2088;
float x2090 = x2082 - x2089;
x70[x2074] = x2090;
x178[x2074] = 0.0f;

}
for(int x2095=0; x2095 < 150; x2095++) {
float x2096 = x179[x2095];
float x2097 = x2096;
float x2098 = x201[x2095];
float x2099 = x2097;
float x2100 = x2099 * x2099;
float x2101 = x2098 + x2100;
x201[x2095] = x2101;
float x2103 = x78[x2095];
float x2105 = x201[x2095];
float x2104 = 0.05f * x2099;
double x2106 = (double)x2105;
double x2107 = x2106 + 9.99999993922529E-9;
double x2108 = sqrt(x2107);
float x2109 = (float)x2108;
float x2110 = x2104 / x2109;
float x2111 = x2103 - x2110;
x78[x2095] = x2111;
x179[x2095] = 0.0f;

}
for(int x2116=0; x2116 < 22500; x2116++) {
float x2117 = x180[x2116];
float x2118 = x2117;
float x2119 = x202[x2116];
float x2120 = x2118;
float x2121 = x2120 * x2120;
float x2122 = x2119 + x2121;
x202[x2116] = x2122;
float x2124 = x79[x2116];
float x2126 = x202[x2116];
float x2125 = 0.05f * x2120;
double x2127 = (double)x2126;
double x2128 = x2127 + 9.99999993922529E-9;
double x2129 = sqrt(x2128);
float x2130 = (float)x2129;
float x2131 = x2125 / x2130;
float x2132 = x2124 - x2131;
x79[x2116] = x2132;
x180[x2116] = 0.0f;

}
for(int x2137=0; x2137 < 22500; x2137++) {
float x2138 = x181[x2137];
float x2139 = x2138;
float x2140 = x203[x2137];
float x2141 = x2139;
float x2142 = x2141 * x2141;
float x2143 = x2140 + x2142;
x203[x2137] = x2143;
float x2145 = x88[x2137];
float x2147 = x203[x2137];
float x2146 = 0.05f * x2141;
double x2148 = (double)x2147;
double x2149 = x2148 + 9.99999993922529E-9;
double x2150 = sqrt(x2149);
float x2151 = (float)x2150;
float x2152 = x2146 / x2151;
float x2153 = x2145 - x2152;
x88[x2137] = x2153;
x181[x2137] = 0.0f;

}
for(int x2158=0; x2158 < 150; x2158++) {
float x2159 = x182[x2158];
float x2160 = x2159;
float x2161 = x204[x2158];
float x2162 = x2160;
float x2163 = x2162 * x2162;
float x2164 = x2161 + x2163;
x204[x2158] = x2164;
float x2166 = x96[x2158];
float x2168 = x204[x2158];
float x2167 = 0.05f * x2162;
double x2169 = (double)x2168;
double x2170 = x2169 + 9.99999993922529E-9;
double x2171 = sqrt(x2170);
float x2172 = (float)x2171;
float x2173 = x2167 / x2172;
float x2174 = x2166 - x2173;
x96[x2158] = x2174;
x182[x2158] = 0.0f;

}
for(int x2179=0; x2179 < 22500; x2179++) {
float x2180 = x183[x2179];
float x2181 = x2180;
float x2182 = x205[x2179];
float x2183 = x2181;
float x2184 = x2183 * x2183;
float x2185 = x2182 + x2184;
x205[x2179] = x2185;
float x2187 = x97[x2179];
float x2189 = x205[x2179];
float x2188 = 0.05f * x2183;
double x2190 = (double)x2189;
double x2191 = x2190 + 9.99999993922529E-9;
double x2192 = sqrt(x2191);
float x2193 = (float)x2192;
float x2194 = x2188 / x2193;
float x2195 = x2187 - x2194;
x97[x2179] = x2195;
x183[x2179] = 0.0f;

}
for(int x2200=0; x2200 < 22500; x2200++) {
float x2201 = x184[x2200];
float x2202 = x2201;
float x2203 = x206[x2200];
float x2204 = x2202;
float x2205 = x2204 * x2204;
float x2206 = x2203 + x2205;
x206[x2200] = x2206;
float x2208 = x105[x2200];
float x2210 = x206[x2200];
float x2209 = 0.05f * x2204;
double x2211 = (double)x2210;
double x2212 = x2211 + 9.99999993922529E-9;
double x2213 = sqrt(x2212);
float x2214 = (float)x2213;
float x2215 = x2209 / x2214;
float x2216 = x2208 - x2215;
x105[x2200] = x2216;
x184[x2200] = 0.0f;

}
for(int x2221=0; x2221 < 22500; x2221++) {
float x2222 = x185[x2221];
float x2223 = x2222;
float x2224 = x207[x2221];
float x2225 = x2223;
float x2226 = x2225 * x2225;
float x2227 = x2224 + x2226;
x207[x2221] = x2227;
float x2229 = x113[x2221];
float x2231 = x207[x2221];
float x2230 = 0.05f * x2225;
double x2232 = (double)x2231;
double x2233 = x2232 + 9.99999993922529E-9;
double x2234 = sqrt(x2233);
float x2235 = (float)x2234;
float x2236 = x2230 / x2235;
float x2237 = x2229 - x2236;
x113[x2221] = x2237;
x185[x2221] = 0.0f;

}
for(int x2242=0; x2242 < 22500; x2242++) {
float x2243 = x186[x2242];
float x2244 = x2243;
float x2245 = x208[x2242];
float x2246 = x2244;
float x2247 = x2246 * x2246;
float x2248 = x2245 + x2247;
x208[x2242] = x2248;
float x2250 = x121[x2242];
float x2252 = x208[x2242];
float x2251 = 0.05f * x2246;
double x2253 = (double)x2252;
double x2254 = x2253 + 9.99999993922529E-9;
double x2255 = sqrt(x2254);
float x2256 = (float)x2255;
float x2257 = x2251 / x2256;
float x2258 = x2250 - x2257;
x121[x2242] = x2258;
x186[x2242] = 0.0f;

}
for(int x2263=0; x2263 < 150; x2263++) {
float x2264 = x187[x2263];
float x2265 = x2264;
float x2266 = x209[x2263];
float x2267 = x2265;
float x2268 = x2267 * x2267;
float x2269 = x2266 + x2268;
x209[x2263] = x2269;
float x2271 = x129[x2263];
float x2273 = x209[x2263];
float x2272 = 0.05f * x2267;
double x2274 = (double)x2273;
double x2275 = x2274 + 9.99999993922529E-9;
double x2276 = sqrt(x2275);
float x2277 = (float)x2276;
float x2278 = x2272 / x2277;
float x2279 = x2271 - x2278;
x129[x2263] = x2279;
x187[x2263] = 0.0f;

}
for(int x2284=0; x2284 < 22500; x2284++) {
float x2285 = x188[x2284];
float x2286 = x2285;
float x2287 = x210[x2284];
float x2288 = x2286;
float x2289 = x2288 * x2288;
float x2290 = x2287 + x2289;
x210[x2284] = x2290;
float x2292 = x130[x2284];
float x2294 = x210[x2284];
float x2293 = 0.05f * x2288;
double x2295 = (double)x2294;
double x2296 = x2295 + 9.99999993922529E-9;
double x2297 = sqrt(x2296);
float x2298 = (float)x2297;
float x2299 = x2293 / x2298;
float x2300 = x2292 - x2299;
x130[x2284] = x2300;
x188[x2284] = 0.0f;

}
for(int x2305=0; x2305 < 22500; x2305++) {
float x2306 = x189[x2305];
float x2307 = x2306;
float x2308 = x211[x2305];
float x2309 = x2307;
float x2310 = x2309 * x2309;
float x2311 = x2308 + x2310;
x211[x2305] = x2311;
float x2313 = x138[x2305];
float x2315 = x211[x2305];
float x2314 = 0.05f * x2309;
double x2316 = (double)x2315;
double x2317 = x2316 + 9.99999993922529E-9;
double x2318 = sqrt(x2317);
float x2319 = (float)x2318;
float x2320 = x2314 / x2319;
float x2321 = x2313 - x2320;
x138[x2305] = x2321;
x189[x2305] = 0.0f;

}
for(int x2326=0; x2326 < 150; x2326++) {
float x2327 = x190[x2326];
float x2328 = x2327;
float x2329 = x212[x2326];
float x2330 = x2328;
float x2331 = x2330 * x2330;
float x2332 = x2329 + x2331;
x212[x2326] = x2332;
float x2334 = x146[x2326];
float x2336 = x212[x2326];
float x2335 = 0.05f * x2330;
double x2337 = (double)x2336;
double x2338 = x2337 + 9.99999993922529E-9;
double x2339 = sqrt(x2338);
float x2340 = (float)x2339;
float x2341 = x2335 / x2340;
float x2342 = x2334 - x2341;
x146[x2326] = x2342;
x190[x2326] = 0.0f;

}
for(int x2347=0; x2347 < 22500; x2347++) {
float x2348 = x191[x2347];
float x2349 = x2348;
float x2350 = x213[x2347];
float x2351 = x2349;
float x2352 = x2351 * x2351;
float x2353 = x2350 + x2352;
x213[x2347] = x2353;
float x2355 = x147[x2347];
float x2357 = x213[x2347];
float x2356 = 0.05f * x2351;
double x2358 = (double)x2357;
double x2359 = x2358 + 9.99999993922529E-9;
double x2360 = sqrt(x2359);
float x2361 = (float)x2360;
float x2362 = x2356 / x2361;
float x2363 = x2355 - x2362;
x147[x2347] = x2363;
x191[x2347] = 0.0f;

}
for(int x2368=0; x2368 < 22500; x2368++) {
float x2369 = x192[x2368];
float x2370 = x2369;
float x2371 = x214[x2368];
float x2372 = x2370;
float x2373 = x2372 * x2372;
float x2374 = x2371 + x2373;
x214[x2368] = x2374;
float x2376 = x155[x2368];
float x2378 = x214[x2368];
float x2377 = 0.05f * x2372;
double x2379 = (double)x2378;
double x2380 = x2379 + 9.99999993922529E-9;
double x2381 = sqrt(x2380);
float x2382 = (float)x2381;
float x2383 = x2377 / x2382;
float x2384 = x2376 - x2383;
x155[x2368] = x2384;
x192[x2368] = 0.0f;

}
for(int x2389=0; x2389 < 150; x2389++) {
float x2390 = x193[x2389];
float x2391 = x2390;
float x2392 = x215[x2389];
float x2393 = x2391;
float x2394 = x2393 * x2393;
float x2395 = x2392 + x2394;
x215[x2389] = x2395;
float x2397 = x163[x2389];
float x2399 = x215[x2389];
float x2398 = 0.05f * x2393;
double x2400 = (double)x2399;
double x2401 = x2400 + 9.99999993922529E-9;
double x2402 = sqrt(x2401);
float x2403 = (float)x2402;
float x2404 = x2398 / x2403;
float x2405 = x2397 - x2404;
x163[x2389] = x2405;
x193[x2389] = 0.0f;

}
for(int x2410=0; x2410 < 750; x2410++) {
float x2411 = x194[x2410];
float x2412 = x2411;
float x2413 = x216[x2410];
float x2414 = x2412;
float x2415 = x2414 * x2414;
float x2416 = x2413 + x2415;
x216[x2410] = x2416;
float x2418 = x164[x2410];
float x2420 = x216[x2410];
float x2419 = 0.05f * x2414;
double x2421 = (double)x2420;
double x2422 = x2421 + 9.99999993922529E-9;
double x2423 = sqrt(x2422);
float x2424 = (float)x2423;
float x2425 = x2419 / x2424;
float x2426 = x2418 - x2425;
x164[x2410] = x2426;
x194[x2410] = 0.0f;

}
for(int x2431=0; x2431 < 5; x2431++) {
float x2432 = x195[x2431];
float x2433 = x2432;
float x2434 = x217[x2431];
float x2435 = x2433;
float x2436 = x2435 * x2435;
float x2437 = x2434 + x2436;
x217[x2431] = x2437;
float x2439 = x173[x2431];
float x2441 = x217[x2431];
float x2440 = 0.05f * x2435;
double x2442 = (double)x2441;
double x2443 = x2442 + 9.99999993922529E-9;
double x2444 = sqrt(x2443);
float x2445 = (float)x2444;
float x2446 = x2440 / x2445;
float x2447 = x2439 - x2446;
x173[x2431] = x2447;
x195[x2431] = 0.0f;

}
int64_t x2452 = (long)mallocAddr;
int64_t x2453 = x2452 - x219;
memset((void*)x219, 0, x2453);
mallocAddr = (void*)x219;

}
float x2458 = x223;
double x2459 = (double)x2458;
x218[x222] = x2459;
double x2461 = ((double)clock() / CLOCKS_PER_SEC);
double x2462 = x2461 - x220;
printf("epoc %d, average_loss %f, time %lf\n",x222,x2458,x2462);

}
double x2466 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2470 = (long)fopen(x0, "w");
fprintf((FILE *)x2470, "unit: %s\n", "1 epoch");
for(int x2472=0; x2472 < 6; x2472++) {
double x2473 = x218[x2472];
fprintf((FILE *)x2470, "%lf\n", x2473);

}
double x2467 = x220 - x2;
double x2468 = x2466 - x220;
double x2469 = x2468 / 6.0;
fprintf((FILE *)x2470, "run time: %lf %lf\n", x2467, x2469);
fclose((FILE*)x2470);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

