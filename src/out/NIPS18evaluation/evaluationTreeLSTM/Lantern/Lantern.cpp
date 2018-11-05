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

long HEAP_SIZE = 4294967304; // 1073741826; // 1048576; // 536870912; // 268435456; // 2097152; 1610612739; //
void *mallocBase = calloc(HEAP_SIZE, 1);
void *mallocAddr = mallocBase;
void *waterMark = mallocBase;
void *myMalloc(size_t bytes) {
  void *res = mallocAddr;
  mallocAddr = (void *)((char *)mallocAddr + bytes);
  if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE)
    fprintf(stderr, "CPU memory breached limit of HEAP_SIZE\n");
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
float* x59 = (float*)myMalloc(45000 * sizeof(float));;
float* x60 = (float*)myMalloc(150 * sizeof(float));;
float* x61 = (float*)myMalloc(150 * sizeof(float));;
float* x62 = (float*)myMalloc(45000 * sizeof(float));;
for(int x63=0; x63 < 45000; x63++) {
float x64 = (float)rand()/RAND_MAX;
float x65 = x64 - 0.5f;
float x66 = x65 * 0.01f;
x62[x63] = x66;

}
float* x70 = (float*)myMalloc(45000 * sizeof(float));;
float* x71 = (float*)myMalloc(150 * sizeof(float));;
float* x72 = (float*)myMalloc(150 * sizeof(float));;
float* x73 = (float*)myMalloc(45000 * sizeof(float));;
for(int x74=0; x74 < 45000; x74++) {
float x75 = (float)rand()/RAND_MAX;
float x76 = x75 - 0.5f;
float x77 = x76 * 0.01f;
x73[x74] = x77;

}
float* x81 = (float*)myMalloc(45000 * sizeof(float));;
float* x82 = (float*)myMalloc(150 * sizeof(float));;
float* x83 = (float*)myMalloc(150 * sizeof(float));;
float* x84 = (float*)myMalloc(22500 * sizeof(float));;
for(int x86=0; x86 < 22500; x86++) {
float x87 = (float)rand()/RAND_MAX;
float x88 = x87 - 0.5f;
float x89 = x88 * 0.01f;
x84[x86] = x89;

}
float* x93 = (float*)myMalloc(22500 * sizeof(float));;
float* x94 = (float*)myMalloc(22500 * sizeof(float));;
for(int x95=0; x95 < 22500; x95++) {
float x96 = (float)rand()/RAND_MAX;
float x97 = x96 - 0.5f;
float x98 = x97 * 0.01f;
x94[x95] = x98;

}
float* x102 = (float*)myMalloc(22500 * sizeof(float));;
float* x103 = (float*)myMalloc(150 * sizeof(float));;
float* x104 = (float*)myMalloc(150 * sizeof(float));;
float* x105 = (float*)myMalloc(22500 * sizeof(float));;
for(int x106=0; x106 < 22500; x106++) {
float x107 = (float)rand()/RAND_MAX;
float x108 = x107 - 0.5f;
float x109 = x108 * 0.01f;
x105[x106] = x109;

}
float* x113 = (float*)myMalloc(22500 * sizeof(float));;
float* x114 = (float*)myMalloc(22500 * sizeof(float));;
for(int x115=0; x115 < 22500; x115++) {
float x116 = (float)rand()/RAND_MAX;
float x117 = x116 - 0.5f;
float x118 = x117 * 0.01f;
x114[x115] = x118;

}
float* x122 = (float*)myMalloc(22500 * sizeof(float));;
float* x123 = (float*)myMalloc(22500 * sizeof(float));;
for(int x124=0; x124 < 22500; x124++) {
float x125 = (float)rand()/RAND_MAX;
float x126 = x125 - 0.5f;
float x127 = x126 * 0.01f;
x123[x124] = x127;

}
float* x131 = (float*)myMalloc(22500 * sizeof(float));;
float* x132 = (float*)myMalloc(22500 * sizeof(float));;
for(int x133=0; x133 < 22500; x133++) {
float x134 = (float)rand()/RAND_MAX;
float x135 = x134 - 0.5f;
float x136 = x135 * 0.01f;
x132[x133] = x136;

}
float* x140 = (float*)myMalloc(22500 * sizeof(float));;
float* x141 = (float*)myMalloc(150 * sizeof(float));;
float* x142 = (float*)myMalloc(150 * sizeof(float));;
float* x143 = (float*)myMalloc(22500 * sizeof(float));;
for(int x144=0; x144 < 22500; x144++) {
float x145 = (float)rand()/RAND_MAX;
float x146 = x145 - 0.5f;
float x147 = x146 * 0.01f;
x143[x144] = x147;

}
float* x151 = (float*)myMalloc(22500 * sizeof(float));;
float* x152 = (float*)myMalloc(22500 * sizeof(float));;
for(int x153=0; x153 < 22500; x153++) {
float x154 = (float)rand()/RAND_MAX;
float x155 = x154 - 0.5f;
float x156 = x155 * 0.01f;
x152[x153] = x156;

}
float* x160 = (float*)myMalloc(22500 * sizeof(float));;
float* x161 = (float*)myMalloc(150 * sizeof(float));;
float* x162 = (float*)myMalloc(150 * sizeof(float));;
float* x163 = (float*)myMalloc(22500 * sizeof(float));;
for(int x164=0; x164 < 22500; x164++) {
float x165 = (float)rand()/RAND_MAX;
float x166 = x165 - 0.5f;
float x167 = x166 * 0.01f;
x163[x164] = x167;

}
float* x171 = (float*)myMalloc(22500 * sizeof(float));;
float* x172 = (float*)myMalloc(22500 * sizeof(float));;
for(int x173=0; x173 < 22500; x173++) {
float x174 = (float)rand()/RAND_MAX;
float x175 = x174 - 0.5f;
float x176 = x175 * 0.01f;
x172[x173] = x176;

}
float* x180 = (float*)myMalloc(22500 * sizeof(float));;
float* x181 = (float*)myMalloc(150 * sizeof(float));;
float* x182 = (float*)myMalloc(150 * sizeof(float));;
float* x183 = (float*)myMalloc(750 * sizeof(float));;
for(int x185=0; x185 < 750; x185++) {
float x186 = (float)rand()/RAND_MAX;
float x187 = x186 - 0.5f;
float x188 = x187 * 0.01f;
x183[x185] = x188;

}
float* x192 = (float*)myMalloc(750 * sizeof(float));;
float* x193 = (float*)myMalloc(5 * sizeof(float));;
float* x194 = (float*)myMalloc(5 * sizeof(float));;
float* x195 = (float*)myMalloc(1 * sizeof(float));;
float* x196 = (float*)myMalloc(1 * sizeof(float));;
float* x197 = (float*)myMalloc(150 * sizeof(float));;
float* x198 = (float*)myMalloc(150 * sizeof(float));;
float* x199 = (float*)myMalloc(150 * sizeof(float));;
float* x200 = (float*)myMalloc(150 * sizeof(float));;
float* x201 = (float*)myMalloc(150 * sizeof(float));;
float* x202 = (float*)myMalloc(22500 * sizeof(float));;
float* x203 = (float*)myMalloc(22500 * sizeof(float));;
float* x204 = (float*)myMalloc(22500 * sizeof(float));;
float* x205 = (float*)myMalloc(22500 * sizeof(float));;
float* x206 = (float*)myMalloc(22500 * sizeof(float));;
float* x207 = (float*)myMalloc(22500 * sizeof(float));;
float* x208 = (float*)myMalloc(22500 * sizeof(float));;
float* x209 = (float*)myMalloc(22500 * sizeof(float));;
float* x210 = (float*)myMalloc(22500 * sizeof(float));;
float* x211 = (float*)myMalloc(150 * sizeof(float));;
float* x212 = (float*)myMalloc(22500 * sizeof(float));;
float* x213 = (float*)myMalloc(150 * sizeof(float));;
float* x214 = (float*)myMalloc(150 * sizeof(float));;
float* x215 = (float*)myMalloc(150 * sizeof(float));;
float* x216 = (float*)myMalloc(45000 * sizeof(float));;
float* x217 = (float*)myMalloc(150 * sizeof(float));;
float* x218 = (float*)myMalloc(45000 * sizeof(float));;
float* x219 = (float*)myMalloc(150 * sizeof(float));;
float* x220 = (float*)myMalloc(45000 * sizeof(float));;
float* x221 = (float*)myMalloc(5 * sizeof(float));;
float* x222 = (float*)myMalloc(750 * sizeof(float));;
double* x223 = (double*)myMalloc(6 * sizeof(double));;
int64_t x224 = (long)mallocAddr;
double x225 = ((double)clock() / CLOCKS_PER_SEC);
for(int x227=0; x227 < 6; x227++) {
float x228 = 0.0f;
for(int x229=0; x229 < x24; x229++) {
int32_t x230 = x229 % x24;
int32_t x231 = x230 * 4;
int* x232 = x26[x231];
int32_t x233 = x231 + 1;
int* x234 = x26[x233];
int32_t x235 = x231 + 2;
int* x236 = x26[x235];
int32_t x237 = x231 + 3;
int* x238 = x26[x237];
function<void(int32_t,function<void(float**)>,float**)> x243 = [&](int32_t x244,function<void(float**)> x245,float** x246) {
float** x249 = x246;
float* x250 = x249[0];
float* x251 = x249[1];
float* x252 = x249[2];
float* x253 = x249[3];
float* x254 = x249[4];
float* x255 = x249[5];
int32_t x247 = x244;
bool x256 = x247 >= 0;
if (x256) {
int32_t x257 = x236[x247];
float** x1847 = (float**)myMalloc(6 * sizeof(float*));;
x1847[0] = x195;
x1847[1] = x196;
x1847[2] = x197;
x1847[3] = x198;
x1847[4] = x199;
x1847[5] = x200;
function<void(float**)> x248 = x245;
function<void(float**)> x486 = [&](float** x487) {
float* x488 = x487[0];
float* x489 = x487[1];
float* x490 = x487[2];
float* x491 = x487[3];
float* x492 = x487[4];
float* x493 = x487[5];
float** x494 = (float**)myMalloc(6 * sizeof(float*));;
x494[0] = x488;
x494[1] = x489;
x494[2] = x490;
x494[3] = x491;
x494[4] = x492;
x494[5] = x493;
x248(x494);
};
function<void(float**)> x478 = [&](float** x479) {
float* x480 = x479[0];
float* x481 = x479[1];
float* x482 = x479[2];
float* x483 = x479[3];
float* x484 = x479[4];
float* x485 = x479[5];
float** x503 = (float**)myMalloc(6 * sizeof(float*));;
x503[0] = x480;
x503[1] = x481;
x503[2] = x482;
x503[3] = x483;
x503[4] = x484;
x503[5] = x485;
x486(x503);
};
function<void(float**)> x258 = [&](float** x259) {
float* x260 = x259[0];
float* x261 = x259[1];
float* x262 = x259[2];
float* x263 = x259[3];
float* x264 = x259[4];
float* x265 = x259[5];
int32_t x266 = x238[x247];
float** x1837 = (float**)myMalloc(6 * sizeof(float*));;
x1837[0] = x195;
x1837[1] = x196;
x1837[2] = x197;
x1837[3] = x198;
x1837[4] = x199;
x1837[5] = x200;
function<void(float**)> x267 = [&](float** x268) {
float* x269 = x268[0];
float* x270 = x268[1];
float* x271 = x268[2];
float* x272 = x268[3];
float* x273 = x268[4];
float* x274 = x268[5];
int32_t x275 = x236[x247];
bool x276 = x275 < 0;
if (x276) {
int32_t x277 = x234[x247];
float* x278 = x7[x277];
float* x279 = (float*)myMalloc(300 * sizeof(float));;
float* x280 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x50,300,x278,1,0,x280,1);
float* x282 = (float*)myMalloc(150 * sizeof(float));;
int32_t x283 = 0;
int32_t x284 = 0;
int32_t x285 = 0;
for(int x287=0; x287 < 150; x287++) {
int32_t x288 = x284;
float x289 = x280[x288];
int32_t x290 = x285;
float x291 = x60[x290];
float x292 = x289 + x291;
x280[x288] = x292;
x283 += 1;
x284 += 1;
x285 += 1;

}
float* x299 = (float*)myMalloc(150 * sizeof(float));;
for(int x300=0; x300 < 150; x300++) {
float x301 = x280[x300];
float x302 = -1.0f * x301;
double x303 = (double)x302;
double x304 = exp(x303);
float x305 = (float)x304;
float x306 = x305 + 1.0f;
float x307 = 1.0f / x306;
x299[x300] = x307;

}
float* x311 = (float*)myMalloc(150 * sizeof(float));;
float* x312 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x62,300,x278,1,0,x312,1);
float* x314 = (float*)myMalloc(150 * sizeof(float));;
int32_t x315 = 0;
int32_t x316 = 0;
int32_t x317 = 0;
for(int x318=0; x318 < 150; x318++) {
int32_t x319 = x316;
float x320 = x312[x319];
int32_t x321 = x317;
float x322 = x71[x321];
float x323 = x320 + x322;
x312[x319] = x323;
x315 += 1;
x316 += 1;
x317 += 1;

}
float* x330 = (float*)myMalloc(150 * sizeof(float));;
for(int x331=0; x331 < 150; x331++) {
float x332 = x312[x331];
float x333 = -1.0f * x332;
double x334 = (double)x333;
double x335 = exp(x334);
float x336 = (float)x335;
float x337 = x336 + 1.0f;
float x338 = 1.0f / x337;
x330[x331] = x338;

}
float* x342 = (float*)myMalloc(150 * sizeof(float));;
float* x343 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x73,300,x278,1,0,x343,1);
float* x345 = (float*)myMalloc(150 * sizeof(float));;
int32_t x346 = 0;
int32_t x347 = 0;
int32_t x348 = 0;
for(int x349=0; x349 < 150; x349++) {
int32_t x350 = x347;
float x351 = x343[x350];
int32_t x352 = x348;
float x353 = x82[x352];
float x354 = x351 + x353;
x343[x350] = x354;
x346 += 1;
x347 += 1;
x348 += 1;

}
float* x361 = (float*)myMalloc(150 * sizeof(float));;
for(int x362=0; x362 < 150; x362++) {
float x363 = x343[x362];
double x364 = (double)x363;
double x365 = tanh(x364);
float x366 = (float)x365;
x361[x362] = x366;

}
float* x370 = (float*)myMalloc(150 * sizeof(float));;
float* x371 = (float*)myMalloc(150 * sizeof(float));;
int32_t x372 = 0;
int32_t x373 = 0;
int32_t x374 = 0;
for(int x375=0; x375 < 150; x375++) {
int32_t x376 = x372;
int32_t x377 = x373;
float x378 = x299[x377];
int32_t x379 = x374;
float x380 = x361[x379];
float x381 = x378 * x380;
x371[x376] = x381;
x372 += 1;
x373 += 1;
x374 += 1;

}
float* x388 = (float*)myMalloc(150 * sizeof(float));;
float* x389 = (float*)myMalloc(150 * sizeof(float));;
for(int x390=0; x390 < 150; x390++) {
float x391 = x371[x390];
double x392 = (double)x391;
double x393 = tanh(x392);
float x394 = (float)x393;
x389[x390] = x394;

}
float* x398 = (float*)myMalloc(150 * sizeof(float));;
float* x399 = (float*)myMalloc(150 * sizeof(float));;
int32_t x400 = 0;
int32_t x401 = 0;
int32_t x402 = 0;
for(int x403=0; x403 < 150; x403++) {
int32_t x404 = x400;
int32_t x405 = x401;
float x406 = x330[x405];
int32_t x407 = x402;
float x408 = x389[x407];
float x409 = x406 * x408;
x399[x404] = x409;
x400 += 1;
x401 += 1;
x402 += 1;

}
float* x416 = (float*)myMalloc(150 * sizeof(float));;
int32_t x417 = x232[x247];
float* x418 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x183,150,x371,1,0,x418,1);
float* x420 = (float*)myMalloc(5 * sizeof(float));;
int32_t x421 = 0;
int32_t x422 = 0;
int32_t x423 = 0;
for(int x425=0; x425 < 5; x425++) {
int32_t x426 = x422;
float x427 = x418[x426];
int32_t x428 = x423;
float x429 = x193[x428];
float x430 = x427 + x429;
x418[x426] = x430;
x421 += 1;
x422 += 1;
x423 += 1;

}
float x437 = -3.4028235E38f;
for(int x438=0; x438 < 5; x438++) {
float x439 = x437;
float x440 = x418[x438];
bool x441 = x440 > x439;
float x442;
if (x441) {
x442 = x440;
} else {
x442 = x439;
}
x437 = x442;

}
float x446 = x437;
float x447 = 0.0f;
for(int x448=0; x448 < 5; x448++) {
float x449 = x447;
float x450 = x418[x448];
float x451 = x437;
float x452 = x450 - x451;
double x453 = (double)x452;
double x454 = exp(x453);
float x455 = (float)x454;
float x456 = x449 + x455;
x447 = x456;

}
float x460 = x447;
float* x465 = (float*)myMalloc(5 * sizeof(float));;
double x461 = (double)x460;
double x462 = log(x461);
float x463 = (float)x462;
float x464 = x446 + x463;
for(int x466=0; x466 < 5; x466++) {
float x467 = x418[x466];
float x468 = x467 - x464;
x465[x466] = x468;

}
float* x472 = (float*)myMalloc(5 * sizeof(float));;
float x473 = x465[x417];
float* x475 = (float*)myMalloc(1 * sizeof(float));;
float x474 = -1.0f * x473;
x475[0] = x474;
float* x477 = (float*)myMalloc(1 * sizeof(float));;
float** x512 = (float**)myMalloc(6 * sizeof(float*));;
x512[0] = x475;
x512[1] = x477;
x512[2] = x371;
x512[3] = x388;
x512[4] = x399;
x512[5] = x416;
x478(x512);
float x520 = x472[x417];
float x521 = x477[0];
float x522 = -1.0f * x521;
float x523 = x520 + x522;
x472[x417] = x523;
float x525 = 0.0f;
for(int x526=0; x526 < 5; x526++) {
float x527 = x525;
float x528 = x472[x526];
float x529 = x527 + x528;
x525 = x529;

}
float x533 = x525;
float* x534 = (float*)myMalloc(1 * sizeof(float));;
x534[0] = x533;
float x536 = x534[0];
for(int x537=0; x537 < 5; x537++) {
float x538 = x420[x537];
float x539 = x472[x537];
float x540 = x465[x537];
double x541 = (double)x540;
double x542 = exp(x541);
float x543 = (float)x542;
float x544 = x543 * x536;
float x545 = x539 - x544;
float x546 = x538 + x545;
x420[x537] = x546;

}
int32_t x550 = 0;
int32_t x551 = 0;
int32_t x552 = 0;
for(int x553=0; x553 < 5; x553++) {
int32_t x554 = x551;
float x555 = x194[x554];
int32_t x556 = x552;
float x557 = x420[x556];
float x558 = x555 + x557;
x194[x554] = x558;
x550 += 1;
x551 += 1;
x552 += 1;

}
// add_cartesian
int32_t x566 = 0;
for(int x567=0; x567 < 5; x567++) {
for(int x568=0; x568 < 150; x568++) {
int32_t x569 = x566;
int32_t x570 = x569 + x568;
float x571 = x192[x570];
float x572 = x371[x568];
float x573 = x420[x567];
float x574 = x572 * x573;
float x575 = x571 + x574;
x192[x570] = x575;

}
x566 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x183,150,x420,1,1,x388,1);
int32_t x583 = 0;
int32_t x584 = 0;
int32_t x585 = 0;
for(int x586=0; x586 < 150; x586++) {
int32_t x587 = x583;
float x588 = x342[x587];
float x589 = x330[x587];
int32_t x590 = x584;
float x591 = x389[x590];
int32_t x592 = x585;
float x593 = x416[x592];
float x594 = x593 * x591;
float x595 = x588 + x594;
x342[x587] = x595;
float x597 = x398[x590];
float x598 = x330[x587];
float x599 = x389[x590];
float x600 = x416[x592];
float x601 = x600 * x598;
float x602 = x597 + x601;
x398[x590] = x602;
x585 += 1;
x583 += 1;
x584 += 1;

}
for(int x609=0; x609 < 150; x609++) {
float x610 = x388[x609];
float x611 = x389[x609];
float x614 = x398[x609];
float x612 = x611 * x611;
float x613 = 1.0f - x612;
float x615 = x613 * x614;
float x616 = x610 + x615;
x388[x609] = x616;

}
int32_t x620 = 0;
int32_t x621 = 0;
int32_t x622 = 0;
for(int x623=0; x623 < 150; x623++) {
int32_t x624 = x620;
float x625 = x311[x624];
float x626 = x299[x624];
int32_t x627 = x621;
float x628 = x361[x627];
int32_t x629 = x622;
float x630 = x388[x629];
float x631 = x630 * x628;
float x632 = x625 + x631;
x311[x624] = x632;
float x634 = x370[x627];
float x635 = x299[x624];
float x636 = x361[x627];
float x637 = x388[x629];
float x638 = x637 * x635;
float x639 = x634 + x638;
x370[x627] = x639;
x622 += 1;
x620 += 1;
x621 += 1;

}
for(int x646=0; x646 < 150; x646++) {
float x647 = x345[x646];
float x648 = x361[x646];
float x651 = x370[x646];
float x649 = x648 * x648;
float x650 = 1.0f - x649;
float x652 = x650 * x651;
float x653 = x647 + x652;
x345[x646] = x653;

}
int32_t x657 = 0;
int32_t x658 = 0;
int32_t x659 = 0;
for(int x660=0; x660 < 150; x660++) {
int32_t x661 = x658;
float x662 = x83[x661];
int32_t x663 = x659;
float x664 = x345[x663];
float x665 = x662 + x664;
x83[x661] = x665;
x657 += 1;
x658 += 1;
x659 += 1;

}
// add_cartesian
int32_t x673 = 0;
for(int x674=0; x674 < 150; x674++) {
for(int x675=0; x675 < 300; x675++) {
int32_t x676 = x673;
int32_t x677 = x676 + x675;
float x678 = x81[x677];
float x679 = x278[x675];
float x680 = x345[x674];
float x681 = x679 * x680;
float x682 = x678 + x681;
x81[x677] = x682;

}
x673 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x73,300,x345,1,1,x279,1);
for(int x690=0; x690 < 150; x690++) {
float x691 = x314[x690];
float x692 = x330[x690];
float x695 = x342[x690];
float x693 = 1.0f - x692;
float x694 = x693 * x692;
float x696 = x694 * x695;
float x697 = x691 + x696;
x314[x690] = x697;

}
int32_t x701 = 0;
int32_t x702 = 0;
int32_t x703 = 0;
for(int x704=0; x704 < 150; x704++) {
int32_t x705 = x702;
float x706 = x72[x705];
int32_t x707 = x703;
float x708 = x314[x707];
float x709 = x706 + x708;
x72[x705] = x709;
x701 += 1;
x702 += 1;
x703 += 1;

}
// add_cartesian
int32_t x717 = 0;
for(int x718=0; x718 < 150; x718++) {
for(int x719=0; x719 < 300; x719++) {
int32_t x720 = x717;
int32_t x721 = x720 + x719;
float x722 = x70[x721];
float x723 = x278[x719];
float x724 = x314[x718];
float x725 = x723 * x724;
float x726 = x722 + x725;
x70[x721] = x726;

}
x717 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x62,300,x314,1,1,x279,1);
for(int x734=0; x734 < 150; x734++) {
float x735 = x282[x734];
float x736 = x299[x734];
float x739 = x311[x734];
float x737 = 1.0f - x736;
float x738 = x737 * x736;
float x740 = x738 * x739;
float x741 = x735 + x740;
x282[x734] = x741;

}
int32_t x745 = 0;
int32_t x746 = 0;
int32_t x747 = 0;
for(int x748=0; x748 < 150; x748++) {
int32_t x749 = x746;
float x750 = x61[x749];
int32_t x751 = x747;
float x752 = x282[x751];
float x753 = x750 + x752;
x61[x749] = x753;
x745 += 1;
x746 += 1;
x747 += 1;

}
// add_cartesian
int32_t x761 = 0;
for(int x762=0; x762 < 150; x762++) {
for(int x763=0; x763 < 300; x763++) {
int32_t x764 = x761;
int32_t x765 = x764 + x763;
float x766 = x59[x765];
float x767 = x278[x763];
float x768 = x282[x762];
float x769 = x767 * x768;
float x770 = x766 + x769;
x59[x765] = x770;

}
x761 += 300;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x50,300,x282,1,1,x279,1);
} else {
float* x779 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x84,150,x262,1,0,x779,1);
float* x781 = (float*)myMalloc(150 * sizeof(float));;
float* x782 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x94,150,x271,1,0,x782,1);
float* x784 = (float*)myMalloc(150 * sizeof(float));;
int32_t x785 = 0;
int32_t x786 = 0;
int32_t x787 = 0;
for(int x788=0; x788 < 150; x788++) {
int32_t x789 = x786;
float x790 = x779[x789];
int32_t x791 = x787;
float x792 = x782[x791];
float x793 = x790 + x792;
x779[x789] = x793;
x785 += 1;
x786 += 1;
x787 += 1;

}
int32_t x800 = 0;
int32_t x801 = 0;
int32_t x802 = 0;
for(int x803=0; x803 < 150; x803++) {
int32_t x804 = x801;
float x805 = x779[x804];
int32_t x806 = x802;
float x807 = x103[x806];
float x808 = x805 + x807;
x779[x804] = x808;
x800 += 1;
x801 += 1;
x802 += 1;

}
float* x815 = (float*)myMalloc(150 * sizeof(float));;
for(int x816=0; x816 < 150; x816++) {
float x817 = x779[x816];
float x818 = -1.0f * x817;
double x819 = (double)x818;
double x820 = exp(x819);
float x821 = (float)x820;
float x822 = x821 + 1.0f;
float x823 = 1.0f / x822;
x815[x816] = x823;

}
float* x827 = (float*)myMalloc(150 * sizeof(float));;
float* x828 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x105,150,x262,1,0,x828,1);
float* x830 = (float*)myMalloc(150 * sizeof(float));;
float* x831 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x114,150,x271,1,0,x831,1);
float* x833 = (float*)myMalloc(150 * sizeof(float));;
int32_t x834 = 0;
int32_t x835 = 0;
int32_t x836 = 0;
for(int x837=0; x837 < 150; x837++) {
int32_t x838 = x835;
float x839 = x828[x838];
int32_t x840 = x836;
float x841 = x831[x840];
float x842 = x839 + x841;
x828[x838] = x842;
x834 += 1;
x835 += 1;
x836 += 1;

}
int32_t x849 = 0;
int32_t x850 = 0;
int32_t x851 = 0;
for(int x852=0; x852 < 150; x852++) {
int32_t x853 = x850;
float x854 = x828[x853];
int32_t x855 = x851;
float x856 = x141[x855];
float x857 = x854 + x856;
x828[x853] = x857;
x849 += 1;
x850 += 1;
x851 += 1;

}
float* x864 = (float*)myMalloc(150 * sizeof(float));;
for(int x865=0; x865 < 150; x865++) {
float x866 = x828[x865];
float x867 = -1.0f * x866;
double x868 = (double)x867;
double x869 = exp(x868);
float x870 = (float)x869;
float x871 = x870 + 1.0f;
float x872 = 1.0f / x871;
x864[x865] = x872;

}
float* x876 = (float*)myMalloc(150 * sizeof(float));;
float* x877 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x123,150,x262,1,0,x877,1);
float* x879 = (float*)myMalloc(150 * sizeof(float));;
float* x880 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x132,150,x271,1,0,x880,1);
float* x882 = (float*)myMalloc(150 * sizeof(float));;
int32_t x883 = 0;
int32_t x884 = 0;
int32_t x885 = 0;
for(int x886=0; x886 < 150; x886++) {
int32_t x887 = x884;
float x888 = x877[x887];
int32_t x889 = x885;
float x890 = x880[x889];
float x891 = x888 + x890;
x877[x887] = x891;
x883 += 1;
x884 += 1;
x885 += 1;

}
int32_t x898 = 0;
int32_t x899 = 0;
int32_t x900 = 0;
for(int x901=0; x901 < 150; x901++) {
int32_t x902 = x899;
float x903 = x877[x902];
int32_t x904 = x900;
float x905 = x141[x904];
float x906 = x903 + x905;
x877[x902] = x906;
x898 += 1;
x899 += 1;
x900 += 1;

}
float* x913 = (float*)myMalloc(150 * sizeof(float));;
for(int x914=0; x914 < 150; x914++) {
float x915 = x877[x914];
float x916 = -1.0f * x915;
double x917 = (double)x916;
double x918 = exp(x917);
float x919 = (float)x918;
float x920 = x919 + 1.0f;
float x921 = 1.0f / x920;
x913[x914] = x921;

}
float* x925 = (float*)myMalloc(150 * sizeof(float));;
float* x926 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x143,150,x262,1,0,x926,1);
float* x928 = (float*)myMalloc(150 * sizeof(float));;
float* x929 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x152,150,x271,1,0,x929,1);
float* x931 = (float*)myMalloc(150 * sizeof(float));;
int32_t x932 = 0;
int32_t x933 = 0;
int32_t x934 = 0;
for(int x935=0; x935 < 150; x935++) {
int32_t x936 = x933;
float x937 = x926[x936];
int32_t x938 = x934;
float x939 = x929[x938];
float x940 = x937 + x939;
x926[x936] = x940;
x932 += 1;
x933 += 1;
x934 += 1;

}
int32_t x947 = 0;
int32_t x948 = 0;
int32_t x949 = 0;
for(int x950=0; x950 < 150; x950++) {
int32_t x951 = x948;
float x952 = x926[x951];
int32_t x953 = x949;
float x954 = x161[x953];
float x955 = x952 + x954;
x926[x951] = x955;
x947 += 1;
x948 += 1;
x949 += 1;

}
float* x962 = (float*)myMalloc(150 * sizeof(float));;
for(int x963=0; x963 < 150; x963++) {
float x964 = x926[x963];
float x965 = -1.0f * x964;
double x966 = (double)x965;
double x967 = exp(x966);
float x968 = (float)x967;
float x969 = x968 + 1.0f;
float x970 = 1.0f / x969;
x962[x963] = x970;

}
float* x974 = (float*)myMalloc(150 * sizeof(float));;
float* x975 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x163,150,x262,1,0,x975,1);
float* x977 = (float*)myMalloc(150 * sizeof(float));;
float* x978 = (float*)myMalloc(150 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x172,150,x271,1,0,x978,1);
float* x980 = (float*)myMalloc(150 * sizeof(float));;
int32_t x981 = 0;
int32_t x982 = 0;
int32_t x983 = 0;
for(int x984=0; x984 < 150; x984++) {
int32_t x985 = x982;
float x986 = x975[x985];
int32_t x987 = x983;
float x988 = x978[x987];
float x989 = x986 + x988;
x975[x985] = x989;
x981 += 1;
x982 += 1;
x983 += 1;

}
int32_t x996 = 0;
int32_t x997 = 0;
int32_t x998 = 0;
for(int x999=0; x999 < 150; x999++) {
int32_t x1000 = x997;
float x1001 = x975[x1000];
int32_t x1002 = x998;
float x1003 = x181[x1002];
float x1004 = x1001 + x1003;
x975[x1000] = x1004;
x996 += 1;
x997 += 1;
x998 += 1;

}
float* x1011 = (float*)myMalloc(150 * sizeof(float));;
for(int x1012=0; x1012 < 150; x1012++) {
float x1013 = x975[x1012];
double x1014 = (double)x1013;
double x1015 = tanh(x1014);
float x1016 = (float)x1015;
x1011[x1012] = x1016;

}
float* x1020 = (float*)myMalloc(150 * sizeof(float));;
float* x1021 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1022 = 0;
int32_t x1023 = 0;
int32_t x1024 = 0;
for(int x1025=0; x1025 < 150; x1025++) {
int32_t x1026 = x1022;
int32_t x1027 = x1023;
float x1028 = x815[x1027];
int32_t x1029 = x1024;
float x1030 = x1011[x1029];
float x1031 = x1028 * x1030;
x1021[x1026] = x1031;
x1022 += 1;
x1023 += 1;
x1024 += 1;

}
float* x1038 = (float*)myMalloc(150 * sizeof(float));;
float* x1039 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1040 = 0;
int32_t x1041 = 0;
int32_t x1042 = 0;
for(int x1043=0; x1043 < 150; x1043++) {
int32_t x1044 = x1040;
int32_t x1045 = x1041;
float x1046 = x864[x1045];
int32_t x1047 = x1042;
float x1048 = x264[x1047];
float x1049 = x1046 * x1048;
x1039[x1044] = x1049;
x1040 += 1;
x1041 += 1;
x1042 += 1;

}
float* x1056 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1057 = 0;
int32_t x1058 = 0;
int32_t x1059 = 0;
for(int x1060=0; x1060 < 150; x1060++) {
int32_t x1061 = x1058;
float x1062 = x1021[x1061];
int32_t x1063 = x1059;
float x1064 = x1039[x1063];
float x1065 = x1062 + x1064;
x1021[x1061] = x1065;
x1057 += 1;
x1058 += 1;
x1059 += 1;

}
float* x1072 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1073 = 0;
int32_t x1074 = 0;
int32_t x1075 = 0;
for(int x1076=0; x1076 < 150; x1076++) {
int32_t x1077 = x1073;
int32_t x1078 = x1074;
float x1079 = x913[x1078];
int32_t x1080 = x1075;
float x1081 = x273[x1080];
float x1082 = x1079 * x1081;
x1072[x1077] = x1082;
x1073 += 1;
x1074 += 1;
x1075 += 1;

}
float* x1089 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1090 = 0;
int32_t x1091 = 0;
int32_t x1092 = 0;
for(int x1093=0; x1093 < 150; x1093++) {
int32_t x1094 = x1091;
float x1095 = x1021[x1094];
int32_t x1096 = x1092;
float x1097 = x1072[x1096];
float x1098 = x1095 + x1097;
x1021[x1094] = x1098;
x1090 += 1;
x1091 += 1;
x1092 += 1;

}
float* x1105 = (float*)myMalloc(150 * sizeof(float));;
for(int x1106=0; x1106 < 150; x1106++) {
float x1107 = x1021[x1106];
double x1108 = (double)x1107;
double x1109 = tanh(x1108);
float x1110 = (float)x1109;
x1105[x1106] = x1110;

}
float* x1114 = (float*)myMalloc(150 * sizeof(float));;
float* x1115 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1116 = 0;
int32_t x1117 = 0;
int32_t x1118 = 0;
for(int x1119=0; x1119 < 150; x1119++) {
int32_t x1120 = x1116;
int32_t x1121 = x1117;
float x1122 = x962[x1121];
int32_t x1123 = x1118;
float x1124 = x1105[x1123];
float x1125 = x1122 * x1124;
x1115[x1120] = x1125;
x1116 += 1;
x1117 += 1;
x1118 += 1;

}
float* x1132 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1133 = x232[x247];
float* x1134 = (float*)myMalloc(5 * sizeof(float));;
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x183,150,x1021,1,0,x1134,1);
float* x1136 = (float*)myMalloc(5 * sizeof(float));;
int32_t x1137 = 0;
int32_t x1138 = 0;
int32_t x1139 = 0;
for(int x1140=0; x1140 < 5; x1140++) {
int32_t x1141 = x1138;
float x1142 = x1134[x1141];
int32_t x1143 = x1139;
float x1144 = x193[x1143];
float x1145 = x1142 + x1144;
x1134[x1141] = x1145;
x1137 += 1;
x1138 += 1;
x1139 += 1;

}
float x1152 = -3.4028235E38f;
for(int x1153=0; x1153 < 5; x1153++) {
float x1154 = x1152;
float x1155 = x1134[x1153];
bool x1156 = x1155 > x1154;
float x1157;
if (x1156) {
x1157 = x1155;
} else {
x1157 = x1154;
}
x1152 = x1157;

}
float x1161 = x1152;
float x1162 = 0.0f;
for(int x1163=0; x1163 < 5; x1163++) {
float x1164 = x1162;
float x1165 = x1134[x1163];
float x1166 = x1152;
float x1167 = x1165 - x1166;
double x1168 = (double)x1167;
double x1169 = exp(x1168);
float x1170 = (float)x1169;
float x1171 = x1164 + x1170;
x1162 = x1171;

}
float x1175 = x1162;
float* x1180 = (float*)myMalloc(5 * sizeof(float));;
double x1176 = (double)x1175;
double x1177 = log(x1176);
float x1178 = (float)x1177;
float x1179 = x1161 + x1178;
for(int x1181=0; x1181 < 5; x1181++) {
float x1182 = x1134[x1181];
float x1183 = x1182 - x1179;
x1180[x1181] = x1183;

}
float* x1187 = (float*)myMalloc(5 * sizeof(float));;
float x1188 = x1180[x1133];
float* x1190 = (float*)myMalloc(1 * sizeof(float));;
float x1189 = -1.0f * x1188;
x1190[0] = x1189;
float* x1192 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1193 = 0;
int32_t x1194 = 0;
int32_t x1195 = 0;
int32_t x1196 = x1194;
float x1197 = x1190[x1196];
int32_t x1198 = x1195;
float x1199 = x260[x1198];
float x1200 = x1197 + x1199;
x1190[x1196] = x1200;
x1193 += 1;
int32_t x1203 = 0;
int32_t x1204 = 0;
int32_t x1205 = 0;
int32_t x1206 = x1204;
float x1207 = x1190[x1206];
int32_t x1208 = x1205;
float x1209 = x269[x1208];
float x1210 = x1207 + x1209;
x1190[x1206] = x1210;
x1203 += 1;
float** x1213 = (float**)myMalloc(6 * sizeof(float*));;
x1213[0] = x1190;
x1213[1] = x1192;
x1213[2] = x1021;
x1213[3] = x1038;
x1213[4] = x1115;
x1213[5] = x1132;
x478(x1213);
int32_t x1221 = 0;
int32_t x1222 = 0;
int32_t x1223 = 0;
int32_t x1224 = x1222;
float x1225 = x270[x1224];
int32_t x1226 = x1223;
float x1227 = x1192[x1226];
float x1228 = x1225 + x1227;
x270[x1224] = x1228;
x1221 += 1;
int32_t x1231 = 0;
int32_t x1232 = 0;
int32_t x1233 = 0;
int32_t x1234 = x1232;
float x1235 = x261[x1234];
int32_t x1236 = x1233;
float x1237 = x1192[x1236];
float x1238 = x1235 + x1237;
x261[x1234] = x1238;
x1231 += 1;
float x1241 = x1187[x1133];
float x1242 = x1192[0];
float x1243 = -1.0f * x1242;
float x1244 = x1241 + x1243;
x1187[x1133] = x1244;
float x1246 = 0.0f;
for(int x1247=0; x1247 < 5; x1247++) {
float x1248 = x1246;
float x1249 = x1187[x1247];
float x1250 = x1248 + x1249;
x1246 = x1250;

}
float x1254 = x1246;
float* x1255 = (float*)myMalloc(1 * sizeof(float));;
x1255[0] = x1254;
float x1257 = x1255[0];
for(int x1258=0; x1258 < 5; x1258++) {
float x1259 = x1136[x1258];
float x1260 = x1187[x1258];
float x1261 = x1180[x1258];
double x1262 = (double)x1261;
double x1263 = exp(x1262);
float x1264 = (float)x1263;
float x1265 = x1264 * x1257;
float x1266 = x1260 - x1265;
float x1267 = x1259 + x1266;
x1136[x1258] = x1267;

}
int32_t x1271 = 0;
int32_t x1272 = 0;
int32_t x1273 = 0;
for(int x1274=0; x1274 < 5; x1274++) {
int32_t x1275 = x1272;
float x1276 = x194[x1275];
int32_t x1277 = x1273;
float x1278 = x1136[x1277];
float x1279 = x1276 + x1278;
x194[x1275] = x1279;
x1271 += 1;
x1272 += 1;
x1273 += 1;

}
// add_cartesian
int32_t x1287 = 0;
for(int x1288=0; x1288 < 5; x1288++) {
for(int x1289=0; x1289 < 150; x1289++) {
int32_t x1290 = x1287;
int32_t x1291 = x1290 + x1289;
float x1292 = x192[x1291];
float x1293 = x1021[x1289];
float x1294 = x1136[x1288];
float x1295 = x1293 * x1294;
float x1296 = x1292 + x1295;
x192[x1291] = x1296;

}
x1287 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x183,150,x1136,1,1,x1038,1);
int32_t x1304 = 0;
int32_t x1305 = 0;
int32_t x1306 = 0;
for(int x1307=0; x1307 < 150; x1307++) {
int32_t x1308 = x1304;
float x1309 = x974[x1308];
float x1310 = x962[x1308];
int32_t x1311 = x1305;
float x1312 = x1105[x1311];
int32_t x1313 = x1306;
float x1314 = x1132[x1313];
float x1315 = x1314 * x1312;
float x1316 = x1309 + x1315;
x974[x1308] = x1316;
float x1318 = x1114[x1311];
float x1319 = x962[x1308];
float x1320 = x1105[x1311];
float x1321 = x1132[x1313];
float x1322 = x1321 * x1319;
float x1323 = x1318 + x1322;
x1114[x1311] = x1323;
x1306 += 1;
x1304 += 1;
x1305 += 1;

}
for(int x1330=0; x1330 < 150; x1330++) {
float x1331 = x1038[x1330];
float x1332 = x1105[x1330];
float x1335 = x1114[x1330];
float x1333 = x1332 * x1332;
float x1334 = 1.0f - x1333;
float x1336 = x1334 * x1335;
float x1337 = x1331 + x1336;
x1038[x1330] = x1337;

}
int32_t x1341 = 0;
int32_t x1342 = 0;
int32_t x1343 = 0;
for(int x1344=0; x1344 < 150; x1344++) {
int32_t x1345 = x1342;
float x1346 = x1089[x1345];
int32_t x1347 = x1343;
float x1348 = x1038[x1347];
float x1349 = x1346 + x1348;
x1089[x1345] = x1349;
x1341 += 1;
x1342 += 1;
x1343 += 1;

}
int32_t x1356 = 0;
int32_t x1357 = 0;
int32_t x1358 = 0;
for(int x1359=0; x1359 < 150; x1359++) {
int32_t x1360 = x1356;
float x1361 = x925[x1360];
float x1362 = x913[x1360];
int32_t x1363 = x1357;
float x1364 = x273[x1363];
int32_t x1365 = x1358;
float x1366 = x1089[x1365];
float x1367 = x1366 * x1364;
float x1368 = x1361 + x1367;
x925[x1360] = x1368;
float x1370 = x274[x1363];
float x1371 = x913[x1360];
float x1372 = x273[x1363];
float x1373 = x1089[x1365];
float x1374 = x1373 * x1371;
float x1375 = x1370 + x1374;
x274[x1363] = x1375;
x1358 += 1;
x1356 += 1;
x1357 += 1;

}
int32_t x1382 = 0;
int32_t x1383 = 0;
int32_t x1384 = 0;
for(int x1385=0; x1385 < 150; x1385++) {
int32_t x1386 = x1383;
float x1387 = x1056[x1386];
int32_t x1388 = x1384;
float x1389 = x1038[x1388];
float x1390 = x1387 + x1389;
x1056[x1386] = x1390;
x1382 += 1;
x1383 += 1;
x1384 += 1;

}
int32_t x1397 = 0;
int32_t x1398 = 0;
int32_t x1399 = 0;
for(int x1400=0; x1400 < 150; x1400++) {
int32_t x1401 = x1397;
float x1402 = x876[x1401];
float x1403 = x864[x1401];
int32_t x1404 = x1398;
float x1405 = x264[x1404];
int32_t x1406 = x1399;
float x1407 = x1056[x1406];
float x1408 = x1407 * x1405;
float x1409 = x1402 + x1408;
x876[x1401] = x1409;
float x1411 = x265[x1404];
float x1412 = x864[x1401];
float x1413 = x264[x1404];
float x1414 = x1056[x1406];
float x1415 = x1414 * x1412;
float x1416 = x1411 + x1415;
x265[x1404] = x1416;
x1399 += 1;
x1397 += 1;
x1398 += 1;

}
int32_t x1423 = 0;
int32_t x1424 = 0;
int32_t x1425 = 0;
for(int x1426=0; x1426 < 150; x1426++) {
int32_t x1427 = x1423;
float x1428 = x827[x1427];
float x1429 = x815[x1427];
int32_t x1430 = x1424;
float x1431 = x1011[x1430];
int32_t x1432 = x1425;
float x1433 = x1038[x1432];
float x1434 = x1433 * x1431;
float x1435 = x1428 + x1434;
x827[x1427] = x1435;
float x1437 = x1020[x1430];
float x1438 = x815[x1427];
float x1439 = x1011[x1430];
float x1440 = x1038[x1432];
float x1441 = x1440 * x1438;
float x1442 = x1437 + x1441;
x1020[x1430] = x1442;
x1425 += 1;
x1423 += 1;
x1424 += 1;

}
for(int x1449=0; x1449 < 150; x1449++) {
float x1450 = x977[x1449];
float x1451 = x1011[x1449];
float x1454 = x1020[x1449];
float x1452 = x1451 * x1451;
float x1453 = 1.0f - x1452;
float x1455 = x1453 * x1454;
float x1456 = x1450 + x1455;
x977[x1449] = x1456;

}
int32_t x1460 = 0;
int32_t x1461 = 0;
int32_t x1462 = 0;
for(int x1463=0; x1463 < 150; x1463++) {
int32_t x1464 = x1461;
float x1465 = x182[x1464];
int32_t x1466 = x1462;
float x1467 = x977[x1466];
float x1468 = x1465 + x1467;
x182[x1464] = x1468;
x1460 += 1;
x1461 += 1;
x1462 += 1;

}
int32_t x1475 = 0;
int32_t x1476 = 0;
int32_t x1477 = 0;
for(int x1478=0; x1478 < 150; x1478++) {
int32_t x1479 = x1476;
float x1480 = x980[x1479];
int32_t x1481 = x1477;
float x1482 = x977[x1481];
float x1483 = x1480 + x1482;
x980[x1479] = x1483;
x1475 += 1;
x1476 += 1;
x1477 += 1;

}
// add_cartesian
int32_t x1491 = 0;
for(int x1492=0; x1492 < 150; x1492++) {
for(int x1493=0; x1493 < 150; x1493++) {
int32_t x1494 = x1491;
int32_t x1495 = x1494 + x1493;
float x1496 = x180[x1495];
float x1497 = x271[x1493];
float x1498 = x980[x1492];
float x1499 = x1497 * x1498;
float x1500 = x1496 + x1499;
x180[x1495] = x1500;

}
x1491 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x172,150,x980,1,1,x272,1);
// add_cartesian
int32_t x1509 = 0;
for(int x1510=0; x1510 < 150; x1510++) {
for(int x1511=0; x1511 < 150; x1511++) {
int32_t x1512 = x1509;
int32_t x1513 = x1512 + x1511;
float x1514 = x171[x1513];
float x1515 = x262[x1511];
float x1516 = x977[x1510];
float x1517 = x1515 * x1516;
float x1518 = x1514 + x1517;
x171[x1513] = x1518;

}
x1509 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x163,150,x977,1,1,x263,1);
for(int x1526=0; x1526 < 150; x1526++) {
float x1527 = x928[x1526];
float x1528 = x962[x1526];
float x1531 = x974[x1526];
float x1529 = 1.0f - x1528;
float x1530 = x1529 * x1528;
float x1532 = x1530 * x1531;
float x1533 = x1527 + x1532;
x928[x1526] = x1533;

}
int32_t x1537 = 0;
int32_t x1538 = 0;
int32_t x1539 = 0;
for(int x1540=0; x1540 < 150; x1540++) {
int32_t x1541 = x1538;
float x1542 = x162[x1541];
int32_t x1543 = x1539;
float x1544 = x928[x1543];
float x1545 = x1542 + x1544;
x162[x1541] = x1545;
x1537 += 1;
x1538 += 1;
x1539 += 1;

}
int32_t x1552 = 0;
int32_t x1553 = 0;
int32_t x1554 = 0;
for(int x1555=0; x1555 < 150; x1555++) {
int32_t x1556 = x1553;
float x1557 = x931[x1556];
int32_t x1558 = x1554;
float x1559 = x928[x1558];
float x1560 = x1557 + x1559;
x931[x1556] = x1560;
x1552 += 1;
x1553 += 1;
x1554 += 1;

}
// add_cartesian
int32_t x1568 = 0;
for(int x1569=0; x1569 < 150; x1569++) {
for(int x1570=0; x1570 < 150; x1570++) {
int32_t x1571 = x1568;
int32_t x1572 = x1571 + x1570;
float x1573 = x160[x1572];
float x1574 = x271[x1570];
float x1575 = x931[x1569];
float x1576 = x1574 * x1575;
float x1577 = x1573 + x1576;
x160[x1572] = x1577;

}
x1568 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x152,150,x931,1,1,x272,1);
// add_cartesian
int32_t x1586 = 0;
for(int x1587=0; x1587 < 150; x1587++) {
for(int x1588=0; x1588 < 150; x1588++) {
int32_t x1589 = x1586;
int32_t x1590 = x1589 + x1588;
float x1591 = x151[x1590];
float x1592 = x262[x1588];
float x1593 = x928[x1587];
float x1594 = x1592 * x1593;
float x1595 = x1591 + x1594;
x151[x1590] = x1595;

}
x1586 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x143,150,x928,1,1,x263,1);
for(int x1603=0; x1603 < 150; x1603++) {
float x1604 = x879[x1603];
float x1605 = x913[x1603];
float x1608 = x925[x1603];
float x1606 = 1.0f - x1605;
float x1607 = x1606 * x1605;
float x1609 = x1607 * x1608;
float x1610 = x1604 + x1609;
x879[x1603] = x1610;

}
int32_t x1614 = 0;
int32_t x1615 = 0;
int32_t x1616 = 0;
for(int x1617=0; x1617 < 150; x1617++) {
int32_t x1618 = x1615;
float x1619 = x142[x1618];
int32_t x1620 = x1616;
float x1621 = x879[x1620];
float x1622 = x1619 + x1621;
x142[x1618] = x1622;
x1614 += 1;
x1615 += 1;
x1616 += 1;

}
int32_t x1629 = 0;
int32_t x1630 = 0;
int32_t x1631 = 0;
for(int x1632=0; x1632 < 150; x1632++) {
int32_t x1633 = x1630;
float x1634 = x882[x1633];
int32_t x1635 = x1631;
float x1636 = x879[x1635];
float x1637 = x1634 + x1636;
x882[x1633] = x1637;
x1629 += 1;
x1630 += 1;
x1631 += 1;

}
// add_cartesian
int32_t x1645 = 0;
for(int x1646=0; x1646 < 150; x1646++) {
for(int x1647=0; x1647 < 150; x1647++) {
int32_t x1648 = x1645;
int32_t x1649 = x1648 + x1647;
float x1650 = x140[x1649];
float x1651 = x271[x1647];
float x1652 = x882[x1646];
float x1653 = x1651 * x1652;
float x1654 = x1650 + x1653;
x140[x1649] = x1654;

}
x1645 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x132,150,x882,1,1,x272,1);
// add_cartesian
int32_t x1663 = 0;
for(int x1664=0; x1664 < 150; x1664++) {
for(int x1665=0; x1665 < 150; x1665++) {
int32_t x1666 = x1663;
int32_t x1667 = x1666 + x1665;
float x1668 = x131[x1667];
float x1669 = x262[x1665];
float x1670 = x879[x1664];
float x1671 = x1669 * x1670;
float x1672 = x1668 + x1671;
x131[x1667] = x1672;

}
x1663 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x123,150,x879,1,1,x263,1);
for(int x1680=0; x1680 < 150; x1680++) {
float x1681 = x830[x1680];
float x1682 = x864[x1680];
float x1685 = x876[x1680];
float x1683 = 1.0f - x1682;
float x1684 = x1683 * x1682;
float x1686 = x1684 * x1685;
float x1687 = x1681 + x1686;
x830[x1680] = x1687;

}
int32_t x1691 = 0;
int32_t x1692 = 0;
int32_t x1693 = 0;
for(int x1694=0; x1694 < 150; x1694++) {
int32_t x1695 = x1692;
float x1696 = x142[x1695];
int32_t x1697 = x1693;
float x1698 = x830[x1697];
float x1699 = x1696 + x1698;
x142[x1695] = x1699;
x1691 += 1;
x1692 += 1;
x1693 += 1;

}
int32_t x1706 = 0;
int32_t x1707 = 0;
int32_t x1708 = 0;
for(int x1709=0; x1709 < 150; x1709++) {
int32_t x1710 = x1707;
float x1711 = x833[x1710];
int32_t x1712 = x1708;
float x1713 = x830[x1712];
float x1714 = x1711 + x1713;
x833[x1710] = x1714;
x1706 += 1;
x1707 += 1;
x1708 += 1;

}
// add_cartesian
int32_t x1722 = 0;
for(int x1723=0; x1723 < 150; x1723++) {
for(int x1724=0; x1724 < 150; x1724++) {
int32_t x1725 = x1722;
int32_t x1726 = x1725 + x1724;
float x1727 = x122[x1726];
float x1728 = x271[x1724];
float x1729 = x833[x1723];
float x1730 = x1728 * x1729;
float x1731 = x1727 + x1730;
x122[x1726] = x1731;

}
x1722 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x114,150,x833,1,1,x272,1);
// add_cartesian
int32_t x1740 = 0;
for(int x1741=0; x1741 < 150; x1741++) {
for(int x1742=0; x1742 < 150; x1742++) {
int32_t x1743 = x1740;
int32_t x1744 = x1743 + x1742;
float x1745 = x113[x1744];
float x1746 = x262[x1742];
float x1747 = x830[x1741];
float x1748 = x1746 * x1747;
float x1749 = x1745 + x1748;
x113[x1744] = x1749;

}
x1740 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x105,150,x830,1,1,x263,1);
for(int x1757=0; x1757 < 150; x1757++) {
float x1758 = x781[x1757];
float x1759 = x815[x1757];
float x1762 = x827[x1757];
float x1760 = 1.0f - x1759;
float x1761 = x1760 * x1759;
float x1763 = x1761 * x1762;
float x1764 = x1758 + x1763;
x781[x1757] = x1764;

}
int32_t x1768 = 0;
int32_t x1769 = 0;
int32_t x1770 = 0;
for(int x1771=0; x1771 < 150; x1771++) {
int32_t x1772 = x1769;
float x1773 = x104[x1772];
int32_t x1774 = x1770;
float x1775 = x781[x1774];
float x1776 = x1773 + x1775;
x104[x1772] = x1776;
x1768 += 1;
x1769 += 1;
x1770 += 1;

}
int32_t x1783 = 0;
int32_t x1784 = 0;
int32_t x1785 = 0;
for(int x1786=0; x1786 < 150; x1786++) {
int32_t x1787 = x1784;
float x1788 = x784[x1787];
int32_t x1789 = x1785;
float x1790 = x781[x1789];
float x1791 = x1788 + x1790;
x784[x1787] = x1791;
x1783 += 1;
x1784 += 1;
x1785 += 1;

}
// add_cartesian
int32_t x1799 = 0;
for(int x1800=0; x1800 < 150; x1800++) {
for(int x1801=0; x1801 < 150; x1801++) {
int32_t x1802 = x1799;
int32_t x1803 = x1802 + x1801;
float x1804 = x102[x1803];
float x1805 = x271[x1801];
float x1806 = x784[x1800];
float x1807 = x1805 * x1806;
float x1808 = x1804 + x1807;
x102[x1803] = x1808;

}
x1799 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x94,150,x784,1,1,x272,1);
// add_cartesian
int32_t x1817 = 0;
for(int x1818=0; x1818 < 150; x1818++) {
for(int x1819=0; x1819 < 150; x1819++) {
int32_t x1820 = x1817;
int32_t x1821 = x1820 + x1819;
float x1822 = x93[x1821];
float x1823 = x262[x1819];
float x1824 = x781[x1818];
float x1825 = x1823 * x1824;
float x1826 = x1822 + x1825;
x93[x1821] = x1826;

}
x1817 += 150;

}
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x84,150,x781,1,1,x263,1);
}
};
x243(x266,x267,x1837);
};
x243(x257,x258,x1847);
} else {
float** x1874 = (float**)myMalloc(6 * sizeof(float*));;
x1874[0] = x195;
x1874[1] = x196;
x1874[2] = x197;
x1874[3] = x198;
x1874[4] = x199;
x1874[5] = x200;
function<void(float**)> x248 = x245;
function<void(float**)> x1857 = [&](float** x1858) {
float* x1859 = x1858[0];
float* x1860 = x1858[1];
float* x1861 = x1858[2];
float* x1862 = x1858[3];
float* x1863 = x1858[4];
float* x1864 = x1858[5];
float** x1865 = (float**)myMalloc(6 * sizeof(float*));;
x1865[0] = x1859;
x1865[1] = x1860;
x1865[2] = x1861;
x1865[3] = x1862;
x1865[4] = x1863;
x1865[5] = x1864;
x248(x1865);
};
x1857(x1874);
}
};
float* x239 = (float*)myMalloc(1 * sizeof(float));;
float* x240 = (float*)myMalloc(1 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
float* x242 = (float*)myMalloc(1 * sizeof(float));;
float** x1898 = (float**)myMalloc(6 * sizeof(float*));;
x1898[0] = x195;
x1898[1] = x196;
x1898[2] = x197;
x1898[3] = x198;
x1898[4] = x199;
x1898[5] = x200;
function<void(float**)> x1885 = [&](float** x1886) {
float* x1887 = x1886[0];
float* x1888 = x1886[1];
float* x1889 = x1886[2];
float* x1890 = x1886[3];
float* x1891 = x1886[4];
float* x1892 = x1886[5];
x1888[0] = 1.0f;
// backend is lantern.TensorDsl$BackendCPU@c97ce40
float x1895 = x1887[0];
x242[0] = x1895;
};
x243(0,x1885,x1898);
float x1907 = x242[0];
float x1908 = x228;
float x1909 = (float)x229;
float x1910 = x1908 * x1909;
int32_t x1911 = x229 + 1;
float x1912 = (float)x1911;
float x1913 = x1910 / x1912;
float x1914 = x1907 / x1912;
float x1915 = x1913 + x1914;
x228 = x1915;
for(int x1917=0; x1917 < 150; x1917++) {
float x1918 = x104[x1917];
float x1919 = x1918;
float x1920 = x1919;
bool x1921 = x1920 > 1.0f;
if (x1921) {
x1919 = 1.0f;
} else {
}
float x1925 = x1919;
bool x1926 = x1925 < -1.0f;
if (x1926) {
x1919 = -1.0f;
} else {
}
float x1930 = x201[x1917];
float x1931 = x1919;
float x1932 = x1931 * x1931;
float x1933 = x1930 + x1932;
x201[x1917] = x1933;
float x1935 = x103[x1917];
float x1937 = x201[x1917];
float x1936 = 0.05f * x1931;
double x1938 = (double)x1937;
double x1939 = x1938 + 9.99999993922529E-9;
double x1940 = sqrt(x1939);
float x1941 = (float)x1940;
float x1942 = x1936 / x1941;
float x1943 = x1935 - x1942;
x103[x1917] = x1943;
x104[x1917] = 0.0f;

}
for(int x1948=0; x1948 < 22500; x1948++) {
float x1949 = x131[x1948];
float x1950 = x1949;
float x1951 = x1950;
bool x1952 = x1951 > 1.0f;
if (x1952) {
x1950 = 1.0f;
} else {
}
float x1956 = x1950;
bool x1957 = x1956 < -1.0f;
if (x1957) {
x1950 = -1.0f;
} else {
}
float x1961 = x202[x1948];
float x1962 = x1950;
float x1963 = x1962 * x1962;
float x1964 = x1961 + x1963;
x202[x1948] = x1964;
float x1966 = x123[x1948];
float x1968 = x202[x1948];
float x1967 = 0.05f * x1962;
double x1969 = (double)x1968;
double x1970 = x1969 + 9.99999993922529E-9;
double x1971 = sqrt(x1970);
float x1972 = (float)x1971;
float x1973 = x1967 / x1972;
float x1974 = x1966 - x1973;
x123[x1948] = x1974;
x131[x1948] = 0.0f;

}
for(int x1979=0; x1979 < 22500; x1979++) {
float x1980 = x102[x1979];
float x1981 = x1980;
float x1982 = x1981;
bool x1983 = x1982 > 1.0f;
if (x1983) {
x1981 = 1.0f;
} else {
}
float x1987 = x1981;
bool x1988 = x1987 < -1.0f;
if (x1988) {
x1981 = -1.0f;
} else {
}
float x1992 = x203[x1979];
float x1993 = x1981;
float x1994 = x1993 * x1993;
float x1995 = x1992 + x1994;
x203[x1979] = x1995;
float x1997 = x94[x1979];
float x1999 = x203[x1979];
float x1998 = 0.05f * x1993;
double x2000 = (double)x1999;
double x2001 = x2000 + 9.99999993922529E-9;
double x2002 = sqrt(x2001);
float x2003 = (float)x2002;
float x2004 = x1998 / x2003;
float x2005 = x1997 - x2004;
x94[x1979] = x2005;
x102[x1979] = 0.0f;

}
for(int x2010=0; x2010 < 22500; x2010++) {
float x2011 = x180[x2010];
float x2012 = x2011;
float x2013 = x2012;
bool x2014 = x2013 > 1.0f;
if (x2014) {
x2012 = 1.0f;
} else {
}
float x2018 = x2012;
bool x2019 = x2018 < -1.0f;
if (x2019) {
x2012 = -1.0f;
} else {
}
float x2023 = x204[x2010];
float x2024 = x2012;
float x2025 = x2024 * x2024;
float x2026 = x2023 + x2025;
x204[x2010] = x2026;
float x2028 = x172[x2010];
float x2030 = x204[x2010];
float x2029 = 0.05f * x2024;
double x2031 = (double)x2030;
double x2032 = x2031 + 9.99999993922529E-9;
double x2033 = sqrt(x2032);
float x2034 = (float)x2033;
float x2035 = x2029 / x2034;
float x2036 = x2028 - x2035;
x172[x2010] = x2036;
x180[x2010] = 0.0f;

}
for(int x2041=0; x2041 < 22500; x2041++) {
float x2042 = x113[x2041];
float x2043 = x2042;
float x2044 = x2043;
bool x2045 = x2044 > 1.0f;
if (x2045) {
x2043 = 1.0f;
} else {
}
float x2049 = x2043;
bool x2050 = x2049 < -1.0f;
if (x2050) {
x2043 = -1.0f;
} else {
}
float x2054 = x205[x2041];
float x2055 = x2043;
float x2056 = x2055 * x2055;
float x2057 = x2054 + x2056;
x205[x2041] = x2057;
float x2059 = x105[x2041];
float x2061 = x205[x2041];
float x2060 = 0.05f * x2055;
double x2062 = (double)x2061;
double x2063 = x2062 + 9.99999993922529E-9;
double x2064 = sqrt(x2063);
float x2065 = (float)x2064;
float x2066 = x2060 / x2065;
float x2067 = x2059 - x2066;
x105[x2041] = x2067;
x113[x2041] = 0.0f;

}
for(int x2072=0; x2072 < 22500; x2072++) {
float x2073 = x160[x2072];
float x2074 = x2073;
float x2075 = x2074;
bool x2076 = x2075 > 1.0f;
if (x2076) {
x2074 = 1.0f;
} else {
}
float x2080 = x2074;
bool x2081 = x2080 < -1.0f;
if (x2081) {
x2074 = -1.0f;
} else {
}
float x2085 = x206[x2072];
float x2086 = x2074;
float x2087 = x2086 * x2086;
float x2088 = x2085 + x2087;
x206[x2072] = x2088;
float x2090 = x152[x2072];
float x2092 = x206[x2072];
float x2091 = 0.05f * x2086;
double x2093 = (double)x2092;
double x2094 = x2093 + 9.99999993922529E-9;
double x2095 = sqrt(x2094);
float x2096 = (float)x2095;
float x2097 = x2091 / x2096;
float x2098 = x2090 - x2097;
x152[x2072] = x2098;
x160[x2072] = 0.0f;

}
for(int x2103=0; x2103 < 22500; x2103++) {
float x2104 = x140[x2103];
float x2105 = x2104;
float x2106 = x2105;
bool x2107 = x2106 > 1.0f;
if (x2107) {
x2105 = 1.0f;
} else {
}
float x2111 = x2105;
bool x2112 = x2111 < -1.0f;
if (x2112) {
x2105 = -1.0f;
} else {
}
float x2116 = x207[x2103];
float x2117 = x2105;
float x2118 = x2117 * x2117;
float x2119 = x2116 + x2118;
x207[x2103] = x2119;
float x2121 = x132[x2103];
float x2123 = x207[x2103];
float x2122 = 0.05f * x2117;
double x2124 = (double)x2123;
double x2125 = x2124 + 9.99999993922529E-9;
double x2126 = sqrt(x2125);
float x2127 = (float)x2126;
float x2128 = x2122 / x2127;
float x2129 = x2121 - x2128;
x132[x2103] = x2129;
x140[x2103] = 0.0f;

}
for(int x2134=0; x2134 < 22500; x2134++) {
float x2135 = x171[x2134];
float x2136 = x2135;
float x2137 = x2136;
bool x2138 = x2137 > 1.0f;
if (x2138) {
x2136 = 1.0f;
} else {
}
float x2142 = x2136;
bool x2143 = x2142 < -1.0f;
if (x2143) {
x2136 = -1.0f;
} else {
}
float x2147 = x208[x2134];
float x2148 = x2136;
float x2149 = x2148 * x2148;
float x2150 = x2147 + x2149;
x208[x2134] = x2150;
float x2152 = x163[x2134];
float x2154 = x208[x2134];
float x2153 = 0.05f * x2148;
double x2155 = (double)x2154;
double x2156 = x2155 + 9.99999993922529E-9;
double x2157 = sqrt(x2156);
float x2158 = (float)x2157;
float x2159 = x2153 / x2158;
float x2160 = x2152 - x2159;
x163[x2134] = x2160;
x171[x2134] = 0.0f;

}
for(int x2165=0; x2165 < 22500; x2165++) {
float x2166 = x151[x2165];
float x2167 = x2166;
float x2168 = x2167;
bool x2169 = x2168 > 1.0f;
if (x2169) {
x2167 = 1.0f;
} else {
}
float x2173 = x2167;
bool x2174 = x2173 < -1.0f;
if (x2174) {
x2167 = -1.0f;
} else {
}
float x2178 = x209[x2165];
float x2179 = x2167;
float x2180 = x2179 * x2179;
float x2181 = x2178 + x2180;
x209[x2165] = x2181;
float x2183 = x143[x2165];
float x2185 = x209[x2165];
float x2184 = 0.05f * x2179;
double x2186 = (double)x2185;
double x2187 = x2186 + 9.99999993922529E-9;
double x2188 = sqrt(x2187);
float x2189 = (float)x2188;
float x2190 = x2184 / x2189;
float x2191 = x2183 - x2190;
x143[x2165] = x2191;
x151[x2165] = 0.0f;

}
for(int x2196=0; x2196 < 22500; x2196++) {
float x2197 = x93[x2196];
float x2198 = x2197;
float x2199 = x2198;
bool x2200 = x2199 > 1.0f;
if (x2200) {
x2198 = 1.0f;
} else {
}
float x2204 = x2198;
bool x2205 = x2204 < -1.0f;
if (x2205) {
x2198 = -1.0f;
} else {
}
float x2209 = x210[x2196];
float x2210 = x2198;
float x2211 = x2210 * x2210;
float x2212 = x2209 + x2211;
x210[x2196] = x2212;
float x2214 = x84[x2196];
float x2216 = x210[x2196];
float x2215 = 0.05f * x2210;
double x2217 = (double)x2216;
double x2218 = x2217 + 9.99999993922529E-9;
double x2219 = sqrt(x2218);
float x2220 = (float)x2219;
float x2221 = x2215 / x2220;
float x2222 = x2214 - x2221;
x84[x2196] = x2222;
x93[x2196] = 0.0f;

}
for(int x2227=0; x2227 < 150; x2227++) {
float x2228 = x182[x2227];
float x2229 = x2228;
float x2230 = x2229;
bool x2231 = x2230 > 1.0f;
if (x2231) {
x2229 = 1.0f;
} else {
}
float x2235 = x2229;
bool x2236 = x2235 < -1.0f;
if (x2236) {
x2229 = -1.0f;
} else {
}
float x2240 = x211[x2227];
float x2241 = x2229;
float x2242 = x2241 * x2241;
float x2243 = x2240 + x2242;
x211[x2227] = x2243;
float x2245 = x181[x2227];
float x2247 = x211[x2227];
float x2246 = 0.05f * x2241;
double x2248 = (double)x2247;
double x2249 = x2248 + 9.99999993922529E-9;
double x2250 = sqrt(x2249);
float x2251 = (float)x2250;
float x2252 = x2246 / x2251;
float x2253 = x2245 - x2252;
x181[x2227] = x2253;
x182[x2227] = 0.0f;

}
for(int x2258=0; x2258 < 22500; x2258++) {
float x2259 = x122[x2258];
float x2260 = x2259;
float x2261 = x2260;
bool x2262 = x2261 > 1.0f;
if (x2262) {
x2260 = 1.0f;
} else {
}
float x2266 = x2260;
bool x2267 = x2266 < -1.0f;
if (x2267) {
x2260 = -1.0f;
} else {
}
float x2271 = x212[x2258];
float x2272 = x2260;
float x2273 = x2272 * x2272;
float x2274 = x2271 + x2273;
x212[x2258] = x2274;
float x2276 = x114[x2258];
float x2278 = x212[x2258];
float x2277 = 0.05f * x2272;
double x2279 = (double)x2278;
double x2280 = x2279 + 9.99999993922529E-9;
double x2281 = sqrt(x2280);
float x2282 = (float)x2281;
float x2283 = x2277 / x2282;
float x2284 = x2276 - x2283;
x114[x2258] = x2284;
x122[x2258] = 0.0f;

}
for(int x2289=0; x2289 < 150; x2289++) {
float x2290 = x142[x2289];
float x2291 = x2290;
float x2292 = x2291;
bool x2293 = x2292 > 1.0f;
if (x2293) {
x2291 = 1.0f;
} else {
}
float x2297 = x2291;
bool x2298 = x2297 < -1.0f;
if (x2298) {
x2291 = -1.0f;
} else {
}
float x2302 = x213[x2289];
float x2303 = x2291;
float x2304 = x2303 * x2303;
float x2305 = x2302 + x2304;
x213[x2289] = x2305;
float x2307 = x141[x2289];
float x2309 = x213[x2289];
float x2308 = 0.05f * x2303;
double x2310 = (double)x2309;
double x2311 = x2310 + 9.99999993922529E-9;
double x2312 = sqrt(x2311);
float x2313 = (float)x2312;
float x2314 = x2308 / x2313;
float x2315 = x2307 - x2314;
x141[x2289] = x2315;
x142[x2289] = 0.0f;

}
for(int x2320=0; x2320 < 150; x2320++) {
float x2321 = x162[x2320];
float x2322 = x2321;
float x2323 = x2322;
bool x2324 = x2323 > 1.0f;
if (x2324) {
x2322 = 1.0f;
} else {
}
float x2328 = x2322;
bool x2329 = x2328 < -1.0f;
if (x2329) {
x2322 = -1.0f;
} else {
}
float x2333 = x214[x2320];
float x2334 = x2322;
float x2335 = x2334 * x2334;
float x2336 = x2333 + x2335;
x214[x2320] = x2336;
float x2338 = x161[x2320];
float x2340 = x214[x2320];
float x2339 = 0.05f * x2334;
double x2341 = (double)x2340;
double x2342 = x2341 + 9.99999993922529E-9;
double x2343 = sqrt(x2342);
float x2344 = (float)x2343;
float x2345 = x2339 / x2344;
float x2346 = x2338 - x2345;
x161[x2320] = x2346;
x162[x2320] = 0.0f;

}
for(int x2351=0; x2351 < 150; x2351++) {
float x2352 = x61[x2351];
float x2353 = x2352;
float x2354 = x2353;
bool x2355 = x2354 > 1.0f;
if (x2355) {
x2353 = 1.0f;
} else {
}
float x2359 = x2353;
bool x2360 = x2359 < -1.0f;
if (x2360) {
x2353 = -1.0f;
} else {
}
float x2364 = x215[x2351];
float x2365 = x2353;
float x2366 = x2365 * x2365;
float x2367 = x2364 + x2366;
x215[x2351] = x2367;
float x2369 = x60[x2351];
float x2371 = x215[x2351];
float x2370 = 0.05f * x2365;
double x2372 = (double)x2371;
double x2373 = x2372 + 9.99999993922529E-9;
double x2374 = sqrt(x2373);
float x2375 = (float)x2374;
float x2376 = x2370 / x2375;
float x2377 = x2369 - x2376;
x60[x2351] = x2377;
x61[x2351] = 0.0f;

}
for(int x2382=0; x2382 < 45000; x2382++) {
float x2383 = x59[x2382];
float x2384 = x2383;
float x2385 = x2384;
bool x2386 = x2385 > 1.0f;
if (x2386) {
x2384 = 1.0f;
} else {
}
float x2390 = x2384;
bool x2391 = x2390 < -1.0f;
if (x2391) {
x2384 = -1.0f;
} else {
}
float x2395 = x216[x2382];
float x2396 = x2384;
float x2397 = x2396 * x2396;
float x2398 = x2395 + x2397;
x216[x2382] = x2398;
float x2400 = x50[x2382];
float x2402 = x216[x2382];
float x2401 = 0.05f * x2396;
double x2403 = (double)x2402;
double x2404 = x2403 + 9.99999993922529E-9;
double x2405 = sqrt(x2404);
float x2406 = (float)x2405;
float x2407 = x2401 / x2406;
float x2408 = x2400 - x2407;
x50[x2382] = x2408;
x59[x2382] = 0.0f;

}
for(int x2413=0; x2413 < 150; x2413++) {
float x2414 = x83[x2413];
float x2415 = x2414;
float x2416 = x2415;
bool x2417 = x2416 > 1.0f;
if (x2417) {
x2415 = 1.0f;
} else {
}
float x2421 = x2415;
bool x2422 = x2421 < -1.0f;
if (x2422) {
x2415 = -1.0f;
} else {
}
float x2426 = x217[x2413];
float x2427 = x2415;
float x2428 = x2427 * x2427;
float x2429 = x2426 + x2428;
x217[x2413] = x2429;
float x2431 = x82[x2413];
float x2433 = x217[x2413];
float x2432 = 0.05f * x2427;
double x2434 = (double)x2433;
double x2435 = x2434 + 9.99999993922529E-9;
double x2436 = sqrt(x2435);
float x2437 = (float)x2436;
float x2438 = x2432 / x2437;
float x2439 = x2431 - x2438;
x82[x2413] = x2439;
x83[x2413] = 0.0f;

}
for(int x2444=0; x2444 < 45000; x2444++) {
float x2445 = x81[x2444];
float x2446 = x2445;
float x2447 = x2446;
bool x2448 = x2447 > 1.0f;
if (x2448) {
x2446 = 1.0f;
} else {
}
float x2452 = x2446;
bool x2453 = x2452 < -1.0f;
if (x2453) {
x2446 = -1.0f;
} else {
}
float x2457 = x218[x2444];
float x2458 = x2446;
float x2459 = x2458 * x2458;
float x2460 = x2457 + x2459;
x218[x2444] = x2460;
float x2462 = x73[x2444];
float x2464 = x218[x2444];
float x2463 = 0.05f * x2458;
double x2465 = (double)x2464;
double x2466 = x2465 + 9.99999993922529E-9;
double x2467 = sqrt(x2466);
float x2468 = (float)x2467;
float x2469 = x2463 / x2468;
float x2470 = x2462 - x2469;
x73[x2444] = x2470;
x81[x2444] = 0.0f;

}
for(int x2475=0; x2475 < 150; x2475++) {
float x2476 = x72[x2475];
float x2477 = x2476;
float x2478 = x2477;
bool x2479 = x2478 > 1.0f;
if (x2479) {
x2477 = 1.0f;
} else {
}
float x2483 = x2477;
bool x2484 = x2483 < -1.0f;
if (x2484) {
x2477 = -1.0f;
} else {
}
float x2488 = x219[x2475];
float x2489 = x2477;
float x2490 = x2489 * x2489;
float x2491 = x2488 + x2490;
x219[x2475] = x2491;
float x2493 = x71[x2475];
float x2495 = x219[x2475];
float x2494 = 0.05f * x2489;
double x2496 = (double)x2495;
double x2497 = x2496 + 9.99999993922529E-9;
double x2498 = sqrt(x2497);
float x2499 = (float)x2498;
float x2500 = x2494 / x2499;
float x2501 = x2493 - x2500;
x71[x2475] = x2501;
x72[x2475] = 0.0f;

}
for(int x2506=0; x2506 < 45000; x2506++) {
float x2507 = x70[x2506];
float x2508 = x2507;
float x2509 = x2508;
bool x2510 = x2509 > 1.0f;
if (x2510) {
x2508 = 1.0f;
} else {
}
float x2514 = x2508;
bool x2515 = x2514 < -1.0f;
if (x2515) {
x2508 = -1.0f;
} else {
}
float x2519 = x220[x2506];
float x2520 = x2508;
float x2521 = x2520 * x2520;
float x2522 = x2519 + x2521;
x220[x2506] = x2522;
float x2524 = x62[x2506];
float x2526 = x220[x2506];
float x2525 = 0.05f * x2520;
double x2527 = (double)x2526;
double x2528 = x2527 + 9.99999993922529E-9;
double x2529 = sqrt(x2528);
float x2530 = (float)x2529;
float x2531 = x2525 / x2530;
float x2532 = x2524 - x2531;
x62[x2506] = x2532;
x70[x2506] = 0.0f;

}
for(int x2537=0; x2537 < 5; x2537++) {
float x2538 = x194[x2537];
float x2539 = x2538;
float x2540 = x2539;
bool x2541 = x2540 > 1.0f;
if (x2541) {
x2539 = 1.0f;
} else {
}
float x2545 = x2539;
bool x2546 = x2545 < -1.0f;
if (x2546) {
x2539 = -1.0f;
} else {
}
float x2550 = x221[x2537];
float x2551 = x2539;
float x2552 = x2551 * x2551;
float x2553 = x2550 + x2552;
x221[x2537] = x2553;
float x2555 = x193[x2537];
float x2557 = x221[x2537];
float x2556 = 0.05f * x2551;
double x2558 = (double)x2557;
double x2559 = x2558 + 9.99999993922529E-9;
double x2560 = sqrt(x2559);
float x2561 = (float)x2560;
float x2562 = x2556 / x2561;
float x2563 = x2555 - x2562;
x193[x2537] = x2563;
x194[x2537] = 0.0f;

}
for(int x2568=0; x2568 < 750; x2568++) {
float x2569 = x192[x2568];
float x2570 = x2569;
float x2571 = x2570;
bool x2572 = x2571 > 1.0f;
if (x2572) {
x2570 = 1.0f;
} else {
}
float x2576 = x2570;
bool x2577 = x2576 < -1.0f;
if (x2577) {
x2570 = -1.0f;
} else {
}
float x2581 = x222[x2568];
float x2582 = x2570;
float x2583 = x2582 * x2582;
float x2584 = x2581 + x2583;
x222[x2568] = x2584;
float x2586 = x183[x2568];
float x2588 = x222[x2568];
float x2587 = 0.05f * x2582;
double x2589 = (double)x2588;
double x2590 = x2589 + 9.99999993922529E-9;
double x2591 = sqrt(x2590);
float x2592 = (float)x2591;
float x2593 = x2587 / x2592;
float x2594 = x2586 - x2593;
x183[x2568] = x2594;
x192[x2568] = 0.0f;

}
int64_t x2599 = (long)mallocAddr;
int64_t x2600 = x2599 - x224;
memset((void*)x224, 0, x2600);
mallocAddr = (void*)x224;

}
float x2605 = x228;
double x2606 = (double)x2605;
x223[x227] = x2606;
double x2608 = ((double)clock() / CLOCKS_PER_SEC);
double x2609 = x2608 - x225;
printf("epoc %d, average_loss %f, time %lf\n",x227,x2605,x2609);

}
double x2613 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2617 = (long)fopen(x0, "w");
fprintf((FILE *)x2617, "unit: %s\n", "1 epoch");
for(int x2619=0; x2619 < 6; x2619++) {
double x2620 = x223[x2619];
fprintf((FILE *)x2617, "%lf\n", x2620);

}
double x2614 = x225 - x2;
double x2615 = x2613 - x225;
double x2616 = x2615 / 6.0;
fprintf((FILE *)x2617, "run time: %lf %lf\n", x2614, x2616);
fclose((FILE*)x2617);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

