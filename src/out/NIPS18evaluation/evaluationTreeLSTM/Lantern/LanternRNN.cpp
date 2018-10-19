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
float** x948 = (float**)myMalloc(4 * sizeof(float*));;
x948[0] = x120;
x948[1] = x121;
x948[2] = x122;
x948[3] = x123;
function<void(float**)> x129 = x126;
function<void(float**)> x287 = [&](float** x288) {
float* x289 = x288[0];
float* x290 = x288[1];
float* x291 = x288[2];
float* x292 = x288[3];
float** x293 = (float**)myMalloc(4 * sizeof(float*));;
x293[0] = x289;
x293[1] = x290;
x293[2] = x291;
x293[3] = x292;
x129(x293);
};
function<void(float**)> x281 = [&](float** x282) {
float* x283 = x282[0];
float* x284 = x282[1];
float* x285 = x282[2];
float* x286 = x282[3];
float** x300 = (float**)myMalloc(4 * sizeof(float*));;
x300[0] = x283;
x300[1] = x284;
x300[2] = x285;
x300[3] = x286;
x287(x300);
};
function<void(float**)> x661 = [&](float** x662) {
float* x663 = x662[0];
float* x664 = x662[1];
float* x665 = x662[2];
float* x666 = x662[3];
float** x667 = (float**)myMalloc(4 * sizeof(float*));;
x667[0] = x663;
x667[1] = x664;
x667[2] = x665;
x667[3] = x666;
x129(x667);
};
function<void(float**)> x655 = [&](float** x656) {
float* x657 = x656[0];
float* x658 = x656[1];
float* x659 = x656[2];
float* x660 = x656[3];
float** x674 = (float**)myMalloc(4 * sizeof(float*));;
x674[0] = x657;
x674[1] = x658;
x674[2] = x659;
x674[3] = x660;
x661(x674);
};
function<void(float**)> x137 = [&](float** x138) {
float* x139 = x138[0];
float* x140 = x138[1];
float* x141 = x138[2];
float* x142 = x138[3];
int32_t x143 = x116[x128];
float** x940 = (float**)myMalloc(4 * sizeof(float*));;
x940[0] = x120;
x940[1] = x121;
x940[2] = x122;
x940[3] = x123;
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
for(int x158=0; x158 < 150; x158++) {
float x159 = 0.0f;
int32_t x161 = x158 * 300;
for(int x160=0; x160 < 300; x160++) {
int32_t x162 = x161 + x160;
float x163 = x50[x162];
float x164 = x153[x160];
float x165 = x163 * x164;
x159 += x165;

}
float x169 = x159;
x156[x158] = x169;

}
float* x173 = (float*)myMalloc(150 * sizeof(float));;
float* x174 = (float*)myMalloc(150 * sizeof(float));;
int32_t x175 = 0;
int32_t x176 = 0;
int32_t x177 = 0;
for(int x178=0; x178 < 150; x178++) {
int32_t x179 = x175;
int32_t x180 = x176;
float x181 = x156[x180];
int32_t x182 = x177;
float x183 = x59[x182];
float x184 = x181 + x183;
x174[x179] = x184;
x175 += 1;
x176 += 1;
x177 += 1;

}
float* x191 = (float*)myMalloc(150 * sizeof(float));;
float* x192 = (float*)myMalloc(150 * sizeof(float));;
for(int x193=0; x193 < 150; x193++) {
float x194 = x174[x193];
double x195 = (double)x194;
double x196 = tanh(x195);
float x197 = (float)x196;
x192[x193] = x197;

}
float* x201 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(5, 150), List(150)
float* x203 = (float*)myMalloc(5 * sizeof(float));;
for(int x205=0; x205 < 5; x205++) {
float x206 = 0.0f;
int32_t x208 = x205 * 150;
for(int x207=0; x207 < 150; x207++) {
int32_t x209 = x208 + x207;
float x210 = x78[x209];
float x211 = x192[x207];
float x212 = x210 * x211;
x206 += x212;

}
float x216 = x206;
x203[x205] = x216;

}
float* x220 = (float*)myMalloc(5 * sizeof(float));;
float* x221 = (float*)myMalloc(5 * sizeof(float));;
int32_t x222 = 0;
int32_t x223 = 0;
int32_t x224 = 0;
for(int x225=0; x225 < 5; x225++) {
int32_t x226 = x222;
int32_t x227 = x223;
float x228 = x203[x227];
int32_t x229 = x224;
float x230 = x87[x229];
float x231 = x228 + x230;
x221[x226] = x231;
x222 += 1;
x223 += 1;
x224 += 1;

}
float* x238 = (float*)myMalloc(5 * sizeof(float));;
float x239 = -3.4028235E38f;
for(int x240=0; x240 < 5; x240++) {
float x241 = x239;
float x242 = x221[x240];
bool x243 = x242 > x241;
float x244;
if (x243) {
x244 = x242;
} else {
x244 = x241;
}
x239 = x244;

}
float x248 = x239;
float x249 = 0.0f;
for(int x250=0; x250 < 5; x250++) {
float x251 = x249;
float x252 = x221[x250];
float x253 = x239;
float x254 = x252 - x253;
double x255 = (double)x254;
double x256 = exp(x255);
float x257 = (float)x256;
float x258 = x251 + x257;
x249 = x258;

}
float x262 = x249;
float* x267 = (float*)myMalloc(5 * sizeof(float));;
double x263 = (double)x262;
double x264 = log(x263);
float x265 = (float)x264;
float x266 = x248 + x265;
for(int x268=0; x268 < 5; x268++) {
float x269 = x221[x268];
float x270 = x269 - x266;
x267[x268] = x270;

}
float* x274 = (float*)myMalloc(5 * sizeof(float));;
int32_t x275 = x110[x128];
float x276 = x267[x275];
float* x278 = (float*)myMalloc(1 * sizeof(float));;
float x277 = -1.0f * x276;
x278[0] = x277;
float* x280 = (float*)myMalloc(1 * sizeof(float));;
float** x307 = (float**)myMalloc(4 * sizeof(float*));;
x307[0] = x278;
x307[1] = x280;
x307[2] = x192;
x307[3] = x201;
x281(x307);
float x313 = x274[x275];
float x314 = x280[0];
float x315 = -1.0f * x314;
float x316 = x313 + x315;
x274[x275] = x316;
float x318 = 0.0f;
for(int x319=0; x319 < 5; x319++) {
float x320 = x318;
float x321 = x274[x319];
float x322 = x320 + x321;
x318 = x322;

}
float x326 = x318;
float* x327 = (float*)myMalloc(1 * sizeof(float));;
x327[0] = x326;
float x329 = x327[0];
for(int x330=0; x330 < 5; x330++) {
float x331 = x238[x330];
float x332 = x274[x330];
float x333 = x267[x330];
double x334 = (double)x333;
double x335 = exp(x334);
float x336 = (float)x335;
float x337 = x336 * x329;
float x338 = x332 - x337;
float x339 = x331 + x338;
x238[x330] = x339;

}
int32_t x343 = 0;
int32_t x344 = 0;
int32_t x345 = 0;
for(int x346=0; x346 < 5; x346++) {
int32_t x347 = x343;
float x348 = x220[x347];
float x349 = x203[x347];
int32_t x350 = x344;
float x351 = x87[x350];
int32_t x352 = x345;
float x353 = x238[x352];
float x354 = x348 + x353;
x220[x347] = x354;
float x356 = x94[x350];
float x357 = x203[x347];
float x358 = x87[x350];
float x359 = x238[x352];
float x360 = x356 + x359;
x94[x350] = x360;
x345 += 1;
x343 += 1;
x344 += 1;

}
// add_cartesian
int32_t x368 = 0;
for(int x369=0; x369 < 5; x369++) {
for(int x370=0; x370 < 150; x370++) {
int32_t x371 = x368;
int32_t x372 = x371 + x370;
float x373 = x93[x372];
float x374 = x192[x370];
float x375 = x220[x369];
float x376 = x374 * x375;
float x377 = x373 + x376;
x93[x372] = x377;

}
x368 += 150;

}
int32_t x384 = 0;
for(int x385=0; x385 < 5; x385++) {
for(int x386=0; x386 < 150; x386++) {
float x387 = x201[x386];
int32_t x388 = x384;
int32_t x389 = x388 + x386;
float x390 = x78[x389];
float x391 = x220[x385];
float x392 = x390 * x391;
float x393 = x387 + x392;
x201[x386] = x393;

}
x384 += 150;

}
for(int x400=0; x400 < 150; x400++) {
float x401 = x191[x400];
float x402 = x192[x400];
float x405 = x201[x400];
float x403 = x402 * x402;
float x404 = 1.0f - x403;
float x406 = x404 * x405;
float x407 = x401 + x406;
x191[x400] = x407;

}
int32_t x411 = 0;
int32_t x412 = 0;
int32_t x413 = 0;
for(int x414=0; x414 < 150; x414++) {
int32_t x415 = x411;
float x416 = x173[x415];
float x417 = x156[x415];
int32_t x418 = x412;
float x419 = x59[x418];
int32_t x420 = x413;
float x421 = x191[x420];
float x422 = x416 + x421;
x173[x415] = x422;
float x424 = x89[x418];
float x425 = x156[x415];
float x426 = x59[x418];
float x427 = x191[x420];
float x428 = x424 + x427;
x89[x418] = x428;
x413 += 1;
x411 += 1;
x412 += 1;

}
// add_cartesian
int32_t x436 = 0;
for(int x437=0; x437 < 150; x437++) {
for(int x438=0; x438 < 300; x438++) {
int32_t x439 = x436;
int32_t x440 = x439 + x438;
float x441 = x88[x440];
float x442 = x153[x438];
float x443 = x173[x437];
float x444 = x442 * x443;
float x445 = x441 + x444;
x88[x440] = x445;

}
x436 += 300;

}
int32_t x452 = 0;
for(int x453=0; x453 < 150; x453++) {
for(int x454=0; x454 < 300; x454++) {
float x455 = x154[x454];
int32_t x456 = x452;
int32_t x457 = x456 + x454;
float x458 = x50[x457];
float x459 = x173[x453];
float x460 = x458 * x459;
float x461 = x455 + x460;
x154[x454] = x461;

}
x452 += 300;

}
} else {
// dot: List(150, 150), WrappedArray(150)
float* x470 = (float*)myMalloc(150 * sizeof(float));;
for(int x471=0; x471 < 150; x471++) {
float x472 = 0.0f;
int32_t x474 = x471 * 150;
for(int x473=0; x473 < 150; x473++) {
int32_t x475 = x474 + x473;
float x476 = x60[x475];
float x477 = x141[x473];
float x478 = x476 * x477;
x472 += x478;

}
float x482 = x472;
x470[x471] = x482;

}
float* x486 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(150, 150), WrappedArray(150)
float* x488 = (float*)myMalloc(150 * sizeof(float));;
for(int x489=0; x489 < 150; x489++) {
float x490 = 0.0f;
int32_t x492 = x489 * 150;
for(int x491=0; x491 < 150; x491++) {
int32_t x493 = x492 + x491;
float x494 = x69[x493];
float x495 = x148[x491];
float x496 = x494 * x495;
x490 += x496;

}
float x500 = x490;
x488[x489] = x500;

}
float* x504 = (float*)myMalloc(150 * sizeof(float));;
float* x505 = (float*)myMalloc(150 * sizeof(float));;
int32_t x506 = 0;
int32_t x507 = 0;
int32_t x508 = 0;
for(int x509=0; x509 < 150; x509++) {
int32_t x510 = x506;
int32_t x511 = x507;
float x512 = x470[x511];
int32_t x513 = x508;
float x514 = x488[x513];
float x515 = x512 + x514;
x505[x510] = x515;
x506 += 1;
x507 += 1;
x508 += 1;

}
float* x522 = (float*)myMalloc(150 * sizeof(float));;
float* x523 = (float*)myMalloc(150 * sizeof(float));;
int32_t x524 = 0;
int32_t x525 = 0;
int32_t x526 = 0;
for(int x527=0; x527 < 150; x527++) {
int32_t x528 = x524;
int32_t x529 = x525;
float x530 = x505[x529];
int32_t x531 = x526;
float x532 = x77[x531];
float x533 = x530 + x532;
x523[x528] = x533;
x524 += 1;
x525 += 1;
x526 += 1;

}
float* x540 = (float*)myMalloc(150 * sizeof(float));;
float* x541 = (float*)myMalloc(150 * sizeof(float));;
for(int x542=0; x542 < 150; x542++) {
float x543 = x523[x542];
double x544 = (double)x543;
double x545 = tanh(x544);
float x546 = (float)x545;
x541[x542] = x546;

}
float* x550 = (float*)myMalloc(150 * sizeof(float));;
// dot: List(5, 150), List(150)
float* x552 = (float*)myMalloc(5 * sizeof(float));;
for(int x553=0; x553 < 5; x553++) {
float x554 = 0.0f;
int32_t x556 = x553 * 150;
for(int x555=0; x555 < 150; x555++) {
int32_t x557 = x556 + x555;
float x558 = x78[x557];
float x559 = x541[x555];
float x560 = x558 * x559;
x554 += x560;

}
float x564 = x554;
x552[x553] = x564;

}
float* x568 = (float*)myMalloc(5 * sizeof(float));;
float* x569 = (float*)myMalloc(5 * sizeof(float));;
int32_t x570 = 0;
int32_t x571 = 0;
int32_t x572 = 0;
for(int x573=0; x573 < 5; x573++) {
int32_t x574 = x570;
int32_t x575 = x571;
float x576 = x552[x575];
int32_t x577 = x572;
float x578 = x87[x577];
float x579 = x576 + x578;
x569[x574] = x579;
x570 += 1;
x571 += 1;
x572 += 1;

}
float* x586 = (float*)myMalloc(5 * sizeof(float));;
float* x587 = (float*)myMalloc(1 * sizeof(float));;
int32_t x588 = 0;
int32_t x589 = 0;
int32_t x590 = 0;
int32_t x591 = x588;
int32_t x592 = x589;
float x593 = x139[x592];
int32_t x594 = x590;
float x595 = x146[x594];
float x596 = x593 + x595;
x587[x591] = x596;
x588 += 1;
float* x599 = (float*)myMalloc(1 * sizeof(float));;
float x600 = -3.4028235E38f;
for(int x601=0; x601 < 5; x601++) {
float x602 = x600;
float x603 = x569[x601];
bool x604 = x603 > x602;
float x605;
if (x604) {
x605 = x603;
} else {
x605 = x602;
}
x600 = x605;

}
float x609 = x600;
float x610 = 0.0f;
for(int x611=0; x611 < 5; x611++) {
float x612 = x610;
float x613 = x569[x611];
float x614 = x600;
float x615 = x613 - x614;
double x616 = (double)x615;
double x617 = exp(x616);
float x618 = (float)x617;
float x619 = x612 + x618;
x610 = x619;

}
float x623 = x610;
float* x628 = (float*)myMalloc(5 * sizeof(float));;
double x624 = (double)x623;
double x625 = log(x624);
float x626 = (float)x625;
float x627 = x609 + x626;
for(int x629=0; x629 < 5; x629++) {
float x630 = x569[x629];
float x631 = x630 - x627;
x628[x629] = x631;

}
float* x635 = (float*)myMalloc(5 * sizeof(float));;
int32_t x636 = x110[x128];
float x637 = x628[x636];
float* x639 = (float*)myMalloc(1 * sizeof(float));;
float x638 = -1.0f * x637;
x639[0] = x638;
float* x641 = (float*)myMalloc(1 * sizeof(float));;
float* x642 = (float*)myMalloc(1 * sizeof(float));;
int32_t x643 = 0;
int32_t x644 = 0;
int32_t x645 = 0;
int32_t x646 = x643;
int32_t x647 = x644;
float x648 = x587[x647];
int32_t x649 = x645;
float x650 = x639[x649];
float x651 = x648 + x650;
x642[x646] = x651;
x643 += 1;
float* x654 = (float*)myMalloc(1 * sizeof(float));;
float** x681 = (float**)myMalloc(4 * sizeof(float*));;
x681[0] = x642;
x681[1] = x654;
x681[2] = x541;
x681[3] = x550;
x655(x681);
int32_t x687 = 0;
int32_t x688 = 0;
int32_t x689 = 0;
int32_t x690 = x687;
float x691 = x599[x690];
float x692 = x587[x690];
int32_t x693 = x688;
float x694 = x639[x693];
int32_t x695 = x689;
float x696 = x654[x695];
float x697 = x691 + x696;
x599[x690] = x697;
float x699 = x641[x693];
float x700 = x587[x690];
float x701 = x639[x693];
float x702 = x654[x695];
float x703 = x699 + x702;
x641[x693] = x703;
x689 += 1;
float x706 = x635[x636];
float x707 = x641[0];
float x708 = -1.0f * x707;
float x709 = x706 + x708;
x635[x636] = x709;
float x711 = 0.0f;
for(int x712=0; x712 < 5; x712++) {
float x713 = x711;
float x714 = x635[x712];
float x715 = x713 + x714;
x711 = x715;

}
float x719 = x711;
float* x720 = (float*)myMalloc(1 * sizeof(float));;
x720[0] = x719;
float x722 = x720[0];
for(int x723=0; x723 < 5; x723++) {
float x724 = x586[x723];
float x725 = x635[x723];
float x726 = x628[x723];
double x727 = (double)x726;
double x728 = exp(x727);
float x729 = (float)x728;
float x730 = x729 * x722;
float x731 = x725 - x730;
float x732 = x724 + x731;
x586[x723] = x732;

}
int32_t x736 = 0;
int32_t x737 = 0;
int32_t x738 = 0;
int32_t x739 = x736;
float x740 = x140[x739];
float x741 = x139[x739];
int32_t x742 = x737;
float x743 = x146[x742];
int32_t x744 = x738;
float x745 = x599[x744];
float x746 = x740 + x745;
x140[x739] = x746;
float x748 = x147[x742];
float x749 = x139[x739];
float x750 = x146[x742];
float x751 = x599[x744];
float x752 = x748 + x751;
x147[x742] = x752;
x738 += 1;
int32_t x755 = 0;
int32_t x756 = 0;
int32_t x757 = 0;
for(int x758=0; x758 < 5; x758++) {
int32_t x759 = x755;
float x760 = x568[x759];
float x761 = x552[x759];
int32_t x762 = x756;
float x763 = x87[x762];
int32_t x764 = x757;
float x765 = x586[x764];
float x766 = x760 + x765;
x568[x759] = x766;
float x768 = x94[x762];
float x769 = x552[x759];
float x770 = x87[x762];
float x771 = x586[x764];
float x772 = x768 + x771;
x94[x762] = x772;
x757 += 1;
x755 += 1;
x756 += 1;

}
// add_cartesian
int32_t x780 = 0;
for(int x781=0; x781 < 5; x781++) {
for(int x782=0; x782 < 150; x782++) {
int32_t x783 = x780;
int32_t x784 = x783 + x782;
float x785 = x93[x784];
float x786 = x541[x782];
float x787 = x568[x781];
float x788 = x786 * x787;
float x789 = x785 + x788;
x93[x784] = x789;

}
x780 += 150;

}
int32_t x796 = 0;
for(int x797=0; x797 < 5; x797++) {
for(int x798=0; x798 < 150; x798++) {
float x799 = x550[x798];
int32_t x800 = x796;
int32_t x801 = x800 + x798;
float x802 = x78[x801];
float x803 = x568[x797];
float x804 = x802 * x803;
float x805 = x799 + x804;
x550[x798] = x805;

}
x796 += 150;

}
for(int x812=0; x812 < 150; x812++) {
float x813 = x540[x812];
float x814 = x541[x812];
float x817 = x550[x812];
float x815 = x814 * x814;
float x816 = 1.0f - x815;
float x818 = x816 * x817;
float x819 = x813 + x818;
x540[x812] = x819;

}
int32_t x823 = 0;
int32_t x824 = 0;
int32_t x825 = 0;
for(int x826=0; x826 < 150; x826++) {
int32_t x827 = x823;
float x828 = x522[x827];
float x829 = x505[x827];
int32_t x830 = x824;
float x831 = x77[x830];
int32_t x832 = x825;
float x833 = x540[x832];
float x834 = x828 + x833;
x522[x827] = x834;
float x836 = x92[x830];
float x837 = x505[x827];
float x838 = x77[x830];
float x839 = x540[x832];
float x840 = x836 + x839;
x92[x830] = x840;
x825 += 1;
x823 += 1;
x824 += 1;

}
int32_t x847 = 0;
int32_t x848 = 0;
int32_t x849 = 0;
for(int x850=0; x850 < 150; x850++) {
int32_t x851 = x847;
float x852 = x486[x851];
float x853 = x470[x851];
int32_t x854 = x848;
float x855 = x488[x854];
int32_t x856 = x849;
float x857 = x522[x856];
float x858 = x852 + x857;
x486[x851] = x858;
float x860 = x504[x854];
float x861 = x470[x851];
float x862 = x488[x854];
float x863 = x522[x856];
float x864 = x860 + x863;
x504[x854] = x864;
x849 += 1;
x847 += 1;
x848 += 1;

}
// add_cartesian
int32_t x872 = 0;
for(int x873=0; x873 < 150; x873++) {
for(int x874=0; x874 < 150; x874++) {
int32_t x875 = x872;
int32_t x876 = x875 + x874;
float x877 = x91[x876];
float x878 = x148[x874];
float x879 = x504[x873];
float x880 = x878 * x879;
float x881 = x877 + x880;
x91[x876] = x881;

}
x872 += 150;

}
int32_t x888 = 0;
for(int x889=0; x889 < 150; x889++) {
for(int x890=0; x890 < 150; x890++) {
float x891 = x149[x890];
int32_t x892 = x888;
int32_t x893 = x892 + x890;
float x894 = x69[x893];
float x895 = x504[x889];
float x896 = x894 * x895;
float x897 = x891 + x896;
x149[x890] = x897;

}
x888 += 150;

}
// add_cartesian
int32_t x905 = 0;
for(int x906=0; x906 < 150; x906++) {
for(int x907=0; x907 < 150; x907++) {
int32_t x908 = x905;
int32_t x909 = x908 + x907;
float x910 = x90[x909];
float x911 = x141[x907];
float x912 = x486[x906];
float x913 = x911 * x912;
float x914 = x910 + x913;
x90[x909] = x914;

}
x905 += 150;

}
int32_t x921 = 0;
for(int x922=0; x922 < 150; x922++) {
for(int x923=0; x923 < 150; x923++) {
float x924 = x142[x923];
int32_t x925 = x921;
int32_t x926 = x925 + x923;
float x927 = x60[x926];
float x928 = x486[x922];
float x929 = x927 * x928;
float x930 = x924 + x929;
x142[x923] = x930;

}
x921 += 150;

}
}
};
x124(x143,x144,x940);
};
x124(x136,x137,x948);
} else {
float** x969 = (float**)myMalloc(4 * sizeof(float*));;
x969[0] = x120;
x969[1] = x121;
x969[2] = x122;
x969[3] = x123;
function<void(float**)> x129 = x126;
function<void(float**)> x956 = [&](float** x957) {
float* x958 = x957[0];
float* x959 = x957[1];
float* x960 = x957[2];
float* x961 = x957[3];
float** x962 = (float**)myMalloc(4 * sizeof(float*));;
x962[0] = x958;
x962[1] = x959;
x962[2] = x960;
x962[3] = x961;
x129(x962);
};
x956(x969);
}
};
float* x117 = (float*)myMalloc(1 * sizeof(float));;
float* x118 = (float*)myMalloc(1 * sizeof(float));;
float* x119 = (float*)myMalloc(1 * sizeof(float));;
float** x989 = (float**)myMalloc(4 * sizeof(float*));;
x989[0] = x120;
x989[1] = x121;
x989[2] = x122;
x989[3] = x123;
function<void(float**)> x978 = [&](float** x979) {
float* x980 = x979[0];
float* x981 = x979[1];
float* x982 = x979[2];
float* x983 = x979[3];
float x984 = x981[0];
x981[0] = 1.0f;
float x986 = x980[0];
x119[0] = x986;
};
x124(0,x978,x989);
float x996 = x119[0];
float x997 = x107;
float x998 = (float)x108;
float x999 = x997 * x998;
int32_t x1000 = x108 + 1;
float x1001 = (float)x1000;
float x1002 = x999 / x1001;
float x1003 = x996 / x1001;
float x1004 = x1002 + x1003;
x107 = x1004;
for(int x1006=0; x1006 < 45000; x1006++) {
float x1007 = x88[x1006];
float x1008 = x1007;
float x1009 = x95[x1006];
float x1010 = x1008;
float x1011 = x1010 * x1010;
float x1012 = x1009 + x1011;
x95[x1006] = x1012;
float x1014 = x50[x1006];
float x1016 = x95[x1006];
float x1015 = 0.05f * x1010;
double x1017 = (double)x1016;
double x1018 = x1017 + 9.99999993922529E-9;
double x1019 = sqrt(x1018);
float x1020 = (float)x1019;
float x1021 = x1015 / x1020;
float x1022 = x1014 - x1021;
x50[x1006] = x1022;
x88[x1006] = 0.0f;

}
for(int x1027=0; x1027 < 150; x1027++) {
float x1028 = x89[x1027];
float x1029 = x1028;
float x1030 = x96[x1027];
float x1031 = x1029;
float x1032 = x1031 * x1031;
float x1033 = x1030 + x1032;
x96[x1027] = x1033;
float x1035 = x59[x1027];
float x1037 = x96[x1027];
float x1036 = 0.05f * x1031;
double x1038 = (double)x1037;
double x1039 = x1038 + 9.99999993922529E-9;
double x1040 = sqrt(x1039);
float x1041 = (float)x1040;
float x1042 = x1036 / x1041;
float x1043 = x1035 - x1042;
x59[x1027] = x1043;
x89[x1027] = 0.0f;

}
for(int x1048=0; x1048 < 22500; x1048++) {
float x1049 = x90[x1048];
float x1050 = x1049;
float x1051 = x97[x1048];
float x1052 = x1050;
float x1053 = x1052 * x1052;
float x1054 = x1051 + x1053;
x97[x1048] = x1054;
float x1056 = x60[x1048];
float x1058 = x97[x1048];
float x1057 = 0.05f * x1052;
double x1059 = (double)x1058;
double x1060 = x1059 + 9.99999993922529E-9;
double x1061 = sqrt(x1060);
float x1062 = (float)x1061;
float x1063 = x1057 / x1062;
float x1064 = x1056 - x1063;
x60[x1048] = x1064;
x90[x1048] = 0.0f;

}
for(int x1069=0; x1069 < 22500; x1069++) {
float x1070 = x91[x1069];
float x1071 = x1070;
float x1072 = x98[x1069];
float x1073 = x1071;
float x1074 = x1073 * x1073;
float x1075 = x1072 + x1074;
x98[x1069] = x1075;
float x1077 = x69[x1069];
float x1079 = x98[x1069];
float x1078 = 0.05f * x1073;
double x1080 = (double)x1079;
double x1081 = x1080 + 9.99999993922529E-9;
double x1082 = sqrt(x1081);
float x1083 = (float)x1082;
float x1084 = x1078 / x1083;
float x1085 = x1077 - x1084;
x69[x1069] = x1085;
x91[x1069] = 0.0f;

}
for(int x1090=0; x1090 < 150; x1090++) {
float x1091 = x92[x1090];
float x1092 = x1091;
float x1093 = x99[x1090];
float x1094 = x1092;
float x1095 = x1094 * x1094;
float x1096 = x1093 + x1095;
x99[x1090] = x1096;
float x1098 = x77[x1090];
float x1100 = x99[x1090];
float x1099 = 0.05f * x1094;
double x1101 = (double)x1100;
double x1102 = x1101 + 9.99999993922529E-9;
double x1103 = sqrt(x1102);
float x1104 = (float)x1103;
float x1105 = x1099 / x1104;
float x1106 = x1098 - x1105;
x77[x1090] = x1106;
x92[x1090] = 0.0f;

}
for(int x1111=0; x1111 < 750; x1111++) {
float x1112 = x93[x1111];
float x1113 = x1112;
float x1114 = x100[x1111];
float x1115 = x1113;
float x1116 = x1115 * x1115;
float x1117 = x1114 + x1116;
x100[x1111] = x1117;
float x1119 = x78[x1111];
float x1121 = x100[x1111];
float x1120 = 0.05f * x1115;
double x1122 = (double)x1121;
double x1123 = x1122 + 9.99999993922529E-9;
double x1124 = sqrt(x1123);
float x1125 = (float)x1124;
float x1126 = x1120 / x1125;
float x1127 = x1119 - x1126;
x78[x1111] = x1127;
x93[x1111] = 0.0f;

}
for(int x1132=0; x1132 < 5; x1132++) {
float x1133 = x94[x1132];
float x1134 = x1133;
float x1135 = x101[x1132];
float x1136 = x1134;
float x1137 = x1136 * x1136;
float x1138 = x1135 + x1137;
x101[x1132] = x1138;
float x1140 = x87[x1132];
float x1142 = x101[x1132];
float x1141 = 0.05f * x1136;
double x1143 = (double)x1142;
double x1144 = x1143 + 9.99999993922529E-9;
double x1145 = sqrt(x1144);
float x1146 = (float)x1145;
float x1147 = x1141 / x1146;
float x1148 = x1140 - x1147;
x87[x1132] = x1148;
x94[x1132] = 0.0f;

}
int64_t x1153 = (long)mallocAddr;
int64_t x1154 = x1153 - x103;
memset((void*)x103, 0, x1154);
mallocAddr = (void*)x103;

}
float x1159 = x107;
double x1160 = (double)x1159;
x102[x106] = x1160;
double x1162 = ((double)clock() / CLOCKS_PER_SEC);
double x1163 = x1162 - x104;
printf("epoc %d, average_loss %f, time %lf\n",x106,x1159,x1163);

}
double x1167 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1171 = (long)fopen(x0, "w");
fprintf((FILE *)x1171, "unit: %s\n", "1 epoch");
for(int x1173=0; x1173 < 6; x1173++) {
double x1174 = x102[x1173];
fprintf((FILE *)x1171, "%lf\n", x1174);

}
double x1168 = x104 - x2;
double x1169 = x1167 - x104;
double x1170 = x1169 / 6.0;
fprintf((FILE *)x1171, "run time: %lf %lf\n", x1168, x1170);
fclose((FILE*)x1171);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

