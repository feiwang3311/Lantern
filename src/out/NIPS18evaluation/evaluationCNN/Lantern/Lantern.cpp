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
void *mallocBase = malloc(HEAP_SIZE);
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
srand(42);
struct timeval begin_0, end_0, diff_0;
gettimeofday(&begin_0, NULL);
float* x4 = (float*)myMalloc(250 * sizeof(float));;
for(int x6=0; x6 < 250; x6++) {
float x7 = (float)rand()/RAND_MAX;
float x8 = x7 - 0.5f;
float x9 = x8 * 0.2f;
x4[x6] = x9;

}
float* x13 = (float*)myMalloc(250 * sizeof(float));;
for(int x14=0; x14 < 250; x14++) {
x13[x14] = 0.0f;

}
float* x18 = (float*)myMalloc(10 * sizeof(float));;
for(int x20=0; x20 < 10; x20++) {
x18[x20] = 0.0f;

}
float* x24 = (float*)myMalloc(10 * sizeof(float));;
for(int x25=0; x25 < 10; x25++) {
x24[x25] = 0.0f;

}
float* x29 = (float*)myMalloc(5000 * sizeof(float));;
for(int x31=0; x31 < 5000; x31++) {
float x32 = (float)rand()/RAND_MAX;
float x33 = x32 - 0.5f;
float x34 = x33 * 0.06324556f;
x29[x31] = x34;

}
float* x38 = (float*)myMalloc(5000 * sizeof(float));;
for(int x39=0; x39 < 5000; x39++) {
x38[x39] = 0.0f;

}
float* x43 = (float*)myMalloc(20 * sizeof(float));;
for(int x45=0; x45 < 20; x45++) {
x43[x45] = 0.0f;

}
float* x49 = (float*)myMalloc(20 * sizeof(float));;
for(int x50=0; x50 < 20; x50++) {
x49[x50] = 0.0f;

}
float* x54 = (float*)myMalloc(16000 * sizeof(float));;
for(int x56=0; x56 < 16000; x56++) {
float x57 = (float)rand()/RAND_MAX;
float x58 = x57 - 0.5f;
float x59 = x58 * 0.0559017f;
x54[x56] = x59;

}
float* x63 = (float*)myMalloc(16000 * sizeof(float));;
for(int x64=0; x64 < 16000; x64++) {
x63[x64] = 0.0f;

}
float* x68 = (float*)myMalloc(50 * sizeof(float));;
for(int x70=0; x70 < 50; x70++) {
x68[x70] = 0.0f;

}
float* x74 = (float*)myMalloc(50 * sizeof(float));;
for(int x75=0; x75 < 50; x75++) {
x74[x75] = 0.0f;

}
float* x79 = (float*)myMalloc(500 * sizeof(float));;
for(int x81=0; x81 < 500; x81++) {
float x82 = (float)rand()/RAND_MAX;
float x83 = x82 - 0.5f;
float x84 = x83 * 0.14142136f;
x79[x81] = x84;

}
float* x88 = (float*)myMalloc(500 * sizeof(float));;
for(int x89=0; x89 < 500; x89++) {
x88[x89] = 0.0f;

}
float* x93 = (float*)myMalloc(10 * sizeof(float));;
for(int x94=0; x94 < 10; x94++) {
x93[x94] = 0.0f;

}
float* x98 = (float*)myMalloc(10 * sizeof(float));;
for(int x99=0; x99 < 10; x99++) {
x98[x99] = 0.0f;

}
int64_t* x103 = (int64_t*)myMalloc(2 * sizeof(int64_t));;
int64_t* x104 = (int64_t*)myMalloc(2 * sizeof(int64_t));;
int32_t x115 = 0;
int32_t x116 = x115;
int32_t x117 = x116;
int32_t x111 = open("../data/bin/mnist_train_target.bin",0);
int32_t x112 = fsize(x111);
int32_t x114 = x112 / 4;
int* x113 = (int*)mmap(0, x112, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x111, 0);
int32_t x105 = open("../data/bin/mnist_train.bin",0);
int32_t x106 = fsize(x105);
float* x107 = (float*)mmap(0, x106, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x105, 0);
for(int x119=0; x119 < x114; x119++) {
int32_t x120 = x117;
int32_t x122 = x113[x119];
float* x121 = x107+x120;
for(int x124=0; x124 < 784; x124++) {
float x125 = x121[x124];
float x126 = x125 - 0.1307f;
float x127 = x126 / 0.3081f;
x121[x124] = x127;

}
x117 += 784;

}
int32_t x134 = x117;
int64_t x108 = (int64_t)x106;
int64_t x109 = x108 / 4LL;
int32_t x110 = (int32_t)x109;
bool x135 = x134 == x110;
if (x135) {
} else {
printf("Data length doesn't match\n");
exit(0);
}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x143 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x144 = (float)x143;
float x145 = x144 / 1000000.0f;
printf("Data normalized (all prepare time) in %lf sec\n",x145);
double* x147 = (double*)myMalloc(10 * sizeof(double));;
int64_t x148 = (long)mallocAddr;
int32_t x163 = x114 / 100;
int32_t x1427 = x114 / 10;
double x1432 = (double)x114;
int64_t x1449 = (int64_t)x114;
float x1453 = (float)x114;
for(int x149=0; x149 < 10; x149++) {
struct timeval begin_1, end_1, diff_1;
int32_t x151 = 0;
int32_t x152 = x151;
int32_t x153 = x152;
float x154 = 0.0f;
float x155 = x154;
float x156 = x155;
int32_t x157 = x149 + 1;
printf("Start training epoch %d\n",x157);
gettimeofday(&begin_1, NULL);
int32_t x160 = 0;
int32_t x161 = x160;
int32_t x162 = x161;
for(int x165=0; x165 < x163; x165++) {
int32_t x166 = x162;
x153 += 100;
float* x171 = (float*)myMalloc(1 * sizeof(float));;
x171[0] = 0.0f;
float* x173 = (float*)myMalloc(1 * sizeof(float));;
x173[0] = 0.0f;
float* x175 = (float*)myMalloc(1 * sizeof(float));;
for(int x177=0; x177 < 1; x177++) {
x175[x177] = 0.0f;

}
float* x181 = (float*)myMalloc(1 * sizeof(float));;
for(int x182=0; x182 < 1; x182++) {
x181[x182] = 0.0f;

}
float* x186 = (float*)myMalloc(576000 * sizeof(float));;
int32_t x187 = 0;
for(int x189=0; x189 < 100; x189++) {
for(int x190=0; x190 < 10; x190++) {
for(int x192=0; x192 < 576; x192++) {
int32_t x193 = x187;
float x194 = x18[x190];
x186[x193] = x194;
x187 += 1;

}

}

}
float* x167 = x107+x166;
for(int x203=0; x203 < 100; x203++) {
int32_t x206 = x203 * 5760;
float* x207 = x186+x206;
int32_t x208 = 0;
int32_t x209 = 0;
int32_t x204 = x203 * 784;
float* x205 = x167+x204;
for(int x210=0; x210 < 10; x210++) {
int32_t x211 = x209;
int32_t x212 = x211;
int32_t x213 = 0;
int32_t x214 = x208;
float* x215 = x207+x214;
int32_t x216 = x213;
int32_t x218 = x212;
float* x219 = x4+x218;
int32_t x220 = 0;
int32_t x221 = 0;
float* x217 = x205+x216;
for(int x223=0; x223 < 24; x223++) {
int32_t x224 = x221;
int32_t x225 = x224;
for(int x226=0; x226 < 24; x226++) {
float x227 = 0.0f;
int32_t x228 = 0;
int32_t x229 = x225;
int32_t x230 = x229;
for(int x232=0; x232 < 5; x232++) {
int32_t x233 = x230;
int32_t x235 = x228;
float* x236 = x219+x235;
float* x234 = x217+x233;
for(int x237=0; x237 < 5; x237++) {
float x238 = x234[x237];
float x239 = x236[x237];
float x240 = x238 * x239;
x227 += x240;

}
x228 += 5;
x230 += 28;

}
int32_t x248 = x220;
float x249 = x215[x248];
float x250 = x227;
float x251 = x249 + x250;
x215[x248] = x251;
x220 += 1;
x225 += 1;

}
x221 += 28;

}
x212 += 25;
x213 += 784;
x209 += 25;
x208 += 576;

}

}
float* x268 = (float*)myMalloc(576000 * sizeof(float));;
for(int x270=0; x270 < 576000; x270++) {
x268[x270] = 0.0f;

}
float* x274 = (float*)myMalloc(576000 * sizeof(float));;
for(int x275=0; x275 < 576000; x275++) {
float x276 = x186[x275];
bool x277 = x276 < 0.0f;
if (x277) {
x274[x275] = 0.0f;
} else {
float x280 = x186[x275];
x274[x275] = x280;
}

}
float* x286 = (float*)myMalloc(576000 * sizeof(float));;
for(int x287=0; x287 < 576000; x287++) {
x286[x287] = 0.0f;

}
float* x291 = (float*)myMalloc(144000 * sizeof(float));;
for(int x293=0; x293 < 144000; x293++) {
x291[x293] = -3.4028235E38f;

}
int* x297 = (int32_t*)myMalloc(144000 * sizeof(int32_t));;
for(int x298=0; x298 < 100; x298++) {
int32_t x299 = x298 * 5760;
float* x300 = x274+x299;
int32_t x301 = x298 * 1440;
float* x302 = x291+x301;
int* x303 = x297+x301;
int32_t x304 = 0;
int32_t x305 = 0;
for(int x306=0; x306 < 10; x306++) {
int32_t x307 = x304;
int32_t x308 = x307;
int32_t x309 = x305;
int32_t x310 = x309;
for(int x312=0; x312 < 12; x312++) {
int32_t x313 = x308;
int32_t x314 = x313;
int32_t x315 = x310;
int32_t x316 = x315;
for(int x317=0; x317 < 12; x317++) {
int32_t x318 = x316;
int32_t x319 = x318;
int32_t x320 = x319;
int32_t x321 = x320;
int32_t x322 = x321;
float x323 = x300[x322];
int32_t x324 = x314;
float x325 = x302[x324];
bool x326 = x323 > x325;
if (x326) {
float x327 = x300[x322];
x302[x324] = x327;
int32_t x329 = x322 + x299;
x303[x324] = x329;
} else {
}
x321 += 1;
int32_t x334 = x321;
float x335 = x300[x334];
float x336 = x302[x324];
bool x337 = x335 > x336;
if (x337) {
float x338 = x300[x334];
x302[x324] = x338;
int32_t x340 = x334 + x299;
x303[x324] = x340;
} else {
}
x321 += 1;
x319 += 24;
int32_t x346 = x319;
int32_t x347 = x346;
int32_t x348 = x347;
float x349 = x300[x348];
float x350 = x302[x324];
bool x351 = x349 > x350;
if (x351) {
float x352 = x300[x348];
x302[x324] = x352;
int32_t x354 = x348 + x299;
x303[x324] = x354;
} else {
}
x347 += 1;
int32_t x359 = x347;
float x360 = x300[x359];
float x361 = x302[x324];
bool x362 = x360 > x361;
if (x362) {
float x363 = x300[x359];
x302[x324] = x363;
int32_t x365 = x359 + x299;
x303[x324] = x365;
} else {
}
x347 += 1;
x319 += 24;
x314 += 1;
x316 += 2;

}
x308 += 12;
x310 += 48;

}
x304 += 144;
x305 += 576;

}

}
float* x385 = (float*)myMalloc(144000 * sizeof(float));;
for(int x386=0; x386 < 144000; x386++) {
x385[x386] = 0.0f;

}
float* x390 = (float*)myMalloc(128000 * sizeof(float));;
int32_t x391 = 0;
for(int x392=0; x392 < 100; x392++) {
for(int x393=0; x393 < 20; x393++) {
for(int x395=0; x395 < 64; x395++) {
int32_t x396 = x391;
float x397 = x43[x393];
x390[x396] = x397;
x391 += 1;

}

}

}
for(int x406=0; x406 < 100; x406++) {
int32_t x407 = x406 * 1440;
float* x408 = x291+x407;
int32_t x409 = x406 * 1280;
float* x410 = x390+x409;
int32_t x411 = 0;
int32_t x412 = 0;
for(int x413=0; x413 < 20; x413++) {
int32_t x414 = x412;
int32_t x415 = x414;
int32_t x416 = 0;
int32_t x417 = x411;
float* x418 = x410+x417;
for(int x419=0; x419 < 10; x419++) {
int32_t x420 = x416;
float* x421 = x408+x420;
int32_t x422 = x415;
float* x423 = x29+x422;
int32_t x424 = 0;
int32_t x425 = 0;
for(int x427=0; x427 < 8; x427++) {
int32_t x428 = x425;
int32_t x429 = x428;
for(int x430=0; x430 < 8; x430++) {
float x431 = 0.0f;
int32_t x432 = 0;
int32_t x433 = x429;
int32_t x434 = x433;
for(int x435=0; x435 < 5; x435++) {
int32_t x436 = x434;
float* x437 = x421+x436;
int32_t x438 = x432;
float* x439 = x423+x438;
for(int x440=0; x440 < 5; x440++) {
float x441 = x437[x440];
float x442 = x439[x440];
float x443 = x441 * x442;
x431 += x443;

}
x432 += 5;
x434 += 12;

}
int32_t x451 = x424;
float x452 = x418[x451];
float x453 = x431;
float x454 = x452 + x453;
x418[x451] = x454;
x424 += 1;
x429 += 1;

}
x425 += 12;

}
x415 += 25;
x416 += 144;

}
x412 += 250;
x411 += 64;

}

}
float* x473 = (float*)myMalloc(128000 * sizeof(float));;
for(int x475=0; x475 < 128000; x475++) {
x473[x475] = 0.0f;

}
float* x479 = (float*)myMalloc(128000 * sizeof(float));;
for(int x480=0; x480 < 128000; x480++) {
float x481 = x390[x480];
bool x482 = x481 < 0.0f;
if (x482) {
x479[x480] = 0.0f;
} else {
float x485 = x390[x480];
x479[x480] = x485;
}

}
float* x491 = (float*)myMalloc(128000 * sizeof(float));;
for(int x492=0; x492 < 128000; x492++) {
x491[x492] = 0.0f;

}
float* x496 = (float*)myMalloc(32000 * sizeof(float));;
for(int x498=0; x498 < 32000; x498++) {
x496[x498] = -3.4028235E38f;

}
int* x502 = (int32_t*)myMalloc(32000 * sizeof(int32_t));;
for(int x503=0; x503 < 100; x503++) {
int32_t x504 = x503 * 1280;
float* x505 = x479+x504;
int32_t x506 = x503 * 320;
float* x507 = x496+x506;
int* x508 = x502+x506;
int32_t x509 = 0;
int32_t x510 = 0;
for(int x511=0; x511 < 20; x511++) {
int32_t x512 = x509;
int32_t x513 = x512;
int32_t x514 = x510;
int32_t x515 = x514;
for(int x517=0; x517 < 4; x517++) {
int32_t x518 = x513;
int32_t x519 = x518;
int32_t x520 = x515;
int32_t x521 = x520;
for(int x522=0; x522 < 4; x522++) {
int32_t x523 = x521;
int32_t x524 = x523;
int32_t x525 = x524;
int32_t x526 = x525;
int32_t x527 = x526;
float x528 = x505[x527];
int32_t x529 = x519;
float x530 = x507[x529];
bool x531 = x528 > x530;
if (x531) {
float x532 = x505[x527];
x507[x529] = x532;
int32_t x534 = x527 + x504;
x508[x529] = x534;
} else {
}
x526 += 1;
int32_t x539 = x526;
float x540 = x505[x539];
float x541 = x507[x529];
bool x542 = x540 > x541;
if (x542) {
float x543 = x505[x539];
x507[x529] = x543;
int32_t x545 = x539 + x504;
x508[x529] = x545;
} else {
}
x526 += 1;
x524 += 8;
int32_t x551 = x524;
int32_t x552 = x551;
int32_t x553 = x552;
float x554 = x505[x553];
float x555 = x507[x529];
bool x556 = x554 > x555;
if (x556) {
float x557 = x505[x553];
x507[x529] = x557;
int32_t x559 = x553 + x504;
x508[x529] = x559;
} else {
}
x552 += 1;
int32_t x564 = x552;
float x565 = x505[x564];
float x566 = x507[x529];
bool x567 = x565 > x566;
if (x567) {
float x568 = x505[x564];
x507[x529] = x568;
int32_t x570 = x564 + x504;
x508[x529] = x570;
} else {
}
x552 += 1;
x524 += 8;
x519 += 1;
x521 += 2;

}
x513 += 4;
x515 += 16;

}
x509 += 16;
x510 += 64;

}

}
float* x590 = (float*)myMalloc(32000 * sizeof(float));;
for(int x591=0; x591 < 32000; x591++) {
x590[x591] = 0.0f;

}
// dot: ArrayBuffer(100, 320), WrappedArray(320, 50)
float* x596 = (float*)myMalloc(5000 * sizeof(float));;
for(int x597=0; x597 < 100; x597++) {
int32_t x602 = x597 * 320;
int32_t x612 = x597 * 50;
for(int x598=0; x598 < 50; x598++) {
float x599 = 0.0f;
for(int x601=0; x601 < 320; x601++) {
int32_t x603 = x602 + x601;
float x604 = x496[x603];
int32_t x605 = x601 * 50;
int32_t x606 = x605 + x598;
float x607 = x54[x606];
float x608 = x604 * x607;
x599 += x608;

}
float x614 = x599;
int32_t x613 = x612 + x598;
x596[x613] = x614;

}

}
float* x620 = (float*)myMalloc(5000 * sizeof(float));;
for(int x621=0; x621 < 5000; x621++) {
x620[x621] = 0.0f;

}
float* x625 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x626 = 0;
int32_t x627 = 0;
int32_t x628 = 0;
for(int x629=0; x629 < 100; x629++) {
int32_t x630 = x627;
int32_t x631 = x628;
int32_t x632 = x626;
int32_t x633 = x632;
int32_t x634 = x630;
int32_t x635 = x631;
for(int x636=0; x636 < 50; x636++) {
int32_t x637 = x633;
int32_t x638 = x634;
float x639 = x596[x638];
int32_t x640 = x635;
float x641 = x68[x640];
float x642 = x639 + x641;
x625[x637] = x642;
x633 += 1;
x634 += 1;
x635 += 1;

}
x626 += 50;
x627 += 50;

}
float* x653 = (float*)myMalloc(5000 * sizeof(float));;
for(int x654=0; x654 < 5000; x654++) {
x653[x654] = 0.0f;

}
float* x658 = (float*)myMalloc(5000 * sizeof(float));;
float* x659 = (float*)myMalloc(5000 * sizeof(float));;
for(int x660=0; x660 < 5000; x660++) {
float x661 = (float)rand()/RAND_MAX;
bool x662 = x661 > 0.5f;
if (x662) {
float x663 = x625[x660];
float x664 = x663 * 2.0f;
x658[x660] = x664;
x659[x660] = 2.0f;
} else {
x658[x660] = 0.0f;
x659[x660] = 0.0f;
}

}
float* x674 = (float*)myMalloc(5000 * sizeof(float));;
for(int x675=0; x675 < 5000; x675++) {
x674[x675] = 0.0f;

}
// dot: List(100, 50), WrappedArray(50, 10)
float* x680 = (float*)myMalloc(1000 * sizeof(float));;
for(int x681=0; x681 < 100; x681++) {
int32_t x685 = x681 * 50;
int32_t x695 = x681 * 10;
for(int x682=0; x682 < 10; x682++) {
float x683 = 0.0f;
for(int x684=0; x684 < 50; x684++) {
int32_t x686 = x685 + x684;
float x687 = x658[x686];
int32_t x688 = x684 * 10;
int32_t x689 = x688 + x682;
float x690 = x79[x689];
float x691 = x687 * x690;
x683 += x691;

}
float x697 = x683;
int32_t x696 = x695 + x682;
x680[x696] = x697;

}

}
float* x703 = (float*)myMalloc(1000 * sizeof(float));;
for(int x705=0; x705 < 1000; x705++) {
x703[x705] = 0.0f;

}
float* x709 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x710 = 0;
int32_t x711 = 0;
int32_t x712 = 0;
for(int x713=0; x713 < 100; x713++) {
int32_t x714 = x711;
int32_t x715 = x712;
int32_t x716 = x710;
int32_t x717 = x716;
int32_t x718 = x714;
int32_t x719 = x715;
for(int x720=0; x720 < 10; x720++) {
int32_t x721 = x717;
int32_t x722 = x718;
float x723 = x680[x722];
int32_t x724 = x719;
float x725 = x93[x724];
float x726 = x723 + x725;
x709[x721] = x726;
x717 += 1;
x718 += 1;
x719 += 1;

}
x710 += 10;
x711 += 10;

}
float* x737 = (float*)myMalloc(1000 * sizeof(float));;
for(int x738=0; x738 < 1000; x738++) {
x737[x738] = 0.0f;

}
float* x742 = (float*)myMalloc(100 * sizeof(float));;
int32_t x743 = 0;
for(int x744=0; x744 < 100; x744++) {
float x745 = -3.4028235E38f;
for(int x746=0; x746 < 10; x746++) {
int32_t x747 = x743;
float x748 = x709[x747];
float x749 = x745;
bool x750 = x748 > x749;
if (x750) {
float x751 = x709[x747];
x745 = x751;
} else {
}
x743 += 1;

}
float x758 = x745;
x742[x744] = x758;

}
float* x762 = (float*)myMalloc(1000 * sizeof(float));;
for(int x763=0; x763 < 1000; x763++) {
x762[x763] = 0.0f;

}
int32_t x767 = 0;
for(int x768=0; x768 < 100; x768++) {
for(int x769=0; x769 < 10; x769++) {
int32_t x770 = x767;
float x771 = x709[x770];
float x772 = x742[x768];
float x773 = x771 - x772;
double x774 = (double)x773;
double x775 = exp(x774);
float x776 = (float)x775;
x762[x770] = x776;
x767 += 1;

}

}
float* x783 = (float*)myMalloc(100 * sizeof(float));;
for(int x784=0; x784 < 100; x784++) {
x783[x784] = 0.0f;

}
for(int x788=0; x788 < 100; x788++) {
int32_t x789 = x788;
int32_t x790 = x788 * 10;
int32_t x791 = x790;
for(int x792=0; x792 < 10; x792++) {
int32_t x793 = x789;
float x794 = x783[x793];
int32_t x795 = x791;
float x796 = x762[x795];
float x797 = x794 + x796;
x783[x793] = x797;
x791 += 1;

}

}
x767 = 0;
for(int x805=0; x805 < 100; x805++) {
float x806 = x742[x805];
float x807 = x783[x805];
double x808 = (double)x807;
double x809 = log(x808);
float x810 = (float)x809;
float x811 = x806 + x810;
for(int x812=0; x812 < 10; x812++) {
int32_t x813 = x767;
float x814 = x709[x813];
float x815 = x814 - x811;
x762[x813] = x815;
x767 += 1;

}

}
float* x822 = (float*)myMalloc(1000 * sizeof(float));;
for(int x823=0; x823 < 1000; x823++) {
x822[x823] = 0.0f;

}
float* x827 = (float*)myMalloc(100 * sizeof(float));;
int32_t x828 = 0;
int32_t x168 = x165 * 100;
int* x169 = x113+x168;
for(int x829=0; x829 < 100; x829++) {
int32_t x830 = x828;
int32_t x831 = x169[x829];
int32_t x832 = x830 + x831;
float x833 = x762[x832];
float x834 = -1.0f * x833;
x827[x829] = x834;
x828 += 10;

}
float* x839 = (float*)myMalloc(100 * sizeof(float));;
for(int x840=0; x840 < 100; x840++) {
x839[x840] = 0.0f;

}
float x844 = 0.0f;
for(int x845=0; x845 < 100; x845++) {
float x846 = x844;
float x847 = x827[x845];
float x848 = x846 + x847;
x844 = x848;

}
float x852 = x844;
float* x853 = (float*)myMalloc(1 * sizeof(float));;
x853[0] = x852;
float* x855 = (float*)myMalloc(1 * sizeof(float));;
for(int x856=0; x856 < 1; x856++) {
x855[x856] = 0.0f;

}
float x860 = x855[0];
x855[0] = 1.0f;
float x862 = x853[0];
x181[0] = x862;
// += tensor of dim 0
float x865 = x855[0];
for(int x866=0; x866 < 100; x866++) {
float x867 = x839[x866];
float x868 = x867 + x865;
x839[x866] = x868;

}
int32_t x872 = 0;
for(int x873=0; x873 < 100; x873++) {
int32_t x874 = x872;
int32_t x875 = x169[x873];
int32_t x876 = x874 + x875;
float x877 = x822[x876];
float x878 = x839[x873];
float x879 = -1.0f * x878;
float x880 = x877 + x879;
x822[x876] = x880;
x872 += 10;

}
float* x885 = (float*)myMalloc(100 * sizeof(float));;
for(int x886=0; x886 < 100; x886++) {
x885[x886] = 0.0f;

}
for(int x890=0; x890 < 100; x890++) {
int32_t x891 = x890;
int32_t x892 = x890 * 10;
int32_t x893 = x892;
for(int x894=0; x894 < 10; x894++) {
int32_t x895 = x891;
float x896 = x885[x895];
int32_t x897 = x893;
float x898 = x822[x897];
float x899 = x896 + x898;
x885[x895] = x899;
x893 += 1;

}

}
int32_t x906 = 0;
for(int x907=0; x907 < 100; x907++) {
for(int x908=0; x908 < 10; x908++) {
int32_t x909 = x906;
float x910 = x737[x909];
float x911 = x822[x909];
float x912 = x762[x909];
float x916 = x885[x907];
double x913 = (double)x912;
double x914 = exp(x913);
float x915 = (float)x914;
float x917 = x915 * x916;
float x918 = x911 - x917;
float x919 = x910 + x918;
x737[x909] = x919;
x906 += 1;

}

}
int32_t x926 = 0;
int32_t x927 = 0;
int32_t x928 = 0;
for(int x929=0; x929 < 100; x929++) {
int32_t x930 = x926;
int32_t x931 = x927;
int32_t x932 = x928;
int32_t x933 = x930;
int32_t x934 = x931;
int32_t x935 = x932;
for(int x936=0; x936 < 10; x936++) {
int32_t x937 = x933;
float x938 = x703[x937];
float x939 = x680[x937];
int32_t x940 = x934;
float x941 = x93[x940];
int32_t x942 = x935;
float x943 = x737[x942];
float x944 = x938 + x943;
x703[x937] = x944;
float x946 = x98[x940];
float x947 = x680[x937];
float x948 = x93[x940];
float x949 = x737[x942];
float x950 = x946 + x949;
x98[x940] = x950;
x935 += 1;
x933 += 1;
x934 += 1;

}
x928 += 10;
x926 += 10;

}
for(int x961=0; x961 < 100; x961++) {
int32_t x964 = x961 * 50;
int32_t x970 = x961 * 10;
for(int x962=0; x962 < 10; x962++) {
int32_t x971 = x970 + x962;
for(int x963=0; x963 < 50; x963++) {
int32_t x965 = x964 + x963;
float x966 = x674[x965];
int32_t x967 = x963 * 10;
int32_t x968 = x967 + x962;
float x969 = x79[x968];
float x972 = x703[x971];
float x973 = x969 * x972;
float x974 = x966 + x973;
x674[x965] = x974;
float x976 = x88[x968];
float x977 = x658[x965];
float x978 = x703[x971];
float x979 = x977 * x978;
float x980 = x976 + x979;
x88[x968] = x980;

}

}

}
float* x988 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x989 = 0;
int32_t x990 = 0;
int32_t x991 = 0;
for(int x992=0; x992 < 100; x992++) {
int32_t x993 = x990;
int32_t x994 = x991;
int32_t x995 = x989;
int32_t x996 = x995;
int32_t x997 = x993;
int32_t x998 = x994;
for(int x999=0; x999 < 50; x999++) {
int32_t x1000 = x996;
int32_t x1001 = x997;
float x1002 = x659[x1001];
int32_t x1003 = x998;
float x1004 = x674[x1003];
float x1005 = x1002 * x1004;
x988[x1000] = x1005;
x996 += 1;
x997 += 1;
x998 += 1;

}
x989 += 50;
x990 += 50;
x991 += 50;

}
for(int x1017=0; x1017 < 5000; x1017++) {
float x1018 = x653[x1017];
float x1019 = x988[x1017];
float x1020 = x1018 + x1019;
x653[x1017] = x1020;

}
int32_t x1024 = 0;
int32_t x1025 = 0;
int32_t x1026 = 0;
for(int x1027=0; x1027 < 100; x1027++) {
int32_t x1028 = x1024;
int32_t x1029 = x1025;
int32_t x1030 = x1026;
int32_t x1031 = x1028;
int32_t x1032 = x1029;
int32_t x1033 = x1030;
for(int x1034=0; x1034 < 50; x1034++) {
int32_t x1035 = x1031;
float x1036 = x620[x1035];
float x1037 = x596[x1035];
int32_t x1038 = x1032;
float x1039 = x68[x1038];
int32_t x1040 = x1033;
float x1041 = x653[x1040];
float x1042 = x1036 + x1041;
x620[x1035] = x1042;
float x1044 = x74[x1038];
float x1045 = x596[x1035];
float x1046 = x68[x1038];
float x1047 = x653[x1040];
float x1048 = x1044 + x1047;
x74[x1038] = x1048;
x1033 += 1;
x1031 += 1;
x1032 += 1;

}
x1026 += 50;
x1024 += 50;

}
for(int x1059=0; x1059 < 100; x1059++) {
int32_t x1062 = x1059 * 320;
int32_t x1068 = x1059 * 50;
for(int x1060=0; x1060 < 50; x1060++) {
int32_t x1069 = x1068 + x1060;
for(int x1061=0; x1061 < 320; x1061++) {
int32_t x1063 = x1062 + x1061;
float x1064 = x590[x1063];
int32_t x1065 = x1061 * 50;
int32_t x1066 = x1065 + x1060;
float x1067 = x54[x1066];
float x1070 = x620[x1069];
float x1071 = x1067 * x1070;
float x1072 = x1064 + x1071;
x590[x1063] = x1072;
float x1074 = x63[x1066];
float x1075 = x496[x1063];
float x1076 = x620[x1069];
float x1077 = x1075 * x1076;
float x1078 = x1074 + x1077;
x63[x1066] = x1078;

}

}

}
for(int x1086=0; x1086 < 32000; x1086++) {
int32_t x1087 = x502[x1086];
float x1088 = x590[x1086];
x491[x1087] = x1088;

}
for(int x1092=0; x1092 < 128000; x1092++) {
float x1093 = x390[x1092];
bool x1094 = x1093 < 0.0f;
float x1097;
if (x1094) {
x1097 = 0.0f;
} else {
float x1095 = x491[x1092];
x1097 = x1095;
}
x473[x1092] = x1097;

}
for(int x1101=0; x1101 < 100; x1101++) {
int32_t x1102 = x1101 * 1280;
int32_t x1103 = x1102;
int32_t x1104 = 0;
int32_t x1107 = x1101 * 1440;
for(int x1105=0; x1105 < 20; x1105++) {
float x1106 = 0.0f;
int32_t x1108 = x1107;
for(int x1109=0; x1109 < 8; x1109++) {
int32_t x1110 = x1108;
int32_t x1111 = x1110;
for(int x1112=0; x1112 < 8; x1112++) {
int32_t x1113 = x1103;
float x1114 = x473[x1113];
x1106 += x1114;
int32_t x1116 = x1111;
int32_t x1117 = x1116;
int32_t x1118 = x1104;
int32_t x1119 = x1118;
for(int x1120=0; x1120 < 10; x1120++) {
int32_t x1121 = x1117;
int32_t x1122 = x1121;
for(int x1123=0; x1123 < 5; x1123++) {
for(int x1124=0; x1124 < 5; x1124++) {
int32_t x1125 = x1122;
int32_t x1126 = x1125 + x1124;
float x1127 = x385[x1126];
int32_t x1128 = x1119;
float x1129 = x29[x1128];
float x1130 = x1114 * x1129;
float x1131 = x1127 + x1130;
x385[x1126] = x1131;
float x1133 = x38[x1128];
float x1134 = x291[x1126];
float x1135 = x1114 * x1134;
float x1136 = x1133 + x1135;
x38[x1128] = x1136;
x1119 += 1;

}
x1122 += 12;

}
x1117 += 144;

}
x1111 += 1;
x1103 += 1;

}
x1108 += 12;

}
float x1154 = x49[x1105];
float x1155 = x1106;
float x1156 = x1154 + x1155;
x49[x1105] = x1156;
x1104 += 250;

}

}
for(int x1163=0; x1163 < 144000; x1163++) {
int32_t x1164 = x297[x1163];
float x1165 = x385[x1163];
x286[x1164] = x1165;

}
for(int x1169=0; x1169 < 576000; x1169++) {
float x1170 = x186[x1169];
bool x1171 = x1170 < 0.0f;
float x1174;
if (x1171) {
x1174 = 0.0f;
} else {
float x1172 = x286[x1169];
x1174 = x1172;
}
x268[x1169] = x1174;

}
for(int x1178=0; x1178 < 100; x1178++) {
int32_t x1179 = x1178 * 5760;
int32_t x1180 = x1179;
int32_t x1181 = 0;
int32_t x1184 = x1178 * 784;
for(int x1182=0; x1182 < 10; x1182++) {
float x1183 = 0.0f;
int32_t x1185 = x1184;
for(int x1186=0; x1186 < 24; x1186++) {
int32_t x1187 = x1185;
int32_t x1188 = x1187;
for(int x1189=0; x1189 < 24; x1189++) {
int32_t x1190 = x1180;
float x1191 = x268[x1190];
x1183 += x1191;
int32_t x1193 = x1188;
int32_t x1194 = x1193;
int32_t x1195 = x1181;
int32_t x1196 = x1195;
int32_t x1197 = x1194;
int32_t x1198 = x1197;
for(int x1199=0; x1199 < 5; x1199++) {
for(int x1200=0; x1200 < 5; x1200++) {
int32_t x1201 = x1196;
float x1202 = x13[x1201];
int32_t x1203 = x1198;
int32_t x1204 = x1203 + x1200;
float x1205 = x167[x1204];
float x1206 = x1191 * x1205;
float x1207 = x1202 + x1206;
x13[x1201] = x1207;
x1196 += 1;

}
x1198 += 28;

}
x1194 += 784;
x1188 += 1;
x1180 += 1;

}
x1185 += 28;

}
float x1223 = x24[x1182];
float x1224 = x1183;
float x1225 = x1223 + x1224;
x24[x1182] = x1225;
x1181 += 25;

}

}
float x1232 = x181[0];
x156 += x1232;
for(int x1234=0; x1234 < 5000; x1234++) {
float x1235 = x38[x1234];
bool x1236 = x1235 > 1000.0f;
if (x1236) {
x38[x1234] = 1000.0f;
} else {
}
float x1240 = x38[x1234];
bool x1241 = x1240 < -1000.0f;
if (x1241) {
x38[x1234] = -1000.0f;
} else {
}

}
float* x1247 = (float*)myMalloc(5000 * sizeof(float));;
for(int x1248=0; x1248 < 5000; x1248++) {
float x1249 = x38[x1248];
float x1250 = x1249 * 5.0E-4f;
x1247[x1248] = x1250;

}
for(int x1254=0; x1254 < 5000; x1254++) {
float x1255 = x29[x1254];
float x1256 = x1247[x1254];
float x1257 = x1255 - x1256;
x29[x1254] = x1257;

}
for(int x1261=0; x1261 < 5000; x1261++) {
float x1262 = x38[x1261];
x38[x1261] = 0.0f;

}
for(int x1266=0; x1266 < 16000; x1266++) {
float x1267 = x63[x1266];
bool x1268 = x1267 > 1000.0f;
if (x1268) {
x63[x1266] = 1000.0f;
} else {
}
float x1272 = x63[x1266];
bool x1273 = x1272 < -1000.0f;
if (x1273) {
x63[x1266] = -1000.0f;
} else {
}

}
float* x1279 = (float*)myMalloc(16000 * sizeof(float));;
for(int x1280=0; x1280 < 16000; x1280++) {
float x1281 = x63[x1280];
float x1282 = x1281 * 5.0E-4f;
x1279[x1280] = x1282;

}
for(int x1286=0; x1286 < 16000; x1286++) {
float x1287 = x54[x1286];
float x1288 = x1279[x1286];
float x1289 = x1287 - x1288;
x54[x1286] = x1289;

}
for(int x1293=0; x1293 < 16000; x1293++) {
float x1294 = x63[x1293];
x63[x1293] = 0.0f;

}
for(int x1298=0; x1298 < 50; x1298++) {
float x1299 = x74[x1298];
bool x1300 = x1299 > 1000.0f;
if (x1300) {
x74[x1298] = 1000.0f;
} else {
}
float x1304 = x74[x1298];
bool x1305 = x1304 < -1000.0f;
if (x1305) {
x74[x1298] = -1000.0f;
} else {
}

}
float* x1311 = (float*)myMalloc(50 * sizeof(float));;
for(int x1312=0; x1312 < 50; x1312++) {
float x1313 = x74[x1312];
float x1314 = x1313 * 5.0E-4f;
x1311[x1312] = x1314;

}
for(int x1318=0; x1318 < 50; x1318++) {
float x1319 = x68[x1318];
float x1320 = x1311[x1318];
float x1321 = x1319 - x1320;
x68[x1318] = x1321;

}
for(int x1325=0; x1325 < 50; x1325++) {
float x1326 = x74[x1325];
x74[x1325] = 0.0f;

}
for(int x1330=0; x1330 < 250; x1330++) {
float x1331 = x13[x1330];
bool x1332 = x1331 > 1000.0f;
if (x1332) {
x13[x1330] = 1000.0f;
} else {
}
float x1336 = x13[x1330];
bool x1337 = x1336 < -1000.0f;
if (x1337) {
x13[x1330] = -1000.0f;
} else {
}

}
float* x1343 = (float*)myMalloc(250 * sizeof(float));;
for(int x1344=0; x1344 < 250; x1344++) {
float x1345 = x13[x1344];
float x1346 = x1345 * 5.0E-4f;
x1343[x1344] = x1346;

}
for(int x1350=0; x1350 < 250; x1350++) {
float x1351 = x4[x1350];
float x1352 = x1343[x1350];
float x1353 = x1351 - x1352;
x4[x1350] = x1353;

}
for(int x1357=0; x1357 < 250; x1357++) {
float x1358 = x13[x1357];
x13[x1357] = 0.0f;

}
for(int x1362=0; x1362 < 10; x1362++) {
float x1363 = x98[x1362];
bool x1364 = x1363 > 1000.0f;
if (x1364) {
x98[x1362] = 1000.0f;
} else {
}
float x1368 = x98[x1362];
bool x1369 = x1368 < -1000.0f;
if (x1369) {
x98[x1362] = -1000.0f;
} else {
}

}
float* x1375 = (float*)myMalloc(10 * sizeof(float));;
for(int x1376=0; x1376 < 10; x1376++) {
float x1377 = x98[x1376];
float x1378 = x1377 * 5.0E-4f;
x1375[x1376] = x1378;

}
for(int x1382=0; x1382 < 10; x1382++) {
float x1383 = x93[x1382];
float x1384 = x1375[x1382];
float x1385 = x1383 - x1384;
x93[x1382] = x1385;

}
for(int x1389=0; x1389 < 10; x1389++) {
float x1390 = x98[x1389];
x98[x1389] = 0.0f;

}
for(int x1394=0; x1394 < 500; x1394++) {
float x1395 = x88[x1394];
bool x1396 = x1395 > 1000.0f;
if (x1396) {
x88[x1394] = 1000.0f;
} else {
}
float x1400 = x88[x1394];
bool x1401 = x1400 < -1000.0f;
if (x1401) {
x88[x1394] = -1000.0f;
} else {
}

}
float* x1407 = (float*)myMalloc(500 * sizeof(float));;
for(int x1408=0; x1408 < 500; x1408++) {
float x1409 = x88[x1408];
float x1410 = x1409 * 5.0E-4f;
x1407[x1408] = x1410;

}
for(int x1414=0; x1414 < 500; x1414++) {
float x1415 = x79[x1414];
float x1416 = x1407[x1414];
float x1417 = x1415 - x1416;
x79[x1414] = x1417;

}
for(int x1421=0; x1421 < 500; x1421++) {
float x1422 = x88[x1421];
x88[x1421] = 0.0f;

}
int32_t x1426 = x153;
int32_t x1428 = x1426 % x1427;
bool x1429 = x1428 == 0;
if (x1429) {
float x1434 = x156;
double x1430 = (double)x1426;
double x1431 = 100.0 * x1430;
double x1433 = x1431 / x1432;
float x1435 = (float)x1426;
float x1436 = x1434 / x1435;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x149,x1426,x114,x1433,x1436);
fflush(stdout);
} else {
}
mallocAddr = (void*)x148;
x162 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1447 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1448 = x1447 / 1000LL;
int64_t x1450 = x1447 / x1449;
printf("Training completed in %ldms (%ld us/images)\n",x1448,x1450);
float x1452 = x156;
float x1454 = x1452 / x1453;
double x1455 = (double)x1454;
x147[x149] = x1455;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1461 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1466 = (long)fopen(x0, "w");
fprintf((FILE *)x1466, "unit: %s\n", "1 epoch");
for(int x1468=0; x1468 < 10; x1468++) {
double x1469 = x147[x1468];
fprintf((FILE *)x1466, "%lf\n", x1469);

}
float x1462 = (float)x1461;
float x1463 = x1462 / 1000000.0f;
float x1464 = x1463 - x145;
float x1465 = x1464 / 10.0f;
fprintf((FILE *)x1466, "run time: %lf %lf\n", x145, x1465);
fclose((FILE*)x1466);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

