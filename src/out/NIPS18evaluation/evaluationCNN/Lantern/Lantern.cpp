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
double* x147 = (double*)myMalloc(4 * sizeof(double));;
int64_t x148 = (long)mallocAddr;
int32_t x164 = x114 / 100;
int32_t x1427 = x114 / 10;
double x1432 = (double)x114;
int64_t x1449 = (int64_t)x114;
float x1453 = (float)x114;
for(int x150=0; x150 < 4; x150++) {
struct timeval begin_1, end_1, diff_1;
int32_t x152 = 0;
int32_t x153 = x152;
int32_t x154 = x153;
float x155 = 0.0f;
float x156 = x155;
float x157 = x156;
int32_t x158 = x150 + 1;
printf("Start training epoch %d\n",x158);
gettimeofday(&begin_1, NULL);
int32_t x161 = 0;
int32_t x162 = x161;
int32_t x163 = x162;
for(int x166=0; x166 < x164; x166++) {
int32_t x167 = x163;
x154 += 100;
float* x172 = (float*)myMalloc(1 * sizeof(float));;
x172[0] = 0.0f;
float* x174 = (float*)myMalloc(1 * sizeof(float));;
x174[0] = 0.0f;
float* x176 = (float*)myMalloc(1 * sizeof(float));;
for(int x178=0; x178 < 1; x178++) {
x176[x178] = 0.0f;

}
float* x182 = (float*)myMalloc(1 * sizeof(float));;
for(int x183=0; x183 < 1; x183++) {
x182[x183] = 0.0f;

}
float* x187 = (float*)myMalloc(576000 * sizeof(float));;
int32_t x188 = 0;
for(int x190=0; x190 < 100; x190++) {
for(int x191=0; x191 < 10; x191++) {
for(int x193=0; x193 < 576; x193++) {
int32_t x194 = x188;
float x195 = x18[x191];
x187[x194] = x195;
x188 += 1;

}

}

}
float* x168 = x107+x167;
for(int x204=0; x204 < 100; x204++) {
int32_t x207 = x204 * 5760;
float* x208 = x187+x207;
int32_t x209 = 0;
int32_t x210 = 0;
int32_t x205 = x204 * 784;
float* x206 = x168+x205;
for(int x211=0; x211 < 10; x211++) {
int32_t x212 = x210;
int32_t x213 = x212;
int32_t x214 = 0;
int32_t x215 = x209;
float* x216 = x208+x215;
int32_t x217 = x214;
int32_t x219 = x213;
float* x220 = x4+x219;
int32_t x221 = 0;
int32_t x222 = 0;
float* x218 = x206+x217;
for(int x224=0; x224 < 24; x224++) {
int32_t x225 = x222;
int32_t x226 = x225;
for(int x227=0; x227 < 24; x227++) {
float x228 = 0.0f;
int32_t x229 = 0;
int32_t x230 = x226;
int32_t x231 = x230;
for(int x233=0; x233 < 5; x233++) {
int32_t x234 = x231;
int32_t x236 = x229;
float* x237 = x220+x236;
float* x235 = x218+x234;
for(int x238=0; x238 < 5; x238++) {
float x239 = x235[x238];
float x240 = x237[x238];
float x241 = x239 * x240;
x228 += x241;

}
x229 += 5;
x231 += 28;

}
int32_t x249 = x221;
float x250 = x216[x249];
float x251 = x228;
float x252 = x250 + x251;
x216[x249] = x252;
x221 += 1;
x226 += 1;

}
x222 += 28;

}
x213 += 25;
x214 += 784;
x210 += 25;
x209 += 576;

}

}
float* x269 = (float*)myMalloc(576000 * sizeof(float));;
for(int x271=0; x271 < 576000; x271++) {
x269[x271] = 0.0f;

}
float* x275 = (float*)myMalloc(576000 * sizeof(float));;
for(int x276=0; x276 < 576000; x276++) {
float x277 = x187[x276];
bool x278 = x277 < 0.0f;
if (x278) {
x275[x276] = 0.0f;
} else {
float x281 = x187[x276];
x275[x276] = x281;
}

}
float* x287 = (float*)myMalloc(576000 * sizeof(float));;
for(int x288=0; x288 < 576000; x288++) {
x287[x288] = 0.0f;

}
float* x292 = (float*)myMalloc(144000 * sizeof(float));;
for(int x294=0; x294 < 144000; x294++) {
x292[x294] = -3.4028235E38f;

}
int* x298 = (int32_t*)myMalloc(144000 * sizeof(int32_t));;
for(int x299=0; x299 < 100; x299++) {
int32_t x300 = x299 * 5760;
float* x301 = x275+x300;
int32_t x302 = x299 * 1440;
float* x303 = x292+x302;
int* x304 = x298+x302;
int32_t x305 = 0;
int32_t x306 = 0;
for(int x307=0; x307 < 10; x307++) {
int32_t x308 = x305;
int32_t x309 = x308;
int32_t x310 = x306;
int32_t x311 = x310;
for(int x313=0; x313 < 12; x313++) {
int32_t x314 = x309;
int32_t x315 = x314;
int32_t x316 = x311;
int32_t x317 = x316;
for(int x318=0; x318 < 12; x318++) {
int32_t x319 = x317;
int32_t x320 = x319;
int32_t x321 = x320;
int32_t x322 = x321;
int32_t x323 = x322;
float x324 = x301[x323];
int32_t x325 = x315;
float x326 = x303[x325];
bool x327 = x324 > x326;
if (x327) {
float x328 = x301[x323];
x303[x325] = x328;
int32_t x330 = x323 + x300;
x304[x325] = x330;
} else {
}
x322 += 1;
int32_t x335 = x322;
float x336 = x301[x335];
float x337 = x303[x325];
bool x338 = x336 > x337;
if (x338) {
float x339 = x301[x335];
x303[x325] = x339;
int32_t x341 = x335 + x300;
x304[x325] = x341;
} else {
}
x322 += 1;
x320 += 24;
int32_t x347 = x320;
int32_t x348 = x347;
int32_t x349 = x348;
float x350 = x301[x349];
float x351 = x303[x325];
bool x352 = x350 > x351;
if (x352) {
float x353 = x301[x349];
x303[x325] = x353;
int32_t x355 = x349 + x300;
x304[x325] = x355;
} else {
}
x348 += 1;
int32_t x360 = x348;
float x361 = x301[x360];
float x362 = x303[x325];
bool x363 = x361 > x362;
if (x363) {
float x364 = x301[x360];
x303[x325] = x364;
int32_t x366 = x360 + x300;
x304[x325] = x366;
} else {
}
x348 += 1;
x320 += 24;
x315 += 1;
x317 += 2;

}
x309 += 12;
x311 += 48;

}
x305 += 144;
x306 += 576;

}

}
float* x386 = (float*)myMalloc(144000 * sizeof(float));;
for(int x387=0; x387 < 144000; x387++) {
x386[x387] = 0.0f;

}
float* x391 = (float*)myMalloc(128000 * sizeof(float));;
int32_t x392 = 0;
for(int x393=0; x393 < 100; x393++) {
for(int x394=0; x394 < 20; x394++) {
for(int x396=0; x396 < 64; x396++) {
int32_t x397 = x392;
float x398 = x43[x394];
x391[x397] = x398;
x392 += 1;

}

}

}
for(int x407=0; x407 < 100; x407++) {
int32_t x408 = x407 * 1440;
float* x409 = x292+x408;
int32_t x410 = x407 * 1280;
float* x411 = x391+x410;
int32_t x412 = 0;
int32_t x413 = 0;
for(int x414=0; x414 < 20; x414++) {
int32_t x415 = x413;
int32_t x416 = x415;
int32_t x417 = 0;
int32_t x418 = x412;
float* x419 = x411+x418;
for(int x420=0; x420 < 10; x420++) {
int32_t x421 = x417;
float* x422 = x409+x421;
int32_t x423 = x416;
float* x424 = x29+x423;
int32_t x425 = 0;
int32_t x426 = 0;
for(int x428=0; x428 < 8; x428++) {
int32_t x429 = x426;
int32_t x430 = x429;
for(int x431=0; x431 < 8; x431++) {
float x432 = 0.0f;
int32_t x433 = 0;
int32_t x434 = x430;
int32_t x435 = x434;
for(int x436=0; x436 < 5; x436++) {
int32_t x437 = x435;
float* x438 = x422+x437;
int32_t x439 = x433;
float* x440 = x424+x439;
for(int x441=0; x441 < 5; x441++) {
float x442 = x438[x441];
float x443 = x440[x441];
float x444 = x442 * x443;
x432 += x444;

}
x433 += 5;
x435 += 12;

}
int32_t x452 = x425;
float x453 = x419[x452];
float x454 = x432;
float x455 = x453 + x454;
x419[x452] = x455;
x425 += 1;
x430 += 1;

}
x426 += 12;

}
x416 += 25;
x417 += 144;

}
x413 += 250;
x412 += 64;

}

}
float* x474 = (float*)myMalloc(128000 * sizeof(float));;
for(int x476=0; x476 < 128000; x476++) {
x474[x476] = 0.0f;

}
float* x480 = (float*)myMalloc(128000 * sizeof(float));;
for(int x481=0; x481 < 128000; x481++) {
float x482 = x391[x481];
bool x483 = x482 < 0.0f;
if (x483) {
x480[x481] = 0.0f;
} else {
float x486 = x391[x481];
x480[x481] = x486;
}

}
float* x492 = (float*)myMalloc(128000 * sizeof(float));;
for(int x493=0; x493 < 128000; x493++) {
x492[x493] = 0.0f;

}
float* x497 = (float*)myMalloc(32000 * sizeof(float));;
for(int x499=0; x499 < 32000; x499++) {
x497[x499] = -3.4028235E38f;

}
int* x503 = (int32_t*)myMalloc(32000 * sizeof(int32_t));;
for(int x504=0; x504 < 100; x504++) {
int32_t x505 = x504 * 1280;
float* x506 = x480+x505;
int32_t x507 = x504 * 320;
float* x508 = x497+x507;
int* x509 = x503+x507;
int32_t x510 = 0;
int32_t x511 = 0;
for(int x512=0; x512 < 20; x512++) {
int32_t x513 = x510;
int32_t x514 = x513;
int32_t x515 = x511;
int32_t x516 = x515;
for(int x517=0; x517 < 4; x517++) {
int32_t x518 = x514;
int32_t x519 = x518;
int32_t x520 = x516;
int32_t x521 = x520;
for(int x522=0; x522 < 4; x522++) {
int32_t x523 = x521;
int32_t x524 = x523;
int32_t x525 = x524;
int32_t x526 = x525;
int32_t x527 = x526;
float x528 = x506[x527];
int32_t x529 = x519;
float x530 = x508[x529];
bool x531 = x528 > x530;
if (x531) {
float x532 = x506[x527];
x508[x529] = x532;
int32_t x534 = x527 + x505;
x509[x529] = x534;
} else {
}
x526 += 1;
int32_t x539 = x526;
float x540 = x506[x539];
float x541 = x508[x529];
bool x542 = x540 > x541;
if (x542) {
float x543 = x506[x539];
x508[x529] = x543;
int32_t x545 = x539 + x505;
x509[x529] = x545;
} else {
}
x526 += 1;
x524 += 8;
int32_t x551 = x524;
int32_t x552 = x551;
int32_t x553 = x552;
float x554 = x506[x553];
float x555 = x508[x529];
bool x556 = x554 > x555;
if (x556) {
float x557 = x506[x553];
x508[x529] = x557;
int32_t x559 = x553 + x505;
x509[x529] = x559;
} else {
}
x552 += 1;
int32_t x564 = x552;
float x565 = x506[x564];
float x566 = x508[x529];
bool x567 = x565 > x566;
if (x567) {
float x568 = x506[x564];
x508[x529] = x568;
int32_t x570 = x564 + x505;
x509[x529] = x570;
} else {
}
x552 += 1;
x524 += 8;
x519 += 1;
x521 += 2;

}
x514 += 4;
x516 += 16;

}
x510 += 16;
x511 += 64;

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
float x604 = x497[x603];
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
int32_t x169 = x166 * 100;
int* x170 = x113+x169;
for(int x829=0; x829 < 100; x829++) {
int32_t x830 = x828;
int32_t x831 = x170[x829];
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
x182[0] = x862;
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
int32_t x875 = x170[x873];
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
float x1075 = x497[x1063];
float x1076 = x620[x1069];
float x1077 = x1075 * x1076;
float x1078 = x1074 + x1077;
x63[x1066] = x1078;

}

}

}
for(int x1086=0; x1086 < 32000; x1086++) {
int32_t x1087 = x503[x1086];
float x1088 = x590[x1086];
x492[x1087] = x1088;

}
for(int x1092=0; x1092 < 128000; x1092++) {
float x1093 = x391[x1092];
bool x1094 = x1093 < 0.0f;
float x1097;
if (x1094) {
x1097 = 0.0f;
} else {
float x1095 = x492[x1092];
x1097 = x1095;
}
x474[x1092] = x1097;

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
float x1114 = x474[x1113];
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
float x1127 = x386[x1126];
int32_t x1128 = x1119;
float x1129 = x29[x1128];
float x1130 = x1114 * x1129;
float x1131 = x1127 + x1130;
x386[x1126] = x1131;
float x1133 = x38[x1128];
float x1134 = x292[x1126];
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
int32_t x1164 = x298[x1163];
float x1165 = x386[x1163];
x287[x1164] = x1165;

}
for(int x1169=0; x1169 < 576000; x1169++) {
float x1170 = x187[x1169];
bool x1171 = x1170 < 0.0f;
float x1174;
if (x1171) {
x1174 = 0.0f;
} else {
float x1172 = x287[x1169];
x1174 = x1172;
}
x269[x1169] = x1174;

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
float x1191 = x269[x1190];
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
float x1205 = x168[x1204];
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
float x1232 = x182[0];
x157 += x1232;
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
int32_t x1426 = x154;
int32_t x1428 = x1426 % x1427;
bool x1429 = x1428 == 0;
if (x1429) {
float x1434 = x157;
double x1430 = (double)x1426;
double x1431 = 100.0 * x1430;
double x1433 = x1431 / x1432;
float x1435 = (float)x1426;
float x1436 = x1434 / x1435;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x150,x1426,x114,x1433,x1436);
fflush(stdout);
} else {
}
mallocAddr = (void*)x148;
x163 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1447 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1448 = x1447 / 1000LL;
int64_t x1450 = x1447 / x1449;
printf("Training completed in %ldms (%ld us/images)\n",x1448,x1450);
float x1452 = x157;
float x1454 = x1452 / x1453;
double x1455 = (double)x1454;
x147[x150] = x1455;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1461 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1466 = (long)fopen(x0, "w");
fprintf((FILE *)x1466, "unit: %s\n", "1 epoch");
for(int x1468=0; x1468 < 4; x1468++) {
double x1469 = x147[x1468];
fprintf((FILE *)x1466, "%lf\n", x1469);

}
float x1462 = (float)x1461;
float x1463 = x1462 / 1000000.0f;
float x1464 = x1463 - x145;
float x1465 = x1464 / 4.0f;
fprintf((FILE *)x1466, "run time: %lf %lf\n", x145, x1465);
fclose((FILE*)x1466);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

