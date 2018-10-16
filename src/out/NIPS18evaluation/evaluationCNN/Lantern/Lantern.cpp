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
int32_t x1334 = x114 / 10;
double x1339 = (double)x114;
int64_t x1356 = (int64_t)x114;
float x1360 = (float)x114;
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
// dot: ArrayBuffer(100, 320), List(320, 50)
float* x596 = (float*)myMalloc(5000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 100,50,320,1,x497,320,x54,50,0,x596,50);
float* x598 = (float*)myMalloc(5000 * sizeof(float));;
for(int x599=0; x599 < 5000; x599++) {
x598[x599] = 0.0f;

}
float* x603 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x604 = 0;
int32_t x605 = 0;
int32_t x606 = 0;
for(int x607=0; x607 < 100; x607++) {
int32_t x608 = x605;
int32_t x609 = x606;
int32_t x610 = x604;
int32_t x611 = x610;
int32_t x612 = x608;
int32_t x613 = x609;
for(int x614=0; x614 < 50; x614++) {
int32_t x615 = x611;
int32_t x616 = x612;
float x617 = x596[x616];
int32_t x618 = x613;
float x619 = x68[x618];
float x620 = x617 + x619;
x603[x615] = x620;
x611 += 1;
x612 += 1;
x613 += 1;

}
x604 += 50;
x605 += 50;

}
float* x631 = (float*)myMalloc(5000 * sizeof(float));;
for(int x632=0; x632 < 5000; x632++) {
x631[x632] = 0.0f;

}
float* x636 = (float*)myMalloc(5000 * sizeof(float));;
float* x637 = (float*)myMalloc(5000 * sizeof(float));;
for(int x638=0; x638 < 5000; x638++) {
float x639 = (float)rand()/RAND_MAX;
bool x640 = x639 > 0.5f;
if (x640) {
float x641 = x603[x638];
float x642 = x641 * 2.0f;
x636[x638] = x642;
x637[x638] = 2.0f;
} else {
x636[x638] = 0.0f;
x637[x638] = 0.0f;
}

}
float* x652 = (float*)myMalloc(5000 * sizeof(float));;
for(int x653=0; x653 < 5000; x653++) {
x652[x653] = 0.0f;

}
// dot: List(100, 50), List(50, 10)
float* x658 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 100,10,50,1,x636,50,x79,10,0,x658,10);
float* x660 = (float*)myMalloc(1000 * sizeof(float));;
for(int x662=0; x662 < 1000; x662++) {
x660[x662] = 0.0f;

}
float* x666 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x667 = 0;
int32_t x668 = 0;
int32_t x669 = 0;
for(int x670=0; x670 < 100; x670++) {
int32_t x671 = x668;
int32_t x672 = x669;
int32_t x673 = x667;
int32_t x674 = x673;
int32_t x675 = x671;
int32_t x676 = x672;
for(int x677=0; x677 < 10; x677++) {
int32_t x678 = x674;
int32_t x679 = x675;
float x680 = x658[x679];
int32_t x681 = x676;
float x682 = x93[x681];
float x683 = x680 + x682;
x666[x678] = x683;
x674 += 1;
x675 += 1;
x676 += 1;

}
x667 += 10;
x668 += 10;

}
float* x694 = (float*)myMalloc(1000 * sizeof(float));;
for(int x695=0; x695 < 1000; x695++) {
x694[x695] = 0.0f;

}
float* x699 = (float*)myMalloc(100 * sizeof(float));;
int32_t x700 = 0;
for(int x701=0; x701 < 100; x701++) {
float x702 = -3.4028235E38f;
for(int x703=0; x703 < 10; x703++) {
int32_t x704 = x700;
float x705 = x666[x704];
float x706 = x702;
bool x707 = x705 > x706;
if (x707) {
float x708 = x666[x704];
x702 = x708;
} else {
}
x700 += 1;

}
float x715 = x702;
x699[x701] = x715;

}
float* x719 = (float*)myMalloc(1000 * sizeof(float));;
for(int x720=0; x720 < 1000; x720++) {
x719[x720] = 0.0f;

}
int32_t x724 = 0;
for(int x725=0; x725 < 100; x725++) {
for(int x726=0; x726 < 10; x726++) {
int32_t x727 = x724;
float x728 = x666[x727];
float x729 = x699[x725];
float x730 = x728 - x729;
double x731 = (double)x730;
double x732 = exp(x731);
float x733 = (float)x732;
x719[x727] = x733;
x724 += 1;

}

}
float* x740 = (float*)myMalloc(100 * sizeof(float));;
for(int x741=0; x741 < 100; x741++) {
x740[x741] = 0.0f;

}
for(int x745=0; x745 < 100; x745++) {
int32_t x746 = x745;
int32_t x747 = x745 * 10;
int32_t x748 = x747;
for(int x749=0; x749 < 10; x749++) {
int32_t x750 = x746;
float x751 = x740[x750];
int32_t x752 = x748;
float x753 = x719[x752];
float x754 = x751 + x753;
x740[x750] = x754;
x748 += 1;

}

}
x724 = 0;
for(int x762=0; x762 < 100; x762++) {
float x763 = x699[x762];
float x764 = x740[x762];
double x765 = (double)x764;
double x766 = log(x765);
float x767 = (float)x766;
float x768 = x763 + x767;
for(int x769=0; x769 < 10; x769++) {
int32_t x770 = x724;
float x771 = x666[x770];
float x772 = x771 - x768;
x719[x770] = x772;
x724 += 1;

}

}
float* x779 = (float*)myMalloc(1000 * sizeof(float));;
for(int x780=0; x780 < 1000; x780++) {
x779[x780] = 0.0f;

}
float* x784 = (float*)myMalloc(100 * sizeof(float));;
int32_t x785 = 0;
int32_t x169 = x166 * 100;
int* x170 = x113+x169;
for(int x786=0; x786 < 100; x786++) {
int32_t x787 = x785;
int32_t x788 = x170[x786];
int32_t x789 = x787 + x788;
float x790 = x719[x789];
float x791 = -1.0f * x790;
x784[x786] = x791;
x785 += 10;

}
float* x796 = (float*)myMalloc(100 * sizeof(float));;
for(int x797=0; x797 < 100; x797++) {
x796[x797] = 0.0f;

}
float x801 = 0.0f;
for(int x802=0; x802 < 100; x802++) {
float x803 = x801;
float x804 = x784[x802];
float x805 = x803 + x804;
x801 = x805;

}
float x809 = x801;
float* x810 = (float*)myMalloc(1 * sizeof(float));;
x810[0] = x809;
float* x812 = (float*)myMalloc(1 * sizeof(float));;
for(int x813=0; x813 < 1; x813++) {
x812[x813] = 0.0f;

}
float x817 = x812[0];
x812[0] = 1.0f;
float x819 = x810[0];
x182[0] = x819;
// += tensor of dim 0
float x822 = x812[0];
for(int x823=0; x823 < 100; x823++) {
float x824 = x796[x823];
float x825 = x824 + x822;
x796[x823] = x825;

}
int32_t x829 = 0;
for(int x830=0; x830 < 100; x830++) {
int32_t x831 = x829;
int32_t x832 = x170[x830];
int32_t x833 = x831 + x832;
float x834 = x779[x833];
float x835 = x796[x830];
float x836 = -1.0f * x835;
float x837 = x834 + x836;
x779[x833] = x837;
x829 += 10;

}
float* x842 = (float*)myMalloc(100 * sizeof(float));;
for(int x843=0; x843 < 100; x843++) {
x842[x843] = 0.0f;

}
for(int x847=0; x847 < 100; x847++) {
int32_t x848 = x847;
int32_t x849 = x847 * 10;
int32_t x850 = x849;
for(int x851=0; x851 < 10; x851++) {
int32_t x852 = x848;
float x853 = x842[x852];
int32_t x854 = x850;
float x855 = x779[x854];
float x856 = x853 + x855;
x842[x852] = x856;
x850 += 1;

}

}
int32_t x863 = 0;
for(int x864=0; x864 < 100; x864++) {
for(int x865=0; x865 < 10; x865++) {
int32_t x866 = x863;
float x867 = x694[x866];
float x868 = x779[x866];
float x869 = x719[x866];
float x873 = x842[x864];
double x870 = (double)x869;
double x871 = exp(x870);
float x872 = (float)x871;
float x874 = x872 * x873;
float x875 = x868 - x874;
float x876 = x867 + x875;
x694[x866] = x876;
x863 += 1;

}

}
int32_t x883 = 0;
int32_t x884 = 0;
int32_t x885 = 0;
for(int x886=0; x886 < 100; x886++) {
int32_t x887 = x883;
int32_t x888 = x884;
int32_t x889 = x885;
int32_t x890 = x887;
int32_t x891 = x888;
int32_t x892 = x889;
for(int x893=0; x893 < 10; x893++) {
int32_t x894 = x890;
float x895 = x660[x894];
float x896 = x658[x894];
int32_t x897 = x891;
float x898 = x93[x897];
int32_t x899 = x892;
float x900 = x694[x899];
float x901 = x895 + x900;
x660[x894] = x901;
float x903 = x98[x897];
float x904 = x658[x894];
float x905 = x93[x897];
float x906 = x694[x899];
float x907 = x903 + x906;
x98[x897] = x907;
x892 += 1;
x890 += 1;
x891 += 1;

}
x885 += 10;
x883 += 10;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,50,10,1,x660,10,x79,10,1,x652,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,100,1,x636,50,x660,10,1,x88,10);
float* x920 = (float*)myMalloc(5000 * sizeof(float));;
int32_t x921 = 0;
int32_t x922 = 0;
int32_t x923 = 0;
for(int x924=0; x924 < 100; x924++) {
int32_t x925 = x922;
int32_t x926 = x923;
int32_t x927 = x921;
int32_t x928 = x927;
int32_t x929 = x925;
int32_t x930 = x926;
for(int x931=0; x931 < 50; x931++) {
int32_t x932 = x928;
int32_t x933 = x929;
float x934 = x637[x933];
int32_t x935 = x930;
float x936 = x652[x935];
float x937 = x934 * x936;
x920[x932] = x937;
x928 += 1;
x929 += 1;
x930 += 1;

}
x921 += 50;
x922 += 50;
x923 += 50;

}
for(int x949=0; x949 < 5000; x949++) {
float x950 = x631[x949];
float x951 = x920[x949];
float x952 = x950 + x951;
x631[x949] = x952;

}
int32_t x956 = 0;
int32_t x957 = 0;
int32_t x958 = 0;
for(int x959=0; x959 < 100; x959++) {
int32_t x960 = x956;
int32_t x961 = x957;
int32_t x962 = x958;
int32_t x963 = x960;
int32_t x964 = x961;
int32_t x965 = x962;
for(int x966=0; x966 < 50; x966++) {
int32_t x967 = x963;
float x968 = x598[x967];
float x969 = x596[x967];
int32_t x970 = x964;
float x971 = x68[x970];
int32_t x972 = x965;
float x973 = x631[x972];
float x974 = x968 + x973;
x598[x967] = x974;
float x976 = x74[x970];
float x977 = x596[x967];
float x978 = x68[x970];
float x979 = x631[x972];
float x980 = x976 + x979;
x74[x970] = x980;
x965 += 1;
x963 += 1;
x964 += 1;

}
x958 += 50;
x956 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 100,320,50,1,x598,50,x54,50,1,x590,320);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,100,1,x497,320,x598,50,1,x63,50);
for(int x993=0; x993 < 32000; x993++) {
int32_t x994 = x503[x993];
float x995 = x590[x993];
x492[x994] = x995;

}
for(int x999=0; x999 < 128000; x999++) {
float x1000 = x391[x999];
bool x1001 = x1000 < 0.0f;
float x1004;
if (x1001) {
x1004 = 0.0f;
} else {
float x1002 = x492[x999];
x1004 = x1002;
}
x474[x999] = x1004;

}
for(int x1008=0; x1008 < 100; x1008++) {
int32_t x1009 = x1008 * 1280;
int32_t x1010 = x1009;
int32_t x1011 = 0;
int32_t x1014 = x1008 * 1440;
for(int x1012=0; x1012 < 20; x1012++) {
float x1013 = 0.0f;
int32_t x1015 = x1014;
for(int x1016=0; x1016 < 8; x1016++) {
int32_t x1017 = x1015;
int32_t x1018 = x1017;
for(int x1019=0; x1019 < 8; x1019++) {
int32_t x1020 = x1010;
float x1021 = x474[x1020];
x1013 += x1021;
int32_t x1023 = x1018;
int32_t x1024 = x1023;
int32_t x1025 = x1011;
int32_t x1026 = x1025;
for(int x1027=0; x1027 < 10; x1027++) {
int32_t x1028 = x1024;
int32_t x1029 = x1028;
for(int x1030=0; x1030 < 5; x1030++) {
for(int x1031=0; x1031 < 5; x1031++) {
int32_t x1032 = x1029;
int32_t x1033 = x1032 + x1031;
float x1034 = x386[x1033];
int32_t x1035 = x1026;
float x1036 = x29[x1035];
float x1037 = x1021 * x1036;
float x1038 = x1034 + x1037;
x386[x1033] = x1038;
float x1040 = x38[x1035];
float x1041 = x292[x1033];
float x1042 = x1021 * x1041;
float x1043 = x1040 + x1042;
x38[x1035] = x1043;
x1026 += 1;

}
x1029 += 12;

}
x1024 += 144;

}
x1018 += 1;
x1010 += 1;

}
x1015 += 12;

}
float x1061 = x49[x1012];
float x1062 = x1013;
float x1063 = x1061 + x1062;
x49[x1012] = x1063;
x1011 += 250;

}

}
for(int x1070=0; x1070 < 144000; x1070++) {
int32_t x1071 = x298[x1070];
float x1072 = x386[x1070];
x287[x1071] = x1072;

}
for(int x1076=0; x1076 < 576000; x1076++) {
float x1077 = x187[x1076];
bool x1078 = x1077 < 0.0f;
float x1081;
if (x1078) {
x1081 = 0.0f;
} else {
float x1079 = x287[x1076];
x1081 = x1079;
}
x269[x1076] = x1081;

}
for(int x1085=0; x1085 < 100; x1085++) {
int32_t x1086 = x1085 * 5760;
int32_t x1087 = x1086;
int32_t x1088 = 0;
int32_t x1091 = x1085 * 784;
for(int x1089=0; x1089 < 10; x1089++) {
float x1090 = 0.0f;
int32_t x1092 = x1091;
for(int x1093=0; x1093 < 24; x1093++) {
int32_t x1094 = x1092;
int32_t x1095 = x1094;
for(int x1096=0; x1096 < 24; x1096++) {
int32_t x1097 = x1087;
float x1098 = x269[x1097];
x1090 += x1098;
int32_t x1100 = x1095;
int32_t x1101 = x1100;
int32_t x1102 = x1088;
int32_t x1103 = x1102;
int32_t x1104 = x1101;
int32_t x1105 = x1104;
for(int x1106=0; x1106 < 5; x1106++) {
for(int x1107=0; x1107 < 5; x1107++) {
int32_t x1108 = x1103;
float x1109 = x13[x1108];
int32_t x1110 = x1105;
int32_t x1111 = x1110 + x1107;
float x1112 = x168[x1111];
float x1113 = x1098 * x1112;
float x1114 = x1109 + x1113;
x13[x1108] = x1114;
x1103 += 1;

}
x1105 += 28;

}
x1101 += 784;
x1095 += 1;
x1087 += 1;

}
x1092 += 28;

}
float x1130 = x24[x1089];
float x1131 = x1090;
float x1132 = x1130 + x1131;
x24[x1089] = x1132;
x1088 += 25;

}

}
float x1139 = x182[0];
x157 += x1139;
for(int x1141=0; x1141 < 5000; x1141++) {
float x1142 = x38[x1141];
bool x1143 = x1142 > 1000.0f;
if (x1143) {
x38[x1141] = 1000.0f;
} else {
}
float x1147 = x38[x1141];
bool x1148 = x1147 < -1000.0f;
if (x1148) {
x38[x1141] = -1000.0f;
} else {
}

}
float* x1154 = (float*)myMalloc(5000 * sizeof(float));;
for(int x1155=0; x1155 < 5000; x1155++) {
float x1156 = x38[x1155];
float x1157 = x1156 * 5.0E-4f;
x1154[x1155] = x1157;

}
for(int x1161=0; x1161 < 5000; x1161++) {
float x1162 = x29[x1161];
float x1163 = x1154[x1161];
float x1164 = x1162 - x1163;
x29[x1161] = x1164;

}
for(int x1168=0; x1168 < 5000; x1168++) {
float x1169 = x38[x1168];
x38[x1168] = 0.0f;

}
for(int x1173=0; x1173 < 16000; x1173++) {
float x1174 = x63[x1173];
bool x1175 = x1174 > 1000.0f;
if (x1175) {
x63[x1173] = 1000.0f;
} else {
}
float x1179 = x63[x1173];
bool x1180 = x1179 < -1000.0f;
if (x1180) {
x63[x1173] = -1000.0f;
} else {
}

}
float* x1186 = (float*)myMalloc(16000 * sizeof(float));;
for(int x1187=0; x1187 < 16000; x1187++) {
float x1188 = x63[x1187];
float x1189 = x1188 * 5.0E-4f;
x1186[x1187] = x1189;

}
for(int x1193=0; x1193 < 16000; x1193++) {
float x1194 = x54[x1193];
float x1195 = x1186[x1193];
float x1196 = x1194 - x1195;
x54[x1193] = x1196;

}
for(int x1200=0; x1200 < 16000; x1200++) {
float x1201 = x63[x1200];
x63[x1200] = 0.0f;

}
for(int x1205=0; x1205 < 50; x1205++) {
float x1206 = x74[x1205];
bool x1207 = x1206 > 1000.0f;
if (x1207) {
x74[x1205] = 1000.0f;
} else {
}
float x1211 = x74[x1205];
bool x1212 = x1211 < -1000.0f;
if (x1212) {
x74[x1205] = -1000.0f;
} else {
}

}
float* x1218 = (float*)myMalloc(50 * sizeof(float));;
for(int x1219=0; x1219 < 50; x1219++) {
float x1220 = x74[x1219];
float x1221 = x1220 * 5.0E-4f;
x1218[x1219] = x1221;

}
for(int x1225=0; x1225 < 50; x1225++) {
float x1226 = x68[x1225];
float x1227 = x1218[x1225];
float x1228 = x1226 - x1227;
x68[x1225] = x1228;

}
for(int x1232=0; x1232 < 50; x1232++) {
float x1233 = x74[x1232];
x74[x1232] = 0.0f;

}
for(int x1237=0; x1237 < 250; x1237++) {
float x1238 = x13[x1237];
bool x1239 = x1238 > 1000.0f;
if (x1239) {
x13[x1237] = 1000.0f;
} else {
}
float x1243 = x13[x1237];
bool x1244 = x1243 < -1000.0f;
if (x1244) {
x13[x1237] = -1000.0f;
} else {
}

}
float* x1250 = (float*)myMalloc(250 * sizeof(float));;
for(int x1251=0; x1251 < 250; x1251++) {
float x1252 = x13[x1251];
float x1253 = x1252 * 5.0E-4f;
x1250[x1251] = x1253;

}
for(int x1257=0; x1257 < 250; x1257++) {
float x1258 = x4[x1257];
float x1259 = x1250[x1257];
float x1260 = x1258 - x1259;
x4[x1257] = x1260;

}
for(int x1264=0; x1264 < 250; x1264++) {
float x1265 = x13[x1264];
x13[x1264] = 0.0f;

}
for(int x1269=0; x1269 < 10; x1269++) {
float x1270 = x98[x1269];
bool x1271 = x1270 > 1000.0f;
if (x1271) {
x98[x1269] = 1000.0f;
} else {
}
float x1275 = x98[x1269];
bool x1276 = x1275 < -1000.0f;
if (x1276) {
x98[x1269] = -1000.0f;
} else {
}

}
float* x1282 = (float*)myMalloc(10 * sizeof(float));;
for(int x1283=0; x1283 < 10; x1283++) {
float x1284 = x98[x1283];
float x1285 = x1284 * 5.0E-4f;
x1282[x1283] = x1285;

}
for(int x1289=0; x1289 < 10; x1289++) {
float x1290 = x93[x1289];
float x1291 = x1282[x1289];
float x1292 = x1290 - x1291;
x93[x1289] = x1292;

}
for(int x1296=0; x1296 < 10; x1296++) {
float x1297 = x98[x1296];
x98[x1296] = 0.0f;

}
for(int x1301=0; x1301 < 500; x1301++) {
float x1302 = x88[x1301];
bool x1303 = x1302 > 1000.0f;
if (x1303) {
x88[x1301] = 1000.0f;
} else {
}
float x1307 = x88[x1301];
bool x1308 = x1307 < -1000.0f;
if (x1308) {
x88[x1301] = -1000.0f;
} else {
}

}
float* x1314 = (float*)myMalloc(500 * sizeof(float));;
for(int x1315=0; x1315 < 500; x1315++) {
float x1316 = x88[x1315];
float x1317 = x1316 * 5.0E-4f;
x1314[x1315] = x1317;

}
for(int x1321=0; x1321 < 500; x1321++) {
float x1322 = x79[x1321];
float x1323 = x1314[x1321];
float x1324 = x1322 - x1323;
x79[x1321] = x1324;

}
for(int x1328=0; x1328 < 500; x1328++) {
float x1329 = x88[x1328];
x88[x1328] = 0.0f;

}
int32_t x1333 = x154;
int32_t x1335 = x1333 % x1334;
bool x1336 = x1335 == 0;
if (x1336) {
float x1341 = x157;
double x1337 = (double)x1333;
double x1338 = 100.0 * x1337;
double x1340 = x1338 / x1339;
float x1342 = (float)x1333;
float x1343 = x1341 / x1342;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x150,x1333,x114,x1340,x1343);
fflush(stdout);
} else {
}
mallocAddr = (void*)x148;
x163 += 78400;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1354 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1355 = x1354 / 1000LL;
int64_t x1357 = x1354 / x1356;
printf("Training completed in %ldms (%ld us/images)\n",x1355,x1357);
float x1359 = x157;
float x1361 = x1359 / x1360;
double x1362 = (double)x1361;
x147[x150] = x1362;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1368 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1373 = (long)fopen(x0, "w");
fprintf((FILE *)x1373, "unit: %s\n", "1 epoch");
for(int x1375=0; x1375 < 4; x1375++) {
double x1376 = x147[x1375];
fprintf((FILE *)x1373, "%lf\n", x1376);

}
float x1369 = (float)x1368;
float x1370 = x1369 / 1000000.0f;
float x1371 = x1370 - x145;
float x1372 = x1371 / 4.0f;
fprintf((FILE *)x1373, "run time: %lf %lf\n", x145, x1372);
fclose((FILE*)x1373);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

