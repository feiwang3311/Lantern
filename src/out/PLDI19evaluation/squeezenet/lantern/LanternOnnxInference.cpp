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
srand(42);
struct timeval begin_0, end_0, diff_0;
gettimeofday(&begin_0, NULL);
int32_t x5 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
int32_t x6 = fsize(x5);
int64_t x8 = (int64_t)x6;
int64_t x9 = x8 / 3073LL;
int32_t x10 = (int32_t)x9;
int32_t x11 = x10 * 3072;
float* x12 = (float*)myMalloc(x11 * sizeof(float));;
int* x13 = (int32_t*)myMalloc(x10 * sizeof(int32_t));;
char* x7 = (char*)mmap(0, x6, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x5, 0);
for(int x15=0; x15 < x10; x15++) {
int32_t x16 = x15 * 3073;
char x17 = x7[x16];
int32_t x18 = (int32_t)(unsigned char)x17;
x13[x15] = x18;
int32_t x24 = x16 + 1;
int32_t x22 = x15 * 3072;
for(int x21=0; x21 < 3072; x21++) {
int32_t x25 = x24 + x21;
char x26 = x7[x25];
int32_t x23 = x22 + x21;
float x27 = (float)(unsigned char)x26;
float x28 = x27 / 255.0f;
x12[x23] = x28;

}

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x36 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x37 = (float)x36;
float x38 = x37 / 1000000.0f;
printf("Data reading in %lf sec\n",x38);
int64_t x95 = (long)mallocAddr;
// inferencing loop starts here
int32_t x103 = x10 / 64;
int32_t x40 = open("/home/fei/bitbucket/Lantern/src/out/PLDI19evaluation/squeezenet/squeezenetCifar10.onnx.bin",0);
int32_t x41 = fsize(x40);
float* x42 = (float*)mmap(0, x41, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x40, 0);
float* x85 = x42+2592;
float* x75 = x42+0;
float* x50 = x42+4224;
float* x92 = x42+2688;
float* x73 = x42+5264;
float* x66 = x42+4240;
float* x47 = x42+14544;
float* x89 = x42+5328;
float* x67 = x42+16656;
float* x54 = x42+14608;
float* x45 = x42+17696;
float* x53 = x42+16672;
float* x79 = x42+26976;
float* x61 = x42+17760;
float* x65 = x42+31136;
float* x52 = x42+27040;
float* x87 = x42+35264;
float* x77 = x42+31168;
float* x83 = x42+72256;
float* x48 = x42+35392;
float* x57 = x42+80576;
float* x69 = x42+72384;
float* x63 = x42+84704;
float* x49 = x42+80608;
float* x58 = x42+121696;
float* x78 = x42+84832;
float* x94 = x42+134112;
float* x84 = x42+121824;
float* x88 = x42+143376;
float* x90 = x42+134160;
float* x71 = x42+226512;
float* x81 = x42+143568;
float* x44 = x42+245136;
float* x56 = x42+226704;
float* x74 = x42+254400;
float* x64 = x42+245184;
float* x86 = x42+337536;
float* x60 = x42+254592;
float* x51 = x42+362304;
float* x76 = x42+337728;
float* x82 = x42+378752;
float* x91 = x42+362368;
float* x55 = x42+526464;
float* x70 = x42+379008;
float* x62 = x42+559488;
float* x43 = x42+526720;
float* x68 = x42+575936;
float* x80 = x42+559552;
float* x59 = x42+723648;
float* x72 = x42+576192;
float* x93 = x42+805824;
float* x46 = x42+723904;
int64_t x2675 = (int64_t)x10;
for(int x98=0; x98 < 4; x98++) {
struct timeval begin_1, end_1, diff_1;
int32_t x100 = x98 + 1;
printf("Start inferencing epoch %d\n",x100);
gettimeofday(&begin_1, NULL);
for(int x105=0; x105 < x103; x105++) {
int32_t x106 = x105 * 64;
int32_t x107 = x106 * 3072;
float* x108 = x12+x107;
int* x109 = x13+x106;
float* x110 = (float*)myMalloc(6291456 * sizeof(float));;
int32_t x111 = 0;
for(int x113=0; x113 < 64; x113++) {
for(int x115=0; x115 < 96; x115++) {
for(int x117=0; x117 < 1024; x117++) {
int32_t x118 = x111;
float x119 = x85[x115];
x110[x118] = x119;
x111 += 1;

}

}

}
float* x128 = (float*)myMalloc(1769472 * sizeof(float));;
for(int x129=0; x129 < 64; x129++) {
int32_t x130 = x129 * 3072;
float* x131 = x108+x130;
int32_t x132 = x129 * 98304;
float* x133 = x110+x132;
int32_t x134 = x129 * 27648;
float* x135 = x128+x134;
for(int x137=0; x137 < 27; x137++) {
int32_t x138 = x137 / 9;
int32_t x142 = x138 * 3;
int32_t x143 = x142 * 3;
int32_t x144 = x143 * 32;
int32_t x145 = x144 * 32;
int32_t x139 = x137 % 9;
int32_t x140 = x139 / 3;
int32_t x146 = x140 * 3;
int32_t x147 = x146 * 32;
int32_t x148 = x147 * 32;
int32_t x149 = x145 + x148;
int32_t x141 = x139 % 3;
int32_t x150 = x141 * 32;
int32_t x151 = x150 * 32;
int32_t x152 = x149 + x151;
float* x153 = x135+x152;
int32_t x154 = x138 * 32;
int32_t x155 = x154 * 32;
float* x156 = x131+x155;
int32_t x169 = 1 - x141;
bool x170 = x169 > 0;
int32_t x171;
if (x170) {
x171 = x169;
} else {
x171 = 0;
}
int32_t x172 = 3 - x141;
int32_t x173 = x172 - 1;
int32_t x174 = 1 - x173;
bool x175 = x174 > 0;
int32_t x176;
if (x175) {
x176 = x174;
} else {
x176 = 0;
}
int32_t x177 = 32 - x176;
int32_t x178 = x177 - x171;
bool x179 = x178 <= 0;
bool x183 = x171 > 0;
int32_t x168 = -1 + x141;
bool x196 = x176 > 0;
for(int x158=0; x158 < 32; x158++) {
int32_t x159 = x158 - 1;
int32_t x160 = x159 + x140;
bool x161 = x160 < 0;
bool x162 = x160 >= 32;
bool x163 = x161 || x162;
if (x163) {
int32_t x164 = x158 * 32;
float* x165 = x153+x164;
memset(x165, 0, 4 * 32);;
} else {
if (x179) {
int32_t x164 = x158 * 32;
float* x180 = x153+x164;
memset(x180, 0, 4 * 32);;
} else {
int32_t x164 = x158 * 32;
if (x183) {
float* x184 = x153+x164;
memset(x184, 0, 4 * x171);;
} else {
}
// may have segfault here
int32_t x189 = x164 + x171;
float* x190 = x153+x189;
int32_t x191 = x160 * 32;
int32_t x192 = x191 + x168;
int32_t x193 = x192 + x171;
float* x194 = x156+x193;
memcpy(x190, x194, 4 * x178);;
if (x196) {
int32_t x197 = x164 + 32;
int32_t x198 = x197 - x176;
float* x199 = x153+x198;
memset(x199, 0, 4 * x176);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 96,1024,27,1,x75,27,x135,1024,1,x133,1024);

}
float* x214 = (float*)myMalloc(6291456 * sizeof(float));;
for(int x216=0; x216 < 6291456; x216++) {
float x217 = x110[x216];
bool x218 = x217 < 0.0f;
if (x218) {
x214[x216] = 0.0f;
} else {
float x221 = x110[x216];
x214[x216] = x221;
}

}
float* x227 = (float*)myMalloc(1572864 * sizeof(float));;
for(int x229=0; x229 < 1572864; x229++) {
x227[x229] = -3.4028235E38f;

}
int* x233 = (int32_t*)myMalloc(1572864 * sizeof(int32_t));;
for(int x234=0; x234 < 64; x234++) {
int32_t x235 = x234 * 98304;
float* x236 = x214+x235;
int32_t x237 = x234 * 24576;
float* x238 = x227+x237;
int* x239 = x233+x237;
int32_t x240 = 0;
int32_t x241 = 0;
for(int x242=0; x242 < 96; x242++) {
int32_t x243 = x240;
int32_t x244 = x243;
int32_t x245 = x241;
int32_t x246 = x245;
for(int x248=0; x248 < 16; x248++) {
int32_t x249 = x244;
int32_t x250 = x249;
int32_t x251 = x246;
int32_t x252 = x251;
for(int x253=0; x253 < 16; x253++) {
int32_t x254 = x252;
int32_t x255 = x254;
int32_t x256 = x255;
int32_t x257 = x256;
int32_t x258 = x257;
float x259 = x236[x258];
int32_t x260 = x250;
float x261 = x238[x260];
bool x262 = x259 > x261;
if (x262) {
float x263 = x236[x258];
x238[x260] = x263;
int32_t x265 = x258 + x235;
x239[x260] = x265;
} else {
}
x257 += 1;
int32_t x270 = x257;
float x271 = x236[x270];
float x272 = x238[x260];
bool x273 = x271 > x272;
if (x273) {
float x274 = x236[x270];
x238[x260] = x274;
int32_t x276 = x270 + x235;
x239[x260] = x276;
} else {
}
x257 += 1;
x255 += 32;
int32_t x282 = x255;
int32_t x283 = x282;
int32_t x284 = x283;
float x285 = x236[x284];
float x286 = x238[x260];
bool x287 = x285 > x286;
if (x287) {
float x288 = x236[x284];
x238[x260] = x288;
int32_t x290 = x284 + x235;
x239[x260] = x290;
} else {
}
x283 += 1;
int32_t x295 = x283;
float x296 = x236[x295];
float x297 = x238[x260];
bool x298 = x296 > x297;
if (x298) {
float x299 = x236[x295];
x238[x260] = x299;
int32_t x301 = x295 + x235;
x239[x260] = x301;
} else {
}
x283 += 1;
x255 += 32;
x250 += 1;
x252 += 2;

}
x244 += 16;
x246 += 64;

}
x240 += 256;
x241 += 1024;

}

}
float* x321 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x322 = 0;
for(int x323=0; x323 < 64; x323++) {
for(int x324=0; x324 < 16; x324++) {
for(int x326=0; x326 < 256; x326++) {
int32_t x327 = x322;
float x328 = x50[x324];
x321[x327] = x328;
x322 += 1;

}

}

}
float* x337 = (float*)myMalloc(1572864 * sizeof(float));;
for(int x338=0; x338 < 64; x338++) {
int32_t x339 = x338 * 24576;
float* x340 = x227+x339;
int32_t x341 = x338 * 4096;
float* x342 = x321+x341;
float* x343 = x337+x339;
for(int x344=0; x344 < 96; x344++) {
int32_t x345 = x344 / 1;
int32_t x349 = x345 * 16;
int32_t x350 = x349 * 16;
int32_t x346 = x344 % 1;
int32_t x347 = x346 / 1;
int32_t x351 = x347 * 16;
int32_t x352 = x351 * 16;
int32_t x353 = x350 + x352;
int32_t x348 = x346 % 1;
int32_t x354 = x348 * 16;
int32_t x355 = x354 * 16;
int32_t x356 = x353 + x355;
float* x357 = x343+x356;
float* x358 = x340+x350;
for(int x359=0; x359 < 16; x359++) {
int32_t x361 = x359 * 16;
float* x362 = x357+x361;
int32_t x360 = x359 + x347;
int32_t x363 = x360 * 16;
int32_t x364 = x363 + x348;
float* x365 = x358+x364;
memcpy(x362, x365, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 16,256,96,1,x92,96,x343,256,1,x342,256);

}
float* x374 = (float*)myMalloc(262144 * sizeof(float));;
for(int x376=0; x376 < 262144; x376++) {
float x377 = x321[x376];
bool x378 = x377 < 0.0f;
if (x378) {
x374[x376] = 0.0f;
} else {
float x381 = x321[x376];
x374[x376] = x381;
}

}
float* x387 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x388 = 0;
for(int x389=0; x389 < 64; x389++) {
for(int x390=0; x390 < 64; x390++) {
for(int x391=0; x391 < 256; x391++) {
int32_t x392 = x388;
float x393 = x73[x390];
x387[x392] = x393;
x388 += 1;

}

}

}
float* x402 = (float*)myMalloc(262144 * sizeof(float));;
for(int x403=0; x403 < 64; x403++) {
int32_t x404 = x403 * 4096;
float* x405 = x374+x404;
int32_t x406 = x403 * 16384;
float* x407 = x387+x406;
float* x408 = x402+x404;
for(int x409=0; x409 < 16; x409++) {
int32_t x410 = x409 / 1;
int32_t x414 = x410 * 16;
int32_t x415 = x414 * 16;
int32_t x411 = x409 % 1;
int32_t x412 = x411 / 1;
int32_t x416 = x412 * 16;
int32_t x417 = x416 * 16;
int32_t x418 = x415 + x417;
int32_t x413 = x411 % 1;
int32_t x419 = x413 * 16;
int32_t x420 = x419 * 16;
int32_t x421 = x418 + x420;
float* x422 = x408+x421;
float* x423 = x405+x415;
for(int x424=0; x424 < 16; x424++) {
int32_t x426 = x424 * 16;
float* x427 = x422+x426;
int32_t x425 = x424 + x412;
int32_t x428 = x425 * 16;
int32_t x429 = x428 + x413;
float* x430 = x423+x429;
memcpy(x427, x430, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,16,1,x66,16,x408,256,1,x407,256);

}
float* x439 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x441=0; x441 < 1048576; x441++) {
float x442 = x387[x441];
bool x443 = x442 < 0.0f;
if (x443) {
x439[x441] = 0.0f;
} else {
float x446 = x387[x441];
x439[x441] = x446;
}

}
float* x452 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x453 = 0;
for(int x454=0; x454 < 64; x454++) {
for(int x455=0; x455 < 64; x455++) {
for(int x456=0; x456 < 256; x456++) {
int32_t x457 = x453;
float x458 = x47[x455];
x452[x457] = x458;
x453 += 1;

}

}

}
float* x467 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x468=0; x468 < 64; x468++) {
int32_t x469 = x468 * 4096;
float* x470 = x374+x469;
int32_t x471 = x468 * 16384;
float* x472 = x452+x471;
int32_t x473 = x468 * 36864;
float* x474 = x467+x473;
for(int x476=0; x476 < 144; x476++) {
int32_t x477 = x476 / 9;
int32_t x481 = x477 * 3;
int32_t x482 = x481 * 3;
int32_t x483 = x482 * 16;
int32_t x484 = x483 * 16;
int32_t x478 = x476 % 9;
int32_t x479 = x478 / 3;
int32_t x485 = x479 * 3;
int32_t x486 = x485 * 16;
int32_t x487 = x486 * 16;
int32_t x488 = x484 + x487;
int32_t x480 = x478 % 3;
int32_t x489 = x480 * 16;
int32_t x490 = x489 * 16;
int32_t x491 = x488 + x490;
float* x492 = x474+x491;
int32_t x493 = x477 * 16;
int32_t x494 = x493 * 16;
float* x495 = x470+x494;
int32_t x507 = 1 - x480;
bool x508 = x507 > 0;
int32_t x509;
if (x508) {
x509 = x507;
} else {
x509 = 0;
}
int32_t x510 = 3 - x480;
int32_t x511 = x510 - 1;
int32_t x512 = 1 - x511;
bool x513 = x512 > 0;
int32_t x514;
if (x513) {
x514 = x512;
} else {
x514 = 0;
}
int32_t x515 = 16 - x514;
int32_t x516 = x515 - x509;
bool x517 = x516 <= 0;
bool x521 = x509 > 0;
int32_t x506 = -1 + x480;
bool x534 = x514 > 0;
for(int x496=0; x496 < 16; x496++) {
int32_t x497 = x496 - 1;
int32_t x498 = x497 + x479;
bool x499 = x498 < 0;
bool x500 = x498 >= 16;
bool x501 = x499 || x500;
if (x501) {
int32_t x502 = x496 * 16;
float* x503 = x492+x502;
memset(x503, 0, 4 * 16);;
} else {
if (x517) {
int32_t x502 = x496 * 16;
float* x518 = x492+x502;
memset(x518, 0, 4 * 16);;
} else {
int32_t x502 = x496 * 16;
if (x521) {
float* x522 = x492+x502;
memset(x522, 0, 4 * x509);;
} else {
}
// may have segfault here
int32_t x527 = x502 + x509;
float* x528 = x492+x527;
int32_t x529 = x498 * 16;
int32_t x530 = x529 + x506;
int32_t x531 = x530 + x509;
float* x532 = x495+x531;
memcpy(x528, x532, 4 * x516);;
if (x534) {
int32_t x535 = x502 + 16;
int32_t x536 = x535 - x514;
float* x537 = x492+x536;
memset(x537, 0, 4 * x514);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,144,1,x89,144,x474,256,1,x472,256);

}
float* x552 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x553=0; x553 < 1048576; x553++) {
float x554 = x452[x553];
bool x555 = x554 < 0.0f;
if (x555) {
x552[x553] = 0.0f;
} else {
float x558 = x452[x553];
x552[x553] = x558;
}

}
float* x564 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x565 = 0;
for(int x566=0; x566 < 64; x566++) {
int32_t x567 = x566 * 16384;
float* x568 = x439+x567;
for(int x570=0; x570 < 16384; x570++) {
int32_t x571 = x565;
float x572 = x568[x570];
x564[x571] = x572;
x565 += 1;

}
float* x577 = x552+x567;
for(int x578=0; x578 < 16384; x578++) {
int32_t x579 = x565;
float x580 = x577[x578];
x564[x579] = x580;
x565 += 1;

}

}
float* x587 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x588 = 0;
for(int x589=0; x589 < 64; x589++) {
for(int x590=0; x590 < 16; x590++) {
for(int x591=0; x591 < 256; x591++) {
int32_t x592 = x588;
float x593 = x67[x590];
x587[x592] = x593;
x588 += 1;

}

}

}
float* x602 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x603=0; x603 < 64; x603++) {
int32_t x604 = x603 * 32768;
float* x605 = x564+x604;
int32_t x606 = x603 * 4096;
float* x607 = x587+x606;
float* x608 = x602+x604;
for(int x610=0; x610 < 128; x610++) {
int32_t x611 = x610 / 1;
int32_t x615 = x611 * 16;
int32_t x616 = x615 * 16;
int32_t x612 = x610 % 1;
int32_t x613 = x612 / 1;
int32_t x617 = x613 * 16;
int32_t x618 = x617 * 16;
int32_t x619 = x616 + x618;
int32_t x614 = x612 % 1;
int32_t x620 = x614 * 16;
int32_t x621 = x620 * 16;
int32_t x622 = x619 + x621;
float* x623 = x608+x622;
float* x624 = x605+x616;
for(int x625=0; x625 < 16; x625++) {
int32_t x627 = x625 * 16;
float* x628 = x623+x627;
int32_t x626 = x625 + x613;
int32_t x629 = x626 * 16;
int32_t x630 = x629 + x614;
float* x631 = x624+x630;
memcpy(x628, x631, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 16,256,128,1,x54,128,x608,256,1,x607,256);

}
float* x640 = (float*)myMalloc(262144 * sizeof(float));;
for(int x641=0; x641 < 262144; x641++) {
float x642 = x587[x641];
bool x643 = x642 < 0.0f;
if (x643) {
x640[x641] = 0.0f;
} else {
float x646 = x587[x641];
x640[x641] = x646;
}

}
float* x652 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x653 = 0;
for(int x654=0; x654 < 64; x654++) {
for(int x655=0; x655 < 64; x655++) {
for(int x656=0; x656 < 256; x656++) {
int32_t x657 = x653;
float x658 = x45[x655];
x652[x657] = x658;
x653 += 1;

}

}

}
float* x667 = (float*)myMalloc(262144 * sizeof(float));;
for(int x668=0; x668 < 64; x668++) {
int32_t x669 = x668 * 4096;
float* x670 = x640+x669;
int32_t x671 = x668 * 16384;
float* x672 = x652+x671;
float* x673 = x667+x669;
for(int x674=0; x674 < 16; x674++) {
int32_t x675 = x674 / 1;
int32_t x679 = x675 * 16;
int32_t x680 = x679 * 16;
int32_t x676 = x674 % 1;
int32_t x677 = x676 / 1;
int32_t x681 = x677 * 16;
int32_t x682 = x681 * 16;
int32_t x683 = x680 + x682;
int32_t x678 = x676 % 1;
int32_t x684 = x678 * 16;
int32_t x685 = x684 * 16;
int32_t x686 = x683 + x685;
float* x687 = x673+x686;
float* x688 = x670+x680;
for(int x689=0; x689 < 16; x689++) {
int32_t x691 = x689 * 16;
float* x692 = x687+x691;
int32_t x690 = x689 + x677;
int32_t x693 = x690 * 16;
int32_t x694 = x693 + x678;
float* x695 = x688+x694;
memcpy(x692, x695, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,16,1,x53,16,x673,256,1,x672,256);

}
float* x704 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x705=0; x705 < 1048576; x705++) {
float x706 = x652[x705];
bool x707 = x706 < 0.0f;
if (x707) {
x704[x705] = 0.0f;
} else {
float x710 = x652[x705];
x704[x705] = x710;
}

}
float* x716 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x717 = 0;
for(int x718=0; x718 < 64; x718++) {
for(int x719=0; x719 < 64; x719++) {
for(int x720=0; x720 < 256; x720++) {
int32_t x721 = x717;
float x722 = x79[x719];
x716[x721] = x722;
x717 += 1;

}

}

}
float* x731 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x732=0; x732 < 64; x732++) {
int32_t x733 = x732 * 4096;
float* x734 = x640+x733;
int32_t x735 = x732 * 16384;
float* x736 = x716+x735;
int32_t x737 = x732 * 36864;
float* x738 = x731+x737;
for(int x739=0; x739 < 144; x739++) {
int32_t x740 = x739 / 9;
int32_t x744 = x740 * 3;
int32_t x745 = x744 * 3;
int32_t x746 = x745 * 16;
int32_t x747 = x746 * 16;
int32_t x741 = x739 % 9;
int32_t x742 = x741 / 3;
int32_t x748 = x742 * 3;
int32_t x749 = x748 * 16;
int32_t x750 = x749 * 16;
int32_t x751 = x747 + x750;
int32_t x743 = x741 % 3;
int32_t x752 = x743 * 16;
int32_t x753 = x752 * 16;
int32_t x754 = x751 + x753;
float* x755 = x738+x754;
int32_t x756 = x740 * 16;
int32_t x757 = x756 * 16;
float* x758 = x734+x757;
int32_t x770 = 1 - x743;
bool x771 = x770 > 0;
int32_t x772;
if (x771) {
x772 = x770;
} else {
x772 = 0;
}
int32_t x773 = 3 - x743;
int32_t x774 = x773 - 1;
int32_t x775 = 1 - x774;
bool x776 = x775 > 0;
int32_t x777;
if (x776) {
x777 = x775;
} else {
x777 = 0;
}
int32_t x778 = 16 - x777;
int32_t x779 = x778 - x772;
bool x780 = x779 <= 0;
bool x784 = x772 > 0;
int32_t x769 = -1 + x743;
bool x797 = x777 > 0;
for(int x759=0; x759 < 16; x759++) {
int32_t x760 = x759 - 1;
int32_t x761 = x760 + x742;
bool x762 = x761 < 0;
bool x763 = x761 >= 16;
bool x764 = x762 || x763;
if (x764) {
int32_t x765 = x759 * 16;
float* x766 = x755+x765;
memset(x766, 0, 4 * 16);;
} else {
if (x780) {
int32_t x765 = x759 * 16;
float* x781 = x755+x765;
memset(x781, 0, 4 * 16);;
} else {
int32_t x765 = x759 * 16;
if (x784) {
float* x785 = x755+x765;
memset(x785, 0, 4 * x772);;
} else {
}
// may have segfault here
int32_t x790 = x765 + x772;
float* x791 = x755+x790;
int32_t x792 = x761 * 16;
int32_t x793 = x792 + x769;
int32_t x794 = x793 + x772;
float* x795 = x758+x794;
memcpy(x791, x795, 4 * x779);;
if (x797) {
int32_t x798 = x765 + 16;
int32_t x799 = x798 - x777;
float* x800 = x755+x799;
memset(x800, 0, 4 * x777);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,144,1,x61,144,x738,256,1,x736,256);

}
float* x815 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x816=0; x816 < 1048576; x816++) {
float x817 = x716[x816];
bool x818 = x817 < 0.0f;
if (x818) {
x815[x816] = 0.0f;
} else {
float x821 = x716[x816];
x815[x816] = x821;
}

}
float* x827 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x828 = 0;
for(int x829=0; x829 < 64; x829++) {
int32_t x830 = x829 * 16384;
float* x831 = x704+x830;
for(int x832=0; x832 < 16384; x832++) {
int32_t x833 = x828;
float x834 = x831[x832];
x827[x833] = x834;
x828 += 1;

}
float* x839 = x815+x830;
for(int x840=0; x840 < 16384; x840++) {
int32_t x841 = x828;
float x842 = x839[x840];
x827[x841] = x842;
x828 += 1;

}

}
float* x849 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x850 = 0;
for(int x851=0; x851 < 64; x851++) {
for(int x852=0; x852 < 32; x852++) {
for(int x853=0; x853 < 256; x853++) {
int32_t x854 = x850;
float x855 = x65[x852];
x849[x854] = x855;
x850 += 1;

}

}

}
float* x864 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x865=0; x865 < 64; x865++) {
int32_t x866 = x865 * 32768;
float* x867 = x827+x866;
int32_t x868 = x865 * 8192;
float* x869 = x849+x868;
float* x870 = x864+x866;
for(int x871=0; x871 < 128; x871++) {
int32_t x872 = x871 / 1;
int32_t x876 = x872 * 16;
int32_t x877 = x876 * 16;
int32_t x873 = x871 % 1;
int32_t x874 = x873 / 1;
int32_t x878 = x874 * 16;
int32_t x879 = x878 * 16;
int32_t x880 = x877 + x879;
int32_t x875 = x873 % 1;
int32_t x881 = x875 * 16;
int32_t x882 = x881 * 16;
int32_t x883 = x880 + x882;
float* x884 = x870+x883;
float* x885 = x867+x877;
for(int x886=0; x886 < 16; x886++) {
int32_t x888 = x886 * 16;
float* x889 = x884+x888;
int32_t x887 = x886 + x874;
int32_t x890 = x887 * 16;
int32_t x891 = x890 + x875;
float* x892 = x885+x891;
memcpy(x889, x892, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 32,256,128,1,x52,128,x870,256,1,x869,256);

}
float* x901 = (float*)myMalloc(524288 * sizeof(float));;
for(int x903=0; x903 < 524288; x903++) {
float x904 = x849[x903];
bool x905 = x904 < 0.0f;
if (x905) {
x901[x903] = 0.0f;
} else {
float x908 = x849[x903];
x901[x903] = x908;
}

}
float* x914 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x915 = 0;
for(int x916=0; x916 < 64; x916++) {
for(int x917=0; x917 < 128; x917++) {
for(int x918=0; x918 < 256; x918++) {
int32_t x919 = x915;
float x920 = x87[x917];
x914[x919] = x920;
x915 += 1;

}

}

}
float* x929 = (float*)myMalloc(524288 * sizeof(float));;
for(int x930=0; x930 < 64; x930++) {
int32_t x931 = x930 * 8192;
float* x932 = x901+x931;
int32_t x933 = x930 * 32768;
float* x934 = x914+x933;
float* x935 = x929+x931;
for(int x936=0; x936 < 32; x936++) {
int32_t x937 = x936 / 1;
int32_t x941 = x937 * 16;
int32_t x942 = x941 * 16;
int32_t x938 = x936 % 1;
int32_t x939 = x938 / 1;
int32_t x943 = x939 * 16;
int32_t x944 = x943 * 16;
int32_t x945 = x942 + x944;
int32_t x940 = x938 % 1;
int32_t x946 = x940 * 16;
int32_t x947 = x946 * 16;
int32_t x948 = x945 + x947;
float* x949 = x935+x948;
float* x950 = x932+x942;
for(int x951=0; x951 < 16; x951++) {
int32_t x953 = x951 * 16;
float* x954 = x949+x953;
int32_t x952 = x951 + x939;
int32_t x955 = x952 * 16;
int32_t x956 = x955 + x940;
float* x957 = x950+x956;
memcpy(x954, x957, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,256,32,1,x77,32,x935,256,1,x934,256);

}
float* x966 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x968=0; x968 < 2097152; x968++) {
float x969 = x914[x968];
bool x970 = x969 < 0.0f;
if (x970) {
x966[x968] = 0.0f;
} else {
float x973 = x914[x968];
x966[x968] = x973;
}

}
float* x979 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x980 = 0;
for(int x981=0; x981 < 64; x981++) {
for(int x982=0; x982 < 128; x982++) {
for(int x983=0; x983 < 256; x983++) {
int32_t x984 = x980;
float x985 = x83[x982];
x979[x984] = x985;
x980 += 1;

}

}

}
float* x994 = (float*)myMalloc(4718592 * sizeof(float));;
for(int x995=0; x995 < 64; x995++) {
int32_t x996 = x995 * 8192;
float* x997 = x901+x996;
int32_t x998 = x995 * 32768;
float* x999 = x979+x998;
int32_t x1000 = x995 * 73728;
float* x1001 = x994+x1000;
for(int x1003=0; x1003 < 288; x1003++) {
int32_t x1004 = x1003 / 9;
int32_t x1008 = x1004 * 3;
int32_t x1009 = x1008 * 3;
int32_t x1010 = x1009 * 16;
int32_t x1011 = x1010 * 16;
int32_t x1005 = x1003 % 9;
int32_t x1006 = x1005 / 3;
int32_t x1012 = x1006 * 3;
int32_t x1013 = x1012 * 16;
int32_t x1014 = x1013 * 16;
int32_t x1015 = x1011 + x1014;
int32_t x1007 = x1005 % 3;
int32_t x1016 = x1007 * 16;
int32_t x1017 = x1016 * 16;
int32_t x1018 = x1015 + x1017;
float* x1019 = x1001+x1018;
int32_t x1020 = x1004 * 16;
int32_t x1021 = x1020 * 16;
float* x1022 = x997+x1021;
int32_t x1034 = 1 - x1007;
bool x1035 = x1034 > 0;
int32_t x1036;
if (x1035) {
x1036 = x1034;
} else {
x1036 = 0;
}
int32_t x1037 = 3 - x1007;
int32_t x1038 = x1037 - 1;
int32_t x1039 = 1 - x1038;
bool x1040 = x1039 > 0;
int32_t x1041;
if (x1040) {
x1041 = x1039;
} else {
x1041 = 0;
}
int32_t x1042 = 16 - x1041;
int32_t x1043 = x1042 - x1036;
bool x1044 = x1043 <= 0;
bool x1048 = x1036 > 0;
int32_t x1033 = -1 + x1007;
bool x1061 = x1041 > 0;
for(int x1023=0; x1023 < 16; x1023++) {
int32_t x1024 = x1023 - 1;
int32_t x1025 = x1024 + x1006;
bool x1026 = x1025 < 0;
bool x1027 = x1025 >= 16;
bool x1028 = x1026 || x1027;
if (x1028) {
int32_t x1029 = x1023 * 16;
float* x1030 = x1019+x1029;
memset(x1030, 0, 4 * 16);;
} else {
if (x1044) {
int32_t x1029 = x1023 * 16;
float* x1045 = x1019+x1029;
memset(x1045, 0, 4 * 16);;
} else {
int32_t x1029 = x1023 * 16;
if (x1048) {
float* x1049 = x1019+x1029;
memset(x1049, 0, 4 * x1036);;
} else {
}
// may have segfault here
int32_t x1054 = x1029 + x1036;
float* x1055 = x1019+x1054;
int32_t x1056 = x1025 * 16;
int32_t x1057 = x1056 + x1033;
int32_t x1058 = x1057 + x1036;
float* x1059 = x1022+x1058;
memcpy(x1055, x1059, 4 * x1043);;
if (x1061) {
int32_t x1062 = x1029 + 16;
int32_t x1063 = x1062 - x1041;
float* x1064 = x1019+x1063;
memset(x1064, 0, 4 * x1041);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,256,288,1,x48,288,x1001,256,1,x999,256);

}
float* x1079 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x1080=0; x1080 < 2097152; x1080++) {
float x1081 = x979[x1080];
bool x1082 = x1081 < 0.0f;
if (x1082) {
x1079[x1080] = 0.0f;
} else {
float x1085 = x979[x1080];
x1079[x1080] = x1085;
}

}
float* x1091 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1092 = 0;
for(int x1093=0; x1093 < 64; x1093++) {
int32_t x1094 = x1093 * 32768;
float* x1095 = x966+x1094;
for(int x1097=0; x1097 < 32768; x1097++) {
int32_t x1098 = x1092;
float x1099 = x1095[x1097];
x1091[x1098] = x1099;
x1092 += 1;

}
float* x1104 = x1079+x1094;
for(int x1105=0; x1105 < 32768; x1105++) {
int32_t x1106 = x1092;
float x1107 = x1104[x1105];
x1091[x1106] = x1107;
x1092 += 1;

}

}
float* x1114 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x1115=0; x1115 < 1048576; x1115++) {
x1114[x1115] = -3.4028235E38f;

}
int* x1119 = (int32_t*)myMalloc(1048576 * sizeof(int32_t));;
for(int x1120=0; x1120 < 64; x1120++) {
int32_t x1121 = x1120 * 65536;
float* x1122 = x1091+x1121;
int32_t x1123 = x1120 * 16384;
float* x1124 = x1114+x1123;
int* x1125 = x1119+x1123;
int32_t x1126 = 0;
int32_t x1127 = 0;
for(int x1128=0; x1128 < 256; x1128++) {
int32_t x1129 = x1126;
int32_t x1130 = x1129;
int32_t x1131 = x1127;
int32_t x1132 = x1131;
for(int x1134=0; x1134 < 8; x1134++) {
int32_t x1135 = x1130;
int32_t x1136 = x1135;
int32_t x1137 = x1132;
int32_t x1138 = x1137;
for(int x1139=0; x1139 < 8; x1139++) {
int32_t x1140 = x1138;
int32_t x1141 = x1140;
int32_t x1142 = x1141;
int32_t x1143 = x1142;
int32_t x1144 = x1143;
float x1145 = x1122[x1144];
int32_t x1146 = x1136;
float x1147 = x1124[x1146];
bool x1148 = x1145 > x1147;
if (x1148) {
float x1149 = x1122[x1144];
x1124[x1146] = x1149;
int32_t x1151 = x1144 + x1121;
x1125[x1146] = x1151;
} else {
}
x1143 += 1;
int32_t x1156 = x1143;
float x1157 = x1122[x1156];
float x1158 = x1124[x1146];
bool x1159 = x1157 > x1158;
if (x1159) {
float x1160 = x1122[x1156];
x1124[x1146] = x1160;
int32_t x1162 = x1156 + x1121;
x1125[x1146] = x1162;
} else {
}
x1143 += 1;
x1141 += 16;
int32_t x1168 = x1141;
int32_t x1169 = x1168;
int32_t x1170 = x1169;
float x1171 = x1122[x1170];
float x1172 = x1124[x1146];
bool x1173 = x1171 > x1172;
if (x1173) {
float x1174 = x1122[x1170];
x1124[x1146] = x1174;
int32_t x1176 = x1170 + x1121;
x1125[x1146] = x1176;
} else {
}
x1169 += 1;
int32_t x1181 = x1169;
float x1182 = x1122[x1181];
float x1183 = x1124[x1146];
bool x1184 = x1182 > x1183;
if (x1184) {
float x1185 = x1122[x1181];
x1124[x1146] = x1185;
int32_t x1187 = x1181 + x1121;
x1125[x1146] = x1187;
} else {
}
x1169 += 1;
x1141 += 16;
x1136 += 1;
x1138 += 2;

}
x1130 += 8;
x1132 += 32;

}
x1126 += 64;
x1127 += 256;

}

}
float* x1207 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x1208 = 0;
for(int x1209=0; x1209 < 64; x1209++) {
for(int x1210=0; x1210 < 32; x1210++) {
for(int x1211=0; x1211 < 64; x1211++) {
int32_t x1212 = x1208;
float x1213 = x57[x1210];
x1207[x1212] = x1213;
x1208 += 1;

}

}

}
float* x1222 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x1223=0; x1223 < 64; x1223++) {
int32_t x1224 = x1223 * 16384;
float* x1225 = x1114+x1224;
int32_t x1226 = x1223 * 2048;
float* x1227 = x1207+x1226;
float* x1228 = x1222+x1224;
for(int x1229=0; x1229 < 256; x1229++) {
int32_t x1230 = x1229 / 1;
int32_t x1234 = x1230 * 8;
int32_t x1235 = x1234 * 8;
int32_t x1231 = x1229 % 1;
int32_t x1232 = x1231 / 1;
int32_t x1236 = x1232 * 8;
int32_t x1237 = x1236 * 8;
int32_t x1238 = x1235 + x1237;
int32_t x1233 = x1231 % 1;
int32_t x1239 = x1233 * 8;
int32_t x1240 = x1239 * 8;
int32_t x1241 = x1238 + x1240;
float* x1242 = x1228+x1241;
float* x1243 = x1225+x1235;
for(int x1244=0; x1244 < 8; x1244++) {
int32_t x1246 = x1244 * 8;
float* x1247 = x1242+x1246;
int32_t x1245 = x1244 + x1232;
int32_t x1248 = x1245 * 8;
int32_t x1249 = x1248 + x1233;
float* x1250 = x1243+x1249;
memcpy(x1247, x1250, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 32,64,256,1,x69,256,x1228,64,1,x1227,64);

}
float* x1259 = (float*)myMalloc(131072 * sizeof(float));;
for(int x1261=0; x1261 < 131072; x1261++) {
float x1262 = x1207[x1261];
bool x1263 = x1262 < 0.0f;
if (x1263) {
x1259[x1261] = 0.0f;
} else {
float x1266 = x1207[x1261];
x1259[x1261] = x1266;
}

}
float* x1272 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x1273 = 0;
for(int x1274=0; x1274 < 64; x1274++) {
for(int x1275=0; x1275 < 128; x1275++) {
for(int x1276=0; x1276 < 64; x1276++) {
int32_t x1277 = x1273;
float x1278 = x63[x1275];
x1272[x1277] = x1278;
x1273 += 1;

}

}

}
float* x1287 = (float*)myMalloc(131072 * sizeof(float));;
for(int x1288=0; x1288 < 64; x1288++) {
int32_t x1289 = x1288 * 2048;
float* x1290 = x1259+x1289;
int32_t x1291 = x1288 * 8192;
float* x1292 = x1272+x1291;
float* x1293 = x1287+x1289;
for(int x1294=0; x1294 < 32; x1294++) {
int32_t x1295 = x1294 / 1;
int32_t x1299 = x1295 * 8;
int32_t x1300 = x1299 * 8;
int32_t x1296 = x1294 % 1;
int32_t x1297 = x1296 / 1;
int32_t x1301 = x1297 * 8;
int32_t x1302 = x1301 * 8;
int32_t x1303 = x1300 + x1302;
int32_t x1298 = x1296 % 1;
int32_t x1304 = x1298 * 8;
int32_t x1305 = x1304 * 8;
int32_t x1306 = x1303 + x1305;
float* x1307 = x1293+x1306;
float* x1308 = x1290+x1300;
for(int x1309=0; x1309 < 8; x1309++) {
int32_t x1311 = x1309 * 8;
float* x1312 = x1307+x1311;
int32_t x1310 = x1309 + x1297;
int32_t x1313 = x1310 * 8;
int32_t x1314 = x1313 + x1298;
float* x1315 = x1308+x1314;
memcpy(x1312, x1315, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,32,1,x49,32,x1293,64,1,x1292,64);

}
float* x1324 = (float*)myMalloc(524288 * sizeof(float));;
for(int x1325=0; x1325 < 524288; x1325++) {
float x1326 = x1272[x1325];
bool x1327 = x1326 < 0.0f;
if (x1327) {
x1324[x1325] = 0.0f;
} else {
float x1330 = x1272[x1325];
x1324[x1325] = x1330;
}

}
float* x1336 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x1337 = 0;
for(int x1338=0; x1338 < 64; x1338++) {
for(int x1339=0; x1339 < 128; x1339++) {
for(int x1340=0; x1340 < 64; x1340++) {
int32_t x1341 = x1337;
float x1342 = x58[x1339];
x1336[x1341] = x1342;
x1337 += 1;

}

}

}
float* x1351 = (float*)myMalloc(1179648 * sizeof(float));;
for(int x1352=0; x1352 < 64; x1352++) {
int32_t x1353 = x1352 * 2048;
float* x1354 = x1259+x1353;
int32_t x1355 = x1352 * 8192;
float* x1356 = x1336+x1355;
int32_t x1357 = x1352 * 18432;
float* x1358 = x1351+x1357;
for(int x1359=0; x1359 < 288; x1359++) {
int32_t x1360 = x1359 / 9;
int32_t x1364 = x1360 * 3;
int32_t x1365 = x1364 * 3;
int32_t x1366 = x1365 * 8;
int32_t x1367 = x1366 * 8;
int32_t x1361 = x1359 % 9;
int32_t x1362 = x1361 / 3;
int32_t x1368 = x1362 * 3;
int32_t x1369 = x1368 * 8;
int32_t x1370 = x1369 * 8;
int32_t x1371 = x1367 + x1370;
int32_t x1363 = x1361 % 3;
int32_t x1372 = x1363 * 8;
int32_t x1373 = x1372 * 8;
int32_t x1374 = x1371 + x1373;
float* x1375 = x1358+x1374;
int32_t x1376 = x1360 * 8;
int32_t x1377 = x1376 * 8;
float* x1378 = x1354+x1377;
int32_t x1390 = 1 - x1363;
bool x1391 = x1390 > 0;
int32_t x1392;
if (x1391) {
x1392 = x1390;
} else {
x1392 = 0;
}
int32_t x1393 = 3 - x1363;
int32_t x1394 = x1393 - 1;
int32_t x1395 = 1 - x1394;
bool x1396 = x1395 > 0;
int32_t x1397;
if (x1396) {
x1397 = x1395;
} else {
x1397 = 0;
}
int32_t x1398 = 8 - x1397;
int32_t x1399 = x1398 - x1392;
bool x1400 = x1399 <= 0;
bool x1404 = x1392 > 0;
int32_t x1389 = -1 + x1363;
bool x1417 = x1397 > 0;
for(int x1379=0; x1379 < 8; x1379++) {
int32_t x1380 = x1379 - 1;
int32_t x1381 = x1380 + x1362;
bool x1382 = x1381 < 0;
bool x1383 = x1381 >= 8;
bool x1384 = x1382 || x1383;
if (x1384) {
int32_t x1385 = x1379 * 8;
float* x1386 = x1375+x1385;
memset(x1386, 0, 4 * 8);;
} else {
if (x1400) {
int32_t x1385 = x1379 * 8;
float* x1401 = x1375+x1385;
memset(x1401, 0, 4 * 8);;
} else {
int32_t x1385 = x1379 * 8;
if (x1404) {
float* x1405 = x1375+x1385;
memset(x1405, 0, 4 * x1392);;
} else {
}
// may have segfault here
int32_t x1410 = x1385 + x1392;
float* x1411 = x1375+x1410;
int32_t x1412 = x1381 * 8;
int32_t x1413 = x1412 + x1389;
int32_t x1414 = x1413 + x1392;
float* x1415 = x1378+x1414;
memcpy(x1411, x1415, 4 * x1399);;
if (x1417) {
int32_t x1418 = x1385 + 8;
int32_t x1419 = x1418 - x1397;
float* x1420 = x1375+x1419;
memset(x1420, 0, 4 * x1397);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,288,1,x78,288,x1358,64,1,x1356,64);

}
float* x1435 = (float*)myMalloc(524288 * sizeof(float));;
for(int x1436=0; x1436 < 524288; x1436++) {
float x1437 = x1336[x1436];
bool x1438 = x1437 < 0.0f;
if (x1438) {
x1435[x1436] = 0.0f;
} else {
float x1441 = x1336[x1436];
x1435[x1436] = x1441;
}

}
float* x1447 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x1448 = 0;
for(int x1449=0; x1449 < 64; x1449++) {
int32_t x1450 = x1449 * 8192;
float* x1451 = x1324+x1450;
for(int x1453=0; x1453 < 8192; x1453++) {
int32_t x1454 = x1448;
float x1455 = x1451[x1453];
x1447[x1454] = x1455;
x1448 += 1;

}
float* x1460 = x1435+x1450;
for(int x1461=0; x1461 < 8192; x1461++) {
int32_t x1462 = x1448;
float x1463 = x1460[x1461];
x1447[x1462] = x1463;
x1448 += 1;

}

}
float* x1470 = (float*)myMalloc(196608 * sizeof(float));;
int32_t x1471 = 0;
for(int x1472=0; x1472 < 64; x1472++) {
for(int x1474=0; x1474 < 48; x1474++) {
for(int x1475=0; x1475 < 64; x1475++) {
int32_t x1476 = x1471;
float x1477 = x94[x1474];
x1470[x1476] = x1477;
x1471 += 1;

}

}

}
float* x1486 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x1487=0; x1487 < 64; x1487++) {
int32_t x1488 = x1487 * 16384;
float* x1489 = x1447+x1488;
int32_t x1490 = x1487 * 3072;
float* x1491 = x1470+x1490;
float* x1492 = x1486+x1488;
for(int x1493=0; x1493 < 256; x1493++) {
int32_t x1494 = x1493 / 1;
int32_t x1498 = x1494 * 8;
int32_t x1499 = x1498 * 8;
int32_t x1495 = x1493 % 1;
int32_t x1496 = x1495 / 1;
int32_t x1500 = x1496 * 8;
int32_t x1501 = x1500 * 8;
int32_t x1502 = x1499 + x1501;
int32_t x1497 = x1495 % 1;
int32_t x1503 = x1497 * 8;
int32_t x1504 = x1503 * 8;
int32_t x1505 = x1502 + x1504;
float* x1506 = x1492+x1505;
float* x1507 = x1489+x1499;
for(int x1508=0; x1508 < 8; x1508++) {
int32_t x1510 = x1508 * 8;
float* x1511 = x1506+x1510;
int32_t x1509 = x1508 + x1496;
int32_t x1512 = x1509 * 8;
int32_t x1513 = x1512 + x1497;
float* x1514 = x1507+x1513;
memcpy(x1511, x1514, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 48,64,256,1,x84,256,x1492,64,1,x1491,64);

}
float* x1523 = (float*)myMalloc(196608 * sizeof(float));;
for(int x1525=0; x1525 < 196608; x1525++) {
float x1526 = x1470[x1525];
bool x1527 = x1526 < 0.0f;
if (x1527) {
x1523[x1525] = 0.0f;
} else {
float x1530 = x1470[x1525];
x1523[x1525] = x1530;
}

}
float* x1536 = (float*)myMalloc(786432 * sizeof(float));;
int32_t x1537 = 0;
for(int x1538=0; x1538 < 64; x1538++) {
for(int x1540=0; x1540 < 192; x1540++) {
for(int x1541=0; x1541 < 64; x1541++) {
int32_t x1542 = x1537;
float x1543 = x88[x1540];
x1536[x1542] = x1543;
x1537 += 1;

}

}

}
float* x1552 = (float*)myMalloc(196608 * sizeof(float));;
for(int x1553=0; x1553 < 64; x1553++) {
int32_t x1554 = x1553 * 3072;
float* x1555 = x1523+x1554;
int32_t x1556 = x1553 * 12288;
float* x1557 = x1536+x1556;
float* x1558 = x1552+x1554;
for(int x1559=0; x1559 < 48; x1559++) {
int32_t x1560 = x1559 / 1;
int32_t x1564 = x1560 * 8;
int32_t x1565 = x1564 * 8;
int32_t x1561 = x1559 % 1;
int32_t x1562 = x1561 / 1;
int32_t x1566 = x1562 * 8;
int32_t x1567 = x1566 * 8;
int32_t x1568 = x1565 + x1567;
int32_t x1563 = x1561 % 1;
int32_t x1569 = x1563 * 8;
int32_t x1570 = x1569 * 8;
int32_t x1571 = x1568 + x1570;
float* x1572 = x1558+x1571;
float* x1573 = x1555+x1565;
for(int x1574=0; x1574 < 8; x1574++) {
int32_t x1576 = x1574 * 8;
float* x1577 = x1572+x1576;
int32_t x1575 = x1574 + x1562;
int32_t x1578 = x1575 * 8;
int32_t x1579 = x1578 + x1563;
float* x1580 = x1573+x1579;
memcpy(x1577, x1580, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 192,64,48,1,x90,48,x1558,64,1,x1557,64);

}
float* x1589 = (float*)myMalloc(786432 * sizeof(float));;
for(int x1591=0; x1591 < 786432; x1591++) {
float x1592 = x1536[x1591];
bool x1593 = x1592 < 0.0f;
if (x1593) {
x1589[x1591] = 0.0f;
} else {
float x1596 = x1536[x1591];
x1589[x1591] = x1596;
}

}
float* x1602 = (float*)myMalloc(786432 * sizeof(float));;
int32_t x1603 = 0;
for(int x1604=0; x1604 < 64; x1604++) {
for(int x1605=0; x1605 < 192; x1605++) {
for(int x1606=0; x1606 < 64; x1606++) {
int32_t x1607 = x1603;
float x1608 = x71[x1605];
x1602[x1607] = x1608;
x1603 += 1;

}

}

}
float* x1617 = (float*)myMalloc(1769472 * sizeof(float));;
for(int x1618=0; x1618 < 64; x1618++) {
int32_t x1619 = x1618 * 3072;
float* x1620 = x1523+x1619;
int32_t x1621 = x1618 * 12288;
float* x1622 = x1602+x1621;
int32_t x1623 = x1618 * 27648;
float* x1624 = x1617+x1623;
for(int x1626=0; x1626 < 432; x1626++) {
int32_t x1627 = x1626 / 9;
int32_t x1631 = x1627 * 3;
int32_t x1632 = x1631 * 3;
int32_t x1633 = x1632 * 8;
int32_t x1634 = x1633 * 8;
int32_t x1628 = x1626 % 9;
int32_t x1629 = x1628 / 3;
int32_t x1635 = x1629 * 3;
int32_t x1636 = x1635 * 8;
int32_t x1637 = x1636 * 8;
int32_t x1638 = x1634 + x1637;
int32_t x1630 = x1628 % 3;
int32_t x1639 = x1630 * 8;
int32_t x1640 = x1639 * 8;
int32_t x1641 = x1638 + x1640;
float* x1642 = x1624+x1641;
int32_t x1643 = x1627 * 8;
int32_t x1644 = x1643 * 8;
float* x1645 = x1620+x1644;
int32_t x1657 = 1 - x1630;
bool x1658 = x1657 > 0;
int32_t x1659;
if (x1658) {
x1659 = x1657;
} else {
x1659 = 0;
}
int32_t x1660 = 3 - x1630;
int32_t x1661 = x1660 - 1;
int32_t x1662 = 1 - x1661;
bool x1663 = x1662 > 0;
int32_t x1664;
if (x1663) {
x1664 = x1662;
} else {
x1664 = 0;
}
int32_t x1665 = 8 - x1664;
int32_t x1666 = x1665 - x1659;
bool x1667 = x1666 <= 0;
bool x1671 = x1659 > 0;
int32_t x1656 = -1 + x1630;
bool x1684 = x1664 > 0;
for(int x1646=0; x1646 < 8; x1646++) {
int32_t x1647 = x1646 - 1;
int32_t x1648 = x1647 + x1629;
bool x1649 = x1648 < 0;
bool x1650 = x1648 >= 8;
bool x1651 = x1649 || x1650;
if (x1651) {
int32_t x1652 = x1646 * 8;
float* x1653 = x1642+x1652;
memset(x1653, 0, 4 * 8);;
} else {
if (x1667) {
int32_t x1652 = x1646 * 8;
float* x1668 = x1642+x1652;
memset(x1668, 0, 4 * 8);;
} else {
int32_t x1652 = x1646 * 8;
if (x1671) {
float* x1672 = x1642+x1652;
memset(x1672, 0, 4 * x1659);;
} else {
}
// may have segfault here
int32_t x1677 = x1652 + x1659;
float* x1678 = x1642+x1677;
int32_t x1679 = x1648 * 8;
int32_t x1680 = x1679 + x1656;
int32_t x1681 = x1680 + x1659;
float* x1682 = x1645+x1681;
memcpy(x1678, x1682, 4 * x1666);;
if (x1684) {
int32_t x1685 = x1652 + 8;
int32_t x1686 = x1685 - x1664;
float* x1687 = x1642+x1686;
memset(x1687, 0, 4 * x1664);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 192,64,432,1,x81,432,x1624,64,1,x1622,64);

}
float* x1702 = (float*)myMalloc(786432 * sizeof(float));;
for(int x1703=0; x1703 < 786432; x1703++) {
float x1704 = x1602[x1703];
bool x1705 = x1704 < 0.0f;
if (x1705) {
x1702[x1703] = 0.0f;
} else {
float x1708 = x1602[x1703];
x1702[x1703] = x1708;
}

}
float* x1714 = (float*)myMalloc(1572864 * sizeof(float));;
int32_t x1715 = 0;
for(int x1716=0; x1716 < 64; x1716++) {
int32_t x1717 = x1716 * 12288;
float* x1718 = x1589+x1717;
for(int x1720=0; x1720 < 12288; x1720++) {
int32_t x1721 = x1715;
float x1722 = x1718[x1720];
x1714[x1721] = x1722;
x1715 += 1;

}
float* x1727 = x1702+x1717;
for(int x1728=0; x1728 < 12288; x1728++) {
int32_t x1729 = x1715;
float x1730 = x1727[x1728];
x1714[x1729] = x1730;
x1715 += 1;

}

}
float* x1737 = (float*)myMalloc(196608 * sizeof(float));;
int32_t x1738 = 0;
for(int x1739=0; x1739 < 64; x1739++) {
for(int x1740=0; x1740 < 48; x1740++) {
for(int x1741=0; x1741 < 64; x1741++) {
int32_t x1742 = x1738;
float x1743 = x44[x1740];
x1737[x1742] = x1743;
x1738 += 1;

}

}

}
float* x1752 = (float*)myMalloc(1572864 * sizeof(float));;
for(int x1753=0; x1753 < 64; x1753++) {
int32_t x1754 = x1753 * 24576;
float* x1755 = x1714+x1754;
int32_t x1756 = x1753 * 3072;
float* x1757 = x1737+x1756;
float* x1758 = x1752+x1754;
for(int x1760=0; x1760 < 384; x1760++) {
int32_t x1761 = x1760 / 1;
int32_t x1765 = x1761 * 8;
int32_t x1766 = x1765 * 8;
int32_t x1762 = x1760 % 1;
int32_t x1763 = x1762 / 1;
int32_t x1767 = x1763 * 8;
int32_t x1768 = x1767 * 8;
int32_t x1769 = x1766 + x1768;
int32_t x1764 = x1762 % 1;
int32_t x1770 = x1764 * 8;
int32_t x1771 = x1770 * 8;
int32_t x1772 = x1769 + x1771;
float* x1773 = x1758+x1772;
float* x1774 = x1755+x1766;
for(int x1775=0; x1775 < 8; x1775++) {
int32_t x1777 = x1775 * 8;
float* x1778 = x1773+x1777;
int32_t x1776 = x1775 + x1763;
int32_t x1779 = x1776 * 8;
int32_t x1780 = x1779 + x1764;
float* x1781 = x1774+x1780;
memcpy(x1778, x1781, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 48,64,384,1,x56,384,x1758,64,1,x1757,64);

}
float* x1790 = (float*)myMalloc(196608 * sizeof(float));;
for(int x1791=0; x1791 < 196608; x1791++) {
float x1792 = x1737[x1791];
bool x1793 = x1792 < 0.0f;
if (x1793) {
x1790[x1791] = 0.0f;
} else {
float x1796 = x1737[x1791];
x1790[x1791] = x1796;
}

}
float* x1802 = (float*)myMalloc(786432 * sizeof(float));;
int32_t x1803 = 0;
for(int x1804=0; x1804 < 64; x1804++) {
for(int x1805=0; x1805 < 192; x1805++) {
for(int x1806=0; x1806 < 64; x1806++) {
int32_t x1807 = x1803;
float x1808 = x74[x1805];
x1802[x1807] = x1808;
x1803 += 1;

}

}

}
float* x1817 = (float*)myMalloc(196608 * sizeof(float));;
for(int x1818=0; x1818 < 64; x1818++) {
int32_t x1819 = x1818 * 3072;
float* x1820 = x1790+x1819;
int32_t x1821 = x1818 * 12288;
float* x1822 = x1802+x1821;
float* x1823 = x1817+x1819;
for(int x1824=0; x1824 < 48; x1824++) {
int32_t x1825 = x1824 / 1;
int32_t x1829 = x1825 * 8;
int32_t x1830 = x1829 * 8;
int32_t x1826 = x1824 % 1;
int32_t x1827 = x1826 / 1;
int32_t x1831 = x1827 * 8;
int32_t x1832 = x1831 * 8;
int32_t x1833 = x1830 + x1832;
int32_t x1828 = x1826 % 1;
int32_t x1834 = x1828 * 8;
int32_t x1835 = x1834 * 8;
int32_t x1836 = x1833 + x1835;
float* x1837 = x1823+x1836;
float* x1838 = x1820+x1830;
for(int x1839=0; x1839 < 8; x1839++) {
int32_t x1841 = x1839 * 8;
float* x1842 = x1837+x1841;
int32_t x1840 = x1839 + x1827;
int32_t x1843 = x1840 * 8;
int32_t x1844 = x1843 + x1828;
float* x1845 = x1838+x1844;
memcpy(x1842, x1845, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 192,64,48,1,x64,48,x1823,64,1,x1822,64);

}
float* x1854 = (float*)myMalloc(786432 * sizeof(float));;
for(int x1855=0; x1855 < 786432; x1855++) {
float x1856 = x1802[x1855];
bool x1857 = x1856 < 0.0f;
if (x1857) {
x1854[x1855] = 0.0f;
} else {
float x1860 = x1802[x1855];
x1854[x1855] = x1860;
}

}
float* x1866 = (float*)myMalloc(786432 * sizeof(float));;
int32_t x1867 = 0;
for(int x1868=0; x1868 < 64; x1868++) {
for(int x1869=0; x1869 < 192; x1869++) {
for(int x1870=0; x1870 < 64; x1870++) {
int32_t x1871 = x1867;
float x1872 = x86[x1869];
x1866[x1871] = x1872;
x1867 += 1;

}

}

}
float* x1881 = (float*)myMalloc(1769472 * sizeof(float));;
for(int x1882=0; x1882 < 64; x1882++) {
int32_t x1883 = x1882 * 3072;
float* x1884 = x1790+x1883;
int32_t x1885 = x1882 * 12288;
float* x1886 = x1866+x1885;
int32_t x1887 = x1882 * 27648;
float* x1888 = x1881+x1887;
for(int x1889=0; x1889 < 432; x1889++) {
int32_t x1890 = x1889 / 9;
int32_t x1894 = x1890 * 3;
int32_t x1895 = x1894 * 3;
int32_t x1896 = x1895 * 8;
int32_t x1897 = x1896 * 8;
int32_t x1891 = x1889 % 9;
int32_t x1892 = x1891 / 3;
int32_t x1898 = x1892 * 3;
int32_t x1899 = x1898 * 8;
int32_t x1900 = x1899 * 8;
int32_t x1901 = x1897 + x1900;
int32_t x1893 = x1891 % 3;
int32_t x1902 = x1893 * 8;
int32_t x1903 = x1902 * 8;
int32_t x1904 = x1901 + x1903;
float* x1905 = x1888+x1904;
int32_t x1906 = x1890 * 8;
int32_t x1907 = x1906 * 8;
float* x1908 = x1884+x1907;
int32_t x1920 = 1 - x1893;
bool x1921 = x1920 > 0;
int32_t x1922;
if (x1921) {
x1922 = x1920;
} else {
x1922 = 0;
}
int32_t x1923 = 3 - x1893;
int32_t x1924 = x1923 - 1;
int32_t x1925 = 1 - x1924;
bool x1926 = x1925 > 0;
int32_t x1927;
if (x1926) {
x1927 = x1925;
} else {
x1927 = 0;
}
int32_t x1928 = 8 - x1927;
int32_t x1929 = x1928 - x1922;
bool x1930 = x1929 <= 0;
bool x1934 = x1922 > 0;
int32_t x1919 = -1 + x1893;
bool x1947 = x1927 > 0;
for(int x1909=0; x1909 < 8; x1909++) {
int32_t x1910 = x1909 - 1;
int32_t x1911 = x1910 + x1892;
bool x1912 = x1911 < 0;
bool x1913 = x1911 >= 8;
bool x1914 = x1912 || x1913;
if (x1914) {
int32_t x1915 = x1909 * 8;
float* x1916 = x1905+x1915;
memset(x1916, 0, 4 * 8);;
} else {
if (x1930) {
int32_t x1915 = x1909 * 8;
float* x1931 = x1905+x1915;
memset(x1931, 0, 4 * 8);;
} else {
int32_t x1915 = x1909 * 8;
if (x1934) {
float* x1935 = x1905+x1915;
memset(x1935, 0, 4 * x1922);;
} else {
}
// may have segfault here
int32_t x1940 = x1915 + x1922;
float* x1941 = x1905+x1940;
int32_t x1942 = x1911 * 8;
int32_t x1943 = x1942 + x1919;
int32_t x1944 = x1943 + x1922;
float* x1945 = x1908+x1944;
memcpy(x1941, x1945, 4 * x1929);;
if (x1947) {
int32_t x1948 = x1915 + 8;
int32_t x1949 = x1948 - x1927;
float* x1950 = x1905+x1949;
memset(x1950, 0, 4 * x1927);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 192,64,432,1,x60,432,x1888,64,1,x1886,64);

}
float* x1965 = (float*)myMalloc(786432 * sizeof(float));;
for(int x1966=0; x1966 < 786432; x1966++) {
float x1967 = x1866[x1966];
bool x1968 = x1967 < 0.0f;
if (x1968) {
x1965[x1966] = 0.0f;
} else {
float x1971 = x1866[x1966];
x1965[x1966] = x1971;
}

}
float* x1977 = (float*)myMalloc(1572864 * sizeof(float));;
int32_t x1978 = 0;
for(int x1979=0; x1979 < 64; x1979++) {
int32_t x1980 = x1979 * 12288;
float* x1981 = x1854+x1980;
for(int x1982=0; x1982 < 12288; x1982++) {
int32_t x1983 = x1978;
float x1984 = x1981[x1982];
x1977[x1983] = x1984;
x1978 += 1;

}
float* x1989 = x1965+x1980;
for(int x1990=0; x1990 < 12288; x1990++) {
int32_t x1991 = x1978;
float x1992 = x1989[x1990];
x1977[x1991] = x1992;
x1978 += 1;

}

}
float* x1999 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x2000 = 0;
for(int x2001=0; x2001 < 64; x2001++) {
for(int x2002=0; x2002 < 64; x2002++) {
for(int x2003=0; x2003 < 64; x2003++) {
int32_t x2004 = x2000;
float x2005 = x51[x2002];
x1999[x2004] = x2005;
x2000 += 1;

}

}

}
float* x2014 = (float*)myMalloc(1572864 * sizeof(float));;
for(int x2015=0; x2015 < 64; x2015++) {
int32_t x2016 = x2015 * 24576;
float* x2017 = x1977+x2016;
int32_t x2018 = x2015 * 4096;
float* x2019 = x1999+x2018;
float* x2020 = x2014+x2016;
for(int x2021=0; x2021 < 384; x2021++) {
int32_t x2022 = x2021 / 1;
int32_t x2026 = x2022 * 8;
int32_t x2027 = x2026 * 8;
int32_t x2023 = x2021 % 1;
int32_t x2024 = x2023 / 1;
int32_t x2028 = x2024 * 8;
int32_t x2029 = x2028 * 8;
int32_t x2030 = x2027 + x2029;
int32_t x2025 = x2023 % 1;
int32_t x2031 = x2025 * 8;
int32_t x2032 = x2031 * 8;
int32_t x2033 = x2030 + x2032;
float* x2034 = x2020+x2033;
float* x2035 = x2017+x2027;
for(int x2036=0; x2036 < 8; x2036++) {
int32_t x2038 = x2036 * 8;
float* x2039 = x2034+x2038;
int32_t x2037 = x2036 + x2024;
int32_t x2040 = x2037 * 8;
int32_t x2041 = x2040 + x2025;
float* x2042 = x2035+x2041;
memcpy(x2039, x2042, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,64,384,1,x76,384,x2020,64,1,x2019,64);

}
float* x2051 = (float*)myMalloc(262144 * sizeof(float));;
for(int x2052=0; x2052 < 262144; x2052++) {
float x2053 = x1999[x2052];
bool x2054 = x2053 < 0.0f;
if (x2054) {
x2051[x2052] = 0.0f;
} else {
float x2057 = x1999[x2052];
x2051[x2052] = x2057;
}

}
float* x2063 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2064 = 0;
for(int x2065=0; x2065 < 64; x2065++) {
for(int x2066=0; x2066 < 256; x2066++) {
for(int x2067=0; x2067 < 64; x2067++) {
int32_t x2068 = x2064;
float x2069 = x82[x2066];
x2063[x2068] = x2069;
x2064 += 1;

}

}

}
float* x2078 = (float*)myMalloc(262144 * sizeof(float));;
for(int x2079=0; x2079 < 64; x2079++) {
int32_t x2080 = x2079 * 4096;
float* x2081 = x2051+x2080;
int32_t x2082 = x2079 * 16384;
float* x2083 = x2063+x2082;
float* x2084 = x2078+x2080;
for(int x2085=0; x2085 < 64; x2085++) {
int32_t x2086 = x2085 / 1;
int32_t x2090 = x2086 * 8;
int32_t x2091 = x2090 * 8;
int32_t x2087 = x2085 % 1;
int32_t x2088 = x2087 / 1;
int32_t x2092 = x2088 * 8;
int32_t x2093 = x2092 * 8;
int32_t x2094 = x2091 + x2093;
int32_t x2089 = x2087 % 1;
int32_t x2095 = x2089 * 8;
int32_t x2096 = x2095 * 8;
int32_t x2097 = x2094 + x2096;
float* x2098 = x2084+x2097;
float* x2099 = x2081+x2091;
for(int x2100=0; x2100 < 8; x2100++) {
int32_t x2102 = x2100 * 8;
float* x2103 = x2098+x2102;
int32_t x2101 = x2100 + x2088;
int32_t x2104 = x2101 * 8;
int32_t x2105 = x2104 + x2089;
float* x2106 = x2099+x2105;
memcpy(x2103, x2106, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,64,64,1,x91,64,x2084,64,1,x2083,64);

}
float* x2115 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x2116=0; x2116 < 1048576; x2116++) {
float x2117 = x2063[x2116];
bool x2118 = x2117 < 0.0f;
if (x2118) {
x2115[x2116] = 0.0f;
} else {
float x2121 = x2063[x2116];
x2115[x2116] = x2121;
}

}
float* x2127 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2128 = 0;
for(int x2129=0; x2129 < 64; x2129++) {
for(int x2130=0; x2130 < 256; x2130++) {
for(int x2131=0; x2131 < 64; x2131++) {
int32_t x2132 = x2128;
float x2133 = x55[x2130];
x2127[x2132] = x2133;
x2128 += 1;

}

}

}
float* x2142 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x2143=0; x2143 < 64; x2143++) {
int32_t x2144 = x2143 * 4096;
float* x2145 = x2051+x2144;
int32_t x2146 = x2143 * 16384;
float* x2147 = x2127+x2146;
int32_t x2148 = x2143 * 36864;
float* x2149 = x2142+x2148;
for(int x2151=0; x2151 < 576; x2151++) {
int32_t x2152 = x2151 / 9;
int32_t x2156 = x2152 * 3;
int32_t x2157 = x2156 * 3;
int32_t x2158 = x2157 * 8;
int32_t x2159 = x2158 * 8;
int32_t x2153 = x2151 % 9;
int32_t x2154 = x2153 / 3;
int32_t x2160 = x2154 * 3;
int32_t x2161 = x2160 * 8;
int32_t x2162 = x2161 * 8;
int32_t x2163 = x2159 + x2162;
int32_t x2155 = x2153 % 3;
int32_t x2164 = x2155 * 8;
int32_t x2165 = x2164 * 8;
int32_t x2166 = x2163 + x2165;
float* x2167 = x2149+x2166;
int32_t x2168 = x2152 * 8;
int32_t x2169 = x2168 * 8;
float* x2170 = x2145+x2169;
int32_t x2182 = 1 - x2155;
bool x2183 = x2182 > 0;
int32_t x2184;
if (x2183) {
x2184 = x2182;
} else {
x2184 = 0;
}
int32_t x2185 = 3 - x2155;
int32_t x2186 = x2185 - 1;
int32_t x2187 = 1 - x2186;
bool x2188 = x2187 > 0;
int32_t x2189;
if (x2188) {
x2189 = x2187;
} else {
x2189 = 0;
}
int32_t x2190 = 8 - x2189;
int32_t x2191 = x2190 - x2184;
bool x2192 = x2191 <= 0;
bool x2196 = x2184 > 0;
int32_t x2181 = -1 + x2155;
bool x2209 = x2189 > 0;
for(int x2171=0; x2171 < 8; x2171++) {
int32_t x2172 = x2171 - 1;
int32_t x2173 = x2172 + x2154;
bool x2174 = x2173 < 0;
bool x2175 = x2173 >= 8;
bool x2176 = x2174 || x2175;
if (x2176) {
int32_t x2177 = x2171 * 8;
float* x2178 = x2167+x2177;
memset(x2178, 0, 4 * 8);;
} else {
if (x2192) {
int32_t x2177 = x2171 * 8;
float* x2193 = x2167+x2177;
memset(x2193, 0, 4 * 8);;
} else {
int32_t x2177 = x2171 * 8;
if (x2196) {
float* x2197 = x2167+x2177;
memset(x2197, 0, 4 * x2184);;
} else {
}
// may have segfault here
int32_t x2202 = x2177 + x2184;
float* x2203 = x2167+x2202;
int32_t x2204 = x2173 * 8;
int32_t x2205 = x2204 + x2181;
int32_t x2206 = x2205 + x2184;
float* x2207 = x2170+x2206;
memcpy(x2203, x2207, 4 * x2191);;
if (x2209) {
int32_t x2210 = x2177 + 8;
int32_t x2211 = x2210 - x2189;
float* x2212 = x2167+x2211;
memset(x2212, 0, 4 * x2189);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,64,576,1,x70,576,x2149,64,1,x2147,64);

}
float* x2227 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x2228=0; x2228 < 1048576; x2228++) {
float x2229 = x2127[x2228];
bool x2230 = x2229 < 0.0f;
if (x2230) {
x2227[x2228] = 0.0f;
} else {
float x2233 = x2127[x2228];
x2227[x2228] = x2233;
}

}
float* x2239 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x2240 = 0;
for(int x2241=0; x2241 < 64; x2241++) {
int32_t x2242 = x2241 * 16384;
float* x2243 = x2115+x2242;
for(int x2244=0; x2244 < 16384; x2244++) {
int32_t x2245 = x2240;
float x2246 = x2243[x2244];
x2239[x2245] = x2246;
x2240 += 1;

}
float* x2251 = x2227+x2242;
for(int x2252=0; x2252 < 16384; x2252++) {
int32_t x2253 = x2240;
float x2254 = x2251[x2252];
x2239[x2253] = x2254;
x2240 += 1;

}

}
float* x2261 = (float*)myMalloc(524288 * sizeof(float));;
for(int x2262=0; x2262 < 524288; x2262++) {
x2261[x2262] = -3.4028235E38f;

}
int* x2266 = (int32_t*)myMalloc(524288 * sizeof(int32_t));;
for(int x2267=0; x2267 < 64; x2267++) {
int32_t x2268 = x2267 * 32768;
float* x2269 = x2239+x2268;
int32_t x2270 = x2267 * 8192;
float* x2271 = x2261+x2270;
int* x2272 = x2266+x2270;
int32_t x2273 = 0;
int32_t x2274 = 0;
for(int x2276=0; x2276 < 512; x2276++) {
int32_t x2277 = x2273;
int32_t x2278 = x2277;
int32_t x2279 = x2274;
int32_t x2280 = x2279;
for(int x2281=0; x2281 < 4; x2281++) {
int32_t x2282 = x2278;
int32_t x2283 = x2282;
int32_t x2284 = x2280;
int32_t x2285 = x2284;
for(int x2286=0; x2286 < 4; x2286++) {
int32_t x2287 = x2285;
int32_t x2288 = x2287;
int32_t x2289 = x2288;
int32_t x2290 = x2289;
int32_t x2291 = x2290;
float x2292 = x2269[x2291];
int32_t x2293 = x2283;
float x2294 = x2271[x2293];
bool x2295 = x2292 > x2294;
if (x2295) {
float x2296 = x2269[x2291];
x2271[x2293] = x2296;
int32_t x2298 = x2291 + x2268;
x2272[x2293] = x2298;
} else {
}
x2290 += 1;
int32_t x2303 = x2290;
float x2304 = x2269[x2303];
float x2305 = x2271[x2293];
bool x2306 = x2304 > x2305;
if (x2306) {
float x2307 = x2269[x2303];
x2271[x2293] = x2307;
int32_t x2309 = x2303 + x2268;
x2272[x2293] = x2309;
} else {
}
x2290 += 1;
x2288 += 8;
int32_t x2315 = x2288;
int32_t x2316 = x2315;
int32_t x2317 = x2316;
float x2318 = x2269[x2317];
float x2319 = x2271[x2293];
bool x2320 = x2318 > x2319;
if (x2320) {
float x2321 = x2269[x2317];
x2271[x2293] = x2321;
int32_t x2323 = x2317 + x2268;
x2272[x2293] = x2323;
} else {
}
x2316 += 1;
int32_t x2328 = x2316;
float x2329 = x2269[x2328];
float x2330 = x2271[x2293];
bool x2331 = x2329 > x2330;
if (x2331) {
float x2332 = x2269[x2328];
x2271[x2293] = x2332;
int32_t x2334 = x2328 + x2268;
x2272[x2293] = x2334;
} else {
}
x2316 += 1;
x2288 += 8;
x2283 += 1;
x2285 += 2;

}
x2278 += 4;
x2280 += 16;

}
x2273 += 16;
x2274 += 64;

}

}
float* x2354 = (float*)myMalloc(65536 * sizeof(float));;
int32_t x2355 = 0;
for(int x2356=0; x2356 < 64; x2356++) {
for(int x2357=0; x2357 < 64; x2357++) {
for(int x2358=0; x2358 < 16; x2358++) {
int32_t x2359 = x2355;
float x2360 = x62[x2357];
x2354[x2359] = x2360;
x2355 += 1;

}

}

}
float* x2369 = (float*)myMalloc(524288 * sizeof(float));;
for(int x2370=0; x2370 < 64; x2370++) {
int32_t x2371 = x2370 * 8192;
float* x2372 = x2261+x2371;
int32_t x2373 = x2370 * 1024;
float* x2374 = x2354+x2373;
float* x2375 = x2369+x2371;
for(int x2376=0; x2376 < 512; x2376++) {
int32_t x2377 = x2376 / 1;
int32_t x2381 = x2377 * 4;
int32_t x2382 = x2381 * 4;
int32_t x2378 = x2376 % 1;
int32_t x2379 = x2378 / 1;
int32_t x2383 = x2379 * 4;
int32_t x2384 = x2383 * 4;
int32_t x2385 = x2382 + x2384;
int32_t x2380 = x2378 % 1;
int32_t x2386 = x2380 * 4;
int32_t x2387 = x2386 * 4;
int32_t x2388 = x2385 + x2387;
float* x2389 = x2375+x2388;
float* x2390 = x2372+x2382;
for(int x2391=0; x2391 < 4; x2391++) {
int32_t x2393 = x2391 * 4;
float* x2394 = x2389+x2393;
int32_t x2392 = x2391 + x2379;
int32_t x2395 = x2392 * 4;
int32_t x2396 = x2395 + x2380;
float* x2397 = x2390+x2396;
memcpy(x2394, x2397, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,16,512,1,x43,512,x2375,16,1,x2374,16);

}
float* x2406 = (float*)myMalloc(65536 * sizeof(float));;
for(int x2408=0; x2408 < 65536; x2408++) {
float x2409 = x2354[x2408];
bool x2410 = x2409 < 0.0f;
if (x2410) {
x2406[x2408] = 0.0f;
} else {
float x2413 = x2354[x2408];
x2406[x2408] = x2413;
}

}
float* x2419 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x2420 = 0;
for(int x2421=0; x2421 < 64; x2421++) {
for(int x2422=0; x2422 < 256; x2422++) {
for(int x2423=0; x2423 < 16; x2423++) {
int32_t x2424 = x2420;
float x2425 = x68[x2422];
x2419[x2424] = x2425;
x2420 += 1;

}

}

}
float* x2434 = (float*)myMalloc(65536 * sizeof(float));;
for(int x2435=0; x2435 < 64; x2435++) {
int32_t x2436 = x2435 * 1024;
float* x2437 = x2406+x2436;
int32_t x2438 = x2435 * 4096;
float* x2439 = x2419+x2438;
float* x2440 = x2434+x2436;
for(int x2441=0; x2441 < 64; x2441++) {
int32_t x2442 = x2441 / 1;
int32_t x2446 = x2442 * 4;
int32_t x2447 = x2446 * 4;
int32_t x2443 = x2441 % 1;
int32_t x2444 = x2443 / 1;
int32_t x2448 = x2444 * 4;
int32_t x2449 = x2448 * 4;
int32_t x2450 = x2447 + x2449;
int32_t x2445 = x2443 % 1;
int32_t x2451 = x2445 * 4;
int32_t x2452 = x2451 * 4;
int32_t x2453 = x2450 + x2452;
float* x2454 = x2440+x2453;
float* x2455 = x2437+x2447;
for(int x2456=0; x2456 < 4; x2456++) {
int32_t x2458 = x2456 * 4;
float* x2459 = x2454+x2458;
int32_t x2457 = x2456 + x2444;
int32_t x2460 = x2457 * 4;
int32_t x2461 = x2460 + x2445;
float* x2462 = x2455+x2461;
memcpy(x2459, x2462, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,64,1,x80,64,x2440,16,1,x2439,16);

}
float* x2471 = (float*)myMalloc(262144 * sizeof(float));;
for(int x2472=0; x2472 < 262144; x2472++) {
float x2473 = x2419[x2472];
bool x2474 = x2473 < 0.0f;
if (x2474) {
x2471[x2472] = 0.0f;
} else {
float x2477 = x2419[x2472];
x2471[x2472] = x2477;
}

}
float* x2483 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x2484 = 0;
for(int x2485=0; x2485 < 64; x2485++) {
for(int x2486=0; x2486 < 256; x2486++) {
for(int x2487=0; x2487 < 16; x2487++) {
int32_t x2488 = x2484;
float x2489 = x59[x2486];
x2483[x2488] = x2489;
x2484 += 1;

}

}

}
float* x2498 = (float*)myMalloc(589824 * sizeof(float));;
for(int x2499=0; x2499 < 64; x2499++) {
int32_t x2500 = x2499 * 1024;
float* x2501 = x2406+x2500;
int32_t x2502 = x2499 * 4096;
float* x2503 = x2483+x2502;
int32_t x2504 = x2499 * 9216;
float* x2505 = x2498+x2504;
for(int x2506=0; x2506 < 576; x2506++) {
int32_t x2507 = x2506 / 9;
int32_t x2511 = x2507 * 3;
int32_t x2512 = x2511 * 3;
int32_t x2513 = x2512 * 4;
int32_t x2514 = x2513 * 4;
int32_t x2508 = x2506 % 9;
int32_t x2509 = x2508 / 3;
int32_t x2515 = x2509 * 3;
int32_t x2516 = x2515 * 4;
int32_t x2517 = x2516 * 4;
int32_t x2518 = x2514 + x2517;
int32_t x2510 = x2508 % 3;
int32_t x2519 = x2510 * 4;
int32_t x2520 = x2519 * 4;
int32_t x2521 = x2518 + x2520;
float* x2522 = x2505+x2521;
int32_t x2523 = x2507 * 4;
int32_t x2524 = x2523 * 4;
float* x2525 = x2501+x2524;
int32_t x2537 = 1 - x2510;
bool x2538 = x2537 > 0;
int32_t x2539;
if (x2538) {
x2539 = x2537;
} else {
x2539 = 0;
}
int32_t x2540 = 3 - x2510;
int32_t x2541 = x2540 - 1;
int32_t x2542 = 1 - x2541;
bool x2543 = x2542 > 0;
int32_t x2544;
if (x2543) {
x2544 = x2542;
} else {
x2544 = 0;
}
int32_t x2545 = 4 - x2544;
int32_t x2546 = x2545 - x2539;
bool x2547 = x2546 <= 0;
bool x2551 = x2539 > 0;
int32_t x2536 = -1 + x2510;
bool x2564 = x2544 > 0;
for(int x2526=0; x2526 < 4; x2526++) {
int32_t x2527 = x2526 - 1;
int32_t x2528 = x2527 + x2509;
bool x2529 = x2528 < 0;
bool x2530 = x2528 >= 4;
bool x2531 = x2529 || x2530;
if (x2531) {
int32_t x2532 = x2526 * 4;
float* x2533 = x2522+x2532;
memset(x2533, 0, 4 * 4);;
} else {
if (x2547) {
int32_t x2532 = x2526 * 4;
float* x2548 = x2522+x2532;
memset(x2548, 0, 4 * 4);;
} else {
int32_t x2532 = x2526 * 4;
if (x2551) {
float* x2552 = x2522+x2532;
memset(x2552, 0, 4 * x2539);;
} else {
}
// may have segfault here
int32_t x2557 = x2532 + x2539;
float* x2558 = x2522+x2557;
int32_t x2559 = x2528 * 4;
int32_t x2560 = x2559 + x2536;
int32_t x2561 = x2560 + x2539;
float* x2562 = x2525+x2561;
memcpy(x2558, x2562, 4 * x2546);;
if (x2564) {
int32_t x2565 = x2532 + 4;
int32_t x2566 = x2565 - x2544;
float* x2567 = x2522+x2566;
memset(x2567, 0, 4 * x2544);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,576,1,x72,576,x2505,16,1,x2503,16);

}
float* x2582 = (float*)myMalloc(262144 * sizeof(float));;
for(int x2583=0; x2583 < 262144; x2583++) {
float x2584 = x2483[x2583];
bool x2585 = x2584 < 0.0f;
if (x2585) {
x2582[x2583] = 0.0f;
} else {
float x2588 = x2483[x2583];
x2582[x2583] = x2588;
}

}
float* x2594 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x2595 = 0;
for(int x2596=0; x2596 < 64; x2596++) {
int32_t x2597 = x2596 * 4096;
float* x2598 = x2471+x2597;
for(int x2600=0; x2600 < 4096; x2600++) {
int32_t x2601 = x2595;
float x2602 = x2598[x2600];
x2594[x2601] = x2602;
x2595 += 1;

}
float* x2607 = x2582+x2597;
for(int x2608=0; x2608 < 4096; x2608++) {
int32_t x2609 = x2595;
float x2610 = x2607[x2608];
x2594[x2609] = x2610;
x2595 += 1;

}

}
float* x2617 = (float*)myMalloc(640 * sizeof(float));;
int32_t x2618 = 0;
for(int x2619=0; x2619 < 64; x2619++) {
for(int x2621=0; x2621 < 10; x2621++) {
int32_t x2622 = x2618;
float x2623 = x93[x2621];
x2617[x2622] = x2623;
x2618 += 1;

}

}
float* x2630 = (float*)myMalloc(524288 * sizeof(float));;
for(int x2631=0; x2631 < 64; x2631++) {
int32_t x2632 = x2631 * 8192;
float* x2633 = x2594+x2632;
int32_t x2634 = x2631 * 10;
float* x2635 = x2617+x2634;
float* x2636 = x2630+x2632;
for(int x2637=0; x2637 < 8192; x2637++) {
int32_t x2639 = x2637 % 16;
int32_t x2641 = x2639 % 4;
int32_t x2638 = x2637 / 16;
int32_t x2642 = x2638 * 4;
int32_t x2643 = x2642 * 4;
int32_t x2640 = x2639 / 4;
int32_t x2644 = x2640 * 4;
int32_t x2645 = x2643 + x2644;
int32_t x2646 = x2645 + x2641;
float* x2647 = x2636+x2646;
float* x2648 = x2633+x2643;
for(int x2650=0; x2650 < 1; x2650++) {
float* x2652 = x2647+x2650;
int32_t x2651 = x2650 + x2640;
int32_t x2653 = x2651 * 4;
int32_t x2654 = x2653 + x2641;
float* x2655 = x2648+x2654;
memcpy(x2652, x2655, 4 * 1);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 10,1,8192,1,x46,8192,x2636,1,1,x2635,1);

}
// resize to WrappedArray(64, 10)
int64_t x2665 = (long)mallocAddr;
int64_t x2666 = x2665 - x95;
memset((void*)x95, 0, x2666);
mallocAddr = (void*)x95;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x2673 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x2674 = x2673 / 1000LL;
int64_t x2676 = x2673 / x2675;
printf("Inferencing completed in %ldms (%ld us/images)\n",x2674,x2676);

}
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

